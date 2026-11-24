using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Holds runner-level evaluation snapshots for the betting simulator.
    /// Snapshots are cached in memory and persisted to disk so a restart does
    /// not force a full retrain. The most recently published snapshot is the
    /// default; callers can select older snapshots by key.
    /// </summary>
    public sealed class BettingSimulatorStore
    {
        private sealed class SnapshotIndex
        {
            public string? LatestKey { get; set; }
            public List<SimulatorSnapshotIndexEntry> Entries { get; set; } = new();
        }

        private readonly object _sync = new();
        private readonly string _directory;
        private readonly string _indexPath;
        private readonly Dictionary<string, BettingSimulationSnapshot> _cache =
            new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, SimulatorSnapshotIndexEntry> _index =
            new(StringComparer.OrdinalIgnoreCase);
        private string? _latestKey;
        private bool _persistenceEnabled = true;

        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        public BettingSimulatorStore(IConfiguration configuration)
        {
            var configured = configuration.GetValue<string>("Training:SimulatorSnapshotDirectory");
            _directory = string.IsNullOrWhiteSpace(configured)
                ? Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "HorseRacingML",
                    "simulator")
                : Path.GetFullPath(configured);
            _indexPath = Path.Combine(_directory, "index.json");

            try
            {
                Directory.CreateDirectory(_directory);
            }
            catch (Exception ex)
            {
                _persistenceEnabled = false;
                Console.Error.WriteLine(
                    $"[Simulator] Snapshot persistence disabled ({_directory}): {ex.Message}");
            }

            if (_persistenceEnabled)
            {
                LoadIndex();
            }
        }

        public BettingSimulationSnapshot? Get(string? key = null)
        {
            lock (_sync)
            {
                var resolved = ResolveKey(key);
                if (string.IsNullOrWhiteSpace(resolved))
                {
                    return null;
                }

                if (_cache.TryGetValue(resolved, out var cached))
                {
                    return cached;
                }

                if (!_index.TryGetValue(resolved, out var entry))
                {
                    return null;
                }

                var path = Path.Combine(_directory, entry.FileName);
                if (!File.Exists(path))
                {
                    return null;
                }

                try
                {
                    var json = File.ReadAllText(path);
                    var snapshot = JsonSerializer.Deserialize<BettingSimulationSnapshot>(
                        json,
                        SerializerOptions);
                    if (snapshot is null)
                    {
                        return null;
                    }

                    snapshot.Summary ??= new BettingSimulationSnapshotSummary();
                    snapshot.Runners ??= new List<SimulatorRunnerRow>();
                    snapshot.Races ??= new List<SimulatorRace>();
                    snapshot.Summary.Key = resolved;
                    if (snapshot.Races is { Count: > 0 } && snapshot.Runners.Count == 0)
                    {
                        snapshot.Runners = snapshot.Races
                            .Where(race => race is not null)
                            .SelectMany(race => race.Runners ?? new List<SimulatorRunnerRow>())
                            .ToList();
                    }

                    _cache[resolved] = snapshot;
                    return snapshot;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"[Simulator] Could not load snapshot {entry.FileName}: {ex.Message}");
                    return null;
                }
            }
        }

        public string Set(BettingSimulationSnapshot snapshot)
        {
            if (snapshot is null)
            {
                throw new ArgumentNullException(nameof(snapshot));
            }

            lock (_sync)
            {
                var key = BuildKey(snapshot);
                snapshot.Summary.Key = key;
                _cache[key] = snapshot;
                _latestKey = key;

                var fileName = $"{SanitiseFileName(key)}.json";
                try
                {
                    // Races already contain every runner. Avoid writing the
                    // flat runner list twice; Get() rebuilds it on load for
                    // callers that only use the legacy shape.
                    if (_persistenceEnabled)
                    {
                        var persisted = snapshot.Races is { Count: > 0 }
                            ? new BettingSimulationSnapshot
                            {
                                Summary = snapshot.Summary,
                                Races = snapshot.Races,
                                Runners = new List<SimulatorRunnerRow>()
                            }
                            : snapshot;
                        WriteAtomic(Path.Combine(_directory, fileName), persisted);
                    }

                    _index[key] = new SimulatorSnapshotIndexEntry
                    {
                        Key = key,
                        FileName = fileName,
                        UpdatedUtc = DateTime.UtcNow,
                        Summary = snapshot.Summary
                    };
                    SaveIndex();
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"[Simulator] Could not persist snapshot {key}: {ex.Message}");
                }

                return key;
            }
        }

        public IReadOnlyList<BettingSimulationSnapshotSummary> List()
        {
            lock (_sync)
            {
                return _index.Values
                    .Where(entry => entry.Summary is not null)
                    .OrderByDescending(entry => entry.UpdatedUtc)
                    .Select(entry => entry.Summary)
                    .ToList();
            }
        }

        public string? LatestKey
        {
            get
            {
                lock (_sync)
                {
                    return _latestKey;
                }
            }
        }

        public void Remove(string key)
        {
            lock (_sync)
            {
                if (_index.TryGetValue(key, out var entry))
                {
                    try
                    {
                        var path = Path.Combine(_directory, entry.FileName);
                        if (File.Exists(path))
                        {
                            File.Delete(path);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine(
                            $"[Simulator] Could not delete snapshot {entry.FileName}: {ex.Message}");
                    }

                    _index.Remove(key);
                    _cache.Remove(key);
                    if (string.Equals(_latestKey, key, StringComparison.OrdinalIgnoreCase))
                    {
                        _latestKey = _index.Values
                            .OrderByDescending(item => item.UpdatedUtc)
                            .Select(item => item.Key)
                            .FirstOrDefault();
                    }

                    SaveIndex();
                }
            }
        }

        public void Clear()
        {
            lock (_sync)
            {
                foreach (var entry in _index.Values)
                {
                    try
                    {
                        var path = Path.Combine(_directory, entry.FileName);
                        if (File.Exists(path))
                        {
                            File.Delete(path);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine(
                            $"[Simulator] Could not delete snapshot {entry.FileName}: {ex.Message}");
                    }
                }

                _cache.Clear();
                _index.Clear();
                _latestKey = null;
                SaveIndex();
            }
        }

        private string? ResolveKey(string? key)
        {
            if (!string.IsNullOrWhiteSpace(key))
            {
                return key;
            }

            return _latestKey ?? _index.Values
                .OrderByDescending(entry => entry.UpdatedUtc)
                .Select(entry => entry.Key)
                .FirstOrDefault();
        }

        private static string BuildKey(BettingSimulationSnapshot snapshot)
        {
            var summary = snapshot.Summary;
            var hash = string.IsNullOrWhiteSpace(summary.ModelHash)
                ? "model"
                : summary.ModelHash![..Math.Min(12, summary.ModelHash.Length)];
            var start = (summary.FirstRaceDate ?? summary.ValidationStart).ToString("yyyyMMdd");
            var end = (summary.LastRaceDate ?? summary.ValidationEnd).ToString("yyyyMMdd");
            var mode = summary.UsedSavedModel ? "saved" : "retrained";
            return $"{hash}-{start}-{end}-{mode}";
        }

        private static string SanitiseFileName(string key)
        {
            var invalid = Path.GetInvalidFileNameChars();
            return new string(key.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray());
        }

        private void LoadIndex()
        {
            try
            {
                if (!File.Exists(_indexPath))
                {
                    return;
                }

                var json = File.ReadAllText(_indexPath);
                var index = JsonSerializer.Deserialize<SnapshotIndex>(json, SerializerOptions);
                if (index is null)
                {
                    return;
                }

                _latestKey = index.LatestKey;
                foreach (var entry in index.Entries ?? new List<SimulatorSnapshotIndexEntry>())
                {
                    if (entry is null)
                    {
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(entry.Key) &&
                        !string.IsNullOrWhiteSpace(entry.FileName))
                    {
                        _index[entry.Key] = entry;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Simulator] Could not read snapshot index: {ex.Message}");
            }
        }

        private void SaveIndex()
        {
            if (!_persistenceEnabled)
            {
                return;
            }

            try
            {
                var index = new SnapshotIndex
                {
                    LatestKey = _latestKey,
                    Entries = _index.Values
                        .OrderByDescending(entry => entry.UpdatedUtc)
                        .ToList()
                };
                WriteAtomic(_indexPath, index);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Simulator] Could not write snapshot index: {ex.Message}");
            }
        }

        private static void WriteAtomic<T>(string path, T value)
        {
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var temporary = path + ".tmp";
            File.WriteAllText(temporary, JsonSerializer.Serialize(value, SerializerOptions));
            File.Move(temporary, path, overwrite: true);
        }
    }
}
