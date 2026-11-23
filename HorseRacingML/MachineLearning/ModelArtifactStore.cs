using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace HorseRacingML.ML
{
    /// <summary>
    /// A persisted model file plus enough identity to record it in MLParameters.
    /// </summary>
    public sealed record ModelArtifact(string FileName, string FullPath, string Sha256, string FeatureSchemaHash);

    /// <summary>
    /// Versioned, atomic storage for trained model artifacts.
    ///
    /// Every successful training run writes a new file; only an explicitly
    /// activated model becomes the live model. The active pointer is a small
    /// JSON file written atomically so a crash cannot leave a half-written
    /// model as the live one.
    /// </summary>
    public static class ModelArtifactStore
    {
        public const string ActiveFileName = "active.json";
        public const string LegacyFileName = "aiweights.json";
        public const string NormalizationFileName = "normalization.json";
        public const string StringMapsFileName = "string_maps.json";

        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            WriteIndented = true
        };

        private sealed record ActiveModelPointer
        {
            public string FileName { get; init; } = string.Empty;
            public string Sha256 { get; init; } = string.Empty;
            public string FeatureSchemaHash { get; init; } = string.Empty;
            public DateTime ActivatedUtc { get; init; } = DateTime.UtcNow;
        }

        public static string ResolveDirectory(string? configuredDirectory)
        {
            if (!string.IsNullOrWhiteSpace(configuredDirectory))
            {
                return Path.GetFullPath(configuredDirectory);
            }

            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(localAppData, "HorseRacingML", "models");
        }

        public static ModelArtifact Save(TrainedModel model, string? configuredDirectory, bool activate)
        {
            if (model is null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            var directory = ResolveDirectory(configuredDirectory);
            Directory.CreateDirectory(directory);

            model.TrainedUtc = model.TrainedUtc == default ? DateTime.UtcNow : model.TrainedUtc;
            if (string.IsNullOrWhiteSpace(model.FeatureSchemaHash))
            {
                model.FeatureSchemaHash = ComputeFeatureSchemaHash(model.Metadata);
            }

            var json = JsonSerializer.Serialize(model, SerializerOptions);
            var hash = ComputeSha256(json);
            var shortHash = hash[..12];
            var fileName = $"model-{model.TrainedUtc:yyyyMMddHHmmssfff}-{shortHash}.json";
            var fullPath = Path.Combine(directory, fileName);

            if (!File.Exists(fullPath))
            {
                WriteAtomic(fullPath, json);
            }

            // Keep the conventional file name for older code paths and for
            // operators who expect to find aiweights.json in the model folder.
            WriteAtomic(Path.Combine(directory, LegacyFileName), json);
            WriteAtomic(
                Path.Combine(directory, NormalizationFileName),
                JsonSerializer.Serialize(model.Normalization, SerializerOptions));
            WriteAtomic(
                Path.Combine(directory, StringMapsFileName),
                JsonSerializer.Serialize(model.Metadata.StringMaps, SerializerOptions));

            if (activate)
            {
                var pointer = new ActiveModelPointer
                {
                    FileName = fileName,
                    Sha256 = hash,
                    FeatureSchemaHash = model.FeatureSchemaHash,
                    ActivatedUtc = DateTime.UtcNow
                };
                WriteAtomic(
                    Path.Combine(directory, ActiveFileName),
                    JsonSerializer.Serialize(pointer, SerializerOptions));
            }

            return new ModelArtifact(fileName, fullPath, hash, model.FeatureSchemaHash);
        }

        public static void Activate(
            string? configuredDirectory,
            string fileName,
            string sha256,
            string featureSchemaHash)
        {
            if (string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentException("A model artifact file name is required.", nameof(fileName));
            }

            var directory = ResolveDirectory(configuredDirectory);
            Directory.CreateDirectory(directory);
            var pointer = new ActiveModelPointer
            {
                FileName = fileName,
                Sha256 = sha256,
                FeatureSchemaHash = featureSchemaHash,
                ActivatedUtc = DateTime.UtcNow
            };
            WriteAtomic(
                Path.Combine(directory, ActiveFileName),
                JsonSerializer.Serialize(pointer, SerializerOptions));
        }

        /// <summary>
        /// Resolves the active model: active pointer first, then conventional
        /// names, then the old bin/weights locations from earlier versions.
        /// </summary>
        public static string? ResolveActivePath(string? configuredDirectory)
        {
            var directory = ResolveDirectory(configuredDirectory);
            try
            {
                var pointerPath = Path.Combine(directory, ActiveFileName);
                if (File.Exists(pointerPath))
                {
                    var pointer = JsonSerializer.Deserialize<ActiveModelPointer>(File.ReadAllText(pointerPath));
                    if (!string.IsNullOrWhiteSpace(pointer?.FileName))
                    {
                        var candidate = Path.Combine(directory, pointer.FileName);
                        if (File.Exists(candidate))
                        {
                            return candidate;
                        }
                    }
                }

                var legacy = Path.Combine(directory, LegacyFileName);
                if (File.Exists(legacy))
                {
                    return legacy;
                }
            }
            catch
            {
                // Fall through to the legacy search.
            }

            foreach (var candidate in EnumerateLegacyCandidates())
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }

        public static string ResolveArtifactPath(string? configuredDirectory, string? fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName))
            {
                return ResolveActivePath(configuredDirectory) ?? string.Empty;
            }

            if (Path.IsPathRooted(fileName))
            {
                return fileName;
            }

            return Path.Combine(ResolveDirectory(configuredDirectory), fileName);
        }

        public static string ComputeFeatureSchemaHash(FeatureMetadata metadata)
        {
            if (metadata is null)
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            foreach (var key in metadata.Keys)
            {
                builder.Append(key);
                builder.Append('=');
                builder.Append(metadata.FeatureDimensions.TryGetValue(key, out var dim) ? dim : 0);
                builder.Append(';');
                if (metadata.StringMaps.TryGetValue(key, out var map))
                {
                    foreach (var entry in map.OrderBy(pair => pair.Key, StringComparer.Ordinal))
                    {
                        builder.Append(entry.Key);
                        builder.Append(':');
                        builder.Append(entry.Value);
                        builder.Append(',');
                    }
                }

                builder.Append('\n');
            }

            return ComputeSha256(builder.ToString());
        }

        private static string ComputeSha256(string value)
        {
            var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(value));
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        private static void WriteAtomic(string path, string contents)
        {
            var directory = Path.GetDirectoryName(path) ?? ".";
            Directory.CreateDirectory(directory);
            var tempPath = Path.Combine(directory, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
            try
            {
                using (var stream = new FileStream(
                           tempPath,
                           FileMode.CreateNew,
                           FileAccess.Write,
                           FileShare.None,
                           bufferSize: 64 * 1024,
                           FileOptions.WriteThrough))
                using (var writer = new StreamWriter(stream, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)))
                {
                    writer.Write(contents);
                    writer.Flush();
                    stream.Flush(flushToDisk: true);
                }

                File.Move(tempPath, path, overwrite: true);
            }
            finally
            {
                if (File.Exists(tempPath))
                {
                    try
                    {
                        File.Delete(tempPath);
                    }
                    catch
                    {
                        // Best effort cleanup.
                    }
                }
            }
        }

        private static IEnumerable<string> EnumerateLegacyCandidates()
        {
            static IEnumerable<string> Expand(string? directory)
            {
                if (string.IsNullOrWhiteSpace(directory))
                {
                    yield break;
                }

                yield return Path.Combine(directory, "weights", LegacyFileName);
                yield return Path.Combine(directory, LegacyFileName);
            }

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var path in Expand(AppContext.BaseDirectory))
            {
                if (seen.Add(path))
                {
                    yield return path;
                }
            }

            foreach (var path in Expand(Directory.GetCurrentDirectory()))
            {
                if (seen.Add(path))
                {
                    yield return path;
                }
            }

            var current = AppContext.BaseDirectory;
            for (var i = 0; i < 5 && !string.IsNullOrEmpty(current); i++)
            {
                current = Path.GetDirectoryName(current.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                if (string.IsNullOrEmpty(current))
                {
                    break;
                }

                foreach (var path in Expand(current))
                {
                    if (seen.Add(path))
                    {
                        yield return path;
                    }
                }
            }
        }
    }
}
