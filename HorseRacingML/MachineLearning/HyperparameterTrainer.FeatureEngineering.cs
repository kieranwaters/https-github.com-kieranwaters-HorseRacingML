using Dapper;
using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Data.SqlTypes;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using static Dapper.SqlMapper;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.ML
{
    public partial class HyperparameterTrainer
    {
        private static readonly DateTime BaseDate = new DateTime(2005, 1, 1);
        private const int PastRaceCount = 5;
        // Maximum history to keep per horse; must cover largest window
        private const int HistoryLength = 120;
        // Windows (in races) for which performance metrics will be generated
        private static readonly int[] PerformanceWindows =
            { 1, 3, 5, 10, 15, 20, 25, 30, 50, 100 };
        private static readonly IReadOnlyDictionary<int, float> ClassRatingBaselines =
            new Dictionary<int, float>
            {
                [1] = 105f,
                [2] = 100f,
                [3] = 95f,
                [4] = 90f,
                [5] = 85f,
                [6] = 80f,
                [7] = 75f
            };
        private const int TrainerJockeyRecentStarts = 50;
        private const int TrainerJockeyRecentDays = 180;
        private const float TypicalRestDays = 30f;
        private const float MsPerLength = 200f;
        private static readonly Regex DistanceHyphenBetweenDigitsRegex =
            new(@"(?<=\d)-(?=\d)", RegexOptions.Compiled);
        private static readonly char[] DistanceTokenSeparators =
        {
            ' ',
            '\t',
            '+',
            '\u00A0'
        };
        private static readonly char[] DistanceTokenTrimChars =
        {
            ',',
            '.',
            ';',
            ':',
            '!',
            '?',
            '"',
            '\'',
            '(',
            ')',
            '[',
            ']',
            '{',
            '}'
        };
        private static readonly IReadOnlyDictionary<string, float> DistanceTokenValues =
            new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase)
            {
                ["nse"] = 0.05f,
                ["nose"] = 0.05f,
                ["ns"] = 0.05f,
                ["shd"] = 0.1f,
                ["sht-hd"] = 0.1f,
                ["short-head"] = 0.1f,
                ["shorthead"] = 0.1f,
                ["hd"] = 0.2f,
                ["head"] = 0.2f,
                ["snk"] = 0.25f,
                ["short-neck"] = 0.25f,
                ["shortneck"] = 0.25f,
                ["nk"] = 0.3f,
                ["neck"] = 0.3f,
                ["dht"] = 0f,
                ["dh"] = 0f,
                ["dead-heat"] = 0f,
                ["deadheat"] = 0f,
                ["dist"] = 30f
            };
        private static readonly Regex HorseNameBracketTextRegex =
            new Regex("\\s*\\([^\\)]*\\)|\\s*\\[[^\\]]*\\]", RegexOptions.Compiled);
        private static readonly Regex UpcomingClassRegex =
            new("class\\s*(?:[:\\-]?\\s*)?(?<value>[0-9]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex HorseNameWhitespaceRegex = new Regex("\\s+", RegexOptions.Compiled);
        private static readonly object HorseIndexLock = new();
        private static bool _horsePerformanceIndexesEnsured;
        private readonly ConcurrentDictionary<int, PrefetchedHorseRace[]> _prefetchedHistoryCache = new();
        private static string SelectColumn(
            HashSet<string> available,
            string tableAlias,
            string columnName,
            string sqlType,
            string? alias = null)
        {
            if (available is null)
            {
                throw new ArgumentNullException(nameof(available));
            }

            alias ??= columnName;
            if (available.Contains(columnName))
            {
                var qualified = string.Concat(tableAlias, ".", columnName);
                return string.Equals(alias, columnName, StringComparison.Ordinal)
                    ? qualified
                    : string.Concat(qualified, " AS ", alias);
            }

            return string.Concat("CAST(NULL AS ", sqlType, ") AS ", alias);
        }

        private class RollingStat
        {
            public int Starts;
            public int Wins;
            public Queue<(DateTime date, bool win)> Recent = new();
        }

        private readonly record struct HistoryEntry(
            DateTime Date,
            float NormFinish,
            short? Finish,
            string? Going,
            string? Surface,
            int CourseId,
            string? Bucket,
            int? RaceClass,
            float RaceSpeed,
            float Speed,
            float SpeedDiff,
            int Age,
            bool Won,
            float Rating,
            float Weight,
            bool HasSpeed,
            bool HasWinningTime,
            float? WinningTimeMs,
            float? DistanceYards);
        private static float ClampNormalizedPosition(float value)
        {
            if (float.IsNaN(value))
            {
                return 0f;
            }

            return Math.Clamp(value, -5f, 5f);
        }
        private readonly record struct PrefetchedHorseRace(
            DateTime RaceDate,
            short? FinishPos,
            int? DistanceYards,
            int? WinningTimeMilliseconds,
            decimal? DistanceBeatenLengths,
            int RunnerResultId,
            short? OfficialRating,
            byte? Class);
        private sealed class FeatureEngineeringState
        {
            private readonly HyperparameterTrainer _trainer;
            private readonly ISet<string>? _identifierKeys;
            private readonly ISet<string> _preserveKeys;
            private readonly Dictionary<int, List<HistoryEntry>> _horseHistory = new();
            private readonly Dictionary<int, RollingStat> _trainerStats = new();
            private readonly Dictionary<int, RollingStat> _jockeyStats = new();
            private readonly Dictionary<(int trainerId, int jockeyId), (int starts, int wins)> _trainerJockeyStats = new();
            private readonly Dictionary<(int trainerId, int jockeyId, string surface), (int starts, int wins)> _trainerJockeySurfaceStats = new();
            private readonly Dictionary<(int trainerId, int jockeyId, int courseId), (int starts, int wins)> _trainerJockeyCourseStats = new();
            private readonly Dictionary<(int trainerId, int courseId), (int starts, int wins)> _trainerCourseStats = new();
            private readonly Dictionary<(int jockeyId, int courseId), (int starts, int wins)> _jockeyCourseStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _surfaceStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _goingStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _goingCourseStats = new();
            private readonly Dictionary<int, Dictionary<int, (int starts, int wins, float sumNorm, float lastNorm)>> _courseStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _distanceBucketStats = new();
            private readonly Dictionary<(int horseId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)> _horseClassStats = new();
            private readonly Dictionary<(int trainerId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)> _trainerClassStats = new();
            private readonly Dictionary<(int jockeyId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)> _jockeyClassStats = new();
            private readonly Dictionary<(int horseId, string going, string bucket), (int starts, int wins, float sumNorm, float lastNorm)> _goingDistanceStats = new();
            private readonly Dictionary<(int courseId, string bucket, int draw), (int starts, int wins)> _drawStats = new();
            private readonly Dictionary<(int courseId, string bucket), (int starts, int wins)> _drawBaselineStats = new();
            private readonly Dictionary<int, (double sum, int count)> _horseDistanceAll = new();
            private readonly Dictionary<int, (double sum, int count)> _horseDistanceWins = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _trainerSurfaceStats = new();
            private readonly Dictionary<int, int> _horseLastDistance = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _trainerGoingStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _trainerDistanceStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _jockeySurfaceStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _jockeyGoingStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _jockeyDistanceStats = new();
            private readonly Dictionary<(int jockeyId, string going, string bucket), (int starts, int wins, float sumNorm, float lastNorm)> _jockeyGoingDistanceStats = new();
            private readonly Dictionary<int, List<short?>> _jockeyFinishHistory = new();
            private readonly Dictionary<int, List<short?>> _trainerFinishHistory = new();
            public FeatureEngineeringState(HyperparameterTrainer trainer, ISet<string>? identifierKeys = null)
            {
                _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
                _identifierKeys = identifierKeys;
                if (identifierKeys is null)
                {
                    _preserveKeys = BackfillRequiredKeys;
                }
                else
                {
                    var combined = new HashSet<string>(BackfillRequiredKeys, StringComparer.OrdinalIgnoreCase);
                    foreach (var key in identifierKeys)
                    {
                        combined.Add(key);
                    }

                    _preserveKeys = combined;
                }
            }
            private static float ComputeStandardDeviation(IReadOnlyList<float> values)
            {
                if (values == null || values.Count == 0)
                {
                    return 0f;
                }

                float mean = values.Average();
                float variance = 0f;
                foreach (var value in values)
                {
                    float diff = value - mean;
                    variance += diff * diff;
                }

                return (float)Math.Sqrt(variance / values.Count);
            }
            private static int CalculateVolatility(IEnumerable<short?> finishPositions)
            {
                if (finishPositions == null)
                {
                    return 0;
                }

                var validFinishes = finishPositions.Where(fp => fp.HasValue).Select(fp => (float)fp.Value).ToList();
                if (validFinishes.Count < 2)
                {
                    return 0;
                }

                float averageFinish = validFinishes.Average();
                int shockResults = 0;
                foreach (var finish in validFinishes)
                {
                    if (Math.Abs(finish - averageFinish) > 5)
                    {
                        shockResults++;
                    }
                }

                return shockResults;
            }
            private static bool IsNullOrWhiteSpace(object? value)
            {
                if (value == null)
                {
                    return true;
                }

                if (value is string s)
                {
                    return string.IsNullOrWhiteSpace(s);
                }

                return false;
            }

            private static string? NormalizeStringValue(object? value)
            {
                if (value == null)
                {
                    return null;
                }

                string? text = value switch
                {
                    string s => s,
                    IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
                    _ => value.ToString()
                };

                if (string.IsNullOrWhiteSpace(text))
                {
                    return null;
                }

                return text.Trim();
            }
            private static void GroupNormalizedAction(
                IEnumerable<Dictionary<string, object?>> rows,
                string key,
                Action<string, IReadOnlyList<Dictionary<string, object?>>> action,
                Func<string?, string?>? normalizer = null)
            {
                if (rows is null)
                {
                    return;
                }

                if (string.IsNullOrWhiteSpace(key))
                {
                    throw new ArgumentException("Key cannot be null or whitespace.", nameof(key));
                }

                if (action is null)
                {
                    throw new ArgumentNullException(nameof(action));
                }

                normalizer ??= RacingRepository.NormalizeHistoricalNameKey;
                var groups = new Dictionary<string, List<Dictionary<string, object?>>>(StringComparer.OrdinalIgnoreCase);

                foreach (var row in rows)
                {
                    if (row is null)
                    {
                        continue;
                    }

                    if (!row.TryGetValue(key, out var value) || value is null)
                    {
                        continue;
                    }

                    var raw = NormalizeStringValue(value);
                    if (string.IsNullOrWhiteSpace(raw))
                    {
                        continue;
                    }

                    var normalized = normalizer(raw);
                    if (string.IsNullOrWhiteSpace(normalized))
                    {
                        continue;
                    }

                    if (!groups.TryGetValue(normalized, out var list))
                    {
                        list = new List<Dictionary<string, object?>>();
                        groups[normalized] = list;
                    }

                    list.Add(row);
                }

                foreach (var (normalized, groupRows) in groups)
                {
                    if (groupRows.Count == 0)
                    {
                        continue;
                    }

                    action(normalized, groupRows);
                }
            }

            private static void GroupNormalizedAction(
                IEnumerable<Dictionary<string, object?>> rows,
                string key,
                Action<IReadOnlyList<Dictionary<string, object?>>> action,
                Func<string?, string?>? normalizer = null)
            {
                if (action is null)
                {
                    throw new ArgumentNullException(nameof(action));
                }

                GroupNormalizedAction(rows, key, (_, groupedRows) => action(groupedRows), normalizer);
            }
            public static float ClampNormalizedPosition(float value)
            {
                if (float.IsNaN(value))
                {
                    return 0f;
                }

                return Math.Clamp(value, -5f, 5f);
            }
            private static float? CombineAverages(float? first, float? second)
            {
                if (first.HasValue && second.HasValue)
                {
                    return (first.Value + second.Value) / 2f;
                }

                if (first.HasValue)
                {
                    return first.Value;
                }

                if (second.HasValue)
                {
                    return second.Value;
                }

                return null;
            }
            private static float? ResolveFloat(Dictionary<string, object?> row, string key)
            {
                if (row is null || string.IsNullOrWhiteSpace(key))
                {
                    return null;
                }

                if (!row.TryGetValue(key, out var value) || value == null)
                {
                    return null;
                }

                try
                {
                    return Convert.ToSingle(value);
                }
                catch
                {
                    return null;
                }
            }
            internal static List<HistoryEntry> TakeRecentEntries(
                List<HistoryEntry> history,
                int desiredCount,
                Func<HistoryEntry, bool> predicate)
            {
                var selected = new List<HistoryEntry>(desiredCount > 0 ? desiredCount : 0);

                if (history == null || history.Count == 0 || desiredCount <= 0 || predicate == null)
                {
                    return selected;
                }

                for (int i = history.Count - 1; i >= 0 && selected.Count < desiredCount; i--)
                {
                    var candidate = history[i];
                    if (predicate(candidate))
                    {
                        selected.Add(candidate);
                    }
                }

                selected.Reverse();
                return selected;
            }
            public void ProcessRace(List<Dictionary<string, object?>> rows, bool includeRace, bool updateState)
            {
                if (rows is null)
                {
                    throw new ArgumentNullException(nameof(rows));
                }
                if (rows.Count == 0 || (!includeRace && !updateState))
                {
                    return;
                }
                rows.RemoveAll(row =>
                {
                    if (!PreparedDataset.TryGetRequiredInt32(row, "HorseId", out _))
                    {
                        return true;
                    }
                    if (!PreparedDataset.TryGetRequiredInt32(row, "RaceId", out _))
                    {
                        return true;
                    }
                    return false;
                });
                if (rows.Count == 0 || (!includeRace && !updateState))
                {
                    return;
                }
                foreach (var runnerRow in rows)
                {
                    bool distanceKnown = false;
                    float beatenLengths = 0f;
                    if (runnerRow.TryGetValue("DistanceBeatenLengths", out var lenObj) && lenObj != null)
                    {
                        beatenLengths = Convert.ToSingle(lenObj);
                        distanceKnown = true;
                    }
                    if (runnerRow.TryGetValue("DistanceBeatenText", out var txtObj) && txtObj is string txt)
                    {
                        var parsed = ParseDistanceBeaten(txt);
                        if (parsed.HasValue)
                        {
                            beatenLengths = parsed.Value;
                            distanceKnown = true;
                        }
                    }
                    if (!distanceKnown &&
                        runnerRow.TryGetValue("FinishPos", out var finishObj) &&
                        PreparedDataset.TryConvertToInt32(finishObj, out var finishValue) &&
                        finishValue == 1)
                    {
                        distanceKnown = true;
                        beatenLengths = 0f;
                    }
                    runnerRow["DistanceBeatenKnown"] = distanceKnown;
                    runnerRow["DistanceBeatenLengths"] = beatenLengths;
                }
                // =================================================================================
                // Stage 1: Pre-computation loop
                // Purpose: Compute horse-specific historical features required for race-level stats.
                // =================================================================================
                foreach (var row in rows)
                {
                    int horseId = PreparedDataset.GetRequiredInt32(row, "HorseId");
                    DateTime date = (DateTime)row["RaceDate"];
                    int? classValue = row.TryGetValue("Class", out var classObj) &&
                        PreparedDataset.TryConvertToInt32(classObj, out var parsedClass)
                            ? parsedClass
                            : (int?)null;
                    string? goingValue = NormalizeStringValue(
                            row.TryGetValue("Going", out var goingObj) ? goingObj : null);
                    bool goingMissing = string.IsNullOrEmpty(goingValue);
                    string? surfaceValue = NormalizeStringValue(
                        row.TryGetValue("Surface", out var surfaceObj) ? surfaceObj : null);
                    bool surfaceMissing = string.IsNullOrEmpty(surfaceValue);
                    int? distanceYardsValue = row.TryGetValue("DistanceYards", out var distanceObj) &&
                        PreparedDataset.TryConvertToInt32(distanceObj, out var parsedDistance)
                            ? parsedDistance
                            : (int?)null;
                    bool distanceMissing = !distanceYardsValue.HasValue || distanceYardsValue.Value <= 0;
                    string bucket = distanceMissing ? "Unknown" : DistanceBucket(distanceYardsValue ?? 0);
                    float rating = row.TryGetValue("OfficialRating", out var ratingObj) && ratingObj != null ? Convert.ToSingle(ratingObj) : 0f;
                    float weight = row.TryGetValue("WeightLbs", out var weightObj) && weightObj != null ? Convert.ToSingle(weightObj) : 0f;
                    int age = row.TryGetValue("Age", out var ageObj) && PreparedDataset.TryConvertToInt32(ageObj, out var ageValue)
                        ? ageValue
                        : 0;
                    if (!_horseHistory.TryGetValue(horseId, out var history))
                    {
                        history = new List<HistoryEntry>();
                        _horseHistory[horseId] = history;
                    }
                    if (history.Count > 0)
                    {
                        var previous = history[^1];
                        row["RaceSpeed"] = previous.HasWinningTime ? previous.RaceSpeed : 0f;
                        row["RunnerSpeed"] = previous.HasSpeed ? previous.Speed : 0f;
                        row["SpeedDiff"] = previous.HasSpeed ? previous.SpeedDiff : 0f;
                        row["SpeedRatio"] = previous.HasSpeed && previous.RaceSpeed != 0f
                            ? previous.Speed / previous.RaceSpeed
                            : 0f;
                        row["SpeedMissing"] = !previous.HasSpeed;
                    }
                    else
                    {
                        row["RaceSpeed"] = 0f;
                        row["RunnerSpeed"] = 0f;
                        row["SpeedDiff"] = 0f;
                        row["SpeedRatio"] = 0f;
                        row["SpeedMissing"] = true;
                    }

                    row["HistoricalDataMissing"] = history.Count == 0;
                    row["RatingChangeFromLast"] = history.Count > 0 ? rating - history[^1].Rating : 0f;
                    row["AgeProgression"] = history.Count > 0 ? age - history[^1].Age : 0f;
                    row["WeightChangeFromLast"] = history.Count > 0 ? weight - history[^1].Weight : 0f;
                    row["ClassChangeFromLast"] = history.Count > 0 && history[^1].RaceClass.HasValue && classValue.HasValue
                        ? classValue.Value - history[^1].RaceClass!.Value
                        : 0;
                    row["DaysSinceLastRace"] = history.Count > 0 ? (float)(date - history[^1].Date).TotalDays : 0f;
                    row["LastFinishPos"] = history.Count > 0 ? history[^1].Finish ?? 0 : 0;
                    int lastWinIdx = history.FindLastIndex(h => h.Won);
                    bool hasLastWin = lastWinIdx >= 0;
                    row["HasLastWin"] = hasLastWin;
                    if (hasLastWin)
                    {
                        row["DaysSinceLastWin"] = (float?)(date - history[lastWinIdx].Date).TotalDays;
                        row["RacesSinceLastWin"] = history.Count - 1 - lastWinIdx;
                    }
                    else
                    {
                        row["DaysSinceLastWin"] = null;
                        row["RacesSinceLastWin"] = null;
                    }

                    for (int i = 0; i < PastRaceCount; i++)
                    {
                        var key = $"Last{i + 1}NormPos";
                        row[key] = i < history.Count
                             ? history[history.Count - 1 - i].NormFinish
                            : 0f;
                    }

                    int careerStarts = history.Count;
                    int careerWins = history.Count(h => h.Won);
                    row["CareerStarts"] = careerStarts;
                    row["LifetimeWinRate"] = _trainer.SmoothedWinRate(careerWins, careerStarts);
                    int lifetimeTop3 = history.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 3);
                    int lifetimeTop5 = history.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 5);
                    row["Top3RateLifetime"] = _trainer.SmoothedWinRate(lifetimeTop3, history.Count);
                    row["Top5RateLifetime"] = _trainer.SmoothedWinRate(lifetimeTop5, history.Count);
                    var lifetimeNorms = history
                        .Where(h => h.Finish.HasValue && h.Finish.Value > 0)
                        .Select(h => h.NormFinish)
                        .ToList();
                    row["NormFinishStdDevLifetime"] = lifetimeNorms.Count >= 2
                        ? ComputeStandardDeviation(lifetimeNorms)
                        : 0f;

                    foreach (var window in PerformanceWindows)
                    {
                        int count = Math.Min(window, history.Count);
                        List<HistoryEntry> recent;
                        if (count > 0)
                        {
                            recent = history.GetRange(history.Count - count, count);
                        var surfaceSpeedRecent = recent.Where(h => (surfaceMissing || h.Surface == surfaceValue) && h.HasSpeed).ToList();
                        row[$"AvgSpeedOnSurfaceLast{window}"] = surfaceSpeedRecent.Any() ? surfaceSpeedRecent.Average(h => h.Speed) : 0f;

                        var goingSpeedRecent = recent.Where(h => (goingMissing || h.Going == goingValue) && h.HasSpeed).ToList();
                        row[$"AvgSpeedOnGoingLast{window}"] = goingSpeedRecent.Any() ? goingSpeedRecent.Average(h => h.Speed) : 0f;

                        var bucketSpeedRecent = recent.Where(h => h.Bucket == bucket && h.HasSpeed).ToList();
                        row[$"AvgSpeedAtDistanceBucketLast{window}"] = bucketSpeedRecent.Any() ? bucketSpeedRecent.Average(h => h.Speed) : 0f;
                        int wins = recent.Count(h => h.Finish == 1);
                        row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(wins, count);
                        row[$"AvgNormPosLast{window}"] = recent.Sum(h => h.NormFinish) / count;
                        row[$"AvgRatingLast{window}"] = recent.Sum(h => h.Rating) / count;
                        
                            var speedRecent = TakeRecentEntries(history, window, h => h.HasSpeed);
                            if (speedRecent.Count > 0)
                            {
                                row[$"AvgSpeedLast{window}"] = speedRecent.Average(h => h.Speed);
                                row[$"AvgSpeedDiffLast{window}"] = speedRecent.Average(h => h.SpeedDiff);
                            }
                            else
                            {
                                row[$"AvgSpeedLast{window}"] = 0f;
                                row[$"AvgSpeedDiffLast{window}"] = 0f;
                            }
                        }
                        else
                        {
                            row[$"AvgSpeedOnSurfaceLast{window}"] = 0f;
                            row[$"AvgSpeedOnGoingLast{window}"] = 0f;
                            row[$"AvgSpeedAtDistanceBucketLast{window}"] = 0f;
                            recent = new List<HistoryEntry>();
                            row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                            row[$"AvgNormPosLast{window}"] = 0f;
                            row[$"AvgRatingLast{window}"] = rating;
                            row[$"AvgSpeedLast{window}"] = 0f;
                            row[$"AvgSpeedDiffLast{window}"] = 0f;
                        }

                        int recentTop3 = recent.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 3);
                        int recentTop5 = recent.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 5);
                        row[$"Top3RateLast{window}"] = _trainer.SmoothedWinRate(recentTop3, count);
                        row[$"Top5RateLast{window}"] = _trainer.SmoothedWinRate(recentTop5, count);

                        var recentNorms = recent
                            .Where(h => h.Finish.HasValue && h.Finish.Value > 0)
                            .Select(h => h.NormFinish)
                            .ToList();
                        row[$"NormFinishStdDevLast{window}"] = recentNorms.Count >= 2
                            ? ComputeStandardDeviation(recentNorms)
                            : 0f;
                    }
                }

                // =================================================================================
                // Stage 2: Compute race-level statistics
                // Purpose: Aggregate stats from all runners now that per-horse features are ready.
                // =================================================================================
                var raceStat = ComputeRaceStats(rows);
                var tempWinningTimes = new Dictionary<int, int?>();
                foreach (var row in rows)
                {
                    int horseId = PreparedDataset.GetRequiredInt32(row, "HorseId");
                    tempWinningTimes[horseId] = row.TryGetValue("WinningTimeMs", out var winningObj) && PreparedDataset.TryConvertToInt32(winningObj, out var winningValue)
                        ? winningValue
                        : (int?)null;
                }
                // =================================================================================
                // Stage 3: Main feature engineering loop
                // Purpose: Compute all remaining features, including those relative to the field.
                // =================================================================================
                foreach (var row in rows)
                {
                    int horseId = PreparedDataset.GetRequiredInt32(row, "HorseId");
                    DateTime date = (DateTime)row["RaceDate"];
                    var normalizedRaceDate = date.Date;
                    if (TryGetTimeOfDay(row, out var timeOfDay))
                    {
                        float minutes = (float)timeOfDay.TotalMinutes;
                        float timeAngle = 2f * MathF.PI * minutes / (24f * 60f);
                        row["TimeOfDaySin"] = MathF.Sin(timeAngle);
                        row["TimeOfDayCos"] = MathF.Cos(timeAngle);
                        row["RaceDate"] = normalizedRaceDate;
                    }
                    else
                    {
                        row["TimeOfDaySin"] = 0f;
                        row["TimeOfDayCos"] = 0f;
                        int month = date.Month;
                        float monthAngle = 2f * MathF.PI * month / 12f;
                        row["RaceMonthSin"] = MathF.Sin(monthAngle);
                        row["RaceMonthCos"] = MathF.Cos(monthAngle);
                        int dayOfWeek = (int)date.DayOfWeek;
                        float dowAngle = 2f * MathF.PI * dayOfWeek / 7f;
                        row["RaceDayOfWeekSin"] = MathF.Sin(dowAngle);
                        row["RaceDayOfWeekCos"] = MathF.Cos(dowAngle);
                        row["IsWeekend"] = date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday;
                        int season = (month % 12) / 3;
                        float seasonAngle = 2f * MathF.PI * season / 4f;
                        row["SeasonSin"] = MathF.Sin(seasonAngle);
                        row["SeasonCos"] = MathF.Cos(seasonAngle);
                        row.Remove("RaceMonth");
                        row.Remove("RaceDayOfWeek");
                        row.Remove("Season");
                        row.Remove("ActualOff");
                        row.Remove("ScheduledOff");
                        row["RaceDate"] = normalizedRaceDate;
                    }

                    short? finish = row["FinishPos"] != null ? (short?)Convert.ToInt16(row["FinishPos"]) : null;
                    int raceId = PreparedDataset.GetRequiredInt32(row, "RaceId");
                    int runnerCount = row.TryGetValue("RunnerCount", out var runnerCountObj) &&
                                        PreparedDataset.TryConvertToInt32(runnerCountObj, out var runnerValue)
                        ? runnerValue
                        : raceStat.RunnerCount;
                    int? trainerId = row.TryGetValue("TrainerId", out var tObj) &&
                                      PreparedDataset.TryConvertToInt32(tObj, out var trainerValue)
                        ? trainerValue
                        : (int?)null;
                    int? jockeyId = row.TryGetValue("JockeyId", out var jObj) &&
                                     PreparedDataset.TryConvertToInt32(jObj, out var jockeyValue)
                        ? jockeyValue
                        : (int?)null;
                    int? classValue = row.TryGetValue("Class", out var classObj) &&
                        PreparedDataset.TryConvertToInt32(classObj, out var parsedClass)
                            ? parsedClass
                            : (int?)null;
                    bool classMissing = !classValue.HasValue;
                    int classVal = classValue ?? 0;
                    row["ClassMissing"] = classMissing;
                    bool ratingMissing = !(row.TryGetValue("OfficialRating", out var ratingObj) && ratingObj != null);
                    float rating = ratingMissing ? raceStat.AvgRating : Convert.ToSingle(ratingObj);
                    if (!ratingMissing && (float.IsNaN(rating) || rating <= 0f))
                    {
                        ratingMissing = true;
                        rating = raceStat.AvgRating;
                    }
                    row["RatingMissing"] = ratingMissing;
                    if ((ratingMissing && (rating <= 0f || float.IsNaN(rating))) ||
                            (!ratingMissing && (float.IsNaN(rating) || rating <= 0f)))
                    {
                        var classBaseline = ResolveClassRatingBaseline(classValue);
                        if (classBaseline.HasValue)
                        {
                            rating = classBaseline.Value;
                        }
                        else if (!float.IsNaN(raceStat.AvgRating) && raceStat.AvgRating > 0f)
                        {
                            rating = raceStat.AvgRating;
                        }
                    }
                    row["RatingDiffFromField"] = rating - raceStat.AvgRating;
                    string? goingValue = NormalizeStringValue(
                            row.TryGetValue("Going", out var goingObj) ? goingObj : null);
                    bool goingMissing = string.IsNullOrEmpty(goingValue);
                    if (!goingMissing)
                    {
                        row["Going"] = goingValue;
                    }
                    row["GoingMissing"] = goingMissing;
                    string going = goingValue ?? "Unknown";
                    string? surfaceValue = NormalizeStringValue(
                        row.TryGetValue("Surface", out var surfaceObj) ? surfaceObj : null);
                    bool surfaceMissing = string.IsNullOrEmpty(surfaceValue);
                    if (!surfaceMissing)
                    {
                        row["Surface"] = surfaceValue;
                    }
                    row["SurfaceMissing"] = surfaceMissing;
                    string surface = surfaceValue ?? "Unknown";
                    string? distanceTextValue = NormalizeStringValue(
                        row.TryGetValue("DistanceText", out var distanceTextObj) ? distanceTextObj : null);
                    bool distanceTextMissing = string.IsNullOrEmpty(distanceTextValue);
                    if (!distanceTextMissing)
                    {
                        row["DistanceText"] = distanceTextValue;
                    }
                    int? distanceYardsValue = row.TryGetValue("DistanceYards", out var distanceObj) &&
                        PreparedDataset.TryConvertToInt32(distanceObj, out var parsedDistance)
                            ? parsedDistance
                            : (int?)null;
                    bool distanceMissing = !distanceYardsValue.HasValue || distanceYardsValue.Value <= 0;
                    int distanceYards = distanceYardsValue ?? 0;
                    string bucket = distanceMissing ? "Unknown" : DistanceBucket(distanceYards);
                    row["DistanceMissing"] = distanceMissing;
                    row["DistanceTextMissing"] = distanceTextMissing;
                    row["RaceDate"] = normalizedRaceDate;
                    bool backBookMissing = !row.TryGetValue("BackBookPercentage", out var backBookObj) || backBookObj == null;
                    bool layBookMissing = !row.TryGetValue("LayBookPercentage", out var layBookObj) || layBookObj == null;
                    row["BackBookPercentageMissing"] = backBookMissing;
                    row["LayBookPercentageMissing"] = layBookMissing;
                    row["RaceMetadataMissing"] = classMissing || goingMissing ||
                        surfaceMissing || distanceMissing || distanceTextMissing;
                    bool drawMissing = row["Draw"] == null;
                    int draw = 0;
                    if (!drawMissing)
                    {
                        if (!PreparedDataset.TryConvertToInt32(row["Draw"], out draw))
                        {
                            drawMissing = true;
                        }
                    }
                    row["DrawMissing"] = drawMissing;
                    float runnerSpeed = 0f;
                    bool weightMissing = row["WeightLbs"] == null;
                    float weight = weightMissing ? 0f : Convert.ToSingle(row["WeightLbs"]);
                    row["WeightMissing"] = weightMissing;
                    row["RelativeDraw"] = runnerCount > 0 ? (float)draw / runnerCount : 0f;
                    int saddlecloth = 0;
                    bool saddleclothMissing = !(row.TryGetValue("SaddleclothNumber", out var saddleclothObj) &&
                                                 PreparedDataset.TryConvertToInt32(saddleclothObj, out saddlecloth));
                    row["SaddleclothMissing"] = saddleclothMissing;
                    row["SaddleclothRelative"] = !saddleclothMissing && runnerCount > 0 ? (float)saddlecloth / runnerCount : 0f;
                    row["SaddleclothDiffFromMean"] = !saddleclothMissing && raceStat.HasSaddleclothStats
                        ? saddlecloth - raceStat.AvgSaddlecloth
                        : 0f;
                    row["WeightDiffFromMean"] = weight - raceStat.AvgWeight;
                    bool hasWeightStats = raceStat.HasWeightStats;
                    row["IsTopWeight"] = !weightMissing && hasWeightStats && Math.Abs(weight - raceStat.MaxWeight) < 0.001f;
                    row["IsBottomWeight"] = !weightMissing && hasWeightStats && Math.Abs(weight - raceStat.MinWeight) < 0.001f;
                    row["FieldRatingStdDev"] = raceStat.StdRating;
                    row["PurseLevel"] = raceStat.TotalPurse;
                    int age = row.TryGetValue("Age", out var ageObj) && PreparedDataset.TryConvertToInt32(ageObj, out var ageValue)
                        ? ageValue
                        : 0;
                    if (!row.TryGetValue("Age", out var existingAge) || existingAge is null)
                    {
                        row["Age"] = age;
                    }
                    row["AgeRelative"] = age - raceStat.AvgAge;
                    int? currentWinningTimeMs = row.TryGetValue("WinningTimeMs", out var winningObj) && PreparedDataset.TryConvertToInt32(winningObj, out var winningValue)
                        ? winningValue
                        : (int?)null;
                    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    // ✅ MOVED: Initialize and compute horse history features UNCONDITIONALLY
                    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    if (!_horseHistory.TryGetValue(horseId, out var history))
                    {
                        history = new List<HistoryEntry>();
                        _horseHistory[horseId] = history;
                    }

                    if (history.Count > 0)
                    {
                        var previous = history[^1];
                        if (previous.HasWinningTime && previous.WinningTimeMs.HasValue && previous.WinningTimeMs.Value > 0f)
                        {
                            row["WinningTimeMs"] = (int)previous.WinningTimeMs.Value;
                        }
                        else
                        {
                            row["WinningTimeMs"] = null;
                        }
                        row["RaceSpeed"] = previous.HasWinningTime ? previous.RaceSpeed : 0f;
                        row["RunnerSpeed"] = previous.HasSpeed ? previous.Speed : 0f;
                        row["SpeedDiff"] = previous.HasSpeed ? previous.SpeedDiff : 0f;
                        row["SpeedRatio"] = previous.HasSpeed && previous.RaceSpeed != 0f
                            ? previous.Speed / previous.RaceSpeed
                            : 0f;
                        row["SpeedMissing"] = !previous.HasSpeed;
                    }
                    else
                    {
                        row["WinningTimeMs"] = null;
                        row["RaceSpeed"] = 0f;
                        row["RunnerSpeed"] = 0f;
                        row["SpeedDiff"] = 0f;
                        row["SpeedRatio"] = 0f;
                        row["SpeedMissing"] = true;
                    }

                    row["HistoricalDataMissing"] = history.Count == 0;
                    row["RatingChangeFromLast"] = history.Count > 0 ? rating - history[^1].Rating : 0f;
                    row["WeightChangeFromLast"] = history.Count > 0 ? weight - history[^1].Weight : 0f;
                    row["AgeProgression"] = history.Count > 0 ? age - history[^1].Age : 0f;
                    row["ClassChangeFromLast"] = history.Count > 0 && history[^1].RaceClass.HasValue && classValue.HasValue
                        ? classValue.Value - history[^1].RaceClass!.Value
                        : 0;
                    row["DaysSinceLastRace"] = history.Count > 0 ? (float)(date - history[^1].Date).TotalDays : 0f;
                    row["LastFinishPos"] = history.Count > 0 ? history[^1].Finish ?? 0 : 0;

                    int lastWinIdx = history.FindLastIndex(h => h.Won);
                    bool hasLastWin = lastWinIdx >= 0;
                    row["HasLastWin"] = hasLastWin;
                    if (hasLastWin)
                    {
                        row["DaysSinceLastWin"] = (float?)(date - history[lastWinIdx].Date).TotalDays;
                        row["RacesSinceLastWin"] = history.Count - 1 - lastWinIdx;
                    }
                    else
                    {
                        row["DaysSinceLastWin"] = null;
                        row["RacesSinceLastWin"] = null;
                    }

                    var daysSinceLast = (float)row["DaysSinceLastRace"];
                    row["LayoffShort"] = daysSinceLast < 30f;
                    row["LayoffMedium"] = daysSinceLast >= 30f && daysSinceLast <= 90f;
                    row["LayoffLong"] = daysSinceLast > 90f;
                    row["LayoffNormalized"] = daysSinceLast / TypicalRestDays;

                    for (int i = 0; i < PastRaceCount; i++)
                    {
                        var key = $"Last{i + 1}NormPos";
                        row[key] = i < history.Count
                             ? history[history.Count - 1 - i].NormFinish
                            : 0f;
                    }

                    int normCount = Math.Min(PastRaceCount, history.Count);
                    if (normCount > 1)
                    {
                        var recentNorms = history
                            .GetRange(history.Count - normCount, normCount)
                            .Select(h => h.NormFinish)
                            .ToList();
                        float xMean = (normCount - 1) / 2f;
                        float yMean = recentNorms.Average();
                        float num = 0f, den = 0f;
                        for (int j = 0; j < recentNorms.Count; j++)
                        {
                            float x = j;
                            float y = recentNorms[j];
                            num += (x - xMean) * (y - yMean);
                            den += (x - xMean) * (x - xMean);
                        }
                        row["NormPosSlope"] = den != 0f ? num / den : 0f;
                    }
                    else
                    {
                        row["NormPosSlope"] = 0f;
                    }

                    int speedCount = Math.Min(PastRaceCount, history.Count);
                    List<HistoryEntry> recentSpeedEntries = speedCount > 0
                        ? TakeRecentEntries(history, speedCount, h => h.HasSpeed)
                        : new List<HistoryEntry>();
                    List<float> recentSpeeds = recentSpeedEntries
                        .Select(entry => entry.Speed)
                        .ToList();
                    float speedSlope = 0f;
                    float speedStdDev = 0f;
                    if (recentSpeeds.Count > 0)
                    {
                        float speedMean = recentSpeeds.Average();
                        float variance = 0f;
                        foreach (var speed in recentSpeeds)
                        {
                            float diff = speed - speedMean;
                            variance += diff * diff;
                        }
                        speedStdDev = (float)Math.Sqrt(variance / recentSpeeds.Count);
                        if (recentSpeeds.Count > 1)
                        {
                            float xMean = (recentSpeeds.Count - 1) / 2f;
                            float num = 0f, den = 0f;
                            for (int j = 0; j < recentSpeeds.Count; j++)
                            {
                                float x = j;
                                float y = recentSpeeds[j];
                                num += (x - xMean) * (y - speedMean);
                                den += (x - xMean) * (x - xMean);
                            }
                            speedSlope = den != 0f ? num / den : 0f;
                        }
                    }
                    row["SpeedSlope"] = speedSlope;
                    row["SpeedStdDev"] = speedStdDev;

                    int ratingCount = Math.Min(PastRaceCount, history.Count);
                    if (ratingCount > 0)
                    {
                        var recentRatings = history
                            .GetRange(history.Count - ratingCount, ratingCount)
                            .Select(h => h.Rating)
                            .ToList();
                        if (ratingCount > 1)
                        {
                            float ratingMean = recentRatings.Average();
                            float xMean = (ratingCount - 1) / 2f;
                            float num = 0f, den = 0f;
                            for (int j = 0; j < recentRatings.Count; j++)
                            {
                                float x = j;
                                float y = recentRatings[j];
                                num += (x - xMean) * (y - ratingMean);
                                den += (x - xMean) * (x - xMean);
                            }
                            row["RatingSlope"] = den != 0f ? num / den : 0f;
                        }
                        else
                        {
                            row["RatingSlope"] = 0f;
                        }
                    }
                    else
                    {
                        row["RatingSlope"] = 0f;
                    }

                    row["RecentImprovement"] = (float)row["Last1NormPos"] - (float)row[$"Last{PastRaceCount}NormPos"];

                    int careerStarts = history.Count;
                    int careerWins = history.Count(h => h.Won);
                    row["CareerStarts"] = careerStarts;
                    row["LifetimeWinRate"] = _trainer.SmoothedWinRate(careerWins, careerStarts);

                    int lifetimeTop3 = history.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 3);
                    int lifetimeTop5 = history.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 5);
                    int lifetimeMeasuredStarts = history.Count;
                    row["Top3RateLifetime"] = _trainer.SmoothedWinRate(lifetimeTop3, lifetimeMeasuredStarts);
                    row["Top5RateLifetime"] = _trainer.SmoothedWinRate(lifetimeTop5, lifetimeMeasuredStarts);

                    var lifetimeNorms = history
                        .Where(h => h.Finish.HasValue && h.Finish.Value > 0)
                        .Select(h => h.NormFinish)
                        .ToList();
                    row["NormFinishStdDevLifetime"] = lifetimeNorms.Count >= 2
                        ? ComputeStandardDeviation(lifetimeNorms)
                        : 0f;

                    foreach (var window in PerformanceWindows)
                    {
                        int count = Math.Min(window, history.Count);
                        List<HistoryEntry> recent;
                        if (count > 0)
                        {
                            recent = history.GetRange(history.Count - count, count);
                            var surfaceSpeedRecent = recent.Where(h => (surfaceMissing || h.Surface == surface) && h.HasSpeed).ToList();
                            row[$"AvgSpeedOnSurfaceLast{window}"] = surfaceSpeedRecent.Any() ? surfaceSpeedRecent.Average(h => h.Speed) : 0f;

                            var goingSpeedRecent = recent.Where(h => (goingMissing || h.Going == going) && h.HasSpeed).ToList();
                            row[$"AvgSpeedOnGoingLast{window}"] = goingSpeedRecent.Any() ? goingSpeedRecent.Average(h => h.Speed) : 0f;

                            var bucketSpeedRecent = recent.Where(h => h.Bucket == bucket && h.HasSpeed).ToList();
                            row[$"AvgSpeedAtDistanceBucketLast{window}"] = bucketSpeedRecent.Any() ? bucketSpeedRecent.Average(h => h.Speed) : 0f;
                            int wins = recent.Count(h => h.Finish == 1);
                            row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(wins, count);
                            row[$"AvgNormPosLast{window}"] = recent.Sum(h => h.NormFinish) / count;
                            row[$"AvgRatingLast{window}"] = recent.Sum(h => h.Rating) / count;
                        }
                        else
                        {
                            row[$"AvgSpeedOnSurfaceLast{window}"] = 0f;
                            row[$"AvgSpeedOnGoingLast{window}"] = 0f;
                            row[$"AvgSpeedAtDistanceBucketLast{window}"] = 0f;
                            recent = new List<HistoryEntry>();
                            row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                            row[$"AvgNormPosLast{window}"] = 0f;
                            row[$"AvgRatingLast{window}"] = rating;
                        }

                        int recentTop3 = recent.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 3);
                        int recentTop5 = recent.Count(h => h.Finish.HasValue && h.Finish.Value > 0 && h.Finish.Value <= 5);
                        row[$"Top3RateLast{window}"] = _trainer.SmoothedWinRate(recentTop3, count);
                        row[$"Top5RateLast{window}"] = _trainer.SmoothedWinRate(recentTop5, count);

                        var recentNorms = recent
                            .Where(h => h.Finish.HasValue && h.Finish.Value > 0)
                            .Select(h => h.NormFinish)
                            .ToList();
                        row[$"NormFinishStdDevLast{window}"] = recentNorms.Count >= 2
                            ? ComputeStandardDeviation(recentNorms)
                            : 0f;
                    }
                    float winRateLast5 = row.ContainsKey("WinRateLast5") && row["WinRateLast5"] != null ? Convert.ToSingle(row["WinRateLast5"]) : 0f;
                    row["WinRateRatioRelativeToField"] = raceStat.AvgWinRateLast5 > 0 ? winRateLast5 / raceStat.AvgWinRateLast5 : 0f;

                    float avgSpeedLast5 = row.ContainsKey("AvgSpeedLast5") && row["AvgSpeedLast5"] != null ? Convert.ToSingle(row["AvgSpeedLast5"]) : 0f;
                    row["SpeedRatioRelativeToField"] = raceStat.AvgSpeedLast5 > 0 ? avgSpeedLast5 / raceStat.AvgSpeedLast5 : 0f;
                    row["SpeedZScore"] = raceStat.StdDevSpeedLast5 > 0 ? (avgSpeedLast5 - raceStat.AvgSpeedLast5) / raceStat.StdDevSpeedLast5 : 0f;

                    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    // ✅ END: Horse history features — now always computed

                    var horseClassKey = (horseId, classVal);
                    if (!_horseClassStats.TryGetValue(horseClassKey, out var horseClassStat))
                        horseClassStat = (0, 0, 0f, 0f);
                    row["ClassWinRate"] = classMissing
                         ? 0f
                         : _trainer.SmoothedWinRate(horseClassStat.wins, horseClassStat.starts);
                    row["ClassAvgNorm"] = !classMissing && horseClassStat.starts > 0
                        ? horseClassStat.sumNorm / horseClassStat.starts
                        : 0f;
                    row["LastClassNormPos"] = classMissing ? 0f : horseClassStat.lastNorm;

                    float? trainerWinRateValue = ResolveFloat(row, "TrainerWinRate");
                    float? trainerSurfaceAvg = ResolveFloat(row, "TrainerSurfaceAvgNorm");
                    float? trainerGoingAvg = ResolveFloat(row, "TrainerGoingAvgNorm");
                    float? trainerAvgNorm = trainerSurfaceAvg ?? trainerGoingAvg;
                    float? lastTrainerSurface = ResolveFloat(row, "LastTrainerSurfaceNormPos");
                    float? lastTrainerGoing = ResolveFloat(row, "LastTrainerGoingNormPos");

                    (int trainerId, int classVal) trainerClassKey = default;
                    (int starts, int wins, float sumNorm, float lastNorm) trainerClassStat = default;
                    bool hasTrainerClass = trainerId.HasValue && !classMissing;
                    float globalDefaultWinRate = _trainer.SmoothedWinRate(0, 0);
                    float trainerDefaultWinRate = globalDefaultWinRate;
                    if (hasTrainerClass)
                    {
                        trainerClassKey = (trainerId.Value, classVal);
                        if (!_trainerClassStats.TryGetValue(trainerClassKey, out trainerClassStat))
                            trainerClassStat = (0, 0, 0f, 0f);
                    }
                    bool hasTrainerClassHistory = hasTrainerClass && trainerClassStat.starts > 0;
                    if (hasTrainerClassHistory)
                    {
                        row["TrainerClassWinRate"] = _trainer.SmoothedWinRate(trainerClassStat.wins, trainerClassStat.starts);
                        row["TrainerClassAvgNorm"] = trainerClassStat.sumNorm / trainerClassStat.starts;
                        row["LastTrainerClassNormPos"] = trainerClassStat.lastNorm;
                    }
                    else
                    {
                        float trainerClassWinFallback = trainerWinRateValue ?? trainerDefaultWinRate;
                        float trainerClassAvgFallback = trainerAvgNorm
                            ?? ClampNormalizedPosition(1f - trainerClassWinFallback);
                        float trainerClassLastFallback = lastTrainerSurface
                            ?? lastTrainerGoing
                            ?? trainerClassAvgFallback;
                        row["TrainerClassWinRate"] = trainerClassWinFallback;
                        row["TrainerClassAvgNorm"] = trainerClassAvgFallback;
                        row["LastTrainerClassNormPos"] = trainerClassLastFallback;
                    }

                    float? jockeyWinRateValue = ResolveFloat(row, "JockeyWinRate");
                    float? jockeySurfaceAvg = ResolveFloat(row, "JockeySurfaceAvgNorm");
                    float? jockeyGoingAvg = ResolveFloat(row, "JockeyGoingAvgNorm");
                    float? jockeyAvgNorm = jockeySurfaceAvg ?? jockeyGoingAvg;
                    float? lastJockeySurface = ResolveFloat(row, "LastJockeySurfaceNormPos");
                    float? lastJockeyGoing = ResolveFloat(row, "LastJockeyGoingNormPos");
                    float? lastJockeyDistance = ResolveFloat(row, "LastJockeyDistanceBucketNormPos");

                    (int jockeyId, int classVal) jockeyClassKey = default;
                    (int starts, int wins, float sumNorm, float lastNorm) jockeyClassStat = default;
                    bool hasJockeyClass = jockeyId.HasValue && !classMissing;
                    if (hasJockeyClass)
                    {
                        jockeyClassKey = (jockeyId.Value, classVal);
                        if (!_jockeyClassStats.TryGetValue(jockeyClassKey, out jockeyClassStat))
                            jockeyClassStat = (0, 0, 0f, 0f);
                    }
                    bool hasJockeyClassHistory = hasJockeyClass && jockeyClassStat.starts > 0;
                    if (hasJockeyClassHistory)
                    {
                        row["JockeyClassWinRate"] = _trainer.SmoothedWinRate(jockeyClassStat.wins, jockeyClassStat.starts);
                        row["JockeyClassAvgNorm"] = jockeyClassStat.sumNorm / jockeyClassStat.starts;
                        row["LastJockeyClassNormPos"] = jockeyClassStat.lastNorm;
                    }
                    else
                    {
                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ❌ REMOVED: The entire history block was here before
                        // ✅ Now it runs unconditionally above
                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                        float jockeyClassWinFallback = jockeyWinRateValue
                            ?? ResolveFloat(row, "TrainerClassWinRate")
                            ?? ResolveFloat(row, "TrainerWinRate")
                            ?? globalDefaultWinRate;
                        float jockeyClassAvgFallback = jockeyAvgNorm
                            ?? ResolveFloat(row, "TrainerClassAvgNorm")
                            ?? ClampNormalizedPosition(1f - jockeyClassWinFallback);
                        float jockeyClassLastFallback = lastJockeySurface
                            ?? lastJockeyGoing
                            ?? lastJockeyDistance
                            ?? ResolveFloat(row, "LastTrainerClassNormPos")
                            ?? jockeyClassAvgFallback;
                        row["JockeyClassWinRate"] = jockeyClassWinFallback;
                        row["JockeyClassAvgNorm"] = jockeyClassAvgFallback;
                        row["LastJockeyClassNormPos"] = jockeyClassLastFallback;
                    }

                    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    // ✅ The rest of the method continues unchanged from here
                    // (going/surface/course stats, speed computation, draw bias, etc.)
                    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                    if (!_goingStats.TryGetValue(horseId, out var gDict))
                    {
                        gDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                        _goingStats[horseId] = gDict;
                    }
                    if (!gDict.TryGetValue(going, out var gStats))
                        gStats = (0, 0, 0f, 0f);
                    row["GoingWinRate"] = _trainer.SmoothedWinRate(gStats.wins, gStats.starts);
                    row["GoingAvgNorm"] = gStats.starts > 0 ? gStats.sumNorm / gStats.starts : 0f;
                    row["LastGoingNormPos"] = gStats.lastNorm;
                    var goingFeatureKey = going.Replace(" ", "");
                    row[$"LayoffNormalized_{goingFeatureKey}"] = (float)row["LayoffNormalized"];

                    if (!_surfaceStats.TryGetValue(horseId, out var sDict))
                    {
                        sDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                        _surfaceStats[horseId] = sDict;
                    }
                    if (!sDict.TryGetValue(surface, out var sStats))
                        sStats = (0, 0, 0f, 0f);
                    row["SurfaceWinRate"] = _trainer.SmoothedWinRate(sStats.wins, sStats.starts);
                    row["SurfaceAvgNorm"] = sStats.starts > 0 ? sStats.sumNorm / sStats.starts : 0f;
                    row["LastSurfaceNormPos"] = sStats.lastNorm;

                    int courseId = row.TryGetValue("CourseId", out var courseObj) && PreparedDataset.TryConvertToInt32(courseObj, out var courseIdValue)
                        ? courseIdValue
                        : 0;

                    if (!_courseStats.TryGetValue(horseId, out var cDict))
                    {
                        cDict = new();
                        _courseStats[horseId] = cDict;
                    }
                    if (!cDict.TryGetValue(courseId, out var cStats))
                        cStats = (0, 0, 0f, 0f);
                    row["CourseWinRate"] = _trainer.SmoothedWinRate(cStats.wins, cStats.starts);
                    row["LastCourseNormPos"] = cStats.lastNorm;

                    string gcKey = going + "_" + courseId;
                    if (!_goingCourseStats.TryGetValue(horseId, out var gcDict))
                    {
                        gcDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                        _goingCourseStats[horseId] = gcDict;
                    }
                    if (!gcDict.TryGetValue(gcKey, out var gcStats))
                        gcStats = (0, 0, 0f, 0f);
                    row["GoingCourseWinRate"] = _trainer.SmoothedWinRate(gcStats.wins, gcStats.starts);
                    row["GoingCourseAvgNorm"] = gcStats.starts > 0 ? gcStats.sumNorm / gcStats.starts : 0f;
                    row["LastGoingCourseNormPos"] = gcStats.lastNorm;

                    bool winningTimeAvailable = currentWinningTimeMs.HasValue && currentWinningTimeMs.Value > 0;
                    float raceSpeed = winningTimeAvailable && distanceYards > 0
                        ? distanceYards / (float)currentWinningTimeMs!.Value
                        : 0f;

                    bool distanceBeatenKnown = row.TryGetValue("DistanceBeatenKnown", out var distanceKnownObj) &&
                        distanceKnownObj is bool distanceKnownBool && distanceKnownBool;
                    bool hasRunnerSpeed = winningTimeAvailable && distanceBeatenKnown;
                    runnerSpeed = 0f;
                    if (hasRunnerSpeed)
                    {
                        float beaten = Convert.ToSingle(row["DistanceBeatenLengths"]);
                        float runnerTime = currentWinningTimeMs!.Value + beaten * MsPerLength;
                        runnerSpeed = runnerTime > 0f ? distanceYards / runnerTime : 0f;
                    }
                    float speedDiff = hasRunnerSpeed ? runnerSpeed - raceSpeed : 0f;
                    row["DistanceBucket"] = bucket;
                    row[$"RelativeDraw_{bucket}"] = (float)row["RelativeDraw"];

                    var drawKey = (courseId, bucket, draw);
                    _drawStats.TryGetValue(drawKey, out var drawStat);
                    var baseKey = (courseId, bucket);
                    _drawBaselineStats.TryGetValue(baseKey, out var baseStat);
                    row["DrawBias"] = _trainer.SmoothedWinRate(drawStat.wins, drawStat.starts) -
                                     _trainer.SmoothedWinRate(baseStat.wins, baseStat.starts);

                    if (!_distanceBucketStats.TryGetValue(horseId, out var dDict))
                    {
                        dDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                        _distanceBucketStats[horseId] = dDict;
                    }
                    if (!dDict.TryGetValue(bucket, out var dStats))
                        dStats = (0, 0, 0f, 0f);
                    row["DistanceBucketWinRate"] = _trainer.SmoothedWinRate(dStats.wins, dStats.starts);
                    row["LastDistanceBucketNormPos"] = dStats.lastNorm;

                    var gdKey = (horseId, going, bucket);
                    if (!_goingDistanceStats.TryGetValue(gdKey, out var gdStats))
                        gdStats = (0, 0, 0f, 0f);
                    row["GoingDistanceWinRate"] = _trainer.SmoothedWinRate(gdStats.wins, gdStats.starts);
                    row["GoingDistanceAvgNorm"] = gdStats.starts > 0 ? gdStats.sumNorm / gdStats.starts : 0f;
                    row["LastGoingDistanceNormPos"] = gdStats.lastNorm;

                    foreach (var window in PerformanceWindows)
                    {
                        int count = Math.Min(window, history.Count);
                        if (count > 0)
                        {
                            var recent = history.GetRange(history.Count - count, count);
                            var goingRecent = recent.Where(h => goingMissing || h.Going == going).ToList();
                            row[$"GoingWinRateLast{window}"] = _trainer.SmoothedWinRate(goingRecent.Count(h => h.Finish == 1), goingRecent.Count);
                            row[$"GoingAvgNormLast{window}"] = goingRecent.Count > 0
                                ? goingRecent.Sum(h => h.NormFinish) / goingRecent.Count
                                : 0f;

                            var surfaceRecent = recent.Where(h => surfaceMissing || h.Surface == surface).ToList();
                            row[$"SurfaceWinRateLast{window}"] = _trainer.SmoothedWinRate(surfaceRecent.Count(h => h.Finish == 1), surfaceRecent.Count);
                            row[$"SurfaceAvgNormLast{window}"] = surfaceRecent.Count > 0
                                ? surfaceRecent.Sum(h => h.NormFinish) / surfaceRecent.Count
                                : 0f;

                            var courseRecent = recent.Where(h => h.CourseId == courseId).ToList();
                            row[$"CourseWinRateLast{window}"] = _trainer.SmoothedWinRate(courseRecent.Count(h => h.Finish == 1), courseRecent.Count);
                            row[$"CourseAvgNormLast{window}"] = courseRecent.Count > 0
                                ? courseRecent.Sum(h => h.NormFinish) / courseRecent.Count
                                : 0f;

                            var bucketRecent = recent.Where(h => h.Bucket == bucket).ToList();
                            row[$"DistanceBucketWinRateLast{window}"] = _trainer.SmoothedWinRate(bucketRecent.Count(h => h.Finish == 1), bucketRecent.Count);
                            row[$"DistanceBucketAvgNormLast{window}"] = bucketRecent.Count > 0
                                ? bucketRecent.Sum(h => h.NormFinish) / bucketRecent.Count
                                : 0f;

                            var speedRecent = TakeRecentEntries(history, window, h => h.HasSpeed);
                            if (speedRecent.Count > 0)
                            {
                                row[$"AvgSpeedDiffLast{window}"] = speedRecent.Average(h => h.SpeedDiff);
                            }
                            else
                            {
                                row[$"AvgSpeedDiffLast{window}"] = 0f;
                            }
                        }
                        else
                        {
                            row[$"GoingWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                            row[$"GoingAvgNormLast{window}"] = 0f;
                            row[$"SurfaceWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                            row[$"SurfaceAvgNormLast{window}"] = 0f;
                            row[$"CourseWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                            row[$"CourseAvgNormLast{window}"] = 0f;
                            row[$"DistanceBucketWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                            row[$"DistanceBucketAvgNormLast{window}"] = 0f;
                            row[$"AvgSpeedDiffLast{window}"] = 0f;
                        }
                    }

                    _horseDistanceAll.TryGetValue(horseId, out var allDist);
                    _horseDistanceWins.TryGetValue(horseId, out var winDist);
                    double avgDist = allDist.count > 0 ? allDist.sum / allDist.count : distanceYards;
                    row["DistanceRatioFromAverage"] = avgDist > 0 ? distanceYards / (float)avgDist : 1f;

                    if (_horseLastDistance.TryGetValue(horseId, out var lastDist))
                        row["DistanceChangeFromLast"] = distanceYards - lastDist;
                    else
                        row["DistanceChangeFromLast"] = 0f;

                    double pref = winDist.count > 0 ? winDist.sum / winDist.count : (allDist.count > 0 ? allDist.sum / allDist.count : distanceYards);
                    row["DistanceFromPreferred"] = (float)Math.Abs(distanceYards - pref);

                    globalDefaultWinRate = _trainer.SmoothedWinRate(0, 0);
                    RollingStat trainerStat = default;
                    bool hasTrainerStat = false;
                    trainerDefaultWinRate = globalDefaultWinRate;

                    if (trainerId.HasValue)
                    {
                        if (!_trainerStats.TryGetValue(trainerId.Value, out trainerStat))
                            trainerStat = new RollingStat();
                        hasTrainerStat = true;
                        trainerDefaultWinRate = _trainer.SmoothedWinRate(trainerStat.Wins, trainerStat.Starts);

                        if (!_trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict))
                        {
                            tsDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                            _trainerSurfaceStats[trainerId.Value] = tsDict;
                        }
                        if (!tsDict.TryGetValue(surface, out var tsStats))
                            tsStats = (0, 0, 0f, 0f);
                        row["TrainerSurfaceWinRate"] = tsStats.starts > 0
                            ? _trainer.SmoothedWinRate(tsStats.wins, tsStats.starts)
                            : trainerDefaultWinRate;
                        row["TrainerSurfaceAvgNorm"] = tsStats.starts > 0 ? tsStats.sumNorm / tsStats.starts : 0f;
                        row["LastTrainerSurfaceNormPos"] = tsStats.lastNorm;

                        if (!_trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict))
                        {
                            tgDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                            _trainerGoingStats[trainerId.Value] = tgDict;
                        }
                        if (!tgDict.TryGetValue(going, out var tgStats))
                            tgStats = (0, 0, 0f, 0f);
                        row["TrainerGoingWinRate"] = tgStats.starts > 0
                            ? _trainer.SmoothedWinRate(tgStats.wins, tgStats.starts)
                            : trainerDefaultWinRate;
                        row["TrainerGoingAvgNorm"] = tgStats.starts > 0 ? tgStats.sumNorm / tgStats.starts : 0f;
                        row["LastTrainerGoingNormPos"] = tgStats.lastNorm;

                        if (!_trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict))
                        {
                            tdDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                            _trainerDistanceStats[trainerId.Value] = tdDict;
                        }
                        if (!tdDict.TryGetValue(bucket, out var tdStats))
                            tdStats = (0, 0, 0f, 0f);
                        row["TrainerDistanceBucketWinRate"] = tdStats.starts > 0
                            ? _trainer.SmoothedWinRate(tdStats.wins, tdStats.starts)
                            : trainerDefaultWinRate;
                        row["TrainerDistanceBucketAvgNorm"] = tdStats.starts > 0 ? tdStats.sumNorm / tdStats.starts : 0f;
                        row["LastTrainerDistanceBucketNormPos"] = tdStats.lastNorm;

                        while (trainerStat.Recent.Count > 0 &&
                                                        (date - trainerStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                            trainerStat.Recent.Dequeue();
                        var tRecent = trainerStat.Recent.ToList();
                        int tCount = Math.Min(tRecent.Count, TrainerJockeyRecentStarts);
                        int tWinsRecent = tRecent.Skip(tRecent.Count - tCount).Count(r => r.win);
                        row[$"TrainerWinRateLast{TrainerJockeyRecentStarts}"] = _trainer.SmoothedWinRate(tWinsRecent, tCount);
                        row["TrainerWinRateRecentDays"] = _trainer.SmoothedWinRate(tRecent.Count(r => r.win), tRecent.Count);
                        row["TrainerWinRate"] = trainerDefaultWinRate;
                    }
                    else
                    {
                        row["TrainerWinRate"] = globalDefaultWinRate;
                        row[$"TrainerWinRateLast{TrainerJockeyRecentStarts}"] = globalDefaultWinRate;
                        row["TrainerWinRateRecentDays"] = globalDefaultWinRate;
                        row["TrainerSurfaceWinRate"] = globalDefaultWinRate;
                        row["TrainerSurfaceAvgNorm"] = 0f;
                        row["LastTrainerSurfaceNormPos"] = 0f;
                        row["TrainerGoingWinRate"] = globalDefaultWinRate;
                        row["TrainerGoingAvgNorm"] = 0f;
                        row["LastTrainerGoingNormPos"] = 0f;
                        row["TrainerDistanceBucketWinRate"] = globalDefaultWinRate;
                        row["TrainerDistanceBucketAvgNorm"] = 0f;
                        row["LastTrainerDistanceBucketNormPos"] = 0f;
                    }

                    RollingStat jockeyStat = default;
                    bool hasJockeyStat = false;
                    float jockeyDefaultWinRate = globalDefaultWinRate;
                    if (jockeyId.HasValue)
                    {
                        if (!_jockeyStats.TryGetValue(jockeyId.Value, out jockeyStat))
                            jockeyStat = new RollingStat();
                        hasJockeyStat = true;
                        jockeyDefaultWinRate = _trainer.SmoothedWinRate(jockeyStat.Wins, jockeyStat.Starts);

                        if (!_jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict))
                        {
                            jsDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                            _jockeySurfaceStats[jockeyId.Value] = jsDict;
                        }
                        if (!jsDict.TryGetValue(surface, out var jsStats))
                            jsStats = (0, 0, 0f, 0f);
                        row["JockeySurfaceWinRate"] = jsStats.starts > 0
                            ? _trainer.SmoothedWinRate(jsStats.wins, jsStats.starts)
                            : jockeyDefaultWinRate;
                        row["JockeySurfaceAvgNorm"] = jsStats.starts > 0 ? jsStats.sumNorm / jsStats.starts : 0f;
                        row["LastJockeySurfaceNormPos"] = jsStats.lastNorm;

                        if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict))
                        {
                            jgDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                            _jockeyGoingStats[jockeyId.Value] = jgDict;
                        }
                        if (!jgDict.TryGetValue(going, out var jgStats))
                            jgStats = (0, 0, 0f, 0f);
                        row["JockeyGoingWinRate"] = jgStats.starts > 0
                            ? _trainer.SmoothedWinRate(jgStats.wins, jgStats.starts)
                            : jockeyDefaultWinRate;
                        row["JockeyGoingAvgNorm"] = jgStats.starts > 0 ? jgStats.sumNorm / jgStats.starts : 0f;
                        row["LastJockeyGoingNormPos"] = jgStats.lastNorm;

                        if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict))
                        {
                            jdDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                            _jockeyDistanceStats[jockeyId.Value] = jdDict;
                        }
                        if (!jdDict.TryGetValue(bucket, out var jdStats))
                            jdStats = (0, 0, 0f, 0f);
                        row["JockeyDistanceBucketWinRate"] = jdStats.starts > 0
                            ? _trainer.SmoothedWinRate(jdStats.wins, jdStats.starts)
                            : jockeyDefaultWinRate;
                        row["JockeyDistanceBucketAvgNorm"] = jdStats.starts > 0 ? jdStats.sumNorm / jdStats.starts : 0f;
                        row["LastJockeyDistanceBucketNormPos"] = jdStats.lastNorm;

                        var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                        if (!_jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat))
                            jockeyGoingDistanceStat = (0, 0, 0f, 0f);
                        if (jockeyGoingDistanceStat.starts > 0)
                        {
                            row["JockeyGoingDistanceWinRate"] = _trainer.SmoothedWinRate(jockeyGoingDistanceStat.wins, jockeyGoingDistanceStat.starts);
                            row["JockeyGoingDistanceAvgNorm"] = jockeyGoingDistanceStat.sumNorm / jockeyGoingDistanceStat.starts;
                            row["LastJockeyGoingDistanceNormPos"] = jockeyGoingDistanceStat.lastNorm;
                        }
                        else
                        {
                            float? jockeyGoingWin = ResolveFloat(row, "JockeyGoingWinRate");
                            float? jockeyDistanceWin = ResolveFloat(row, "JockeyDistanceBucketWinRate");
                            float? jockeyGoingAvgNorm = ResolveFloat(row, "JockeyGoingAvgNorm");
                            float? jockeyDistanceAvgNorm = ResolveFloat(row, "JockeyDistanceBucketAvgNorm");
                            float? lastJockeyDistanceNorm = ResolveFloat(row, "LastJockeyDistanceBucketNormPos");
                            row["JockeyGoingDistanceWinRate"] = CombineAverages(jockeyGoingWin, jockeyDistanceWin) ?? jockeyDefaultWinRate;
                            row["JockeyGoingDistanceAvgNorm"] = CombineAverages(jockeyGoingAvgNorm, jockeyDistanceAvgNorm) ?? 0f;
                            row["LastJockeyGoingDistanceNormPos"] = CombineAverages(ResolveFloat(row, "LastJockeyGoingNormPos"), lastJockeyDistanceNorm) ?? 0f;
                        }

                        while (jockeyStat.Recent.Count > 0 &&
                           (date - jockeyStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                            jockeyStat.Recent.Dequeue();
                        var jRecent = jockeyStat.Recent.ToList();
                        int jCount = Math.Min(jRecent.Count, TrainerJockeyRecentStarts);
                        int jWinsRecent = jRecent.Skip(jRecent.Count - jCount).Count(r => r.win);
                        row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = _trainer.SmoothedWinRate(jWinsRecent, jCount);
                        row["JockeyWinRateRecentDays"] = _trainer.SmoothedWinRate(jRecent.Count(r => r.win), jRecent.Count);
                        row["JockeyWinRate"] = jockeyDefaultWinRate;
                    }
                    else
                    {
                        row["JockeyWinRate"] = globalDefaultWinRate;
                        row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = globalDefaultWinRate;
                        row["JockeyWinRateRecentDays"] = globalDefaultWinRate;
                        row["JockeySurfaceWinRate"] = globalDefaultWinRate;
                        row["JockeySurfaceAvgNorm"] = 0f;
                        row["LastJockeySurfaceNormPos"] = 0f;
                        row["JockeyGoingWinRate"] = globalDefaultWinRate;
                        row["JockeyGoingAvgNorm"] = 0f;
                        row["LastJockeyGoingNormPos"] = 0f;
                        row["JockeyDistanceBucketWinRate"] = globalDefaultWinRate;
                        row["JockeyDistanceBucketAvgNorm"] = 0f;
                        row["LastJockeyDistanceBucketNormPos"] = 0f;
                        row["JockeyGoingDistanceWinRate"] = globalDefaultWinRate;
                        row["JockeyGoingDistanceAvgNorm"] = 0f;
                        row["LastJockeyGoingDistanceNormPos"] = 0f;
                    }

                    if (trainerId.HasValue)
                    {
                        var tcKey = (trainerId.Value, courseId);
                        _trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                        row["TrainerCourseWinRate"] = tcStat.starts > 0
                            ? _trainer.SmoothedWinRate(tcStat.wins, tcStat.starts)
                            : (hasTrainerStat ? trainerDefaultWinRate : globalDefaultWinRate);
                    }
                    else
                    {
                        row["TrainerCourseWinRate"] = globalDefaultWinRate;
                    }

                    if (jockeyId.HasValue)
                    {
                        var jcKey = (jockeyId.Value, courseId);
                        if (!_jockeyCourseStats.TryGetValue(jcKey, out var jcStat))
                        {
                            jcStat = (0, 0);
                        }
                        row["JockeyCourseWinRate"] = jcStat.starts > 0
                            ? _trainer.SmoothedWinRate(jcStat.wins, jcStat.starts)
                            : (hasJockeyStat ? jockeyDefaultWinRate : globalDefaultWinRate);
                    }
                    else
                    {
                        row["JockeyCourseWinRate"] = globalDefaultWinRate;
                    }
                    if (!_trainerFinishHistory.TryGetValue(trainerId ?? 0, out var trainerHistory))
                    {
                        trainerHistory = new List<short?>();
                    }
                    row["TrainerVolatility"] = CalculateVolatility(trainerHistory);

                    if (!_jockeyFinishHistory.TryGetValue(jockeyId ?? 0, out var jockeyHistory))
                    {
                        jockeyHistory = new List<short?>();
                    }
                    row["JockeyVolatility"] = CalculateVolatility(jockeyHistory);

                    row["HorseVolatility"] = CalculateVolatility(history.TakeLast(10).Select(h => h.Finish));
                    float trainerJockeyDefaultWinRate = hasTrainerStat && hasJockeyStat
                        ? (trainerDefaultWinRate + jockeyDefaultWinRate) / 2f
                        : hasTrainerStat ? trainerDefaultWinRate
                        : hasJockeyStat ? jockeyDefaultWinRate
                        : globalDefaultWinRate;

                    if (trainerId.HasValue && jockeyId.HasValue)
                    {
                        var pairKey = (trainerId.Value, jockeyId.Value);
                        if (!_trainerJockeyStats.TryGetValue(pairKey, out var pairStat))
                            pairStat = (0, 0);
                        row["TrainerJockeyWinRate"] = pairStat.starts > 0
                            ? _trainer.SmoothedWinRate(pairStat.wins, pairStat.starts)
                            : trainerJockeyDefaultWinRate;

                        var trainerJockeySurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                        if (!_trainerJockeySurfaceStats.TryGetValue(trainerJockeySurfaceKey, out var pairSurfaceStat))
                            pairSurfaceStat = (0, 0);
                        row["TrainerJockeySurfaceWinRate"] =
                            pairSurfaceStat.starts > 0
                                ? _trainer.SmoothedWinRate(pairSurfaceStat.wins, pairSurfaceStat.starts)
                                : trainerJockeyDefaultWinRate;

                        var trainerJockeyCourseKey = (trainerId.Value, jockeyId.Value, courseId);
                        if (!_trainerJockeyCourseStats.TryGetValue(trainerJockeyCourseKey, out var pairCourseStat))
                            pairCourseStat = (0, 0);
                        row["TrainerJockeyCourseWinRate"] =
                            pairCourseStat.starts > 0
                                ? _trainer.SmoothedWinRate(pairCourseStat.wins, pairCourseStat.starts)
                                : trainerJockeyDefaultWinRate;
                    }
                    else
                    {
                        row["TrainerJockeyWinRate"] = trainerJockeyDefaultWinRate;
                        row["TrainerJockeySurfaceWinRate"] = trainerJockeyDefaultWinRate;
                        row["TrainerJockeyCourseWinRate"] = trainerJockeyDefaultWinRate;
                    }

                }

                // =================================================================================
                // Stage 4: State update loop
                // Purpose: Update all historical stats after all features for the race are computed.
                // =================================================================================
                if (updateState)
                {
                    foreach (var row in rows)
                    {
                        // =================================================================================
                        // Stage 4.1: Re-fetch variables needed for state updates
                        // =================================================================================
                        int horseId = PreparedDataset.GetRequiredInt32(row, "HorseId");
                        var history = _horseHistory[horseId];
                        DateTime date = (DateTime)row["RaceDate"];
                        short? finish = row["FinishPos"] != null ? (short?)Convert.ToInt16(row["FinishPos"]) : null;
                        int runnerCount = row.TryGetValue("RunnerCount", out var runnerCountObj) &&
                                            PreparedDataset.TryConvertToInt32(runnerCountObj, out var runnerValue)
                            ? runnerValue
                            : raceStat.RunnerCount;
                        int? trainerId = row.TryGetValue("TrainerId", out var tObj) &&
                                          PreparedDataset.TryConvertToInt32(tObj, out var trainerValue)
                            ? trainerValue
                            : (int?)null;
                        int? jockeyId = row.TryGetValue("JockeyId", out var jObj) &&
                                         PreparedDataset.TryConvertToInt32(jObj, out var jockeyValue)
                            ? jockeyValue
                            : (int?)null;
                        int? classValue = row.TryGetValue("Class", out var classObj) &&
                            PreparedDataset.TryConvertToInt32(classObj, out var parsedClass)
                                ? parsedClass
                                : (int?)null;
                        bool classMissing = !classValue.HasValue;
                        int classVal = classValue ?? 0;
                        bool ratingMissing = !(row.TryGetValue("OfficialRating", out var ratingObj) && ratingObj != null);
                        float rating = ratingMissing ? raceStat.AvgRating : Convert.ToSingle(ratingObj);
                        if (!ratingMissing && (float.IsNaN(rating) || rating <= 0f))
                        {
                            ratingMissing = true;
                            rating = raceStat.AvgRating;
                        }
                        if ((ratingMissing && (rating <= 0f || float.IsNaN(rating))) ||
                                (!ratingMissing && (float.IsNaN(rating) || rating <= 0f)))
                        {
                            var classBaseline = ResolveClassRatingBaseline(classValue);
                            if (classBaseline.HasValue)
                            {
                                rating = classBaseline.Value;
                            }
                            else if (!float.IsNaN(raceStat.AvgRating) && raceStat.AvgRating > 0f)
                            {
                                rating = raceStat.AvgRating;
                            }
                        }
                        string? goingValue = NormalizeStringValue(
                                row.TryGetValue("Going", out var goingObj) ? goingObj : null);
                        bool goingMissing = string.IsNullOrEmpty(goingValue);
                        string going = goingValue ?? "Unknown";
                        string? surfaceValue = NormalizeStringValue(
                            row.TryGetValue("Surface", out var surfaceObj) ? surfaceObj : null);
                        bool surfaceMissing = string.IsNullOrEmpty(surfaceValue);
                        string surface = surfaceValue ?? "Unknown";
                        int? distanceYardsValue = row.TryGetValue("DistanceYards", out var distanceObj) &&
                            PreparedDataset.TryConvertToInt32(distanceObj, out var parsedDistance)
                                ? parsedDistance
                                : (int?)null;
                        bool distanceMissing = !distanceYardsValue.HasValue || distanceYardsValue.Value <= 0;
                        int distanceYards = distanceYardsValue ?? 0;
                        string bucket = distanceMissing ? "Unknown" : DistanceBucket(distanceYards);
                        int age = row.TryGetValue("Age", out var ageObj) && PreparedDataset.TryConvertToInt32(ageObj, out var ageValue)
                            ? ageValue
                            : 0;
                        float weight = row["WeightLbs"] == null ? 0f : Convert.ToSingle(row["WeightLbs"]);
                        bool drawMissing = row["Draw"] == null;
                        int draw = 0;
                        if (!drawMissing)
                        {
                            if (!PreparedDataset.TryConvertToInt32(row["Draw"], out draw))
                            {
                                drawMissing = true;
                            }
                        }
                        int courseId = row.TryGetValue("CourseId", out var courseObj) && PreparedDataset.TryConvertToInt32(courseObj, out var courseIdValue)
                            ? courseIdValue
                            : 0;

                        int? currentWinningTimeMs = row.TryGetValue("WinningTimeMs", out var winningObj) && PreparedDataset.TryConvertToInt32(winningObj, out var winningValue)
                            ? winningValue
                            : (int?)null;
                        bool winningTimeAvailable = currentWinningTimeMs.HasValue && currentWinningTimeMs.Value > 0;
                        float raceSpeed = winningTimeAvailable && distanceYards > 0
                            ? distanceYards / (float)currentWinningTimeMs!.Value
                            : 0f;
                        bool distanceBeatenKnown = row.TryGetValue("DistanceBeatenKnown", out var distanceKnownObj) &&
                            distanceKnownObj is bool distanceKnownBool && distanceKnownBool;
                        bool hasRunnerSpeed = winningTimeAvailable && distanceBeatenKnown;
                        float runnerSpeed = 0f;
                        if (hasRunnerSpeed)
                        {
                            float beaten = Convert.ToSingle(row["DistanceBeatenLengths"]);
                            float runnerTime = currentWinningTimeMs!.Value + beaten * MsPerLength;
                            runnerSpeed = runnerTime > 0f ? distanceYards / runnerTime : 0f;
                        }
                        float speedDiff = hasRunnerSpeed ? runnerSpeed - raceSpeed : 0f;

                        // =================================================================================
                        // Stage 4.2: Update horse, trainer, and jockey stats
                        // =================================================================================
                        float normFinish = (finish.HasValue && runnerCount > 1)
                            ? (runnerCount - finish.Value) / (float)(runnerCount - 1)
                            : 0f;

                        history.Add(new HistoryEntry(
                            date,
                            normFinish,
                            finish,
                            goingMissing ? null : going,
                            surfaceMissing ? null : surface,
                            courseId,
                            distanceMissing ? null : bucket,
                            classValue,
                            raceSpeed,
                            runnerSpeed,
                            speedDiff,
                            age,
                            finish.HasValue && finish.Value == 1,
                            rating,
                            weight,
                            hasRunnerSpeed,
                            winningTimeAvailable,
                            winningTimeAvailable ? (float?)currentWinningTimeMs : null,
                            distanceMissing ? null : (float?)distanceYards));

                        if (history.Count > HistoryLength)
                            history.RemoveAt(0);

                        if (!classMissing)
                        {
                            var horseClassKey = (horseId, classVal);
                            if (!_horseClassStats.TryGetValue(horseClassKey, out var horseClassStat))
                                horseClassStat = (0, 0, 0f, 0f);
                            horseClassStat.starts++;
                            horseClassStat.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) horseClassStat.wins++;
                            horseClassStat.lastNorm = normFinish;
                            _horseClassStats[horseClassKey] = horseClassStat;
                        }

                        if (!surfaceMissing)
                        {
                            if (!_surfaceStats.TryGetValue(horseId, out var sDict))
                            {
                                sDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                _surfaceStats[horseId] = sDict;
                            }
                            if (!sDict.TryGetValue(surface, out var sStats))
                                sStats = (0, 0, 0f, 0f);
                            sStats.starts++;
                            sStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) sStats.wins++;
                            sStats.lastNorm = normFinish;
                            sDict[surface] = sStats;
                        }

                        if (!goingMissing)
                        {
                            if (!_goingStats.TryGetValue(horseId, out var gDict))
                            {
                                gDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                _goingStats[horseId] = gDict;
                            }
                            if (!gDict.TryGetValue(going, out var gStats))
                                gStats = (0, 0, 0f, 0f);
                            gStats.starts++;
                            gStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) gStats.wins++;
                            gStats.lastNorm = normFinish;
                            gDict[going] = gStats;

                            string gcKey = going + "_" + courseId;
                            if (!_goingCourseStats.TryGetValue(horseId, out var gcDict))
                            {
                                gcDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                _goingCourseStats[horseId] = gcDict;
                            }
                            if (!gcDict.TryGetValue(gcKey, out var gcStats))
                                gcStats = (0, 0, 0f, 0f);
                            gcStats.starts++;
                            gcStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) gcStats.wins++;
                            gcStats.lastNorm = normFinish;
                            gcDict[gcKey] = gcStats;
                        }

                        if (!_courseStats.TryGetValue(horseId, out var cDict))
                        {
                            cDict = new();
                            _courseStats[horseId] = cDict;
                        }
                        if (!cDict.TryGetValue(courseId, out var cStats))
                            cStats = (0, 0, 0f, 0f);
                        cStats.starts++;
                        cStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) cStats.wins++;
                        cStats.lastNorm = normFinish;
                        cDict[courseId] = cStats;

                        if (!distanceMissing)
                        {
                            if (!_distanceBucketStats.TryGetValue(horseId, out var dDict))
                            {
                                dDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                _distanceBucketStats[horseId] = dDict;
                            }
                            if (!dDict.TryGetValue(bucket, out var dStats))
                                dStats = (0, 0, 0f, 0f);
                            dStats.starts++;
                            dStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) dStats.wins++;
                            dStats.lastNorm = normFinish;
                            dDict[bucket] = dStats;

                            if (!goingMissing)
                            {
                                var updateKey = (horseId, going, bucket);
                                _goingDistanceStats.TryGetValue(updateKey, out var updateStats);
                                updateStats.starts++;
                                updateStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) updateStats.wins++;
                                updateStats.lastNorm = normFinish;
                                _goingDistanceStats[updateKey] = updateStats;
                            }

                            _horseDistanceAll.TryGetValue(horseId, out var allDist);
                            allDist.sum += distanceYards;
                            allDist.count++;
                            _horseDistanceAll[horseId] = allDist;
                            _horseLastDistance[horseId] = distanceYards;

                            if (finish.HasValue && finish.Value == 1)
                            {
                                _horseDistanceWins.TryGetValue(horseId, out var winDist);
                                winDist.sum += distanceYards;
                                winDist.count++;
                                _horseDistanceWins[horseId] = winDist;
                            }
                        }

                        var drawKey = (courseId, bucket, draw);
                        _drawStats.TryGetValue(drawKey, out var drawStat);
                        drawStat.starts++;
                        if (finish.HasValue && finish.Value == 1) drawStat.wins++;
                        _drawStats[drawKey] = drawStat;

                        var baseKey = (courseId, bucket);
                        _drawBaselineStats.TryGetValue(baseKey, out var baseStat);
                        baseStat.starts++;
                        if (finish.HasValue && finish.Value == 1) baseStat.wins++;
                        _drawBaselineStats[baseKey] = baseStat;


                        if (trainerId.HasValue)
                        {
                            if (!_trainerStats.TryGetValue(trainerId.Value, out var trainerStat))
                                trainerStat = new RollingStat();
                            trainerStat.Starts++;
                            bool tWin = finish.HasValue && finish.Value == 1;
                            if (tWin) trainerStat.Wins++;
                            trainerStat.Recent.Enqueue((date, tWin));
                            while (trainerStat.Recent.Count > TrainerJockeyRecentStarts)
                                trainerStat.Recent.Dequeue();
                            _trainerStats[trainerId.Value] = trainerStat;

                            if (!_trainerFinishHistory.TryGetValue(trainerId.Value, out var trainerHistory))
                            {
                                trainerHistory = new List<short?>();
                                _trainerFinishHistory[trainerId.Value] = trainerHistory;
                            }
                            trainerHistory.Add(finish);
                            if (trainerHistory.Count > 10)
                            {
                                trainerHistory.RemoveAt(0);
                            }
                        }

                        if (jockeyId.HasValue)
                        {
                            if (!_jockeyStats.TryGetValue(jockeyId.Value, out var jockeyStat))
                                jockeyStat = new RollingStat();
                            jockeyStat.Starts++;
                            bool jWin = finish.HasValue && finish.Value == 1;
                            if (jWin) jockeyStat.Wins++;
                            jockeyStat.Recent.Enqueue((date, jWin));
                            while (jockeyStat.Recent.Count > TrainerJockeyRecentStarts)
                                jockeyStat.Recent.Dequeue();
                            _jockeyStats[jockeyId.Value] = jockeyStat;

                            if (!_jockeyFinishHistory.TryGetValue(jockeyId.Value, out var jockeyHistory))
                            {
                                jockeyHistory = new List<short?>();
                                _jockeyFinishHistory[jockeyId.Value] = jockeyHistory;
                            }
                            jockeyHistory.Add(finish);
                            if (jockeyHistory.Count > 10)
                            {
                                jockeyHistory.RemoveAt(0);
                            }
                        }

                        if (trainerId.HasValue && !classMissing)
                        {
                            var trainerClassKey = (trainerId.Value, classVal);
                            if (!_trainerClassStats.TryGetValue(trainerClassKey, out var trainerClassStat))
                                trainerClassStat = (0, 0, 0f, 0f);
                            trainerClassStat.starts++;
                            trainerClassStat.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) trainerClassStat.wins++;
                            trainerClassStat.lastNorm = normFinish;
                            _trainerClassStats[trainerClassKey] = trainerClassStat;
                        }

                        if (jockeyId.HasValue && !classMissing)
                        {
                            var jockeyClassKey = (jockeyId.Value, classVal);
                            if (!_jockeyClassStats.TryGetValue(jockeyClassKey, out var jockeyClassStat))
                                jockeyClassStat = (0, 0, 0f, 0f);
                            jockeyClassStat.starts++;
                            jockeyClassStat.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) jockeyClassStat.wins++;
                            jockeyClassStat.lastNorm = normFinish;
                            _jockeyClassStats[jockeyClassKey] = jockeyClassStat;
                        }

                        if (trainerId.HasValue)
                        {
                            if (!surfaceMissing)
                            {
                                if (!_trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict)) tsDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                if (!tsDict.TryGetValue(surface, out var tsStats)) tsStats = (0, 0, 0f, 0f);
                                tsStats.starts++;
                                tsStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) tsStats.wins++;
                                tsStats.lastNorm = normFinish;
                                tsDict[surface] = tsStats;
                                _trainerSurfaceStats[trainerId.Value] = tsDict;
                            }
                            if (!goingMissing)
                            {
                                if (!_trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict)) tgDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                if (!tgDict.TryGetValue(going, out var tgStats)) tgStats = (0, 0, 0f, 0f);
                                tgStats.starts++;
                                tgStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) tgStats.wins++;
                                tgStats.lastNorm = normFinish;
                                tgDict[going] = tgStats;
                                _trainerGoingStats[trainerId.Value] = tgDict;
                            }
                            if (!distanceMissing)
                            {
                                if (!_trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict)) tdDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                if (!tdDict.TryGetValue(bucket, out var tdStats)) tdStats = (0, 0, 0f, 0f);
                                tdStats.starts++;
                                tdStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) tdStats.wins++;
                                tdStats.lastNorm = normFinish;
                                tdDict[bucket] = tdStats;
                                _trainerDistanceStats[trainerId.Value] = tdDict;
                            }
                            var tcKey = (trainerId.Value, courseId);
                            _trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                            tcStat.starts++;
                            if (finish.HasValue && finish.Value == 1) tcStat.wins++;
                            _trainerCourseStats[tcKey] = tcStat;
                        }

                        if (jockeyId.HasValue)
                        {
                            if (!surfaceMissing)
                            {
                                if (!_jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict)) jsDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                if (!jsDict.TryGetValue(surface, out var jsStats)) jsStats = (0, 0, 0f, 0f);
                                jsStats.starts++;
                                jsStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) jsStats.wins++;
                                jsStats.lastNorm = normFinish;
                                jsDict[surface] = jsStats;
                                _jockeySurfaceStats[jockeyId.Value] = jsDict;
                            }
                            if (!goingMissing)
                            {
                                if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict)) jgDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                if (!jgDict.TryGetValue(going, out var jgStats)) jgStats = (0, 0, 0f, 0f);
                                jgStats.starts++;
                                jgStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) jgStats.wins++;
                                jgStats.lastNorm = normFinish;
                                jgDict[going] = jgStats;
                                _jockeyGoingStats[jockeyId.Value] = jgDict;
                            }
                            if (!distanceMissing)
                            {
                                if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict)) jdDict = new Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>(StringComparer.OrdinalIgnoreCase);
                                if (!jdDict.TryGetValue(bucket, out var jdStats)) jdStats = (0, 0, 0f, 0f);
                                jdStats.starts++;
                                jdStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) jdStats.wins++;
                                jdStats.lastNorm = normFinish;
                                jdDict[bucket] = jdStats;
                                _jockeyDistanceStats[jockeyId.Value] = jdDict;
                            }
                            if (!goingMissing && !distanceMissing)
                            {
                                var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                                _jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat);
                                jockeyGoingDistanceStat.starts++;
                                jockeyGoingDistanceStat.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) jockeyGoingDistanceStat.wins++;
                                jockeyGoingDistanceStat.lastNorm = normFinish;
                                _jockeyGoingDistanceStats[jockeyGoingDistanceKey] = jockeyGoingDistanceStat;
                            }
                            var jcKey = (jockeyId.Value, courseId);
                            if (!_jockeyCourseStats.TryGetValue(jcKey, out var jcStat))
                            {
                                jcStat = (0, 0);
                            }
                            jcStat.starts++;
                            if (finish.HasValue && finish.Value == 1)
                            {
                                jcStat.wins++;
                            }
                            _jockeyCourseStats[jcKey] = jcStat;
                        }

                        if (trainerId.HasValue && jockeyId.HasValue)
                        {
                            var pairKey = (trainerId.Value, jockeyId.Value);
                            _trainerJockeyStats.TryGetValue(pairKey, out var pairStat);
                            pairStat.starts++;
                            if (finish.HasValue && finish.Value == 1)
                            {
                                pairStat.wins++;
                            }
                            _trainerJockeyStats[pairKey] = pairStat;

                            if (!surfaceMissing)
                            {
                                var pairSurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                                _trainerJockeySurfaceStats.TryGetValue(pairSurfaceKey, out var pairSurfaceStat);
                                pairSurfaceStat.starts++;
                                if (finish.HasValue && finish.Value == 1) pairSurfaceStat.wins++;
                                _trainerJockeySurfaceStats[pairSurfaceKey] = pairSurfaceStat;
                            }

                            var pairCourseKey = (trainerId.Value, jockeyId.Value, courseId);
                            _trainerJockeyCourseStats.TryGetValue(pairCourseKey, out var pairCourseStat);
                            pairCourseStat.starts++;
                            if (finish.HasValue && finish.Value == 1) pairCourseStat.wins++;
                            _trainerJockeyCourseStats[pairCourseKey] = pairCourseStat;
                        }
                    }
                }

                if (includeRace)
                {
                    float ResolveRunnerSpeed(Dictionary<string, object?> row)
                    {
                        if (row == null)
                        {
                            return 0f;
                        }
                        if (!row.TryGetValue("AvgSpeedLast5", out var value) || value == null)
                        {
                            return 0f;
                        }
                        try
                        {
                            return Convert.ToSingle(value);
                        }
                        catch
                        {
                            return 0f;
                        }
                    }

                    float raceAvgWinRate = rows
                        .Select(r => r.ContainsKey("WinRateLast5") && r["WinRateLast5"] != null ? Convert.ToSingle(r["WinRateLast5"]) : 0f)
                        .DefaultIfEmpty(0f)
                        .Average();

                    foreach (var raceRow in rows)
                    {
                        var runnerSpeed = ResolveRunnerSpeed(raceRow);
                        raceRow["RaceAvgSpeedLast5"] = runnerSpeed;
                        raceRow["RaceAvgWinRateLast5"] = raceAvgWinRate;

                        if (!raceRow.TryGetValue("RatingSlope", out var slope) || slope is null)
                        {
                            raceRow["RatingSlope"] = 0f;
                        }
                        if (!raceRow.TryGetValue("Age", out var ageValue) || ageValue is null)
                        {
                            raceRow["Age"] = 0;
                        }
                        if (!raceRow.TryGetValue("RaceAvgSpeedLast5", out var raceAvgSpeedValue) || raceAvgSpeedValue is null)
                        {
                            raceRow["RaceAvgSpeedLast5"] = 0f;
                        }

                        TrimRunnerRow(raceRow, _preserveKeys);
                    }
                }
            }
            private static bool TryGetTimeOfDay(Dictionary<string, object?> row, out TimeSpan timeOfDay)
            {
                if (TryReadTimeValue(row, "ActualOff", out timeOfDay) ||
                    TryReadTimeValue(row, "ScheduledOff", out timeOfDay))
                {
                    return true;
                }

                if (row.TryGetValue("RaceDate", out var dateObj) && dateObj != null)
                {
                    switch (dateObj)
                    {
                        case DateTime dt:
                            timeOfDay = dt.TimeOfDay;
                            return true;
                        case DateTimeOffset dto:
                            timeOfDay = dto.TimeOfDay;
                            return true;
                    }
                }

                timeOfDay = default;
                return false;
            }

            public static bool TryReadTimeValue(Dictionary<string, object?> row, string key, out TimeSpan time)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    time = default;
                    return false;
                }

                switch (value)
                {
                    case TimeSpan ts:
                        time = ts;
                        return true;
                    case DateTime dt:
                        time = dt.TimeOfDay;
                        return true;
                    case DateTimeOffset dto:
                        time = dto.TimeOfDay;
                        return true;
                    case string s when TimeSpan.TryParse(s, CultureInfo.InvariantCulture, out var parsed):
                        time = parsed;
                        return true;
                    default:
                        time = default;
                        return false;
                }
            }
            private static string DistanceBucket(int yards)
                => yards < 1760 ? "Sprint" : yards < 2640 ? "Middle" : "Long";
            private static string NormalizeHorseNameForLookup(string horseName)
            {
                if (string.IsNullOrWhiteSpace(horseName))
                {
                    return string.Empty;
                }

                var trimmed = horseName.Trim();
                var withoutBracketed = HorseNameBracketTextRegex.Replace(trimmed, " ");
                var collapsed = HorseNameWhitespaceRegex.Replace(withoutBracketed, " ").Trim();

                return string.IsNullOrEmpty(collapsed) ? trimmed : collapsed;
            }
            private RaceStats ComputeRaceStats(List<Dictionary<string, object?>> rows)
            {
                int cnt = rows.Count;
                var drawValues = rows.Where(r => r["Draw"] != null).Select(r => Convert.ToSingle(r["Draw"])).ToList();
                float avgDraw = drawValues.Count > 0 ? drawValues.Average() : 0f;
                var weightValues = rows.Where(r => r["WeightLbs"] != null).Select(r => Convert.ToSingle(r["WeightLbs"])).ToList();
                float avgWeight = weightValues.Count > 0 ? weightValues.Average() : 0f;
                float minWeight = weightValues.Count > 0 ? weightValues.Min() : 0f;
                float maxWeight = weightValues.Count > 0 ? weightValues.Max() : 0f;
                var saddleclothValues = rows.Where(r => r.TryGetValue("SaddleclothNumber", out var scObj) && scObj != null).Select(r => Convert.ToSingle(r["SaddleclothNumber"])).ToList();
                float avgSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Average() : 0f;
                float minSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Min() : 0f;
                float maxSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Max() : 0f;
                float avgAge = rows.Where(r => r["Age"] != null).Select(r => Convert.ToSingle(r["Age"])).DefaultIfEmpty(0f).Average();
                var ratingValues = rows.Where(r => r.TryGetValue("OfficialRating", out var orObj) && orObj != null).Select(r => Convert.ToSingle(r["OfficialRating"])).ToList();
                float avgRating = ratingValues.DefaultIfEmpty(0f).Average();
                float stdRating = 0f;
                if (ratingValues.Count > 0)
                {
                    float variance = ratingValues.Select(r => (r - avgRating) * (r - avgRating)).Average();
                    stdRating = (float)Math.Sqrt(variance);
                }
                float purse = rows.Select(r => r.TryGetValue("Purse", out var pObj) && pObj != null ? Convert.ToSingle(pObj) : 0f).FirstOrDefault();
                float avgSaddle = saddleclothValues.Count > 0 ? saddleclothValues.Average() : 0f;
                float minSaddle = saddleclothValues.Count > 0 ? saddleclothValues.Min() : 0f;
                float maxSaddle = saddleclothValues.Count > 0 ? saddleclothValues.Max() : 0f;
                var winRateLast5Values = rows.Select(r => r.ContainsKey("WinRateLast5") && r["WinRateLast5"] != null ? Convert.ToSingle(r["WinRateLast5"]) : 0f).ToList();
                float avgWinRateLast5 = winRateLast5Values.Any() ? winRateLast5Values.Average() : 0f;
                float stdDevWinRateLast5 = 0f;
                if (winRateLast5Values.Count > 1)
                {
                    float variance = winRateLast5Values.Select(r => (r - avgWinRateLast5) * (r - avgWinRateLast5)).Average();
                    stdDevWinRateLast5 = (float)Math.Sqrt(variance);
                }
                var speedLast5Values = rows.Select(r => r.ContainsKey("AvgSpeedLast5") && r["AvgSpeedLast5"] != null ? Convert.ToSingle(r["AvgSpeedLast5"]) : 0f).ToList();
                float avgSpeedLast5 = speedLast5Values.Any() ? speedLast5Values.Average() : 0f;
                float stdDevSpeedLast5 = 0f;
                if (speedLast5Values.Count > 1)
                {
                    float variance = speedLast5Values.Select(r => (r - avgSpeedLast5) * (r - avgSpeedLast5)).Average();
                    stdDevSpeedLast5 = (float)Math.Sqrt(variance);
                }
                return new RaceStats(
                    cnt,
                    avgDraw,
                    avgWeight,
                    minWeight,
                    maxWeight,
                    avgAge,
                    avgRating,
                    stdRating,
                    purse,
                    saddleclothValues.Count > 0,
                    avgSaddle,
                    minSaddle,
                    maxSaddle,
                    weightValues.Count > 0,
                    avgWinRateLast5,
                    stdDevWinRateLast5,
                    avgSpeedLast5,
                    stdDevSpeedLast5);
            }
            internal static float? TestParseDistanceBeaten(string text)
            {
                return ParseDistanceBeaten(text);
            }
            private readonly record struct RaceStats(
                       int RunnerCount,
                       float AvgDraw,
                       float AvgWeight,
                       float MinWeight,
                       float MaxWeight,
                       float AvgAge,
                       float AvgRating,
                       float StdRating,
                       float TotalPurse,
                       bool HasSaddleclothStats,
                       float AvgSaddlecloth,
                       float MinSaddlecloth,
                       float MaxSaddlecloth,
                       bool HasWeightStats,
                       float AvgWinRateLast5,
                       float StdDevWinRateLast5,
                       float AvgSpeedLast5,
                       float StdDevSpeedLast5);


            private static float? ParseDistanceBeaten(string text)
            {
                if (string.IsNullOrWhiteSpace(text))
                {
                    return null;
                }

                var lowered = text.Trim().ToLowerInvariant();
                if (DistanceTokenValues.TryGetValue(lowered, out var directValue))
                {
                    return directValue;
                }

                lowered = lowered.Replace('\u00A0', ' ');
                lowered = lowered.Replace("¼", ".25").Replace("½", ".5").Replace("¾", ".75");
                lowered = lowered.Replace("short head", "short-head");
                lowered = lowered.Replace("short neck", "short-neck");
                lowered = lowered.Replace("dead heat", "dead-heat");
                lowered = DistanceHyphenBetweenDigitsRegex.Replace(lowered, " ");

                static string TrimTrailingLetters(string value)
                {
                    var end = value.Length;
                    while (end > 0 && char.IsLetter(value[end - 1]))
                    {
                        end--;
                    }

                    return end == value.Length ? value : value[..end];
                }

                double total = 0d;
                var hasValue = false;

                foreach (var rawToken in lowered.Split(DistanceTokenSeparators, StringSplitOptions.RemoveEmptyEntries))
                {
                    var token = rawToken.Trim(DistanceTokenTrimChars);
                    if (token.Length == 0)
                    {
                        continue;
                    }

                    if (DistanceTokenValues.TryGetValue(token, out var mappedValue))
                    {
                        total += mappedValue;
                        hasValue = true;
                        continue;
                    }

                    var numericCandidate = TrimTrailingLetters(token);
                    if (numericCandidate.Length == 0)
                    {
                        continue;
                    }

                    if (DistanceTokenValues.TryGetValue(numericCandidate, out mappedValue))
                    {
                        total += mappedValue;
                        hasValue = true;
                        continue;
                    }

                    if (numericCandidate.Contains('/'))
                    {
                        var frac = numericCandidate.Split('/');
                        if (frac.Length == 2 &&
                            double.TryParse(frac[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var numerator) &&
                            double.TryParse(frac[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var denominator) &&
                            denominator != 0d)
                        {
                            total += numerator / denominator;
                            hasValue = true;
                        }

                        continue;
                    }

                    if (double.TryParse(numericCandidate, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out var number))
                    {
                        total += number;
                        hasValue = true;
                    }
                }

                if (!hasValue)
                {
                    return null;
                }

                return (float)total;
            }

            internal static float? TestParseDistanceBeatenInternal(string text)
                => ParseDistanceBeaten(text);

            private float SmoothedWinRate(int wins, int starts)
                => _trainer.ComputeSmoothedWinRate(wins, starts);


            private static object? NormalizeDbValue(object? value)
            {
                if (value is null || value is DBNull)
                {
                    return null;
                }

                switch (value)
                {
                    case string s:
                        {
                            var trimmed = s.Trim();
                            return trimmed.Length == 0 ? null : trimmed;
                        }
                    case char c:
                        {
                            var text = c.ToString().Trim();
                            return text.Length == 0 ? null : text;
                        }
                    case char[] chars:
                        {
                            if (chars.Length == 0)
                            {
                                return null;
                            }

                            var text = new string(chars).Trim();
                            return text.Length == 0 ? null : text;
                        }
                    case ReadOnlyMemory<char> memory:
                        {
                            var text = memory.ToString().Trim();
                            return text.Length == 0 ? null : text;
                        }
                    case SqlString sqlString when !sqlString.IsNull:
                        {
                            var text = sqlString.Value?.Trim();
                            return string.IsNullOrEmpty(text) ? null : text;
                        }
                    case SqlChars sqlChars when !sqlChars.IsNull:
                        {
                            var text = new string(sqlChars.Value ?? Array.Empty<char>()).Trim();
                            return text.Length == 0 ? null : text;
                        }
                    case SqlBoolean sqlBool when !sqlBool.IsNull:
                        return sqlBool.Value;
                    case SqlByte sqlByte when !sqlByte.IsNull:
                        return sqlByte.Value;
                    case SqlInt16 sqlInt16 when !sqlInt16.IsNull:
                        return sqlInt16.Value;
                    case SqlInt32 sqlInt32 when !sqlInt32.IsNull:
                        return sqlInt32.Value;
                    case SqlInt64 sqlInt64 when !sqlInt64.IsNull:
                        return sqlInt64.Value;
                    case SqlSingle sqlSingle when !sqlSingle.IsNull:
                        return sqlSingle.Value;
                    case SqlDouble sqlDouble when !sqlDouble.IsNull:
                        return sqlDouble.Value;
                    case SqlDecimal sqlDecimal when !sqlDecimal.IsNull:
                        return sqlDecimal.Value;
                    case SqlMoney sqlMoney when !sqlMoney.IsNull:
                        return sqlMoney.Value;
                    case SqlDateTime sqlDate when !sqlDate.IsNull:
                        return sqlDate.Value;
                }

                return value;
            }

            private static void Shuffle<T>(IList<T> list, Random random)
            {
                if (list is null)
                    throw new ArgumentNullException(nameof(list));
                if (random is null)
                    throw new ArgumentNullException(nameof(random));

                for (int i = list.Count - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (list[i], list[j]) = (list[j], list[i]);
                }
            }

            private static readonly HashSet<string> NonTrainingKeys = new HashSet<string>
        {
            "HorseId",
            "CourseId",
            "TrainerId",
            "JockeyId",
            "OutcomeCode",
            "DistanceBeatenText",
            "SP_Fraction",
            "SP_Decimal",
            "OpeningFraction",
            "TouchedHighFraction",
            "TouchedLowFraction",
            "HorseName",
            "JockeyName",
            "TrainerName",
            "Title",
            "RaceMonth",
            "RaceDayOfWeek",
            "Season",
            "ActualOff",
            "ScheduledOff",
            "RaceDate",
            "CourseName",
            "DistanceText",
            "Status",
            "WeightText",
            "FavTag",
            "SaddleclothNumber",
            "Purse"
        };

            private static readonly ISet<string> BackfillRequiredKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "HorseId",
            "HorseName",
            "TrainerId",
            "TrainerName",
            "JockeyId",
            "JockeyName",
            "CourseId",
            "RaceDate"
        };

            private static void TrimRunnerRow(Dictionary<string, object?> row, ISet<string>? preserveKeys = null)
            {
                if (row is null)
                    throw new ArgumentNullException(nameof(row));

                foreach (var key in NonTrainingKeys)
                {
                    if (preserveKeys != null && preserveKeys.Contains(key))
                    {
                        continue;
                    }

                    row.Remove(key);
                }
            }
            private static HashSet<int>? ToNonNullableSet(ISet<int?>? source)
            {
                if (source is null)
                {
                    return null;
                }

                var result = new HashSet<int>();
                foreach (var value in source)
                {
                    if (value.HasValue)
                    {
                        result.Add(value.Value);
                    }
                }

                return result;
            }
            public PreparedDataset PrepareDataset(
                ISet<int?>? includeRaceIds = null,
                ISet<int?>? stateRaceWhitelist = null,
                bool includeIdentifiers = false,
                bool applyRepositoryBackfills = true)
            {
                using var conn = new SqlConnection(_trainer._connectionString);
                conn.Open();

                var includeRaceIdSet = ToNonNullableSet(includeRaceIds);
                var stateRaceWhitelistSet = ToNonNullableSet(stateRaceWhitelist);
                var requiredRaceIds = new HashSet<int>();
                if (includeRaceIdSet is not null)
                {
                    requiredRaceIds.UnionWith(includeRaceIdSet);
                }

                if (stateRaceWhitelistSet is not null)
                {
                    requiredRaceIds.UnionWith(stateRaceWhitelistSet);
                }
                var raceColumns = PreparedDataset.LoadColumnNames(conn, "Race");
                var runnerColumns = PreparedDataset.LoadColumnNames(conn, "RunnerResult");
                string scheduledOffColumn = raceColumns.Contains("ScheduledOff")
                    ? "r.ScheduledOff AS ScheduledOff"
                    : "CAST(NULL AS time(0)) AS ScheduledOff";
                string actualOffColumn = raceColumns.Contains("ActualOff")
                    ? "r.ActualOff AS ActualOff"
                    : "CAST(NULL AS time(0)) AS ActualOff";
                string titleColumn = SelectColumn(raceColumns, "r", "Title", "nvarchar(512)");
                string raceTypeColumn = SelectColumn(raceColumns, "r", "RaceType", "nvarchar(128)");
                string classColumn = SelectColumn(raceColumns, "r", "Class", "int");
                string surfaceColumn = SelectColumn(raceColumns, "r", "Surface", "nvarchar(64)");
                string goingColumn = SelectColumn(raceColumns, "r", "Going", "nvarchar(30)");
                string distanceYardsColumn = SelectColumn(raceColumns, "r", "DistanceYards", "int");
                string distanceTextColumn = SelectColumn(raceColumns, "r", "DistanceText", "nvarchar(64)");
                string runnerCountColumn = SelectColumn(raceColumns, "r", "RunnerCount", "int");
                string statusColumn = SelectColumn(raceColumns, "r", "Status", "nvarchar(32)");
                string winningTimeColumn = SelectColumn(raceColumns, "r", "WinningTimeMs", "int");
                string saddleclothColumn = SelectColumn(runnerColumns, "rr", "SaddleclothNumber", "int");
                string drawColumn = SelectColumn(runnerColumns, "rr", "Draw", "int");
                string ageColumn = SelectColumn(runnerColumns, "rr", "Age", "int");
                string weightLbsColumn = SelectColumn(runnerColumns, "rr", "WeightLbs", "int");
                string weightTextColumn = SelectColumn(runnerColumns, "rr", "WeightText", "nvarchar(50)");
                string officialRatingColumn = SelectColumn(runnerColumns, "rr", "OfficialRating", "int", "OfficialRating");
                string outcomeCodeColumn = SelectColumn(runnerColumns, "rr", "OutcomeCode", "nvarchar(50)");
                string distanceBeatenTextColumn = SelectColumn(runnerColumns, "rr", "DistanceBeatenText", "nvarchar(50)");
                string distanceBeatenLengthsColumn = SelectColumn(runnerColumns, "rr", "DistanceBeatenLengths", "decimal(9,4)");
                string spFractionColumn = SelectColumn(runnerColumns, "rr", "SP_Fraction", "nvarchar(50)");
                string spDecimalColumn = SelectColumn(runnerColumns, "rr", "SP_Decimal", "decimal(18,6)");
                string favTagColumn = SelectColumn(runnerColumns, "rr", "FavTag", "nvarchar(16)");
                string openingFractionColumn = SelectColumn(runnerColumns, "rr", "OpeningFraction", "nvarchar(50)");
                string touchedHighColumn = SelectColumn(runnerColumns, "rr", "TouchedHighFraction", "nvarchar(50)");
                string touchedLowColumn = SelectColumn(runnerColumns, "rr", "TouchedLowFraction", "nvarchar(50)");
                var filters = new List<string>();
                var parameters = new DynamicParameters();
                bool filterByRaceIds = requiredRaceIds.Count > 0;
                if (filterByRaceIds)
                {
                    filters.Add("r.RaceId IN @FilteredRaceIds");
                }

                string whereClause = filters.Count > 0
                    ? string.Concat(" WHERE ", string.Join(" AND ", filters))
                    : string.Empty;
                var sql = $@"SELECT c.Name AS CourseName,
                               h.Name AS HorseName,
                               j.Name AS JockeyName,
                               t.Name AS TrainerName,
                               r.RaceId,
                               r.CourseId,
                               r.RaceDate,
                               {scheduledOffColumn},
                               {actualOffColumn},
                               {titleColumn},
                               {raceTypeColumn},
                               {classColumn},
                               {surfaceColumn},
                               {goingColumn},
                               {distanceYardsColumn},
                               {distanceTextColumn},
                               {runnerCountColumn},
                               {statusColumn},
                               {winningTimeColumn},
                               rr.HorseId,
                               rr.TrainerId,
                               rr.JockeyId,
                               {saddleclothColumn},
                               {drawColumn},
                               {ageColumn},
                               {weightLbsColumn},
                               {weightTextColumn},
                                {officialRatingColumn},
                               rr.FinishPos,
                               {outcomeCodeColumn},
                               {distanceBeatenTextColumn},
                               {distanceBeatenLengthsColumn},
                               {spFractionColumn},
                               {spDecimalColumn},
                               {favTagColumn},
                               {openingFractionColumn},
                               {touchedHighColumn},
                               {touchedLowColumn}
                        FROM Race r
                        JOIN Course c ON r.CourseId = c.CourseId
                        JOIN RunnerResult rr ON r.RaceId = rr.RaceId
                        LEFT JOIN Horse h ON rr.HorseId = h.HorseId
                        LEFT JOIN Trainer t ON rr.TrainerId = t.TrainerId
                        LEFT JOIN Jockey j ON rr.JockeyId = j.JockeyId
                        {whereClause}
                        ORDER BY r.RaceDate, r.RaceId, rr.RunnerResultId";

                ISet<string>? identifierKeys = null;
                if (includeIdentifiers)
                {
                    identifierKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "HorseName",
                "HorseId",
                "TrainerName",
                "TrainerId",
                "JockeyName",
                "JockeyId",
                "SaddleclothNumber",
                "Draw"
            };
                }

                var featureState = new FeatureEngineeringState(_trainer, identifierKeys);
                var races = new List<PreparedRace>();
                var rnd = new Random();
                var currentRows = new List<Dictionary<string, object?>>();
                int? currentRaceId = null;
                bool ShouldInclude(int raceId) => includeRaceIdSet is null || includeRaceIdSet.Contains(raceId);
                bool ShouldUpdate(int raceId) => stateRaceWhitelistSet is null || stateRaceWhitelistSet.Contains(raceId);
                const int raceProgressInterval = 250;
                long totalRunnerRows = 0;
                long includedRunnerRows = 0;
                int processedRaceCount = 0;
                int includedRaceCount = 0;

                Console.WriteLine("[TrainAI] Loading races and runner rows from the database...");

                void FinalizeRace(List<Dictionary<string, object?>> rows, int raceId)
                {
                    if (rows is null || rows.Count == 0)
                    {
                        return;
                    }

                    Shuffle(rows, rnd);
                    bool include = ShouldInclude(raceId);
                    bool update = ShouldUpdate(raceId);
                    if (!include && !update)
                    {
                        return;
                    }

                    ResolveHorseIdentifiers(conn, rows);
                    featureState.ProcessRace(rows, include, update);

                    processedRaceCount++;
                    if (include)
                    {
                        races.Add(new PreparedRace(raceId, rows));
                        includedRaceCount++;
                        includedRunnerRows += rows.Count;
                    }
                }
                void ProcessRecords(IEnumerable<dynamic> records)
                {
                    foreach (var record in records)
                    {
                        var source = (IDictionary<string, object?>)record;
                        var row = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                        foreach (var kvp in source)
                        {
                            row[kvp.Key] = NormalizeDbValue(kvp.Value);
                        }

                        if (!PreparedDataset.TryGetRequiredInt32(row, "RaceId", out var raceId))
                        {
                            continue;
                        }

                        if (currentRaceId.HasValue && raceId != currentRaceId.Value)
                        {
                            FinalizeRace(currentRows, currentRaceId.Value);
                            currentRows = new List<Dictionary<string, object?>>();
                        }

                        currentRows.Add(row);
                        totalRunnerRows++;
                        currentRaceId = raceId;
                    }
                }

                const int filteredRaceIdChunkSize = 2000;
                if (!filterByRaceIds || requiredRaceIds.Count <= filteredRaceIdChunkSize)
                {
                    if (filterByRaceIds)
                    {
                        var chunkParameters = new DynamicParameters(parameters);
                        chunkParameters.Add("FilteredRaceIds", requiredRaceIds.ToArray());
                        ProcessRecords(conn.Query(sql, chunkParameters, commandTimeout: 600000000, buffered: false));
                    }
                    else
                    {
                        ProcessRecords(conn.Query(sql, parameters, commandTimeout: 600000000, buffered: false));
                    }
                }
                else
                {
                    var raceIdList = requiredRaceIds.ToList();
                    raceIdList.Sort();
                    for (var offset = 0; offset < raceIdList.Count; offset += filteredRaceIdChunkSize)
                    {
                        var chunkLength = Math.Min(filteredRaceIdChunkSize, raceIdList.Count - offset);
                        var chunk = raceIdList.GetRange(offset, chunkLength).ToArray();
                        var chunkParameters = new DynamicParameters(parameters);
                        chunkParameters.Add("FilteredRaceIds", chunk);
                        ProcessRecords(conn.Query(sql, chunkParameters, commandTimeout: 600000000, buffered: false));
                    }
                }

                if (currentRaceId.HasValue && currentRows.Count > 0)
                {
                    FinalizeRace(currentRows, currentRaceId.Value);
                }
                Console.WriteLine(
                    $"[TrainAI] Raw data streaming complete. Feature engineering ran for {processedRaceCount} races ({includedRaceCount} included) across {totalRunnerRows:N0} runner rows.");

                if (applyRepositoryBackfills)
                {
                    ApplyRepositoryBackfills(races, includeIdentifiers);
                }
                else if (!includeIdentifiers)
                {
                    foreach (var race in races)
                    {
                        var rows = race?.Rows;
                        if (rows is null)
                        {
                            continue;
                        }

                        foreach (var row in rows)
                        {
                            if (row is null)
                            {
                                continue;
                            }

                            foreach (var key in BackfillRequiredKeys)
                            {
                                row.Remove(key);
                            }
                        }
                    }
                }

                Console.WriteLine(
                    $"[TrainAI] Dataset preparation finished. {races.Count} races ready ({includedRunnerRows:N0} runner rows retained).");
                return new PreparedDataset(races);
            }
            private void ApplyRepositoryBackfills(List<PreparedRace> races, bool includeIdentifiers)
            {
                if (races is null || races.Count == 0)
                {
                    Console.WriteLine("[TrainAI] No races supplied for repository backfills.");
                    return;
                }

                var repository = _trainer.EnsureRacingRepository();
                if (repository is null)
                {
                    Console.WriteLine("[TrainAI] Skipping repository backfills because the racing repository is unavailable.");
                    return;
                }

                Console.WriteLine($"[TrainAI] Applying repository backfills for {races.Count} races.");
                var horseIdentitySummary = CollectHorseIdentitySummary(races);
                var prefetchedHistoryByHorseId = PrefetchHorseHistoryByHorseId(horseIdentitySummary);
                if (prefetchedHistoryByHorseId.Count > 0)
                {
                    Console.WriteLine($"[TrainAI] Prefetched historical performance for {prefetchedHistoryByHorseId.Count} horses.");
                }
                var winRateCache = new ConcurrentDictionary<HorseCacheKey, (float WinRate, int Wins, int Starts)?>(HorseCacheKeyComparer.Instance);
                var speedCache = new ConcurrentDictionary<HorseCacheKey, float?>(HorseCacheKeyComparer.Instance);
                var distanceCache = new ConcurrentDictionary<HorseCacheKey, int?>(HorseCacheKeyComparer.Instance);
                if (repository is not null)
                {
                    PreloadRepositoryCaches(
                        repository,
                        horseIdentitySummary.Identities,
                        prefetchedHistoryByHorseId,
                        winRateCache,
                        speedCache,
                        distanceCache);
                }
                const int backfillProgressInterval = 250;
                int processed = 0;
                var totalRaces = races.Count;
                var stopwatch = Stopwatch.StartNew();
                var maxThreads = Math.Max(1, Environment.ProcessorCount);
                var parallelOptions = new ParallelOptions
                {
                    MaxDegreeOfParallelism = Math.Min(maxThreads, Math.Max(1, totalRaces))
                };

                void ReportProgress(int currentCount)
                {
                    if (currentCount <= 0)
                    {
                        return;
                    }

                    if (currentCount % backfillProgressInterval != 0 && currentCount != totalRaces)
                    {
                        return;
                    }

                    var elapsed = stopwatch.Elapsed;
                    if (elapsed.Ticks == 0)
                    {
                        Console.WriteLine($"[TrainAI] Backfill progress: {currentCount}/{totalRaces} races enriched.");
                        return;
                    }

                    var averagePerRace = TimeSpan.FromTicks(elapsed.Ticks / currentCount);
                    var remainingRaces = Math.Max(0, totalRaces - currentCount);
                    var estimatedRemaining = remainingRaces > 0
                        ? TimeSpan.FromTicks(averagePerRace.Ticks * remainingRaces)
                        : TimeSpan.Zero;

                    Console.WriteLine(
                        $"[TrainAI] Backfill progress: {currentCount}/{totalRaces} races enriched. " +
                        $"Elapsed {FormatDuration(elapsed)}, avg {averagePerRace.TotalSeconds:F5}s/race, ETA {FormatDuration(estimatedRemaining)}.");
                }
                var raceMetadataLookup = BuildRaceMetadataLookup(repository, races);
                Parallel.ForEach(races, parallelOptions, race =>
                {
                    var raceRows = race?.Rows;
                    if (raceRows == null || raceRows.Count == 0)
                    {
                        var skippedCount = Interlocked.Increment(ref processed);
                        ReportProgress(skippedCount);
                        return;
                    }
                    double winRateSum = 0d;
                    int winRateCount = 0;
                    var rowCount = raceRows.Count;
                    var perRunnerWinRate = new float?[rowCount];

                    for (var rowIndex = 0; rowIndex < rowCount; rowIndex++)
                    {
                        var row = raceRows[rowIndex];
                        if (row is null)
                        {
                            continue;
                        }
                        if (PreparedDataset.TryGetRequiredInt32(row, "RaceId", out var raceId) &&
                            raceMetadataLookup.TryGetValue(raceId, out var metadata) &&
                            metadata != null)
                        {
                            ApplyRaceMetadataBackfill(row, metadata);
                        }
                        bool distanceBackfilled = false;

                        if (TryGetHorseIdentity(row, out var identity) &&
                            identity.HorseId.HasValue &&
                            identity.RaceDate.HasValue &&
                            prefetchedHistoryByHorseId.TryGetValue(identity.HorseId.Value, out var history) &&
                            history.Count > 0)
                        {
                            var stats = ComputePrefetchedHorseStats(history, identity.RaceDate.Value);
                            if (stats.HasValue)
                            {
                                if (stats.Value.WinRate.HasValue)
                                {
                                    var sanitized = ClampProbability(stats.Value.WinRate.Value);
                                    if (!TryGetFloat(row, "WinRateLast5", out var existingWinRate) || existingWinRate <= 0f)
                                    {
                                        row["WinRateLast5"] = sanitized;
                                    }
                                }

                                if (stats.Value.AverageSpeed.HasValue)
                                {
                                    if (!TryGetFloat(row, "AvgSpeedLast5", out var existingSpeed) ||
                                        existingSpeed <= 0f ||
                                        float.IsNaN(existingSpeed) ||
                                        float.IsInfinity(existingSpeed))
                                    {
                                        row["AvgSpeedLast5"] = stats.Value.AverageSpeed.Value;
                                    }

                                    if (TryGetFloat(row, "AvgSpeedLast5", out var finalSpeed) &&
                                        finalSpeed > 0f &&
                                        !float.IsNaN(finalSpeed) &&
                                        !float.IsInfinity(finalSpeed))
                                    {
                                        row["RaceAvgSpeedLast5"] = finalSpeed;
                                    }
                                }

                                if (stats.Value.LastDistanceYards.HasValue &&
                                    PreparedDataset.TryGetValueWithAliases(row, "DistanceYards", out var distanceObj, requireNonNull: true) &&
                                    distanceObj != null &&
                                    PreparedDataset.TryConvertToInt32(distanceObj, out var currentDistance) &&
                                    currentDistance > 0)
                                {
                                    row["DistanceChangeFromLast"] = (float)(currentDistance - stats.Value.LastDistanceYards.Value);
                                    distanceBackfilled = true;
                                }
                            }
                            ApplyPrefetchedRatingFallbacks(row, history, identity.RaceDate.Value);
                        }
                        var resolvedWinRate = ResolveWinRateLast5(row, winRateCache);
                        if (resolvedWinRate.HasValue)
                        {
                            var winRateValue = resolvedWinRate.Value;
                            perRunnerWinRate[rowIndex] = winRateValue;
                            winRateSum += winRateValue;
                            winRateCount++;
                        }

                        var resolvedSpeed = ResolveAvgSpeedLast5(row, speedCache);
                        if (resolvedSpeed.HasValue)
                        {
                            row["AvgSpeedLast5"] = resolvedSpeed.Value;
                            row["RaceAvgSpeedLast5"] = resolvedSpeed.Value;
                        }

                        if (!distanceBackfilled)
                        {
                            ResolveDistanceChangeFromLast(row, distanceCache);
                        }
                        ApplyTrainerClassFallbacks(row);
                        ApplyJockeyClassFallbacks(row);
                    }

                    if (winRateCount > 0)
                    {
                        for (var rowIndex = 0; rowIndex < rowCount; rowIndex++)
                        {
                            var row = raceRows[rowIndex];
                            if (row is null)
                            {
                                continue;
                            }

                            float valueToAssign;
                            var runnerWinRate = perRunnerWinRate[rowIndex];
                            if (runnerWinRate.HasValue)
                            {
                                valueToAssign = winRateCount > 1
                                    ? (float)((winRateSum - runnerWinRate.Value) / (winRateCount - 1))
                                    : runnerWinRate.Value;
                            }
                            else
                            {
                                valueToAssign = (float)(winRateSum / winRateCount);
                            }

                            row["RaceAvgWinRateLast5"] = valueToAssign;
                        }
                    }
                    var current = Interlocked.Increment(ref processed);
                    ReportProgress(current);
                });

                stopwatch.Stop();
                Console.WriteLine($"[TrainAI] Repository backfills complete in {FormatDuration(stopwatch.Elapsed)}.");
                if (!includeIdentifiers)
                {
                    foreach (var race in races)
                    {
                        var raceRows = race?.Rows;
                        if (raceRows is null)
                        {
                            continue;
                        }

                        foreach (var row in raceRows)
                        {
                            if (row is null)
                            {
                                continue;
                            }

                            foreach (var key in BackfillRequiredKeys)
                            {
                                row.Remove(key);
                            }
                        }
                    }
                }
            }
            private static float? GetMeaningfulFloat(Dictionary<string, object?> row, string key, bool allowZero = false)
            {
                if (row is null)
                {
                    return null;
                }

                if (TryGetFloat(row, key, out var value))
                {
                    if (float.IsNaN(value) || float.IsInfinity(value))
                    {
                        return null;
                    }

                    if (allowZero || Math.Abs(value) > 1e-6f)
                    {
                        return value;
                    }
                }

                return null;
            }
            private static void ApplyTrainerClassFallbacks(Dictionary<string, object?> row)
            {
                if (row is null)
                {
                    return;
                }

                float? trainerClassWin = GetMeaningfulFloat(row, "TrainerClassWinRate");
                if (!trainerClassWin.HasValue)
                {
                    float? fallbackWin = GetMeaningfulFloat(row, "TrainerWinRate")
                        ?? GetMeaningfulFloat(row, $"TrainerWinRateLast{TrainerJockeyRecentStarts}")
                        ?? GetMeaningfulFloat(row, "TrainerWinRateRecentDays")
                        ?? GetMeaningfulFloat(row, "TrainerSurfaceWinRate")
                        ?? GetMeaningfulFloat(row, "TrainerGoingWinRate")
                        ?? GetMeaningfulFloat(row, "TrainerDistanceBucketWinRate")
                        ?? GetMeaningfulFloat(row, "TrainerCourseWinRate");

                    if (fallbackWin.HasValue)
                    {
                        trainerClassWin = ClampProbability(fallbackWin.Value);
                        row["TrainerClassWinRate"] = trainerClassWin.Value;
                    }
                }

                float? trainerClassAvg = GetMeaningfulFloat(row, "TrainerClassAvgNorm");
                if (!trainerClassAvg.HasValue)
                {
                    trainerClassAvg = GetMeaningfulFloat(row, "TrainerSurfaceAvgNorm")
                        ?? GetMeaningfulFloat(row, "TrainerGoingAvgNorm")
                        ?? GetMeaningfulFloat(row, "TrainerDistanceBucketAvgNorm");

                    if (!trainerClassAvg.HasValue)
                    {
                        var sourceWin = trainerClassWin ?? GetMeaningfulFloat(row, "TrainerWinRate");
                        if (sourceWin.HasValue)
                        {
                            trainerClassAvg = ClampNormalizedPosition(1f - ClampProbability(sourceWin.Value));
                        }
                    }

                    if (trainerClassAvg.HasValue)
                    {
                        row["TrainerClassAvgNorm"] = trainerClassAvg.Value;
                    }
                }

                float? lastTrainerClass = GetMeaningfulFloat(row, "LastTrainerClassNormPos");
                if (!lastTrainerClass.HasValue)
                {
                    lastTrainerClass = GetMeaningfulFloat(row, "LastTrainerSurfaceNormPos")
                        ?? GetMeaningfulFloat(row, "LastTrainerGoingNormPos")
                        ?? GetMeaningfulFloat(row, "LastTrainerDistanceBucketNormPos")
                        ?? trainerClassAvg;

                    if (!lastTrainerClass.HasValue && trainerClassWin.HasValue)
                    {
                        lastTrainerClass = ClampNormalizedPosition(1f - ClampProbability(trainerClassWin.Value));
                    }

                    if (lastTrainerClass.HasValue)
                    {
                        row["LastTrainerClassNormPos"] = lastTrainerClass.Value;
                    }
                }
            }

            private static void ApplyJockeyClassFallbacks(Dictionary<string, object?> row)
            {
                if (row is null)
                {
                    return;
                }

                float? jockeyClassWin = GetMeaningfulFloat(row, "JockeyClassWinRate");
                if (!jockeyClassWin.HasValue)
                {
                    float? fallbackWin = GetMeaningfulFloat(row, "JockeyWinRate")
                        ?? GetMeaningfulFloat(row, $"JockeyWinRateLast{TrainerJockeyRecentStarts}")
                        ?? GetMeaningfulFloat(row, "JockeyWinRateRecentDays")
                        ?? GetMeaningfulFloat(row, "TrainerClassWinRate")
                        ?? GetMeaningfulFloat(row, "TrainerWinRate")
                        ?? GetMeaningfulFloat(row, "JockeySurfaceWinRate")
                        ?? GetMeaningfulFloat(row, "JockeyGoingWinRate")
                        ?? GetMeaningfulFloat(row, "JockeyDistanceBucketWinRate")
                        ?? GetMeaningfulFloat(row, "TrainerJockeyWinRate");

                    if (fallbackWin.HasValue)
                    {
                        jockeyClassWin = ClampProbability(fallbackWin.Value);
                        row["JockeyClassWinRate"] = jockeyClassWin.Value;
                    }
                }

                float? jockeyClassAvg = GetMeaningfulFloat(row, "JockeyClassAvgNorm");
                if (!jockeyClassAvg.HasValue)
                {
                    jockeyClassAvg = GetMeaningfulFloat(row, "JockeySurfaceAvgNorm")
                        ?? GetMeaningfulFloat(row, "JockeyGoingAvgNorm")
                        ?? GetMeaningfulFloat(row, "JockeyDistanceBucketAvgNorm")
                        ?? GetMeaningfulFloat(row, "TrainerClassAvgNorm");

                    if (!jockeyClassAvg.HasValue)
                    {
                        var sourceWin = jockeyClassWin
                            ?? GetMeaningfulFloat(row, "JockeyWinRate")
                            ?? GetMeaningfulFloat(row, "TrainerClassWinRate");
                        if (sourceWin.HasValue)
                        {
                            jockeyClassAvg = ClampNormalizedPosition(1f - ClampProbability(sourceWin.Value));
                        }
                    }

                    if (jockeyClassAvg.HasValue)
                    {
                        row["JockeyClassAvgNorm"] = jockeyClassAvg.Value;
                    }
                }

                float? lastJockeyClass = GetMeaningfulFloat(row, "LastJockeyClassNormPos");
                if (!lastJockeyClass.HasValue)
                {
                    lastJockeyClass = GetMeaningfulFloat(row, "LastJockeySurfaceNormPos")
                        ?? GetMeaningfulFloat(row, "LastJockeyGoingNormPos")
                        ?? GetMeaningfulFloat(row, "LastJockeyDistanceBucketNormPos")
                        ?? jockeyClassAvg
                        ?? GetMeaningfulFloat(row, "LastTrainerClassNormPos");

                    if (!lastJockeyClass.HasValue && jockeyClassWin.HasValue)
                    {
                        lastJockeyClass = ClampNormalizedPosition(1f - ClampProbability(jockeyClassWin.Value));
                    }

                    if (lastJockeyClass.HasValue)
                    {
                        row["LastJockeyClassNormPos"] = lastJockeyClass.Value;
                    }
                }
            }
            private static IDictionary<int, RaceFeatureBackfill> BuildRaceMetadataLookup(
                IRacingRepository? repository,
                IReadOnlyList<PreparedRace> races)
            {
                if (repository is null || races is null || races.Count == 0)
                {
                    return new Dictionary<int, RaceFeatureBackfill>();
                }

                var missingRaceIds = new HashSet<int>();
                foreach (var race in races)
                {
                    if (race?.Rows == null)
                    {
                        continue;
                    }

                    foreach (var row in race.Rows)
                    {
                        if (row is null)
                        {
                            continue;
                        }

                        if (!PreparedDataset.TryGetRequiredInt32(row, "RaceId", out var raceId))
                        {
                            continue;
                        }

                        if (NeedsRaceMetadataBackfill(row))
                        {
                            missingRaceIds.Add(raceId);
                        }
                    }
                }

                if (missingRaceIds.Count == 0)
                {
                    return new Dictionary<int, RaceFeatureBackfill>();
                }

                return repository.GetRaceFeatureBackfills(missingRaceIds);
            }

            private static bool NeedsRaceMetadataBackfill(Dictionary<string, object?> row)
            {
                return !HasMeaningfulString(row, "RaceType") ||
                       !HasMeaningfulString(row, "Surface") ||
                       !HasMeaningfulString(row, "Going") ||
                       !HasMeaningfulString(row, "DistanceText") ||
                       !HasMeaningfulNumeric(row, "DistanceYards") ||
                       !HasMeaningfulNumeric(row, "RunnerCount");
            }

            private static void ApplyRaceMetadataBackfill(Dictionary<string, object?> row, RaceFeatureBackfill metadata)
            {
                if (row is null || metadata is null)
                {
                    return;
                }

                if (!HasMeaningfulString(row, "RaceType") && !string.IsNullOrWhiteSpace(metadata.RaceType))
                {
                    row["RaceType"] = metadata.RaceType?.Trim();
                }

                if (!HasMeaningfulString(row, "Surface") && !string.IsNullOrWhiteSpace(metadata.Surface))
                {
                    row["Surface"] = metadata.Surface?.Trim();
                }

                if (!HasMeaningfulString(row, "Going") && !string.IsNullOrWhiteSpace(metadata.Going))
                {
                    row["Going"] = metadata.Going?.Trim();
                }

                if (!HasMeaningfulNumeric(row, "DistanceYards") && metadata.DistanceYards.HasValue)
                {
                    row["DistanceYards"] = metadata.DistanceYards.Value;
                }

                if (!HasMeaningfulString(row, "DistanceText") && !string.IsNullOrWhiteSpace(metadata.DistanceText))
                {
                    row["DistanceText"] = metadata.DistanceText?.Trim();
                }

                if (!HasMeaningfulNumeric(row, "RunnerCount") && metadata.RunnerCount.HasValue)
                {
                    row["RunnerCount"] = metadata.RunnerCount.Value;
                }
            }
            private static readonly HashSet<string> NonMeaningfulStringTokens = new(StringComparer.OrdinalIgnoreCase)
        {
            "Unknown",
            "Missing",
            "N/A",
            "N\\A",
            "NA",
            "-"
        };

            private static bool IsMeaningfulStringValue(string? value)
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    return false;
                }

                var trimmed = value.Trim();
                return trimmed.Length > 0 && !NonMeaningfulStringTokens.Contains(trimmed);
            }

            private static bool HasMeaningfulString(Dictionary<string, object?> row, string key)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    return false;
                }

                if (value is string s)
                {
                    return IsMeaningfulStringValue(s);
                }

                if (value is System.Data.SqlTypes.SqlString sql)
                {
                    return !sql.IsNull && IsMeaningfulStringValue(sql.Value);
                }

                return false;
            }
            private static bool HasMeaningfulNumeric(Dictionary<string, object?> row, string key)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    return false;
                }

                if (PreparedDataset.TryConvertToInt32(value, out var intValue))
                {
                    return intValue != 0;
                }

                if (value is IConvertible convertible)
                {
                    try
                    {
                        return Math.Abs(convertible.ToDouble(null)) > 0.0;
                    }
                    catch
                    {
                        return false;
                    }
                }

                return false;
            }
            private static string FormatDuration(TimeSpan time)
            {
                if (time <= TimeSpan.Zero)
                {
                    return "0s";
                }

                var builder = new StringBuilder();

                if (time.Days > 0)
                {
                    builder.Append(time.Days).Append('d').Append(' ');
                }

                if (time.Hours > 0 || builder.Length > 0)
                {
                    builder.Append(time.Hours).Append('h').Append(' ');
                }

                if (time.Minutes > 0 || builder.Length > 0)
                {
                    builder.Append(time.Minutes).Append('m').Append(' ');
                }

                if (time.Seconds > 0)
                {
                    builder.Append(time.Seconds).Append('s');
                }
                else if (builder.Length == 0)
                {
                    builder.Append(time.TotalSeconds < 1 ? "<1s" : "0s");
                }

                return builder.ToString().Trim();
            }
            private HorseIdentitySummary CollectHorseIdentitySummary(List<PreparedRace> races)
            {
                var identities = new List<HorseIdentity>();
                var horseIds = new HashSet<int>();
                var seen = new HashSet<HorseCacheKey>(HorseCacheKeyComparer.Instance);
                DateTime? maxRaceDate = null;

                if (races is null || races.Count == 0)
                {
                    return new HorseIdentitySummary(identities, horseIds, maxRaceDate);
                }

                foreach (var race in races)
                {
                    if (race?.Rows == null || race.Rows.Count == 0)
                    {
                        continue;
                    }

                    foreach (var row in race.Rows)
                    {
                        if (row is null)
                        {
                            continue;
                        }

                        if (!TryGetHorseIdentity(row, out var identity))
                        {
                            continue;
                        }

                        if (identity.HorseId.HasValue)
                        {
                            horseIds.Add(identity.HorseId.Value);
                        }

                        if (identity.RaceDate.HasValue)
                        {
                            var date = identity.RaceDate.Value.Date;
                            if (!maxRaceDate.HasValue || date > maxRaceDate.Value)
                            {
                                maxRaceDate = date;
                            }
                        }

                        var key = identity.ToCacheKey();
                        if (seen.Add(key))
                        {
                            identities.Add(identity);
                        }
                    }
                }

                return new HorseIdentitySummary(identities, horseIds, maxRaceDate);
            }

            private void PreloadRepositoryCaches(
                IRacingRepository repository,
                IReadOnlyList<HorseIdentity> identities,
                Dictionary<int, List<PrefetchedHorseRace>> prefetchedHistory,
                ConcurrentDictionary<HorseCacheKey, (float WinRate, int Wins, int Starts)?> winRateCache,
                ConcurrentDictionary<HorseCacheKey, float?> speedCache,
                ConcurrentDictionary<HorseCacheKey, int?> distanceCache)
            {
                if (identities is null || identities.Count == 0)
                {
                    return;
                }

                var horsesToPreload = new List<HorseIdentity>();
                foreach (var identity in identities)
                {
                    if (!identity.HorseId.HasValue && string.IsNullOrWhiteSpace(identity.NormalizedName))
                    {
                        continue;
                    }

                    if (identity.HorseId.HasValue && prefetchedHistory.ContainsKey(identity.HorseId.Value))
                    {
                        continue;
                    }

                    horsesToPreload.Add(identity);
                }

                if (horsesToPreload.Count == 0)
                {
                    return;
                }

                var uniqueRequests = new List<(HorseCacheKey CacheKey, HorseMetricRequest Request)>(horsesToPreload.Count);
                var seen = new HashSet<HorseCacheKey>(HorseCacheKeyComparer.Instance);

                foreach (var identity in horsesToPreload)
                {
                    var cacheKey = identity.ToCacheKey();
                    if (!seen.Add(cacheKey))
                    {
                        continue;
                    }

                    var request = new HorseMetricRequest(
                        identity.HorseId,
                        identity.NormalizedName,
                        identity.RaceDate,
                        identity.RawName);
                    uniqueRequests.Add((cacheKey, request));
                }

                if (uniqueRequests.Count == 0)
                {
                    return;
                }

                const int windowSize = 5;
                var requestList = uniqueRequests.Select(entry => entry.Request).ToList();

                var batchedWinRates = repository.GetRecentHorseWinStatsBatch(requestList, windowSize);
                var batchedSpeeds = repository.GetRecentHorseAverageSpeedsBatch(requestList, windowSize);
                var batchedDistances = repository.GetLastRaceDistancesBatch(requestList);

                foreach (var (cacheKey, request) in uniqueRequests)
                {
                    if (batchedWinRates.TryGetValue(request, out var winStats) && winStats.HasValue && winStats.Value.Starts > 0)
                    {
                        var winRate = ClampProbability(_trainer.ComputeSmoothedWinRate(winStats.Value.Wins, winStats.Value.Starts));
                        winRateCache[cacheKey] = (winRate, winStats.Value.Wins, winStats.Value.Starts);
                    }
                    else
                    {
                        winRateCache[cacheKey] = null;
                    }

                    if (batchedSpeeds.TryGetValue(request, out var speedValue) &&
                        speedValue.HasValue &&
                        speedValue.Value > 0f &&
                        !float.IsNaN(speedValue.Value) &&
                        !float.IsInfinity(speedValue.Value))
                    {
                        speedCache[cacheKey] = speedValue.Value;
                    }
                    else
                    {
                        speedCache[cacheKey] = null;
                    }

                    if (batchedDistances.TryGetValue(request, out var distanceValue))
                    {
                        distanceCache[cacheKey] = distanceValue;
                    }
                    else
                    {
                        distanceCache[cacheKey] = null;
                    }
                }
            }

            private void EnsureHorsePerformanceIndexes()
            {
                if (_horsePerformanceIndexesEnsured)
                {
                    return;
                }

                lock (HorseIndexLock)
                {
                    if (_horsePerformanceIndexesEnsured)
                    {
                        return;
                    }

                    using var connection = new SqlConnection(_trainer._connectionString);
                    connection.Open();
                    const string sql = @"IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunnerResult_HorseId_RaceId' AND object_id = OBJECT_ID(N'dbo.RunnerResult'))
BEGIN
    CREATE INDEX IX_RunnerResult_HorseId_RaceId ON dbo.RunnerResult(HorseId, RaceId)
        INCLUDE (RunnerResultId, FinishPos, DistanceBeatenLengths);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Race_RaceDate' AND object_id = OBJECT_ID(N'dbo.Race'))
BEGIN
    CREATE INDEX IX_Race_RaceDate ON dbo.Race(RaceDate, RaceId)
        INCLUDE (DistanceYards, WinningTimeMs);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Horse_Name' AND object_id = OBJECT_ID(N'dbo.Horse'))
BEGIN
    CREATE INDEX IX_Horse_Name ON dbo.Horse(Name, HorseId);
END;";

                    connection.Execute(sql);
                    _horsePerformanceIndexesEnsured = true;
                }
            }

            private (float WinRate, int Wins, int Starts)? LoadWinRateLast5(HorseIdentity identity, IRacingRepository repository)
            {
                try
                {
                    var stats = repository.GetRecentHorseWinStats(identity.RawName, identity.HorseId, identity.RaceDate, 5);
                    if (stats.HasValue && stats.Value.Starts > 0)
                    {
                        var winRate = ClampProbability(_trainer.ComputeSmoothedWinRate(stats.Value.Wins, stats.Value.Starts));
                        return (winRate, stats.Value.Wins, stats.Value.Starts);
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[AI] Failed to resolve WinRateLast5 for {identity.Display}: {ex.Message}");
                }

                return null;
            }

            private float? LoadAvgSpeedLast5(HorseIdentity identity, IRacingRepository repository)
            {
                try
                {
                    const int windowSize = 5;
                    var entries = repository.GetRecentHorseSpeedEntries(identity.RawName, identity.HorseId, identity.RaceDate, windowSize);
                    if (entries == null || entries.Count == 0)
                    {
                        return null;
                    }

                    var speeds = new List<float>(entries.Count);
                    foreach (var entry in entries)
                    {
                        if (!entry.DistanceYards.HasValue || entry.DistanceYards.Value <= 0)
                        {
                            continue;
                        }

                        if (!entry.WinningTimeMilliseconds.HasValue || entry.WinningTimeMilliseconds.Value <= 0)
                        {
                            continue;
                        }

                        var runnerTimeMs = (float)entry.WinningTimeMilliseconds.Value;
                        if (entry.DistanceBeatenLengths.HasValue)
                        {
                            runnerTimeMs += (float)entry.DistanceBeatenLengths.Value * MsPerLength;
                        }

                        if (runnerTimeMs <= 0f)
                        {
                            continue;
                        }

                        var speed = entry.DistanceYards.Value / runnerTimeMs;
                        if (!float.IsNaN(speed) && !float.IsInfinity(speed) && speed > 0f)
                        {
                            speeds.Add(speed);
                        }
                    }

                    if (speeds.Count == 0)
                    {
                        return null;
                    }

                    var average = speeds.Average();
                    if (float.IsNaN(average) || float.IsInfinity(average) || average <= 0f)
                    {
                        return null;
                    }

                    return average;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[AI] Failed to resolve AvgSpeedLast5 for {identity.Display}: {ex.Message}");
                    return null;
                }
            }

            private int? LoadLastRaceDistance(HorseIdentity identity, IRacingRepository repository)
            {
                try
                {
                    return repository.GetLastRaceDistance(identity.RawName, identity.HorseId, identity.RaceDate);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[AI] Failed to resolve DistanceChangeFromLast for {identity.Display}: {ex.Message}");
                    return null;
                }
            }

            private Dictionary<int, List<PrefetchedHorseRace>> PrefetchHorseHistoryByHorseId(HorseIdentitySummary identitySummary)
            {
                var fetchedResults = new ConcurrentDictionary<int, List<PrefetchedHorseRace>>();
                var horseIds = identitySummary.HorseIds;
                if (horseIds.Count == 0)
                {
                    return new Dictionary<int, List<PrefetchedHorseRace>>();
                }

                const string sql = @"SELECT
    rr.HorseId,
    r.RaceDate,
    rr.FinishPos,
    r.DistanceYards,
    r.WinningTimeMs AS WinningTimeMilliseconds,
    rr.DistanceBeatenLengths,
     rr.RunnerResultId,
    rr.OfficialRating,
    r.Class
FROM RunnerResult rr
JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId IN @HorseIds
  AND (@MaxRaceDateExclusive IS NULL OR r.RaceDate < @MaxRaceDateExclusive)
ORDER BY rr.HorseId, r.RaceDate, rr.RunnerResultId;";

                var cachedResults = new Dictionary<int, List<PrefetchedHorseRace>>();
                foreach (var horseId in horseIds)
                {
                    if (_trainer._prefetchedHistoryCache.TryGetValue(horseId, out var cachedHistory))
                    {
                        cachedResults[horseId] = new List<PrefetchedHorseRace>(cachedHistory);
                    }
                }

                var horseIdArray = horseIds.Where(id => !cachedResults.ContainsKey(id)).ToArray();
                const int batchSize = 500;
                const int commandTimeoutSeconds = 3600;
                DateTime? maxRaceDateExclusive = identitySummary.MaxRaceDate?.Date.AddDays(1);
                if (horseIdArray.Length > 0)
                {
                    EnsureHorsePerformanceIndexes();
                    var parallelOptions = new ParallelOptions
                    {
                        MaxDegreeOfParallelism = Math.Max(1, Math.Min(Math.Min(Environment.ProcessorCount, horseIdArray.Length), 4))
                    };

                    using var connectionFactory = new ThreadLocal<SqlConnection>(() =>
                    {
                        var connection = new SqlConnection(_trainer._connectionString);
                        connection.Open();
                        return connection;
                    }, trackAllValues: true);

                    try
                    {
                        Parallel.ForEach(Partitioner.Create(0, horseIdArray.Length, batchSize), parallelOptions, range =>
                        {
                            var count = range.Item2 - range.Item1;
                            if (count <= 0)
                            {
                                return;
                            }

                            var batch = new int[count];
                            Array.Copy(horseIdArray, range.Item1, batch, 0, count);
                            if (batch.Length == 0)
                            {
                                return;
                            }

                            var conn = connectionFactory.Value;

                            var rows = conn.Query<PrefetchedHorseRaceRow>(sql, new
                            {
                                HorseIds = batch,
                                MaxRaceDateExclusive = maxRaceDateExclusive
                            }, commandTimeout: commandTimeoutSeconds).ToList();

                            foreach (var row in rows)
                            {
                                var list = fetchedResults.GetOrAdd(row.HorseId, _ => new List<PrefetchedHorseRace>());
                                lock (list)
                                {
                                    list.Add(new PrefetchedHorseRace(
                                        row.RaceDate.Date,
                                        row.FinishPos,
                                        row.DistanceYards,
                                        row.WinningTimeMilliseconds,
                                        row.DistanceBeatenLengths,
                                         row.RunnerResultId,
                                        row.OfficialRating,
                                        row.Class));
                                }
                            }
                        });
                    }
                    finally
                    {
                        foreach (var connection in connectionFactory.Values)
                        {
                            connection.Dispose();
                        }
                    }
                }

                var finalized = new Dictionary<int, List<PrefetchedHorseRace>>(cachedResults.Count + fetchedResults.Count);
                foreach (var kvp in cachedResults)
                {
                    finalized[kvp.Key] = kvp.Value;
                }

                foreach (var kvp in fetchedResults)
                {
                    var list = kvp.Value;
                    list.Sort((left, right) =>
                    {
                        int compare = left.RaceDate.CompareTo(right.RaceDate);
                        if (compare != 0)
                        {
                            return compare;
                        }

                        return left.RunnerResultId.CompareTo(right.RunnerResultId);
                    });

                    _trainer._prefetchedHistoryCache[kvp.Key] = list.ToArray();
                    finalized[kvp.Key] = list;
                }

                return finalized;
            }
            private PrefetchedHorseStats? ComputePrefetchedHorseStats(List<PrefetchedHorseRace> history, DateTime raceDate)
            {
                if (history is null || history.Count == 0)
                {
                    return null;
                }

                var targetDate = raceDate.Date;
                int lastIndex = FindLastIndexBefore(history, targetDate);
                if (lastIndex < 0)
                {
                    return null;
                }

                int firstIndex = Math.Max(0, lastIndex - 4);
                int starts = lastIndex - firstIndex + 1;

                int wins = 0;
                float speedSum = 0f;
                int speedCount = 0;

                for (int i = firstIndex; i <= lastIndex; i++)
                {
                    var entry = history[i];
                    if (entry.FinishPos.HasValue && entry.FinishPos.Value == 1)
                    {
                        wins++;
                    }

                    if (entry.DistanceYards.HasValue && entry.DistanceYards.Value > 0 &&
                        entry.WinningTimeMilliseconds.HasValue && entry.WinningTimeMilliseconds.Value > 0)
                    {
                        var runnerTime = (float)entry.WinningTimeMilliseconds.Value;
                        if (entry.DistanceBeatenLengths.HasValue)
                        {
                            runnerTime += (float)entry.DistanceBeatenLengths.Value * MsPerLength;
                        }

                        if (runnerTime > 0f)
                        {
                            var speed = entry.DistanceYards.Value / runnerTime;
                            if (!float.IsNaN(speed) && !float.IsInfinity(speed) && speed > 0f)
                            {
                                speedSum += speed;
                                speedCount++;
                            }
                        }
                    }
                }

                float? winRate = starts > 0 ? _trainer.ComputeSmoothedWinRate(wins, starts) : (float?)null;
                float? averageSpeed = speedCount > 0 ? speedSum / speedCount : (float?)null;
                int? lastDistance = history[lastIndex].DistanceYards;

                return new PrefetchedHorseStats(winRate, averageSpeed, lastDistance);
            }

            private static int FindLastIndexBefore(List<PrefetchedHorseRace> history, DateTime targetDate)
            {
                int low = 0;
                int high = history.Count - 1;
                int result = -1;

                while (low <= high)
                {
                    int mid = low + ((high - low) / 2);
                    var midDate = history[mid].RaceDate.Date;
                    if (midDate < targetDate)
                    {
                        result = mid;
                        low = mid + 1;
                    }
                    else
                    {
                        high = mid - 1;
                    }
                }

                return result;
            }

            private sealed class PrefetchedHorseRaceRow
            {
                public int HorseId { get; init; }
                public DateTime RaceDate { get; init; }
                public short? FinishPos { get; init; }
                public int? DistanceYards { get; init; }
                public int? WinningTimeMilliseconds { get; init; }
                public decimal? DistanceBeatenLengths { get; init; }
                public int RunnerResultId { get; init; }
                public short? OfficialRating { get; init; }
                public byte? Class { get; init; }
            }
            private readonly record struct PrefetchedHorseStats(
                float? WinRate,
                float? AverageSpeed,
                int? LastDistanceYards);

            private float? ResolveWinRateLast5(Dictionary<string, object?> row, ConcurrentDictionary<HorseCacheKey, (float WinRate, int Wins, int Starts)?> cache)
            {
                if (row is null)
                {
                    return null;
                }

                if (TryGetFloat(row, "WinRateLast5", out var existingWinRate) && existingWinRate >= 0f)
                {
                    return existingWinRate;
                }

                if (TryGetFloat(row, "LifetimeWinRate", out var lifetimeWinRate))
                {
                    var sanitized = ClampProbability(lifetimeWinRate);
                    row["WinRateLast5"] = sanitized;
                    return sanitized;
                }

                if (!TryGetHorseIdentity(row, out var identity))
                {
                    return null;
                }

                var cacheKey = identity.ToCacheKey();
                var repository = _trainer.EnsureRacingRepository();
                if (repository is null)
                {
                    cache[cacheKey] = null;
                    return null;
                }
                if (!cache.TryGetValue(cacheKey, out var cached))
                {
                    cached = LoadWinRateLast5(identity, repository);
                    cache[cacheKey] = cached;
                }

                if (cached.HasValue)
                {
                    row["WinRateLast5"] = cached.Value.WinRate;
                    return cached.Value.WinRate;
                }

                return null;
            }

            private float? ResolveAvgSpeedLast5(Dictionary<string, object?> row, ConcurrentDictionary<HorseCacheKey, float?> cache)
            {
                if (row is null)
                {
                    return null;
                }

                if (TryGetFloat(row, "AvgSpeedLast5", out var existingSpeed) && existingSpeed > 0f &&
                    !float.IsNaN(existingSpeed) && !float.IsInfinity(existingSpeed))
                {
                    return existingSpeed;
                }

                if (!TryGetHorseIdentity(row, out var identity))
                {
                    return null;
                }

                var cacheKey = identity.ToCacheKey();
                var repository = _trainer.EnsureRacingRepository();
                if (repository is null)
                {
                    cache[cacheKey] = null;
                    return null;
                }
                if (!cache.TryGetValue(cacheKey, out var cached))
                {
                    cached = LoadAvgSpeedLast5(identity, repository);
                    cache[cacheKey] = cached;
                }
                if (cached.HasValue && cached.Value > 0f &&
                !float.IsNaN(cached.Value) && !float.IsInfinity(cached.Value))
                {
                    return cached.Value;
                }

                return null;
            }

            private void ResolveDistanceChangeFromLast(Dictionary<string, object?> row, ConcurrentDictionary<HorseCacheKey, int?> cache)
            {
                if (row is null)
                {
                    return;
                }

                if (!PreparedDataset.TryGetValueWithAliases(row, "DistanceYards", out var distanceObj, requireNonNull: true) ||
                    !PreparedDataset.TryConvertToInt32(distanceObj!, out var currentDistance) || currentDistance <= 0)
                {
                    return;
                }

                if (!TryGetHorseIdentity(row, out var identity))
                {
                    return;
                }

                var cacheKey = identity.ToCacheKey();
                var repository = _trainer.EnsureRacingRepository();
                if (repository is null)
                {
                    cache[cacheKey] = null;
                    return;
                }
                if (!cache.TryGetValue(cacheKey, out var cached))
                {
                    cached = LoadLastRaceDistance(identity, repository);
                    cache[cacheKey] = cached;
                }

                if (cached.HasValue)
                {
                    row["DistanceChangeFromLast"] = (float)(currentDistance - cached.Value);
                }
            }

            private static bool TryGetHorseIdentity(Dictionary<string, object?> row, out HorseIdentity identity)
            {
                identity = default;
                if (row is null)
                {
                    return false;
                }

                int? horseId = null;
                if (row.TryGetValue("HorseId", out var horseObj) && horseObj != null && PreparedDataset.TryConvertToInt32(horseObj, out var horseIdValue))
                {
                    if (horseIdValue > 0)
                    {
                        horseId = horseIdValue;
                    }
                }

                string? rawName = null;
                if (row.TryGetValue("HorseName", out var horseNameObj) && horseNameObj != null)
                {
                    rawName = horseNameObj as string ?? horseNameObj.ToString();
                }

                string? normalized = null;
                if (!string.IsNullOrWhiteSpace(rawName))
                {
                    normalized = NormalizeHorseNameForLookup(rawName);
                }

                DateTime? raceDate = null;
                if (row.TryGetValue("RaceDate", out var raceDateObj) && raceDateObj != null)
                {
                    if (raceDateObj is DateTime dt)
                    {
                        raceDate = dt.Date;
                    }
                    else if (raceDateObj is DateTimeOffset dto)
                    {
                        raceDate = dto.Date;
                    }
                }

                if (!horseId.HasValue && string.IsNullOrWhiteSpace(normalized))
                {
                    return false;
                }

                identity = new HorseIdentity(horseId, rawName, normalized, raceDate);
                return true;
            }
            private void ApplyPrefetchedRatingFallbacks(
                Dictionary<string, object?> row,
                List<PrefetchedHorseRace> history,
                DateTime raceDate)
            {
                if (row is null || history is null || history.Count == 0)
                {
                    return;
                }

                if (PerformanceWindows.Length == 0)
                {
                    return;
                }

                var cutoffDate = raceDate.Date;
                int maxWindow = PerformanceWindows[^1];
                if (maxWindow <= 0)
                {
                    return;
                }

                var ratings = new List<float>(Math.Min(history.Count, maxWindow));
                for (int i = history.Count - 1; i >= 0 && ratings.Count < maxWindow; i--)
                {
                    var entry = history[i];
                    if (entry.RaceDate.Date >= cutoffDate)
                    {
                        continue;
                    }

                    float? rating = null;
                    if (entry.OfficialRating.HasValue && entry.OfficialRating.Value > 0)
                    {
                        rating = entry.OfficialRating.Value;
                    }
                    else
                    {
                        rating = ResolveClassRatingBaseline(entry.Class);
                    }

                    if (!rating.HasValue && ratings.Count > 0)
                    {
                        rating = ratings[^1];
                    }

                    if (rating.HasValue && rating.Value > 0f && !float.IsNaN(rating.Value))
                    {
                        ratings.Add(rating.Value);
                    }
                }

                if (ratings.Count == 0)
                {
                    return;
                }

                foreach (var window in PerformanceWindows)
                {
                    var key = $"AvgRatingLast{window}";
                    if (TryGetFloat(row, key, out var existing) && existing > 0f)
                    {
                        continue;
                    }

                    int count = Math.Min(window, ratings.Count);
                    if (count <= 0)
                    {
                        continue;
                    }

                    float sum = 0f;
                    for (int i = 0; i < count; i++)
                    {
                        sum += ratings[i];
                    }

                    row[key] = sum / count;
                }
            }
            private static bool TryGetFloat(Dictionary<string, object?> row, string key, out float value)
            {
                value = 0f;
                if (row is null || string.IsNullOrWhiteSpace(key))
                {
                    return false;
                }

                if (!row.TryGetValue(key, out var obj) || obj is null)
                {
                    return false;
                }

                try
                {
                    value = Convert.ToSingle(obj);
                    if (float.IsNaN(value) || float.IsInfinity(value))
                    {
                        return false;
                    }

                    return true;
                }
                catch
                {
                    return false;
                }
            }

            private static float ClampProbability(float value)
            {
                if (value < 0f)
                {
                    return 0f;
                }

                if (value > 1f)
                {
                    return 1f;
                }

                return value;
            }
            private readonly struct HorseIdentitySummary
            {
                public HorseIdentitySummary(
                    List<HorseIdentity> identities,
                    HashSet<int> horseIds,
                    DateTime? maxRaceDate)
                {
                    Identities = identities;
                    HorseIds = horseIds;
                    MaxRaceDate = maxRaceDate;
                }

                public List<HorseIdentity> Identities { get; }

                public HashSet<int> HorseIds { get; }

                public DateTime? MaxRaceDate { get; }
            }
            private static float? ResolveClassRatingBaseline(int? classValue)
            {
                if (!classValue.HasValue)
                {
                    return null;
                }

                var classInt = classValue.Value;
                if (classInt <= 0)
                {
                    return null;
                }

                if (ClassRatingBaselines.TryGetValue(classInt, out var baseline))
                {
                    return baseline;
                }

                classInt = Math.Min(classInt, 12);
                return 110f - (classInt * 5f);
            }

            private static float? ResolveClassRatingBaseline(byte? classValue)
                => ResolveClassRatingBaseline(classValue.HasValue ? (int?)classValue.Value : null);
            private readonly struct HorseIdentity
            {
                public HorseIdentity(int? horseId, string? rawName, string? normalizedName, DateTime? raceDate)
                {
                    HorseId = horseId;
                    RawName = rawName;
                    NormalizedName = normalizedName;
                    RaceDate = raceDate?.Date;
                }

                public int? HorseId { get; }
                public string? RawName { get; }
                public string? NormalizedName { get; }
                public DateTime? RaceDate { get; }
                public string Display => !string.IsNullOrWhiteSpace(RawName)
                    ? RawName!
                    : (HorseId.HasValue ? $"horse:{HorseId.Value}" : "unknown horse");
                public HorseCacheKey ToCacheKey() => new(HorseId, NormalizedName, RaceDate);
            }

            private readonly struct HorseCacheKey : IEquatable<HorseCacheKey>
            {
                public HorseCacheKey(int? horseId, string? horseNameKey, DateTime? raceDate)
                {
                    HorseId = horseId;
                    HorseNameKey = horseNameKey;
                    RaceDate = raceDate?.Date;
                }

                public int? HorseId { get; }
                public string? HorseNameKey { get; }
                public DateTime? RaceDate { get; }

                public bool Equals(HorseCacheKey other)
                {
                    return HorseId == other.HorseId &&
                           string.Equals(HorseNameKey, other.HorseNameKey, StringComparison.OrdinalIgnoreCase) &&
                           Nullable.Equals(RaceDate, other.RaceDate);
                }

                public override bool Equals(object? obj) => obj is HorseCacheKey other && Equals(other);

                public override int GetHashCode()
                {
                    var hash = new HashCode();
                    hash.Add(HorseId.GetValueOrDefault());
                    hash.Add(HorseId.HasValue);
                    hash.Add(HorseNameKey ?? string.Empty, StringComparer.OrdinalIgnoreCase);
                    hash.Add(RaceDate?.Date ?? default);
                    return hash.ToHashCode();
                }
            }

            private sealed class HorseCacheKeyComparer : IEqualityComparer<HorseCacheKey>
            {
                public static HorseCacheKeyComparer Instance { get; } = new();

                public bool Equals(HorseCacheKey x, HorseCacheKey y) => x.Equals(y);

                public int GetHashCode(HorseCacheKey obj) => obj.GetHashCode();
            }

            public PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
            {
                if (upcoming is null)
                    throw new ArgumentNullException(nameof(upcoming));
                if (flows is null)
                    throw new ArgumentNullException(nameof(flows));
                if (flows.Count == 0)
                {
                    return null;
                }
                var results = PrepareUpcomingRaces(new[] { (upcoming, flows) });
                return results.Count > 0 ? results[0] : null;
            }

            public IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                    throw new ArgumentNullException(nameof(requests));

                if (requests.Count == 0)
                {
                    return Array.Empty<PreparedRace?>();
                }

                var results = new PreparedRace?[requests.Count];
                var valid = new List<(int Index, UpcomingRace Upcoming, IReadOnlyList<RunnerFlow> Flows)>(requests.Count);
                for (int i = 0; i < requests.Count; i++)
                {
                    var (upcoming, flows) = requests[i];
                    if (upcoming == null)
                    {
                        results[i] = null;
                        continue;
                    }

                    if (flows == null || flows.Count == 0)
                    {
                        results[i] = null;
                        continue;
                    }

                    valid.Add((i, upcoming, flows));
                }

                if (valid.Count == 0)
                {
                    return results;
                }

                using var conn = new SqlConnection(_trainer._connectionString);
                conn.Open();
                var (sql, runnerColumns, featureState) = BuildUpcomingPreparationContext(conn);

                var sorted = valid
                    .OrderBy(v => v.Upcoming.RaceDate.Date)
                    .ToList();

                var maxTargetDate = sorted[^1].Upcoming.RaceDate.Date;

                var historicalRecords = conn.Query(sql, new { TargetDate = maxTargetDate }, commandTimeout: 6000, buffered: false);
                var historicalRaces = MaterializeHistoricalRaces(conn, historicalRecords);

                int historyIndex = 0;
                foreach (var entry in sorted)
                {
                    var targetDate = entry.Upcoming.RaceDate.Date;
                    while (historyIndex < historicalRaces.Count && historicalRaces[historyIndex].RaceDate < targetDate)
                    {
                        featureState.ProcessRace(historicalRaces[historyIndex].Rows, includeRace: false, updateState: true);
                        historyIndex++;
                    }

                    var prepared = PrepareUpcomingRaceFromState(conn, entry.Upcoming, entry.Flows, runnerColumns, featureState);
                    results[entry.Index] = prepared;
                }

                return results;
            }

            private (string Sql, HashSet<string> RunnerColumns, FeatureEngineeringState FeatureState) BuildUpcomingPreparationContext(SqlConnection conn)
            {
                var raceColumns = PreparedDataset.LoadColumnNames(conn, "Race");
                var runnerColumns = PreparedDataset.LoadColumnNames(conn, "RunnerResult");
                string scheduledOffColumn = raceColumns.Contains("ScheduledOff")
                    ? "r.ScheduledOff AS ScheduledOff"
                    : "CAST(NULL AS time(0)) AS ScheduledOff";
                string actualOffColumn = raceColumns.Contains("ActualOff")
                    ? "r.ActualOff AS ActualOff"
                    : "CAST(NULL AS time(0)) AS ActualOff";
                string purseColumn = raceColumns.Contains("Purse")
                    ? "r.Purse AS Purse"
                    : "CAST(NULL AS decimal(18, 2)) AS Purse";
                string titleColumn = SelectColumn(raceColumns, "r", "Title", "nvarchar(512)");
                string raceTypeColumn = SelectColumn(raceColumns, "r", "RaceType", "nvarchar(128)");
                string classColumn = SelectColumn(raceColumns, "r", "Class", "int");
                string surfaceColumn = SelectColumn(raceColumns, "r", "Surface", "nvarchar(64)");
                string goingColumn = SelectColumn(raceColumns, "r", "Going", "nvarchar(30)");
                string distanceYardsColumn = SelectColumn(raceColumns, "r", "DistanceYards", "int");
                string distanceTextColumn = SelectColumn(raceColumns, "r", "DistanceText", "nvarchar(64)");
                string runnerCountColumn = SelectColumn(raceColumns, "r", "RunnerCount", "int");
                string statusColumn = SelectColumn(raceColumns, "r", "Status", "nvarchar(32)");
                string winningTimeColumn = SelectColumn(raceColumns, "r", "WinningTimeMs", "int");
                string saddleclothColumn = SelectColumn(runnerColumns, "rr", "SaddleclothNumber", "int");
                string drawColumn = SelectColumn(runnerColumns, "rr", "Draw", "int");
                string ageColumn = SelectColumn(runnerColumns, "rr", "Age", "int");
                string weightLbsColumn = SelectColumn(runnerColumns, "rr", "WeightLbs", "int");
                string weightTextColumn = SelectColumn(runnerColumns, "rr", "WeightText", "nvarchar(50)");
                string officialRatingColumn = SelectColumn(runnerColumns, "rr", "OfficialRating", "int", "OfficialRating");
                string outcomeCodeColumn = SelectColumn(runnerColumns, "rr", "OutcomeCode", "nvarchar(50)");
                string distanceBeatenTextColumn = SelectColumn(runnerColumns, "rr", "DistanceBeatenText", "nvarchar(50)");
                string distanceBeatenLengthsColumn = SelectColumn(runnerColumns, "rr", "DistanceBeatenLengths", "decimal(9,4)");
                string spFractionColumn = SelectColumn(runnerColumns, "rr", "SP_Fraction", "nvarchar(50)");
                string spDecimalColumn = SelectColumn(runnerColumns, "rr", "SP_Decimal", "decimal(18,6)");
                string favTagColumn = SelectColumn(runnerColumns, "rr", "FavTag", "nvarchar(16)");
                string openingFractionColumn = SelectColumn(runnerColumns, "rr", "OpeningFraction", "nvarchar(50)");
                string touchedHighColumn = SelectColumn(runnerColumns, "rr", "TouchedHighFraction", "nvarchar(50)");
                string touchedLowColumn = SelectColumn(runnerColumns, "rr", "TouchedLowFraction", "nvarchar(50)");
                var sql = $@"SELECT c.Name AS CourseName,
                                   h.Name AS HorseName,
                                   j.Name AS JockeyName,
                                   t.Name AS TrainerName,
                                   r.RaceId,
                                   r.CourseId,
                                   r.RaceDate,
                                   {scheduledOffColumn},
                                   {actualOffColumn},
                                   {titleColumn},
                                   {raceTypeColumn},
                                   {classColumn},
                                   {surfaceColumn},
                                   {goingColumn},
                                   {distanceYardsColumn},
                                   {distanceTextColumn},
                                   {runnerCountColumn},
                                   {statusColumn},
                                   {winningTimeColumn},
                                   {purseColumn},
                                   rr.HorseId,
                                   rr.TrainerId,
                                   rr.JockeyId,
                                   {saddleclothColumn},
                                   {drawColumn},
                                   {ageColumn},
                                   {weightLbsColumn},
                                   {weightTextColumn},
                                   {officialRatingColumn},
                                   rr.FinishPos,
                                   {outcomeCodeColumn},
                                   {distanceBeatenTextColumn},
                                   {distanceBeatenLengthsColumn},
                                   {spFractionColumn},
                                   {spDecimalColumn},
                                   {favTagColumn},
                                   {openingFractionColumn},
                                   {touchedHighColumn},
                                   {touchedLowColumn}
                            FROM Race r
                            JOIN Course c ON r.CourseId = c.CourseId
                            JOIN RunnerResult rr ON r.RaceId = rr.RaceId
                            LEFT JOIN Horse h ON rr.HorseId = h.HorseId
                            LEFT JOIN Trainer t ON rr.TrainerId = t.TrainerId
                            LEFT JOIN Jockey j ON rr.JockeyId = j.JockeyId
                            WHERE r.RaceDate < @TargetDate
                            ORDER BY r.RaceDate, r.RaceId, rr.RunnerResultId";

                var identifierKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "HorseName",
                "HorseId",
                "TrainerName",
                "TrainerId",
                "JockeyName",
                "JockeyId",
                "SaddleclothNumber",
                "Draw"
            };

                var featureState = new FeatureEngineeringState(_trainer, identifierKeys);
                return (sql, runnerColumns, featureState);
            }

            private List<(DateTime RaceDate, List<Dictionary<string, object?>> Rows)> MaterializeHistoricalRaces(
                SqlConnection conn,
                IEnumerable<object> records)
            {
                var races = new List<(DateTime RaceDate, List<Dictionary<string, object?>> Rows)>();
                var currentRows = new List<Dictionary<string, object?>>(capacity: 32);
                int? currentRaceId = null;
                DateTime? currentRaceDate = null;

                foreach (var record in records)
                {
                    if (record is null)
                    {
                        continue;
                    }

                    var source = (IDictionary<string, object?>)record;
                    var row = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                    foreach (var kvp in source)
                    {
                        row[kvp.Key] = NormalizeDbValue(kvp.Value);
                    }

                    if (!PreparedDataset.TryGetRequiredInt32(row, "RaceId", out var raceId))
                    {
                        continue;
                    }
                    if (!row.TryGetValue("RaceDate", out var raceDateObj) || raceDateObj is not DateTime raceDateValue)
                    {
                        continue;
                    }

                    var raceDate = raceDateValue.Date;
                    if (currentRaceId.HasValue && raceId != currentRaceId.Value)
                    {
                        if (currentRows.Count > 0 && currentRaceDate.HasValue)
                        {
                            ResolveHorseIdentifiers(conn, currentRows);
                            races.Add((currentRaceDate.Value, currentRows));
                        }

                        currentRows = new List<Dictionary<string, object?>>(currentRows.Count);
                    }

                    currentRows.Add(row);
                    currentRaceId = raceId;
                    currentRaceDate = raceDate;
                }

                if (currentRows.Count > 0 && currentRaceDate.HasValue)
                {
                    races.Add((currentRaceDate.Value, currentRows));
                }

                return races;
            }

            private PreparedRace? PrepareUpcomingRaceFromState(
                SqlConnection conn,
                UpcomingRace upcoming,
                IReadOnlyList<RunnerFlow> flows,
                IReadOnlyCollection<string> runnerColumns,
                FeatureEngineeringState featureState)
            {
                var syntheticRaceId = CreateSyntheticRaceId(upcoming);
                var syntheticRows = BuildUpcomingRaceRows(conn, upcoming, flows, syntheticRaceId, runnerColumns);
                if (syntheticRows.Count == 0)
                {
                    return null;
                }

                featureState.ProcessRace(syntheticRows, includeRace: true, updateState: false);
                return new PreparedRace(syntheticRaceId, syntheticRows);
            }

            private static int CreateSyntheticRaceId(UpcomingRace upcoming)
            {
                unchecked
                {
                    int hash = 17;
                    hash = hash * 31 + upcoming.RaceDate.GetHashCode();
                    if (!string.IsNullOrWhiteSpace(upcoming.MarketId))
                    {
                        hash = hash * 31 + StringComparer.OrdinalIgnoreCase.GetHashCode(upcoming.MarketId.Trim());
                    }
                    if (!string.IsNullOrWhiteSpace(upcoming.Title))
                    {
                        hash = hash * 31 + StringComparer.OrdinalIgnoreCase.GetHashCode(upcoming.Title.Trim());
                    }

                    return unchecked((int)(0x80000000 | ((uint)hash & 0x7FFFFFFF)));
                }
            }

            private List<Dictionary<string, object?>> BuildUpcomingRaceRows(
                SqlConnection conn,
                UpcomingRace upcoming,
                IReadOnlyList<RunnerFlow> flows,
                int raceId,
                IReadOnlyCollection<string> runnerColumns)
            {
                var rows = new List<Dictionary<string, object?>>(flows.Count);
                var (courseId, courseName) = _trainer.ResolveCourse(conn, upcoming);
                int runnerCount = flows.Count;
                var validFlows = new List<RunnerFlow>(flows.Count);
                var horseNames = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                var jockeyNames = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                var trainerNames = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                var horseIdsByName = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                var horseIdsByFlow = new Dictionary<RunnerFlow, int>();
                var explicitHorseIds = new HashSet<int>();

                foreach (var flow in flows)
                {
                    if (flow == null)
                    {
                        continue;
                    }

                    var horseName = flow.HorseName;
                    if (string.IsNullOrWhiteSpace(horseName))
                    {
                        continue;
                    }
                    validFlows.Add(flow);
                    if (!horseNames.ContainsKey(horseName))
                    {
                        horseNames[horseName] = horseName;
                    }
                    if (flow.FeatureValues != null &&
                        flow.FeatureValues.TryGetValue("HorseId", out var horseIdValue) &&
                        PreparedDataset.TryConvertToInt32(horseIdValue, out var parsedHorseId) &&
                        parsedHorseId > 0)
                    {
                        horseIdsByFlow[flow] = parsedHorseId;
                        explicitHorseIds.Add(parsedHorseId);
                        if (!horseIdsByName.ContainsKey(horseName))
                        {
                            horseIdsByName[horseName] = parsedHorseId;
                        }
                    }

                    if (!string.IsNullOrWhiteSpace(flow.JockeyName) && !jockeyNames.ContainsKey(flow.JockeyName!))
                    {
                        jockeyNames[flow.JockeyName!] = flow.JockeyName!;
                    }
                    if (!string.IsNullOrWhiteSpace(flow.TrainerName))
                    {
                        var trainerName = flow.TrainerName!.Trim();
                        if (!string.IsNullOrWhiteSpace(trainerName) && !trainerNames.ContainsKey(trainerName))
                        {
                            trainerNames[trainerName] = trainerName;
                        }
                    }
                }

                var horseNameList = horseNames.Values.ToList();
                var jockeyNameList = jockeyNames.Values.ToList();
                var trainerNameList = trainerNames.Values.ToList();
                var lookupData = LoadRunnerLookupData(
                    conn,
                    upcoming,
                    runnerColumns,
                    horseNameList,
                    jockeyNameList,
                    explicitHorseIds);
                var horseIdLookup = lookupData.HorseIds;
                var jockeyIdLookup = lookupData.JockeyIds;
                var runnerSnapshots = lookupData.RunnerSnapshots;
                var trainerIdLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

                if (trainerNames.Count > 0)
                {
                    var trainerCandidateMap = BuildNameCandidateMap(trainerNameList);
                    if (trainerCandidateMap.Count > 0)
                    {
                        const string trainerSql = "SELECT Name, MIN(TrainerId) AS TrainerId FROM Trainer WHERE Name IN @Names GROUP BY Name";
                        var trainerCandidateList = trainerCandidateMap.Keys.ToArray();
                        foreach (var (name, trainerId) in conn.Query<(string Name, int TrainerId)>(trainerSql, new { Names = trainerCandidateList }))
                        {
                            if (!trainerCandidateMap.TryGetValue(name, out var originals) || originals == null)
                            {
                                continue;
                            }

                            foreach (var original in originals)
                            {
                                if (!trainerIdLookup.ContainsKey(original))
                                {
                                    trainerIdLookup[original] = trainerId;
                                }
                            }
                        }
                    }

                    var unmatched = new HashSet<string>(trainerNames.Keys, StringComparer.OrdinalIgnoreCase);
                    foreach (var matched in trainerIdLookup.Keys)
                    {
                        unmatched.Remove(matched);
                    }

                    if (unmatched.Count > 0)
                    {
                        var normalizedMap = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
                        foreach (var original in unmatched)
                        {
                            if (string.IsNullOrWhiteSpace(original))
                            {
                                continue;
                            }

                            IEnumerable<string> candidates = trainerCandidateMap.Count > 0
                                ? trainerCandidateMap.Where(kvp => kvp.Value.Contains(original)).Select(kvp => kvp.Key)
                                : RacingRepository.BuildHistoricalNameCandidates(original);

                            foreach (var candidate in candidates)
                            {
                                var normalized = RacingRepository.NormalizeHistoricalNameKey(candidate);
                                if (string.IsNullOrEmpty(normalized))
                                {
                                    continue;
                                }

                                if (!normalizedMap.TryGetValue(normalized, out var originals))
                                {
                                    originals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                                    normalizedMap[normalized] = originals;
                                }

                                originals.Add(original);
                            }
                        }

                        if (normalizedMap.Count > 0)
                        {
                            PopulateNormalizedLookup(
                                conn,
                                "Trainer",
                                "TrainerId",
                                normalizedMap,
                                trainerIdLookup);
                        }
                    }
                }

                foreach (var pair in horseIdsByName)
                {
                    horseIdLookup[pair.Key] = pair.Value;
                }

                foreach (var flow in validFlows)
                {
                    var horseName = flow.HorseName!;
                    var row = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["RaceId"] = raceId,
                        ["CourseId"] = courseId,
                        ["CourseName"] = courseName,
                        ["RaceDate"] = upcoming.RaceDate,
                        ["ScheduledOff"] = upcoming.ScheduledOff,
                        ["ActualOff"] = null,
                        ["Title"] = upcoming.Title,
                        ["RaceType"] = upcoming.RaceType,
                        ["Surface"] = upcoming.Surface,
                        ["Going"] = upcoming.Going,
                        ["DistanceYards"] = upcoming.DistanceYards,
                        ["DistanceText"] = upcoming.DistanceText,
                        ["RunnerCount"] = upcoming.RunnerCount ?? Math.Min(runnerCount, byte.MaxValue),
                        ["Status"] = null,
                        ["WinningTimeMs"] = null,
                        ["HorseName"] = horseName,
                        ["JockeyName"] = flow.JockeyName,
                        ["SaddleclothNumber"] = flow.ClothNumber,
                        ["Draw"] = flow.Draw,
                        ["Age"] = null,
                        ["WeightLbs"] = null,
                        ["WeightText"] = null,
                        ["OfficialRating"] = null,
                        ["FinishPos"] = null,
                        ["OutcomeCode"] = null,
                        ["DistanceBeatenText"] = null,
                        ["SP_Fraction"] = null,
                        ["SP_Decimal"] = null,
                        ["FavTag"] = null,
                        ["OpeningFraction"] = null,
                        ["TouchedHighFraction"] = null,
                        ["TouchedLowFraction"] = null,
                        ["BackPrice1"] = flow.BackPrice1,
                        ["BackPrice2"] = flow.BackPrice2,
                        ["BackPrice3"] = flow.BackPrice3,
                        ["LayPrice1"] = flow.LayPrice1,
                        ["LayPrice2"] = flow.LayPrice2,
                        ["LayPrice3"] = flow.LayPrice3
                    };
                    var resolvedClass = ResolveUpcomingRaceClass(upcoming, flow);
                    if (resolvedClass.HasValue)
                    {
                        row["Class"] = resolvedClass.Value;
                    }
                    // Values that depend on historical lookups are populated below when data is available.
                    row["Purse"] = null;
                    row["TrainerId"] = null;
                    row["TrainerName"] = null;
                    var trainerNameKey = flow.TrainerName?.Trim();

                    int? resolvedHorseId = null;
                    if (horseIdsByFlow.TryGetValue(flow, out var horseIdFromFlow))
                    {
                        resolvedHorseId = horseIdFromFlow;
                    }
                    else if (horseIdLookup.TryGetValue(horseName, out var horseIdFromLookup))
                    {
                        resolvedHorseId = horseIdFromLookup;
                    }

                    if (resolvedHorseId.HasValue)
                    {
                        row["HorseId"] = resolvedHorseId.Value;
                    }
                    else
                    {
                        Console.WriteLine($"\t\tNo match found in Horse.Name for '{horseName}'; using synthetic horse identifier.");
                        row["HorseId"] = GenerateSyntheticId("horse:" + horseName);
                    }

                    if (!string.IsNullOrWhiteSpace(flow.JockeyName) &&
                        jockeyIdLookup.TryGetValue(flow.JockeyName!, out var jockeyId))
                    {
                        row["JockeyId"] = jockeyId;
                    }
                    else if (!string.IsNullOrWhiteSpace(flow.JockeyName))
                    {
                        Console.WriteLine($"\t\tNo match found in Jockey.Name for '{flow.JockeyName}'; jockey history will be unavailable.");
                    }

                    if (runnerColumns != null &&
                        row.TryGetValue("HorseId", out var horseIdObj) &&
                        horseIdObj is int horseIdFromRow &&
                        runnerSnapshots.TryGetValue(horseIdFromRow, out var snapshot))
                    {
                        if (row["TrainerId"] == null && snapshot.TrainerId.HasValue)
                        {
                            row["TrainerId"] = snapshot.TrainerId.Value;
                        }

                        if (row["TrainerName"] == null && !string.IsNullOrWhiteSpace(snapshot.TrainerName))
                        {
                            row["TrainerName"] = snapshot.TrainerName;
                        }

                        if (row["Age"] == null && snapshot.Age.HasValue)
                        {
                            row["Age"] = Convert.ToInt32(snapshot.Age.Value);
                        }

                        if (row["WeightLbs"] == null && snapshot.WeightLbs.HasValue)
                        {
                            row["WeightLbs"] = Convert.ToInt32(snapshot.WeightLbs.Value);
                        }

                        if (row["WeightText"] == null && snapshot.WeightText != null)
                        {
                            row["WeightText"] = snapshot.WeightText;
                        }

                        if (snapshot.OfficialRating.HasValue)
                        {
                            row["OfficialRating"] = Convert.ToInt32(snapshot.OfficialRating.Value);
                        }
                    }

                    if ((row["TrainerId"] == null || !PreparedDataset.TryConvertToInt32(row["TrainerId"], out _)) &&
                        !string.IsNullOrWhiteSpace(trainerNameKey) &&
                        trainerIdLookup.TryGetValue(trainerNameKey, out var trainerId))
                    {
                        row["TrainerId"] = trainerId;
                    }

                    if (row["TrainerName"] == null && !string.IsNullOrWhiteSpace(trainerNameKey))
                    {
                        row["TrainerName"] = trainerNameKey;
                    }

                    rows.Add(row);
                }
                return rows;
            }
            internal List<Dictionary<string, object?>> TestBuildUpcomingRaceRows(
               SqlConnection conn,
               UpcomingRace upcoming,
               IReadOnlyList<RunnerFlow> flows,
               int raceId,
               IReadOnlyCollection<string> runnerColumns) =>
               BuildUpcomingRaceRows(conn, upcoming, flows, raceId, runnerColumns);
            internal static (float AvgSpeed, float AvgSpeedDiff) TestComputeAverageSpeedForWindow(
               IReadOnlyList<(bool HasSpeed, float Speed, float SpeedDiff)> history,
               int window)
            {
                if (history == null)
                {
                    throw new ArgumentNullException(nameof(history));
                }

                if (window <= 0 || history.Count == 0)
                {
                    return (0f, 0f);
                }

                var entries = new List<HistoryEntry>(history.Count);
                var date = BaseDate;
                foreach (var sample in history)
                {
                    date = date.AddDays(1);
                    entries.Add(new HistoryEntry(
                        date,
                        NormFinish: 0f,
                        Finish: null,
                        Going: null,
                        Surface: null,
                        CourseId: 0,
                        Bucket: null,
                        RaceClass: null,
                        RaceSpeed: 0f,
                        Speed: sample.Speed,
                        SpeedDiff: sample.SpeedDiff,
                        Age: 0,
                        Won: false,
                        Rating: 0f,
                        Weight: 0f,
                        HasSpeed: sample.HasSpeed,
                        HasWinningTime: false,
                        WinningTimeMs: null,
                        DistanceYards: null));
                }

                var selected = FeatureEngineeringState.TakeRecentEntries(entries, window, h => h.HasSpeed);
                if (selected.Count == 0)
                {
                    return (0f, 0f);
                }

                return (
                    selected.Average(h => h.Speed),
                    selected.Average(h => h.SpeedDiff));
            }
            private static int? ResolveUpcomingRaceClass(UpcomingRace? upcoming, RunnerFlow? flow)
            {
                if (flow?.FeatureValues != null &&
                    flow.FeatureValues.TryGetValue("Class", out var classObj) &&
                    PreparedDataset.TryConvertToInt32(classObj, out var classFromFlow) &&
                    classFromFlow > 0)
                {
                    return classFromFlow;
                }

                static int? ParseClassFromText(string? text)
                {
                    if (string.IsNullOrWhiteSpace(text))
                    {
                        return null;
                    }

                    var match = UpcomingClassRegex.Match(text);
                    if (!match.Success)
                    {
                        return null;
                    }

                    return int.TryParse(match.Groups["value"].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) && value > 0
                        ? value
                        : null;
                }

                return ParseClassFromText(flow?.RaceDetails)
                       ?? ParseClassFromText(flow?.RaceTitle)
                       ?? ParseClassFromText(flow?.RaceType)
                       ?? ParseClassFromText(upcoming?.RaceDetails)
                       ?? ParseClassFromText(upcoming?.Title)
                       ?? ParseClassFromText(upcoming?.RaceType);
            }

            protected RunnerLookupData LoadRunnerLookupData(
                 SqlConnection conn,
                 UpcomingRace upcoming,
                 IReadOnlyCollection<string> runnerColumns,
                 IReadOnlyCollection<string> horseNames,
                 IReadOnlyCollection<string> jockeyNames,
                 IReadOnlyCollection<int>? horseIdsFromFlows = null)

            {
                var custom = _trainer.OverrideRunnerLookupData(
                    conn,
                    upcoming,
                    runnerColumns,
                    horseNames,
                    jockeyNames,
                    horseIdsFromFlows);

                if (custom is not null)
                {
                    return custom;
                }

                var horseIdLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                Dictionary<string, HashSet<string>>? horseCandidateMap = null;
                if (horseNames.Count > 0)
                {
                    horseCandidateMap = BuildNameCandidateMap(horseNames);
                    if (horseCandidateMap.Count > 0)
                    {
                        const string horseSql = "SELECT Name, MIN(HorseId) AS HorseId FROM Horse WHERE Name IN @Names GROUP BY Name";
                        var candidateList = horseCandidateMap.Keys.ToArray();
                        foreach (var (name, horseId) in conn.Query<(string Name, int HorseId)>(horseSql, new { Names = candidateList }))
                        {
                            if (!horseCandidateMap.TryGetValue(name, out var originals) || originals == null)
                            {
                                continue;
                            }

                            foreach (var original in originals)
                            {
                                if (!horseIdLookup.ContainsKey(original))
                                {
                                    horseIdLookup[original] = horseId;
                                }
                            }
                        }
                    }
                }
                if (horseNames.Count > 0)
                {
                    var unmatched = new HashSet<string>(horseNames, StringComparer.OrdinalIgnoreCase);
                    foreach (var matched in horseIdLookup.Keys)
                    {
                        unmatched.Remove(matched);
                    }

                    if (unmatched.Count > 0)
                    {
                        var normalizedMap = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
                        foreach (var original in unmatched)
                        {
                            if (string.IsNullOrWhiteSpace(original))
                            {
                                continue;
                            }

                            IEnumerable<string> candidates = horseCandidateMap != null && horseCandidateMap.Count > 0
                                ? horseCandidateMap.Where(kvp => kvp.Value.Contains(original)).Select(kvp => kvp.Key)
                                : RacingRepository.BuildHistoricalNameCandidates(original);

                            foreach (var candidate in candidates)
                            {
                                var normalized = RacingRepository.NormalizeHistoricalNameKey(candidate);
                                if (string.IsNullOrEmpty(normalized))
                                {
                                    continue;
                                }

                                if (!normalizedMap.TryGetValue(normalized, out var originals))
                                {
                                    originals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                                    normalizedMap[normalized] = originals;
                                }

                                originals.Add(original);
                            }
                        }

                        if (normalizedMap.Count > 0)
                        {
                            PopulateNormalizedLookup(
                                conn,
                                "Horse",
                                "HorseId",
                                normalizedMap,
                                horseIdLookup);
                        }
                    }
                }

                var jockeyIdLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                Dictionary<string, HashSet<string>>? jockeyCandidateMap = null;
                if (jockeyNames.Count > 0)
                {
                    jockeyCandidateMap = BuildNameCandidateMap(jockeyNames);
                    if (jockeyCandidateMap.Count > 0)
                    {
                        const string jockeySql = "SELECT Name, MIN(JockeyId) AS JockeyId FROM Jockey WHERE Name IN @Names GROUP BY Name";
                        var candidateList = jockeyCandidateMap.Keys.ToArray();
                        foreach (var (name, jockeyId) in conn.Query<(string Name, int JockeyId)>(jockeySql, new { Names = candidateList }))
                        {
                            if (!jockeyCandidateMap.TryGetValue(name, out var originals) || originals == null)
                            {
                                continue;
                            }

                            foreach (var original in originals)
                            {
                                if (!jockeyIdLookup.ContainsKey(original))
                                {
                                    jockeyIdLookup[original] = jockeyId;
                                }
                            }
                        }
                    }
                }
                if (jockeyNames.Count > 0)
                {
                    var unmatched = new HashSet<string>(jockeyNames, StringComparer.OrdinalIgnoreCase);
                    foreach (var matched in jockeyIdLookup.Keys)
                    {
                        unmatched.Remove(matched);
                    }

                    if (unmatched.Count > 0)
                    {
                        var normalizedMap = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
                        foreach (var original in unmatched)
                        {
                            if (string.IsNullOrWhiteSpace(original))
                            {
                                continue;
                            }

                            IEnumerable<string> candidates = jockeyCandidateMap != null && jockeyCandidateMap.Count > 0
                                ? jockeyCandidateMap.Where(kvp => kvp.Value.Contains(original)).Select(kvp => kvp.Key)
                                : RacingRepository.BuildHistoricalNameCandidates(original);

                            foreach (var candidate in candidates)
                            {
                                var normalized = RacingRepository.NormalizeHistoricalNameKey(candidate);
                                if (string.IsNullOrEmpty(normalized))
                                {
                                    continue;
                                }

                                if (!normalizedMap.TryGetValue(normalized, out var originals))
                                {
                                    originals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                                    normalizedMap[normalized] = originals;
                                }

                                originals.Add(original);
                            }
                        }

                        if (normalizedMap.Count > 0)
                        {
                            PopulateNormalizedLookup(
                                conn,
                                "Jockey",
                                "JockeyId",
                                normalizedMap,
                                jockeyIdLookup);
                        }
                    }
                }
                var runnerSnapshots = new Dictionary<int, RunnerSnapshot>();
                var knownHorseIds = new HashSet<int>();
                foreach (var horseId in horseIdLookup.Values)
                {
                    if (horseId > 0)
                    {
                        knownHorseIds.Add(horseId);
                    }
                }
                if (horseIdsFromFlows != null)
                {
                    foreach (var horseId in horseIdsFromFlows)
                    {
                        if (horseId > 0)
                        {
                            knownHorseIds.Add(horseId);
                        }
                    }
                }
                if (knownHorseIds.Count > 0)
                {
                    string weightTextColumn = runnerColumns.Contains("WeightText")
                        ? "rr.WeightText"
                        : "CAST(NULL AS nvarchar(50))";
                    string officialRatingColumn = runnerColumns.Contains("OfficialRating")
                        ? "rr.OfficialRating"
                        : "CAST(NULL AS smallint)";
                    string ageColumn = runnerColumns.Contains("Age")
                        ? "rr.Age"
                        : "CAST(NULL AS smallint)";
                    string weightLbsColumn = runnerColumns.Contains("WeightLbs")
                        ? "rr.WeightLbs"
                        : "CAST(NULL AS smallint)";

                    string snapshotSql = $@"SELECT ranked.HorseId,
                                                ranked.TrainerId,
                                                t.Name AS TrainerName,
                                                ranked.Age,
                                                ranked.WeightLbs,
                                                ranked.WeightText,
                                                ranked.OfficialRating
                                         FROM (
                                              SELECT rr.HorseId,
                                                     rr.TrainerId,
                                                     {ageColumn} AS Age,
                                                     {weightLbsColumn} AS WeightLbs,
                                                     {weightTextColumn} AS WeightText,
                                                     {officialRatingColumn} AS OfficialRating,
                                                     ROW_NUMBER() OVER (PARTITION BY rr.HorseId ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC) AS RowNum
                                              FROM RunnerResult rr
                                              JOIN Race r ON r.RaceId = rr.RaceId
                                              WHERE rr.HorseId IN @HorseIds AND r.RaceDate < @TargetDate
                                         ) ranked
                                         LEFT JOIN Trainer t ON ranked.TrainerId = t.TrainerId
                                         WHERE ranked.RowNum = 1";

                    var horseIdList = knownHorseIds.ToList();
                    foreach (var snapshot in conn.Query<RunnerSnapshot>(snapshotSql, new
                    {
                        HorseIds = horseIdList,
                        TargetDate = upcoming.RaceDate.Date
                    }))
                    {
                        runnerSnapshots[snapshot.HorseId] = snapshot;
                    }
                }

                return new RunnerLookupData(horseIdLookup, jockeyIdLookup, runnerSnapshots);
            }
            private static void ResolveHorseIdentifiers(
                SqlConnection conn,
                List<Dictionary<string, object?>> rows)
            {
                if (conn == null)
                {
                    throw new ArgumentNullException(nameof(conn));
                }

                if (rows == null || rows.Count == 0)
                {
                    return;
                }

                var unresolvedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var row in rows)
                {
                    if (row == null)
                    {
                        continue;
                    }

                    if (row.TryGetValue("HorseId", out var horseIdValue) &&
                        PreparedDataset.TryConvertToInt32(horseIdValue, out var existingId) &&
                        existingId > 0)
                    {
                        continue;
                    }

                    if (!row.TryGetValue("HorseName", out var horseNameObj) || horseNameObj == null)
                    {
                        continue;
                    }

                    var horseName = horseNameObj as string ?? horseNameObj.ToString();
                    if (string.IsNullOrWhiteSpace(horseName))
                    {
                        continue;
                    }

                    unresolvedNames.Add(horseName);
                }

                if (unresolvedNames.Count == 0)
                {
                    return;
                }

                var resolved = ResolveHorseIdsByName(conn, unresolvedNames);
                if (resolved.Count > 0)
                {
                    foreach (var row in rows)
                    {
                        if (row == null)
                        {
                            continue;
                        }

                        if (row.TryGetValue("HorseId", out var horseIdValue) &&
                            PreparedDataset.TryConvertToInt32(horseIdValue, out var existingId) &&
                            existingId > 0)
                        {
                            continue;
                        }

                        if (!row.TryGetValue("HorseName", out var horseNameObj) || horseNameObj == null)
                        {
                            continue;
                        }

                        var horseName = horseNameObj as string ?? horseNameObj.ToString();
                        if (string.IsNullOrWhiteSpace(horseName))
                        {
                            continue;
                        }

                        if (resolved.TryGetValue(horseName, out var horseId) && horseId > 0)
                        {
                            row["HorseId"] = horseId;
                        }
                    }
                }

                foreach (var row in rows)
                {
                    if (row == null)
                    {
                        continue;
                    }

                    if (row.TryGetValue("HorseId", out var horseIdValue) &&
                        PreparedDataset.TryConvertToInt32(horseIdValue, out var existingId) &&
                        existingId > 0)
                    {
                        continue;
                    }

                    if (!row.TryGetValue("HorseName", out var horseNameObj) || horseNameObj == null)
                    {
                        continue;
                    }

                    var horseName = horseNameObj as string ?? horseNameObj.ToString();
                    if (string.IsNullOrWhiteSpace(horseName))
                    {
                        continue;
                    }

                    row["HorseId"] = GenerateSyntheticId("horse:" + horseName);
                }
            }

            private static Dictionary<string, int> ResolveHorseIdsByName(
                SqlConnection conn,
                IReadOnlyCollection<string> horseNames)
            {
                if (horseNames == null)
                {
                    throw new ArgumentNullException(nameof(horseNames));
                }

                var horseIdLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                if (horseNames.Count == 0)
                {
                    return horseIdLookup;
                }

                var horseCandidateMap = BuildNameCandidateMap(horseNames);
                if (horseCandidateMap.Count > 0)
                {
                    const string horseSql = "SELECT Name, MIN(HorseId) AS HorseId FROM Horse WHERE Name IN @Names GROUP BY Name";
                    var candidateList = horseCandidateMap.Keys.ToArray();
                    foreach (var (name, horseId) in conn.Query<(string Name, int HorseId)>(horseSql, new { Names = candidateList }))
                    {
                        if (!horseCandidateMap.TryGetValue(name, out var originals) || originals == null)
                        {
                            continue;
                        }

                        foreach (var original in originals)
                        {
                            if (!horseIdLookup.ContainsKey(original))
                            {
                                horseIdLookup[original] = horseId;
                            }
                        }
                    }
                }

                var unmatched = new HashSet<string>(horseNames, StringComparer.OrdinalIgnoreCase);
                foreach (var matched in horseIdLookup.Keys)
                {
                    unmatched.Remove(matched);
                }

                if (unmatched.Count > 0)
                {
                    var normalizedMap = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
                    foreach (var original in unmatched)
                    {
                        if (string.IsNullOrWhiteSpace(original))
                        {
                            continue;
                        }

                        IEnumerable<string> candidates = horseCandidateMap.Count > 0
                            ? horseCandidateMap.Where(kvp => kvp.Value.Contains(original)).Select(kvp => kvp.Key)
                            : RacingRepository.BuildHistoricalNameCandidates(original);

                        foreach (var candidate in candidates)
                        {
                            var normalized = RacingRepository.NormalizeHistoricalNameKey(candidate);
                            if (string.IsNullOrEmpty(normalized))
                            {
                                continue;
                            }

                            if (!normalizedMap.TryGetValue(normalized, out var originals))
                            {
                                originals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                                normalizedMap[normalized] = originals;
                            }

                            originals.Add(original);
                        }
                    }

                    if (normalizedMap.Count > 0)
                    {
                        PopulateNormalizedLookup(conn, "Horse", "HorseId", normalizedMap, horseIdLookup);
                    }
                }

                return horseIdLookup;
            }
            private static void PopulateNormalizedLookup(
                SqlConnection conn,
                string tableName,
                string idColumn,
                IReadOnlyDictionary<string, HashSet<string>> normalizedMap,
                IDictionary<string, int> lookup)
            {
                if (normalizedMap == null || normalizedMap.Count == 0)
                {
                    return;
                }

                var pending = new HashSet<string>(normalizedMap.Keys, StringComparer.OrdinalIgnoreCase);
                if (pending.Count == 0)
                {
                    return;
                }

                var sql = $"SELECT {idColumn} AS Id, Name FROM {tableName} WHERE Name IS NOT NULL";
                foreach (var (Id, Name) in conn.Query<(int Id, string Name)>(sql))
                {
                    if (string.IsNullOrWhiteSpace(Name))
                    {
                        continue;
                    }

                    var normalized = RacingRepository.NormalizeHistoricalNameKey(Name);
                    if (string.IsNullOrEmpty(normalized) || !pending.Contains(normalized))
                    {
                        continue;
                    }

                    if (!normalizedMap.TryGetValue(normalized, out var originals) || originals == null)
                    {
                        continue;
                    }

                    foreach (var original in originals)
                    {
                        if (!lookup.ContainsKey(original))
                        {
                            lookup[original] = Id;
                        }
                    }

                    pending.Remove(normalized);
                    if (pending.Count == 0)
                    {
                        break;
                    }
                }
            }
            private static Dictionary<string, HashSet<string>> BuildNameCandidateMap(IReadOnlyCollection<string> names)
            {
                var map = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
                if (names == null || names.Count == 0)
                {
                    return map;
                }

                foreach (var name in names)
                {
                    if (string.IsNullOrWhiteSpace(name))
                    {
                        continue;
                    }

                    foreach (var candidate in RacingRepository.BuildHistoricalNameCandidates(name))
                    {
                        if (string.IsNullOrWhiteSpace(candidate))
                        {
                            continue;
                        }

                        if (!map.TryGetValue(candidate, out var originals))
                        {
                            originals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                            map[candidate] = originals;
                        }

                        originals.Add(name);
                    }
                }

                return map;
            }
        }
        public virtual PreparedDataset PrepareDataset(
           ISet<int?>? includeRaceIds = null,
           ISet<int?>? stateRaceWhitelist = null,
        bool includeIdentifiers = false,
           bool applyRepositoryBackfills = true)
        {
            var cacheKey = PreparedDatasetCacheKey.Create(includeRaceIds, stateRaceWhitelist, includeIdentifiers, applyRepositoryBackfills);

            lock (_preparedDatasetCacheLock)
            {
                if (_preparedDatasetCache.TryGetValue(cacheKey, out var cachedReference) &&
                    cachedReference.TryGetTarget(out var cached))
                {
                    return cached;
                }

                if (_preparedDatasetCache.Count > 0)
                {
                    var expiredKeys = new List<PreparedDatasetCacheKey>();
                    foreach (var pair in _preparedDatasetCache)
                    {
                        if (!pair.Value.TryGetTarget(out _))
                        {
                            expiredKeys.Add(pair.Key);
                        }
                    }

                    if (expiredKeys.Count > 0)
                    {
                        foreach (var key in expiredKeys)
                        {
                            _preparedDatasetCache.Remove(key);
                        }
                    }
                }
            }

            var prepared = CreatePreparedDataset(includeRaceIds, stateRaceWhitelist, includeIdentifiers, applyRepositoryBackfills);

            lock (_preparedDatasetCacheLock)
            {
                _preparedDatasetCache[cacheKey] = new WeakReference<PreparedDataset>(prepared);
            }

            return prepared;
        }
        protected virtual PreparedDataset CreatePreparedDataset(
            ISet<int?>? includeRaceIds,
            ISet<int?>? stateRaceWhitelist,
            bool includeIdentifiers,
            bool applyRepositoryBackfills)
        {
            var state = new FeatureEngineeringState(this);
            return state.PrepareDataset(includeRaceIds, stateRaceWhitelist, includeIdentifiers, applyRepositoryBackfills);
        }
        public static float? TestParseDistanceBeaten(string text)
        {
            return FeatureEngineeringState.TestParseDistanceBeaten(text);
        }
        public virtual PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
        {
            var results = PrepareUpcomingRaces(new[] { (upcoming, flows) });
            return results.Count > 0 ? results[0] : null;
        }

        public virtual IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
            IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
        {
            var state = new FeatureEngineeringState(this);
            return state.PrepareUpcomingRaces(requests);
        }
        protected virtual (int CourseId, string? CourseName) ResolveCourse(SqlConnection conn, UpcomingRace upcoming)
        {
            if (!string.IsNullOrWhiteSpace(upcoming.VenueName))
            {
                const string exactSql = "SELECT TOP (1) CourseId, Name FROM Course WHERE Name = @Name ORDER BY CourseId";
                var exact = conn.QuerySingleOrDefault<(int CourseId, string Name)?>(exactSql, new { Name = upcoming.VenueName });
                if (exact.HasValue)
                {
                    return (exact.Value.CourseId, exact.Value.Name);
                }

                const string courseSql = "SELECT CourseId, Name FROM Course";
                var allCourses = conn.Query<(int CourseId, string Name)>(courseSql).ToList();
                var normalizedVenue = NormalizeLookupKey(upcoming.VenueName);
                (int CourseId, string Name)? best = null;
                int bestScore = int.MaxValue;
                foreach (var course in allCourses)
                {
                    var candidate = NormalizeLookupKey(course.Name);
                    int score = 0;
                    if (candidate == normalizedVenue)
                    {
                        score -= 3;
                    }
                    else if (!string.IsNullOrEmpty(candidate) &&
                             (candidate.Contains(normalizedVenue) || normalizedVenue.Contains(candidate)))
                    {
                        score -= 1;
                    }
                    else
                    {
                        score += 1;
                    }

                    if (score < bestScore || (score == bestScore && (!best.HasValue || course.CourseId < best.Value.CourseId)))
                    {
                        best = course;
                        bestScore = score;
                    }
                }

                if (best.HasValue)
                {
                    return (best.Value.CourseId, best.Value.Name);
                }
            }

            Console.WriteLine($"\t\tNo match found in Course.Name for venue '{upcoming.VenueName ?? "<null>"}'; using synthetic course metadata.");
            int syntheticCourseId = GenerateSyntheticId("course:" + (upcoming.VenueName ?? upcoming.MarketId ?? string.Empty));
            return (syntheticCourseId, upcoming.VenueName);
        }

        protected virtual RunnerLookupData? OverrideRunnerLookupData(
            SqlConnection conn,
            UpcomingRace upcoming,
            IReadOnlyCollection<string> runnerColumns,
            IReadOnlyCollection<string> horseNames,
            IReadOnlyCollection<string> jockeyNames,
            IReadOnlyCollection<int>? horseIdsFromFlows)
        {
            return null;
        }

        private static int GenerateSyntheticId(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return int.MaxValue;
            }

            unchecked
            {
                int hash = 17;
                hash = hash * 31 + StringComparer.OrdinalIgnoreCase.GetHashCode(value.Trim());
                return 0x60000000 | (hash & 0x0FFFFFFF);
            }
        }

        private static string NormalizeLookupKey(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var lower = value.Trim().ToLowerInvariant();
            lower = System.Text.RegularExpressions.Regex.Replace(lower, "[^a-z0-9]+", " ");
            lower = System.Text.RegularExpressions.Regex.Replace(lower, "\\s+", " ").Trim();
            return lower;
        }
        internal List<Dictionary<string, object?>> TestBuildUpcomingRaceRows(
            SqlConnection conn,
            UpcomingRace upcoming,
            IReadOnlyList<RunnerFlow> flows,
            int raceId,
            IReadOnlyCollection<string> runnerColumns)
        {
            var state = new FeatureEngineeringState(this);
            return state.TestBuildUpcomingRaceRows(conn, upcoming, flows, raceId, runnerColumns);
        }

        internal static (float AvgSpeed, float AvgSpeedDiff) TestComputeAverageSpeedForWindow(
            IReadOnlyList<(bool HasSpeed, float Speed, float SpeedDiff)> history,
            int window)
        {
            return FeatureEngineeringState.TestComputeAverageSpeedForWindow(history, window);
        }
        protected sealed class RunnerSnapshot
        {
            public int HorseId { get; set; }
            public int? TrainerId { get; set; }
            public string? TrainerName { get; set; }
            public short? Age { get; set; }
            public short? WeightLbs { get; set; }
            public string? WeightText { get; set; }
            public short? OfficialRating { get; set; }
        }

        protected sealed class RunnerLookupData
        {
            public RunnerLookupData(
                Dictionary<string, int> horseIds,
                Dictionary<string, int> jockeyIds,
                Dictionary<int, RunnerSnapshot> runnerSnapshots)
            {
                HorseIds = horseIds ?? throw new ArgumentNullException(nameof(horseIds));
                JockeyIds = jockeyIds ?? throw new ArgumentNullException(nameof(jockeyIds));
                RunnerSnapshots = runnerSnapshots ?? throw new ArgumentNullException(nameof(runnerSnapshots));
            }

            public Dictionary<string, int> HorseIds { get; }
            public Dictionary<string, int> JockeyIds { get; }
            public Dictionary<int, RunnerSnapshot> RunnerSnapshots { get; }
        }
    }
}
