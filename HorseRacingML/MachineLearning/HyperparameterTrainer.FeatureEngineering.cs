using Dapper;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;
using System;
using System.Collections.Generic;
using System.Globalization;
using HorseRacingML.Data;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;
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
        private const int TrainerJockeyRecentStarts = 50;
        private const int TrainerJockeyRecentDays = 180;
        private const float TypicalRestDays = 30f;
        private const float MsPerLength = 200f;
        private static readonly Regex HorseNameBracketTextRegex =
            new Regex("\\s*\\([^\\)]*\\)|\\s*\\[[^\\]]*\\]", RegexOptions.Compiled);
        private static readonly Regex UpcomingClassRegex =
            new("class\\s*(?:[:\\-]?\\s*)?(?<value>[0-9]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex HorseNameWhitespaceRegex = new Regex("\\s+", RegexOptions.Compiled);
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

        private sealed class FeatureEngineeringState
        {
            private readonly HyperparameterTrainer _trainer;
            private readonly ISet<string>? _identifierKeys;
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

            public FeatureEngineeringState(HyperparameterTrainer trainer, ISet<string>? identifierKeys = null)
            {
                _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
                _identifierKeys = identifierKeys;
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
            private static float ClampNormalizedPosition(float value)
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

                    runnerRow["DistanceBeatenKnown"] = distanceKnown;
                    runnerRow["DistanceBeatenLengths"] = beatenLengths;


                    var raceStat = ComputeRaceStats(rows);

                    foreach (var row in rows)
                    {
                        int horseId = PreparedDataset.GetRequiredInt32(row, "HorseId");
                        DateTime date = (DateTime)row["RaceDate"];
                        if (TryGetTimeOfDay(row, out var timeOfDay))
                        {
                            float minutes = (float)timeOfDay.TotalMinutes;
                            float timeAngle = 2f * MathF.PI * minutes / (24f * 60f);
                            row["TimeOfDaySin"] = MathF.Sin(timeAngle);
                            row["TimeOfDayCos"] = MathF.Cos(timeAngle);
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
                            row.Remove("RaceDate");
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
                            row["DistanceMissing"] = distanceMissing;
                            row["DistanceTextMissing"] = distanceTextMissing;

                            bool backBookMissing = row["BackBookPercentage"] == null;
                            bool layBookMissing = row["LayBookPercentage"] == null;
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

                            bool ratingMissing = !(row.TryGetValue("OfficialRating", out var ratingObj) && ratingObj != null);
                            float rating = ratingMissing ? raceStat.AvgRating : Convert.ToSingle(ratingObj);
                            row["RatingMissing"] = ratingMissing;
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
                            row["RatingDiffFromField"] = rating - raceStat.AvgRating;
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
                                    // Oldest first for regression
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
                                row["RecentImprovement"] =
                                        (float)row["Last1NormPos"] - (float)row[$"Last{PastRaceCount}NormPos"];
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

                                    // Ensure `recent` is available regardless of branch to avoid scope issues.
                                    List<HistoryEntry> recent;
                                    if (count > 0)
                                    {
                                        recent = history.GetRange(history.Count - count, count);
                                        int wins = recent.Count(h => h.Finish == 1);

                                        row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(wins, count);
                                        row[$"AvgNormPosLast{window}"] = recent.Sum(h => h.NormFinish) / count;
                                        row[$"AvgRatingLast{window}"] = recent.Sum(h => h.Rating) / count;

                                    }
                                    else
                                    {
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
                                // Going + Course preference
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
                                int? winningMs = row.TryGetValue("WinningTimeMs", out var winningObj) && PreparedDataset.TryConvertToInt32(winningObj, out var winningValue)
                                    ? winningValue
                                : (int?)null;
                                bool winningTimeAvailable = winningMs.HasValue && winningMs.Value > 0;
                                float raceSpeed = winningTimeAvailable && distanceYards > 0
                                    ? distanceYards / (float)winningMs.Value
                                    : 0f;
                                bool distanceBeatenKnown = row.TryGetValue("DistanceBeatenKnown", out var distanceKnownObj) &&
                                    distanceKnownObj is bool distanceKnownBool && distanceKnownBool;

                                bool hasRunnerSpeed = winningTimeAvailable && distanceBeatenKnown;
                                runnerSpeed = 0f;
                                if (hasRunnerSpeed)
                                {
                                    float beaten = Convert.ToSingle(row["DistanceBeatenLengths"]);
                                    float runnerTime = winningMs!.Value + beaten * MsPerLength;
                                    runnerSpeed = runnerTime > 0f ? distanceYards / runnerTime : 0f;
                                }
                                float speedDiff = hasRunnerSpeed ? runnerSpeed - raceSpeed : 0f;
                                string bucket = distanceMissing ? "Unknown" : DistanceBucket(distanceYards);
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

                                        var goingRecent = recent.Where(h => h.Going == going).ToList();
                                        row[$"GoingWinRateLast{window}"] = _trainer.SmoothedWinRate(goingRecent.Count(h => h.Finish == 1), goingRecent.Count);
                                        row[$"GoingAvgNormLast{window}"] = goingRecent.Count > 0
                                            ? goingRecent.Sum(h => h.NormFinish) / goingRecent.Count
                                            : 0f;
                                        var surfaceRecent = recent.Where(h => h.Surface == surface).ToList();
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
                                        row[$"GoingWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                        row[$"GoingAvgNormLast{window}"] = 0f;
                                        row[$"SurfaceWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                        row[$"SurfaceAvgNormLast{window}"] = 0f;
                                        row[$"CourseWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                        row[$"CourseAvgNormLast{window}"] = 0f;
                                        row[$"DistanceBucketWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                        row[$"DistanceBucketAvgNormLast{window}"] = 0f;
                                        row[$"AvgSpeedLast{window}"] = 0f;
                                        row[$"AvgSpeedDiffLast{window}"] = 0f;
                                    }
                                }
                                // Preferred distance deviation
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

                                var globalDefaultWinRate = _trainer.SmoothedWinRate(0, 0);
                                RollingStat trainerStat = default;
                                bool hasTrainerStat = false;
                                float trainerDefaultWinRate = globalDefaultWinRate;

                                // Trainer statistics
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

                                    if (updateState)
                                    {
                                        trainerStat.Starts++;
                                        bool tWin = finish.HasValue && finish.Value == 1;
                                        if (tWin) trainerStat.Wins++;
                                        trainerStat.Recent.Enqueue((date, tWin));
                                        while (trainerStat.Recent.Count > TrainerJockeyRecentStarts)
                                            trainerStat.Recent.Dequeue();
                                        _trainerStats[trainerId.Value] = trainerStat;
                                    }
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

                                    if (updateState)
                                    {
                                        jockeyStat.Starts++;
                                        bool jWin = finish.HasValue && finish.Value == 1;
                                        if (jWin) jockeyStat.Wins++;
                                        jockeyStat.Recent.Enqueue((date, jWin));
                                        while (jockeyStat.Recent.Count > TrainerJockeyRecentStarts)
                                            jockeyStat.Recent.Dequeue();
                                        _jockeyStats[jockeyId.Value] = jockeyStat;
                                    }
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
                                if (updateState)
                                {
                                    // Compute normalized finish
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
                                        winningTimeAvailable ? (float?)winningMs : null,
                                        distanceMissing ? null : (float?)distanceYards));
                                    if (!classMissing)
                                    {
                                        horseClassStat.starts++;
                                        horseClassStat.sumNorm += normFinish;
                                        if (finish.HasValue && finish.Value == 1) horseClassStat.wins++;
                                        horseClassStat.lastNorm = normFinish;
                                        _horseClassStats[horseClassKey] = horseClassStat;
                                    }

                                    if (hasTrainerClass)
                                    {
                                        trainerClassStat.starts++;
                                        trainerClassStat.sumNorm += normFinish;
                                        if (finish.HasValue && finish.Value == 1) trainerClassStat.wins++;
                                        trainerClassStat.lastNorm = normFinish;
                                        _trainerClassStats[trainerClassKey] = trainerClassStat;
                                    }

                                    if (hasJockeyClass)
                                    {
                                        jockeyClassStat.starts++;
                                        jockeyClassStat.sumNorm += normFinish;
                                        if (finish.HasValue && finish.Value == 1) jockeyClassStat.wins++;
                                        jockeyClassStat.lastNorm = normFinish;
                                        _jockeyClassStats[jockeyClassKey] = jockeyClassStat;
                                    }
                                    if (history.Count > HistoryLength)
                                        history.RemoveAt(0);
                                    if (!surfaceMissing)
                                    {
                                        sStats.starts++;
                                        sStats.sumNorm += normFinish;
                                        if (finish.HasValue && finish.Value == 1) sStats.wins++;
                                        sStats.lastNorm = normFinish;
                                        sDict[surface] = sStats;
                                    }

                                    if (!goingMissing)
                                    {
                                        gStats.starts++;
                                        gStats.sumNorm += normFinish;
                                        if (finish.HasValue && finish.Value == 1) gStats.wins++;
                                        gStats.lastNorm = normFinish;
                                        gDict[going] = gStats;

                                        gcStats.starts++;
                                        gcStats.sumNorm += normFinish;
                                        if (finish.HasValue && finish.Value == 1) gcStats.wins++;
                                        gcStats.lastNorm = normFinish;
                                        gcDict[gcKey] = gcStats;
                                    }
                                    cStats.starts++;
                                    cStats.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) cStats.wins++;
                                    cStats.lastNorm = normFinish;
                                    cDict[courseId] = cStats;

                                    if (!distanceMissing)
                                    {
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

                                        allDist.sum += distanceYards;
                                        allDist.count++;
                                        _horseDistanceAll[horseId] = allDist;
                                        _horseLastDistance[horseId] = distanceYards;
                                        if (finish.HasValue && finish.Value == 1)
                                        {
                                            winDist.sum += distanceYards;
                                            winDist.count++;
                                        }
                                        _horseDistanceWins[horseId] = winDist;
                                    }
                                    drawStat.starts++;
                                    if (finish.HasValue && finish.Value == 1) drawStat.wins++;
                                    _drawStats[drawKey] = drawStat;
                                    baseStat.starts++;
                                    if (finish.HasValue && finish.Value == 1) baseStat.wins++;
                                    _drawBaselineStats[baseKey] = baseStat;
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
                            TrimRunnerRow(raceRow, _identifierKeys);
                        }
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
                    weightValues.Count > 0);
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
                       bool HasWeightStats);
        }

        private static float? ParseDistanceBeaten(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return null;
            text = text.Trim().ToLowerInvariant();
            var map = new Dictionary<string, float>
            {
                {"nse", 0.05f},
                {"nose", 0.05f},
                {"shd", 0.1f},
                {"sht-hd", 0.1f},
                {"hd", 0.2f},
                {"snk", 0.25f},
                {"nk", 0.3f},
                {"dist", 30f}
            };
            if (map.TryGetValue(text, out var val))
                return val;
            text = text.Replace("¼", ".25").Replace("½", ".5").Replace("¾", ".75");
            double total = 0;
            foreach (var part in text.Split(new[] { ' ', '+' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (double.TryParse(part, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out var num))
                {
                    total += num;
                }
                else if (part.Contains('/'))
                {
                    var frac = part.Split('/');
                    if (frac.Length == 2 &&
                        double.TryParse(frac[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) &&
                        double.TryParse(frac[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var d) &&
                        d != 0)
                    {
                        total += n / d;
                    }
                }
            }
            return total > 0 ? (float)total : (float?)null;
        }

        private float SmoothedWinRate(int wins, int starts)
            => ComputeSmoothedWinRate(wins, starts);


        private static object? NormalizeDbValue(object? value)
        {
            return value == null || value is DBNull ? null : value;
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

        public virtual PreparedDataset PrepareDataset(
                ISet<int>? includeRaceIds = null,
                ISet<int>? stateRaceWhitelist = null,
                bool includeIdentifiers = false)
        {
            using var conn = new SqlConnection(_connectionString);
            conn.Open();

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
                                   rr.HorseId,
                                   rr.TrainerId,
                                   rr.JockeyId,
                                   {saddleclothColumn},
                                   {drawColumn},
                                   {ageColumn},
                                   {weightLbsColumn},
                                   {weightTextColumn},
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

            var featureState = new FeatureEngineeringState(this, identifierKeys);
            var races = new List<PreparedRace>();
            var rnd = new Random();
            var currentRows = new List<Dictionary<string, object?>>();
            int? currentRaceId = null;
            bool ShouldInclude(int raceId) => includeRaceIds is null || includeRaceIds.Contains(raceId);
            bool ShouldUpdate(int raceId) => stateRaceWhitelist is null || stateRaceWhitelist.Contains(raceId);

            foreach (var record in conn.Query(sql, commandTimeout: 6000, buffered: false))
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
                    Shuffle(currentRows, rnd);
                    int previousRaceId = currentRaceId.Value;
                    bool include = ShouldInclude(previousRaceId);
                    bool update = ShouldUpdate(previousRaceId);
                    if (include || update)
                    {
                        ResolveHorseIdentifiers(conn, currentRows);
                        featureState.ProcessRace(currentRows, include, update);
                        if (include)
                        {
                            races.Add(new PreparedRace(previousRaceId, currentRows));
                        }
                    }
                    currentRows = new List<Dictionary<string, object?>>();
                }

                currentRows.Add(row);
                currentRaceId = raceId;
            }

            if (currentRaceId.HasValue && currentRows.Count > 0)
            {
                Shuffle(currentRows, rnd);
                int finalRaceId = currentRaceId.Value;
                bool include = ShouldInclude(finalRaceId);
                bool update = ShouldUpdate(finalRaceId);
                if (include || update)
                {
                    ResolveHorseIdentifiers(conn, currentRows);
                    featureState.ProcessRace(currentRows, include, update);
                    if (include)
                    {
                        races.Add(new PreparedRace(finalRaceId, currentRows));
                    }
                }
            }

            return new PreparedDataset(races);
        }
        public virtual PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
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

        public virtual IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
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

            using var conn = new SqlConnection(_connectionString);
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

            var featureState = new FeatureEngineeringState(this, identifierKeys);
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
            var (courseId, courseName) = ResolveCourse(conn, upcoming);
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
            var lookupData = LoadRunnerLookupData(conn, upcoming, runnerColumns, horseNameList, jockeyNameList, explicitHorseIds);
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

        protected virtual RunnerLookupData LoadRunnerLookupData(
             SqlConnection conn,
             UpcomingRace upcoming,
             IReadOnlyCollection<string> runnerColumns,
             IReadOnlyCollection<string> horseNames,
             IReadOnlyCollection<string> jockeyNames,
             IReadOnlyCollection<int>? horseIdsFromFlows = null)
        {
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
            if (resolved.Count == 0)
            {
                return;
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

                if (resolved.TryGetValue(horseName, out var horseId) && horseId > 0)
                {
                    row["HorseId"] = horseId;
                }
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
    }
}