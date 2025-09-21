using Dapper;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Tensorflow;
using Tensorflow.NumPy;
using static HorseRacingML.ML.HyperparameterTrainer.TrainingDataset;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static Tensorflow.Binding;
using TensorShape = Tensorflow.Shape;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Builds and trains a simple TensorFlow model using supplied hyperparameters.
    /// GPU is used when available. Training data is loaded from SQL Server using the
    /// HorseRacingML schema instead of randomly generated dummy values.
    /// </summary>
    public class HyperparameterTrainer
    {
        private readonly string _connectionString;
        private readonly float _winRateAlpha;
        private readonly float _winRateBeta;

        public HyperparameterTrainer(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
            _winRateAlpha = configuration.GetValue<float>("WinRateAlpha", 1f);
            _winRateBeta = configuration.GetValue<float>("WinRateBeta", 2f);
        }
        public class TrainingDataset
        {
            public TrainingDataset(
                List<RaceExample> races,
                List<string> featureKeys,
                Dictionary<string, int> featureDimensions,
                Dictionary<string, Dictionary<string, int>> stringMaps,
                NormalizationParameters normalization)
            {
                Races = races;
                FeatureKeys = featureKeys;
                FeatureDimensions = featureDimensions;
                StringMaps = stringMaps;
                Normalization = normalization;
                FeatureCount = featureDimensions.Values.Sum();
            }
            public class PreparedDataset
            {
                public PreparedDataset(List<PreparedRace> races)
                {
                    Races = races;
                }

                public List<PreparedRace> Races { get; }
                public IEnumerable<Dictionary<string, object?>> Rows => Races.SelectMany(r => r.Rows);
                public int RowCount => Races.Sum(r => r.Rows.Count);
                private readonly Dictionary<(int FoldCount, int FoldIndex), EncodingCacheEntry> _encodingCache = new();

                internal bool TryGetEncodingCache(int foldIndex, int foldCount, out EncodingCacheEntry? entry)
                {
                    return _encodingCache.TryGetValue((foldCount, foldIndex), out entry);
                }

                internal EncodingCacheEntry SetEncodingCache(
                    int foldIndex,
                    int foldCount,
                    List<RaceExample> races,
                    List<string> featureKeys,
                    Dictionary<string, int> featureDimensions,
                    Dictionary<string, Dictionary<string, int>> stringMaps)
                {
                    var entry = new EncodingCacheEntry(races, featureKeys, featureDimensions, stringMaps);
                    _encodingCache[(foldCount, foldIndex)] = entry;
                    return entry;
                }

                internal sealed class EncodingCacheEntry
                {
                    public EncodingCacheEntry(
                        List<RaceExample> races,
                        List<string> featureKeys,
                        Dictionary<string, int> featureDimensions,
                        Dictionary<string, Dictionary<string, int>> stringMaps)
                    {
                        Races = races;
                        FeatureKeys = featureKeys;
                        FeatureDimensions = featureDimensions;
                        StringMaps = stringMaps;
                        FeatureCount = featureDimensions.Values.Sum();
                    }

                    public List<RaceExample> Races { get; }
                    public List<string> FeatureKeys { get; }
                    public Dictionary<string, int> FeatureDimensions { get; }
                    public Dictionary<string, Dictionary<string, int>> StringMaps { get; }
                    public int FeatureCount { get; }
                }
            }

            public class PreparedRace
            {
                public PreparedRace(int raceId, List<Dictionary<string, object?>> rows)
                {
                    RaceId = raceId;
                    Rows = rows;
                }

                public int RaceId { get; }
                public List<Dictionary<string, object?>> Rows { get; }
            }
            public List<RaceExample> Races { get; }
            public List<string> FeatureKeys { get; }
            public Dictionary<string, int> FeatureDimensions { get; }
            public Dictionary<string, Dictionary<string, int>> StringMaps { get; }
            public NormalizationParameters Normalization { get; }
            public int FeatureCount { get; }
        }

        public class RaceExample
        {
            public RaceExample(int raceId, List<RunnerExample> runners)
            {
                RaceId = raceId;
                Runners = runners;
            }

            public int RaceId { get; }
            public List<RunnerExample> Runners { get; }
        }

        public class RunnerExample
        {
            public RunnerExample(int raceId, float[] features, float label)
            {
                RaceId = raceId;
                Features = features;
                Label = label;
            }

            public int RaceId { get; }
            public float[] Features { get; }
            public float Label { get; }
        }
        private static double ComputeWinnerAccuracy(IReadOnlyList<int> raceIds,
            IReadOnlyList<float> preds, IReadOnlyList<float> labels)
        {
            var grouped = raceIds.Select((raceId, idx) => new { raceId, idx })
                                 .GroupBy(x => x.raceId);

            int correct = 0;
            int total = 0;

            foreach (var group in grouped)
            {
                total++;
                var bestPred = group.OrderByDescending(g => preds[g.idx]).First().idx;
                var trueIdx = group.OrderByDescending(g => labels[g.idx]).First().idx;
                if (bestPred == trueIdx) correct++;
            }

            return total == 0 ? 0 : (double)correct / total;
        }
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
            string Going,
            string Surface,
            int CourseId,
            string Bucket,
            int RaceClass,
            float Speed,
            float SpeedDiff,
            int Age,
            bool Won,
            float Rating,
            float Weight);
        private static float[] EncodeFeature(
            string key,
            object? value,
            int dim,
            Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            // Determine the base dimension (excluding missing indicator) and whether an
            // additional slot is reserved for missing values.
            int baseDim = stringMaps.TryGetValue(key, out var map)
                ? map.Count
                : 1;
            bool hasMissingIndicator = dim > baseDim;

            if (value == null)
            {
                var arr = new float[dim];
                if (hasMissingIndicator)
                    arr[baseDim] = 1f;
                return arr;
            }

            float[] encoded = value switch
            {
                // Convert to days relative to a recent base date to avoid huge tick values
                DateTime dt => new[] { (float)(dt - BaseDate).TotalDays },
                // Seconds are a reasonable scale for durations
                TimeSpan ts => new[] { (float)ts.TotalSeconds },
                string s => EncodeString(key, s, baseDim, stringMaps),
                bool b => new[] { b ? 1f : 0f },
                // Scale down very large numeric values to keep them in a manageable range
                _ => EncodeNumeric(value, baseDim)
            };

            if (!hasMissingIndicator)
                return encoded;

            var result = new float[dim];
            Array.Copy(encoded, result, baseDim);
            return result;
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
            => (wins + _winRateAlpha) / (starts + _winRateBeta);

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

        private static void TrimRunnerRow(Dictionary<string, object?> row)
        {
            if (row is null)
                throw new ArgumentNullException(nameof(row));

            foreach (var key in NonTrainingKeys)
            {
                row.Remove(key);
            }
        }

        private sealed class FeatureEngineeringState
        {
            private readonly HyperparameterTrainer _trainer;
            private readonly Dictionary<int, List<HistoryEntry>> _horseHistory = new();
            private readonly Dictionary<int, RollingStat> _trainerStats = new();
            private readonly Dictionary<int, RollingStat> _jockeyStats = new();
            private readonly Dictionary<(int trainerId, int jockeyId), (int starts, int wins)> _trainerJockeyStats = new();
            private readonly Dictionary<(int trainerId, int jockeyId, string surface), (int starts, int wins)> _trainerJockeySurfaceStats = new();
            private readonly Dictionary<(int trainerId, int courseId), (int starts, int wins)> _trainerCourseStats = new();
            private readonly Dictionary<(int jockeyId, int courseId), (int starts, int wins)> _jockeyCourseStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _surfaceStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _goingStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _goingCourseStats = new();
            private readonly Dictionary<int, Dictionary<int, (int starts, int wins, float sumNorm, float lastNorm)>> _courseStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _ageStats = new();
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

            public FeatureEngineeringState(HyperparameterTrainer trainer)
            {
                _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
            }

            public void ProcessRace(List<Dictionary<string, object?>> rows)
            {
                if (rows is null)
                {
                    throw new ArgumentNullException(nameof(rows));
                }
                if (rows.Count == 0)
                {
                    return;
                }

                foreach (var row in rows)
                {
                    bool distanceKnown = false;
                    float beatenLengths = 0f;
                    if (row.TryGetValue("DistanceBeatenLengths", out var lenObj) && lenObj != null)
                    {
                        beatenLengths = Convert.ToSingle(lenObj);
                        distanceKnown = true;
                    }

                    if (row.TryGetValue("DistanceBeatenText", out var txtObj) && txtObj is string txt)
                    {
                        var parsed = ParseDistanceBeaten(txt);
                        if (parsed.HasValue)
                        {
                            beatenLengths = parsed.Value;
                            distanceKnown = true;
                        }
                    }

                    row["DistanceBeatenKnown"] = distanceKnown;
                    row["DistanceBeatenLengths"] = beatenLengths;


                    var raceStat = ComputeRaceStats(rows);

                    foreach (var row in rows)
                    {
                        int horseId = Convert.ToInt32(row["HorseId"]);
                        DateTime date = (DateTime)row["RaceDate"];
                        TimeSpan? off = null;
                        if (row.TryGetValue("ActualOff", out var offObj) && offObj != null)
                            off = (TimeSpan)offObj;
                        else if (row.TryGetValue("ScheduledOff", out offObj) && offObj != null)
                            off = (TimeSpan)offObj;
                        float timeAngle = 0f;
                        if (off.HasValue)
                        {
                            float minutes = (float)off.Value.TotalMinutes;
                            timeAngle = 2f * MathF.PI * minutes / (24f * 60f);
                        }
                        row["TimeOfDaySin"] = MathF.Sin(timeAngle);
                        row["TimeOfDayCos"] = MathF.Cos(timeAngle);

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
                        int raceId = Convert.ToInt32(row["RaceId"]);
                        int runnerCount = row["RunnerCount"] != null ? Convert.ToInt32(row["RunnerCount"]) : raceStat.RunnerCount;
                        int? trainerId = row.TryGetValue("TrainerId", out var tObj) && tObj != null
                            ? Convert.ToInt32(tObj)
                            : (int?)null;
                        int? jockeyId = row.TryGetValue("JockeyId", out var jObj) && jObj != null
                            ? Convert.ToInt32(jObj)
                            : (int?)null;

                        bool drawMissing = row["Draw"] == null;
                        int draw = !drawMissing ? Convert.ToInt32(row["Draw"]) : 0;
                        row["DrawMissing"] = drawMissing;

                        float runnerSpeed = 0f;

                        bool weightMissing = row["WeightLbs"] == null;
                        float weight = weightMissing ? 0f : Convert.ToSingle(row["WeightLbs"]);
                        row["WeightMissing"] = weightMissing;

                        bool ratingMissing = !(row.TryGetValue("OfficialRating", out var ratingObj) && ratingObj != null);
                        float rating = ratingMissing ? raceStat.AvgRating : Convert.ToSingle(ratingObj);
                        row["RatingMissing"] = ratingMissing;
                        row["RelativeDraw"] = runnerCount > 0 ? (float)draw / runnerCount : 0f;
                        bool saddleclothMissing = !(row.TryGetValue("SaddleclothNumber", out var saddleclothObj) && saddleclothObj != null);
                        int saddlecloth = saddleclothMissing ? 0 : Convert.ToInt32(saddleclothObj);
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
                        int age = row["Age"] != null ? Convert.ToInt32(row["Age"]) : 0;
                        row["AgeRelative"] = age - raceStat.AvgAge;
                        int classVal = row["Class"] != null ? Convert.ToInt32(row["Class"]) : 0;
                        var horseClassKey = (horseId, classVal);
                        if (!_horseClassStats.TryGetValue(horseClassKey, out var horseClassStat))
                            horseClassStat = (0, 0, 0f, 0f);
                        row["ClassWinRate"] = _trainer.SmoothedWinRate(horseClassStat.wins, horseClassStat.starts);
                        row["ClassAvgNorm"] = horseClassStat.starts > 0
                            ? horseClassStat.sumNorm / horseClassStat.starts
                            : 0f;
                        row["LastClassNormPos"] = horseClassStat.lastNorm;

                        (int trainerId, int classVal) trainerClassKey = default;
                        (int starts, int wins, float sumNorm, float lastNorm) trainerClassStat = default;
                        bool hasTrainerClass = false;
                        if (trainerId.HasValue)
                        {
                            trainerClassKey = (trainerId.Value, classVal);
                            if (!_trainerClassStats.TryGetValue(trainerClassKey, out trainerClassStat))
                                trainerClassStat = (0, 0, 0f, 0f);
                            row["TrainerClassWinRate"] = _trainer.SmoothedWinRate(trainerClassStat.wins, trainerClassStat.starts);
                            row["TrainerClassAvgNorm"] = trainerClassStat.starts > 0
                                ? trainerClassStat.sumNorm / trainerClassStat.starts
                                : 0f;
                            row["LastTrainerClassNormPos"] = trainerClassStat.lastNorm;
                            hasTrainerClass = true;
                        }
                        else
                        {
                            row["TrainerClassWinRate"] = 0f;
                            row["TrainerClassAvgNorm"] = 0f;
                            row["LastTrainerClassNormPos"] = 0f;
                        }

                        (int jockeyId, int classVal) jockeyClassKey = default;
                        (int starts, int wins, float sumNorm, float lastNorm) jockeyClassStat = default;
                        bool hasJockeyClass = false;
                        if (jockeyId.HasValue)
                        {
                            jockeyClassKey = (jockeyId.Value, classVal);
                            if (!_jockeyClassStats.TryGetValue(jockeyClassKey, out jockeyClassStat))
                                jockeyClassStat = (0, 0, 0f, 0f);
                            row["JockeyClassWinRate"] = _trainer.SmoothedWinRate(jockeyClassStat.wins, jockeyClassStat.starts);
                            row["JockeyClassAvgNorm"] = jockeyClassStat.starts > 0
                                                ? jockeyClassStat.sumNorm / jockeyClassStat.starts
                                                : 0f;
                            row["LastJockeyClassNormPos"] = jockeyClassStat.lastNorm;
                            hasJockeyClass = true;
                        }
                        else
                        {
                            row["JockeyClassWinRate"] = 0f;
                            row["JockeyClassAvgNorm"] = 0f;
                            row["LastJockeyClassNormPos"] = 0f;
                        }
                        if (!_horseHistory.TryGetValue(horseId, out var history))
                        {
                            history = new List<HistoryEntry>();
                            _horseHistory[horseId] = history;
                        }
                        row["RatingChangeFromLast"] = history.Count > 0 ? rating - history[^1].Rating : 0f;
                        row["WeightChangeFromLast"] = history.Count > 0 ? weight - history[^1].Weight : 0f;
                        row["AgeProgression"] = history.Count > 0 ? age - history[^1].Age : 0f;
                        row["ClassChangeFromLast"] = history.Count > 0 ? classVal - history[^1].RaceClass : 0;
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
                        if (speedCount > 0)
                        {
                            var recentSpeeds = history
                                .GetRange(history.Count - speedCount, speedCount)
                                .Select(h => h.Speed)
                                .ToList();
                            float speedMean = recentSpeeds.Average();
                            float variance = 0f;
                            foreach (var s in recentSpeeds)
                            {
                                float diff = s - speedMean;
                                variance += diff * diff;
                            }
                            row["SpeedStdDev"] = (float)Math.Sqrt(variance / recentSpeeds.Count);
                            if (speedCount > 1)
                            {
                                float xMean = (speedCount - 1) / 2f;
                                float num = 0f, den = 0f;
                                for (int j = 0; j < recentSpeeds.Count; j++)
                                {
                                    float x = j;
                                    float y = recentSpeeds[j];
                                    num += (x - xMean) * (y - speedMean);
                                    den += (x - xMean) * (x - xMean);
                                }
                                row["SpeedSlope"] = den != 0f ? num / den : 0f;
                            }
                            else
                            {
                                row["SpeedSlope"] = 0f;
                            }
                        }
                        else
                        {
                            row["SpeedSlope"] = 0f;
                            row["SpeedStdDev"] = 0f;
                        }
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
                                row[$"AvgSpeedLast{window}"] = recent.Sum(h => h.Speed) / count;
                                row[$"AvgSpeedDiffLast{window}"] = recent.Sum(h => h.SpeedDiff) / count;
                                row[$"AvgRatingLast{window}"] = recent.Sum(h => h.Rating) / count;

                            }
                            else
                            {
                                recent = new();
                                row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                row[$"AvgNormPosLast{window}"] = 0f;
                                row[$"AvgSpeedLast{window}"] = 0f;
                                row[$"AvgSpeedDiffLast{window}"] = 0f;
                                row[$"AvgRatingLast{window}"] = 0f;

                            }
                        }
                        // Going performance
                        string going = row["Going"] as string ?? "Unknown";
                        if (!_goingStats.TryGetValue(horseId, out var gDict))
                        {
                            gDict = new();
                            _goingStats[horseId] = gDict;
                        }
                        if (!gDict.TryGetValue(going, out var gStats))
                            gStats = (0, 0, 0f, 0f);
                        row["GoingWinRate"] = _trainer.SmoothedWinRate(gStats.wins, gStats.starts);
                        row["GoingAvgNorm"] = gStats.starts > 0 ? gStats.sumNorm / gStats.starts : 0f;
                        row["LastGoingNormPos"] = gStats.lastNorm;
                        var goingFeatureKey = going.Replace(" ", "");
                        row[$"LayoffNormalized_{goingFeatureKey}"] = (float)row["LayoffNormalized"];
                        string surface = row["Surface"] as string ?? "Unknown";
                        if (!_surfaceStats.TryGetValue(horseId, out var sDict))
                        {
                            sDict = new();
                            _surfaceStats[horseId] = sDict;
                        }
                        if (!sDict.TryGetValue(surface, out var sStats))
                            sStats = (0, 0, 0f, 0f);
                        row["SurfaceWinRate"] = _trainer.SmoothedWinRate(sStats.wins, sStats.starts);
                        row["SurfaceAvgNorm"] = sStats.starts > 0 ? sStats.sumNorm / sStats.starts : 0f;
                        row["LastSurfaceNormPos"] = sStats.lastNorm;
                        // Going + Course preference
                        int courseId = row["CourseId"] != null ? Convert.ToInt32(row["CourseId"]) : 0;
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
                            gcDict = new();
                            _goingCourseStats[horseId] = gcDict;
                        }
                        if (!gcDict.TryGetValue(gcKey, out var gcStats))
                            gcStats = (0, 0, 0f, 0f);
                        row["GoingCourseWinRate"] = _trainer.SmoothedWinRate(gcStats.wins, gcStats.starts);
                        row["GoingCourseAvgNorm"] = gcStats.starts > 0 ? gcStats.sumNorm / gcStats.starts : 0f;
                        row["LastGoingCourseNormPos"] = gcStats.lastNorm;
                        // Age restriction context
                        string ageRes = row["AgeRestriction"] as string ?? "Unknown";
                        if (!_ageStats.TryGetValue(horseId, out var aDict))
                        {
                            aDict = new();
                            _ageStats[horseId] = aDict;
                        }
                        if (!aDict.TryGetValue(ageRes, out var aStats))
                            aStats = (0, 0, 0f, 0f);
                        row["AgeRestrictionWinRate"] = _trainer.SmoothedWinRate(aStats.wins, aStats.starts);
                        row["LastAgeRestrictionNormPos"] = aStats.lastNorm;

                        // Distance specialization
                        int distanceYards = row["DistanceYards"] != null ? Convert.ToInt32(row["DistanceYards"]) : 0;
                        int? winningMs = row["WinningTimeMs"] != null ? Convert.ToInt32(row["WinningTimeMs"]) : (int?)null;
                        bool speedMissing = !(winningMs.HasValue && winningMs.Value > 0);
                        float raceSpeed = speedMissing
                            ? 0f
                            : distanceYards / (float)winningMs.Value;
                        row["RaceSpeed"] = raceSpeed;
                        if (!speedMissing)
                        {
                            float beaten = row["DistanceBeatenLengths"] != null ? Convert.ToSingle(row["DistanceBeatenLengths"]) : 0f;
                            float runnerTime = winningMs.Value + beaten * MsPerLength;
                            runnerSpeed = runnerTime > 0f ? distanceYards / runnerTime : 0f;
                            row["RunnerSpeed"] = runnerSpeed;
                        }
                        else
                        {
                            runnerSpeed = 0f;
                            row["RunnerSpeed"] = 0f;
                        }
                        float speedDiff = runnerSpeed - raceSpeed;
                        row["SpeedMissing"] = speedMissing;
                        row["SpeedDiff"] = speedDiff;
                        row["SpeedRatio"] = raceSpeed != 0f ? runnerSpeed / raceSpeed : 0f;
                        string bucket = DistanceBucket(distanceYards);
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
                            dDict = new();
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

                        // Trainer statistics
                        if (trainerId.HasValue)
                        {
                            if (!_trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict))
                            {
                                tsDict = new();
                                _trainerSurfaceStats[trainerId.Value] = tsDict;
                            }
                            if (!tsDict.TryGetValue(surface, out var tsStats))
                                tsStats = (0, 0, 0f, 0f);
                            row["TrainerSurfaceWinRate"] = _trainer.SmoothedWinRate(tsStats.wins, tsStats.starts);
                            row["TrainerSurfaceAvgNorm"] = tsStats.starts > 0 ? tsStats.sumNorm / tsStats.starts : 0f;
                            row["LastTrainerSurfaceNormPos"] = tsStats.lastNorm;

                            if (!_trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict))
                            {
                                tgDict = new();
                                _trainerGoingStats[trainerId.Value] = tgDict;
                            }
                            if (!tgDict.TryGetValue(going, out var tgStats))
                                tgStats = (0, 0, 0f, 0f);
                            row["TrainerGoingWinRate"] = _trainer.SmoothedWinRate(tgStats.wins, tgStats.starts);
                            row["TrainerGoingAvgNorm"] = tgStats.starts > 0 ? tgStats.sumNorm / tgStats.starts : 0f;
                            row["LastTrainerGoingNormPos"] = tgStats.lastNorm;

                            if (!_trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict))
                            {
                                tdDict = new();
                                _trainerDistanceStats[trainerId.Value] = tdDict;
                            }
                            if (!tdDict.TryGetValue(bucket, out var tdStats))
                                tdStats = (0, 0, 0f, 0f);
                            row["TrainerDistanceBucketWinRate"] = _trainer.SmoothedWinRate(tdStats.wins, tdStats.starts);
                            row["TrainerDistanceBucketAvgNorm"] = tdStats.starts > 0 ? tdStats.sumNorm / tdStats.starts : 0f;
                            row["LastTrainerDistanceBucketNormPos"] = tdStats.lastNorm;
                            if (!_trainerStats.TryGetValue(trainerId.Value, out var trainerStat))
                                trainerStat = new RollingStat();
                            while (trainerStat.Recent.Count > 0 &&
                                                    (date - trainerStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                                trainerStat.Recent.Dequeue();

                            var tRecent = trainerStat.Recent.ToList();
                            int tCount = Math.Min(tRecent.Count, TrainerJockeyRecentStarts);
                            int tWinsRecent = tRecent.Skip(tRecent.Count - tCount).Count(r => r.win);
                            row[$"TrainerWinRateLast{TrainerJockeyRecentStarts}"] = _trainer.SmoothedWinRate(tWinsRecent, tCount);
                            row["TrainerWinRateRecentDays"] = _trainer.SmoothedWinRate(tRecent.Count(r => r.win), tRecent.Count);
                            row["TrainerWinRate"] = _trainer.SmoothedWinRate(trainerStat.Wins, trainerStat.Starts);

                            trainerStat.Starts++;
                            bool tWin = finish.HasValue && finish.Value == 1;
                            if (tWin) trainerStat.Wins++;
                            trainerStat.Recent.Enqueue((date, tWin));
                            while (trainerStat.Recent.Count > TrainerJockeyRecentStarts)
                                trainerStat.Recent.Dequeue();
                            _trainerStats[trainerId.Value] = trainerStat;
                        }
                        else
                        {
                            row["TrainerWinRate"] = 0f;
                            row[$"TrainerWinRateLast{TrainerJockeyRecentStarts}"] = 0f;
                            row["TrainerWinRateRecentDays"] = 0f;
                            row["TrainerSurfaceWinRate"] = 0f;
                            row["TrainerSurfaceAvgNorm"] = 0f;
                            row["LastTrainerSurfaceNormPos"] = 0f;
                            row["TrainerGoingWinRate"] = 0f;
                            row["TrainerGoingAvgNorm"] = 0f;
                            row["LastTrainerGoingNormPos"] = 0f;
                            row["TrainerDistanceBucketWinRate"] = 0f;
                            row["TrainerDistanceBucketAvgNorm"] = 0f;
                            row["LastTrainerDistanceBucketNormPos"] = 0f;
                        }
                        if (jockeyId.HasValue)
                        {
                            if (!_jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict))
                            {
                                jsDict = new();
                                _jockeySurfaceStats[jockeyId.Value] = jsDict;
                            }
                            if (!jsDict.TryGetValue(surface, out var jsStats))
                                jsStats = (0, 0, 0f, 0f);
                            row["JockeySurfaceWinRate"] = _trainer.SmoothedWinRate(jsStats.wins, jsStats.starts);
                            row["JockeySurfaceAvgNorm"] = jsStats.starts > 0 ? jsStats.sumNorm / jsStats.starts : 0f;
                            row["LastJockeySurfaceNormPos"] = jsStats.lastNorm;

                            if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict))
                            {
                                jgDict = new();
                                _jockeyGoingStats[jockeyId.Value] = jgDict;
                            }
                            if (!jgDict.TryGetValue(going, out var jgStats))
                                jgStats = (0, 0, 0f, 0f);
                            row["JockeyGoingWinRate"] = _trainer.SmoothedWinRate(jgStats.wins, jgStats.starts);
                            row["JockeyGoingAvgNorm"] = jgStats.starts > 0 ? jgStats.sumNorm / jgStats.starts : 0f;
                            row["LastJockeyGoingNormPos"] = jgStats.lastNorm;

                            if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict))
                            {
                                jdDict = new();
                                _jockeyDistanceStats[jockeyId.Value] = jdDict;
                            }
                            if (!jdDict.TryGetValue(bucket, out var jdStats))
                                jdStats = (0, 0, 0f, 0f);
                            row["JockeyDistanceBucketWinRate"] = _trainer.SmoothedWinRate(jdStats.wins, jdStats.starts);
                            row["JockeyDistanceBucketAvgNorm"] = jdStats.starts > 0 ? jdStats.sumNorm / jdStats.starts : 0f;
                            row["LastJockeyDistanceBucketNormPos"] = jdStats.lastNorm;
                            var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                            if (!_jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat))
                                jockeyGoingDistanceStat = (0, 0, 0f, 0f);
                            row["JockeyGoingDistanceWinRate"] = _trainer.SmoothedWinRate(jockeyGoingDistanceStat.wins, jockeyGoingDistanceStat.starts);
                            row["JockeyGoingDistanceAvgNorm"] = jockeyGoingDistanceStat.starts > 0
                                ? jockeyGoingDistanceStat.sumNorm / jockeyGoingDistanceStat.starts
                                : 0f;
                            row["LastJockeyGoingDistanceNormPos"] = jockeyGoingDistanceStat.lastNorm;
                            if (!_jockeyStats.TryGetValue(jockeyId.Value, out var jockeyStat))
                                jockeyStat = new RollingStat();

                            while (jockeyStat.Recent.Count > 0 &&
                                   (date - jockeyStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                                jockeyStat.Recent.Dequeue();

                            var jRecent = jockeyStat.Recent.ToList();
                            int jCount = Math.Min(jRecent.Count, TrainerJockeyRecentStarts);
                            int jWinsRecent = jRecent.Skip(jRecent.Count - jCount).Count(r => r.win);
                            row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = _trainer.SmoothedWinRate(jWinsRecent, jCount);
                            row["JockeyWinRateRecentDays"] = _trainer.SmoothedWinRate(jRecent.Count(r => r.win), jRecent.Count);
                            row["JockeyWinRate"] = _trainer.SmoothedWinRate(jockeyStat.Wins, jockeyStat.Starts);

                            jockeyStat.Starts++;
                            bool jWin = finish.HasValue && finish.Value == 1;
                            if (jWin) jockeyStat.Wins++;
                            jockeyStat.Recent.Enqueue((date, jWin));
                            while (jockeyStat.Recent.Count > TrainerJockeyRecentStarts)
                                jockeyStat.Recent.Dequeue();
                            _jockeyStats[jockeyId.Value] = jockeyStat;
                        }
                        else
                        {
                            row["JockeyWinRate"] = 0f;
                            row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = 0f;
                            row["JockeyWinRateRecentDays"] = 0f;
                            row["JockeySurfaceWinRate"] = 0f;
                            row["JockeySurfaceAvgNorm"] = 0f;
                            row["LastJockeySurfaceNormPos"] = 0f;
                            row["JockeyGoingWinRate"] = 0f;
                            row["JockeyGoingAvgNorm"] = 0f;
                            row["LastJockeyGoingNormPos"] = 0f;
                            row["JockeyDistanceBucketWinRate"] = 0f;
                            row["JockeyDistanceBucketAvgNorm"] = 0f;
                            row["LastJockeyDistanceBucketNormPos"] = 0f;
                            row["JockeyGoingDistanceWinRate"] = 0f;
                            row["JockeyGoingDistanceAvgNorm"] = 0f;
                            row["LastJockeyGoingDistanceNormPos"] = 0f;
                        }
                        if (trainerId.HasValue)
                        {
                            var tcKey = (trainerId.Value, courseId);
                            _trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                            row["TrainerCourseWinRate"] = _trainer.SmoothedWinRate(tcStat.wins, tcStat.starts);
                        }
                        else
                        {
                            row["TrainerCourseWinRate"] = 0f;
                        }
                        if (jockeyId.HasValue)
                        {
                            var jcKey = (jockeyId.Value, courseId);
                            if (!_jockeyCourseStats.TryGetValue(jcKey, out var jcStat))
                            {
                                jcStat = (0, 0);
                            }
                            row["JockeyCourseWinRate"] = _trainer.SmoothedWinRate(jcStat.wins, jcStat.starts);
                        }
                        else
                        {
                            row["JockeyCourseWinRate"] = 0f;
                        }
                        if (trainerId.HasValue && jockeyId.HasValue)
                        {
                            var pairKey = (trainerId.Value, jockeyId.Value);
                            if (!_trainerJockeyStats.TryGetValue(pairKey, out var pairStat))
                                pairStat = (0, 0);
                            row["TrainerJockeyWinRate"] = _trainer.SmoothedWinRate(pairStat.wins, pairStat.starts);
                            var trainerJockeySurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                            if (!_trainerJockeySurfaceStats.TryGetValue(trainerJockeySurfaceKey, out var pairSurfaceStat))
                                pairSurfaceStat = (0, 0);
                            row["TrainerJockeySurfaceWinRate"] =
                                _trainer.SmoothedWinRate(pairSurfaceStat.wins, pairSurfaceStat.starts);
                        }
                        else
                        {
                            row["TrainerJockeyWinRate"] = 0f;
                            row["TrainerJockeySurfaceWinRate"] = 0f;
                            row["TrainerJockeyCourseWinRate"] = 0f;
                        }
                        // Compute normalized finish
                        float normFinish = (finish.HasValue && runnerCount > 1)
                            ? (runnerCount - finish.Value) / (float)(runnerCount - 1)
                            : 0f;
                        history.Add(new HistoryEntry(
                            date,
                            normFinish,
                            finish,
                            going,
                            surface,
                            courseId,
                            bucket,
                            classVal,
                            runnerSpeed,
                            speedDiff,
                            age,
                            finish.HasValue && finish.Value == 1,
                            rating,
                            weight));
                        horseClassStat.starts++;
                        horseClassStat.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) horseClassStat.wins++;
                        horseClassStat.lastNorm = normFinish;
                        _horseClassStats[horseClassKey] = horseClassStat;

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
                        sStats.starts++;
                        sStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) sStats.wins++;
                        sStats.lastNorm = normFinish;
                        sDict[surface] = sStats;
                        // Update going stats after race
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
                        cStats.starts++;
                        cStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) cStats.wins++;
                        cStats.lastNorm = normFinish;
                        cDict[courseId] = cStats;
                        aStats.starts++;
                        aStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) aStats.wins++;
                        aStats.lastNorm = normFinish;
                        aDict[ageRes] = aStats;

                        dStats.starts++;
                        dStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) dStats.wins++;
                        dStats.lastNorm = normFinish;
                        dDict[bucket] = dStats;
                        var updateKey = (horseId, going, bucket);
                        _goingDistanceStats.TryGetValue(updateKey, out var updateStats);
                        updateStats.starts++;
                        updateStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) updateStats.wins++;
                        updateStats.lastNorm = normFinish;
                        _goingDistanceStats[updateKey] = updateStats;
                        drawStat.starts++;
                        if (finish.HasValue && finish.Value == 1) drawStat.wins++;
                        _drawStats[drawKey] = drawStat;
                        baseStat.starts++;
                        if (finish.HasValue && finish.Value == 1) baseStat.wins++;
                        _drawBaselineStats[baseKey] = baseStat;
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
                        if (trainerId.HasValue)
                        {
                            if (!_trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict)) tsDict = new();
                            if (!tsDict.TryGetValue(surface, out var tsStats)) tsStats = (0, 0, 0f, 0f);
                            tsStats.starts++; tsStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) tsStats.wins++;
                            tsStats.lastNorm = normFinish;
                            tsDict[surface] = tsStats;

                            if (!_trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict)) tgDict = new();
                            if (!tgDict.TryGetValue(going, out var tgStats)) tgStats = (0, 0, 0f, 0f);
                            tgStats.starts++; tgStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) tgStats.wins++;
                            tgStats.lastNorm = normFinish;
                            tgDict[going] = tgStats;

                            if (!_trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict)) tdDict = new();
                            if (!tdDict.TryGetValue(bucket, out var tdStats)) tdStats = (0, 0, 0f, 0f);
                            tdStats.starts++; tdStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) tdStats.wins++;
                            tdStats.lastNorm = normFinish;
                            tdDict[bucket] = tdStats;
                            var tcKey = (trainerId.Value, courseId);
                            _trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                            tcStat.starts++;
                            if (finish.HasValue && finish.Value == 1) tcStat.wins++;
                            _trainerCourseStats[tcKey] = tcStat;
                        }
                        if (jockeyId.HasValue)
                        {
                            if (!_jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict)) jsDict = new();
                            if (!jsDict.TryGetValue(surface, out var jsStats)) jsStats = (0, 0, 0f, 0f);
                            jsStats.starts++; jsStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) jsStats.wins++;
                            jsStats.lastNorm = normFinish;
                            jsDict[surface] = jsStats;

                            if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict)) jgDict = new();
                            if (!jgDict.TryGetValue(going, out var jgStats)) jgStats = (0, 0, 0f, 0f);
                            jgStats.starts++; jgStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) jgStats.wins++;
                            jgStats.lastNorm = normFinish;
                            jgDict[going] = jgStats;

                            if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict)) jdDict = new();
                            if (!jdDict.TryGetValue(bucket, out var jdStats)) jdStats = (0, 0, 0f, 0f);
                            jdStats.starts++; jdStats.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) jdStats.wins++;
                            jdStats.lastNorm = normFinish;
                            jdDict[bucket] = jdStats;
                            var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                            _jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat);
                            jockeyGoingDistanceStat.starts++;
                            jockeyGoingDistanceStat.sumNorm += normFinish;
                            if (finish.HasValue && finish.Value == 1) jockeyGoingDistanceStat.wins++;
                            jockeyGoingDistanceStat.lastNorm = normFinish;
                            _jockeyGoingDistanceStats[jockeyGoingDistanceKey] = jockeyGoingDistanceStat;
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
                            var pairSurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                            _trainerJockeySurfaceStats.TryGetValue(pairSurfaceKey, out var pairSurfaceStat);
                            pairSurfaceStat.starts++;
                            if (finish.HasValue && finish.Value == 1) pairSurfaceStat.wins++;
                            _trainerJockeySurfaceStats[pairSurfaceKey] = pairSurfaceStat;
                        }

                    }

                    float raceAvgSpeed = rows
                        .Select(r => r.ContainsKey("AvgSpeedLast5") && r["AvgSpeedLast5"] != null ? Convert.ToSingle(r["AvgSpeedLast5"]) : 0f)
                        .DefaultIfEmpty(0f)
                        .Average();
                    float raceAvgWinRate = rows
                        .Select(r => r.ContainsKey("WinRateLast5") && r["WinRateLast5"] != null ? Convert.ToSingle(r["WinRateLast5"]) : 0f)
                        .DefaultIfEmpty(0f)
                        .Average();

                    foreach (var row in rows)
                    {
                        row["RaceAvgSpeedLast5"] = raceAvgSpeed;
                        row["RaceAvgWinRateLast5"] = raceAvgWinRate;
                        TrimRunnerRow(row);
                    }
                }
            }

            private static string DistanceBucket(int yards)
                => yards < 1760 ? "Sprint" : yards < 2640 ? "Middle" : "Long";

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
        public PreparedDataset PrepareDataset()
        {
    using var conn = new SqlConnection(_connectionString);

    var sql = @"SELECT c.Name AS CourseName,
                               h.Name AS HorseName,
                               j.Name AS JockeyName,
                               t.Name AS TrainerName,
                               r.RaceId,
                               r.CourseId,
                               r.RaceDate,
                               r.ScheduledOff,
                               r.ActualOff,
                               r.Title,
                               r.RaceType,
                               r.Class,
                               r.AgeRestriction,
                               r.Surface,
                               r.Going,
                               r.DistanceYards,
                               r.DistanceText,
                               r.RunnerCount,
                               r.Status,
                               r.WinningTimeMs,
                               rr.HorseId,
                               rr.TrainerId,
                               rr.JockeyId,
                               rr.SaddleclothNumber,
                               rr.Draw,
                               rr.Age,
                               rr.WeightLbs,
                               rr.WeightText,
                               rr.FinishPos,
                               rr.OutcomeCode,
                               rr.DistanceBeatenText,
                               rr.SP_Fraction,
                               rr.SP_Decimal,
                               rr.FavTag,
                               rr.OpeningFraction,
                               rr.TouchedHighFraction,
                               rr.TouchedLowFraction
                        FROM Race r
                        JOIN Course c ON r.CourseId = c.CourseId
                        JOIN RunnerResult rr ON r.RaceId = rr.RaceId
                        LEFT JOIN Horse h ON rr.HorseId = h.HorseId
                        LEFT JOIN Trainer t ON rr.TrainerId = t.TrainerId
                        LEFT JOIN Jockey j ON rr.JockeyId = j.JockeyId
                        ORDER BY r.RaceDate, r.RaceId, rr.RunnerResultId";

    var featureState = new FeatureEngineeringState(this);
    var races = new List<PreparedRace>();
    var rnd = new Random();
    var currentRows = new List<Dictionary<string, object?>>();
    int? currentRaceId = null;

    foreach (var record in conn.Query(sql, commandTimeout: 6000, buffered: false))
    {
        var source = (IDictionary<string, object?>)record;
        var row = new Dictionary<string, object?>();
        foreach (var kvp in source)
        {
            row[kvp.Key] = NormalizeDbValue(kvp.Value);
        }

        int raceId = Convert.ToInt32(row["RaceId"] ?? throw new InvalidOperationException("Missing RaceId."));
        if (currentRaceId.HasValue && raceId != currentRaceId.Value)
        {
            Shuffle(currentRows, rnd);
            featureState.ProcessRace(currentRows);
            races.Add(new PreparedRace(currentRaceId.Value, currentRows));
            currentRows = new List<Dictionary<string, object?>>();
        }

        currentRows.Add(row);
        currentRaceId = raceId;
    }

    if (currentRaceId.HasValue && currentRows.Count > 0)
    {
        Shuffle(currentRows, rnd);
        featureState.ProcessRace(currentRows);
        races.Add(new PreparedRace(currentRaceId.Value, currentRows));
    }

    return new PreparedDataset(races);
}

private static float[] EncodeNumeric(object value, int dim)
        {
            float f = Convert.ToSingle(value);
            if (Math.Abs(f) > 1_000_000f)
                f /= 1_000_000f;
            var arr = new float[dim];
            arr[0] = f;
            return arr;
        }

        private static float[] EncodeString(
            string key,
            string s,
            int dim,
            Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            var vec = new float[dim];
            if (stringMaps.TryGetValue(key, out var map) && map.TryGetValue(s, out var idx))
            {
                vec[idx] = 1f;
            }
            else
            {
                // Reserve the last index for unknown categories
                vec[dim - 1] = 1f;
            }
            return vec;
        }
        
        private sealed class DatasetFeatureMetadata
        {
            public DatasetFeatureMetadata(
                List<string> featureKeys,
                Dictionary<string, int> featureDimensions,
                Dictionary<string, Dictionary<string, int>> stringMaps)
            {
                FeatureKeys = featureKeys;
                FeatureDimensions = featureDimensions;
                StringMaps = stringMaps;
                FeatureCount = featureDimensions.Values.Sum();
            }

            public List<string> FeatureKeys { get; }
            public Dictionary<string, int> FeatureDimensions { get; }
            public Dictionary<string, Dictionary<string, int>> StringMaps { get; }
            public int FeatureCount { get; }
        }

        private DatasetFeatureMetadata BuildFeatureMetadata(
            PreparedDataset prepared,
            List<Dictionary<string, object?>> metadataRows)
        {
            if (prepared is null)
                throw new ArgumentNullException(nameof(prepared));
            if (metadataRows is null)
                throw new ArgumentNullException(nameof(metadataRows));

            var keys = prepared.Rows
                .SelectMany(r => r.Keys)
                .Distinct()
                .ToList();
            keys.Remove("FinishPos");
            keys.Remove("RaceId");
            keys.Remove("HorseId");
            keys.Remove("CourseId");
            keys.Remove("TrainerId");
            keys.Remove("JockeyId");
            keys.Remove("OutcomeCode");
            keys.Remove("DistanceBeatenText");
            keys.Remove("DistanceBeatenLengths");
            keys.Remove("DistanceBeatenKnown");
            keys.Remove("SP_Fraction");
            keys.Remove("SP_Decimal");
            keys.Remove("OpeningFraction");
            keys.Remove("TouchedHighFraction");
            keys.Remove("TouchedLowFraction");
            keys.Remove("HorseName");
            keys.Remove("JockeyName");
            keys.Remove("TrainerName");
            keys.Remove("Title");
            keys.Remove("RaceMonth");
            keys.Remove("RaceDayOfWeek");
            keys.Remove("Season");
            keys.Remove("ActualOff");
            keys.Remove("ScheduledOff");
            keys.Remove("RaceDate");
            keys.Remove("CourseName");
            keys.Remove("DistanceText");
            keys.Remove("Status");
            keys.Remove("WeightText");
            keys.Remove("FavTag");
            keys.Remove("SaddleclothNumber");
            keys.Remove("Purse");

            var featureDims = new Dictionary<string, int>();
            var stringMaps = new Dictionary<string, Dictionary<string, int>>();
            foreach (var k in keys)
            {
                var values = metadataRows
                    .Select(r => r.ContainsKey(k) ? r[k] : null)
                    .ToList();
                bool hasMissing = values.Count == 0 || values.Any(v => v == null);
                var nonNullValues = values.Where(v => v != null).ToList();
                object? sample = nonNullValues.Count > 0
                    ? nonNullValues[0]
                    : prepared.Rows
                        .Select(r => r.ContainsKey(k) ? r[k] : null)
                        .FirstOrDefault(v => v != null);
                if (sample == null)
                {
                    continue;
                }
                var firstValue = nonNullValues.Count > 0 ? nonNullValues[0]! : sample;
                int baseDim;
                if (firstValue is string)
                {
                    var distinct = nonNullValues.Count > 0
                        ? nonNullValues.Cast<string>().Distinct().ToList()
                        : new List<string>();
                    var map = distinct
                        .Select((v, idx) => new { v, idx })
                        .ToDictionary(x => x.v, x => x.idx);
                    map["__unknown__"] = distinct.Count;
                    stringMaps[k] = map;
                    baseDim = map.Count;
                }
                else
                {
                    baseDim = 1;
                }
                if (k == "DaysSinceLastWin" || k == "RacesSinceLastWin")
                {
                    featureDims[k] = baseDim; // presence handled by HasLastWin
                }
                else
                {
                    featureDims[k] = baseDim + (hasMissing ? 1 : 0);
                }
            }
            featureDims["HasLastWin"] = 1;

            featureDims["TimeOfDaySin"] = 1;
            featureDims["TimeOfDayCos"] = 1;
            featureDims["DistanceChangeFromLast"] = 1;
            featureDims["DistanceRatioFromAverage"] = 1;
            featureDims["CareerStarts"] = 1;
            featureDims["LifetimeWinRate"] = 1;
            featureDims["DrawBias"] = 1;
            featureDims["DistanceBeatenLengths"] = 1;
            featureDims["DistanceBeatenKnown"] = 1;
            featureDims["SaddleclothMissing"] = 1;
            featureDims["SaddleclothRelative"] = 1;
            featureDims["SaddleclothDiffFromMean"] = 1;
            featureDims["IsTopWeight"] = 1;
            featureDims["IsBottomWeight"] = 1;
            featureDims["JockeyGoingDistanceWinRate"] = 1;
            featureDims["JockeyGoingDistanceAvgNorm"] = 1;
            featureDims["LastJockeyGoingDistanceNormPos"] = 1;
            featureDims["TrainerJockeyCourseWinRate"] = 1;
            featureDims["ClassWinRate"] = 1;
            featureDims["ClassAvgNorm"] = 1;
            featureDims["LastClassNormPos"] = 1;
            featureDims["TrainerClassWinRate"] = 1;
            featureDims["TrainerClassAvgNorm"] = 1;
            featureDims["LastTrainerClassNormPos"] = 1;
            featureDims["JockeyClassWinRate"] = 1;
            featureDims["JockeyClassAvgNorm"] = 1;
            featureDims["LastJockeyClassNormPos"] = 1;
            foreach (var window in PerformanceWindows)
            {
                featureDims[$"AvgRatingLast{window}"] = 1;
            }
            featureDims["RatingSlope"] = 1;
            var featureKeys = featureDims.Keys.ToList();

            return new DatasetFeatureMetadata(featureKeys, featureDims, stringMaps);
        }

        private static List<RaceExample> EncodeRaces(
            IEnumerable<PreparedRace> races,
            List<string> featureKeys,
            Dictionary<string, int> featureDimensions,
            Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            if (races is null)
                throw new ArgumentNullException(nameof(races));
            if (featureKeys is null)
                throw new ArgumentNullException(nameof(featureKeys));
            if (featureDimensions is null)
                throw new ArgumentNullException(nameof(featureDimensions));
            if (stringMaps is null)
                throw new ArgumentNullException(nameof(stringMaps));

            int featureCount = featureDimensions.Values.Sum();
            var result = new List<RaceExample>();
            foreach (var preparedRace in races)
            {
                var runners = new List<RunnerExample>(preparedRace.Rows.Count);
                foreach (var row in preparedRace.Rows)
                {
                    var features = new float[featureCount];
                    int offset = 0;
                    foreach (var key in featureKeys)
                    {
                        int dim = featureDimensions[key];
                        row.TryGetValue(key, out var value);
                        var vec = EncodeFeature(key, value, dim, stringMaps);
                        Array.Copy(vec, 0, features, offset, dim);
                        offset += dim;
                    }

                    float label = row.TryGetValue("FinishPos", out var f) && f != null && Convert.ToInt32(f) == 1 ? 1f : 0f;
                    runners.Add(new RunnerExample(preparedRace.RaceId, features, label));
                }

                result.Add(new RaceExample(preparedRace.RaceId, runners));
            }

            return result;
        }

        private TrainingDataset BuildTrainingDataset(
            PreparedDataset prepared,
            List<Dictionary<string, object?>> metadataRows,
            HashSet<int> normalizationRaceIds)
        {
            if (prepared is null)
                throw new ArgumentNullException(nameof(prepared));
            if (metadataRows is null)
                throw new ArgumentNullException(nameof(metadataRows));
            if (normalizationRaceIds is null)
                throw new ArgumentNullException(nameof(normalizationRaceIds));

            var metadata = BuildFeatureMetadata(prepared, metadataRows);
            int featureCount = metadata.FeatureCount;
            var races = EncodeRaces(prepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);

            var normalizationExamples = races
                .Where(r => normalizationRaceIds.Contains(r.RaceId))
                .SelectMany(r => r.Runners)
                .ToList();
            var means = new float[featureCount];
            var stdDevs = new float[featureCount];
            if (normalizationExamples.Count > 0)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    double sum = 0;
                    foreach (var example in normalizationExamples)
                    {
                        sum += example.Features[j];
                    }
                    means[j] = (float)(sum / normalizationExamples.Count);

                    double variance = 0;
                    foreach (var example in normalizationExamples)
                    {
                        double diff = example.Features[j] - means[j];
                        variance += diff * diff;
                    }
                    stdDevs[j] = (float)Math.Sqrt(variance / normalizationExamples.Count);
                    if (stdDevs[j] == 0f)
                    {
                        stdDevs[j] = 1f;
                    }
                }
            }
            else
            {
                for (int j = 0; j < featureCount; j++)
                {
                    stdDevs[j] = 1f;
                }
            }

            var normalization = new NormalizationParameters
            {
                Mean = means,
                StdDev = stdDevs
            };
            var mapPath = Path.Combine(AppContext.BaseDirectory, "string_maps.json");
            File.WriteAllText(mapPath, JsonSerializer.Serialize(metadata.StringMaps));
            var normPath = Path.Combine(AppContext.BaseDirectory, "normalization.json");
            File.WriteAllText(normPath, JsonSerializer.Serialize(normalization));
            return new TrainingDataset(races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps, normalization);
        }
        public TrainingDataset LoadTrainingDataset()
        {
            var prepared = PrepareDataset();
            var allRaceIds = prepared.Races
                .Select(r => r.RaceId)
                .ToHashSet();
    return BuildTrainingDataset(prepared, prepared.Rows.ToList(), allRaceIds);
}
        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss, double TrainBrier, double ValidationBrier) Train(MLParameter param, int foldIndex, int foldCount)
        {
            var dataset = LoadTrainingDataset();
            return Train(param, foldIndex, foldCount, dataset);
        }
        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss, double TrainBrier, double ValidationBrier) Train(MLParameter param, PreparedDataset dataset, int foldIndex, int foldCount)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));

            int totalRaces = dataset.Races.Count;
            if (foldCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(foldCount));
            if (foldIndex < 0 || foldIndex >= foldCount)
                throw new ArgumentOutOfRangeException(nameof(foldIndex));

            int foldSize = totalRaces / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalRaces : valStart + foldSize;

            var trainRaceIds = dataset.Races
                .Select((race, idx) => new { race, idx })
                .Where(x => x.idx < valStart || x.idx >= valEnd)
                .Select(x => x.race.RaceId)
                .ToHashSet();

            var trainRows = dataset.Races
                .Where(r => trainRaceIds.Contains(r.RaceId))
                .SelectMany(r => r.Rows)
                .ToList();

            if (trainRows.Count == 0)
            {
                trainRaceIds = dataset.Races
                    .Select(r => r.RaceId)
                    .ToHashSet();
                trainRows = dataset.Rows.ToList();
            }

    var metadataRows = trainRows.Count == 0 ? dataset.Rows.ToList() : trainRows;
    if (!dataset.TryGetEncodingCache(foldIndex, foldCount, out var cacheEntry))
            {
                var metadata = BuildFeatureMetadata(dataset, metadataRows);
                var races = EncodeRaces(dataset.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);
                cacheEntry = dataset.SetEncodingCache(foldIndex, foldCount, races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);
                var mapPath = Path.Combine(AppContext.BaseDirectory, "string_maps.json");
                File.WriteAllText(mapPath, JsonSerializer.Serialize(cacheEntry.StringMaps));
            }

            var normalization = new NormalizationParameters
            {
                Mean = new float[cacheEntry.FeatureCount],
                StdDev = new float[cacheEntry.FeatureCount]
            };

            var trainingDataset = new TrainingDataset(
                cacheEntry.Races,
                cacheEntry.FeatureKeys,
                cacheEntry.FeatureDimensions,
                cacheEntry.StringMaps,
                normalization);
            return Train(param, foldIndex, foldCount, trainingDataset);
        }

            public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss, double TrainBrier, double ValidationBrier) Train(MLParameter param, int foldIndex, int foldCount, TrainingDataset dataset)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));

            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

            int totalRaces = dataset.Races.Count;
            int foldSize = totalRaces / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalRaces : valStart + foldSize;
            var valRaceSet = dataset.Races
                .Skip(valStart)
                .Take(valEnd - valStart)
                .Select(r => r.RaceId)
                .ToHashSet();

            var trainExamples = new List<RunnerExample>();
            var valExamples = new List<RunnerExample>();
            foreach (var race in dataset.Races)
            {
                if (valRaceSet.Contains(race.RaceId))
                {
                    valExamples.AddRange(race.Runners);
                }
                else
                {
                    trainExamples.AddRange(race.Runners);
                }
            }

            int featureCount = dataset.FeatureCount;
            var trainFeatures = trainExamples.Select(r => r.Features).ToList();
            var trainLabels = trainExamples.Select(r => r.Label).ToArray();
            var trainRaceIds = trainExamples.Select(r => r.RaceId).ToArray();

            var valFeatures = valExamples.Select(r => r.Features).ToList();
            var valLabels = valExamples.Select(r => r.Label).ToArray();
            var valRaceIds = valExamples.Select(r => r.RaceId).ToArray();

            var means = dataset.Normalization.Mean;
            if (means.Length != featureCount)
            {
                means = new float[featureCount];
                dataset.Normalization.Mean = means;
            }
            else
            {
                Array.Clear(means, 0, featureCount);
            }

            var stdDevs = dataset.Normalization.StdDev;
            if (stdDevs.Length != featureCount)
            {
                stdDevs = new float[featureCount];
                dataset.Normalization.StdDev = stdDevs;
            }
            else
            {
                Array.Clear(stdDevs, 0, featureCount);
            }

            if (trainFeatures.Count > 0)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    double sum = 0;
                    foreach (var feature in trainFeatures)
                    {
                        sum += feature[j];
                    }
                    means[j] = (float)(sum / trainFeatures.Count);
                }

                for (int j = 0; j < featureCount; j++)
                {
                    double variance = 0;
                    foreach (var feature in trainFeatures)
                    {
                        double diff = feature[j] - means[j];
                        variance += diff * diff;
                    }
                    stdDevs[j] = (float)Math.Sqrt(variance / trainFeatures.Count);
                    if (stdDevs[j] == 0f)
                    {
                        stdDevs[j] = 1f;
                    }
                }
            }
            else
            {
                Array.Clear(means, 0, featureCount);
                for (int j = 0; j < featureCount; j++)
                {
                    stdDevs[j] = 1f;
                }
            }

            void Normalize(IList<float[]> data)
            {
                foreach (var arr in data)
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        arr[i] = (arr[i] - means[i]) / stdDevs[i];
                    }
                }
            }

            Normalize(trainFeatures);
            Normalize(valFeatures);

            static float[,] BuildFeatureMatrix(List<float[]> source, int featureCount)
            {
                var matrix = new float[source.Count, featureCount];
                for (int i = 0; i < source.Count; i++)
                {
                    var row = source[i];
                    for (int j = 0; j < featureCount; j++)
                    {
                        matrix[i, j] = row[j];
                    }
                }
                return matrix;
            }

            static float[,] BuildLabelMatrix(float[] labels)
            {
                var matrix = new float[labels.Length, 1];
                for (int i = 0; i < labels.Length; i++)
                {
                    matrix[i, 0] = labels[i];
                }
                return matrix;
            }

            var trainFeatureTensor = np.array(BuildFeatureMatrix(trainFeatures, featureCount), dtype: tf.float32);
            var trainLabelTensor = np.array(BuildLabelMatrix(trainLabels), dtype: tf.float32);
            var valFeatureTensor = np.array(BuildFeatureMatrix(valFeatures, featureCount), dtype: tf.float32);
            var valLabelTensor = np.array(BuildLabelMatrix(valLabels), dtype: tf.float32);
            var graph = tf.Graph().as_default();

            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "y");
            Tensor layer = x;
            int inputDim = featureCount;
            var hiddenWeightVars = new List<ResourceVariable>();
            var hiddenBiasVars = new List<ResourceVariable>();
            for (int i = 0; i < param.Layers; i++)
            {
                var w = tf.Variable(tf.random.normal((inputDim, param.Units)), name: $"w{i}");
                var b = tf.Variable(tf.zeros(param.Units), name: $"b{i}");
                hiddenWeightVars.Add(w);
                hiddenBiasVars.Add(b);
                layer = tf.nn.relu(tf.matmul(layer, w) + b);
                if (param.Dropout > 0)
                {
                    layer = tf.nn.dropout(layer, rate: (float)param.Dropout);
                }
                inputDim = param.Units;
            }

            var wOut = tf.Variable(tf.random.normal((inputDim, 1)), name: "wOut");
            var bOut = tf.Variable(tf.zeros(1), name: "bOut");
            var logits = tf.matmul(layer, wOut) + bOut;
            var loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: y, logits: logits));
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(loss);

            var prediction = tf.sigmoid(logits);
            var rnd = new Random();
            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());

            var normPath = Path.Combine(AppContext.BaseDirectory, "normalization.json");
            File.WriteAllText(normPath, JsonSerializer.Serialize(dataset.Normalization));

            var trainPreds = new float[trainLabels.Length];
            var valPreds = new float[valLabels.Length];
            var epochPredBuffer = new float[trainLabels.Length];

            double ComputeDatasetMetrics(NDArray featureTensor, NDArray labelTensor, float[] preds, int count)
            {
                if (preds.Length != count)
                    throw new ArgumentException("Prediction buffer size must match example count.", nameof(preds));

                if (count == 0)
                {
                    return 0;
                }

                const int evalBatchSize = 8192;
                double weightedLoss = 0;
                int totalExamples = 0;

                for (int start = 0; start < count; start += evalBatchSize)
                {
                    int batchCount = Math.Min(evalBatchSize, count - start);

                    var featureSlice = featureTensor[new Slice(start, start + batchCount), Slice.All];
                    var labelSlice = labelTensor[new Slice(start, start + batchCount), Slice.All];

                    var results = sess.run(new[] { loss, prediction },
                        new FeedItem(x, featureSlice),
                        new FeedItem(y, labelSlice));

                    var chunkLoss = results[0].ToArray<float>()[0];
                    var chunkPreds = results[1].ToArray<float>();
                    Array.Copy(chunkPreds, 0, preds, start, batchCount);
                    weightedLoss += chunkLoss * batchCount;
                    totalExamples += batchCount;
                }

                return totalExamples > 0 ? weightedLoss / totalExamples : 0;
            }

            void Denormalize(IList<float[]> data)
            {
                foreach (var arr in data)
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        arr[i] = arr[i] * stdDevs[i] + means[i];
                    }
                }
            }

            double ComputeBrier(float[] preds, float[] labels)
            {
                if (preds.Length == 0)
                    return 0;

                double sum = 0;
                for (int i = 0; i < preds.Length; i++)
                {
                    double diff = preds[i] - labels[i];
                    sum += diff * diff;
                }
                return sum / preds.Length;
            }

            double trainLoss = 0;
            double valLoss = 0;
            double trainBrier = 0;
            double valBrier = 0;
            double trainAcc = 0;
            double valAcc = 0;

            try
            {
                for (int epoch = 0; epoch < param.Epochs; epoch++)
                {
                    var indices = new int[trainFeatures.Count];
                    for (int i = 0; i < indices.Length; i++)
                    {
                        indices[i] = i;
                    }
                    for (int i = indices.Length - 1; i > 0; i--)
                    {
                        int j = rnd.Next(i + 1);
                        (indices[i], indices[j]) = (indices[j], indices[i]);
                    }
                    var fullBatchIdx = new int[param.BatchSize];
                    int[]? shortBatchIdx = null;

                    for (int start = 0; start < indices.Length; start += param.BatchSize)
                    {
                        int batchCount = Math.Min(param.BatchSize, indices.Length - start);
                        int[] batchIdx;
                        if (batchCount == param.BatchSize)
                        {
                            Array.Copy(indices, start, fullBatchIdx, 0, batchCount);
                            batchIdx = fullBatchIdx;
                        }
                        else
                        {
                            if (shortBatchIdx == null || shortBatchIdx.Length != batchCount)
                            {
                                shortBatchIdx = new int[batchCount];
                            }
                            Array.Copy(indices, start, shortBatchIdx, 0, batchCount);
                            batchIdx = shortBatchIdx;
                        }
                        var batchX = trainFeatureTensor[batchIdx];
                        var batchY = trainLabelTensor[batchIdx];
                        sess.run(optimizer, new FeedItem(x, batchX), new FeedItem(y, batchY));
                    }

                    var epochLoss = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, epochPredBuffer, trainLabels.Length);
                    var epochAcc = trainLabels.Length > 0 ? ComputeWinnerAccuracy(trainRaceIds, epochPredBuffer, trainLabels) : 0;
                    Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
                }

                trainLoss = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, trainPreds, trainLabels.Length);
                valLoss = ComputeDatasetMetrics(valFeatureTensor, valLabelTensor, valPreds, valLabels.Length);
                trainBrier = ComputeBrier(trainPreds, trainLabels);
                valBrier = ComputeBrier(valPreds, valLabels);

                trainAcc = trainPreds.Length > 0 ? ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels) : 0;
                valAcc = valPreds.Length > 0 ? ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels) : 0;
            }
            finally
            {
                Denormalize(trainFeatures);
                Denormalize(valFeatures);
            }
            var hiddenLayers = new List<LayerWeights>();
            foreach (var (wVar, bVar) in hiddenWeightVars.Zip(hiddenBiasVars, (wVar, bVar) => (wVar, bVar)))
            {
                var weightArray = ToJagged2D(sess.run(wVar));
                var biasArray = sess.run(bVar).ToArray<float>();
                hiddenLayers.Add(new LayerWeights { Weights = weightArray, Bias = biasArray });
            }

            var outputLayer = new LayerWeights
            {
                Weights = ToJagged2D(sess.run(wOut)),
                Bias = sess.run(bOut).ToArray<float>()
            };
            var featureKeys = dataset.FeatureKeys;
            var featureDims = dataset.FeatureDimensions;
            var stringMaps = dataset.StringMaps;

            var model = new TrainedModel
            {
                HiddenLayers = hiddenLayers,
                OutputLayer = outputLayer,
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string>(featureKeys),
                    FeatureDimensions = new Dictionary<string, int>(featureDims),
                    StringMaps = stringMaps.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new Dictionary<string, int>(kvp.Value))
                },
                Normalization = new NormalizationParameters
                {
                    Mean = (float[])means.Clone(),
                    StdDev = (float[])stdDevs.Clone()
                }
            };

            var weightPath = Path.Combine(AppContext.BaseDirectory, "aiweights.json");
            try
            {
                var options = new JsonSerializerOptions { WriteIndented = true };
                File.WriteAllText(weightPath, JsonSerializer.Serialize(model, options));
            }
            catch
            {
                // Failing to persist weights shouldn't abort training; simply swallow
                // any IO issues so training metrics are still returned.
            }
            Console.WriteLine($"Training complete - train brier: {trainBrier:F4} - val brier: {valBrier:F4}");
            return (trainAcc, trainLoss, valAcc, valLoss, trainBrier, valBrier);
            static float[][] ToJagged2D(NDArray array)
            {
                if (array.ndim != 2)
                    throw new InvalidOperationException($"Expected 2-D tensor but received rank {array.ndim}");

                var shape = array.shape;
                int rows = (int)shape[0];
                int cols = (int)shape[1];
                var flat = array.ToArray<float>();
                var result = new float[rows][];
                for (int r = 0; r < rows; r++)
                {
                    var row = new float[cols];
                    Array.Copy(flat, r * cols, row, 0, cols);
                    result[r] = row;
                }
                return result;
            }
        }
    
    }
}