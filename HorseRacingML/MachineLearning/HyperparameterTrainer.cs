using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using HorseRacingML.Models;
using Dapper;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Globalization;
using System.Text.Json;
using NpShape = Tensorflow.NumPy.Shape;
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

        private void AddDerivedFeatures(List<Dictionary<string, object>> rows)
        {
            foreach (var row in rows)
            {
                bool distanceKnown = false;
                float beatenLengths = 0f;
                if (row.TryGetValue("DistanceBeatenLengths", out var lenObj) && lenObj != null)
                {
                    beatenLengths = Convert.ToSingle(lenObj);
                    distanceKnown = true;
                }
                else if (row.TryGetValue("DistanceBeatenText", out var txtObj) && txtObj is string txt)
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
            }
                // Precompute average draw and weight for each race to allow
                // relative features on a per-runner basis.
                var raceStats = rows
                    .GroupBy(r => Convert.ToInt32(r["RaceId"]))
                    .ToDictionary(
                        g => g.Key,
                        g =>
                        {
                            int cnt = g.Count();
                            var drawValues = g.Where(r => r["Draw"] != null)
                                                                          .Select(r => Convert.ToSingle(r["Draw"]))
                                                                          .ToList();
                            float avgDraw = drawValues.Count > 0 ? drawValues.Average() : 0f;

                            var weightValues = g.Where(r => r["WeightLbs"] != null)
                                                .Select(r => Convert.ToSingle(r["WeightLbs"]))
                                                .ToList();
                            float avgWeight = weightValues.Count > 0 ? weightValues.Average() : 0f;
                            float minWeight = weightValues.Count > 0 ? weightValues.Min() : 0f;
                            float maxWeight = weightValues.Count > 0 ? weightValues.Max() : 0f;

                            var saddleclothValues = g
                                .Where(r => r.TryGetValue("SaddleclothNumber", out var scObj) && scObj != null)
                                .Select(r => Convert.ToSingle(r["SaddleclothNumber"]))
                                .ToList();
                            float avgSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Average() : 0f;
                            float minSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Min() : 0f;
                            float maxSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Max() : 0f;
                            float avgAge = g.Where(r => r["Age"] != null)
                                             .Select(r => Convert.ToSingle(r["Age"]))
                                             .DefaultIfEmpty(0f)
                                             .Average();
                            var ratingValues = g.Where(r => r.TryGetValue("OfficialRating", out var orObj) && orObj != null)
                                               .Select(r => Convert.ToSingle(r["OfficialRating"]))
                                               .ToList();
                            float avgRating = ratingValues.DefaultIfEmpty(0f).Average();
                            float stdRating = 0f;
                            if (ratingValues.Count > 0)
                            {
                                float variance = ratingValues
                                    .Select(r => (r - avgRating) * (r - avgRating))
                                    .Average();
                                stdRating = (float)Math.Sqrt(variance);
                            }
                            // Purse/total prize money is stored once per race. If available
                            // extract it from any runner row.
                            float purse = g.Select(r => r.TryGetValue("Purse", out var pObj) && pObj != null
                                                    ? Convert.ToSingle(pObj)
                                                    : 0f)
                                           .FirstOrDefault();
                            return (RunnerCount: cnt,
                                    AvgDraw: avgDraw,
                                    AvgWeight: avgWeight,
                                     MinWeight: minWeight,
                                    MaxWeight: maxWeight,
                                    AvgAge: avgAge,
                                    AvgRating: avgRating,
                                    StdRating: stdRating,
                                    TotalPurse: purse,
                                    AvgSaddlecloth: avgSaddlecloth,
                                    MinSaddlecloth: minSaddlecloth,
                                    MaxSaddlecloth: maxSaddlecloth,
                                    HasWeightStats: weightValues.Count > 0,
                                    HasSaddleclothStats: saddleclothValues.Count > 0);
                        });

                var ordered = rows
                    .OrderBy(r => (DateTime)r["RaceDate"])
                    .ThenBy(r => Convert.ToInt32(r["RaceId"]))
                    .ToList();

            var horseHistory = new Dictionary<int, List<HistoryEntry>>();
            Dictionary<int, RollingStat> trainerStats = new();
                var jockeyStats = new Dictionary<int, RollingStat>();
            var trainerJockeyStats = new Dictionary<(int trainerId, int jockeyId), (int starts, int wins)>();
            var trainerJockeySurfaceStats =
                new Dictionary<(int trainerId, int jockeyId, string surface), (int starts, int wins)>();
            var trainerCourseStats =
                new Dictionary<(int trainerId, int courseId), (int starts, int wins)>();
            var jockeyCourseStats =
                new Dictionary<(int jockeyId, int courseId), (int starts, int wins)>();
            // New dictionaries for going, age restriction, and distance preferences
            var surfaceStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var goingStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var goingCourseStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var courseStats = new Dictionary<int, Dictionary<int, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var ageStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var distanceBucketStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var horseClassStats = new Dictionary<(int horseId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)>();
            var trainerClassStats = new Dictionary<(int trainerId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)>();
            var jockeyClassStats = new Dictionary<(int jockeyId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)>();
            var goingDistanceStats = new Dictionary<(int horseId, string going, string bucket), (int starts, int wins, float sumNorm, float lastNorm)>();
                var drawStats = new Dictionary<(int courseId, string bucket, int draw), (int starts, int wins)>();
                var drawBaselineStats = new Dictionary<(int courseId, string bucket), (int starts, int wins)>();
                var horseDistanceAll = new Dictionary<int, (double sum, int count)>();
                var horseDistanceWins = new Dictionary<int, (double sum, int count)>();
                var trainerSurfaceStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var horseLastDistance = new Dictionary<int, int>();
                var trainerGoingStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var trainerDistanceStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var jockeySurfaceStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var jockeyGoingStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
                var jockeyDistanceStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var jockeyGoingDistanceStats = new Dictionary<(int jockeyId, string going, string bucket), (int starts, int wins, float sumNorm, float lastNorm)>();
            static string DistanceBucket(int yards)
                    => yards < 1760 ? "Sprint" : yards < 2640 ? "Middle" : "Long";

                foreach (var row in ordered)
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
                    var raceStat = raceStats[raceId];
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
                if (!horseClassStats.TryGetValue(horseClassKey, out var horseClassStat))
                    horseClassStat = (0, 0, 0f, 0f);
                row["ClassWinRate"] = SmoothedWinRate(horseClassStat.wins, horseClassStat.starts);
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
                    if (!trainerClassStats.TryGetValue(trainerClassKey, out trainerClassStat))
                        trainerClassStat = (0, 0, 0f, 0f);
                    row["TrainerClassWinRate"] = SmoothedWinRate(trainerClassStat.wins, trainerClassStat.starts);
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
                    if (!jockeyClassStats.TryGetValue(jockeyClassKey, out jockeyClassStat))
                        jockeyClassStat = (0, 0, 0f, 0f);
                    row["JockeyClassWinRate"] = SmoothedWinRate(jockeyClassStat.wins, jockeyClassStat.starts);
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
                if (!horseHistory.TryGetValue(horseId, out var history))
                    {
                    history = new List<HistoryEntry>();
                    horseHistory[horseId] = history;
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
                    row["LifetimeWinRate"] = SmoothedWinRate(careerWins, careerStarts);
                    foreach (var window in PerformanceWindows)
                    {
                        int count = Math.Min(window, history.Count);

                    // Ensure `recent` is available regardless of branch to avoid scope issues.
                    List<HistoryEntry> recent;
                    if (count > 0)
                        {
                            recent = history.GetRange(history.Count - count, count);
                        int wins = recent.Count(h => h.Finish == 1);
                        row[$"WinRateLast{window}"] = SmoothedWinRate(wins, count);
                        row[$"AvgNormPosLast{window}"] = recent.Sum(h => h.NormFinish) / count;
                        row[$"AvgSpeedLast{window}"] = recent.Sum(h => h.Speed) / count;
                        row[$"AvgSpeedDiffLast{window}"] = recent.Sum(h => h.SpeedDiff) / count;
                        row[$"AvgRatingLast{window}"] = recent.Sum(h => h.Rating) / count;
                    
                }
                        else
                        {
                            recent = new();
                            row[$"WinRateLast{window}"] = SmoothedWinRate(0, 0);
                            row[$"AvgNormPosLast{window}"] = 0f;
                            row[$"AvgSpeedLast{window}"] = 0f;
                            row[$"AvgSpeedDiffLast{window}"] = 0f;
                        row[$"AvgRatingLast{window}"] = 0f;
                    
                }
                    }
                    // Going performance
                    string going = row["Going"] as string ?? "Unknown";
                    if (!goingStats.TryGetValue(horseId, out var gDict))
                    {
                        gDict = new();
                        goingStats[horseId] = gDict;
                    }
                    if (!gDict.TryGetValue(going, out var gStats))
                        gStats = (0, 0, 0f, 0f);
                    row["GoingWinRate"] = SmoothedWinRate(gStats.wins, gStats.starts);
                    row["GoingAvgNorm"] = gStats.starts > 0 ? gStats.sumNorm / gStats.starts : 0f;
                    row["LastGoingNormPos"] = gStats.lastNorm;
                var goingFeatureKey = going.Replace(" ", "");
                row[$"LayoffNormalized_{goingFeatureKey}"] = (float)row["LayoffNormalized"];
                string surface = row["Surface"] as string ?? "Unknown";
                    if (!surfaceStats.TryGetValue(horseId, out var sDict))
                    {
                        sDict = new();
                        surfaceStats[horseId] = sDict;
                    }
                    if (!sDict.TryGetValue(surface, out var sStats))
                        sStats = (0, 0, 0f, 0f);
                    row["SurfaceWinRate"] = SmoothedWinRate(sStats.wins, sStats.starts);
                    row["SurfaceAvgNorm"] = sStats.starts > 0 ? sStats.sumNorm / sStats.starts : 0f;
                    row["LastSurfaceNormPos"] = sStats.lastNorm;
                    // Going + Course preference
                    int courseId = row["CourseId"] != null ? Convert.ToInt32(row["CourseId"]) : 0;
                    if (!courseStats.TryGetValue(horseId, out var cDict))
                    {
                        cDict = new();
                        courseStats[horseId] = cDict;
                    }
                    if (!cDict.TryGetValue(courseId, out var cStats))
                        cStats = (0, 0, 0f, 0f);
                    row["CourseWinRate"] = SmoothedWinRate(cStats.wins, cStats.starts);
                    row["LastCourseNormPos"] = cStats.lastNorm;
                    string gcKey = going + "_" + courseId;
                    if (!goingCourseStats.TryGetValue(horseId, out var gcDict))
                    {
                        gcDict = new();
                        goingCourseStats[horseId] = gcDict;
                    }
                    if (!gcDict.TryGetValue(gcKey, out var gcStats))
                        gcStats = (0, 0, 0f, 0f);
                    row["GoingCourseWinRate"] = SmoothedWinRate(gcStats.wins, gcStats.starts);
                    row["GoingCourseAvgNorm"] = gcStats.starts > 0 ? gcStats.sumNorm / gcStats.starts : 0f;
                    row["LastGoingCourseNormPos"] = gcStats.lastNorm;
                    // Age restriction context
                    string ageRes = row["AgeRestriction"] as string ?? "Unknown";
                    if (!ageStats.TryGetValue(horseId, out var aDict))
                    {
                        aDict = new();
                        ageStats[horseId] = aDict;
                    }
                    if (!aDict.TryGetValue(ageRes, out var aStats))
                        aStats = (0, 0, 0f, 0f);
                    row["AgeRestrictionWinRate"] = SmoothedWinRate(aStats.wins, aStats.starts);
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
                    drawStats.TryGetValue(drawKey, out var drawStat);
                    var baseKey = (courseId, bucket);
                    drawBaselineStats.TryGetValue(baseKey, out var baseStat);
                    row["DrawBias"] = SmoothedWinRate(drawStat.wins, drawStat.starts) -
                                     SmoothedWinRate(baseStat.wins, baseStat.starts);
                    if (!distanceBucketStats.TryGetValue(horseId, out var dDict))
                    {
                        dDict = new();
                        distanceBucketStats[horseId] = dDict;
                    }
                    if (!dDict.TryGetValue(bucket, out var dStats))
                        dStats = (0, 0, 0f, 0f);
                    row["DistanceBucketWinRate"] = SmoothedWinRate(dStats.wins, dStats.starts);
                    row["LastDistanceBucketNormPos"] = dStats.lastNorm;
                    var gdKey = (horseId, going, bucket);
                    if (!goingDistanceStats.TryGetValue(gdKey, out var gdStats))
                        gdStats = (0, 0, 0f, 0f);
                    row["GoingDistanceWinRate"] = SmoothedWinRate(gdStats.wins, gdStats.starts);
                    row["GoingDistanceAvgNorm"] = gdStats.starts > 0 ? gdStats.sumNorm / gdStats.starts : 0f;
                    row["LastGoingDistanceNormPos"] = gdStats.lastNorm;
                    foreach (var window in PerformanceWindows)
                    {
                        int count = Math.Min(window, history.Count);
                    if (count > 0)
                    {
                        var recent = history.GetRange(history.Count - count, count);

                        var goingRecent = recent.Where(h => h.Going == going).ToList();
                        row[$"GoingWinRateLast{window}"] = SmoothedWinRate(goingRecent.Count(h => h.Finish == 1), goingRecent.Count);
                        row[$"GoingAvgNormLast{window}"] = goingRecent.Count > 0
                            ? goingRecent.Sum(h => h.NormFinish) / goingRecent.Count
                            : 0f;
                        var surfaceRecent = recent.Where(h => h.Surface == surface).ToList();
                        row[$"SurfaceWinRateLast{window}"] = SmoothedWinRate(surfaceRecent.Count(h => h.Finish == 1), surfaceRecent.Count);
                        row[$"SurfaceAvgNormLast{window}"] = surfaceRecent.Count > 0
                            ? surfaceRecent.Sum(h => h.NormFinish) / surfaceRecent.Count
                            : 0f;
                        var courseRecent = recent.Where(h => h.CourseId == courseId).ToList();
                        row[$"CourseWinRateLast{window}"] = SmoothedWinRate(courseRecent.Count(h => h.Finish == 1), courseRecent.Count);
                        row[$"CourseAvgNormLast{window}"] = courseRecent.Count > 0
                            ? courseRecent.Sum(h => h.NormFinish) / courseRecent.Count
                            : 0f;

                        var bucketRecent = recent.Where(h => h.Bucket == bucket).ToList();
                        row[$"DistanceBucketWinRateLast{window}"] = SmoothedWinRate(bucketRecent.Count(h => h.Finish == 1), bucketRecent.Count);
                        row[$"DistanceBucketAvgNormLast{window}"] = bucketRecent.Count > 0
                            ? bucketRecent.Sum(h => h.NormFinish) / bucketRecent.Count
                            : 0f;
                    }
                    else
                        {
                            row[$"GoingWinRateLast{window}"] = SmoothedWinRate(0, 0);
                            row[$"GoingAvgNormLast{window}"] = 0f;
                            row[$"SurfaceWinRateLast{window}"] = SmoothedWinRate(0, 0);
                            row[$"SurfaceAvgNormLast{window}"] = 0f;
                            row[$"CourseWinRateLast{window}"] = SmoothedWinRate(0, 0);
                            row[$"CourseAvgNormLast{window}"] = 0f;
                            row[$"DistanceBucketWinRateLast{window}"] = SmoothedWinRate(0, 0);
                            row[$"DistanceBucketAvgNormLast{window}"] = 0f;
                        }
                    }
                    // Preferred distance deviation
                    horseDistanceAll.TryGetValue(horseId, out var allDist);
                    horseDistanceWins.TryGetValue(horseId, out var winDist);
                    double avgDist = allDist.count > 0 ? allDist.sum / allDist.count : distanceYards;
                    row["DistanceRatioFromAverage"] = avgDist > 0 ? distanceYards / (float)avgDist : 1f;
                    if (horseLastDistance.TryGetValue(horseId, out var lastDist))
                        row["DistanceChangeFromLast"] = distanceYards - lastDist;
                    else
                        row["DistanceChangeFromLast"] = 0f;
                    double pref = winDist.count > 0 ? winDist.sum / winDist.count : (allDist.count > 0 ? allDist.sum / allDist.count : distanceYards);
                    row["DistanceFromPreferred"] = (float)Math.Abs(distanceYards - pref);

                    // Trainer statistics
                    if (trainerId.HasValue)
                    {
                        if (!trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict))
                        {
                            tsDict = new();
                            trainerSurfaceStats[trainerId.Value] = tsDict;
                        }
                        if (!tsDict.TryGetValue(surface, out var tsStats))
                            tsStats = (0, 0, 0f, 0f);
                        row["TrainerSurfaceWinRate"] = SmoothedWinRate(tsStats.wins, tsStats.starts);
                        row["TrainerSurfaceAvgNorm"] = tsStats.starts > 0 ? tsStats.sumNorm / tsStats.starts : 0f;
                        row["LastTrainerSurfaceNormPos"] = tsStats.lastNorm;

                        if (!trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict))
                        {
                            tgDict = new();
                            trainerGoingStats[trainerId.Value] = tgDict;
                        }
                        if (!tgDict.TryGetValue(going, out var tgStats))
                            tgStats = (0, 0, 0f, 0f);
                        row["TrainerGoingWinRate"] = SmoothedWinRate(tgStats.wins, tgStats.starts);
                        row["TrainerGoingAvgNorm"] = tgStats.starts > 0 ? tgStats.sumNorm / tgStats.starts : 0f;
                        row["LastTrainerGoingNormPos"] = tgStats.lastNorm;

                        if (!trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict))
                        {
                            tdDict = new();
                            trainerDistanceStats[trainerId.Value] = tdDict;
                        }
                        if (!tdDict.TryGetValue(bucket, out var tdStats))
                            tdStats = (0, 0, 0f, 0f);
                        row["TrainerDistanceBucketWinRate"] = SmoothedWinRate(tdStats.wins, tdStats.starts);
                        row["TrainerDistanceBucketAvgNorm"] = tdStats.starts > 0 ? tdStats.sumNorm / tdStats.starts : 0f;
                        row["LastTrainerDistanceBucketNormPos"] = tdStats.lastNorm;
                        if (!trainerStats.TryGetValue(trainerId.Value, out var trainerStat))
                            trainerStat = new RollingStat();

                        while (trainerStat.Recent.Count > 0 &&
                               (date - trainerStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                            trainerStat.Recent.Dequeue();

                        var tRecent = trainerStat.Recent.ToList();
                        int tCount = Math.Min(tRecent.Count, TrainerJockeyRecentStarts);
                        int tWinsRecent = tRecent.Skip(tRecent.Count - tCount).Count(r => r.win);
                        row[$"TrainerWinRateLast{TrainerJockeyRecentStarts}"] = SmoothedWinRate(tWinsRecent, tCount);
                        row["TrainerWinRateRecentDays"] = SmoothedWinRate(tRecent.Count(r => r.win), tRecent.Count);
                        row["TrainerWinRate"] = SmoothedWinRate(trainerStat.Wins, trainerStat.Starts);

                        trainerStat.Starts++;
                        bool tWin = finish.HasValue && finish.Value == 1;
                        if (tWin) trainerStat.Wins++;
                        trainerStat.Recent.Enqueue((date, tWin));
                        while (trainerStat.Recent.Count > TrainerJockeyRecentStarts)
                            trainerStat.Recent.Dequeue();
                        trainerStats[trainerId.Value] = trainerStat;
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
                        if (!jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict))
                        {
                            jsDict = new();
                            jockeySurfaceStats[jockeyId.Value] = jsDict;
                        }
                        if (!jsDict.TryGetValue(surface, out var jsStats))
                            jsStats = (0, 0, 0f, 0f);
                        row["JockeySurfaceWinRate"] = SmoothedWinRate(jsStats.wins, jsStats.starts);
                        row["JockeySurfaceAvgNorm"] = jsStats.starts > 0 ? jsStats.sumNorm / jsStats.starts : 0f;
                        row["LastJockeySurfaceNormPos"] = jsStats.lastNorm;

                        if (!jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict))
                        {
                            jgDict = new();
                            jockeyGoingStats[jockeyId.Value] = jgDict;
                        }
                        if (!jgDict.TryGetValue(going, out var jgStats))
                            jgStats = (0, 0, 0f, 0f);
                        row["JockeyGoingWinRate"] = SmoothedWinRate(jgStats.wins, jgStats.starts);
                        row["JockeyGoingAvgNorm"] = jgStats.starts > 0 ? jgStats.sumNorm / jgStats.starts : 0f;
                        row["LastJockeyGoingNormPos"] = jgStats.lastNorm;

                        if (!jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict))
                        {
                            jdDict = new();
                            jockeyDistanceStats[jockeyId.Value] = jdDict;
                        }
                        if (!jdDict.TryGetValue(bucket, out var jdStats))
                            jdStats = (0, 0, 0f, 0f);
                        row["JockeyDistanceBucketWinRate"] = SmoothedWinRate(jdStats.wins, jdStats.starts);
                        row["JockeyDistanceBucketAvgNorm"] = jdStats.starts > 0 ? jdStats.sumNorm / jdStats.starts : 0f;
                        row["LastJockeyDistanceBucketNormPos"] = jdStats.lastNorm;
                    var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                    if (!jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat))
                        jockeyGoingDistanceStat = (0, 0, 0f, 0f);
                    row["JockeyGoingDistanceWinRate"] = SmoothedWinRate(jockeyGoingDistanceStat.wins, jockeyGoingDistanceStat.starts);
                    row["JockeyGoingDistanceAvgNorm"] = jockeyGoingDistanceStat.starts > 0
                        ? jockeyGoingDistanceStat.sumNorm / jockeyGoingDistanceStat.starts
                        : 0f;
                    row["LastJockeyGoingDistanceNormPos"] = jockeyGoingDistanceStat.lastNorm;
                    if (!jockeyStats.TryGetValue(jockeyId.Value, out var jockeyStat))
                            jockeyStat = new RollingStat();

                        while (jockeyStat.Recent.Count > 0 &&
                               (date - jockeyStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                            jockeyStat.Recent.Dequeue();

                        var jRecent = jockeyStat.Recent.ToList();
                        int jCount = Math.Min(jRecent.Count, TrainerJockeyRecentStarts);
                        int jWinsRecent = jRecent.Skip(jRecent.Count - jCount).Count(r => r.win);
                        row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = SmoothedWinRate(jWinsRecent, jCount);
                        row["JockeyWinRateRecentDays"] = SmoothedWinRate(jRecent.Count(r => r.win), jRecent.Count);
                        row["JockeySurfaceWinRate"] = 0f;
                        row["JockeySurfaceAvgNorm"] = 0f;
                        row["LastJockeySurfaceNormPos"] = 0f;
                        row["JockeyGoingWinRate"] = 0f;
                        row["JockeyGoingAvgNorm"] = 0f;
                        row["LastJockeyGoingNormPos"] = 0f;
                        row["JockeyDistanceBucketWinRate"] = 0f;
                        row["JockeyDistanceBucketAvgNorm"] = 0f;
                        row["LastJockeyDistanceBucketNormPos"] = 0f;
                        row["JockeyWinRate"] = SmoothedWinRate(jockeyStat.Wins, jockeyStat.Starts);

                        jockeyStat.Starts++;
                        bool jWin = finish.HasValue && finish.Value == 1;
                        if (jWin) jockeyStat.Wins++;
                        jockeyStat.Recent.Enqueue((date, jWin));
                        while (jockeyStat.Recent.Count > TrainerJockeyRecentStarts)
                            jockeyStat.Recent.Dequeue();
                        jockeyStats[jockeyId.Value] = jockeyStat;
                    }
                    else
                    {
                        row["JockeyWinRate"] = 0f;
                        row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = 0f;
                        row["JockeyWinRateRecentDays"] = 0f;
                    row["JockeyGoingDistanceWinRate"] = 0f;
                    row["JockeyGoingDistanceAvgNorm"] = 0f;
                    row["LastJockeyGoingDistanceNormPos"] = 0f;
                }
                    if (trainerId.HasValue)
                    {
                        var tcKey = (trainerId.Value, courseId);
                        trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                        row["TrainerCourseWinRate"] = SmoothedWinRate(tcStat.wins, tcStat.starts);
                    }
                    else
                    {
                        row["TrainerCourseWinRate"] = 0f;
                    }
                    if (jockeyId.HasValue)
                    {
                        var jcKey = (jockeyId.Value, courseId);
                    if (!jockeyCourseStats.TryGetValue(jcKey, out var jcStat))
                    {
                        jcStat = (0, 0);
                    }
                    row["JockeyCourseWinRate"] = SmoothedWinRate(jcStat.wins, jcStat.starts);
                    }
                    else
                    {
                        row["JockeyCourseWinRate"] = 0f;
                    }
                    if (trainerId.HasValue && jockeyId.HasValue)
                    {
                        var pairKey = (trainerId.Value, jockeyId.Value);
                        if (!trainerJockeyStats.TryGetValue(pairKey, out var pairStat))
                            pairStat = (0, 0);
                        row["TrainerJockeyWinRate"] = SmoothedWinRate(pairStat.wins, pairStat.starts);
                    var trainerJockeySurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                    if (!trainerJockeySurfaceStats.TryGetValue(trainerJockeySurfaceKey, out var pairSurfaceStat))
                        pairSurfaceStat = (0, 0);
                    row["TrainerJockeySurfaceWinRate"] =
                        SmoothedWinRate(pairSurfaceStat.wins, pairSurfaceStat.starts);
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
                horseClassStats[horseClassKey] = horseClassStat;

                if (hasTrainerClass)
                {
                    trainerClassStat.starts++;
                    trainerClassStat.sumNorm += normFinish;
                    if (finish.HasValue && finish.Value == 1) trainerClassStat.wins++;
                    trainerClassStat.lastNorm = normFinish;
                    trainerClassStats[trainerClassKey] = trainerClassStat;
                }

                if (hasJockeyClass)
                {
                    jockeyClassStat.starts++;
                    jockeyClassStat.sumNorm += normFinish;
                    if (finish.HasValue && finish.Value == 1) jockeyClassStat.wins++;
                    jockeyClassStat.lastNorm = normFinish;
                    jockeyClassStats[jockeyClassKey] = jockeyClassStat;
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
                    goingDistanceStats.TryGetValue(updateKey, out var updateStats);
                    updateStats.starts++;
                    updateStats.sumNorm += normFinish;
                    if (finish.HasValue && finish.Value == 1) updateStats.wins++;
                    updateStats.lastNorm = normFinish;
                    goingDistanceStats[updateKey] = updateStats;
                    drawStat.starts++;
                    if (finish.HasValue && finish.Value == 1) drawStat.wins++;
                    drawStats[drawKey] = drawStat;
                    baseStat.starts++;
                    if (finish.HasValue && finish.Value == 1) baseStat.wins++;
                    drawBaselineStats[baseKey] = baseStat;
                    allDist.sum += distanceYards;
                    allDist.count++;
                    horseDistanceAll[horseId] = allDist;
                    horseLastDistance[horseId] = distanceYards;
                    if (finish.HasValue && finish.Value == 1)
                    {
                        winDist.sum += distanceYards;
                        winDist.count++;
                    }
                    horseDistanceWins[horseId] = winDist;
                    if (trainerId.HasValue)
                    {
                        if (!trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict)) tsDict = new();
                        if (!tsDict.TryGetValue(surface, out var tsStats)) tsStats = (0, 0, 0f, 0f);
                        tsStats.starts++; tsStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) tsStats.wins++;
                        tsStats.lastNorm = normFinish;
                        tsDict[surface] = tsStats;

                        if (!trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict)) tgDict = new();
                        if (!tgDict.TryGetValue(going, out var tgStats)) tgStats = (0, 0, 0f, 0f);
                        tgStats.starts++; tgStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) tgStats.wins++;
                        tgStats.lastNorm = normFinish;
                        tgDict[going] = tgStats;

                        if (!trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict)) tdDict = new();
                        if (!tdDict.TryGetValue(bucket, out var tdStats)) tdStats = (0, 0, 0f, 0f);
                        tdStats.starts++; tdStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) tdStats.wins++;
                        tdStats.lastNorm = normFinish;
                        tdDict[bucket] = tdStats;
                        var tcKey = (trainerId.Value, courseId);
                        trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                        tcStat.starts++;
                        if (finish.HasValue && finish.Value == 1) tcStat.wins++;
                        trainerCourseStats[tcKey] = tcStat;
                    }
                    if (jockeyId.HasValue)
                    {
                        if (!jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict)) jsDict = new();
                        if (!jsDict.TryGetValue(surface, out var jsStats)) jsStats = (0, 0, 0f, 0f);
                        jsStats.starts++; jsStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) jsStats.wins++;
                        jsStats.lastNorm = normFinish;
                        jsDict[surface] = jsStats;

                        if (!jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict)) jgDict = new();
                        if (!jgDict.TryGetValue(going, out var jgStats)) jgStats = (0, 0, 0f, 0f);
                        jgStats.starts++; jgStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) jgStats.wins++;
                        jgStats.lastNorm = normFinish;
                        jgDict[going] = jgStats;

                        if (!jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict)) jdDict = new();
                        if (!jdDict.TryGetValue(bucket, out var jdStats)) jdStats = (0, 0, 0f, 0f);
                        jdStats.starts++; jdStats.sumNorm += normFinish;
                        if (finish.HasValue && finish.Value == 1) jdStats.wins++;
                        jdStats.lastNorm = normFinish;
                        jdDict[bucket] = jdStats;
                    var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                    jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat);
                    jockeyGoingDistanceStat.starts++;
                    jockeyGoingDistanceStat.sumNorm += normFinish;
                    if (finish.HasValue && finish.Value == 1) jockeyGoingDistanceStat.wins++;
                    jockeyGoingDistanceStat.lastNorm = normFinish;
                    jockeyGoingDistanceStats[jockeyGoingDistanceKey] = jockeyGoingDistanceStat;
                    var jcKey = (jockeyId.Value, courseId);
                    if (!jockeyCourseStats.TryGetValue(jcKey, out var jcStat))
                    {
                        jcStat = (0, 0);
                    }
                    jcStat.starts++;
                    if (finish.HasValue && finish.Value == 1)
                    {
                        jcStat.wins++;
                    }
                    jockeyCourseStats[jcKey] = jcStat;
                    }
                    if (trainerId.HasValue && jockeyId.HasValue)
                    {
                        var pairKey = (trainerId.Value, jockeyId.Value);
                        trainerJockeyStats.TryGetValue(pairKey, out var pairStat);
                        pairStat.starts++;
                    if (finish.HasValue && finish.Value == 1)
                    {
                        pairStat.wins++;
                    }
                    trainerJockeyStats[pairKey] = pairStat;
                    var pairSurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                    trainerJockeySurfaceStats.TryGetValue(pairSurfaceKey, out var pairSurfaceStat);
                    pairSurfaceStat.starts++;
                    if (finish.HasValue && finish.Value == 1) pairSurfaceStat.wins++;
                    trainerJockeySurfaceStats[pairSurfaceKey] = pairSurfaceStat;
                }
            }
                var racePerfStats = rows
                   .GroupBy(r => Convert.ToInt32(r["RaceId"]))
                   .ToDictionary(
                       g => g.Key,
                       g =>
                       {
                           float avgSpeed = g
                               .Select(r => r.ContainsKey("AvgSpeedLast5")
                                   ? Convert.ToSingle(r["AvgSpeedLast5"]) : 0f)
                               .DefaultIfEmpty(0f)
                               .Average();
                           float avgWinRate = g
                               .Select(r => r.ContainsKey("WinRateLast5")
                                   ? Convert.ToSingle(r["WinRateLast5"]) : 0f)
                               .DefaultIfEmpty(0f)
                               .Average();
                           return (AvgSpeed: avgSpeed, AvgWinRate: avgWinRate);
                       });

                foreach (var row in rows)
                {
                    int raceId = Convert.ToInt32(row["RaceId"]);
                    var stats = racePerfStats[raceId];
                    row["RaceAvgSpeedLast5"] = stats.AvgSpeed;
                    row["RaceAvgWinRateLast5"] = stats.AvgWinRate;
                }
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
        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss, double TrainBrier, double ValidationBrier) Train(MLParameter param, int foldIndex, int foldCount)
        {
            // Enable GPU if available
            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

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
                        LEFT JOIN Jockey j ON rr.JockeyId = j.JockeyId";

            var rnd = new Random();
            var rows = conn.Query(sql, commandTimeout: 6000)
               .Select(r => ((IDictionary<string, object>)r)
                    .ToDictionary(k => k.Key, k => k.Value))
                // Randomize runner order within each race to avoid leaking
                // finish position via default row ordering from the database.
                .GroupBy(r => Convert.ToInt32(r["RaceId"]))
                .SelectMany(g => g.OrderBy(_ => rnd.Next()))
                                 .ToList();
            var raceGroups = rows
               .GroupBy(r => Convert.ToInt32(r["RaceId"]))
               .ToList();
            int totalRaces = raceGroups.Count;
            int foldSize = totalRaces / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalRaces : valStart + foldSize;
            var valRaceSet = raceGroups
                .Skip(valStart)
                .Take(valEnd - valStart)
                .Select(g => g.Key)
                .ToHashSet();

            var trainRows = new List<Dictionary<string, object>>();
            var valRows = new List<Dictionary<string, object>>();
            foreach (var row in rows)
            {
                int raceId = Convert.ToInt32(row["RaceId"]);
                if (valRaceSet.Contains(raceId))
                    valRows.Add(row);
                else
                    trainRows.Add(row);
            }

            AddDerivedFeatures(trainRows);
            AddDerivedFeatures(valRows);

            var keys = trainRows.Concat(valRows)
                .SelectMany(r => r.Keys)
                .Distinct()
                .ToList();
            keys.Remove("FinishPos"); // we'll use this as the label
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

            var allRows = trainRows.Concat(valRows).ToList();
            var featureDims = new Dictionary<string, int>();
            var stringMaps = new Dictionary<string, Dictionary<string, int>>();
            foreach (var k in keys)
            {
                var values = trainRows
                    .Select(r => r.ContainsKey(k) ? r[k] : null)
                    .ToList();
                bool hasMissing = values.Any(v => v == null);
                var nonNullValues = values.Where(v => v != null).ToList();
                if (nonNullValues.Count == 0)
                {
                    continue;
                }
                var sample = nonNullValues[0];
                int baseDim;
                if (sample is string)
                {
                    var distinct = nonNullValues.Cast<string>().Distinct().ToList();
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
            var mapPath = Path.Combine(AppContext.BaseDirectory, "string_maps.json");
            File.WriteAllText(mapPath, JsonSerializer.Serialize(stringMaps));
            // Ensure availability flag is included as a feature
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

            int featureCount = featureDims.Values.Sum();

            var trainFeatures = new List<float[]>();
            var trainLabels = new List<float>();
            var trainRaceIds = new List<int>();
            foreach (var row in trainRows)
            {
                var features = new float[featureCount];
                int offset = 0;
                foreach (var key in featureKeys)
                {
                    int dim = featureDims[key];
                    row.TryGetValue(key, out var value);
                    var vec = EncodeFeature(key, value, dim, stringMaps);
                    Array.Copy(vec, 0, features, offset, dim);
                    offset += dim;
                }
                trainFeatures.Add(features);
                trainLabels.Add(row.TryGetValue("FinishPos", out var f) && f != null && Convert.ToInt32(f) == 1 ? 1f : 0f);
                trainRaceIds.Add(Convert.ToInt32(row["RaceId"]));
            }

            var valFeatures = new List<float[]>();
            var valLabels = new List<float>();
            var valRaceIds = new List<int>();
            foreach (var row in valRows)
            {
                var features = new float[featureCount];
                int offset = 0;
                foreach (var key in featureKeys)
                {
                    int dim = featureDims[key];
                    row.TryGetValue(key, out var value);
                    var vec = EncodeFeature(key, value, dim, stringMaps);
                    Array.Copy(vec, 0, features, offset, dim);
                    offset += dim;
                }
                valFeatures.Add(features);
                valLabels.Add(row.TryGetValue("FinishPos", out var f) && f != null && Convert.ToInt32(f) == 1 ? 1f : 0f);
                valRaceIds.Add(Convert.ToInt32(row["RaceId"]));
            }
            var means = new float[featureCount];
            var stdDevs = new float[featureCount];
            if (trainFeatures.Count > 0)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    double sum = 0;
                    foreach (var f in trainFeatures)
                    {
                        sum += f[j];
                    }
                    means[j] = (float)(sum / trainFeatures.Count);

                    double var = 0;
                    foreach (var f in trainFeatures)
                    {
                        double diff = f[j] - means[j];
                        var += diff * diff;
                    }
                    stdDevs[j] = (float)Math.Sqrt(var / trainFeatures.Count);
                    if (stdDevs[j] == 0f) stdDevs[j] = 1f;
                }

                var normParams = new NormalizationParameters { Mean = means, StdDev = stdDevs };
                var normPath = Path.Combine(AppContext.BaseDirectory, "normalization.json");
                File.WriteAllText(normPath, JsonSerializer.Serialize(normParams));
            }
            void Normalize(List<float[]> data)
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

            var graph = tf.Graph().as_default();

            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "y");
            Tensor layer = x;
            int inputDim = featureCount;
            var hiddenWeightVars = new List<VariableV1>();
            var hiddenBiasVars = new List<VariableV1>();
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

            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());
            double ComputeDatasetMetrics(List<float[]> feats, List<float> labs, List<int> races, out float[] preds)
            {
                preds = new float[feats.Count];
                var grouped = races.Select((raceId, idx) => new { raceId, idx })
                                   .GroupBy(x => x.raceId);
                double totLoss = 0;
                int cnt = 0;
                foreach (var grp in grouped)
                {
                    var indices = grp.Select(g => g.idx).ToList();
                    var batchX = np.array(indices.SelectMany(i => feats[i]).ToArray())
                        .reshape(new NpShape(indices.Count, featureCount));
                    var batchY = np.array(indices.Select(i => labs[i]).ToArray())
                        .reshape(new NpShape(indices.Count, 1));
                    totLoss += sess.run(loss, new FeedItem(x, batchX), new FeedItem(y, batchY)).ToArray<float>()[0];
                    var p = sess.run(prediction, new FeedItem(x, batchX)).ToArray<float>();
                    for (int j = 0; j < indices.Count; j++) preds[indices[j]] = p[j];
                    cnt++;
                }
                return cnt > 0 ? totLoss / cnt : 0;
            }
            for (int epoch = 0; epoch < param.Epochs; epoch++)
            {
                var indices = Enumerable.Range(0, trainFeatures.Count)
                                        .OrderBy(_ => rnd.Next())
                                        .ToList();
                for (int start = 0; start < indices.Count; start += param.BatchSize)
                {
                    var batchIdx = indices.Skip(start)
                                          .Take(Math.Min(param.BatchSize, indices.Count - start))
                                          .ToList();
                    var batchX = np.array(batchIdx.SelectMany(i => trainFeatures[i]).ToArray())
                                         .reshape(new NpShape(batchIdx.Count, featureCount));
                    var batchY = np.array(batchIdx.Select(i => trainLabels[i]).ToArray())
                                         .reshape(new NpShape(batchIdx.Count, 1));
                    sess.run(optimizer, new FeedItem(x, batchX), new FeedItem(y, batchY));
                }
                var epochLoss = ComputeDatasetMetrics(trainFeatures, trainLabels, trainRaceIds, out var epochPreds);
                var epochAcc = ComputeWinnerAccuracy(trainRaceIds, epochPreds, trainLabels);
                Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
            }

            var trainLoss = ComputeDatasetMetrics(trainFeatures, trainLabels, trainRaceIds, out var trainPreds);
            var valLoss = ComputeDatasetMetrics(valFeatures, valLabels, valRaceIds, out var valPreds);
            double trainBrier = trainPreds.Zip(trainLabels, (p, l) => Math.Pow(p - l, 2)).Average();
            double valBrier = valPreds.Zip(valLabels, (p, l) => Math.Pow(p - l, 2)).Average();

            double trainAcc = ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels);
            double valAcc = ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels);
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
                int rows = shape[0];
                int cols = shape[1];
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