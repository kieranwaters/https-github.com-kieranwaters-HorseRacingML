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
        private const int StringVectorSize = 4;
        private static readonly DateTime BaseDate = new DateTime(2005, 1, 1);
        private const int PastRaceCount = 3;
        private const int HistoryLength = 30;
        // Windows (in races) for which performance metrics will be generated
        private static readonly int[] PerformanceWindows = { 1, 3, 5, 10, 15, 20, 25, 30 };
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
        private static float[] EncodeFeature(
            string key,
            object? value,
            int dim,
            Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            if (value == null)
                return new float[dim];

            return value switch
            {
                // Convert to days relative to a recent base date to avoid huge tick values
                DateTime dt => new[] { (float)(dt - BaseDate).TotalDays },
                // Seconds are a reasonable scale for durations
                TimeSpan ts => new[] { (float)ts.TotalSeconds },
                string s => EncodeString(key, s, dim, stringMaps),
                bool b => new[] { b ? 1f : 0f },
                // Scale down very large numeric values to keep them in a manageable range
                _ => EncodeNumeric(value, dim)
            };
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
                if (row.TryGetValue("DistanceBeatenLengths", out var lenObj) && lenObj != null)
                {
                    row["DistanceBeatenLengths"] = Convert.ToSingle(lenObj);
                }
                else if (row.TryGetValue("DistanceBeatenText", out var txtObj) && txtObj is string txt)
                {
                    row["DistanceBeatenLengths"] = ParseDistanceBeaten(txt) ?? 0f;
                }
                else
                {
                    row["DistanceBeatenLengths"] = 0f;
                }
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
                        float avgDraw = g.Where(r => r["Draw"] != null)
                                         .Select(r => Convert.ToSingle(r["Draw"]))
                                         .DefaultIfEmpty(0f)
                                         .Average();
                        float avgWeight = g.Where(r => r["WeightLbs"] != null)
                                           .Select(r => Convert.ToSingle(r["WeightLbs"]))
                                           .DefaultIfEmpty(0f)
                                           .Average();
                        float avgAge = g.Where(r => r["Age"] != null)
                                         .Select(r => Convert.ToSingle(r["Age"]))
                                         .DefaultIfEmpty(0f)
                                         .Average();
                        return (RunnerCount: cnt, AvgDraw: avgDraw, AvgWeight: avgWeight, AvgAge: avgAge);
                    });

            var ordered = rows
                .OrderBy(r => (DateTime)r["RaceDate"])
                .ThenBy(r => Convert.ToInt32(r["RaceId"]))
                .ToList();

            var horseHistory = new Dictionary<int, List<(DateTime date, float normFinish, short? finish, string going, string surface, int courseId, string bucket, int raceClass, float speed, int age)>>();
            var trainerStats = new Dictionary<int, RollingStat>();
            var jockeyStats = new Dictionary<int, RollingStat>();
            var trainerJockeyStats = new Dictionary<(int trainerId, int jockeyId), (int starts, int wins)>();
            // New dictionaries for going, age restriction, and distance preferences
            var surfaceStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var goingStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var goingCourseStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var courseStats = new Dictionary<int, Dictionary<int, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var ageStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var distanceBucketStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var horseDistanceAll = new Dictionary<int, (double sum, int count)>();
            var horseDistanceWins = new Dictionary<int, (double sum, int count)>();

            static string DistanceBucket(int yards)
                => yards < 1760 ? "Sprint" : yards < 2640 ? "Middle" : "Long";

            foreach (var row in ordered)
            {
                int horseId = Convert.ToInt32(row["HorseId"]);
                DateTime date = (DateTime)row["RaceDate"];
                row["RaceMonth"] = date.Month;
                row["RaceDayOfWeek"] = (int)date.DayOfWeek;
                row["IsWeekend"] = date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday;
                row["Season"] = (date.Month % 12) / 3;
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

                int draw = row["Draw"] != null ? Convert.ToInt32(row["Draw"]) : 0;
                float runnerSpeed = 0f;
                float weight = row["WeightLbs"] != null ? Convert.ToSingle(row["WeightLbs"]) : 0f;
                row["RelativeDraw"] = runnerCount > 0 ? (float)draw / runnerCount : 0f;
                row["WeightDiffFromMean"] = weight - raceStat.AvgWeight;
                int age = row["Age"] != null ? Convert.ToInt32(row["Age"]) : 0;
                row["AgeRelative"] = age - raceStat.AvgAge;
                int classVal = row["Class"] != null ? Convert.ToInt32(row["Class"]) : 0;
                if (!horseHistory.TryGetValue(horseId, out var history))
                {
                    history = new List<(DateTime date, float normFinish, short? finish, string going, string surface, int courseId, string bucket, int raceClass, float speed, int age)>();
                    horseHistory[horseId] = history;
                }
                row["AgeProgression"] = history.Count > 0 ? age - history[^1].age : 0f;
                row["ClassChangeFromLast"] = history.Count > 0 ? classVal - history[^1].raceClass : 0;
                row["DaysSinceLastRace"] = history.Count > 0 ? (float)(date - history[^1].date).TotalDays : 0f;
                row["LastFinishPos"] = history.Count > 0 ? history[^1].finish ?? 0 : 0;
                var daysSinceLast = (float)row["DaysSinceLastRace"];
                row["LayoffShort"] = daysSinceLast < 30f;
                row["LayoffMedium"] = daysSinceLast >= 30f && daysSinceLast <= 90f;
                row["LayoffLong"] = daysSinceLast > 90f;
                row["LayoffNormalized"] = daysSinceLast / TypicalRestDays;
                for (int i = 0; i < PastRaceCount; i++)
                {
                    var key = $"Last{i + 1}NormPos";
                    row[key] = i < history.Count
                        ? history[history.Count - 1 - i].normFinish
                        : 0f;
                }
                int normCount = Math.Min(PastRaceCount, history.Count);
                if (normCount > 1)
                {
                    // Oldest first for regression
                    var recentNorms = history
                        .GetRange(history.Count - normCount, normCount)
                        .Select(h => h.normFinish)
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
                row["RecentImprovement"] =
                    (float)row["Last1NormPos"] - (float)row[$"Last{PastRaceCount}NormPos"];
                foreach (var window in PerformanceWindows)
                {
                    int count = Math.Min(window, history.Count);

                    // Ensure `recent` is available regardless of branch to avoid scope issues.
                    List<(DateTime date, float normFinish, short? finish, string going, string surface, int courseId, string bucket, int raceClass, float speed, int age)> recent;
                    if (count > 0)
                    {
                        recent = history.GetRange(history.Count - count, count);
                        int wins = recent.Count(h => h.finish == 1);
                        row[$"WinRateLast{window}"] = SmoothedWinRate(wins, count);
                        row[$"AvgNormPosLast{window}"] = recent.Sum(h => h.normFinish) / count;
                        row[$"AvgSpeedLast{window}"] = recent.Sum(h => h.speed) / count;
                    }
                    else
                    {
                        recent = new();
                        row[$"WinRateLast{window}"] = SmoothedWinRate(0, 0);
                        row[$"AvgNormPosLast{window}"] = 0f;
                        row[$"AvgSpeedLast{window}"] = 0f;
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
                row["RaceSpeed"] = winningMs.HasValue && winningMs.Value > 0
                    ? distanceYards / (float)winningMs.Value
                    : 0f;
                if (winningMs.HasValue && winningMs.Value > 0)
                {
                    float beaten = row["DistanceBeatenLengths"] != null ? Convert.ToSingle(row["DistanceBeatenLengths"]) : 0f;
                    float runnerTime = winningMs.Value + beaten * MsPerLength;
                    runnerSpeed = runnerTime > 0f ? distanceYards / runnerTime : 0f;
                    row["RunnerSpeed"] = runnerSpeed;
                }
                else
                {
                    row["RunnerSpeed"] = 0f;
                }
                string bucket = DistanceBucket(distanceYards);
                row["DistanceBucket"] = bucket;
                if (!distanceBucketStats.TryGetValue(horseId, out var dDict))
                {
                    dDict = new();
                    distanceBucketStats[horseId] = dDict;
                }
                if (!dDict.TryGetValue(bucket, out var dStats))
                    dStats = (0, 0, 0f, 0f);
                row["DistanceBucketWinRate"] = SmoothedWinRate(dStats.wins, dStats.starts);
                row["LastDistanceBucketNormPos"] = dStats.lastNorm;
                foreach (var window in PerformanceWindows)
                {
                    int count = Math.Min(window, history.Count);
                    if (count > 0)
                    {
                        var recent = history.GetRange(history.Count - count, count);

                        var goingRecent = recent.Where(h => h.going == going).ToList();
                        row[$"GoingWinRateLast{window}"] = SmoothedWinRate(goingRecent.Count(h => h.finish == 1), goingRecent.Count);
                        row[$"GoingAvgNormLast{window}"] = goingRecent.Count > 0
                            ? goingRecent.Sum(h => h.normFinish) / goingRecent.Count
                            : 0f;
                        var surfaceRecent = recent.Where(h => h.surface == surface).ToList();
                        row[$"SurfaceWinRateLast{window}"] = SmoothedWinRate(surfaceRecent.Count(h => h.finish == 1), surfaceRecent.Count);
                        row[$"SurfaceAvgNormLast{window}"] = surfaceRecent.Count > 0
                            ? surfaceRecent.Sum(h => h.normFinish) / surfaceRecent.Count
                            : 0f;
                        var courseRecent = recent.Where(h => h.courseId == courseId).ToList();
                        row[$"CourseWinRateLast{window}"] = SmoothedWinRate(courseRecent.Count(h => h.finish == 1), courseRecent.Count);
                        row[$"CourseAvgNormLast{window}"] = courseRecent.Count > 0
                            ? courseRecent.Sum(h => h.normFinish) / courseRecent.Count
                            : 0f;

                        var bucketRecent = recent.Where(h => h.bucket == bucket).ToList();
                        row[$"DistanceBucketWinRateLast{window}"] = SmoothedWinRate(bucketRecent.Count(h => h.finish == 1), bucketRecent.Count);
                        row[$"DistanceBucketAvgNormLast{window}"] = bucketRecent.Count > 0
                            ? bucketRecent.Sum(h => h.normFinish) / bucketRecent.Count
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
                double pref = winDist.count > 0 ? winDist.sum / winDist.count : (allDist.count > 0 ? allDist.sum / allDist.count : distanceYards);
                row["DistanceFromPreferred"] = (float)Math.Abs(distanceYards - pref);

                // Trainer statistics
                if (trainerId.HasValue)
                {
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
                }
                if (jockeyId.HasValue)
                {
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
                }
                if (trainerId.HasValue && jockeyId.HasValue)
                {
                    var pairKey = (trainerId.Value, jockeyId.Value);
                    if (!trainerJockeyStats.TryGetValue(pairKey, out var pairStat))
                        pairStat = (0, 0);
                    row["TrainerJockeyWinRate"] = SmoothedWinRate(pairStat.wins, pairStat.starts);
                }
                else
                {
                    row["TrainerJockeyWinRate"] = 0f;
                }
                // Compute normalized finish
                float normFinish = (finish.HasValue && runnerCount > 1)
                    ? (runnerCount - finish.Value) / (float)(runnerCount - 1)
                    : 0f;
                history.Add((date, normFinish, finish, going, surface, courseId, bucket, classVal, runnerSpeed, age));
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

                allDist.sum += distanceYards;
                allDist.count++;
                horseDistanceAll[horseId] = allDist;
                if (finish.HasValue && finish.Value == 1)
                {
                    winDist.sum += distanceYards;
                    winDist.count++;
                }
                horseDistanceWins[horseId] = winDist;
                if (trainerId.HasValue && jockeyId.HasValue)
                {
                    var pairKey = (trainerId.Value, jockeyId.Value);
                    trainerJockeyStats.TryGetValue(pairKey, out var pairStat);
                    pairStat.starts++;
                    if (finish.HasValue && finish.Value == 1) pairStat.wins++;
                    trainerJockeyStats[pairKey] = pairStat;
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
            return vec;
        }
        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss) Train(MLParameter param, int foldIndex, int foldCount)
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
            var rows = conn.Query(sql)
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
            keys.Remove("ActualOff");
            keys.Remove("SP_Fraction");
            keys.Remove("SP_Decimal");
            keys.Remove("OpeningFraction");
            keys.Remove("TouchedHighFraction");
            keys.Remove("TouchedLowFraction");
            keys.Remove("HorseName");
            keys.Remove("JockeyName");
            keys.Remove("TrainerName");
            keys.Remove("Title");

            var allRows = trainRows.Concat(valRows).ToList();
            var featureDims = new Dictionary<string, int>();
            var stringMaps = new Dictionary<string, Dictionary<string, int>>();
            foreach (var k in keys)
            {
                var values = allRows
                    .Select(r => r.ContainsKey(k) ? r[k] : null)
                    .Where(v => v != null)
                    .ToList();
                if (values.Count == 0)
                {
                    continue;
                }
                var sample = values[0];
                if (sample is string)
                {
                    var distinct = values.Cast<string>().Distinct().ToList();
                    var map = distinct
                        .Select((v, idx) => new { v, idx })
                        .ToDictionary(x => x.v, x => x.idx);
                    stringMaps[k] = map;
                    featureDims[k] = map.Count;
                }
                else
                {
                    featureDims[k] = 1;
                }
            }
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
            for (int i = 0; i < param.Layers; i++)
            {
                var w = tf.Variable(tf.random.normal((inputDim, param.Units)), name: $"w{i}");
                var b = tf.Variable(tf.zeros(param.Units), name: $"b{i}");
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
                        .reshape(new Shape(indices.Count, featureCount));
                    var batchY = np.array(indices.Select(i => labs[i]).ToArray())
                        .reshape(new Shape(indices.Count, 1));
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
                                         .reshape(new Shape(batchIdx.Count, featureCount));
                    var batchY = np.array(batchIdx.Select(i => trainLabels[i]).ToArray())
                                         .reshape(new Shape(batchIdx.Count, 1));
                    sess.run(optimizer, new FeedItem(x, batchX), new FeedItem(y, batchY));
                }
                var epochLoss = ComputeDatasetMetrics(trainFeatures, trainLabels, trainRaceIds, out var epochPreds);
                var epochAcc = ComputeWinnerAccuracy(trainRaceIds, epochPreds, trainLabels);
                Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
            }

            var trainLoss = ComputeDatasetMetrics(trainFeatures, trainLabels, trainRaceIds, out var trainPreds);
            var valLoss = ComputeDatasetMetrics(valFeatures, valLabels, valRaceIds, out var valPreds);


            double trainAcc = ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels);
            double valAcc = ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels);

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}