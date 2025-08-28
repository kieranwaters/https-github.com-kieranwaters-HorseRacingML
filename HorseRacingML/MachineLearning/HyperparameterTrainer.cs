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
using System.Security.Cryptography;
using System.Text;
using System.IO;
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

        public HyperparameterTrainer(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
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
        private static float[] EncodeFeature(object? value, int dim)
        {
            if (value == null)
                return new float[dim];

            return value switch
            {
                // Convert to days relative to a recent base date to avoid huge tick values
                DateTime dt => new[] { (float)(dt - BaseDate).TotalDays },
                // Seconds are a reasonable scale for durations
                TimeSpan ts => new[] { (float)ts.TotalSeconds },
                string s => EncodeString(s, dim),
                bool b => new[] { b ? 1f : 0f },
                // Scale down very large numeric values to keep them in a manageable range
                _ => EncodeNumeric(value, dim)
            };
        }
        private static void AddDerivedFeatures(List<Dictionary<string, object>> rows)
        {
            var ordered = rows
                .OrderBy(r => (DateTime)r["RaceDate"])
                .ThenBy(r => Convert.ToInt32(r["RaceId"]))
                .ToList();

            var horseHistory = new Dictionary<int, List<(DateTime date, float normFinish, short? finish, string going, int courseId, string bucket)>>();
            var trainerStats = new Dictionary<int, (int starts, int wins)>();
            var jockeyStats = new Dictionary<int, (int starts, int wins)>();

            // New dictionaries for going, age restriction, and distance preferences
            var goingStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
            var goingCourseStats = new Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>>();
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
                short? finish = row["FinishPos"] != null ? (short?)Convert.ToInt16(row["FinishPos"]) : null;
                int runnerCount = row["RunnerCount"] != null ? Convert.ToInt32(row["RunnerCount"]) : 0;

                if (!horseHistory.TryGetValue(horseId, out var history))
                {
                    history = new List<(DateTime, float, short?, string, int, string)>();
                    horseHistory[horseId] = history;
                }

                // Basic history features
                row["DaysSinceLastRace"] = history.Count > 0 ? (float)(date - history[^1].date).TotalDays : 0f;
                row["LastFinishPos"] = history.Count > 0 ? history[^1].finish ?? 0 : 0;
                for (int i = 0; i < PastRaceCount; i++)
                {
                    var key = $"Last{i + 1}NormPos";
                    row[key] = i < history.Count
                        ? history[history.Count - 1 - i].normFinish
                        : 0f;
                }
                foreach (var window in PerformanceWindows)
                {
                    int count = Math.Min(window, history.Count);
                    if (count > 0)
                    {
                        var recent = history.GetRange(history.Count - count, count);
                        row[$"WinRateLast{window}"] = recent.Count(h => h.finish == 1) / (float)count;
                        row[$"AvgNormPosLast{window}"] = recent.Sum(h => h.normFinish) / count;
                    }
                    else
                    {
                        row[$"WinRateLast{window}"] = 0f;
                        row[$"AvgNormPosLast{window}"] = 0f;
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
                row["GoingWinRate"] = gStats.starts > 0 ? (float)gStats.wins / gStats.starts : 0f;
                row["GoingAvgNorm"] = gStats.starts > 0 ? gStats.sumNorm / gStats.starts : 0f;
                row["LastGoingNormPos"] = gStats.lastNorm;

                // Going + Course preference
                int courseId = row["CourseId"] != null ? Convert.ToInt32(row["CourseId"]) : 0;
                string gcKey = going + "_" + courseId;
                if (!goingCourseStats.TryGetValue(horseId, out var gcDict))
                {
                    gcDict = new();
                    goingCourseStats[horseId] = gcDict;
                }
                if (!gcDict.TryGetValue(gcKey, out var gcStats))
                    gcStats = (0, 0, 0f, 0f);
                row["GoingCourseWinRate"] = gcStats.starts > 0 ? (float)gcStats.wins / gcStats.starts : 0f;

                // Age restriction context
                string ageRes = row["AgeRestriction"] as string ?? "Unknown";
                if (!ageStats.TryGetValue(horseId, out var aDict))
                {
                    aDict = new();
                    ageStats[horseId] = aDict;
                }
                if (!aDict.TryGetValue(ageRes, out var aStats))
                    aStats = (0, 0, 0f, 0f);
                row["AgeRestrictionWinRate"] = aStats.starts > 0 ? (float)aStats.wins / aStats.starts : 0f;
                row["LastAgeRestrictionNormPos"] = aStats.lastNorm;

                // Distance specialization
                int distanceYards = row["DistanceYards"] != null ? Convert.ToInt32(row["DistanceYards"]) : 0;
                string bucket = DistanceBucket(distanceYards);
                row["DistanceBucket"] = bucket;
                if (!distanceBucketStats.TryGetValue(horseId, out var dDict))
                {
                    dDict = new();
                    distanceBucketStats[horseId] = dDict;
                }
                if (!dDict.TryGetValue(bucket, out var dStats))
                    dStats = (0, 0, 0f, 0f);
                row["DistanceBucketWinRate"] = dStats.starts > 0 ? (float)dStats.wins / dStats.starts : 0f;
                row["LastDistanceBucketNormPos"] = dStats.lastNorm;
                foreach (var window in PerformanceWindows)
                {
                    int count = Math.Min(window, history.Count);
                    if (count > 0)
                    {
                        var recent = history.GetRange(history.Count - count, count);

                        var goingRecent = recent.Where(h => h.going == going).ToList();
                        row[$"GoingWinRateLast{window}"] = goingRecent.Count > 0
                            ? goingRecent.Count(h => h.finish == 1) / (float)goingRecent.Count
                            : 0f;
                        row[$"GoingAvgNormLast{window}"] = goingRecent.Count > 0
                            ? goingRecent.Sum(h => h.normFinish) / goingRecent.Count
                            : 0f;

                        var courseRecent = recent.Where(h => h.courseId == courseId).ToList();
                        row[$"CourseWinRateLast{window}"] = courseRecent.Count > 0
                            ? courseRecent.Count(h => h.finish == 1) / (float)courseRecent.Count
                            : 0f;
                        row[$"CourseAvgNormLast{window}"] = courseRecent.Count > 0
                            ? courseRecent.Sum(h => h.normFinish) / courseRecent.Count
                            : 0f;

                        var bucketRecent = recent.Where(h => h.bucket == bucket).ToList();
                        row[$"DistanceBucketWinRateLast{window}"] = bucketRecent.Count > 0
                            ? bucketRecent.Count(h => h.finish == 1) / (float)bucketRecent.Count
                            : 0f;
                        row[$"DistanceBucketAvgNormLast{window}"] = bucketRecent.Count > 0
                            ? bucketRecent.Sum(h => h.normFinish) / bucketRecent.Count
                            : 0f;
                    }
                    else
                    {
                        row[$"GoingWinRateLast{window}"] = 0f;
                        row[$"GoingAvgNormLast{window}"] = 0f;
                        row[$"CourseWinRateLast{window}"] = 0f;
                        row[$"CourseAvgNormLast{window}"] = 0f;
                        row[$"DistanceBucketWinRateLast{window}"] = 0f;
                        row[$"DistanceBucketAvgNormLast{window}"] = 0f;
                    }
                }
                // Preferred distance deviation
                horseDistanceAll.TryGetValue(horseId, out var allDist);
                horseDistanceWins.TryGetValue(horseId, out var winDist);
                double pref = winDist.count > 0 ? winDist.sum / winDist.count : (allDist.count > 0 ? allDist.sum / allDist.count : distanceYards);
                row["DistanceFromPreferred"] = (float)Math.Abs(distanceYards - pref);

                // Trainer statistics
                if (row.TryGetValue("TrainerId", out var tObj) && tObj != null)
                {
                    int tId = Convert.ToInt32(tObj);
                    if (!trainerStats.TryGetValue(tId, out var stats))
                        stats = (0, 0);
                    row["TrainerWinRate"] = stats.starts > 0 ? (float)stats.wins / stats.starts : 0f;
                    stats.starts++;
                    if (finish.HasValue && finish.Value == 1) stats.wins++;
                    trainerStats[tId] = stats;
                }
                else
                {
                    row["TrainerWinRate"] = 0f;
                }

                // Jockey statistics
                if (row.TryGetValue("JockeyId", out var jObj) && jObj != null)
                {
                    int jId = Convert.ToInt32(jObj);
                    if (!jockeyStats.TryGetValue(jId, out var stats))
                        stats = (0, 0);
                    row["JockeyWinRate"] = stats.starts > 0 ? (float)stats.wins / stats.starts : 0f;
                    stats.starts++;
                    if (finish.HasValue && finish.Value == 1) stats.wins++;
                    jockeyStats[jId] = stats;
                }
                else
                {
                    row["JockeyWinRate"] = 0f;
                }

                // Compute normalized finish
                float normFinish = (finish.HasValue && runnerCount > 1)
                    ? (runnerCount - finish.Value) / (float)(runnerCount - 1)
                    : 0f;
                history.Add((date, normFinish, finish, going, courseId, bucket));
                if (history.Count > HistoryLength)
                    history.RemoveAt(0);

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

        private static float[] EncodeString(string s, int dim)
        {
            var hash = MD5.HashData(Encoding.UTF8.GetBytes(s));
            var vec = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                vec[i] = hash[i] / 255f;
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
                               rr.DistanceBeatenLengths,
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
            keys.Remove("WinningTimeMs");
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
            foreach (var k in keys)
            {
                var sample = allRows.Select(r => r.ContainsKey(k) ? r[k] : null)
                                    .FirstOrDefault(v => v != null);
                if (sample == null)
                {
                    continue;
                }
                featureDims[k] = sample is string ? StringVectorSize : 1;
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
                    var vec = EncodeFeature(value, dim);
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
                    var vec = EncodeFeature(value, dim);
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

            var trainFeatNd = np.array(trainFeatures.ToArray(), dtype: np.float32);
            var trainLabelNd = np.array(trainLabels.ToArray(), dtype: np.float32)
                                  .reshape(new Shape(trainLabels.Count, 1));

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