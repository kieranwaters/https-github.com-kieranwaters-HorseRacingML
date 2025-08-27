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
        private static void AddDerivedFeatures(List<Dictionary<string, object>> rows)
        {
            var ordered = rows
                .OrderBy(r => Convert.ToInt32(r["HorseId"]))
                .ThenBy(r => (DateTime)r["RaceDate"])
                .ToList();

            var lastRace = new Dictionary<int, (DateTime date, short? finish)>();

            foreach (var row in ordered)
            {
                int horseId = Convert.ToInt32(row["HorseId"]);
                DateTime date = (DateTime)row["RaceDate"];
                short? finish = row["FinishPos"] != null ? (short?)Convert.ToInt16(row["FinishPos"]) : null;

                if (lastRace.TryGetValue(horseId, out var info))
                {
                    row["DaysSinceLastRace"] = (float)(date - info.date).TotalDays;
                    row["LastFinishPos"] = info.finish ?? 0;
                }
                else
                {
                    row["DaysSinceLastRace"] = 0f;
                    row["LastFinishPos"] = 0;
                }

                lastRace[horseId] = (date, finish);
            }
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

            var rows = conn.Query(sql)
                .Select(r => ((IDictionary<string, object>)r)
                    .ToDictionary(k => k.Key, k => k.Value))
                .ToList();
            if (rows.Count == 0)
            {
                throw new InvalidOperationException("No training data found.");
            }

            var raceGroups = rows.Select((r, idx) => new { raceId = Convert.ToInt32(r["RaceId"]), idx })
                                 .GroupBy(x => x.raceId)
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

            var keys = rows[0].Keys.ToList();
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

            var featureDims = keys.ToDictionary(k => k, k =>
            {
                var sample = trainRows.Concat(valRows).Select(r => r[k]).FirstOrDefault(v => v != null);
                return sample is string ? StringVectorSize : 1;
            });

            int featureCount = featureDims.Values.Sum();

            var trainFeatures = new List<float[]>();
            var trainLabels = new List<float>();
            var trainRaceIds = new List<int>();
            foreach (var row in trainRows)
            {
                var features = new float[featureCount];
                int offset = 0;
                foreach (var key in keys)
                {
                    int dim = featureDims[key];
                    var vec = EncodeFeature(row[key], dim);
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
                foreach (var key in keys)
                {
                            int dim = featureDims[key];
                            var vec = EncodeFeature(row[key], dim);
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
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1), name: "y");
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
            logits = tf.reshape(logits, new Shape(-1));

            var loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: y, logits: logits));
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(loss);

            var prediction = tf.sigmoid(logits);

            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());
            var rnd = new Random();
            for (int epoch = 0; epoch < param.Epochs; epoch++)
            {
                var raceToIndices = trainRaceIds
                    .Select((raceId, idx) => new { raceId, idx })
                    .GroupBy(x => x.raceId)
                    .ToDictionary(g => g.Key, g => g.Select(x => x.idx).ToList());

                var shuffledRaces = raceToIndices.Keys
                    .OrderBy(_ => rnd.Next())
                    .ToList();

                double totalLoss = 0;
                double totalAcc = 0;
                int batchCount = 0;
                var currentBatch = new List<int>();

                void ProcessBatch(List<int> batchIdx)
                {
                    var races = batchIdx.Select(i => new { idx = i, raceId = trainRaceIds[i] })
                                      .GroupBy(g => g.raceId);

                    foreach (var race in races)
                    {
                        var indices = race.Select(g => g.idx).ToList();
                        var batchX = np.array(indices.SelectMany(i => trainFeatures[i]).ToArray())
                            .reshape(new Shape(indices.Count, featureCount));
                        var batchY = np.array(indices.Select(i => trainLabels[i]).ToArray());

                        sess.run(optimizer, new FeedItem(x, batchX), new FeedItem(y, batchY));

                        double raceLoss = sess.run(loss, new FeedItem(x, batchX), new FeedItem(y, batchY)).ToArray<float>()[0];
                        var racePreds = sess.run(prediction, new FeedItem(x, batchX)).ToArray<float>();

                        var raceIds = indices.Select(i => trainRaceIds[i]).ToList();
                        var raceLabels = indices.Select(i => trainLabels[i]).ToList();

                        totalLoss += raceLoss;
                        totalAcc += ComputeWinnerAccuracy(raceIds, racePreds, raceLabels);
                        batchCount++;
                    }
                }
                foreach (var raceId in shuffledRaces)
                {
                    var indices = raceToIndices[raceId];
                    if (currentBatch.Count + indices.Count > param.BatchSize && currentBatch.Count > 0)
                    {
                        ProcessBatch(currentBatch);
                        currentBatch.Clear();
                    }

                    currentBatch.AddRange(indices);

                    if (currentBatch.Count >= param.BatchSize)
                    {
                        ProcessBatch(currentBatch);
                        currentBatch.Clear();
                    }
                }

                if (currentBatch.Count > 0)
                {
                    ProcessBatch(currentBatch);
                }
                double epochLoss = batchCount > 0 ? totalLoss / batchCount : 0;
                double epochAcc = batchCount > 0 ? totalAcc / batchCount : 0;
                Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
            }

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
                    var batchY = np.array(indices.Select(i => labs[i]).ToArray());
                    totLoss += sess.run(loss, new FeedItem(x, batchX), new FeedItem(y, batchY)).ToArray<float>()[0];
                    var p = sess.run(prediction, new FeedItem(x, batchX)).ToArray<float>();
                    for (int j = 0; j < indices.Count; j++) preds[indices[j]] = p[j];
                    cnt++;
                }
                return cnt > 0 ? totLoss / cnt : 0;
            }

            var trainLoss = ComputeDatasetMetrics(trainFeatures, trainLabels, trainRaceIds, out var trainPreds);
            var valLoss = ComputeDatasetMetrics(valFeatures, valLabels, valRaceIds, out var valPreds);


            double trainAcc = ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels);
            double valAcc = ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels);

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}