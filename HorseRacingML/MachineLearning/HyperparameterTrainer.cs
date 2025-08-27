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
                var best = group.OrderByDescending(g => preds[g.idx]).First().idx;
                if (labels[best] == 1f) correct++;
            }

            return total == 0 ? 0 : (double)correct / total;
        }
        private const int StringVectorSize = 4;

        private static float[] EncodeFeature(object? value, int dim)
        {
            if (value == null)
                return new float[dim];

            return value switch
            {
                DateTime dt => new[] { (float)dt.Ticks },
                TimeSpan ts => new[] { (float)ts.TotalSeconds },
                string s => EncodeString(s, dim),
                bool b => new[] { b ? 1f : 0f },
                _ => new[] { Convert.ToSingle(value) }
            };
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

            AddDerivedFeatures(rows);

            var keys = rows[0].Keys.ToList();
            keys.Remove("FinishPos"); // we'll use this as the label
            keys.Remove("RaceId");
            keys.Remove("HorseId");
            keys.Remove("CourseId");
            keys.Remove("TrainerId");
            keys.Remove("JockeyId");

            var featureDims = keys.ToDictionary(k => k, k =>
            {
                var sample = rows.Select(r => r[k]).FirstOrDefault(v => v != null);
                return sample is string ? StringVectorSize : 1;
            });

            int featureCount = featureDims.Values.Sum();

            var featureList = new List<float[]>();
            var labelList = new List<float>();
            var raceIdList = new List<int>();

            foreach (var row in rows)
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
                featureList.Add(features);
                labelList.Add(row.TryGetValue("FinishPos", out var f) && f != null && Convert.ToInt32(f) == 1 ? 1f : 0f);
                raceIdList.Add(Convert.ToInt32(row["RaceId"]));
            }
            var means = new float[featureCount];
            var stdDevs = new float[featureCount];
            if (featureList.Count > 0)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    double sum = 0;
                    foreach (var f in featureList)
                    {
                        sum += f[j];
                    }
                    means[j] = (float)(sum / featureList.Count);

                    double var = 0;
                    foreach (var f in featureList)
                    {
                        double diff = f[j] - means[j];
                        var += diff * diff;
                    }
                    stdDevs[j] = (float)Math.Sqrt(var / featureList.Count);
                    if (stdDevs[j] == 0f) stdDevs[j] = 1f;
                }

                var normParams = new NormalizationParameters { Mean = means, StdDev = stdDevs };
                var normPath = Path.Combine(AppContext.BaseDirectory, "normalization.json");
                File.WriteAllText(normPath, JsonSerializer.Serialize(normParams));
            }

            int totalCount = featureList.Count;
            int foldSize = totalCount / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalCount : valStart + foldSize;

            var valFeatures = featureList.Skip(valStart).Take(valEnd - valStart).ToList();
            var valLabels = labelList.Skip(valStart).Take(valEnd - valStart).ToList();
            var valRaceIds = raceIdList.Skip(valStart).Take(valEnd - valStart).ToList();

            var trainFeatures = featureList.Take(valStart).Concat(featureList.Skip(valEnd)).ToList();
            var trainLabels = labelList.Take(valStart).Concat(labelList.Skip(valEnd)).ToList();
            var trainRaceIds = raceIdList.Take(valStart).Concat(raceIdList.Skip(valEnd)).ToList();
            int posCount = trainLabels.Count(l => l == 1f);
            int negCount = trainLabels.Count - posCount;
            double ratio = negCount == 0 ? 0 : (double)posCount / negCount;
            Console.WriteLine($"Training labels - positive: {posCount}, negative: {negCount}, ratio: {ratio:F4}");
            float posWeight = posCount == 0 ? 1f : (float)negCount / posCount;
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

            int trainCount = trainFeatures.Count;
            int valCount = valFeatures.Count;
            var xTrain = np.array(trainFeatures
                .SelectMany(f => f)
                .ToArray())
                .reshape(new Shape(trainCount, featureCount));
            var yTrain = np.array(trainLabels.ToArray())
                .reshape(new Shape(trainCount, 1));

            var xVal = np.array(valFeatures
                .SelectMany(f => f)
                .ToArray())
                .reshape(new Shape(valCount, featureCount));
            var yVal = np.array(valLabels.ToArray())
                .reshape(new Shape(valCount, 1));

            var graph = tf.Graph().as_default();

            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "y");
            var posWeightTensor = tf.constant(posWeight);
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

            var loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels: y, logits: logits, pos_weight: posWeightTensor));
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(loss);

            var prediction = tf.sigmoid(logits);

            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());
            var rnd = new Random();
            for (int epoch = 0; epoch < param.Epochs; epoch++)
            {
                var indices = Enumerable.Range(0, trainCount)
                    .OrderBy(_ => rnd.Next())
                    .ToArray();
                double totalLoss = 0;
                double totalAcc = 0;
                int batchCount = 0;

                for (int start = 0; start < trainCount; start += param.BatchSize)
                {
                    var batchIdx = indices.Skip(start)
                        .Take(Math.Min(param.BatchSize, trainCount - start))
                        .ToArray();

                    var batchX = np.array(batchIdx.SelectMany(i => trainFeatures[i]).ToArray())
                        .reshape(new Shape(batchIdx.Length, featureCount));
                    var batchY = np.array(batchIdx.Select(i => trainLabels[i]).ToArray())
                        .reshape(new Shape(batchIdx.Length, 1));

                    sess.run(optimizer, new FeedItem(x, batchX), new FeedItem(y, batchY));

                    double batchLoss = sess.run(loss, new FeedItem(x, batchX), new FeedItem(y, batchY)).ToArray<float>()[0];
                    var batchPreds = sess.run(prediction, new FeedItem(x, batchX)).ToArray<float>();

                    var batchRaceIds = batchIdx.Select(i => trainRaceIds[i]).ToList();
                    var batchLabels = batchIdx.Select(i => trainLabels[i]).ToList();

                    totalLoss += batchLoss;
                    totalAcc += ComputeWinnerAccuracy(batchRaceIds, batchPreds, batchLabels);
                    batchCount++;
                }

                double epochLoss = batchCount > 0 ? totalLoss / batchCount : 0;
                double epochAcc = batchCount > 0 ? totalAcc / batchCount : 0;
                Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
            }

            double trainLoss = sess.run(loss, new FeedItem(x, xTrain), new FeedItem(y, yTrain)).ToArray<float>()[0];
            double valLoss = sess.run(loss, new FeedItem(x, xVal), new FeedItem(y, yVal)).ToArray<float>()[0];
            var trainPreds = sess.run(prediction, new FeedItem(x, xTrain)).ToArray<float>();
            var valPreds = sess.run(prediction, new FeedItem(x, xVal)).ToArray<float>();

            double trainAcc = ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels);
            double valAcc = ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels);

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}