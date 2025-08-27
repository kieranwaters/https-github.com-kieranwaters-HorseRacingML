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

        private static float ConvertToFloat(object? value)
        {
            if (value == null) return 0f;
            return value switch
            {
                DateTime dt => dt.Ticks,
                TimeSpan ts => (float)ts.TotalSeconds,
                string s => Math.Abs(s.GetHashCode()),
                bool b => b ? 1f : 0f,
                _ => Convert.ToSingle(value)
            };
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

            var featureList = new List<float[]>();
            var labelList = new List<float>();

            foreach (var row in rows)
            {
                var dict = (IDictionary<string, object>)row;
                var features = new float[keys.Count];
                for (int i = 0; i < keys.Count; i++)
                {
                    features[i] = ConvertToFloat(row[keys[i]]);
                }
                featureList.Add(features);
                labelList.Add(row.TryGetValue("FinishPos", out var f) && f != null && Convert.ToInt32(f) == 1 ? 1f : 0f);
            }

            int totalCount = featureList.Count;

            int featureCount = keys.Count;
            int foldSize = totalCount / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalCount : valStart + foldSize;

            var valFeatures = featureList.Skip(valStart).Take(valEnd - valStart).ToList();
            var valLabels = labelList.Skip(valStart).Take(valEnd - valStart).ToList();

            var trainFeatures = featureList.Take(valStart).Concat(featureList.Skip(valEnd)).ToList();
            var trainLabels = labelList.Take(valStart).Concat(labelList.Skip(valEnd)).ToList();

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
            var accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(prediction), y), tf.float32));

            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());

            for (int epoch = 0; epoch < param.Epochs; epoch++)
            {
                sess.run(optimizer, new FeedItem(x, xTrain), new FeedItem(y, yTrain));
                double epochLoss = sess.run(loss, new FeedItem(x, xTrain), new FeedItem(y, yTrain)).ToArray<float>()[0];
                double epochAcc = sess.run(accuracy, new FeedItem(x, xTrain), new FeedItem(y, yTrain)).ToArray<float>()[0];
                Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - acc: {epochAcc:F4}");
            }

            double trainLoss = sess.run(loss, new FeedItem(x, xTrain), new FeedItem(y, yTrain)).ToArray<float>()[0];
            double trainAcc = sess.run(accuracy, new FeedItem(x, xTrain), new FeedItem(y, yTrain)).ToArray<float>()[0];
            double valLoss = sess.run(loss, new FeedItem(x, xVal), new FeedItem(y, yVal)).ToArray<float>()[0];
            double valAcc = sess.run(accuracy, new FeedItem(x, xVal), new FeedItem(y, yVal)).ToArray<float>()[0];

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}