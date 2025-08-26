using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using HorseRacingML.Models;
using Dapper;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;

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

        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss) Train(MLParameter param)
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

            var rows = conn.Query(sql).ToList();
            if (rows.Count == 0)
            {
                throw new InvalidOperationException("No training data found.");
            }

            var keys = ((IDictionary<string, object>)rows[0]).Keys.ToList();
            keys.Remove("FinishPos"); // we'll use this as the label

            var featureList = new List<float[]>();
            var labelList = new List<float>();

            foreach (var row in rows)
            {
                var dict = (IDictionary<string, object>)row;
                var features = new float[keys.Count];
                for (int i = 0; i < keys.Count; i++)
                {
                    features[i] = ConvertToFloat(dict[keys[i]]);
                }
                featureList.Add(features);
                labelList.Add(dict.TryGetValue("FinishPos", out var f) && f != null && Convert.ToInt32(f) == 1 ? 1f : 0f);
            }

            var x = np.array(featureList.ToArray());
            var y = np.array(labelList.ToArray()).reshape(featureList.Count, 1);

            int featureCount = x.shape[1];

            // Build a simple sequential model
            var model = keras.Sequential();
            model.add(keras.layers.Dense(units: param.Units, activation: keras.activations.Relu, input_shape: new Shape(featureCount)));
            if (param.Dropout > 0)
                model.add(keras.layers.Dropout((float)param.Dropout));

            for (int i = 1; i < param.Layers; i++)
            {
                model.add(keras.layers.Dense(units: param.Units, activation: keras.activations.Relu));
                if (param.Dropout > 0)
                    model.add(keras.layers.Dropout((float)param.Dropout));
            }

            // Output layer (logits)
            model.add(keras.layers.Dense(units: 1));

            var optimizer = keras.optimizers.Adam((float)param.LearningRate);
            var loss = keras.losses.BinaryCrossentropy(from_logits: true);
            var metric = keras.metrics.BinaryAccuracy();

            model.compile(optimizer: optimizer, loss: loss, metrics: new Tensorflow.Keras.Metrics.IMetricFunc[] { metric });

            // Train the model (use validation_split to get validation metrics)
            var history = model.fit(x: x, y: y,
                                   batch_size: param.BatchSize,
                                   epochs: param.Epochs,
                                   validation_split: 0.2f,
                                   verbose: 0);

            var hist = history.history;
            double trainLoss = ((NDArray)hist["loss"])[-1].AsScalar<double>();
            string accKey = hist.ContainsKey("binary_accuracy") ? "binary_accuracy" : "accuracy";
            double trainAcc = ((NDArray)hist[accKey])[-1].AsScalar<double>();
            double valLoss = ((NDArray)hist["val_loss"])[-1].AsScalar<double>();
            string valAccKey = hist.ContainsKey("val_binary_accuracy") ? "val_binary_accuracy" : "val_accuracy";
            double valAcc = ((NDArray)hist[valAccKey])[-1].AsScalar<double>();

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}