using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Keras;
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

            int totalCount = featureList.Count;
            int trainCount = (int)(totalCount * 0.8);

            var xTrain = np.array(featureList.Take(trainCount).ToArray());
            var yTrain = np.array(labelList.Take(trainCount).ToArray()).reshape(new Shape(trainCount, 1));
            var xVal = np.array(featureList.Skip(trainCount).ToArray());
            var yVal = np.array(labelList.Skip(trainCount).ToArray()).reshape(new Shape(totalCount - trainCount, 1));

            int featureCount = (int)xTrain.shape[1];

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
            model.compile(optimizer: optimizer,
                          loss: LossesOnly.Bfair_Crossentropy(from_logits: true),
                          metrics: new[] { MetricsCalc.Bfair_Calc() });

            // Train the model
            model.fit(xTrain, yTrain,
                      batch_size: param.BatchSize,
                      epochs: param.Epochs,
                      verbose: 0);

            var trainResults = model.evaluate(xTrain, yTrain, verbose: 0);
            var valResults = model.evaluate(xVal, yVal, verbose: 0);

            double trainLoss = trainResults[0];
            double trainAcc = trainResults[1];
            double valLoss = valResults[0];
            double valAcc = valResults[1];

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}