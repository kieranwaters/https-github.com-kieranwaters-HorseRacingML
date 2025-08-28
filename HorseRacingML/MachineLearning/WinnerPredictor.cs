using HorseRacingML.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Predicts a race winner from a set of runners. A LightGBM model is trained
    /// once and persisted to disk so that it can be reused for multiple
    /// predictions. When no suitable training data is available the predictor
    /// falls back to a simple heuristic.
    /// </summary>
    public static class WinnerPredictor
    {
        private static readonly MLContext Ml = new MLContext(seed: 1);
        private static PredictionEngine<RunnerFeatures, RunnerPrediction>? _engine;
        private static readonly string ModelPath = Path.Combine(AppContext.BaseDirectory, "winnerModel.zip");
        private static readonly string MetricsPath = Path.Combine(AppContext.BaseDirectory, "winnerModel.metrics.json");

        /// <summary>
        /// Trains a model from historical runner results and saves it to disk.
        /// </summary>
        /// <param name="runners">Historical runner results.</param>
        /// <param name="modelPath">Optional path to save the trained model.</param>
        public static void TrainModel(IEnumerable<RunnerResult> runners, string? modelPath = null)
        {
            if (runners == null)
                throw new ArgumentException("Training data required", nameof(runners));

            var training = runners
                .Where(r => r.FinishPos.HasValue)
                .Select(r => new RunnerFeatures
                {
                    Odds = GetOdds(r),
                    Weight = r.WeightLbs ?? 0,
                    Draw = r.Draw ?? 0,
                    Age = r.Age ?? 0,
                    Label = r.FinishPos == 1
                })
                .ToList();

            if (training.Count == 0)
                return;

            var data = Ml.Data.LoadFromEnumerable(training);
            var pipeline = Ml.Transforms.Concatenate("Features",
                                                     nameof(RunnerFeatures.Odds),
                                                     nameof(RunnerFeatures.Weight),
                                                     nameof(RunnerFeatures.Draw),
                                                     nameof(RunnerFeatures.Age))
                                 .Append(Ml.BinaryClassification.Trainers.LightGbm());

            var split = Ml.Data.TrainTestSplit(data, testFraction: 0.2);
            var model = pipeline.Fit(split.TrainSet);
            var predictions = model.Transform(split.TestSet);
            var metrics = Ml.BinaryClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}  AUC: {metrics.AreaUnderRocCurve:P2}  LogLoss: {metrics.LogLoss:F4}");

            var path = modelPath ?? ModelPath;
            var existing = LoadMetrics(MetricsPath);


            if (IsBetter(metrics, existing))
            {
                Ml.Model.Save(model, data.Schema, path);
                SaveMetrics(metrics, MetricsPath);
                _engine = Ml.Model.CreatePredictionEngine<RunnerFeatures, RunnerPrediction>(model);
            }
            else if (File.Exists(path))
            {
                using var stream = File.OpenRead(path);
                var oldModel = Ml.Model.Load(stream, out _);
                _engine = Ml.Model.CreatePredictionEngine<RunnerFeatures, RunnerPrediction>(oldModel);
            }
            else
            {
                Ml.Model.Save(model, data.Schema, path);
                SaveMetrics(metrics, MetricsPath);
                _engine = Ml.Model.CreatePredictionEngine<RunnerFeatures, RunnerPrediction>(model);
            }
        }

        /// <summary>
        /// Ensures a model is available and refreshes it when the saved model is
        /// older than <paramref name="maxAge"/>.
        /// </summary>
        /// <param name="runners">Runner results used to retrain when required.</param>
        /// <param name="maxAge">Maximum allowed age of the model before retraining.</param>
        public static void RefreshModel(IEnumerable<RunnerResult> runners, TimeSpan? maxAge = null)
        {
            var age = maxAge ?? TimeSpan.FromDays(7);
            var info = new FileInfo(ModelPath);

            bool needTrain = _engine == null || !info.Exists || DateTime.UtcNow - info.LastWriteTimeUtc > age;

            if (needTrain)
            {
                TrainModel(runners, ModelPath);
            }
            else if (_engine == null)
            {
                using var stream = File.OpenRead(ModelPath);
                var model = Ml.Model.Load(stream, out _);
                _engine = Ml.Model.CreatePredictionEngine<RunnerFeatures, RunnerPrediction>(model);
            }
        }

        /// <summary>
        /// Predicts the winner of the specified race using a pre-trained model.
        /// </summary>
        public static RunnerResult PredictWinner(IEnumerable<RunnerResult> runners, int raceId)
        {
            if (runners == null)
                throw new ArgumentException("At least one runner must be supplied", nameof(runners));

            var raceRunners = runners.Where(r => r.RaceId == raceId).ToList();
            if (raceRunners.Count == 0)
                throw new ArgumentException($"No runners found for race {raceId}", nameof(raceId));

            RefreshModel(runners);

            if (_engine != null)
            {
                RunnerResult? best = null;
                float bestProb = float.NegativeInfinity;

                foreach (var r in raceRunners)
                {
                    var input = new RunnerFeatures
                    {
                        Odds = GetOdds(r),
                        Weight = r.WeightLbs ?? 0,
                        Draw = r.Draw ?? 0,
                        Age = r.Age ?? 0
                    };

                    var pred = _engine.Predict(input);
                    if (best == null || pred.Probability > bestProb)
                    {
                        best = r;
                        bestProb = pred.Probability;
                    }
                }

                if (best != null)
                    return best;
            }

            // Fallback heuristic if there was no training data or prediction failed.
            RunnerResult? fallback = null;
            double fallbackScore = double.NegativeInfinity;

            foreach (var r in raceRunners)
            {
                double odds = GetOdds(r);
                double score = 1.0 / odds;

                if (r.WeightLbs.HasValue)
                    score += 1.0 / (1 + r.WeightLbs.Value);
                if (r.Draw.HasValue)
                    score += 1.0 / (1 + r.Draw.Value);
                if (r.Age.HasValue)
                    score += 1.0 / (1 + r.Age.Value);

                if (fallback == null || score > fallbackScore)
                {
                    fallback = r;
                    fallbackScore = score;
                }
            }

            return fallback!;
        }

        private static float GetOdds(RunnerResult r)
        {
            float odds = (float)(r.SP_Decimal ?? 0m);
            return odds > 0 ? odds : 1000f;
        }

        private class RunnerFeatures
        {
            public float Odds { get; set; }
            public float Weight { get; set; }
            public float Draw { get; set; }
            public float Age { get; set; }
            public bool Label { get; set; }
        }
        private class ModelMetrics
        {
            public double Accuracy { get; set; }
            public double AreaUnderRocCurve { get; set; }
            public double LogLoss { get; set; }
        }

        private static ModelMetrics? LoadMetrics(string path)
        {
            if (!File.Exists(path))
                return null;

            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<ModelMetrics>(json);
        }

        private static void SaveMetrics(CalibratedBinaryClassificationMetrics metrics, string path)
        {
            var info = new ModelMetrics
            {
                Accuracy = metrics.Accuracy,
                AreaUnderRocCurve = metrics.AreaUnderRocCurve,
                LogLoss = metrics.LogLoss
            };
            var json = JsonSerializer.Serialize(info);
            File.WriteAllText(path, json);
        }

        private static bool IsBetter(CalibratedBinaryClassificationMetrics metrics, ModelMetrics? old)
        {
            if (old == null)
                return true;

            return metrics.Accuracy >= old.Accuracy &&
                   metrics.AreaUnderRocCurve >= old.AreaUnderRocCurve &&
                    metrics.LogLoss <= old.LogLoss;
        }
        private class RunnerPrediction
        {
            public bool PredictedLabel { get; set; }
            public float Probability { get; set; }
        }
    }
}