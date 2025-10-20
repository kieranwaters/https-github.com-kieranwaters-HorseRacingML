using HorseRacingML.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Globalization;

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
        private static readonly string StringMapPath = Path.Combine(AppContext.BaseDirectory, "string_maps.json");
        private static readonly Dictionary<string, Dictionary<string, int>> StringMaps = LoadStringMaps();
        /// <summary>
        /// Trains a model from historical runner results and saves it to disk.
        /// </summary>
        /// <param name="runners">Historical runner results.</param>
        /// <param name="modelPath">Optional path to save the trained model.</param>
        public static void TrainModel(IEnumerable<RunnerResult> runners, string? modelPath = null)
        {
            if (runners == null)
                throw new ArgumentException("Training data required", nameof(runners));

            var ordered = runners.Where(r => r.FinishPos.HasValue)
                                  .OrderBy(r => r.RaceId)
                                  .ToList();

            var raceStats = ordered
                .GroupBy(r => r.RaceId)
                .ToDictionary(g => g.Key,
                              g => g.Where(r => r.Draw.HasValue)
                                    .Select(r => (float)r.Draw!)
                                    .DefaultIfEmpty(0f)
                                    .Average());

            var history = new Dictionary<int, List<RunnerResult>>();
            var training = new List<RunnerFeatures>();
            foreach (var r in ordered)
            {
                if (!history.TryGetValue(r.HorseId, out var hist))
                {
                    hist = new List<RunnerResult>();
                    history[r.HorseId] = hist;
                }

                int starts = hist.Count;
                int wins = hist.Count(h => h.FinishPos == 1);
                float lastDistance = starts > 0 ? hist.Last().DistanceYards : r.DistanceYards;
                float avgDistance = starts > 0 ? (float)hist.Average(h => (double)h.DistanceYards) : r.DistanceYards;
                float distChange = r.DistanceYards - lastDistance;
                float distRatio = starts > 0 ? r.DistanceYards / avgDistance : 1f;
                var (beaten, beatenKnown) = GetDistanceBeaten(r);
                float drawBias = r.Draw.HasValue && raceStats.TryGetValue(r.RaceId, out var avgDraw)
                    ? r.Draw.Value - avgDraw
                    : 0f;

                training.Add(new RunnerFeatures
                {
                    Odds = r.SP_Decimal.HasValue && r.SP_Decimal.Value > 0m
                        ? (float)r.SP_Decimal.Value
                        : 1000f,
                    Weight = r.WeightLbs ?? 0,
                    WeightMissing = r.WeightLbs.HasValue ? 0f : 1f,
                    Draw = r.Draw ?? 0,
                    DrawMissing = r.Draw.HasValue ? 0f : 1f,
                    Age = r.Age ?? 0,
                    AgeMissing = r.Age.HasValue ? 0f : 1f,
                    Going = EncodeGoing(r.Going),
                    Surface = EncodeSurface(r.Surface),
                    Course = r.CourseId ?? 0,
                    Distance = r.DistanceYards,
                    TimeOfDaySin = 0f,
                    TimeOfDayCos = 0f,
                    DistanceChangeFromLast = distChange,
                    DistanceRatioFromAverage = distRatio,
                    CareerStarts = starts,
                    LifetimeWinRate = starts > 0 ? (float)wins / starts : 0f,
                    DrawBias = drawBias,
                    DistanceBeatenLengths = beaten,
                    DistanceBeatenKnown = beatenKnown ? 1f : 0f,
                    Label = r.FinishPos == 1
                });

                hist.Add(r);
            }

            if (training.Count == 0)
                return;

            var data = Ml.Data.LoadFromEnumerable(training);
            var pipeline = Ml.Transforms.Concatenate("Features",
                                                     nameof(RunnerFeatures.Odds),
                                                     nameof(RunnerFeatures.Weight),
                                                     nameof(RunnerFeatures.WeightMissing),
                                                     nameof(RunnerFeatures.Draw),
                                                     nameof(RunnerFeatures.DrawMissing),
                                                     nameof(RunnerFeatures.Age),
                                                     nameof(RunnerFeatures.AgeMissing),
                                                     nameof(RunnerFeatures.Going),
                                                     nameof(RunnerFeatures.Surface),
                                                     nameof(RunnerFeatures.Course),
                                                     nameof(RunnerFeatures.Distance),
                                                     nameof(RunnerFeatures.TimeOfDaySin),
                                                     nameof(RunnerFeatures.TimeOfDayCos),
                                                     nameof(RunnerFeatures.DistanceChangeFromLast),
                                                     nameof(RunnerFeatures.DistanceRatioFromAverage),
                                                     nameof(RunnerFeatures.CareerStarts),
                                                     nameof(RunnerFeatures.LifetimeWinRate),
                                                     nameof(RunnerFeatures.DrawBias),
                                                     nameof(RunnerFeatures.DistanceBeatenLengths),
                                                     nameof(RunnerFeatures.DistanceBeatenKnown))
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
        private static Dictionary<string, Dictionary<string, int>> LoadStringMaps()
        {
            try
            {
                if (File.Exists(StringMapPath))
                {
                    var json = File.ReadAllText(StringMapPath);
                    return JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, int>>>(json)
                           ?? new Dictionary<string, Dictionary<string, int>>();
                }
            }
            catch
            {
                // Ignore any errors and fall back to empty mappings
            }
            return new Dictionary<string, Dictionary<string, int>>();
        }
        private static float EncodeGoing(string? going)
        {
            if (string.IsNullOrWhiteSpace(going))
                return 0f;
            if (StringMaps.TryGetValue("Going", out var map) && map.TryGetValue(going, out var idx))
                return idx;
            return going.ToLower() switch
            {
                "heavy" => 1f,
                "soft" => 2f,
                "good to soft" => 3f,
                "good" => 4f,
                "good to firm" => 5f,
                "firm" => 6f,
                "standard" => 7f,
                "standard to slow" => 8f,
                "standard to fast" => 9f,
                "yielding" => 10f,
                _ => 0f
            };
        }

        private static float EncodeSurface(string? surface)
        {
            if (string.IsNullOrWhiteSpace(surface))
                return 0f;
            if (StringMaps.TryGetValue("Surface", out var map) && map.TryGetValue(surface, out var idx))
                return idx;
            return surface.ToLower() switch
            {
                "turf" => 1f,
                "dirt" => 2f,
                "all weather" => 3f,
                "synthetic" => 4f,
                _ => 0f
            };
        }
        private class RunnerFeatures
        {
            public float Odds { get; set; }
            public float Weight { get; set; }
            public float WeightMissing { get; set; }
            public float Draw { get; set; }
            public float DrawMissing { get; set; }
            public float Age { get; set; }
            public float AgeMissing { get; set; }
            public float Going { get; set; }
            public float Surface { get; set; }
            public float Course { get; set; }
            public float Distance { get; set; }
            public float TimeOfDaySin { get; set; }
            public float TimeOfDayCos { get; set; }
            public float DistanceChangeFromLast { get; set; }
            public float DistanceRatioFromAverage { get; set; }
            public float CareerStarts { get; set; }
            public float LifetimeWinRate { get; set; }
            public float DrawBias { get; set; }
            public float DistanceBeatenLengths { get; set; }
            public float DistanceBeatenKnown { get; set; }
            public bool Label { get; set; }
        }
        private static (float beaten, bool known) GetDistanceBeaten(RunnerResult r)
        {
            if (r.DistanceBeatenLengths.HasValue)
                return ((float)r.DistanceBeatenLengths.Value, true);

            if (!string.IsNullOrWhiteSpace(r.DistanceBeatenText))
            {
                var parsed = ParseDistanceBeaten(r.DistanceBeatenText);
                if (parsed.HasValue)
                    return (parsed.Value, true);
            }
            return (0f, false);
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
                    total += num;
                else if (part.Contains('/'))
                {
                    var frac = part.Split('/');
                    if (frac.Length == 2 &&
                        double.TryParse(frac[0], out var n) &&
                        double.TryParse(frac[1], out var d) && d != 0)
                        total += n / d;
                }
            }
            return (float)total;
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