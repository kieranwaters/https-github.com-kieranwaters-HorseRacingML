using HorseRacingML.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Contexts;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Predicts a race winner from a set of runners.
    /// Historical runner results are used to train a LightGBM model so that
    /// non-linear interactions between odds, weight, draw and age can be
    /// captured.  When no suitable training data is supplied the method falls
    /// back to a simple heuristic.
    /// </summary>
    public static class WinnerPredictor
    {
        /// <param name="runners">Collection of runners which may include multiple races.</param>
        /// <param name="raceId">The specific race to evaluate.</param>
        /// <returns>The <see cref="RunnerResult"/> predicted to win the race.</returns>
        /// <exception cref="ArgumentException">Thrown when no runners are supplied or none match the race.</exception>
        public static RunnerResult PredictWinner(IEnumerable<RunnerResult> runners, int raceId)
        {
            if (runners == null)
                throw new ArgumentException("At least one runner must be supplied", nameof(runners));

            var raceRunners = runners.Where(r => r.RaceId == raceId).ToList();
            if (raceRunners.Count == 0)
                throw new ArgumentException($"No runners found for race {raceId}", nameof(raceId));

            // Build training data from historical races.
            var training = runners
                .Where(r => r.RaceId != raceId && r.FinishPos.HasValue)
                .Select(r => new RunnerFeatures
                {
                    Odds = GetOdds(r),
                    Weight = r.WeightLbs ?? 0,
                    Draw = r.Draw ?? 0,
                    Age = r.Age ?? 0,
                    Label = r.FinishPos == 1
                })
                .ToList();

            if (training.Count > 0)
            {
                var ml = new MLContext(seed: 1);
                var data = ml.Data.LoadFromEnumerable(training);
                var pipeline = ml.Transforms.Concatenate("Features",
                                                         nameof(RunnerFeatures.Odds),
                                                         nameof(RunnerFeatures.Weight),
                                                         nameof(RunnerFeatures.Draw),
                                                         nameof(RunnerFeatures.Age))
                                 .Append(ml.BinaryClassification.Trainers.LightGbm());

                var model = pipeline.Fit(data);
                var engine = ml.Model.CreatePredictionEngine<RunnerFeatures, RunnerPrediction>(model);

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

                    var pred = engine.Predict(input);
                    if (best == null || pred.Probability > bestProb)
                    {
                        best = r;
                        bestProb = pred.Probability;
                    }
                }

                if (best != null)
                    return best;
            }

            // Fallback heuristic if there was no training data.
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

        private class RunnerPrediction
        {
            public bool PredictedLabel { get; set; }
            public float Probability { get; set; }
        }
    }
}