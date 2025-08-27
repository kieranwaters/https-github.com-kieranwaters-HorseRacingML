using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.Models;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Provides simple logic to predict a race winner from a set of runners.
    /// This is a heuristic model that scores runners using available features
    /// such as betting odds, weight, draw and age. The runner with the highest
    /// score is returned as the predicted winner.
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

            RunnerResult? best = null;
            double bestScore = double.NegativeInfinity;

            foreach (var r in raceRunners)
            {
                // Lower starting price implies higher probability of winning
                double odds = (double)(r.SP_Decimal ?? 0m);
                if (odds <= 0) odds = 1000; // treat missing odds as very unlikely
                double score = 1.0 / odds;

                // Horses carrying less weight, with a favourable draw or younger age
                // are given a slightly higher score.
                if (r.WeightLbs.HasValue)
                    score += 1.0 / (1 + r.WeightLbs.Value);
                if (r.Draw.HasValue)
                    score += 1.0 / (1 + r.Draw.Value);
                if (r.Age.HasValue)
                    score += 1.0 / (1 + r.Age.Value);

                if (best == null || score > bestScore)
                {
                    best = r;
                    bestScore = score;
                }
            }

            // best will always be non-null here due to earlier check
            return best!;
        }
    }
}