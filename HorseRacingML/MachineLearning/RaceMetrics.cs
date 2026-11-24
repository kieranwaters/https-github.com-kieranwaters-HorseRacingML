using System;
using System.Collections.Generic;
using System.Linq;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Race-aware evaluation metrics.
    ///
    /// A horse race is a single-choice problem: exactly one runner wins (dead
    /// heats aside), and the model's probabilities are normalised over the
    /// field. The runner-level binary log loss/Brier that the trainer used to
    /// report is a proper composite score, but it is dominated by the many
    /// losers in large fields and is not the metric live betting is judged on.
    /// The race log loss (<c>-log p(winner)</c>) and top-k accuracy are the
    /// primary metrics; the runner-level scores are kept for comparison with
    /// older runs.
    /// </summary>
    public sealed record RaceEvaluationMetrics(
        double RaceLogLoss,
        double Top1Accuracy,
        double Top3Accuracy,
        double RunnerLogLoss,
        double Brier,
        int RaceCount,
        int RunnerCount)
    {
        public static RaceEvaluationMetrics Empty { get; } = new(0, 0, 0, 0, 0, 0, 0);
    }

    public static class RaceMetrics
    {
        private const double MinimumProbability = 1e-7;

        public static RaceEvaluationMetrics Compute(
            IReadOnlyList<int> raceIds,
            IReadOnlyList<float> predictions,
            IReadOnlyList<float> labels)
        {
            if (raceIds is null || predictions is null || labels is null)
            {
                return RaceEvaluationMetrics.Empty;
            }

            int count = Math.Min(raceIds.Count, Math.Min(predictions.Count, labels.Count));
            if (count == 0)
            {
                return RaceEvaluationMetrics.Empty;
            }

            var grouped = new Dictionary<int, List<int>>();
            var raceOrder = new List<int>();
            for (int i = 0; i < count; i++)
            {
                int raceId = raceIds[i];
                if (!grouped.TryGetValue(raceId, out var indices))
                {
                    indices = new List<int>(8);
                    grouped[raceId] = indices;
                    raceOrder.Add(raceId);
                }

                indices.Add(i);
            }

            double raceLogLoss = 0;
            double runnerLogLoss = 0;
            double brier = 0;
            int raceCount = 0;
            int runnerCount = 0;
            int top1 = 0;
            int top3 = 0;

            foreach (int raceId in raceOrder)
            {
                var indices = grouped[raceId];
                if (indices.Count == 0)
                {
                    continue;
                }

                double probabilitySum = 0;
                foreach (int index in indices)
                {
                    float probability = predictions[index];
                    if (float.IsFinite(probability) && probability > 0f)
                    {
                        probabilitySum += probability;
                    }
                }

                if (probabilitySum <= double.Epsilon || !double.IsFinite(probabilitySum))
                {
                    continue;
                }

                var normalised = new double[indices.Count];
                bool hasWinner = false;
                double winnerMass = 0;
                for (int k = 0; k < indices.Count; k++)
                {
                    int index = indices[k];
                    double probability = float.IsFinite(predictions[index]) && predictions[index] > 0f
                        ? predictions[index] / probabilitySum
                        : 0;
                    normalised[k] = probability;

                    double label = labels[index];
                    if (label >= 0.5)
                    {
                        hasWinner = true;
                        winnerMass += probability;
                    }

                    double q = Math.Clamp(probability, MinimumProbability, 1d - MinimumProbability);
                    bool isWinner = label >= 0.5;
                    double y = isWinner ? 1d : 0d;
                    runnerLogLoss -= isWinner
                        ? Math.Log(q)
                        : Math.Log(1d - q);
                    brier += (q - y) * (q - y);
                    runnerCount++;
                }

                if (!hasWinner || winnerMass <= double.Epsilon)
                {
                    // An unpriced/dead-heat race with no usable winner cannot
                    // contribute a meaningful conditional likelihood.
                    continue;
                }

                raceLogLoss -= Math.Log(Math.Clamp(winnerMass, MinimumProbability, 1d));
                raceCount++;

                // Top-k accuracy: which runners did the model rank highest?
                // A dead heat counts as correct if any winner appears in the
                // top-k, and the metric is still one observation per race.
                var ranked = Enumerable.Range(0, normalised.Length)
                    .OrderByDescending(k => normalised[k])
                    .ToList();
                if (ranked.Count > 0)
                {
                    int best = ranked[0];
                    if (labels[indices[best]] >= 0.5)
                    {
                        top1++;
                    }
                }

                int topK = Math.Min(3, ranked.Count);
                bool winnerInTopK = false;
                for (int k = 0; k < topK; k++)
                {
                    if (labels[indices[ranked[k]]] >= 0.5)
                    {
                        winnerInTopK = true;
                        break;
                    }
                }

                if (winnerInTopK)
                {
                    top3++;
                }
            }

            if (raceCount == 0 || runnerCount == 0)
            {
                return RaceEvaluationMetrics.Empty;
            }

            return new RaceEvaluationMetrics(
                raceLogLoss / raceCount,
                (double)top1 / raceCount,
                (double)top3 / raceCount,
                runnerLogLoss / runnerCount,
                brier / runnerCount,
                raceCount,
                runnerCount);
        }

        public static double ComputeRaceLogLoss(
            IReadOnlyList<int> raceIds,
            IReadOnlyList<float> predictions,
            IReadOnlyList<float> labels)
        {
            return Compute(raceIds, predictions, labels).RaceLogLoss;
        }

        public static (double Top1, double Top3) ComputeAccuracy(
            IReadOnlyList<int> raceIds,
            IReadOnlyList<float> predictions,
            IReadOnlyList<float> labels)
        {
            var metrics = Compute(raceIds, predictions, labels);
            return (metrics.Top1Accuracy, metrics.Top3Accuracy);
        }

        /// <summary>
        /// Combines several probability vectors by averaging in log space and
        /// renormalising per race. Averaging logits is the standard way to
        /// ensemble conditional-logit models because it preserves the relative
        /// confidence of each member instead of letting one overconfident member
        /// dominate.
        /// </summary>
        public static float[] AverageLogProbabilities(
            IReadOnlyList<IReadOnlyList<float>> memberPredictions,
            IReadOnlyList<int> raceIds)
        {
            if (memberPredictions is null || memberPredictions.Count == 0)
            {
                return Array.Empty<float>();
            }

            int count = memberPredictions[0].Count;
            if (count == 0 || raceIds.Count != count)
            {
                return Array.Empty<float>();
            }

            var output = new float[count];
            int start = 0;
            while (start < count)
            {
                int end = start + 1;
                while (end < count && raceIds[end] == raceIds[start])
                {
                    end++;
                }

                for (int i = start; i < end; i++)
                {
                    double logSum = 0;
                    int members = 0;
                    foreach (var predictions in memberPredictions)
                    {
                        if (predictions.Count != count)
                        {
                            continue;
                        }

                        double probability = predictions[i];
                        if (!double.IsFinite(probability))
                        {
                            probability = 0;
                        }

                        // Clamp rather than skip: a member that assigns a zero
                        // probability must still pull the ensemble down.
                        logSum += Math.Log(Math.Max(probability, 1e-12));
                        members++;
                    }

                    output[i] = members > 0
                        ? (float)Math.Exp(logSum / members)
                        : 0f;
                }

                double sum = 0;
                for (int i = start; i < end; i++)
                {
                    if (float.IsFinite(output[i]) && output[i] > 0)
                    {
                        sum += output[i];
                    }
                }

                if (sum > double.Epsilon && double.IsFinite(sum))
                {
                    for (int i = start; i < end; i++)
                    {
                        output[i] = (float)(Math.Max(0f, output[i]) / sum);
                    }
                }

                start = end;
            }

            return output;
        }
    }
}
