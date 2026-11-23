using System;
using System.Collections.Generic;

namespace HorseRacingML.ML
{
    public sealed class MarketBlendMetrics
    {
        public double LogLoss { get; init; }
        public double Brier { get; init; }
        public double RaceLogLoss { get; init; }
        public int RaceCount { get; init; }
        public int RunnerCount { get; init; }
    }

    public sealed class MarketBlendParameters
    {
        public double ModelWeight { get; init; }
        public double MarketWeight { get; init; }
        public MarketBlendMetrics Metrics { get; init; } = new();
        public bool HasData => Metrics.RunnerCount > 0;
    }

    /// <summary>
    /// Two-parameter log-linear blend between a model and the no-vig market:
    /// p_i is proportional to p_model_i^a * p_market_i^b.  The market-only
    /// baseline is the same family with a fixed at zero, so the gate compares
    /// the model against the best-tempered market rather than the raw market.
    /// </summary>
    public static class MarketBlendCalibration
    {
        private const double MinimumProbability = 1e-12;
        private const double MaximumWeight = 5d;
        private const int MaximumFitRaces = 4000;

        private sealed class RaceData
        {
            public double[] LogModel = Array.Empty<double>();
            public double[] LogMarket = Array.Empty<double>();
            public double[] Labels = Array.Empty<double>();
        }

        public static MarketBlendParameters Fit(
            IReadOnlyList<float> modelPredictions,
            IReadOnlyList<float> labels,
            IReadOnlyList<int> raceIds,
            IReadOnlyList<HyperparameterTrainer.RunnerExample> examples)
        {
            return FitCore(modelPredictions, labels, raceIds, examples, fitModelWeight: true);
        }

        public static MarketBlendParameters FitMarketOnly(
            IReadOnlyList<float> modelPredictions,
            IReadOnlyList<float> labels,
            IReadOnlyList<int> raceIds,
            IReadOnlyList<HyperparameterTrainer.RunnerExample> examples)
        {
            return FitCore(modelPredictions, labels, raceIds, examples, fitModelWeight: false);
        }

        public static MarketBlendMetrics Evaluate(
            IReadOnlyList<float> modelPredictions,
            IReadOnlyList<float> labels,
            IReadOnlyList<int> raceIds,
            IReadOnlyList<HyperparameterTrainer.RunnerExample> examples,
            double modelWeight,
            double marketWeight)
        {
            var races = BuildRaces(modelPredictions, labels, raceIds, examples);
            return EvaluateRaces(races, modelWeight, marketWeight);
        }

        /// <summary>
        /// Applies the blend to the full field, using the centred log market
        /// probability (with a neutral value for missing-price runners) from
        /// each <see cref="HyperparameterTrainer.RunnerExample"/>. The result
        /// is a coherent per-race probability vector.
        /// </summary>
        public static float[] ApplyFullField(
            IReadOnlyList<float> modelPredictions,
            IReadOnlyList<int> raceIds,
            IReadOnlyList<HyperparameterTrainer.RunnerExample> examples,
            double modelWeight,
            double marketWeight)
        {
            int count = modelPredictions.Count;
            var output = new float[count];
            if (count == 0 || raceIds.Count != count || examples.Count != count)
            {
                return output;
            }

            double a = ClampWeight(modelWeight);
            double b = ClampWeight(marketWeight);
            int start = 0;
            while (start < count)
            {
                int end = start + 1;
                while (end < count && raceIds[end] == raceIds[start])
                {
                    end++;
                }

                int length = end - start;
                var logits = new double[length];
                double max = double.NegativeInfinity;
                for (int i = 0; i < length; i++)
                {
                    int index = start + i;
                    double probability = modelPredictions[index];
                    if (!double.IsFinite(probability) || probability <= 0d)
                    {
                        probability = MinimumProbability;
                    }

                    double value = a * Math.Log(probability) +
                                   b * examples[index].MarketLogProbability;
                    logits[i] = value;
                    if (value > max)
                    {
                        max = value;
                    }
                }

                double sum = 0d;
                for (int i = 0; i < length; i++)
                {
                    logits[i] = Math.Exp(logits[i] - max);
                    sum += logits[i];
                }

                if (sum <= double.Epsilon || !double.IsFinite(sum))
                {
                    for (int i = 0; i < length; i++)
                    {
                        output[start + i] = (float)(1d / length);
                    }
                }
                else
                {
                    for (int i = 0; i < length; i++)
                    {
                        output[start + i] = (float)(logits[i] / sum);
                    }
                }

                start = end;
            }

            return output;
        }

        private static MarketBlendParameters FitCore(
            IReadOnlyList<float> modelPredictions,
            IReadOnlyList<float> labels,
            IReadOnlyList<int> raceIds,
            IReadOnlyList<HyperparameterTrainer.RunnerExample> examples,
            bool fitModelWeight)
        {
            var races = BuildRaces(modelPredictions, labels, raceIds, examples);
            if (races.Count == 0)
            {
                return new MarketBlendParameters
                {
                    ModelWeight = 0d,
                    MarketWeight = 1d,
                    Metrics = new MarketBlendMetrics()
                };
            }

            // Two parameters are cheap to fit, so a capped subset is enough and
            // keeps the per-candidate cost independent of the full history size.
            int stride = Math.Max(1, (races.Count + MaximumFitRaces - 1) / MaximumFitRaces);
            var fitRaces = new List<RaceData>(Math.Min(races.Count, MaximumFitRaces));
            for (int i = 0; i < races.Count; i += stride)
            {
                fitRaces.Add(races[i]);
            }

            double bestModelWeight = fitModelWeight ? 1d : 0d;
            double bestMarketWeight = 1d;
            double bestLoss = double.MaxValue;

            var modelWeights = fitModelWeight ? Grid(0d, 1.5d, 0.1d) : new[] { 0d };
            var marketWeights = Grid(0.4d, 1.6d, 0.1d);
            Search(modelWeights, marketWeights);

            var refinedModelWeights = fitModelWeight
                ? Grid(Math.Max(0d, bestModelWeight - 0.1d), Math.Min(2d, bestModelWeight + 0.1d), 0.02d)
                : new[] { 0d };
            var refinedMarketWeights = Grid(
                Math.Max(0.1d, bestMarketWeight - 0.1d),
                Math.Min(2d, bestMarketWeight + 0.1d),
                0.02d);
            Search(refinedModelWeights, refinedMarketWeights);

            if (!double.IsFinite(bestLoss))
            {
                bestModelWeight = 0d;
                bestMarketWeight = 1d;
            }

            return new MarketBlendParameters
            {
                ModelWeight = bestModelWeight,
                MarketWeight = bestMarketWeight,
                Metrics = EvaluateRaces(races, bestModelWeight, bestMarketWeight)
            };

            void Search(IReadOnlyList<double> models, IReadOnlyList<double> markets)
            {
                foreach (double modelWeight in models)
                {
                    foreach (double marketWeight in markets)
                    {
                        var metrics = EvaluateRaces(fitRaces, modelWeight, marketWeight);
                        if (metrics.RunnerCount > 0 && metrics.LogLoss < bestLoss)
                        {
                            bestLoss = metrics.LogLoss;
                            bestModelWeight = modelWeight;
                            bestMarketWeight = marketWeight;
                        }
                    }
                }
            }
        }

        private static List<RaceData> BuildRaces(
            IReadOnlyList<float> modelPredictions,
            IReadOnlyList<float> labels,
            IReadOnlyList<int> raceIds,
            IReadOnlyList<HyperparameterTrainer.RunnerExample> examples)
        {
            var races = new List<RaceData>();
            int count = modelPredictions.Count;
            if (count == 0 ||
                labels.Count != count ||
                raceIds.Count != count ||
                examples.Count != count)
            {
                return races;
            }

            int start = 0;
            while (start < count)
            {
                int end = start + 1;
                while (end < count && raceIds[end] == raceIds[start])
                {
                    end++;
                }

                var indices = new List<int>();
                bool hasWinner = false;
                for (int i = start; i < end; i++)
                {
                    double marketProbability = examples[i].MarketProbability;
                    if (marketProbability > 1e-6 && marketProbability < 1d - 1e-6)
                    {
                        indices.Add(i);
                        if (labels[i] >= 0.5f)
                        {
                            hasWinner = true;
                        }
                    }
                }

                if (indices.Count >= 2 && hasWinner)
                {
                    double modelSum = 0d;
                    double marketSum = 0d;
                    foreach (int index in indices)
                    {
                        modelSum += Math.Max(MinimumProbability, modelPredictions[index]);
                        marketSum += Math.Max(MinimumProbability, examples[index].MarketProbability);
                    }

                    if (modelSum > double.Epsilon && marketSum > double.Epsilon)
                    {
                        var race = new RaceData
                        {
                            LogModel = new double[indices.Count],
                            LogMarket = new double[indices.Count],
                            Labels = new double[indices.Count]
                        };
                        for (int k = 0; k < indices.Count; k++)
                        {
                            int index = indices[k];
                            race.LogModel[k] = Math.Log(Math.Max(
                                MinimumProbability,
                                modelPredictions[index] / modelSum));
                            race.LogMarket[k] = Math.Log(Math.Max(
                                MinimumProbability,
                                examples[index].MarketProbability / marketSum));
                            race.Labels[k] = labels[index];
                        }

                        races.Add(race);
                    }
                }

                start = end;
            }

            return races;
        }

        private static MarketBlendMetrics EvaluateRaces(
            IReadOnlyList<RaceData> races,
            double modelWeight,
            double marketWeight)
        {
            if (races.Count == 0)
            {
                return new MarketBlendMetrics();
            }

            double a = ClampWeight(modelWeight);
            double b = ClampWeight(marketWeight);
            double logLoss = 0d;
            double brier = 0d;
            double raceLogLoss = 0d;
            int runners = 0;
            int raceCount = 0;

            foreach (var race in races)
            {
                int length = race.Labels.Length;
                var logits = new double[length];
                double max = double.NegativeInfinity;
                for (int i = 0; i < length; i++)
                {
                    double value = a * race.LogModel[i] + b * race.LogMarket[i];
                    logits[i] = value;
                    if (value > max)
                    {
                        max = value;
                    }
                }

                double sum = 0d;
                for (int i = 0; i < length; i++)
                {
                    logits[i] = Math.Exp(logits[i] - max);
                    sum += logits[i];
                }

                if (sum <= double.Epsilon || !double.IsFinite(sum))
                {
                    continue;
                }

                double winnerLogProbability = 0d;
                for (int i = 0; i < length; i++)
                {
                    double probability = Math.Clamp(logits[i] / sum, 1e-7, 1d - 1e-7);
                    double label = race.Labels[i];
                    logLoss -= label * Math.Log(probability) +
                               (1d - label) * Math.Log(1d - probability);
                    brier += (probability - label) * (probability - label);
                    if (label >= 0.5d)
                    {
                        winnerLogProbability -= Math.Log(Math.Max(probability, MinimumProbability));
                    }
                    runners++;
                }

                raceLogLoss += winnerLogProbability;
                raceCount++;
            }

            if (runners == 0 || raceCount == 0)
            {
                return new MarketBlendMetrics();
            }

            return new MarketBlendMetrics
            {
                LogLoss = logLoss / runners,
                Brier = brier / runners,
                RaceLogLoss = raceLogLoss / raceCount,
                RaceCount = raceCount,
                RunnerCount = runners
            };
        }

        private static double ClampWeight(double value)
        {
            if (!double.IsFinite(value))
            {
                return 0d;
            }

            return Math.Clamp(value, 0d, MaximumWeight);
        }

        private static double[] Grid(double minimum, double maximum, double step)
        {
            if (maximum < minimum)
            {
                (minimum, maximum) = (maximum, minimum);
            }

            var values = new List<double>();
            int steps = (int)Math.Round((maximum - minimum) / step);
            for (int i = 0; i <= steps; i++)
            {
                values.Add(Math.Round(minimum + i * step, 6));
            }

            return values.ToArray();
        }
    }
}
