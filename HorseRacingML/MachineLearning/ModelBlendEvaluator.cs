using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.ML;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.ML
{
    /// <summary>
    /// One-off comparison/blend evaluator for two saved RaceSoftmax artifacts.
    ///
    /// The two models rank runners differently and have different calibration.
    /// This evaluator scores both on the same chronological dev/test rows and
    /// searches a log-linear blend
    ///   p_i ∝ p_new_i^a · p_old_i^b · p_market_i^c
    /// on dev, then reports the frozen blend on test. It also reports a
    /// temperature-scaled version of the new model.
    /// </summary>
    public static class ModelBlendEvaluator
    {
        public sealed class BlendMetrics
        {
            public double RunnerLogLoss { get; init; }
            public double Brier { get; init; }
            public double RaceLogLoss { get; init; }
            public double Top1Accuracy { get; init; }
            public double Top3Accuracy { get; init; }
            public int RaceCount { get; init; }
            public int RunnerCount { get; init; }
            public double NewWeight { get; init; }
            public double OldWeight { get; init; }
            public double MarketWeight { get; init; }
            public double Temperature { get; init; }
        }

        public sealed class Report
        {
            public string NewModel { get; init; } = string.Empty;
            public string OldModel { get; init; } = string.Empty;
            public DateTime DevStart { get; init; }
            public DateTime DevEnd { get; init; }
            public DateTime TestStart { get; init; }
            public DateTime TestEnd { get; init; }
            public int DevRaces { get; init; }
            public int TestRaces { get; init; }
            public BlendMetrics? OldDev { get; init; }
            public BlendMetrics? OldTest { get; init; }
            public BlendMetrics? NewDev { get; init; }
            public BlendMetrics? NewTest { get; init; }
            public BlendMetrics? BestBrierBlendDev { get; init; }
            public BlendMetrics? BestBrierBlendTest { get; init; }
            public BlendMetrics? BestLogLossBlendDev { get; init; }
            public BlendMetrics? BestLogLossBlendTest { get; init; }
            public BlendMetrics? BestRaceLogLossBlendDev { get; init; }
            public BlendMetrics? BestRaceLogLossBlendTest { get; init; }
            public BlendMetrics? BestConstrainedBlendDev { get; init; }
            public BlendMetrics? BestConstrainedBlendTest { get; init; }
            public BlendMetrics? BestTemperatureDev { get; init; }
            public BlendMetrics? BestTemperatureTest { get; init; }
        }

        private sealed class RaceObservation
        {
            public int RaceId { get; init; }
            public double[] LogNew { get; init; } = Array.Empty<double>();
            public double[] LogOld { get; init; } = Array.Empty<double>();
            public double[] LogMarket { get; init; } = Array.Empty<double>();
            public double[] Labels { get; init; } = Array.Empty<double>();
        }

        public static Report Run(
            IConfiguration configuration,
            RacingRepository repository,
            HyperparameterTrainer trainer,
            string newModelPath,
            string oldModelPath,
            DateTime devStart,
            DateTime devEnd,
            DateTime testStart,
            DateTime testEnd,
            Action<string>? log = null)
        {
            if (configuration is null) throw new ArgumentNullException(nameof(configuration));
            if (repository is null) throw new ArgumentNullException(nameof(repository));
            if (trainer is null) throw new ArgumentNullException(nameof(trainer));
            log ??= Console.WriteLine;

            var newRuntime = TrainedModelRuntime.TryLoad(newModelPath, out var newStatus)
                ?? throw new InvalidOperationException($"Could not load new model: {newStatus}");
            var oldRuntime = TrainedModelRuntime.TryLoad(oldModelPath, out var oldStatus)
                ?? throw new InvalidOperationException($"Could not load old model: {oldStatus}");
            log($"[blend] new model: {Path.GetFileName(newModelPath)} ({newStatus})");
            log($"[blend] old model: {Path.GetFileName(oldModelPath)} ({oldStatus})");

            var devIds = new HashSet<int>(repository.GetRaceIdsBetweenDates(devStart, devEnd));
            var testIds = new HashSet<int>(repository.GetRaceIdsBetweenDates(testStart, testEnd));
            var included = new HashSet<int>(devIds);
            included.UnionWith(testIds);
            log($"[blend] preparing {devIds.Count} dev and {testIds.Count} test races; full history replay");

            var prepared = trainer.PrepareDataset(
                includeRaceIds: included,
                stateRaceWhitelist: null,
                includeIdentifiers: false,
                minimumRaceDate: null);

            var devObservations = new List<RaceObservation>();
            var testObservations = new List<RaceObservation>();
            int skipped = 0;
            foreach (var race in prepared.Races)
            {
                var observation = BuildObservation(race, newRuntime, oldRuntime);
                if (observation is null)
                {
                    skipped++;
                    continue;
                }

                if (devIds.Contains(observation.RaceId))
                {
                    devObservations.Add(observation);
                }
                else if (testIds.Contains(observation.RaceId))
                {
                    testObservations.Add(observation);
                }
            }

            log($"[blend] observations: dev {devObservations.Count}, test {testObservations.Count}, skipped {skipped}");
            if (devObservations.Count == 0 || testObservations.Count == 0)
            {
                throw new InvalidOperationException("Blend evaluation produced an empty dev or test set.");
            }

            var oldDev = Evaluate(devObservations, 0, 1, 0);
            var oldTest = Evaluate(testObservations, 0, 1, 0);
            var newDev = Evaluate(devObservations, 1, 0, 0);
            var newTest = Evaluate(testObservations, 1, 0, 0);
            log($"[blend] old dev: runner LL {oldDev.RunnerLogLoss:F4}, Brier {oldDev.Brier:F4}, race LL {oldDev.RaceLogLoss:F4}, top1 {oldDev.Top1Accuracy:P2}, top3 {oldDev.Top3Accuracy:P2}");
            log($"[blend] new dev: runner LL {newDev.RunnerLogLoss:F4}, Brier {newDev.Brier:F4}, race LL {newDev.RaceLogLoss:F4}, top1 {newDev.Top1Accuracy:P2}, top3 {newDev.Top3Accuracy:P2}");

            var devGrid = SearchGrid(devObservations,
                objective: BlendObjective.Brier,
                minimumTop1: null,
                log);
            var (bestBrierWeights, bestBrierDev) = Refine(
                devObservations,
                devGrid,
                BlendObjective.Brier,
                minimumTop1: null);
            var bestBrierTest = Evaluate(testObservations, bestBrierWeights.A, bestBrierWeights.B, bestBrierWeights.C);

            var logLossGrid = SearchGrid(devObservations,
                objective: BlendObjective.RunnerLogLoss,
                minimumTop1: null,
                log);
            var (bestLogLossWeights, bestLogLossDev) = Refine(
                devObservations,
                logLossGrid,
                BlendObjective.RunnerLogLoss,
                minimumTop1: null);
            var bestLogLossTest = Evaluate(testObservations, bestLogLossWeights.A, bestLogLossWeights.B, bestLogLossWeights.C);

            var raceGrid = SearchGrid(devObservations,
                objective: BlendObjective.RaceLogLoss,
                minimumTop1: null,
                log);
            var (bestRaceWeights, bestRaceDev) = Refine(
                devObservations,
                raceGrid,
                BlendObjective.RaceLogLoss,
                minimumTop1: null);
            var bestRaceTest = Evaluate(testObservations, bestRaceWeights.A, bestRaceWeights.B, bestRaceWeights.C);

            // Constrained blend: improve runner Brier without losing top-1
            // against the old model on dev.
            var constrainedGrid = SearchGrid(
                devObservations,
                BlendObjective.Brier,
                minimumTop1: oldDev.Top1Accuracy,
                log);
            var (bestConstrainedWeights, bestConstrainedDev) = Refine(
                devObservations,
                constrainedGrid,
                BlendObjective.Brier,
                minimumTop1: oldDev.Top1Accuracy);
            var bestConstrainedTest = Evaluate(
                testObservations,
                bestConstrainedWeights.A,
                bestConstrainedWeights.B,
                bestConstrainedWeights.C);

            // Temperature scaling of the new model, which preserves ranking.
            var (bestTemperature, bestTemperatureDev) = FitTemperature(devObservations);
            var bestTemperatureTest = Evaluate(
                testObservations,
                a: 1.0 / bestTemperature,
                b: 0,
                c: 0,
                temperature: bestTemperature);

            var report = new Report
            {
                NewModel = Path.GetFileName(newModelPath),
                OldModel = Path.GetFileName(oldModelPath),
                DevStart = devStart,
                DevEnd = devEnd,
                TestStart = testStart,
                TestEnd = testEnd,
                DevRaces = devObservations.Count,
                TestRaces = testObservations.Count,
                OldDev = oldDev,
                OldTest = oldTest,
                NewDev = newDev,
                NewTest = newTest,
                BestBrierBlendDev = WithWeights(bestBrierDev, bestBrierWeights),
                BestBrierBlendTest = WithWeights(bestBrierTest, bestBrierWeights),
                BestLogLossBlendDev = WithWeights(bestLogLossDev, bestLogLossWeights),
                BestLogLossBlendTest = WithWeights(bestLogLossTest, bestLogLossWeights),
                BestRaceLogLossBlendDev = WithWeights(bestRaceDev, bestRaceWeights),
                BestRaceLogLossBlendTest = WithWeights(bestRaceTest, bestRaceWeights),
                BestConstrainedBlendDev = WithWeights(bestConstrainedDev, bestConstrainedWeights),
                BestConstrainedBlendTest = WithWeights(bestConstrainedTest, bestConstrainedWeights),
                BestTemperatureDev = WithWeights(bestTemperatureDev, new Weights(1.0 / bestTemperature, 0, 0), bestTemperature),
                BestTemperatureTest = WithWeights(bestTemperatureTest, new Weights(1.0 / bestTemperature, 0, 0), bestTemperature)
            };

            log($"[blend] best Brier blend: new {bestBrierWeights.A:F2}, old {bestBrierWeights.B:F2}, market {bestBrierWeights.C:F2}; dev Brier {bestBrierDev.Brier:F4}, test Brier {bestBrierTest.Brier:F4}, test race LL {bestBrierTest.RaceLogLoss:F4}, test top1 {bestBrierTest.Top1Accuracy:P2}");
            log($"[blend] best temperature: {bestTemperature:F2}; dev Brier {bestTemperatureDev.Brier:F4}, test Brier {bestTemperatureTest.Brier:F4}, test top1 {bestTemperatureTest.Top1Accuracy:P2}");
            return report;
        }

        private readonly record struct Weights(double A, double B, double C);

        private enum BlendObjective
        {
            Brier,
            RunnerLogLoss,
            RaceLogLoss
        }

        private static RaceObservation? BuildObservation(
            HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace race,
            TrainedModelRuntime newRuntime,
            TrainedModelRuntime oldRuntime)
        {
            var rows = race.Rows;
            if (rows.Count < 2)
            {
                return null;
            }

            var rowDicts = rows
                .Select(row => (IDictionary<string, object?>)row)
                .ToList();
            if (!newRuntime.TryPredictRace(rowDicts, out var newProbabilities) ||
                !oldRuntime.TryPredictRace(rowDicts, out var oldProbabilities) ||
                newProbabilities.Length != rows.Count ||
                oldProbabilities.Length != rows.Count)
            {
                return null;
            }

            var indices = new List<int>(rows.Count);
            for (int i = 0; i < rows.Count; i++)
            {
                if (rows[i].TryGetValue("MarketProbMissing", out var missing) &&
                    missing is bool isMissing && isMissing)
                {
                    continue;
                }

                if (!TryGetDouble(rows[i], "MarketImpliedProb", out var market) ||
                    market <= 1e-6)
                {
                    continue;
                }

                indices.Add(i);
            }

            if (indices.Count < 2)
            {
                return null;
            }

            bool hasWinner = indices.Any(index =>
                rows[index].TryGetValue("FinishPos", out var finish) &&
                finish is not null &&
                HyperparameterTrainer.TrainingDataset.PreparedDataset.TryConvertToInt32(
                    finish,
                    out var position) &&
                position == 1);
            if (!hasWinner)
            {
                return null;
            }

            double newSum = indices.Sum(index => Math.Max(0, newProbabilities[index]));
            double oldSum = indices.Sum(index => Math.Max(0, oldProbabilities[index]));
            double marketSum = indices.Sum(index =>
                TryGetDouble(rows[index], "MarketImpliedProb", out var value)
                    ? Math.Max(0, value)
                    : 0);
            if (newSum <= 0 || oldSum <= 0 || marketSum <= 0)
            {
                return null;
            }

            var logNew = new double[indices.Count];
            var logOld = new double[indices.Count];
            var logMarket = new double[indices.Count];
            var labels = new double[indices.Count];
            for (int k = 0; k < indices.Count; k++)
            {
                int index = indices[k];
                TryGetDouble(rows[index], "MarketImpliedProb", out var market);
                logNew[k] = Math.Log(Math.Max(newProbabilities[index] / newSum, 1e-12));
                logOld[k] = Math.Log(Math.Max(oldProbabilities[index] / oldSum, 1e-12));
                logMarket[k] = Math.Log(Math.Max(market / marketSum, 1e-12));
                labels[k] = rows[index].TryGetValue("FinishPos", out var finish) &&
                            finish is not null &&
                            HyperparameterTrainer.TrainingDataset.PreparedDataset.TryConvertToInt32(
                                finish,
                                out var position) &&
                            position == 1
                    ? 1d
                    : 0d;
            }

            return new RaceObservation
            {
                RaceId = race.RaceId,
                LogNew = logNew,
                LogOld = logOld,
                LogMarket = logMarket,
                Labels = labels
            };
        }

        private static (Weights Weights, BlendMetrics Metrics) SearchGrid(
            IReadOnlyList<RaceObservation> observations,
            BlendObjective objective,
            double? minimumTop1,
            Action<string> log)
        {
            Weights best = new(1, 0, 0);
            BlendMetrics bestMetrics = Evaluate(observations, 1, 0, 0);
            double bestScore = double.MaxValue;
            var values = Enumerable.Range(0, 13).Select(i => i * 0.1).ToList();
            foreach (var a in values)
            {
                foreach (var b in values)
                {
                    foreach (var c in values)
                    {
                        if (a + b + c <= 1e-9)
                        {
                            continue;
                        }

                        var metrics = Evaluate(observations, a, b, c);
                        if (minimumTop1.HasValue &&
                            metrics.Top1Accuracy < minimumTop1.Value)
                        {
                            continue;
                        }

                        double score = objective switch
                        {
                            BlendObjective.Brier => metrics.Brier,
                            BlendObjective.RunnerLogLoss => metrics.RunnerLogLoss,
                            _ => metrics.RaceLogLoss
                        };
                        if (score < bestScore)
                        {
                            bestScore = score;
                            best = new Weights(a, b, c);
                            bestMetrics = metrics;
                        }
                    }
                }
            }

            log($"[blend] coarse {objective}: new {best.A:F1}, old {best.B:F1}, market {best.C:F1}, score {bestScore:F4}");
            return (best, bestMetrics);
        }

        private static (Weights Weights, BlendMetrics Metrics) Refine(
            IReadOnlyList<RaceObservation> observations,
            (Weights Weights, BlendMetrics Metrics) coarse,
            BlendObjective objective,
            double? minimumTop1)
        {
            Weights best = coarse.Weights;
            BlendMetrics bestMetrics = coarse.Metrics;
            double bestScore = objective switch
            {
                BlendObjective.Brier => coarse.Metrics.Brier,
                BlendObjective.RunnerLogLoss => coarse.Metrics.RunnerLogLoss,
                _ => coarse.Metrics.RaceLogLoss
            };

            var aValues = Steps(Math.Max(0, best.A - 0.1), Math.Min(1.5, best.A + 0.1), 0.02);
            var bValues = Steps(Math.Max(0, best.B - 0.1), Math.Min(1.5, best.B + 0.1), 0.02);
            var cValues = Steps(Math.Max(0, best.C - 0.1), Math.Min(1.5, best.C + 0.1), 0.02);
            foreach (var a in aValues)
            {
                foreach (var b in bValues)
                {
                    foreach (var c in cValues)
                    {
                        if (a + b + c <= 1e-9)
                        {
                            continue;
                        }

                        var metrics = Evaluate(observations, a, b, c);
                        if (minimumTop1.HasValue &&
                            metrics.Top1Accuracy < minimumTop1.Value)
                        {
                            continue;
                        }

                        double score = objective switch
                        {
                            BlendObjective.Brier => metrics.Brier,
                            BlendObjective.RunnerLogLoss => metrics.RunnerLogLoss,
                            _ => metrics.RaceLogLoss
                        };
                        if (score < bestScore)
                        {
                            bestScore = score;
                            best = new Weights(a, b, c);
                            bestMetrics = metrics;
                        }
                    }
                }
            }

            return (best, bestMetrics);
        }

        private static (double Temperature, BlendMetrics Metrics) FitTemperature(
            IReadOnlyList<RaceObservation> observations)
        {
            double bestTemperature = 1.0;
            BlendMetrics bestMetrics = Evaluate(observations, 1, 0, 0);
            for (double temperature = 0.5; temperature <= 2.0; temperature += 0.05)
            {
                var metrics = Evaluate(observations, 1.0 / temperature, 0, 0, temperature);
                if (metrics.Brier < bestMetrics.Brier)
                {
                    bestTemperature = temperature;
                    bestMetrics = metrics;
                }
            }

            return (Math.Round(bestTemperature, 2), bestMetrics);
        }

        private static BlendMetrics Evaluate(
            IReadOnlyList<RaceObservation> observations,
            double a,
            double b,
            double c,
            double temperature = 1.0)
        {
            double logLoss = 0;
            double brier = 0;
            double raceLogLoss = 0;
            int runners = 0;
            int races = 0;
            int top1 = 0;
            int top3 = 0;

            foreach (var race in observations)
            {
                int count = race.Labels.Length;
                var logits = new double[count];
                double max = double.NegativeInfinity;
                for (int i = 0; i < count; i++)
                {
                    double value = a * race.LogNew[i] + b * race.LogOld[i] + c * race.LogMarket[i];
                    logits[i] = value;
                    if (value > max)
                    {
                        max = value;
                    }
                }

                if (double.IsNegativeInfinity(max) || double.IsNaN(max))
                {
                    continue;
                }

                double sum = 0;
                for (int i = 0; i < count; i++)
                {
                    logits[i] = Math.Exp(logits[i] - max);
                    sum += logits[i];
                }

                if (sum <= double.Epsilon || !double.IsFinite(sum))
                {
                    continue;
                }

                var probabilities = new double[count];
                double winnerMass = 0;
                bool hasWinner = false;
                for (int i = 0; i < count; i++)
                {
                    double probability = Math.Clamp(logits[i] / sum, 1e-7, 1 - 1e-7);
                    probabilities[i] = probability;
                    double label = race.Labels[i];
                    logLoss -= label * Math.Log(probability) +
                               (1 - label) * Math.Log(1 - probability);
                    brier += (probability - label) * (probability - label);
                    if (label >= 0.5)
                    {
                        hasWinner = true;
                        winnerMass += probability;
                    }

                    runners++;
                }

                if (!hasWinner || winnerMass <= 0)
                {
                    continue;
                }

                raceLogLoss -= Math.Log(Math.Clamp(winnerMass, 1e-7, 1));
                races++;
                var ranked = Enumerable.Range(0, count)
                    .OrderByDescending(i => probabilities[i])
                    .ToList();
                if (ranked.Count > 0 && race.Labels[ranked[0]] >= 0.5)
                {
                    top1++;
                }

                bool top3Hit = false;
                for (int k = 0; k < Math.Min(3, ranked.Count); k++)
                {
                    if (race.Labels[ranked[k]] >= 0.5)
                    {
                        top3Hit = true;
                        break;
                    }
                }

                if (top3Hit)
                {
                    top3++;
                }
            }

            return new BlendMetrics
            {
                RunnerLogLoss = runners == 0 ? 0 : logLoss / runners,
                Brier = runners == 0 ? 0 : brier / runners,
                RaceLogLoss = races == 0 ? 0 : raceLogLoss / races,
                Top1Accuracy = races == 0 ? 0 : (double)top1 / races,
                Top3Accuracy = races == 0 ? 0 : (double)top3 / races,
                RaceCount = races,
                RunnerCount = runners,
                NewWeight = a,
                OldWeight = b,
                MarketWeight = c,
                Temperature = temperature
            };
        }

        private static BlendMetrics WithWeights(BlendMetrics metrics, Weights weights, double temperature = 1.0)
        {
            return new BlendMetrics
            {
                RunnerLogLoss = metrics.RunnerLogLoss,
                Brier = metrics.Brier,
                RaceLogLoss = metrics.RaceLogLoss,
                Top1Accuracy = metrics.Top1Accuracy,
                Top3Accuracy = metrics.Top3Accuracy,
                RaceCount = metrics.RaceCount,
                RunnerCount = metrics.RunnerCount,
                NewWeight = weights.A,
                OldWeight = weights.B,
                MarketWeight = weights.C,
                Temperature = temperature
            };
        }

        private static List<double> Steps(double start, double end, double step)
        {
            var values = new List<double>();
            for (double value = start; value <= end + 1e-9; value += step)
            {
                values.Add(Math.Round(value, 4));
            }

            return values;
        }

        private static bool TryGetDouble(
            IReadOnlyDictionary<string, object?> row,
            string key,
            out double value)
        {
            value = 0;
            if (!row.TryGetValue(key, out var raw) || raw is null)
            {
                return false;
            }

            try
            {
                value = Convert.ToDouble(raw, CultureInfo.InvariantCulture);
                return double.IsFinite(value);
            }
            catch
            {
                return false;
            }
        }
    }
}
