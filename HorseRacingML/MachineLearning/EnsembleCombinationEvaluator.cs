using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using HorseRacingML.Data;
using HorseRacingML.ML;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Compares how the members of one persisted ensemble are combined:
    /// equal geometric mean (the live path), equal arithmetic mean, dev-fitted
    /// weighted arithmetic/geometric combinations, and each member alone.
    /// No retraining and no artifact changes.
    /// </summary>
    public static class EnsembleCombinationEvaluator
    {
        public sealed class Metrics
        {
            public double RunnerLogLoss { get; set; }
            public double Brier { get; set; }
            public double RaceLogLoss { get; set; }
            public double Top1Accuracy { get; set; }
            public double Top3Accuracy { get; set; }
            public int RaceCount { get; set; }
            public int RunnerCount { get; set; }
            public double Weight { get; set; }
        }

        public sealed class Report
        {
            public string ModelFile { get; set; } = string.Empty;
            public int MemberCount { get; set; }
            public DateTime DevStart { get; set; }
            public DateTime DevEnd { get; set; }
            public DateTime TestStart { get; set; }
            public DateTime TestEnd { get; set; }
            public int DevRaces { get; set; }
            public int TestRaces { get; set; }
            public List<Metrics> MemberDev { get; set; } = new();
            public List<Metrics> MemberTest { get; set; } = new();
            public Metrics? GeometricEqualDev { get; set; }
            public Metrics? GeometricEqualTest { get; set; }
            public Metrics? ArithmeticEqualDev { get; set; }
            public Metrics? ArithmeticEqualTest { get; set; }
            public Metrics? BestArithmeticBrierDev { get; set; }
            public Metrics? BestArithmeticBrierTest { get; set; }
            public Metrics? BestArithmeticLogLossDev { get; set; }
            public Metrics? BestArithmeticLogLossTest { get; set; }
            public Metrics? BestGeometricBrierDev { get; set; }
            public Metrics? BestGeometricBrierTest { get; set; }
            public Metrics? BestGeometricLogLossDev { get; set; }
            public Metrics? BestGeometricLogLossTest { get; set; }
        }

        private sealed class Observation
        {
            public double[][] MemberLogs { get; set; } = Array.Empty<double[]>();
            public double[][] MemberProbabilities { get; set; } = Array.Empty<double[]>();
            public double[] Labels { get; set; } = Array.Empty<double>();
        }

        public static Report Run(
            IConfiguration configuration,
            RacingRepository repository,
            HyperparameterTrainer trainer,
            string modelPath,
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

            var json = File.ReadAllText(modelPath);
            var model = JsonSerializer.Deserialize<TrainedModel>(json)
                ?? throw new InvalidOperationException("Could not deserialize ensemble model.");
            var modelDirectory = Path.GetDirectoryName(modelPath);
            var memberModels = model.Members is { Count: > 0 }
                ? model.Members
                : new List<TrainedModel> { model };
            var members = new List<TrainedModelRuntime>();
            foreach (var member in memberModels)
            {
                var runtime = TrainedModelRuntime.FromModel(member, modelDirectory, out var status);
                if (runtime is null)
                {
                    throw new InvalidOperationException($"Could not load ensemble member: {status}");
                }

                members.Add(runtime);
            }

            log($"[combine] {Path.GetFileName(modelPath)}: {members.Count} member(s)");

            var devIds = new HashSet<int>(repository.GetRaceIdsBetweenDates(devStart, devEnd));
            var testIds = new HashSet<int>(repository.GetRaceIdsBetweenDates(testStart, testEnd));
            var included = new HashSet<int>(devIds);
            included.UnionWith(testIds);
            log($"[combine] preparing {devIds.Count} dev and {testIds.Count} test races; full history replay");
            var prepared = trainer.PrepareDataset(
                includeRaceIds: included,
                stateRaceWhitelist: null,
                includeIdentifiers: false,
                minimumRaceDate: null);

            var dev = new List<Observation>();
            var test = new List<Observation>();
            int skipped = 0;
            foreach (var race in prepared.Races)
            {
                var observation = BuildObservation(race, members);
                if (observation is null)
                {
                    skipped++;
                    continue;
                }

                if (devIds.Contains(race.RaceId))
                {
                    dev.Add(observation);
                }
                else if (testIds.Contains(race.RaceId))
                {
                    test.Add(observation);
                }
            }

            log($"[combine] observations: dev {dev.Count}, test {test.Count}, skipped {skipped}");
            if (dev.Count == 0 || test.Count == 0)
            {
                throw new InvalidOperationException("Ensemble comparison produced an empty dev or test set.");
            }

            var report = new Report
            {
                ModelFile = Path.GetFileName(modelPath),
                MemberCount = members.Count,
                DevStart = devStart,
                DevEnd = devEnd,
                TestStart = testStart,
                TestEnd = testEnd,
                DevRaces = dev.Count,
                TestRaces = test.Count
            };

            for (int i = 0; i < members.Count; i++)
            {
                report.MemberDev.Add(Evaluate(dev, new[] { i }, geometric: true, weight: 1));
                report.MemberTest.Add(Evaluate(test, new[] { i }, geometric: true, weight: 1));
            }

            report.GeometricEqualDev = EvaluateAll(dev, geometric: true, equal: true);
            report.GeometricEqualTest = EvaluateAll(test, geometric: true, equal: true);
            report.ArithmeticEqualDev = EvaluateAll(dev, geometric: false, equal: true);
            report.ArithmeticEqualTest = EvaluateAll(test, geometric: false, equal: true);

            var arithmeticCandidates = new List<(double Weight, Metrics Metrics)>();
            var geometricCandidates = new List<(double Weight, Metrics Metrics)>();
            for (double weight = 0; weight <= 1.0001; weight += 0.02)
            {
                double rounded = Math.Round(weight, 2);
                arithmeticCandidates.Add((rounded, EvaluateAll(dev, geometric: false, equal: false, weight: rounded)));
                geometricCandidates.Add((rounded, EvaluateAll(dev, geometric: true, equal: false, weight: rounded)));
            }

            var bestArithmeticBrier = arithmeticCandidates.OrderBy(candidate => candidate.Metrics.Brier).First();
            var bestArithmeticLogLoss = arithmeticCandidates.OrderBy(candidate => candidate.Metrics.RunnerLogLoss).First();
            var bestGeometricBrier = geometricCandidates.OrderBy(candidate => candidate.Metrics.Brier).First();
            var bestGeometricLogLoss = geometricCandidates.OrderBy(candidate => candidate.Metrics.RunnerLogLoss).First();

            report.BestArithmeticBrierDev = bestArithmeticBrier.Metrics;
            report.BestArithmeticBrierTest = EvaluateAll(test, geometric: false, equal: false, weight: bestArithmeticBrier.Weight);
            report.BestArithmeticLogLossDev = bestArithmeticLogLoss.Metrics;
            report.BestArithmeticLogLossTest = EvaluateAll(test, geometric: false, equal: false, weight: bestArithmeticLogLoss.Weight);
            report.BestGeometricBrierDev = bestGeometricBrier.Metrics;
            report.BestGeometricBrierTest = EvaluateAll(test, geometric: true, equal: false, weight: bestGeometricBrier.Weight);
            report.BestGeometricLogLossDev = bestGeometricLogLoss.Metrics;
            report.BestGeometricLogLossTest = EvaluateAll(test, geometric: true, equal: false, weight: bestGeometricLogLoss.Weight);

            log($"[combine] geometric equal dev Brier {report.GeometricEqualDev.Brier:F6}, test Brier {report.GeometricEqualTest.Brier:F6}");
            log($"[combine] arithmetic equal dev Brier {report.ArithmeticEqualDev.Brier:F6}, test Brier {report.ArithmeticEqualTest.Brier:F6}");
            log($"[combine] best arithmetic Brier weight {bestArithmeticBrier.Weight:F2}: test Brier {report.BestArithmeticBrierTest.Brier:F6}, test runner LL {report.BestArithmeticBrierTest.RunnerLogLoss:F6}, test top1 {report.BestArithmeticBrierTest.Top1Accuracy:P2}");

            return report;
        }

        private static Observation? BuildObservation(
            HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace race,
            IReadOnlyList<TrainedModelRuntime> members)
        {
            var rows = race.Rows;
            if (rows.Count < 2 || members.Count == 0)
            {
                return null;
            }

            var rowDicts = rows
                .Select(row => (IDictionary<string, object?>)row)
                .ToList();
            var memberProbabilities = new List<double[]>(members.Count);
            foreach (var member in members)
            {
                if (!member.TryPredictRace(rowDicts, out var probabilities) ||
                    probabilities.Length != rows.Count)
                {
                    return null;
                }

                memberProbabilities.Add(probabilities);
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

            var logs = new double[members.Count][];
            var memberProbabilityArrays = new double[members.Count][];
            for (int memberIndex = 0; memberIndex < members.Count; memberIndex++)
            {
                var source = memberProbabilities[memberIndex];
                double sum = indices.Sum(index => Math.Max(0, source[index]));
                if (sum <= 0)
                {
                    return null;
                }

                logs[memberIndex] = new double[indices.Count];
                memberProbabilityArrays[memberIndex] = new double[indices.Count];
                for (int k = 0; k < indices.Count; k++)
                {
                    int index = indices[k];
                    memberProbabilityArrays[memberIndex][k] = source[index] / sum;
                    logs[memberIndex][k] = Math.Log(Math.Max(source[index] / sum, 1e-12));
                }
            }

            var labels = new double[indices.Count];
            for (int k = 0; k < indices.Count; k++)
            {
                int index = indices[k];
                labels[k] = rows[index].TryGetValue("FinishPos", out var finish) &&
                            finish is not null &&
                            HyperparameterTrainer.TrainingDataset.PreparedDataset.TryConvertToInt32(
                                finish,
                                out var position) &&
                            position == 1
                    ? 1d
                    : 0d;
            }

            return new Observation
            {
                MemberLogs = logs,
                MemberProbabilities = memberProbabilityArrays,
                Labels = labels
            };
        }

        private static Metrics EvaluateAll(
            IReadOnlyList<Observation> observations,
            bool geometric,
            bool equal,
            double weight = 1.0)
        {
            return Evaluate(
                observations,
                Enumerable.Range(0, observations[0].MemberLogs.Length).ToArray(),
                geometric,
                weight,
                equal);
        }

        private static Metrics Evaluate(
            IReadOnlyList<Observation> observations,
            IReadOnlyList<int> memberIndices,
            bool geometric,
            double weight,
            bool equal = false)
        {
            int memberCount = observations[0].MemberLogs.Length;
            double logLoss = 0;
            double brier = 0;
            double raceLogLoss = 0;
            int runners = 0;
            int races = 0;
            int top1 = 0;
            int top3 = 0;

            foreach (var observation in observations)
            {
                int count = observation.Labels.Length;
                var probabilities = new double[count];
                if (geometric)
                {
                    var logs = new double[count];
                    double max = double.NegativeInfinity;
                    for (int i = 0; i < count; i++)
                    {
                        double value = 0;
                        if (memberIndices.Count == 1)
                        {
                            value = observation.MemberLogs[memberIndices[0]][i];
                        }
                        else if (equal)
                        {
                            foreach (int memberIndex in memberIndices)
                            {
                                value += observation.MemberLogs[memberIndex][i];
                            }

                            value /= memberIndices.Count;
                        }
                        else
                        {
                            int first = memberIndices[0];
                            int second = memberIndices[1];
                            value = weight * observation.MemberLogs[first][i] +
                                    (1 - weight) * observation.MemberLogs[second][i];
                        }

                        logs[i] = value;
                        if (value > max)
                        {
                            max = value;
                        }
                    }

                    double sum = 0;
                    for (int i = 0; i < count; i++)
                    {
                        logs[i] = Math.Exp(logs[i] - max);
                        sum += logs[i];
                    }

                    for (int i = 0; i < count; i++)
                    {
                        probabilities[i] = sum <= 0 ? 1.0 / count : logs[i] / sum;
                    }
                }
                else
                {
                    for (int i = 0; i < count; i++)
                    {
                        double value = 0;
                        if (memberIndices.Count == 1)
                        {
                            value = observation.MemberProbabilities[memberIndices[0]][i];
                        }
                        else if (equal)
                        {
                            foreach (int memberIndex in memberIndices)
                            {
                                value += observation.MemberProbabilities[memberIndex][i];
                            }

                            value /= memberIndices.Count;
                        }
                        else
                        {
                            int first = memberIndices[0];
                            int second = memberIndices[1];
                            value = weight * observation.MemberProbabilities[first][i] +
                                    (1 - weight) * observation.MemberProbabilities[second][i];
                        }

                        probabilities[i] = value;
                    }

                    double sum = probabilities.Sum();
                    if (sum <= 0)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            probabilities[i] = 1.0 / count;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < count; i++)
                        {
                            probabilities[i] /= sum;
                        }
                    }
                }

                double winnerMass = 0;
                bool hasWinner = false;
                for (int i = 0; i < count; i++)
                {
                    double probability = Math.Clamp(probabilities[i], 1e-7, 1 - 1e-7);
                    double label = observation.Labels[i];
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
                if (ranked.Count > 0 && observation.Labels[ranked[0]] >= 0.5)
                {
                    top1++;
                }

                bool top3Hit = false;
                for (int k = 0; k < Math.Min(3, ranked.Count); k++)
                {
                    if (observation.Labels[ranked[k]] >= 0.5)
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

            return new Metrics
            {
                RunnerLogLoss = runners == 0 ? 0 : logLoss / runners,
                Brier = runners == 0 ? 0 : brier / runners,
                RaceLogLoss = races == 0 ? 0 : raceLogLoss / races,
                Top1Accuracy = races == 0 ? 0 : (double)top1 / races,
                Top3Accuracy = races == 0 ? 0 : (double)top3 / races,
                RaceCount = races,
                RunnerCount = runners,
                Weight = weight
            };
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
