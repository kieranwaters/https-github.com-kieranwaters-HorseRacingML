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
    /// Deployable calibration comparison for one persisted ensemble.
    ///
    /// All calibration parameters are fitted on the dev window and frozen for
    /// test. Variants:
    ///   baseline (no calibration),
    ///   global Platt scaling,
    ///   Platt scaling by market-rank bucket,
    ///   Platt scaling by field-size bucket.
    /// </summary>
    public static class CalibrationEvaluator
    {
        public sealed class Report
        {
            public string ModelFile { get; set; } = string.Empty;
            public DateTime DevStart { get; set; }
            public DateTime DevEnd { get; set; }
            public DateTime TestStart { get; set; }
            public DateTime TestEnd { get; set; }
            public int DevRaces { get; set; }
            public int TestRaces { get; set; }
            public RaceEvaluationMetrics? BaselineDev { get; set; }
            public RaceEvaluationMetrics? BaselineTest { get; set; }
            public RaceEvaluationMetrics? GlobalPlattDev { get; set; }
            public RaceEvaluationMetrics? GlobalPlattTest { get; set; }
            public RaceEvaluationMetrics? RankBucketPlattDev { get; set; }
            public RaceEvaluationMetrics? RankBucketPlattTest { get; set; }
            public RaceEvaluationMetrics? FieldBucketPlattDev { get; set; }
            public RaceEvaluationMetrics? FieldBucketPlattTest { get; set; }
            public List<PlattParameters> GlobalParameters { get; set; } = new();
            public List<PlattParameters> RankBucketParameters { get; set; } = new();
            public List<PlattParameters> FieldBucketParameters { get; set; } = new();
        }

        public sealed class PlattParameters
        {
            public string Bucket { get; set; } = string.Empty;
            public int Runners { get; set; }
            public double Intercept { get; set; }
            public double Slope { get; set; }
        }

        private sealed class Observation
        {
            public int RaceId { get; init; }
            public double[] ModelProbabilities { get; init; } = Array.Empty<double>();
            public double[] MarketProbabilities { get; init; } = Array.Empty<double>();
            public double[] Labels { get; init; } = Array.Empty<double>();
            public int[] MarketRanks { get; init; } = Array.Empty<int>();
            public int FieldSize => ModelProbabilities.Length;
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

            var model = JsonSerializer.Deserialize<TrainedModel>(File.ReadAllText(modelPath))
                ?? throw new InvalidOperationException("Could not deserialize the model.");
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

            log($"[calibration] {Path.GetFileName(modelPath)}: {members.Count} member(s)");

            var devIds = new HashSet<int>(repository.GetRaceIdsBetweenDates(devStart, devEnd));
            var testIds = new HashSet<int>(repository.GetRaceIdsBetweenDates(testStart, testEnd));
            var included = new HashSet<int>(devIds);
            included.UnionWith(testIds);
            log($"[calibration] preparing {devIds.Count} dev and {testIds.Count} test races; full history replay");
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

            log($"[calibration] observations: dev {dev.Count}, test {test.Count}, skipped {skipped}");
            if (dev.Count == 0 || test.Count == 0)
            {
                throw new InvalidOperationException("Calibration evaluation produced an empty dev or test set.");
            }

            var report = new Report
            {
                ModelFile = Path.GetFileName(modelPath),
                DevStart = devStart,
                DevEnd = devEnd,
                TestStart = testStart,
                TestEnd = testEnd,
                DevRaces = dev.Count,
                TestRaces = test.Count,
                BaselineDev = Evaluate(dev),
                BaselineTest = Evaluate(test)
            };

            var global = FitPlatt(dev, (_, _) => true, "all");
            report.GlobalParameters.Add(global);
            report.GlobalPlattDev = EvaluateWithCalibration(dev, _ => new[] { global });
            report.GlobalPlattTest = EvaluateWithCalibration(test, _ => new[] { global });

            foreach (int bucket in new[] { 0, 1, 2, 3, 4 })
            {
                var parameters = FitPlatt(
                    dev,
                    (observation, runnerIndex) =>
                        RankBucketForRank(observation.MarketRanks[runnerIndex]) == bucket,
                    $"rank-{bucket}",
                    fallback: global);
                report.RankBucketParameters.Add(parameters);
            }

            report.RankBucketPlattDev = EvaluateWithCalibration(
                dev,
                observation => RankBucketParametersFor(observation, report.RankBucketParameters));
            report.RankBucketPlattTest = EvaluateWithCalibration(
                test,
                observation => RankBucketParametersFor(observation, report.RankBucketParameters));

            foreach (int bucket in new[] { 0, 1, 2 })
            {
                var parameters = FitPlatt(
                    dev,
                    (observation, _) => FieldBucket(observation.FieldSize) == bucket,
                    $"field-{bucket}",
                    fallback: global);
                report.FieldBucketParameters.Add(parameters);
            }

            report.FieldBucketPlattDev = EvaluateWithCalibration(
                dev,
                observation => FieldBucketParametersFor(observation, report.FieldBucketParameters));
            report.FieldBucketPlattTest = EvaluateWithCalibration(
                test,
                observation => FieldBucketParametersFor(observation, report.FieldBucketParameters));

            log($"[calibration] baseline dev LL {report.BaselineDev.RunnerLogLoss:F6}, Brier {report.BaselineDev.Brier:F6}, race {report.BaselineDev.RaceLogLoss:F6}, top1 {report.BaselineDev.Top1Accuracy:P2}");
            log($"[calibration] global dev LL {report.GlobalPlattDev.RunnerLogLoss:F6}, Brier {report.GlobalPlattDev.Brier:F6}, test LL {report.GlobalPlattTest.RunnerLogLoss:F6}, Brier {report.GlobalPlattTest.Brier:F6}");
            log($"[calibration] rank-bucket dev LL {report.RankBucketPlattDev.RunnerLogLoss:F6}, Brier {report.RankBucketPlattDev.Brier:F6}, test LL {report.RankBucketPlattTest.RunnerLogLoss:F6}, Brier {report.RankBucketPlattTest.Brier:F6}");
            log($"[calibration] field-bucket dev LL {report.FieldBucketPlattDev.RunnerLogLoss:F6}, Brier {report.FieldBucketPlattDev.Brier:F6}, test LL {report.FieldBucketPlattTest.RunnerLogLoss:F6}, Brier {report.FieldBucketPlattTest.Brier:F6}");

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
            var market = new List<double>(rows.Count);
            for (int i = 0; i < rows.Count; i++)
            {
                if (rows[i].TryGetValue("MarketProbMissing", out var missing) &&
                    missing is bool isMissing && isMissing)
                {
                    continue;
                }

                if (!TryGetDouble(rows[i], "MarketImpliedProb", out var probability) ||
                    probability <= 1e-6)
                {
                    continue;
                }

                indices.Add(i);
                market.Add(probability);
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

            var modelProbabilities = new double[indices.Count];
            double modelSum = 0;
            for (int k = 0; k < indices.Count; k++)
            {
                double sum = 0;
                foreach (var member in memberProbabilities)
                {
                    sum += Math.Max(0, member[indices[k]]);
                }

                modelProbabilities[k] = sum / memberProbabilities.Count;
                modelSum += modelProbabilities[k];
            }

            if (modelSum <= 0)
            {
                return null;
            }

            for (int k = 0; k < modelProbabilities.Length; k++)
            {
                modelProbabilities[k] /= modelSum;
            }

            double marketSum = market.Sum();
            if (marketSum <= 0)
            {
                return null;
            }

            var marketProbabilities = market.Select(value => value / marketSum).ToArray();
            var ranks = Enumerable.Range(0, marketProbabilities.Length)
                .OrderByDescending(i => marketProbabilities[i])
                .Select((index, rank) => (index, rank))
                .ToArray();
            var marketRanks = new int[marketProbabilities.Length];
            foreach (var (index, rank) in ranks)
            {
                marketRanks[index] = rank + 1;
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
                RaceId = race.RaceId,
                ModelProbabilities = modelProbabilities,
                MarketProbabilities = marketProbabilities,
                Labels = labels,
                MarketRanks = marketRanks
            };
        }

        private static RaceEvaluationMetrics Evaluate(IReadOnlyList<Observation> observations)
        {
            var predictions = new List<float>();
            var labels = new List<float>();
            var raceIds = new List<int>();
            foreach (var observation in observations)
            {
                double sum = observation.ModelProbabilities.Sum();
                if (sum <= 0)
                {
                    continue;
                }

                for (int i = 0; i < observation.ModelProbabilities.Length; i++)
                {
                    predictions.Add((float)(observation.ModelProbabilities[i] / sum));
                    labels.Add((float)observation.Labels[i]);
                    raceIds.Add(observation.RaceId);
                }
            }

            return RaceMetrics.Compute(raceIds, predictions, labels);
        }

        private static RaceEvaluationMetrics EvaluateWithCalibration(
            IReadOnlyList<Observation> observations,
            Func<Observation, IReadOnlyList<PlattParameters>> parametersFor)
        {
            var predictions = new List<float>();
            var labels = new List<float>();
            var raceIds = new List<int>();
            foreach (var observation in observations)
            {
                var parameters = parametersFor(observation);
                var calibrated = new double[observation.ModelProbabilities.Length];
                double sum = 0;
                for (int i = 0; i < calibrated.Length; i++)
                {
                    var bucket = parameters[Math.Min(i, parameters.Count - 1)];
                    double p = Math.Clamp(observation.ModelProbabilities[i], 1e-6, 1 - 1e-6);
                    double logit = Math.Log(p / (1 - p));
                    double calibratedLogit = bucket.Intercept + bucket.Slope * logit;
                    double calibratedProbability = 1.0 / (1.0 + Math.Exp(-calibratedLogit));
                    calibrated[i] = Math.Clamp(calibratedProbability, 1e-9, 1 - 1e-9);
                    sum += calibrated[i];
                }

                if (sum <= 0)
                {
                    continue;
                }

                for (int i = 0; i < calibrated.Length; i++)
                {
                    predictions.Add((float)(calibrated[i] / sum));
                    labels.Add((float)observation.Labels[i]);
                    raceIds.Add(observation.RaceId);
                }
            }

            return RaceMetrics.Compute(raceIds, predictions, labels);
        }

        private static IReadOnlyList<PlattParameters> RankBucketParametersFor(
            Observation observation,
            IReadOnlyList<PlattParameters> parameters)
        {
            var output = new PlattParameters[observation.FieldSize];
            for (int i = 0; i < output.Length; i++)
            {
                int bucket = RankBucketForRank(observation.MarketRanks[i]);
                output[i] = parameters[Math.Clamp(bucket, 0, parameters.Count - 1)];
            }

            return output;
        }

        private static IReadOnlyList<PlattParameters> FieldBucketParametersFor(
            Observation observation,
            IReadOnlyList<PlattParameters> parameters)
        {
            int bucket = FieldBucket(observation.FieldSize);
            var output = new PlattParameters[observation.FieldSize];
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = parameters[Math.Clamp(bucket, 0, parameters.Count - 1)];
            }

            return output;
        }

        private static PlattParameters FitPlatt(
            IReadOnlyList<Observation> observations,
            Func<Observation, int, bool> bucketPredicate,
            string bucketName,
            PlattParameters? fallback = null)
        {
            var rows = new List<(double Logit, double Label)>();
            foreach (var observation in observations)
            {
                for (int i = 0; i < observation.ModelProbabilities.Length; i++)
                {
                    if (!bucketPredicate(observation, i))
                    {
                        continue;
                    }

                    double p = Math.Clamp(observation.ModelProbabilities[i], 1e-6, 1 - 1e-6);
                    rows.Add((Math.Log(p / (1 - p)), observation.Labels[i]));
                }
            }

            if (rows.Count < 200)
            {
                return fallback ?? new PlattParameters
                {
                    Bucket = bucketName,
                    Runners = rows.Count,
                    Intercept = 0,
                    Slope = 1
                };
            }

            double bestLoss = double.MaxValue;
            double bestA = 0;
            double bestB = 1;
            for (double a = -0.8; a <= 0.8001; a += 0.05)
            {
                for (double b = 0.5; b <= 1.5001; b += 0.05)
                {
                    double loss = 0;
                    foreach (var (logit, label) in rows)
                    {
                        double z = a + b * logit;
                        double p = 1.0 / (1.0 + Math.Exp(-z));
                        p = Math.Clamp(p, 1e-7, 1 - 1e-7);
                        loss -= label * Math.Log(p) + (1 - label) * Math.Log(1 - p);
                    }

                    if (loss < bestLoss)
                    {
                        bestLoss = loss;
                        bestA = a;
                        bestB = b;
                    }
                }
            }

            return new PlattParameters
            {
                Bucket = bucketName,
                Runners = rows.Count,
                Intercept = Math.Round(bestA, 4),
                Slope = Math.Round(bestB, 4)
            };
        }

        private static int RankBucketForRank(int rank)
        {
            return rank switch
            {
                1 => 0,
                2 => 1,
                3 => 2,
                <= 6 => 3,
                _ => 4
            };
        }

        private static int FieldBucket(int fieldSize)
        {
            return fieldSize <= 8 ? 0 : fieldSize <= 12 ? 1 : 2;
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
