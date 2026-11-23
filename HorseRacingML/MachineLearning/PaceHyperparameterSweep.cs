using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.ML
{
    public sealed class SweepCandidateReport
    {
        public string Name { get; set; } = string.Empty;
        public int Units { get; set; }
        public int Layers { get; set; }
        public double Dropout { get; set; }
        public double LearningRate { get; set; }
        public int BatchSize { get; set; }
        public int RaceBatchSize { get; set; }
        public int Seed { get; set; }
        public int EpochBudget { get; set; }
        public int? BestEpoch { get; set; }
        public double Temperature { get; set; }
        public double DevModelLogLoss { get; set; }
        public double DevModelBrier { get; set; }
        public double DevMarketLogLoss { get; set; }
        public double DevMarketBrier { get; set; }
        public double? DevFairMarketLogLoss { get; set; }
        public double? DevFairMarketBrier { get; set; }
        public double? DevBlendLogLoss { get; set; }
        public double? DevBlendBrier { get; set; }
        public double? DevBlendModelWeight { get; set; }
        public double? DevBlendMarketWeight { get; set; }
        public bool PassedMarketGate { get; set; }
        public double SelectionScore { get; set; }
        public double? TestModelLogLoss { get; set; }
        public double? TestModelBrier { get; set; }
        public double? TestMarketLogLoss { get; set; }
        public double? TestMarketBrier { get; set; }
        public double? TestFairMarketLogLoss { get; set; }
        public double? TestFairMarketBrier { get; set; }
        public double? TestBlendLogLoss { get; set; }
        public double? TestBlendBrier { get; set; }
        public double? TestTemperature { get; set; }
    }

    public sealed class PaceHyperparameterSweepReport
    {
        public bool PaceFeaturesEnabled { get; init; }
        public int FeatureCount { get; init; }
        public int ReferenceParameterId { get; init; }
        public int TrainRaces { get; init; }
        public int DevRaces { get; init; }
        public int TestRaces { get; init; }
        public int TrainRows { get; init; }
        public int DevRows { get; init; }
        public int TestRows { get; init; }
        public double MinimumBlendImprovement { get; init; }
        public List<SweepCandidateReport> Candidates { get; init; } = new();
        public SweepCandidateReport? Winner { get; init; }

        public string FormatSummary()
        {
            var lines = new List<string>
            {
                $"Pace features: {(PaceFeaturesEnabled ? "ON" : "OFF")} (reference {ReferenceParameterId}, {FeatureCount} encoded features)",
                $"Races train/dev/test: {TrainRaces}/{DevRaces}/{TestRaces}; rows: {TrainRows}/{DevRows}/{TestRows}",
                "Top dev candidates:"
            };

            foreach (var candidate in Candidates
                         .OrderByDescending(c => c.PassedMarketGate)
                         .ThenBy(c => c.SelectionScore)
                         .ThenBy(c => c.DevModelBrier)
                         .Take(10))
            {
                lines.Add(
                    $"  {candidate.Name,-22} U{candidate.Units,-3} L{candidate.Layers} D{candidate.Dropout:0.00} LR{candidate.LearningRate:0.0000} B{candidate.BatchSize,-4} RB{candidate.RaceBatchSize,-3} " +
                    $"dev LL {candidate.DevModelLogLoss:F6} Brier {candidate.DevModelBrier:F6} blendLL {candidate.DevBlendLogLoss?.ToString("F6", CultureInfo.InvariantCulture) ?? "n/a"} gate={(candidate.PassedMarketGate ? "PASS" : "fail")}");
            }

            if (Winner is null)
            {
                lines.Add("No sweep winner was produced.");
                return string.Join(Environment.NewLine, lines);
            }

            lines.Add(
                $"WINNER {Winner.Name}: test LL {Winner.TestModelLogLoss:F6} Brier {Winner.TestModelBrier:F6}; " +
                $"test blend LL {Winner.TestBlendLogLoss?.ToString("F6", CultureInfo.InvariantCulture) ?? "n/a"} " +
                $"Brier {Winner.TestBlendBrier?.ToString("F6", CultureInfo.InvariantCulture) ?? "n/a"}; " +
                $"dev blend LL {Winner.DevBlendLogLoss?.ToString("F6", CultureInfo.InvariantCulture) ?? "n/a"}; " +
                $"best epoch {Winner.BestEpoch?.ToString(CultureInfo.InvariantCulture) ?? "n/a"}");
            return string.Join(Environment.NewLine, lines);
        }
    }

    public static class PaceHyperparameterSweep
    {
        public static PaceHyperparameterSweepReport Run(
            IConfiguration configuration,
            RacingRepository repository,
            HyperparameterTrainer trainer,
            MLParameter reference,
            int? maxCandidates = null,
            int candidateOffset = 0,
            Action<string>? log = null)
        {
            if (configuration is null) throw new ArgumentNullException(nameof(configuration));
            if (repository is null) throw new ArgumentNullException(nameof(repository));
            if (trainer is null) throw new ArgumentNullException(nameof(trainer));
            if (reference is null) throw new ArgumentNullException(nameof(reference));

            log ??= Console.WriteLine;

            var minimumRaceDate = configuration.GetValue<DateTime?>("Training:MinimumMarketRaceDate")
                ?? configuration.GetValue<DateTime?>("Training:MinimumRaceDate")
                ?? new DateTime(2017, 1, 1);
            bool requirePreOffPrice = configuration.GetValue<bool?>("Training:RequirePreOffPrice") ?? true;
            bool keepUnpricedRunners = configuration.GetValue<bool?>("Training:KeepUnpricedRunners") ?? true;
            double devFraction = Math.Clamp(configuration.GetValue<double?>("Training:DevFraction") ?? 0.10, 0.01, 0.40);
            double testFraction = Math.Clamp(configuration.GetValue<double?>("Training:TestFraction") ?? 0.10, 0.01, 0.40);
            decimal commissionRate = Math.Clamp(configuration.GetValue<decimal?>("Betting:CommissionRate") ?? 0.05m, 0m, 1m);
            double minimumBlendImprovement = Math.Max(
                0d,
                configuration.GetValue<double?>("Training:MinimumBlendLogLossImprovement") ?? 0.0002d);
            bool marketBlind = reference.MarketBlind ?? configuration.GetValue<bool?>("Training:MarketBlind") ?? false;

            var allRaceIds = requirePreOffPrice
                ? repository.GetRaceIdsWithOpeningPrice(minimumRaceDate, DateTime.Today)
                : repository.GetRaceIdsBetweenDates(minimumRaceDate, DateTime.Today);
            if (allRaceIds.Count < 5)
            {
                throw new InvalidOperationException(
                    $"Not enough eligible races from {minimumRaceDate:yyyy-MM-dd} to run the sweep.");
            }

            int testCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * testFraction));
            int devCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * devFraction));
            if (testCount + devCount >= allRaceIds.Count)
            {
                devCount = Math.Max(1, (allRaceIds.Count - testCount) / 2);
            }
            if (testCount + devCount >= allRaceIds.Count)
            {
                throw new InvalidOperationException("Unable to create non-empty train/dev/test splits.");
            }

            int testStartIndex = allRaceIds.Count - testCount;
            int devStartIndex = testStartIndex - devCount;
            var trainingRaceIds = new HashSet<int>(allRaceIds.Take(devStartIndex));
            var devRaceIds = new HashSet<int>(allRaceIds.Skip(devStartIndex).Take(devCount));
            var testRaceIds = new HashSet<int>(allRaceIds.Skip(testStartIndex));
            var trainDevRaceIds = new HashSet<int>(trainingRaceIds);
            trainDevRaceIds.UnionWith(devRaceIds);
            var allIncluded = new HashSet<int>(allRaceIds);

            log($"[sweep] pace features {(trainer.EnablePaceFeatures ? "ON" : "OFF")}; reference parameter {reference.Id}");
            log($"[sweep] split: train={trainingRaceIds.Count}, dev={devRaceIds.Count}, test={testRaceIds.Count} races");

            var full = trainer.LoadEncodedTrainingDataset(
                includeRaceIds: allIncluded,
                includeIdentifiers: true,
                minimumRaceDate: null,
                requireMarketPrice: !keepUnpricedRunners,
                metadataRaceIds: trainingRaceIds,
                marketBlind: marketBlind);
            var devView = HyperparameterTrainer.SplitDataset(full, trainingRaceIds, devRaceIds);
            var testView = HyperparameterTrainer.SplitDataset(full, trainDevRaceIds, testRaceIds);

            int trainRows = devView.TrainingRaces.Sum(r => r.Runners.Count);
            int devRows = devView.ValidationRaces.Sum(r => r.Runners.Count);
            int testRows = testView.ValidationRaces.Sum(r => r.Runners.Count);
            log($"[sweep] rows: train={trainRows}, dev={devRows}, test={testRows}; features={full.FeatureCount}");

            if (trainRows == 0 || devRows == 0 || testRows == 0)
            {
                throw new InvalidOperationException("A train/dev/test split was empty after encoding.");
            }

            int epochBudget = Math.Max(12, (reference.BestEpoch ?? reference.Epochs) + 5);
            var candidates = BuildCandidates(reference, epochBudget);
            if (candidateOffset > 0)
            {
                candidates = candidates.Skip(candidateOffset).ToList();
            }
            if (maxCandidates.HasValue && maxCandidates.Value > 0 && candidates.Count > maxCandidates.Value)
            {
                candidates = candidates.Take(maxCandidates.Value).ToList();
            }

            log($"[sweep] evaluating {candidates.Count} candidates with an epoch budget of {epochBudget}");

            var reports = new List<SweepCandidateReport>(candidates.Count);
            foreach (var candidate in candidates)
            {
                var model = PaceFeatureBenchmark.Clone(reference, reference.Seed ?? configuration.GetValue<int?>("Training:Seed") ?? 42);
                model.RunDate = DateTime.UtcNow;
                model.Folds = 1;
                model.Fold = null;
                model.Units = candidate.Units;
                model.Layers = candidate.Layers;
                model.Dropout = candidate.Dropout;
                model.LearningRate = candidate.LearningRate;
                model.BatchSize = candidate.BatchSize;
                model.RaceBatchSize = candidate.RaceBatchSize;
                model.Epochs = candidate.EpochBudget;
                model.Objective ??= configuration.GetValue<string>("Training:Objective") ?? "RaceSoftmax";
                model.MarketOffsetMode ??= HyperparameterTrainer.NormalizeOffsetMode(
                    configuration.GetValue<string>("Training:MarketOffsetMode") ?? "LogProb");
                model.MarketBlind ??= marketBlind;
                model.WinRateAlpha ??= configuration.GetValue<double?>("WinRateAlpha") ?? 1.0;
                model.WinRateBeta ??= configuration.GetValue<double?>("WinRateBeta") ?? 9.0;
                model.NormalizationMode ??= string.Equals(model.Objective, "Binary", StringComparison.OrdinalIgnoreCase)
                    ? "PerRaceNormalized"
                    : "PerRaceSoftmax";

                log($"[sweep] {candidate.Name}: U{model.Units} L{model.Layers} D{model.Dropout:0.00} LR{model.LearningRate:0.0000} B{model.BatchSize} RB{model.RaceBatchSize}");
                var result = trainer.Train(
                    model,
                    0,
                    1,
                    devView,
                    persistWeights: false,
                    earlyStopOnValidation: true);
                var (devMarketLogLoss, devMarketBrier) =
                    PaceFeatureBenchmark.ComputeMarketMetrics(result.ValidationExamples, commissionRate);

                // Model-only selection. The post-model market blend has not
                // improved on the pure model, so the sweep no longer uses it
                // as a gate or a selection score.
                MarketBlendParameters? devFair = null;
                MarketBlendParameters? devBlend = null;
                bool passedGate = true;
                double selectionScore = result.ValidationBrier;

                var report = new SweepCandidateReport
                {
                    Name = candidate.Name,
                    Units = model.Units,
                    Layers = model.Layers,
                    Dropout = model.Dropout,
                    LearningRate = model.LearningRate,
                    BatchSize = model.BatchSize,
                    RaceBatchSize = model.RaceBatchSize ?? 64,
                    Seed = result.Seed,
                    EpochBudget = model.Epochs,
                    BestEpoch = result.BestEpoch,
                    Temperature = result.ValidationTemperature,
                    DevModelLogLoss = result.ValidationLoss,
                    DevModelBrier = result.ValidationBrier,
                    DevMarketLogLoss = devMarketLogLoss,
                    DevMarketBrier = devMarketBrier,
                    DevFairMarketLogLoss = devFair?.Metrics.LogLoss,
                    DevFairMarketBrier = devFair?.Metrics.Brier,
                    DevBlendLogLoss = devBlend?.Metrics.LogLoss,
                    DevBlendBrier = devBlend?.Metrics.Brier,
                    DevBlendModelWeight = devBlend?.ModelWeight,
                    DevBlendMarketWeight = devBlend?.MarketWeight,
                    PassedMarketGate = passedGate,
                    SelectionScore = selectionScore
                };
                reports.Add(report);

                log($"[sweep] {candidate.Name}: dev LL {report.DevModelLogLoss:F6} Brier {report.DevModelBrier:F6} " +
                    $"blendLL {report.DevBlendLogLoss?.ToString("F6", CultureInfo.InvariantCulture) ?? "n/a"} " +
                    $"gate={(report.PassedMarketGate ? "PASS" : "fail")}");
            }

            var winner = reports
                .OrderBy(c => c.DevModelBrier)
                .ThenBy(c => c.DevModelLogLoss)
                .FirstOrDefault();

            if (winner is not null)
            {
                int chosenEpochs = winner.BestEpoch.HasValue
                    ? Math.Max(1, winner.BestEpoch.Value + 1)
                    : winner.EpochBudget;
                var finalModel = PaceFeatureBenchmark.Clone(reference, winner.Seed);
                finalModel.RunDate = DateTime.UtcNow;
                finalModel.Folds = 1;
                finalModel.Fold = null;
                finalModel.Units = winner.Units;
                finalModel.Layers = winner.Layers;
                finalModel.Dropout = winner.Dropout;
                finalModel.LearningRate = winner.LearningRate;
                finalModel.BatchSize = winner.BatchSize;
                finalModel.RaceBatchSize = winner.RaceBatchSize;
                finalModel.Epochs = chosenEpochs;
                finalModel.Objective ??= configuration.GetValue<string>("Training:Objective") ?? "RaceSoftmax";
                finalModel.MarketOffsetMode ??= HyperparameterTrainer.NormalizeOffsetMode(
                    configuration.GetValue<string>("Training:MarketOffsetMode") ?? "LogProb");
                finalModel.MarketBlind ??= marketBlind;
                finalModel.WinRateAlpha ??= configuration.GetValue<double?>("WinRateAlpha") ?? 1.0;
                finalModel.WinRateBeta ??= configuration.GetValue<double?>("WinRateBeta") ?? 9.0;
                finalModel.NormalizationMode ??= string.Equals(finalModel.Objective, "Binary", StringComparison.OrdinalIgnoreCase)
                    ? "PerRaceNormalized"
                    : "PerRaceSoftmax";

                log($"[sweep] retraining winner {winner.Name} on train+dev with {chosenEpochs} epoch(s)");
                var testResult = trainer.Train(
                    finalModel,
                    0,
                    1,
                    testView,
                    persistWeights: false,
                    temperatureOverride: (float)winner.Temperature,
                    earlyStopOnValidation: false,
                    blendModelWeight: 1d,
                    blendMarketWeight: 0d);
                var (testMarketLogLoss, testMarketBrier) =
                    PaceFeatureBenchmark.ComputeMarketMetrics(testResult.ValidationExamples, commissionRate);

                MarketBlendMetrics? testFair = null;
                MarketBlendMetrics? testBlend = null;
                if (winner.DevFairMarketLogLoss.HasValue &&
                    winner.DevBlendModelWeight.HasValue &&
                    winner.DevBlendMarketWeight.HasValue)
                {
                    testFair = MarketBlendCalibration.Evaluate(
                        testResult.ValidationPredictions,
                        testResult.ValidationLabels,
                        testResult.ValidationRaceIds,
                        testResult.ValidationExamples,
                        0d,
                        winner.DevBlendMarketWeight.Value);
                    testBlend = MarketBlendCalibration.Evaluate(
                        testResult.ValidationPredictions,
                        testResult.ValidationLabels,
                        testResult.ValidationRaceIds,
                        testResult.ValidationExamples,
                        winner.DevBlendModelWeight.Value,
                        winner.DevBlendMarketWeight.Value);
                }

                winner.TestModelLogLoss = testResult.ValidationLoss;
                winner.TestModelBrier = testResult.ValidationBrier;
                winner.TestMarketLogLoss = testMarketLogLoss;
                winner.TestMarketBrier = testMarketBrier;
                winner.TestFairMarketLogLoss = testFair?.LogLoss;
                winner.TestFairMarketBrier = testFair?.Brier;
                winner.TestBlendLogLoss = testBlend?.LogLoss;
                winner.TestBlendBrier = testBlend?.Brier;
                winner.TestTemperature = testResult.ValidationTemperature;
            }

            var sweepReport = new PaceHyperparameterSweepReport
            {
                PaceFeaturesEnabled = trainer.EnablePaceFeatures,
                FeatureCount = full.FeatureCount,
                ReferenceParameterId = reference.Id,
                TrainRaces = trainingRaceIds.Count,
                DevRaces = devRaceIds.Count,
                TestRaces = testRaceIds.Count,
                TrainRows = trainRows,
                DevRows = devRows,
                TestRows = testRows,
                MinimumBlendImprovement = minimumBlendImprovement,
                Candidates = reports,
                Winner = winner
            };

            log(sweepReport.FormatSummary());
            return sweepReport;
        }

        private static List<SweepCandidate> BuildCandidates(MLParameter reference, int epochBudget)
        {
            int units = reference.Units > 0 ? reference.Units : 96;
            int layers = reference.Layers > 0 ? reference.Layers : 2;
            double dropout = reference.Dropout >= 0 ? reference.Dropout : 0.1;
            double learningRate = reference.LearningRate > 0 ? reference.LearningRate : 0.0007;
            int batchSize = reference.BatchSize > 0 ? reference.BatchSize : 500;
            int raceBatchSize = reference.RaceBatchSize ?? 64;

            return new List<SweepCandidate>
            {
                new("base", units, layers, dropout, learningRate, batchSize, raceBatchSize, epochBudget),
                new("units64", 64, layers, dropout, learningRate, batchSize, raceBatchSize, epochBudget),
                new("units128", 128, layers, dropout, learningRate, batchSize, raceBatchSize, epochBudget),
                new("units160", 160, layers, dropout, learningRate, batchSize, raceBatchSize, epochBudget),
                new("layers3", units, 3, dropout, learningRate, batchSize, raceBatchSize, epochBudget),
                new("dropout005", units, layers, 0.05, learningRate, batchSize, raceBatchSize, epochBudget),
                new("dropout015", units, layers, 0.15, learningRate, batchSize, raceBatchSize, epochBudget),
                new("dropout020", units, layers, 0.20, learningRate, batchSize, raceBatchSize, epochBudget),
                new("lr00005", units, layers, dropout, 0.0005, batchSize, raceBatchSize, epochBudget),
                new("lr0001", units, layers, dropout, 0.0010, batchSize, raceBatchSize, epochBudget),
                new("batch256", units, layers, dropout, learningRate, 256, raceBatchSize, epochBudget),
                new("batch1024", units, layers, dropout, learningRate, 1024, raceBatchSize, epochBudget),
                new("u128-d015", 128, layers, 0.15, learningRate, batchSize, raceBatchSize, epochBudget),
                new("u64-lr0001", 64, layers, dropout, 0.0010, batchSize, raceBatchSize, epochBudget),
                new("l3-d015", units, 3, 0.15, learningRate, batchSize, raceBatchSize, epochBudget),
                new("b256-lr00005", units, layers, dropout, 0.0005, 256, raceBatchSize, epochBudget),
                new("racebatch32", units, layers, dropout, learningRate, batchSize, 32, epochBudget),
                new("racebatch128", units, layers, dropout, learningRate, batchSize, 128, epochBudget),
                // Stronger regularisation for the high-cardinality course
                // and course-distance dimensions.
                new("units192", 192, layers, dropout, learningRate, batchSize, raceBatchSize, epochBudget),
                new("units256", 256, layers, dropout, learningRate, batchSize, raceBatchSize, epochBudget),
                new("dropout025", units, layers, 0.25, learningRate, batchSize, raceBatchSize, epochBudget),
                new("dropout030", units, layers, 0.30, learningRate, batchSize, raceBatchSize, epochBudget),
                new("u128-d020", 128, layers, 0.20, learningRate, batchSize, raceBatchSize, epochBudget),
                new("u160-d020", 160, layers, 0.20, learningRate, batchSize, raceBatchSize, epochBudget),
                new("l3-d020", units, 3, 0.20, learningRate, batchSize, raceBatchSize, epochBudget),
                new("l3-d025", units, 3, 0.25, learningRate, batchSize, raceBatchSize, epochBudget),
                new("lr0003", units, layers, dropout, 0.0003, batchSize, raceBatchSize, epochBudget),
                new("u128-l3-d025", 128, 3, 0.25, learningRate, batchSize, raceBatchSize, epochBudget)
            };
        }

        private readonly record struct SweepCandidate(
            string Name,
            int Units,
            int Layers,
            double Dropout,
            double LearningRate,
            int BatchSize,
            int RaceBatchSize,
            int EpochBudget);
    }
}
