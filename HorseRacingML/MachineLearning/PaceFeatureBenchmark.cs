using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.ML
{
    public sealed class PaceFeatureBenchmarkReport
    {
        public int ParameterId { get; init; }
        public bool PaceFeaturesEnabled { get; init; }
        public string FeatureBlocks { get; init; } = "Pace";
        public int FeatureCount { get; init; }
        public int TrainRaces { get; init; }
        public int DevRaces { get; init; }
        public int TestRaces { get; init; }
        public int TrainRows { get; init; }
        public int DevRows { get; init; }
        public int TestRows { get; init; }
        public int? BestEpoch { get; init; }
        public double Temperature { get; init; }

        public double DevModelLogLoss { get; init; }
        public double DevModelBrier { get; init; }
        public double DevMarketLogLoss { get; init; }
        public double DevMarketBrier { get; init; }
        public double? DevFairMarketLogLoss { get; init; }
        public double? DevFairMarketBrier { get; init; }
        public double? DevBlendLogLoss { get; init; }
        public double? DevBlendBrier { get; init; }
        public double? DevBlendModelWeight { get; init; }
        public double? DevBlendMarketWeight { get; init; }

        public double TestModelLogLoss { get; init; }
        public double TestModelBrier { get; init; }
        public double TestMarketLogLoss { get; init; }
        public double TestMarketBrier { get; init; }
        public double? TestFairMarketLogLoss { get; init; }
        public double? TestFairMarketBrier { get; init; }
        public double? TestBlendLogLoss { get; init; }
        public double? TestBlendBrier { get; init; }

        public double? ReferenceDevBrier { get; init; }
        public double? ReferenceDevLogLoss { get; init; }
        public double? ReferenceTestBrier { get; init; }
        public double? ReferenceTestLogLoss { get; init; }
        public double? DevBrierDelta { get; init; }
        public double? DevLogLossDelta { get; init; }
        public double? TestBrierDelta { get; init; }
        public double? TestLogLossDelta { get; init; }
        public string? ModelArtifactFileName { get; init; }
        public string? ModelHash { get; init; }
        public string? FeatureSchemaHash { get; init; }

        public string FormatSummary()
        {
            static string D(double? value) => value.HasValue
                ? value.Value.ToString("F6", CultureInfo.InvariantCulture)
                : "n/a";
            static string Delta(double? value) => value.HasValue
                ? value.Value.ToString("+0.000000;-0.000000;0.000000", CultureInfo.InvariantCulture)
                : "n/a";

            var lines = new List<string>
            {
                $"Feature blocks: {FeatureBlocks} (parameter {ParameterId}, {FeatureCount} features)",
                $"Races train/dev/test: {TrainRaces}/{DevRaces}/{TestRaces}; rows: {TrainRows}/{DevRows}/{TestRows}; best epoch: {BestEpoch?.ToString(CultureInfo.InvariantCulture) ?? "n/a"}; T={Temperature:F3}",
                $"DEV  model LL {DevModelLogLoss:F6} Brier {DevModelBrier:F6} | closing market LL {DevMarketLogLoss:F6} Brier {DevMarketBrier:F6} | fair market LL {D(DevFairMarketLogLoss)} | blend LL {D(DevBlendLogLoss)} Brier {D(DevBlendBrier)}",
                $"TEST model LL {TestModelLogLoss:F6} Brier {TestModelBrier:F6} | closing market LL {TestMarketLogLoss:F6} Brier {TestMarketBrier:F6} | fair market LL {D(TestFairMarketLogLoss)} | blend LL {D(TestBlendLogLoss)} Brier {D(TestBlendBrier)}",
                $"REFERENCE parameter {ParameterId}: dev Brier {D(ReferenceDevBrier)} LL {D(ReferenceDevLogLoss)}; test Brier {D(ReferenceTestBrier)} LL {D(ReferenceTestLogLoss)}",
                $"DELTA vs reference: dev Brier {Delta(DevBrierDelta)} LL {Delta(DevLogLossDelta)}; test Brier {Delta(TestBrierDelta)} LL {Delta(TestLogLossDelta)}"
            };

            return string.Join(Environment.NewLine, lines);
        }
    }

    /// <summary>
    /// Runs one controlled benchmark of a saved hyperparameter combination using
    /// the same chronological train/dev/test split as TrainAI. It never
    /// persists weights; the caller can compare a run with
    /// Training:EnablePaceFeatures=false against a run with it enabled.
    /// </summary>
    public static class PaceFeatureBenchmark
    {
        public static PaceFeatureBenchmarkReport Run(
            IConfiguration configuration,
            RacingRepository repository,
            HyperparameterTrainer trainer,
            MLParameter reference,
            Action<string>? log = null,
            bool persistWeights = false,
            bool activateModel = false)
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
            bool marketBlind = reference.MarketBlind ?? configuration.GetValue<bool?>("Training:MarketBlind") ?? false;

            var allRaceIds = requirePreOffPrice
                ? repository.GetRaceIdsWithOpeningPrice(minimumRaceDate, DateTime.Today)
                : repository.GetRaceIdsBetweenDates(minimumRaceDate, DateTime.Today);
            if (allRaceIds.Count < 5)
            {
                throw new InvalidOperationException(
                    $"Not enough eligible races from {minimumRaceDate:yyyy-MM-dd} to run the benchmark.");
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

            var featureBlocks = FeatureBlockNames.Format(trainer.EnabledFeatureBlocks);
            log($"[benchmark] parameter {reference.Id}: feature blocks {featureBlocks}");
            log($"[benchmark] split: train={trainingRaceIds.Count}, dev={devRaceIds.Count}, test={testRaceIds.Count} races");

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
            log($"[benchmark] rows: train={trainRows}, dev={devRows}, test={testRows}; features={full.FeatureCount}");

            if (trainRows == 0 || devRows == 0 || testRows == 0)
            {
                throw new InvalidOperationException("A train/dev/test split was empty after encoding.");
            }

            var model = Clone(reference, reference.Seed ?? configuration.GetValue<int?>("Training:Seed") ?? 42);
            model.RunDate = DateTime.UtcNow;
            model.Folds = 1;
            model.Fold = null;
            model.Objective ??= configuration.GetValue<string>("Training:Objective") ?? "RaceSoftmax";
            model.MarketOffsetMode ??= HyperparameterTrainer.NormalizeOffsetMode(
                configuration.GetValue<string>("Training:MarketOffsetMode") ?? "LogProb");
            model.MarketBlind ??= marketBlind;
            model.RaceBatchSize ??= Math.Max(1, configuration.GetValue<int?>("Training:RaceBatchSize") ?? 64);
            model.WinRateAlpha ??= configuration.GetValue<double?>("WinRateAlpha") ?? 1.0;
            model.WinRateBeta ??= configuration.GetValue<double?>("WinRateBeta") ?? 9.0;
            model.NormalizationMode ??= string.Equals(model.Objective, "Binary", StringComparison.OrdinalIgnoreCase)
                ? "PerRaceNormalized"
                : "PerRaceSoftmax";
            // The saved best row is the final model, so its Epochs is already
            // BestEpoch + 1. Give the dev pass the same early-stopping room as
            // the original sweep instead of stopping exactly at the frozen
            // epoch count.
            model.Epochs = Math.Max(
                Math.Max(1, reference.Epochs + 10),
                reference.BestEpoch.HasValue ? reference.BestEpoch.Value + 1 : 1);

            log($"[benchmark] training dev model ({model.Layers} layers, {model.Units} units, dropout {model.Dropout:F3}, lr {model.LearningRate}, epochs {model.Epochs})");
            var devResult = trainer.Train(
                model,
                0,
                1,
                devView,
                persistWeights: false,
                earlyStopOnValidation: true);
            var (devMarketLogLoss, devMarketBrier) = ComputeMarketMetrics(devResult.ValidationExamples, commissionRate);

            MarketBlendParameters? devFair = null;
            MarketBlendParameters? devBlend = null;
            if (devResult.ValidationExamples.Count > 0)
            {
                devFair = MarketBlendCalibration.FitMarketOnly(
                    devResult.ValidationPredictions,
                    devResult.ValidationLabels,
                    devResult.ValidationRaceIds,
                    devResult.ValidationExamples);
                devBlend = MarketBlendCalibration.Fit(
                    devResult.ValidationPredictions,
                    devResult.ValidationLabels,
                    devResult.ValidationRaceIds,
                    devResult.ValidationExamples);
            }

            int chosenEpochs = devResult.BestEpoch.HasValue
                ? Math.Max(1, devResult.BestEpoch.Value + 1)
                : model.Epochs;
            var finalModel = Clone(model, devResult.Seed);
            finalModel.Epochs = chosenEpochs;
            finalModel.FeatureBlocks = FeatureBlockNames.Format(trainer.EnabledFeatureBlocks);
            finalModel.BlendModelWeight = 1d;
            finalModel.BlendMarketWeight = 0d;

            log($"[benchmark] retraining on train+dev with {chosenEpochs} epoch(s); temperature {devResult.ValidationTemperature:F4}");
            var testResult = trainer.Train(
                finalModel,
                0,
                1,
                testView,
                persistWeights: persistWeights,
                temperatureOverride: (float)devResult.ValidationTemperature,
                earlyStopOnValidation: false,
                blendModelWeight: 1d,
                blendMarketWeight: 0d,
                activateModel: activateModel);
            var (testMarketLogLoss, testMarketBrier) = ComputeMarketMetrics(testResult.ValidationExamples, commissionRate);

            MarketBlendMetrics? testFair = null;
            MarketBlendMetrics? testBlend = null;
            if (devFair is not null && devBlend is not null)
            {
                testFair = MarketBlendCalibration.Evaluate(
                    testResult.ValidationPredictions,
                    testResult.ValidationLabels,
                    testResult.ValidationRaceIds,
                    testResult.ValidationExamples,
                    0d,
                    devFair.MarketWeight);
                testBlend = MarketBlendCalibration.Evaluate(
                    testResult.ValidationPredictions,
                    testResult.ValidationLabels,
                    testResult.ValidationRaceIds,
                    testResult.ValidationExamples,
                    devBlend.ModelWeight,
                    devBlend.MarketWeight);
            }

            var report = new PaceFeatureBenchmarkReport
            {
                ParameterId = reference.Id,
                PaceFeaturesEnabled = trainer.EnablePaceFeatures,
                FeatureBlocks = featureBlocks,
                FeatureCount = full.FeatureCount,
                TrainRaces = trainingRaceIds.Count,
                DevRaces = devRaceIds.Count,
                TestRaces = testRaceIds.Count,
                TrainRows = trainRows,
                DevRows = devRows,
                TestRows = testRows,
                BestEpoch = devResult.BestEpoch,
                Temperature = devResult.ValidationTemperature,
                DevModelLogLoss = devResult.ValidationLoss,
                DevModelBrier = devResult.ValidationBrier,
                DevMarketLogLoss = devMarketLogLoss,
                DevMarketBrier = devMarketBrier,
                DevFairMarketLogLoss = devFair?.Metrics.LogLoss,
                DevFairMarketBrier = devFair?.Metrics.Brier,
                DevBlendLogLoss = devBlend?.Metrics.LogLoss,
                DevBlendBrier = devBlend?.Metrics.Brier,
                DevBlendModelWeight = devBlend?.ModelWeight,
                DevBlendMarketWeight = devBlend?.MarketWeight,
                TestModelLogLoss = testResult.ValidationLoss,
                TestModelBrier = testResult.ValidationBrier,
                TestMarketLogLoss = testMarketLogLoss,
                TestMarketBrier = testMarketBrier,
                TestFairMarketLogLoss = testFair?.LogLoss,
                TestFairMarketBrier = testFair?.Brier,
                TestBlendLogLoss = testBlend?.LogLoss,
                TestBlendBrier = testBlend?.Brier,
                ReferenceDevBrier = reference.DevBrier,
                ReferenceDevLogLoss = reference.DevLoss,
                ReferenceTestBrier = reference.TestBrier,
                ReferenceTestLogLoss = reference.TestLoss,
                DevBrierDelta = reference.DevBrier.HasValue
                    ? devResult.ValidationBrier - reference.DevBrier.Value
                    : null,
                DevLogLossDelta = reference.DevLoss.HasValue
                    ? devResult.ValidationLoss - reference.DevLoss.Value
                    : null,
                TestBrierDelta = reference.TestBrier.HasValue
                    ? testResult.ValidationBrier - reference.TestBrier.Value
                    : null,
                TestLogLossDelta = reference.TestLoss.HasValue
                    ? testResult.ValidationLoss - reference.TestLoss.Value
                    : null,
                ModelArtifactFileName = testResult.ModelArtifactFileName,
                ModelHash = testResult.ModelHash,
                FeatureSchemaHash = testResult.FeatureSchemaHash
            };

            log(report.FormatSummary());
            return report;
        }

        internal static MLParameter Clone(MLParameter source, int seed)
        {
            return new MLParameter
            {
                Id = source.Id,
                RunDate = source.RunDate,
                EarliestTrainingDate = source.EarliestTrainingDate,
                Units = source.Units,
                Dropout = source.Dropout,
                Layers = source.Layers,
                LearningRate = source.LearningRate,
                Epochs = source.Epochs,
                BatchSize = source.BatchSize,
                Folds = 1,
                Fold = null,
                Objective = source.Objective,
                MarketOffsetMode = source.MarketOffsetMode,
                MarketBlind = source.MarketBlind,
                BlendModelWeight = source.BlendModelWeight,
                BlendMarketWeight = source.BlendMarketWeight,
                RaceBatchSize = source.RaceBatchSize,
                WinRateAlpha = source.WinRateAlpha,
                WinRateBeta = source.WinRateBeta,
                NormalizationMode = source.NormalizationMode,
                Seed = seed
            };
        }

        internal static (double LogLoss, double Brier) ComputeMarketMetrics(
            IReadOnlyList<HyperparameterTrainer.RunnerExample>? examples,
            decimal commissionRate)
        {
            if (examples is null || examples.Count == 0)
            {
                return (0d, 0d);
            }

            double logLoss = 0d;
            double brier = 0d;
            int count = 0;

            foreach (var race in examples.GroupBy(e => e.RaceId))
            {
                var entries = race
                    .Where(e => e.StartingPriceDecimal.HasValue && e.StartingPriceDecimal.Value > 1m)
                    .Select(e => new
                    {
                        e.Label,
                        Odds = BettingMath.CalculateEffectiveDecimalOdds(
                            e.StartingPriceDecimal!.Value,
                            commissionRate)
                    })
                    .ToList();

                if (entries.Count == 0)
                {
                    continue;
                }

                double sum = entries.Sum(e => 1.0 / (double)e.Odds);
                if (sum <= double.Epsilon || !double.IsFinite(sum))
                {
                    continue;
                }

                foreach (var entry in entries)
                {
                    double probability = Math.Clamp(
                        1.0 / ((double)entry.Odds * sum),
                        1e-7,
                        1d - 1e-7);
                    double label = entry.Label;
                    logLoss -= label * Math.Log(probability) + (1d - label) * Math.Log(1d - probability);
                    brier += (probability - label) * (probability - label);
                    count++;
                }
            }

            return count == 0
                ? (0d, 0d)
                : (logLoss / count, brier / count);
        }
    }
}
