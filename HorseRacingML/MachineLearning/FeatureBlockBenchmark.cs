using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using static HorseRacingML.ML.HyperparameterTrainer;

namespace HorseRacingML.ML
{
    public sealed class FeatureBlockBenchmarkRow
    {
        public string Block { get; init; } = "Baseline";
        public int FeatureCount { get; init; }
        public int FeatureKeyCount { get; init; }
        public int TrainRaces { get; init; }
        public int DevRaces { get; init; }
        public int TestRaces { get; init; }
        public int TrainRows { get; init; }
        public int DevRows { get; init; }
        public int TestRows { get; init; }
        public int? BestEpoch { get; init; }
        public double Temperature { get; init; }
        public double DevBrier { get; init; }
        public double DevLogLoss { get; init; }
        public double TestBrier { get; init; }
        public double TestLogLoss { get; init; }
        public double DevMarketBrier { get; init; }
        public double DevMarketLogLoss { get; init; }
        public double TestMarketBrier { get; init; }
        public double TestMarketLogLoss { get; init; }
        public double? DevBlendBrier { get; init; }
        public double? DevBlendLogLoss { get; init; }
        public double? TestBlendBrier { get; init; }
        public double? TestBlendLogLoss { get; init; }
        public double? DevBrierDeltaVsBaseline { get; set; }
        public double? TestBrierDeltaVsBaseline { get; set; }
        public double? DevLogLossDeltaVsBaseline { get; set; }
        public double? TestLogLossDeltaVsBaseline { get; set; }
        public double? DevBlendLogLossDeltaVsBaseline { get; set; }
        public double? TestBlendLogLossDeltaVsBaseline { get; set; }
    }

    public sealed class FeatureBlockBenchmarkReport
    {
        public int ParameterId { get; init; }
        public string BaselineFeatureBlocks { get; init; } = "Pace,MarketMovement";
        public IReadOnlyList<FeatureBlockBenchmarkRow> Rows { get; init; } =
            Array.Empty<FeatureBlockBenchmarkRow>();

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
                $"Feature-block benchmark (parameter {ParameterId}); each row = {BaselineFeatureBlocks} + block",
                "block            dims      devBrier   devLL      dDevBrier  devBlendLL dDevBlendLL  testBrier  testLL     dTestBrier testBlendLL dTestBlendLL"
            };

            foreach (var row in Rows)
            {
                lines.Add(
                    $"{row.Block,-16} {row.FeatureCount,8}  {D(row.DevBrier),10} {D(row.DevLogLoss),10} " +
                    $"{Delta(row.DevBrierDeltaVsBaseline),10} {D(row.DevBlendLogLoss),10} " +
                    $"{Delta(row.DevBlendLogLossDeltaVsBaseline),11} {D(row.TestBrier),10} " +
                    $"{D(row.TestLogLoss),10} {Delta(row.TestBrierDeltaVsBaseline),10} " +
                    $"{D(row.TestBlendLogLoss),11} {Delta(row.TestBlendLogLossDeltaVsBaseline),11}");
            }

            return string.Join(Environment.NewLine, lines);
        }
    }

    /// <summary>
    /// Loads one encoded dataset with every block enabled, then trains
    /// each ablation on a feature-column subset of that dataset. Loading the
    /// history is by far the most expensive part of a benchmark, so this keeps
    /// the comparison controlled and avoids re-reading the database for every
    /// block.
    /// </summary>
    public static class FeatureBlockBenchmark
    {
        public static FeatureBlockBenchmarkReport Run(
            IConfiguration configuration,
            RacingRepository repository,
            HyperparameterTrainer trainer,
            MLParameter reference,
            Action<string>? log = null,
            IReadOnlyCollection<string>? onlyBlocks = null)
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

            log($"[blocks] parameter {reference.Id}: loading history with all blocks for {allIncluded.Count} races");
            var full = trainer.LoadEncodedTrainingDataset(
                includeRaceIds: allIncluded,
                includeIdentifiers: true,
                minimumRaceDate: null,
                requireMarketPrice: !keepUnpricedRunners,
                metadataRaceIds: trainingRaceIds,
                marketBlind: marketBlind);

            foreach (var key in full.FeatureKeys.Where(k => k.StartsWith("RaceContext", StringComparison.OrdinalIgnoreCase)))
            {
                log($"[blocks] schema {key}: {full.FeatureDimensions[key]} dimension(s)");
            }

            var newBlockKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var block in FeatureBlockNames.IndividualBlocks())
            {
                if (block == FeatureBlocks.Pace || block == FeatureBlocks.MarketMovement)
                {
                    continue;
                }

                foreach (var key in full.FeatureKeys)
                {
                    if (HyperparameterTrainer.IsFeatureInBlock(key, block))
                    {
                        newBlockKeys.Add(key);
                    }
                }
            }

            // MarketMovement is now part of the baseline: it was the only
            // block that improved both dev and test in the first ablation.
            // Keep its columns in every subsequent comparison so each new
            // block is judged on top of the current best feature set.
            var baselineKeys = full.FeatureKeys
                .Where(key => !newBlockKeys.Contains(key))
                .ToList();

            var blockOrder = new (string Name, FeatureBlocks Block, HashSet<string> Keys)[]
            {
                ("Baseline", FeatureBlocks.None, new HashSet<string>(StringComparer.OrdinalIgnoreCase)),
                ("TrainerForm", FeatureBlocks.TrainerForm, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.TrainerForm)),
                ("WeatherBalance", FeatureBlocks.WeatherBalance, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.WeatherBalance)),
                ("RaceContext", FeatureBlocks.RaceContext, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.RaceContext)),
                ("FormContext", FeatureBlocks.FormContext, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.FormContext)),
                ("PriceData", FeatureBlocks.PriceData, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.PriceData)),
                ("WeatherExtended", FeatureBlocks.WeatherExtended, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.WeatherExtended)),
                ("TrackGeometry", FeatureBlocks.TrackGeometry, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.TrackGeometry)),
                ("DeepForm", FeatureBlocks.DeepForm, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.DeepForm)),
                ("MarketLiquidity", FeatureBlocks.MarketLiquidity, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.MarketLiquidity)),
                ("Sectionals", FeatureBlocks.Sectionals, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.Sectionals)),
                ("Breeding", FeatureBlocks.Breeding, CollectBlockKeys(full.FeatureKeys, FeatureBlocks.Breeding))
            };

            var requestedBlocks = onlyBlocks is null || onlyBlocks.Count == 0
                ? null
                : new HashSet<string>(onlyBlocks, StringComparer.OrdinalIgnoreCase);

            var rows = new List<FeatureBlockBenchmarkRow>();
            foreach (var (name, _, blockKeys) in blockOrder)
            {
                if (name != "Baseline" &&
                    requestedBlocks is not null &&
                    !requestedBlocks.Contains(name))
                {
                    continue;
                }

                var keys = new HashSet<string>(baselineKeys, StringComparer.OrdinalIgnoreCase);
                keys.UnionWith(blockKeys);
                var orderedKeys = full.FeatureKeys.Where(keys.Contains).ToList();

                if (name != "Baseline" && blockKeys.Count == 0)
                {
                    log($"[blocks] {name}: no columns were discovered in this dataset; the row will match the baseline.");
                }

                var subset = SubsetFeatures(full, orderedKeys);
                log($"[blocks] {name}: {orderedKeys.Count} keys / {subset.FeatureCount} dimensions");
                var devView = HyperparameterTrainer.SplitDataset(subset, trainingRaceIds, devRaceIds);
                var testView = HyperparameterTrainer.SplitDataset(subset, trainDevRaceIds, testRaceIds);

                int trainRows = devView.TrainingRaces.Sum(r => r.Runners.Count);
                int devRows = devView.ValidationRaces.Sum(r => r.Runners.Count);
                int testRows = testView.ValidationRaces.Sum(r => r.Runners.Count);
                if (trainRows == 0 || devRows == 0 || testRows == 0)
                {
                    throw new InvalidOperationException($"The {name} block produced an empty train/dev/test split.");
                }

                var model = PaceFeatureBenchmark.Clone(
                    reference,
                    reference.Seed ?? configuration.GetValue<int?>("Training:Seed") ?? 42);
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
                model.Epochs = Math.Max(
                    Math.Max(1, reference.Epochs + 10),
                    reference.BestEpoch.HasValue ? reference.BestEpoch.Value + 1 : 1);

                log($"[blocks] {name}: training dev model ({model.Layers} layers, {model.Units} units, {model.Epochs} epochs)");
                var devResult = trainer.Train(
                    model,
                    0,
                    1,
                    devView,
                    persistWeights: false,
                    earlyStopOnValidation: true);
                var (devMarketLogLoss, devMarketBrier) =
                    PaceFeatureBenchmark.ComputeMarketMetrics(devResult.ValidationExamples, commissionRate);

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
                var finalModel = PaceFeatureBenchmark.Clone(model, devResult.Seed);
                finalModel.Epochs = chosenEpochs;
                log($"[blocks] {name}: retraining on train+dev with {chosenEpochs} epochs");
                var testResult = trainer.Train(
                    finalModel,
                    0,
                    1,
                    testView,
                    persistWeights: false,
                    temperatureOverride: (float)devResult.ValidationTemperature,
                    earlyStopOnValidation: false,
                    blendModelWeight: devBlend?.ModelWeight ?? 1d,
                    blendMarketWeight: devBlend?.MarketWeight ?? 0d);
                var (testMarketLogLoss, testMarketBrier) =
                    PaceFeatureBenchmark.ComputeMarketMetrics(testResult.ValidationExamples, commissionRate);

                MarketBlendMetrics? testBlend = null;
                if (devFair is not null && devBlend is not null)
                {
                    testBlend = MarketBlendCalibration.Evaluate(
                        testResult.ValidationPredictions,
                        testResult.ValidationLabels,
                        testResult.ValidationRaceIds,
                        testResult.ValidationExamples,
                        devBlend.ModelWeight,
                        devBlend.MarketWeight);
                }

                rows.Add(new FeatureBlockBenchmarkRow
                {
                    Block = name,
                    FeatureCount = subset.FeatureCount,
                    FeatureKeyCount = orderedKeys.Count,
                    TrainRaces = trainingRaceIds.Count,
                    DevRaces = devRaceIds.Count,
                    TestRaces = testRaceIds.Count,
                    TrainRows = trainRows,
                    DevRows = devRows,
                    TestRows = testRows,
                    BestEpoch = devResult.BestEpoch,
                    Temperature = devResult.ValidationTemperature,
                    DevBrier = devResult.ValidationBrier,
                    DevLogLoss = devResult.ValidationLoss,
                    TestBrier = testResult.ValidationBrier,
                    TestLogLoss = testResult.ValidationLoss,
                    DevMarketBrier = devMarketBrier,
                    DevMarketLogLoss = devMarketLogLoss,
                    TestMarketBrier = testMarketBrier,
                    TestMarketLogLoss = testMarketLogLoss,
                    DevBlendBrier = devBlend?.Metrics.Brier,
                    DevBlendLogLoss = devBlend?.Metrics.LogLoss,
                    TestBlendBrier = testBlend?.Brier,
                    TestBlendLogLoss = testBlend?.LogLoss
                });

                // Each row holds its own feature-vector copy for the duration
                // of the training call. Release it before building the next one.
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            var baseline = rows.First();
            foreach (var row in rows)
            {
                row.DevBrierDeltaVsBaseline = row.DevBrier - baseline.DevBrier;
                row.TestBrierDeltaVsBaseline = row.TestBrier - baseline.TestBrier;
                row.DevLogLossDeltaVsBaseline = row.DevLogLoss - baseline.DevLogLoss;
                row.TestLogLossDeltaVsBaseline = row.TestLogLoss - baseline.TestLogLoss;
                if (row.DevBlendLogLoss.HasValue && baseline.DevBlendLogLoss.HasValue)
                {
                    row.DevBlendLogLossDeltaVsBaseline =
                        row.DevBlendLogLoss.Value - baseline.DevBlendLogLoss.Value;
                }
                if (row.TestBlendLogLoss.HasValue && baseline.TestBlendLogLoss.HasValue)
                {
                    row.TestBlendLogLossDeltaVsBaseline =
                        row.TestBlendLogLoss.Value - baseline.TestBlendLogLoss.Value;
                }
            }

            var report = new FeatureBlockBenchmarkReport
            {
                ParameterId = reference.Id,
                BaselineFeatureBlocks = "Pace,MarketMovement",
                Rows = rows
            };
            log(report.FormatSummary());
            return report;
        }

        private static HashSet<string> CollectBlockKeys(
            IList<string> featureKeys,
            FeatureBlocks block)
        {
            var keys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var key in featureKeys)
            {
                if (HyperparameterTrainer.IsFeatureInBlock(key, block))
                {
                    keys.Add(key);
                }
            }

            return keys;
        }

        internal static HyperparameterTrainer.TrainingDataset SubsetFeatures(
            HyperparameterTrainer.TrainingDataset source,
            IList<string> keys)
        {
            var sourceOffsets = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            int sourceOffset = 0;
            foreach (var key in source.FeatureKeys)
            {
                sourceOffsets[key] = sourceOffset;
                sourceOffset += source.FeatureDimensions[key];
            }

            var dimensions = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var stringMaps = new Dictionary<string, Dictionary<string, int>>(StringComparer.OrdinalIgnoreCase);
            var targetOffsets = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            int featureCount = 0;
            foreach (var key in keys)
            {
                int dimension = source.FeatureDimensions[key];
                dimensions[key] = dimension;
                targetOffsets[key] = featureCount;
                featureCount += dimension;
                if (source.StringMaps.TryGetValue(key, out var map))
                {
                    stringMaps[key] = map;
                }
            }

            var normalization = new NormalizationParameters
            {
                Mean = new float[featureCount],
                StdDev = new float[featureCount]
            };

            List<RaceExample> ConvertRaces(IEnumerable<RaceExample> races)
            {
                var converted = new List<RaceExample>();
                foreach (var race in races)
                {
                    var runners = new List<RunnerExample>(race.Runners.Count);
                    foreach (var runner in race.Runners)
                    {
                        var features = new float[featureCount];
                        foreach (var key in keys)
                        {
                            int dimension = dimensions[key];
                            Array.Copy(
                                runner.Features,
                                sourceOffsets[key],
                                features,
                                targetOffsets[key],
                                dimension);
                        }

                        runners.Add(new RunnerExample(
                            runner.RaceId,
                            features,
                            runner.Label,
                            runner.HorseId,
                            runner.HorseName,
                            runner.StartingPriceDecimal,
                            runner.MarketLogit,
                            runner.HasMarketPrice,
                            runner.MarketProbability,
                            runner.MarketLogProbability,
                            runner.FinishPosition,
                            runner.FieldSize));
                    }

                    converted.Add(new RaceExample(race.RaceId, race.RaceDate, runners));
                }

                return converted;
            }

            return new HyperparameterTrainer.TrainingDataset(
                ConvertRaces(source.TrainingRaces),
                ConvertRaces(source.ValidationRaces),
                keys.ToList(),
                dimensions,
                stringMaps,
                normalization);
        }
    }
}
