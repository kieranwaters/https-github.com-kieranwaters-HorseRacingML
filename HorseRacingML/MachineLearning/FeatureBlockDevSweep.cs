using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Dev-only feature-block ablation against the active feature schema.
    /// The dataset is encoded once with every block enabled; the baseline keeps
    /// only the blocks active in the reference parameter and each candidate adds
    /// one block. No test retraining and no artifact persistence.
    /// </summary>
    public static class FeatureBlockDevSweep
    {
        public sealed class CandidateReport
        {
            public string Block { get; init; } = string.Empty;
            public int FeatureKeyCount { get; init; }
            public int FeatureCount { get; init; }
            public double DevRaceLogLoss { get; init; }
            public double DevTop1Accuracy { get; init; }
            public double DevTop3Accuracy { get; init; }
            public double DevRunnerLogLoss { get; init; }
            public double DevBrier { get; init; }
            public double FullFieldRunnerLogLoss { get; init; }
            public double FullFieldBrier { get; init; }
            public int? BestEpoch { get; init; }
            public double Temperature { get; init; }
            public int RaceCount { get; init; }
            public int RunnerCount { get; init; }
        }

        public sealed class Report
        {
            public DateTime DevStart { get; init; }
            public DateTime DevEnd { get; init; }
            public int TrainRaces { get; init; }
            public int DevRaces { get; init; }
            public int TrainRows { get; init; }
            public int DevRows { get; init; }
            public string ActiveBlocks { get; init; } = string.Empty;
            public List<CandidateReport> Candidates { get; init; } = new();
        }

        public static Report Run(
            IConfiguration configuration,
            RacingRepository repository,
            MLParameter template,
            DateTime devStart,
            DateTime devEnd,
            IReadOnlyList<FeatureBlocks> blocks,
            int seed,
            Action<string>? log = null,
            Action<Report>? progress = null,
            Report? resume = null)
        {
            if (configuration is null) throw new ArgumentNullException(nameof(configuration));
            if (repository is null) throw new ArgumentNullException(nameof(repository));
            if (template is null) throw new ArgumentNullException(nameof(template));
            log ??= Console.WriteLine;

            var trainingStart = template.TrainingStartDate
                ?? configuration.GetValue<DateTime?>("Training:MinimumRaceDate")
                ?? new DateTime(2017, 1, 1);
            var trainIds = repository.GetRaceIdsBetweenDates(trainingStart, devStart.AddDays(-1));
            var devIds = repository.GetRaceIdsBetweenDates(devStart, devEnd);
            var trainSet = new HashSet<int>(trainIds);
            var devSet = new HashSet<int>(devIds);
            var allSet = new HashSet<int>(trainSet);
            allSet.UnionWith(devSet);

            var allConfig = new ConfigurationBuilder()
                .AddConfiguration(configuration)
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["Training:FeatureBlocks"] = "all"
                })
                .Build();
            var allTrainer = new HyperparameterTrainer(allConfig);
            log($"[blocks-dev] split: train {trainSet.Count}, dev {devSet.Count}; loading all blocks once");
            var full = allTrainer.LoadEncodedTrainingDataset(
                includeRaceIds: allSet,
                includeIdentifiers: true,
                minimumRaceDate: null,
                requireMarketPrice: false,
                metadataRaceIds: trainSet,
                marketBlind: false);
            int trainRows = full.TrainingRaces.Sum(race => race.Runners.Count);
            int devRows = full.ValidationRaces.Sum(race => race.Runners.Count);
            log($"[blocks-dev] features {full.FeatureCount}; rows train {trainRows}, dev {devRows}");

            var activeBlocks = FeatureBlockNames.Parse(
                template.FeatureBlocks,
                FeatureBlocks.Default);
            var blockFor = new Dictionary<string, FeatureBlocks>(StringComparer.OrdinalIgnoreCase);
            foreach (var key in full.FeatureKeys)
            {
                var matched = FeatureBlocks.None;
                foreach (var block in FeatureBlockNames.IndividualBlocks())
                {
                    if (HyperparameterTrainer.IsFeatureInBlock(key, block))
                    {
                        matched = block;
                        break;
                    }
                }

                blockFor[key] = matched;
            }

            var baselineKeys = full.FeatureKeys
                .Where(key =>
                {
                    var matched = blockFor[key];
                    return matched == FeatureBlocks.None || activeBlocks.HasFlag(matched);
                })
                .ToList();
            log($"[blocks-dev] active schema {FeatureBlockNames.Format(activeBlocks)}: {baselineKeys.Count} feature keys");

            var candidates = new List<CandidateReport>();
            Report BuildReport() => new Report
            {
                DevStart = devStart,
                DevEnd = devEnd,
                TrainRaces = trainSet.Count,
                DevRaces = devSet.Count,
                TrainRows = trainRows,
                DevRows = devRows,
                ActiveBlocks = FeatureBlockNames.Format(activeBlocks),
                Candidates = candidates.ToList()
            };

            var resumed = resume?.Candidates ?? new List<CandidateReport>();
            var baseline = resumed.FirstOrDefault(candidate =>
                string.Equals(candidate.Block, "Baseline", StringComparison.OrdinalIgnoreCase));
            if (baseline is null)
            {
                baseline = TrainCandidate(
                    allTrainer,
                    full,
                    baselineKeys,
                    trainSet,
                    devSet,
                    template,
                    seed,
                    "Baseline",
                    log);
                candidates.Add(baseline);
                LogMemory(log, "baseline");
                progress?.Invoke(BuildReport());
            }
            else
            {
                candidates.Add(baseline);
                foreach (var existing in resumed.Where(candidate =>
                    !string.Equals(candidate.Block, "Baseline", StringComparison.OrdinalIgnoreCase)))
                {
                    candidates.Add(existing);
                }

                log($"[blocks-dev] resumed {candidates.Count - 1} completed candidate(s) from checkpoint");
            }

            foreach (var block in blocks)
            {
                var blockName = FeatureBlockNames.Format(block);
                if (candidates.Any(candidate =>
                    string.Equals(candidate.Block, blockName, StringComparison.OrdinalIgnoreCase)))
                {
                    log($"[blocks-dev] {blockName}: already present in checkpoint; skipping");
                    continue;
                }

                var blockKeys = full.FeatureKeys
                    .Where(key => HyperparameterTrainer.IsFeatureInBlock(key, block))
                    .ToList();
                if (blockKeys.Count == 0)
                {
                    log($"[blocks-dev] {block}: no columns discovered");
                    continue;
                }

                var keys = new HashSet<string>(baselineKeys, StringComparer.OrdinalIgnoreCase);
                keys.UnionWith(blockKeys);
                var orderedKeys = full.FeatureKeys.Where(keys.Contains).ToList();
                var candidate = TrainCandidate(
                    allTrainer,
                    full,
                    orderedKeys,
                    trainSet,
                    devSet,
                    template,
                    seed,
                    blockName,
                    log);
                candidates.Add(candidate);
                LogMemory(log, blockName);
                progress?.Invoke(BuildReport());
            }

            var report = BuildReport();

            log("[blocks-dev] summary:");
            foreach (var candidate in candidates)
            {
                var baselineDelta = candidate.Block == "Baseline"
                    ? 0
                    : candidate.DevRaceLogLoss - baseline.DevRaceLogLoss;
                log(
                    $"[blocks-dev] {candidate.Block}: race LL {candidate.DevRaceLogLoss:F4} " +
                    $"(delta {baselineDelta:+0.0000;-0.0000;0.0000}), top1 {candidate.DevTop1Accuracy:P2}, " +
                    $"top3 {candidate.DevTop3Accuracy:P2}, runner LL {candidate.DevRunnerLogLoss:F4}, " +
                    $"Brier {candidate.DevBrier:F4}, epoch {candidate.BestEpoch}");
            }

            return report;
        }

        private static void LogMemory(Action<string> log, string label)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            using var process = Process.GetCurrentProcess();
            log($"[blocks-dev] {label}: managed memory {GC.GetTotalMemory(false) / 1024d / 1024d:F0} MB, working set {process.WorkingSet64 / 1024d / 1024d:F0} MB");
        }

        private static CandidateReport TrainCandidate(
            HyperparameterTrainer trainer,
            HyperparameterTrainer.TrainingDataset full,
            IList<string> keys,
            ISet<int> trainSet,
            ISet<int> devSet,
            MLParameter template,
            int seed,
            string name,
            Action<string> log)
        {
            var subset = FeatureBlockBenchmark.SubsetFeatures(full, keys);
            var view = HyperparameterTrainer.SplitDataset(subset, trainSet, devSet);
            var parameter = new MLParameter
            {
                RunDate = DateTime.UtcNow,
                Units = template.Units,
                Layers = template.Layers,
                Dropout = template.Dropout,
                LearningRate = template.LearningRate,
                Epochs = Math.Max(10, (template.BestEpoch ?? 0) + 5),
                BatchSize = template.BatchSize,
                Folds = 1,
                Objective = "RaceSoftmax",
                MarketOffsetMode = template.MarketOffsetMode ?? "LogProb",
                MarketBlind = template.MarketBlind ?? false,
                RaceBatchSize = template.RaceBatchSize ?? 64,
                WinRateAlpha = template.WinRateAlpha,
                WinRateBeta = template.WinRateBeta,
                FeatureBlocks = template.FeatureBlocks,
                NormalizationMode = "PerRaceSoftmax",
                Seed = seed
            };

            log($"[blocks-dev] {name}: {keys.Count} keys / {subset.FeatureCount} dimensions; training");
            var result = trainer.Train(
                parameter,
                0,
                1,
                view,
                persistWeights: false,
                earlyStopOnValidation: true);
            var priced = ComputePricedMetrics(result);
            log(
                $"[blocks-dev] {name}: race LL {priced.RaceLogLoss:F4}, top1 {priced.Top1Accuracy:P2}, " +
                $"top3 {priced.Top3Accuracy:P2}, runner LL {priced.RunnerLogLoss:F4}, Brier {priced.Brier:F4}, " +
                $"epoch {result.BestEpoch}");

            return new CandidateReport
            {
                Block = name,
                FeatureKeyCount = keys.Count,
                FeatureCount = subset.FeatureCount,
                DevRaceLogLoss = priced.RaceLogLoss,
                DevTop1Accuracy = priced.Top1Accuracy,
                DevTop3Accuracy = priced.Top3Accuracy,
                DevRunnerLogLoss = priced.RunnerLogLoss,
                DevBrier = priced.Brier,
                FullFieldRunnerLogLoss = result.ValidationLoss,
                FullFieldBrier = result.ValidationBrier,
                BestEpoch = result.BestEpoch,
                Temperature = result.ValidationTemperature,
                RaceCount = priced.RaceCount,
                RunnerCount = priced.RunnerCount
            };
        }

        private static RaceEvaluationMetrics ComputePricedMetrics(HyperparameterTrainer.TrainingResult result)
        {
            var predictions = new List<float>();
            var labels = new List<float>();
            var raceIds = new List<int>();
            int count = Math.Min(
                result.ValidationPredictions.Count,
                Math.Min(
                    result.ValidationLabels.Count,
                    Math.Min(result.ValidationRaceIds.Count, result.ValidationExamples.Count)));
            for (int i = 0; i < count; i++)
            {
                if (!result.ValidationExamples[i].HasMarketPrice)
                {
                    continue;
                }

                predictions.Add(result.ValidationPredictions[i]);
                labels.Add(result.ValidationLabels[i]);
                raceIds.Add(result.ValidationRaceIds[i]);
            }

            return RaceMetrics.Compute(raceIds, predictions, labels);
        }
    }
}
