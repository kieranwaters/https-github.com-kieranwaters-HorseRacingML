using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Dev-only sweep of the winner-versus-field pairwise ranking weight.
    /// The encoded dataset and the chronological dev split are built once;
    /// every candidate trains on the same rows and is scored on the same dev
    /// rows. No final/test retraining and no artifact persistence.
    /// </summary>
    public static class PairwiseLossSweep
    {
        public sealed class CandidateReport
        {
            public double PairwiseWeight { get; init; }
            public int Seed { get; init; }
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

        public sealed class WeightSummary
        {
            public double PairwiseWeight { get; init; }
            public int CandidateCount { get; init; }
            public double MeanRaceLogLoss { get; init; }
            public double MeanTop1Accuracy { get; init; }
            public double MeanTop3Accuracy { get; init; }
            public double MeanRunnerLogLoss { get; init; }
            public double MeanBrier { get; init; }
        }

        public sealed class Report
        {
            public DateTime DevStart { get; init; }
            public DateTime DevEnd { get; init; }
            public int TrainRaces { get; init; }
            public int DevRaces { get; init; }
            public int TrainRows { get; init; }
            public int DevRows { get; init; }
            public int FeatureCount { get; init; }
            public double ReferenceTop1Accuracy { get; init; }
            public double? BestBrierWeight { get; init; }
            public double? BestLogLossWeight { get; init; }
            public List<WeightSummary> Summaries { get; init; } = new();
            public List<CandidateReport> Candidates { get; init; } = new();
        }

        public static Report Run(
            IConfiguration configuration,
            RacingRepository repository,
            MLParameter template,
            DateTime devStart,
            DateTime devEnd,
            IReadOnlyList<double> pairwiseWeights,
            IReadOnlyList<int> seeds,
            Action<string>? log = null)
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

            log($"[pairwise] split: train {trainSet.Count} races, dev {devSet.Count} races; loading features once");
            var trainer = new HyperparameterTrainer(configuration);
            var full = trainer.LoadEncodedTrainingDataset(
                includeRaceIds: allSet,
                includeIdentifiers: true,
                minimumRaceDate: null,
                requireMarketPrice: false,
                metadataRaceIds: trainSet,
                marketBlind: false);
            var devView = HyperparameterTrainer.SplitDataset(full, trainSet, devSet);
            int trainRows = devView.TrainingRaces.Sum(race => race.Runners.Count);
            int devRows = devView.ValidationRaces.Sum(race => race.Runners.Count);
            log($"[pairwise] features {full.FeatureCount}; rows train {trainRows}, dev {devRows}");

            var candidates = new List<CandidateReport>();
            foreach (double weight in pairwiseWeights)
            {
                var weightConfig = new ConfigurationBuilder()
                    .AddConfiguration(configuration)
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["Training:PairwiseLossWeight"] = weight.ToString(CultureInfo.InvariantCulture)
                    })
                    .Build();
                var weightTrainer = new HyperparameterTrainer(weightConfig);
                foreach (int seed in seeds)
                {
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

                    log($"[pairwise] weight {weight:0.00}, seed {seed}: training");
                    var result = weightTrainer.Train(
                        parameter,
                        0,
                        1,
                        devView,
                        persistWeights: false,
                        earlyStopOnValidation: true);
                    var priced = ComputePricedMetrics(result);
                    candidates.Add(new CandidateReport
                    {
                        PairwiseWeight = weight,
                        Seed = seed,
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
                    });
                    log(
                        $"[pairwise] weight {weight:0.00}, seed {seed}: dev race LL {priced.RaceLogLoss:F4}, " +
                        $"top1 {priced.Top1Accuracy:P2}, top3 {priced.Top3Accuracy:P2}, " +
                        $"runner LL {priced.RunnerLogLoss:F4}, Brier {priced.Brier:F4}, epoch {result.BestEpoch}");
                }
            }

            var summaries = pairwiseWeights
                .Select(weight =>
                {
                    var rows = candidates.Where(candidate => Math.Abs(candidate.PairwiseWeight - weight) < 1e-9).ToList();
                    return new WeightSummary
                    {
                        PairwiseWeight = weight,
                        CandidateCount = rows.Count,
                        MeanRaceLogLoss = rows.Count == 0 ? 0 : rows.Average(row => row.DevRaceLogLoss),
                        MeanTop1Accuracy = rows.Count == 0 ? 0 : rows.Average(row => row.DevTop1Accuracy),
                        MeanTop3Accuracy = rows.Count == 0 ? 0 : rows.Average(row => row.DevTop3Accuracy),
                        MeanRunnerLogLoss = rows.Count == 0 ? 0 : rows.Average(row => row.DevRunnerLogLoss),
                        MeanBrier = rows.Count == 0 ? 0 : rows.Average(row => row.DevBrier)
                    };
                })
                .ToList();

            double referenceTop1 = template.DevAccuracy
                ?? (candidates.Count == 0
                    ? 0
                    : candidates.Max(candidate => candidate.DevTop1Accuracy));
            var eligible = summaries
                .Where(summary => summary.MeanTop1Accuracy >= referenceTop1 - 1e-9)
                .ToList();
            if (eligible.Count == 0)
            {
                eligible = summaries;
            }

            var bestBrier = eligible.OrderBy(summary => summary.MeanBrier).FirstOrDefault();
            var bestLogLoss = eligible.OrderBy(summary => summary.MeanRunnerLogLoss).FirstOrDefault();

            var report = new Report
            {
                DevStart = devStart,
                DevEnd = devEnd,
                TrainRaces = trainSet.Count,
                DevRaces = devSet.Count,
                TrainRows = trainRows,
                DevRows = devRows,
                FeatureCount = full.FeatureCount,
                ReferenceTop1Accuracy = referenceTop1,
                BestBrierWeight = bestBrier?.PairwiseWeight,
                BestLogLossWeight = bestLogLoss?.PairwiseWeight,
                Summaries = summaries,
                Candidates = candidates
            };

            log("[pairwise] summary:");
            foreach (var summary in summaries)
            {
                log(
                    $"[pairwise] w={summary.PairwiseWeight:0.00}: race LL {summary.MeanRaceLogLoss:F4}, " +
                    $"top1 {summary.MeanTop1Accuracy:P2}, top3 {summary.MeanTop3Accuracy:P2}, " +
                    $"runner LL {summary.MeanRunnerLogLoss:F4}, Brier {summary.MeanBrier:F4}");
            }

            return report;
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
