using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Optional LightGBM LambdaRank comparison for the tabular feature set.
    ///
    /// This is deliberately a benchmark, not the persisted model path yet: it
    /// loads the same chronological train/dev/test split as the MLP benchmark,
    /// trains a group-wise ranker, converts its scores to within-race
    /// probabilities, and reports race log loss/top-k against the no-vig market
    /// and SP. If it wins consistently, its scores can be added as another
    /// ensemble member without changing the feature contract.
    /// </summary>
    public static class LightGbmRankingBenchmark
    {
        public sealed class Report
        {
            public int FeatureCount { get; init; }
            public int TrainRaces { get; init; }
            public int DevRaces { get; init; }
            public int TestRaces { get; init; }
            public int TrainRows { get; init; }
            public int DevRows { get; init; }
            public int TestRows { get; init; }
            public double DevLogLoss { get; init; }
            public double DevBrier { get; init; }
            public double DevRaceLogLoss { get; init; }
            public double DevTop1Accuracy { get; init; }
            public double DevTop3Accuracy { get; init; }
            public double TestLogLoss { get; init; }
            public double TestBrier { get; init; }
            public double TestRaceLogLoss { get; init; }
            public double TestTop1Accuracy { get; init; }
            public double TestTop3Accuracy { get; init; }
            public double DevFairRaceLogLoss { get; init; }
            public double DevFairTop1Accuracy { get; init; }
            public double DevFairTop3Accuracy { get; init; }
            public double TestFairRaceLogLoss { get; init; }
            public double TestFairTop1Accuracy { get; init; }
            public double TestFairTop3Accuracy { get; init; }
            public double DevSpRaceLogLoss { get; init; }
            public double DevSpTop1Accuracy { get; init; }
            public double TestSpRaceLogLoss { get; init; }
            public double TestSpTop1Accuracy { get; init; }
        }

        private sealed class RankingRow
        {
            [ColumnName("Label")]
            public float Label { get; set; }

            [ColumnName("GroupId")]
            public uint GroupId { get; set; }

            [ColumnName("Features")]
            public float[] Features { get; set; } = Array.Empty<float>();
        }

        private sealed class RankingPrediction
        {
            [ColumnName("Score")]
            public float Score { get; set; }
        }

        public static Report Run(
            IConfiguration configuration,
            RacingRepository repository,
            HyperparameterTrainer trainer,
            MLParameter reference,
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
            decimal commission = Math.Clamp(configuration.GetValue<decimal?>("Betting:CommissionRate") ?? 0.05m, 0m, 1m);

            var allRaceIds = requirePreOffPrice
                ? repository.GetRaceIdsWithOpeningPrice(minimumRaceDate, DateTime.Today)
                : repository.GetRaceIdsBetweenDates(minimumRaceDate, DateTime.Today);
            int? maxRaces = configuration.GetValue<int?>("Training:BenchmarkMaxRaces");
            if (maxRaces.HasValue && maxRaces.Value > 0 &&
                allRaceIds.Count > maxRaces.Value)
            {
                allRaceIds = allRaceIds
                    .Skip(allRaceIds.Count - maxRaces.Value)
                    .ToList();
            }

            if (allRaceIds.Count < 5)
            {
                throw new InvalidOperationException(
                    $"Not enough eligible races from {minimumRaceDate:yyyy-MM-dd} for the LightGBM benchmark.");
            }

            DateTime? benchmarkStateStart = maxRaces.HasValue && allRaceIds.Count > 0
                ? repository.GetRaceDate(allRaceIds[0])
                : null;

            int testCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * testFraction));
            int devCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * devFraction));
            if (testCount + devCount >= allRaceIds.Count)
            {
                devCount = Math.Max(1, (allRaceIds.Count - testCount) / 2);
            }

            int testStart = allRaceIds.Count - testCount;
            int devStart = testStart - devCount;
            var trainingRaceIds = new HashSet<int>(allRaceIds.Take(devStart));
            var devRaceIds = new HashSet<int>(allRaceIds.Skip(devStart).Take(devCount));
            var testRaceIds = new HashSet<int>(allRaceIds.Skip(testStart));
            var trainDevRaceIds = new HashSet<int>(trainingRaceIds);
            trainDevRaceIds.UnionWith(devRaceIds);
            var allIncluded = new HashSet<int>(allRaceIds);

            log($"[lightgbm] split: train={trainingRaceIds.Count}, dev={devRaceIds.Count}, test={testRaceIds.Count} races");
            var full = trainer.LoadEncodedTrainingDataset(
                includeRaceIds: allIncluded,
                includeIdentifiers: true,
                minimumRaceDate: benchmarkStateStart,
                requireMarketPrice: !keepUnpricedRunners,
                metadataRaceIds: trainingRaceIds,
                marketBlind: false);
            var devView = HyperparameterTrainer.SplitDataset(full, trainingRaceIds, devRaceIds);
            var testView = HyperparameterTrainer.SplitDataset(full, trainDevRaceIds, testRaceIds);

            var ml = new MLContext(seed: reference.Seed ?? 42);
            var schema = SchemaDefinition.Create(typeof(RankingRow));
            schema[nameof(RankingRow.Features)].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, full.FeatureCount);

            var trainData = ml.Data.LoadFromEnumerable(
                BuildRows(devView.TrainingRaces),
                schema);
            var devData = ml.Data.LoadFromEnumerable(
                BuildRows(devView.ValidationRaces),
                schema);
            var testData = ml.Data.LoadFromEnumerable(
                BuildRows(testView.ValidationRaces),
                schema);

            var options = new LightGbmRankingTrainer.Options
            {
                LabelColumnName = "Label",
                RowGroupColumnName = "GroupId",
                FeatureColumnName = "Features",
                NumberOfLeaves = 31,
                MinimumExampleCountPerLeaf = 50,
                LearningRate = 0.05,
                NumberOfIterations = 200,
                MaximumBinCountPerFeature = 255
            };

            log($"[lightgbm] training on {devView.TrainingRaces.Sum(r => r.Runners.Count)} rows, {full.FeatureCount} features");
            var model = ml.Ranking.Trainers.LightGbm(options).Fit(trainData);

            var devMetrics = Evaluate(
                ml,
                model,
                devData,
                devView.ValidationRaces,
                commission);
            var devFair = EvaluateFairMarket(devView.ValidationRaces);
            var devSp = EvaluateSpMarket(devView.ValidationRaces, commission);
            var testMetrics = Evaluate(
                ml,
                model,
                testData,
                testView.ValidationRaces,
                commission);
            var testFair = EvaluateFairMarket(testView.ValidationRaces);
            var testSp = EvaluateSpMarket(testView.ValidationRaces, commission);

            log(
                $"[lightgbm] dev race LL {devMetrics.RaceLogLoss:F4} (fair {devFair.RaceLogLoss:F4}, SP {devSp.RaceLogLoss:F4}), " +
                $"top1 {devMetrics.Top1Accuracy:P2}, top3 {devMetrics.Top3Accuracy:P2}");
            log(
                $"[lightgbm] test race LL {testMetrics.RaceLogLoss:F4} (fair {testFair.RaceLogLoss:F4}, SP {testSp.RaceLogLoss:F4}), " +
                $"top1 {testMetrics.Top1Accuracy:P2}, top3 {testMetrics.Top3Accuracy:P2}");

            return new Report
            {
                FeatureCount = full.FeatureCount,
                TrainRaces = trainingRaceIds.Count,
                DevRaces = devRaceIds.Count,
                TestRaces = testRaceIds.Count,
                TrainRows = devView.TrainingRaces.Sum(r => r.Runners.Count),
                DevRows = devView.ValidationRaces.Sum(r => r.Runners.Count),
                TestRows = testView.ValidationRaces.Sum(r => r.Runners.Count),
                DevLogLoss = devMetrics.RunnerLogLoss,
                DevBrier = devMetrics.Brier,
                DevRaceLogLoss = devMetrics.RaceLogLoss,
                DevTop1Accuracy = devMetrics.Top1Accuracy,
                DevTop3Accuracy = devMetrics.Top3Accuracy,
                TestLogLoss = testMetrics.RunnerLogLoss,
                TestBrier = testMetrics.Brier,
                TestRaceLogLoss = testMetrics.RaceLogLoss,
                TestTop1Accuracy = testMetrics.Top1Accuracy,
                TestTop3Accuracy = testMetrics.Top3Accuracy,
                DevFairRaceLogLoss = devFair.RaceLogLoss,
                DevFairTop1Accuracy = devFair.Top1Accuracy,
                DevFairTop3Accuracy = devFair.Top3Accuracy,
                TestFairRaceLogLoss = testFair.RaceLogLoss,
                TestFairTop1Accuracy = testFair.Top1Accuracy,
                TestFairTop3Accuracy = testFair.Top3Accuracy,
                DevSpRaceLogLoss = devSp.RaceLogLoss,
                DevSpTop1Accuracy = devSp.Top1Accuracy,
                TestSpRaceLogLoss = testSp.RaceLogLoss,
                TestSpTop1Accuracy = testSp.Top1Accuracy
            };
        }

        private static List<RankingRow> BuildRows(IReadOnlyList<HyperparameterTrainer.RaceExample> races)
        {
            var rows = new List<RankingRow>();
            uint groupId = 0;
            foreach (var race in races)
            {
                groupId++;
                foreach (var runner in race.Runners)
                {
                    rows.Add(new RankingRow
                    {
                        Label = runner.Label,
                        GroupId = groupId,
                        Features = runner.Features
                    });
                }
            }

            return rows;
        }

        private static RaceEvaluationMetrics Evaluate(
            MLContext ml,
            ITransformer model,
            IDataView data,
            IReadOnlyList<HyperparameterTrainer.RaceExample> races,
            decimal commission)
        {
            var scored = model.Transform(data);
            var scores = ml.Data
                .CreateEnumerable<RankingPrediction>(scored, reuseRowObject: false)
                .Select(row => row.Score)
                .ToArray();

            var predictions = new float[scores.Length];
            int runnerIndex = 0;
            foreach (var race in races)
            {
                int count = race.Runners.Count;
                float max = float.NegativeInfinity;
                for (int i = 0; i < count; i++)
                {
                    max = Math.Max(max, scores[runnerIndex + i]);
                }

                double sum = 0;
                for (int i = 0; i < count; i++)
                {
                    double value = Math.Exp(scores[runnerIndex + i] - max);
                    predictions[runnerIndex + i] = (float)value;
                    sum += value;
                }

                for (int i = 0; i < count; i++)
                {
                    predictions[runnerIndex + i] = (float)(predictions[runnerIndex + i] / sum);
                }

                runnerIndex += count;
            }

            var raceIds = new List<int>();
            var labels = new List<float>();
            foreach (var race in races)
            {
                foreach (var runner in race.Runners)
                {
                    raceIds.Add(race.RaceId);
                    labels.Add(runner.Label);
                }
            }

            return RaceMetrics.Compute(raceIds, predictions, labels);
        }

        private static RaceEvaluationMetrics EvaluateFairMarket(
            IReadOnlyList<HyperparameterTrainer.RaceExample> races)
        {
            var raceIds = new List<int>();
            var predictions = new List<float>();
            var labels = new List<float>();
            foreach (var race in races)
            {
                foreach (var runner in race.Runners)
                {
                    raceIds.Add(race.RaceId);
                    predictions.Add(runner.MarketProbability);
                    labels.Add(runner.Label);
                }
            }

            return RaceMetrics.Compute(raceIds, predictions, labels);
        }

        private static RaceEvaluationMetrics EvaluateSpMarket(
            IReadOnlyList<HyperparameterTrainer.RaceExample> races,
            decimal commission)
        {
            var raceIds = new List<int>();
            var predictions = new List<float>();
            var labels = new List<float>();
            foreach (var race in races)
            {
                foreach (var runner in race.Runners)
                {
                    if (!runner.StartingPriceDecimal.HasValue ||
                        runner.StartingPriceDecimal.Value <= 1m)
                    {
                        continue;
                    }

                    decimal odds = BettingMath.CalculateEffectiveDecimalOdds(
                        runner.StartingPriceDecimal.Value,
                        commission);
                    if (odds <= 1m)
                    {
                        continue;
                    }

                    raceIds.Add(race.RaceId);
                    predictions.Add((float)(1.0 / (double)odds));
                    labels.Add(runner.Label);
                }
            }

            return RaceMetrics.Compute(raceIds, predictions, labels);
        }
    }
}
