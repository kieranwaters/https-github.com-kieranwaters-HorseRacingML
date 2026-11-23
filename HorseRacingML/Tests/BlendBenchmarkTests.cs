using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace HorseRacingML.Tests
{
    public class BlendBenchmarkTests
    {
        private const string ConnectionString =
            "Data Source=.\\SQLEXPRESS;Initial Catalog=HorseRacingML;Integrated Security=True;TrustServerCertificate=True;Connect Timeout=30";

        [Fact]
        public void CompareOffsetModesAndBlend()
        {
            if (!string.Equals(Environment.GetEnvironmentVariable("RUN_BLEND_BENCHMARK"), "1", StringComparison.Ordinal))
            {
                return;
            }

            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = ConnectionString,
                    ["Training:Objective"] = "RaceSoftmax",
                    ["Training:UseMarketOffset"] = "true",
                    ["Training:MarketOffsetMode"] = "LogProb",
                    ["Training:MarketBlind"] = "false",
                    ["Training:RaceBatchSize"] = "64",
                    ["Training:Seed"] = "42",
                    ["Training:L2WeightDecay"] = "0.0001",
                    ["Training:EarlyStoppingPatience"] = "3",
                    ["Training:EvaluationFrequency"] = "2",
                    ["Training:LiveStateCacheMinutes"] = "5",
                    ["WinRateAlpha"] = "1.0",
                    ["WinRateBeta"] = "9.0"
                })
                .Build();

            var repository = new RacingRepository(ConnectionString);
            var trainer = new HyperparameterTrainer(configuration);
            var reportPath = Path.Combine(Path.GetTempPath(), "horse-racing-blend-benchmark.txt");
            File.WriteAllText(reportPath, $"Blend benchmark started {DateTime.Now:yyyy-MM-dd HH:mm:ss}{Environment.NewLine}");

            var allRaceIds = repository.GetRaceIdsWithOpeningPrice(new DateTime(2017, 1, 1), DateTime.Today);
            int testCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * 0.10));
            int devCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * 0.10));
            int testStart = allRaceIds.Count - testCount;
            int devStart = testStart - devCount;

            var orderedTrainIds = allRaceIds.Take(devStart).ToList();
            int cap = 0;
            if (int.TryParse(Environment.GetEnvironmentVariable("BENCH_MAX_TRAIN_RACES"), out var configuredCap))
            {
                cap = Math.Max(1000, configuredCap);
            }
            if (cap > 0 && orderedTrainIds.Count > cap)
            {
                orderedTrainIds = orderedTrainIds.Skip(orderedTrainIds.Count - cap).ToList();
            }

            var trainingRaceIds = new HashSet<int>(orderedTrainIds);
            var devRaceIds = new HashSet<int>(allRaceIds.Skip(devStart).Take(devCount));
            var testRaceIds = new HashSet<int>(allRaceIds.Skip(testStart));
            var trainDevRaceIds = new HashSet<int>(trainingRaceIds);
            trainDevRaceIds.UnionWith(devRaceIds);
            var included = new HashSet<int>(trainingRaceIds);
            included.UnionWith(devRaceIds);
            included.UnionWith(testRaceIds);

            Write(reportPath, $"Races: train={trainingRaceIds.Count}, dev={devRaceIds.Count}, test={testRaceIds.Count}, cap={cap}");
            Write(reportPath, "Loading market-feature dataset...");
            var fullMarket = trainer.LoadEncodedTrainingDataset(
                included,
                includeIdentifiers: true,
                requireMarketPrice: false,
                metadataRaceIds: trainingRaceIds,
                marketBlind: false);
            var devMarket = HyperparameterTrainer.SplitDataset(fullMarket, trainingRaceIds, devRaceIds);
            var testMarket = HyperparameterTrainer.SplitDataset(fullMarket, trainDevRaceIds, testRaceIds);

            Write(reportPath, "Loading market-blind dataset...");
            var fullBlind = trainer.LoadEncodedTrainingDataset(
                included,
                includeIdentifiers: true,
                requireMarketPrice: false,
                metadataRaceIds: trainingRaceIds,
                marketBlind: true);
            var devBlind = HyperparameterTrainer.SplitDataset(fullBlind, trainingRaceIds, devRaceIds);
            var testBlind = HyperparameterTrainer.SplitDataset(fullBlind, trainDevRaceIds, testRaceIds);

            var candidates = new List<CandidateResult>();
            candidates.Add(RunCandidate("baseline-logit", "Logit", false, devMarket, testMarket, trainer, reportPath));
            candidates.Add(RunCandidate("logprob", "LogProb", false, devMarket, testMarket, trainer, reportPath));
            candidates.Add(RunCandidate("blind-logprob", "LogProb", true, devBlind, testBlind, trainer, reportPath));

            var best = candidates
                .Where(c => c.DevBlend.HasData)
                .OrderBy(c => c.DevBlend.Metrics.LogLoss)
                .FirstOrDefault();
            if (best is null)
            {
                Write(reportPath, "No candidate produced a dev blend; stopping.");
                return;
            }

            Write(reportPath, $"Best dev candidate: {best.Name} (dev blend LL {best.DevBlend.Metrics.LogLoss:F6})");
            foreach (var candidate in candidates)
            {
                var final = RetrainAndEvaluate(candidate, trainer, reportPath);
                Write(reportPath,
                    $"TEST {candidate.Name}: stand-alone LL {final.Standalone.LogLoss:F6}, Brier {final.Standalone.Brier:F6}; " +
                    $"fair market LL {final.FairMarket.LogLoss:F6}; blend LL {final.Blend.LogLoss:F6}, Brier {final.Blend.Brier:F6}, " +
                    $"race LL {final.Blend.RaceLogLoss:F6}, a={candidate.DevBlend.ModelWeight:F3}, b={candidate.DevBlend.MarketWeight:F3}; " +
                    $"blend improvement {final.FairMarket.LogLoss - final.Blend.LogLoss:F6}");
            }

            Write(reportPath, $"Blend benchmark finished {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        }

        private static CandidateResult RunCandidate(
            string name,
            string offsetMode,
            bool marketBlind,
            HyperparameterTrainer.TrainingDataset devSet,
            HyperparameterTrainer.TrainingDataset testSet,
            HyperparameterTrainer trainer,
            string reportPath)
        {
            Write(reportPath, $"--- Training {name} (offset={offsetMode}, blind={marketBlind}) ---");
            var parameter = CreateParameter(offsetMode, marketBlind);
            var result = trainer.Train(parameter, 0, 1, devSet, persistWeights: false, earlyStopOnValidation: true);
            var fair = MarketBlendCalibration.FitMarketOnly(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples);
            var blend = MarketBlendCalibration.Fit(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples);
            var priced = HyperparameterTrainer.ComputePricedSubsetMetrics(result);

            Write(reportPath,
                $"DEV {name}: stand-alone LL {result.ValidationLoss:F6}, Brier {result.ValidationBrier:F6}, " +
                $"priced LL {priced.LogLoss:F6}, Brier {priced.Brier:F6}; fair market LL {fair.Metrics.LogLoss:F6}; " +
                $"blend LL {blend.Metrics.LogLoss:F6}, a={blend.ModelWeight:F3}, b={blend.MarketWeight:F3}, " +
                $"improvement {fair.Metrics.LogLoss - blend.Metrics.LogLoss:F6}; temperature {result.ValidationTemperature:F3}, best epoch {result.BestEpoch}");

            return new CandidateResult
            {
                Name = name,
                Parameter = parameter,
                DevResult = result,
                DevFair = fair,
                DevBlend = blend,
                TestSet = testSet
            };
        }

        private static FinalResult RetrainAndEvaluate(
            CandidateResult candidate,
            HyperparameterTrainer trainer,
            string reportPath)
        {
            Write(reportPath, $"--- Final retrain {candidate.Name} ---");
            var parameter = new MLParameter
            {
                Units = candidate.Parameter.Units,
                Dropout = candidate.Parameter.Dropout,
                Layers = candidate.Parameter.Layers,
                LearningRate = candidate.Parameter.LearningRate,
                Epochs = candidate.DevResult.BestEpoch.HasValue
                    ? Math.Max(1, candidate.DevResult.BestEpoch.Value + 1)
                    : candidate.Parameter.Epochs,
                BatchSize = candidate.Parameter.BatchSize,
                Folds = 1,
                Objective = candidate.Parameter.Objective,
                MarketOffsetMode = candidate.Parameter.MarketOffsetMode,
                MarketBlind = candidate.Parameter.MarketBlind,
                RaceBatchSize = candidate.Parameter.RaceBatchSize,
                WinRateAlpha = candidate.Parameter.WinRateAlpha,
                WinRateBeta = candidate.Parameter.WinRateBeta,
                Seed = candidate.Parameter.Seed
            };

            var result = trainer.Train(
                parameter,
                0,
                1,
                candidate.TestSet,
                persistWeights: false,
                temperatureOverride: (float)candidate.DevResult.ValidationTemperature,
                earlyStopOnValidation: false,
                blendModelWeight: candidate.DevBlend.ModelWeight,
                blendMarketWeight: candidate.DevBlend.MarketWeight);

            var standalone = new MarketBlendMetrics
            {
                LogLoss = result.ValidationLoss,
                Brier = result.ValidationBrier
            };
            var fair = MarketBlendCalibration.Evaluate(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples,
                0d,
                candidate.DevFair.MarketWeight);
            var blend = MarketBlendCalibration.Evaluate(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples,
                candidate.DevBlend.ModelWeight,
                candidate.DevBlend.MarketWeight);

            return new FinalResult
            {
                Standalone = standalone,
                FairMarket = fair,
                Blend = blend
            };
        }

        private static MLParameter CreateParameter(string offsetMode, bool marketBlind)
        {
            return new MLParameter
            {
                Units = 64,
                Dropout = 0.05,
                Layers = 2,
                LearningRate = 0.001,
                Epochs = 8,
                BatchSize = 500,
                Folds = 1,
                Objective = "RaceSoftmax",
                MarketOffsetMode = offsetMode,
                MarketBlind = marketBlind,
                RaceBatchSize = 64,
                WinRateAlpha = 1.0,
                WinRateBeta = 9.0,
                Seed = 42
            };
        }

        private static void Write(string path, string message)
        {
            Console.WriteLine(message);
            File.AppendAllText(path, message + Environment.NewLine);
        }

        private sealed class CandidateResult
        {
            public string Name { get; init; } = string.Empty;
            public MLParameter Parameter { get; init; } = new();
            public HyperparameterTrainer.TrainingResult DevResult { get; init; } = new();
            public MarketBlendParameters DevFair { get; init; } = new();
            public MarketBlendParameters DevBlend { get; init; } = new();
            public HyperparameterTrainer.TrainingDataset TestSet { get; init; } = null!;
        }

        private sealed class FinalResult
        {
            public MarketBlendMetrics Standalone { get; init; } = new();
            public MarketBlendMetrics FairMarket { get; init; } = new();
            public MarketBlendMetrics Blend { get; init; } = new();
        }
    }
}
