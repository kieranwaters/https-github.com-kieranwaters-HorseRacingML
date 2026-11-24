using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace HorseRacingML.Tests
{
    public class BettingSimulatorTests
    {
        [Fact]
        public void FiltersByRaceTypeAndFieldSize()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true), (0.5, 2.0m, false)),
                CreateRace(2, "Hurdle", 5, "2024-01-02", (0.7, 4.0m, true), (0.3, 2.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                RaceType = "Flat",
                MinRunners = 6,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal(1, bet.RaceId);
            Assert.Equal("Flat", bet.RaceType);
            Assert.Equal(8, bet.RunnerCount);
        }

        [Fact]
        public void MinimumEdgeExcludesLowValueRaces()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true), (0.5, 2.0m, false)),
                CreateRace(2, "Hurdle", 5, "2024-01-02", (0.7, 4.0m, true), (0.3, 2.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 55m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal(2, bet.RaceId);
            Assert.True(bet.Edge > 1.0d);
            Assert.Equal(1.8d, bet.Edge, 6);
        }

        [Fact]
        public void FlatStakesProduceTheExpectedBankrollAndDrawdown()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 2.0m, true)),
                CreateRace(2, "Flat", 8, "2024-01-02", (0.5, 2.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            Assert.Equal(2, result.BetCount);
            Assert.Equal(1, result.WinCount);
            Assert.Equal(1, result.LossCount);
            Assert.Equal(20m, result.TotalStaked);
            Assert.Equal(100m, result.EndingBankroll);
            Assert.Equal(0m, result.Profit);
            Assert.Equal(0m, result.Roi);
            Assert.Equal(10m, result.MaxDrawdown);
            Assert.Equal(1, result.LongestWinStreak);
            Assert.Equal(1, result.LongestLosingStreak);
            Assert.Equal(1m, result.ProfitFactor);
        }

        [Fact]
        public void SelectsOnlyTheHighestEdgeRunnerInEachRace()
        {
            var snapshot = BuildSnapshot(
                CreateRace(
                    1,
                    "Flat",
                    8,
                    "2024-01-01",
                    (0.4, 4.0m, true),
                    (0.6, 2.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal("Horse 1", bet.HorseName);
            Assert.Equal(0.4d, bet.AiProbability, 6);
        }

        [Fact]
        public void FiltersByOddsAndAiProbability()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true)),
                CreateRace(2, "Flat", 8, "2024-01-02", (0.5, 8.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                MaxOdds = 5m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal(1, bet.RaceId);
            Assert.True(bet.DecimalOdds <= 5m);
        }

        [Fact]
        public void CommissionReducesProfit()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 2.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 10m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = -100m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal(1.9m, bet.EffectiveDecimalOdds, 6);
            Assert.Equal(9m, bet.Profit);
            Assert.Equal(109m, result.EndingBankroll);
        }

        [Fact]
        public void KellyIsCappedByMaxKellyAndBankroll()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.9, 5.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Kelly,
                MaxKellyPercent = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.InRange(bet.Stake, 0m, 10m);
            Assert.True(bet.Stake <= 100m);
        }

        [Fact]
        public void BankrollPercentStakeUsesCurrentBankroll()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true)),
                CreateRace(2, "Flat", 8, "2024-01-02", (0.5, 3.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.BankrollPercent,
                BankrollPercent = 10m
            });

            Assert.Equal(2, result.BetCount);
            Assert.Equal(10m, result.Bets[0].Stake);
            Assert.Equal(12m, result.Bets[1].Stake);
        }

        [Fact]
        public void UndatedRacesAreBetAfterDatedRaces()
        {
            var snapshot = BuildSnapshot(
                CreateRaceDetailed(1, new DateTime(2024, 1, 1), "Flat", 8, null, null, null, null, (0.5, 2.0m, true)),
                CreateRaceDetailed(2, null, "Flat", 8, null, null, null, null, (0.5, 2.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            Assert.Equal(2, result.BetCount);
            Assert.Equal(1, result.Bets[0].RaceId);
            Assert.Equal(2, result.Bets[1].RaceId);
            Assert.Null(result.Bets[1].RaceDate);
        }

        [Fact]
        public void BetfairPriceSourceIsUsedWhenAvailable()
        {
            var snapshot = BuildSnapshot(
                CreateRaceDetailed(1, new DateTime(2024, 1, 1), "Flat", 8, 4.0m, null, null, null, (0.4, 2.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = -100m,
                PriceSource = SimulatorPriceSource.BetfairSp,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal(4.0m, bet.DecimalOdds);
            Assert.Equal(30m, bet.Profit);
        }

        [Fact]
        public void PriceSlippageIsAppliedBeforeSettlement()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 2.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = -100m,
                PriceSlippagePercent = 10m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal(1.9m, bet.DecimalOdds, 6);
            Assert.Equal(9m, bet.Profit);
        }

        [Fact]
        public void MaxStakeAmountCapsTheStake()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.9, 5.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 1_000m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Kelly,
                MaxKellyPercent = 100m,
                MaxStakeAmount = 2m
            });

            var bet = Assert.Single(result.Bets);
            Assert.Equal(2m, bet.Stake);
        }

        [Fact]
        public void MinStakeSkipsTinyBets()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.9, 5.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 1m,
                MinStake = 10m
            });

            Assert.Empty(result.Bets);
        }

        [Fact]
        public void DailyLossLimitStopsFurtherBetsOnTheSameDay()
        {
            var date = new DateTime(2024, 1, 1);
            var snapshot = BuildSnapshot(
                CreateRaceDetailed(1, date, "Flat", 8, null, null, null, null, (0.5, 2.0m, false)),
                CreateRaceDetailed(2, date, "Flat", 8, null, null, null, null, (0.5, 2.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m,
                MaxDailyLossPercent = 5m
            });

            Assert.Single(result.Bets);
            Assert.Equal(1, result.Bets[0].RaceId);
        }

        [Fact]
        public void DailyStakeLimitCapsTheDay()
        {
            var date = new DateTime(2024, 1, 1);
            var snapshot = BuildSnapshot(
                CreateRaceDetailed(1, date, "Flat", 8, null, null, null, null, (0.5, 2.0m, true)),
                CreateRaceDetailed(2, date, "Flat", 8, null, null, null, null, (0.5, 2.0m, true)),
                CreateRaceDetailed(3, date, "Flat", 8, null, null, null, null, (0.5, 2.0m, true)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m,
                MaxDailyStakePercent = 15m
            });

            Assert.Equal(2, result.BetCount);
            Assert.Equal(15m, result.TotalStaked);
        }

        [Fact]
        public void BootstrapStatisticsAreBoundedAndDeterministic()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true), (0.5, 2.0m, false)),
                CreateRace(2, "Flat", 8, "2024-01-02", (0.6, 3.0m, false), (0.4, 2.0m, false)),
                CreateRace(3, "Flat", 8, "2024-01-03", (0.7, 3.0m, true), (0.3, 2.0m, false)));

            var filter = new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m,
                BootstrapSamples = 250,
                BootstrapSeed = 7
            };

            var first = BettingSimulationEngine.Simulate(snapshot, filter);
            var second = BettingSimulationEngine.Simulate(snapshot, filter);

            Assert.Equal(250, first.BootstrapSamples);
            Assert.InRange(first.ProbabilityOfProfit, 0d, 1d);
            Assert.True(first.RoiConfidenceLow <= first.RoiConfidenceHigh);
            Assert.Equal(first.RoiConfidenceLow, second.RoiConfidenceLow, 10);
            Assert.Equal(first.ProbabilityOfProfit, second.ProbabilityOfProfit, 10);
        }

        [Fact]
        public void ExpectedProfitAndWinsMatchProbabilities()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true), (0.5, 2.0m, false)),
                CreateRace(2, "Flat", 8, "2024-01-02", (0.4, 4.0m, true), (0.6, 2.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            Assert.Equal(2, result.BetCount);
            // Race 1: 0.5 * 3.0 - 1 = 0.5 edge; Race 2: 0.4 * 4.0 - 1 = 0.6 edge.
            Assert.Equal(11m, result.ExpectedProfit, 6);
            Assert.Equal(0.9d, result.ExpectedWins, 6);
            Assert.Equal(2, result.WinCount);
        }

        [Fact]
        public void SelectionStrategiesCanChooseDifferentRunners()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.7, 2.0m, true), (0.5, 3.0m, false)));

            var edge = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = -100m,
                SelectionStrategy = SimulatorSelectionStrategy.HighestEdge,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });
            var shortest = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = -100m,
                SelectionStrategy = SimulatorSelectionStrategy.ShortestPrice,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            Assert.Equal("Horse 2", Assert.Single(edge.Bets).HorseName);
            Assert.Equal("Horse 1", Assert.Single(shortest.Bets).HorseName);
        }

        [Fact]
        public void MultipleBetsPerRaceAreSupported()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true), (0.4, 4.0m, false), (0.3, 5.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = -100m,
                BetsPerRace = 2,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            Assert.Equal(2, result.BetCount);
        }

        [Fact]
        public void DefaultMinimumEdgeExcludesZeroExpectedValueBets()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 2.0m, true), (0.3, 3.0m, false)));

            var result = BettingSimulationEngine.Simulate(snapshot, new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m
            });

            Assert.Empty(result.Bets);
            Assert.Equal(0, result.RacesWithCandidate);
            Assert.Equal(0, result.RacesWithStake);
        }

        [Fact]
        public void CrossFieldValidationRejectsInvertedRanges()
        {
            var filter = new SimulatorFilterRequest
            {
                MinRunners = 10,
                MaxRunners = 5,
                MinOdds = 5m,
                MaxOdds = 2m,
                MinEdgePercent = 10m,
                MaxEdgePercent = 1m,
                FromDate = new DateTime(2024, 2, 1),
                ToDate = new DateTime(2024, 1, 1)
            };

            var results = new List<ValidationResult>();
            var valid = Validator.TryValidateObject(
                filter,
                new ValidationContext(filter),
                results,
                validateAllProperties: true);

            Assert.False(valid);
            Assert.Contains(results, result => result.ErrorMessage!.Contains("runners", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(results, result => result.ErrorMessage!.Contains("odds", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(results, result => result.ErrorMessage!.Contains("edge", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(results, result => result.ErrorMessage!.Contains("from date", StringComparison.OrdinalIgnoreCase));
        }

        [Fact]
        public void EmptyCourseSelectionIncludesEveryCourse()
        {
            var ascot = CreateRace(
                1,
                "Flat",
                8,
                "2024-01-01",
                (0.5, 3.0m, true),
                (0.3, 2.0m, false));
            var chester = CreateRace(
                2,
                "Flat",
                8,
                "2024-01-02",
                (0.5, 3.0m, true),
                (0.3, 2.0m, false));
            ascot[0].CourseName = "Ascot";
            chester[0].CourseName = "Chester";

            var result = BettingSimulationEngine.Simulate(
                BuildSnapshot(ascot, chester),
                new SimulatorFilterRequest
                {
                    StartingBankroll = 100m,
                    CommissionPercent = 0m,
                    MinProbabilitySpread = 0m,
                    MinEdgePercent = 0m,
                    StakeMode = SimulatorStakeMode.Flat,
                    FlatStake = 10m,
                    Courses = new List<string>()
                });

            Assert.Equal(2, result.BetCount);
        }

        [Fact]
        public void AnalyticsBuildComparisonBaselinesAndWalkForward()
        {
            var snapshot = BuildSnapshot(
                CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true), (0.5, 2.0m, false)),
                CreateRace(2, "Flat", 8, "2024-01-10", (0.6, 3.0m, false), (0.4, 2.0m, false)),
                CreateRace(3, "Flat", 8, "2024-01-20", (0.7, 3.0m, true), (0.3, 2.0m, false)),
                CreateRace(4, "Flat", 8, "2024-01-30", (0.5, 3.0m, true), (0.5, 2.0m, false)));

            var filter = new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                CommissionPercent = 0m,
                MinProbabilitySpread = 0m,
                MinEdgePercent = 0m,
                StakeMode = SimulatorStakeMode.Flat,
                FlatStake = 10m,
                ConfirmPercent = 30m
            };

            var comparison = SimulatorAnalytics.BuildComparison(snapshot, filter);
            Assert.NotNull(comparison);
            Assert.Contains(comparison.Tune.BetCount, new[] { 1, 2, 3 });
            Assert.NotNull(comparison.Confirm);

            var baselines = SimulatorAnalytics.BuildBaselines(snapshot, filter);
            Assert.Equal(3, baselines.Count);
            Assert.Contains(baselines, row => row.Name.Contains("favourite", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(baselines, row => row.Name.Contains("top pick", StringComparison.OrdinalIgnoreCase));

            var walkForward = SimulatorAnalytics.BuildWalkForward(snapshot, filter);
            Assert.Equal(4, walkForward.Count);
            Assert.All(walkForward, period => Assert.NotNull(period.Label));
        }

        [Fact]
        public void SnapshotBuilderNormalisesProbabilitiesAndCapturesPrices()
        {
            var examples = new List<HyperparameterTrainer.RunnerExample>
            {
                new(
                    1,
                    new float[] { 0.1f },
                    0f,
                    1,
                    "Horse 1",
                    2.0m,
                    betfairSp: 4.0m,
                    betfairLastTradedPrice: 3.5m),
                new(
                    1,
                    new float[] { 0.2f },
                    1f,
                    2,
                    "Horse 2",
                    5.0m,
                    betfairSp: 6.0m,
                    betfairLastTradedPrice: 5.5m)
            };
            var result = new HyperparameterTrainer.TrainingResult
            {
                ValidationPredictions = new[] { 0.25f, 0.75f },
                ValidationLabels = new[] { 0f, 1f },
                ValidationRaceIds = new[] { 1, 1 },
                ValidationExamples = examples
            };
            var summaries = new Dictionary<int, RaceSummary>
            {
                [1] = new RaceSummary
                {
                    RaceId = 1,
                    RaceDate = new DateTime(2024, 1, 1),
                    Title = "Flat Race",
                    CourseName = "Test",
                    Surface = "Turf",
                    Going = "Good",
                    RunnerCount = 2
                }
            };

            var snapshot = SimulatorSnapshotBuilder.Build(
                result,
                new[] { 0.25f, 0.75f },
                summaries,
                new MLParameter { Objective = "RaceSoftmax", ModelHash = "abc", TestEndDate = new DateTime(2024, 1, 1) },
                new DateTime(2024, 1, 1),
                new DateTime(2024, 1, 31),
                1,
                10,
                1,
                "test");

            Assert.Single(snapshot.Races);
            Assert.Equal(2, snapshot.Runners.Count);
            Assert.Equal(1d, snapshot.Runners.Sum(row => row.Probability), 6);
            Assert.Equal(4.0m, snapshot.Runners[0].BetfairSp);
            Assert.Equal(3.5m, snapshot.Runners[0].BetfairLastTradedPrice);
            Assert.True(snapshot.Summary.HasBetfairPrices);
            Assert.Equal(2, snapshot.Summary.PricedRunnerCount);
            Assert.Equal(new DateTime(2024, 1, 1), snapshot.Summary.FirstRaceDate);
        }

        [Fact]
        public void StorePersistsAndReloadsSnapshots()
        {
            var directory = Path.Combine(
                Path.GetTempPath(),
                "horseracing-simulator-tests",
                Guid.NewGuid().ToString("N"));
            try
            {
                var configuration = new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["Training:SimulatorSnapshotDirectory"] = directory
                    })
                    .Build();
                var snapshot = BuildSnapshot(
                    CreateRace(1, "Flat", 8, "2024-01-01", (0.5, 3.0m, true)));
                snapshot.Summary.ValidationStart = new DateTime(2024, 1, 1);
                snapshot.Summary.ValidationEnd = new DateTime(2024, 1, 31);
                snapshot.Summary.ModelHash = "abcdef1234567890";
                snapshot.Summary.Source = "test";
                snapshot.Races = new List<SimulatorRace>
                {
                    new()
                    {
                        RaceId = 1,
                        RaceDate = new DateTime(2024, 1, 1),
                        RunnerCount = 1,
                        Runners = snapshot.Runners
                    }
                };

                var store = new BettingSimulatorStore(configuration);
                var key = store.Set(snapshot);
                Assert.True(File.Exists(Path.Combine(directory, "index.json")));

                var reloadedStore = new BettingSimulatorStore(configuration);
                var reloaded = reloadedStore.Get(key);
                Assert.NotNull(reloaded);
                Assert.Equal(1, reloaded!.Summary.RunnerCount);
                Assert.Equal("test", reloaded.Summary.Source);
                Assert.Single(reloaded.Runners);
                Assert.Single(reloaded.Races);
                Assert.Single(reloadedStore.List());
            }
            finally
            {
                if (Directory.Exists(directory))
                {
                    Directory.Delete(directory, recursive: true);
                }
            }
        }

        private static BettingSimulationSnapshot BuildSnapshot(params SimulatorRunnerRow[][] races)
        {
            var rows = races.SelectMany(race => race).ToList();
            return new BettingSimulationSnapshot
            {
                Summary = new BettingSimulationSnapshotSummary
                {
                    ValidationStart = new DateTime(2024, 1, 1),
                    ValidationEnd = new DateTime(2024, 1, 31),
                    FirstRaceDate = rows.Where(row => row.RaceDate.HasValue).Select(row => row.RaceDate).Min(),
                    LastRaceDate = rows.Where(row => row.RaceDate.HasValue).Select(row => row.RaceDate).Max(),
                    RunnerCount = rows.Count
                },
                Runners = rows
            };
        }

        private static SimulatorRunnerRow[] CreateRace(
            int raceId,
            string raceType,
            int runnerCount,
            string date,
            params (double Probability, decimal Odds, bool Won)[] runners)
        {
            return CreateRaceDetailed(
                raceId,
                DateTime.Parse(date),
                raceType,
                runnerCount,
                null,
                null,
                null,
                null,
                runners);
        }

        private static SimulatorRunnerRow[] CreateRaceDetailed(
            int raceId,
            DateTime? date,
            string raceType,
            int runnerCount,
            decimal? betfairSp,
            decimal? betfairLastTraded,
            short? distanceYards,
            string? ageRestriction,
            params (double Probability, decimal Odds, bool Won)[] runners)
        {
            return runners
                .Select((runner, index) => new SimulatorRunnerRow
                {
                    RaceId = raceId,
                    RaceDate = date,
                    RaceTitle = $"{raceType} race {raceId}",
                    CourseName = "Test Course",
                    RaceType = raceType,
                    Class = 4,
                    AgeRestriction = ageRestriction,
                    Surface = "Turf",
                    Going = "Good",
                    DistanceYards = distanceYards,
                    RunnerCount = runnerCount,
                    HorseName = $"Horse {index + 1}",
                    FinishPosition = runner.Won ? 1 : 2,
                    DecimalOdds = runner.Odds,
                    BetfairSp = betfairSp,
                    BetfairLastTradedPrice = betfairLastTraded,
                    Probability = runner.Probability,
                    RawProbability = runner.Probability,
                    Won = runner.Won
                })
                .ToArray();
        }
    }
}
