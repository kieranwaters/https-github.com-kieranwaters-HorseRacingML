using System;
using System.Collections.Generic;
using System.Reflection;
using HorseRacingML.ML;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace HorseRacingML.Tests
{
    public class MarketDataFeatureTests
    {
        [Fact]
        public void MarketLiquidityBlockIsOptIn()
        {
            Assert.False(HyperparameterTrainer.IsFeatureEnabled(
                "BetfairMarketMatchedLog",
                FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled(
                "BetfairMarketMatchedLog",
                FeatureBlocks.MarketLiquidity));
            Assert.Equal(
                FeatureBlocks.MarketLiquidity,
                FeatureBlockNames.Parse("liquidity"));
            Assert.Contains(
                "MarketLiquidity",
                FeatureBlockNames.Format(FeatureBlocks.MarketLiquidity));

            // No live Betfair API capture exists yet, so a model trained with
            // this block must not be loaded for live scoring.
            Assert.False(HyperparameterTrainer.IsLiveAvailableFeature(
                "BetfairMarketMatchedLog"));
        }

        [Fact]
        public void TouchedHighPopulatesRangeFeatures()
        {
            var (state, processRace) = CreateState(FeatureBlocks.Pace | FeatureBlocks.MarketMovement);
            var runners = CreateRace();
            runners[0]["OpeningFraction"] = "6/1";
            runners[0]["TouchedLowFraction"] = "4/1";
            runners[0]["TouchedHighFraction"] = "8/1";

            processRace.Invoke(state, new object?[] { runners, true, true });

            Assert.False(Convert.ToBoolean(runners[0]["MarketMoveHighMissing"]));
            Assert.Equal(MathF.Log(9f / 5f), Convert.ToSingle(runners[0]["MarketMoveRangeLog"]), 3);
            Assert.Equal(MathF.Log(9f / 7f), Convert.ToSingle(runners[0]["MarketMoveOpeningToHighLog"]), 3);
            Assert.Equal(0.5f, Convert.ToSingle(runners[0]["MarketMoveOpeningInRange"]), 3);
            Assert.True(Convert.ToBoolean(runners[0]["MarketMoveBackedFlag"]));
            Assert.False(Convert.ToBoolean(runners[0]["MarketMoveDriftFlag"]));
        }

        [Fact]
        public void MissingTouchedHighSetsTheFlagAndNeutralRange()
        {
            var (state, processRace) = CreateState(FeatureBlocks.Pace | FeatureBlocks.MarketMovement);
            var runners = CreateRace();
            runners[0]["OpeningFraction"] = "6/1";
            runners[0]["TouchedLowFraction"] = "4/1";

            processRace.Invoke(state, new object?[] { runners, true, true });

            Assert.True(Convert.ToBoolean(runners[0]["MarketMoveHighMissing"]));
            Assert.Equal(0f, Convert.ToSingle(runners[0]["MarketMoveRangeLog"]), 3);
            Assert.Equal(0.5f, Convert.ToSingle(runners[0]["MarketMoveOpeningInRange"]), 3);
        }

        [Fact]
        public void BetfairLiquidityColumnsBecomeModelFeatures()
        {
            var (state, processRace) = CreateState(FeatureBlocks.Pace | FeatureBlocks.MarketLiquidity);
            var runners = CreateRace();
            runners[0]["BetfairTotalMatched"] = 10000m;
            runners[0]["BetfairTradedVolume"] = 1000m;
            runners[0]["BetfairPreOffHigh"] = 5.0m;
            runners[0]["BetfairPreOffLow"] = 2.5m;
            runners[0]["BetfairProjectedSP"] = 3.0m;
            runners[0]["BetfairAvailableToBack"] = 100m;
            runners[0]["BetfairAvailableToLay"] = 50m;

            processRace.Invoke(state, new object?[] { runners, true, true });

            Assert.Equal(MathF.Log(10001f), Convert.ToSingle(runners[0]["BetfairMarketMatchedLog"]), 3);
            Assert.Equal(MathF.Log(1001f), Convert.ToSingle(runners[0]["BetfairRunnerVolumeLog"]), 3);
            Assert.Equal(0.1f, Convert.ToSingle(runners[0]["BetfairRunnerVolumeShare"]), 3);
            Assert.Equal(MathF.Log(2f), Convert.ToSingle(runners[0]["BetfairPreOffRangeLog"]), 3);
            Assert.Equal(MathF.Log(3f), Convert.ToSingle(runners[0]["BetfairProjectedSPLog"]), 3);
            Assert.Equal(1f / 3f, Convert.ToSingle(runners[0]["BetfairBookPressure"]), 3);
            Assert.False(Convert.ToBoolean(runners[0]["BetfairLiquidityMissing"]));
            Assert.False(Convert.ToBoolean(runners[0]["BetfairExchangeRangeMissing"]));
            Assert.False(Convert.ToBoolean(runners[0]["BetfairProjectedSPMissing"]));
        }

        private static (object State, MethodInfo ProcessRace) CreateState(FeatureBlocks blocks)
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] =
                        "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;",
                    ["Training:FeatureBlocks"] = FeatureBlockNames.Format(blocks)
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType(
                "FeatureEngineeringState",
                BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod(
                "ProcessRace",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);
            return (state!, processRace!);
        }

        private static List<Dictionary<string, object?>> CreateRace()
        {
            return new List<Dictionary<string, object?>>
            {
                CreateRunner(horseId: 1, finishPos: 1),
                CreateRunner(horseId: 2, finishPos: 2)
            };
        }

        private static Dictionary<string, object?> CreateRunner(int horseId, int finishPos)
        {
            return new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["RaceId"] = 1,
                ["HorseId"] = horseId,
                ["CourseId"] = 1,
                ["RaceDate"] = new DateTime(2024, 1, 1, 13, 0, 0),
                ["ScheduledOff"] = new TimeSpan(13, 0, 0),
                ["ActualOff"] = new TimeSpan(13, 0, 0),
                ["Title"] = "Test Race",
                ["RaceType"] = "Handicap",
                ["Class"] = 3,
                ["AgeRestriction"] = "3yo+",
                ["Surface"] = "Turf",
                ["Going"] = "Good",
                ["DistanceYards"] = 1760,
                ["DistanceText"] = "1m",
                ["RunnerCount"] = 2,
                ["Status"] = "Result",
                ["WinningTimeMs"] = 100000,
                ["TrainerId"] = 10 + horseId,
                ["JockeyId"] = 20 + horseId,
                ["SaddleclothNumber"] = horseId,
                ["Draw"] = horseId,
                ["Age"] = 5,
                ["WeightLbs"] = 126,
                ["WeightText"] = "9-0",
                ["OfficialRating"] = 70,
                ["FinishPos"] = (short)finishPos,
                ["OutcomeCode"] = null,
                ["DistanceBeatenText"] = finishPos == 1 ? null : "2",
                ["DistanceBeatenLengths"] = finishPos == 1 ? null : 2m,
                ["SP_Fraction"] = null,
                ["SP_Decimal"] = 4.0m,
                ["FavTag"] = null,
                ["OpeningFraction"] = horseId == 1 ? "3/1" : "5/1",
                ["TouchedHighFraction"] = null,
                ["TouchedLowFraction"] = null,
                ["Purse"] = 1000m,
                ["HorseName"] = $"Horse {horseId}",
                ["TrainerName"] = "Trainer",
                ["JockeyName"] = "Jockey",
                ["CourseName"] = "Course"
            };
        }
    }
}
