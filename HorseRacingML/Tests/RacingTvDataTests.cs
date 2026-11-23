using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace HorseRacingML.Tests
{
    public class RacingTvDataTests
    {
        [Fact]
        public void SectionalAndBreedingBlocksParseIndependently()
        {
            Assert.Equal(
                FeatureBlocks.Pace | FeatureBlocks.Sectionals,
                FeatureBlockNames.Parse("pace,sectionals"));
            Assert.Equal(
                FeatureBlocks.Pace | FeatureBlocks.Breeding,
                FeatureBlockNames.Parse("pace,pedigree"));
            Assert.True(FeatureBlockNames.Parse("all").HasFlag(FeatureBlocks.Sectionals));
            Assert.True(FeatureBlockNames.Parse("all").HasFlag(FeatureBlocks.Breeding));
            Assert.True(FeatureBlocks.Default.HasFlag(FeatureBlocks.Sectionals));
            Assert.True(FeatureBlocks.Default.HasFlag(FeatureBlocks.Breeding));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("SectionalAvgFinishSpeedPctLast5", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("SectionalAvgFinishSpeedPctLast5", FeatureBlocks.Sectionals));
            Assert.False(HyperparameterTrainer.IsFeatureEnabled("PedigreeSireWinRate", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("PedigreeSireWinRate", FeatureBlocks.Breeding));
        }

        [Fact]
        public void CourseAliasesMatchTheStoredCourseNames()
        {
            Assert.True(RacingTvDataParser.CourseNamesMatch("Epsom Downs", "Epsom Downs"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Epsom Downs", "Epsom"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Lingfield Park", "Lingfield"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Hamilton Park", "Hamilton"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Royal Ascot", "Ascot"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Stratford-on-Avon", "Stratford"));
            Assert.False(RacingTvDataParser.CourseNamesMatch("Epsom Downs", "Newmarket"));
        }

        [Fact]
        public void RaceMatchingUsesTheUtcScheduledTime()
        {
            var candidates = new List<RaceBackfillCandidate>
            {
                new()
                {
                    RaceId = 10,
                    CourseName = "Epsom Downs",
                    ScheduledOff = new TimeSpan(12, 25, 0),
                    Title = "The Derby"
                },
                new()
                {
                    RaceId = 11,
                    CourseName = "Epsom Downs",
                    ScheduledOff = new TimeSpan(13, 0, 0),
                    Title = "Test Handicap"
                }
            };

            var matched = RacingTvDataParser.MatchRace(
                candidates,
                "Epsom Downs",
                new DateTimeOffset(2024, 6, 1, 13, 25, 0, TimeSpan.FromHours(1)),
                "The Derby");

            Assert.NotNull(matched);
            Assert.Equal(10, matched!.RaceId);
        }

        [Fact]
        public void SectionalSummaryUsesTheLastThreeFurlongs()
        {
            var race = new RacingTvSectionalRaceInfo
            {
                TimeIndexScore = 6.8,
                MeetingAverageTimeIndex = 5.9,
                WinTimeVsPar = -1.2,
                Runners = Enumerable.Range(0, 10)
                    .Select(_ => new RacingTvSectionalRunner())
                    .ToList()
            };
            var runner = new RacingTvSectionalRunner
            {
                ClothNumber = 9,
                FinishPosition = 1,
                HorseName = "Zodiac Bear",
                FinishingSpeedPercentage = "107.86",
                Sectionals = new List<RacingTvSectionalSplit>
                {
                    Split(0, 19.47, 7, null),
                    Split(1, 12.84, 7, 0.55),
                    Split(2, 12.21, 7, 0.73),
                    Split(3, 11.88, 6, 0.83),
                    Split(4, 11.54, 6, 0.93),
                    Split(5, 11.30, 1, 1.0),
                    Split(6, 12.09, 1, 0.77)
                }
            };

            var row = RacingTvDataParser.BuildRunnerSectional(runner, race, 1, 2, "123");

            Assert.NotNull(row);
            Assert.Equal((byte?)7, row!.SectionCount);
            Assert.Equal(12.09f, row.Last1FSeconds!.Value, 2);
            Assert.Equal(23.39f, row.Last2FSeconds!.Value, 2);
            Assert.Equal(34.93f, row.Last3FSeconds!.Value, 2);
            Assert.Equal(0.9f, row.Last3FPercentile!.Value, 2);
            Assert.Equal(107.86f, row.FinishSpeedPct!.Value, 2);
            Assert.Equal(1f, row.ClosingPositionNorm!.Value, 2);
            Assert.True(row.PaceStyle > 0f);
        }

        [Fact]
        public void CurrentRaceSectionalAndPedigreeDoNotLeakIntoTheirOwnFeatures()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
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

            var raceOne = new List<Dictionary<string, object?>>
            {
                CreateRunner(1, 1, 1, "Galileo", "Kind", "Danehill", 107.5f, 0.9f, 34.5f),
                CreateRunner(1, 2, 2, "Galileo", "Other Mare", "Sadler's Wells", 105.0f, 0.6f, 35.1f)
            };
            processRace!.Invoke(state, new object?[] { raceOne, true, true });

            var firstWinner = raceOne[0];
            Assert.False(Convert.ToBoolean(firstWinner["SectionalHasHistory"]));
            Assert.Equal(0, Convert.ToInt32(firstWinner["PedigreeSireStarts"]));
            Assert.True(Convert.ToBoolean(firstWinner["PedigreeKnown"]));

            var raceTwo = new List<Dictionary<string, object?>>
            {
                CreateRunner(2, 1, 1, "Galileo", "Kind", "Danehill", 108.0f, 0.95f, 34.0f),
                CreateRunner(2, 3, 2, "Galileo", "Third Mare", "Danehill", 104.0f, 0.5f, 36.0f)
            };
            processRace.Invoke(state, new object?[] { raceTwo, true, true });

            var secondWinner = raceTwo[0];
            Assert.True(Convert.ToBoolean(secondWinner["SectionalHasHistory"]));
            Assert.Equal(1, Convert.ToInt32(secondWinner["SectionalStarts"]));
            Assert.Equal(107.5f, Convert.ToSingle(secondWinner["SectionalLastFinishSpeedPct"]), 2);
            Assert.True(Convert.ToSingle(secondWinner["PedigreeSireStarts"]) >= 2f);
            Assert.True(Convert.ToSingle(secondWinner["PedigreeSireWinRate"]) > 0f);
        }

        private static Dictionary<string, object?> CreateRunner(
            int raceId,
            int horseId,
            int finishPos,
            string sireName,
            string damName,
            string damSireName,
            float finishSpeedPct,
            float last3FPercentile,
            float last3FSeconds)
        {
            return new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["RaceId"] = raceId,
                ["HorseId"] = horseId,
                ["CourseId"] = 1,
                ["RaceDate"] = new DateTime(2024, 1, raceId, 13, 0, 0),
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
                ["SP_Decimal"] = 2.0m,
                ["OpeningFraction"] = null,
                ["TouchedHighFraction"] = null,
                ["TouchedLowFraction"] = null,
                ["Purse"] = 1000m,
                ["HorseName"] = $"Horse {horseId}",
                ["TrainerName"] = "Trainer",
                ["JockeyName"] = "Jockey",
                ["CourseName"] = "Course",
                ["CurrentSectionalCount"] = 6,
                ["CurrentSectionalLast1FSeconds"] = 12f,
                ["CurrentSectionalLast2FSeconds"] = 24f,
                ["CurrentSectionalLast3FSeconds"] = last3FSeconds,
                ["CurrentSectionalLast3FPercentile"] = last3FPercentile,
                ["CurrentSectionalEarlyPositionNorm"] = 0.4f,
                ["CurrentSectionalClosingPositionNorm"] = last3FPercentile,
                ["CurrentSectionalPaceStyle"] = last3FPercentile - 0.4f,
                ["CurrentSectionalFinishSpeedPct"] = finishSpeedPct,
                ["SireName"] = sireName,
                ["DamName"] = damName,
                ["DamSireName"] = damSireName
            };
        }

        private static RacingTvSectionalSplit Split(
            int number,
            double time,
            int position,
            double? percentile)
        {
            return new RacingTvSectionalSplit
            {
                SectorNumber = number,
                SectorTime = time,
                SectorPosition = position,
                SectorTimePercentile = percentile
            };
        }
    }
}
