using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace HorseRacingML.Tests
{
    public class FeatureAndBettingTests
    {
        [Theory]
        [InlineData("All Weather", "allweather")]
        [InlineData("All-weather", "allweather")]
        [InlineData("Allweather", "allweather")]
        [InlineData("Turf", "turf")]
        [InlineData("Dirt", "dirt")]
        public void SurfaceValuesAreCanonicalised(string input, string expected)
        {
            Assert.Equal(expected, FeatureText.NormalizeSurface(input));
        }

        [Fact]
        public void HorseNamesIgnoreBracketedSuffixes()
        {
            Assert.Equal("solar flair", FeatureText.NormalizeHorseName("Solar Flair (IRE)"));
            Assert.Equal("solar flair", FeatureText.NormalizeHorseName("  Solar   Flair "));
        }

        [Fact]
        public void EffectiveOddsAccountForCommission()
        {
            Assert.Equal(4.8m, BettingMath.CalculateEffectiveDecimalOdds(5m, 0.05m));
            Assert.Equal(5m, BettingMath.CalculateEffectiveDecimalOdds(5m, 0m));
        }

        [Fact]
        public void RoiUsesTurnoverNotStartingBankroll()
        {
            var simulation = new ValidationSimulationResult
            {
                StartingBankroll = 100m,
                EndingBankroll = 110m,
                TotalStaked = 200m,
                BetCount = 2,
                WinCount = 1
            };

            Assert.Equal(10m, simulation.Profit);
            Assert.Equal(0.05m, simulation.Roi);
        }

        [Theory]
        [InlineData("DistanceBeatenLengths")]
        [InlineData("DistanceBeatenKnown")]
        [InlineData("WinningTimeMs")]
        [InlineData("RunnerSpeed")]
        [InlineData("SpeedRatio")]
        public void ResultAndLiveUnavailableFeaturesAreExcluded(string key)
        {
            Assert.False(HyperparameterTrainer.IsLiveAvailableFeature(key));
        }

        [Theory]
        [InlineData("Draw")]
        [InlineData("Going")]
        [InlineData("Surface")]
        [InlineData("CareerStarts")]
        [InlineData("LifetimeWinRate")]
        [InlineData("FrontRunRateLast5")]
        [InlineData("ProminentRateLast5")]
        [InlineData("LoneSpeedFlag")]
        [InlineData("PacePressure")]
        [InlineData("FrontRunnerCount")]
        [InlineData("RainPrev1dMm")]
        [InlineData("RainPrev7dMm")]
        [InlineData("RainRaceDayMm")]
        [InlineData("WeatherGoingIndex")]
        [InlineData("WeatherSoftRisk")]
        [InlineData("DaysSinceLastMeeting")]
        [InlineData("GoingIndex")]
        [InlineData("GoingWeatherDelta")]
        [InlineData("TrainerWinRate")]
        [InlineData("AgeRelative")]
        public void LiveAvailableFeaturesAreKept(string key)
        {
            Assert.True(HyperparameterTrainer.IsLiveAvailableFeature(key));
        }

        [Fact]
        public void FeatureBlocksAreOptInAndIndependent()
        {
            Assert.Equal(FeatureBlocks.Default, FeatureBlockNames.Parse(null));
            Assert.Equal(FeatureBlocks.Default, FeatureBlockNames.Parse("default"));
            Assert.True(FeatureBlockNames.Parse(null).HasFlag(FeatureBlocks.Sectionals));
            Assert.True(FeatureBlockNames.Parse(null).HasFlag(FeatureBlocks.Breeding));
            Assert.Equal(
                FeatureBlocks.Pace | FeatureBlocks.FormContext,
                FeatureBlockNames.Parse("pace,form"));
            Assert.Equal(FeatureBlocks.None, FeatureBlockNames.Parse("none"));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("TrainerWinRate", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("TrainerWinRate", FeatureBlocks.TrainerForm));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("AgeRelative", FeatureBlocks.TrainerForm));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("FormLayoffVsTypical", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("FormLayoffVsTypical", FeatureBlocks.FormContext));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("WeatherBalance7dMm", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("WeatherBalance7dMm", FeatureBlocks.WeatherBalance));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("RaceContextMeetingDay", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("RaceContextMeetingDay", FeatureBlocks.RaceContext));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("MarketMoveRangeLog", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("MarketMoveRangeLog", FeatureBlocks.MarketMovement));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("OfficialRating", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("OfficialRating", FeatureBlocks.PriceData));
            Assert.False(HyperparameterTrainer.IsFeatureEnabled("FieldRatingMean", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("FieldRatingMean", FeatureBlocks.PriceData));
            Assert.False(HyperparameterTrainer.IsFeatureEnabled("OfficialRatingRankRelative", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("OfficialRatingRankRelative", FeatureBlocks.PriceData));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("WeatherHumidity", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("WeatherHumidity", FeatureBlocks.WeatherExtended));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("TrackGeometryStraight", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("TrackGeometryStraight", FeatureBlocks.TrackGeometry));

            Assert.False(HyperparameterTrainer.IsFeatureEnabled("DeepFormH2HStarts", FeatureBlocks.Pace));
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("DeepFormH2HStarts", FeatureBlocks.DeepForm));

            // LastTrainer* was already in the pre-block schema and must stay
            // there when only Pace is selected.
            Assert.True(HyperparameterTrainer.IsFeatureEnabled("LastTrainerSurfaceNormPos", FeatureBlocks.Pace));

            Assert.True(CourseGeometryCatalog.TryGet("Cheltenham", out var cheltenham));
            Assert.True(cheltenham.LeftHanded);
            Assert.Contains(3, cheltenham.FestivalMonths);
            Assert.False(CourseGeometryCatalog.TryGet("Not A Racecourse", out _));
        }

        [Fact]
        public void ProcessRaceDoesNotLeakTheCurrentResultIntoHistoryFeatures()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType("FeatureEngineeringState", BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod("ProcessRace", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);

            var raceOne = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 1, horseId: 1, finishPos: 1, beatenText: null, day: 1),
                CreateRunner(raceId: 1, horseId: 2, finishPos: 2, beatenText: "2", day: 1)
            };

            processRace!.Invoke(state, new object?[] { raceOne, true, true });

            var winner = raceOne[0];
            Assert.Equal(0, Convert.ToInt32(winner["CareerStarts"]));
            Assert.False(Convert.ToBoolean(winner["HasLastWin"]));
            Assert.True(Convert.ToBoolean(winner["IsFirstStart"]));
            Assert.Equal(0.5f, Convert.ToSingle(winner["Last1NormPos"]), 3);
            Assert.Equal(0f, Convert.ToSingle(winner["DistanceBeatenLengths"]), 3);
            Assert.False(Convert.ToBoolean(winner["DistanceBeatenKnown"]));

            var loser = raceOne[1];
            Assert.Equal(2f, Convert.ToSingle(loser["DistanceBeatenLengths"]), 3);
            Assert.True(Convert.ToBoolean(loser["DistanceBeatenKnown"]));

            var raceTwo = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 2, horseId: 1, finishPos: 1, beatenText: null, day: 2),
                CreateRunner(raceId: 2, horseId: 3, finishPos: 2, beatenText: "1", day: 2)
            };

            processRace.Invoke(state, new object?[] { raceTwo, true, true });

            var horseOne = raceTwo[0];
            Assert.Equal(1, Convert.ToInt32(horseOne["CareerStarts"]));
            Assert.True(Convert.ToBoolean(horseOne["HasLastWin"]));
            Assert.Equal(1f, Convert.ToSingle(horseOne["Last1NormPos"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["DaysSinceLastRace"]), 3);
        }

        [Fact]
        public void ProcessRaceBuildsFieldStrengthFromOfficialRatings()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType("FeatureEngineeringState", BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod("ProcessRace", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);

            var race = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 10, horseId: 1, finishPos: 1, beatenText: null, day: 3),
                CreateRunner(raceId: 10, horseId: 2, finishPos: 2, beatenText: "2", day: 3),
                CreateRunner(raceId: 10, horseId: 3, finishPos: 3, beatenText: "4", day: 3)
            };
            race[0]["OfficialRating"] = 90;
            race[1]["OfficialRating"] = 80;
            race[2]["OfficialRating"] = 70;

            processRace!.Invoke(state, new object?[] { race, true, true });

            var top = race[0];
            var middle = race[1];
            var bottom = race[2];

            Assert.Equal(80f, Convert.ToSingle(top["FieldRatingMean"]), 3);
            Assert.Equal(90f, Convert.ToSingle(top["FieldRatingMax"]), 3);
            Assert.Equal(70f, Convert.ToSingle(top["FieldRatingMin"]), 3);
            Assert.Equal(20f, Convert.ToSingle(top["FieldRatingRange"]), 3);
            Assert.Equal(80f, Convert.ToSingle(top["FieldRatingTop3Mean"]), 3);
            Assert.Equal(8.164966f, Convert.ToSingle(top["FieldRatingStdDev"]), 3);
            Assert.Equal(3, Convert.ToInt32(top["FieldRatingKnownCount"]));
            Assert.Equal(1f, Convert.ToSingle(top["FieldRatingCoverage"]), 3);

            Assert.Equal(0f, Convert.ToSingle(top["OfficialRatingVsFieldMax"]), 3);
            Assert.Equal(-10f, Convert.ToSingle(middle["OfficialRatingVsFieldMax"]), 3);
            Assert.Equal(-20f, Convert.ToSingle(bottom["OfficialRatingVsFieldMax"]), 3);

            Assert.Equal(1f / 3f, Convert.ToSingle(top["OfficialRatingRankRelative"]), 3);
            Assert.Equal(2f / 3f, Convert.ToSingle(middle["OfficialRatingRankRelative"]), 3);
            Assert.Equal(1f, Convert.ToSingle(bottom["OfficialRatingRankRelative"]), 3);
        }

        [Fact]
        public void OptInFeatureBlocksWriteTheirExpectedColumns()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;",
                    ["Training:FeatureBlocks"] = "all"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType("FeatureEngineeringState", BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod("ProcessRace", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);

            var runner = CreateRunner(raceId: 1, horseId: 1, finishPos: 1, beatenText: null, day: 1);
            runner["OpeningFraction"] = "2/1";
            runner["TouchedLowFraction"] = "7/4";
            runner["RainPrev1dMm"] = 1.0;
            runner["RainPrev3dMm"] = 6.0;
            runner["RainPrev7dMm"] = 10.0;
            runner["WeatherEt0Prev7dMm"] = 5.0;

            var race = new List<Dictionary<string, object?>> { runner };
            processRace!.Invoke(state, new object?[] { race, true, true });

            Assert.True(runner.ContainsKey("TrainerWinRate"));
            Assert.True(runner.ContainsKey("AgeRelative"));
            Assert.True(runner.ContainsKey("WeatherBalance7dMm"));
            Assert.Equal(5f, Convert.ToSingle(runner["WeatherBalance7dMm"]), 3);
            Assert.True(runner.ContainsKey("RaceContextCourse"));
            Assert.Equal("1", Convert.ToString(runner["RaceContextCourse"]));
            Assert.Equal(1, Convert.ToInt32(runner["RaceContextMeetingDay"]));
            Assert.Equal(1, Convert.ToInt32(runner["RaceContextMeetingRaceIndex"]));
            Assert.True(runner.ContainsKey("FormLayoffVsTypical"));
            Assert.True(runner.ContainsKey("FormLastRunNonFinisher"));
            Assert.True(runner.ContainsKey("MarketMoveOpeningToLowLog"));
            Assert.True(Convert.ToBoolean(runner["MarketMoveShortenedFlag"]));

            // Trimming must not remove the categorical block columns before
            // the metadata pass sees them.
            var trim = typeof(HyperparameterTrainer).GetMethod(
                "TrimRunnerRow",
                BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(trim);
            trim!.Invoke(null, new object?[] { runner, null });
            Assert.True(runner.ContainsKey("RaceContextCourse"));
            Assert.True(runner.ContainsKey("RaceContextCourseDistance"));

            // The accumulator must turn the course string into a categorical
            // dimension rather than silently dropping it.
            var accumulatorType = typeof(HyperparameterTrainer).GetNestedType(
                "FeatureMetadataAccumulator",
                BindingFlags.NonPublic);
            Assert.NotNull(accumulatorType);
            var accumulator = Activator.CreateInstance(
                accumulatorType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { false, FeatureBlocks.Pace | FeatureBlocks.RaceContext },
                culture: null);
            Assert.NotNull(accumulator);
            var observe = accumulatorType!.GetMethod("ObserveRow", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var build = accumulatorType.GetMethod("Build", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(observe);
            Assert.NotNull(build);
            observe!.Invoke(accumulator, new object?[] { runner });
            var metadata = build!.Invoke(accumulator, null);
            Assert.NotNull(metadata);
            var featureKeys = (List<string>)metadata!.GetType().GetProperty("FeatureKeys")!.GetValue(metadata)!;
            var featureDimensions = (Dictionary<string, int>)metadata.GetType().GetProperty("FeatureDimensions")!.GetValue(metadata)!;
            Assert.Contains("RaceContextCourse", featureKeys);
            Assert.Contains("RaceContextCourseDistance", featureKeys);
            Assert.True(featureDimensions["RaceContextCourse"] > 1);
            Assert.True(featureDimensions["RaceContextCourseDistance"] > 1);
        }

        [Fact]
        public void ProcessRaceCarriesRollingStatsIntoLaterRaces()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType("FeatureEngineeringState", BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod("ProcessRace", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);

            var raceOne = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 1, horseId: 1, finishPos: 1, beatenText: null, day: 1),
                CreateRunner(raceId: 1, horseId: 2, finishPos: 2, beatenText: "2", day: 1)
            };

            processRace!.Invoke(state, new object?[] { raceOne, true, true });

            // Nothing about race one may appear in race one's own features.
            var horseOneRaceOne = raceOne[0];
            const float noHistoryPrior = 1f / 9f; // alpha=1, beta=9
            Assert.Equal(0, Convert.ToInt32(horseOneRaceOne["CareerStarts"]));
            Assert.True(Convert.ToBoolean(horseOneRaceOne["IsFirstStart"]));
            Assert.Equal(0, Convert.ToInt32(horseOneRaceOne["GoingStarts"]));
            Assert.Equal(0.5f, Convert.ToSingle(horseOneRaceOne["Last1NormPos"]), 3);
            Assert.Equal(noHistoryPrior, Convert.ToSingle(horseOneRaceOne["TrainerWinRate"]), 3);
            Assert.Equal(noHistoryPrior, Convert.ToSingle(horseOneRaceOne["JockeyWinRate"]), 3);

            var raceTwo = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 2, horseId: 1, finishPos: 1, beatenText: null, day: 2),
                CreateRunner(raceId: 2, horseId: 3, finishPos: 2, beatenText: "1", day: 2)
            };

            processRace.Invoke(state, new object?[] { raceTwo, true, true });

            var horseOne = raceTwo[0];
            const float oneWinOneStart = 2f / 10f; // (1 + alpha) / (1 + beta), alpha=1, beta=9
            Assert.Equal(1, Convert.ToInt32(horseOne["CareerStarts"]));
            Assert.True(Convert.ToBoolean(horseOne["HasLastWin"]));
            Assert.Equal(1, Convert.ToInt32(horseOne["GoingStarts"]));
            Assert.Equal(1, Convert.ToInt32(horseOne["CourseStarts"]));
            Assert.Equal(1, Convert.ToInt32(horseOne["ClassStarts"]));
            Assert.Equal(1f, Convert.ToSingle(horseOne["Last1NormPos"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["AvgNormPosLast1"]), 3);
            Assert.Equal(70f, Convert.ToSingle(horseOne["AvgRatingLast1"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["JockeyWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerWinRateLast50"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerCourseWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["JockeyCourseWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerJockeyWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerJockeySurfaceWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerClassWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["JockeyClassWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerSurfaceWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["JockeySurfaceWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerGoingWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["JockeyGoingWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["TrainerDistanceBucketWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["JockeyDistanceBucketWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["GoingWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["SurfaceWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["ClassWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["DistanceBucketWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["GoingCourseWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["AgeRestrictionWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["GoingDistanceWinRate"]), 3);
            Assert.Equal(oneWinOneStart, Convert.ToSingle(horseOne["JockeyGoingDistanceWinRate"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["LastCourseNormPos"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["LastAgeRestrictionNormPos"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["LastJockeyGoingDistanceNormPos"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["DaysSinceLastRace"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["DistanceRatioFromAverage"]), 3);
            Assert.Equal(0f, Convert.ToSingle(horseOne["DistanceChangeFromLast"]), 3);
            Assert.Equal(2f / 10f - 2f / 11f, Convert.ToSingle(horseOne["DrawBias"]), 3);
        }

        [Fact]
        public void RunStyleHistoryFeedsRollingAndFieldPaceFeatures()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType("FeatureEngineeringState", BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod("ProcessRace", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);

            var raceOne = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 1, horseId: 1, finishPos: 1, beatenText: null, day: 1),
                CreateRunner(raceId: 1, horseId: 2, finishPos: 2, beatenText: "2", day: 1),
                CreateRunner(raceId: 1, horseId: 3, finishPos: 3, beatenText: "4", day: 1)
            };
            raceOne[0]["Comment"] = "Led, ridden out";
            raceOne[1]["Comment"] = "Held up in rear, ran on";
            raceOne[2]["Comment"] = "Mid-division, pushed along";
            foreach (var row in raceOne) row["RunnerCount"] = raceOne.Count;

            processRace!.Invoke(state, new object?[] { raceOne, true, true });

            // The current race's own comment must not appear in its features.
            Assert.Equal(0, Convert.ToInt32(raceOne[0]["PaceStartsLast1"]));
            Assert.False(Convert.ToBoolean(raceOne[0]["HasPaceHistory"]));
            Assert.Equal(0.10f, Convert.ToSingle(raceOne[0]["FrontRunRateLast1"]), 3);
            Assert.False(Convert.ToBoolean(raceOne[0]["LastRunStyleKnown"]));

            var raceTwo = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 2, horseId: 1, finishPos: 1, beatenText: null, day: 2),
                CreateRunner(raceId: 2, horseId: 2, finishPos: 2, beatenText: "1", day: 2),
                CreateRunner(raceId: 2, horseId: 3, finishPos: 3, beatenText: "3", day: 2)
            };
            raceTwo[0]["Comment"] = "Led, ridden out";
            raceTwo[1]["Comment"] = "Held up in rear, ran on";
            raceTwo[2]["Comment"] = "Mid-division, pushed along";
            foreach (var row in raceTwo) row["RunnerCount"] = raceTwo.Count;

            processRace.Invoke(state, new object?[] { raceTwo, true, true });

            var horseOne = raceTwo[0];
            Assert.Equal(1, Convert.ToInt32(horseOne["PaceStartsLast1"]));
            Assert.Equal(1f, Convert.ToSingle(horseOne["LastRunStyle"]), 3);
            Assert.True(Convert.ToBoolean(horseOne["LastRunStyleKnown"]));
            Assert.True(Convert.ToBoolean(horseOne["HasPaceHistory"]));
            Assert.Equal(0.325f, Convert.ToSingle(horseOne["FrontRunRateLast1"]), 3);
            Assert.Equal(0.4375f, Convert.ToSingle(horseOne["ProminentRateLast1"]), 3);
            Assert.Equal(1f, Convert.ToSingle(horseOne["AvgPaceScoreLast1"]), 3);

            var horseTwo = raceTwo[1];
            Assert.True(Convert.ToBoolean(horseTwo["HasPaceHistory"]));
            Assert.Equal(5f, Convert.ToSingle(horseTwo["LastRunStyle"]), 3);
            Assert.Equal(0.34f, Convert.ToSingle(horseTwo["HeldUpRateLast1"]), 3);

            var horseThree = raceTwo[2];
            Assert.Equal(0.37f, Convert.ToSingle(horseThree["MidfieldRateLast1"]), 3);

            var raceThree = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 3, horseId: 1, finishPos: 1, beatenText: null, day: 3),
                CreateRunner(raceId: 3, horseId: 2, finishPos: 2, beatenText: "1", day: 3),
                CreateRunner(raceId: 3, horseId: 4, finishPos: 3, beatenText: "2", day: 3)
            };
            foreach (var row in raceThree) row["RunnerCount"] = raceThree.Count;

            processRace.Invoke(state, new object?[] { raceThree, true, true });

            var frontRunner = raceThree[0];
            var heldUpRunner = raceThree[1];
            var unknownRunner = raceThree[2];

            Assert.Equal(0.46f, Convert.ToSingle(frontRunner["FrontRunRateLast5"]), 3);
            Assert.Equal(0.06f, Convert.ToSingle(heldUpRunner["FrontRunRateLast5"]), 3);
            Assert.Equal(1, Convert.ToInt32(frontRunner["FrontRunnerCount"]));
            Assert.Equal(1, Convert.ToInt32(frontRunner["ProminentRunnerCount"]));
            Assert.True(Convert.ToBoolean(frontRunner["LoneSpeedFlag"]));
            Assert.True(Convert.ToBoolean(frontRunner["IsLoneFrontRunner"]));
            Assert.False(Convert.ToBoolean(heldUpRunner["IsFrontRunner"]));
            Assert.Equal(2, Convert.ToInt32(frontRunner["KnownPaceRunnerCount"]));
            Assert.Equal(1, Convert.ToInt32(frontRunner["UnknownPaceRunnerCount"]));
            Assert.Equal(0.5f, Convert.ToSingle(frontRunner["FieldPaceContention"]), 3);
            Assert.Equal(0.46f, Convert.ToSingle(frontRunner["FieldMaxFrontRunRate"]), 3);
            Assert.Equal(0.06f, Convert.ToSingle(frontRunner["FieldMinFrontRunRate"]), 3);
            Assert.Equal(0.26f, Convert.ToSingle(frontRunner["FieldAvgFrontRunRate"]), 3);
            Assert.Equal(0.06f, Convert.ToSingle(frontRunner["PacePressure"]), 3);
            Assert.Equal(0.46f, Convert.ToSingle(heldUpRunner["PacePressure"]), 3);
            Assert.Equal(0.52f, Convert.ToSingle(unknownRunner["PacePressure"]), 3);
            Assert.Equal(0.20f, Convert.ToSingle(frontRunner["OwnFrontRunMinusFieldMean"]), 3);
            Assert.Equal(-0.20f, Convert.ToSingle(heldUpRunner["OwnFrontRunMinusFieldMean"]), 3);
            Assert.Equal(0f, Convert.ToSingle(unknownRunner["OwnFrontRunMinusFieldMean"]), 3);
            Assert.Equal(0.472f, Convert.ToSingle(frontRunner["FieldMaxHeldUpRate"]), 3);
            Assert.True(frontRunner.ContainsKey("FrontRunRateRankRelative"));
            Assert.True(frontRunner.ContainsKey("MarketRankMinusFrontRunRank"));
        }

        [Fact]
        public void PaceFeaturesCanBeDisabledForAblation()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;",
                    ["Training:EnablePaceFeatures"] = "false"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType("FeatureEngineeringState", BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod("ProcessRace", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);

            var race = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 1, horseId: 1, finishPos: 1, beatenText: null, day: 1),
                CreateRunner(raceId: 1, horseId: 2, finishPos: 2, beatenText: "2", day: 1)
            };
            race[0]["Comment"] = "Led, ridden out";
            race[1]["Comment"] = "Held up in rear, ran on";
            foreach (var row in race) row["RunnerCount"] = race.Count;

            processRace!.Invoke(state, new object?[] { race, true, true });

            Assert.False(race[0].ContainsKey("FrontRunRateLast5"));
            Assert.False(race[0].ContainsKey("PaceStartsLast1"));
            Assert.False(race[0].ContainsKey("LoneSpeedFlag"));
            Assert.False(race[0].ContainsKey("FieldMaxFrontRunRate"));
            Assert.True(race[0].ContainsKey("WinRateLast5"));
        }

        [Fact]
        public void MarketFeaturesUsePreOffOpeningPricesAndDeVig()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var stateType = typeof(HyperparameterTrainer).GetNestedType("FeatureEngineeringState", BindingFlags.NonPublic);
            Assert.NotNull(stateType);

            var state = Activator.CreateInstance(
                stateType!,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                args: new object?[] { trainer, null },
                culture: null);
            Assert.NotNull(state);

            var processRace = stateType!.GetMethod("ProcessRace", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(processRace);

            var race = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 1, horseId: 1, finishPos: 1, beatenText: null, day: 1),
                CreateRunner(raceId: 1, horseId: 2, finishPos: 2, beatenText: "2", day: 1)
            };

            // Opening prices: 2/1 (3.0) and 1/1 (2.0). SP_Decimal is 2.0 for
            // both, so a test that accidentally used SP would produce 0.5/0.5
            // instead of the de-vigged 0.4/0.6 below.
            race[0]["OpeningFraction"] = "2/1";
            race[1]["OpeningFraction"] = "1/1";

            processRace!.Invoke(state, new object?[] { race, true, true });

            var outsider = race[0];
            var favourite = race[1];

            Assert.Equal(0.4f, Convert.ToSingle(outsider["MarketImpliedProb"]), 3);
            Assert.Equal(0.6f, Convert.ToSingle(favourite["MarketImpliedProb"]), 3);
            Assert.Equal(-0.1f, Convert.ToSingle(outsider["MarketProbDiffFromMean"]), 3);
            Assert.Equal(0.1f, Convert.ToSingle(favourite["MarketProbDiffFromMean"]), 3);
            Assert.Equal((float)Math.Log(3.0), Convert.ToSingle(outsider["MarketLogOdds"]), 3);
            Assert.Equal(0.5f, Convert.ToSingle(favourite["MarketProbRankRelative"]), 3);
            Assert.False(Convert.ToBoolean(outsider["MarketProbMissing"]));

            // Live rows carry MarketDecimalOdds (the current back price). It must
            // take precedence over OpeningFraction and be de-vigged the same way.
            var liveRace = new List<Dictionary<string, object?>>
            {
                CreateRunner(raceId: 2, horseId: 4, finishPos: 2, beatenText: "1", day: 2),
                CreateRunner(raceId: 2, horseId: 5, finishPos: 1, beatenText: null, day: 2)
            };
            liveRace[0]["MarketDecimalOdds"] = 4.0m;
            liveRace[1]["MarketDecimalOdds"] = 2.0m;
            liveRace[0]["OpeningFraction"] = "1/1";
            liveRace[1]["OpeningFraction"] = "2/1";

            processRace.Invoke(state, new object?[] { liveRace, true, true });

            Assert.Equal(1f / 3f, Convert.ToSingle(liveRace[0]["MarketImpliedProb"]), 3);
            Assert.Equal(2f / 3f, Convert.ToSingle(liveRace[1]["MarketImpliedProb"]), 3);
        }

        [Fact]
        public void TrainCompletesWithPrebuiltTensorsAndEarlyStopping()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;",
                    ["Training:L2WeightDecay"] = "0.0001",
                    ["Training:EarlyStoppingPatience"] = "2",
                    ["Training:EvaluationFrequency"] = "1"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var featureKeys = new List<string> { "FeatureA", "FeatureB" };
            var featureDims = new Dictionary<string, int>
            {
                ["FeatureA"] = 1,
                ["FeatureB"] = 1
            };
            var normalization = new NormalizationParameters
            {
                Mean = new float[2],
                StdDev = new float[] { 1f, 1f }
            };

            var trainingRaces = new List<HyperparameterTrainer.RaceExample>();
            var validationRaces = new List<HyperparameterTrainer.RaceExample>();
            var random = new Random(42);
            for (int raceId = 0; raceId < 20; raceId++)
            {
                var runners = new List<HyperparameterTrainer.RunnerExample>();
                for (int runner = 0; runner < 6; runner++)
                {
                    float a = (float)random.NextDouble();
                    float b = (float)random.NextDouble();
                    float label = runner == 0 ? 1f : 0f;
                    runners.Add(new HyperparameterTrainer.RunnerExample(
                        raceId,
                        new[] { a, b },
                        label,
                        runner,
                        $"Horse {runner}",
                        null));
                }

                var race = new HyperparameterTrainer.RaceExample(raceId, DateTime.Today, runners);
                if (raceId < 15)
                {
                    trainingRaces.Add(race);
                }
                else
                {
                    validationRaces.Add(race);
                }
            }

            var dataset = new HyperparameterTrainer.TrainingDataset(
                trainingRaces,
                validationRaces,
                featureKeys,
                featureDims,
                new Dictionary<string, Dictionary<string, int>>(),
                normalization);

            var parameter = new MLParameter
            {
                Units = 8,
                Dropout = 0.1,
                Layers = 1,
                LearningRate = 0.01,
                Epochs = 3,
                BatchSize = 16,
                Folds = 1
            };

            var result = trainer.Train(parameter, 0, 1, dataset, persistWeights: false);

            Assert.True(double.IsFinite(result.TrainLoss));
            Assert.True(double.IsFinite(result.ValidationLoss));
            Assert.Equal(validationRaces.Sum(r => r.Runners.Count), result.ValidationPredictions.Count);
        }

        [Fact]
        public void RunnerExampleNormalisesFinishTargetWithinField()
        {
            var winner = new HyperparameterTrainer.RunnerExample(
                1, new[] { 0f }, 1f, 1, "Winner", null,
                finishPosition: 1, fieldSize: 8);
            var middle = new HyperparameterTrainer.RunnerExample(
                1, new[] { 0f }, 0f, 2, "Middle", null,
                finishPosition: 4, fieldSize: 8);
            var last = new HyperparameterTrainer.RunnerExample(
                1, new[] { 0f }, 0f, 3, "Last", null,
                finishPosition: 8, fieldSize: 8);
            var nonFinisher = new HyperparameterTrainer.RunnerExample(
                1, new[] { 0f }, 0f, 4, "Fell", null,
                finishPosition: null, fieldSize: 8);
            var inconsistent = new HyperparameterTrainer.RunnerExample(
                1, new[] { 0f }, 0f, 5, "Invalid", null,
                finishPosition: 9, fieldSize: 8);
            var soloRunner = new HyperparameterTrainer.RunnerExample(
                2, new[] { 0f }, 1f, 6, "Solo", null,
                finishPosition: 1, fieldSize: 1);

            Assert.True(winner.HasFinishTarget);
            Assert.Equal(1f, winner.FinishTarget, 6);
            Assert.True(middle.HasFinishTarget);
            Assert.Equal(4f / 7f, middle.FinishTarget, 6);
            Assert.True(last.HasFinishTarget);
            Assert.Equal(0f, last.FinishTarget, 6);
            Assert.False(nonFinisher.HasFinishTarget);
            Assert.False(inconsistent.HasFinishTarget);
            Assert.True(soloRunner.HasFinishTarget);
            Assert.Equal(1f, soloRunner.FinishTarget, 6);
        }

        [Fact]
        public void RaceSoftmaxTrainingWithAuxiliaryLossesCompletes()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;",
                    ["Training:L2WeightDecay"] = "0.0001",
                    ["Training:EarlyStoppingPatience"] = "2",
                    ["Training:EvaluationFrequency"] = "1",
                    ["Training:AuxiliaryWinLossWeight"] = "0.15",
                    ["Training:AuxiliaryFinishLossWeight"] = "0.35"
                })
                .Build();

            var trainer = new HyperparameterTrainer(configuration);
            var featureKeys = new List<string> { "FeatureA", "FeatureB" };
            var featureDims = new Dictionary<string, int>
            {
                ["FeatureA"] = 1,
                ["FeatureB"] = 1
            };
            var normalization = new NormalizationParameters
            {
                Mean = new float[2],
                StdDev = new float[] { 1f, 1f }
            };

            var trainingRaces = new List<HyperparameterTrainer.RaceExample>();
            var validationRaces = new List<HyperparameterTrainer.RaceExample>();
            var random = new Random(42);
            for (int raceId = 0; raceId < 20; raceId++)
            {
                var runners = new List<HyperparameterTrainer.RunnerExample>();
                for (int runner = 0; runner < 5; runner++)
                {
                    float a = (float)random.NextDouble();
                    float b = (float)random.NextDouble();
                    float label = runner == 0 ? 1f : 0f;
                    runners.Add(new HyperparameterTrainer.RunnerExample(
                        raceId,
                        new[] { a, b },
                        label,
                        runner,
                        $"Horse {runner}",
                        null,
                        finishPosition: runner + 1,
                        fieldSize: 5));
                }

                var race = new HyperparameterTrainer.RaceExample(raceId, DateTime.Today, runners);
                if (raceId < 15)
                {
                    trainingRaces.Add(race);
                }
                else
                {
                    validationRaces.Add(race);
                }
            }

            var dataset = new HyperparameterTrainer.TrainingDataset(
                trainingRaces,
                validationRaces,
                featureKeys,
                featureDims,
                new Dictionary<string, Dictionary<string, int>>(),
                normalization);

            var parameter = new MLParameter
            {
                Units = 8,
                Dropout = 0.1,
                Layers = 1,
                LearningRate = 0.01,
                Epochs = 3,
                BatchSize = 16,
                RaceBatchSize = 8,
                Folds = 1,
                Objective = "RaceSoftmax",
                Seed = 42
            };

            var result = trainer.Train(parameter, 0, 1, dataset, persistWeights: false);

            Assert.True(double.IsFinite(result.TrainLoss));
            Assert.True(double.IsFinite(result.ValidationLoss));
            Assert.Equal(0.15f, result.AuxiliaryWinLossWeight);
            Assert.Equal(0.35f, result.AuxiliaryFinishLossWeight);
            Assert.Equal(validationRaces.Sum(r => r.Runners.Count), result.ValidationPredictions.Count);
        }

        [Fact]
        public void PricedSubsetMetricsIgnoreUnpricedRunnersAndUnpricedWinners()
        {
            var examples = new List<HyperparameterTrainer.RunnerExample>
            {
                new(1, new[] { 0f }, 1f, 1, "A", 2m, 0f, true),
                new(1, new[] { 0f }, 0f, 2, "B", 3m, 0f, true),
                new(2, new[] { 0f }, 0f, 3, "C", 2m, 0f, true),
                new(2, new[] { 0f }, 1f, 4, "D", 3m, 0f, false),
                new(3, new[] { 0f }, 1f, 5, "E", 2m, 0f, true),
                new(3, new[] { 0f }, 0f, 6, "F", 3m, 0f, false),
                new(3, new[] { 0f }, 0f, 7, "G", 4m, 0f, true)
            };

            var result = new HyperparameterTrainer.TrainingResult
            {
                ValidationPredictions = Array.AsReadOnly(new[] { 0.7f, 0.3f, 0.6f, 0.4f, 0.5f, 0.3f, 0.2f }),
                ValidationLabels = Array.AsReadOnly(new[] { 1f, 0f, 0f, 1f, 1f, 0f, 0f }),
                ValidationRaceIds = Array.AsReadOnly(new[] { 1, 1, 2, 2, 3, 3, 3 }),
                ValidationExamples = new System.Collections.ObjectModel.ReadOnlyCollection<HyperparameterTrainer.RunnerExample>(examples)
            };

            var metrics = HyperparameterTrainer.ComputePricedSubsetMetrics(result);

            // Race 2 is excluded because its winner is unpriced. Race 3 keeps
            // only the two priced runners and renormalises their predictions.
            Assert.Equal(2, metrics.RaceCount);
            Assert.Equal(4, metrics.RunnerCount);
            Assert.Equal(1.0, metrics.Accuracy, 6);
            Assert.True(double.IsFinite(metrics.LogLoss));
            Assert.True(double.IsFinite(metrics.Brier));
        }

        [Fact]
        public void RaceSoftmaxInferenceReturnsCoherentProbabilities()
        {
            var model = new TrainedModel
            {
                Objective = "RaceSoftmax",
                MarketOffsetScale = 0f,
                Temperature = 1f,
                HiddenLayers = new List<LayerWeights>
                {
                    new LayerWeights
                    {
                        Weights = new[] { new[] { 1f }, new[] { 1f } },
                        Bias = new[] { 0f }
                    }
                },
                OutputLayer = new LayerWeights
                {
                    Weights = new[] { new[] { 1f } },
                    Bias = new[] { 0f }
                },
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "FeatureA", "FeatureB" },
                    FeatureDimensions = new Dictionary<string, int>
                    {
                        ["FeatureA"] = 1,
                        ["FeatureB"] = 1
                    },
                    StringMaps = new Dictionary<string, Dictionary<string, int>>()
                },
                Normalization = new NormalizationParameters
                {
                    Mean = new[] { 0f, 0f },
                    StdDev = new[] { 1f, 1f }
                }
            };

            var path = Path.Combine(Path.GetTempPath(), $"race-softmax-{Guid.NewGuid():N}.json");
            File.WriteAllText(path, JsonSerializer.Serialize(model));
            try
            {
                var flows = new List<RunnerFlow>
                {
                    new RunnerFlow
                    {
                        HorseName = "Strong",
                        FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                        {
                            ["FeatureA"] = 2.0,
                            ["FeatureB"] = 0.0
                        },
                        HasPreparedFeatures = true
                    },
                    new RunnerFlow
                    {
                        HorseName = "Weak",
                        FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                        {
                            ["FeatureA"] = 0.0,
                            ["FeatureB"] = 1.0
                        },
                        HasPreparedFeatures = true
                    }
                };

                var calculator = new AIOddsCalculator(path);
                Assert.True(calculator.TryCalculateRaceOdds(flows, out var probabilities));
                Assert.Equal(2, probabilities.Length);
                Assert.Equal(1.0, probabilities.Sum(), 6);
                Assert.True(probabilities[0] > probabilities[1]);

                var configuration = new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                    })
                    .Build();
                var trainer = new HyperparameterTrainer(configuration);
                var race = new HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace(
                    1,
                    flows.Select(flow => new Dictionary<string, object?>(flow.FeatureValues!, StringComparer.OrdinalIgnoreCase)).ToList());
                var trainerProbabilities = trainer.PredictRaceProbabilities(race, model);
                Assert.Equal(2, trainerProbabilities.Count);
                Assert.Equal(1.0, trainerProbabilities.Sum(result => result.Probability), 6);
                Assert.Equal(probabilities[0], trainerProbabilities[0].Probability, 6);
                Assert.Equal(probabilities[1], trainerProbabilities[1].Probability, 6);
            }
            finally
            {
                try { File.Delete(path); } catch { }
            }
        }

        [Fact]
        public void BlendCalibrationIgnoresAnUninformativeModel()
        {
            var examples = new[]
            {
                new HyperparameterTrainer.RunnerExample(
                    1, new[] { 0f }, 1f, 1, "Winner", null,
                    marketProbability: 0.7f, marketLogProbability: (float)Math.Log(0.7)),
                new HyperparameterTrainer.RunnerExample(
                    1, new[] { 0f }, 0f, 2, "Loser", null,
                    marketProbability: 0.3f, marketLogProbability: (float)Math.Log(0.3))
            };
            var result = MarketBlendCalibration.Fit(
                new[] { 0.5f, 0.5f },
                new[] { 1f, 0f },
                new[] { 1, 1 },
                examples);

            Assert.True(result.HasData);
            Assert.True(result.ModelWeight <= 0.1,
                $"Expected the blend to ignore the model, got a={result.ModelWeight:F3}.");
            Assert.True(result.Metrics.LogLoss < 0.4);
        }

        [Fact]
        public void BlendCalibrationUsesAnInformativeModel()
        {
            var examples = new[]
            {
                new HyperparameterTrainer.RunnerExample(
                    1, new[] { 0f }, 1f, 1, "Winner", null,
                    marketProbability: 0.6f, marketLogProbability: (float)Math.Log(0.6)),
                new HyperparameterTrainer.RunnerExample(
                    1, new[] { 0f }, 0f, 2, "Loser", null,
                    marketProbability: 0.4f, marketLogProbability: (float)Math.Log(0.4))
            };
            var modelPredictions = new[] { 0.9f, 0.1f };
            var labels = new[] { 1f, 0f };
            var raceIds = new[] { 1, 1 };

            var marketOnly = MarketBlendCalibration.FitMarketOnly(
                modelPredictions, labels, raceIds, examples);
            var blend = MarketBlendCalibration.Fit(
                modelPredictions, labels, raceIds, examples);

            Assert.True(blend.ModelWeight > 0d);
            Assert.True(blend.Metrics.LogLoss < marketOnly.Metrics.LogLoss);
        }

        [Fact]
        public void LogProbOffsetReproducesTheMarketPrior()
        {
            var model = new TrainedModel
            {
                Objective = "RaceSoftmax",
                MarketOffsetMode = "LogProb",
                MarketOffsetScale = 1f,
                BlendModelWeight = 1f,
                BlendMarketWeight = 0f,
                Temperature = 1f,
                HiddenLayers = new List<LayerWeights>(),
                OutputLayer = new LayerWeights
                {
                    Weights = new[] { new[] { 0f } },
                    Bias = new[] { 0f }
                },
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "FeatureA" },
                    FeatureDimensions = new Dictionary<string, int> { ["FeatureA"] = 1 },
                    StringMaps = new Dictionary<string, Dictionary<string, int>>()
                },
                Normalization = new NormalizationParameters
                {
                    Mean = new[] { 0f },
                    StdDev = new[] { 1f }
                }
            };

            var path = Path.Combine(Path.GetTempPath(), $"logprob-offset-{Guid.NewGuid():N}.json");
            File.WriteAllText(path, JsonSerializer.Serialize(model));
            try
            {
                var flows = new List<RunnerFlow>
                {
                    new RunnerFlow
                    {
                        HorseName = "Favourite",
                        FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                        {
                            ["FeatureA"] = 1.0,
                            ["MarketImpliedProb"] = 0.7
                        },
                        HasPreparedFeatures = true
                    },
                    new RunnerFlow
                    {
                        HorseName = "Outsider",
                        FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                        {
                            ["FeatureA"] = 1.0,
                            ["MarketImpliedProb"] = 0.3
                        },
                        HasPreparedFeatures = true
                    }
                };

                var calculator = new AIOddsCalculator(path);
                Assert.True(calculator.TryCalculateRaceOdds(flows, out var probabilities));
                Assert.Equal(0.7, probabilities[0], 6);
                Assert.Equal(0.3, probabilities[1], 6);
            }
            finally
            {
                try { File.Delete(path); } catch { }
            }
        }

        [Fact]
        public void MissingMarketOffsetUsesThePricedRunnerMean()
        {
            var model = new TrainedModel
            {
                Objective = "RaceSoftmax",
                MarketOffsetMode = "LogProb",
                MarketOffsetScale = 1f,
                BlendModelWeight = 1f,
                BlendMarketWeight = 0f,
                Temperature = 1f,
                HiddenLayers = new List<LayerWeights>(),
                OutputLayer = new LayerWeights
                {
                    Weights = new[] { new[] { 0f } },
                    Bias = new[] { 0f }
                },
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "FeatureA" },
                    FeatureDimensions = new Dictionary<string, int> { ["FeatureA"] = 1 },
                    StringMaps = new Dictionary<string, Dictionary<string, int>>()
                },
                Normalization = new NormalizationParameters
                {
                    Mean = new[] { 0f },
                    StdDev = new[] { 1f }
                }
            };

            var path = Path.Combine(Path.GetTempPath(), $"neutral-offset-{Guid.NewGuid():N}.json");
            File.WriteAllText(path, JsonSerializer.Serialize(model));
            try
            {
                var flows = new List<RunnerFlow>
                {
                    new RunnerFlow
                    {
                        HorseName = "Priced",
                        FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                        {
                            ["FeatureA"] = 1.0,
                            ["MarketImpliedProb"] = 0.8
                        },
                        HasPreparedFeatures = true
                    },
                    new RunnerFlow
                    {
                        HorseName = "Unpriced",
                        FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                        {
                            ["FeatureA"] = 1.0
                        },
                        HasPreparedFeatures = true
                    }
                };

                var calculator = new AIOddsCalculator(path);
                Assert.True(calculator.TryCalculateRaceOdds(flows, out var probabilities));
                Assert.Equal(0.5, probabilities[0], 6);
                Assert.Equal(0.5, probabilities[1], 6);
            }
            finally
            {
                try { File.Delete(path); } catch { }
            }
        }

        private static Dictionary<string, object?> CreateRunner(int raceId, int horseId, int finishPos, string? beatenText, int day)
        {
            return new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["RaceId"] = raceId,
                ["HorseId"] = horseId,
                ["CourseId"] = 1,
                ["RaceDate"] = new DateTime(2024, 1, day, 13, 0, 0),
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
                ["DistanceBeatenText"] = beatenText,
                ["DistanceBeatenLengths"] = beatenText is null ? null : 2m,
                ["SP_Fraction"] = null,
                ["SP_Decimal"] = 2.0m,
                ["FavTag"] = null,
                ["OpeningFraction"] = null,
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
