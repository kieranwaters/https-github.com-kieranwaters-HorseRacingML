using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.Tests
{
    public class UpcomingRaceLookupTests
    {
        [Fact]
        public void FindUpcomingRace_RejectedWhenMetadataMismatch_SyntheticFallbackUsed()
        {
            var raceDate = new DateTime(2024, 5, 12);
            var candidates = new List<UpcomingRace>
            {
                new UpcomingRace
                {
                    UpcomingRaceId = 10,
                    RaceDate = raceDate,
                    MarketId = "1.234",
                    Title = "Kempton Sprint",
                    VenueName = "Kempton"
                },
                new UpcomingRace
                {
                    UpcomingRaceId = 11,
                    RaceDate = raceDate,
                    MarketId = "1.345",
                    Title = "Lingfield Dash",
                    VenueName = "Lingfield"
                }
            };

            var normalizedTitle = RacingRepository.NormalizeLookupKey("Ascot Gold Cup");
            var normalizedVenue = RacingRepository.NormalizeLookupKey("Ascot");

            var (best, score, titleAligned, venueAligned) = RacingRepository.SelectBestUpcomingRaceCandidate(
                candidates,
                normalizedTitle,
                normalizedVenue);

            Assert.NotNull(best);
            Assert.True(score > 0);
            Assert.False(titleAligned);
            Assert.False(venueAligned);

            var shouldUse = BetfairMarketScraper.ShouldUseUpcomingCandidate(
                best!,
                marketId: "1.999",
                raceTitle: "Ascot Gold Cup",
                venueName: "Ascot");

            Assert.False(shouldUse);
        }
        [Theory]
        [InlineData("Sophar Sogood", "sopharsogood")]
        [InlineData("Sophar So Good", "sopharsogood")]
        [InlineData("Masterdream (IRE)", "masterdream")]
        [InlineData("Rock & Roll", "rockandroll")]
        public void NormalizeHistoricalNameKey_StripsFormatting(string source, string expected)
        {
            var normalized = RacingRepository.NormalizeHistoricalNameKey(source);
            Assert.Equal(expected, normalized);
        }

        [Fact]
        public void BuildHistoricalNameCandidates_IncludesNormalizedVariant()
        {
            var candidates = RacingRepository.BuildHistoricalNameCandidates("Sophar Sogood");

            Assert.Contains("Sophar Sogood", candidates);
            Assert.Contains("sopharsogood", candidates, StringComparer.OrdinalIgnoreCase);
        }
        [Fact]
        public void BuildUpcomingRaceRows_UsesBatchedLookupsWhenAvailable()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var trainer = new BatchedLookupTrainer(configuration);
            var upcoming = new UpcomingRace
            {
                RaceDate = new DateTime(2024, 8, 21),
                MarketId = "1.555",
                Title = "Summer Stakes",
                VenueName = "Test Course",
                VenueCountry = "GB",
                RaceType = "Handicap",
                Surface = "Turf",
                Going = "Good",
                DistanceYards = 1760,
                DistanceText = "1m",
                RunnerCount = (byte)12,
                ScheduledOff = new TimeSpan(15, 0, 0)
            };

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Known Runner",
                    JockeyName = "Known Jockey",
                    ClothNumber = 1,
                    Draw = 3
                },
                new RunnerFlow
                {
                    HorseName = "Unknown Runner",
                    JockeyName = "Unknown Jockey",
                    ClothNumber = 2,
                    Draw = 7
                }
            };
            var runnerColumns = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "WeightText",
                "OfficialRating",
                "Age",
                "WeightLbs"
            };

            var rows = trainer.TestBuildUpcomingRaceRows(conn: null!, upcoming, flows, raceId: 999, runnerColumns);

            Assert.Equal(2, rows.Count);

            var knownRow = Assert.Single(rows.Where(r => string.Equals((string?)r["HorseName"], "Known Runner", StringComparison.OrdinalIgnoreCase)));
            var unknownRow = Assert.Single(rows.Where(r => string.Equals((string?)r["HorseName"], "Unknown Runner", StringComparison.OrdinalIgnoreCase)));

            Assert.Equal(999, Assert.IsType<int>(knownRow["RaceId"]));
            Assert.Equal(42, Assert.IsType<int>(knownRow["HorseId"]));
            Assert.Equal(9, Assert.IsType<int>(knownRow["JockeyId"]));
            Assert.Equal(77, Assert.IsType<int>(knownRow["TrainerId"]));
            Assert.Equal("Sample Trainer", Assert.IsType<string>(knownRow["TrainerName"]));
            Assert.Equal(5, Assert.IsType<int>(knownRow["Age"]));
            Assert.Equal(126, Assert.IsType<int>(knownRow["WeightLbs"]));
            Assert.Equal("9-0", Assert.IsType<string>(knownRow["WeightText"]));
            Assert.Equal(101, Assert.IsType<int>(knownRow["OfficialRating"]));

            Assert.NotEqual(42, Assert.IsType<int>(unknownRow["HorseId"]));
            Assert.False(unknownRow.ContainsKey("JockeyId"));
            Assert.Null(unknownRow["TrainerId"]);
            Assert.Null(unknownRow["TrainerName"]);
            Assert.Null(unknownRow["Age"]);
            Assert.Null(unknownRow["WeightLbs"]);
            Assert.Null(unknownRow["WeightText"]);
            Assert.Null(unknownRow["OfficialRating"]);
        }

        [Fact]
        public void LoadFeatureLookup_UsesPersistedUpcomingRaceMetadata()
        {
            var raceDate = new DateTime(2024, 7, 18);
            var marketId = "1.987654";
            var repo = new InMemoryRacingRepository();

            var preparedRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Solar Flair",
                ["TrainerName"] = "Jane Trainer",
                ["TrainerRunsLast365"] = 128,
                ["HorseRunsLast365"] = 9
            };
            var preparedRace = new PreparedRace(12345, new List<Dictionary<string, object?>>(new[] { preparedRow }));
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();
            var trainer = new FakeTrainer(configuration, preparedRace);

            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 100m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Solar Flair",
                    ClothNumber = 1
                }
            };

            var upcoming = new UpcomingRace
            {
                MarketId = marketId,
                RaceDate = raceDate,
                ScheduledOff = new TimeSpan(15, 30, 0),
                VenueName = "Sandown",
                VenueCountry = "GB",
                Title = "Summer Trophy",
                RaceDetails = "Handicap (Class 3)",
                RunnerCount = (byte)flows.Count,
                BackBookPercentage = 101.2m,
                LayBookPercentage = 103.4m
            };

            var upcomingId = repo.UpsertUpcomingRace(upcoming);
            Assert.True(upcomingId > 0);
            upcoming.UpcomingRaceId = upcomingId;

            var features = scraper.TestLoadFeatures(
                raceDate,
                upcoming.Title,
                upcoming.VenueName,
                upcoming.VenueCountry,
                upcoming.ScheduledOff,
                upcoming.RaceDetails,
                upcoming.Going,
                upcoming.BackBookPercentage,
                upcoming.LayBookPercentage,
                upcoming.MarketId,
                flows,
                preparedRows: null,
                persistedUpcoming: upcoming);

            Assert.True(repo.FindUpcomingRaceCalled);
            Assert.NotNull(features);
            Assert.Equal("Solar Flair", features!["HorseName"]);
            Assert.Equal("Jane Trainer", features["TrainerName"]);
            Assert.Equal(128, features["TrainerRunsLast365"]);
            Assert.Equal(9, features["HorseRunsLast365"]);
        }
        [Fact]
        public void PopulateFeatureVectors_FallsBackToTrainerWhenLookupEmpty()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var fallbackRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Fallback Hero",
                ["Class"] = 3,
                ["Going"] = "Soft",
                ["HasLastWin"] = true,
                ["DistanceChangeFromLast"] = 4f,
                ["DistanceRatioFromAverage"] = 1.1f,
                ["CareerStarts"] = 8,
                ["LifetimeWinRate"] = 0.25f,
                ["IsTopWeight"] = true,
                ["IsBottomWeight"] = false,
                ["JockeyGoingDistanceWinRate"] = 0.4f,
                ["JockeyGoingDistanceAvgNorm"] = 0.35f,
                ["LastJockeyGoingDistanceNormPos"] = 0.2f,
                ["TrainerJockeyCourseWinRate"] = 0.5f
            };
            var preparedRace = new PreparedRace(777, new List<Dictionary<string, object?>> { fallbackRow });

            var trainer = new FallbackTrainer(configuration, preparedRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 20m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Fallback Hero"
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.555",
                raceTitle: "Fallback Stakes",
                venueName: "Fallback Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 9, 14),
                offTime: new TimeSpan(15, 15, 0),
                raceDetails: "Handicap",
                going: "Good",
                backBookPercentage: 101m,
                layBookPercentage: 103m,
                raceUrl: null,
                flows: flows);

            Assert.True(trainer.FallbackCalled);

            var runner = Assert.Single(report.Runners);
            Assert.True(Assert.IsType<bool>(runner.FeatureValues["HasLastWin"]));
            Assert.Equal(4f, Convert.ToSingle(runner.FeatureValues["DistanceChangeFromLast"]));
            Assert.Equal(1.1f, Convert.ToSingle(runner.FeatureValues["DistanceRatioFromAverage"]));
            Assert.Equal(8, Convert.ToInt32(runner.FeatureValues["CareerStarts"]));
            Assert.Equal(3, Convert.ToInt32(runner.FeatureValues["Class"]));
            Assert.Equal("Soft", Assert.IsType<string>(runner.FeatureValues["Going"]));
            Assert.Equal(0.25f, Convert.ToSingle(runner.FeatureValues["LifetimeWinRate"]));
            Assert.Equal(true, runner.FeatureValues["IsTopWeight"]);
            Assert.Equal(false, runner.FeatureValues["IsBottomWeight"]);
            Assert.Equal(0.4f, Convert.ToSingle(runner.FeatureValues["JockeyGoingDistanceWinRate"]));
            Assert.Equal(0.35f, Convert.ToSingle(runner.FeatureValues["JockeyGoingDistanceAvgNorm"]));
            Assert.Equal(0.2f, Convert.ToSingle(runner.FeatureValues["LastJockeyGoingDistanceNormPos"]));
            Assert.Equal(0.5f, Convert.ToSingle(runner.FeatureValues["TrainerJockeyCourseWinRate"]));
        }

        [Fact]
        public void PopulateFeatureVectors_ComputesDistanceChangeFromRepositoryFallback()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var preparedRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Repository Hero",
                ["DistanceYards"] = 2200
            };

            var preparedRace = new PreparedRace(888, new List<Dictionary<string, object?>> { preparedRow });

            var trainer = new FallbackTrainer(configuration, preparedRace);
            var repo = new MinimalRacingRepository
            {
                LastDistanceResult = 2100
            };
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 20m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Repository Hero"
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.777",
                raceTitle: "Repository Stakes",
                venueName: "Repository Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 9, 14),
                offTime: new TimeSpan(15, 15, 0),
                raceDetails: "Handicap",
                going: "Good",
                backBookPercentage: 101m,
                layBookPercentage: 103m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.True(repo.LastDistanceLookupCalled);
            Assert.Equal(100f, Convert.ToSingle(runner.FeatureValues["DistanceChangeFromLast"]));
        }
        [Fact]
        public void PopulateFeatureVectors_BackfillsHistoricalFeaturesFromTrainer()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var primaryRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Backfill Hero",
            };
            var primaryRace = new PreparedRace(888, new List<Dictionary<string, object?>> { primaryRow });

            var fallbackRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Backfill Hero",
                ["Class"] = 4,
                ["Going"] = "Good",
                ["HasLastWin"] = true,
                ["DistanceChangeFromLast"] = -2f,
                ["DistanceRatioFromAverage"] = 0.95f,
                ["CareerStarts"] = 14,
                ["LifetimeWinRate"] = 0.285f,
                ["IsTopWeight"] = false,
                ["IsBottomWeight"] = true,
                ["JockeyGoingDistanceWinRate"] = 0.42f,
                ["JockeyGoingDistanceAvgNorm"] = 0.31f,
                ["LastJockeyGoingDistanceNormPos"] = 0.18f,
                ["TrainerJockeyCourseWinRate"] = 0.37f,
                ["DistanceBeatenLengths"] = 1.5f,
                ["DistanceBeatenKnown"] = true,
                ["WinningTimeMs"] = 95432,
                ["ClassWinRate"] = 0.55f,
                ["ClassAvgNorm"] = 0.27f,
                ["LastClassNormPos"] = 0.19f,
                ["TrainerClassWinRate"] = 0.48f,
                ["TrainerClassAvgNorm"] = 0.22f,
                ["LastTrainerClassNormPos"] = 0.31f,
                ["JockeyClassWinRate"] = 0.33f,
                ["JockeyClassAvgNorm"] = 0.21f,
                ["LastJockeyClassNormPos"] = 0.17f,
                ["GoingCourseWinRate"] = 0.41f,
                ["GoingCourseAvgNorm"] = 0.36f,
                ["LastGoingCourseNormPos"] = 0.29f,
                ["DistanceBucketWinRate"] = 0.45f,
                ["LastDistanceBucketNormPos"] = 0.24f,
                ["GoingDistanceWinRate"] = 0.34f,
                ["GoingDistanceAvgNorm"] = 0.28f,
                ["LastGoingDistanceNormPos"] = 0.23f,
                ["RaceSpeed"] = 0.92f,
                ["RunnerSpeed"] = 0.89f,
                ["SpeedMissing"] = false,
                ["SpeedDiff"] = -0.03f,
                ["SpeedRatio"] = 0.97f,
                ["CourseWinRateLast5"] = 0.52f
            
        };
            var fallbackRace = new PreparedRace(889, new List<Dictionary<string, object?>> { fallbackRow });

            var trainer = new BackfillTrainer(configuration, primaryRace, fallbackRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 15m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Backfill Hero",
                    ClothNumber = 3
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.777",
                raceTitle: "Backfill Stakes",
                venueName: "Backfill Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 9, 15),
                offTime: new TimeSpan(14, 45, 0),
                raceDetails: "Handicap",
                going: "Good to Firm",
                backBookPercentage: 102m,
                layBookPercentage: 105m,
                raceUrl: null,
                flows: flows);

            Assert.True(trainer.BackfillCalled);

            var runner = Assert.Single(report.Runners);
            Assert.True(Convert.ToBoolean(runner.FeatureValues["HasLastWin"]));
            Assert.Equal(-2f, Convert.ToSingle(runner.FeatureValues["DistanceChangeFromLast"]));
            Assert.Equal(0.95f, Convert.ToSingle(runner.FeatureValues["DistanceRatioFromAverage"]));
            Assert.Equal(14, Convert.ToInt32(runner.FeatureValues["CareerStarts"]));
            Assert.Equal(0.285f, Convert.ToSingle(runner.FeatureValues["LifetimeWinRate"]));
            Assert.Equal(4, Convert.ToInt32(runner.FeatureValues["Class"]));
            Assert.Equal("Good", Assert.IsType<string>(runner.FeatureValues["Going"]));
            Assert.Equal(false, Convert.ToBoolean(runner.FeatureValues["IsTopWeight"]));
            Assert.Equal(true, Convert.ToBoolean(runner.FeatureValues["IsBottomWeight"]));
            Assert.Equal(0.42f, Convert.ToSingle(runner.FeatureValues["JockeyGoingDistanceWinRate"]));
            Assert.Equal(0.31f, Convert.ToSingle(runner.FeatureValues["JockeyGoingDistanceAvgNorm"]));
            Assert.Equal(0.18f, Convert.ToSingle(runner.FeatureValues["LastJockeyGoingDistanceNormPos"]));
            Assert.Equal(0.37f, Convert.ToSingle(runner.FeatureValues["TrainerJockeyCourseWinRate"]));
            Assert.Equal(1.5f, Convert.ToSingle(runner.FeatureValues["DistanceBeatenLengths"]));
            Assert.True(Convert.ToBoolean(runner.FeatureValues["DistanceBeatenKnown"]));
            Assert.Equal(95432, Convert.ToInt32(runner.FeatureValues["WinningTimeMs"]));
            Assert.Equal(0.55f, Convert.ToSingle(runner.FeatureValues["ClassWinRate"]));
            Assert.Equal(0.27f, Convert.ToSingle(runner.FeatureValues["ClassAvgNorm"]));
            Assert.Equal(0.19f, Convert.ToSingle(runner.FeatureValues["LastClassNormPos"]));
            Assert.Equal(0.48f, Convert.ToSingle(runner.FeatureValues["TrainerClassWinRate"]));
            Assert.Equal(0.22f, Convert.ToSingle(runner.FeatureValues["TrainerClassAvgNorm"]));
            Assert.Equal(0.31f, Convert.ToSingle(runner.FeatureValues["LastTrainerClassNormPos"]));
            Assert.Equal(0.33f, Convert.ToSingle(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0.21f, Convert.ToSingle(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.17f, Convert.ToSingle(runner.FeatureValues["LastJockeyClassNormPos"]));
            Assert.Equal(0.41f, Convert.ToSingle(runner.FeatureValues["GoingCourseWinRate"]));
            Assert.Equal(0.36f, Convert.ToSingle(runner.FeatureValues["GoingCourseAvgNorm"]));
            Assert.Equal(0.26f, Convert.ToSingle(runner.FeatureValues["LastAgeRestrictionNormPos"]));
            Assert.Equal(0.45f, Convert.ToSingle(runner.FeatureValues["DistanceBucketWinRate"]));
            Assert.Equal(0.24f, Convert.ToSingle(runner.FeatureValues["LastDistanceBucketNormPos"]));
            Assert.Equal(0.34f, Convert.ToSingle(runner.FeatureValues["GoingDistanceWinRate"]));
            Assert.Equal(0.28f, Convert.ToSingle(runner.FeatureValues["GoingDistanceAvgNorm"]));
            Assert.Equal(0.23f, Convert.ToSingle(runner.FeatureValues["LastGoingDistanceNormPos"]));
            Assert.Equal(0.92f, Convert.ToSingle(runner.FeatureValues["RaceSpeed"]));
            Assert.Equal(0.89f, Convert.ToSingle(runner.FeatureValues["RunnerSpeed"]));
            Assert.False(Convert.ToBoolean(runner.FeatureValues["SpeedMissing"]));
            Assert.Equal(-0.03f, Convert.ToSingle(runner.FeatureValues["SpeedDiff"]));
            Assert.Equal(0.97f, Convert.ToSingle(runner.FeatureValues["SpeedRatio"]));
            Assert.Equal(0.52f, Convert.ToSingle(runner.FeatureValues["CourseWinRateLast5"]));
        }
        [Fact]
        public void PopulateFeatureVectors_AllowsMissingSpeedFeaturesWhenRawDataUnavailable()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var primaryRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Speed Optional"
            };
            var primaryRace = new PreparedRace(910, new List<Dictionary<string, object?>> { primaryRow });

            var fallbackRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Speed Optional",
                ["Class"] = 4,
                ["Going"] = "Soft",
                ["HasLastWin"] = false,
                ["DistanceChangeFromLast"] = 1.5f,
                ["DistanceRatioFromAverage"] = 1.02f,
                ["CareerStarts"] = 11,
                ["LifetimeWinRate"] = 0.182f,
                ["IsTopWeight"] = false,
                ["IsBottomWeight"] = false,
                ["JockeyGoingDistanceWinRate"] = 0.33f,
                ["JockeyGoingDistanceAvgNorm"] = 0.27f,
                ["LastJockeyGoingDistanceNormPos"] = 0.21f,
                ["TrainerJockeyCourseWinRate"] = 0.3f,
                ["DistanceBeatenKnown"] = false,
                ["DistanceBeatenLengths"] = 0f,
                ["WinningTimeMs"] = 0f,
                ["ClassWinRate"] = 0.41f,
                ["ClassAvgNorm"] = 0.23f,
                ["LastClassNormPos"] = 0.15f,
                ["TrainerClassWinRate"] = 0.38f,
                ["TrainerClassAvgNorm"] = 0.19f,
                ["LastTrainerClassNormPos"] = 0.24f,
                ["JockeyClassWinRate"] = 0.29f,
                ["JockeyClassAvgNorm"] = 0.2f,
                ["LastJockeyClassNormPos"] = 0.16f,
                ["GoingCourseWinRate"] = 0.32f,
                ["GoingCourseAvgNorm"] = 0.28f,
                ["LastGoingCourseNormPos"] = 0.22f,
                ["DistanceBucketWinRate"] = 0.35f,
                ["LastDistanceBucketNormPos"] = 0.2f,
                ["GoingDistanceWinRate"] = 0.31f,
                ["GoingDistanceAvgNorm"] = 0.25f,
                ["LastGoingDistanceNormPos"] = 0.2f,
                ["RaceSpeed"] = 0f,
                ["RunnerSpeed"] = 0f,
                ["SpeedMissing"] = true,
                ["SpeedDiff"] = 0f,
                ["SpeedRatio"] = 0f,
                ["CourseWinRateLast5"] = 0.44f,
                ["TrainerWinRate"] = 0.21f,
                ["TrainerWinRateLast50"] = 0.24f,
                ["TrainerWinRateRecentDays"] = 0.2f,
                ["TrainerSurfaceWinRate"] = 0.27f,
                ["TrainerSurfaceAvgNorm"] = 0.18f,
                ["LastTrainerSurfaceNormPos"] = 0.14f,
                ["TrainerGoingWinRate"] = 0.25f,
                ["TrainerGoingAvgNorm"] = 0.2f,
                ["LastTrainerGoingNormPos"] = 0.18f,
                ["TrainerDistanceBucketWinRate"] = 0.28f,
                ["TrainerDistanceBucketAvgNorm"] = 0.19f,
                ["LastTrainerDistanceBucketNormPos"] = 0.17f,
                ["TrainerCourseWinRate"] = 0.23f,
                ["TrainerJockeyWinRate"] = 0.27f,
                ["TrainerJockeySurfaceWinRate"] = 0.25f,
                ["TrainerJockeyCourseWinRate"] = 0.24f,
                ["JockeyWinRate"] = 0.18f,
                ["JockeyWinRateLast50"] = 0.2f,
                ["JockeyWinRateRecentDays"] = 0.17f,
                ["JockeySurfaceWinRate"] = 0.22f,
                ["JockeySurfaceAvgNorm"] = 0.16f,
                ["LastJockeySurfaceNormPos"] = 0.13f,
                ["JockeyGoingWinRate"] = 0.2f,
                ["JockeyGoingAvgNorm"] = 0.17f,
                ["LastJockeyGoingNormPos"] = 0.15f,
                ["JockeyDistanceBucketWinRate"] = 0.24f,
                ["JockeyDistanceBucketAvgNorm"] = 0.18f,
                ["LastJockeyDistanceBucketNormPos"] = 0.16f,
                ["JockeyCourseWinRate"] = 0.19f,
                ["JockeyClassWinRate"] = 0.23f,
                ["JockeyClassAvgNorm"] = 0.18f,
                ["LastJockeyClassNormPos"] = 0.16f,
                ["JockeyGoingDistanceWinRate"] = 0.21f,
                ["JockeyGoingDistanceAvgNorm"] = 0.18f,
                ["LastJockeyGoingDistanceNormPos"] = 0.16f,
                ["RaceAvgWinRateLast5"] = 0.27f,
                ["GoingWinRateLast5"] = 0.29f,
                ["GoingAvgNormLast5"] = 0.2f,
                ["SurfaceWinRateLast5"] = 0.3f,
                ["SurfaceAvgNormLast5"] = 0.22f,
                ["CourseWinRateLast5"] = 0.28f,
                ["CourseAvgNormLast5"] = 0.2f,
                ["DistanceBucketWinRateLast5"] = 0.26f,
                ["DistanceBucketAvgNormLast5"] = 0.19f,
                ["WinRateLast5"] = 0.24f,
                ["AvgNormPosLast5"] = 0.32f,
                ["AvgSpeedLast5"] = 0f,
                ["AvgSpeedDiffLast5"] = 0f,
                ["AvgRatingLast5"] = 0.25f
            };
            var fallbackRace = new PreparedRace(911, new List<Dictionary<string, object?>> { fallbackRow });

            var trainer = new BackfillTrainer(configuration, primaryRace, fallbackRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 12m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Speed Optional",
                    ClothNumber = 4
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.888",
                raceTitle: "Speedless Stakes",
                venueName: "Optional Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 10, 3),
                offTime: new TimeSpan(15, 20, 0),
                raceDetails: "Handicap",
                going: "Soft",
                backBookPercentage: 103m,
                layBookPercentage: 105m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.True(runner.HasPreparedFeatures);
            Assert.True(Convert.ToBoolean(runner.FeatureValues["SpeedMissing"]));
            Assert.DoesNotContain("WinningTimeMs", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("RaceSpeed", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("RunnerSpeed", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("SpeedDiff", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("SpeedRatio", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("DistanceBeatenLengths", runner.FeaturePopulation.MissingKeys);
        }

        [Fact]
        public void PopulateFeatureVectors_RetainsRaceSpeedWhenRunnerSpeedUnavailable()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var primaryRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Race Speed Only"
            };
            var primaryRace = new PreparedRace(915, new List<Dictionary<string, object?>> { primaryRow });

            var fallbackRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Race Speed Only",
                ["Class"] = 3,
                ["Going"] = "Good",
                ["DistanceYards"] = 1760,
                ["DistanceBeatenKnown"] = false,
                ["DistanceBeatenLengths"] = 0f,
                ["WinningTimeMs"] = 95600,
                ["RaceSpeed"] = 1760f / 95600f,
                ["RunnerSpeed"] = 0f,
                ["SpeedMissing"] = true
            };
            var fallbackRace = new PreparedRace(916, new List<Dictionary<string, object?>> { fallbackRow });

            var trainer = new BackfillTrainer(configuration, primaryRace, fallbackRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 9m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Race Speed Only",
                    ClothNumber = 2
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.889",
                raceTitle: "Partial Speed Stakes",
                venueName: "Speed Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 11, 4),
                offTime: new TimeSpan(13, 55, 0),
                raceDetails: "Handicap",
                going: "Good",
                backBookPercentage: 101m,
                layBookPercentage: 103m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.True(Convert.ToBoolean(runner.FeatureValues["SpeedMissing"]));
            Assert.DoesNotContain("WinningTimeMs", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("RaceSpeed", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("RunnerSpeed", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("SpeedDiff", runner.FeaturePopulation.MissingKeys);
            Assert.DoesNotContain("SpeedRatio", runner.FeaturePopulation.MissingKeys);
        }
        [Fact]
        public void PopulateFeatureVectors_AllowsPartialHistoricalCoverage()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var preparedRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Partial Coverage",
                ["CareerStarts"] = 8,
                ["WinRateLast5"] = 0.2f
            };
            var preparedRace = new PreparedRace(917, new List<Dictionary<string, object?>> { preparedRow });

            var trainer = new FakeTrainer(configuration, preparedRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 15m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Partial Coverage",
                    ClothNumber = 3,
                    BackPrice1 = 4.5m
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.901",
                raceTitle: "Partial Feature Stakes",
                venueName: "Partial Downs",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 12, 1),
                offTime: new TimeSpan(14, 0, 0),
                raceDetails: "Handicap",
                going: "Good",
                backBookPercentage: 102m,
                layBookPercentage: 104m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.True(runner.HasPreparedFeatures);
            Assert.NotNull(runner.AiOdds);
            Assert.DoesNotContain(
                "Missing historical features",
                runner.AiProbabilityFallbackReason ?? string.Empty,
                StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Class", runner.FeaturePopulation.MissingKeys, StringComparer.OrdinalIgnoreCase);
        }
        [Fact]
        public void PopulateFeatureVectors_FallbackPreservesTrainerStatistics()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var fallbackRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Neutral Override",
                ["TrainerName"] = "Sample Trainer",
                ["TrainerId"] = 77,
                ["TrainerWinRate"] = 0.37f,
                ["TrainerSurfaceWinRate"] = 0.41f,
                ["TrainerGoingWinRate"] = 0.35f,
                ["TrainerDistanceBucketWinRate"] = 0.33f
            };
            var fallbackRace = new PreparedRace(901, new List<Dictionary<string, object?>> { fallbackRow });

            var trainer = new NeutralOverrideTrainer(configuration, fallbackRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 10m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Neutral Override",
                    TrainerName = "Sample Trainer"
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.555",
                raceTitle: "Neutral Stakes",
                venueName: "Neutral Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 7, 10),
                offTime: new TimeSpan(12, 30, 0),
                raceDetails: "Handicap",
                going: "Good",
                backBookPercentage: 102m,
                layBookPercentage: 105m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(77, Convert.ToInt32(runner.FeatureValues["TrainerId"]));
            Assert.Equal(0.37f, Convert.ToSingle(runner.FeatureValues["TrainerWinRate"]));
            Assert.Equal(0.41f, Convert.ToSingle(runner.FeatureValues["TrainerSurfaceWinRate"]));
            Assert.Equal(0.35f, Convert.ToSingle(runner.FeatureValues["TrainerGoingWinRate"]));
            Assert.Equal(0.33f, Convert.ToSingle(runner.FeatureValues["TrainerDistanceBucketWinRate"]));
        }
        [Fact]
        public void PopulateFeatureVectors_ComputesRaceAverageWinRateLast5()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var row1 = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Alpha Runner",
                ["WinRateLast5"] = 0.4f
            };

            var row2 = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Beta Runner",
                ["WinRateLast5"] = 0.6f
            };

            var preparedRace = new PreparedRace(999, new List<Dictionary<string, object?>> { row1, row2 });

            var trainer = new FallbackTrainer(configuration, preparedRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 20m, settings);

            var flows = new List<RunnerFlow>
            {
new RunnerFlow { HorseName = "Alpha Runner" },
                new RunnerFlow { HorseName = "Beta Runner" }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.777",
                raceTitle: "Win Rate Stakes",
                venueName: "Sample Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 9, 21),
                offTime: new TimeSpan(14, 30, 0),
                raceDetails: "Handicap",
                going: "Good",
                backBookPercentage: 102m,
                layBookPercentage: 104m,
                raceUrl: null,
                flows: flows);

            Assert.True(trainer.FallbackCalled);

            var alpha = report.Runners.First(r => string.Equals(r.HorseName, "Alpha Runner", StringComparison.OrdinalIgnoreCase));
            var beta = report.Runners.First(r => string.Equals(r.HorseName, "Beta Runner", StringComparison.OrdinalIgnoreCase));

            Assert.Equal(0.4f, Convert.ToSingle(alpha.FeatureValues["WinRateLast5"]));
            Assert.Equal(0.6f, Convert.ToSingle(beta.FeatureValues["WinRateLast5"]));

            Assert.Equal(0.5f, Convert.ToSingle(alpha.FeatureValues["RaceAvgWinRateLast5"]));
            Assert.Equal(0.5f, Convert.ToSingle(beta.FeatureValues["RaceAvgWinRateLast5"]));
        }
        [Fact]
        public void AverageSpeedWindow_UsesOlderHistoryWhenRecentMissing()
        {
            var history = new List<(bool HasSpeed, float Speed, float SpeedDiff)>
            {
                (true, 1f, 0.1f),
                (true, 2f, 0.2f),
                (true, 3f, 0.3f),
                (true, 4f, 0.4f),
                (false, 0f, 0f),
                (true, 6f, 0.6f)
            };

            var (avgSpeed, avgDiff) = HyperparameterTrainer.TestComputeAverageSpeedForWindow(history, window: 5);

            Assert.Equal(3.2f, avgSpeed, 3);
            Assert.Equal(0.32f, avgDiff, 3);
        }
        [Fact]
        public void PopulateFeatureVectors_BackfillsRaceMetadataFromTrainer()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                })
                .Build();

            var primaryRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Metadata Marvel",
            };
            var primaryRace = new PreparedRace(890, new List<Dictionary<string, object?>> { primaryRow });

            var fallbackRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Metadata Marvel",
                ["Class"] = (byte)4,
                ["RaceType"] = "Handicap",
                ["Surface"] = "Turf",
                ["Going"] = "Good to Firm",
                ["DistanceYards"] = 1540,
                ["DistanceText"] = "7f",
                ["DistanceBucket"] = "Sprint",
                ["BackBookPercentage"] = 102.5m,
                ["LayBookPercentage"] = 104.2m,
                ["RunnerCount"] = (byte)12
            };
            var fallbackRace = new PreparedRace(891, new List<Dictionary<string, object?>> { fallbackRow });

            var trainer = new BackfillTrainer(configuration, primaryRace, fallbackRace);
            var repo = new MinimalRacingRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 25m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Metadata Marvel",
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.999",
                raceTitle: "Metadata Stakes",
                venueName: "Metadata Park",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 9, 20),
                offTime: new TimeSpan(15, 30, 0),
                raceDetails: null,
                going: null,
                backBookPercentage: null,
                layBookPercentage: null,
                raceUrl: null,
                flows: flows);

            Assert.True(trainer.BackfillCalled);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(4, Convert.ToInt32(runner.FeatureValues["Class"]));
            Assert.Equal("Handicap", Convert.ToString(runner.FeatureValues["RaceType"]));
            Assert.Equal("3yo+", Convert.ToString(runner.FeatureValues["AgeRestriction"]));
            Assert.Equal("Turf", Convert.ToString(runner.FeatureValues["Surface"]));
            Assert.Equal("Good to Firm", Convert.ToString(runner.FeatureValues["Going"]));
            Assert.Equal(1540, Convert.ToInt32(runner.FeatureValues["DistanceYards"]));
            Assert.Equal("7f", Convert.ToString(runner.FeatureValues["DistanceText"]));
            Assert.Equal("Sprint", Convert.ToString(runner.FeatureValues["DistanceBucket"]));
            Assert.Equal(102.5m, Convert.ToDecimal(runner.FeatureValues["BackBookPercentage"]));
            Assert.Equal(104.2m, Convert.ToDecimal(runner.FeatureValues["LayBookPercentage"]));
            Assert.Equal(12, Convert.ToInt32(runner.FeatureValues["RunnerCount"]));
        }

        private sealed class InMemoryRacingRepository : IRacingRepository
        {
            private readonly Dictionary<string, UpcomingRace> _upcomingByMarket = new(StringComparer.OrdinalIgnoreCase);
            private int _nextId = 1;

            public bool FindUpcomingRaceCalled { get; private set; }
            public void ClearDayReportTables()
            {
                _upcomingByMarket.Clear();
            }
            public UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName)
            {
                FindUpcomingRaceCalled = true;
                return _upcomingByMarket.Values.FirstOrDefault(r =>
                    r.RaceDate.Date == raceDate.Date &&
                    string.Equals(r.Title ?? string.Empty, raceTitle ?? string.Empty, StringComparison.OrdinalIgnoreCase) &&
                    string.Equals(r.VenueName ?? string.Empty, venueName ?? string.Empty, StringComparison.OrdinalIgnoreCase));
            }
            [Fact]
            public void PopulateFeatureVectors_SkipsScoringWhenTrainerDataMissing()
            {
                var configuration = new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                    })
                    .Build();

                var trainer = new FakeTrainer(configuration, upcomingRace: null);
                var repo = new MinimalRacingRepository();
                var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
                var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 25m, settings);
                var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Database Gap",
                    BackPrice1 = 4m
                }
            };

                var report = scraper.TestBuildRaceReport(
                    marketId: "1.1001",
                    raceTitle: "Gap Stakes",
                    venueName: "Data Park",
                    venueCountry: "GB",
                    raceDate: new DateTime(2024, 10, 5),
                    offTime: new TimeSpan(13, 15, 0),
                    raceDetails: "Handicap",
                    going: "Good",
                    backBookPercentage: 101m,
                    layBookPercentage: 103m,
                    raceUrl: null,
                    flows: flows);

                var runner = Assert.Single(report.Runners);
                Assert.False(runner.HasPreparedFeatures);
                Assert.NotNull(runner.AiProbabilityFallbackReason);
                Assert.Contains("Missing historical features; database coverage required.", runner.AiProbabilityFallbackReason);
                Assert.True(runner.AiProbabilityMarketDerived);
            }
            [Fact]
            public void LoadFeatureLookup_CachesPreparedRaceByMetadata()
            {
                var raceDate = new DateTime(2024, 9, 10);
                var configuration = new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                    })
                    .Build();

                var trainer = new CountingTrainer(configuration);
                var repo = new InMemoryRacingRepository();
                var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
                var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 25m, settings);
                var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Cached Runner",
                    ClothNumber = 1,
                }
            };

                var upcoming = new UpcomingRace
                {
                    MarketId = "1.456789",
                    RaceDate = raceDate,
                    Title = "Cache Test Stakes",
                    VenueName = "CacheVille",
                    RunnerCount = (byte)flows.Count
                };

                repo.UpsertUpcomingRace(upcoming);

                var features1 = scraper.TestLoadFeatures(
                    raceDate,
                    upcoming.Title,
                    upcoming.VenueName,
                    venueCountry: null,
                    scheduledOff: null,
                    raceDetails: null,
                    going: null,
                    backBookPercentage: null,
                    layBookPercentage: null,
                    marketId: upcoming.MarketId,
                    flows: flows,
                    preparedRows: null,
                    persistedUpcoming: upcoming);

                var features2 = scraper.TestLoadFeatures(
                    raceDate,
                    upcoming.Title,
                    upcoming.VenueName,
                    venueCountry: null,
                    scheduledOff: null,
                    going: null,
                    raceDetails: null,
                    backBookPercentage: null,
                    layBookPercentage: null,
                    marketId: upcoming.MarketId,
                    flows: flows,
                    preparedRows: null,
                    persistedUpcoming: upcoming);

                Assert.NotNull(features1);
                Assert.NotNull(features2);
                Assert.Equal(1, trainer.PrepareCalls);
            }
            [Fact]
            public void ApplyMarketFallback_UsesMarketOddsForUnmatchedRunners()
            {
                var configuration = new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                    })
                    .Build();

                var trainer = new FakeTrainer(configuration, upcomingRace: null);
                var repo = new InMemoryRacingRepository();
                var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
                var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 10m, settings);

                var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Unknown Runner",
                    BackPrice1 = 5m,
                    MatchedDatabaseRecord = false
                },
                new RunnerFlow
                {
                    HorseName = "Known Runner",
                    BackPrice1 = 4m,
                    AiOdds = 0.3,
                    MatchedDatabaseRecord = true
                }
            };

                scraper.TestApplyMarketFallback(flows);

                var unknown = flows[0];
                Assert.True(unknown.AiProbabilityMarketDerived);
                Assert.True(unknown.AiOdds.HasValue);
                Assert.Equal(0.2, unknown.AiOdds!.Value, 12);

                var known = flows[1];
                Assert.False(known.AiProbabilityMarketDerived);
                Assert.Equal(0.3, known.AiOdds);
            }
            public int? GetHistoricalRaceCountByHorseName(string? horseName)
            {
                return null;
            }
            public HistoricalRaceCountPrefetchResult GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
            {
                if (horseNames == null)
                {
                    throw new ArgumentNullException(nameof(horseNames));
                }

                return HistoricalRaceCountPrefetchResult.Empty;
            }
            public HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName)
            {
                return null;
            }
            public int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }
            public UpcomingRace? GetUpcomingRaceByMarketId(string? marketId)
            {
                if (string.IsNullOrWhiteSpace(marketId))
                {
                    return null;
                }

                return _upcomingByMarket.TryGetValue(marketId, out var race) ? race : null;
            }

            public void InsertRaceScreen(RaceScreen screen)
            {
            }

            public void InsertRunnerFlow(RunnerFlow flow)
            {
            }
            public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
            {
                if (flows is null)
                {
                    throw new ArgumentNullException(nameof(flows));
                }
            }
            public int UpsertUpcomingRace(UpcomingRace race)
            {
                if (race is null)
                {
                    throw new ArgumentNullException(nameof(race));
                }

                if (string.IsNullOrWhiteSpace(race.MarketId))
                {
                    throw new ArgumentException("MarketId is required for upcoming races in the test repository.");
                }

                if (!_upcomingByMarket.TryGetValue(race.MarketId, out var existing))
                {
                    race.UpcomingRaceId = _nextId++;
                    _upcomingByMarket[race.MarketId] = Clone(race);
                }
                else
                {
                    race.UpcomingRaceId = existing.UpcomingRaceId;
                    _upcomingByMarket[race.MarketId] = Clone(race);
                }

                return race.UpcomingRaceId;
            }

            private static UpcomingRace Clone(UpcomingRace race)
            {
                return new UpcomingRace
                {
                    UpcomingRaceId = race.UpcomingRaceId,
                    MarketId = race.MarketId,
                    RaceDate = race.RaceDate,
                    ScheduledOff = race.ScheduledOff,
                    VenueName = race.VenueName,
                    VenueCountry = race.VenueCountry,
                    Title = race.Title,
                    RaceDetails = race.RaceDetails,
                    RaceType = race.RaceType,
                    Surface = race.Surface,
                    Going = race.Going,
                    DistanceYards = race.DistanceYards,
                    DistanceText = race.DistanceText,
                    RunnerCount = race.RunnerCount,
                    BackBookPercentage = race.BackBookPercentage,
                    LayBookPercentage = race.LayBookPercentage
                };
            }
        }

        private sealed class FakeTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            private readonly PreparedRace? _upcomingRace;
            private readonly PreparedDataset _dataset;
            public FakeTrainer(IConfiguration configuration, PreparedRace? upcomingRace)
                : base(configuration)
            {
                _upcomingRace = upcomingRace;
                var races = upcomingRace is null
                    ? new List<PreparedRace>()
                    : new List<PreparedRace> { upcomingRace };
                _dataset = new PreparedDataset(races);
            }

            public override PreparedDataset PrepareDataset(
                ISet<int>? includeRaceIds = null,
                ISet<int>? stateRaceWhitelist = null,
                bool includeIdentifiers = false)
            {
                return _dataset;
            }

            public override PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
            {
                if (upcoming is null)
                {
                    throw new ArgumentNullException(nameof(upcoming));
                }
                if (flows is null)
                {
                    throw new ArgumentNullException(nameof(flows));
                }

                return _upcomingRace;
            }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                {
                    throw new ArgumentNullException(nameof(requests));
                }
                if (requests.Count == 0)
                {
                    return Array.Empty<PreparedRace?>();
                }
                var results = new PreparedRace?[requests.Count];
                for (int i = 0; i < results.Length; i++)
                {
                    results[i] = _upcomingRace;
                }

                return results;
            }
        }
        private sealed class CountingTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            private readonly PreparedRace _preparedRace;

            public CountingTrainer(IConfiguration configuration)
                : base(configuration)
            {
                var row = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                {
                    ["HorseName"] = "Cached Runner",
                    ["SaddleclothNumber"] = 1,
                    ["CareerStarts"] = 3
                };
                _preparedRace = new PreparedRace(777, new List<Dictionary<string, object?>>(new[] { row }));
            }

            public int PrepareCalls { get; private set; }

            public override PreparedDataset PrepareDataset(ISet<int>? includeRaceIds = null, ISet<int>? stateRaceWhitelist = null, bool includeIdentifiers = false)
            {
                return new PreparedDataset(new List<PreparedRace>());
            }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                {
                    throw new ArgumentNullException(nameof(requests));
                }

                PrepareCalls++;
                var results = new PreparedRace?[requests.Count];
                for (int i = 0; i < results.Length; i++)
                {
                    results[i] = _preparedRace;
                }

                return results;
            }

            public override PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
            {
                if (upcoming is null)
                {
                    throw new ArgumentNullException(nameof(upcoming));
                }
                if (flows is null)
                {
                    throw new ArgumentNullException(nameof(flows));
                }

                return _preparedRace;
            }
        }
        private sealed class BatchedLookupTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            private readonly RunnerLookupData _lookupData;

            public BatchedLookupTrainer(IConfiguration configuration)
                : base(configuration)
            {
                var horseIds = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
                {
                    ["Known Runner"] = 42
                };
                var jockeyIds = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
                {
                    ["Known Jockey"] = 9
                };
                var snapshots = new Dictionary<int, RunnerSnapshot>
                {
                    [42] = new RunnerSnapshot
                    {
                        HorseId = 42,
                        TrainerId = 77,
                        TrainerName = "Sample Trainer",
                        Age = (short)5,
                        WeightLbs = (short)126,
                        WeightText = "9-0",
                        OfficialRating = (short)101
                    }
                };

                _lookupData = new RunnerLookupData(horseIds, jockeyIds, snapshots);
            }

            protected override RunnerLookupData LoadRunnerLookupData(
                SqlConnection conn,
                UpcomingRace upcoming,
                IReadOnlyCollection<string> runnerColumns,
                IReadOnlyCollection<string> horseNames,
                IReadOnlyCollection<string> jockeyNames,
                IReadOnlyCollection<int>? horseIdsFromFlows = null)
            {
                return _lookupData;
            }

            public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
            {
            }
            protected override (int CourseId, string? CourseName) ResolveCourse(SqlConnection conn, UpcomingRace upcoming)
            {
                return (555, upcoming.VenueName);
            }
        }
        private sealed class MinimalRacingRepository : IRacingRepository
        {
            public int? LastDistanceResult { get; set; }
            public bool LastDistanceLookupCalled { get; private set; }
            public void ClearDayReportTables()
            {
            }

            public UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName)
            {
                return null;
            }

            public int? GetHistoricalRaceCountByHorseName(string? horseName)
            {
                return null;
            }

            public HistoricalRaceCountPrefetchResult GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
            {
                return HistoricalRaceCountPrefetchResult.Empty;
            }
            public HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName)
            {
                return null;
            }
            public int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate)
            {
                LastDistanceLookupCalled = true;
                return LastDistanceResult;
            }

            public UpcomingRace? GetUpcomingRaceByMarketId(string? marketId)
            {
                return null;
            }

            public void InsertRaceScreen(RaceScreen screen)
            {
            }

            public void InsertRunnerFlow(RunnerFlow flow)
            {
            }

            public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
            {
            }

            public int UpsertUpcomingRace(UpcomingRace race)
            {
                return 0;
            }
        }

        private sealed class FallbackTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            private readonly PreparedRace _fallbackRace;

            public FallbackTrainer(IConfiguration configuration, PreparedRace fallbackRace)
                : base(configuration)
            {
                _fallbackRace = fallbackRace ?? throw new ArgumentNullException(nameof(fallbackRace));
            }

            public bool FallbackCalled { get; private set; }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                {
                    throw new ArgumentNullException(nameof(requests));
                }

                var results = new PreparedRace?[requests.Count];
                for (int i = 0; i < results.Length; i++)
                {
                    results[i] = null;
                }

                return results;
            }

            public override PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
            {
                FallbackCalled = true;
                return _fallbackRace;
            }
        }
        private sealed class BackfillTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            private readonly PreparedRace _primaryRace;
            private readonly PreparedRace _fallbackRace;

            public BackfillTrainer(IConfiguration configuration, PreparedRace primaryRace, PreparedRace fallbackRace)
                : base(configuration)
            {
                _primaryRace = primaryRace ?? throw new ArgumentNullException(nameof(primaryRace));
                _fallbackRace = fallbackRace ?? throw new ArgumentNullException(nameof(fallbackRace));
            }

            public bool BackfillCalled { get; private set; }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                {
                    throw new ArgumentNullException(nameof(requests));
                }

                var results = new PreparedRace?[requests.Count];
                for (int i = 0; i < results.Length; i++)
                {
                    results[i] = _primaryRace;
                }

                return results;
            }

            public override PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
            {
                BackfillCalled = true;
                return _fallbackRace;
            }
        }
        private sealed class NeutralOverrideTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            private readonly PreparedRace _fallbackRace;
            private int _callCount;

            public NeutralOverrideTrainer(IConfiguration configuration, PreparedRace fallbackRace)
                : base(configuration)
            {
                _fallbackRace = fallbackRace ?? throw new ArgumentNullException(nameof(fallbackRace));
            }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                {
                    throw new ArgumentNullException(nameof(requests));
                }

                _callCount++;
                var results = new PreparedRace?[requests.Count];
                for (int i = 0; i < results.Length; i++)
                {
                    results[i] = _callCount == 1 ? null : _fallbackRace;
                }

                return results;
            }
        }
    }
}

