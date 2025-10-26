using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using Xunit;

namespace HorseRacingML.Tests
{
    public class DayReportFeatureDisplayTests
    {
        [Fact]
        public void CreateRunnerReport_PopulatesJockeyClassFallbacksForDisplay()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["JockeyWinRate"] = 0.33f,
                        ["JockeyGoingAvgNorm"] = 0.21f,
                        ["LastJockeyGoingNormPos"] = 0.17f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.True(runner.FeatureValues.TryGetValue("JockeyClassWinRate", out var winRateObj));
            Assert.Equal(0.33f, Assert.IsType<float>(winRateObj));
            Assert.Equal(0.21f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.17f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }

        [Fact]
        public void CreateRunnerReport_PopulatesJockeyClassFallbacksWhenSourceMissing()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase),
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }
        [Fact]
        public void CreateRunnerReport_UsesGeneralClassMetricsWhenJockeyClassUnavailable()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["ClassWinRate"] = 0.44f,
                        ["ClassAvgNorm"] = 0.31f,
                        ["LastClassNormPos"] = 0.27f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0.44f, Assert.IsType<float>(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0.31f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.27f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }

        private sealed class StubRepository : IRacingRepository
        {
            public void ClearDayReportTables()
            {
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

            public UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName)
            {
                return null;
            }

            public UpcomingRace? GetUpcomingRaceByMarketId(string? marketId)
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

            public int GetHistoricalWinCountByHorseName(string? horseName)
            {
                return 0;
            }

            public HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName)
            {
                return null;
            }

            public int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }

            public IReadOnlyList<RunnerResult> GetLastSavedResults(string? raceTitle, DateTime? raceDate)
            {
                return Array.Empty<RunnerResult>();
            }

            public int? GetWinningTimeMilliseconds(int raceId)
            {
                return null;
            }

            public int? GetRaceIdByRunnerResult(int runnerResultId)
            {
                return null;
            }

            public decimal? GetDistanceBeatenLengths(int runnerResultId)
            {
                return null;
            }

            public decimal? GetLastDistanceBeatenLengths(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }

            public int? GetLastWinningTimeMilliseconds(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }

            public byte? GetMostRecentRaceClass(string? horseName, int? horseId)
            {
                return null;
            }

            public IReadOnlyList<RaceClassRating> GetHistoricalRaceClassRatings(string? horseName, int? horseId, int maxCount)
            {
                return Array.Empty<RaceClassRating>();
            }

            public (int Wins, int Starts)? GetRecentHorseWinStats(string? horseName, int? horseId, DateTime? beforeDate, int windowSize)
            {
                return null;
            }

            public IReadOnlyList<HorseSpeedEntry> GetRecentHorseSpeedEntries(string? horseName, int? horseId, DateTime? beforeDate, int windowSize)
            {
                return Array.Empty<HorseSpeedEntry>();
            }
        }

        private sealed class StubTrainer : HyperparameterTrainer
        {
            public StubTrainer()
                : base(new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                    })
                    .Build())
            {
            }

            public override TrainingDataset.PreparedDataset PrepareDataset(ISet<int>? includeRaceIds = null, ISet<int>? stateRaceWhitelist = null, bool includeIdentifiers = false)
            {
                return new TrainingDataset.PreparedDataset(new List<TrainingDataset.PreparedDataset.PreparedRace>());
            }

            public override IReadOnlyList<TrainingDataset.PreparedDataset.PreparedRace?> PrepareUpcomingRaces(IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                return Array.Empty<TrainingDataset.PreparedDataset.PreparedRace?>();
            }
        }
    }
}