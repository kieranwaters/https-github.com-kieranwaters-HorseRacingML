using System;
using System.Collections.Generic;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Microsoft.Extensions.Configuration;
using Xunit;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.Tests
{
    public class HistoricalRaceCountPrefetchTests
    {
        [Fact]
        public void BuildRaceReport_UsesPrefetchedHistoricalCounts_WhenAvailable()
        {
            var repo = new FakeHistoricalRepository
            {
                BulkResult = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
                {
                    ["Alpha Runner"] = 12,
                    ["Beta Runner"] = 8
                }
            };

            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 100m);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow { HorseName = "Alpha Runner" },
                new RunnerFlow { HorseName = "Beta Runner (IRE)" }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 6, 1),
                offTime: new TimeSpan(14, 30, 0),
                raceDetails: "Class 4",
                backBookPercentage: null,
                layBookPercentage: null,
                raceUrl: null,
                flows: flows);

            Assert.Equal(1, repo.BulkCallCount);
            Assert.Equal(0, repo.SingleCallCount);
            Assert.Collection(
                report.Runners,
                runner => Assert.Equal(12, runner.HistoricalRaceCount),
                runner => Assert.Equal(8, runner.HistoricalRaceCount));

            Assert.Equal(new[] { "Alpha Runner", "Beta Runner (IRE)" }, repo.LastBulkNames);
        }
        public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
        {
            if (flows is null)
            {
                throw new ArgumentNullException(nameof(flows));
            }
        }
        [Fact]
        public void BuildRaceReport_FallsBackToSingleLookup_WhenPrefetchMissing()
        {
            var repo = new FakeHistoricalRepository
            {
                BulkResult = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
                {
                    ["Known Runner"] = 3
                },
                SingleLookup = name =>
                {
                    if (string.Equals(name, "Fallback Runner", StringComparison.OrdinalIgnoreCase))
                    {
                        return 7;
                    }

                    return null;
                }
            };

            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 50m);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow { HorseName = "Known Runner" },
                new RunnerFlow { HorseName = "Fallback Runner" }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "2.468",
                raceTitle: "Fallback Race",
                venueName: "Fallback Venue",
                venueCountry: "IE",
                raceDate: new DateTime(2024, 6, 2),
                offTime: new TimeSpan(15, 45, 0),
                raceDetails: "Handicap",
                backBookPercentage: null,
                layBookPercentage: null,
                raceUrl: null,
                flows: flows);

            Assert.Equal(1, repo.BulkCallCount);
            Assert.Equal(1, repo.SingleCallCount);
            Assert.Collection(
                report.Runners,
                runner => Assert.Equal(3, runner.HistoricalRaceCount),
                runner => Assert.Equal(7, runner.HistoricalRaceCount));
        }

        private sealed class FakeHistoricalRepository : IRacingRepository
        {
            public Dictionary<string, int> BulkResult { get; init; } = new(StringComparer.OrdinalIgnoreCase);
            public Func<string?, int?>? SingleLookup { get; init; }
            public int BulkCallCount { get; private set; }
            public int SingleCallCount { get; private set; }
            public IReadOnlyList<string> LastBulkNames { get; private set; } = Array.Empty<string>();

            public void ClearDayReportTables()
            {
            }

            public UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName)
            {
                return null;
            }

            public int? GetHistoricalRaceCountByHorseName(string? horseName)
            {
                SingleCallCount++;
                return SingleLookup?.Invoke(horseName);
            }

            public IReadOnlyDictionary<string, int> GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
            {
                BulkCallCount++;
                LastBulkNames = new List<string>(horseNames);
                return new Dictionary<string, int>(BulkResult, StringComparer.OrdinalIgnoreCase);
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
                if (flows is null)
                {
                    throw new ArgumentNullException(nameof(flows));
                }
            }
            public int UpsertUpcomingRace(UpcomingRace race)
            {
                return 0;
            }
        }

        private sealed class StubTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            public StubTrainer()
                : base(new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                    })
                    .Build())
            {
            }

            public override TrainingDataset.PreparedDataset PrepareDataset(
                ISet<int>? includeRaceIds = null,
                ISet<int>? stateRaceWhitelist = null,
                bool includeIdentifiers = false)
            {
                return new TrainingDataset.PreparedDataset(new List<TrainingDataset.PreparedDataset.PreparedRace>());
            }
            public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
            {
            }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                {
                    throw new ArgumentNullException(nameof(requests));
                }

                return new PreparedRace?[requests.Count];
            }
        }
    }
}