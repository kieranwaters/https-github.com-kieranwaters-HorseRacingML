using System;
using System.Collections.Generic;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Xunit;
using Microsoft.Extensions.Configuration;
using System.Linq;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;
using Microsoft.Data.SqlClient;
using HorseRacingML.Services;

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
        [Fact]
        public void BuildUpcomingRaceRows_UsesBatchedLookupsWhenAvailable()
        {
            var configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
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
                Class = (byte)2,
                AgeRestriction = "3yo+",
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
                    Draw = 3,
                    SelectionId = "101"
                },
                new RunnerFlow
                {
                    HorseName = "Unknown Runner",
                    JockeyName = "Unknown Jockey",
                    ClothNumber = 2,
                    Draw = 7,
                    SelectionId = "202"
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
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
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
                    ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
                })
                .Build();

            var fallbackRow = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseName"] = "Fallback Hero",
                ["SelectionId"] = "321",
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
                    HorseName = "Fallback Hero",
                    SelectionId = "321"
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
            Assert.Equal(0.25f, Convert.ToSingle(runner.FeatureValues["LifetimeWinRate"]));
            Assert.Equal(true, runner.FeatureValues["IsTopWeight"]);
            Assert.Equal(false, runner.FeatureValues["IsBottomWeight"]);
            Assert.Equal(0.4f, Convert.ToSingle(runner.FeatureValues["JockeyGoingDistanceWinRate"]));
            Assert.Equal(0.35f, Convert.ToSingle(runner.FeatureValues["JockeyGoingDistanceAvgNorm"]));
            Assert.Equal(0.2f, Convert.ToSingle(runner.FeatureValues["LastJockeyGoingDistanceNormPos"]));
            Assert.Equal(0.5f, Convert.ToSingle(runner.FeatureValues["TrainerJockeyCourseWinRate"]));
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
            public void LoadFeatureLookup_CachesPreparedRaceByMetadata()
            {
                var raceDate = new DateTime(2024, 9, 10);
                var configuration = new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;"
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
                    SelectionId = "111"
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
            public int? GetHistoricalRaceCountByHorseName(string? horseName)
            {
                return null;
            }
            public IReadOnlyDictionary<string, int> GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
            {
                if (horseNames == null)
                {
                    throw new ArgumentNullException(nameof(horseNames));
                }

                return new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
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
                    Class = race.Class,
                    AgeRestriction = race.AgeRestriction,
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

            public IReadOnlyDictionary<string, int> GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
            {
                return new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
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
    }
}