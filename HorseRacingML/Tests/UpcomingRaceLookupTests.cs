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

            var scraper = new BetfairMarketScraper(repo, trainer, bankroll: 100m);
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

        private sealed class InMemoryRacingRepository : IRacingRepository
        {
            private readonly Dictionary<string, UpcomingRace> _upcomingByMarket = new(StringComparer.OrdinalIgnoreCase);
            private int _nextId = 1;

            public bool FindUpcomingRaceCalled { get; private set; }

            public UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName)
            {
                FindUpcomingRaceCalled = true;
                return _upcomingByMarket.Values.FirstOrDefault(r =>
                    r.RaceDate.Date == raceDate.Date &&
                    string.Equals(r.Title ?? string.Empty, raceTitle ?? string.Empty, StringComparison.OrdinalIgnoreCase) &&
                    string.Equals(r.VenueName ?? string.Empty, venueName ?? string.Empty, StringComparison.OrdinalIgnoreCase));
            }
            public int? GetHistoricalRaceCountByHorseName(string? horseName)
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
                _dataset = new PreparedDataset(new List<PreparedRace>());
            }

            public override PreparedDataset PrepareDataset(ISet<int>? includeRaceIds = null, ISet<int>? stateRaceWhitelist = null, bool includeIdentifiers = false)
            {
                return _dataset;
            }

            public override PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
            {
                return _upcomingRace;
            }
        }
    }
}