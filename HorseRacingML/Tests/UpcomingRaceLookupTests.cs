using System;
using System.Collections.Generic;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Xunit;

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
    }
}