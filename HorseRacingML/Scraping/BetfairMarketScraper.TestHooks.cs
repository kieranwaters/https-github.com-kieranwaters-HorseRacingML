using System;
using System.Collections.Generic;
using HorseRacingML.Models;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        internal RaceDayReport TestBuildRaceReport(
            string marketId,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            DateTime? raceDate,
            TimeSpan? offTime,
            string? raceDetails,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? raceUrl,
            IEnumerable<RunnerFlow> flows)
        {
            return TestBuildRaceReport(
                marketId,
                raceTitle,
                venueName,
                venueCountry,
                raceDate,
                offTime,
                raceDetails,
                going,
                backBookPercentage,
                layBookPercentage,
                raceUrl,
                flows,
                raceType: null);
        }

        internal RaceDayReport TestBuildRaceReport(
            string marketId,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            DateTime? raceDate,
            TimeSpan? offTime,
            string? raceDetails,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? raceUrl,
            IEnumerable<RunnerFlow> flows,
            string? raceType)
        {
            if (flows == null)
            {
                throw new ArgumentNullException(nameof(flows));
            }

            return BuildRaceReport(
                marketId,
                raceTitle,
                venueName,
                venueCountry,
                raceDate,
                offTime,
                raceDetails,
                raceType,
                going,
                backBookPercentage,
                layBookPercentage,
                raceUrl,
                flows);
        }

        internal Dictionary<string, object?>? TestLoadFeatures(
            DateTime? raceDate,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            TimeSpan? scheduledOff,
            string? raceDetails,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows,
            IReadOnlyList<IDictionary<string, object?>>? preparedRows,
            UpcomingRace? persistedUpcoming)
        {
            return TestLoadFeatures(
                raceDate,
                raceTitle,
                venueName,
                venueCountry,
                scheduledOff,
                raceDetails,
                going,
                backBookPercentage,
                layBookPercentage,
                marketId,
                flows,
                preparedRows,
                persistedUpcoming,
                raceType: null);
        }

        internal Dictionary<string, object?>? TestLoadFeatures(
            DateTime? raceDate,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            TimeSpan? scheduledOff,
            string? raceDetails,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows,
            IReadOnlyList<IDictionary<string, object?>>? preparedRows,
            UpcomingRace? persistedUpcoming,
            string? raceType)
        {
            if (flows == null)
            {
                throw new ArgumentNullException(nameof(flows));
            }

            var lookup = LoadFeatureLookup(
                raceDate,
                raceTitle,
                venueName,
                venueCountry,
                scheduledOff,
                raceDetails,
                raceType,
                going,
                backBookPercentage,
                layBookPercentage,
                marketId,
                flows,
                preparedRows,
                persistedUpcoming);

            return flows.Count > 0 ? lookup.FindByHorse(flows[0].HorseName) : null;
        }
        internal void TestApplyMarketFallback(IReadOnlyList<RunnerFlow> flows)
        {
            ApplyMarketFallbackForUnmatchedRunners(flows);
        }
    }
}