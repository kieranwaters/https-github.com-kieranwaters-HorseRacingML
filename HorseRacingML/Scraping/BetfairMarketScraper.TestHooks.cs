using System;
using System.Collections.Generic;
using HorseRacingML.Models;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        internal Dictionary<string, object?>? TestLoadFeatures(
            DateTime? raceDate,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            TimeSpan? scheduledOff,
            string? raceDetails,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows,
            IReadOnlyList<IDictionary<string, object?>>? preparedRows,
            UpcomingRace? persistedUpcoming)
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
                backBookPercentage,
                layBookPercentage,
                marketId,
                flows,
                preparedRows,
                persistedUpcoming);

            return flows.Count > 0 ? lookup.FindByHorse(flows[0].HorseName) : null;
        }
    }
}