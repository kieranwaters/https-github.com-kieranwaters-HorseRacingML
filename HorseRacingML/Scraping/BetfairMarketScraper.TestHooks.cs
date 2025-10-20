using System;
using System.Collections.Generic;
using HorseRacingML.Models;
using System.Linq;

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
                persistedUpcoming);

            if (flows.Count == 0)
            {
                return null;
            }

            var flow = flows[0];

            return lookup.FindByRunner(flow)
                ?? lookup.FindByHorse(flow.HorseName);
        }
        internal void TestApplyMarketFallback(IReadOnlyList<RunnerFlow> flows)
        {
            ApplyMarketFallbackForUnmatchedRunners(flows);
        }
        internal void TestApplyScrapedFeatureFallbacks(
            Dictionary<string, object?> featureVector,
            RunnerFlow flow,
            DateTime? raceDate = null,
            TimeSpan? scheduledOff = null,
            string? raceTitle = null,
            string? raceDetails = null,
            string? raceType = null,
            string? going = null,
            string? venueName = null,
            string? venueCountry = null,
            decimal? backBookPercentage = null,
            decimal? layBookPercentage = null,
            IReadOnlyList<RunnerFlow>? flows = null,
            ISet<string>? missingScrapeFields = null)
        {
            ApplyScrapedFeatureFallbacks(
                featureVector,
                flow,
                raceDate,
                scheduledOff,
                raceTitle,
                raceDetails,
                raceType,
                going,
                venueName,
                venueCountry,
                backBookPercentage,
                layBookPercentage,
                flows,
                missingScrapeFields);
        }
        internal FeaturePopulationSummary TestBuildFeaturePopulationSummary(Dictionary<string, object?> featureVector)
        {
            return BuildFeaturePopulationSummary(featureVector);
        }

        internal void TestSetNeuralFeatureKeys(IEnumerable<string>? keys)
        {
            UpdateNeuralFeatureKeys(keys?.ToList());
        }
    }
}