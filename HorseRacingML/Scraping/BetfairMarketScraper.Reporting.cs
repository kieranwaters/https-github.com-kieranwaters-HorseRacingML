using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        public void PopulateWinnerProbabilities(IEnumerable<RaceDayReport> races)
        {
            if (races == null)
            {
                return;
            }

            var raceList = races as IList<RaceDayReport> ?? races.ToList();
            if (raceList.Count == 0)
            {
                return;
            }
            Console.WriteLine($"[DayReport][Stage] Populating AI winner probabilities for {raceList.Count} race(s).");
            ResetUpcomingRaceLookupCache();
            var processedRaceCount = 0;
            foreach (var race in raceList)
            {
                if (race == null)
                {
                    continue;
                }

                if (race.Runners == null || race.Runners.Count == 0)
                {
                    Console.WriteLine("\t[DayReport] Skipping race with no runners available for probability population.");
                    continue;
                }

                var raceName = race.RaceTitle ?? race.MarketId ?? "unknown race";
                var raceDateStr = race.RaceDate.HasValue ? race.RaceDate.Value.ToString("yyyy-MM-dd") : "unknown date";
                Console.WriteLine($"\t[DayReport] Processing race {raceName} on {raceDateStr} with {race.Runners.Count} runner(s).");

                var upcomingRecord = TryResolveUpcomingRace(race);
                if (upcomingRecord != null)
                {
                    race.UpcomingRaceId = upcomingRecord.UpcomingRaceId;
                    Console.WriteLine($"        [DayReport] Using upcoming race {upcomingRecord.UpcomingRaceId} (market {upcomingRecord.MarketId}) for probability alignment.");
                }
                else
                {
                    int? recordedUpcomingId = TryRecordUpcomingRace(race);
                    if (recordedUpcomingId.HasValue)
                    {
                        race.UpcomingRaceId = recordedUpcomingId.Value;
                        Console.WriteLine($"        [DayReport] Recorded upcoming race {recordedUpcomingId.Value} for market {race.MarketId ?? "<unknown>"} to backfill features.");
                    }
                    else
                    {
                        var upcomingId = race.UpcomingRaceId?.ToString() ?? "n/a";
                        Console.WriteLine($"\t[DayReport] Race {raceName} is upcoming; recorded upcoming race id {upcomingId}. Using saved AI probabilities for reporting.");
                    }
                }
                var totalRunners = race.Runners.Count;
                var validRunners = new List<RunnerDayReport>();
                var suppressedProbabilityCount = 0;

                foreach (var runner in race.Runners)
                {
                    if (runner == null)
                    {
                        continue;
                    }

                    if (!runner.AiProbability.HasValue || !double.IsFinite(runner.AiProbability.Value) || runner.AiProbability.Value <= 0)
                    {
                        suppressedProbabilityCount++;
                        continue;
                    }

                    validRunners.Add(runner);
                }

                var retainedProbabilityCount = validRunners.Count;
                var backfilledMarketProbabilityCount = validRunners.Count(r =>
                    r.MarketProbability.HasValue &&
                    r.MarketDecimalOdds.HasValue &&
                    r.MarketDecimalOdds.Value > 1m);

                Console.WriteLine($"\t[DayReport] {retainedProbabilityCount}/{totalRunners} runner(s) retain valid AI probabilities.");
                if (suppressedProbabilityCount > 0)
                {
                    Console.WriteLine($"\t[DayReport] {retainedProbabilityCount}/{race.Runners.Count} runner(s) retain valid AI probabilities.");
                    if (suppressedProbabilityCount > 0)
                    {
                        Console.WriteLine($"\t[DayReport] Filtered {suppressedProbabilityCount} runner(s) due to missing or invalid AI probabilities.");
                    }

                    if (backfilledMarketProbabilityCount > 0)
                    {
                        Console.WriteLine($"\t[DayReport] Backfilled market probabilities for {backfilledMarketProbabilityCount} runner(s) from decimal odds.");
                    }
                    var winner = validRunners
                         .OrderByDescending(r => r.AiProbability!.Value)
                          .FirstOrDefault();

                    if (winner == null)
                    {
                        Console.WriteLine("\t[DayReport] No runners produced AI probabilities; skipping race output.");
                        continue;
                    }

                    var identifier = !string.IsNullOrWhiteSpace(winner.HorseName)
                        ? winner.HorseName
                        : "unknown";

                    var aiProb = winner.AiProbability!.Value;
                    var aiOddsStr = winner.AiDecimalOdds?.ToString("F3") ?? "n/a";
                    Console.WriteLine($"\t[DayReport] Top AI runner {identifier} => {aiProb:P4} (decimal odds {aiOddsStr}).");

                    if (winner.MarketProbability.HasValue && winner.Differential.HasValue)
                    {
                        Console.WriteLine($"\t[DayReport] Market probability {winner.MarketProbability.Value:P4}; differential {winner.Differential.Value:P4}.");
                    }
                    else
                    {
                        Console.WriteLine("\t[DayReport] Market probability unavailable; differential not computed.");
                    }

                    if (winner.KellyFraction.HasValue)
                    {
                        Console.WriteLine($"\t[DayReport] Kelly fraction {winner.KellyFraction.Value:P4}; suggested stake {(winner.SuggestedStake?.ToString("F2") ?? "n/a")} from bankroll {_bankroll:F2}.");
                    }
                    else
                    {
                        Console.WriteLine("\t[DayReport] Kelly fraction unavailable; stake not suggested.");
                    }
                }
                processedRaceCount++;
            }
            Console.WriteLine($"[DayReport][Stage] Winner probability population complete ({processedRaceCount} race(s) processed).");
        }

        private FeatureLookup LoadFeatureLookup(
            DateTime? raceDate,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            TimeSpan? scheduledOff,
            string? raceDetails,
            string? raceType,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows,
            UpcomingRace? persistedUpcoming)
        {
            if (!raceDate.HasValue)
            {
                return FeatureLookup.Empty;
            }

            UpcomingRace? upcoming = null;
            var persistedAccepted = false;

            if (persistedUpcoming != null)
            {
                if (ShouldUseUpcomingCandidate(persistedUpcoming, marketId, raceTitle, venueName))
                {
                    upcoming = persistedUpcoming;
                    persistedAccepted = true;
                    CacheUpcomingRace(persistedUpcoming);
                }
                else
                {
                    Console.WriteLine($"        Persisted upcoming race metadata for market {persistedUpcoming.MarketId ?? marketId ?? "<unknown>"} rejected due to metadata misalignment with scraped race (Title/VenueName).");
                }
            }

            if (upcoming == null && !string.IsNullOrWhiteSpace(marketId))
            {
                try
                {
                    var byMarket = GetUpcomingRaceByMarketIdCached(marketId);
                    if (byMarket != null)
                    {
                        upcoming = byMarket;
                        Console.WriteLine($"    Located UpcomingRaces row by market id: UpcomingRaceId={byMarket.UpcomingRaceId}, MarketId={byMarket.MarketId ?? "<null>"}.");
                    }
                    else
                    {
                        Console.WriteLine("    No UpcomingRaces row matched market id search.");
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Failed to query upcoming race metadata by market id: {ex.Message}");
                }

                if (upcoming != null && !ShouldUseUpcomingCandidate(upcoming, marketId, raceTitle, venueName))
                {
                    Console.WriteLine($"        UpcomingRaces row {upcoming.UpcomingRaceId} rejected due to metadata misalignment with scraped race (Title/VenueName).");
                    upcoming = null;
                }
            }

            if (upcoming == null)
            {
                try
                {
                    var byMetadata = FindUpcomingRaceCached(raceDate.Value, raceTitle, venueName);
                    if (byMetadata != null)
                    {
                        upcoming = byMetadata;
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Failed to query upcoming race metadata: {ex.Message}");
                }

                if (upcoming != null && !ShouldUseUpcomingCandidate(upcoming, marketId, raceTitle, venueName))
                {
                    Console.WriteLine($"        UpcomingRaces row {upcoming.UpcomingRaceId} rejected due to metadata misalignment with scraped race (Title/VenueName).");
                    upcoming = null;
                }
            }

            if (upcoming == null)
            {
                upcoming = BuildSyntheticUpcomingRace(
                    raceDate.Value,
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
                    flows);

                if (upcoming == null)
                {
                    Console.WriteLine(" Unable to create synthetic upcoming race metadata; feature lookup aborted.");
                    return FeatureLookup.Empty;
                }

                Console.WriteLine("     Constructed synthetic upcoming race metadata for feature synthesis.");
            }
            var cacheKey = new RacePreparationKey(
                upcoming.RaceDate.Date,
                upcoming.Title ?? raceTitle,
                upcoming.VenueName ?? venueName,
                upcoming.VenueCountry ?? venueCountry);

            lock (_featureLookupCacheLock)
            {
                if (_featureLookupCache.TryGetValue(cacheKey, out var cachedLookup))
                {
                    Console.WriteLine($"  Using cached synthetic feature rows for upcoming race {upcoming.MarketId ?? marketId ?? "<unknown>"}.");
                    return cachedLookup;
                }
            }

            try
            {
                var preparedResults = _trainer.PrepareUpcomingRaces(new[] { (upcoming, flows) });
                var prepared = preparedResults.Count > 0 ? preparedResults[0] : null;
                if (prepared == null)
                {
                    Console.WriteLine("\tSynthetic feature preparation for upcoming race returned no rows.");
                    lock (_featureLookupCacheLock)
                    {
                        _featureLookupCache[cacheKey] = FeatureLookup.Empty;
                    }

                    return FeatureLookup.Empty;
                }

                Console.WriteLine($"  Loaded synthetic feature rows for upcoming race {upcoming.MarketId}; runner count={prepared.Rows.Count}.");
                var lookup = FeatureLookup.FromPreparedRace(prepared);
                lock (_featureLookupCacheLock)
                {
                    _featureLookupCache[cacheKey] = lookup;
                }

                return lookup;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"      Failed to build synthetic feature vector for upcoming race {upcoming.MarketId}:{ex.Message}");
                return FeatureLookup.Empty;
            }
        }

        private FeatureLookup BuildFallbackFeatureLookup(
            DateTime? raceDate,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            TimeSpan? scheduledOff,
            string? raceDetails,
            string? raceType,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows,
            UpcomingRace? persistedUpcoming)
        {
            var fallbackMetadata = string.Format(
                CultureInfo.InvariantCulture,
                "RaceDate={0}, Title='{1}', Venue='{2}', Country='{3}', Off={4}, RaceType='{5}', Going='{6}', BackBook={7}, LayBook={8}, MarketId='{9}', Flows={10}",
                raceDate?.ToString(CultureInfo.InvariantCulture) ?? "<null>",
                raceTitle ?? "<null>",
                venueName ?? "<null>",
                venueCountry ?? "<null>",
                scheduledOff?.ToString() ?? "<null>",
                raceType ?? "<null>",
                going ?? "<null>",
                backBookPercentage?.ToString(CultureInfo.InvariantCulture) ?? "<null>",
                layBookPercentage?.ToString(CultureInfo.InvariantCulture) ?? "<null>",
                marketId ?? "<null>",
                flows?.Count ?? 0);
            Console.WriteLine("    [FallbackLookup] Attempting to build fallback feature lookup with metadata: " + fallbackMetadata);

            if (!raceDate.HasValue || flows == null || flows.Count == 0)
            {
                SetFallbackFeatureError("insufficient race metadata was available to build fallback feature vectors.");
                Console.WriteLine("    [FallbackLookup] Skipping fallback preparation because race metadata or flows were missing.");
                return FeatureLookup.Empty;
            }

            UpcomingRace? upcoming = null;
            if (persistedUpcoming != null && ShouldUseUpcomingCandidate(persistedUpcoming, marketId, raceTitle, venueName))
            {
                upcoming = persistedUpcoming;
                var persistedMessage = string.Format(
                    CultureInfo.InvariantCulture,
                    "    [FallbackLookup] Using persisted upcoming race candidate: UpcomingRaceId={0}, MarketId={1}.",
                    upcoming.UpcomingRaceId,
                    upcoming.MarketId ?? "<null>");
                Console.WriteLine(persistedMessage);
            }

            if (upcoming == null && !string.IsNullOrWhiteSpace(marketId))
            {
                try
                {
                    var byMarket = GetUpcomingRaceByMarketIdCached(marketId);
                    if (byMarket != null)
                    {
                        upcoming = byMarket;
                        Console.WriteLine($"    Located UpcomingRaces row by market id: UpcomingRaceId={byMarket.UpcomingRaceId}, MarketId={byMarket.MarketId ?? "<null>"}.");
                    }
                    else
                    {
                        Console.WriteLine("    No UpcomingRaces row matched market id search.");
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Failed to query upcoming race metadata by market id: {ex.Message}");
                }

                if (upcoming != null && !ShouldUseUpcomingCandidate(upcoming, marketId, raceTitle, venueName))
                {
                    Console.WriteLine($"        UpcomingRaces row {upcoming.UpcomingRaceId} rejected due to metadata misalignment with scraped race (Title/VenueName).");
                    upcoming = null;
                }
            }

            if (upcoming == null)
            {
                Console.WriteLine("    [FallbackLookup] No persisted upcoming race matched; searching repository by metadata.");
                try
                {
                    upcoming = FindUpcomingRaceCached(raceDate.Value, raceTitle, venueName);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Failed to query upcoming race metadata for fallback lookup: {ex.Message}");
                }

                if (upcoming != null && !ShouldUseUpcomingCandidate(upcoming, marketId, raceTitle, venueName))
                {
                    Console.WriteLine("    [FallbackLookup] Repository upcoming race candidate rejected due to metadata mismatch.");
                    upcoming = null;
                }
            }

            if (upcoming == null)
            {
                Console.WriteLine("    [FallbackLookup] Attempting to build synthetic upcoming race metadata for fallback preparation.");
                upcoming = BuildSyntheticUpcomingRace(
                    raceDate.Value,
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
                    flows);
            }

            if (upcoming == null)
            {
                SetFallbackFeatureError("no upcoming race metadata matched the scraped race; fallback feature synthesis skipped.");
                Console.WriteLine("    [FallbackLookup] Failed to resolve upcoming race metadata; fallback lookup is empty.");
                return FeatureLookup.Empty;
            }

            var cacheKey = new RacePreparationKey(
                upcoming.RaceDate,
                upcoming.Title ?? raceTitle,
                upcoming.VenueName ?? venueName,
                upcoming.VenueCountry ?? venueCountry);

            lock (_featureLookupCacheLock)
            {
                if (_featureLookupCache.TryGetValue(cacheKey, out var cachedLookup))
                {
                    if (_featureLookupErrorCache.TryGetValue(cacheKey, out var cachedError))
                    {
                        SetFallbackFeatureError(cachedError);
                    }
                    else
                    {
                        SetFallbackFeatureError(null);
                    }

                    Console.WriteLine($"  Using cached synthetic feature rows for upcoming race {upcoming.MarketId ?? marketId ?? "<unknown>"}.");
                    var cacheSummary = string.Format(
                        CultureInfo.InvariantCulture,
                        "    [FallbackLookup] Cached lookup entry contains {0} runner(s); error cache entry {1}.",
                        cachedLookup.Count,
                        cachedError == null ? "cleared" : "present");
                    Console.WriteLine(cacheSummary);
                    return cachedLookup;
                }
            }

            try
            {
                var trainerMessage = string.Format(
                    CultureInfo.InvariantCulture,
                    "    [FallbackLookup] Preparing upcoming race via trainer with {0} runner flow(s). Using metadata: UpcomingRaceId={1}, MarketId={2}.",
                    flows.Count,
                    upcoming.UpcomingRaceId,
                    upcoming.MarketId ?? marketId ?? "<unknown>");
                Console.WriteLine(trainerMessage);
                var prepared = _trainer.PrepareUpcomingRace(upcoming, flows);
                if (prepared == null || prepared.Rows.Count == 0)
                {
                    const string error = "trainer returned no rows when preparing fallback features; check that the HorseRacingDb connection string points to a populated SQL Server database.";
                    SetFallbackFeatureError(error);
                    lock (_featureLookupCacheLock)
                    {
                        _featureLookupCache[cacheKey] = FeatureLookup.Empty;
                        _featureLookupErrorCache[cacheKey] = error;
                    }

                    return FeatureLookup.Empty;
                }

                var preparedSummary = string.Format(
                    CultureInfo.InvariantCulture,
                    "    [FallbackLookup] Trainer prepared {0} row(s); building feature lookup for fallback usage.",
                    prepared.Rows.Count);
                Console.WriteLine(preparedSummary);
                var lookup = FeatureLookup.FromPreparedRace(prepared);
                SetFallbackFeatureError(null);
                lock (_featureLookupCacheLock)
                {
                    _featureLookupCache[cacheKey] = lookup;
                    _featureLookupErrorCache[cacheKey] = null;
                }

                var lookupSummary = string.Format(
                    CultureInfo.InvariantCulture,
                    "    [FallbackLookup] Fallback lookup built successfully with {0} runner feature set(s).",
                    lookup.Count);
                Console.WriteLine(lookupSummary);
                return lookup;
            }
            catch (Exception ex)
            {
                var message = $"trainer failed to build fallback feature vectors ({ex.Message}). Ensure the HorseRacingDb connection string is valid and the database is accessible.";
                SetFallbackFeatureError(message);
                Console.Error.WriteLine($"      Failed to build fallback synthetic feature vector for upcoming race {upcoming.MarketId ?? marketId ?? "<unknown>"}:{ex.Message}");
                lock (_featureLookupCacheLock)
                {
                    _featureLookupCache[cacheKey] = FeatureLookup.Empty;
                    _featureLookupErrorCache[cacheKey] = message;
                }
                Console.WriteLine("    [FallbackLookup] Trainer threw during preparation; cached empty lookup with error message.");
                return FeatureLookup.Empty;
            }
        }
        private UpcomingRace? TryResolveUpcomingRace(RaceDayReport race)
        {
            UpcomingRace? upcoming = null;

            if (!string.IsNullOrWhiteSpace(race.MarketId))
            {
                try
                {
                    upcoming = _repo.GetUpcomingRaceByMarketId(race.MarketId);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\t[DayReport] Failed to resolve upcoming race by market {race.MarketId}: {ex.Message}");
                }
            }

            if (upcoming != null)
            {
                return upcoming;
            }

            if (!race.RaceDate.HasValue)
            {
                return null;
            }

            try
            {
                return _repo.FindUpcomingRace(race.RaceDate.Value, race.RaceTitle, race.VenueName);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\t[DayReport] Failed to resolve upcoming race by metadata for {race.RaceTitle ?? race.MarketId}: {ex.Message}");
                return null;
            }
        }

        private int? TryRecordUpcomingRace(RaceDayReport race)
        {
            if (race == null || !race.RaceDate.HasValue)
            {
                return null;
            }

            if (string.IsNullOrWhiteSpace(race.MarketId))
            {
                return null;
            }

            try
            {
                var metadata = ParseRaceMetadata(race);

                short? distanceYards = null;
                if (metadata.DistanceYards > 0)
                {
                    distanceYards = (short)Math.Min(metadata.DistanceYards, short.MaxValue);
                }

                byte? runnerCount = null;
                if (race.Runners != null && race.Runners.Count > 0)
                {
                    runnerCount = (byte)Math.Min(race.Runners.Count, byte.MaxValue);
                }

                var upcoming = new UpcomingRace
                {
                    MarketId = race.MarketId,
                    RaceDate = race.RaceDate.Value.Date,
                    ScheduledOff = race.OffTime,
                    VenueName = string.IsNullOrWhiteSpace(race.VenueName) ? null : race.VenueName!.Trim(),
                    VenueCountry = string.IsNullOrWhiteSpace(race.VenueCountry) ? null : race.VenueCountry!.Trim(),
                    Title = string.IsNullOrWhiteSpace(race.RaceTitle) ? null : race.RaceTitle!.Trim(),
                    RaceDetails = string.IsNullOrWhiteSpace(race.RaceDetails) ? null : race.RaceDetails!.Trim(),
                    RaceType = metadata.RaceType,
                    Surface = metadata.Surface,
                    Going = string.IsNullOrWhiteSpace(race.Going) ? metadata.Going : race.Going!.Trim(),
                    DistanceYards = distanceYards,
                    DistanceText = metadata.DistanceText,
                    RunnerCount = runnerCount,
                    BackBookPercentage = race.BackBookPercentage,
                    LayBookPercentage = race.LayBookPercentage
                };

                lock (_repoLock)
                {
                    var upcomingId = _repo.UpsertUpcomingRace(upcoming);
                    CacheUpcomingRace(upcoming);
                    Console.WriteLine($"\t[DayReport] Recorded upcoming race {upcomingId} for {race.VenueName?.Trim() ?? "unknown venue"} on {race.RaceDate:yyyy-MM-dd}.");
                    return upcomingId;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\t[DayReport] Failed to record upcoming race for {race.RaceTitle ?? race.MarketId}: {ex.Message}");
                return null;
            }
        }

        public static bool ShouldUseUpcomingCandidate(
            UpcomingRace upcoming,
            string? marketId,
            string? raceTitle,
            string? venueName)
        {
            if (upcoming == null)
            {
                return false;
            }

            if (!string.IsNullOrWhiteSpace(upcoming.MarketId) &&
                !string.IsNullOrWhiteSpace(marketId) &&
                !string.Equals(upcoming.MarketId, marketId, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            var normalizedTitle = RacingRepository.NormalizeLookupKey(raceTitle);
            var normalizedVenue = RacingRepository.NormalizeLookupKey(venueName);
            var candidateTitle = RacingRepository.NormalizeLookupKey(upcoming.Title);
            var candidateVenue = RacingRepository.NormalizeLookupKey(upcoming.VenueName);

            var hasTitle = !string.IsNullOrEmpty(normalizedTitle);
            var hasVenue = !string.IsNullOrEmpty(normalizedVenue);
            var titleAligned = hasTitle && IsNormalizedMatch(normalizedTitle, candidateTitle);
            var venueAligned = hasVenue && IsNormalizedMatch(normalizedVenue, candidateVenue);

            if ((hasTitle || hasVenue) && !titleAligned && !venueAligned)
            {
                return false;
            }

            return true;
        }

        private static bool IsNormalizedMatch(string expected, string candidate)
        {
            if (string.IsNullOrEmpty(expected) || string.IsNullOrEmpty(candidate))
            {
                return false;
            }

            if (candidate == expected)
            {
                return true;
            }

            return candidate.Contains(expected, StringComparison.Ordinal) ||
                   expected.Contains(candidate, StringComparison.Ordinal);
        }

        private UpcomingRace? BuildSyntheticUpcomingRace(
            DateTime raceDate,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            TimeSpan? scheduledOff,
            string? raceDetails,
            string? raceType,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows)
        {
            string sanitizedMarketId;
            if (!string.IsNullOrWhiteSpace(marketId))
            {
                sanitizedMarketId = marketId.Trim();
            }
            else if (!string.IsNullOrWhiteSpace(raceTitle) || !string.IsNullOrWhiteSpace(venueName))
            {
                sanitizedMarketId = $"synthetic:{raceDate:yyyyMMdd}:{(venueName ?? raceTitle ?? "race")}";
            }
            else
            {
                return null;
            }

            var metadataSource = new RaceDayReport
            {
                RaceTitle = raceTitle,
                RaceDetails = string.IsNullOrWhiteSpace(raceDetails) ? null : raceDetails.Trim(),
                RaceType = string.IsNullOrWhiteSpace(raceType) ? null : raceType.Trim(),
                Going = string.IsNullOrWhiteSpace(going) ? null : going.Trim()
            };
            var parsedMetadata = ParseRaceMetadata(metadataSource);

            short? distanceYards = parsedMetadata.DistanceYards > 0
                ? (short)Math.Min(parsedMetadata.DistanceYards, short.MaxValue)
                : null;

            byte? runnerCount = (flows != null && flows.Count > 0)
                ? (byte)Math.Min(flows.Count, byte.MaxValue)
                : (byte?)null;

            return new UpcomingRace
            {
                MarketId = sanitizedMarketId,
                RaceDate = raceDate.Date,
                ScheduledOff = scheduledOff,
                VenueName = string.IsNullOrWhiteSpace(venueName) ? null : venueName.Trim(),
                VenueCountry = string.IsNullOrWhiteSpace(venueCountry) ? null : venueCountry.Trim(),
                Title = string.IsNullOrWhiteSpace(raceTitle) ? null : raceTitle.Trim(),
                RaceDetails = string.IsNullOrWhiteSpace(raceDetails) ? null : raceDetails.Trim(),
                RaceType = parsedMetadata.RaceType,
                Surface = parsedMetadata.Surface,
                Going = string.IsNullOrWhiteSpace(going) ? parsedMetadata.Going : going.Trim(),
                DistanceYards = distanceYards,
                DistanceText = parsedMetadata.DistanceText,
                RunnerCount = runnerCount,
                BackBookPercentage = backBookPercentage,
                LayBookPercentage = layBookPercentage
            };
        }

        private readonly struct RacePreparationKey : IEquatable<RacePreparationKey>
        {
            public RacePreparationKey(DateTime raceDate, string? title, string? venue, string? country)
            {
                RaceDate = raceDate.Date;
                TitleKey = RacingRepository.NormalizeLookupKey(title);
                VenueKey = RacingRepository.NormalizeLookupKey(venue);
                CountryKey = string.IsNullOrWhiteSpace(country)
                    ? string.Empty
                    : country.Trim().ToUpperInvariant();
            }

            public DateTime RaceDate { get; }
            public string TitleKey { get; }
            public string VenueKey { get; }
            public string CountryKey { get; }

            public bool Equals(RacePreparationKey other)
            {
                return RaceDate == other.RaceDate &&
                    string.Equals(TitleKey, other.TitleKey, StringComparison.Ordinal) &&
                    string.Equals(VenueKey, other.VenueKey, StringComparison.Ordinal) &&
                    string.Equals(CountryKey, other.CountryKey, StringComparison.Ordinal);
            }

            public override bool Equals(object? obj) => obj is RacePreparationKey other && Equals(other);

            public override int GetHashCode()
            {
                var hash = new HashCode();
                hash.Add(RaceDate);
                hash.Add(TitleKey, StringComparer.Ordinal);
                hash.Add(VenueKey, StringComparer.Ordinal);
                hash.Add(CountryKey, StringComparer.Ordinal);
                return hash.ToHashCode();
            }
        }

        private sealed class FeatureLookup
        {
            private readonly Dictionary<string, List<Dictionary<string, object?>>> _byIdentifier;
            private readonly int _rowCount;

            private FeatureLookup(
                Dictionary<string, List<Dictionary<string, object?>>> byIdentifier,
                int rowCount)
            {
                _byIdentifier = byIdentifier;
                _rowCount = rowCount;
            }

            public static FeatureLookup Empty { get; } = new FeatureLookup(
                new Dictionary<string, List<Dictionary<string, object?>>>(StringComparer.OrdinalIgnoreCase),
                rowCount: 0);

            public static FeatureLookup FromPreparedRace(PreparedRace race)
            {
                if (race == null)
                    throw new ArgumentNullException(nameof(race));

                var lookup = new Dictionary<string, List<Dictionary<string, object?>>>(StringComparer.OrdinalIgnoreCase);
                foreach (var row in race.Rows)
                {
                    AddRowIdentifiers(lookup, row);
                }

                return new FeatureLookup(lookup, race.Rows.Count);
            }
            public int Count => _rowCount;

            public Dictionary<string, object?>? FindByRunner(RunnerFlow? flow)
            {
                if (flow == null)
                {
                    return null;
                }

                var normalizedHorse = NormalizeName(flow.HorseName);
                var normalizedJockey = NormalizeName(flow.JockeyName);
                var normalizedTrainer = NormalizeName(flow.TrainerName);
                var clothKey = NormalizeNumeric(flow.ClothNumber, treatZeroOrNegativeAsMissing: true);
                var drawKey = NormalizeNumeric(flow.Draw, treatZeroOrNegativeAsMissing: true);

                var horseIdKey = ExtractNumericFeature(flow, "HorseId");
                var runnerResultKey = ExtractNumericFeature(flow, "RunnerResultId");

                return FindByIdentifiers(
                    normalizedHorse,
                    normalizedJockey,
                    normalizedTrainer,
                    clothKey,
                    drawKey,
                    horseIdKey,
                    runnerResultKey);
            }

            public Dictionary<string, object?>? FindByHorse(string? horseName)
            {
                var normalizedHorse = NormalizeName(horseName);
                if (string.IsNullOrEmpty(normalizedHorse))
                {
                    return null;
                }

                return FindByIdentifiers(normalizedHorse, null, null, null, null, null, null);
            }

            private Dictionary<string, object?>? FindByIdentifiers(
                string? normalizedHorse,
                string? normalizedJockey,
                string? normalizedTrainer,
                string? clothKey,
                string? drawKey,
                string? horseIdKey,
                string? runnerResultKey)
            {
                foreach (var key in BuildIdentifierKeys(
                             normalizedHorse,
                             normalizedJockey,
                             normalizedTrainer,
                             clothKey,
                             drawKey,
                             horseIdKey,
                             runnerResultKey))
                {
                    if (_byIdentifier.TryGetValue(key, out var candidates) && candidates != null && candidates.Count > 0)
                    {
                        var selected = SelectBestCandidate(candidates, key);
                        if (selected != null)
                        {
                            return selected;
                        }
                    }
                }

                return null;
            }

            private static void AddRowIdentifiers(
                Dictionary<string, List<Dictionary<string, object?>>> lookup,
                IDictionary<string, object?>? row)
            {
                if (row == null)
                {
                    return;
                }

                var copy = row is Dictionary<string, object?> dict
                    ? new Dictionary<string, object?>(dict, StringComparer.OrdinalIgnoreCase)
                    : new Dictionary<string, object?>(row, StringComparer.OrdinalIgnoreCase);

                string? horseName = null;
                if (copy.TryGetValue("HorseName", out var horseObj) && horseObj is string horseText)
                {
                    horseName = horseText;
                }

                string? jockeyName = null;
                if (copy.TryGetValue("JockeyName", out var jockeyObj) && jockeyObj is string jockeyText)
                {
                    jockeyName = jockeyText;
                }

                string? trainerName = null;
                if (copy.TryGetValue("TrainerName", out var trainerObj) && trainerObj is string trainerText)
                {
                    trainerName = trainerText;
                }

                var clothKey = NormalizeNumeric(copy.TryGetValue("SaddleclothNumber", out var clothObj)
                        ? clothObj
                        : null,
                    treatZeroOrNegativeAsMissing: true);
                var drawKey = NormalizeNumeric(copy.TryGetValue("Draw", out var drawObj)
                        ? drawObj
                        : null,
                    treatZeroOrNegativeAsMissing: true);
                var horseIdKey = NormalizeNumeric(copy.TryGetValue("HorseId", out var horseIdObj)
                        ? horseIdObj
                        : null,
                    treatZeroOrNegativeAsMissing: true);
                var runnerResultKey = NormalizeNumeric(copy.TryGetValue("RunnerResultId", out var runnerObj)
                        ? runnerObj
                        : null,
                    treatZeroOrNegativeAsMissing: true);
                var normalizedHorse = NormalizeName(horseName);
                var normalizedJockey = NormalizeName(jockeyName);
                var normalizedTrainer = NormalizeName(trainerName);

                foreach (var key in BuildIdentifierKeys(
                             normalizedHorse,
                             normalizedJockey,
                             normalizedTrainer,
                             clothKey,
                             drawKey,
                             horseIdKey,
                             runnerResultKey))
                {
                    if (!lookup.TryGetValue(key, out var list))
                    {
                        list = new List<Dictionary<string, object?>>();
                        lookup[key] = list;
                    }

                    if (!list.Contains(copy))
                    {
                        list.Add(copy);
                    }
                }
            }

            private static Dictionary<string, object?>? SelectBestCandidate(
                IReadOnlyList<Dictionary<string, object?>> candidates,
                string identifierKey)
            {
                if (candidates == null || candidates.Count == 0)
                {
                    return null;
                }

                if (candidates.Count == 1)
                {
                    return candidates[0];
                }
                var candidateInfos = new List<(Dictionary<string, object?> Candidate, int Index, int MissingCount, int Score)>();

                for (var i = 0; i < candidates.Count; i++)
                {
                    var candidate = candidates[i];
                    if (candidate == null)
                    {
                        continue;
                    }

                    var missingKeys = GetMissingHistoricalFeatureKeys(candidate);
                    var missingCount = missingKeys?.Count ?? 0;
                    var score = CalculateCandidateScore(candidate);

                    candidateInfos.Add((candidate, i, missingCount, score));
                }

                if (candidateInfos.Count == 0)
                {
                    return candidates[0];
                }

                var ordered = candidateInfos
                    .OrderBy(info => info.MissingCount)
                    .ThenByDescending(info => info.Score)
                    .ThenBy(info => info.Index)
                    .ToList();

                var aggregate = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                var contributingIndexes = new HashSet<int>();

                var trackedKeys = ResolveTrackedFeatureKeys();
                foreach (var key in trackedKeys)
                {
                    if (string.IsNullOrWhiteSpace(key))
                    {
                        continue;
                    }

                    var normalizedKey = key.Trim();
                    if (normalizedKey.Length == 0)
                    {
                        continue;
                    }

                    if (TryGetMeaningfulValue(aggregate, normalizedKey, out _))
                    {
                        continue;
                    }

                    foreach (var info in ordered)
                    {
                        if (!TryGetMeaningfulValue(info.Candidate, normalizedKey, out var value))
                        {
                            continue;
                        }

                        aggregate[normalizedKey] = value;
                        contributingIndexes.Add(info.Index);
                        break;
                    }
                }

                foreach (var info in ordered)
                {
                    foreach (var kvp in info.Candidate)
                    {
                        var key = kvp.Key;
                        if (string.IsNullOrWhiteSpace(key))
                        {
                            continue;
                        }

                        var normalizedKey = key.Trim();
                        if (TryGetMeaningfulValue(aggregate, normalizedKey, out _))
                        {
                            continue;
                        }

                        if (!TryGetMeaningfulValue(info.Candidate, normalizedKey, out var value))
                        {
                            continue;
                        }

                        aggregate[normalizedKey] = value;
                        contributingIndexes.Add(info.Index);
                    }
                }

                if (aggregate.Count == 0)
                {
                    return ordered[0].Candidate;
                }

                var primary = ordered[0];
                if (primary.Index > 0 || contributingIndexes.Count > 1)
                {
                    var completenessText = primary.MissingCount == 0
                        ? "complete"
                        : $"missingRequired={primary.MissingCount}";
                    string detailText;

                    if (contributingIndexes.Count > 1)
                    {
                        var mergedText = string.Join(",", contributingIndexes
                            .OrderBy(idx => idx)
                            .Select(idx => (idx + 1).ToString(CultureInfo.InvariantCulture)));
                        detailText = $"mergedCandidates={mergedText}";
                    }
                    else
                    {
                        detailText = $"candidate {primary.Index + 1}/{candidates.Count}";
                    }

                    Console.WriteLine(
                        $"    [FallbackLookup] Merged feature row(s) for '{identifierKey}' ({detailText}, score={primary.Score}, {completenessText}).");
                }

                return aggregate;
            }


            private static int CalculateCandidateScore(Dictionary<string, object?> candidate)
            {
                var score = 0;

                if (TryConvertToInt32(candidate, "RunnerResultId", out var runnerResultId) && runnerResultId > 0)
                {
                    score += 16;
                }

                if (TryConvertToBool(candidate, "DistanceBeatenKnown", out var beatenKnown) && beatenKnown)
                {
                    score += 8;
                }
                else if (TryConvertToFloat(candidate, "DistanceBeatenLengths", out var beatenLengths) && beatenLengths > 0f)
                {
                    score += 6;
                }

                if (TryConvertToFloat(candidate, "WinningTimeMs", out var winningTime) && winningTime > 0f)
                {
                    score += 4;
                }

                if (TryConvertToFloat(candidate, "RaceAvgSpeedLast5", out var raceAvgSpeed) && raceAvgSpeed > 0f)
                {
                    score += 2;
                }

                if (TryConvertToFloat(candidate, "RaceAvgWinRateLast5", out var raceAvgWinRate) && raceAvgWinRate > 0f)
                {
                    score += 1;
                }

                if (TryConvertToFloat(candidate, "AvgSpeedLast5", out var runnerSpeed) && runnerSpeed > 0f)
                {
                    score += 1;
                }

                if (TryConvertToFloat(candidate, "WinRateLast5", out var runnerWinRate) && runnerWinRate > 0f)
                {
                    score += 1;
                }

                return score;
            }

            private static bool TryConvertToFloat(Dictionary<string, object?> source, string key, out float value)
            {
                value = 0f;
                if (!source.TryGetValue(key, out var raw) || raw == null)
                {
                    return false;
                }

                try
                {
                    switch (raw)
                    {
                        case float f when !float.IsNaN(f) && !float.IsInfinity(f):
                            value = f;
                            return true;
                        case double d when !double.IsNaN(d) && !double.IsInfinity(d):
                            value = (float)d;
                            return !float.IsNaN(value) && !float.IsInfinity(value);
                        case decimal m:
                            value = (float)m;
                            return !float.IsNaN(value) && !float.IsInfinity(value);
                        case int i:
                            value = i;
                            return true;
                        case long l:
                            value = l;
                            return true;
                        case short s:
                            value = s;
                            return true;
                        case byte b:
                            value = b;
                            return true;
                        case string s when float.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var parsedInvariant):
                            value = parsedInvariant;
                            return true;
                        case string s when float.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.CurrentCulture, out var parsedCurrent):
                            value = parsedCurrent;
                            return true;
                    }
                }
                catch
                {
                }

                return false;
            }

            private static bool TryConvertToInt32(Dictionary<string, object?> source, string key, out int value)
            {
                value = 0;
                if (!source.TryGetValue(key, out var raw) || raw == null)
                {
                    return false;
                }

                try
                {
                    switch (raw)
                    {
                        case int i:
                            value = i;
                            return true;
                        case long l when l <= int.MaxValue && l >= int.MinValue:
                            value = (int)l;
                            return true;
                        case short s:
                            value = s;
                            return true;
                        case byte b:
                            value = b;
                            return true;
                        case sbyte sb:
                            value = sb;
                            return true;
                        case ushort us when us <= int.MaxValue:
                            value = (int)us;
                            return true;
                        case uint ui when ui <= int.MaxValue:
                            value = (int)ui;
                            return true;
                        case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedInvariant):
                            value = parsedInvariant;
                            return true;
                        case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.CurrentCulture, out var parsedCurrent):
                            value = parsedCurrent;
                            return true;
                    }
                }
                catch
                {
                }

                return false;
            }

            private static bool TryConvertToBool(Dictionary<string, object?> source, string key, out bool value)
            {
                value = false;
                if (!source.TryGetValue(key, out var raw) || raw == null)
                {
                    return false;
                }

                switch (raw)
                {
                    case bool b:
                        value = b;
                        return true;
                    case string s when bool.TryParse(s, out var parsed):
                        value = parsed;
                        return true;
                    case int i when i == 0 || i == 1:
                        value = i == 1;
                        return true;
                }

                return false;
            }
            private static IEnumerable<string> BuildIdentifierKeys(
                string? normalizedHorse,
                string? normalizedJockey,
                string? normalizedTrainer,
                string? clothKey,
                string? drawKey,
                string? horseIdKey,
                string? runnerResultKey)
            {
                var components = new List<(string Prefix, string? Value)>
                {
                    ("horse", normalizedHorse),
                    ("horseId", horseIdKey),
                    ("runnerResult", runnerResultKey),
                    ("cloth", clothKey),
                    ("draw", drawKey),
                    ("jockey", normalizedJockey),
                    ("trainer", normalizedTrainer)
                };

                var available = components
                    .Where(c => !string.IsNullOrEmpty(c.Value))
                    .Select(c => (c.Prefix, Value: c.Value!))
                    .ToArray();

                if (available.Length == 0)
                {
                    return Array.Empty<string>();
                }

                var weights = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
                {
                    ["horse"] = 64,
                    ["horseId"] = 64,
                    ["runnerResult"] = 48,
                    ["jockey"] = 16,
                    ["trainer"] = 12,
                    ["cloth"] = 8,
                    ["draw"] = 8
                };

                var combinations = new List<(string Key, int Weight, int PartCount)>();
                var total = 1 << available.Length;

                for (var mask = 1; mask < total; mask++)
                {
                    var parts = new List<string>();
                    var weight = 0;

                    for (var bit = 0; bit < available.Length; bit++)
                    {
                        if ((mask & (1 << bit)) == 0)
                        {
                            continue;
                        }

                        var entry = available[bit];
                        parts.Add($"{entry.Prefix}:{entry.Value}");

                        if (weights.TryGetValue(entry.Prefix, out var partWeight))
                        {
                            weight += partWeight;
                        }
                    }

                    if (parts.Count == 0)
                    {
                        continue;
                    }

                    combinations.Add((string.Join("|", parts), weight, parts.Count));
                }

                combinations.Sort((left, right) =>
                {
                    var weightComparison = right.Weight.CompareTo(left.Weight);
                    if (weightComparison != 0)
                    {
                        return weightComparison;
                    }

                    var partComparison = right.PartCount.CompareTo(left.PartCount);
                    if (partComparison != 0)
                    {
                        return partComparison;
                    }

                    return string.Compare(left.Key, right.Key, StringComparison.OrdinalIgnoreCase);
                });

                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var orderedKeys = new List<string>();

                foreach (var combination in combinations)
                {
                    if (seen.Add(combination.Key))
                    {
                        orderedKeys.Add(combination.Key);
                    }
                }

                return orderedKeys;
            }
            private static string? ExtractNumericFeature(RunnerFlow flow, string key)
            {
                if (flow?.FeatureValues == null || string.IsNullOrWhiteSpace(key))
                {
                    return null;
                }

                if (!flow.FeatureValues.TryGetValue(key, out var value) || value == null)
                {
                    return null;
                }

                return NormalizeNumeric(value, treatZeroOrNegativeAsMissing: true);
            }
            private static string? NormalizeNumeric(object? value, bool treatZeroOrNegativeAsMissing = false)
            {
                if (value == null)
                {
                    return null;
                }

                try
                {
                    int result;
                    switch (value)
                    {
                        case byte b:
                            result = b;
                            break;
                        case short s:
                            result = s;
                            break;
                        case int i:
                            result = i;
                            break;
                        case long l:
                            result = checked((int)l);
                            break;
                        case float f:
                            result = (int)Math.Round(f);
                            break;
                        case double d:
                            result = (int)Math.Round(d);
                            break;
                        case decimal m:
                            result = (int)Math.Round(m);
                            break;
                        case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed):
                            result = parsed;
                            break;
                        default:
                            result = Convert.ToInt32(value, CultureInfo.InvariantCulture);
                            break;
                    }

                    if (treatZeroOrNegativeAsMissing && result <= 0)
                    {
                        return null;
                    }

                    return result.ToString(CultureInfo.InvariantCulture);
                }
                catch
                {
                    return null;
                }
            }
        }
    }
}