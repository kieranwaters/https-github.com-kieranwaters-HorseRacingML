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
            ResetUpcomingRaceLookupCache();
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
            }
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
            IReadOnlyList<IDictionary<string, object?>>? preparedRows,
            UpcomingRace? persistedUpcoming)
        {
            if (preparedRows != null && preparedRows.Count > 0)
            {
                Console.WriteLine($"  Using {preparedRows.Count} pre-provided feature row(s) for lookup.");
                return FeatureLookup.FromRows(preparedRows);
            }

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
                        Console.WriteLine($"    Located UpcomingRaces row: UpcomingRaceId={byMetadata.UpcomingRaceId}, MarketId={byMetadata.MarketId ?? "<null>"}.");
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
            else
            {
                if (persistedAccepted)
                {
                    Console.WriteLine($"        Using persisted upcoming race metadata for market {upcoming.MarketId ?? marketId ?? "<unknown>"}.");
                }
                else
                {
                    Console.WriteLine($"        Using UpcomingRaces metadata from repository for market {upcoming.MarketId ?? marketId ?? "<unknown>"}.");
                }
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
            if (!raceDate.HasValue || flows == null || flows.Count == 0)
            {
                SetFallbackFeatureError("insufficient race metadata was available to build fallback feature vectors.");
                return FeatureLookup.Empty;
            }

            UpcomingRace? upcoming = null;
            if (persistedUpcoming != null && ShouldUseUpcomingCandidate(persistedUpcoming, marketId, raceTitle, venueName))
            {
                upcoming = persistedUpcoming;
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
                    upcoming = FindUpcomingRaceCached(raceDate.Value, raceTitle, venueName);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Failed to query upcoming race metadata for fallback lookup: {ex.Message}");
                }

                if (upcoming != null && !ShouldUseUpcomingCandidate(upcoming, marketId, raceTitle, venueName))
                {
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
            }

            if (upcoming == null)
            {
                SetFallbackFeatureError("no upcoming race metadata matched the scraped race; fallback feature synthesis skipped.");
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
                    return cachedLookup;
                }
            }

            try
            {
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

                var lookup = FeatureLookup.FromPreparedRace(prepared);
                SetFallbackFeatureError(null);
                lock (_featureLookupCacheLock)
                {
                    _featureLookupCache[cacheKey] = lookup;
                    _featureLookupErrorCache[cacheKey] = null;
                }

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
                return FeatureLookup.Empty;
            }
        }


        private static bool IsFutureRace(DateTime raceDate, TimeSpan? offTime)
        {
            var now = DateTime.Now;
            if (raceDate.Date > now.Date)
            {
                return true;
            }

            if (raceDate.Date < now.Date)
            {
                return false;
            }

            if (offTime.HasValue)
            {
                return offTime.Value > now.TimeOfDay;
            }

            return true;
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
                    Class = metadata.Class,
                    AgeRestriction = metadata.AgeRestriction,
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
                Class = parsedMetadata.Class,
                AgeRestriction = parsedMetadata.AgeRestriction,
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
            private readonly Dictionary<string, Dictionary<string, object?>> _byIdentifier;

            private FeatureLookup(Dictionary<string, Dictionary<string, object?>> byIdentifier)
            {
                _byIdentifier = byIdentifier;
            }

            public static FeatureLookup Empty { get; } = new FeatureLookup(
                new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase));

            public static FeatureLookup FromPreparedRace(PreparedRace race)
            {
                if (race == null)
                    throw new ArgumentNullException(nameof(race));

                var lookup = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);
                foreach (var row in race.Rows)
                {
                    AddRowIdentifiers(lookup, row);
                }

                return new FeatureLookup(lookup);
            }

            public static FeatureLookup FromRows(IReadOnlyList<IDictionary<string, object?>> rows)
            {
                if (rows == null)
                    throw new ArgumentNullException(nameof(rows));

                if (rows.Count == 0)
                {
                    return Empty;
                }

                var lookup = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);
                foreach (var row in rows)
                {
                    AddRowIdentifiers(lookup, row);
                }

                return new FeatureLookup(lookup);
            }

            public Dictionary<string, object?>? FindByRunner(RunnerFlow? flow)
            {
                if (flow == null)
                {
                    return null;
                }

                var normalizedHorse = NormalizeName(flow.HorseName);
                var normalizedJockey = NormalizeName(flow.JockeyName);
                var normalizedTrainer = NormalizeName(flow.TrainerName);
                var clothKey = flow.ClothNumber.HasValue
                    ? flow.ClothNumber.Value.ToString(CultureInfo.InvariantCulture)
                    : null;
                var drawKey = flow.Draw.HasValue
                    ? flow.Draw.Value.ToString(CultureInfo.InvariantCulture)
                    : null;

                return FindByIdentifiers(normalizedHorse, normalizedJockey, normalizedTrainer, clothKey, drawKey);
            }

            public Dictionary<string, object?>? FindByHorse(string? horseName)
            {
                var normalizedHorse = NormalizeName(horseName);
                if (string.IsNullOrEmpty(normalizedHorse))
                {
                    return null;
                }

                return FindByIdentifiers(normalizedHorse, null, null, null, null);
            }

            private Dictionary<string, object?>? FindByIdentifiers(
                string? normalizedHorse,
                string? normalizedJockey,
                string? normalizedTrainer,
                string? clothKey,
                string? drawKey)
            {
                foreach (var key in BuildIdentifierKeys(normalizedHorse, normalizedJockey, normalizedTrainer, clothKey, drawKey))
                {
                    if (_byIdentifier.TryGetValue(key, out var features))
                    {
                        return features;
                    }
                }

                return null;
            }

            private static void AddRowIdentifiers(
                Dictionary<string, Dictionary<string, object?>> lookup,
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
                    : null);
                var drawKey = NormalizeNumeric(copy.TryGetValue("Draw", out var drawObj)
                    ? drawObj
                    : null);

                var normalizedHorse = NormalizeName(horseName);
                var normalizedJockey = NormalizeName(jockeyName);
                var normalizedTrainer = NormalizeName(trainerName);

                foreach (var key in BuildIdentifierKeys(normalizedHorse, normalizedJockey, normalizedTrainer, clothKey, drawKey))
                {
                    if (!lookup.ContainsKey(key))
                    {
                        lookup[key] = copy;
                    }
                }
            }

            private static IEnumerable<string> BuildIdentifierKeys(
                string? normalizedHorse,
                string? normalizedJockey,
                string? normalizedTrainer,
                string? clothKey,
                string? drawKey)
            {
                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var keys = new List<string>();

                void TryAdd(string? candidate)
                {
                    if (string.IsNullOrEmpty(candidate))
                    {
                        return;
                    }

                    if (seen.Add(candidate))
                    {
                        keys.Add(candidate);
                    }
                }

                if (!string.IsNullOrEmpty(normalizedHorse))
                {
                    TryAdd($"horse:{normalizedHorse}");

                    if (!string.IsNullOrEmpty(clothKey))
                    {
                        TryAdd($"horse+cloth:{normalizedHorse}|{clothKey}");
                    }

                    if (!string.IsNullOrEmpty(drawKey))
                    {
                        TryAdd($"horse+draw:{normalizedHorse}|{drawKey}");
                    }

                    if (!string.IsNullOrEmpty(normalizedJockey))
                    {
                        TryAdd($"horse+jockey:{normalizedHorse}|{normalizedJockey}");
                    }

                    if (!string.IsNullOrEmpty(normalizedTrainer))
                    {
                        TryAdd($"horse+trainer:{normalizedHorse}|{normalizedTrainer}");
                    }
                }

                if (!string.IsNullOrEmpty(clothKey))
                {
                    TryAdd($"cloth:{clothKey}");
                }

                if (!string.IsNullOrEmpty(drawKey))
                {
                    TryAdd($"draw:{drawKey}");
                }

                if (!string.IsNullOrEmpty(normalizedJockey))
                {
                    TryAdd($"jockey:{normalizedJockey}");
                }

                if (!string.IsNullOrEmpty(normalizedTrainer))
                {
                    TryAdd($"trainer:{normalizedTrainer}");
                }

                return keys;
            }

            private static string? NormalizeNumeric(object? value)
            {
                if (value == null)
                {
                    return null;
                }

                try
                {
                    switch (value)
                    {
                        case byte b:
                            return b.ToString(CultureInfo.InvariantCulture);
                        case short s:
                            return s.ToString(CultureInfo.InvariantCulture);
                        case int i:
                            return i.ToString(CultureInfo.InvariantCulture);
                        case long l:
                            return l.ToString(CultureInfo.InvariantCulture);
                        case float f:
                            return ((int)Math.Round(f)).ToString(CultureInfo.InvariantCulture);
                        case double d:
                            return ((int)Math.Round(d)).ToString(CultureInfo.InvariantCulture);
                        case decimal m:
                            return ((int)Math.Round(m)).ToString(CultureInfo.InvariantCulture);
                        case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed):
                            return parsed.ToString(CultureInfo.InvariantCulture);
                    }

                    var converted = Convert.ToInt32(value, CultureInfo.InvariantCulture);
                    return converted.ToString(CultureInfo.InvariantCulture);
                }
                catch
                {
                    return null;
                }
            }
        }
    }
}