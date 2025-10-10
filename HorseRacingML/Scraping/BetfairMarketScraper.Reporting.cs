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
using System.Security.Claims;
using System.Text.RegularExpressions;
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
                    var upcomingId = race.UpcomingRaceId?.ToString() ?? "n/a";
                    Console.WriteLine($"\t[DayReport] Race {raceName} is upcoming; recorded upcoming race id {upcomingId}. Using saved AI probabilities for reporting.");
                }

                foreach (var runner in race.Runners)
                {
                    if (runner == null)
                    {
                        continue;
                    }

                    var runnerIdentifier = !string.IsNullOrWhiteSpace(runner.HorseName)
                        ? runner.HorseName
                        : (!string.IsNullOrWhiteSpace(runner.SelectionId) ? runner.SelectionId : "unknown");

                    var historyDescription = runner.HistoricalRaceCount.HasValue
                        ? runner.HistoricalRaceCount.Value.ToString("N0", CultureInfo.InvariantCulture)
                        : "unknown";
                    Console.WriteLine($"\t[DayReport] {runnerIdentifier}: historical races used for AI features = {historyDescription}.");

                    if (!runner.AiProbability.HasValue || !double.IsFinite(runner.AiProbability.Value) || runner.AiProbability.Value <= 0)
                    {
                        runner.AiProbability = null;
                        runner.AiDecimalOdds = null;
                        runner.Differential = null;
                        runner.KellyFraction = null;
                        runner.SuggestedStake = null;
                        continue;
                    }

                    if (!runner.AiDecimalOdds.HasValue)
                    {
                        runner.AiDecimalOdds = BettingMath.CalculateAiDecimalOdds(runner.AiProbability.Value);
                    }

                    if (!runner.MarketProbability.HasValue && runner.MarketDecimalOdds.HasValue && runner.MarketDecimalOdds.Value > 0m)
                    {
                        runner.MarketProbability = 1.0 / (double)runner.MarketDecimalOdds.Value;
                    }

                    runner.Differential = runner.MarketProbability.HasValue
                        ? runner.AiProbability.Value - runner.MarketProbability.Value
                        : (double?)null;

                    if (runner.MarketDecimalOdds.HasValue && runner.MarketDecimalOdds.Value > 1m)
                    {
                        var kelly = BettingMath.CalculateKellyFraction(runner.AiProbability.Value, (double)runner.MarketDecimalOdds.Value, _maxKellyFraction);
                        runner.KellyFraction = kelly;
                        runner.SuggestedStake = (kelly > 0m && _bankroll > 0m)
                            ? CalculateStakeWithLimits(_bankroll, kelly)
                            : (decimal?)null;
                        if (runner.SuggestedStake <= 0m)
                        {
                            runner.SuggestedStake = null;
                        }
                    }
                    else
                    {
                        runner.KellyFraction = null;
                        runner.SuggestedStake = null;
                    }
                }

                var winner = race.Runners
                    .Where(r => r?.AiProbability.HasValue == true)
                    .OrderByDescending(r => r!.AiProbability!.Value)
                    .FirstOrDefault();

                if (winner == null)
                {
                    Console.WriteLine("\t[DayReport] No runners produced AI probabilities; skipping race output.");
                    continue;
                }

                var identifier = !string.IsNullOrWhiteSpace(winner.HorseName)
                    ? winner.HorseName
                    : (!string.IsNullOrWhiteSpace(winner.SelectionId) ? winner.SelectionId : "unknown");

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
                    var byMarket = _repo.GetUpcomingRaceByMarketId(marketId);
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
                    upcoming = _repo.FindUpcomingRace(raceDate.Value, raceTitle, venueName);
                    if (upcoming != null)
                    {
                        Console.WriteLine($"    Located UpcomingRaces row: UpcomingRaceId={upcoming.UpcomingRaceId}, MarketId={upcoming.MarketId ?? "<null>"}.");
                    }
                    else
                    {
                        Console.WriteLine($"    No UpcomingRaces row matched date/title/venue search (RaceDate, Title, VenueName).");
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
                    var byMarket = _repo.GetUpcomingRaceByMarketId(marketId);
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
                    upcoming = _repo.FindUpcomingRace(raceDate.Value, raceTitle, venueName);
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
                return FeatureLookup.Empty;
            }

            try
            {
                var prepared = _trainer.PrepareUpcomingRace(upcoming, flows);
                if (prepared == null || prepared.Rows.Count == 0)
                {
                    return FeatureLookup.Empty;
                }

                return FeatureLookup.FromPreparedRace(prepared);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"      Failed to build fallback synthetic feature vector for upcoming race {upcoming.MarketId ?? marketId ?? "<unknown>"}:{ex.Message}");
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
            private readonly Dictionary<string, Dictionary<string, object?>> _byHorse;
            private readonly Dictionary<string, Dictionary<string, object?>> _bySelectionId;

            private FeatureLookup(
                Dictionary<string, Dictionary<string, object?>> byHorse,
                Dictionary<string, Dictionary<string, object?>> bySelectionId)
            {
                _byHorse = byHorse;
                _bySelectionId = bySelectionId;
            }

            public static FeatureLookup Empty { get; } = new FeatureLookup(
                new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase),
                new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase));

            public static FeatureLookup FromPreparedRace(PreparedRace race)
            {
                if (race == null)
                    throw new ArgumentNullException(nameof(race));

                var byHorse = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);
                var bySelection = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);

                foreach (var row in race.Rows)
                {
                    var copy = new Dictionary<string, object?>(row, StringComparer.OrdinalIgnoreCase);

                    if (row.TryGetValue("HorseName", out var horseObj) && horseObj is string horse && !string.IsNullOrWhiteSpace(horse))
                    {
                        var normalized = NormalizeName(horse);
                        if (!string.IsNullOrEmpty(normalized) && !byHorse.ContainsKey(normalized))
                        {
                            byHorse[normalized] = copy;
                        }
                    }

                    var normalizedSelection = NormalizeSelectionId(row.TryGetValue("SelectionId", out var selectionObj)
                        ? selectionObj
                        : null);
                    if (!string.IsNullOrEmpty(normalizedSelection) && !bySelection.ContainsKey(normalizedSelection))
                    {
                        bySelection[normalizedSelection] = copy;
                    }
                }

                return new FeatureLookup(byHorse, bySelection);
            }

            public static FeatureLookup FromRows(IReadOnlyList<IDictionary<string, object?>> rows)
            {
                if (rows == null)
                    throw new ArgumentNullException(nameof(rows));

                if (rows.Count == 0)
                {
                    return Empty;
                }

                var byHorse = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);
                var bySelection = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);

                foreach (var row in rows)
                {
                    if (row == null)
                    {
                        continue;
                    }

                    var copy = row is Dictionary<string, object?> dict
                        ? new Dictionary<string, object?>(dict, StringComparer.OrdinalIgnoreCase)
                        : new Dictionary<string, object?>(row, StringComparer.OrdinalIgnoreCase);

                    if (copy.TryGetValue("HorseName", out var horseObj) && horseObj is string horse && !string.IsNullOrWhiteSpace(horse))
                    {
                        var normalized = NormalizeName(horse);
                        if (!string.IsNullOrEmpty(normalized) && !byHorse.ContainsKey(normalized))
                        {
                            byHorse[normalized] = copy;
                        }
                    }

                    var normalizedSelection = NormalizeSelectionId(copy.TryGetValue("SelectionId", out var selectionObj)
                        ? selectionObj
                        : null);
                    if (!string.IsNullOrEmpty(normalizedSelection) && !bySelection.ContainsKey(normalizedSelection))
                    {
                        bySelection[normalizedSelection] = copy;
                    }
                }

                return new FeatureLookup(byHorse, bySelection);
            }

            public Dictionary<string, object?>? FindBySelectionId(string? selectionId)
            {
                var normalized = NormalizeSelectionId(selectionId);
                if (string.IsNullOrEmpty(normalized))
                {
                    return null;
                }

                return _bySelectionId.TryGetValue(normalized, out var features) ? features : null;
            }

            private static string? NormalizeSelectionId(object? value)
            {
                if (value == null)
                {
                    return null;
                }

                if (value is string s)
                {
                    var trimmed = s.Trim();
                    return trimmed.Length == 0 ? null : trimmed;
                }

                var text = Convert.ToString(value, CultureInfo.InvariantCulture);
                if (string.IsNullOrWhiteSpace(text))
                {
                    return null;
                }

                var normalized = text.Trim();
                return normalized.Length == 0 ? null : normalized;
            }

            public Dictionary<string, object?>? FindByHorse(string? horseName)
            {
                if (string.IsNullOrWhiteSpace(horseName))
                {
                    return null;
                }

                var normalized = NormalizeName(horseName);
                if (string.IsNullOrEmpty(normalized))
                {
                    return null;
                }

                return _byHorse.TryGetValue(normalized, out var features) ? features : null;
            }
        }
    }
}