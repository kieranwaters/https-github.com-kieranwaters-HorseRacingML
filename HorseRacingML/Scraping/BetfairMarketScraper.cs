using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using OpenQA.Selenium;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text.RegularExpressions;
using System.Linq;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        private readonly IRacingRepository _repo;
        private readonly HyperparameterTrainer _trainer;
        private readonly object _repoLock = new();
        private readonly decimal _bankroll;
        private readonly decimal? _maxKellyFraction;
        private readonly bool _useMarketFallbackForAiDegeneracy;
        private readonly decimal _kellyDampener;
        private readonly MaxStakeMode _maxStakeMode;
        private readonly decimal? _maxStakePercentOfBankroll;
        private readonly decimal? _maxStakeFixedAmount;
        private readonly bool _computeAiProbabilities;
        private readonly Dictionary<RacePreparationKey, FeatureLookup> _featureLookupCache = new();
        private readonly Dictionary<RacePreparationKey, string?> _featureLookupErrorCache = new();
        private readonly object _featureLookupCacheLock = new();
        private readonly Dictionary<string, HandleScheduleMetadata> _handleScheduleCache = new(StringComparer.Ordinal);
        private readonly object _handleScheduleCacheLock = new();
        private readonly Dictionary<string, UpcomingRace?> _upcomingByMarketIdCache = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<UpcomingRaceLookupKey, UpcomingRace?> _upcomingByMetadataCache = new();
        private readonly object _upcomingCacheLock = new();
        private decimal _availableBankroll;
        private int _betSlipSelectionsFilled;
        private readonly HashSet<string> _missingScrapedFieldDescriptions = new(StringComparer.OrdinalIgnoreCase);
        private readonly System.Collections.Generic.IReadOnlyDictionary<string, string?>? _raceGoingLookup;
        private readonly string? _scheduleRegion;
        private readonly System.Collections.Generic.IReadOnlyDictionary<string, string?>? _raceGoingByVenueLookup;

        private const string MarketHeaderXPath = "/html/body/ui-view/div/div/div[2]/div/ui-view/div/div/div[1]/div[1]/div/bf-sports-header/div/div/div/div[1]/div/span[1]";
        private const string PlaceBetsButtonSelector = "#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > div > div > div.bf-col-xxl-7-24.bf-col-xl-8-24.bf-col-lg-8-24.bf-col-md-9-24.bf-col-sm-10-24.bf-col-10-24.right-side-column > div > div > bf-aside > div > div.bf-row.aside-top-row.no-bottom-gutter > div > betslip > div > bf-tabs > section > div:nth-child(2) > div > div > section > potentials > section > form > betslip-potentials-footer > footer > div.potentials-footer__actions > div > highlighted-button > ours-button > button";
        private const string ConfirmBetsButtonSelector = "#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > div > div > div.bf-col-xxl-7-24.bf-col-xl-8-24.bf-col-lg-8-24.bf-col-md-9-24.bf-col-sm-10-24.bf-col-10-24.right-side-column > div > div > bf-aside > div > div.bf-row.aside-top-row.no-bottom-gutter > div > betslip > div > bf-tabs > section > div:nth-child(2) > div > div > section > confirmation > section > betslip-confirmation-footer > footer > div.confirmation-footer__actions > highlighted-button > ours-button > button";
        private string? _lastFallbackFeatureError;
        private HyperparameterSummary? _loadedHyperparameters;

        public BetfairMarketScraper(
            IRacingRepository repo,
            HyperparameterTrainer trainer,
            decimal bankroll,
            decimal? maxKellyFraction = null,
            decimal? kellyDampener = null,
            bool useMarketFallbackForAiDegeneracy = true,
            MaxStakeMode maxStakeMode = MaxStakeMode.None,
            decimal? maxStakePercentOfBankroll = null,
            decimal? maxStakeFixedAmount = null,
            bool computeAiProbabilities = true,
            System.Collections.Generic.IReadOnlyDictionary<string, string?>? raceGoingLookup = null,
            string? scheduleRegion = null,
            System.Collections.Generic.IReadOnlyDictionary<string, string?>? raceGoingByVenueLookup = null)
        {
            _repo = repo ?? throw new ArgumentNullException(nameof(repo));
            _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
            _bankroll = bankroll;
            _availableBankroll = bankroll;
            _maxKellyFraction = maxKellyFraction;
            _betSlipSelectionsFilled = 0;
            _kellyDampener = (kellyDampener.HasValue && kellyDampener.Value > 0m)
                ? (kellyDampener.Value > 1m ? 1m : kellyDampener.Value)
                : 1m;
            _useMarketFallbackForAiDegeneracy = useMarketFallbackForAiDegeneracy;
            _maxStakeMode = maxStakeMode;
            _maxStakePercentOfBankroll = maxStakePercentOfBankroll;
            _maxStakeFixedAmount = maxStakeFixedAmount;
            _raceGoingLookup = raceGoingLookup;
            _raceGoingByVenueLookup = raceGoingByVenueLookup;
            _computeAiProbabilities = computeAiProbabilities;
            _scheduleRegion = string.IsNullOrWhiteSpace(scheduleRegion) ? null : scheduleRegion.Trim();
        }

        public BetfairMarketScraper(
            IRacingRepository repo,
            HyperparameterTrainer trainer,
            decimal bankroll,
            AutomationSettingsSnapshot settings,
            bool useMarketFallbackForAiDegeneracy = true,
            System.Collections.Generic.IReadOnlyDictionary<string, string?>? raceGoingLookup = null,
            bool computeAiProbabilities = true,
            string? scheduleRegion = null,
            System.Collections.Generic.IReadOnlyDictionary<string, string?>? raceGoingByVenueLookup = null)
            : this(
                repo,
                trainer,
                bankroll,
                settings?.MaxKellyFraction,
                settings?.KellyDampener,
                useMarketFallbackForAiDegeneracy,
                settings?.MaxStakeMode ?? MaxStakeMode.None,
                settings?.MaxStakePercentOfBankroll,
                settings?.MaxStakeFixedAmount,
                computeAiProbabilities,
               raceGoingLookup,
                scheduleRegion,
                raceGoingByVenueLookup)
        {
            if (settings == null)
            {
                throw new ArgumentNullException(nameof(settings));
            }
        }
        internal static string? NormalizeVenueName(string? venue)
        {
            if (string.IsNullOrWhiteSpace(venue))
            {
                return null;
            }

            var normalized = Regex.Replace(venue, "\\s+", " ").Trim();
            if (normalized.Length == 0)
            {
                return null;
            }

            normalized = Regex.Replace(normalized, @"\bgoing\b.*$", string.Empty, RegexOptions.IgnoreCase).Trim();
            normalized = Regex.Replace(normalized, @"[\-|,:]$", string.Empty).Trim();

            return normalized.Length == 0 ? null : normalized;
        }

        public HyperparameterSummary? LoadedHyperparameters => _loadedHyperparameters;
        private static void UpdateNeuralFeatureKeys(IReadOnlyList<string>? keys)
        {
            if (keys == null || keys.Count == 0)
            {
                _neuralFeatureKeys = Array.Empty<string>();
                return;
            }

            var normalized = keys
                .Where(key => !string.IsNullOrWhiteSpace(key))
                .Select(key => key.Trim())
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            _neuralFeatureKeys = normalized.Length > 0 ? normalized : Array.Empty<string>();
        }

        private static IReadOnlyList<string> GetCachedNeuralFeatureKeys() => _neuralFeatureKeys;
        public IReadOnlyCollection<string> MissingScrapeFieldDescriptions => _missingScrapedFieldDescriptions;

        public IReadOnlyList<RaceDayReport> ScrapeOpenRaceTabsForReport(IWebDriver driver, IEnumerable<string>? handlesToProcess = null)
        {
            var result = ScrapeOpenRaceTabsInternal(driver, executeBets: false, captureReport: true, handlesToProcess: handlesToProcess);
            var deduplicatedRaces = RaceDayReportDeduplicator.ByMarketId(result.Races);
            PopulateWinnerProbabilities(deduplicatedRaces);
            result.Races.Clear();
            result.Races.AddRange(deduplicatedRaces);
            return result.Races.AsReadOnly();
        }

        public IReadOnlyList<BetRecommendation> ScrapeOpenRaceTabs(
                    IWebDriver driver,
                    IEnumerable<string>? handlesToProcess = null)
        {
            var result = ScrapeOpenRaceTabsInternal(
                driver,
                executeBets: true,
                captureReport: false,
                handlesToProcess: handlesToProcess);
            return result.Recommendations.AsReadOnly();
        }
        private void SetFallbackFeatureError(string? message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                _lastFallbackFeatureError = null;
            }
            else
            {
                _lastFallbackFeatureError = message.Trim();
            }
        }

        private string BuildMissingHistoricalFeatureReason()
        {
            return string.IsNullOrWhiteSpace(_lastFallbackFeatureError)
                ? "Missing historical features; database coverage required."
                : $"Missing historical features; {_lastFallbackFeatureError}";
        }
        private sealed class HandleScheduleMetadata
        {
            public HandleScheduleMetadata(DateTime sortKey, string? url, DateTime capturedUtc)
            {
                SortKey = sortKey;
                Url = url;
                CapturedUtc = capturedUtc;
            }

            public DateTime SortKey { get; }
            public string? Url { get; }
            public DateTime CapturedUtc { get; }
        }

        private readonly struct UpcomingRaceLookupKey : IEquatable<UpcomingRaceLookupKey>
        {
            public UpcomingRaceLookupKey(DateTime raceDate, string titleKey, string venueKey)
            {
                RaceDate = raceDate;
                TitleKey = titleKey;
                VenueKey = venueKey;
            }

            public DateTime RaceDate { get; }
            public string TitleKey { get; }
            public string VenueKey { get; }

            public bool Equals(UpcomingRaceLookupKey other)
            {
                return RaceDate.Equals(other.RaceDate) &&
                       string.Equals(TitleKey, other.TitleKey, StringComparison.Ordinal) &&
                       string.Equals(VenueKey, other.VenueKey, StringComparison.Ordinal);
            }

            public override bool Equals(object? obj) => obj is UpcomingRaceLookupKey other && Equals(other);

            public override int GetHashCode()
            {
                unchecked
                {
                    var hash = RaceDate.GetHashCode();
                    hash = (hash * 397) ^ StringComparer.Ordinal.GetHashCode(TitleKey);
                    hash = (hash * 397) ^ StringComparer.Ordinal.GetHashCode(VenueKey);
                    return hash;
                }
            }
        }

        private void ResetUpcomingRaceLookupCache()
        {
            lock (_upcomingCacheLock)
            {
                _upcomingByMarketIdCache.Clear();
                _upcomingByMetadataCache.Clear();
            }
        }

        private void CacheUpcomingRace(UpcomingRace? upcoming)
        {
            if (upcoming == null)
            {
                return;
            }

            lock (_upcomingCacheLock)
            {
                CacheUpcomingRaceUnsafe(upcoming);
            }
        }

        private void CacheUpcomingRaceUnsafe(UpcomingRace upcoming)
        {
            if (!string.IsNullOrWhiteSpace(upcoming.MarketId))
            {
                _upcomingByMarketIdCache[upcoming.MarketId.Trim()] = upcoming;
            }

            if (upcoming.RaceDate != default)
            {
                var key = new UpcomingRaceLookupKey(
                    upcoming.RaceDate.Date,
                    RacingRepository.NormalizeLookupKey(upcoming.Title),
                    RacingRepository.NormalizeLookupKey(upcoming.VenueName));
                _upcomingByMetadataCache[key] = upcoming;
            }
        }

        private void CacheUpcomingRaceLookupResult(string? marketId, UpcomingRaceLookupKey? metadataKey, UpcomingRace? result)
        {
            lock (_upcomingCacheLock)
            {
                if (!string.IsNullOrWhiteSpace(marketId))
                {
                    _upcomingByMarketIdCache[marketId.Trim()] = result;
                }

                if (metadataKey.HasValue)
                {
                    _upcomingByMetadataCache[metadataKey.Value] = result;
                }

                if (result != null)
                {
                    CacheUpcomingRaceUnsafe(result);
                }
            }
        }

        private UpcomingRace? GetUpcomingRaceByMarketIdCached(string marketId)
        {
            var trimmed = marketId.Trim();

            lock (_upcomingCacheLock)
            {
                if (_upcomingByMarketIdCache.TryGetValue(trimmed, out var cached))
                {
                    return cached;
                }
            }

            UpcomingRace? result = null;
            try
            {
                result = _repo.GetUpcomingRaceByMarketId(trimmed);
            }
            catch
            {
                CacheUpcomingRaceLookupResult(trimmed, null, null);
                throw;
            }

            CacheUpcomingRaceLookupResult(trimmed, null, result);
            return result;
        }

        private UpcomingRace? FindUpcomingRaceCached(DateTime raceDate, string? raceTitle, string? venueName)
        {
            var key = new UpcomingRaceLookupKey(
                raceDate.Date,
                RacingRepository.NormalizeLookupKey(raceTitle),
                RacingRepository.NormalizeLookupKey(venueName));

            lock (_upcomingCacheLock)
            {
                if (_upcomingByMetadataCache.TryGetValue(key, out var cached))
                {
                    return cached;
                }
            }

            UpcomingRace? result = null;
            try
            {
                result = _repo.FindUpcomingRace(raceDate, raceTitle, venueName);
            }
            catch
            {
                CacheUpcomingRaceLookupResult(null, key, null);
                throw;
            }

            CacheUpcomingRaceLookupResult(null, key, result);
            return result;
        }


    }
}