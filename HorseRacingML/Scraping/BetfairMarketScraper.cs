using System;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Linq;
using System.Threading;
using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using HorseRacingML.Data;
using HorseRacingML.Models;
using System.Collections.Generic;
using HorseRacingML.ML;
using HorseRacingML.Services;
using System.IO;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

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
        private readonly Dictionary<RacePreparationKey, FeatureLookup> _featureLookupCache = new();
        private readonly object _featureLookupCacheLock = new();
        private decimal _availableBankroll;
        private int _betSlipSelectionsFilled;
        private const string MarketHeaderXPath = "/html/body/ui-view/div/div/div[2]/div/ui-view/div/div/div[1]/div[1]/div/bf-sports-header/div/div/div/div[1]/div/span[1]";
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
            decimal? maxStakeFixedAmount = null)
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
        }

        public BetfairMarketScraper(
            IRacingRepository repo,
            HyperparameterTrainer trainer,
            decimal bankroll,
            AutomationSettingsSnapshot settings,
            bool useMarketFallbackForAiDegeneracy = true)
            : this(
                repo,
                trainer,
                bankroll,
                settings?.MaxKellyFraction,
                settings?.KellyDampener,
                useMarketFallbackForAiDegeneracy,
                settings?.MaxStakeMode ?? MaxStakeMode.None,
                settings?.MaxStakePercentOfBankroll,
                settings?.MaxStakeFixedAmount)
        {
            if (settings == null)
            {
                throw new ArgumentNullException(nameof(settings));
            }
        }
        public HyperparameterSummary? LoadedHyperparameters => _loadedHyperparameters;
        public IReadOnlyList<RaceDayReport> ScrapeOpenRaceTabsForReport(IWebDriver driver, IEnumerable<string>? handlesToProcess = null)
        {
            var result = ScrapeOpenRaceTabsInternal(driver, executeBets: false, captureReport: true, handlesToProcess: handlesToProcess);
            PopulateWinnerProbabilities(result.Races);
            return result.Races;
        }
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
                    Console.WriteLine($"	[DayReport] Using upcoming race {upcomingRecord.UpcomingRaceId} (market {upcomingRecord.MarketId}) for probability alignment.");
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
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows,
            IReadOnlyList<IDictionary<string, object?>>? preparedRows,
            UpcomingRace? persistedUpcoming)
        {
            if (preparedRows != null && preparedRows.Count > 0)
            {
                Console.WriteLine(
                    $"	Using {preparedRows.Count} pre-provided feature row(s) for lookup.");
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
                    Console.WriteLine($"	Persisted upcoming race metadata for market {persistedUpcoming.MarketId ?? marketId ?? "<unknown>"} rejected due to metadata misalignment with scraped race (Title/VenueName).");
                }
            }

            if (upcoming == null)
            {
                try
                {
                    upcoming = _repo.FindUpcomingRace(raceDate.Value, raceTitle, venueName);
                    if (upcoming != null)
                    {
                        Console.WriteLine($"	Located UpcomingRaces row: UpcomingRaceId={upcoming.UpcomingRaceId}, MarketId={upcoming.MarketId ?? "<null>"}.");
                    }
                    else
                    {
                        Console.WriteLine($"	No UpcomingRaces row matched date/title/venue search (RaceDate, Title, VenueName).");
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"	Failed to query upcoming race metadata: {ex.Message}");
                }

                if (upcoming != null &&
                    !ShouldUseUpcomingCandidate(upcoming, marketId, raceTitle, venueName))
                {
                    Console.WriteLine($"	UpcomingRaces row {upcoming.UpcomingRaceId} rejected due to metadata misalignment with scraped race (Title/VenueName).");
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
                    backBookPercentage,
                    layBookPercentage,
                    marketId,
                    flows);
                if (upcoming == null)
                {
                    Console.WriteLine("	Unable to create synthetic upcoming race metadata; feature lookup aborted.");
                    return FeatureLookup.Empty;
                }

                Console.WriteLine("	Constructed synthetic upcoming race metadata for feature synthesis.");
            }
            else
            {
                if (persistedAccepted)
                {
                    Console.WriteLine($"	Using persisted upcoming race metadata for market {upcoming.MarketId ?? marketId ?? "<unknown>"}.");
                }
                else
                {
                    Console.WriteLine($"	Using UpcomingRaces metadata from repository for market {upcoming.MarketId ?? marketId ?? "<unknown>"}.");
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
                    Console.WriteLine(
                        $"  Using cached synthetic feature rows for upcoming race {upcoming.MarketId ?? marketId ?? "<unknown"}.");
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

                Console.WriteLine(
                    $"  Loaded synthetic feature rows for upcoming race {upcoming.MarketId}; runner count={prepared.Rows.Count}.");
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
                    Going = metadata.Going,
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
        private BetfairScrapeResult ScrapeOpenRaceTabsInternal(IWebDriver driver, bool executeBets, bool captureReport, IEnumerable<string>? handlesToProcess = null)
        {
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10)); // short explicit wait
            var handles = handlesToProcess?.ToList() ?? driver.WindowHandles.ToList(); // collect tab handles
            var weightPath = ResolveAiWeightPath(); // resolve AI weights path
            _loadedHyperparameters = null;

            if (File.Exists(weightPath))
            {
                var info = new FileInfo(weightPath); // file info for logging
                Console.WriteLine($"\tAI weight file located at {weightPath} ({info.Length} bytes, last modified {info.LastWriteTimeUtc:u})."); // status
            }
            else
            {
                Console.WriteLine($"\tAI weight file missing at {weightPath}; AI probabilities may fall back to defaults."); // warn missing weights
            }

            var aiCalculator = new AIOddsCalculator(weightPath); // init AI calc
            Console.WriteLine($"\tAI model status: {aiCalculator.ModelStatus}"); // log model status
            _loadedHyperparameters = aiCalculator.Hyperparameters;
            lock (_featureLookupCacheLock)
            {
                _featureLookupCache.Clear();
            }
            var result = new BetfairScrapeResult(); // aggregate result
            var recommendations = new List<BetRecommendation>(); // all bet recs

            foreach (var handle in handles)
            {
                driver.SwitchTo().Window(handle); // switch tab
                Console.WriteLine($"Processing tab: {driver.Url}"); // log url
                var raceUrl = driver.Url;
                if (!driver.Url.Contains("/horse-racing/", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("\tSkipping non-racing tab"); // skip non-racing
                    continue; // next tab
                }

                try
                {
                    wait.Until(d => d.FindElements(By.CssSelector(".runner-line")).Count > 0); // wait for runner rows
                }
                catch (WebDriverTimeoutException)
                {
                    Console.Error.WriteLine("\tTimed out waiting for runner rows"); // timeout
                    continue; // next tab
                }

                var marketId = ExtractMarketId(driver.Url); // parse market id
                if (string.IsNullOrEmpty(marketId))
                {
                    Console.Error.WriteLine($"\tFailed to extract market ID from URL: {driver.Url}"); // log failure
                    continue; // next tab
                }

                var title = ReadFirstNonEmptyText(driver,
                    "[data-testid='marketTitle']",
                    "[data-testid='market-title']",
                    "[data-testid='eventTitle']",
                    "h1[data-testid='marketTitle']",
                    "h1[data-testid='eventTitle']",
                    ".market-title",
                    "header h1",
                    ".page-title h1"); // market title with fallbacks
                if (string.IsNullOrWhiteSpace(title))
                {
                    var xpathTitle = TextOrEmpty(driver, By.XPath(MarketHeaderXPath));
                    if (!string.IsNullOrWhiteSpace(xpathTitle))
                    {
                        Console.WriteLine("\tResolved market title using dedicated header XPath.");
                        title = xpathTitle;
                    }
                }
                if (string.IsNullOrWhiteSpace(title))
                {
                    title = ExtractDocumentTitle(driver);
                    if (!string.IsNullOrWhiteSpace(title))
                    {
                        Console.WriteLine("\tResolved market title using document title fallback.");
                    }
                }
                var normalizedTitle = NormalizeMarketTitle(title);
                if (!string.IsNullOrWhiteSpace(normalizedTitle))
                {
                    if (!string.IsNullOrWhiteSpace(title) && !string.Equals(normalizedTitle, title.Trim(), StringComparison.Ordinal))
                    {
                        Console.WriteLine($"\tNormalized market title to '{normalizedTitle}'.");
                    }

                    title = normalizedTitle;
                }
                else if (!string.IsNullOrWhiteSpace(title))
                {
                    title = title.Trim();
                }
                var venueText = TextOrEmpty(driver, By.CssSelector(".venue-name")); // venue text
                var eventDateText = TextOrEmpty(driver, By.CssSelector(".event-date")); // event date raw
                var raceDetailsText = ReadFirstNonEmptyText(driver,
                    ".market-name",
                    "[data-testid='marketName']",
                    "[data-testid='market-name']",
                    "[data-testid='marketDescription']"); // race details
                var offTimeText = TextOrEmpty(driver, By.CssSelector("[data-testid='startTime']")); // off time raw
                var backBookText = TextOrEmpty(driver, By.CssSelector(".rh-back-book-percentage-label")); // back book %
                var layBookText = TextOrEmpty(driver, By.CssSelector(".rh-lay-book-percentage-label")); // lay book %

                TimeSpan? offTime = TimeSpan.TryParse(offTimeText, out var t) ? t : (TimeSpan?)null; // parse off time
                var (venueTime, venueName, venueCountry) = ParseVenueDetails(venueText); // parse venue details
                if (!offTime.HasValue && venueTime.HasValue) { offTime = venueTime; } // fallback to venue time

                var parsedRaceDate = ParseEventDate(eventDateText, DateTime.Today); // parse date
                var backBookPercentage = ParsePercentage(backBookText); // parse %
                var layBookPercentage = ParsePercentage(layBookText); // parse %

                Console.WriteLine($"\tScraping market {marketId} - {title}"); // progress

                try
                {
                    var screen = new RaceScreen
                    {
                        MarketId = marketId, // ids
                        OffTime = offTime, // off time
                        Title = title, // title
                        RaceDate = parsedRaceDate ?? DateTime.Today, // date
                        VenueName = venueName, // venue
                        VenueCountry = venueCountry, // country
                        EventDateText = string.IsNullOrWhiteSpace(eventDateText) ? null : eventDateText.Trim(), // raw text
                        RaceDetails = string.IsNullOrWhiteSpace(raceDetailsText) ? null : raceDetailsText.Trim(), // details
                        BackBookPercentage = backBookPercentage, // %
                        LayBookPercentage = layBookPercentage, // %
                        RaceUrl = string.IsNullOrWhiteSpace(raceUrl) ? null : raceUrl.Trim() // url
                    }; // init screen row

                    lock (_repoLock) { _repo.InsertRaceScreen(screen); } // persist
                    Console.WriteLine($"\tInserted race screen for {marketId}"); // log ok
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tInsertRaceScreen failed for market {marketId}: {ex.Message}"); // log error
                    continue; // next tab
                }

                var rows = driver.FindElements(By.CssSelector(".runner-line")); // current runner rows
                if (rows.Count == 0)
                {
                    Console.Error.WriteLine($"\tNo runner rows found for market {marketId}"); // guard
                    continue; // next tab
                }

                Console.WriteLine($"\tFound {rows.Count} runners for market {marketId}"); // log count
                UpcomingRace? persistedUpcoming = null;
                if (parsedRaceDate.HasValue)
                {
                    try
                    {
                        var metadataSource = new RaceDayReport
                        {
                            RaceTitle = title,
                            RaceDetails = raceDetailsText
                        };
                        var metadata = ParseRaceMetadata(metadataSource);

                        short? distanceYards = metadata.DistanceYards > 0
                            ? (short)Math.Min(metadata.DistanceYards, short.MaxValue)
                            : null;

                        byte? runnerCount = rows.Count > 0
                            ? (byte)Math.Min(rows.Count, byte.MaxValue)
                            : (byte?)null;

                        persistedUpcoming = new UpcomingRace
                        {
                            MarketId = marketId,
                            RaceDate = parsedRaceDate.Value.Date,
                            ScheduledOff = offTime,
                            VenueName = string.IsNullOrWhiteSpace(venueName) ? null : venueName.Trim(),
                            VenueCountry = string.IsNullOrWhiteSpace(venueCountry) ? null : venueCountry.Trim(),
                            Title = string.IsNullOrWhiteSpace(title) ? null : title.Trim(),
                            RaceDetails = string.IsNullOrWhiteSpace(raceDetailsText) ? null : raceDetailsText.Trim(),
                            RaceType = metadata.RaceType,
                            Class = metadata.Class,
                            AgeRestriction = metadata.AgeRestriction,
                            Surface = metadata.Surface,
                            Going = metadata.Going,
                            DistanceYards = distanceYards,
                            DistanceText = metadata.DistanceText,
                            RunnerCount = runnerCount,
                            BackBookPercentage = backBookPercentage,
                            LayBookPercentage = layBookPercentage
                        };

                        lock (_repoLock)
                        {
                            var upcomingId = _repo.UpsertUpcomingRace(persistedUpcoming);
                            persistedUpcoming.UpcomingRaceId = upcomingId;
                        }

                        Console.WriteLine($"\tRecorded upcoming race {persistedUpcoming.UpcomingRaceId} for market {marketId}.");
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"\tFailed to record upcoming race metadata for market {marketId}: {ex.Message}");
                        persistedUpcoming = null;
                    }
                }
                var flows = new List<RunnerFlow>(); // collect runner flows
                var runnerEntries = new List<(IWebElement Row, RunnerFlow Flow)>(); // row→flow mapping
                var selectionIdAttributes = new[] { "data-selection-id", "data-selection-key", "data-selection-uid", "data-selectionid", "data-runner-id" }; // possible id attrs

                foreach (var row in rows)
                {
                    string? selectionId = null; // selection id holder

                    foreach (var attribute in selectionIdAttributes)
                    {
                        selectionId = row.GetAttribute(attribute); // read attr
                        if (!string.IsNullOrEmpty(selectionId)) { break; } // found
                    }

                    if (string.IsNullOrEmpty(selectionId))
                    {
                        foreach (var attribute in selectionIdAttributes)
                        {
                            try
                            {
                                var childWithId = row.FindElement(By.CssSelector($"[{attribute}]")); // search child
                                selectionId = childWithId.GetAttribute(attribute); // read id
                                if (!string.IsNullOrEmpty(selectionId)) { break; } // stop if found
                            }
                            catch (NoSuchElementException)
                            {
                                selectionId = null; // ignore
                            }
                        }
                    }

                    var selectionIdMissing = string.IsNullOrWhiteSpace(selectionId); // missing id?
                    if (selectionIdMissing) { selectionId = null; } // normalize

                    var js = (IJavaScriptExecutor)driver; // cast to JS
                    const string runnerExtractionScript = @"
        const row = arguments[0];
        const textOrEmpty = el => el && el.textContent ? el.textContent.trim() : '';
        const queryText = selector => selector ? textOrEmpty(row.querySelector(selector)) : '';
        const extractPriceText = raw => { if (!raw) { return ''; } const text = raw.trim(); if (!text) { return ''; } if (/^[£€$]/.test(text)) { return ''; } return text; };
        const extractPriceFromButton = raw => { if (!raw) { return ''; } const text = raw.trim(); if (!text) { return ''; } const tokens = text.split(/\s+/); for (const token of tokens) { if (!token || /^[£€$]/.test(token)) { continue; } if (/^[0-9]+(\.[0-9]+)?$/.test(token)) { return token; } } return extractPriceText(text); };
        const indexTokens = { 1: ['1','one'], 2: ['2','two'], 3: ['3','three'] };
        const findOursPriceButton = (type, index) => {
            const cellIndex = type === 'back' ? 4 : 5;
            const cell = row.querySelector(`td:nth-of-type(${cellIndex})`);
            if (cell) {
                const oursButtons = Array.from(cell.querySelectorAll('ours-price-button'));
                if (oursButtons.length >= index) {
                    const button = oursButtons[index - 1];
                    const priceLabel = button.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price');
                    if (priceLabel) { const text = extractPriceText(textOrEmpty(priceLabel)); if (text) { return text; } }
                    const fallbackButtonText = extractPriceFromButton(textOrEmpty(button.querySelector('button')));
                    if (fallbackButtonText) { return fallbackButtonText; }
                }
            }
            const allButtons = Array.from(row.querySelectorAll('ours-price-button'));
            const startIndex = type === 'lay' ? 3 : 0;
            if (allButtons.length >= index + startIndex) {
                const button = allButtons[startIndex + index - 1];
                const priceLabel = button.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price');
                if (priceLabel) { const text = extractPriceText(textOrEmpty(priceLabel)); if (text) { return text; } }
                const fallbackButtonText = extractPriceFromButton(textOrEmpty(button.querySelector('button')));
                if (fallbackButtonText) { return fallbackButtonText; }
            }
            return '';
        };
        const findPrice = (type, index) => {
            const tokens = indexTokens[index] || [String(index)];
            const selectors = [];
            for (const token of tokens) {
                selectors.push(`.bet-button.${type}-selection-button.${type}-${token} .bet-button-price`);
                selectors.push(`.bet-button.price-button.${type}-${token} .bet-button-price`);
                selectors.push(`.bet-button.${type}-cell.${type}-${token} .bet-button-price`);
                selectors.push(`.bet-buttons-${type}-cell.${type}-${token} .bet-button-price`);
                selectors.push(`.${type}-cell.${type}-${token} .bet-button-price`);
                selectors.push(`.${type}-${token} .bet-button-price`);
            }
            const oursButtonPrice = findOursPriceButton(type, index);
            if (oursButtonPrice) { return oursButtonPrice; }
            const dataTestId = row.querySelector(`[data-testid='runner-${type}-${index}-price']`);
            if (dataTestId) { const text = textOrEmpty(dataTestId); if (text) { return text; } }
            for (const selector of selectors) {
                const el = row.querySelector(selector);
                if (el) { const text = textOrEmpty(el); if (text) { return text; } }
            }
            return '';
        };
        const result = { cloth: queryText('.runner-number'), draw: queryText('.draw'), horse: queryText('.name .runner-name'), jockey: queryText('.name .jockey-name') };
        for (let i = 1; i <= 3; i++) { result[`back${i}`] = findPrice('back', i); result[`lay${i}`] = findPrice('lay', i); }
        const fallbackButtons = Array.from(row.querySelectorAll('bet-button'));
        const fallbackPrices = fallbackButtons.map(btn => { const label = btn.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price'); if (label) { const priceText = extractPriceText(textOrEmpty(label)); if (priceText) { return priceText; } } const buttonText = btn.querySelector('button'); return extractPriceFromButton(textOrEmpty(buttonText || btn)); });
        for (let i = 1; i <= 3; i++) {
            const backKey = `back${i}`; const layKey = `lay${i}`;
            if (!result[backKey] && fallbackPrices.length >= i) { result[backKey] = fallbackPrices[i - 1]; }
            if (!result[layKey] && fallbackPrices.length >= i + 3) { result[layKey] = fallbackPrices[i + 2]; }
        }
        return result;
    ";

                    var elementData = (IDictionary<string, object>)js.ExecuteScript(runnerExtractionScript, row); // execute script
                    string Get(string key) => elementData.TryGetValue(key, out var v) ? v?.ToString() ?? string.Empty : string.Empty; // helper

                    var runnerFlow = new RunnerFlow // create flow
                    {
                        MarketId = marketId, // market id
                        SelectionId = selectionId, // selection id
                        ClothNumber = TryParseByte(Get("cloth")), // cloth
                        Draw = TryParseByte(Get("draw")), // draw
                        HorseName = Get("horse"), // horse
                        JockeyName = Get("jockey"), // jockey
                        BackPrice1 = ParseDecimal(Get("back1")), // b1
                        BackPrice2 = ParseDecimal(Get("back2")), // b2
                        BackPrice3 = ParseDecimal(Get("back3")), // b3
                        LayPrice1 = ParseDecimal(Get("lay1")), // l1
                        LayPrice2 = ParseDecimal(Get("lay2")), // l2
                        LayPrice3 = ParseDecimal(Get("lay3")) // l3
                    };

                    if (selectionIdMissing && string.IsNullOrWhiteSpace(runnerFlow.HorseName))
                    {
                        Console.Error.WriteLine($"\tUnable to determine selection id or horse name for a runner in market {marketId}; skipping row."); // cannot ID
                        continue; // next row
                    }

                    if (selectionIdMissing)
                    {
                        var fallbackIdentifier = !string.IsNullOrWhiteSpace(runnerFlow.HorseName)
                            ? runnerFlow.HorseName!.Trim()
                            : runnerFlow.ClothNumber.HasValue
                                ? $"cloth #{runnerFlow.ClothNumber.Value.ToString(CultureInfo.InvariantCulture)}"
                                : "unknown runner";
                        Console.WriteLine($"\tRunner {fallbackIdentifier} in market {marketId} missing selection id; relying on horse name for identification."); // info
                    }

                    flows.Add(runnerFlow); // add to list
                    runnerEntries.Add((row, runnerFlow)); // keep mapping
                }

                PopulateFeatureVectors(
                    parsedRaceDate,
                    title,
                    venueName,
                    venueCountry,
                    offTime,
                    raceDetailsText,
                    backBookPercentage,
                    layBookPercentage,
                    marketId,
                    flows,
                    rows.Count,
                    preparedRows: null,
                    persistedUpcoming: persistedUpcoming); // build features

                foreach (var rf in flows) // process each runner
                {
                    if (rf.FeatureValues != null && !rf.FeatureValues.ContainsKey("RunnerCount") && rows.Count > 0)
                    {
                        rf.FeatureValues["RunnerCount"] = rows.Count; // ensure runner count
                    }
                    var rfIdentifier = !string.IsNullOrWhiteSpace(rf.HorseName)
                        ? rf.HorseName!
                        : (rf.SelectionId ?? "unknown");
                    if (rf.FeatureValues != null)
                    {
                        if (rf.BackPrice1.HasValue) { rf.FeatureValues["BackPrice1"] = rf.BackPrice1.Value; } // copy b1
                        if (rf.BackPrice2.HasValue) { rf.FeatureValues["BackPrice2"] = rf.BackPrice2.Value; } // copy b2
                        if (rf.BackPrice3.HasValue) { rf.FeatureValues["BackPrice3"] = rf.BackPrice3.Value; } // copy b3
                        if (rf.LayPrice1.HasValue) { rf.FeatureValues["LayPrice1"] = rf.LayPrice1.Value; } // copy l1
                        if (rf.LayPrice2.HasValue) { rf.FeatureValues["LayPrice2"] = rf.LayPrice2.Value; } // copy l2
                        if (rf.LayPrice3.HasValue) { rf.FeatureValues["LayPrice3"] = rf.LayPrice3.Value; } // copy l3
                    }

                    try
                    {
                        var probability = aiCalculator.CalculateOdds(rf); // compute AI odds
                        if (double.IsFinite(probability) && probability > 0 && probability <= 1)
                        {
                            rf.AiOdds = probability;
                            rf.AiProbabilityMarketDerived = false;
                        }
                        else
                        {
                            rf.AiOdds = null;
                            rf.AiProbabilityMarketDerived = false;
                            var probabilityText = double.IsFinite(probability)
                                ? probability.ToString("0.####", CultureInfo.InvariantCulture)
                                : "non-finite";
                            Console.WriteLine($"\t\tDiscarding non-positive AI probability {probabilityText} for {rfIdentifier}; treating as missing.");
                        }
                    }
                    catch (Exception ex)
                    {
                        rf.AiOdds = null; // set null on fail
                        rf.AiProbabilityMarketDerived = false;
                        Console.Error.WriteLine($"\tFailed to calculate AI odds for selection {rfIdentifier} in market {marketId}: {ex.Message}"); // log
                    }
                    string aiText;
                    if (rf.AiOdds.HasValue)
                    {
                        var rawProbability = rf.AiOdds.Value;
                        aiText = rawProbability.ToString("0.####", CultureInfo.InvariantCulture);
                        if (rawProbability > 0 && rawProbability < 1e-3)
                        {
                            var scientific = rawProbability.ToString("0.###E+0", CultureInfo.InvariantCulture);
                            aiText = $"{aiText} (~{scientific})";
                        }
                    }
                    else
                    {
                        aiText = "null";
                    }
                    var backText = rf.BackPrice1.HasValue ? rf.BackPrice1.Value.ToString("0.##", CultureInfo.InvariantCulture) : "null"; // back text
                    Console.WriteLine($"\tRunner snapshot {rfIdentifier}: back1={backText}, aiProbabilityRaw={aiText}"); // per-runner log

                    if (!rf.AiOdds.HasValue) { Console.WriteLine($"\t\tAI probability missing for {rfIdentifier}; downstream filters will treat this runner as zero edge."); } // warn missing ai
                    if (!rf.BackPrice1.HasValue) { Console.WriteLine($"\t\tNo back price available for {rfIdentifier}; cannot compare against market probability."); } // warn missing price
                }

                var validAiBefore = flows.Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0).Select(f => f.AiOdds!.Value).ToList(); // gather valid ai
                var missingAiCount = flows.Count - validAiBefore.Count; // count missing

                if (validAiBefore.Count == 0)
                {
                    Console.WriteLine($"\tAll {flows.Count} runner(s) in market {marketId} are missing AI probabilities before normalization."); // summary
                }
                else
                {
                    var sumProb = validAiBefore.Sum(); // sum
                    var minProb = validAiBefore.Min(); // min
                    var maxProb = validAiBefore.Max(); // max
                    Console.WriteLine($"\tAI probability summary before normalization for market {marketId}: valid={validAiBefore.Count}, missing={missingAiCount}, sum={sumProb.ToString("0.####", CultureInfo.InvariantCulture)}, min={minProb.ToString("0.####", CultureInfo.InvariantCulture)}, max={maxProb.ToString("0.####", CultureInfo.InvariantCulture)}"); // log
                }

                NormalizeAiOdds(flows, _useMarketFallbackForAiDegeneracy); // normalize

                if (captureReport)
                {
                    var report = BuildRaceReport(
                        marketId,
                        title,
                        venueName,
                        venueCountry,
                        parsedRaceDate,
                        offTime,
                        string.IsNullOrWhiteSpace(raceDetailsText) ? null : raceDetailsText.Trim(),
                        backBookPercentage,
                        layBookPercentage,
                         raceUrl,
                        flows
                    ); // build report

                    result.Races.Add(report); // collect report
                }

                var raceRecommendations = CreateRecommendations(flows, marketId, title, venueName, parsedRaceDate, executeBets)
                    .OrderByDescending(r => r.Differential)
                    .ThenByDescending(r => r.KellyFraction)
                    .ToList(); // rank recs

                Console.WriteLine($"\t{raceRecommendations.Count} runner(s) passed value filters for market {marketId}."); // log count

                if (!executeBets)
                {
                    if (raceRecommendations.Count > 0) { Console.WriteLine("\tReport mode: positive expected value runner(s) identified; skipping bet execution."); } else { Console.WriteLine($"\tNo positive value opportunity identified for market {marketId}"); } // report only
                }
                else
                {
                    if (raceRecommendations.Count > 1) { Console.WriteLine("\t\tMultiple runners qualified in the same market; sequential Kelly stakes will size each independently in tab order."); } // info

                    if (raceRecommendations.Count > 0)
                    {
                        var bankrollBeforeClicks = _availableBankroll; // snapshot bankroll
                        var clickedRecommendations = ExecuteBackAllClicks(driver, runnerEntries, raceRecommendations); // click bets

                        if (clickedRecommendations.Count > 0)
                        {
                            var top = clickedRecommendations.First(); // first rec
                            Console.WriteLine($"\tKelly stake {top.Stake.ToString("0.##", CultureInfo.InvariantCulture)} on {top.HorseName ?? "unknown"} (diff {top.Differential.ToString("0.####", CultureInfo.InvariantCulture)})"); // log stake
                            recommendations.AddRange(clickedRecommendations); // keep all
                            PopulateBetSlipStakes(driver, clickedRecommendations, bankrollBeforeClicks); // fill stakes
                        }
                        else
                        {
                            Console.WriteLine("\tNo qualifying Back-All clicks were executed for this market"); // none clicked
                        }
                    }
                    else
                    {
                        Console.WriteLine($"\tNo positive value opportunity identified for market {marketId}"); // no value
                    }
                }

                try
                {
                    lock (_repoLock)
                    {
                        _repo.InsertRunnerFlows(flows);
                    }

                    foreach (var inner in flows)
                    {
                        var innerIdentifier = !string.IsNullOrWhiteSpace(inner.HorseName)
                            ? inner.HorseName!.Trim()
                            : (!string.IsNullOrWhiteSpace(inner.SelectionId) ? inner.SelectionId! : "unknown");
                        Console.WriteLine($"\tInserted runner {innerIdentifier} for market {marketId}"); // log ok
                    }
                }
                catch (Exception ex)
                {
                    foreach (var inner in flows)
                    {
                        var innerIdentifier = !string.IsNullOrWhiteSpace(inner.HorseName)
                            ? inner.HorseName!.Trim()
                            : (inner.SelectionId ?? "unknown");
                        Console.Error.WriteLine($"\tInsertRunnerFlow failed for market {marketId}, runner {innerIdentifier}: {ex.Message}"); // log error
                    }
                }
            }

            result.Recommendations.AddRange(recommendations.OrderByDescending(r => r.Differential).ThenByDescending(r => r.KellyFraction)); // finalize ordering
            return result; // done
        }

        private RaceDayReport BuildRaceReport(
            string marketId,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            DateTime? raceDate,
            TimeSpan? offTime,
            string? raceDetails,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? raceUrl,
            IEnumerable<RunnerFlow> flows)
        {
            var report = new RaceDayReport
            {
                MarketId = marketId,
                RaceTitle = string.IsNullOrWhiteSpace(raceTitle) ? null : raceTitle.Trim(),
                VenueName = string.IsNullOrWhiteSpace(venueName) ? null : venueName.Trim(),
                VenueCountry = string.IsNullOrWhiteSpace(venueCountry) ? null : venueCountry.Trim(),
                RaceDate = raceDate,
                OffTime = offTime,
                RaceDetails = string.IsNullOrWhiteSpace(raceDetails) ? null : raceDetails.Trim(),
                BackBookPercentage = backBookPercentage,
                LayBookPercentage = layBookPercentage,
                RaceUrl = string.IsNullOrWhiteSpace(raceUrl) ? null : raceUrl.Trim()
            };

            var runnerList = flows as IList<RunnerFlow> ?? flows.ToList();

            IReadOnlyDictionary<string, int>? prefetchedCounts = null;
            try
            {
                var missingNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var flow in runnerList)
                {
                    if (flow == null)
                    {
                        continue;
                    }

                    if (flow.HistoricalRaceCount.HasValue)
                    {
                        continue;
                    }

                    if (flow.FeatureValues != null &&
                        flow.FeatureValues.TryGetValue("CareerStarts", out var existingValue) &&
                        TryConvertToInt32(existingValue).HasValue)
                    {
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(flow.HorseName))
                    {
                        missingNames.Add(flow.HorseName);
                    }
                }

                if (missingNames.Count > 0)
                {
                    prefetchedCounts = _repo.GetHistoricalRaceCountsByHorseNames(missingNames);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tFailed to prefetch historical race counts: {ex.Message}");
            }

            foreach (var flow in runnerList)
            {
                report.Runners.Add(CreateRunnerReport(flow, prefetchedCounts));
            }

            return report;
        }


        private RunnerDayReport CreateRunnerReport(
            RunnerFlow flow,
            IReadOnlyDictionary<string, int>? prefetchedCounts)
        {
            var runner = new RunnerDayReport
            {
                SelectionId = flow.SelectionId,
                ClothNumber = flow.ClothNumber,
                Draw = flow.Draw,
                HorseName = flow.HorseName,
                JockeyName = flow.JockeyName,
                FeatureValues = flow.FeatureValues != null
                    ? new Dictionary<string, object?>(flow.FeatureValues, StringComparer.OrdinalIgnoreCase)
                    : new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            };

            if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 0m)
            {
                runner.MarketDecimalOdds = flow.BackPrice1.Value;
                if (flow.BackPrice1.Value > 1m)
                {
                    runner.MarketProbability = 1.0 / (double)flow.BackPrice1.Value;
                }
            }
            if (flow.LayPrice1.HasValue && flow.LayPrice1.Value > 0m)
            {
                runner.LayDecimalOdds = flow.LayPrice1.Value;
            }
            if (flow.AiOdds.HasValue && double.IsFinite(flow.AiOdds.Value) && flow.AiOdds.Value > 0)
            {
                runner.AiProbability = flow.AiOdds.Value;
                runner.AiDecimalOdds = BettingMath.CalculateAiDecimalOdds(flow.AiOdds.Value);
            }
            runner.AiProbabilityMarketDerived = flow.AiProbabilityMarketDerived;
            if (runner.AiProbability.HasValue && runner.MarketProbability.HasValue)
            {
                runner.Differential = runner.AiProbability.Value - runner.MarketProbability.Value;
            }

            if (runner.AiProbability.HasValue && runner.MarketDecimalOdds.HasValue && runner.MarketDecimalOdds.Value > 1m)
            {
                var kelly = BettingMath.CalculateKellyFraction(
                    runner.AiProbability.Value,
                    (double)runner.MarketDecimalOdds.Value,
                    _maxKellyFraction);
                runner.KellyFraction = kelly;
                if (kelly > 0m && _bankroll > 0m)
                {
                    var stake = CalculateStakeWithLimits(_bankroll, kelly);
                    runner.SuggestedStake = stake > 0m ? stake : null;
                }
            }
            if (runner.AiProbability.HasValue && runner.LayDecimalOdds.HasValue && runner.LayDecimalOdds.Value > 1m)
            {
                var layKelly = BettingMath.CalculateLayKellyFraction(
                    runner.AiProbability.Value,
                    (double)runner.LayDecimalOdds.Value,
                    _maxKellyFraction);

                if (layKelly > 0m)
                {
                    runner.LayKellyFraction = layKelly;

                    if (_bankroll > 0m)
                    {
                        var layStake = BettingMath.CalculateLayStake(_bankroll, layKelly, runner.LayDecimalOdds.Value);
                        if (layStake > 0m)
                        {
                            runner.LaySuggestedStake = layStake;
                        }
                    }
                }
            }
            if (flow.HistoricalRaceCount.HasValue)
            {
                runner.HistoricalRaceCount = flow.HistoricalRaceCount;
            }
            else if (flow.FeatureValues != null &&
                     flow.FeatureValues.TryGetValue("CareerStarts", out var historyValue))
            {
                runner.HistoricalRaceCount = TryConvertToInt32(historyValue);
                flow.HistoricalRaceCount = runner.HistoricalRaceCount;
            }
            else
            {
                var resolvedHistoryCount = ResolveHistoricalRaceCountFromPrefetch(flow, prefetchedCounts)
                   ?? ResolveHistoricalRaceCount(flow);
                if (resolvedHistoryCount.HasValue)
                {
                    runner.HistoricalRaceCount = resolvedHistoryCount;
                    flow.HistoricalRaceCount = resolvedHistoryCount;

                    if (runner.FeatureValues == null)
                    {
                        runner.FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                    }

                    runner.FeatureValues["CareerStarts"] = resolvedHistoryCount.Value;

                    if (flow.FeatureValues == null)
                    {
                        flow.FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                    }

                    flow.FeatureValues["CareerStarts"] = resolvedHistoryCount.Value;
                }
            }
            return runner;
        }
        private static int? ResolveHistoricalRaceCountFromPrefetch(
            RunnerFlow flow,
            IReadOnlyDictionary<string, int>? prefetchedCounts)
        {
            if (prefetchedCounts == null || prefetchedCounts.Count == 0)
            {
                return null;
            }

            if (flow == null || string.IsNullOrWhiteSpace(flow.HorseName))
            {
                return null;
            }

            foreach (var candidate in RacingRepository.BuildHistoricalNameCandidates(flow.HorseName))
            {
                if (prefetchedCounts.TryGetValue(candidate, out var count))
                {
                    return count;
                }
            }

            return null;
        }
        private int? ResolveHistoricalRaceCount(RunnerFlow flow)
        {
            if (flow == null)
            {
                return null;
            }

            if (!string.IsNullOrWhiteSpace(flow.HorseName))
            {
                try
                {
                    return _repo.GetHistoricalRaceCountByHorseName(flow.HorseName);
                }
                catch (Exception ex)
                {
                    var identifier = flow.HorseName ?? flow.SelectionId ?? "unknown";
                    Console.Error.WriteLine($"\tFailed to resolve historical race count for {identifier}: {ex.Message}");
                }
            }

            return null;
        }
        private static int? TryConvertToInt32(object? value)
        {
            if (value == null)
            {
                return null;
            }

            switch (value)
            {
                case int i:
                    return i;
                case long l when l <= int.MaxValue && l >= int.MinValue:
                    return (int)l;
                case short s:
                    return s;
                case byte b:
                    return b;
                case sbyte sb:
                    return sb;
                case ushort us when us <= int.MaxValue:
                    return (int)us;
                case uint ui when ui <= int.MaxValue:
                    return (int)ui;
                case float f when !float.IsNaN(f) && f <= int.MaxValue && f >= int.MinValue:
                    return (int)Math.Round(f);
                case double d when !double.IsNaN(d) && d <= int.MaxValue && d >= int.MinValue:
                    return (int)Math.Round(d);
                case decimal m when m <= int.MaxValue && m >= int.MinValue:
                    return (int)Math.Round(m);
                case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedInvariant):
                    return parsedInvariant;
                case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.CurrentCulture, out var parsedCurrent):
                    return parsedCurrent;
            }

            if (value is IConvertible convertible)
            {
                try
                {
                    return convertible.ToInt32(CultureInfo.InvariantCulture);
                }
                catch
                {
                    // ignored
                }
            }

            return null;
        }
        private void PopulateFeatureVectors(
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
          int runnerCount,
          IReadOnlyList<IDictionary<string, object?>>? preparedRows = null,
          UpcomingRace? persistedUpcoming = null)
        {
            if (flows == null || flows.Count == 0)
            {
                return;
            }

            var featureLookup = LoadFeatureLookup(
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
            foreach (var flow in flows)
            {
                var matchedFeatures = featureLookup.FindByHorse(flow.HorseName)
                    ?? featureLookup.FindBySaddlecloth(flow.ClothNumber);
                flow.HasPreparedFeatures = matchedFeatures != null;
                Dictionary<string, object?> featureVector;
                if (matchedFeatures != null)
                {
                    featureVector = new Dictionary<string, object?>(matchedFeatures, StringComparer.OrdinalIgnoreCase);
                }
                else
                {
                    featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                    var missingFeatureIdentifier = !string.IsNullOrWhiteSpace(flow.HorseName)
                        ? flow.HorseName!
                        : (flow.SelectionId ?? "unknown");
                    Console.WriteLine(
                        $"\t\tNo prepared feature row matched for {missingFeatureIdentifier}; neural model will fall back to legacy odds.");
                }

                if (runnerCount > 0)
                {
                    featureVector["RunnerCount"] = runnerCount;
                }

                if (featureVector.Count > 0)
                {
                    flow.FeatureValues = featureVector;
                }
                if (featureVector.TryGetValue("CareerStarts", out var careerStartsValue))
                {
                    flow.HistoricalRaceCount = TryConvertToInt32(careerStartsValue);
                }
                else
                {
                    flow.HistoricalRaceCount = null;
                }
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
                RaceDetails = raceDetails
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
                Going = parsedMetadata.Going,
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
        private sealed class BetfairScrapeResult
        {
            public List<BetRecommendation> Recommendations { get; } = new();
            public List<RaceDayReport> Races { get; } = new();
        }

        public IReadOnlyList<BetRecommendation> ScrapeOpenRaceTabs(IWebDriver driver)
        {
            return ScrapeOpenRaceTabsInternal(driver, executeBets: true, captureReport: false).Recommendations;
        }
        private sealed class FeatureLookup
        {
            private readonly Dictionary<string, Dictionary<string, object?>> _byHorse;
            private readonly Dictionary<int, Dictionary<string, object?>> _bySaddlecloth;
            private readonly IReadOnlyList<IDictionary<string, object?>> _rows;

            private FeatureLookup(
                Dictionary<string, Dictionary<string, object?>> byHorse,
                Dictionary<int, Dictionary<string, object?>> bySaddlecloth,
                IReadOnlyList<IDictionary<string, object?>> rows)
            {
                _byHorse = byHorse;
                _bySaddlecloth = bySaddlecloth;
                _rows = rows;
            }

            public static FeatureLookup Empty { get; } = new FeatureLookup(
                new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase),
                new Dictionary<int, Dictionary<string, object?>>(),
                Array.Empty<IDictionary<string, object?>>());

            public static FeatureLookup FromPreparedRace(PreparedRace race)
            {
                if (race == null)
                    throw new ArgumentNullException(nameof(race));

                var byHorse = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);
                var byCloth = new Dictionary<int, Dictionary<string, object?>>();
                var rows = new List<IDictionary<string, object?>>(race.Rows.Count);

                foreach (var row in race.Rows)
                {
                    var copy = new Dictionary<string, object?>(row, StringComparer.OrdinalIgnoreCase);
                    rows.Add(copy);

                    if (row.TryGetValue("HorseName", out var horseObj) && horseObj is string horse && !string.IsNullOrWhiteSpace(horse))
                    {
                        var normalized = NormalizeName(horse);
                        if (!string.IsNullOrEmpty(normalized) && !byHorse.ContainsKey(normalized))
                        {
                            byHorse[normalized] = copy;
                        }
                    }

                    if (PreparedDataset.TryGetRequiredInt32(row, "SaddleclothNumber", out var saddlecloth))
                    {
                        byCloth[saddlecloth] = copy;
                    }
                }

                return new FeatureLookup(byHorse, byCloth, rows);
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
                var byCloth = new Dictionary<int, Dictionary<string, object?>>();
                var copies = new List<IDictionary<string, object?>>(rows.Count);

                foreach (var row in rows)
                {
                    if (row == null)
                    {
                        continue;
                    }

                    var copy = row is Dictionary<string, object?> dict
                        ? new Dictionary<string, object?>(dict, StringComparer.OrdinalIgnoreCase)
                        : new Dictionary<string, object?>(row, StringComparer.OrdinalIgnoreCase);
                    copies.Add(copy);

                    if (copy.TryGetValue("HorseName", out var horseObj) && horseObj is string horse && !string.IsNullOrWhiteSpace(horse))
                    {
                        var normalized = NormalizeName(horse);
                        if (!string.IsNullOrEmpty(normalized) && !byHorse.ContainsKey(normalized))
                        {
                            byHorse[normalized] = copy;
                        }
                    }

                    if (PreparedDataset.TryGetRequiredInt32(copy, "SaddleclothNumber", out var saddlecloth))
                    {
                        byCloth[saddlecloth] = copy;
                    }
                }

                if (copies.Count == 0)
                {
                    return Empty;
                }

                return new FeatureLookup(byHorse, byCloth, copies);
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

            public Dictionary<string, object?>? FindBySaddlecloth(byte? clothNumber)
            {
                if (!clothNumber.HasValue)
                {
                    return null;
                }

                return _bySaddlecloth.TryGetValue(clothNumber.Value, out var features) ? features : null;
            }
        }
        private IEnumerable<BetRecommendation> CreateRecommendations(
            IEnumerable<RunnerFlow> flows,
            string marketId,
            string? raceTitle,
            string? venueName,
            DateTime? raceDate,
            bool applyKellyDampener)
        {
            if (_availableBankroll <= 0m)
            {
                return Enumerable.Empty<BetRecommendation>();
            }

            var recommendations = new List<BetRecommendation>();

            foreach (var flow in flows)
            {
                var identifier = !string.IsNullOrWhiteSpace(flow.HorseName)
                    ? flow.HorseName!
                    : (flow.SelectionId ?? "unknown");

                if (!flow.AiOdds.HasValue || !double.IsFinite(flow.AiOdds.Value) || flow.AiOdds.Value <= 0)
                {
                    Console.WriteLine($"\tSkipping {identifier}: AI probability unavailable or non-positive.");
                    continue;
                }

                if (!flow.BackPrice1.HasValue || flow.BackPrice1.Value <= 1m)
                {
                    var backText = flow.BackPrice1.HasValue
                        ? flow.BackPrice1.Value.ToString("0.##", CultureInfo.InvariantCulture)
                        : "null";
                    Console.WriteLine($"\tSkipping {identifier}: back price {backText} is not usable for value comparison.");
                    continue;
                }

                var decimalOdds = flow.BackPrice1.Value;
                var aiProbability = flow.AiOdds.Value;
                var marketProbability = 1.0 / (double)decimalOdds;
                var differential = aiProbability - marketProbability;

                Console.WriteLine($"\tRunner {identifier}: decimalOdds={decimalOdds.ToString("0.##", CultureInfo.InvariantCulture)}, aiProb={aiProbability.ToString("0.####", CultureInfo.InvariantCulture)}, marketProb={marketProbability.ToString("0.####", CultureInfo.InvariantCulture)}, diff={differential.ToString("0.####", CultureInfo.InvariantCulture)}");

                if (differential <= 0)
                {
                    Console.WriteLine($"\t\tRejected {identifier}: differential {differential.ToString("0.####", CultureInfo.InvariantCulture)} is not positive after ignoring exchange commission.");
                    continue;
                }

                var kellyFraction = CalculateKellyFraction(aiProbability, (double)decimalOdds);
                if (applyKellyDampener && _kellyDampener < 1m)
                {
                    kellyFraction *= _kellyDampener;
                }

                if (kellyFraction > 1m)
                {
                    kellyFraction = 1m;
                }
                if (kellyFraction <= 0)
                {
                    Console.WriteLine($"\t\tRejected {identifier}: Kelly fraction {kellyFraction.ToString("0.####", CultureInfo.InvariantCulture)} is non-positive.");
                    continue;
                }

                Console.WriteLine($"\t\tAccepted {identifier}: Kelly fraction {kellyFraction.ToString("0.####", CultureInfo.InvariantCulture)} (commission not yet deducted).");

                recommendations.Add(new BetRecommendation
                {
                    MarketId = marketId,
                    SelectionId = flow.SelectionId,
                    HorseName = flow.HorseName,
                    RaceTitle = raceTitle,
                    VenueName = venueName,
                    RaceDate = raceDate,
                    DecimalOdds = decimalOdds,
                    AiDecimalOdds = CalculateAiDecimalOdds(aiProbability),
                    AiProbability = aiProbability,
                    MarketProbability = marketProbability,
                    Differential = differential,
                    KellyFraction = kellyFraction,
                    Stake = 0m
                });
            }

            return recommendations;
        }
        private IReadOnlyList<BetRecommendation> ExecuteBackAllClicks(
            IWebDriver driver,
            IReadOnlyList<(IWebElement Row, RunnerFlow Flow)> runnerEntries,
            IReadOnlyList<BetRecommendation> recommendations)
        {
            if (runnerEntries.Count == 0 || recommendations.Count == 0)
            {
                return Array.Empty<BetRecommendation>();
            }

            var clicked = new List<BetRecommendation>();

            foreach (var recommendation in recommendations)
            {
                var identifier = recommendation.HorseName ?? recommendation.SelectionId ?? "unknown";

                if (recommendation.Differential <= 0)
                {
                    Console.WriteLine($"\tSkipping {identifier}: differential {recommendation.Differential.ToString("0.####", CultureInfo.InvariantCulture)} <= 0.");
                    continue;
                }
                if (_availableBankroll <= 0m)
                {
                    Console.WriteLine($"\tBankroll exhausted before sizing stake for {identifier}.");
                    break;
                }

                if (recommendation.KellyFraction <= 0m)
                {
                    Console.WriteLine($"\tSkipping {identifier}: Kelly fraction {recommendation.KellyFraction.ToString("0.####", CultureInfo.InvariantCulture)} <= 0.");
                    continue;
                }

                Console.WriteLine($"\tSizing stake for {identifier}: bankroll {_availableBankroll.ToString("0.##", CultureInfo.InvariantCulture)}, Kelly {recommendation.KellyFraction.ToString("0.####", CultureInfo.InvariantCulture)}");
                var stake = CalculateStakeWithLimits(_availableBankroll, recommendation.KellyFraction);
                if (stake <= 0m)
                {
                    Console.WriteLine($"\t\tSequential Kelly returned zero stake for {identifier}; check rounding or Kelly cap constraints.");
                    continue;
                }
                var match = runnerEntries.FirstOrDefault(entry =>
                    (!string.IsNullOrEmpty(recommendation.SelectionId) &&
                        string.Equals(entry.Flow.SelectionId, recommendation.SelectionId, StringComparison.Ordinal)) ||
                    (!string.IsNullOrWhiteSpace(recommendation.HorseName) &&
                        !string.IsNullOrWhiteSpace(entry.Flow.HorseName) &&
                        string.Equals(entry.Flow.HorseName, recommendation.HorseName, StringComparison.OrdinalIgnoreCase)));

                if (match.Row == null)
                {
                    Console.Error.WriteLine($"\tUnable to locate row for {recommendation.HorseName ?? recommendation.SelectionId ?? "unknown"} to click Back-All");
                    continue;
                }

                if (TryClickBackAllButton(driver, match.Row, recommendation))
                {
                    var clickedRecommendation = recommendation with { Stake = stake };
                    clicked.Add(clickedRecommendation);
                    _availableBankroll -= stake;
                    if (_availableBankroll < 0m)
                    {
                        _availableBankroll = 0m;
                    }

                    Console.WriteLine($"\t\tStake {stake.ToString("0.##", CultureInfo.InvariantCulture)} accepted for {identifier}; bankroll now {_availableBankroll.ToString("0.##", CultureInfo.InvariantCulture)}");

                    Thread.Sleep(TimeSpan.FromMilliseconds(400));
                }
                else
                {
                    Console.WriteLine($"\t\tBack-All click failed or was skipped for {identifier}; bankroll remains {_availableBankroll.ToString("0.##", CultureInfo.InvariantCulture)}");
                }
            }
            return clicked;
        }
        private bool TryClickBackAllButton(IWebDriver driver, IWebElement row, BetRecommendation recommendation)
        {
            var identifier = recommendation.HorseName ?? recommendation.SelectionId ?? "unknown";

            try
            {
                var button = FindBackAllButton(row);
                var js = (IJavaScriptExecutor)driver;

                if (button == null)
                {
                    if (TryClickBackAllViaScript(js, row))
                    {
                        Console.WriteLine($"\tClicked Back-All for {identifier} using script fallback");
                        return true;
                    }
                    else
                    {
                        Console.Error.WriteLine($"\tBack-All button not found for {identifier}");
                    }
                    return false;
                }

                try
                {
                    js.ExecuteScript("arguments[0].scrollIntoView({block:'center'});", button);
                }
                catch (Exception)
                {
                }

                try
                {
                    button.Click();
                    Console.WriteLine($"\tClicked Back-All for {identifier}");
                    return true;
                }
                catch (Exception)
                {
                }

                try
                {
                    js.ExecuteScript("arguments[0].click();", button);
                    Console.WriteLine($"\tClicked Back-All for {identifier} using JavaScript");
                    return true;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to click Back-All for {identifier}: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tUnexpected error clicking Back-All for {identifier}: {ex.Message}");
            }
            return false;
        }
        private static IWebElement? FindBackAllButton(IWebElement row)
        {
            static bool MatchesBackAll(IWebElement element)
            {
                if (element == null)
                {
                    return false;
                }
                bool IsBackAllContext(IWebElement el)
                {
                    IWebElement? current = el;
                    for (var depth = 0; depth < 6 && current != null; depth++)
                    {
                        try
                        {
                            var classAttribute = current.GetAttribute("class");
                            if (!string.IsNullOrWhiteSpace(classAttribute))
                            {
                                var lowered = classAttribute.ToLowerInvariant();
                                if (lowered.Contains("back-all"))
                                {
                                    return true;
                                }

                                if (lowered.Contains("bet-buttons") &&
                                    lowered.Contains("back-cell") &&
                                    lowered.Contains("last-back-cell"))
                                {
                                    return true;
                                }
                            }

                            var betType = current.GetAttribute("bet-type");
                            if (!string.IsNullOrWhiteSpace(betType) &&
                                betType.Equals("back", StringComparison.OrdinalIgnoreCase))
                            {
                                var handicap = current.GetAttribute("bet-handicap");
                                if (string.IsNullOrWhiteSpace(handicap) ||
                                    handicap.Equals("0", StringComparison.OrdinalIgnoreCase))
                                {
                                    return true;
                                }
                            }

                            current = TryFindElement(current, By.XPath(".."));
                        }
                        catch (Exception)
                        {
                            break;
                        }
                    }

                    return false;
                }

                bool IsBackButton(IWebElement el)
                {
                    try
                    {
                        var classAttribute = el.GetAttribute("class");
                        if (!string.IsNullOrWhiteSpace(classAttribute))
                        {
                            var lowered = classAttribute.ToLowerInvariant();
                            if (lowered.Contains("lay"))
                            {
                                return false;
                            }

                            if (lowered.Contains("back"))
                            {
                                return true;
                            }
                        }

                        var typeAttribute = el.GetAttribute("type");
                        if (!string.IsNullOrWhiteSpace(typeAttribute) &&
                            typeAttribute.Equals("back", StringComparison.OrdinalIgnoreCase))
                        {
                            return true;
                        }
                    }
                    catch (Exception)
                    {
                    }

                    return string.Equals(el.TagName, "button", StringComparison.OrdinalIgnoreCase);
                }

                string? ReadText(IWebElement el)
                {
                    var text = el.Text;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        return text;
                    }

                    var textContent = el.GetAttribute("textContent");
                    if (!string.IsNullOrWhiteSpace(textContent))
                    {
                        return textContent;
                    }

                    var innerText = el.GetAttribute("innerText");
                    return !string.IsNullOrWhiteSpace(innerText) ? innerText : null;
                }

                bool TextContainsBackAll(string? value)
                {
                    return !string.IsNullOrWhiteSpace(value) &&
                        value.IndexOf("back all", StringComparison.OrdinalIgnoreCase) >= 0;
                }

                if (TextContainsBackAll(ReadText(element)))
                {
                    return true;
                }

                foreach (var attribute in new[] { "aria-label", "title", "data-testid", "data-trackid", "data-action" })
                {
                    if (TextContainsBackAll(element.GetAttribute(attribute)))
                    {
                        return true;
                    }
                    var attributeValue = element.GetAttribute(attribute);
                    if (!string.IsNullOrWhiteSpace(attributeValue) &&
                        attributeValue.IndexOf("back-all", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        return true;
                    }
                }

                try
                {
                    var labelCandidates = element.FindElements(By.CssSelector("label, span, strong, div, p"));
                    foreach (var label in labelCandidates)
                    {
                        if (!ReferenceEquals(label, element) && TextContainsBackAll(ReadText(label)))
                        {
                            return true;
                        }
                    }
                }
                catch (Exception)
                {
                }

                return false;
            }

            IWebElement? ValidateCandidate(IWebElement? candidate)
            {
                if (candidate == null)
                {
                    return null;
                }

                if (MatchesBackAll(candidate))
                {
                    return candidate;
                }

                try
                {
                    var nestedButton = TryFindElement(candidate, By.TagName("button"));
                    if (nestedButton != null && !ReferenceEquals(nestedButton, candidate) && MatchesBackAll(nestedButton))
                    {
                        return nestedButton;
                    }
                }
                catch (Exception)
                {
                }

                return null;
            }

            var selectors = new[]
            {
                By.CssSelector("button[data-testid='back-all']"),
                By.CssSelector("button[data-testid='button-back-all']"),
                By.CssSelector("button[data-testid='back-all-button']"),
                By.CssSelector("button[data-testid*='back-all']"),
                By.CssSelector("button[aria-label*='back all' i]"),
                By.CssSelector("button[title*='back all' i]"),
                By.CssSelector("ours-price-button button[data-testid*='back-all']"),
                By.CssSelector("td:nth-of-type(4) ours-price-button button"),
                By.CssSelector("td.bet-buttons.back-cell.last-back-cell > ours-price-button > button"),
                By.CssSelector("td.bet-buttons.back-cell.last-back-cell ours-price-button button"),
                By.CssSelector("[data-testid='back-all'] button"),
                By.CssSelector("[data-testid*='back-all'] button"),
                By.CssSelector("button.back-all"),
                By.CssSelector("button.back-all-button"),
                By.CssSelector("button[class*='back-all']"),
            };

            foreach (var selector in selectors)
            {
                try
                {
                    var button = ValidateCandidate(TryFindElement(row, selector));
                    if (button != null)
                    {
                        return button;
                    }
                }
                catch (Exception)
                {
                }
            }

            try
            {
                var candidates = row.FindElements(By.CssSelector("ours-price-button button, button, [role='button'], .bet-button"));
                foreach (var candidate in candidates)
                {
                    var validated = ValidateCandidate(candidate);
                    if (validated != null)
                    {
                        return validated;
                    }
                }
            }
            catch (Exception)
            {
            }

            return null;
        }
        
        private void PopulateBetSlipStakes(
            IWebDriver driver,
            IReadOnlyList<BetRecommendation> recommendations,
            decimal startingBankroll)
        {
            if (recommendations.Count == 0)
            {
                return;
            }

            var expectedCount = _betSlipSelectionsFilled + recommendations.Count;
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(5));
            try
            {
                wait.Until(d => FindBetSlipStakeInputs(d).Count >= expectedCount);
            }
            catch (WebDriverTimeoutException)
            {
                // proceed with whatever entries are available
            }

            var inputs = FindBetSlipStakeInputs(driver);

            if (inputs.Count < _betSlipSelectionsFilled)
            {
                _betSlipSelectionsFilled = 0;
            }

            if (inputs.Count <= _betSlipSelectionsFilled)
            {
                return;
            }

            var startIndex = _betSlipSelectionsFilled;
            var available = inputs.Skip(startIndex).Take(recommendations.Count).ToList();

            if (available.Count == 0)
            {
                return;
            }

            var js = (IJavaScriptExecutor)driver;
            var remainingPot = startingBankroll;

            for (var i = 0; i < available.Count && i < recommendations.Count; i++)
            {
                var recommendation = recommendations[i];
                var stake = CalculateStakeWithLimits(remainingPot, recommendation.KellyFraction);
                var input = available[i];
                var identifier = recommendation.HorseName ?? recommendation.SelectionId ?? "unknown";
                Console.WriteLine($"\tBet slip allocation for {identifier}: remaining pot {remainingPot.ToString("0.##", CultureInfo.InvariantCulture)}, stake {stake.ToString("0.##", CultureInfo.InvariantCulture)}");
                try
                {
                    SetStakeInputValue(js, input, stake);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to populate stake input: {ex.Message}");
                }

                remainingPot -= stake;
                if (remainingPot < 0m)
                {
                    remainingPot = 0m;
                }
                Console.WriteLine($"\t\tRemaining pot after allocation: {remainingPot.ToString("0.##", CultureInfo.InvariantCulture)}");
            }

            _betSlipSelectionsFilled += available.Count;
        }
        private decimal CalculateStakeWithLimits(decimal bankroll, decimal kellyFraction)
        {
            var stake = BettingMath.CalculateSequentialStake(bankroll, kellyFraction);
            if (stake <= 0m)
            {
                return 0m;
            }

            var maxStake = DetermineMaxStake(bankroll);
            if (maxStake.HasValue && maxStake.Value > 0m && stake > maxStake.Value)
            {
                stake = maxStake.Value;
            }

            if (stake > bankroll)
            {
                stake = bankroll;
            }

            if (stake < 0m)
            {
                stake = 0m;
            }

            return decimal.Round(stake, 2, MidpointRounding.ToZero);
        }

        private decimal? DetermineMaxStake(decimal bankroll)
        {
            decimal? raw = _maxStakeMode switch
            {
                MaxStakeMode.PercentageOfBankroll when _maxStakePercentOfBankroll.HasValue && _maxStakePercentOfBankroll.Value > 0m
                    => bankroll * _maxStakePercentOfBankroll.Value,
                MaxStakeMode.FixedAmount when _maxStakeFixedAmount.HasValue && _maxStakeFixedAmount.Value > 0m
                    => _maxStakeFixedAmount.Value,
                _ => null
            };

            if (!raw.HasValue)
            {
                return null;
            }

            var capped = raw.Value;
            if (capped > bankroll)
            {
                capped = bankroll;
            }

            if (capped <= 0m)
            {
                return null;
            }

            var rounded = decimal.Round(capped, 2, MidpointRounding.ToZero);
            if (rounded <= 0m)
            {
                return null;
            }

            if (rounded < 1m && bankroll >= 1m)
            {
                rounded = 1m;
            }

            return rounded;
        }
        private decimal CalculateKellyFraction(double probability, double decimalOdds)
        {
            if (probability <= 0 || probability >= 1 || decimalOdds <= 1)
            {
                return 0m;
            }

            var b = decimalOdds - 1.0;
            if (Math.Abs(b) < double.Epsilon)
            {
                return 0m;
            }

            var q = 1.0 - probability;
            var fraction = (b * probability - q) / b;

            if (!double.IsFinite(fraction))
            {
                return 0m;
            }

            var result = (decimal)fraction;
            if (result < 0m)
            {
                result = 0m;
            }

            if (_maxKellyFraction.HasValue && result > _maxKellyFraction.Value)
            {
                result = _maxKellyFraction.Value;
            }

            if (result > 1m)
            {
                result = 1m;
            }

            return result;
        }
        private static string? FindFirstFeatureString(IEnumerable<RunnerFlow> flows, params string[] keys)
        {
            foreach (var flow in flows)
            {
                if (flow.FeatureValues == null)
                {
                    continue;
                }

                foreach (var key in keys)
                {
                    if (flow.FeatureValues.TryGetValue(key, out var value) && value != null)
                    {
                        var text = ConvertToInvariantString(value)?.Trim();
                        if (!string.IsNullOrWhiteSpace(text))
                        {
                            return text;
                        }
                    }
                }
            }

            return null;
        }
    
}
}