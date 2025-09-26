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
        private readonly RacingRepository _repo;
        private readonly HyperparameterTrainer _trainer;
        private readonly object _repoLock = new();
        private readonly decimal _bankroll;
        private readonly decimal? _maxKellyFraction;
        private readonly bool _useMarketFallbackForAiDegeneracy;
        private decimal _availableBankroll;
        private int _betSlipSelectionsFilled;
        public BetfairMarketScraper(
            RacingRepository repo,
            HyperparameterTrainer trainer,
            decimal bankroll,
            decimal? maxKellyFraction = null,
            bool useMarketFallbackForAiDegeneracy = true)
        {
            _repo = repo ?? throw new ArgumentNullException(nameof(repo));
            _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
            _bankroll = bankroll;
            _availableBankroll = bankroll;
            _maxKellyFraction = maxKellyFraction;
            _betSlipSelectionsFilled = 0;
            _useMarketFallbackForAiDegeneracy = useMarketFallbackForAiDegeneracy;
        }
        public IReadOnlyList<RaceDayReport> ScrapeOpenRaceTabsForReport(IWebDriver driver)
        {
            var result = ScrapeOpenRaceTabsInternal(driver, executeBets: false, captureReport: true);
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

            var model = _trainer.LoadLatestTrainedModel();
            if (model == null)
            {
                Console.Error.WriteLine("[DayReport] Unable to load trained AI model; skipping probability population.");
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
                    continue;
                }

                var runnersByHorse = new Dictionary<string, RunnerDayReport>(StringComparer.Ordinal);
                foreach (var runner in race.Runners)
                {
                    if (runner?.HorseName == null)
                    {
                        continue;
                    }

                    var normalizedHorse = NormalizeName(runner.HorseName);
                    if (string.IsNullOrEmpty(normalizedHorse) || runnersByHorse.ContainsKey(normalizedHorse))
                    {
                        continue;
                    }

                    runnersByHorse[normalizedHorse] = runner;
                }
                int? raceId = null;
                try
                {
                    if (race.RaceDate.HasValue)
                    {
                        raceId = _repo.FindRaceId(race.RaceDate.Value, race.RaceTitle, race.VenueName);
                    }

                    if (!raceId.HasValue && race.RaceDate.HasValue)
                    {
                        raceId = _repo.FindRaceId(race.RaceDate.Value, race.RaceTitle, null);
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\t[DayReport] Failed to resolve race identifier for {race.RaceTitle ?? race.MarketId}: {ex.Message}");
                    continue;
                }

                if (!raceId.HasValue)
                {
                    var upcomingId = TryRecordUpcomingRace(race);
                    race.UpcomingRaceId = upcomingId;
                    continue;
                }

                PreparedDataset prepared;
                try
                {
                    prepared = _trainer.PrepareDataset(new HashSet<int> { raceId.Value }, null, includeIdentifiers: true);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\t[DayReport] Failed to prepare feature set for race {raceId.Value}: {ex.Message}");
                    continue;
                }

                var preparedRace = prepared.Races.FirstOrDefault(r => r.RaceId == raceId.Value);
                if (preparedRace == null)
                {
                    continue;
                }

                var predictions = _trainer.PredictRaceProbabilities(preparedRace, model);
                if (predictions.Count == 0)
                {
                    continue;
                }

                var winner = predictions
                    .OrderByDescending(p => p.Probability)
                    .First();

                RunnerDayReport? runnerMatch = null;
                if (!string.IsNullOrEmpty(winner.HorseName))
                {
                    var horseKey = NormalizeName(winner.HorseName);
                    if (!string.IsNullOrEmpty(horseKey) && runnersByHorse.TryGetValue(horseKey, out var horseMatch))
                    {
                        runnerMatch = horseMatch;
                    }
                }

                if (runnerMatch == null && winner.ClothNumber.HasValue)
                {
                    runnerMatch = race.Runners.FirstOrDefault(r => r?.ClothNumber == winner.ClothNumber.Value);
                }

                if (runnerMatch == null && !string.IsNullOrEmpty(winner.SelectionId))
                {
                    runnerMatch = race.Runners.FirstOrDefault(r =>
                        !string.IsNullOrWhiteSpace(r?.SelectionId) &&
                        string.Equals(r.SelectionId, winner.SelectionId, StringComparison.OrdinalIgnoreCase));
                }

                if (runnerMatch == null)
                {
                    var winnerIdentifier = !string.IsNullOrWhiteSpace(winner.HorseName)
                        ? winner.HorseName
                        : (!string.IsNullOrWhiteSpace(winner.SelectionId) ? winner.SelectionId : "unknown");
                    Console.Error.WriteLine($"\t[DayReport] Failed to match predicted winner {winnerIdentifier} for race {race.RaceTitle ?? race.MarketId}.");
                    continue;
                }

                var probability = winner.Probability;
                runnerMatch.AiProbability = probability;
                runnerMatch.AiDecimalOdds = BettingMath.CalculateAiDecimalOdds(probability);

                if (runnerMatch.MarketProbability.HasValue)
                {
                    runnerMatch.Differential = probability - runnerMatch.MarketProbability.Value;
                }
                else
                {
                    runnerMatch.Differential = null;
                }

                if (runnerMatch.MarketDecimalOdds.HasValue && runnerMatch.MarketDecimalOdds.Value > 1m)
                {
                    var kelly = BettingMath.CalculateKellyFraction(probability, (double)runnerMatch.MarketDecimalOdds.Value, _maxKellyFraction);
                    runnerMatch.KellyFraction = kelly;
                    runnerMatch.SuggestedStake = (kelly > 0m && _bankroll > 0m)
                        ? BettingMath.CalculateSequentialStake(_bankroll, kelly)
                        : (decimal?)null;
                }
                else
                {
                    runnerMatch.KellyFraction = null;
                    runnerMatch.SuggestedStake = null;
                }

                foreach (var runner in race.Runners.ToList())
                {
                    if (!ReferenceEquals(runner, runnerMatch))
                    {
                        race.Runners.Remove(runner);
                    }
                }
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
        private static IEnumerable<string> EnumerateDetailTokens(string? source)
        {
            if (string.IsNullOrWhiteSpace(source))
            {
                yield break;
            }

            var normalized = source.Replace('\u00A0', ' ').Trim();
            if (!string.IsNullOrEmpty(normalized))
            {
                yield return normalized;
            }

            var separators = new[] { '|', '/', '\\', ',', ';', '–', '—', '·' };
            foreach (var segment in normalized.Split(separators, StringSplitOptions.RemoveEmptyEntries))
            {
                var trimmed = segment.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    yield return trimmed;
                }

                foreach (var token in trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    var inner = token.Trim();
                    if (!string.IsNullOrEmpty(inner))
                    {
                        yield return inner;
                    }
                }
            }
        }

        private static bool HasDistanceToken(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                return false;
            }

            return DistanceComponentRegex.IsMatch(token);
        }

        private static int ParseDistanceToYards(string? token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                return 0;
            }

            var text = token.ToLowerInvariant();
            var total = 0;
            foreach (Match match in DistanceComponentRegex.Matches(text))
            {
                if (!double.TryParse(match.Groups["value"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                {
                    continue;
                }

                switch (match.Groups["unit"].Value)
                {
                    case "m":
                        total += (int)Math.Round(value * 1760d);
                        break;
                    case "f":
                        total += (int)Math.Round(value * 220d);
                        break;
                    case "y":
                        total += (int)Math.Round(value);
                        break;
                }
            }

            return total;
        }

        private static string? TryDetectRaceType(IEnumerable<string> tokens, string? title)
        {
            foreach (var token in tokens)
            {
                var found = RaceTypeKeywords.FirstOrDefault(k => token.IndexOf(k, StringComparison.OrdinalIgnoreCase) >= 0);
                if (!string.IsNullOrEmpty(found))
                {
                    return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(found);
                }
            }

            if (!string.IsNullOrWhiteSpace(title))
            {
                foreach (var keyword in RaceTypeKeywords)
                {
                    if (title.IndexOf(keyword, StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(keyword);
                    }
                }
            }

            return null;
        }

        private static bool IsGoingToken(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                return false;
            }

            var normalized = token.ToLowerInvariant();
            if (normalized.Contains("going"))
            {
                return true;
            }

            var keywords = new[] { "heavy", "soft", "yielding", "good", "firm", "standard", "slow", "fast" };
            return keywords.Any(k => Regex.IsMatch(normalized, $"\\b{k}\\b"));
        }

        private static string? DetermineSurface(string? going, IEnumerable<string> tokens)
        {
            static bool ContainsAwIndicator(string value)
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    return false;
                }

                var normalized = value.ToLowerInvariant();
                return normalized.Contains("all weather")
                    || normalized.Contains("all-weather")
                    || normalized.Contains("a/w")
                    || normalized.Equals("aw", StringComparison.OrdinalIgnoreCase)
                    || normalized.Contains("polytrack")
                    || normalized.Contains("tapeta")
                    || normalized.Contains("fibresand");
            }

            if (!string.IsNullOrEmpty(going) && going.IndexOf("standard", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return "All Weather";
            }

            if (tokens.Any(ContainsAwIndicator))
            {
                return "All Weather";
            }

            return "Turf";
        }

        private readonly record struct ParsedRaceMetadata(
            string? DistanceText,
            int DistanceYards,
            byte? Class,
            string? AgeRestriction,
            string? Going,
            string? Surface,
            string? RaceType);
      
        private static readonly Regex BracketedNameContentRegex =
            new Regex(@"\s*[\(\[][^\)\]]*[\)\]]\s*", RegexOptions.Compiled);


        private BetfairScrapeResult ScrapeOpenRaceTabsInternal(IWebDriver driver, bool executeBets, bool captureReport)
        {
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10)); // short explicit wait
            var handles = driver.WindowHandles.ToList(); // collect tab handles
            var weightPath = ResolveAiWeightPath(); // resolve AI weights path

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

            var result = new BetfairScrapeResult(); // aggregate result
            var recommendations = new List<BetRecommendation>(); // all bet recs

            foreach (var handle in handles)
            {
                driver.SwitchTo().Window(handle); // switch tab
                Console.WriteLine($"Processing tab: {driver.Url}"); // log url

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

                var title = TextOrEmpty(driver, By.CssSelector("[data-testid='marketTitle']")); // market title
                var venueText = TextOrEmpty(driver, By.CssSelector(".venue-name")); // venue text
                var eventDateText = TextOrEmpty(driver, By.CssSelector(".event-date")); // event date raw
                var raceDetailsText = TextOrEmpty(driver, By.CssSelector(".market-name")); // race details
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
                        LayBookPercentage = layBookPercentage // %
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

                PopulateFeatureVectors(parsedRaceDate, title, venueName, flows, rows.Count); // build features

                foreach (var rf in flows) // process each runner
                {
                    if (rf.FeatureValues != null && !rf.FeatureValues.ContainsKey("RunnerCount") && rows.Count > 0)
                    {
                        rf.FeatureValues["RunnerCount"] = rows.Count; // ensure runner count
                    }

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
                        if (double.IsFinite(probability) && probability >= 0) { rf.AiOdds = probability; } else { rf.AiOdds = null; var badId = rf.SelectionId ?? rf.HorseName ?? "unknown"; Console.Error.WriteLine($"\tInvalid AI odds calculated for selection {badId} in market {marketId}"); } // validate
                    }
                    catch (Exception ex)
                    {
                        rf.AiOdds = null; // set null on fail
                        var failId = rf.SelectionId ?? rf.HorseName ?? "unknown"; // id for log
                        Console.Error.WriteLine($"\tFailed to calculate AI odds for selection {failId} in market {marketId}: {ex.Message}"); // log
                    }

                    var rfIdentifier = !string.IsNullOrWhiteSpace(rf.HorseName) ? rf.HorseName! : (rf.SelectionId ?? "unknown"); // readable id
                    var aiText = rf.AiOdds.HasValue ? rf.AiOdds.Value.ToString("0.####", CultureInfo.InvariantCulture) : "null"; // ai text
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
                        flows
                    ); // build report

                    result.Races.Add(report); // collect report
                }

                var raceRecommendations = CreateRecommendations(flows, marketId, title, venueName, parsedRaceDate)
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

                foreach (var inner in flows) // persist each flow
                {
                    try
                    {
                        lock (_repoLock) { _repo.InsertRunnerFlow(inner); } // save flow
                        var innerIdentifier = !string.IsNullOrWhiteSpace(inner.HorseName)
                            ? inner.HorseName!.Trim()
                            : (!string.IsNullOrWhiteSpace(inner.SelectionId) ? inner.SelectionId! : "unknown");
                        Console.WriteLine($"\tInserted runner {innerIdentifier} for market {marketId}"); // log ok
                    }
                    catch (Exception ex)
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
                LayBookPercentage = layBookPercentage
            };

            foreach (var flow in flows)
            {
                report.Runners.Add(CreateRunnerReport(flow));
            }

            return report;
        }

        private RunnerDayReport CreateRunnerReport(RunnerFlow flow)
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

            if (flow.AiOdds.HasValue && double.IsFinite(flow.AiOdds.Value) && flow.AiOdds.Value > 0)
            {
                runner.AiProbability = flow.AiOdds.Value;
                runner.AiDecimalOdds = BettingMath.CalculateAiDecimalOdds(flow.AiOdds.Value);
            }

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
                    runner.SuggestedStake = BettingMath.CalculateSequentialStake(_bankroll, kelly);
                }
            }

            return runner;
        }
        private void PopulateFeatureVectors(
          DateTime? raceDate,
          string? raceTitle,
          string? venueName,
          IReadOnlyList<RunnerFlow> flows,
          int runnerCount,
          IReadOnlyList<IDictionary<string, object?>>? preparedRows = null)
        {
            if (flows == null || flows.Count == 0)
            {
                return;
            }

            var featureLookup = LoadFeatureLookup(raceDate, raceTitle, venueName, flows, preparedRows);
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
            }
        }

        private FeatureLookup LoadFeatureLookup(
            DateTime? raceDate,
            string? raceTitle,
            string? venueName,
            IReadOnlyList<RunnerFlow> flows,
            IReadOnlyList<IDictionary<string, object?>>? preparedRows)
        {
            if (preparedRows != null && preparedRows.Count > 0)
            {
                Console.WriteLine(
                    $"\tUsing {preparedRows.Count} pre-provided feature row(s) for lookup.");
                return FeatureLookup.FromRows(preparedRows);
            }
            if (!raceDate.HasValue)
            {
                return FeatureLookup.Empty;
            }

            int? raceId;
            try
            {
                raceId = _repo.FindRaceId(raceDate.Value, raceTitle, venueName);
                if (raceId.HasValue)
                {
                    Console.WriteLine(
                        $"\tResolved race lookup: date={raceDate.Value:yyyy-MM-dd}, title='{raceTitle ?? "<null>"}', venue='{venueName ?? "<null>"}' => raceId={raceId.Value}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tUnable to resolve race details for feature lookup: {ex.Message}");
                return FeatureLookup.Empty;
            }

            if (!raceId.HasValue)
            {
                Console.WriteLine(
                    $"\tNo race ID found for date={raceDate.Value:yyyy-MM-dd}, title='{raceTitle ?? "<null>"}', venue='{venueName ?? "<null>"}'.");
                var upcoming = _repo.FindUpcomingRace(raceDate.Value, raceTitle, venueName);
                if (upcoming == null)
                {
                    Console.WriteLine("\tNo matching upcoming race metadata available for feature synthesis.");
                    return FeatureLookup.Empty;
                }

                try
                {
                    var prepared = _trainer.PrepareUpcomingRace(upcoming, flows);
                    if (prepared == null)
                    {
                        Console.WriteLine("\tSynthetic feature preparation for upcoming race returned no rows.");
                        return FeatureLookup.Empty;
                    }

                    Console.WriteLine(
                        $"\tLoaded synthetic feature rows for upcoming race {upcoming.MarketId}; runner count={prepared.Rows.Count}.");
                    return FeatureLookup.FromPreparedRace(prepared);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to build synthetic feature vector for upcoming race {upcoming.MarketId}: {ex.Message}");
                    return FeatureLookup.Empty;
                }
            }

            try
            {
                var include = new HashSet<int> { raceId.Value };
                var prepared = _trainer.PrepareDataset(include, null, includeIdentifiers: true);
                var race = prepared.Races.FirstOrDefault(r => r.RaceId == raceId.Value);
                if (race == null)
                {
                    Console.WriteLine($"\tFeature preparation returned no race data for raceId={raceId.Value}.");
                    return FeatureLookup.Empty;
                }

                Console.WriteLine(
                    $"\tLoaded feature rows for raceId={raceId.Value}; runner count={race.Runners.Count}.");

                return FeatureLookup.FromPreparedRace(race);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tFailed to build feature vector for race {raceId.Value}: {ex.Message}");
                return FeatureLookup.Empty;
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
            DateTime? raceDate)
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
                var stake = CalculateSequentialStake(_availableBankroll, recommendation.KellyFraction);
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
            var selectors = new[]
            {
                By.CssSelector("button[data-testid='back-all']"),
                By.CssSelector("button.back-all"),
                By.CssSelector("button.back-all-button"),
                By.CssSelector("button[class*='back-all']")
            };

            foreach (var selector in selectors)
            {
                var button = TryFindElement(row, selector);
                if (button != null)
                {
                    return button;
                }
            }

            try
            {
                var buttons = row.FindElements(By.TagName("button"));
                foreach (var candidate in buttons)
                {
                    var text = candidate.Text;
                    if (string.IsNullOrWhiteSpace(text))
                    {
                        text = candidate.GetAttribute("textContent");
                    }

                    if (!string.IsNullOrWhiteSpace(text) && text.IndexOf("back all", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        return candidate;
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
                var stake = CalculateSequentialStake(remainingPot, recommendation.KellyFraction);
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