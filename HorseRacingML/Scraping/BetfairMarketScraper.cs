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
    public class BetfairMarketScraper
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
        private static string ResolveAiWeightPath()
        {
            static IEnumerable<string> EnumerateCandidates()
            {
                static IEnumerable<string> ExpandDirectory(string? directory)
                {
                    if (string.IsNullOrWhiteSpace(directory))
                    {
                        yield break;
                    }

                    yield return Path.Combine(directory, "weights", "aiweights.json");
                    yield return Path.Combine(directory, "aiweights.json");
                }

                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

                foreach (var path in ExpandDirectory(AppContext.BaseDirectory))
                {
                    if (seen.Add(path))
                    {
                        yield return path;
                    }
                }

                foreach (var path in ExpandDirectory(Directory.GetCurrentDirectory()))
                {
                    if (seen.Add(path))
                    {
                        yield return path;
                    }
                }

                var current = AppContext.BaseDirectory;
                for (var i = 0; i < 5 && !string.IsNullOrEmpty(current); i++)
                {
                    current = Path.GetDirectoryName(current?.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                    if (string.IsNullOrEmpty(current))
                    {
                        break;
                    }

                    foreach (var path in ExpandDirectory(current))
                    {
                        if (seen.Add(path))
                        {
                            yield return path;
                        }
                    }
                }
            }

            foreach (var candidate in EnumerateCandidates())
            {
                if (File.Exists(candidate))
                {
                    Console.WriteLine($"\tUsing AI weight file at {candidate}");
                    return candidate;
                }
            }

            var fallback = Path.Combine(AppContext.BaseDirectory, "weights", "aiweights.json");
            Console.Error.WriteLine($"\tAI weight file not found; expected locations include {fallback}");
            return fallback;
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
                if (!string.IsNullOrEmpty(winner.SelectionId))
                {
                    runnerMatch = race.Runners.FirstOrDefault(r => string.Equals(r.SelectionId, winner.SelectionId, StringComparison.OrdinalIgnoreCase));
                }

                if (runnerMatch == null && winner.ClothNumber.HasValue)
                {
                    runnerMatch = race.Runners.FirstOrDefault(r => r.ClothNumber == winner.ClothNumber.Value);
                }

                if (runnerMatch == null && !string.IsNullOrEmpty(winner.HorseName))
                {
                    var horseKey = NormalizeName(winner.HorseName);
                    runnerMatch = race.Runners.FirstOrDefault(r => NormalizeName(r.HorseName) == horseKey);
                }

                if (runnerMatch == null)
                {
                    Console.Error.WriteLine($"\t[DayReport] Failed to match predicted winner for race {race.RaceTitle ?? race.MarketId}.");
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
        private static bool TryConvertToByte(object? value, out byte result)
        {
            switch (value)
            {
                case byte b:
                    result = b;
                    return true;
                case sbyte sb when sb >= byte.MinValue:
                    result = (byte)sb;
                    return true;
                case short s when s >= byte.MinValue && s <= byte.MaxValue:
                    result = (byte)s;
                    return true;
                case ushort us when us <= byte.MaxValue:
                    result = (byte)us;
                    return true;
                case int i when i >= byte.MinValue && i <= byte.MaxValue:
                    result = (byte)i;
                    return true;
                case uint ui when ui <= byte.MaxValue:
                    result = (byte)ui;
                    return true;
                case long l when l >= byte.MinValue && l <= byte.MaxValue:
                    result = (byte)l;
                    return true;
                case ulong ul when ul <= byte.MaxValue:
                    result = (byte)ul;
                    return true;
                case float f when f >= byte.MinValue && f <= byte.MaxValue:
                    result = (byte)Math.Round(f);
                    return true;
                case double d when d >= byte.MinValue && d <= byte.MaxValue:
                    result = (byte)Math.Round(d);
                    return true;
                case decimal m when m >= byte.MinValue && m <= byte.MaxValue:
                    result = (byte)Math.Round(m);
                    return true;
                case string s when byte.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsedInv):
                    result = parsedInv;
                    return true;
                case string s when byte.TryParse(s, NumberStyles.Any, CultureInfo.CurrentCulture, out var parsedCur):
                    result = parsedCur;
                    return true;
                default:
                    result = 0;
                    return false;
            }
        }

        
        private BetfairScrapeResult ScrapeOpenRaceTabsInternal(IWebDriver driver, bool executeBets, bool captureReport)
        {
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10));
            var handles = driver.WindowHandles.ToList();
            var weightPath = ResolveAiWeightPath();
            if (File.Exists(weightPath))
            {
                var info = new FileInfo(weightPath);
                Console.WriteLine($"\tAI weight file located at {weightPath} ({info.Length} bytes, last modified {info.LastWriteTimeUtc:u}).");
            }
            else
            {
                Console.WriteLine($"\tAI weight file missing at {weightPath}; AI probabilities may fall back to defaults.");
            }

            var aiCalculator = new AIOddsCalculator(weightPath);
            Console.WriteLine($"\tAI model status: {aiCalculator.ModelStatus}");
            var result = new BetfairScrapeResult();
            var recommendations = new List<BetRecommendation>();
            foreach (var handle in handles)
            {
                driver.SwitchTo().Window(handle);
                Console.WriteLine($"Processing tab: {driver.Url}");
                if (!driver.Url.Contains("/horse-racing/", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("\tSkipping non-racing tab");
                    continue;
                }

                try
                {
                    // Betfair recently changed the markup for runner rows.  The old
                    // selector looked for elements with a "data-test-id" attribute of
                    // "runner" which no longer exists.  Runner rows are now rendered
                    // with the class "runner-line", but the data-selection-id may be on
                    // a nested element rather than the row itself.  Wait until at least
                    // one element with this class is present.

                    wait.Until(d => d.FindElements(By.CssSelector(".runner-line")).Count > 0);
                }
                catch (WebDriverTimeoutException)
                {
                    Console.Error.WriteLine("\tTimed out waiting for runner rows");
                    continue;
                }

                var marketId = ExtractMarketId(driver.Url);
                if (string.IsNullOrEmpty(marketId))
                {
                    Console.Error.WriteLine($"\tFailed to extract market ID from URL: {driver.Url}");
                    continue;
                }

                var title = TextOrEmpty(driver, By.CssSelector("[data-testid='marketTitle']"));
                var venueText = TextOrEmpty(driver, By.CssSelector(".venue-name"));
                var eventDateText = TextOrEmpty(driver, By.CssSelector(".event-date"));
                var raceDetailsText = TextOrEmpty(driver, By.CssSelector(".market-name"));
                var offTimeText = TextOrEmpty(driver, By.CssSelector("[data-testid='startTime']"));
                var backBookText = TextOrEmpty(driver, By.CssSelector(".rh-back-book-percentage-label"));
                var layBookText = TextOrEmpty(driver, By.CssSelector(".rh-lay-book-percentage-label"));
                TimeSpan? offTime = TimeSpan.TryParse(offTimeText, out var t) ? t : (TimeSpan?)null;
                var (venueTime, venueName, venueCountry) = ParseVenueDetails(venueText);
                if (!offTime.HasValue && venueTime.HasValue)
                {
                    offTime = venueTime;
                }
                var parsedRaceDate = ParseEventDate(eventDateText, DateTime.Today);
                var backBookPercentage = ParsePercentage(backBookText);
                var layBookPercentage = ParsePercentage(layBookText);
                Console.WriteLine($"\tScraping market {marketId} - {title}");

                try
                {
                    var screen = new RaceScreen
                    {
                        MarketId = marketId,
                        OffTime = offTime,
                        Title = title,
                        RaceDate = parsedRaceDate ?? DateTime.Today,
                        VenueName = venueName,
                        VenueCountry = venueCountry,
                        EventDateText = string.IsNullOrWhiteSpace(eventDateText) ? null : eventDateText.Trim(),
                        RaceDetails = string.IsNullOrWhiteSpace(raceDetailsText) ? null : raceDetailsText.Trim(),
                        BackBookPercentage = backBookPercentage,
                        LayBookPercentage = layBookPercentage
                    };
                    lock (_repoLock)
                    {
                        _repo.InsertRaceScreen(screen);
                    }
                    Console.WriteLine($"\tInserted race screen for {marketId}");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tInsertRaceScreen failed for market {marketId}: {ex.Message}");
                    continue;
                }

                // Retrieve the runner rows using the updated selector described above
                // (any element with the runner-line class)
                var rows = driver.FindElements(By.CssSelector(".runner-line"));
                if (rows.Count == 0)
                {
                    Console.Error.WriteLine($"\tNo runner rows found for market {marketId}");
                    continue;
                }
                Console.WriteLine($"\tFound {rows.Count} runners for market {marketId}");
                var featureLookup = LoadFeatureLookup(parsedRaceDate, title, venueName);
                var flows = new List<RunnerFlow>();
                var runnerEntries = new List<(IWebElement Row, RunnerFlow Flow)>();
                foreach (var row in rows)
                {
                    // Some runner rows nest the data-selection-id on a child element, so
                    // fall back to searching within the row if it's missing on the row
                    // itself.
                    var selectionId = row.GetAttribute("data-selection-id");
                    if (string.IsNullOrEmpty(selectionId))
                    {
                        try
                        {
                            var childWithId = row.FindElement(By.CssSelector("[data-selection-id]"));
                            selectionId = childWithId.GetAttribute("data-selection-id");
                        }
                        catch (NoSuchElementException)
                        {
                            selectionId = null;
                        }
                    }
                    var js = (IJavaScriptExecutor)driver;
                    const string runnerExtractionScript = @"
                        const row = arguments[0];
                        const textOrEmpty = el => el && el.textContent ? el.textContent.trim() : '';
                        const queryText = selector => selector ? textOrEmpty(row.querySelector(selector)) : '';

                        const extractPriceText = raw => {
                            if (!raw) {
                                return '';
                            }
                            const text = raw.trim();
                            if (!text) {
                                return '';
                            }
                            if (/^[£€$]/.test(text)) {
                                return '';
                            }
                            return text;
                        };

                        const extractPriceFromButton = raw => {
                            if (!raw) {
                                return '';
                            }
                            const text = raw.trim();
                            if (!text) {
                                return '';
                            }
                            const tokens = text.split(/\s+/);
                            for (const token of tokens) {
                                if (!token || /^[£€$]/.test(token)) {
                                    continue;
                                }
                                if (/^[0-9]+(\.[0-9]+)?$/.test(token)) {
                                    return token;
                                }
                            }
                            return extractPriceText(text);
                        };

                        const indexTokens = {
                            1: ['1', 'one'],
                            2: ['2', 'two'],
                            3: ['3', 'three']
                        };

                        const findOursPriceButton = (type, index) => {
                            const cellIndex = type === 'back' ? 4 : 5;
                            const cell = row.querySelector(`td:nth-of-type(${cellIndex})`);
                            if (cell) {
                                const oursButtons = Array.from(cell.querySelectorAll('ours-price-button'));
                                if (oursButtons.length >= index) {
                                    const button = oursButtons[index - 1];
                                    const priceLabel = button.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price');
                                    if (priceLabel) {
                                        const text = extractPriceText(textOrEmpty(priceLabel));
                                        if (text) {
                                            return text;
                                        }
                                    }

                                    const fallbackButtonText = extractPriceFromButton(textOrEmpty(button.querySelector('button')));
                                    if (fallbackButtonText) {
                                        return fallbackButtonText;
                                    }
                                }
                            }

                            const allButtons = Array.from(row.querySelectorAll('ours-price-button'));
                            const startIndex = type === 'lay' ? 3 : 0;
                            if (allButtons.length >= index + startIndex) {
                                const button = allButtons[startIndex + index - 1];
                                const priceLabel = button.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price');
                                if (priceLabel) {
                                    const text = extractPriceText(textOrEmpty(priceLabel));
                                    if (text) {
                                        return text;
                                    }
                                }

                                const fallbackButtonText = extractPriceFromButton(textOrEmpty(button.querySelector('button')));
                                if (fallbackButtonText) {
                                    return fallbackButtonText;
                                }
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
                            if (oursButtonPrice) {
                                return oursButtonPrice;
                            }

                            const dataTestId = row.querySelector(`[data-testid='runner-${type}-${index}-price']`);
                            if (dataTestId) {
                                const text = textOrEmpty(dataTestId);
                                if (text) {
                                    return text;
                                }
                            }

                            for (const selector of selectors) {
                                const el = row.querySelector(selector);
                                if (el) {
                                    const text = textOrEmpty(el);
                                    if (text) {
                                        return text;
                                    }
                                }
                            }

                            return '';
                        };

                        const result = {
                            cloth: queryText('.runner-number'),
                            draw: queryText('.draw'),
                            horse: queryText('.name .runner-name'),
                            jockey: queryText('.name .jockey-name')
                        };

                        for (let i = 1; i <= 3; i++) {
                            result[`back${i}`] = findPrice('back', i);
                            result[`lay${i}`] = findPrice('lay', i);
                        }

                        const fallbackButtons = Array.from(row.querySelectorAll('bet-button'));
                        const fallbackPrices = fallbackButtons
                             .map(btn => {
                                const label = btn.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price');
                                if (label) {
                                    const priceText = extractPriceText(textOrEmpty(label));
                                    if (priceText) {
                                        return priceText;
                                    }
                                }
                                const buttonText = btn.querySelector('button');
                                return extractPriceFromButton(textOrEmpty(buttonText || btn));
                            })

                        for (let i = 1; i <= 3; i++) {
                            const backKey = `back${i}`;
                            const layKey = `lay${i}`;
                            if (!result[backKey] && fallbackPrices.length >= i) {
                                result[backKey] = fallbackPrices[i - 1];
                            }
                            if (!result[layKey] && fallbackPrices.length >= i + 3) {
                                result[layKey] = fallbackPrices[i + 2];
                            }
                        }

                        return result;
                    ";

                    var elementData = (IDictionary<string, object>)js.ExecuteScript(runnerExtractionScript, row);

                    string Get(string key) => elementData.TryGetValue(key, out var v) ? v?.ToString() ?? string.Empty : string.Empty;
                    var flow = new RunnerFlow
                    {
                        MarketId = marketId,
                        SelectionId = selectionId,
                        ClothNumber = TryParseByte(Get("cloth")),
                        Draw = TryParseByte(Get("draw")),
                        HorseName = Get("horse"),
                        JockeyName = Get("jockey"),
                        BackPrice1 = ParseDecimal(Get("back1")),
                        BackPrice2 = ParseDecimal(Get("back2")),
                        BackPrice3 = ParseDecimal(Get("back3")),
                        LayPrice1 = ParseDecimal(Get("lay1")),
                        LayPrice2 = ParseDecimal(Get("lay2")),
                        LayPrice3 = ParseDecimal(Get("lay3"))
                    };
                    var matchedFeatures = featureLookup.FindByHorse(flow.HorseName)
                        ?? featureLookup.FindBySaddlecloth(flow.ClothNumber);
                    flow.HasPreparedFeatures = matchedFeatures != null;
                    var featureVector = matchedFeatures != null
                        ? new Dictionary<string, object?>(matchedFeatures, StringComparer.OrdinalIgnoreCase)
                        : new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);

                    if (rows.Count > 0)
                    {
                        featureVector["RunnerCount"] = rows.Count;
                    }

                    if (flow.BackPrice1.HasValue)
                    {
                        featureVector["BackPrice1"] = flow.BackPrice1.Value;
                    }
                    if (flow.BackPrice2.HasValue)
                    {
                        featureVector["BackPrice2"] = flow.BackPrice2.Value;
                    }
                    if (flow.BackPrice3.HasValue)
                    {
                        featureVector["BackPrice3"] = flow.BackPrice3.Value;
                    }
                    if (flow.LayPrice1.HasValue)
                    {
                        featureVector["LayPrice1"] = flow.LayPrice1.Value;
                    }
                    if (flow.LayPrice2.HasValue)
                    {
                        featureVector["LayPrice2"] = flow.LayPrice2.Value;
                    }
                    if (flow.LayPrice3.HasValue)
                    {
                        featureVector["LayPrice3"] = flow.LayPrice3.Value;
                    }

                    if (featureVector.Count > 0)
                    {
                        flow.FeatureValues = featureVector;
                    }
                    if (!flow.HasPreparedFeatures)
                    {
                        var missingFeatureIdentifier = !string.IsNullOrWhiteSpace(flow.HorseName)
                            ? flow.HorseName!
                            : (flow.SelectionId ?? "unknown");
                        Console.WriteLine(
                            $"\t\tNo prepared feature row matched for {missingFeatureIdentifier}; " +
                            "neural model will fall back to legacy odds.");
                    }
                    try
                    {
                        var probability = aiCalculator.CalculateOdds(flow);
                        if (double.IsFinite(probability) && probability >= 0)
                        {
                            flow.AiOdds = probability;
                        }
                        else
                        {
                            flow.AiOdds = null;
                            Console.Error.WriteLine($"\tInvalid AI odds calculated for selection {selectionId ?? "unknown"} in market {marketId}");
                        }
                    }
                    catch (Exception ex)
                    {
                        flow.AiOdds = null;
                        Console.Error.WriteLine($"\tFailed to calculate AI odds for selection {selectionId ?? "unknown"} in market {marketId}: {ex.Message}");
                    }
                    var identifier = !string.IsNullOrWhiteSpace(flow.HorseName)
                        ? flow.HorseName!
                        : (flow.SelectionId ?? "unknown");
                    var aiText = flow.AiOdds.HasValue
                        ? flow.AiOdds.Value.ToString("0.####", CultureInfo.InvariantCulture)
                        : "null";
                    var backText = flow.BackPrice1.HasValue
                        ? flow.BackPrice1.Value.ToString("0.##", CultureInfo.InvariantCulture)
                        : "null";
                    Console.WriteLine($"\tRunner snapshot {identifier}: back1={backText}, aiProbabilityRaw={aiText}");
                    if (!flow.AiOdds.HasValue)
                    {
                        Console.WriteLine($"\t\tAI probability missing for {identifier}; downstream filters will treat this runner as zero edge.");
                    }
                    if (!flow.BackPrice1.HasValue)
                    {
                        Console.WriteLine($"\t\tNo back price available for {identifier}; cannot compare against market probability.");
                    }


                    flows.Add(flow);
                    runnerEntries.Add((row, flow));
                }
                var validAiBefore = flows
                    .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                    .Select(f => f.AiOdds!.Value)
                    .ToList();
                var missingAiCount = flows.Count - validAiBefore.Count;
                if (validAiBefore.Count == 0)
                {
                    Console.WriteLine($"\tAll {flows.Count} runner(s) in market {marketId} are missing AI probabilities before normalization.");
                }
                else
                {
                    var sumProb = validAiBefore.Sum();
                    var minProb = validAiBefore.Min();
                    var maxProb = validAiBefore.Max();
                    Console.WriteLine($"\tAI probability summary before normalization for market {marketId}: valid={validAiBefore.Count}, missing={missingAiCount}, sum={sumProb.ToString("0.####", CultureInfo.InvariantCulture)}, min={minProb.ToString("0.####", CultureInfo.InvariantCulture)}, max={maxProb.ToString("0.####", CultureInfo.InvariantCulture)}");
                }
                NormalizeAiOdds(flows, _useMarketFallbackForAiDegeneracy);
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
                        flows);
                    result.Races.Add(report);
                }
                var raceRecommendations = CreateRecommendations(flows, marketId, title, venueName, parsedRaceDate)
                    .OrderByDescending(r => r.Differential)
                    .ThenByDescending(r => r.KellyFraction)
                    .ToList();
                Console.WriteLine($"\t{raceRecommendations.Count} runner(s) passed value filters for market {marketId}.");
                if (!executeBets)
                {
                    if (raceRecommendations.Count > 0)
                    {
                        Console.WriteLine("\tReport mode: positive expected value runner(s) identified; skipping bet execution.");
                    }
                    else
                    {
                        Console.WriteLine($"\tNo positive value opportunity identified for market {marketId}");
                    }
                }
                else
                {
                    if (raceRecommendations.Count > 1)
                    {
                        Console.WriteLine("\t\tMultiple runners qualified in the same market; sequential Kelly stakes will size each independently in tab order.");
                    }
                    if (raceRecommendations.Count > 0)
                    {
                        var bankrollBeforeClicks = _availableBankroll;
                        var clickedRecommendations = ExecuteBackAllClicks(driver, runnerEntries, raceRecommendations);

                        if (clickedRecommendations.Count > 0)
                        {
                            var top = clickedRecommendations.First();
                            Console.WriteLine($"\tKelly stake {top.Stake.ToString("0.##", CultureInfo.InvariantCulture)} on {top.HorseName ?? "unknown"} (diff {top.Differential.ToString("0.####", CultureInfo.InvariantCulture)})");
                            recommendations.AddRange(clickedRecommendations);
                            PopulateBetSlipStakes(driver, clickedRecommendations, bankrollBeforeClicks);
                        }
                        else
                        {
                            Console.WriteLine("\tNo qualifying Back-All clicks were executed for this market");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"\tNo positive value opportunity identified for market {marketId}");
                    }
                }

                foreach (var flow in flows)
                {
                    try
                    {
                        lock (_repoLock)
                        {
                            _repo.InsertRunnerFlow(flow);
                        }
                        Console.WriteLine($"\tInserted runner {flow.SelectionId} for market {marketId}");
                    }
                    catch (Exception ex)
                    {
                        var selId = flow.SelectionId ?? "unknown";
                        Console.Error.WriteLine($"\tInsertRunnerFlow failed for market {marketId}, selection {selId}: {ex.Message}");
                    }
                }
            }
            result.Recommendations.AddRange(recommendations
                .OrderByDescending(r => r.Differential)
                .ThenByDescending(r => r.KellyFraction));
            return result;
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

        private sealed class BetfairScrapeResult
        {
            public List<BetRecommendation> Recommendations { get; } = new();
            public List<RaceDayReport> Races { get; } = new();
        }

        public IReadOnlyList<BetRecommendation> ScrapeOpenRaceTabs(IWebDriver driver)
        {
            return ScrapeOpenRaceTabsInternal(driver, executeBets: true, captureReport: false).Recommendations;
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
        private FeatureLookup LoadFeatureLookup(DateTime? raceDate, string? raceTitle, string? venueName)
        {
            if (!raceDate.HasValue)
            {
                return FeatureLookup.Empty;
            }

            int? raceId;
            try
            {
                raceId = _repo.FindRaceId(raceDate.Value, raceTitle, venueName);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tUnable to resolve race details for feature lookup: {ex.Message}");
                return FeatureLookup.Empty;
            }

            if (!raceId.HasValue)
            {
                return FeatureLookup.Empty;
            }

            try
            {
                var include = new HashSet<int> { raceId.Value };
                var prepared = _trainer.PrepareDataset(include, null, includeIdentifiers: true);
                var race = prepared.Races.FirstOrDefault(r => r.RaceId == raceId.Value);
                if (race == null)
                {
                    return FeatureLookup.Empty;
                }

                return FeatureLookup.FromPreparedRace(race);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tFailed to build feature vector for race {raceId.Value}: {ex.Message}");
                return FeatureLookup.Empty;
            }
        }

        private static string NormalizeName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var normalized = Regex.Replace(value.Trim().ToLowerInvariant(), "[^a-z0-9]+", " ");
            normalized = Regex.Replace(normalized, "\\s+", " ").Trim();
            return normalized;
        }

        private sealed class FeatureLookup
        {
            private readonly Dictionary<string, Dictionary<string, object?>> _byHorse;
            private readonly Dictionary<int, Dictionary<string, object?>> _bySaddlecloth;

            private FeatureLookup(
                Dictionary<string, Dictionary<string, object?>> byHorse,
                Dictionary<int, Dictionary<string, object?>> bySaddlecloth)
            {
                _byHorse = byHorse;
                _bySaddlecloth = bySaddlecloth;
            }

            public static FeatureLookup Empty { get; } = new FeatureLookup(
                new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase),
                new Dictionary<int, Dictionary<string, object?>>());

            public static FeatureLookup FromPreparedRace(PreparedRace race)
            {
                var byHorse = new Dictionary<string, Dictionary<string, object?>>(StringComparer.OrdinalIgnoreCase);
                var byCloth = new Dictionary<int, Dictionary<string, object?>>();

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

                    if (PreparedDataset.TryGetRequiredInt32(row, "SaddleclothNumber", out var saddlecloth))
                    {
                        byCloth[saddlecloth] = copy;
                    }
                }

                return new FeatureLookup(byHorse, byCloth);
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
        private static decimal CalculateSequentialStake(decimal bankroll, decimal kellyFraction)
        {
            if (bankroll <= 0m || kellyFraction <= 0m)
            {
                return 0m;
            }

            var stake = bankroll * kellyFraction;
            if (stake <= 0m)
            {
                return 0m;
            }

            if (stake > bankroll)
            {
                stake = bankroll;
            }

            stake = decimal.Round(stake, 2, MidpointRounding.ToZero);

            return stake;
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

        private static IReadOnlyList<IWebElement> FindBetSlipStakeInputs(IWebDriver driver)
        {
            var selectors = new[]
            {
                "input.betslip-size-input",
                "input[data-testid='betslip-stake-input']",
                "input[data-testid='bet-slip-input']",
                "input[data-testid='size-input']",
                "div.betslip-selection input[type='text']"
            };

            foreach (var selector in selectors)
            {
                try
                {
                    var found = driver
                        .FindElements(By.CssSelector(selector))
                        .Where(e => e.Displayed)
                        .ToList();

                    if (found.Count > 0)
                    {
                        return found;
                    }
                }
                catch (NoSuchElementException)
                {
                }
            }

            return Array.Empty<IWebElement>();
        }

        private static void SetStakeInputValue(IJavaScriptExecutor js, IWebElement input, decimal stake)
        {
            var text = stake > 0m
                ? stake.ToString("0.##", CultureInfo.InvariantCulture)
                : "0";

            js.ExecuteScript(@"const el = arguments[0];
                const value = arguments[1];
                if (!el) { return; }
                el.focus();
                el.value = value;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            ", input, text);
        }
        private static bool TryClickBackAllViaScript(IJavaScriptExecutor js, IWebElement row)
        {
            try
            {
                var result = js.ExecuteScript(@"const runnerRow = arguments[0];
                    if (!runnerRow) { return false; }
                    const toLower = el => (el.textContent || '').trim().toLowerCase();
                    const buttons = Array.from(runnerRow.querySelectorAll('button, [role=""button""], .bet-button'));
                    for (const btn of buttons) {
                        if (toLower(btn).includes('back all')) {
                            btn.scrollIntoView({block: 'center'});
                            btn.click();
                            return true;
                        }
                    }
                    const fallback = Array.from(runnerRow.querySelectorAll('*')).find(el => toLower(el).includes('back all'));
                    if (fallback) {
                        fallback.scrollIntoView({block: 'center'});
                        fallback.click();
                        return true;
                    }
                    return false;", row);

                return result is bool success && success;
            }
            catch (Exception)
            {
                return false;
            }
        }
        private static IWebElement? TryFindElement(ISearchContext context, By by)
        {
            try
            {
                return context.FindElement(by);
            }
            catch (NoSuchElementException)
            {
                return null;
            }
            catch (StaleElementReferenceException)
            {
                return null;
            }
        }
        private static decimal CalculateAiDecimalOdds(double aiProbability)
        {
            if (aiProbability <= 0)
            {
                return 0m;
            }

            var inverted = 1.0 / aiProbability;

            if (!double.IsFinite(inverted) || inverted <= 0)
            {
                return 0m;
            }

            if (inverted > (double)decimal.MaxValue)
            {
                return decimal.MaxValue;
            }

            return (decimal)inverted;
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
        private static void NormalizeAiOdds(ICollection<RunnerFlow> flows, bool useMarketFallbackForDegeneracy)
        {
            var valid = flows
                .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                .ToList();
            var totalRunners = flows.Count;
            var missingCount = totalRunners - valid.Count;

            if (valid.Count == 0)
            {
                if (totalRunners > 0)
                {
                    Console.WriteLine($"\tNormalizeAiOdds: no valid AI probabilities across {totalRunners} runner(s); skipping normalization.");
                }
                return;
            }
            const double degeneracyTolerance = 1e-8;
            double minValue = valid.Min(f => f.AiOdds!.Value);
            double maxValue = valid.Max(f => f.AiOdds!.Value);

            if (maxValue - minValue <= degeneracyTolerance)
            {
                Console.WriteLine("\t\tDetected degenerate AI probability distribution; raw outputs are identical across runners.");

                if (TryResolveDegenerateDistribution(flows))
                {
                    return;
                }

                if (!useMarketFallbackForDegeneracy)
                {
                    Console.WriteLine("\t\tPreserving raw AI outputs (market fallback disabled).");
                }
                else
                {
                    Console.WriteLine("\t\tFalling back to market-implied probabilities.");

                    bool anyFallbackApplied = false;
                    foreach (var flow in flows)
                    {
                        if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 1m)
                        {
                            flow.AiOdds = 1.0 / (double)flow.BackPrice1.Value;
                            anyFallbackApplied = true;
                        }
                        else
                        {
                            flow.AiOdds = null;
                        }
                    }

                    if (!anyFallbackApplied)
                    {
                        Console.WriteLine("\t\tUnable to apply market fallback due to missing back prices; AI odds will remain unavailable for this race.");
                        return;
                    }

                    valid = flows
                        .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                        .ToList();
                    missingCount = flows.Count - valid.Count;
                    if (valid.Count == 0)
                    {
                        Console.WriteLine("\t\tMarket fallback produced no usable probabilities; skipping normalization.");
                        return;
                    }

                    minValue = valid.Min(f => f.AiOdds!.Value);
                    maxValue = valid.Max(f => f.AiOdds!.Value);
                }
            }
        NormalizationSummary:
            var sum = valid.Sum(f => f.AiOdds!.Value);
            Console.WriteLine($"\tNormalizeAiOdds: normalizing {valid.Count} runner(s) (missing={missingCount}, raw sum={sum.ToString("0.####", CultureInfo.InvariantCulture)}).");
            if (missingCount > 0)
            {
                Console.WriteLine($"\t\tMissing AI predictions detected for {missingCount} runner(s); surviving probabilities may be inflated relative to the full field.");
            }

            if (sum <= double.Epsilon)
            {
                var uniform = 1.0 / valid.Count;
                foreach (var flow in valid)
                {
                    flow.AiOdds = uniform;
                }
                Console.WriteLine($"\t\tRaw probabilities summed to ~0; distributing uniform probability {uniform.ToString("0.####", CultureInfo.InvariantCulture)} across valid runners.");
                return;
            }

            foreach (var flow in valid)
            {
                flow.AiOdds = Math.Max(flow.AiOdds!.Value / sum, 0);
            }

            var normalizedSum = valid.Sum(f => f.AiOdds!.Value);
            Console.WriteLine($"\t\tNormalized probability sum: {normalizedSum.ToString("0.####", CultureInfo.InvariantCulture)}.");
        }
        private static bool TryResolveDegenerateDistribution(ICollection<RunnerFlow> flows)
        {
            if (flows is null || flows.Count == 0)
            {
                return false;
            }

            var winner = ResolveLikelyWinnerFromFeatures(flows);
            if (winner == null || !winner.AiOdds.HasValue || !double.IsFinite(winner.AiOdds.Value) || winner.AiOdds.Value <= 0)
            {
                return false;
            }

            var winnerProbability = winner.AiOdds.Value;
            var others = flows.Where(f => !ReferenceEquals(f, winner)).ToList();
            var leftoverMass = Math.Max(1.0 - winnerProbability, 0);

            if (others.Count > 0)
            {
                var weights = new Dictionary<RunnerFlow, double>();
                foreach (var flow in others)
                {
                    double weight = 0d;

                    if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 1m)
                    {
                        weight = 1.0 / (double)flow.BackPrice1.Value;
                    }
                    else if (flow.AiOdds.HasValue && double.IsFinite(flow.AiOdds.Value) && flow.AiOdds.Value > 0)
                    {
                        weight = flow.AiOdds.Value;
                    }
                    else
                    {
                        weight = 1d;
                    }

                    if (!double.IsFinite(weight) || weight < 0)
                    {
                        weight = 0d;
                    }

                    weights[flow] = weight;
                }

                var weightSum = weights.Values.Sum();
                if (weightSum <= double.Epsilon)
                {
                    var uniform = others.Count > 0 ? leftoverMass / others.Count : 0d;
                    foreach (var flow in others)
                    {
                        flow.AiOdds = uniform;
                    }
                }
                else
                {
                    foreach (var kvp in weights)
                    {
                        var share = leftoverMass * (kvp.Value / weightSum);
                        kvp.Key.AiOdds = Math.Max(share, 0d);
                    }
                }
            }

            winner.AiOdds = winnerProbability;

            var validAfter = flows
                .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                .ToList();
            var missingAfter = flows.Count - validAfter.Count;
            if (missingAfter > 0)
            {
                Console.WriteLine($"\t\tMissing AI predictions detected for {missingAfter} runner(s); surviving probabilities may be inflated relative to the full field.");
            }

            var winnerName = !string.IsNullOrWhiteSpace(winner.HorseName)
                ? winner.HorseName!
                : (winner.SelectionId ?? "unknown");

            Console.WriteLine($"\t\tInterpreting degenerate predictions as winner-only probability; assigning {winnerProbability.ToString("0.####", CultureInfo.InvariantCulture)} to {winnerName}.");

            if (others.Count > 0 && leftoverMass > 0)
            {
                Console.WriteLine($"\t\tRedistributed remaining {leftoverMass.ToString("0.####", CultureInfo.InvariantCulture)} probability mass across {others.Count} runner(s) using market-derived weights.");
            }

            var normalized = validAfter.Sum(f => f.AiOdds!.Value);
            Console.WriteLine($"\t\tNormalized probability sum: {normalized.ToString("0.####", CultureInfo.InvariantCulture)}.");

            return true;
        }

        private static RunnerFlow? ResolveLikelyWinnerFromFeatures(IEnumerable<RunnerFlow> flows)
        {
            if (flows is null)
            {
                return null;
            }

            var flowList = flows.ToList();
            if (flowList.Count == 0)
            {
                return null;
            }

            var booleanKeys = new[] { "AiLikelyWinner", "LikelyWinner", "PredictedWinner", "IsAiWinner" };
            foreach (var flow in flowList)
            {
                if (flow.FeatureValues == null)
                {
                    continue;
                }

                foreach (var key in booleanKeys)
                {
                    if (IsFeatureTrue(flow.FeatureValues, key))
                    {
                        return flow;
                    }
                }
            }

            var selectionHint = FindFirstFeatureString(flowList, "AiLikelyWinnerSelectionId", "LikelyWinnerSelectionId", "PredictedWinnerSelectionId");
            if (!string.IsNullOrWhiteSpace(selectionHint))
            {
                var match = flowList.FirstOrDefault(f => !string.IsNullOrWhiteSpace(f.SelectionId) && string.Equals(f.SelectionId, selectionHint, StringComparison.OrdinalIgnoreCase));
                if (match != null)
                {
                    return match;
                }
            }

            var horseIdHint = FindFirstFeatureString(flowList, "AiLikelyWinnerHorseId", "LikelyWinnerHorseId", "PredictedWinnerHorseId");
            if (!string.IsNullOrWhiteSpace(horseIdHint))
            {
                foreach (var flow in flowList)
                {
                    if (flow.FeatureValues != null && flow.FeatureValues.TryGetValue("HorseId", out var horseIdObj) && horseIdObj != null)
                    {
                        var horseIdText = ConvertToInvariantString(horseIdObj);
                        if (!string.IsNullOrWhiteSpace(horseIdText) && string.Equals(horseIdText, horseIdHint, StringComparison.OrdinalIgnoreCase))
                        {
                            return flow;
                        }
                    }
                }
            }

            var horseNameHint = FindFirstFeatureString(flowList, "AiLikelyWinnerHorse", "AiLikelyWinnerHorseName", "LikelyWinnerHorse", "LikelyWinnerHorseName", "PredictedWinnerHorse", "PredictedWinnerHorseName");
            if (!string.IsNullOrWhiteSpace(horseNameHint))
            {
                var match = flowList.FirstOrDefault(f => !string.IsNullOrWhiteSpace(f.HorseName) && string.Equals(f.HorseName, horseNameHint, StringComparison.OrdinalIgnoreCase));
                if (match != null)
                {
                    return match;
                }
            }

            return null;
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

        private static bool IsFeatureTrue(Dictionary<string, object?> features, string key)
        {
            if (!features.TryGetValue(key, out var value) || value is null)
            {
                return false;
            }

            switch (value)
            {
                case bool b:
                    return b;
                case string s:
                    var trimmed = s.Trim();
                    if (bool.TryParse(trimmed, out var parsed))
                    {
                        return parsed;
                    }

                    if (double.TryParse(trimmed, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var numericFromString))
                    {
                        return Math.Abs(numericFromString) > double.Epsilon;
                    }

                    return false;
                default:
                    if (value is IConvertible convertible)
                    {
                        try
                        {
                            return convertible.ToBoolean(CultureInfo.InvariantCulture);
                        }
                        catch
                        {
                            try
                            {
                                var numeric = convertible.ToDouble(CultureInfo.InvariantCulture);
                                return Math.Abs(numeric) > double.Epsilon;
                            }
                            catch
                            {
                                return false;
                            }
                        }
                    }

                    return false;
            }
        }

        private static string? ConvertToInvariantString(object value)
        {
            return value switch
            {
                null => null,
                string s => s,
                IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
                _ => value.ToString()
            };
        }
        private static decimal? ParsePercentage(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var cleaned = text.Trim();
            if (cleaned.EndsWith("%", StringComparison.Ordinal))
            {
                cleaned = cleaned[..^1];
            }

            cleaned = cleaned.Replace("%", string.Empty);

            return ParseDecimal(cleaned);
        }

        private static (TimeSpan? Time, string? Venue, string? Country) ParseVenueDetails(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return (null, null, null);
            }

            var trimmed = text.Trim();
            var match = Regex.Match(trimmed, @"^(?<time>\d{1,2}:\d{2})\s+(?<venue>.*?)(?:\s+\((?<country>[A-Z]{2,})\))?$");

            if (match.Success)
            {
                TimeSpan? parsedTime = null;
                if (TimeSpan.TryParse(match.Groups["time"].Value, out var ts))
                {
                    parsedTime = ts;
                }

                var venueName = match.Groups["venue"].Value.Trim();
                var country = match.Groups["country"].Success ? match.Groups["country"].Value.Trim() : null;

                return (parsedTime, string.IsNullOrWhiteSpace(venueName) ? null : venueName, country);
            }

            return (null, trimmed, null);
        }

        private static DateTime? ParseEventDate(string text, DateTime today)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var trimmed = text.Trim();
            var tokens = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            string? dayToken = null;
            string? monthToken = null;
            string? yearToken = null;

            foreach (var token in tokens)
            {
                var clean = token.Trim();
                if (dayToken == null && int.TryParse(clean, NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
                {
                    dayToken = clean;
                    continue;
                }

                if (dayToken != null && monthToken == null)
                {
                    monthToken = clean.TrimEnd(',', '.');
                    continue;
                }

                if (dayToken != null && monthToken != null && yearToken == null && int.TryParse(clean, NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
                {
                    yearToken = clean;
                    break;
                }
            }

            if (dayToken != null && monthToken != null)
            {
                var hasExplicitYear = int.TryParse(yearToken, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedYear);
                var candidateYear = hasExplicitYear ? parsedYear : today.Year;
                var formats = new[] { "d MMM yyyy", "dd MMM yyyy", "d MMMM yyyy", "dd MMMM yyyy" };

                foreach (var format in formats)
                {
                    if (DateTime.TryParseExact($"{dayToken} {monthToken} {candidateYear}", format, CultureInfo.InvariantCulture, DateTimeStyles.None, out var candidate))
                    {
                        candidate = candidate.Date;
                        if (!hasExplicitYear)
                        {
                            if (candidate < today.AddDays(-7))
                            {
                                candidate = candidate.AddYears(1);
                            }
                            else if (candidate > today.AddDays(200))
                            {
                                candidate = candidate.AddYears(-1);
                            }
                        }
                        return candidate;
                    }
                }
            }

            if (DateTime.TryParse(trimmed, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out var fallback))
            {
                return fallback.Date;
            }

            return null;
        }
        private static string TextOrEmpty(IWebDriver d, By by)
        {
            try { return d.FindElement(by).Text.Trim(); }
            catch { return string.Empty; }
        }

        private static string SafeText(IWebElement e, By by)
        {
            try { return e.FindElement(by).Text.Trim(); }
            catch { return string.Empty; }
        }

        private static decimal? ParseDecimal(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var normalized = text
                .Replace('\u00a0', ' ')
                .Replace(",", string.Empty)
                .Trim();

            if (string.IsNullOrEmpty(normalized))
            {
                return null;
            }

            var priceMatch = Regex.Match(normalized, @"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])");
            if (priceMatch.Success
                && decimal.TryParse(priceMatch.Groups[1].Value, NumberStyles.Number, CultureInfo.InvariantCulture, out var decimalOdds))
            {
                return decimalOdds;
            }

            if (normalized.Contains('/'))
            {
                var slashParts = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                if (slashParts.Length == 2
                    && int.TryParse(slashParts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var numerator)
                    && int.TryParse(slashParts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var denominator)
                    && denominator != 0)
                {
                    var fractional = (decimal)numerator / denominator;
                    return fractional + 1m;
                }
            }

            return null;
        }

        private static byte? TryParseByte(string text)
        {
            return byte.TryParse(text, out var v) ? v : (byte?)null;
        }

        private static string? ExtractMarketId(string url)
        {
            // Betfair have changed their URL structure over time.  In some cases the
            // market id appears in a traditional "market/1.234" or "marketId="
            // format, but newer "plus" pages render it as "...-betting-123456".
            // Support both patterns so scraping works regardless of the style of URL
            // that is loaded.
            var m = Regex.Match(url,
                @"(?:/market/|marketId=)(?<id1>[0-9.]+)|(?:betting-)(?<id2>\d+)");

            if (m.Groups["id1"].Success)
            {
                return m.Groups["id1"].Value;
            }

            if (m.Groups["id2"].Success)
            {
                return m.Groups["id2"].Value;
            }

            return null;
        }
    
}
}