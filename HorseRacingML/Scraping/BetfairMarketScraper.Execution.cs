using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using OneOf.Types;
using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        private string? GetGoingForMarket(string? marketId)
        {
            if (string.IsNullOrWhiteSpace(marketId) || _raceGoingLookup == null)
            {
                return null;
            }

            var key = marketId.Trim();
            if (key.Length == 0)
            {
                return null;
            }

            if (_raceGoingLookup.TryGetValue(key, out var going))
            {
                var trimmed = going?.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    return trimmed;
                }
            }

            return null;
        }
        private string? GetGoingForVenue(string? venue)
        {
            if (string.IsNullOrWhiteSpace(venue) || _raceGoingByVenueLookup == null)
            {
                return null;
            }

            var normalized = NormalizeVenueName(venue);
            if (string.IsNullOrEmpty(normalized))
            {
                return null;
            }

            if (_raceGoingByVenueLookup.TryGetValue(normalized, out var going))
            {
                var trimmed = going?.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    return trimmed;
                }
            }

            return null;
        }
        private BetfairScrapeResult ScrapeOpenRaceTabsInternal(IWebDriver driver, bool executeBets, bool captureReport, IEnumerable<string>? handlesToProcess = null)
        {
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10)); // short explicit wait
            var handles = handlesToProcess?.ToList() ?? driver.WindowHandles.ToList(); // collect tab handles
            var orderedHandles = OrderHandlesByScheduledStart(driver, handles);
            Console.WriteLine($"[DayReport][Stage] Preparing to process {orderedHandles.Count} open tab(s) for the day report.");
            var weightPath = ResolveAiWeightPath(); // resolve AI weights path
            _loadedHyperparameters = null;
            AIOddsCalculator? aiCalculator = null;
            _neuralFeatureKeys = null;
            if (_computeAiProbabilities)
            {
                Console.WriteLine("[DayReport][Stage] Initializing AI odds calculator and loading model weights.");
                if (File.Exists(weightPath))
                {
                    var info = new FileInfo(weightPath); // file info for logging
                    Console.WriteLine($"\tAI weight file located at {weightPath} ({info.Length} bytes, last modified {info.LastWriteTimeUtc:u})."); // status
                }
                else
                {
                    Console.WriteLine($"\tAI weight file missing at {weightPath}; AI probabilities may fall back to defaults."); // warn missing weights
                }

                aiCalculator = new AIOddsCalculator(weightPath); // init AI calc
                _neuralFeatureKeys = aiCalculator.FeatureKeys?.ToArray();
                UpdateNeuralFeatureKeys(aiCalculator.GetRawFeatureKeys());
                Console.WriteLine($"\tAI model status: {aiCalculator.ModelStatus}"); // log model status
                _loadedHyperparameters = aiCalculator.Hyperparameters;
            }
            else
            {
                Console.WriteLine("\tAI probability recalculation disabled for this scrape; skipping model load.");
                UpdateNeuralFeatureKeys(null);
                _neuralFeatureKeys = null;
            }
            lock (_featureLookupCacheLock)
            {
                _featureLookupCache.Clear();
                _preloadedPreparedRaces.Clear();
            }
            var result = new BetfairScrapeResult(); // aggregate result
            var recommendations = new List<BetRecommendation>(); // all bet recs
            var processedMarketIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var pendingEvaluations = executeBets ? null : new List<PendingRaceEvaluation>();
            var totalHandles = orderedHandles.Count;
            var handleIndex = 0;
            foreach (var handle in orderedHandles)
            {
                handleIndex++;
                driver.SwitchTo().Window(handle); // switch tab
                Console.WriteLine($"[DayReport][Stage] [{handleIndex}/{totalHandles}] Processing tab: {driver.Url}"); // log url
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
                if (!processedMarketIds.Add(marketId))
                {
                    Console.WriteLine($"\tSkipping market {marketId}: already processed in this scrape session.");
                    continue; // avoid scraping the same race twice when duplicate tabs are open
                }
                var goingText = GetGoingForMarket(marketId);
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
                if (string.IsNullOrWhiteSpace(goingText) && !string.IsNullOrWhiteSpace(venueText))
                {
                    var venueFallback = GetGoingForVenue(venueText);
                    if (!string.IsNullOrWhiteSpace(venueFallback))
                    {
                        goingText = venueFallback;
                    }
                }
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
                var originalRaceDetails = string.IsNullOrWhiteSpace(raceDetailsText) ? null : raceDetailsText.Trim();
                var (raceTypeText, cleanedRaceDetails) = SplitRaceTypeFromDetails(originalRaceDetails);
                var trimmedTitle = string.IsNullOrWhiteSpace(title) ? null : title.Trim();
                var trimmedRaceDetails = string.IsNullOrWhiteSpace(cleanedRaceDetails) ? null : cleanedRaceDetails.Trim();
                var trimmedRaceType = string.IsNullOrWhiteSpace(raceTypeText) ? null : raceTypeText.Trim();
                var trimmedGoing = string.IsNullOrWhiteSpace(goingText) ? null : goingText.Trim();
                var trimmedVenueName = string.IsNullOrWhiteSpace(venueName) ? null : venueName.Trim();
                var trimmedVenueCountry = string.IsNullOrWhiteSpace(venueCountry) ? null : venueCountry.Trim();
                string? scheduleGoing = null;
                if (!string.IsNullOrWhiteSpace(trimmedVenueName))
                {
                    scheduleGoing = GetGoingForVenue(trimmedVenueName);
                }

                if (string.IsNullOrWhiteSpace(scheduleGoing) &&
                    !string.IsNullOrWhiteSpace(venueName) &&
                    !string.Equals(venueName, trimmedVenueName, StringComparison.Ordinal))
                {
                    scheduleGoing = GetGoingForVenue(venueName);
                }

                if (!string.IsNullOrWhiteSpace(scheduleGoing))
                {
                    scheduleGoing = scheduleGoing.Trim();
                }

                var metadataSource = new RaceDayReport
                {
                    RaceTitle = trimmedTitle,
                    RaceDetails = trimmedRaceDetails,
                    RaceType = trimmedRaceType,
                    Going = string.IsNullOrWhiteSpace(trimmedGoing) ? scheduleGoing : trimmedGoing
                };

                ParsedRaceMetadata? parsedMetadata = null;
                try
                {
                    parsedMetadata = ParseRaceMetadata(metadataSource);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to parse race metadata for market {marketId}: {ex.Message}");
                }

                var metadataGoing = parsedMetadata?.Going;
                if (!string.IsNullOrWhiteSpace(metadataGoing))
                {
                    metadataGoing = metadataGoing.Trim();
                }

                var resolvedGoing = !string.IsNullOrWhiteSpace(trimmedGoing)
                    ? trimmedGoing
                    : (!string.IsNullOrWhiteSpace(scheduleGoing)
                        ? scheduleGoing
                        : metadataGoing);

                var goingSource = !string.IsNullOrWhiteSpace(trimmedGoing)
                    ? "race page"
                    : (!string.IsNullOrWhiteSpace(scheduleGoing)
                        ? "schedule venue lookup"
                        : (!string.IsNullOrWhiteSpace(metadataGoing) ? "parsed metadata" : "unavailable"));

                var goingDisplay = string.IsNullOrWhiteSpace(resolvedGoing) ? "<null>" : resolvedGoing;
                try
                {
                    var screen = new RaceScreen
                    {
                        MarketId = marketId, // ids
                        OffTime = offTime, // off time
                        Title = trimmedTitle ?? title, // title
                        RaceDate = parsedRaceDate ?? DateTime.Today, // date
                        VenueName = trimmedVenueName ?? venueName, // venue
                        VenueCountry = trimmedVenueCountry ?? venueCountry, // country
                        EventDateText = string.IsNullOrWhiteSpace(eventDateText) ? null : eventDateText.Trim(), // raw text
                        RaceDetails = trimmedRaceDetails,
                        RaceType = trimmedRaceType,
                        Going = resolvedGoing,
                        BackBookPercentage = backBookPercentage, // %
                        LayBookPercentage = layBookPercentage, // %
                        RaceUrl = string.IsNullOrWhiteSpace(raceUrl) ? null : raceUrl.Trim() // url
                    }; // init screen row

                    lock (_repoLock) { _repo.InsertRaceScreen(screen); } // persist
                    Console.WriteLine($"\tInserted race screen for {marketId} with going '{goingDisplay}'."); // log ok
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
                ExpandRunnerTimeformDetails(driver, rows); // ensure details expanded for all runners
                if (!string.IsNullOrWhiteSpace(_scheduleRegion))
                {
                    trimmedVenueCountry = _scheduleRegion;
                    venueCountry = _scheduleRegion;
                }
                var runnerCount = rows.Count > 0
                    ? (byte)Math.Min(rows.Count, byte.MaxValue)
                    : (byte?)null;

                UpcomingRace? persistedUpcoming = null;
                if (parsedRaceDate.HasValue)
                {
                    try
                    {
                        var metadata = parsedMetadata ?? ParseRaceMetadata(metadataSource);
                        parsedMetadata = metadata;
                        metadataGoing = string.IsNullOrWhiteSpace(metadata.Going) ? null : metadata.Going.Trim();
                        if (string.IsNullOrWhiteSpace(resolvedGoing) && !string.IsNullOrWhiteSpace(metadataGoing))
                        {
                            resolvedGoing = metadataGoing;
                            goingSource = "parsed metadata";
                            goingDisplay = resolvedGoing;
                        }
                        short? distanceYards = metadata.DistanceYards > 0
                            ? (short)Math.Min(metadata.DistanceYards, short.MaxValue)
                            : null;
                        var upcomingGoing = !string.IsNullOrWhiteSpace(resolvedGoing)
                            ? resolvedGoing
                            : metadataGoing;
                        persistedUpcoming = new UpcomingRace
                        {
                            MarketId = marketId,
                            RaceDate = parsedRaceDate.Value.Date,
                            ScheduledOff = offTime,
                            VenueName = trimmedVenueName,
                            VenueCountry = trimmedVenueCountry,
                            Title = trimmedTitle,
                            RaceDetails = trimmedRaceDetails,
                            RaceType = metadata.RaceType,
                            Surface = metadata.Surface,
                            Going = string.IsNullOrWhiteSpace(upcomingGoing) ? null : upcomingGoing,
                            DistanceYards = distanceYards,
                            DistanceText = metadata.DistanceText,
                            RunnerCount = runnerCount,
                            BackBookPercentage = backBookPercentage,
                            LayBookPercentage = layBookPercentage
                        };

                        lock (_repoLock)
                        {
                            var upcomingId = _repo.UpsertUpcomingRace(persistedUpcoming);
                            CacheUpcomingRace(persistedUpcoming);
                            persistedUpcoming.UpcomingRaceId = upcomingId;
                        }
                        if (!string.IsNullOrWhiteSpace(persistedUpcoming.Going))
                        {
                            resolvedGoing = persistedUpcoming.Going.Trim();
                            goingSource = "upcoming race record";
                            goingDisplay = string.IsNullOrWhiteSpace(resolvedGoing) ? "<null>" : resolvedGoing;

                        }
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"\tFailed to record upcoming race metadata for market {marketId}: {ex.Message}");
                        persistedUpcoming = null;
                    }
                }
                goingDisplay = string.IsNullOrWhiteSpace(resolvedGoing) ? "<null>" : resolvedGoing;
                Console.WriteLine($"\t[DayReport] Resolved going for market {marketId}: '{goingDisplay}' (source: {goingSource}).");
                var flows = new List<RunnerFlow>(); // collect runner flows
                var runnerEntries = new List<(IWebElement Row, RunnerFlow Flow)>(); // row→flow mapping

                foreach (var row in rows)
                {
                    var js = (IJavaScriptExecutor)driver; // cast to JS
                    const string runnerExtractionScript = """
        const row = arguments[0];
        const textOrEmpty = el => el && el.textContent ? el.textContent.trim() : '';
        const normalizeSpaces = value => value ? value.replace(/\s+/g, ' ').trim() : '';

        const detailRootSet = new Set();
        const detailRoots = [];
        const candidateRows = (() => {
            const items = [];
            if (row) { items.push(row); }
            const tableRow = row && row.closest ? row.closest('tr') : null;
            if (tableRow && !items.includes(tableRow)) { items.push(tableRow); }
            const related = [
                row && row.previousElementSibling,
                row && row.nextElementSibling,
                tableRow && tableRow.previousElementSibling,
                tableRow && tableRow.nextElementSibling
            ];
            for (const sibling of related) {
                if (sibling && !items.includes(sibling)) {
                    items.push(sibling);
                }
            }
            return items.filter(Boolean);
        })();
        const belongsToCurrentRow = element => {
            if (!element) { return false; }
            const visited = new Set();
            const stack = [element];
            while (stack.length) {
                const node = stack.pop();
                if (!node || visited.has(node)) { continue; }
                visited.add(node);
                if (candidateRows.includes(node)) { return true; }
                if (node.parentElement) { stack.push(node.parentElement); }
                if (node.assignedSlot) { stack.push(node.assignedSlot); }
                if (typeof node.getRootNode === 'function') {
                    const rootNode = node.getRootNode();
                    if (rootNode && rootNode.host) { stack.push(rootNode.host); }
                }
            }
            return false;
        };
        const addRoot = node => {
            if (!node || detailRootSet.has(node)) { return; }
            detailRootSet.add(node);
            detailRoots.push(node);
            if (node.shadowRoot) { addRoot(node.shadowRoot); }
            if (node.tagName === 'SLOT' && typeof node.assignedElements === 'function') {
                const assigned = node.assignedElements();
                if (assigned && assigned.length) {
                    for (const child of assigned) { addRoot(child); }
                }
            }
        };
        const ascendAncestors = start => {
            let current = start;
            for (let depth = 0; depth < 10 && current; depth++) {
                const parent = current.parentElement;
                if (parent) {
                    addRoot(parent);
                    current = parent;
                    continue;
                }
                if (typeof current.getRootNode === 'function') {
                    const rootNode = current.getRootNode();
                    if (rootNode && rootNode.host) {
                        addRoot(rootNode.host);
                        current = rootNode.host;
                        continue;
                    }
                }
                break;
            }
        };
        const addRelated = node => {
            if (!node) { return; }
            addRoot(node);
            ascendAncestors(node);
        };
        addRelated(row);
        if (row.previousElementSibling) { addRelated(row.previousElementSibling); }
        if (row.nextElementSibling) { addRelated(row.nextElementSibling); }
        const tableRow = row.closest ? row.closest('tr') : null;
        if (tableRow && tableRow.nextElementSibling) { addRelated(tableRow.nextElementSibling); }
        const queryWithin = (root, selector) => {
            if (!root || !selector) { return null; }
            if (root instanceof Element && root.matches(selector)) { return root; }
            if (typeof root.querySelector === 'function') {
                const direct = root.querySelector(selector);
                if (direct) { return direct; }
            }
            if (typeof root.querySelectorAll !== 'function') { return null; }
            const all = root.querySelectorAll('*');
            for (const el of all) {
                if (el.shadowRoot) {
                    const shadowResult = queryWithin(el.shadowRoot, selector);
                    if (shadowResult) { return shadowResult; }
                }
                if (el.tagName === 'SLOT' && typeof el.assignedElements === 'function') {
                    const assigned = el.assignedElements();
                    if (assigned && assigned.length) {
                        for (const assignedEl of assigned) {
                            const assignedResult = queryWithin(assignedEl, selector);
                            if (assignedResult) { return assignedResult; }
                        }
                    }
                }
            }
            return null;
        };
        const queryText = selector => {
            if (!selector) { return ''; }
            const selectors = Array.isArray(selector) ? selector : [selector];
            for (const sel of selectors) {
                if (!sel) { continue; }
                for (const root of detailRoots) {
                    const element = queryWithin(root, sel);
                    if (element) {
                        const text = textOrEmpty(element);
                        if (text) { return text; }
                    }
                }
            }
            return '';
        };
        const queryDetail = selectors => {
            if (!selectors) { return ''; }
            const values = Array.isArray(selectors) ? selectors : [selectors];
            for (const selector of values) {
                if (!selector) { continue; }
                for (const root of detailRoots) {
                    const element = queryWithin(root, selector);
                    if (!element || !belongsToCurrentRow(element)) { continue; }
                    const text = textOrEmpty(element);
                    if (text) { return normalizeSpaces(text); }
                }
            }
            return '';
        };
        const extractPriceText = raw => {
            if (!raw) { return ''; }
            const text = raw.trim();
            if (!text) { return ''; }
            if (/^[£€$]/.test(text)) { return ''; }
            return text;
        };
        const extractPriceFromButton = raw => {
            if (!raw) { return ''; }
            const text = raw.trim();
            if (!text) { return ''; }
            const tokens = text.split(/\s+/);
            for (const token of tokens) {
                if (!token || /^[£€$]/.test(token)) { continue; }
                if (/^[0-9]+(\.[0-9]+)?$/.test(token)) { return token; }
            }
            return extractPriceText(text);
        };
        const getButtonPrice = button => {
            if (!button) { return ''; }
            const attrValue = button.getAttribute('data-price')
                || button.getAttribute('data-bet-price')
                || button.getAttribute('value');
            if (attrValue && /^[0-9]+(\.[0-9]+)?$/.test(attrValue.trim())) {
                return attrValue.trim();
            }
            const label = button.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price, span.price, .price');
            const labelText = textOrEmpty(label);
            if (labelText) {
                const parsed = extractPriceFromButton(labelText);
                if (parsed) { return parsed; }
            }
            const text = textOrEmpty(button);
            return extractPriceFromButton(text);
        };
        const findOursPriceButton = (type, index) => {
            const cellIndex = type === 'back' ? 4 : 5;
            const cell = row.querySelector(`td:nth-of-type(${cellIndex})`);
            if (cell) {
                const oursButtons = Array.from(cell.querySelectorAll('ours-price-button'));
                if (oursButtons.length >= index) {
                    return oursButtons[index - 1];
                }
            }
            return null;
        };
        const findPrice = (type, index) => {
            const preferred = findOursPriceButton(type, index);
            if (preferred) {
                const price = getButtonPrice(preferred);
                if (price) { return price; }
            }
            const cellIndex = type === 'back' ? 4 : 5;
            const cell = row.querySelector(`td:nth-of-type(${cellIndex})`);
            if (cell) {
                const explicit = cell.querySelector(`[data-${type}-price], [data-price]`);
                if (explicit) {
                    const value = explicit.getAttribute(`data-${type}-price`) || explicit.getAttribute('data-price');
                    if (value && /^[0-9]+(\.[0-9]+)?$/.test(value.trim())) {
                        return value.trim();
                    }
                }
                const buttonCandidates = Array.from(cell.querySelectorAll('button, bet-button, ours-price-button'));
                for (const candidate of buttonCandidates) {
                    const price = getButtonPrice(candidate);
                    if (price) { return price; }
                }
            }
            const allButtons = Array.from(row.querySelectorAll('button, bet-button, ours-price-button, .bet-button-price, .price'));
            const collected = [];
            for (const candidate of allButtons) {
                const price = getButtonPrice(candidate);
                if (price) { collected.push(price); }
            }
            if (collected.length >= index) {
                return collected[index - 1];
            }
            return '';
        };
        const clothSelectors = [
            '.runner-number',
            '.runner-numbers .runner-number',
            '.runner-numbers .saddle-cloth',
            '.runner-numbers.double p.saddle-cloth',
            'p.saddle-cloth',
            '.saddle-cloth'
        ];
        const drawSelectors = [
            '.draw',
            '.runner-numbers .draw',
            '.runner-numbers.double p.stall-draw',
            'p.stall-draw',
            '.stall-draw'
        ];
        const result = {
            cloth: queryText(clothSelectors),
            draw: queryText(drawSelectors),
            horse: queryText('.name .runner-name'),
            jockey: queryText('.name .jockey-name'),
            back1: findPrice('back', 1),
            back2: findPrice('back', 2),
            back3: findPrice('back', 3),
            lay1: findPrice('lay', 1),
            lay2: findPrice('lay', 2),
            lay3: findPrice('lay', 3)
        };
        const fallbackButtons = Array.from(row.querySelectorAll('bet-button'));
        if (fallbackButtons.length > 0) {
            const fallbackPrices = fallbackButtons.map(btn => {
                const label = btn.querySelector('button label:nth-of-type(1), button span:nth-of-type(1), .bet-button-price');
                if (label) {
                    const priceText = extractPriceText(textOrEmpty(label));
                    if (priceText) { return priceText; }
                }
                const buttonText = btn.querySelector('button');
                return extractPriceFromButton(textOrEmpty(buttonText || btn));
            });
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
        }
        const ageWeightSelectors = [
            '.runner-timeform-wrapper__horse-details .runner-timeform-wrapper__details.runner-timeform-wrapper__age-weight-rating',
            '.runner-timeform-wrapper__details.runner-timeform-wrapper__age-weight-rating',
            '.runner-expanded-details .runner-timeform-wrapper__details.runner-timeform-wrapper__age-weight-rating',
            '.runner-expanded-details .runner-timeform-wrapper__details--age-weight',
            '.runner-info-expanded [data-testid="horse-age-weight"]',
            '.runner-info-expanded .runner-timeform-wrapper__details--age-weight',
            '[data-testid="runner-age-weight"]',
            '[data-test-id="runner-age-weight"]'
        ];
        const trainerSelectors = [
            '.runner-timeform-wrapper__horse-details .runner-timeform-wrapper__details.runner-timeform-wrapper__trainer',
            '.runner-timeform-wrapper__details.runner-timeform-wrapper__trainer',
            '.runner-expanded-details .runner-timeform-wrapper__details.runner-timeform-wrapper__trainer',
            '.runner-expanded-details .runner-timeform-wrapper__details--trainer',
            '.runner-info-expanded [data-testid="horse-trainer"]',
            '.runner-info-expanded .runner-timeform-wrapper__details--trainer',
            '[data-testid="runner-trainer"]',
            '[data-test-id="runner-trainer"]'
        ];
        result['ageWeight'] = queryDetail(ageWeightSelectors);
        result['trainer'] = queryDetail(trainerSelectors);
        return result;
    """;

                    var elementData = (IDictionary<string, object>)js.ExecuteScript(runnerExtractionScript, row); // execute script
                    string Get(string key) => elementData.TryGetValue(key, out var v) ? v?.ToString() ?? string.Empty : string.Empty; // helper

                    var runnerFlow = new RunnerFlow // create flow
                    {
                        MarketId = marketId, // market id
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
                    runnerFlow.UpcomingRaceId = persistedUpcoming?.UpcomingRaceId;
                    runnerFlow.RaceDate = persistedUpcoming?.RaceDate ?? parsedRaceDate?.Date;
                    runnerFlow.ScheduledOff = persistedUpcoming?.ScheduledOff ?? offTime;
                    runnerFlow.VenueName = persistedUpcoming?.VenueName ?? trimmedVenueName;
                    runnerFlow.VenueCountry = persistedUpcoming?.VenueCountry ?? trimmedVenueCountry;
                    runnerFlow.RaceTitle = persistedUpcoming?.Title ?? trimmedTitle;
                    runnerFlow.RaceDetails = persistedUpcoming?.RaceDetails ?? trimmedRaceDetails;
                    runnerFlow.RaceType = persistedUpcoming?.RaceType ?? trimmedRaceType ?? parsedMetadata?.RaceType;
                    runnerFlow.Surface = persistedUpcoming?.Surface ?? parsedMetadata?.Surface;
                    var runnerGoing = !string.IsNullOrWhiteSpace(resolvedGoing)
                        ? resolvedGoing
                        : (!string.IsNullOrWhiteSpace(persistedUpcoming?.Going)
                            ? persistedUpcoming.Going.Trim()
                            : metadataGoing);
                    runnerFlow.Going = string.IsNullOrWhiteSpace(runnerGoing) ? null : runnerGoing;
                    var parsedDistance = parsedMetadata?.DistanceYards;
                    if (!parsedDistance.HasValue && persistedUpcoming?.DistanceYards.HasValue == true)
                    {
                        parsedDistance = persistedUpcoming.DistanceYards;
                    }
                    runnerFlow.DistanceYards = parsedDistance.HasValue && parsedDistance.Value > 0
                        ? (short)Math.Min(parsedDistance.Value, short.MaxValue)
                        : null;
                    runnerFlow.DistanceText = persistedUpcoming?.DistanceText ?? parsedMetadata?.DistanceText;
                    runnerFlow.RunnerCount = persistedUpcoming?.RunnerCount ?? runnerCount;
                    runnerFlow.BackBookPercentage = persistedUpcoming?.BackBookPercentage ?? backBookPercentage;
                    runnerFlow.LayBookPercentage = persistedUpcoming?.LayBookPercentage ?? layBookPercentage;
                    var (age, weightLbs, weightText) = ParseRunnerAgeWeight(Get("ageWeight"));
                    runnerFlow.Age = age;
                    runnerFlow.WeightLbs = weightLbs;
                    runnerFlow.WeightText = string.IsNullOrWhiteSpace(weightText) ? null : weightText.Trim();
                    runnerFlow.TrainerName = NormalizeTrainerName(Get("trainer"));

                    flows.Add(runnerFlow); // add to list
                    runnerEntries.Add((row, runnerFlow)); // keep mapping
                }

                if (flows.Count == 0)
                {
                    Console.WriteLine($"\tNo runners captured for market {marketId}; skipping AI processing.");
                    continue;
                }

                var effectiveRaceDate = parsedRaceDate ?? DateTime.Today;
                if (!executeBets)
                {
                    pendingEvaluations!.Add(new PendingRaceEvaluation
                    {
                        MarketId = marketId,
                        Title = trimmedTitle ?? title,
                        VenueName = trimmedVenueName ?? venueName,
                        VenueCountry = trimmedVenueCountry ?? venueCountry,
                        RaceDate = parsedRaceDate?.Date,
                        EffectiveRaceDate = effectiveRaceDate,
                        OffTime = offTime,
                        CleanedRaceDetails = cleanedRaceDetails,
                        RaceTypeText = raceTypeText,
                        Going = resolvedGoing,
                        BackBookPercentage = backBookPercentage,
                        LayBookPercentage = layBookPercentage,
                        RaceUrl = raceUrl,
                        PersistedUpcoming = persistedUpcoming,
                        Flows = flows,
                        RunnerCount = rows.Count
                    });
                    continue;
                }
                if (_computeAiProbabilities && aiCalculator != null)
                {
                    PopulateFeatureVectors(
                        effectiveRaceDate,
                        title,
                        venueName,
                        venueCountry,
                        offTime,
                        cleanedRaceDetails,
                        raceTypeText,
                        resolvedGoing,
                        backBookPercentage,
                        layBookPercentage,
                        marketId,
                        flows,
                        rows.Count,
                        persistedUpcoming: persistedUpcoming); // build features

                    var flowsSnapshot = flows.ToList(); // snapshot for safe iteration
                    var hasAnyBackPrice = flowsSnapshot.Any(f => f.BackPrice1.HasValue); // detect available prices
                    var evaluationCandidates = new List<RunnerFlow>();

                    foreach (var rf in flowsSnapshot) // process each runner
                    {
                        if (rf.FeatureValues != null && !rf.FeatureValues.ContainsKey("RunnerCount") && rows.Count > 0)
                        {
                            rf.FeatureValues["RunnerCount"] = rows.Count; // ensure runner count
                        }
                        var rfIdentifier = DescribeRunner(rf);
                        if (hasAnyBackPrice && !rf.BackPrice1.HasValue)
                        {
                            Console.WriteLine($"            No back price available for {rfIdentifier}; assuming this runner is a non-runner and excluding it from analysis.");
                            flows.Remove(rf); // drop non-runner from active list
                            continue; // skip downstream processing
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
                        rf.AiProbabilityClampedToMarket = false;
                        rf.AiProbabilityClampTarget = null;
                        rf.AiProbabilityMarketDerived = false;
                        rf.AiProbabilityFallbackReason = null;
                        evaluationCandidates.Add(rf);
                    }

                    if (evaluationCandidates.Count > 0)
                    {
                        Parallel.ForEach(
                            evaluationCandidates,
                            rf =>
                            {
                                var rfIdentifier = DescribeRunner(rf);
                                try
                                {
                                    var probability = aiCalculator.CalculateOdds(rf); // compute AI odds
                                    if (double.IsFinite(probability) && probability > 0 && probability <= 1)
                                    {
                                        rf.AiOdds = probability;
                                        rf.AiProbabilityMarketDerived = false;
                                        rf.AiProbabilityFallbackReason = null;
                                    }
                                    else
                                    {
                                        rf.AiOdds = null;
                                        rf.AiProbabilityMarketDerived = false;
                                        rf.AiProbabilityFallbackReason = null;
                                        var probabilityText = double.IsFinite(probability)
                                            ? probability.ToString("0.####", CultureInfo.InvariantCulture)
                                            : "non-finite";
                                        Console.WriteLine($"            Discarding non-positive AI probability {probabilityText} for {rfIdentifier}; treating as missing.");
                                    }
                                }
                                catch (Exception ex)
                                {
                                    rf.AiOdds = null; // set null on fail
                                    rf.AiProbabilityMarketDerived = false;
                                    rf.AiProbabilityFallbackReason = null;
                                    rf.AiProbabilityClampedToMarket = false;
                                    rf.AiProbabilityClampTarget = null;
                                    Console.Error.WriteLine($"  Failed to calculate AI odds for runner {rfIdentifier} in market {marketId}: {ex.Message}"); // log
                                }

                            });
                    }
                    foreach (var rf in flowsSnapshot)
                    {
                        if (!flows.Contains(rf))
                        {
                            continue;
                        }

                        var rfIdentifier = DescribeRunner(rf);
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
                    var missingAiCount = flows.Count - validAiBefore.Count;
                    if (validAiBefore.Count == 0)
                    {
                        Console.WriteLine($"\tAI probability summary before normalization for market {marketId}: no valid AI probabilities; missing={missingAiCount}."); // log missing summary
                    }
                    else
                    {
                        var sumProb = validAiBefore.Sum(); // sum
                        var minProb = validAiBefore.Min(); // min
                        var maxProb = validAiBefore.Max(); // max
                        Console.WriteLine($"\tAI probability summary before normalization for market {marketId}: valid={validAiBefore.Count}, missing={missingAiCount}, sum={sumProb.ToString("0.####", CultureInfo.InvariantCulture)}, min={minProb.ToString("0.####", CultureInfo.InvariantCulture)}, max={maxProb.ToString("0.####", CultureInfo.InvariantCulture)}"); // log
                    }

                    NormalizeAiOdds(flows, _useMarketFallbackForAiDegeneracy); // normalize
                    ApplyMarketFallbackForUnmatchedRunners(flows); // ensure unmatched runners use market odds
                }
                else
                {
                    var flowsSnapshot = flows.ToList();
                    var hasAnyBackPrice = flowsSnapshot.Any(f => f.BackPrice1.HasValue);

                    foreach (var rf in flowsSnapshot)
                    {
                        var rfIdentifier = DescribeRunner(rf);

                        if (hasAnyBackPrice && !rf.BackPrice1.HasValue)
                        {
                            Console.WriteLine($"            No back price available for {rfIdentifier}; assuming this runner is a non-runner and excluding it from analysis.");
                            flows.Remove(rf);
                        }
                    }
                }

                if (captureReport)
                {
                    var report = BuildRaceReport(
                        marketId,
                        title,
                        venueName,
                        venueCountry,
                        effectiveRaceDate,
                        offTime,
                        cleanedRaceDetails,
                        raceTypeText,
                        resolvedGoing,
                        backBookPercentage,
                        layBookPercentage,
                        raceUrl,
                        flows
                    ); // build report

                    result.Races.Add(report); // collect report
                }

                var raceRecommendations = CreateRecommendations(flows, marketId, title, venueName, effectiveRaceDate, executeBets)
                    .OrderByDescending(r => r.Differential)
                    .ThenByDescending(r => r.KellyFraction)
                    .ToList(); // rank recs

                if (!executeBets)
                {
                    if (raceRecommendations.Count > 0) { Console.WriteLine("\tReport mode: positive expected value runner(s) identified; skipping bet execution."); } else { Console.WriteLine($"\tNo positive value opportunity identified for market {marketId}"); } // report only
                }
                else
                {
                    if (raceRecommendations.Count > 0)
                    {
                        if (!TryRefreshRunnerEntriesForBetting(driver, marketId, raceUrl, flows, out var refreshedEntries))
                        {
                            Console.Error.WriteLine($"\tSkipping bet execution for market {marketId} because the refreshed runner list could not be resolved.");
                            continue;
                        }

                        runnerEntries = refreshedEntries;
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
                }

                try
                {
                    lock (_repoLock)
                    {
                        _repo.InsertRunnerFlows(flows);
                    }

                    //foreach (var inner in flows)
                    //{
                    //    var innerIdentifier = DescribeRunner(inner);
                    //    Console.WriteLine($"\tInserted runner {innerIdentifier} for market {marketId}"); // log ok
                    //}
                }
                catch (Exception ex)
                {
                    foreach (var inner in flows)
                    {
                        var innerIdentifier = DescribeRunner(inner);
                        Console.Error.WriteLine($"\tInsertRunnerFlow failed for market {marketId}, runner {innerIdentifier}: {ex.Message}"); // log error
                    }
                }

                result.Recommendations.AddRange(recommendations.OrderByDescending(r => r.Differential).ThenByDescending(r => r.KellyFraction)); // finalize ordering
            }
            if (pendingEvaluations != null && pendingEvaluations.Count > 0)
            {
                ProcessDeferredRaceEvaluations(pendingEvaluations, aiCalculator, captureReport, result, recommendations);
            }
            Console.WriteLine("[DayReport][Stage] Completed scraping and AI evaluation for all open race tabs.");
            return result; // done
        }
    }

private void ProcessDeferredRaceEvaluations(
            List<PendingRaceEvaluation> pendingEvaluations,
            AIOddsCalculator? aiCalculator,
            bool captureReport,
            BetfairScrapeResult result,
            List<BetRecommendation> recommendations)
        {
            if (pendingEvaluations == null || pendingEvaluations.Count == 0)
            {
                return;
            }

            if (_computeAiProbabilities && aiCalculator != null)
            {
                PreloadFeatureLookupsForBatch(pendingEvaluations);
            }

            foreach (var evaluation in pendingEvaluations)
            {
                var flows = evaluation.Flows;
                if (flows == null || flows.Count == 0)
                {
                    Console.WriteLine($"\tNo runners captured for deferred market {evaluation.MarketId}; skipping.");
                    continue;
                }

                Console.WriteLine($"\t[Deferred] Processing cached market {evaluation.MarketId} with {flows.Count} runner(s).");

                if (_computeAiProbabilities && aiCalculator != null)
                {
                    PopulateFeatureVectors(
                        evaluation.EffectiveRaceDate,
                        evaluation.Title,
                        evaluation.VenueName,
                        evaluation.VenueCountry,
                        evaluation.OffTime,
                        evaluation.CleanedRaceDetails,
                        evaluation.RaceTypeText,
                        evaluation.Going,
                        evaluation.BackBookPercentage,
                        evaluation.LayBookPercentage,
                        evaluation.MarketId,
                        flows,
                        evaluation.RunnerCount,
                        persistedUpcoming: evaluation.PersistedUpcoming);

                    var flowsSnapshot = flows.ToList();
                    var hasAnyBackPrice = flowsSnapshot.Any(f => f.BackPrice1.HasValue);
                    var evaluationCandidates = new List<RunnerFlow>();

                    foreach (var rf in flowsSnapshot)
                    {
                        if (rf == null)
                        {
                            continue;
                        }

                        var rfIdentifier = DescribeRunner(rf);

                        if (hasAnyBackPrice && !rf.BackPrice1.HasValue)
                        {
                            Console.WriteLine($"            [Deferred] No back price available for {rfIdentifier}; assuming non-runner and excluding from analysis.");
                            flows.Remove(rf);
                            continue;
                        }

                        if (rf.FeatureValues != null && !rf.FeatureValues.ContainsKey("RunnerCount") && evaluation.RunnerCount > 0)
                        {
                            rf.FeatureValues["RunnerCount"] = evaluation.RunnerCount;
                        }

                        if (rf.FeatureValues != null)
                        {
                            if (rf.BackPrice1.HasValue) { rf.FeatureValues["BackPrice1"] = rf.BackPrice1.Value; }
                            if (rf.BackPrice2.HasValue) { rf.FeatureValues["BackPrice2"] = rf.BackPrice2.Value; }
                            if (rf.BackPrice3.HasValue) { rf.FeatureValues["BackPrice3"] = rf.BackPrice3.Value; }
                            if (rf.LayPrice1.HasValue) { rf.FeatureValues["LayPrice1"] = rf.LayPrice1.Value; }
                            if (rf.LayPrice2.HasValue) { rf.FeatureValues["LayPrice2"] = rf.LayPrice2.Value; }
                            if (rf.LayPrice3.HasValue) { rf.FeatureValues["LayPrice3"] = rf.LayPrice3.Value; }
                        }

                        rf.AiProbabilityClampedToMarket = false;
                        rf.AiProbabilityClampTarget = null;
                        rf.AiProbabilityMarketDerived = false;
                        rf.AiProbabilityFallbackReason = null;
                        evaluationCandidates.Add(rf);
                    }

                    if (evaluationCandidates.Count > 0)
                    {
                        Parallel.ForEach(
                            evaluationCandidates,
                            rf =>
                            {
                                var rfIdentifier = DescribeRunner(rf);
                                try
                                {
                                    var probability = aiCalculator.CalculateOdds(rf);
                                    if (double.IsFinite(probability) && probability > 0 && probability <= 1)
                                    {
                                        rf.AiOdds = probability;
                                        rf.AiProbabilityMarketDerived = false;
                                        rf.AiProbabilityFallbackReason = null;
                                    }
                                    else
                                    {
                                        rf.AiOdds = null;
                                        rf.AiProbabilityMarketDerived = false;
                                        rf.AiProbabilityFallbackReason = null;
                                        var probabilityText = double.IsFinite(probability)
                                            ? probability.ToString("0.####", CultureInfo.InvariantCulture)
                                            : "non-finite";
                                        Console.WriteLine($"            [Deferred] Discarding non-positive AI probability {probabilityText} for {rfIdentifier}; treating as missing.");
                                    }
                                }
                                catch (Exception ex)
                                {
                                    rf.AiOdds = null;
                                    rf.AiProbabilityMarketDerived = false;
                                    rf.AiProbabilityFallbackReason = null;
                                    rf.AiProbabilityClampedToMarket = false;
                                    rf.AiProbabilityClampTarget = null;
                                    Console.Error.WriteLine($"  [Deferred] Failed to calculate AI odds for runner {rfIdentifier} in market {evaluation.MarketId}: {ex.Message}");
                                }
                            });
                    }

                    foreach (var rf in flowsSnapshot)
                    {
                        if (!flows.Contains(rf))
                        {
                            continue;
                        }

                        var rfIdentifier = DescribeRunner(rf);
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

                        var backText = rf.BackPrice1.HasValue
                            ? rf.BackPrice1.Value.ToString("0.##", CultureInfo.InvariantCulture)
                            : "null";

                        Console.WriteLine($"\t[Deferred] Runner snapshot {rfIdentifier}: back1={backText}, aiProbabilityRaw={aiText}");

                        if (!rf.AiOdds.HasValue)
                        {
                            Console.WriteLine($"\t\t[Deferred] AI probability missing for {rfIdentifier}; downstream filters will treat this runner as zero edge.");
                        }

                        if (!rf.BackPrice1.HasValue)
                        {
                            Console.WriteLine($"\t\t[Deferred] No back price available for {rfIdentifier}; cannot compare against market probability.");
                        }
                    }

                    var validAiBefore = flows
                        .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                        .Select(f => f.AiOdds!.Value)
                        .ToList();

                    var missingAiCount = flows.Count - validAiBefore.Count;
                    if (validAiBefore.Count == 0)
                    {
                        Console.WriteLine($"\t[Deferred] AI probability summary before normalization for market {evaluation.MarketId}: no valid AI probabilities; missing={missingAiCount}.");
                    }
                    else
                    {
                        var sumProb = validAiBefore.Sum();
                        var minProb = validAiBefore.Min();
                        var maxProb = validAiBefore.Max();
                        Console.WriteLine($"\t[Deferred] AI probability summary before normalization for market {evaluation.MarketId}: valid={validAiBefore.Count}, missing={missingAiCount}, sum={sumProb.ToString("0.####", CultureInfo.InvariantCulture)}, min={minProb.ToString("0.####", CultureInfo.InvariantCulture)}, max={maxProb.ToString("0.####", CultureInfo.InvariantCulture)}");
                    }

                    NormalizeAiOdds(flows, _useMarketFallbackForAiDegeneracy);
                    ApplyMarketFallbackForUnmatchedRunners(flows);
                }
                else
                {
                    var flowsSnapshot = flows.ToList();
                    var hasAnyBackPrice = flowsSnapshot.Any(f => f.BackPrice1.HasValue);

                    foreach (var rf in flowsSnapshot)
                    {
                        if (rf == null)
                        {
                            continue;
                        }

                        var rfIdentifier = DescribeRunner(rf);

                        if (hasAnyBackPrice && !rf.BackPrice1.HasValue)
                        {
                            Console.WriteLine($"            [Deferred] No back price available for {rfIdentifier}; assuming non-runner and excluding from analysis.");
                            flows.Remove(rf);
                        }
                    }
                }

                if (captureReport)
                {
                    var report = BuildRaceReport(
                        evaluation.MarketId,
                        evaluation.Title,
                        evaluation.VenueName,
                        evaluation.VenueCountry,
                        evaluation.EffectiveRaceDate,
                        evaluation.OffTime,
                        evaluation.CleanedRaceDetails,
                        evaluation.RaceTypeText,
                        evaluation.Going,
                        evaluation.BackBookPercentage,
                        evaluation.LayBookPercentage,
                        evaluation.RaceUrl,
                        flows);

                    result.Races.Add(report);
                }

                var raceRecommendations = CreateRecommendations(
                        flows,
                        evaluation.MarketId,
                        evaluation.Title,
                        evaluation.VenueName,
                        evaluation.EffectiveRaceDate,
                        executeBets: false)
                    .OrderByDescending(r => r.Differential)
                    .ThenByDescending(r => r.KellyFraction)
                    .ToList();

                if (raceRecommendations.Count > 0)
                {
                    Console.WriteLine("\t[Deferred] Report mode: positive expected value runner(s) identified; skipping bet execution.");
                }
                else
                {
                    Console.WriteLine($"\t[Deferred] No positive value opportunity identified for market {evaluation.MarketId}.");
                }

                try
                {
                    lock (_repoLock)
                    {
                        _repo.InsertRunnerFlows(flows);
                    }
                }
                catch (Exception ex)
                {
                    foreach (var inner in flows)
                    {
                        var innerIdentifier = DescribeRunner(inner);
                        Console.Error.WriteLine($"\t[Deferred] InsertRunnerFlow failed for market {evaluation.MarketId}, runner {innerIdentifier}: {ex.Message}");
                    }
                }

                recommendations.AddRange(raceRecommendations);
                result.Recommendations.AddRange(raceRecommendations.OrderByDescending(r => r.Differential).ThenByDescending(r => r.KellyFraction));
            }
        }

        private void PreloadFeatureLookupsForBatch(List<PendingRaceEvaluation> pendingEvaluations)
        {
            if (pendingEvaluations == null || pendingEvaluations.Count == 0)
            {
                return;
            }

            var trainerRequests = new List<(RacePreparationKey Key, UpcomingRace Upcoming, IReadOnlyList<RunnerFlow> Flows)>();

            foreach (var evaluation in pendingEvaluations)
            {
                if (evaluation.Flows == null || evaluation.Flows.Count == 0)
                {
                    continue;
                }

                if (!TryResolveUpcomingForLookup(
                        evaluation.RaceDate,
                        evaluation.Title,
                        evaluation.VenueName,
                        evaluation.VenueCountry,
                        evaluation.OffTime,
                        evaluation.CleanedRaceDetails,
                        evaluation.RaceTypeText,
                        evaluation.Going,
                        evaluation.BackBookPercentage,
                        evaluation.LayBookPercentage,
                        evaluation.MarketId,
                        evaluation.Flows,
                        evaluation.PersistedUpcoming,
                        out var upcoming,
                        out var cacheKey,
                        out var persistedAccepted))
                {
                    continue;
                }

                if (persistedAccepted && evaluation.PersistedUpcoming != null)
                {
                    CacheUpcomingRace(evaluation.PersistedUpcoming);
                }

                trainerRequests.Add((cacheKey, upcoming, evaluation.Flows));
            }

            if (trainerRequests.Count == 0)
            {
                return;
            }

            var requestPayload = trainerRequests
                .Select(entry => (entry.Upcoming, entry.Flows))
                .ToList();

            var preparedResults = _trainer.PrepareUpcomingRaces(requestPayload);

            lock (_featureLookupCacheLock)
            {
                for (var i = 0; i < trainerRequests.Count; i++)
                {
                    var prepared = preparedResults.Count > i ? preparedResults[i] : null;
                    if (prepared != null)
                    {
                        _preloadedPreparedRaces[trainerRequests[i].Key] = prepared;
                    }
                    else
                    {
                        _preloadedPreparedRaces.Remove(trainerRequests[i].Key);
                    }
                }
            }
        }
    } }