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
using System.IO;

namespace HorseRacingML.Scraping
{
    public class BetfairMarketScraper
    {
        private readonly RacingRepository _repo;
        private readonly object _repoLock = new();
        private readonly decimal _bankroll;
        private readonly decimal? _maxKellyFraction;
        public BetfairMarketScraper(RacingRepository repo, decimal bankroll, decimal? maxKellyFraction = null)
        {
            _repo = repo;
            _bankroll = bankroll;
            _maxKellyFraction = maxKellyFraction;
        }
        public IReadOnlyList<BetRecommendation> ScrapeOpenRaceTabs(IWebDriver driver)
        {
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10));
            var handles = driver.WindowHandles.ToList();
            var weightPath = Path.Combine(AppContext.BaseDirectory, "aiweights.json");
            var aiCalculator = new AIOddsCalculator(weightPath);
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
                    var elementData = (IDictionary<string, object>)js.ExecuteScript(@"
                        const row = arguments[0];
                        const q = s => {
                            const el = row.querySelector(s);
                            return el ? el.textContent.trim() : '';
                        };
                        return {
                            cloth: q('.runner-number'),
                            draw: q('.draw'),
                            horse: q('.name .runner-name'),
                            jockey: q('.name .jockey-name'),
                            back1: q('.bet-button.back-selection-button.back-1 .bet-button-price'),
                            back2: q('.bet-button.back-selection-button.back-2 .bet-button-price'),
                            back3: q('.bet-button.back-selection-button.back-3 .bet-button-price'),
                            lay1: q('.bet-button.lay-selection-button.lay-1 .bet-button-price'),
                            lay2: q('.bet-button.lay-selection-button.lay-2 .bet-button-price'),
                            lay3: q('.bet-button.lay-selection-button.lay-3 .bet-button-price')
                        };
                    ", row);

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

                    flows.Add(flow);
                    runnerEntries.Add((row, flow));
                }

                NormalizeAiOdds(flows);
                var raceRecommendations = CreateRecommendations(flows, marketId, title, venueName, parsedRaceDate)
                    .OrderByDescending(r => r.Differential)
                    .ThenByDescending(r => r.KellyFraction)
                    .ToList();

                if (raceRecommendations.Count > 0)
                {
                    recommendations.AddRange(raceRecommendations);
                    var top = raceRecommendations.First();
                    Console.WriteLine($"\tKelly stake {top.Stake.ToString("0.##", CultureInfo.InvariantCulture)} on {top.HorseName ?? "unknown"} (diff {top.Differential.ToString("0.####", CultureInfo.InvariantCulture)})");
                    ExecuteBackAllClicks(driver, runnerEntries, raceRecommendations);
                }
                else
                {
                    Console.WriteLine($"\tNo positive value opportunity identified for market {marketId}");
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
            return recommendations
                .OrderByDescending(r => r.Differential)
                .ThenByDescending(r => r.KellyFraction)
                .ToList();
        }
        private IEnumerable<BetRecommendation> CreateRecommendations(
            IEnumerable<RunnerFlow> flows,
            string marketId,
            string? raceTitle,
            string? venueName,
            DateTime? raceDate)
        {
            if (_bankroll <= 0m)
            {
                return Enumerable.Empty<BetRecommendation>();
            }

            var recommendations = new List<BetRecommendation>();

            foreach (var flow in flows)
            {
                if (!flow.AiOdds.HasValue || !double.IsFinite(flow.AiOdds.Value) || flow.AiOdds.Value <= 0)
                {
                    continue;
                }

                if (!flow.BackPrice1.HasValue || flow.BackPrice1.Value <= 1m)
                {
                    continue;
                }

                var decimalOdds = flow.BackPrice1.Value;
                var aiProbability = flow.AiOdds.Value;
                var marketProbability = 1.0 / (double)decimalOdds;
                var differential = aiProbability - marketProbability;

                if (differential <= 0)
                {
                    continue;
                }

                var kellyFraction = CalculateKellyFraction(aiProbability, (double)decimalOdds);
                if (kellyFraction <= 0)
                {
                    continue;
                }

                var stake = _bankroll * kellyFraction;
                if (stake <= 0)
                {
                    continue;
                }

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
                    Stake = stake
                });
            }

            return recommendations;
        }
        private void ExecuteBackAllClicks(
            IWebDriver driver,
            IReadOnlyList<(IWebElement Row, RunnerFlow Flow)> runnerEntries,
            IReadOnlyList<BetRecommendation> recommendations)
        {
            if (runnerEntries.Count == 0 || recommendations.Count == 0)
            {
                return;
            }

            foreach (var recommendation in recommendations)
            {
                if (recommendation.Differential <= 0)
                {
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

                TryClickBackAllButton(driver, match.Row, recommendation);
                Thread.Sleep(TimeSpan.FromMilliseconds(400));
            }
        }
        private void TryClickBackAllButton(IWebDriver driver, IWebElement row, BetRecommendation recommendation)
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
                    }
                    else
                    {
                        Console.Error.WriteLine($"\tBack-All button not found for {identifier}");
                    }
                    return;
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
                    return;
                }
                catch (Exception)
                {
                }

                try
                {
                    js.ExecuteScript("arguments[0].click();", button);
                    Console.WriteLine($"\tClicked Back-All for {identifier} using JavaScript");
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
        private static void NormalizeAiOdds(ICollection<RunnerFlow> flows)
        {
            var valid = flows
                .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                .ToList();

            if (valid.Count == 0)
            {
                return;
            }

            var sum = valid.Sum(f => f.AiOdds!.Value);

            if (sum <= double.Epsilon)
            {
                var uniform = 1.0 / valid.Count;
                foreach (var flow in valid)
                {
                    flow.AiOdds = uniform;
                }
                return;
            }

            foreach (var flow in valid)
            {
                flow.AiOdds = Math.Max(flow.AiOdds!.Value / sum, 0);
            }
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
            return decimal.TryParse(text, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : (decimal?)null;
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