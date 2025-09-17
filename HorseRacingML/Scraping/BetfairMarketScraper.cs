using System;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Linq;
using System.Threading.Tasks;
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

        public BetfairMarketScraper(RacingRepository repo)
        {
            _repo = repo;
        }
        public void ScrapeOpenRaceTabs(IWebDriver driver)
        {
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10));
            var handles = driver.WindowHandles.ToList();
            var weightPath = Path.Combine(AppContext.BaseDirectory, "aiweights.json");
            var aiCalculator = new AIOddsCalculator(weightPath);
            Parallel.ForEach(handles, handle =>
            {
                driver.SwitchTo().Window(handle);
                Console.WriteLine($"Processing tab: {driver.Url}");
                if (!driver.Url.Contains("/horse-racing/", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("\tSkipping non-racing tab");
                    return;
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
                    return;
                }

                var marketId = ExtractMarketId(driver.Url);
                if (string.IsNullOrEmpty(marketId))
                {
                    Console.Error.WriteLine($"\tFailed to extract market ID from URL: {driver.Url}");
                    return;
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
                    return;
                }

                // Retrieve the runner rows using the updated selector described above
                // (any element with the runner-line class)
                var rows = driver.FindElements(By.CssSelector(".runner-line"));
                if (rows.Count == 0)
                {
                    Console.Error.WriteLine($"\tNo runner rows found for market {marketId}");
                    return;
                }
                Console.WriteLine($"\tFound {rows.Count} runners for market {marketId}");
                var flows = new List<RunnerFlow>();
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
                }

                NormalizeAiOdds(flows);

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
            });
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