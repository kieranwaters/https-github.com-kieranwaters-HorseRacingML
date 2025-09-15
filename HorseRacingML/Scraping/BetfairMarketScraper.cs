using System;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Linq;
using System.Threading.Tasks;
using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using HorseRacingML.Data;
using HorseRacingML.Models;

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
                var offTimeText = TextOrEmpty(driver, By.CssSelector("[data-testid='startTime']"));
                TimeSpan? offTime = TimeSpan.TryParse(offTimeText, out var t) ? t : (TimeSpan?)null;
                Console.WriteLine($"\tScraping market {marketId} - {title}");

                try
                {
                    var screen = new RaceScreen
                    {
                        MarketId = marketId,
                        OffTime = offTime,
                        Title = title,
                        RaceDate = DateTime.Today
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

                    var flow = new RunnerFlow
                    {
                        MarketId = marketId,
                        SelectionId = selectionId,
                        ClothNumber = TryParseByte(SafeText(row, By.CssSelector(".runner-number"))),
                        Draw = TryParseByte(SafeText(row, By.CssSelector(".draw"))),
                        HorseName = SafeText(row, By.CssSelector(".runner-name .runner-name")),
                        JockeyName = SafeText(row, By.CssSelector(".jockey-name")),
                        BackPrice1 = ParseDecimal(SafeText(row, By.CssSelector(".bet-button.back-selection-button.back-1 .bet-button-price"))),
                        BackPrice2 = ParseDecimal(SafeText(row, By.CssSelector(".bet-button.back-selection-button.back-2 .bet-button-price"))),
                        BackPrice3 = ParseDecimal(SafeText(row, By.CssSelector(".bet-button.back-selection-button.back-3 .bet-button-price"))),
                        LayPrice1 = ParseDecimal(SafeText(row, By.CssSelector(".bet-button.lay-selection-button.lay-1 .bet-button-price"))),
                        LayPrice2 = ParseDecimal(SafeText(row, By.CssSelector(".bet-button.lay-selection-button.lay-2 .bet-button-price"))),
                        LayPrice3 = ParseDecimal(SafeText(row, By.CssSelector(".bet-button.lay-selection-button.lay-3 .bet-button-price")))
                    };

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