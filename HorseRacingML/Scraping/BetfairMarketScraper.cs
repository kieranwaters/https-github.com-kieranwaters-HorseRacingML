using System;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Linq;
using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using HorseRacingML.Data;
using HorseRacingML.Models;

namespace HorseRacingML.Scraping
{
    public class BetfairMarketScraper
    {
        private readonly RacingRepository _repo;

        public BetfairMarketScraper(RacingRepository repo)
        {
            _repo = repo;
        }

        public void ScrapeOpenRaceTabs(IWebDriver driver)
        {
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10));
            var handles = driver.WindowHandles.ToList();
            foreach (var handle in handles)
            {
                driver.SwitchTo().Window(handle);
                if (!driver.Url.Contains("/horse-racing/", StringComparison.OrdinalIgnoreCase))
                    continue;

                try
                {
                    wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='runner']")).Count > 0);
                }
                catch
                {
                    continue;
                }

                var marketId = ExtractMarketId(driver.Url);
                var title = TextOrEmpty(driver, By.CssSelector("[data-testid='marketTitle']"));
                var offTimeText = TextOrEmpty(driver, By.CssSelector("[data-testid='startTime']"));
                TimeSpan? offTime = TimeSpan.TryParse(offTimeText, out var t) ? t : (TimeSpan?)null;

                _repo.InsertRaceScreen(new RaceScreen
                {
                    MarketId = marketId,
                    OffTime = offTime,
                    Title = title,
                    RaceDate = DateTime.Today
                });

                var rows = driver.FindElements(By.CssSelector("[data-test-id='runner']"));
                foreach (var row in rows)
                {
                    var flow = new RunnerFlow
                    {
                        MarketId = marketId,
                        SelectionId = row.GetAttribute("data-selection-id"),
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
                    _repo.InsertRunnerFlow(flow);
                }
            }
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
            var m = Regex.Match(url, @"(?:/market/|marketId=)([0-9.]+)");
            return m.Success ? m.Groups[1].Value : null;
        }
    }
}