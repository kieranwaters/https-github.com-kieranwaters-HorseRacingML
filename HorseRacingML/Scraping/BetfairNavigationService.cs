using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using SeleniumExtras.WaitHelpers;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Linq;

namespace HorseRacingML.Scraping
{
    public class BetfairNavigationService : IDisposable
    {
        private const string HardCodedUsername = "kierandpwaters@gmail.com";
        private const string HardCodedPassword = "AZQ2v.b=$e$!e!u";

        private readonly string _username;
        private readonly string _password;
        private readonly IWebDriver _driver;
        private readonly decimal _configuredBankroll;
        private decimal _bankroll;
        private readonly decimal? _maxKellyFraction;
        private readonly bool _useMarketFallbackForAiDegeneracy;
        private readonly decimal _kellyDampener;
        private static readonly Regex NonNumericCharactersRegex = new("[^0-9.,-]", RegexOptions.Compiled);

        public BetfairNavigationService(IConfiguration config)
        {
            _username = config["Betfair:Username"] ?? HardCodedUsername;
            _password = config["Betfair:Password"] ?? HardCodedPassword;
            _configuredBankroll = config.GetValue<decimal?>("Betting:Bankroll") ?? 100m;
            _bankroll = _configuredBankroll;
            _maxKellyFraction = config.GetValue<decimal?>("Betting:MaxKellyFraction");
            _useMarketFallbackForAiDegeneracy = config.GetValue<bool?>("Betting:UseMarketFallbackForAiDegeneracy") ?? true;
            _kellyDampener = config.GetValue<decimal?>("Betting:KellyDampener") ?? 1m;
            if (_kellyDampener <= 0m)
            {
                _kellyDampener = 1m;
            }
            else if (_kellyDampener > 1m)
            {
                _kellyDampener = 1m;
            }
            var options = new ChromeOptions();
            options.AddArguments(
                "--disable-extensions",
                "--blink-settings=imagesEnabled=false",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage");

            _driver = new ChromeDriver(options);
        }
        public async Task<RaceDayReport?> RefreshRaceAsync(
            string raceUrl,
            RacingRepository repo,
            HyperparameterTrainer trainer)
        {
            if (string.IsNullOrWhiteSpace(raceUrl))
            {
                return null;
            }

            await LoginAsync();

            var priorHandles = _driver.WindowHandles.ToList();
            var priorHandleSet = new HashSet<string>(priorHandles);
            var newHandles = new List<string>();
            var openedNewTab = false;

            try
            {
                try
                {
                    ((IJavaScriptExecutor)_driver).ExecuteScript("window.open(arguments[0],'_blank');", raceUrl);
                    await Task.Delay(500);
                    newHandles = _driver.WindowHandles
                        .Where(h => !priorHandleSet.Contains(h))
                        .ToList();
                    openedNewTab = newHandles.Count > 0;
                }
                catch (Exception)
                {
                    newHandles.Clear();
                }

                if (!openedNewTab)
                {
                    _driver.Navigate().GoToUrl(raceUrl);
                    await Task.Delay(500);
                    var current = _driver.CurrentWindowHandle;
                    if (!string.IsNullOrWhiteSpace(current))
                    {
                        newHandles = new List<string> { current };
                    }
                }

                if (newHandles.Count == 0)
                {
                    return null;
                }

                var bankroll = GetEffectiveBankroll();
                var scraper = new BetfairMarketScraper(
                    repo,
                    trainer,
                    bankroll,
                    _maxKellyFraction,
                    _kellyDampener,
                    _useMarketFallbackForAiDegeneracy);
                var races = scraper.ScrapeOpenRaceTabsForReport(_driver, newHandles);

                var targetMarketId = BetfairMarketScraper.ExtractMarketId(raceUrl);
                RaceDayReport? refreshed = null;
                if (!string.IsNullOrWhiteSpace(targetMarketId))
                {
                    refreshed = races.FirstOrDefault(r =>
                        string.Equals(r.MarketId, targetMarketId, StringComparison.OrdinalIgnoreCase));
                }

                return refreshed ?? races.FirstOrDefault();
            }
            finally
            {
                if (openedNewTab)
                {
                    foreach (var handle in newHandles)
                    {
                        try
                        {
                            _driver.SwitchTo().Window(handle);
                            _driver.Close();
                        }
                        catch (WebDriverException)
                        {
                            // Ignore failures when closing transient tabs.
                        }
                    }

                    var fallback = priorHandles.FirstOrDefault(h => _driver.WindowHandles.Contains(h));
                    if (!string.IsNullOrEmpty(fallback))
                    {
                        _driver.SwitchTo().Window(fallback);
                    }
                    else if (_driver.WindowHandles.Count > 0)
                    {
                        _driver.SwitchTo().Window(_driver.WindowHandles[0]);
                    }
                }
                else if (priorHandles.Count > 0)
                {
                    var current = priorHandles[0];
                    if (_driver.WindowHandles.Contains(current))
                    {
                        _driver.SwitchTo().Window(current);
                    }
                }
            }
        }
        public IWebDriver Driver => _driver;

        public IReadOnlyList<BetRecommendation> ScrapeOpenRaceTabs(RacingRepository repo, HyperparameterTrainer trainer)
        {
            var bankroll = GetEffectiveBankroll();
            var scraper = new BetfairMarketScraper(
                repo,
                trainer,
                bankroll,
                _maxKellyFraction,
                _kellyDampener,
                _useMarketFallbackForAiDegeneracy);
            return scraper.ScrapeOpenRaceTabs(_driver);
        }

        private decimal GetEffectiveBankroll()
        {
            var refreshed = TryRefreshBankrollFromPage();
            if (refreshed.HasValue && refreshed.Value > 0m)
            {
                _bankroll = refreshed.Value;
            }

            if (_bankroll <= 0m)
            {
                _bankroll = _configuredBankroll;
            }

            return _bankroll;
        }

        private decimal? TryRefreshBankrollFromPage()
        {
            string? balanceText = null;
            var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(5));

            var strategies = new Func<IWebDriver, IWebElement?>[]
            {
                drv => FindDisplayedElement(drv, By.XPath("/html/body/ui-view/div/div/div[1]/div[1]/div/bf-ssc-header/div/div/div/div/div/table/tbody/tr/td[4]/div/div/div/form/div[3]/div[1]/table/tbody/tr[1]/td[2]")),
                drv => FindDisplayedElement(drv, By.CssSelector("#ssc-wallet-balance-value")),
                drv => FindDisplayedElement(drv, By.CssSelector("span.ssc-wallet__balance-value"))
            };

            foreach (var strategy in strategies)
            {
                try
                {
                    var element = wait.Until(driver =>
                    {
                        try
                        {
                            var candidate = strategy(driver);
                            if (candidate != null)
                            {
                                var text = candidate.Text;
                                if (string.IsNullOrWhiteSpace(text))
                                {
                                    text = candidate.GetAttribute("textContent");
                                }

                                if (!string.IsNullOrWhiteSpace(text))
                                {
                                    return candidate;
                                }
                            }
                        }
                        catch (StaleElementReferenceException)
                        {
                            return null;
                        }

                        return null;
                    });

                    if (element != null)
                    {
                        balanceText = element.Text;
                        if (string.IsNullOrWhiteSpace(balanceText))
                        {
                            balanceText = element.GetAttribute("textContent");
                        }
                    }

                    if (!string.IsNullOrWhiteSpace(balanceText))
                    {
                        break;
                    }
                }
                catch (WebDriverTimeoutException)
                {
                    // continue to next strategy
                }
            }

            if (string.IsNullOrWhiteSpace(balanceText))
            {
                return null;
            }

            if (TryParseCurrency(balanceText, out var value) && value >= 0m)
            {
                return value;
            }

            return null;
        }

        private static IWebElement? FindDisplayedElement(IWebDriver driver, By by)
        {
            try
            {
                var element = driver.FindElement(by);
                return element.Displayed ? element : null;
            }
            catch (NoSuchElementException)
            {
                return null;
            }
        }

        private static bool TryParseCurrency(string? text, out decimal value)
        {
            value = 0m;
            if (string.IsNullOrWhiteSpace(text))
            {
                return false;
            }

            var trimmed = text.Trim();
            var cultures = new[]
            {
                CultureInfo.GetCultureInfo("en-GB"),
                CultureInfo.InvariantCulture
            };

            foreach (var culture in cultures)
            {
                if (decimal.TryParse(trimmed, NumberStyles.Currency | NumberStyles.AllowThousands, culture, out value))
                {
                    return true;
                }
            }

            var sanitized = NonNumericCharactersRegex.Replace(trimmed, string.Empty);
            sanitized = sanitized.Replace(",", string.Empty);

            return decimal.TryParse(
                sanitized,
                NumberStyles.Number | NumberStyles.AllowLeadingSign,
                CultureInfo.InvariantCulture,
                out value);
        }

        public async Task LoginAsync()
        {
            _driver.Navigate().GoToUrl("https://www.betfair.com/exchange/plus/");

            var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(45));

            try
            {
                var cookie = wait.Until(ExpectedConditions.ElementToBeClickable(By.Id("onetrust-accept-btn-handler")));
                try
                {
                    cookie.Click();
                }
                catch (ElementClickInterceptedException)
                {
                    ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", cookie);
                }
            }
            catch (WebDriverTimeoutException)
            {
                // The cookie banner does not always appear.
            }

            bool IsLoggedIn() =>
                _driver.Url.Contains("loginStatus=SUCCESS", StringComparison.OrdinalIgnoreCase) ||
                _driver.PageSource.Contains("My Account", StringComparison.OrdinalIgnoreCase);

            if (!IsLoggedIn())
            {
                IWebElement? iframe = null;
                try
                {
                    iframe = wait.Until(d =>
                    {
                        var frames = d.FindElements(By.CssSelector("iframe[src*='identitysso'],iframe[id*='sso'],iframe[name*='sso']"));
                        return frames.Count > 0 ? frames[0] : null;
                    });
                }
                catch (WebDriverTimeoutException)
                {
                    // Some regions show a full page form instead of an iframe.
                }

                if (iframe != null)
                {
                    _driver.SwitchTo().Frame(iframe);
                }

                var userField = wait.Until(d =>
                {
                    var elements = d.FindElements(By.CssSelector("#ssc-liu,#username,input[name='username']"));
                    return elements.Count > 0 ? elements[0] : null;
                });
                userField.Clear();
                userField.SendKeys(_username);

                var passwordField = wait.Until(d =>
                {
                    var elements = d.FindElements(By.CssSelector("#ssc-lipw,#password,input[type='password']"));
                    return elements.Count > 0 ? elements[0] : null;
                });
                passwordField.Clear();
                passwordField.SendKeys(_password);

                wait.Until(_ =>
                    !string.IsNullOrEmpty(userField.GetAttribute("value")) &&
                    !string.IsNullOrEmpty(passwordField.GetAttribute("value")));

                var loginButton = wait.Until(d =>
                {
                    var elements = d.FindElements(By.CssSelector("input#ssc-lis,input[type='submit'][id='ssc-lis'],input[type='submit'][value='Log In']"));
                    return elements.Count > 0 ? elements[0] : null;
                });

                ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", loginButton);

                try
                {
                    wait.Until(ExpectedConditions.ElementToBeClickable(loginButton));
                    loginButton.Click();
                }
                catch (Exception)
                {
                    try
                    {
                        ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", loginButton);
                    }
                    catch (Exception)
                    {
                        try
                        {
                            loginButton.Submit();
                        }
                        catch (Exception)
                        {
                            passwordField.SendKeys(Keys.Enter);
                        }
                    }
                }

                await Task.Delay(1200);
                _driver.SwitchTo().DefaultContent();

                wait.Until(d =>
                    IsLoggedIn() ||
                    !d.Url.Contains("login", StringComparison.OrdinalIgnoreCase) ||
                    !d.Url.Contains("identitysso", StringComparison.OrdinalIgnoreCase));
            }

            wait.Until(d =>
                ((IJavaScriptExecutor)d).ExecuteScript("return document.readyState").ToString() == "complete");

            var navigationRoot = wait.Until(d =>
                d.FindElement(By.CssSelector("bf-navigation-lhm .navigation-container")));

            ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollTop=0;", navigationRoot);

            IWebElement? horseRacingLink = null;
            Func<IWebDriver, IWebElement?> findHorseRacingLink = drv =>
            {
                var selectors = new[]
                {
                    By.CssSelector("bf-navigation-lhm a.navigation-link[href*='horse-racing-betting-7']"),
                    By.CssSelector("bf-navigation-lhm a.navigation-link[link-sport='7']"),
                    By.XPath("//bf-navigation-lhm//a[normalize-space()='Horse Racing']")
                };

                foreach (var selector in selectors)
                {
                    var candidates = drv.FindElements(selector);
                    foreach (var candidate in candidates)
                    {
                        if (candidate.Displayed)
                        {
                            return candidate;
                        }
                    }
                }

                return null;
            };

            for (int i = 0; i < 10 && horseRacingLink == null; i++)
            {
                horseRacingLink = findHorseRacingLink(_driver);
                if (horseRacingLink == null)
                {
                    ((IJavaScriptExecutor)_driver).ExecuteScript(
                        "arguments[0].scrollTop=arguments[0].scrollTop+180;",
                        navigationRoot);
                    await Task.Delay(120);
                }
            }

            if (horseRacingLink == null)
            {
                horseRacingLink = wait.Until(d =>
                {
                    var element = ((IJavaScriptExecutor)d).ExecuteScript(
                        "return document.querySelector(\"#main-wrapper > div > div.scrollable-panes-height-taker > div > div > div.bf-row.no-bottom-gutter.nested-scrollable-pane-parent.extra-links-space-3.hidden-when-left-side-collapsed.extra-links-space-2 > div > bf-navigation-lhm > div > div > ng-include > div > div > ul > li > tree-section > span > div > div > div > ul > li:nth-child(16) > a\")");
                    return element is IWebElement we ? we : null;
                });
            }

            ((IJavaScriptExecutor)_driver).ExecuteScript(
                "arguments[0].scrollIntoView({block:'center'});",
                horseRacingLink);

            try
            {
                wait.Until(ExpectedConditions.ElementToBeClickable(horseRacingLink));
                horseRacingLink.Click();
            }
            catch (Exception)
            {
                ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", horseRacingLink);
            }

            wait.Until(d =>
                d.Url.Contains("horse-racing", StringComparison.OrdinalIgnoreCase) ||
                d.Url.Contains("horse-racing-betting-7", StringComparison.OrdinalIgnoreCase));
        }
        public DayReportViewModel GenerateDayReport(RacingRepository repo, HyperparameterTrainer trainer)
        {
            repo.ClearDayReportTables();
            var bankroll = GetEffectiveBankroll();
            var scraper = new BetfairMarketScraper(
                repo,
                trainer,
                bankroll,
                _maxKellyFraction,
                 _kellyDampener,
                _useMarketFallbackForAiDegeneracy);
            var races = scraper.ScrapeOpenRaceTabsForReport(_driver);
            var orderedRaces = races
                .OrderBy(r => GetRaceScheduleSortKey(r))
                .ThenBy(r => r.RaceTitle ?? string.Empty, StringComparer.OrdinalIgnoreCase)
                .ThenBy(r => r.MarketId, StringComparer.Ordinal)
                .ToList();
            return new DayReportViewModel
            {
                GeneratedAt = DateTime.UtcNow,
                Bankroll = bankroll,
                AiHyperparameters = scraper.LoadedHyperparameters,
                Races = orderedRaces
            };
        }

        private static DateTime GetRaceScheduleSortKey(RaceDayReport race)
        {
            var schedule = GetRaceScheduleDateTime(race);
            if (!schedule.HasValue)
            {
                return DateTime.MaxValue;
            }

            var value = schedule.Value;
            return value.Kind == DateTimeKind.Unspecified
                ? DateTime.SpecifyKind(value, DateTimeKind.Utc)
                : value.ToUniversalTime();
        }

        private static DateTime? GetRaceScheduleDateTime(RaceDayReport race)
        {
            if (race == null)
            {
                return null;
            }

            if (race.RaceDate.HasValue)
            {
                var date = race.RaceDate.Value;

                if (race.OffTime.HasValue)
                {
                    return date.Date.Add(race.OffTime.Value);
                }

                return date;
            }

            if (race.OffTime.HasValue)
            {
                return DateTime.Today.Add(race.OffTime.Value);
            }

            return null;
        }
        public async Task OpenHorseRaceMeetingsInNewTabsAsync(int delayBetweenTabsMs = 0)
        {
            var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(20)); wait.Until(d => ((IJavaScriptExecutor)d).ExecuteScript("return document.readyState").ToString() == "complete"); wait.Until(d => d.FindElements(By.CssSelector("a,button")).Count > 0); bool Is24HourTime(string s) { if (string.IsNullOrWhiteSpace(s)) return false; var t = s.Trim(); return DateTime.TryParseExact(t, new[] { "H:mm", "HH:mm" }, CultureInfo.InvariantCulture, DateTimeStyles.None, out _); } // filter by 24h times
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase); // dedupe hrefs
            var scrollRoot = (IWebElement)((IJavaScriptExecutor)_driver).ExecuteScript("return document.scrollingElement||document.body"); // scrollable root
            for (int pass = 0; pass < 6; pass++)
            {
                var candidates = _driver.FindElements(By.CssSelector("a,button")); foreach (var el in candidates) { try { if (!el.Displayed) continue; string txt = el.Text; if (string.IsNullOrWhiteSpace(txt)) txt = el.GetAttribute("innerText"); txt = txt?.Trim() ?? string.Empty; if (!Is24HourTime(txt)) continue; var href = (string)((IJavaScriptExecutor)_driver).ExecuteScript("const n=arguments[0];return (n.closest&&n.closest('a')&&n.closest('a').href)||n.href||null;", el); if (string.IsNullOrWhiteSpace(href)) continue; if (!href.Contains("/horse-racing/", StringComparison.OrdinalIgnoreCase)) continue; seen.Add(href); } catch (StaleElementReferenceException) { continue; } } // collect time links on page
            ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollTop=arguments[0].scrollTop+Math.min(1200,window.innerHeight);", scrollRoot); await Task.Delay(150);
            } // light scroll to load lazy content
            if (seen.Count == 0)
            { // fallback to any horse-racing links if no time-labelled buttons found
                var links = _driver.FindElements(By.CssSelector("a[href*='/horse-racing/']")); foreach (var a in links) { var href = a.GetAttribute("href"); if (!string.IsNullOrWhiteSpace(href)) seen.Add(href); }
            }
            foreach (var url in seen) { ((IJavaScriptExecutor)_driver).ExecuteScript("window.open(arguments[0],'_blank');", url); if (delayBetweenTabsMs > 0) await Task.Delay(delayBetweenTabsMs); }
            await Task.CompletedTask;
        }

        public void Dispose()
        {
            _driver.Quit();
            _driver.Dispose();
        }
    }
}
