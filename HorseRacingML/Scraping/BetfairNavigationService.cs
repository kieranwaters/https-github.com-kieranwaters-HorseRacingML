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
using System.Threading;
using HorseRacingML.Services;
using System.Collections;

namespace HorseRacingML.Scraping
{
    public class BetfairNavigationService : IDisposable
    {
        private const string HardCodedUsername = "kierandpwaters@gmail.com";
        private const string HardCodedPassword = "AZQ2v.b=$e$!e!u";
        private const string HorseRacingScheduleUrl = "https://www.betfair.com/exchange/plus/horse-racing";

        private readonly string _username;
        private readonly string _password;
        private IWebDriver _driver;
        private readonly decimal _configuredBankroll;
        private decimal _bankroll;
        private readonly bool _useMarketFallbackForAiDegeneracy;
        private readonly AutomationSettingsService _automationSettings;
        private string _primaryWindowHandle;
        private readonly object _driverLock = new();
        private static readonly Regex NonNumericCharactersRegex = new("[^0-9.,-]", RegexOptions.Compiled);
        private static readonly TimeSpan MinimumAutomationDelay = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan MaximumAutomationDelay = TimeSpan.FromMinutes(1.5);
        private static readonly Regex RaceTimeRegex = new(@"\b([01]?\d|2[0-3]):[0-5]\d\b", RegexOptions.Compiled);
        private static readonly Regex ScheduleRegionCountSuffixRegex = new(@"\s*\(\s*\d+\s*\)\s*$", RegexOptions.Compiled);
        private static readonly Regex CollapseWhitespaceRegex = new(@"\s+", RegexOptions.Compiled);
        private static readonly string[] ScheduleRegionButtonSelectors =
        {
            "li.country-tab > div",
            "li.country-tab > span",
            "li.country-tab > button",
            "li.country-tab > a",
            "#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > ui-view > div > div > div > div > div.content-page-center-column.racing-homepage-center-column > div:nth-child(2) > div > bf-todays-racing-mod > div > div > bf-todays-racing > section > div:nth-child(2) > div > div:nth-child(1) > ul > li.country-tab.active > div",
            "#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > ui-view > div > div > div > div > div.content-page-center-column.racing-homepage-center-column > div:nth-child(2) > div > bf-todays-racing-mod > div > div > bf-todays-racing > section > div:nth-child(2) > div > div:nth-child(1) > ul > li:nth-child(2) > div",
            "#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > ui-view > div > div > div > div > div.content-page-center-column.racing-homepage-center-column > div:nth-child(2) > div > bf-todays-racing-mod > div > div > bf-todays-racing > section > div:nth-child(2) > div > div:nth-child(1) > ul > li:nth-child(3) > div",
            "#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > ui-view > div > div > div > div > div.content-page-center-column.racing-homepage-center-column > div:nth-child(2) > div > bf-todays-racing-mod > div > div > bf-todays-racing > section > div:nth-child(2) > div > div:nth-child(1) > ul > li:nth-child(4) > div"
        };
        private readonly object _automationLock = new();
        private CancellationTokenSource? _automationCancellation;
        private Task? _automationTask;
        private readonly Dictionary<string, string?> _raceGoingByMarketId = new(StringComparer.OrdinalIgnoreCase);
        private readonly object _raceGoingLock = new();
        private static readonly TimeSpan SchedulePageReadyTimeout = TimeSpan.FromSeconds(120);

        public BetfairNavigationService(IConfiguration config, AutomationSettingsService automationSettings)
        {
            _username = config["Betfair:Username"] ?? HardCodedUsername;
            _password = config["Betfair:Password"] ?? HardCodedPassword;
            _configuredBankroll = config.GetValue<decimal?>("Betting:Bankroll") ?? 100m;
            _bankroll = _configuredBankroll;
            _useMarketFallbackForAiDegeneracy = config.GetValue<bool?>("Betting:UseMarketFallbackForAiDegeneracy") ?? true;
            _automationSettings = automationSettings ?? throw new ArgumentNullException(nameof(automationSettings));
            _driver = CreateWebDriver();
            _primaryWindowHandle = _driver.CurrentWindowHandle;
        }

        private IWebDriver CreateWebDriver()
        {
            var options = new ChromeOptions();
            options.AddArguments(
                "--disable-extensions",
                "--blink-settings=imagesEnabled=false",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage");

            return new ChromeDriver(options);
        }
        private void ResetWebDriver()
        {
            lock (_driverLock)
            {
                try
                {
                    _driver.Quit();
                }
                catch (Exception)
                {
                    // Ignore failures while tearing down the previous session.
                }

                try
                {
                    _driver.Dispose();
                }
                catch (Exception)
                {
                    // Ignore failures while disposing the previous driver instance.
                }

                _driver = CreateWebDriver();
                _primaryWindowHandle = _driver.CurrentWindowHandle;
                _bankroll = _configuredBankroll;

                lock (_raceGoingLock)
                {
                    _raceGoingByMarketId.Clear();
                }
            }
        }

        private static bool IsInvalidSessionException(Exception ex)
        {
            if (ex is WebDriverException webDriverException)
            {
                var message = webDriverException.Message ?? string.Empty;
                if (message.IndexOf("invalid session id", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return true;
                }
            }

            return ex.InnerException != null && IsInvalidSessionException(ex.InnerException);
        }

        private bool TryHandleInvalidSession(Exception ex)
        {
            if (!IsInvalidSessionException(ex))
            {
                return false;
            }

            ResetWebDriver();
            return true;
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

            string? originalHandle = null;
            string? originalUrl = null;
            try
            {
                originalHandle = _driver.CurrentWindowHandle;
            }
            catch (WebDriverException)
            {
                originalHandle = null;
            }

            try
            {
                originalUrl = _driver.Url;
            }
            catch (WebDriverException)
            {
                originalUrl = null;
            }

            await LoginAsync();

            var priorHandles = _driver.WindowHandles.ToList();
            var priorHandleSet = new HashSet<string>(priorHandles);
            var newHandles = new List<string>();
            var openedNewTab = false;
            var restoreUrl = string.IsNullOrWhiteSpace(raceUrl) ? originalUrl : raceUrl;

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
                var settings = _automationSettings.GetSnapshot();
                var scraper = new BetfairMarketScraper(
                    repo,
                    trainer,
                    bankroll,
                    settings,
                    _useMarketFallbackForAiDegeneracy,
                    GetRaceGoingSnapshot());
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

                    string? handleToRestore = null;
                    if (!string.IsNullOrEmpty(originalHandle) && _driver.WindowHandles.Contains(originalHandle))
                    {
                        handleToRestore = originalHandle;
                    }
                    else
                    {
                        handleToRestore = priorHandles.FirstOrDefault(h => _driver.WindowHandles.Contains(h));
                    }

                    if (string.IsNullOrEmpty(handleToRestore) && _driver.WindowHandles.Count > 0)
                    {
                        handleToRestore = _driver.WindowHandles[0];
                    }

                    if (!string.IsNullOrEmpty(handleToRestore))
                    {
                        try
                        {
                            _driver.SwitchTo().Window(handleToRestore);
                        }
                        catch (WebDriverException)
                        {
                            // Ignore failures when switching back to the original tab.
                        }
                    }
                }
                else if (priorHandles.Count > 0)
                {
                    var current = !string.IsNullOrEmpty(originalHandle) && _driver.WindowHandles.Contains(originalHandle)
                        ? originalHandle
                        : priorHandles[0];
                    if (!string.IsNullOrEmpty(current) && _driver.WindowHandles.Contains(current))
                    {
                        try
                        {
                            _driver.SwitchTo().Window(current);
                        }
                        catch (WebDriverException)
                        {
                            // Ignore failures when switching tabs.
                        }
                    }
                }

                if (!string.IsNullOrWhiteSpace(restoreUrl))
                {
                    try
                    {
                        var currentUrl = _driver.Url;
                        if (string.IsNullOrWhiteSpace(currentUrl) ||
                            !string.Equals(currentUrl, restoreUrl, StringComparison.OrdinalIgnoreCase))
                        {
                            _driver.Navigate().GoToUrl(restoreUrl);
                        }
                    }
                    catch (WebDriverException)
                    {
                        // Ignore failures when restoring the previous URL.
                    }
                }
            }
        }
        public IWebDriver Driver => _driver;

        public IReadOnlyList<BetRecommendation> ScrapeOpenRaceTabs(
           RacingRepository repo,
           HyperparameterTrainer trainer,
           out IReadOnlyCollection<string> missingScrapeFields,
           bool refreshBankrollFromPage = true)
        {
            var bankroll = GetEffectiveBankroll(refreshBankrollFromPage);
            var settings = _automationSettings.GetSnapshot();
            var scraper = new BetfairMarketScraper(
                repo,
                trainer,
                bankroll,
                settings,
                _useMarketFallbackForAiDegeneracy,
                GetRaceGoingSnapshot());
            var recommendations = scraper.ScrapeOpenRaceTabs(_driver);
            missingScrapeFields = scraper.MissingScrapeFieldDescriptions.ToArray();
            return recommendations;
        }

        private decimal GetEffectiveBankroll(bool refreshFromPage = true)
        {
            if (refreshFromPage)
            {
                var refreshed = TryRefreshBankrollFromPage();
                if (refreshed.HasValue && refreshed.Value > 0m)
                {
                    _bankroll = refreshed.Value;
                }
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
                 drv => FindDisplayedElement(drv, By.XPath("/html/body/ui-view/div/div/div[1]/div[1]/div/bf-ssc-header/div/div/div/div/div/div/table/tbody/tr/td[4]/div/div/div/form/div[3]/div[1]/table/tbody/tr[1]/td[2]")),
                drv => FindDisplayedElement(drv, By.CssSelector("td.ssc-wla[rel='main']")),
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

            wait.Until(d => IsScheduleUrl(d.Url));
        }
        public DayReportViewModel GenerateDayReport(RacingRepository repo, HyperparameterTrainer trainer)
        {
            ReturnToPrimaryWindow();
            CaptureRaceGoingFromSchedule();

            decimal bankroll;
            HyperparameterSummary? hyperparameters;
            List<RaceDayReport> orderedRaces;

            using (repo.BeginDayReportScope())
            {
                repo.ClearDayReportTables();
                bankroll = GetEffectiveBankroll();
                var settings = _automationSettings.GetSnapshot();
                var scraper = new BetfairMarketScraper(
                    repo,
                    trainer,
                    bankroll,
                    settings,
                    _useMarketFallbackForAiDegeneracy,
                    GetRaceGoingSnapshot());
                var races = scraper.ScrapeOpenRaceTabsForReport(_driver);
                orderedRaces = races
                    .OrderBy(r => GetRaceScheduleSortKey(r))
                    .ThenBy(r => r.RaceTitle ?? string.Empty, StringComparer.OrdinalIgnoreCase)
                    .ThenBy(r => r.MarketId, StringComparer.Ordinal)
                    .ToList();
                hyperparameters = scraper.LoadedHyperparameters;
                ReturnToPrimaryWindow();
            }

            return new DayReportViewModel
            {
                GeneratedAt = DateTime.UtcNow,
                Bankroll = bankroll,
                AiHyperparameters = hyperparameters,
                Races = orderedRaces
            };
        }
        public async Task OpenHorseRaceMeetingsInNewTabsAsync(
            int delayBetweenTabsMs = 0,
            bool closeExistingRaceTabs = true,
            TimeSpan? raceWindow = null,
            DateTime? windowReferenceUtc = null,
            TimeSpan? scheduleStartTime = null,
            TimeSpan? scheduleEndTime = null)
        {
            if (raceWindow.HasValue && (scheduleStartTime.HasValue || scheduleEndTime.HasValue))
            {
                throw new ArgumentException(
                    "Race window and schedule time filters cannot be used at the same time.",
                    nameof(raceWindow));
            }
            HashSet<string> existingMarketIds;
            if (closeExistingRaceTabs)
            {
                CloseAdditionalRaceTabs();
                existingMarketIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            }
            else
            {
                existingMarketIds = CaptureOpenRaceMarketIds();
            }

            ReturnToPrimaryWindow();
            EnsureSchedulePageReady();
            var scheduleRegionChanged = await TrySelectScheduleRegionAsync(scheduleRegion);
            if (scheduleRegionChanged)
            {
                await Task.Delay(200);
            }
            CaptureRaceGoingFromSchedule();
            static string? ExtractTimeToken(string? value)
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    return null;
                }

                var match = RaceTimeRegex.Match(value);
                return match.Success ? match.Value : null;
            }

            static bool Is24HourTime(string s)
            {
                return ExtractTimeToken(s) != null;
            }

            static DateTime? TryResolveRaceDateTime(string? text, DateTime reference)
            {
                var timeToken = ExtractTimeToken(text);
                if (string.IsNullOrEmpty(timeToken))
                {
                    return null;
                }

                if (!TimeSpan.TryParseExact(
                        timeToken,
                        new[] { @"h\:mm", @"hh\:mm" },
                        CultureInfo.InvariantCulture,
                        out var timeOfDay))
                {
                    if (!DateTime.TryParseExact(
                            timeToken,
                            new[] { "H:mm", "HH:mm" },
                            CultureInfo.InvariantCulture,
                            DateTimeStyles.None,
                            out var parsedDateTime))
                    {
                        return null;
                    }

                    timeOfDay = parsedDateTime.TimeOfDay;
                }

                var candidate = reference.Date.Add(timeOfDay);

                if (candidate < reference.AddMinutes(-5))
                {
                    candidate = candidate.AddDays(1);
                }

                return candidate;
            }

            DateTime windowReferenceLocal = DateTime.Now;
            DateTime windowStart = DateTime.MinValue;
            DateTime windowEnd = DateTime.MaxValue;
            if (raceWindow.HasValue)
            {
                var reference = windowReferenceUtc?.ToLocalTime() ?? DateTime.Now;
                windowReferenceLocal = reference;
                windowStart = reference;
                windowEnd = reference.Add(raceWindow.Value);
            }
            var hasScheduleFilters = scheduleStartTime.HasValue || scheduleEndTime.HasValue;
            var seen = raceWindow.HasValue
                ? null
                : new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var windowCandidates = raceWindow.HasValue
                ? new Dictionary<string, DateTime>(StringComparer.OrdinalIgnoreCase)
                : null;
            var scrollRoot = (IWebElement)((IJavaScriptExecutor)_driver)
                .ExecuteScript("return document.scrollingElement||document.body");

            for (int pass = 0; pass < 6; pass++)
            {
                var candidates = _driver.FindElements(By.CssSelector("a,button"));
                foreach (var el in candidates)
                {
                    try
                    {
                        if (!el.Displayed)
                        {
                            continue;
                        }

                        var textValue = el.Text;
                        if (string.IsNullOrWhiteSpace(textValue))
                        {
                            textValue = el.GetAttribute("innerText");
                        }

                        textValue = textValue?.Trim() ?? string.Empty;
                        if (!Is24HourTime(textValue))
                        {
                            continue;
                        }

                        var href = (string?)((IJavaScriptExecutor)_driver).ExecuteScript(
                            "const n=arguments[0];return (n.closest&&n.closest('a')&&n.closest('a').href)||n.href||null;",
                            el);
                        if (string.IsNullOrWhiteSpace(href))
                        {
                            continue;
                        }

                        if (!href.Contains("/horse-racing/", StringComparison.OrdinalIgnoreCase))
                        {
                            continue;
                        }
                        string? marketIdFromLink = null;
                        try
                        {
                            marketIdFromLink = BetfairMarketScraper.ExtractMarketId(href);
                        }
                        catch (Exception)
                        {
                            marketIdFromLink = null;
                        }
                        try
                        {
                            var going = ExtractGoingForElement(el);
                            RecordRaceGoing(href, going);
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"[Navigation] Failed to record going for race link: {ex.Message}");
                        }
                        if (!string.IsNullOrWhiteSpace(marketIdFromLink) && existingMarketIds.Contains(marketIdFromLink))
                        {
                            continue;
                        }
                        DateTime? raceTime = null;
                        if (raceWindow.HasValue || hasScheduleFilters)
                        {
                            raceTime = TryResolveRaceDateTime(textValue, windowReferenceLocal);
                            if (!raceTime.HasValue)
                            {
                                continue;
                            }
                        }
                        if (raceWindow.HasValue)
                        {

                            if (raceTime.Value < windowStart || raceTime.Value > windowEnd)
                            {
                                continue;
                            }

                            if (windowCandidates != null)
                            {
                                if (!windowCandidates.TryGetValue(href, out var existing) || raceTime.Value < existing)
                                {
                                    windowCandidates[href] = raceTime.Value;
                                }
                            }
                        }
                        else
                        {
                            if (hasScheduleFilters)
                            {
                                var timeOfDay = raceTime!.Value.TimeOfDay;
                                if (scheduleStartTime.HasValue && timeOfDay < scheduleStartTime.Value)
                                {
                                    continue;
                                }

                                if (scheduleEndTime.HasValue && timeOfDay > scheduleEndTime.Value)
                                {
                                    continue;
                                }
                            }

                            seen!.Add(href);
                        }
                    }
                    catch (StaleElementReferenceException)
                    {
                        continue;
                    }
                }

                ((IJavaScriptExecutor)_driver).ExecuteScript(
                    "arguments[0].scrollTop=arguments[0].scrollTop+Math.min(1200,window.innerHeight);",
                    scrollRoot);
                await Task.Delay(150);
            }

            if (!raceWindow.HasValue && seen!.Count == 0)
            {
                var links = _driver.FindElements(By.CssSelector("a[href*='/horse-racing/']"));
                foreach (var anchor in links)
                {
                    var href = anchor.GetAttribute("href");
                    if (!string.IsNullOrWhiteSpace(href))
                    {
                        string? marketIdFromLink = null;
                        try
                        {
                            marketIdFromLink = BetfairMarketScraper.ExtractMarketId(href);
                        }
                        catch (Exception)
                        {
                            marketIdFromLink = null;
                        }
                        try
                        {
                            var going = ExtractGoingForElement(anchor);
                            RecordRaceGoing(href, going);
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"[Navigation] Failed to capture going from fallback link: {ex.Message}");
                        }
                        if (!string.IsNullOrWhiteSpace(marketIdFromLink) && existingMarketIds.Contains(marketIdFromLink))
                        {
                            continue;
                        }
                        if (hasScheduleFilters)
                        {
                            var textValue = anchor.Text;
                            if (string.IsNullOrWhiteSpace(textValue))
                            {
                                textValue = anchor.GetAttribute("innerText");
                            }

                            textValue = textValue?.Trim() ?? string.Empty;
                            var raceTime = TryResolveRaceDateTime(textValue, windowReferenceLocal);
                            if (!raceTime.HasValue)
                            {
                                continue;
                            }

                            var timeOfDay = raceTime.Value.TimeOfDay;
                            if (scheduleStartTime.HasValue && timeOfDay < scheduleStartTime.Value)
                            {
                                continue;
                            }

                            if (scheduleEndTime.HasValue && timeOfDay > scheduleEndTime.Value)
                            {
                                continue;
                            }
                        }
                        seen.Add(href);
                    }
                }
            }

            IEnumerable<string> urlsToOpen;
            if (raceWindow.HasValue)
            {
                urlsToOpen = (windowCandidates != null && windowCandidates.Count > 0)
                    ? windowCandidates
                        .OrderBy(kvp => kvp.Value)
                        .Select(kvp => kvp.Key)
                        .ToList()
                    : Array.Empty<string>();
            }
            else
            {
                urlsToOpen = seen!;
            }

            var newlyOpenedMarketIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var url in urlsToOpen)
            {
                string? marketId = null;
                try
                {
                    marketId = BetfairMarketScraper.ExtractMarketId(url);
                }
                catch (Exception)
                {
                    marketId = null;
                }

                if (!string.IsNullOrWhiteSpace(marketId))
                {
                    if (existingMarketIds.Contains(marketId) || !newlyOpenedMarketIds.Add(marketId))
                    {
                        continue;
                    }
                }
                ((IJavaScriptExecutor)_driver).ExecuteScript("window.open(arguments[0],'_blank');", url);
                if (delayBetweenTabsMs > 0)
                {
                    await Task.Delay(delayBetweenTabsMs);
                }
            }
        }
        private async Task<bool> TrySelectScheduleRegionAsync(string? region)
        {
            if (string.IsNullOrWhiteSpace(region))
            {
                return false;
            }

            var normalizedTarget = NormalizeScheduleRegionText(region);
            if (normalizedTarget.Length == 0 || normalizedTarget.Equals("All", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            const int maxAttempts = 5;
            for (var attempt = 0; attempt < maxAttempts; attempt++)
            {
                IReadOnlyCollection<IWebElement> candidates;
                try
                {
                    candidates = FindScheduleRegionButtons();
                }
                catch (WebDriverException)
                {
                    candidates = Array.Empty<IWebElement>();
                }

                if (candidates.Count == 0)
                {
                    await Task.Delay(200);
                    continue;
                }

                foreach (var candidate in candidates)
                {
                    try
                    {
                        var candidateText = NormalizeScheduleRegionText(GetElementTextSafe(candidate));
                        if (candidateText.Length == 0)
                        {
                            continue;
                        }

                        if (!candidateText.Equals(normalizedTarget, StringComparison.OrdinalIgnoreCase))
                        {
                            continue;
                        }

                        ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", candidate);

                        try
                        {
                            if (candidate.Enabled)
                            {
                                candidate.Click();
                            }
                            else
                            {
                                ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", candidate);
                            }
                        }
                        catch (Exception)
                        {
                            ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", candidate);
                        }

                        try
                        {
                            var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(5))
                            {
                                PollingInterval = TimeSpan.FromMilliseconds(200)
                            };
                            wait.IgnoreExceptionTypes(
                                typeof(StaleElementReferenceException),
                                typeof(NoSuchElementException));
                            wait.Until(d => IsScheduleRegionActive(d, normalizedTarget));
                        }
                        catch (WebDriverTimeoutException)
                        {
                            // Ignore if the active state does not update within the timeout.
                        }

                        return true;
                    }
                    catch (StaleElementReferenceException)
                    {
                        break;
                    }
                }

                await Task.Delay(200);
            }

            return false;
        }

        private IReadOnlyCollection<IWebElement> FindScheduleRegionButtons()
        {
            try
            {
                var general = _driver.FindElements(By.CssSelector("li.country-tab > div, li.country-tab > span, li.country-tab > button, li.country-tab > a"));
                if (general.Count > 0)
                {
                    return general;
                }
            }
            catch (WebDriverException)
            {
                // Ignore if the general selector is invalid for the current page layout.
            }

            foreach (var selector in ScheduleRegionButtonSelectors)
            {
                try
                {
                    var candidates = _driver.FindElements(By.CssSelector(selector));
                    if (candidates.Count > 0)
                    {
                        return candidates;
                    }
                }
                catch (WebDriverException)
                {
                    // Ignore selectors that are not valid for the current layout.
                }
            }

            return Array.Empty<IWebElement>();
        }

        private static string NormalizeScheduleRegionText(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var trimmed = value.Trim();
            trimmed = ScheduleRegionCountSuffixRegex.Replace(trimmed, string.Empty);
            trimmed = CollapseWhitespaceRegex.Replace(trimmed, " ");
            return trimmed.Trim();
        }

        private static string GetElementTextSafe(IWebElement element)
        {
            if (element == null)
            {
                return string.Empty;
            }

            try
            {
                var text = element.Text;
                if (!string.IsNullOrWhiteSpace(text))
                {
                    return text;
                }
            }
            catch (WebDriverException)
            {
            }

            try
            {
                var text = element.GetAttribute("innerText");
                if (!string.IsNullOrWhiteSpace(text))
                {
                    return text;
                }
            }
            catch (WebDriverException)
            {
            }

            try
            {
                var text = element.GetAttribute("textContent");
                if (!string.IsNullOrWhiteSpace(text))
                {
                    return text;
                }
            }
            catch (WebDriverException)
            {
            }

            return string.Empty;
        }

        private static bool IsScheduleRegionActive(IWebDriver driver, string normalizedRegion)
        {
            if (string.IsNullOrWhiteSpace(normalizedRegion))
            {
                return false;
            }

            try
            {
                var activeElements = driver.FindElements(By.CssSelector("li.country-tab.active > div, li.country-tab.active > span, li.country-tab.active > button, li.country-tab.active > a"));
                foreach (var element in activeElements)
                {
                    var text = NormalizeScheduleRegionText(GetElementTextSafe(element));
                    if (text.Length == 0)
                    {
                        continue;
                    }

                    if (text.Equals(normalizedRegion, StringComparison.OrdinalIgnoreCase))
                    {
                        return true;
                    }
                }
            }
            catch (WebDriverException)
            {
                return false;
            }

            return false;
        }

        private void EnsureSchedulePageReady()
        {
            const int maxAttempts = 2;
            WebDriverTimeoutException? lastTimeout = null;

            for (var attempt = 0; attempt < maxAttempts; attempt++)
            {
                try
                {
                    EnsureScheduleNavigation();

                    var wait = new WebDriverWait(_driver, SchedulePageReadyTimeout)
                    {
                        PollingInterval = TimeSpan.FromMilliseconds(250)
                    };
                    wait.IgnoreExceptionTypes(
                        typeof(JavaScriptException),
                        typeof(NoSuchElementException),
                        typeof(StaleElementReferenceException));

                    wait.Until(d =>
                    {
                        var state = ((IJavaScriptExecutor)d).ExecuteScript("return document.readyState");
                        var stateText = state?.ToString();
                        return string.Equals("complete", stateText, StringComparison.OrdinalIgnoreCase)
                            || string.Equals("interactive", stateText, StringComparison.OrdinalIgnoreCase);
                    });

                    wait.Until(d =>
                    {
                        if (!IsScheduleUrl(d.Url))
                        {
                            return false;
                        }

                        if (IsScheduleContentAvailable(d))
                        {
                            return true;
                        }

                        var elements = d.FindElements(By.CssSelector("a,button"));
                        return elements != null && elements.Count > 0;
                    });

                    return;
                }
                catch (WebDriverTimeoutException ex)
                {
                    lastTimeout = ex;
                    if (attempt + 1 == maxAttempts)
                    {
                        break;
                    }

                    try
                    {
                        _driver.Navigate().Refresh();
                    }
                    catch (Exception)
                    {
                        // Ignore refresh failures; we'll retry with the existing page state.
                    }

                    Thread.Sleep(TimeSpan.FromSeconds(2));
                }
            }

            throw lastTimeout ?? new WebDriverTimeoutException("Failed to ensure the Betfair schedule page is ready.");
        }
        private void EnsureScheduleNavigation()
        {
            var currentUrl = TryGetCurrentUrl();
            if (IsScheduleUrl(currentUrl))
            {
                return;
            }

            for (var attempt = 0; attempt < 2; attempt++)
            {
                try
                {
                    _driver.Navigate().GoToUrl(HorseRacingScheduleUrl);
                    return;
                }
                catch (WebDriverException ex) when (TryHandleInvalidSession(ex))
                {
                    // Driver was reset due to an invalid session; retry navigation with the new instance.
                }
            }
        }

        private string? TryGetCurrentUrl()
        {
            try
            {
                return _driver.Url;
            }
            catch (WebDriverException)
            {
                return null;
            }
        }

        private static bool IsScheduleUrl(string? url)
        {
            if (string.IsNullOrWhiteSpace(url))
            {
                return false;
            }

            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
            {
                return false;
            }

            var path = uri.AbsolutePath?.TrimEnd('/') ?? string.Empty;
            if (path.IndexOf("/market/", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return false;
            }

            if (path.IndexOf("horse-racing-betting", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return true;
            }

            return path.EndsWith("/horse-racing", StringComparison.OrdinalIgnoreCase);
        }
        private static bool IsScheduleContentAvailable(IWebDriver driver)
        {
            try
            {
                var selectors = new[]
                {
                    "[data-testid='racing-schedule']",
                    "[data-testid='racing-event']",
                    ".meeting-description",
                    "[data-testid='racing-race-list'] li",
                    "li .meeting-description"
                };

                foreach (var selector in selectors)
                {
                    var elements = driver.FindElements(By.CssSelector(selector));
                    if (elements.Any(e => e.Displayed))
                    {
                        return true;
                    }
                }

                var raceLinks = driver.FindElements(By.CssSelector("a[href*='/horse-racing/']"));
                if (raceLinks.Count > 0)
                {
                    return true;
                }

                try
                {
                    var body = driver.FindElement(By.TagName("body"));
                    var text = body?.Text;
                    if (!string.IsNullOrWhiteSpace(text) &&
                        text.IndexOf("no races", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        return true;
                    }
                }
                catch (NoSuchElementException)
                {
                    // Ignore if the body element is not yet available.
                }
            }
            catch (WebDriverException)
            {
                return false;
            }

            return false;
        }
        private IReadOnlyDictionary<string, string?> GetRaceGoingSnapshot() { lock (_raceGoingLock) { return new Dictionary<string, string?>(_raceGoingByMarketId, StringComparer.OrdinalIgnoreCase); } }
        private void RecordRaceGoing(string? href, string? going) { if (string.IsNullOrWhiteSpace(href)) { return; } var marketId = BetfairMarketScraper.ExtractMarketId(href); if (string.IsNullOrWhiteSpace(marketId)) { return; } var trimmedGoing = string.IsNullOrWhiteSpace(going) ? null : going!.Trim(); lock (_raceGoingLock) { if (!string.IsNullOrEmpty(trimmedGoing)) { _raceGoingByMarketId[marketId] = trimmedGoing; } else if (!_raceGoingByMarketId.ContainsKey(marketId)) { _raceGoingByMarketId[marketId] = null; } } }
        private void CaptureRaceGoingFromSchedule() { lock (_raceGoingLock) { _raceGoingByMarketId.Clear(); } try { var js = (IJavaScriptExecutor)_driver; const string script = @"
const results=[];const textOrEmpty=node=>{if(!node)return '';const raw=node.textContent||node.innerText||'';return raw.trim();};const items=Array.from(document.querySelectorAll('li')).filter(li=>li.querySelector('.meeting-description'));for(const item of items){let href='';const anchor=item.querySelector(""a[href*='/horse-racing/']"");if(anchor&&anchor.href){href=anchor.href;}else{const button=item.querySelector('button');if(button){const link=button.closest(""a[href*='/horse-racing/']"");if(link&&link.href){href=link.href;}}}if(!href){continue;}const going=textOrEmpty(item.querySelector(""div.racetrack-conditions, .racetrack-conditions, [data-testid='racetrack-conditions']""));results.push({href,going});}return results;"; var raw = js.ExecuteScript(script); if (raw is IEnumerable<object> entries) { foreach (var entry in entries) { string? href = null; string? going = null; switch (entry) { case IReadOnlyDictionary<string, object?> dict: if (dict.TryGetValue("href", out var hrefValue)) { href = hrefValue?.ToString(); } if (dict.TryGetValue("going", out var goingValue)) { going = goingValue?.ToString(); } break; case IDictionary legacyDict: if (legacyDict.Contains("href")) { href = legacyDict["href"]?.ToString(); } if (legacyDict.Contains("going")) { going = legacyDict["going"]?.ToString(); } break; } RecordRaceGoing(href, going); } } } catch (Exception ex) { Console.Error.WriteLine($"[Navigation] Failed to capture going information from schedule: {ex.Message}"); } }
        private string? ExtractGoingForElement(IWebElement element) { if (element == null) { return null; } try { var js = (IJavaScriptExecutor)_driver; const string script = @"
const el=arguments[0];const selectors=['div.racetrack-conditions','.racetrack-conditions',""[data-testid='racetrack-conditions']""];const textOrEmpty=node=>{if(!node)return '';const raw=node.textContent||node.innerText||'';return raw.trim();};let current=el;while(current){for(const selector of selectors){const candidate=current.querySelector?current.querySelector(selector):null;if(candidate){const value=textOrEmpty(candidate);if(value){return value;}}}current=current.parentElement;}return '';"; var result = js.ExecuteScript(script, element); if (result is string text) { var trimmed = text.Trim(); return string.IsNullOrEmpty(trimmed) ? null : trimmed; } } catch (StaleElementReferenceException) { return null; } catch (Exception ex) { Console.Error.WriteLine($"[Navigation] Failed to extract going text: {ex.Message}"); } return null; }

        public void StartAutomatedBettingLoop(
            RacingRepository repo,
            HyperparameterTrainer trainer,
            TimeSpan raceWindow,
            TimeSpan refreshLeadTime,
            TimeSpan initialDelay)
        {
            if (repo == null)
            {
                throw new ArgumentNullException(nameof(repo));
            }

            if (trainer == null)
            {
                throw new ArgumentNullException(nameof(trainer));
            }

            if (refreshLeadTime < TimeSpan.Zero)
            {
                refreshLeadTime = TimeSpan.Zero;
            }

            if (raceWindow <= TimeSpan.Zero)
            {
                throw new ArgumentOutOfRangeException(nameof(raceWindow), "Race window must be positive.");
            }

            if (initialDelay < TimeSpan.Zero)
            {
                initialDelay = TimeSpan.Zero;
            }

            lock (_automationLock)
            {
                StopAutomatedBettingLoop();

                var cts = new CancellationTokenSource();
                _automationCancellation = cts;
                var token = cts.Token;

                _automationTask = Task.Run(
                    () => RunAutomatedBettingLoopAsync(repo, trainer, raceWindow, refreshLeadTime, initialDelay, token),
                    token);
            }
        }

        public void StopAutomatedBettingLoop()
        {
            lock (_automationLock)
            {
                var cts = _automationCancellation;
                if (cts != null)
                {
                    try
                    {
                        cts.Cancel();
                    }
                    catch (ObjectDisposedException)
                    {
                        // Ignore if already disposed.
                    }

                    cts.Dispose();
                }

                _automationCancellation = null;
                _automationTask = null;
            }
        }

        private async Task RunAutomatedBettingLoopAsync(
            RacingRepository repo,
            HyperparameterTrainer trainer,
            TimeSpan raceWindow,
            TimeSpan refreshLeadTime,
            TimeSpan initialDelay,
            CancellationToken cancellationToken)
        {
            var normalizedInitialDelay = ClampAutomationDelay(initialDelay, allowZero: true);
            if (normalizedInitialDelay > TimeSpan.Zero)
            {
                try
                {
                    await Task.Delay(normalizedInitialDelay, cancellationToken);
                }
                catch (TaskCanceledException)
                {
                    return;
                }
            }
            while (!cancellationToken.IsCancellationRequested)
            {
                var cycleStartUtc = DateTime.UtcNow;

                try
                {
                    await LoginAsync();
                    await OpenHorseRaceMeetingsInNewTabsAsync(
                        closeExistingRaceTabs: false,
                        raceWindow: raceWindow,
                        windowReferenceUtc: cycleStartUtc);

                    var recommendations = ScrapeOpenRaceTabs(
                        repo,
                        trainer,
                        out _,
                        refreshBankrollFromPage: false);

                    UpdateBankrollFromExecutedBets(recommendations);
                }
                catch (Exception ex)
                {
                    if (TryHandleInvalidSession(ex))
                    {
                        Console.Error.WriteLine("[Automation] Restarted browser session after it became invalid.");
                    }
                    else
                    {
                        Console.Error.WriteLine($"[Automation] Failed to process betting cycle: {ex.Message}");
                    }
                }

                var cycleEndUtc = DateTime.UtcNow;
                var requestedInterval = raceWindow - refreshLeadTime;
                var normalizedInterval = ClampAutomationDelay(requestedInterval, allowZero: false);

                var elapsed = cycleEndUtc - cycleStartUtc;
                var delay = normalizedInterval - elapsed;
                delay = ClampAutomationDelay(delay, allowZero: false);

                try
                {
                    await Task.Delay(delay, cancellationToken);
                }
                catch (TaskCanceledException)
                {
                    break;
                }
            }
        }
        private void UpdateBankrollFromExecutedBets(IEnumerable<BetRecommendation> recommendations)
        {
            if (recommendations == null)
            {
                return;
            }

            var totalStaked = recommendations
                .Where(r => r != null && r.Stake > 0m)
                .Sum(r => r.Stake);

            if (totalStaked <= 0m)
            {
                return;
            }

            _bankroll -= totalStaked;
            if (_bankroll < 0m)
            {
                _bankroll = 0m;
            }
        }
        private HashSet<string> CaptureOpenRaceMarketIds()
        {
            var existing = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            IReadOnlyCollection<string> handles;

            try
            {
                handles = _driver.WindowHandles;
            }
            catch (WebDriverException)
            {
                return existing;
            }

            string? originalHandle = null;
            try
            {
                originalHandle = _driver.CurrentWindowHandle;
            }
            catch (WebDriverException)
            {
                originalHandle = null;
            }

            foreach (var handle in handles)
            {
                try
                {
                    _driver.SwitchTo().Window(handle);
                    var url = _driver.Url;
                    var marketId = BetfairMarketScraper.ExtractMarketId(url);
                    if (!string.IsNullOrWhiteSpace(marketId))
                    {
                        existing.Add(marketId);
                    }
                }
                catch (WebDriverException)
                {
                    continue;
                }
            }

            try
            {
                if (!string.IsNullOrEmpty(originalHandle) && _driver.WindowHandles.Contains(originalHandle))
                {
                    _driver.SwitchTo().Window(originalHandle);
                }
            }
            catch (WebDriverException)
            {
                ReturnToPrimaryWindow();
            }

            return existing;
        }

        private static TimeSpan ClampAutomationDelay(TimeSpan value, bool allowZero)
        {
            if (value <= TimeSpan.Zero)
            {
                return allowZero ? TimeSpan.Zero : MinimumAutomationDelay;
            }

            if (value > MaximumAutomationDelay)
            {
                return MaximumAutomationDelay;
            }

            return value;
        }
        public async Task<bool> TrySelectHorseRacingDayAsync(int daysFromToday)
        {
            if (daysFromToday == 0)
            {
                return true;
            }

            if (daysFromToday < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(daysFromToday), "Day offset cannot be negative.");
            }

            ReturnToPrimaryWindow();

            var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(10));
            var targetDate = DateTime.Today.AddDays(daysFromToday);
            var searchTerms = new List<string>();

            if (daysFromToday == 1)
            {
                searchTerms.Add("tomorrow");
            }

            searchTerms.Add(targetDate.ToString("ddd dd MMM", CultureInfo.InvariantCulture));
            searchTerms.Add(targetDate.ToString("ddd d MMM", CultureInfo.InvariantCulture));
            searchTerms.Add(targetDate.ToString("dd MMM", CultureInfo.InvariantCulture));
            searchTerms.Add(targetDate.ToString("d MMM", CultureInfo.InvariantCulture));
            searchTerms.Add(targetDate.ToString("dd MMMM", CultureInfo.InvariantCulture));
            searchTerms.Add(targetDate.ToString("d MMMM", CultureInfo.InvariantCulture));
            searchTerms.Add(targetDate.ToString("dddd", CultureInfo.InvariantCulture));
            searchTerms.Add(targetDate.ToString("ddd", CultureInfo.InvariantCulture));

            var selectors = new List<By>();
            var seenSelectors = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var term in searchTerms.Where(t => !string.IsNullOrWhiteSpace(t)))
            {
                var trimmed = term.Trim();
                var lower = trimmed.ToLowerInvariant();

                string BuildContainsXPath(string node) =>
                    $"//{node}[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{lower}')]";

                foreach (var node in new[] { "button", "a", "span", "div" })
                {
                    var xpath = BuildContainsXPath(node);
                    if (seenSelectors.Add(xpath))
                    {
                        selectors.Add(By.XPath(xpath));
                    }
                }

                var roleXPath =
                    $"//*[@role='button' and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{lower}')]";
                if (seenSelectors.Add(roleXPath))
                {
                    selectors.Add(By.XPath(roleXPath));
                }
            }

            var isoDate = targetDate.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
            foreach (var selector in new[]
            {
                $"[data-day='{isoDate}']",
                $"[data-date='{isoDate}']",
                $"[data-testid='day-{isoDate}']",
                $"button[aria-label*='{isoDate}']",
                $"a[aria-label*='{isoDate}']"
            })
            {
                if (seenSelectors.Add(selector))
                {
                    selectors.Add(By.CssSelector(selector));
                }
            }

            foreach (var selector in selectors)
            {
                try
                {
                    var element = wait.Until(driver =>
                    {
                        var elements = driver.FindElements(selector);
                        foreach (var candidate in elements)
                        {
                            if (candidate.Displayed)
                            {
                                return candidate;
                            }
                        }

                        return null;
                    });

                    if (element == null)
                    {
                        continue;
                    }

                    try
                    {
                        var clickable = wait.Until(ExpectedConditions.ElementToBeClickable(selector));
                        clickable.Click();
                    }
                    catch (WebDriverException)
                    {
                        try
                        {
                            ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", element);
                        }
                        catch (Exception)
                        {
                            continue;
                        }
                    }

                    await Task.Delay(750);
                    return true;
                }
                catch (WebDriverTimeoutException)
                {
                    continue;
                }
            }

            return false;
        }

        private void CloseAdditionalRaceTabs()
        {
            var handles = _driver.WindowHandles.ToList();
            foreach (var handle in handles)
            {
                if (string.Equals(handle, _primaryWindowHandle, StringComparison.Ordinal))
                {
                    continue;
                }

                try
                {
                    _driver.SwitchTo().Window(handle);
                    _driver.Close();
                }
                catch (WebDriverException)
                {
                    // Ignore failures when closing extraneous tabs.
                }
            }

            ReturnToPrimaryWindow();
        }

        private void ReturnToPrimaryWindow()
        {
            try
            {
                if (!string.IsNullOrEmpty(_primaryWindowHandle) &&
                    _driver.WindowHandles.Contains(_primaryWindowHandle))
                {
                    _driver.SwitchTo().Window(_primaryWindowHandle);
                }
                else if (_driver.WindowHandles.Count > 0)
                {
                    _driver.SwitchTo().Window(_driver.WindowHandles[0]);
                }
            }
            catch (WebDriverException)
            {
                // Ignore if the window cannot be focused; caller will handle subsequent failures.
            }
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
        public void Dispose()
        {
            StopAutomatedBettingLoop();
            lock (_driverLock)
            {
                try
                {
                    _driver.Quit();
                }
                catch (Exception)
                {
                    // Ignore cleanup errors while shutting down the driver.
                }

                try
                {
                    _driver.Dispose();
                }
                catch (Exception)
                {
                    // Ignore cleanup errors while shutting down the driver.
                }
            }
        }
    }
}
