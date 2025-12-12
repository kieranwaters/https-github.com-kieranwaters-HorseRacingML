using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using OneOf.Types;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using SeleniumExtras.WaitHelpers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

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
        private bool _isHeadless;
        private readonly decimal _configuredBankroll;
        private decimal _bankroll;
        private readonly bool _useMarketFallbackForAiDegeneracy;
        private readonly AutomationSettingsService _automationSettings;
        private string _primaryWindowHandle;
        private readonly object _driverLock = new();
        private readonly AIOddsCalculator _aiOddsCalculator;
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
        private readonly Dictionary<string, string?> _raceGoingByVenue = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _raceTabHandles = new(StringComparer.OrdinalIgnoreCase);
        private readonly object _raceTabLock = new();
        private static readonly TimeSpan SchedulePageReadyTimeout = TimeSpan.FromSeconds(120);
        private string? _activeScheduleRegion;

        public BetfairNavigationService(IConfiguration config, AutomationSettingsService automationSettings, AIOddsCalculator aiOddsCalculator)
        {
            _username = config["Betfair:Username"] ?? HardCodedUsername;
            _password = config["Betfair:Password"] ?? HardCodedPassword;
            _aiOddsCalculator = aiOddsCalculator;
            _configuredBankroll = config.GetValue<decimal?>("Betting:Bankroll") ?? 100m;
            _bankroll = _configuredBankroll;
            _useMarketFallbackForAiDegeneracy = config.GetValue<bool?>("Betting:UseMarketFallbackForAiDegeneracy") ?? true;
            _automationSettings = automationSettings ?? throw new ArgumentNullException(nameof(automationSettings));
            _isHeadless = false;
            _driver = CreateWebDriver(_isHeadless);
            _primaryWindowHandle = _driver.CurrentWindowHandle;
        }
        public string? GetActiveScheduleRegion()
        {
            return _activeScheduleRegion;
        }
        private IWebDriver CreateWebDriver(bool headless)
        {
            var options = new ChromeOptions();
            options.AddArguments(
                "--disable-extensions",
                "--blink-settings=imagesEnabled=false",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage");

            if (headless)
            {
                options.AddArgument("--headless=new");
            }

            return new ChromeDriver(options);
        }

        public void EnsureDriverMode(bool headless)
        {
            lock (_driverLock)
            {
                if (_isHeadless == headless && _driver != null)
                {
                    try
                    {
                        var _ = _driver.CurrentWindowHandle; // Check if alive
                        return;
                    }
                    catch
                    {
                        // Driver is dead, proceed to reset
                    }
                }

                try
                {
                    _driver.Quit();
                }
                catch { }

                try
                {
                    _driver.Dispose();
                }
                catch { }

                _isHeadless = headless;
                _driver = CreateWebDriver(_isHeadless);
                _primaryWindowHandle = _driver.CurrentWindowHandle;
                _bankroll = _configuredBankroll;

                lock (_raceGoingLock)
                {
                    _raceGoingByMarketId.Clear();
                    _raceGoingByVenue.Clear();
                }
                lock (_raceTabLock)
                {
                    _raceTabHandles.Clear();
                }
            }
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

                _driver = CreateWebDriver(_isHeadless);
                _primaryWindowHandle = _driver.CurrentWindowHandle;
                _bankroll = _configuredBankroll;

                lock (_raceGoingLock)
                {
                    _raceGoingByMarketId.Clear();
                    _raceGoingByVenue.Clear();
                }
                lock (_raceTabLock)
                {
                    _raceTabHandles.Clear();
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
        // ... (methods kept: TryGetExistingRaceTab, RefreshRaceAsync, ShouldRecalculateAiForMarket, Driver prop, GetEffectiveBankroll, TryRefreshBankrollFromPage, FindDisplayedElement, TryParseCurrency, LoginAsync) ...

        private bool TryGetExistingRaceTab(string marketId, out string? handle)
        {
            if (string.IsNullOrWhiteSpace(marketId))
            {
                handle = null;
                return false;
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

            handle = null;
            bool found = false;
            string? resolvedHandle = null;

            bool ValidateCandidate(string? candidate, out string? candidateHandle)
            {
                candidateHandle = null;

                if (string.IsNullOrWhiteSpace(candidate))
                {
                    return false;
                }

                var handles = _driver.WindowHandles;
                if (!handles.Contains(candidate))
                {
                    return false;
                }

                try
                {
                    _driver.SwitchTo().Window(candidate);
                }
                catch (WebDriverException)
                {
                    return false;
                }

                try
                {
                    var currentUrl = _driver.Url;
                    var currentMarketId = BetfairMarketScraper.ExtractMarketId(currentUrl);
                    if (!string.IsNullOrWhiteSpace(currentMarketId) &&
                        string.Equals(currentMarketId, marketId, StringComparison.OrdinalIgnoreCase))
                    {
                        candidateHandle = candidate;
                        return true;
                    }
                }
                catch (WebDriverException)
                {
                    // Ignore failures while reading the current tab state.
                }

                return false;
            }

            string? cachedHandle = null;
            lock (_raceTabLock)
            {
                if (_raceTabHandles.TryGetValue(marketId, out var known))
                {
                    cachedHandle = known;
                }
            }

            if (!string.IsNullOrWhiteSpace(cachedHandle) &&
                ValidateCandidate(cachedHandle, out var cachedValidatedHandle))
            {
                resolvedHandle = cachedValidatedHandle;
                found = true;
            }
            else if (!string.IsNullOrWhiteSpace(cachedHandle))
            {
                lock (_raceTabLock)
                {
                    _raceTabHandles.Remove(marketId);
                }
            }

            if (!found)
            {
                foreach (var candidate in _driver.WindowHandles)
                {
                    if (ValidateCandidate(candidate, out var validatedHandle))
                    {
                        resolvedHandle = validatedHandle;
                        found = true;
                        lock (_raceTabLock)
                        {
                            _raceTabHandles[marketId] = validatedHandle;
                        }
                        break;
                    }
                }
            }

            if (!string.IsNullOrEmpty(originalHandle) && _driver.WindowHandles.Contains(originalHandle))
            {
                try
                {
                    _driver.SwitchTo().Window(originalHandle);
                }
                catch (WebDriverException)
                {
                    // Ignore failures when restoring the original tab.
                }
            }

            handle = resolvedHandle;
            return found;
        }

        public async Task<RaceDayReport?> RefreshRaceAsync(
            string raceUrl,
            RacingRepository repo,
            HyperparameterTrainer trainer,
            bool includeAiProbabilities = true,
            bool refreshBankrollFromPage = true,
            bool requireLogin = true)
        {
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

                if (requireLogin)
                {
                    await LoginAsync();
                }

                var priorHandles = _driver.WindowHandles.ToList();
                var priorHandleSet = new HashSet<string>(priorHandles);
                var newHandles = new List<string>();
                var openedNewTab = false;
                var reusedExistingTab = false;
                var restoreUrl = string.IsNullOrWhiteSpace(raceUrl) ? originalUrl : raceUrl;
                var targetMarketId = BetfairMarketScraper.ExtractMarketId(raceUrl);

                if (!string.IsNullOrWhiteSpace(targetMarketId) &&
                    TryGetExistingRaceTab(targetMarketId, out var existingHandle) &&
                    !string.IsNullOrWhiteSpace(existingHandle))
                {
                    newHandles.Add(existingHandle);
                    reusedExistingTab = true;
                }

                try
                {
                    if (!reusedExistingTab)
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
                    }

                    if (newHandles.Count == 0)
                    {
                        return null;
                    }

                    var bankroll = GetEffectiveBankroll(refreshBankrollFromPage);
                    var settings = _automationSettings.GetSnapshot();
                    var scraper = new BetfairMarketScraper(
                        repo,
                        trainer,
                         _aiOddsCalculator,
                        bankroll,
                        settings,
                        _useMarketFallbackForAiDegeneracy,
                        GetRaceGoingSnapshot(),
                        computeAiProbabilities: includeAiProbabilities,
                        scheduleRegion: GetActiveScheduleRegion(),
                        raceGoingByVenueLookup: GetRaceGoingByVenueSnapshot());
                    var races = scraper.ScrapeOpenRaceTabsForReport(_driver, newHandles);

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
                    if (!string.IsNullOrWhiteSpace(targetMarketId) && newHandles.Count > 0)
                    {
                        var handleToRemember = newHandles[0];
                        if (!string.IsNullOrWhiteSpace(handleToRemember))
                        {
                            lock (_raceTabLock)
                            {
                                _raceTabHandles[targetMarketId] = handleToRemember;
                            }
                        }
                    }

                    string? handleToRestore = null;
                    if (!string.IsNullOrEmpty(originalHandle) && _driver.WindowHandles.Contains(originalHandle))
                    {
                        handleToRestore = originalHandle;
                    }
                    else if (priorHandles.Count > 0)
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
        }
        public bool ShouldRecalculateAiForMarket(string? marketId, string? currentGoing, out string? latestGoing)
        {
            latestGoing = null;

            if (string.IsNullOrWhiteSpace(marketId))
            {
                return false;
            }

            var normalizedMarketId = marketId.Trim();
            if (normalizedMarketId.Length == 0)
            {
                return false;
            }

            var normalizedCurrent = string.IsNullOrWhiteSpace(currentGoing)
                ? null
                : currentGoing.Trim();

            var snapshot = GetRaceGoingSnapshot();
            if (!snapshot.TryGetValue(normalizedMarketId, out var knownGoing) || string.IsNullOrWhiteSpace(knownGoing))
            {
                latestGoing = string.IsNullOrWhiteSpace(knownGoing) ? null : knownGoing.Trim();
                return false;
            }

            latestGoing = string.IsNullOrWhiteSpace(knownGoing) ? null : knownGoing.Trim();
            return !string.Equals(latestGoing, normalizedCurrent, StringComparison.OrdinalIgnoreCase);
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
                _aiOddsCalculator,
                bankroll,
                settings,
                _useMarketFallbackForAiDegeneracy,
                GetRaceGoingSnapshot(),
                scheduleRegion: GetActiveScheduleRegion(),
                raceGoingByVenueLookup: GetRaceGoingByVenueSnapshot());
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

        // Renamed and modified to support URL collection for batching
        public async Task<List<string>> CollectRaceUrlsAsync(
            TimeSpan? raceWindow = null,
            DateTime? windowReferenceUtc = null,
            TimeSpan? scheduleStartTime = null,
            TimeSpan? scheduleEndTime = null,
            string? scheduleRegion = null)
        {
            if (raceWindow.HasValue && (scheduleStartTime.HasValue || scheduleEndTime.HasValue))
            {
                throw new ArgumentException(
                    "Race window and schedule time filters cannot be used at the same time.",
                    nameof(raceWindow));
            }

            ReturnToPrimaryWindow();
            EnsureSchedulePageReady();
            UpdateActiveScheduleRegion(CaptureActiveScheduleRegion());
            var scheduleRegionChanged = await TrySelectScheduleRegionAsync(scheduleRegion);
            if (scheduleRegionChanged)
            {
                await Task.Delay(200);
            }
            UpdateActiveScheduleRegion(CaptureActiveScheduleRegion());
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

            if (raceWindow.HasValue)
            {
                return (windowCandidates != null && windowCandidates.Count > 0)
                    ? windowCandidates
                        .OrderBy(kvp => kvp.Value)
                        .Select(kvp => kvp.Key)
                        .ToList()
                    : new List<string>();
            }
            else
            {
                return seen!.ToList();
            }
        }

        public async Task OpenBatchAsync(IEnumerable<string> urls, int delayBetweenTabsMs = 0)
        {
            var existingMarketIds = CaptureOpenRaceMarketIds();
            var newlyOpenedMarketIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var url in urls)
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

        // Kept for backward compatibility if needed, but redirects to new flow
        public async Task OpenHorseRaceMeetingsInNewTabsAsync(
            int delayBetweenTabsMs = 0,
            bool closeExistingRaceTabs = true,
            TimeSpan? raceWindow = null,
            DateTime? windowReferenceUtc = null,
            TimeSpan? scheduleStartTime = null,
            TimeSpan? scheduleEndTime = null,
            string? scheduleRegion = null)
        {
            if (closeExistingRaceTabs)
            {
                CloseAdditionalRaceTabs();
            }

            var urls = await CollectRaceUrlsAsync(raceWindow, windowReferenceUtc, scheduleStartTime, scheduleEndTime, scheduleRegion);
            await OpenBatchAsync(urls, delayBetweenTabsMs);
        }

        // ... (rest of methods) ...
        private async Task<bool> TrySelectScheduleRegionAsync(string? region)
        {
            if (string.IsNullOrWhiteSpace(region))
            {
                UpdateActiveScheduleRegion(CaptureActiveScheduleRegion());
                return false;
            }

            var normalizedTarget = NormalizeScheduleRegionText(region);
            if (normalizedTarget.Length == 0 || normalizedTarget.Equals("All", StringComparison.OrdinalIgnoreCase))
            {
                if (normalizedTarget.Equals("All", StringComparison.OrdinalIgnoreCase))
                {
                    UpdateActiveScheduleRegion(normalizedTarget);
                }
                else
                {
                    UpdateActiveScheduleRegion(CaptureActiveScheduleRegion());
                }
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
                        UpdateActiveScheduleRegion(normalizedTarget);
                        return true;
                    }
                    catch (StaleElementReferenceException)
                    {
                        break;
                    }
                }

                await Task.Delay(200);
            }
            UpdateActiveScheduleRegion(CaptureActiveScheduleRegion());
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
        private void UpdateActiveScheduleRegion(string? value)
        {
            var normalized = NormalizeScheduleRegionText(value);
            if (normalized.Length == 0 || normalized.Equals("All", StringComparison.OrdinalIgnoreCase))
            {
                _activeScheduleRegion = null;
                return;
            }

            _activeScheduleRegion = normalized;
        }

        private string? CaptureActiveScheduleRegion()
        {
            try
            {
                var js = (IJavaScriptExecutor)_driver;
                const string script = @"
const active = document.querySelector('li.country-tab.active, li.country-tab.selected');
if (!active) {
    return '';
}
const target = active.querySelector('div, span, button, a');
if (!target) {
    return '';
}
const text = target.textContent || target.innerText || '';
return text.trim();";
                var result = js.ExecuteScript(script);
                if (result is string text)
                {
                    var normalized = NormalizeScheduleRegionText(text);
                    return normalized.Length == 0 ? null : normalized;
                }
            }
            catch (WebDriverException)
            {
                // Ignore failures when attempting to detect the active schedule region.
            }
            catch (InvalidOperationException)
            {
                // Ignore if the driver is not in a valid state to execute script.
            }

            return null;
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
        private IReadOnlyDictionary<string, string?> GetRaceGoingByVenueSnapshot() { lock (_raceGoingLock) { return new Dictionary<string, string?>(_raceGoingByVenue, StringComparer.OrdinalIgnoreCase); } }
        private void RecordRaceGoing(string? href, string? going, string? venue = null)
        {
            var trimmedGoing = string.IsNullOrWhiteSpace(going) ? null : going!.Trim();

            lock (_raceGoingLock)
            {
                if (!string.IsNullOrWhiteSpace(href))
                {
                    var marketId = BetfairMarketScraper.ExtractMarketId(href);
                    if (!string.IsNullOrWhiteSpace(marketId))
                    {
                        if (!string.IsNullOrEmpty(trimmedGoing))
                        {
                            _raceGoingByMarketId[marketId] = trimmedGoing;
                        }
                        else if (!_raceGoingByMarketId.ContainsKey(marketId))
                        {
                            _raceGoingByMarketId[marketId] = null;
                        }
                    }
                }

                if (!string.IsNullOrWhiteSpace(venue))
                {
                    var normalizedVenue = BetfairMarketScraper.NormalizeVenueName(venue);
                    if (!string.IsNullOrEmpty(normalizedVenue))
                    {
                        if (!string.IsNullOrEmpty(trimmedGoing))
                        {
                            _raceGoingByVenue[normalizedVenue] = trimmedGoing;
                        }
                        else if (!_raceGoingByVenue.ContainsKey(normalizedVenue))
                        {
                            _raceGoingByVenue[normalizedVenue] = null;
                        }
                    }
                }
            }
        }

        private void CaptureRaceGoingFromSchedule()
        {
            lock (_raceGoingLock)
            {
                _raceGoingByMarketId.Clear();
                _raceGoingByVenue.Clear();
            }

            try
            {
                const string listXPath = "/html/body/ui-view/div/div/div[2]/div/ui-view/ui-view/div/div/div/div/div[1]/div[2]/div/bf-todays-racing-mod/div/div/bf-todays-racing/section/div[2]/div/div[2]/div/li";
                var raceItems = _driver.FindElements(By.XPath(listXPath));

                foreach (var raceItem in raceItems)
                {
                    string? location = null;
                    string? going = null;
                    string? locationRaw = null;
                    string? goingRaw = null;

                    try
                    {
                        var locationElement = raceItem.FindElement(By.XPath("./div/div[1]/div[1]"));
                        locationRaw = locationElement.Text;
                        location = locationRaw?.Trim();
                    }
                    catch (NoSuchElementException)
                    {
                        location = null;
                    }

                    try
                    {
                        var goingElement = raceItem.FindElement(By.XPath("./div/div[1]/div[2]"));
                        goingRaw = goingElement.Text;
                        going = CleanGoingText(goingRaw);
                    }
                    catch (NoSuchElementException)
                    {
                        Console.WriteLine("\t[DayReport][GoingXPath] Going XPath './div/div[1]/div[2]' not found.");
                        going = null;
                    }

                    var cleanedGoingDisplay = string.IsNullOrWhiteSpace(going) ? "<null>" : going;
                    Console.WriteLine(
                        $"\t[DayReport][GoingXPath] Location '{FormatGoingLogValue(locationRaw)}' => Going raw '{FormatGoingLogValue(goingRaw)}', cleaned '{cleanedGoingDisplay}'.");

                    if (string.IsNullOrWhiteSpace(location) && string.IsNullOrWhiteSpace(going))
                    {
                        Console.WriteLine("\t[DayReport][GoingXPath] Race item missing both location and going; skipping.");
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(location) || !string.IsNullOrWhiteSpace(going))
                    {
                        RecordRaceGoing(href: null, going, location);
                    }

                    var anchors = raceItem.FindElements(By.XPath(".//a[contains(@href, '/horse-racing/')]")).ToList();
                    if (anchors.Count == 0)
                    {

                        continue;
                    }

                    foreach (var anchor in anchors)
                    {
                        var href = anchor.GetAttribute("href");
                        if (string.IsNullOrWhiteSpace(href))
                        {
                            continue;
                        }

                        RecordRaceGoing(href, going, location);
                    }

                    var goingDisplay = string.IsNullOrWhiteSpace(going) ? "<unknown>" : going;
                    var locationDisplay = string.IsNullOrWhiteSpace(location) ? "<unknown location>" : location;
                    Console.WriteLine($"	[DayReport] Captured going '{goingDisplay}' for '{locationDisplay}'.");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Navigation] Failed to capture going information from schedule: {ex.Message}");
            }

            static string? CleanGoingText(string? raw)
            {
                if (string.IsNullOrWhiteSpace(raw))
                {
                    return null;
                }

                var cleaned = raw.Trim();
                if (cleaned.Length == 0)
                {
                    return null;
                }

                if (cleaned.StartsWith("Going", StringComparison.OrdinalIgnoreCase))
                {
                    var separatorIndex = cleaned.IndexOf(':');
                    if (separatorIndex >= 0 && separatorIndex + 1 < cleaned.Length)
                    {
                        cleaned = cleaned.Substring(separatorIndex + 1);
                    }
                    else
                    {
                        cleaned = cleaned.Substring(5);
                    }
                }

                cleaned = cleaned.Replace("in places", string.Empty, StringComparison.OrdinalIgnoreCase);
                cleaned = cleaned.Replace("Going", string.Empty, StringComparison.OrdinalIgnoreCase);
                cleaned = cleaned.Replace("-", " ");
                cleaned = cleaned.Replace("|", " ");
                cleaned = cleaned.Trim();

                return cleaned.Length == 0 ? null : cleaned;
            }
            static string FormatGoingLogValue(string? raw)
            {
                if (raw == null)
                {
                    return "<null>";
                }

                var sanitized = raw.Replace("\r", "\\r").Replace("\n", "\\n");

                if (sanitized.Length == 0)
                {
                    return "<empty>";
                }

                if (string.IsNullOrWhiteSpace(raw))
                {
                    return "<whitespace>";
                }

                return sanitized;
            }
        }

        private bool HasCapturedRaceGoing()
        {
            lock (_raceGoingLock)
            {
                return _raceGoingByMarketId.Count > 0 || _raceGoingByVenue.Count > 0;
            }
        }


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
        public BetfairScrapeResult GenerateDayReport(
           RacingRepository repo,
           HyperparameterTrainer trainer,
           bool showFeatureSignificance)
        {
            var bankroll = GetEffectiveBankroll(refreshFromPage: true);
            var settings = _automationSettings.GetSnapshot();
            var scraper = new BetfairMarketScraper(
                repo,
                trainer,
                _aiOddsCalculator,
                bankroll,
                settings,
                useMarketFallbackForAiDegeneracy: _useMarketFallbackForAiDegeneracy,
                raceGoingLookup: GetRaceGoingSnapshot(),
                computeAiProbabilities: true,
                scheduleRegion: GetActiveScheduleRegion(),
                raceGoingByVenueLookup: GetRaceGoingByVenueSnapshot(),
                computeFeatureContributions: showFeatureSignificance);

            var result = new BetfairScrapeResult();
            var races = scraper.ScrapeOpenRaceTabsForReport(_driver);
            if (races != null)
            {
                result.Races.AddRange(races);
            }

            return result;
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
