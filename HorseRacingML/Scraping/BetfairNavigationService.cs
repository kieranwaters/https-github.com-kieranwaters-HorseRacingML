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
        private readonly bool _useMarketFallbackForAiDegeneracy;
        private readonly AutomationSettingsService _automationSettings;
        private readonly string _primaryWindowHandle;
        private static readonly Regex NonNumericCharactersRegex = new("[^0-9.,-]", RegexOptions.Compiled);
        private readonly object _automationLock = new();
        private CancellationTokenSource? _automationCancellation;
        private Task? _automationTask;

        public BetfairNavigationService(IConfiguration config, AutomationSettingsService automationSettings)
        {
            _username = config["Betfair:Username"] ?? HardCodedUsername;
            _password = config["Betfair:Password"] ?? HardCodedPassword;
            _configuredBankroll = config.GetValue<decimal?>("Betting:Bankroll") ?? 100m;
            _bankroll = _configuredBankroll;
            _useMarketFallbackForAiDegeneracy = config.GetValue<bool?>("Betting:UseMarketFallbackForAiDegeneracy") ?? true;
            _automationSettings = automationSettings ?? throw new ArgumentNullException(nameof(automationSettings));
            var options = new ChromeOptions();
            options.AddArguments(
                "--disable-extensions",
                "--blink-settings=imagesEnabled=false",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage");

            _driver = new ChromeDriver(options);
            _primaryWindowHandle = _driver.CurrentWindowHandle;
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
                var settings = _automationSettings.GetSnapshot();
                var scraper = new BetfairMarketScraper(
                    repo,
                    trainer,
                    bankroll,
                    settings,
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
            var settings = _automationSettings.GetSnapshot();
            var scraper = new BetfairMarketScraper(
                repo,
                trainer,
                bankroll,
                settings,
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

            wait.Until(d =>
                d.Url.Contains("horse-racing", StringComparison.OrdinalIgnoreCase) ||
                d.Url.Contains("horse-racing-betting-7", StringComparison.OrdinalIgnoreCase));
        }
        public DayReportViewModel GenerateDayReport(RacingRepository repo, HyperparameterTrainer trainer)
        {
            repo.ClearDayReportTables();
            var bankroll = GetEffectiveBankroll();
            var settings = _automationSettings.GetSnapshot();
            var scraper = new BetfairMarketScraper(
                repo,
                trainer,
                bankroll,
                settings,
                _useMarketFallbackForAiDegeneracy);
            var races = scraper.ScrapeOpenRaceTabsForReport(_driver);
            var orderedRaces = races
                .OrderBy(r => GetRaceScheduleSortKey(r))
                .ThenBy(r => r.RaceTitle ?? string.Empty, StringComparer.OrdinalIgnoreCase)
                .ThenBy(r => r.MarketId, StringComparer.Ordinal)
                .ToList();
            ReturnToPrimaryWindow();
            return new DayReportViewModel
            {
                GeneratedAt = DateTime.UtcNow,
                Bankroll = bankroll,
                AiHyperparameters = scraper.LoadedHyperparameters,
                Races = orderedRaces
            };
        }
        public async Task OpenHorseRaceMeetingsInNewTabsAsync(
            int delayBetweenTabsMs = 0,
            bool closeExistingRaceTabs = true,
            TimeSpan? raceWindow = null,
            DateTime? windowReferenceUtc = null)
        {
            if (closeExistingRaceTabs)
            {
                CloseAdditionalRaceTabs();
            }

            ReturnToPrimaryWindow();

            var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(20));
            wait.Until(d =>
                ((IJavaScriptExecutor)d).ExecuteScript("return document.readyState").ToString() == "complete");
            wait.Until(d => d.FindElements(By.CssSelector("a,button")).Count > 0);

            static bool Is24HourTime(string s)
            {
                if (string.IsNullOrWhiteSpace(s))
                {
                    return false;
                }

                var trimmed = s.Trim();
                return DateTime.TryParseExact(
                    trimmed,
                    new[] { "H:mm", "HH:mm" },
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.None,
                    out _);
            }

            static DateTime? TryResolveRaceDateTime(string? text, DateTime reference)
            {
                if (string.IsNullOrWhiteSpace(text))
                {
                    return null;
                }

                var trimmed = text.Trim();
                if (!TimeSpan.TryParseExact(
                        trimmed,
                        new[] { @"h\:mm", @"hh\:mm" },
                        CultureInfo.InvariantCulture,
                        out var timeOfDay))
                {
                    if (!DateTime.TryParseExact(
                            trimmed,
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

                        if (raceWindow.HasValue)
                        {
                            var raceTime = TryResolveRaceDateTime(textValue, windowReferenceLocal);
                            if (!raceTime.HasValue)
                            {
                                continue;
                            }

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

            foreach (var url in urlsToOpen)
            {
                ((IJavaScriptExecutor)_driver).ExecuteScript("window.open(arguments[0],'_blank');", url);
                if (delayBetweenTabsMs > 0)
                {
                    await Task.Delay(delayBetweenTabsMs);
                }
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
            if (initialDelay > TimeSpan.Zero)
            {
                try
                {
                    await Task.Delay(initialDelay, cancellationToken);
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
                    await OpenHorseRaceMeetingsInNewTabsAsync(
                        closeExistingRaceTabs: true,
                        raceWindow: raceWindow,
                        windowReferenceUtc: cycleStartUtc);

                    ScrapeOpenRaceTabs(repo, trainer);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[Automation] Failed to process betting cycle: {ex.Message}");
                }

                var cycleEndUtc = DateTime.UtcNow;
                var targetInterval = raceWindow - refreshLeadTime;
                if (targetInterval <= TimeSpan.Zero)
                {
                    continue;
                }

                var elapsed = cycleEndUtc - cycleStartUtc;
                var delay = targetInterval - elapsed;
                if (delay < TimeSpan.Zero)
                {
                    delay = TimeSpan.Zero;
                }

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
            _driver.Quit();
            _driver.Dispose();
        }
    }
}
