using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using SeleniumExtras.WaitHelpers;
using HorseRacingML.Data;
using HorseRacingML.Models;
using System.Globalization;
using System.Text.RegularExpressions;

namespace HorseRacingML.Scraping;

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

    public BetfairNavigationService(IConfiguration config)
    {
        _username = config["Betfair:Username"] ?? HardCodedUsername;
        _password = config["Betfair:Password"] ?? HardCodedPassword;
        _configuredBankroll = config.GetValue<decimal?>("Betting:Bankroll") ?? 100m;
        _bankroll = _configuredBankroll;
        _maxKellyFraction = config.GetValue<decimal?>("Betting:MaxKellyFraction");

        var options = new ChromeOptions();
        options.AddArguments(
            "--disable-extensions",
            "--blink-settings=imagesEnabled=false",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage");
        _driver = new ChromeDriver(options);
    }

    public IWebDriver Driver => _driver;
    public IReadOnlyList<BetRecommendation> ScrapeOpenRaceTabs(RacingRepository repo)
    {
        var bankroll = GetEffectiveBankroll();
        var scraper = new BetfairMarketScraper(repo, bankroll, _maxKellyFraction);
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

        var sanitized = Regex.Replace(trimmed, "[^0-9.,-]", string.Empty);
        sanitized = sanitized.Replace(",", string.Empty);

        return decimal.TryParse(sanitized, NumberStyles.Number | NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value);
    }
    public async Task LoginAsync()
    {
        _driver.Navigate().GoToUrl("https://www.betfair.com/exchange/plus/"); var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(45)); // go + longer timeout
        try { var cookie = wait.Until(ExpectedConditions.ElementToBeClickable(By.Id("onetrust-accept-btn-handler"))); try { cookie.Click(); } catch (ElementClickInterceptedException) { ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", cookie); } } catch (WebDriverTimeoutException) { } // accept cookies if shown
        Func<bool> isLoggedIn = () => _driver.Url.Contains("loginStatus=SUCCESS", StringComparison.OrdinalIgnoreCase) || _driver.PageSource.Contains("My Account", StringComparison.OrdinalIgnoreCase); // quick login heuristic
        if (!isLoggedIn()) // only try to log in if we’re not already
        {
            IWebElement? iframe = null; try { iframe = wait.Until(d => { var f = d.FindElements(By.CssSelector("iframe[src*='identitysso'],iframe[id*='sso'],iframe[name*='sso']")); return f.Count > 0 ? f[0] : null; }); } catch (WebDriverTimeoutException) { } // detect login iframe
            if (iframe != null) { _driver.SwitchTo().Frame(iframe); } // switch if present
            IWebElement user = wait.Until(d => { var els = d.FindElements(By.CssSelector("#ssc-liu,#username,input[name='username']")); return els.Count > 0 ? els[0] : null; }); user.Clear(); user.SendKeys(_username); // username
            IWebElement pass = wait.Until(d => { var els = d.FindElements(By.CssSelector("#ssc-lipw,#password,input[type='password']")); return els.Count > 0 ? els[0] : null; }); pass.Clear(); pass.SendKeys(_password); // password
            wait.Until(_ => !string.IsNullOrEmpty(user.GetAttribute("value")) && !string.IsNullOrEmpty(pass.GetAttribute("value"))); // ensure filled
            IWebElement loginBtn = wait.Until(d => { var els = d.FindElements(By.CssSelector("input#ssc-lis,input[type='submit'][id='ssc-lis'],input[type='submit'][value='Log In']")); return els.Count > 0 ? els[0] : null; }); ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", loginBtn); // bring into view
            try { wait.Until(ExpectedConditions.ElementToBeClickable(loginBtn)); loginBtn.Click(); } catch (Exception) { try { ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", loginBtn); } catch (Exception) { try { loginBtn.Submit(); } catch (Exception) { pass.SendKeys(Keys.Enter); } } } // click with fallbacks
            await Task.Delay(1200); _driver.SwitchTo().DefaultContent(); // leave frame if any
            wait.Until(d => isLoggedIn() || !d.Url.Contains("login", StringComparison.OrdinalIgnoreCase) || !d.Url.Contains("identitysso", StringComparison.OrdinalIgnoreCase)); // wait until logged in
        }
        wait.Until(d => ((IJavaScriptExecutor)d).ExecuteScript("return document.readyState").ToString() == "complete"); // page ready
                                                                                                                        // ---- Click the left-menu "Horse Racing" exact item (Angular loads this lazily)
        IWebElement nav = wait.Until(d => d.FindElement(By.CssSelector("bf-navigation-lhm .navigation-container"))); // left nav root
        ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollTop=0;", nav); // reset nav scroll
        Func<IWebDriver, IWebElement?> findHorse = drv => { var bys = new[] { By.CssSelector("bf-navigation-lhm a.navigation-link[href*='horse-racing-betting-7']"), By.CssSelector("bf-navigation-lhm a.navigation-link[link-sport='7']"), By.XPath("//bf-navigation-lhm//a[normalize-space()='Horse Racing']") }; foreach (var by in bys) { var els = drv.FindElements(by); foreach (var e in els) { if (e.Displayed) return e; } } return null; }; // find strategies
        IWebElement? horse = null; for (int i = 0; i < 10 && horse == null; i++) { horse = findHorse(_driver); if (horse == null) { ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollTop=arguments[0].scrollTop+180;", nav); await Task.Delay(120); } } // incremental scroll to reveal
        if (horse == null) { horse = wait.Until(d => { var el = ((IJavaScriptExecutor)d).ExecuteScript("return document.querySelector(\"#main-wrapper > div > div.scrollable-panes-height-taker > div > div > div.bf-row.no-bottom-gutter.nested-scrollable-pane-parent.extra-links-space-3.hidden-when-left-side-collapsed.extra-links-space-2 > div > bf-navigation-lhm > div > div > ng-include > div > div > ul > li > tree-section > span > div > div > div > ul > li:nth-child(16) > a\")"); return el is IWebElement we ? we : null; }); } // your exact path as a last resort
    ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", horse); // bring into view
        try { wait.Until(ExpectedConditions.ElementToBeClickable(horse)); horse.Click(); } catch (Exception) { ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", horse); } // click with JS fallback
        wait.Until(d => d.Url.Contains("horse-racing", StringComparison.OrdinalIgnoreCase) || d.Url.Contains("horse-racing-betting-7", StringComparison.OrdinalIgnoreCase)); // ensure navigation happened
    }


    public async Task OpenHorseRaceMeetingsInNewTabsAsync(int delayBetweenTabsMs = 0)
    {
        var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(10));
        wait.Until(d => d.FindElements(By.CssSelector("a[href*='/horse-racing/']")).Count > 0);

        var links = _driver.FindElements(By.CssSelector("a[href*='/horse-racing/']"));
        var urls = new HashSet<string>();
        foreach (var link in links)
        {
            var href = link.GetAttribute("href");
            if (!string.IsNullOrEmpty(href))
            {
                urls.Add(href);
            }
        }

        foreach (var url in urls)
        {
            ((IJavaScriptExecutor)_driver).ExecuteScript("window.open(arguments[0], '_blank');", url);
            if (delayBetweenTabsMs > 0)
            {
                await Task.Delay(delayBetweenTabsMs);
            }
        }
        await Task.CompletedTask;
    }
    public void Dispose()
    {
        _driver.Quit();
        _driver.Dispose();
    }
}