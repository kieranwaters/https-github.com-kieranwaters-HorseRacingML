using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using SeleniumExtras.WaitHelpers;

namespace HorseRacingML.Scraping;

public class BetfairNavigationService : IDisposable
{
    private const string HardCodedUsername = "kierandpwaters@gmail.com";
    private const string HardCodedPassword = "AZQ2v.b=$e$!e!u";
    private readonly string _username;
    private readonly string _password;
    private readonly IWebDriver _driver;

    public BetfairNavigationService(IConfiguration config)
    {
        _username = config["Betfair:Username"] ?? HardCodedUsername;
        _password = config["Betfair:Password"] ?? HardCodedPassword;

        var options = new ChromeOptions();
        _driver = new ChromeDriver(options);
    }

    public IWebDriver Driver => _driver;

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


    public async Task OpenHorseRaceMeetingsInNewTabsAsync()
    {
        var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(10));
        wait.Until(d => d.FindElements(By.CssSelector("a[href*='/horse-racing/']")).Count > 0);

        var links = _driver.FindElements(By.CssSelector("a[href*='/horse-racing/']"));
        var seen = new HashSet<string>();
        foreach (var link in links)
        {
            var href = link.GetAttribute("href");
            if (!string.IsNullOrEmpty(href) && seen.Add(href))
            {
                ((IJavaScriptExecutor)_driver).ExecuteScript("window.open(arguments[0], '_blank');", href);
                await Task.Delay(100);
            }
        }
    }
    public void Dispose()
    {
        _driver.Quit();
        _driver.Dispose();
    }
}