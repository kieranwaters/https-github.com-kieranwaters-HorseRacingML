using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Interactions;
using OpenQA.Selenium.Support.UI;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Scrapes race result pages from sportinglife.com by iterating dates and
    /// clicking through available meeting tabs and race times.
    /// </summary>
    public class RaceResultsScraper
    {
        /// <summary>
        /// Starts scraping from <paramref name="startDate"/> until <paramref name="endDate"/>
        /// inclusive. A visible Chrome browser is used so the user can watch the
        /// scraping process.
        /// </summary>
        public void Scrape(DateTime startDate, DateTime endDate)
        {
            var options = new ChromeOptions(); options.AddArgument("--start-maximized"); //visible browser
            using var driver = new ChromeDriver(options);
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10)); //ample time for SPA updates
            for (var date = startDate; date <= endDate; date = date.AddDays(1))
            {
                try
                {
                    var url = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}"; driver.Navigate().GoToUrl(url); //nav to date
                    AcceptTermsIfPresent(driver); //cookie banner if shown
                    string[] regions = { "UK & Ireland", "International" }; //process both regions independently
                    foreach (var region in regions)
                    {
                        try
                        {
                            var regionBtn = driver.FindElements(By.XPath($"//*[self::button or self::span][contains(normalize-space(.), '{region}') and ancestor::*[@data-test-id='new-switch-button']]")).FirstOrDefault(e => e.Displayed && e.Enabled); //find visible region control
                            if (regionBtn == null) { continue; } //region not present on this date
                            ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", regionBtn); //robust click through React overlay
                            wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0); //wait for meeting tabs
                            if (!ScrapeMeetingTabs(driver, wait)) { Console.WriteLine($"No meeting tabs under region '{region}' on {date:yyyy-MM-dd}"); } //scrape meetings
                        }
                        catch (WebDriverException ex) { Console.WriteLine($"Region '{region}' error on {date:yyyy-MM-dd}: {ex.Message}"); } //log and move on
                    }
                    //fallback if no region switchers existed at all
                    if (driver.FindElements(By.CssSelector("[data-test-id='new-switch-button']")).Count == 0) { if (!ScrapeMeetingTabs(driver, wait)) { Console.WriteLine("No meeting tabs found on page."); } } //process page without regions
                }
                catch (WebDriverException ex) { Console.WriteLine($"Skipping {date:yyyy-MM-dd}: {ex.Message}"); } //continue to next day
            }
        }

        private static bool ScrapeMeetingTabs(IWebDriver driver, WebDriverWait wait)
        {
            IReadOnlyCollection<IWebElement> initialTabs;
            try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0); initialTabs = driver.FindElements(By.CssSelector("[data-test-id='generic-tab']")); } catch { Console.WriteLine("Meeting tab elements not found with expected selectors."); return false; } //ensure tabs exist
            int tabIndex = 0; //index over live queries
            while (true)
            {
                var tabs = driver.FindElements(By.CssSelector("[data-test-id='generic-tab']")); if (tabIndex >= tabs.Count) break; //no more tabs
                var tab = tabs[tabIndex];
                try { ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", tab); ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", tab); } catch { tabIndex++; continue; } //robust click
                try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id='race-container']")).Count > 0); } catch { tabIndex++; continue; } //wait for races or an empty container
                var raceLinks = driver.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]")); //anchors for each race
                foreach (var a in raceLinks)
                {
                    var href = a.GetAttribute("href"); if (string.IsNullOrWhiteSpace(href)) continue; //skip if no link
                    ((IJavaScriptExecutor)driver).ExecuteScript("window.open(arguments[0], '_blank');", href); //open race in new tab
                }
                tabIndex++; //next meeting tab
            }
            return true; //processed at least the tab loop
        }

        /// <summary>
        /// Attempts to accept terms and conditions if a pop-up is present.
        /// </summary>
        /// <param name="driver">The Selenium WebDriver instance.</param>
        private static bool AcceptTermsIfPresent(IWebDriver driver)
        {
            try
            {
                var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(2));
                var acceptButton = wait.Until(d =>
                {
                    var buttons = d.FindElements(By.XPath("//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow all cookies')]"));
                    return buttons.FirstOrDefault(b => b.Displayed && b.Enabled);
                });

                acceptButton?.Click();
                return acceptButton != null;
            }
            catch (WebDriverTimeoutException)
            {
                // Pop-up not present; nothing to accept.
                return false;
            }
        }
    }
}