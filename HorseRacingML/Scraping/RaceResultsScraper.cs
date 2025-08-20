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
            var options = new ChromeOptions();
            options.AddArgument("--start-maximized");
            // No headless argument -> Chrome is visible

            using var driver = new ChromeDriver(options);
            // give pages plenty of time to load dynamic content
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10));

            for (var date = startDate; date <= endDate; date = date.AddDays(1))
            {

                var url = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}";
                driver.Navigate().GoToUrl(url);

                AcceptTermsIfPresent(driver);

                // Some pages expose region buttons (e.g. "UK & Ireland", "International")
                // that must be clicked before the meeting tabs become visible. Try to
                // iterate through those first, falling back to processing meetings
                // directly if none are found.
                var regionNames = new[] { "UK & Ireland", "International" };
                var regionFound = false;
                foreach (var region in regionNames)
                {
                    var regionButton = driver
                        .FindElements(By.XPath($"//button[contains(., '{region}')]"))
                        .FirstOrDefault(b => b.Displayed && b.Enabled);
                    if (regionButton != null)
                    {
                        regionFound = true;
                        regionButton.Click();
                        ScrapeMeetingTabs(driver, wait);
                    }
                }

                if (!regionFound)
                {
                    ScrapeMeetingTabs(driver, wait);
                }
            }
        }

        /// <summary>
        /// Clicks through all meeting tabs on the current page and opens each race
        /// link in a new browser tab.
        /// </summary>
        private static void ScrapeMeetingTabs(IWebDriver driver, WebDriverWait wait)
        {
            // iterate through available meeting tabs; fall back to new-switch-button if
            // generic tabs are not present. If neither is found, simply return so the
            // scraper can continue without failing.
            string? tabSelector = null;
            foreach (var selector in new[] { "[data-test-id='generic-tab']", "[data-test-id='new-switch-button']" })
            {
                try
                {
                    wait.Until(d => d.FindElements(By.CssSelector(selector)).Count > 0);
                    tabSelector = selector;
                    break;
                }
                catch (WebDriverTimeoutException)
                {
                    // try the next selector
                }
            }

            if (tabSelector == null)
            {
                // no meeting tabs were found; nothing to do on this page
                return;
            }
            int tabIndex = 0;
            while (true)
            {
                var tabs = driver.FindElements(By.CssSelector(tabSelector));
                if (tabIndex >= tabs.Count)
                    break;

                tabs[tabIndex].Click();
                // wait for time-short elements to appear
                try
                {
                    wait.Until(d => d.FindElements(By.CssSelector(".time-short")).Count > 0);
                }
                catch (WebDriverTimeoutException)
                {
                    // if no races are available under this tab, skip to next tab
                    tabIndex++;
                    continue;
                }

                var races = driver.FindElements(By.CssSelector(".time-short a"));
                foreach (var race in races)
                {
                    var href = race.GetAttribute("href");
                    if (!string.IsNullOrEmpty(href))
                    {
                        // open in a new tab
                        ((IJavaScriptExecutor)driver).ExecuteScript("window.open(arguments[0], '_blank');", href);
                    }
                }

                tabIndex++;
            }
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