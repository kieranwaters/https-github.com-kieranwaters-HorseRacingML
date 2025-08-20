using System;
using System.Linq;
using System.Threading;
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
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10));

            for (var date = startDate; date <= endDate; date = date.AddDays(1))
            {
                var url = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}";
                driver.Navigate().GoToUrl(url);

                AcceptTermsIfPresent(driver);

                // iterate through all generic tabs
                wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0);
                int tabIndex = 0;
                while (true)
                {
                    var tabs = driver.FindElements(By.CssSelector("[data-test-id='generic-tab']"));
                    if (tabIndex >= tabs.Count)
                        break;

                    tabs[tabIndex].Click();
                    // wait for time-short elements to appear
                    wait.Until(d => d.FindElements(By.CssSelector(".time-short")).Count > 0);

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
        }

        /// <summary>
        /// Attempts to accept terms and conditions if a pop-up is present.
        /// </summary>
        /// <param name="driver">The Selenium WebDriver instance.</param>
        private static void AcceptTermsIfPresent(IWebDriver driver)
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
            }
            catch (WebDriverTimeoutException)
            {
                // Pop-up not present; nothing to accept.
            }
        }
    }
}
