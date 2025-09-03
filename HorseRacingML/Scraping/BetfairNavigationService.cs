using System;
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
        _driver.Navigate().GoToUrl("https://www.betfair.com/exchange/plus/");
        var wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(30));

        // Accept cookies / terms if prompted
        try
        {
            var cookieButton = wait.Until(ExpectedConditions.ElementToBeClickable(By.Id("onetrust-accept-btn-handler")));
            try
            {
                cookieButton.Click();
            }
            catch (ElementClickInterceptedException)
            {
                ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", cookieButton);
            }
        }
        catch (WebDriverTimeoutException)
        {
            // cookie dialog not present
        }

        // Open login form
        try
        {
            var loginButton = wait.Until(d => d.FindElement(By.CssSelector("[data-testid='loginButton']")));
            loginButton.Click();
        }
        catch (WebDriverTimeoutException)
        {
            // already on login page
        }

        try
        {
            var loginFrame = wait.Until(d => d.FindElement(By.CssSelector("iframe[src*='identitysso']")));
            _driver.SwitchTo().Frame(loginFrame);
        }
        catch (WebDriverTimeoutException)
        {
            // Login form isn't hosted in an iframe.
        }

        // Fill in credentials once the fields are visible and interactable.
        // Multiple selectors are used so we don't wait the full timeout for a
        // single missing element before trying an alternative selector.
        IWebElement userBox = wait.Until(d =>
        {
            var boxes = d.FindElements(By.CssSelector("#ssc-liu, #username"));
            return boxes.Count > 0 ? boxes[0] : null;
        });
        userBox.Clear();
        userBox.SendKeys(_username);

        IWebElement passBox = wait.Until(d =>
        {
            var boxes = d.FindElements(By.CssSelector("#ssc-lipw, #password"));
            return boxes.Count > 0 ? boxes[0] : null;
        });
        passBox.Clear();
        passBox.SendKeys(_password);

        // Ensure fields have been populated before submitting the form
        wait.Until(_ => !string.IsNullOrEmpty(userBox.GetAttribute("value")) &&
                       !string.IsNullOrEmpty(passBox.GetAttribute("value")));

        var submit = wait.Until(d =>
        {
            var buttons = d.FindElements(By.CssSelector(
                "button[data-testid='login-submit'], button[type='submit'], button[data-testid='login-form-submit-button']"));
            return buttons.Count > 0 ? buttons[0] : null;
        });
        submit.Click();

        // wait for navigation after login and return to default content
        await Task.Delay(TimeSpan.FromSeconds(1));
        _driver.SwitchTo().DefaultContent();
        wait.Until(d => !d.Url.Contains("login", StringComparison.OrdinalIgnoreCase));
    }

    public void Dispose()
    {
        _driver.Quit();
        _driver.Dispose();
    }
}