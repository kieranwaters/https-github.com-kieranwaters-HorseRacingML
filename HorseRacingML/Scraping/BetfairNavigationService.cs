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
            var cookieButton = wait.Until(d => d.FindElement(By.Id("onetrust-accept-btn-handler")));
            cookieButton.Click();
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

        // Fill in credentials once the fields are visible and interactable
        IWebElement userBox;
        try
        {
            // Preferred selector for username field
            userBox = wait.Until(d => d.FindElement(By.CssSelector("#ssc-liu")));
        }
        catch (WebDriverTimeoutException)
        {
            // Fallback for when the preferred selector fails
            userBox = wait.Until(d => d.FindElement(By.Id("username")));
        }
        userBox.Clear();
        userBox.SendKeys(_username);

        IWebElement passBox;
        try
        {
            // Preferred selector for password field
            passBox = wait.Until(d => d.FindElement(By.CssSelector("#ssc-lipw")));
        }
        catch (WebDriverTimeoutException)
        {
            passBox = wait.Until(d => d.FindElement(By.Id("password")));
        }
        passBox.Clear();
        passBox.SendKeys(_password);

        var submit = wait.Until(d => d.FindElement(By.CssSelector("button[data-testid='login-submit']")));
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