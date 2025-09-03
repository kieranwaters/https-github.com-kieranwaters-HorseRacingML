using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;

namespace HorseRacingML.Scraping;

public class BetfairNavigationService : IDisposable
{
    private readonly string _username;
    private readonly string _password;
    private readonly IWebDriver _driver;

    public BetfairNavigationService(IConfiguration config)
    {
        _username = config["Betfair:Username"] ?? string.Empty;
        _password = config["Betfair:Password"] ?? string.Empty;

        var options = new ChromeOptions();
        options.AddArgument("--headless=new");
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

        // Fill in credentials
        var userBox = wait.Until(d => d.FindElement(By.Id("username")));
        userBox.SendKeys(_username);
        var passBox = _driver.FindElement(By.Id("password"));
        passBox.SendKeys(_password);

        var submit = _driver.FindElement(By.CssSelector("button[data-testid='login-submit']"));
        submit.Click();

        // wait for navigation after login
        await Task.Delay(TimeSpan.FromSeconds(1));
        wait.Until(d => !d.Url.Contains("login", StringComparison.OrdinalIgnoreCase));
    }

    public void Dispose()
    {
        _driver.Quit();
        _driver.Dispose();
    }
}