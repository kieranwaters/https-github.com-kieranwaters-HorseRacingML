using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Browser-backed client for attheraces.com. The site uses a JavaScript
    /// client challenge, so a normal headless HTTP request receives a 403.
    /// A persistent Chrome profile is kept outside the repository; the first
    /// run may show a short challenge page before the result pages load.
    /// </summary>
    public sealed class AtTheRacesClient : IDisposable
    {
        private readonly string _baseUrl;
        private readonly string _profileDirectory;
        private readonly int _challengeTimeoutSeconds;
        private readonly int _maxRateLimitRetries;
        private readonly int _rateLimitBackoffSeconds;
        private readonly int _resultsPageRetries;
        private readonly bool _headless;
        private readonly bool _fetchFinishSpeed;
        private int _currentDelayMs;
        private IWebDriver? _driver;
        private ChromeDriverService? _service;
        private bool _sessionReady;
        private bool _resultsPageFetchedOnce;

        public AtTheRacesClient(IConfiguration configuration)
        {
            _baseUrl = (configuration.GetValue<string>("AtTheRaces:BaseUrl")
                ?? "https://www.attheraces.com").TrimEnd('/');
            _currentDelayMs = Math.Clamp(
                configuration.GetValue<int?>("AtTheRaces:RequestDelayMs") ?? 1000,
                0,
                10_000);
            _challengeTimeoutSeconds = Math.Clamp(
                configuration.GetValue<int?>("AtTheRaces:ChallengeTimeoutSeconds") ?? 120,
                10,
                180);
            _maxRateLimitRetries = Math.Clamp(
                configuration.GetValue<int?>("AtTheRaces:MaxRateLimitRetries") ?? 3,
                0,
                10);
            _rateLimitBackoffSeconds = Math.Clamp(
                configuration.GetValue<int?>("AtTheRaces:RateLimitBackoffSeconds") ?? 60,
                5,
                900);
            _resultsPageRetries = Math.Clamp(
                configuration.GetValue<int?>("AtTheRaces:ResultsPageRetries") ?? 1,
                0,
                5);
            _headless = configuration.GetValue<bool?>("AtTheRaces:Headless") ?? false;
            _fetchFinishSpeed = configuration.GetValue<bool?>("AtTheRaces:FetchFinishSpeed") ?? true;
            string configuredProfile = configuration.GetValue<string>("AtTheRaces:ProfileDirectory") ?? string.Empty;
            _profileDirectory = string.IsNullOrWhiteSpace(configuredProfile)
                ? Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "HorseRacingML",
                    "atr-chrome-profile")
                : configuredProfile;
        }

        public string LoadResultsPage(DateTime date)
        {
            string url = $"{_baseUrl}/results/{date:yyyy-MM-dd}";
            if (!_resultsPageFetchedOnce)
            {
                // The first page of a process must be a real navigation so the
                // JavaScript challenge can run. Every later date uses fetch().
                var first = LoadResultsPageWithNavigation(url);
                _resultsPageFetchedOnce = true;
                return first;
            }

            for (int attempt = 0; attempt <= _maxRateLimitRetries; attempt++)
            {
                try
                {
                    string html = FetchViaJavaScript(url);
                    if (_currentDelayMs > 0)
                    {
                        Thread.Sleep(_currentDelayMs);
                    }

                    if (LooksLikeResultsPage(html))
                    {
                        return html;
                    }

                    // A challenge/blank response is cheaper to recover from
                    // with one real navigation than by retrying the fetch.
                    return LoadResultsPageWithNavigation(url);
                }
                catch (AtTheRacesRateLimitException ex)
                {
                    var wait = ex.RetryAfter > TimeSpan.Zero ? ex.RetryAfter : Backoff(attempt + 1);
                    if (wait > TimeSpan.FromMinutes(5))
                    {
                        wait = TimeSpan.FromMinutes(5);
                    }
                    _currentDelayMs = Math.Min(3000, Math.Max(1000, _currentDelayMs * 2));
                    Console.WriteLine(
                        $"[atr] results page rate limited (429); waiting {wait.TotalSeconds:0}s " +
                        $"and raising the request delay to {_currentDelayMs}ms.");
                    Thread.Sleep(wait);
                }
                catch (InvalidOperationException ex) when (
                    attempt < _maxRateLimitRetries &&
                    (ex.Message.Contains("HTTP_ERROR:403", StringComparison.Ordinal) ||
                     ex.Message.Contains("403", StringComparison.Ordinal)))
                {
                    return LoadResultsPageWithNavigation(url);
                }
            }

            return LoadResultsPageWithNavigation(url);
        }

        private string LoadResultsPageWithNavigation(string url)
        {
            for (int attempt = 0; attempt <= _resultsPageRetries; attempt++)
            {
                try
                {
                    string html = Navigate(url, waitForResultsPage: true);
                    _sessionReady = true;
                    return html;
                }
                catch (Exception ex) when (
                    attempt < _resultsPageRetries &&
                    ex is InvalidOperationException or WebDriverException)
                {
                    var wait = Backoff(attempt + 1);
                    Console.WriteLine(
                        $"[atr] results page not ready ({ex.Message.Trim()}); " +
                        $"waiting {wait.TotalSeconds:0}s before retry {attempt + 2}/{_resultsPageRetries + 1}.");
                    Thread.Sleep(wait);
                }
            }

            throw new InvalidOperationException(
                "At The Races results page could not be loaded after repeated retries. " +
                "The site may be rate-limiting this IP; wait a few minutes and resume the backfill.");
        }

        /// <summary>
        /// When false, the tools tab is not requested and the finishing-speed
        /// percentage is derived from the split times instead. This removes one
        /// request for every race that has timed sectionals.
        /// </summary>
        public bool FetchFinishSpeed => _fetchFinishSpeed;

        public string LoadSectionalTimes(AtTheRacesRaceLink race)
        {
            EnsureSession();
            return FetchSectionalFragment(BuildSectionalUrl(race, "times"), race);
        }

        public string LoadSectionalTools(AtTheRacesRaceLink race)
        {
            EnsureSession();
            return FetchSectionalFragment(BuildSectionalUrl(race, "tools"), race);
        }

        /// <summary>
        /// ATR's AJAX endpoints can be read with a same-origin fetch once the
        /// challenge cookie is present. This avoids a full browser navigation
        /// (and its page-load wait) for every race.
        /// </summary>
        private string FetchSectionalFragment(string url, AtTheRacesRaceLink race)
        {
            int attempts = _maxRateLimitRetries + 2;
            for (int attempt = 0; attempt < attempts; attempt++)
            {
                string html;
                try
                {
                    html = FetchViaJavaScript(url);
                    if (_currentDelayMs > 0)
                    {
                        Thread.Sleep(_currentDelayMs);
                    }
                }
                catch (AtTheRacesRateLimitException ex)
                {
                    if (attempt >= _maxRateLimitRetries)
                    {
                        throw new InvalidOperationException(
                            "At The Races rate-limited the sectional backfill. " +
                            "Wait a few minutes, then resume the same command.",
                            ex);
                    }

                    var wait = ex.RetryAfter > TimeSpan.Zero
                        ? ex.RetryAfter
                        : Backoff(attempt + 1);
                    if (wait > TimeSpan.FromMinutes(5))
                    {
                        wait = TimeSpan.FromMinutes(5);
                    }
                    _currentDelayMs = Math.Min(3000, Math.Max(1000, _currentDelayMs * 2));
                    Console.WriteLine(
                        $"[atr] rate limited (429); waiting {wait.TotalSeconds:0}s and " +
                        $"raising the request delay to {_currentDelayMs}ms.");
                    Thread.Sleep(wait);
                    continue;
                }
                catch (InvalidOperationException ex) when (
                    attempt < attempts - 1 &&
                    ex.Message.Contains("HTTP_ERROR:403", StringComparison.Ordinal))
                {
                    // The challenge cookie is stale. Reloading the racecard
                    // re-establishes it, then retry.
                    RefreshSession(race);
                    continue;
                }

                if (!LooksLikeChallenge(html))
                {
                    return html;
                }

                RefreshSession(race);
            }

            throw new InvalidOperationException(
                "At The Races returned a client challenge for a sectional request. " +
                "The persistent Chrome profile may need to be refreshed.");
        }

        private void RefreshSession(AtTheRacesRaceLink race)
        {
            Navigate($"{_baseUrl}{race.PagePath}", waitForResultsPage: false);
            _sessionReady = true;
        }

        private string FetchViaJavaScript(string url)
        {
            EnsureDriver();
            var driver = _driver!;
            object? result;
            try
            {
                result = ((IJavaScriptExecutor)driver).ExecuteAsyncScript(
                    @"
var url = arguments[0];
var done = arguments[arguments.length - 1];
fetch(url, {
    credentials: 'same-origin',
    cache: 'no-store',
    headers: { 'X-Requested-With': 'XMLHttpRequest' }
}).then(function (response) {
    if (!response.ok) {
        var retryAfter = response.headers.get('Retry-After') || '';
        done('HTTP_ERROR:' + response.status + ':' + retryAfter);
        return;
    }
    return response.text().then(function (text) { done(text); });
}).catch(function (error) {
    done('FETCH_ERROR:' + error);
});
",
                    url);
            }
            catch (WebDriverTimeoutException ex)
            {
                throw new InvalidOperationException(
                    "At The Races did not answer the sectional request in time.",
                    ex);
            }

            string html = result as string ?? string.Empty;
            if (html.StartsWith("HTTP_ERROR:", StringComparison.Ordinal) ||
                html.StartsWith("FETCH_ERROR:", StringComparison.Ordinal))
            {
                var parts = html.Split(':');
                if (parts.Length >= 2 &&
                    parts[1].Trim().Equals("429", StringComparison.Ordinal))
                {
                    if (parts.Length >= 3 &&
                        int.TryParse(parts[2].Trim(), out int retryAfterSeconds) &&
                        retryAfterSeconds > 0)
                    {
                        throw new AtTheRacesRateLimitException(
                            TimeSpan.FromSeconds(retryAfterSeconds));
                    }

                    throw new AtTheRacesRateLimitException(TimeSpan.Zero);
                }

                throw new InvalidOperationException(
                    $"At The Races sectional request failed: {html}");
            }

            return html;
        }

        private void EnsureSession()
        {
            if (_sessionReady)
            {
                return;
            }

            Navigate($"{_baseUrl}/", waitForResultsPage: false);
            _sessionReady = true;
        }

        private static bool LooksLikeChallenge(string html)
            => (html ?? string.Empty).Contains(
                "Client Challenge",
                StringComparison.OrdinalIgnoreCase);

        private static bool LooksLikeResultsPage(string html)
            => !LooksLikeChallenge(html) &&
               (html?.Length ?? 0) > 50_000 &&
               html!.Contains("/racecard/", StringComparison.OrdinalIgnoreCase);

        private string BuildSectionalUrl(AtTheRacesRaceLink race, string page)
        {
            string pagePath = race.PagePath.StartsWith("/", StringComparison.Ordinal)
                ? race.PagePath
                : "/" + race.PagePath;
            return $"{_baseUrl}/ajax/sectionals-razor/{race.RaceId}/{page}" +
                   $"?page={pagePath}&raceid={race.RaceId}";
        }

        private string Navigate(string url, bool waitForResultsPage)
        {
            EnsureDriver();
            var driver = _driver!;
            driver.Navigate().GoToUrl(url);
            WaitForDocument(driver, waitForResultsPage);
            if (_currentDelayMs > 0)
            {
                Thread.Sleep(_currentDelayMs);
            }

            return driver.PageSource;
        }

        private void WaitForDocument(IWebDriver driver, bool waitForResultsPage)
        {
            var stopwatch = Stopwatch.StartNew();
            while (stopwatch.Elapsed < TimeSpan.FromSeconds(_challengeTimeoutSeconds))
            {
                try
                {
                    string title = driver.Title ?? string.Empty;
                    string source = driver.PageSource ?? string.Empty;
                    bool challenge = title.Contains("Client Challenge", StringComparison.OrdinalIgnoreCase);
                    if (!challenge &&
                        source.Length > (waitForResultsPage ? 50_000 : 2_000) &&
                        (!waitForResultsPage || source.Contains("/racecard/", StringComparison.OrdinalIgnoreCase)))
                    {
                        return;
                    }
                }
                catch (WebDriverException)
                {
                    // The page is still navigating. Retry until the timeout.
                }

                Thread.Sleep(500);
            }

            throw new InvalidOperationException(
                "At The Races did not finish loading its client challenge. " +
                (_headless
                    ? "ATR blocks headless Chrome; rerun without AtTheRaces:Headless."
                    : "Run the backfill from an interactive desktop session and try again."));
        }

        private TimeSpan Backoff(int attempt)
            => TimeSpan.FromSeconds(
                Math.Clamp(
                    _rateLimitBackoffSeconds * Math.Pow(2, Math.Max(0, attempt - 1)),
                    _rateLimitBackoffSeconds,
                    900));

        private void EnsureDriver()
        {
            if (_driver is not null)
            {
                return;
            }

            Directory.CreateDirectory(_profileDirectory);
            var options = new ChromeOptions();
            if (_headless)
            {
                options.AddArgument("--headless=new");
            }

            options.AddArgument("--disable-gpu");
            options.AddArgument("--disable-dev-shm-usage");
            options.AddArgument("--no-sandbox");
            options.AddArgument("--window-size=1400,1000");
            options.AddArgument("--disable-blink-features=AutomationControlled");
            options.AddArgument("--user-data-dir=" + _profileDirectory);
            options.AddArgument("--profile-directory=Default");
            options.AddArgument("--no-first-run");
            options.AddArgument("--no-default-browser-check");
            options.AddExcludedArgument("enable-automation");
            options.AddAdditionalOption("useAutomationExtension", false);

            _service = ChromeDriverService.CreateDefaultService();
            _service.HideCommandPromptWindow = true;
            _service.Port = GetFreeTcpPort();

            try
            {
                _driver = new ChromeDriver(_service, options, TimeSpan.FromSeconds(90));
                _driver.Manage().Timeouts().PageLoad = TimeSpan.FromSeconds(90);
                _driver.Manage().Timeouts().AsynchronousJavaScript = TimeSpan.FromSeconds(60);
                _driver.Manage().Timeouts().ImplicitWait = TimeSpan.Zero;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    "Could not start Chrome for At The Races. Close any Chrome window using " +
                    $"the profile at '{_profileDirectory}' and try again. {ex.Message}",
                    ex);
            }
        }

        private static int GetFreeTcpPort()
        {
            var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            int port = ((IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();
            return port;
        }

        public void Dispose()
        {
            try
            {
                _driver?.Quit();
            }
            catch
            {
                // The browser may already be gone.
            }
            finally
            {
                _driver?.Dispose();
                _service?.Dispose();
                _driver = null;
                _service = null;
            }
        }
    }

    public sealed class AtTheRacesRateLimitException : Exception
    {
        public AtTheRacesRateLimitException(TimeSpan retryAfter)
            : base("At The Races returned HTTP 429 (rate limited).")
        {
            RetryAfter = retryAfter;
        }

        public TimeSpan RetryAfter { get; }
    }
}
