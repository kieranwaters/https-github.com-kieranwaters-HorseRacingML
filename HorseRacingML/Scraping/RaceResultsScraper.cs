using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using System.Globalization;
using HorseRacingML.Data;
using HorseRacingML.Models;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using System.Collections.Generic;
using System.Text.Json;
using System.Net;
using System.Net.Sockets;
using HorseRacingML.Services;
using System.Linq;

namespace HorseRacingML.Scraping
{
    public class RaceResultsScraper
    {
        private readonly RacingRepository _repo;
        private readonly ScrapingStatusService _status;
        private const int MaxParallelDrivers = 8;
        public RaceResultsScraper(RacingRepository repo, ScrapingStatusService status)
        {
            _repo = repo;
            _status = status;
        }
        private static int GetFreeTcpPort()
        {
            using var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            int port = ((IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();
            return port;
        }
        public void ScrapeFromEarliest()
        {
            var minDate = new DateTime(2000, 1, 1);
            while (true)
            {
                var earliest = _repo.GetEarliestRaceDate();
                if (earliest <= minDate) break;
                var end = earliest.AddDays(-1);
                var start = end.AddDays(-(MaxParallelDrivers - 1));
                if (start < minDate) start = minDate;
                _status.Update($"Scraping {start:yyyy-MM-dd} to {end:yyyy-MM-dd}");
                Scrape(start, end);
                if (start == minDate) break;
            }
            _status.Update("Scraping finished.");
        }
        private void ScrapeMeetingTabs(IWebDriver driver, WebDriverWait wait, DateTime raceDate, string dayHandle, List<RunnerResult> sessionResults)
        {
            try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id*='no-meetings']")).Count > 0); } catch { Console.WriteLine("Meeting tab elements not found."); return; } // ensure tabs or no-meetings
            int tabIndex = 0; while (true)
            {
                var tabs = driver.FindElements(By.CssSelector("[data-test-id='generic-tab']")); if (tabIndex >= tabs.Count) break; // done
                var tab = tabs[tabIndex];
                try { ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", tab); ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", tab); } catch { tabIndex++; continue; } // click meeting tab
                try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id='race-container']")).Count > 0); } catch { tabIndex++; continue; } // wait races
                var raceLinks = driver.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]"));
                var meetingHandle = driver.CurrentWindowHandle; // remember meeting page handle
                var raceHandles = new List<string>();
                foreach (var a in raceLinks)
                {
                    var href = a.GetAttribute("href");
                    if (string.IsNullOrWhiteSpace(href)) continue; // skip bad links
                    var beforeOpen = driver.WindowHandles.ToList();
                    ((IJavaScriptExecutor)driver).ExecuteScript("window.open(arguments[0], '_blank');", href); // open race in new tab
                    var handle = driver.WindowHandles.Except(beforeOpen).FirstOrDefault();
                    if (!string.IsNullOrEmpty(handle)) raceHandles.Add(handle); else Console.WriteLine("Failed to detect new tab handle, skipping.");
                }
                foreach (var newHandle in raceHandles)
                {
                    try { driver.SwitchTo().Window(newHandle); }
                    catch (WebDriverException ex) { Console.WriteLine($"Switch to race tab error: {ex.Message}"); continue; } // switch to race
                    int attempts = 0;
                    while (true)
                    {
                        try
                        {
                            ParseRacePage(driver, wait, raceDate, sessionResults);
                            break;
                        }
                        catch (StaleElementReferenceException ex)
                        {
                            attempts++;
                            if (attempts >= 3)
                            {
                                Console.WriteLine($"Parse error: {ex.Message}");
                                break;
                            }
                            Thread.Sleep(200);
                            try { driver.Navigate().Refresh(); } catch { }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Parse error: {ex.Message}");
                            break;
                        }
                    }
                    try
                    {
                        var handlesNow = driver.WindowHandles;
                        if (handlesNow.Count > 1) { driver.Close(); } else { Console.WriteLine("Skip Close(): only one window left."); } // avoid closing last window
                    }
                    catch (WebDriverException ex) { Console.WriteLine($"Close tab error: {ex.Message}"); } // safe close
                    try
                    {
                        var handlesAfterClose = driver.WindowHandles; // remaining windows
                        string target = null;
                        if (handlesAfterClose.Contains(meetingHandle)) target = meetingHandle;
                        else if (handlesAfterClose.Contains(dayHandle)) target = dayHandle;
                        else if (handlesAfterClose.Count > 0) target = handlesAfterClose.First(); // choose best remaining
                        if (target == null) { Console.WriteLine("No remaining window to switch to, breaking out of races."); break; } // nothing to switch
                        driver.SwitchTo().Window(target); // back to meeting or day
                        if (target == dayHandle) { meetingHandle = dayHandle; } // reset meeting handle if we lost it
                    }
                    catch (WebDriverException ex) { Console.WriteLine($"Switch back error: {ex.Message}"); break; } // bail out cleanly for this meeting
                }
                tabIndex++; // next meeting
                try { if (!driver.WindowHandles.Contains(dayHandle)) { dayHandle = driver.WindowHandles.FirstOrDefault() ?? dayHandle; } } catch { } // keep dayHandle valid
            }
        }

        private static decimal? FractionToDecimal(string frac)
        {
            if (string.IsNullOrWhiteSpace(frac)) return null; frac = frac.Trim().ToLowerInvariant(); frac = frac.Replace("jf", "").Replace("cf", "").Replace("f", "").Trim(); var m = System.Text.RegularExpressions.Regex.Match(frac, @"(\d+)\s*/\s*(\d+)"); if (!m.Success) return null; var a = decimal.Parse(m.Groups[1].Value); var b = decimal.Parse(m.Groups[2].Value); return Math.Round(1 + (a / b), 3); // decimal incl stake
        }
        public void ScrapeFromLatest()
        {
            var today = DateTime.Today;
            while (true)
            {
                var latest = _repo.GetLatestRaceDate();
                if (latest > today) latest = today;
                var start = latest.AddDays(1);
                if (start > today) break;
                var end = start.AddDays(MaxParallelDrivers - 1);
                if (end > today) end = today;
                _status.Update($"Scraping {start:yyyy-MM-dd} to {end:yyyy-MM-dd}");
                Scrape(start, end);
            }
            _status.Update("Scraping finished.");
        }//
        public void Scrape(DateTime startDate, DateTime endDate)
        {
            var start = startDate.Date;
            var end = endDate.Date;
            var today = DateTime.Today;
            if (end > today) end = today;
            if (start > end) return;

            var dates = Enumerable.Range(0, (end - start).Days + 1)
                                   .Select(i => end.AddDays(-i));
            var queue = new ConcurrentQueue<DateTime>(dates);
            var tasks = new List<Task>();
            int workers = Math.Min(MaxParallelDrivers, queue.Count);

            for (int i = 0; i < workers; i++)
            {
                tasks.Add(Task.Run(() =>
                {
                    using var svc = ChromeDriverService.CreateDefaultService();
                    svc.HideCommandPromptWindow = true;
                    svc.Port = GetFreeTcpPort(); // let OS choose a free port to avoid collisions
                    using var driver = new ChromeDriver(svc, BuildChromeOptions(), TimeSpan.FromSeconds(60));
                    driver.Manage().Timeouts().PageLoad = TimeSpan.FromSeconds(45);
                    driver.Manage().Timeouts().AsynchronousJavaScript = TimeSpan.FromSeconds(15);
                    driver.Manage().Timeouts().ImplicitWait = TimeSpan.FromSeconds(0); // timeouts

                    while (queue.TryDequeue(out var date))
                    {
                        ScrapeDay(driver, date);
                        var msg = $"[{date:yyyy-MM-dd}] parsed and inserted";
                        Console.WriteLine(msg);
                        _status.Update(msg);
                        try { _ = driver.WindowHandles.Count; }
                        catch (Exception ex) { Console.Error.WriteLine($"[Warning] Driver session not healthy before next day: {ex.Message}"); break; }
                    }
                }));
            }

            Task.WaitAll(tasks.ToArray());
        }

        private static string ExtractFavouriteTag(string frac) { if (string.IsNullOrWhiteSpace(frac)) return null; frac = frac.ToUpperInvariant(); if (frac.Contains("JF")) return "JF"; if (frac.Contains("CF")) return "CF"; if (frac.EndsWith("F")) return "F"; return null; } // F/JF/CF

        private static decimal? ParseBeatenLengths(string s) { if (string.IsNullOrWhiteSpace(s)) return null; s = s.Trim().ToLowerInvariant(); if (s == "nk" || s == "neck") return 0.3m; if (s == "hd" || s == "head") return 0.2m; if (s == "shd" || s == "short head" || s == "shorthead" || s == "s.h" || s == "sh") return 0.1m; if (s == "nse" || s == "nose") return 0.05m; s = s.Replace("¾", ".75").Replace("½", ".5").Replace("¼", ".25"); if (decimal.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var v)) return v; return null; } // adds s.h/sh/nse/nose/neck/head


        private static string ExtractOddsToken(string s, string key) { if (string.IsNullOrWhiteSpace(s)) return null; var m = System.Text.RegularExpressions.Regex.Match(s, $@"\b{System.Text.RegularExpressions.Regex.Escape(key)}\s+(\d+/\d+\w*)", System.Text.RegularExpressions.RegexOptions.IgnoreCase); return m.Success ? m.Groups[1].Value : null; } // "op 33/1"

        private static (string? low, string? high) ExtractTouchedTokens(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return (null, null);
            var lowMatch = System.Text.RegularExpressions.Regex.Match(s, @"(tchd|low)\s+(\d+/\d+\w*)", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            var highMatch = System.Text.RegularExpressions.Regex.Match(s, @"high\s+(\d+/\d+\w*)", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            string? low = lowMatch.Success ? lowMatch.Groups[2].Value : null; string? high = highMatch.Success ? highMatch.Groups[1].Value : null; return (low, high); // low/high
        }

        private static int? TryParseInt(string? s) { return int.TryParse(s?.Trim(), out var v) ? v : null; } // int?
        private static byte? TryParseByte(string? s) { return byte.TryParse(s?.Trim(), out var v) ? v : null; } // byte?
        private static DateTime? ParseDateSafe(string? s) { if (string.IsNullOrWhiteSpace(s)) return null; if (DateTime.TryParse(s, out var d)) return d.Date; return null; } // date?
        private static string TextOrEmpty(IWebDriver d, By by)
        {
            for (int i = 0; i < 3; i++)
            {
                try { return d.FindElement(by).Text.Trim(); }
                catch (StaleElementReferenceException)
                {
                    Thread.Sleep(100);
                }
                catch { return ""; }
            }
            return "";
        } // driver text
        private static string SafeText(IWebElement e, By by)
        {
            for (int i = 0; i < 3; i++)
            {
                try { return e.FindElement(by).Text.Trim(); }
                catch (StaleElementReferenceException)
                {
                    Thread.Sleep(100);
                }
                catch { return ""; }
            }
            return "";
        } // element text
        private void ScrapeDay(IWebDriver driver, DateTime date)
        {
            try
            {
                var url = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}"; driver.Navigate().GoToUrl(url); // go to date page
                AcceptTermsIfPresent(driver); // cookies
                var dayHandle = driver.CurrentWindowHandle; // remember the top-level window for this day
                var dayResults = new List<RunnerResult>();
                string[] regions = { "UK & Ireland", "International" }; bool hasRegionToggle = driver.FindElements(By.CssSelector("[data-test-id='new-switch-button']")).Count > 0; // detect region toggle
                bool anyMeetingsProcessed = false; // track if we managed to scrape anything
                if (hasRegionToggle)
                {
                    foreach (var region in regions)
                    {
                        try
                        {
                            IWebElement? regionBtn = null;
                            for (int attempt = 0; attempt < 3 && regionBtn == null; attempt++)
                            {
                                try
                                {
                                    regionBtn = driver
                                        .FindElements(By.XPath($"//*[self::button or self::span][contains(normalize-space(.), '{region}') and ancestor::*[@data-test-id='new-switch-button']]"))
                                        .FirstOrDefault(e =>
                                        {
                                            try { return e.Displayed && e.Enabled; } catch { return false; }
                                        });
                                }
                                catch (StaleElementReferenceException)
                                {
                                    Thread.Sleep(100);
                                }
                            }
                            if (regionBtn == null) { Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Region '{region}' not present"); continue; } // skip missing region

                            bool clicked = false;
                            for (int attempt = 0; attempt < 3 && !clicked; attempt++)
                            {
                                try
                                {
                                    ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", regionBtn); // click region
                                    clicked = true;
                                }
                                catch (StaleElementReferenceException)
                                {
                                    regionBtn = null; // re-find button if it went stale
                                    try
                                    {
                                        regionBtn = driver
                                            .FindElements(By.XPath($"//*[self::button or self::span][contains(normalize-space(.), '{region}') and ancestor::*[@data-test-id='new-switch-button']]"))
                                            .FirstOrDefault(e =>
                                            {
                                                try { return e.Displayed && e.Enabled; } catch { return false; }
                                            });
                                    }
                                    catch (Exception) { }
                                }
                                catch (WebDriverException)
                                {
                                    Thread.Sleep(100);
                                }
                            }
                            if (!clicked) { Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Region '{region}' click failed"); continue; }

                            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(5));
                            wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id*='no-meetings']")).Count > 0); // wait meetings
                            ScrapeMeetingTabs(driver, new WebDriverWait(driver, TimeSpan.FromSeconds(6)), date, dayHandle, dayResults); anyMeetingsProcessed = true; // scrape meetings for this region
                        }
                        catch (WebDriverException ex) { Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Region '{region}' error: {ex.Message}"); }
                        catch (Exception ex) { Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Unexpected region error '{region}': {ex.Message}"); }
                    }
                    if (!anyMeetingsProcessed) { ScrapeMeetingTabs(driver, new WebDriverWait(driver, TimeSpan.FromSeconds(12)), date, dayHandle, dayResults); } // fallback if toggle failed
                }
                else { ScrapeMeetingTabs(driver, new WebDriverWait(driver, TimeSpan.FromSeconds(12)), date, dayHandle, dayResults); } // no toggle present, scrape directly
                if (dayResults.Count > 0) _repo.BulkInsertRunnerResults(dayResults);
            }
            catch (WebDriverException ex) { Console.Error.WriteLine($"[Skip Day] {date:yyyy-MM-dd} WebDriverException: {ex.Message}"); }
            catch (OperationCanceledException ex) { Console.Error.WriteLine($"[Skip Day] {date:yyyy-MM-dd} OperationCanceled: {ex.Message}"); }
            catch (Exception ex) { Console.Error.WriteLine($"[Skip Day] {date:yyyy-MM-dd} Unexpected: {ex.Message}"); }
        }
        private ChromeOptions BuildChromeOptions()
        {
            var options = new ChromeOptions();
            options.AddArgument("--headless=new");
            options.AddArgument("--disable-dev-shm-usage");
            options.AddArgument("--disable-gpu");
            options.AddArgument("--no-sandbox");
            options.AddUserProfilePreference("profile.managed_default_content_settings.images", 2);
            options.AddUserProfilePreference("profile.managed_default_content_settings.fonts", 2);
            options.AddUserProfilePreference("profile.managed_default_content_settings.stylesheets", 2);
            options.AddUserProfilePreference("profile.managed_default_content_settings.plugins", 2);
            return options;
        }
        private void ParseRacePage(IWebDriver driver, WebDriverWait wait, DateTime defaultDate, List<RunnerResult> sessionResults)
        {
            string headerText = TextOrEmpty(driver, By.CssSelector("p[class*='CourseListingHeader__StyledMainTitle']")); // header text
            if (string.IsNullOrWhiteSpace(headerText)) headerText = TextOrEmpty(driver, By.CssSelector("[data-test-id='course-title'], [class*='CourseListingHeader__StyledMainTitle']")); // alt header
            string courseName = ExtractCourseNameFromHeader(headerText); // course from header
            if (string.IsNullOrEmpty(courseName)) courseName = GuessCourseFromBreadcrumb(driver); // fallback course
            var dateText = TextOrEmpty(driver, By.CssSelector("p[class*='CourseListingHeader__StyledMainSubTitle'], [data-test-id='course-subtitle']")); // date text
            DateTime raceDate = ParseDateSafe(dateText) ?? defaultDate; // parsed date
            var raceTitle = TextOrEmpty(driver, By.CssSelector("h1[data-test-id='racecard-race-name'], h1[class*='RacingRacecardSummary__StyledTitle']")); // race title

            string metaLine = TextOrEmpty(driver, By.CssSelector("li[class*='RacingRacecardSummary__StyledAdditionalInfo'], [data-test-id='racecard-additional-info']")); // primary meta
            if (string.IsNullOrWhiteSpace(metaLine))
            {
                try
                {
                    metaLine = (string)((IJavaScriptExecutor)driver).ExecuteScript("var cand=[\"li[class*='RacingRacecardSummary__StyledAdditionalInfo']\",\"[data-test-id='racecard-additional-info']\",\"[class*='RacingRacecardSummary__StyledAdditionalInfoWrapper']\",\"[data-test-id='race-summary']\",\"[data-test-id='result-additional-info']\"];for(var i=0;i<cand.length;i++){var el=document.querySelector(cand[i]);if(el){return (el.innerText||el.textContent||'');}}return ''"); // scan likely containers
                }
                catch { metaLine = ""; } // ignore
            }
            if (string.IsNullOrWhiteSpace(metaLine))
            {
                try
                {
                    metaLine = (string)((IJavaScriptExecutor)driver).ExecuteScript("var w=document.querySelector(\"[data-test-id='race-container']\");return w?(w.innerText||''):''"); // broad section grab
                }
                catch { }
            }
            metaLine = Normalize(metaLine.Replace("•", "|").Replace("·", "|")); // normalize bullets

            var status = TextOrEmpty(driver, By.CssSelector(".RacingRacecardSummary__StyledEndState, [data-test-id='racecard-end-state']")); // end state
            if (string.IsNullOrWhiteSpace(status)) status = TextOrEmpty(driver, By.XPath("//li[contains(@class,'RacingRacecardSummary__StyledAdditionalInfo')]//span[contains(@class,'EndState') or contains(.,'Weighed In') or contains(.,'Abandoned') or contains(.,'Void')]")); // alt end state
            status = Normalize(status); // tidy
            string ageRestriction = null, distanceText = null, going = null, runners = null, offTime = null, winTime = null, surface = null; byte? classVal = null;
            try
            {
                string detailsJson = (string)((IJavaScriptExecutor)driver).ExecuteScript("var cont=document.querySelector(\"[data-test-id='race-summary'],[data-test-id='racecard-additional-info'],[data-test-id='result-additional-info'],ul[class*='MainDetailsList']\"); if(!cont) return ''; var res={}; var items=cont.querySelectorAll('li'); for(var i=0;i<items.length;i++){var spans=items[i].querySelectorAll('span'); if(spans.length>=2){var key=spans[0].innerText.trim().toLowerCase(); var val=spans[1].innerText.trim(); if(key&&val) res[key]=val;}} return JSON.stringify(res);");
                if (!string.IsNullOrWhiteSpace(detailsJson))
                {
                    var dict = JsonSerializer.Deserialize<Dictionary<string, string>>(detailsJson);
                    if (dict != null)
                    {
                        string v;
                        if (dict.TryGetValue("age restriction", out v)) ageRestriction = Normalize(v);
                        if (dict.TryGetValue("distance", out v)) distanceText = Normalize(v);
                        if (dict.TryGetValue("going", out v)) going = Normalize(v);
                        if (dict.TryGetValue("runners", out v)) runners = Normalize(v);
                        if (dict.TryGetValue("off time", out v)) offTime = Normalize(v);
                        if (dict.TryGetValue("winning time", out v)) winTime = Normalize(v);
                        if (dict.TryGetValue("surface", out v)) surface = Normalize(v);
                        if (dict.TryGetValue("class", out v)) { var m = System.Text.RegularExpressions.Regex.Match(v, @"\d+"); if (m.Success) classVal = byte.Parse(m.Value); }
                    }
                }
            }
            catch { }
            var meta = SplitMeta(metaLine); // split meta tokens
            if (string.IsNullOrWhiteSpace(ageRestriction) && meta.TryGetValue("age", out var a1)) ageRestriction = a1; // age
            if (string.IsNullOrWhiteSpace(distanceText) && meta.TryGetValue("dist", out var d1)) distanceText = d1; // distance
            if (string.IsNullOrWhiteSpace(going) && meta.TryGetValue("going", out var g1)) going = g1; // going
            if (string.IsNullOrWhiteSpace(runners) && meta.TryGetValue("runners", out var r1)) runners = r1; // runners
            if (string.IsNullOrWhiteSpace(offTime) && meta.TryGetValue("off", out var o1)) offTime = o1; // off time
            if (string.IsNullOrWhiteSpace(winTime) && meta.TryGetValue("win", out var w1)) winTime = w1; // winning time
            if (!classVal.HasValue && meta.TryGetValue("class", out var c1))
            {
                var mClassToken = System.Text.RegularExpressions.Regex.Match(c1, @"\d+");
                if (mClassToken.Success) classVal = byte.Parse(mClassToken.Value);
            }
            if (!classVal.HasValue) classVal = ExtractClass(metaLine); // class
            if (string.IsNullOrWhiteSpace(surface)) surface = InferSurface(metaLine); // surface guess

            if (string.IsNullOrWhiteSpace(going)) { going = TextOrEmpty(driver, By.CssSelector("[data-test-id='going'], [class*='Going']")); going = Normalize(going); } // going fallback
            if (string.IsNullOrWhiteSpace(distanceText)) { distanceText = TextOrEmpty(driver, By.CssSelector("[data-test-id='distance'], [class*='Distance']")); distanceText = Normalize(distanceText); } // distance fallback
            if (string.IsNullOrWhiteSpace(winTime))
            {
                var wt = TextOrEmpty(driver, By.CssSelector("[data-test-id='winning-time'], [class*='WinningTime']")); // winning time container
                if (!string.IsNullOrWhiteSpace(wt))
                {
                    var mWT = System.Text.RegularExpressions.Regex.Match(wt, @"([0-9]+m\s*[0-9.]+s|[0-9.]+s)"); // extract "Xm Y.s" or "Y.s"
                    if (mWT.Success) winTime = mWT.Groups[1].Value; // set win time
                }
            }
            if (!classVal.HasValue)
            {
                var clsTxt = TextOrEmpty(driver, By.CssSelector("[data-test-id='race-class'], [class*='RaceClass']")); // class label
                var mC = System.Text.RegularExpressions.Regex.Match(clsTxt ?? "", @"Class\s*(\d)"); // parse digit
                if (mC.Success) classVal = byte.Parse(mC.Groups[1].Value); // set class
            }

            if (string.IsNullOrWhiteSpace(distanceText) || string.IsNullOrWhiteSpace(winTime) || !classVal.HasValue || string.IsNullOrWhiteSpace(going))
            {
                try
                {
                    string json = (string)((IJavaScriptExecutor)driver).ExecuteScript("var s=[].slice.call(document.querySelectorAll('script[type=\"application/ld+json\"]'));for(var i=0;i<s.length;i++){try{var o=JSON.parse(s[i].textContent||s[i].innerText||'{}');if(o&&((o['@type']&&String(o['@type']).toLowerCase().includes('horserace'))||(o['sport']&&String(o['sport']).toLowerCase().includes('horse'))))return JSON.stringify(o);if(o&&o['@graph']){for(var j=0;j<o['@graph'].length;j++){var g=o['@graph'][j];if(g&&g['@type']&&String(g['@type']).toLowerCase().includes('horserace'))return JSON.stringify(g);}}}catch(e){}}return ''"); // pull JSON-LD
                    if (!string.IsNullOrWhiteSpace(json))
                    {
                        var mDist = System.Text.RegularExpressions.Regex.Match(json, @"""(?:distance|raceDistance|distanceName)""\s*:\s*""(.*?)""", System.Text.RegularExpressions.RegexOptions.IgnoreCase); // distance in JSON
                        if (mDist.Success && string.IsNullOrWhiteSpace(distanceText)) distanceText = mDist.Groups[1].Value; // set distance
                        var mGoing = System.Text.RegularExpressions.Regex.Match(json, @"""(?:going|ground)""\s*:\s*""(.*?)""", System.Text.RegularExpressions.RegexOptions.IgnoreCase); // going in JSON
                        if (mGoing.Success && string.IsNullOrWhiteSpace(going)) going = mGoing.Groups[1].Value; // set going
                        var mClass = System.Text.RegularExpressions.Regex.Match(json, @"""(?:raceClass|class)""\s*:\s*""?(\d)""?", System.Text.RegularExpressions.RegexOptions.IgnoreCase); // class in JSON
                        if (mClass.Success && !classVal.HasValue) classVal = byte.Parse(mClass.Groups[1].Value); // set class
                        var mWTjson = System.Text.RegularExpressions.Regex.Match(json, @"""(?:winningTime|time)""\s*:\s*""(.*?)""", System.Text.RegularExpressions.RegexOptions.IgnoreCase); // time in JSON
                        if (mWTjson.Success && string.IsNullOrWhiteSpace(winTime)) winTime = mWTjson.Groups[1].Value; // set win time
                    }
                }
                catch { }
            }

            if (string.IsNullOrWhiteSpace(distanceText) || string.IsNullOrWhiteSpace(going) || string.IsNullOrWhiteSpace(offTime) || string.IsNullOrWhiteSpace(winTime) || string.IsNullOrWhiteSpace(runners) || !classVal.HasValue || string.IsNullOrWhiteSpace(surface))
            {
                string allText = (string)((IJavaScriptExecutor)driver).ExecuteScript("return (document.body.innerText||'')"); // page text
                if (string.IsNullOrWhiteSpace(distanceText)) { var mDist = System.Text.RegularExpressions.Regex.Match(allText, @"\b(\d+\s*m(?:\s*\d+\s*f)?(?:\s*\d+\s*y)?)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mDist.Success) distanceText = mDist.Groups[1].Value; } // distance
                if (string.IsNullOrWhiteSpace(going)) { var mGoing = System.Text.RegularExpressions.Regex.Match(allText, @"\b(Heavy|Soft|Good to Soft|Good|Good to Firm|Firm|Standard(?: to (?:Slow|Fast))?|Yielding)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mGoing.Success) going = mGoing.Groups[1].Value; } // going
                if (string.IsNullOrWhiteSpace(offTime)) { var mOff = System.Text.RegularExpressions.Regex.Match(allText, @"Off time\s*[:\-]?\s*([0-2]?\d:[0-5]\d)", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mOff.Success) offTime = mOff.Groups[1].Value; } // off time
                if (string.IsNullOrWhiteSpace(winTime)) { var mWin = System.Text.RegularExpressions.Regex.Match(allText, @"Winning time\s*[:\-]?\s*([0-9]+\s*m\s*[0-9.]+s|[0-9.]+s)", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mWin.Success) winTime = mWin.Groups[1].Value; } // winning time
                if (string.IsNullOrWhiteSpace(runners)) { var mRun = System.Text.RegularExpressions.Regex.Match(allText, @"\b(\d+)\s+Runners\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mRun.Success) runners = mRun.Groups[1].Value + " Runners"; } // runners
                if (!classVal.HasValue) { var mClass = System.Text.RegularExpressions.Regex.Match(allText, @"\bClass\s*(\d)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mClass.Success) classVal = byte.Parse(mClass.Groups[1].Value); } // class
                if (string.IsNullOrWhiteSpace(surface)) surface = InferSurface(allText); // surface from text
            }

            int? runnerCount = TryParseOrdinalInt(runners?.Split(' ').FirstOrDefault()); // runner count
            int distanceYards = ParseDistanceToYards(distanceText); // yards
            TimeSpan? scheduledOff = ExtractScheduledOffFromHeader(headerText) ?? ParseClockTime(offTime); // scheduled off
            TimeSpan? actualOff = ParseClockTime(offTime); // actual off
            int? winningMs = ParseWinningMs(winTime); // winning ms

            var courseId = _repo.InsertCourse(new Course { Name = courseName }); // upsert course
            var raceEntity = new Race
            {
                CourseId = courseId, // course
                RaceDate = raceDate, // date
                ScheduledOff = scheduledOff ?? TimeSpan.Zero, // sched off
                ActualOff = actualOff, // actual off
                Title = raceTitle, // title
                RaceType = string.Empty, // type
                Class = classVal, // class
                AgeRestriction = ageRestriction, // age
                Surface = surface, // surface
                Going = going, // going
                DistanceYards = distanceYards, // yards
                DistanceText = distanceText ?? string.Empty, // text
                RunnerCount = runnerCount.HasValue ? (byte?)runnerCount : null, // runners
                Status = status, // status
                WinningTimeMs = winningMs, // ms
                WinningTimeText = winTime // text
            };
            var raceId = _repo.InsertRace(raceEntity); // insert race

            wait.Until(d => d.FindElements(By.CssSelector("[class*='ResultRunner__StyledResultRunnerWrapper']")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id='horse-sub-info']")).Count > 0); // wait rows
            var rows = driver.FindElements(By.CssSelector("[class*='ResultRunner__StyledResultRunnerWrapper']")); // rows
            var results = new List<RunnerResult>(); // collect

            foreach (var row in rows)
            {
                string posRaw = SafeText(row, By.CssSelector("[data-test-id='position-no'], .position-no .ordinal")); // position text
                int? finishPos = TryParseOrdinalInt(posRaw); // numeric pos
                string outcome = finishPos.HasValue ? "" : ParseOutcomeCode(posRaw); // outcome code
                string cloth = SafeText(row, By.CssSelector("[data-test-id='saddle-cloth-no']")); // saddle cloth
                byte? saddle = TryParseByte(cloth); // saddle no
                string drawRaw = (string)((IJavaScriptExecutor)driver).ExecuteScript("var el=arguments[0].querySelector('[data-test-id=\"stall-no\"]'); return el?(el.textContent||'').trim():'';", row); // stall text
                byte? stall = TryParseByteLoose(drawRaw); // stall no
                string horseName = SafeText(row, By.CssSelector("a[href*='/racing/profiles/horse/'], [data-test-id='horse-name']")); // horse
                string ageTxt = (string)((IJavaScriptExecutor)driver).ExecuteScript("var el=arguments[0].querySelector('[data-test-id=\"horse-sub-info\"] span:first-child'); return el?(el.textContent||'').trim():'';", row); // age text
                byte? age = TryParseByte(ageTxt); // age
                string weightTxt = (string)((IJavaScriptExecutor)driver).ExecuteScript("var el=arguments[0].querySelector('[data-test-id=\"horse-sub-info\"] span:last-child'); return el?(el.textContent||'').trim():'';", row); // weight text
                byte? weightLbs = null; // weight lbs
                var m = System.Text.RegularExpressions.Regex.Match(weightTxt ?? "", @"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$"); // "s-st"
                if (m.Success)
                {
                    int stones = int.Parse(m.Groups[1].Value); // stones
                    int pounds = int.Parse(m.Groups[2].Value); // pounds
                    int total = stones * 14 + pounds; // total lbs
                    if (total >= byte.MinValue && total <= byte.MaxValue) weightLbs = (byte)total; // clamp
                }
                var (trainer, jockey) = ExtractTrainerJockey(row); // names
                string spFrac = SafeText(row, By.CssSelector("span[class*='BetLinkStyle'], [data-test-id='sp-odds']")); // sp frac
                decimal? spDec = FractionToDecimal(spFrac); // sp dec
                string favTag = ExtractFavouriteTag(spFrac); // fav tag
                string beatenTxt = SafeText(row, By.CssSelector("[class*='StyledFinishDistance'], [data-test-id='finish-distance']")); // beaten text
                decimal? beatenLen = ParseBeatenLengths(beatenTxt); // beaten len
                string opTxt = SafeText(row, By.XPath(".//*[contains(.,'op ') and contains(@class,'small')]")); // opening odds
                string tchTxt = SafeText(row, By.XPath(".//*[contains(.,'tchd ') or contains(.,'tch ') and contains(@class,'small')]")); // touched odds
                string opFrac = ExtractOddsToken(opTxt, "op"); // op frac
                (string? tchLow, string? tchHigh) = ExtractTouchedTokens(tchTxt); // low/high
                string comment = SafeText(row, By.CssSelector("[data-test-id='ride-description'], [class*='StyledRideDescription']")); // comment
                var trainerId = string.IsNullOrWhiteSpace(trainer) ? (int?)null : _repo.InsertTrainer(new Trainer { Name = trainer }); // trainer id
                var jockeyId = string.IsNullOrWhiteSpace(jockey) ? (int?)null : _repo.InsertJockey(new Jockey { Name = jockey }); // jockey id
                var horseId = _repo.InsertHorse(new Horse { Name = horseName }); // horse id
                var result = new RunnerResult
                {
                    RaceId = raceId, // race
                    HorseId = horseId, // horse
                    TrainerId = trainerId, // trainer
                    JockeyId = jockeyId, // jockey
                    SaddleclothNumber = saddle, // saddle
                    Draw = stall, // draw
                    Age = age, // age
                    WeightLbs = weightLbs, // weight
                    WeightText = weightTxt, // text
                    FinishPos = finishPos.HasValue ? (short?)finishPos.Value : null, // pos
                    OutcomeCode = outcome, // outcome
                    DistanceBeatenText = beatenTxt, // beaten
                    DistanceBeatenLengths = beatenLen, // beaten len
                    SP_Fraction = spFrac, // sp
                    SP_Decimal = spDec, // sp dec
                    FavTag = favTag, // fav
                    OpeningFraction = opFrac, // op
                    TouchedHighFraction = tchHigh, // high
                    TouchedLowFraction = tchLow, // low
                    Comment = comment // comment
                };
                results.Add(result); // add row
            }
            if (results.Count > 0) sessionResults.AddRange(results); // accumulate for bulk insert
        }

        private static byte? TryParseByteLoose(string? s) { if (string.IsNullOrWhiteSpace(s)) return null; var m = System.Text.RegularExpressions.Regex.Match(s, @"\d+"); return m.Success && byte.TryParse(m.Value, out var v) ? v : (byte?)null; } // handles "(2)", " 2 ", etc.
        private static (string trainer, string jockey) ExtractTrainerJockey(IWebElement row)
        {
            try
            {
                string trainer = "", jockey = ""; // init
                trainer = SafeText(row, By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]//span[normalize-space(translate(., '\u00A0',' '))='T:']/following-sibling::span[contains(@class,'StyledPersonName')][1]")); if (!string.IsNullOrWhiteSpace(trainer)) trainer = trainer.Trim(); // primary trainer: span after T:
                jockey = SafeText(row, By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]//span[normalize-space(translate(., '\u00A0',' '))='J:']/following-sibling::span[contains(@class,'StyledPersonName')][1]")); if (!string.IsNullOrWhiteSpace(jockey)) jockey = jockey.Trim(); // primary jockey: span after J:
                if (string.IsNullOrWhiteSpace(trainer)) { trainer = SafeText(row, By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]//span[starts-with(normalize-space(translate(., '\u00A0',' ')),'T:')]/following-sibling::span[contains(@class,'StyledPersonName')][1]")); if (!string.IsNullOrWhiteSpace(trainer)) trainer = trainer.Trim(); } // tolerate stray spacing
                if (string.IsNullOrWhiteSpace(jockey)) { jockey = SafeText(row, By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]//span[starts-with(normalize-space(translate(., '\u00A0',' ')),'J:')]/following-sibling::span[contains(@class,'StyledPersonName')][1]")); if (!string.IsNullOrWhiteSpace(jockey)) jockey = jockey.Trim(); } // tolerate stray spacing
                if (string.IsNullOrWhiteSpace(trainer) || string.IsNullOrWhiteSpace(jockey))
                {
                    var names = row.FindElements(By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]//span[contains(@class,'StyledPersonName')]")); // fallback: first two name spans
                    if (string.IsNullOrWhiteSpace(trainer) && names.Count >= 1) trainer = names[0].Text.Trim(); // trainer = first
                    if (string.IsNullOrWhiteSpace(jockey) && names.Count >= 2) jockey = names[1].Text.Trim(); // jockey = second
                }
                if (string.IsNullOrWhiteSpace(trainer) || string.IsNullOrWhiteSpace(jockey))
                {
                    var block = SafeText(row, By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]")).Replace('\u00A0', ' '); // final text fallback
                    if (string.IsNullOrWhiteSpace(trainer)) { var mT = System.Text.RegularExpressions.Regex.Match(block, @"(?:^|\s)T:\s*([A-Za-z .&'\-]+?)(?=\s+J:|$)", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mT.Success) trainer = mT.Groups[1].Value.Trim(); }
                    if (string.IsNullOrWhiteSpace(jockey)) { var mJ = System.Text.RegularExpressions.Regex.Match(block, @"(?:^|\s)J:\s*([A-Za-z .&'\-]+)$", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (mJ.Success) jockey = mJ.Groups[1].Value.Trim(); }
                }
                return (trainer, jockey); // done
            }
            catch { return ("", ""); } // safe default
        }
        private static int? TryParseOrdinalInt(string? s)
        {
            if (string.IsNullOrWhiteSpace(s)) return null; s = s.Trim(); var m = System.Text.RegularExpressions.Regex.Match(s, @"^\s*(\d+)\s*(st|nd|rd|th)?\s*$", System.Text.RegularExpressions.RegexOptions.IgnoreCase); if (m.Success && int.TryParse(m.Groups[1].Value, out var v)) return v; return int.TryParse(s, out var v2) ? v2 : (int?)null; // handles 1st/2nd/3rd/4th
        }

        private static string GuessCourseFromBreadcrumb(IWebDriver d) { try { return d.FindElements(By.CssSelector("p[class*='CourseListingHeader__StyledMainTitle']")).FirstOrDefault()?.Text?.Split(' ').LastOrDefault() ?? ""; } catch { return ""; } } // fallback

        private static string ExtractCourseNameFromHeader(string header)
        {
            if (string.IsNullOrWhiteSpace(header)) return ""; var toks = header.Trim().Split(' ', 2); if (toks.Length < 2) return ""; return toks[1].Trim(); // from "14:10 Beverley"
        }

        private static TimeSpan? ExtractScheduledOffFromHeader(string header) { if (string.IsNullOrWhiteSpace(header)) return null; var time = header.Split(' ').FirstOrDefault(); return ParseClockTime(time); } // HH:mm

        private static string Normalize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return ""; s = s.Replace('\u00A0', ' '); s = System.Text.RegularExpressions.Regex.Replace(s, @"\s+", " "); return s.Trim(); // collapse spaces
        }

        private static Dictionary<string, string> SplitMeta(string meta)
        {
            var r = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase); if (string.IsNullOrWhiteSpace(meta)) return r;
            meta = Normalize(meta); var parts = meta.Split('|', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries); // tokens

            foreach (var p in parts)
            {
                if (string.IsNullOrEmpty(t)) continue;
                var clean = CleanMetaValue(t);

                if (t.Contains("YO", StringComparison.OrdinalIgnoreCase)) r["age"] = clean; // age
                else if (t.Contains("Runners", StringComparison.OrdinalIgnoreCase)) r["runners"] = clean; // runners
                else if (t.StartsWith("Off time", StringComparison.OrdinalIgnoreCase)) r["off"] = CleanMetaValue(t.Split(':', 2)[1]); // off time
                else if (t.StartsWith("Winning time", StringComparison.OrdinalIgnoreCase)) r["win"] = CleanMetaValue(t.Split(':', 2)[1]); // win time
                else if (System.Text.RegularExpressions.Regex.IsMatch(t, @"\bClass\s*\d", System.Text.RegularExpressions.RegexOptions.IgnoreCase)) r["class"] = clean; // class token
                else if (IsGoingToken(t)) r["going"] = clean; // going
                else if (HasDistanceToken(t)) r["dist"] = clean; // distance
            }

            return r;
        }
        private static string CleanMetaValue(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return "";
            value = value.Trim();
            value = value.TrimEnd('.', ';', ',');
            return value;
        }
        private static int? ParseWinningMs(string? t)
        {
            if (string.IsNullOrWhiteSpace(t) || t == "-") return null; // "1m 15.11s" or "59.87s"
            try
            {
                t = t.Trim();
                var match = System.Text.RegularExpressions.Regex.Match(t, @"([0-9]+m\s*[0-9.]+s|[0-9.]+s)", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                if (match.Success) t = match.Groups[1].Value;
                t = t.Trim().TrimEnd('.', ';', ',');

                double seconds = 0;
                if (t.Contains('m'))
                {
                    var parts = t.Split('m', 2, StringSplitOptions.TrimEntries);
                    if (parts.Length != 2) return null;
                    var mins = int.Parse(parts[0], CultureInfo.InvariantCulture);
                    var secsText = parts[1].Trim().TrimEnd('s', 'S');
                    var secs = double.Parse(secsText, CultureInfo.InvariantCulture);
                    seconds = mins * 60 + secs;
                }
                else
                {
                    var secsText = t.TrimEnd('s', 'S');
                    seconds = double.Parse(secsText, CultureInfo.InvariantCulture);
                }
                return (int)Math.Round(seconds * 1000.0);
            }
            catch { return null; }
        }
        private static bool IsGoingToken(string s) { if (string.IsNullOrWhiteSpace(s)) return false; var keys = new[] { "Good", "Firm", "Soft", "Heavy", "Standard", "Yielding" }; return keys.Any(k => s.Contains(k, StringComparison.OrdinalIgnoreCase)); } // going marker

        private static bool HasDistanceToken(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return false;
            s = s.ToLowerInvariant();
            return System.Text.RegularExpressions.Regex.IsMatch(s, @"\b\d+\s*(miles?|m|furlongs?|f|yards?|y)\b");
        } // match typical distance patterns like "2m 5f"
        private static int ParseDistanceToYards(string? text)
        {
            if (string.IsNullOrWhiteSpace(text)) return 0; int yards = 0; // accumulate
            foreach (var tok in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (tok.EndsWith("m", StringComparison.OrdinalIgnoreCase) && int.TryParse(tok.TrimEnd('m', 'M'), out var m)) yards += m * 1760; // miles
                else if (tok.EndsWith("f", StringComparison.OrdinalIgnoreCase) && int.TryParse(tok.TrimEnd('f', 'F'), out var f)) yards += f * 220; // furlongs
                else if (tok.EndsWith("y", StringComparison.OrdinalIgnoreCase) && int.TryParse(tok.TrimEnd('y', 'Y'), out var y)) yards += y; // yards
            }
            return yards;
        }

        private static TimeSpan? ParseClockTime(string? t) { if (string.IsNullOrWhiteSpace(t)) return null; return TimeSpan.TryParse(t, CultureInfo.InvariantCulture, out var ts) ? ts : null; } // HH:mm
        private static string ParseOutcomeCode(string posText)
        {
            if (string.IsNullOrWhiteSpace(posText)) return ""; var t = posText.Trim().ToUpperInvariant(); var letters = new string(t.Where(char.IsLetter).ToArray()); if (letters.Length == 0) return "";
            switch (letters) { case "PU": case "F": case "UR": case "RO": case "DSQ": case "BD": case "SU": case "RR": case "REF": case "WD": case "NR": case "VOID": case "CO": case "DNF": return letters; default: return letters; } // map codes
        }

        private static (byte? age, byte? lbs, string weightTxt) ParseAgeWeight(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return (null, null, ""); byte? age = null; var ageMatch = System.Text.RegularExpressions.Regex.Match(s, @"\((\d{1,2})\)"); if (ageMatch.Success) age = byte.Parse(ageMatch.Groups[1].Value); string wt = System.Text.RegularExpressions.Regex.Match(s, @"\d{1,2}-\d{1,2}").Value; byte? lbs = null; if (!string.IsNullOrEmpty(wt)) { var parts = wt.Split('-'); lbs = (byte)(int.Parse(parts[0]) * 14 + int.Parse(parts[1])); }
            return (age, lbs, wt); // age & lbs
        }
        private static string InferRaceType(string title) { if (string.IsNullOrWhiteSpace(title)) return ""; var keys = new[] { "Handicap", "Maiden", "Novice", "Apprentice", "Claiming", "Selling", "Stakes" }; return keys.FirstOrDefault(k => title.IndexOf(k, StringComparison.OrdinalIgnoreCase) >= 0) ?? ""; } // type
        private static byte? ExtractClass(string meta) { if (string.IsNullOrWhiteSpace(meta)) return null; var m = System.Text.RegularExpressions.Regex.Match(meta, @"Class\s+(\d)"); return m.Success ? (byte?)byte.Parse(m.Groups[1].Value) : null; } // Class 1..7
        private static string InferSurface(string meta) { if (string.IsNullOrWhiteSpace(meta)) return null; if (meta.Contains("All Weather", StringComparison.OrdinalIgnoreCase) || meta.Contains("Allweather", StringComparison.OrdinalIgnoreCase)) return "Allweather"; return "Turf"; } // surface

        private static bool AcceptTermsIfPresent(IWebDriver driver)
        {
            try
            {
                var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(2)); var acceptButton = wait.Until(d => { var buttons = d.FindElements(By.XPath("//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow all cookies')]")); return buttons.FirstOrDefault(b => b.Displayed && b.Enabled); }); acceptButton?.Click(); return acceptButton != null; // clicked
            }
            catch (WebDriverTimeoutException) { return false; } // none
        }
    }
}
//using HorseRacingML.Data;
//using HorseRacingML.Models;
//using HorseRacingML.Services;
//using HtmlAgilityPack;
//using System;
//using System.Collections.Generic;
//using System.Globalization;
//using System.Linq;
//using System.Net.Http;
//using System.Net.Http.Headers;
//using System.Text.Json;
//using System.Text.RegularExpressions;
//using System.Threading;
//using System.Threading.Tasks;
//using System.Xml;
//using System.Collections.Concurrent;
//using System.Xml.XPath;

//namespace HorseRacingML.Scraping
//{
//    public class RaceResultsScraper
//    {
//        private readonly RacingRepository _repo;
//        private readonly ScrapingStatusService _status;
//        private readonly HttpClient _httpClient;

//        private const int MaxConcurrentRequests = 6;
//        private static readonly Uri BaseUri = new Uri("https://www.sportinglife.com/");
//        private static readonly string[] InternationalQuerySuffixes = new[]
//        {
//            "?tab=international",
//            "?region=international",
//            "?switch=international",
//            "?country=international",
//            "?countries=international"
//        };

//        private sealed record RaceLinkExtractionResult(
//            IReadOnlyCollection<string> Links,
//            IReadOnlyCollection<Uri> AlternateRegionUris,
//            bool SawInternationalToggle);


//        public RaceResultsScraper(RacingRepository repo, ScrapingStatusService status)
//            : this(repo, status, CreateDefaultClient())
//        {
//        }

//        internal RaceResultsScraper(RacingRepository repo, ScrapingStatusService status, HttpClient httpClient)
//        {
//            _repo = repo;
//            _status = status;
//            _httpClient = httpClient;
//        }
//        private enum DayScrapeStatus
//        {
//            Inserted,
//            NoResults,
//            NoLinks
//        }

//        private async Task ScrapeAsync(DateTime startDate, DateTime endDate)
//        {
//            var start = startDate.Date;
//            var end = endDate.Date;
//            var today = DateTime.Today;

//            if (end > today) end = today;
//            if (start > end) return;

//            var dates = Enumerable.Range(0, (end - start).Days + 1)
//                                   .Select(i => end.AddDays(-i))
//                                   .ToList();

//            using var semaphore = new SemaphoreSlim(MaxConcurrentRequests);
//            var tasks = dates.Select(async date =>
//            {
//                await semaphore.WaitAsync();
//                try
//                {
//                    var result = await ScrapeDayAsync(date);

//                    string msg;
//                    var writeToStdOut = true;

//                    switch (result)
//                    {
//                        case DayScrapeStatus.Inserted:
//                            msg = $"[{date:yyyy-MM-dd}] parsed and inserted";
//                            break;
//                        case DayScrapeStatus.NoResults:
//                            msg = $"[{date:yyyy-MM-dd}] parsed (no race results)";
//                            break;
//                        case DayScrapeStatus.NoLinks:
//                            msg = $"[{date:yyyy-MM-dd}] No race links detected";
//                            writeToStdOut = false;
//                            break;
//                        default:
//                            msg = $"[{date:yyyy-MM-dd}] parsed";
//                            break;
//                    }

//                    if (writeToStdOut)
//                    {
//                        Console.WriteLine(msg);
//                    }

//                    _status.Update(msg);
//                }
//                catch (Exception ex)
//                {
//                    Console.Error.WriteLine($"[Skip Day] {date:yyyy-MM-dd} {ex.Message}");
//                }
//                finally
//                {
//                    semaphore.Release();
//                }
//            }).ToArray();

//            await Task.WhenAll(tasks);
//        }

//        private async Task<DayScrapeStatus> ScrapeDayAsync(DateTime date)
//        {
//            var dayUri = new Uri(BaseUri, $"racing/results/{date:yyyy-MM-dd}");
//            var (dayDoc, dayHtml) = await LoadDocumentAsync(dayUri);
//            var visitedPages = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
//            {
//                dayUri.ToString()
//            };
//            var fallbackPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
//            var pendingPages = new Queue<Uri>();
//            var raceLinkSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

//            static bool ContainsInternationalText(string? html) =>
//                !string.IsNullOrEmpty(html) &&
//                html.IndexOf("International", StringComparison.OrdinalIgnoreCase) >= 0;

//            void EnqueuePage(Uri uri)
//            {
//                var key = uri.ToString();
//                if (visitedPages.Add(key))
//                {
//                    pendingPages.Enqueue(uri);
//                }
//            }

//            void EnqueueFallbackCandidates(Uri baseUri)
//            {
//                var basePath = baseUri.GetLeftPart(UriPartial.Path);
//                if (!fallbackPaths.Add(basePath))
//                {
//                    return;
//                }

//                foreach (var suffix in InternationalQuerySuffixes)
//                {
//                    if (Uri.TryCreate(basePath + suffix, UriKind.Absolute, out var candidate))
//                    {
//                        EnqueuePage(candidate);
//                    }
//                }
//            }

//            var initialExtraction = ExtractRaceLinks(dayDoc, dayHtml, date, dayUri);
//            foreach (var link in initialExtraction.Links)
//            {
//                raceLinkSet.Add(link);
//            }

//            foreach (var altUri in initialExtraction.AlternateRegionUris)
//            {
//                EnqueuePage(altUri);
//            }

//            if (initialExtraction.SawInternationalToggle || ContainsInternationalText(dayHtml))
//            {
//                EnqueueFallbackCandidates(dayUri);
//            }

//            while (pendingPages.Count > 0)
//            {
//                var pageUri = pendingPages.Dequeue();
//                HtmlDocument pageDoc;
//                string pageHtml;

//                try
//                {
//                    (pageDoc, pageHtml) = await LoadDocumentAsync(pageUri);
//                }
//                catch (HttpRequestException ex)
//                {
//                    Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Alternate page fetch failed {pageUri}: {ex.Message}");
//                    continue;
//                }
//                catch (TaskCanceledException ex)
//                {
//                    Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Alternate page fetch timed out {pageUri}: {ex.Message}");
//                    continue;
//                }

//                var extraction = ExtractRaceLinks(pageDoc, pageHtml, date, pageUri);
//                foreach (var link in extraction.Links)
//                {
//                    raceLinkSet.Add(link);
//                }

//                foreach (var nextUri in extraction.AlternateRegionUris)
//                {
//                    EnqueuePage(nextUri);
//                }

//                if (extraction.SawInternationalToggle || ContainsInternationalText(pageHtml))
//                {
//                    EnqueueFallbackCandidates(pageUri);
//                }
//            }

//            var raceLinks = raceLinkSet.OrderBy(l => l, StringComparer.OrdinalIgnoreCase).ToList();
//            if (raceLinks.Count == 0)
//            {
//                Console.Error.WriteLine($"[{date:yyyy-MM-dd}] No race links detected");
//                return DayScrapeStatus.NoLinks;
//            }
//            Console.WriteLine($"[{date:yyyy-MM-dd}] Found {raceLinks.Count} race links");
//            var resultsBag = new ConcurrentBag<RunnerResult>();
//            using var semaphore = new SemaphoreSlim(MaxConcurrentRequests);

//            var raceTasks = raceLinks.Select(async raceUrl =>
//            {
//                await semaphore.WaitAsync();
//                try
//                {
//                    var (raceDoc, raceHtml) = await LoadDocumentAsync(new Uri(raceUrl));
//                    var results = ParseRacePage(raceDoc, raceHtml, date);
//                    foreach (var result in results)
//                    {
//                        resultsBag.Add(result);
//                    }
//                }
//                catch (HttpRequestException ex)
//                {
//                    Console.Error.WriteLine($"[Skip Race] {date:yyyy-MM-dd} {raceUrl}: {ex.Message}");
//                }
//                catch (TaskCanceledException ex)
//                {
//                    Console.Error.WriteLine($"[Skip Race] {date:yyyy-MM-dd} {raceUrl}: {ex.Message}");
//                }
//                catch (Exception ex)
//                {
//                    Console.Error.WriteLine($"[Skip Race] {date:yyyy-MM-dd} {raceUrl}: {ex.Message}");
//                }
//                finally
//                {
//                    semaphore.Release();
//                }
//            }).ToArray();

//            await Task.WhenAll(raceTasks);

//            if (!resultsBag.IsEmpty)
//            {
//                _repo.BulkInsertRunnerResults(resultsBag.ToList());
//                return DayScrapeStatus.Inserted;
//            }

//            return DayScrapeStatus.NoResults;
//        }

//        private static HttpClient CreateDefaultClient()
//        {
//            var client = new HttpClient();
//            client.DefaultRequestHeaders.UserAgent.ParseAdd("HorseRacingMLBot/1.0 (+https://github.com/kieranwaters/HorseRacingML)");
//            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("text/html"));
//            client.Timeout = TimeSpan.FromSeconds(30);
//            return client;
//        }

//        public void ScrapeFromEarliest()
//        {
//            var minDate = new DateTime(2000, 1, 1);
//            var earliest = _repo.GetEarliestRaceDate();
//            if (earliest <= minDate)
//            {
//                _status.Update("Scraping finished.");
//                return;
//            }

//            var currentEnd = earliest.AddDays(-1);
//            while (currentEnd >= minDate)
//            {
//                var start = currentEnd.AddDays(-(MaxConcurrentRequests - 1));
//                if (start < minDate)
//                {
//                    start = minDate;
//                }

//                _status.Update($"Scraping {start:yyyy-MM-dd} to {currentEnd:yyyy-MM-dd}");
//                Scrape(start, currentEnd);

//                if (start == minDate)
//                {
//                    break;
//                }

//                var newEarliest = _repo.GetEarliestRaceDate();
//                var candidateEnd = newEarliest.AddDays(-1);

//                if (candidateEnd < currentEnd)
//                {
//                    currentEnd = candidateEnd;
//                }
//                else
//                {
//                    currentEnd = start.AddDays(-1);
//                }
//            }

//            _status.Update("Scraping finished.");
//        }

//        public void ScrapeFromLatest()
//        {
//            var today = DateTime.Today;
//            while (true)
//            {
//                var latest = _repo.GetLatestRaceDate();
//                if (latest > today) latest = today;

//                var start = latest.AddDays(1);
//                if (start > today) break;

//                var end = start.AddDays(MaxConcurrentRequests - 1);
//                if (end > today) end = today;

//                _status.Update($"Scraping {start:yyyy-MM-dd} to {end:yyyy-MM-dd}");
//                Scrape(start, end);
//            }

//            _status.Update("Scraping finished.");
//        }

//        public void Scrape(DateTime startDate, DateTime endDate)
//        {
//            ScrapeAsync(startDate, endDate).GetAwaiter().GetResult();
//        }
//        private RaceLinkExtractionResult ExtractRaceLinks(HtmlDocument doc, string rawHtml, DateTime date, Uri pageUri)
//        {
//            var links = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
//            var dateToken = date.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
//            var alternatePages = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
//            var sawInternationalToggle = false;

//            var sampleLinks = new List<string>();
//            var sampleSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
//            void AddSample(string value)
//            {
//                if (string.IsNullOrWhiteSpace(value)) return;
//                if (sampleSet.Count >= 5) return;
//                if (sampleSet.Add(value))
//                {
//                    sampleLinks.Add(value);
//                }
//            }

//            bool TryCollectAlternatePage(string? raw)
//            {
//                if (string.IsNullOrWhiteSpace(raw))
//                {
//                    return false;
//                }

//                var candidateText = HtmlEntity.DeEntitize(raw).Trim();
//                if (candidateText.Length == 0)
//                {
//                    return false;
//                }

//                if (candidateText.StartsWith("javascript:", StringComparison.OrdinalIgnoreCase))
//                {
//                    return false;
//                }

//                if (candidateText.StartsWith("//", StringComparison.Ordinal))
//                {
//                    candidateText = $"https:{candidateText}";
//                }

//                Uri? candidateUri = null;
//                if (Uri.TryCreate(candidateText, UriKind.Absolute, out var absolute))
//                {
//                    candidateUri = absolute;
//                }
//                else if (Uri.TryCreate(pageUri, candidateText, out var relativeToPage))
//                {
//                    candidateUri = relativeToPage;
//                }
//                else if (Uri.TryCreate(BaseUri, candidateText, out var relativeToBase))
//                {
//                    candidateUri = relativeToBase;
//                }

//                if (candidateUri == null)
//                {
//                    return false;
//                }

//                if (!string.Equals(candidateUri.Host, BaseUri.Host, StringComparison.OrdinalIgnoreCase))
//                {
//                    return false;
//                }

//                var segments = candidateUri.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
//                var resultsIndex = Array.IndexOf(segments, "results");
//                if (resultsIndex < 0 || resultsIndex + 1 >= segments.Length)
//                {
//                    return false;
//                }

//                var dateSegment = segments[resultsIndex + 1];
//                if (!string.Equals(dateSegment, dateToken, StringComparison.OrdinalIgnoreCase))
//                {
//                    return false;
//                }

//                if (segments.Length > resultsIndex + 2)
//                {
//                    return false;
//                }

//                var query = candidateUri.Query;
//                if (string.IsNullOrEmpty(query))
//                {
//                    return false;
//                }

//                if (query.IndexOf("international", StringComparison.OrdinalIgnoreCase) < 0)
//                {
//                    return false;
//                }

//                if (!alternatePages.Add(candidateUri.ToString()))
//                {
//                    return false;
//                }

//                sawInternationalToggle = true;
//                return true;
//            }

//            try
//            {
//                var switchNodes = doc.DocumentNode.SelectNodes("//*[@data-test-id][contains(@data-test-id,'switch') or contains(@data-test-id,'toggle') or contains(@data-test-id,'country')]");
//                if (switchNodes != null)
//                {
//                    foreach (var node in switchNodes)
//                    {
//                        var text = Normalize(node.InnerText ?? string.Empty);
//                        if (!string.IsNullOrEmpty(text) && text.IndexOf("international", StringComparison.OrdinalIgnoreCase) >= 0)
//                        {
//                            sawInternationalToggle = true;
//                            break;
//                        }

//                        var attr = node.GetAttributeValue("data-test-id", string.Empty);
//                        if (attr.IndexOf("international", StringComparison.OrdinalIgnoreCase) >= 0)
//                        {
//                            sawInternationalToggle = true;
//                            break;
//                        }
//                    }
//                }
//            }
//            catch
//            {
//                // Ignore toggle detection errors – debugging will capture missing toggles.
//            }

//            if (!sawInternationalToggle && !string.IsNullOrEmpty(rawHtml))
//            {
//                if (rawHtml.IndexOf("new-switch-button", StringComparison.OrdinalIgnoreCase) >= 0 &&
//                    rawHtml.IndexOf("international", StringComparison.OrdinalIgnoreCase) >= 0)
//                {
//                    sawInternationalToggle = true;
//                }
//            }


//            int regexMatches = 0;
//            int regexAdded = 0;
//            int regexDateOnly = 0;
//            int regexInvalid = 0;
//            int regexSkipped = 0;

//            if (!string.IsNullOrWhiteSpace(rawHtml))
//            {
//                var regexSource = rawHtml;
//                if (regexSource.IndexOf("\\/", StringComparison.Ordinal) >= 0)
//                {
//                    regexSource = regexSource.Replace("\\/", "/");
//                }

//                if (regexSource.IndexOf("\\u002f", StringComparison.OrdinalIgnoreCase) >= 0)
//                {
//                    regexSource = regexSource
//                        .Replace("\\u002F", "/")
//                        .Replace("\\u002f", "/");
//                }

//                var regex = new Regex($"((?:https?:)?//(?:www\\.)?sportinglife\\.com)?/racing/results/{dateToken}/[^\"'#<\\s]+", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
//                foreach (Match match in regex.Matches(regexSource))
//                {
//                    regexMatches++;

//                    var captured = match.Value;
//                    TryCollectAlternatePage(captured);
//                    if (!TryNormalizeRaceUrl(captured, out var normalized, out var isDateOnly))
//                    {
//                        regexInvalid++;
//                        continue;
//                    }

//                    AddSample(normalized);

//                    if (isDateOnly)
//                    {
//                        regexDateOnly++;
//                        continue;
//                    }

//                    if (links.Add(normalized))
//                    {
//                        regexAdded++;
//                    }
//                    else
//                    {
//                        regexSkipped++;
//                    }
//                }
//            }

//            var anchorNodes = doc.DocumentNode.SelectNodes("//a[@href]");
//            var anchorNodeCount = anchorNodes?.Count ?? 0;
//            int anchorMatches = 0;
//            int anchorAdded = 0;
//            int anchorDateOnly = 0;
//            int anchorInvalid = 0;
//            int anchorSkipped = 0;

//            if (anchorNodes != null)
//            {
//                foreach (var node in anchorNodes)
//                {
//                    var href = node.GetAttributeValue("href", string.Empty);
//                    if (string.IsNullOrWhiteSpace(href)) continue;
//                    if (!href.Contains("/racing/results/", StringComparison.OrdinalIgnoreCase)) continue;
//                    if (!href.Contains(dateToken, StringComparison.OrdinalIgnoreCase)) continue;

//                    anchorMatches++;
//                    TryCollectAlternatePage(href);
//                    if (!TryNormalizeRaceUrl(href, out var normalized, out var isDateOnly))
//                    {
//                        anchorInvalid++;
//                        continue;
//                    }

//                    AddSample(normalized);

//                    if (isDateOnly)
//                    {
//                        anchorDateOnly++;
//                        continue;
//                    }

//                    if (links.Add(normalized))
//                    {
//                        anchorAdded++;
//                    }
//                    else
//                    {
//                        anchorSkipped++;
//                    }
//                }
//            }

//            int jsonMatches = 0;
//            int jsonAdded = 0;
//            int jsonInvalid = 0;
//            int jsonDateOnly = 0;
//            int jsonSkipped = 0;

//            foreach (var candidate in ExtractLinksFromEmbeddedJson(doc, date))
//            {
//                jsonMatches++;
//                TryCollectAlternatePage(candidate);

//                if (!TryNormalizeRaceUrl(candidate, out var normalized, out var isDateOnly))
//                {
//                    jsonInvalid++;
//                    continue;
//                }

//                AddSample(normalized);

//                if (isDateOnly)
//                {
//                    jsonDateOnly++;
//                    continue;
//                }

//                if (links.Add(normalized))
//                {
//                    jsonAdded++;
//                }
//                else
//                {
//                    jsonSkipped++;
//                }
//            }

//            if (links.Count == 0)
//            {
//                List<string> tabLabels = new();
//                void CollectTabText(string xpath)
//                {
//                    var nodes = doc.DocumentNode.SelectNodes(xpath);
//                    if (nodes == null) return;

//                    foreach (var node in nodes)
//                    {
//                        var text = Normalize(node.InnerText ?? string.Empty);
//                        if (!string.IsNullOrWhiteSpace(text))
//                        {
//                            tabLabels.Add(text);
//                        }
//                    }
//                }

//                CollectTabText("//*[@data-test-id='generic-tab']");
//                CollectTabText("//*[@data-test-id='results-tab']");
//                CollectTabText("//*[@data-test-id='meeting-tab']");
//                CollectTabText("//*[@data-test-id][contains(@data-test-id,'country')]");
//                CollectTabText("//*[@data-test-id][contains(@data-test-id,'tab')]");
//                CollectTabText("//*[contains(@class,'Tab')]");

//                var distinctTabs = tabLabels
//                    .Where(t => !string.IsNullOrWhiteSpace(t))
//                    .Distinct(StringComparer.OrdinalIgnoreCase)
//                    .ToList();

//                const int maxTabLog = 10;
//                if (distinctTabs.Count > maxTabLog)
//                {
//                    distinctTabs = distinctTabs.Take(maxTabLog - 1).Concat(new[] { "…" }).ToList();
//                }

//                var noMeetingNodes = doc.DocumentNode.SelectNodes("//*[@data-test-id][contains(@data-test-id,'no-meetings')]");
//                var meetingCards = doc.DocumentNode.SelectNodes("//*[@data-test-id][contains(@data-test-id,'meeting-card')]");
//                var raceContainers = doc.DocumentNode.SelectNodes("//*[@data-test-id='race-container']");

//                var truncatedSamples = sampleLinks
//                    .Select(s => s.Length > 120 ? s.Substring(0, 120) + "…" : s)
//                    .ToList();

//                var debugParts = new List<string>
//                {
//                    $"regexMatches={regexMatches}",
//                    $"regexAdded={regexAdded}",
//                    $"regexDateOnly={regexDateOnly}",
//                    $"regexInvalid={regexInvalid}",
//                    $"regexSkipped={regexSkipped}",
//                    $"anchorNodes={anchorNodeCount}",
//                    $"anchorMatches={anchorMatches}",
//                    $"anchorAdded={anchorAdded}",
//                    $"anchorDateOnly={anchorDateOnly}",
//                    $"anchorInvalid={anchorInvalid}",
//                    $"anchorSkipped={anchorSkipped}",
//                    $"jsonMatches={jsonMatches}",
//                    $"jsonAdded={jsonAdded}",
//                    $"jsonDateOnly={jsonDateOnly}",
//                    $"jsonInvalid={jsonInvalid}",
//                    $"jsonSkipped={jsonSkipped}",
//                    $"raceContainers={raceContainers?.Count ?? 0}",
//                    $"meetingCards={meetingCards?.Count ?? 0}",
//                    $"noMeetingsFlags={noMeetingNodes?.Count ?? 0}",
//                    $"tabs={(distinctTabs.Count > 0 ? "[" + string.Join(", ", distinctTabs) + "]" : "<none>")}",
//                    $"samples={(truncatedSamples.Count > 0 ? "[" + string.Join(", ", truncatedSamples) + "]" : "<none>")}",
//                     $"htmlLength={(rawHtml?.Length ?? 0)}",
//                    $"alternatePages={alternatePages.Count}",
//                    $"sawInternationalToggle={sawInternationalToggle}",
//                    $"pageUri={pageUri}"
//                };

//                Console.Error.WriteLine($"[Debug] {date:yyyy-MM-dd} ExtractRaceLinks: {string.Join(", ", debugParts)}");
//            }

//            var orderedLinks = links.OrderBy(l => l, StringComparer.OrdinalIgnoreCase).ToList();
//            var alternateUris = alternatePages
//                .Select(u =>
//                {
//                    try { return new Uri(u); }
//                    catch { return null; }
//                })
//                .Where(u => u != null)
//                .Cast<Uri>()
//                .ToList();

//            return new RaceLinkExtractionResult(orderedLinks, alternateUris, sawInternationalToggle);
//        }
//        private bool TryNormalizeRaceUrl(string rawUrl, out string normalized, out bool isDateOnly)
//        {
//            normalized = string.Empty;
//            isDateOnly = false;

//            if (string.IsNullOrWhiteSpace(rawUrl))
//            {
//                return false;
//            }

//            var candidate = rawUrl.Trim();

//            try
//            {
//                if (candidate.StartsWith("//", StringComparison.Ordinal))
//                {
//                    candidate = $"https:{candidate}";
//                }

//                var uri = candidate.StartsWith("http", StringComparison.OrdinalIgnoreCase)
//                    ? new Uri(candidate, UriKind.Absolute)
//                    : new Uri(BaseUri, candidate);

//                normalized = NormalizeUrl(uri.ToString());

//                var segments = uri.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
//                var resultsIndex = Array.IndexOf(segments, "results");
//                if (resultsIndex < 0)
//                {
//                    return false;
//                }

//                var remaining = segments.Length - (resultsIndex + 1);
//                if (remaining <= 0)
//                {
//                    isDateOnly = true;
//                    return true;
//                }

//                if (remaining == 1)
//                {
//                    var segment = segments[resultsIndex + 1];
//                    if (Regex.IsMatch(segment, @"^\d{4}-\d{2}-\d{2}$", RegexOptions.CultureInvariant))
//                    {
//                        isDateOnly = true;

//                        return true;
//                    }

//                    if (string.Equals(segment, "calendar", StringComparison.OrdinalIgnoreCase) ||
//                        string.Equals(segment, "fast-results", StringComparison.OrdinalIgnoreCase) ||
//                        string.Equals(segment, "today", StringComparison.OrdinalIgnoreCase) ||
//                        string.Equals(segment, "tomorrow", StringComparison.OrdinalIgnoreCase) ||
//                        string.Equals(segment, "yesterday", StringComparison.OrdinalIgnoreCase))
//                    {
//                        return false;
//                    }

//                    if (string.IsNullOrEmpty(uri.Query) && Regex.IsMatch(segment, @"^(?:today|yesterday|tomorrow)$", RegexOptions.IgnoreCase))
//                    {
//                        return false;
//                    }
//                }

//                return true;
//            }
//            catch (UriFormatException)
//            {
//                normalized = string.Empty;
//                return false;
//            }
//        }
//        private IEnumerable<string> ExtractLinksFromEmbeddedJson(HtmlDocument doc, DateTime date)
//        {
//            var seenNodes = new HashSet<HtmlNode>();
//            var yielded = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

//            string[] primarySelectors =
//            {
//                "//script[@id='__NEXT_DATA__']",
//                "//script[@id='__NUXT_DATA__']",
//                "//script[contains(@id,'__NEXT_DATA__')]",
//                "//script[contains(@id,'__NUXT_DATA__')]",
//                "//script[contains(@type,'application/json')]"
//            };

//            foreach (var selector in primarySelectors)
//            {
//                var nodes = doc.DocumentNode.SelectNodes(selector);
//                if (nodes == null) continue;

//                foreach (var node in nodes)
//                {
//                    if (!seenNodes.Add(node)) continue;

//                    foreach (var link in ExtractLinksFromScriptNode(node, date))
//                    {
//                        if (yielded.Add(link))
//                        {
//                            yield return link;
//                        }
//                    }
//                }
//            }

//            var otherScripts = doc.DocumentNode.SelectNodes("//script[not(@src)]");
//            if (otherScripts == null) yield break;

//            foreach (var node in otherScripts)
//            {
//                if (!seenNodes.Add(node)) continue;

//                foreach (var link in ExtractLinksFromScriptNode(node, date))
//                {
//                    if (yielded.Add(link))
//                    {
//                        yield return link;
//                    }
//                }
//            }
//        }

//        private IEnumerable<string> ExtractLinksFromScriptNode(HtmlNode node, DateTime date)
//        {
//            var raw = HtmlEntity.DeEntitize(node.InnerText ?? string.Empty);
//            if (string.IsNullOrWhiteSpace(raw)) yield break;

//            if (!raw.Contains("/racing/results/", StringComparison.OrdinalIgnoreCase)) yield break;

//            foreach (var link in ExtractLinksFromJsonText(raw, date))
//            {
//                yield return link;
//            }
//        }

//        private IEnumerable<string> ExtractLinksFromJsonText(string text, DateTime date)
//        {
//            if (string.IsNullOrWhiteSpace(text)) yield break;

//            foreach (var link in ExtractLinksFromJsonStructure(text, date))
//            {
//                yield return link;
//            }

//            var firstBrace = text.IndexOf('{');
//            var lastBrace = text.LastIndexOf('}');
//            if (firstBrace >= 0 && lastBrace > firstBrace)
//            {
//                var span = text.Substring(firstBrace, lastBrace - firstBrace + 1);
//                foreach (var link in ExtractLinksFromJsonStructure(span, date))
//                {
//                    yield return link;
//                }
//            }

//            foreach (Match match in Regex.Matches(text, @"((?:https?:)?//(?:www\\.)?sportinglife\\.com)?/racing/results/[^\""'#<\\s]+", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
//            {
//                yield return match.Value;
//            }
//        }
//        private IEnumerable<string> ExtractLinksFromJsonStructure(string json, DateTime date)
//        {
//            if (string.IsNullOrWhiteSpace(json)) yield break;

//            if (!TryParseJsonDocument(json, out var jsonDoc)) yield break;

//            using (jsonDoc)
//            {
//                foreach (var link in ExtractLinksFromJsonElement(jsonDoc.RootElement, date, false))
//                {
//                    yield return link;
//                }
//            }
//        }

//        private static bool TryParseJsonDocument(string json, out JsonDocument jsonDocument)
//        {
//            try
//            {
//                jsonDocument = JsonDocument.Parse(json);
//                return true;
//            }
//            catch (JsonException)
//            {
//                jsonDocument = null!;
//                return false;
//            }
//        }
//        private IEnumerable<string> ExtractLinksFromJsonElement(JsonElement element, DateTime date, bool dateContext)
//        {
//            switch (element.ValueKind)
//            {
//                case JsonValueKind.Object:
//                    {
//                        bool objectHasDate = dateContext;

//                        if (!objectHasDate)
//                        {
//                            foreach (var property in element.EnumerateObject())
//                            {
//                                if (property.Value.ValueKind == JsonValueKind.String && ContainsDateToken(property.Value.GetString(), date))
//                                {
//                                    objectHasDate = true;
//                                    break;
//                                }

//                                if (property.Value.ValueKind == JsonValueKind.Number && MatchesDateNumber(property.Value, date))
//                                {
//                                    objectHasDate = true;
//                                    break;
//                                }
//                            }
//                        }

//                        foreach (var property in element.EnumerateObject())
//                        {
//                            foreach (var link in ExtractLinksFromJsonElement(property.Value, date, objectHasDate))
//                            {
//                                yield return link;
//                            }
//                        }

//                        break;
//                    }

//                case JsonValueKind.Array:
//                    foreach (var item in element.EnumerateArray())
//                    {
//                        foreach (var link in ExtractLinksFromJsonElement(item, date, dateContext))
//                        {
//                            yield return link;
//                        }
//                    }

//                    break;

//                case JsonValueKind.String:
//                    {
//                        var value = element.GetString();
//                        if (string.IsNullOrWhiteSpace(value)) break;

//                        var hasDate = ContainsDateToken(value, date);

//                        if (value.Contains("/racing/results/", StringComparison.OrdinalIgnoreCase))
//                        {
//                            if (dateContext || hasDate)
//                            {
//                                yield return value;
//                            }
//                        }

//                        break;
//                    }
//            }
//        }

//        private static bool ContainsDateToken(string? value, DateTime date)
//        {
//            if (string.IsNullOrWhiteSpace(value)) return false;

//            if (value.IndexOf(date.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (value.IndexOf(date.ToString("yyyyMMdd", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (value.IndexOf(date.ToString("dd MMM yyyy", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (value.IndexOf(date.ToString("dd MMMM yyyy", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (value.IndexOf(date.ToString("ddd dd MMM yyyy", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (value.IndexOf(date.ToString("ddd dd MMMM yyyy", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (value.IndexOf(date.ToString("dd-MM-yyyy", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (value.IndexOf(date.ToString("dd/MM/yyyy", CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase) >= 0)
//            {
//                return true;
//            }

//            if (DateTime.TryParse(value, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var parsed))
//            {
//                if (parsed.Date == date.Date)
//                {
//                    return true;
//                }
//            }

//            return false;
//        }

//        private static bool MatchesDateNumber(JsonElement element, DateTime date)
//        {
//            if (!element.TryGetInt64(out var value))
//            {
//                return false;
//            }

//            try
//            {
//                if (value >= 1_000_000_000 && value <= 4_000_000_000)
//                {
//                    var seconds = DateTimeOffset.FromUnixTimeSeconds(value);
//                    if (seconds.Date == date.Date)
//                    {
//                        return true;
//                    }
//                }

//                if (value >= 1_000_000_000_000 && value <= 4_000_000_000_000)
//                {
//                    var milliseconds = DateTimeOffset.FromUnixTimeMilliseconds(value);
//                    if (milliseconds.Date == date.Date)
//                    {
//                        return true;
//                    }
//                }
//            }
//            catch
//            {
//                return false;
//            }

//            return false;
//        }
//        private async Task<(HtmlDocument Document, string RawHtml)> LoadDocumentAsync(Uri uri)
//        {
//            using var response = await _httpClient.GetAsync(uri);
//            response.EnsureSuccessStatusCode();
//            var html = await response.Content.ReadAsStringAsync();
//            var doc = new HtmlDocument();
//            doc.LoadHtml(html);
//            return (doc, html);
//        }


//        private static string NormalizeUrl(string url)
//        {
//            try
//            {
//                var builder = new UriBuilder(url)
//                {
//                    Fragment = string.Empty,
//                    Query = string.Empty
//                };
//                return builder.Uri.ToString().TrimEnd('/');
//            }
//            catch
//            {
//                return url;
//            }
//        }

//        private IEnumerable<RunnerResult> ParseRacePage(HtmlDocument doc, string rawHtml, DateTime defaultDate)
//        {
//            var rows = FindRunnerRows(doc);
//            if (rows.Count == 0)
//            {
//                return Array.Empty<RunnerResult>();
//            }

//            var headerText = HtmlText(doc.DocumentNode,
//                "//p[contains(@class,'CourseListingHeader__StyledMainTitle')]",
//                "//*[@data-test-id='course-title']",
//                "//*[@data-test-id='racecard-meeting-name']");
//            var courseName = ExtractCourseNameFromHeader(headerText);
//            if (string.IsNullOrWhiteSpace(courseName))
//            {
//                courseName = GuessCourseFromBreadcrumb(doc);
//            }

//            var dateText = HtmlText(doc.DocumentNode,
//                "//p[contains(@class,'CourseListingHeader__StyledMainSubTitle')]",
//                "//*[@data-test-id='course-subtitle']",
//                "//*[@data-test-id='racecard-subtitle']");
//            var raceDate = ParseDateSafe(dateText) ?? defaultDate;

//            var raceTitle = HtmlText(doc.DocumentNode,
//                "//h1[@data-test-id='racecard-race-name']",
//                "//h1[contains(@class,'RacingRacecardSummary__StyledTitle')]");

//            var metaLine = BuildMetaLine(doc.DocumentNode);

//            var status = HtmlText(doc.DocumentNode,
//                "//*[contains(@class,'RacingRacecardSummary__StyledEndState')]",
//                "//*[@data-test-id='racecard-end-state']");

//            string? ageRestriction = null;
//            string? distanceText = null;
//            string? going = null;
//            string? runners = null;
//            string? offTime = null;
//            string? winTime = null;
//            string? surface = null;
//            byte? classVal = null;

//            var meta = SplitMeta(metaLine);
//            if (meta.TryGetValue("age", out var ageVal)) ageRestriction = ageVal;
//            if (meta.TryGetValue("dist", out var distVal)) distanceText = distVal;
//            if (meta.TryGetValue("going", out var goingVal)) going = goingVal;
//            if (meta.TryGetValue("runners", out var runnersVal)) runners = runnersVal;
//            if (meta.TryGetValue("off", out var offVal)) offTime = offVal;
//            if (meta.TryGetValue("win", out var winVal)) winTime = winVal;
//            if (!classVal.HasValue) classVal = ExtractClass(metaLine);
//            if (string.IsNullOrWhiteSpace(surface)) surface = InferSurface(metaLine);

//            ApplyJsonLdMetadata(doc, ref ageRestriction, ref distanceText, ref going, ref runners, ref offTime, ref winTime, ref surface, ref classVal);

//            if (string.IsNullOrWhiteSpace(distanceText))
//            {
//                var m = Regex.Match(rawHtml, @"\b(\d+\s*m(?:\s*\d+\s*f)?(?:\s*\d+\s*y)?)\b", RegexOptions.IgnoreCase);
//                if (m.Success) distanceText = Normalize(m.Groups[1].Value);
//            }

//            if (string.IsNullOrWhiteSpace(going))
//            {
//                var m = Regex.Match(rawHtml, @"\b(Heavy|Soft|Good to Soft|Good|Good to Firm|Firm|Standard(?: to (?:Slow|Fast))?|Yielding)\b", RegexOptions.IgnoreCase);
//                if (m.Success) going = Normalize(m.Groups[1].Value);
//            }

//            if (string.IsNullOrWhiteSpace(offTime))
//            {
//                var m = Regex.Match(rawHtml, @"Off\s*time\s*[:\-]?\s*([0-2]?\d:[0-5]\d)", RegexOptions.IgnoreCase);
//                if (m.Success) offTime = Normalize(m.Groups[1].Value);
//            }

//            if (string.IsNullOrWhiteSpace(winTime))
//            {
//                var m = Regex.Match(rawHtml, @"Winning\s*time\s*[:\-]?\s*([0-9]+\s*m\s*[0-9.]+s|[0-9.]+s)", RegexOptions.IgnoreCase);
//                if (m.Success) winTime = Normalize(m.Groups[1].Value);
//            }

//            if (string.IsNullOrWhiteSpace(runners))
//            {
//                var m = Regex.Match(rawHtml, @"\b(\d+)\s+Runners\b", RegexOptions.IgnoreCase);
//                if (m.Success) runners = $"{m.Groups[1].Value} Runners";
//            }

//            if (!classVal.HasValue)
//            {
//                var m = Regex.Match(rawHtml, @"Class\s*(\d)", RegexOptions.IgnoreCase);
//                if (m.Success) classVal = byte.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
//            }

//            if (string.IsNullOrWhiteSpace(surface))
//            {
//                surface = InferSurface(rawHtml);
//            }

//            var runnerCount = TryParseOrdinalInt(runners?.Split(' ').FirstOrDefault());
//            var distanceYards = ParseDistanceToYards(distanceText);
//            var scheduledOff = ExtractScheduledOffFromHeader(headerText) ?? ParseClockTime(offTime);
//            var actualOff = ParseClockTime(offTime);
//            var winningMs = ParseWinningMs(winTime);

//            var courseId = _repo.InsertCourse(new Course { Name = courseName });
//            var raceEntity = new Race
//            {
//                CourseId = courseId,
//                RaceDate = raceDate,
//                ScheduledOff = scheduledOff ?? TimeSpan.Zero,
//                ActualOff = actualOff,
//                Title = raceTitle,
//                RaceType = InferRaceType(raceTitle),
//                Class = classVal,
//                AgeRestriction = ageRestriction,
//                Surface = surface,
//                Going = going,
//                DistanceYards = distanceYards,
//                DistanceText = distanceText ?? string.Empty,
//                RunnerCount = runnerCount.HasValue ? (byte?)runnerCount : null,
//                Status = status,
//                WinningTimeMs = winningMs,
//                WinningTimeText = winTime
//            };

//            var raceId = _repo.InsertRace(raceEntity);

//            var results = new List<RunnerResult>();
//            foreach (var row in rows)
//            {
//                var posRaw = HtmlText(row, ".//*[@data-test-id='position-no']", ".//*[contains(@class,'position-no')]");
//                if (string.IsNullOrWhiteSpace(posRaw))
//                {
//                    posRaw = HtmlTextByAttributeContains(row, "data-test-id", "position", "pos");
//                }
//                if (string.IsNullOrWhiteSpace(posRaw))
//                {
//                    posRaw = HtmlTextByAttributeContains(row, "class", "position", "pos");
//                }
//                var finishPos = TryParseOrdinalInt(posRaw);
//                var outcome = finishPos.HasValue ? string.Empty : ParseOutcomeCode(posRaw);

//                var cloth = HtmlText(row, ".//*[@data-test-id='saddle-cloth-no']");
//                if (string.IsNullOrWhiteSpace(cloth))
//                {
//                    cloth = HtmlTextByAttributeContains(row, "data-test-id", "saddle", "cloth", "number");
//                }
//                if (string.IsNullOrWhiteSpace(cloth))
//                {
//                    cloth = HtmlTextByAttributeContains(row, "class", "saddle", "cloth", "number");
//                }
//                var saddle = TryParseByte(cloth);

//                var drawRaw = HtmlText(row, ".//*[@data-test-id='stall-no']");
//                if (string.IsNullOrWhiteSpace(drawRaw))
//                {
//                    drawRaw = HtmlTextByAttributeContains(row, "data-test-id", "stall", "draw", "gate");
//                }
//                if (string.IsNullOrWhiteSpace(drawRaw))
//                {
//                    drawRaw = HtmlTextByAttributeContains(row, "class", "stall", "draw", "gate");
//                }
//                var stall = TryParseByteLoose(drawRaw);

//                var horseName = HtmlText(row, ".//a[contains(@href,'/racing/profiles/horse/')]");
//                if (string.IsNullOrWhiteSpace(horseName))
//                {
//                    horseName = HtmlText(row, ".//*[@data-test-id='horse-name']");
//                }
//                if (string.IsNullOrWhiteSpace(horseName))
//                {
//                    horseName = HtmlTextByAttributeContains(row, "data-test-id", "horse-name", "runner-name", "horse");
//                }
//                if (string.IsNullOrWhiteSpace(horseName))
//                {
//                    var fallbackLinks = row.SelectNodes(".//a[contains(@href,'/racing/')]");
//                    if (fallbackLinks != null)
//                    {
//                        foreach (var link in fallbackLinks)
//                        {
//                            var href = link.GetAttributeValue("href", string.Empty);
//                            if (href.IndexOf("trainer", StringComparison.OrdinalIgnoreCase) >= 0 ||
//                                href.IndexOf("jockey", StringComparison.OrdinalIgnoreCase) >= 0)
//                            {
//                                continue;
//                            }

//                            var text = Normalize(link.InnerText ?? string.Empty);
//                            if (!string.IsNullOrWhiteSpace(text))
//                            {
//                                horseName = text;
//                                break;
//                            }
//                        }
//                    }
//                }
//                if (string.IsNullOrWhiteSpace(horseName))
//                {
//                    continue;
//                }

//                var subInfo = row.SelectSingleNode(".//*[@data-test-id='horse-sub-info']")
//                    ?? row.SelectSingleNode(".//*[@data-test-id][contains(translate(@data-test-id,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'sub-info')]")
//                    ?? row.SelectSingleNode(".//*[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'sub-info')]");
//                string ageTxt = string.Empty;
//                string weightTxt = string.Empty;
//                if (subInfo != null)
//                {
//                    var spans = subInfo.SelectNodes(".//span");
//                    if (spans != null && spans.Count > 0)
//                    {
//                        ageTxt = Normalize(spans[0].InnerText);
//                        if (spans.Count > 1)
//                        {
//                            weightTxt = Normalize(spans[spans.Count - 1].InnerText);
//                        }
//                    }
//                }

//                var age = TryParseByte(ageTxt);

//                byte? weightLbs = null;
//                var weightMatch = Regex.Match(weightTxt ?? string.Empty, @"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$");
//                if (weightMatch.Success)
//                {
//                    var stones = int.Parse(weightMatch.Groups[1].Value, CultureInfo.InvariantCulture);
//                    var pounds = int.Parse(weightMatch.Groups[2].Value, CultureInfo.InvariantCulture);
//                    var total = stones * 14 + pounds;
//                    if (total >= byte.MinValue && total <= byte.MaxValue)
//                    {
//                        weightLbs = (byte)total;
//                    }
//                }

//                var (trainer, jockey) = ExtractTrainerJockey(row);

//                var spFrac = HtmlText(row, ".//span[contains(@class,'BetLinkStyle')]");
//                if (string.IsNullOrWhiteSpace(spFrac))
//                {
//                    spFrac = HtmlText(row, ".//*[@data-test-id='sp-odds']");
//                }
//                if (string.IsNullOrWhiteSpace(spFrac))
//                {
//                    spFrac = HtmlTextByAttributeContains(row, "data-test-id", "sp", "odds", "price");
//                }
//                if (string.IsNullOrWhiteSpace(spFrac))
//                {
//                    spFrac = HtmlTextByAttributeContains(row, "class", "sp", "odds", "price");
//                }
//                var spDec = FractionToDecimal(spFrac);
//                var favTag = ExtractFavouriteTag(spFrac);

//                var beatenTxt = HtmlText(row, ".//*[contains(@class,'StyledFinishDistance')]", ".//*[@data-test-id='finish-distance']");
//                if (string.IsNullOrWhiteSpace(beatenTxt))
//                {
//                    beatenTxt = HtmlTextByAttributeContains(row, "data-test-id", "distance", "beaten");
//                }
//                if (string.IsNullOrWhiteSpace(beatenTxt))
//                {
//                    beatenTxt = HtmlTextByAttributeContains(row, "class", "distance", "beaten");
//                }
//                var beatenLen = ParseBeatenLengths(beatenTxt);

//                var rowText = Normalize(row.InnerText ?? string.Empty);
//                var opFrac = ExtractOddsToken(rowText, "op");
//                var (tchLow, tchHigh) = ExtractTouchedTokens(rowText);

//                var comment = HtmlText(row, ".//*[@data-test-id='ride-description']", ".//*[contains(@class,'StyledRideDescription')]");
//                if (string.IsNullOrWhiteSpace(comment))
//                {
//                    comment = HtmlTextByAttributeContains(row, "data-test-id", "comment", "summary", "verdict", "analysis");
//                }
//                if (string.IsNullOrWhiteSpace(comment))
//                {
//                    comment = HtmlTextByAttributeContains(row, "class", "comment", "summary", "verdict", "analysis");
//                }
//                var trainerId = string.IsNullOrWhiteSpace(trainer) ? (int?)null : _repo.InsertTrainer(new Trainer { Name = trainer });
//                var jockeyId = string.IsNullOrWhiteSpace(jockey) ? (int?)null : _repo.InsertJockey(new Jockey { Name = jockey });
//                var horseId = _repo.InsertHorse(new Horse { Name = horseName });

//                var result = new RunnerResult
//                {
//                    RaceId = raceId,
//                    HorseId = horseId,
//                    TrainerId = trainerId,
//                    JockeyId = jockeyId,
//                    SaddleclothNumber = saddle,
//                    Draw = stall,
//                    Age = age,
//                    WeightLbs = weightLbs,
//                    WeightText = weightTxt,
//                    FinishPos = finishPos.HasValue ? (short?)finishPos.Value : null,
//                    OutcomeCode = outcome,
//                    DistanceBeatenText = beatenTxt,
//                    DistanceBeatenLengths = beatenLen,
//                    SP_Fraction = spFrac,
//                    SP_Decimal = spDec,
//                    FavTag = favTag,
//                    OpeningFraction = opFrac,
//                    TouchedHighFraction = tchHigh,
//                    TouchedLowFraction = tchLow,
//                    Comment = comment
//                };

//                results.Add(result);
//            }

//            return results;
//        }
//        private static List<HtmlNode> FindRunnerRows(HtmlDocument doc)
//        {
//            var result = new List<HtmlNode>();
//            var seen = new HashSet<HtmlNode>();

//            string[] selectors =
//            {
//                "//*[contains(@class,'ResultRunner__StyledResultRunnerWrapper')]",
//                "//*[@data-test-id='result-runner']",
//                "//*[@data-test-id='race-result-runner']",
//                "//*[@data-test-id='racecard-result-runner']",
//                "//*[@data-test-id='result-runner-row']",
//                "//*[@data-test-id='full-result-runner']",
//                "//*[@data-test-id='result-runner-item']",
//                "//*[contains(@data-test-id,'result-runner')]",
//                "//*[contains(@class,'ResultRunner')]",
//                "//*[contains(@class,'result-runner')]",
//                "//*[contains(@class,'RaceResultRunner')]",
//                "//*[contains(@data-test-id,'runner-row')]",
//                "//*[contains(@data-test-id,'result-row')]",
//                "//*[contains(@class,'result-row')]"
//            };

//            foreach (var selector in selectors)
//            {
//                HtmlNodeCollection? nodes = null;
//                try
//                {
//                    nodes = doc.DocumentNode.SelectNodes(selector);
//                }
//                catch (XPathException)
//                {
//                    continue;
//                }

//                if (nodes == null) continue;

//                foreach (var node in nodes)
//                {
//                    if (node == null) continue;

//                    if (result.Any(existing => existing != null && !ReferenceEquals(existing, node) && existing.Ancestors().Contains(node)))
//                    {
//                        continue;
//                    }

//                    if (!seen.Add(node))
//                    {
//                        continue;
//                    }

//                    result.Add(node);
//                }
//            }

//            if (result.Count == 0)
//            {
//                var tableRows = doc.DocumentNode.SelectNodes("//tr[.//a[contains(@href,'/racing/profiles/horse/')]]");
//                if (tableRows != null)
//                {
//                    foreach (var row in tableRows)
//                    {
//                        if (row == null) continue;
//                        if (!seen.Add(row)) continue;
//                        result.Add(row);
//                    }
//                }
//            }

//            if (result.Count > 1)
//            {
//                result = result
//                    .Where(node => node != null && !result.Any(other => other != null && !ReferenceEquals(other, node) && other.Ancestors().Contains(node)))
//                    .ToList();
//            }

//            result = result
//                .Where(node => node != null && NodeContainsHorseCandidate(node))
//                .ToList();

//            return result;
//        }

//        private static bool NodeContainsHorseCandidate(HtmlNode node)
//        {
//            if (node.SelectSingleNode(".//a[contains(@href,'/racing/profiles/horse/')]") != null)
//            {
//                return true;
//            }

//            foreach (var descendant in node.Descendants())
//            {
//                var dataId = descendant.GetAttributeValue("data-test-id", string.Empty);
//                if (!string.IsNullOrWhiteSpace(dataId) &&
//                    (dataId.IndexOf("horse", StringComparison.OrdinalIgnoreCase) >= 0 ||
//                     dataId.IndexOf("runner", StringComparison.OrdinalIgnoreCase) >= 0))
//                {
//                    return true;
//                }

//                var classAttr = descendant.GetAttributeValue("class", string.Empty);
//                if (!string.IsNullOrWhiteSpace(classAttr) &&
//                    (classAttr.IndexOf("horse", StringComparison.OrdinalIgnoreCase) >= 0 ||
//                     classAttr.IndexOf("runner", StringComparison.OrdinalIgnoreCase) >= 0))
//                {
//                    return true;
//                }
//            }

//            return false;
//        }
//        private static string HtmlTextByAttributeContains(HtmlNode node, string attributeName, params string[] tokens)
//        {
//            if (tokens == null || tokens.Length == 0) return string.Empty;

//            foreach (var current in EnumerateNodeAndDescendants(node))
//            {
//                var attr = current.GetAttributeValue(attributeName, string.Empty);
//                if (string.IsNullOrWhiteSpace(attr)) continue;

//                foreach (var token in tokens)
//                {
//                    if (attr.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0)
//                    {
//                        var text = Normalize(current.InnerText ?? string.Empty);
//                        if (!string.IsNullOrWhiteSpace(text))
//                        {
//                            return text;
//                        }
//                    }
//                }
//            }

//            return string.Empty;
//        }

//        private static IEnumerable<HtmlNode> EnumerateNodeAndDescendants(HtmlNode node)
//        {
//            yield return node;

//            foreach (var child in node.ChildNodes)
//            {
//                foreach (var descendant in EnumerateNodeAndDescendants(child))
//                {
//                    yield return descendant;
//                }
//            }
//        }

//        private static string HtmlText(HtmlNode node, params string[] xpaths)
//        {
//            foreach (var xpath in xpaths)
//            {
//                var selected = node.SelectSingleNode(xpath);
//                if (selected == null) continue;
//                var text = Normalize(selected.InnerText ?? string.Empty);
//                if (!string.IsNullOrWhiteSpace(text))
//                {
//                    return text;
//                }
//            }

//            return string.Empty;
//        }

//        private static string BuildMetaLine(HtmlNode root)
//        {
//            var metaNodes = root.SelectNodes("//*[contains(@class,'RacingRacecardSummary__StyledAdditionalInfo')]" +
//                                             "|//*[@data-test-id='racecard-additional-info']" +
//                                             "|//*[@data-test-id='race-summary']" +
//                                             "|//*[@data-test-id='result-additional-info']");
//            if (metaNodes == null || metaNodes.Count == 0)
//            {
//                return string.Empty;
//            }

//            var parts = new List<string>();
//            foreach (var node in metaNodes)
//            {
//                var text = Normalize(node.InnerText ?? string.Empty);
//                if (!string.IsNullOrWhiteSpace(text))
//                {
//                    parts.Add(text.Replace("•", "|").Replace("·", "|"));
//                }
//            }

//            return Normalize(string.Join("|", parts));
//        }

//        private static void ApplyJsonLdMetadata(HtmlDocument doc,
//            ref string? ageRestriction,
//            ref string? distanceText,
//            ref string? going,
//            ref string? runners,
//            ref string? offTime,
//            ref string? winTime,
//            ref string? surface,
//            ref byte? classVal)
//        {
//            var scripts = doc.DocumentNode.SelectNodes("//script[@type='application/ld+json']");
//            if (scripts == null) return;

//            foreach (var script in scripts)
//            {
//                var json = script.InnerText;
//                if (string.IsNullOrWhiteSpace(json)) continue;

//                try
//                {
//                    using var document = JsonDocument.Parse(json);
//                    ApplyJsonElement(document.RootElement, ref ageRestriction, ref distanceText, ref going, ref runners, ref offTime, ref winTime, ref surface, ref classVal);
//                }
//                catch
//                {
//                    // ignore invalid JSON blocks
//                }
//            }
//        }

//        private static void ApplyJsonElement(JsonElement element,
//            ref string? ageRestriction,
//            ref string? distanceText,
//            ref string? going,
//            ref string? runners,
//            ref string? offTime,
//            ref string? winTime,
//            ref string? surface,
//            ref byte? classVal)
//        {
//            switch (element.ValueKind)
//            {
//                case JsonValueKind.Array:
//                    foreach (var item in element.EnumerateArray())
//                    {
//                        ApplyJsonElement(item, ref ageRestriction, ref distanceText, ref going, ref runners, ref offTime, ref winTime, ref surface, ref classVal);
//                    }
//                    break;
//                case JsonValueKind.Object:
//                    if (element.TryGetProperty("ageRestriction", out var ageProp) && string.IsNullOrWhiteSpace(ageRestriction))
//                    {
//                        ageRestriction = Normalize(ageProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("distance", out var distProp) && string.IsNullOrWhiteSpace(distanceText))
//                    {
//                        distanceText = Normalize(distProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("raceDistance", out var distAlt) && string.IsNullOrWhiteSpace(distanceText))
//                    {
//                        distanceText = Normalize(distAlt.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("distanceName", out var distName) && string.IsNullOrWhiteSpace(distanceText))
//                    {
//                        distanceText = Normalize(distName.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("going", out var goingProp) && string.IsNullOrWhiteSpace(going))
//                    {
//                        going = Normalize(goingProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("ground", out var groundProp) && string.IsNullOrWhiteSpace(going))
//                    {
//                        going = Normalize(groundProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("runners", out var runnersProp) && string.IsNullOrWhiteSpace(runners))
//                    {
//                        runners = Normalize(runnersProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("numberOfRunners", out var runnerCountProp) && string.IsNullOrWhiteSpace(runners))
//                    {
//                        if (runnerCountProp.ValueKind == JsonValueKind.Number && runnerCountProp.TryGetInt32(out var count))
//                        {
//                            runners = $"{count} Runners";
//                        }
//                        else if (runnerCountProp.ValueKind == JsonValueKind.String)
//                        {
//                            runners = Normalize(runnerCountProp.GetString() ?? string.Empty);
//                        }
//                    }

//                    if (element.TryGetProperty("offTime", out var offProp) && string.IsNullOrWhiteSpace(offTime))
//                    {
//                        offTime = Normalize(offProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("startTime", out var startProp) && string.IsNullOrWhiteSpace(offTime))
//                    {
//                        offTime = Normalize(startProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("winningTime", out var winProp) && string.IsNullOrWhiteSpace(winTime))
//                    {
//                        winTime = Normalize(winProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("time", out var timeProp) && string.IsNullOrWhiteSpace(winTime))
//                    {
//                        winTime = Normalize(timeProp.GetString() ?? string.Empty);
//                    }

//                    if (element.TryGetProperty("surface", out var surfaceProp) && string.IsNullOrWhiteSpace(surface))
//                    {
//                        surface = Normalize(surfaceProp.GetString() ?? string.Empty);
//                    }

//                    if (!classVal.HasValue)
//                    {
//                        if (element.TryGetProperty("raceClass", out var classProp))
//                        {
//                            var text = Normalize(classProp.GetString() ?? string.Empty);
//                            var match = Regex.Match(text, @"(\d)");
//                            if (match.Success)
//                            {
//                                classVal = byte.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
//                            }
//                        }
//                        else if (element.TryGetProperty("class", out var classAltProp))
//                        {
//                            var text = Normalize(classAltProp.GetString() ?? string.Empty);
//                            var match = Regex.Match(text, @"(\d)");
//                            if (match.Success)
//                            {
//                                classVal = byte.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
//                            }
//                        }
//                    }

//                    if (element.TryGetProperty("@graph", out var graphProp))
//                    {
//                        ApplyJsonElement(graphProp, ref ageRestriction, ref distanceText, ref going, ref runners, ref offTime, ref winTime, ref surface, ref classVal);
//                    }
//                    break;
//            }
//        }

//        private static (string trainer, string jockey) ExtractTrainerJockey(HtmlNode row)
//        {
//            string trainer = string.Empty;
//            string jockey = string.Empty;

//            var nameSpans = row.SelectNodes(".//*[contains(@class,'StyledTrainerJockeyBlock')]//span[contains(@class,'StyledPersonName')]");
//            if (nameSpans != null && nameSpans.Count > 0)
//            {
//                trainer = Normalize(nameSpans[0].InnerText ?? string.Empty);
//                if (nameSpans.Count > 1)
//                {
//                    jockey = Normalize(nameSpans[1].InnerText ?? string.Empty);
//                }
//            }
//            if (string.IsNullOrWhiteSpace(trainer))
//            {
//                trainer = HtmlTextByAttributeContains(row, "data-test-id", "trainer");
//            }

//            if (string.IsNullOrWhiteSpace(jockey))
//            {
//                jockey = HtmlTextByAttributeContains(row, "data-test-id", "jockey");
//            }

//            if (string.IsNullOrWhiteSpace(trainer))
//            {
//                trainer = HtmlTextByAttributeContains(row, "class", "trainer");
//            }

//            if (string.IsNullOrWhiteSpace(jockey))
//            {
//                jockey = HtmlTextByAttributeContains(row, "class", "jockey");
//            }

//            if (string.IsNullOrWhiteSpace(trainer) || string.IsNullOrWhiteSpace(jockey))
//            {
//                var block = row.SelectSingleNode(".//*[contains(@class,'StyledTrainerJockeyBlock')]");
//                var text = block != null ? Normalize(block.InnerText ?? string.Empty) : string.Empty;
//                if (string.IsNullOrWhiteSpace(text))
//                {
//                    text = Normalize(row.InnerText ?? string.Empty);
//                }

//                if (string.IsNullOrWhiteSpace(trainer))
//                {
//                    var m = Regex.Match(text, @"(?:^|\s)T:\s*([A-Za-z .&'\-]+?)(?=\s+J:|$)", RegexOptions.IgnoreCase);
//                    if (m.Success) trainer = m.Groups[1].Value.Trim();
//                }

//                if (string.IsNullOrWhiteSpace(jockey))
//                {
//                    var m = Regex.Match(text, @"(?:^|\s)J:\s*([A-Za-z .&'\-]+)$", RegexOptions.IgnoreCase);
//                    if (m.Success) jockey = m.Groups[1].Value.Trim();
//                }
//            }

//            return (trainer, jockey);
//        }

//        private static string GuessCourseFromBreadcrumb(HtmlDocument doc)
//        {
//            var crumb = doc.DocumentNode.SelectSingleNode("//nav[contains(@class,'Breadcrumb')]//a[last()]");
//            if (crumb != null)
//            {
//                var text = Normalize(crumb.InnerText ?? string.Empty);
//                if (!string.IsNullOrWhiteSpace(text))
//                {
//                    return text;
//                }
//            }

//            var titles = doc.DocumentNode.SelectNodes("//p[contains(@class,'CourseListingHeader__StyledMainTitle')]");
//            if (titles != null)
//            {
//                foreach (var title in titles)
//                {
//                    var text = Normalize(title.InnerText ?? string.Empty);
//                    var tokens = text.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
//                    if (tokens.Length == 2)
//                    {
//                        return tokens[1].Trim();
//                    }
//                }
//            }

//            return string.Empty;
//        }

//        private static decimal? FractionToDecimal(string frac)
//        {
//            if (string.IsNullOrWhiteSpace(frac)) return null;
//            frac = frac.Trim().ToLowerInvariant();
//            frac = frac.Replace("jf", string.Empty).Replace("cf", string.Empty).Replace("f", string.Empty).Trim();
//            var match = Regex.Match(frac, @"(\d+)\s*/\s*(\d+)");
//            if (!match.Success) return null;
//            var a = decimal.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
//            var b = decimal.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture);
//            return Math.Round(1 + (a / b), 3);
//        }

//        private static string ExtractFavouriteTag(string frac)
//        {
//            if (string.IsNullOrWhiteSpace(frac)) return null;
//            frac = frac.ToUpperInvariant();
//            if (frac.Contains("JF")) return "JF";
//            if (frac.Contains("CF")) return "CF";
//            if (frac.EndsWith("F")) return "F";
//            return null;
//        }

//        private static decimal? ParseBeatenLengths(string s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return null;
//            s = s.Trim().ToLowerInvariant();
//            return s switch
//            {
//                "nk" or "neck" => 0.3m,
//                "hd" or "head" => 0.2m,
//                "shd" or "short head" or "shorthead" or "s.h" or "sh" => 0.1m,
//                "nse" or "nose" => 0.05m,
//                _ => TryParseDecimalLength(s)
//            };
//        }

//        private static decimal? TryParseDecimalLength(string s)
//        {
//            s = s.Replace("¾", ".75").Replace("½", ".5").Replace("¼", ".25");
//            return decimal.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var value) ? value : null;
//        }

//        private static string ExtractOddsToken(string s, string key)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return null;
//            var match = Regex.Match(s, $@"\b{Regex.Escape(key)}\s+(\d+/\d+\w*)", RegexOptions.IgnoreCase);
//            return match.Success ? match.Groups[1].Value : null;
//        }

//        private static (string? low, string? high) ExtractTouchedTokens(string s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return (null, null);
//            var lowMatch = Regex.Match(s, @"(tchd|low)\s+(\d+/\d+\w*)", RegexOptions.IgnoreCase);
//            var highMatch = Regex.Match(s, @"high\s+(\d+/\d+\w*)", RegexOptions.IgnoreCase);
//            string? low = lowMatch.Success ? lowMatch.Groups[2].Value : null;
//            string? high = highMatch.Success ? highMatch.Groups[1].Value : null;
//            return (low, high);
//        }

//        private static byte? TryParseByte(string? s) => byte.TryParse(s?.Trim(), out var value) ? value : null;

//        private static byte? TryParseByteLoose(string? s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return null;
//            var match = Regex.Match(s, @"\d+");
//            return match.Success && byte.TryParse(match.Value, out var value) ? value : (byte?)null;
//        }

//        private static DateTime? ParseDateSafe(string? s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return null;
//            return DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.AllowWhiteSpaces, out var date) ? date.Date : null;
//        }

//        private static int? TryParseOrdinalInt(string? s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return null;
//            s = s.Trim();
//            var match = Regex.Match(s, @"^(\d+)\s*(st|nd|rd|th)?$", RegexOptions.IgnoreCase);
//            if (match.Success && int.TryParse(match.Groups[1].Value, out var value)) return value;
//            return int.TryParse(s, out var direct) ? direct : (int?)null;
//        }

//        private static string ExtractCourseNameFromHeader(string header)
//        {
//            if (string.IsNullOrWhiteSpace(header)) return string.Empty;
//            var tokens = header.Trim().Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
//            return tokens.Length < 2 ? string.Empty : tokens[1].Trim();
//        }

//        private static TimeSpan? ExtractScheduledOffFromHeader(string header)
//        {
//            if (string.IsNullOrWhiteSpace(header)) return null;
//            var timeToken = header.Split(' ', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault();
//            return ParseClockTime(timeToken);
//        }

//        private static string Normalize(string s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return string.Empty;
//            s = HtmlEntity.DeEntitize(s);
//            s = s.Replace('\u00A0', ' ');
//            s = Regex.Replace(s, @"\s+", " ");
//            return s.Trim();
//        }

//        private static Dictionary<string, string> SplitMeta(string meta)
//        {
//            var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
//            if (string.IsNullOrWhiteSpace(meta)) return result;

//            meta = Normalize(meta);
//            var parts = meta.Split('|', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
//            foreach (var part in parts)
//            {
//                var token = part.Trim();
//                if (token.Contains("YO", StringComparison.OrdinalIgnoreCase)) result["age"] = token;
//                else if (token.Contains("Runners", StringComparison.OrdinalIgnoreCase)) result["runners"] = token;
//                else if (token.StartsWith("Off time", StringComparison.OrdinalIgnoreCase)) result["off"] = token.Split(':', 2)[1].Trim();
//                else if (token.StartsWith("Winning time", StringComparison.OrdinalIgnoreCase)) result["win"] = token.Split(':', 2)[1].Trim();
//                else if (IsGoingToken(token)) result["going"] = token;
//                else if (HasDistanceToken(token)) result["dist"] = token;
//            }

//            return result;
//        }

//        private static bool IsGoingToken(string s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return false;
//            var keys = new[] { "Good", "Firm", "Soft", "Heavy", "Standard", "Yielding" };
//            return keys.Any(k => s.Contains(k, StringComparison.OrdinalIgnoreCase));
//        }

//        private static bool HasDistanceToken(string s)
//        {
//            if (string.IsNullOrWhiteSpace(s)) return false;
//            return Regex.IsMatch(s.ToLowerInvariant(), @"\b\d+\s*(miles?|m|furlongs?|f|yards?|y)\b");
//        }

//        private static int ParseDistanceToYards(string? text)
//        {
//            if (string.IsNullOrWhiteSpace(text)) return 0;
//            var yards = 0;
//            foreach (var token in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
//            {
//                if (token.EndsWith("m", StringComparison.OrdinalIgnoreCase) && int.TryParse(token.TrimEnd('m', 'M'), out var miles)) yards += miles * 1760;
//                else if (token.EndsWith("f", StringComparison.OrdinalIgnoreCase) && int.TryParse(token.TrimEnd('f', 'F'), out var furlongs)) yards += furlongs * 220;
//                else if (token.EndsWith("y", StringComparison.OrdinalIgnoreCase) && int.TryParse(token.TrimEnd('y', 'Y'), out var yardsVal)) yards += yardsVal;
//            }
//            return yards;
//        }

//        private static TimeSpan? ParseClockTime(string? t)
//        {
//            if (string.IsNullOrWhiteSpace(t)) return null;
//            return TimeSpan.TryParse(t, CultureInfo.InvariantCulture, out var ts) ? ts : null;
//        }

//        private static int? ParseWinningMs(string? t)
//        {
//            if (string.IsNullOrWhiteSpace(t) || t == "-") return null;
//            try
//            {
//                t = t.Trim();
//                double seconds;
//                if (t.Contains('m'))
//                {
//                    var parts = t.Replace("s", string.Empty).Split('m', StringSplitOptions.TrimEntries);
//                    var mins = int.Parse(parts[0], CultureInfo.InvariantCulture);
//                    var secs = double.Parse(parts[1], CultureInfo.InvariantCulture);
//                    seconds = mins * 60 + secs;
//                }
//                else
//                {
//                    seconds = double.Parse(t.TrimEnd('s'), CultureInfo.InvariantCulture);
//                }

//                return (int)Math.Round(seconds * 1000.0);
//            }
//            catch
//            {
//                return null;
//            }
//        }

//        private static string ParseOutcomeCode(string posText)
//        {
//            if (string.IsNullOrWhiteSpace(posText)) return string.Empty;
//            var letters = new string(posText.Trim().ToUpperInvariant().Where(char.IsLetter).ToArray());
//            if (letters.Length == 0) return string.Empty;
//            return letters;
//        }

//        private static string InferRaceType(string title)
//        {
//            if (string.IsNullOrWhiteSpace(title)) return string.Empty;
//            var keys = new[] { "Handicap", "Maiden", "Novice", "Apprentice", "Claiming", "Selling", "Stakes" };
//            return keys.FirstOrDefault(k => title.IndexOf(k, StringComparison.OrdinalIgnoreCase) >= 0) ?? string.Empty;
//        }

//        private static byte? ExtractClass(string meta)
//        {
//            if (string.IsNullOrWhiteSpace(meta)) return null;
//            var match = Regex.Match(meta, @"Class\s+(\d)");
//            return match.Success ? (byte?)byte.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) : null;
//        }

//        private static string InferSurface(string meta)
//        {
//            if (string.IsNullOrWhiteSpace(meta)) return null;
//            if (meta.Contains("All Weather", StringComparison.OrdinalIgnoreCase) || meta.Contains("Allweather", StringComparison.OrdinalIgnoreCase))
//            {
//                return "Allweather";
//            }
//            return "Turf";
//        }
//    }
//}