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

namespace HorseRacingML.Scraping
{
    public class RaceResultsScraper
    {
        private readonly RacingRepository _repo;
        private static readonly DateTime HardcodedToday = new DateTime(2025, 9, 14);
        public RaceResultsScraper(RacingRepository repo)
        {
            _repo = repo;
        }
        private static int GetFreeTcpPort()
        {
            using var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            int port = ((IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();
            return port;
        }
        private void ScrapeMeetingTabs(IWebDriver driver, WebDriverWait wait, DateTime raceDate, string dayHandle)
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
                            ParseRacePage(driver, wait, raceDate);
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
        public void Scrape(DateTime startDate, DateTime endDate)
        {
            var start = startDate.Date;
            var end = HardcodedToday.Date;
            if (start > end) start = end;

            var dates = Enumerable.Range(0, (end - start).Days + 1)
                                   .Select(i => end.AddDays(-i));
            var queue = new ConcurrentQueue<DateTime>(dates);
            var tasks = new List<Task>();
            int workers = Math.Min(6, queue.Count);

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
                        Console.WriteLine($"[{date:yyyy-MM-dd}] parsed and inserted");
                        try { _ = driver.WindowHandles.Count; }
                        catch (Exception ex) { Console.Error.WriteLine($"[Warning] Driver session not healthy before next day: {ex.Message}"); break; }
                    }
                }));
            }

            Task.WaitAll(tasks.ToArray());
        }
        private void ScrapeDay(IWebDriver driver, DateTime date)
        {
            try
            {
                var url = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}"; driver.Navigate().GoToUrl(url); // go to date page
                AcceptTermsIfPresent(driver); // cookies
                var dayHandle = driver.CurrentWindowHandle; // remember the top-level window for this day
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
                            ScrapeMeetingTabs(driver, new WebDriverWait(driver, TimeSpan.FromSeconds(6)), date, dayHandle); anyMeetingsProcessed = true; // scrape meetings for this region
                        }
                        catch (WebDriverException ex) { Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Region '{region}' error: {ex.Message}"); }
                        catch (Exception ex) { Console.Error.WriteLine($"[{date:yyyy-MM-dd}] Unexpected region error '{region}': {ex.Message}"); }
                    }
                    if (!anyMeetingsProcessed) { ScrapeMeetingTabs(driver, new WebDriverWait(driver, TimeSpan.FromSeconds(12)), date, dayHandle); } // fallback if toggle failed
                }
                else { ScrapeMeetingTabs(driver, new WebDriverWait(driver, TimeSpan.FromSeconds(12)), date, dayHandle); } // no toggle present, scrape directly
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
        private void ParseRacePage(IWebDriver driver, WebDriverWait wait, DateTime defaultDate)
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

            if (results.Count > 0) _repo.InsertRunnerResults(results); // batch insert
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
                var t = p.Trim();

                if (t.Contains("YO", StringComparison.OrdinalIgnoreCase)) r["age"] = t; // age
                else if (t.Contains("Runners", StringComparison.OrdinalIgnoreCase)) r["runners"] = t; // runners
                else if (t.StartsWith("Off time", StringComparison.OrdinalIgnoreCase)) r["off"] = t.Split(':', 2)[1].Trim(); // off time
                else if (t.StartsWith("Winning time", StringComparison.OrdinalIgnoreCase)) r["win"] = t.Split(':', 2)[1].Trim(); // win time
                else if (IsGoingToken(t)) r["going"] = t; // going
                else if (HasDistanceToken(t)) r["dist"] = t; // distance
            }

            return r;
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

        private static int? ParseWinningMs(string? t)
        {
            if (string.IsNullOrWhiteSpace(t) || t == "-") return null; // "1m 15.11s" or "59.87s"
            try
            {
                t = t.Trim(); double seconds = 0;
                if (t.Contains('m')) { var parts = t.Replace("s", "").Split('m', StringSplitOptions.TrimEntries); var mins = int.Parse(parts[0]); var secs = double.Parse(parts[1], CultureInfo.InvariantCulture); seconds = mins * 60 + secs; }
                else { seconds = double.Parse(t.TrimEnd('s'), CultureInfo.InvariantCulture); }
                return (int)Math.Round(seconds * 1000.0);
            }
            catch { return null; }
        }

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