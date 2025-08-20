using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Interactions;
using OpenQA.Selenium.Support.UI;
using System.Globalization;

namespace HorseRacingML.Scraping
    {
        public class RaceResultsScraper
        {
            public void Scrape(DateTime startDate, DateTime endDate)
            {
                var options = new ChromeOptions(); options.AddArgument("--start-maximized"); //visible
                using var driver = new ChromeDriver(options); var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10)); //SPA waits
                for (var date = startDate; date <= endDate; date = date.AddDays(1))
                {
                    try
                    {
                        var url = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}"; driver.Navigate().GoToUrl(url); //go date
                        AcceptTermsIfPresent(driver); //cookies
                        string[] regions = { "UK & Ireland", "International" }; //toggle regions if present
                        foreach (var region in regions)
                        {
                            try
                            {
                                var regionBtn = driver.FindElements(By.XPath($"//*[self::button or self::span][contains(normalize-space(.), '{region}') and ancestor::*[@data-test-id='new-switch-button']]")).FirstOrDefault(e => e.Displayed && e.Enabled); //find region pill
                                if (regionBtn == null) { continue; } //not present for this day
                                ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", regionBtn); //robust click
                                wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0); //meeting tabs loaded
                                ScrapeMeetingTabs(driver, wait, date); //meetings under this region
                            }
                            catch (WebDriverException ex) { Console.WriteLine($"Region '{region}' error on {date:yyyy-MM-dd}: {ex.Message}"); } //log and move
                        }
                        if (driver.FindElements(By.CssSelector("[data-test-id='new-switch-button']")).Count == 0) { ScrapeMeetingTabs(driver, wait, date); } //pages without regions
                    }
                    catch (WebDriverException ex) { Console.WriteLine($"Skipping {date:yyyy-MM-dd}: {ex.Message}"); } //continue next day
                }
            }
            private static void ScrapeMeetingTabs(IWebDriver driver, WebDriverWait wait, DateTime raceDate)
            {
                try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0); } catch { Console.WriteLine("Meeting tab elements not found."); return; } //ensure tabs exist
                int tabIndex = 0; while (true)
                {
                    var tabs = driver.FindElements(By.CssSelector("[data-test-id='generic-tab']")); if (tabIndex >= tabs.Count) break; //done
                    var tab = tabs[tabIndex]; try { ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", tab); ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", tab); } catch { tabIndex++; continue; } //click meeting
                    try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id='race-container']")).Count > 0); } catch { tabIndex++; continue; } //wait races
                var raceLinks = driver.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]"));
                var mainHandle = driver.CurrentWindowHandle; //remember listing tab

                // open all races first so they can load in parallel
                var existingHandles = driver.WindowHandles.ToList();
                foreach (var a in raceLinks)
                {
                    var href = a.GetAttribute("href");
                    if (string.IsNullOrWhiteSpace(href)) continue; //skip
                    ((IJavaScriptExecutor)driver).ExecuteScript("window.open(arguments[0], '_blank');", href); //open race
                    }
                // collect newly opened race tabs
                var newHandles = driver.WindowHandles.Except(existingHandles).ToList();

                // now parse each race tab and close it
                foreach (var handle in newHandles)
                {
                    driver.SwitchTo().Window(handle); //focus race tab
                    try { ParseRacePage(driver, wait, raceDate); } catch (Exception ex) { Console.WriteLine($"Parse error: {ex.Message}"); }
                    driver.Close();
                }

                driver.SwitchTo().Window(mainHandle); //back to meeting listing
                tabIndex++; //next meeting
                }
            }
            private static void ParseRacePage(IWebDriver driver, WebDriverWait wait, DateTime defaultDate)
            {
                //Course + title + meta (from header) ----------------------------------------------------------
                string course = TextOrEmpty(driver, By.CssSelector("p.CourseListingHeader__StyledMainTitle")); //e.g. "14:10 Beverley"
                if (string.IsNullOrWhiteSpace(course)) course = TextOrEmpty(driver, By.CssSelector("h1[data-test-id='racecard-race-name']")); //fallback
                string courseName = ExtractCourseNameFromHeader(TextOrEmpty(driver, By.CssSelector("p.CourseListingHeader__StyledMainTitle"))); //from "14:10 Beverley"
                if (string.IsNullOrEmpty(courseName)) courseName = GuessCourseFromBreadcrumb(driver); //last resort
                string dateText = TextOrEmpty(driver, By.CssSelector("p.CourseListingHeader__StyledMainSubTitle")); //e.g. "Wednesday 13 April 2005"
                DateTime raceDate = ParseDateSafe(dateText) ?? defaultDate; //page date or fallback
                string raceTitle = TextOrEmpty(driver, By.CssSelector("h1[data-test-id='racecard-race-name']")); //race title
                string addInfo = TextOrEmpty(driver, By.CssSelector("li.RacingRacecardSummary__StyledAdditionalInfo")); //line with trip/going/runners/etc (joined)
                string status = TextOrEmpty(driver, By.CssSelector(".RacingRacecardSummary__StyledEndState")); //e.g. "Weighed In"
                                                                                                               //Parse meta line: "3YO plus | 1m 100y | Good | 16 Runners | Unknown | Off time: - | Winning time: -"
                var meta = SplitMeta(addInfo); string ageRestriction = meta.TryGetValue("age", out var a1) ? a1 : null; string distanceText = meta.TryGetValue("dist", out var d1) ? d1 : null; string going = meta.TryGetValue("going", out var g1) ? g1 : null; string runners = meta.TryGetValue("runners", out var r1) ? r1 : null; string offTime = meta.TryGetValue("off", out var o1) ? o1 : null; string winTime = meta.TryGetValue("win", out var w1) ? w1 : null; //pull segments
                int? runnerCount = TryParseInt(runners?.Split(' ').FirstOrDefault()); //first token before "Runners"
                int distanceYards = ParseDistanceToYards(distanceText); //canonical yards
                TimeSpan? scheduledOff = ExtractScheduledOffFromHeader(TextOrEmpty(driver, By.CssSelector("p.CourseListingHeader__StyledMainTitle"))); //time in the header
                TimeSpan? actualOff = ParseClockTime(offTime); //actual off
                int? winningMs = ParseWinningMs(winTime); //ms or null
            Console.WriteLine($"Parsing race: {raceTitle} at {courseName} on {raceDate:yyyy-MM-dd}");
            Console.WriteLine($"  Distance: {distanceText}, Going: {going}, Runners: {runnerCount}, Status: {status}");
            //TODO: Upsert Course(courseName,country: null) -> get courseId
            //TODO: Upsert Race with: CourseId, RaceDate=raceDate, ScheduledOff=scheduledOff??TimeSpan.Zero, ActualOff=actualOff, Title=raceTitle, RaceType=InferRaceType(raceTitle), Class=ExtractClass(addInfo), AgeRestriction=ageRestriction, Surface=InferSurface(addInfo), Going=going, DistanceYards=distanceYards, DistanceText=distanceText, RunnerCount=(byte?)runnerCount, Status=status, WinningTimeMs=winningMs, WinningTimeText=winTime
            //Runners --------------------------------------------------------------------------------------
            wait.Until(d => d.FindElements(By.CssSelector("[class*='ResultRunner__StyledResultRunnerWrapper']")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id='horse-sub-info']")).Count > 0); //ensure table visible
                var rows = driver.FindElements(By.CssSelector("[class*='ResultRunner__StyledResultRunnerWrapper']")); foreach (var row in rows)
                {
                    string posText = SafeText(row, By.CssSelector("[data-test-id='position-no'], .position-no .ordinal")); int? finishPos = TryParseInt(posText); string outcome = ParseOutcomeCode(posText); //position/outcome
                    string cloth = SafeText(row, By.CssSelector("[data-test-id='saddle-cloth-no']")); byte? saddle = TryParseByte(cloth); //saddle cloth
                    string draw = SafeText(row, By.CssSelector("[data-test-id='stall-no']")); byte? stall = TryParseByte(draw); //draw
                    string horseName = SafeText(row, By.CssSelector("a[href*='/racing/profiles/horse/'], [data-test-id='horse-name']")); //horse
                    string ageWeight = SafeText(row, By.CssSelector("[data-test-id='horse-sub-info']")); //e.g. "(4) 9-12"
                    (byte? age, byte? weightLbs, string weightTxt) = ParseAgeWeight(ageWeight); //parse (age, lbs, text)
                                                                                                //trainer/jockey block: two anchors in the block
                    string trainer = SafeText(row, By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]//a[1]")); string jockey = SafeText(row, By.XPath(".//div[contains(@class,'StyledTrainerJockeyBlock')]//a[2]")); //names
                                                                                                                                                                                                                                 //SP odds on right: span with BetLink class inside runner row
                    string spFrac = SafeText(row, By.CssSelector("span[class*='BetLinkStyle']")); decimal? spDec = FractionToDecimal(spFrac); string favTag = ExtractFavouriteTag(spFrac); //odds + F/JF tag
                                                                                                                                                                                           //Distances between finishers
                    string beatenTxt = SafeText(row, By.CssSelector("[class*='StyledFinishDistance']")); decimal? beatenLen = ParseBeatenLengths(beatenTxt); //hd/nk/1.25 etc
                                                                                                                                                             //Opening/touched (if present in row’s small text)
                    string opTxt = SafeText(row, By.XPath(".//*[contains(.,'op ') and contains(@class,'small')]")); string tchTxt = SafeText(row, By.XPath(".//*[contains(.,'tchd ') or contains(.,'tch ') and contains(@class,'small')]")); string opFrac = ExtractOddsToken(opTxt, "op"); (string? tchLow, string? tchHigh) = ExtractTouchedTokens(tchTxt); //market moves
                                                                                                                                                                                                                                                                                                                                                              //Comment line
                    string comment = SafeText(row, By.CssSelector("[data-test-id='ride-description'], [class*='StyledRideDescription']")); //race reader note
                Console.WriteLine($"    Pos: {finishPos?.ToString() ?? outcome} Horse: {horseName} Jockey: {jockey} Trainer: {trainer} SP: {spFrac} Beaten: {beatenTxt}");                                                                                                                   //TODO: Upsert Horse/Trainer/Jockey -> get IDs
                                                                                                                                                                                                                                                                                             //TODO: Insert RunnerResult with parsed fields mapped to your schema
            }
            }
            private static string GuessCourseFromBreadcrumb(IWebDriver d) { try { return d.FindElements(By.CssSelector("p.CourseListingHeader__StyledMainTitle")).FirstOrDefault()?.Text?.Split(' ').LastOrDefault() ?? ""; } catch { return ""; } } //fallback
            private static string ExtractCourseNameFromHeader(string header) { if (string.IsNullOrWhiteSpace(header)) return ""; var parts = header.Trim().Split(' ', 2); if (parts.Length < 2) return ""; return parts[1].Trim(); } //from "14:10 Beverley"
            private static TimeSpan? ExtractScheduledOffFromHeader(string header) { if (string.IsNullOrWhiteSpace(header)) return null; var time = header.Split(' ').FirstOrDefault(); return ParseClockTime(time); } //first token is "14:10"
            private static Dictionary<string, string> SplitMeta(string meta)
            {
                var r = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase); if (string.IsNullOrWhiteSpace(meta)) return r; var parts = meta.Split('|', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries); //tokens
                foreach (var p in parts)
                {
                    if (p.Contains("YO", StringComparison.OrdinalIgnoreCase)) r["age"] = p.Trim(); else if (p.Contains("Runners", StringComparison.OrdinalIgnoreCase)) r["runners"] = p.Trim(); else if (p.StartsWith("Off time", StringComparison.OrdinalIgnoreCase)) r["off"] = p.Split(':', 2)[1].Trim(); else if (p.StartsWith("Winning time", StringComparison.OrdinalIgnoreCase)) r["win"] = p.Split(':', 2)[1].Trim(); else if (IsGoingToken(p)) r["going"] = p.Trim(); else if (HasDistanceToken(p)) r["dist"] = p.Trim();
                }
                return r;
            }
            private static bool IsGoingToken(string s) { if (string.IsNullOrWhiteSpace(s)) return false; var keys = new[] { "Good", "Firm", "Soft", "Heavy", "Standard" }; return keys.Any(k => s.Contains(k, StringComparison.OrdinalIgnoreCase)); } //going marker
            private static bool HasDistanceToken(string s) { if (string.IsNullOrWhiteSpace(s)) return false; return s.Any(ch => "mfyl".Contains(char.ToLowerInvariant(ch))); } //m/f/y/l hints
            private static int ParseDistanceToYards(string? text)
            {
                if (string.IsNullOrWhiteSpace(text)) return 0; int yards = 0; //accumulate
                                                                              //supports forms like "1m 100y", "6f 16y", "1m 2f", "7f", "2m 5f 110y"
                foreach (var tok in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    if (tok.EndsWith("m", StringComparison.OrdinalIgnoreCase) && int.TryParse(tok.TrimEnd('m', 'M'), out var m)) yards += m * 1760; else if (tok.EndsWith("f", StringComparison.OrdinalIgnoreCase) && int.TryParse(tok.TrimEnd('f', 'F'), out var f)) yards += f * 220; else if (tok.EndsWith("y", StringComparison.OrdinalIgnoreCase) && int.TryParse(tok.TrimEnd('y', 'Y'), out var y)) yards += y;
                }
                return yards;
            }
            private static TimeSpan? ParseClockTime(string? t) { if (string.IsNullOrWhiteSpace(t)) return null; return TimeSpan.TryParse(t, CultureInfo.InvariantCulture, out var ts) ? ts : null; } //HH:mm
            private static int? ParseWinningMs(string? t)
            {
                if (string.IsNullOrWhiteSpace(t) || t == "-") return null; //supports "1m 15.11s" or "59.87s"
                try
                {
                    t = t.Trim(); double seconds = 0; if (t.Contains('m')) { var parts = t.Replace("s", "").Split('m', StringSplitOptions.TrimEntries); var mins = int.Parse(parts[0]); var secs = double.Parse(parts[1], CultureInfo.InvariantCulture); seconds = mins * 60 + secs; } else { seconds = double.Parse(t.TrimEnd('s'), CultureInfo.InvariantCulture); }
                    return (int)Math.Round(seconds * 1000.0);
                }
                catch { return null; }
            }
        private static string ParseOutcomeCode(string posText)
        {
            if (string.IsNullOrWhiteSpace(posText)) return ""; //no text => no special outcome
            var t = posText.Trim().ToUpperInvariant(); //normalize
            var letters = new string(t.Where(char.IsLetter).ToArray()); //keep only letters (e.g., "PU","UR","F")
            if (letters.Length == 0) return ""; //pure number like "1" or "2nd" => finished position, no code
            switch (letters) //map common UK result codes; return as-is otherwise
            {
                case "PU": case "F": case "UR": case "RO": case "DSQ": case "BD": case "SU": case "RR": case "REF": case "WD": case "NR": case "VOID": case "CO": case "DNF": return letters; //known codes
                default: return letters; //fallback to whatever the page shows
            }
        }

        private static (byte? age, byte? lbs, string weightTxt) ParseAgeWeight(string s)
            {
                if (string.IsNullOrWhiteSpace(s)) return (null, null, ""); //expects like "(4) 9-12"
                byte? age = null; var ageMatch = System.Text.RegularExpressions.Regex.Match(s, @"\((\d{1,2})\)"); if (ageMatch.Success) age = byte.Parse(ageMatch.Groups[1].Value); string wt = System.Text.RegularExpressions.Regex.Match(s, @"\d{1,2}-\d{1,2}").Value; byte? lbs = null; if (!string.IsNullOrEmpty(wt)) { var parts = wt.Split('-'); lbs = (byte)(int.Parse(parts[0]) * 14 + int.Parse(parts[1])); }
                return (age, lbs, wt);
            }
            private static decimal? FractionToDecimal(string frac)
            {
                if (string.IsNullOrWhiteSpace(frac)) return null; frac = frac.Trim().ToLowerInvariant(); frac = frac.Replace("jf", "").Replace("f", "").Trim(); //remove fav flags
                var m = System.Text.RegularExpressions.Regex.Match(frac, @"(\d+)\s*/\s*(\d+)"); if (!m.Success) return null; var a = decimal.Parse(m.Groups[1].Value); var b = decimal.Parse(m.Groups[2].Value); return Math.Round(1 + (a / b), 3); //decimal price incl. stake
            }
            private static string ExtractFavouriteTag(string frac) { if (string.IsNullOrWhiteSpace(frac)) return null; frac = frac.ToUpperInvariant(); if (frac.Contains("JF")) return "JF"; if (frac.Contains("CF")) return "CF"; if (frac.EndsWith("F")) return "F"; return null; } //F/JF/CF
            private static decimal? ParseBeatenLengths(string s)
            {
                if (string.IsNullOrWhiteSpace(s)) return null; s = s.Trim().ToLowerInvariant(); if (s is "nk") return 0.3m; if (s is "hd") return 0.2m; if (s is "shd") return 0.1m; s = s.Replace("¾", ".75").Replace("½", ".5").Replace("¼", ".25"); if (decimal.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var v)) return v; return null; //simple mapping
            }
            private static string ExtractOddsToken(string s, string key) { if (string.IsNullOrWhiteSpace(s)) return null; var m = System.Text.RegularExpressions.Regex.Match(s, $@"\b{System.Text.RegularExpressions.Regex.Escape(key)}\s+(\d+/\d+\w*)", System.Text.RegularExpressions.RegexOptions.IgnoreCase); return m.Success ? m.Groups[1].Value : null; } //e.g. "op 33/1"
            private static (string? low, string? high) ExtractTouchedTokens(string s) { if (string.IsNullOrWhiteSpace(s)) return (null, null); var low = System.Text.RegularExpressions.Regex.Match(s, @"(tchd|low)\s+(\d+/\d+\w*)", System.Text.RegularExpressions.RegexOptions.IgnoreCase).Groups.Cast<System.Text.RegularExpressions.Group>().Skip(2).FirstOrDefault()?.Value; var high = System.Text.RegularExpressions.Regex.Match(s, @"high\s+(\d+/\d+\w*)", System.Text.RegularExpressions.RegexOptions.IgnoreCase).Groups.Cast<System.Text.RegularExpressions.Group>().Skip(1).FirstOrDefault()?.Value; return (low, high); } //touched low/high
            private static int? TryParseInt(string? s) { return int.TryParse(s?.Trim(), out var v) ? v : null; } //nullable int
            private static byte? TryParseByte(string? s) { return byte.TryParse(s?.Trim(), out var v) ? v : null; } //nullable byte
            private static DateTime? ParseDateSafe(string? s) { if (string.IsNullOrWhiteSpace(s)) return null; if (DateTime.TryParse(s, out var d)) return d.Date; return null; } //date or null
            private static string TextOrEmpty(IWebDriver d, By by) { try { return d.FindElement(by).Text.Trim(); } catch { return ""; } } //driver text
            private static string SafeText(IWebElement e, By by) { try { return e.FindElement(by).Text.Trim(); } catch { return ""; } } //element text
            private static string InferRaceType(string title) { if (string.IsNullOrWhiteSpace(title)) return ""; var keys = new[] { "Handicap", "Maiden", "Novice", "Apprentice", "Claiming", "Selling", "Stakes" }; return keys.FirstOrDefault(k => title.IndexOf(k, StringComparison.OrdinalIgnoreCase) >= 0) ?? ""; } //simple classifier
            private static byte? ExtractClass(string meta) { if (string.IsNullOrWhiteSpace(meta)) return null; var m = System.Text.RegularExpressions.Regex.Match(meta, @"Class\s+(\d)"); return m.Success ? (byte?)byte.Parse(m.Groups[1].Value) : null; } //Class 1..7
            private static string InferSurface(string meta) { if (string.IsNullOrWhiteSpace(meta)) return null; if (meta.Contains("All Weather", StringComparison.OrdinalIgnoreCase) || meta.Contains("Allweather", StringComparison.OrdinalIgnoreCase)) return "Allweather"; return "Turf"; } //basic surf
            private static bool AcceptTermsIfPresent(IWebDriver driver)
            {
                try
                {
                    var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(2)); var acceptButton = wait.Until(d => { var buttons = d.FindElements(By.XPath("//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow all cookies')]")); return buttons.FirstOrDefault(b => b.Displayed && b.Enabled); }); acceptButton?.Click(); return acceptButton != null; //clicked if found
                }
                catch (WebDriverTimeoutException) { return false; } //no popup
            }
        }        
    }
