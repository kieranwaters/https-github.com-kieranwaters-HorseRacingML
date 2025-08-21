using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using System.Globalization;

namespace HorseRacingML.Scraping
{
    public class RaceResultsScraper
    {
        public void Scrape(DateTime startDate, DateTime endDate)
        {
            var options = new ChromeOptions(); options.AddArgument("--start-maximized"); // visible
            using var driver = new ChromeDriver(options); var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10)); // waits

            for (var date = startDate; date <= endDate; date = date.AddDays(1))
            {
                try
                {
                    var url = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}"; driver.Navigate().GoToUrl(url); // go date
                    AcceptTermsIfPresent(driver); // cookies

                    string[] regions = { "UK & Ireland", "International" }; // toggle regions
                    foreach (var region in regions)
                    {
                        try
                        {
                            var regionBtn = driver.FindElements(By.XPath($"//*[self::button or self::span][contains(normalize-space(.), '{region}') and ancestor::*[@data-test-id='new-switch-button']]")).FirstOrDefault(e => e.Displayed && e.Enabled); // region pill
                            if (regionBtn == null) { continue; } // not present

                            ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", regionBtn); // click
                            wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0); // meeting tabs
                            ScrapeMeetingTabs(driver, wait, date); // scrape meetings
                        }
                        catch (WebDriverException ex) { Console.WriteLine($"Region '{region}' error on {date:yyyy-MM-dd}: {ex.Message}"); } // log
                    }

                    if (driver.FindElements(By.CssSelector("[data-test-id='new-switch-button']")).Count == 0) { ScrapeMeetingTabs(driver, wait, date); } // pages w/o regions
                }
                catch (WebDriverException ex) { Console.WriteLine($"Skipping {date:yyyy-MM-dd}: {ex.Message}"); } // next day
            }
        }

        private static void ScrapeMeetingTabs(IWebDriver driver, WebDriverWait wait, DateTime raceDate)
        {
            try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='generic-tab']")).Count > 0); } catch { Console.WriteLine("Meeting tab elements not found."); return; } // ensure tabs

            int tabIndex = 0;
            while (true)
            {
                var tabs = driver.FindElements(By.CssSelector("[data-test-id='generic-tab']")); if (tabIndex >= tabs.Count) break; // done
                var tab = tabs[tabIndex];

                try
                {
                    ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].scrollIntoView({block:'center'});", tab); ((IJavaScriptExecutor)driver).ExecuteScript("arguments[0].click();", tab); // click
                }
                catch { tabIndex++; continue; } // next meeting

                try { wait.Until(d => d.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id='race-container']")).Count > 0); } catch { tabIndex++; continue; } // wait races

                var raceLinks = driver.FindElements(By.CssSelector("[data-test-id='race-container'] a[href]")); var mainHandle = driver.CurrentWindowHandle; // remember tab

                var existingHandles = driver.WindowHandles.ToList(); // before open
                foreach (var a in raceLinks)
                {
                    var href = a.GetAttribute("href"); if (string.IsNullOrWhiteSpace(href)) continue; // skip
                    ((IJavaScriptExecutor)driver).ExecuteScript("window.open(arguments[0], '_blank');", href); // open race
                }

                var newHandles = driver.WindowHandles.Except(existingHandles).ToList(); // new tabs

                foreach (var handle in newHandles)
                {
                    if (handle == mainHandle) { continue; } // avoid closing primary window
                    driver.SwitchTo().Window(handle); // focus
                    try { ParseRacePage(driver, wait, raceDate); } catch (Exception ex) { Console.WriteLine($"Parse error: {ex.Message}"); } // parse
                    driver.Close(); // close tab
                    driver.SwitchTo().Window(mainHandle); // ensure main window remains active
                }
                tabIndex++; // next meeting
            }
        }

        private static void ParseRacePage(IWebDriver driver, WebDriverWait wait, DateTime defaultDate)
        {
            var headerText = TextOrEmpty(driver, By.CssSelector("p[class*='CourseListingHeader__StyledMainTitle']"));
            if (string.IsNullOrWhiteSpace(headerText))
                headerText = TextOrEmpty(driver, By.CssSelector("[data-test-id='course-title'], [class*='CourseListingHeader__StyledMainTitle']"));
            string courseName = ExtractCourseNameFromHeader(headerText);
            if (string.IsNullOrEmpty(courseName)) courseName = GuessCourseFromBreadcrumb(driver);
            var dateText = TextOrEmpty(driver, By.CssSelector("p[class*='CourseListingHeader__StyledMainSubTitle'], [data-test-id='course-subtitle']"));
            DateTime raceDate = ParseDateSafe(dateText) ?? defaultDate;
            var raceTitle = TextOrEmpty(driver, By.CssSelector("h1[data-test-id='racecard-race-name'], h1[class*='RacingRacecardSummary__StyledTitle']"));
            var metaLine = TextOrEmpty(driver, By.CssSelector("li[class*='RacingRacecardSummary__StyledAdditionalInfo'], [data-test-id='racecard-additional-info']"));
            metaLine = Normalize(metaLine);
            var status = TextOrEmpty(driver, By.CssSelector(".RacingRacecardSummary__StyledEndState, [data-test-id='racecard-end-state']"));
            if (string.IsNullOrWhiteSpace(status))
                status = TextOrEmpty(driver, By.XPath("//li[contains(@class,'RacingRacecardSummary__StyledAdditionalInfo')]//span[contains(@class,'EndState') or contains(.,'Weighed In') or contains(.,'Abandoned') or contains(.,'Void')]"));
            status = Normalize(status);
            var meta = SplitMeta(metaLine);
            string ageRestriction = meta.TryGetValue("age", out var a1) ? a1 : null;
            string distanceText = meta.TryGetValue("dist", out var d1) ? d1 : null;
            string going = meta.TryGetValue("going", out var g1) ? g1 : null;
            string runners = meta.TryGetValue("runners", out var r1) ? r1 : null;
            string offTime = meta.TryGetValue("off", out var o1) ? o1 : null;
            string winTime = meta.TryGetValue("win", out var w1) ? w1 : null;
            int? runnerCount = TryParseOrdinalInt(runners?.Split(' ').FirstOrDefault());
            int distanceYards = ParseDistanceToYards(distanceText);
            TimeSpan? scheduledOff = ExtractScheduledOffFromHeader(headerText);
            TimeSpan? actualOff = ParseClockTime(offTime);
            int? winningMs = ParseWinningMs(winTime);
            Console.WriteLine($"Parsing race: {raceTitle} at {courseName} on {raceDate:yyyy-MM-dd}");
            Console.WriteLine($"  Distance: {distanceText}, Going: {going}, Runners: {runnerCount}, Status: {status}");
            wait.Until(d => d.FindElements(By.CssSelector("[class*='ResultRunner__StyledResultRunnerWrapper']")).Count > 0 || d.FindElements(By.CssSelector("[data-test-id='horse-sub-info']")).Count > 0);
            var rows = driver.FindElements(By.CssSelector("[class*='ResultRunner__StyledResultRunnerWrapper']"));
            foreach (var row in rows)
            {
                string posRaw = SafeText(row, By.CssSelector("[data-test-id='position-no'], .position-no .ordinal"));
                int? finishPos = TryParseOrdinalInt(posRaw);
                string outcome = finishPos.HasValue ? "" : ParseOutcomeCode(posRaw);
                string cloth = SafeText(row, By.CssSelector("[data-test-id='saddle-cloth-no']"));
                byte? saddle = TryParseByte(cloth);
                string draw = SafeText(row, By.CssSelector("[data-test-id='stall-no']"));
                byte? stall = TryParseByte(draw);
                string horseName = SafeText(row, By.CssSelector("a[href*='/racing/profiles/horse/'], [data-test-id='horse-name']"));
                string ageWeight = SafeText(row, By.CssSelector("[data-test-id='horse-sub-info']"));
                (byte? age, byte? weightLbs, string weightTxt) = ParseAgeWeight(ageWeight);
                var (trainer, jockey) = ExtractTrainerJockey(row); // NEW trainer/jockey logic
                string spFrac = SafeText(row, By.CssSelector("span[class*='BetLinkStyle'], [data-test-id='sp-odds']"));
                decimal? spDec = FractionToDecimal(spFrac);
                string favTag = ExtractFavouriteTag(spFrac);
                string beatenTxt = SafeText(row, By.CssSelector("[class*='StyledFinishDistance'], [data-test-id='finish-distance']"));
                decimal? beatenLen = ParseBeatenLengths(beatenTxt);
                string opTxt = SafeText(row, By.XPath(".//*[contains(.,'op ') and contains(@class,'small')]"));
                string tchTxt = SafeText(row, By.XPath(".//*[contains(.,'tchd ') or contains(.,'tch ') and contains(@class,'small')]"));
                string opFrac = ExtractOddsToken(opTxt, "op");
                (string? tchLow, string? tchHigh) = ExtractTouchedTokens(tchTxt);
                string comment = SafeText(row, By.CssSelector("[data-test-id='ride-description'], [class*='StyledRideDescription']"));
                Console.WriteLine($"    Pos: {(finishPos.HasValue ? finishPos.ToString() : outcome)} Horse: {horseName} Jockey: {jockey} Trainer: {trainer} SP: {spFrac} Beaten: {beatenTxt}");
            }
        }

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

        // ----- Helpers -----
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

        private static bool HasDistanceToken(string s) { if (string.IsNullOrWhiteSpace(s)) return false; return s.Any(ch => "mfyl".Contains(char.ToLowerInvariant(ch))); } // m/f/y/l hints

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
        private static string TextOrEmpty(IWebDriver d, By by) { try { return d.FindElement(by).Text.Trim(); } catch { return ""; } } // driver text
        private static string SafeText(IWebElement e, By by) { try { return e.FindElement(by).Text.Trim(); } catch { return ""; } } // element text

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
