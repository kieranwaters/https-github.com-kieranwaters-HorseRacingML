using HorseRacingML.Models;
using OpenQA.Selenium;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Tensorflow.Keras.Engine;
using System.Text;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        private static readonly Regex DistanceComponentRegex = new("(?<value>[0-9]+(?:\\.[0-9]+)?)\\s*(?<unit>[mfy])", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex ClassRegex = new("class\\s*(?<value>[0-9])", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeRestrictionRegex = new("(?<value>[0-9]{1,2}\\s*(?:yo\\+?|yo|yrs?\\+?|years?\\+?))", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex MarketTitleExchangeSuffixRegex = new(@"\s*(?:[»|,-]\s*)?BetfairT?\s*Exchange.*$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex MarketTitleSiteSuffixRegex = new(@"\s*(?:[-–—]|\|)\s*Betfair.*$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex LeadingRaceTypeRegex = new(
            @"^\s*(?<type>(?:[A-Za-z'\-]+(?:\s+[A-Za-z'\-]+)*)|(?:G[1-3])|(?:Group\s+[1-3])|(?:Grade\s+[1-3]))",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly string[] RaceTypeKeywords =
        {
            "handicap",
            "maiden",
            "novice",
            "stakes",
            "selling",
            "claiming",
            "listed",
            "group",
            "g1",
            "g2",
            "g3",
            "nursery",
            "hurdle",
            "chase",
            "bumper",
            "nh flat",
            "condition",
            "fillies",
            "mares",
            "apprentice",
            "amateur"
        };
        private static string NormalizeMarketTitle(string? rawTitle)
        {
            if (string.IsNullOrWhiteSpace(rawTitle))
            {
                return string.Empty;
            }

            var normalized = rawTitle.Replace('\u00A0', ' ').Trim();

            var bettingIndex = normalized.IndexOf(" Betting Odds", StringComparison.OrdinalIgnoreCase);
            if (bettingIndex >= 0)
            {
                normalized = normalized.Substring(0, bettingIndex).Trim();
            }

            normalized = MarketTitleExchangeSuffixRegex.Replace(normalized, string.Empty);
            normalized = MarketTitleSiteSuffixRegex.Replace(normalized, string.Empty);

            while (normalized.Contains("  "))
            {
                normalized = normalized.Replace("  ", " ");
            }

            return normalized.Trim();
        }
        private static string ResolveAiWeightPath()
        {
            static IEnumerable<string> EnumerateCandidates()
            {
                static IEnumerable<string> ExpandDirectory(string? directory)
                {
                    if (string.IsNullOrWhiteSpace(directory))
                    {
                        yield break;
                    }

                    yield return Path.Combine(directory, "weights", "aiweights.json");
                    yield return Path.Combine(directory, "aiweights.json");
                }

                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

                foreach (var path in ExpandDirectory(AppContext.BaseDirectory))
                {
                    if (seen.Add(path))
                    {
                        yield return path;
                    }
                }

                foreach (var path in ExpandDirectory(Directory.GetCurrentDirectory()))
                {
                    if (seen.Add(path))
                    {
                        yield return path;
                    }
                }

                var current = AppContext.BaseDirectory;
                for (var i = 0; i < 5 && !string.IsNullOrEmpty(current); i++)
                {
                    current = Path.GetDirectoryName(current?.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                    if (string.IsNullOrEmpty(current))
                    {
                        break;
                    }

                    foreach (var path in ExpandDirectory(current))
                    {
                        if (seen.Add(path))
                        {
                            yield return path;
                        }
                    }
                }
            }

            foreach (var candidate in EnumerateCandidates())
            {
                if (File.Exists(candidate))
                {
                    Console.WriteLine($"\tUsing AI weight file at {candidate}");
                    return candidate;
                }
            }

            var fallback = Path.Combine(AppContext.BaseDirectory, "weights", "aiweights.json");
            Console.Error.WriteLine($"\tAI weight file not found; expected locations include {fallback}");
            return fallback;
        }

        private static ParsedRaceMetadata ParseRaceMetadata(RaceDayReport race)
        {
            var tokens = EnumerateDetailTokens(race.RaceDetails)
                .Concat(EnumerateDetailTokens(race.RaceTitle))
                .Concat(EnumerateDetailTokens(race.RaceType))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            string? distanceToken = tokens.FirstOrDefault(HasDistanceToken);
            if (distanceToken == null && !string.IsNullOrWhiteSpace(race.RaceDetails))
            {
                distanceToken = race.RaceDetails;
            }

            var distanceYards = ParseDistanceToYards(distanceToken);

            byte? classValue = null;
            foreach (var token in tokens)
            {
                var match = ClassRegex.Match(token);
                if (match.Success && byte.TryParse(match.Groups["value"].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                {
                    classValue = parsed;
                    break;
                }
            }

            string? going = tokens.FirstOrDefault(IsGoingToken);

            string? ageRestriction = null;
            foreach (var token in tokens)
            {
                var match = AgeRestrictionRegex.Match(token);
                if (match.Success)
                {
                    ageRestriction = match.Groups["value"].Value.Replace(" ", string.Empty);
                    break;
                }
            }

            var raceType = TryDetectRaceType(tokens, race.RaceTitle);
            var surface = DetermineSurface(going, tokens);

            return new ParsedRaceMetadata(
                string.IsNullOrWhiteSpace(distanceToken) ? null : distanceToken.Trim(),
                distanceYards,
                classValue,
                string.IsNullOrWhiteSpace(ageRestriction) ? null : ageRestriction,
                string.IsNullOrWhiteSpace(going) ? null : going.Trim(),
                surface,
                raceType);
        }
        private static (string? RaceTypeText, string? CleanedDetails) SplitRaceTypeFromDetails(string? raceDetails)
        {
            if (string.IsNullOrWhiteSpace(raceDetails))
            {
                return (null, null);
            }

            var trimmed = raceDetails.Trim();
            if (trimmed.Length == 0)
            {
                return (null, null);
            }

            var match = LeadingRaceTypeRegex.Match(trimmed);
            if (!match.Success)
            {
                return (null, trimmed);
            }

            var typeSegment = match.Groups["type"].Value.Trim();
            if (string.IsNullOrEmpty(typeSegment))
            {
                return (null, trimmed);
            }

            var remainder = trimmed.Substring(match.Length);
            remainder = remainder.TrimStart(' ', '\t', '-', '–', '—', '|', '/', ',', ';');
            remainder = remainder.Trim();

            return (typeSegment, string.IsNullOrWhiteSpace(remainder) ? null : remainder);
        }

        private static IEnumerable<string> EnumerateDetailTokens(string? source)
        {
            if (string.IsNullOrWhiteSpace(source))
            {
                yield break;
            }

            var normalized = source.Replace('\u00A0', ' ').Trim();
            if (!string.IsNullOrEmpty(normalized))
            {
                yield return normalized;
            }

            var separators = new[] { '|', '/', '\\', ',', ';', '–', '—', '·' };
            foreach (var segment in normalized.Split(separators, StringSplitOptions.RemoveEmptyEntries))
            {
                var trimmed = segment.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    yield return trimmed;
                }

                foreach (var token in trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    var inner = token.Trim();
                    if (!string.IsNullOrEmpty(inner))
                    {
                        yield return inner;
                    }
                }
            }
        }

        private static bool HasDistanceToken(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                return false;
            }

            return DistanceComponentRegex.IsMatch(token);
        }

        private static int ParseDistanceToYards(string? token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                return 0;
            }

            var text = token.ToLowerInvariant();
            var total = 0;
            foreach (Match match in DistanceComponentRegex.Matches(text))
            {
                if (!double.TryParse(match.Groups["value"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                {
                    continue;
                }

                var unit = match.Groups["unit"].Value;
                switch (unit)
                {
                    case "m":
                        total += (int)Math.Round(value * 1760);
                        break;
                    case "f":
                        total += (int)Math.Round(value * 220);
                        break;
                    case "y":
                        total += (int)Math.Round(value);
                        break;
                }
            }

            return total;
        }
        private static bool TryClickBackAllViaScript(IJavaScriptExecutor js, IWebElement row)
        {
            try
            {
                var result = js.ExecuteScript(@"
            const runnerRow = arguments[0];
            if (!runnerRow) { return false; }

            const matchesBackAll = el => {
                if (!el) { return false; }
const hasBackAllContext = target => {
                    if (!target) { return false; }
                    const classAttr = (target.getAttribute && (target.getAttribute('class') || '') || '').toLowerCase();
                    if (classAttr.includes('back-all')) { return true; }
                    if (classAttr.includes('lay')) { return false; }

                    if (target.closest) {
                        const cell = target.closest('td.bet-buttons.back-cell.last-back-cell');
                        if (cell) { return true; }

                        const betTypeHolder = target.closest('[bet-type]');
                        if (betTypeHolder) {
                            const betType = (betTypeHolder.getAttribute('bet-type') || '').toLowerCase();
                            const handicap = (betTypeHolder.getAttribute('bet-handicap') || '').toLowerCase();
                            if (betType === 'back' && (!handicap || handicap === '0')) {
                                return true; }
                        }
                    }

                    return false;
                };

                const read = target => {
                    if (!target) { return ''; }
                    const text = (target.textContent || '').toLowerCase();
                    if (text.includes('back all')) { return 'back all'; }
                    const aria = (target.getAttribute && (target.getAttribute('aria-label') || '') || '').toLowerCase();
                    if (aria.includes('back all')) { return 'back all'; }
                    const title = (target.getAttribute && (target.getAttribute('title') || '') || '').toLowerCase();
                    if (title.includes('back all')) { return 'back all'; }
                    const testId = (target.getAttribute && (target.getAttribute('data-testid') || '') || '').toLowerCase();
                    if (testId.includes('back-all')) { return 'back all'; }
                    const dataTrack = (target.getAttribute && (target.getAttribute('data-trackid') || '') || '').toLowerCase();
                    if (dataTrack.includes('back all') || dataTrack.includes('back-all')) { return 'back all'; }
                    return text;
                };

                const text = read(el);
                if (text && text.includes('back all')) { return true; }

                const labels = Array.from(el.querySelectorAll ? el.querySelectorAll('label, span, strong, div, p') : []);
                for (const label of labels) {
                    if (read(label).includes('back all')) { return true; }
                }

                if (hasBackAllContext(el)) {
                    const buttonClass = (el.getAttribute && (el.getAttribute('class') || '') || '').toLowerCase();
                    if (!buttonClass.includes('lay')) { return true; }
                }

                return false;
            };

            const buttonSelectors = 'button, [role=""button""], .bet-button, ours-price-button button, td.bet-buttons.back-cell.last-back-cell > ours-price-button > button, td.bet-buttons.back-cell.last-back-cell ours-price-button button, td[bet-type=""back""] ours-price-button button';
            const buttons = Array.from(runnerRow.querySelectorAll(buttonSelectors));

            for (const btn of buttons) {
                if (matchesBackAll(btn)) {
                    btn.scrollIntoView({ block: 'center' });
                    btn.click();
                    return true;
                }
                const nestedButton = btn.querySelector ? btn.querySelector('button') : null;
                if (nestedButton && matchesBackAll(nestedButton)) {
                    nestedButton.scrollIntoView({ block: 'center' });
                    nestedButton.click();
                    return true;
                }
            }

            const resolveAbsoluteBackAllButton = () => {
                const tableRow = runnerRow.closest('tr');
                if (!tableRow) { return null; }
                const parent = tableRow.parentElement;
                if (!parent) { return null; }
                const rows = Array.from(parent.children);
                const index = rows.indexOf(tableRow);
                if (index < 0) { return null; }
                const nth = index + 1;
                const absoluteBase = `#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > div > div > div.bf-col-xxl-17-24.bf-col-xl-16-24.bf-col-lg-16-24.bf-col-md-15-24.bf-col-sm-14-24.bf-col-14-24.center-column.bfMarketSettingsSpace.bf-module-loading.nested-scrollable-pane-parent.market-settings-space > div.scrollable-panes-height-taker.height-taker-helper > div > div.bf-row.main-mv-container > div > bf-main-market > bf-main-marketview > div > div.main-mv-runners-list-wrapper > bf-marketview-runners-list.runners-list-unpinned > div > div > div > table > tbody > tr:nth-child(${nth}) > td.bet-buttons.back-cell.last-back-cell > ours-price-button`;
                const candidates = [
                    `${absoluteBase} > button`,
                    `${absoluteBase}:nth-of-type(1) > button`,
                    `${absoluteBase}:nth-of-type(2) > button`,
                    `${absoluteBase}:nth-of-type(3) > button`
                ];
                for (const selector of candidates) {
                    const candidate = document.querySelector(selector);
                    if (matchesBackAll(candidate)) {
                        return candidate;
                    }
                }
                return null;
            };

            const absoluteButton = resolveAbsoluteBackAllButton();
            if (absoluteButton) {
                absoluteButton.scrollIntoView({ block: 'center' });
                absoluteButton.click();
                return true;
            }

            const fallback = Array.from(runnerRow.querySelectorAll('*'))
                .find(matchesBackAll);

            if (fallback) {
                fallback.scrollIntoView({ block: 'center' });
                fallback.click();
                return true;
            }

            return false;
        ", row);

                return result is bool success && success;
            }
            catch (Exception)
            {
                return false;
            }
        }
        private static string? TryDetectRaceType(IEnumerable<string> tokens, string? title)
        {
            foreach (var token in tokens)
            {
                var normalized = token.Trim().ToLowerInvariant();
                foreach (var keyword in RaceTypeKeywords)
                {
                    if (normalized.Contains(keyword))
                    {
                        return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(keyword);
                    }
                }
            }

            if (!string.IsNullOrWhiteSpace(title))
            {
                var normalized = title.ToLowerInvariant();
                foreach (var keyword in RaceTypeKeywords)
                {
                    if (normalized.Contains(keyword))
                    {
                        return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(keyword);
                    }
                }
            }

            return null;
        }

        private static bool IsGoingToken(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                return false;
            }

            var normalized = token.ToLowerInvariant();
            if (normalized.Contains("going"))
            {
                return true;
            }

            var keywords = new[] { "heavy", "soft", "yielding", "good", "firm", "standard", "slow", "fast" };
            return keywords.Any(k => Regex.IsMatch(normalized, $"\\b{k}\\b"));
        }

        private static string? DetermineSurface(string? going, IEnumerable<string> tokens)
        {
            static bool ContainsAwIndicator(string value)
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    return false;
                }

                var normalized = value.ToLowerInvariant();
                return normalized.Contains("all weather")
                    || normalized.Contains("all-weather")
                    || normalized.Contains("a/w")
                    || normalized.Equals("aw", StringComparison.OrdinalIgnoreCase)
                    || normalized.Contains("polytrack")
                    || normalized.Contains("tapeta")
                    || normalized.Contains("fibresand");
            }

            if (!string.IsNullOrEmpty(going) && going.IndexOf("standard", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return "All Weather";
            }

            if (tokens.Any(ContainsAwIndicator))
            {
                return "All Weather";
            }

            return "Turf";
        }

        private readonly record struct ParsedRaceMetadata(
            string? DistanceText,
            int DistanceYards,
            byte? Class,
            string? AgeRestriction,
            string? Going,
            string? Surface,
            string? RaceType);

        private static readonly Regex BracketedNameContentRegex =
            new Regex(@"\s*[\(\[][^\)\]]*[\)\]]\s*", RegexOptions.Compiled);

        private static string NormalizeName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var withoutBracketed = BracketedNameContentRegex.Replace(value, " ");
            var lower = withoutBracketed.Trim().ToLowerInvariant();
            var decomposed = lower.Normalize(NormalizationForm.FormD);

            var builder = new StringBuilder(decomposed.Length);
            foreach (var c in decomposed)
            {
                var category = CharUnicodeInfo.GetUnicodeCategory(c);
                if (category == UnicodeCategory.NonSpacingMark
                    || category == UnicodeCategory.SpacingCombiningMark
                    || category == UnicodeCategory.EnclosingMark)
                {
                    continue;
                }

                if (char.IsLetterOrDigit(c))
                {
                    builder.Append(c);
                }
            }

            return builder.ToString();
        }

        private static IReadOnlyList<IWebElement> FindBetSlipStakeInputs(IWebDriver driver)
        {
            var selectors = new[]
            {
                "input.betslip-size-input",
                "input[data-testid='betslip-stake-input']",
                "input[data-testid='bet-slip-input']",
                "input[data-testid='size-input']",
                "div.betslip-selection input[type='text']"
            };

            foreach (var selector in selectors)
            {
                try
                {
                    var found = driver
                        .FindElements(By.CssSelector(selector))
                        .Where(e => e.Displayed)
                        .ToList();

                    if (found.Count > 0)
                    {
                        return found;
                    }
                }
                catch (NoSuchElementException)
                {
                }
            }

            return Array.Empty<IWebElement>();
        }

        private static void SetStakeInputValue(IJavaScriptExecutor js, IWebElement input, decimal stake)
        {
            var text = stake > 0m
                ? stake.ToString("0.##", CultureInfo.InvariantCulture)
                : "0";

            js.ExecuteScript(@"const el = arguments[0];
                const value = arguments[1];
                if (!el) { return; }

                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
                const nativeNumberValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'valueAsNumber')?.set;
const typeAttr = (el.getAttribute('type') || '').toLowerCase();
                const supportsNumberSetter = ['number', 'range', 'time', 'datetime-local', 'month', 'week'].includes(typeAttr);
                el.focus();

                if (nativeInputValueSetter) {
                    nativeInputValueSetter.call(el, value);
                } else {
                    el.value = value;
                }

                if (supportsNumberSetter && nativeNumberValueSetter && !Number.isNaN(Number(value))) {
                    try {
                        nativeNumberValueSetter.call(el, Number(value));
                    } catch (err) {
                        // Some input types (e.g. Betfair text inputs) reject valueAsNumber; ignore.
                    }
                }

                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            ", input, text);
        }
        private static IWebElement? TryFindElement(ISearchContext context, By by)
        {
            try
            {
                return context.FindElement(by);
            }
            catch (NoSuchElementException)
            {
                return null;
            }
            catch (StaleElementReferenceException)
            {
                return null;
            }
        }
        private static decimal CalculateAiDecimalOdds(double aiProbability)
        {
            if (aiProbability <= 0)
            {
                return 0m;
            }

            var inverted = 1.0 / aiProbability;

            if (!double.IsFinite(inverted) || inverted <= 0)
            {
                return 0m;
            }

            if (inverted > (double)decimal.MaxValue)
            {
                return decimal.MaxValue;
            }

            return (decimal)inverted;
        }

        private static void NormalizeAiOdds(ICollection<RunnerFlow> flows, bool useMarketFallbackForDegeneracy)
      {
            var valid = flows
                .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                .ToList();
            var totalRunners = flows.Count;
            var missingCount = totalRunners - valid.Count;

            if (valid.Count == 0)
            {
                if (totalRunners > 0)
                {
                    Console.WriteLine($"\tNormalizeAiOdds: no valid AI probabilities across {totalRunners} runner(s); skipping normalization.");
                }
                return;
            }
            const double degeneracyTolerance = 1e-8;
            double minValue = valid.Min(f => f.AiOdds!.Value);
            double maxValue = valid.Max(f => f.AiOdds!.Value);

            if (maxValue - minValue <= degeneracyTolerance)
            {
                Console.WriteLine("\t\tDetected degenerate AI probability distribution; raw outputs are identical across runners.");

                if (TryApplyLegacyFallback(flows))
                {
                    valid = flows
                        .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                        .ToList();
                    missingCount = flows.Count - valid.Count;
                    minValue = valid.Min(f => f.AiOdds!.Value);
                    maxValue = valid.Max(f => f.AiOdds!.Value);
                }
                else if (TryResolveDegenerateDistribution(flows))
                {
                    valid = flows
                         .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                         .ToList();
                    missingCount = flows.Count - valid.Count;
                    if (valid.Count == 0)
                    {
                        Console.WriteLine("\t\tUnable to normalize after resolving degeneracy; all AI probabilities were discarded.");
                        return;
                    }

                    minValue = valid.Min(f => f.AiOdds!.Value);
                    maxValue = valid.Max(f => f.AiOdds!.Value);
                }
                else if (useMarketFallbackForDegeneracy)
                {
                    Console.WriteLine("\t\tFalling back to market-implied probabilities.");

                    bool anyFallbackApplied = false;
                    foreach (var flow in flows)
                    {
                        if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 1m)
                        {
                            flow.AiOdds = 1.0 / (double)flow.BackPrice1.Value;
                            flow.AiProbabilityMarketDerived = true;
                            anyFallbackApplied = true;
                        }
                        else
                        {
                            flow.AiOdds = null;
                            flow.AiProbabilityMarketDerived = false;
                        }
                    }

                    if (!anyFallbackApplied)
                    {
                        Console.WriteLine("\t\tUnable to apply market fallback due to missing back prices; AI odds will remain unavailable for this race.");
                        return;
                    }

                    valid = flows
                        .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                        .ToList();
                    missingCount = flows.Count - valid.Count;
                    if (valid.Count == 0)
                    {
                        Console.WriteLine("\t\tMarket fallback produced no usable probabilities; skipping normalization.");
                        return;
                    }

                    minValue = valid.Min(f => f.AiOdds!.Value);
                    maxValue = valid.Max(f => f.AiOdds!.Value);
                }
                else
                {
                    Console.WriteLine("\t\tPreserving raw AI outputs (market fallback disabled).");
                    Console.WriteLine("\t\tSkipping normalization to avoid fabricating probabilities from degenerate model output.");
                    return;
                }
            }

        NormalizationSummary:
            var sum = valid.Sum(f => f.AiOdds!.Value);
            Console.WriteLine($"\tNormalizeAiOdds: normalizing {valid.Count} runner(s) (missing={missingCount}, raw sum={sum.ToString("0.####", CultureInfo.InvariantCulture)}).");
            if (missingCount > 0)
            {
                Console.WriteLine($"\t\tMissing AI predictions detected for {missingCount} runner(s); surviving probabilities may be inflated relative to the full field.");
            }

            if (sum <= double.Epsilon)
            {
                var uniform = 1.0 / valid.Count;
                foreach (var flow in valid)
                {
                    flow.AiOdds = uniform;
                }
                Console.WriteLine($"\t\tRaw probabilities summed to ~0; distributing uniform probability {uniform.ToString("0.####", CultureInfo.InvariantCulture)} across valid runners.");
                return;
            }

            foreach (var flow in valid)
            {
                flow.AiOdds = Math.Max(flow.AiOdds!.Value / sum, 0);
            }

            var normalizedSum = valid.Sum(f => f.AiOdds!.Value);
            Console.WriteLine($"\t\tNormalized probability sum: {normalizedSum.ToString("0.####", CultureInfo.InvariantCulture)}.");
        }
        private static bool TryApplyLegacyFallback(ICollection<RunnerFlow> flows)
        {
            if (flows is null || flows.Count == 0)
            {
                return false;
            }

            const double tolerance = 1e-8;
            var legacyProbabilities = flows
                .Where(f => f.LegacyProbability.HasValue && double.IsFinite(f.LegacyProbability.Value) && f.LegacyProbability.Value > 0)
                .Select(f => f.LegacyProbability!.Value)
                .ToList();

            if (legacyProbabilities.Count == 0)
            {
                return false;
            }

            var minLegacy = legacyProbabilities.Min();
            var maxLegacy = legacyProbabilities.Max();
            if (maxLegacy - minLegacy <= tolerance)
            {
                return false;
            }

            foreach (var flow in flows)
            {
                if (flow.LegacyProbability.HasValue && double.IsFinite(flow.LegacyProbability.Value) && flow.LegacyProbability.Value > 0)
                {
                    flow.AiOdds = flow.LegacyProbability.Value;
                }
                else
                {
                    flow.AiOdds = null;
                    flow.AiProbabilityMarketDerived = false;
                }
            }

            Console.WriteLine("\t\tApplied legacy probability fallback due to degenerate trained model outputs.");
            return true;
        }
        private static bool TryResolveDegenerateDistribution(ICollection<RunnerFlow> flows)
        {
            if (flows is null || flows.Count == 0)
            {
                return false;
            }

            var winner = ResolveLikelyWinnerFromFeatures(flows);
            if (winner == null || !winner.AiOdds.HasValue || !double.IsFinite(winner.AiOdds.Value) || winner.AiOdds.Value <= 0)
            {
                return false;
            }

            var winnerProbability = winner.AiOdds.Value;
            const double probabilityFloor = 1e-6;
            bool substitutedWinnerProbability = false;
            if (winnerProbability <= probabilityFloor)
            {
                if (winner.BackPrice1.HasValue && winner.BackPrice1.Value > 1m)
                {
                    winnerProbability = 1.0 / (double)winner.BackPrice1.Value;
                    substitutedWinnerProbability = true;
                }
                else
                {
                    winnerProbability = probabilityFloor;
                }
            }
            var others = flows.Where(f => !ReferenceEquals(f, winner)).ToList();
            var leftoverMass = Math.Max(1.0 - winnerProbability, 0);

            if (others.Count > 0)
            {
                var weights = new Dictionary<RunnerFlow, double>();
                foreach (var flow in others)
                {
                    double weight = 0d;

                    if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 1m)
                    {
                        weight = 1.0 / (double)flow.BackPrice1.Value;
                    }
                    else if (flow.AiOdds.HasValue && double.IsFinite(flow.AiOdds.Value) && flow.AiOdds.Value > 0)
                    {
                        weight = flow.AiOdds.Value;
                    }
                    else
                    {
                        weight = 1d;
                    }

                    if (!double.IsFinite(weight) || weight < 0)
                    {
                        weight = 0d;
                    }

                    weights[flow] = weight;
                }

                var weightSum = weights.Values.Sum();
                if (weightSum <= double.Epsilon)
                {
                    var uniform = others.Count > 0 ? leftoverMass / others.Count : 0d;
                    foreach (var flow in others)
                    {
                        flow.AiOdds = uniform;
                    }
                }
                else
                {
                    foreach (var kvp in weights)
                    {
                        var share = leftoverMass * (kvp.Value / weightSum);
                        kvp.Key.AiOdds = Math.Max(share, 0d);
                    }
                }
            }

            winner.AiOdds = winnerProbability;

            var validAfter = flows
                .Where(f => f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value >= 0)
                .ToList();
            var missingAfter = flows.Count - validAfter.Count;
            if (missingAfter > 0)
            {
                Console.WriteLine($"\t\tMissing AI predictions detected for {missingAfter} runner(s); surviving probabilities may be inflated relative to the full field.");
            }

            var winnerName = !string.IsNullOrWhiteSpace(winner.HorseName)
                ? winner.HorseName!
                : (winner.SelectionId ?? "unknown");

            if (substitutedWinnerProbability)
            {
                Console.WriteLine($"\t\tWinner probability substituted with market-implied value {winnerProbability.ToString("0.####", CultureInfo.InvariantCulture)} for {winnerName}.");
            }
            else
            {
                Console.WriteLine($"\t\tInterpreting degenerate predictions as winner-only probability; assigning {winnerProbability.ToString("0.####", CultureInfo.InvariantCulture)} to {winnerName}.");
            }
            if (others.Count > 0 && leftoverMass > 0)
            {
                Console.WriteLine($"\t\tRedistributed remaining {leftoverMass.ToString("0.####", CultureInfo.InvariantCulture)} probability mass across {others.Count} runner(s) using market-derived weights.");
            }

            var normalized = validAfter.Sum(f => f.AiOdds!.Value);
            Console.WriteLine($"\t\tNormalized probability sum: {normalized.ToString("0.####", CultureInfo.InvariantCulture)}.");

            return true;
        }

        private static RunnerFlow? ResolveLikelyWinnerFromFeatures(IEnumerable<RunnerFlow> flows)
        {
            if (flows is null)
            {
                return null;
            }

            var flowList = flows.ToList();
            if (flowList.Count == 0)
            {
                return null;
            }

            var booleanKeys = new[] { "AiLikelyWinner", "LikelyWinner", "PredictedWinner", "IsAiWinner" };
            foreach (var flow in flowList)
            {
                if (flow.FeatureValues == null)
                {
                    continue;
                }

                foreach (var key in booleanKeys)
                {
                    if (IsFeatureTrue(flow.FeatureValues, key))
                    {
                        return flow;
                    }
                }
            }

            var selectionHint = FindFirstFeatureString(flowList, "AiLikelyWinnerSelectionId", "LikelyWinnerSelectionId", "PredictedWinnerSelectionId");
            if (!string.IsNullOrWhiteSpace(selectionHint))
            {
                var match = flowList.FirstOrDefault(f => !string.IsNullOrWhiteSpace(f.SelectionId) && string.Equals(f.SelectionId, selectionHint, StringComparison.OrdinalIgnoreCase));
                if (match != null)
                {
                    return match;
                }
            }

            var horseIdHint = FindFirstFeatureString(flowList, "AiLikelyWinnerHorseId", "LikelyWinnerHorseId", "PredictedWinnerHorseId");
            if (!string.IsNullOrWhiteSpace(horseIdHint))
            {
                foreach (var flow in flowList)
                {
                    if (flow.FeatureValues != null && flow.FeatureValues.TryGetValue("HorseId", out var horseIdObj) && horseIdObj != null)
                    {
                        var horseIdText = ConvertToInvariantString(horseIdObj);
                        if (!string.IsNullOrWhiteSpace(horseIdText) && string.Equals(horseIdText, horseIdHint, StringComparison.OrdinalIgnoreCase))
                        {
                            return flow;
                        }
                    }
                }
            }

            var horseNameHint = FindFirstFeatureString(flowList, "AiLikelyWinnerHorse", "AiLikelyWinnerHorseName", "LikelyWinnerHorse", "LikelyWinnerHorseName", "PredictedWinnerHorse", "PredictedWinnerHorseName");
            if (!string.IsNullOrWhiteSpace(horseNameHint))
            {
                var match = flowList.FirstOrDefault(f => !string.IsNullOrWhiteSpace(f.HorseName) && string.Equals(f.HorseName, horseNameHint, StringComparison.OrdinalIgnoreCase));
                if (match != null)
                {
                    return match;
                }
            }

            var marketFavourite = flowList
               .Where(f => f != null && f.AiOdds.HasValue && f.BackPrice1.HasValue && f.BackPrice1.Value > 1m)
               .OrderBy(f => f.BackPrice1.Value)
               .FirstOrDefault();
            if (marketFavourite != null)
            {
                return marketFavourite;
            }

            var aiCandidate = flowList
                .Where(f => f != null && f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value > 0)
                .OrderByDescending(f => f.AiOdds.Value)
                .FirstOrDefault();
            if (aiCandidate != null)
            {
                return aiCandidate;
            }

            var fallbackMarket = flowList
                .Where(f => f != null && f.BackPrice1.HasValue && f.BackPrice1.Value > 1m)
                .OrderBy(f => f.BackPrice1.Value)
                .FirstOrDefault();
            if (fallbackMarket != null)
            {
                return fallbackMarket;
            }

            return flowList.FirstOrDefault();
        }

        private static string ReadFirstNonEmptyText(IWebDriver driver, params string[] selectors)
        {
            if (driver == null || selectors == null || selectors.Length == 0)
            {
                return string.Empty;
            }

            foreach (var selector in selectors)
            {
                if (string.IsNullOrWhiteSpace(selector))
                {
                    continue;
                }

                var text = TextOrEmpty(driver, By.CssSelector(selector));
                if (!string.IsNullOrWhiteSpace(text))
                {
                    return text.Trim();
                }
            }

            return string.Empty;
        }

        private static string ExtractDocumentTitle(IWebDriver driver)
        {
            if (driver == null)
            {
                return string.Empty;
            }

            try
            {
                var documentTitle = driver.Title;
                if (string.IsNullOrWhiteSpace(documentTitle))
                {
                    return string.Empty;
                }

                var normalized = documentTitle.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .FirstOrDefault()?.Trim() ?? documentTitle.Trim();

                if (string.IsNullOrWhiteSpace(normalized))
                {
                    return string.Empty;
                }

                normalized = Regex.Replace(normalized, @"\s+-\s+betfair.*$", string.Empty, RegexOptions.IgnoreCase);
                return normalized.Trim();
            }
            catch
            {
                return string.Empty;
            }
        }
        private static bool IsFeatureTrue(Dictionary<string, object?> features, string key)
        {
            if (!features.TryGetValue(key, out var value) || value is null)
            {
                return false;
            }

            switch (value)
            {
                case bool b:
                    return b;
                case string s:
                    var trimmed = s.Trim();
                    if (bool.TryParse(trimmed, out var parsed))
                    {
                        return parsed;
                    }

                    if (double.TryParse(trimmed, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var numericFromString))
                    {
                        return Math.Abs(numericFromString) > double.Epsilon;
                    }

                    return false;
                default:
                    if (value is IConvertible convertible)
                    {
                        try
                        {
                            return convertible.ToBoolean(CultureInfo.InvariantCulture);
                        }
                        catch
                        {
                            try
                            {
                                var numeric = convertible.ToDouble(CultureInfo.InvariantCulture);
                                return Math.Abs(numeric) > double.Epsilon;
                            }
                            catch
                            {
                                return false;
                            }
                        }
                    }

                    return false;
            }
        }

        private static string? ConvertToInvariantString(object value)
        {
            return value switch
            {
                null => null,
                string s => s,
                IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
                _ => value.ToString()
            };
        }

        private static decimal? ParsePercentage(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var cleaned = text.Trim();
            if (cleaned.EndsWith("%", StringComparison.Ordinal))
            {
                cleaned = cleaned[..^1];
            }

            cleaned = cleaned.Replace("%", string.Empty);

            return ParseDecimal(cleaned);
        }

        private static (TimeSpan? Time, string? Venue, string? Country) ParseVenueDetails(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return (null, null, null);
            }

            var trimmed = text.Trim();
            var match = Regex.Match(trimmed, @"^(?<time>\d{1,2}:\d{2})\s+(?<venue>.*?)(?:\s+\((?<country>[A-Z]{2,})\))?$");

            if (match.Success)
            {
                TimeSpan? parsedTime = null;
                if (TimeSpan.TryParse(match.Groups["time"].Value, out var ts))
                {
                    parsedTime = ts;
                }

                var venueName = match.Groups["venue"].Value.Trim();
                var country = match.Groups["country"].Success ? match.Groups["country"].Value.Trim() : null;

                return (parsedTime, string.IsNullOrWhiteSpace(venueName) ? null : venueName, country);
            }

            return (null, trimmed, null);
        }

        private static DateTime? ParseEventDate(string text, DateTime today)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var trimmed = text.Trim();
            var tokens = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            string? dayToken = null;
            string? monthToken = null;
            string? yearToken = null;

            foreach (var token in tokens)
            {
                var clean = token.Trim();
                if (dayToken == null && int.TryParse(clean, NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
                {
                    dayToken = clean;
                    continue;
                }

                if (dayToken != null && monthToken == null)
                {
                    monthToken = clean.TrimEnd(',', '.');
                    continue;
                }

                if (dayToken != null && monthToken != null && yearToken == null && int.TryParse(clean, NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
                {
                    yearToken = clean;
                    break;
                }
            }

            if (dayToken != null && monthToken != null)
            {
                var hasExplicitYear = int.TryParse(yearToken, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedYear);
                var candidateYear = hasExplicitYear ? parsedYear : today.Year;
                var formats = new[] { "d MMM yyyy", "dd MMM yyyy", "d MMMM yyyy", "dd MMMM yyyy" };

                foreach (var format in formats)
                {
                    if (DateTime.TryParseExact($"{dayToken} {monthToken} {candidateYear}", format, CultureInfo.InvariantCulture, DateTimeStyles.None, out var candidate))
                    {
                        candidate = candidate.Date;
                        if (!hasExplicitYear)
                        {
                            if (candidate < today.AddDays(-7))
                            {
                                candidate = candidate.AddYears(1);
                            }
                            else if (candidate > today.AddDays(200))
                            {
                                candidate = candidate.AddYears(-1);
                            }
                        }
                        return candidate;
                    }
                }
            }

            if (DateTime.TryParse(trimmed, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out var fallback))
            {
                return fallback.Date;
            }

            return null;
        }

        private static string TextOrEmpty(IWebDriver d, By by)
        {
            try { return d.FindElement(by).Text.Trim(); }
            catch { return string.Empty; }
        }

        private static decimal? ParseDecimal(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var normalized = text
                .Replace('\u00a0', ' ')
                .Replace(",", string.Empty)
                .Trim();

            if (string.IsNullOrEmpty(normalized))
            {
                return null;
            }

            var priceMatch = Regex.Match(normalized, @"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])");
            if (priceMatch.Success
                && decimal.TryParse(priceMatch.Groups[1].Value, NumberStyles.Number, CultureInfo.InvariantCulture, out var decimalOdds))
            {
                return decimalOdds;
            }

            if (normalized.Contains('/'))
            {
                var slashParts = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                if (slashParts.Length == 2
                    && int.TryParse(slashParts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var numerator)
                    && int.TryParse(slashParts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var denominator)
                    && denominator != 0)
                {
                    var fractional = (decimal)numerator / denominator;
                    return fractional + 1m;
                }
            }

            return null;
        }

        private static byte? TryParseByte(string text)
        {
            return byte.TryParse(text, out var v) ? v : (byte?)null;
        }

        internal static string? ExtractMarketId(string url)
        {
            // Betfair have changed their URL structure over time.  In some cases the
            // market id appears in a traditional "market/1.234" or "marketId="
            // format, but newer "plus" pages render it as "...-betting-123456".
            // Support both patterns so scraping works regardless of the style of URL
            // that is loaded.
            var m = Regex.Match(url,
                @"(?:/market/|marketId=)(?<id1>[0-9.]+)|(?:betting-)(?<id2>\d+)");

            if (m.Groups["id1"].Success)
            {
                return m.Groups["id1"].Value;
            }

            if (m.Groups["id2"].Success)
            {
                return m.Groups["id2"].Value;
            }

            return null;
        }
    }
}