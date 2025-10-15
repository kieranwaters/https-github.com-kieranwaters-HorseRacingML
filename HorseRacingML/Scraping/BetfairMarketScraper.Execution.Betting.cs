using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Services;
using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        private IEnumerable<BetRecommendation> CreateRecommendations(
            IEnumerable<RunnerFlow> flows,
            string marketId,
            string? raceTitle,
            string? venueName,
            DateTime? raceDate,
            bool applyKellyDampener)
        {
            if (_availableBankroll <= 0m)
            {
                return Enumerable.Empty<BetRecommendation>();
            }
            var flowList = flows as IList<RunnerFlow> ?? flows.ToList();
            if (flowList.Count == 0)
            {
                return Enumerable.Empty<BetRecommendation>();
            }

            HistoricalRaceCountPrefetchResult? prefetchedCounts = null;
            try
            {
                var missingNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var flow in flowList)
                {
                    if (flow == null)
                    {
                        continue;
                    }

                    if (flow.HistoricalRaceCount.HasValue)
                    {
                        continue;
                    }

                    if (flow.FeatureValues != null &&
                        flow.FeatureValues.TryGetValue("CareerStarts", out var existingValue) &&
                        TryConvertToInt32(existingValue).HasValue)
                    {
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(flow.HorseName))
                    {
                        missingNames.Add(flow.HorseName);
                    }
                }

                if (missingNames.Count > 0)
                {
                    prefetchedCounts = _repo.GetHistoricalRaceCountsByHorseNames(missingNames);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tFailed to prefetch historical race counts: {ex.Message}");
            }

            foreach (var flow in flowList)
            {
                if (flow == null)
                {
                    continue;
                }

                if (flow.HistoricalRaceCount.HasValue)
                {
                    continue;
                }

                int? resolvedFromFeatures = null;
                if (flow.FeatureValues != null &&
                    flow.FeatureValues.TryGetValue("CareerStarts", out var historyValue))
                {
                    resolvedFromFeatures = TryConvertToInt32(historyValue);
                }

                if (resolvedFromFeatures.HasValue)
                {
                    flow.HistoricalRaceCount = resolvedFromFeatures;
                    continue;
                }

                var resolvedHistoryCount = ResolveHistoricalRaceCountFromPrefetch(flow, prefetchedCounts)
                    ?? ResolveHistoricalRaceCount(flow);

                if (resolvedHistoryCount.HasValue)
                {
                    flow.HistoricalRaceCount = resolvedHistoryCount;

                    if (flow.FeatureValues == null)
                    {
                        flow.FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                    }

                    flow.FeatureValues["CareerStarts"] = resolvedHistoryCount.Value;
                }
            }

            RunnerFlow? zeroHistoryRunner = null;
            foreach (var flow in flowList)
            {
                if (flow == null)
                {
                    continue;
                }

                int? historyCount = flow.HistoricalRaceCount;
                if (!historyCount.HasValue &&
                    flow.FeatureValues != null &&
                    flow.FeatureValues.TryGetValue("CareerStarts", out var historyValue))
                {
                    historyCount = TryConvertToInt32(historyValue);
                }

                if (historyCount.HasValue && historyCount.Value <= 0)
                {
                    zeroHistoryRunner = flow;
                    break;
                }
            }

            if (zeroHistoryRunner != null)
            {
                var identifier = DescribeRunner(zeroHistoryRunner);
                Console.WriteLine($"\tSkipping market {marketId}: runner {identifier} has zero recorded historical races; skipping bets for this race.");
                return Enumerable.Empty<BetRecommendation>();
            }

            var recommendations = new List<BetRecommendation>();

            foreach (var flow in flowList)
            {
                var identifier = !string.IsNullOrWhiteSpace(flow.HorseName)
                    ? flow.HorseName!
                    : "unknown";

                if (!flow.AiOdds.HasValue || !double.IsFinite(flow.AiOdds.Value) || flow.AiOdds.Value <= 0)
                {
                    Console.WriteLine($"\tSkipping {identifier}: AI probability unavailable or non-positive.");
                    continue;
                }

                if (!flow.BackPrice1.HasValue || flow.BackPrice1.Value <= 1m)
                {
                    var backText = flow.BackPrice1.HasValue
                        ? flow.BackPrice1.Value.ToString("0.##", CultureInfo.InvariantCulture)
                        : "null";
                    Console.WriteLine($"\tSkipping {identifier}: back price {backText} is not usable for value comparison.");
                    continue;
                }

                var decimalOdds = flow.BackPrice1.Value;
                var aiProbability = flow.AiOdds.Value;
                var marketProbability = 1.0 / (double)decimalOdds;
                var differential = aiProbability - marketProbability;

                int? runnerHistoryCount = flow.HistoricalRaceCount;
                if (!runnerHistoryCount.HasValue &&
                    flow.FeatureValues != null &&
                    flow.FeatureValues.TryGetValue("CareerStarts", out var flowHistoryValue))
                {
                    runnerHistoryCount = TryConvertToInt32(flowHistoryValue);
                }

                var historyText = runnerHistoryCount.HasValue
                    ? runnerHistoryCount.Value.ToString("N0", CultureInfo.InvariantCulture)
                    : "unknown";

                Console.WriteLine($"\tRunner {identifier}: decimalOdds={decimalOdds.ToString("0.##", CultureInfo.InvariantCulture)}, aiProb={aiProbability.ToString("0.####", CultureInfo.InvariantCulture)}, marketProb={marketProbability.ToString("0.####", CultureInfo.InvariantCulture)}, diff={differential.ToString("0.####", CultureInfo.InvariantCulture)}, historyCount={historyText}");

                if (differential <= 0)
                {
                    Console.WriteLine($"\t\tRejected {identifier}: differential {differential.ToString("0.####", CultureInfo.InvariantCulture)} is not positive after ignoring exchange commission.");
                    continue;
                }

                var kellyFraction = CalculateKellyFraction(aiProbability, (double)decimalOdds);
                if (applyKellyDampener && _kellyDampener < 1m)
                {
                    kellyFraction *= _kellyDampener;
                }

                if (kellyFraction > 1m)
                {
                    kellyFraction = 1m;
                }
                if (kellyFraction <= 0)
                {
                    Console.WriteLine($"\t\tRejected {identifier}: Kelly fraction {kellyFraction.ToString("0.####", CultureInfo.InvariantCulture)} is non-positive.");
                    continue;
                }

                Console.WriteLine($"\t\tAccepted {identifier}: Kelly fraction {kellyFraction.ToString("0.####", CultureInfo.InvariantCulture)} (commission not yet deducted).");
                var runnerKey = GetPrimaryRunnerMatchKey(flow.HorseName, flow.ClothNumber, flow.Draw, flow.JockeyName, flow.TrainerName);
                recommendations.Add(new BetRecommendation
                {
                    MarketId = marketId,
                    HorseName = flow.HorseName,
                    RunnerKey = runnerKey,
                    RaceTitle = raceTitle,
                    VenueName = venueName,
                    RaceDate = raceDate,
                    DecimalOdds = decimalOdds,
                    AiDecimalOdds = CalculateAiDecimalOdds(aiProbability),
                    AiProbability = aiProbability,
                    MarketProbability = marketProbability,
                    Differential = differential,
                    KellyFraction = kellyFraction,
                    Stake = 0m
                });
            }

            return recommendations;
        }
        private IReadOnlyList<BetRecommendation> ExecuteBackAllClicks(
            IWebDriver driver,
            IReadOnlyList<(IWebElement Row, RunnerFlow Flow)> runnerEntries,
            IReadOnlyList<BetRecommendation> recommendations)
        {
            if (runnerEntries.Count == 0 || recommendations.Count == 0)
            {
                return Array.Empty<BetRecommendation>();
            }

            var clicked = new List<BetRecommendation>();

            foreach (var recommendation in recommendations)
            {
                var identifier = recommendation.HorseName ?? recommendation.RunnerKey ?? "unknown";

                if (recommendation.Differential <= 0)
                {
                    Console.WriteLine($"\tSkipping {identifier}: differential {recommendation.Differential.ToString("0.####", CultureInfo.InvariantCulture)} <= 0.");
                    continue;
                }
                if (_availableBankroll <= 0m)
                {
                    Console.WriteLine($"\tBankroll exhausted before sizing stake for {identifier}.");
                    break;
                }

                if (recommendation.KellyFraction <= 0m)
                {
                    Console.WriteLine($"\tSkipping {identifier}: Kelly fraction {recommendation.KellyFraction.ToString("0.####", CultureInfo.InvariantCulture)} <= 0.");
                    continue;
                }

                Console.WriteLine($"\tSizing stake for {identifier}: bankroll {_availableBankroll.ToString("0.##", CultureInfo.InvariantCulture)}, Kelly {recommendation.KellyFraction.ToString("0.####", CultureInfo.InvariantCulture)}");
                var stake = CalculateStakeWithLimits(_availableBankroll, recommendation.KellyFraction);
                if (stake <= 0m)
                {
                    Console.WriteLine($"\t\tSequential Kelly returned zero stake for {identifier}; check rounding or Kelly cap constraints.");
                    continue;
                }
                var match = runnerEntries.FirstOrDefault(entry =>
                    DoesRecommendationMatchFlow(entry.Flow, recommendation));

                if (match.Row == null)
                {
                    Console.Error.WriteLine($"\tUnable to locate row for {identifier} to click Back-All");
                    continue;
                }

                identifier = DescribeRunner(match.Flow);

                if (TryClickBackAllButton(driver, match.Row, recommendation))
                {
                    var clickedRecommendation = recommendation with { Stake = stake };
                    clicked.Add(clickedRecommendation);
                    _availableBankroll -= stake;
                    if (_availableBankroll < 0m)
                    {
                        _availableBankroll = 0m;
                    }

                    Console.WriteLine($"\t\tStake {stake.ToString("0.##", CultureInfo.InvariantCulture)} accepted for {identifier}; bankroll now {_availableBankroll.ToString("0.##", CultureInfo.InvariantCulture)}");

                    Thread.Sleep(TimeSpan.FromMilliseconds(400));
                }
                else
                {
                    Console.WriteLine($"\t\tBack-All click failed or was skipped for {identifier}; bankroll remains {_availableBankroll.ToString("0.##", CultureInfo.InvariantCulture)}");
                }
            }
            return clicked;
        }
        private static bool DoesRecommendationMatchFlow(RunnerFlow? flow, BetRecommendation recommendation)
        {
            if (flow == null)
            {
                return false;
            }

            if (!string.IsNullOrWhiteSpace(recommendation.RunnerKey))
            {
                foreach (var key in BuildRunnerMatchKeys(flow.HorseName, flow.ClothNumber, flow.Draw, flow.JockeyName, flow.TrainerName))
                {
                    if (string.Equals(key, recommendation.RunnerKey, StringComparison.OrdinalIgnoreCase))
                    {
                        return true;
                    }
                }
            }

            if (AreNamesEquivalent(flow.HorseName, recommendation.HorseName))
            {
                return true;
            }

            return false;
        }
        private bool TryClickBackAllButton(IWebDriver driver, IWebElement row, BetRecommendation recommendation)
        {
            var identifier = recommendation.HorseName ?? recommendation.RunnerKey ?? "unknown";

            try
            {
                var button = FindBackAllButton(row);
                var js = (IJavaScriptExecutor)driver;

                if (button == null)
                {
                    if (TryClickBackAllViaScript(js, row))
                    {
                        Console.WriteLine($"\tClicked Back-All for {identifier} using script fallback");
                        return true;
                    }
                    else
                    {
                        Console.Error.WriteLine($"\tBack-All button not found for {identifier}");
                    }
                    return false;
                }

                try
                {
                    js.ExecuteScript("arguments[0].scrollIntoView({block:'center'});", button);
                }
                catch (Exception)
                {
                }

                try
                {
                    button.Click();
                    Console.WriteLine($"\tClicked Back-All for {identifier}");
                    return true;
                }
                catch (Exception)
                {
                }

                try
                {
                    js.ExecuteScript("arguments[0].click();", button);
                    Console.WriteLine($"\tClicked Back-All for {identifier} using JavaScript");
                    return true;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to click Back-All for {identifier}: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tUnexpected error clicking Back-All for {identifier}: {ex.Message}");
            }
            return false;
        }
        private static IWebElement? FindBackAllButton(IWebElement row)
        {
            static bool MatchesBackAll(IWebElement element)
            {
                if (element == null)
                {
                    return false;
                }
                bool IsBackAllContext(IWebElement el)
                {
                    IWebElement? current = el;
                    for (var depth = 0; depth < 6 && current != null; depth++)
                    {
                        try
                        {
                            var classAttribute = current.GetAttribute("class");
                            if (!string.IsNullOrWhiteSpace(classAttribute))
                            {
                                var lowered = classAttribute.ToLowerInvariant();
                                if (lowered.Contains("back-all"))
                                {
                                    return true;
                                }

                                if (lowered.Contains("bet-buttons") &&
                                    lowered.Contains("back-cell") &&
                                    lowered.Contains("last-back-cell"))
                                {
                                    return true;
                                }
                            }

                            var betType = current.GetAttribute("bet-type");
                            if (!string.IsNullOrWhiteSpace(betType) &&
                                betType.Equals("back", StringComparison.OrdinalIgnoreCase))
                            {
                                var handicap = current.GetAttribute("bet-handicap");
                                if (string.IsNullOrWhiteSpace(handicap) ||
                                    handicap.Equals("0", StringComparison.OrdinalIgnoreCase))
                                {
                                    return true;
                                }
                            }

                            current = TryFindElement(current, By.XPath(".."));
                        }
                        catch (Exception)
                        {
                            break;
                        }
                    }

                    return false;
                }

                bool IsBackButton(IWebElement el)
                {
                    try
                    {
                        var classAttribute = el.GetAttribute("class");
                        if (!string.IsNullOrWhiteSpace(classAttribute))
                        {
                            var lowered = classAttribute.ToLowerInvariant();
                            if (lowered.Contains("lay"))
                            {
                                return false;
                            }

                            if (lowered.Contains("back"))
                            {
                                return true;
                            }
                        }

                        var typeAttribute = el.GetAttribute("type");
                        if (!string.IsNullOrWhiteSpace(typeAttribute) &&
                            typeAttribute.Equals("back", StringComparison.OrdinalIgnoreCase))
                        {
                            return true;
                        }
                    }
                    catch (Exception)
                    {
                    }

                    return string.Equals(el.TagName, "button", StringComparison.OrdinalIgnoreCase);
                }

                string? ReadText(IWebElement el)
                {
                    var text = el.Text;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        return text;
                    }

                    var textContent = el.GetAttribute("textContent");
                    if (!string.IsNullOrWhiteSpace(textContent))
                    {
                        return textContent;
                    }

                    var innerText = el.GetAttribute("innerText");
                    return !string.IsNullOrWhiteSpace(innerText) ? innerText : null;
                }

                bool TextContainsBackAll(string? value)
                {
                    return !string.IsNullOrWhiteSpace(value) &&
                        value.IndexOf("back all", StringComparison.OrdinalIgnoreCase) >= 0;
                }

                if (TextContainsBackAll(ReadText(element)))
                {
                    return true;
                }

                foreach (var attribute in new[] { "aria-label", "title", "data-testid", "data-trackid", "data-action" })
                {
                    if (TextContainsBackAll(element.GetAttribute(attribute)))
                    {
                        return true;
                    }
                    var attributeValue = element.GetAttribute(attribute);
                    if (!string.IsNullOrWhiteSpace(attributeValue) &&
                        attributeValue.IndexOf("back-all", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        return true;
                    }
                }

                try
                {
                    var labelCandidates = element.FindElements(By.CssSelector("label, span, strong, div, p"));
                    foreach (var label in labelCandidates)
                    {
                        if (!ReferenceEquals(label, element) && TextContainsBackAll(ReadText(label)))
                        {
                            return true;
                        }
                    }
                }
                catch (Exception)
                {
                }

                return false;
            }

            IWebElement? ValidateCandidate(IWebElement? candidate)
            {
                if (candidate == null)
                {
                    return null;
                }

                if (MatchesBackAll(candidate))
                {
                    return candidate;
                }

                try
                {
                    var nestedButton = TryFindElement(candidate, By.TagName("button"));
                    if (nestedButton != null && !ReferenceEquals(nestedButton, candidate) && MatchesBackAll(nestedButton))
                    {
                        return nestedButton;
                    }
                }
                catch (Exception)
                {
                }

                return null;
            }

            var selectors = new[]
            {
                By.CssSelector("button[data-testid='back-all']"),
                By.CssSelector("button[data-testid='button-back-all']"),
                By.CssSelector("button[data-testid='back-all-button']"),
                By.CssSelector("button[data-testid*='back-all']"),
                By.CssSelector("button[aria-label*='back all' i]"),
                By.CssSelector("button[title*='back all' i]"),
                By.CssSelector("ours-price-button button[data-testid*='back-all']"),
                By.CssSelector("td:nth-of-type(4) ours-price-button button"),
                By.CssSelector("td.bet-buttons.back-cell.last-back-cell > ours-price-button > button"),
                By.CssSelector("td.bet-buttons.back-cell.last-back-cell ours-price-button button"),
                By.CssSelector("[data-testid='back-all'] button"),
                By.CssSelector("[data-testid*='back-all'] button"),
                By.CssSelector("button.back-all"),
                By.CssSelector("button.back-all-button"),
                By.CssSelector("button[class*='back-all']"),
            };

            foreach (var selector in selectors)
            {
                try
                {
                    var button = ValidateCandidate(TryFindElement(row, selector));
                    if (button != null)
                    {
                        return button;
                    }
                }
                catch (Exception)
                {
                }
            }

            try
            {
                var candidates = row.FindElements(By.CssSelector("ours-price-button button, button, [role='button'], .bet-button"));
                foreach (var candidate in candidates)
                {
                    var validated = ValidateCandidate(candidate);
                    if (validated != null)
                    {
                        return validated;
                    }
                }
            }
            catch (Exception)
            {
            }

            return null;
        }

        private void PopulateBetSlipStakes(
            IWebDriver driver,
            IReadOnlyList<BetRecommendation> recommendations,
            decimal startingBankroll)
        {
            if (recommendations.Count == 0)
            {
                return;
            }

            var expectedCount = _betSlipSelectionsFilled + recommendations.Count;
            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(5));
            try
            {
                wait.Until(d => FindBetSlipStakeInputs(d).Count >= expectedCount);
            }
            catch (WebDriverTimeoutException)
            {
                // proceed with whatever entries are available
            }

            var inputs = FindBetSlipStakeInputs(driver);

            if (inputs.Count < _betSlipSelectionsFilled)
            {
                _betSlipSelectionsFilled = 0;
            }

            if (inputs.Count <= _betSlipSelectionsFilled)
            {
                return;
            }

            var startIndex = _betSlipSelectionsFilled;
            var available = inputs.Skip(startIndex).Take(recommendations.Count).ToList();

            if (available.Count == 0)
            {
                return;
            }

            var js = (IJavaScriptExecutor)driver;
            var remainingPot = startingBankroll;

            for (var i = 0; i < available.Count && i < recommendations.Count; i++)
            {
                var recommendation = recommendations[i];
                var stake = CalculateStakeWithLimits(remainingPot, recommendation.KellyFraction);
                var input = available[i];
                var identifier = recommendation.HorseName ?? recommendation.RunnerKey ?? "unknown";
                Console.WriteLine($"\tBet slip allocation for {identifier}: remaining pot {remainingPot.ToString("0.##", CultureInfo.InvariantCulture)}, stake {stake.ToString("0.##", CultureInfo.InvariantCulture)}");
                try
                {
                    SetStakeInputValue(js, input, stake);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to populate stake input: {ex.Message}");
                }

                remainingPot -= stake;
                if (remainingPot < 0m)
                {
                    remainingPot = 0m;
                }
                Console.WriteLine($"\t\tRemaining pot after allocation: {remainingPot.ToString("0.##", CultureInfo.InvariantCulture)}");
            }

            _betSlipSelectionsFilled += available.Count;
            if (available.Count > 0 && TrySubmitBetSlip(driver))
            {
                _betSlipSelectionsFilled = 0;
            }
        }

        private bool TrySubmitBetSlip(IWebDriver driver)
        {
            if (driver == null)
            {
                return false;
            }

            try
            {
                var placeClicked = TryClickBetSlipButton(driver, PlaceBetsButtonSelector, "Place bets", TimeSpan.FromSeconds(3));
                if (!placeClicked)
                {
                    return false;
                }

                var confirmClicked = TryClickBetSlipButton(driver, ConfirmBetsButtonSelector, "Confirm bets", TimeSpan.FromSeconds(8));
                if (!confirmClicked)
                {
                    Console.WriteLine("\tConfirm bets button not clicked; verify slip manually.");
                }

                return confirmClicked;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tUnexpected error while submitting bet slip: {ex.Message}");
                return false;
            }
        }

        private bool TryClickBetSlipButton(IWebDriver driver, string selector, string description, TimeSpan timeout)
        {
            if (driver == null)
            {
                return false;
            }

            IWebElement? button = WaitForDisplayedElement(driver, selector, timeout);
            if (button == null)
            {
                Console.WriteLine($"\t{description} button not found within {timeout.TotalSeconds:0.#}s.");
                return false;
            }

            var js = driver as IJavaScriptExecutor;
            try
            {
                js?.ExecuteScript("arguments[0].scrollIntoView({block:'center'});", button);
            }
            catch (Exception)
            {
            }

            for (var attempt = 0; attempt < 2; attempt++)
            {
                try
                {
                    button.Click();
                    Console.WriteLine($"\tClicked {description} button.");
                    return true;
                }
                catch (StaleElementReferenceException)
                {
                    button = WaitForDisplayedElement(driver, selector, TimeSpan.FromSeconds(1));
                    if (button == null)
                    {
                        break;
                    }
                }
                catch (ElementClickInterceptedException)
                {
                    Thread.Sleep(TimeSpan.FromMilliseconds(200));
                }
                catch (Exception)
                {
                    break;
                }
            }

            if (js != null && button != null)
            {
                try
                {
                    js.ExecuteScript("arguments[0].click();", button);
                    Console.WriteLine($"\tClicked {description} button via JavaScript.");
                    return true;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to click {description} button via JavaScript: {ex.Message}");
                }
            }

            Console.WriteLine($"\tUnable to click {description} button.");
            return false;
        }

        private static IWebElement? WaitForDisplayedElement(IWebDriver driver, string selector, TimeSpan timeout)
        {
            if (driver == null)
            {
                return null;
            }

            var wait = new WebDriverWait(driver, timeout);
            wait.IgnoreExceptionTypes(typeof(NoSuchElementException), typeof(StaleElementReferenceException));

            try
            {
                return wait.Until(d =>
                {
                    try
                    {
                        var elements = d.FindElements(By.CssSelector(selector));
                        foreach (var element in elements)
                        {
                            if (element.Displayed && element.Enabled)
                            {
                                return element;
                            }
                        }

                        return null;
                    }
                    catch (StaleElementReferenceException)
                    {
                        return null;
                    }
                });
            }
            catch (WebDriverTimeoutException)
            {
                return null;
            }
        }
        private decimal CalculateStakeWithLimits(decimal bankroll, decimal kellyFraction)
        {
            var stake = BettingMath.CalculateSequentialStake(bankroll, kellyFraction);
            if (stake <= 0m)
            {
                return 0m;
            }

            var maxStake = DetermineMaxStake(bankroll);
            if (maxStake.HasValue && maxStake.Value > 0m && stake > maxStake.Value)
            {
                stake = maxStake.Value;
            }

            if (stake > bankroll)
            {
                stake = bankroll;
            }

            if (stake < 0m)
            {
                stake = 0m;
            }

            return decimal.Round(stake, 2, MidpointRounding.ToZero);
        }

        private decimal? DetermineMaxStake(decimal bankroll)
        {
            decimal? raw = _maxStakeMode switch
            {
                MaxStakeMode.PercentageOfBankroll when _maxStakePercentOfBankroll.HasValue && _maxStakePercentOfBankroll.Value > 0m
                    => bankroll * _maxStakePercentOfBankroll.Value,
                MaxStakeMode.FixedAmount when _maxStakeFixedAmount.HasValue && _maxStakeFixedAmount.Value > 0m
                    => _maxStakeFixedAmount.Value,
                _ => null
            };

            if (!raw.HasValue)
            {
                return null;
            }

            var capped = raw.Value;
            if (capped > bankroll)
            {
                capped = bankroll;
            }

            if (capped <= 0m)
            {
                return null;
            }

            var rounded = decimal.Round(capped, 2, MidpointRounding.ToZero);
            if (rounded <= 0m)
            {
                return null;
            }

            if (rounded < 1m && bankroll >= 1m)
            {
                rounded = 1m;
            }

            return rounded;
        }
        private decimal CalculateKellyFraction(double probability, double decimalOdds)
        {
            if (probability <= 0 || probability >= 1 || decimalOdds <= 1)
            {
                return 0m;
            }

            var b = decimalOdds - 1.0;
            if (Math.Abs(b) < double.Epsilon)
            {
                return 0m;
            }

            var q = 1.0 - probability;
            var fraction = (b * probability - q) / b;

            if (!double.IsFinite(fraction))
            {
                return 0m;
            }

            var result = (decimal)fraction;
            if (result < 0m)
            {
                result = 0m;
            }

            if (_maxKellyFraction.HasValue && result > _maxKellyFraction.Value)
            {
                result = _maxKellyFraction.Value;
            }

            if (result > 1m)
            {
                result = 1m;
            }

            return result;
        }
        private static string? FindFirstFeatureString(IEnumerable<RunnerFlow> flows, params string[] keys)
        {
            foreach (var flow in flows)
            {
                if (flow.FeatureValues == null)
                {
                    continue;
                }

                foreach (var key in keys)
                {
                    if (flow.FeatureValues.TryGetValue(key, out var value) && value != null)
                    {
                        var text = ConvertToInvariantString(value)?.Trim();
                        if (!string.IsNullOrWhiteSpace(text))
                        {
                            return text;
                        }
                    }
                }
            }

            return null;
        }


    }
}