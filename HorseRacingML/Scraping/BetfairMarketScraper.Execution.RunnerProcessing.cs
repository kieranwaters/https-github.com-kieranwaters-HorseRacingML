using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Tensorflow;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        private static readonly string[] RunnerClothSelectors =
        {
            ".cloth-number",
            ".runner-numbers .saddle-cloth",
            ".runner-numbers.double p.saddle-cloth",
            "p.saddle-cloth",
            ".saddle-cloth"
        };

        private static readonly string[] RunnerDrawSelectors =
        {
            ".draw",
            ".runner-numbers .draw",
            ".runner-numbers.double p.stall-draw",
            "p.stall-draw",
            ".stall-draw"
        };

        private static readonly string[] RunnerJockeySelectors =
        {
            ".name .jockey-name",
            ".runner-info-expanded [data-testid='horse-jockey']",
            ".runner-info-expanded [data-testid='runner-jockey']",
            "[data-testid='runner-jockey']",
            ".runner-timeform-wrapper__details--jockey"
        };

        private static readonly string[] RunnerTrainerSelectors =
        {
            ".runner-timeform-wrapper__horse-details .runner-timeform-wrapper__details.runner-timeform-wrapper__trainer",
            ".runner-timeform-wrapper__details.runner-timeform-wrapper__trainer",
            ".runner-expanded-details .runner-timeform-wrapper__details.runner-timeform-wrapper__trainer",
            ".runner-expanded-details .runner-timeform-wrapper__details--trainer",
            ".runner-info-expanded [data-testid='horse-trainer']",
            ".runner-info-expanded .runner-timeform-wrapper__details--trainer",
            "[data-testid='runner-trainer']",
            "[data-test-id='runner-trainer']"
        };
        private bool TryRefreshRunnerEntriesForBetting(
            IWebDriver driver,
            string marketId,
            string? raceUrl,
            IReadOnlyList<RunnerFlow> flows,
            out List<(IWebElement Row, RunnerFlow Flow)> refreshedEntries)
        {
            refreshedEntries = new List<(IWebElement Row, RunnerFlow Flow)>();

            if (driver == null)
            {
                return false;
            }
            string? originalUrl = null;
            try
            {
                originalUrl = driver.Url;
            }
            catch (WebDriverException)
            {
                originalUrl = raceUrl;
            }
            try
            {
                driver.Navigate().Refresh();
                Console.WriteLine($"\tRefreshed Betfair market {marketId} before executing bets to reload stake defaults.");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tFailed to refresh market {marketId} before executing bets: {ex.Message}");
                return false;
            }
            var expectedMarketId = marketId;
            string? refreshedUrl = null;
            try
            {
                refreshedUrl = driver.Url;
            }
            catch (WebDriverException)
            {
                refreshedUrl = null;
            }

            var refreshedMarketId = !string.IsNullOrWhiteSpace(refreshedUrl)
                ? ExtractMarketId(refreshedUrl)
                : null;

            if (string.IsNullOrWhiteSpace(refreshedMarketId) ||
                !string.Equals(refreshedMarketId, expectedMarketId, StringComparison.OrdinalIgnoreCase))
            {
                var targetUrl = !string.IsNullOrWhiteSpace(raceUrl) ? raceUrl : originalUrl;
                if (!string.IsNullOrWhiteSpace(targetUrl))
                {
                    try
                    {
                        driver.Navigate().GoToUrl(targetUrl);
                        Console.WriteLine($"\tReloaded market {marketId} after refresh redirected to {refreshedUrl ?? "unknown URL"}.");
                        refreshedUrl = driver.Url;
                    }
                    catch (Exception navEx)
                    {
                        Console.Error.WriteLine($"\tFailed to restore market {marketId} after refresh: {navEx.Message}");
                        return false;
                    }
                }
                else
                {
                    Console.Error.WriteLine($"\tUnable to determine race URL for market {marketId} after refresh redirected to {refreshedUrl ?? "unknown URL"}.");
                    return false;
                }
            }

            refreshedMarketId = !string.IsNullOrWhiteSpace(refreshedUrl)
                ? ExtractMarketId(refreshedUrl)
                : null;

            if (string.IsNullOrWhiteSpace(refreshedMarketId) ||
                !string.Equals(refreshedMarketId, expectedMarketId, StringComparison.OrdinalIgnoreCase))
            {
                Console.Error.WriteLine($"\tFailed to ensure market {marketId} was loaded after refresh (current url: {refreshedUrl ?? "unknown"}).");
                return false;
            }

            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(10));
            try
            {
                wait.Until(d => d.FindElements(By.CssSelector(".runner-line")).Count > 0);
            }
            catch (WebDriverTimeoutException)
            {
                Console.Error.WriteLine($"\tTimed out waiting for runner rows after refreshing market {marketId}.");
                return false;
            }

            var rows = driver.FindElements(By.CssSelector(".runner-line"));
            if (rows.Count == 0)
            {
                Console.Error.WriteLine($"\tNo runner rows found after refreshing market {marketId}.");
                return false;
            }
            var byMatchKey = new Dictionary<string, List<RunnerFlow>>(StringComparer.OrdinalIgnoreCase);
            var matchedFlows = new HashSet<RunnerFlow>();

            foreach (var flow in flows)
            {
                if (flow == null)
                {
                    continue;
                }

                foreach (var key in BuildRunnerMatchKeys(flow.HorseName, flow.ClothNumber, flow.Draw, flow.JockeyName, flow.TrainerName))
                {
                    if (!byMatchKey.TryGetValue(key, out var list))
                    {
                        list = new List<RunnerFlow>();
                        byMatchKey[key] = list;
                    }

                    if (!list.Contains(flow))
                    {
                        list.Add(flow);
                    }
                }
            }

            foreach (var row in rows)
            {
                var horseName = TryExtractRunnerName(row);
                var clothNumber = TryExtractRunnerNumber(row, RunnerClothSelectors);
                var draw = TryExtractRunnerNumber(row, RunnerDrawSelectors);
                var jockey = TryExtractRunnerDetail(row, RunnerJockeySelectors);
                var trainer = TryExtractRunnerDetail(row, RunnerTrainerSelectors);

                var candidateList = new List<RunnerFlow>();
                var seenCandidates = new HashSet<RunnerFlow>();

                foreach (var key in BuildRunnerMatchKeys(horseName, clothNumber, draw, jockey, trainer))
                {
                    if (!byMatchKey.TryGetValue(key, out var flowsByKey))
                    {
                        continue;
                    }

                    foreach (var candidate in flowsByKey)
                    {
                        if (candidate == null || matchedFlows.Contains(candidate))
                        {
                            continue;
                        }

                        if (seenCandidates.Add(candidate))
                        {
                            candidateList.Add(candidate);
                        }
                    }
                }

                if (candidateList.Count == 0 && !string.IsNullOrWhiteSpace(horseName))
                {
                    foreach (var flow in flows)
                    {
                        if (flow == null || matchedFlows.Contains(flow))
                        {
                            continue;
                        }

                        if (AreNamesEquivalent(flow.HorseName, horseName) && seenCandidates.Add(flow))
                        {
                            candidateList.Add(flow);
                        }
                    }
                }

                var matched = FindBestRunnerMatch(candidateList, horseName, clothNumber, draw, jockey, trainer);

                if (matched != null)
                {
                    refreshedEntries.Add((row, matched));
                    matchedFlows.Add(matched);

                    foreach (var key in BuildRunnerMatchKeys(matched.HorseName, matched.ClothNumber, matched.Draw, matched.JockeyName, matched.TrainerName))
                    {
                        if (!byMatchKey.TryGetValue(key, out var flowsByKey))
                        {
                            continue;
                        }

                        flowsByKey.Remove(matched);
                        if (flowsByKey.Count == 0)
                        {
                            byMatchKey.Remove(key);
                        }
                    }
                }
            }

            if (refreshedEntries.Count == 0)
            {
                Console.Error.WriteLine($"\tUnable to match refreshed runner rows to existing flows for market {marketId}.");
                return false;
            }

            return true;
        }

        private static string? TryExtractRunnerName(IWebElement row)
        {
            if (row == null)
            {
                return null;
            }

            var selectors = new[]
            {
                ".name .runner-name",
                "[data-testid='runner-name']",
                ".runner-name",
                ".name"
            };

            foreach (var selector in selectors)
            {
                try
                {
                    var element = TryFindElement(row, By.CssSelector(selector));
                    if (element == null)
                    {
                        continue;
                    }

                    var text = element.Text;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        return text.Trim();
                    }
                }
                catch (Exception)
                {
                }
            }

            return null;
        }

        private static byte? TryExtractRunnerNumber(IWebElement row, IReadOnlyList<string> selectors)
        {
            if (row == null || selectors == null)
            {
                return null;
            }

            foreach (var selector in selectors)
            {
                if (string.IsNullOrWhiteSpace(selector))
                {
                    continue;
                }

                try
                {
                    var element = TryFindElement(row, By.CssSelector(selector));
                    if (element == null)
                    {
                        continue;
                    }

                    var text = element.Text;
                    var parsed = TryParseByte(text);
                    if (parsed.HasValue)
                    {
                        return parsed;
                    }
                }
                catch (Exception)
                {
                }
            }

            return null;
        }

        private static string? TryExtractRunnerDetail(IWebElement row, IReadOnlyList<string> selectors)
        {
            if (row == null || selectors == null)
            {
                return null;
            }

            foreach (var selector in selectors)
            {
                if (string.IsNullOrWhiteSpace(selector))
                {
                    continue;
                }

                try
                {
                    var element = TryFindElement(row, By.CssSelector(selector));
                    if (element == null)
                    {
                        continue;
                    }

                    var text = element.Text;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        return text.Trim();
                    }
                }
                catch (Exception)
                {
                }
            }

            return null;
        }
        private static Dictionary<string, object?> CreateFeatureDictionary(IDictionary<string, object?>? source)
        {
            var result = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);

            if (source == null)
            {
                return result;
            }

            foreach (var entry in source)
            {
                if (entry.Key == null)
                {
                    continue;
                }

                var trimmedKey = entry.Key.Trim();
                if (trimmedKey.Length == 0)
                {
                    continue;
                }

                result[trimmedKey] = entry.Value;
            }

            return result;
        }
        private RaceDayReport BuildRaceReport(
            string marketId,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            DateTime? raceDate,
            TimeSpan? offTime,
            string? raceDetails,
            string? raceType,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? raceUrl,
            IEnumerable<RunnerFlow> flows)
        {
            var report = new RaceDayReport
            {
                MarketId = marketId,
                RaceTitle = string.IsNullOrWhiteSpace(raceTitle) ? null : raceTitle.Trim(),
                VenueName = string.IsNullOrWhiteSpace(venueName) ? null : venueName.Trim(),
                VenueCountry = string.IsNullOrWhiteSpace(venueCountry) ? null : venueCountry.Trim(),
                RaceDate = raceDate,
                OffTime = offTime,
                RaceDetails = string.IsNullOrWhiteSpace(raceDetails) ? null : raceDetails.Trim(),
                RaceType = string.IsNullOrWhiteSpace(raceType) ? null : raceType.Trim(),
                Going = string.IsNullOrWhiteSpace(going) ? null : going.Trim(),
                BackBookPercentage = backBookPercentage,
                LayBookPercentage = layBookPercentage,
                RaceUrl = string.IsNullOrWhiteSpace(raceUrl) ? null : raceUrl.Trim()
            };

            var runnerList = flows as IList<RunnerFlow> ?? flows.ToList();

            IReadOnlyDictionary<string, int>? prefetchedCounts = null;
            try
            {
                var missingNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var flow in runnerList)
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

            foreach (var flow in runnerList)
            {
                report.Runners.Add(CreateRunnerReport(flow, prefetchedCounts));
            }

            report.RaceFallbackSummary = BuildRaceFallbackSummary(runnerList);

            return report;
        }


        private static string? BuildRaceFallbackSummary(ICollection<RunnerFlow> flows)
        {
            if (flows is null || flows.Count == 0)
            {
                return null;
            }

            var validRunners = flows
                .Where(flow => flow != null)
                .ToList();

            if (validRunners.Count == 0)
            {
                return null;
            }

            var marketDerived = validRunners
                .Where(flow => flow.AiProbabilityMarketDerived)
                .ToList();

            if (marketDerived.Count == 0)
            {
                return null;
            }

            var reasons = new List<string>();
            var reasonSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var flow in marketDerived)
            {
                if (string.IsNullOrWhiteSpace(flow.AiProbabilityFallbackReason))
                {
                    continue;
                }

                var detail = flow.AiProbabilityFallbackReason.Trim();
                if (detail.Length == 0)
                {
                    continue;
                }

                if (reasonSet.Add(detail))
                {
                    reasons.Add(detail);
                }
            }

            var totalRunners = validRunners.Count;
            var baseMessage = BuildFallbackBaseMessage(totalRunners, marketDerived.Count);

            if (string.IsNullOrEmpty(baseMessage))
            {
                return null;
            }

            if (reasons.Count == 0)
            {
                return baseMessage;
            }

            var reasonLabel = reasons.Count == 1 ? "Reason" : "Reasons";
            return $"{baseMessage} {reasonLabel}: {string.Join("; ", reasons)}.";
        }

        private static string? BuildFallbackBaseMessage(int totalRunners, int marketDerivedCount)
        {
            if (totalRunners <= 0 || marketDerivedCount <= 0 || marketDerivedCount > totalRunners)
            {
                return null;
            }

            if (marketDerivedCount == totalRunners)
            {
                return "All AI win probabilities defaulted to market-implied odds for this race.";
            }

            var remaining = totalRunners - marketDerivedCount;
            var runnerWord = remaining == 1 ? "runner" : "runners";

            return $"{marketDerivedCount} of {totalRunners} AI win probabilities defaulted to market-implied odds for this race. The remaining {remaining} {runnerWord} retained their model-derived probabilities.";
        }


        private RunnerDayReport CreateRunnerReport(
            RunnerFlow flow,
            IReadOnlyDictionary<string, int>? prefetchedCounts)
        {
            var runner = new RunnerDayReport
            {
                ClothNumber = flow.ClothNumber,
                Draw = flow.Draw,
                HorseName = flow.HorseName,
                JockeyName = flow.JockeyName,
                FeatureValues = CreateFeatureDictionary(flow.FeatureValues),
                FeaturePopulation = (flow.FeaturePopulationSummary ?? FeaturePopulationSummary.Empty).WithSortedKeys()
            };

            PopulateRunnerPricing(flow, runner);
            void EnsureCareerStartsFeature(int count)
            {
                if (runner.FeatureValues == null)
                {
                    runner.FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                }

                runner.FeatureValues["CareerStarts"] = count;

                if (flow.FeatureValues == null)
                {
                    flow.FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                }

                flow.FeatureValues["CareerStarts"] = count;
            }

            int? historyCount = null;
            if (flow.HistoricalRaceCount.HasValue)
            {
                historyCount = flow.HistoricalRaceCount.Value;
            }

            int? featureHistoryCount = null;
            if (flow.FeatureValues != null &&
                flow.FeatureValues.TryGetValue("CareerStarts", out var historyValue))
            {
                featureHistoryCount = TryConvertToInt32(historyValue);
                if (!historyCount.HasValue)
                {
                    historyCount = featureHistoryCount;
                }
            }

            var requiresLookup = !historyCount.HasValue || historyCount.Value <= 0;
            if (requiresLookup)
            {
                var resolvedHistoryCount = ResolveHistoricalRaceCountFromPrefetch(flow, prefetchedCounts)
                    ?? ResolveHistoricalRaceCount(flow);

                if (resolvedHistoryCount.HasValue && resolvedHistoryCount.Value > 0)
                {
                    historyCount = resolvedHistoryCount.Value;
                }
            }

            if (!historyCount.HasValue && featureHistoryCount.HasValue)
            {
                historyCount = featureHistoryCount.Value;
            }

            if (historyCount.HasValue)
            {
                runner.HistoricalRaceCount = historyCount;
                flow.HistoricalRaceCount = historyCount;
                EnsureCareerStartsFeature(historyCount.Value);
            }
            return runner;
        }
        private void PopulateRunnerPricing(RunnerFlow flow, RunnerDayReport runner)
        {
            if (flow == null || runner == null)
            {
                return;
            }

            runner.AiProbabilityMarketDerived = flow.AiProbabilityMarketDerived;
            runner.AiProbabilityClampedToMarket = flow.AiProbabilityClampedToMarket;
            runner.AiProbabilityFallbackReason = flow.AiProbabilityFallbackReason;

            runner.MarketDecimalOdds = null;
            runner.MarketProbability = null;
            if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 0m)
            {
                runner.MarketDecimalOdds = flow.BackPrice1.Value;
                if (flow.BackPrice1.Value > 1m)
                {
                    runner.MarketProbability = 1.0 / (double)flow.BackPrice1.Value;
                }
            }

            runner.LayDecimalOdds = null;
            if (flow.LayPrice1.HasValue && flow.LayPrice1.Value > 0m)
            {
                runner.LayDecimalOdds = flow.LayPrice1.Value;
            }

            runner.AiProbability = null;
            runner.AiDecimalOdds = null;
            if (flow.AiOdds.HasValue && double.IsFinite(flow.AiOdds.Value))
            {
                var candidate = flow.AiOdds.Value;
                if (candidate > 0 && candidate <= 1)
                {
                    runner.AiProbability = candidate;
                    var decimalOdds = BettingMath.CalculateAiDecimalOdds(candidate);
                    runner.AiDecimalOdds = decimalOdds > 0m ? decimalOdds : null;
                }
            }

            runner.Differential = null;
            if (runner.AiProbability.HasValue && runner.MarketProbability.HasValue)
            {
                runner.Differential = runner.AiProbability.Value - runner.MarketProbability.Value;
            }

            runner.KellyFraction = null;
            runner.SuggestedStake = null;
            if (runner.AiProbability.HasValue && runner.MarketDecimalOdds.HasValue && runner.MarketDecimalOdds.Value > 1m)
            {
                var kelly = BettingMath.CalculateKellyFraction(
                    runner.AiProbability.Value,
                    (double)runner.MarketDecimalOdds.Value,
                    _maxKellyFraction);

                if (kelly > 0m)
                {
                    runner.KellyFraction = kelly;
                    if (_bankroll > 0m)
                    {
                        var stake = CalculateStakeWithLimits(_bankroll, kelly);
                        if (stake > 0m)
                        {
                            runner.SuggestedStake = stake;
                        }
                    }
                }
            }

            runner.LayKellyFraction = null;
            runner.LaySuggestedStake = null;
            if (runner.AiProbability.HasValue && runner.LayDecimalOdds.HasValue && runner.LayDecimalOdds.Value > 1m)
            {
                var layKelly = BettingMath.CalculateLayKellyFraction(
                    runner.AiProbability.Value,
                    (double)runner.LayDecimalOdds.Value,
                    _maxKellyFraction);

                if (layKelly > 0m)
                {
                    runner.LayKellyFraction = layKelly;

                    if (_bankroll > 0m)
                    {
                        var layStake = BettingMath.CalculateLayStake(_bankroll, layKelly, runner.LayDecimalOdds.Value);
                        if (layStake > 0m)
                        {
                            runner.LaySuggestedStake = layStake;
                        }
                    }
                }
            }
        }
        private static int? ResolveHistoricalRaceCountFromPrefetch(
            RunnerFlow flow,
            IReadOnlyDictionary<string, int>? prefetchedCounts)
        {
            if (prefetchedCounts == null || prefetchedCounts.Count == 0)
            {
                return null;
            }

            if (flow == null || string.IsNullOrWhiteSpace(flow.HorseName))
            {
                return null;
            }

            foreach (var candidate in RacingRepository.BuildHistoricalNameCandidates(flow.HorseName))
            {
                if (prefetchedCounts.TryGetValue(candidate, out var count))
                {
                    return count;
                }
            }

            return null;
        }
        private int? ResolveHistoricalRaceCount(RunnerFlow flow)
        {
            if (flow == null)
            {
                return null;
            }

            if (!string.IsNullOrWhiteSpace(flow.HorseName))
            {
                try
                {
                    return _repo.GetHistoricalRaceCountByHorseName(flow.HorseName);
                }
                catch (Exception ex)
                {
                    var identifier = DescribeRunner(flow);
                    Console.Error.WriteLine($"\tFailed to resolve historical race count for {identifier}: {ex.Message}");
                }
            }

            return null;
        }
        private static int? TryConvertToInt32(object? value)
        {
            if (value == null)
            {
                return null;
            }

            switch (value)
            {
                case int i:
                    return i;
                case long l when l <= int.MaxValue && l >= int.MinValue:
                    return (int)l;
                case short s:
                    return s;
                case byte b:
                    return b;
                case sbyte sb:
                    return sb;
                case ushort us when us <= int.MaxValue:
                    return (int)us;
                case uint ui when ui <= int.MaxValue:
                    return (int)ui;
                case float f when !float.IsNaN(f) && f <= int.MaxValue && f >= int.MinValue:
                    return (int)Math.Round(f);
                case double d when !double.IsNaN(d) && d <= int.MaxValue && d >= int.MinValue:
                    return (int)Math.Round(d);
                case decimal m when m <= int.MaxValue && m >= int.MinValue:
                    return (int)Math.Round(m);
                case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedInvariant):
                    return parsedInvariant;
                case string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.CurrentCulture, out var parsedCurrent):
                    return parsedCurrent;
            }

            if (value is IConvertible convertible)
            {
                try
                {
                    return convertible.ToInt32(CultureInfo.InvariantCulture);
                }
                catch
                {
                    // ignored
                }
            }

            return null;
        }
        private void PopulateFeatureVectors(
DateTime? raceDate,
            string? raceTitle,
            string? venueName,
            string? venueCountry,
            TimeSpan? scheduledOff,
            string? raceDetails,
            string? raceType,
            string? going,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            string? marketId,
            IReadOnlyList<RunnerFlow> flows,
            int runnerCount,
            IReadOnlyList<IDictionary<string, object?>>? preparedRows = null,
            UpcomingRace? persistedUpcoming = null)
        {
            if (flows == null || flows.Count == 0)
            {
                return;
            }
            var raceMissingFields = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var lastDistanceCache = new Dictionary<(int? HorseId, string NameKey), int?>();
            IReadOnlyDictionary<string, int>? prefetchedCounts = null;
            try
            {
                var missingNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var flow in flows)
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
            var featureLookup = LoadFeatureLookup(
                raceDate,
                raceTitle,
                venueName,
                venueCountry,
                scheduledOff,
                raceDetails,
                raceType,
                going,
                backBookPercentage,
                layBookPercentage,
                marketId,
                flows,
                preparedRows,
                persistedUpcoming);
            var fallbackLookup = FeatureLookup.Empty;
            var fallbackAttempted = false;
            foreach (var flow in flows)
            {
                var matchedFeatures = featureLookup.FindByRunner(flow);
                var matchedPreparedRow = matchedFeatures != null;
                var matchedDatabaseRow = matchedPreparedRow;
                var identifier = DescribeRunner(flow);

                if (!matchedPreparedRow)
                {
                    if (!fallbackAttempted)
                    {
                        fallbackAttempted = true;
                        fallbackLookup = BuildFallbackFeatureLookup(
                            raceDate,
                            raceTitle,
                            venueName,
                            venueCountry,
                            scheduledOff,
                            raceDetails,
                            raceType,
                            going,
                            backBookPercentage,
                            layBookPercentage,
                            marketId,
                            flows,
                            persistedUpcoming);
                    }

                    if (fallbackLookup != FeatureLookup.Empty)
                    {
                        matchedFeatures = fallbackLookup.FindByRunner(flow);
                        matchedPreparedRow = matchedFeatures != null;
                        if (matchedPreparedRow)
                        {
                            Console.WriteLine(
                                $"\t\tUsing trainer fallback feature vector for {identifier}; synthetic preparation succeeded.");
                        }
                    }
                }
                Dictionary<string, object?> featureVector;
                if (matchedPreparedRow)
                {
                    featureVector = CreateFeatureDictionary(matchedFeatures);
                }
                else
                {
                    featureVector = CreateFeatureDictionary(null);
                    var missingFeatureIdentifier = DescribeRunner(flow);
                    Console.WriteLine(
                        $"\t\tNo prepared feature row matched for {missingFeatureIdentifier}; synthesizing feature vector from live scrape.");
                }
                flow.MatchedDatabaseRecord = matchedDatabaseRow;
                var missingHistoricalKeys = GetMissingHistoricalFeatureKeys(featureVector);
                Dictionary<string, object?>? fallbackFeatures = null;
                var appliedFallback = false;
                if (missingHistoricalKeys.Count > 0)
                {
                    if (!fallbackAttempted)
                    {
                        fallbackAttempted = true;
                        fallbackLookup = BuildFallbackFeatureLookup(
                            raceDate,
                            raceTitle,
                            venueName,
                            venueCountry,
                            scheduledOff,
                            raceDetails,
                            raceType,
                            going,
                            backBookPercentage,
                            layBookPercentage,
                            marketId,
                            flows,
                            persistedUpcoming);
                    }

                    if (fallbackLookup != FeatureLookup.Empty)
                    {
                        var missingFeatureIdentifier = DescribeRunner(flow);

                        var fallbackCandidate = fallbackLookup.FindByRunner(flow)
                                 ?? (!string.IsNullOrWhiteSpace(flow?.HorseName)
                                     ? fallbackLookup.FindByHorse(flow.HorseName)
                                     : null);

                        if (fallbackCandidate != null)
                        {
                            fallbackFeatures = CreateFeatureDictionary(fallbackCandidate);
                            BackfillHistoricalFeatures(featureVector, fallbackFeatures);
                            appliedFallback = true;
                            missingHistoricalKeys = GetMissingHistoricalFeatureKeys(featureVector);
                        }

                        if (missingHistoricalKeys.Count > 0)
                        {
                            var missingSummary = string.Join(", ", missingHistoricalKeys);
                            Console.WriteLine(
                                $"\t\tUnable to backfill {missingHistoricalKeys.Count} historical feature(s) for {missingFeatureIdentifier}: {missingSummary}.");
                        }

                        ApplyScrapedFeatureFallbacks(
                            featureVector,
                            flow,
                            raceDate,
                            scheduledOff,
                            raceTitle,
                            raceDetails,
                            raceType,
                            going,
                            venueName,
                            venueCountry,
                            backBookPercentage,
                            layBookPercentage,
                            flows,
                            raceMissingFields);
                        EnsureDistanceChangeFromLast(
                            featureVector,
                            flow,
                            raceDate,
                            lastDistanceCache);
                        ApplyNeutralFeatureFallbacks(featureVector);

                        flow.FeatureValues = featureVector;
                        flow.HasPreparedFeatures = featureVector.Count > 0;
                        flow.FeaturePopulationSummary = BuildFeaturePopulationSummary(featureVector);

                        if (!matchedPreparedRow && flow.HasPreparedFeatures)
                        {
                            Console.WriteLine(
                                $"\t\tUsing scraped fallback feature vector for {identifier}; attempting neural model scoring with live data only.");
                        }

                        int? resolvedCareerStarts = null;
                        if (featureVector.TryGetValue("CareerStarts", out var careerStartsValue))
                        {
                            resolvedCareerStarts = TryConvertToInt32(careerStartsValue);
                        }

                        if (!resolvedCareerStarts.HasValue && flow.HistoricalRaceCount.HasValue)
                        {
                            resolvedCareerStarts = flow.HistoricalRaceCount.Value;
                        }

                        if (!resolvedCareerStarts.HasValue)
                        {
                            resolvedCareerStarts = ResolveHistoricalRaceCountFromPrefetch(flow, prefetchedCounts)
                                ?? ResolveHistoricalRaceCount(flow);
                        }

                        if (resolvedCareerStarts.HasValue)
                        {
                            featureVector["CareerStarts"] = resolvedCareerStarts.Value;
                            flow.HistoricalRaceCount = resolvedCareerStarts.Value;
                        }
                        else
                        {
                            flow.HistoricalRaceCount = null;
                        }
                    }
                    if (raceMissingFields.Count > 0)
                    {
                        foreach (var entry in raceMissingFields)
                        {
                            if (!string.IsNullOrWhiteSpace(entry))
                            {
                                _missingScrapedFieldDescriptions.Add(entry);
                            }
                        }
                    }
                }
            }
        }
        private static float? TryConvertToSingle(object? value)
        {
            if (value == null)
            {
                return null;
            }

            switch (value)
            {
                case float f when !float.IsNaN(f) && !float.IsInfinity(f):
                    return f;
                case double d when !double.IsNaN(d) && !double.IsInfinity(d):
                    if (d > float.MaxValue || d < float.MinValue)
                    {
                        return null;
                    }
                    return (float)d;
                case decimal m:
                    try
                    {
                        var convertedDecimal = (float)m;
                        if (!float.IsNaN(convertedDecimal) && !float.IsInfinity(convertedDecimal))
                        {
                            return convertedDecimal;
                        }
                    }
                    catch
                    {
                        // ignored
                    }
                    return null;
                case int i:
                    return i;
                case long l:
                    return l;
                case short s:
                    return s;
                case byte b:
                    return b;
                case sbyte sb:
                    return sb;
                case ushort us:
                    return us;
                case uint ui:
                    return ui;
                case string s when float.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var parsedInvariant):
                    return parsedInvariant;
                case string s when float.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.CurrentCulture, out var parsedCurrent):
                    return parsedCurrent;
            }

            if (value is IConvertible convertible)
            {
                try
                {
                    var converted = convertible.ToSingle(CultureInfo.InvariantCulture);
                    if (!float.IsNaN(converted) && !float.IsInfinity(converted))
                    {
                        return converted;
                    }
                }
                catch
                {
                    // ignored
                }
            }

            return null;
        }
        private static FeaturePopulationSummary BuildFeaturePopulationSummary(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return FeaturePopulationSummary.Empty;
            }

            var populated = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
            var missing = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var kvp in featureVector)
            {
                var key = kvp.Key;
                if (string.IsNullOrWhiteSpace(key))
                {
                    continue;
                }

                var trimmedKey = key.Trim();
                if (trimmedKey.Length == 0)
                {
                    continue;
                }

                if (HasMeaningfulValue(kvp.Value))
                {
                    populated.Add(trimmedKey);
                }
                else
                {
                    missing.Add(trimmedKey);
                }
            }

            foreach (var key in HistoricalFeatureBackfillKeys)
            {
                if (string.IsNullOrWhiteSpace(key))
                {
                    continue;
                }

                if (populated.Contains(key))
                {
                    continue;
                }

                if (!featureVector.TryGetValue(key, out var value) || !HasMeaningfulValue(value))
                {
                    missing.Add(key);
                }
            }

            return new FeaturePopulationSummary
            {
                PopulatedCount = populated.Count,
                MissingCount = missing.Count,
                PopulatedKeys = populated.Count > 0
                    ? populated.ToList()
                    : Array.Empty<string>(),
                MissingKeys = missing.Count > 0
                    ? missing.ToList()
                    : Array.Empty<string>()
            };
        }
        private static void EnsureRaceAverageWinRateLast5(IReadOnlyList<RunnerFlow> flows)
        {
            if (flows == null || flows.Count == 0)
            {
                return;
            }

            double sum = 0d;
            int participantCount = 0;

            foreach (var flow in flows)
            {
                if (flow == null)
                {
                    continue;
                }

                participantCount++;

                float runnerWinRate = 0f;
                if (flow.FeatureValues != null &&
                    TryGetMeaningfulValue(flow.FeatureValues, "WinRateLast5", out var winRateValue))
                {
                    var converted = TryConvertToSingle(winRateValue);
                    if (converted.HasValue && !float.IsNaN(converted.Value) && !float.IsInfinity(converted.Value))
                    {
                        runnerWinRate = converted.Value;
                    }
                }

                sum += runnerWinRate;
            }

            if (participantCount == 0)
            {
                return;
            }

            var average = (float)(sum / participantCount);
            if (float.IsNaN(average) || float.IsInfinity(average))
            {
                return;
            }

            foreach (var flow in flows)
            {
                if (flow?.FeatureValues == null)
                {
                    continue;
                }

                if (!TryGetMeaningfulValue(flow.FeatureValues, "RaceAvgWinRateLast5", out var existingValue))
                {
                    flow.FeatureValues["RaceAvgWinRateLast5"] = average;
                    continue;
                }

                var existing = TryConvertToSingle(existingValue);
                if (!existing.HasValue || float.IsNaN(existing.Value) || float.IsInfinity(existing.Value))
                {
                    flow.FeatureValues["RaceAvgWinRateLast5"] = average;
                }
            }
        }
        private static bool TryGetMeaningfulValue(
                    Dictionary<string, object?>? source,
                    string key,
                    out object? value)
        {
            value = null;
            if (source == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (!source.TryGetValue(key, out var existing) || !HasMeaningfulValue(existing))
            {
                return false;
            }

            if (HistoricalFeatureBackfillKeySet.Contains(key) && IsNeutralFallbackValue(key, existing))
            {
                return false;
            }

            value = existing;
            return true;
        }

        private static bool IsNeutralFallbackValue(string key, object? value)
        {
            if (value == null)
            {
                return false;
            }

            if (!NeutralFeatureFallbacks.TryGetValue(key, out var fallback) || fallback == null)
            {
                return false;
            }

            if (value is string valueText && fallback is string fallbackText)
            {
                return string.Equals(
                    valueText.Trim(),
                    fallbackText.Trim(),
                    StringComparison.OrdinalIgnoreCase);
            }

            if (value is IConvertible convertibleValue && fallback is IConvertible convertibleFallback)
            {
                try
                {
                    var numericValue = convertibleValue.ToDouble(CultureInfo.InvariantCulture);
                    var numericFallback = convertibleFallback.ToDouble(CultureInfo.InvariantCulture);
                    return Math.Abs(numericValue - numericFallback) < 1e-9;
                }
                catch
                {
                    // ignored - fall through to equality comparison
                }
            }

            return Equals(value, fallback);
        }

        private static readonly Lazy<string[]> HistoricalFeatureBackfillKeysLazy = new(() =>
             new[]
             {
                "Class",
                "RaceType",
                "AgeRestriction",
                "Surface",
                "Going",
                "DistanceYards",
                "DistanceText",
                "DistanceBucket",
                "BackBookPercentage",
                "LayBookPercentage",
                "RunnerCount",
                "HasLastWin",
                "DistanceChangeFromLast",
                "DistanceRatioFromAverage",
                "DaysSinceLastRace",
                "DaysSinceLastWin",
                "RacesSinceLastWin",
                "LayoffNormalized",
                "CareerStarts",
                "LifetimeWinRate",
                "DistanceBeatenLengths",
                "DistanceBeatenKnown",
                "DrawBias",
                "SaddleclothDiffFromMean",
                "IsTopWeight",
                "IsBottomWeight",
                "JockeyGoingDistanceWinRate",
                "JockeyGoingDistanceAvgNorm",
                "LastJockeyGoingDistanceNormPos",
                "TrainerJockeyCourseWinRate",
                "ClassWinRate",
                "ClassAvgNorm",
                "LastClassNormPos",
                "TrainerClassWinRate",
                "TrainerClassAvgNorm",
                "LastTrainerClassNormPos",
                "JockeyClassWinRate",
                "JockeyClassAvgNorm",
                "LastJockeyClassNormPos",
                "GoingCourseWinRate",
                "GoingCourseAvgNorm",
                "LastGoingCourseNormPos",
                "AgeRestrictionWinRate",
                "LastAgeRestrictionNormPos",
                "DistanceBucketWinRate",
                "LastDistanceBucketNormPos",
                "GoingDistanceWinRate",
                "GoingDistanceAvgNorm",
                "LastGoingDistanceNormPos",
                "RaceAvgWinRateLast5",
                "WinningTimeMs",
                "RaceSpeed",
                "RunnerSpeed",
                "SpeedMissing",
                "SpeedDiff",
                "SpeedRatio"
            }
            .Concat(PerformanceWindowPrefixes.SelectMany(prefix =>
                PerformanceWindowSizes.Select(window => prefix + window)))
            .ToArray());

        private static readonly Lazy<HashSet<string>> HistoricalFeatureBackfillKeySetLazy = new(() =>
            new HashSet<string>(HistoricalFeatureBackfillKeys, StringComparer.OrdinalIgnoreCase));

        private static string[] HistoricalFeatureBackfillKeys => HistoricalFeatureBackfillKeysLazy.Value;

        private static HashSet<string> HistoricalFeatureBackfillKeySet => HistoricalFeatureBackfillKeySetLazy.Value;

        private static readonly HashSet<string> HistoricalFeatureBackfillExcludedKeys = new(
            new[]
            {
                "HorseName",
                "HorseId",
                "TrainerName",
                "TrainerId",
                "JockeyName",
                "JockeyId",
                "RaceId",
                "RaceDate",
                "RunnerResultId",
                "MarketId"
            },
            StringComparer.OrdinalIgnoreCase);


        private static IReadOnlyList<string> GetMissingHistoricalFeatureKeys(Dictionary<string, object?> featureVector)
        {
            var missing = new List<string>();

            if (featureVector == null || featureVector.Count == 0)
            {
                missing.AddRange(HistoricalFeatureBackfillKeys);
                return missing;
            }

            foreach (var key in HistoricalFeatureBackfillKeys)
            {
                if (!TryGetMeaningfulValue(featureVector, key, out _))
                {
                    missing.Add(key);
                }
            }

            return missing;
        }
        private static void BackfillHistoricalFeatures(
            Dictionary<string, object?> target,
            Dictionary<string, object?> source)
        {
            if (target == null || source == null || source.Count == 0)
            {
                return;
            }

            foreach (var key in HistoricalFeatureBackfillKeys)
            {
                if (!TryGetMeaningfulValue(target, key, out _) &&
                    TryGetMeaningfulValue(source, key, out var replacement))
                {
                    target[key] = replacement;
                }
            }

            foreach (var kvp in source)
            {
                var key = kvp.Key;
                if (string.IsNullOrWhiteSpace(key))
                {
                    continue;
                }

                if (HistoricalFeatureBackfillExcludedKeys.Contains(key))
                {
                    continue;
                }

                if (!TryGetMeaningfulValue(target, key, out _) && HasMeaningfulValue(kvp.Value))
                {
                    target[key] = kvp.Value;
                }
            }
        }

        
        private static bool HasMeaningfulValue(object? value)
        {
            if (value == null)
            {
                return false;
            }

            if (value is string s)
            {
                return !string.IsNullOrWhiteSpace(s);
            }

            if (value is float f)
            {
                return !float.IsNaN(f);
            }

            if (value is double d)
            {
                return !double.IsNaN(d);
            }

            return true;
        }
        private void EnsureDistanceChangeFromLast(
            Dictionary<string, object?> featureVector,
            RunnerFlow flow,
            DateTime? raceDate,
            Dictionary<(int? HorseId, string NameKey), int?> cache)
        {
            if (featureVector == null || cache == null)
            {
                return;
            }

            if (TryGetMeaningfulValue(featureVector, "DistanceChangeFromLast", out _))
            {
                return;
            }

            if (!TryGetMeaningfulValue(featureVector, "DistanceYards", out var distanceObj))
            {
                return;
            }

            var currentDistance = TryConvertToInt32(distanceObj);
            if (!currentDistance.HasValue || currentDistance.Value <= 0)
            {
                return;
            }

            int? horseId = null;
            if (featureVector.TryGetValue("HorseId", out var horseIdValue))
            {
                horseId = TryConvertToInt32(horseIdValue);
            }

            if (!horseId.HasValue && flow?.FeatureValues != null &&
                flow.FeatureValues.TryGetValue("HorseId", out var flowHorseId))
            {
                horseId = TryConvertToInt32(flowHorseId);
            }

            string? horseName = flow?.HorseName;
            if (string.IsNullOrWhiteSpace(horseName) &&
                featureVector.TryGetValue("HorseName", out var horseObj) &&
                horseObj is string horseStr)
            {
                horseName = horseStr;
            }

            var cacheKey = (horseId, NormalizeHorseNameKeyForCache(horseName));
            if (!cache.TryGetValue(cacheKey, out var lastDistance))
            {
                try
                {
                    lastDistance = _repo.GetLastRaceDistance(horseName, horseId, raceDate);
                }
                catch (Exception ex)
                {
                    var identifier = !string.IsNullOrWhiteSpace(horseName)
                        ? horseName
                        : DescribeRunner(flow);
                    Console.Error.WriteLine($"  Failed to resolve last race distance for {identifier}: {ex.Message}");
                    lastDistance = null;
                }

                cache[cacheKey] = lastDistance;
            }

            if (lastDistance.HasValue)
            {
                featureVector["DistanceChangeFromLast"] = (float)(currentDistance.Value - lastDistance.Value);
            }
            else
            {
                featureVector["DistanceChangeFromLast"] = 0f;
            }
        }

        private static string NormalizeHorseNameKeyForCache(string? name)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                return string.Empty;
            }

            return name.Trim().ToLowerInvariant();
        }
        private static bool NeedsHistoricalFeatureBackfill(Dictionary<string, object?> featureVector) =>
            GetMissingHistoricalFeatureKeys(featureVector).Count > 0;

    }
}