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
        private const string FeatureFallbackPrefix = "\t[FeatureFallback]";

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
        private const float MsPerLength = 200f;
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
            var normalized = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);

            if (source != null)
            {
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

                    normalized[trimmedKey] = entry.Value;
                }
            }

            var result = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);

            if (normalized.Count == 0)
            {
                return result;
            }

            foreach (var key in normalized.Keys)
            {
                if (TryGetMeaningfulValue(normalized, key, out var meaningful))
                {
                    result[key] = meaningful;
                }
            }

            return result;
        }
        private static List<EncodedFeatureValue> CloneEncodedFeatureValues(IEnumerable<EncodedFeatureValue>? source)
        {
            var result = new List<EncodedFeatureValue>();

            if (source == null)
            {
                return result;
            }

            foreach (var value in source)
            {
                if (value == null)
                {
                    continue;
                }

                result.Add(new EncodedFeatureValue
                {
                    Index = value.Index,
                    FeatureKey = value.FeatureKey ?? string.Empty,
                    Label = value.Label ?? string.Empty,
                    Value = value.Value,
                    Active = value.Active
                });
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

            HistoricalRaceCountPrefetchResult? prefetchedCounts = null;
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

                    if (TryGetMeaningfulValue(flow.FeatureValues, "CareerStarts", out var existingValue) &&
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
                    var prefetchSummary = string.Format(
                        CultureInfo.InvariantCulture,
                        "\t[FeaturePopulation] Prefetched historical race counts for {0} runner name(s); matched {1} name(s) ({2} unique horse id(s)) spanning {3} historical race(s).",
                        missingNames.Count,
                        prefetchedCounts?.MatchedHorseCount ?? 0,
                        prefetchedCounts?.MatchedHorseIdCount ?? 0,
                        prefetchedCounts?.TotalHistoricalRaces ?? 0);
                    Console.WriteLine(prefetchSummary);
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
           HistoricalRaceCountPrefetchResult? prefetchedCounts)
        {
            var identifier = DescribeRunner(flow);
            if (flow.FeatureValues != null)
            {
                EnsureDisplayJockeyClassFallbacks(flow.FeatureValues, identifier);
            }
            var displayFeatureValues = CreateFeatureDictionary(flow.FeatureValues);
            if (displayFeatureValues != null && displayFeatureValues.Count > 0)
            {
                EnsureDisplayJockeyClassFallbacks(displayFeatureValues, identifier);
                LogDisplayJockeyClassFeatureOutcomes(displayFeatureValues, identifier);
            }
            var runner = new RunnerDayReport
            {
                ClothNumber = flow.ClothNumber,
                Draw = flow.Draw,
                HorseName = flow.HorseName,
                JockeyName = flow.JockeyName,
                EncodedFeatureValues = CloneEncodedFeatureValues(flow.EncodedFeatureValues),
                FeatureValues = displayFeatureValues,
                HasPreparedFeatures = flow.HasPreparedFeatures,
                HasPartialPreparedFeatures = flow.HasPartialPreparedFeatures,
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

            static string FormatHistoryCount(int? value) =>
                value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) : "<null>";

            if (flow.HistoricalRaceCount.HasValue)
            {
                historyCount = flow.HistoricalRaceCount.Value;
            }

            int? featureHistoryCount = null;
            if (TryGetMeaningfulValue(flow.FeatureValues, "CareerStarts", out var historyValue))
            {
                featureHistoryCount = TryConvertToInt32(historyValue);
                if (!historyCount.HasValue)
                {
                    historyCount = featureHistoryCount;
                }
            }

            Console.WriteLine(
                string.Format(
                    CultureInfo.InvariantCulture,
                    "\t[FeaturePopulation] Evaluating historical race count for {0}: flow={1}; feature={2}.",
                    identifier,
                    FormatHistoryCount(flow.HistoricalRaceCount),
                    FormatHistoryCount(featureHistoryCount)));

            var requiresLookup = !historyCount.HasValue || historyCount.Value <= 0;
            if (requiresLookup)
            {
                Console.WriteLine(
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "\t[FeaturePopulation] Historical race count missing or non-positive for {0}; attempting repository lookup.",
                        identifier));

                string? prefetchCandidate;
                var resolvedHistoryCount = ResolveHistoricalRaceCountFromPrefetch(flow, prefetchedCounts, out prefetchCandidate);
                string? resolutionSource = null;

                if (resolvedHistoryCount.HasValue && resolvedHistoryCount.Value > 0)
                {
                    resolutionSource = !string.IsNullOrWhiteSpace(prefetchCandidate)
                        ? string.Format(
                            CultureInfo.InvariantCulture,
                            "prefetch candidate '{0}'",
                            prefetchCandidate)
                        : "prefetch match";
                }
                else
                {
                    var repositoryResult = ResolveHistoricalRaceCount(flow, out var repositoryHorseName);
                    resolvedHistoryCount = repositoryResult;

                    if (resolvedHistoryCount.HasValue && resolvedHistoryCount.Value > 0)
                    {
                        resolutionSource = string.IsNullOrWhiteSpace(repositoryHorseName)
                            ? "repository lookup"
                            : string.Format(
                                CultureInfo.InvariantCulture,
                                "repository lookup for '{0}'",
                                repositoryHorseName);
                    }
                }

                if (resolvedHistoryCount.HasValue && resolvedHistoryCount.Value > 0)
                {
                    historyCount = resolvedHistoryCount.Value;
                    Console.WriteLine(
                        string.Format(
                            CultureInfo.InvariantCulture,
                            "\t[FeaturePopulation] {2} resolved historical race count {0} for {1}.",
                            historyCount.Value,
                            identifier,
                            resolutionSource ?? "Lookup"));
                }
                else
                {
                    Console.WriteLine(
                        string.Format(
                            CultureInfo.InvariantCulture,
                            "\t[FeaturePopulation] Repository lookup did not resolve a historical race count for {0}.",
                            identifier));
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
                if (historyCount.Value <= 0)
                {
                    Console.WriteLine(
                        string.Format(
                            CultureInfo.InvariantCulture,
                            "\t[FeaturePopulation] Warning: non-positive historical race count {0} captured for {1}.",
                            historyCount.Value,
                            identifier));
                }
                else
                {
                    Console.WriteLine(
                        string.Format(
                            CultureInfo.InvariantCulture,
                            "\t[FeaturePopulation] Using historical race count {0} for {1}; CareerStarts feature updated.",
                            historyCount.Value,
                            identifier));
                }
            }
            else
            {
                Console.WriteLine(
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "\t[FeaturePopulation] Historical race count unavailable for {0}; CareerStarts feature not populated.",
                        identifier));
            }
            return runner;
        }
        private static void LogFeatureFallback(string featureName, string message, string? runnerIdentifier = null)
        {
            var context = string.IsNullOrWhiteSpace(runnerIdentifier)
                ? string.Empty
                : $" ({runnerIdentifier})";
            Console.WriteLine(
                string.Format(
                    CultureInfo.InvariantCulture,
                    "{0} {1}{2}: {3}",
                    FeatureFallbackPrefix,
                    featureName,
                    context,
                    message));
        }
        private static string AppendResolvedValue(string message, float? value)
        {
            var formatted = value.HasValue
                ? value.Value.ToString("0.0000", CultureInfo.InvariantCulture)
                : "<null>";

            return string.Format(
                CultureInfo.InvariantCulture,
                "{0}; resolved value {1}",
                message,
                formatted);
        }
        private static void EnsureDisplayJockeyClassFallbacks(
            Dictionary<string, object?> featureValues,
            string? runnerIdentifier = null)
        {
            var jockeyClassWinRate = ResolveJockeyClassWinRate(featureValues, runnerIdentifier);
            var jockeyClassAvgNorm = ResolveJockeyClassAvgNorm(featureValues, jockeyClassWinRate, runnerIdentifier);
            var lastJockeyClass = ResolveLastJockeyClassNorm(featureValues, jockeyClassAvgNorm, runnerIdentifier);

            FillIfMissing(featureValues, "JockeyClassWinRate", jockeyClassWinRate);
            FillIfMissing(featureValues, "JockeyClassAvgNorm", jockeyClassAvgNorm);
            FillIfMissing(featureValues, "LastJockeyClassNormPos", lastJockeyClass);
        }
        private static void LogDisplayJockeyClassFeatureOutcomes(
            Dictionary<string, object?>? featureValues,
            string? runnerIdentifier)
        {
            if (featureValues == null || featureValues.Count == 0)
            {
                return;
            }

            LogFeatureFallback(
                "JockeyClassWinRate",
                BuildDisplayFeatureValueMessage(featureValues, "JockeyClassWinRate"),
                runnerIdentifier);

            LogFeatureFallback(
                "JockeyClassAvgNorm",
                BuildDisplayFeatureValueMessage(featureValues, "JockeyClassAvgNorm"),
                runnerIdentifier);

            LogFeatureFallback(
                "LastJockeyClassNormPos",
                BuildDisplayFeatureValueMessage(featureValues, "LastJockeyClassNormPos"),
                runnerIdentifier);
        }

        private static string BuildDisplayFeatureValueMessage(
            Dictionary<string, object?> featureValues,
            string key)
        {
            if (featureValues == null)
            {
                return "final day report value unavailable (feature dictionary null)";
            }

            if (!featureValues.TryGetValue(key, out var rawValue))
            {
                return "final day report value <missing>";
            }

            var converted = TryConvertToSingle(rawValue);
            var formatted = converted.HasValue
                ? converted.Value.ToString("0.0000", CultureInfo.InvariantCulture)
                : rawValue?.ToString() ?? "<null>";

            var fallbackSuffix = rawValue != null && IsNeutralFallbackValue(featureValues, key, rawValue)
                ? " (neutral fallback applied)"
                : string.Empty;

            return string.Format(
                CultureInfo.InvariantCulture,
                "final day report value {0}{1}",
                formatted,
                fallbackSuffix);
        }
        private static float? ResolveJockeyClassWinRate(
            Dictionary<string, object?> featureValues,
            string? runnerIdentifier = null)
        {
            if (featureValues == null)
            {
                LogFeatureFallback(
                    "JockeyClassWinRate",
                    "feature collection unavailable; unable to resolve win rate",
                    runnerIdentifier);
                return null;
            }

            var primaryWinRate = ResolveFeatureValue(featureValues, "JockeyWinRate")
                ?? ResolveFeatureValue(featureValues, "JockeyWinRateRecentDays")
                ?? ResolveFeatureValue(featureValues, "JockeyWinRateLast50");

            if (primaryWinRate.HasValue)
            {
                LogFeatureFallback(
                    "JockeyClassWinRate",
                    AppendResolvedValue(
                        "resolved from jockey-level win rate metrics (JockeyWinRate/JockeyWinRateRecentDays/JockeyWinRateLast50)",
                        primaryWinRate),
                    runnerIdentifier);
                return primaryWinRate;
            }

            var blendedWinRate = CombineAverages(
                ResolveFeatureValue(featureValues, "JockeyGoingWinRate"),
                ResolveFeatureValue(featureValues, "JockeyDistanceBucketWinRate"))
                ?? CombineAverages(
                    ResolveFeatureValue(featureValues, "JockeySurfaceWinRate"),
                    ResolveFeatureValue(featureValues, "JockeyCourseWinRate"));

            if (blendedWinRate.HasValue)
            {
                LogFeatureFallback(
                    "JockeyClassWinRate",
                    "resolved by blending going/surface/distance win rates",
                    runnerIdentifier);
                return blendedWinRate;
            }

            var trainerJockeyWinRate = ResolveNonNeutralFeatureValue(featureValues, "TrainerJockeyWinRate");
            if (trainerJockeyWinRate.HasValue)
            {
                LogFeatureFallback(
                    "JockeyClassWinRate",
                    AppendResolvedValue(
                        "resolved from trainer-jockey combination win rate",
                        trainerJockeyWinRate),
                    runnerIdentifier);
                return trainerJockeyWinRate;
            }

            var fallbackWinRate = ResolveNonNeutralFeatureValue(featureValues, "ClassWinRate")
                ?? ResolveNonNeutralFeatureValue(featureValues, "TrainerClassWinRate")
                ?? ResolveNonNeutralFeatureValue(featureValues, "TrainerWinRate");

            if (fallbackWinRate.HasValue)
            {
                LogFeatureFallback(
                    "JockeyClassWinRate",
                    AppendResolvedValue(
                        "resolved from class/trainer win rate fallbacks",
                        fallbackWinRate),
                    runnerIdentifier);
                return fallbackWinRate;
            }
            var generalWinRate = ResolveGeneralWinRateFallback(featureValues);
            if (generalWinRate.HasValue)
            {
                LogFeatureFallback(
                    "JockeyClassWinRate",
                    AppendResolvedValue(
                        "resolved from general performance win rate metrics (WinRateLast*/LifetimeWinRate)",
                        generalWinRate),
                    runnerIdentifier);
                return generalWinRate;
            }
            LogFeatureFallback(
                "JockeyClassWinRate",
                AppendResolvedValue(
                    "no jockey, trainer, or class win rate statistics available; using neutral default",
                    0f),
                runnerIdentifier);
            return null;
        }

        private static float? ResolveJockeyClassAvgNorm(
            Dictionary<string, object?> featureValues,
            float? jockeyClassWinRate,
            string? runnerIdentifier = null)
        {
            if (featureValues == null)
            {
                LogFeatureFallback(
                    "JockeyClassAvgNorm",
                    "feature collection unavailable; unable to resolve average normalized finish",
                    runnerIdentifier);
                return null;
            }

            var surfaceAvg = ResolveNonNeutralFeatureValue(featureValues, "JockeySurfaceAvgNorm");
            var goingAvg = ResolveNonNeutralFeatureValue(featureValues, "JockeyGoingAvgNorm");
            var distanceAvg = ResolveNonNeutralFeatureValue(featureValues, "JockeyDistanceBucketAvgNorm");
            var goingDistanceAvg = ResolveNonNeutralFeatureValue(featureValues, "JockeyGoingDistanceAvgNorm");

            var blendedAvg = CombineAverages(surfaceAvg, goingAvg)
                ?? CombineAverages(distanceAvg, goingDistanceAvg);

            var resolvedAvg = blendedAvg
                ?? surfaceAvg
                ?? goingAvg
                ?? distanceAvg
                ?? goingDistanceAvg;

            var resolutionLogged = false;

            if (!resolvedAvg.HasValue && jockeyClassWinRate.HasValue)
            {
                var derived = ClampNormalizedPosition(1f - jockeyClassWinRate.Value);
                LogFeatureFallback(
                    "JockeyClassAvgNorm",
                    AppendResolvedValue(
                        "derived from resolved jockey class win rate because no normalized finish history was available",
                        derived),
                    runnerIdentifier);
                resolvedAvg = derived;
                resolutionLogged = true;
            }

            if (!resolvedAvg.HasValue)
            {
                resolvedAvg = ResolveNonNeutralFeatureValue(featureValues, "ClassAvgNorm")
                    ?? ResolveNonNeutralFeatureValue(featureValues, "TrainerClassAvgNorm");
            }

            if (resolvedAvg.HasValue)
            {
                if (!resolutionLogged)
                {
                    if (resolvedAvg == blendedAvg)
                    {
                        LogFeatureFallback(
                            "JockeyClassAvgNorm",
                            AppendResolvedValue(
                                "resolved by blending available normalized finish metrics",
                                resolvedAvg),
                            runnerIdentifier);
                    }
                    else if (resolvedAvg == surfaceAvg || resolvedAvg == goingAvg || resolvedAvg == distanceAvg || resolvedAvg == goingDistanceAvg)
                    {
                        LogFeatureFallback(
                            "JockeyClassAvgNorm",
                            AppendResolvedValue(
                                "resolved from a single jockey normalized finish metric (surface/going/distance)",
                                resolvedAvg),
                            runnerIdentifier);
                    }
                    else
                    {
                        LogFeatureFallback(
                            "JockeyClassAvgNorm",
                            AppendResolvedValue(
                                "resolved from class/trainer normalized finish fallbacks",
                                resolvedAvg),
                            runnerIdentifier);
                    }
                }

                return resolvedAvg;
            }
            var generalAvgNorm = ResolveGeneralNormalizedFinishFallback(featureValues, jockeyClassWinRate);
            if (generalAvgNorm.HasValue)
            {
                LogFeatureFallback(
                    "JockeyClassAvgNorm",
                    AppendResolvedValue(
                        "resolved from general performance normalized finish metrics (AvgNormPosLast*/WinRateLast*)",
                        generalAvgNorm),
                    runnerIdentifier);
                return generalAvgNorm;
            }
            LogFeatureFallback(
                "JockeyClassAvgNorm",
                AppendResolvedValue(
                    "no jockey, trainer, or class normalized finish statistics available; using neutral default",
                    0f),
                runnerIdentifier);
            return null;
        }
        private static float? ResolveLastJockeyClassNorm(
            Dictionary<string, object?> featureValues,
            float? jockeyClassAvgNorm,
            string? runnerIdentifier = null)
        {
            if (featureValues == null)
            {
                return null;
            }

            var lastSurface = ResolveNonNeutralFeatureValue(featureValues, "LastJockeySurfaceNormPos");
            var lastGoing = ResolveNonNeutralFeatureValue(featureValues, "LastJockeyGoingNormPos");
            var lastDistance = ResolveNonNeutralFeatureValue(featureValues, "LastJockeyDistanceBucketNormPos");
            var lastGoingDistance = ResolveNonNeutralFeatureValue(featureValues, "LastJockeyGoingDistanceNormPos");

            var resolved = lastSurface
                ?? lastGoing
                ?? lastDistance
                ?? lastGoingDistance
                ?? CombineAverages(lastGoing, lastDistance);

            if (!resolved.HasValue && jockeyClassAvgNorm.HasValue)
            {
                var derived = ClampNormalizedPosition(jockeyClassAvgNorm.Value);
                LogFeatureFallback(
                    "LastJockeyClassNormPos",
                    AppendResolvedValue(
                        "derived from resolved jockey class average because no last-position history was available",
                        derived),
                    runnerIdentifier);
                resolved = derived;
            }

            if (resolved.HasValue)
            {
                return resolved;
            }

            var lastClassNorm = ResolveNonNeutralFeatureValue(featureValues, "LastClassNormPos")
                ?? ResolveNonNeutralFeatureValue(featureValues, "LastTrainerClassNormPos");

            if (lastClassNorm.HasValue)
            {
                LogFeatureFallback(
                    "LastJockeyClassNormPos",
                    AppendResolvedValue(
                        "resolved from class/trainer last normalized finish fallbacks",
                        lastClassNorm),
                    runnerIdentifier);
            }

            if (lastClassNorm.HasValue)
            {
                return lastClassNorm;
            }

            LogFeatureFallback(
                "LastJockeyClassNormPos",
                AppendResolvedValue(
                    "no jockey, trainer, or class last-position statistics available; using neutral default",
                    0f),
                runnerIdentifier);

            return null;
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
            runner.AiTrainedModelApplied = flow.AiTrainedModelApplied;
            runner.AiUsedLegacyModel = flow.AiUsedLegacyModel;
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
           HistoricalRaceCountPrefetchResult? prefetchedCounts,
           out string? matchedCandidate)
        {
            matchedCandidate = null;

            if (prefetchedCounts == null || prefetchedCounts.IsEmpty)
            {
                return null;
            }

            if (flow == null || string.IsNullOrWhiteSpace(flow.HorseName))
            {
                return null;
            }

            foreach (var candidate in RacingRepository.BuildHistoricalNameCandidates(flow.HorseName))
            {
                if (prefetchedCounts.TryGetCountByCandidate(candidate, out var count))
                {
                    matchedCandidate = candidate;
                    return count;
                }
            }

            return null;
        }
        private int? ResolveHistoricalRaceCount(RunnerFlow flow, out string? resolvedHorseName)
        {
            resolvedHorseName = null;

            if (flow == null)
            {
                return null;
            }

            if (!string.IsNullOrWhiteSpace(flow.HorseName))
            {
                try
                {
                    resolvedHorseName = flow.HorseName;
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
        private void ResolveCareerStartsFeature(
           Dictionary<string, object?>? featureVector,
           RunnerFlow? flow,
           HistoricalRaceCountPrefetchResult? prefetchedCounts,
           string identifier)
        {
            if (flow == null)
            {
                return;
            }

            int? resolvedCareerStarts = null;
            float? resolvedLifetimeWinRate = null;
            string? lifetimeSource = null;
            if (featureVector != null &&
                TryGetMeaningfulValue(featureVector, "CareerStarts", out var careerStartsValue))
            {
                var parsedCareerStarts = TryConvertToInt32(careerStartsValue);
                if (parsedCareerStarts.HasValue && parsedCareerStarts.Value > 0)
                {
                    resolvedCareerStarts = parsedCareerStarts.Value;
                }
            }
            if (featureVector != null &&
                TryGetMeaningfulValue(featureVector, "LifetimeWinRate", out var lifetimeValue))
            {
                var parsedLifetime = TryConvertToSingle(lifetimeValue);
                if (parsedLifetime.HasValue)
                {
                    resolvedLifetimeWinRate = parsedLifetime.Value;
                    lifetimeSource = "feature vector";
                }
            }

            if (!resolvedLifetimeWinRate.HasValue &&
                flow?.FeatureValues != null &&
                TryGetMeaningfulValue(flow.FeatureValues, "LifetimeWinRate", out var flowLifetimeValue))
            {
                var parsedLifetime = TryConvertToSingle(flowLifetimeValue);
                if (parsedLifetime.HasValue)
                {
                    resolvedLifetimeWinRate = parsedLifetime.Value;
                    lifetimeSource = "existing flow value";
                }
            }
            if (resolvedCareerStarts.HasValue)
            {
                var existingSource = flow.HistoricalRaceCount.HasValue
                    ? "existing flow value"
                    : "feature vector";

                if (!flow.HistoricalRaceCount.HasValue)
                {
                    flow.HistoricalRaceCount = resolvedCareerStarts;
                }

                if (featureVector != null)
                {
                    featureVector["CareerStarts"] = resolvedCareerStarts.Value;
                }
                if (!resolvedLifetimeWinRate.HasValue)
                {
                    resolvedLifetimeWinRate = _trainer.ComputeSmoothedWinRate(0, resolvedCareerStarts.Value);
                    lifetimeSource = "default smoothing";
                }

                AssignLifetimeWinRate(featureVector, flow, resolvedLifetimeWinRate.Value);
                return;
            }

            if (flow.HistoricalRaceCount.HasValue)
            {
                resolvedCareerStarts = flow.HistoricalRaceCount.Value;
                if (featureVector != null)
                {
                    featureVector["CareerStarts"] = resolvedCareerStarts.Value;
                }

                Console.WriteLine(
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "\t\t[FeaturePopulation] Resolved career starts for {0}: {1} (source: existing flow value).",
                        identifier,
                        resolvedCareerStarts.Value));
                if (!resolvedLifetimeWinRate.HasValue)
                {
                    resolvedLifetimeWinRate = _trainer.ComputeSmoothedWinRate(0, resolvedCareerStarts.Value);
                    lifetimeSource = "existing flow value";
                }

                AssignLifetimeWinRate(featureVector, flow, resolvedLifetimeWinRate.Value);
                Console.WriteLine(
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "\t\t[FeaturePopulation] Resolved lifetime win rate for {0}: {1:F4} (source: {2}).",
                        identifier,
                        resolvedLifetimeWinRate.Value,
                        lifetimeSource ?? "existing flow value"));
                return;
            }

            string? prefetchCandidate;
            var resolvedFromPrefetch = ResolveHistoricalRaceCountFromPrefetch(flow, prefetchedCounts, out prefetchCandidate);
            string? resolutionSource = null;
            int? resolvedCareerWins = null;
            if (resolvedFromPrefetch.HasValue)
            {
                resolvedCareerStarts = resolvedFromPrefetch;
                resolutionSource = !string.IsNullOrWhiteSpace(prefetchCandidate)
                    ? string.Format(
                        CultureInfo.InvariantCulture,
                        "prefetch candidate '{0}'",
                        prefetchCandidate)
                    : "prefetch match";
                if (!string.IsNullOrWhiteSpace(prefetchCandidate) &&
                    prefetchedCounts != null &&
                    prefetchedCounts.TryGetWinCountByCandidate(prefetchCandidate, out var wins))
                {
                    resolvedCareerWins = wins;
                    if (resolvedCareerStarts.Value >= 0)
                    {
                        resolvedLifetimeWinRate = _trainer.ComputeSmoothedWinRate(wins, resolvedCareerStarts.Value);
                        lifetimeSource = resolutionSource;
                    }
                }
            }
            else
            {
                var repositoryResult = ResolveHistoricalRaceCount(flow, out var repositoryHorseName);
                if (repositoryResult.HasValue)
                {
                    resolvedCareerStarts = repositoryResult;
                    resolutionSource = string.IsNullOrWhiteSpace(repositoryHorseName)
                        ? "repository lookup"
                        : string.Format(
                            CultureInfo.InvariantCulture,
                        "repository lookup for '{0}'",
                        repositoryHorseName);

                    if (!resolvedLifetimeWinRate.HasValue)
                    {
                        var lookupName = string.IsNullOrWhiteSpace(repositoryHorseName)
                            ? flow.HorseName
                            : repositoryHorseName;

                        if (!string.IsNullOrWhiteSpace(lookupName))
                        {
                            try
                            {
                                var wins = _repo.GetHistoricalWinCountByHorseName(lookupName);
                                resolvedCareerWins = wins;
                                resolvedLifetimeWinRate = _trainer.ComputeSmoothedWinRate(wins, resolvedCareerStarts.Value);
                                lifetimeSource = resolutionSource;
                            }
                            catch (Exception ex)
                            {
                                Console.Error.WriteLine(
                                    string.Format(
                                        CultureInfo.InvariantCulture,
                                        "\tFailed to resolve lifetime win rate for {0} via repository lookup: {1}",
                                        identifier,
                                        ex.Message));
                            }
                        }
                    }
                }
            }

            if (!resolvedCareerStarts.HasValue)
            {
                return;
            }

            if (featureVector != null)
            {
                featureVector["CareerStarts"] = resolvedCareerStarts.Value;
            }

            flow.HistoricalRaceCount = resolvedCareerStarts.Value;

            var sourceSuffix = string.IsNullOrWhiteSpace(resolutionSource)
                ? string.Empty
                : string.Format(
                    CultureInfo.InvariantCulture,
                    " (source: {0})",
                    resolutionSource);

            Console.WriteLine(
                $"\t\t[FeaturePopulation] Resolved career starts for {identifier}: {resolvedCareerStarts.Value}{sourceSuffix}.");
            if (!resolvedLifetimeWinRate.HasValue)
            {
                if (resolvedCareerWins.HasValue)
                {
                    resolvedLifetimeWinRate = _trainer.ComputeSmoothedWinRate(resolvedCareerWins.Value, resolvedCareerStarts.Value);
                    lifetimeSource = resolutionSource;
                }
                else
                {
                    resolvedLifetimeWinRate = _trainer.ComputeSmoothedWinRate(0, resolvedCareerStarts.Value);
                    lifetimeSource = string.IsNullOrWhiteSpace(resolutionSource)
                        ? "default smoothing"
                        : resolutionSource;
                }
            }

            AssignLifetimeWinRate(featureVector, flow, resolvedLifetimeWinRate.Value);
            Console.WriteLine(
                string.Format(
                    CultureInfo.InvariantCulture,
                    "\t\t[FeaturePopulation] Resolved lifetime win rate for {0}: {1:F4} (source: {2}).",
                    identifier,
                    resolvedLifetimeWinRate.Value,
                    lifetimeSource ?? resolutionSource ?? "default smoothing"));
        }

        private static void AssignLifetimeWinRate(
            Dictionary<string, object?>? featureVector,
            RunnerFlow? flow,
            float lifetimeWinRate)
        {
            if (featureVector != null)
            {
                featureVector["LifetimeWinRate"] = lifetimeWinRate;
            }

            if (flow == null)
            {
                return;
            }

            if (flow.FeatureValues == null)
            {
                flow.FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
            }

            flow.FeatureValues["LifetimeWinRate"] = lifetimeWinRate;

            EnsureWinRatePerformanceWindows(featureVector, flow.FeatureValues, lifetimeWinRate);
        }

        private static void EnsureWinRatePerformanceWindows(
            Dictionary<string, object?>? featureVector,
            Dictionary<string, object?> flowFeatureValues,
            float lifetimeWinRate)
        {
            if (featureVector != null)
            {
                EnsureWinRatePerformanceWindows(featureVector, lifetimeWinRate);
            }

            if (!ReferenceEquals(featureVector, flowFeatureValues))
            {
                EnsureWinRatePerformanceWindows(flowFeatureValues, lifetimeWinRate);
            }
        }

        private static void EnsureWinRatePerformanceWindows(Dictionary<string, object?> target, float lifetimeWinRate)
        {
            if (target == null)
            {
                return;
            }

            foreach (var window in PerformanceWindowSizes)
            {
                var winRateKey = $"WinRateLast{window}";
                if (!TryGetMeaningfulValue(target, winRateKey, out _))
                {
                    target[winRateKey] = lifetimeWinRate;
                }

                var normKey = $"AvgNormPosLast{window}";
                if (!TryGetMeaningfulValue(target, normKey, out _))
                {
                    target[normKey] = ClampNormalizedPosition(1f - lifetimeWinRate);
                }
            }
        }
        private void RestorePersistedHistoricalFeatures(
            Dictionary<string, object?> featureVector,
            Dictionary<string, object?>? persistedFeatures,
            string identifier)
        {
            if (featureVector == null || persistedFeatures == null || persistedFeatures.Count == 0)
            {
                return;
            }

            var trackedKeys = ResolveTrackedFeatureKeys();
            if (trackedKeys == null || trackedKeys.Count == 0)
            {
                return;
            }

            var restoredKeys = new List<string>();
            var restoredLookup = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var key in trackedKeys)
            {
                if (string.IsNullOrWhiteSpace(key))
                {
                    continue;
                }

                if (string.Equals(key, "CareerStarts", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (TryGetMeaningfulValue(featureVector, key, out _))
                {
                    continue;
                }

                if (!TryGetMeaningfulValue(persistedFeatures, key, out var persistedValue))
                {
                    continue;
                }

                featureVector[key] = persistedValue;

                if (restoredLookup.Add(key))
                {
                    restoredKeys.Add(key);
                }
            }

            if (restoredKeys.Count > 0)
            {
                Console.WriteLine(
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "\t\t[FeaturePopulation] Restored {0} persisted feature(s) for {1}: {2}.",
                        restoredKeys.Count,
                        identifier,
                        string.Join(", ", restoredKeys)));
            }
        }

        private static void EnsureHasLastWinFlag(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return;
            }

            if (TryGetMeaningfulValue(featureVector, "HasLastWin", out _))
            {
                return;
            }

            if (featureVector.TryGetValue("DaysSinceLastWin", out var daysObj))
            {
                var daysSince = TryConvertToInt32(daysObj);
                if (daysSince.HasValue)
                {
                    featureVector["HasLastWin"] = daysSince.Value >= 0;
                    return;
                }
            }

            if (featureVector.TryGetValue("RacesSinceLastWin", out var racesObj))
            {
                var racesSince = TryConvertToInt32(racesObj);
                if (racesSince.HasValue)
                {
                    featureVector["HasLastWin"] = racesSince.Value >= 0;

                    return;
                }
            }

            featureVector["HasLastWin"] = false;
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
            UpcomingRace? persistedUpcoming = null)
        {
            var populationContext = string.Format(
                CultureInfo.InvariantCulture,
                "\t[FeaturePopulation] PopulateFeatureVectors invoked. MarketId={0}, RaceDate={1}, Title='{2}', Venue='{3}', RunnerCount={4}.",
                marketId ?? "<null>",
                raceDate?.ToString(CultureInfo.InvariantCulture) ?? "<null>",
                raceTitle ?? "<null>",
                venueName ?? "<null>",
                flows?.Count ?? 0);

            Console.WriteLine(populationContext);

            if (flows == null || flows.Count == 0)
            {
                Console.WriteLine("\t[FeaturePopulation] No runner flows provided; skipping feature population.");
                return;
            }
            var lastDistanceCache = new Dictionary<(int? HorseId, string NameKey), int?>();
            HistoricalRaceCountPrefetchResult? prefetchedCounts = null;
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
                    var prefetchSummary = string.Format(
                        CultureInfo.InvariantCulture,
                        "\t[FeaturePopulation] Prefetched historical race counts for {0} runner name(s); matched {1} name(s) ({2} unique horse id(s)) spanning {3} historical race(s).",
                        missingNames.Count,
                        prefetchedCounts?.MatchedHorseCount ?? 0,
                        prefetchedCounts?.MatchedHorseIdCount ?? 0,
                        prefetchedCounts?.TotalHistoricalRaces ?? 0);
                    Console.WriteLine(prefetchSummary);
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
                //preparedRows,
                persistedUpcoming);
            var primaryLookupSummary = string.Format(
                CultureInfo.InvariantCulture,
                "\t[FeaturePopulation] Primary feature lookup contains {0} runner(s).",
                featureLookup.Count);
            Console.WriteLine(primaryLookupSummary);
            var fallbackLookup = FeatureLookup.Empty;
            var fallbackAttempted = false;
            foreach (var flow in flows)
            {
                if (flow != null)
                {
                    flow.HasPartialPreparedFeatures = false;
                }
                var persistedFeatureSnapshot = flow?.FeatureValues != null && flow.FeatureValues.Count > 0
                    ? CreateFeatureDictionary(flow.FeatureValues)
                    : null;
                var matchedFeatures = featureLookup.FindByRunner(flow);
                var matchedLookupRow = matchedFeatures != null;
                var identifier = DescribeRunner(flow);

                if (!matchedLookupRow)
                {
                    Console.WriteLine($"\t\t[FeaturePopulation] No primary lookup row matched for {identifier}; fallback evaluation pending.");
                }

                if (!matchedLookupRow)
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
                        var fallbackStatus = fallbackLookup == FeatureLookup.Empty
                            ? BuildMissingHistoricalFeatureReason()
                            : string.Format(
                                CultureInfo.InvariantCulture,
                                "trainer lookup produced {0} runner feature set(s).",
                                fallbackLookup.Count);
                        Console.WriteLine($"\t\t[FeaturePopulation] Trainer fallback attempt completed; result={fallbackStatus}");
                    }

                    if (fallbackLookup != FeatureLookup.Empty)
                    {
                        matchedFeatures = fallbackLookup.FindByRunner(flow);
                        matchedLookupRow = matchedFeatures != null;
                        if (matchedLookupRow)
                        {
                            Console.WriteLine(
                                $"\t\tUsing trainer fallback feature vector for {identifier}; synthetic preparation succeeded.");
                        }
                    }
                }
                Dictionary<string, object?> featureVector;
                bool usedTrainerFallback = false;
                if (matchedLookupRow)
                {
                    featureVector = CreateFeatureDictionary(matchedFeatures);
                }
                else
                {
                    featureVector = CreateFeatureDictionary(null);
                    Console.WriteLine(
                        $"\t\tNo lookup feature row matched for {identifier}; feature vector requires database backfill.");
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
                    missingScrapeFields: null,
                    persistedFeatures: persistedFeatureSnapshot);
                RestorePersistedHistoricalFeatures(featureVector, persistedFeatureSnapshot, identifier);
                var missingHistoricalKeys = GetMissingHistoricalFeatureKeys(featureVector);
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
                        var fallbackCandidate = fallbackLookup.FindByRunner(flow)
                                 ?? (!string.IsNullOrWhiteSpace(flow?.HorseName)
                                     ? fallbackLookup.FindByHorse(flow.HorseName)
                                     : null);

                        if (fallbackCandidate != null)
                        {
                            var fallbackFeatures = CreateFeatureDictionary(fallbackCandidate);
                            if (fallbackFeatures.Count > 0)
                            {
                                var mergedFallback = CreateFeatureDictionary(fallbackFeatures);
                                BackfillHistoricalFeatures(mergedFallback, featureVector);
                                ApplyScrapedFeatureFallbacks(
                                    mergedFallback,
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
                                    missingScrapeFields: null,
                                    persistedFeatures: persistedFeatureSnapshot);
                                featureVector = mergedFallback;
                                usedTrainerFallback = true;
                                missingHistoricalKeys = GetMissingHistoricalFeatureKeys(featureVector);

                                if (missingHistoricalKeys.Count == 0)
                                {
                                    Console.WriteLine(
                                        $"\t\tUsing trainer fallback feature vector for {identifier}; database preparation succeeded.");
                                }
                            }
                        }
                    }
                    EnsureDistanceBeatenFromNumeric(featureVector, flow);
                }

                RestorePersistedHistoricalFeatures(featureVector, persistedFeatureSnapshot, identifier);
                ResolveCareerStartsFeature(featureVector, flow, prefetchedCounts, identifier);
                EnsureHasLastWinFlag(featureVector);
                missingHistoricalKeys = GetMissingHistoricalFeatureKeys(featureVector);

                if (missingHistoricalKeys.Count > 0)
                {
                    var missingSummary = string.Join(", ", missingHistoricalKeys);
                    Console.WriteLine(
                        $"\t\tProceeding with {missingHistoricalKeys.Count} missing historical feature(s) for {identifier}: {missingSummary}.");
                    var missingDetails = DescribeMissingHistoricalFeatureDetails(featureVector);
                    foreach (var detail in missingDetails)
                    {
                        Console.WriteLine(
                            string.Format(
                                CultureInfo.InvariantCulture,
                                "\t\t\t[FeaturePopulation] Missing {0} for {1}.",
                                detail,
                                identifier));
                    }
                    EnsureDistanceBeatenFromNumeric(featureVector, flow);
                    EnsureWinningTimeFromRace(featureVector, flow);
                    EnsureSpeedMetrics(featureVector);
                    EnsureHasLastWinFlag(featureVector);
                    Console.WriteLine(
                        "\t\t[FeaturePopulation] Historical backfill remained incomplete; continuing with available data.");
                    var hasFeatureVector = featureVector.Count > 0;
                    flow.FeatureValues = featureVector;
                    flow.HasPreparedFeatures = featureVector.Count > 0;
                    flow.HasPartialPreparedFeatures = flow.HasPreparedFeatures;
                    flow.FeaturePopulationSummary = BuildFeaturePopulationSummary(featureVector);
                    flow.MatchedDatabaseRecord = matchedLookupRow || usedTrainerFallback;
                    flow.AiProbabilityFallbackReason = BuildMissingHistoricalFeatureReason();
                    continue;
                }
                if (missingHistoricalKeys.Count == 0 &&
                    AreAlwaysRequiredHistoricalFeaturesSatisfied(featureVector))
                {
                    Console.WriteLine(
                        $"\t\t[FeaturePopulation] HistoricalFeaturesAlwaysRequired satisfied for {identifier}.");
                }
                EnsureDistanceBeatenFromNumeric(featureVector, flow);
                EnsureWinningTimeFromRace(featureVector, flow);
                EnsureSpeedMetrics(featureVector);
                EnsureDistanceChangeFromLast(
                    featureVector,
                    flow,
                    raceDate,
                    lastDistanceCache);

                flow.FeatureValues = featureVector;
                flow.HasPreparedFeatures = featureVector.Count > 0;
                flow.HasPartialPreparedFeatures = false;
                flow.FeaturePopulationSummary = BuildFeaturePopulationSummary(featureVector);
                flow.MatchedDatabaseRecord = matchedLookupRow || usedTrainerFallback;

                if (!matchedLookupRow && !usedTrainerFallback)
                {
                    Console.WriteLine(
                        $"\t\tSkipping AI scoring for {identifier} due to missing database-backed features.");
                    Console.WriteLine(
                        $"\t\t[FeaturePopulation] Runner missing features summary: {BuildFeaturePopulationSummary(featureVector)}.");
                    flow.HasPreparedFeatures = false;
                    flow.HasPartialPreparedFeatures = false;
                    flow.AiProbabilityFallbackReason = BuildMissingHistoricalFeatureReason();
                    continue;
                }

            }
            EnsureRaceAverageWinRateLast5(flows);
            EnsureRaceAverageSpeedLast5(flows);
            EnsureRunnerClassAndGoingDistanceFeatures(flows);

            foreach (var flow in flows)
            {
                if (flow?.FeatureValues == null)
                {
                    continue;
                }

                flow.FeaturePopulationSummary = BuildFeaturePopulationSummary(flow.FeatureValues);
            }
        }
        private static IReadOnlyList<string> DescribeMissingHistoricalFeatureDetails(Dictionary<string, object?> featureVector)
        {
            var trackedKeys = ResolveTrackedFeatureKeys();
            var details = new List<string>();

            if (trackedKeys == null || trackedKeys.Count == 0)
            {
                return details;
            }

            if (featureVector == null || featureVector.Count == 0)
            {
                foreach (var key in trackedKeys)
                {
                    if (string.IsNullOrWhiteSpace(key))
                    {
                        continue;
                    }

                    details.Add(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0} (feature vector unavailable)",
                        key.Trim()));
                }

                return details;
            }

            foreach (var rawKey in trackedKeys)
            {
                if (string.IsNullOrWhiteSpace(rawKey))
                {
                    continue;
                }

                var key = rawKey.Trim();

                if (!ShouldRequireFeature(featureVector, key))
                {
                    continue;
                }

                if (TryGetMeaningfulValue(featureVector, key, out _))
                {
                    continue;
                }

                var reason = DescribeMissingHistoricalFeatureReason(featureVector, key);
                details.Add(string.Format(
                    CultureInfo.InvariantCulture,
                    "{0} ({1})",
                    key,
                    reason));
            }

            return details;
        }

        private static string DescribeMissingHistoricalFeatureReason(Dictionary<string, object?> featureVector, string key)
        {
            if (featureVector == null || featureVector.Count == 0)
            {
                return "feature vector unavailable";
            }

            if (!featureVector.TryGetValue(key, out var rawValue))
            {
                return "value not present";
            }

            if (rawValue == null)
            {
                return "value null";
            }

            if (rawValue is string s)
            {
                return string.IsNullOrWhiteSpace(s)
                    ? "value empty"
                    : (IsNeutralFallbackValue(featureVector, key, rawValue)
                        ? string.Format(
                            CultureInfo.InvariantCulture,
                            "neutral fallback '{0}'",
                            FormatFeatureValue(rawValue))
                        : "value ignored");
            }

            if (rawValue is double d)
            {
                if (double.IsNaN(d))
                {
                    return "value NaN";
                }

                if (double.IsInfinity(d))
                {
                    return "value non-finite";
                }
            }

            if (rawValue is float f)
            {
                if (float.IsNaN(f))
                {
                    return "value NaN";
                }

                if (float.IsInfinity(f))
                {
                    return "value non-finite";
                }
            }

            if (IsNeutralFallbackValue(featureVector, key, rawValue))
            {
                return string.Format(
                    CultureInfo.InvariantCulture,
                    "neutral fallback '{0}'",
                    FormatFeatureValue(rawValue));
            }

            if (!HasMeaningfulValue(rawValue))
            {
                return string.Format(
                    CultureInfo.InvariantCulture,
                    "value '{0}' not meaningful",
                    FormatFeatureValue(rawValue));
            }

            return string.Format(
                CultureInfo.InvariantCulture,
                "value '{0}' filtered",
                FormatFeatureValue(rawValue));
        }

        private static string FormatFeatureValue(object? value)
        {
            if (value == null)
            {
                return "null";
            }

            if (value is IFormattable formattable)
            {
                return formattable.ToString(null, CultureInfo.InvariantCulture) ?? "null";
            }

            return value.ToString() ?? "null";
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
        private static IReadOnlyList<string> ResolveTrackedFeatureKeys()
        {
            var neuralKeys = GetCachedNeuralFeatureKeys();
            if (neuralKeys.Count > 0)
            {
                var combined = new List<string>(neuralKeys);
                var seen = new HashSet<string>(combined, StringComparer.OrdinalIgnoreCase);

                var essentialKeys = EssentialTrackedFeatureKeys ?? Array.Empty<string>();

                foreach (var key in essentialKeys)
                {
                    if (string.IsNullOrWhiteSpace(key))
                    {
                        continue;
                    }

                    if (seen.Add(key))
                    {
                        combined.Add(key);
                    }
                }

                return combined;
            }

            return HistoricalFeatureBackfillKeys ?? Array.Empty<string>();
        }
        private static FeaturePopulationSummary BuildFeaturePopulationSummary(Dictionary<string, object?> featureVector)
        {
            var missingList = GetMissingHistoricalFeatureKeys(featureVector);

            var normalizedMissing = missingList
                .Where(k => !string.IsNullOrWhiteSpace(k))
                .Select(k => k.Trim())
                .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (featureVector == null)
            {
                return new FeaturePopulationSummary
                {
                    PopulatedCount = 0,
                    MissingCount = normalizedMissing.Count,
                    PopulatedKeys = Array.Empty<string>(),
                    MissingKeys = normalizedMissing
                };
            }

            var missingLookup = new HashSet<string>(normalizedMissing, StringComparer.OrdinalIgnoreCase);
            var populated = new List<string>();

            var trackedKeys = ResolveTrackedFeatureKeys();

            foreach (var key in trackedKeys)
            {
                if (string.IsNullOrWhiteSpace(key))
                {
                    continue;
                }

                var normalizedKey = key.Trim();

                if (!ShouldRequireFeature(featureVector, normalizedKey))
                {
                    continue;
                }

                if (missingLookup.Contains(normalizedKey))
                {
                    continue;
                }

                if (HasMeaningfulOrFallbackValue(featureVector, normalizedKey))
                {
                    populated.Add(normalizedKey);
                }
            }

            populated = populated
                .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                .ToList();

            return new FeaturePopulationSummary
            {
                PopulatedCount = populated.Count,
                MissingCount = normalizedMissing.Count,
                PopulatedKeys = populated.Count > 0
                    ? populated
                    : Array.Empty<string>(),
                MissingKeys = normalizedMissing.Count > 0
                    ? normalizedMissing
                    : Array.Empty<string>()
            };
        }
        private static bool HasMeaningfulOrFallbackValue(Dictionary<string, object?>? featureVector, string key)
        {
            if (TryGetMeaningfulValue(featureVector, key, out _))
            {
                return true;
            }

            if (featureVector == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (!featureVector.TryGetValue(key, out var existing) || !HasMeaningfulValue(existing))
            {
                return false;
            }

            return IsNeutralFallbackValue(featureVector, key, existing);
        }
        private void EnsureRaceAverageSpeedLast5(IReadOnlyList<RunnerFlow> flows)
        {
            if (flows == null || flows.Count == 0)
            {
                return;
            }
            foreach (var flow in flows)
            {
                if (flow?.FeatureValues == null)
                {
                    continue;
                }

                float? resolvedSpeed = null;

                if (TryGetMeaningfulValue(flow.FeatureValues, "AvgSpeedLast5", out var speedValue))
                {
                    var converted = TryConvertToSingle(speedValue);
                    if (converted.HasValue &&
                        !float.IsNaN(converted.Value) &&
                        !float.IsInfinity(converted.Value) &&
                        converted.Value > 0f)
                    {
                        resolvedSpeed = converted.Value;
                    }
                }

                if (!resolvedSpeed.HasValue)
                {
                    var fallback = TryResolveAvgSpeedLast5FromDatabase(flow.FeatureValues, flow);
                    if (fallback.HasValue && fallback.Value > 0f)
                    {
                        flow.FeatureValues["AvgSpeedLast5"] = fallback.Value;
                        resolvedSpeed = fallback.Value;
                    }
                }

                if (resolvedSpeed.HasValue)
                {
                    flow.FeatureValues["RaceAvgSpeedLast5"] = resolvedSpeed.Value;
                }
                else if (flow.FeatureValues.ContainsKey("RaceAvgSpeedLast5"))
                {
                    flow.FeatureValues.Remove("RaceAvgSpeedLast5");
                }
            }
        }
        private void EnsureRunnerClassAndGoingDistanceFeatures(IReadOnlyList<RunnerFlow> flows)
        {
            if (flows == null || flows.Count == 0)
            {
                return;
            }

            foreach (var flow in flows)
            {
                if (flow?.FeatureValues == null)
                {
                    continue;
                }

                ApplyTrainerClassFeatures(flow);
                ApplyJockeyClassFeatures(flow);
                ApplyRunnerClassFeatures(flow);
            }
        }
        private void ApplyRunnerClassFeatures(RunnerFlow flow)
        {
            if (flow?.FeatureValues == null)
            {
                return;
            }

            var featureValues = flow.FeatureValues;

            var classWinRate = ResolveFeatureValue(featureValues, "ClassWinRate");

            if (!classWinRate.HasValue)
            {
                var lifetimeWinRate = ResolveFeatureValue(featureValues, "LifetimeWinRate");
                if (!lifetimeWinRate.HasValue)
                {
                    lifetimeWinRate = CombineAverages(
                        ResolveFeatureValue(featureValues, "TrainerWinRate"),
                        ResolveFeatureValue(featureValues, "JockeyWinRate"));
                }

                classWinRate = lifetimeWinRate
                    ?? ResolveFeatureValue(featureValues, "TrainerClassWinRate")
                    ?? ResolveFeatureValue(featureValues, "TrainerWinRate")
                    ?? ResolveFeatureValue(featureValues, "JockeyClassWinRate")
                    ?? ResolveFeatureValue(featureValues, "JockeyWinRate");
            }

            if (classWinRate.HasValue)
            {
                featureValues["ClassWinRate"] = classWinRate.Value;
            }
            else
            {
                featureValues.Remove("ClassWinRate");
            }

            float? classAvgNorm = ResolveFeatureValue(featureValues, "ClassAvgNorm");
            if (!classAvgNorm.HasValue)
            {
                classAvgNorm = CombineAverages(
                    ResolveFeatureValue(featureValues, "TrainerClassAvgNorm"),
                    ResolveFeatureValue(featureValues, "JockeyClassAvgNorm"));
            }
            if (!classAvgNorm.HasValue && classWinRate.HasValue)
            {
                classAvgNorm = ClampNormalizedPosition(1f - classWinRate.Value);
            }

            if (classAvgNorm.HasValue)
            {
                featureValues["ClassAvgNorm"] = classAvgNorm.Value;
            }
            else if (!TryGetMeaningfulValue(featureValues, "ClassAvgNorm", out _))
            {
                featureValues.Remove("ClassAvgNorm");
            }

            if (!TryGetMeaningfulValue(featureValues, "LastClassNormPos", out _))
            {
                var lastClass = classAvgNorm
                    ?? ResolveFeatureValue(featureValues, "AvgNormPosLast5")
                    ?? CombineAverages(
                        ResolveFeatureValue(featureValues, "TrainerClassAvgNorm"),
                        ResolveFeatureValue(featureValues, "JockeyClassAvgNorm"));

                if (!lastClass.HasValue && classWinRate.HasValue)
                {
                    lastClass = ClampNormalizedPosition(1f - classWinRate.Value);
                }

                if (lastClass.HasValue)
                {
                    featureValues["LastClassNormPos"] = lastClass.Value;
                }
                else
                {
                    featureValues.Remove("LastClassNormPos");
                }
            }
        }
        private void ApplyTrainerClassFeatures(RunnerFlow flow)
        {
            if (flow?.FeatureValues == null)
            {
                return;
            }

            var featureValues = flow.FeatureValues;
            var trainerWinRate = ResolveFeatureValue(featureValues, "TrainerWinRate");
            if (!trainerWinRate.HasValue)
            {
                trainerWinRate = ResolveFeatureValue(featureValues, "LifetimeWinRate");
            }
            var trainerSurfaceAvg = ResolveFeatureValue(featureValues, "TrainerSurfaceAvgNorm");
            var trainerGoingAvg = ResolveFeatureValue(featureValues, "TrainerGoingAvgNorm");
            var trainerAvgNorm = trainerSurfaceAvg ?? trainerGoingAvg;
            var lastTrainerSurface = ResolveFeatureValue(featureValues, "LastTrainerSurfaceNormPos");
            var lastTrainerGoing = ResolveFeatureValue(featureValues, "LastTrainerGoingNormPos");
            var lastTrainerNorm = lastTrainerSurface ?? lastTrainerGoing ?? trainerAvgNorm;
            ApplyFeatureValueOrFallback(featureValues, "TrainerClassWinRate", trainerWinRate); 
            featureValues.Remove("TrainerClassWinRate");
            

            float? trainerClassNorm = null;
            if (trainerAvgNorm.HasValue)
            {
                trainerClassNorm = trainerAvgNorm.Value;
            }
            else if (trainerWinRate.HasValue)
            {
                trainerClassNorm = ClampNormalizedPosition(1f - trainerWinRate.Value);
            }
            else
            {
                var lifetimeWinRate = ResolveFeatureValue(featureValues, "LifetimeWinRate");
                if (lifetimeWinRate.HasValue)
                {
                    trainerClassNorm = ClampNormalizedPosition(1f - lifetimeWinRate.Value);
                    ApplyFeatureValueOrFallback(featureValues, "TrainerClassWinRate", lifetimeWinRate);
                }
            }
            if (trainerClassNorm.HasValue)
            {
                featureValues["TrainerClassAvgNorm"] = trainerClassNorm.Value;
            }
            else
            {
                featureValues.Remove("TrainerClassAvgNorm");
            }

            var trainerLastClass = lastTrainerNorm ?? trainerClassNorm;
            if (trainerLastClass.HasValue)
            {
                featureValues["LastTrainerClassNormPos"] = trainerLastClass.Value;
            }
            else
            {
                featureValues.Remove("LastTrainerClassNormPos");
            }
        }
        private void ApplyJockeyClassFeatures(RunnerFlow flow)
        {
            if (flow?.FeatureValues == null)
            {
                return;
            }

            var featureValues = flow.FeatureValues;
            var jockeySurfaceAvg = ResolveFeatureValue(featureValues, "JockeySurfaceAvgNorm");
            var jockeyGoingAvg = ResolveFeatureValue(featureValues, "JockeyGoingAvgNorm");
            var jockeyDistanceAvg = ResolveFeatureValue(featureValues, "JockeyDistanceBucketAvgNorm");
            var lifetimeWinRate = ResolveFeatureValue(featureValues, "LifetimeWinRate");

            var goingDistanceWin = CombineAverages(
                ResolveFeatureValue(featureValues, "JockeyGoingWinRate"),
                ResolveFeatureValue(featureValues, "JockeyDistanceBucketWinRate"));
            if (!goingDistanceWin.HasValue)
            {
                var fallbackJockeyWinRate = ResolveFeatureValue(featureValues, "JockeyWinRate");
                if (!fallbackJockeyWinRate.HasValue)
                {
                    fallbackJockeyWinRate = lifetimeWinRate;
                }
                goingDistanceWin = fallbackJockeyWinRate;
            }
            ApplyFeatureValueOrFallback(featureValues, "JockeyGoingDistanceWinRate", goingDistanceWin);

            var goingDistanceAvg = CombineAverages(jockeyGoingAvg, jockeyDistanceAvg);
            if (!goingDistanceAvg.HasValue)
            {
                goingDistanceAvg = jockeySurfaceAvg
                    ?? ResolveFeatureValue(featureValues, "JockeyClassAvgNorm")
                    ?? (lifetimeWinRate.HasValue
                        ? ClampNormalizedPosition(1f - lifetimeWinRate.Value)
                        : (float?)null);
            }
            ApplyFeatureValueOrFallback(featureValues, "JockeyGoingDistanceAvgNorm", goingDistanceAvg);

            var lastJockeySurface = ResolveFeatureValue(featureValues, "LastJockeySurfaceNormPos");
            var lastJockeyGoing = ResolveFeatureValue(featureValues, "LastJockeyGoingNormPos");
            var lastJockeyDistance = ResolveFeatureValue(featureValues, "LastJockeyDistanceBucketNormPos");

            var lastGoingDistance = CombineAverages(lastJockeyGoing, lastJockeyDistance);
            if (!lastGoingDistance.HasValue)
            {
                lastGoingDistance = lastJockeySurface
                    ?? ResolveFeatureValue(featureValues, "LastJockeyClassNormPos")
                    ?? (goingDistanceAvg.HasValue ? goingDistanceAvg : (float?)null);
            }
            ApplyFeatureValueOrFallback(featureValues, "LastJockeyGoingDistanceNormPos", lastGoingDistance);
            var jockeyWinRate = ResolveJockeyClassWinRate(featureValues);
            if (jockeyWinRate.HasValue)
            {
                featureValues["JockeyClassWinRate"] = jockeyWinRate.Value;
            }
            else
            {
                if (!jockeyWinRate.HasValue)
                {
                    jockeyWinRate = lifetimeWinRate;
                }
                if (jockeyWinRate.HasValue)
                {
                    featureValues["JockeyClassWinRate"] = jockeyWinRate.Value;
                }
                else
                {
                    featureValues.Remove("JockeyClassWinRate");
                }
            }

            var jockeyClassNorm = ResolveJockeyClassAvgNorm(featureValues, jockeyWinRate);
            if (!jockeyClassNorm.HasValue && lifetimeWinRate.HasValue)
            {
                jockeyClassNorm = ClampNormalizedPosition(1f - lifetimeWinRate.Value);
            }
            if (jockeyClassNorm.HasValue)
            {
                featureValues["JockeyClassAvgNorm"] = jockeyClassNorm.Value;
            }
            else
            {
                featureValues.Remove("JockeyClassAvgNorm");
            }

            var lastJockeyClass = ResolveLastJockeyClassNorm(featureValues, jockeyClassNorm);
            if (lastJockeyClass.HasValue)
            {
                featureValues["LastJockeyClassNormPos"] = lastJockeyClass.Value;
            }
            else
            {
                featureValues.Remove("LastJockeyClassNormPos");
            }
            var trainerCourseWin = ResolveFeatureValue(featureValues, "TrainerCourseWinRate");
            var jockeyCourseWin = ResolveFeatureValue(featureValues, "JockeyCourseWinRate");
            var trainerWinRate = ResolveFeatureValue(featureValues, "TrainerWinRate") ?? lifetimeWinRate;
            var jockeyOverallWin = ResolveFeatureValue(featureValues, "JockeyWinRate") ?? lifetimeWinRate;

            var trainerJockeyCourseWin = CombineAverages(trainerCourseWin, jockeyCourseWin)
                ?? CombineAverages(trainerWinRate, jockeyOverallWin)
                ?? lifetimeWinRate;

            ApplyFeatureValueOrFallback(featureValues, "TrainerJockeyCourseWinRate", trainerJockeyCourseWin);
        }
        private float? TryResolveAvgSpeedLast5FromDatabase(
            Dictionary<string, object?> featureValues,
            RunnerFlow flow)
        {
            if (featureValues == null)
            {
                return null;
            }

            int? horseId = null;
            if (featureValues.TryGetValue("HorseId", out var horseObj))
            {
                horseId = TryConvertToInt32(horseObj);
            }

            if (!horseId.HasValue && flow?.FeatureValues != null &&
                flow.FeatureValues.TryGetValue("HorseId", out var flowHorseObj))
            {
                horseId = TryConvertToInt32(flowHorseObj);
            }

            horseId = NormalizeHorseIdentifier(horseId);

            string? horseName = flow?.HorseName;
            if (string.IsNullOrWhiteSpace(horseName) &&
                featureValues.TryGetValue("HorseName", out var horseNameObj) &&
                horseNameObj is string horseNameStr)
            {
                horseName = horseNameStr;
            }

            DateTime? raceDate = flow?.RaceDate;

            try
            {
                const int windowSize = 5;
                var entries = _repo.GetRecentHorseSpeedEntries(horseName, horseId, raceDate, windowSize);
                if (entries == null || entries.Count == 0)
                {
                    return null;
                }

                var speeds = new List<float>();

                foreach (var entry in entries)
                {
                    if (!entry.DistanceYards.HasValue || entry.DistanceYards.Value <= 0)
                    {
                        continue;
                    }

                    if (!entry.WinningTimeMilliseconds.HasValue || entry.WinningTimeMilliseconds.Value <= 0)
                    {
                        continue;
                    }

                    var runnerTimeMs = (float)entry.WinningTimeMilliseconds.Value;
                    if (entry.DistanceBeatenLengths.HasValue)
                    {
                        runnerTimeMs += (float)entry.DistanceBeatenLengths.Value * MsPerLength;
                    }

                    if (runnerTimeMs <= 0f)
                    {
                        continue;
                    }

                    var speed = entry.DistanceYards.Value / runnerTimeMs;
                    if (float.IsNaN(speed) || float.IsInfinity(speed) || speed <= 0f)
                    {
                        continue;
                    }

                    speeds.Add(speed);
                }

                if (speeds.Count == 0)
                {
                    return null;
                }
                var average = speeds.Average();
                if (float.IsNaN(average) || float.IsInfinity(average) || average <= 0f)
                {
                    return null;
                }               
                return average;
            }
            catch (Exception ex)
            {
                var identifier = DescribeRunner(flow);
                return null;
            }
        }
        private void EnsureRaceAverageWinRateLast5(IReadOnlyList<RunnerFlow> flows)
        {
            if (flows == null || flows.Count == 0)
            {
                return;
            }
            double sum = 0d;
            int validParticipantCount = 0;
            var perRunnerWinRates = new Dictionary<RunnerFlow, float>();

            foreach (var flow in flows)
            {
                var featureValues = flow.FeatureValues;
                object? winRateValue;
                var usedLifetimeFallback = false;
                var usedDatabaseFallback = false;
                int? databaseFallbackWins = null;
                int? databaseFallbackStarts = null;

                if (!TryGetMeaningfulValue(featureValues, "WinRateLast5", out winRateValue))
                {
                    float? lifetimeFallback = null;
                    object? lifetimeRaw = null;

                    if (featureValues.TryGetValue("LifetimeWinRate", out lifetimeRaw))
                    {
                        var lifetimeConverted = TryConvertToSingle(lifetimeRaw);
                        if (lifetimeConverted.HasValue &&
                            !float.IsNaN(lifetimeConverted.Value) &&
                            !float.IsInfinity(lifetimeConverted.Value))
                        {
                            var sanitized = lifetimeConverted.Value;
                            if (sanitized < 0f)
                            {
                                sanitized = 0f;
                            }
                            else if (sanitized > 1f)
                            {
                                sanitized = 1f;
                            }

                            lifetimeFallback = sanitized;
                        }
                    }

                    if (lifetimeFallback.HasValue)
                    {
                        featureValues["WinRateLast5"] = lifetimeFallback.Value;
                        EnsureWinRatePerformanceWindows(featureValues, lifetimeFallback.Value);
                        winRateValue = lifetimeFallback.Value;
                        usedLifetimeFallback = true;
                        
                    }
                    else
                    {
                        var databaseFallback = TryResolveWinRateLast5FromDatabase(featureValues, flow);
                        if (databaseFallback.HasValue)
                        {
                            featureValues["WinRateLast5"] = databaseFallback.Value.WinRate;
                            EnsureWinRatePerformanceWindows(featureValues, flow?.FeatureValues ?? featureValues, databaseFallback.Value.WinRate);
                            winRateValue = databaseFallback.Value.WinRate;
                            usedDatabaseFallback = true;
                            databaseFallbackWins = databaseFallback.Value.Wins;
                            databaseFallbackStarts = databaseFallback.Value.Starts;
                            
                        }
                        else
                        {
                            var raw = featureValues.TryGetValue("WinRateLast5", out var candidate)
                                ? candidate
                                : null;
                            
                            continue;
                        }
                    }
                }

                var converted = TryConvertToSingle(winRateValue);
                sum += converted.Value;
                validParticipantCount++;
                perRunnerWinRates[flow] = converted.Value;
            }

            if (validParticipantCount == 0)
            {
                
                return;
            }
            var average = (float)(sum / validParticipantCount);
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

                float valueToAssign;
                if (perRunnerWinRates.TryGetValue(flow, out var runnerWinRate))
                {
                    if (validParticipantCount > 1)
                    {
                        valueToAssign = (float)((sum - runnerWinRate) / (validParticipantCount - 1));
                    }
                    else
                    {
                        valueToAssign = runnerWinRate;
                    }
                }
                else
                {
                    valueToAssign = average;
                }

                flow.FeatureValues["RaceAvgWinRateLast5"] = valueToAssign;
               
            }
        }
        private (float WinRate, int Wins, int Starts)? TryResolveWinRateLast5FromDatabase(
            Dictionary<string, object?> featureValues,
            RunnerFlow flow)
        {
            if (featureValues == null)
            {
                return null;
            }

            int? horseId = null;
            if (featureValues.TryGetValue("HorseId", out var horseObj))
            {
                horseId = TryConvertToInt32(horseObj);
            }

            if (!horseId.HasValue && flow?.FeatureValues != null &&
                flow.FeatureValues.TryGetValue("HorseId", out var flowHorseObj))
            {
                horseId = TryConvertToInt32(flowHorseObj);
            }

            horseId = NormalizeHorseIdentifier(horseId);

            string? horseName = flow?.HorseName;
            if (string.IsNullOrWhiteSpace(horseName) &&
                featureValues.TryGetValue("HorseName", out var horseNameObj) &&
                horseNameObj is string horseNameStr)
            {
                horseName = horseNameStr;
            }

            DateTime? raceDate = flow?.RaceDate;

            try
            {
                const int windowSize = 5;
                var stats = _repo.GetRecentHorseWinStats(horseName, horseId, raceDate, windowSize);
                if (!stats.HasValue || stats.Value.Starts <= 0)
                {
                    return null;
                }

                var winRate = _trainer.ComputeSmoothedWinRate(stats.Value.Wins, stats.Value.Starts);
                if (float.IsNaN(winRate) || float.IsInfinity(winRate))
                {
                    return null;
                }

                if (winRate < 0f)
                {
                    winRate = 0f;
                }
                else if (winRate > 1f)
                {
                    winRate = 1f;
                }

                return (winRate, stats.Value.Wins, stats.Value.Starts);
            }
            catch (Exception ex)
            {
                var identifier = DescribeRunner(flow);
                Console.Error.WriteLine($"[RaceAvgWinRateLast5] Failed to resolve historical win rate for {identifier}: {ex.Message}");
                return null;
            }
        }
        private static bool IsMeaningfulNeutralFallback(
            Dictionary<string, object?> source,
            string key,
            object? value)
        {
            if (source == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (IsClassDependentKey(key) && HasMeaningfulNonNeutralValue(source, "Class"))
            {
                return true;
            }

            if (IsGoingDistanceDependentKey(key) &&
                HasMeaningfulNonNeutralValue(source, "Going") &&
                (HasMeaningfulNonNeutralValue(source, "DistanceBucket") ||
                 HasMeaningfulNonNeutralValue(source, "DistanceText") ||
                 HasMeaningfulNonNeutralValue(source, "DistanceYards")))
            {
                return true;
            }

            if (IsRatingAggregateKey(key))
            {
                if (TryExtractPerformanceWindow(key, out var window) &&
                    TryGetCareerStarts(source, out var starts) &&
                    window > 0 && starts > 0 && starts < window)
                {
                    return true;
                }

                if (HasMeaningfulNonNeutralValue(source, "OfficialRating"))
                {
                    return true;
                }
            }

            return false;
        }
        public static bool TryGetMeaningfulValue(
            Dictionary<string, object?>? source,
            string key,
            out object? value)
        {
            value = null;

            if (source == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (!source.TryGetValue(key, out var candidate) || !HasMeaningfulValue(candidate))
            {
                return false;
            }

            if (IsNeutralFallbackValue(source, key, candidate) &&
                !IsMeaningfulNeutralFallback(source, key, candidate))
            {
                return false;
            }

            value = candidate;
            return true;
        }

        private static bool HasMeaningfulNonNeutralValue(
            Dictionary<string, object?> source,
            string key)
        {
            if (source == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (!source.TryGetValue(key, out var candidate) || !HasMeaningfulValue(candidate))
            {
                return false;
            }

            return !IsNeutralFallbackValue(source, key, candidate);
        }
        private static void ApplyFeatureValueOrFallback(
            Dictionary<string, object?>? featureValues,
            string key,
            float? value)
        {
            if (featureValues == null || string.IsNullOrWhiteSpace(key))
            {
                return;
            }

            if (value.HasValue)
            {
                featureValues[key] = value.Value;
                return;
            }

            if (!NeutralFeatureFallbacks.TryGetValue(key, out var fallback) || fallback == null)
            {
                featureValues.Remove(key);
                return;
            }

            var fallbackValue = TryConvertToSingle(fallback);
            if (fallbackValue.HasValue)
            {
                featureValues[key] = fallbackValue.Value;
                return;
            }

            featureValues.Remove(key);
        }
        private static bool TryGetCareerStarts(
            Dictionary<string, object?> source,
            out int starts)
        {
            starts = 0;
            if (source == null)
            {
                return false;
            }

            if (!source.TryGetValue("CareerStarts", out var startsObj))
            {
                return false;
            }

            var parsed = TryConvertToInt32(startsObj);
            if (!parsed.HasValue)
            {
                return false;
            }

            starts = parsed.Value;
            return true;
        }
        private static float? ResolveNonNeutralFeatureValue(
            Dictionary<string, object?>? source,
            string key)
        {
            if (source == null)
            {
                return null;
            }

            return HasMeaningfulNonNeutralValue(source, key)
                ? ResolveFeatureValue(source, key)
                : null;
        }
        private static bool TryExtractPerformanceWindow(string key, out int window)
        {
            window = 0;
            if (string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            int end = key.Length - 1;
            while (end >= 0 && char.IsDigit(key[end]))
            {
                end--;
            }

            if (end == key.Length - 1)
            {
                return false;
            }

            var span = key.AsSpan(end + 1);
            if (!int.TryParse(span, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
            {
                return false;
            }

            window = parsed;
            return true;
        }

        private static bool IsNeutralFallbackValue(Dictionary<string, object?>? source, string key, object? value)
        {
            if (value == null)
            {
                return false;
            }
            if (NeutralFallbackAllowedKeys.Contains(key))
            {
                return false;
            }
            if (!NeutralFeatureFallbacks.TryGetValue(key, out var fallback) || fallback == null)
            {
                return false;
            }
            if (!NeutralFallbackMissingKeys.Contains(key))
            {
                return false;
            }
            if (source != null &&
                string.Equals(key, "DistanceBeatenLengths", StringComparison.OrdinalIgnoreCase) &&
                source.TryGetValue("DistanceBeatenKnown", out var knownObj) &&
                knownObj is bool known && known)
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
        private static readonly Lazy<HashSet<string>> NeutralFallbackMissingKeysLazy = new(CreateNeutralFallbackMissingKeys);
        private static HashSet<string> NeutralFallbackMissingKeys => NeutralFallbackMissingKeysLazy.Value;
        private static HashSet<string> CreateNeutralFallbackMissingKeys()
        {
            return new HashSet<string>(
                new[]
                {
                    "BackBookPercentage",
                    "LayBookPercentage",
                    "RunnerCount",
                    "Class",
                    "RaceType",
                    "Surface",
                    "Going",
                    "DistanceYards",
                    "DistanceText",
                    "DistanceBucket",
                    "DistanceBeatenLengths",
                    "ClassWinRate",
                    "ClassAvgNorm",
                    "LastClassNormPos",
                    "TrainerClassWinRate",
                    "TrainerClassAvgNorm",
                    "LastTrainerClassNormPos",
                    "TrainerJockeyWinRate",
                    "TrainerJockeySurfaceWinRate",
                    "TrainerJockeyCourseWinRate",
                    "JockeyClassWinRate",
                    "JockeyClassAvgNorm",
                    "LastJockeyClassNormPos",
                    "JockeyGoingDistanceWinRate",
                    "JockeyGoingDistanceAvgNorm",
                    "LastJockeyGoingDistanceNormPos",
                    "TrainerJockeyCourseWinRate"
                }
                .Concat(PerformanceWindowPrefixes.SelectMany(prefix =>
                    PerformanceWindowSizes.Select(window => string.Concat(prefix, window.ToString(CultureInfo.InvariantCulture)))))
                .ToArray(),
                StringComparer.OrdinalIgnoreCase);
        }
        private static readonly HashSet<string> NeutralFallbackAllowedKeys = new(
            new[]
            {
                "SpeedMissing",
                "DistanceBeatenKnown",
                "HasLastWin"
            },
            StringComparer.OrdinalIgnoreCase);

        private static readonly HashSet<string> RaceSpeedFeatureKeys = new(
            new[]
            {
                "WinningTimeMs",
                "RaceSpeed"
            },
            StringComparer.OrdinalIgnoreCase);

        private static readonly HashSet<string> RunnerSpeedFeatureKeys = new(
            new[]
            {
                "RunnerSpeed",
                "SpeedDiff",
                "SpeedRatio"
            },
            StringComparer.OrdinalIgnoreCase);

        private static readonly HashSet<string> DistanceBeatenSensitiveFeatureKeys = new(
            new[]
            {
                "DistanceBeatenLengths",
                "DistanceBeatenKnown"
            },
            StringComparer.OrdinalIgnoreCase);

        private static readonly Lazy<string[]> HistoricalFeatureBackfillKeysLazy = new(() =>
             new[]
             {
                "Class",
                "RaceType",
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
         .Concat(PerformanceWindowPrefixes.SelectMany((string prefix) =>
                PerformanceWindowSizes.Select(window => string.Concat(prefix, window.ToString(CultureInfo.InvariantCulture)))))
            .ToArray());
        private static readonly string[] EssentialTrackedFeatureKeys =
        {
            "HasLastWin",
            "JockeyClassWinRate",
            "JockeyClassAvgNorm"
        };
        private static string[] HistoricalFeatureBackfillKeys => HistoricalFeatureBackfillKeysLazy.Value;
        private static readonly HashSet<string> HistoricalFeaturesAlwaysRequired = new(//here
            new[]
            {
                "Class",
                "RaceType",
                "Surface",
                "Going",
                "DistanceYards",
                "DistanceText",
                "DistanceBucket",
                "BackBookPercentage",
                "LayBookPercentage",
                "RunnerCount"
            },
            StringComparer.OrdinalIgnoreCase);

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
            var trackedKeys = ResolveTrackedFeatureKeys();

            if (featureVector == null || featureVector.Count == 0)
            {
                missing.AddRange(trackedKeys);
                return missing;
            }

            foreach (var key in trackedKeys)
            {
                if (!ShouldRequireFeature(featureVector, key))
                {
                    continue;
                }
                if (!HasMeaningfulOrFallbackValue(featureVector, key))
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

            bool ratingAggregateInserted = false;

            foreach (var key in HistoricalFeatureBackfillKeys)
            {
                var requireFeature = ShouldRequireFeature(target, key);
                if (!requireFeature && !IsRatingAggregateKey(key))
                {
                    continue;
                }
                if (!TryGetMeaningfulValue(target, key, out _) &&
                    TryGetBackfillValue(target, source, key, out var replacement))
                {
                    target[key] = replacement;
                    if (IsRatingAggregateKey(key))
                    {
                        ratingAggregateInserted = true;
                    }
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

                if (!TryGetMeaningfulValue(target, key, out _) &&
                    TryGetBackfillValue(target, source, key, out var replacement))
                {
                    target[key] = replacement;
                    if (IsRatingAggregateKey(key))
                    {
                        ratingAggregateInserted = true;
                    }
                }
            }

            if (ratingAggregateInserted)
            {
                target["RatingAggregatesMissing"] = !HasAnyMeaningfulRating(target);
            }
        }
        private static bool TryGetBackfillValue(
            Dictionary<string, object?> target,
            Dictionary<string, object?>? source,
            string key,
            out object? value)
        {
            value = null;
            if (source == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (!source.TryGetValue(key, out var candidate) || !HasMeaningfulValue(candidate))
            {
                return false;
            }

            if (!IsNeutralFallbackValue(source, key, candidate))
            {
                value = candidate;
                return true;
            }

            if (IsMeaningfulNeutralFallback(source, key, candidate) ||
                IsMeaningfulNeutralFallback(target, key, candidate))
            {
                value = candidate;
                return true;
            }

            return false;
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
        private static bool AreAlwaysRequiredHistoricalFeaturesSatisfied(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null || featureVector.Count == 0)
            {
                return false;
            }

            foreach (var key in HistoricalFeaturesAlwaysRequired)
            {
                if (!ShouldRequireFeature(featureVector, key))
                {
                    continue;
                }

                if (!TryGetMeaningfulValue(featureVector, key, out _))
                {
                    return false;
                }
            }

            return true;
        }
        private static bool ShouldRequireFeature(Dictionary<string, object?>? featureVector, string key)
        {
            if (featureVector == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }
            if (IsMetadataDependentKey(featureVector, key))
            {
                return false;
            }
            if (IsRaceSpeedFeature(key) && IsWinningTimeUnavailable(featureVector))
            {
                return false;
            }

            if (IsRunnerSpeedFeature(key) && IsRunnerSpeedDataUnavailable(featureVector))
            {
                return false;
            }

            if (IsDistanceBeatenSensitiveKey(key) && IsDistanceBeatenDataUnavailable(featureVector))
            {
                return false;
            }

            if (IsRaceMetadataUnavailable(featureVector, key))
            {
                return false;
            }
            if (IsDrawRelatedKey(key) && IsDrawDataUnavailable(featureVector))
            {
                return false;
            }

            if (IsAgeRestrictionKey(key) && IsAgeRestrictionDataUnavailable(featureVector))
            {
                return false;
            }

            if (IsRatingAggregateKey(key) && AreRatingAggregatesUnavailable(featureVector))
            {
                return false;
            }

            return true;
        }
        private static bool IsMetadataDependentKey(Dictionary<string, object?> featureVector, string key)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (IsClassDependentKey(key) && !HasMeaningfulClass(featureVector))
            {
                return true;
            }

            if (IsGoingDependentKey(key) && !HasMeaningfulGoing(featureVector))
            {
                return true;
            }

            if (IsDistanceBucketDependentKey(key) && !HasMeaningfulDistanceBucket(featureVector))
            {
                return true;
            }

            return false;
        }

        private static bool HasMeaningfulClass(Dictionary<string, object?> featureVector) =>
            TryGetMeaningfulValue(featureVector, "Class", out _);

        private static bool HasMeaningfulGoing(Dictionary<string, object?> featureVector) =>
            TryGetMeaningfulValue(featureVector, "Going", out _);

        private static bool HasMeaningfulDistanceBucket(Dictionary<string, object?> featureVector) =>
            TryGetMeaningfulValue(featureVector, "DistanceBucket", out _);

        private static bool IsClassDependentKey(string key) =>
            key.IndexOf("Class", StringComparison.OrdinalIgnoreCase) >= 0;

        private static bool IsGoingDependentKey(string key) =>
            key.IndexOf("Going", StringComparison.OrdinalIgnoreCase) >= 0 ||
            key.StartsWith("LayoffNormalized_", StringComparison.OrdinalIgnoreCase);
        private static bool IsGoingDistanceDependentKey(string key) =>
            key.IndexOf("GoingDistance", StringComparison.OrdinalIgnoreCase) >= 0;

        private static bool IsDistanceBucketDependentKey(string key) =>
            string.Equals(key, "DistanceBucket", StringComparison.OrdinalIgnoreCase) ||
            key.IndexOf("DistanceBucket", StringComparison.OrdinalIgnoreCase) >= 0;
        private static bool IsRaceSpeedFeature(string key) =>
            RaceSpeedFeatureKeys.Contains(key);

        private static bool IsRunnerSpeedFeature(string key)
        {
            if (RunnerSpeedFeatureKeys.Contains(key))
            {
                return true;
            }

            return key.StartsWith("AvgSpeed", StringComparison.OrdinalIgnoreCase) ||
                key.StartsWith("AvgSpeedDiff", StringComparison.OrdinalIgnoreCase);
        }
        private static bool HasMissingFlag(Dictionary<string, object?> featureVector, string flagKey)
        {
            return featureVector.TryGetValue(flagKey, out var flagValue) &&
                   flagValue is bool flagBool && flagBool;
        }

        private static bool IsRaceMetadataUnavailable(Dictionary<string, object?> featureVector, string key)
        {
            if (featureVector == null || string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (HasMissingFlag(featureVector, "ClassMissing") &&
                key.Contains("Class", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (HasMissingFlag(featureVector, "GoingMissing"))
            {
                if (key.Contains("Going", StringComparison.OrdinalIgnoreCase) ||
                    key.StartsWith("LayoffNormalized_", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            if (HasMissingFlag(featureVector, "SurfaceMissing") &&
                key.Contains("Surface", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (HasMissingFlag(featureVector, "DistanceMissing"))
            {
                if (string.Equals(key, "DistanceYards", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(key, "DistanceText", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(key, "DistanceBucket", StringComparison.OrdinalIgnoreCase) ||
                    key.Contains("DistanceBucket", StringComparison.OrdinalIgnoreCase) ||
                    key.Contains("DistanceChange", StringComparison.OrdinalIgnoreCase) ||
                    key.Contains("DistanceRatio", StringComparison.OrdinalIgnoreCase) ||
                    key.Contains("GoingDistance", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            if (HasMissingFlag(featureVector, "BackBookPercentageMissing") &&
                key.Contains("BackBook", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (HasMissingFlag(featureVector, "LayBookPercentageMissing") &&
                key.Contains("LayBook", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            return false;
        }

        private static bool IsDrawRelatedKey(string key)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (string.Equals(key, "Draw", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            return key.IndexOf("Draw", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static bool IsDrawDataUnavailable(Dictionary<string, object?> featureVector)
        {
            return HasMissingFlag(featureVector, "DrawMissing");
        }

        private static bool IsAgeRestrictionKey(string key) =>
            key.IndexOf("AgeRestriction", StringComparison.OrdinalIgnoreCase) >= 0;

        private static bool IsAgeRestrictionDataUnavailable(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return true;
            }

            if (HasMissingFlag(featureVector, "AgeRestrictionMissing"))
            {
                return true;
            }

            return !TryGetMeaningfulValue(featureVector, "AgeRestriction", out _);
        }

        private static bool IsRatingAggregateKey(string key)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            if (key.StartsWith("AvgRating", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (key.StartsWith("RatingDiff", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            return string.Equals(key, "Rating", StringComparison.OrdinalIgnoreCase);
        }

        private static bool AreRatingAggregatesUnavailable(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return true;
            }

            if (HasMissingFlag(featureVector, "RatingAggregatesMissing"))
            {
                return true;
            }

            foreach (var kvp in featureVector)
            {
                if (!IsRatingAggregateKey(kvp.Key))
                {
                    continue;
                }

                if (TryGetMeaningfulValue(featureVector, kvp.Key, out _))
                {
                    return false;
                }
            }

            return true;
        }

        private static bool HasAnyMeaningfulRating(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null || featureVector.Count == 0)
            {
                return false;
            }

            foreach (var kvp in featureVector)
            {
                if (!IsRatingAggregateKey(kvp.Key))
                {
                    continue;
                }

                if (TryGetMeaningfulValue(featureVector, kvp.Key, out _))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsDistanceBeatenSensitiveKey(string key) =>
            DistanceBeatenSensitiveFeatureKeys.Contains(key);

        private static bool IsWinningTimeUnavailable(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return true;
            }

            if (!featureVector.TryGetValue("WinningTimeMs", out var winningObj))
            {
                return true;
            }

            var winning = TryConvertToSingle(winningObj);
            return !winning.HasValue || winning.Value <= 0f;
        }

        private static bool IsRunnerSpeedDataUnavailable(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return true;
            }

            if (featureVector.TryGetValue("SpeedMissing", out var speedMissingObj) &&
                speedMissingObj is bool speedMissing && speedMissing)
            {
                return true;
            }

            if (!featureVector.TryGetValue("RunnerSpeed", out var runnerSpeedObj))
            {
                return true;
            }

            var runnerSpeed = TryConvertToSingle(runnerSpeedObj);
            return !runnerSpeed.HasValue || runnerSpeed.Value <= 0f;
        }

        private static bool IsDistanceBeatenDataUnavailable(Dictionary<string, object?> featureVector)
        {
            if (!featureVector.TryGetValue("DistanceBeatenKnown", out var knownObj))
            {
                return false;
            }

            return knownObj is bool known && !known;
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
            horseId = NormalizeHorseIdentifier(horseId);
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
        private void EnsureDistanceBeatenFromNumeric(Dictionary<string, object?> featureVector, RunnerFlow flow)
        {
            if (featureVector == null)
            {
                return;
            }

            if (featureVector.TryGetValue("DistanceBeatenKnown", out var knownObj) &&
                knownObj is bool knownBool && knownBool)
            {
                if (featureVector.TryGetValue("DistanceBeatenLengths", out var existingObj))
                {
                    var existing = TryConvertToSingle(existingObj);
                    if (existing.HasValue && existing.Value > 0f)
                    {
                        return;
                    }
                }
            }

            if (TryGetMeaningfulValue(featureVector, "DistanceBeatenLengths", out var lenObj))
            {
                var converted = TryConvertToSingle(lenObj);
                if (converted.HasValue)
                {
                    featureVector["DistanceBeatenLengths"] = converted.Value;
                    featureVector["DistanceBeatenKnown"] = true;
                    return;
                }
            }

            int? runnerResultId = null;
            if (featureVector.TryGetValue("RunnerResultId", out var runnerObj))
            {
                runnerResultId = TryConvertToInt32(runnerObj);
            }

            try
            {
                if (runnerResultId.HasValue && runnerResultId.Value > 0)
                {
                    var lengths = _repo.GetDistanceBeatenLengths(runnerResultId.Value);
                    if (lengths.HasValue)
                    {
                        featureVector["DistanceBeatenLengths"] = (float)lengths.Value;
                        featureVector["DistanceBeatenKnown"] = true;
                        return;
                    }
                }

                int? horseId = null;
                if (featureVector.TryGetValue("HorseId", out var horseObj))
                {
                    horseId = TryConvertToInt32(horseObj);
                }

                if (!horseId.HasValue && flow?.FeatureValues != null &&
                    flow.FeatureValues.TryGetValue("HorseId", out var flowHorseId))
                {
                    horseId = TryConvertToInt32(flowHorseId);
                }
                horseId = NormalizeHorseIdentifier(horseId);
                string? horseName = flow?.HorseName;
                if (string.IsNullOrWhiteSpace(horseName) &&
                    featureVector.TryGetValue("HorseName", out var horseNameObj) &&
                    horseNameObj is string horseNameStr)
                {
                    horseName = horseNameStr;
                }

                if (horseId.HasValue || !string.IsNullOrWhiteSpace(horseName))
                {
                    var cutoffDate = flow?.RaceDate;
                    var lengths = _repo.GetLastDistanceBeatenLengths(horseName, horseId, cutoffDate);
                    if (lengths.HasValue)
                    {
                        featureVector["DistanceBeatenLengths"] = (float)lengths.Value;
                        featureVector["DistanceBeatenKnown"] = true;
                    }
                }
            }
            catch (Exception ex)
            {
                var identifier = DescribeRunner(flow);
                Console.Error.WriteLine($"\t\tFailed to resolve distance beaten for {identifier}: {ex.Message}");
            }
        }
        private void EnsureWinningTimeFromRace(Dictionary<string, object?> featureVector, RunnerFlow flow)
        {
            if (featureVector == null)
            {
                return;
            }

            if (featureVector.TryGetValue("WinningTimeMs", out var winningObj))
            {
                var existing = TryConvertToSingle(winningObj);
                if (existing.HasValue && existing.Value > 0f)
                {
                    featureVector["WinningTimeMs"] = existing.Value;
                    return;
                }
            }

            int? horseId = null;
            if (featureVector.TryGetValue("HorseId", out var horseObj))
            {
                horseId = TryConvertToInt32(horseObj);
            }

            if (!horseId.HasValue && flow?.FeatureValues != null &&
                flow.FeatureValues.TryGetValue("HorseId", out var flowHorseId))
            {
                horseId = TryConvertToInt32(flowHorseId);
            }

            horseId = NormalizeHorseIdentifier(horseId);

            string? horseName = flow?.HorseName;
            if (string.IsNullOrWhiteSpace(horseName) &&
                featureVector.TryGetValue("HorseName", out var horseNameObj) &&
                horseNameObj is string horseNameStr)
            {
                horseName = horseNameStr;
            }

            DateTime? raceDate = flow?.RaceDate;
            if (!raceDate.HasValue && featureVector.TryGetValue("RaceDate", out var raceDateObj))
            {
                raceDate = raceDateObj switch
                {
                    DateTime dt => dt,
                    DateTimeOffset dto => dto.DateTime,
                    _ => null
                };
            }

            try
            {
                var resolved = _repo.GetLastWinningTimeMilliseconds(horseName, horseId, raceDate);
                if (resolved.HasValue && resolved.Value > 0)
                {
                    featureVector["WinningTimeMs"] = (float)resolved.Value;

                    try
                    {
                        var lastDistance = _repo.GetLastRaceDistance(horseName, horseId, raceDate);
                        if (lastDistance.HasValue && lastDistance.Value > 0)
                        {
                            featureVector["LastRaceDistanceYards"] = lastDistance.Value;
                        }
                    }
                    catch (Exception distanceEx)
                    {
                        var identifier = DescribeRunner(flow);
                        Console.Error.WriteLine($"\t\tFailed to resolve previous race distance for {identifier}: {distanceEx.Message}");
                    }

                    return;
                }
            }
            catch (Exception ex)
            {
                var identifier = DescribeRunner(flow);
                Console.Error.WriteLine($"\t\tFailed to resolve previous winning time for {identifier}: {ex.Message}");
            }

            try
            {
                int? fallbackRaceId = featureVector.TryGetValue("RaceId", out var raceIdObj)
                    ? TryConvertToInt32(raceIdObj)
                    : null;

                if (!fallbackRaceId.HasValue || fallbackRaceId.Value <= 0)
                {
                    int? runnerResultId = featureVector.TryGetValue("RunnerResultId", out var runnerObj)
                        ? TryConvertToInt32(runnerObj)
                        : null;

                    if (!runnerResultId.HasValue && flow?.FeatureValues != null &&
                        flow.FeatureValues.TryGetValue("RunnerResultId", out var flowRunnerObj))
                    {
                        runnerResultId = TryConvertToInt32(flowRunnerObj);
                    }

                    if (runnerResultId.HasValue && runnerResultId.Value > 0)
                    {
                        try
                        {
                            fallbackRaceId = _repo.GetRaceIdByRunnerResult(runnerResultId.Value);
                        }
                        catch (Exception runnerResultEx)
                        {
                            var identifierLookup = DescribeRunner(flow);
                            Console.Error.WriteLine($"\t\tFailed to resolve race id for {identifierLookup}: {runnerResultEx.Message}");
                        }
                    }
                }

                if (!fallbackRaceId.HasValue || fallbackRaceId.Value <= 0)
                {
                    return;
                }

                var fallback = _repo.GetWinningTimeMilliseconds(fallbackRaceId.Value);
                if (fallback.HasValue && fallback.Value > 0)
                {
                    featureVector["WinningTimeMs"] = (float)fallback.Value;
                }
            }
            catch (Exception ex)
            {
                var identifier = DescribeRunner(flow);
                Console.Error.WriteLine($"\t\tFailed to resolve fallback winning time for {identifier}: {ex.Message}");
            }
        }
        private void EnsureSpeedMetrics(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return;
            }

            if (!featureVector.TryGetValue("WinningTimeMs", out var winningObj))
            {
                return;
            }

            var winningMs = TryConvertToSingle(winningObj);
            if (!winningMs.HasValue || winningMs.Value <= 0f)
            {
                return;
            }

            float? distanceForSpeed = null;
            if (featureVector.TryGetValue("LastRaceDistanceYards", out var lastDistanceObj))
            {
                var converted = TryConvertToSingle(lastDistanceObj);
                if (converted.HasValue && converted.Value > 0f)
                {
                    distanceForSpeed = converted.Value;
                }
            }

            if (!distanceForSpeed.HasValue)
            {
                if (!featureVector.TryGetValue("DistanceYards", out var distanceObj))
                {
                    return;
                }

                var distanceYards = TryConvertToSingle(distanceObj);
                if (!distanceYards.HasValue || distanceYards.Value <= 0f)
                {
                    return;
                }

                distanceForSpeed = distanceYards.Value;
            }

            var raceSpeed = distanceForSpeed.Value / winningMs.Value;
            featureVector["RaceSpeed"] = raceSpeed;

            bool distanceKnown = featureVector.TryGetValue("DistanceBeatenKnown", out var knownObj) &&
                                 knownObj is bool knownBool && knownBool;

            float runnerSpeed = 0f;
            bool hasRunnerSpeed = false;
            if (distanceKnown && featureVector.TryGetValue("DistanceBeatenLengths", out var lenObj))
            {
                var beaten = TryConvertToSingle(lenObj);
                if (beaten.HasValue)
                {
                    var runnerTime = winningMs.Value + beaten.Value * MsPerLength;
                    if (runnerTime > 0f)
                    {
                        runnerSpeed = distanceForSpeed.Value / runnerTime;
                        hasRunnerSpeed = runnerSpeed > 0f;
                    }
                }
            }

            featureVector["RunnerSpeed"] = hasRunnerSpeed ? runnerSpeed : 0f;
            featureVector["SpeedDiff"] = hasRunnerSpeed ? runnerSpeed - raceSpeed : 0f;
            featureVector["SpeedRatio"] = hasRunnerSpeed && raceSpeed != 0f ? runnerSpeed / raceSpeed : 0f;
            featureVector["SpeedMissing"] = !hasRunnerSpeed;
            featureVector.Remove("LastRaceDistanceYards");
        }
        private static string NormalizeHorseNameKeyForCache(string? name)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                return string.Empty;
            }

            return name.Trim().ToLowerInvariant();
        }
        private static int? NormalizeHorseIdentifier(int? horseId)
        {
            if (!horseId.HasValue)
            {
                return null;
            }

            var value = horseId.Value;
            if (value <= 0 || IsSyntheticIdentifier(value))
            {
                return null;
            }

            return value;
        }

        private static bool IsSyntheticIdentifier(int value)
        {
            if (value == int.MaxValue)
            {
                return true;
            }

            const int syntheticPrefix = unchecked((int)0x60000000);
            const int mask = unchecked((int)0xF0000000);
            return (value & mask) == syntheticPrefix;
        }
    }
}