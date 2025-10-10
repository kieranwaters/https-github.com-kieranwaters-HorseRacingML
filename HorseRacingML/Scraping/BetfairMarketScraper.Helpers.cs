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
        private static readonly Regex AgeParentheticalRegex = new(@"\((?<age>\d{1,2})\)", RegexOptions.Compiled);
        private static readonly Regex AgeWordRegex = new(@"\b(?<age>\d{1,2})\s*(?:yo|yr|yrs|year|years)\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeLeadingRegex = new(@"^(?<age>\d{1,2})(?=\s|$)", RegexOptions.Compiled);
        private static readonly Regex WeightDashRegex = new(@"\b(?<stone>\d{1,2})\s*[-/]\s*(?<pounds>\d{1,2})\b", RegexOptions.Compiled);
        private static readonly Regex WeightStoneRegex = new(@"\b(?<stone>\d{1,2})\s*(?:st|stone|stones)\s*(?<pounds>\d{1,2})?\s*(?:lb|lbs|pound|pounds)?\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex WeightLbsRegex = new(@"\b(?<pounds>\d{2,3})\s*(?:lb|lbs|pound|pounds)\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex TrainerPrefixRegex = new(@"^(?:trainer|trainers?|t:)\s*[:\-]?\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly string[] DistanceBuckets = { "Sprint", "Middle", "Long" };
        private static readonly int[] PerformanceWindowSizes = { 1, 3, 5, 10, 15, 20, 25, 30, 50, 100 };
        private static readonly string[] PerformanceWindowPrefixes =
        {
            "WinRateLast",
            "AvgNormPosLast",
            "AvgSpeedLast",
            "AvgSpeedDiffLast",
            "AvgRatingLast",
            "GoingWinRateLast",
            "GoingAvgNormLast",
            "SurfaceWinRateLast",
            "SurfaceAvgNormLast",
            "CourseWinRateLast",
            "CourseAvgNormLast",
            "DistanceBucketWinRateLast",
            "DistanceBucketAvgNormLast"
        };
        private static readonly IReadOnlyDictionary<string, object?> NeutralFeatureFallbacks =
            new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["BackBookPercentage"] = 0d,
                ["LayBookPercentage"] = 0d,
                ["RunnerCount"] = 0,
                ["Class"] = 0,
                ["RaceType"] = "Unknown",
                ["AgeRestriction"] = "Unknown",
                ["Surface"] = "Unknown",
                ["Going"] = "Unknown",
                ["DistanceYards"] = 0,
                ["DistanceText"] = "Unknown",
                ["DistanceBucket"] = "Unknown",
                ["HasLastWin"] = false,
                ["DistanceChangeFromLast"] = 0f,
                ["DistanceRatioFromAverage"] = 1f,
                ["CareerStarts"] = 0,
                ["LifetimeWinRate"] = 0f,
                ["DistanceBeatenLengths"] = 0f,
                ["DistanceBeatenKnown"] = false,
                ["DrawBias"] = 0f,
                ["SaddleclothDiffFromMean"] = 0f,
                ["IsTopWeight"] = false,
                ["IsBottomWeight"] = false,
                ["WinningTimeMs"] = 0f,
                ["RaceSpeed"] = 0f,
                ["RunnerSpeed"] = 0f,
                ["SpeedMissing"] = true,
                ["SpeedDiff"] = 0f,
                ["SpeedRatio"] = 0f,
                ["RaceAvgWinRateLast5"] = 0f,
                ["TrainerWinRate"] = 0f,
                ["TrainerWinRateLast50"] = 0f,
                ["TrainerWinRateRecentDays"] = 0f,
                ["TrainerSurfaceWinRate"] = 0f,
                ["TrainerSurfaceAvgNorm"] = 0f,
                ["LastTrainerSurfaceNormPos"] = 0f,
                ["TrainerGoingWinRate"] = 0f,
                ["TrainerGoingAvgNorm"] = 0f,
                ["LastTrainerGoingNormPos"] = 0f,
                ["TrainerDistanceBucketWinRate"] = 0f,
                ["TrainerDistanceBucketAvgNorm"] = 0f,
                ["LastTrainerDistanceBucketNormPos"] = 0f,
                ["TrainerCourseWinRate"] = 0f,
                ["TrainerClassWinRate"] = 0f,
                ["TrainerClassAvgNorm"] = 0f,
                ["LastTrainerClassNormPos"] = 0f,
                ["TrainerJockeyWinRate"] = 0f,
                ["TrainerJockeySurfaceWinRate"] = 0f,
                ["TrainerJockeyCourseWinRate"] = 0f,
                ["JockeyWinRate"] = 0f,
                ["JockeyWinRateLast50"] = 0f,
                ["JockeyWinRateRecentDays"] = 0f,
                ["JockeySurfaceWinRate"] = 0f,
                ["JockeySurfaceAvgNorm"] = 0f,
                ["LastJockeySurfaceNormPos"] = 0f,
                ["JockeyGoingWinRate"] = 0f,
                ["JockeyGoingAvgNorm"] = 0f,
                ["LastJockeyGoingNormPos"] = 0f,
                ["JockeyDistanceBucketWinRate"] = 0f,
                ["JockeyDistanceBucketAvgNorm"] = 0f,
                ["LastJockeyDistanceBucketNormPos"] = 0f,
                ["JockeyCourseWinRate"] = 0f,
                ["JockeyClassWinRate"] = 0f,
                ["JockeyClassAvgNorm"] = 0f,
                ["LastJockeyClassNormPos"] = 0f,
                ["JockeyGoingDistanceWinRate"] = 0f,
                ["JockeyGoingDistanceAvgNorm"] = 0f,
                ["LastJockeyGoingDistanceNormPos"] = 0f,
                ["GoingCourseWinRate"] = 0f,
                ["GoingCourseAvgNorm"] = 0f,
                ["LastGoingCourseNormPos"] = 0f,
                ["AgeRestrictionWinRate"] = 0f,
                ["LastAgeRestrictionNormPos"] = 0f,
                ["DistanceBucketWinRate"] = 0f,
                ["LastDistanceBucketNormPos"] = 0f,
                ["GoingDistanceWinRate"] = 0f,
                ["GoingDistanceAvgNorm"] = 0f,
                ["LastGoingDistanceNormPos"] = 0f,
                ["ClassWinRate"] = 0f,
                ["ClassAvgNorm"] = 0f,
                ["LastClassNormPos"] = 0f
            };
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
        static bool TryGetInt(object? source, out int value)
        {
            switch (source)
            {
                case null:
                    value = 0;
                    return false;
                case int i:
                    value = i;
                    return true;
                case long l when l >= int.MinValue && l <= int.MaxValue:
                    value = (int)l;
                    return true;
                case short s:
                    value = s;
                    return true;
                case byte b:
                    value = b;
                    return true;
                case float f when float.IsFinite(f):
                    value = (int)Math.Round(f);
                    return true;
                case double d when double.IsFinite(d):
                    value = (int)Math.Round(d);
                    return true;
                case decimal m:
                    value = (int)Math.Round(m);
                    return true;
                case string str when int.TryParse(str, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed):
                    value = parsed;
                    return true;
                default:
                    try
                    {
                        value = System.Convert.ToInt32(source, CultureInfo.InvariantCulture);
                        return true;
                    }
                    catch
                    {
                        value = 0;
                        return false;
                    }
            }
        }

        static string DistanceBucketFromYards(int yards)
            => yards < 1760 ? "Sprint" : yards < 2640 ? "Middle" : "Long";

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
        private void ApplyScrapedFeatureFallbacks(
            Dictionary<string, object?> featureVector,
            RunnerFlow flow,
            DateTime? raceDate,
            TimeSpan? scheduledOff,
            string? raceTitle,
            string? raceDetails,
            string? raceType,
            string? going,
            string? venueName,
            string? venueCountry,
            decimal? backBookPercentage,
            decimal? layBookPercentage,
            IReadOnlyList<RunnerFlow>? flows,
            ISet<string>? missingScrapeFields = null)
        {
            if (featureVector == null)
            {
                return;
            }

            static bool HasMissingValue(Dictionary<string, object?> target, string key)
            {
                if (!target.TryGetValue(key, out var existing) || existing == null)
                {
                    return true;
                }

                if (existing is string s)
                {
                    return string.IsNullOrWhiteSpace(s);
                }

                return false;
            }

            void SetIfMissing(string key, object? value)
            {
                if (value == null)
                {
                    return;
                }

                if (HasMissingValue(featureVector, key))
                {
                    featureVector[key] = value;
                }
            }

            void MarkMissing(string description)
            {
                if (missingScrapeFields == null)
                {
                    return;
                }

                var trimmed = description?.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    missingScrapeFields.Add(trimmed);
                }
            }

            if (flows != null && flows.Count > 0)
            {
                SetIfMissing("RunnerCount", Math.Min(flows.Count, byte.MaxValue));
            }
            else
            {
                MarkMissing("runner count");
            }
            int resolvedRunnerCount = flows?.Count(f => f != null) ?? 0;
            if (resolvedRunnerCount <= 0 &&
                featureVector.TryGetValue("RunnerCount", out var runnerCountObj) &&
                TryGetInt(runnerCountObj, out var runnerCountValue))
            {
                resolvedRunnerCount = runnerCountValue;
            }

            var clothValues = flows == null
                ? Array.Empty<int>()
                : flows
                    .Where(f => f?.ClothNumber.HasValue == true)
                    .Select(f => (int)f!.ClothNumber!.Value)
                    .ToArray();
            bool hasSaddleclothStats = clothValues.Length > 0;
            float avgSaddlecloth = (float)(hasSaddleclothStats ? clothValues.Average() : 0f);
            if (backBookPercentage.HasValue)
            {
                SetIfMissing("BackBookPercentage", Convert.ToDouble(backBookPercentage.Value));
            }
            else
            {
                MarkMissing("back book percentage");
            }

            if (layBookPercentage.HasValue)
            {
                SetIfMissing("LayBookPercentage", Convert.ToDouble(layBookPercentage.Value));
            }
            else
            {
                MarkMissing("lay book percentage");
            }

            if (flow?.Draw.HasValue == true)
            {
                SetIfMissing("Draw", flow.Draw.Value);
                SetIfMissing("DrawMissing", false);
            }
            else
            {
                SetIfMissing("DrawMissing", true);
                MarkMissing("draw number");
            }

            float relativeDraw = resolvedRunnerCount > 0 && flow?.Draw.HasValue == true
                ? flow.Draw.Value / (float)resolvedRunnerCount
                : 0f;
            SetIfMissing("RelativeDraw", relativeDraw);

            if (flow?.ClothNumber.HasValue == true)
            {
                SetIfMissing("SaddleclothNumber", flow.ClothNumber.Value);
                SetIfMissing("SaddleclothMissing", false);
            }
            else
            {
                SetIfMissing("SaddleclothMissing", true);
                MarkMissing("saddlecloth number");
            }
            int? saddleclothNumber = flow?.ClothNumber;
            float saddleclothRelative = saddleclothNumber.HasValue && resolvedRunnerCount > 0
                ? saddleclothNumber.Value / (float)resolvedRunnerCount
                : 0f;
            SetIfMissing("SaddleclothRelative", saddleclothRelative);

            float saddleclothDiff = saddleclothNumber.HasValue && hasSaddleclothStats
                ? saddleclothNumber.Value - avgSaddlecloth
                : 0f;
            SetIfMissing("SaddleclothDiffFromMean", saddleclothDiff);

            SetIfMissing("DrawBias", 0f);
            if (flow?.Age.HasValue == true)
            {
                SetIfMissing("Age", flow.Age.Value);
                SetIfMissing("AgeMissing", false);
            }
            else
            {
                SetIfMissing("AgeMissing", true);
                MarkMissing("runner age");
            }

            if (flow?.WeightLbs.HasValue == true)
            {
                SetIfMissing("WeightLbs", flow.WeightLbs.Value);
                if (!string.IsNullOrWhiteSpace(flow.WeightText))
                {
                    SetIfMissing("WeightText", flow.WeightText.Trim());
                }
                SetIfMissing("WeightMissing", false);
            }
            else
            {
                if (!string.IsNullOrWhiteSpace(flow?.WeightText))
                {
                    SetIfMissing("WeightText", flow.WeightText!.Trim());
                }

                SetIfMissing("WeightMissing", true);
                MarkMissing("runner weight");
            }

            if (!string.IsNullOrWhiteSpace(flow?.TrainerName))
            {
                SetIfMissing("TrainerName", flow!.TrainerName!.Trim());
            }
            else
            {
                MarkMissing("trainer name");
            }
            var metadataSource = new RaceDayReport
            {
                RaceTitle = raceTitle,
                RaceDetails = raceDetails,
                RaceType = raceType,
                Going = going,
                VenueName = venueName,
                VenueCountry = venueCountry
            };

            var parsed = ParseRaceMetadata(metadataSource);

            if (parsed.Class.HasValue)
            {
                SetIfMissing("Class", parsed.Class);
            }
            else
            {
                MarkMissing("race class");
            }

            if (!string.IsNullOrWhiteSpace(parsed.AgeRestriction))
            {
                SetIfMissing("AgeRestriction", parsed.AgeRestriction);
            }
            else
            {
                MarkMissing("age restriction");
            }
            if (HasMissingValue(featureVector, "AgeRestriction"))
            {
                var ageSamples = flows == null
                    ? Array.Empty<byte>()
                    : flows
                        .Where(f => f?.Age.HasValue == true)
                        .Select(f => f!.Age!.Value)
                        .ToArray();

                if (ageSamples.Length > 0)
                {
                    Array.Sort(ageSamples);
                    var minAge = ageSamples[0];
                    var maxAge = ageSamples[^1];
                    string restriction = minAge == maxAge
                        ? $"{minAge}yo"
                        : $"{minAge}yo+";
                    featureVector["AgeRestriction"] = restriction;
                }
                else
                {
                    featureVector["AgeRestriction"] = "Unknown";
                }
            }

            var resolvedRaceType = string.IsNullOrWhiteSpace(raceType)
                ? parsed.RaceType
                : raceType.Trim();
            if (!string.IsNullOrWhiteSpace(resolvedRaceType))
            {
                SetIfMissing("RaceType", resolvedRaceType);
            }
            else
            {
                MarkMissing("race type");
            }

            var resolvedGoing = string.IsNullOrWhiteSpace(going)
                ? parsed.Going
                : going.Trim();
            if (!string.IsNullOrWhiteSpace(resolvedGoing))
            {
                var normalizedGoing = resolvedGoing.Trim();
                if (featureVector.TryGetValue("Going", out var existingGoing) && existingGoing != null)
                {
                    var existingText = existingGoing switch
                    {
                        string s => s.Trim(),
                        _ => existingGoing.ToString()?.Trim()
                    };

                    if (!string.Equals(existingText, normalizedGoing, StringComparison.OrdinalIgnoreCase))
                    {
                        featureVector["Going"] = normalizedGoing;
                    }
                }
                else
                {
                    featureVector["Going"] = normalizedGoing;
                }
            }
            else
            {
                MarkMissing("going description");
            }

            if (!string.IsNullOrWhiteSpace(parsed.Surface))
            {
                SetIfMissing("Surface", parsed.Surface);
            }
            else
            {
                MarkMissing("surface type");
            }

            int distanceYardsValue = parsed.DistanceYards > 0
                  ? parsed.DistanceYards
                  : (featureVector.TryGetValue("DistanceYards", out var distanceObj) && TryGetInt(distanceObj, out var resolvedDistance)
                      ? resolvedDistance
                      : 0);

            if (distanceYardsValue > 0)
            {
                SetIfMissing("DistanceYards", distanceYardsValue);
            }
            else
            {
                MarkMissing("race distance");
            }

            if (!string.IsNullOrWhiteSpace(parsed.DistanceText))
            {
                SetIfMissing("DistanceText", parsed.DistanceText);
            }
            else
            {
                MarkMissing("distance description");
            }
            string? distanceBucket = null;
            if (distanceYardsValue > 0)
            {
                distanceBucket = DistanceBucketFromYards(distanceYardsValue);
                SetIfMissing("DistanceBucket", distanceBucket);
            }
            bool needsDistanceChange = HasMissingValue(featureVector, "DistanceChangeFromLast");
            bool needsDistanceRatio = HasMissingValue(featureVector, "DistanceRatioFromAverage");
            HorseDistanceStats? distanceStats = null;
            if ((needsDistanceChange || needsDistanceRatio) && !string.IsNullOrWhiteSpace(flow?.HorseName))
            {
                try
                {
                    distanceStats = _repo.GetHorseDistanceStatsByHorseName(flow.HorseName);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to resolve historical distance stats for {flow?.HorseName ?? "unknown horse"}: {ex.Message}");
                }
            }

            if (needsDistanceChange)
            {
                float change = 0f;
                if (distanceStats.HasValue && distanceStats.Value.LastDistanceYards.HasValue && distanceYardsValue > 0)
                {
                    change = distanceYardsValue - distanceStats.Value.LastDistanceYards.Value;
                }

                featureVector["DistanceChangeFromLast"] = change;
            }

            if (needsDistanceRatio)
            {
                float ratio = 1f;
                if (distanceStats.HasValue &&
                    distanceStats.Value.AverageDistanceYards.HasValue &&
                    distanceStats.Value.AverageDistanceYards.Value > 0d &&
                    distanceYardsValue > 0)
                {
                    ratio = distanceYardsValue / (float)distanceStats.Value.AverageDistanceYards.Value;
                }

                featureVector["DistanceRatioFromAverage"] = ratio;
            }
            foreach (var bucket in DistanceBuckets)
            {
                float bucketValue = distanceBucket != null && bucket.Equals(distanceBucket, StringComparison.OrdinalIgnoreCase)
                    ? relativeDraw
                    : 0f;
                SetIfMissing($"RelativeDraw_{bucket}", bucketValue);
            }
            if (raceDate.HasValue)
            {
                var date = raceDate.Value.Date;

                if (scheduledOff.HasValue)
                {
                    var minutes = scheduledOff.Value.TotalMinutes;
                    var timeAngle = 2d * Math.PI * minutes / (24d * 60d);
                    SetIfMissing("TimeOfDaySin", Math.Sin(timeAngle));
                    SetIfMissing("TimeOfDayCos", Math.Cos(timeAngle));
                }
                else
                {
                    SetIfMissing("TimeOfDaySin", 0d);
                    SetIfMissing("TimeOfDayCos", 0d);
                    MarkMissing("scheduled off time");
                }

                int month = date.Month;
                var monthAngle = 2d * Math.PI * month / 12d;
                SetIfMissing("RaceMonthSin", Math.Sin(monthAngle));
                SetIfMissing("RaceMonthCos", Math.Cos(monthAngle));

                int dayOfWeek = (int)date.DayOfWeek;
                var dowAngle = 2d * Math.PI * dayOfWeek / 7d;
                SetIfMissing("RaceDayOfWeekSin", Math.Sin(dowAngle));
                SetIfMissing("RaceDayOfWeekCos", Math.Cos(dowAngle));

                SetIfMissing("IsWeekend", date.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday);

                int season = (month % 12) / 3;
                var seasonAngle = 2d * Math.PI * season / 4d;
                SetIfMissing("SeasonSin", Math.Sin(seasonAngle));
                SetIfMissing("SeasonCos", Math.Cos(seasonAngle));
            }
            else
            {
                MarkMissing("race date");
            }
            ApplyNeutralFeatureFallbacks(featureVector);
        }

        private static void ApplyNeutralFeatureFallbacks(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return;
            }

            foreach (var kvp in NeutralFeatureFallbacks)
            {
                if (!featureVector.TryGetValue(kvp.Key, out var existing) || existing == null)
                {
                    featureVector[kvp.Key] = kvp.Value;
                    continue;
                }

                if (existing is string s && string.IsNullOrWhiteSpace(s))
                {
                    featureVector[kvp.Key] = kvp.Value;
                }
            }

            foreach (var window in PerformanceWindowSizes)
            {
                foreach (var prefix in PerformanceWindowPrefixes)
                {
                    var key = prefix + window;
                    if (!featureVector.TryGetValue(key, out var existing) || existing == null)
                    {
                        featureVector[key] = 0f;
                    }
                }
            }
        }
        
        private static ParsedRaceMetadata ParseRaceMetadata(RaceDayReport race)
        {
            var tokens = EnumerateDetailTokens(race.RaceDetails)
                .Concat(EnumerateDetailTokens(race.RaceTitle))
                .Concat(EnumerateDetailTokens(race.RaceType))
                .Concat(EnumerateDetailTokens(race.Going))
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
        private static readonly IReadOnlyDictionary<string, string> RaceTypeTokenMappings =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["handicap"] = "Handicap",
                ["hcap"] = "Handicap",
                ["hcp"] = "Handicap",
                ["nursery"] = "Nursery",
                ["novice"] = "Novice",
                ["nov"] = "Novice",
                ["stakes"] = "Stakes",
                ["stks"] = "Stakes",
                ["stk"] = "Stakes",
                ["listed"] = "Listed",
                ["maiden"] = "Maiden",
                ["mdn"] = "Maiden",
                ["claiming"] = "Claiming",
                ["claim"] = "Claiming",
                ["claimer"] = "Claiming",
                ["clm"] = "Claiming",
                ["selling"] = "Selling",
                ["sell"] = "Selling",
                ["allowance"] = "Allowance",
                ["allow"] = "Allowance",
                ["conditions"] = "Conditions",
                ["condition"] = "Conditions",
                ["apprentice"] = "Apprentice",
                ["app"] = "Apprentice",
                ["amateur"] = "Amateur",
                ["mares"] = "Mares",
                ["mare"] = "Mares",
                ["fillies"] = "Fillies",
                ["filly"] = "Fillies",
                ["hurdle"] = "Hurdle",
                ["hrd"] = "Hurdle",
                ["hdle"] = "Hurdle",
                ["chase"] = "Chase",
                ["chs"] = "Chase",
                ["ch"] = "Chase",
                ["steeplechase"] = "Chase",
                ["bumper"] = "NH Flat",
                ["nhflat"] = "NH Flat",
                ["nhf"] = "NH Flat",
                ["inhf"] = "NH Flat",
                ["flat"] = "Flat",
                ["beg"] = "Beginners",
                ["beginner"] = "Beginners",
                ["beginners"] = "Beginners",
                ["g1"] = "Group 1",
                ["g2"] = "Group 2",
                ["g3"] = "Group 3",
                ["group1"] = "Group 1",
                ["group2"] = "Group 2",
                ["group3"] = "Group 3",
                ["grade1"] = "Grade 1",
                ["grade2"] = "Grade 2",
                ["grade3"] = "Grade 3",
                ["gr1"] = "Grade 1",
                ["gr2"] = "Grade 2",
                ["gr3"] = "Grade 3"
            };
        private const string RunnerDetailsExpansionScript = @"
            const row = arguments[0];
            const index = arguments[1];
            if (!row) { return false; }

            const normalize = value => {
                if (value === null || value === undefined) { return ''; }
                return String(value).toLowerCase().trim();
            };

            const hasExpandedState = element => {
                if (!element) { return false; }
                const attributes = [
                    element.getAttribute && element.getAttribute('aria-expanded'),
                    element.getAttribute && element.getAttribute('data-state'),
                    element.getAttribute && element.getAttribute('data-expanded'),
                    element.getAttribute && element.getAttribute('data-open'),
                    element.getAttribute && element.getAttribute('aria-pressed')
                ].map(normalize);
                if (attributes.includes('true') || attributes.includes('expanded') || attributes.includes('open')) { return true; }
                const classAttr = normalize(element.getAttribute && element.getAttribute('class'));
                if (classAttr.includes('expanded') || classAttr.includes('open') || classAttr.includes('active')) { return true; }
                return false;
            };

            const hasVisibleDetails = element => {
                if (!element) { return false; }
                const detailSelectors = [
                    '.runner-timeform-details',
                    '.runner-timeform',
                    '.timeform-expandable',
                    '.runner-expanded-details',
                    '.runner-info-expanded'
                ];
                for (const selector of detailSelectors) {
                    const detail = element.querySelector && element.querySelector(selector);
                    if (detail) {
                        const style = window.getComputedStyle(detail);
                        if (style && style.display !== 'none' && style.visibility !== 'hidden' && style.height !== '0px' && style.maxHeight !== '0px') {
                            return true;
                        }
                    }
                }
                return false;
            };

            const resolveIcon = () => {
                const selectors = [
                    '.runner-timeform-info-icon.runner-timeform-info-icon--clickable',
                    '.runner-timeform-info-icon--clickable',
                    '.runner-timeform-info-icon',
                    '[data-testid=""runner-timeform-info-icon""]',
                    '[data-test-id=""runner-timeform-info-icon""]',
                    '.timeform-icon',
                    '.details-icon'
                ];
                for (const selector of selectors) {
                    const candidate = row.querySelector(selector);
                    if (candidate) {
                        const clickable = candidate.closest && candidate.closest('.runner-timeform-info-icon--clickable');
                        return clickable || candidate;
                    }
                }
                const svg = row.querySelector('.runner-timeform-info-icon svg, .runner-timeform-info-icon path');
                if (svg) {
                    const clickable = svg.closest && (svg.closest('.runner-timeform-info-icon--clickable') || svg.closest('.runner-timeform-info-icon'));
                    if (clickable) { return clickable; }
                    return svg;
                }
                if (typeof index === 'number' && Number.isFinite(index)) {
                    const base = `#main-wrapper > div > div.scrollable-panes-height-taker > div > ui-view > div > div > div.bf-col-xxl-17-24.bf-col-xl-16-24.bf-col-lg-16-24.bf-col-md-15-24.bf-col-sm-14-24.bf-col-14-24.center-column.bfMarketSettingsSpace.bf-module-loading.nested-scrollable-pane-parent.market-settings-space > div.scrollable-panes-height-taker.height-taker-helper > div > div.bf-row.main-mv-container > div > bf-main-market > bf-main-marketview > div > div.main-mv-runners-list-wrapper > bf-marketview-runners-list.runners-list-unpinned > div > div > div > table > tbody > tr:nth-child(${index}) > td.new-runner-info.with-runner-timeform-info > div.runner-timeform-info-icon.runner-timeform-info-icon--clickable`;
                    const absoluteSelectors = [base, `${base} > svg`, `${base} > svg > path`];
                    for (const selector of absoluteSelectors) {
                        const candidate = document.querySelector(selector);
                        if (candidate) {
                            const clickable = candidate.closest && candidate.closest('.runner-timeform-info-icon--clickable');
                            return clickable || candidate;
                        }
                    }
                }
                return null;
            };

            const icon = resolveIcon();
            if (!icon) { return false; }

            const container = icon.closest && icon.closest('.new-runner-info, .runner-timeform-info, .runner-info, tr, td');
            if (hasExpandedState(icon) || hasExpandedState(container) || hasVisibleDetails(container)) {
                return true;
            }

            if (icon.scrollIntoView) {
                icon.scrollIntoView({ block: 'center', inline: 'center' });
            }

            if (icon.dispatchEvent) {
                icon.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
            }
            if (icon.click) {
                icon.click();
            }

            const expandedNow = hasExpandedState(icon) || hasExpandedState(container) || hasVisibleDetails(container);
            return expandedNow;
        ";
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
        private static void ExpandRunnerTimeformDetails(IWebDriver driver, IReadOnlyList<IWebElement> runnerRows)
        {
            if (driver is not IJavaScriptExecutor js || runnerRows == null || runnerRows.Count == 0)
            {
                return;
            }

            for (var i = 0; i < runnerRows.Count; i++)
            {
                var row = runnerRows[i];
                if (row == null)
                {
                    continue;
                }

                try
                {
                    var result = js.ExecuteScript(RunnerDetailsExpansionScript, row, i + 1);
                    var expanded = result is bool flag && flag;

                    if (!expanded)
                    {
                        TryFallbackRunnerDetailsClick(js, row);
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"\tFailed to expand runner details for row {i + 1}: {ex.Message}");
                }
            }
        }

        private static void TryFallbackRunnerDetailsClick(IJavaScriptExecutor js, IWebElement row)
        {
            if (row == null)
            {
                return;
            }

            try
            {
                IWebElement? target = row
                    .FindElements(By.CssSelector(
                        ".runner-timeform-info-icon--clickable, .runner-timeform-info-icon.runner-timeform-info-icon--clickable"))
                    .FirstOrDefault();

                if (target == null)
                {
                    var svgCandidate = row
                        .FindElements(By.CssSelector(
                            ".runner-timeform-info-icon svg, .runner-timeform-info-icon path"))
                        .FirstOrDefault();

                    if (svgCandidate != null)
                    {
                        IWebElement? clickable = null;
                        try
                        {
                            clickable = svgCandidate.FindElement(By.XPath(
                                "ancestor-or-self::*[contains(@class,'runner-timeform-info-icon--clickable')][1]"));
                        }
                        catch (NoSuchElementException)
                        {
                            try
                            {
                                clickable = svgCandidate.FindElement(By.XPath(".."));
                            }
                            catch (NoSuchElementException)
                            {
                                clickable = null;
                            }
                        }

                        target = clickable ?? svgCandidate;
                    }
                }

                if (target == null)
                {
                    return;
                }

                try
                {
                    target.Click();
                }
                catch (Exception)
                {
                    js.ExecuteScript(
                        "if (arguments[0]) { arguments[0].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window })); }",
                        target);
                }
            }
            catch (Exception)
            {
                // ignore fallback failures; details expansion is best-effort
            }
        }
        private static string? TryDetectRaceType(IEnumerable<string> tokens, string? title)
        {
            static string? ComposeRaceType(IReadOnlyCollection<string> components)
            {
                if (components.Count == 0)
                {
                    return null;
                }

                return components.Count == 1
                    ? components.First()
                    : string.Join(" ", components);
            }

            var componentList = ExtractRaceTypeComponents(tokens);
            var raceType = ComposeRaceType(componentList);
            if (!string.IsNullOrEmpty(raceType))
            {
                return raceType;
            }

            if (!string.IsNullOrWhiteSpace(title))
            {
                var titleComponents = ExtractRaceTypeComponents(EnumerateDetailTokens(title));
                raceType = ComposeRaceType(titleComponents);
                if (!string.IsNullOrEmpty(raceType))
                {
                    return raceType;
                }
            }
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
        private static IReadOnlyList<string> ExtractRaceTypeComponents(IEnumerable<string> source)
        {
            var results = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var token in source)
            {
                if (string.IsNullOrWhiteSpace(token))
                {
                    continue;
                }

                foreach (var component in ExpandRaceTypeToken(token))
                {
                    if (seen.Add(component))
                    {
                        results.Add(component);
                    }
                }
            }

            return results;
        }

        private static IEnumerable<string> ExpandRaceTypeToken(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                yield break;
            }

            var builder = new StringBuilder();

            foreach (var c in token)
            {
                if (char.IsLetterOrDigit(c))
                {
                    builder.Append(c);
                    continue;
                }

                if (builder.Length == 0)
                {
                    continue;
                }

                var part = builder.ToString();
                builder.Clear();

                if (TryMapRaceTypeToken(part, out var mapped))
                {
                    yield return mapped;
                }
            }

            if (builder.Length > 0)
            {
                var part = builder.ToString();
                if (TryMapRaceTypeToken(part, out var mapped))
                {
                    yield return mapped;
                }
            }

            if (TryMapRaceTypeToken(token, out var entireTokenMapped))
            {
                yield return entireTokenMapped;
            }
        }

        private static bool TryMapRaceTypeToken(string token, out string canonical)
        {
            canonical = string.Empty;
            if (string.IsNullOrWhiteSpace(token))
            {
                return false;
            }

            var normalized = NormalizeRaceTypeToken(token);
            if (string.IsNullOrEmpty(normalized))
            {
                return false;
            }

            if (RaceTypeTokenMappings.TryGetValue(normalized, out canonical!))
            {
                return true;
            }

            return false;
        }

        private static string NormalizeRaceTypeToken(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var builder = new StringBuilder(value.Length);
            foreach (var c in value)
            {
                if (char.IsLetterOrDigit(c))
                {
                    builder.Append(char.ToLowerInvariant(c));
                }
            }

            return builder.ToString();
        }
        private static (byte? Age, byte? WeightLbs, string? WeightText) ParseRunnerAgeWeight(string? raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
            {
                return (null, null, null);
            }

            var normalized = Regex.Replace(raw.Replace('\u00A0', ' '), "\\s+", " ").Trim();
            if (normalized.Length == 0)
            {
                return (null, null, null);
            }

            static byte? TryParseByte(string? value)
            {
                if (byte.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                {
                    return parsed;
                }

                return null;
            }

            byte? age = null;
            var ageMatch = AgeParentheticalRegex.Match(normalized);
            if (ageMatch.Success)
            {
                age = TryParseByte(ageMatch.Groups["age"].Value);
            }

            if (!age.HasValue)
            {
                var wordMatch = AgeWordRegex.Match(normalized);
                if (wordMatch.Success)
                {
                    age = TryParseByte(wordMatch.Groups["age"].Value);
                }
            }

            if (!age.HasValue)
            {
                var leadingMatch = AgeLeadingRegex.Match(normalized);
                if (leadingMatch.Success)
                {
                    age = TryParseByte(leadingMatch.Groups["age"].Value);
                }
            }

            string? weightText = null;
            byte? weightLbs = null;

            var dashMatch = WeightDashRegex.Match(normalized);
            if (dashMatch.Success)
            {
                var stone = TryParseByte(dashMatch.Groups["stone"].Value);
                var pounds = TryParseByte(dashMatch.Groups["pounds"].Value);
                if (stone.HasValue && pounds.HasValue)
                {
                    weightText = $"{stone.Value}-{pounds.Value}";
                    var total = stone.Value * 14 + pounds.Value;
                    if (total <= byte.MaxValue)
                    {
                        weightLbs = (byte)total;
                    }
                }
            }

            if (!weightLbs.HasValue)
            {
                var stoneMatch = WeightStoneRegex.Match(normalized);
                if (stoneMatch.Success)
                {
                    var stone = TryParseByte(stoneMatch.Groups["stone"].Value);
                    var poundsGroup = stoneMatch.Groups["pounds"];
                    var pounds = poundsGroup.Success ? TryParseByte(poundsGroup.Value) ?? (byte)0 : (byte)0;

                    if (stone.HasValue)
                    {
                        weightText = $"{stone.Value}-{pounds}";
                        var total = stone.Value * 14 + pounds;
                        if (total <= byte.MaxValue)
                        {
                            weightLbs = (byte)total;
                        }
                    }
                }
            }

            if (!weightLbs.HasValue)
            {
                var lbsMatch = WeightLbsRegex.Match(normalized);
                if (lbsMatch.Success)
                {
                    var pounds = TryParseByte(lbsMatch.Groups["pounds"].Value);
                    if (pounds.HasValue)
                    {
                        weightText = pounds.Value.ToString(CultureInfo.InvariantCulture);
                        weightLbs = pounds.Value;
                    }
                }
            }

            return (age, weightLbs, string.IsNullOrWhiteSpace(weightText) ? null : weightText);
        }

        private static string? NormalizeTrainerName(string? raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
            {
                return null;
            }

            var normalized = TrainerPrefixRegex.Replace(raw.Replace('\u00A0', ' '), string.Empty).Trim();
            return string.IsNullOrWhiteSpace(normalized) ? null : normalized;
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
                            AppendMarketFallbackReason(flow, "Degenerate model outputs; using market-implied probability");
                            anyFallbackApplied = true;
                        }
                        else
                        {
                            flow.AiOdds = null;
                            flow.AiProbabilityMarketDerived = false;
                            AppendMarketFallbackReason(flow, "Degenerate model outputs; using market-implied probability");
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
                    flow.AiProbabilityClampedToMarket = false;
                    flow.AiProbabilityClampTarget = null;
                }
                Console.WriteLine($"\t\tRaw probabilities summed to ~0; distributing uniform probability {uniform.ToString("0.####", CultureInfo.InvariantCulture)} across valid runners.");
                return;
            }

            var clamped = valid
                .Where(f => f.AiProbabilityClampedToMarket
                            && f.AiProbabilityClampTarget.HasValue
                            && double.IsFinite(f.AiProbabilityClampTarget.Value)
                            && f.AiProbabilityClampTarget.Value > 0)
                .ToList();

            if (clamped.Count == 0)
            {
                foreach (var flow in valid)
                {
                    flow.AiOdds = Math.Max(flow.AiOdds!.Value / sum, 0d);
                }

                if (valid.Count > 1)
                {
                    const double minimumNormalizedProbability = 1e-6;
                    var normalizedMinimum = valid.Min(f => f.AiOdds!.Value);
                    var uniformProbability = 1.0 / valid.Count;
                    var targetFloor = Math.Min(minimumNormalizedProbability, uniformProbability * 0.5);

                    if (normalizedMinimum < targetFloor && uniformProbability > normalizedMinimum)
                    {
                        var blendWeight = (targetFloor - normalizedMinimum) / (uniformProbability - normalizedMinimum);
                        blendWeight = Math.Clamp(blendWeight, 0d, 1d);

                        if (blendWeight > 0d)
                        {
                            foreach (var flow in valid)
                            {
                                var blended = (1d - blendWeight) * flow.AiOdds!.Value + blendWeight * uniformProbability;
                                flow.AiOdds = Math.Max(blended, 0d);
                            }

                            var blendedSum = valid.Sum(f => f.AiOdds!.Value);
                            if (blendedSum > 0d && Math.Abs(blendedSum - 1d) > 1e-12)
                            {
                                foreach (var flow in valid)
                                {
                                    flow.AiOdds = Math.Max(flow.AiOdds!.Value / blendedSum, 0d);
                                }
                            }

                            Console.WriteLine(
                                $"\t\tApplied probability smoothing with blend={blendWeight.ToString("0.####E+0", CultureInfo.InvariantCulture)} " +
                                $"to enforce minimum normalized probability {targetFloor.ToString("0.####", CultureInfo.InvariantCulture)}.");
                        }
                    }
                }

                var normalizedSum = valid.Sum(f => f.AiOdds!.Value);
                Console.WriteLine($"\t\tNormalized probability sum: {normalizedSum.ToString("0.####", CultureInfo.InvariantCulture)}.");
                return;
            }

            var adjustable = valid.Except(clamped).ToList();
            var reservedProbability = clamped.Sum(f => f.AiProbabilityClampTarget!.Value);
            reservedProbability = Math.Max(reservedProbability, 0d);

            Console.WriteLine($"\t\tReserved {reservedProbability.ToString("0.####", CultureInfo.InvariantCulture)} probability mass for {clamped.Count} market-clamped runner(s).");

            if (reservedProbability >= 1d)
            {
                if (reservedProbability <= double.Epsilon)
                {
                    var uniform = 1.0 / valid.Count;
                    foreach (var flow in valid)
                    {
                        flow.AiOdds = uniform;
                        flow.AiProbabilityClampedToMarket = false;
                        flow.AiProbabilityClampTarget = null;
                    }
                    Console.WriteLine("\t\tReserved mass exhausted distribution; reverted to uniform probabilities.");
                    return;
                }

                var scale = 1d / reservedProbability;
                foreach (var flow in clamped)
                {
                    flow.AiOdds = Math.Max(flow.AiProbabilityClampTarget!.Value * scale, 0d);
                }
                foreach (var flow in adjustable)
                {
                    flow.AiOdds = 0d;
                }

                var normalizedReserved = clamped.Sum(f => f.AiOdds!.Value);
                Console.WriteLine($"\t\tReserved probability exceeded total mass; scaled clamped runners to {normalizedReserved.ToString("0.####", CultureInfo.InvariantCulture)} and zeroed others.");

                return;
            }

            foreach (var flow in clamped)
            {
                flow.AiOdds = Math.Max(flow.AiProbabilityClampTarget!.Value, 0d);
            }

            var available = 1d - reservedProbability;
            if (adjustable.Count > 0)
            {
                var adjustableRawSum = adjustable.Sum(f => f.AiOdds!.Value);
                if (adjustableRawSum <= double.Epsilon)
                {
                    var uniformAdjustable = available / adjustable.Count;
                    foreach (var flow in adjustable)
                    {
                        flow.AiOdds = Math.Max(uniformAdjustable, 0d);
                    }
                    Console.WriteLine("\t\tAdjustable runners had non-positive mass; distributed remaining probability uniformly amongst them.");
                }
                else
                {
                    var scale = available / adjustableRawSum;
                    foreach (var flow in adjustable)
                    {
                        flow.AiOdds = Math.Max(flow.AiOdds!.Value * scale, 0d);
                    }
                }

                if (adjustable.Count > 1 && available > 0d)
                {
                    const double minimumNormalizedProbability = 1e-6;
                    var normalizedMinimum = adjustable.Min(f => f.AiOdds!.Value);
                    var uniformProbability = available / adjustable.Count;
                    var targetFloor = Math.Min(minimumNormalizedProbability, uniformProbability * 0.5);

                    if (uniformProbability > 0d && normalizedMinimum < targetFloor)
                    {
                        var blendWeight = (targetFloor - normalizedMinimum) / (uniformProbability - normalizedMinimum);
                        blendWeight = Math.Clamp(blendWeight, 0d, 1d);

                        if (blendWeight > 0d)
                        {
                            foreach (var flow in adjustable)
                            {
                                var blended = (1d - blendWeight) * flow.AiOdds!.Value + blendWeight * uniformProbability;
                                flow.AiOdds = Math.Max(blended, 0d);
                            }

                            var blendedSum = adjustable.Sum(f => f.AiOdds!.Value);
                            if (blendedSum > 0d)
                            {
                                var rescale = available / blendedSum;
                                foreach (var flow in adjustable)
                                {
                                    flow.AiOdds = Math.Max(flow.AiOdds!.Value * rescale, 0d);
                                }
                            }

                            Console.WriteLine(
                                $"\t\tApplied probability smoothing to adjustable runners with blend={blendWeight.ToString("0.####E+0", CultureInfo.InvariantCulture)} " +
                                $"to enforce minimum normalized probability {targetFloor.ToString("0.####", CultureInfo.InvariantCulture)}.");
                        }
                    }
                }
            }
            else
            {
                var actualReserved = clamped.Sum(f => f.AiOdds!.Value);
                if (Math.Abs(actualReserved - 1d) > 1e-12)
                {
                    var scale = actualReserved > 0d ? 1d / actualReserved : 0d;
                    foreach (var flow in clamped)
                    {
                        flow.AiOdds = scale > 0d ? Math.Max(flow.AiOdds!.Value * scale, 0d) : 0d;
                    }
                }
            }

            var clampSumAfter = clamped.Sum(f => f.AiOdds!.Value);
            var adjustableSumAfter = adjustable.Sum(f => f.AiOdds!.Value);
            var totalAfterClamp = clampSumAfter + adjustableSumAfter;

            if (Math.Abs(totalAfterClamp - 1d) > 1e-8 && totalAfterClamp > 0d)
            {
                if (adjustable.Count > 0)
                {
                    var desiredAdjustableTotal = Math.Max(1d - clampSumAfter, 0d);
                    if (adjustableSumAfter > 0d)
                    {
                        var rescaleAdjustable = desiredAdjustableTotal / adjustableSumAfter;
                        foreach (var flow in adjustable)
                        {
                            flow.AiOdds = Math.Max(flow.AiOdds!.Value * rescaleAdjustable, 0d);
                        }
                    }
                    else if (desiredAdjustableTotal > 0d)
                    {
                        var uniformAdjustable = desiredAdjustableTotal / adjustable.Count;
                        foreach (var flow in adjustable)
                        {
                            flow.AiOdds = Math.Max(uniformAdjustable, 0d);
                        }
                    }

                    clampSumAfter = clamped.Sum(f => f.AiOdds!.Value);
                    adjustableSumAfter = adjustable.Sum(f => f.AiOdds!.Value);
                    totalAfterClamp = clampSumAfter + adjustableSumAfter;
                }

                if (Math.Abs(totalAfterClamp - 1d) > 1e-8 && totalAfterClamp > 0d)
                {
                    var scale = 1d / totalAfterClamp;
                    foreach (var flow in valid)
                    {
                        flow.AiOdds = Math.Max(flow.AiOdds!.Value * scale, 0d);
                    }
                }
            }

            var normalizedSumWithClamp = valid.Sum(f => f.AiOdds!.Value);
            Console.WriteLine($"\t\tNormalized probability sum: {normalizedSumWithClamp.ToString("0.####", CultureInfo.InvariantCulture)}.");
        }
        private static void ApplyMarketFallbackForUnmatchedRunners(IReadOnlyList<RunnerFlow> flows)
        {
            if (flows == null || flows.Count == 0) return; // return early if list is null or empty

            foreach (var flow in flows)
            {
                if (flow == null || flow.MatchedDatabaseRecord) continue; // skip matched or null entries

                var identifier = !string.IsNullOrWhiteSpace(flow.HorseName) ? flow.HorseName! : (flow.SelectionId ?? "unknown"); // determine identifier

                if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 1m)
                {
                    var marketProbability = 1.0 / (double)flow.BackPrice1.Value; // calculate implied probability
                    flow.AiOdds = marketProbability; flow.AiProbabilityMarketDerived = true;
                    flow.AiProbabilityClampedToMarket = false;
                    flow.AiProbabilityClampTarget = null;
                    AppendMarketFallbackReason(flow, "Database record not found; using market-implied probability");
                    Console.WriteLine($"\t\tNo database match for {identifier}; using market-implied probability {marketProbability.ToString("0.####", CultureInfo.InvariantCulture)} as AI odds.");
                }
                else
                {
                    flow.AiOdds = null; flow.AiProbabilityMarketDerived = false;
                    flow.AiProbabilityFallbackReason = null;
                    Console.WriteLine($"\t\tNo database match for {identifier} and no usable back price; AI odds remain unavailable.");
                }
            }
        }
        private static void AppendMarketFallbackReason(RunnerFlow? flow, string detail)
        {
            if (flow == null || string.IsNullOrWhiteSpace(detail))
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(flow.AiProbabilityFallbackReason))
            {
                flow.AiProbabilityFallbackReason = detail;
                return;
            }

            if (flow.AiProbabilityFallbackReason.IndexOf(detail, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return;
            }

            flow.AiProbabilityFallbackReason = $"{flow.AiProbabilityFallbackReason}; {detail}";
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
                    flow.AiProbabilityClampedToMarket = false;
                    flow.AiProbabilityClampTarget = null;
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
                        flow.AiProbabilityClampedToMarket = false;
                        flow.AiProbabilityClampTarget = null;
                    }
                }
                else
                {
                    foreach (var kvp in weights)
                    {
                        var share = leftoverMass * (kvp.Value / weightSum);
                        kvp.Key.AiOdds = Math.Max(share, 0d);
                        kvp.Key.AiProbabilityClampedToMarket = false;
                        kvp.Key.AiProbabilityClampTarget = null;
                    }
                }
            }

            winner.AiOdds = winnerProbability;
            winner.AiProbabilityClampedToMarket = false;
            winner.AiProbabilityClampTarget = null;

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
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var trimmed = text.Trim();

            if (byte.TryParse(trimmed, NumberStyles.Integer, CultureInfo.InvariantCulture, out var directValue))
            {
                return directValue;
            }

            var digitMatch = Regex.Match(trimmed, @"\d+");
            if (digitMatch.Success && byte.TryParse(digitMatch.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var matchedValue))
            {
                return matchedValue;
            }

            return null;
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