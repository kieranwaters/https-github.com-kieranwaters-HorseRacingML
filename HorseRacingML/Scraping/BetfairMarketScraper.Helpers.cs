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
using System.Collections;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        private static readonly Regex DistanceComponentRegex = new("(?<value>[0-9]+(?:\\.[0-9]+)?)\\s*(?<unit>[mfy])", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex ClassRegex = new("class\\s*(?<value>[0-9])", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex MarketTitleExchangeSuffixRegex = new(@"\s*(?:[»|,-]\s*)?BetfairT?\s*Exchange.*$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex MarketTitleSiteSuffixRegex = new(@"\s*(?:[-–—]|\|)\s*Betfair.*$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex LeadingRaceTypeRegex = new(
            @"^\s*(?<type>(?:[A-Za-z'\-]+(?:\s+[A-Za-z'\-]+)*)|(?:G[1-3])|(?:Group\s+[1-3])|(?:Grade\s+[1-3]))",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeParentheticalRegex = new(@"\((?<age>\d{1,2})\)", RegexOptions.Compiled);
        private static readonly Regex AgeLabelRegex = new(@"\b(?:age|aged)\s*[:\-]?\s*(?<age>\d{1,2})\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeWordRegex = new(@"\b(?<age>\d{1,2})\s*(?:yo|yr|yrs|year|years)\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeLeadingRegex = new(@"^(?<age>\d{1,2})(?=\s|$)", RegexOptions.Compiled);
        private static readonly Regex WeightDashRegex = new(@"\b(?<stone>\d{1,2})\s*[-/]\s*(?<pounds>\d{1,2})\b", RegexOptions.Compiled);
        private static readonly Regex WeightStoneRegex = new(@"\b(?<stone>\d{1,2})\s*(?:st|stone|stones)\s*(?<pounds>\d{1,2})?\s*(?:lb|lbs|pound|pounds)?\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex WeightLbsRegex = new(@"\b(?<pounds>\d{2,3})\s*(?:lb|lbs|pound|pounds)\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex TrainerPrefixRegex = new(@"^(?:trainer|trainers?|t:)\s*[:\-]?\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly string[] DistanceBuckets = { "Sprint", "Middle", "Long" };
        private static readonly int[] PerformanceWindowSizes = { 1, 3, 5, 10, 15, 20, 25, 30, 50, 100 };
        private static readonly Regex AgeRestrictionRangeRegex = new(@"\b(?<min>\d{1,2})\s*(?:[-–]\s*|\s+to\s+)(?<max>\d{1,2})\s*(?:y[\./-]?\s*o|yrs?|years?)\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeRestrictionWordRegex = new(@"\b(?<age>\d{1,2})\s*(?:y[\./-]?\s*o|yrs?|years?)\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeRestrictionLabelRegex = new(@"\b(?:aged|age)\s*(?<age>\d{1,2})\b", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex AgeRestrictionQualifierRegex = new(@"^(?<qualifier>\+|plus|and\s*up(?:wards)?|&\s*up|upwards|up|over|only)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private static readonly Regex MarketTitleDateCandidateRegex = new(
            @"(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s+)?\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}(?:\s+\d{2,4})?",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);
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
        private static readonly IReadOnlyDictionary<int, float> ClassRatingBaselines =
           new Dictionary<int, float>
           {
               [1] = 105f,
               [2] = 100f,
               [3] = 95f,
               [4] = 90f,
               [5] = 85f,
               [6] = 80f,
               [7] = 75f
           };
        private static readonly string[] ClassDependentFeatureKeys =
        {
            "ClassWinRate",
            "ClassAvgNorm",
            "LastClassNormPos",
            "TrainerClassWinRate",
            "TrainerClassAvgNorm",
            "LastTrainerClassNormPos",
            "JockeyClassWinRate",
            "JockeyClassAvgNorm",
            "LastJockeyClassNormPos",
            "JockeyGoingDistanceWinRate",
            "JockeyGoingDistanceAvgNorm",
            "LastJockeyGoingDistanceNormPos",
            "TrainerJockeyCourseWinRate"
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
        private void PromoteClassDependentFallbacks(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return;
            }

            if (!TryGetMeaningfulValue(featureVector, "Class", out _))
            {
                return;
            }

            foreach (var key in ClassDependentFeatureKeys)
            {
                if (TryGetMeaningfulValue(featureVector, key, out _))
                {
                    continue;
                }

                if (!NeutralFeatureFallbacks.TryGetValue(key, out var fallback) || fallback == null)
                {
                    fallback = 0f;
                }

                featureVector[key] = fallback;
            }
        }

        private void PromotePerformanceWindowFallbacks(Dictionary<string, object?> featureVector)
        {
            if (featureVector == null)
            {
                return;
            }

            var insertedAny = false;

            foreach (var prefix in PerformanceWindowPrefixes)
            {
                foreach (var window in PerformanceWindowSizes)
                {
                    var key = string.Concat(prefix, window.ToString(CultureInfo.InvariantCulture));
                    if (TryGetMeaningfulValue(featureVector, key, out _))
                    {
                        continue;
                    }

                    if (!NeutralFeatureFallbacks.TryGetValue(key, out var fallback) || fallback == null)
                    {
                        fallback = 0f;
                    }

                    featureVector[key] = fallback;
                    insertedAny = true;
                }
            }

            if (!insertedAny)
            {
                return;
            }

            if (featureVector.TryGetValue("RatingAggregatesMissing", out var ratingMissingObj) &&
                ratingMissingObj is bool ratingMissing && ratingMissing)
            {
                featureVector["RatingAggregatesMissing"] = false;
            }
        }
        private void PromoteCalculatedHistoricalFallbacks(
            Dictionary<string, object?> featureVector,
            RunnerFlow? flow,
            IReadOnlyList<RunnerFlow>? flows)
        {
            if (featureVector == null)
            {
                return;
            }

            var ratingWindows = new List<int>();
            foreach (var window in PerformanceWindowSizes)
            {
                var ratingKey = $"AvgRatingLast{window}";
                if (!TryGetMeaningfulValue(featureVector, ratingKey, out _))
                {
                    ratingWindows.Add(window);
                }
            }

            float? ratingFallback = null;
            if (ratingWindows.Count > 0)
            {
                ratingFallback = ResolveBaselineRating(featureVector, flow, flows);
                var historicalRatings = ResolveHistoricalRatings(featureVector, flow, ratingFallback);
                if (historicalRatings.Count > 0)
                {
                    foreach (var window in ratingWindows)
                    {
                        var ratingKey = $"AvgRatingLast{window}";
                        if (TryGetMeaningfulValue(featureVector, ratingKey, out _))
                        {
                            continue;
                        }

                        int count = Math.Min(window, historicalRatings.Count);
                        if (count > 0)
                        {
                            featureVector[ratingKey] = historicalRatings.Take(count).Average();
                        }
                    }
                }

                if (ratingFallback.HasValue)
                {
                    foreach (var window in ratingWindows)
                    {
                        var ratingKey = $"AvgRatingLast{window}";
                        if (!TryGetMeaningfulValue(featureVector, ratingKey, out _))
                        {
                            featureVector[ratingKey] = ratingFallback.Value;
                        }
                    }
                }
            }

            var lifetimeWinRate = ResolveFeatureValue(featureVector, "LifetimeWinRate");
            if (!lifetimeWinRate.HasValue)
            {
                lifetimeWinRate = ResolveFeatureValue(flow?.FeatureValues, "LifetimeWinRate");
            }

            if (!lifetimeWinRate.HasValue)
            {
                var trainerWin = ResolveFeatureValue(featureVector, "TrainerWinRate");
                var jockeyWin = ResolveFeatureValue(featureVector, "JockeyWinRate");
                lifetimeWinRate = CombineAverages(trainerWin, jockeyWin);
            }

            if (lifetimeWinRate.HasValue)
            {
                foreach (var window in PerformanceWindowSizes)
                {
                    var winRateKey = $"WinRateLast{window}";
                    if (!TryGetMeaningfulValue(featureVector, winRateKey, out _))
                    {
                        featureVector[winRateKey] = lifetimeWinRate.Value;
                    }

                    var normKey = $"AvgNormPosLast{window}";
                    if (!TryGetMeaningfulValue(featureVector, normKey, out _))
                    {
                        featureVector[normKey] = ClampNormalizedPosition(1f - lifetimeWinRate.Value);
                    }
                }
            }

            var avgNormLast5 = ResolveFeatureValue(featureVector, "AvgNormPosLast5");
            if (!avgNormLast5.HasValue && lifetimeWinRate.HasValue)
            {
                avgNormLast5 = ClampNormalizedPosition(1f - lifetimeWinRate.Value);
            }

            var classWinRate = ResolveFeatureValue(featureVector, "ClassWinRate");
            if (!classWinRate.HasValue)
            {
                classWinRate = lifetimeWinRate ?? ResolveFeatureValue(featureVector, "TrainerWinRate")
                    ?? ResolveFeatureValue(featureVector, "JockeyWinRate");
            }

            if (classWinRate.HasValue)
            {
                FillIfMissing(featureVector, "ClassWinRate", classWinRate);
                var classAvgNorm = ResolveFeatureValue(featureVector, "ClassAvgNorm")
                    ?? ClampNormalizedPosition(1f - classWinRate.Value);
                FillIfMissing(featureVector, "ClassAvgNorm", classAvgNorm);

                if (!TryGetMeaningfulValue(featureVector, "LastClassNormPos", out _))
                {
                    featureVector["LastClassNormPos"] = classAvgNorm;
                }
            }
            else if (avgNormLast5.HasValue)
            {
                FillIfMissing(featureVector, "LastClassNormPos", avgNormLast5);
            }

            var trainerWinRate = ResolveFeatureValue(featureVector, "TrainerWinRate");
            var trainerSurfaceAvg = ResolveFeatureValue(featureVector, "TrainerSurfaceAvgNorm");
            var trainerGoingAvg = ResolveFeatureValue(featureVector, "TrainerGoingAvgNorm");
            var trainerAvgNorm = trainerSurfaceAvg ?? trainerGoingAvg;
            var lastTrainerSurface = ResolveFeatureValue(featureVector, "LastTrainerSurfaceNormPos");
            var lastTrainerGoing = ResolveFeatureValue(featureVector, "LastTrainerGoingNormPos");
            var lastTrainerNorm = lastTrainerSurface ?? lastTrainerGoing;

            FillIfMissing(featureVector, "TrainerClassWinRate", trainerWinRate);
            FillIfMissing(featureVector, "TrainerClassAvgNorm",
                trainerAvgNorm ?? (trainerWinRate.HasValue ? ClampNormalizedPosition(1f - trainerWinRate.Value) : (float?)null));
            FillIfMissing(featureVector, "LastTrainerClassNormPos",
                lastTrainerNorm ?? trainerAvgNorm ?? (trainerWinRate.HasValue ? ClampNormalizedPosition(1f - trainerWinRate.Value) : (float?)null));

            FillIfMissing(featureVector, "TrainerGoingWinRate", trainerWinRate);
            FillIfMissing(featureVector, "TrainerGoingAvgNorm", trainerGoingAvg ?? trainerSurfaceAvg);
            FillIfMissing(featureVector, "LastTrainerGoingNormPos", lastTrainerGoing ?? lastTrainerSurface);
            FillIfMissing(featureVector, "LastTrainerSurfaceNormPos", lastTrainerSurface ?? lastTrainerGoing);

            var jockeyWinRate = ResolveFeatureValue(featureVector, "JockeyWinRate");
            var jockeySurfaceAvg = ResolveFeatureValue(featureVector, "JockeySurfaceAvgNorm");
            var jockeyGoingAvg = ResolveFeatureValue(featureVector, "JockeyGoingAvgNorm");
            var jockeyAvgNorm = jockeySurfaceAvg ?? jockeyGoingAvg;
            var lastJockeySurface = ResolveFeatureValue(featureVector, "LastJockeySurfaceNormPos");
            var lastJockeyGoing = ResolveFeatureValue(featureVector, "LastJockeyGoingNormPos");
            var lastJockeyDistance = ResolveFeatureValue(featureVector, "LastJockeyDistanceBucketNormPos");

            FillIfMissing(featureVector, "JockeyClassWinRate", jockeyWinRate);
            FillIfMissing(featureVector, "JockeyClassAvgNorm",
                jockeyAvgNorm ?? (jockeyWinRate.HasValue ? ClampNormalizedPosition(1f - jockeyWinRate.Value) : (float?)null));
            FillIfMissing(featureVector, "LastJockeyClassNormPos",
                lastJockeySurface ?? lastJockeyGoing ?? (jockeyAvgNorm ?? (jockeyWinRate.HasValue ? ClampNormalizedPosition(1f - jockeyWinRate.Value) : (float?)null)));

            var jockeyGoingWin = ResolveFeatureValue(featureVector, "JockeyGoingWinRate");
            var jockeyDistanceWin = ResolveFeatureValue(featureVector, "JockeyDistanceBucketWinRate");
            FillIfMissing(featureVector, "JockeyGoingDistanceWinRate", CombineAverages(jockeyGoingWin, jockeyDistanceWin));

            var jockeyDistanceAvg = ResolveFeatureValue(featureVector, "JockeyDistanceBucketAvgNorm");
            FillIfMissing(featureVector, "JockeyGoingDistanceAvgNorm", CombineAverages(jockeyGoingAvg, jockeyDistanceAvg));
            FillIfMissing(featureVector, "LastJockeyGoingDistanceNormPos", CombineAverages(lastJockeyGoing, lastJockeyDistance));

            var trainerSurfaceWin = ResolveFeatureValue(featureVector, "TrainerSurfaceWinRate");
            var jockeySurfaceWin = ResolveFeatureValue(featureVector, "JockeySurfaceWinRate");
            var trainerCourseWin = ResolveFeatureValue(featureVector, "TrainerCourseWinRate");
            var jockeyCourseWin = ResolveFeatureValue(featureVector, "JockeyCourseWinRate");

            FillIfMissing(featureVector, "TrainerJockeyWinRate", CombineAverages(trainerWinRate, jockeyWinRate));
            FillIfMissing(featureVector, "TrainerJockeySurfaceWinRate", CombineAverages(trainerSurfaceWin, jockeySurfaceWin));
            FillIfMissing(featureVector, "TrainerJockeyCourseWinRate", CombineAverages(trainerCourseWin, jockeyCourseWin));
        }

        private static float ClampNormalizedPosition(float value)
        {
            if (float.IsNaN(value))
            {
                return 0f;
            }

            return Math.Clamp(value, -5f, 5f);
        }

        private static float? ResolveFeatureValue(Dictionary<string, object?>? source, string key)
        {
            if (source == null)
            {
                return null;
            }

            if (!TryGetMeaningfulValue(source, key, out var obj))
            {
                return null;
            }

            return TryConvertToSingle(obj);
        }

        private float? ResolveBaselineRating(
            Dictionary<string, object?> featureVector,
            RunnerFlow? flow,
            IReadOnlyList<RunnerFlow>? flows)
        {
            var rating = ResolveFeatureValue(featureVector, "OfficialRating");
            if (rating.HasValue)
            {
                return rating.Value;
            }

            rating = ResolveFeatureValue(flow?.FeatureValues, "OfficialRating");
            if (rating.HasValue)
            {
                return rating.Value;
            }

            if (flows != null)
            {
                var collected = new List<float>();
                foreach (var candidate in flows)
                {
                    if (candidate?.FeatureValues == null)
                    {
                        continue;
                    }

                    var candidateRating = ResolveFeatureValue(candidate.FeatureValues, "OfficialRating");
                    if (candidateRating.HasValue)
                    {
                        collected.Add(candidateRating.Value);
                    }
                }

                if (collected.Count > 0)
                {
                    return collected.Average();
                }
            }

            var classValue = ResolveFeatureValue(featureVector, "Class");
            if (!classValue.HasValue && flow?.FeatureValues != null)
            {
                classValue = ResolveFeatureValue(flow.FeatureValues, "Class");
            }

            var classBaseline = ResolveClassRatingBaseline(classValue);
            if (classBaseline.HasValue)
            {
                return classBaseline.Value;
            }

            return null;
        }

        private IReadOnlyList<float> ResolveHistoricalRatings(
            Dictionary<string, object?> featureVector,
            RunnerFlow? flow,
            float? ratingFallback)
        {
            int? horseId = null;
            if (featureVector.TryGetValue("HorseId", out var horseIdObj))
            {
                horseId = TryConvertToInt32(horseIdObj);
            }

            if (!horseId.HasValue && flow?.FeatureValues != null &&
                flow.FeatureValues.TryGetValue("HorseId", out var flowHorseId))
            {
                horseId = TryConvertToInt32(flowHorseId);
            }

            horseId = NormalizeHorseIdentifier(horseId);

            string? horseName = flow?.HorseName;
            if (string.IsNullOrWhiteSpace(horseName) && featureVector.TryGetValue("HorseName", out var horseNameObj) &&
                horseNameObj is string horseNameStr)
            {
                horseName = horseNameStr;
            }

            if (string.IsNullOrWhiteSpace(horseName) && flow?.FeatureValues != null &&
                flow.FeatureValues.TryGetValue("HorseName", out var flowHorseNameObj) &&
                flowHorseNameObj is string flowHorseName)
            {
                horseName = flowHorseName;
            }

            var history = _repo.GetHistoricalRaceClassRatings(
                horseName,
                horseId,
                PerformanceWindowSizes.Max());

            if (history == null || history.Count == 0)
            {
                return Array.Empty<float>();
            }

            var resolved = new List<float>(history.Count);
            foreach (var entry in history)
            {
                var rating = entry.OfficialRating.HasValue ? (float?)entry.OfficialRating.Value : null;
                if (!rating.HasValue)
                {
                    rating = ResolveClassRatingBaseline(entry.Class);
                }

                if (!rating.HasValue)
                {
                    rating = ratingFallback;
                }

                if (rating.HasValue)
                {
                    resolved.Add(rating.Value);
                }
            }

            return resolved;
        }

        private static float? ResolveClassRatingBaseline(float? classValue)
        {
            if (!classValue.HasValue)
            {
                return null;
            }

            var rounded = (int)Math.Round(classValue.Value);
            return ResolveClassRatingBaseline(rounded);
        }

        private static float? ResolveClassRatingBaseline(byte? classValue)
            => ResolveClassRatingBaseline(classValue.HasValue ? (int?)classValue.Value : null);

        private static float? ResolveClassRatingBaseline(int? classValue)
        {
            if (!classValue.HasValue)
            {
                return null;
            }

            var classInt = classValue.Value;
            if (classInt <= 0)
            {
                return null;
            }

            if (ClassRatingBaselines.TryGetValue(classInt, out var baseline))
            {
                return baseline;
            }

            classInt = Math.Min(classInt, 12);
            return 110f - (classInt * 5f);
        }


        private static float? CombineAverages(float? first, float? second)
        {
            if (first.HasValue && second.HasValue)
            {
                return (first.Value + second.Value) / 2f;
            }

            if (first.HasValue)
            {
                return first.Value;
            }

            if (second.HasValue)
            {
                return second.Value;
            }

            return null;
        }

        private static void FillIfMissing(Dictionary<string, object?> featureVector, string key, float? value)
        {
            if (featureVector == null || string.IsNullOrWhiteSpace(key))
            {
                return;
            }

            if (TryGetMeaningfulValue(featureVector, key, out _))
            {
                return;
            }

            if (value.HasValue)
            {
                featureVector[key] = value.Value;
                return;
            }

            if (!NeutralFeatureFallbacks.TryGetValue(key, out var fallback) || fallback == null)
            {
                return;
            }

            var fallbackValue = TryConvertToSingle(fallback);
            if (!fallbackValue.HasValue)
            {
                return;
            }

            featureVector[key] = fallbackValue.Value;
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
        private static DateTime? ExtractDateFromTitle(string? normalizedTitle)
        {
            if (string.IsNullOrWhiteSpace(normalizedTitle))
            {
                return null;
            }

            static IEnumerable<string> EnumerateCandidates(string source)
            {
                foreach (Match match in MarketTitleDateCandidateRegex.Matches(source))
                {
                    if (match.Success)
                    {
                        yield return match.Value;
                    }
                }
            }

            static DateTime? TryParseCandidate(string candidate)
            {
                if (string.IsNullOrWhiteSpace(candidate))
                {
                    return null;
                }

                var sanitized = Regex.Replace(candidate, @"(\d)(st|nd|rd|th)", "$1", RegexOptions.IgnoreCase);

                // Try a range of formats that may appear in market titles.
                var formats = new[]
                {
                    "ddd d MMM yyyy",
                    "ddd d MMMM yyyy",
                    "dddd d MMM yyyy",
                    "dddd d MMMM yyyy",
                    "d MMM yyyy",
                    "d MMMM yyyy",
                    "ddd d MMM",
                    "ddd d MMMM",
                    "dddd d MMM",
                    "dddd d MMMM",
                    "d MMM",
                    "d MMMM"
                };

                foreach (var format in formats)
                {
                    if (DateTime.TryParseExact(
                        sanitized,
                        format,
                        CultureInfo.InvariantCulture,
                        DateTimeStyles.AllowWhiteSpaces | DateTimeStyles.AssumeLocal,
                        out var parsed))
                    {
                        if (format.Contains("yyyy", StringComparison.Ordinal))
                        {
                            return parsed.Date;
                        }

                        var today = DateTime.Today;
                        try
                        {
                            var resolved = new DateTime(today.Year, parsed.Month, parsed.Day);
                            if (resolved < today.AddDays(-30))
                            {
                                resolved = resolved.AddYears(1);
                            }

                            return resolved.Date;
                        }
                        catch
                        {
                            // Ignore invalid day/month combinations.
                        }
                    }
                }

                return null;
            }

            foreach (var candidate in EnumerateCandidates(normalizedTitle))
            {
                var parsed = TryParseCandidate(candidate);
                if (parsed.HasValue)
                {
                    return parsed.Value;
                }
            }

            return null;
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

            bool HasMissingValue(string key)
            {
                if (!featureVector.TryGetValue(key, out var existing) || existing == null)
                {
                    return true;
                }

                if (existing is string existingText && string.IsNullOrWhiteSpace(existingText))
                {
                    return true;
                }

                return IsNeutralFallbackValue(featureVector, key, existing);
            }

            void SetIfMissing(string key, object? value)
            {
                if (value == null)
                {
                    return;
                }

                if (value is string stringValue && string.IsNullOrWhiteSpace(stringValue))
                {
                    return;
                }

                if (HasMissingValue(key))
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
            bool hasBackBook = TryGetMeaningfulValue(featureVector, "BackBookPercentage", out _);
            featureVector["BackBookPercentageMissing"] = !hasBackBook;
            if (!hasBackBook)
            {
                featureVector.Remove("BackBookPercentage");
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
                featureVector["WeightLbs"] = flow.WeightLbs.Value;

                if (!string.IsNullOrWhiteSpace(flow.WeightText))
                {
                    var trimmedWeight = flow.WeightText.Trim();
                    SetIfMissing("WeightText", trimmedWeight);
                    featureVector["WeightText"] = trimmedWeight;
                }

                SetIfMissing("WeightMissing", false);
                featureVector["WeightMissing"] = false;
                if (flows != null && flows.Count > 0)
                {
                    var comparableWeights = flows
                        .Where(f => f?.WeightLbs.HasValue == true)
                        .Select(f => f!.WeightLbs!.Value)
                        .ToArray();

                    if (comparableWeights.Length > 0)
                    {
                        var maxWeight = comparableWeights.Max();
                        var minWeight = comparableWeights.Min();

                        SetIfMissing("IsTopWeight", flow.WeightLbs.Value >= maxWeight);
                        SetIfMissing("IsBottomWeight", flow.WeightLbs.Value <= minWeight);
                    }
                }
            }
            else
            {
                if (!string.IsNullOrWhiteSpace(flow?.WeightText))
                {
                    var trimmedWeight = flow!.WeightText!.Trim();
                    SetIfMissing("WeightText", trimmedWeight);
                    featureVector["WeightText"] = trimmedWeight;
                }
                else if (featureVector.ContainsKey("WeightText"))
                {
                    featureVector.Remove("WeightText");
                }

                SetIfMissing("WeightMissing", true);
                featureVector["WeightMissing"] = true;
                if (featureVector.ContainsKey("WeightLbs"))
                {
                    featureVector.Remove("WeightLbs");
                }

                MarkMissing("runner weight");
            }
            if (flow?.WeightLbs.HasValue != true && flows != null && flows.Count > 0)
            {
                if (featureVector.TryGetValue("WeightLbs", out var runnerWeightObj) &&
                    TryGetInt(runnerWeightObj, out var runnerWeight) && runnerWeight > 0)
                {
                    var inferredWeights = flows
                        .Where(f => f?.WeightLbs.HasValue == true)
                        .Select(f => f!.WeightLbs!.Value)
                        .ToArray();

                    if (inferredWeights.Length > 0)
                    {
                        var maxWeight = inferredWeights.Max();
                        var minWeight = inferredWeights.Min();

                        SetIfMissing("IsTopWeight", runnerWeight >= maxWeight);
                        SetIfMissing("IsBottomWeight", runnerWeight <= minWeight);
                    }
                }
            }

            if (!string.IsNullOrWhiteSpace(flow?.TrainerName))
            {
                var trimmedTrainer = flow!.TrainerName!.Trim();
                SetIfMissing("TrainerName", trimmedTrainer);
                featureVector["TrainerName"] = trimmedTrainer;
            
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
                VenueCountry = venueCountry,
                Runners = flows == null
                    ? new List<RunnerDayReport>()
                    : flows
                        .Where(f => f != null && f.Age.HasValue && f.Age.Value > 0)
                        .Select(f =>
                            new RunnerDayReport
                            {
                                FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                                {
                                    ["Age"] = f!.Age!.Value
                                }
                            })
                        .ToList()
            };

            var parsed = ParseRaceMetadata(metadataSource, flows);

            if (!string.IsNullOrWhiteSpace(parsed.AgeRestriction))
            {
                SetIfMissing("AgeRestriction", parsed.AgeRestriction);
            }
            bool classFromHistory = false;
            try
            {
                int? horseId = null;
                if (featureVector.TryGetValue("HorseId", out var horseIdObj))
                {
                    horseId = TryConvertToInt32(horseIdObj);
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
                    var resolvedClass = _repo.GetMostRecentRaceClass(horseName, horseId);
                    if (resolvedClass.HasValue)
                    {
                        featureVector["Class"] = resolvedClass.Value;
                        classFromHistory = true;
                    }
                }
            }
            catch (Exception ex)
            {
                var identifier = DescribeRunner(flow);
                Console.Error.WriteLine($"\t\tFailed to resolve historical class for {identifier}: {ex.Message}");
            }

            if (!classFromHistory && parsed.Class.HasValue)
            {
                SetIfMissing("Class", parsed.Class);
            }
            bool hasClass = TryGetMeaningfulValue(featureVector, "Class", out _);
            featureVector["ClassMissing"] = !hasClass;
            if (!hasClass)
            {
                featureVector.Remove("Class");
                MarkMissing("race class");
            }
            var resolvedRaceType = string.IsNullOrWhiteSpace(raceType)
                ? parsed.RaceType
                : raceType.Trim();
            if (!string.IsNullOrWhiteSpace(resolvedRaceType))
            {
                SetIfMissing("RaceType", resolvedRaceType);
            }
            else if (!TryGetMeaningfulValue(featureVector, "RaceType", out _))
            {
                featureVector.Remove("RaceType");
                MarkMissing("race type");
            }

            var resolvedGoing = string.IsNullOrWhiteSpace(going)
                ? parsed.Going
                : going.Trim();
            bool hasGoing = TryGetMeaningfulValue(featureVector, "Going", out _);
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
                    featureVector.Remove("Going");
                    featureVector["Going"] = normalizedGoing;
                }
                hasGoing = true;
            }
            featureVector["GoingMissing"] = !hasGoing;
            if (!hasGoing)
            {
                featureVector.Remove("Going");
            }
            bool hasSurface = TryGetMeaningfulValue(featureVector, "Surface", out _);
            featureVector["SurfaceMissing"] = !hasSurface;
            if (!hasSurface)
            {
                featureVector.Remove("Surface");
                MarkMissing("surface type");
            }

            if (!string.IsNullOrWhiteSpace(parsed.Surface))
            {
                SetIfMissing("Surface", parsed.Surface);
            }
            else
            {
                MarkMissing("surface type");
            }

            if (!string.IsNullOrWhiteSpace(parsed.DistanceText))
            {
                SetIfMissing("DistanceText", parsed.DistanceText);
            }
            bool hasDistanceText = TryGetMeaningfulValue(featureVector, "DistanceText", out _);
            featureVector["DistanceTextMissing"] = !hasDistanceText;
            if (!hasDistanceText)
            {
                featureVector.Remove("DistanceText");
                MarkMissing("distance description");
            }

            int distanceYardsValue = 0;
            bool hasDistance = false;
            if (TryGetMeaningfulValue(featureVector, "DistanceYards", out var distanceObj) &&
                TryGetInt(distanceObj, out var resolvedDistance) && resolvedDistance > 0)
            {
                distanceYardsValue = resolvedDistance;
                hasDistance = true;
            }
            else if (parsed.DistanceYards > 0)
            {
                distanceYardsValue = parsed.DistanceYards;
                featureVector["DistanceYards"] = distanceYardsValue;
                hasDistance = true;
            }

            featureVector["DistanceMissing"] = !hasDistance;
            if (!hasDistance)
            {
                featureVector.Remove("DistanceYards");
                MarkMissing("race distance");
            }

            string? distanceBucket = null;
            if (hasDistance)
            {
                distanceBucket = DistanceBucketFromYards(distanceYardsValue);
                SetIfMissing("DistanceBucket", distanceBucket);
            }
            else
            {
                featureVector.Remove("DistanceBucket");
            }

            featureVector["RaceMetadataMissing"] = !hasClass || !hasGoing || !hasSurface || !hasDistance || !hasDistanceText;
            bool needsDistanceChange = HasMissingValue("DistanceChangeFromLast");
            bool needsDistanceRatio = HasMissingValue("DistanceRatioFromAverage");
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
            PromoteCalculatedHistoricalFallbacks(featureVector, flow, flows);
        }
        private static ParsedRaceMetadata ParseRaceMetadata(
            RaceDayReport race,
            IEnumerable<RunnerFlow>? flows = null)
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

            var raceType = TryDetectRaceType(tokens, race.RaceTitle);
            var surface = DetermineSurface(going, tokens);

            var ageRestriction = ResolveAgeRestriction(race, tokens, flows);

            return new ParsedRaceMetadata(
                string.IsNullOrWhiteSpace(distanceToken) ? null : distanceToken.Trim(),
                distanceYards,
                classValue,
                string.IsNullOrWhiteSpace(going) ? null : going.Trim(),
                surface,
                raceType,
                ageRestriction);
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
        private const string RunnerDetailsExpansionFunctionBody = @"
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
        private const string RunnerDetailsExpansionScript = "const row = arguments[0]; const index = arguments[1];" + RunnerDetailsExpansionFunctionBody;

        private const string BulkRunnerDetailsExpansionScript = @"
            const rows = Array.from(arguments[0] || []);
            const expandBody = arguments[1];
            const expandSingle = new Function('row', 'index', expandBody);
            const detailSelectors = [
                '.runner-timeform-details',
                '.runner-timeform',
                '.timeform-expandable',
                '.runner-expanded-details',
                '.runner-info-expanded'
            ];

            const hasVisibleDetails = element => {
                if (!element) { return false; }
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

            const results = [];
            for (let i = 0; i < rows.length; i++) {
                const row = rows[i];
                if (!row) {
                    results.push({ expanded: false, needsFallback: false });
                    continue;
                }

                let expanded = false;
                try {
                    expanded = !!expandSingle(row, i + 1);
                } catch (error) {
                    expanded = false;
                }

                if (expanded) {
                    results.push({ expanded: true, needsFallback: false });
                    continue;
                }

                const hasDetails = hasVisibleDetails(row);
                results.push({ expanded: hasDetails, needsFallback: !hasDetails });
            }

            return results;
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

            var rowsArray = runnerRows.OfType<IWebElement>().ToArray();
            if (rowsArray.Length == 0)
            {
                return;
            }

            IReadOnlyList<object>? bulkResults = null;
            try
            {
                var raw = js.ExecuteScript(BulkRunnerDetailsExpansionScript, rowsArray, RunnerDetailsExpansionFunctionBody);
                if (raw is IReadOnlyList<object> list)
                {
                    bulkResults = list;
                }
                else if (raw is object[] array)
                {
                    bulkResults = array;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tFailed to expand runner details in bulk: {ex.Message}");
            }

            if (bulkResults == null || bulkResults.Count != rowsArray.Length)
            {
                LegacyExpandRunnerDetails(js, rowsArray);
                return;
            }

            for (var i = 0; i < rowsArray.Length; i++)
            {
                var row = rowsArray[i];
                if (row == null)
                {
                    continue;
                }

                var parsed = ParseBulkExpansionResult(bulkResults[i]);
                if (parsed == null)
                {
                    var expansionSucceeded = TryExpandRunnerDetailsWithScript(js, row, i);
                    if (!expansionSucceeded)
                    {
                        TryFallbackRunnerDetailsClick(js, row);
                    }

                    continue;
                }

                var (expanded, needsFallback) = parsed.Value;
                if (!expanded && needsFallback)
                {
                    TryFallbackRunnerDetailsClick(js, row);
                }
            }
        }

        private static void LegacyExpandRunnerDetails(IJavaScriptExecutor js, IReadOnlyList<IWebElement> rows)
        {
            for (var i = 0; i < rows.Count; i++)
            {
                var row = rows[i];
                if (row == null)
                {
                    continue;
                }

                var expanded = TryExpandRunnerDetailsWithScript(js, row, i);
                if (!expanded)
                {
                    TryFallbackRunnerDetailsClick(js, row);
                }
            }
        }

        private static bool TryExpandRunnerDetailsWithScript(IJavaScriptExecutor js, IWebElement row, int index)
        {
            try
            {
                var result = js.ExecuteScript(RunnerDetailsExpansionScript, row, index + 1);
                return result is bool flag && flag;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\tFailed to expand runner details for row {index + 1}: {ex.Message}");
                return false;
            }
        }

        private static (bool Expanded, bool NeedsFallback)? ParseBulkExpansionResult(object? value)
        {
            if (value is IDictionary<string, object?> generic)
            {
                return (
                    generic.TryGetValue("expanded", out var expandedObj) && expandedObj is bool expanded && expanded,
                    generic.TryGetValue("needsFallback", out var fallbackObj) && fallbackObj is bool needsFallback && needsFallback);
            }

            if (value is IDictionary dictionary)
            {
                return (
                    TryReadBoolean(dictionary, "expanded"),
                    TryReadBoolean(dictionary, "needsFallback"));
            }

            return null;
        }

        private static bool TryReadBoolean(IDictionary dictionary, string key)
        {
            if (dictionary.Contains(key) && dictionary[key] is bool flag)
            {
                return flag;
            }

            return false;
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
                var labeledMatch = AgeLabelRegex.Match(normalized);
                if (labeledMatch.Success)
                {
                    age = TryParseByte(labeledMatch.Groups["age"].Value);
                }
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
            string? Going,
            string? Surface,
            string? RaceType,
            string? AgeRestriction);

        private static readonly Regex BracketedNameContentRegex =
            new Regex(@"\s*[\(\[][^\)\]]*[\)\]]\s*", RegexOptions.Compiled);
        private static string? ResolveAgeRestriction(
            RaceDayReport race,
            IReadOnlyCollection<string> tokens,
            IEnumerable<RunnerFlow>? flows)
        {
            if (race == null)
            {
                return null;
            }

            var sources = new List<string?>
            {
                race.RaceDetails,
                race.RaceTitle,
                race.RaceType,
                race.Going
            };

            if (tokens != null)
            {
                foreach (var token in tokens)
                {
                    if (!string.IsNullOrWhiteSpace(token))
                    {
                        sources.Add(token);
                    }
                }
            }

            foreach (var source in sources)
            {
                var restriction = ExtractAgeRestrictionFromText(source);
                if (!string.IsNullOrWhiteSpace(restriction))
                {
                    return restriction;
                }
            }

            var youngestAge = GetYoungestRunnerAge(race, flows);
            if (youngestAge.HasValue)
            {
                return string.Concat(youngestAge.Value.ToString(CultureInfo.InvariantCulture), "yo+");
            }

            return null;
        }

        private static int? GetYoungestRunnerAge(RaceDayReport race, IEnumerable<RunnerFlow>? flows)
        {
            int? youngest = null;

            if (flows != null)
            {
                foreach (var flow in flows)
                {
                    if (flow?.Age.HasValue == true && flow.Age.Value > 0)
                    {
                        youngest = youngest.HasValue
                            ? Math.Min(youngest.Value, flow.Age.Value)
                            : flow.Age.Value;
                    }
                }
            }

            if (race?.Runners != null)
            {
                foreach (var runner in race.Runners)
                {
                    if (runner?.FeatureValues == null)
                    {
                        continue;
                    }

                    if (!runner.FeatureValues.TryGetValue("Age", out var ageObj))
                    {
                        continue;
                    }

                    if (TryGetInt(ageObj, out var parsedAge) && parsedAge > 0)
                    {
                        youngest = youngest.HasValue
                            ? Math.Min(youngest.Value, parsedAge)
                            : parsedAge;
                    }
                }
            }

            return youngest;
        }

        private static string? ExtractAgeRestrictionFromText(string? source)
        {
            if (string.IsNullOrWhiteSpace(source))
            {
                return null;
            }

            var normalized = NormalizeAgeRestrictionSource(source);
            if (normalized.Length == 0)
            {
                return null;
            }

            var rangeMatch = AgeRestrictionRangeRegex.Match(normalized);
            if (rangeMatch.Success)
            {
                var min = rangeMatch.Groups["min"].Value;
                var max = rangeMatch.Groups["max"].Value;
                if (!string.IsNullOrEmpty(min) && !string.IsNullOrEmpty(max))
                {
                    return string.Concat(min, "-", max, "yo");
                }
            }

            var wordMatch = AgeRestrictionWordRegex.Match(normalized);
            if (wordMatch.Success)
            {
                var age = wordMatch.Groups["age"].Value;
                var suffix = normalized[(wordMatch.Index + wordMatch.Length)..];
                var qualifier = ExtractAgeQualifier(suffix);
                return BuildAgeRestriction(age, qualifier);
            }

            var labelMatch = AgeRestrictionLabelRegex.Match(normalized);
            if (labelMatch.Success)
            {
                var age = labelMatch.Groups["age"].Value;
                var suffix = normalized[(labelMatch.Index + labelMatch.Length)..];
                var qualifier = ExtractAgeQualifier(suffix);
                return BuildAgeRestriction(age, qualifier);
            }

            return null;
        }
        private static string NormalizeAgeRestrictionSource(string source)
        {
            var normalized = Regex.Replace(source.Replace('\u00A0', ' '), @"\s+", " ").Trim(); // use @ to make \s work
            if (normalized.Length == 0) return string.Empty;
            normalized = Regex.Replace(normalized, @"(?i)y\s*[\-\./]\s*o", "yo"); // use @ here too
            return normalized;
        }//
        private static AgeRestrictionQualifier ExtractAgeQualifier(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return AgeRestrictionQualifier.None;
            }

            var trimmed = text.TrimStart(' ', '-', '–', '—', ',', ';', '/', '\\');
            if (trimmed.Length == 0)
            {
                return AgeRestrictionQualifier.None;
            }

            var match = AgeRestrictionQualifierRegex.Match(trimmed);
            if (!match.Success)
            {
                return AgeRestrictionQualifier.None;
            }

            var qualifier = match.Groups["qualifier"].Value;
            return qualifier.Equals("only", StringComparison.OrdinalIgnoreCase)
                ? AgeRestrictionQualifier.Only
                : AgeRestrictionQualifier.Plus;
        }

        private static string? BuildAgeRestriction(string age, AgeRestrictionQualifier qualifier)
        {
            if (string.IsNullOrWhiteSpace(age))
            {
                return null;
            }

            return qualifier switch
            {
                AgeRestrictionQualifier.Plus => string.Concat(age, "yo+"),
                _ => string.Concat(age, "yo")
            };
        }

        private enum AgeRestrictionQualifier
        {
            None,
            Plus,
            Only
        }

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
            static void SyncClampTargetWithOdds(RunnerFlow flow)
            {
                if (!flow.AiProbabilityClampedToMarket)
                {
                    return;
                }

                if (!flow.AiProbabilityClampTarget.HasValue)
                {
                    return;
                }

                if (!flow.AiOdds.HasValue || !double.IsFinite(flow.AiOdds.Value))
                {
                    return;
                }

                flow.AiProbabilityClampTarget = Math.Max(flow.AiOdds.Value, 0d);
            }
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
            bool treatAsDegenerate = false;
            string degeneracyMessage = string.Empty;

            if (maxValue - minValue <= degeneracyTolerance)
            {
                degeneracyMessage = "\t\tDetected degenerate AI probability distribution; raw outputs are identical across runners.";
                treatAsDegenerate = true;
            }
            else if (IsSaturatedProbabilityDistribution(valid, out var saturationDetail))
            {
                degeneracyMessage = $"\t\tDetected saturated AI probability distribution; {saturationDetail}.";
                treatAsDegenerate = true;
            }

            if (treatAsDegenerate)
            {
                Console.WriteLine(degeneracyMessage);

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
                if (ApplyPostNormalizationLowProbabilityClamp(flows))
                {
                    NormalizeAiOdds(flows, useMarketFallbackForDegeneracy);
                }
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
            if (ApplyPostNormalizationLowProbabilityClamp(flows))
            {
                NormalizeAiOdds(flows, useMarketFallbackForDegeneracy);
            }
        }
        private static bool ApplyPostNormalizationLowProbabilityClamp(ICollection<RunnerFlow>? flows)
        {
            if (flows is null || flows.Count == 0)
            {
                return false;
            }

            var clampApplied = false;

            foreach (var flow in flows)
            {
                if (flow == null)
                {
                    continue;
                }

                if (flow.AiProbabilityClampedToMarket)
                {
                    continue;
                }

                if (!flow.AiOdds.HasValue || !double.IsFinite(flow.AiOdds.Value) || flow.AiOdds.Value <= 0d)
                {
                    continue;
                }

                if (flow.AiOdds.Value >= LowAiProbabilityClampThreshold)
                {
                    continue;
                }

                if (TryClampLowAiProbabilityToMarket(flow))
                {
                    clampApplied = true;
                }
            }

            if (clampApplied)
            {
                Console.WriteLine("\t\tClamped normalized AI probabilities below threshold to market-implied values; re-normalizing distribution.");
            }

            return clampApplied;
        }
        private static void ApplyMarketFallbackForUnmatchedRunners(IReadOnlyList<RunnerFlow> flows)
        {
            if (flows == null || flows.Count == 0) return; // return early if list is null or empty
            bool fallbackApplied = false;
            foreach (var flow in flows)
            {
                if (flow == null || flow.MatchedDatabaseRecord)
                {
                    continue; // skip matched or null entries
                }

                var hasValidAiProbability = flow.AiOdds.HasValue
                    && double.IsFinite(flow.AiOdds.Value)
                    && flow.AiOdds.Value > 0d;

                if (hasValidAiProbability)
                {
                    continue; // retain existing AI odds when available
                }

                var identifier = !string.IsNullOrWhiteSpace(flow.HorseName) ? flow.HorseName! : "unknown"; // determine identifier

                if (flow.BackPrice1.HasValue && flow.BackPrice1.Value > 1m)
                {
                    var marketProbability = 1.0 / (double)flow.BackPrice1.Value; // calculate implied probability
                    flow.AiOdds = marketProbability; flow.AiProbabilityMarketDerived = true;
                    flow.AiProbabilityClampedToMarket = false;
                    flow.AiProbabilityClampTarget = null;
                    AppendMarketFallbackReason(flow, "Database record not found; using market-implied probability");
                    Console.WriteLine($"\t\tNo database match for {identifier}; using market-implied probability {marketProbability.ToString("0.####", CultureInfo.InvariantCulture)} as AI odds.");
                    fallbackApplied = true;
                }
                else
                {
                    flow.AiOdds = null; flow.AiProbabilityMarketDerived = false;
                    flow.AiProbabilityFallbackReason = null;
                    Console.WriteLine($"\t\tNo database match for {identifier} and no usable back price; AI odds remain unavailable.");
                }
                if (fallbackApplied)
                {
                    RenormalizeAiProbabilities(flows);
                }
            }
        }
        private static bool TryClampLowAiProbabilityToMarket(RunnerFlow? flow)
        {
            if (flow == null)
            {
                return false;
            }

            if (!flow.BackPrice1.HasValue || flow.BackPrice1.Value <= 1m)
            {
                return false;
            }

            var marketProbability = 1.0 / (double)flow.BackPrice1.Value;

            flow.AiOdds = marketProbability;
            flow.AiProbabilityClampedToMarket = true;
            flow.AiProbabilityClampTarget = marketProbability;
            flow.AiProbabilityMarketDerived = true;
            AppendMarketFallbackReason(flow, "Model probability below clamp threshold; using market-implied probability");

            return true;
        }

        private static void RenormalizeAiProbabilities(IEnumerable<RunnerFlow>? flows)
        {
            if (flows == null)
            {
                return;
            }

            var valid = flows
                .Where(f => f != null && f.AiOdds.HasValue && double.IsFinite(f.AiOdds.Value) && f.AiOdds.Value > 0d)
                .ToList();

            if (valid.Count == 0)
            {
                return;
            }

            var sum = valid.Sum(f => f.AiOdds!.Value);
            if (!double.IsFinite(sum) || sum <= double.Epsilon)
            {
                return;
            }

            const double tolerance = 1e-8;
            if (Math.Abs(sum - 1d) <= tolerance)
            {
                return;
            }

            var scale = 1d / sum;

            foreach (var flow in valid)
            {
                flow.AiOdds = Math.Max(flow.AiOdds!.Value * scale, 0d);

                if (flow.AiProbabilityClampTarget.HasValue && double.IsFinite(flow.AiProbabilityClampTarget.Value))
                {
                    flow.AiProbabilityClampTarget = Math.Max(flow.AiProbabilityClampTarget.Value * scale, 0d);
                }
            }

            var normalizedSum = valid.Sum(f => f.AiOdds!.Value);
            Console.WriteLine(
                $"\t\tApplied post-fallback normalization to AI probabilities; " +
                $"scale={scale.ToString("0.####", CultureInfo.InvariantCulture)}, " +
                $"normalized sum={normalizedSum.ToString("0.####", CultureInfo.InvariantCulture)}.");
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
        private static bool IsSaturatedProbabilityDistribution(IReadOnlyCollection<RunnerFlow> flows, out string detail)
        {
            detail = string.Empty;

            if (flows is null || flows.Count <= 2)
            {
                return false;
            }

            const double nearCertaintyThreshold = 1d - 1e-6;
            const double nearZeroThreshold = 1e-6;

            int nearCertaintyCount = 0;
            int nearZeroCount = 0;

            foreach (var flow in flows)
            {
                if (!flow.AiOdds.HasValue || !double.IsFinite(flow.AiOdds.Value))
                {
                    continue;
                }

                var value = flow.AiOdds.Value;
                if (value >= nearCertaintyThreshold)
                {
                    nearCertaintyCount++;
                }
                else if (value <= nearZeroThreshold)
                {
                    nearZeroCount++;
                }
            }

            int majorityThreshold = Math.Max(3, (flows.Count + 1) / 2);

            if (nearCertaintyCount >= majorityThreshold)
            {
                detail = $"{nearCertaintyCount}/{flows.Count} runner(s) exceed {nearCertaintyThreshold.ToString("0.######", CultureInfo.InvariantCulture)} raw probability";
                return true;
            }

            int nearZeroMajority = Math.Max(3, flows.Count - 1);
            if (nearZeroCount >= nearZeroMajority)
            {
                detail = $"{nearZeroCount}/{flows.Count} runner(s) fall below {nearZeroThreshold.ToString("0.######", CultureInfo.InvariantCulture)} raw probability";
                return true;
            }

            return false;
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
            var uniformProbability = flows.Count > 0 ? 1.0 / flows.Count : 0d;

            double minimumResidualMass = 0d;
            if (others.Count > 0 && uniformProbability > 0d)
            {
                const double residualUniformFraction = 0.5d;
                minimumResidualMass = uniformProbability * residualUniformFraction * others.Count;
                minimumResidualMass = Math.Clamp(minimumResidualMass, 0d, 1d - probabilityFloor);

                var maximumWinnerProbability = 1d - minimumResidualMass;
                if (winnerProbability > maximumWinnerProbability)
                {
                    var originalWinnerProbability = winnerProbability;
                    winnerProbability = Math.Max(maximumWinnerProbability, probabilityFloor);
                    Console.WriteLine(
                        $"\t\tWinner probability {originalWinnerProbability.ToString("0.####", CultureInfo.InvariantCulture)} " +
                        $"exceeded degeneracy guard; clamped to {winnerProbability.ToString("0.####", CultureInfo.InvariantCulture)} " +
                        $"to reserve {minimumResidualMass.ToString("0.####", CultureInfo.InvariantCulture)} probability mass for rivals."
                    );
                }
            }
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
                : "unknown";

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
        private static bool AreNamesEquivalent(string? left, string? right)
        {
            if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
            {
                return false;
            }

            var normalizedLeft = NormalizeName(left);
            var normalizedRight = NormalizeName(right);

            if (string.IsNullOrEmpty(normalizedLeft) || string.IsNullOrEmpty(normalizedRight))
            {
                return false;
            }

            return string.Equals(normalizedLeft, normalizedRight, StringComparison.Ordinal);
        }

        private readonly record struct RunnerMatchScore(int Score, int MatchedFields)
        {
            public bool IsValid => Score > 0 && MatchedFields > 0;
        }

        private static RunnerMatchScore CalculateRunnerMatchScore(
            RunnerFlow flow,
            string? horseName,
            byte? clothNumber,
            byte? draw,
            string? jockeyName,
            string? trainerName)
        {
            if (flow == null)
            {
                return default;
            }

            var score = 0;
            var matchedFields = 0;

            if (!string.IsNullOrWhiteSpace(horseName) && !string.IsNullOrWhiteSpace(flow.HorseName))
            {
                if (AreNamesEquivalent(flow.HorseName, horseName))
                {
                    score += 100;
                    matchedFields++;
                }
                else
                {
                    return default;
                }
            }

            if (clothNumber.HasValue && flow.ClothNumber.HasValue)
            {
                if (clothNumber.Value == flow.ClothNumber.Value)
                {
                    score += 25;
                    matchedFields++;
                }
                else
                {
                    return default;
                }
            }

            if (draw.HasValue && flow.Draw.HasValue)
            {
                if (draw.Value == flow.Draw.Value)
                {
                    score += 20;
                    matchedFields++;
                }
                else
                {
                    return default;
                }
            }

            if (!string.IsNullOrWhiteSpace(jockeyName) && !string.IsNullOrWhiteSpace(flow.JockeyName))
            {
                if (AreNamesEquivalent(flow.JockeyName, jockeyName))
                {
                    score += 10;
                    matchedFields++;
                }
                else
                {
                    return default;
                }
            }

            var normalizedTrainer = NormalizeTrainerName(trainerName);
            var normalizedFlowTrainer = NormalizeTrainerName(flow.TrainerName);

            if (!string.IsNullOrWhiteSpace(normalizedTrainer) && !string.IsNullOrWhiteSpace(normalizedFlowTrainer))
            {
                if (AreNamesEquivalent(normalizedFlowTrainer, normalizedTrainer))
                {
                    score += 10;
                    matchedFields++;
                }
                else
                {
                    return default;
                }
            }

            return new RunnerMatchScore(score, matchedFields);
        }

        private static RunnerFlow? FindBestRunnerMatch(
            IEnumerable<RunnerFlow> candidates,
            string? horseName,
            byte? clothNumber,
            byte? draw,
            string? jockeyName,
            string? trainerName)
        {
            if (candidates == null)
            {
                return null;
            }

            RunnerFlow? best = null;
            var bestScore = 0;
            var bestMatchedFields = 0;

            foreach (var candidate in candidates)
            {
                var score = CalculateRunnerMatchScore(candidate, horseName, clothNumber, draw, jockeyName, trainerName);
                if (!score.IsValid)
                {
                    continue;
                }

                if (score.Score > bestScore || (score.Score == bestScore && score.MatchedFields > bestMatchedFields))
                {
                    best = candidate;
                    bestScore = score.Score;
                    bestMatchedFields = score.MatchedFields;
                }
            }

            return best;
        }

        private static IEnumerable<string> BuildRunnerMatchKeys(
            string? horseName,
            byte? clothNumber,
            byte? draw,
            string? jockeyName,
            string? trainerName)
        {
            var results = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            void Add(string? key)
            {
                if (string.IsNullOrWhiteSpace(key))
                {
                    return;
                }

                if (seen.Add(key))
                {
                    results.Add(key);
                }
            }

            Add(BuildRunnerMatchKeyInternal(horseName, clothNumber, draw, jockeyName, trainerName));
            Add(BuildRunnerMatchKeyInternal(horseName, clothNumber, draw, jockeyName, null));
            Add(BuildRunnerMatchKeyInternal(horseName, clothNumber, draw, null, trainerName));
            Add(BuildRunnerMatchKeyInternal(horseName, clothNumber, draw, null, null));
            Add(BuildRunnerMatchKeyInternal(horseName, clothNumber, null, null, null));
            Add(BuildRunnerMatchKeyInternal(horseName, null, null, null, null));
            Add(BuildRunnerMatchKeyInternal(null, clothNumber, draw, null, null));
            Add(BuildRunnerMatchKeyInternal(null, clothNumber, null, null, null));
            Add(BuildRunnerMatchKeyInternal(null, null, draw, null, null));
            Add(BuildRunnerMatchKeyInternal(null, null, null, jockeyName, trainerName));
            Add(BuildRunnerMatchKeyInternal(null, null, null, jockeyName, null));
            Add(BuildRunnerMatchKeyInternal(null, null, null, null, trainerName));

            return results;
        }

        private static string? GetPrimaryRunnerMatchKey(
            string? horseName,
            byte? clothNumber,
            byte? draw,
            string? jockeyName,
            string? trainerName) =>
            BuildRunnerMatchKeys(horseName, clothNumber, draw, jockeyName, trainerName).FirstOrDefault();

        private static string BuildRunnerMatchKeyInternal(
            string? horseName,
            byte? clothNumber,
            byte? draw,
            string? jockeyName,
            string? trainerName)
        {
            var parts = new List<string>();

            if (!string.IsNullOrWhiteSpace(horseName))
            {
                parts.Add($"horse:{horseName.Trim()}");
            }

            if (clothNumber.HasValue)
            {
                parts.Add($"cloth:{clothNumber.Value.ToString(CultureInfo.InvariantCulture)}");
            }

            if (draw.HasValue)
            {
                parts.Add($"draw:{draw.Value.ToString(CultureInfo.InvariantCulture)}");
            }

            if (!string.IsNullOrWhiteSpace(jockeyName))
            {
                parts.Add($"jockey:{jockeyName.Trim()}");
            }

            if (!string.IsNullOrWhiteSpace(trainerName))
            {
                parts.Add($"trainer:{trainerName.Trim()}");
            }

            return parts.Count > 0 ? string.Join("|", parts) : string.Empty;
        }
        private static string DescribeRunner(RunnerFlow? flow)
        {
            if (flow == null)
            {
                return "unknown";
            }

            var parts = new List<string>();

            if (!string.IsNullOrWhiteSpace(flow.HorseName))
            {
                parts.Add(flow.HorseName!.Trim());
            }

            if (flow.ClothNumber.HasValue)
            {
                parts.Add($"cloth #{flow.ClothNumber.Value.ToString(CultureInfo.InvariantCulture)}");
            }

            if (flow.Draw.HasValue)
            {
                parts.Add($"draw #{flow.Draw.Value.ToString(CultureInfo.InvariantCulture)}");
            }

            if (!string.IsNullOrWhiteSpace(flow.JockeyName))
            {
                parts.Add($"jockey {flow.JockeyName!.Trim()}");
            }

            if (!string.IsNullOrWhiteSpace(flow.TrainerName))
            {
                parts.Add($"trainer {flow.TrainerName!.Trim()}");
            }

            return parts.Count > 0 ? string.Join(", ", parts) : "unknown";
        }
        internal static string? ExtractMarketId(string url)
        {
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