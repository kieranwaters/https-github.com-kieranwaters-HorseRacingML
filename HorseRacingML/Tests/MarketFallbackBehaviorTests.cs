using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.ML;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using Xunit;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.Tests
{
    public class MarketFallbackBehaviorTests
    {
        [Fact]
        public void ApplyMarketFallback_NormalizesProbabilities()
        {
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Matched Runner",
                    MatchedDatabaseRecord = true,
                    AiOdds = 0.25d,
                    BackPrice1 = 4m
                },
                new RunnerFlow
                {
                    HorseName = "Synthetic Features Runner",
                    MatchedDatabaseRecord = false,
                    AiOdds = 0.35d,
                    BackPrice1 = 3.5m
                },
                new RunnerFlow
                {
                    HorseName = "Fallback Runner",
                    MatchedDatabaseRecord = false,
                    AiOdds = null,
                    BackPrice1 = 5m
                }
            };

            InvokeApplyMarketFallback(flows);

            var matched = flows[0];
            Assert.False(matched.AiProbabilityMarketDerived);
            Assert.True(matched.AiOdds.HasValue);
            Assert.Null(matched.AiProbabilityFallbackReason);

            var synthetic = flows[1];
            Assert.False(synthetic.AiProbabilityMarketDerived);
            Assert.True(synthetic.AiOdds.HasValue);
            Assert.Null(synthetic.AiProbabilityFallbackReason);

            var fallback = flows[2];
            Assert.True(fallback.AiProbabilityMarketDerived);
            Assert.True(fallback.AiOdds.HasValue);
            var expectedMarketProbability = 1.0d / (double)fallback.BackPrice1!.Value;
            var rawSum = 0.25d + 0.35d + expectedMarketProbability;
            var expectedScale = 1.0d / rawSum;

            Assert.Equal(0.25d * expectedScale, matched.AiOdds!.Value, 12);
            Assert.Equal(0.35d * expectedScale, synthetic.AiOdds!.Value, 12);
            Assert.Equal(expectedMarketProbability * expectedScale, fallback.AiOdds!.Value, 12);
            Assert.Equal(1.0d, matched.AiOdds!.Value + synthetic.AiOdds!.Value + fallback.AiOdds!.Value, 12);
            Assert.False(string.IsNullOrWhiteSpace(fallback.AiProbabilityFallbackReason));
        }

        [Fact]
        public void BuildRaceReport_SummarizesMarketFallbackReason()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner Alpha",
                    AiOdds = 0.6d,
                    AiProbabilityMarketDerived = true,
                    AiProbabilityFallbackReason = "Degenerate model outputs; using market-implied probability",
                    BackPrice1 = 2m
                },
                new RunnerFlow
                {
                    HorseName = "Runner Beta",
                    AiOdds = 0.4d,
                    AiProbabilityMarketDerived = true,
                    AiProbabilityFallbackReason = "Degenerate model outputs; using market-implied probability",
                    BackPrice1 = 3.5m
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.111",
                raceTitle: "Fallback Test",
                venueName: "Test Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 6, 15),
                offTime: new TimeSpan(13, 45, 0),
                raceDetails: "Handicap",
                going: null,
                backBookPercentage: null,
                layBookPercentage: null,
                raceUrl: null,
                flows: flows);

            const string expected = "All AI win probabilities defaulted to market-implied odds for this race. Reason: Degenerate model outputs; using market-implied probability.";
            Assert.Equal(expected, report.RaceFallbackSummary);
        }
        [Theory]
        [InlineData("DaysSinceLastWin", 12)]
        [InlineData("RacesSinceLastWin", 3)]
        public void BuildRaceReport_PopulatesHasLastWinFlagWhenHistoryPresent(string historyKey, int historyValue)
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "History Runner",
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        [historyKey] = historyValue
                    }
                }
            };

            var race = scraper.TestBuildRaceReport(
                marketId: "1.999",
                raceTitle: "History Test",
                venueName: "Test Course",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 6, 1),
                offTime: new TimeSpan(14, 30, 0),
                raceDetails: "Test",
                going: "Good",
                backBookPercentage: null,
                layBookPercentage: null,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(race.Runners);
            Assert.True(runner.FeatureValues.TryGetValue("HasLastWin", out var flag));
            Assert.True(Convert.ToBoolean(flag));
        }
        [Fact]
        public void BuildRaceReport_DefaultsHasLastWinFlagWhenHistoryUnavailable()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "No History Runner",
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                }
            };

            var race = scraper.TestBuildRaceReport(
                marketId: "1.1000",
                raceTitle: "Default History Test",
                venueName: "Test Course",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 6, 1),
                offTime: new TimeSpan(14, 30, 0),
                raceDetails: "Test",
                going: "Good",
                backBookPercentage: null,
                layBookPercentage: null,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(race.Runners);
            Assert.True(runner.FeatureValues.TryGetValue("HasLastWin", out var flag));
            Assert.False(Convert.ToBoolean(flag));
        }
        [Fact]
        public void BuildRaceReport_ClonesEncodedFeatureValues()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var encoded = new List<EncodedFeatureValue>
            {
                new EncodedFeatureValue
                {
                    Index = 0,
                    FeatureKey = "Pace",
                    Label = "Pace=Forward",
                    Value = 0.75d,
                    Active = true
                },
                new EncodedFeatureValue
                {
                    Index = 1,
                    FeatureKey = "Pace",
                    Label = "Pace=HeldUp",
                    Value = 0.1d,
                    Active = false
                }
            };

            var flow = new RunnerFlow
            {
                HorseName = "Runner Gamma",
                EncodedFeatureValues = encoded,
                FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                {
                    ["BackPrice1"] = 3.6m
                },
                FeaturePopulationSummary = new FeaturePopulationSummary
                {
                    PopulatedKeys = new[] { "BackPrice1" },
                    MissingKeys = new[] { "HistoricalWins" }
                },
                AiTrainedModelApplied = true
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.222",
                raceTitle: "Encoded Clone Test",
                venueName: "Test Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 8, 10),
                offTime: new TimeSpan(14, 30, 0),
                raceDetails: "Handicap",
                going: "Good",
                backBookPercentage: 102.5m,
                layBookPercentage: 104.2m,
                raceUrl: "https://example.com/1.222",
                flows: new[] { flow });

            var runner = Assert.Single(report.Runners);
            Assert.NotSame(encoded, runner.EncodedFeatureValues);
            Assert.Equal(encoded.Count, runner.EncodedFeatureValues.Count);
            Assert.True(runner.EncodedFeatureValues.SequenceEqual(encoded, new EncodedFeatureValueComparer()));
        }

        private sealed class EncodedFeatureValueComparer : IEqualityComparer<EncodedFeatureValue>
        {
            public bool Equals(EncodedFeatureValue? x, EncodedFeatureValue? y)
            {
                if (x == y)
                {
                    return true;
                }

                if (x is null || y is null)
                {
                    return false;
                }

                return x.Index == y.Index &&
                       string.Equals(x.FeatureKey, y.FeatureKey, StringComparison.Ordinal) &&
                       string.Equals(x.Label, y.Label, StringComparison.Ordinal) &&
                       Math.Abs(x.Value - y.Value) < 1e-12 &&
                       x.Active == y.Active;
            }

            public int GetHashCode(EncodedFeatureValue obj)
            {
                return HashCode.Combine(obj.Index, obj.FeatureKey, obj.Label, obj.Value, obj.Active);
            }
        }
        [Fact]
        public void ApplyScrapedFeatureFallbacks_OverridesTrainerAndWeightWithLiveValues()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["TrainerName"] = "Historical Trainer",
                ["WeightLbs"] = 118,
                ["WeightText"] = "8-6",
                ["WeightMissing"] = false
            };

            var flow = new RunnerFlow
            {
                HorseName = "Runner Alpha",
                TrainerName = "Live Trainer",
                WeightLbs = 132,
                WeightText = "9-6"
            };

            scraper.TestApplyScrapedFeatureFallbacks(
                featureVector,
                flow,
                raceDate: new DateTime(2024, 6, 1),
                raceDetails: "Handicap",
                flows: new List<RunnerFlow> { flow });

            Assert.Equal("Live Trainer", Assert.IsType<string>(featureVector["TrainerName"]));
            Assert.Equal(132, Convert.ToInt32(featureVector["WeightLbs"]));
            Assert.Equal("9-6", Assert.IsType<string>(featureVector["WeightText"]));
            Assert.False(Convert.ToBoolean(featureVector["WeightMissing"]));
        }
        [Fact]
        public void ApplyScrapedFeatureFallbacks_PopulatesDerivedHistoricalFallbacks()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["TrainerWinRate"] = 0.2f,
                ["TrainerSurfaceAvgNorm"] = 0.4f,
                ["LastTrainerSurfaceNormPos"] = 0.45f,
                ["JockeyWinRate"] = 0.18f,
                ["JockeyGoingWinRate"] = 0.22f,
                ["JockeyDistanceBucketWinRate"] = 0.16f,
                ["JockeyGoingAvgNorm"] = 0.55f,
                ["JockeyDistanceBucketAvgNorm"] = 0.6f,
                ["LastJockeyGoingNormPos"] = 0.48f,
                ["LastJockeyDistanceBucketNormPos"] = 0.5f,
                ["TrainerSurfaceWinRate"] = 0.24f,
                ["JockeySurfaceWinRate"] = 0.2f,
                ["TrainerCourseWinRate"] = 0.26f,
                ["JockeyCourseWinRate"] = 0.18f,
                ["LifetimeWinRate"] = 0.3f,
                ["OfficialRating"] = 92f,
                ["Class"] = 4
            };

            var flow = new RunnerFlow
            {
                HorseName = "Runner Delta"
            };

            scraper.TestApplyScrapedFeatureFallbacks(
                featureVector,
                flow,
                raceDate: new DateTime(2024, 6, 1),
                raceDetails: "Handicap",
                flows: new List<RunnerFlow> { flow });

            Assert.Equal(92f, Convert.ToSingle(featureVector["AvgRatingLast5"]));
            Assert.Equal(0.3f, Convert.ToSingle(featureVector["ClassWinRate"]));
            Assert.Equal(0.7f, MathF.Round(Convert.ToSingle(featureVector["ClassAvgNorm"]), 1));
            Assert.Equal(0.2f, Convert.ToSingle(featureVector["TrainerClassWinRate"]));
            Assert.Equal(0.4f, Convert.ToSingle(featureVector["TrainerClassAvgNorm"]));
            Assert.Equal(0.19f, MathF.Round(Convert.ToSingle(featureVector["TrainerJockeyWinRate"]), 2));
            Assert.Equal(0.19f, MathF.Round(Convert.ToSingle(featureVector["JockeyGoingDistanceWinRate"]), 2));
            Assert.Equal(0.49f, MathF.Round(Convert.ToSingle(featureVector["LastJockeyGoingDistanceNormPos"]), 2));
            Assert.Equal(0.22f, MathF.Round(Convert.ToSingle(featureVector["TrainerJockeySurfaceWinRate"]), 2));
            Assert.Equal(0.22f, MathF.Round(Convert.ToSingle(featureVector["TrainerJockeyCourseWinRate"]), 2));
        }
        [Fact]
        public void PromoteCalculatedHistoricalFallbacks_UsesPerRaceClassBaselines()
        {
            var repo = new StubRepository
            {
                HistoricalRatings = new List<RaceClassRating>
                {
                    new RaceClassRating
                    {
                        RaceDate = new DateTime(2024, 6, 10),
                        OfficialRating = null,
                        Class = 3
                    },
                    new RaceClassRating
                    {
                        RaceDate = new DateTime(2024, 5, 1),
                        OfficialRating = 88,
                        Class = 4
                    },
                    new RaceClassRating
                    {
                        RaceDate = new DateTime(2024, 4, 1),
                        OfficialRating = null,
                        Class = 5
                    }
                }
            };

            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HorseId"] = 123,
                ["HorseName"] = "Runner Echo"
            };

            var flow = new RunnerFlow
            {
                HorseName = "Runner Echo",
                FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                {
                    ["HorseId"] = 123
                }
            };

            scraper.TestApplyScrapedFeatureFallbacks(
                featureVector,
                flow,
                raceDate: new DateTime(2024, 7, 1),
                flows: new List<RunnerFlow> { flow });

            Assert.True(featureVector.TryGetValue("AvgRatingLast1", out var last1Obj));
            Assert.True(featureVector.TryGetValue("AvgRatingLast3", out var last3Obj));
            Assert.True(featureVector.TryGetValue("AvgRatingLast5", out var last5Obj));

            Assert.Equal(95f, Convert.ToSingle(last1Obj));
            Assert.Equal(89.33f, MathF.Round(Convert.ToSingle(last3Obj), 2));
            Assert.Equal(89.33f, MathF.Round(Convert.ToSingle(last5Obj), 2));
        }

        [Fact]
        public void BuildRaceReport_SummarizesPartialMarketFallbackReason()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner Alpha",
                    AiOdds = 0.65d,
                    AiProbabilityMarketDerived = false,
                    BackPrice1 = 2m
                },
                new RunnerFlow
                {
                    HorseName = "Runner Beta",
                    AiOdds = 0.35d,
                    AiProbabilityMarketDerived = true,
                    AiProbabilityFallbackReason = "Model probability below clamp threshold; using market-implied probability",
                    BackPrice1 = 3.5m
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.112",
                raceTitle: "Partial Fallback Test",
                venueName: "Test Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 6, 16),
                offTime: new TimeSpan(14, 15, 0),
                raceDetails: "Maiden",
                going: null,
                backBookPercentage: null,
                layBookPercentage: null,
                raceUrl: null,
                flows: flows);

            const string expected = "1 of 2 AI win probabilities defaulted to market-implied odds for this race. The remaining 1 runner retained their model-derived probabilities. Reason: Model probability below clamp threshold; using market-implied probability.";
            Assert.Equal(expected, report.RaceFallbackSummary);
        }
        [Fact]
        public void ApplyScrapedFeatureFallbacks_SetsMetadataMissingFlagsWhenUnavailable()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
            var flow = new RunnerFlow
            {
                HorseName = "Runner Delta"
            };

            scraper.TestApplyScrapedFeatureFallbacks(
                featureVector,
                flow,
                flows: new[] { flow });

            Assert.True(featureVector.TryGetValue("ClassMissing", out var classMissing));
            Assert.True(Convert.ToBoolean(classMissing));
            Assert.True(featureVector.TryGetValue("GoingMissing", out var goingMissing));
            Assert.True(Convert.ToBoolean(goingMissing));
            Assert.True(featureVector.TryGetValue("SurfaceMissing", out var surfaceMissing));
            Assert.True(Convert.ToBoolean(surfaceMissing));
            Assert.True(featureVector.TryGetValue("DistanceMissing", out var distanceMissing));
            Assert.True(Convert.ToBoolean(distanceMissing));
            Assert.True(featureVector.TryGetValue("DistanceTextMissing", out var distanceTextMissing));
            Assert.True(Convert.ToBoolean(distanceTextMissing));
            Assert.True(featureVector.TryGetValue("BackBookPercentageMissing", out var backBookMissing));
            Assert.True(Convert.ToBoolean(backBookMissing));
            Assert.True(featureVector.TryGetValue("LayBookPercentageMissing", out var layBookMissing));
            Assert.True(Convert.ToBoolean(layBookMissing));
            Assert.True(featureVector.TryGetValue("RaceMetadataMissing", out var metadataMissing));
            Assert.True(Convert.ToBoolean(metadataMissing));

            Assert.False(featureVector.ContainsKey("Going"));
            Assert.False(featureVector.ContainsKey("Surface"));
            Assert.False(featureVector.ContainsKey("DistanceYards"));
            Assert.False(featureVector.ContainsKey("DistanceText"));
        }


        [Fact]
        public void ClampLowAiProbability_UsesMarketOddsAndRecordsFallbackReason()
        {
            var flow = new RunnerFlow
            {
                HorseName = "Clamped Runner",
                BackPrice1 = 4m,
                AiOdds = 1e-8,
                AiProbabilityMarketDerived = false,
                AiProbabilityFallbackReason = null,
                AiProbabilityClampedToMarket = false,
                AiProbabilityClampTarget = null
            };

            var method = typeof(BetfairMarketScraper).GetMethod(
                "TryClampLowAiProbabilityToMarket",
                BindingFlags.NonPublic | BindingFlags.Static);

            Assert.NotNull(method);

            var result = method!.Invoke(null, new object?[] { flow });

            Assert.IsType<bool>(result);
            Assert.True((bool)result!);

            var expectedProbability = 1.0d / 4.0d;

            Assert.True(flow.AiOdds.HasValue);
            Assert.Equal(expectedProbability, flow.AiOdds!.Value, 12);
            Assert.True(flow.AiProbabilityClampedToMarket);
            Assert.True(flow.AiProbabilityClampTarget.HasValue);
            Assert.Equal(expectedProbability, flow.AiProbabilityClampTarget!.Value, 12);
            Assert.True(flow.AiProbabilityMarketDerived);
            Assert.False(string.IsNullOrWhiteSpace(flow.AiProbabilityFallbackReason));
            Assert.Contains("Model probability below clamp threshold", flow.AiProbabilityFallbackReason);
        }
        [Fact]
        public void FeaturePopulationSummary_IncludesNeuralFeaturesWhenHistoryMissing()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            scraper.TestSetNeuralFeatureKeys(new[] { "AvgRatingLast5", "Class" });

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HistoricalDataMissing"] = true
            };

            var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

            Assert.Equal(0, summary.PopulatedCount);
            Assert.Equal(2, summary.MissingCount);
            Assert.Contains("AvgRatingLast5", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
            Assert.Contains("Class", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
        }
        [Fact]
        public void FeaturePopulationSummary_UsesNeuralFeatureKeysAndExcludesFallbackValues()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            scraper.TestSetNeuralFeatureKeys(new[] { "Age", "Class", "BackBookPercentage" });

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["Age"] = 5,
                ["BackBookPercentage"] = 102.5m,
                ["Class"] = 0
            };

            var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

            Assert.Equal(2, summary.PopulatedCount);
            Assert.Contains("Age", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
            Assert.Contains("BackBookPercentage", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
            Assert.Equal(1, summary.MissingCount);
            Assert.Contains("Class", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
        }
        [Fact]
        public void FeaturePopulationSummary_ClassDependentFallbacksPopulatedWhenContextAvailable()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            scraper.TestSetNeuralFeatureKeys(new[]
            {
                "ClassWinRate",
                "AvgRatingLast5",
                "TrainerJockeyCourseWinRate"
            });

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["Class"] = 4,
                ["CareerStarts"] = 3,
                ["ClassWinRate"] = 0f,
                ["AvgRatingLast5"] = 0f,
                ["TrainerJockeyCourseWinRate"] = 0f
            };

            var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

            Assert.Equal(3, summary.PopulatedCount);
            Assert.Contains("ClassWinRate", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
            Assert.Contains("AvgRatingLast5", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
            Assert.DoesNotContain("ClassWinRate", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
            Assert.DoesNotContain("AvgRatingLast5", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
            Assert.Contains("TrainerJockeyCourseWinRate", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
            Assert.DoesNotContain("TrainerJockeyCourseWinRate", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
        }
        [Fact]
        public void FeaturePopulationSummary_TreatsFalseHasLastWinAsPopulated()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            scraper.TestSetNeuralFeatureKeys(null);

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["HasLastWin"] = false
            };

            var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

            Assert.Contains("HasLastWin", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
            Assert.DoesNotContain("HasLastWin", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
        }
        [Fact]
        public void FeaturePopulationSummary_RatingFallbacksPopulatedWhenHistoryInsufficient()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            scraper.TestSetNeuralFeatureKeys(new[] { "AvgRatingLast10" });

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["CareerStarts"] = 4,
                ["AvgRatingLast10"] = 0f
            };

            var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

            Assert.Contains("AvgRatingLast10", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
            Assert.DoesNotContain("AvgRatingLast10", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
        }
        [Fact]
        public void FeaturePopulationSummary_RatingFallbacksRemainMissingWhenHistoryAvailable()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            scraper.TestSetNeuralFeatureKeys(new[] { "AvgRatingLast10" });

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["CareerStarts"] = 12,
                ["AvgRatingLast10"] = 0f
            };

            var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

            Assert.Contains("AvgRatingLast10", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
            Assert.DoesNotContain("AvgRatingLast10", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
        }        
        [Fact]
        public void FeaturePopulationSummary_RetainsEssentialKeysWhenModelMetadataMissing()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            try
            {
                scraper.TestSetNeuralFeatureKeys(new[] { "AvgRatingLast5", "Class" });

                var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                {
                    ["HasLastWin"] = true
                };

                var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

                Assert.Contains("HasLastWin", summary.PopulatedKeys, StringComparer.OrdinalIgnoreCase);
                Assert.DoesNotContain("HasLastWin", summary.MissingKeys, StringComparer.OrdinalIgnoreCase);
            }
            finally
            {
                scraper.TestSetNeuralFeatureKeys(null);
            }
        }
        [Fact]
        public void FeaturePopulationSummary_TracksEncodedDimensions()
        {
            var model = new TrainedModel
            {
                HiddenLayers = new List<LayerWeights>(),
                OutputLayer = new LayerWeights
                {
                    Weights = new[]
                    {
                        new float[] { 0f, 0f, 0f, 0f }
                    },
                    Bias = new float[] { 0f }
                },
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "FormRating", "RaceCode" },
                    FeatureDimensions = new Dictionary<string, int>
                    {
                        ["FormRating"] = 1,
                        ["RaceCode"] = 3
                    },
                    StringMaps = new Dictionary<string, Dictionary<string, int>>
                    {
                        ["RaceCode"] = new Dictionary<string, int>
                        {
                            ["A"] = 0,
                            ["B"] = 1
                        }
                    }
                },
                Normalization = new NormalizationParameters
                {
                    Mean = new[] { 0f, 0f, 0f, 0f },
                    StdDev = new[] { 1f, 1f, 1f, 1f }
                }
            };

            var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".json");
            try
            {
                File.WriteAllText(tempPath, JsonSerializer.Serialize(model));
                var calculator = new AIOddsCalculator(tempPath);

                var flow = new RunnerFlow
                {
                    HorseName = "Encoded Runner",
                    HasPreparedFeatures = false,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["FormRating"] = 82
                    }
                };

                calculator.CalculateOdds(flow);

                Assert.Equal(4, flow.FeaturePopulationSummary.TotalEncodedDimensions);
                Assert.Equal(1, flow.FeaturePopulationSummary.ActiveEncodedDimensions);
                Assert.Equal(4, flow.EncodedFeatureValues.Count);
                Assert.Equal(1, flow.EncodedFeatureValues.Count(value => value.Active));
                Assert.Contains("RaceCode", flow.FeaturePopulationSummary.MissingKeys, StringComparer.OrdinalIgnoreCase);
            }
            finally
            {
                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }
            }
        }
        [Fact]
        public void FeaturePopulationSummary_DoesNotRequireMissingRaceMetadata()
        {
            var repo = new StubRepository();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, new StubTrainer(), bankroll: 25m, settings);

            scraper.TestSetNeuralFeatureKeys(new[]
            {
                "Class",
                "TrainerClassWinRate",
                "Going",
                "TrainerGoingWinRate",
                "DistanceBucket",
                "DistanceBucketWinRate"
            });

            var featureVector = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);

            var summary = scraper.TestBuildFeaturePopulationSummary(featureVector);

            Assert.Equal(0, summary.PopulatedCount);
            Assert.Equal(0, summary.MissingCount);
            Assert.Empty(summary.MissingKeys);
        }
        [Fact]
        public void AIOddsCalculator_ActivatesMissingIndicatorSlots()
        {
            var model = new TrainedModel
            {
                OutputLayer = new LayerWeights
                {
                    Weights = new[]
                    {
                        new[] { 0f },
                        new[] { 0f }
                    },
                    Bias = new[] { 0f }
                },
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "TrainerClassWinRate" },
                    FeatureDimensions = new Dictionary<string, int>
                    {
                        ["TrainerClassWinRate"] = 2
                    }
                },
                Normalization = new NormalizationParameters
                {
                    Mean = new[] { 0f, 0f },
                    StdDev = new[] { 1f, 1f }
                }
            };

            var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".json");
            try
            {
                File.WriteAllText(tempPath, JsonSerializer.Serialize(model));
                var calculator = new AIOddsCalculator(tempPath);

                var flow = new RunnerFlow
                {
                    HorseName = "Missing Trainer Class",
                    HasPreparedFeatures = true,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                };

                calculator.CalculateOdds(flow);

                var encoded = Assert.IsType<List<EncodedFeatureValue>>(flow.EncodedFeatureValues);
                Assert.Equal(2, encoded.Count);
                Assert.All(encoded, value => Assert.True(value.Active));
                Assert.Equal(0d, encoded[0].Value);
                Assert.Equal(1d, encoded[1].Value);
            }
            finally
            {
                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }
            }
        }
        private static void InvokeApplyMarketFallback(IReadOnlyList<RunnerFlow> flows)
        {
            if (flows == null)
            {
                throw new ArgumentNullException(nameof(flows));
            }

            var method = typeof(BetfairMarketScraper).GetMethod(
                "ApplyMarketFallbackForUnmatchedRunners",
                BindingFlags.NonPublic | BindingFlags.Static);

            Assert.NotNull(method);
            method!.Invoke(null, new object[] { flows });
        }

        private sealed class StubRepository : IRacingRepository
        {
            public void ClearDayReportTables()
            {
            }
            public IReadOnlyList<RaceClassRating> HistoricalRatings { get; set; } = Array.Empty<RaceClassRating>();
            public void InsertRaceScreen(RaceScreen screen)
            {
            }

            public void InsertRunnerFlow(RunnerFlow flow)
            {
            }

            public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
            {
                if (flows == null)
                {
                    throw new ArgumentNullException(nameof(flows));
                }
            }

            public int UpsertUpcomingRace(UpcomingRace race)
            {
                return 0;
            }

            public UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName)
            {
                return null;
            }

            public UpcomingRace? GetUpcomingRaceByMarketId(string? marketId)
            {
                return null;
            }

            public int? GetHistoricalRaceCountByHorseName(string? horseName)
            {
                return null;
            }

            public HistoricalRaceCountPrefetchResult GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
            {
                return HistoricalRaceCountPrefetchResult.Empty;
            }
            public int GetHistoricalWinCountByHorseName(string? horseName)
            {
                return 0;
            }
            public HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName)
            {
                return null;
            }
            public IReadOnlyList<RunnerResult> GetLastSavedResults(string? raceTitle, DateTime? raceDate)
            {
                return Array.Empty<RunnerResult>();
            }
            public int? GetWinningTimeMilliseconds(int raceId) => null;
            public int? GetRaceIdByRunnerResult(int runnerResultId) => null;
            public decimal? GetDistanceBeatenLengths(int runnerResultId) => null;
            public decimal? GetLastDistanceBeatenLengths(string? horseName, int? horseId, DateTime? beforeDate) => null;
            public int? GetLastWinningTimeMilliseconds(string? horseName, int? horseId, DateTime? beforeDate) => null;
            public byte? GetMostRecentRaceClass(string? horseName, int? horseId) => null;
            public int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }
            public IReadOnlyList<RaceClassRating> GetHistoricalRaceClassRatings(string? horseName, int? horseId, int maxCount)
            {
                if (HistoricalRatings.Count == 0 || maxCount <= 0)
                {
                    return Array.Empty<RaceClassRating>();
                }

                return HistoricalRatings
                    .Take(maxCount)
                    .ToList();
            }
        }

        private sealed class StubTrainer : HorseRacingML.ML.HyperparameterTrainer
        {
            public StubTrainer()
                : base(new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30"
                    })
                    .Build())
            {
            }

            public override TrainingDataset.PreparedDataset PrepareDataset(
                ISet<int>? includeRaceIds = null,
                ISet<int>? stateRaceWhitelist = null,
                bool includeIdentifiers = false)
            {
                return new TrainingDataset.PreparedDataset(new List<TrainingDataset.PreparedDataset.PreparedRace>());
            }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                if (requests is null)
                {
                    throw new ArgumentNullException(nameof(requests));
                }

                return Array.Empty<PreparedRace?>();
            }
        }
    }
}