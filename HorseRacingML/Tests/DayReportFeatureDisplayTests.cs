using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using System.Linq;
using System;
using System.Collections.Generic;
using Xunit;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.Tests
{
    public class DayReportFeatureDisplayTests
    {
        [Fact]
        public void CreateRunnerReport_PopulatesJockeyClassFallbacksForDisplay()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["JockeyWinRate"] = 0.33f,
                        ["JockeyGoingAvgNorm"] = 0.21f,
                        ["LastJockeyGoingNormPos"] = 0.17f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.True(runner.FeatureValues.TryGetValue("JockeyClassWinRate", out var winRateObj));
            Assert.Equal(0.33f, Assert.IsType<float>(winRateObj));
            Assert.Equal(0.21f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.17f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }

        [Fact]
        public void CreateRunnerReport_PopulatesJockeyClassFallbacksWhenSourceMissing()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 10m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase),
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }
        [Fact]
        public void CreateRunnerReport_PopulatesTrainerClassFallbacksForDisplay()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 10m, settings);
            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["TrainerWinRate"] = 0.27f,
                        ["TrainerSurfaceAvgNorm"] = 0.22f,
                        ["LastTrainerSurfaceNormPos"] = 0.18f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.True(runner.FeatureValues.TryGetValue("TrainerClassWinRate", out var winRateObj));
            Assert.Equal(0.27f, Assert.IsType<float>(winRateObj));
            Assert.Equal(0.22f, Assert.IsType<float>(runner.FeatureValues["TrainerClassAvgNorm"]));
            Assert.Equal(0.18f, Assert.IsType<float>(runner.FeatureValues["LastTrainerClassNormPos"]));
        }

        [Fact]
        public void CreateRunnerReport_UsesGeneralClassMetricsWhenJockeyClassUnavailable()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["ClassWinRate"] = 0.44f,
                        ["ClassAvgNorm"] = 0.31f,
                        ["LastClassNormPos"] = 0.27f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0.44f, Assert.IsType<float>(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0.31f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.27f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }
        [Fact]
        public void CreateRunnerReport_PopulatesTrainerClassFallbacksWhenSourceMissing()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", bankroll: 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase),
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["TrainerClassWinRate"]));
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["TrainerClassAvgNorm"]));
            Assert.Equal(0f, Assert.IsType<float>(runner.FeatureValues["LastTrainerClassNormPos"]));
        }
        [Fact]
        public void CreateRunnerReport_PopulatesTrainerAndJockeyPerformanceSummaries()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 10m, settings);

            var trainerSamples = new List<ParticipantHistoricalRaceSummary>
            {
                new ParticipantHistoricalRaceSummary { FinishPosition = 1, RunnerCount = 5 },
                new ParticipantHistoricalRaceSummary { FinishPosition = 2, RunnerCount = 5 },
                new ParticipantHistoricalRaceSummary { FinishPosition = 5, RunnerCount = 5 }
            };
            var jockeySamples = new List<ParticipantHistoricalRaceSummary>
            {
                new ParticipantHistoricalRaceSummary { FinishPosition = 3, RunnerCount = 10 },
                new ParticipantHistoricalRaceSummary { FinishPosition = 1, RunnerCount = 10 },
                new ParticipantHistoricalRaceSummary { FinishPosition = 9, RunnerCount = 10 }
            };

            repo.SetRecentTrainerResults(7, "Trainer Example", trainerSamples);
            repo.SetRecentJockeyResults(11, "Jockey Example", jockeySamples);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    TrainerName = "Trainer Example",
                    JockeyName = "Jockey Example",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["TrainerId"] = 7,
                        ["JockeyId"] = 11
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);

            var expectedTrainerMetrics = ComputeExpectedMetrics((1, 5), (2, 5), (5, 5));
            Assert.Equal(expectedTrainerMetrics.Count, runner.TrainerHistoricalRaceCount);
            AssertApproximatelyEqual(
                expectedTrainerMetrics.Top3,
                runner.TrainerTop3FinishRate,
                6,
                "Trainer top 3 finish rate");
            AssertApproximatelyEqual(
                expectedTrainerMetrics.Top5,
                runner.TrainerTop5FinishRate,
                6,
                "Trainer top 5 finish rate");
            AssertApproximatelyEqual(
                expectedTrainerMetrics.Volatility,
                runner.TrainerFinishPositionVolatility,
                6,
                "Trainer finish volatility");

            var expectedJockeyMetrics = ComputeExpectedMetrics((3, 10), (1, 10), (9, 10));
            Assert.Equal(expectedJockeyMetrics.Count, runner.JockeyHistoricalRaceCount);
            AssertApproximatelyEqual(
                expectedJockeyMetrics.Top3,
                runner.JockeyTop3FinishRate,
                6,
                "Jockey top 3 finish rate");
            AssertApproximatelyEqual(
                expectedJockeyMetrics.Top5,
                runner.JockeyTop5FinishRate,
                6,
                "Jockey top 5 finish rate");
            AssertApproximatelyEqual(
                expectedJockeyMetrics.Volatility,
                runner.JockeyFinishPositionVolatility,
                6,
                "Jockey finish volatility");
        }
        [Fact]
        public void CreateRunnerReport_IgnoresNeutralTrainerJockeyWinRateFallback()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["TrainerJockeyWinRate"] = 0f,
                        ["TrainerJockeyCourseWinRate"] = 0f,
                        ["TrainerJockeySurfaceWinRate"] = 0f,
                        ["ClassWinRate"] = 0.44f,
                        ["ClassAvgNorm"] = 0.31f,
                        ["LastClassNormPos"] = 0.27f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0.44f, Assert.IsType<float>(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0.31f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.27f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }

        [Fact]
        public void CreateRunnerReport_SkipsNeutralClassFallbacksWhenGeneralStatsAvailable()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", bankroll: 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["Class"] = 4,
                        ["ClassWinRate"] = 0f,
                        ["TrainerClassWinRate"] = 0f,
                        ["TrainerWinRate"] = 0f,
                        ["ClassAvgNorm"] = 0f,
                        ["TrainerClassAvgNorm"] = 0f,
                        ["LastClassNormPos"] = 0f,
                        ["LastTrainerClassNormPos"] = 0f,
                        ["WinRateLast5"] = 0.28f,
                        ["AvgNormPosLast5"] = 0.36f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0.28f, Assert.IsType<float>(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0.36f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.36f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }
        [Fact]
        public void CreateRunnerReport_AppliesMaxStakeLimitToLayRecommendations()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.FixedAmount, null, 10m);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 100m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Lay Runner",
                    ClothNumber = 1,
                    LayPrice1 = 2.0m,
                    BackPrice1 = 2.0m,
                    AiOdds = 0.25,
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0.5m, runner.LayKellyFraction);
            Assert.Equal(10m, runner.LaySuggestedStake);
        }

        [Fact]
        public void CreateRunnerReport_UsesRecentPerformanceWhenClassAndTrainerDataMissing()
        {
            var repo = new StubRepository();
            var trainer = new StubTrainer();
            var settings = new AutomationSettingsSnapshot(1m, null, MaxStakeMode.None, null, null);
            var scraper = new BetfairMarketScraper(repo, trainer, "model.path", 10m, settings);

            var flows = new List<RunnerFlow>
            {
                new RunnerFlow
                {
                    HorseName = "Runner One",
                    JockeyName = "Sample Jockey",
                    ClothNumber = 1,
                    FeatureValues = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["WinRateLast5"] = 0.28f,
                        ["AvgNormPosLast5"] = 0.36f
                    },
                    HasPreparedFeatures = true
                }
            };

            var report = scraper.TestBuildRaceReport(
                marketId: "1.234",
                raceTitle: "Sample Race",
                venueName: "Sample Venue",
                venueCountry: "GB",
                raceDate: new DateTime(2024, 1, 1),
                offTime: new TimeSpan(12, 0, 0),
                raceDetails: null,
                going: "Good",
                backBookPercentage: 100m,
                layBookPercentage: 101m,
                raceUrl: null,
                flows: flows);

            var runner = Assert.Single(report.Runners);
            Assert.Equal(0.28f, Assert.IsType<float>(runner.FeatureValues["JockeyClassWinRate"]));
            Assert.Equal(0.36f, Assert.IsType<float>(runner.FeatureValues["JockeyClassAvgNorm"]));
            Assert.Equal(0.36f, Assert.IsType<float>(runner.FeatureValues["LastJockeyClassNormPos"]));
        }
        private static (int Count, double? Top3, double? Top5, double? Volatility) ComputeExpectedMetrics(params (int Finish, int Runners)[] results)
        {
            var numeric = results
                .Where(result => result.Finish > 0)
                .ToList();

            if (numeric.Count == 0)
            {
                return (0, null, null, null);
            }

            int count = numeric.Count;
            double? top3 = count > 0 ? numeric.Count(r => r.Finish <= 3) / (double)count : (double?)null;
            double? top5 = count > 0 ? numeric.Count(r => r.Finish <= 5) / (double)count : (double?)null;

            var normalized = new List<double>();
            foreach (var sample in numeric)
            {
                if (sample.Runners > 1 && sample.Finish <= sample.Runners)
                {
                    normalized.Add((sample.Runners - sample.Finish) / (double)(sample.Runners - 1));
                }
            }

            double? volatility = null;
            if (normalized.Count >= 2)
            {
                volatility = ComputeStandardDeviation(normalized);
            }
            else if (normalized.Count == 1)
            {
                volatility = 0d;
            }
            else
            {
                var fallback = numeric.Select(r => (double)r.Finish).ToList();
                if (fallback.Count >= 2)
                {
                    double maxFinish = fallback.Max();
                    if (maxFinish > 1d)
                    {
                        var approx = fallback
                            .Select(f => 1d - (f - 1d) / (maxFinish - 1d))
                            .ToList();
                        volatility = ComputeStandardDeviation(approx);
                    }
                }
            }

            return (count, top3, top5, volatility);
        }

        private static double ComputeStandardDeviation(IReadOnlyList<double> values)
        {
            if (values == null || values.Count == 0)
            {
                return 0d;
            }

            var mean = values.Average();
            var variance = values.Sum(value =>
            {
                var diff = value - mean;
                return diff * diff;
            }) / values.Count;

            return Math.Sqrt(variance);
        }
        private static void AssertApproximatelyEqual(double? expected, double? actual, int precision, string context)
        {
            if (expected.HasValue)
            {
                Assert.True(actual.HasValue, $"{context} should be populated.");
                Assert.Equal(expected.Value, actual.Value, precision);
            }
            else
            {
                Assert.Null(actual);
            }
        }
        private sealed class StubRepository : IRacingRepository
        {
            public void ClearDayReportTables()
            {
            }

            public void InsertRaceScreen(RaceScreen screen)
            {
            }

            public void InsertRunnerFlow(RunnerFlow flow)
            {
            }

            public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
            {
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

            public int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }

            public IReadOnlyList<RunnerResult> GetLastSavedResults(string? raceTitle, DateTime? raceDate)
            {
                return Array.Empty<RunnerResult>();
            }

            public int? GetWinningTimeMilliseconds(int raceId)
            {
                return null;
            }

            public int? GetRaceIdByRunnerResult(int runnerResultId)
            {
                return null;
            }

            public decimal? GetDistanceBeatenLengths(int runnerResultId)
            {
                return null;
            }

            public decimal? GetLastDistanceBeatenLengths(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }

            public int? GetLastWinningTimeMilliseconds(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
            }

            public byte? GetMostRecentRaceClass(string? horseName, int? horseId)
            {
                return null;
            }
            public byte? GetMostRecentRaceClassForHorseJockey(
                string? horseName,
                int? horseId,
                string? jockeyName,
                int? jockeyId)
            {
                return null;
            }
            public IReadOnlyList<RaceClassRating> GetHistoricalRaceClassRatings(string? horseName, int? horseId, int maxCount)
            {
                return Array.Empty<RaceClassRating>();
            }

            public (int Wins, int Starts)? GetRecentHorseWinStats(string? horseName, int? horseId, DateTime? beforeDate, int windowSize)
            {
                return null;
            }
            public IReadOnlyDictionary<HorseMetricRequest, (int Wins, int Starts)?> GetRecentHorseWinStatsBatch(
                IEnumerable<HorseMetricRequest> requests,
                int windowSize)
            {
                return new Dictionary<HorseMetricRequest, (int Wins, int Starts)?>();
            }
            public IReadOnlyList<HorseSpeedEntry> GetRecentHorseSpeedEntries(string? horseName, int? horseId, DateTime? beforeDate, int windowSize)
            {
                return Array.Empty<HorseSpeedEntry>();
            }
            public IReadOnlyDictionary<HorseMetricRequest, float?> GetRecentHorseAverageSpeedsBatch(
                IEnumerable<HorseMetricRequest> requests,
                int windowSize)
            {
                return new Dictionary<HorseMetricRequest, float?>();
            }
            public IReadOnlyList<HorseHistoricalRaceSummary> GetRecentHorseResults(string? horseName, int? horseId, int maxCount)
            {
                if (maxCount <= 0)
                {
                    return Array.Empty<HorseHistoricalRaceSummary>();
                }

                var key = (horseId, NormalizeHorseNameKey(horseName));
                if (_runnerHistoryCache.TryGetValue(key, out var cached))
                {
                    if (cached.Count <= maxCount)
                    {
                        return cached;
                    }

                    var limited = new List<HorseHistoricalRaceSummary>(maxCount);
                    for (var i = 0; i < maxCount; i++)
                    {
                        limited.Add(cached[i]);
                    }

                    return limited;
                }

                return Array.Empty<HorseHistoricalRaceSummary>();
            }
            public IReadOnlyList<ParticipantHistoricalRaceSummary> GetRecentTrainerResults(string? trainerName, int? trainerId, int maxCount)
            {
                return GetParticipantResults(trainerName, trainerId, maxCount, _trainerHistoryCache);
            }
            public IReadOnlyList<ParticipantHistoricalRaceSummary> GetRecentJockeyResults(string? jockeyName, int? jockeyId, int maxCount)
            {
                return GetParticipantResults(jockeyName, jockeyId, maxCount, _jockeyHistoryCache);
            }
            public IDictionary<int, RaceFeatureBackfill> GetRaceFeatureBackfills(IEnumerable<int> raceIds)
            {
                return new Dictionary<int, RaceFeatureBackfill>();
            }
            public void SetRecentHorseResults(int? horseId, string? horseName, IReadOnlyList<HorseHistoricalRaceSummary> results)
            {
                var key = (horseId, NormalizeHorseNameKey(horseName));
                _runnerHistoryCache[key] = results ?? Array.Empty<HorseHistoricalRaceSummary>();
            }
            public IReadOnlyDictionary<HorseMetricRequest, int?> GetLastRaceDistancesBatch(IEnumerable<HorseMetricRequest> requests)
            {
                return new Dictionary<HorseMetricRequest, int?>();
            }
            public void SetRecentTrainerResults(int? trainerId, string? trainerName, IReadOnlyList<ParticipantHistoricalRaceSummary> results)
            {
                var key = (NormalizeParticipantId(trainerId), NormalizeHorseNameKey(trainerName));
                _trainerHistoryCache[key] = results ?? Array.Empty<ParticipantHistoricalRaceSummary>();
            }
            public void SetRecentJockeyResults(int? jockeyId, string? jockeyName, IReadOnlyList<ParticipantHistoricalRaceSummary> results)
            {
                var key = (NormalizeParticipantId(jockeyId), NormalizeHorseNameKey(jockeyName));
                _jockeyHistoryCache[key] = results ?? Array.Empty<ParticipantHistoricalRaceSummary>();
            }
            private static IReadOnlyList<ParticipantHistoricalRaceSummary> GetParticipantResults(
                string? name,
                int? id,
                int maxCount,
                Dictionary<(int? Id, string NameKey), IReadOnlyList<ParticipantHistoricalRaceSummary>> cache)
            {
                if (maxCount <= 0)
                {
                    return Array.Empty<ParticipantHistoricalRaceSummary>();
                }

                var key = (NormalizeParticipantId(id), NormalizeHorseNameKey(name));
                if (cache.TryGetValue(key, out var cached))
                {
                    if (cached.Count <= maxCount)
                    {
                        return cached;
                    }

                    var limited = new List<ParticipantHistoricalRaceSummary>(maxCount);
                    for (var i = 0; i < maxCount; i++)
                    {
                        limited.Add(cached[i]);
                    }

                    return limited;
                }

                return Array.Empty<ParticipantHistoricalRaceSummary>();
            }
            private static string NormalizeHorseNameKey(string? horseName)
            {
                return string.IsNullOrWhiteSpace(horseName)
                    ? string.Empty
                    : horseName.Trim().ToUpperInvariant();
            }

            private static int? NormalizeParticipantId(int? id) => id.HasValue && id.Value > 0 ? id : null;

            private readonly Dictionary<(int? HorseId, string NameKey), IReadOnlyList<HorseHistoricalRaceSummary>> _runnerHistoryCache = new();
            private readonly Dictionary<(int? Id, string NameKey), IReadOnlyList<ParticipantHistoricalRaceSummary>> _trainerHistoryCache = new();
            private readonly Dictionary<(int? Id, string NameKey), IReadOnlyList<ParticipantHistoricalRaceSummary>> _jockeyHistoryCache = new();
        }

        private sealed class StubTrainer : HyperparameterTrainer
        {
            public StubTrainer()
                : base(new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=HorseRacingML;Integrated Security=True;Persist Security Info=False;Pooling=False;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Application Name=\"SQL Server Management Studio\";Command Timeout=30",
                        ["ML:ModelPath"] = "model.path"
                    })
                    .Build())
            {
            }

            public override PreparedDataset PrepareDataset(
                ISet<int?>? includeRaceIds = null,
                ISet<int?>? stateRaceWhitelist = null,
                bool includeIdentifiers = false,
                bool applyRepositoryBackfills = true)
            {
                return new PreparedDataset(new List<PreparedRace>());
            }

            public override IReadOnlyList<PreparedRace?> PrepareUpcomingRaces(
                IReadOnlyList<(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)> requests)
            {
                return Array.Empty<PreparedRace?>();
            }
        }
    }
}