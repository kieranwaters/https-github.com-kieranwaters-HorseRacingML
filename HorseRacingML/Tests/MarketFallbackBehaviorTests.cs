using System;
using System.Collections.Generic;
using System.Reflection;
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

            public HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName)
            {
                return null;
            }

            public int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate)
            {
                return null;
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