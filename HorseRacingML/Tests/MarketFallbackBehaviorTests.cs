using System;
using System.Collections.Generic;
using System.Reflection;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Xunit;

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
    }
}