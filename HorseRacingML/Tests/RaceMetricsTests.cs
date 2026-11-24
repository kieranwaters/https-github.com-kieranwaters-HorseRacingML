using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.ML;
using Xunit;

namespace HorseRacingML.Tests
{
    public class RaceMetricsTests
    {
        [Fact]
        public void Compute_ReportsRaceLogLossAndTopK()
        {
            var raceIds = new[] { 1, 1, 1, 2, 2, 2 };
            var predictions = new[] { 0.5f, 0.3f, 0.2f, 0.1f, 0.6f, 0.3f };
            var labels = new[] { 0f, 1f, 0f, 0f, 0f, 1f };

            var metrics = RaceMetrics.Compute(raceIds, predictions, labels);

            Assert.Equal(2, metrics.RaceCount);
            Assert.Equal(6, metrics.RunnerCount);
            Assert.Equal(-Math.Log(0.3), metrics.RaceLogLoss, 6);
            Assert.Equal(0.0, metrics.Top1Accuracy, 6);
            Assert.Equal(1.0, metrics.Top3Accuracy, 6);
        }

        [Fact]
        public void Compute_Top1Correct_WhenWinnerIsHighestProbability()
        {
            var raceIds = new[] { 1, 1, 1 };
            var predictions = new[] { 0.7f, 0.2f, 0.1f };
            var labels = new[] { 1f, 0f, 0f };

            var metrics = RaceMetrics.Compute(raceIds, predictions, labels);

            Assert.Equal(1.0, metrics.Top1Accuracy, 6);
            Assert.Equal(1.0, metrics.Top3Accuracy, 6);
            Assert.Equal(-Math.Log(0.7), metrics.RaceLogLoss, 6);
        }

        [Fact]
        public void AverageLogProbabilities_RenormalisesWithinRace()
        {
            var raceIds = new[] { 1, 1, 2, 2 };
            var memberA = new[] { 0.8f, 0.2f, 0.3f, 0.7f };
            var memberB = new[] { 0.6f, 0.4f, 0.5f, 0.5f };

            var averaged = RaceMetrics.AverageLogProbabilities(
                new IReadOnlyList<float>[] { memberA, memberB },
                raceIds);

            Assert.Equal(4, averaged.Length);
            Assert.Equal(1.0, averaged.Take(2).Sum(), 6);
            Assert.Equal(1.0, averaged.Skip(2).Sum(), 6);
            Assert.True(averaged[0] > averaged[1]);
            Assert.True(averaged[3] > averaged[2]);
            double geometric0 = Math.Sqrt(0.3 * 0.5);
            double geometric1 = Math.Sqrt(0.7 * 0.5);
            double raceSum = geometric0 + geometric1;
            Assert.Equal(geometric0 / raceSum, averaged[2], 6);
            Assert.Equal(geometric1 / raceSum, averaged[3], 6);
        }

        [Fact]
        public void Compute_HandlesDeadHeatWinnerMass()
        {
            var raceIds = new[] { 1, 1, 1 };
            var predictions = new[] { 0.4f, 0.4f, 0.2f };
            var labels = new[] { 1f, 1f, 0f };

            var metrics = RaceMetrics.Compute(raceIds, predictions, labels);

            Assert.Equal(-Math.Log(0.8), metrics.RaceLogLoss, 6);
            Assert.Equal(1.0, metrics.Top1Accuracy, 6);
        }
    }
}
