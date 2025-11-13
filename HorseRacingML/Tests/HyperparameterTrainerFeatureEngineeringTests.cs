using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using static HorseRacingML.ML.HyperparameterTrainer;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.Tests
{
    public class HyperparameterTrainerFeatureEngineeringTests
    {
        private readonly IConfiguration _configuration;
        private readonly HyperparameterTrainer _trainer;

        public HyperparameterTrainerFeatureEngineeringTests()
        {
            _configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "DataSource=file::memory:"
                })
                .Build();
            _trainer = new HyperparameterTrainer(_configuration);
        }

        [Fact]
        public void ProcessRace_CalculatesAveragesCorrectly_AfterFix()
        {
            // Arrange
            var state = new HyperparameterTrainer.FeatureEngineeringState(_trainer);
            var horseId = 1;
            var history = new List<HyperparameterTrainer.HistoryEntry>
            {
                // Older races
                new(new DateTime(2023, 1, 1), 0.5f, 2, "Good", "Turf", 1, "Sprint", 1, 40f, 41f, 1f, 3, true, 80f, 130f, true, true, 20000, 1000),
                new(new DateTime(2023, 2, 1), 0.6f, 3, "Soft", "Turf", 1, "Sprint", 1, 38f, 37f, -1f, 3, false, 81f, 130f, true, true, 22000, 1000),
                new(new DateTime(2023, 3, 1), 0.9f, 1, "Good", "Turf", 1, "Sprint", 1, 42f, 43f, 1f, 3, true, 85f, 130f, true, true, 19000, 1000),
                new(new DateTime(2023, 4, 1), 0.4f, 5, "Heavy", "Turf", 1, "Sprint", 1, 35f, 36f, 1f, 3, false, 83f, 130f, true, true, 24000, 1000),
                new(new DateTime(2023, 5, 1), 0.8f, 2, "Good", "Turf", 1, "Sprint", 1, 41f, 42f, 1f, 3, false, 88f, 130f, true, true, 19500, 1000),
                 // This is the 6th race, but the 4th on "Good" going. The window is 5.
                new(new DateTime(2023, 6, 1), 0.7f, 3, "Good", "All-Weather", 1, "Sprint", 1, 40f, 40f, 0f, 3, false, 90f, 130f, true, true, 20000, 1000),
            };

            // Manually set the history for the horse
            var horseHistoryField = typeof(HyperparameterTrainer.FeatureEngineeringState).GetField("_horseHistory", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var horseHistoryDict = (Dictionary<int, List<HyperparameterTrainer.HistoryEntry>>)horseHistoryField.GetValue(state);
            horseHistoryDict[horseId] = history;

            var raceData = new List<Dictionary<string, object?>>
            {
                new()
                {
                    { "RaceId", 100 },
                    { "HorseId", horseId },
                    { "RaceDate", new DateTime(2023, 7, 1) },
                    { "Going", "Good" },
                    { "Surface", "Turf" },
                    { "RunnerCount", 10 },
                    { "FinishPos", (short)1 },
                    { "OfficialRating", 95f }
                }
            };

            // Act
            state.ProcessRace(raceData, includeRace: true, updateState: false);

            // Assert
            var resultRow = raceData.First();

            // 1. Test AvgRatingLast5
            // Expected: Average of last 5 ratings -> (81+85+83+88+90)/5 = 85.4
            var avgRating = (float)resultRow["AvgRatingLast5"];
            Assert.True(Math.Abs(85.4f - avgRating) < 0.01, $"Expected AvgRatingLast5 to be ~85.4 but got {avgRating}");

            // 2. Test AvgSpeedOnGoingLast5
            // History has 4 races on "Good" going. Speeds are 41, 43, 42, 40. Average = 41.5
            var avgSpeedOnGoing = (float)resultRow["AvgSpeedOnGoingLast5"];
            Assert.True(Math.Abs(41.5f - avgSpeedOnGoing) < 0.01, $"Expected AvgSpeedOnGoingLast5 to be ~41.5 but got {avgSpeedOnGoing}");

            // 3. Test GoingAvgNormLast5
            // History has 4 races on "Good" going. NormFinish are 0.5, 0.9, 0.8, 0.7. Average = 0.725
            var goingAvgNorm = (float)resultRow["GoingAvgNormLast5"];
            Assert.True(Math.Abs(0.725f - goingAvgNorm) < 0.01, $"Expected GoingAvgNormLast5 to be ~0.725 but got {goingAvgNorm}");
        }

        [Fact]
        public void ExcludeFeaturesWithInsufficientData_RemovesCorrectFeatures()
        {
            // Arrange
            var calculator = new AIOddsCalculator("fakepath"); // Path doesn't matter for this test
            var method = typeof(AIOddsCalculator).GetMethod("ExcludeFeaturesWithInsufficientData", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            var rawFeatures = new Dictionary<string, object?>(System.StringComparer.OrdinalIgnoreCase)
            {
                { "CareerStarts", 20 },
                { "WinRateLast10", 0.5f },
                { "AvgNormPosLast10", 0.6f },
                { "WinRateLast25", 0.4f }, // Should be removed
                { "AvgNormPosLast25", 0.5f }, // Should be removed
                { "WinRateLast100", 0.3f }, // Should be removed
                { "AvgNormPosLast100", 0.4f } // Should be removed
            };

            // Act
            method.Invoke(calculator, new object[] { rawFeatures });

            // Assert
            Assert.Contains("WinRateLast10", rawFeatures.Keys);
            Assert.Contains("AvgNormPosLast10", rawFeatures.Keys);
            Assert.DoesNotContain("WinRateLast25", rawFeatures.Keys);
            Assert.DoesNotContain("AvgNormPosLast25", rawFeatures.Keys);
            Assert.DoesNotContain("WinRateLast100", rawFeatures.Keys);
            Assert.DoesNotContain("AvgNormPosLast100", rawFeatures.Keys);
        }
    }
}
