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
                new(new DateTime(2023, 1, 1), 0.5f, 2, "Good", "Turf", 1, "Sprint", 1, 40f, 41f, 1f, 3, true, 80f, 130f, true, true, 20000, 1000, 0f, false),
                new(new DateTime(2023, 2, 1), 0.6f, 3, "Soft", "Turf", 1, "Sprint", 1, 38f, 37f, -1f, 3, false, 81f, 130f, true, true, 22000, 1000, 0f, false),
                new(new DateTime(2023, 3, 1), 0.9f, 1, "Good", "Turf", 1, "Sprint", 1, 42f, 43f, 1f, 3, true, 85f, 130f, true, true, 19000, 1000, 0f, true),
                new(new DateTime(2023, 4, 1), 0.4f, 5, "Heavy", "Turf", 1, "Sprint", 1, 35f, 36f, 1f, 3, false, 83f, 130f, true, true, 24000, 1000, 0f, false),
                new(new DateTime(2023, 5, 1), 0.8f, 2, "Good", "Turf", 1, "Sprint", 1, 41f, 42f, 1f, 3, false, 88f, 130f, true, true, 19500, 1000, 0f, true),
                 // This is the 6th race, but the 4th on "Good" going. The window is 5.
                new(new DateTime(2023, 6, 1), 0.7f, 3, "Good", "All-Weather", 1, "Sprint", 1, 40f, 40f, 0f, 3, false, 90f, 130f, true, true, 20000, 1000, 0f, false),
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
        public void ProcessRace_StrictWindowLogic_RemovesFeaturesIfInsufficientHistory()
        {
            // Arrange
            var state = new HyperparameterTrainer.FeatureEngineeringState(_trainer);
            var horseId = 1;
            var history = new List<HyperparameterTrainer.HistoryEntry>
            {
                new(new DateTime(2023, 1, 1), 0.5f, 2, "Good", "Turf", 1, "Sprint", 1, 40f, 41f, 1f, 3, true, 80f, 130f, true, true, 20000, 1000, 0f, false),
                new(new DateTime(2023, 2, 1), 0.6f, 3, "Soft", "Turf", 1, "Sprint", 1, 38f, 37f, -1f, 3, false, 81f, 130f, true, true, 22000, 1000, 0f, false),
                // Only 2 races in history.
            };

            var horseHistoryField = typeof(HyperparameterTrainer.FeatureEngineeringState).GetField("_horseHistory", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var horseHistoryDict = (Dictionary<int, List<HyperparameterTrainer.HistoryEntry>>)horseHistoryField.GetValue(state);
            horseHistoryDict[horseId] = history;

            var raceData = new List<Dictionary<string, object?>>
            {
                new()
                {
                    { "RaceId", 100 },
                    { "HorseId", horseId },
                    { "RaceDate", new DateTime(2023, 3, 1) },
                    { "Going", "Good" },
                    { "Surface", "Turf" },
                    { "RunnerCount", 10 },
                    { "FinishPos", (short)1 }
                }
            };

            // Act
            state.ProcessRace(raceData, includeRace: true, updateState: false);
            var row = raceData.First();

            // Assert
            // Window 1 (History 2 >= 1): Should exist
            Assert.True(row.ContainsKey("WinRateLast1"));
            Assert.True(row.ContainsKey("AvgNormPosLast1"));

            // Window 2 (History 2 >= 2): Should exist (newly added window)
            Assert.True(row.ContainsKey("WinRateLast2"));
            Assert.True(row.ContainsKey("AvgNormPosLast2"));

            // Window 3 (History 2 < 3): Should NOT exist
            Assert.False(row.ContainsKey("WinRateLast3"));
            Assert.False(row.ContainsKey("AvgNormPosLast3"));

            // Window 5 (History 2 < 5): Should NOT exist
            Assert.False(row.ContainsKey("WinRateLast5"));
            Assert.False(row.ContainsKey("AvgNormPosLast5"));

            // Stage 1 features
            Assert.False(row.ContainsKey("WinRateLast5")); // Stage 1
            Assert.False(row.ContainsKey("AvgSpeedLast5")); // Stage 1
        }

        [Fact]
        public void ProcessRace_GeneratesNewWindowFeatures_2_and_4()
        {
            // Arrange
            var state = new HyperparameterTrainer.FeatureEngineeringState(_trainer);
            var horseId = 1;
            var history = new List<HyperparameterTrainer.HistoryEntry>();
            for (int i = 0; i < 5; i++)
            {
                history.Add(new(new DateTime(2023, 1, 1).AddDays(i), 0.5f, 2, "Good", "Turf", 1, "Sprint", 1, 40f, 41f, 1f, 3, true, 80f, 130f, true, true, 20000, 1000, 0f, false));
            }
            // History count is 5.

            var horseHistoryField = typeof(HyperparameterTrainer.FeatureEngineeringState).GetField("_horseHistory", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var horseHistoryDict = (Dictionary<int, List<HyperparameterTrainer.HistoryEntry>>)horseHistoryField.GetValue(state);
            horseHistoryDict[horseId] = history;

            var raceData = new List<Dictionary<string, object?>>
            {
                new()
                {
                    { "RaceId", 100 },
                    { "HorseId", horseId },
                    { "RaceDate", new DateTime(2023, 3, 1) },
                    { "Going", "Good" },
                    { "Surface", "Turf" },
                    { "RunnerCount", 10 },
                    { "FinishPos", (short)1 }
                }
            };

            // Act
            state.ProcessRace(raceData, includeRace: true, updateState: false);
            var row = raceData.First();

            // Assert
            // Window 2 should be present
            Assert.True(row.ContainsKey("WinRateLast2"));
            Assert.True(row.ContainsKey("AvgNormPosLast2"));

            // Window 4 should be present
            Assert.True(row.ContainsKey("WinRateLast4"));
            Assert.True(row.ContainsKey("AvgNormPosLast4"));
        }
        [Fact]
        public void ProcessRace_HandlesPulledUpOutcome()
        {
            // Arrange
            var state = new HyperparameterTrainer.FeatureEngineeringState(_trainer);
            var raceData = new List<Dictionary<string, object?>>
            {
                new()
                {
                    { "RaceId", 1 },
                    { "HorseId", 1 },
                    { "RaceDate", new DateTime(2023, 1, 1) },
                    { "RunnerCount", 3 },
                    { "FinishPos", null },
                    { "OutcomeCode", "PU" }
                },
                new()
                {
                    { "RaceId", 1 },
                    { "HorseId", 2 },
                    { "RaceDate", new DateTime(2023, 1, 1) },
                    { "RunnerCount", 3 },
                    { "FinishPos", (short)1 }
                },
                new()
                {
                    { "RaceId", 1 },
                    { "HorseId", 3 },
                    { "RaceDate", new DateTime(2023, 1, 1) },
                    { "RunnerCount", 3 },
                    { "FinishPos", (short)2 }
                }
            };

            // Act
            state.ProcessRace(raceData, includeRace: true, updateState: true);

            // Assert
            var pulledUpHorse = raceData.First(r => (int)r["HorseId"] == 1);
            Assert.Equal(4, pulledUpHorse["FinishPos"]);
        }
    }
}
