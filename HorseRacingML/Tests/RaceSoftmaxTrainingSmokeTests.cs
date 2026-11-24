using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.ML;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace HorseRacingML.Tests
{
    public class RaceSoftmaxTrainingSmokeTests
    {
        [Fact]
        public void TrainRaceSoftmax_WithPairwiseLoss_RunsAndProducesFiniteMetrics()
        {
            var config = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "Data Source=.\\SQLEXPRESS;Initial Catalog=HorseRacingML;Integrated Security=True;TrustServerCertificate=True",
                    ["Training:Objective"] = "RaceSoftmax",
                    ["Training:UseMarketOffset"] = "false",
                    ["Training:MarketBlind"] = "false",
                    ["Training:PairwiseLossWeight"] = "0.1",
                    ["Training:RaceBatchSize"] = "2",
                    ["Training:UseMarketBlend"] = "false",
                    ["Weather:Enabled"] = "false"
                })
                .Build();

            var trainer = new HyperparameterTrainer(config);
            var dataset = BuildDataset();
            var parameter = new MLParameter
            {
                Units = 4,
                Layers = 1,
                Dropout = 0,
                LearningRate = 0.01,
                Epochs = 1,
                BatchSize = 10,
                RaceBatchSize = 2,
                Folds = 1,
                Objective = "RaceSoftmax",
                MarketBlind = false,
                Seed = 42,
                FeatureBlocks = "None"
            };

            var result = trainer.Train(
                parameter,
                0,
                1,
                dataset,
                persistWeights: false,
                earlyStopOnValidation: true);

            Assert.True(double.IsFinite(result.ValidationRaceLogLoss));
            Assert.True(double.IsFinite(result.ValidationLoss));
            Assert.True(double.IsFinite(result.ValidationBrier));
            Assert.InRange(result.ValidationTop3Accuracy, 0.0, 1.0);
            Assert.NotNull(result.Model);
        }

        private static HyperparameterTrainer.TrainingDataset BuildDataset()
        {
            var trainingRaces = new List<HyperparameterTrainer.RaceExample>
            {
                BuildRace(1, new[] { 1f, 0f, 0.2f }),
                BuildRace(2, new[] { 0f, 1f, 0.4f })
            };
            var validationRaces = new List<HyperparameterTrainer.RaceExample>
            {
                BuildRace(3, new[] { 1f, 0f, 0.1f })
            };

            return new HyperparameterTrainer.TrainingDataset(
                trainingRaces,
                validationRaces,
                new List<string> { "X1", "X2" },
                new Dictionary<string, int> { ["X1"] = 1, ["X2"] = 1 },
                new Dictionary<string, Dictionary<string, int>>(),
                new NormalizationParameters
                {
                    Mean = new[] { 0f, 0f },
                    StdDev = new[] { 1f, 1f }
                });
        }

        private static HyperparameterTrainer.RaceExample BuildRace(int raceId, float[] winnerFeatures)
        {
            var runners = new List<HyperparameterTrainer.RunnerExample>
            {
                new(
                    raceId,
                    winnerFeatures,
                    1f,
                    horseId: 1,
                    horseName: "Winner",
                    startingPriceDecimal: 2m,
                    finishPosition: 1,
                    fieldSize: 3),
                new(
                    raceId,
                    new[] { 0.1f, 0.2f },
                    0f,
                    horseId: 2,
                    horseName: "Second",
                    startingPriceDecimal: 4m,
                    finishPosition: 2,
                    fieldSize: 3),
                new(
                    raceId,
                    new[] { 0.2f, 0.1f },
                    0f,
                    horseId: 3,
                    horseName: "Third",
                    startingPriceDecimal: 6m,
                    finishPosition: 3,
                    fieldSize: 3)
            };

            return new HyperparameterTrainer.RaceExample(
                raceId,
                new DateTime(2025, 1, raceId),
                runners);
        }
    }
}
