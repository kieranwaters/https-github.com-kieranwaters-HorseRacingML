using System;
using System.Collections.Generic;
using HorseRacingML.ML;
using Xunit;

namespace HorseRacingML.Tests
{
    public class MarketBlendCalibrationTests
    {
        [Fact]
        public void FitMarketOnly_OnCalibratedSyntheticMarket_SelectsNearRawMarket()
        {
            var random = new Random(123);
            var predictions = new List<float>();
            var labels = new List<float>();
            var raceIds = new List<int>();
            var examples = new List<HyperparameterTrainer.RunnerExample>();

            for (int race = 0; race < 1500; race++)
            {
                int field = random.Next(3, 13);
                var probabilities = new double[field];
                double total = 0;
                for (int i = 0; i < field; i++)
                {
                    probabilities[i] = Math.Exp(random.NextDouble() * 3.0);
                    total += probabilities[i];
                }

                for (int i = 0; i < field; i++)
                {
                    probabilities[i] /= total;
                }

                int winner = 0;
                double draw = random.NextDouble();
                double cumulative = 0;
                for (int i = 0; i < field; i++)
                {
                    cumulative += probabilities[i];
                    if (draw <= cumulative)
                    {
                        winner = i;
                        break;
                    }
                }

                for (int i = 0; i < field; i++)
                {
                    int raceId = race + 1;
                    float label = i == winner ? 1f : 0f;
                    float probability = (float)probabilities[i];
                    predictions.Add(probability);
                    labels.Add(label);
                    raceIds.Add(raceId);
                    examples.Add(new HyperparameterTrainer.RunnerExample(
                        raceId,
                        Array.Empty<float>(),
                        label,
                        horseId: i + 1,
                        horseName: null,
                        startingPriceDecimal: null,
                        marketLogit: (float)Math.Log(probability / (1 - probability)),
                        hasMarketPrice: true,
                        marketProbability: probability,
                        marketLogProbability: (float)Math.Log(probability),
                        finishPosition: i == winner ? 1 : i + 2,
                        fieldSize: field));
                }
            }

            var market = MarketBlendCalibration.FitMarketOnly(
                predictions,
                labels,
                raceIds,
                examples);

            Assert.True(market.HasData);
            Assert.InRange(market.MarketWeight, 0.5, 2.0);
            Assert.True(market.Metrics.RaceLogLoss > 0);
            Assert.InRange(market.Metrics.Top1Accuracy, 0.0, 1.0);
            Assert.InRange(market.Metrics.Top3Accuracy, market.Metrics.Top1Accuracy, 1.0);
        }
    }
}
