using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HorseRacingML.ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using Xunit;

namespace HorseRacingML.Tests
{
    public class LightGbmRankingSmokeTests
    {
        private sealed class Row
        {
            [ColumnName("Label")]
            public float Label { get; set; }

            [ColumnName("GroupId")]
            public uint GroupId { get; set; }

            [ColumnName("Features")]
            public float[] Features { get; set; } = Array.Empty<float>();
        }

        private sealed class Prediction
        {
            [ColumnName("Score")]
            public float Score { get; set; }
        }

        [Fact]
        public void LightGbmRankingTrainer_CanTrainAndScoreSyntheticGroups()
        {
            var ml = new MLContext(seed: 42);
            var rows = new List<Row>();
            for (int race = 0; race < 20; race++)
            {
                for (int runner = 0; runner < 4; runner++)
                {
                    rows.Add(new Row
                    {
                        Label = runner == 0 ? 1f : 0f,
                        GroupId = (uint)(race + 1),
                        Features = new[]
                        {
                            runner == 0 ? 1f : 0f,
                            runner * 0.1f,
                            (race % 3) * 0.2f
                        }
                    });
                }
            }

            var schema = SchemaDefinition.Create(typeof(Row));
            schema[nameof(Row.Features)].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, 3);
            var data = ml.Data.LoadFromEnumerable(rows, schema);

            var options = new LightGbmRankingTrainer.Options
            {
                LabelColumnName = "Label",
                RowGroupColumnName = "GroupId",
                FeatureColumnName = "Features",
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 2,
                LearningRate = 0.1,
                NumberOfIterations = 10
            };
            var model = ml.Ranking.Trainers.LightGbm(options).Fit(data);
            var scored = model.Transform(data);
            var predictions = ml.Data
                .CreateEnumerable<Prediction>(scored, reuseRowObject: false)
                .Select(row => row.Score)
                .ToArray();

            Assert.Equal(rows.Count, predictions.Length);
            Assert.All(predictions, score => Assert.True(float.IsFinite(score)));
        }

        [Fact]
        public void LightGbmMember_SavesLoadsAndScoresThroughRuntime()
        {
            var races = BuildRaces();
            var ml = new MLContext(seed: 42);
            var model = LightGbmRankingModel.Train(
                ml,
                races,
                featureCount: 4,
                iterations: 10,
                leaves: 4,
                learningRate: 0.1,
                minimumRowsPerLeaf: 2);

            string path = Path.Combine(
                Path.GetTempPath(),
                $"lightgbm-runtime-{Guid.NewGuid():N}.zip");
            try
            {
                LightGbmRankingModel.Save(ml, model, races, 4, path);

                var metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "X1", "X2", "X3", "X4" },
                    FeatureDimensions = new Dictionary<string, int>
                    {
                        ["X1"] = 1,
                        ["X2"] = 1,
                        ["X3"] = 1,
                        ["X4"] = 1
                    },
                    StringMaps = new Dictionary<string, Dictionary<string, int>>()
                };
                var member = new TrainedModel
                {
                    Objective = "RaceSoftmax",
                    MarketOffsetScale = 0f,
                    MarketOffsetMode = "LogProb",
                    BlendModelWeight = 1f,
                    BlendMarketWeight = 0f,
                    LightGbmModelFile = path,
                    Metadata = metadata,
                    Normalization = new NormalizationParameters
                    {
                        Mean = new float[4],
                        StdDev = Enumerable.Repeat(1f, 4).ToArray()
                    }
                };
                member.FeatureSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(metadata);

                var runtime = TrainedModelRuntime.FromModel(
                    member,
                    Path.GetDirectoryName(path),
                    out var status);
                Assert.NotNull(runtime);
                Assert.True(runtime!.IsRaceSoftmax);
                Assert.True(runtime.TryPredictEncodedRace(races[0].Runners, out var probabilities));
                Assert.Equal(races[0].Runners.Count, probabilities.Length);
                Assert.Equal(1.0, probabilities.Sum(), 6);
                Assert.All(probabilities, probability => Assert.InRange(probability, 0.0, 1.0));
            }
            finally
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
        }

        private static List<HyperparameterTrainer.RaceExample> BuildRaces()
        {
            var races = new List<HyperparameterTrainer.RaceExample>();
            for (int raceIndex = 0; raceIndex < 20; raceIndex++)
            {
                var runners = new List<HyperparameterTrainer.RunnerExample>();
                for (int runnerIndex = 0; runnerIndex < 4; runnerIndex++)
                {
                    runners.Add(new HyperparameterTrainer.RunnerExample(
                        raceIndex + 1,
                        new[]
                        {
                            runnerIndex == 0 ? 1f : 0f,
                            runnerIndex * 0.1f,
                            (raceIndex % 3) * 0.2f,
                            1f
                        },
                        runnerIndex == 0 ? 1f : 0f,
                        horseId: runnerIndex + 1,
                        horseName: null,
                        startingPriceDecimal: null,
                        finishPosition: runnerIndex + 1,
                        fieldSize: 4));
                }

                races.Add(new HyperparameterTrainer.RaceExample(
                    raceIndex + 1,
                    new DateTime(2025, 1, 1).AddDays(raceIndex),
                    runners));
            }

            return races;
        }
    }
}
