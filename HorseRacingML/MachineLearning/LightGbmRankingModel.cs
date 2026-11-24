using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Shared LightGBM LambdaRank training/scoring code used by the benchmark
    /// and by the optional persisted ensemble member.
    /// </summary>
    public static class LightGbmRankingModel
    {
        public sealed class RankingRow
        {
            [ColumnName("Label")]
            public float Label { get; set; }

            [ColumnName("GroupId")]
            public uint GroupId { get; set; }

            [ColumnName("Features")]
            public float[] Features { get; set; } = Array.Empty<float>();
        }

        public sealed class RankingPrediction
        {
            [ColumnName("Score")]
            public float Score { get; set; }
        }

        public static ITransformer Train(
            MLContext mlContext,
            IReadOnlyList<HyperparameterTrainer.RaceExample> races,
            int featureCount,
            int iterations = 200,
            int leaves = 31,
            double learningRate = 0.05,
            int minimumRowsPerLeaf = 50)
        {
            if (mlContext is null) throw new ArgumentNullException(nameof(mlContext));
            if (races is null) throw new ArgumentNullException(nameof(races));

            var schema = SchemaDefinition.Create(typeof(RankingRow));
            schema[nameof(RankingRow.Features)].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, featureCount);
            var data = mlContext.Data.LoadFromEnumerable(BuildRows(races), schema);
            var options = new LightGbmRankingTrainer.Options
            {
                LabelColumnName = "Label",
                RowGroupColumnName = "GroupId",
                FeatureColumnName = "Features",
                NumberOfLeaves = leaves,
                MinimumExampleCountPerLeaf = minimumRowsPerLeaf,
                LearningRate = learningRate,
                NumberOfIterations = iterations,
                MaximumBinCountPerFeature = 255
            };
            return mlContext.Ranking.Trainers.LightGbm(options).Fit(data);
        }

        public static float[] PredictProbabilities(
            MLContext mlContext,
            ITransformer model,
            IReadOnlyList<HyperparameterTrainer.RaceExample> races)
        {
            if (mlContext is null) throw new ArgumentNullException(nameof(mlContext));
            if (model is null) throw new ArgumentNullException(nameof(model));
            if (races is null) throw new ArgumentNullException(nameof(races));
            if (races.Count == 0)
            {
                return Array.Empty<float>();
            }

            int featureCount = races[0].Runners.Count > 0
                ? races[0].Runners[0].Features.Length
                : 0;
            var schema = SchemaDefinition.Create(typeof(RankingRow));
            schema[nameof(RankingRow.Features)].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, featureCount);
            var data = mlContext.Data.LoadFromEnumerable(BuildRows(races), schema);
            var scored = model.Transform(data);
            var scores = mlContext.Data
                .CreateEnumerable<RankingPrediction>(scored, reuseRowObject: false)
                .Select(row => row.Score)
                .ToArray();

            var probabilities = new float[scores.Length];
            int runnerIndex = 0;
            foreach (var race in races)
            {
                int count = race.Runners.Count;
                float max = float.NegativeInfinity;
                for (int i = 0; i < count; i++)
                {
                    max = Math.Max(max, scores[runnerIndex + i]);
                }

                double sum = 0;
                for (int i = 0; i < count; i++)
                {
                    double value = Math.Exp(scores[runnerIndex + i] - max);
                    probabilities[runnerIndex + i] = (float)value;
                    sum += value;
                }

                for (int i = 0; i < count; i++)
                {
                    probabilities[runnerIndex + i] = (float)(probabilities[runnerIndex + i] / sum);
                }

                runnerIndex += count;
            }

            return probabilities;
        }

        public static void Save(
            MLContext mlContext,
            ITransformer model,
            IReadOnlyList<HyperparameterTrainer.RaceExample> trainingRaces,
            int featureCount,
            string path)
        {
            if (mlContext is null) throw new ArgumentNullException(nameof(mlContext));
            if (model is null) throw new ArgumentNullException(nameof(model));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("A model path is required.", nameof(path));

            var schema = SchemaDefinition.Create(typeof(RankingRow));
            schema[nameof(RankingRow.Features)].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, featureCount);
            var data = mlContext.Data.LoadFromEnumerable(BuildRows(trainingRaces), schema);
            mlContext.Model.Save(model, data.Schema, path);
        }

        public static ITransformer? TryLoad(MLContext mlContext, string path)
        {
            if (mlContext is null || string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
            {
                return null;
            }

            try
            {
                return mlContext.Model.Load(path, out _);
            }
            catch
            {
                return null;
            }
        }

        public static List<RankingRow> BuildRows(
            IReadOnlyList<HyperparameterTrainer.RaceExample> races)
        {
            var rows = new List<RankingRow>();
            uint groupId = 0;
            foreach (var race in races)
            {
                groupId++;
                foreach (var runner in race.Runners)
                {
                    rows.Add(new RankingRow
                    {
                        Label = runner.Label,
                        GroupId = groupId,
                        Features = runner.Features
                    });
                }
            }

            return rows;
        }

        public static List<RankingRow> BuildPredictionRows(
            IReadOnlyList<float[]> features)
        {
            var rows = new List<RankingRow>(features.Count);
            foreach (var featureVector in features)
            {
                rows.Add(new RankingRow
                {
                    Label = 0f,
                    GroupId = 1,
                    Features = featureVector
                });
            }

            return rows;
        }

        public static float[] SoftmaxScores(IReadOnlyList<float> scores)
        {
            var probabilities = new float[scores.Count];
            if (scores.Count == 0)
            {
                return probabilities;
            }

            float max = float.NegativeInfinity;
            foreach (var score in scores)
            {
                max = Math.Max(max, score);
            }

            double sum = 0;
            for (int i = 0; i < scores.Count; i++)
            {
                probabilities[i] = (float)Math.Exp(scores[i] - max);
                sum += probabilities[i];
            }

            if (sum <= double.Epsilon || !double.IsFinite(sum))
            {
                return probabilities;
            }

            for (int i = 0; i < scores.Count; i++)
            {
                probabilities[i] = (float)(probabilities[i] / sum);
            }

            return probabilities;
        }
    }
}
