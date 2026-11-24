using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.ML;
using Xunit;

namespace HorseRacingML.Tests
{
    public class EnsembleRuntimeTests
    {
        [Fact]
        public void EnsembleRuntime_AveragesMemberProbabilitiesInLogSpace()
        {
            var member1 = BuildModel(weight: 1f);
            var member2 = BuildModel(weight: 2f);
            var ensemble = BuildModel(weight: 1f);
            ensemble.Members = new List<TrainedModel> { member1, member2 };
            ensemble.FeatureSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(ensemble.Metadata);

            var runtime = TrainedModelRuntime.FromModel(ensemble, out var status);
            Assert.NotNull(runtime);
            Assert.True(runtime!.IsEnsemble);
            Assert.Equal(2, runtime.MemberCount);

            var rows = new List<IDictionary<string, object?>>
            {
                new Dictionary<string, object?> { ["X"] = 0f },
                new Dictionary<string, object?> { ["X"] = 1f }
            };

            Assert.True(runtime.TryPredictRace(rows, out var probabilities));
            Assert.Equal(2, probabilities.Length);
            Assert.True(probabilities[1] > probabilities[0]);
            Assert.Equal(1.0, probabilities.Sum(), 6);

            // Member 1: softmax([0, 1]); member 2: softmax([0, 2]).
            // The ensemble averages log probabilities and renormalises.
            double p0 = Math.Sqrt((1.0 / (1.0 + Math.Exp(1.0))) * (1.0 / (1.0 + Math.Exp(2.0))));
            double p1 = Math.Sqrt((Math.Exp(1.0) / (1.0 + Math.Exp(1.0))) * (Math.Exp(2.0) / (1.0 + Math.Exp(2.0))));
            double sum = p0 + p1;
            Assert.Equal(p0 / sum, probabilities[0], 5);
            Assert.Equal(p1 / sum, probabilities[1], 5);
        }

        [Fact]
        public void EnsembleArtifact_RoundTripsWithoutSerializationCycle()
        {
            var member1 = BuildModel(weight: 1f);
            var member2 = BuildModel(weight: 2f);
            var container = new TrainedModel
            {
                Objective = "RaceSoftmax",
                Metadata = member1.Metadata,
                Normalization = member1.Normalization,
                Members = new List<TrainedModel> { member1, member2 },
                TrainedUtc = DateTime.UtcNow
            };
            container.FeatureSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(container.Metadata);

            string json = System.Text.Json.JsonSerializer.Serialize(container);
            var loaded = System.Text.Json.JsonSerializer.Deserialize<TrainedModel>(json);
            Assert.NotNull(loaded);

            var runtime = TrainedModelRuntime.FromModel(loaded!, out var status);
            Assert.NotNull(runtime);
            Assert.True(runtime!.IsEnsemble);
            Assert.Equal(2, runtime.MemberCount);

            var rows = new List<IDictionary<string, object?>>
            {
                new Dictionary<string, object?> { ["X"] = 0.5f },
                new Dictionary<string, object?> { ["X"] = 1.5f }
            };
            Assert.True(runtime.TryPredictRace(rows, out var probabilities));
            Assert.Equal(1.0, probabilities.Sum(), 6);
        }

        [Fact]
        public void EncodedInferenceMatchesRawInference()
        {
            var model = BuildModel(weight: 1f);
            var runtime = TrainedModelRuntime.FromModel(model, out var status);
            Assert.NotNull(runtime);

            var rows = new List<IDictionary<string, object?>>
            {
                new Dictionary<string, object?> { ["X"] = 0f },
                new Dictionary<string, object?> { ["X"] = 1f }
            };
            Assert.True(runtime!.TryPredictRace(rows, out var raw));

            var examples = new List<HyperparameterTrainer.RunnerExample>
            {
                new(1, new[] { 0f }, 0f, 1, "A", 2m),
                new(1, new[] { 1f }, 1f, 2, "B", 3m)
            };
            Assert.True(runtime.MatchesFeatureSchema(
                new List<string> { "X" },
                new Dictionary<string, int> { ["X"] = 1 }));
            Assert.True(runtime.TryPredictEncodedRace(examples, out var encoded));
            Assert.Equal(raw[0], encoded[0], 6);
            Assert.Equal(raw[1], encoded[1], 6);
            Assert.False(runtime.MatchesFeatureSchema(
                new List<string> { "Y" },
                new Dictionary<string, int> { ["Y"] = 1 }));
        }

        [Fact]
        public void FieldCalibration_ChangesAndRenormalisesProbabilities()
        {
            var model = BuildModel(weight: 1f);
            model.FieldCalibration = new List<FieldCalibrationBucket>
            {
                new() { MaxFieldSize = 8, Intercept = 0.1, Slope = 1.2 },
                new() { MaxFieldSize = int.MaxValue, Intercept = 0.1, Slope = 1.2 }
            };
            var runtime = TrainedModelRuntime.FromModel(model, out var status);
            Assert.NotNull(runtime);

            var rows = new List<IDictionary<string, object?>>
            {
                new Dictionary<string, object?> { ["X"] = 0f },
                new Dictionary<string, object?> { ["X"] = 1f }
            };
            Assert.True(runtime!.TryPredictRace(rows, out var calibrated));
            Assert.Equal(1.0, calibrated.Sum(), 6);
            Assert.NotEqual(0.5, calibrated[0], 3);
            Assert.True(calibrated[1] > calibrated[0]);
        }

        private static TrainedModel BuildModel(float weight)
        {
            return new TrainedModel
            {
                HiddenLayers = new List<LayerWeights>(),
                OutputLayer = new LayerWeights
                {
                    Weights = new[] { new[] { weight } },
                    Bias = new[] { 0f }
                },
                Temperature = 1f,
                Objective = "RaceSoftmax",
                MarketOffsetScale = 0f,
                MarketOffsetMode = "LogProb",
                MarketBlind = true,
                BlendModelWeight = 1f,
                BlendMarketWeight = 0f,
                NormalizationMode = "PerRaceSoftmax",
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "X" },
                    FeatureDimensions = new Dictionary<string, int> { ["X"] = 1 },
                    StringMaps = new Dictionary<string, Dictionary<string, int>>()
                },
                Normalization = new NormalizationParameters
                {
                    Mean = new[] { 0f },
                    StdDev = new[] { 1f }
                },
                TrainedUtc = DateTime.UtcNow
            };
        }
    }
}
