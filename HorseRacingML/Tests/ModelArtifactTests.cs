using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HorseRacingML.ML;
using Xunit;

namespace HorseRacingML.Tests
{
    public class ModelArtifactTests
    {
        [Fact]
        public void SavingAnActivatedModelWritesVersionedLegacyAndPointerFiles()
        {
            var directory = CreateTempDirectory();
            try
            {
                var model = CreateModel();
                var artifact = ModelArtifactStore.Save(model, directory, activate: true);

                Assert.True(File.Exists(artifact.FullPath));
                Assert.True(File.Exists(Path.Combine(directory, ModelArtifactStore.LegacyFileName)));
                Assert.True(File.Exists(Path.Combine(directory, ModelArtifactStore.ActiveFileName)));
                Assert.Equal(artifact.FullPath, ModelArtifactStore.ResolveActivePath(directory));
                Assert.Equal(
                    ModelArtifactStore.ComputeFeatureSchemaHash(model.Metadata),
                    artifact.FeatureSchemaHash);
            }
            finally
            {
                DeleteDirectory(directory);
            }
        }

        [Fact]
        public void InactiveSaveDoesNotChangeTheActiveModel()
        {
            var directory = CreateTempDirectory();
            try
            {
                var active = ModelArtifactStore.Save(CreateModel(), directory, activate: true);
                var candidate = ModelArtifactStore.Save(CreateModel(), directory, activate: false);

                Assert.Equal(active.FullPath, ModelArtifactStore.ResolveActivePath(directory));

                ModelArtifactStore.Activate(
                    directory,
                    candidate.FileName,
                    candidate.Sha256,
                    candidate.FeatureSchemaHash);

                Assert.Equal(candidate.FullPath, ModelArtifactStore.ResolveActivePath(directory));
            }
            finally
            {
                DeleteDirectory(directory);
            }
        }

        [Fact]
        public void SchemaHashIgnoresStringMapInsertionOrder()
        {
            var first = new FeatureMetadata
            {
                Keys = new List<string> { "Going" },
                FeatureDimensions = new Dictionary<string, int> { ["Going"] = 2 },
                StringMaps = new Dictionary<string, Dictionary<string, int>>
                {
                    ["Going"] = new Dictionary<string, int> { ["good"] = 1, ["soft"] = 2 }
                }
            };
            var second = new FeatureMetadata
            {
                Keys = new List<string> { "Going" },
                FeatureDimensions = new Dictionary<string, int> { ["Going"] = 2 },
                StringMaps = new Dictionary<string, Dictionary<string, int>>
                {
                    ["Going"] = new Dictionary<string, int> { ["soft"] = 2, ["good"] = 1 }
                }
            };

            Assert.Equal(
                ModelArtifactStore.ComputeFeatureSchemaHash(first),
                ModelArtifactStore.ComputeFeatureSchemaHash(second));
        }

        [Fact]
        public void RuntimeRejectsAFeatureSchemaHashThatDoesNotMatchTheMetadata()
        {
            var model = CreateModel();
            model.FeatureSchemaHash = "0000000000000000000000000000000000000000000000000000000000000000";

            var runtime = TrainedModelRuntime.FromModel(model, out var status);

            Assert.Null(runtime);
            Assert.Contains("schema hash", status, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void RaceSoftmaxRuntimeScoresACompleteFieldDeterministically()
        {
            var model = CreateModel();
            model.Objective = "RaceSoftmax";
            model.FeatureSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(model.Metadata);

            var runtime = TrainedModelRuntime.FromModel(model, out var status);
            Assert.NotNull(runtime);
            Assert.True(runtime!.IsRaceSoftmax);
            Assert.Equal("Model loaded.", status);

            var rows = new List<IDictionary<string, object?>>
            {
                new Dictionary<string, object?> { ["Draw"] = 1d },
                new Dictionary<string, object?> { ["Draw"] = 2d },
                new Dictionary<string, object?> { ["Draw"] = 3d }
            };

            Assert.True(runtime.TryPredictRace(rows, out var first));
            Assert.True(runtime.TryPredictRace(rows, out var second));

            Assert.Equal(3, first.Length);
            Assert.Equal(1d, first.Sum(), 6);
            Assert.Equal(first, second);
        }

        private static TrainedModel CreateModel()
        {
            return new TrainedModel
            {
                Objective = "Binary",
                Temperature = 1f,
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string> { "Draw" },
                    FeatureDimensions = new Dictionary<string, int> { ["Draw"] = 1 },
                    StringMaps = new Dictionary<string, Dictionary<string, int>>()
                },
                Normalization = new NormalizationParameters
                {
                    Mean = new[] { 0f },
                    StdDev = new[] { 1f }
                },
                HiddenLayers = new List<LayerWeights>
                {
                    new()
                    {
                        Weights = new[] { new[] { 1f } },
                        Bias = new[] { 0f }
                    }
                },
                OutputLayer = new LayerWeights
                {
                    Weights = new[] { new[] { 1f } },
                    Bias = new[] { 0f }
                }
            };
        }

        private static string CreateTempDirectory()
        {
            var path = Path.Combine(
                Path.GetTempPath(),
                "HorseRacingMLTests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return path;
        }

        private static void DeleteDirectory(string path)
        {
            try
            {
                if (Directory.Exists(path))
                {
                    Directory.Delete(path, recursive: true);
                }
            }
            catch
            {
                // Best effort cleanup.
            }
        }
    }
}
