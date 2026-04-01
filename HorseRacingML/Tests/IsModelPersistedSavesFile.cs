using HorseRacingML.Data;
using HorseRacingML.ML;
using System.Collections.Generic;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using System.IO;
using Xunit;

namespace HorseRacingML.Tests
{
    public class IsModelPersistedSavesFile
    {
        private readonly IConfiguration _configuration;
        private readonly HyperparameterTrainer _trainer;
        private readonly string _modelPath;

        private class TestHyperparameterTrainer : HyperparameterTrainer
        {
            public TestHyperparameterTrainer(IConfiguration configuration, IRacingRepository repository)
                : base(configuration)
            {
                SetRacingRepository(repository);
            }
        }

        public IsModelPersistedSavesFile()
        {
            _modelPath = Path.Combine(Path.GetTempPath(), "aiweights.json");
            _configuration = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ConnectionStrings:HorseRacingDb"] = "DataSource=file::memory:",
                    ["ML:ModelPath"] = _modelPath
                })
                .Build();
            _trainer = new TestHyperparameterTrainer(_configuration, new RacingRepository(""));
        }//

        [Fact]
        public void IsModelPersisted_ReturnsFalse_WhenModelNotPersisted()
        {
            // Arrange
            if (File.Exists(_modelPath))
            {
                File.Delete(_modelPath);
            }

            // Act
            var result = _trainer.IsModelPersisted();

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void IsModelPersisted_ReturnsTrue_WhenModelIsPersisted()
        {
            // Arrange
            if (File.Exists(_modelPath))
            {
                File.Delete(_modelPath);
            }
            var model = new MLParameter();
            var dataset = _trainer.LoadTrainingDataset();
            _trainer.Train(model, 0, 1, dataset, persistWeights: true);

            // Act
            var result = _trainer.IsModelPersisted();

            // Assert
            Assert.True(result);
        }
    }
}