using System.Collections.Generic;
using HorseRacingML.ML;
using Xunit;

namespace HorseRacingML.Tests
{
    public class FeatureEncodingTests
    {
        [Fact]
        public void EncodeFeatureVector_UsesUnknownBucketForUnseenStrings()
        {
            var row = new Dictionary<string, object?>(System.StringComparer.OrdinalIgnoreCase)
            {
                ["RaceType"] = "Hunter Chase"
            };

            var featureKeys = new List<string> { "RaceType" };
            var stringMap = new Dictionary<string, int>(System.StringComparer.OrdinalIgnoreCase)
            {
                ["Flat"] = 0,
                ["__unknown__"] = 1
            };

            var stringMaps = new Dictionary<string, Dictionary<string, int>>(System.StringComparer.OrdinalIgnoreCase)
            {
                ["RaceType"] = stringMap
            };

            var featureDimensions = new Dictionary<string, int>(System.StringComparer.OrdinalIgnoreCase)
            {
                ["RaceType"] = stringMap.Count + 1 // reserve space for missing indicator
            };

            int featureCount = featureDimensions["RaceType"];
            var encoded = HyperparameterTrainer.EncodeFeatureVector(
                row,
                featureKeys,
                featureDimensions,
                stringMaps,
                featureCount);

            Assert.Equal(featureCount, encoded.Length);
            Assert.Equal(0f, encoded[featureCount - 1]);
            Assert.Equal(1f, encoded[1]);
        }
    }
}