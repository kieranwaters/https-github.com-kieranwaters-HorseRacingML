using HorseRacingML.ML;
using Xunit;

namespace HorseRacingML.Tests
{
    public class DistanceParsingTests
    {
        [Theory]
        [InlineData("1l", 1f)]
        [InlineData("1-1/2", 1.5f)]
        [InlineData("1/2l", 0.5f)]
        [InlineData("1¼l nk", 1.55f)]
        [InlineData("dht", 0f)]
        [InlineData("short-head", 0.1f)]
        public void ParseDistanceBeaten_ParsesCommonFormats(string input, float expected)
        {
            var parsed = HyperparameterTrainer.TestParseDistanceBeaten(input);

            Assert.NotNull(parsed);
            Assert.Equal(expected, parsed!.Value, 3);
        }

        [Fact]
        public void ParseDistanceBeaten_ReturnsNullForUnknownText()
        {
            var parsed = HyperparameterTrainer.TestParseDistanceBeaten("unknown");

            Assert.Null(parsed);
        }
    }
}