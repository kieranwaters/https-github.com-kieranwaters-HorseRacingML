using System.Collections.Generic;
using HorseRacingML.Scraping;
using Xunit;

namespace HorseRacingML.Tests
{
    public class MarketOddsParserTests
    {
        [Theory]
        [InlineData("Evens", 2.0)]
        [InlineData("EVS", 2.0)]
        [InlineData("7/4", 2.75)]
        [InlineData("85/40", 3.125)]
        [InlineData("11/10F", 2.1)]
        [InlineData("11/10J", 2.1)]
        [InlineData("2.5", 2.5)]
        public void ParsesFractionalAndDecimalOdds(string token, double expected)
        {
            var parsed = MarketOddsParser.Parse(token);
            Assert.NotNull(parsed);
            Assert.Equal((decimal)expected, parsed!.Value, 3);
        }

        [Fact]
        public void ExtractsTouchedTokensIncludingEvens()
        {
            var (low, high) = MarketOddsParser.ExtractTouchedTokens("op 11/10 tchd Evens");
            Assert.Equal("Evens", low);
            Assert.Null(high);
        }

        [Fact]
        public void BuildsAHalfOpenRangeFromTheHistoricalSeries()
        {
            var history = new List<string?> { "6/1", "13/2", "7/1", "15/2", "8/1" };

            var range = MarketOddsParser.BuildRange(
                history,
                openingToken: "6/1",
                touchedLowToken: null,
                touchedHighToken: null);

            Assert.Equal("6/1", range.OpeningToken);
            Assert.Equal("6/1", range.LowToken);
            Assert.Equal("8/1", range.HighToken);
            Assert.True(range.HasRange);
            Assert.Equal(5, range.PointCount);
        }

        [Fact]
        public void ABackedHorseUsesTheTouchedShortPriceAsTheLow()
        {
            var history = new List<string?> { "4/1", "7/2", "10/3" };

            var range = MarketOddsParser.BuildRange(
                history,
                openingToken: "4/1",
                touchedLowToken: "10/3",
                touchedHighToken: null);

            Assert.Equal("10/3", range.LowToken);
            Assert.Equal("4/1", range.HighToken);
            Assert.True(range.HasRange);
        }

        [Fact]
        public void ADriftingHorseUsesTheTouchedLongPriceAsTheHigh()
        {
            var history = new List<string?>
            {
                "25/1", "28/1", "33/1", "28/1", "33/1", "40/1", "33/1"
            };

            var range = MarketOddsParser.BuildRange(
                history,
                openingToken: "25/1",
                touchedLowToken: null,
                touchedHighToken: "40/1");

            Assert.Equal("25/1", range.LowToken);
            Assert.Equal("40/1", range.HighToken);
            Assert.True(range.HasRange);
        }

        [Fact]
        public void ASingleTouchedPriceIsNotReportedAsAFullRange()
        {
            var range = MarketOddsParser.BuildRange(
                new List<string?>(),
                openingToken: null,
                touchedLowToken: "4/1",
                touchedHighToken: null);

            Assert.Equal("4/1", range.LowToken);
            Assert.Null(range.HighToken);
            Assert.False(range.HasRange);
            Assert.Equal(1, range.PointCount);
        }

        [Fact]
        public void OpeningAndTouchedPriceFormATwoPointRange()
        {
            var range = MarketOddsParser.BuildRange(
                new List<string?>(),
                openingToken: "6/1",
                touchedLowToken: "4/1",
                touchedHighToken: null);

            Assert.Equal("4/1", range.LowToken);
            Assert.Equal("6/1", range.HighToken);
            Assert.True(range.HasRange);
            Assert.Equal(2, range.PointCount);
        }
    }
}
