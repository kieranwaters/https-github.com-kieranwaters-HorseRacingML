using System;
using System.Reflection;
using Xunit;

namespace HorseRacingML.Tests
{
    public class BackfillStringMeaningfulnessTests
    {
        [Theory]
        [InlineData("Unknown")]
        [InlineData("unknown")]
        [InlineData(" Missing ")]
        [InlineData("N/A")]
        [InlineData("-")]
        public void IsMeaningfulStringValue_TreatsPlaceholderValuesAsNotMeaningful(string input)
        {
            var method = typeof(HorseRacingML.ML.HyperparameterTrainer)
                .GetMethod("IsMeaningfulStringValue", BindingFlags.NonPublic | BindingFlags.Static);

            Assert.NotNull(method);

            var result = (bool?)method!.Invoke(null, new object?[] { input });

            Assert.False(result.GetValueOrDefault(true));
        }

        [Theory]
        [InlineData("Handicap")]
        [InlineData("Soft")]
        [InlineData("Turf")]
        public void IsMeaningfulStringValue_AllowsValidValues(string input)
        {
            var method = typeof(HorseRacingML.ML.HyperparameterTrainer)
                .GetMethod("IsMeaningfulStringValue", BindingFlags.NonPublic | BindingFlags.Static);

            Assert.NotNull(method);

            var result = (bool?)method!.Invoke(null, new object?[] { input });

            Assert.True(result.GetValueOrDefault(false));
        }
    }
}