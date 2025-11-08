
using HorseRacingML.Models;
using Xunit;

namespace HorseRacingML.Tests
{
    public class MLParameterValidatorTests
    {
        [Theory]
        [InlineData(1)]
        [InlineData(64)]
        [InlineData(MLParameterValidator.MaxBatchSize)]
        public void EnsureBatchSize_AllowsValuesWithinRange(int batchSize)
        {
            var result = MLParameterValidator.EnsureBatchSize(batchSize, fallback: -1);

            Assert.Equal(batchSize, result);
        }

        [Theory]
        [InlineData(null)]
        [InlineData(0)]
        [InlineData(-5)]
        public void EnsureBatchSize_FallsBackWhenNotPositive(int? batchSize)
        {
            const int fallback = -1;

            var result = MLParameterValidator.EnsureBatchSize(batchSize, fallback);

            Assert.Equal(fallback, result);
        }

        [Fact]
        public void EnsureBatchSize_ClampsToMaximumWhenExceeded()
        {
            var requested = MLParameterValidator.MaxBatchSize + 1024;

            var result = MLParameterValidator.EnsureBatchSize(requested, fallback: -1);

            Assert.Equal(MLParameterValidator.MaxBatchSize, result);
        }
    }
}