using Tensorflow.Keras.Metrics;
using static Tensorflow.KerasApi;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Provides custom metric functions used throughout the project.
    /// </summary>
    public static class MetricsCalc
    {
        /// <summary>
        /// Wrapper around TensorFlow's <see cref="BinaryAccuracy"/> metric.
        /// Added to satisfy references expecting a custom metric named
        /// <c>Bfair_Calc</c>.
        /// </summary>
        /// <returns>A binary accuracy metric.</returns>
        public static IMetricFunc Bfair_Calc()
            => keras.metrics.BinaryAccuracy();
    }
}