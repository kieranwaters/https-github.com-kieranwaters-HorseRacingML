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
        /// Wraps TensorFlow's binary accuracy metric.
        /// </summary>
        public static IMetricFunc Bfair_Calc() =>
            keras.metrics.BinaryAccuracy();
    }
}