namespace HorseRacingML.ML
{
    /// <summary>
    /// Provides custom metric functions used throughout the project.
    /// </summary>
    public static class MetricsCalc
    {
        /// <summary>
        /// Returns the identifier for TensorFlow's binary accuracy metric.
        /// Similar to the loss wrapper, this keeps existing references while
        /// relying on the string-based API exposed by TensorFlow.NET's Keras
        /// bindings.
        /// </summary>
        public static string Bfair_Calc() => "binary_accuracy";
    }
}
