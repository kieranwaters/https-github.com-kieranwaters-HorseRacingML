namespace HorseRacingML.ML
{
    /// <summary>
    /// Provides custom loss functions used throughout the project.
    /// </summary>
    public static class LossesOnly
    {
        /// <summary>
        /// Returns the identifier for TensorFlow's binary cross entropy loss.
        /// The Keras API in TensorFlow.NET accepts loss names as strings, so
        /// this method simply exposes the expected name while retaining the
        /// original call pattern.
        /// </summary>
        /// <param name="from_logits">Unused – retained for compatibility.</param>
        public static string Bfair_Crossentropy(bool from_logits = false) =>
            "binary_crossentropy";
    }
}