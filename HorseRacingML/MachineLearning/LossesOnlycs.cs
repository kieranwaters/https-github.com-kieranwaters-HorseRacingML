using Tensorflow.Keras.Losses;
using static Tensorflow.KerasApi;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Provides custom loss functions used throughout the project.
    /// </summary>
    public static class LossesOnly
    {
        /// <summary>
        /// Wrapper around TensorFlow's <see cref="BinaryCrossentropy"/> loss.
        /// Added to satisfy references expecting a custom loss named
        /// <c>Bfair_Crossentropy</c>.
        /// </summary>
        /// <param name="from_logits">Whether the predictions are logits.</param>
        /// <returns>A binary cross entropy loss function.</returns>
        public static ILossFunc Bfair_Crossentropy(bool from_logits = false)
            => keras.losses.BinaryCrossentropy(from_logits: from_logits);
    }
}