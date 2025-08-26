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
        /// Wraps TensorFlow's binary cross entropy loss.
        /// </summary>
        /// <param name="from_logits">Whether the model outputs logits.</param>
        public static ILossFunc Bfair_Crossentropy(bool from_logits = false) =>
            keras.losses.BinaryCrossentropy(from_logits: from_logits);
    }
}