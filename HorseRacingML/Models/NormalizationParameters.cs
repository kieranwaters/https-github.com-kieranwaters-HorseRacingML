using System;

namespace HorseRacingML.Models
{
    /// <summary>
    /// Parameters used for feature normalization.
    /// </summary>
    public class NormalizationParameters
    {
        public float[] Mean { get; set; } = Array.Empty<float>();
        public float[] StdDev { get; set; } = Array.Empty<float>();
    }
}