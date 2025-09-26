using System;

namespace HorseRacingML.Models
{
    public static class MLParameterValidator
    {
        private const int MaxUnits = 4096;
        private const int MaxLayers = 12;

        public static int EnsureUnits(int? units, int fallback)
        {
            if (units.HasValue)
            {
                var value = units.Value;
                if (value > 0 && value <= MaxUnits)
                {
                    return value;
                }
            }

            return fallback;
        }

        public static int EnsureLayers(int? layers, int fallback)
        {
            if (layers.HasValue)
            {
                var value = layers.Value;
                if (value > 0 && value <= MaxLayers)
                {
                    return value;
                }
            }

            return fallback;
        }

        public static double EnsureDropout(double? dropout, double fallback)
        {
            if (dropout.HasValue)
            {
                var value = dropout.Value;
                if (double.IsFinite(value) && value >= 0d && value < 1d)
                {
                    return value;
                }
            }

            return fallback;
        }

        public static double EnsureLearningRate(double? learningRate, double fallback)
        {
            if (learningRate.HasValue)
            {
                var value = learningRate.Value;
                if (double.IsFinite(value) && value > 0d)
                {
                    return value;
                }
            }

            return fallback;
        }

        public static int EnsurePositive(int? value, int fallback)
        {
            if (value.HasValue && value.Value > 0)
            {
                return value.Value;
            }

            return fallback;
        }
    }
}