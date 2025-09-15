using System;
using System.IO;
using System.Text.Json;
using HorseRacingML.Models;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Loads pre-saved weights produced by <see cref="HyperparameterTrainer"/>
    /// and provides a simple probability/odds calculation used by the automated
    /// betting feature.  The weights are derived from the user-supplied
    /// hyperparameters on the custom hyperparameter page.
    /// </summary>
    public class AIOddsCalculator
    {
        private readonly float[] _weights;
        private readonly float _bias;

        private class WeightFile
        {
            public float Bias { get; set; }
            public float[] Weights { get; set; } = Array.Empty<float>();
        }

        public AIOddsCalculator(string path)
        {
            if (!File.Exists(path))
            {
                _weights = Array.Empty<float>();
                _bias = 0f;
                return;
            }

            var json = File.ReadAllText(path);
            var data = JsonSerializer.Deserialize<WeightFile>(json);
            _weights = data?.Weights ?? Array.Empty<float>();
            _bias = data?.Bias ?? 0f;
        }

        /// <summary>
        /// Calculates AI odds for a runner using a logistic model.  The features used
        /// here are deliberately simple – back price, draw and lay price – but this
        /// can be extended as the model evolves.
        /// </summary>
        public double CalculateOdds(RunnerFlow flow)
        {
            var features = new[]
            {
                (float)(flow.BackPrice1 ?? 0m),
                flow.Draw ?? 0,
                (float)(flow.LayPrice1 ?? 0m)
            };

            double z = _bias;
            for (int i = 0; i < Math.Min(features.Length, _weights.Length); i++)
            {
                z += features[i] * _weights[i];
            }

            var prob = 1.0 / (1.0 + Math.Exp(-z));
            return prob > 0 ? 1.0 / prob : double.PositiveInfinity;
        }
    }
}

