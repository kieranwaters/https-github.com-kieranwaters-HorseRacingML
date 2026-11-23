using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using HorseRacingML.Models;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Loads a persisted model and evaluates runners through
    /// <see cref="TrainedModelRuntime"/>. The runtime is also used by
    /// <see cref="HyperparameterTrainer.PredictRaceProbabilities"/>, so the
    /// live path and the evaluation path share one forward pass.
    /// </summary>
    public class AIOddsCalculator
    {
        private sealed class WeightFile
        {
            public float Bias { get; set; }
            public float[] Weights { get; set; } = Array.Empty<float>();
        }

        private readonly TrainedModelRuntime? _runtime;
        private readonly double[] _legacyWeights;
        private readonly double _legacyBias;
        private readonly string _modelStatus;

        public AIOddsCalculator(string path)
        {
            _legacyWeights = Array.Empty<double>();
            _legacyBias = 0d;

            _runtime = TrainedModelRuntime.TryLoad(path, out var status);
            _modelStatus = status;

            if (_runtime is not null)
            {
                Console.WriteLine($"[AI] {status}");
                return;
            }

            Console.Error.WriteLine($"[AI] {status}");
            if (!File.Exists(path))
            {
                return;
            }

            try
            {
                var json = File.ReadAllText(path);
                var data = JsonSerializer.Deserialize<WeightFile>(json);
                _legacyWeights = data?.Weights?.Select(w => (double)w).ToArray() ?? Array.Empty<double>();
                _legacyBias = data?.Bias ?? 0d;
                if (_legacyWeights.Length > 0)
                {
                    _modelStatus = $"Loaded legacy logistic model with {_legacyWeights.Length} coefficients.";
                }
            }
            catch (JsonException)
            {
                _legacyWeights = Array.Empty<double>();
                _legacyBias = 0d;
            }
        }

        public string ModelStatus => _modelStatus;

        public bool HasTrainedModel => _runtime is not null;

        public bool HasLegacyModel => _legacyWeights.Length > 0;

        public bool IsRaceSoftmax => _runtime?.IsRaceSoftmax == true;

        public int FeatureCount => _runtime?.FeatureCount ?? 0;

        /// <summary>
        /// Evaluates a complete field. Race-softmax models must never be scored
        /// runner-by-runner because their output is only coherent after the
        /// within-race softmax.
        /// </summary>
        public bool TryCalculateRaceOdds(IReadOnlyList<RunnerFlow> flows, out double[] probabilities)
        {
            probabilities = Array.Empty<double>();
            if (_runtime is null || !_runtime.IsRaceSoftmax || flows is null || flows.Count == 0)
            {
                return false;
            }

            if (flows.Any(flow => flow is null || !flow.HasPreparedFeatures))
            {
                return false;
            }

            var rows = flows.Select(flow => (IDictionary<string, object?>)BuildRawFeatureMap(flow)).ToList();
            if (!_runtime.TryPredictRace(rows, out probabilities) ||
                probabilities.Length != flows.Count)
            {
                probabilities = Array.Empty<double>();
                return false;
            }

            return true;
        }

        public double CalculateOdds(RunnerFlow flow)
        {
            var legacyProbability = TryCalculateLegacyProbability(flow);
            if (flow is not null)
            {
                flow.LegacyProbability = legacyProbability;
            }

            if (TryCalculateWithTrainedModel(flow, out var probability))
            {
                return probability;
            }

            if (legacyProbability.HasValue)
            {
                return legacyProbability.Value;
            }

            return CalculateLegacyOdds(flow);
        }

        private bool TryCalculateWithTrainedModel(RunnerFlow flow, out double probability)
        {
            probability = 0d;
            if (_runtime is null || flow is null || !flow.HasPreparedFeatures)
            {
                return false;
            }

            var row = BuildRawFeatureMap(flow);
            if (!_runtime.TryPredictLogit(row, out var logit, out _))
            {
                return false;
            }

            var value = Sigmoid(logit / _runtime.Model.Temperature);
            if (!double.IsFinite(value) || value < 0)
            {
                return false;
            }

            probability = value;
            return true;
        }

        private Dictionary<string, object?> BuildRawFeatureMap(RunnerFlow flow)
        {
            var raw = flow.FeatureValues != null
                ? new Dictionary<string, object?>(flow.FeatureValues, StringComparer.OrdinalIgnoreCase)
                : new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);

            var metadata = _runtime?.Model.Metadata;
            if (metadata is null)
            {
                return raw;
            }

            void EnsureFeature(string key, object? value)
            {
                if (!metadata.FeatureDimensions.ContainsKey(key))
                {
                    return;
                }

                if (!raw.ContainsKey(key))
                {
                    raw[key] = value;
                }
            }

            EnsureFeature("Draw", flow.Draw.HasValue ? (int)flow.Draw.Value : 0);
            EnsureFeature("DrawMissing", !flow.Draw.HasValue);
            EnsureFeature("SaddleclothMissing", !flow.ClothNumber.HasValue);

            if (flow.BackPrice1.HasValue) EnsureFeature("BackPrice1", flow.BackPrice1.Value);
            if (flow.BackPrice2.HasValue) EnsureFeature("BackPrice2", flow.BackPrice2.Value);
            if (flow.BackPrice3.HasValue) EnsureFeature("BackPrice3", flow.BackPrice3.Value);
            if (flow.LayPrice1.HasValue) EnsureFeature("LayPrice1", flow.LayPrice1.Value);
            if (flow.LayPrice2.HasValue) EnsureFeature("LayPrice2", flow.LayPrice2.Value);
            if (flow.LayPrice3.HasValue) EnsureFeature("LayPrice3", flow.LayPrice3.Value);

            return raw;
        }

        private double? TryCalculateLegacyProbability(RunnerFlow flow)
        {
            if (!HasLegacyModel || flow is null)
            {
                return null;
            }

            var probability = CalculateLegacyOdds(flow);
            return double.IsFinite(probability) && probability > 0 && probability <= 1
                ? probability
                : null;
        }

        private double CalculateLegacyOdds(RunnerFlow flow)
        {
            if (_legacyWeights.Length == 0 || flow is null)
            {
                return 0d;
            }

            var features = new[]
            {
                (double)(flow.BackPrice1 ?? 0m),
                flow.Draw ?? 0,
                (double)(flow.LayPrice1 ?? 0m)
            };

            var z = _legacyBias;
            for (var i = 0; i < Math.Min(features.Length, _legacyWeights.Length); i++)
            {
                z += features[i] * _legacyWeights[i];
            }

            var probability = Sigmoid(z);
            return double.IsFinite(probability) && probability >= 0 ? probability : 0d;
        }

        private static double Sigmoid(double value)
        {
            if (value >= 0)
            {
                var negative = Math.Exp(-value);
                return 1d / (1d + negative);
            }

            var positive = Math.Exp(value);
            return positive / (1d + positive);
        }
    }
}
