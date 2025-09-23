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
    /// Loads the trained model produced by <see cref="HyperparameterTrainer"/> and
    /// evaluates runners using the same feature space that was used during
    /// training. When a full model cannot be loaded, the calculator falls back to
    /// the legacy logistic regression that considered only a handful of market
    /// prices.
    /// </summary>
    public class AIOddsCalculator
    {
        private readonly List<double[][]> _hiddenWeights = new();
        private readonly List<double[]> _hiddenBiases = new();
        private double[][]? _outputWeights;
        private double[]? _outputBias;
        private FeatureMetadata? _metadata;
        private double[]? _mean;
        private double[]? _std;
        private int _featureCount;
        private readonly bool _hasTrainedModel;
        private readonly double[] _legacyWeights;
        private readonly double _legacyBias;

        private static readonly DateTime BaseDate = new DateTime(2005, 1, 1);

        private class WeightFile
        {
            public float Bias { get; set; }
            public float[] Weights { get; set; } = Array.Empty<float>();
        }
        private Dictionary<string, object?> BuildRawFeatureMap(RunnerFlow flow)
        {
            var raw = flow.FeatureValues != null
                ? new Dictionary<string, object?>(flow.FeatureValues, StringComparer.OrdinalIgnoreCase)
                : new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);

            if (_metadata == null)
            {
                return raw;
            }

            void EnsureFeature(string key, object? value)
            {
                if (!_metadata.FeatureDimensions.ContainsKey(key))
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

            if (flow.BackPrice1.HasValue)
            {
                EnsureFeature("BackPrice1", flow.BackPrice1.Value);
            }

            if (flow.BackPrice2.HasValue)
            {
                EnsureFeature("BackPrice2", flow.BackPrice2.Value);
            }

            if (flow.BackPrice3.HasValue)
            {
                EnsureFeature("BackPrice3", flow.BackPrice3.Value);
            }

            if (flow.LayPrice1.HasValue)
            {
                EnsureFeature("LayPrice1", flow.LayPrice1.Value);
            }

            if (flow.LayPrice2.HasValue)
            {
                EnsureFeature("LayPrice2", flow.LayPrice2.Value);
            }

            if (flow.LayPrice3.HasValue)
            {
                EnsureFeature("LayPrice3", flow.LayPrice3.Value);
            }

            return raw;
        }
        public AIOddsCalculator(string path)
        {
            _legacyWeights = Array.Empty<double>();
            _legacyBias = 0d;
            if (!File.Exists(path))
            {
                return;
            }

            var json = File.ReadAllText(path);
            if (TryLoadTrainedModel(json))
            {
                _hasTrainedModel = true;
                return;
            }

            try
            {
                var data = JsonSerializer.Deserialize<WeightFile>(json);
                _legacyWeights = data?.Weights?.Select(w => (double)w).ToArray() ?? Array.Empty<double>();
                _legacyBias = data?.Bias ?? 0d;
            }
            catch (JsonException)
            {
                _legacyWeights = Array.Empty<double>();
                _legacyBias = 0d;
            }
        }

        public double CalculateOdds(RunnerFlow flow)
        {
            if (TryCalculateWithTrainedModel(flow, out var probability))
            {
                return probability;
            }

            return CalculateLegacyOdds(flow);
        }

        private bool TryLoadTrainedModel(string json)
        {
            TrainedModel? model;
            try
            {
                model = JsonSerializer.Deserialize<TrainedModel>(json);
            }
            catch (JsonException)
            {
                return false;
            }

            if (model == null || model.OutputLayer?.Weights == null || model.OutputLayer.Weights.Length == 0)
            {
                return false;
            }

            model.Metadata ??= new FeatureMetadata();
            model.Metadata.Keys ??= new List<string>();
            model.Metadata.FeatureDimensions ??= new Dictionary<string, int>();
            model.Metadata.StringMaps ??= new Dictionary<string, Dictionary<string, int>>();

            _metadata = model.Metadata;
            _mean = model.Normalization?.Mean?.Select(f => (double)f).ToArray();
            _std = model.Normalization?.StdDev?.Select(f => (double)f).ToArray();
            if (_metadata.Keys.Count == 0 || _mean == null || _std == null)
            {
                return false;
            }

            foreach (var layer in model.HiddenLayers ?? new List<LayerWeights>())
            {
                _hiddenWeights.Add(ToDoubleJagged(layer.Weights));
                _hiddenBiases.Add(ToDoubleArray(layer.Bias));
            }

            _outputWeights = ToDoubleJagged(model.OutputLayer.Weights);
            _outputBias = ToDoubleArray(model.OutputLayer.Bias);

            _featureCount = _metadata.Keys.Sum(key =>
                _metadata.FeatureDimensions.TryGetValue(key, out var dim) ? dim : 0);
            if (_featureCount == 0 || _featureCount != _mean.Length || _featureCount != _std.Length)
            {
                return false;
            }

            return true;
        }

        private bool TryCalculateWithTrainedModel(RunnerFlow flow, out double probability)
        {
            probability = 0d;
            if (!_hasTrainedModel || _metadata == null || _mean == null || _std == null ||
                _outputWeights == null || _outputBias == null)
            {
                return false;
            }

            Dictionary<string, object?> rawFeatures = BuildRawFeatureMap(flow);
            var encoded = EncodeFeatures(rawFeatures);
            if (encoded == null || encoded.Length != _featureCount)
            {
                return false;
            }

            var normalized = new double[_featureCount];
            for (int i = 0; i < _featureCount; i++)
            {
                var std = _std[i];
                if (Math.Abs(std) < 1e-8)
                {
                    return false;
                }
                normalized[i] = (encoded[i] - _mean[i]) / std;
            }

            var activations = normalized;
            for (int i = 0; i < _hiddenWeights.Count; i++)
            {
                activations = ApplyRelu(Multiply(activations, _hiddenWeights[i], _hiddenBiases[i]));
            }

            var output = Multiply(activations, _outputWeights, _outputBias);
            if (output.Length == 0)
            {
                return false;
            }

            var logit = output[0];
            var prob = 1.0 / (1.0 + Math.Exp(-logit));
            if (!double.IsFinite(prob) || prob < 0)
            {
                return false;
            }

            probability = prob;
            return true;
        }

        private double[]? EncodeFeatures(Dictionary<string, object?> raw)
        {
            if (_metadata == null)
            {
                return null;
            }

            var vector = new double[_featureCount];
            int offset = 0;
            foreach (var key in _metadata.Keys)
            {
                if (!_metadata.FeatureDimensions.TryGetValue(key, out var dim) || dim <= 0)
                {
                    continue;
                }

                raw.TryGetValue(key, out var value);
                var encoded = EncodeFeature(key, value, dim);
                for (int i = 0; i < dim; i++)
                {
                    vector[offset + i] = encoded[i];
                }

                offset += dim;
            }

            return vector;
        }

        private double[] EncodeFeature(string key, object? value, int dim)
        {
            int baseDim = 1;
            if (_metadata != null && _metadata.StringMaps.TryGetValue(key, out var map))
            {
                baseDim = map.Count;
            }

            bool hasMissingIndicator = dim > baseDim;
            if (value == null)
            {
                var arr = new double[dim];
                if (hasMissingIndicator && baseDim < dim)
                {
                    arr[baseDim] = 1d;
                }
                return arr;
            }

            double[] encoded;
            switch (value)
            {
                case DateTime dt:
                    encoded = new[] { (dt - BaseDate).TotalDays };
                    break;
                case TimeSpan ts:
                    encoded = new[] { ts.TotalSeconds };
                    break;
                case string s:
                    encoded = EncodeString(key, s, baseDim);
                    break;
                case bool b:
                    encoded = new[] { b ? 1d : 0d };
                    break;
                default:
                    encoded = EncodeNumeric(value, baseDim);
                    break;
            }

            if (!hasMissingIndicator)
            {
                return encoded.Length == dim ? encoded : Pad(encoded, dim);
            }

            var result = new double[dim];
            Array.Copy(encoded, result, Math.Min(baseDim, encoded.Length));
            return result;
        }

        private double[] EncodeString(string key, string value, int dim)
        {
            if (_metadata != null && _metadata.StringMaps.TryGetValue(key, out var map))
            {
                if (!map.TryGetValue(value, out var index) && map.TryGetValue("__unknown__", out var unknown))
                {
                    index = unknown;
                }

                var vec = new double[dim];
                if (index >= 0 && index < dim)
                {
                    vec[index] = 1d;
                }
                return vec;
            }

            var fallback = new double[dim];
            if (dim > 0)
            {
                fallback[0] = 1d;
            }
            return fallback;
        }

        private static double[] EncodeNumeric(object value, int dim)
        {
            double number = value switch
            {
                double d => d,
                float f => f,
                decimal m => (double)m,
                int i => i,
                long l => l,
                short s => s,
                byte b => b,
                IConvertible convertible => convertible.ToDouble(CultureInfo.InvariantCulture),
                _ => System.Convert.ToDouble(value, CultureInfo.InvariantCulture)
            };

            if (Math.Abs(number) > 1_000_000d)
            {
                number /= 1_000_000d;
            }

            var arr = new double[dim];
            if (dim > 0)
            {
                arr[0] = number;
            }
            return arr;
        }

        private static double[] Pad(double[] source, int dim)
        {
            if (source.Length == dim)
            {
                return source;
            }

            var result = new double[dim];
            Array.Copy(source, result, Math.Min(source.Length, dim));
            return result;
        }

        private static double[] Multiply(double[] inputs, double[][] weights, double[] bias)
        {
            if (weights.Length != inputs.Length)
            {
                throw new InvalidOperationException("Weight matrix input dimension mismatch.");
            }

            int outputs = weights.Length == 0 ? 0 : weights[0].Length;
            var result = new double[outputs];
            for (int j = 0; j < outputs; j++)
            {
                double sum = bias.Length > j ? bias[j] : 0d;
                for (int i = 0; i < inputs.Length; i++)
                {
                    var row = weights[i];
                    if (row.Length > j)
                    {
                        sum += inputs[i] * row[j];
                    }
                }
                result[j] = sum;
            }

            return result;
        }

        private static double[] ApplyRelu(double[] values)
        {
            var result = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                result[i] = values[i] > 0 ? values[i] : 0d;
            }
            return result;
        }

        private double CalculateLegacyOdds(RunnerFlow flow)
        {
            if (_legacyWeights.Length == 0)
            {
                return 0d;
            }

            var features = new[]
            {
                (double)(flow.BackPrice1 ?? 0m),
                flow.Draw ?? 0,
                (double)(flow.LayPrice1 ?? 0m)
            };

            double z = _legacyBias;
            for (int i = 0; i < Math.Min(features.Length, _legacyWeights.Length); i++)
            {
                z += features[i] * _legacyWeights[i];
            }

            var prob = 1.0 / (1.0 + Math.Exp(-z));
            if (!double.IsFinite(prob) || prob < 0)
            {
                return 0d;
            }

            return prob;
        }

        private static double[][] ToDoubleJagged(float[][] source)
        {
            if (source == null || source.Length == 0)
            {
                return Array.Empty<double[]>();
            }

            var result = new double[source.Length][];
            for (int i = 0; i < source.Length; i++)
            {
                var row = source[i] ?? Array.Empty<float>();
                result[i] = row.Select(f => (double)f).ToArray();
            }
            return result;
        }

        private static double[] ToDoubleArray(float[] source)
        {
            return source?.Select(f => (double)f).ToArray() ?? Array.Empty<double>();
        }
    }
}