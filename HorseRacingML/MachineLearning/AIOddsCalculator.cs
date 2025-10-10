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
        private readonly string _modelStatus;
        private HyperparameterSummary? _hyperparameters;

        private static readonly DateTime BaseDate = new DateTime(2005, 1, 1);

        private class WeightFile
        {
            public float Bias { get; set; }
            public float[] Weights { get; set; } = Array.Empty<float>();
        }
        private bool TryCalculateWithTrainedModel(RunnerFlow flow, out double probability)
        {
            probability = 0d;
            LogDebug(flow, "Attempting to calculate probability with trained model");
            if (!_hasTrainedModel)
            {
                AppendFallbackDetail(flow, "Trained model unavailable");
                LogDebug(flow, "Trained model is unavailable; will fall back to legacy odds");
                return false;
            }

            if (!flow.HasPreparedFeatures)
            {
                LogFallback(flow, "no prepared feature vector matched the runner");
                return false;
            }

            if (_metadata == null || _mean == null || _std == null ||
                _outputWeights == null || _outputBias == null)
            {
                LogFallback(flow, "trained model metadata is incomplete");
                return false;
            }

            Dictionary<string, object?> rawFeatures = BuildRawFeatureMap(flow);
            LogDebug(flow, $"Built raw feature map with {rawFeatures.Count} entries");
            if (_metadata != null)
            {
                LogDebug(flow,
                    $"Feature metadata defines {_metadata.Keys.Count} raw keys expanding to {_featureCount} encoded dimensions");
                LogMissingRawFeatures(flow, rawFeatures);
            }
            var encoded = EncodeFeatures(rawFeatures);
            if (encoded == null || encoded.Values.Length != _featureCount)
            {
                LogFallback(flow, encoded == null
                    ? "feature encoding returned null"
                    : $"feature encoding length mismatch (expected {_featureCount}, observed {encoded.Values.Length})");
                return false;
            }
            LogDebug(flow, $"Encoded feature vector length {_featureCount}");
            bool hasSignal = false;
            for (int i = 0; i < encoded.Values.Length; i++)
            {
                if (!encoded.Active[i])
                {
                    continue;
                }

                var value = encoded.Values[i];
                if (double.IsFinite(value) && Math.Abs(value) > 1e-9)
                {
                    hasSignal = true;
                    break;
                }
            }

            if (!hasSignal)
            {
                Console.WriteLine($"[AI] Encoded feature vector for {DescribeRunner(flow)} contains no usable signal; neural output will rely on bias terms.");
            }
            var normalized = new double[_featureCount];
            bool loggedNonFiniteValue = false;
            bool loggedInvalidStd = false;
            bool loggedNonFiniteMean = false;
            for (int i = 0; i < _featureCount; i++)
            {
                if (!encoded.Active[i])
                {
                    normalized[i] = 0d;
                    continue;
                }

                var value = encoded.Values[i];
                if (!double.IsFinite(value))
                {
                    value = 0d;
                    if (!loggedNonFiniteValue)
                    {
                        LogDebug(flow, $"Encountered non-finite encoded value at index {i}; substituting 0");
                        loggedNonFiniteValue = true;
                    }
                }

                var std = _std[i];
                if (!double.IsFinite(std) || Math.Abs(std) < 1e-8)
                {
                    normalized[i] = 0d;
                    if (!loggedInvalidStd)
                    {
                        LogDebug(flow, $"Standard deviation at index {i} was not usable ({std}); substituting 0");
                        loggedInvalidStd = true;
                    }
                    continue;
                }

                var mean = _mean[i];
                if (!double.IsFinite(mean))
                {
                    mean = 0d;
                    if (!loggedNonFiniteMean)
                    {
                        LogDebug(flow, $"Encountered non-finite mean at index {i}; substituting 0");
                        loggedNonFiniteMean = true;
                    }
                }

                normalized[i] = (value - mean) / std;
            }

            var activations = normalized;
            for (int i = 0; i < _hiddenWeights.Count; i++)
            {
                LogDebug(flow, $"Feeding hidden layer {i + 1} with vector length {activations.Length}");
                activations = ApplyRelu(Multiply(activations, _hiddenWeights[i], _hiddenBiases[i]));
                LogDebug(flow, $"Hidden layer {i + 1} output length {activations.Length}");
            }

            var output = Multiply(activations, _outputWeights, _outputBias);
            if (output.Length == 0)
            {
                LogFallback(flow, "forward pass produced an empty output vector");
                return false;
            }

            var logit = Math.Clamp(output[0], -700d, 700d);
            var prob = Sigmoid(logit);
            if (!double.IsFinite(prob) || prob < 0)
            {
                LogFallback(flow, "model produced a non-finite probability");
                return false;
            }

            probability = prob;
            var formattedProbability = probability.ToString("0.0000", CultureInfo.InvariantCulture);
            if (probability > 0 && probability < 1e-3)
            {
                var scientific = probability.ToString("0.###E+0", CultureInfo.InvariantCulture);
                formattedProbability = $"{formattedProbability} (~{scientific})";
            }

            LogDebug(flow, $"Calculated probability {formattedProbability}");
            return true;
        }
        private void LogMissingRawFeatures(RunnerFlow flow, IReadOnlyDictionary<string, object?> raw)
        {
            if (flow == null || _metadata == null || raw == null)
            {
                return;
            }

            List<string>? missing = null;
            foreach (var key in _metadata.Keys)
            {
                if (raw.TryGetValue(key, out var value) && value != null)
                {
                    continue;
                }

                missing ??= new List<string>();
                missing.Add(key);
            }

            if (missing == null || missing.Count == 0)
            {
                return;
            }

            const int maxToShow = 20;
            var preview = string.Join(", ", missing
                .Take(maxToShow)
                .Select(key =>
                {
                    var defaults = DescribeDefaultEncoding(key);
                    return string.IsNullOrEmpty(defaults) ? key : $"{key} ({defaults})";
                }));
            if (missing.Count > maxToShow)
            {
                preview += $", … (+{missing.Count - maxToShow} more)";
            }

            LogDebug(flow,
                $"Missing raw feature values for {missing.Count} feature keys: {preview}. Those encoded dimensions will be suppressed");
        }

        // Feature metadata is generated from the training pipeline and includes
        // both the raw feature keys and how many encoded dimensions each key
        // expands into (one-hot buckets, missing-value indicators, etc). During
        // live scoring we might not have values for every key that appeared in
        // training—especially for historic-only fields such as class, draw bias
        // or derived pace figures. Historically the scorer mirrored the
        // training-time convention of setting a dedicated "missing" indicator to
        // 1 (or routing into the "__unknown__" one-hot bucket) whenever a raw
        // value was unavailable. The current behaviour instead suppresses those
        // encoded dimensions entirely so that downstream layers ignore the
        // feature; all entries stay at 0 and are marked inactive so the
        // normaliser and neural network weights do not use them.


        private string DescribeDefaultEncoding(string key)
        {
            if (_metadata == null || !_metadata.FeatureDimensions.TryGetValue(key, out var dim) || dim <= 0)
            {
                return string.Empty;
            }

            int baseDim = 1;
            Dictionary<string, int>? map = null;
            if (_metadata.StringMaps.TryGetValue(key, out var existingMap) && existingMap.Count > 0)
            {
                map = existingMap;
                baseDim = Math.Max(1, existingMap.Count);
            }

            if (map != null)
            {
                var hasUnknown = map.TryGetValue("__unknown__", out var unknownIndex) && unknownIndex >= 0 && unknownIndex < dim;
                return hasUnknown
                    ? $"raw null -> feature omitted; no one-hot bucket selected and missing indicator remains 0"
                    : "raw null -> feature omitted; all encoded dimensions forced to 0";
            }

            return "raw null -> feature omitted; all encoded dimensions forced to 0";
        }
        private static void LogDebug(RunnerFlow flow, string message)
        {
            if (flow == null)
            {
                Console.WriteLine($"[AI][debug] {message} (runner flow was null).");
                return;
            }

            var runnerId = DescribeRunner(flow);
            var marketId = string.IsNullOrWhiteSpace(flow.MarketId) ? string.Empty : $" in market {flow.MarketId}";
            Console.WriteLine($"[AI][debug] {message} for {runnerId}{marketId}.");
        }

        private static double Sigmoid(double logit)
        {
            if (logit >= 0)
            {
                var neg = Math.Exp(-logit);
                return 1d / (1d + neg);
            }

            var pos = Math.Exp(logit);
            return pos / (1d + pos);
        }
        public double CalculateOdds(RunnerFlow flow)
        {
            var legacyProbability = TryCalculateLegacyProbability(flow);
            if (flow != null)
            {
                flow.LegacyProbability = legacyProbability;
            }

            if (TryCalculateWithTrainedModel(flow, out var probability))
            {
                if (flow != null)
                {
                    flow.AiProbabilityMarketDerived = false;
                    flow.AiProbabilityFallbackReason = null;
                }
                return probability;
            }

            if (legacyProbability.HasValue)
            {
                if (_hasTrainedModel)
                {
                    LogFallback(flow, "falling back to legacy odds");
                }
                if (flow != null)
                {
                    flow.AiProbabilityMarketDerived = true;
                    AppendFallbackDetail(flow, "Using legacy logistic regression probability");
                }
                return legacyProbability.Value;
            }

            var fallback = CalculateLegacyOdds(flow);
            if (_hasTrainedModel)
            {
                LogFallback(flow, "falling back to legacy odds");
            }
            if (flow != null)
            {
                flow.AiProbabilityMarketDerived = true;
                AppendFallbackDetail(flow, "Using legacy logistic regression probability");
            }
            return fallback;
        }
        private double? TryCalculateLegacyProbability(RunnerFlow flow)
        {
            if (!HasLegacyModel || flow == null)
            {
                return null;
            }

            var probability = CalculateLegacyOdds(flow);
            if (!double.IsFinite(probability) || probability <= 0 || probability > 1)
            {
                return null;
            }

            return probability;
        }

        private static void LogFallback(RunnerFlow flow, string reason)
        {
            if (flow == null)
            {
                return;
            }
            AppendFallbackDetail(flow, reason);
            var runnerId = DescribeRunner(flow);
            var marketId = string.IsNullOrWhiteSpace(flow.MarketId) ? "<unknown>" : flow.MarketId;
            Console.WriteLine($"[AI] Unable to use trained model for {runnerId} in market {marketId}: {reason}.");
        }
        private static void AppendFallbackDetail(RunnerFlow? flow, string detail)
        {
            if (flow == null || string.IsNullOrWhiteSpace(detail))
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(flow.AiProbabilityFallbackReason))
            {
                flow.AiProbabilityFallbackReason = detail;
                return;
            }

            if (flow.AiProbabilityFallbackReason.IndexOf(detail, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return;
            }

            flow.AiProbabilityFallbackReason = $"{flow.AiProbabilityFallbackReason}; {detail}";
        }
        private static string DescribeRunner(RunnerFlow flow)
        {
            var parts = new List<string>();
            if (!string.IsNullOrWhiteSpace(flow.HorseName))
            {
                parts.Add(flow.HorseName!.Trim());
            }
            else if (!string.IsNullOrWhiteSpace(flow.SelectionId))
            {
                parts.Add(flow.SelectionId!.Trim());
            }
            else
            {
                parts.Add("unknown runner");
            }

            if (flow.ClothNumber.HasValue)
            {
                parts.Add($"cloth {flow.ClothNumber.Value}");
            }

            if (flow.Draw.HasValue)
            {
                parts.Add($"draw {flow.Draw.Value}");
            }

            return string.Join(", ", parts);
        }
        public AIOddsCalculator(string path)
        {
            _legacyWeights = Array.Empty<double>();
            _legacyBias = 0d;
            _modelStatus = "AI model not initialized.";
            if (!File.Exists(path))
            {
                Console.Error.WriteLine($"[AI] Weight file not found at {path}; falling back to legacy logistic model.");
                _modelStatus = $"Weight file not found at {path}; defaulting to zero-probability outputs.";
                return;
            }

            var json = File.ReadAllText(path);
            if (TryLoadTrainedModel(json))
            {
                _hasTrainedModel = true;
                Console.WriteLine($"[AI] Loaded trained model with {_featureCount} features from {Path.GetFileName(path)}.");
                _modelStatus = $"Loaded trained model with {_featureCount} features from {Path.GetFileName(path)}.";
                return;
            }

            try
            {
                var data = JsonSerializer.Deserialize<WeightFile>(json);
                _legacyWeights = data?.Weights?.Select(w => (double)w).ToArray() ?? Array.Empty<double>();
                _legacyBias = data?.Bias ?? 0d;
                if (_legacyWeights.Length == 0)
                {
                    Console.Error.WriteLine($"[AI] Legacy weight file {Path.GetFileName(path)} did not contain any usable coefficients; probabilities will default to zero.");
                    _modelStatus = $"Legacy weight file {Path.GetFileName(path)} was empty; probabilities will default to zero.";
                }
                else
                {
                    Console.WriteLine($"[AI] Loaded legacy logistic weights ({_legacyWeights.Length}) from {Path.GetFileName(path)}.");
                    _modelStatus = $"Loaded legacy logistic model with {_legacyWeights.Length} coefficients from {Path.GetFileName(path)}.";
                }
            }
            catch (JsonException)
            {
                _legacyWeights = Array.Empty<double>();
                _legacyBias = 0d;
                Console.Error.WriteLine($"[AI] Failed to parse weight file {Path.GetFileName(path)}; probabilities will default to zero.");
                _modelStatus = $"Failed to parse weight file {Path.GetFileName(path)}; probabilities will default to zero.";
            }
        }
        private sealed class EncodedVector
        {
            public EncodedVector(double[] values, bool[] active)
            {
                Values = values;
                Active = active;
            }

            public double[] Values { get; }
            public bool[] Active { get; }
        }
        public string ModelStatus => _modelStatus;

        public bool HasTrainedModel => _hasTrainedModel;
        public HyperparameterSummary? Hyperparameters => _hyperparameters;
        public bool HasLegacyModel => _legacyWeights.Length > 0;
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

            EnsureFeature("Draw", flow.Draw);
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
            _hyperparameters = model.Hyperparameters;
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

        private EncodedVector? EncodeFeatures(Dictionary<string, object?> raw)
        {
            if (_metadata == null)
            {
                return null;
            }

            var vector = new double[_featureCount];
            var active = new bool[_featureCount];
            int offset = 0;
            foreach (var key in _metadata.Keys)
            {
                if (!_metadata.FeatureDimensions.TryGetValue(key, out var dim) || dim <= 0)
                {
                    continue;
                }

                raw.TryGetValue(key, out var value);
                var encoded = EncodeFeature(key, value, dim, out var isPresent);
                for (int i = 0; i < dim; i++)
                {
                    vector[offset + i] = encoded[i];
                    active[offset + i] = isPresent;
                }

                offset += dim;
            }

            return new EncodedVector(vector, active);
        }

        private double[] EncodeFeature(string key, object? value, int dim, out bool isPresent)
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
                isPresent = false;
                return arr;
            }

            isPresent = true;

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