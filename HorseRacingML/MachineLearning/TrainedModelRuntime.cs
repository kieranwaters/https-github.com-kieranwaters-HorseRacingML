using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Single inference implementation for both the live scraper and the
    /// trainer's evaluation helpers. Keeping the forward pass, feature
    /// encoding, temperature, market offset and blend in one place prevents
    /// the live path from drifting away from the validated path.
    /// </summary>
    public sealed class TrainedModelRuntime
    {
        private readonly TrainedModel _model;
        private readonly float[] _mean;
        private readonly float[] _std;
        private readonly int _featureCount;
        private readonly double _temperature;
        private readonly double _marketOffsetScale;
        private readonly double _blendModelWeight;
        private readonly double _blendMarketWeight;
        private readonly bool _useLogProbOffset;

        private TrainedModelRuntime(TrainedModel model)
        {
            _model = model;
            _mean = model.Normalization?.Mean ?? Array.Empty<float>();
            _std = model.Normalization?.StdDev ?? Array.Empty<float>();
            _featureCount = model.Metadata?.Keys?.Sum(key =>
                model.Metadata.FeatureDimensions.TryGetValue(key, out var dim) ? dim : 0) ?? 0;
            _temperature = model.Temperature > 0f && double.IsFinite(model.Temperature)
                ? model.Temperature
                : 1d;
            _marketOffsetScale = double.IsFinite(model.MarketOffsetScale)
                ? model.MarketOffsetScale
                : 0d;
            _blendModelWeight = double.IsFinite(model.BlendModelWeight)
                ? model.BlendModelWeight
                : 1d;
            _blendMarketWeight = double.IsFinite(model.BlendMarketWeight)
                ? model.BlendMarketWeight
                : 0d;
            _useLogProbOffset = string.Equals(
                model.MarketOffsetMode,
                "LogProb",
                StringComparison.OrdinalIgnoreCase);

            IsRaceSoftmax = string.Equals(
                model.Objective,
                "RaceSoftmax",
                StringComparison.OrdinalIgnoreCase);
        }

        public bool IsRaceSoftmax { get; }

        public int FeatureCount => _featureCount;

        public TrainedModel Model => _model;

        public static TrainedModelRuntime? TryLoad(string path, out string status)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                status = $"Weight file not found at {path}; falling back to legacy scoring.";
                return null;
            }

            try
            {
                var json = File.ReadAllText(path);
                var model = JsonSerializer.Deserialize<TrainedModel>(json);
                if (model is null)
                {
                    status = $"Weight file {Path.GetFileName(path)} did not contain a model.";
                    return null;
                }

                var runtime = FromModel(model, out status);
                if (runtime is not null)
                {
                    status = $"Loaded model {Path.GetFileName(path)} ({runtime.FeatureCount} features, " +
                             $"{model.Objective}, schema {model.FeatureSchemaHash[..Math.Min(12, model.FeatureSchemaHash.Length)]}).";
                }

                return runtime;
            }
            catch (Exception ex)
            {
                status = $"Failed to load model {Path.GetFileName(path)}: {ex.Message}";
                return null;
            }
        }

        public static TrainedModelRuntime? FromModel(TrainedModel model, out string status)
        {
            if (model is null)
            {
                status = "Model was null.";
                return null;
            }

            if (model.OutputLayer?.Weights is null || model.OutputLayer.Weights.Length == 0)
            {
                status = "Model has no output weights.";
                return null;
            }

            model.Metadata ??= new FeatureMetadata();
            model.Metadata.Keys ??= new List<string>();
            model.Metadata.FeatureDimensions ??= new Dictionary<string, int>();
            model.Metadata.StringMaps ??= new Dictionary<string, Dictionary<string, int>>();

            if (!string.IsNullOrWhiteSpace(model.FeatureSchemaHash))
            {
                var computedSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(model.Metadata);
                if (!string.Equals(computedSchemaHash, model.FeatureSchemaHash, StringComparison.OrdinalIgnoreCase))
                {
                    status = "Model feature schema hash does not match its feature metadata.";
                    return null;
                }
            }

            var runtime = new TrainedModelRuntime(model);
            if (runtime._featureCount == 0 ||
                runtime._mean.Length != runtime._featureCount ||
                runtime._std.Length != runtime._featureCount)
            {
                status = "Model normalization does not match its feature schema.";
                return null;
            }

            var unsafeFeatures = model.Metadata.Keys
                .Where(key => !HyperparameterTrainer.IsLiveAvailableFeature(key))
                .Take(10)
                .ToList();
            if (unsafeFeatures.Count > 0)
            {
                status = $"Model contains features that are not available live: {string.Join(", ", unsafeFeatures)}. Retrain the model.";
                return null;
            }

            var nonNormalizedCategories = model.Metadata.StringMaps
                .SelectMany(pair => pair.Value.Keys)
                .Where(key => key != "__unknown__" &&
                              !string.Equals(key, FeatureText.NormalizeCategory(key), StringComparison.Ordinal))
                .Take(5)
                .ToList();
            if (nonNormalizedCategories.Count > 0)
            {
                status = $"Model contains non-normalized category values: {string.Join(", ", nonNormalizedCategories)}. Retrain the model.";
                return null;
            }

            status = "Model loaded.";
            return runtime;
        }

        public bool TryPredictRace(IReadOnlyList<IDictionary<string, object?>> rows, out double[] probabilities)
        {
            probabilities = Array.Empty<double>();
            if (!IsRaceSoftmax || rows is null || rows.Count == 0)
            {
                return false;
            }

            var logits = new double[rows.Count];
            var marketProbabilities = new double[rows.Count];
            for (var i = 0; i < rows.Count; i++)
            {
                if (!TryPredictLogit(rows[i], out logits[i], out marketProbabilities[i]))
                {
                    return false;
                }
            }

            var offsets = BuildMarketOffsets(marketProbabilities, _useLogProbOffset);
            var values = new double[rows.Count];
            var max = double.NegativeInfinity;
            for (var i = 0; i < rows.Count; i++)
            {
                var value = (logits[i] + _marketOffsetScale * offsets[i]) / _temperature;
                value = _blendModelWeight * value + _blendMarketWeight * offsets[i];
                values[i] = value;
                if (value > max)
                {
                    max = value;
                }
            }

            var sum = 0d;
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = Math.Exp(values[i] - max);
                sum += values[i];
            }

            if (sum <= double.Epsilon || !double.IsFinite(sum))
            {
                return false;
            }

            probabilities = new double[values.Length];
            for (var i = 0; i < values.Length; i++)
            {
                probabilities[i] = values[i] / sum;
            }

            return true;
        }

        public bool TryPredictLogit(
            IDictionary<string, object?> row,
            out double logit,
            out double marketProbability)
        {
            logit = 0d;
            marketProbability = 0d;

            if (row is null || _model.Metadata is null)
            {
                return false;
            }

            float[] features;
            try
            {
                features = HyperparameterTrainer.EncodeFeatureVector(
                    row,
                    _model.Metadata.Keys,
                    _model.Metadata.FeatureDimensions,
                    _model.Metadata.StringMaps,
                    _featureCount);
            }
            catch
            {
                return false;
            }

            if (features.Length != _featureCount)
            {
                return false;
            }

            var normalized = Normalize(features);
            logit = ForwardLogit(normalized);
            if (!double.IsFinite(logit))
            {
                return false;
            }

            marketProbability = ExtractMarketProbability(row);
            return true;
        }

        private double[] Normalize(float[] features)
        {
            var normalized = new double[features.Length];
            for (var i = 0; i < features.Length; i++)
            {
                var value = double.IsFinite(features[i]) ? (double)features[i] : 0d;
                var std = _std[i];
                if (!double.IsFinite(std) || Math.Abs(std) < 1e-8)
                {
                    normalized[i] = 0d;
                    continue;
                }

                var mean = double.IsFinite(_mean[i]) ? _mean[i] : 0d;
                normalized[i] = (value - mean) / std;
                if (IsRaceSoftmax)
                {
                    normalized[i] = Math.Clamp(normalized[i], -5d, 5d);
                }
            }

            return normalized;
        }

        private double ForwardLogit(double[] inputs)
        {
            var activations = inputs;
            foreach (var layer in _model.HiddenLayers ?? new List<LayerWeights>())
            {
                activations = ApplyRelu(ApplyLayer(activations, layer));
            }

            var output = ApplyLayer(activations, _model.OutputLayer);
            return output.Length == 0 ? 0d : output[0];
        }

        private static double[] ApplyLayer(double[] inputs, LayerWeights layer)
        {
            if (layer?.Weights is null || layer.Weights.Length != inputs.Length)
            {
                return Array.Empty<double>();
            }

            var outputs = layer.Weights.Length == 0 ? 0 : layer.Weights[0].Length;
            var result = new double[outputs];
            for (var j = 0; j < outputs; j++)
            {
                var sum = layer.Bias is not null && layer.Bias.Length > j ? (double)layer.Bias[j] : 0d;
                for (var i = 0; i < inputs.Length; i++)
                {
                    var row = layer.Weights[i];
                    if (row is not null && row.Length > j)
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
            for (var i = 0; i < values.Length; i++)
            {
                result[i] = values[i] > 0d ? values[i] : 0d;
            }

            return result;
        }

        private static double ExtractMarketProbability(IDictionary<string, object?> row)
        {
            if (!row.TryGetValue("MarketImpliedProb", out var value) || value is null)
            {
                return 0d;
            }

            try
            {
                var probability = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                return probability > 1e-6 && probability < 1d - 1e-6 ? probability : 0d;
            }
            catch
            {
                return 0d;
            }
        }

        internal static double[] BuildMarketOffsets(double[] marketProbabilities, bool useLogProbOffset)
        {
            var offsets = new double[marketProbabilities.Length];
            if (!useLogProbOffset)
            {
                for (var i = 0; i < marketProbabilities.Length; i++)
                {
                    var probability = marketProbabilities[i];
                    offsets[i] = probability > 1e-6 && probability < 1d - 1e-6
                        ? Math.Log(probability / (1d - probability))
                        : 0d;
                }

                return offsets;
            }

            var priced = 0;
            var sum = 0d;
            for (var i = 0; i < marketProbabilities.Length; i++)
            {
                var probability = marketProbabilities[i];
                if (probability > 1e-6 && probability < 1d - 1e-6)
                {
                    offsets[i] = Math.Log(probability);
                    sum += offsets[i];
                    priced++;
                }
            }

            var mean = priced > 0 ? sum / priced : 0d;
            var allSum = 0d;
            for (var i = 0; i < offsets.Length; i++)
            {
                if (marketProbabilities[i] <= 1e-6 || marketProbabilities[i] >= 1d - 1e-6)
                {
                    offsets[i] = mean;
                }

                allSum += offsets[i];
            }

            var centre = offsets.Length > 0 ? allSum / offsets.Length : 0d;
            for (var i = 0; i < offsets.Length; i++)
            {
                offsets[i] -= centre;
            }

            return offsets;
        }
    }
}
