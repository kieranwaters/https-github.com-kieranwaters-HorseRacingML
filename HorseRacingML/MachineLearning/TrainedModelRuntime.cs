using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;

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
        private const double MinimumProbability = 1e-12;

        private readonly TrainedModel _model;
        private readonly IReadOnlyList<TrainedModelRuntime> _members;
        private readonly float[] _mean;
        private readonly float[] _std;
        private readonly int _featureCount;
        private readonly double _temperature;
        private readonly double _marketOffsetScale;
        private readonly double _blendModelWeight;
        private readonly double _blendMarketWeight;
        private readonly bool _useLogProbOffset;
        private readonly bool[] _marketFeatureMask;
        private readonly MLContext? _lightGbmContext;
        private readonly ITransformer? _lightGbmModel;
        private readonly IReadOnlyList<FieldCalibrationBucket> _fieldCalibration;

        private TrainedModelRuntime(TrainedModel model)
            : this(model, Array.Empty<TrainedModelRuntime>())
        {
        }

        private TrainedModelRuntime(
            TrainedModel model,
            IReadOnlyList<TrainedModelRuntime> members,
            MLContext? lightGbmContext = null,
            ITransformer? lightGbmModel = null)
        {
            _model = model;
            _members = members ?? Array.Empty<TrainedModelRuntime>();
            _lightGbmContext = lightGbmContext;
            _lightGbmModel = lightGbmModel;
            _fieldCalibration = (model.FieldCalibration ?? new List<FieldCalibrationBucket>())
                .OrderBy(bucket => bucket.MaxFieldSize)
                .ToList();
            _mean = model.Normalization?.Mean ?? Array.Empty<float>();
            _std = model.Normalization?.StdDev ?? Array.Empty<float>();
            _featureCount = model.Metadata?.Keys?.Sum(key =>
                model.Metadata.FeatureDimensions.TryGetValue(key, out var dim) ? dim : 0) ?? 0;
            if (_featureCount == 0 && _members.Count > 0)
            {
                _featureCount = _members[0].FeatureCount;
            }
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

            _marketFeatureMask = Array.Empty<bool>();
            if (model.MarketBlind && model.Metadata is not null)
            {
                _marketFeatureMask = new bool[_featureCount];
                int offset = 0;
                foreach (var key in model.Metadata.Keys)
                {
                    int dimension = model.Metadata.FeatureDimensions.TryGetValue(key, out var dim)
                        ? dim
                        : 0;
                    if (HyperparameterTrainer.IsMarketFeatureKey(key))
                    {
                        for (int j = 0; j < dimension && offset + j < _featureCount; j++)
                        {
                            _marketFeatureMask[offset + j] = true;
                        }
                    }

                    offset += dimension;
                }
            }

            IsRaceSoftmax = string.Equals(
                model.Objective,
                "RaceSoftmax",
                StringComparison.OrdinalIgnoreCase);
        }

        public bool IsRaceSoftmax { get; }

        public bool IsEnsemble => _members.Count > 0;

        public int MemberCount => _members.Count > 0 ? _members.Count : 1;

        public int FeatureCount => _featureCount;

        public TrainedModel Model => _model;

        /// <summary>
        /// True when an encoded dataset uses exactly the same feature keys and
        /// dimensions as this artifact (or every ensemble member does).
        /// </summary>
        public bool MatchesFeatureSchema(
            IReadOnlyList<string> featureKeys,
            IDictionary<string, int> featureDimensions)
        {
            if (featureKeys is null || featureDimensions is null)
            {
                return false;
            }

            if (_members.Count > 0)
            {
                return _members.All(member => member.MatchesFeatureSchema(featureKeys, featureDimensions));
            }

            if (_model.Metadata is null)
            {
                return false;
            }

            var modelKeys = _model.Metadata.Keys ?? new List<string>();
            if (modelKeys.Count != featureKeys.Count)
            {
                return false;
            }

            for (int i = 0; i < modelKeys.Count; i++)
            {
                if (!string.Equals(modelKeys[i], featureKeys[i], StringComparison.Ordinal))
                {
                    return false;
                }

                var modelDimension = _model.Metadata.FeatureDimensions.TryGetValue(modelKeys[i], out var dim)
                    ? dim
                    : 0;
                var datasetDimension = featureDimensions.TryGetValue(featureKeys[i], out var datasetDim)
                    ? datasetDim
                    : 0;
                if (modelDimension != datasetDimension)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Scores a race whose runner feature vectors were already encoded in
        /// the same schema as this artifact. Used by the simulator when the
        /// window is after the model's training cutoff.
        /// </summary>
        public bool TryPredictEncodedRace(
            IReadOnlyList<HyperparameterTrainer.RunnerExample> runners,
            out double[] probabilities)
        {
            probabilities = Array.Empty<double>();
            if (!IsRaceSoftmax || runners is null || runners.Count == 0)
            {
                return false;
            }

            if (_lightGbmModel is not null)
            {
                if (!TryPredictLightGbmEncoded(runners, out probabilities))
                {
                    return false;
                }

                ApplyFieldCalibration(probabilities, runners.Count);
                return true;
            }

            if (_members.Count > 0)
            {
                var logSums = new double[runners.Count];
                int membersUsed = 0;
                foreach (var member in _members)
                {
                    if (!member.TryPredictEncodedRace(runners, out var memberProbabilities) ||
                        memberProbabilities.Length != runners.Count)
                    {
                        return false;
                    }

                    for (int i = 0; i < runners.Count; i++)
                    {
                        double probability = memberProbabilities[i];
                        if (!double.IsFinite(probability))
                        {
                            return false;
                        }

                        logSums[i] += Math.Log(Math.Max(probability, MinimumProbability));
                    }

                    membersUsed++;
                }

                if (membersUsed == 0)
                {
                    return false;
                }

                double total = 0d;
                var combined = new double[runners.Count];
                for (int i = 0; i < runners.Count; i++)
                {
                    combined[i] = Math.Exp(logSums[i] / membersUsed);
                    total += combined[i];
                }

                if (total <= double.Epsilon || !double.IsFinite(total))
                {
                    return false;
                }

                probabilities = new double[runners.Count];
                for (int i = 0; i < runners.Count; i++)
                {
                    probabilities[i] = combined[i] / total;
                }

                ApplyFieldCalibration(probabilities, runners.Count);
                return true;
            }

            var logits = new double[runners.Count];
            var marketProbabilities = new double[runners.Count];
            for (int i = 0; i < runners.Count; i++)
            {
                var features = runners[i]?.Features;
                if (features is null || features.Length != _featureCount)
                {
                    return false;
                }

                var normalized = Normalize(features);
                logits[i] = ForwardLogit(normalized);
                if (!double.IsFinite(logits[i]))
                {
                    return false;
                }

                marketProbabilities[i] = runners[i].MarketProbability;
            }

            var offsets = BuildMarketOffsets(marketProbabilities, _useLogProbOffset);
            var values = new double[runners.Count];
            var max = double.NegativeInfinity;
            for (int i = 0; i < runners.Count; i++)
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
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Math.Exp(values[i] - max);
                sum += values[i];
            }

            if (sum <= double.Epsilon || !double.IsFinite(sum))
            {
                return false;
            }

            probabilities = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                probabilities[i] = values[i] / sum;
            }

            ApplyFieldCalibration(probabilities, runners.Count);
            return true;
        }

        private void ApplyFieldCalibration(double[] probabilities, int fieldSize)
        {
            if (_fieldCalibration.Count == 0 ||
                probabilities.Length == 0 ||
                fieldSize < 2)
            {
                return;
            }

            FieldCalibrationBucket? bucket = null;
            foreach (var candidate in _fieldCalibration)
            {
                if (fieldSize <= candidate.MaxFieldSize)
                {
                    bucket = candidate;
                    break;
                }
            }

            bucket ??= _fieldCalibration[^1];
            if (bucket is null ||
                (Math.Abs(bucket.Intercept) < 1e-12 && Math.Abs(bucket.Slope - 1d) < 1e-12))
            {
                return;
            }

            double sum = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                double p = Math.Clamp(probabilities[i], 1e-6, 1d - 1e-6);
                double logit = Math.Log(p / (1d - p));
                double calibratedLogit = bucket.Intercept + bucket.Slope * logit;
                double calibrated = 1d / (1d + Math.Exp(-calibratedLogit));
                probabilities[i] = Math.Clamp(calibrated, 1e-9, 1d - 1e-9);
                sum += probabilities[i];
            }

            if (sum <= double.Epsilon || !double.IsFinite(sum))
            {
                return;
            }

            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] /= sum;
            }
        }

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

                var runtime = FromModel(model, Path.GetDirectoryName(path), out status);
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
            return FromModel(model, null, out status);
        }

        public static TrainedModelRuntime? FromModel(
            TrainedModel model,
            string? modelDirectory,
            out string status)
        {
            if (model is null)
            {
                status = "Model was null.";
                return null;
            }

            model.Members ??= new List<TrainedModel>();
            if (model.Members.Count > 0)
            {
                var members = new List<TrainedModelRuntime>(model.Members.Count);
                foreach (var member in model.Members)
                {
                    var memberRuntime = FromModel(member, modelDirectory, out var memberStatus);
                    if (memberRuntime is null)
                    {
                        status = $"Ensemble member is invalid: {memberStatus}";
                        return null;
                    }

                    if (!memberRuntime.IsRaceSoftmax)
                    {
                        status = "Ensemble members must all use the RaceSoftmax objective.";
                        return null;
                    }

                    // Members may deliberately use different feature schemas
                    // (for example a market-blind member in a market-aware
                    // ensemble). Each member encodes its own feature vector.
                    members.Add(memberRuntime);
                }

                if (model.Metadata is null || model.Metadata.Keys.Count == 0)
                {
                    model.Metadata = members[0].Model.Metadata;
                }

                if (model.Normalization is null ||
                    model.Normalization.Mean.Length == 0 ||
                    model.Normalization.StdDev.Length == 0)
                {
                    model.Normalization = members[0].Model.Normalization;
                }

                if (!string.IsNullOrWhiteSpace(model.FeatureSchemaHash))
                {
                    var computedSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(model.Metadata);
                    if (!string.Equals(computedSchemaHash, model.FeatureSchemaHash, StringComparison.OrdinalIgnoreCase))
                    {
                        status = "Ensemble feature schema hash does not match its feature metadata.";
                        return null;
                    }
                }

                var ensemble = new TrainedModelRuntime(model, members);
                status = $"Loaded RaceSoftmax ensemble with {members.Count} member(s).";
                return ensemble;
            }

            if (!string.IsNullOrWhiteSpace(model.LightGbmModelFile))
            {
                model.Metadata ??= new FeatureMetadata();
                model.Metadata.Keys ??= new List<string>();
                model.Metadata.FeatureDimensions ??= new Dictionary<string, int>();
                model.Metadata.StringMaps ??= new Dictionary<string, Dictionary<string, int>>();
                model.Objective = "RaceSoftmax";

                if (model.Metadata.Keys.Count == 0)
                {
                    status = "LightGBM member has no feature metadata.";
                    return null;
                }

                if (!string.IsNullOrWhiteSpace(model.FeatureSchemaHash))
                {
                    var computedSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(model.Metadata);
                    if (!string.Equals(computedSchemaHash, model.FeatureSchemaHash, StringComparison.OrdinalIgnoreCase))
                    {
                        status = "LightGBM member feature schema hash does not match its metadata.";
                        return null;
                    }
                }

                var unsafeLightGbmFeatures = model.Metadata.Keys
                    .Where(key => !HyperparameterTrainer.IsLiveAvailableFeature(key))
                    .Take(10)
                    .ToList();
                if (unsafeLightGbmFeatures.Count > 0)
                {
                    status = $"LightGBM member contains features that are not available live: {string.Join(", ", unsafeLightGbmFeatures)}.";
                    return null;
                }

                string path = Path.IsPathRooted(model.LightGbmModelFile)
                    ? model.LightGbmModelFile
                    : Path.Combine(modelDirectory ?? string.Empty, model.LightGbmModelFile);
                if (!File.Exists(path))
                {
                    status = $"LightGBM member file not found at {path}.";
                    return null;
                }

                var mlContext = new MLContext(seed: 42);
                var lightGbmModel = LightGbmRankingModel.TryLoad(mlContext, path);
                if (lightGbmModel is null)
                {
                    status = $"Could not load LightGBM member from {path}.";
                    return null;
                }

                var lightGbmRuntime = new TrainedModelRuntime(
                    model,
                    Array.Empty<TrainedModelRuntime>(),
                    mlContext,
                    lightGbmModel);
                status = $"Loaded LightGBM ranking member from {Path.GetFileName(path)}.";
                return lightGbmRuntime;
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

            if (_lightGbmModel is not null)
            {
                if (!TryPredictLightGbm(rows, out probabilities))
                {
                    return false;
                }

                ApplyFieldCalibration(probabilities, rows.Count);
                return true;
            }

            if (_members.Count > 0)
            {
                if (!TryPredictEnsemble(rows, out probabilities))
                {
                    return false;
                }

                ApplyFieldCalibration(probabilities, rows.Count);
                return true;
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

            ApplyFieldCalibration(probabilities, rows.Count);
            return true;
        }

        private bool TryPredictLightGbm(
            IReadOnlyList<IDictionary<string, object?>> rows,
            out double[] probabilities)
        {
            probabilities = Array.Empty<double>();
            if (_lightGbmModel is null || _lightGbmContext is null || _model.Metadata is null)
            {
                return false;
            }

            var features = new List<float[]>(rows.Count);
            foreach (var row in rows)
            {
                try
                {
                    features.Add(HyperparameterTrainer.EncodeFeatureVector(
                        row,
                        _model.Metadata.Keys,
                        _model.Metadata.FeatureDimensions,
                        _model.Metadata.StringMaps,
                        _featureCount));
                }
                catch
                {
                    return false;
                }
            }

            return ScoreLightGbm(features, out probabilities);
        }

        private bool TryPredictLightGbmEncoded(
            IReadOnlyList<HyperparameterTrainer.RunnerExample> runners,
            out double[] probabilities)
        {
            probabilities = Array.Empty<double>();
            if (_lightGbmModel is null || _lightGbmContext is null)
            {
                return false;
            }

            var features = runners.Select(runner => runner.Features).ToList();
            return ScoreLightGbm(features, out probabilities);
        }

        private bool ScoreLightGbm(
            IReadOnlyList<float[]> features,
            out double[] probabilities)
        {
            probabilities = Array.Empty<double>();
            if (_lightGbmModel is null || _lightGbmContext is null || features.Count == 0)
            {
                return false;
            }

            var rows = LightGbmRankingModel.BuildPredictionRows(features);
            var schema = SchemaDefinition.Create(typeof(LightGbmRankingModel.RankingRow));
            schema[nameof(LightGbmRankingModel.RankingRow.Features)].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, _featureCount);
            var data = _lightGbmContext.Data.LoadFromEnumerable(rows, schema);
            var scored = _lightGbmModel.Transform(data);
            var scores = _lightGbmContext.Data
                .CreateEnumerable<LightGbmRankingModel.RankingPrediction>(
                    scored,
                    reuseRowObject: false)
                .Select(row => row.Score)
                .ToArray();
            if (scores.Length != features.Count)
            {
                return false;
            }

            var softmax = LightGbmRankingModel.SoftmaxScores(scores);
            probabilities = softmax.Select(value => (double)value).ToArray();
            return true;
        }

        private bool TryPredictEnsemble(
            IReadOnlyList<IDictionary<string, object?>> rows,
            out double[] probabilities)
        {
            probabilities = Array.Empty<double>();
            int count = rows.Count;
            var logSums = new double[count];
            var memberCounts = new int[count];

            foreach (var member in _members)
            {
                if (!member.TryPredictRace(rows, out var memberProbabilities) ||
                    memberProbabilities.Length != count)
                {
                    return false;
                }

                for (int i = 0; i < count; i++)
                {
                    double probability = memberProbabilities[i];
                    if (!double.IsFinite(probability))
                    {
                        return false;
                    }

                    logSums[i] += Math.Log(Math.Max(probability, MinimumProbability));
                    memberCounts[i]++;
                }
            }

            var values = new double[count];
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                if (memberCounts[i] == 0)
                {
                    return false;
                }

                values[i] = Math.Exp(logSums[i] / memberCounts[i]);
                sum += values[i];
            }

            if (sum <= double.Epsilon || !double.IsFinite(sum))
            {
                return false;
            }

            probabilities = new double[count];
            for (int i = 0; i < count; i++)
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

            if (_members.Count > 0 || _lightGbmModel is not null)
            {
                return false;
            }

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

            if (_marketFeatureMask.Length == normalized.Length)
            {
                for (int i = 0; i < normalized.Length; i++)
                {
                    if (_marketFeatureMask[i])
                    {
                        normalized[i] = 0d;
                    }
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
