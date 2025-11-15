using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;
using OpenQA.Selenium.BiDi.Script;
using Microsoft.Extensions.Configuration;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using HorseRacingML.Models;
using System.Text.Json;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System;
using Tensorflow;
using Tensorflow.NumPy;
using System.IO;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using static Tensorflow.TensorShapeProto.Types;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;
using TensorShape = Tensorflow.Shape;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Builds and trains a simple TensorFlow model using supplied hyperparameters.
    /// GPU is used when available. Training data is loaded from SQL Server using the
    /// HorseRacingML schema instead of randomly generated dummy values.
    /// </summary>
    public partial class HyperparameterTrainer
    {
        private readonly string _connectionString;
        private readonly float _winRateAlpha;
        private readonly float _winRateBeta;
        private IRacingRepository? _racingRepository;
        private readonly object _preparedDatasetCacheLock = new();
        private readonly Dictionary<PreparedDatasetCacheKey, WeakReference<PreparedDataset>> _preparedDatasetCache = new();
        private readonly object _featureMetadataCacheLock = new();
        private readonly Dictionary<(bool IncludeIdentifiers, string RaceSignature), DatasetFeatureMetadata> _featureMetadataCache = new();
        private readonly object _normalizationCacheLock = new();
        private readonly Dictionary<string, NormalizationParameters> _normalizationCache = new(StringComparer.Ordinal);
        public HyperparameterTrainer(IConfiguration configuration, IRacingRepository? racingRepository = null)
        {
            if (configuration is null)
            {
                throw new ArgumentNullException(nameof(configuration));
            }

            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
            _winRateAlpha = configuration.GetValue<float>("WinRateAlpha", 1f);
            _winRateBeta = configuration.GetValue<float>("WinRateBeta", 2f);
            _racingRepository = racingRepository;

        }
        public class TrainingResult
        {
            public double TrainAccuracy { get; init; }
            public double TrainLoss { get; init; }
            public double TrainBrier { get; init; }
            public double TrainFocalLoss { get; init; }
            public double ValidationAccuracy { get; init; }
            public double ValidationLoss { get; init; }
            public double ValidationBrier { get; init; }
            public double ValidationFocalLoss { get; init; }
            public IReadOnlyList<float> TrainingPredictions { get; init; } = Array.Empty<float>();
            public IReadOnlyList<float> TrainingLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> TrainingRaceIds { get; init; } = Array.Empty<int>();
            public IReadOnlyList<float> ValidationPredictions { get; init; } = Array.Empty<float>();
            public IReadOnlyList<float> ValidationLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> ValidationRaceIds { get; init; } = Array.Empty<int>();
            public IReadOnlyList<RunnerExample> ValidationExamples { get; init; } = Array.Empty<RunnerExample>();
            public IReadOnlyList<FeatureCorrelation> FeatureCorrelations { get; init; } = Array.Empty<FeatureCorrelation>();
        }
        private sealed record PreparedDatasetCacheKey(
            string IncludeRaceIdsSignature,
            string StateWhitelistSignature,
            bool IncludeIdentifiers,
            bool ApplyRepositoryBackfills)
        {
            public static PreparedDatasetCacheKey Create(
                ISet<int?>? includeRaceIds,
                ISet<int?>? stateRaceWhitelist,
                bool includeIdentifiers,
                bool applyRepositoryBackfills)
            {
                static string BuildSignature(ISet<int?>? source, string label)
                {
                    if (source is null)
                    {
                        return string.Concat(label, ":*");
                    }

                    if (source.Count == 0)
                    {
                        return string.Concat(label, ":");
                    }

                    var values = source
                        .Where(v => v.HasValue)
                        .Select(v => v.Value)
                        .OrderBy(v => v)
                        .ToArray();

                    return string.Concat(label, ":", string.Join(',', values));
                }

                return new PreparedDatasetCacheKey(
                    BuildSignature(includeRaceIds, nameof(includeRaceIds)),
                    BuildSignature(stateRaceWhitelist, nameof(stateRaceWhitelist)),
                    includeIdentifiers,
                    applyRepositoryBackfills);
            }
        }
        public sealed class FeatureCorrelation
        {
            public string FeatureKey { get; init; } = string.Empty;
            public string? Dimension { get; init; }
            public double Correlation { get; init; }
            public IReadOnlyList<RunnerExample> ValidationExamples { get; init; } = Array.Empty<RunnerExample>();
        }
        private static void NormalizeBatchSize(MLParameter param)
        {
            if (param is null)
            {
                throw new ArgumentNullException(nameof(param));
            }

            var normalized = MLParameterValidator.EnsureBatchSize(param.BatchSize, fallback: 0);
            if (normalized <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(param.BatchSize),
                    param.BatchSize,
                    "Batch size must be greater than zero.");
            }

            param.BatchSize = normalized;
        }


        public TrainingDataset LoadTrainingDataset(bool includeIdentifiers = false)
        {
            var prepared = PrepareDataset(includeRaceIds: null, stateRaceWhitelist: null, includeIdentifiers: includeIdentifiers);
            var emptyValidation = new PreparedDataset(new List<PreparedRace>());
            return BuildTrainingDataset(prepared, emptyValidation, includeIdentifiers);
        }
        public TrainingDataset LoadValidationDataset(ISet<int> trainingRaceIdsForState, ISet<int> validationRaceIds, bool includeIdentifiers = false)
        {
            if (trainingRaceIdsForState is null)
                throw new ArgumentNullException(nameof(trainingRaceIdsForState));
            if (validationRaceIds is null)
                throw new ArgumentNullException(nameof(validationRaceIds));

            var model = LoadModel();
            if (model == null)
            {
                throw new InvalidOperationException("Failed to load the AI model from weights file for validation.");
            }

            var validationPrepared = PrepareDataset(
                ToNullableSet(validationRaceIds),
                ToNullableSet(trainingRaceIdsForState),
                includeIdentifiers,
                applyRepositoryBackfills: false);

            var emptyTraining = new PreparedDataset(new List<PreparedRace>());

            var metadataSource = validationPrepared;
            var metadataRows = metadataSource.Rows.ToList();

            var signature = BuildRaceSignature(validationPrepared.Races);
            DatasetFeatureMetadata metadata;
            lock (_featureMetadataCacheLock)
            {
                if (!_featureMetadataCache.TryGetValue((includeIdentifiers, signature), out metadata))
                {
                    metadata = BuildFeatureMetadata(metadataSource, metadataRows);
                    _featureMetadataCache[(includeIdentifiers, signature)] = metadata;
                }
            }

            var validationRaces = EncodeRaces(validationPrepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);

            var normalizationKey = BuildNormalizationCacheKey(includeIdentifiers, signature);

            return new TrainingDataset(new List<RaceExample>(), validationRaces, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps, model.Normalization, normalizationKey);
        }
        public TrainingResult Train(MLParameter param, int foldIndex, int foldCount, bool persistWeights = true)
        {
            NormalizeBatchSize(param);
            var dataset = LoadTrainingDataset(includeIdentifiers: true);
            return Train(param, foldIndex, foldCount, dataset, persistWeights);
        }
        private static double ComputeWinnerAccuracy(IReadOnlyList<int> raceIds,
            IReadOnlyList<float> preds, IReadOnlyList<float> labels)
        {
            var grouped = raceIds.Select((raceId, idx) => new { raceId, idx })
                                 .GroupBy(x => x.raceId);

            int correct = 0;
            int total = 0;

            foreach (var group in grouped)
            {
                total++;
                var bestPred = group.OrderByDescending(g => preds[g.idx]).First().idx;
                var trueIdx = group.OrderByDescending(g => labels[g.idx]).First().idx;
                if (bestPred == trueIdx) correct++;
            }

            return total == 0 ? 0 : (double)correct / total;
        }
        private static HashSet<int?>? ToNullableSet(ISet<int>? source)
        {
            if (source is null)
            {
                return null;
            }

            var result = new HashSet<int?>();
            foreach (var value in source)
            {
                result.Add(value);
            }

            return result;
        }
        public float ComputeSmoothedWinRate(int wins, int starts)
        {
            if (wins < 0)
            {
                wins = 0;
            }

            if (starts < 0)
            {
                starts = 0;
            }

            return (wins + _winRateAlpha) / (starts + _winRateBeta);
        }
        public float SmoothedWinRate(int wins, int starts)
        {
            return ComputeSmoothedWinRate(wins, starts);
        }
        protected void SetRacingRepository(IRacingRepository? repository)
        {
            _racingRepository = repository;
        }

        public IRacingRepository? EnsureRacingRepository()
        {
            if (_racingRepository != null)
            {
                return _racingRepository;
            }

            if (string.IsNullOrWhiteSpace(_connectionString))
            {
                return null;
            }

            try
            {
                _racingRepository = new RacingRepository(_connectionString);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AI] Failed to initialize racing repository for historical backfills: {ex.Message}");
                _racingRepository = null;
            }

            return _racingRepository;
        }
        private static double ComputeBrier(float[] preds, float[] labels)
        {
            if (preds.Length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < preds.Length; i++)
            {
                double diff = preds[i] - labels[i];
                sum += diff * diff;
            }
            return sum / preds.Length;
        }
        private TrainingDataset BuildTrainingDataset(
        PreparedDataset trainingPrepared,
        PreparedDataset validationPrepared,
        bool includeIdentifiers = false)
        {
            if (trainingPrepared is null)
                throw new ArgumentNullException(nameof(trainingPrepared));
            if (validationPrepared is null)
                throw new ArgumentNullException(nameof(validationPrepared));

            var metadataSource = trainingPrepared.RowCount > 0 ? trainingPrepared : validationPrepared;
            var metadataRows = trainingPrepared.RowCount > 0
                ? trainingPrepared.Rows.ToList()
                : metadataSource.Rows.ToList();

            var signature = BuildRaceSignature(trainingPrepared.Races.Concat(validationPrepared.Races));
            DatasetFeatureMetadata metadata;
            lock (_featureMetadataCacheLock)
            {
                if (!_featureMetadataCache.TryGetValue((includeIdentifiers, signature), out metadata))
                {
                    metadata = BuildFeatureMetadata(metadataSource, metadataRows);
                    _featureMetadataCache[(includeIdentifiers, signature)] = metadata;
                }
            }
            int featureCount = metadata.FeatureCount;
            Console.WriteLine($"[AI] Prepared training dataset with {featureCount} features derived from {metadata.FeatureKeys.Count} source columns.");
            var trainRaces = EncodeRaces(trainingPrepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);
            var validationRaces = EncodeRaces(validationPrepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);

            var normalizationKey = BuildNormalizationCacheKey(includeIdentifiers, signature);
            var normalization = GetNormalizationParameters(normalizationKey, featureCount, out var loadedFromCache);
            if (loadedFromCache)
            {
                Console.WriteLine($"[AI] Using cached normalization statistics for dataset key '{normalizationKey}'.");
            }

            var mapPath = Path.Combine(AppContext.BaseDirectory, "string_maps.json");
            var serializedMaps = JsonSerializer.Serialize(metadata.StringMaps);
            WriteTextIfChanged(mapPath, serializedMaps);

            return new TrainingDataset(trainRaces, validationRaces, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps, normalization, normalizationKey);
        }
        private static void WriteTextIfChanged(string path, string content)
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentException("Path must be provided.", nameof(path));
            }

            content ??= string.Empty;

            try
            {
                if (File.Exists(path))
                {
                    var existing = File.ReadAllText(path);
                    if (string.Equals(existing, content, StringComparison.Ordinal))
                    {
                        return;
                    }
                }
            }
            catch (IOException)
            {
                // If the read fails we still attempt to write the fresh content below.
            }
            catch (UnauthorizedAccessException)
            {
                // If we cannot read the existing file, fall back to writing the new content.
            }

            File.WriteAllText(path, content);
        }
        private static string BuildRaceSignature(IEnumerable<PreparedRace> races)
        {
            if (races is null)
            {
                return string.Empty;
            }

            var ids = races
                .Where(r => r is not null)
                .Select(r => r!.RaceId)
                .Distinct()
                .OrderBy(id => id)
                .ToArray();

            return ids.Length == 0 ? string.Empty : string.Join(',', ids);
        }
        private static string BuildNormalizationCacheKey(bool includeIdentifiers, string raceSignature)
        {
            var signature = string.IsNullOrEmpty(raceSignature) ? "*" : raceSignature;
            var identifierPart = includeIdentifiers ? "id:1" : "id:0";
            return string.Concat(identifierPart, '|', signature);
        }
        private static NormalizationParameters CloneNormalization(NormalizationParameters source)
        {
            if (source is null)
            {
                throw new ArgumentNullException(nameof(source));
            }

            return new NormalizationParameters
            {
                Mean = source.Mean is null ? Array.Empty<float>() : (float[])source.Mean.Clone(),
                StdDev = source.StdDev is null ? Array.Empty<float>() : (float[])source.StdDev.Clone()
            };
        }
        private NormalizationParameters GetNormalizationParameters(string cacheKey, int featureCount, out bool fromCache)
        {
            fromCache = false;
            if (!string.IsNullOrEmpty(cacheKey))
            {
                lock (_normalizationCacheLock)
                {
                    if (_normalizationCache.TryGetValue(cacheKey, out var cached) &&
                        cached.Mean.Length == featureCount &&
                        cached.StdDev.Length == featureCount)
                    {
                        fromCache = true;
                        Console.WriteLine($"[AI] Loaded cached normalization for key '{cacheKey}'.");
                        return CloneNormalization(cached);
                    }
                }
            }

            return new NormalizationParameters
            {
                Mean = new float[featureCount],
                StdDev = new float[featureCount]
            };
        }
        private void CacheNormalization(string cacheKey, NormalizationParameters normalization)
        {
            if (string.IsNullOrEmpty(cacheKey) || normalization is null)
            {
                return;
            }

            lock (_normalizationCacheLock)
            {
                _normalizationCache[cacheKey] = CloneNormalization(normalization);
            }
        }
        private static bool HasComputedNormalization(float[] means, float[] stdDevs, int featureCount)
        {
            if (means is null || stdDevs is null)
            {
                return false;
            }

            if (means.Length != featureCount || stdDevs.Length != featureCount)
            {
                return false;
            }

            for (int i = 0; i < stdDevs.Length; i++)
            {
                float value = stdDevs[i];
                if (value <= 0f || float.IsNaN(value) || float.IsInfinity(value))
                {
                    return false;
                }
            }

            return true;
        }
        public TrainingResult Train(MLParameter param, TrainingDataset.PreparedDataset dataset, int foldIndex, int foldCount, bool persistWeights = true)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));
            NormalizeBatchSize(param);
            int totalRaces = dataset.Races.Count;
            if (foldCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(foldCount));
            if (foldIndex < 0 || foldIndex >= foldCount)
                throw new ArgumentOutOfRangeException(nameof(foldIndex));

            int foldSize = totalRaces / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalRaces : valStart + foldSize;

            var trainRaceIds = dataset.Races
                .Select((race, idx) => new { race, idx })
                .Where(x => x.idx < valStart || x.idx >= valEnd)
                .Select(x => x.race.RaceId)
                .ToHashSet();

            if (trainRaceIds.Count == 0)
            {
                trainRaceIds = dataset.Races
                    .Select(r => r.RaceId)
                    .ToHashSet();
            }

            var validationRaceIds = dataset.Races
                .Skip(valStart)
                .Take(valEnd - valStart)
                .Select(r => r.RaceId)
                .ToHashSet();

            var trainingRaces = new List<PreparedRace>();
            var validationRaces = new List<PreparedRace>();

            foreach (var race in dataset.Races)
            {
                if (trainRaceIds.Contains(race.RaceId))
                {
                    trainingRaces.Add(race);
                }
                else if (validationRaceIds.Contains(race.RaceId))
                {
                    validationRaces.Add(race);
                }
            }

            var trainingPrepared = new TrainingDataset.PreparedDataset(trainingRaces);

            var validationPrepared = validationRaceIds.Count > 0
                ? new TrainingDataset.PreparedDataset(validationRaces)
                : new TrainingDataset.PreparedDataset(new List<PreparedRace>());

            var trainingDataset = BuildTrainingDataset(trainingPrepared, validationPrepared, includeIdentifiers: false);
            return Train(param, foldIndex, foldCount, trainingDataset, persistWeights);
        }
        public TrainingResult Train(MLParameter param, int foldIndex, int foldCount, TrainingDataset dataset, bool persistWeights = true)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));
            NormalizeBatchSize(param);
            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

                _ = foldIndex;
                _ = foldCount;

                var trainExamples = dataset.TrainingRaces
                    .SelectMany(r => r.Runners)
                    .ToList();
                var valExamples = dataset.ValidationRaces
                    .SelectMany(r => r.Runners)
                    .ToList();

                int featureCount = dataset.FeatureCount;
            var trainFeatures = trainExamples.Select(r => (float[])r.Features.Clone()).ToList();
            var trainLabels = trainExamples.Select(r => r.Label).ToArray();
            var trainRaceIds = trainExamples.Select(r => r.RaceId).ToArray();

            var valFeatures = valExamples.Select(r => (float[])r.Features.Clone()).ToList();
            var valLabels = valExamples.Select(r => r.Label).ToArray();
            var valRaceIds = valExamples.Select(r => r.RaceId).ToArray();
            bool restoreFeatureState = false;

            var normalizationKey = dataset.NormalizationCacheKey;
            var means = dataset.Normalization.Mean ?? Array.Empty<float>();
            var stdDevs = dataset.Normalization.StdDev ?? Array.Empty<float>();
            bool reuseNormalization = HasComputedNormalization(means, stdDevs, featureCount);

            if (!reuseNormalization)
            {
                if (means.Length != featureCount)
                {
                    means = new float[featureCount];
                    dataset.Normalization.Mean = means;
                }
                else
                {
                    Array.Clear(means, 0, featureCount);
                }

                if (stdDevs.Length != featureCount)
                {
                    stdDevs = new float[featureCount];
                    dataset.Normalization.StdDev = stdDevs;
                }
                else
                {
                    Array.Clear(stdDevs, 0, featureCount);
                }

                if (trainFeatures.Count > 0)
                {
                    Parallel.For(0, featureCount, j =>
                    {
                        double sum = 0;
                        foreach (var feature in trainFeatures)
                        {
                            sum += feature[j];
                        }
                        means[j] = (float)(sum / trainFeatures.Count);
                    });

                    Parallel.For(0, featureCount, j =>
                    {
                        double variance = 0;
                        foreach (var feature in trainFeatures)
                        {
                            double diff = feature[j] - means[j];
                            variance += diff * diff;
                        }
                        stdDevs[j] = (float)Math.Sqrt(variance / trainFeatures.Count);
                        if (stdDevs[j] == 0f)
                        {
                            stdDevs[j] = 1f;
                        }
                    });
                }
                else
                {
                    Array.Clear(means, 0, featureCount);
                    for (int j = 0; j < featureCount; j++)
                    {
                        stdDevs[j] = 1f;
                    }
                }

                CacheNormalization(normalizationKey, dataset.Normalization);
            }
            else
            {
                Console.WriteLine($"[AI] Reusing cached normalization statistics for key '{normalizationKey}'.");
            }

            means = dataset.Normalization.Mean;
            stdDevs = dataset.Normalization.StdDev;


            void Normalize(IList<float[]> data)
            {
                Parallel.ForEach(data, arr =>
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        arr[i] = (arr[i] - means[i]) / stdDevs[i];
                    }
                });
            }

            Normalize(trainFeatures);
            Normalize(valFeatures);

            static float[,] BuildFeatureMatrix(List<float[]> source, int featureCount)
            {
                var matrix = new float[source.Count, featureCount];
                for (int i = 0; i < source.Count; i++)
                {
                    var row = source[i];
                    for (int j = 0; j < featureCount; j++)
                    {
                        matrix[i, j] = row[j];
                    }
                }
                return matrix;
            }

            static float[,] BuildLabelMatrix(float[] labels)
            {
                var matrix = new float[labels.Length, 1];
                for (int i = 0; i < labels.Length; i++)
                {
                    matrix[i, 0] = labels[i];
                }
                return matrix;
            }
            static void Shuffle(int[] values, Random random)
            {
                for (int i = values.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (values[i], values[j]) = (values[j], values[i]);
                }
            }

            var trainFeatureTensor = Tensorflow.NumPy.np.array(
                BuildFeatureMatrix(trainFeatures, featureCount),
                dtype: tf.float32);
            var trainLabelTensor = Tensorflow.NumPy.np.array(
                BuildLabelMatrix(trainLabels),
                dtype: tf.float32);
            bool hasValidationExamples = valExamples.Count > 0 && valLabels.Length > 0;
            NDArray? valFeatureTensor = null;
            NDArray? valLabelTensor = null;
            if (hasValidationExamples)
            {
                valFeatureTensor = Tensorflow.NumPy.np.array(
                    BuildFeatureMatrix(valFeatures, featureCount),
                    dtype: tf.float32);
                valLabelTensor = Tensorflow.NumPy.np.array(
                    BuildLabelMatrix(valLabels),
                    dtype: tf.float32);
            }
            var graph = tf.Graph().as_default();

            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "y");
            Tensor layer = x;
            int inputDim = featureCount;
            var hiddenWeightVars = new List<ResourceVariable>();
            var hiddenBiasVars = new List<ResourceVariable>();
            for (int i = 0; i < param.Layers; i++)
            {
                var w = tf.Variable(tf.random.normal((inputDim, param.Units)), name: $"w{i}");
                var b = tf.Variable(tf.zeros(param.Units), name: $"b{i}");
                hiddenWeightVars.Add(w);
                hiddenBiasVars.Add(b);
                layer = tf.nn.relu(tf.matmul(layer, w) + b);
                if (param.Dropout > 0)
                {
                    layer = tf.nn.dropout(layer, rate: (float)param.Dropout);
                }
                inputDim = param.Units;
            }

            var wOut = tf.Variable(tf.random.normal((inputDim, 1)), name: "wOut");
            var bOut = tf.Variable(tf.zeros(1), name: "bOut");
            var logits = tf.matmul(layer, wOut) + bOut;
            var loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: y, logits: logits));
            var focalLoss = tf.reduce_mean(SigmoidFocalLoss(y, logits));
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(focalLoss);

            var prediction = tf.sigmoid(logits);
            var rnd = new Random();
            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());
            var allVariables = hiddenWeightVars.Concat(hiddenBiasVars).Concat(new[] { wOut, bOut }).ToList();
            var bestWeights = new List<NDArray>();
            var normPath = Path.Combine(AppContext.BaseDirectory, "normalization.json");
            var normalizationJson = JsonSerializer.Serialize(dataset.Normalization);
            WriteTextIfChanged(normPath, normalizationJson);

            var trainPreds = new float[trainLabels.Length];
            var valPreds = new float[valLabels.Length];
            var epochPredBuffer = new float[trainLabels.Length];

            (double loss, double focalLoss) ComputeDatasetMetrics(NDArray featureTensor, NDArray labelTensor, float[] preds, int count)
            {
                if (preds.Length != count)
                    throw new ArgumentException("Prediction buffer size must match example count.", nameof(preds));

                if (count == 0)
                {
                    return (0, 0);
                }

                const int evalBatchSize = 8192;
                double weightedLoss = 0;
                double weightedFocalLoss = 0;
                int totalExamples = 0;

                for (int start = 0; start < count; start += evalBatchSize)
                {
                    int batchCount = Math.Min(evalBatchSize, count - start);

                    var featureSlice = featureTensor[new Slice(start, start + batchCount), Slice.All];
                    var labelSlice = labelTensor[new Slice(start, start + batchCount), Slice.All];

                    var results = sess.run(new[] { loss, focalLoss, prediction },
                        new FeedItem(x, featureSlice),
                        new FeedItem(y, labelSlice));

                    var chunkLoss = results[0].ToArray<float>()[0];
                    var chunkFocalLoss = results[1].ToArray<float>()[0];
                    var chunkPreds = results[2].ToArray<float>();
                    Array.Copy(chunkPreds, 0, preds, start, batchCount);
                    weightedLoss += chunkLoss * batchCount;
                    weightedFocalLoss += chunkFocalLoss * batchCount;
                    totalExamples += batchCount;
                }

                return totalExamples > 0 ? (weightedLoss / totalExamples, weightedFocalLoss / totalExamples) : (0, 0);
            }

            void Denormalize(IList<float[]> data)
            {
                Parallel.ForEach(data, arr =>
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        arr[i] = arr[i] * stdDevs[i] + means[i];
                    }
                });
            }

            double ComputeBrier(float[] preds, float[] labels)
            {
                if (preds.Length == 0)
                    return 0;

                double sum = 0;
                for (int i = 0; i < preds.Length; i++)
                {
                    double diff = preds[i] - labels[i];
                    sum += diff * diff;
                }
                return sum / preds.Length;
            }

            double trainLoss = 0;
            double valLoss = 0;
            double trainBrier = 0;
            double valBrier = 0;
            double trainFocalLoss = 0;
            double valFocalLoss = 0;
            double trainAcc = 0;
            double valAcc = 0;
            double bestValLoss = double.MaxValue;
            int patience = 10;
            int patienceCounter = 0;
            string? bestModelPath = null;
            List<FeatureCorrelation> featureCorrelations = new();
            HyperparameterStarted(param, trainLabels.Length, valLabels.Length, featureCount);

            try
            {
                int n = trainLabels.Length;
                if (n > 0)
                {
                    bestValLoss = double.MaxValue;
                    patience = 5;
                    int wait = 0;
                    var indices = Enumerable.Range(0, n).ToArray();
                    var batchIndexBuffer = new int[param.BatchSize];
                    for (int epoch = 0; epoch < param.Epochs; epoch++)
                    {
                        Shuffle(indices, rnd);

                        for (int start = 0; start < n; start += param.BatchSize)
                        {
                            int batchCount = Math.Min(param.BatchSize, n - start);
                            var batchIndices = indices.AsSpan(start, batchCount);
                            batchIndices.CopyTo(batchIndexBuffer);
                            var batchIndexArray = batchIndexBuffer.AsSpan(0, batchCount).ToArray();
                            using var batchIndexTensor = Tensorflow.NumPy.np.array(batchIndexArray, dtype: tf.int32);
                            using var batchFeatures = trainFeatureTensor[batchIndexTensor];
                            using var batchLabels = trainLabelTensor[batchIndexTensor];

                            sess.run(optimizer,
                                new FeedItem(x, batchFeatures),
                                new FeedItem(y, batchLabels));
                        }

                        (var epochLoss, var epochFocalLoss) = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, epochPredBuffer, n);
                        var epochAcc = ComputeWinnerAccuracy(trainRaceIds, epochPredBuffer, trainLabels);

                        if (hasValidationExamples && valFeatureTensor != null && valLabelTensor != null)
                        {
                            (var epochValLoss, var epochValFocalLoss) = ComputeDatasetMetrics(valFeatureTensor, valLabelTensor, valPreds, valLabels.Length);
                            var epochValAcc = ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels);
                            Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - focal_loss: {epochFocalLoss:F4} - winner acc: {epochAcc:F4} - val_loss: {epochValLoss:F4} - val_focal_loss: {epochValFocalLoss:F4} - val_acc: {epochValAcc:F4}");

                            if (epochValLoss < bestValLoss)
                            {
                                bestValLoss = epochValLoss;
                                patienceCounter = 0;
                                bestModelPath = Path.GetTempFileName();
                                var saver = tf.train.Saver();
                                saver.save(sess, bestModelPath);
                            }
                            else
                            {
                                patienceCounter++;
                            }

                            if (patienceCounter >= patience)
                            {
                                Console.WriteLine("Early stopping due to no improvement in validation loss.");
                                break;
                            }
                        }
                        else
                        {
                            Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
                        }
                    }

                    if (hasValidationExamples && bestWeights.Count > 0)
                    {
                        for (int i = 0; i < allVariables.Count; i++)
                        {
                            sess.run(allVariables[i].assign(bestWeights[i]));
                        }
                    }
                    (trainLoss, trainFocalLoss) = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, trainPreds, trainLabels.Length);
                    trainBrier = ComputeBrier(trainPreds, trainLabels);

                    if (hasValidationExamples && valFeatureTensor is not null && valLabelTensor is not null)
                    {
                        (valLoss, valFocalLoss) = ComputeDatasetMetrics(valFeatureTensor, valLabelTensor, valPreds, valLabels.Length);
                        valBrier = ComputeBrier(valPreds, valLabels);
                    }
                    else
                    {
                        Array.Clear(valPreds, 0, valPreds.Length);
                        valLoss = 0;
                        valBrier = 0;
                        valFocalLoss = 0;
                    }

                    trainAcc = trainPreds.Length > 0 ? ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels) : 0;
                    valAcc = hasValidationExamples && valPreds.Length > 0 ? ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels) : 0;
                    bool useValidationCorrelations = hasValidationExamples && valFeatures.Count > 0 && valPreds.Length > 0;
                    var correlationFeatures = useValidationCorrelations ? valFeatures : trainFeatures;
                    var correlationPreds = useValidationCorrelations ? valPreds : trainPreds;
                    var correlationLabels = useValidationCorrelations ? valLabels : trainLabels;

                    if (param.EnableFeatureCorrelations)
                    {
                        featureCorrelations = ComputeFeatureCorrelations(
                            correlationFeatures,
                            correlationPreds,
                            correlationLabels,
                            dataset.FeatureKeys,
                            dataset.FeatureDimensions,
                            dataset.StringMaps);
                    }
                    else
                    {
                        Console.WriteLine("[TrainAI] Skipping feature correlation computation because it was disabled for this run.");
                    }
                }
            }
            catch (Exception ex)
            {
                HyperparameterFailed(param, ex);
                throw;
            }
            finally
            {
                if (restoreFeatureState)
                {
                    Denormalize(trainFeatures);
                    Denormalize(valFeatures);
                }
                if (bestModelPath != null && File.Exists(bestModelPath))
                {
                    var saver = tf.train.Saver();
                    saver.restore(sess, bestModelPath);
                    File.Delete(bestModelPath);
                }
            }
            var hiddenLayers = new List<LayerWeights>();
            foreach (var (wVar, bVar) in hiddenWeightVars.Zip(hiddenBiasVars, (wVar, bVar) => (wVar, bVar)))
            {
                var weightArray = ToJagged2D(sess.run(wVar));
                var biasArray = sess.run(bVar).ToArray<float>();
                hiddenLayers.Add(new LayerWeights { Weights = weightArray, Bias = biasArray });
            }

            var outputLayer = new LayerWeights
            {
                Weights = ToJagged2D(sess.run(wOut)),
                Bias = sess.run(bOut).ToArray<float>()
            };
            var featureKeys = dataset.FeatureKeys;
            var featureDims = dataset.FeatureDimensions;
            var stringMaps = dataset.StringMaps;
            var trainedAtUtc = DateTime.UtcNow;
            if (param.RunDate != default)
            {
                trainedAtUtc = param.RunDate.Kind switch
                {
                    DateTimeKind.Unspecified => DateTime.SpecifyKind(param.RunDate, DateTimeKind.Utc),
                    DateTimeKind.Utc => param.RunDate,
                    _ => param.RunDate.ToUniversalTime()
                };
            }
            var model = new TrainedModel
            {
                HiddenLayers = hiddenLayers,
                OutputLayer = outputLayer,
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string>(featureKeys),
                    FeatureDimensions = new Dictionary<string, int>(featureDims),
                    StringMaps = stringMaps.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new Dictionary<string, int>(kvp.Value))
                },
                Normalization = new NormalizationParameters
                {
                    Mean = (float[])means.Clone(),
                    StdDev = (float[])stdDevs.Clone()
                },
                Hyperparameters = new HyperparameterSummary
                {
                    Layers = param.Layers,
                    Units = param.Units,
                    Dropout = param.Dropout,
                    LearningRate = param.LearningRate,
                    Epochs = param.Epochs,
                    BatchSize = param.BatchSize,
                    Folds = param.Folds,
                    Fold = param.Fold,
                    TrainedAtUtc = trainedAtUtc
                }
            };

            if (persistWeights)
            {
                var weightsDirectory = Path.Combine(AppContext.BaseDirectory, "weights");
                var legacyWeightPath = Path.Combine(AppContext.BaseDirectory, "aiweights.json");
                try
                {
                    Directory.CreateDirectory(weightsDirectory);

                    var options = new JsonSerializerOptions { WriteIndented = true };
                    var serialized = JsonSerializer.Serialize(model, options);
                    var weightPath = Path.Combine(weightsDirectory, "aiweights.json");
                    File.WriteAllText(weightPath, serialized);

                    try
                    {
                        File.WriteAllText(legacyWeightPath, serialized);
                    }
                    catch
                    {
                        // Updating the legacy path is best-effort only.
                    }
                }
                catch
                {
                    // Updating the legacy path is best-effort only.
                }
                }
            HyperparameterCompleted(param, trainAcc, valAcc, trainLoss, valLoss, trainBrier, valBrier, trainFocalLoss, valFocalLoss);
            return new TrainingResult
            {
                TrainAccuracy = trainAcc,
                TrainLoss = trainLoss,
                TrainBrier = trainBrier,
                TrainFocalLoss = trainFocalLoss,
                ValidationAccuracy = valAcc,
                ValidationLoss = valLoss,
                ValidationBrier = valBrier,
                ValidationFocalLoss = valFocalLoss,
                TrainingPredictions = Array.AsReadOnly(trainPreds),
                TrainingLabels = Array.AsReadOnly(trainLabels),
                TrainingRaceIds = Array.AsReadOnly(trainRaceIds),
                ValidationPredictions = Array.AsReadOnly(valPreds),
                ValidationLabels = Array.AsReadOnly(valLabels),
                ValidationRaceIds = Array.AsReadOnly(valRaceIds),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(valExamples),
                FeatureCorrelations = new ReadOnlyCollection<FeatureCorrelation>(featureCorrelations)
            };
            static float[][] ToJagged2D(NDArray array)
            {
                if (array.ndim != 2)
                    throw new InvalidOperationException($"Expected 2-D tensor but received rank {array.ndim}");

                var shape = array.shape;
                int rows = (int)shape[0];
                int cols = (int)shape[1];
                var flat = array.ToArray<float>();
                var result = new float[rows][];
                for (int r = 0; r < rows; r++)
                {
                    var row = new float[cols];
                    Array.Copy(flat, r * cols, row, 0, cols);
                    result[r] = row;
                }
                return result;
            }
        }
        private static List<FeatureCorrelation> ComputeFeatureCorrelations(
            IList<float[]> normalizedFeatures,
            IReadOnlyList<float> predictions,
            IReadOnlyList<float>? labelValues,
            IList<string> featureKeys,
            IDictionary<string, int> featureDimensions,
            IDictionary<string, Dictionary<string, int>> stringMaps)
        {
            if (normalizedFeatures is null)
                throw new ArgumentNullException(nameof(normalizedFeatures));
            if (predictions is null)
                throw new ArgumentNullException(nameof(predictions));
            if (featureKeys is null)
                throw new ArgumentNullException(nameof(featureKeys));
            if (featureDimensions is null)
                throw new ArgumentNullException(nameof(featureDimensions));
            if (stringMaps is null)
                throw new ArgumentNullException(nameof(stringMaps));

            int exampleCount = Math.Min(normalizedFeatures.Count, predictions.Count);
            if (exampleCount == 0)
            {
                Console.WriteLine("[TrainAI] Skipping feature correlation computation because no examples were provided.");
                return new List<FeatureCorrelation>();
            }
            var targets = (IReadOnlyList<float>)predictions;
            double sumY = 0;
            double sumY2 = 0;
            for (int i = 0; i < exampleCount; i++)
            {
                double y = targets[i];
                sumY += y;
                sumY2 += y * y;
            }
            static bool HasVariance(int count, double sum, double sumSquares)
            {
                if (count == 0)
                {
                    return false;
                }

                double varianceComponent = count * sumSquares - sum * sum;
                return varianceComponent > 0;
            }

            if (!HasVariance(exampleCount, sumY, sumY2) && labelValues is { Count: > 0 })
            {
                exampleCount = Math.Min(exampleCount, labelValues.Count);
                if (exampleCount == 0)
                {
                    Console.WriteLine("[TrainAI] Skipping feature correlation computation because neither predictions nor labels provided variance.");
                    return new List<FeatureCorrelation>();
                }

                targets = labelValues;
                sumY = 0;
                sumY2 = 0;
                for (int i = 0; i < exampleCount; i++)
                {
                    double y = targets[i];
                    sumY += y;
                    sumY2 += y * y;
                }
                Console.WriteLine("[TrainAI] Skipping feature correlation computation because neither predictions nor labels provided variance.");
            }

            var correlations = new ConcurrentBag<FeatureCorrelation>();
            int totalDimensions = featureKeys.Sum(key => featureDimensions.TryGetValue(key, out var dim) ? dim : 0);
            Console.WriteLine($"[TrainAI] Computing feature correlations for {featureKeys.Count} feature keys spanning {totalDimensions} dimensions using {exampleCount} examples.");

            var featureOffsets = new Dictionary<string, int>();
            int currentOffset = 0;
            foreach (var featureKey in featureKeys)
            {
                featureOffsets[featureKey] = currentOffset;
                if (featureDimensions.TryGetValue(featureKey, out var dim))
                {
                    currentOffset += dim;
                }
            }

            Parallel.ForEach(featureKeys, featureKey =>
            {
                if (!featureDimensions.TryGetValue(featureKey, out var dim) || dim <= 0)
                {
                    return;
                }
                int offset = featureOffsets[featureKey];

                string[]? categories = null;
                if (stringMaps.TryGetValue(featureKey, out var map) && map.Count > 0)
                {
                    categories = new string[map.Count];
                    foreach (var kvp in map)
                    {
                        var label = kvp.Key;
                        if (string.Equals(label, "__unknown__", StringComparison.Ordinal))
                        {
                            label = "Unknown";
                        }

                        if (kvp.Value >= 0 && kvp.Value < categories.Length)
                        {
                            categories[kvp.Value] = label;
                        }
                    }
                }

                int baseDim = categories?.Length ?? 1;
                bool hasMissingIndicator = dim > baseDim;

                for (int d = 0; d < dim; d++)
                {
                    double sumX = 0;
                    double sumX2 = 0;
                    double sumXY = 0;

                    for (int row = 0; row < exampleCount; row++)
                    {
                        var featureRow = normalizedFeatures[row];
                        if (featureRow.Length <= offset + d)
                        {
                            continue;
                        }

                        double x = featureRow[offset + d];
                        double y = targets[row];
                        sumX += x;
                        sumX2 += x * x;
                        sumXY += x * y;
                    }

                    double numerator = exampleCount * sumXY - sumX * sumY;
                    double denomLeft = exampleCount * sumX2 - sumX * sumX;
                    double denomRight = exampleCount * sumY2 - sumY * sumY;
                    double denom = Math.Sqrt(Math.Max(denomLeft, 0) * Math.Max(denomRight, 0));
                    double corr = denom > 0 ? numerator / denom : 0;

                    string? dimensionLabel = null;
                    if (categories != null && d < categories.Length)
                    {
                        dimensionLabel = categories[d];
                    }
                    else if (categories != null && hasMissingIndicator && d == categories.Length)
                    {
                        dimensionLabel = "Missing";
                    }
                    else if (categories is null)
                    {
                        dimensionLabel = d == 0 ? "Value" : "Missing";
                    }
                    if (string.Equals(dimensionLabel, "Missing", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }
                    if (string.IsNullOrEmpty(dimensionLabel))
                    {
                        dimensionLabel = $"Dim {d + 1}";
                    }

                    correlations.Add(new FeatureCorrelation
                    {
                        FeatureKey = featureKey,
                        Dimension = dimensionLabel,
                        Correlation = corr
                    });
                }
            });

            Console.WriteLine($"[TrainAI] Feature correlation computation complete. Generated {correlations.Count} correlation entries.");
            return correlations.ToList();
        }
        public TrainingDataset LoadTrainingDataset(ISet<int> trainingRaceIds, ISet<int> validationRaceIds, bool includeIdentifiers = false)
        {
            if (trainingRaceIds is null)
                throw new ArgumentNullException(nameof(trainingRaceIds));
            if (validationRaceIds is null)
                throw new ArgumentNullException(nameof(validationRaceIds));

            var combinedRaceIds = new HashSet<int>(trainingRaceIds);
            combinedRaceIds.UnionWith(validationRaceIds);

            var combinedPrepared = PrepareDataset(
                ToNullableSet(combinedRaceIds),
                ToNullableSet(trainingRaceIds),
                includeIdentifiers);

            var trainingPrepared = new PreparedDataset(
                combinedPrepared.Races
                    .Where(r => trainingRaceIds.Contains(r.RaceId))
                    .ToList());

            var validationPrepared = validationRaceIds.Count > 0
                ? new PreparedDataset(
                    combinedPrepared.Races
                        .Where(r => validationRaceIds.Contains(r.RaceId))
                        .ToList())
                : new PreparedDataset(new List<PreparedRace>());

            return BuildTrainingDataset(trainingPrepared, validationPrepared, includeIdentifiers);
        }
        public HyperparameterSummary? LoadPersistedHyperparameters()
        {
            var weightDirectory = Path.Combine(AppContext.BaseDirectory, "weights");
            var weightPath = Path.Combine(weightDirectory, "aiweights.json");
            if (!File.Exists(weightPath))
            {
                weightPath = Path.Combine(AppContext.BaseDirectory, "aiweights.json");
                if (!File.Exists(weightPath))
                {
                    return null;
                }
            }

            try
            {
                var json = File.ReadAllText(weightPath);
                var model = JsonSerializer.Deserialize<TrainedModel>(json);
                return model?.Hyperparameters;
            }
            catch (IOException)
            {
                return null;
            }
            catch (UnauthorizedAccessException)
            {
                return null;
            }
            catch (JsonException)
            {
                return null;
            }
        }
        private static void HyperparameterStarted(MLParameter param, int trainExampleCount, int validationExampleCount, int featureCount)
        {
            Console.WriteLine(
                "[Hyperparameter] Training started | layers: {0}, units: {1}, dropout: {2}, lr: {3}, epochs: {4}, batch: {5}, train examples: {6}, val examples: {7}, features: {8}",
                param.Layers,
                param.Units,
                param.Dropout,
                param.LearningRate,
                param.Epochs,
                param.BatchSize,
                trainExampleCount,
                validationExampleCount,
                featureCount);
        }

        private static Tensor SigmoidFocalLoss(Tensor labels, Tensor logits, float alpha = 0.25f, float gamma = 2.0f)
        {
            var prob = tf.sigmoid(logits);
            var ce = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits);
            var prob_t = (labels * prob) + ((1 - labels) * (1 - prob));
            var loss = ce * tf.pow(1 - prob_t, gamma);
            var alpha_t = (labels * alpha) + ((1 - labels) * (1 - alpha));
            loss = alpha_t * loss;
            return loss;
        }

        private static void HyperparameterCompleted(MLParameter param, double trainAccuracy, double validationAccuracy, double trainLoss, double validationLoss, double trainBrier, double validationBrier, double trainFocalLoss, double validationFocalLoss)
        {
            Console.WriteLine(
                "[Hyperparameter] Training complete | layers: {0}, units: {1}, epochs: {2}, train acc: {3:F4}, val acc: {4:F4}, train loss: {5:F4}, val loss: {6:F4}, train brier: {7:F4}, val brier: {8:F4}, train focal: {9:F4}, val focal: {10:F4}",
                param.Layers,
                param.Units,
                param.Epochs,
                trainAccuracy,
                validationAccuracy,
                trainLoss,
                validationLoss,
                trainBrier,
                validationBrier,
                trainFocalLoss,
                validationFocalLoss);
        }
        private static void HyperparameterFailed(MLParameter param, Exception exception)
        {
            Console.Error.WriteLine(
                "[Hyperparameter] Training failed | layers: {0}, units: {1}, dropout: {2}, lr: {3}, epochs: {4}, batch: {5}. Error: {6}",
                param.Layers,
                param.Units,
                param.Dropout,
                param.LearningRate,
                param.Epochs,
                param.BatchSize,
                exception);
        }
        public TrainingResult Evaluate(TrainingDataset dataset)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));

            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

            var trainExamples = dataset.TrainingRaces.SelectMany(r => r.Runners).ToList();
            var valExamples = dataset.ValidationRaces.SelectMany(r => r.Runners).ToList();
            int featureCount = dataset.FeatureCount;

            var valFeatures = valExamples.Select(r => (float[])r.Features.Clone()).ToList();
            var valLabels = valExamples.Select(r => r.Label).ToArray();
            var valRaceIds = valExamples.Select(r => r.RaceId).ToArray();

            var means = dataset.Normalization.Mean ?? Array.Empty<float>();
            var stdDevs = dataset.Normalization.StdDev ?? Array.Empty<float>();

            void Normalize(IList<float[]> data)
            {
                Parallel.ForEach(data, arr =>
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        arr[i] = (arr[i] - means[i]) / stdDevs[i];
                    }
                });
            }

            Normalize(valFeatures);

            var model = LoadModel();
            if (model == null)
            {
                throw new InvalidOperationException("Failed to load the AI model from weights file.");
            }

            var valPreds = new float[valLabels.Length];
            // Evaluation logic using the loaded model
            // This part needs to be implemented based on the model structure
            // For now, let's assume a simple forward pass
            for (int i = 0; i < valFeatures.Count; i++)
            {
                var input = valFeatures[i];
                var output = ForwardPass(model, input);
                valPreds[i] = output;
            }

            var valAcc = valPreds.Length > 0 ? ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels) : 0;
            var valLoss = 0; // Loss calculation would require the loss function
            var valBrier = ComputeBrier(valPreds, valLabels);
            var valFocalLoss = 0; // Focal loss calculation

            return new TrainingResult
            {
                ValidationAccuracy = valAcc,
                ValidationLoss = valLoss,
                ValidationBrier = valBrier,
                ValidationFocalLoss = valFocalLoss,
                ValidationPredictions = new ReadOnlyCollection<float>(valPreds),
                ValidationLabels = new ReadOnlyCollection<float>(valLabels),
                ValidationRaceIds = new ReadOnlyCollection<int>(valRaceIds),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(valExamples),
            };
        }

        private float ForwardPass(TrainedModel model, float[] input)
        {
            var layerInput = input;
            foreach (var layer in model.HiddenLayers)
            {
                var layerOutput = new float[layer.Bias.Length];
                for (int i = 0; i < layer.Bias.Length; i++)
                {
                    float sum = 0;
                    for (int j = 0; j < layerInput.Length; j++)
                    {
                        sum += layerInput[j] * layer.Weights[j][i];
                    }
                    sum += layer.Bias[i];
                    layerOutput[i] = Math.Max(0, sum); // ReLU activation
                }
                layerInput = layerOutput;
            }

            float output = 0;
            for (int i = 0; i < layerInput.Length; i++)
            {
                output += layerInput[i] * model.OutputLayer.Weights[i][0];
            }
            output += model.OutputLayer.Bias[0];

            return 1 / (1 + (float)Math.Exp(-output)); // Sigmoid activation
        }

        private TrainedModel LoadModel()
        {
            var weightsDirectory = Path.Combine(AppContext.BaseDirectory, "weights");
            var weightPath = Path.Combine(weightsDirectory, "aiweights.json");
            if (!File.Exists(weightPath))
            {
                weightPath = Path.Combine(AppContext.BaseDirectory, "aiweights.json");
            }
            if (!File.Exists(weightPath))
            {
                return null;
            }
            var json = File.ReadAllText(weightPath);
            return JsonSerializer.Deserialize<TrainedModel>(json);
        }

    }
}