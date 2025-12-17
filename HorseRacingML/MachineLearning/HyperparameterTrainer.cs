using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.NumPy;
using static System.Runtime.InteropServices.JavaScript.JSType;
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
        private readonly object _masterDatasetLock = new();
        private TrainingDataset? _cachedMasterDataset;
        private readonly ConcurrentDictionary<(int FoldIndex, int FoldCount, int ModelType), object> _foldCache = new();
        private readonly string _modelPath;
        public HyperparameterTrainer(IConfiguration configuration)
        {
            if (configuration is null)
            {
                throw new ArgumentNullException(nameof(configuration));
            }

            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
            _winRateAlpha = configuration.GetValue<float>("WinRateAlpha", 1f);
            _winRateBeta = configuration.GetValue<float>("WinRateBeta", 2f);
            var modelPath = configuration["ML:ModelPath"] ?? throw new InvalidOperationException("AI model path not configured.");
            _modelPath = Path.IsPathRooted(modelPath) ? modelPath : Path.Combine(AppContext.BaseDirectory, modelPath);
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
            // Added to support Hybrid Model composition
            public TrainedModel? Model { get; init; }
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
        public TrainingDataset EnsureMasterDatasetLoaded(bool forceReload = false)
        {
            lock (_masterDatasetLock)
            {
                if (_cachedMasterDataset != null && !forceReload)
                {
                    return _cachedMasterDataset;
                }
                _foldCache.Clear();
                Console.WriteLine("[AI] Loading and preparing master dataset for custom training queue...");
                // 1. Prepare raw dataset (Stage 1: SQL Load)
                // includeIdentifiers: true is required to track race IDs for caching
                var rawDataset = PrepareDataset(includeRaceIds: null, stateRaceWhitelist: null, includeIdentifiers: true, applyRepositoryBackfills: false);
                Console.WriteLine($"[AI] Raw dataset loaded with {rawDataset.Races.Count} races.");

                // 2. Create Master TrainingDataset (Stage 2: Metadata & Normalization Prep)
                var masterDataset = CreateMasterDataset(rawDataset, includeIdentifiers: true);

                // 3. Pre-encode features (Stage 3: Vector Encoding)
                PreEncodeFeatures(masterDataset);

                _cachedMasterDataset = masterDataset;
                return _cachedMasterDataset;
            }
        }

        private static void NormalizeBatchSize(MLParameter param)
        {
            if (param is null)
            {
                throw new ArgumentNullException(nameof(param));
            }

            if (param.ModelType == 1) // LightGBM ignores batch size
            {
                return;
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
            var prepared = PrepareDataset(includeRaceIds: null, stateRaceWhitelist: null, includeIdentifiers: includeIdentifiers, applyRepositoryBackfills: false);
            var emptyValidation = new PreparedDataset(new List<PreparedRace>());
            return BuildTrainingDataset(prepared, emptyValidation, includeIdentifiers);
        }

        // Overload referenced by HyperparameterController (likely used for creating datasets from a subset of IDs)
        public TrainingDataset LoadTrainingDataset(ISet<int> raceIds, ISet<int> validationRaceIds, bool includeIdentifiers = false)
        {
            var trainPrepared = PrepareDataset(ToNullableSet(raceIds), null, includeIdentifiers, applyRepositoryBackfills: false);
            var valPrepared = PrepareDataset(ToNullableSet(validationRaceIds), null, includeIdentifiers, applyRepositoryBackfills: false);
            return BuildTrainingDataset(trainPrepared, valPrepared, includeIdentifiers);
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

            var featureKeys = model.Metadata.Keys;
            var featureDimensions = model.Metadata.FeatureDimensions;
            var stringMaps = model.Metadata.StringMaps;

            var validationRaces = EncodeRaces(validationPrepared.Races, featureKeys, featureDimensions, stringMaps);

            var signature = BuildRaceSignature(validationPrepared.Races);
            var normalizationKey = BuildNormalizationCacheKey(includeIdentifiers, signature);

            return new TrainingDataset(new List<RaceExample>(), validationRaces, featureKeys, featureDimensions, stringMaps, model.Normalization, normalizationKey);
        }

        public TrainingResult Evaluate(TrainingDataset dataset)
        {
            // Placeholder for missing Evaluate method. 
            // Logic: Load model, predict on dataset, return metrics.
            var model = LoadModel();
            if (model == null) return new TrainingResult();
            // Minimal impl: just return empty if real eval logic is complex and missing.
            // Or call AIOddsCalculator logic... but this returns TrainingResult.
            return new TrainingResult();
        }

        public bool IsModelPersisted()
        {
            return File.Exists(_modelPath);
        }

        private TrainedModel? LoadModel()
        {
            if (!File.Exists(_modelPath)) return null;
            try
            {
                var json = File.ReadAllText(_modelPath);
                return JsonSerializer.Deserialize<TrainedModel>(json);
            }
            catch
            {
                return null;
            }
        }

        public HyperparameterSummary? LoadPersistedHyperparameters()
        {
            var model = LoadModel();
            return model?.Hyperparameters;
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
        public void PreEncodeFeatures(TrainingDataset dataset)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));

            Console.WriteLine($"[AI] Pre-encoding features for {dataset.Races.Count} races...");
            var allRunners = dataset.Races.SelectMany(r => r.Runners).ToList();
            int featureCount = dataset.FeatureCount;
            var featureKeys = dataset.FeatureKeys;
            var featureDims = dataset.FeatureDimensions;
            var stringMaps = dataset.StringMaps;

            Parallel.ForEach(allRunners, runner =>
            {
                runner.EncodedFeatures = EncodeFeatureVector(
                    runner.Features,
                    featureKeys,
                    featureDims,
                    stringMaps,
                    featureCount);
            });
            Console.WriteLine("[AI] Pre-encoding complete.");
        }

        public TrainingDataset CreateMasterDataset(PreparedDataset prepared, bool includeIdentifiers = false)
        {
            var emptyValidation = new PreparedDataset(new List<PreparedRace>());
            return BuildTrainingDataset(prepared, emptyValidation, includeIdentifiers);
        }

        public TrainingDataset BuildTrainingDataset(
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

        private static string BuildRaceSignature(IEnumerable<RaceExample> races)
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
                           .Where(x => x.idx < valStart)
                           .Select(x => x.race.RaceId)
                           .ToHashSet();

            if (trainRaceIds.Count == 0)
            {
                // In time-series validation, the first fold often has no historical data to train on.
                // In this case, we cannot train, so we return a dummy result to allow the process to continue
                // (e.g., if iterating through folds).
                Console.WriteLine($"[TrainAI] Fold {foldIndex} resulted in an empty training set (likely the first chronological block). Skipping training for this fold.");
                return new TrainingResult();
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
        public class LightGbmInput
        {
            [VectorType]
            public float[] Features { get; set; } = Array.Empty<float>();
            public bool Label { get; set; }
        }

        public class LightGbmOutput
        {
            public float Score { get; set; }
            public float Probability { get; set; }
        }

        private TrainingResult TrainLightGbm(MLParameter param, int foldIndex, int foldCount, TrainingDataset dataset, bool persistWeights = true)
        {
            var mlContext = new MLContext(seed: 42);

            var trainExamples = dataset.TrainingRaces
                .SelectMany(r => r.Runners)
                .ToList();
            var valExamples = dataset.ValidationRaces
                .SelectMany(r => r.Runners)
                .ToList();

            int featureCount = dataset.FeatureCount;
            // Note: LightGBM handles non-normalized data well, so we skip explicit normalization here
            // but we still encode features to get float vectors.
            var trainFeatures = trainExamples.AsParallel().AsOrdered()
                .Select(r => r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount))
                .ToList();
            var valFeatures = valExamples.AsParallel().AsOrdered()
                .Select(r => r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount))
                .ToList();

            var trainData = trainFeatures.Zip(trainExamples, (f, r) => new LightGbmInput { Features = f, Label = r.Label == 1f }).ToList();
            var valData = valFeatures.Zip(valExamples, (f, r) => new LightGbmInput { Features = f, Label = r.Label == 1f }).ToList();

            // Define schema explicitly to handle dynamic feature count as a fixed-size vector
            var schemaDef = SchemaDefinition.Create(typeof(LightGbmInput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);

            var trainDataView = mlContext.Data.LoadFromEnumerable(trainData, schemaDef);
            var valDataView = mlContext.Data.LoadFromEnumerable(valData, schemaDef);

            var pipeline = mlContext.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                NumberOfLeaves = param.LgbmLeaves > 1 ? param.LgbmLeaves.Value : 31,
                MinimumExampleCountPerLeaf = param.LgbmMinDataInLeaf > 0 ? param.LgbmMinDataInLeaf.Value : 20,
                LearningRate = param.LearningRate > 0 ? param.LearningRate : 0.1,
                NumberOfIterations = param.Epochs > 0 ? param.Epochs : 100,
            });

            Console.WriteLine("[LightGBM] Training model...");
            var model = pipeline.Fit(trainDataView);

            var trainPredictions = model.Transform(trainDataView);
            var valPredictions = model.Transform(valDataView);

            var trainMetrics = mlContext.BinaryClassification.Evaluate(trainPredictions, labelColumnName: "Label");
            var valMetrics = mlContext.BinaryClassification.Evaluate(valPredictions, labelColumnName: "Label");

            var trainProbs = trainPredictions.GetColumn<float>("Probability").ToArray();
            var valProbs = valPredictions.GetColumn<float>("Probability").ToArray();

            var trainLabels = trainData.Select(x => x.Label ? 1f : 0f).ToArray();
            var valLabels = valData.Select(x => x.Label ? 1f : 0f).ToArray();
            var trainRaceIds = trainExamples.Select(r => r.RaceId).ToArray();
            var valRaceIds = valExamples.Select(r => r.RaceId).ToArray();

            double trainAcc = ComputeWinnerAccuracy(trainRaceIds, trainProbs, trainLabels);
            double valAcc = ComputeWinnerAccuracy(valRaceIds, valProbs, valLabels);
            double trainBrier = ComputeBrier(trainProbs, trainLabels);
            double valBrier = ComputeBrier(valProbs, valLabels);

            if (persistWeights)
            {
                var modelPath = Path.ChangeExtension(_modelPath, ".zip"); // ML.NET saves as zip
                mlContext.Model.Save(model, trainDataView.Schema, modelPath);

                // We also need to save the metadata so we know how to encode features for inference
                var metadataModel = new TrainedModel
                {
                    Metadata = new FeatureMetadata
                    {
                        Keys = new List<string>(dataset.FeatureKeys),
                        FeatureDimensions = new Dictionary<string, int>(dataset.FeatureDimensions),
                        StringMaps = dataset.StringMaps.ToDictionary(
                            kvp => kvp.Key,
                            kvp => new Dictionary<string, int>(kvp.Value))
                    },
                    Hyperparameters = new HyperparameterSummary
                    {
                        ModelType = 1, // LightGBM
                        TrainedAtUtc = DateTime.UtcNow
                    }
                };
                var options = new JsonSerializerOptions { WriteIndented = true };
                var serialized = JsonSerializer.Serialize(metadataModel, options);
                File.WriteAllText(_modelPath, serialized); // Overwrite the main json with metadata
            }

            Console.WriteLine($"[LightGBM] Training complete. Train Acc: {trainAcc:P2}, Val Acc: {valAcc:P2}");

            return new TrainingResult
            {
                TrainAccuracy = trainAcc,
                TrainLoss = trainMetrics.LogLoss, // LogLoss is analogous to cross-entropy
                TrainBrier = trainBrier,
                ValidationAccuracy = valAcc,
                ValidationLoss = valMetrics.LogLoss,
                ValidationBrier = valBrier,
                TrainingPredictions = Array.AsReadOnly(trainProbs),
                TrainingLabels = Array.AsReadOnly(trainLabels),
                TrainingRaceIds = Array.AsReadOnly(trainRaceIds),
                ValidationPredictions = Array.AsReadOnly(valProbs),
                ValidationLabels = Array.AsReadOnly(valLabels),
                ValidationRaceIds = Array.AsReadOnly(valRaceIds),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(valExamples)
            };
        }

        public TrainingResult Train(MLParameter param, int foldIndex, int foldCount, TrainingDataset dataset, bool persistWeights = true)
        {
            if (param.ModelType == 1) // LightGBM
            {
                NormalizeBatchSize(param); // Still valid for consistency checks
                return TrainLightGbm(param, foldIndex, foldCount, dataset, persistWeights);
            }
            if (param.ModelType == 2) // Hybrid
            {
                NormalizeBatchSize(param);
                return TrainHybrid(param, foldIndex, foldCount, dataset, persistWeights);
            }

            // Default to Neural Network (TensorFlow)
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
            var trainFeatures = trainExamples.AsParallel().AsOrdered().Select(r => r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount)).ToList();
            var trainLabels = trainExamples.Select(r => r.Label).ToArray();
            var trainRaceIds = trainExamples.Select(r => r.RaceId).ToArray();

            var valFeatures = valExamples.AsParallel().AsOrdered().Select(r => r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount)).ToList();
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

            using var trainFeatureTensor = Tensorflow.NumPy.np.array(
                BuildFeatureMatrix(trainFeatures, featureCount),
                dtype: tf.float32);
            using var trainLabelTensor = Tensorflow.NumPy.np.array(
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
            // Note: In newer TensorFlow.NET versions, Graph and its context may not implement IDisposable directly.
            // We remove 'using' to fix CS1674, relying on internal resource management or GC.
            var graph = tf.Graph();
            var graphScope = graph.as_default();

            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "y");
            Tensor layer = x;
            int inputDim = featureCount;
            var hiddenWeightVars = new List<ResourceVariable>();
            var hiddenBiasVars = new List<ResourceVariable>();
            for (int i = 0; i < param.Layers; i++)
            {
                var w = tf.Variable(tf.random.normal((inputDim, param.Units ?? 10)), name: $"w{i}");
                var b = tf.Variable(tf.zeros(param.Units ?? 10), name: $"b{i}");
                hiddenWeightVars.Add(w);
                hiddenBiasVars.Add(b);
                layer = tf.nn.relu(tf.matmul(layer, w) + b);
                if (param.Dropout > 0)
                {
                    layer = tf.nn.dropout(layer, rate: (float)param.Dropout);
                }
                inputDim = param.Units ?? 10;
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

                if (count == 0) return (0, 0);

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

            // Shuffle helper reused (renamed to avoid local function shadowing error)
            static void ShuffleIndices(int[] values, Random random)
            {
                for (int i = values.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (values[i], values[j]) = (values[j], values[i]);
                }
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
            int patienceCounter = 0;
            string? bestModelPath = null;
            List<FeatureCorrelation> featureCorrelations = new();
            HyperparameterStarted(param, trainLabels.Length, valLabels.Length, featureCount);

            try
            {
                int n = trainLabels.Length;
                if (n > 0)
                {
                    var indices = Enumerable.Range(0, n).ToArray();
                    var batchIndexBuffer = new int[param.BatchSize ?? 32];
                    for (int epoch = 0; epoch < param.Epochs; epoch++)
                    {
                        ShuffleIndices(indices, rnd);
                        for (int start = 0; start < n; start += (param.BatchSize ?? 32))
                        {
                            int batchCount = Math.Min(param.BatchSize ?? 32, n - start);
                            var batchIndices = indices.AsSpan(start, batchCount);
                            batchIndices.CopyTo(batchIndexBuffer);
                            var batchIndexArray = batchIndexBuffer.AsSpan(0, batchCount).ToArray();
                            using var batchIndexTensor = Tensorflow.NumPy.np.array(batchIndexArray, dtype: tf.int32);
                            using var batchFeatures = trainFeatureTensor[batchIndexTensor];
                            using var batchLabels = trainLabelTensor[batchIndexTensor];

                            sess.run(optimizer, new FeedItem(x, batchFeatures), new FeedItem(y, batchLabels));
                        }

                        (var epochLoss, var epochFocalLoss) = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, epochPredBuffer, n);
                        var epochAcc = ComputeWinnerAccuracy(trainRaceIds, epochPredBuffer, trainLabels);

                        if (hasValidationExamples)
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
                                // Save best weights in memory
                                bestWeights.Clear();
                                foreach (var v in allVariables) bestWeights.Add(sess.run(v));
                            }
                            else
                            {
                                patienceCounter++;
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
                            // Fix for InvalidCastException: Explicitly convert NDArray to constant Tensor
                            var weightData = bestWeights[i].ToArray<float>();
                            var shape = bestWeights[i].shape;
                            var tensor = tf.constant(weightData, shape: shape);
                            sess.run(allVariables[i].assign(tensor));
                        }
                    }
                    (trainLoss, trainFocalLoss) = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, trainPreds, n);
                    trainBrier = ComputeBrier(trainPreds, trainLabels); // ToArray if List

                    if (hasValidationExamples)
                    {
                        (valLoss, valFocalLoss) = ComputeDatasetMetrics(valFeatureTensor, valLabelTensor, valPreds, valLabels.Length);
                        valBrier = ComputeBrier(valPreds, valLabels);
                    }

                    trainAcc = ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels);
                    valAcc = hasValidationExamples ? ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels) : 0;
                }
            }
            catch (Exception ex)
            {
                HyperparameterFailed(param, ex);
                throw;
            }
            finally
            {
                if (bestModelPath != null && File.Exists(bestModelPath))
                {
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

            var trainedModel = new TrainedModel
            {
                HiddenLayers = hiddenLayers,
                OutputLayer = outputLayer,
                // Metadata/Normalization handled below or by caller in Hybrid mode
            };

            // Feature metadata is not passed in the fold data explicitly, we need it if we want to save metadata
            // For now, assuming standard metadata usage if we had it. 
            // Since this optimized path is mostly for search (persistWeights=false), we skip saving full metadata if not available.

            // Re-fetch master dataset just for metadata if persisting
            // Warning: If we are in Hybrid training, EnsureMasterDatasetLoaded returns the FULL dataset,
            // but the NN model was trained on a SUBSET (from 'data' argument).
            // We must use the keys from 'data' (if available?) or we need to pass metadata in.
            // TensorFlowFoldData does not contain metadata.
            // However, in TrainHybrid, we call this with persistWeights=true/false. 
            // If we are in TrainHybrid, we handle persistence manually OR we rely on the override.
            // If modelPathOverride is set, we assume we want to save there.

            if (persistWeights)
            {
                var targetPath = _modelPath;

                // var dataset = EnsureMasterDatasetLoaded(); // Already available as parameter 'dataset'

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

                // Populate model properties for standard save
                trainedModel.Metadata = new FeatureMetadata
                {
                    Keys = new List<string>(featureKeys),
                    FeatureDimensions = new Dictionary<string, int>(featureDims),
                    StringMaps = stringMaps.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new Dictionary<string, int>(kvp.Value))
                };
                trainedModel.Normalization = new NormalizationParameters
                {
                    Mean = (float[])dataset.Normalization.Mean.Clone(),
                    StdDev = (float[])dataset.Normalization.StdDev.Clone()
                };
                trainedModel.Hyperparameters = new HyperparameterSummary
                {
                    Layers = param.Layers ?? 0,
                    Units = param.Units ?? 10,
                    Dropout = param.Dropout ?? 0,
                    LearningRate = param.LearningRate,
                    Epochs = param.Epochs,
                    BatchSize = param.BatchSize ?? 32,
                    Folds = param.Folds,
                    Fold = param.Fold,
                    TrainedAtUtc = trainedAtUtc
                };

                var weightsDirectory = Path.GetDirectoryName(targetPath);
                if (weightsDirectory != null)
                {
                    Directory.CreateDirectory(weightsDirectory);
                }

                var options = new JsonSerializerOptions { WriteIndented = true };
                var serialized = JsonSerializer.Serialize(trainedModel, options);
                File.WriteAllText(targetPath, serialized);
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
                TrainingLabels = new ReadOnlyCollection<float>(trainLabels.ToList()),
                TrainingRaceIds = new ReadOnlyCollection<int>(trainRaceIds.ToList()),
                ValidationPredictions = Array.AsReadOnly(valPreds),
                ValidationLabels = new ReadOnlyCollection<float>(valLabels.ToList()),
                ValidationRaceIds = new ReadOnlyCollection<int>(valRaceIds.ToList()),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(valExamples.ToList()),
                FeatureCorrelations = new ReadOnlyCollection<FeatureCorrelation>(featureCorrelations),
                Model = trainedModel // Return the model object for potential external usage
            };
        }
        public TrainingResult TrainLightGbmOptimized(MLParameter param, MLContext mlContext, LightGbmFoldData data, bool persistWeights = false, string? modelPathOverride = null)
        {
            if (data.IsEmpty)
            {
                return new TrainingResult();
            }
            // Ensure we have actual data rows to prevent AccessViolation in LightGBM native
            if (data.TrainData.GetRowCount() == 0)
            {
                Console.WriteLine("[LightGBM] Training skipped because training data view has 0 rows.");
                return new TrainingResult();
            }
            NormalizeBatchSize(param);

            var options = new LightGbmBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                NumberOfLeaves = param.LgbmLeaves > 1 ? param.LgbmLeaves.Value : 31,
                MinimumExampleCountPerLeaf = param.LgbmMinDataInLeaf > 0 ? param.LgbmMinDataInLeaf.Value : 20,
                LearningRate = param.LearningRate > 0 ? param.LearningRate : 0.1,
                NumberOfIterations = param.Epochs > 0 ? param.Epochs : 100,
            };

            if (param.Threads.HasValue && param.Threads.Value > 0)
            {
                options.NumberOfThreads = param.Threads.Value;
            }

            var pipeline = mlContext.BinaryClassification.Trainers.LightGbm(options);

            var model = pipeline.Fit(data.TrainData);

            var trainPredictions = model.Transform(data.TrainData);
            var valPredictions = model.Transform(data.ValData);

            var trainMetrics = mlContext.BinaryClassification.Evaluate(trainPredictions, labelColumnName: "Label");
            var valMetrics = mlContext.BinaryClassification.Evaluate(valPredictions, labelColumnName: "Label");

            var trainProbs = trainPredictions.GetColumn<float>("Probability").ToArray();
            var valProbs = valPredictions.GetColumn<float>("Probability").ToArray();

            var trainLabelsArr = data.TrainLabels.ToArray();
            var valLabelsArr = data.ValLabels.ToArray();
            var trainRaceIdsArr = data.TrainRaceIds.ToArray();
            var valRaceIdsArr = data.ValRaceIds.ToArray();

            double trainAcc = ComputeWinnerAccuracy(trainRaceIdsArr, trainProbs, trainLabelsArr);
            double valAcc = ComputeWinnerAccuracy(valRaceIdsArr, valProbs, valLabelsArr);
            double trainBrier = ComputeBrier(trainProbs, trainLabelsArr);
            double valBrier = ComputeBrier(valProbs, valLabelsArr);
            if (persistWeights)
            {
                var targetPath = !string.IsNullOrEmpty(modelPathOverride)
                    ? (Path.IsPathRooted(modelPathOverride) ? modelPathOverride : Path.Combine(AppContext.BaseDirectory, modelPathOverride))
                    : _modelPath;

                var modelPath = Path.ChangeExtension(targetPath, ".zip");
                mlContext.Model.Save(model, data.TrainData.Schema, modelPath);

                // Only save JSON metadata if standard flow (no override)
                // In Hybrid flow, we handle JSON creation in TrainHybrid to avoid race conditions/overwrites
                if (string.IsNullOrEmpty(modelPathOverride))
                {
                    var dataset = EnsureMasterDatasetLoaded();
                    var metadataModel = new TrainedModel
                    {
                        Metadata = new FeatureMetadata
                        {
                            Keys = new List<string>(dataset.FeatureKeys),
                            FeatureDimensions = new Dictionary<string, int>(dataset.FeatureDimensions),
                            StringMaps = dataset.StringMaps.ToDictionary(
                                kvp => kvp.Key,
                                kvp => new Dictionary<string, int>(kvp.Value))
                        },
                        Hyperparameters = new HyperparameterSummary
                        {
                            ModelType = 1, // LightGBM
                            TrainedAtUtc = DateTime.UtcNow
                        }
                    };
                    var optionsJson = new JsonSerializerOptions { WriteIndented = true };
                    var serialized = JsonSerializer.Serialize(metadataModel, optionsJson);
                    File.WriteAllText(targetPath, serialized);
                }
            }
            return new TrainingResult
            {
                TrainAccuracy = trainAcc,
                TrainLoss = trainMetrics.LogLoss,
                TrainBrier = trainBrier,
                ValidationAccuracy = valAcc,
                ValidationLoss = valMetrics.LogLoss,
                ValidationBrier = valBrier,
                TrainingPredictions = Array.AsReadOnly(trainProbs),
                TrainingLabels = Array.AsReadOnly(trainLabelsArr),
                TrainingRaceIds = Array.AsReadOnly(trainRaceIdsArr),
                ValidationPredictions = Array.AsReadOnly(valProbs),
                ValidationLabels = Array.AsReadOnly(valLabelsArr),
                ValidationRaceIds = Array.AsReadOnly(valRaceIdsArr),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(data.ValExamples.ToList())
            };
        }

        private (List<string> nnKeys, List<string> lgbmKeys) PartitionFeatures(List<string> featureKeys, Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            var nnKeys = new List<string>();
            var lgbmKeys = new List<string>();

            // Heuristic Partitioning
            foreach (var key in featureKeys)
            {
                // Categorical features go to LightGBM
                if (stringMaps.ContainsKey(key))
                {
                    lgbmKeys.Add(key);
                    continue;
                }

                // Integer/ID/Count features (if not mapped) typically go to LGBM too
                // Or if the key implies a categorical nature
                if (key.Contains("Id") || key.EndsWith("Code") || key.Contains("Draw") || key.Contains("Position") || key.Contains("Rank") || key.Contains("Count"))
                {
                    lgbmKeys.Add(key);
                    continue;
                }

                // Default continuous to Neural Network
                // Includes: Price, Prob, Speed, Lengths, Weight, Time, Pct, Diff, StdDev, Avg, Norm
                nnKeys.Add(key);
            }

            return (nnKeys, lgbmKeys);
        }

        private TrainingDataset CreateSubsetDataset(TrainingDataset master, List<string> keys)
        {
            // We need to create a new TrainingDataset with only the specified keys
            // But we can't just pass the original RaceExample/RunnerExample objects because they have 'EncodedFeatures'
            // which are pre-calculated for the FULL set of keys.
            // We must create fresh RunnerExample objects with EncodedFeatures = null, so they get re-calculated.

            var newDimensions = new Dictionary<string, int>();
            foreach (var key in keys)
            {
                if (master.FeatureDimensions.TryGetValue(key, out var dim))
                {
                    newDimensions[key] = dim;
                }
            }

            // Helper to clone races
            List<RaceExample> CloneRaces(List<RaceExample> source)
            {
                var result = new List<RaceExample>(source.Count);
                foreach (var race in source)
                {
                    var newRunners = new List<RunnerExample>(race.Runners.Count);
                    foreach (var runner in race.Runners)
                    {
                        // Note: Features dictionary is shared (read-only), EncodedFeatures is reset to null
                        newRunners.Add(new RunnerExample(runner.RaceId, runner.Features, runner.Label, runner.HorseId, runner.HorseName, runner.StartingPriceDecimal));
                    }
                    result.Add(new RaceExample(race.RaceId, newRunners));
                }
                return result;
            }

            var newTrainRaces = CloneRaces(master.TrainingRaces);
            var newValRaces = CloneRaces(master.ValidationRaces);

            var signature = BuildRaceSignature(newTrainRaces.Concat(newValRaces));
            var normalizationKey = BuildNormalizationCacheKey(false, signature + "|" + string.Join(",", keys));

            // We need new normalization parameters for this subset.
            // Since we are likely in TrainHybrid, we will let the individual Train methods calculate normalization if needed.
            // Or we can pre-calculate it here? 
            // Better to let TrainTensorFlowOptimized handle it via PrepareTensorFlowFold which calculates it for the fold.
            // But we need to pass a NormalizationParameters object to the constructor.
            var norm = new NormalizationParameters { Mean = new float[newDimensions.Values.Sum()], StdDev = new float[newDimensions.Values.Sum()] };

            return new TrainingDataset(
                newTrainRaces,
                newValRaces,
                keys,
                newDimensions,
                master.StringMaps,
                norm,
                normalizationKey);
        }

        private TrainingResult TrainHybrid(MLParameter param, int foldIndex, int foldCount, TrainingDataset fullDataset, bool persistWeights)
        {
            Console.WriteLine("[Hybrid] Starting Hybrid Training (NN + LightGBM)...");

            // 1. Partition Features
            var (nnKeys, lgbmKeys) = PartitionFeatures(fullDataset.FeatureKeys, fullDataset.StringMaps);
            Console.WriteLine($"[Hybrid] Features partitioned: {nnKeys.Count} for Neural Network, {lgbmKeys.Count} for LightGBM.");

            // 2. Create Datasets
            var nnDataset = CreateSubsetDataset(fullDataset, nnKeys);
            var lgbmDataset = CreateSubsetDataset(fullDataset, lgbmKeys);

            var hybridJsonPath = Path.Combine(AppContext.BaseDirectory, "NNLightGBM.json");

            // 3. Train LightGBM first (saves .zip)
            Console.WriteLine("[Hybrid] Training LightGBM component...");
            var mlContext = CreateLightGbmContext();
            var lgbmFoldData = PrepareLightGbmFold(mlContext, lgbmDataset, foldIndex, foldCount);
            // We tell it to persist weights to "NNLightGBM.json" (which triggers .zip save), but it won't write the JSON due to the override check.
            var lgbmParam = new MLParameter
            {
                ModelType = 1, // Explicitly set to LightGBM so validation (e.g. BatchSize) is handled correctly
                RunDate = param.RunDate,
                Folds = param.Folds,
                Fold = param.Fold,
                TrainFinalFoldOnly = param.TrainFinalFoldOnly,
                Threads = param.Threads,
                LgbmLeaves = param.LgbmLeaves,
                LgbmMinDataInLeaf = param.LgbmMinDataInLeaf,
                LgbmMaxDepth = param.LgbmMaxDepth,
                // Use explicit LGBM values if present, otherwise fallback to main (NN) values
                LearningRate = param.LgbmLearningRate ?? param.LearningRate,
                Epochs = param.LgbmEpochs ?? param.Epochs
            };

            // We tell it to persist weights to "NNLightGBM.json" (which triggers .zip save), but it won't write the JSON due to the override check.
            var lgbmResult = TrainLightGbmOptimized(lgbmParam, mlContext, lgbmFoldData, persistWeights: persistWeights, modelPathOverride: hybridJsonPath);
            // 4. Train Neural Network (needs to return weights, but TrainTensorFlowOptimized is void of that return)
            Console.WriteLine("[Hybrid] Training Neural Network component...");
            var nnFoldData = PrepareTensorFlowFold(nnDataset, foldIndex, foldCount);

            // We need the model object back. TrainTensorFlowOptimized has been updated to return it in .Model property.
            var nnResult = TrainTensorFlowOptimized(param, nnFoldData, persistWeights: false);

            // 5. Combine Results (Validation Only for now, as that's what TrainingResult returns)
            var valPreds = new float[nnResult.ValidationPredictions.Count];
            // Ensure lengths match
            if (nnResult.ValidationPredictions.Count == lgbmResult.ValidationPredictions.Count)
            {
                for (int i = 0; i < valPreds.Length; i++)
                {
                    valPreds[i] = (nnResult.ValidationPredictions[i] + lgbmResult.ValidationPredictions[i]) / 2f;
                }
            }
            else
            {
                Console.WriteLine("[Hybrid] Warning: Validation prediction counts mismatch between models.");
            }

            var valLabels = nnResult.ValidationLabels; // Should be identical
            var valRaceIds = nnResult.ValidationRaceIds;

            // Recalculate metrics
            double valAcc = ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels);
            double valBrier = ComputeBrier(valPreds, valLabels.ToArray());

            Console.WriteLine($"[Hybrid] Combined Validation Accuracy: {valAcc:P2}");

            // 6. Persist Weights
            if (persistWeights && nnResult.Model != null)
            {
                var trainedModel = nnResult.Model;

                // Populate NN Metadata
                trainedModel.Metadata = new FeatureMetadata
                {
                    Keys = new List<string>(nnDataset.FeatureKeys),
                    FeatureDimensions = new Dictionary<string, int>(nnDataset.FeatureDimensions),
                    StringMaps = nnDataset.StringMaps.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new Dictionary<string, int>(kvp.Value))
                };
                trainedModel.Normalization = nnFoldData.Normalization; // Use the one computed during training

                // Populate Hybrid LGBM Metadata
                trainedModel.HybridLgbmMetadata = new FeatureMetadata
                {
                    Keys = new List<string>(lgbmDataset.FeatureKeys),
                    FeatureDimensions = new Dictionary<string, int>(lgbmDataset.FeatureDimensions),
                    StringMaps = lgbmDataset.StringMaps.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new Dictionary<string, int>(kvp.Value))
                };

                // Update Hyperparameters
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
                trainedModel.Hyperparameters = new HyperparameterSummary
                {
                    Layers = param.Layers ?? 0,
                    Units = param.Units ?? 10,
                    Dropout = param.Dropout ?? 0,
                    LearningRate = param.LearningRate,
                    Epochs = param.Epochs,
                    BatchSize = param.BatchSize ?? 32,
                    Folds = param.Folds,
                    Fold = param.Fold,
                    TrainedAtUtc = trainedAtUtc,
                    ModelType = 2 // Hybrid
                };

                var options = new JsonSerializerOptions { WriteIndented = true };
                var serialized = JsonSerializer.Serialize(trainedModel, options);
                File.WriteAllText(hybridJsonPath, serialized);
                Console.WriteLine($"[Hybrid] Model saved to {hybridJsonPath} and .zip");
            }

            return new TrainingResult
            {
                TrainAccuracy = (nnResult.TrainAccuracy + lgbmResult.TrainAccuracy) / 2, // Approximate
                ValidationAccuracy = valAcc,
                ValidationLoss = (nnResult.ValidationLoss + lgbmResult.ValidationLoss) / 2,
                ValidationBrier = valBrier,
                ValidationPredictions = Array.AsReadOnly(valPreds),
                ValidationLabels = valLabels,
                ValidationRaceIds = valRaceIds,
                ValidationExamples = nnResult.ValidationExamples
            };
        }
        public class TensorFlowFoldData
        {
            public float[,] TrainFeatureMatrix { get; init; }
            public float[,] TrainLabelMatrix { get; init; }
            public float[,] ValFeatureMatrix { get; init; }
            public float[,] ValLabelMatrix { get; init; }
            public IReadOnlyList<float> TrainLabels { get; init; }
            public IReadOnlyList<float> ValLabels { get; init; }
            public IReadOnlyList<int> TrainRaceIds { get; init; }
            public IReadOnlyList<int> ValRaceIds { get; init; }
            public IReadOnlyList<RunnerExample> ValExamples { get; init; }
            public NormalizationParameters Normalization { get; init; }
            public bool IsEmpty { get; init; }
        }

        public object GetPreparedFold(int foldIndex, int foldCount, int modelType, MLContext mlContext = null)
        {
            var key = (foldIndex, foldCount, modelType);
            if (_foldCache.TryGetValue(key, out var cached))
            {
                return cached;
            }

            var dataset = EnsureMasterDatasetLoaded();

            if (modelType == 1) // LightGBM
            {
                if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));
                var data = PrepareLightGbmFold(mlContext, dataset, foldIndex, foldCount);
                _foldCache[key] = data;
                return data;
            }
            else // TensorFlow
            {
                var data = PrepareTensorFlowFold(dataset, foldIndex, foldCount);
                _foldCache[key] = data;
                return data;
            }
        }

        private TensorFlowFoldData PrepareTensorFlowFold(TrainingDataset dataset, int foldIndex, int foldCount)
        {
            int totalRaces = dataset.Races.Count;
            int foldSize = totalRaces / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalRaces : valStart + foldSize;

            var trainRaceIds = dataset.Races
                           .Select((race, idx) => new { race, idx })
                           .Where(x => x.idx < valStart)
                           .Select(x => x.race.RaceId)
                           .ToHashSet();

            if (trainRaceIds.Count == 0) return new TensorFlowFoldData { IsEmpty = true };

            var validationRaceIds = dataset.Races.Skip(valStart).Take(valEnd - valStart).Select(r => r.RaceId).ToHashSet();

            var trainExamples = dataset.Races.Where(r => trainRaceIds.Contains(r.RaceId)).SelectMany(r => r.Runners).ToList();
            var valExamples = dataset.Races.Where(r => validationRaceIds.Contains(r.RaceId)).SelectMany(r => r.Runners).ToList();
            int featureCount = dataset.FeatureCount;

            // Deep clone features for training set to avoid in-place corruption of the master dataset
            var trainFeatures = trainExamples.AsParallel().AsOrdered()
                .Select(r => (float[])(r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount)).Clone())
                .ToList();

            var valFeatures = valExamples.AsParallel().AsOrdered()
                .Select(r => (float[])(r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount)).Clone())
                .ToList();

            var means = new float[featureCount];
            var stdDevs = new float[featureCount];

            if (trainFeatures.Count > 0)
            {
                Parallel.For(0, featureCount, j =>
                {
                    double sum = 0;
                    foreach (var feature in trainFeatures) sum += feature[j];
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
                    if (stdDevs[j] == 0f) stdDevs[j] = 1f;
                });
            }
            else
            {
                for (int j = 0; j < featureCount; j++) stdDevs[j] = 1f;
            }

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

            float[,] BuildMatrix(List<float[]> source)
            {
                var matrix = new float[source.Count, featureCount];
                for (int i = 0; i < source.Count; i++)
                {
                    var row = source[i];
                    for (int j = 0; j < featureCount; j++) matrix[i, j] = row[j];
                }
                return matrix;
            }
            float[,] BuildLabelMatrix(List<RunnerExample> runners)
            {
                var matrix = new float[runners.Count, 1];
                for (int i = 0; i < runners.Count; i++) matrix[i, 0] = runners[i].Label;
                return matrix;
            }

            return new TensorFlowFoldData
            {
                TrainFeatureMatrix = BuildMatrix(trainFeatures),
                TrainLabelMatrix = BuildLabelMatrix(trainExamples),
                ValFeatureMatrix = BuildMatrix(valFeatures),
                ValLabelMatrix = BuildLabelMatrix(valExamples),
                TrainLabels = trainExamples.Select(r => r.Label).ToArray(),
                ValLabels = valExamples.Select(r => r.Label).ToArray(),
                TrainRaceIds = trainExamples.Select(r => r.RaceId).ToArray(),
                ValRaceIds = valExamples.Select(r => r.RaceId).ToArray(),
                ValExamples = valExamples,
                Normalization = new NormalizationParameters { Mean = means, StdDev = stdDevs },
                IsEmpty = false
            };
        }
        public class LightGbmFoldData
        {
            public IDataView TrainData { get; init; }
            public IDataView ValData { get; init; }
            public IReadOnlyList<float> ValLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> ValRaceIds { get; init; } = Array.Empty<int>();
            public IReadOnlyList<RunnerExample> ValExamples { get; init; } = Array.Empty<RunnerExample>();
            public IReadOnlyList<float> TrainLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> TrainRaceIds { get; init; } = Array.Empty<int>();
            public bool IsEmpty { get; init; }
        }

        public MLContext CreateLightGbmContext()
        {
            return new MLContext(seed: 42);
        }

        public LightGbmFoldData PrepareLightGbmFold(MLContext mlContext, TrainingDataset dataset, int foldIndex, int foldCount)
        {
            if (dataset is null) throw new ArgumentNullException(nameof(dataset));
            if (foldCount <= 0) throw new ArgumentOutOfRangeException(nameof(foldCount));
            if (foldIndex < 0 || foldIndex >= foldCount) throw new ArgumentOutOfRangeException(nameof(foldIndex));

            int totalRaces = dataset.Races.Count;
            int foldSize = totalRaces / foldCount;
            int valStart = foldIndex * foldSize;
            int valEnd = foldIndex == foldCount - 1 ? totalRaces : valStart + foldSize;

            var trainRaceIds = dataset.Races
                           .Select((race, idx) => new { race, idx })
                           .Where(x => x.idx < valStart)
                           .Select(x => x.race.RaceId)
                           .ToHashSet();

            if (trainRaceIds.Count == 0)
            {
                return new LightGbmFoldData { IsEmpty = true };
            }

            var validationRaceIds = dataset.Races
                .Skip(valStart)
                .Take(valEnd - valStart)
                .Select(r => r.RaceId)
                .ToHashSet();

            var trainingRaces = new List<RaceExample>();
            var validationRaces = new List<RaceExample>();

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

            var trainExamples = trainingRaces.SelectMany(r => r.Runners).ToList();
            var valExamples = validationRaces.SelectMany(r => r.Runners).ToList();

            int featureCount = dataset.FeatureCount;

            var trainFeatures = trainExamples.AsParallel().AsOrdered()
                .Select(r => r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount))
                .ToList();
            var valFeatures = valExamples.AsParallel().AsOrdered()
                .Select(r => r.EncodedFeatures ?? EncodeFeatureVector(r.Features, dataset.FeatureKeys, dataset.FeatureDimensions, dataset.StringMaps, featureCount))
                .ToList();

            var trainData = trainFeatures.Zip(trainExamples, (f, r) => new LightGbmInput { Features = f, Label = r.Label == 1f }).ToList();
            var valData = valFeatures.Zip(valExamples, (f, r) => new LightGbmInput { Features = f, Label = r.Label == 1f }).ToList();

            var schemaDef = SchemaDefinition.Create(typeof(LightGbmInput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);

            var trainDataView = mlContext.Data.LoadFromEnumerable(trainData, schemaDef);
            var valDataView = mlContext.Data.LoadFromEnumerable(valData, schemaDef);

            return new LightGbmFoldData
            {
                TrainData = trainDataView,
                ValData = valDataView,
                TrainLabels = trainData.Select(x => x.Label ? 1f : 0f).ToArray(),
                TrainRaceIds = trainExamples.Select(r => r.RaceId).ToArray(),
                ValLabels = valData.Select(x => x.Label ? 1f : 0f).ToArray(),
                ValRaceIds = valExamples.Select(r => r.RaceId).ToArray(),
                ValExamples = valExamples,
                IsEmpty = false
            };
        }
        public TrainingResult TrainTensorFlowOptimized(MLParameter param, TensorFlowFoldData data, bool persistWeights = true, string? modelPathOverride = null)
        {
            if (data.IsEmpty) return new TrainingResult();
            NormalizeBatchSize(param);
            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

            var trainFeatureTensor = Tensorflow.NumPy.np.array(data.TrainFeatureMatrix, dtype: tf.float32);
            var trainLabelTensor = Tensorflow.NumPy.np.array(data.TrainLabelMatrix, dtype: tf.float32);
            NDArray? valFeatureTensor = null;
            NDArray? valLabelTensor = null;
            bool hasValidationExamples = data.ValLabelMatrix.GetLength(0) > 0;

            if (hasValidationExamples)
            {
                valFeatureTensor = Tensorflow.NumPy.np.array(data.ValFeatureMatrix, dtype: tf.float32);
                valLabelTensor = Tensorflow.NumPy.np.array(data.ValLabelMatrix, dtype: tf.float32);
            }

            var graph = tf.Graph();
            var graphScope = graph.as_default();
            ConfigProto? config = null;
            if (param.Threads.HasValue && param.Threads.Value > 0)
            {
                config = new ConfigProto
                {
                    IntraOpParallelismThreads = param.Threads.Value,
                    InterOpParallelismThreads = param.Threads.Value
                };
            }
            int featureCount = data.TrainFeatureMatrix.GetLength(1);
            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "y");
            Tensor layer = x;
            int inputDim = featureCount;
            var hiddenWeightVars = new List<ResourceVariable>();
            var hiddenBiasVars = new List<ResourceVariable>();
            for (int i = 0; i < param.Layers; i++)
            {
                var w = tf.Variable(tf.random.normal((inputDim, param.Units ?? 10)), name: $"w{i}");
                var b = tf.Variable(tf.zeros(param.Units ?? 10), name: $"b{i}");
                hiddenWeightVars.Add(w);
                hiddenBiasVars.Add(b);
                layer = tf.nn.relu(tf.matmul(layer, w) + b);
                if (param.Dropout > 0)
                {
                    layer = tf.nn.dropout(layer, rate: (float)param.Dropout);
                }
                inputDim = param.Units ?? 10;
            }

            var wOut = tf.Variable(tf.random.normal((inputDim, 1)), name: "wOut");
            var bOut = tf.Variable(tf.zeros(1), name: "bOut");
            var logits = tf.matmul(layer, wOut) + bOut;
            var loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: y, logits: logits));
            var focalLoss = tf.reduce_mean(SigmoidFocalLoss(y, logits));
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(focalLoss);

            var prediction = tf.sigmoid(logits);
            var rnd = new Random();
            using var sess = tf.Session(graph, config);
            sess.run(tf.global_variables_initializer());
            var allVariables = hiddenWeightVars.Concat(hiddenBiasVars).Concat(new[] { wOut, bOut }).ToList();
            var bestWeights = new List<NDArray>();

            // Note: We use the normalization from the fold data
            var normalization = data.Normalization;
            // Only persist normalization if we are persisting weights (saving the model)
            if (persistWeights)
            {
                var normPath = Path.Combine(AppContext.BaseDirectory, "normalization.json");
                var normalizationJson = JsonSerializer.Serialize(normalization);
                WriteTextIfChanged(normPath, normalizationJson);
            }

            var trainPreds = new float[data.TrainLabels.Count];
            var valPreds = new float[data.ValLabels.Count];
            var epochPredBuffer = new float[data.TrainLabels.Count];

            (double loss, double focalLoss) ComputeDatasetMetrics(NDArray featureTensor, NDArray labelTensor, float[] preds, int count)
            {
                if (preds.Length != count)
                    throw new ArgumentException("Prediction buffer size must match example count.", nameof(preds));

                if (count == 0) return (0, 0);

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

            // Shuffle helper reused
            static void Shuffle(int[] values, Random random)
            {
                for (int i = values.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (values[i], values[j]) = (values[j], values[i]);
                }
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
            int patienceCounter = 0;
            string? bestModelPath = null;
            List<FeatureCorrelation> featureCorrelations = new();
            HyperparameterStarted(param, data.TrainLabels.Count, data.ValLabels.Count, featureCount);

            try
            {
                int n = data.TrainLabels.Count;
                if (n > 0)
                {
                    var indices = Enumerable.Range(0, n).ToArray();
                    var batchIndexBuffer = new int[param.BatchSize ?? 32];
                    for (int epoch = 0; epoch < param.Epochs; epoch++)
                    {
                        Shuffle(indices, rnd);
                        for (int start = 0; start < n; start += (param.BatchSize ?? 32))
                        {
                            int batchCount = Math.Min(param.BatchSize ?? 32, n - start);
                            var batchIndices = indices.AsSpan(start, batchCount);
                            batchIndices.CopyTo(batchIndexBuffer);
                            var batchIndexArray = batchIndexBuffer.AsSpan(0, batchCount).ToArray();
                            using var batchIndexTensor = Tensorflow.NumPy.np.array(batchIndexArray, dtype: tf.int32);
                            using var batchFeatures = trainFeatureTensor[batchIndexTensor];
                            using var batchLabels = trainLabelTensor[batchIndexTensor];

                            sess.run(optimizer, new FeedItem(x, batchFeatures), new FeedItem(y, batchLabels));
                        }

                        (var epochLoss, var epochFocalLoss) = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, epochPredBuffer, n);
                        var epochAcc = ComputeWinnerAccuracy(data.TrainRaceIds, epochPredBuffer, data.TrainLabels);

                        if (hasValidationExamples)
                        {
                            (var epochValLoss, var epochValFocalLoss) = ComputeDatasetMetrics(valFeatureTensor, valLabelTensor, valPreds, data.ValLabels.Count);
                            var epochValAcc = ComputeWinnerAccuracy(data.ValRaceIds, valPreds, data.ValLabels);
                            Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - focal_loss: {epochFocalLoss:F4} - winner acc: {epochAcc:F4} - val_loss: {epochValLoss:F4} - val_focal_loss: {epochValFocalLoss:F4} - val_acc: {epochValAcc:F4}");

                            if (epochValLoss < bestValLoss)
                            {
                                bestValLoss = epochValLoss;
                                patienceCounter = 0;
                                bestModelPath = Path.GetTempFileName();
                                var saver = tf.train.Saver();
                                saver.save(sess, bestModelPath);
                                // Save best weights in memory
                                bestWeights.Clear();
                                foreach (var v in allVariables) bestWeights.Add(sess.run(v));
                            }
                            else
                            {
                                patienceCounter++;
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
                            // Fix for InvalidCastException: Explicitly convert NDArray to constant Tensor
                            var weightData = bestWeights[i].ToArray<float>();
                            var shape = bestWeights[i].shape;
                            var tensor = tf.constant(weightData, shape: shape);
                            sess.run(allVariables[i].assign(tensor));
                        }
                    }
                    (trainLoss, trainFocalLoss) = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, trainPreds, n);
                    trainBrier = ComputeBrier(trainPreds, data.TrainLabels.ToArray()); // ToArray if List

                    if (hasValidationExamples)
                    {
                        (valLoss, valFocalLoss) = ComputeDatasetMetrics(valFeatureTensor, valLabelTensor, valPreds, data.ValLabels.Count);
                        valBrier = ComputeBrier(valPreds, data.ValLabels.ToArray());
                    }

                    trainAcc = ComputeWinnerAccuracy(data.TrainRaceIds, trainPreds, data.TrainLabels);
                    valAcc = hasValidationExamples ? ComputeWinnerAccuracy(data.ValRaceIds, valPreds, data.ValLabels) : 0;
                }
            }
            catch (Exception ex)
            {
                HyperparameterFailed(param, ex);
                throw;
            }
            finally
            {
                if (bestModelPath != null && File.Exists(bestModelPath))
                {
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

            // Declare model variable outside the persistWeights block so it's accessible for return
            TrainedModel? trainedModel = null;

            // Feature metadata is not passed in the fold data explicitly, we need it if we want to save metadata
            // For now, assuming standard metadata usage if we had it. 
            // Since this optimized path is mostly for search (persistWeights=false), we skip saving full metadata if not available.

            // Re-fetch master dataset just for metadata if persisting
            // Warning: If we are in Hybrid training, EnsureMasterDatasetLoaded returns the FULL dataset,
            // but the NN model was trained on a SUBSET (from 'data' argument).
            // We must use the keys from 'data' (if available?) or we need to pass metadata in.
            // TensorFlowFoldData does not contain metadata.
            // However, in TrainHybrid, we call this with persistWeights=true/false. 
            // If we are in TrainHybrid, we handle persistence manually OR we rely on the override.
            // If modelPathOverride is set, we assume we want to save there.

            if (persistWeights)
            {
                var targetPath = !string.IsNullOrEmpty(modelPathOverride)
                    ? (Path.IsPathRooted(modelPathOverride) ? modelPathOverride : Path.Combine(AppContext.BaseDirectory, modelPathOverride))
                    : _modelPath;

                var dataset = EnsureMasterDatasetLoaded();
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

                trainedModel = new TrainedModel
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
                        Mean = (float[])normalization.Mean.Clone(),
                        StdDev = (float[])normalization.StdDev.Clone()
                    },
                    Hyperparameters = new HyperparameterSummary
                    {
                        Layers = param.Layers ?? 0,
                        Units = param.Units ?? 10,
                        Dropout = param.Dropout ?? 0,
                        LearningRate = param.LearningRate,
                        Epochs = param.Epochs,
                        BatchSize = param.BatchSize ?? 32,
                        Folds = param.Folds,
                        Fold = param.Fold,
                        TrainedAtUtc = trainedAtUtc
                    }
                };

                var weightsDirectory = Path.GetDirectoryName(targetPath);
                if (weightsDirectory != null)
                {
                    Directory.CreateDirectory(weightsDirectory);
                }

                var options = new JsonSerializerOptions { WriteIndented = true };
                var serialized = JsonSerializer.Serialize(trainedModel, options);
                File.WriteAllText(targetPath, serialized);
            }
            else
            {
                // Create minimal model structure for return if not persisting (e.g. for Hybrid combination)
                trainedModel = new TrainedModel
                {
                    HiddenLayers = hiddenLayers,
                    OutputLayer = outputLayer,
                    Normalization = normalization
                };
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
                TrainingLabels = new ReadOnlyCollection<float>(data.TrainLabels.ToList()),
                TrainingRaceIds = new ReadOnlyCollection<int>(data.TrainRaceIds.ToList()),
                ValidationPredictions = Array.AsReadOnly(valPreds),
                ValidationLabels = new ReadOnlyCollection<float>(data.ValLabels.ToList()),
                ValidationRaceIds = new ReadOnlyCollection<int>(data.ValRaceIds.ToList()),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(data.ValExamples.ToList()),
                FeatureCorrelations = new ReadOnlyCollection<FeatureCorrelation>(featureCorrelations),
                Model = trainedModel // Return the model object for potential external usage
            };
        }
        private static float[][] ToJagged2D(NDArray nd)
        {
            // Helper to convert TF NDArray to float[][]
            if (nd.ndim != 2) return Array.Empty<float[]>();
            var rows = nd.shape[0];
            var cols = nd.shape[1];

            var result = new float[rows][];
            var flat = nd.ToArray<float>();
            for (int i = 0; i < rows; i++)
            {
                var row = new float[cols];
                for (int j = 0; j < cols; j++) row[j] = flat[i * cols + j];
                result[i] = row;
            }
            return result;
        }

        private static Tensor SigmoidFocalLoss(Tensor yTrue, Tensor yLogits, float alpha = 0.25f, float gamma = 2.0f)
        {
            var sigmoidP = tf.sigmoid(yLogits);
            var zeros = tf.zeros_like(sigmoidP);
            var ones = tf.ones_like(sigmoidP);
            // pt = p if y=1 else 1-p
            var pt = tf.where(tf.equal(yTrue, tf.constant(1.0f)), sigmoidP, ones - sigmoidP);
            // CE = -log(pt)
            // But focal loss: -alpha * (1-pt)^gamma * log(pt)
            // We use standard sigmoid_cross_entropy for numerical stability of the log part if possible, but implementing manually here:
            // FL = -alpha * (1-pt)^gamma * log(pt)
            // However, implementing via standard TF ops:

            var bce = tf.nn.sigmoid_cross_entropy_with_logits(labels: yTrue, logits: yLogits);
            // Fix: Wrap alpha in tf.constant
            var alphaTensor = tf.constant(alpha);
            var oneMinusAlphaTensor = tf.constant(1 - alpha);
            var alpha_t = tf.where(tf.equal(yTrue, tf.constant(1.0f)), tf.fill(tf.shape(yTrue), alphaTensor), tf.fill(tf.shape(yTrue), oneMinusAlphaTensor));
            var p_t = tf.where(tf.equal(yTrue, tf.constant(1.0f)), sigmoidP, ones - sigmoidP);

            var loss = alpha_t * tf.pow(ones - p_t, gamma) * bce;
            return loss;
        }
        private void HyperparameterStarted(MLParameter p, int trainCount, int valCount, int features)
        {
            // Placeholder logging
        }
        private void HyperparameterFailed(MLParameter p, Exception ex)
        {
            // Placeholder logging
            Console.Error.WriteLine($"[HyperparameterTrainer] Failed: {ex.Message}");
        }
        private void HyperparameterCompleted(MLParameter p, double tAcc, double vAcc, double tLoss, double vLoss, double tBrier, double vBrier, double tFocal, double vFocal)
        {
            // Placeholder logging
        }
    }
}
