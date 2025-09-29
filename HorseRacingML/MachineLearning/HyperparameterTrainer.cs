using HorseRacingML.Models;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Tensorflow;
using Tensorflow.NumPy;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;
using static Tensorflow.Binding;
using TensorShape = Tensorflow.Shape;
using System.Collections.ObjectModel;

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

        public HyperparameterTrainer(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
            _winRateAlpha = configuration.GetValue<float>("WinRateAlpha", 1f);
            _winRateBeta = configuration.GetValue<float>("WinRateBeta", 2f);
        }




        public class TrainingResult
        {
            public double TrainAccuracy { get; init; }
            public double TrainLoss { get; init; }
            public double TrainBrier { get; init; }
            public double ValidationAccuracy { get; init; }
            public double ValidationLoss { get; init; }
            public double ValidationBrier { get; init; }
            public IReadOnlyList<float> TrainingPredictions { get; init; } = Array.Empty<float>();
            public IReadOnlyList<float> TrainingLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> TrainingRaceIds { get; init; } = Array.Empty<int>();
            public IReadOnlyList<float> ValidationPredictions { get; init; } = Array.Empty<float>();
            public IReadOnlyList<float> ValidationLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> ValidationRaceIds { get; init; } = Array.Empty<int>();
            public IReadOnlyList<RunnerExample> ValidationExamples { get; init; } = Array.Empty<RunnerExample>();
        }


        public TrainingDataset LoadTrainingDataset(bool includeIdentifiers = false)
        {
            var prepared = PrepareDataset(includeIdentifiers: includeIdentifiers);
            var emptyValidation = new PreparedDataset(new List<PreparedRace>());
            return BuildTrainingDataset(prepared, emptyValidation);
        }

        public TrainingResult Train(MLParameter param, int foldIndex, int foldCount, bool persistWeights = true)
        {
            var dataset = LoadTrainingDataset();
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


        private TrainingDataset BuildTrainingDataset(
        PreparedDataset trainingPrepared,
        PreparedDataset validationPrepared)
        {
            if (trainingPrepared is null)
                throw new ArgumentNullException(nameof(trainingPrepared));
            if (validationPrepared is null)
                throw new ArgumentNullException(nameof(validationPrepared));

            var metadataSource = trainingPrepared.RowCount > 0 ? trainingPrepared : validationPrepared;
            var metadataRows = trainingPrepared.RowCount > 0
                ? trainingPrepared.Rows.ToList()
                : metadataSource.Rows.ToList();

            var metadata = BuildFeatureMetadata(metadataSource, metadataRows);
            int featureCount = metadata.FeatureCount;
            var trainRaces = EncodeRaces(trainingPrepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);
            var validationRaces = EncodeRaces(validationPrepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);

            var normalization = new NormalizationParameters
            {
                Mean = new float[featureCount],
                StdDev = new float[featureCount]
            };

            var mapPath = Path.Combine(AppContext.BaseDirectory, "string_maps.json");
            File.WriteAllText(mapPath, JsonSerializer.Serialize(metadata.StringMaps));

            return new TrainingDataset(trainRaces, validationRaces, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps, normalization);
        }

        public TrainingResult Train(MLParameter param, TrainingDataset.PreparedDataset dataset, int foldIndex, int foldCount, bool persistWeights = true)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));

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

            var trainingPrepared = PrepareDataset(trainRaceIds, trainRaceIds);
            TrainingDataset.PreparedDataset validationPrepared;
            if (validationRaceIds.Count > 0)
            {
                validationPrepared = PrepareDataset(validationRaceIds, trainRaceIds);
            }
            else
            {
                validationPrepared = new TrainingDataset.PreparedDataset(new List<TrainingDataset.PreparedDataset.PreparedRace>());
            }

            var trainingDataset = BuildTrainingDataset(trainingPrepared, validationPrepared);
            return Train(param, foldIndex, foldCount, trainingDataset, persistWeights);
        }
        public TrainingResult Train(MLParameter param, int foldIndex, int foldCount, TrainingDataset dataset, bool persistWeights = true)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));

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
            var trainFeatures = trainExamples.Select(r => r.Features).ToList();
            var trainLabels = trainExamples.Select(r => r.Label).ToArray();
            var trainRaceIds = trainExamples.Select(r => r.RaceId).ToArray();

            var valFeatures = valExamples.Select(r => r.Features).ToList();
            var valLabels = valExamples.Select(r => r.Label).ToArray();
            var valRaceIds = valExamples.Select(r => r.RaceId).ToArray();

            var means = dataset.Normalization.Mean;
            if (means.Length != featureCount)
            {
                means = new float[featureCount];
                dataset.Normalization.Mean = means;
            }
            else
            {
                Array.Clear(means, 0, featureCount);
            }

            var stdDevs = dataset.Normalization.StdDev;
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
                for (int j = 0; j < featureCount; j++)
                {
                    double sum = 0;
                    foreach (var feature in trainFeatures)
                    {
                        sum += feature[j];
                    }
                    means[j] = (float)(sum / trainFeatures.Count);
                }

                for (int j = 0; j < featureCount; j++)
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
                }
            }
            else
            {
                Array.Clear(means, 0, featureCount);
                for (int j = 0; j < featureCount; j++)
                {
                    stdDevs[j] = 1f;
                }
            }

            void Normalize(IList<float[]> data)
            {
                foreach (var arr in data)
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        arr[i] = (arr[i] - means[i]) / stdDevs[i];
                    }
                }
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

            var trainFeatureTensor = np.array(BuildFeatureMatrix(trainFeatures, featureCount), dtype: tf.float32);
            var trainLabelTensor = np.array(BuildLabelMatrix(trainLabels), dtype: tf.float32);
            var valFeatureTensor = np.array(BuildFeatureMatrix(valFeatures, featureCount), dtype: tf.float32);
            var valLabelTensor = np.array(BuildLabelMatrix(valLabels), dtype: tf.float32);
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
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(loss);

            var prediction = tf.sigmoid(logits);
            var rnd = new Random();
            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());

            var normPath = Path.Combine(AppContext.BaseDirectory, "normalization.json");
            File.WriteAllText(normPath, JsonSerializer.Serialize(dataset.Normalization));

            var trainPreds = new float[trainLabels.Length];
            var valPreds = new float[valLabels.Length];
            var epochPredBuffer = new float[trainLabels.Length];

            double ComputeDatasetMetrics(NDArray featureTensor, NDArray labelTensor, float[] preds, int count)
            {
                if (preds.Length != count)
                    throw new ArgumentException("Prediction buffer size must match example count.", nameof(preds));

                if (count == 0)
                {
                    return 0;
                }

                const int evalBatchSize = 8192;
                double weightedLoss = 0;
                int totalExamples = 0;

                for (int start = 0; start < count; start += evalBatchSize)
                {
                    int batchCount = Math.Min(evalBatchSize, count - start);

                    var featureSlice = featureTensor[new Slice(start, start + batchCount), Slice.All];
                    var labelSlice = labelTensor[new Slice(start, start + batchCount), Slice.All];

                    var results = sess.run(new[] { loss, prediction },
                        new FeedItem(x, featureSlice),
                        new FeedItem(y, labelSlice));

                    var chunkLoss = results[0].ToArray<float>()[0];
                    var chunkPreds = results[1].ToArray<float>();
                    Array.Copy(chunkPreds, 0, preds, start, batchCount);
                    weightedLoss += chunkLoss * batchCount;
                    totalExamples += batchCount;
                }

                return totalExamples > 0 ? weightedLoss / totalExamples : 0;
            }

            void Denormalize(IList<float[]> data)
            {
                foreach (var arr in data)
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        arr[i] = arr[i] * stdDevs[i] + means[i];
                    }
                }
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
            double trainAcc = 0;
            double valAcc = 0;
            HyperparameterStarted(param, trainLabels.Length, valLabels.Length, featureCount);

            try
            {
                if (param.BatchSize <= 0)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(param.BatchSize),
                        param.BatchSize,
                        "Batch size must be greater than zero.");
                }

                for (int epoch = 0; epoch < param.Epochs; epoch++)
                {
                    var indices = new int[trainFeatures.Count];
                    for (int i = 0; i < indices.Length; i++)
                    {
                        indices[i] = i;
                    }
                    for (int i = indices.Length - 1; i > 0; i--)
                    {
                        int j = rnd.Next(i + 1);
                        (indices[i], indices[j]) = (indices[j], indices[i]);
                    }
                    for (int start = 0; start < indices.Length; start += param.BatchSize)
                    {
                        int batchCount = Math.Min(param.BatchSize, indices.Length - start);
                        if (batchCount <= 0)
                        {
                            continue;
                        }

                        var batchIdx = indices[start..(start + batchCount)];
                        var batchFeatures = new List<float[]>(batchCount);
                        var batchLabels = new float[batchCount];
                        for (int b = 0; b < batchCount; b++)
                        {
                            int dataIndex = batchIdx[b];
                            batchFeatures.Add(trainFeatures[dataIndex]);
                            batchLabels[b] = trainLabels[dataIndex];
                        }

                        var batchX = np.array(BuildFeatureMatrix(batchFeatures, featureCount), dtype: tf.float32);
                        var batchY = np.array(BuildLabelMatrix(batchLabels), dtype: tf.float32);
                        sess.run(optimizer, new FeedItem(x, batchX), new FeedItem(y, batchY));
                    }

                    var epochLoss = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, epochPredBuffer, trainLabels.Length);
                    var epochAcc = trainLabels.Length > 0 ? ComputeWinnerAccuracy(trainRaceIds, epochPredBuffer, trainLabels) : 0;
                    Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
                }

                trainLoss = ComputeDatasetMetrics(trainFeatureTensor, trainLabelTensor, trainPreds, trainLabels.Length);
                valLoss = ComputeDatasetMetrics(valFeatureTensor, valLabelTensor, valPreds, valLabels.Length);
                trainBrier = ComputeBrier(trainPreds, trainLabels);
                valBrier = ComputeBrier(valPreds, valLabels);

                trainAcc = trainPreds.Length > 0 ? ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels) : 0;
                valAcc = valPreds.Length > 0 ? ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels) : 0;
            }
            catch (Exception ex)
            {
                HyperparameterFailed(param, ex);
                throw;
            }
            finally
            {
                Denormalize(trainFeatures);
                Denormalize(valFeatures);
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
            HyperparameterCompleted(param, trainAcc, valAcc, trainLoss, valLoss, trainBrier, valBrier);
            return new TrainingResult
            {
                TrainAccuracy = trainAcc,
                TrainLoss = trainLoss,
                TrainBrier = trainBrier,
                ValidationAccuracy = valAcc,
                ValidationLoss = valLoss,
                ValidationBrier = valBrier,
                TrainingPredictions = Array.AsReadOnly(trainPreds),
                TrainingLabels = Array.AsReadOnly(trainLabels),
                TrainingRaceIds = Array.AsReadOnly(trainRaceIds),
                ValidationPredictions = Array.AsReadOnly(valPreds),
                ValidationLabels = Array.AsReadOnly(valLabels),
                ValidationRaceIds = Array.AsReadOnly(valRaceIds),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(valExamples)
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
        public TrainingDataset LoadTrainingDataset(ISet<int> trainingRaceIds, ISet<int> validationRaceIds, bool includeIdentifiers = false)
        {
            if (trainingRaceIds is null)
                throw new ArgumentNullException(nameof(trainingRaceIds));
            if (validationRaceIds is null)
                throw new ArgumentNullException(nameof(validationRaceIds));

            var trainingPrepared = PrepareDataset(trainingRaceIds, trainingRaceIds, includeIdentifiers);
            PreparedDataset validationPrepared;
            if (validationRaceIds.Count > 0)
            {
                validationPrepared = PrepareDataset(validationRaceIds, trainingRaceIds, includeIdentifiers: includeIdentifiers);
            }
            else
            {
                validationPrepared = new PreparedDataset(new List<PreparedRace>());
            }

            return BuildTrainingDataset(trainingPrepared, validationPrepared);
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

        private static void HyperparameterCompleted(MLParameter param, double trainAccuracy, double validationAccuracy, double trainLoss, double validationLoss, double trainBrier, double validationBrier)
        {
            Console.WriteLine(
                "[Hyperparameter] Training complete | layers: {0}, units: {1}, epochs: {2}, train acc: {3:F4}, val acc: {4:F4}, train loss: {5:F4}, val loss: {6:F4}, train brier: {7:F4}, val brier: {8:F4}",
                param.Layers,
                param.Units,
                param.Epochs,
                trainAccuracy,
                validationAccuracy,
                trainLoss,
                validationLoss,
                trainBrier,
                validationBrier);
        }
        public record RunnerProbability(int RaceId, int? ClothNumber, string? HorseName, string? SelectionId, double Probability);

        public TrainedModel? LoadLatestTrainedModel()
        {
            var weightDirectory = Path.Combine(AppContext.BaseDirectory, "weights");
            var weightPath = Path.Combine(weightDirectory, "aiweights.json");
            if (!File.Exists(weightPath))
            {
                Console.Error.WriteLine($"[AI] Trained weight file not found at {weightPath}.");
                return null;
            }

            try
            {
                var json = File.ReadAllText(weightPath);
                var model = JsonSerializer.Deserialize<TrainedModel>(json);
                if (model == null)
                {
                    Console.Error.WriteLine("[AI] Failed to deserialize trained weight file; model was null.");
                }
                return model;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AI] Failed to load trained model: {ex.Message}");
                return null;
            }
        }

        public IReadOnlyList<RunnerProbability> PredictRaceProbabilities(TrainingDataset.PreparedDataset.PreparedRace race, TrainedModel model)
        {
            if (race is null)
                throw new ArgumentNullException(nameof(race));
            if (model is null)
                throw new ArgumentNullException(nameof(model));

            if (model.Metadata is null || model.Metadata.Keys is null || model.Metadata.FeatureDimensions is null)
            {
                return Array.Empty<RunnerProbability>();
            }

            var featureKeys = model.Metadata.Keys;
            var featureDimensions = model.Metadata.FeatureDimensions;
            var stringMaps = model.Metadata.StringMaps ?? new Dictionary<string, Dictionary<string, int>>();
            int featureCount = featureDimensions.Values.Sum();
            if (featureCount == 0)
            {
                return Array.Empty<RunnerProbability>();
            }

            var normalization = model.Normalization ?? new NormalizationParameters();
            var results = new List<RunnerProbability>(race.Rows.Count);
            foreach (var row in race.Rows)
            {
                if (row is null)
                {
                    continue;
                }

                float[] featureVector;
                try
                {
                    featureVector = HyperparameterTrainer.EncodeFeatureVector(row, featureKeys, featureDimensions, stringMaps, featureCount);
                }
                catch
                {
                    continue;
                }

                var normalized = NormalizeFeatures(featureVector, normalization);
                var probability = ForwardPass(normalized, model.HiddenLayers, model.OutputLayer);
                if (!double.IsFinite(probability) || probability <= 0)
                {
                    continue;
                }

                int? clothNumber = null;
                if (row.TryGetValue("SaddleclothNumber", out var clothObj) && TrainingDataset.PreparedDataset.TryConvertToInt32(clothObj, out var cloth))
                {
                    clothNumber = cloth;
                }

                string? selectionId = null;
                if (row.TryGetValue("SelectionId", out var selectionObj) && selectionObj is not null)
                {
                    selectionId = selectionObj.ToString();
                }

                string? horseName = null;
                if (row.TryGetValue("HorseName", out var horseObj) && horseObj is not null)
                {
                    horseName = horseObj.ToString();
                }

                results.Add(new RunnerProbability(race.RaceId, clothNumber, horseName, selectionId, probability));
            }

            return results;
        }

        private static double[] NormalizeFeatures(float[] features, NormalizationParameters normalization)
        {
            if (features is null)
                throw new ArgumentNullException(nameof(features));
            if (normalization is null)
                throw new ArgumentNullException(nameof(normalization));

            var normalized = new double[features.Length];
            var means = normalization.Mean ?? Array.Empty<float>();
            var stds = normalization.StdDev ?? Array.Empty<float>();

            for (int i = 0; i < features.Length; i++)
            {
                double value = double.IsFinite(features[i]) ? features[i] : 0d;
                double mean = i < means.Length && double.IsFinite(means[i]) ? means[i] : 0d;
                double std = i < stds.Length && double.IsFinite(stds[i]) ? stds[i] : 0d;
                if (Math.Abs(std) < 1e-8)
                {
                    normalized[i] = 0d;
                }
                else
                {
                    normalized[i] = (value - mean) / std;
                }
            }

            return normalized;
        }

        private static double ForwardPass(double[] inputs, List<LayerWeights> hiddenLayers, LayerWeights outputLayer)
        {
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            if (hiddenLayers is null)
                throw new ArgumentNullException(nameof(hiddenLayers));
            if (outputLayer is null)
                throw new ArgumentNullException(nameof(outputLayer));

            var activations = inputs;
            foreach (var layer in hiddenLayers)
            {
                activations = ApplyRelu(ApplyLayer(activations, layer));
            }

            var output = ApplyLayer(activations, outputLayer);
            if (output.Length == 0)
            {
                return 0d;
            }

            return Sigmoid(output[0]);
        }

        private static double[] ApplyLayer(double[] inputs, LayerWeights layer)
        {
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            if (layer is null)
                throw new ArgumentNullException(nameof(layer));

            var weights = layer.Weights ?? Array.Empty<float[]>();
            var bias = layer.Bias ?? Array.Empty<float>();
            int outputCount = Math.Max(bias.Length, weights.Length > 0 ? weights[0]?.Length ?? 0 : 0);
            var outputs = new double[outputCount];

            for (int j = 0; j < outputCount; j++)
            {
                double sum = j < bias.Length ? bias[j] : 0d;
                int inputCount = Math.Min(inputs.Length, weights.Length);
                for (int i = 0; i < inputCount; i++)
                {
                    var weightRow = weights[i];
                    if (weightRow == null || j >= weightRow.Length)
                    {
                        continue;
                    }

                    sum += inputs[i] * weightRow[j];
                }

                outputs[j] = sum;
            }

            return outputs;
        }

        private static double[] ApplyRelu(double[] values)
        {
            if (values is null)
                throw new ArgumentNullException(nameof(values));

            var result = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                var value = values[i];
                result[i] = value > 0d ? value : 0d;
            }

            return result;
        }

        private static double Sigmoid(double value)
        {
            if (value >= 0)
            {
                var z = Math.Exp(-value);
                return 1d / (1d + z);
            }
            else
            {
                var z = Math.Exp(value);
                return z / (1d + z);
            }
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

    }
}