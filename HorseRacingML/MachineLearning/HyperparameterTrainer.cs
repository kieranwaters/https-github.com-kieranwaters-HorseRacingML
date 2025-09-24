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
        public TrainingDataset LoadTrainingDataset()
        {
            var prepared = PrepareDataset();
            var emptyValidation = new PreparedDataset(new List<PreparedRace>());
            return BuildTrainingDataset(prepared, emptyValidation);
        }

        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss, double TrainBrier, double ValidationBrier) Train(MLParameter param, int foldIndex, int foldCount, bool persistWeights = true)
        {
            var dataset = LoadTrainingDataset();
            return Train(param, foldIndex, foldCount, dataset, persistWeights);
        }

        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss, double TrainBrier, double ValidationBrier) Train(MLParameter param, TrainingDataset.PreparedDataset dataset, int foldIndex, int foldCount, bool persistWeights = true)
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
        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss, double TrainBrier, double ValidationBrier) Train(MLParameter param, int foldIndex, int foldCount, TrainingDataset dataset, bool persistWeights = true)
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
                }
            };

            if (persistWeights)
            {
                var weightPath = Path.Combine(AppContext.BaseDirectory, "aiweights.json");
                try
                {
                    var options = new JsonSerializerOptions { WriteIndented = true };
                    File.WriteAllText(weightPath, JsonSerializer.Serialize(model, options));
                }
                catch
                {
                    // Failing to persist weights shouldn't abort training; simply swallow
                    // any IO issues so training metrics are still returned.
                }
            }
            HyperparameterCompleted(param, trainAcc, valAcc, trainLoss, valLoss, trainBrier, valBrier);
            return (trainAcc, trainLoss, valAcc, valLoss, trainBrier, valBrier);
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