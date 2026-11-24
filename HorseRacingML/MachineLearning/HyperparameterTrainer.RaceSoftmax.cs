using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using HorseRacingML.Models;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using TensorShape = Tensorflow.Shape;

namespace HorseRacingML.ML
{
    public partial class HyperparameterTrainer
    {
        private sealed class RaceBatchData
        {
            public float[,] Features = new float[0, 0];
            public float[] Labels = Array.Empty<float>();
            public float[] MarketOffsets = Array.Empty<float>();
            public float[] Mask = Array.Empty<float>();
            public float[] SampleWeights = Array.Empty<float>();
            public float[] FinishTargets = Array.Empty<float>();
            public float[] FinishMask = Array.Empty<float>();
            public int[] FlatIndices = Array.Empty<int>();
            public int RaceCount;
            public int MaxRunners;
        }

        /// <summary>
        /// Conditional-logit (race softmax) trainer. Each race is one softmax
        /// group, so the model learns the probability that a runner wins against
        /// the other runners in the same race. The market log-odds are used as an
        /// additive offset, which makes the network learn the residual edge over
        /// the price instead of re-learning the price from scratch.
        /// </summary>
        private TrainingResult TrainRaceSoftmax(
            MLParameter param,
            TrainingDataset dataset,
            bool persistWeights,
            float? temperatureOverride,
            bool earlyStopOnValidation,
            double blendModelWeight,
            double blendMarketWeight,
            bool activateModel = false,
            CancellationToken cancellationToken = default)
        {
            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

            int seed = param.Seed ?? _seed;
            bool marketBlind = param.MarketBlind ?? _marketBlind;
            string marketOffsetMode = NormalizeOffsetMode(param.MarketOffsetMode ?? _marketOffsetMode);
            bool useMarketOffset = _useMarketOffset && !marketBlind;
            bool useLogProbOffset = string.Equals(marketOffsetMode, "LogProb", StringComparison.OrdinalIgnoreCase);

            int featureCount = dataset.FeatureCount;
            bool[]? marketFeatureMask = null;
            if (marketBlind)
            {
                marketFeatureMask = new bool[featureCount];
                int maskOffset = 0;
                foreach (var key in dataset.FeatureKeys)
                {
                    int dimension = dataset.FeatureDimensions.TryGetValue(key, out var dim) ? dim : 0;
                    if (IsMarketFeatureKey(key))
                    {
                        for (int j = 0; j < dimension && maskOffset + j < featureCount; j++)
                        {
                            marketFeatureMask[maskOffset + j] = true;
                        }
                    }

                    maskOffset += dimension;
                }
            }

            var trainRaces = dataset.TrainingRaces;
            var valRaces = dataset.ValidationRaces;
            var trainExamples = trainRaces.SelectMany(r => r.Runners).ToList();
            var valExamples = valRaces.SelectMany(r => r.Runners).ToList();
            var trainLabels = trainExamples.Select(r => r.Label).ToArray();
            var valLabels = valExamples.Select(r => r.Label).ToArray();
            var trainRaceIds = trainExamples.Select(r => r.RaceId).ToArray();
            var valRaceIds = valExamples.Select(r => r.RaceId).ToArray();
            var trainFlatStarts = BuildFlatStarts(trainRaces);
            var valFlatStarts = BuildFlatStarts(valRaces);

            ComputeNormalization(trainExamples, featureCount, out var means, out var stdDevs);

            var graph = tf.Graph().as_default();
            try { tf.set_random_seed(seed); } catch { }
            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1), name: "y");
            var market = tf.placeholder(tf.float32, shape: new TensorShape(-1), name: "market");
            var mask = tf.placeholder(tf.float32, shape: new TensorShape(-1), name: "mask");
            var sampleWeight = tf.placeholder(tf.float32, shape: new TensorShape(-1), name: "sampleWeight");
            var finishTarget = tf.placeholder(tf.float32, shape: new TensorShape(-1), name: "finishTarget");
            var finishMask = tf.placeholder(tf.float32, shape: new TensorShape(-1), name: "finishMask");
            var batchShape = tf.placeholder(tf.int32, shape: new TensorShape(2), name: "batchShape");

            Tensor hidden = x;
            // The evaluation branch shares the same variables but never applies
            // dropout, so dev/test metrics and the persisted model match the
            // deterministic forward pass used by TrainedModelRuntime.
            Tensor inferenceHidden = x;
            int inputDim = featureCount;
            var hiddenWeightVars = new List<ResourceVariable>();
            var hiddenBiasVars = new List<ResourceVariable>();
            for (int i = 0; i < param.Layers; i++)
            {
                float stdDev = (float)Math.Sqrt(2.0 / Math.Max(1, inputDim));
                var weight = tf.Variable(tf.random.normal((inputDim, param.Units), stddev: stdDev), name: $"w{i}");
                var bias = tf.Variable(tf.zeros(param.Units), name: $"b{i}");
                hiddenWeightVars.Add(weight);
                hiddenBiasVars.Add(bias);
                hidden = tf.nn.relu(tf.matmul(hidden, weight) + bias);
                inferenceHidden = tf.nn.relu(tf.matmul(inferenceHidden, weight) + bias);
                if (param.Dropout > 0)
                {
                    hidden = tf.nn.dropout(hidden, rate: (float)param.Dropout);
                }
                inputDim = param.Units;
            }

            var wOut = tf.Variable(
                tf.random.normal((inputDim, 1), stddev: (float)Math.Sqrt(1.0 / Math.Max(1, inputDim))),
                name: "wOut");
            var bOut = tf.Variable(tf.zeros(1), name: "bOut");
            var marketOffsetScale = tf.Variable(tf.constant(1f), name: "marketOffsetScale");

            // Auxiliary multi-task heads. They share the trunk but have their
            // own scalar output weights, so the main softmax head keeps its
            // inference contract (one logit per runner). Both heads are
            // training-only regularisers: the persisted artifact contains just
            // the trunk and the race-softmax output layer.
            float auxWinWeight = _auxiliaryWinLossWeight;
            float auxFinishWeight = _auxiliaryFinishLossWeight;
            ResourceVariable? wAuxWin = null;
            ResourceVariable? bAuxWin = null;
            ResourceVariable? wAuxFinish = null;
            ResourceVariable? bAuxFinish = null;
            Tensor auxWinLoss = tf.constant(0f);
            Tensor auxFinishLoss = tf.constant(0f);

            if (auxWinWeight > 0f)
            {
                float auxStdDev = (float)Math.Sqrt(1.0 / Math.Max(1, inputDim));
                wAuxWin = tf.Variable(
                    tf.random.normal((inputDim, 1), stddev: auxStdDev),
                    name: "wAuxWin");
                bAuxWin = tf.Variable(tf.zeros(1), name: "bAuxWin");
                var auxWinLogits = tf.reshape(
                    tf.matmul(hidden, wAuxWin) + bAuxWin,
                    new object[] { -1 });
                var perRunnerWinLoss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels: y,
                    logits: auxWinLogits);
                var winWeights = sampleWeight * mask;
                auxWinLoss = tf.reduce_sum(perRunnerWinLoss * winWeights) /
                             tf.maximum(tf.reduce_sum(winWeights), tf.constant(1f));
            }

            if (auxFinishWeight > 0f)
            {
                float auxStdDev = (float)Math.Sqrt(1.0 / Math.Max(1, inputDim));
                wAuxFinish = tf.Variable(
                    tf.random.normal((inputDim, 1), stddev: auxStdDev),
                    name: "wAuxFinish");
                bAuxFinish = tf.Variable(tf.zeros(1), name: "bAuxFinish");
                var auxFinishLogits = tf.reshape(
                    tf.matmul(hidden, wAuxFinish) + bAuxFinish,
                    new object[] { -1 });
                // Soft-label cross-entropy against the normalised finishing
                // position. It is numerically stable for 0/1 targets and keeps
                // the finish loss on a similar scale to the win loss.
                var perRunnerFinishLoss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels: finishTarget,
                    logits: auxFinishLogits);
                var finishWeights = sampleWeight * finishMask;
                auxFinishLoss = tf.reduce_sum(perRunnerFinishLoss * finishWeights) /
                                tf.maximum(tf.reduce_sum(finishWeights), tf.constant(1f));
            }

            var flatLogits = tf.reshape(tf.matmul(hidden, wOut) + bOut, new object[] { -1 });
            var inferenceFlatLogits = tf.reshape(tf.matmul(inferenceHidden, wOut) + bOut, new object[] { -1 });
            if (useMarketOffset)
            {
                flatLogits = flatLogits + marketOffsetScale * market;
                inferenceFlatLogits = inferenceFlatLogits + marketOffsetScale * market;
            }

            // Reshape the flat batch into [races, maxRunners]. Padded slots are
            // masked out of the softmax. Using a dense softmax keeps the loss
            // differentiable with TensorFlow.NET's registered gradients, whereas
            // the segment ops do not have gradients in this binding.
            var logits = tf.reshape(flatLogits, batchShape);
            var labels = tf.reshape(y, batchShape);
            var sampleWeight2D = tf.reshape(sampleWeight, batchShape);
            var mask2D = tf.reshape(mask, batchShape);
            var maskedLogits = logits + (1f - mask2D) * (-1e9f);
            var softmax = tf.nn.softmax(maskedLogits);
            var logProbabilities = tf.log(softmax + 1e-9f);

            var inferenceLogits = tf.reshape(inferenceFlatLogits, batchShape);
            var inferenceMaskedLogits = inferenceLogits + (1f - mask2D) * (-1e9f);
            var inferenceSoftmax = tf.nn.softmax(inferenceMaskedLogits);

            var weightedLabels = labels * sampleWeight2D;
            var dataLoss = -tf.reduce_sum(weightedLabels * logProbabilities) /
                           tf.maximum(tf.reduce_sum(weightedLabels), tf.constant(1f));
            var probabilities = tf.reshape(softmax, new object[] { -1 });
            var inferenceProbabilities = tf.reshape(inferenceSoftmax, new object[] { -1 });

            // Top-of-the-field ranking regulariser. The softmax loss is a
            // proper conditional likelihood, but a field of longshots can
            // still dominate the gradient. This extra term asks the winner's
            // logit to exceed the mean loser logit, which is closer to the
            // top-1 accuracy objective used to choose a live bet.
            Tensor pairwiseLoss = tf.constant(0f);
            if (_pairwiseLossWeight > 0f)
            {
                var winnerMask2D = labels;
                var loserMask2D = mask2D * (1f - labels);
                var winnerCount = tf.reduce_sum(winnerMask2D, axis: new[] { 1 });
                var loserCount = tf.reduce_sum(loserMask2D, axis: new[] { 1 });
                var winnerLogitSum = tf.reduce_sum(logits * winnerMask2D, axis: new[] { 1 });
                var loserLogitSum = tf.reduce_sum(logits * loserMask2D, axis: new[] { 1 });
                var winnerMeanLogit = winnerLogitSum / tf.maximum(winnerCount, tf.constant(1f));
                var loserMeanLogit = loserLogitSum / tf.maximum(loserCount, tf.constant(1f));
                var margin = winnerMeanLogit - loserMeanLogit;
                var softplus = tf.maximum(-margin, tf.constant(0f)) +
                               tf.log(1f + tf.exp(-tf.abs(-margin)));
                var validRaces = tf.cast(
                    tf.logical_and(winnerCount > 0f, loserCount > 0f),
                    tf.float32);
                pairwiseLoss = tf.reduce_sum(softplus * validRaces) /
                               tf.maximum(tf.reduce_sum(validRaces), tf.constant(1f));
            }

            Tensor l2Penalty = tf.constant(0f);
            foreach (var weight in hiddenWeightVars)
            {
                l2Penalty += tf.reduce_sum(tf.square(weight));
            }
            l2Penalty += tf.reduce_sum(tf.square(wOut));
            l2Penalty += tf.square(marketOffsetScale);
            if (wAuxWin is not null)
            {
                l2Penalty += tf.reduce_sum(tf.square(wAuxWin));
            }
            if (wAuxFinish is not null)
            {
                l2Penalty += tf.reduce_sum(tf.square(wAuxFinish));
            }
            var totalLoss = dataLoss + (l2Penalty * _l2WeightDecay);
            if (auxWinWeight > 0f)
            {
                totalLoss = totalLoss + (auxWinWeight * auxWinLoss);
            }
            if (auxFinishWeight > 0f)
            {
                totalLoss = totalLoss + (auxFinishWeight * auxFinishLoss);
            }
            if (_pairwiseLossWeight > 0f)
            {
                totalLoss = totalLoss + (_pairwiseLossWeight * pairwiseLoss);
            }
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(totalLoss);

            var trainableVariables = tf.trainable_variables();
            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());

            var rnd = new Random(seed);
            DateTime recencyReferenceDate = trainRaces.Count > 0
                ? trainRaces.Max(race => race.RaceDate)
                : default;
            int raceBatchSize = Math.Max(1, param.RaceBatchSize ?? _raceBatchSize);
            int evalRaceBatchSize = Math.Max(64, raceBatchSize);
            int epochCount = Math.Max(1, param.Epochs);
            bool hasValidation = valRaces.Count > 0 && valLabels.Length > 0;

            var trainPreds = new float[trainLabels.Length];
            var valPreds = new float[valLabels.Length];
            double bestValidationLoss = double.MaxValue;
            int bestEpoch = -1;
            NDArray[]? bestVariableValues = null;
            double validationTemperature = 1.0;

            try
            {
                if (param.BatchSize <= 0 && param.RaceBatchSize is null)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(param.BatchSize),
                        param.BatchSize,
                        "Batch size must be greater than zero.");
                }

                for (int epoch = 0; epoch < epochCount; epoch++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var order = Enumerable.Range(0, trainRaces.Count).ToArray();
                    ShuffleIndices(order, rnd);

                    for (int start = 0; start < order.Length; start += raceBatchSize)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        int raceCount = Math.Min(raceBatchSize, order.Length - start);
                        if (raceCount <= 0)
                        {
                            continue;
                        }

                        var batch = BuildRaceBatch(
                            trainRaces,
                            order,
                            start,
                            raceCount,
                            featureCount,
                            means,
                            stdDevs,
                            trainFlatStarts,
                            includeFlatIndices: false,
                            useLogProbOffset,
                            referenceDate: recencyReferenceDate,
                            recencyHalfLifeDays: _recencyHalfLifeDays,
                            marketFeatureMask: marketFeatureMask);

                        sess.run(
                            optimizer,
                            new FeedItem(x, np.array(batch.Features, dtype: tf.float32)),
                            new FeedItem(y, np.array(batch.Labels, dtype: tf.float32)),
                            new FeedItem(market, np.array(batch.MarketOffsets, dtype: tf.float32)),
                            new FeedItem(mask, np.array(batch.Mask, dtype: tf.float32)),
                            new FeedItem(sampleWeight, np.array(batch.SampleWeights, dtype: tf.float32)),
                            new FeedItem(finishTarget, np.array(batch.FinishTargets, dtype: tf.float32)),
                            new FeedItem(finishMask, np.array(batch.FinishMask, dtype: tf.float32)),
                            new FeedItem(batchShape, np.array(new[] { batch.RaceCount, batch.MaxRunners }, dtype: tf.int32)));
                    }

                    bool shouldLog = ((epoch + 1) % _evaluationFrequency == 0) || epoch == epochCount - 1;
                    if (hasValidation)
                    {
                        EvaluateRacePredictions(
                            sess,
                            inferenceProbabilities,
                            x,
                            market,
                            mask,
                            batchShape,
                            valRaces,
                            valFlatStarts,
                            valPreds,
                            featureCount,
                            means,
                            stdDevs,
                            evalRaceBatchSize,
                            useLogProbOffset,
                            marketFeatureMask);

                        NormalizePerRace(valPreds, valRaceIds);
                        var epochMetrics = RaceMetrics.Compute(valRaceIds, valPreds, valLabels);
                        double epochValidationLoss = epochMetrics.RaceLogLoss;
                        double epochAccuracy = epochMetrics.Top1Accuracy;

                        if (shouldLog)
                        {
                            Console.WriteLine(
                                $"Epoch {epoch + 1}/{epochCount} [race-softmax] - val race LL: {epochValidationLoss:F4} - runner LL: {epochMetrics.RunnerLogLoss:F4} - top1: {epochAccuracy:F4} - top3: {epochMetrics.Top3Accuracy:F4}");
                        }

                        if (earlyStopOnValidation &&
                            double.IsFinite(epochValidationLoss) &&
                            (bestVariableValues is null || epochValidationLoss < bestValidationLoss - 1e-6))
                        {
                            bestValidationLoss = epochValidationLoss;
                            bestEpoch = epoch;
                            bestVariableValues = sess.run(trainableVariables);
                        }
                        else if (earlyStopOnValidation &&
                                 _earlyStoppingPatience > 0 &&
                                 bestEpoch >= 0 &&
                                 epoch - bestEpoch >= _earlyStoppingPatience)
                        {
                            Console.WriteLine(
                                $"Early stopping at epoch {epoch + 1}/{epochCount}; best validation loss {bestValidationLoss:F4} at epoch {bestEpoch + 1}.");
                            break;
                        }
                    }
                    else if (shouldLog)
                    {
                        Console.WriteLine($"Epoch {epoch + 1}/{epochCount} [race-softmax] - training completed.");
                    }
                }

                if (!earlyStopOnValidation)
                {
                    bestEpoch = epochCount - 1;
                }

                if (bestVariableValues is not null && trainableVariables.Length > 0)
                {
                    var restoreOps = trainableVariables
                        .Select((variable, index) => tf.assign(variable, bestVariableValues[index]))
                        .ToArray();
                    sess.run(restoreOps);
                }

                EvaluateRacePredictions(
                    sess,
                    inferenceProbabilities,
                    x,
                    market,
                    mask,
                    batchShape,
                    trainRaces,
                    trainFlatStarts,
                    trainPreds,
                    featureCount,
                    means,
                    stdDevs,
                    evalRaceBatchSize,
                    useLogProbOffset,
                    marketFeatureMask);

                NormalizePerRace(trainPreds, trainRaceIds);

                if (hasValidation)
                {
                    EvaluateRacePredictions(
                        sess,
                        inferenceProbabilities,
                        x,
                        market,
                        mask,
                        batchShape,
                        valRaces,
                        valFlatStarts,
                        valPreds,
                        featureCount,
                        means,
                        stdDevs,
                        evalRaceBatchSize,
                        useLogProbOffset,
                        marketFeatureMask);

                    NormalizePerRace(valPreds, valRaceIds);
                    validationTemperature = temperatureOverride.HasValue
                        ? temperatureOverride.Value
                        : FitTemperaturePerRaceCategorical(valPreds, valLabels, valRaceIds);
                }
                else if (temperatureOverride.HasValue)
                {
                    validationTemperature = temperatureOverride.Value;
                }

                if (Math.Abs(validationTemperature - 1.0) > 1e-9)
                {
                    ApplyTemperaturePerRace(valPreds, valRaceIds, validationTemperature);
                    ApplyTemperaturePerRace(trainPreds, trainRaceIds, validationTemperature);
                }
            }
            catch (Exception ex)
            {
                HyperparameterFailed(param, ex);
                throw;
            }

            var trainMetrics = RaceMetrics.Compute(trainRaceIds, trainPreds, trainLabels);
            var valMetrics = RaceMetrics.Compute(valRaceIds, valPreds, valLabels);
            double trainLoss = trainMetrics.RunnerLogLoss;
            double trainBrier = trainMetrics.Brier;
            double valLoss = valMetrics.RunnerLogLoss;
            double valBrier = valMetrics.Brier;
            double trainAccuracy = trainMetrics.Top1Accuracy;
            double valAccuracy = valMetrics.Top1Accuracy;

            float offsetScaleValue = 1f;
            try
            {
                var scaleArray = sess.run(marketOffsetScale).ToArray<float>();
                if (scaleArray.Length > 0 && float.IsFinite(scaleArray[0]))
                {
                    offsetScaleValue = scaleArray[0];
                }
            }
            catch
            {
                offsetScaleValue = 1f;
            }

            var hiddenLayers = new List<LayerWeights>();
            foreach (var (weightVar, biasVar) in hiddenWeightVars.Zip(hiddenBiasVars, (weightVar, biasVar) => (weightVar, biasVar)))
            {
                hiddenLayers.Add(new LayerWeights
                {
                    Weights = ToJagged2DArray(sess.run(weightVar)),
                    Bias = sess.run(biasVar).ToArray<float>()
                });
            }

            var outputLayer = new LayerWeights
            {
                Weights = ToJagged2DArray(sess.run(wOut)),
                Bias = sess.run(bOut).ToArray<float>()
            };

            var model = new TrainedModel
            {
                HiddenLayers = hiddenLayers,
                OutputLayer = outputLayer,
                Temperature = temperatureOverride ?? (float)validationTemperature,
                Objective = "RaceSoftmax",
                AuxiliaryWinLossWeight = auxWinWeight,
                AuxiliaryFinishLossWeight = auxFinishWeight,
                PairwiseLossWeight = _pairwiseLossWeight,
                MarketOffsetScale = useMarketOffset ? offsetScaleValue : 0f,
                MarketOffsetMode = marketOffsetMode,
                MarketBlind = marketBlind,
                BlendModelWeight = (float)blendModelWeight,
                BlendMarketWeight = (float)blendMarketWeight,
                NormalizationMode = "PerRaceSoftmax",
                Metadata = new FeatureMetadata
                {
                    Keys = new List<string>(dataset.FeatureKeys),
                    FeatureDimensions = new Dictionary<string, int>(dataset.FeatureDimensions),
                    StringMaps = dataset.StringMaps.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new Dictionary<string, int>(kvp.Value))
                },
                Normalization = new NormalizationParameters
                {
                    Mean = (float[])means.Clone(),
                    StdDev = (float[])stdDevs.Clone()
                }
            };
            model.FeatureSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(model.Metadata);
            model.TrainedUtc = DateTime.UtcNow;

            ModelArtifact? artifact = null;
            if (persistWeights)
            {
                artifact = PersistTrainedModel(model, activateModel);
            }

            HyperparameterCompleted(
                param,
                trainAccuracy,
                valAccuracy,
                trainLoss,
                valLoss,
                trainBrier,
                valBrier,
                trainMetrics.RaceLogLoss,
                valMetrics.RaceLogLoss,
                valMetrics.Top3Accuracy);

            // A hyperparameter sweep trains many graphs in one process. Reset the
            // default graph after the session has been disposed so native graph
            // memory from the previous candidate can be reclaimed instead of
            // accumulating until the process is killed.
            try { Tensorflow.ops.reset_default_graph(); } catch { }
            GC.Collect();
            GC.WaitForPendingFinalizers();

            return new TrainingResult
            {
                TrainAccuracy = trainAccuracy,
                TrainLoss = trainLoss,
                TrainBrier = trainBrier,
                TrainRaceLogLoss = trainMetrics.RaceLogLoss,
                TrainTop3Accuracy = trainMetrics.Top3Accuracy,
                ValidationAccuracy = valAccuracy,
                ValidationLoss = valLoss,
                ValidationBrier = valBrier,
                ValidationRaceLogLoss = valMetrics.RaceLogLoss,
                ValidationTop3Accuracy = valMetrics.Top3Accuracy,
                ValidationTemperature = validationTemperature,
                Objective = "RaceSoftmax",
                AuxiliaryWinLossWeight = auxWinWeight,
                AuxiliaryFinishLossWeight = auxFinishWeight,
                BestEpoch = hasValidation ? bestEpoch : null,
                MarketOffsetScale = useMarketOffset ? offsetScaleValue : 0f,
                MarketOffsetMode = marketOffsetMode,
                MarketBlind = marketBlind,
                Seed = seed,
                NormalizationMode = "PerRaceSoftmax",
                TrainingPredictions = Array.AsReadOnly(trainPreds),
                TrainingLabels = Array.AsReadOnly(trainLabels),
                TrainingRaceIds = Array.AsReadOnly(trainRaceIds),
                ValidationPredictions = Array.AsReadOnly(valPreds),
                ValidationLabels = Array.AsReadOnly(valLabels),
                ValidationRaceIds = Array.AsReadOnly(valRaceIds),
                ValidationExamples = new System.Collections.ObjectModel.ReadOnlyCollection<RunnerExample>(valExamples),
                ModelArtifactFileName = artifact?.FileName,
                ModelHash = artifact?.Sha256,
                FeatureSchemaHash = model.FeatureSchemaHash,
                Model = model
            };
        }

        private static int[] BuildFlatStarts(IReadOnlyList<RaceExample> races)
        {
            var starts = new int[races.Count];
            int offset = 0;
            for (int i = 0; i < races.Count; i++)
            {
                starts[i] = offset;
                offset += races[i].Runners.Count;
            }
            return starts;
        }

        private static void ComputeNormalization(
            IReadOnlyList<RunnerExample> examples,
            int featureCount,
            out float[] means,
            out float[] stdDevs)
        {
            means = new float[featureCount];
            stdDevs = new float[featureCount];
            if (examples.Count == 0)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    stdDevs[j] = 1f;
                }
                return;
            }

            for (int j = 0; j < featureCount; j++)
            {
                double sum = 0;
                for (int i = 0; i < examples.Count; i++)
                {
                    sum += examples[i].Features[j];
                }
                means[j] = (float)(sum / examples.Count);
            }

            for (int j = 0; j < featureCount; j++)
            {
                double variance = 0;
                for (int i = 0; i < examples.Count; i++)
                {
                    double diff = examples[i].Features[j] - means[j];
                    variance += diff * diff;
                }
                stdDevs[j] = (float)Math.Sqrt(variance / examples.Count);
                if (stdDevs[j] == 0f || !float.IsFinite(stdDevs[j]))
                {
                    stdDevs[j] = 1f;
                }
            }
        }

        private static RaceBatchData BuildRaceBatch(
            IReadOnlyList<RaceExample> races,
            int[] raceOrder,
            int orderStart,
            int raceCount,
            int featureCount,
            float[] means,
            float[] stdDevs,
            int[] flatStarts,
            bool includeFlatIndices,
            bool useLogProbOffset,
            DateTime referenceDate = default,
            double recencyHalfLifeDays = 0d,
            bool[]? marketFeatureMask = null)
        {
            int maxRunners = 1;
            for (int r = 0; r < raceCount; r++)
            {
                maxRunners = Math.Max(maxRunners, races[raceOrder[orderStart + r]].Runners.Count);
            }

            int slots = raceCount * maxRunners;
            var batch = new RaceBatchData
            {
                Features = new float[slots, featureCount],
                Labels = new float[slots],
                MarketOffsets = new float[slots],
                Mask = new float[slots],
                SampleWeights = new float[slots],
                FinishTargets = new float[slots],
                FinishMask = new float[slots],
                FlatIndices = includeFlatIndices
                    ? Enumerable.Repeat(-1, slots).ToArray()
                    : Array.Empty<int>(),
                RaceCount = raceCount,
                MaxRunners = maxRunners
            };

            for (int r = 0; r < raceCount; r++)
            {
                int raceIndex = raceOrder[orderStart + r];
                var race = races[raceIndex];
                int flatStart = flatStarts[raceIndex];
                float raceWeight = 1f;
                if (recencyHalfLifeDays > 0d &&
                    referenceDate != default &&
                    race.RaceDate != default)
                {
                    double ageDays = Math.Max(0d, (referenceDate - race.RaceDate).TotalDays);
                    raceWeight = (float)Math.Pow(0.5d, ageDays / recencyHalfLifeDays);
                }
                for (int k = 0; k < race.Runners.Count; k++)
                {
                    int slot = r * maxRunners + k;
                    var runner = race.Runners[k];
                    for (int j = 0; j < featureCount; j++)
                    {
                        float normalized = (runner.Features[j] - means[j]) / stdDevs[j];
                        if (marketFeatureMask is not null && marketFeatureMask[j])
                        {
                            // A market-blind ensemble member must not see the
                            // price, even though the shared schema contains the
                            // market columns.
                            normalized = 0f;
                        }

                        batch.Features[slot, j] = Math.Clamp(normalized, -5f, 5f);
                    }

                    batch.Labels[slot] = runner.Label;
                    batch.MarketOffsets[slot] = useLogProbOffset
                        ? runner.MarketLogProbability
                        : runner.MarketLogit;
                    batch.Mask[slot] = 1f;
                    batch.SampleWeights[slot] = raceWeight;
                    if (runner.HasFinishTarget)
                    {
                        batch.FinishTargets[slot] = runner.FinishTarget;
                        batch.FinishMask[slot] = 1f;
                    }
                    if (includeFlatIndices)
                    {
                        batch.FlatIndices[slot] = flatStart + k;
                    }
                }
            }

            return batch;
        }

        private static void EvaluateRacePredictions(
            Session session,
            Tensor probabilities,
            Tensor x,
            Tensor market,
            Tensor mask,
            Tensor batchShape,
            IReadOnlyList<RaceExample> races,
            int[] flatStarts,
            float[] destination,
            int featureCount,
            float[] means,
            float[] stdDevs,
            int raceBatchSize,
            bool useLogProbOffset,
            bool[]? marketFeatureMask = null)
        {
            if (races.Count == 0)
            {
                return;
            }

            var order = Enumerable.Range(0, races.Count).ToArray();
            for (int start = 0; start < order.Length; start += raceBatchSize)
            {
                int raceCount = Math.Min(raceBatchSize, order.Length - start);
                var batch = BuildRaceBatch(
                    races,
                    order,
                    start,
                    raceCount,
                    featureCount,
                    means,
                    stdDevs,
                    flatStarts,
                    includeFlatIndices: true,
                    useLogProbOffset,
                    marketFeatureMask: marketFeatureMask);

                var results = session.run(
                    new[] { probabilities },
                    new FeedItem(x, np.array(batch.Features, dtype: tf.float32)),
                    new FeedItem(market, np.array(batch.MarketOffsets, dtype: tf.float32)),
                    new FeedItem(mask, np.array(batch.Mask, dtype: tf.float32)),
                    new FeedItem(batchShape, np.array(new[] { batch.RaceCount, batch.MaxRunners }, dtype: tf.int32)));

                var chunk = results[0].ToArray<float>();
                var indices = batch.FlatIndices;
                for (int i = 0; i < chunk.Length && i < indices.Length; i++)
                {
                    if (indices[i] >= 0)
                    {
                        destination[indices[i]] = chunk[i];
                    }
                }
            }
        }

        /// <summary>
        /// Normalises predictions to sum to one within each race. Softmax
        /// predictions are already normalised; this is intentionally idempotent
        /// so the same metric path can be used for the binary fallback.
        /// </summary>
        internal static void NormalizePerRace(float[] predictions, IReadOnlyList<int> raceIds)
        {
            if (predictions.Length == 0 || raceIds.Count == 0)
            {
                return;
            }

            int start = 0;
            while (start < predictions.Length)
            {
                int end = start + 1;
                while (end < predictions.Length && raceIds[end] == raceIds[start])
                {
                    end++;
                }

                double sum = 0;
                for (int i = start; i < end; i++)
                {
                    if (float.IsFinite(predictions[i]) && predictions[i] > 0f)
                    {
                        sum += predictions[i];
                    }
                }

                if (sum > 0 && double.IsFinite(sum))
                {
                    for (int i = start; i < end; i++)
                    {
                        predictions[i] = (float)(Math.Max(0f, predictions[i]) / sum);
                    }
                }

                start = end;
            }
        }

        /// <summary>
        /// Fits a single temperature by minimising the race-level conditional
        /// log loss (<c>-log p(winner)</c>). This is the metric RaceSoftmax is
        /// trained with and the metric live betting is judged on. The old
        /// runner-level binary fit is kept for the legacy Binary objective.
        /// </summary>
        private static double FitTemperaturePerRaceCategorical(
            float[] predictions,
            float[] labels,
            IReadOnlyList<int> raceIds)
        {
            if (predictions.Length == 0 || predictions.Length != labels.Length || raceIds.Count != predictions.Length)
            {
                return 1.0;
            }

            double bestTemperature = 1.0;
            double bestLoss = double.MaxValue;

            double Evaluate(double temperature)
            {
                if (!(temperature > 0) || !double.IsFinite(temperature))
                {
                    return double.MaxValue;
                }

                double loss = 0;
                int races = 0;
                int index = 0;
                while (index < predictions.Length)
                {
                    int end = index + 1;
                    while (end < predictions.Length && raceIds[end] == raceIds[index])
                    {
                        end++;
                    }

                    double sum = 0;
                    for (int i = index; i < end; i++)
                    {
                        double p = Math.Clamp((double)predictions[i], 1e-12, 1.0);
                        sum += Math.Pow(p, 1.0 / temperature);
                    }

                    if (sum <= double.Epsilon || !double.IsFinite(sum))
                    {
                        index = end;
                        continue;
                    }

                    double winnerMass = 0;
                    bool hasWinner = false;
                    for (int i = index; i < end; i++)
                    {
                        if (labels[i] < 0.5f)
                        {
                            continue;
                        }

                        hasWinner = true;
                        double p = Math.Clamp((double)predictions[i], 1e-12, 1.0);
                        winnerMass += Math.Pow(p, 1.0 / temperature);
                    }

                    if (hasWinner && winnerMass > double.Epsilon)
                    {
                        double q = Math.Clamp(winnerMass / sum, 1e-12, 1d);
                        loss -= Math.Log(q);
                        races++;
                    }

                    index = end;
                }

                return races == 0 ? double.MaxValue : loss / races;
            }

            for (double temperature = 0.25; temperature <= 4.0; temperature += 0.05)
            {
                double loss = Evaluate(temperature);
                if (loss < bestLoss)
                {
                    bestLoss = loss;
                    bestTemperature = temperature;
                }
            }

            // Refine around the coarse optimum so the final temperature is not
            // quantised to a 0.05 step.
            double coarse = bestTemperature;
            for (double temperature = Math.Max(0.1, coarse - 0.1);
                 temperature <= Math.Min(6.0, coarse + 0.1);
                 temperature += 0.01)
            {
                double loss = Evaluate(temperature);
                if (loss < bestLoss)
                {
                    bestLoss = loss;
                    bestTemperature = temperature;
                }
            }

            return Math.Round(bestTemperature, 4);
        }

        private static double FitTemperaturePerRace(
            float[] predictions,
            float[] labels,
            IReadOnlyList<int> raceIds)
        {
            if (predictions.Length == 0 || predictions.Length != labels.Length || raceIds.Count != predictions.Length)
            {
                return 1.0;
            }

            double bestTemperature = 1.0;
            double bestLoss = double.MaxValue;

            for (double temperature = 0.25; temperature <= 4.0; temperature += 0.05)
            {
                double loss = 0;
                int index = 0;
                while (index < predictions.Length)
                {
                    int end = index + 1;
                    while (end < predictions.Length && raceIds[end] == raceIds[index])
                    {
                        end++;
                    }

                    double sum = 0;
                    for (int i = index; i < end; i++)
                    {
                        double p = Math.Clamp((double)predictions[i], 1e-12, 1.0);
                        sum += Math.Pow(p, 1.0 / temperature);
                    }

                    if (sum <= double.Epsilon || !double.IsFinite(sum))
                    {
                        index = end;
                        continue;
                    }

                    for (int i = index; i < end; i++)
                    {
                        double p = Math.Clamp((double)predictions[i], 1e-12, 1.0);
                        double q = Math.Pow(p, 1.0 / temperature) / sum;
                        q = Math.Clamp(q, 1e-7, 1 - 1e-7);
                        double target = labels[i];
                        loss -= target * Math.Log(q) + (1 - target) * Math.Log(1 - q);
                    }

                    index = end;
                }

                if (loss < bestLoss)
                {
                    bestLoss = loss;
                    bestTemperature = temperature;
                }
            }

            return bestTemperature;
        }

        private static void ApplyTemperaturePerRace(
            float[] predictions,
            IReadOnlyList<int> raceIds,
            double temperature)
        {
            if (predictions.Length == 0 || temperature <= 0 || !double.IsFinite(temperature))
            {
                return;
            }

            int start = 0;
            while (start < predictions.Length)
            {
                int end = start + 1;
                while (end < predictions.Length && raceIds[end] == raceIds[start])
                {
                    end++;
                }

                double sum = 0;
                for (int i = start; i < end; i++)
                {
                    double p = Math.Clamp((double)predictions[i], 1e-12, 1.0);
                    sum += Math.Pow(p, 1.0 / temperature);
                }

                if (sum > double.Epsilon && double.IsFinite(sum))
                {
                    for (int i = start; i < end; i++)
                    {
                        double p = Math.Clamp((double)predictions[i], 1e-12, 1.0);
                        predictions[i] = (float)(Math.Pow(p, 1.0 / temperature) / sum);
                    }
                }

                start = end;
            }
        }

        private static double ComputeLogLossScore(float[] predictions, float[] labels)
        {
            if (predictions.Length == 0)
            {
                return 0;
            }

            double sum = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                double p = Math.Clamp((double)predictions[i], 1e-7, 1 - 1e-7);
                double target = labels[i];
                sum -= target * Math.Log(p) + (1 - target) * Math.Log(1 - p);
            }
            return sum / predictions.Length;
        }

        private static double ComputeBrierScore(float[] predictions, float[] labels)
        {
            if (predictions.Length == 0)
            {
                return 0;
            }

            double sum = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                double diff = predictions[i] - labels[i];
                sum += diff * diff;
            }
            return sum / predictions.Length;
        }

        private static void ShuffleIndices(int[] values, Random random)
        {
            for (int i = values.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (values[i], values[j]) = (values[j], values[i]);
            }
        }

        private static float[][] ToJagged2DArray(NDArray array)
        {
            if (array.ndim != 2)
            {
                throw new InvalidOperationException($"Expected 2-D tensor but received rank {array.ndim}");
            }

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

        private ModelArtifact? PersistTrainedModel(TrainedModel model, bool activate)
        {
            try
            {
                var artifact = ModelArtifactStore.Save(model, _modelDirectory, activate);
                Console.WriteLine(
                    $"[AI] Persisted model artifact {artifact.FileName} " +
                    $"(sha256 {artifact.Sha256[..Math.Min(12, artifact.Sha256.Length)]}, active={activate}).");
                return artifact;
            }
            catch (Exception ex)
            {
                // Activating a model is part of the training contract: if the
                // artifact cannot be written, the run must fail rather than
                // leave MLParameters pointing at a missing file.
                if (activate)
                {
                    throw new InvalidOperationException(
                        $"The trained model could not be persisted to '{_modelDirectory}'.", ex);
                }

                Console.Error.WriteLine($"[AI] Could not persist model artifact: {ex.Message}");
                return null;
            }
        }
    }
}
