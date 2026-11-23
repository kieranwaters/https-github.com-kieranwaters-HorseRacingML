using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
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
        private readonly float _l2WeightDecay;
        private readonly float _auxiliaryWinLossWeight;
        private readonly float _auxiliaryFinishLossWeight;
        private readonly int _earlyStoppingPatience;
        private readonly int _evaluationFrequency;
        private readonly int _liveStateCacheMinutes;
        private readonly string _objective;
        private readonly bool _useMarketOffset;
        private readonly string _marketOffsetMode;
        private readonly bool _marketBlind;
        private readonly int _raceBatchSize;
        private readonly int _seed;
        private readonly double _recencyHalfLifeDays;
        private readonly FeatureBlocks _featureBlocks;
        private readonly bool _enablePaceFeatures;
        private readonly string _modelDirectory;
        private readonly bool _weatherEnabled;
        private readonly int _weatherLookbackDays;
        private readonly WeatherRepository? _weatherRepository;
        private readonly OpenMeteoWeatherClient? _weatherClient;
        private readonly RacingTvClient? _racingTvClient;
        private readonly RacingTvRepository? _racingTvRepository;
        private readonly System.Collections.Concurrent.ConcurrentDictionary<int, DateTime> _pedigreeFetchAttempts = new();

        private readonly object _liveStateLock = new();
        private FeatureEngineeringState? _liveState;
        private DateTime _liveStateBuiltUtc;
        private readonly object _weatherFetchLock = new();
        private DateTime _weatherFetchDisabledUntilUtc;

        public HyperparameterTrainer(
            IConfiguration configuration,
            RacingTvClient? racingTvClient = null)
        {
            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
            _racingTvClient = racingTvClient;
            _racingTvRepository = racingTvClient is null
                ? null
                : new RacingTvRepository(_connectionString);
            _winRateAlpha = configuration.GetValue<float>("WinRateAlpha", 1f);
            // A prior of alpha/beta = 1/9 is close to the base win rate. The
            // previous 1/2 prior made an unraced horse look like a 50% winner
            // and made 0/0 indistinguishable from 50/100.
            _winRateBeta = configuration.GetValue<float>("WinRateBeta", 9f);
            _l2WeightDecay = Math.Max(0f, configuration.GetValue<float?>("Training:L2WeightDecay") ?? 1e-4f);
            _auxiliaryWinLossWeight = ReadNonNegativeWeight(
                configuration,
                "Training:AuxiliaryWinLossWeight",
                0.2f);
            _auxiliaryFinishLossWeight = ReadNonNegativeWeight(
                configuration,
                "Training:AuxiliaryFinishLossWeight",
                0.2f);
            _earlyStoppingPatience = Math.Max(0, configuration.GetValue<int?>("Training:EarlyStoppingPatience") ?? 5);
            _evaluationFrequency = Math.Max(1, configuration.GetValue<int?>("Training:EvaluationFrequency") ?? 5);
            _liveStateCacheMinutes = Math.Max(1, configuration.GetValue<int?>("Training:LiveStateCacheMinutes") ?? 5);
            _objective = (configuration.GetValue<string>("Training:Objective") ?? "RaceSoftmax").Trim();
            _useMarketOffset = configuration.GetValue<bool?>("Training:UseMarketOffset") ?? true;
            _marketOffsetMode = NormalizeOffsetMode(
                configuration.GetValue<string>("Training:MarketOffsetMode") ?? "LogProb");
            _marketBlind = configuration.GetValue<bool?>("Training:MarketBlind") ?? false;
            _raceBatchSize = Math.Max(1, configuration.GetValue<int?>("Training:RaceBatchSize") ?? 64);
            _seed = configuration.GetValue<int?>("Training:Seed") ?? 42;
            _recencyHalfLifeDays = Math.Max(
                0d,
                configuration.GetValue<double?>("Training:RecencyHalfLifeDays") ?? 0d);
            // Controlled ablation switches. The default schema contains the
            // basic, movement, price, context, geometry, sectional and
            // breeding blocks; the remaining blocks can still be enabled
            // explicitly.
            var featureBlocks = FeatureBlockNames.Parse(
                configuration.GetValue<string>("Training:FeatureBlocks"),
                FeatureBlocks.Default);
            // Keep the original >= pace-features switch working.
            if (configuration.GetValue<bool?>("Training:EnablePaceFeatures") == false)
            {
                featureBlocks &= ~FeatureBlocks.Pace;
            }
            _featureBlocks = featureBlocks;
            _enablePaceFeatures = featureBlocks.HasFlag(FeatureBlocks.Pace);
            _modelDirectory = ModelArtifactStore.ResolveDirectory(
                configuration.GetValue<string>("Training:ModelDirectory"));
            _weatherEnabled = configuration.GetValue<bool?>("Weather:Enabled") ?? true;
            _weatherLookbackDays = Math.Clamp(configuration.GetValue<int?>("Weather:LookbackDays") ?? 20, 15, 30);
            if (_weatherEnabled)
            {
                _weatherRepository = new WeatherRepository(_connectionString);
                _weatherClient = new OpenMeteoWeatherClient(configuration);
            }
        }

        internal string ModelDirectory => _modelDirectory;

        internal bool EnablePaceFeatures => _enablePaceFeatures;

        internal float AuxiliaryWinLossWeight => _auxiliaryWinLossWeight;

        internal float AuxiliaryFinishLossWeight => _auxiliaryFinishLossWeight;

        public FeatureBlocks EnabledFeatureBlocks => _featureBlocks;

        internal bool IsFeatureBlockEnabled(FeatureBlocks block) => _featureBlocks.HasFlag(block);

        /// <summary>
        /// Last-resort pedigree lookup for a live runner that has no row in
        /// HorsePedigree yet (typically a first-time starter). The Racing TV
        /// horse search and profile endpoints are used, then the result is
        /// persisted so the next prediction is a database hit. Failures are
        /// non-fatal: the Breeding block falls back to neutral values.
        /// </summary>
        internal HorsePedigree? TryFetchPedigreeFromRacingTv(int horseId, string horseName)
        {
            if (!_featureBlocks.HasFlag(FeatureBlocks.Breeding) ||
                _racingTvClient is null ||
                _racingTvRepository is null ||
                horseId <= 0 ||
                string.IsNullOrWhiteSpace(horseName))
            {
                return null;
            }

            string normalizedName = FeatureText.NormalizeHorseName(horseName);
            if (normalizedName.Length == 0)
            {
                return null;
            }

            // Do not hammer the source for the same unknown horse on every
            // market scan. A missing profile is retried after a day.
            if (_pedigreeFetchAttempts.TryGetValue(horseId, out var lastAttempt) &&
                (DateTime.UtcNow - lastAttempt).TotalHours < 24)
            {
                return null;
            }
            _pedigreeFetchAttempts[horseId] = DateTime.UtcNow;

            try
            {
                var search = _racingTvClient.SearchHorsesAsync(horseName).GetAwaiter().GetResult();
                var hit = search?.Hits?
                    .Where(candidate => candidate.Horse is not null)
                    .FirstOrDefault(candidate =>
                        FeatureText.NormalizeHorseName(candidate.Horse!.Name) == normalizedName);
                if (hit?.Horse?.Slug is null)
                {
                    return null;
                }

                var profile = _racingTvClient
                    .GetHorseProfileAsync(hit.Horse.Slug)
                    .GetAwaiter()
                    .GetResult();
                var sourceHorse = profile?.Horse ?? hit.Horse;
                if (sourceHorse is null ||
                    (string.IsNullOrWhiteSpace(sourceHorse.SireName) &&
                     string.IsNullOrWhiteSpace(sourceHorse.DamName)))
                {
                    return null;
                }

                var row = new HorsePedigree
                {
                    HorseId = horseId,
                    SireName = CleanPedigreeName(sourceHorse.SireName),
                    DamName = CleanPedigreeName(sourceHorse.DamName),
                    DamSireName = CleanPedigreeName(sourceHorse.DamSireName),
                    Sex = sourceHorse.Gender,
                    Colour = sourceHorse.Colors is { Count: > 0 }
                        ? string.Join("/", sourceHorse.Colors)
                        : null,
                    FoalingDate = sourceHorse.FoalingDate,
                    SourceHorseId = sourceHorse.Id.HasValue &&
                                    sourceHorse.Id.Value > 0 &&
                                    sourceHorse.Id.Value <= int.MaxValue
                        ? (int?)sourceHorse.Id.Value
                        : null,
                    SourceSlug = sourceHorse.Slug,
                    Source = "RacingTV",
                    FetchedUtc = DateTime.UtcNow
                };
                _racingTvRepository.UpsertHorsePedigree(new[] { row });
                return row;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[breeding] Racing TV pedigree lookup failed for '{horseName}': {ex.Message}");
                return null;
            }
        }

        private static string? CleanPedigreeName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return null;
            }

            string cleaned = value.Trim();
            int bracket = cleaned.IndexOf('(');
            if (bracket > 0)
            {
                cleaned = cleaned[..bracket].Trim();
            }
            return cleaned.Length == 0 ? null : cleaned;
        }

        internal static string NormalizeOffsetMode(string? mode)
        {
            return string.Equals(mode?.Trim(), "Logit", StringComparison.OrdinalIgnoreCase)
                ? "Logit"
                : "LogProb";
        }

        /// <summary>
        /// Reads an auxiliary-loss weight. Negative, NaN and infinite values
        /// disable the loss rather than poisoning the graph.
        /// </summary>
        private static float ReadNonNegativeWeight(
            IConfiguration configuration,
            string key,
            float fallback)
        {
            float value = configuration.GetValue<float?>(key) ?? fallback;
            return float.IsFinite(value) && value > 0f ? value : 0f;
        }

        public class TrainingResult
        {
            public double TrainAccuracy { get; init; }
            public double TrainLoss { get; init; }
            public double TrainBrier { get; init; }
            public double ValidationAccuracy { get; init; }
            public double ValidationLoss { get; init; }
            public double ValidationBrier { get; init; }
            public double ValidationTemperature { get; init; } = 1.0;
            public double ValidationMarketLogLoss { get; init; }
            public double ValidationMarketBrier { get; init; }
            public string Objective { get; init; } = "RaceSoftmax";
            public float AuxiliaryWinLossWeight { get; init; }
            public float AuxiliaryFinishLossWeight { get; init; }
            public int? BestEpoch { get; init; }
            public float MarketOffsetScale { get; init; }
            public string MarketOffsetMode { get; init; } = "Logit";
            public bool MarketBlind { get; init; }
            public int Seed { get; init; }
            public string NormalizationMode { get; init; } = "PerRaceSoftmax";
            public IReadOnlyList<float> TrainingPredictions { get; init; } = Array.Empty<float>();
            public IReadOnlyList<float> TrainingLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> TrainingRaceIds { get; init; } = Array.Empty<int>();
            public IReadOnlyList<float> ValidationPredictions { get; init; } = Array.Empty<float>();
            public IReadOnlyList<float> ValidationLabels { get; init; } = Array.Empty<float>();
            public IReadOnlyList<int> ValidationRaceIds { get; init; } = Array.Empty<int>();
            public IReadOnlyList<RunnerExample> ValidationExamples { get; init; } = Array.Empty<RunnerExample>();
            public string? ModelArtifactFileName { get; init; }
            public string? ModelHash { get; init; }
            public string? FeatureSchemaHash { get; init; }
        }


        public TrainingDataset LoadTrainingDataset(bool includeIdentifiers = false)
        {
            return LoadEncodedTrainingDataset(includeIdentifiers: includeIdentifiers);
        }

        /// <summary>
        /// Loads the history in compact form. The database is read twice: the
        /// first pass derives the feature schema, the second encodes each race
        /// into float vectors as it arrives. The large per-runner dictionaries
        /// are released race-by-race instead of being retained for the whole
        /// history, which is what made the full dataset exceed available RAM.
        /// </summary>
        public TrainingDataset LoadEncodedTrainingDataset(
            ISet<int>? includeRaceIds = null,
            bool includeIdentifiers = false,
            DateTime? minimumRaceDate = null,
            bool requireMarketPrice = false,
            ISet<int>? metadataRaceIds = null,
            bool? marketBlind = null)
        {
            var metadata = BuildFeatureMetadataFromDatabase(
                metadataRaceIds ?? includeRaceIds,
                null,
                includeIdentifiers,
                minimumRaceDate,
                requireMarketPrice,
                marketBlind ?? _marketBlind);
            var races = EncodeRacesFromDatabase(
                metadata,
                includeRaceIds,
                null,
                includeIdentifiers,
                minimumRaceDate,
                requireMarketPrice);

            var normalization = new NormalizationParameters
            {
                Mean = new float[metadata.FeatureCount],
                StdDev = new float[metadata.FeatureCount]
            };

            return new TrainingDataset(
                races,
                new List<RaceExample>(),
                metadata.FeatureKeys,
                metadata.FeatureDimensions,
                metadata.StringMaps,
                normalization);
        }

        /// <summary>
        /// Splits an already-encoded chronological dataset without copying any
        /// feature vectors. The lists only hold references to the same compact
        /// <see cref="RaceExample"/> instances.
        /// </summary>
        public static TrainingDataset SplitDataset(
            TrainingDataset source,
            ISet<int> trainingRaceIds,
            ISet<int> validationRaceIds)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (trainingRaceIds is null)
                throw new ArgumentNullException(nameof(trainingRaceIds));
            if (validationRaceIds is null)
                throw new ArgumentNullException(nameof(validationRaceIds));

            var trainingRaces = source.Races
                .Where(race => trainingRaceIds.Contains(race.RaceId))
                .ToList();
            var validationRaces = source.Races
                .Where(race => validationRaceIds.Contains(race.RaceId))
                .ToList();

            return new TrainingDataset(
                trainingRaces,
                validationRaces,
                source.FeatureKeys,
                source.FeatureDimensions,
                source.StringMaps,
                source.Normalization);
        }

        public TrainingResult Train(
            MLParameter param,
            int foldIndex,
            int foldCount,
            bool persistWeights = true,
            bool activateModel = false,
            CancellationToken cancellationToken = default)
        {
            var dataset = LoadTrainingDataset();
            return Train(param, foldIndex, foldCount, dataset, persistWeights, cancellationToken: cancellationToken, activateModel: activateModel);
        }
        public record PricedSubsetMetrics(double Accuracy, double LogLoss, double Brier, int RaceCount, int RunnerCount);

        /// <summary>
        /// Evaluates only the runners that had a pre-off price and only races
        /// whose winner was priced. Predictions are renormalised within each
        /// race. This is a diagnostic subset: live rows have prices for every
        /// runner, so the full-field metric remains the closer deployment proxy.
        /// </summary>
        public static PricedSubsetMetrics ComputePricedSubsetMetrics(TrainingResult result)
        {
            if (result is null)
                throw new ArgumentNullException(nameof(result));

            var predictions = result.ValidationPredictions;
            var labels = result.ValidationLabels;
            var raceIds = result.ValidationRaceIds;
            var examples = result.ValidationExamples;
            int count = predictions.Count;
            if (count == 0 || labels.Count != count || raceIds.Count != count || examples.Count != count)
            {
                return new PricedSubsetMetrics(0, 0, 0, 0, 0);
            }

            double logLoss = 0;
            double brier = 0;
            int runnerCount = 0;
            int raceCount = 0;
            int correct = 0;

            foreach (var group in raceIds.Select((raceId, index) => (raceId, index)).GroupBy(x => x.raceId))
            {
                var indices = group
                    .Select(x => x.index)
                    .Where(i => examples[i].HasMarketPrice)
                    .ToList();
                if (indices.Count < 2)
                {
                    continue;
                }

                var trueIndices = indices.Where(i => labels[i] >= 0.5f).ToList();
                if (trueIndices.Count == 0)
                {
                    // The winner had no price; a priced-subset accuracy or loss
                    // cannot be interpreted for this race.
                    continue;
                }

                double sum = 0;
                foreach (var i in indices)
                {
                    if (float.IsFinite(predictions[i]) && predictions[i] > 0f)
                    {
                        sum += predictions[i];
                    }
                }
                if (sum <= double.Epsilon || !double.IsFinite(sum))
                {
                    continue;
                }

                int bestIndex = indices[0];
                double bestProbability = double.NegativeInfinity;
                foreach (var i in indices)
                {
                    double probability = Math.Max(0f, predictions[i]) / sum;
                    if (probability > bestProbability)
                    {
                        bestProbability = probability;
                        bestIndex = i;
                    }

                    double q = Math.Clamp(probability, 1e-7, 1d - 1e-7);
                    double y = labels[i];
                    logLoss -= y * Math.Log(q) + (1d - y) * Math.Log(1d - q);
                    brier += (q - y) * (q - y);
                    runnerCount++;
                }

                raceCount++;
                if (trueIndices.Contains(bestIndex))
                {
                    correct++;
                }
            }

            if (runnerCount == 0 || raceCount == 0)
            {
                return new PricedSubsetMetrics(0, 0, 0, 0, 0);
            }

            return new PricedSubsetMetrics(
                (double)correct / raceCount,
                logLoss / runnerCount,
                brier / runnerCount,
                raceCount,
                runnerCount);
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

            var metadata = BuildFeatureMetadata(metadataSource, metadataRows, _marketBlind);
            int featureCount = metadata.FeatureCount;
            var trainRaces = EncodeRaces(trainingPrepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);
            var validationRaces = EncodeRaces(validationPrepared.Races, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps);

            var normalization = new NormalizationParameters
            {
                Mean = new float[featureCount],
                StdDev = new float[featureCount]
            };

            return new TrainingDataset(trainRaces, validationRaces, metadata.FeatureKeys, metadata.FeatureDimensions, metadata.StringMaps, normalization);
        }

        public TrainingResult Train(
            MLParameter param,
            TrainingDataset.PreparedDataset dataset,
            int foldIndex,
            int foldCount,
            bool persistWeights = true,
            float? temperatureOverride = null,
            bool activateModel = false,
            CancellationToken cancellationToken = default)
        {
            var trainingDataset = BuildFoldTrainingDataset(dataset, foldIndex, foldCount);
            return Train(
                param,
                foldIndex,
                foldCount,
                trainingDataset,
                persistWeights,
                temperatureOverride,
                cancellationToken: cancellationToken,
                activateModel: activateModel);
        }

        /// <summary>
        /// Slices the prepared chronological rows for one walk-forward fold and
        /// encodes them once. A hyperparameter sweep can cache the result and reuse
        /// it for every model with the same fold layout.
        /// </summary>
        public TrainingDataset BuildFoldTrainingDataset(
            TrainingDataset.PreparedDataset dataset,
            int foldIndex,
            int foldCount)
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

            if (valStart <= 0)
            {
                throw new InvalidOperationException("Walk-forward fold has no earlier races to train on.");
            }

            // The prepared rows come from a single chronological pass, and every row's
            // features only depend on races that occurred before it. Re-preparing the
            // database for a fold would therefore reproduce exactly these values, so
            // slice the prepared rows instead of scanning and engineering the whole
            // history again for every fold and every model.
            var trainingPrepared = new TrainingDataset.PreparedDataset(
                dataset.Races.Take(valStart).ToList());
            var validationPrepared = valEnd > valStart
                ? new TrainingDataset.PreparedDataset(
                    dataset.Races.Skip(valStart).Take(valEnd - valStart).ToList())
                : new TrainingDataset.PreparedDataset(
                    new List<TrainingDataset.PreparedDataset.PreparedRace>());

            return BuildTrainingDataset(trainingPrepared, validationPrepared);
        }

        /// <summary>
        /// Slices an already-encoded chronological dataset for a walk-forward
        /// fold. This is the low-memory counterpart used by the training
        /// screens: no dictionaries are rebuilt and no feature vectors are
        /// copied.
        /// </summary>
        public TrainingDataset BuildFoldTrainingDataset(
            TrainingDataset dataset,
            int foldIndex,
            int foldCount)
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

            if (valStart <= 0)
            {
                throw new InvalidOperationException("Walk-forward fold has no earlier races to train on.");
            }

            var trainingRaces = dataset.Races.Take(valStart).ToList();
            var validationRaces = valEnd > valStart
                ? dataset.Races.Skip(valStart).Take(valEnd - valStart).ToList()
                : new List<RaceExample>();

            return new TrainingDataset(
                trainingRaces,
                validationRaces,
                dataset.FeatureKeys,
                dataset.FeatureDimensions,
                dataset.StringMaps,
                dataset.Normalization);
        }

        public TrainingResult Train(
            MLParameter param,
            int foldIndex,
            int foldCount,
            TrainingDataset dataset,
            bool persistWeights = true,
            float? temperatureOverride = null,
            bool earlyStopOnValidation = true,
            double blendModelWeight = 1d,
            double blendMarketWeight = 0d,
            bool activateModel = false,
            CancellationToken cancellationToken = default)
        {
            if (dataset is null)
                throw new ArgumentNullException(nameof(dataset));

            var objective = string.IsNullOrWhiteSpace(param.Objective) ? _objective : param.Objective;
            if (!string.Equals(objective, "Binary", StringComparison.OrdinalIgnoreCase))
            {
                return TrainRaceSoftmax(
                    param,
                    dataset,
                    persistWeights,
                    temperatureOverride,
                    earlyStopOnValidation,
                    blendModelWeight,
                    blendMarketWeight,
                    activateModel,
                    cancellationToken);
            }

            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

            int binarySeed = param.Seed ?? _seed;

            _ = foldIndex;
            _ = foldCount;

            var trainExamples = dataset.TrainingRaces
                .SelectMany(r => r.Runners)
                .ToList();
            var valExamples = dataset.ValidationRaces
                .SelectMany(r => r.Runners)
                .ToList();

            int featureCount = dataset.FeatureCount;
            var trainLabels = trainExamples.Select(r => r.Label).ToArray();
            var trainRaceIds = trainExamples.Select(r => r.RaceId).ToArray();
            var valLabels = valExamples.Select(r => r.Label).ToArray();
            var valRaceIds = valExamples.Select(r => r.RaceId).ToArray();

            var means = new float[featureCount];
            var stdDevs = new float[featureCount];

            if (trainExamples.Count > 0)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < trainExamples.Count; i++)
                    {
                        sum += trainExamples[i].Features[j];
                    }
                    means[j] = (float)(sum / trainExamples.Count);
                }

                for (int j = 0; j < featureCount; j++)
                {
                    double variance = 0;
                    for (int i = 0; i < trainExamples.Count; i++)
                    {
                        double diff = trainExamples[i].Features[j] - means[j];
                        variance += diff * diff;
                    }
                    stdDevs[j] = (float)Math.Sqrt(variance / trainExamples.Count);
                    if (stdDevs[j] == 0f)
                    {
                        stdDevs[j] = 1f;
                    }
                }
            }
            else
            {
                for (int j = 0; j < featureCount; j++)
                {
                    stdDevs[j] = 1f;
                }
            }

            // Normalise one batch at a time rather than materialising a full
            // float[,] copy of the history and then a second full NDArray copy.
            // The encoded feature vectors are already compact, so peak memory
            // stays close to the dataset itself.
            static void FillBatch(
                IReadOnlyList<RunnerExample> source,
                int start,
                int count,
                int featureCount,
                float[] means,
                float[] stdDevs,
                float[,] features,
                float[,] labels)
            {
                for (int i = 0; i < count; i++)
                {
                    var example = source[start + i];
                    var row = example.Features;
                    for (int j = 0; j < featureCount; j++)
                    {
                        features[i, j] = (row[j] - means[j]) / stdDevs[j];
                    }
                    labels[i, 0] = example.Label;
                }
            }

            var graph = tf.Graph().as_default();
            try { tf.set_random_seed(binarySeed); } catch { }

            var x = tf.placeholder(tf.float32, shape: new TensorShape(-1, featureCount), name: "x");
            var y = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "y");
            var finishTarget = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "finishTarget");
            var finishMask = tf.placeholder(tf.float32, shape: new TensorShape(-1, 1), name: "finishMask");
            Tensor trainingLayer = x;
            Tensor inferenceLayer = x;
            int inputDim = featureCount;
            var hiddenWeightVars = new List<ResourceVariable>();
            var hiddenBiasVars = new List<ResourceVariable>();
            for (int i = 0; i < param.Layers; i++)
            {
                // He initialisation. N(0, 1) (the previous default) made the initial
                // activations and logits enormous for 300+ features, so the network
                // spent its first epochs recovering instead of learning.
                var stdDev = (float)Math.Sqrt(2.0 / Math.Max(1, inputDim));
                var w = tf.Variable(tf.random.normal((inputDim, param.Units), stddev: stdDev), name: $"w{i}");
                var b = tf.Variable(tf.zeros(param.Units), name: $"b{i}");
                hiddenWeightVars.Add(w);
                hiddenBiasVars.Add(b);
                trainingLayer = tf.nn.relu(tf.matmul(trainingLayer, w) + b);
                inferenceLayer = tf.nn.relu(tf.matmul(inferenceLayer, w) + b);
                if (param.Dropout > 0)
                {
                    trainingLayer = tf.nn.dropout(trainingLayer, rate: (float)param.Dropout);
                }
                inputDim = param.Units;
            }

            var wOut = tf.Variable(tf.random.normal((inputDim, 1), stddev: (float)Math.Sqrt(1.0 / Math.Max(1, inputDim))), name: "wOut");
            var bOut = tf.Variable(tf.zeros(1), name: "bOut");
            var trainingLogits = tf.matmul(trainingLayer, wOut) + bOut;
            var inferenceLogits = tf.matmul(inferenceLayer, wOut) + bOut;

            // The binary objective already supervises the winner label, so only
            // the finishing-position auxiliary head is added here. RaceSoftmax
            // also receives an auxiliary pointwise win head.
            float binaryAuxFinishWeight = _auxiliaryFinishLossWeight;
            ResourceVariable? wAuxFinish = null;
            ResourceVariable? bAuxFinish = null;
            Tensor auxFinishLoss = tf.constant(0f);
            if (binaryAuxFinishWeight > 0f)
            {
                float auxStdDev = (float)Math.Sqrt(1.0 / Math.Max(1, inputDim));
                wAuxFinish = tf.Variable(
                    tf.random.normal((inputDim, 1), stddev: auxStdDev),
                    name: "wAuxFinish");
                bAuxFinish = tf.Variable(tf.zeros(1), name: "bAuxFinish");
                var auxFinishLogits = tf.matmul(trainingLayer, wAuxFinish) + bAuxFinish;
                var perRunnerFinishLoss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels: finishTarget,
                    logits: auxFinishLogits);
                auxFinishLoss = tf.reduce_sum(perRunnerFinishLoss * finishMask) /
                                tf.maximum(tf.reduce_sum(finishMask), tf.constant(1f));
            }

            var trainingDataLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: y, logits: trainingLogits));
            var evaluationLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: y, logits: inferenceLogits));
            Tensor l2Penalty = tf.constant(0f);
            foreach (var weight in hiddenWeightVars)
            {
                l2Penalty += tf.reduce_sum(tf.square(weight));
            }
            l2Penalty += tf.reduce_sum(tf.square(wOut));
            if (wAuxFinish is not null)
            {
                l2Penalty += tf.reduce_sum(tf.square(wAuxFinish));
            }
            var totalLoss = trainingDataLoss + (l2Penalty * _l2WeightDecay);
            if (binaryAuxFinishWeight > 0f)
            {
                totalLoss = totalLoss + (binaryAuxFinishWeight * auxFinishLoss);
            }
            var optimizer = tf.train.AdamOptimizer((float)param.LearningRate).minimize(totalLoss);

            // Inference always uses the non-dropout branch so validation metrics
            // and persisted predictions are deterministic and match the live
            // forward pass in AIOddsCalculator.
            var prediction = tf.sigmoid(inferenceLogits);
            var rnd = new Random(binarySeed);
            var trainableVariables = tf.trainable_variables();
            using var sess = tf.Session(graph);
            sess.run(tf.global_variables_initializer());

            var trainPreds = new float[trainLabels.Length];
            var valPreds = new float[valLabels.Length];
            var epochPredBuffer = new float[trainLabels.Length];

            double ComputeDatasetMetrics(IReadOnlyList<RunnerExample> examples, float[] preds)
            {
                int count = examples.Count;
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
                    var featureBatch = new float[batchCount, featureCount];
                    var labelBatch = new float[batchCount, 1];
                    FillBatch(examples, start, batchCount, featureCount, means, stdDevs, featureBatch, labelBatch);

                    var results = sess.run(new[] { evaluationLoss, prediction },
                        new FeedItem(x, np.array(featureBatch, dtype: tf.float32)),
                        new FeedItem(y, np.array(labelBatch, dtype: tf.float32)));

                    var chunkLoss = results[0].ToArray<float>()[0];
                    var chunkPreds = results[1].ToArray<float>();
                    Array.Copy(chunkPreds, 0, preds, start, batchCount);
                    weightedLoss += chunkLoss * batchCount;
                    totalExamples += batchCount;
                }

                return totalExamples > 0 ? weightedLoss / totalExamples : 0;
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

            double ComputeLogLoss(float[] preds, float[] labels)
            {
                if (preds.Length == 0)
                {
                    return 0;
                }

                double sum = 0;
                for (int i = 0; i < preds.Length; i++)
                {
                    double p = Math.Clamp((double)preds[i], 1e-7, 1 - 1e-7);
                    double y = labels[i];
                    sum -= y * Math.Log(p) + (1 - y) * Math.Log(1 - p);
                }

                return sum / preds.Length;
            }

            static double FitTemperature(IReadOnlyList<float> predictions, IReadOnlyList<float> labels)
            {
                if (predictions.Count == 0 || predictions.Count != labels.Count)
                {
                    return 1.0;
                }

                double bestTemperature = 1.0;
                double bestLoss = double.MaxValue;
                for (double temperature = 0.25; temperature <= 4.0; temperature += 0.05)
                {
                    double loss = 0;
                    for (int i = 0; i < predictions.Count; i++)
                    {
                        double p = Math.Clamp((double)predictions[i], 1e-7, 1 - 1e-7);
                        double logit = Math.Log(p / (1 - p));
                        double calibrated = Sigmoid(logit / temperature);
                        calibrated = Math.Clamp(calibrated, 1e-7, 1 - 1e-7);
                        double y = labels[i];
                        loss -= y * Math.Log(calibrated) + (1 - y) * Math.Log(1 - calibrated);
                    }

                    if (loss < bestLoss)
                    {
                        bestLoss = loss;
                        bestTemperature = temperature;
                    }
                }

                return bestTemperature;
            }

            static double ApplyTemperature(float probability, double temperature)
            {
                double p = Math.Clamp((double)probability, 1e-7, 1 - 1e-7);
                double logit = Math.Log(p / (1 - p));
                return Sigmoid(logit / temperature);
            }

            double trainLoss = 0;
            double valLoss = 0;
            double trainBrier = 0;
            double valBrier = 0;
            double trainAcc = 0;
            double valAcc = 0;
            double validationTemperature = 1.0;
            int bestEpoch = -1;
            double bestValidationLoss = double.MaxValue;
            NDArray[]? bestVariableValues = null;
            var valEpochPredBuffer = new float[valLabels.Length];
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
                    cancellationToken.ThrowIfCancellationRequested();
                    var indices = new int[trainExamples.Count];
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
                        cancellationToken.ThrowIfCancellationRequested();
                        int batchCount = Math.Min(param.BatchSize, indices.Length - start);
                        if (batchCount <= 0)
                        {
                            continue;
                        }

                        // Build and normalise only the current mini-batch. The
                        // previous path first copied every training row into a
                        // full float[,] and then into a full NDArray.
                        var batchFeatures = new float[batchCount, featureCount];
                        var batchLabels = new float[batchCount, 1];
                        var batchFinishTargets = new float[batchCount, 1];
                        var batchFinishMask = new float[batchCount, 1];
                        for (int i = 0; i < batchCount; i++)
                        {
                            var example = trainExamples[indices[start + i]];
                            var row = example.Features;
                            for (int j = 0; j < featureCount; j++)
                            {
                                batchFeatures[i, j] = (row[j] - means[j]) / stdDevs[j];
                            }
                            batchLabels[i, 0] = example.Label;
                            if (example.HasFinishTarget)
                            {
                                batchFinishTargets[i, 0] = example.FinishTarget;
                                batchFinishMask[i, 0] = 1f;
                            }
                        }

                        if (binaryAuxFinishWeight > 0f)
                        {
                            sess.run(
                                optimizer,
                                new FeedItem(x, np.array(batchFeatures, dtype: tf.float32)),
                                new FeedItem(y, np.array(batchLabels, dtype: tf.float32)),
                                new FeedItem(finishTarget, np.array(batchFinishTargets, dtype: tf.float32)),
                                new FeedItem(finishMask, np.array(batchFinishMask, dtype: tf.float32)));
                        }
                        else
                        {
                            sess.run(
                                optimizer,
                                new FeedItem(x, np.array(batchFeatures, dtype: tf.float32)),
                                new FeedItem(y, np.array(batchLabels, dtype: tf.float32)));
                        }
                    }

                    bool shouldLog = ((epoch + 1) % _evaluationFrequency == 0) || epoch == param.Epochs - 1;
                    if (shouldLog)
                    {
                        _ = ComputeDatasetMetrics(trainExamples, epochPredBuffer);
                        NormalizePerRace(epochPredBuffer, trainRaceIds);
                        var epochLoss = ComputeLogLossScore(epochPredBuffer, trainLabels);
                        var epochAcc = trainLabels.Length > 0 ? ComputeWinnerAccuracy(trainRaceIds, epochPredBuffer, trainLabels) : 0;
                        Console.WriteLine($"Epoch {epoch + 1}/{param.Epochs} - per-race loss: {epochLoss:F4} - winner acc: {epochAcc:F4}");
                    }

                    if (earlyStopOnValidation && valLabels.Length > 0)
                    {
                        _ = ComputeDatasetMetrics(valExamples, valEpochPredBuffer);
                        NormalizePerRace(valEpochPredBuffer, valRaceIds);
                        double epochValidationLoss = ComputeLogLossScore(valEpochPredBuffer, valLabels);

                        if (double.IsFinite(epochValidationLoss) &&
                            (bestVariableValues is null || epochValidationLoss < bestValidationLoss - 1e-6))
                        {
                            bestValidationLoss = epochValidationLoss;
                            bestEpoch = epoch;
                            bestVariableValues = sess.run(trainableVariables);
                        }
                        else if (_earlyStoppingPatience > 0 &&
                                 bestEpoch >= 0 &&
                                 epoch - bestEpoch >= _earlyStoppingPatience)
                        {
                            Console.WriteLine(
                                $"Early stopping at epoch {epoch + 1}/{param.Epochs}; best validation loss {bestValidationLoss:F4} at epoch {bestEpoch + 1}.");
                            break;
                        }
                    }
                }

                if (!earlyStopOnValidation)
                {
                    bestEpoch = param.Epochs - 1;
                }

                if (bestVariableValues is not null && trainableVariables.Length > 0)
                {
                    var restoreOps = trainableVariables
                        .Select((variable, index) => tf.assign(variable, bestVariableValues[index]))
                        .ToArray();
                    sess.run(restoreOps);
                }

                _ = ComputeDatasetMetrics(trainExamples, trainPreds);
                _ = ComputeDatasetMetrics(valExamples, valPreds);

                // The live pipeline normalises probabilities within each race
                // before they are used, so validation and early stopping must
                // use the same probability space.
                NormalizePerRace(trainPreds, trainRaceIds);
                NormalizePerRace(valPreds, valRaceIds);

                validationTemperature = temperatureOverride.HasValue
                    ? temperatureOverride.Value
                    : (valPreds.Length > 0 && valLabels.Length == valPreds.Length
                        ? FitTemperaturePerRace(valPreds, valLabels, valRaceIds)
                        : 1.0);

                ApplyTemperaturePerRace(valPreds, valRaceIds, validationTemperature);
                ApplyTemperaturePerRace(trainPreds, trainRaceIds, validationTemperature);

                trainLoss = ComputeLogLossScore(trainPreds, trainLabels);
                trainBrier = ComputeBrierScore(trainPreds, trainLabels);
                valLoss = ComputeLogLossScore(valPreds, valLabels);
                valBrier = ComputeBrierScore(valPreds, valLabels);

                trainAcc = trainPreds.Length > 0 ? ComputeWinnerAccuracy(trainRaceIds, trainPreds, trainLabels) : 0;
                valAcc = valPreds.Length > 0 ? ComputeWinnerAccuracy(valRaceIds, valPreds, valLabels) : 0;
            }
            catch (Exception ex)
            {
                HyperparameterFailed(param, ex);
                throw;
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
                Temperature = temperatureOverride ?? (float)validationTemperature,
                Objective = "Binary",
                AuxiliaryWinLossWeight = 0f,
                AuxiliaryFinishLossWeight = binaryAuxFinishWeight,
                MarketOffsetScale = 0f,
                NormalizationMode = "PerRaceNormalized",
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
            model.FeatureSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(model.Metadata);
            model.TrainedUtc = DateTime.UtcNow;

            ModelArtifact? artifact = null;
            if (persistWeights)
            {
                artifact = PersistTrainedModel(model, activateModel);
            }

            HyperparameterCompleted(param, trainAcc, valAcc, trainLoss, valLoss, trainBrier, valBrier);

            try { Tensorflow.ops.reset_default_graph(); } catch { }
            GC.Collect();
            GC.WaitForPendingFinalizers();

            return new TrainingResult
            {
                TrainAccuracy = trainAcc,
                TrainLoss = trainLoss,
                TrainBrier = trainBrier,
                ValidationAccuracy = valAcc,
                ValidationLoss = valLoss,
                ValidationBrier = valBrier,
                ValidationTemperature = validationTemperature,
                Objective = "Binary",
                AuxiliaryWinLossWeight = 0f,
                AuxiliaryFinishLossWeight = binaryAuxFinishWeight,
                BestEpoch = bestEpoch >= 0 ? bestEpoch : null,
                MarketOffsetScale = 0f,
                Seed = param.Seed ?? _seed,
                NormalizationMode = "PerRaceNormalized",
                TrainingPredictions = Array.AsReadOnly(trainPreds),
                TrainingLabels = Array.AsReadOnly(trainLabels),
                TrainingRaceIds = Array.AsReadOnly(trainRaceIds),
                ValidationPredictions = Array.AsReadOnly(valPreds),
                ValidationLabels = Array.AsReadOnly(valLabels),
                ValidationRaceIds = Array.AsReadOnly(valRaceIds),
                ValidationExamples = new ReadOnlyCollection<RunnerExample>(valExamples),
                ModelArtifactFileName = artifact?.FileName,
                ModelHash = artifact?.Sha256,
                FeatureSchemaHash = model.FeatureSchemaHash
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
        public TrainingDataset LoadTrainingDataset(
            ISet<int> trainingRaceIds,
            ISet<int> validationRaceIds,
            bool includeIdentifiers = false,
            bool requireMarketPrice = false,
            ISet<int>? metadataRaceIds = null,
            bool? marketBlind = null)
        {
            if (trainingRaceIds is null)
                throw new ArgumentNullException(nameof(trainingRaceIds));
            if (validationRaceIds is null)
                throw new ArgumentNullException(nameof(validationRaceIds));

            // One chronological pass is enough: state advances over every race
            // regardless of which races are returned, so the second scan that this
            // method used to perform reproduced identical feature values.
            return LoadTrainingDatasetSinglePass(
                trainingRaceIds,
                validationRaceIds,
                includeIdentifiers,
                requireMarketPrice: requireMarketPrice,
                metadataRaceIds: metadataRaceIds,
                marketBlind: marketBlind);
        }

        /// <summary>
        /// Builds both the holdout dataset and the full-data dataset from one
        /// feature-engineering pass. Used by TrainAI so that the final retrain does
        /// not scan and engineer the whole history a second time.
        /// </summary>
        public (TrainingDataset Holdout, TrainingDataset Full) LoadTrainingDatasetsSinglePass(
            ISet<int> trainingRaceIds,
            ISet<int> validationRaceIds,
            bool includeIdentifiers = false,
            DateTime? minimumRaceDate = null,
            bool requireMarketPrice = false,
            ISet<int>? metadataRaceIds = null,
            bool? marketBlind = null)
        {
            if (trainingRaceIds is null) throw new ArgumentNullException(nameof(trainingRaceIds));
            if (validationRaceIds is null) throw new ArgumentNullException(nameof(validationRaceIds));

            var included = new HashSet<int>(trainingRaceIds);
            included.UnionWith(validationRaceIds);

            // Both views share the same encoded RaceExample instances; only the
            // small list of references is duplicated.
            var full = LoadEncodedTrainingDataset(
                included,
                includeIdentifiers,
                minimumRaceDate,
                requireMarketPrice,
                metadataRaceIds,
                marketBlind);
            var holdout = SplitDataset(full, trainingRaceIds, validationRaceIds);
            return (holdout, full);
        }

        /// <summary>
        /// Builds a train/validation dataset with a single feature-engineering pass.
        /// Existing validation rows keep the chronological history of earlier races,
        /// including earlier validation rows, exactly like the two-pass overload.
        /// </summary>
        public TrainingDataset LoadTrainingDatasetSinglePass(
            ISet<int> trainingRaceIds,
            ISet<int> validationRaceIds,
            bool includeIdentifiers = false,
            DateTime? minimumRaceDate = null,
            bool requireMarketPrice = false,
            ISet<int>? metadataRaceIds = null,
            bool? marketBlind = null)
        {
            if (trainingRaceIds is null) throw new ArgumentNullException(nameof(trainingRaceIds));
            if (validationRaceIds is null) throw new ArgumentNullException(nameof(validationRaceIds));

            var included = new HashSet<int>(trainingRaceIds);
            included.UnionWith(validationRaceIds);

            var full = LoadEncodedTrainingDataset(
                included,
                includeIdentifiers,
                minimumRaceDate,
                requireMarketPrice,
                metadataRaceIds,
                marketBlind);
            return SplitDataset(full, trainingRaceIds, validationRaceIds);
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
            var weightPath = ModelArtifactStore.ResolveActivePath(_modelDirectory);
            if (string.IsNullOrWhiteSpace(weightPath) || !File.Exists(weightPath))
            {
                Console.Error.WriteLine($"[AI] Trained weight file not found in {_modelDirectory}.");
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

        public IReadOnlyList<RunnerProbability> PredictRaceProbabilities(
            TrainingDataset.PreparedDataset.PreparedRace race,
            TrainedModel model)
        {
            if (race is null)
                throw new ArgumentNullException(nameof(race));
            if (model is null)
                throw new ArgumentNullException(nameof(model));

            var runtime = TrainedModelRuntime.FromModel(model, out _);
            if (runtime is null || !runtime.IsRaceSoftmax)
            {
                return Array.Empty<RunnerProbability>();
            }

            var rows = race.Rows
                .Where(row => row is not null)
                .Select(row => (IDictionary<string, object?>)row)
                .ToList();
            if (rows.Count == 0 || !runtime.TryPredictRace(rows, out var probabilities))
            {
                return Array.Empty<RunnerProbability>();
            }

            var results = new List<RunnerProbability>(rows.Count);
            for (var i = 0; i < rows.Count && i < probabilities.Length; i++)
            {
                var row = rows[i];
                int? clothNumber = null;
                if (row.TryGetValue("SaddleclothNumber", out var clothObj) &&
                    TrainingDataset.PreparedDataset.TryConvertToInt32(clothObj, out var cloth))
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

                var probability = probabilities[i];
                if (double.IsFinite(probability) && probability > 0)
                {
                    results.Add(new RunnerProbability(
                        race.RaceId,
                        clothNumber,
                        horseName,
                        selectionId,
                        probability));
                }
            }

            return results;
        }

        private static double GetMarketProbability(Dictionary<string, object?> row)
        {
            if (!row.TryGetValue("MarketImpliedProb", out var value) || value is null)
            {
                return 0d;
            }

            try
            {
                double probability = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                if (probability > 1e-6 && probability < 1d - 1e-6)
                {
                    return probability;
                }
            }
            catch
            {
                // Missing/invalid market probability means no offset.
            }

            return 0d;
        }

        private static double[] BuildMarketOffsets(double[] marketProbabilities, bool useLogProbOffset)
        {
            var offsets = new double[marketProbabilities.Length];
            if (!useLogProbOffset)
            {
                for (int i = 0; i < marketProbabilities.Length; i++)
                {
                    double probability = marketProbabilities[i];
                    offsets[i] = probability > 1e-6 && probability < 1d - 1e-6
                        ? Math.Log(probability / (1d - probability))
                        : 0d;
                }

                return offsets;
            }

            int priced = 0;
            double sum = 0d;
            for (int i = 0; i < marketProbabilities.Length; i++)
            {
                double probability = marketProbabilities[i];
                if (probability > 1e-6 && probability < 1d - 1e-6)
                {
                    offsets[i] = Math.Log(probability);
                    sum += offsets[i];
                    priced++;
                }
            }

            double mean = priced > 0 ? sum / priced : 0d;
            double allSum = 0d;
            for (int i = 0; i < offsets.Length; i++)
            {
                if (marketProbabilities[i] <= 1e-6 || marketProbabilities[i] >= 1d - 1e-6)
                {
                    offsets[i] = mean;
                }
                allSum += offsets[i];
            }

            double centre = offsets.Length > 0 ? allSum / offsets.Length : 0d;
            for (int i = 0; i < offsets.Length; i++)
            {
                offsets[i] -= centre;
            }

            return offsets;
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
            return Sigmoid(ForwardPassLogit(inputs, hiddenLayers, outputLayer));
        }

        private static double ForwardPassLogit(double[] inputs, List<LayerWeights> hiddenLayers, LayerWeights outputLayer)
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

            return output[0];
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