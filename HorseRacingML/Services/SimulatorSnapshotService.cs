using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Builds and publishes a simulator snapshot. This was extracted from the
    /// controller so the background job can call it directly and so Train AI,
    /// Test AI and the simulator share one snapshot shape.
    /// </summary>
    public sealed class SimulatorSnapshotService
    {
        private readonly RacingRepository _repository;
        private readonly HyperparameterTrainer _trainer;
        private readonly BettingSimulatorStore _store;
        private readonly ILogger<SimulatorSnapshotService> _logger;
        private readonly DateTime _minimumRaceDate;
        private readonly int _maxTrainingRaces;
        private readonly int _maxRunners;
        private readonly bool _keepUnpricedRunners;
        private readonly bool _requirePreOffPrice;
        private readonly int _seed;
        private readonly int _raceBatchSize;
        private readonly double _winRateAlpha;
        private readonly double _winRateBeta;
        private readonly string? _modelDirectory;

        public SimulatorSnapshotService(
            RacingRepository repository,
            HyperparameterTrainer trainer,
            BettingSimulatorStore store,
            IConfiguration configuration,
            ILogger<SimulatorSnapshotService> logger)
        {
            _repository = repository;
            _trainer = trainer;
            _store = store;
            _logger = logger;
            _minimumRaceDate = configuration.GetValue<DateTime?>("Training:SimulatorMinimumRaceDate")
                ?? new DateTime(2017, 1, 1);
            _maxTrainingRaces = Math.Max(
                1_000,
                configuration.GetValue<int?>("Training:SimulatorMaxTrainingRaces") ?? 200_000);
            _maxRunners = Math.Max(
                10_000,
                configuration.GetValue<int?>("Training:SimulatorMaxRunners") ?? 1_000_000);
            _keepUnpricedRunners = configuration.GetValue<bool?>("Training:KeepUnpricedRunners") ?? true;
            _requirePreOffPrice = configuration.GetValue<bool?>("Training:RequirePreOffPrice") ?? true;
            _seed = configuration.GetValue<int?>("Training:Seed") ?? 42;
            _raceBatchSize = Math.Max(1, configuration.GetValue<int?>("Training:RaceBatchSize") ?? 64);
            _winRateAlpha = configuration.GetValue<double?>("WinRateAlpha") ?? 1.0;
            _winRateBeta = configuration.GetValue<double?>("WinRateBeta") ?? 9.0;
            _modelDirectory = ModelArtifactStore.ResolveDirectory(
                configuration.GetValue<string>("Training:ModelDirectory"));
        }

        public async Task<(bool Success, string? Error, string? SnapshotKey)> PrepareAndStoreAsync(
            SimulatorRunRequest request,
            Action<string, int?> report,
            CancellationToken cancellationToken)
        {
            request ??= new SimulatorRunRequest();
            report ??= (_, _) => { };

            int months = Math.Clamp(
                request.Months,
                SimulatorRunRequest.MinimumMonths,
                SimulatorRunRequest.MaximumMonths);
            int? windowMonths = request.FromDate.HasValue || request.ToDate.HasValue
                ? null
                : months;
            var latestRaceDate = _repository.GetLatestRaceDate();
            var validationEnd = request.ToDate?.Date ?? latestRaceDate.Date;
            var validationStart = request.FromDate?.Date ?? validationEnd.AddMonths(-months);
            if (validationStart > validationEnd)
            {
                return (false, "The window start date must not be after the end date.", null);
            }

            report($"Preparing window {validationStart:yyyy-MM-dd} → {validationEnd:yyyy-MM-dd}", 2);

            var savedParameter = _repository.GetBestMLParameter();
            if (savedParameter is null)
            {
                return (false, "No saved AI parameters were found. Please train the AI first.", null);
            }

            var trainingStart = savedParameter.EarliestTrainingDate?.Date ?? _minimumRaceDate;
            if (trainingStart < _minimumRaceDate)
            {
                trainingStart = _minimumRaceDate;
            }

            report($"Loading race lists (training from {trainingStart:yyyy-MM-dd})", 5);
            var validationRaceIds = GetRaceIds(validationStart, validationEnd);
            if (validationRaceIds.Count == 0)
            {
                return (false,
                    $"No eligible races were found between {validationStart:yyyy-MM-dd} and {validationEnd:yyyy-MM-dd}. " +
                    $"The latest race in the database is {latestRaceDate:yyyy-MM-dd}.", null);
            }

            var trainingRaceIds = GetRaceIds(trainingStart, validationStart.AddDays(-1));
            if (trainingRaceIds.Count == 0)
            {
                return (false, "No eligible training data is available before the validation window.", null);
            }

            var validationSet = new HashSet<int>(validationRaceIds);
            var trainingSet = new HashSet<int>(trainingRaceIds);
            var parameter = BuildParameter(savedParameter);
            bool usedSavedModel = false;
            string source = windowMonths.HasValue
                ? $"Simulator window ({months} {(months == 1 ? "month" : "months")})"
                : $"Simulator window ({validationStart:yyyy-MM-dd} → {validationEnd:yyyy-MM-dd})";
            HyperparameterTrainer.TrainingResult result;
            IReadOnlyList<float> predictions;
            int trainingRaceCount = trainingRaceIds.Count;
            int validationRaceCount = validationRaceIds.Count;

            // The active model is frozen after train+dev, so its weights end
            // at TrainingEndDate. A window that starts after that date can be
            // scored directly, even if it overlaps the held-out test rows.
            // TestEndDate is the last race in the database and made direct
            // scoring impossible for the default window.
            var cutoff = savedParameter.TrainingEndDate?.Date ?? savedParameter.TestEndDate?.Date;
            bool canUseSavedModel = request.UseSavedModel &&
                                    cutoff.HasValue &&
                                    validationStart > cutoff.Value;
            bool savedModelPredicted = false;
            HyperparameterTrainer.TrainingResult? savedResult = null;
            IReadOnlyList<float>? savedPredictions = null;
            if (canUseSavedModel)
            {
                try
                {
                    report(
                        $"Scoring {validationRaceIds.Count:N0} validation races with the active model",
                        20);

                    // Encode only the validation races. The feature schema is
                    // derived from the training-only date range that produced
                    // the saved model's schema, so the hash check remains
                    // meaningful. The full training history is never
                    // materialised in memory.
                    var metadataStart = savedParameter.EarliestTrainingDate?.Date ?? trainingStart;
                    DateTime? metadataEnd = savedParameter.DevStartDate?.Date.AddDays(-1)
                        ?? savedParameter.TrainingEndDate?.Date
                        ?? cutoff!.Value;
                    var metadataRaceIds = metadataEnd < metadataStart
                        ? new List<int>()
                        : GetRaceIds(metadataStart, metadataEnd.Value);
                    var metadataSet = new HashSet<int>(metadataRaceIds);
                    if (metadataSet.Count == 0)
                    {
                        _logger.LogInformation(
                            "Simulator will retrain: the saved model's metadata date range was empty.");
                    }
                    else
                    {
                        var validationDataset = _trainer.LoadEncodedTrainingDataset(
                            includeRaceIds: validationSet,
                            includeIdentifiers: true,
                            requireMarketPrice: !_keepUnpricedRunners,
                            metadataRaceIds: metadataSet,
                            marketBlind: savedParameter.MarketBlind);
                        if (validationDataset.Races.Count > 0)
                        {
                            savedModelPredicted = TryPredictWithSavedModel(
                                validationDataset,
                                out var predictedResult,
                                out var predictedValues);
                            savedResult = predictedResult;
                            savedPredictions = predictedValues;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(
                        ex,
                        "Saved model inference failed; the simulator will retrain a replica.");
                }
            }

            if (savedModelPredicted && savedResult is not null && savedPredictions is not null)
            {
                usedSavedModel = true;
                result = savedResult;
                predictions = savedPredictions;
                trainingRaceCount = savedParameter.TrainingRaceCount ?? trainingRaceCount;
                source = windowMonths.HasValue
                    ? $"Saved model window ({months} {(months == 1 ? "month" : "months")})"
                    : $"Saved model window ({validationStart:yyyy-MM-dd} → {validationEnd:yyyy-MM-dd})";
                report(
                    $"Scored {result.ValidationExamples.Count:N0} runners with the saved model",
                    85);
            }
            else
            {
                if (request.UseSavedModel && !canUseSavedModel)
                {
                    var cutoffText = cutoff.HasValue
                        ? $" (the active model's training cutoff is {cutoff.Value:yyyy-MM-dd})"
                        : string.Empty;
                    report(
                        $"Window overlaps the model's training period{cutoffText}; retraining a replica",
                        20);
                }
                else if (request.UseSavedModel && canUseSavedModel && !savedModelPredicted)
                {
                    report("Saved model could not be applied to this feature set; retraining a replica", 20);
                }

                if (trainingRaceIds.Count > _maxTrainingRaces)
                {
                    return (false,
                        $"This window would train on {trainingRaceIds.Count:N0} races, above the safety limit of " +
                        $"{_maxTrainingRaces:N0}. Narrow the window, or raise " +
                        "Training:SimulatorMaxTrainingRaces in appsettings.", null);
                }

                report(
                    $"Building features for {trainingRaceIds.Count:N0} training races and " +
                    $"{validationRaceIds.Count:N0} validation races",
                    25);
                var dataset = _trainer.LoadTrainingDatasetSinglePass(
                    trainingSet,
                    validationSet,
                    includeIdentifiers: true,
                    requireMarketPrice: !_keepUnpricedRunners,
                    metadataRaceIds: trainingSet,
                    marketBlind: savedParameter.MarketBlind);
                if (dataset.TrainingRaces.Count == 0 || dataset.ValidationRaces.Count == 0)
                {
                    return (false, "The priced training or validation dataset was empty after preparation.", null);
                }

                if (dataset.RowCount > _maxRunners)
                {
                    var cutoffHint = cutoff.HasValue
                        ? $" Start the window after {cutoff.Value:yyyy-MM-dd} so the active model can score it directly, or raise"
                        : " Raise";
                    return (false,
                        $"This window needs {dataset.RowCount:N0} runner rows to retrain before " +
                        $"{validationStart:yyyy-MM-dd}, above the safety limit of {_maxRunners:N0}." +
                        $"{cutoffHint} Training:SimulatorMaxRunners in appsettings.", null);
                }

                trainingRaceCount = dataset.TrainingRaces.Count;
                validationRaceCount = dataset.ValidationRaces.Count;
                report($"Training on {dataset.TrainingRaces.Count:N0} races", 30);
                using var heartbeatCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                var heartbeat = RunTrainingHeartbeat(report, heartbeatCts.Token);
                try
                {
                    var temperatureOverride = savedParameter.DevTemperature.HasValue
                        ? (float)savedParameter.DevTemperature.Value
                        : 1.0f;
                    result = await Task.Run(() => _trainer.Train(
                        parameter,
                        0,
                        1,
                        dataset,
                        persistWeights: false,
                        temperatureOverride: temperatureOverride,
                        earlyStopOnValidation: false,
                        blendModelWeight: savedParameter.BlendModelWeight ?? 1d,
                        blendMarketWeight: savedParameter.BlendMarketWeight ?? 0d),
                        cancellationToken);
                }
                finally
                {
                    heartbeatCts.Cancel();
                    try
                    {
                        await heartbeat;
                    }
                    catch (OperationCanceledException)
                    {
                        // Expected when training finishes.
                    }
                }

                predictions = MarketBlendCalibration.ApplyFullField(
                    result.ValidationPredictions,
                    result.ValidationRaceIds,
                    result.ValidationExamples,
                    savedParameter.BlendModelWeight ?? 1d,
                    savedParameter.BlendMarketWeight ?? 0d);
            }

            cancellationToken.ThrowIfCancellationRequested();
            report(
                $"Building snapshot from {result.ValidationExamples.Count:N0} runner predictions",
                90);
            var raceSummaries = _repository.GetRaceSummaries(result.ValidationRaceIds);
            var snapshot = SimulatorSnapshotBuilder.Build(
                result,
                predictions,
                raceSummaries,
                savedParameter,
                validationStart,
                validationEnd,
                windowMonths,
                trainingRaceCount,
                validationRaceCount,
                source,
                usedSavedModel);

            var key = _store.Set(snapshot);
            report(
                $"Snapshot ready: {snapshot.Summary.ValidationRaceCount:N0} races, " +
                $"{snapshot.Summary.RunnerCount:N0} runners",
                100);
            _logger.LogInformation(
                "Simulator snapshot {Key} ready: {Races} races, {Runners} runners, saved model: {Saved}.",
                key,
                snapshot.Summary.ValidationRaceCount,
                snapshot.Summary.RunnerCount,
                usedSavedModel);
            return (true, null, key);
        }

        private MLParameter BuildParameter(MLParameter savedParameter)
        {
            return new MLParameter
            {
                Units = MLParameterValidator.EnsureUnits(savedParameter.Units, 0),
                Dropout = MLParameterValidator.EnsureDropout(savedParameter.Dropout, 0),
                Layers = MLParameterValidator.EnsureLayers(savedParameter.Layers, 0),
                LearningRate = MLParameterValidator.EnsureLearningRate(savedParameter.LearningRate, 0.005),
                Epochs = savedParameter.BestEpoch.HasValue
                    ? Math.Max(1, savedParameter.BestEpoch.Value + 1)
                    : MLParameterValidator.EnsurePositive(savedParameter.Epochs, 12),
                BatchSize = MLParameterValidator.EnsurePositive(savedParameter.BatchSize, 500),
                Objective = savedParameter.Objective ?? "Binary",
                MarketOffsetMode = savedParameter.MarketOffsetMode ?? "Logit",
                MarketBlind = savedParameter.MarketBlind ?? false,
                BlendModelWeight = savedParameter.BlendModelWeight ?? 1d,
                BlendMarketWeight = savedParameter.BlendMarketWeight ?? 0d,
                Seed = savedParameter.Seed ?? _seed,
                RaceBatchSize = savedParameter.RaceBatchSize ?? _raceBatchSize,
                WinRateAlpha = savedParameter.WinRateAlpha ?? _winRateAlpha,
                WinRateBeta = savedParameter.WinRateBeta ?? _winRateBeta,
                Folds = 1
            };
        }

        private bool TryPredictWithSavedModel(
            HyperparameterTrainer.TrainingDataset dataset,
            out HyperparameterTrainer.TrainingResult result,
            out IReadOnlyList<float> predictions)
        {
            result = new HyperparameterTrainer.TrainingResult();
            predictions = Array.Empty<float>();

            var path = ModelArtifactStore.ResolveActivePath(_modelDirectory);
            if (string.IsNullOrWhiteSpace(path))
            {
                return false;
            }

            var runtime = TrainedModelRuntime.TryLoad(path, out var status);
            if (runtime is null || !runtime.IsRaceSoftmax)
            {
                _logger.LogInformation("Simulator will retrain: {Status}", status);
                return false;
            }

            if (!runtime.MatchesFeatureSchema(dataset.FeatureKeys, dataset.FeatureDimensions))
            {
                _logger.LogInformation(
                    "Simulator will retrain: saved model feature schema does not match the current dataset.");
                return false;
            }

            var datasetMetadata = new FeatureMetadata
            {
                Keys = dataset.FeatureKeys,
                FeatureDimensions = dataset.FeatureDimensions,
                StringMaps = dataset.StringMaps
            };
            var datasetSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(datasetMetadata);
            var models = runtime.Model.Members is { Count: > 0 }
                ? runtime.Model.Members
                : new List<TrainedModel> { runtime.Model };
            foreach (var model in models)
            {
                if (!string.IsNullOrWhiteSpace(model.FeatureSchemaHash) &&
                    !string.Equals(
                        model.FeatureSchemaHash,
                        datasetSchemaHash,
                        StringComparison.OrdinalIgnoreCase))
                {
                    _logger.LogInformation(
                        "Simulator will retrain: saved model categorical feature maps do not match the current dataset.");
                    return false;
                }
            }

            var examples = new List<HyperparameterTrainer.RunnerExample>();
            var raceIds = new List<int>();
            var probabilities = new List<float>();
            // A validation-only dataset keeps its encoded races in
            // TrainingRaces because LoadEncodedTrainingDataset treats the
            // included ids as its training view. A full train/validation
            // dataset exposes the held-out window through ValidationRaces.
            var scoringRaces = dataset.ValidationRaces.Count > 0
                ? dataset.ValidationRaces
                : dataset.TrainingRaces;
            foreach (var race in scoringRaces)
            {
                if (!runtime.TryPredictEncodedRace(race.Runners, out var raceProbabilities) ||
                    raceProbabilities.Length != race.Runners.Count)
                {
                    _logger.LogInformation(
                        "Simulator will retrain: saved model could not score race {RaceId}.",
                        race.RaceId);
                    return false;
                }

                for (int i = 0; i < race.Runners.Count; i++)
                {
                    var example = race.Runners[i];
                    examples.Add(example);
                    raceIds.Add(race.RaceId);
                    probabilities.Add((float)raceProbabilities[i]);
                }
            }

            result = new HyperparameterTrainer.TrainingResult
            {
                Objective = runtime.Model.Objective,
                ValidationPredictions = probabilities,
                ValidationLabels = examples.Select(example => example.Label).ToList(),
                ValidationRaceIds = raceIds,
                ValidationExamples = examples,
                ModelArtifactFileName = System.IO.Path.GetFileName(path),
                FeatureSchemaHash = runtime.Model.FeatureSchemaHash
            };
            predictions = probabilities;
            return true;
        }

        private static async Task RunTrainingHeartbeat(
            Action<string, int?> report,
            CancellationToken cancellationToken)
        {
            var stopwatch = Stopwatch.StartNew();
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    await Task.Delay(TimeSpan.FromSeconds(15), cancellationToken);
                    var elapsed = stopwatch.Elapsed.TotalMinutes;
                    int progress = Math.Min(80, 25 + (int)Math.Floor(elapsed * 4d));
                    report($"Training in progress ({elapsed:0.0} min elapsed)", progress);
                }
            }
            catch (OperationCanceledException)
            {
                // Normal completion path.
            }
            finally
            {
                stopwatch.Stop();
            }
        }

        private List<int> GetRaceIds(DateTime startInclusive, DateTime endInclusive)
        {
            return _requirePreOffPrice
                ? _repository.GetRaceIdsWithOpeningPrice(startInclusive, endInclusive)
                : _repository.GetRaceIdsBetweenDates(startInclusive, endInclusive);
        }
    }
}
