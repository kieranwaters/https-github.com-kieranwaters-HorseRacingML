using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using HorseRacingML.Data;
using Microsoft.ML;
using HorseRacingML.Jobs;
using HorseRacingML.Models;
using System.Threading;
using System.Text.Json;
using HorseRacingML.ML;
using HorseRacingML.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace HorseRacingML.Controllers
{
    public class HyperparameterController : Controller
    {
        private readonly RacingRepository _repository;
        private readonly HyperparameterTrainer _trainer;
        private readonly BackgroundJobQueue _jobQueue;
        private readonly BackgroundJobStore _jobStore;
        private readonly ILogger<HyperparameterController> _logger;
        private readonly BettingSimulatorStore _simulatorStore;

        private readonly decimal? _maxKellyFraction;
        private readonly decimal _commissionRate;
        private readonly double _validationFraction;
        private readonly bool _requireMarketEdge;
        private readonly bool _useMarketBlend;
        private readonly int _maxParallelTrials;
        private readonly DateTime _minimumMarketRaceDate;
        private readonly double _devFraction;
        private readonly double _testFraction;
        private readonly string _objective;
        private readonly string _marketOffsetMode;
        private readonly bool _marketBlind;
        private readonly double _minimumBlendImprovement;
        private readonly int _raceBatchSize;
        private readonly int _seed;
        private readonly int _seedCount;
        private readonly int _ensembleSize;
        private readonly int _marketBlindEnsembleMembers;
        private readonly bool _lightGbmEnsembleMember;
        private readonly int _lightGbmIterations;
        private readonly double _winRateAlpha;
        private readonly double _winRateBeta;
        private readonly bool _keepUnpricedRunners;
        private readonly bool _requirePreOffPrice;

        public HyperparameterController(
            RacingRepository repository,
            HyperparameterTrainer trainer,
            IConfiguration configuration,
            BackgroundJobQueue jobQueue,
            BackgroundJobStore jobStore,
            ILogger<HyperparameterController> logger,
            BettingSimulatorStore simulatorStore)
        {
            _repository = repository;
            _trainer = trainer;
            _jobQueue = jobQueue;
            _jobStore = jobStore;
            _logger = logger;
            _simulatorStore = simulatorStore;
            _maxKellyFraction = configuration.GetValue<decimal?>("Betting:MaxKellyFraction");
            _commissionRate = Math.Clamp(configuration.GetValue<decimal?>("Betting:CommissionRate") ?? 0.05m, 0m, 1m);
            _validationFraction = Math.Clamp(configuration.GetValue<double?>("Training:ValidationFraction") ?? 0.10, 0.01, 0.50);
            _requireMarketEdge = configuration.GetValue<bool?>("Training:RequireMarketEdge") ?? true;
            // The post-model market blend has not been better than the pure
            // model in the recent ablation runs, so selection is model-only by
            // default. Set Training:UseMarketBlend=true to restore the old path.
            _useMarketBlend = configuration.GetValue<bool?>("Training:UseMarketBlend") ?? false;
            _maxParallelTrials = Math.Max(1, configuration.GetValue<int?>("Training:MaxParallelTrials") ?? 1);
            // Selects which races are returned to the trainer; it is deliberately
            // not passed as minimumRaceDate, because the feature state still needs
            // the full history to build each runner's rolling statistics.
            _minimumMarketRaceDate = configuration.GetValue<DateTime?>("Training:MinimumMarketRaceDate")
                ?? configuration.GetValue<DateTime?>("Training:MinimumRaceDate")
                ?? new DateTime(2017, 1, 1);
            _devFraction = Math.Clamp(configuration.GetValue<double?>("Training:DevFraction") ?? 0.10, 0.01, 0.40);
            _testFraction = Math.Clamp(configuration.GetValue<double?>("Training:TestFraction") ?? 0.10, 0.01, 0.40);
            _objective = configuration.GetValue<string>("Training:Objective") ?? "RaceSoftmax";
            _marketOffsetMode = HyperparameterTrainer.NormalizeOffsetMode(
                configuration.GetValue<string>("Training:MarketOffsetMode") ?? "LogProb");
            _marketBlind = configuration.GetValue<bool?>("Training:MarketBlind") ?? false;
            _minimumBlendImprovement = Math.Max(
                0d,
                configuration.GetValue<double?>("Training:MinimumBlendLogLossImprovement") ?? 0.0002d);
            _raceBatchSize = Math.Max(1, configuration.GetValue<int?>("Training:RaceBatchSize") ?? 64);
            _seed = configuration.GetValue<int?>("Training:Seed") ?? 42;
            _seedCount = Math.Max(1, configuration.GetValue<int?>("Training:SeedCount") ?? 1);
            _ensembleSize = Math.Clamp(
                configuration.GetValue<int?>("Training:EnsembleSize") ?? 3,
                1,
                10);
            _marketBlindEnsembleMembers = Math.Clamp(
                configuration.GetValue<int?>("Training:MarketBlindEnsembleMembers") ?? 1,
                0,
                5);
            _lightGbmEnsembleMember = configuration.GetValue<bool?>(
                "Training:LightGbmEnsembleMember") ?? true;
            _lightGbmIterations = Math.Clamp(
                configuration.GetValue<int?>("Training:LightGbmIterations") ?? 200,
                10,
                2000);
            _winRateAlpha = configuration.GetValue<double?>("WinRateAlpha") ?? 1.0;
            _winRateBeta = configuration.GetValue<double?>("WinRateBeta") ?? 9.0;
            // Unpriced runners still carry form/pace/draw information and are
            // part of the real softmax field. Keep them by default; the market
            // features are marked missing and the market offset is zero.
            _keepUnpricedRunners = configuration.GetValue<bool?>("Training:KeepUnpricedRunners") ?? true;
            _requirePreOffPrice = configuration.GetValue<bool?>("Training:RequirePreOffPrice") ?? true;
        }

        private List<int> GetTrainingRaceIds(DateTime startInclusive, DateTime endInclusive)
        {
            return _requirePreOffPrice
                ? _repository.GetRaceIdsWithOpeningPrice(startInclusive, endInclusive)
                : _repository.GetRaceIdsBetweenDates(startInclusive, endInclusive);
        }

        [HttpGet]
        public IActionResult Custom()
        {
            return View(new MLParameterBatch
            {
                Parameters = new List<MLParameter>
                {
                    new MLParameter { RunDate = DateTime.UtcNow }
                }
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult Custom(MLParameterBatch batch)
        {
            if (batch.ValidationStartDate.HasValue)
            {
                // A date split ignores walk-forward folds, so an empty Folds
                // cell must not block the whole run with a model-binding error.
                foreach (var key in ModelState.Keys
                             .Where(key => key.EndsWith(".Folds", StringComparison.OrdinalIgnoreCase))
                             .ToList())
                {
                    ModelState.Remove(key);
                }
            }

            if (!ModelState.IsValid)
            {
                return View(batch);
            }

            var payload = new Dictionary<string, string?>
            {
                ["batchJson"] = JsonSerializer.Serialize(batch)
            };

            if (!_jobQueue.TryEnqueue(BackgroundJobKind.CustomTraining, "Custom training", payload))
            {
                TempData["Message"] = "A training job is already running or queued.";
            }
            else
            {
                TempData["Message"] = "Custom training has been queued.";
            }

            return RedirectToAction("Index", "Home");
        }

        [NonAction]
        public async Task<IActionResult> RunCustomTraining(MLParameterBatch batch, CancellationToken cancellationToken = default)
        {
            if (!ModelState.IsValid)
            {
                return View(batch);
            }

            await Task.Run(() =>
            {
                var parameters = batch.Parameters ?? new List<MLParameter>();
                _logger.LogInformation(
                    "Custom training started for {ParameterSetCount} parameter set(s).",
                    parameters.Count);
                Console.WriteLine($"[Hyperparameter] Starting custom run for {parameters.Count} parameter set(s).");
                if (_maxParallelTrials > 1)
                {
                    Console.WriteLine(
                        $"[Hyperparameter] MaxParallelTrials={_maxParallelTrials}. If TensorFlow.NET reports graph/session errors, set Training:MaxParallelTrials back to 1.");
                }

                // Optional single-split mode: train on everything before the
                // requested date, validate once on races from that date onward.
                // This is the "only test the last two years" equivalent of
                // running just the final walk-forward fold.
                bool singleSplit = batch.ValidationStartDate.HasValue;
                DateTime validationStart = batch.ValidationStartDate?.Date ?? default;
                if (singleSplit &&
                    (validationStart <= _minimumMarketRaceDate.Date || validationStart >= DateTime.Today))
                {
                    Console.WriteLine(
                        $"[Hyperparameter] Validation start {validationStart:yyyy-MM-dd} must be after {_minimumMarketRaceDate:yyyy-MM-dd} and before today; nothing to train.");
                    return;
                }

                HyperparameterTrainer.TrainingDataset? dataset = null;
                if (!singleSplit)
                {
                    var pricedRaceIds = GetTrainingRaceIds(_minimumMarketRaceDate, DateTime.Today);
                    if (pricedRaceIds.Count == 0)
                    {
                        Console.WriteLine(
                            $"[Hyperparameter] No eligible races were found on or after {_minimumMarketRaceDate:yyyy-MM-dd}; nothing to train on.");
                        return;
                    }

                    // The full history is still replayed for the rolling feature
                    // state; only eligible race rows are returned.
                    dataset = _trainer.LoadEncodedTrainingDataset(
                        new HashSet<int>(pricedRaceIds),
                        requireMarketPrice: !_keepUnpricedRunners,
                        marketBlind: _marketBlind);
                    if (dataset.Races.Count == 0 || dataset.RowCount == 0)
                    {
                        Console.WriteLine("[Hyperparameter] No usable races remained after filtering; nothing to train on.");
                        return;
                    }

                    long encodedBytes = (long)dataset.RowCount * dataset.FeatureCount * sizeof(float);
                    Console.WriteLine(
                        $"[Hyperparameter] Dataset prepared with {dataset.Races.Count} eligible races (from {_minimumMarketRaceDate:yyyy-MM-dd}), {dataset.RowCount} runner rows, {dataset.FeatureCount} features (~{encodedBytes / (1024d * 1024d * 1024d):F1} GiB encoded).");
                }

                var runs = new List<ModelRun>();
                for (int i = 0; i < parameters.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var model = parameters[i];
                    if (!singleSplit)
                    {
                        if (model.Folds <= 0)
                        {
                            Console.WriteLine($"[Hyperparameter] Skipped model {i + 1} due to invalid fold count {model.Folds}.");
                            continue;
                        }

                        if (model.Folds < 2)
                        {
                            Console.WriteLine($"[Hyperparameter] Forcing at least 2 folds for model {i + 1}; a single fold would train and validate on the same races.");
                            model.Folds = 2;
                        }
                    }

                    model.RunDate = DateTime.UtcNow;
                    model.EarliestTrainingDate ??= _repository.GetEarliestRaceDateBetween(_minimumMarketRaceDate, DateTime.Today);
                    model.Objective ??= _objective;
                    model.RaceBatchSize ??= _raceBatchSize;
                    model.Seed ??= _seed;
                    model.WinRateAlpha ??= _winRateAlpha;
                    model.WinRateBeta ??= _winRateBeta;
                    runs.Add(new ModelRun(model, i + 1) { SingleSplit = singleSplit });
                }

                if (singleSplit)
                {
                    var trainingRaceList = GetTrainingRaceIds(_minimumMarketRaceDate, validationStart.AddDays(-1));
                    var validationRaceList = GetTrainingRaceIds(validationStart, DateTime.Today);
                    if (trainingRaceList.Count == 0 || validationRaceList.Count == 0)
                    {
                        Console.WriteLine(
                            $"[Hyperparameter] Date split {_minimumMarketRaceDate:yyyy-MM-dd}..{validationStart.AddDays(-1):yyyy-MM-dd} train / {validationStart:yyyy-MM-dd}..{DateTime.Today:yyyy-MM-dd} validation has an empty side; nothing to train.");
                        return;
                    }

                    var trainingRaceIds = new HashSet<int>(trainingRaceList);
                    var validationRaceIds = new HashSet<int>(validationRaceList);
                    var splitRaceIds = new HashSet<int>(trainingRaceIds);
                    splitRaceIds.UnionWith(validationRaceIds);

                    var splitSource = _trainer.LoadEncodedTrainingDataset(
                        splitRaceIds,
                        requireMarketPrice: !_keepUnpricedRunners,
                        marketBlind: _marketBlind);
                    var splitDataset = HyperparameterTrainer.SplitDataset(
                        splitSource,
                        trainingRaceIds,
                        validationRaceIds);
                    if (splitDataset.TrainingRaces.Count == 0 || splitDataset.ValidationRaces.Count == 0)
                    {
                        Console.WriteLine("[Hyperparameter] Date split was empty after encoding; nothing to train.");
                        return;
                    }

                    var trainingStart = _repository.GetRaceDate(trainingRaceList[0]);
                    var trainingEnd = _repository.GetRaceDate(trainingRaceList[^1]);
                    var validationEnd = _repository.GetRaceDate(validationRaceList[^1]);
                    Console.WriteLine(
                        $"[Hyperparameter] Single date split: {splitDataset.TrainingRaces.Count} training races before {validationStart:yyyy-MM-dd}, {splitDataset.ValidationRaces.Count} validation races from {validationStart:yyyy-MM-dd}.");

                    foreach (var run in runs)
                    {
                        run.TrainingStartDate = trainingStart;
                        run.TrainingEndDate = trainingEnd;
                        run.ValidationStartDate = validationStart;
                        run.ValidationEndDate = validationEnd;
                        run.TrainingRaceCount = splitDataset.TrainingRaces.Count;
                        run.ValidationRaceCount = splitDataset.ValidationRaces.Count;
                        run.Model.EarliestTrainingDate ??= trainingStart;
                    }

                    if (_maxParallelTrials <= 1)
                    {
                        foreach (var run in runs)
                        {
                            TrainModelFold(run, 0, 1, splitDataset, cancellationToken, singleSplit: true);
                        }
                    }
                    else
                    {
                        Parallel.ForEach(
                            runs,
                            new ParallelOptions
                            {
                                MaxDegreeOfParallelism = _maxParallelTrials,
                                CancellationToken = cancellationToken
                            },
                            run => TrainModelFold(run, 0, 1, splitDataset, cancellationToken, singleSplit: true));
                    }
                }
                else
                {
                    foreach (var group in runs.GroupBy(run => run.Model.Folds).OrderBy(g => g.Key))
                    {
                        for (int fold = 1; fold < group.Key; fold++)
                        {
                            cancellationToken.ThrowIfCancellationRequested();
                            var foldDataset = _trainer.BuildFoldTrainingDataset(dataset!, fold, group.Key);
                            var foldRuns = group.ToList();
                            if (_maxParallelTrials <= 1)
                            {
                                foreach (var run in foldRuns)
                                {
                                    TrainModelFold(run, fold, group.Key, foldDataset, cancellationToken);
                                }
                            }
                            else
                            {
                                Parallel.ForEach(
                                    foldRuns,
                                    new ParallelOptions
                                    {
                                        MaxDegreeOfParallelism = _maxParallelTrials,
                                        CancellationToken = cancellationToken
                                    },
                                    run => TrainModelFold(run, fold, group.Key, foldDataset, cancellationToken));
                            }
                        }
                    }
                }

                // Individual fold/model rows are inserted above as they finish.
                // Only the fold-average rows are inserted here.
                foreach (var run in runs.OrderBy(run => run.ModelIndex))
                {
                    if (run.SingleSplit)
                    {
                        continue;
                    }

                    run.AddAveragedRow();
                    foreach (var row in run.Rows)
                    {
                        _repository.InsertMLParameter(row);
                    }
                }

                _logger.LogInformation("Custom training finished.");
                Console.WriteLine("[Hyperparameter] Custom run finished.");
            });

            return RedirectToAction("Index", "Home");
        }

        private sealed class ModelRun
        {
            public ModelRun(MLParameter model, int modelIndex)
            {
                Model = model;
                ModelIndex = modelIndex;
            }

            public MLParameter Model { get; }
            public int ModelIndex { get; }
            public bool SingleSplit { get; set; }
            public DateTime? TrainingStartDate { get; set; }
            public DateTime? TrainingEndDate { get; set; }
            public DateTime? ValidationStartDate { get; set; }
            public DateTime? ValidationEndDate { get; set; }
            public int TrainingRaceCount { get; set; }
            public int ValidationRaceCount { get; set; }
            public List<MLParameter> Rows { get; } = new();
            public List<HyperparameterTrainer.TrainingResult> Results { get; } = new();

            public void AddAveragedRow()
            {
                if (Results.Count == 0)
                {
                    return;
                }

                var first = Results.FirstOrDefault();
                Rows.Add(new MLParameter
                {
                    RunDate = Model.RunDate,
                    EarliestTrainingDate = Model.EarliestTrainingDate,
                    Units = Model.Units,
                    Dropout = Model.Dropout,
                    Layers = Model.Layers,
                    LearningRate = Model.LearningRate,
                    Epochs = Model.Epochs,
                    BatchSize = Model.BatchSize,
                    Folds = Model.Folds,
                    Fold = null,
                    TrainAccuracy = Results.Average(result => result.TrainAccuracy),
                    TrainLoss = Results.Average(result => result.TrainLoss),
                    TrainBrier = Results.Average(result => result.TrainBrier),
                    ValidationAccuracy = Results.Average(result => result.ValidationAccuracy),
                    ValidationLoss = Results.Average(result => result.ValidationLoss),
                    ValidationBrier = Results.Average(result => result.ValidationBrier),
                    DevAccuracy = Results.Average(result => result.ValidationAccuracy),
                    DevLoss = Results.Average(result => result.ValidationLoss),
                    DevBrier = Results.Average(result => result.ValidationBrier),
                    DevTemperature = Results.Average(result => result.ValidationTemperature),
                    Objective = first?.Objective ?? Model.Objective,
                    NormalizationMode = first?.NormalizationMode,
                    MarketOffsetScale = Results.Average(result => result.MarketOffsetScale),
                    BestEpoch = first?.BestEpoch,
                    Seed = Model.Seed,
                    RaceBatchSize = Model.RaceBatchSize,
                    WinRateAlpha = Model.WinRateAlpha,
                    WinRateBeta = Model.WinRateBeta,
                    SelectedOn = "FoldAverage",
                    IsActive = false
                });
            }
        }

        private void TrainModelFold(
            ModelRun run,
            int foldIndex,
            int foldCount,
            HyperparameterTrainer.TrainingDataset foldDataset,
            CancellationToken cancellationToken = default,
            bool singleSplit = false)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var model = run.Model;
            Console.WriteLine(singleSplit
                ? $"[Hyperparameter]  Model {run.ModelIndex} single date split - training in progress..."
                : $"[Hyperparameter]  Model {run.ModelIndex} walk-forward fold {foldIndex + 1}/{foldCount} - training in progress...");
            var result = _trainer.Train(
                model,
                foldIndex,
                foldCount,
                foldDataset,
                persistWeights: false,
                cancellationToken: cancellationToken);
            Console.WriteLine(singleSplit
                ? $"[Hyperparameter]  Model {run.ModelIndex} single date split complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}."
                : $"[Hyperparameter]  Model {run.ModelIndex} fold {foldIndex + 1}/{foldCount} complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");

            run.Results.Add(result);
            var row = new MLParameter
            {
                RunDate = model.RunDate,
                EarliestTrainingDate = model.EarliestTrainingDate,
                Units = model.Units,
                Dropout = model.Dropout,
                Layers = model.Layers,
                LearningRate = model.LearningRate,
                Epochs = model.Epochs,
                BatchSize = model.BatchSize,
                Folds = singleSplit ? 1 : model.Folds,
                Fold = singleSplit ? (int?)null : foldIndex + 1,
                TrainAccuracy = result.TrainAccuracy,
                TrainLoss = result.TrainLoss,
                TrainBrier = result.TrainBrier,
                ValidationAccuracy = result.ValidationAccuracy,
                ValidationLoss = result.ValidationLoss,
                ValidationBrier = result.ValidationBrier,
                DevAccuracy = result.ValidationAccuracy,
                DevLoss = result.ValidationLoss,
                DevBrier = result.ValidationBrier,
                DevTemperature = result.ValidationTemperature,
                Objective = result.Objective,
                NormalizationMode = result.NormalizationMode,
                MarketOffsetScale = result.MarketOffsetScale,
                BestEpoch = result.BestEpoch,
                Seed = result.Seed,
                RaceBatchSize = model.RaceBatchSize,
                WinRateAlpha = model.WinRateAlpha,
                WinRateBeta = model.WinRateBeta,
                TrainingStartDate = run.TrainingStartDate,
                TrainingEndDate = run.TrainingEndDate,
                DevStartDate = run.ValidationStartDate,
                DevEndDate = run.ValidationEndDate,
                TrainingRaceCount = run.TrainingRaceCount,
                DevRaceCount = run.ValidationRaceCount,
                TrainingRowCount = result.TrainingPredictions.Count,
                DevRowCount = result.ValidationExamples.Count,
                SelectedOn = singleSplit ? "DateSplit" : "Fold",
                IsActive = false
            };

            // Persist as soon as this model/fold finishes so partial results
            // survive a long sweep or a later model failing.
            var rowId = _repository.InsertMLParameter(row);
            Console.WriteLine(singleSplit
                ? $"[Hyperparameter] Completed model {run.ModelIndex} single date split; saved MLParameters row {rowId}."
                : $"[Hyperparameter] Completed model {run.ModelIndex} fold {foldIndex + 1}/{foldCount}; saved MLParameters row {rowId}.");
        }

        [HttpGet]
        public IActionResult TrainAI()
        {
            return View(new MLParameterBatch
            {
                Parameters = new List<MLParameter>
                {
                    new MLParameter { RunDate = DateTime.UtcNow }
                }
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult TrainAI(MLParameterBatch batch)
        {
            if (!ModelState.IsValid)
            {
                return View(batch);
            }

            var payload = new Dictionary<string, string?>
            {
                ["batchJson"] = JsonSerializer.Serialize(batch)
            };

            if (!_jobQueue.TryEnqueue(BackgroundJobKind.TrainAI, "Train AI", payload))
            {
                TempData["Message"] = "A training job is already running or queued.";
            }
            else
            {
                TempData["Message"] = "Training has been queued.";
            }

            return RedirectToAction("Index", "Home");
        }

        [NonAction]
        public async Task<IActionResult> RunTrainAI(MLParameterBatch batch, CancellationToken cancellationToken = default)
        {
            if (!ModelState.IsValid)
            {
                return View(batch);
            }

            await Task.Run(() =>
            {
                var parameters = batch.Parameters ?? new List<MLParameter>();
                _logger.LogInformation(
                    "Train AI started for {ParameterSetCount} parameter set(s).",
                    parameters.Count);
                Console.WriteLine($"[TrainAI] Starting training run for {parameters.Count} parameter set(s).");

                var allRaceIds = GetTrainingRaceIds(_minimumMarketRaceDate, DateTime.Today);
                if (allRaceIds.Count < 5)
                {
                    Console.WriteLine(
                        $"[TrainAI] Not enough eligible races on or after {_minimumMarketRaceDate:yyyy-MM-dd} to create train/dev/test splits.");
                    return;
                }

                int testCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * _testFraction));
                int devCount = Math.Max(1, (int)Math.Round(allRaceIds.Count * _devFraction));
                if (testCount + devCount >= allRaceIds.Count)
                {
                    devCount = Math.Max(1, (allRaceIds.Count - testCount) / 2);
                }
                if (testCount + devCount >= allRaceIds.Count)
                {
                    Console.WriteLine("[TrainAI] Unable to create non-empty train/dev/test splits.");
                    return;
                }

                int testStartIndex = allRaceIds.Count - testCount;
                int devStartIndex = testStartIndex - devCount;
                var testRaceIds = new HashSet<int>(allRaceIds.Skip(testStartIndex));
                var devRaceIds = new HashSet<int>(allRaceIds.Skip(devStartIndex).Take(devCount));
                var trainingRaceIds = new HashSet<int>(allRaceIds.Take(devStartIndex));
                var trainDevRaceIds = new HashSet<int>(trainingRaceIds);
                trainDevRaceIds.UnionWith(devRaceIds);
                var allIncluded = new HashSet<int>(allRaceIds);

                Console.WriteLine(
                    $"[TrainAI] Chronological split: {trainingRaceIds.Count} train races, {devRaceIds.Count} dev races, {testRaceIds.Count} test races.");

                // Build encoded features once. The feature schema is derived from
                // the training races only, so dev/test category values cannot
                // influence the model's columns.
                var full = _trainer.LoadEncodedTrainingDataset(
                    allIncluded,
                    includeIdentifiers: true,
                    requireMarketPrice: !_keepUnpricedRunners,
                    metadataRaceIds: trainingRaceIds,
                    marketBlind: _marketBlind);
                var devView = HyperparameterTrainer.SplitDataset(full, trainingRaceIds, devRaceIds);
                var testView = HyperparameterTrainer.SplitDataset(full, trainDevRaceIds, testRaceIds);

                int trainRows = devView.TrainingRaces.Sum(r => r.Runners.Count);
                int devRows = devView.ValidationRaces.Sum(r => r.Runners.Count);
                int testRows = testView.ValidationRaces.Sum(r => r.Runners.Count);
                Console.WriteLine(
                    $"[TrainAI] Market window starts {_minimumMarketRaceDate:yyyy-MM-dd}; rows: train={trainRows}, dev={devRows}, test={testRows} ({(_keepUnpricedRunners ? "all runners included, unpriced marked missing" : "unpriced runners excluded")}).");

                if (trainRows == 0 || devRows == 0 || testRows == 0)
                {
                    Console.WriteLine("[TrainAI] A train/dev/test split was empty after filtering; existing weights were left unchanged.");
                    return;
                }

                DateTime? trainStart = allRaceIds.Count > 0 ? _repository.GetRaceDate(allRaceIds[0]) : null;
                DateTime? trainEnd = devStartIndex > 0 ? _repository.GetRaceDate(allRaceIds[devStartIndex - 1]) : null;
                DateTime? devStart = devCount > 0 ? _repository.GetRaceDate(allRaceIds[devStartIndex]) : null;
                DateTime? devEnd = devCount > 0 ? _repository.GetRaceDate(allRaceIds[testStartIndex - 1]) : null;
                DateTime? testStart = testCount > 0 ? _repository.GetRaceDate(allRaceIds[testStartIndex]) : null;
                DateTime? testEnd = testCount > 0 ? _repository.GetRaceDate(allRaceIds[^1]) : null;

                int modelIndex = 0;
                int totalRuns = parameters.Count * _seedCount;
                MLParameter? bestModel = null;
                HyperparameterTrainer.TrainingResult? bestResult = null;
                MarketBlendParameters? bestBlend = null;
                MarketBlendParameters? bestFairMarket = null;
                var candidates = new List<ModelCandidate>();
                double bestSelectionScore = double.MaxValue;
                double bestDevMarketLogLoss = 0;
                double bestDevMarketBrier = double.MaxValue;

                foreach (var parameter in parameters)
                {
                    for (int seedIndex = 0; seedIndex < _seedCount; seedIndex++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        modelIndex++;
                        var model = CloneParameter(parameter, _seed + seedIndex);
                        model.RunDate = DateTime.UtcNow;
                        model.EarliestTrainingDate ??= trainStart;
                        model.Folds = 1;
                        model.Fold = null;
                        model.Objective ??= _objective;
                        model.MarketOffsetMode ??= _marketOffsetMode;
                        model.MarketBlind ??= _marketBlind;
                        model.RaceBatchSize ??= _raceBatchSize;
                        model.WinRateAlpha ??= _winRateAlpha;
                        model.WinRateBeta ??= _winRateBeta;
                        model.NormalizationMode ??= string.Equals(model.Objective, "Binary", StringComparison.OrdinalIgnoreCase)
                            ? "PerRaceNormalized"
                            : "PerRaceSoftmax";
                        model.FeatureBlocks ??= FeatureBlockNames.Format(_trainer.EnabledFeatureBlocks);

                        Console.WriteLine(
                            $"[TrainAI] Starting model {modelIndex}/{totalRuns} (seed {model.Seed}, {model.Objective}, offset: {model.MarketOffsetMode}, blind: {model.MarketBlind}, layers: {model.Layers}, units: {model.Units}, dropout: {model.Dropout}, lr: {model.LearningRate}).");

                        var result = _trainer.Train(
                            model,
                            0,
                            1,
                            devView,
                            persistWeights: false,
                            earlyStopOnValidation: true,
                            cancellationToken: cancellationToken);
                        var devMarket = ComputeMarketMetrics(result.ValidationExamples);
                        double devMarketLogLoss = devMarket.RunnerLogLoss;
                        double devMarketBrier = devMarket.Brier;
                        var devModelMetrics = ComputePricedRaceMetrics(result);
                        bool isRaceSoftmax = string.Equals(result.Objective, "RaceSoftmax", StringComparison.OrdinalIgnoreCase);

                        // The fair baseline uses the same no-vig probabilities the
                        // model sees, and the blend gate asks whether adding the
                        // model improves that baseline. Closing SP remains a
                        // separate, harder benchmark.
                        MarketBlendMetrics? devFairMarketRaw = null;
                        MarketBlendParameters? devFairMarket = null;
                        MarketBlendParameters? devBlend = null;
                        if (isRaceSoftmax && result.ValidationExamples.Count > 0)
                        {
                            // No fitting here: the raw no-vig market is the
                            // honest comparator for the model. The fitted
                            // market-only blend is kept as a diagnostic only.
                            devFairMarketRaw = MarketBlendCalibration.Evaluate(
                                result.ValidationPredictions,
                                result.ValidationLabels,
                                result.ValidationRaceIds,
                                result.ValidationExamples,
                                0d,
                                1d);
                            if (_useMarketBlend)
                            {
                                devFairMarket = MarketBlendCalibration.FitMarketOnly(
                                    result.ValidationPredictions,
                                    result.ValidationLabels,
                                    result.ValidationRaceIds,
                                    result.ValidationExamples);
                                devBlend = MarketBlendCalibration.Fit(
                                    result.ValidationPredictions,
                                    result.ValidationLabels,
                                    result.ValidationRaceIds,
                                    result.ValidationExamples);
                            }
                        }

                        model.TrainAccuracy = result.TrainAccuracy;
                        model.TrainLoss = result.TrainLoss;
                        model.TrainBrier = result.TrainBrier;
                        model.ValidationAccuracy = result.ValidationAccuracy;
                        model.ValidationLoss = result.ValidationLoss;
                        model.ValidationBrier = result.ValidationBrier;
                        model.DevAccuracy = result.ValidationAccuracy;
                        model.DevLoss = result.ValidationLoss;
                        model.DevBrier = result.ValidationBrier;
                        model.DevRaceLogLoss = devModelMetrics.RaceLogLoss;
                        model.DevTop3Accuracy = devModelMetrics.Top3Accuracy;
                        model.DevTemperature = result.ValidationTemperature;
                        model.DevMarketLogLoss = devMarketLogLoss;
                        model.DevMarketBrier = devMarketBrier;
                        model.DevMarketRaceLogLoss = devMarket.RaceLogLoss;
                        model.DevMarketTop1Accuracy = devMarket.Top1Accuracy;
                        model.DevMarketTop3Accuracy = devMarket.Top3Accuracy;
                        model.DevFairMarketLogLoss = devFairMarketRaw?.LogLoss;
                        model.DevFairMarketBrier = devFairMarketRaw?.Brier;
                        model.DevBlendLogLoss = devBlend?.Metrics.LogLoss;
                        model.DevBlendBrier = devBlend?.Metrics.Brier;
                        model.DevBlendRaceLogLoss = devBlend?.Metrics.RaceLogLoss;
                        model.BlendModelWeight = _useMarketBlend ? devBlend?.ModelWeight : 1.0;
                        model.BlendMarketWeight = _useMarketBlend ? devBlend?.MarketWeight : 0.0;
                        model.Objective = result.Objective;
                        model.NormalizationMode = result.NormalizationMode;
                        model.MarketOffsetScale = result.MarketOffsetScale;
                        model.MarketOffsetMode = result.MarketOffsetMode;
                        model.MarketBlind = result.MarketBlind;
                        model.BestEpoch = result.BestEpoch;
                        model.Seed = result.Seed;
                        model.TrainingStartDate = trainStart;
                        model.TrainingEndDate = trainEnd;
                        model.DevStartDate = devStart;
                        model.DevEndDate = devEnd;
                        model.TrainingRaceCount = trainingRaceIds.Count;
                        model.DevRaceCount = devRaceIds.Count;
                        model.TrainingRowCount = trainRows;
                        model.DevRowCount = devRows;

                        bool passedMarketGate;
                        double selectionScore;
                        if (isRaceSoftmax)
                        {
                            if (_useMarketBlend)
                            {
                                passedMarketGate = !_requireMarketEdge ||
                                    (devBlend is not null &&
                                     devFairMarket is not null &&
                                     devBlend.HasData &&
                                     devFairMarket.HasData &&
                                     devBlend.ModelWeight > 0.01d &&
                                     devBlend.Metrics.RaceLogLoss <
                                     devFairMarket.Metrics.RaceLogLoss - _minimumBlendImprovement);
                                selectionScore = devBlend?.Metrics.RaceLogLoss ?? double.MaxValue;
                            }
                            else
                            {
                                // Model-only selection, but still require an
                                // edge over the raw no-vig market. This is the
                                // prior the model actually sees, so it is the
                                // fairest gate. Closing SP remains a separate
                                // (harder) benchmark in the report.
                                passedMarketGate = !_requireMarketEdge ||
                                    (devFairMarketRaw is not null &&
                                     devModelMetrics.RaceLogLoss <
                                     devFairMarketRaw.RaceLogLoss - _minimumBlendImprovement);
                                selectionScore = devModelMetrics.RaceLogLoss;
                            }
                        }
                        else
                        {
                            passedMarketGate = !_requireMarketEdge || result.ValidationBrier < devMarketBrier;
                            selectionScore = result.ValidationBrier;
                        }

                        model.SelectedOn = passedMarketGate ? "Dev" : "MarketRejected";

                        _repository.InsertMLParameter(model);

                        var devPriced = HyperparameterTrainer.ComputePricedSubsetMetrics(result);
                        var blendText = devBlend is not null && devFairMarket is not null
                            ? $"blend race LL {devBlend.Metrics.RaceLogLoss:F4} (a={devBlend.ModelWeight:F2}, b={devBlend.MarketWeight:F2}) vs fair market {devFairMarket.Metrics.RaceLogLoss:F4}"
                            : "blend not fitted";
                        Console.WriteLine(
                            $"[TrainAI] Model {modelIndex}: dev race LL {devModelMetrics.RaceLogLoss:F4} vs fair market {devFairMarketRaw?.RaceLogLoss:F4}, SP market {devMarket.RaceLogLoss:F4}; dev top1 {devModelMetrics.Top1Accuracy:P2} vs fair {devFairMarketRaw?.Top1Accuracy:P2} and SP {devMarket.Top1Accuracy:P2}; dev top3 {devModelMetrics.Top3Accuracy:P2} vs SP {devMarket.Top3Accuracy:P2}; dev Brier {result.ValidationBrier:F4}; temperature {result.ValidationTemperature:F2}, best epoch {result.BestEpoch}. {blendText}. Priced subset: acc {devPriced.Accuracy:P2}, log loss {devPriced.LogLoss:F4}, Brier {devPriced.Brier:F4} over {devPriced.RaceCount} races.");

                        if (result.ValidationExamples.Count == 0)
                        {
                            Console.WriteLine("[TrainAI] No dev examples were produced; refusing to select this model.");
                            continue;
                        }

                        if (!passedMarketGate)
                        {
                            Console.WriteLine(isRaceSoftmax
                                ? "[TrainAI] Dev race log loss did not beat the raw no-vig market baseline; model is recorded as MarketRejected and is not eligible for activation."
                                : "[TrainAI] Dev Brier did not beat the dev closing market baseline; model is recorded as MarketRejected and is not eligible for activation.");
                            continue;
                        }

                        candidates.Add(new ModelCandidate
                        {
                            Model = CloneParameter(model, model.Seed ?? _seed),
                            Result = result,
                            SelectionScore = selectionScore
                        });

                        if (selectionScore < bestSelectionScore)
                        {
                            bestModel = CloneParameter(model, model.Seed ?? _seed);
                            bestResult = result;
                            bestBlend = devBlend;
                            bestFairMarket = devFairMarketRaw is null
                                ? null
                                : new MarketBlendParameters
                                {
                                    ModelWeight = 0d,
                                    MarketWeight = 1d,
                                    Metrics = devFairMarketRaw
                                };
                            bestSelectionScore = selectionScore;
                            bestDevMarketLogLoss = devMarketLogLoss;
                            bestDevMarketBrier = devMarketBrier;
                            Console.WriteLine(isRaceSoftmax
                                ? $"[TrainAI] Model {modelIndex} is the best dev model so far (race LL {bestSelectionScore:F4} vs fair {devFairMarketRaw?.RaceLogLoss:F4})."
                                : $"[TrainAI] Model {modelIndex} is the best dev model so far (Brier {bestSelectionScore:F4} vs market {bestDevMarketBrier:F4}).");
                        }
                    }
                }

                if (bestModel is null || bestResult is null)
                {
                    Console.WriteLine("[TrainAI] No model passed the dev gates; existing weights were left unchanged.");
                }
                else
                {
                    // Freeze the epoch count selected on dev, then evaluate once
                    // on the untouched test window. No early stopping or
                    // temperature fitting is performed on test.
                    int chosenEpochs = bestResult.BestEpoch.HasValue
                        ? bestResult.BestEpoch.Value + 1
                        : bestModel.Epochs;
                    bestModel.Epochs = Math.Max(1, chosenEpochs);

                    HyperparameterTrainer.TrainingResult TrainSingleFinal()
                    {
                        return _trainer.Train(
                            bestModel,
                            0,
                            1,
                            testView,
                            persistWeights: true,
                            temperatureOverride: (float)bestResult.ValidationTemperature,
                            earlyStopOnValidation: false,
                            blendModelWeight: bestBlend?.ModelWeight ?? 1d,
                            blendMarketWeight: bestBlend?.MarketWeight ?? 0d,
                            activateModel: true,
                            cancellationToken: cancellationToken);
                    }

                    HyperparameterTrainer.TrainingResult testResult;
                    int ensembleMemberCount = 1;
                    ModelArtifact? ensembleArtifact = null;
                    RaceEvaluationMetrics? ensembleDevMetrics = null;
                    bool ensembleEligible =
                        _ensembleSize > 1 &&
                        candidates.Count > 0 &&
                        string.Equals(bestResult.Objective, "RaceSoftmax", StringComparison.OrdinalIgnoreCase) &&
                        (candidates.Count > 1 ||
                         _marketBlindEnsembleMembers > 0 ||
                         _lightGbmEnsembleMember);

                    if (!ensembleEligible)
                    {
                        testResult = TrainSingleFinal();
                    }
                    else
                    {
                        var selected = candidates
                            .OrderBy(candidate => candidate.SelectionScore)
                            .Take(_ensembleSize)
                            .ToList();
                        var memberResults = new List<HyperparameterTrainer.TrainingResult>();
                        var memberModels = new List<TrainedModel>();
                        var devMemberPredictions = selected
                            .Select(candidate => candidate.Result.ValidationPredictions)
                            .ToList();
                        foreach (var candidate in selected)
                        {
                            cancellationToken.ThrowIfCancellationRequested();
                            var memberModel = CloneParameter(candidate.Model, candidate.Model.Seed ?? _seed);
                            memberModel.Epochs = candidate.Result.BestEpoch.HasValue
                                ? Math.Max(1, candidate.Result.BestEpoch.Value + 1)
                                : Math.Max(1, candidate.Model.Epochs);
                            var memberResult = _trainer.Train(
                                memberModel,
                                0,
                                1,
                                testView,
                                persistWeights: false,
                                temperatureOverride: (float)candidate.Result.ValidationTemperature,
                                earlyStopOnValidation: false,
                                blendModelWeight: candidate.Model.BlendModelWeight ?? 1d,
                                blendMarketWeight: candidate.Model.BlendMarketWeight ?? 0d,
                                activateModel: false,
                                cancellationToken: cancellationToken);
                            if (memberResult.Model is null)
                            {
                                continue;
                            }

                            memberResults.Add(memberResult);
                            memberModels.Add(memberResult.Model);
                        }

                        // Optional market-blind members. They share the encoded
                        // dataset but the trainer zeros every market-derived
                        // column and disables the market offset, so the member
                        // can only learn form/pace/context signal. Adding it to
                        // the ensemble diversifies away some market-prior noise.
                        for (int blindIndex = 0;
                             blindIndex < _marketBlindEnsembleMembers;
                             blindIndex++)
                        {
                            cancellationToken.ThrowIfCancellationRequested();
                            var blindModel = CloneParameter(
                                bestModel,
                                (_seed + blindIndex) % int.MaxValue);
                            blindModel.MarketBlind = true;
                            blindModel.Objective = "RaceSoftmax";
                            blindModel.FeatureBlocks = bestModel.FeatureBlocks;
                            blindModel.Epochs = bestResult.BestEpoch.HasValue
                                ? Math.Max(1, bestResult.BestEpoch.Value + 1)
                                : Math.Max(1, bestModel.Epochs);

                            var blindDev = _trainer.Train(
                                blindModel,
                                0,
                                1,
                                devView,
                                persistWeights: false,
                                earlyStopOnValidation: true,
                                cancellationToken: cancellationToken);
                            if (blindDev.BestEpoch.HasValue)
                            {
                                blindModel.Epochs = Math.Max(1, blindDev.BestEpoch.Value + 1);
                            }

                            // A blind member should add diversity, not drag the
                            // ensemble down. Keep it only when its dev race log
                            // loss is within a small tolerance of the best
                            // market-aware member.
                            if (blindDev.ValidationRaceLogLoss >
                                bestResult.ValidationRaceLogLoss + 0.05)
                            {
                                Console.WriteLine(
                                    $"[TrainAI] Skipping market-blind member {blindIndex + 1}: dev race LL {blindDev.ValidationRaceLogLoss:F4} is too far behind {bestResult.ValidationRaceLogLoss:F4}.");
                                continue;
                            }

                            devMemberPredictions.Add(blindDev.ValidationPredictions);
                            var blindTest = _trainer.Train(
                                blindModel,
                                0,
                                1,
                                testView,
                                persistWeights: false,
                                temperatureOverride: (float)blindDev.ValidationTemperature,
                                earlyStopOnValidation: false,
                                blendModelWeight: 1d,
                                blendMarketWeight: 0d,
                                activateModel: false,
                                cancellationToken: cancellationToken);
                            if (blindTest.Model is null)
                            {
                                continue;
                            }

                            Console.WriteLine(
                                $"[TrainAI] Added market-blind member {blindIndex + 1}: dev race LL {blindDev.ValidationRaceLogLoss:F4}, test race LL {RaceMetrics.Compute(blindTest.ValidationRaceIds, blindTest.ValidationPredictions, blindTest.ValidationLabels).RaceLogLoss:F4}.");
                            memberResults.Add(blindTest);
                            memberModels.Add(blindTest.Model);
                        }

                        // Optional LightGBM LambdaRank member. It is trained on
                        // exactly the same encoded feature matrix and only added
                        // when its dev race log loss is not far behind the MLP.
                        if (_lightGbmEnsembleMember && testView.Races.Count > 0)
                        {
                            try
                            {
                                var mlContext = new MLContext(seed: _seed);
                                var devLightGbmModel = LightGbmRankingModel.Train(
                                    mlContext,
                                    devView.TrainingRaces,
                                    devView.FeatureCount,
                                    _lightGbmIterations);
                                var devLightGbmPredictions = LightGbmRankingModel.PredictProbabilities(
                                    mlContext,
                                    devLightGbmModel,
                                    devView.ValidationRaces);
                                var referenceResult = selected[0].Result;
                                var devLightGbmMetrics = RaceMetrics.Compute(
                                    referenceResult.ValidationRaceIds,
                                    devLightGbmPredictions,
                                    referenceResult.ValidationLabels);

                                if (devLightGbmMetrics.RaceLogLoss >
                                    bestResult.ValidationRaceLogLoss + 0.10)
                                {
                                    Console.WriteLine(
                                        $"[TrainAI] Skipping LightGBM member: dev race LL {devLightGbmMetrics.RaceLogLoss:F4} is too far behind {bestResult.ValidationRaceLogLoss:F4}.");
                                }
                                else
                                {
                                    var testLightGbmModel = LightGbmRankingModel.Train(
                                        mlContext,
                                        testView.TrainingRaces,
                                        testView.FeatureCount,
                                        _lightGbmIterations);
                                    var testLightGbmPredictions = LightGbmRankingModel.PredictProbabilities(
                                        mlContext,
                                        testLightGbmModel,
                                        testView.ValidationRaces);

                                    Directory.CreateDirectory(_trainer.ModelDirectory);
                                    string lightGbmFileName =
                                        $"lightgbm-{DateTime.UtcNow:yyyyMMddHHmmssfff}-{Guid.NewGuid():N}.zip";
                                    string lightGbmPath = Path.Combine(
                                        _trainer.ModelDirectory,
                                        lightGbmFileName);
                                    LightGbmRankingModel.Save(
                                        mlContext,
                                        testLightGbmModel,
                                        testView.TrainingRaces,
                                        testView.FeatureCount,
                                        lightGbmPath);

                                    var lightGbmMember = new TrainedModel
                                    {
                                        HiddenLayers = new List<LayerWeights>(),
                                        OutputLayer = new LayerWeights(),
                                        Temperature = 1f,
                                        Objective = "RaceSoftmax",
                                        MarketOffsetScale = 0f,
                                        MarketOffsetMode = "LogProb",
                                        MarketBlind = false,
                                        BlendModelWeight = 1f,
                                        BlendMarketWeight = 0f,
                                        NormalizationMode = "PerRaceSoftmax",
                                        Metadata = new FeatureMetadata
                                        {
                                            Keys = new List<string>(testView.FeatureKeys),
                                            FeatureDimensions = new Dictionary<string, int>(
                                                testView.FeatureDimensions,
                                                StringComparer.OrdinalIgnoreCase),
                                            StringMaps = testView.StringMaps.ToDictionary(
                                                pair => pair.Key,
                                                pair => new Dictionary<string, int>(pair.Value),
                                                StringComparer.OrdinalIgnoreCase)
                                        },
                                        Normalization = new NormalizationParameters
                                        {
                                            Mean = new float[testView.FeatureCount],
                                            StdDev = Enumerable.Repeat(1f, testView.FeatureCount).ToArray()
                                        },
                                        LightGbmModelFile = lightGbmFileName,
                                        TrainedUtc = DateTime.UtcNow
                                    };
                                    lightGbmMember.FeatureSchemaHash =
                                        ModelArtifactStore.ComputeFeatureSchemaHash(lightGbmMember.Metadata);

                                    var testRunners = testView.ValidationRaces
                                        .SelectMany(race => race.Runners)
                                        .ToList();
                                    var lightGbmResult = new HyperparameterTrainer.TrainingResult
                                    {
                                        TrainAccuracy = 0,
                                        TrainLoss = 0,
                                        TrainBrier = 0,
                                        ValidationAccuracy = 0,
                                        ValidationLoss = 0,
                                        ValidationBrier = 0,
                                        ValidationTemperature = 1.0,
                                        Objective = "RaceSoftmax",
                                        MarketOffsetScale = 0f,
                                        MarketOffsetMode = "LogProb",
                                        MarketBlind = false,
                                        Seed = _seed,
                                        NormalizationMode = "PerRaceSoftmax",
                                        ValidationPredictions = testLightGbmPredictions,
                                        ValidationLabels = testRunners.Select(runner => runner.Label).ToArray(),
                                        ValidationRaceIds = testRunners.Select(runner => runner.RaceId).ToArray(),
                                        ValidationExamples = testRunners,
                                        Model = lightGbmMember
                                    };

                                    devMemberPredictions.Add(devLightGbmPredictions);
                                    memberResults.Add(lightGbmResult);
                                    memberModels.Add(lightGbmMember);
                                    Console.WriteLine(
                                        $"[TrainAI] Added LightGBM member: dev race LL {devLightGbmMetrics.RaceLogLoss:F4}, file {lightGbmFileName}.");
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine(
                                    $"[TrainAI] LightGBM member failed and will be skipped: {ex.Message}");
                            }
                        }

                        if (memberModels.Count < 2)
                        {
                            Console.WriteLine("[TrainAI] Not enough members for a seed ensemble; persisting a single model instead.");
                            testResult = TrainSingleFinal();
                        }
                        else
                        {
                            var first = memberResults[0];
                            var memberPredictions = memberResults
                                .Select(result => result.ValidationPredictions)
                                .ToList();
                            var ensemblePredictions = RaceMetrics.AverageLogProbabilities(
                                memberPredictions,
                                first.ValidationRaceIds);
                            var ensembleMetrics = RaceMetrics.Compute(
                                first.ValidationRaceIds,
                                ensemblePredictions,
                                first.ValidationLabels);
                            // Dev predictions come from the selected dev
                            // models, so their labels/examples must come from
                            // the dev result, not from the test result that was
                            // just retrained.
                            var devReference = selected[0].Result;
                            ensembleDevMetrics = ComputePricedRaceMetrics(
                                RaceMetrics.AverageLogProbabilities(
                                    devMemberPredictions,
                                    devReference.ValidationRaceIds),
                                devReference.ValidationLabels,
                                devReference.ValidationRaceIds,
                                devReference.ValidationExamples);

                            // The top-level object is a container only. It must
                            // not be one of the members: a member list that
                            // contains the top-level object would create a
                            // serialization cycle.
                            var ensemble = new TrainedModel
                            {
                                HiddenLayers = new List<LayerWeights>(),
                                OutputLayer = new LayerWeights(),
                                Temperature = 1f,
                                Objective = "RaceSoftmax",
                                MarketOffsetScale = memberModels[0].MarketOffsetScale,
                                MarketOffsetMode = memberModels[0].MarketOffsetMode,
                                MarketBlind = false,
                                BlendModelWeight = 1f,
                                BlendMarketWeight = 0f,
                                NormalizationMode = memberModels[0].NormalizationMode,
                                Metadata = memberModels[0].Metadata,
                                Normalization = memberModels[0].Normalization,
                                Members = memberModels,
                                TrainedUtc = DateTime.UtcNow
                            };
                            ensemble.FeatureSchemaHash = ModelArtifactStore.ComputeFeatureSchemaHash(ensemble.Metadata);
                            ensembleArtifact = ModelArtifactStore.Save(
                                ensemble,
                                _trainer.ModelDirectory,
                                activate: true);
                            ensembleMemberCount = memberModels.Count;

                            testResult = new HyperparameterTrainer.TrainingResult
                            {
                                TrainAccuracy = first.TrainAccuracy,
                                TrainLoss = first.TrainLoss,
                                TrainBrier = first.TrainBrier,
                                TrainRaceLogLoss = first.TrainRaceLogLoss,
                                TrainTop3Accuracy = first.TrainTop3Accuracy,
                                ValidationAccuracy = ensembleMetrics.Top1Accuracy,
                                ValidationLoss = ensembleMetrics.RunnerLogLoss,
                                ValidationBrier = ensembleMetrics.Brier,
                                ValidationRaceLogLoss = ensembleMetrics.RaceLogLoss,
                                ValidationTop3Accuracy = ensembleMetrics.Top3Accuracy,
                                ValidationTemperature = 1.0,
                                Objective = first.Objective,
                                AuxiliaryWinLossWeight = first.AuxiliaryWinLossWeight,
                                AuxiliaryFinishLossWeight = first.AuxiliaryFinishLossWeight,
                                BestEpoch = first.BestEpoch,
                                MarketOffsetScale = first.MarketOffsetScale,
                                MarketOffsetMode = first.MarketOffsetMode,
                                MarketBlind = first.MarketBlind,
                                Seed = first.Seed,
                                NormalizationMode = first.NormalizationMode,
                                TrainingPredictions = first.TrainingPredictions,
                                TrainingLabels = first.TrainingLabels,
                                TrainingRaceIds = first.TrainingRaceIds,
                                ValidationPredictions = ensemblePredictions,
                                ValidationLabels = first.ValidationLabels,
                                ValidationRaceIds = first.ValidationRaceIds,
                                ValidationExamples = first.ValidationExamples,
                                ModelArtifactFileName = ensembleArtifact.FileName,
                                ModelHash = ensembleArtifact.Sha256,
                                FeatureSchemaHash = ensemble.FeatureSchemaHash
                            };

                            Console.WriteLine(
                                $"[TrainAI] Persisted a {ensembleMemberCount}-member seed ensemble (artifact {ensembleArtifact.FileName}).");
                        }
                    }

                    var testMarket = ComputeMarketMetrics(testResult.ValidationExamples);
                    double testMarketLogLoss = testMarket.RunnerLogLoss;
                    double testMarketBrier = testMarket.Brier;
                    var testModelMetrics = ComputePricedRaceMetrics(testResult);

                    MarketBlendMetrics? testFairMarket = null;
                    MarketBlendMetrics? testBlend = null;
                    if (bestFairMarket is not null)
                    {
                        // Freeze the market-only baseline and blend weights
                        // selected on dev. Fitting either on test would tune on
                        // the holdout.
                        testFairMarket = MarketBlendCalibration.Evaluate(
                            testResult.ValidationPredictions,
                            testResult.ValidationLabels,
                            testResult.ValidationRaceIds,
                            testResult.ValidationExamples,
                            0d,
                            1d);
                    }
                    if (bestBlend is not null)
                    {
                        testBlend = MarketBlendCalibration.Evaluate(
                            testResult.ValidationPredictions,
                            testResult.ValidationLabels,
                            testResult.ValidationRaceIds,
                            testResult.ValidationExamples,
                            bestBlend.ModelWeight,
                            bestBlend.MarketWeight);
                    }

                    var bestDevMarket = ComputeMarketMetrics(bestResult.ValidationExamples);
                    var bestDevModelMetrics = ComputePricedRaceMetrics(bestResult);

                    var finalRow = CloneParameter(bestModel, bestModel.Seed ?? _seed);
                    finalRow.RunDate = DateTime.UtcNow;
                    finalRow.EarliestTrainingDate = trainStart;
                    finalRow.Folds = 1;
                    finalRow.Fold = null;
                    finalRow.Objective = testResult.Objective;
                    finalRow.NormalizationMode = testResult.NormalizationMode;
                    finalRow.MarketOffsetScale = testResult.MarketOffsetScale;
                    finalRow.MarketOffsetMode = testResult.MarketOffsetMode;
                    finalRow.MarketBlind = testResult.MarketBlind;
                    finalRow.BlendModelWeight = _useMarketBlend ? bestBlend?.ModelWeight : 1.0;
                    finalRow.BlendMarketWeight = _useMarketBlend ? bestBlend?.MarketWeight : 0.0;
                    finalRow.BestEpoch = bestResult.BestEpoch;
                    finalRow.Seed = testResult.Seed;
                    finalRow.RaceBatchSize = bestModel.RaceBatchSize;
                    finalRow.WinRateAlpha = bestModel.WinRateAlpha;
                    finalRow.WinRateBeta = bestModel.WinRateBeta;
                    finalRow.TrainAccuracy = testResult.TrainAccuracy;
                    finalRow.TrainLoss = testResult.TrainLoss;
                    finalRow.TrainBrier = testResult.TrainBrier;
                    finalRow.DevAccuracy = bestDevModelMetrics.Top1Accuracy;
                    finalRow.DevLoss = bestDevModelMetrics.RunnerLogLoss;
                    finalRow.DevBrier = bestDevModelMetrics.Brier;
                    finalRow.DevRaceLogLoss = bestDevModelMetrics.RaceLogLoss;
                    finalRow.DevTop3Accuracy = bestDevModelMetrics.Top3Accuracy;
                    if (ensembleDevMetrics is not null)
                    {
                        finalRow.DevAccuracy = ensembleDevMetrics.Top1Accuracy;
                        finalRow.DevLoss = ensembleDevMetrics.RunnerLogLoss;
                        finalRow.DevBrier = ensembleDevMetrics.Brier;
                        finalRow.DevRaceLogLoss = ensembleDevMetrics.RaceLogLoss;
                        finalRow.DevTop3Accuracy = ensembleDevMetrics.Top3Accuracy;
                    }
                    finalRow.DevTemperature = bestResult.ValidationTemperature;
                    finalRow.DevMarketLogLoss = bestDevMarketLogLoss;
                    finalRow.DevMarketBrier = bestDevMarketBrier;
                    finalRow.DevMarketRaceLogLoss = bestDevMarket.RaceLogLoss;
                    finalRow.DevMarketTop1Accuracy = bestDevMarket.Top1Accuracy;
                    finalRow.DevMarketTop3Accuracy = bestDevMarket.Top3Accuracy;
                    finalRow.DevFairMarketLogLoss = bestFairMarket?.Metrics.LogLoss;
                    finalRow.DevFairMarketBrier = bestFairMarket?.Metrics.Brier;
                    finalRow.DevBlendLogLoss = bestBlend?.Metrics.LogLoss;
                    finalRow.DevBlendBrier = bestBlend?.Metrics.Brier;
                    finalRow.DevBlendRaceLogLoss = bestBlend?.Metrics.RaceLogLoss;
                    finalRow.ValidationAccuracy = bestResult.ValidationAccuracy;
                    finalRow.ValidationLoss = bestResult.ValidationLoss;
                    finalRow.ValidationBrier = bestResult.ValidationBrier;
                    finalRow.TestAccuracy = testModelMetrics.Top1Accuracy;
                    finalRow.TestLoss = testModelMetrics.RunnerLogLoss;
                    finalRow.TestBrier = testModelMetrics.Brier;
                    finalRow.TestRaceLogLoss = testModelMetrics.RaceLogLoss;
                    finalRow.TestTop3Accuracy = testModelMetrics.Top3Accuracy;
                    finalRow.TestTemperature = testResult.ValidationTemperature;
                    finalRow.TestMarketLogLoss = testMarketLogLoss;
                    finalRow.TestMarketBrier = testMarketBrier;
                    finalRow.TestMarketRaceLogLoss = testMarket.RaceLogLoss;
                    finalRow.TestMarketTop1Accuracy = testMarket.Top1Accuracy;
                    finalRow.TestMarketTop3Accuracy = testMarket.Top3Accuracy;
                    finalRow.TestFairMarketLogLoss = testFairMarket?.LogLoss;
                    finalRow.TestFairMarketBrier = testFairMarket?.Brier;
                    finalRow.TestBlendLogLoss = testBlend?.LogLoss;
                    finalRow.TestBlendBrier = testBlend?.Brier;
                    finalRow.TestBlendRaceLogLoss = testBlend?.RaceLogLoss;
                    finalRow.TrainingStartDate = trainStart;
                    finalRow.TrainingEndDate = devEnd;
                    finalRow.DevStartDate = devStart;
                    finalRow.DevEndDate = devEnd;
                    finalRow.TestStartDate = testStart;
                    finalRow.TestEndDate = testEnd;
                    finalRow.TrainingRaceCount = trainDevRaceIds.Count;
                    finalRow.DevRaceCount = devRaceIds.Count;
                    finalRow.TestRaceCount = testRaceIds.Count;
                    finalRow.TrainingRowCount = testView.TrainingRaces.Sum(r => r.Runners.Count);
                    finalRow.DevRowCount = bestResult.ValidationExamples.Count;
                    finalRow.TestRowCount = testResult.ValidationExamples.Count;
                    finalRow.SelectedOn = "Test";
                    finalRow.ModelArtifact = testResult.ModelArtifactFileName;
                    finalRow.ModelHash = testResult.ModelHash;
                    finalRow.FeatureSchemaHash = testResult.FeatureSchemaHash;
                    finalRow.EnsembleSize = ensembleMemberCount;
                    finalRow.IsActive = true;
                    finalRow.Id = _repository.InsertMLParameter(finalRow);
                    _repository.ActivateMLParameter(
                        finalRow.Id,
                        finalRow.ModelArtifact,
                        finalRow.ModelHash,
                        finalRow.FeatureSchemaHash);

                    // Keep the held-out test predictions for the betting
                    // simulator. Filters can then be changed without retraining.
                    try
                    {
                        var testPredictions = MarketBlendCalibration.ApplyFullField(
                            testResult.ValidationPredictions,
                            testResult.ValidationRaceIds,
                            testResult.ValidationExamples,
                            finalRow.BlendModelWeight ?? 1d,
                            finalRow.BlendMarketWeight ?? 0d);
                        var testSummaries = _repository.GetRaceSummaries(testResult.ValidationRaceIds);
                        var snapshot = SimulatorSnapshotBuilder.Build(
                            testResult,
                            testPredictions,
                            testSummaries,
                            finalRow,
                            testStart ?? DateTime.Today,
                            testEnd ?? DateTime.Today,
                            months: null,
                            trainDevRaceIds.Count,
                            testRaceIds.Count,
                            "Train AI test window");
                        _simulatorStore.Set(snapshot);
                        Console.WriteLine(
                            $"[TrainAI] Betting simulator snapshot prepared: {snapshot.Summary.ValidationRaceCount} races, {snapshot.Summary.RunnerCount} runners. Filters can now be applied without retraining.");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(
                            $"[TrainAI] Could not prepare the betting simulator snapshot: {ex.Message}");
                    }

                    var testPriced = HyperparameterTrainer.ComputePricedSubsetMetrics(testResult);
                    var blendReport = testBlend is not null && testFairMarket is not null
                        ? $" Test blend LL {testBlend.LogLoss:F4} vs fair market {testFairMarket.LogLoss:F4}."
                        : string.Empty;
                    var reportedDevMetrics = ensembleDevMetrics ?? bestDevModelMetrics;
                    Console.WriteLine(
                        $"[TrainAI] Persisted the best model ({bestModel.Layers} layers, {bestModel.Units} units, {testResult.Objective}, offset {testResult.MarketOffsetMode}, blind {testResult.MarketBlind}, members {ensembleMemberCount}) trained on {trainDevRaceIds.Count} races. Dev race LL: {reportedDevMetrics.RaceLogLoss:F4} (fair {bestFairMarket?.Metrics.RaceLogLoss:F4}, SP {bestDevMarket.RaceLogLoss:F4}); Test race LL: {testModelMetrics.RaceLogLoss:F4} (fair {testFairMarket?.RaceLogLoss:F4}, SP {testMarket.RaceLogLoss:F4}); Test top1 {testModelMetrics.Top1Accuracy:P2} vs SP {testMarket.Top1Accuracy:P2}; Test top3 {testModelMetrics.Top3Accuracy:P2} vs SP {testMarket.Top3Accuracy:P2}; Test Brier: {testModelMetrics.Brier:F4} (SP {testMarketBrier:F4}).{blendReport} Test priced subset: acc {testPriced.Accuracy:P2}, loss {testPriced.LogLoss:F4}, Brier {testPriced.Brier:F4} over {testPriced.RaceCount} races.");
                }

                _logger.LogInformation("Train AI finished.");
                Console.WriteLine("[TrainAI] Training run finished.");
            });

            return RedirectToAction("Index", "Home");
        }

        [HttpGet]
        public IActionResult TestAI()
        {
            return View(new AITestResultViewModel
            {
                ValidationStart = DateTime.Today.AddMonths(-1),
                ValidationEnd = DateTime.Today,
                SelectedValidationMonths = 1,
                StartingBankroll = 100m
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestAI(AITestResultViewModel request)
        {
            int months = request.SelectedValidationMonths;
            if (months <= 0)
            {
                months = SimulatorRunRequest.MinimumMonths;
            }
            else if (months > SimulatorRunRequest.MaximumMonths)
            {
                months = SimulatorRunRequest.MaximumMonths;
            }

            var monthLabel = months == 1 ? "month" : "months";

            var validationEnd = DateTime.Today;
            var validationStart = validationEnd.AddMonths(-months);
            var startingBankroll = request.StartingBankroll < 0 ? 0m : request.StartingBankroll;
            var viewModel = new AITestResultViewModel
            {
                ValidationStart = validationStart,
                ValidationEnd = validationEnd,
                SelectedValidationMonths = months,
                StartingBankroll = startingBankroll
            };

            var savedParameter = _repository.GetBestMLParameter();
            if (savedParameter is null)
            {
                viewModel.Message = "No saved AI parameters were found. Please train the AI before running a test.";
                return View(viewModel);
            }

            var validationRaceIds = GetTrainingRaceIds(validationStart, validationEnd);
            if (validationRaceIds.Count == 0)
            {
                viewModel.Message = $"No eligible races were found in the last {months} {monthLabel} to use for validation.";
                return View(viewModel);
            }

            // Train only on races that occurred before the validation window.
            // GetRaceIdsOutsideRange used to include races after the window,
            // which leaked future form into a historical backtest.
            var trainingRaceIds = GetTrainingRaceIds(
                _minimumMarketRaceDate,
                validationStart.AddDays(-1));
            if (trainingRaceIds.Count == 0)
            {
                viewModel.Message = "No eligible training data is available before the validation window.";
                return View(viewModel);
            }

            var validationSet = new HashSet<int>(validationRaceIds);
            var trainingSet = new HashSet<int>(trainingRaceIds);
            var dataset = _trainer.LoadTrainingDataset(
                trainingSet,
                validationSet,
                includeIdentifiers: true,
                requireMarketPrice: !_keepUnpricedRunners,
                metadataRaceIds: trainingSet,
                marketBlind: savedParameter.MarketBlind);
            if (dataset.TrainingRaces.Count == 0 || dataset.ValidationRaces.Count == 0)
            {
                viewModel.Message = "The priced training or validation dataset was empty after preparation.";
                return View(viewModel);
            }

            var parameter = new MLParameter
            {
                Units = MLParameterValidator.EnsureUnits(savedParameter.Units, 0),
                Dropout = MLParameterValidator.EnsureDropout(savedParameter.Dropout, 0),
                Layers = MLParameterValidator.EnsureLayers(savedParameter.Layers, 0),
                LearningRate = MLParameterValidator.EnsureLearningRate(savedParameter.LearningRate, 0.005),
                Epochs = savedParameter.BestEpoch.HasValue
                    ? Math.Max(1, savedParameter.BestEpoch.Value + 1)
                    : MLParameterValidator.EnsurePositive(savedParameter.Epochs, 12),
                BatchSize = MLParameterValidator.EnsurePositive(savedParameter.BatchSize, 500),
                // Old MLParameters rows predate the objective column and were
                // trained with the pointwise sigmoid objective.
                Objective = savedParameter.Objective ?? "Binary",
                // Old rows predate the offset-mode column and used log-odds.
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

            // Freeze the calibration learned during training. Fitting a new
            // temperature on the evaluation window would tune on the test set.
            // Old rows without a stored dev temperature use T=1 rather than
            // fitting calibration on the evaluation window.
            float? temperatureOverride = savedParameter.DevTemperature.HasValue
                ? (float)savedParameter.DevTemperature.Value
                : 1.0f;
            int savedEnsembleSize = Math.Clamp(savedParameter.EnsembleSize ?? 1, 1, 10);
            bool savedIsRaceSoftmax = string.Equals(
                savedParameter.Objective,
                "RaceSoftmax",
                StringComparison.OrdinalIgnoreCase);
            var result = await Task.Run(() =>
            {
                HyperparameterTrainer.TrainingResult TrainMember(int seed)
                {
                    var member = CloneParameter(parameter, seed);
                    member.Epochs = parameter.Epochs;
                    return _trainer.Train(
                        member,
                        0,
                        1,
                        dataset,
                        persistWeights: false,
                        temperatureOverride: temperatureOverride,
                        earlyStopOnValidation: false,
                        blendModelWeight: savedParameter.BlendModelWeight ?? 1d,
                        blendMarketWeight: savedParameter.BlendMarketWeight ?? 0d);
                }

                if (savedEnsembleSize <= 1 || !savedIsRaceSoftmax)
                {
                    return TrainMember(savedParameter.Seed ?? _seed);
                }

                var memberResults = new List<HyperparameterTrainer.TrainingResult>();
                for (int i = 0; i < savedEnsembleSize; i++)
                {
                    var memberResult = TrainMember((savedParameter.Seed ?? _seed) + i);
                    if (memberResult.Model is not null)
                    {
                        memberResults.Add(memberResult);
                    }
                }

                if (memberResults.Count <= 1)
                {
                    return memberResults.FirstOrDefault() ?? TrainMember(savedParameter.Seed ?? _seed);
                }

                var first = memberResults[0];
                var predictions = RaceMetrics.AverageLogProbabilities(
                    memberResults.Select(member => member.ValidationPredictions).ToList(),
                    first.ValidationRaceIds);
                var metrics = RaceMetrics.Compute(
                    first.ValidationRaceIds,
                    predictions,
                    first.ValidationLabels);
                return new HyperparameterTrainer.TrainingResult
                {
                    TrainAccuracy = first.TrainAccuracy,
                    TrainLoss = first.TrainLoss,
                    TrainBrier = first.TrainBrier,
                    TrainRaceLogLoss = first.TrainRaceLogLoss,
                    TrainTop3Accuracy = first.TrainTop3Accuracy,
                    ValidationAccuracy = metrics.Top1Accuracy,
                    ValidationLoss = metrics.RunnerLogLoss,
                    ValidationBrier = metrics.Brier,
                    ValidationRaceLogLoss = metrics.RaceLogLoss,
                    ValidationTop3Accuracy = metrics.Top3Accuracy,
                    ValidationTemperature = first.ValidationTemperature,
                    Objective = first.Objective,
                    AuxiliaryWinLossWeight = first.AuxiliaryWinLossWeight,
                    AuxiliaryFinishLossWeight = first.AuxiliaryFinishLossWeight,
                    BestEpoch = first.BestEpoch,
                    MarketOffsetScale = first.MarketOffsetScale,
                    MarketOffsetMode = first.MarketOffsetMode,
                    MarketBlind = first.MarketBlind,
                    Seed = first.Seed,
                    NormalizationMode = first.NormalizationMode,
                    TrainingPredictions = first.TrainingPredictions,
                    TrainingLabels = first.TrainingLabels,
                    TrainingRaceIds = first.TrainingRaceIds,
                    ValidationPredictions = predictions,
                    ValidationLabels = first.ValidationLabels,
                    ValidationRaceIds = first.ValidationRaceIds,
                    ValidationExamples = first.ValidationExamples
                };
            });
            var market = ComputeMarketMetrics(result.ValidationExamples);
            double marketLogLoss = market.RunnerLogLoss;
            double marketBrier = market.Brier;
            var modelMetrics = ComputePricedRaceMetrics(result);

            double blendModelWeight = savedParameter.BlendModelWeight ?? 1d;
            double blendMarketWeight = savedParameter.BlendMarketWeight ?? 0d;
            var fairMarket = MarketBlendCalibration.Evaluate(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples,
                0d,
                1d);
            var blend = MarketBlendCalibration.Evaluate(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples,
                blendModelWeight,
                blendMarketWeight);

            viewModel.ParameterUsed = savedParameter;
            viewModel.TrainAccuracy = result.TrainAccuracy;
            viewModel.TrainLoss = result.TrainLoss;
            viewModel.TrainBrier = result.TrainBrier;
            viewModel.ValidationAccuracy = result.ValidationAccuracy;
            viewModel.ValidationLoss = result.ValidationLoss;
            viewModel.ValidationBrier = result.ValidationBrier;
            viewModel.ValidationRaceLogLoss = modelMetrics.RaceLogLoss;
            viewModel.ValidationTop3Accuracy = modelMetrics.Top3Accuracy;
            viewModel.ValidationTemperature = result.ValidationTemperature;
            viewModel.MarketLogLoss = marketLogLoss;
            viewModel.MarketBrier = marketBrier;
            viewModel.MarketRaceLogLoss = market.RaceLogLoss;
            viewModel.MarketTop1Accuracy = market.Top1Accuracy;
            viewModel.MarketTop3Accuracy = market.Top3Accuracy;
            viewModel.FairMarketLogLoss = fairMarket.LogLoss;
            viewModel.FairMarketBrier = fairMarket.Brier;
            viewModel.FairMarketRaceLogLoss = fairMarket.RaceLogLoss;
            viewModel.BlendLogLoss = blend.LogLoss;
            viewModel.BlendBrier = blend.Brier;
            viewModel.BlendRaceLogLoss = blend.RaceLogLoss;
            viewModel.BlendModelWeight = blendModelWeight;
            viewModel.BlendMarketWeight = blendMarketWeight;
            viewModel.ValidationRaceCount = dataset.ValidationRaces.Count;
            viewModel.TrainingRaceCount = dataset.TrainingRaces.Count;
            var pricedSubset = HyperparameterTrainer.ComputePricedSubsetMetrics(result);
            var blendText = blend.RunnerCount > 0
                ? $" Blend log loss {blend.LogLoss:F4} vs fair market {fairMarket.LogLoss:F4}."
                : string.Empty;
            viewModel.Message = $"Validation over {viewModel.ValidationRaceCount} races: race log loss {modelMetrics.RaceLogLoss:F4} vs fair {fairMarket.RaceLogLoss:F4} and SP {market.RaceLogLoss:F4}; top1 {modelMetrics.Top1Accuracy:P2} vs SP {market.Top1Accuracy:P2}; top3 {modelMetrics.Top3Accuracy:P2} vs SP {market.Top3Accuracy:P2}; Brier {result.ValidationBrier:F4} vs SP {marketBrier:F4}.{blendText} Priced subset: accuracy {pricedSubset.Accuracy:P2}, log loss {pricedSubset.LogLoss:F4}, Brier {pricedSubset.Brier:F4} over {pricedSubset.RaceCount} races.";

            var blendedPredictions = MarketBlendCalibration.ApplyFullField(
                result.ValidationPredictions,
                result.ValidationRaceIds,
                result.ValidationExamples,
                blendModelWeight,
                blendMarketWeight);
            var raceSummaries = _repository.GetRaceSummaries(result.ValidationRaceIds);
            var snapshot = SimulatorSnapshotBuilder.Build(
                result,
                blendedPredictions,
                raceSummaries,
                savedParameter,
                validationStart,
                validationEnd,
                months,
                dataset.TrainingRaces.Count,
                dataset.ValidationRaces.Count,
                $"Test AI window ({months} {(months == 1 ? "month" : "months")})");

            var simulationFilter = CreateDefaultSimulatorFilter(snapshot);
            simulationFilter.StartingBankroll = startingBankroll;
            // Test AI does not display the bootstrap interval; skip it.
            simulationFilter.BootstrapSamples = 0;
            // The historical Test AI path required strictly positive value.
            // A tiny epsilon preserves that behaviour while sharing the engine.
            simulationFilter.MinEdgePercent = 0.000001m;
            viewModel.Simulation = RunValidationSimulation(snapshot, simulationFilter);

            // Publish the same runner-level predictions to the betting
            // simulator so the user can re-filter them without retraining.
            try
            {
                _simulatorStore.Set(snapshot);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not publish the Test AI prediction snapshot.");
            }

            return View(viewModel);
        }

        /// <summary>
        /// Betting simulator landing page. If a previous training/test run has
        /// already published a prediction snapshot, the default filters are
        /// applied immediately. Otherwise the prepare form is shown.
        /// </summary>
        [HttpGet]
        public IActionResult BettingSimulator(string? snapshotKey = null)
        {
            var snapshot = _simulatorStore.Get(snapshotKey);
            if (snapshot is null && !string.IsNullOrWhiteSpace(snapshotKey))
            {
                snapshot = _simulatorStore.Get();
            }

            var filter = CreateDefaultSimulatorFilter(snapshot);
            filter.SnapshotKey = snapshot?.Summary.Key;

            var viewModel = BuildSimulatorView(snapshot, filter, runAnalysis: true);
            if (TempData["Message"] is string message)
            {
                viewModel.Message = message;
            }

            return View(viewModel);
        }

        /// <summary>
        /// Queues a fresh snapshot for the selected evaluation window. The
        /// heavy history replay/training runs on the background worker so the
        /// browser request is not held open for several minutes.
        /// </summary>
        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult SimulatorRun(SimulatorRunRequest request)
        {
            if (!ModelState.IsValid)
            {
                var snapshot = _simulatorStore.Get();
                var invalid = BuildSimulatorView(
                    snapshot,
                    CreateDefaultSimulatorFilter(snapshot),
                    runAnalysis: false);
                invalid.Run = request;
                return View("BettingSimulator", invalid);
            }

            var payload = new Dictionary<string, string?>
            {
                ["requestJson"] = JsonSerializer.Serialize(request)
            };

            if (!_jobQueue.TryEnqueue(
                    BackgroundJobKind.SimulatorPrepare,
                    "Betting simulator snapshot",
                    payload))
            {
                TempData["Message"] =
                    "A simulator snapshot preparation is already queued or running. Check the Jobs page or refresh this page shortly.";
            }
            else
            {
                var window = request.FromDate.HasValue && request.ToDate.HasValue
                    ? $"{request.FromDate:yyyy-MM-dd} → {request.ToDate:yyyy-MM-dd}"
                    : $"{request.Months}-month window ending at the latest race date";
                TempData["Message"] =
                    $"Snapshot preparation for the {window} has been queued. " +
                    "It runs in the background; this page will update when it is ready.";
            }

            return RedirectToAction(nameof(BettingSimulator));
        }

        private void PopulateSimulatorJob(BettingSimulatorViewModel viewModel)
        {
            var latestJob = _jobStore.List()
                .FirstOrDefault(job => job.Kind == BackgroundJobKind.SimulatorPrepare);
            if (latestJob is null)
            {
                return;
            }

            viewModel.PrepareJob = new SimulatorPrepareJobViewModel
            {
                JobId = latestJob.Id,
                Status = latestJob.Status.ToString(),
                Message = latestJob.Message,
                Progress = latestJob.Progress,
                Error = latestJob.Error,
                CreatedUtc = latestJob.CreatedUtc,
                StartedUtc = latestJob.StartedUtc,
                CompletedUtc = latestJob.CompletedUtc
            };
        }

        /// <summary>
        /// Re-runs the betting loop against the cached snapshot. This is a GET
        /// so the result is bookmarkable, refresh-safe and shareable.
        /// </summary>
        [AcceptVerbs("GET", "POST")]
        public IActionResult SimulatorApply(SimulatorFilterRequest filter)
        {
            filter ??= new SimulatorFilterRequest();
            var snapshot = _simulatorStore.Get(filter.SnapshotKey);
            if (snapshot is null && !string.IsNullOrWhiteSpace(filter.SnapshotKey))
            {
                snapshot = _simulatorStore.Get();
                filter.SnapshotKey = snapshot?.Summary.Key;
            }

            if (snapshot is null)
            {
                var viewModel = BuildSimulatorView(
                    null,
                    CreateDefaultSimulatorFilter(null),
                    runAnalysis: false);
                viewModel.Message =
                    "No prediction snapshot is loaded. Prepare an evaluation window first, or run Train AI/Test AI and return to this page.";
                return View("BettingSimulator", viewModel);
            }

            if (!ModelState.IsValid)
            {
                var invalid = BuildSimulatorView(snapshot, filter, runAnalysis: false);
                invalid.Result = null;
                invalid.PagedBets = Array.Empty<SimulatorBet>();
                invalid.Message = "Some filter values are invalid. Correct the highlighted fields and apply again.";
                return View("BettingSimulator", invalid);
            }

            return View("BettingSimulator", BuildSimulatorView(snapshot, filter, runAnalysis: true));
        }

        /// <summary>
        /// Downloads the full un-paged bet list for the current filters.
        /// </summary>
        [AcceptVerbs("GET", "POST")]
        public IActionResult SimulatorExport(SimulatorFilterRequest filter)
        {
            filter ??= new SimulatorFilterRequest();
            var snapshot = _simulatorStore.Get(filter.SnapshotKey);
            if (snapshot is null && !string.IsNullOrWhiteSpace(filter.SnapshotKey))
            {
                snapshot = _simulatorStore.Get();
                filter.SnapshotKey = snapshot?.Summary.Key;
            }

            if (snapshot is null)
            {
                return NotFound("No simulator snapshot is loaded.");
            }

            var result = BettingSimulationEngine.Simulate(snapshot, filter);
            var builder = new System.Text.StringBuilder();
            builder.AppendLine(
                "Date,RaceId,Race,Course,Type,Class,Runners,Horse,Odds,NetOdds,BSP,LTP,AIProb,MarketProb,Edge,Differential,Stake,Profit,ExpectedProfit,BankrollAfter,Result,CLV");
            foreach (var bet in result.Bets)
            {
                builder.Append(Csv(bet.RaceDate?.ToString("yyyy-MM-dd")))
                    .Append(',').Append(bet.RaceId)
                    .Append(',').Append(Csv(bet.RaceTitle))
                    .Append(',').Append(Csv(bet.CourseName))
                    .Append(',').Append(Csv(bet.RaceType))
                    .Append(',').Append(bet.Class?.ToString() ?? string.Empty)
                    .Append(',').Append(bet.RunnerCount)
                    .Append(',').Append(Csv(bet.HorseName))
                    .Append(',').Append(bet.DecimalOdds.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.EffectiveDecimalOdds.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.BetfairSp?.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty)
                    .Append(',').Append(bet.BetfairLastTradedPrice?.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty)
                    .Append(',').Append(bet.AiProbability.ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.MarketProbability.ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.Edge.ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.Differential.ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.Stake.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.Profit.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.ExpectedProfit.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.BankrollAfter.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture))
                    .Append(',').Append(bet.Won ? "Won" : "Lost")
                    .Append(',').Append(bet.ClvPercent.ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture))
                    .AppendLine();
            }

            var fileName = $"simulator-{snapshot.Summary.Key}-{DateTime.UtcNow:yyyyMMddHHmmss}.csv";
            return File(
                System.Text.Encoding.UTF8.GetBytes(builder.ToString()),
                "text/csv",
                fileName);

            static string Csv(string? value)
            {
                if (string.IsNullOrEmpty(value))
                {
                    return string.Empty;
                }

                return value.Contains(',') || value.Contains('"') || value.Contains('\n')
                    ? $"\"{value.Replace("\"", "\"\"")}\""
                    : value;
            }
        }

        [HttpGet]
        public IActionResult SimulatorJobStatus()
        {
            var latestJob = _jobStore.List()
                .FirstOrDefault(job => job.Kind == BackgroundJobKind.SimulatorPrepare);
            if (latestJob is null)
            {
                return Json(new { isActive = false, status = "None" });
            }

            bool isActive = latestJob.Status is BackgroundJobStatus.Queued
                or BackgroundJobStatus.Running;
            return Json(new
            {
                jobId = latestJob.Id,
                isActive,
                status = latestJob.Status.ToString(),
                message = latestJob.Message,
                progress = latestJob.Progress,
                error = latestJob.Error,
                completedUtc = latestJob.CompletedUtc
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult SimulatorCancel()
        {
            var latestJob = _jobStore.List()
                .FirstOrDefault(job =>
                    job.Kind == BackgroundJobKind.SimulatorPrepare &&
                    (job.Status == BackgroundJobStatus.Queued ||
                     job.Status == BackgroundJobStatus.Running));
            if (latestJob is null)
            {
                TempData["Message"] = "There is no active simulator preparation to cancel.";
            }
            else
            {
                _jobStore.RequestCancel(latestJob.Id);
                TempData["Message"] = "Cancellation requested.";
            }

            return RedirectToAction(nameof(BettingSimulator));
        }

        private BettingSimulatorViewModel BuildSimulatorView(
            BettingSimulationSnapshot? snapshot,
            SimulatorFilterRequest filter,
            bool runAnalysis)
        {
            filter ??= new SimulatorFilterRequest();
            var viewModel = new BettingSimulatorViewModel
            {
                Run = new SimulatorRunRequest
                {
                    Months = snapshot?.Summary.Months ?? 6
                },
                Filter = filter,
                Snapshot = snapshot?.Summary,
                Snapshots = _simulatorStore.List(),
                SelectedSnapshotKey = snapshot?.Summary.Key,
                Page = filter.Page,
                PageSize = filter.PageSize
            };

            PopulateSimulatorJob(viewModel);

            if (snapshot is null)
            {
                return viewModel;
            }

            viewModel.Result = BettingSimulationEngine.Simulate(snapshot, filter);
            viewModel.PageSize = Math.Clamp(filter.PageSize, 10, 1_000);
            viewModel.PageCount = Math.Max(
                1,
                (int)Math.Ceiling((double)viewModel.Result.BetCount / viewModel.PageSize));
            viewModel.Page = Math.Clamp(filter.Page, 1, viewModel.PageCount);
            viewModel.PagedBets = viewModel.Result.Bets
                .Skip((viewModel.Page - 1) * viewModel.PageSize)
                .Take(viewModel.PageSize)
                .ToList();

            if (runAnalysis)
            {
                viewModel.Comparison = SimulatorAnalytics.BuildComparison(snapshot, filter);
                viewModel.Baselines = SimulatorAnalytics.BuildBaselines(snapshot, filter);
                viewModel.WalkForward = SimulatorAnalytics.BuildWalkForward(snapshot, filter);
            }

            var activeModel = _repository.GetBestMLParameter();
            if (!string.IsNullOrWhiteSpace(activeModel?.ModelHash) &&
                !string.IsNullOrWhiteSpace(snapshot.Summary.ModelHash) &&
                !string.Equals(
                    activeModel.ModelHash,
                    snapshot.Summary.ModelHash,
                    StringComparison.OrdinalIgnoreCase))
            {
                viewModel.StaleModelWarning =
                    "This snapshot was produced by a different model than the currently active one. " +
                    "Prepare a fresh snapshot before using these results for live decisions.";
            }

            return viewModel;
        }

        private SimulatorFilterRequest CreateDefaultSimulatorFilter(
            BettingSimulationSnapshot? snapshot)
        {
            return new SimulatorFilterRequest
            {
                StartingBankroll = 100m,
                MaxKellyPercent = Math.Clamp(_maxKellyFraction ?? 0.10m, 0m, 1m) * 100m,
                CommissionPercent = Math.Clamp(_commissionRate, 0m, 1m) * 100m,
                RaceType = "Any",
                Surface = "Any",
                Going = "Any",
                AgeRestriction = "Any",
                MinOdds = 1.01m,
                MinEdgePercent = SimulatorFilterRequest.DefaultMinimumEdgePercent,
                MinProbabilitySpread = SimulatorFilterRequest.DefaultMinimumProbabilitySpread,
                StakeMode = SimulatorStakeMode.Kelly,
                SelectionStrategy = SimulatorSelectionStrategy.HighestDifferential,
                PriceSource = SimulatorPriceSource.StartingPrice,
                BetsPerRace = 1,
                MaxStakePercent = 100m,
                MaxDailyStakePercent = 100m,
                MaxDailyLossPercent = 100m,
                ConfirmPercent = 30m,
                BootstrapSamples = 1_000,
                Page = 1,
                PageSize = 100,
                SnapshotKey = snapshot?.Summary.Key
            };
        }

        private sealed class ModelCandidate
        {
            public MLParameter Model { get; init; } = new();
            public HyperparameterTrainer.TrainingResult Result { get; init; } = null!;
            public double SelectionScore { get; init; }
        }

        private static MLParameter CloneParameter(MLParameter source, int seed)
        {
            return new MLParameter
            {
                RunDate = source.RunDate,
                EarliestTrainingDate = source.EarliestTrainingDate,
                Units = source.Units,
                Dropout = source.Dropout,
                Layers = source.Layers,
                LearningRate = source.LearningRate,
                Epochs = source.Epochs,
                BatchSize = source.BatchSize,
                Folds = source.Folds,
                Fold = source.Fold,
                Objective = source.Objective,
                MarketOffsetMode = source.MarketOffsetMode,
                MarketBlind = source.MarketBlind,
                BlendModelWeight = source.BlendModelWeight,
                BlendMarketWeight = source.BlendMarketWeight,
                NormalizationMode = source.NormalizationMode,
                FeatureBlocks = source.FeatureBlocks,
                RaceBatchSize = source.RaceBatchSize,
                WinRateAlpha = source.WinRateAlpha,
                WinRateBeta = source.WinRateBeta,
                Seed = seed
            };
        }

        private static RaceEvaluationMetrics ComputePricedRaceMetrics(
            HyperparameterTrainer.TrainingResult result)
        {
            return ComputePricedRaceMetrics(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples);
        }

        private static RaceEvaluationMetrics ComputePricedRaceMetrics(
            IReadOnlyList<float> predictions,
            IReadOnlyList<float> labels,
            IReadOnlyList<int> raceIds,
            IReadOnlyList<HyperparameterTrainer.RunnerExample> examples)
        {
            int count = Math.Min(
                predictions.Count,
                Math.Min(labels.Count, Math.Min(raceIds.Count, examples.Count)));
            var pricedPredictions = new List<float>(count);
            var pricedLabels = new List<float>(count);
            var pricedRaceIds = new List<int>(count);
            for (int i = 0; i < count; i++)
            {
                if (!examples[i].HasMarketPrice)
                {
                    continue;
                }

                pricedPredictions.Add(predictions[i]);
                pricedLabels.Add(labels[i]);
                pricedRaceIds.Add(raceIds[i]);
            }

            return RaceMetrics.Compute(pricedRaceIds, pricedPredictions, pricedLabels);
        }

        private RaceEvaluationMetrics ComputeMarketMetrics(
            IReadOnlyList<HyperparameterTrainer.RunnerExample>? examples)
        {
            if (examples is null || examples.Count == 0)
            {
                return RaceEvaluationMetrics.Empty;
            }

            var predictions = new List<float>();
            var labels = new List<float>();
            var raceIds = new List<int>();

            foreach (var race in examples.GroupBy(e => e.RaceId))
            {
                foreach (var entry in race)
                {
                    if (!entry.StartingPriceDecimal.HasValue ||
                        entry.StartingPriceDecimal.Value <= 1m)
                    {
                        continue;
                    }

                    decimal odds = BettingMath.CalculateEffectiveDecimalOdds(
                        entry.StartingPriceDecimal.Value,
                        _commissionRate);
                    if (odds <= 1m)
                    {
                        continue;
                    }

                    predictions.Add((float)(1.0 / (double)odds));
                    labels.Add(entry.Label);
                    raceIds.Add(entry.RaceId);
                }
            }

            // RaceMetrics normalises within each race and returns both the
            // race-level and runner-level scores, so the SP baseline and the
            // model are measured on exactly the same probability space.
            return RaceMetrics.Compute(raceIds, predictions, labels);
        }

        private static ValidationSimulationResult RunValidationSimulation(
            BettingSimulationSnapshot snapshot,
            SimulatorFilterRequest filter)
        {
            var result = BettingSimulationEngine.Simulate(snapshot, filter);
            var bets = result.Bets
                .Select(bet => new ValidationBetResult
                {
                    RaceId = bet.RaceId,
                    RaceDate = bet.RaceDate,
                    RaceTitle = bet.RaceTitle,
                    CourseName = bet.CourseName,
                    HorseName = bet.HorseName,
                    RaceType = bet.RaceType,
                    Class = bet.Class,
                    AgeRestriction = bet.AgeRestriction,
                    Surface = bet.Surface,
                    Going = bet.Going,
                    DistanceYards = bet.DistanceYards,
                    DistanceText = bet.DistanceText,
                    RunnerCount = bet.RunnerCount is > 0 and <= byte.MaxValue
                        ? (byte)bet.RunnerCount
                        : null,
                    DecimalOdds = bet.DecimalOdds,
                    EffectiveDecimalOdds = bet.EffectiveDecimalOdds,
                    CommissionRate = Math.Clamp(filter.CommissionPercent / 100m, 0m, 1m),
                    AiDecimalOdds = BettingMath.CalculateAiDecimalOdds(bet.AiProbability),
                    AiProbability = bet.AiProbability,
                    MarketProbability = bet.MarketProbability,
                    Differential = bet.Differential,
                    Stake = bet.Stake,
                    Won = bet.Won,
                    BankrollAfter = bet.BankrollAfter
                })
                .ToList();

            return new ValidationSimulationResult
            {
                StartingBankroll = result.StartingBankroll,
                EndingBankroll = result.EndingBankroll,
                TotalStaked = result.TotalStaked,
                BetCount = result.BetCount,
                WinCount = result.WinCount,
                Bets = bets
            };
        }
    }
}
