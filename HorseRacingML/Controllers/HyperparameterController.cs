using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HorseRacingML.Data;
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
        private readonly ILogger<HyperparameterController> _logger;

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
        private readonly double _winRateAlpha;
        private readonly double _winRateBeta;
        private readonly bool _keepUnpricedRunners;
        private readonly bool _requirePreOffPrice;

        public HyperparameterController(
            RacingRepository repository,
            HyperparameterTrainer trainer,
            IConfiguration configuration,
            BackgroundJobQueue jobQueue,
            ILogger<HyperparameterController> logger)
        {
            _repository = repository;
            _trainer = trainer;
            _jobQueue = jobQueue;
            _logger = logger;
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
                        var (devMarketLogLoss, devMarketBrier) = ComputeMarketMetrics(result.ValidationExamples);
                        bool isRaceSoftmax = string.Equals(result.Objective, "RaceSoftmax", StringComparison.OrdinalIgnoreCase);

                        // The fair baseline uses the same no-vig probabilities the
                        // model sees, and the blend gate asks whether adding the
                        // model improves that baseline. Closing SP remains a
                        // separate, harder benchmark.
                        MarketBlendParameters? devFairMarket = null;
                        MarketBlendParameters? devBlend = null;
                        if (_useMarketBlend && isRaceSoftmax && result.ValidationExamples.Count > 0)
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

                        model.TrainAccuracy = result.TrainAccuracy;
                        model.TrainLoss = result.TrainLoss;
                        model.TrainBrier = result.TrainBrier;
                        model.ValidationAccuracy = result.ValidationAccuracy;
                        model.ValidationLoss = result.ValidationLoss;
                        model.ValidationBrier = result.ValidationBrier;
                        model.DevAccuracy = result.ValidationAccuracy;
                        model.DevLoss = result.ValidationLoss;
                        model.DevBrier = result.ValidationBrier;
                        model.DevTemperature = result.ValidationTemperature;
                        model.DevMarketLogLoss = devMarketLogLoss;
                        model.DevMarketBrier = devMarketBrier;
                        model.DevFairMarketLogLoss = devFairMarket?.Metrics.LogLoss;
                        model.DevFairMarketBrier = devFairMarket?.Metrics.Brier;
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
                                     devBlend.Metrics.LogLoss <
                                     devFairMarket.Metrics.LogLoss - _minimumBlendImprovement);
                                selectionScore = devBlend?.Metrics.LogLoss ?? double.MaxValue;
                            }
                            else
                            {
                                // Model-only selection: keep the market metrics
                                // as diagnostics, but do not reject a model just
                                // because the post-model blend did not improve
                                // on the market prior.
                                passedMarketGate = true;
                                selectionScore = result.ValidationBrier;
                            }
                        }
                        else
                        {
                            passedMarketGate = !_requireMarketEdge || result.ValidationBrier < devMarketBrier;
                            selectionScore = result.ValidationBrier;
                        }

                        model.SelectedOn = passedMarketGate ? "Dev" : "Rejected";

                        _repository.InsertMLParameter(model);

                        var devPriced = HyperparameterTrainer.ComputePricedSubsetMetrics(result);
                        var blendText = devBlend is not null && devFairMarket is not null
                            ? $"blend LL {devBlend.Metrics.LogLoss:F4} (a={devBlend.ModelWeight:F2}, b={devBlend.MarketWeight:F2}) vs fair market {devFairMarket.Metrics.LogLoss:F4}"
                            : "blend not fitted";
                        Console.WriteLine(
                            $"[TrainAI] Model {modelIndex}: train acc {result.TrainAccuracy:F4}, dev acc {result.ValidationAccuracy:F4}, dev Brier {result.ValidationBrier:F4}, closing market Brier {devMarketBrier:F4}, closing market log loss {devMarketLogLoss:F4}, temperature {result.ValidationTemperature:F2}, best epoch {result.BestEpoch}. {blendText}. Priced subset: acc {devPriced.Accuracy:P2}, log loss {devPriced.LogLoss:F4}, Brier {devPriced.Brier:F4} over {devPriced.RaceCount} races.");

                        if (result.ValidationExamples.Count == 0)
                        {
                            Console.WriteLine("[TrainAI] No dev examples were produced; refusing to select this model.");
                            continue;
                        }

                        if (!passedMarketGate)
                        {
                            Console.WriteLine(isRaceSoftmax
                                ? "[TrainAI] The dev blend did not improve on the fair no-vig market baseline; model is recorded as rejected and is not eligible for selection."
                                : "[TrainAI] Dev Brier did not beat the dev closing market baseline; model is recorded as rejected and is not eligible for selection.");
                            continue;
                        }

                        if (selectionScore < bestSelectionScore)
                        {
                            bestModel = CloneParameter(model, model.Seed ?? _seed);
                            bestResult = result;
                            bestBlend = devBlend;
                            bestFairMarket = devFairMarket;
                            bestSelectionScore = selectionScore;
                            bestDevMarketLogLoss = devMarketLogLoss;
                            bestDevMarketBrier = devMarketBrier;
                            Console.WriteLine(isRaceSoftmax
                                ? $"[TrainAI] Model {modelIndex} is the best dev blend so far (log loss {bestSelectionScore:F4} vs fair market {devFairMarket?.Metrics.LogLoss:F4})."
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

                    var testResult = _trainer.Train(
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
                    var (testMarketLogLoss, testMarketBrier) = ComputeMarketMetrics(testResult.ValidationExamples);

                    MarketBlendMetrics? testFairMarket = null;
                    MarketBlendMetrics? testBlend = null;
                    if (bestBlend is not null && bestFairMarket is not null)
                    {
                        // Freeze the dev-fitted market sharpening and blend
                        // weights. Fitting either on test would tune on the
                        // holdout.
                        testFairMarket = MarketBlendCalibration.Evaluate(
                            testResult.ValidationPredictions,
                            testResult.ValidationLabels,
                            testResult.ValidationRaceIds,
                            testResult.ValidationExamples,
                            0d,
                            bestFairMarket.MarketWeight);
                        testBlend = MarketBlendCalibration.Evaluate(
                            testResult.ValidationPredictions,
                            testResult.ValidationLabels,
                            testResult.ValidationRaceIds,
                            testResult.ValidationExamples,
                            bestBlend.ModelWeight,
                            bestBlend.MarketWeight);
                    }

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
                    finalRow.DevAccuracy = bestResult.ValidationAccuracy;
                    finalRow.DevLoss = bestResult.ValidationLoss;
                    finalRow.DevBrier = bestResult.ValidationBrier;
                    finalRow.DevTemperature = bestResult.ValidationTemperature;
                    finalRow.DevMarketLogLoss = bestDevMarketLogLoss;
                    finalRow.DevMarketBrier = bestDevMarketBrier;
                    finalRow.DevFairMarketLogLoss = bestFairMarket?.Metrics.LogLoss;
                    finalRow.DevFairMarketBrier = bestFairMarket?.Metrics.Brier;
                    finalRow.DevBlendLogLoss = bestBlend?.Metrics.LogLoss;
                    finalRow.DevBlendBrier = bestBlend?.Metrics.Brier;
                    finalRow.DevBlendRaceLogLoss = bestBlend?.Metrics.RaceLogLoss;
                    finalRow.ValidationAccuracy = bestResult.ValidationAccuracy;
                    finalRow.ValidationLoss = bestResult.ValidationLoss;
                    finalRow.ValidationBrier = bestResult.ValidationBrier;
                    finalRow.TestAccuracy = testResult.ValidationAccuracy;
                    finalRow.TestLoss = testResult.ValidationLoss;
                    finalRow.TestBrier = testResult.ValidationBrier;
                    finalRow.TestTemperature = testResult.ValidationTemperature;
                    finalRow.TestMarketLogLoss = testMarketLogLoss;
                    finalRow.TestMarketBrier = testMarketBrier;
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
                    finalRow.IsActive = true;
                    finalRow.Id = _repository.InsertMLParameter(finalRow);
                    _repository.ActivateMLParameter(
                        finalRow.Id,
                        finalRow.ModelArtifact,
                        finalRow.ModelHash,
                        finalRow.FeatureSchemaHash);

                    var testPriced = HyperparameterTrainer.ComputePricedSubsetMetrics(testResult);
                    var blendReport = testBlend is not null && testFairMarket is not null
                        ? $" Test blend LL {testBlend.LogLoss:F4} vs fair market {testFairMarket.LogLoss:F4}."
                        : string.Empty;
                    Console.WriteLine(
                        $"[TrainAI] Persisted the best model ({bestModel.Layers} layers, {bestModel.Units} units, {testResult.Objective}, offset {testResult.MarketOffsetMode}, blind {testResult.MarketBlind}) trained on {trainDevRaceIds.Count} races. Dev Brier: {bestResult.ValidationBrier:F4} (closing market {bestDevMarketBrier:F4}); Test Brier: {testResult.ValidationBrier:F4} (closing market {testMarketBrier:F4}); Test log loss: {testResult.ValidationLoss:F4} (closing market {testMarketLogLoss:F4}).{blendReport} Test priced subset: acc {testPriced.Accuracy:P2}, loss {testPriced.LogLoss:F4}, Brier {testPriced.Brier:F4} over {testPriced.RaceCount} races.");
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
                months = 1;
            }
            else if (months > 24)
            {
                months = 24;
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
            var result = await Task.Run(() => _trainer.Train(
                parameter,
                0,
                1,
                dataset,
                persistWeights: false,
                temperatureOverride: temperatureOverride,
                earlyStopOnValidation: false,
                blendModelWeight: savedParameter.BlendModelWeight ?? 1d,
                blendMarketWeight: savedParameter.BlendMarketWeight ?? 0d));
            var (marketLogLoss, marketBrier) = ComputeMarketMetrics(result.ValidationExamples);

            double blendModelWeight = savedParameter.BlendModelWeight ?? 1d;
            double blendMarketWeight = savedParameter.BlendMarketWeight ?? 0d;
            var fairMarket = MarketBlendCalibration.Evaluate(
                result.ValidationPredictions,
                result.ValidationLabels,
                result.ValidationRaceIds,
                result.ValidationExamples,
                0d,
                blendMarketWeight);
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
            viewModel.ValidationTemperature = result.ValidationTemperature;
            viewModel.MarketLogLoss = marketLogLoss;
            viewModel.MarketBrier = marketBrier;
            viewModel.FairMarketLogLoss = fairMarket.LogLoss;
            viewModel.FairMarketBrier = fairMarket.Brier;
            viewModel.BlendLogLoss = blend.LogLoss;
            viewModel.BlendBrier = blend.Brier;
            viewModel.BlendModelWeight = blendModelWeight;
            viewModel.BlendMarketWeight = blendMarketWeight;
            viewModel.ValidationRaceCount = dataset.ValidationRaces.Count;
            viewModel.TrainingRaceCount = dataset.TrainingRaces.Count;
            var pricedSubset = HyperparameterTrainer.ComputePricedSubsetMetrics(result);
            var blendText = blend.RunnerCount > 0
                ? $" Blend log loss {blend.LogLoss:F4} vs fair market {fairMarket.LogLoss:F4}."
                : string.Empty;
            viewModel.Message = $"Validation accuracy over {viewModel.ValidationRaceCount} races: {result.ValidationAccuracy:P2}. Model Brier {result.ValidationBrier:F4} vs closing market {marketBrier:F4}.{blendText} Priced subset: accuracy {pricedSubset.Accuracy:P2}, log loss {pricedSubset.LogLoss:F4}, Brier {pricedSubset.Brier:F4} over {pricedSubset.RaceCount} races.";

            var blendedPredictions = MarketBlendCalibration.ApplyFullField(
                result.ValidationPredictions,
                result.ValidationRaceIds,
                result.ValidationExamples,
                blendModelWeight,
                blendMarketWeight);
            var raceSummaries = _repository.GetRaceSummaries(result.ValidationRaceIds);
            viewModel.Simulation = RunValidationSimulation(result, raceSummaries, startingBankroll, blendedPredictions);

            return View(viewModel);
        }

        /// <summary>
        /// The stored RaceType column is empty in the imported data, so derive a
        /// usable type from the race title for the ROI breakdowns.
        /// </summary>
        private static string ResolveRaceType(string? storedRaceType, string? title, string? surface)
        {
            if (!string.IsNullOrWhiteSpace(storedRaceType))
            {
                return storedRaceType!;
            }

            var value = title ?? string.Empty;
            if (value.Contains("Hurdle", StringComparison.OrdinalIgnoreCase))
            {
                return "Hurdle";
            }

            if (value.Contains("Chase", StringComparison.OrdinalIgnoreCase))
            {
                return "Chase";
            }

            if (value.Contains("NH Flat", StringComparison.OrdinalIgnoreCase) ||
                value.Contains("Flat Race", StringComparison.OrdinalIgnoreCase) ||
                value.Contains("Bumper", StringComparison.OrdinalIgnoreCase))
            {
                return "NH Flat";
            }

            if (value.Contains("Flat", StringComparison.OrdinalIgnoreCase) ||
                !string.IsNullOrWhiteSpace(surface))
            {
                return "Flat";
            }

            return "Unknown";
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

        private (double LogLoss, double Brier) ComputeMarketMetrics(
            IReadOnlyList<HyperparameterTrainer.RunnerExample>? examples)
        {
            if (examples is null || examples.Count == 0)
            {
                return (0d, 0d);
            }

            double logLoss = 0d;
            double brier = 0d;
            int count = 0;

            foreach (var race in examples.GroupBy(e => e.RaceId))
            {
                var entries = race
                    .Where(e => e.StartingPriceDecimal.HasValue && e.StartingPriceDecimal.Value > 1m)
                    .Select(e => new
                    {
                        e.Label,
                        Odds = BettingMath.CalculateEffectiveDecimalOdds(e.StartingPriceDecimal!.Value, _commissionRate)
                    })
                    .ToList();

                if (entries.Count == 0)
                {
                    continue;
                }

                double sum = entries.Sum(e => 1.0 / (double)e.Odds);
                if (sum <= double.Epsilon || !double.IsFinite(sum))
                {
                    continue;
                }

                foreach (var entry in entries)
                {
                    double q = (1.0 / (double)entry.Odds) / sum;
                    q = Math.Clamp(q, 1e-7, 1d - 1e-7);
                    double y = entry.Label;
                    logLoss -= y * Math.Log(q) + (1d - y) * Math.Log(1d - q);
                    brier += (q - y) * (q - y);
                    count++;
                }
            }

            return count == 0 ? (0d, 0d) : (logLoss / count, brier / count);
        }

        private ValidationSimulationResult RunValidationSimulation(
            HyperparameterTrainer.TrainingResult result,
            IDictionary<int, RaceSummary> raceSummaries,
            decimal startingBankroll,
            IReadOnlyList<float>? predictionsOverride = null)
        {
            raceSummaries ??= new Dictionary<int, RaceSummary>();

            var predictions = predictionsOverride ?? result.ValidationPredictions ?? Array.Empty<float>();
            var examples = result.ValidationExamples ?? Array.Empty<HyperparameterTrainer.RunnerExample>();
            var raceIds = result.ValidationRaceIds ?? Array.Empty<int>();

            if (predictions.Count == 0 || examples.Count == 0 || raceIds.Count == 0 ||
                predictions.Count != examples.Count || raceIds.Count != examples.Count)
            {
                return new ValidationSimulationResult
                {
                    StartingBankroll = startingBankroll,
                    EndingBankroll = startingBankroll,
                    TotalStaked = 0m,
                    BetCount = 0,
                    WinCount = 0,
                    Bets = Array.Empty<ValidationBetResult>()
                };
            }

            decimal bankroll = startingBankroll;
            decimal totalStaked = 0m;
            int wins = 0;
            var bets = new List<ValidationBetResult>();

            var grouped = Enumerable.Range(0, examples.Count)
                .Select(i => new
                {
                    RaceId = raceIds[i],
                    Example = examples[i],
                    Probability = Math.Clamp((double)predictions[i], 0d, 1d)
                })
                .GroupBy(x => x.RaceId);

            foreach (var raceGroup in grouped)
            {
                if (bankroll <= 0m)
                {
                    break;
                }

                var entries = raceGroup.ToList();
                if (entries.Any(e => !double.IsFinite(e.Probability) || e.Probability <= 0))
                {
                    continue;
                }

                var minProbability = entries.Min(e => e.Probability);
                var maxProbability = entries.Max(e => e.Probability);
                if (maxProbability - minProbability < 0.01)
                {
                    // Match the live normaliser: do not turn an uninformative
                    // near-uniform vector into longshot bets.
                    continue;
                }

                var probabilitySum = entries.Sum(e => e.Probability);
                if (probabilitySum <= double.Epsilon || !double.IsFinite(probabilitySum))
                {
                    continue;
                }

                var candidate = entries
                    .Select(entry =>
                    {
                        if (!entry.Example.StartingPriceDecimal.HasValue || entry.Example.StartingPriceDecimal.Value <= 1m)
                        {
                            return null;
                        }

                        var decimalOdds = entry.Example.StartingPriceDecimal.Value;
                        var effectiveOdds = BettingMath.CalculateEffectiveDecimalOdds(decimalOdds, _commissionRate);
                        if (effectiveOdds <= 1m)
                        {
                            return null;
                        }

                        var normalizedProbability = entry.Probability / probabilitySum;
                        var marketProbability = 1.0 / (double)effectiveOdds;
                        if (!double.IsFinite(marketProbability) || marketProbability <= 0)
                        {
                            return null;
                        }

                        var differential = normalizedProbability - marketProbability;
                        return new
                        {
                            entry.RaceId,
                            entry.Example,
                            Probability = normalizedProbability,
                            DecimalOdds = decimalOdds,
                            EffectiveDecimalOdds = effectiveOdds,
                            MarketProbability = marketProbability,
                            Differential = differential
                        };
                    })
                    .Where(c => c != null && c.Differential > 0)
                    .Select(c => c!)
                    .OrderByDescending(c => c.Differential)
                    .ThenByDescending(c => c.Probability)
                    .FirstOrDefault();

                if (candidate is null)
                {
                    continue;
                }

                var kellyFraction = BettingMath.CalculateKellyFraction(candidate.Probability, (double)candidate.EffectiveDecimalOdds, _maxKellyFraction);
                if (kellyFraction <= 0m)
                {
                    continue;
                }

                var stake = BettingMath.CalculateSequentialStake(bankroll, kellyFraction);
                if (stake <= 0m)
                {
                    continue;
                }

                bankroll -= stake;
                totalStaked += stake;

                bool won = candidate.Example.Label >= 0.5f;
                if (won)
                {
                    wins++;
                    bankroll += stake * candidate.EffectiveDecimalOdds;
                }

                raceSummaries.TryGetValue(candidate.RaceId, out var summary);
                bets.Add(new ValidationBetResult
                {
                    RaceId = candidate.RaceId,
                    RaceDate = summary?.RaceDate,
                    RaceTitle = summary?.Title,
                    CourseName = summary?.CourseName,
                    HorseName = candidate.Example.HorseName,
                    RaceType = ResolveRaceType(summary?.RaceType, summary?.Title, summary?.Surface),
                    Class = summary?.Class,
                    AgeRestriction = summary?.AgeRestriction,
                    Surface = summary?.Surface,
                    Going = summary?.Going,
                    DistanceYards = summary?.DistanceYards,
                    DistanceText = summary?.DistanceText,
                    RunnerCount = summary?.RunnerCount ?? (byte?)entries.Count,
                    DecimalOdds = candidate.DecimalOdds,
                    EffectiveDecimalOdds = candidate.EffectiveDecimalOdds,
                    CommissionRate = _commissionRate,
                    AiDecimalOdds = BettingMath.CalculateAiDecimalOdds(candidate.Probability),
                    AiProbability = candidate.Probability,
                    MarketProbability = candidate.MarketProbability,
                    Differential = candidate.Differential,
                    Stake = stake,
                    Won = won,
                    BankrollAfter = bankroll
                });
            }

            return new ValidationSimulationResult
            {
                StartingBankroll = startingBankroll,
                EndingBankroll = bankroll,
                TotalStaked = totalStaked,
                BetCount = bets.Count,
                WinCount = wins,
                Bets = bets
            };
        }
    }
}
