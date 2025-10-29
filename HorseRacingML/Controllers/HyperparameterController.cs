using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.ML;
using HorseRacingML.Services;
using System.Globalization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Controllers
{
    public class HyperparameterController : Controller
    {
        private readonly RacingRepository _repository;
        private readonly HyperparameterTrainer _trainer;

        private readonly decimal? _maxKellyFraction;

        public HyperparameterController(RacingRepository repository, HyperparameterTrainer trainer, IConfiguration configuration)
        {
            _repository = repository;
            _trainer = trainer;
            _maxKellyFraction = configuration.GetValue<decimal?>("Betting:MaxKellyFraction");
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
        public async Task<IActionResult> Custom(MLParameterBatch batch)
        {
            if (!ModelState.IsValid)
            {
                return View(batch);
            }

            await Task.Run(() =>
            {
                var parameters = batch.Parameters ?? new List<MLParameter>();
                Console.WriteLine($"[Hyperparameter] Starting custom run for {parameters.Count} parameter set(s).");

                var dataset = _trainer.PrepareDataset();
                Console.WriteLine($"[Hyperparameter] Dataset prepared with {dataset.Races.Count} races and {dataset.RowCount} runner rows.");

                int modelIndex = 0;
                foreach (var model in parameters)
                {
                    modelIndex++;
                    Console.WriteLine($"[Hyperparameter] Starting model {modelIndex}/{parameters.Count} (layers: {model.Layers}, units: {model.Units}, dropout: {model.Dropout}, lr: {model.LearningRate}).");

                    if (model.Folds <= 0)
                    {
                        Console.WriteLine($"[Hyperparameter] Skipped storing results for model {modelIndex} due to invalid fold count {model.Folds}.");
                        continue;
                    }

                    model.RunDate = DateTime.UtcNow;

                    double totalTrainAccuracy = 0;
                    double totalTrainLoss = 0;
                    double totalTrainBrier = 0;
                    double totalValidationAccuracy = 0;
                    double totalValidationLoss = 0;
                    double totalValidationBrier = 0;

                    for (int i = 0; i < model.Folds; i++)
                    {
                        Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} - training in progress...");
                        var result = _trainer.Train(model, dataset, i, model.Folds, persistWeights: false);
                        Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");

                        totalTrainAccuracy += result.TrainAccuracy;
                        totalTrainLoss += result.TrainLoss;
                        totalTrainBrier += result.TrainBrier;
                        totalValidationAccuracy += result.ValidationAccuracy;
                        totalValidationLoss += result.ValidationLoss;
                        totalValidationBrier += result.ValidationBrier;

                        if (model.Folds > 1)
                        {
                            var foldModel = new MLParameter
                            {
                                RunDate = model.RunDate,
                                Units = model.Units,
                                Dropout = model.Dropout,
                                Layers = model.Layers,
                                LearningRate = model.LearningRate,
                                Epochs = model.Epochs,
                                BatchSize = model.BatchSize,
                                Folds = model.Folds,
                                Fold = i + 1,
                                TrainAccuracy = result.TrainAccuracy,
                                TrainLoss = result.TrainLoss,
                                TrainBrier = result.TrainBrier,
                                ValidationAccuracy = result.ValidationAccuracy,
                                ValidationLoss = result.ValidationLoss,
                                ValidationBrier = result.ValidationBrier
                            };

                            _repository.InsertMLParameter(foldModel);
                        }
                    }

                    var averagedModel = new MLParameter
                    {
                        RunDate = model.RunDate,
                        Units = model.Units,
                        Dropout = model.Dropout,
                        Layers = model.Layers,
                        LearningRate = model.LearningRate,
                        Epochs = model.Epochs,
                        BatchSize = model.BatchSize,
                        Folds = model.Folds,
                        Fold = null,
                        TrainAccuracy = totalTrainAccuracy / model.Folds,
                        TrainLoss = totalTrainLoss / model.Folds,
                        TrainBrier = totalTrainBrier / model.Folds,
                        ValidationAccuracy = totalValidationAccuracy / model.Folds,
                        ValidationLoss = totalValidationLoss / model.Folds,
                        ValidationBrier = totalValidationBrier / model.Folds
                    };

                    _repository.InsertMLParameter(averagedModel);

                    Console.WriteLine($"[Hyperparameter] Completed model {modelIndex}/{parameters.Count}.");
                }

                Console.WriteLine("[Hyperparameter] Custom run finished.");
            });

            return RedirectToAction("Index", "Home");
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
        public async Task<IActionResult> TrainAI(MLParameterBatch batch)
        {
            if (!ModelState.IsValid)
            {
                return View(batch);
            }

            await Task.Run(() =>
            {
                var parameters = batch.Parameters ?? new List<MLParameter>();
                Console.WriteLine($"[TrainAI] Starting training run for {parameters.Count} parameter set(s).");

                var dataset = _trainer.LoadTrainingDataset();
                Console.WriteLine($"[TrainAI] Dataset loaded with {dataset.TrainingRaces.Count} races covering {dataset.TrainingRaces.Sum(r => r.Runners.Count)} runner rows.");

                int modelIndex = 0;
                foreach (var model in parameters)
                {
                    modelIndex++;
                    Console.WriteLine($"[TrainAI] Starting model {modelIndex}/{parameters.Count} (layers: {model.Layers}, units: {model.Units}, dropout: {model.Dropout}, lr: {model.LearningRate}).");

                    model.RunDate = DateTime.UtcNow;
                    model.Folds = 1;
                    model.Fold = null;

                    var result = _trainer.Train(model, 0, 1, dataset, persistWeights: true);

                    model.TrainAccuracy = result.TrainAccuracy;
                    model.TrainLoss = result.TrainLoss;
                    model.TrainBrier = result.TrainBrier;
                    model.ValidationAccuracy = result.ValidationAccuracy;
                    model.ValidationLoss = result.ValidationLoss;
                    model.ValidationBrier = result.ValidationBrier;

                    _repository.InsertMLParameter(model);

                    Console.WriteLine($"[TrainAI] Completed model {modelIndex}/{parameters.Count}. Train acc: {result.TrainAccuracy:F4}, train loss: {result.TrainLoss:F4}.");
                }

                Console.WriteLine("[TrainAI] Training run finished.");
            });

            return RedirectToAction("Index", "Home");
        }

        [HttpGet]
        public IActionResult TestAI(
           string? units,
           string? dropout,
           string? layers,
           string? learningRate,
           string? epochs,
           string? batchSize,
           string? folds)
        {
            static int? ParseNullableInt(string? value)
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    return null;
                }

                if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                {
                    return parsed;
                }

                if (int.TryParse(value, NumberStyles.Integer, CultureInfo.CurrentCulture, out parsed))
                {
                    return parsed;
                }

                return null;
            }

            static double? ParseNullableDouble(string? value)
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    return null;
                }

                const NumberStyles style = NumberStyles.Float | NumberStyles.AllowThousands;
                if (double.TryParse(value, style, CultureInfo.InvariantCulture, out var parsed))
                {
                    return parsed;
                }

                if (double.TryParse(value, style, CultureInfo.CurrentCulture, out parsed))
                {
                    return parsed;
                }

                return null;
            }

            return View(new AITestResultViewModel
            {
                ValidationStart = DateTime.Today.AddMonths(-1),
                ValidationEnd = DateTime.Today,
                SelectedValidationMonths = 1,
                StartingBankroll = 100m,
                RequestedUnits = ParseNullableInt(units),
                RequestedDropout = ParseNullableDouble(dropout),
                RequestedLayers = ParseNullableInt(layers),
                RequestedLearningRate = ParseNullableDouble(learningRate),
                RequestedEpochs = ParseNullableInt(epochs),
                RequestedBatchSize = ParseNullableInt(batchSize),
                RequestedFolds = ParseNullableInt(folds)
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestAI(AITestResultViewModel request)
        {
            int? ReadRequestedInt(int? current, string key)
            {
                if (current.HasValue)
                {
                    return current;
                }

                var raw = Request.Form[key];
                if (string.IsNullOrWhiteSpace(raw))
                {
                    return null;
                }

                if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedInt))
                {
                    return parsedInt;
                }

                if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.CurrentCulture, out parsedInt))
                {
                    return parsedInt;
                }

                return null;
            }

            double? ReadRequestedDouble(double? current, string key)
            {
                if (current.HasValue)
                {
                    return current;
                }

                var raw = Request.Form[key];
                if (string.IsNullOrWhiteSpace(raw))
                {
                    return null;
                }

                const NumberStyles style = NumberStyles.Float | NumberStyles.AllowThousands;
                if (double.TryParse(raw, style, CultureInfo.InvariantCulture, out var parsedDouble))
                {
                    return parsedDouble;
                }

                if (double.TryParse(raw, style, CultureInfo.CurrentCulture, out parsedDouble))
                {
                    return parsedDouble;
                }

                return null;
            }

            var requestedUnits = ReadRequestedInt(request.RequestedUnits, nameof(request.RequestedUnits));
            var requestedDropout = ReadRequestedDouble(request.RequestedDropout, nameof(request.RequestedDropout));
            var requestedLayers = ReadRequestedInt(request.RequestedLayers, nameof(request.RequestedLayers));
            var requestedLearningRate = ReadRequestedDouble(request.RequestedLearningRate, nameof(request.RequestedLearningRate));
            var requestedEpochs = ReadRequestedInt(request.RequestedEpochs, nameof(request.RequestedEpochs));
            var requestedBatchSize = ReadRequestedInt(request.RequestedBatchSize, nameof(request.RequestedBatchSize));
            var requestedFolds = ReadRequestedInt(request.RequestedFolds, nameof(request.RequestedFolds));

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

            viewModel.RequestedUnits = requestedUnits;
            viewModel.RequestedDropout = requestedDropout;
            viewModel.RequestedLayers = requestedLayers;
            viewModel.RequestedLearningRate = requestedLearningRate;
            viewModel.RequestedEpochs = requestedEpochs;
            viewModel.RequestedBatchSize = requestedBatchSize;
            viewModel.RequestedFolds = requestedFolds;

            bool hasRequestedHyperparameters =
                requestedUnits.HasValue ||
                requestedDropout.HasValue ||
                requestedLayers.HasValue ||
                requestedLearningRate.HasValue ||
                requestedEpochs.HasValue ||
                requestedBatchSize.HasValue ||
                requestedFolds.HasValue;

            var savedParameter = _repository.GetMostRecentMLParameter();
            if (savedParameter is null && !hasRequestedHyperparameters)
            {
                viewModel.Message = "No saved AI parameters were found. Please train the AI before running a test.";
                return View(viewModel);
            }

            if (savedParameter is null)
            {
                savedParameter = new MLParameter
                {
                    RunDate = DateTime.UtcNow
                };
            }

            var validationRaceIds = _repository.GetRaceIdsBetweenDates(validationStart, validationEnd);
            if (validationRaceIds.Count == 0)
            {
                viewModel.Message = $"No races were found in the last {months} {monthLabel} to use for validation.";
                return View(viewModel);
            }

            var trainingRaceIds = _repository.GetRaceIdsOutsideRange(validationStart, validationEnd);
            if (trainingRaceIds.Count == 0)
            {
                viewModel.Message = "No training data is available outside the validation window.";
                return View(viewModel);
            }
            var dataset = _trainer.LoadTrainingDataset(new HashSet<int>(trainingRaceIds), new HashSet<int>(validationRaceIds), includeIdentifiers: true);
            if (dataset.TrainingRaces.Count == 0)
            {
                viewModel.Message = "The training dataset was empty after preparation.";
                return View(viewModel);
            }

            var persistedHyperparameters = _trainer.LoadPersistedHyperparameters();
            static bool HasValidUnits(int units) => MLParameterValidator.EnsureUnits(units, int.MinValue) == units;
            static bool HasValidDropout(double dropout) => !double.IsNaN(MLParameterValidator.EnsureDropout(dropout, double.NaN));
            static bool HasValidLayers(int layers) => MLParameterValidator.EnsureLayers(layers, int.MinValue) == layers;
            static bool HasValidLearningRate(double learningRate) => !double.IsNaN(MLParameterValidator.EnsureLearningRate(learningRate, double.NaN));
            static bool HasValidPositive(int value) => MLParameterValidator.EnsurePositive(value, int.MinValue) == value;

            int ResolveUnitsFallback()
            {
                if (persistedHyperparameters != null && HasValidUnits(persistedHyperparameters.Units))
                {
                    return persistedHyperparameters.Units;
                }

                return 1;
            }

            double ResolveDropoutFallback()
            {
                if (persistedHyperparameters != null && HasValidDropout(persistedHyperparameters.Dropout))
                {
                    return persistedHyperparameters.Dropout;
                }

                return 0.1d;
            }

            int ResolveLayersFallback()
            {
                if (persistedHyperparameters != null && HasValidLayers(persistedHyperparameters.Layers))
                {
                    return persistedHyperparameters.Layers;
                }

                return 1;
            }

            double ResolveLearningRateFallback()
            {
                if (persistedHyperparameters != null && HasValidLearningRate(persistedHyperparameters.LearningRate))
                {
                    return persistedHyperparameters.LearningRate;
                }

                return 0.005d;
            }

            int ResolveEpochsFallback()
            {
                if (persistedHyperparameters != null && HasValidPositive(persistedHyperparameters.Epochs))
                {
                    return persistedHyperparameters.Epochs;
                }

                return 1;
            }

            int ResolveBatchSizeFallback()
            {
                if (persistedHyperparameters != null && HasValidPositive(persistedHyperparameters.BatchSize))
                {
                    return persistedHyperparameters.BatchSize;
                }

                return 1;
            }

            int ResolveFoldsFallback()
            {
                if (persistedHyperparameters != null && HasValidPositive(persistedHyperparameters.Folds))
                {
                    return persistedHyperparameters.Folds;
                }

                return 1;
            }

            var units = requestedUnits.HasValue && HasValidUnits(requestedUnits.Value)
                ? requestedUnits.Value
                : HasValidUnits(savedParameter.Units)
                    ? savedParameter.Units
                    : ResolveUnitsFallback();
            var dropout = requestedDropout.HasValue && HasValidDropout(requestedDropout.Value)
                ? requestedDropout.Value
                : HasValidDropout(savedParameter.Dropout)
                    ? savedParameter.Dropout
                    : ResolveDropoutFallback();
            var layers = requestedLayers.HasValue && HasValidLayers(requestedLayers.Value)
                ? requestedLayers.Value
                : HasValidLayers(savedParameter.Layers)
                    ? savedParameter.Layers
                    : ResolveLayersFallback();
            var learningRate = requestedLearningRate.HasValue && HasValidLearningRate(requestedLearningRate.Value)
                ? requestedLearningRate.Value
                : HasValidLearningRate(savedParameter.LearningRate)
                    ? savedParameter.LearningRate
                    : ResolveLearningRateFallback();
            var epochs = requestedEpochs.HasValue && HasValidPositive(requestedEpochs.Value)
                ? requestedEpochs.Value
                : HasValidPositive(savedParameter.Epochs)
                    ? savedParameter.Epochs
                    : ResolveEpochsFallback();
            var batchSize = requestedBatchSize.HasValue && HasValidPositive(requestedBatchSize.Value)
                ? requestedBatchSize.Value
                : HasValidPositive(savedParameter.BatchSize)
                    ? savedParameter.BatchSize
                    : ResolveBatchSizeFallback();
            var folds = requestedFolds.HasValue && HasValidPositive(requestedFolds.Value)
                ? requestedFolds.Value
                : HasValidPositive(savedParameter.Folds)
                    ? savedParameter.Folds
                    : ResolveFoldsFallback();
            var parameter = new MLParameter
            {
                Units = units,
                Dropout = dropout,
                Layers = layers,
                LearningRate = learningRate,
                Epochs = epochs,
                BatchSize = batchSize,
                Folds = folds,
                Fold = null,
                RunDate = savedParameter.RunDate
            };

            var result = await Task.Run(() => _trainer.Train(parameter, 0, 1, dataset, persistWeights: false));

            savedParameter.Units = parameter.Units;
            savedParameter.Dropout = parameter.Dropout;
            savedParameter.Layers = parameter.Layers;
            savedParameter.LearningRate = parameter.LearningRate;
            savedParameter.Epochs = parameter.Epochs;
            savedParameter.BatchSize = parameter.BatchSize;
            savedParameter.Folds = parameter.Folds;
            savedParameter.Fold = parameter.Fold;
            viewModel.ParameterUsed = new MLParameter
            {
                RunDate = parameter.RunDate,
                Units = parameter.Units,
                Dropout = parameter.Dropout,
                Layers = parameter.Layers,
                LearningRate = parameter.LearningRate,
                Epochs = parameter.Epochs,
                BatchSize = parameter.BatchSize,
                Folds = parameter.Folds,
                Fold = parameter.Fold
            };
            viewModel.TrainAccuracy = result.TrainAccuracy;
            viewModel.TrainLoss = result.TrainLoss;
            viewModel.TrainBrier = result.TrainBrier;
            viewModel.ValidationAccuracy = result.ValidationAccuracy;
            viewModel.ValidationLoss = result.ValidationLoss;
            viewModel.ValidationBrier = result.ValidationBrier;
            viewModel.ValidationRaceCount = dataset.ValidationRaces.Count;
            viewModel.TrainingRaceCount = dataset.TrainingRaces.Count;
            viewModel.Message = $"Validation accuracy over {viewModel.ValidationRaceCount} races: {result.ValidationAccuracy:P2}.";

            var raceSummaries = _repository.GetRaceSummaries(result.ValidationRaceIds);
            viewModel.Simulation = RunValidationSimulation(result, raceSummaries, startingBankroll);

            return View(viewModel);
        }
        private ValidationSimulationResult RunValidationSimulation(
            HyperparameterTrainer.TrainingResult result,
            IDictionary<int, RaceSummary> raceSummaries,
            decimal startingBankroll)
        {
            raceSummaries ??= new Dictionary<int, RaceSummary>();

            var predictions = result.ValidationPredictions ?? Array.Empty<float>();
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

                var candidate = raceGroup
                    .Select(entry =>
                    {
                        if (!entry.Example.StartingPriceDecimal.HasValue || entry.Example.StartingPriceDecimal.Value <= 1m)
                        {
                            return null;
                        }

                        var decimalOdds = entry.Example.StartingPriceDecimal.Value;
                        if (decimalOdds <= 1m)
                        {
                            return null;
                        }

                        var marketProbability = 1.0 / (double)decimalOdds;
                        if (!double.IsFinite(marketProbability) || marketProbability <= 0)
                        {
                            return null;
                        }

                        var differential = entry.Probability - marketProbability;
                        return new
                        {
                            entry.RaceId,
                            entry.Example,
                            entry.Probability,
                            DecimalOdds = decimalOdds,
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

                var kellyFraction = BettingMath.CalculateKellyFraction(candidate.Probability, (double)candidate.DecimalOdds, _maxKellyFraction);
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
                    bankroll += stake * candidate.DecimalOdds;
                }

                raceSummaries.TryGetValue(candidate.RaceId, out var summary);
                bets.Add(new ValidationBetResult
                {
                    RaceId = candidate.RaceId,
                    RaceDate = summary?.RaceDate,
                    RaceTitle = summary?.Title,
                    CourseName = summary?.CourseName,
                    HorseName = candidate.Example.HorseName,
                    DecimalOdds = candidate.DecimalOdds,
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
