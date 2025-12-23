using CsvHelper;
using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Services;
using HorseRacingML.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using static HorseRacingML.ML.HyperparameterTrainer;

namespace HorseRacingML.Controllers
{
    public class HyperparameterController : Controller
    {
        private readonly RacingRepository _repository;
        private readonly HyperparameterTrainer _trainer;
        private readonly AIOddsCalculator _aiOddsCalculator;

        private readonly decimal? _maxKellyFraction;

        public HyperparameterController(RacingRepository repository, HyperparameterTrainer trainer, IConfiguration configuration, AIOddsCalculator aiOddsCalculator)
        {
            _repository = repository;
            _trainer = trainer;
            _aiOddsCalculator = aiOddsCalculator;
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
            // Retained for backward compatibility if standard form submit is used,
            // though the new JS queue will use TrainCustomModel.
            if (!ModelState.IsValid)
            {
                return View(batch);
            }

            await Task.Run(() =>
            {
                var parameters = batch.Parameters ?? new List<MLParameter>();
                Console.WriteLine($"[Hyperparameter] Starting custom run for {parameters.Count} parameter set(s).");

                var masterDataset = _trainer.EnsureMasterDatasetLoaded();

                var mlContext = _trainer.CreateLightGbmContext();

                int modelIndex = 0;
                foreach (var model in parameters)
                {
                    modelIndex++;
                    Console.WriteLine($"[Hyperparameter] Starting model {modelIndex}/{parameters.Count} (ModelType: {model.ModelType}).");

                    var foldCount = model.Folds;
                    if (foldCount <= 0)
                    {
                        foldCount = 1;
                        Console.WriteLine($"[Hyperparameter]  Fold count was not specified or invalid. Defaulting to {foldCount} for model {modelIndex}.");
                    }

                    model.Folds = foldCount;
                    model.RunDate = DateTime.UtcNow;
                    if (model.TrainFinalFoldOnly && model.Folds > 0)
                    {
                        var lastFoldIndex = model.Folds - 1;
                        Console.WriteLine($"[Hyperparameter]  Training ONLY final fold {lastFoldIndex + 1}/{model.Folds}...");

                        HyperparameterTrainer.TrainingResult result;

                        if (model.ModelType == 2)
                        {
                            // Hybrid: Use the high-level Train method which delegates to TrainHybrid
                            result = _trainer.Train(model, lastFoldIndex, model.Folds, masterDataset, persistWeights: false);
                        }
                        else
                        {
                            var preparedData = _trainer.GetPreparedFold(lastFoldIndex, model.Folds, model.ModelType, mlContext);

                            if (model.ModelType == 1)
                            {
                                result = _trainer.TrainLightGbmOptimized(model, mlContext, (HyperparameterTrainer.LightGbmFoldData)preparedData, persistWeights: false);
                            }
                            else
                            {
                                result = _trainer.TrainTensorFlowOptimized(model, (HyperparameterTrainer.TensorFlowFoldData)preparedData, persistWeights: false);
                            }
                        }

                        model.TrainAccuracy = result.TrainAccuracy;
                        model.TrainLoss = result.TrainLoss;
                        model.TrainBrier = result.TrainBrier;
                        model.TrainFocalLoss = result.TrainFocalLoss;
                        model.ValidationAccuracy = result.ValidationAccuracy;
                        model.ValidationLoss = result.ValidationLoss;
                        model.ValidationBrier = result.ValidationBrier;
                        model.ValidationFocalLoss = result.ValidationFocalLoss;
                        model.Fold = model.Folds;

                        if (result.TrainingRaceIds.Count > 0)
                        {
                            _repository.InsertMLParameter(model);
                            Console.WriteLine($"[Hyperparameter]  Final fold complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");
                        }
                        else
                        {
                            Console.WriteLine($"[Hyperparameter]  Final fold skipped due to empty training set.");
                        }
                    }
                    else
                    {
                        double totalTrainAccuracy = 0;
                        double totalTrainLoss = 0;
                        double totalTrainBrier = 0;
                        double totalTrainFocalLoss = 0;
                        double totalValidationAccuracy = 0;
                        double totalValidationLoss = 0;
                        double totalValidationBrier = 0;
                        double totalValidationFocalLoss = 0;

                        int validFolds = 0;

                        for (int i = 0; i < model.Folds; i++)
                        {
                            Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} - training in progress...");

                            HyperparameterTrainer.TrainingResult result;

                            if (model.ModelType == 2)
                            {
                                result = _trainer.Train(model, i, model.Folds, masterDataset, persistWeights: false);
                            }
                            else
                            {
                                var preparedData = _trainer.GetPreparedFold(i, model.Folds, model.ModelType, mlContext);

                                if (model.ModelType == 1)
                                {
                                    result = _trainer.TrainLightGbmOptimized(model, mlContext, (HyperparameterTrainer.LightGbmFoldData)preparedData, persistWeights: false);
                                }
                                else
                                {
                                    result = _trainer.TrainTensorFlowOptimized(model, (HyperparameterTrainer.TensorFlowFoldData)preparedData, persistWeights: false);
                                }
                            }

                            if (result.TrainingRaceIds.Count == 0)
                            {
                                Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} skipped due to empty training set.");
                                continue;
                            }

                            validFolds++;
                            Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");

                            totalTrainAccuracy += result.TrainAccuracy;
                            totalTrainLoss += result.TrainLoss;
                            totalTrainBrier += result.TrainBrier;
                            totalTrainFocalLoss += result.TrainFocalLoss;
                            totalValidationAccuracy += result.ValidationAccuracy;
                            totalValidationLoss += result.ValidationLoss;
                            totalValidationBrier += result.ValidationBrier;
                            totalValidationFocalLoss += result.ValidationFocalLoss;
                        }

                        if (validFolds > 0)
                        {
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
                                TrainAccuracy = totalTrainAccuracy / validFolds,
                                TrainLoss = totalTrainLoss / validFolds,
                                TrainBrier = totalTrainBrier / validFolds,
                                TrainFocalLoss = totalTrainFocalLoss / validFolds,
                                ValidationAccuracy = totalValidationAccuracy / validFolds,
                                ValidationLoss = totalValidationLoss / validFolds,
                                ValidationBrier = totalValidationBrier / validFolds,
                                ValidationFocalLoss = totalValidationFocalLoss / validFolds
                            };

                            _repository.InsertMLParameter(averagedModel);
                        }
                    }

                    Console.WriteLine($"[Hyperparameter] Completed model {modelIndex}/{parameters.Count}.");
                }

                Console.WriteLine("[Hyperparameter] Custom run finished.");

            });

            return RedirectToAction("Index", "Home");
        }

        [HttpPost]
        public async Task<IActionResult> TrainCustomModel([FromBody] MLParameter model)
        {
            if (model == null)
            {
                return BadRequest("Invalid model parameters.");
            }

            try
            {
                await Task.Run(() =>
                {
                    // Ensure dataset is loaded and cached
                    var masterDataset = _trainer.EnsureMasterDatasetLoaded();

                    if (model.ModelType == 1)
                    {
                        Console.WriteLine($"[Hyperparameter] Starting custom LightGBM model training (leaves: {model.LgbmLeaves}, min data: {model.LgbmMinDataInLeaf}, max depth: {model.LgbmMaxDepth}, lr: {model.LearningRate}).");
                    }
                    else if (model.ModelType == 2)
                    {
                        Console.WriteLine($"[Hyperparameter] Starting custom Hybrid (NN + LGBM) training.");
                    }
                    else
                    {
                        Console.WriteLine($"[Hyperparameter] Starting custom Neural Network model training (layers: {model.Layers}, units: {model.Units}, dropout: {model.Dropout}, lr: {model.LearningRate}).");
                    }

                    var foldCount = model.Folds;
                    if (foldCount <= 0)
                    {
                        foldCount = 1;
                        Console.WriteLine($"[Hyperparameter] Fold count was not specified or invalid. Defaulting to {foldCount}.");
                    }
                    model.Folds = foldCount;
                    model.RunDate = DateTime.UtcNow;

                    var mlContext = _trainer.CreateLightGbmContext();
                    // We don't cache LightGBM fold data across HTTP requests for now, as it adds complexity.
                    // Each request rebuilds the fold data if needed.

                    if (model.TrainFinalFoldOnly && model.Folds > 0)
                    {
                        var lastFoldIndex = model.Folds - 1;
                        Console.WriteLine($"[Hyperparameter] Training ONLY final fold {lastFoldIndex + 1}/{model.Folds}...");

                        HyperparameterTrainer.TrainingResult result;

                        if (model.ModelType == 2)
                        {
                            // Hybrid model
                            result = _trainer.Train(model, lastFoldIndex, model.Folds, masterDataset, persistWeights: false);
                        }
                        else
                        {
                            var preparedData = _trainer.GetPreparedFold(lastFoldIndex, model.Folds, model.ModelType, mlContext);

                            if (model.ModelType == 1)
                            {
                                result = _trainer.TrainLightGbmOptimized(model, mlContext, (HyperparameterTrainer.LightGbmFoldData)preparedData, persistWeights: false);
                            }
                            else
                            {
                                result = _trainer.TrainTensorFlowOptimized(model, (HyperparameterTrainer.TensorFlowFoldData)preparedData, persistWeights: false);
                            }
                        }

                        model.TrainAccuracy = result.TrainAccuracy;
                        model.TrainLoss = result.TrainLoss;
                        model.TrainBrier = result.TrainBrier;
                        model.TrainFocalLoss = result.TrainFocalLoss;
                        model.ValidationAccuracy = result.ValidationAccuracy;
                        model.ValidationLoss = result.ValidationLoss;
                        model.ValidationBrier = result.ValidationBrier;
                        model.ValidationFocalLoss = result.ValidationFocalLoss;
                        model.Fold = model.Folds;

                        if (result.TrainingRaceIds.Count > 0)
                        {
                            _repository.InsertMLParameter(model);
                            Console.WriteLine($"[Hyperparameter] Final fold complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");
                        }
                        else
                        {
                            Console.WriteLine($"[Hyperparameter] Final fold skipped due to empty training set.");
                        }
                    }
                    else
                    {
                        double totalTrainAccuracy = 0;
                        double totalTrainLoss = 0;
                        double totalTrainBrier = 0;
                        double totalTrainFocalLoss = 0;
                        double totalValidationAccuracy = 0;
                        double totalValidationLoss = 0;
                        double totalValidationBrier = 0;
                        double totalValidationFocalLoss = 0;

                        int validFolds = 0;

                        for (int i = 0; i < model.Folds; i++)
                        {
                            Console.WriteLine($"[Hyperparameter] Fold {i + 1}/{model.Folds} - training in progress...");

                            HyperparameterTrainer.TrainingResult result;

                            if (model.ModelType == 2)
                            {
                                result = _trainer.Train(model, i, model.Folds, masterDataset, persistWeights: false);
                            }
                            else
                            {
                                var preparedData = _trainer.GetPreparedFold(i, model.Folds, model.ModelType, mlContext);

                                if (model.ModelType == 1)
                                {
                                    result = _trainer.TrainLightGbmOptimized(model, mlContext, (HyperparameterTrainer.LightGbmFoldData)preparedData, persistWeights: false);
                                }
                                else
                                {
                                    result = _trainer.TrainTensorFlowOptimized(model, (HyperparameterTrainer.TensorFlowFoldData)preparedData, persistWeights: false);
                                }
                            }

                            if (result.TrainingRaceIds.Count == 0)
                            {
                                Console.WriteLine($"[Hyperparameter] Fold {i + 1}/{model.Folds} skipped due to empty training set.");
                                continue;
                            }

                            validFolds++;
                            Console.WriteLine($"[Hyperparameter] Fold {i + 1}/{model.Folds} complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");

                            totalTrainAccuracy += result.TrainAccuracy;
                            totalTrainLoss += result.TrainLoss;
                            totalTrainBrier += result.TrainBrier;
                            totalTrainFocalLoss += result.TrainFocalLoss;
                            totalValidationAccuracy += result.ValidationAccuracy;
                            totalValidationLoss += result.ValidationLoss;
                            totalValidationBrier += result.ValidationBrier;
                            totalValidationFocalLoss += result.ValidationFocalLoss;
                        }

                        if (validFolds > 0)
                        {
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
                                TrainAccuracy = totalTrainAccuracy / validFolds,
                                TrainLoss = totalTrainLoss / validFolds,
                                TrainBrier = totalTrainBrier / validFolds,
                                TrainFocalLoss = totalTrainFocalLoss / validFolds,
                                ValidationAccuracy = totalValidationAccuracy / validFolds,
                                ValidationLoss = totalValidationLoss / validFolds,
                                ValidationBrier = totalValidationBrier / validFolds,
                                ValidationFocalLoss = totalValidationFocalLoss / validFolds
                            };

                            _repository.InsertMLParameter(averagedModel);
                        }
                    }
                });

                return Json(new { success = true });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Hyperparameter] Error training custom model: {ex}");
                return Json(new { success = false, message = ex.Message });
            }
        }
        [HttpGet]
        public IActionResult TrainAI()
        {
            return View(new TrainAIViewModel
            {
                Batch = new MLParameterBatch
                {
                    Parameters = new List<MLParameter>
                    {
                        new MLParameter { RunDate = DateTime.UtcNow }
                    }
                }
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TrainAI(TrainAIViewModel viewModel)
        {
            if (viewModel is null)
            {
                throw new ArgumentNullException(nameof(viewModel));
            }

            viewModel.Batch ??= new MLParameterBatch();
            var batch = viewModel.Batch;
            batch.Parameters ??= new List<MLParameter>();

            if (!ModelState.IsValid)
            {
                return View(viewModel);
            }

            var results = new List<TrainAIViewModel.TrainAIModelResult>();

            await Task.Run(() =>
            {
                var parameters = batch.Parameters;
                Console.WriteLine($"[TrainAI] Starting training run for {parameters.Count} parameter set(s).");

                var dataset = _trainer.LoadTrainingDataset(includeIdentifiers: true);
                Console.WriteLine($"[TrainAI] Dataset loaded with {dataset.TrainingRaces.Count} races covering {dataset.TrainingRaces.Sum(r => r.Runners.Count)} runner rows.");

                int modelIndex = 0;
                foreach (var model in parameters)
                {
                    modelIndex++;
                    Console.WriteLine($"[TrainAI] Starting model {modelIndex}/{parameters.Count} (ModelType: {model.ModelType}).");

                    model.RunDate = DateTime.UtcNow;
                    model.Folds = 1;
                    model.Fold = null;

                    var mlContext = _trainer.CreateLightGbmContext();
                    HyperparameterTrainer.TrainingResult result;

                    if (model.ModelType == 2)
                    {
                        // Hybrid: use main Train method to handle splitting and saving
                        result = _trainer.Train(model, 0, 1, dataset, persistWeights: true);
                    }
                    else
                    {
                        var preparedData = _trainer.GetPreparedFold(0, 1, model.ModelType, mlContext);
                        if (model.ModelType == 1)
                        {
                            result = _trainer.TrainLightGbmOptimized(model, mlContext, (HyperparameterTrainer.LightGbmFoldData)preparedData, persistWeights: true);
                        }
                        else
                        {
                            result = _trainer.TrainTensorFlowOptimized(model, (HyperparameterTrainer.TensorFlowFoldData)preparedData, persistWeights: true);
                        }
                    }

                    model.TrainAccuracy = result.TrainAccuracy;
                    model.TrainLoss = result.TrainLoss;
                    model.TrainBrier = result.TrainBrier;
                    model.TrainFocalLoss = result.TrainFocalLoss;
                    model.ValidationAccuracy = result.ValidationAccuracy;
                    model.ValidationLoss = result.ValidationLoss;
                    model.ValidationBrier = result.ValidationBrier;
                    model.ValidationFocalLoss = result.ValidationFocalLoss;

                    _repository.InsertMLParameter(model);

                    var correlationInfos = (result.FeatureCorrelations ?? Array.Empty<HyperparameterTrainer.FeatureCorrelation>())
                        .Select(c => new TrainAIViewModel.FeatureCorrelationInfo
                        {
                            FeatureKey = c.FeatureKey,
                            Dimension = c.Dimension,
                            Correlation = c.Correlation
                        })
                        .ToList();

                    results.Add(new TrainAIViewModel.TrainAIModelResult
                    {
                        Parameters = model,
                        FeatureCorrelations = correlationInfos
                    });

                    Console.WriteLine($"[TrainAI] Completed model {modelIndex}/{parameters.Count}. Train acc: {result.TrainAccuracy:F4}, train loss: {result.TrainLoss:F4}.");
                }

                Console.WriteLine("[TrainAI] Training run finished.");
            });

            viewModel.Results ??= new List<TrainAIViewModel.TrainAIModelResult>();
            viewModel.Results.Clear();
            viewModel.Results.AddRange(results);

            // Clear model state so the freshly computed results are rendered instead of
            // being suppressed by the existing form values submitted with the request.
            ModelState.Clear();

            return View(viewModel);
        }
        [HttpGet]
        public IActionResult TestAI(
            [FromQuery(Name = "units")] int? requestedUnits,
            [FromQuery(Name = "dropout")] double? requestedDropout,
            [FromQuery(Name = "layers")] int? requestedLayers,
            [FromQuery(Name = "learningRate")] double? requestedLearningRate,
            [FromQuery(Name = "epochs")] int? requestedEpochs,
            [FromQuery(Name = "batchSize")] int? requestedBatchSize,
            [FromQuery(Name = "folds")] int? requestedFolds,
            [FromQuery(Name = "lgbmLeaves")] int? requestedLgbmLeaves,
            [FromQuery(Name = "lgbmMinData")] int? requestedLgbmMinData,
            [FromQuery(Name = "lgbmMaxDepth")] int? requestedLgbmMaxDepth,
            [FromQuery(Name = "modelType")] int requestedModelType = 0)
        {
            var viewModel = new AITestResultViewModel
            {
                RequestedUnits = requestedUnits,
                RequestedDropout = requestedDropout,
                RequestedLayers = requestedLayers,
                RequestedLearningRate = requestedLearningRate,
                RequestedEpochs = requestedEpochs,
                RequestedBatchSize = requestedBatchSize,
                RequestedFolds = requestedFolds,
                RequestedLgbmLeaves = requestedLgbmLeaves,
                RequestedLgbmMinDataInLeaf = requestedLgbmMinData,
                RequestedLgbmMaxDepth = requestedLgbmMaxDepth,
                RequestedModelType = requestedModelType,
                Countries = _repository.GetCountries().ToList()
            };

            return View(viewModel);
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestAI(AITestResultViewModel request)
        {
            static bool HasValidUnits(int units) => MLParameterValidator.EnsureUnits(units, int.MinValue) == units;
            static bool HasValidDropout(double dropout) => !double.IsNaN(MLParameterValidator.EnsureDropout(dropout, double.NaN));
            static bool HasValidLayers(int layers) => MLParameterValidator.EnsureLayers(layers, int.MinValue) == layers;
            static bool HasValidLearningRate(double learningRate) => !double.IsNaN(MLParameterValidator.EnsureLearningRate(learningRate, double.NaN));
            static bool HasValidPositive(int value) => MLParameterValidator.EnsurePositive(value, int.MinValue) == value;
            static bool TryNormalizeBatchSize(int value, out int normalized)
            {
                normalized = MLParameterValidator.EnsureBatchSize(value, fallback: 0);
                return normalized > 0;
            }
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

            var requestedLgbmLeaves = ReadRequestedInt(request.RequestedLgbmLeaves, nameof(request.RequestedLgbmLeaves));
            var requestedLgbmMinData = ReadRequestedInt(request.RequestedLgbmMinDataInLeaf, nameof(request.RequestedLgbmMinDataInLeaf));
            var requestedLgbmMaxDepth = ReadRequestedInt(request.RequestedLgbmMaxDepth, nameof(request.RequestedLgbmMaxDepth));

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
            viewModel.RequestedLgbmLeaves = requestedLgbmLeaves;
            viewModel.RequestedLgbmMinDataInLeaf = requestedLgbmMinData;
            viewModel.RequestedLgbmMaxDepth = requestedLgbmMaxDepth;
            viewModel.Countries = _repository.GetCountries().ToList();
            var validationRaceIds = _repository.GetRaceIdsBetweenDates(validationStart, validationEnd, request.SelectedCountry);
            if (validationRaceIds.Count == 0)
            {
                viewModel.Message = $"No races were found in the last {months} {monthLabel} to use for validation.";
                return View(viewModel);
            }

            var trainingRaceIds = _repository.GetRaceIdsOutsideRange(validationStart, validationEnd, request.SelectedCountry);
            if (trainingRaceIds.Count == 0)
            {
                viewModel.Message = "No training data is available outside the validation window.";
                return View(viewModel);
            }
            TrainingDataset dataset;
            if (request.UseExistingWeights)
            {
                if (!_trainer.IsModelPersisted())
                {
                    viewModel.Message = "The AI model weights file (aiweights.json) was not found. Please train a model before running this test.";
                    return View(viewModel);
                }
                dataset = _trainer.LoadValidationDataset(new HashSet<int>(trainingRaceIds), new HashSet<int>(validationRaceIds), includeIdentifiers: true);
            }
            else
            {
                dataset = _trainer.LoadTrainingDataset(new HashSet<int>(trainingRaceIds), new HashSet<int>(validationRaceIds), includeIdentifiers: true);
            }

            if (dataset.ValidationRaces.Count == 0)
            {
                viewModel.Message = "The validation dataset was empty after preparation.";
                return View(viewModel);
            }


            MLParameter parameter;
            string? batchSizeAdjustmentMessage = null;
            if (request.UseExistingWeights)
            {
                parameter = new MLParameter();
            }
            else
            {
                int normalizedBatchSize = 0;
                var invalidHyperparameters = new List<string>();

                bool isLgbm = request.RequestedModelType == 1;

                if (isLgbm)
                {
                    if (requestedLgbmLeaves.HasValue && requestedLgbmLeaves.Value < 2)
                    {
                        invalidHyperparameters.Add("Leaves (must be >= 2)");
                    }
                }
                else
                {
                    bool unitsValid = !requestedUnits.HasValue || HasValidUnits(requestedUnits.Value);
                    bool layersValid = !requestedLayers.HasValue || HasValidLayers(requestedLayers.Value);

                    if (!unitsValid)
                    {
                        invalidHyperparameters.Add("Units per layer");
                    }

                    if (requestedDropout.HasValue && !HasValidDropout(requestedDropout.Value))
                    {
                        invalidHyperparameters.Add("Dropout");
                    }

                    if (!layersValid)
                    {
                        invalidHyperparameters.Add("Layers");
                    }

                    if (layersValid && requestedLayers.GetValueOrDefault() > 0 && (!requestedUnits.HasValue || requestedUnits.Value <= 0))
                    {
                        invalidHyperparameters.Add("Units per layer (must be positive when Layers > 0)");
                    }
                    if (requestedBatchSize.HasValue && !TryNormalizeBatchSize(requestedBatchSize.Value, out normalizedBatchSize))
                    {
                        invalidHyperparameters.Add("Batch size");
                    }
                }

                if (requestedLearningRate.HasValue && !HasValidLearningRate(requestedLearningRate.Value))
                {
                    invalidHyperparameters.Add("Learning rate");
                }

                if (requestedEpochs.HasValue && !HasValidPositive(requestedEpochs.Value))
                {
                    invalidHyperparameters.Add("Epochs");
                }

                if (requestedFolds.HasValue && !HasValidPositive(requestedFolds.Value))
                {
                    invalidHyperparameters.Add("Folds");
                }

                if (invalidHyperparameters.Count > 0)
                {
                    var invalidList = string.Join(", ", invalidHyperparameters);
                    ModelState.AddModelError(string.Empty, $"The following hyperparameter values are invalid: {invalidList}. Please correct them and try again.");
                    return View(viewModel);
                }

                bool hasRequestedHyperparameters;
                if (isLgbm)
                {
                    hasRequestedHyperparameters =
                       requestedLgbmLeaves.HasValue &&
                       requestedLgbmMinData.HasValue &&
                       requestedLgbmMaxDepth.HasValue &&
                       requestedLearningRate.HasValue &&
                       requestedEpochs.HasValue &&
                       requestedFolds.HasValue;
                }
                else
                {
                    hasRequestedHyperparameters =
                       requestedUnits.HasValue &&
                       requestedDropout.HasValue &&
                       requestedLayers.HasValue &&
                       requestedLearningRate.HasValue &&
                       requestedEpochs.HasValue &&
                       requestedBatchSize.HasValue &&
                       requestedFolds.HasValue;
                }

                if (!hasRequestedHyperparameters)
                {
                    viewModel.Message = "Please supply all hyperparameter values before running the AI test.";
                    return View(viewModel);
                }

                if (!isLgbm && normalizedBatchSize < requestedBatchSize.Value)
                {
                    batchSizeAdjustmentMessage = $"Batch size reduced to {normalizedBatchSize} to respect the maximum of {MLParameterValidator.MaxBatchSize}.";
                }

                parameter = new MLParameter
                {
                    ModelType = request.RequestedModelType,
                    Units = isLgbm ? null : requestedUnits,
                    Dropout = isLgbm ? null : requestedDropout,
                    Layers = isLgbm ? null : requestedLayers,
                    LgbmLeaves = isLgbm ? requestedLgbmLeaves : null,
                    LgbmMinDataInLeaf = isLgbm ? requestedLgbmMinData : null,
                    LgbmMaxDepth = isLgbm ? requestedLgbmMaxDepth : null,
                    LearningRate = requestedLearningRate.Value,
                    Epochs = requestedEpochs.Value,
                    BatchSize = isLgbm ? null : normalizedBatchSize,
                    Folds = requestedFolds.Value,
                    Fold = null,
                    RunDate = DateTime.UtcNow
                };
            }

            var result = new HyperparameterTrainer.TrainingResult();
            if (request.UseExistingWeights)
            {
                result = await Task.Run(() => _trainer.Evaluate(dataset));
            }
            else
            {
                result = await Task.Run(() => _trainer.Train(parameter, 0, 1, dataset, persistWeights: false));
            }
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
                Fold = parameter.Fold,
                EnableFeatureCorrelations = parameter.EnableFeatureCorrelations
            };
            viewModel.TrainAccuracy = result.TrainAccuracy;
            viewModel.TrainLoss = result.TrainLoss;
            viewModel.TrainBrier = result.TrainBrier;
            viewModel.TrainFocalLoss = result.TrainFocalLoss;
            viewModel.ValidationAccuracy = result.ValidationAccuracy;
            viewModel.ValidationLoss = result.ValidationLoss;
            viewModel.ValidationBrier = result.ValidationBrier;
            viewModel.ValidationFocalLoss = result.ValidationFocalLoss;
            viewModel.ValidationRaceCount = dataset.ValidationRaces.Count;
            viewModel.TrainingRaceCount = dataset.TrainingRaces.Count;
            var accuracyMessage = $"Validation accuracy over {viewModel.ValidationRaceCount} races: {result.ValidationAccuracy:P2}.";
            viewModel.Message = batchSizeAdjustmentMessage is null
                ? accuracyMessage
                : string.Concat(batchSizeAdjustmentMessage, " ", accuracyMessage);

            var raceSummaries = _repository.GetRaceSummaries(result.ValidationRaceIds);
            viewModel.Simulation = RunValidationSimulation(result, raceSummaries, startingBankroll, viewModel);

            return View(viewModel);
        }
        private ValidationSimulationResult RunValidationSimulation(
            HyperparameterTrainer.TrainingResult result,
            IDictionary<int, RaceSummary> raceSummaries,
            decimal startingBankroll,
            AITestResultViewModel viewModel)
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
            var dailyRaceResults = new List<RaceResultViewModel>();
            decimal bankrollBefore = bankroll;

            var allRaceData = Enumerable.Range(0, examples.Count)
                 .Select(i => new
                 {
                     RaceId = raceIds[i],
                     Example = examples[i]
                 })
                .GroupBy(x => x.RaceId)
                .Select(g =>
                {
                    raceSummaries.TryGetValue(g.Key, out var summary);
                    return new
                    {
                        RaceGroup = g.ToList(),
                        RaceDate = summary?.RaceDate ?? DateTime.MinValue,
                        Summary = summary
                    };
                })
                 .OrderBy(r => r.RaceDate)
                .ToList();
            var allHorseNames = allRaceData
               .SelectMany(race => race.RaceGroup)
               .Select(runner => runner.Example.HorseName)
               .Where(name => !string.IsNullOrWhiteSpace(name))
               .Distinct()
               .ToList();
            var historicalCounts = _repository.GetHistoricalRaceCountsByHorseNames(allHorseNames).CountsByOriginal;
            var raceIndex = 0;
            foreach (var race in allRaceData)
            {
                raceIndex++;
                var raceDate = race.RaceDate.ToString("yyyy-MM-dd");
                var raceTitle = race.Summary?.Title ?? "Unknown Race";
                Console.WriteLine($"[AI] Processing race {raceIndex}/{allRaceData.Count}: {raceDate} - {raceTitle}");
                bankrollBefore = bankroll;
                bankrollBefore = bankroll;
                var raceGroup = race.RaceGroup;
                var summary = race.Summary;

                var runnersForOddsCalc = raceGroup.Select(r => r.Example).ToList();
                var calculatedProbs = _aiOddsCalculator.CalculateProbabilities(runnersForOddsCalc, useHybrid: viewModel.UseHybrid);

                var runnersWithProbs = raceGroup
                    .Select(r => new
                    {
                        r.Example,
                        CalculatedProbability = (double)(calculatedProbs
                            .FirstOrDefault(p => p.HorseId == r.Example.HorseId)?.Probability ?? 0m)
                    })
                    .ToList();

                var predictedWinnerDetails = runnersWithProbs
                    .OrderByDescending(r => r.CalculatedProbability)
                    .FirstOrDefault();

                var actualWinnerDetails = runnersWithProbs.FirstOrDefault(r => r.Example.Label >= 0.5f);

                var isCorrectPrediction = predictedWinnerDetails != null && actualWinnerDetails != null &&
                                          predictedWinnerDetails.Example.HorseId == actualWinnerDetails.Example.HorseId;

                var stake = 0m;
                var aiOdds = 0m;
                var bookmakerOdds = 0m;

                if (predictedWinnerDetails != null)
                {
                    if (predictedWinnerDetails.Example.StartingPriceDecimal.HasValue)
                    {
                        bookmakerOdds = predictedWinnerDetails.Example.StartingPriceDecimal.Value;
                    }
                    aiOdds = BettingMath.CalculateAiDecimalOdds(predictedWinnerDetails.CalculatedProbability);

                    if (bankroll > 0m && bookmakerOdds > 1m)
                    {
                        var probability = predictedWinnerDetails.CalculatedProbability;
                        var marketProbability = 1.0 / (double)bookmakerOdds;
                        var edge = probability - marketProbability;

                        var kellyFraction = BettingMath.CalculateKellyFraction(edge, (double)bookmakerOdds, _maxKellyFraction);

                        if (viewModel.KellyDampener > 0)
                        {
                            kellyFraction /= viewModel.KellyDampener;
                        }

                        if (kellyFraction > 0m)
                        {
                            stake = BettingMath.CalculateSequentialStake(bankroll, kellyFraction);
                            if (stake > 0m)
                            {
                                bankroll -= stake;
                                totalStaked += stake;

                                if (isCorrectPrediction)
                                {
                                    wins++;
                                    bankroll += stake * bookmakerOdds;
                                }

                                bets.Add(new ValidationBetResult
                                {
                                    RaceId = race.RaceGroup.First().RaceId,
                                    RaceDate = summary?.RaceDate,
                                    RaceTitle = summary?.Title,
                                    CourseName = summary?.CourseName,
                                    HorseName = predictedWinnerDetails.Example.HorseName,
                                    DecimalOdds = bookmakerOdds,
                                    AiDecimalOdds = aiOdds,
                                    AiProbability = probability,
                                    MarketProbability = marketProbability,
                                    Differential = edge,
                                    Stake = stake,
                                    Won = isCorrectPrediction,
                                    Bankroll = bankrollBefore
                                });
                            }
                        }
                    }
                }
                var runnersForRace = runnersWithProbs.Select(r =>
                {
                    historicalCounts.TryGetValue(r.Example.HorseName, out var count);
                    return new RunnerViewModel
                    {
                        HorseName = r.Example.HorseName,
                        HistoricalRaceCount = count > 0 ? count : (int?)null,
                        AiProbability = r.CalculatedProbability,
                        BookmakerOdds = r.Example.StartingPriceDecimal ?? 0m,
                        IsWinner = r.Example.Label >= 0.5f
                    };
                }).ToList();
                dailyRaceResults.Add(new RaceResultViewModel
                {
                    RaceId = race.RaceGroup.First().RaceId,
                    RaceTitle = summary?.Title,
                    PredictedWinner = predictedWinnerDetails?.Example.HorseName,
                    ActualWinner = actualWinnerDetails?.Example.HorseName,
                    IsCorrectPrediction = isCorrectPrediction,
                    Bankroll = bankrollBefore,
                    Stake = stake,
                    AiOdds = aiOdds,
                    BookmakerOdds = predictedWinnerDetails?.Example.StartingPriceDecimal ?? 0m,
                    Runners = runnersForRace
                });
            }

            viewModel.DailyResults = dailyRaceResults
                .GroupBy(r =>
                {
                    raceSummaries.TryGetValue(r.RaceId, out var summary);
                    return summary?.RaceDate?.Date ?? DateTime.MinValue.Date;
                })
                .OrderBy(g => g.Key)
                .Select(g => new DayResultViewModel
                {
                    RaceDate = g.Key,
                    Races = g.ToList()
                })
                .ToList();
            return new ValidationSimulationResult
            {
                StartingBankroll = startingBankroll,
                EndingBankroll = bankroll,
                TotalStaked = totalStaked,
                BetCount = bets.Count,
                WinCount = wins,
                Bets = bets,
                AverageRoiPerRace = bets.Count > 0 ? bets.Average(b => b.Profit / b.Stake) : 0m
            };
        }
    }
}
