using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using HorseRacingML.Models;
using HorseRacingML.Data;
using HorseRacingML.ML;

namespace HorseRacingML.Controllers
{
    public class HyperparameterController : Controller
    {
        private readonly RacingRepository _repository;
        private readonly HyperparameterTrainer _trainer;

        public HyperparameterController(
            RacingRepository repository,
            HyperparameterTrainer trainer)
        {
            _repository = repository;
            _trainer = trainer;
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Custom(MLParameterBatch batch)
        {
            if (ModelState.IsValid)
            {
                await Task.Run(() =>
                {
                    var parameters = batch.Parameters;
                    Console.WriteLine($"[Hyperparameter] Starting custom run for {parameters.Count} parameter set(s).");

                    var dataset = _trainer.PrepareDataset();
                    Console.WriteLine($"[Hyperparameter] Dataset prepared with {dataset.Races.Count} races and {dataset.RowCount} runner rows.");

                    int modelIndex = 0;
                    foreach (var model in parameters)
                    {
                        modelIndex++;
                        Console.WriteLine($"[Hyperparameter] Starting model {modelIndex}/{parameters.Count} (layers: {model.Layers}, units: {model.Units}, dropout: {model.Dropout}, lr: {model.LearningRate}).");

                        model.RunDate = DateTime.UtcNow;
                        double totalTrainAccuracy = 0;
                        double totalValidationAccuracy = 0;
                        double totalTrainLoss = 0;
                        double totalValidationLoss = 0;
                        double totalTrainBrier = 0;
                        double totalValidationBrier = 0;

                        for (int i = 0; i < model.Folds; i++)
                        {
                            Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} - training in progress...");
                            var result = _trainer.Train(model, dataset, i, model.Folds);
                            Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");

                            totalTrainAccuracy += result.TrainAccuracy;
                            totalValidationAccuracy += result.ValidationAccuracy;
                            totalTrainLoss += result.TrainLoss;
                            totalValidationLoss += result.ValidationLoss;
                            totalTrainBrier += result.TrainBrier;
                            totalValidationBrier += result.ValidationBrier;
                        }

                        if (model.Folds > 0)
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
                                TrainAccuracy = totalTrainAccuracy / model.Folds,
                                TrainLoss = totalTrainLoss / model.Folds,
                                TrainBrier = totalTrainBrier / model.Folds,
                                ValidationAccuracy = totalValidationAccuracy / model.Folds,
                                ValidationLoss = totalValidationLoss / model.Folds,
                                ValidationBrier = totalValidationBrier / model.Folds,
                                Fold = 0
                            };
                            _repository.InsertMLParameter(averagedModel);
                            Console.WriteLine($"[Hyperparameter] Stored averaged results across {model.Folds} folds.");
                        }
                        else
                        {
                            Console.WriteLine($"[Hyperparameter] Skipped storing results for model {modelIndex} due to invalid fold count {model.Folds}.");
                        }

                        Console.WriteLine($"[Hyperparameter] Completed model {modelIndex}/{parameters.Count}.");
                    }
                    Console.WriteLine("[Hyperparameter] Custom run finished.");
                });
                return RedirectToAction("Index", "Home");
            }

            return View(batch);
        }

        [HttpGet]
        public IActionResult Custom()
        {
            return View(new MLParameterBatch
            {
                Parameters = new System.Collections.Generic.List<MLParameter>
                {
                    new MLParameter { RunDate = DateTime.UtcNow }
                }
            });
        }
    }
}