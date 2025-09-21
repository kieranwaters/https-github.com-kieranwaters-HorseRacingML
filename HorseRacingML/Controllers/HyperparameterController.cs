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
                    Console.WriteLine($"[Hyperparameter] Dataset prepared with {dataset.Races.Count} races and {dataset.Rows.Count} runner rows.");

                    int modelIndex = 0;
                    foreach (var model in parameters)
                    {
                        modelIndex++;
                        Console.WriteLine($"[Hyperparameter] Starting model {modelIndex}/{parameters.Count} (layers: {model.Layers}, units: {model.Units}, dropout: {model.Dropout}, lr: {model.LearningRate}).");

                        model.RunDate = DateTime.UtcNow;
                        for (int i = 0; i < model.Folds; i++)
                        {
                            Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} - training in progress...");
                            var result = _trainer.Train(model, dataset, i, model.Folds);
                            Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} complete. Train acc: {result.TrainAccuracy:F4}, val acc: {result.ValidationAccuracy:F4}.");

                            var foldModel = new MLParameter
                            {
                                RunDate = model.RunDate,
                                Units = model.Units,
                                Dropout = model.Dropout,
                                Layers = model.Layers,
                                LearningRate = model.LearningRate,
                                Epochs = model.Epochs,
                                BatchSize = model.BatchSize,
                                TrainAccuracy = result.TrainAccuracy,
                                TrainLoss = result.TrainLoss,
                                TrainBrier = result.TrainBrier,
                                ValidationAccuracy = result.ValidationAccuracy,
                                ValidationLoss = result.ValidationLoss,
                                ValidationBrier = result.ValidationBrier,
                                Fold = i + 1
                            };
                            _repository.InsertMLParameter(foldModel);
                            Console.WriteLine($"[Hyperparameter]  Fold {i + 1}/{model.Folds} results stored.");
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