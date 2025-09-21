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
                    var dataset = _trainer.PrepareDataset();
                    foreach (var model in batch.Parameters)
                    {
                        model.RunDate = DateTime.UtcNow;
                        for (int i = 0; i < model.Folds; i++)
                        {
                            var result = _trainer.Train(model, dataset, i, model.Folds);
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
                        }
                    }
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