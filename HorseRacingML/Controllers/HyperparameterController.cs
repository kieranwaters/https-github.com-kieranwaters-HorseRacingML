using System;
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

        public HyperparameterController(RacingRepository repository, HyperparameterTrainer trainer)
        {
            _repository = repository;
            _trainer = trainer;
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult Custom(MLParameter model)
        {
            if (ModelState.IsValid)
            {
                for (int i = 0; i < model.Folds; i++)
                {
                    var result = _trainer.Train(model, i, model.Folds);
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
                        ValidationAccuracy = result.ValidationAccuracy,
                        ValidationLoss = result.ValidationLoss,
                        Fold = i + 1
                    };
                    _repository.InsertMLParameter(foldModel);
                }
                return RedirectToAction("Index", "Home");
            }

            return View(model);
        }

        [HttpGet]
        public IActionResult Custom()
        {
            return View(new MLParameter { RunDate = DateTime.UtcNow });
        }
    }
}