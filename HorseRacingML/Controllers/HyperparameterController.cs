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

        [HttpGet]
        public IActionResult Custom()
        {
            return View(new MLParameter { RunDate = DateTime.UtcNow });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult Custom(MLParameter model)
        {
            if (ModelState.IsValid)
            {
                var result = _trainer.Train(model);
                model.TrainAccuracy = result.TrainAccuracy;
                model.TrainLoss = result.TrainLoss;
                model.ValidationAccuracy = result.ValidationAccuracy;
                model.ValidationLoss = result.ValidationLoss;
                _repository.InsertMLParameter(model);
                return RedirectToAction("Index", "Home");
            }

            return View(model);
        }
    }
}