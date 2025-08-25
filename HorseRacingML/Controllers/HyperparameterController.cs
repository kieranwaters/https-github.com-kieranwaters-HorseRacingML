
using System;
using Microsoft.AspNetCore.Mvc;
using HorseRacingML.Models;
using HorseRacingML.Data;

namespace HorseRacingML.Controllers
{
    public class HyperparameterController : Controller
    {
        private readonly RacingRepository _repository;

        public HyperparameterController(RacingRepository repository)
        {
            _repository = repository;
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
                _repository.InsertMLParameter(model);
                return RedirectToAction("Index", "Home");
            }

            return View(model);
        }
    }
}