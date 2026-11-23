using System;
using HorseRacingML.Jobs;
using Microsoft.AspNetCore.Mvc;

namespace HorseRacingML.Controllers
{
    public class JobsController : Controller
    {
        private readonly BackgroundJobStore _store;

        public JobsController(BackgroundJobStore store)
        {
            _store = store;
        }

        public IActionResult Index()
        {
            return View(_store.List());
        }

        [HttpGet]
        public IActionResult Status(Guid id)
        {
            var job = _store.Get(id);
            if (job is null)
            {
                return NotFound();
            }

            return Json(new
            {
                job.Id,
                Kind = job.Kind.ToString(),
                Status = job.Status.ToString(),
                job.DisplayName,
                job.Message,
                job.Progress,
                job.Error,
                job.CreatedUtc,
                job.StartedUtc,
                job.CompletedUtc
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult Cancel(Guid id)
        {
            _store.RequestCancel(id);
            TempData["Message"] = "Cancellation requested.";
            return RedirectToAction(nameof(Index));
        }
    }
}
