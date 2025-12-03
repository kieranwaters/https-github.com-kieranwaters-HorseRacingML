using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Globalization;
using System.Threading.Tasks;

namespace HorseRacingML.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly RacingRepository _repo;
        private readonly RaceResultsScraper _raceResultsScraper;
        private readonly BetfairNavigationService _navigationService;
        private readonly AutomationSettingsService _automationSettings;
        private readonly HyperparameterTrainer _trainer;

        public HomeController(
            ILogger<HomeController> logger,
            RacingRepository repo,
            RaceResultsScraper raceResultsScraper,
            BetfairNavigationService navigationService,
            AutomationSettingsService automationSettings,
            HyperparameterTrainer trainer)
        {
            _logger = logger;
            _repo = repo;
            _raceResultsScraper = raceResultsScraper;
            _navigationService = navigationService;
            _automationSettings = automationSettings;
            _trainer = trainer;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        public IActionResult ScrapeOptions()
        {
            return View();
        }

        public async Task<IActionResult> ScrapeRaceResults()
        {
            await Task.Run(() => _raceResultsScraper.ScrapeFromTodayBackwards());
            return RedirectToAction(nameof(ScrapeOptions));
        }

        public async Task<IActionResult> ScrapeEasternResults(DateTime startDate)
        {
            await Task.Run(() => _raceResultsScraper.ScrapeEasternFromDate(startDate));
            return RedirectToAction(nameof(ScrapeOptions));
        }

        public IActionResult DayReportOptions()
        {
            return View(new DayReportFilterViewModel());
        }

        [HttpGet]
        public async Task<IActionResult> DayReport(string startTime, string endTime, string region, bool showFeatureSignificance)
        {
            await _navigationService.LoginAsync();

            TimeSpan? start = null;
            TimeSpan? end = null;

            if (TimeSpan.TryParse(startTime, out var s)) start = s;
            if (TimeSpan.TryParse(endTime, out var e)) end = e;

            await _navigationService.OpenHorseRaceMeetingsInNewTabsAsync(
                scheduleStartTime: start,
                scheduleEndTime: end,
                scheduleRegion: region,
                closeExistingRaceTabs: true);

            var model = _navigationService.GenerateDayReport(_repo, _trainer, showFeatureSignificance);

            return View(model);
        }

        public IActionResult AutomateBets()
        {
            var snapshot = _automationSettings.GetSnapshot();
            var model = new AutomateBetsViewModel
            {
                KellyDampener = snapshot.KellyDampener,
                MaxKellyFraction = snapshot.MaxKellyFraction,
                MaxStakeMode = snapshot.MaxStakeMode,
                MaxStakePercent = snapshot.MaxStakePercentOfBankroll.HasValue ? snapshot.MaxStakePercentOfBankroll.Value * 100m : null,
                MaxStakeAmount = snapshot.MaxStakeFixedAmount,
                StatusMessage = "Automation loop running..."
            };
            return View(model);
        }

        [HttpPost]
        public IActionResult UpdateAutomationSettings([FromBody] UpdateAutomationSettingsRequest request)
        {
            if (request == null) return BadRequest();

            var update = new AutomationSettingsUpdate(
                request.KellyDampener,
                request.MaxKellyFraction,
                request.MaxStakeMode,
                request.MaxStakePercent,
                request.MaxStakeAmount
            );

            _automationSettings.UpdateSettings(update);
            return Json(new { success = true, settings = _automationSettings.GetSnapshot() });
        }

        public IActionResult CalculateFavouritesAccuracy()
        {
            var (inc, exc, logLoss) = _repo.GetFavouriteAccuracy();
            var model = new FavouriteAccuracyViewModel
            {
                IncludingJoint = inc,
                ExcludingJoint = exc,
                LogLoss = logLoss
            };
            return View(model);
        }
    }
}
