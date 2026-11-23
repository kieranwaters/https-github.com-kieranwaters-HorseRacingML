using HorseRacingML.Jobs;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Data;
using HorseRacingML.Services;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Threading.Tasks;
using HorseRacingML.ML;
using System.Globalization;
using System.Linq;

namespace HorseRacingML.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly RacingRepository _repository;
        private readonly ScrapingStatusService _status;
        private readonly BackgroundJobQueue _jobQueue;
        private readonly BackgroundJobStore _jobStore;

        public HomeController(
            ILogger<HomeController> logger,
            RacingRepository repository,
            ScrapingStatusService status,
            BackgroundJobQueue jobQueue,
            BackgroundJobStore jobStore)
        {
            _logger = logger;
            _repository = repository;
            _status = status;
            _jobQueue = jobQueue;
            _jobStore = jobStore;
        }
        public IActionResult CalculateFavouritesAccuracy()
        {
            var (includingJoint, excludingJoint, logLoss) = _repository.GetFavouriteAccuracy();
            var model = new FavouriteAccuracyViewModel
            {
                IncludingJoint = includingJoint,
                ExcludingJoint = excludingJoint,
                LogLoss = logLoss
            };
            return View(model);
        }
        public async Task<IActionResult> AutomateBets(
            [FromServices] BetfairNavigationService betfair,
            [FromServices] HyperparameterTrainer trainer)
        {
            await betfair.LoginAsync();
            await betfair.OpenHorseRaceMeetingsInNewTabsAsync();
            var recommendations = betfair.ScrapeOpenRaceTabs(_repository, trainer);
            if (recommendations.Count > 0)
            {
                var best = recommendations
                    .OrderByDescending(r => r.Differential)
                    .ThenByDescending(r => r.KellyFraction)
                    .First();

                var horse = string.IsNullOrWhiteSpace(best.HorseName) ? "selection" : best.HorseName;
                var race = string.IsNullOrWhiteSpace(best.RaceTitle) ? "race" : best.RaceTitle;
                var venue = string.IsNullOrWhiteSpace(best.VenueName) ? string.Empty : $" at {best.VenueName}";
                var odds = best.DecimalOdds.ToString("0.00", CultureInfo.InvariantCulture);
                var aiProb = (best.AiProbability * 100).ToString("0.##", CultureInfo.InvariantCulture);
                var aiReturn = best.AiDecimalOdds.ToString("0.00", CultureInfo.InvariantCulture);
                var marketProb = (best.MarketProbability * 100).ToString("0.##", CultureInfo.InvariantCulture);
                var diff = (best.Differential * 100).ToString("0.##", CultureInfo.InvariantCulture);
                var stake = best.Stake.ToString("0.##", CultureInfo.InvariantCulture);
                var kelly = (best.KellyFraction * 100m).ToString("0.##", CultureInfo.InvariantCulture);

                var message = $"Best value bet: {horse}{venue} ({race})  odds {odds}, AI win {aiProb}% (AI return {aiReturn}) vs market {marketProb}% (diff {diff}%). Kelly stake {stake} ({kelly}% bankroll).";
                _status.Update(message);
            }
            else
            {
                const string message = "No positive expected value opportunities were found while scanning markets.";
                TempData["Message"] = message;
                _status.Update(message);
            }
            return RedirectToAction("Index");
        }
        [HttpGet]
        //public IActionResult ScrapeRaceResults([FromServices] RaceResultsScraper scraper)
        //{
        //    var startDate = new DateTime(2025, 9, 15);
        //    var endDate = new DateTime(2025, 9, 21);

        //    _status.Update($"Scraping results from {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd} started at {DateTime.Now:G}");
        //    Task.Run(() =>
        //    {
        //        try
        //        {
        //            scraper.Scrape(startDate, endDate);
        //            _status.Update($"Scraping completed at {DateTime.Now:G}");
        //        }
        //        catch (Exception ex)
        //        {
        //            _status.Update($"Scraping failed: {ex.Message}");
        //        }
        //    });
        //    TempData["Message"] = $"Scraping of race results from {startDate:dd/MM/yy} to {endDate:dd/MM/yy} has started.";
        //    return RedirectToAction("Index");
        //}
        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult ScrapeRaceResults()
        {
            if (!_jobQueue.TryEnqueue(BackgroundJobKind.ScrapeRaceResults, "Scrape race results", new Dictionary<string, string?>()))
            {
                TempData["Message"] = "A scrape job is already running or queued.";
            }
            else
            {
                TempData["Message"] = "Scraping of recent race results has been queued.";
            }

            return RedirectToAction("Index");
        }

        [HttpGet]
        public IActionResult WeatherStatus([FromServices] WeatherStatusService weather)
        {
            return View(weather.BuildReport());
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult BackfillWeather()
        {
            if (!_jobQueue.TryEnqueue(
                    BackgroundJobKind.BackfillWeather,
                    "Backfill weather",
                    new Dictionary<string, string?>()))
            {
                TempData["Message"] = "A weather backfill is already running or queued.";
            }
            else
            {
                TempData["Message"] =
                    "Weather backfill has been queued. It applies course coordinates and downloads " +
                    "Open-Meteo daily weather for every course with race history.";
            }

            return RedirectToAction("Index");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult BackfillSectionals()
        {
            if (!_jobQueue.TryEnqueue(
                    BackgroundJobKind.BackfillSectionals,
                    "Backfill sectionals",
                    new Dictionary<string, string?>()))
            {
                TempData["Message"] = "A sectionals backfill is already running or queued.";
            }
            else
            {
                TempData["Message"] =
                    "Sectional backfill has been queued. It downloads Racing TV / RaceiQ furlong " +
                    "splits for every matched British and Irish race from March 2023 onward.";
            }

            return RedirectToAction("Index");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult BackfillBreeding()
        {
            if (!_jobQueue.TryEnqueue(
                    BackgroundJobKind.BackfillBreeding,
                    "Backfill breeding",
                    new Dictionary<string, string?>()))
            {
                TempData["Message"] = "A breeding backfill is already running or queued.";
            }
            else
            {
                TempData["Message"] =
                    "Breeding backfill has been queued. It downloads sire, dam and damsire for " +
                    "runners from the Racing TV public racecards.";
            }

            return RedirectToAction("Index");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult RefreshWeather()
        {
            return EnqueueWeatherJob(
                WeatherBackfillMode.Refresh,
                "Refresh weather",
                "Weather refresh has been queued. It re-downloads daily weather (including the " +
                "extended variables) even for course-years that already have race rows.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult PopulateWeatherFeatures()
        {
            return EnqueueWeatherJob(
                WeatherBackfillMode.CacheOnly,
                "Populate weather features",
                "Weather feature rebuild has been queued. It recomputes race features from the " +
                "cached CourseWeatherDaily rows without calling Open-Meteo.");
        }

        private IActionResult EnqueueWeatherJob(
            WeatherBackfillMode mode,
            string displayName,
            string queuedMessage)
        {
            var payload = new Dictionary<string, string?>
            {
                ["mode"] = mode.ToString()
            };

            if (!_jobQueue.TryEnqueue(BackgroundJobKind.BackfillWeather, displayName, payload))
            {
                TempData["Message"] = "A weather job is already running or queued.";
            }
            else
            {
                TempData["Message"] = queuedMessage;
            }

            return RedirectToAction(nameof(WeatherStatus));
        }

        [NonAction]
        public void RunAutomateBetsWork(JobExecutionContext context)
        {
            var betfair = context.Services.GetRequiredService<BetfairNavigationService>();
            var trainer = context.Services.GetRequiredService<HyperparameterTrainer>();
            try
            {
                context.CancellationToken.ThrowIfCancellationRequested();
                betfair.LoginAsync().GetAwaiter().GetResult();
                context.CancellationToken.ThrowIfCancellationRequested();
                betfair.OpenHorseRaceMeetingsInNewTabsAsync().GetAwaiter().GetResult();
                context.CancellationToken.ThrowIfCancellationRequested();
                var recommendations = betfair.ScrapeOpenRaceTabs(_repository, trainer);
                if (recommendations.Count > 0)
                {
                    var best = recommendations
                        .OrderByDescending(r => r.Differential)
                        .ThenByDescending(r => r.KellyFraction)
                        .First();

                    var horse = string.IsNullOrWhiteSpace(best.HorseName) ? "selection" : best.HorseName;
                    var race = string.IsNullOrWhiteSpace(best.RaceTitle) ? "race" : best.RaceTitle;
                    var venue = string.IsNullOrWhiteSpace(best.VenueName) ? string.Empty : $" at {best.VenueName}";
                    var message = $"Best value bet: {horse}{venue} ({race}).";
                    _status.Update(message);
                    context.Report(message, 100);
                }
                else
                {
                    const string message = "No positive expected value opportunities were found while scanning markets.";
                    _status.Update(message);
                    context.Report(message, 100);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Automated betting failed.");
                _status.Update($"Automated betting failed: {ex.Message}");
                context.Report($"Automated betting failed: {ex.Message}", null);
                throw;
            }
        }

        /// <summary>
        /// One-off/periodic backfill of parsed run styles from historical
        /// comments. Optional maxRows is useful for a smoke test before running
        /// the whole history.
        /// </summary>
        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult BackfillRunStyles(int? maxRows = null)
        {
            var payload = new Dictionary<string, string?>();
            if (maxRows.HasValue)
            {
                payload["maxRows"] = maxRows.Value.ToString();
            }

            if (!_jobQueue.TryEnqueue(BackgroundJobKind.BackfillRunStyles, "Backfill run styles", payload))
            {
                TempData["Message"] = "A backfill job is already running or queued.";
            }
            else
            {
                TempData["Message"] = "Run-style backfill has been queued.";
            }

            return RedirectToAction("Index");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult AutomateBets()
        {
            if (!_jobQueue.TryEnqueue(BackgroundJobKind.AutomateBets, "Automate bets", new Dictionary<string, string?>()))
            {
                TempData["Message"] = "An automated betting job is already running or queued.";
            }
            else
            {
                TempData["Message"] = "Automated betting has been queued.";
            }

            return RedirectToAction("Index");
        }

        public IActionResult Index()
        {
            ViewData["StatusMessage"] = _status.Message;
            ViewData["Jobs"] = _jobStore.List();
            return View();
        }
        public async Task<IActionResult> DayReport(
            [FromServices] BetfairNavigationService betfair,
            [FromServices] HyperparameterTrainer trainer)
        {
            await betfair.LoginAsync();
            await betfair.OpenHorseRaceMeetingsInNewTabsAsync();
            var report = betfair.GenerateDayReport(_repository, trainer);
            _status.Update($"Day report generated at {DateTime.Now:G}.");
            return View(report);
        }

        public IActionResult Privacy()
        {
            return View();
        }
        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}