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

        public HomeController(ILogger<HomeController> logger, RacingRepository repository, ScrapingStatusService status)
        {
            _logger = logger;
            _repository = repository;
            _status = status;
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
        public IActionResult ScrapeRaceResults([FromServices] RaceResultsScraper scraper)
        {
            _status.Update($"Scraping started at {DateTime.Now:G}");
            Task.Run(() =>
            {
                try
                {
                    scraper.ScrapeFromTodayBackwards();
                    _status.Update($"Scraping completed at {DateTime.Now:G}");
                }
                catch (Exception ex)
                {
                    _status.Update($"Scraping failed: {ex.Message}");
                }
            });
            TempData["Message"] = "Scraping of recent race results has started.";
            return RedirectToAction("Index");
        }

        public IActionResult Index()
        {
            ViewData["StatusMessage"] = _status.Message;
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