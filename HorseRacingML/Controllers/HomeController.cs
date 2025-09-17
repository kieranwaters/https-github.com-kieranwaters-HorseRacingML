using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Data;
using HorseRacingML.Services;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Threading.Tasks;
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
        public async Task<IActionResult> AutomateBets([FromServices] BetfairNavigationService betfair)
        {
            await betfair.LoginAsync();
            await betfair.OpenHorseRaceMeetingsInNewTabsAsync();
            var recommendations = betfair.ScrapeOpenRaceTabs(_repository);
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
                var marketProb = (best.MarketProbability * 100).ToString("0.##", CultureInfo.InvariantCulture);
                var diff = (best.Differential * 100).ToString("0.##", CultureInfo.InvariantCulture);
                var stake = best.Stake.ToString("0.##", CultureInfo.InvariantCulture);
                var kelly = (best.KellyFraction * 100m).ToString("0.##", CultureInfo.InvariantCulture);

                var message = $"Best value bet: {horse}{venue} ({race}) – odds {odds}, AI win {aiProb}% vs market {marketProb}% (diff {diff}%). Kelly stake {stake} ({kelly}% bankroll).";
                TempData["Message"] = message;
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
        public IActionResult ScrapeRaceResults([FromServices] RaceResultsScraper scraper)
        {
            _status.Update($"Scraping started at {DateTime.Now:G}");
            Task.Run(() =>
            {
                try
                {
                    scraper.ScrapeFromEarliest();
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