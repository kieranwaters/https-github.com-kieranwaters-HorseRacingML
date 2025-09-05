using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Data;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Threading.Tasks;

namespace HorseRacingML.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly RacingRepository _repository;

        public HomeController(ILogger<HomeController> logger, RacingRepository repository)
        {
            _logger = logger;
            _repository = repository;
        }
        public async Task<IActionResult> Scrape(DateTime? startDate, DateTime? endDate)
        {
            var start = startDate ?? new DateTime(2018, 6, 23);
            var end = endDate ?? start;

            await Task.Run(() =>
            {
                var scraper = new RaceResultsScraper(_repository);
                scraper.Scrape(start, end);
            });

            return RedirectToAction("Index");
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }
        public async Task<IActionResult> AutomateBets([FromServices] BetfairNavigationService betfair)
        {
            await betfair.LoginAsync();
            // TODO: add bet placement automation here
            await betfair.OpenHorseRaceMeetingsInNewTabsAsync();
            betfair.ScrapeOpenRaceTabs(_repository);
            return RedirectToAction("Index");
        }
        public IActionResult CalculateFavouritesAccuracy()
        {
            var (includingJoint, excludingJoint) = _repository.GetFavouriteAccuracy();
            var model = new FavouriteAccuracyViewModel
            {
                IncludingJoint = includingJoint,
                ExcludingJoint = excludingJoint
            };
            return View(model);
        }
        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}