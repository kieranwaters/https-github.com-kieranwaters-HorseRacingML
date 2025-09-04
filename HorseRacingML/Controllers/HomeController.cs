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
        public IActionResult Scrape(DateTime? startDate, DateTime? endDate)
        {
            var start = startDate ?? new DateTime(2018, 5, 5);
            var end = endDate ?? start;

            var scraper = new RaceResultsScraper(_repository);
            scraper.Scrape(start, end);

            return RedirectToAction("Index");
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}