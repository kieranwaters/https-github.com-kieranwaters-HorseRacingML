using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;

namespace HorseRacingML.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        public IActionResult Scrape(DateTime? startDate, DateTime? endDate)
        {
            var start = startDate ?? new DateTime(2005, 4, 13);
            var end = endDate ?? start;

            var scraper = new RaceResultsScraper();
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