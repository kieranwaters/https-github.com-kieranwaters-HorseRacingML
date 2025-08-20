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

        /// <summary>
        /// Triggers the race results scraping process using Selenium.
        /// Defaults to scraping results for 13 April 2005 unless start and end
        /// dates are provided via query parameters.
        /// </summary>
        /// <param name="startDate">Optional start date for scraping.</param>
        /// <param name="endDate">Optional end date for scraping.</param>
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