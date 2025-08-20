using System;
using System.Diagnostics;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Microsoft.AspNetCore.Mvc;

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
        /// Currently scrapes results for today only and then redirects back to the home page.
        /// </summary>
        public IActionResult Scrape()
        {
            var scraper = new RaceResultsScraper();
            scraper.Scrape(DateTime.Today, DateTime.Today);
            return RedirectToAction("Index");
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}