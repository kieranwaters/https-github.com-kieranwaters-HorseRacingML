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
using System;

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
        public async Task<IActionResult> DayReport(
            [FromServices] BetfairNavigationService betfair,
            [FromServices] HyperparameterTrainer trainer)
        {
            await betfair.LoginAsync();
            await betfair.OpenHorseRaceMeetingsInNewTabsAsync();

            var report = betfair.GenerateDayReport(_repository, trainer);
            var usedNextDay = false;

            if (ShouldLoadNextDaySchedule(report))
            {
                _logger.LogInformation("Day report contains only USA races; attempting to load the next day's schedule.");
                var switched = await betfair.TrySelectHorseRacingDayAsync(1);
                if (switched)
                {
                    await betfair.OpenHorseRaceMeetingsInNewTabsAsync(closeExistingRaceTabs: true);
                    var nextDayReport = betfair.GenerateDayReport(_repository, trainer);
                    if (nextDayReport?.Races?.Count > 0)
                    {
                        report = nextDayReport;
                        usedNextDay = true;
                    }
                    else
                    {
                        _logger.LogWarning("Next day schedule did not produce any races; retaining USA schedule report.");
                    }
                }
                else
                {
                    _logger.LogWarning("Unable to switch Betfair schedule to the next day; retaining USA schedule report.");
                }
            }

            var statusMessage = usedNextDay
                ? $"Day report (next day schedule) generated at {DateTime.Now:G}."
                : $"Day report generated at {DateTime.Now:G}.";
            _status.Update(statusMessage);
            return View(report);
        }
        private static bool ShouldLoadNextDaySchedule(DayReportViewModel? report)
        {
            if (report?.Races == null || report.Races.Count == 0)
            {
                return false;
            }

            var hasRace = false;
            foreach (var race in report.Races)
            {
                if (race == null)
                {
                    continue;
                }

                hasRace = true;
                if (!IsUsRace(race))
                {
                    return false;
                }
            }

            return hasRace;
        }

        private static bool IsUsRace(RaceDayReport? race)
        {
            var country = race?.VenueCountry;
            if (string.IsNullOrWhiteSpace(country))
            {
                return false;
            }

            var normalized = country.Trim();
            return normalized.Equals("USA", StringComparison.OrdinalIgnoreCase) ||
                   normalized.Equals("US", StringComparison.OrdinalIgnoreCase) ||
                   normalized.Equals("U.S.A.", StringComparison.OrdinalIgnoreCase) ||
                   normalized.Equals("United States", StringComparison.OrdinalIgnoreCase) ||
                   normalized.Equals("United States Of America", StringComparison.OrdinalIgnoreCase);
        }
        public async Task<IActionResult> AutomateBets(
            [FromServices] BetfairNavigationService betfair,
            [FromServices] HyperparameterTrainer trainer)
        {
            await betfair.LoginAsync();
            var raceWindow = TimeSpan.FromHours(2);
            var refreshLeadTime = TimeSpan.FromMinutes(20);
            var cycleStartUtc = DateTime.UtcNow;
            await betfair.OpenHorseRaceMeetingsInNewTabsAsync(
                closeExistingRaceTabs: true,
                raceWindow: raceWindow,
                windowReferenceUtc: cycleStartUtc);
            var cycleEndUtc = DateTime.UtcNow;
            var initialDelay = raceWindow - refreshLeadTime - (cycleEndUtc - cycleStartUtc);
            if (initialDelay < TimeSpan.Zero)
            {
                initialDelay = TimeSpan.Zero;
            }
            betfair.StartAutomatedBettingLoop(_repository, trainer, raceWindow, refreshLeadTime, initialDelay);
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
        [HttpPost]
        public async Task<IActionResult> RefreshRaceOdds(
                    [FromBody] RefreshRaceRequest request,
                    [FromServices] BetfairNavigationService betfair,
                    [FromServices] HyperparameterTrainer trainer)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.RaceUrl))
            {
                return BadRequest(new { success = false, message = "Race URL is required." });
            }

            try
            {
                var refreshed = await betfair.RefreshRaceAsync(request.RaceUrl, _repository, trainer);
                if (refreshed == null)
                {
                    return NotFound(new { success = false, message = "Unable to refresh market data for the selected race." });
                }

                return Json(new { success = true, race = refreshed });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to refresh race odds for URL {RaceUrl}.", request.RaceUrl);
                return StatusCode(500, new { success = false, message = "An unexpected error occurred while refreshing the race." });
            }
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