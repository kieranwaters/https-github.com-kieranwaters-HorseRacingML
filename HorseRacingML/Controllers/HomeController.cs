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
using System.Collections.Generic;

namespace HorseRacingML.Controllers
{
    public class HomeController : Controller
    {
        private const double AiProbabilityDisplayThreshold = 0.01d;
        private readonly ILogger<HomeController> _logger;
        private readonly RacingRepository _repository;
        private readonly ScrapingStatusService _status;
        private readonly AutomationSettingsService _automationSettings;

        public HomeController(
           ILogger<HomeController> logger,
           RacingRepository repository,
           ScrapingStatusService status,
           AutomationSettingsService automationSettings)
        {
            _logger = logger;
            _repository = repository;
            _status = status;
            _automationSettings = automationSettings;
        }
        [HttpGet]
        public IActionResult GetAutomationSettings()
        {
            var snapshot = _automationSettings.GetSnapshot();
            return Json(new
            {
                success = true,
                settings = new
                {
                    kellyDampener = snapshot.KellyDampener,
                    maxKellyFraction = snapshot.MaxKellyFraction,
                    maxStakeMode = snapshot.MaxStakeMode.ToString(),
                    maxStakePercent = snapshot.MaxStakePercentOfBankroll,
                    maxStakeAmount = snapshot.MaxStakeFixedAmount
                }
            });
        }

        [HttpPost]
        public IActionResult UpdateAutomationSettings([FromBody] UpdateAutomationSettingsRequest? request)
        {
            if (request == null)
            {
                return BadRequest(new { success = false, message = "A request payload is required." });
            }

            var update = new AutomationSettingsUpdate(
                request.KellyDampener,
                request.MaxKellyFraction,
                request.MaxStakeMode,
                request.MaxStakeMode == MaxStakeMode.PercentageOfBankroll ? request.MaxStakePercent : null,
                request.MaxStakeMode == MaxStakeMode.FixedAmount ? request.MaxStakeAmount : null);

            var snapshot = _automationSettings.UpdateSettings(update);
            return Json(new
            {
                success = true,
                settings = new
                {
                    kellyDampener = snapshot.KellyDampener,
                    maxKellyFraction = snapshot.MaxKellyFraction,
                    maxStakeMode = snapshot.MaxStakeMode.ToString(),
                    maxStakePercent = snapshot.MaxStakePercentOfBankroll,
                    maxStakeAmount = snapshot.MaxStakeFixedAmount
                }
            });
        }
        [HttpGet]
        public IActionResult DayReportOptions(string? startTime = null, string? endTime = null, string? error = null, string? region = null)
        {
            var normalizedRegion = DayReportFilterViewModel.NormalizeRegion(region);
            var model = new DayReportFilterViewModel
            {
                StartTime = startTime,
                EndTime = endTime,
                ErrorMessage = error,
                Region = normalizedRegion
            };

            return View(model);
        }

        public async Task<IActionResult> DayReport(
            [FromServices] BetfairNavigationService betfair,
            [FromServices] HyperparameterTrainer trainer,
            string? startTime = null,
            string? endTime = null,
            string? region = null)
        {
            var normalizedRegion = DayReportFilterViewModel.NormalizeRegion(region);
            if (!TryParseTimeOfDay(startTime, out var startTimeSpan))
            {
                return View("DayReportOptions", new DayReportFilterViewModel
                {
                    StartTime = startTime,
                    EndTime = endTime,
                    ErrorMessage = "Start time must be in HH:MM format.",
                    Region = normalizedRegion
                });
            }

            if (!TryParseTimeOfDay(endTime, out var endTimeSpan))
            {
                return View("DayReportOptions", new DayReportFilterViewModel
                {
                    StartTime = startTime,
                    EndTime = endTime,
                    ErrorMessage = "End time must be in HH:MM format.",
                    Region = normalizedRegion
                });
            }

            if (startTimeSpan.HasValue && endTimeSpan.HasValue && startTimeSpan.Value > endTimeSpan.Value)
            {
                return View("DayReportOptions", new DayReportFilterViewModel
                {
                    StartTime = startTime,
                    EndTime = endTime,
                    ErrorMessage = "Start time must be earlier than or equal to the end time.",
                    Region = normalizedRegion
                });
            }
            await betfair.OpenHorseRaceMeetingsInNewTabsAsync(
                scheduleStartTime: startTimeSpan,
                scheduleEndTime: endTimeSpan,
                scheduleRegion: normalizedRegion);


            var report = betfair.GenerateDayReport(_repository, trainer);
            var usedNextDay = false;

            if (ShouldLoadNextDaySchedule(report, normalizedRegion))
            {
                _logger.LogInformation("Day report contains only USA races; attempting to load the next day's schedule.");
                var switched = await betfair.TrySelectHorseRacingDayAsync(1);
                if (switched)
                {
                    await betfair.OpenHorseRaceMeetingsInNewTabsAsync(
                        closeExistingRaceTabs: true,
                        scheduleStartTime: startTimeSpan,
                        scheduleEndTime: endTimeSpan);
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
            if (report != null)
            {
                ApplyTimeFilters(report, startTimeSpan, endTimeSpan);
                report.InitialRegion = normalizedRegion;
                ApplyDisplayedAiProbabilities(report);
            }
            else
            {
                report = new DayReportViewModel
                {
                    FilterStartTime = startTimeSpan,
                    FilterEndTime = endTimeSpan,
                    InitialRegion = normalizedRegion
                };
            }

            return View(report);
        }
        private static void ApplyDisplayedAiProbabilities(DayReportViewModel? report)
        {
            if (report?.Races == null)
            {
                return;
            }

            foreach (var race in report.Races)
            {
                if (race?.Runners == null || race.Runners.Count == 0)
                {
                    continue;
                }

                var candidates = new List<(RunnerDayReport Runner, double Probability, bool FromMarket)>(race.Runners.Count);

                foreach (var runner in race.Runners)
                {
                    if (runner == null)
                    {
                        continue;
                    }

                    var probability = 0d;
                    var fromMarket = false;
                    var aiProbability = runner.AiProbability;
                    var hasAiProbability = aiProbability.HasValue && double.IsFinite(aiProbability.Value) && aiProbability.Value > 0d;

                    if (hasAiProbability && aiProbability!.Value >= AiProbabilityDisplayThreshold)
                    {
                        probability = aiProbability.Value;
                    }
                    else if (runner.MarketProbability.HasValue && double.IsFinite(runner.MarketProbability.Value) && runner.MarketProbability.Value > 0d)
                    {
                        probability = runner.MarketProbability.Value;
                        fromMarket = true;
                    }
                    else if (hasAiProbability)
                    {
                        probability = aiProbability!.Value;
                    }
                    else if (runner.AiDecimalOdds.HasValue && runner.AiDecimalOdds.Value > 0m)
                    {
                        probability = 1.0 / (double)runner.AiDecimalOdds.Value;
                    }

                    if (probability < 0d || double.IsNaN(probability) || double.IsInfinity(probability))
                    {
                        probability = 0d;
                        fromMarket = false;
                    }

                    candidates.Add((runner, probability, fromMarket));
                }

                var total = candidates.Sum(entry => entry.Probability);
                if (total <= 0d || double.IsNaN(total) || double.IsInfinity(total))
                {
                    foreach (var entry in candidates)
                    {
                        entry.Runner.DisplayedAiProbability = null;
                        entry.Runner.DisplayedAiProbabilityMarketDerived = false;
                    }

                    continue;
                }

                var scale = 1d / total;
                foreach (var entry in candidates)
                {
                    var scaled = entry.Probability * scale;
                    if (scaled < 0d || double.IsNaN(scaled) || double.IsInfinity(scaled))
                    {
                        entry.Runner.DisplayedAiProbability = null;
                        entry.Runner.DisplayedAiProbabilityMarketDerived = false;
                        continue;
                    }

                    entry.Runner.DisplayedAiProbability = scaled;
                    entry.Runner.DisplayedAiProbabilityMarketDerived = entry.FromMarket;
                }
            }
        }

        private static bool TryParseTimeOfDay(string? value, out TimeSpan? time)
        {
            time = null;

            if (string.IsNullOrWhiteSpace(value))
            {
                return true;
            }

            if (TimeSpan.TryParseExact(value, new[] { "h\\:mm", "hh\\:mm" }, CultureInfo.InvariantCulture, out var parsed))
            {
                time = parsed;
                return true;
            }

            return false;
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
        private static void ApplyTimeFilters(DayReportViewModel report, TimeSpan? start, TimeSpan? end)
        {
            if (report.Races == null || report.Races.Count == 0)
            {
                report.FilterStartTime = start;
                report.FilterEndTime = end;
                return;
            }

            var hasStart = start.HasValue;
            var hasEnd = end.HasValue;

            if (!hasStart && !hasEnd)
            {
                report.FilterStartTime = null;
                report.FilterEndTime = null;
                return;
            }

            var effectiveStart = start ?? TimeSpan.Zero;
            var effectiveEnd = end ?? new TimeSpan(23, 59, 59);

            static TimeSpan? GetRaceTime(RaceDayReport race)
            {
                if (race.OffTime.HasValue)
                {
                    return race.OffTime.Value;
                }

                if (race.RaceDate.HasValue)
                {
                    return race.RaceDate.Value.TimeOfDay;
                }

                return null;
            }

            var filtered = report.Races
                .Where(race =>
                {
                    var raceTime = GetRaceTime(race);
                    if (!raceTime.HasValue)
                    {
                        return true;
                    }

                    if (hasStart && raceTime.Value < effectiveStart)
                    {
                        return false;
                    }

                    if (hasEnd && raceTime.Value > effectiveEnd)
                    {
                        return false;
                    }

                    return true;
                })
                .ToList();

            report.Races = filtered;
            report.FilterStartTime = start;
            report.FilterEndTime = end;
        }
        private static bool ShouldLoadNextDaySchedule(DayReportViewModel? report, string normalizedRegion)
        {
            if (!string.Equals(normalizedRegion, DayReportFilterViewModel.DefaultRegion, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
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
        public IActionResult AutomateBets(
            [FromServices] BetfairNavigationService betfair,
            [FromServices] HyperparameterTrainer trainer)
        {
            const string startingMessage = "Starting automated betting process...";
            _status.Update(startingMessage);

            var snapshot = _automationSettings.GetSnapshot();
            var model = new AutomateBetsViewModel
            {
                KellyDampener = snapshot.KellyDampener,
                MaxKellyFraction = snapshot.MaxKellyFraction,
                MaxStakeMode = snapshot.MaxStakeMode,
                MaxStakePercent = snapshot.MaxStakePercentOfBankroll.HasValue
                    ? snapshot.MaxStakePercentOfBankroll.Value * 100m
                    : (decimal?)null,
                MaxStakeAmount = snapshot.MaxStakeFixedAmount,
                StatusMessage = _status.Message,
                BannerMessage = startingMessage
            };

            _ = Task.Run(async () =>
            {
                try
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
                    var recommendations = betfair.ScrapeOpenRaceTabs(_repository, trainer, out var missingScrapeFields);
                    string message;
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

                        message = $"Best value bet: {horse}{venue} ({race})  odds {odds}, AI win {aiProb}% (AI return {aiReturn}) vs market {marketProb}% (diff {diff}%). Kelly stake {stake} ({kelly}% bankroll).";
                    }
                    else
                    {
                        message = "No positive expected value opportunities were found while scanning markets.";
                    }
                    if (missingScrapeFields != null && missingScrapeFields.Count > 0)
                    {
                        var ordered = missingScrapeFields
                            .Where(s => !string.IsNullOrWhiteSpace(s))
                            .Select(s => s.Trim())
                            .Where(s => s.Length > 0)
                            .Distinct(StringComparer.OrdinalIgnoreCase)
                            .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
                            .ToList();

                        if (ordered.Count > 0)
                        {
                            var details = string.Join(", ", ordered);
                            message += $" Some features were disabled because scraped data was unavailable for: {details}.";
                        }
                    }
                    _status.Update(message);
                }
                catch (Exception ex)
                {
                    _status.Update($"Failed to start automated betting: {ex.Message}");
                }
            });

            return View(model);
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
        [HttpPost]
        public async Task<IActionResult> RefreshRaceMarketOddsMarketOnly(
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
                var refreshed = await betfair.RefreshRaceAsync(
                    request.RaceUrl,
                    _repository,
                    trainer,
                    includeAiProbabilities: false);
                if (refreshed == null)
                {
                    return NotFound(new { success = false, message = "Unable to refresh market data for the selected race." });
                }

                var update = new RaceMarketOddsUpdate
                {
                    MarketId = refreshed.MarketId,
                    BackBookPercentage = refreshed.BackBookPercentage,
                    LayBookPercentage = refreshed.LayBookPercentage,
                    Runners = refreshed.Runners?.Select(runner => new RunnerMarketOddsUpdate
                    {
                        HorseName = runner.HorseName,
                        MarketDecimalOdds = runner.MarketDecimalOdds,
                        LayDecimalOdds = runner.LayDecimalOdds,
                        MarketProbability = runner.MarketProbability ??
                            (runner.MarketDecimalOdds.HasValue && runner.MarketDecimalOdds.Value > 0m
                                ? (double?)(1.0m / runner.MarketDecimalOdds.Value)
                                : null)
                    }).ToList() ?? new List<RunnerMarketOddsUpdate>()
                };

                return Json(new { success = true, update });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to refresh market-only odds for URL {RaceUrl}.", request.RaceUrl);
                return StatusCode(500, new { success = false, message = "An unexpected error occurred while refreshing the market odds." });
            }
        }
        public IActionResult Index()
        {
            ViewData["StatusMessage"] = _status.Message;
            return View();
        }
        [HttpPost]
        public async Task<IActionResult> RefreshRaceMarketOdds(
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
                bool shouldRecalculateAi = false;
                string? latestGoing = null;
                var marketId = request.MarketId?.Trim();
                if (!string.IsNullOrWhiteSpace(marketId))
                {
                    shouldRecalculateAi = betfair.ShouldRecalculateAiForMarket(marketId, request.CurrentGoing, out latestGoing);
                }

                var refreshed = await betfair.RefreshRaceAsync(
                    request.RaceUrl,
                    _repository,
                    trainer,
                    includeAiProbabilities: shouldRecalculateAi);
                if (refreshed == null)
                {
                    return NotFound(new { success = false, message = "Unable to refresh market data for the selected race." });
                }

                var update = new RaceMarketOddsUpdate
                {
                    MarketId = refreshed.MarketId,
                    BackBookPercentage = refreshed.BackBookPercentage,
                    LayBookPercentage = refreshed.LayBookPercentage,
                    Going = string.IsNullOrWhiteSpace(refreshed.Going) ? null : refreshed.Going.Trim(),
                    Runners = refreshed.Runners?.Select(runner => new RunnerMarketOddsUpdate
                    {
                        HorseName = runner.HorseName,
                        MarketDecimalOdds = runner.MarketDecimalOdds,
                        LayDecimalOdds = runner.LayDecimalOdds,
                        MarketProbability = runner.MarketProbability ??
                            (runner.MarketDecimalOdds.HasValue && runner.MarketDecimalOdds.Value > 0m
                                ? (double?)(1.0m / runner.MarketDecimalOdds.Value)
                                : null)
                    }).ToList() ?? new List<RunnerMarketOddsUpdate>()
                };

                var normalizedUpdatedGoing = string.IsNullOrWhiteSpace(update.Going) ? null : update.Going.Trim();
                if (normalizedUpdatedGoing == null && !string.IsNullOrWhiteSpace(latestGoing))
                {
                    normalizedUpdatedGoing = latestGoing.Trim();
                }
                var normalizedRequestGoing = string.IsNullOrWhiteSpace(request.CurrentGoing) ? null : request.CurrentGoing.Trim();
                var goingChanged = !string.Equals(normalizedUpdatedGoing, normalizedRequestGoing, StringComparison.OrdinalIgnoreCase);

                return Json(new
                {
                    success = true,
                    update,
                    goingChanged,
                    newGoing = normalizedUpdatedGoing
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to refresh market-only odds for URL {RaceUrl}.", request.RaceUrl);
                return StatusCode(500, new { success = false, message = "An unexpected error occurred while refreshing the market odds." });
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