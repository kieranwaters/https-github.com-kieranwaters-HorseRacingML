using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.TypeConversion;
using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace HorseRacingML.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly RacingRepository _repo;
        private readonly RaceResultsScraper _raceResultsScraper;
        private readonly BetfairNavigationService _navigationService;
        private readonly AutomationSettingsService _automationSettings;
        private readonly HyperparameterTrainer _trainer;
        private readonly ScrapingStatusService _statusService;

        public HomeController(
            ILogger<HomeController> logger,
            RacingRepository repo,
            RaceResultsScraper raceResultsScraper,
            BetfairNavigationService navigationService,
            AutomationSettingsService automationSettings,
            HyperparameterTrainer trainer,
            ScrapingStatusService statusService)
        {
            _logger = logger;
            _repo = repo;
            _raceResultsScraper = raceResultsScraper;
            _navigationService = navigationService;
            _automationSettings = automationSettings;
            _trainer = trainer;
            _statusService = statusService;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        public IActionResult ScrapeOptions()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> UploadCsv(IFormFile raceCsv, IFormFile runnerCsv, DateTime? selectionDate)
        {
            if (raceCsv == null && runnerCsv == null)
            {
                TempData["Message"] = "Please select at least one CSV file.";
                return RedirectToAction(nameof(ScrapeOptions));
            }

            try
            {
                await Task.Run(() => ProcessCsvFiles(raceCsv, runnerCsv, selectionDate));
                TempData["Message"] = "CSV files processed successfully.";
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing CSV files: {ex}");
                TempData["Message"] = $"Error processing CSV files: {ex.Message}";
            }

            return RedirectToAction(nameof(ScrapeOptions));
        }

        private void ProcessCsvFiles(IFormFile raceCsv, IFormFile runnerCsv, DateTime? selectionDate)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };

            var insertedRids = new List<int>();

            // 1. Parse and Insert Races (if provided)
            if (raceCsv != null)
            {
                using var raceReader = new StreamReader(raceCsv.OpenReadStream());
                using var raceCsvReader = new CsvReader(raceReader, config);

                // Add date format support for 'yy/MM/dd' (e.g. 90/01/01) and standard 'yyyy-MM-dd'
                var options = new TypeConverterOptions { Formats = new[] { "yy/MM/dd", "yyyy-MM-dd" } };
                raceCsvReader.Context.TypeConverterOptionsCache.AddOptions<DateTime>(options);

                var raceRecords = raceCsvReader.GetRecords<RaceCsvModel>().ToList();
                foreach (var r in raceRecords)
                {
                    if (selectionDate.HasValue && r.Date > selectionDate.Value)
                    {
                        continue;
                    }

                    // Course Lookup/Insert
                    var courseId = _repo.InsertCourse(new Course
                    {
                        Name = r.Course ?? "Unknown",
                        Country = r.CountryCode
                    });

                    // Class logic: use 'class' column, fallback to parsing 'rclass' if 0/missing
                    byte? raceClass = (byte?)r.Class;
                    if (raceClass == 0 && !string.IsNullOrWhiteSpace(r.RClass))
                    {
                        // Try parse from "Class 5" or just "5"
                        var digits = new string(r.RClass.Where(char.IsDigit).ToArray());
                        if (byte.TryParse(digits, out var parsedClass))
                        {
                            raceClass = parsedClass;
                        }
                    }
                    if (raceClass == 0) raceClass = null;

                    var race = new Race
                    {
                        Rid = r.Rid,
                        CourseId = courseId,
                        RaceDate = r.Date,
                        ScheduledOff = TimeSpan.TryParse(r.Time, out var t) ? t : (TimeSpan?)null,
                        Title = r.Title ?? string.Empty,
                        RaceType = r.Hurdles ?? string.Empty, // Mapping hurdles info to RaceType roughly
                        Class = raceClass,
                        AgeRestriction = r.Ages, // or Band? User said ages -> Ages allowed
                        Surface = r.Condition, // Mapping condition -> Going usually, user said condition -> Surface condition... wait.
                                               // User said: "condition - Surface condition" and "ncond - condition type (created from condition feature)".
                                               // And later in Q&A: "2) Mapping condition: You have a condition column and an ncond column. Which one should map to the Race table's Going and Surface columns? ... Current assumption: condition -> Going? User: yes"
                                               // So I map condition -> Going. Surface can be inferred or left null for now.
                        Going = r.Condition,
                        DistanceYards = (short)(r.Metric.HasValue ? r.Metric.Value * 1.09361 : 0), // Meters to Yards
                        DistanceText = r.Distance ?? string.Empty,
                        WinningTimeMs = r.WinningTime.HasValue ? (int)(r.WinningTime.Value * 1000) : null,
                        PrizeMoney = r.Prize
                    };

                    _repo.InsertRace(race);
                    insertedRids.Add(r.Rid);
                }
            }

            if (runnerCsv != null)
            {
                using var runnerReader = new StreamReader(runnerCsv.OpenReadStream());
                using var runnerCsvReader = new CsvReader(runnerReader, config);
                var runnerRecords = runnerCsvReader.GetRecords<RunnerCsvModel>().ToList();

                // 4. Resolve RaceIds
                // We need to map CSV rid -> Database RaceId
                var ridMap = _repo.GetRaceIdsByRids(insertedRids);

                // 5. Insert Runners
                var runnerResults = new List<RunnerResult>();
                foreach (var run in runnerRecords)
                {
                    if (!ridMap.TryGetValue(run.Rid, out int raceId))
                    {
                        // If race not found (maybe uploaded previously?), try to find it?
                        // For now, skip or log.
                        // We can also try fetching singular if not in batch map.
                        var ids = _repo.GetRaceIdsByRids(new[] { run.Rid });
                        if (ids.ContainsKey(run.Rid))
                        {
                            raceId = ids[run.Rid];
                            ridMap[run.Rid] = raceId;
                        }
                        else
                        {
                            continue;
                        }
                    }

                    // Entity Lookups
                    var horseId = _repo.InsertHorse(new Horse { Name = run.HorseName ?? "Unknown" });
                    short? trainerId = !string.IsNullOrWhiteSpace(run.TrainerName)
                        ? _repo.InsertTrainer(new Trainer { Name = run.TrainerName })
                        : (short?)null;
                    short? jockeyId = !string.IsNullOrWhiteSpace(run.JockeyName)
                        ? _repo.InsertJockey(new Jockey { Name = run.JockeyName })
                        : (short?)null;

                    // Decimal Price Conversion
                    decimal? spDecimal = null;
                    if (run.DecimalPrice.HasValue && run.DecimalPrice.Value != 0)
                    {
                        try
                        {
                            spDecimal = 1.0m / run.DecimalPrice.Value;
                            // Clamp to fit DECIMAL(8, 3) - Max 99999.999
                            if (spDecimal > 99999.999m) spDecimal = 99999.999m;
                            else if (spDecimal < -99999.999m) spDecimal = -99999.999m;
                        }
                        catch
                        {
                            spDecimal = null;
                        }
                    }

                    // Clamp DistanceBeatenLengths to avoid arithmetic overflow (Schema is DECIMAL(4, 2) -> Max 99.99)
                    decimal? dist = (decimal?)run.Dist;
                    if (dist.HasValue)
                    {
                        if (dist.Value > 99.99m) dist = 99.99m;
                        else if (dist.Value < -99.99m) dist = -99.99m;
                    }

                    // Clamp WeightLbs to 255 (tinyint max)
                    int rawWeight = (run.WeightSt * 14) + run.WeightLb;
                    byte? weightLbs = rawWeight > 255 ? (byte)255 : (byte?)rawWeight;

                    runnerResults.Add(new RunnerResult
                    {
                        RaceId = raceId,
                        Rid = run.Rid,
                        HorseId = horseId,
                        TrainerId = trainerId,
                        JockeyId = jockeyId,
                        SaddleclothNumber = (byte?)run.Saddle,
                        Age = (byte?)run.Age,
                        FinishPos = run.Position == 40 ? (short?)null : (short)run.Position,
                        OutcomeCode = run.Position == 40 ? "DNF" : null, // Assuming 40 means DNF
                        DistanceBeatenLengths = dist, // "dist - how far a horse has finished from a winner"
                        SP_Decimal = spDecimal,
                        FavTag = run.IsFav == 1 ? "F" : null, // Simple mapping
                        WeightLbs = weightLbs,
                        IsPlace = run.ResPlace == 1,
                        // Mappings for other fields if necessary
                    });
                }

                _repo.BulkInsertRunnerResults(runnerResults);
            }
        }

        public async Task<IActionResult> ScrapeRaceResults()
        {
            await Task.Run(() => _raceResultsScraper.ScrapeFromTodayBackwards());
            return RedirectToAction(nameof(ScrapeOptions));
        }

        public async Task<IActionResult> ScrapeEasternResults(DateTime startDate)
        {
            await Task.Run(() => _raceResultsScraper.ScrapeEasternFromDate(startDate));
            return RedirectToAction(nameof(ScrapeOptions));
        }

        public IActionResult DayReportOptions()
        {
            return View(new DayReportFilterViewModel());
        }

        // New action to start the background process
        public IActionResult DayReport(string startTime, string endTime, string region, bool showFeatureSignificance, bool runHeadless, bool useHybrid)
        {
            TimeSpan? start = null;
            TimeSpan? end = null;
            if (TimeSpan.TryParse(startTime, out var s)) start = s;
            if (TimeSpan.TryParse(endTime, out var e)) end = e;

            _statusService.Reset();
            _statusService.Update("Starting browser session...");

            // Run in background to prevent request timeout
            Task.Run(async () =>
            {
                try
                {
                    _navigationService.EnsureDriverMode(runHeadless);
                    await _navigationService.LoginAsync();

                    _statusService.Update("Finding race meetings...");
                    await _navigationService.OpenHorseRaceMeetingsInNewTabsAsync(
                        delayBetweenTabsMs: 0,
                        closeExistingRaceTabs: true,
                        scheduleStartTime: start,
                        scheduleEndTime: end,
                        scheduleRegion: region);

                    _statusService.Update("Scraping race data...");

                    var result = _navigationService.GenerateDayReport(
                        _repo,
                        _trainer,
                        showFeatureSignificance,
                        useHybrid,
                        progressCallback: (processed, total, message) =>
                        {
                            _statusService.UpdateProgress(processed, total, message);
                        });

                    var bankroll = _navigationService.GetEffectiveBankroll();
                    var model = new DayReportViewModel
                    {
                        GeneratedAt = DateTime.UtcNow,
                        Bankroll = bankroll,
                        AiHyperparameters = _trainer.LoadPersistedHyperparameters(),
                        Races = result.Races,
                        FilterStartTime = start,
                        FilterEndTime = end,
                        InitialRegion = region
                    };

                    _statusService.SetFinalReport(model);
                    _statusService.MarkComplete();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Background Day Report generation failed.");
                    _statusService.Update($"Error: {ex.Message}");
                }
            });

            return RedirectToAction(nameof(DayReportProgress));
        }

        public IActionResult DayReportProgress()
        {
            // Simple view that polls GetDayReportProgress
            // If the process is already complete (e.g. user refreshed), we can just render the live view which will redirect.
            return View("DayReportLive", new DayReportStatus
            {
                Message = _statusService.Message,
                IsComplete = _statusService.IsComplete
            });
        }

        [HttpGet]
        public IActionResult GetDayReportProgress()
        {
            var snapshot = _statusService.GetSnapshot();
            return Json(new DayReportStatus
            {
                Message = _statusService.Message,
                TotalRaces = _statusService.TotalRaces,
                ProcessedRaces = _statusService.ProcessedRaces,
                IsComplete = _statusService.IsComplete,
                // We don't necessarily need to return the full list of results here if we just want a progress bar
                // But keeping it for potential future live updates
                Results = snapshot.Select(r => new SimpleRaceResult
                {
                    RaceTitle = r.RaceTitle,
                    VenueName = r.VenueName,
                    RaceTime = r.OffTime?.ToString(@"hh\:mm") ?? "",
                    RaceUrl = r.RaceUrl
                }).ToList()
            });
        }

        public IActionResult DayReportResult()
        {
            var model = _statusService.GetFinalReport();
            if (model == null)
            {
                return RedirectToAction(nameof(DayReportOptions));
            }
            return View("DayReport", model);
        }

        public IActionResult AutomateBets()
        {
            var snapshot = _automationSettings.GetSnapshot();
            var model = new AutomateBetsViewModel
            {
                KellyDampener = snapshot.KellyDampener,
                MaxKellyFraction = snapshot.MaxKellyFraction,
                MaxStakeMode = snapshot.MaxStakeMode,
                MaxStakePercent = snapshot.MaxStakePercentOfBankroll.HasValue ? snapshot.MaxStakePercentOfBankroll.Value * 100m : null,
                MaxStakeAmount = snapshot.MaxStakeFixedAmount,
                StatusMessage = "Automation loop running..."
            };
            return View(model);
        }

        [HttpPost]
        public IActionResult UpdateAutomationSettings([FromBody] UpdateAutomationSettingsRequest request)
        {
            if (request == null) return BadRequest();

            var update = new AutomationSettingsUpdate(
                request.KellyDampener,
                request.MaxKellyFraction,
                request.MaxStakeMode,
                request.MaxStakePercent,
                request.MaxStakeAmount
            );

            _automationSettings.UpdateSettings(update);
            return Json(new { success = true, settings = _automationSettings.GetSnapshot() });
        }

        public IActionResult CalculateFavouritesAccuracy()
        {
            var (inc, exc, logLoss) = _repo.GetFavouriteAccuracy();
            var model = new FavouriteAccuracyViewModel
            {
                IncludingJoint = inc,
                ExcludingJoint = exc,
                LogLoss = logLoss
            };
            return View(model);
        }
    }
}
