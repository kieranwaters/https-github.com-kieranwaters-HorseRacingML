using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Applies the built-in course coordinates, downloads daily weather for the
    /// courses that have race history, and writes the per-race weather features.
    /// The operation is idempotent: weather rows are merged by (course, date)
    /// and race rows by RaceId, so an interrupted run can simply be repeated.
    /// </summary>
    public sealed class WeatherBackfillService
    {
        // Open-Meteo's archive starts in 1940 (ERA5). Clamp the discovered
        // history so a very old Race row cannot produce an invalid request.
        internal static readonly DateTime EarliestSupportedDate = new DateTime(1940, 1, 1);

        private readonly WeatherRepository _repository;
        private readonly OpenMeteoWeatherClient _client;
        private readonly IConfiguration _configuration;
        private readonly int _lookbackDays;
        private readonly int _pastDays;
        private readonly int _forecastDays;

        public WeatherBackfillService(
            WeatherRepository repository,
            OpenMeteoWeatherClient client,
            IConfiguration configuration)
        {
            _repository = repository ?? throw new ArgumentNullException(nameof(repository));
            _client = client ?? throw new ArgumentNullException(nameof(client));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _lookbackDays = Math.Clamp(configuration.GetValue<int?>("Weather:LookbackDays") ?? 20, 15, 30);
            _pastDays = Math.Clamp(configuration.GetValue<int?>("Weather:PastDays") ?? 14, 1, 92);
            _forecastDays = Math.Clamp(configuration.GetValue<int?>("Weather:ForecastDays") ?? 16, 2, 16);
        }

        /// <summary>
        /// Default start date for a backfill: the configured override when
        /// present, otherwise the earliest race date in the database (clamped
        /// to Open-Meteo's archive range). Falls back to 1 January 2010 when
        /// there are no races yet.
        /// </summary>
        public DateTime GetDefaultBackfillStart()
        {
            var configured = _configuration.GetValue<DateTime?>("Weather:BackfillFrom");
            if (configured.HasValue)
            {
                return configured.Value.Date;
            }

            var earliest = _repository.GetEarliestRaceDate()?.Date;
            if (!earliest.HasValue)
            {
                return new DateTime(2010, 1, 1);
            }

            return earliest.Value < EarliestSupportedDate ? EarliestSupportedDate : earliest.Value;
        }

        public int ApplyCourseLocations(Action<string>? log = null, CancellationToken cancellationToken = default)
        {
            var located = 0;
            var unmatched = new List<string>();

            foreach (var course in _repository.GetCourses())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!CourseLocationCatalog.TryMatch(course.Name, out var location))
                {
                    unmatched.Add(course.Name);
                    continue;
                }

                _repository.UpdateCourseLocation(
                    course.CourseId,
                    location.Latitude,
                    location.Longitude,
                    location.TimeZoneId);
                located++;
            }

            log?.Invoke($"Course locations applied: {located} matched, {unmatched.Count} unmatched.");
            if (unmatched.Count > 0)
            {
                log?.Invoke("Unmatched courses (weather features will stay null): " +
                            string.Join(", ", unmatched.Take(40)) +
                            (unmatched.Count > 40 ? ", ..." : string.Empty));
            }

            return located;
        }

        public WeatherBackfillSummary Backfill(
            DateTime fromDate,
            DateTime toDate,
            int? maxCourses = null,
            Action<string>? log = null,
            CancellationToken cancellationToken = default,
            WeatherBackfillMode mode = WeatherBackfillMode.Normal)
        {
            var summary = new WeatherBackfillSummary();
            var today = DateTime.Today;
            fromDate = fromDate.Date;
            toDate = toDate.Date > today ? today : toDate.Date;
            if (toDate < fromDate)
            {
                return summary;
            }

            summary.CoursesLocated = ApplyCourseLocations(log, cancellationToken);

            var courses = _repository.GetCourses()
                .Where(course => course.Latitude.HasValue && course.Longitude.HasValue)
                .OrderBy(course => course.CourseId)
                .ToList();

            if (maxCourses.HasValue && maxCourses.Value > 0)
            {
                courses = courses.Take(maxCourses.Value).ToList();
            }

            var warnings = new List<string>();
            foreach (var course in courses)
            {
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    var processed = BackfillCourse(
                        course,
                        fromDate,
                        toDate,
                        log,
                        cancellationToken,
                        mode,
                        out var daysWritten,
                        out var racesWritten);

                    if (!processed)
                    {
                        summary.CoursesSkipped++;
                        continue;
                    }

                    summary.CoursesProcessed++;
                    summary.WeatherDaysUpserted += daysWritten;
                    summary.RaceRowsUpserted += racesWritten;
                    log?.Invoke($"Weather backfill: {course.Name} ({racesWritten} races, {daysWritten} weather days).");
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (WeatherRateLimitException ex)
                {
                    // Stop cleanly rather than hammering the API for every
                    // remaining course. The run is resumable by design.
                    warnings.Add(ex.Message);
                    log?.Invoke("Weather backfill stopped: " + ex.Message);
                    break;
                }
                catch (Exception ex)
                {
                    summary.CoursesSkipped++;
                    var warning = $"{course.Name}: {ex.Message}";
                    warnings.Add(warning);
                    log?.Invoke("Weather backfill warning: " + warning);
                }
            }

            summary.Warnings = warnings;
            var action = mode == WeatherBackfillMode.CacheOnly ? "Weather feature rebuild" : "Weather backfill";
            log?.Invoke(
                $"{action} finished: {summary.CoursesProcessed} courses, " +
                $"{summary.RaceRowsUpserted} race rows, {summary.WeatherDaysUpserted} weather days, " +
                $"{summary.CoursesSkipped} skipped, {warnings.Count} warnings.");
            return summary;
        }

        /// <summary>
        /// Recomputes the per-race weather features from the daily rows already
        /// stored in CourseWeatherDaily. No HTTP calls are made, so this can fill
        /// derived columns (frost days, rain anomaly) when the Open-Meteo quota is
        /// exhausted. Rows whose daily cache is missing or clearly incomplete are
        /// left untouched rather than overwritten with nulls.
        /// </summary>
        public WeatherBackfillSummary PopulateFromCache(
            DateTime fromDate,
            DateTime toDate,
            int? maxCourses = null,
            Action<string>? log = null,
            CancellationToken cancellationToken = default)
        {
            return Backfill(
                fromDate,
                toDate,
                maxCourses,
                log,
                cancellationToken,
                WeatherBackfillMode.CacheOnly);
        }

        private bool BackfillCourse(
            Course course,
            DateTime fromDate,
            DateTime toDate,
            Action<string>? log,
            CancellationToken cancellationToken,
            WeatherBackfillMode mode,
            out int weatherDaysWritten,
            out int racesWritten)
        {
            weatherDaysWritten = 0;
            racesWritten = 0;

            // Read the whole history up to the end date so the previous meeting
            // and its going are available even when fromDate is mid-season.
            var races = _repository.GetRacesForCourse(course.CourseId, null, toDate);
            var inRange = races.Where(race => race.RaceDate.Date >= fromDate).ToList();
            if (inRange.Count == 0)
            {
                return false;
            }

            foreach (var yearGroup in inRange.GroupBy(race => race.RaceDate.Year).OrderBy(group => group.Key))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var yearRaces = yearGroup.OrderBy(race => race.RaceDate).ThenBy(race => race.ScheduledOff).ToList();
                var fetchStart = yearRaces.Min(race => race.RaceDate.Date).AddDays(-_lookbackDays);
                var fetchEnd = yearRaces.Max(race => race.RaceDate.Date).Date;

                // Resumability: a rerun after a rate-limit stop should not spend
                // quota on a course-year that is already complete. Refresh mode
                // deliberately bypasses this so it can pick up newly added raw
                // columns; cache-only mode always recomputes the derived features.
                if (mode == WeatherBackfillMode.Normal)
                {
                    var existing = _repository.GetRaceWeatherIds(
                        course.CourseId,
                        yearRaces[0].RaceDate.Date,
                        yearRaces[^1].RaceDate.Date);
                    if (yearRaces.All(race => existing.Contains(race.RaceId)))
                    {
                        continue;
                    }
                }

                IReadOnlyList<CourseWeatherDay>? cached = null;
                if (mode is WeatherBackfillMode.Refresh or WeatherBackfillMode.CacheOnly)
                {
                    cached = _repository.GetWeather(course.CourseId, fetchStart, fetchEnd);
                }

                // Refresh is resumable: once a course-year has the extended
                // variables (and all its race rows), a later run skips it. This
                // is what lets a quota-limited refresh advance course by course
                // instead of re-downloading the same head of the list every time.
                if (mode == WeatherBackfillMode.Refresh &&
                    HasExtendedWeather(cached!, fetchStart, fetchEnd))
                {
                    var existing = _repository.GetRaceWeatherIds(
                        course.CourseId,
                        yearRaces[0].RaceDate.Date,
                        yearRaces[^1].RaceDate.Date);
                    if (yearRaces.All(race => existing.Contains(race.RaceId)))
                    {
                        continue;
                    }
                }

                IReadOnlyList<CourseWeatherDay> weather;
                if (mode == WeatherBackfillMode.CacheOnly)
                {
                    weather = cached!;
                    if (!HasEnoughCachedDays(weather.Count, fetchStart, fetchEnd))
                    {
                        // Silence the common "course has no cache for this year"
                        // case; only warn when a partial slice was actually found.
                        if (weather.Count > 0)
                        {
                            log?.Invoke(
                                $"  {course.Name} {yearGroup.Key}: cached weather incomplete " +
                                $"({weather.Count} days); skipped.");
                        }

                        continue;
                    }
                }
                else
                {
                    weather = FetchDaily(course, fetchStart, fetchEnd, log);
                    if (weather.Count == 0)
                    {
                        log?.Invoke($"  {course.Name} {yearGroup.Key}: no weather returned; skipped.");
                        continue;
                    }

                    weatherDaysWritten += _repository.UpsertWeather(weather);
                }

                var dayMap = BuildDayMap(weather);
                var (rainNormal14d, soilMoistureNormal) =
                    _repository.GetWeatherNormals(course.CourseId, yearRaces[0].RaceDate.Date);
                var rows = new List<RaceWeatherFeatures>(yearRaces.Count);

                // Seed the state with the last meeting before this year's first
                // race, otherwise January races look like they follow a break
                // of unknown length and lose their previous-going signal.
                var firstRaceDate = yearRaces[0].RaceDate.Date;
                var priorRace = races
                    .Where(race => race.RaceDate.Date < firstRaceDate)
                    .OrderBy(race => race.RaceDate)
                    .ThenBy(race => race.ScheduledOff)
                    .LastOrDefault();
                var lastRaceDate = priorRace?.RaceDate.Date;
                var lastGoing = priorRace?.Going;

                foreach (var race in races.Where(race => race.RaceDate.Date <= toDate))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int? daysSinceLastMeeting = lastRaceDate.HasValue
                        ? Math.Max(0, (race.RaceDate.Date - lastRaceDate.Value.Date).Days)
                        : null;

                    if (race.RaceDate.Date >= fromDate && race.RaceDate.Year == yearGroup.Key)
                    {
                        var features = WeatherFeatureCalculator.Compute(
                            race.RaceId,
                            course.CourseId,
                            race.RaceDate,
                            race.ScheduledOff,
                            race.Surface,
                            lastGoing,
                            daysSinceLastMeeting,
                            dayMap,
                            includeRaceDayRain: false,
                            source: weather.Any(day => day.Source == "archive") ? "archive" : "forecast",
                            rainNormal14d: rainNormal14d,
                            soilMoistureNormal: soilMoistureNormal);

                        rows.Add(features);
                    }

                    lastRaceDate = race.RaceDate.Date;
                    if (!string.IsNullOrWhiteSpace(race.Going))
                    {
                        lastGoing = race.Going;
                    }
                }

                foreach (var batch in Batch(rows, 500))
                {
                    racesWritten += _repository.UpsertRaceWeather(batch);
                }
            }

            return true;
        }

        /// <summary>
        /// A cached year is safe to rebuild when it covers almost the whole fetch
        /// window. This prevents a sparse slice (for example a single meeting)
        /// from overwriting previously complete race features with nulls.
        /// </summary>
        private static bool HasEnoughCachedDays(int cachedDays, DateTime startDate, DateTime endDate)
        {
            if (cachedDays == 0)
            {
                return false;
            }

            var expectedDays = Math.Max(1, (endDate.Date - startDate.Date).Days + 1);
            var requiredDays = Math.Max(14, (int)Math.Ceiling(expectedDays * 0.9));
            return cachedDays >= requiredDays;
        }

        /// <summary>
        /// True when the cached window already contains the extended variables.
        /// Wind direction is the marker because it is only present when the
        /// expanded Open-Meteo variable list was requested. Used to make refresh
        /// runs resumable after a rate-limit stop.
        /// </summary>
        private static bool HasExtendedWeather(
            IReadOnlyList<CourseWeatherDay> days,
            DateTime startDate,
            DateTime endDate)
        {
            if (days.Count == 0 || !HasEnoughCachedDays(days.Count, startDate, endDate))
            {
                return false;
            }

            var expectedDays = Math.Max(1, (endDate.Date - startDate.Date).Days + 1);
            var requiredDays = Math.Max(14, (int)Math.Ceiling(expectedDays * 0.9));
            var withWindDirection = days.Count(day => day.WindDirectionDominant.HasValue);
            return withWindDirection >= requiredDays;
        }

        private IReadOnlyList<CourseWeatherDay> FetchDaily(
            Course course,
            DateTime startDate,
            DateTime endDate,
            Action<string>? log)
        {
            var today = DateTime.Today;
            if (endDate < startDate || startDate > today.AddDays(_forecastDays))
            {
                return Array.Empty<CourseWeatherDay>();
            }

            var combined = new Dictionary<DateTime, CourseWeatherDay>();
            var archiveEnd = endDate < today.AddDays(-5) ? endDate : today.AddDays(-5);

            if (startDate <= archiveEnd)
            {
                var archive = _client.GetArchive(
                    course.CourseId,
                    course.Latitude!.Value,
                    course.Longitude!.Value,
                    course.TimeZoneId,
                    startDate,
                    archiveEnd);
                foreach (var day in archive)
                {
                    combined[day.WeatherDate.Date] = day;
                }
                log?.Invoke($"  {course.Name}: {archive.Count} archive days {startDate:yyyy-MM-dd}..{archiveEnd:yyyy-MM-dd}.");
            }

            var forecastStart = startDate > today.AddDays(-_pastDays) ? startDate : today.AddDays(-_pastDays);
            var forecastEnd = endDate < today.AddDays(_forecastDays) ? endDate : today.AddDays(_forecastDays);
            if (forecastStart <= forecastEnd)
            {
                var forecast = _client.GetForecast(
                    course.CourseId,
                    course.Latitude!.Value,
                    course.Longitude!.Value,
                    course.TimeZoneId);

                foreach (var day in forecast)
                {
                    var date = day.WeatherDate.Date;
                    if (date < forecastStart || date > forecastEnd)
                    {
                        continue;
                    }

                    // Archive data is authoritative for past dates; only fill
                    // gaps with the forecast product.
                    if (!combined.ContainsKey(date))
                    {
                        combined[date] = day;
                    }
                }
            }

            return combined.Values.OrderBy(day => day.WeatherDate).ToList();
        }

        private static IReadOnlyDictionary<DateTime, CourseWeatherDay> BuildDayMap(
            IReadOnlyList<CourseWeatherDay> days)
        {
            var map = new Dictionary<DateTime, CourseWeatherDay>();
            foreach (var day in days)
            {
                var date = day.WeatherDate.Date;
                if (!map.TryGetValue(date, out var existing) ||
                    (existing.Source != "archive" && day.Source == "archive"))
                {
                    map[date] = day;
                }
            }

            return map;
        }

        internal static IEnumerable<List<T>> Batch<T>(IEnumerable<T> source, int size)
        {
            var batch = new List<T>(size);
            foreach (var item in source)
            {
                batch.Add(item);
                if (batch.Count >= size)
                {
                    yield return batch;
                    batch = new List<T>(size);
                }
            }

            if (batch.Count > 0)
            {
                yield return batch;
            }
        }
    }
}
