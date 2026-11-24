using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Fills sectional gaps from At The Races. ATR publishes the same TPD
    /// tracking data as the Racing TV results pages, but it also covers some
    /// British meetings that the RaceiQ archive does not. The importer only
    /// inserts missing runner rows; it never overwrites a Racing TV row.
    ///
    /// The site is browser-only because of its JavaScript client challenge, so
    /// this is deliberately a foreground CLI/backfill job rather than a
    /// high-frequency live feature.
    /// </summary>
    public sealed class AtTheRacesSectionalBackfillService
    {
        private static readonly TimeSpan TimeTolerance = TimeSpan.FromMinutes(8);
        private readonly AtTheRacesClient _client;
        private readonly RacingTvRepository _repository;
        private readonly DateTime _defaultFrom;

        public AtTheRacesSectionalBackfillService(
            AtTheRacesClient client,
            RacingTvRepository repository,
            IConfiguration configuration)
        {
            _client = client ?? throw new ArgumentNullException(nameof(client));
            _repository = repository ?? throw new ArgumentNullException(nameof(repository));
            _defaultFrom = configuration.GetValue<DateTime?>("AtTheRaces:BackfillFrom")
                ?? new DateTime(2023, 3, 1);
        }

        public DateTime DefaultFrom => _defaultFrom.Date;

        public AtTheRacesBackfillSummary Backfill(
            DateTime fromDate,
            DateTime toDate,
            int? maxDays,
            bool force,
            Action<string>? log,
            CancellationToken cancellationToken = default)
        {
            var summary = new AtTheRacesBackfillSummary();
            DateTime start = fromDate.Date;
            DateTime end = toDate.Date > DateTime.Today ? DateTime.Today : toDate.Date;
            if (end < start)
            {
                log?.Invoke("[atr] End date is before the start date.");
                return summary;
            }

            int processedDays = 0;
            int consecutiveResultsPageFailures = 0;
            for (DateTime date = start; date <= end; date = date.AddDays(1))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (maxDays.HasValue && processedDays >= maxDays.Value)
                {
                    break;
                }

                summary.DaysExamined++;
                var coverage = _repository.GetSectionalCoverage(date);
                var races = _repository.GetRacesForDate(date);
                var gaps = BuildGapList(races, coverage, force);
                if (gaps.Count == 0)
                {
                    continue;
                }

                processedDays++;
                summary.DaysProcessed++;
                IReadOnlyList<AtTheRacesRaceLink> links;
                try
                {
                    links = AtTheRacesSectionalParser.ParseResultsPage(_client.LoadResultsPage(date));
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    summary.Warnings++;
                    consecutiveResultsPageFailures++;
                    log?.Invoke($"[atr] {date:yyyy-MM-dd}: could not load results page: {ex.Message}");
                    if (consecutiveResultsPageFailures >= 3)
                    {
                        log?.Invoke(
                            "[atr] three consecutive results pages failed; stopping so ATR can " +
                            "cool down. Resume from the last successful date with a slower delay.");
                        break;
                    }
                    continue;
                }

                consecutiveResultsPageFailures = 0;
                if (links.Count == 0)
                {
                    summary.Warnings++;
                    summary.RacesWithoutSourceData += gaps.Count;
                    log?.Invoke($"[atr] {date:yyyy-MM-dd}: no ATR race links found.");
                    continue;
                }

                int dayMatched = 0;
                int dayRows = 0;
                int dayWithoutSource = 0;
                var usedAtrRaceIds = new HashSet<long>();
                foreach (var race in gaps)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var match = FindMatch(links, race, date, usedAtrRaceIds);
                    if (match is null)
                    {
                        continue;
                    }

                    usedAtrRaceIds.Add(match.RaceId);
                    dayMatched++;
                    summary.RacesMatched++;
                    try
                    {
                        var sourceRunners = AtTheRacesSectionalParser.ParseSectionalPage(
                            _client.LoadSectionalTimes(match));
                        if (sourceRunners.Count == 0)
                        {
                            dayWithoutSource++;
                            summary.RacesWithoutSourceData++;
                            continue;
                        }

                        IReadOnlyDictionary<string, double> finishPercentages =
                            new Dictionary<string, double>(StringComparer.Ordinal);
                        if (_client.FetchFinishSpeed)
                        {
                            try
                            {
                                finishPercentages = AtTheRacesSectionalParser.ParseFinishPercentages(
                                    _client.LoadSectionalTools(match));
                            }
                            catch (Exception ex) when (ex is not OperationCanceledException)
                            {
                                summary.Warnings++;
                            }
                        }

                        var rows = AtTheRacesSectionalParser.BuildRunnerSectionals(
                            sourceRunners,
                            _repository.GetRunnersForRace(race.RaceId),
                            race.RaceId,
                            match.RaceId,
                            finishPercentages);
                        if (rows.Count == 0)
                        {
                            dayWithoutSource++;
                            summary.RacesWithoutSourceData++;
                            continue;
                        }

                        int inserted = _repository.UpsertRunnerSectionals(rows, overwrite: false);
                        if (inserted > 0)
                        {
                            summary.RacesWithSourceData++;
                            summary.RowsInserted += inserted;
                            dayRows += inserted;
                        }
                        else
                        {
                            summary.RowsAlreadyPresent += rows.Count;
                        }
                    }
                    catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        summary.Warnings++;
                        summary.RacesWithoutSourceData++;
                        log?.Invoke(
                            $"[atr] {date:yyyy-MM-dd} {race.CourseName} {race.ScheduledOff}: {ex.Message}");
                    }
                }

                log?.Invoke(
                    $"[atr] {date:yyyy-MM-dd}: {dayMatched}/{gaps.Count} gaps matched, " +
                    $"{dayRows} rows inserted, {dayWithoutSource} without source data.");
            }

            log?.Invoke(
                $"[atr] finished: {summary.DaysProcessed} days, {summary.RacesMatched} races matched, " +
                $"{summary.RowsInserted} rows inserted, {summary.RowsAlreadyPresent} already present, " +
                $"{summary.RacesWithoutSourceData} without source data, {summary.Warnings} warnings.");
            return summary;
        }

        private static List<RaceBackfillCandidate> BuildGapList(
            IReadOnlyList<RaceBackfillCandidate> races,
            IReadOnlyDictionary<int, (int Rows, int Finishers)> coverage,
            bool force)
        {
            var gaps = new List<RaceBackfillCandidate>();
            foreach (var race in races)
            {
                if (!race.ScheduledOff.HasValue ||
                    string.IsNullOrWhiteSpace(race.TimeZoneId) ||
                    // ATR's tracking feed is British-only; we still include
                    // Northern Ireland because those courses use Europe/London.
                    !race.TimeZoneId.Equals("Europe/London", StringComparison.OrdinalIgnoreCase) ||
                    !coverage.TryGetValue(race.RaceId, out var raceCoverage))
                {
                    continue;
                }

                bool needsSectionals = force ||
                    raceCoverage.Rows == 0 ||
                    (raceCoverage.Finishers > 0 && raceCoverage.Rows < raceCoverage.Finishers);
                if (needsSectionals)
                {
                    gaps.Add(race);
                }
            }

            return gaps;
        }

        private static AtTheRacesRaceLink? FindMatch(
            IReadOnlyList<AtTheRacesRaceLink> links,
            RaceBackfillCandidate race,
            DateTime date,
            HashSet<long> usedAtrRaceIds)
        {
            if (!race.ScheduledOff.HasValue || links.Count == 0)
            {
                return null;
            }

            TimeZoneInfo timeZone;
            try
            {
                timeZone = TimeZoneInfo.FindSystemTimeZoneById(race.TimeZoneId!);
            }
            catch (TimeZoneNotFoundException)
            {
                return null;
            }
            catch (InvalidTimeZoneException)
            {
                return null;
            }

            var utc = DateTime.SpecifyKind(date.Date + race.ScheduledOff.Value, DateTimeKind.Utc);
            var local = TimeZoneInfo.ConvertTimeFromUtc(utc, timeZone);
            return links
                .Where(link => !usedAtrRaceIds.Contains(link.RaceId) &&
                               link.LocalDate.Date == local.Date &&
                               RacingTvDataParser.CourseNamesMatch(race.CourseName, link.Course) &&
                               Math.Abs((link.LocalTime - local.TimeOfDay).TotalSeconds) <=
                               TimeTolerance.TotalSeconds)
                .OrderBy(link => Math.Abs((link.LocalTime - local.TimeOfDay).TotalSeconds))
                .FirstOrDefault();
        }
    }
}
