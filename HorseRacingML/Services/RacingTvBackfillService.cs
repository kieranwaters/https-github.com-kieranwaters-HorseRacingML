using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Resumable backfill for the Sectionals and Breeding feature blocks.
    ///
    /// Sectionals: Racing TV / RaceiQ publishes free furlong splits for British
    /// races from March 2023 and Irish races from January 2024.
    ///
    /// Breeding: the Racing TV racecard for a historical meeting carries
    /// sire, dam and damsire for every declared runner, so one request covers
    /// a whole field. The public racecard archive starts on 1 January 2009.
    ///
    /// The source exposes the same JSON endpoints its public website uses.
    /// Requests are serialised by <see cref="RacingTvClient"/> and the run is
    /// intentionally restartable by date and race.
    /// </summary>
    public sealed class RacingTvBackfillService
    {
        private static readonly Regex BracketSuffix = new(
            @"\s*\([^)]*\)\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private readonly RacingTvClient _client;
        private readonly RacingTvRepository _repository;
        private readonly int _sectionalBackfillFromYear;

        public RacingTvBackfillService(
            RacingTvClient client,
            RacingTvRepository repository,
            IConfiguration configuration)
        {
            _client = client ?? throw new ArgumentNullException(nameof(client));
            _repository = repository ?? throw new ArgumentNullException(nameof(repository));
            _sectionalBackfillFromYear = configuration
                .GetValue<DateTime?>("RacingTv:SectionalBackfillFrom")?.Year ?? 2023;
        }

        public async Task<RacingTvBackfillSummary> BackfillAsync(
            DateTime fromDate,
            DateTime toDate,
            bool includeSectionals,
            bool includeBreeding,
            bool force,
            int? maxDays,
            Action<string>? log,
            CancellationToken cancellationToken = default)
        {
            if (!includeSectionals && !includeBreeding)
            {
                throw new ArgumentException("At least one of sectionals or breeding must be selected.");
            }

            var summary = new RacingTvBackfillSummary();
            DateTime start = fromDate.Date;
            DateTime end = toDate.Date > DateTime.Today ? DateTime.Today : toDate.Date;
            if (includeSectionals && start.Year < _sectionalBackfillFromYear)
            {
                start = new DateTime(_sectionalBackfillFromYear, 1, 1);
            }
            if (end < start)
            {
                log?.Invoke("Racing TV backfill: end date is before the start date.");
                return summary;
            }

            int processedDays = 0;
            for (DateTime date = start; date <= end; date = date.AddDays(1))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (maxDays.HasValue && processedDays >= maxDays.Value)
                {
                    break;
                }

                processedDays++;
                try
                {
                    await BackfillDateAsync(
                        date,
                        includeSectionals,
                        includeBreeding,
                        force,
                        summary,
                        log,
                        cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    summary.Warnings++;
                    log?.Invoke($"[racingtv] {date:yyyy-MM-dd}: {ex.Message}");
                }
            }

            log?.Invoke(
                $"[racingtv] finished: {summary.DaysProcessed} days, {summary.RacesMatched} races matched, " +
                $"{summary.SectionalRowsUpserted} sectional rows, {summary.PedigreeRowsUpserted} pedigree rows, " +
                $"{summary.RacesWithoutSourceData} races without source data, {summary.Warnings} warnings.");
            return summary;
        }

        private async Task BackfillDateAsync(
            DateTime date,
            bool includeSectionals,
            bool includeBreeding,
            bool force,
            RacingTvBackfillSummary summary,
            Action<string>? log,
            CancellationToken cancellationToken)
        {
            var results = await _client.GetResultsAsync(date, cancellationToken).ConfigureAwait(false);
            if (results is null || results.Meetings.Count == 0)
            {
                summary.DaysSkipped++;
                return;
            }

            var databaseRaces = _repository.GetRacesForDate(date);
            if (databaseRaces.Count == 0)
            {
                summary.DaysSkipped++;
                return;
            }

            var sectionalCoverage = includeSectionals
                ? _repository.GetSectionalCoverage(date)
                : new Dictionary<int, (int Rows, int Finishers)>();
            var pedigreeCoverage = includeBreeding
                ? _repository.GetPedigreeCoverage(date)
                : new Dictionary<int, (int Runners, int WithPedigree)>();

            int daySectionalRows = 0;
            int dayPedigreeRows = 0;
            int dayMatchedRaces = 0;
            var usedRaceIds = new HashSet<int>();

            foreach (var meeting in results.Meetings)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var track = meeting.Track;
                if (track is null || string.IsNullOrWhiteSpace(track.Slug))
                {
                    continue;
                }

                string country = track.CountryCode?.Racing ?? string.Empty;
                if (!country.Equals("GBR", StringComparison.OrdinalIgnoreCase) &&
                    !country.Equals("IRE", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                foreach (var sourceRace in meeting.CompletedRaces)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!sourceRace.StartTimeScheduled.HasValue)
                    {
                        continue;
                    }

                    var matchedRace = RacingTvDataParser.MatchRace(
                        databaseRaces,
                        track.Name,
                        sourceRace.StartTimeScheduled.Value,
                        sourceRace.Title);
                    if (matchedRace is null || !usedRaceIds.Add(matchedRace.RaceId))
                    {
                        continue;
                    }

                    // A race is only complete when every finisher has a sectional
                    // row. Checking "any row exists" left races that were interrupted
                    // mid-field, or where a runner was temporarily unmatched, stuck
                    // forever; compare against the finishers instead so a rerun
                    // repairs them. Non-finishers legitimately have no splits.
                    bool needsSectionals = includeSectionals &&
                        (force ||
                         !sectionalCoverage.TryGetValue(matchedRace.RaceId, out var coverage) ||
                         coverage.Rows == 0 ||
                         (coverage.Finishers > 0 && coverage.Rows < coverage.Finishers));
                    bool needsBreeding = includeBreeding &&
                        (force || !pedigreeCoverage.TryGetValue(matchedRace.RaceId, out var pedigree) ||
                         pedigree.Runners == 0 || pedigree.WithPedigree < pedigree.Runners);
                    if (!needsSectionals && !needsBreeding)
                    {
                        continue;
                    }

                    dayMatchedRaces++;
                    summary.RacesMatched++;
                    List<RaceRunnerReference>? runners = null;
                    List<RaceRunnerReference> DbRunners()
                        => runners ??= _repository.GetRunnersForRace(matchedRace.RaceId).ToList();

                    try
                    {
                        if (needsBreeding)
                        {
                            var racecard = await _client.GetRacecardAsync(
                                date,
                                track.Slug,
                                RacingTvDataParser.FormatLocalTime(sourceRace.StartTimeScheduled.Value),
                                cancellationToken).ConfigureAwait(false);
                            if (racecard is not null)
                            {
                                var pedigreeRows = BuildPedigreeRows(racecard, DbRunners());
                                if (pedigreeRows.Count > 0)
                                {
                                    int written = _repository.UpsertHorsePedigree(pedigreeRows);
                                    dayPedigreeRows += written;
                                    summary.PedigreeRowsUpserted += written;
                                }
                            }
                        }

                        if (needsSectionals)
                        {
                            var sectionalRace = await _client.GetSectionalsAsync(
                                date,
                                track.Slug,
                                RacingTvDataParser.FormatLocalTime(sourceRace.StartTimeScheduled.Value),
                                cancellationToken).ConfigureAwait(false);
                            if (sectionalRace?.Race is not null && sectionalRace.Race.Runners.Count > 0)
                            {
                                var sectionalRows = BuildSectionalRows(
                                    sectionalRace,
                                    matchedRace.RaceId,
                                    sourceRace.Id,
                                    DbRunners());
                                if (sectionalRows.Count > 0)
                                {
                                    int written = _repository.UpsertRunnerSectionals(sectionalRows);
                                    daySectionalRows += written;
                                    summary.SectionalRowsUpserted += written;
                                    summary.SectionalRacesUpserted++;
                                }
                                else
                                {
                                    summary.RacesWithoutSourceData++;
                                }
                            }
                            else
                            {
                                summary.RacesWithoutSourceData++;
                            }
                        }
                    }
                    catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        summary.Warnings++;
                        log?.Invoke(
                            $"[racingtv] {date:yyyy-MM-dd} {track.Name} {sourceRace.Title}: {ex.Message}");
                    }
                }
            }

            if (dayMatchedRaces > 0)
            {
                summary.DaysProcessed++;
                log?.Invoke(
                    $"[racingtv] {date:yyyy-MM-dd}: {dayMatchedRaces} races, " +
                    $"{daySectionalRows} sectional rows, {dayPedigreeRows} pedigree rows.");
            }
            else
            {
                summary.DaysSkipped++;
            }
        }

        private static List<HorsePedigree> BuildPedigreeRows(
            RacingTvRacecard racecard,
            IReadOnlyList<RaceRunnerReference> databaseRunners)
        {
            var rows = new List<HorsePedigree>();
            var seenHorseIds = new HashSet<int>();
            foreach (var sourceRunner in racecard.Runners)
            {
                if (sourceRunner is null || sourceRunner.Withdrawn)
                {
                    continue;
                }

                var sourceHorse = sourceRunner.Horse;
                if (sourceHorse is null ||
                    (string.IsNullOrWhiteSpace(sourceHorse.SireName) &&
                     string.IsNullOrWhiteSpace(sourceHorse.DamName)))
                {
                    continue;
                }

                var databaseRunner = RunnerMatcher.FindRunner(
                    databaseRunners,
                    sourceRunner.ClothNumber,
                    sourceRunner.HorseName);
                if (databaseRunner is null || !seenHorseIds.Add(databaseRunner.HorseId))
                {
                    continue;
                }

                rows.Add(new HorsePedigree
                {
                    HorseId = databaseRunner.HorseId,
                    SireName = CleanName(sourceHorse.SireName),
                    DamName = CleanName(sourceHorse.DamName),
                    DamSireName = CleanName(sourceHorse.DamSireName),
                    Sex = MapSex(sourceHorse.Gender),
                    Colour = sourceHorse.Colors is { Count: > 0 }
                        ? string.Join("/", sourceHorse.Colors.Where(value => !string.IsNullOrWhiteSpace(value)))
                        : null,
                    SourceHorseId = ToNullableInt(sourceHorse.Id ?? sourceRunner.HorseId),
                    SourceSlug = sourceHorse.Slug,
                    Source = "RacingTV",
                    FetchedUtc = DateTime.UtcNow
                });
            }

            return rows;
        }

        private static List<RunnerSectional> BuildSectionalRows(
            RacingTvSectionalRace sectionalRace,
            int raceId,
            long? sourceRaceId,
            IReadOnlyList<RaceRunnerReference> databaseRunners)
        {
            var rows = new List<RunnerSectional>();
            var race = sectionalRace.Race!;
            string? externalRaceId = sourceRaceId?.ToString(CultureInfo.InvariantCulture);
            foreach (var sourceRunner in race.Runners)
            {
                if (sourceRunner is null)
                {
                    continue;
                }

                var databaseRunner = RunnerMatcher.FindRunner(
                    databaseRunners,
                    sourceRunner.ClothNumber,
                    sourceRunner.HorseName);
                if (databaseRunner is null)
                {
                    continue;
                }

                var sectional = RacingTvDataParser.BuildRunnerSectional(
                    sourceRunner,
                    race,
                    raceId,
                    databaseRunner.HorseId,
                    externalRaceId);
                if (sectional is not null)
                {
                    rows.Add(sectional);
                }
            }

            return rows;
        }

        private static string? CleanName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return null;
            }

            string cleaned = BracketSuffix.Replace(value.Trim(), string.Empty).Trim();
            return cleaned.Length == 0 ? null : cleaned;
        }

        private static int? ToNullableInt(long? value)
            => value.HasValue && value.Value > 0 && value.Value <= int.MaxValue
                ? (int?)value.Value
                : null;

        private static string? MapSex(string? value)
        {
            return value?.Trim().ToLowerInvariant() switch
            {
                "c" => "Colt",
                "f" => "Filly",
                "g" => "Gelding",
                "m" => "Mare",
                null or "" => null,
                _ => value.Trim()
            };
        }
    }
}
