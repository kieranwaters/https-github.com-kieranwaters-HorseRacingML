using System;
using System.Collections.Generic;
using System.Linq;
using Dapper;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Coverage report for the Racing TV sectionals and breeding tables. The
    /// backfills are resumable, so this is the quickest way to see which years
    /// still need a run.
    /// </summary>
    public sealed class RacingTvStatusService
    {
        private readonly string _connectionString;

        public RacingTvStatusService(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
        }

        public RacingTvStatusReport BuildReport()
        {
            using var connection = new SqlConnection(_connectionString);
            connection.Open();

            var sectional = connection.QuerySingle<RacingTvTableStats>(
                @"SELECT COUNT(*) AS [Rows],
                         COUNT(DISTINCT rs.RaceId) AS Races,
                         COUNT(DISTINCT rs.HorseId) AS Horses,
                         MIN(r.RaceDate) AS MinDate,
                         MAX(r.RaceDate) AS MaxDate
                  FROM dbo.RunnerSectional rs
                  JOIN dbo.Race r ON r.RaceId = rs.RaceId");

            var sectionalSources = connection.Query<RacingTvSourceStats>(
                @"SELECT Source,
                         COUNT(*) AS [Rows],
                         COUNT(DISTINCT RaceId) AS Races
                  FROM dbo.RunnerSectional
                  GROUP BY Source
                  ORDER BY COUNT(*) DESC").ToList();

            var pedigree = connection.QuerySingle<RacingTvTableStats>(
                @"SELECT COUNT(*) AS [Rows],
                         NULL AS Races,
                         COUNT(DISTINCT HorseId) AS Horses,
                         NULL AS MinDate,
                         NULL AS MaxDate
                  FROM dbo.HorsePedigree");

            var sectionalYears = connection.Query<RacingTvYearCoverage>(
                @"SELECT YEAR(r.RaceDate) AS [Year],
                         COUNT(DISTINCT r.RaceId) AS Races,
                         COUNT(DISTINCT CASE WHEN rs.RaceId IS NOT NULL THEN r.RaceId END) AS WithData
                  FROM dbo.Race r
                  LEFT JOIN dbo.RunnerSectional rs ON rs.RaceId = r.RaceId
                  WHERE r.RaceDate >= DATEFROMPARTS(2023, 1, 1)
                  GROUP BY YEAR(r.RaceDate)
                  ORDER BY [Year]").ToList();

            var pedigreeYears = connection.Query<RacingTvYearCoverage>(
                @"SELECT YEAR(r.RaceDate) AS [Year],
                         COUNT(DISTINCT rr.RaceId) AS Races,
                         COUNT(DISTINCT CASE WHEN hp.HorseId IS NOT NULL THEN rr.RaceId END) AS WithData
                  FROM dbo.Race r
                  JOIN dbo.RunnerResult rr ON rr.RaceId = r.RaceId
                  LEFT JOIN dbo.HorsePedigree hp ON hp.HorseId = rr.HorseId
                  WHERE r.RaceDate >= DATEFROMPARTS(2009, 1, 1)
                  GROUP BY YEAR(r.RaceDate)
                  ORDER BY [Year]").ToList();

            return new RacingTvStatusReport
            {
                Sectional = sectional,
                SectionalSources = sectionalSources,
                Pedigree = pedigree,
                SectionalYears = sectionalYears,
                PedigreeYears = pedigreeYears
            };
        }

        public void Print(Action<string> log)
        {
            if (log is null)
            {
                throw new ArgumentNullException(nameof(log));
            }

            var report = BuildReport();
            log("Racing TV coverage");
            log($"  RunnerSectional: {report.Sectional.Rows:N0} rows / {report.Sectional.Races:N0} races / " +
                $"{report.Sectional.Horses:N0} horses" + FormatRange(report.Sectional.MinDate, report.Sectional.MaxDate));
            foreach (var source in report.SectionalSources)
            {
                log($"    {source.Source}: {source.Rows:N0} rows / {source.Races:N0} races");
            }
            log($"  HorsePedigree:   {report.Pedigree.Rows:N0} horses" +
                (report.Pedigree.Horses > 0 ? $" ({report.Pedigree.Horses:N0} distinct)" : string.Empty));

            var missingSectionalYears = report.SectionalYears.Where(year => year.Missing > 0).ToList();
            if (missingSectionalYears.Count > 0)
            {
                log("Sectional coverage still missing:");
                foreach (var year in missingSectionalYears)
                {
                    log($"  {year.Year}: {year.WithData:N0}/{year.Races:N0} races ({year.Percent:0.0}%)");
                }
            }

            var missingPedigreeYears = report.PedigreeYears.Where(year => year.Missing > 0).ToList();
            if (missingPedigreeYears.Count > 0)
            {
                log("Breeding coverage still missing:");
                foreach (var year in missingPedigreeYears.Take(10))
                {
                    log($"  {year.Year}: {year.WithData:N0}/{year.Races:N0} races ({year.Percent:0.0}%)");
                }
                if (missingPedigreeYears.Count > 10)
                {
                    log($"  ... {missingPedigreeYears.Count - 10} more years");
                }
            }

            log("To fill gaps, rerun the resumable backfill, for example:");
            log("  dotnet run --project HorseRacingML -- --backfill-sectionals --from 2023-03-01");
            log("  dotnet run --project HorseRacingML -- --backfill-breeding --from 2009-01-01");
        }

        private static string FormatRange(DateTime? minDate, DateTime? maxDate)
        {
            return minDate.HasValue && maxDate.HasValue
                ? $" ({minDate:yyyy-MM-dd}..{maxDate:yyyy-MM-dd})"
                : string.Empty;
        }
    }

    public sealed class RacingTvStatusReport
    {
        public RacingTvTableStats Sectional { get; set; } = new();
        public IReadOnlyList<RacingTvSourceStats> SectionalSources { get; set; } = Array.Empty<RacingTvSourceStats>();
        public RacingTvTableStats Pedigree { get; set; } = new();
        public IReadOnlyList<RacingTvYearCoverage> SectionalYears { get; set; } = Array.Empty<RacingTvYearCoverage>();
        public IReadOnlyList<RacingTvYearCoverage> PedigreeYears { get; set; } = Array.Empty<RacingTvYearCoverage>();
    }

    public sealed class RacingTvSourceStats
    {
        public string Source { get; set; } = string.Empty;
        public int Rows { get; set; }
        public int Races { get; set; }
    }

    public sealed class RacingTvTableStats
    {
        public int Rows { get; set; }
        public int? Races { get; set; }
        public int Horses { get; set; }
        public DateTime? MinDate { get; set; }
        public DateTime? MaxDate { get; set; }
    }

    public sealed class RacingTvYearCoverage
    {
        public int Year { get; set; }
        public int Races { get; set; }
        public int WithData { get; set; }
        public int Missing => Math.Max(0, Races - WithData);
        public double Percent => Races == 0 ? 100d : 100d * WithData / Races;
    }
}
