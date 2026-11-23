using System;
using System.Collections.Generic;
using System.Linq;
using Dapper;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Reports how much of the race history has weather rows. The backfill is
    /// resumable, so this is the quickest way to see which years or courses
    /// still need a run.
    /// </summary>
    public sealed class WeatherStatusService
    {
        private readonly string _connectionString;

        public WeatherStatusService(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("HorseRacingDb")
                ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' not found.");
        }

        public WeatherStatusReport BuildReport()
        {
            using var connection = new SqlConnection(_connectionString);
            connection.Open();

            var daily = connection.QuerySingle<WeatherTableStats>(
                @"SELECT COUNT(*) AS [Rows],
                         COUNT(DISTINCT CourseId) AS Courses,
                         MIN(WeatherDate) AS MinDate,
                         MAX(WeatherDate) AS MaxDate
                  FROM dbo.CourseWeatherDaily");

            var races = connection.QuerySingle<WeatherTableStats>(
                @"SELECT COUNT(*) AS [Rows],
                         COUNT(DISTINCT CourseId) AS Courses,
                         MIN(WeatherDate) AS MinDate,
                         MAX(WeatherDate) AS MaxDate
                  FROM dbo.RaceWeather");

            var extended = connection.QuerySingle<WeatherExtendedCoverage>(
                @"SELECT COUNT(*) AS [Rows],
                         COUNT(WeatherWindDirectionDegrees) AS WithWindDirection,
                         COUNT(WeatherHumidity) AS WithHumidity,
                         COUNT(WeatherSoilMoisture0To7) AS WithSoilMoisture,
                         COUNT(WeatherRainAnomaly14dMm) AS WithRainAnomaly,
                         COUNT(WeatherFrostDaysPrev7) AS WithFrostDays
                  FROM dbo.RaceWeather");

            var years = connection.Query<WeatherYearCoverage>(
                @"SELECT YEAR(r.RaceDate) AS [Year],
                         COUNT(*) AS Races,
                         COUNT(rw.RaceId) AS WithWeather
                  FROM dbo.Race r
                  LEFT JOIN dbo.RaceWeather rw ON rw.RaceId = r.RaceId
                  GROUP BY YEAR(r.RaceDate)
                  ORDER BY [Year]").ToList();

            var gaps = connection.Query<WeatherCourseGap>(
                @"SELECT c.CourseId,
                         c.Name,
                         COUNT(r.RaceId) AS Races,
                         COUNT(rw.RaceId) AS WithWeather,
                         CASE WHEN c.Latitude IS NULL OR c.Longitude IS NULL
                              THEN CAST(0 AS BIT) ELSE CAST(1 AS BIT) END AS HasLocation
                  FROM dbo.Course c
                  JOIN dbo.Race r ON r.CourseId = c.CourseId
                  LEFT JOIN dbo.RaceWeather rw ON rw.RaceId = r.RaceId
                  GROUP BY c.CourseId, c.Name, c.Latitude, c.Longitude
                  HAVING COUNT(rw.RaceId) = 0
                  ORDER BY COUNT(r.RaceId) DESC").ToList();

            var unlocated = connection.Query<string>(
                @"SELECT c.Name
                  FROM dbo.Course c
                  WHERE (c.Latitude IS NULL OR c.Longitude IS NULL)
                    AND EXISTS (SELECT 1 FROM dbo.Race r WHERE r.CourseId = c.CourseId)
                  ORDER BY c.Name").ToList();

            return new WeatherStatusReport
            {
                Daily = daily,
                Races = races,
                Extended = extended,
                Years = years,
                CourseGaps = gaps,
                UnlocatedCourses = unlocated
            };
        }

        public void Print(Action<string> log)
        {
            if (log is null)
            {
                throw new ArgumentNullException(nameof(log));
            }

            var report = BuildReport();
            log("Weather coverage");
            log($"  CourseWeatherDaily: {report.Daily.Rows:N0} rows / {report.Daily.Courses:N0} courses" +
                FormatRange(report.Daily.MinDate, report.Daily.MaxDate));
            log($"  RaceWeather:        {report.Races.Rows:N0} rows / {report.Races.Courses:N0} courses" +
                FormatRange(report.Races.MinDate, report.Races.MaxDate));
            log($"  Race rows with weather: {report.TotalWithWeather:N0} / {report.TotalRaces:N0} " +
                $"({report.CoveragePercent:0.0}%)");
            log($"  Extended features: wind {report.Extended.WithWindDirection:N0}, " +
                $"humidity {report.Extended.WithHumidity:N0}, " +
                $"soil {report.Extended.WithSoilMoisture:N0}, " +
                $"rain anomaly {report.Extended.WithRainAnomaly:N0}, " +
                $"frost {report.Extended.WithFrostDays:N0} of {report.Extended.Rows:N0} rows");

            var missingYears = report.Years.Where(year => year.Missing > 0).ToList();
            if (missingYears.Count == 0)
            {
                log("  Every year with race rows has weather rows.");
            }
            else
            {
                log("Years still missing weather:");
                foreach (var year in missingYears)
                {
                    log($"  {year.Year}: {year.WithWeather:N0}/{year.Races:N0} " +
                        $"({year.Percent:0.0}%), {year.Missing:N0} missing");
                }
            }

            if (report.CourseGaps.Count > 0)
            {
                log("Courses with races but no weather rows:");
                foreach (var gap in report.CourseGaps)
                {
                    var location = gap.HasLocation ? string.Empty : " [no coordinates]";
                    log($"  {gap.Name}: {gap.Races:N0} races{location}");
                }
            }

            if (report.UnlocatedCourses.Count > 0)
            {
                log("Unlocated courses: " + string.Join(", ", report.UnlocatedCourses));
            }

            log("To fill gaps, rerun the resumable backfill for a date range, for example:");
            log("  dotnet run --project HorseRacingML -- --backfill-weather --from 2015-01-01 --to 2026-12-31");
        }

        private static string FormatRange(DateTime? minDate, DateTime? maxDate)
        {
            return minDate.HasValue && maxDate.HasValue
                ? $" ({minDate:yyyy-MM-dd}..{maxDate:yyyy-MM-dd})"
                : string.Empty;
        }
    }

    public sealed class WeatherTableStats
    {
        public int Rows { get; set; }
        public int Courses { get; set; }
        public DateTime? MinDate { get; set; }
        public DateTime? MaxDate { get; set; }
    }

    public sealed class WeatherExtendedCoverage
    {
        public int Rows { get; set; }
        public int WithWindDirection { get; set; }
        public int WithHumidity { get; set; }
        public int WithSoilMoisture { get; set; }
        public int WithRainAnomaly { get; set; }
        public int WithFrostDays { get; set; }
    }

    public sealed class WeatherYearCoverage
    {
        public int Year { get; set; }
        public int Races { get; set; }
        public int WithWeather { get; set; }
        public int Missing => Math.Max(0, Races - WithWeather);
        public double Percent => Races == 0 ? 100d : 100d * WithWeather / Races;
    }

    public sealed class WeatherCourseGap
    {
        public short CourseId { get; set; }
        public string Name { get; set; } = string.Empty;
        public int Races { get; set; }
        public int WithWeather { get; set; }
        public bool HasLocation { get; set; }
    }

    public sealed class WeatherStatusReport
    {
        public WeatherTableStats Daily { get; set; } = new();
        public WeatherTableStats Races { get; set; } = new();
        public WeatherExtendedCoverage Extended { get; set; } = new();
        public IReadOnlyList<WeatherYearCoverage> Years { get; set; } = Array.Empty<WeatherYearCoverage>();
        public IReadOnlyList<WeatherCourseGap> CourseGaps { get; set; } = Array.Empty<WeatherCourseGap>();
        public IReadOnlyList<string> UnlocatedCourses { get; set; } = Array.Empty<string>();

        public int TotalRaces => Years.Sum(year => year.Races);
        public int TotalWithWeather => Years.Sum(year => year.WithWeather);
        public double CoveragePercent => TotalRaces == 0 ? 100d : 100d * TotalWithWeather / TotalRaces;
    }
}
