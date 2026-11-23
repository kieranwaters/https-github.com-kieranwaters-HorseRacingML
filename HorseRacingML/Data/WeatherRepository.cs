using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Dapper;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;

namespace HorseRacingML.Data
{
    /// <summary>
    /// Persistence for course coordinates, daily weather and the per-race
    /// weather feature rows. Kept separate from RacingRepository so the weather
    /// backfill can be run without loading the scraping dependencies.
    /// </summary>
    public sealed class WeatherRepository
    {
        private readonly string _connectionString;

        public WeatherRepository(string connectionString)
        {
            _connectionString = string.IsNullOrWhiteSpace(connectionString)
                ? throw new ArgumentException("A database connection string is required.", nameof(connectionString))
                : connectionString;
        }

        private SqlConnection OpenConnection()
        {
            var connection = new SqlConnection(_connectionString);
            connection.Open();
            return connection;
        }

        public IReadOnlyList<Course> GetCourses()
        {
            using var connection = OpenConnection();
            return connection.Query<Course>(
                @"SELECT CourseId, Name, Country, Latitude, Longitude, TimeZoneId
                  FROM dbo.Course
                  ORDER BY CourseId").ToList();
        }

        public int UpdateCourseLocation(short courseId, double latitude, double longitude, string? timeZoneId)
        {
            using var connection = OpenConnection();
            return connection.Execute(
                @"UPDATE dbo.Course
                  SET Latitude = @Latitude,
                      Longitude = @Longitude,
                      TimeZoneId = @TimeZoneId
                  WHERE CourseId = @CourseId",
                new { CourseId = courseId, Latitude = latitude, Longitude = longitude, TimeZoneId = timeZoneId });
        }

        public IReadOnlyList<CourseWeatherDay> GetWeather(short courseId, DateTime startDate, DateTime endDate)
        {
            using var connection = OpenConnection();
            return connection.Query<CourseWeatherDay>(
                @"SELECT CourseId, WeatherDate, PrecipitationMm, RainMm, TempMaxC, TempMinC,
                         WindMaxKph, WindGustMaxKph, WindDirectionDominant,
                         RelativeHumidityMean, DewPointMean, PrecipitationHours,
                         SnowfallCm, SoilMoisture0To7, SoilMoisture0To28,
                         Et0Mm, Source, FetchedUtc
                  FROM dbo.CourseWeatherDaily
                  WHERE CourseId = @CourseId
                    AND WeatherDate >= @StartDate
                    AND WeatherDate <= @EndDate
                  ORDER BY WeatherDate",
                new { CourseId = courseId, StartDate = startDate.Date, EndDate = endDate.Date }).ToList();
        }

        /// <summary>
        /// Ten-year calendar-window normals for the 14 days before a race date.
        /// Used to express rainfall and soil moisture as anomalies rather than
        /// absolute values. Returns null when the course has too little history.
        /// </summary>
        private sealed class WeatherNormals
        {
            public double? Rain14d { get; set; }
            public double? SoilMoisture { get; set; }
        }

        public (double? Rain14d, double? SoilMoisture) GetWeatherNormals(
            short courseId,
            DateTime raceDate,
            int years = 10)
        {
            using var connection = OpenConnection();
            var normal = connection.QueryFirstOrDefault<WeatherNormals>(
                @"
WITH windows AS
(
    SELECT YEAR(WeatherDate) AS Y,
           SUM(RainMm) AS Rain14d,
           AVG(SoilMoisture0To7) AS SoilMoisture
    FROM dbo.CourseWeatherDaily
    WHERE CourseId = @CourseId
      AND WeatherDate < @RaceDate
      AND YEAR(WeatherDate) >= YEAR(@RaceDate) - @Years
      AND ((DATEPART(dayofyear, WeatherDate) - DATEPART(dayofyear, @RaceDate) + 3650) % 3650) BETWEEN 1 AND 14
    GROUP BY YEAR(WeatherDate)
)
SELECT AVG(Rain14d) AS Rain14d, AVG(SoilMoisture) AS SoilMoisture
FROM windows
HAVING COUNT(*) >= 3",
                new { CourseId = courseId, RaceDate = raceDate.Date, Years = Math.Clamp(years, 3, 30) });

            return (normal?.Rain14d, normal?.SoilMoisture);
        }

        public int UpsertWeather(IEnumerable<CourseWeatherDay> days)
        {
            var rows = days?.Where(day => day is not null).ToList() ?? new List<CourseWeatherDay>();
            if (rows.Count == 0)
            {
                return 0;
            }

            const string sql = @"
MERGE dbo.CourseWeatherDaily AS target
USING (SELECT @CourseId AS CourseId, @WeatherDate AS WeatherDate) AS source
ON target.CourseId = source.CourseId AND target.WeatherDate = source.WeatherDate
WHEN MATCHED AND (@Source = 'archive' OR target.Source IS NULL OR target.Source <> 'archive') THEN
    UPDATE SET PrecipitationMm = @PrecipitationMm,
               RainMm = @RainMm,
               TempMaxC = @TempMaxC,
               TempMinC = @TempMinC,
               WindMaxKph = @WindMaxKph,
               WindGustMaxKph = @WindGustMaxKph,
               WindDirectionDominant = COALESCE(@WindDirectionDominant, target.WindDirectionDominant),
               RelativeHumidityMean = COALESCE(@RelativeHumidityMean, target.RelativeHumidityMean),
               DewPointMean = COALESCE(@DewPointMean, target.DewPointMean),
               PrecipitationHours = COALESCE(@PrecipitationHours, target.PrecipitationHours),
               SnowfallCm = COALESCE(@SnowfallCm, target.SnowfallCm),
               SoilMoisture0To7 = COALESCE(@SoilMoisture0To7, target.SoilMoisture0To7),
               SoilMoisture0To28 = COALESCE(@SoilMoisture0To28, target.SoilMoisture0To28),
               Et0Mm = @Et0Mm,
               Source = @Source,
               FetchedUtc = SYSUTCDATETIME()
WHEN NOT MATCHED THEN
    INSERT (CourseId, WeatherDate, PrecipitationMm, RainMm, TempMaxC, TempMinC,
            WindMaxKph, WindGustMaxKph, WindDirectionDominant, RelativeHumidityMean,
            DewPointMean, PrecipitationHours, SnowfallCm, SoilMoisture0To7,
            SoilMoisture0To28, Et0Mm, Source, FetchedUtc)
    VALUES (@CourseId, @WeatherDate, @PrecipitationMm, @RainMm, @TempMaxC, @TempMinC,
            @WindMaxKph, @WindGustMaxKph, @WindDirectionDominant, @RelativeHumidityMean,
            @DewPointMean, @PrecipitationHours, @SnowfallCm, @SoilMoisture0To7,
            @SoilMoisture0To28, @Et0Mm, @Source, SYSUTCDATETIME());";

            using var connection = OpenConnection();
            using var transaction = connection.BeginTransaction();
            try
            {
                var affected = connection.Execute(sql, rows, transaction);
                transaction.Commit();
                return affected;
            }
            catch
            {
                transaction.Rollback();
                throw;
            }
        }

        public int UpsertRaceWeather(IEnumerable<RaceWeatherFeatures> rows)
        {
            var items = rows?.Where(row => row is not null).ToList() ?? new List<RaceWeatherFeatures>();
            if (items.Count == 0)
            {
                return 0;
            }

            const string sql = @"
MERGE dbo.RaceWeather AS target
USING (SELECT @RaceId AS RaceId) AS source
ON target.RaceId = source.RaceId
WHEN MATCHED THEN
    UPDATE SET CourseId = @CourseId,
               WeatherDate = @WeatherDate,
               -- Null means not available in this pass; keep any value a
               -- refresh previously populated rather than erasing it. This is
               -- what lets a cache-only rebuild run safely after a refresh.
               RainPrev1dMm = COALESCE(@RainPrev1dMm, target.RainPrev1dMm),
               RainPrev3dMm = COALESCE(@RainPrev3dMm, target.RainPrev3dMm),
               RainPrev7dMm = COALESCE(@RainPrev7dMm, target.RainPrev7dMm),
               RainPrev14dMm = COALESCE(@RainPrev14dMm, target.RainPrev14dMm),
               RainRaceDayMm = COALESCE(@RainRaceDayMm, target.RainRaceDayMm),
               WeatherTempMaxC = COALESCE(@WeatherTempMaxC, target.WeatherTempMaxC),
               WeatherTempMinC = COALESCE(@WeatherTempMinC, target.WeatherTempMinC),
               WeatherWindMaxKph = COALESCE(@WeatherWindMaxKph, target.WeatherWindMaxKph),
               WeatherWindDirectionDegrees = COALESCE(@WeatherWindDirectionDegrees, target.WeatherWindDirectionDegrees),
               WeatherHumidity = COALESCE(@WeatherHumidity, target.WeatherHumidity),
               WeatherDewPointC = COALESCE(@WeatherDewPointC, target.WeatherDewPointC),
               WeatherRainHoursPrev1 = COALESCE(@WeatherRainHoursPrev1, target.WeatherRainHoursPrev1),
               WeatherFrostDaysPrev7 = COALESCE(@WeatherFrostDaysPrev7, target.WeatherFrostDaysPrev7),
               WeatherSoilMoisture0To7 = COALESCE(@WeatherSoilMoisture0To7, target.WeatherSoilMoisture0To7),
               WeatherSoilMoisture0To28 = COALESCE(@WeatherSoilMoisture0To28, target.WeatherSoilMoisture0To28),
               WeatherRainAnomaly14dMm = COALESCE(@WeatherRainAnomaly14dMm, target.WeatherRainAnomaly14dMm),
               WeatherSoilMoistureAnomaly = COALESCE(@WeatherSoilMoistureAnomaly, target.WeatherSoilMoistureAnomaly),
               WeatherEt0Prev7dMm = COALESCE(@WeatherEt0Prev7dMm, target.WeatherEt0Prev7dMm),
               DaysSinceLastMeeting = COALESCE(@DaysSinceLastMeeting, target.DaysSinceLastMeeting),
               LastGoingAtCourse = COALESCE(@LastGoingAtCourse, target.LastGoingAtCourse),
               WeatherGoingIndex = COALESCE(@WeatherGoingIndex, target.WeatherGoingIndex),
               WeatherSoftRisk = COALESCE(@WeatherSoftRisk, target.WeatherSoftRisk),
               Source = COALESCE(@Source, target.Source),
               FetchedUtc = SYSUTCDATETIME()
WHEN NOT MATCHED THEN
    INSERT (RaceId, CourseId, WeatherDate, RainPrev1dMm, RainPrev3dMm, RainPrev7dMm,
            RainPrev14dMm, RainRaceDayMm, WeatherTempMaxC, WeatherTempMinC,
            WeatherWindMaxKph, WeatherWindDirectionDegrees, WeatherHumidity,
            WeatherDewPointC, WeatherRainHoursPrev1, WeatherFrostDaysPrev7,
            WeatherSoilMoisture0To7, WeatherSoilMoisture0To28,
            WeatherRainAnomaly14dMm, WeatherSoilMoistureAnomaly,
            WeatherEt0Prev7dMm, DaysSinceLastMeeting,
            LastGoingAtCourse, WeatherGoingIndex, WeatherSoftRisk, Source, FetchedUtc)
    VALUES (@RaceId, @CourseId, @WeatherDate, @RainPrev1dMm, @RainPrev3dMm, @RainPrev7dMm,
            @RainPrev14dMm, @RainRaceDayMm, @WeatherTempMaxC, @WeatherTempMinC,
            @WeatherWindMaxKph, @WeatherWindDirectionDegrees, @WeatherHumidity,
            @WeatherDewPointC, @WeatherRainHoursPrev1, @WeatherFrostDaysPrev7,
            @WeatherSoilMoisture0To7, @WeatherSoilMoisture0To28,
            @WeatherRainAnomaly14dMm, @WeatherSoilMoistureAnomaly,
            @WeatherEt0Prev7dMm, @DaysSinceLastMeeting,
            @LastGoingAtCourse, @WeatherGoingIndex, @WeatherSoftRisk, @Source, SYSUTCDATETIME());";

            using var connection = OpenConnection();
            using var transaction = connection.BeginTransaction();
            try
            {
                var affected = connection.Execute(sql, items, transaction);
                transaction.Commit();
                return affected;
            }
            catch
            {
                transaction.Rollback();
                throw;
            }
        }

        public IReadOnlyList<RaceWeatherTarget> GetRacesForCourse(
            short courseId,
            DateTime? fromDate = null,
            DateTime? toDate = null)
        {
            using var connection = OpenConnection();
            return connection.Query<RaceWeatherTarget>(
                @"SELECT RaceId, RaceDate, ScheduledOff, Going, Surface
                  FROM dbo.Race
                  WHERE CourseId = @CourseId
                    AND (@FromDate IS NULL OR RaceDate >= @FromDate)
                    AND (@ToDate IS NULL OR RaceDate <= @ToDate)
                  ORDER BY RaceDate, ScheduledOff, RaceId",
                new { CourseId = courseId, FromDate = fromDate?.Date, ToDate = toDate?.Date }).ToList();
        }

        public IReadOnlySet<int> GetRaceWeatherIds(short courseId, DateTime fromDate, DateTime toDate)
        {
            using var connection = OpenConnection();
            var ids = connection.Query<int>(
                @"SELECT rw.RaceId
                  FROM dbo.RaceWeather rw
                  JOIN dbo.Race r ON r.RaceId = rw.RaceId
                  WHERE r.CourseId = @CourseId
                    AND r.RaceDate >= @FromDate
                    AND r.RaceDate <= @ToDate",
                new { CourseId = courseId, FromDate = fromDate.Date, ToDate = toDate.Date });
            return ids.ToHashSet();
        }

        public RaceWeatherTarget? GetLastRaceOnOrBefore(short courseId, DateTime date, TimeSpan? offTime)
        {
            using var connection = OpenConnection();
            return connection.QueryFirstOrDefault<RaceWeatherTarget>(
                @"SELECT TOP (1) RaceId, RaceDate, ScheduledOff, Going, Surface
                  FROM dbo.Race
                  WHERE CourseId = @CourseId
                    AND (RaceDate < @Date
                         OR (RaceDate = @Date
                             AND (@OffTime IS NULL
                                  OR ScheduledOff IS NULL
                                  OR ScheduledOff <= @OffTime)))
                  ORDER BY RaceDate DESC, ScheduledOff DESC, RaceId DESC",
                new { CourseId = courseId, Date = date.Date, OffTime = offTime });
        }

        public RaceWeatherFeatures? GetRaceWeather(int raceId)
        {
            using var connection = OpenConnection();
            return connection.QueryFirstOrDefault<RaceWeatherFeatures>(
                @"SELECT RaceId, CourseId, WeatherDate, RainPrev1dMm, RainPrev3dMm,
                         RainPrev7dMm, RainPrev14dMm, RainRaceDayMm, WeatherTempMaxC,
                         WeatherTempMinC, WeatherWindMaxKph, WeatherEt0Prev7dMm,
                         DaysSinceLastMeeting, LastGoingAtCourse, WeatherGoingIndex,
                         WeatherSoftRisk, Source
                  FROM dbo.RaceWeather
                  WHERE RaceId = @RaceId",
                new { RaceId = raceId });
        }
    }
}
