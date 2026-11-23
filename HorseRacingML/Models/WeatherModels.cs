using System;
using System.Collections.Generic;

namespace HorseRacingML.Models
{
    /// <summary>
    /// One daily row returned by the weather provider for a course. Values are
    /// local-calendar-day aggregates; the race feature builder only uses days
    /// before the race date, so race-day rain cannot leak into a historical
    /// training row.
    /// </summary>
    public sealed class CourseWeatherDay
    {
        public short CourseId { get; set; }
        public DateTime WeatherDate { get; set; }
        public double? PrecipitationMm { get; set; }
        public double? RainMm { get; set; }
        public double? TempMaxC { get; set; }
        public double? TempMinC { get; set; }
        public double? WindMaxKph { get; set; }
        public double? WindGustMaxKph { get; set; }
        public double? WindDirectionDominant { get; set; }
        public double? RelativeHumidityMean { get; set; }
        public double? DewPointMean { get; set; }
        public double? PrecipitationHours { get; set; }
        public double? SnowfallCm { get; set; }
        public double? SoilMoisture0To7 { get; set; }
        public double? SoilMoisture0To28 { get; set; }
        public double? Et0Mm { get; set; }
        public string? Source { get; set; }
        public DateTime? FetchedUtc { get; set; }
    }

    /// <summary>
    /// The weather features stored for one race. The names are the feature keys
    /// consumed by the model, so they must stay stable across the historical
    /// builder and the live scraper.
    /// </summary>
    public sealed class RaceWeatherFeatures
    {
        public int RaceId { get; set; }
        public short CourseId { get; set; }
        public DateTime WeatherDate { get; set; }

        public double? RainPrev1dMm { get; set; }
        public double? RainPrev3dMm { get; set; }
        public double? RainPrev7dMm { get; set; }
        public double? RainPrev14dMm { get; set; }

        /// <summary>
        /// Race-day rainfall. Historically this is left null because only the
        /// pre-race portion is knowable; for an upcoming race it is the
        /// forecast value for the rest of the day.
        /// </summary>
        public double? RainRaceDayMm { get; set; }

        public double? WeatherTempMaxC { get; set; }
        public double? WeatherTempMinC { get; set; }
        public double? WeatherWindMaxKph { get; set; }
        public double? WeatherWindDirectionDegrees { get; set; }
        public double? WeatherHumidity { get; set; }
        public double? WeatherDewPointC { get; set; }
        public double? WeatherRainHoursPrev1 { get; set; }
        public int? WeatherFrostDaysPrev7 { get; set; }
        public double? WeatherSoilMoisture0To7 { get; set; }
        public double? WeatherSoilMoisture0To28 { get; set; }
        public double? WeatherRainAnomaly14dMm { get; set; }
        public double? WeatherSoilMoistureAnomaly { get; set; }
        public double? WeatherEt0Prev7dMm { get; set; }
        public int? DaysSinceLastMeeting { get; set; }
        public string? LastGoingAtCourse { get; set; }

        /// <summary>
        /// Heuristic 0 (firm) to 5 (heavy) going estimate for turf. This is a
        /// transparent prior, not a trained target; the main model decides how
        /// much to use it.
        /// </summary>
        public double? WeatherGoingIndex { get; set; }

        /// <summary>
        /// Probability-like score that the surface is at least Good to Soft.
        /// Null for all-weather and dirt, where turf going does not apply.
        /// </summary>
        public double? WeatherSoftRisk { get; set; }

        public string? Source { get; set; }
    }

    public enum WeatherBackfillMode
    {
        /// <summary>Resumable backfill: fetch only course-years that still lack race rows.</summary>
        Normal = 0,

        /// <summary>Re-fetch the selected range even when race rows already exist, so newly added raw columns are populated.</summary>
        Refresh = 1,

        /// <summary>Recompute race features from CourseWeatherDaily without any HTTP calls.</summary>
        CacheOnly = 2
    }

    public sealed class RaceWeatherTarget
    {
        public int RaceId { get; set; }
        public DateTime RaceDate { get; set; }
        public TimeSpan? ScheduledOff { get; set; }
        public string? Going { get; set; }
        public string? Surface { get; set; }
    }

    public sealed class WeatherBackfillSummary
    {
        public int CoursesLocated { get; set; }
        public int CoursesProcessed { get; set; }
        public int WeatherDaysUpserted { get; set; }
        public int RaceRowsUpserted { get; set; }
        public int CoursesSkipped { get; set; }
        public IReadOnlyList<string> Warnings { get; set; } = Array.Empty<string>();
    }
}
