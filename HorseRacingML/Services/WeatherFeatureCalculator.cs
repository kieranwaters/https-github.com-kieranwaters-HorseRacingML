using System;
using System.Collections.Generic;
using HorseRacingML.ML;
using HorseRacingML.Models;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Converts daily weather observations into the per-race features consumed
    /// by the model. Only days strictly before the race date contribute to the
    /// historical rain windows; race-day rain is a separate, forecast-only
    /// feature so a backtest cannot see the weather that occurred after the
    /// prediction was made.
    /// </summary>
    public static class WeatherFeatureCalculator
    {
        private const float NeutralGoingIndex = 2.0f; // Good

        public static RaceWeatherFeatures Compute(
            int raceId,
            short courseId,
            DateTime raceDate,
            TimeSpan? offTime,
            string? surface,
            string? lastGoing,
            int? daysSinceLastMeeting,
            IReadOnlyDictionary<DateTime, CourseWeatherDay> days,
            bool includeRaceDayRain,
            string? source,
            double? rainNormal14d = null,
            double? soilMoistureNormal = null)
        {
            if (days is null)
            {
                throw new ArgumentNullException(nameof(days));
            }

            var date = raceDate.Date;
            var features = new RaceWeatherFeatures
            {
                RaceId = raceId,
                CourseId = courseId,
                WeatherDate = date,
                Source = source,
                DaysSinceLastMeeting = daysSinceLastMeeting,
                LastGoingAtCourse = string.IsNullOrWhiteSpace(lastGoing) ? null : lastGoing,
                RainPrev1dMm = Sum(days, date, -1, -1, d => d.RainMm),
                RainPrev3dMm = Sum(days, date, -3, -1, d => d.RainMm),
                RainPrev7dMm = Sum(days, date, -7, -1, d => d.RainMm),
                RainPrev14dMm = Sum(days, date, -14, -1, d => d.RainMm),
                WeatherTempMaxC = Average(days, date, -3, -1, d => d.TempMaxC),
                WeatherTempMinC = Average(days, date, -3, -1, d => d.TempMinC),
                WeatherWindMaxKph = Max(days, date, -3, -1, d => d.WindMaxKph),
                WeatherWindDirectionDegrees = Average(days, date, -1, -1, d => d.WindDirectionDominant),
                WeatherHumidity = Average(days, date, -1, -1, d => d.RelativeHumidityMean),
                WeatherDewPointC = Average(days, date, -1, -1, d => d.DewPointMean),
                WeatherRainHoursPrev1 = Sum(days, date, -1, -1, d => d.PrecipitationHours),
                WeatherFrostDaysPrev7 = Count(days, date, -7, -1, d => d.TempMinC is <= 0),
                WeatherSoilMoisture0To7 = Average(days, date, -1, -1, d => d.SoilMoisture0To7),
                WeatherSoilMoisture0To28 = Average(days, date, -1, -1, d => d.SoilMoisture0To28),
                WeatherEt0Prev7dMm = Sum(days, date, -7, -1, d => d.Et0Mm)
            };

            if (features.RainPrev14dMm.HasValue && rainNormal14d.HasValue)
            {
                features.WeatherRainAnomaly14dMm =
                    features.RainPrev14dMm.Value - rainNormal14d.Value;
            }

            if (features.WeatherSoilMoisture0To7.HasValue && soilMoistureNormal.HasValue)
            {
                features.WeatherSoilMoistureAnomaly =
                    features.WeatherSoilMoisture0To7.Value - soilMoistureNormal.Value;
            }

            if (includeRaceDayRain &&
                days.TryGetValue(date, out var raceDay))
            {
                features.RainRaceDayMm = raceDay.PrecipitationMm ?? raceDay.RainMm;
            }

            var normalizedSurface = FeatureText.NormalizeSurface(surface);
            var normalizedGoing = FeatureText.NormalizeCategory(lastGoing);
            var turf = normalizedSurface is not ("allweather" or "dirt" or "standard") &&
                       !normalizedGoing.Contains("standard", StringComparison.Ordinal);

            var lastIndex = GoingIndex(lastGoing);
            var hasWeather = features.RainPrev7dMm.HasValue ||
                             features.WeatherTempMaxC.HasValue ||
                             features.WeatherWindMaxKph.HasValue ||
                             features.WeatherEt0Prev7dMm.HasValue;

            if (!turf || (!hasWeather && !lastIndex.HasValue))
            {
                return features;
            }

            double baseIndex = lastIndex ?? NeutralGoingIndex;
            if (lastIndex.HasValue &&
                daysSinceLastMeeting is > 21)
            {
                // After a long break the last going says less about the current
                // surface, so shrink it towards Good.
                var shrink = Math.Min(0.5, (daysSinceLastMeeting.Value - 21) / 60.0);
                baseIndex += (NeutralGoingIndex - baseIndex) * shrink;
            }

            var rain7 = features.RainPrev7dMm ?? 0;
            var raceDayRain = includeRaceDayRain ? features.RainRaceDayMm ?? 0 : 0;
            var wetEffect = 0.45 * Math.Log(1 + rain7 / 4.0) +
                            0.25 * Math.Log(1 + raceDayRain / 10.0);
            var temp = features.WeatherTempMaxC ?? 15;
            var wind = features.WeatherWindMaxKph ?? 10;
            var et0 = features.WeatherEt0Prev7dMm ?? 0;
            var dryEffect = 0.035 * Math.Max(0, temp - 15) +
                            0.004 * wind +
                            0.05 * et0;

            var index = baseIndex + wetEffect - dryEffect;

            // Sustained rain should not be argued away by a warm, breezy day.
            if (rain7 >= 25)
            {
                index = Math.Max(index, 4.0);
            }
            if (rain7 >= 50)
            {
                index = Math.Max(index, 4.5);
            }

            index = Math.Clamp(index, 0.0, 5.0);
            features.WeatherGoingIndex = index;

            if (hasWeather)
            {
                features.WeatherSoftRisk = 1.0 / (1.0 + Math.Exp(-(index - 3.0) * 1.5));
            }

            return features;
        }

        /// <summary>
        /// Maps a going description to a turf ordinal: 0 firm, 1 good to firm,
        /// 2 good, 3 good to soft/yielding, 4 soft, 5 heavy. Standard and
        /// unknown values return null because they are not on the turf scale.
        /// </summary>
        public static int? GoingIndex(string? going)
        {
            var value = FeatureText.NormalizeCategory(going);
            if (string.IsNullOrWhiteSpace(value) || value == "unknown")
            {
                return null;
            }

            if (value.Contains("heavy", StringComparison.Ordinal))
            {
                return 5;
            }

            if (value.Contains("good to soft", StringComparison.Ordinal) ||
                value.Contains("good to yielding", StringComparison.Ordinal))
            {
                return 3;
            }

            if (value.Contains("soft", StringComparison.Ordinal) ||
                value.Contains("yielding", StringComparison.Ordinal))
            {
                return 4;
            }

            if (value.Contains("good to firm", StringComparison.Ordinal))
            {
                return 1;
            }

            if (value.Contains("firm", StringComparison.Ordinal))
            {
                return 0;
            }

            if (value.Contains("good", StringComparison.Ordinal))
            {
                return 2;
            }

            return null;
        }

        private static double? Sum(
            IReadOnlyDictionary<DateTime, CourseWeatherDay> days,
            DateTime raceDate,
            int fromOffset,
            int toOffset,
            Func<CourseWeatherDay, double?> selector)
        {
            double total = 0;
            var found = false;
            for (var offset = fromOffset; offset <= toOffset; offset++)
            {
                if (!days.TryGetValue(raceDate.AddDays(offset), out var day))
                {
                    continue;
                }

                var value = selector(day);
                if (!value.HasValue || !double.IsFinite(value.Value))
                {
                    continue;
                }

                total += value.Value;
                found = true;
            }

            return found ? total : null;
        }

        private static double? Average(
            IReadOnlyDictionary<DateTime, CourseWeatherDay> days,
            DateTime raceDate,
            int fromOffset,
            int toOffset,
            Func<CourseWeatherDay, double?> selector)
        {
            double total = 0;
            var count = 0;
            for (var offset = fromOffset; offset <= toOffset; offset++)
            {
                if (!days.TryGetValue(raceDate.AddDays(offset), out var day))
                {
                    continue;
                }

                var value = selector(day);
                if (!value.HasValue || !double.IsFinite(value.Value))
                {
                    continue;
                }

                total += value.Value;
                count++;
            }

            return count > 0 ? total / count : null;
        }

        private static int? Count(
            IReadOnlyDictionary<DateTime, CourseWeatherDay> days,
            DateTime raceDate,
            int fromOffset,
            int toOffset,
            Func<CourseWeatherDay, bool> predicate)
        {
            int count = 0;
            bool found = false;
            for (var offset = fromOffset; offset <= toOffset; offset++)
            {
                if (!days.TryGetValue(raceDate.AddDays(offset), out var day))
                {
                    continue;
                }

                found = true;
                if (predicate(day))
                {
                    count++;
                }
            }

            return found ? count : null;
        }

        private static double? Max(
            IReadOnlyDictionary<DateTime, CourseWeatherDay> days,
            DateTime raceDate,
            int fromOffset,
            int toOffset,
            Func<CourseWeatherDay, double?> selector)
        {
            double? best = null;
            for (var offset = fromOffset; offset <= toOffset; offset++)
            {
                if (!days.TryGetValue(raceDate.AddDays(offset), out var day))
                {
                    continue;
                }

                var value = selector(day);
                if (!value.HasValue || !double.IsFinite(value.Value))
                {
                    continue;
                }

                best = best.HasValue ? Math.Max(best.Value, value.Value) : value.Value;
            }

            return best;
        }
    }
}
