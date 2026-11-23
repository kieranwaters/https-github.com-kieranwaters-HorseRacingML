using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using HorseRacingML.Models;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Thin client for the free Open-Meteo APIs. The free tier is intended for
    /// non-commercial use and data is licensed CC BY 4.0, so the application
    /// includes attribution wherever the weather is displayed.
    ///
    /// The client is synchronous because the existing scraper and trainer
    /// pipeline is synchronous. Calls are short and cached in CourseWeatherDaily,
    /// so this is not on a hot path.
    /// </summary>
    public sealed class OpenMeteoWeatherClient : IDisposable
    {
        private const string DailyVariables =
            "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min," +
            "wind_speed_10m_max,wind_gusts_10m_max,et0_fao_evapotranspiration," +
            "relative_humidity_2m_mean,dew_point_2m_mean,wind_direction_10m_dominant," +
            "precipitation_hours,snowfall_sum,soil_moisture_0_to_7cm_mean," +
            "soil_moisture_7_to_28cm_mean";

        private readonly HttpClient _http;
        private readonly string _archiveBaseUrl;
        private readonly string _forecastBaseUrl;
        private readonly int _pastDays;
        private readonly int _forecastDays;

        public OpenMeteoWeatherClient(IConfiguration configuration)
            : this(
                configuration.GetValue<string>("Weather:ArchiveBaseUrl")
                    ?? "https://archive-api.open-meteo.com/v1/archive",
                configuration.GetValue<string>("Weather:ForecastBaseUrl")
                    ?? "https://api.open-meteo.com/v1/forecast",
                configuration.GetValue<int?>("Weather:RequestTimeoutSeconds") ?? 20,
                configuration.GetValue<int?>("Weather:PastDays") ?? 14,
                configuration.GetValue<int?>("Weather:ForecastDays") ?? 16)
        {
        }

        internal OpenMeteoWeatherClient(
            string archiveBaseUrl,
            string forecastBaseUrl,
            int timeoutSeconds,
            int pastDays,
            int forecastDays,
            HttpClient? httpClient = null)
        {
            _archiveBaseUrl = archiveBaseUrl;
            _forecastBaseUrl = forecastBaseUrl;
            _pastDays = Math.Clamp(pastDays, 0, 92);
            _forecastDays = Math.Clamp(forecastDays, 1, 16);
            _http = httpClient ?? new HttpClient();
            _http.Timeout = TimeSpan.FromSeconds(Math.Max(5, timeoutSeconds));
            _http.DefaultRequestHeaders.UserAgent.ParseAdd("HorseRacingML/1.0 (weather backfill)");
        }

        public static string Attribution => "Weather data by Open-Meteo.com (CC BY 4.0)";

        public IReadOnlyList<CourseWeatherDay> GetArchive(
            short courseId,
            double latitude,
            double longitude,
            string? timeZoneId,
            DateTime startDate,
            DateTime endDate)
        {
            if (endDate < startDate)
            {
                return Array.Empty<CourseWeatherDay>();
            }

            var url = string.Format(
                CultureInfo.InvariantCulture,
                "{0}?latitude={1:0.####}&longitude={2:0.####}&start_date={3:yyyy-MM-dd}&end_date={4:yyyy-MM-dd}&daily={5}&timezone={6}&timeformat=iso8601",
                _archiveBaseUrl,
                latitude,
                longitude,
                startDate,
                endDate,
                DailyVariables,
                Uri.EscapeDataString(string.IsNullOrWhiteSpace(timeZoneId) ? "auto" : timeZoneId!));

            return Fetch(courseId, url, "archive");
        }

        public IReadOnlyList<CourseWeatherDay> GetForecast(
            short courseId,
            double latitude,
            double longitude,
            string? timeZoneId)
        {
            var url = string.Format(
                CultureInfo.InvariantCulture,
                "{0}?latitude={1:0.####}&longitude={2:0.####}&daily={3}&timezone={4}&past_days={5}&forecast_days={6}&timeformat=iso8601",
                _forecastBaseUrl,
                latitude,
                longitude,
                DailyVariables,
                Uri.EscapeDataString(string.IsNullOrWhiteSpace(timeZoneId) ? "auto" : timeZoneId!),
                _pastDays,
                _forecastDays);

            return Fetch(courseId, url, "forecast");
        }

        private IReadOnlyList<CourseWeatherDay> Fetch(short courseId, string url, string source)
        {
            string json;
            try
            {
                json = _http.GetStringAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
            }
            catch (HttpRequestException ex) when (ex.StatusCode == HttpStatusCode.TooManyRequests)
            {
                throw new WeatherRateLimitException(
                    $"Open-Meteo rate limit reached for course {courseId} ({source}). " +
                    "The backfill is resumable; rerun it later or self-host Open-Meteo.", ex);
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
            {
                throw new InvalidOperationException(
                    $"Weather request failed for course {courseId} ({source}): {ex.Message}", ex);
            }

            return ParseDaily(json, courseId, source);
        }

        internal static IReadOnlyList<CourseWeatherDay> ParseDaily(string json, short courseId, string source)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                return Array.Empty<CourseWeatherDay>();
            }

            using var document = JsonDocument.Parse(json);
            if (!document.RootElement.TryGetProperty("daily", out var daily) ||
                !daily.TryGetProperty("time", out var times) ||
                times.ValueKind != JsonValueKind.Array)
            {
                return Array.Empty<CourseWeatherDay>();
            }

            var precipitation = GetArray(daily, "precipitation_sum");
            var rain = GetArray(daily, "rain_sum");
            var tempMax = GetArray(daily, "temperature_2m_max");
            var tempMin = GetArray(daily, "temperature_2m_min");
            var windMax = GetArray(daily, "wind_speed_10m_max");
            var windGustMax = GetArray(daily, "wind_gusts_10m_max");
            var windDirection = GetArray(daily, "wind_direction_10m_dominant");
            var humidity = GetArray(daily, "relative_humidity_2m_mean");
            var dewPoint = GetArray(daily, "dew_point_2m_mean");
            var precipitationHours = GetArray(daily, "precipitation_hours");
            var snowfall = GetArray(daily, "snowfall_sum");
            var soilMoisture0To7 = GetArray(daily, "soil_moisture_0_to_7cm_mean");
            // Open-Meteo has 0-7 cm and 7-28 cm layers, not a single 0-28 cm
            // daily variable. Combine them by layer thickness below.
            var soilMoisture7To28 = GetArray(daily, "soil_moisture_7_to_28cm_mean");
            var et0 = GetArray(daily, "et0_fao_evapotranspiration");

            var result = new List<CourseWeatherDay>(times.GetArrayLength());
            for (var i = 0; i < times.GetArrayLength(); i++)
            {
                if (times[i].ValueKind != JsonValueKind.String ||
                    !DateTime.TryParseExact(
                        times[i].GetString(),
                        "yyyy-MM-dd",
                        CultureInfo.InvariantCulture,
                        DateTimeStyles.None,
                        out var date))
                {
                    continue;
                }

                var shallowSoil = ReadDouble(soilMoisture0To7, i);
                var midSoil = ReadDouble(soilMoisture7To28, i);
                var soil0To28 = shallowSoil.HasValue && midSoil.HasValue
                    ? (shallowSoil.Value * 7 + midSoil.Value * 21) / 28
                    : shallowSoil ?? midSoil;

                result.Add(new CourseWeatherDay
                {
                    CourseId = courseId,
                    WeatherDate = date.Date,
                    PrecipitationMm = ReadDouble(precipitation, i),
                    RainMm = ReadDouble(rain, i),
                    TempMaxC = ReadDouble(tempMax, i),
                    TempMinC = ReadDouble(tempMin, i),
                    WindMaxKph = ReadDouble(windMax, i),
                    WindGustMaxKph = ReadDouble(windGustMax, i),
                    WindDirectionDominant = ReadDouble(windDirection, i),
                    RelativeHumidityMean = ReadDouble(humidity, i),
                    DewPointMean = ReadDouble(dewPoint, i),
                    PrecipitationHours = ReadDouble(precipitationHours, i),
                    SnowfallCm = ReadDouble(snowfall, i),
                    SoilMoisture0To7 = shallowSoil,
                    SoilMoisture0To28 = soil0To28,
                    Et0Mm = ReadDouble(et0, i),
                    Source = source,
                    FetchedUtc = DateTime.UtcNow
                });
            }

            return result;
        }

        private static JsonElement? GetArray(JsonElement parent, string propertyName)
        {
            return parent.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.Array
                ? value
                : null;
        }

        private static double? ReadDouble(JsonElement? array, int index)
        {
            if (!array.HasValue || index < 0 || index >= array.Value.GetArrayLength())
            {
                return null;
            }

            var item = array.Value[index];
            if (item.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
            {
                return null;
            }

            return item.TryGetDouble(out var value) && double.IsFinite(value) ? value : null;
        }

        public void Dispose() => _http.Dispose();
    }

    public sealed class WeatherRateLimitException : Exception
    {
        public WeatherRateLimitException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
