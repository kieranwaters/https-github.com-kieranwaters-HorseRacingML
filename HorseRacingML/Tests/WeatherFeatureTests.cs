using System;
using System.Collections.Generic;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Xunit;

namespace HorseRacingML.Tests
{
    public class WeatherFeatureTests
    {
        [Theory]
        [InlineData("Firm", 0)]
        [InlineData("Good to Firm", 1)]
        [InlineData("Good", 2)]
        [InlineData("Good to Soft", 3)]
        [InlineData("Soft", 4)]
        [InlineData("Heavy", 5)]
        [InlineData("Yielding", 4)]
        [InlineData("Good to Yielding", 3)]
        [InlineData("Standard", null)]
        [InlineData("", null)]
        public void GoingDescriptionsMapToTheTurfScale(string going, int? expected)
        {
            Assert.Equal(expected, WeatherFeatureCalculator.GoingIndex(going));
        }

        [Fact]
        public void RainWindowsUseOnlyTheDaysBeforeTheRace()
        {
            var raceDate = new DateTime(2026, 6, 10);
            var days = BuildDays(raceDate, rainByOffset: new Dictionary<int, double>
            {
                [0] = 100,  // race day must not leak into historical windows
                [-1] = 5,
                [-2] = 3,
                [-3] = 2,
                [-6] = 4,
                [-14] = 10
            });

            var features = WeatherFeatureCalculator.Compute(
                1, 1, raceDate, new TimeSpan(15, 0, 0), "Turf", "Good", 7, days,
                includeRaceDayRain: false, source: "archive");

            Assert.Equal(5, features.RainPrev1dMm);
            Assert.Equal(10, features.RainPrev3dMm);
            Assert.Equal(14, features.RainPrev7dMm);
            Assert.Equal(24, features.RainPrev14dMm);
            Assert.Null(features.RainRaceDayMm);
        }

        [Fact]
        public void ForecastRaceDayRainIsOnlyIncludedForLiveRows()
        {
            var raceDate = new DateTime(2026, 6, 10);
            var days = BuildDays(raceDate, rainByOffset: new Dictionary<int, double> { [0] = 12 });

            var features = WeatherFeatureCalculator.Compute(
                1, 1, raceDate, new TimeSpan(15, 0, 0), "Turf", "Good", 7, days,
                includeRaceDayRain: true, source: "forecast");

            Assert.Equal(12, features.RainRaceDayMm);
        }

        [Fact]
        public void HeavyRainMovesTheGoingEstimateTowardsSoft()
        {
            var raceDate = new DateTime(2026, 3, 10);
            var dry = BuildDays(raceDate, rainByOffset: new Dictionary<int, double>
            {
                [-1] = 0, [-2] = 0, [-3] = 0, [-4] = 0, [-5] = 0, [-6] = 0, [-7] = 0
            });
            var wet = BuildDays(raceDate, rainByOffset: new Dictionary<int, double>
            {
                [-1] = 15, [-2] = 12, [-3] = 10, [-4] = 8, [-5] = 5, [-6] = 4, [-7] = 3
            });

            var dryFeatures = WeatherFeatureCalculator.Compute(
                1, 1, raceDate, null, "Turf", "Good", 7, dry, false, "archive");
            var wetFeatures = WeatherFeatureCalculator.Compute(
                1, 1, raceDate, null, "Turf", "Good", 7, wet, false, "archive");

            Assert.NotNull(dryFeatures.WeatherGoingIndex);
            Assert.NotNull(wetFeatures.WeatherGoingIndex);
            Assert.True(wetFeatures.WeatherGoingIndex > dryFeatures.WeatherGoingIndex + 1);
            Assert.True(wetFeatures.WeatherSoftRisk > dryFeatures.WeatherSoftRisk);
        }

        [Theory]
        [InlineData("All Weather")]
        [InlineData("Standard")]
        [InlineData("Dirt")]
        public void NonTurfSurfacesDoNotGetATurfGoingEstimate(string surface)
        {
            var raceDate = new DateTime(2026, 6, 10);
            var days = BuildDays(raceDate, rainByOffset: new Dictionary<int, double> { [-1] = 20 });

            var features = WeatherFeatureCalculator.Compute(
                1, 1, raceDate, new TimeSpan(15, 0, 0), surface, "Standard", 3, days,
                includeRaceDayRain: false, source: "archive");

            Assert.Null(features.WeatherGoingIndex);
            Assert.Null(features.WeatherSoftRisk);
            Assert.Equal(20, features.RainPrev1dMm);
        }

        [Theory]
        [InlineData("Ascot", "Europe/London")]
        [InlineData("Kempton Park", "Europe/London")]
        [InlineData("Newmarket (July)", "Europe/London")]
        [InlineData("Great Yarmouth", "Europe/London")]
        [InlineData("Leopardstown", "Europe/Dublin")]
        [InlineData("Ffos Las Racecourse", "Europe/London")]
        [InlineData("York Racecourse", "Europe/London")]
        public void CourseNamesResolveToLocations(string courseName, string expectedTimeZone)
        {
            Assert.True(CourseLocationCatalog.TryMatch(courseName, out var location));
            Assert.Equal(expectedTimeZone, location.TimeZoneId);
            Assert.NotEqual(0, location.Latitude);
            Assert.NotEqual(0, location.Longitude);
        }

        [Fact]
        public void UnknownCourseNameDoesNotResolve()
        {
            Assert.False(CourseLocationCatalog.TryMatch("Definitely Not A Racecourse", out _));
        }

        [Fact]
        public void ForeignTrackIsNotMatchedByASubstringOfItsName()
        {
            Assert.False(CourseLocationCatalog.TryMatch("New York", out _));
        }

        [Fact]
        public void OpenMeteoDailyJsonIsParsedWithoutThrowingOnNulls()
        {
            const string json = """
            {
              "daily": {
                "time": ["2026-06-01", "2026-06-02"],
                "precipitation_sum": [1.2, null],
                "rain_sum": [1.0, null],
                "temperature_2m_max": [18.5, 19.0],
                "temperature_2m_min": [9.0, 10.0],
                "wind_speed_10m_max": [14.0, null],
                "wind_gusts_10m_max": [30.0, 31.0],
                "et0_fao_evapotranspiration": [3.1, 3.4]
              }
            }
            """;

            var days = OpenMeteoWeatherClient.ParseDaily(json, 7, "archive");

            Assert.Equal(2, days.Count);
            Assert.Equal(new DateTime(2026, 6, 1), days[0].WeatherDate);
            Assert.Equal(1.2, days[0].PrecipitationMm);
            Assert.Null(days[1].PrecipitationMm);
            Assert.Equal(18.5, days[0].TempMaxC);
            Assert.Equal(7, days[0].CourseId);
            Assert.Equal("archive", days[0].Source);
        }

        private static IReadOnlyDictionary<DateTime, CourseWeatherDay> BuildDays(
            DateTime raceDate,
            IReadOnlyDictionary<int, double> rainByOffset)
        {
            var map = new Dictionary<DateTime, CourseWeatherDay>();
            for (var offset = -15; offset <= 0; offset++)
            {
                var date = raceDate.AddDays(offset);
                map[date] = new CourseWeatherDay
                {
                    CourseId = 1,
                    WeatherDate = date,
                    RainMm = rainByOffset.TryGetValue(offset, out var rain) ? rain : 0,
                    PrecipitationMm = rainByOffset.TryGetValue(offset, out var precipitation) ? precipitation : 0,
                    TempMaxC = 12,
                    TempMinC = 5,
                    WindMaxKph = 10,
                    Et0Mm = 1.5,
                    Source = "archive"
                };
            }

            return map;
        }
    }
}
