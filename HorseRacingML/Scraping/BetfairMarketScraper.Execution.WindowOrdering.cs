using OpenQA.Selenium;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace HorseRacingML.Scraping
{
    public partial class BetfairMarketScraper
    {
        private IReadOnlyList<string> OrderHandlesByScheduledStart(IWebDriver driver, IReadOnlyList<string> handles)
        {
            if (handles == null || handles.Count <= 1)
            {
                return handles ?? Array.Empty<string>();
            }

            string? originalHandle = null;
            try
            {
                originalHandle = driver.CurrentWindowHandle;
            }
            catch (WebDriverException)
            {
                originalHandle = null;
            }

            var metadata = new List<(string Handle, DateTime SortKey, int Index)>(handles.Count);

            for (int i = 0; i < handles.Count; i++)
            {
                var handle = handles[i];
                var sortKey = DateTime.MaxValue;

                try
                {
                    driver.SwitchTo().Window(handle);
                    var url = driver.Url;
                    if (!string.IsNullOrWhiteSpace(url) && url.Contains("/horse-racing/", StringComparison.OrdinalIgnoreCase))
                    {
                        var scheduled = TryResolveScheduledStart(driver);
                        if (scheduled.HasValue)
                        {
                            sortKey = scheduled.Value;
                        }
                    }
                }
                catch (WebDriverException)
                {
                    // Ignore handles that cannot be focused; they will keep the default sort key.
                }

                metadata.Add((handle, sortKey, i));
            }

            if (!string.IsNullOrEmpty(originalHandle))
            {
                try
                {
                    driver.SwitchTo().Window(originalHandle);
                }
                catch (WebDriverException)
                {
                    // Ignore failures when restoring the original window.
                }
            }

            return metadata
                .OrderBy(m => m.SortKey)
                .ThenBy(m => m.Index)
                .Select(m => m.Handle)
                .ToList();
        }

        private DateTime? TryResolveScheduledStart(IWebDriver driver)
        {
            var eventDateText = ReadFirstNonEmptyText(
                driver,
                ".event-date",
                "[data-testid='event-date']",
                "[data-testid='eventDate']",
                "[data-testid='market-date']");

            var startTimeText = ReadFirstNonEmptyText(
                driver,
                "[data-testid='startTime']",
                "[data-testid='marketStartTime']",
                "[data-testid='market-start-time']",
                ".market-name",
                ".event-time");

            var venueText = ReadFirstNonEmptyText(driver, ".venue-name");

            var offTime = TryParseRaceTime(startTimeText);

            if (!offTime.HasValue)
            {
                var venueDetails = ParseVenueDetails(venueText);
                if (venueDetails.Time.HasValue)
                {
                    offTime = venueDetails.Time;
                }
            }

            DateTime? raceDate = null;
            if (!string.IsNullOrWhiteSpace(eventDateText))
            {
                raceDate = ParseEventDate(eventDateText, DateTime.Today);
            }

            if (!raceDate.HasValue)
            {
                var titleText = ReadFirstNonEmptyText(
                    driver,
                    "[data-testid='marketTitle']",
                    "[data-testid='market-title']",
                    "h1[data-testid='marketTitle']",
                    "h1[data-testid='eventTitle']",
                    ".market-title",
                    "header h1",
                    ".page-title h1");

                if (!string.IsNullOrWhiteSpace(titleText))
                {
                    var normalized = NormalizeMarketTitle(titleText);
                    if (!string.IsNullOrWhiteSpace(normalized))
                    {
                        var extractedDate = ExtractDateFromTitle(normalized);
                        if (extractedDate.HasValue)
                        {
                            raceDate = extractedDate;
                        }
                    }
                }
            }

            if (!raceDate.HasValue)
            {
                var fallbackTitle = ExtractDocumentTitle(driver);
                if (!string.IsNullOrWhiteSpace(fallbackTitle))
                {
                    var normalized = NormalizeMarketTitle(fallbackTitle);
                    if (!string.IsNullOrWhiteSpace(normalized))
                    {
                        raceDate = ExtractDateFromTitle(normalized);
                    }
                }
            }

            if (!raceDate.HasValue)
            {
                return null;
            }

            if (!offTime.HasValue)
            {
                return raceDate;
            }

            try
            {
                return raceDate.Value.Date + offTime.Value;
            }
            catch (Exception)
            {
                return raceDate.Value;
            }
        }

        private static TimeSpan? TryParseRaceTime(string? text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var trimmed = text.Trim();
            if (trimmed.Length == 0)
            {
                return null;
            }

            if (TimeSpan.TryParse(trimmed, CultureInfo.InvariantCulture, out var parsed))
            {
                return parsed;
            }

            if (DateTime.TryParse(trimmed, CultureInfo.InvariantCulture, DateTimeStyles.None, out var parsedDateTime))
            {
                return parsedDateTime.TimeOfDay;
            }

            var match = Regex.Match(trimmed, @"(\d{1,2}:\d{2})");
            if (match.Success && TimeSpan.TryParse(match.Groups[1].Value, CultureInfo.InvariantCulture, out var fallback))
            {
                return fallback;
            }

            return null;
        }
    }
}