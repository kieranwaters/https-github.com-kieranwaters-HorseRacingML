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
        private static readonly TimeSpan HandleScheduleCacheLifetime = TimeSpan.FromSeconds(45);

        private IReadOnlyList<string> OrderHandlesByScheduledStart(IWebDriver driver, IEnumerable<string> handlesEnumerable)
        {
            var handles = handlesEnumerable?.ToList() ?? new List<string>();
            if (handles.Count <= 2)
            {
                return handles;
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
            var now = DateTime.UtcNow;
            var handleSet = new HashSet<string>(handles);
            var switched = false;

            for (int i = 0; i < handles.Count; i++)
            {
                var handle = handles[i];
                var sortKey = DateTime.MaxValue;

                HandleScheduleMetadata? cached = null;
                lock (_handleScheduleCacheLock)
                {
                    if (_handleScheduleCache.TryGetValue(handle, out var existing) &&
                        now - existing.CapturedUtc <= HandleScheduleCacheLifetime)
                    {
                        cached = existing;
                        _handleScheduleCache[handle] = new HandleScheduleMetadata(existing.SortKey, existing.Url, now);
                    }
                }

                if (cached != null)
                {
                    metadata.Add((handle, cached.SortKey, i));
                    continue;
                }

                string? url = null;

                try
                {
                    driver.SwitchTo().Window(handle);
                    switched = true;
                    url = driver.Url;
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
                lock (_handleScheduleCacheLock)
                {
                    _handleScheduleCache[handle] = new HandleScheduleMetadata(sortKey, url, now);
                }
            }

            var staleThreshold = TimeSpan.FromTicks(HandleScheduleCacheLifetime.Ticks * 4);

            lock (_handleScheduleCacheLock)
            {
                var keysToRemove = _handleScheduleCache
                    .Where(kvp => !handleSet.Contains(kvp.Key) || now - kvp.Value.CapturedUtc > staleThreshold)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in keysToRemove)
                {
                    _handleScheduleCache.Remove(key);
                }
            }

            if (!string.IsNullOrEmpty(originalHandle))
            {
                try
                {
                    if (switched)
                    {
                        driver.SwitchTo().Window(originalHandle);
                    }
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