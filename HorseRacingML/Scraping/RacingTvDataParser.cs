using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using HorseRacingML.Models;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Pure-data helpers for the Racing TV feed. Kept separate from the HTTP
    /// client and database writer so the race matching and sectional
    /// aggregation can be unit tested without a network or SQL Server.
    /// </summary>
    public static class RacingTvDataParser
    {
        private static readonly JsonSerializerOptions RawJsonOptions = new()
        {
            WriteIndented = false
        };

        private static readonly HashSet<string> IgnoredCourseWords = new(StringComparer.Ordinal)
        {
            "racecourse", "races", "race", "park", "downs", "royal", "city"
        };

        public static RunnerSectional? BuildRunnerSectional(
            RacingTvSectionalRunner runner,
            RacingTvSectionalRaceInfo race,
            int raceId,
            int horseId,
            string? sourceRaceId)
        {
            if (runner is null || race is null || raceId <= 0 || horseId <= 0)
            {
                return null;
            }

            var splits = (runner.Sectionals ?? new List<RacingTvSectionalSplit>())
                .Where(split => split is not null && split.SectorTime.HasValue)
                .OrderBy(split => split.SectorNumber)
                .ToList();
            if (splits.Count == 0)
            {
                return null;
            }

            int fieldSize = race.Runners?.Count ?? 0;
            var last1 = splits[^1];
            var last2 = splits.Skip(Math.Max(0, splits.Count - 2)).ToList();
            var last3 = splits.Skip(Math.Max(0, splits.Count - 3)).ToList();

            float? last1Percentile = ToFloat(last1.SectorTimePercentile);
            float? last2Percentile = AverageNullable(last2.Select(split => split.SectorTimePercentile));
            float? last3Percentile = AverageNullable(last3.Select(split => split.SectorTimePercentile));
            float? earlyPosition = NormalizePosition(splits[0].SectorPosition, fieldSize);
            float? closingPosition = NormalizePosition(last1.SectorPosition, fieldSize);
            float? paceStyle = earlyPosition.HasValue && closingPosition.HasValue
                ? closingPosition.Value - earlyPosition.Value
                : (float?)null;

            return new RunnerSectional
            {
                RaceId = raceId,
                HorseId = horseId,
                FinishPos = runner.FinishPosition.HasValue ? (short?)runner.FinishPosition.Value : null,
                ClothNumber = runner.ClothNumber.HasValue && runner.ClothNumber.Value <= byte.MaxValue
                    ? (byte?)runner.ClothNumber.Value
                    : null,
                SectionCount = splits.Count <= byte.MaxValue ? (byte?)splits.Count : (byte?)byte.MaxValue,
                Last1FSeconds = ToFloat(last1.SectorTime),
                Last2FSeconds = SumNullable(last2.Select(split => split.SectorTime)),
                Last3FSeconds = SumNullable(last3.Select(split => split.SectorTime)),
                Last1FPercentile = last1Percentile,
                Last2FPercentile = last2Percentile,
                Last3FPercentile = last3Percentile,
                EarlyPositionNorm = earlyPosition,
                ClosingPositionNorm = closingPosition,
                PaceStyle = paceStyle,
                FinishSpeedPct = ParseNumber(runner.FinishingSpeedPercentage),
                TimeIndexScore = ToFloat(race.TimeIndexScore),
                MeetingAverageTimeIndex = ToFloat(race.MeetingAverageTimeIndex),
                WinTimeVsPar = ToFloat(race.WinTimeVsPar),
                SourceRaceId = sourceRaceId,
                RawJson = JsonSerializer.Serialize(splits, RawJsonOptions),
                Source = "RacingTV",
                FetchedUtc = DateTime.UtcNow
            };
        }

        public static RaceBackfillCandidate? MatchRace(
            IReadOnlyList<RaceBackfillCandidate> candidates,
            string? trackName,
            DateTimeOffset startTime,
            string? title)
        {
            if (candidates is null || candidates.Count == 0)
            {
                return null;
            }

            var courseMatches = candidates
                .Where(candidate => CourseNamesMatch(candidate.CourseName, trackName))
                .ToList();
            if (courseMatches.Count == 0)
            {
                return null;
            }

            var targetTime = startTime.UtcDateTime.TimeOfDay;
            RaceBackfillCandidate? bestByTime = null;
            double bestSeconds = double.MaxValue;
            foreach (var candidate in courseMatches)
            {
                if (!candidate.ScheduledOff.HasValue)
                {
                    continue;
                }

                double difference = Math.Abs((candidate.ScheduledOff.Value - targetTime).TotalSeconds);
                if (difference < bestSeconds)
                {
                    bestSeconds = difference;
                    bestByTime = candidate;
                }
            }

            // Scheduled times are stored as UTC by the result scraper. A
            // small tolerance absorbs the usual race-day drift without
            // allowing two races at the same meeting to collide.
            if (bestByTime is not null && bestSeconds <= 8 * 60)
            {
                return bestByTime;
            }

            string normalizedTitle = NormalizeTitle(title);
            if (normalizedTitle.Length > 0)
            {
                var titleMatches = courseMatches
                    .Where(candidate => NormalizeTitle(candidate.Title) == normalizedTitle)
                    .ToList();
                if (titleMatches.Count == 1)
                {
                    return titleMatches[0];
                }
            }

            return bestByTime is not null && bestSeconds <= 15 * 60 ? bestByTime : null;
        }

        public static bool CourseNamesMatch(string? left, string? right)
        {
            string normalizedLeft = NormalizeCourseName(left);
            string normalizedRight = NormalizeCourseName(right);
            if (normalizedLeft.Length == 0 || normalizedRight.Length == 0)
            {
                return false;
            }

            if (string.Equals(normalizedLeft, normalizedRight, StringComparison.Ordinal))
            {
                return true;
            }

            // A few tracks are published with a sponsor-style prefix or a
            // second name ("Epsom Downs" vs "Epsom"). The generic word removal
            // above already handles most of these; the containment check
            // covers the rest without matching two unrelated short names.
            if (normalizedLeft.Length >= 4 && normalizedRight.Length >= 4 &&
                (normalizedLeft.Contains(normalizedRight, StringComparison.Ordinal) ||
                 normalizedRight.Contains(normalizedLeft, StringComparison.Ordinal)))
            {
                return true;
            }

            return false;
        }

        public static string NormalizeCourseName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            string text = value.Trim().ToLowerInvariant()
                .Replace("&", " and ")
                .Replace("'", string.Empty)
                .Replace("-", " ")
                .Replace(".", " ");

            if (text.Contains("stratford on avon", StringComparison.Ordinal))
            {
                text = "stratford";
            }

            var words = text
                .Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(word => !IgnoredCourseWords.Contains(word))
                .ToList();
            return string.Join(" ", words);
        }

        public static string NormalizeTitle(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var builder = new StringBuilder(value.Length);
            bool previousSpace = false;
            foreach (char character in value.Trim().ToLowerInvariant())
            {
                if (char.IsLetterOrDigit(character))
                {
                    builder.Append(character);
                    previousSpace = false;
                }
                else if (!previousSpace)
                {
                    builder.Append(' ');
                    previousSpace = true;
                }
            }

            return builder.ToString().Trim();
        }

        public static string FormatLocalTime(DateTimeOffset startTime)
            => startTime.ToString("HHmm", CultureInfo.InvariantCulture);

        private static float? NormalizePosition(int? position, int fieldSize)
        {
            if (!position.HasValue || fieldSize <= 1)
            {
                return null;
            }

            float normalized = 1f - ((position.Value - 1f) / (fieldSize - 1f));
            return Math.Clamp(normalized, 0f, 1f);
        }

        private static float? ParseNumber(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return null;
            }

            string cleaned = value.Trim().TrimEnd('s', 'S', '%');
            return float.TryParse(cleaned, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) &&
                   float.IsFinite(parsed)
                ? parsed
                : (float?)null;
        }

        private static float? ToFloat(double? value)
            => value.HasValue && double.IsFinite(value.Value) ? (float?)value.Value : null;

        private static float? SumNullable(IEnumerable<double?> values)
        {
            var present = values.Where(value => value.HasValue && double.IsFinite(value.Value)).ToList();
            return present.Count == 0 ? null : (float?)present.Sum(value => value!.Value);
        }

        private static float? AverageNullable(IEnumerable<double?> values)
        {
            var present = values.Where(value => value.HasValue && double.IsFinite(value.Value)).ToList();
            return present.Count == 0 ? null : (float?)present.Average(value => value!.Value);
        }
    }
}
