using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using HorseRacingML.Models;
using HtmlAgilityPack;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Parses the public At The Races result and sectional fragments. ATR uses
    /// Total Performance Data timing, which also covers British meetings that
    /// are missing from the Racing TV / RaceiQ archive.
    /// </summary>
    public static class AtTheRacesSectionalParser
    {
        private static readonly Regex RaceIdPattern = new(
            @"raceid=(\d+)",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

        private static readonly Regex FinishTimePattern = new(
            @"(?:(?<minutes>\d+)\s*m\s*)?(?<seconds>[0-9]+(?:\.[0-9]+)?)\s*s",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

        private static readonly Regex NumberPattern = new(
            @"[0-9]+(?:\.[0-9]+)?",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex SectionalFinishCellPattern = new(
            @"card-cell--sectional-time-\d+-sectionals",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        public static IReadOnlyList<AtTheRacesRaceLink> ParseResultsPage(string html)
        {
            var document = new HtmlDocument();
            document.LoadHtml(html ?? string.Empty);

            var links = new List<AtTheRacesRaceLink>();
            var seenRaceIds = new HashSet<long>();
            foreach (var article in SelectNodes(document, "//article[.//a[starts-with(@href, '/racecard/')]]"))
            {
                var raceLink = article.SelectSingleNode(".//a[starts-with(@href, '/racecard/')]");
                string href = raceLink?.GetAttributeValue("href", string.Empty) ?? string.Empty;
                if (!TryParseRacePath(href, out var course, out var localDate, out var localTime))
                {
                    continue;
                }

                long raceId = ReadRaceId(article);
                if (raceId <= 0 || !seenRaceIds.Add(raceId))
                {
                    continue;
                }

                links.Add(new AtTheRacesRaceLink
                {
                    RaceId = raceId,
                    Course = course,
                    LocalDate = localDate,
                    LocalTime = localTime,
                    PagePath = href,
                    Title = Clean(raceLink?.InnerText)
                });
            }

            return links;
        }

        public static IReadOnlyList<AtTheRacesRunnerSectional> ParseSectionalPage(string html)
        {
            var document = new HtmlDocument();
            document.LoadHtml(html ?? string.Empty);

            var headerNames = SelectNodes(document, "//div[contains(@class,'card-cell--sectional-header')]")
                .Select(node => Clean(node.InnerText))
                .Where(name => name.Length > 0)
                .ToList();
            if (headerNames.Count > 0 &&
                headerNames[^1].Equals("Finish", StringComparison.OrdinalIgnoreCase))
            {
                headerNames.RemoveAt(headerNames.Count - 1);
            }

            var runners = new Dictionary<string, AtTheRacesRunnerSectional>(StringComparer.Ordinal);
            foreach (var entry in SelectNodes(document, "//div[contains(@class,'card-entry')]"))
            {
                var cells = SelectNodes(entry, ".//div[contains(@class,'card-cell--sectional-time')]");
                if (cells.Count == 0)
                {
                    continue;
                }

                string horseName = Clean(
                    entry.SelectSingleNode(".//a[contains(@class,'horse__link')]")?.InnerText);
                if (horseName.Length == 0)
                {
                    continue;
                }

                var runner = new AtTheRacesRunnerSectional
                {
                    HorseName = horseName,
                    FinishPos = ParseShort(
                        entry.GetAttributeValue("data-officialplacing", string.Empty)),
                    ClothNumber = ParseClothNumber(entry)
                };

                foreach (var cell in cells)
                {
                    string classes = cell.GetAttributeValue("class", string.Empty);
                    if (SectionalFinishCellPattern.IsMatch(classes))
                    {
                        runner.FinishTimeSeconds = ParseFinishTime(Clean(cell.InnerText));
                        continue;
                    }

                    var timeNode = cell.SelectSingleNode(".//span[contains(@class,'visible')]");
                    double? seconds = ParseNumber(timeNode?.InnerText ?? cell.InnerText);
                    if (!seconds.HasValue)
                    {
                        continue;
                    }

                    var inner = cell.SelectSingleNode(".//div[contains(@class,'card-sectional__inner')]");
                    var cumulativeMatch = Regex.Match(
                        inner?.GetAttributeValue("title", string.Empty) ?? string.Empty,
                        @"Running Total:\s*(?<value>[0-9]+(?:\.[0-9]+)?)",
                        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
                    double? cumulative = cumulativeMatch.Success
                        ? double.Parse(cumulativeMatch.Groups["value"].Value, CultureInfo.InvariantCulture)
                        : null;

                    runner.Splits.Add(new AtTheRacesSplit
                    {
                        SectorNumber = runner.Splits.Count,
                        SectorName = runner.Splits.Count < headerNames.Count
                            ? headerNames[runner.Splits.Count]
                            : null,
                        TimeSeconds = seconds,
                        CumulativeSeconds = cumulative,
                        Position = Clean(
                            cell.SelectSingleNode(".//span[contains(@class,'text-pill')]")?.InnerText),
                        PaceCategory = ReadPaceCategory(cell)
                    });
                }

                if (runner.Splits.Count == 0)
                {
                    continue;
                }

                string key = RunnerMatcher.NormalizeHorseName(runner.HorseName) + "|" + runner.ClothNumber;
                if (!runners.TryGetValue(key, out var existing) ||
                    runner.Splits.Count > existing.Splits.Count)
                {
                    runners[key] = runner;
                }
            }

            return runners.Values.ToList();
        }

        public static IReadOnlyDictionary<string, double> ParseFinishPercentages(string html)
        {
            var document = new HtmlDocument();
            document.LoadHtml(html ?? string.Empty);

            var percentages = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var entry in SelectNodes(document, "//div[contains(@class,'card-entry')]"))
            {
                var valueNode = entry.SelectSingleNode(".//div[contains(@class,'card-cell--horse-finish')]");
                if (valueNode is null)
                {
                    continue;
                }

                string name = RunnerMatcher.NormalizeHorseName(
                    entry.SelectSingleNode(".//a[contains(@class,'horse__link')]")?.InnerText);
                if (name.Length == 0)
                {
                    continue;
                }

                double? value = ParseNumber(Clean(valueNode.InnerText));
                if (value.HasValue && value.Value > 0)
                {
                    percentages[name] = value.Value;
                }
            }

            return percentages;
        }

        public static List<RunnerSectional> BuildRunnerSectionals(
            IReadOnlyList<AtTheRacesRunnerSectional> sourceRunners,
            IReadOnlyList<RaceRunnerReference> databaseRunners,
            int raceId,
            long atrRaceId,
            IReadOnlyDictionary<string, double> finishPercentages)
        {
            var candidates = new List<Candidate>();
            var seenHorseIds = new HashSet<int>();

            foreach (var source in sourceRunners ?? Array.Empty<AtTheRacesRunnerSectional>())
            {
                if (source is null || source.Splits.Count == 0)
                {
                    continue;
                }

                var databaseRunner = RunnerMatcher.FindRunner(
                    databaseRunners,
                    source.ClothNumber,
                    source.HorseName);
                if (databaseRunner is null || !seenHorseIds.Add(databaseRunner.HorseId))
                {
                    continue;
                }

                var candidate = new Candidate(source, databaseRunner);
                candidate.Prepare();
                if (candidate.Times.Count == 0)
                {
                    continue;
                }

                candidates.Add(candidate);
            }

            if (candidates.Count == 0)
            {
                return new List<RunnerSectional>();
            }

            var last1Field = candidates
                .Where(candidate => candidate.Last1Seconds.HasValue)
                .Select(candidate => candidate.Last1Seconds!.Value)
                .ToList();
            var last2Field = candidates
                .Where(candidate => candidate.Last2Seconds.HasValue)
                .Select(candidate => candidate.Last2Seconds!.Value)
                .ToList();
            var last3Field = candidates
                .Where(candidate => candidate.Last3Seconds.HasValue)
                .Select(candidate => candidate.Last3Seconds!.Value)
                .ToList();
            var cumulativeFields = new List<List<double>>();
            int maxSplits = candidates.Max(candidate => candidate.Cumulative.Count);
            for (int index = 0; index < maxSplits; index++)
            {
                cumulativeFields.Add(candidates
                    .Where(candidate => candidate.Cumulative.Count > index)
                    .Select(candidate => candidate.Cumulative[index])
                    .ToList());
            }

            var rows = new List<RunnerSectional>(candidates.Count);
            foreach (var candidate in candidates)
            {
                float? last1Percentile = Percentile(candidate.Last1Seconds, last1Field);
                float? last2Percentile = Percentile(candidate.Last2Seconds, last2Field);
                float? last3Percentile = Percentile(candidate.Last3Seconds, last3Field);
                float? earlyPosition = candidate.Cumulative.Count > 0
                    ? PositionNorm(candidate.Cumulative[0], cumulativeFields[0])
                    : null;
                int closingIndex = candidate.Cumulative.Count - 1;
                float? closingPosition = closingIndex >= 0
                    ? PositionNorm(candidate.Cumulative[closingIndex], cumulativeFields[closingIndex])
                    : null;
                float? paceStyle = earlyPosition.HasValue && closingPosition.HasValue
                    ? closingPosition.Value - earlyPosition.Value
                    : null;

                double? finishSpeedPct = null;
                string normalizedName = RunnerMatcher.NormalizeHorseName(candidate.Source.HorseName);
                if (finishPercentages is not null &&
                    finishPercentages.TryGetValue(normalizedName, out double sourceFinishSpeed) &&
                    sourceFinishSpeed > 0)
                {
                    finishSpeedPct = sourceFinishSpeed;
                }
                else if (candidate.TotalSeconds.HasValue &&
                         candidate.Last1Seconds.HasValue &&
                         candidate.Times.Count > 0 &&
                         candidate.Last1Seconds.Value > 0)
                {
                    finishSpeedPct = 100d * candidate.TotalSeconds.Value /
                        (candidate.Last1Seconds.Value * candidate.Times.Count);
                }

                string rawJson = JsonSerializer.Serialize(new
                {
                    source = "AtTheRaces",
                    sourceRaceId = atrRaceId,
                    finishTimeSeconds = candidate.TotalSeconds,
                    finishSpeedPct,
                    splits = candidate.Source.Splits.Select(split => new
                    {
                        split.SectorNumber,
                        split.SectorName,
                        split.TimeSeconds,
                        split.CumulativeSeconds,
                        split.Position,
                        split.PaceCategory
                    })
                });

                rows.Add(new RunnerSectional
                {
                    RaceId = raceId,
                    HorseId = candidate.DatabaseRunner.HorseId,
                    FinishPos = candidate.Source.FinishPos,
                    ClothNumber = candidate.Source.ClothNumber.HasValue &&
                                  candidate.Source.ClothNumber.Value >= 0 &&
                                  candidate.Source.ClothNumber.Value <= byte.MaxValue
                        ? (byte?)candidate.Source.ClothNumber.Value
                        : candidate.DatabaseRunner.SaddleclothNumber,
                    SectionCount = candidate.Times.Count <= byte.MaxValue
                        ? (byte?)candidate.Times.Count
                        : (byte?)byte.MaxValue,
                    Last1FSeconds = candidate.Last1Seconds.HasValue
                        ? (float?)candidate.Last1Seconds.Value
                        : null,
                    Last2FSeconds = candidate.Last2Seconds.HasValue
                        ? (float?)candidate.Last2Seconds.Value
                        : null,
                    Last3FSeconds = candidate.Last3Seconds.HasValue
                        ? (float?)candidate.Last3Seconds.Value
                        : null,
                    Last1FPercentile = last1Percentile,
                    Last2FPercentile = last2Percentile,
                    Last3FPercentile = last3Percentile,
                    EarlyPositionNorm = earlyPosition,
                    ClosingPositionNorm = closingPosition,
                    PaceStyle = paceStyle,
                    FinishSpeedPct = finishSpeedPct.HasValue ? (float?)finishSpeedPct.Value : null,
                    TimeIndexScore = null,
                    MeetingAverageTimeIndex = null,
                    WinTimeVsPar = null,
                    SourceRaceId = atrRaceId.ToString(CultureInfo.InvariantCulture),
                    RawJson = rawJson,
                    Source = "AtTheRaces",
                    FetchedUtc = DateTime.UtcNow
                });
            }

            return rows;
        }

        private static long ReadRaceId(HtmlNode article)
        {
            var link = article.SelectSingleNode(".//a[contains(@href,'raceid=')]");
            var match = RaceIdPattern.Match(link?.GetAttributeValue("href", string.Empty) ?? string.Empty);
            if (match.Success && long.TryParse(match.Groups[1].Value, out long raceId))
            {
                return raceId;
            }

            var videoNode = article.SelectSingleNode(".//*[@data-video-raceid]");
            return long.TryParse(videoNode?.GetAttributeValue("data-video-raceid", string.Empty), out raceId)
                ? raceId
                : 0;
        }

        private static bool TryParseRacePath(
            string href,
            out string course,
            out DateTime localDate,
            out TimeSpan localTime)
        {
            course = string.Empty;
            localDate = default;
            localTime = default;

            var segments = (href ?? string.Empty)
                .Split('/', StringSplitOptions.RemoveEmptyEntries);
            if (segments.Length < 4 ||
                !segments[0].Equals("racecard", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            course = Uri.UnescapeDataString(segments[1]).Replace('-', ' ').Trim();
            string dateText = Uri.UnescapeDataString(segments[2]);
            if (!DateTime.TryParseExact(
                    dateText,
                    new[] { "dd-MMMM-yyyy", "d-MMMM-yyyy" },
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.None,
                    out localDate))
            {
                return false;
            }

            string timeText = Uri.UnescapeDataString(segments[3]);
            if (!int.TryParse(timeText, NumberStyles.None, CultureInfo.InvariantCulture, out int clock) ||
                clock < 0 || clock > 2359)
            {
                return false;
            }

            localTime = new TimeSpan(clock / 100, clock % 100, 0);
            return true;
        }

        private static int? ParseClothNumber(HtmlNode entry)
        {
            var node = entry.SelectSingleNode(
                ".//h2[contains(@class,'horse__details')]//span[contains(@class,'p--small')]");
            var match = Regex.Match(Clean(node?.InnerText), @"^\s*(\d{1,3})\b");
            return match.Success && int.TryParse(match.Groups[1].Value, out int cloth)
                ? cloth
                : null;
        }

        private static short? ParseShort(string value)
            => short.TryParse(Clean(value), NumberStyles.Integer, CultureInfo.InvariantCulture, out short parsed)
                ? parsed
                : null;

        private static double? ParseFinishTime(string value)
        {
            var match = FinishTimePattern.Match(value ?? string.Empty);
            if (!match.Success)
            {
                return null;
            }

            double minutes = match.Groups["minutes"].Success
                ? double.Parse(match.Groups["minutes"].Value, CultureInfo.InvariantCulture)
                : 0;
            double seconds = double.Parse(match.Groups["seconds"].Value, CultureInfo.InvariantCulture);
            return minutes * 60 + seconds;
        }

        private static double? ParseNumber(string value)
        {
            var match = NumberPattern.Match(value ?? string.Empty);
            return match.Success &&
                   double.TryParse(match.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) &&
                   double.IsFinite(parsed)
                ? parsed
                : null;
        }

        private static string? ReadPaceCategory(HtmlNode cell)
        {
            var sectional = cell.SelectSingleNode(".//div[contains(@class,'card-sectional')]");
            string classes = sectional?.GetAttributeValue("class", string.Empty) ?? string.Empty;
            var match = Regex.Match(
                classes,
                @"card-sectional--(?<name>[a-z-]+)",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (!match.Success)
            {
                return null;
            }

            string name = match.Groups["name"].Value;
            return name.Equals("finish", StringComparison.OrdinalIgnoreCase) ? null : name;
        }

        private static float? Percentile(double? value, IReadOnlyList<double> field)
        {
            if (!value.HasValue || field is null || field.Count < 2)
            {
                return null;
            }

            int better = field.Count(candidate => candidate < value.Value - 1e-9);
            return Math.Clamp(1f - (float)better / (field.Count - 1), 0f, 1f);
        }

        private static float? PositionNorm(double value, IReadOnlyList<double> field)
        {
            if (field is null || field.Count < 2)
            {
                return null;
            }

            int better = field.Count(candidate => candidate < value - 1e-9);
            return Math.Clamp(1f - (float)better / (field.Count - 1), 0f, 1f);
        }

        private static List<HtmlNode> SelectNodes(HtmlNode node, string xpath)
            => node.SelectNodes(xpath)?.ToList() ?? new List<HtmlNode>();

        private static List<HtmlNode> SelectNodes(HtmlDocument document, string xpath)
            => document.DocumentNode.SelectNodes(xpath)?.ToList() ?? new List<HtmlNode>();

        private static string Clean(string? value)
            => Regex.Replace(value ?? string.Empty, @"\s+", " ").Trim();

        private sealed class Candidate
        {
            public Candidate(AtTheRacesRunnerSectional source, RaceRunnerReference databaseRunner)
            {
                Source = source;
                DatabaseRunner = databaseRunner;
            }

            public AtTheRacesRunnerSectional Source { get; }
            public RaceRunnerReference DatabaseRunner { get; }
            public List<double> Times { get; } = new();
            public List<double> Cumulative { get; } = new();
            public double? Last1Seconds { get; private set; }
            public double? Last2Seconds { get; private set; }
            public double? Last3Seconds { get; private set; }
            public double? TotalSeconds { get; private set; }

            public void Prepare()
            {
                foreach (var split in Source.Splits.OrderBy(split => split.SectorNumber))
                {
                    if (!split.TimeSeconds.HasValue || !double.IsFinite(split.TimeSeconds.Value))
                    {
                        continue;
                    }

                    Times.Add(split.TimeSeconds.Value);
                }

                if (Times.Count == 0)
                {
                    return;
                }

                Last1Seconds = Times[^1];
                if (Times.Count >= 2)
                {
                    Last2Seconds = Times.Skip(Times.Count - 2).Sum();
                }
                if (Times.Count >= 3)
                {
                    Last3Seconds = Times.Skip(Times.Count - 3).Sum();
                }

                bool hasCumulative = Source.Splits
                    .Take(Times.Count)
                    .All(split => split.CumulativeSeconds.HasValue);
                if (hasCumulative)
                {
                    Cumulative.AddRange(Source.Splits
                        .Take(Times.Count)
                        .Select(split => split.CumulativeSeconds!.Value));
                }
                else
                {
                    double running = 0;
                    foreach (double time in Times)
                    {
                        running += time;
                        Cumulative.Add(running);
                    }
                }

                TotalSeconds = Source.FinishTimeSeconds.HasValue
                    ? Source.FinishTimeSeconds.Value
                    : Times.Sum();
            }
        }
    }
}
