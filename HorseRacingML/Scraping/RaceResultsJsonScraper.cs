using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Text.Json;
using System.Text.RegularExpressions;
using HorseRacingML.Data;
using HorseRacingML.Models;
using HorseRacingML.Services;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Fast historical backfill. Every Sporting Life race page is server-rendered with
    /// the complete result set inside the __NEXT_DATA__ JSON blob, so this scraper uses
    /// HttpClient + JSON instead of Selenium/Chrome. It is roughly an order of magnitude
    /// faster than <see cref="RaceResultsScraper"/> and uses far less CPU.
    /// </summary>
    public class RaceResultsJsonScraper
    {
        private static readonly Regex NextDataRegex = new(
            "<script id=\"__NEXT_DATA__\"[^>]*>(.*?)</script>",
            RegexOptions.Singleline | RegexOptions.Compiled);

        private static readonly Regex RaceLinkRegex = new(
            "/racing/results/(?<date>\\d{4}-\\d{2}-\\d{2})/[^\"'\\s#?<>]+",
            RegexOptions.Compiled);

        private readonly RacingRepository _repo;
        private readonly ScrapingStatusService _status;
        private readonly HttpClient _http;
        private readonly ConcurrentDictionary<string, Lazy<int>> _trainerIds = new(StringComparer.OrdinalIgnoreCase);
        private readonly ConcurrentDictionary<string, Lazy<int>> _jockeyIds = new(StringComparer.OrdinalIgnoreCase);
        private readonly ConcurrentDictionary<string, Lazy<int>> _horseIds = new(StringComparer.OrdinalIgnoreCase);

        public RaceResultsJsonScraper(RacingRepository repo, ScrapingStatusService status)
        {
            _repo = repo;
            _status = status;
            var handler = new SocketsHttpHandler
            {
                AutomaticDecompression = DecompressionMethods.All,
                MaxConnectionsPerServer = 32
            };
            _http = new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(60) };
            _http.DefaultRequestHeaders.UserAgent.ParseAdd(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36");
            _http.DefaultRequestHeaders.Accept.ParseAdd("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
        }

        /// <summary>
        /// Scrapes every date from <paramref name="oldestDate"/> to today that does not
        /// already have a complete result set. Safe to stop and restart: completed dates
        /// are skipped and re-scraping a partial date de-duplicates on (RaceId, HorseId).
        /// </summary>
        public void Backfill(DateTime oldestDate, int parallelism = 16)
            => BackfillRange(oldestDate, DateTime.Today, parallelism);

        /// <summary>
        /// Scrapes every date in the inclusive range that does not already have a
        /// complete result set. Safe to stop and restart.
        /// </summary>
        public void BackfillRange(
            DateTime oldestDate,
            DateTime newestDate,
            int parallelism = 16,
            bool force = false,
            bool priceDataOnly = false,
            bool marketMovementOnly = false)
        {
            var today = DateTime.Today;
            if (newestDate > today) newestDate = today;
            if (oldestDate > newestDate) return;

            var completed = force
                ? new HashSet<DateTime>()
                : _repo.GetDatesWithCompleteResults(oldestDate, newestDate);
            var dates = new List<DateTime>();
            for (var d = newestDate; d >= oldestDate; d = d.AddDays(-1))
            {
                if (!completed.Contains(d)) dates.Add(d);
            }

            Console.WriteLine($"[backfill] {dates.Count} days to scrape ({oldestDate:yyyy-MM-dd}..{newestDate:yyyy-MM-dd}), parallelism {parallelism}");
            _status.Update($"[backfill] {dates.Count} days queued");

            var stopwatch = Stopwatch.StartNew();
            var doneDays = 0;
            long totalRaces = 0;
            long totalRunners = 0;
            var lastReport = DateTime.UtcNow;
            var failures = 0;

            Parallel.ForEach(
                dates,
                new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, parallelism) },
                date =>
                {
                    try
                    {
                        var (races, runners) = ScrapeDay(date, priceDataOnly, marketMovementOnly);
                        Interlocked.Add(ref totalRaces, races);
                        Interlocked.Add(ref totalRunners, runners);
                    }
                    catch (Exception ex)
                    {
                        Interlocked.Increment(ref failures);
                        Console.Error.WriteLine($"[backfill] {date:yyyy-MM-dd} failed: {ex.Message}");
                    }

                    var n = Interlocked.Increment(ref doneDays);
                    if (n % 10 == 0 || n == dates.Count)
                    {
                        var elapsed = stopwatch.Elapsed;
                        var rate = n / Math.Max(1.0, elapsed.TotalMinutes);
                        var remaining = dates.Count - n;
                        var eta = rate > 0 ? TimeSpan.FromMinutes(remaining / rate) : TimeSpan.Zero;
                        var msg = $"[backfill] {n}/{dates.Count} days, {Interlocked.Read(ref totalRaces)} races, {Interlocked.Read(ref totalRunners)} runners, {rate:F1} days/min, ETA {eta:hh\\:mm\\:ss}, {failures} failures";
                        Console.WriteLine(msg);
                        _status.Update(msg);
                    }
                });

            var summary = $"[backfill] finished in {stopwatch.Elapsed:hh\\:mm\\:ss}: {Interlocked.Read(ref totalRaces)} races, {Interlocked.Read(ref totalRunners)} runners, {failures} failed days";
            Console.WriteLine(summary);
            _status.Update(summary);
        }

        private (int races, int runners) ScrapeDay(
            DateTime date,
            bool priceDataOnly = false,
            bool marketMovementOnly = false)
        {
            var dayUrl = $"https://www.sportinglife.com/racing/results/{date:yyyy-MM-dd}";
            var html = GetWithRetry(dayUrl);
            if (html == null) return (0, 0);

            var urls = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (Match match in RaceLinkRegex.Matches(html))
            {
                if (!DateTime.TryParseExact(match.Groups["date"].Value, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var linkDate) || linkDate.Date != date.Date)
                {
                    continue;
                }
                var url = "https://www.sportinglife.com" + match.Value;
                urls.Add(url);
            }

            var races = 0;
            var runners = 0;
            foreach (var url in urls)
            {
                var inserted = 0;
                for (var attempt = 0; attempt < 3; attempt++)
                {
                    try
                    {
                        inserted = ScrapeRace(url, priceDataOnly, marketMovementOnly);
                        break;
                    }
                    catch (Microsoft.Data.SqlClient.SqlException ex) when (ex.Number == 1205 && attempt < 2)
                    {
                        // Another worker's reference-data transaction won the deadlock.
                        Thread.Sleep(250 * (attempt + 1));
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"[backfill] {url} failed: {ex.Message}");
                        break;
                    }
                }

                if (inserted > 0)
                {
                    races++;
                    runners += inserted;
                }
            }

            Console.WriteLine($"[backfill] {date:yyyy-MM-dd}: {urls.Count} races, {runners} runners");
            return (races, runners);
        }

        private int ScrapeRace(
            string url,
            bool priceDataOnly = false,
            bool marketMovementOnly = false)
        {
            var html = GetWithRetry(url);
            if (html == null) return 0;

            var match = NextDataRegex.Match(html);
            if (!match.Success) return 0;

            using var document = JsonDocument.Parse(match.Groups[1].Value);
            var root = document.RootElement;
            if (!root.TryGetProperty("props", out var props) || !props.TryGetProperty("pageProps", out var pageProps)) return 0;
            if (!pageProps.TryGetProperty("race", out var raceElement) || raceElement.ValueKind != JsonValueKind.Object) return 0;
            if (!raceElement.TryGetProperty("race_summary", out var summary)) return 0;

            var dateText = Str(summary, "date");
            if (!DateTime.TryParseExact(dateText, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var raceDate)) return 0;

            var courseName = Str(summary, "course_name");
            if (string.IsNullOrWhiteSpace(courseName)) return 0;

            var distanceText = Str(summary, "distance");
            var going = Str(summary, "going");
            var ageRestriction = Str(summary, "age");
            var winningTime = Str(summary, "winning_time");
            var raceClass = Str(summary, "race_class");
            var surfaceRaw = summary.TryGetProperty("course_surface", out var surfaceEl) ? Str(surfaceEl, "surface") : null;
            var surface = string.Equals(surfaceRaw, "TURF", StringComparison.OrdinalIgnoreCase) ? "Turf" : "Allweather";
            var runnerCount = Int(summary, "ride_count");
            var scheduledOff = RaceResultsScraper.ParseClockTime(Str(summary, "time"));
            var actualOff = RaceResultsScraper.ParseClockTime(Str(summary, "off_time"));

            var courseId = _repo.InsertCourse(new Course { Name = courseName });
            var raceId = _repo.InsertRace(new Race
            {
                CourseId = courseId,
                RaceDate = raceDate,
                ScheduledOff = scheduledOff ?? TimeSpan.Zero,
                ActualOff = actualOff,
                Title = Str(summary, "name") ?? string.Empty,
                RaceType = string.Empty,
                Class = byte.TryParse(raceClass, NumberStyles.Integer, CultureInfo.InvariantCulture, out var classValue) ? classValue : (byte?)null,
                AgeRestriction = ageRestriction,
                Surface = surface,
                Going = going,
                DistanceYards = (short)RaceResultsScraper.ParseDistanceToYards(distanceText),
                DistanceText = distanceText ?? string.Empty,
                RunnerCount = runnerCount.HasValue ? (byte?)Math.Min(runnerCount.Value, byte.MaxValue) : null,
                Status = MapRaceStage(Str(summary, "race_stage")),
                WinningTimeMs = RaceResultsScraper.ParseWinningMs(winningTime),
                WinningTimeText = winningTime,
                Purse = ParsePurse(raceElement),
                Source = "SportingLife",
                SourceRaceId = Truncate(url, 128)
            });

            if (!raceElement.TryGetProperty("rides", out var ridesElement) || ridesElement.ValueKind != JsonValueKind.Array) return 0;

            var rawRides = new List<RawRide>();
            foreach (var ride in ridesElement.EnumerateArray())
            {
                var horse = ride.TryGetProperty("horse", out var h) && h.ValueKind == JsonValueKind.Object ? h : default;
                var trainer = ride.TryGetProperty("trainer", out var t) && t.ValueKind == JsonValueKind.Object ? t : default;
                var jockey = ride.TryGetProperty("jockey", out var j) && j.ValueKind == JsonValueKind.Object ? j : default;
                var betting = ride.TryGetProperty("betting", out var b) && b.ValueKind == JsonValueKind.Object ? b : default;
                var casualty = ride.TryGetProperty("casualty", out var c) && c.ValueKind == JsonValueKind.Object ? c : default;

                var finishPosition = Int(ride, "finish_position");
                var rideStatus = Str(ride, "ride_status");
                var casualtyReason = casualty.ValueKind == JsonValueKind.Object ? Str(casualty, "reason") : null;
                var rideComment = Str(ride, "ride_description");

                // The race page's rides array includes every declared entry, not just the
                // actual field. Non-runners have finish_position 0, no casualty and often
                // a "NonRunner"/"Withdrawn" comment. Only keep real runners and genuine
                // non-finishers (which have a casualty reason or a casualty comment).
                var isNonRunner = string.Equals(rideStatus, "NONRUNNER", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(rideStatus, "WITHDRAWN", StringComparison.OrdinalIgnoreCase)
                    || (!string.IsNullOrWhiteSpace(rideComment) && Regex.IsMatch(rideComment, @"non[\s-]*runner|with[\s-]*drawn", RegexOptions.IgnoreCase));
                if (isNonRunner) continue;

                if (!finishPosition.HasValue || finishPosition.Value <= 0)
                {
                    var hasCasualty = !string.IsNullOrWhiteSpace(casualtyReason);
                    var looksLikeCasualty = !string.IsNullOrWhiteSpace(rideComment) && Regex.IsMatch(
                        rideComment,
                        @"pulled up|puled up|slipped up|fell|unseated|brought down|refused|ran out|disqualified|left at start",
                        RegexOptions.IgnoreCase);
                    if (!hasCasualty && !looksLikeCasualty) continue;
                }

                rawRides.Add(new RawRide
                {
                    Horse = CleanName(horse.ValueKind == JsonValueKind.Object ? Str(horse, "name") : null),
                    Age = horse.ValueKind == JsonValueKind.Object ? Int(horse, "age") : null,
                    Trainer = CleanName(trainer.ValueKind == JsonValueKind.Object ? Str(trainer, "name") : null),
                    Jockey = CleanName(jockey.ValueKind == JsonValueKind.Object ? Str(jockey, "name") : null),
                    Cloth = Int(ride, "cloth_number"),
                    OfficialRating = Int(ride, "official_rating"),
                    Headgear = NamedArray(ride, "headgear"),
                    Medication = NamedArray(ride, "medication"),
                    FinishPos = finishPosition is > 0 ? finishPosition : null,
                    Status = rideStatus,
                    Casualty = casualtyReason,
                    WeightText = Str(ride, "handicap"),
                    Odds = betting.ValueKind == JsonValueKind.Object ? Str(betting, "current_odds") : null,
                    Favourite = betting.ValueKind == JsonValueKind.Object && betting.TryGetProperty("favourite", out var fav) && fav.ValueKind == JsonValueKind.Object
                        ? Str(fav, "betting_favourite")
                        : null,
                    Comment = rideComment,
                    BetMovements = Str(ride, "bet_movements"),
                    HistoricalOdds = ReadHistoricalOdds(betting)
                });
            }

            if (rawRides.Count == 0) return 0;

            var refs = _repo.EnsureReferenceData(
                rawRides.Select(r => r.Horse),
                rawRides.Select(r => r.Trainer),
                rawRides.Select(r => r.Jockey));
            foreach (var kv in refs.Horses) _horseIds.TryAdd(kv.Key, new Lazy<int>(() => kv.Value));
            foreach (var kv in refs.Trainers) _trainerIds.TryAdd(kv.Key, new Lazy<int>(() => kv.Value));
            foreach (var kv in refs.Jockeys) _jockeyIds.TryAdd(kv.Key, new Lazy<int>(() => kv.Value));

            var results = new List<RunnerResult>();
            foreach (var ride in rawRides)
            {
                if (string.IsNullOrWhiteSpace(ride.Horse)) continue;

                var odds = ride.Odds;
                var weightText = ride.WeightText ?? string.Empty;
                var weightMatch = Regex.Match(weightText, @"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$");
                byte? weightLbs = null;
                if (weightMatch.Success)
                {
                    var total = int.Parse(weightMatch.Groups[1].Value) * 14 + int.Parse(weightMatch.Groups[2].Value);
                    if (total >= byte.MinValue && total <= byte.MaxValue) weightLbs = (byte)total;
                }

                var (touchedLow, touchedHigh) = RaceResultsScraper.ExtractTouchedTokens(ride.BetMovements);
                var openingToken = RaceResultsScraper.ExtractOddsToken(ride.BetMovements, "op");
                var oddsRange = MarketOddsParser.BuildRange(
                    ride.HistoricalOdds,
                    openingToken,
                    touchedLow,
                    touchedHigh);

                results.Add(new RunnerResult
                {
                    RaceId = raceId,
                    HorseId = ResolveCachedId(_horseIds, ride.Horse, () => _repo.InsertHorse(new Horse { Name = ride.Horse })),
                    TrainerId = string.IsNullOrWhiteSpace(ride.Trainer) ? null : (short)ResolveCachedId(_trainerIds, ride.Trainer, () => _repo.InsertTrainer(new Trainer { Name = ride.Trainer })),
                    JockeyId = string.IsNullOrWhiteSpace(ride.Jockey) ? null : (short)ResolveCachedId(_jockeyIds, ride.Jockey, () => _repo.InsertJockey(new Jockey { Name = ride.Jockey })),
                    SaddleclothNumber = ride.Cloth.HasValue ? (byte?)Math.Min(ride.Cloth.Value, byte.MaxValue) : null,
                    Draw = null,
                    Age = ride.Age.HasValue ? (byte?)Math.Min(ride.Age.Value, byte.MaxValue) : null,
                    WeightLbs = weightLbs,
                    WeightText = weightText,
                    OfficialRating = ride.OfficialRating.HasValue
                        ? (short?)Math.Min(ride.OfficialRating.Value, short.MaxValue)
                        : null,
                    Headgear = ride.Headgear,
                    Medication = ride.Medication,
                    FinishPos = ride.FinishPos.HasValue ? (short?)Math.Min(ride.FinishPos.Value, short.MaxValue) : null,
                    OutcomeCode = MapOutcome(ride),
                    DistanceBeatenText = null,
                    DistanceBeatenLengths = null,
                    SP_Fraction = odds,
                    SP_Decimal = RaceResultsScraper.FractionToDecimal(odds ?? string.Empty),
                    FavTag = MapFavourite(ride.Favourite),
                    OpeningFraction = oddsRange.OpeningToken ?? openingToken,
                    TouchedHighFraction = oddsRange.HighToken ?? touchedHigh,
                    TouchedLowFraction = oddsRange.LowToken ?? touchedLow,
                    Comment = ride.Comment,
                    Source = "SportingLife",
                    SourceRunnerId = Truncate(
                        $"{url}#{ride.Cloth?.ToString(CultureInfo.InvariantCulture) ?? ride.Horse}",
                        128)
                });
            }

            if (results.Count == 0) return 0;
            if (marketMovementOnly)
            {
                _repo.BulkUpdateMarketMovement(results);
            }
            else if (priceDataOnly)
            {
                _repo.BulkUpdatePriceData(results);
            }
            else
            {
                _repo.BulkInsertRunnerResults(results);
            }
            return results.Count;
        }

        private string? GetWithRetry(string url)
        {
            for (var attempt = 0; attempt < 3; attempt++)
            {
                try
                {
                    return _http.GetStringAsync(url).GetAwaiter().GetResult();
                }
                catch (HttpRequestException ex) when (attempt < 2)
                {
                    Console.Error.WriteLine($"[backfill] GET {url} attempt {attempt + 1}: {ex.StatusCode}");
                    Thread.Sleep(500 * (attempt + 1));
                }
                catch (TaskCanceledException) when (attempt < 2)
                {
                    Thread.Sleep(500 * (attempt + 1));
                }
            }
            return null;
        }

        private static string MapRaceStage(string? stage)
        {
            if (string.IsNullOrWhiteSpace(stage)) return string.Empty;
            return stage.Trim().ToUpperInvariant() switch
            {
                "WEIGHEDIN" => "Weighed In",
                "RESULT" => "Result",
                "ABANDONED" => "Abandoned",
                "VOID" => "Void",
                _ => stage
            };
        }

        private static string MapFavourite(string? favourite)
        {
            if (string.IsNullOrWhiteSpace(favourite)) return string.Empty;
            var value = favourite.Trim().ToUpperInvariant();
            return value switch
            {
                "F" => "F",
                "JF" => "JF",
                "CF" => "CF",
                _ => value
            };
        }

        private static string MapOutcome(RawRide ride)
        {
            if (ride.FinishPos.HasValue && ride.FinishPos.Value > 0) return string.Empty;
            var reason = ride.Casualty ?? string.Empty;
            if (string.IsNullOrWhiteSpace(reason) && !string.IsNullOrWhiteSpace(ride.Comment))
            {
                var comment = ride.Comment;
                if (Regex.IsMatch(comment, @"pulled up|puled up", RegexOptions.IgnoreCase)) reason = "PulledUp";
                else if (Regex.IsMatch(comment, @"slipped up", RegexOptions.IgnoreCase)) reason = "SlippedUp";
                else if (Regex.IsMatch(comment, @"unseated", RegexOptions.IgnoreCase)) reason = "UnseatedRider";
                else if (Regex.IsMatch(comment, @"brought down", RegexOptions.IgnoreCase)) reason = "BroughtDown";
                else if (Regex.IsMatch(comment, @"refused", RegexOptions.IgnoreCase)) reason = "Refused";
                else if (Regex.IsMatch(comment, @"ran out", RegexOptions.IgnoreCase)) reason = "RanOut";
                else if (Regex.IsMatch(comment, @"disqualified", RegexOptions.IgnoreCase)) reason = "Disqualified";
                else if (Regex.IsMatch(comment, @"left at start", RegexOptions.IgnoreCase)) reason = "LeftAtStart";
                else if (Regex.IsMatch(comment, @"\bfell\b", RegexOptions.IgnoreCase)) reason = "Fell";
            }
            if (string.IsNullOrWhiteSpace(reason)) reason = ride.Status ?? string.Empty;
            var normalized = reason.Replace(" ", "").Replace("_", "").ToUpperInvariant();
            return normalized switch
            {
                "PULLEDUP" => "PU",
                "PULEDUP" => "PU",
                "FELL" => "F",
                "UNSEATEDRIDER" => "UR",
                "BROUGHTDOWN" => "BD",
                "REFUSED" => "REF",
                "RANOUT" => "RO",
                "DISQUALIFIED" => "DSQ",
                "SLIPPEDUP" => "SU",
                "LEFTATSTART" => "LFT",
                "WITHDRAWN" => "WD",
                "NONRUNNER" => "NR",
                "RUNNER" => string.Empty,
                _ => RaceResultsScraper.ParseOutcomeCode(normalized)
            };
        }

        private static List<string?> ReadHistoricalOdds(JsonElement betting)
        {
            var odds = new List<string?>();
            if (betting.ValueKind != JsonValueKind.Object ||
                !betting.TryGetProperty("historical_odds", out var historical) ||
                historical.ValueKind != JsonValueKind.Array)
            {
                return odds;
            }

            foreach (var item in historical.EnumerateArray())
            {
                odds.Add(item.ValueKind == JsonValueKind.String ? item.GetString() : null);
            }

            return odds;
        }

        private static string? Str(JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object || !element.TryGetProperty(name, out var value)) return null;
            return value.ValueKind switch
            {
                JsonValueKind.String => value.GetString(),
                JsonValueKind.Number => value.ToString(),
                _ => null
            };
        }

        private static int? Int(JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object || !element.TryGetProperty(name, out var value)) return null;
            if (value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var number)) return number;
            if (value.ValueKind == JsonValueKind.String && int.TryParse(value.GetString(), out number)) return number;
            return null;
        }

        private static string? NamedArray(JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object ||
                !element.TryGetProperty(name, out var value) ||
                value.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            var names = value
                .EnumerateArray()
                .Select(item => Str(item, "name"))
                .Where(item => !string.IsNullOrWhiteSpace(item))
                .ToList();
            return names.Count == 0 ? null : string.Join(", ", names);
        }

        private static decimal? ParsePurse(JsonElement summary)
        {
            if (summary.ValueKind != JsonValueKind.Object ||
                !summary.TryGetProperty("prizes", out var prizes) ||
                prizes.ValueKind != JsonValueKind.Object ||
                !prizes.TryGetProperty("prize", out var prizeArray) ||
                prizeArray.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            decimal total = 0m;
            bool found = false;
            foreach (var prize in prizeArray.EnumerateArray())
            {
                var text = Str(prize, "prize");
                if (string.IsNullOrWhiteSpace(text))
                {
                    continue;
                }

                var amount = text.Split(' ', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault();
                if (decimal.TryParse(amount, NumberStyles.Number, CultureInfo.InvariantCulture, out var value))
                {
                    total += value;
                    found = true;
                }
            }

            return found ? total : (decimal?)null;
        }

        private static readonly Regex BracketTextRegex = new(@"\s*\([^\)]*\)|\s*\[[^\]]*\]", RegexOptions.Compiled);
        private static string CleanName(string? value)
            => string.IsNullOrWhiteSpace(value) ? string.Empty : BracketTextRegex.Replace(value, string.Empty).Trim();

        private static string? Truncate(string? value, int maxLength)
            => string.IsNullOrEmpty(value) || value.Length <= maxLength
                ? value
                : value.Substring(0, maxLength);

        private static int ResolveCachedId(ConcurrentDictionary<string, Lazy<int>> cache, string name, Func<int> insert)
            => cache.GetOrAdd(name, n => new Lazy<int>(insert, LazyThreadSafetyMode.ExecutionAndPublication)).Value;

        private sealed class RawRide
        {
            public string Horse { get; set; } = string.Empty;
            public int? Age { get; set; }
            public string Trainer { get; set; } = string.Empty;
            public string Jockey { get; set; } = string.Empty;
            public int? Cloth { get; set; }
            public int? OfficialRating { get; set; }
            public string? Headgear { get; set; }
            public string? Medication { get; set; }
            public int? FinishPos { get; set; }
            public string? Status { get; set; }
            public string? Casualty { get; set; }
            public string? WeightText { get; set; }
            public string? Odds { get; set; }
            public string? Favourite { get; set; }
            public string? Comment { get; set; }
            public string? BetMovements { get; set; }
            public List<string?> HistoricalOdds { get; set; } = new();
        }
    }
}
