using System;
using System.Collections.Generic;
using System.Formats.Tar;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using ICSharpCode.SharpZipLib.BZip2;
using HorseRacingML.Models;

namespace HorseRacingML.Data
{
    /// <summary>
    /// Imports Betfair Historical Data / Stream API files. The files are
    /// newline-delimited JSON; one market is normally one file, but the event
    /// (E) downloads interleave several markets in one file, so the importer
    /// accumulates each market independently.
    ///
    /// BASIC data (free) provides last traded prices at one-minute intervals
    /// and each runner's BSP. ADVANCED/PRO data additionally provides traded
    /// volume and the available-to-back/lay ladder. Missing fields stay NULL;
    /// the feature block has explicit missing indicators for that case.
    /// </summary>
    public sealed class BetfairHistoricalDataImporter
    {
        private static readonly string[] SupportedExtensions =
        {
            ".json", ".jsonl", ".txt", ".gz", ".tar", ".bz2"
        };

        private static readonly Regex LeadingRunnerNumberRegex = new(
            @"^\s*\d+\s*[\.\)\-:]?\s+",
            RegexOptions.Compiled);

        private static readonly Regex BracketedTextRegex = new(
            @"\s*[\(\[][^\)\]]*[\)\]]\s*",
            RegexOptions.Compiled);

        private static readonly Regex NonAlphaNumericRegex = new(
            @"[^a-z0-9]+",
            RegexOptions.Compiled);

        private static readonly Dictionary<string, string> WindowsTimeZones = new(
            StringComparer.OrdinalIgnoreCase)
        {
            ["Europe/London"] = "GMT Standard Time",
            ["Europe/Dublin"] = "GMT Standard Time",
            ["Europe/Paris"] = "Romance Standard Time",
            ["America/New_York"] = "Eastern Standard Time",
            ["America/Chicago"] = "Central Standard Time",
            ["America/Los_Angeles"] = "Pacific Standard Time",
            ["Australia/Sydney"] = "AUS Eastern Standard Time",
            ["Australia/Melbourne"] = "AUS Eastern Standard Time",
            ["Australia/Brisbane"] = "E. Australia Standard Time",
            ["Australia/Adelaide"] = "Cen. Australia Standard Time",
            ["Australia/Perth"] = "W. Australia Standard Time",
            ["Asia/Hong_Kong"] = "China Standard Time",
            ["Asia/Tokyo"] = "Tokyo Standard Time",
            ["Asia/Dubai"] = "Arabian Standard Time",
            ["Africa/Johannesburg"] = "South Africa Standard Time",
            ["UTC"] = "UTC"
        };

        private const int DefaultMarketAnchorMinutes = 5;

        private readonly BetfairHistoricalRepository _repository;
        private readonly Action<string>? _log;
        private readonly int _marketAnchorMinutes;
        private readonly Dictionary<string, BetfairCourseRow?> _courseCache =
            new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<(short CourseId, DateTime Date), IReadOnlyList<BetfairRaceCandidate>> _raceCache = new();
        private readonly Dictionary<int, IReadOnlyList<BetfairRunnerRow>> _runnerCache = new();
        private IReadOnlyList<BetfairCourseRow>? _courses;

        public BetfairHistoricalDataImporter(
            BetfairHistoricalRepository repository,
            Action<string>? log = null,
            int marketAnchorMinutes = DefaultMarketAnchorMinutes)
        {
            _repository = repository ?? throw new ArgumentNullException(nameof(repository));
            _log = log;
            _marketAnchorMinutes = Math.Max(0, marketAnchorMinutes);
        }

        private IReadOnlyList<BetfairCourseRow> Courses =>
            _courses ??= _repository.GetCourses().ToList();

        /// <summary>
        /// Imports a file, or every supported file under a directory. Returns
        /// counts for a summary report.
        /// </summary>
        public BetfairImportSummary ImportPath(string path, bool dryRun = false)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentException("A file or directory path is required.", nameof(path));
            }

            var summary = new ImportSummaryBuilder();
            if (Directory.Exists(path))
            {
                var allFiles = Directory
                    .EnumerateFiles(path, "*", SearchOption.AllDirectories)
                    .ToList();
                var files = allFiles
                    .Where(IsSupportedFile)
                    .OrderBy(file => file, StringComparer.OrdinalIgnoreCase)
                    .ToList();

                Log($"[betfair] {files.Count} supported file(s) under {path}");
                foreach (var file in files)
                {
                    try
                    {
                        ImportFile(file, dryRun, summary);
                    }
                    catch (Exception ex)
                    {
                        Log($"[betfair] {file} failed: {ex.Message}");
                        summary.Failures++;
                    }
                }
            }
            else if (File.Exists(path))
            {
                ImportFile(path, dryRun, summary);
            }
            else
            {
                throw new FileNotFoundException("Betfair historical data path was not found.", path);
            }

            var result = summary.Build();
            Log(
                $"[betfair] finished: {result.FilesRead} file(s), {result.MarketsRead} market(s), " +
                $"{result.MarketsMatched} matched race(s), {result.RunnersMatched} matched runner(s), " +
                $"{result.MarketsSkipped} skipped market(s)");
            return result;
        }

        private static bool IsSupportedFile(string path)
        {
            var name = path.ToLowerInvariant();
            if (name.EndsWith(".tar.gz", StringComparison.Ordinal) ||
                name.EndsWith(".tar.bz2", StringComparison.Ordinal))
            {
                return true;
            }

            return SupportedExtensions.Any(extension => name.EndsWith(extension, StringComparison.Ordinal));
        }

        private void ImportFile(string path, bool dryRun, ImportSummaryBuilder summary)
        {
            var fileName = Path.GetFileName(path);
            summary.FilesRead++;
            using var stream = File.OpenRead(path);
            if (path.EndsWith(".tar.bz2", StringComparison.OrdinalIgnoreCase))
            {
                using var bzip = new BZip2InputStream(stream);
                using var reader = new TarReader(bzip);
                ImportTarEntries(reader, fileName, dryRun, summary);
            }
            else if (path.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase))
            {
                using var gzip = new GZipStream(stream, CompressionMode.Decompress);
                using var reader = new TarReader(gzip);
                ImportTarEntries(reader, fileName, dryRun, summary);
            }
            else if (path.EndsWith(".tar", StringComparison.OrdinalIgnoreCase))
            {
                using var reader = new TarReader(stream);
                ImportTarEntries(reader, fileName, dryRun, summary);
            }
            else if (path.EndsWith(".bz2", StringComparison.OrdinalIgnoreCase))
            {
                using var bzip = new BZip2InputStream(stream);
                ImportStream(bzip, fileName, dryRun, summary);
            }
            else if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
            {
                using var gzip = new GZipStream(stream, CompressionMode.Decompress);
                ImportStream(gzip, fileName, dryRun, summary);
            }
            else
            {
                ImportStream(stream, fileName, dryRun, summary);
            }

            if (summary.FilesRead % 100 == 0)
            {
                Log($"[betfair] {summary.FilesRead} file(s) read, {summary.MarketsMatched} race(s) matched");
            }
        }

        private void ImportTarEntries(
            TarReader reader,
            string fileName,
            bool dryRun,
            ImportSummaryBuilder summary)
        {
            TarEntry? entry;
            while ((entry = reader.GetNextEntry()) is not null)
            {
                if (entry.EntryType is not (TarEntryType.RegularFile or TarEntryType.V7RegularFile) ||
                    entry.DataStream is null)
                {
                    continue;
                }

                var entryName = entry.Name ?? "entry";
                var sourceName = $"{fileName}:{entryName}";
                if (entryName.EndsWith(".bz2", StringComparison.OrdinalIgnoreCase))
                {
                    // TarReader owns the entry stream; do not close it while
                    // moving to the next entry.
                    using var bzip = new BZip2InputStream(entry.DataStream)
                    {
                        IsStreamOwner = false
                    };
                    ImportStream(bzip, sourceName, dryRun, summary);
                }
                else if (entryName.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
                {
                    using var gzip = new GZipStream(
                        entry.DataStream,
                        CompressionMode.Decompress,
                        leaveOpen: true);
                    ImportStream(gzip, sourceName, dryRun, summary);
                }
                else
                {
                    ImportStream(entry.DataStream, sourceName, dryRun, summary);
                }
            }
        }

        /// <summary>
        /// Parses newline-delimited Betfair stream messages into completed
        /// market objects. Exposed separately from matching so the parser can
        /// be tested without a database.
        /// </summary>
        internal (IReadOnlyList<BetfairMarketImport> Markets, int LinesRead) ParseMarkets(
            Stream stream,
            string? sourceName = null)
        {
            var accumulators = new Dictionary<string, MarketAccumulator>(StringComparer.OrdinalIgnoreCase);
            using var reader = new StreamReader(
                stream,
                Encoding.UTF8,
                detectEncodingFromByteOrderMarks: true,
                leaveOpen: true);
            string? line;
            var linesRead = 0;
            while ((line = reader.ReadLine()) is not null)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                linesRead++;
                try
                {
                    using var document = JsonDocument.Parse(line);
                    ReadMessage(document.RootElement, accumulators);
                }
                catch (JsonException ex)
                {
                    throw new InvalidDataException(
                        $"Invalid JSON in {sourceName ?? "stream"} at line {linesRead}: {ex.Message}",
                        ex);
                }
            }

            return (
                accumulators.Values.Select(ToImport).ToList(),
                linesRead);
        }

        private void ImportStream(
            Stream stream,
            string sourceName,
            bool dryRun,
            ImportSummaryBuilder summary)
        {
            var parsed = ParseMarkets(stream, sourceName);
            summary.LinesRead += parsed.LinesRead;
            if (parsed.Markets.Count == 0)
            {
                return;
            }

            var batch = new List<BetfairMarketUpdate>();
            foreach (var market in parsed.Markets)
            {
                summary.MarketsRead++;
                var update = MatchMarket(market);
                if (update is null)
                {
                    summary.MarketsSkipped++;
                    continue;
                }

                summary.MarketsMatched++;
                summary.RunnersMatched += update.Runners.Count;
                batch.Add(update);
            }

            if (!dryRun && batch.Count > 0)
            {
                _repository.SaveBatch(batch);
            }
        }

        private BetfairMarketImport ToImport(MarketAccumulator accumulator)
        {
            var market = new BetfairMarketImport
            {
                MarketId = accumulator.MarketId,
                MarketName = accumulator.MarketName,
                Venue = accumulator.Venue,
                CountryCode = accumulator.CountryCode,
                TimeZone = accumulator.TimeZone,
                MarketType = accumulator.MarketType,
                NumberOfWinners = accumulator.NumberOfWinners,
                MarketTimeUtc = accumulator.MarketTimeUtc,
                SuspendTimeUtc = accumulator.SuspendTimeUtc,
                BspMarket = accumulator.BspMarket,
                BspReconciled = accumulator.BspReconciled
            };

            // Anchor at the last observed price at or before T-N, where T is
            // the actual suspend time when the stream provides one and the
            // scheduled market time otherwise.
            DateTime? anchorCutoffUtc = null;
            var offTimeUtc = accumulator.SuspendTimeUtc ?? accumulator.MarketTimeUtc;
            if (offTimeUtc.HasValue)
            {
                anchorCutoffUtc = offTimeUtc.Value.AddMinutes(-_marketAnchorMinutes);
            }

            foreach (var runner in accumulator.Runners.Values)
            {
                var anchor = ResolveMarketAnchor(runner, anchorCutoffUtc);
                market.Runners.Add(new BetfairRunnerImport
                {
                    SelectionId = runner.SelectionId,
                    Name = runner.Name,
                    Status = runner.Status,
                    SortPriority = runner.SortPriority,
                    Bsp = runner.Bsp,
                    LastTradedPrice = runner.LastTradedPrice,
                    MarketAnchorPrice = anchor.Price,
                    MarketAnchorCapturedUtc = anchor.CapturedUtc,
                    PreOffHigh = runner.PreOffHigh,
                    PreOffLow = runner.PreOffLow,
                    TradedVolume = ResolveTradedVolume(runner),
                    ProjectedSp = runner.ProjectedSp,
                    AvailableToBack = ResolveAvailable(runner, isBack: true),
                    AvailableToLay = ResolveAvailable(runner, isBack: false)
                });
            }

            market.TotalMatched = market.Runners
                .Where(runner => runner.TradedVolume.HasValue)
                .Select(runner => runner.TradedVolume!.Value)
                .DefaultIfEmpty()
                .Sum();
            if (market.TotalMatched <= 0m)
            {
                market.TotalMatched = null;
            }

            return market;
        }

        private static void ReadMessage(
            JsonElement root,
            IDictionary<string, MarketAccumulator> markets)
        {
            if (root.ValueKind != JsonValueKind.Object ||
                !root.TryGetProperty("mc", out var changes) ||
                changes.ValueKind != JsonValueKind.Array)
            {
                return;
            }

            long published = TryGetInt64(root, "pt") ?? 0;
            foreach (var change in changes.EnumerateArray())
            {
                var marketId = TryGetString(change, "id");
                if (string.IsNullOrWhiteSpace(marketId))
                {
                    continue;
                }

                if (!markets.TryGetValue(marketId, out var market))
                {
                    market = new MarketAccumulator { MarketId = marketId };
                    markets[marketId] = market;
                }

                market.LastPublishedMs = Math.Max(market.LastPublishedMs, published);
                if (change.TryGetProperty("marketDefinition", out var definition) &&
                    definition.ValueKind == JsonValueKind.Object)
                {
                    ApplyMarketDefinition(market, definition);
                }

                if (change.TryGetProperty("rc", out var runnerChanges) &&
                    runnerChanges.ValueKind == JsonValueKind.Array)
                {
                    ApplyRunnerChanges(market, runnerChanges, published);
                }
            }
        }

        private static void ApplyMarketDefinition(MarketAccumulator market, JsonElement definition)
        {
            market.MarketName = TryGetString(definition, "name") ?? market.MarketName;
            market.Venue = TryGetString(definition, "venue") ?? market.Venue;
            market.CountryCode = TryGetString(definition, "countryCode") ?? market.CountryCode;
            market.TimeZone = TryGetString(definition, "timezone") ?? market.TimeZone;
            market.MarketType = TryGetString(definition, "marketType") ?? market.MarketType;
            market.NumberOfWinners = TryGetInt32(definition, "numberOfWinners") ?? market.NumberOfWinners;
            market.BspMarket = TryGetBoolean(definition, "bspMarket") ?? market.BspMarket;
            market.BspReconciled = TryGetBoolean(definition, "bspReconciled") ?? market.BspReconciled;
            market.InPlay = TryGetBoolean(definition, "inPlay") ?? market.InPlay;

            var marketTime = TryGetDateTime(definition, "marketTime");
            if (marketTime.HasValue)
            {
                market.MarketTimeUtc = marketTime;
            }

            var suspendTime = TryGetDateTime(definition, "suspendTime");
            if (suspendTime.HasValue)
            {
                market.SuspendTimeUtc = suspendTime;
            }

            if (!definition.TryGetProperty("runners", out var runners) ||
                runners.ValueKind != JsonValueKind.Array)
            {
                return;
            }

            foreach (var runnerDefinition in runners.EnumerateArray())
            {
                var selectionId = TryGetInt64(runnerDefinition, "id");
                if (!selectionId.HasValue)
                {
                    continue;
                }

                var runner = market.GetOrCreateRunner(selectionId.Value);
                runner.Name = TryGetString(runnerDefinition, "name") ?? runner.Name;
                runner.Status = TryGetString(runnerDefinition, "status") ?? runner.Status;
                runner.SortPriority = TryGetInt32(runnerDefinition, "sortPriority") ?? runner.SortPriority;
                var bsp = TryGetDecimal(runnerDefinition, "bsp");
                if (bsp.HasValue && bsp.Value > 1m)
                {
                    runner.Bsp = bsp;
                }
            }
        }

        private static void ApplyRunnerChanges(
            MarketAccumulator market,
            JsonElement changes,
            long publishedMs)
        {
            var timestamp = publishedMs > 0
                ? DateTimeOffset.FromUnixTimeMilliseconds(publishedMs).UtcDateTime
                : (DateTime?)null;
            var cutoff = market.SuspendTimeUtc ?? market.MarketTimeUtc;
            bool preOff = !market.InPlay &&
                          (!cutoff.HasValue || !timestamp.HasValue || timestamp.Value <= cutoff.Value);

            foreach (var change in changes.EnumerateArray())
            {
                var selectionId = TryGetInt64(change, "id");
                if (!selectionId.HasValue)
                {
                    continue;
                }

                var runner = market.GetOrCreateRunner(selectionId.Value);
                var status = TryGetString(change, "status");
                if (!string.IsNullOrWhiteSpace(status))
                {
                    runner.Status = status;
                }

                if (preOff)
                {
                    ApplyPreOffChange(runner, change, timestamp);
                }
            }
        }

        private static void ApplyPreOffChange(
            RunnerAccumulator runner,
            JsonElement change,
            DateTime? timestampUtc)
        {
            var ltp = TryGetDecimal(change, "ltp");
            if (ltp.HasValue && ltp.Value > 1m)
            {
                runner.LastTradedPrice = ltp;
                if (timestampUtc.HasValue)
                {
                    // Keep every pre-off LTP point: the suspend time may only
                    // appear later in the stream, which moves the T-N cutoff.
                    runner.PreOffTradedPrices.Add((timestampUtc.Value, ltp.Value));
                }
                runner.PreOffHigh = !runner.PreOffHigh.HasValue || ltp.Value > runner.PreOffHigh.Value
                    ? ltp
                    : runner.PreOffHigh;
                runner.PreOffLow = !runner.PreOffLow.HasValue || ltp.Value < runner.PreOffLow.Value
                    ? ltp
                    : runner.PreOffLow;
            }

            if (change.TryGetProperty("trd", out var traded) &&
                traded.ValueKind == JsonValueKind.Array)
            {
                runner.SawTradedLadder = true;
                ApplyLadder(runner.TradedLadder, traded);
            }

            var tradedDelta = TryGetDecimal(change, "tv");
            if (tradedDelta.HasValue && tradedDelta.Value > 0m)
            {
                runner.TradedDeltaTotal += tradedDelta.Value;
            }

            var projectedSp = TryGetDecimal(change, "spn") ?? TryGetDecimal(change, "spf");
            if (projectedSp.HasValue && projectedSp.Value > 1m)
            {
                runner.ProjectedSp = projectedSp;
            }

            if (change.TryGetProperty("batb", out var bestBack) &&
                bestBack.ValueKind == JsonValueKind.Array)
            {
                ApplyBestAvailable(runner, bestBack, isBack: true);
            }
            if (change.TryGetProperty("batl", out var bestLay) &&
                bestLay.ValueKind == JsonValueKind.Array)
            {
                ApplyBestAvailable(runner, bestLay, isBack: false);
            }

            if (change.TryGetProperty("atb", out var allBack) &&
                allBack.ValueKind == JsonValueKind.Array)
            {
                ApplyLadder(runner.BackLadder, allBack);
            }
            if (change.TryGetProperty("atl", out var allLay) &&
                allLay.ValueKind == JsonValueKind.Array)
            {
                ApplyLadder(runner.LayLadder, allLay);
            }
        }

        private static void ApplyBestAvailable(
            RunnerAccumulator runner,
            JsonElement updates,
            bool isBack)
        {
            foreach (var update in updates.EnumerateArray())
            {
                if (update.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                var values = update.EnumerateArray().ToList();
                if (values.Count < 3)
                {
                    continue;
                }

                var level = TryGetDecimal(values[0]);
                if (!level.HasValue || level.Value != 0m)
                {
                    continue;
                }

                var volume = TryGetDecimal(values[2]);
                if (!volume.HasValue)
                {
                    continue;
                }

                if (isBack)
                {
                    runner.AvailableToBack = volume.Value > 0m ? volume : null;
                }
                else
                {
                    runner.AvailableToLay = volume.Value > 0m ? volume : null;
                }
            }
        }

        private static void ApplyLadder(
            IDictionary<decimal, decimal> ladder,
            JsonElement updates)
        {
            foreach (var update in updates.EnumerateArray())
            {
                if (update.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                var values = update.EnumerateArray().ToList();
                var price = values.Count > 0 ? TryGetDecimal(values[0]) : null;
                var volume = values.Count > 1 ? TryGetDecimal(values[1]) : null;
                if (!price.HasValue || !volume.HasValue)
                {
                    continue;
                }

                if (volume.Value <= 0m)
                {
                    ladder.Remove(price.Value);
                }
                else
                {
                    ladder[price.Value] = volume.Value;
                }
            }
        }

        private BetfairMarketUpdate? MatchMarket(BetfairMarketImport market)
        {
            if (!market.MarketTimeUtc.HasValue ||
                string.IsNullOrWhiteSpace(market.Venue) ||
                market.Runners.Count == 0)
            {
                return null;
            }

            // Event downloads interleave WIN and PLACE markets. Only the WIN
            // market's BSP/volume describes the race outcome we are modelling;
            // letting a place market overwrite the same runners would corrupt
            // the columns.
            if ((!string.IsNullOrWhiteSpace(market.MarketType) &&
                 !market.MarketType.Equals("WIN", StringComparison.OrdinalIgnoreCase)) ||
                (market.NumberOfWinners.HasValue && market.NumberOfWinners.Value != 1))
            {
                return null;
            }

            var local = ToLocalTime(market.MarketTimeUtc.Value, market.TimeZone);
            var course = ResolveCourse(market.Venue);
            if (course is null)
            {
                return null;
            }

            if (!_raceCache.TryGetValue((course.CourseId, local.Date), out var races))
            {
                races = _repository.GetRaces(course.CourseId, local.Date);
                _raceCache[(course.CourseId, local.Date)] = races;
            }

            var race = ResolveRace(races, local.TimeOfDay);
            if (race is null)
            {
                return null;
            }

            if (!_runnerCache.TryGetValue(race.RaceId, out var appRunners))
            {
                appRunners = _repository.GetRunners(race.RaceId);
                _runnerCache[race.RaceId] = appRunners;
            }

            var byName = appRunners
                .Where(row => !string.IsNullOrWhiteSpace(row.HorseName))
                .GroupBy(row => NormalizeName(row.HorseName), StringComparer.Ordinal)
                .ToDictionary(group => group.Key, group => group.ToList(), StringComparer.Ordinal);

            var update = new BetfairMarketUpdate
            {
                RaceId = race.RaceId,
                MarketId = market.MarketId,
                MarketName = market.MarketName,
                BspReconciled = market.BspReconciled
            };

            foreach (var runner in market.Runners)
            {
                if (runner.Status is not null &&
                    runner.Status.Equals("REMOVED", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                var normalized = NormalizeName(runner.Name);
                if (normalized.Length == 0 ||
                    !byName.TryGetValue(normalized, out var candidates))
                {
                    continue;
                }

                var matched = ResolveRunner(candidates, runner);
                if (matched is null)
                {
                    continue;
                }

                update.Runners.Add(new BetfairRunnerUpdate
                {
                    RunnerResultId = matched.RunnerResultId,
                    HorseId = matched.HorseId,
                    SelectionId = runner.SelectionId,
                    Bsp = runner.Bsp,
                    PreOffHigh = runner.PreOffHigh,
                    PreOffLow = runner.PreOffLow,
                    LastTradedPrice = runner.LastTradedPrice,
                    MarketAnchorPrice = runner.MarketAnchorPrice,
                    MarketAnchorCapturedUtc = runner.MarketAnchorCapturedUtc,
                    TradedVolume = runner.TradedVolume,
                    ProjectedSp = runner.ProjectedSp,
                    AvailableToBack = runner.AvailableToBack,
                    AvailableToLay = runner.AvailableToLay
                });
            }

            if (update.Runners.Count == 0)
            {
                return null;
            }

            update.TotalMatched = market.TotalMatched;
            return update;
        }

        private static BetfairRunnerRow? ResolveRunner(
            IReadOnlyList<BetfairRunnerRow> candidates,
            BetfairRunnerImport runner)
        {
            if (candidates.Count == 1)
            {
                return candidates[0];
            }

            if (runner.SortPriority.HasValue)
            {
                var byCloth = candidates.FirstOrDefault(
                    candidate => candidate.SaddleclothNumber.HasValue &&
                                 candidate.SaddleclothNumber.Value == runner.SortPriority.Value);
                if (byCloth is not null)
                {
                    return byCloth;
                }
            }

            return null;
        }

        /// <summary>
        /// Last LTP observed at or before the cutoff. The observations are
        /// retained so a suspend time that arrives after the early prices still
        /// produces the correct anchor.
        /// </summary>
        private static (decimal? Price, DateTime? CapturedUtc) ResolveMarketAnchor(
            RunnerAccumulator runner,
            DateTime? cutoffUtc)
        {
            if (!cutoffUtc.HasValue || runner.PreOffTradedPrices.Count == 0)
            {
                return (null, null);
            }

            decimal? price = null;
            DateTime? captured = null;
            foreach (var observation in runner.PreOffTradedPrices)
            {
                if (observation.TimestampUtc > cutoffUtc.Value)
                {
                    continue;
                }

                if (!captured.HasValue || observation.TimestampUtc >= captured.Value)
                {
                    price = observation.Price;
                    captured = observation.TimestampUtc;
                }
            }

            return (price, captured);
        }

        private static decimal? ResolveTradedVolume(RunnerAccumulator runner)
        {
            if (runner.SawTradedLadder && runner.TradedLadder.Count > 0)
            {
                return runner.TradedLadder.Values.Sum();
            }

            return runner.TradedDeltaTotal > 0m ? runner.TradedDeltaTotal : null;
        }

        private static decimal? ResolveAvailable(RunnerAccumulator runner, bool isBack)
        {
            if (isBack)
            {
                return runner.AvailableToBack ??
                       BestLadderVolume(runner.BackLadder, preferHigherPrice: true);
            }

            return runner.AvailableToLay ??
                   BestLadderVolume(runner.LayLadder, preferHigherPrice: false);
        }

        private static decimal? BestLadderVolume(
            IReadOnlyDictionary<decimal, decimal> ladder,
            bool preferHigherPrice)
        {
            if (ladder.Count == 0)
            {
                return null;
            }

            var best = preferHigherPrice
                ? ladder.Keys.Max()
                : ladder.Keys.Min();
            return ladder.TryGetValue(best, out var volume) && volume > 0m ? volume : null;
        }

        private BetfairCourseRow? ResolveCourse(string venue)
        {
            var normalized = NormalizeName(venue);
            if (normalized.Length == 0)
            {
                return null;
            }

            if (_courseCache.TryGetValue(normalized, out var cached))
            {
                return cached;
            }

            BetfairCourseRow? match = Courses.FirstOrDefault(
                course => NormalizeName(course.Name) == normalized);
            if (match is null)
            {
                match = Courses.FirstOrDefault(course =>
                {
                    var candidate = NormalizeName(course.Name);
                    return candidate.Length >= 4 &&
                           (candidate.Contains(normalized, StringComparison.Ordinal) ||
                            normalized.Contains(candidate, StringComparison.Ordinal));
                });
            }
            if (match is null)
            {
                var venueToken = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault();
                if (venueToken is not null && venueToken.Length >= 4)
                {
                    match = Courses.FirstOrDefault(course =>
                        NormalizeName(course.Name)
                            .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                            .Contains(venueToken, StringComparer.Ordinal));
                }
            }

            _courseCache[normalized] = match;
            return match;
        }

        private static BetfairRaceCandidate? ResolveRace(
            IReadOnlyList<BetfairRaceCandidate> races,
            TimeSpan offTime)
        {
            if (races.Count == 0)
            {
                return null;
            }

            const int toleranceMinutes = 20;
            var best = races
                .Where(race => race.ScheduledOff.HasValue)
                .Select(race => new
                {
                    Race = race,
                    Difference = Math.Abs((race.ScheduledOff!.Value - offTime).TotalMinutes)
                })
                .Where(candidate => candidate.Difference <= toleranceMinutes)
                .OrderBy(candidate => candidate.Difference)
                .FirstOrDefault();

            if (best is not null)
            {
                return best.Race;
            }

            return races.Count == 1 ? races[0] : null;
        }

        private static DateTime ToLocalTime(DateTime utc, string? timeZoneId)
        {
            var timeZone = ResolveTimeZone(timeZoneId);
            return TimeZoneInfo.ConvertTimeFromUtc(
                DateTime.SpecifyKind(utc, DateTimeKind.Utc),
                timeZone);
        }

        private static TimeZoneInfo ResolveTimeZone(string? id)
        {
            if (string.IsNullOrWhiteSpace(id))
            {
                return TimeZoneInfo.Utc;
            }

            try
            {
                return TimeZoneInfo.FindSystemTimeZoneById(id);
            }
            catch (TimeZoneNotFoundException)
            {
            }
            catch (InvalidTimeZoneException)
            {
            }

            if (WindowsTimeZones.TryGetValue(id, out var windowsId))
            {
                try
                {
                    return TimeZoneInfo.FindSystemTimeZoneById(windowsId);
                }
                catch (TimeZoneNotFoundException)
                {
                }
                catch (InvalidTimeZoneException)
                {
                }
            }

            return TimeZoneInfo.Utc;
        }

        internal static string NormalizeName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var text = value.Normalize(NormalizationForm.FormD);
            var builder = new StringBuilder(text.Length);
            foreach (var character in text)
            {
                if (CharUnicodeInfo.GetUnicodeCategory(character) != UnicodeCategory.NonSpacingMark)
                {
                    builder.Append(character);
                }
            }

            text = BracketedTextRegex.Replace(builder.ToString(), " ");
            text = LeadingRunnerNumberRegex.Replace(text, string.Empty);
            text = text.Replace('&', ' ');
            text = text.ToLowerInvariant();
            text = NonAlphaNumericRegex.Replace(text, " ");
            return Regex.Replace(text, @"\s+", " ").Trim();
        }

        private static string? TryGetString(JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object ||
                !element.TryGetProperty(name, out var value))
            {
                return null;
            }

            return value.ValueKind switch
            {
                JsonValueKind.String => value.GetString(),
                JsonValueKind.Number => value.ToString(),
                _ => null
            };
        }

        private static long? TryGetInt64(JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object ||
                !element.TryGetProperty(name, out var value))
            {
                return null;
            }

            if (value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out var number))
            {
                return number;
            }

            if (value.ValueKind == JsonValueKind.String &&
                long.TryParse(value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out number))
            {
                return number;
            }

            return null;
        }

        private static int? TryGetInt32(JsonElement element, string name)
        {
            var value = TryGetInt64(element, name);
            return value.HasValue && value.Value >= int.MinValue && value.Value <= int.MaxValue
                ? (int)value.Value
                : null;
        }

        private static bool? TryGetBoolean(JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object ||
                !element.TryGetProperty(name, out var value))
            {
                return null;
            }

            return value.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null
            };
        }

        private static DateTime? TryGetDateTime(JsonElement element, string name)
        {
            var text = TryGetString(element, name);
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            return DateTimeOffset.TryParse(
                text,
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                out var parsed)
                ? parsed.UtcDateTime
                : null;
        }

        private static decimal? TryGetDecimal(JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object ||
                !element.TryGetProperty(name, out var value))
            {
                return null;
            }

            return TryGetDecimal(value);
        }

        private static decimal? TryGetDecimal(JsonElement value)
        {
            if (value.ValueKind == JsonValueKind.Number)
            {
                if (value.TryGetDecimal(out var number))
                {
                    return number;
                }

                return value.TryGetDouble(out var doubleNumber) && double.IsFinite(doubleNumber)
                    ? (decimal)doubleNumber
                    : null;
            }

            if (value.ValueKind == JsonValueKind.String)
            {
                var text = value.GetString();
                if (string.IsNullOrWhiteSpace(text) ||
                    text.Equals("inf", StringComparison.OrdinalIgnoreCase))
                {
                    return null;
                }

                return decimal.TryParse(
                    text,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out var parsed)
                    ? parsed
                    : null;
            }

            return null;
        }

        private void Log(string message)
        {
            _log?.Invoke(message);
        }

        private sealed class MarketAccumulator
        {
            public string MarketId { get; init; } = string.Empty;
            public string? MarketName { get; set; }
            public string? Venue { get; set; }
            public string? CountryCode { get; set; }
            public string? TimeZone { get; set; }
            public string? MarketType { get; set; }
            public int? NumberOfWinners { get; set; }
            public DateTime? MarketTimeUtc { get; set; }
            public DateTime? SuspendTimeUtc { get; set; }
            public bool BspMarket { get; set; }
            public bool BspReconciled { get; set; }
            public bool InPlay { get; set; }
            public long LastPublishedMs { get; set; }
            public Dictionary<long, RunnerAccumulator> Runners { get; } = new();

            public RunnerAccumulator GetOrCreateRunner(long selectionId)
            {
                if (!Runners.TryGetValue(selectionId, out var runner))
                {
                    runner = new RunnerAccumulator { SelectionId = selectionId };
                    Runners[selectionId] = runner;
                }

                return runner;
            }
        }

        private sealed class RunnerAccumulator
        {
            public long SelectionId { get; init; }
            public string? Name { get; set; }
            public string? Status { get; set; }
            public int? SortPriority { get; set; }
            public decimal? Bsp { get; set; }
            public decimal? LastTradedPrice { get; set; }
            public List<(DateTime TimestampUtc, decimal Price)> PreOffTradedPrices { get; } = new();
            public decimal? PreOffHigh { get; set; }
            public decimal? PreOffLow { get; set; }
            public decimal? ProjectedSp { get; set; }
            public decimal? AvailableToBack { get; set; }
            public decimal? AvailableToLay { get; set; }
            public bool SawTradedLadder { get; set; }
            public decimal TradedDeltaTotal { get; set; }
            public Dictionary<decimal, decimal> TradedLadder { get; } = new();
            public Dictionary<decimal, decimal> BackLadder { get; } = new();
            public Dictionary<decimal, decimal> LayLadder { get; } = new();
        }

        private sealed class ImportSummaryBuilder
        {
            public int FilesRead;
            public int LinesRead;
            public int MarketsRead;
            public int MarketsMatched;
            public int RunnersMatched;
            public int MarketsSkipped;
            public int Failures;

            public BetfairImportSummary Build()
                => new(
                    FilesRead,
                    MarketsRead,
                    MarketsMatched,
                    RunnersMatched,
                    MarketsSkipped,
                    LinesRead);
        }
    }
}
