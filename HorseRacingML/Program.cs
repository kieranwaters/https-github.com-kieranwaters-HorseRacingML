using System.Text.Json;
using HorseRacingML.Data;
using HorseRacingML.Jobs;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;

var builder = WebApplication.CreateBuilder(args);

var connectionString = builder.Configuration.GetConnectionString("HorseRacingDb")
    ?? throw new InvalidOperationException("Connection string 'HorseRacingDb' is not configured.");

// Controlled benchmark of the best saved hyperparameter combination. Run it
// twice to isolate the new features:
//   --benchmark-best --disable-pace-features
//   --benchmark-best
if (args.Any(arg => string.Equals(arg, "--benchmark-best", StringComparison.OrdinalIgnoreCase)))
{
    if (args.Any(arg => string.Equals(arg, "--disable-pace-features", StringComparison.OrdinalIgnoreCase)))
    {
        builder.Configuration["Training:EnablePaceFeatures"] = "false";
    }

    static string? ReadOption(string[] arguments, string name)
    {
        var index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var featureBlocksValue = ReadOption(args, "--feature-blocks");
    if (!string.IsNullOrWhiteSpace(featureBlocksValue))
    {
        builder.Configuration["Training:FeatureBlocks"] = featureBlocksValue;
    }

    var minimumRaceDateValue = ReadOption(args, "--minimum-race-date");
    if (!string.IsNullOrWhiteSpace(minimumRaceDateValue))
    {
        builder.Configuration["Training:MinimumRaceDate"] = minimumRaceDateValue;
    }

    var requirePreOffPriceValue = ReadOption(args, "--require-pre-off-price");
    if (!string.IsNullOrWhiteSpace(requirePreOffPriceValue))
    {
        builder.Configuration["Training:RequirePreOffPrice"] = requirePreOffPriceValue;
    }

    var keepUnpricedRunnersValue = ReadOption(args, "--keep-unpriced-runners");
    if (!string.IsNullOrWhiteSpace(keepUnpricedRunnersValue))
    {
        builder.Configuration["Training:KeepUnpricedRunners"] = keepUnpricedRunnersValue;
    }

    var benchmarkRepository = new RacingRepository(connectionString);
    var reference = benchmarkRepository.GetBestMLParameter()
        ?? throw new InvalidOperationException("No eligible saved ML parameter was found.");
    var benchmarkTrainer = new HyperparameterTrainer(builder.Configuration);
    var report = PaceFeatureBenchmark.Run(
        builder.Configuration,
        benchmarkRepository,
        benchmarkTrainer,
        reference,
        Console.WriteLine);

    var outputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var outputPath = outputIndex >= 0 && outputIndex + 1 < args.Length
        ? args[outputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"feature-benchmark-{FeatureBlockNames.Format(benchmarkTrainer.EnabledFeatureBlocks).Replace(',', '-')}-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        outputPath,
        JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[benchmark] report written to {outputPath}");
    return;
}

// Trains the best model-only hyperparameter combination with the configured
// feature blocks, persists the artifact, and records a candidate MLParameters
// row. Activation is opt-in so ROI/CLV can be checked first:
//   --train-best [--feature-blocks Pace,MarketMovement,PriceData] [--activate]
if (args.Any(arg => string.Equals(arg, "--train-best", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadTrainOption(string[] arguments, string name)
    {
        var index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var trainFeatureBlocks = ReadTrainOption(args, "--feature-blocks");
    if (!string.IsNullOrWhiteSpace(trainFeatureBlocks))
    {
        builder.Configuration["Training:FeatureBlocks"] = trainFeatureBlocks;
    }

    var trainMinimumRaceDate = ReadTrainOption(args, "--minimum-race-date");
    if (!string.IsNullOrWhiteSpace(trainMinimumRaceDate))
    {
        builder.Configuration["Training:MinimumRaceDate"] = trainMinimumRaceDate;
    }

    var trainRequirePrice = ReadTrainOption(args, "--require-pre-off-price");
    if (!string.IsNullOrWhiteSpace(trainRequirePrice))
    {
        builder.Configuration["Training:RequirePreOffPrice"] = trainRequirePrice;
    }

    var trainKeepUnpriced = ReadTrainOption(args, "--keep-unpriced-runners");
    if (!string.IsNullOrWhiteSpace(trainKeepUnpriced))
    {
        builder.Configuration["Training:KeepUnpricedRunners"] = trainKeepUnpriced;
    }

    bool activateCandidate = args.Any(
        arg => string.Equals(arg, "--activate", StringComparison.OrdinalIgnoreCase));

    var candidateRepository = new RacingRepository(connectionString);
    var candidateReference = candidateRepository.GetBestMLParameter()
        ?? throw new InvalidOperationException("No eligible saved ML parameter was found.");
    var candidateTrainer = new HyperparameterTrainer(builder.Configuration);
    var candidateReport = PaceFeatureBenchmark.Run(
        builder.Configuration,
        candidateRepository,
        candidateTrainer,
        candidateReference,
        Console.WriteLine,
        persistWeights: true,
        activateModel: activateCandidate);

    var candidate = PaceFeatureBenchmark.Clone(
        candidateReference,
        candidateReference.Seed ?? 42);
    candidate.RunDate = DateTime.UtcNow;
    candidate.EarliestTrainingDate = builder.Configuration.GetValue<DateTime?>("Training:MinimumRaceDate")
        ?? new DateTime(2017, 1, 1);
    candidate.TrainingStartDate = candidate.EarliestTrainingDate;
    candidate.Folds = 1;
    candidate.Fold = null;
    candidate.Objective = candidateReference.Objective ?? "RaceSoftmax";
    candidate.MarketOffsetMode = candidateReference.MarketOffsetMode
        ?? HyperparameterTrainer.NormalizeOffsetMode(
            builder.Configuration.GetValue<string>("Training:MarketOffsetMode") ?? "LogProb");
    candidate.MarketBlind = candidateReference.MarketBlind
        ?? builder.Configuration.GetValue<bool?>("Training:MarketBlind")
        ?? false;
    candidate.RaceBatchSize = candidateReference.RaceBatchSize
        ?? Math.Max(1, builder.Configuration.GetValue<int?>("Training:RaceBatchSize") ?? 64);
    candidate.WinRateAlpha = candidateReference.WinRateAlpha
        ?? builder.Configuration.GetValue<double?>("WinRateAlpha")
        ?? 1d;
    candidate.WinRateBeta = candidateReference.WinRateBeta
        ?? builder.Configuration.GetValue<double?>("WinRateBeta")
        ?? 9d;
    candidate.NormalizationMode = "PerRaceSoftmax";
    candidate.FeatureBlocks = FeatureBlockNames.Format(candidateTrainer.EnabledFeatureBlocks);
    candidate.Epochs = candidateReport.BestEpoch.HasValue
        ? Math.Max(1, candidateReport.BestEpoch.Value + 1)
        : candidateReference.Epochs;
    candidate.BestEpoch = candidateReport.BestEpoch;
    candidate.BlendModelWeight = 1d;
    candidate.BlendMarketWeight = 0d;
    candidate.DevBrier = candidateReport.DevModelBrier;
    candidate.ValidationBrier = candidateReport.DevModelBrier;
    candidate.DevLoss = candidateReport.DevModelLogLoss;
    candidate.ValidationLoss = candidateReport.DevModelLogLoss;
    candidate.DevTemperature = candidateReport.Temperature;
    candidate.DevMarketBrier = candidateReport.DevMarketBrier;
    candidate.DevMarketLogLoss = candidateReport.DevMarketLogLoss;
    candidate.DevBlendBrier = candidateReport.DevBlendBrier;
    candidate.DevBlendLogLoss = candidateReport.DevBlendLogLoss;
    candidate.TestBrier = candidateReport.TestModelBrier;
    candidate.TestLoss = candidateReport.TestModelLogLoss;
    candidate.TestTemperature = candidateReport.Temperature;
    candidate.TestMarketBrier = candidateReport.TestMarketBrier;
    candidate.TestMarketLogLoss = candidateReport.TestMarketLogLoss;
    candidate.TestBlendBrier = candidateReport.TestBlendBrier;
    candidate.TestBlendLogLoss = candidateReport.TestBlendLogLoss;
    candidate.TrainingRaceCount = candidateReport.TrainRaces + candidateReport.DevRaces;
    candidate.DevRaceCount = candidateReport.DevRaces;
    candidate.TestRaceCount = candidateReport.TestRaces;
    candidate.TrainingRowCount = candidateReport.TrainRows + candidateReport.DevRows;
    candidate.DevRowCount = candidateReport.DevRows;
    candidate.TestRowCount = candidateReport.TestRows;
    candidate.SelectedOn = "Test";
    candidate.ModelArtifact = candidateReport.ModelArtifactFileName;
    candidate.ModelHash = candidateReport.ModelHash;
    candidate.FeatureSchemaHash = candidateReport.FeatureSchemaHash;
    candidate.IsActive = activateCandidate;

    var candidateId = candidateRepository.InsertMLParameter(candidate);
    if (activateCandidate)
    {
        candidateRepository.ActivateMLParameter(
            candidateId,
            candidate.ModelArtifact,
            candidate.ModelHash,
            candidate.FeatureSchemaHash);
    }

    Console.WriteLine(
        $"[train-best] candidate Id {candidateId} {(activateCandidate ? "activated" : "saved but not activated")}: " +
        $"dev Brier {candidateReport.DevModelBrier:F6}, test Brier {candidateReport.TestModelBrier:F6}, " +
        $"artifact {candidateReport.ModelArtifactFileName ?? "n/a"}");
    return;
}

// One controlled ablation run per feature block. The history is loaded
// once with every block enabled; each row trains on the same split with only
// that block's columns added to the baseline feature set.
//   --benchmark-blocks --minimum-race-date 2017-01-01 --require-pre-off-price true --keep-unpriced-runners false
if (args.Any(arg => string.Equals(arg, "--benchmark-blocks", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadBlockOption(string[] arguments, string name)
    {
        var index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    builder.Configuration["Training:FeatureBlocks"] = "all";

    var minimumRaceDateValue = ReadBlockOption(args, "--minimum-race-date");
    if (!string.IsNullOrWhiteSpace(minimumRaceDateValue))
    {
        builder.Configuration["Training:MinimumRaceDate"] = minimumRaceDateValue;
    }

    var requirePreOffPriceValue = ReadBlockOption(args, "--require-pre-off-price");
    if (!string.IsNullOrWhiteSpace(requirePreOffPriceValue))
    {
        builder.Configuration["Training:RequirePreOffPrice"] = requirePreOffPriceValue;
    }

    var keepUnpricedRunnersValue = ReadBlockOption(args, "--keep-unpriced-runners");
    if (!string.IsNullOrWhiteSpace(keepUnpricedRunnersValue))
    {
        builder.Configuration["Training:KeepUnpricedRunners"] = keepUnpricedRunnersValue;
    }

    var blocksValue = ReadBlockOption(args, "--blocks");
    var onlyBlocks = string.IsNullOrWhiteSpace(blocksValue)
        ? null
        : blocksValue
            .Split(new[] { ',', ';', '|', ' ' }, StringSplitOptions.RemoveEmptyEntries)
            .ToList();

    var blockRepository = new RacingRepository(connectionString);
    var blockReference = blockRepository.GetBestMLParameter()
        ?? throw new InvalidOperationException("No eligible saved ML parameter was found.");
    var blockTrainer = new HyperparameterTrainer(builder.Configuration);
    var blockReport = FeatureBlockBenchmark.Run(
        builder.Configuration,
        blockRepository,
        blockTrainer,
        blockReference,
        Console.WriteLine,
        onlyBlocks);

    var blockOutputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var blockOutputPath = blockOutputIndex >= 0 && blockOutputIndex + 1 < args.Length
        ? args[blockOutputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"feature-block-benchmark-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        blockOutputPath,
        JsonSerializer.Serialize(blockReport, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[blocks] report written to {blockOutputPath}");
    return;
}

// Targeted hyperparameter sweep around the best saved parameter. The encoded
// dataset is built once; every candidate is selected on dev and only the
// winner is evaluated on test.
//   --tune-pace --disable-pace-features --max-candidates 12
//   --tune-pace --max-candidates 18
if (args.Any(arg => string.Equals(arg, "--tune-pace", StringComparison.OrdinalIgnoreCase)))
{
    if (args.Any(arg => string.Equals(arg, "--disable-pace-features", StringComparison.OrdinalIgnoreCase)))
    {
        builder.Configuration["Training:EnablePaceFeatures"] = "false";
    }

    int? maxCandidates = null;
    var maxCandidatesIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--max-candidates", StringComparison.OrdinalIgnoreCase));
    if (maxCandidatesIndex >= 0 &&
        maxCandidatesIndex + 1 < args.Length &&
        int.TryParse(args[maxCandidatesIndex + 1], out var parsedMaxCandidates))
    {
        maxCandidates = parsedMaxCandidates;
    }

    int candidateOffset = 0;
    var candidateOffsetIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--candidate-offset", StringComparison.OrdinalIgnoreCase));
    if (candidateOffsetIndex >= 0 &&
        candidateOffsetIndex + 1 < args.Length &&
        int.TryParse(args[candidateOffsetIndex + 1], out var parsedOffset))
    {
        candidateOffset = Math.Max(0, parsedOffset);
    }

    var sweepRepository = new RacingRepository(connectionString);
    var reference = sweepRepository.GetBestMLParameter()
        ?? throw new InvalidOperationException("No eligible saved ML parameter was found.");
    var sweepTrainer = new HyperparameterTrainer(builder.Configuration);
    var sweepReport = PaceHyperparameterSweep.Run(
        builder.Configuration,
        sweepRepository,
        sweepTrainer,
        reference,
        maxCandidates,
        candidateOffset,
        Console.WriteLine);

    var outputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var outputPath = outputIndex >= 0 && outputIndex + 1 < args.Length
        ? args[outputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"pace-sweep-{(sweepTrainer.EnablePaceFeatures ? "on" : "off")}-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        outputPath,
        JsonSerializer.Serialize(sweepReport, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[sweep] report written to {outputPath}");
    return;
}

// Operational entry point: dotnet run --project HorseRacingML -- --backfill-run-styles [--max-rows N]
// The web app and the full schema bootstrap are not needed in this mode;
// BackfillRunStyles creates only the narrow RunnerRunStyle table it needs.
if (args.Any(arg => string.Equals(arg, "--backfill-run-styles", StringComparison.OrdinalIgnoreCase)))
{
    int? maxRows = null;
    var maxRowsIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--max-rows", StringComparison.OrdinalIgnoreCase));
    if (maxRowsIndex >= 0 &&
        maxRowsIndex + 1 < args.Length &&
        int.TryParse(args[maxRowsIndex + 1], out var parsedMaxRows))
    {
        maxRows = parsedMaxRows;
    }

    var repository = new RacingRepository(connectionString);
    var classified = repository.BackfillRunStyles(maxRows: maxRows, log: Console.WriteLine);
    Console.WriteLine($"[run-style] classified {classified} rows.");
    return;
}

// Create the local database and its schema on first run so the app does not
// depend on a database that was set up by hand outside of source control.
DatabaseBootstrapper.EnsureDatabase(connectionString);

// Closing-line value report: bet price versus the last recorded pre-off
// RunnerFlow price. Needs the Betfair/Sporting Life snapshot history.
if (args.Any(arg => string.Equals(arg, "--clv-report", StringComparison.OrdinalIgnoreCase)))
{
    var clvRepository = new RacingRepository(connectionString);
    var clvRows = clvRepository.GetClvRows();
    if (clvRows.Count == 0)
    {
        Console.WriteLine("No placed bets with RunnerFlow snapshots yet.");
        return;
    }

    foreach (var row in clvRows)
    {
        Console.WriteLine(
            $"{row.PlacedUtc:yyyy-MM-dd HH:mm} {row.HorseName}: bet {row.BetOdds:F2}, " +
            $"close {row.ClosingOdds:F2}, CLV {(row.Clv.HasValue ? row.Clv.Value.ToString("P2") : "n/a")}");
    }

    var clvValues = clvRows.Where(row => row.Clv.HasValue).Select(row => row.Clv!.Value).ToList();
    if (clvValues.Count > 0)
    {
        Console.WriteLine($"Average CLV: {clvValues.Average():P2} over {clvValues.Count} bets.");
    }
    return;
}

// Resumable price-data backfill: official rating, headgear, medication and
// the opening/touched price range from the Sporting Life result JSON.
// Existing result rows are updated in place; results and rolling state are
// not duplicated.
//   --backfill-price-data --from 2024-01-01 --to 2026-11-18 --parallelism 8
if (args.Any(arg => string.Equals(arg, "--backfill-price-data", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadPriceOption(string[] arguments, string name)
    {
        var index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var fromValue = ReadPriceOption(args, "--from");
    var toValue = ReadPriceOption(args, "--to");
    var parallelismValue = ReadPriceOption(args, "--parallelism");
    var fromDate = DateTime.TryParse(fromValue, out var parsedFrom)
        ? parsedFrom.Date
        : new DateTime(2024, 1, 1);
    var toDate = DateTime.TryParse(toValue, out var parsedTo)
        ? parsedTo.Date
        : DateTime.Today;
    int parallelism = int.TryParse(parallelismValue, out var parsedParallelism)
        ? Math.Clamp(parsedParallelism, 1, 32)
        : 8;

    var priceRepository = new RacingRepository(connectionString);
    var priceStatus = new ScrapingStatusService();
    var priceScraper = new RaceResultsJsonScraper(priceRepository, priceStatus);
    priceScraper.BackfillRange(
        fromDate,
        toDate,
        parallelism,
        force: true,
        priceDataOnly: true);
    return;
}

// Market-movement-only backfill: re-read the historical odds series from the
// Sporting Life result JSON and fill OpeningFraction/TouchedHighFraction/
// TouchedLowFraction without touching the result or price-data columns.
//   --backfill-market-movement --from 2017-01-01 --to 2026-11-18 --parallelism 8
if (args.Any(arg => string.Equals(arg, "--backfill-market-movement", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadMovementOption(string[] arguments, string name)
    {
        var index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var fromValue = ReadMovementOption(args, "--from");
    var toValue = ReadMovementOption(args, "--to");
    var parallelismValue = ReadMovementOption(args, "--parallelism");
    var fromDate = DateTime.TryParse(fromValue, out var parsedFrom)
        ? parsedFrom.Date
        : new DateTime(2017, 1, 1);
    var toDate = DateTime.TryParse(toValue, out var parsedTo)
        ? parsedTo.Date
        : DateTime.Today;
    int parallelism = int.TryParse(parallelismValue, out var parsedParallelism)
        ? Math.Clamp(parsedParallelism, 1, 32)
        : 8;

    var movementRepository = new RacingRepository(connectionString);
    var movementStatus = new ScrapingStatusService();
    var movementScraper = new RaceResultsJsonScraper(movementRepository, movementStatus);
    movementScraper.BackfillRange(
        fromDate,
        toDate,
        parallelism,
        force: true,
        marketMovementOnly: true);
    return;
}

// Import Betfair Historical Data files (BASIC/ADVANCED/PRO). BASIC is free and
// supplies BSP plus one-minute last-traded prices; ADVANCED/PRO also supply
// traded volume and the available-to-back/lay ladder. The command reads
// .tar.bz2/.bz2/.tar.gz archives and extracted .json files directly:
//   --import-betfair-historical "C:\betfair\2024-01" [--dry-run]
if (args.Any(arg => string.Equals(arg, "--import-betfair-historical", StringComparison.OrdinalIgnoreCase)))
{
    var pathIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--import-betfair-historical", StringComparison.OrdinalIgnoreCase));
    if (pathIndex < 0 || pathIndex + 1 >= args.Length)
    {
        Console.Error.WriteLine("Usage: --import-betfair-historical <file-or-directory> [--dry-run]");
        Environment.ExitCode = 2;
        return;
    }

    var importPath = args[pathIndex + 1];
    var dryRun = args.Any(arg => string.Equals(arg, "--dry-run", StringComparison.OrdinalIgnoreCase));
    var betfairRepository = new BetfairHistoricalRepository(connectionString);
    var importer = new BetfairHistoricalDataImporter(betfairRepository, Console.WriteLine);
    var importSummary = importer.ImportPath(importPath, dryRun);
    Console.WriteLine(
        $"[betfair] {(dryRun ? "dry run" : "import")}: " +
        $"{importSummary.MarketsMatched}/{importSummary.MarketsRead} markets matched, " +
        $"{importSummary.RunnersMatched} runners, {importSummary.MarketsSkipped} skipped.");
    return;
}

// Racing TV sectionals and breeding coverage:
//   dotnet run --project HorseRacingML -- --racingtv-status
if (args.Any(arg => string.Equals(arg, "--racingtv-status", StringComparison.OrdinalIgnoreCase)))
{
    var racingTvStatus = new RacingTvStatusService(builder.Configuration);
    racingTvStatus.Print(Console.WriteLine);
    return;
}

// Racing TV backfills. Sectionals are free from March 2023 (GB) and January
// 2024 (IRE). The breeding pass reads sire/dam/damsire from the public
// racecards and is useful from the start of the racecard archive (2010+).
//   --backfill-sectionals [--from 2023-03-01] [--to 2026-12-31] [--max-days 30] [--force]
//   --backfill-breeding   [--from 2010-01-01] [--to 2026-12-31] [--max-days 30] [--force]
if (args.Any(arg => string.Equals(arg, "--backfill-sectionals", StringComparison.OrdinalIgnoreCase)) ||
    args.Any(arg => string.Equals(arg, "--backfill-breeding", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadRacingTvOption(string[] arguments, string name)
    {
        var index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    bool includeSectionals = args.Any(
        arg => string.Equals(arg, "--backfill-sectionals", StringComparison.OrdinalIgnoreCase));
    bool includeBreeding = args.Any(
        arg => string.Equals(arg, "--backfill-breeding", StringComparison.OrdinalIgnoreCase));
    DateTime defaultFrom = includeSectionals
        ? builder.Configuration.GetValue<DateTime?>("RacingTv:SectionalBackfillFrom") ?? new DateTime(2023, 3, 1)
        : builder.Configuration.GetValue<DateTime?>("RacingTv:BreedingBackfillFrom") ?? new DateTime(2010, 1, 1);

    var fromValue = ReadRacingTvOption(args, "--from");
    var toValue = ReadRacingTvOption(args, "--to");
    var maxDaysValue = ReadRacingTvOption(args, "--max-days");
    bool force = args.Any(arg => string.Equals(arg, "--force", StringComparison.OrdinalIgnoreCase));
    DateTime fromDate = DateTime.TryParse(fromValue, out var parsedRacingTvFrom)
        ? parsedRacingTvFrom.Date
        : defaultFrom.Date;
    DateTime toDate = DateTime.TryParse(toValue, out var parsedRacingTvTo)
        ? parsedRacingTvTo.Date
        : DateTime.Today;
    int? maxDays = int.TryParse(maxDaysValue, out var parsedMaxDays) && parsedMaxDays > 0
        ? parsedMaxDays
        : (int?)null;

    using var racingTvClient = new RacingTvClient(builder.Configuration);
    var racingTvRepository = new RacingTvRepository(connectionString);
    var racingTvBackfill = new RacingTvBackfillService(
        racingTvClient,
        racingTvRepository,
        builder.Configuration);
    var racingTvSummary = racingTvBackfill.BackfillAsync(
        fromDate,
        toDate,
        includeSectionals,
        includeBreeding,
        force,
        maxDays,
        Console.WriteLine).GetAwaiter().GetResult();

    Console.WriteLine(
        $"[racingtv] {(includeSectionals ? "sectionals" : "breeding")}: " +
        $"{racingTvSummary.SectionalRowsUpserted} sectional rows, " +
        $"{racingTvSummary.PedigreeRowsUpserted} pedigree rows.");
    return;
}

// Weather coverage report:
//   dotnet run --project HorseRacingML -- --weather-status
if (args.Any(arg => string.Equals(arg, "--weather-status", StringComparison.OrdinalIgnoreCase)))
{
    var statusService = new WeatherStatusService(builder.Configuration);
    statusService.Print(Console.WriteLine);
    return;
}

// One-off weather work:
//   --backfill-weather           resumable: fetch only course-years that still need race rows
//   --refresh-weather            re-fetch the selected range even when race rows already exist
//   --populate-weather-features  recompute race features from CourseWeatherDaily only
// All accept [--from yyyy-MM-dd] [--to yyyy-MM-dd] [--max-courses N].
// Course coordinates are applied automatically before the weather is fetched.
static (DateTime? From, DateTime? To, int? MaxCourses) ParseWeatherRangeArgs(string[] arguments)
{
    DateTime? fromDate = null;
    DateTime? toDate = null;
    int? maxCourses = null;

    var fromIndex = Array.FindIndex(arguments, arg => string.Equals(arg, "--from", StringComparison.OrdinalIgnoreCase));
    if (fromIndex >= 0 && fromIndex + 1 < arguments.Length &&
        DateTime.TryParse(arguments[fromIndex + 1], out var parsedFrom))
    {
        fromDate = parsedFrom.Date;
    }

    var toIndex = Array.FindIndex(arguments, arg => string.Equals(arg, "--to", StringComparison.OrdinalIgnoreCase));
    if (toIndex >= 0 && toIndex + 1 < arguments.Length &&
        DateTime.TryParse(arguments[toIndex + 1], out var parsedTo))
    {
        toDate = parsedTo.Date;
    }

    var maxCoursesIndex = Array.FindIndex(arguments, arg => string.Equals(arg, "--max-courses", StringComparison.OrdinalIgnoreCase));
    if (maxCoursesIndex >= 0 && maxCoursesIndex + 1 < arguments.Length &&
        int.TryParse(arguments[maxCoursesIndex + 1], out var parsedMaxCourses))
    {
        maxCourses = Math.Max(0, parsedMaxCourses);
    }

    return (fromDate, toDate, maxCourses);
}

var weatherMode =
    args.Any(arg => string.Equals(arg, "--populate-weather-features", StringComparison.OrdinalIgnoreCase))
        ? WeatherBackfillMode.CacheOnly
        : args.Any(arg => string.Equals(arg, "--refresh-weather", StringComparison.OrdinalIgnoreCase))
            ? WeatherBackfillMode.Refresh
            : args.Any(arg => string.Equals(arg, "--backfill-weather", StringComparison.OrdinalIgnoreCase))
                ? WeatherBackfillMode.Normal
                : (WeatherBackfillMode?)null;

if (weatherMode.HasValue)
{
    var (fromDate, toDate, maxCourses) = ParseWeatherRangeArgs(args);
    var defaultFrom = builder.Configuration.GetValue<DateTime?>("Weather:BackfillFrom") ?? new DateTime(2010, 1, 1);
    var weatherRepository = new WeatherRepository(connectionString);
    using var weatherClient = new OpenMeteoWeatherClient(builder.Configuration);
    var weatherService = new WeatherBackfillService(weatherRepository, weatherClient, builder.Configuration);
    weatherService.Backfill(
        fromDate ?? defaultFrom,
        toDate ?? DateTime.Today,
        maxCourses,
        Console.WriteLine,
        mode: weatherMode.Value);
    return;
}

// Add services to the container.
builder.Services.AddControllersWithViews();

// Register the repository so parsed data can be inserted into the database
builder.Services.AddScoped(_ => new RacingRepository(connectionString));

// TensorFlow trainer for running models on the GPU
builder.Services.AddSingleton<HyperparameterTrainer>();
builder.Services.AddSingleton<BetfairNavigationService>();
builder.Services.AddTransient<RaceResultsScraper>();
builder.Services.AddSingleton<ScrapingStatusService>();
builder.Services.AddHostedService<MarketSnapshotScheduler>();

// Weather ingestion. The client and repository are singletons because the
// backfill is long-running and the daily rows are cached in SQL Server.
builder.Services.AddSingleton(_ => new WeatherRepository(connectionString));
builder.Services.AddSingleton(_ => new OpenMeteoWeatherClient(builder.Configuration));
builder.Services.AddSingleton<WeatherBackfillService>();
builder.Services.AddSingleton<WeatherStatusService>();

// Racing TV sectionals and breeding. The client serialises and rate-limits
// requests, so a single singleton can safely serve both backfills.
builder.Services.AddSingleton(_ => new RacingTvRepository(connectionString));
builder.Services.AddSingleton(_ => new RacingTvClient(builder.Configuration));
builder.Services.AddSingleton<RacingTvBackfillService>();
builder.Services.AddSingleton<RacingTvStatusService>();

// Background job pipeline. Jobs run outside the HTTP request path and are
// protected against duplicate concurrent submissions by kind.
builder.Services.AddSingleton<BackgroundJobStore>();
builder.Services.AddSingleton<BackgroundJobQueue>();
builder.Services.AddScoped<IBackgroundJobHandler, ScrapeRaceResultsJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, BackfillRunStylesJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, BackfillWeatherJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, BackfillSectionalsJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, BackfillBreedingJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, TrainAIJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, CustomTrainingJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, AutomateBetsJobHandler>();
builder.Services.AddHostedService<BackgroundJobWorker>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();