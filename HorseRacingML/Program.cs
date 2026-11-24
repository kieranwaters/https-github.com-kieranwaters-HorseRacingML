using System.Text.Json;
using HorseRacingML.Controllers;
using HorseRacingML.Data;
using HorseRacingML.Jobs;
using HorseRacingML.ML;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

var builder = WebApplication.CreateBuilder(args);

// Keep a small diagnostic log outside the repo. ASP.NET logs normally go to
// the console, which is lost when a long-running training request kills the
// process or the terminal is closed.
static void AppendDiagnosticLog(string message)
{
    try
    {
        var directory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "HorseRacingML",
            "logs");
        Directory.CreateDirectory(directory);
        File.AppendAllText(
            Path.Combine(directory, $"app-{DateTime.UtcNow:yyyyMMdd}.log"),
            $"{DateTime.UtcNow:O} {message}{Environment.NewLine}");
    }
    catch
    {
        // Diagnostics must never take the app down.
    }
}

AppDomain.CurrentDomain.UnhandledException += (_, eventArgs) =>
    AppendDiagnosticLog($"Unhandled exception: {eventArgs.ExceptionObject}");
TaskScheduler.UnobservedTaskException += (_, eventArgs) =>
{
    AppendDiagnosticLog($"Unobserved task exception: {eventArgs.Exception}");
    eventArgs.SetObserved();
};
AppDomain.CurrentDomain.ProcessExit += (_, _) =>
    AppendDiagnosticLog("Process exiting.");

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

// Optional LightGBM LambdaRank comparison on the same chronological split.
// It is a benchmark only; it does not persist or activate an artifact.
//   --benchmark-lightgbm --minimum-race-date 2017-01-01
if (args.Any(arg => string.Equals(arg, "--benchmark-lightgbm", StringComparison.OrdinalIgnoreCase)))
{
    var lgbmMinimumDateIndex = Array.FindIndex(
        args,
        argument => string.Equals(argument, "--minimum-race-date", StringComparison.OrdinalIgnoreCase));
    if (lgbmMinimumDateIndex >= 0 && lgbmMinimumDateIndex + 1 < args.Length)
    {
        builder.Configuration["Training:MinimumRaceDate"] = args[lgbmMinimumDateIndex + 1];
    }

    var lgbmMaxRacesIndex = Array.FindIndex(
        args,
        argument => string.Equals(argument, "--max-races", StringComparison.OrdinalIgnoreCase));
    if (lgbmMaxRacesIndex >= 0 && lgbmMaxRacesIndex + 1 < args.Length)
    {
        builder.Configuration["Training:BenchmarkMaxRaces"] = args[lgbmMaxRacesIndex + 1];
    }

    var lgbmRepository = new RacingRepository(connectionString);
    var lgbmReference = lgbmRepository.GetBestMLParameter()
        ?? new MLParameter
        {
            Seed = 42,
            Units = 96,
            Layers = 2,
            LearningRate = 0.0007,
            Epochs = 10,
            BatchSize = 500,
            Folds = 1
        };
    var lgbmTrainer = new HyperparameterTrainer(builder.Configuration);
    var lgbmReport = LightGbmRankingBenchmark.Run(
        builder.Configuration,
        lgbmRepository,
        lgbmTrainer,
        lgbmReference,
        Console.WriteLine);

    var lgbmOutputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var lgbmOutputPath = lgbmOutputIndex >= 0 && lgbmOutputIndex + 1 < args.Length
        ? args[lgbmOutputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"lightgbm-benchmark-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        lgbmOutputPath,
        JsonSerializer.Serialize(lgbmReport, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[lightgbm] report written to {lgbmOutputPath}");
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
    var importer = new BetfairHistoricalDataImporter(
        betfairRepository,
        Console.WriteLine,
        builder.Configuration.GetValue<int?>("Training:MarketAnchorMinutes") ?? 5);
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
// racecards and is useful from the start of the racecard archive (2009+).
//   --backfill-sectionals [--from 2023-03-01] [--to 2026-12-31] [--max-days 30] [--force]
//   --backfill-breeding   [--from 2009-01-01] [--to 2026-12-31] [--max-days 30] [--force]
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
        : builder.Configuration.GetValue<DateTime?>("RacingTv:BreedingBackfillFrom") ?? new DateTime(2009, 1, 1);

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

// At The Races sectional gap fill. ATR carries the same TPD timing data as
// the free RaceiQ archive but covers some British meetings that RaceiQ does
// not. It needs an interactive Chrome session because of the site's
// JavaScript client challenge, so this is a foreground backfill:
//   dotnet run --project HorseRacingML -- --backfill-atr-sectionals
//       [--from 2023-03-01] [--to yyyy-MM-dd] [--max-days 30] [--force]
if (args.Any(arg => string.Equals(arg, "--backfill-atr-sectionals", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadAtTheRacesOption(string[] arguments, string name)
    {
        var index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var atrRepository = new RacingTvRepository(connectionString);
    using var atrClient = new AtTheRacesClient(builder.Configuration);
    var atrService = new AtTheRacesSectionalBackfillService(
        atrClient,
        atrRepository,
        builder.Configuration);
    DateTime atrFrom = DateTime.TryParse(
        ReadAtTheRacesOption(args, "--from"),
        out var parsedAtrFrom)
        ? parsedAtrFrom.Date
        : atrService.DefaultFrom;
    DateTime atrTo = DateTime.TryParse(
        ReadAtTheRacesOption(args, "--to"),
        out var parsedAtrTo)
        ? parsedAtrTo.Date
        : DateTime.Today;
    int? atrMaxDays = int.TryParse(
        ReadAtTheRacesOption(args, "--max-days"),
        out var parsedAtrMaxDays) && parsedAtrMaxDays > 0
        ? parsedAtrMaxDays
        : (int?)null;
    bool atrForce = args.Any(
        arg => string.Equals(arg, "--force", StringComparison.OrdinalIgnoreCase));

    Console.WriteLine(
        "[atr] A Chrome window may open while the site's client challenge is solved.");
    var atrSummary = atrService.Backfill(
        atrFrom,
        atrTo,
        atrMaxDays,
        atrForce,
        Console.WriteLine);
    Console.WriteLine(
        $"[atr] {atrSummary.RowsInserted} rows inserted for {atrSummary.RacesWithSourceData} races; " +
        $"{atrSummary.RacesWithoutSourceData} gaps had no timed source data.");
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
    var weatherRepository = new WeatherRepository(connectionString);
    using var weatherClient = new OpenMeteoWeatherClient(builder.Configuration);
    var weatherService = new WeatherBackfillService(weatherRepository, weatherClient, builder.Configuration);
    weatherService.Backfill(
        fromDate ?? weatherService.GetDefaultBackfillStart(),
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
builder.Services.AddSingleton<BettingSimulatorStore>();
builder.Services.AddScoped<SimulatorSnapshotService>();
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
builder.Services.AddScoped<IBackgroundJobHandler, SimulatorPrepareJobHandler>();
builder.Services.AddScoped<IBackgroundJobHandler, AutomateBetsJobHandler>();
builder.Services.AddHostedService<BackgroundJobWorker>();

var app = builder.Build();

// Compare how the members of a persisted ensemble are combined: equal
// geometric, equal arithmetic, dev-fitted weighted combinations, and each
// member alone. No retraining and no artifact changes.
if (args.Any(arg => string.Equals(arg, "--evaluate-ensemble-combination", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadCombinationArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var combinationRepository = new RacingRepository(connectionString);
    int combinationId = int.TryParse(ReadCombinationArg(args, "--id"), out var parsedCombinationId)
        ? parsedCombinationId
        : combinationRepository.GetBestMLParameter()?.Id ?? 85;
    var combinationRow = combinationRepository.GetMLParameterById(combinationId);
    if (combinationRow is null)
    {
        Console.WriteLine($"[combine] could not find MLParameters row {combinationId}");
        return;
    }

    var combinationModelDirectory = builder.Configuration.GetValue<string>("Training:ModelDirectory");
    var combinationModelPath = ModelArtifactStore.ResolveArtifactPath(
        combinationModelDirectory,
        combinationRow.ModelArtifact);
    var combinationReport = EnsembleCombinationEvaluator.Run(
        builder.Configuration,
        combinationRepository,
        new HyperparameterTrainer(builder.Configuration),
        combinationModelPath,
        combinationRow.DevStartDate ?? new DateTime(2024, 12, 13),
        combinationRow.DevEndDate ?? new DateTime(2025, 11, 29),
        combinationRow.TestStartDate ?? new DateTime(2025, 11, 30),
        combinationRow.TestEndDate ?? new DateTime(2026, 11, 21),
        Console.WriteLine);
    var combinationOutputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var combinationOutputPath = combinationOutputIndex >= 0 && combinationOutputIndex + 1 < args.Length
        ? args[combinationOutputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"ensemble-combination-{combinationId}-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        combinationOutputPath,
        JsonSerializer.Serialize(combinationReport, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[combine] report written to {combinationOutputPath}");
    return;
}

// Apply the field-bucket Platt parameters from a calibration report to an
// existing artifact. Writes a new artifact, activates it and updates the
// MLParameters row to point at the calibrated bytes.
if (args.Any(arg => string.Equals(arg, "--apply-field-calibration", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadApplyArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var applyRepository = new RacingRepository(connectionString);
    int applyId = int.TryParse(ReadApplyArg(args, "--id"), out var parsedApplyId)
        ? parsedApplyId
        : applyRepository.GetBestMLParameter()?.Id ?? 85;
    var applyRow = applyRepository.GetMLParameterById(applyId);
    if (applyRow is null)
    {
        Console.WriteLine($"[calibration] could not find MLParameters row {applyId}");
        return;
    }

    var applyModelDirectory = builder.Configuration.GetValue<string>("Training:ModelDirectory");
    var applyModelPath = ModelArtifactStore.ResolveArtifactPath(
        applyModelDirectory,
        applyRow.ModelArtifact);
    var reportPath = ReadApplyArg(args, "--report")
        ?? Path.Combine(Path.GetTempPath(), $"calibration-{applyId}.json");
    if (!File.Exists(reportPath))
    {
        Console.WriteLine($"[calibration] report not found: {reportPath}");
        return;
    }

    var reportDocument = JsonDocument.Parse(File.ReadAllText(reportPath));
    if (!reportDocument.RootElement.TryGetProperty("FieldBucketParameters", out var fieldParameters) ||
        fieldParameters.ValueKind != JsonValueKind.Array)
    {
        Console.WriteLine("[calibration] report has no FieldBucketParameters");
        return;
    }

    var buckets = new List<FieldCalibrationBucket>();
    foreach (var item in fieldParameters.EnumerateArray())
    {
        string bucket = item.TryGetProperty("Bucket", out var bucketElement)
            ? bucketElement.GetString() ?? string.Empty
            : string.Empty;
        double intercept = item.TryGetProperty("Intercept", out var interceptElement)
            ? interceptElement.GetDouble()
            : 0;
        double slope = item.TryGetProperty("Slope", out var slopeElement)
            ? slopeElement.GetDouble()
            : 1;
        int maxFieldSize = bucket switch
        {
            "field-0" => 8,
            "field-1" => 12,
            "field-2" => int.MaxValue,
            _ => buckets.Count == 0 ? 8 : buckets.Count == 1 ? 12 : int.MaxValue
        };
        buckets.Add(new FieldCalibrationBucket
        {
            MaxFieldSize = maxFieldSize,
            Intercept = intercept,
            Slope = slope
        });
    }

    if (buckets.Count == 0)
    {
        Console.WriteLine("[calibration] no field-bucket parameters found");
        return;
    }

    var applyModel = JsonSerializer.Deserialize<TrainedModel>(File.ReadAllText(applyModelPath))
        ?? throw new InvalidOperationException("Could not deserialize the model artifact.");
    applyModel.FieldCalibration = buckets
        .OrderBy(bucket => bucket.MaxFieldSize)
        .ToList();
    applyModel.TrainedUtc = DateTime.UtcNow;
    var calibratedArtifact = ModelArtifactStore.Save(
        applyModel,
        applyModelDirectory,
        activate: true);
    applyRepository.ActivateMLParameter(
        applyId,
        calibratedArtifact.FileName,
        calibratedArtifact.Sha256,
        calibratedArtifact.FeatureSchemaHash);
    Console.WriteLine(
        $"[calibration] applied {buckets.Count} field buckets to row {applyId}; " +
        $"active artifact {calibratedArtifact.FileName}");
    return;
}

// Re-price the cached simulator snapshot with the active model's field
// calibration. This avoids reloading history or retraining a replica when the
// only change is the post-combination calibration.
if (args.Any(arg => string.Equals(arg, "--apply-simulator-calibration", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadSimulatorCalibrationArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var simulatorCalibrationRepository = new RacingRepository(connectionString);
    int simulatorCalibrationId = int.TryParse(
        ReadSimulatorCalibrationArg(args, "--id"),
        out var parsedSimulatorCalibrationId)
        ? parsedSimulatorCalibrationId
        : simulatorCalibrationRepository.GetActiveMLParameter()?.Id ?? 92;
    var simulatorCalibrationRow = simulatorCalibrationRepository.GetMLParameterById(simulatorCalibrationId)
        ?? throw new InvalidOperationException($"MLParameters row {simulatorCalibrationId} was not found.");
    var simulatorCalibrationModelDirectory =
        builder.Configuration.GetValue<string>("Training:ModelDirectory");
    var simulatorCalibrationModelPath = ModelArtifactStore.ResolveArtifactPath(
        simulatorCalibrationModelDirectory,
        simulatorCalibrationRow.ModelArtifact);
    if (!File.Exists(simulatorCalibrationModelPath))
    {
        throw new FileNotFoundException(
            "The active model artifact was not found.",
            simulatorCalibrationModelPath);
    }

    var simulatorCalibrationModel = JsonSerializer.Deserialize<TrainedModel>(
        File.ReadAllText(simulatorCalibrationModelPath))
        ?? throw new InvalidOperationException("Could not deserialize the active model artifact.");
    var simulatorCalibrationStore = new BettingSimulatorStore(builder.Configuration);
    SimulatorCalibrationPatcher.Apply(
        simulatorCalibrationStore,
        simulatorCalibrationRow,
        simulatorCalibrationModel,
        Console.WriteLine);
    return;
}

// Deployable calibration comparison on one persisted ensemble: global Platt,
// market-rank-bucket Platt and field-size-bucket Platt, all fitted on dev and
// frozen for test.
if (args.Any(arg => string.Equals(arg, "--benchmark-calibration", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadCalibrationArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var calibrationRepository = new RacingRepository(connectionString);
    int calibrationId = int.TryParse(ReadCalibrationArg(args, "--id"), out var parsedCalibrationId)
        ? parsedCalibrationId
        : calibrationRepository.GetBestMLParameter()?.Id ?? 85;
    var calibrationRow = calibrationRepository.GetMLParameterById(calibrationId);
    if (calibrationRow is null)
    {
        Console.WriteLine($"[calibration] could not find MLParameters row {calibrationId}");
        return;
    }

    var calibrationModelDirectory = builder.Configuration.GetValue<string>("Training:ModelDirectory");
    var calibrationModelPath = ModelArtifactStore.ResolveArtifactPath(
        calibrationModelDirectory,
        calibrationRow.ModelArtifact);
    var calibrationReport = CalibrationEvaluator.Run(
        builder.Configuration,
        calibrationRepository,
        new HyperparameterTrainer(builder.Configuration),
        calibrationModelPath,
        calibrationRow.DevStartDate ?? new DateTime(2024, 12, 13),
        calibrationRow.DevEndDate ?? new DateTime(2025, 11, 29),
        calibrationRow.TestStartDate ?? new DateTime(2025, 11, 30),
        calibrationRow.TestEndDate ?? new DateTime(2026, 11, 21),
        Console.WriteLine);
    var calibrationOutputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var calibrationOutputPath = calibrationOutputIndex >= 0 && calibrationOutputIndex + 1 < args.Length
        ? args[calibrationOutputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"calibration-{calibrationId}-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        calibrationOutputPath,
        JsonSerializer.Serialize(calibrationReport, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[calibration] report written to {calibrationOutputPath}");
    return;
}

// Dev-only feature-block ablation against the active schema. The dataset is
// encoded once with all blocks; no test retraining and no artifact changes.
if (args.Any(arg => string.Equals(arg, "--benchmark-feature-blocks-dev", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadBlocksArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var blocksRepository = new RacingRepository(connectionString);
    var blocksTemplate = blocksRepository.GetMLParameterById(85)
        ?? blocksRepository.GetBestMLParameter()
        ?? new MLParameter
        {
            Units = 128,
            Layers = 2,
            Dropout = 0.15,
            LearningRate = 0.0007,
            Epochs = 10,
            BatchSize = 500,
            Folds = 1,
            RaceBatchSize = 64,
            Seed = 42
        };
    var blocksText = ReadBlocksArg(args, "--blocks")
        ?? "TrainerForm,FormContext,DeepForm,WeatherBalance,WeatherExtended";
    var blocksToTest = blocksText
        .Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)
        .Select(name => FeatureBlockNames.Parse(name, FeatureBlocks.None))
        .Where(block => block != FeatureBlocks.None)
        .ToList();
    int blocksSeed = int.TryParse(ReadBlocksArg(args, "--seed"), out var parsedBlocksSeed)
        ? parsedBlocksSeed
        : 42;
    var blocksOutputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var blocksOutputPath = blocksOutputIndex >= 0 && blocksOutputIndex + 1 < args.Length
        ? args[blocksOutputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"feature-blocks-dev-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    var blocksJsonOptions = new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true
    };
    FeatureBlockDevSweep.Report? blocksResumeReport = null;
    if (args.Any(arg => string.Equals(arg, "--resume", StringComparison.OrdinalIgnoreCase))
        && File.Exists(blocksOutputPath))
    {
        try
        {
            blocksResumeReport = JsonSerializer.Deserialize<FeatureBlockDevSweep.Report>(
                File.ReadAllText(blocksOutputPath),
                blocksJsonOptions);
            Console.WriteLine(
                $"[blocks-dev] resuming from {blocksOutputPath} " +
                $"({blocksResumeReport?.Candidates.Count ?? 0} candidate(s))");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[blocks-dev] could not resume from checkpoint: {ex.Message}");
        }
    }

    void WriteBlocksReport(FeatureBlockDevSweep.Report report)
    {
        File.WriteAllText(
            blocksOutputPath,
            JsonSerializer.Serialize(report, blocksJsonOptions));
    }

    var blocksReport = FeatureBlockDevSweep.Run(
        builder.Configuration,
        blocksRepository,
        blocksTemplate,
        blocksTemplate.DevStartDate ?? new DateTime(2024, 12, 13),
        blocksTemplate.DevEndDate ?? new DateTime(2025, 11, 29),
        blocksToTest,
        blocksSeed,
        Console.WriteLine,
        WriteBlocksReport,
        blocksResumeReport);
    WriteBlocksReport(blocksReport);
    Console.WriteLine($"[blocks-dev] report written to {blocksOutputPath}");
    return;
}

// Dev-only sweep of Training:PairwiseLossWeight. The dataset is encoded once;
// every weight/seed candidate shares the same train/dev split. No final/test
// retraining and no artifact persistence.
if (args.Any(arg => string.Equals(arg, "--benchmark-pairwise", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadSweepArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var pairwiseRepository = new RacingRepository(connectionString);
    var pairwiseTemplate = pairwiseRepository.GetMLParameterById(85)
        ?? pairwiseRepository.GetBestMLParameter()
        ?? new MLParameter
        {
            Units = 128,
            Layers = 2,
            Dropout = 0.15,
            LearningRate = 0.0007,
            Epochs = 10,
            BatchSize = 500,
            Folds = 1,
            RaceBatchSize = 64,
            Seed = 42
        };
    var pairwiseDevStart = pairwiseTemplate.DevStartDate ?? new DateTime(2024, 12, 13);
    var pairwiseDevEnd = pairwiseTemplate.DevEndDate ?? new DateTime(2025, 11, 29);
    var weightsText = ReadSweepArg(args, "--weights") ?? "0,0.02,0.05,0.1";
    var pairwiseWeights = weightsText
        .Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)
        .Select(value => double.Parse(value, System.Globalization.CultureInfo.InvariantCulture))
        .ToList();
    var seedsText = ReadSweepArg(args, "--seeds") ?? "42,43";
    var pairwiseSeeds = seedsText
        .Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)
        .Select(int.Parse)
        .ToList();

    var pairwiseReport = PairwiseLossSweep.Run(
        builder.Configuration,
        pairwiseRepository,
        pairwiseTemplate,
        pairwiseDevStart,
        pairwiseDevEnd,
        pairwiseWeights,
        pairwiseSeeds,
        Console.WriteLine);
    var pairwiseOutputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var pairwiseOutputPath = pairwiseOutputIndex >= 0 && pairwiseOutputIndex + 1 < args.Length
        ? args[pairwiseOutputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"pairwise-sweep-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        pairwiseOutputPath,
        JsonSerializer.Serialize(pairwiseReport, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine(
        $"[pairwise] best Brier weight {pairwiseReport.BestBrierWeight:0.00}, best log-loss weight {pairwiseReport.BestLogLossWeight:0.00}");
    Console.WriteLine($"[pairwise] report written to {pairwiseOutputPath}");
    return;
}

// One-off comparison of two saved models on the same chronological windows.
// It fits blend weights on dev and reports the frozen blend on test; it does
// not activate or modify an artifact.
if (args.Any(arg => string.Equals(arg, "--blend-evaluate", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadBlendArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var blendRepository = new RacingRepository(connectionString);
    int newId = int.TryParse(ReadBlendArg(args, "--new-id"), out var parsedNewId)
        ? parsedNewId
        : blendRepository.GetBestMLParameter()?.Id ?? 0;
    int oldId = int.TryParse(ReadBlendArg(args, "--old-id"), out var parsedOldId)
        ? parsedOldId
        : 81;
    var newRow = newId > 0 ? blendRepository.GetMLParameterById(newId) : null;
    var oldRow = blendRepository.GetMLParameterById(oldId);
    if (newRow is null || oldRow is null)
    {
        Console.WriteLine($"[blend] could not find MLParameters rows new={newId}, old={oldId}");
        return;
    }

    var modelDirectory = builder.Configuration.GetValue<string>("Training:ModelDirectory");
    var newModelPath = ModelArtifactStore.ResolveArtifactPath(
        modelDirectory,
        newRow.ModelArtifact);
    var oldModelPath = ModelArtifactStore.ResolveArtifactPath(
        modelDirectory,
        oldRow.ModelArtifact);
    var blendDevStart = newRow.DevStartDate ?? new DateTime(2024, 12, 13);
    var blendDevEnd = newRow.DevEndDate ?? new DateTime(2025, 11, 29);
    var blendTestStart = newRow.TestStartDate ?? new DateTime(2025, 11, 30);
    var blendTestEnd = newRow.TestEndDate ?? new DateTime(2026, 11, 21);

    var blendReport = ModelBlendEvaluator.Run(
        builder.Configuration,
        blendRepository,
        new HyperparameterTrainer(builder.Configuration),
        newModelPath,
        oldModelPath,
        blendDevStart,
        blendDevEnd,
        blendTestStart,
        blendTestEnd,
        Console.WriteLine);

    var blendOutputIndex = Array.FindIndex(
        args,
        arg => string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase));
    var blendOutputPath = blendOutputIndex >= 0 && blendOutputIndex + 1 < args.Length
        ? args[blendOutputIndex + 1]
        : Path.Combine(
            Path.GetTempPath(),
            $"blend-evaluation-{DateTime.Now:yyyyMMdd-HHmmss}.json");
    File.WriteAllText(
        blendOutputPath,
        JsonSerializer.Serialize(blendReport, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[blend] report written to {blendOutputPath}");
    return;
}

// Run the same chronological train/dev/test ensemble path as the TrainAI page,
// but from a console command so a full training run can persist its rows and
// active artifact without keeping a browser request open.
if (args.Any(arg => string.Equals(arg, "--train-ensemble", StringComparison.OrdinalIgnoreCase)))
{
    static string? ReadTrainArg(string[] arguments, string name)
    {
        int index = Array.FindIndex(
            arguments,
            argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
    }

    var overrides = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
    var trainMinimumDate = ReadTrainArg(args, "--minimum-race-date");
    if (!string.IsNullOrWhiteSpace(trainMinimumDate))
    {
        overrides["Training:MinimumRaceDate"] = trainMinimumDate;
        overrides["Training:MinimumMarketRaceDate"] = trainMinimumDate;
    }

    var ensembleSizeValue = ReadTrainArg(args, "--ensemble-size");
    if (!string.IsNullOrWhiteSpace(ensembleSizeValue))
    {
        overrides["Training:EnsembleSize"] = ensembleSizeValue;
    }

    var seedCountValue = ReadTrainArg(args, "--seed-count");
    if (!string.IsNullOrWhiteSpace(seedCountValue))
    {
        overrides["Training:SeedCount"] = seedCountValue;
    }

    if (args.Any(arg => string.Equals(arg, "--no-blind", StringComparison.OrdinalIgnoreCase)))
    {
        overrides["Training:MarketBlindEnsembleMembers"] = "0";
    }

    if (args.Any(arg => string.Equals(arg, "--force", StringComparison.OrdinalIgnoreCase)))
    {
        overrides["Training:RequireMarketEdge"] = "false";
    }

    var recencyValue = ReadTrainArg(args, "--recency-half-life-days");
    if (!string.IsNullOrWhiteSpace(recencyValue))
    {
        overrides["Training:RecencyHalfLifeDays"] = recencyValue;
    }

    var trainConfiguration = new ConfigurationBuilder()
        .AddConfiguration(builder.Configuration)
        .AddInMemoryCollection(overrides)
        .Build();
    var trainRepository = new RacingRepository(connectionString);
    var template = trainRepository.GetBestMLParameter();
    var trainParameter = new MLParameter
    {
        RunDate = DateTime.UtcNow,
        Units = int.TryParse(ReadTrainArg(args, "--units"), out var units) ? units : template?.Units ?? 128,
        Layers = int.TryParse(ReadTrainArg(args, "--layers"), out var layers) ? layers : template?.Layers ?? 2,
        Dropout = double.TryParse(ReadTrainArg(args, "--dropout"), out var dropout) ? dropout : template?.Dropout ?? 0.15,
        LearningRate = double.TryParse(ReadTrainArg(args, "--learning-rate"), out var learningRate) ? learningRate : template?.LearningRate ?? 0.0007,
        Epochs = int.TryParse(ReadTrainArg(args, "--epochs"), out var epochs)
            ? epochs
            : Math.Max(10, template?.Epochs ?? 12),
        BatchSize = int.TryParse(ReadTrainArg(args, "--batch-size"), out var batchSize) ? batchSize : template?.BatchSize ?? 500,
        Folds = 1
        // Leave Objective, FeatureBlocks, Seed, RaceBatchSize and MarketBlind
        // null so the controller fills them from the current configuration.
    };

    var trainBatch = new MLParameterBatch
    {
        Parameters = new List<MLParameter> { trainParameter }
    };
    var trainController = new HyperparameterController(
        trainRepository,
        new HyperparameterTrainer(trainConfiguration),
        trainConfiguration,
        app.Services.GetRequiredService<BackgroundJobQueue>(),
        app.Services.GetRequiredService<BackgroundJobStore>(),
        app.Services.GetRequiredService<ILogger<HyperparameterController>>(),
        app.Services.GetRequiredService<BettingSimulatorStore>());
    Console.WriteLine(
        $"[train-ensemble] start | minimum date {trainMinimumDate ?? "(config)"} | ensemble {trainConfigValue(trainConfiguration, "Training:EnsembleSize")} | seeds {trainConfigValue(trainConfiguration, "Training:SeedCount")} | blind {trainConfigValue(trainConfiguration, "Training:MarketBlindEnsembleMembers")}");
    await trainController.RunTrainAI(trainBatch);
    Console.WriteLine("[train-ensemble] finished.");
    return;

    static string trainConfigValue(IConfiguration configuration, string key) =>
        configuration.GetValue<string>(key) ?? "(default)";
}

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