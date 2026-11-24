using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using Microsoft.AspNetCore.Mvc.ModelBinding.Validation;

namespace HorseRacingML.Models
{
    /// <summary>
    /// Request used to prepare a new evaluation window. Preparing a window
    /// replays the saved model over history once; every subsequent filter
    /// change re-uses the cached runner-level predictions.
    /// </summary>
    public sealed class SimulatorRunRequest : IValidatableObject
    {
        public const int MinimumMonths = 1;
        public const int MaximumMonths = 24;

        [Range(MinimumMonths, MaximumMonths, ErrorMessage = "Please choose between 1 and 24 months.")]
        public int Months { get; set; } = 6;

        /// <summary>
        /// Optional explicit window start. When supplied it takes precedence
        /// over <see cref="Months"/>, which makes a backtest reproducible.
        /// </summary>
        public DateTime? FromDate { get; set; }

        /// <summary>
        /// Optional explicit window end. When omitted the latest race date in
        /// the database is used, rather than today's date.
        /// </summary>
        public DateTime? ToDate { get; set; }

        /// <summary>
        /// Use the active persisted model for inference when the evaluation
        /// window starts after the model's training period. Otherwise a
        /// replica is retrained on the history before the window.
        /// </summary>
        public bool UseSavedModel { get; set; } = true;

        public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
        {
            if (FromDate.HasValue && ToDate.HasValue && FromDate.Value.Date > ToDate.Value.Date)
            {
                yield return new ValidationResult(
                    "The window start date must not be after the end date.",
                    new[] { nameof(FromDate), nameof(ToDate) });
            }

            if (!FromDate.HasValue && !ToDate.HasValue && (Months < 1 || Months > 24))
            {
                yield return new ValidationResult(
                    "Please choose between 1 and 24 months.",
                    new[] { nameof(Months) });
            }
        }
    }

    public enum SimulatorStakeMode
    {
        Kelly = 0,
        Flat = 1,
        BankrollPercent = 2
    }

    public enum SimulatorSelectionStrategy
    {
        HighestEdge = 0,
        HighestDifferential = 1,
        HighestKelly = 2,
        ShortestPrice = 3,
        MarketFavourite = 4,
        ModelTopPick = 5
    }

    public enum SimulatorPriceSource
    {
        StartingPrice = 0,
        BetfairSp = 1,
        LastTraded = 2,
        BestAvailable = 3
    }

    /// <summary>
    /// Bets are selected from the cached prediction snapshot using these
    /// rules. Changing a filter only re-runs the betting loop, not the model.
    /// </summary>
    public sealed class SimulatorFilterRequest : IValidatableObject
    {
        public const decimal DefaultMinimumEdgePercent = 0.01m;
        public const decimal DefaultMinimumProbabilitySpread = 0.01m;
        /// <summary>
        /// Key of the persisted snapshot this filter applies to. Empty means
        /// the most recently published snapshot.
        /// </summary>
        [StringLength(64)]
        public string? SnapshotKey { get; set; }

        [Range(0, 1_000_000_000, ErrorMessage = "Starting bankroll must be zero or more.")]
        public decimal StartingBankroll { get; set; } = 100m;

        [Range(0, 60, ErrorMessage = "Minimum runners must be between 0 and 60.")]
        public int? MinRunners { get; set; }

        [Range(0, 60, ErrorMessage = "Maximum runners must be between 0 and 60.")]
        public int? MaxRunners { get; set; }

        [StringLength(64)]
        public string RaceType { get; set; } = "Any";

        [StringLength(64)]
        public string Surface { get; set; } = "Any";

        [StringLength(64)]
        public string Going { get; set; } = "Any";

        [StringLength(64)]
        public string AgeRestriction { get; set; } = "Any";

        [Range(1, 10, ErrorMessage = "Class must be between 1 and 10.")]
        public byte? ClassMin { get; set; }

        [Range(1, 10, ErrorMessage = "Class must be between 1 and 10.")]
        public byte? ClassMax { get; set; }

        [StringLength(64)]
        public string? CourseContains { get; set; }

        /// <summary>
        /// Optional multi-select course filter. When empty or null, all
        /// courses are included. Case-insensitive. The type is nullable so MVC
        /// does not implicitly treat a non-nullable reference type as a
        /// required field, which made the browser refuse an empty selection.
        /// </summary>
        [ValidateNever]
        public List<string>? Courses { get; set; }

        [Range(0, 12_000, ErrorMessage = "Distance must be between 0 and 12,000 yards.")]
        public short? MinDistanceYards { get; set; }

        [Range(0, 12_000, ErrorMessage = "Distance must be between 0 and 12,000 yards.")]
        public short? MaxDistanceYards { get; set; }

        public DateTime? FromDate { get; set; }

        public DateTime? ToDate { get; set; }

        /// <summary>
        /// Minimum expected value per unit staked, net of commission.
        /// 5 means the AI price must imply at least a 5% profit on turnover.
        /// The default is deliberately positive: a zero-EV bet adds variance
        /// without adding expected profit.
        /// </summary>
        [Range(-100, 10_000, ErrorMessage = "Minimum edge must be between -100% and 10,000%.")]
        public decimal MinEdgePercent { get; set; } = DefaultMinimumEdgePercent;

        [Range(-100, 10_000, ErrorMessage = "Maximum edge must be between -100% and 10,000%.")]
        public decimal? MaxEdgePercent { get; set; }

        [Range(1.01, 100_000, ErrorMessage = "Minimum odds must be at least 1.01.")]
        public decimal MinOdds { get; set; } = 1.01m;

        [Range(1.01, 100_000, ErrorMessage = "Maximum odds must be at least 1.01.")]
        public decimal? MaxOdds { get; set; }

        [Range(0, 100, ErrorMessage = "Minimum AI probability must be between 0% and 100%.")]
        public decimal MinAiProbabilityPercent { get; set; }

        [Range(0, 100, ErrorMessage = "Maximum AI probability must be between 0% and 100%.")]
        public decimal? MaxAiProbabilityPercent { get; set; }

        /// <summary>
        /// Near-uniform model outputs are skipped, matching the live path.
        /// The value is measured on the raw per-race probability spread.
        /// </summary>
        [Range(0, 1, ErrorMessage = "Probability spread must be between 0 and 1.")]
        public decimal MinProbabilitySpread { get; set; } = DefaultMinimumProbabilitySpread;

        [Range(0, 100, ErrorMessage = "Commission must be between 0% and 100%.")]
        public decimal CommissionPercent { get; set; } = 5m;

        public SimulatorStakeMode StakeMode { get; set; } = SimulatorStakeMode.Kelly;

        [Range(0, 100, ErrorMessage = "Maximum Kelly must be between 0% and 100%.")]
        public decimal MaxKellyPercent { get; set; } = 10m;

        [Range(0, 1_000_000_000, ErrorMessage = "Flat stake must be zero or more.")]
        public decimal FlatStake { get; set; } = 1m;

        [Range(0, 100, ErrorMessage = "Bankroll percent must be between 0% and 100%.")]
        public decimal BankrollPercent { get; set; } = 1m;

        /// <summary>
        /// Hard cap on a single stake as a percentage of the bankroll. This is
        /// separate from the fractional-Kelly cap. Zero disables the cap.
        /// </summary>
        [Range(0, 100, ErrorMessage = "Maximum stake percent must be between 0% and 100%.")]
        public decimal MaxStakePercent { get; set; } = 100m;

        [Range(0, 1_000_000_000, ErrorMessage = "Maximum stake must be zero or more.")]
        public decimal? MaxStakeAmount { get; set; }

        [Range(0, 1_000_000_000, ErrorMessage = "Minimum stake must be zero or more.")]
        public decimal MinStake { get; set; }

        /// <summary>
        /// Maximum total staked on one race day as a percentage of the
        /// bankroll at the start of that day. Zero disables the cap.
        /// </summary>
        [Range(0, 100, ErrorMessage = "Maximum daily stake percent must be between 0% and 100%.")]
        public decimal MaxDailyStakePercent { get; set; } = 100m;

        /// <summary>
        /// Stop betting for the rest of the day once the day's realised loss
        /// reaches this percentage of the day-start bankroll. Zero disables it.
        /// </summary>
        [Range(0, 100, ErrorMessage = "Maximum daily loss percent must be between 0% and 100%.")]
        public decimal MaxDailyLossPercent { get; set; } = 100m;

        [Range(1, 10, ErrorMessage = "Bets per race must be between 1 and 10.")]
        public int BetsPerRace { get; set; } = 1;

        public SimulatorSelectionStrategy SelectionStrategy { get; set; } =
            SimulatorSelectionStrategy.HighestDifferential;

        public SimulatorPriceSource PriceSource { get; set; } = SimulatorPriceSource.StartingPrice;

        /// <summary>
        /// Price haircut applied to the selected odds before commission. Use
        /// 1-3% to model exchange spread or a worse-than-SP fill.
        /// </summary>
        [Range(0, 50, ErrorMessage = "Price slippage must be between 0% and 50%.")]
        public decimal PriceSlippagePercent { get; set; }

        /// <summary>
        /// The last N percent of the window is held back as a confirmation
        /// sample. Filter tuning should use the first part only.
        /// </summary>
        [Range(0, 50, ErrorMessage = "Confirmation percent must be between 0% and 50%.")]
        public decimal ConfirmPercent { get; set; } = 30m;

        [Range(0, 10_000, ErrorMessage = "Bootstrap samples must be between 0 and 10,000.")]
        public int BootstrapSamples { get; set; } = 1_000;

        public int BootstrapSeed { get; set; } = 12_345;

        /// <summary>1-based page number for the displayed bet list.</summary>
        [Range(1, 1_000_000, ErrorMessage = "Page must be at least 1.")]
        public int Page { get; set; } = 1;

        [Range(10, 1_000, ErrorMessage = "Page size must be between 10 and 1,000.")]
        public int PageSize { get; set; } = 100;

        public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
        {
            if (MinRunners.HasValue && MaxRunners.HasValue && MinRunners.Value > MaxRunners.Value)
            {
                yield return new ValidationResult(
                    "Minimum runners must not exceed maximum runners.",
                    new[] { nameof(MinRunners), nameof(MaxRunners) });
            }

            if (ClassMin.HasValue && ClassMax.HasValue && ClassMin.Value > ClassMax.Value)
            {
                yield return new ValidationResult(
                    "Minimum class must not exceed maximum class.",
                    new[] { nameof(ClassMin), nameof(ClassMax) });
            }

            if (MinOdds > 0m && MaxOdds.HasValue && MinOdds > MaxOdds.Value)
            {
                yield return new ValidationResult(
                    "Minimum odds must not exceed maximum odds.",
                    new[] { nameof(MinOdds), nameof(MaxOdds) });
            }

            if (MaxEdgePercent.HasValue && MinEdgePercent > MaxEdgePercent.Value)
            {
                yield return new ValidationResult(
                    "Minimum edge must not exceed maximum edge.",
                    new[] { nameof(MinEdgePercent), nameof(MaxEdgePercent) });
            }

            if (MaxAiProbabilityPercent.HasValue && MinAiProbabilityPercent > MaxAiProbabilityPercent.Value)
            {
                yield return new ValidationResult(
                    "Minimum AI probability must not exceed maximum AI probability.",
                    new[] { nameof(MinAiProbabilityPercent), nameof(MaxAiProbabilityPercent) });
            }

            if (MinDistanceYards.HasValue && MaxDistanceYards.HasValue &&
                MinDistanceYards.Value > MaxDistanceYards.Value)
            {
                yield return new ValidationResult(
                    "Minimum distance must not exceed maximum distance.",
                    new[] { nameof(MinDistanceYards), nameof(MaxDistanceYards) });
            }

            if (FromDate.HasValue && ToDate.HasValue && FromDate.Value.Date > ToDate.Value.Date)
            {
                yield return new ValidationResult(
                    "The from date must not be after the to date.",
                    new[] { nameof(FromDate), nameof(ToDate) });
            }

            if (!Enum.IsDefined(typeof(SimulatorStakeMode), StakeMode))
            {
                yield return new ValidationResult("Unknown stake mode.", new[] { nameof(StakeMode) });
            }

            if (!Enum.IsDefined(typeof(SimulatorSelectionStrategy), SelectionStrategy))
            {
                yield return new ValidationResult(
                    "Unknown selection strategy.",
                    new[] { nameof(SelectionStrategy) });
            }

            if (!Enum.IsDefined(typeof(SimulatorPriceSource), PriceSource))
            {
                yield return new ValidationResult("Unknown price source.", new[] { nameof(PriceSource) });
            }
        }

        /// <summary>
        /// Shallow copy used by analytics (split, walk-forward, baselines).
        /// </summary>
        public SimulatorFilterRequest Clone()
        {
            var clone = (SimulatorFilterRequest)MemberwiseClone();
            clone.Courses = Courses is null ? new List<string>() : new List<string>(Courses);
            return clone;
        }
    }

    public sealed class BettingSimulatorViewModel
    {
        public SimulatorRunRequest Run { get; set; } = new();
        public SimulatorFilterRequest Filter { get; set; } = new();
        public BettingSimulationSnapshotSummary? Snapshot { get; set; }
        public SimulatorPrepareJobViewModel? PrepareJob { get; set; }
        public BettingSimulationResult? Result { get; set; }
        public SimulatorComparison? Comparison { get; set; }
        public IReadOnlyList<SimulatorBaselineRow> Baselines { get; set; } =
            Array.Empty<SimulatorBaselineRow>();
        public IReadOnlyList<SimulatorPeriodResult> WalkForward { get; set; } =
            Array.Empty<SimulatorPeriodResult>();
        public IReadOnlyList<BettingSimulationSnapshotSummary> Snapshots { get; set; } =
            Array.Empty<BettingSimulationSnapshotSummary>();
        public IReadOnlyList<SimulatorBet> PagedBets { get; set; } =
            Array.Empty<SimulatorBet>();
        public string? SelectedSnapshotKey { get; set; }
        public string? StaleModelWarning { get; set; }
        public string? Message { get; set; }
        public int Page { get; set; } = 1;
        public int PageSize { get; set; } = 100;
        public int PageCount { get; set; }
        public bool HasSnapshot => Snapshot is not null;
        public bool HasResult => Result is not null;
        public bool IsPreparing => PrepareJob?.IsActive == true;
        public bool HasConfirmation => Comparison is not null;
    }

    public sealed class SimulatorPrepareJobViewModel
    {
        public Guid JobId { get; set; }
        public string Status { get; set; } = string.Empty;
        public string? Message { get; set; }
        public int? Progress { get; set; }
        public string? Error { get; set; }
        public DateTime CreatedUtc { get; set; }
        public DateTime? StartedUtc { get; set; }
        public DateTime? CompletedUtc { get; set; }
        public bool IsActive =>
            string.Equals(Status, "Queued", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(Status, "Running", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// A compact, allocation-light copy of the validation predictions. It is
    /// kept in memory for the lifetime of the app so filter changes never
    /// retrain the model.
    /// </summary>
    public sealed class BettingSimulationSnapshot
    {
        public BettingSimulationSnapshotSummary Summary { get; set; } = new();

        /// <summary>Flat runner rows, retained for serialization and tests.</summary>
        public List<SimulatorRunnerRow> Runners { get; set; } = new();

        /// <summary>
        /// Pre-grouped races so filter changes do not re-run a full GroupBy.
        /// Older snapshots or hand-built test snapshots may leave this empty;
        /// the engine falls back to <see cref="Runners"/>.
        /// </summary>
        public List<SimulatorRace> Races { get; set; } = new();
    }

    /// <summary>
    /// A race and its runners. Header fields are copied from the first runner
    /// so the engine can apply race filters without touching every row.
    /// </summary>
    public sealed class SimulatorRace
    {
        public int RaceId { get; set; }
        public DateTime? RaceDate { get; set; }
        public string? RaceTitle { get; set; }
        public string? CourseName { get; set; }
        public string? RaceType { get; set; }
        public byte? Class { get; set; }
        public string? AgeRestriction { get; set; }
        public string? Surface { get; set; }
        public string? Going { get; set; }
        public short? DistanceYards { get; set; }
        public string? DistanceText { get; set; }
        public int RunnerCount { get; set; }
        public List<SimulatorRunnerRow> Runners { get; set; } = new();
    }

    public sealed class BettingSimulationSnapshotSummary
    {
        public string Key { get; set; } = string.Empty;
        public DateTime CreatedUtc { get; set; }
        public string Source { get; set; } = string.Empty;
        public DateTime ValidationStart { get; set; }
        public DateTime ValidationEnd { get; set; }
        public DateTime? FirstRaceDate { get; set; }
        public DateTime? LastRaceDate { get; set; }
        public int? Months { get; set; }
        public int TrainingRaceCount { get; set; }
        public int ValidationRaceCount { get; set; }
        public int RunnerCount { get; set; }
        public int PricedRunnerCount { get; set; }
        public int UnpricedRunnerCount { get; set; }
        public string? Objective { get; set; }
        public string? ModelArtifact { get; set; }
        public string? ModelHash { get; set; }
        public string? FeatureSchemaHash { get; set; }
        public bool UsedSavedModel { get; set; }
        public DateTime? ModelTrainingEnd { get; set; }
        public double BlendModelWeight { get; set; } = 1d;
        public double BlendMarketWeight { get; set; }
        public List<string> RaceTypes { get; set; } = new();
        public List<string> Surfaces { get; set; } = new();
        public List<string> Goings { get; set; } = new();
        public List<string> Courses { get; set; } = new();
        public List<string> AgeRestrictions { get; set; } = new();
        public bool HasBetfairPrices { get; set; }
        public double PriceCoverage => RunnerCount <= 0
            ? 0d
            : (double)PricedRunnerCount / RunnerCount;
    }

    public sealed class SimulatorRunnerRow
    {
        public int RaceId { get; set; }
        public DateTime? RaceDate { get; set; }
        public string? RaceTitle { get; set; }
        public string? CourseName { get; set; }
        public string? RaceType { get; set; }
        public byte? Class { get; set; }
        public string? AgeRestriction { get; set; }
        public string? Surface { get; set; }
        public string? Going { get; set; }
        public short? DistanceYards { get; set; }
        public string? DistanceText { get; set; }
        public int RunnerCount { get; set; }
        public string? HorseName { get; set; }
        public int? FinishPosition { get; set; }
        public decimal? DecimalOdds { get; set; }
        public decimal? BetfairSp { get; set; }
        public decimal? BetfairLastTradedPrice { get; set; }

        /// <summary>Probability after normalising within the race.</summary>
        public double Probability { get; set; }

        /// <summary>Raw model output, before within-race normalisation.</summary>
        public double RawProbability { get; set; }

        public bool Won { get; set; }
    }

    public sealed class BettingSimulationResult
    {
        public decimal StartingBankroll { get; init; }
        public decimal EndingBankroll { get; init; }
        public decimal TotalStaked { get; init; }
        public decimal Profit => EndingBankroll - StartingBankroll;
        public int BetCount { get; init; }
        public int WinCount { get; init; }
        public int LossCount => BetCount - WinCount;
        public int LongestWinStreak { get; init; }
        public int LongestLosingStreak { get; init; }
        public decimal MaxDrawdown { get; init; }
        public decimal MaxDrawdownPercent { get; init; }
        public double AverageEdge { get; init; }
        public double AverageOdds { get; init; }
        public double AverageEffectiveOdds { get; init; }
        public double AverageAiProbability { get; init; }
        public double AverageMarketProbability { get; init; }
        public decimal AverageStake { get; init; }
        public decimal GrossWinnings { get; init; }
        public decimal GrossLosses { get; init; }

        public int RacesConsidered { get; init; }
        public int RacesPassedRaceFilter { get; init; }
        public int RacesSkippedLowSpread { get; init; }
        public int RacesWithCandidate { get; init; }
        public int RacesWithStake { get; init; }
        public int UnpricedRunnerRows { get; init; }
        public bool StoppedForBankroll { get; init; }

        public decimal ExpectedProfit { get; init; }
        public double ExpectedWins { get; init; }
        public double BreakEvenStrikeRate { get; init; }
        public double RoiConfidenceLow { get; init; }
        public double RoiConfidenceHigh { get; init; }
        public decimal ProfitConfidenceLow { get; init; }
        public decimal ProfitConfidenceHigh { get; init; }
        public double ProbabilityOfProfit { get; init; }
        public double StandardError { get; init; }
        public double TStatistic { get; init; }
        public int BootstrapSamples { get; init; }
        public SimulatorPriceSource PriceSource { get; init; }
        public string? SnapshotKey { get; init; }

        public IReadOnlyList<SimulatorBet> Bets { get; init; } = Array.Empty<SimulatorBet>();
        public IReadOnlyList<SimulatorEquityPoint> EquityCurve { get; init; } = Array.Empty<SimulatorEquityPoint>();

        public decimal Roi => TotalStaked <= 0 ? 0 : Profit / TotalStaked;
        public double ExpectedRoi => TotalStaked <= 0 ? 0d : (double)(ExpectedProfit / TotalStaked);
        public double StrikeRate => BetCount <= 0 ? 0 : (double)WinCount / BetCount;
        public decimal ExpectedProfitPerBet => BetCount <= 0 ? 0 : ExpectedProfit / BetCount;
        public decimal RealisedProfitPerBet => BetCount <= 0 ? 0 : Profit / BetCount;
        public decimal? ProfitFactor => GrossLosses > 0
            ? GrossWinnings / GrossLosses
            : GrossWinnings > 0 ? null : 0m;
        public double SelectionRate => RacesPassedRaceFilter <= 0
            ? 0d
            : (double)BetCount / RacesPassedRaceFilter;

        public IReadOnlyList<SimulatorBreakdownRow> ByRaceType =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.RaceType) ? "Unknown" : bet.RaceType!);

        public IReadOnlyList<SimulatorBreakdownRow> ByFieldSize =>
            BuildBreakdown(bet => bet.RunnerCount switch
            {
                <= 0 => "Unknown",
                <= 5 => "1-5 runners",
                <= 8 => "6-8 runners",
                <= 12 => "9-12 runners",
                <= 16 => "13-16 runners",
                _ => "17+ runners"
            });

        public IReadOnlyList<SimulatorBreakdownRow> ByClass =>
            BuildBreakdown(bet => bet.Class.HasValue ? $"Class {bet.Class.Value}" : "Unknown");

        public IReadOnlyList<SimulatorBreakdownRow> ByGoing =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.Going) ? "Unknown" : bet.Going!);

        public IReadOnlyList<SimulatorBreakdownRow> BySurface =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.Surface) ? "Unknown" : bet.Surface!);

        public IReadOnlyList<SimulatorBreakdownRow> ByCourse =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.CourseName) ? "Unknown" : bet.CourseName!);

        public IReadOnlyList<SimulatorBreakdownRow> ByAgeRestriction =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.AgeRestriction) ? "Unknown" : bet.AgeRestriction!);

        public IReadOnlyList<SimulatorBreakdownRow> ByDistance =>
            BuildBreakdown(bet => bet.DistanceYards switch
            {
                null or <= 0 => "Unknown",
                < 1_320 => "Sprint (<6f)",
                < 1_760 => "6f-1m",
                < 2_200 => "1m-1m4f",
                _ => "Staying (1m4f+)"
            });

        public IReadOnlyList<SimulatorBreakdownRow> ByMonth =>
            BuildBreakdown(bet => bet.RaceDate?.ToString("yyyy-MM") ?? "Unknown");

        public IReadOnlyList<SimulatorBreakdownRow> ByEdgeBand =>
            BuildBreakdown(bet => bet.Edge switch
            {
                < 0d => "Negative edge",
                < 0.05d => "0-5%",
                < 0.10d => "5-10%",
                < 0.20d => "10-20%",
                < 0.50d => "20-50%",
                _ => "50%+"
            });

        public IReadOnlyList<SimulatorBreakdownRow> ByOddsBand =>
            BuildBreakdown(bet => bet.DecimalOdds switch
            {
                < 2m => "1.01-1.99",
                < 3m => "2.00-2.99",
                < 5m => "3.00-4.99",
                < 10m => "5.00-9.99",
                _ => "10.00+"
            });

        private IReadOnlyList<SimulatorBreakdownRow> BuildBreakdown(
            Func<SimulatorBet, string> keySelector)
        {
            return Bets
                .GroupBy(keySelector)
                .Select(group => new SimulatorBreakdownRow
                {
                    Group = group.Key,
                    Bets = group.Count(),
                    Wins = group.Count(bet => bet.Won),
                    Staked = group.Sum(bet => bet.Stake),
                    Profit = group.Sum(bet => bet.Profit),
                    ExpectedProfit = group.Sum(bet => bet.ExpectedProfit),
                    AverageStake = group.Average(bet => bet.Stake),
                    AverageEdge = group.Average(bet => bet.Edge),
                    AverageOdds = group.Average(bet => (double)bet.DecimalOdds),
                    GrossWinnings = group.Where(bet => bet.Won).Sum(bet => bet.Profit),
                    GrossLosses = group.Where(bet => !bet.Won).Sum(bet => Math.Abs(bet.Profit))
                })
                .OrderByDescending(row => row.Bets)
                .ThenBy(row => row.Group)
                .ToList();
        }
    }

    public sealed class SimulatorBet
    {
        public int RaceId { get; init; }
        public DateTime? RaceDate { get; init; }
        public string? RaceTitle { get; init; }
        public string? CourseName { get; init; }
        public string? HorseName { get; init; }
        public string? RaceType { get; init; }
        public byte? Class { get; init; }
        public string? AgeRestriction { get; init; }
        public string? Surface { get; init; }
        public string? Going { get; init; }
        public short? DistanceYards { get; init; }
        public string? DistanceText { get; init; }
        public int RunnerCount { get; init; }
        public int? FinishPosition { get; init; }
        public decimal DecimalOdds { get; init; }
        public decimal EffectiveDecimalOdds { get; init; }
        public decimal? BetfairSp { get; init; }
        public decimal? BetfairLastTradedPrice { get; init; }
        public SimulatorPriceSource PriceSource { get; init; }
        public double ClvPercent { get; init; }
        public double AiProbability { get; init; }
        public double MarketProbability { get; init; }
        public double Differential { get; init; }
        public double Edge { get; init; }
        public decimal Stake { get; init; }
        public bool Won { get; init; }
        public decimal Profit { get; init; }
        public decimal ExpectedProfit { get; init; }
        public decimal BankrollAfter { get; init; }
    }

    public sealed class SimulatorEquityPoint
    {
        public int BetNumber { get; init; }
        public DateTime? RaceDate { get; init; }
        public decimal Bankroll { get; init; }
        public decimal Profit { get; init; }
    }

    public sealed class SimulatorBreakdownRow
    {
        public string Group { get; init; } = "Unknown";
        public int Bets { get; init; }
        public int Wins { get; init; }
        public decimal Staked { get; init; }
        public decimal Profit { get; init; }
        public decimal ExpectedProfit { get; init; }
        public decimal AverageStake { get; init; }
        public double AverageEdge { get; init; }
        public double AverageOdds { get; init; }
        public decimal GrossWinnings { get; init; }
        public decimal GrossLosses { get; init; }
        public decimal Roi => Staked <= 0 ? 0 : Profit / Staked;
        public double StrikeRate => Bets <= 0 ? 0 : (double)Wins / Bets;
        public decimal? ProfitFactor => GrossLosses > 0
            ? GrossWinnings / GrossLosses
            : GrossWinnings > 0 ? null : 0m;
    }

    public sealed class SimulatorBreakdownViewModel
    {
        public string Title { get; init; } = string.Empty;
        public IReadOnlyList<SimulatorBreakdownRow> Rows { get; init; } = Array.Empty<SimulatorBreakdownRow>();
    }

    /// <summary>
    /// Tune/confirm split. The tune side is the first part of the window; the
    /// confirmation side must not be used to choose filters.
    /// </summary>
    public sealed class SimulatorComparison
    {
        public DateTime SplitDate { get; init; }
        public decimal ConfirmPercent { get; init; }
        public BettingSimulationResult Tune { get; init; } = new();
        public BettingSimulationResult Confirm { get; init; } = new();
    }

    public sealed class SimulatorPeriodResult
    {
        public string Label { get; init; } = string.Empty;
        public DateTime? Start { get; init; }
        public DateTime? End { get; init; }
        public int Bets { get; init; }
        public decimal Staked { get; init; }
        public decimal Profit { get; init; }
        public decimal ExpectedProfit { get; init; }
        public decimal MaxDrawdown { get; init; }
        public double Roi { get; init; }
        public double StrikeRate { get; init; }
        public decimal? ProfitFactor { get; init; }
    }

    public sealed class SimulatorBaselineRow
    {
        public string Name { get; init; } = string.Empty;
        public string? Note { get; init; }
        public int Bets { get; init; }
        public decimal Staked { get; init; }
        public decimal Profit { get; init; }
        public double Roi { get; init; }
        public double StrikeRate { get; init; }
        public decimal? ProfitFactor { get; init; }
    }

    public sealed class SimulatorSnapshotIndexEntry
    {
        public string Key { get; set; } = string.Empty;
        public string FileName { get; set; } = string.Empty;
        public DateTime UpdatedUtc { get; set; }
        public BettingSimulationSnapshotSummary Summary { get; set; } = new();
    }
}
