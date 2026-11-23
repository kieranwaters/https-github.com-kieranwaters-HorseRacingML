using System;
using System.Collections.Generic;

namespace HorseRacingML.Models
{
    /// <summary>
    /// One Betfair market reconstructed from a historical stream file. The
    /// importer accumulates runner changes line by line and emits this object
    /// once the file has been read.
    /// </summary>
    public sealed class BetfairMarketImport
    {
        public string MarketId { get; set; } = string.Empty;
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
        public decimal? TotalMatched { get; set; }
        public List<BetfairRunnerImport> Runners { get; } = new();
    }

    /// <summary>
    /// Pre-off and reconciled values for one Betfair selection.
    /// </summary>
    public sealed class BetfairRunnerImport
    {
        public long SelectionId { get; set; }
        public string? Name { get; set; }
        public string? Status { get; set; }
        public int? SortPriority { get; set; }
        public decimal? Bsp { get; set; }
        public decimal? LastTradedPrice { get; set; }
        public decimal? PreOffHigh { get; set; }
        public decimal? PreOffLow { get; set; }
        public decimal? TradedVolume { get; set; }
        public decimal? ProjectedSp { get; set; }
        public decimal? AvailableToBack { get; set; }
        public decimal? AvailableToLay { get; set; }
    }

    public sealed record BetfairImportSummary(
        int FilesRead,
        int MarketsRead,
        int MarketsMatched,
        int RunnersMatched,
        int MarketsSkipped,
        int LinesRead);
}
