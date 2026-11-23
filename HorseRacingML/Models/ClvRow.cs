using System;

namespace HorseRacingML.Models
{
    /// <summary>
    /// Closing-line value for a placed bet: the bet price taken versus the
    /// last pre-off RunnerFlow price observed for the same selection.
    /// Positive means the bet was taken at a better price than the close.
    /// </summary>
    public sealed class ClvRow
    {
        public string? MarketId { get; set; }
        public string? SelectionId { get; set; }
        public string? HorseName { get; set; }
        public decimal? BetOdds { get; set; }
        public decimal? ClosingOdds { get; set; }
        public double? Clv { get; set; }
        public DateTime? PlacedUtc { get; set; }
    }
}
