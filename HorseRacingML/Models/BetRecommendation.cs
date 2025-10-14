using System;

namespace HorseRacingML.Models
{
    /// <summary>
    /// Represents a recommended bet derived from the AI model and the
    /// available market prices.  The recommendation stores the bet sizing
    /// information calculated via the Kelly criterion as well as the
    /// differential between the model probability and the market odds.
    /// </summary>
    public record BetRecommendation
    {
        public string MarketId { get; init; } = string.Empty;
        public string? SelectionId { get; init; }
        public string? HorseName { get; init; }
        public string? RunnerKey { get; init; }
        public string? RaceTitle { get; init; }
        public string? VenueName { get; init; }
        public DateTime? RaceDate { get; init; }
        public decimal DecimalOdds { get; init; }
        public decimal AiDecimalOdds { get; init; }
        public double AiProbability { get; init; }
        public double MarketProbability { get; init; }
        public double Differential { get; init; }
        public decimal KellyFraction { get; init; }
        public decimal Stake { get; init; }
    }
}
