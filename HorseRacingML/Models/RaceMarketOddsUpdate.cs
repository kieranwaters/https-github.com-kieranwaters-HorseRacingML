using System.Collections.Generic;

namespace HorseRacingML.Models
{
    public class RaceMarketOddsUpdate
    {
        public string? MarketId { get; set; }
        public decimal? BackBookPercentage { get; set; }
        public decimal? LayBookPercentage { get; set; }
        public List<RunnerMarketOddsUpdate> Runners { get; set; } = new();
        public string? Going { get; set; }
    }

    public class RunnerMarketOddsUpdate
    {
        public string? SelectionId { get; set; }
        public string? HorseName { get; set; }
        public decimal? MarketDecimalOdds { get; set; }
        public decimal? LayDecimalOdds { get; set; }
        public double? MarketProbability { get; set; }
    }
}
