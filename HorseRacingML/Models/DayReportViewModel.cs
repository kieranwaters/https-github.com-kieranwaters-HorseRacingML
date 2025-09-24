using System;
using System.Collections.Generic;

namespace HorseRacingML.Models
{
    public class DayReportViewModel
    {
        public DateTime GeneratedAt { get; set; }
        public decimal Bankroll { get; set; }
        public List<RaceDayReport> Races { get; set; } = new();
    }

    public class RaceDayReport
    {
        public string MarketId { get; set; } = string.Empty;
        public string? RaceTitle { get; set; }
        public string? VenueName { get; set; }
        public string? VenueCountry { get; set; }
        public DateTime? RaceDate { get; set; }
        public TimeSpan? OffTime { get; set; }
        public string? RaceDetails { get; set; }
        public decimal? BackBookPercentage { get; set; }
        public decimal? LayBookPercentage { get; set; }
        public List<RunnerDayReport> Runners { get; set; } = new();
    }

    public class RunnerDayReport
    {
        public string? SelectionId { get; set; }
        public byte? ClothNumber { get; set; }
        public byte? Draw { get; set; }
        public string? HorseName { get; set; }
        public string? JockeyName { get; set; }
        public decimal? MarketDecimalOdds { get; set; }
        public decimal? AiDecimalOdds { get; set; }
        public double? AiProbability { get; set; }
        public double? MarketProbability { get; set; }
        public double? Differential { get; set; }
        public decimal? KellyFraction { get; set; }
        public decimal? SuggestedStake { get; set; }
        public Dictionary<string, object?> FeatureValues { get; set; } = new(StringComparer.OrdinalIgnoreCase);
    }
}