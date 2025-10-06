using System;
using System.Collections.Generic;
using HorseRacingML.ML;

namespace HorseRacingML.Models
{
    public class DayReportViewModel
    {
        public DateTime GeneratedAt { get; set; }
        public decimal Bankroll { get; set; }
        public HyperparameterSummary? AiHyperparameters { get; set; }
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
        public string? RaceType { get; set; }
        public string? Going { get; set; }
        public decimal? BackBookPercentage { get; set; }
        public decimal? LayBookPercentage { get; set; }
        public List<RunnerDayReport> Runners { get; set; } = new();
        public int? UpcomingRaceId { get; set; }
        public string? RaceUrl { get; set; }
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
        public decimal? LayDecimalOdds { get; set; }
        public int? HistoricalRaceCount { get; set; }
        public decimal? LayKellyFraction { get; set; }
        public decimal? LaySuggestedStake { get; set; }
        public Dictionary<string, object?> FeatureValues { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public bool AiProbabilityMarketDerived { get; set; }
    }
}