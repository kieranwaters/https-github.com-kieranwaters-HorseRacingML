using System;
using System.Collections.Generic;

namespace HorseRacingML.Models
{
    // Domain models that map to the SQL schema
    public class Course
    {
        public short CourseId { get; set; }
        public string Name { get; set; } = string.Empty;
        public string? Country { get; set; }
    }

    public class Race
    {
        public int RaceId { get; set; }
        public short CourseId { get; set; }
        public DateTime RaceDate { get; set; }
        public TimeSpan ScheduledOff { get; set; }
        public TimeSpan? ActualOff { get; set; }
        public string Title { get; set; } = string.Empty;
        public string RaceType { get; set; } = string.Empty;
        public byte? Class { get; set; }
        public string? AgeRestriction { get; set; }
        public string? Surface { get; set; }
        public string? Going { get; set; }
        public short DistanceYards { get; set; }
        public string DistanceText { get; set; } = string.Empty;
        public byte? RunnerCount { get; set; }
        public string? Status { get; set; }
        public int? WinningTimeMs { get; set; }
        public string? WinningTimeText { get; set; }
    }

    public class Trainer
    {
        public short TrainerId { get; set; }
        public string Name { get; set; } = string.Empty;
    }

    public class Jockey
    {
        public short JockeyId { get; set; }
        public string Name { get; set; } = string.Empty;
    }

    public class Horse
    {
        public int HorseId { get; set; }
        public string Name { get; set; } = string.Empty;
    }

    public class RunnerResult
    {
        public int RunnerResultId { get; set; }
        public int RaceId { get; set; }
        public int HorseId { get; set; }
        public short? TrainerId { get; set; }
        public short? JockeyId { get; set; }
        public byte? SaddleclothNumber { get; set; }
        public byte? Draw { get; set; }
        public byte? Age { get; set; }
        public byte? WeightLbs { get; set; }
        public string? WeightText { get; set; }
        public short? FinishPos { get; set; }
        public string? OutcomeCode { get; set; }
        public string? DistanceBeatenText { get; set; }
        public decimal? DistanceBeatenLengths { get; set; }
        public string? SP_Fraction { get; set; }
        public decimal? SP_Decimal { get; set; }
        public string? FavTag { get; set; }
        public string? OpeningFraction { get; set; }
        public string? TouchedHighFraction { get; set; }
        public string? TouchedLowFraction { get; set; }
        public string? Comment { get; set; }
        public string? Going { get; set; }
        public string? Surface { get; set; }
        public short? CourseId { get; set; }
        public short DistanceYards { get; set; }
    }
    public sealed class RaceClassRating
    {
        public DateTime RaceDate { get; set; }
        public short? OfficialRating { get; set; }
        public byte? Class { get; set; }
    }
    public class RaceScreen
    {
        public long RaceScreenId { get; set; }
        public string? MarketId { get; set; }
        public DateTime? RaceDate { get; set; }
        public TimeSpan? OffTime { get; set; }
        public string? Title { get; set; }
        public string? VenueName { get; set; }
        public string? VenueCountry { get; set; }
        public string? EventDateText { get; set; }
        public string? RaceDetails { get; set; }
        public string? RaceType { get; set; }
        public string? Going { get; set; }
        public decimal? BackBookPercentage { get; set; }
        public decimal? LayBookPercentage { get; set; }
        public string? RaceUrl { get; set; }
    }


    public class RunnerFlow
    {
        public long RunnerFlowId { get; set; }
        public string? MarketId { get; set; }
        public int? UpcomingRaceId { get; set; }
        public DateTime? RaceDate { get; set; }
        public TimeSpan? ScheduledOff { get; set; }
        public string? VenueName { get; set; }
        public string? VenueCountry { get; set; }
        public string? RaceTitle { get; set; }
        public string? RaceDetails { get; set; }
        public string? RaceType { get; set; }
        public string? Surface { get; set; }
        public string? Going { get; set; }
        public short? DistanceYards { get; set; }
        public string? DistanceText { get; set; }
        public byte? RunnerCount { get; set; }
        public decimal? BackBookPercentage { get; set; }
        public decimal? LayBookPercentage { get; set; }
        public byte? ClothNumber { get; set; }
        public byte? Draw { get; set; }
        public string? HorseName { get; set; }
        public string? JockeyName { get; set; }
        public decimal? BackPrice1 { get; set; }
        public decimal? BackPrice2 { get; set; }
        public decimal? BackPrice3 { get; set; }
        public decimal? LayPrice1 { get; set; }
        public decimal? LayPrice2 { get; set; }
        public decimal? LayPrice3 { get; set; }
        public double? AiOdds { get; set; }
        public double? LegacyProbability { get; set; }
        public Dictionary<string, object?>? FeatureValues { get; set; }
        public bool HasPreparedFeatures { get; set; }
        public List<EncodedFeatureValue>? EncodedFeatureValues { get; set; }
        public bool HasPartialPreparedFeatures { get; set; }
        public int? HistoricalRaceCount { get; set; }
        public bool AiProbabilityMarketDerived { get; set; }
        public bool AiProbabilityClampedToMarket { get; set; }
        public double? AiProbabilityClampTarget { get; set; }
        public string? AiProbabilityFallbackReason { get; set; }

        public byte? Age { get; set; }
        public byte? WeightLbs { get; set; }
        public string? WeightText { get; set; }
        public string? TrainerName { get; set; }
        public bool MatchedDatabaseRecord { get; set; }



        public FeaturePopulationSummary FeaturePopulationSummary { get; set; } = FeaturePopulationSummary.Empty;
        public bool AiUsedLegacyModel { get; internal set; }
        public bool AiTrainedModelApplied { get; internal set; }
    }

    public class UpcomingRace
    {
        public int UpcomingRaceId { get; set; }
        public string MarketId { get; set; } = string.Empty;
        public DateTime RaceDate { get; set; }
        public TimeSpan? ScheduledOff { get; set; }
        public string? VenueName { get; set; }
        public string? VenueCountry { get; set; }
        public string? Title { get; set; }
        public string? RaceDetails { get; set; }
        public string? RaceType { get; set; }
        public string? Surface { get; set; }
        public string? Going { get; set; }
        public short? DistanceYards { get; set; }
        public string? DistanceText { get; set; }
        public byte? RunnerCount { get; set; }
        public decimal? BackBookPercentage { get; set; }
        public decimal? LayBookPercentage { get; set; }
        public DateTime? CreatedUtc { get; set; }
        public DateTime? LastUpdatedUtc { get; set; }
        public bool AiTrainedModelApplied { get; set; }
        public bool AiUsedLegacyModel { get; set; }
    }

    public class RaceSummary
    {
        public int RaceId { get; set; }
        public DateTime? RaceDate { get; set; }
        public string? Title { get; set; }
        public string? CourseName { get; set; }
    }


}