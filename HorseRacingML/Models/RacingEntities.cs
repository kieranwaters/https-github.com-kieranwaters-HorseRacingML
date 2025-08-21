using System;

namespace HorseRacingML.Models
{
    // Domain models that map to the SQL schema
    public class Course
    {
        public int CourseId { get; set; }
        public string Name { get; set; } = string.Empty;
        public string? Country { get; set; }
    }

    public class Race
    {
        public int RaceId { get; set; }
        public int CourseId { get; set; }
        public DateTime RaceDate { get; set; }
        public TimeSpan ScheduledOff { get; set; }
        public TimeSpan? ActualOff { get; set; }
        public string Title { get; set; } = string.Empty;
        public string RaceType { get; set; } = string.Empty;
        public byte? Class { get; set; }
        public string? AgeRestriction { get; set; }
        public string? Surface { get; set; }
        public string? Going { get; set; }
        public int DistanceYards { get; set; }
        public string DistanceText { get; set; } = string.Empty;
        public byte? RunnerCount { get; set; }
        public string? Status { get; set; }
        public int? WinningTimeMs { get; set; }
        public string? WinningTimeText { get; set; }
    }

    public class Trainer
    {
        public int TrainerId { get; set; }
        public string Name { get; set; } = string.Empty;
    }

    public class Jockey
    {
        public int JockeyId { get; set; }
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
        public int? TrainerId { get; set; }
        public int? JockeyId { get; set; }
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
    }
}