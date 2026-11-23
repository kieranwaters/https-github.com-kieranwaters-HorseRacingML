using System;

namespace HorseRacingML.Models
{
    /// <summary>
    /// Sire/dam/damsire and the small amount of stable profile data published by
    /// Racing TV. The raw names are not model features; they key the rolling
    /// progeny statistics used by the Breeding feature block.
    /// </summary>
    public sealed class HorsePedigree
    {
        public int HorseId { get; set; }
        public string? SireName { get; set; }
        public string? DamName { get; set; }
        public string? DamSireName { get; set; }
        public string? Sex { get; set; }
        public string? Colour { get; set; }
        public DateTime? FoalingDate { get; set; }
        public int? SourceHorseId { get; set; }
        public string? SourceSlug { get; set; }
        public string Source { get; set; } = "RacingTV";
        public DateTime FetchedUtc { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// One summary row per runner and race. The raw furlong splits are kept as
    /// JSON for audit and future re-derivation, while the columns used by the
    /// feature pass are already aggregated and leak-free.
    /// </summary>
    public sealed class RunnerSectional
    {
        public int RaceId { get; set; }
        public int HorseId { get; set; }
        public short? FinishPos { get; set; }
        public byte? ClothNumber { get; set; }
        public byte? SectionCount { get; set; }
        public float? Last1FSeconds { get; set; }
        public float? Last2FSeconds { get; set; }
        public float? Last3FSeconds { get; set; }
        public float? Last1FPercentile { get; set; }
        public float? Last2FPercentile { get; set; }
        public float? Last3FPercentile { get; set; }
        public float? EarlyPositionNorm { get; set; }
        public float? ClosingPositionNorm { get; set; }
        public float? PaceStyle { get; set; }
        public float? FinishSpeedPct { get; set; }
        public float? TimeIndexScore { get; set; }
        public float? MeetingAverageTimeIndex { get; set; }
        public float? WinTimeVsPar { get; set; }
        public string? SourceRaceId { get; set; }
        public string? RawJson { get; set; }
        public string Source { get; set; } = "RacingTV";
        public DateTime FetchedUtc { get; set; } = DateTime.UtcNow;
    }

    public sealed class RaceBackfillCandidate
    {
        public int RaceId { get; set; }
        public int CourseId { get; set; }
        public string CourseName { get; set; } = string.Empty;
        public TimeSpan? ScheduledOff { get; set; }
        public string? Title { get; set; }
        public byte? RunnerCount { get; set; }
    }

    public sealed class RaceRunnerReference
    {
        public int RaceId { get; set; }
        public int HorseId { get; set; }
        public string HorseName { get; set; } = string.Empty;
        public byte? SaddleclothNumber { get; set; }
        public short? FinishPos { get; set; }
    }

    public sealed class RacingTvBackfillSummary
    {
        public int DaysProcessed { get; set; }
        public int DaysSkipped { get; set; }
        public int RacesMatched { get; set; }
        public int SectionalRacesUpserted { get; set; }
        public int SectionalRowsUpserted { get; set; }
        public int PedigreeRowsUpserted { get; set; }
        public int RacesWithoutSourceData { get; set; }
        public int Warnings { get; set; }
    }
}
