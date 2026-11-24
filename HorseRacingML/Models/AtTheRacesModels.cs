using System;
using System.Collections.Generic;

namespace HorseRacingML.Models
{
    /// <summary>
    /// A race link discovered on an At The Races results page. The link time is
    /// local track time, matching the value in the public racecard URL.
    /// </summary>
    public sealed class AtTheRacesRaceLink
    {
        public long RaceId { get; set; }
        public string Course { get; set; } = string.Empty;
        public DateTime LocalDate { get; set; }
        public TimeSpan LocalTime { get; set; }
        public string PagePath { get; set; } = string.Empty;
        public string? Title { get; set; }
    }

    public sealed class AtTheRacesSplit
    {
        public int SectorNumber { get; set; }
        public string? SectorName { get; set; }
        public double? TimeSeconds { get; set; }
        public double? CumulativeSeconds { get; set; }
        public string? Position { get; set; }
        public string? PaceCategory { get; set; }
    }

    public sealed class AtTheRacesRunnerSectional
    {
        public string HorseName { get; set; } = string.Empty;
        public int? ClothNumber { get; set; }
        public short? FinishPos { get; set; }
        public double? FinishTimeSeconds { get; set; }
        public List<AtTheRacesSplit> Splits { get; set; } = new();
    }

    public sealed class AtTheRacesBackfillSummary
    {
        public int DaysExamined { get; set; }
        public int DaysProcessed { get; set; }
        public int RacesMatched { get; set; }
        public int RacesWithSourceData { get; set; }
        public int RacesWithoutSourceData { get; set; }
        public int RowsInserted { get; set; }
        public int RowsAlreadyPresent { get; set; }
        public int Warnings { get; set; }
    }
}
