using System.Collections.Generic;

namespace HorseRacingML.Models
{
    public class DayReportStatus
    {
        public string Message { get; set; } = string.Empty;
        public int TotalRaces { get; set; }
        public int ProcessedRaces { get; set; }
        public bool IsComplete { get; set; }
        public List<SimpleRaceResult> Results { get; set; } = new List<SimpleRaceResult>();
    }

    public class SimpleRaceResult
    {
        public string VenueName { get; set; } = string.Empty;
        public string RaceTime { get; set; } = string.Empty;
        public string RaceTitle { get; set; } = string.Empty;
        public string RaceUrl { get; set; } = string.Empty;
        public List<SimpleRunnerResult> Runners { get; set; } = new List<SimpleRunnerResult>();
    }

    public class SimpleRunnerResult
    {
        public string? ClothNumber { get; set; }
        public string HorseName { get; set; } = string.Empty;
        public decimal? BackPrice { get; set; }
        public double AiProbability { get; set; }
        public double Edge { get; set; }
        public decimal Stake { get; set; }
        public bool IsBet { get; set; }
    }
}
