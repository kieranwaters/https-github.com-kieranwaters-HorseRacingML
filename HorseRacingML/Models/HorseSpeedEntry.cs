using System;

namespace HorseRacingML.Models
{
    /// <summary>
    /// Represents the historical speed-relevant data for a single runner result.
    /// </summary>
    public sealed class HorseSpeedEntry
    {
        public DateTime RaceDate { get; set; }
        public int? DistanceYards { get; set; }
        public int? WinningTimeMilliseconds { get; set; }
        public decimal? DistanceBeatenLengths { get; set; }
    }
}