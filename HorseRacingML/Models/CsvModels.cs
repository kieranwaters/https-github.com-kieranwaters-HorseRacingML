using System;

namespace HorseRacingML.Models
{
    public class RaceCsvModel
    {
        public int Rid { get; set; }
        public string? Course { get; set; }
        public string? CountryCode { get; set; }
        public DateTime Date { get; set; }
        public string? Time { get; set; }
        public string? Title { get; set; }
        public string? Hurdles { get; set; }
        public int? Class { get; set; }
        public string? RClass { get; set; }
        public string? Ages { get; set; }
        public string? Condition { get; set; }
        public double? Metric { get; set; }
        public string? Distance { get; set; }
        public double? WinningTime { get; set; }
        public decimal? Prize { get; set; }
    }

    public class RunnerCsvModel
    {
        public int Rid { get; set; }
        public string? HorseName { get; set; }
        public string? TrainerName { get; set; }
        public string? JockeyName { get; set; }
        public decimal? DecimalPrice { get; set; }
        public byte? Saddle { get; set; }
        public byte? Age { get; set; }
        public int Position { get; set; }
        public decimal? Dist { get; set; }
        public int? IsFav { get; set; }
        public int WeightSt { get; set; }
        public int WeightLb { get; set; }
        public int? ResPlace { get; set; }
    }
}
