using System;
using CsvHelper.Configuration.Attributes;

namespace HorseRacingML.Models
{
    public class RaceCsvModel
    {
        [Name("rid")]
        public int Rid { get; set; }
        [Name("course")]
        public string? Course { get; set; }
        [Name("countryCode")]
        public string? CountryCode { get; set; }
        [Name("date")]
        public DateTime Date { get; set; }
        [Name("time")]
        public string? Time { get; set; }
        [Name("title")]
        public string? Title { get; set; }
        [Name("hurdles")]
        public string? Hurdles { get; set; }
        [Name("class")]
        public int? Class { get; set; }
        [Name("rclass")]
        public string? RClass { get; set; }
        [Name("ages")]
        public string? Ages { get; set; }
        [Name("condition")]
        public string? Condition { get; set; }
        [Name("metric")]
        public double? Metric { get; set; }
        [Name("distance")]
        public string? Distance { get; set; }
        [Name("winningTime")]
        public double? WinningTime { get; set; }
        [Name("prize")]
        public decimal? Prize { get; set; }
    }

    public class RunnerCsvModel
    {
        [Name("rid")]
        public int Rid { get; set; }
        [Name("horseName")]
        public string? HorseName { get; set; }
        [Name("trainerName")]
        public string? TrainerName { get; set; }
        [Name("jockeyName")]
        public string? JockeyName { get; set; }
        [Name("decimalPrice")]
        public decimal? DecimalPrice { get; set; }
        [Name("saddle")]
        public double? Saddle { get; set; }
        [Name("age")]
        public double? Age { get; set; }
        [Name("position")]
        public int Position { get; set; }
        [Name("dist")]
        public decimal? Dist { get; set; }
        [Name("isFav")]
        public int? IsFav { get; set; }
        [Name("weightSt")]
        public int WeightSt { get; set; }
        [Name("weightLb")]
        public int WeightLb { get; set; }
        [Name("res_place")]
        public int? ResPlace { get; set; }
    }
}
