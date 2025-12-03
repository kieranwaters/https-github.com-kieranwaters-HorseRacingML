using System;
using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.Configuration.Attributes;
using CsvHelper.TypeConversion;

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
        [TypeConverter(typeof(RacingDistanceConverter))]
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

    public class RacingDistanceConverter : DefaultTypeConverter
    {
        public override object? ConvertFromString(string? text, IReaderRow row, MemberMapData memberMapData)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var cleanText = text.Trim().ToLowerInvariant();

            if (decimal.TryParse(cleanText, out var decimalValue))
            {
                return decimalValue;
            }

            switch (cleanText)
            {
                case "nse":
                case "nose":
                    return 0.05m;
                case "shd":
                case "sht-hd":
                case "short head":
                    return 0.1m;
                case "hd":
                case "head":
                    return 0.2m;
                case "nk":
                case "neck":
                    return 0.3m;
                case "dist":
                case "ds":
                    return 30.0m;
                default:
                    if (cleanText == "½") return 0.5m;
                    if (cleanText == "¼") return 0.25m;
                    if (cleanText == "¾") return 0.75m;
                    return null;
            }
        }
    }
}
