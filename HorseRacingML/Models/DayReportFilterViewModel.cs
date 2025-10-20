using System;
using System.Collections.Generic;
using System.Linq;

namespace HorseRacingML.Models
{
    public class DayReportFilterViewModel
    {
        public const string DefaultRegion = "GB & IRE";

        public static IReadOnlyList<string> AvailableRegions { get; } = new[]
        {
            "GB & IRE",
            "USA",
            "RSA",
            "FRA",
            "AUS",
            "All"
        };

        public string? StartTime { get; set; }
        public string? EndTime { get; set; }
        public string? ErrorMessage { get; set; }
        public string Region { get; set; } = DefaultRegion;
        public static string NormalizeRegion(string? region)
        {
            if (string.IsNullOrWhiteSpace(region))
            {
                return DefaultRegion;
            }

            var trimmed = region.Trim();
            var match = AvailableRegions.FirstOrDefault(option =>
                string.Equals(option, trimmed, StringComparison.OrdinalIgnoreCase));

            return match ?? DefaultRegion;
        }
    }
}