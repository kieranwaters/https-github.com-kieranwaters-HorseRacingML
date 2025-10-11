using System;

namespace HorseRacingML.Models
{
    public class DayReportFilterViewModel
    {
        public string? StartTime { get; set; }
        public string? EndTime { get; set; }
        public string? ErrorMessage { get; set; }

        public string? BuildSummary()
        {
            if (string.IsNullOrWhiteSpace(StartTime) && string.IsNullOrWhiteSpace(EndTime))
            {
                return null;
            }

            if (!string.IsNullOrWhiteSpace(StartTime) && !string.IsNullOrWhiteSpace(EndTime))
            {
                return $"Races between {StartTime} and {EndTime}";
            }

            return !string.IsNullOrWhiteSpace(StartTime)
                ? $"Races from {StartTime} onwards"
                : $"Races until {EndTime}";
        }
    }
}