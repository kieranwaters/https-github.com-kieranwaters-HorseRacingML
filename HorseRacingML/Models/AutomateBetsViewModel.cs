using HorseRacingML.Services;

namespace HorseRacingML.Models
{
    public class AutomateBetsViewModel
    {
        public decimal KellyDampener { get; set; }
        public decimal? MaxKellyFraction { get; set; }
        public MaxStakeMode MaxStakeMode { get; set; }
        public decimal? MaxStakePercent { get; set; }
        public decimal? MaxStakeAmount { get; set; }
        public string? StatusMessage { get; set; }
        public string? BannerMessage { get; set; }
    }
}