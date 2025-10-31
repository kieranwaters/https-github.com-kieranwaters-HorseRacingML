namespace HorseRacingML.Data
{
    public sealed class RaceFeatureBackfill
    {
        public int RaceId { get; set; }
        public string? RaceType { get; set; }
        public string? Surface { get; set; }
        public string? Going { get; set; }
        public int? DistanceYards { get; set; }
        public string? DistanceText { get; set; }
        public int? RunnerCount { get; set; }
    }
}