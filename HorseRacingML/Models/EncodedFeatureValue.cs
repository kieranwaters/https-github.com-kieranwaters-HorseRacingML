namespace HorseRacingML.Models
{
    public class EncodedFeatureValue
    {
        public int Index { get; set; }
        public string FeatureKey { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
        public double Value { get; set; }
        public bool Active { get; set; }
    }
}