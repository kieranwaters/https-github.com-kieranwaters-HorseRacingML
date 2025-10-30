using System.Collections.Generic;

namespace HorseRacingML.Models
{
    public class TrainAIViewModel
    {
        public MLParameterBatch Batch { get; set; } = new();

        public List<TrainAIModelResult> Results { get; set; } = new();

        public class TrainAIModelResult
        {
            public MLParameter Parameters { get; set; } = new();

            public List<FeatureCorrelationInfo> FeatureCorrelations { get; set; } = new();
        }

        public class FeatureCorrelationInfo
        {
            public string FeatureKey { get; set; } = string.Empty;

            public string? Dimension { get; set; }

            public double Correlation { get; set; }
        }
    }
}