
using System;
using System.Collections.Generic;

namespace HorseRacingML.ML
{
    public class LayerWeights
    {
        public float[][] Weights { get; set; } = Array.Empty<float[]>();
        public float[] Bias { get; set; } = Array.Empty<float>();
    }

    public class FeatureMetadata
    {
        public List<string> Keys { get; set; } = new();
        public Dictionary<string, int> FeatureDimensions { get; set; } = new();
        public Dictionary<string, Dictionary<string, int>> StringMaps { get; set; } = new();
    }

    public class TrainedModel
    {
        public List<LayerWeights> HiddenLayers { get; set; } = new();
        public LayerWeights OutputLayer { get; set; } = new LayerWeights();
        public FeatureMetadata Metadata { get; set; } = new FeatureMetadata();
        public NormalizationParameters Normalization { get; set; } = new NormalizationParameters();
    }
}