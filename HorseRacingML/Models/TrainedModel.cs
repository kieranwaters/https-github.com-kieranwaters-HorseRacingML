
using System;
using System.Collections.Generic;

namespace HorseRacingML.ML
{
    public class LayerWeights
    {
        public float[][] Weights { get; set; } = Array.Empty<float[]>();
        public float[] Bias { get; set; } = Array.Empty<float>();
    }
    public class HyperparameterSummary
    {
        public int Layers { get; set; }
        public int Units { get; set; }
        public double Dropout { get; set; }
        public double LearningRate { get; set; }
        public int Epochs { get; set; }
        public int BatchSize { get; set; }
        public int Folds { get; set; }
        public int? Fold { get; set; }
        public DateTime? TrainedAtUtc { get; set; }
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
        public HyperparameterSummary? Hyperparameters { get; set; }
    }
}