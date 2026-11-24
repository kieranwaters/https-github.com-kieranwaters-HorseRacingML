
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
        /// <summary>
        /// Temperature used to calibrate the output sigmoid. Values below 1 make
        /// the model more confident, values above 1 make it less confident.
        /// </summary>
        public float Temperature { get; set; } = 1f;
        /// <summary>
        /// "Binary" for the legacy pointwise sigmoid objective, or
        /// "RaceSoftmax" for the conditional-logit objective.
        /// </summary>
        public string Objective { get; set; } = "Binary";
        /// <summary>
        /// Weight of the auxiliary pointwise win head used while training a
        /// RaceSoftmax model. The head is a training-only regulariser; it is not
        /// part of the persisted inference graph.
        /// </summary>
        public float AuxiliaryWinLossWeight { get; set; }
        /// <summary>
        /// Weight of the auxiliary finishing-position head. Like the win head it
        /// only shapes the shared trunk during training and is not evaluated at
        /// inference time.
        /// </summary>
        public float AuxiliaryFinishLossWeight { get; set; }
        /// <summary>
        /// Weight of the training-only winner-versus-field pairwise ranking
        /// regulariser. It is not part of the inference graph.
        /// </summary>
        public float PairwiseLossWeight { get; set; }
        /// <summary>
        /// Additive market offset scale used by RaceSoftmax models. The offset
        /// itself is selected by <see cref="MarketOffsetMode"/>.
        /// </summary>
        public float MarketOffsetScale { get; set; }
        /// <summary>
        /// "Logit" for the legacy log-odds offset, or "LogProb" for the
        /// race-softmax market prior (centred log market probabilities).
        /// Old weight files default to "Logit".
        /// </summary>
        public string MarketOffsetMode { get; set; } = "Logit";
        /// <summary>
        /// True when the network was trained without market features.
        /// </summary>
        public bool MarketBlind { get; set; }
        /// <summary>
        /// Weight on the model log-probabilities in the final two-parameter
        /// market blend. 1 with a market weight of 0 means "no blend".
        /// </summary>
        public float BlendModelWeight { get; set; } = 1f;
        /// <summary>
        /// Weight on the market log-probabilities in the final blend.
        /// </summary>
        public float BlendMarketWeight { get; set; }
        public string NormalizationMode { get; set; } = "Raw";
        public FeatureMetadata Metadata { get; set; } = new FeatureMetadata();
        public NormalizationParameters Normalization { get; set; } = new NormalizationParameters();

        /// <summary>
        /// Timestamp written when the artifact is persisted. Older files may
        /// contain the default value and are treated as legacy artifacts.
        /// </summary>
        public DateTime TrainedUtc { get; set; }

        /// <summary>
        /// Hash of the feature keys, dimensions and category maps. It lets a
        /// training run and an evaluation run prove they use the same schema.
        /// </summary>
        public string FeatureSchemaHash { get; set; } = string.Empty;

        /// <summary>
        /// Artifact format version. Increased when the inference contract changes.
        /// </summary>
        public string ModelVersion { get; set; } = "2";

        /// <summary>
        /// Optional seed ensemble. When this list is non-empty the top-level
        /// weights are ignored for scoring and each member contributes one
        /// probability vector, averaged in log space within the race. The
        /// members share the feature schema and normalisation; they differ only
        /// in their weights, market offset scale and temperature.
        /// </summary>
        public List<TrainedModel> Members { get; set; } = new();

        public bool IsEnsemble => Members is { Count: > 0 };

        /// <summary>
        /// Optional ML.NET LightGBM ranking model stored next to this JSON
        /// artifact. When set, the member is scored by LightGBM (scores are
        /// softmaxed within the race) instead of the TensorFlow trunk.
        /// </summary>
        public string? LightGbmModelFile { get; set; }

        /// <summary>
        /// Optional deployable field-size-bucket Platt calibration. The final
        /// per-runner probabilities are calibrated after the model/ensemble
        /// combination and renormalised within the race. Buckets are ordered by
        /// maximum field size; the last bucket should normally be int.MaxValue.
        /// </summary>
        public List<FieldCalibrationBucket> FieldCalibration { get; set; } = new();
    }

    public class FieldCalibrationBucket
    {
        public int MaxFieldSize { get; set; } = int.MaxValue;
        public double Intercept { get; set; }
        public double Slope { get; set; } = 1d;
    }
}