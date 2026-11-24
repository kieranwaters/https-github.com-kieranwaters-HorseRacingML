using System;
using System.ComponentModel.DataAnnotations;

namespace HorseRacingML.Models
{
    public class MLParameter
    {
        public int Id { get; set; }
        public DateTime RunDate { get; set; }
        public DateTime? EarliestTrainingDate { get; set; }
        [Required]
        public int Units { get; set; }
        [Required]
        public double Dropout { get; set; }
        [Required]
        public int Layers { get; set; }
        [Required]
        public double LearningRate { get; set; }
        [Required]
        public int Epochs { get; set; }
        [Required]
        public int BatchSize { get; set; }
        [Required]
        public int Folds { get; set; }
        public double? TrainAccuracy { get; set; }
        public double? ValidationAccuracy { get; set; }
        public double? ValidationLoss { get; set; }
        public double? TrainLoss { get; set; }
        public double? TrainBrier { get; set; }
        public int? Fold { get; set; }
        public double? ValidationBrier { get; set; }

        // Evaluation split and objective metadata. Validation* remains the
        // dev/selection metric for backwards compatibility; Test* is the
        // untouched chronological holdout and Dev* mirrors Validation*.
        public string? Objective { get; set; }
        public int? Seed { get; set; }
        public int? BestEpoch { get; set; }
        public double? MarketOffsetScale { get; set; }
        public string? MarketOffsetMode { get; set; }
        public bool? MarketBlind { get; set; }
        public double? BlendModelWeight { get; set; }
        public double? BlendMarketWeight { get; set; }
        public string? NormalizationMode { get; set; }
        public string? FeatureBlocks { get; set; }
        public int? RaceBatchSize { get; set; }
        public DateTime? TrainingStartDate { get; set; }
        public DateTime? TrainingEndDate { get; set; }
        public DateTime? DevStartDate { get; set; }
        public DateTime? DevEndDate { get; set; }
        public DateTime? TestStartDate { get; set; }
        public DateTime? TestEndDate { get; set; }
        public int? TrainingRaceCount { get; set; }
        public int? DevRaceCount { get; set; }
        public int? TestRaceCount { get; set; }
        public int? TrainingRowCount { get; set; }
        public int? DevRowCount { get; set; }
        public int? TestRowCount { get; set; }
        public double? DevAccuracy { get; set; }
        public double? DevLoss { get; set; }
        public double? DevBrier { get; set; }
        public double? DevRaceLogLoss { get; set; }
        public double? DevTop3Accuracy { get; set; }
        public double? DevMarketRaceLogLoss { get; set; }
        public double? DevMarketTop1Accuracy { get; set; }
        public double? DevMarketTop3Accuracy { get; set; }
        public double? DevTemperature { get; set; }
        // DevMarket* uses the closing SP (with commission) and is kept for
        // comparison with older runs. DevFairMarket* uses the same no-vig
        // market probabilities the model was trained on. DevBlend* is the
        // two-parameter model/market blend fitted on dev.
        public double? DevMarketLogLoss { get; set; }
        public double? DevMarketBrier { get; set; }
        public double? DevFairMarketLogLoss { get; set; }
        public double? DevFairMarketBrier { get; set; }
        public double? DevBlendLogLoss { get; set; }
        public double? DevBlendBrier { get; set; }
        public double? DevBlendRaceLogLoss { get; set; }
        public double? TestAccuracy { get; set; }
        public double? TestLoss { get; set; }
        public double? TestBrier { get; set; }
        public double? TestRaceLogLoss { get; set; }
        public double? TestTop3Accuracy { get; set; }
        public double? TestMarketRaceLogLoss { get; set; }
        public double? TestMarketTop1Accuracy { get; set; }
        public double? TestMarketTop3Accuracy { get; set; }
        public double? TestTemperature { get; set; }
        public double? TestMarketLogLoss { get; set; }
        public double? TestMarketBrier { get; set; }
        public double? TestFairMarketLogLoss { get; set; }
        public double? TestFairMarketBrier { get; set; }
        public double? TestBlendLogLoss { get; set; }
        public double? TestBlendBrier { get; set; }
        public double? TestBlendRaceLogLoss { get; set; }
        public double? WinRateAlpha { get; set; }
        public double? WinRateBeta { get; set; }
        public string? SelectedOn { get; set; }

        // Versioned model artifact metadata. ModelArtifact is a file name
        // relative to Training:ModelDirectory; the hash proves which bytes were
        // active and FeatureSchemaHash proves which feature contract was used.
        public string? ModelArtifact { get; set; }
        public string? ModelHash { get; set; }
        public string? FeatureSchemaHash { get; set; }
        public int? EnsembleSize { get; set; }
        public bool? IsActive { get; set; }
    }
}