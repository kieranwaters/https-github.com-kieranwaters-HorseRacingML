using System;
using System.ComponentModel.DataAnnotations;

namespace HorseRacingML.Models
{
    public class MLParameter
    {
        public int Id { get; set; }
        public DateTime RunDate { get; set; }
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
        public bool EnableFeatureCorrelations { get; set; } = true;
        public double? TrainAccuracy { get; set; }
        public double? ValidationAccuracy { get; set; }
        public double? ValidationLoss { get; set; }
        public double? TrainLoss { get; set; }
        public double? TrainBrier { get; set; }
        public double? TrainFocalLoss { get; set; }
        public double? ValidationFocalLoss { get; set; }
        public int? Fold { get; set; }
        public double? ValidationBrier { get; set; }
        public bool TrainFinalFoldOnly { get; set; }
        /// <summary>
        /// 0 = Neural Network (TensorFlow), 1 = LightGBM
        /// </summary>
        public int ModelType { get; set; }
    }
}