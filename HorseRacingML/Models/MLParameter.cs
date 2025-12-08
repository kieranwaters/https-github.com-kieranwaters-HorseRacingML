using System;
using System.ComponentModel.DataAnnotations;

namespace HorseRacingML.Models
{
    public class MLParameter
    {
        public int Id { get; set; }
        public DateTime RunDate { get; set; }
        public int? Units { get; set; }
        public double? Dropout { get; set; }
        public int? Layers { get; set; }
        public double LearningRate { get; set; }
        public int Epochs { get; set; }
        public int? BatchSize { get; set; }
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

        // LightGBM specific parameters
        public int? LgbmLeaves { get; set; }
        public int? LgbmMinDataInLeaf { get; set; }
        public int? LgbmMaxDepth { get; set; }
        public int? Threads { get; set; }
    }
}