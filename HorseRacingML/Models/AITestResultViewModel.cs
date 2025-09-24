using System;

namespace HorseRacingML.Models
{
    public class AITestResultViewModel
    {
        public double? ValidationAccuracy { get; set; }
        public double? ValidationLoss { get; set; }
        public double? ValidationBrier { get; set; }
        public double? TrainAccuracy { get; set; }
        public double? TrainLoss { get; set; }
        public double? TrainBrier { get; set; }
        public int ValidationRaceCount { get; set; }
        public int TrainingRaceCount { get; set; }
        public DateTime ValidationStart { get; set; }
        public DateTime ValidationEnd { get; set; }
        public string? Message { get; set; }
        public MLParameter? ParameterUsed { get; set; }
        public bool HasResult => ValidationAccuracy.HasValue;
    }
}