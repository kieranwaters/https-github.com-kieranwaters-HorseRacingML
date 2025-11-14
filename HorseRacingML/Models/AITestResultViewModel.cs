using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace HorseRacingML.Models
{
    public class RaceResultViewModel
    {
        public int RaceId { get; set; }
        public string RaceTitle { get; set; }
        public string PredictedWinner { get; set; }
        public string ActualWinner { get; set; }
        public bool IsCorrectPrediction { get; set; }
    }

    public class DayResultViewModel
    {
        public DateTime RaceDate { get; set; }
        public List<RaceResultViewModel> Races { get; set; }
    }

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
        public int? RequestedUnits { get; set; }
        public double? RequestedDropout { get; set; }
        public int? RequestedLayers { get; set; }
        public double? RequestedLearningRate { get; set; }
        public int? RequestedEpochs { get; set; }
        public int? RequestedBatchSize { get; set; }
        public int? RequestedFolds { get; set; }
        [Range(1, 24, ErrorMessage = "Please choose between 1 and 24 months.")]
        public int SelectedValidationMonths { get; set; } = 1;
        public double? TrainFocalLoss { get; set; }
        public double? ValidationFocalLoss { get; set; }
        public decimal StartingBankroll { get; set; } = 100m;
        public ValidationSimulationResult? Simulation { get; set; }
        public bool HasResult => ValidationAccuracy.HasValue || ValidationLoss.HasValue;
        public bool UseExistingWeights { get; set; }
        public List<DayResultViewModel> DailyResults { get; set; }
    }
}