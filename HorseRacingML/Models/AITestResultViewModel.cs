using System;
using System.ComponentModel.DataAnnotations;
using System.Collections.Generic;

namespace HorseRacingML.Models
{
    public class RaceResultViewModel
    {
        public int RaceId { get; set; }
        public string RaceTitle { get; set; }
        public string PredictedWinner { get; set; }
        public string ActualWinner { get; set; }
        public bool IsCorrectPrediction { get; set; }
        public decimal Bankroll { get; set; }
        public decimal Stake { get; set; }
        public decimal AiOdds { get; set; }
        public decimal BookmakerOdds { get; set; }
        public List<RunnerViewModel> Runners { get; set; }
    }

    public class RunnerViewModel
    {
        public string HorseName { get; set; }
        public int? HistoricalRaceCount { get; set; }
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
        public string SelectedCountry { get; set; }
        public List<string> Countries { get; set; }
        public double? TrainFocalLoss { get; set; }
        public double? ValidationFocalLoss { get; set; }
        [Range(1, 100, ErrorMessage = "Please choose a dampener between 1 and 100.")]
        public int KellyDampener { get; set; } = 10;
        public decimal StartingBankroll { get; set; } = 100m;
        public ValidationSimulationResult? Simulation { get; set; }
        public bool HasResult => ValidationAccuracy.HasValue || ValidationLoss.HasValue;
        public bool UseExistingWeights { get; set; }
        public List<DayResultViewModel> DailyResults { get; set; }
    }
}