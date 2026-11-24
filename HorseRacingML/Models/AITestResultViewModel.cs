using System;
using System.ComponentModel.DataAnnotations;

namespace HorseRacingML.Models
{
    public class AITestResultViewModel
    {
        public double? ValidationAccuracy { get; set; }
        public double? ValidationLoss { get; set; }
        public double? ValidationBrier { get; set; }
        public double? ValidationRaceLogLoss { get; set; }
        public double? ValidationTop3Accuracy { get; set; }
        public double? ValidationTemperature { get; set; }
        public double? MarketLogLoss { get; set; }
        public double? MarketBrier { get; set; }
        public double? MarketRaceLogLoss { get; set; }
        public double? MarketTop1Accuracy { get; set; }
        public double? MarketTop3Accuracy { get; set; }
        public double? FairMarketLogLoss { get; set; }
        public double? FairMarketBrier { get; set; }
        public double? FairMarketRaceLogLoss { get; set; }
        public double? BlendLogLoss { get; set; }
        public double? BlendBrier { get; set; }
        public double? BlendRaceLogLoss { get; set; }
        public double? BlendModelWeight { get; set; }
        public double? BlendMarketWeight { get; set; }
        public double? TrainAccuracy { get; set; }
        public double? TrainLoss { get; set; }
        public double? TrainBrier { get; set; }
        public int ValidationRaceCount { get; set; }
        public int TrainingRaceCount { get; set; }
        public DateTime ValidationStart { get; set; }
        public DateTime ValidationEnd { get; set; }
        public string? Message { get; set; }
        public MLParameter? ParameterUsed { get; set; }
        [Range(SimulatorRunRequest.MinimumMonths, SimulatorRunRequest.MaximumMonths, ErrorMessage = "Please choose between 1 and 24 months.")]
        public int SelectedValidationMonths { get; set; } = 1;
        public decimal StartingBankroll { get; set; } = 100m;
        public ValidationSimulationResult? Simulation { get; set; }
        public bool HasResult => ValidationAccuracy.HasValue;
    }
}