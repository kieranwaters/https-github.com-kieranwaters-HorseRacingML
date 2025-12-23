using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace HorseRacingML.Models
{
    public class ValidationSimulationResult
    {
        public decimal StartingBankroll { get; init; }
        public decimal EndingBankroll { get; init; }
        public decimal TotalStaked { get; init; }
        public int BetCount { get; init; }
        public int WinCount { get; init; }
        public IReadOnlyList<ValidationBetResult> Bets { get; init; } = Array.Empty<ValidationBetResult>();

        public int LossCount => BetCount - WinCount;
        public decimal Profit => EndingBankroll - StartingBankroll;
        public decimal Roi => StartingBankroll <= 0 ? 0 : Profit / StartingBankroll;
        public decimal AverageRoiPerRace { get; init; }
    }

    public class ValidationBetResult
    {
        public int RaceId { get; init; }
        public DateTime? RaceDate { get; init; }
        public string? RaceTitle { get; init; }
        public string? CourseName { get; init; }
        public string? HorseName { get; init; }
        public decimal DecimalOdds { get; init; }
        public decimal AiDecimalOdds { get; init; }
        public double AiProbability { get; init; }
        public double MarketProbability { get; init; }
        public double Differential { get; init; }
        public decimal Stake { get; init; }
        public bool Won { get; init; }
        public decimal Bankroll { get; init; }
        public decimal Profit => Won ? Stake * (DecimalOdds - 1m) : -Stake;
    }
}