using System;
using System.Collections.Generic;
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
        public decimal Roi => TotalStaked <= 0 ? 0 : Profit / TotalStaked;

        public IReadOnlyList<ValidationBreakdownRow> ByRaceType =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.RaceType) ? "Unknown" : bet.RaceType!);

        public IReadOnlyList<ValidationBreakdownRow> ByFieldSize =>
            BuildBreakdown(bet => bet.RunnerCount switch
            {
                null => "Unknown",
                <= 5 => "1-5 runners",
                <= 8 => "6-8 runners",
                <= 12 => "9-12 runners",
                <= 16 => "13-16 runners",
                _ => "17+ runners"
            });

        public IReadOnlyList<ValidationBreakdownRow> ByClass =>
            BuildBreakdown(bet => bet.Class.HasValue ? $"Class {bet.Class.Value}" : "Unknown");

        public IReadOnlyList<ValidationBreakdownRow> ByGoing =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.Going) ? "Unknown" : bet.Going!);

        public IReadOnlyList<ValidationBreakdownRow> BySurface =>
            BuildBreakdown(bet => string.IsNullOrWhiteSpace(bet.Surface) ? "Unknown" : bet.Surface!);

        private IReadOnlyList<ValidationBreakdownRow> BuildBreakdown(
            Func<ValidationBetResult, string> keySelector)
        {
            return Bets
                .GroupBy(keySelector)
                .Select(group => new ValidationBreakdownRow
                {
                    Group = group.Key,
                    Bets = group.Count(),
                    Wins = group.Count(bet => bet.Won),
                    Staked = group.Sum(bet => bet.Stake),
                    Profit = group.Sum(bet => bet.Profit)
                })
                .OrderByDescending(row => row.Bets)
                .ThenBy(row => row.Group)
                .ToList();
        }
    }

    public class ValidationBreakdownRow
    {
        public string Group { get; init; } = "Unknown";
        public int Bets { get; init; }
        public int Wins { get; init; }
        public decimal Staked { get; init; }
        public decimal Profit { get; init; }
        public decimal Roi => Staked <= 0 ? 0 : Profit / Staked;
        public double StrikeRate => Bets <= 0 ? 0 : (double)Wins / Bets;
    }

    public class ValidationBreakdownViewModel
    {
        public string Title { get; init; } = string.Empty;
        public IReadOnlyList<ValidationBreakdownRow> Rows { get; init; } =
            Array.Empty<ValidationBreakdownRow>();
    }

    public class ValidationBetResult
    {
        public int RaceId { get; init; }
        public DateTime? RaceDate { get; init; }
        public string? RaceTitle { get; init; }
        public string? CourseName { get; init; }
        public string? HorseName { get; init; }
        public string? RaceType { get; init; }
        public byte? Class { get; init; }
        public string? AgeRestriction { get; init; }
        public string? Surface { get; init; }
        public string? Going { get; init; }
        public short? DistanceYards { get; init; }
        public string? DistanceText { get; init; }
        public byte? RunnerCount { get; init; }
        public decimal DecimalOdds { get; init; }
        public decimal AiDecimalOdds { get; init; }
        public double AiProbability { get; init; }
        public double MarketProbability { get; init; }
        public double Differential { get; init; }
        public decimal Stake { get; init; }
        public bool Won { get; init; }
        public decimal BankrollAfter { get; init; }
        public decimal EffectiveDecimalOdds { get; init; }
        public decimal CommissionRate { get; init; }
        public decimal Profit
        {
            get
            {
                var odds = EffectiveDecimalOdds > 1m ? EffectiveDecimalOdds : DecimalOdds;
                return Won ? Stake * (odds - 1m) : -Stake;
            }
        }
    }
}
