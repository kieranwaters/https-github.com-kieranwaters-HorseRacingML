using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.Models;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Higher-level analysis over a cached snapshot: tune/confirm split,
    /// walk-forward stability and simple strategy baselines. All methods are
    /// pure and re-use <see cref="BettingSimulationEngine"/>.
    /// </summary>
    public static class SimulatorAnalytics
    {
        public static SimulatorComparison? BuildComparison(
            BettingSimulationSnapshot snapshot,
            SimulatorFilterRequest filter)
        {
            if (snapshot is null || filter is null || filter.ConfirmPercent <= 0m)
            {
                return null;
            }

            var (start, end) = ResolveDateRange(snapshot, filter);
            if (!start.HasValue || !end.HasValue)
            {
                return null;
            }

            int totalDays = (end.Value.Date - start.Value.Date).Days;
            if (totalDays < 2)
            {
                return null;
            }

            int tuneDays = (int)Math.Floor(totalDays * (double)(1m - filter.ConfirmPercent / 100m));
            tuneDays = Math.Clamp(tuneDays, 1, totalDays - 1);
            var split = start.Value.Date.AddDays(tuneDays);
            if (split >= end.Value.Date)
            {
                return null;
            }

            var tuneFilter = filter.Clone();
            tuneFilter.ConfirmPercent = 0m;
            tuneFilter.Page = 1;
            tuneFilter.BootstrapSamples = Math.Min(filter.BootstrapSamples, 500);
            tuneFilter.FromDate = start.Value.Date;
            tuneFilter.ToDate = split;

            var confirmFilter = filter.Clone();
            confirmFilter.ConfirmPercent = 0m;
            confirmFilter.Page = 1;
            confirmFilter.BootstrapSamples = Math.Min(filter.BootstrapSamples, 500);
            confirmFilter.FromDate = split.AddDays(1);
            confirmFilter.ToDate = end.Value.Date;

            return new SimulatorComparison
            {
                SplitDate = split,
                ConfirmPercent = filter.ConfirmPercent,
                Tune = BettingSimulationEngine.Simulate(snapshot, tuneFilter),
                Confirm = BettingSimulationEngine.Simulate(snapshot, confirmFilter)
            };
        }

        public static IReadOnlyList<SimulatorPeriodResult> BuildWalkForward(
            BettingSimulationSnapshot snapshot,
            SimulatorFilterRequest filter,
            int periods = 4)
        {
            if (snapshot is null || filter is null)
            {
                return Array.Empty<SimulatorPeriodResult>();
            }

            var (start, end) = ResolveDateRange(snapshot, filter);
            if (!start.HasValue || !end.HasValue)
            {
                return Array.Empty<SimulatorPeriodResult>();
            }

            int totalDays = (end.Value.Date - start.Value.Date).Days;
            periods = Math.Clamp(periods, 2, 12);
            if (totalDays < periods)
            {
                return Array.Empty<SimulatorPeriodResult>();
            }

            var results = new List<SimulatorPeriodResult>(periods);
            for (int index = 0; index < periods; index++)
            {
                int fromOffset = (int)Math.Floor((double)totalDays * index / periods);
                int toOffset = index == periods - 1
                    ? totalDays
                    : (int)Math.Floor((double)totalDays * (index + 1) / periods) - 1;
                if (toOffset < fromOffset)
                {
                    toOffset = fromOffset;
                }

                var periodStart = start.Value.Date.AddDays(fromOffset);
                var periodEnd = start.Value.Date.AddDays(toOffset);
                var periodFilter = filter.Clone();
                periodFilter.ConfirmPercent = 0m;
                periodFilter.Page = 1;
                periodFilter.BootstrapSamples = 0;
                periodFilter.FromDate = periodStart;
                periodFilter.ToDate = periodEnd;

                var result = BettingSimulationEngine.Simulate(snapshot, periodFilter);
                results.Add(new SimulatorPeriodResult
                {
                    Label = $"{periodStart:yyyy-MM-dd} → {periodEnd:yyyy-MM-dd}",
                    Start = periodStart,
                    End = periodEnd,
                    Bets = result.BetCount,
                    Staked = result.TotalStaked,
                    Profit = result.Profit,
                    ExpectedProfit = result.ExpectedProfit,
                    MaxDrawdown = result.MaxDrawdown,
                    Roi = (double)result.Roi,
                    StrikeRate = result.StrikeRate,
                    ProfitFactor = result.ProfitFactor
                });
            }

            return results;
        }

        public static IReadOnlyList<SimulatorBaselineRow> BuildBaselines(
            BettingSimulationSnapshot snapshot,
            SimulatorFilterRequest filter)
        {
            if (snapshot is null || filter is null)
            {
                return Array.Empty<SimulatorBaselineRow>();
            }

            decimal stake = Math.Max(1m, decimal.Round(filter.StartingBankroll * 0.01m, 2));
            var rows = new List<SimulatorBaselineRow>();

            var favouriteFilter = BuildBaselineFilter(
                filter,
                stake,
                SimulatorSelectionStrategy.MarketFavourite);
            rows.Add(ToBaselineRow(
                "Back the market favourite",
                "Shortest priced runner in every eligible race.",
                BettingSimulationEngine.Simulate(snapshot, favouriteFilter)));

            var topPickFilter = BuildBaselineFilter(
                filter,
                stake,
                SimulatorSelectionStrategy.ModelTopPick);
            rows.Add(ToBaselineRow(
                "Back the model top pick",
                "Highest AI probability, ignoring price value.",
                BettingSimulationEngine.Simulate(snapshot, topPickFilter)));

            var positiveEvFilter = filter.Clone();
            positiveEvFilter.ConfirmPercent = 0m;
            positiveEvFilter.Page = 1;
            positiveEvFilter.BootstrapSamples = 0;
            positiveEvFilter.StakeMode = SimulatorStakeMode.Flat;
            positiveEvFilter.FlatStake = stake;
            positiveEvFilter.MinStake = 0m;
            positiveEvFilter.MaxStakeAmount = null;
            positiveEvFilter.MaxStakePercent = 100m;
            positiveEvFilter.MaxDailyStakePercent = 100m;
            positiveEvFilter.MaxDailyLossPercent = 100m;
            positiveEvFilter.SelectionStrategy = SimulatorSelectionStrategy.HighestDifferential;
            positiveEvFilter.BetsPerRace = 1;
            positiveEvFilter.MinEdgePercent = 0m;
            positiveEvFilter.MinProbabilitySpread = Math.Min(filter.MinProbabilitySpread, 0.01m);
            rows.Add(ToBaselineRow(
                "All positive-EV bets (flat)",
                "Highest differential per race with no edge cut-off.",
                BettingSimulationEngine.Simulate(snapshot, positiveEvFilter)));

            return rows;
        }

        private static SimulatorFilterRequest BuildBaselineFilter(
            SimulatorFilterRequest source,
            decimal stake,
            SimulatorSelectionStrategy strategy)
        {
            var filter = source.Clone();
            filter.ConfirmPercent = 0m;
            filter.Page = 1;
            filter.BootstrapSamples = 0;
            filter.StakeMode = SimulatorStakeMode.Flat;
            filter.FlatStake = stake;
            filter.MinStake = 0m;
            filter.MaxStakeAmount = null;
            filter.MaxStakePercent = 100m;
            filter.MaxDailyStakePercent = 100m;
            filter.MaxDailyLossPercent = 100m;
            filter.SelectionStrategy = strategy;
            filter.BetsPerRace = 1;
            filter.MinEdgePercent = -100m;
            filter.MaxEdgePercent = null;
            filter.MinAiProbabilityPercent = 0m;
            filter.MaxAiProbabilityPercent = null;
            filter.MinProbabilitySpread = 0m;
            filter.MinOdds = 1.01m;
            filter.MaxOdds = null;
            return filter;
        }

        private static SimulatorBaselineRow ToBaselineRow(
            string name,
            string note,
            BettingSimulationResult result)
        {
            return new SimulatorBaselineRow
            {
                Name = name,
                Note = note,
                Bets = result.BetCount,
                Staked = result.TotalStaked,
                Profit = result.Profit,
                Roi = (double)result.Roi,
                StrikeRate = result.StrikeRate,
                ProfitFactor = result.ProfitFactor
            };
        }

        private static (DateTime? Start, DateTime? End) ResolveDateRange(
            BettingSimulationSnapshot snapshot,
            SimulatorFilterRequest filter)
        {
            var start = filter.FromDate?.Date
                ?? snapshot.Summary.FirstRaceDate?.Date
                ?? snapshot.Summary.ValidationStart.Date;
            var end = filter.ToDate?.Date
                ?? snapshot.Summary.LastRaceDate?.Date
                ?? snapshot.Summary.ValidationEnd.Date;

            if (start > end)
            {
                return (null, null);
            }

            return (start, end);
        }
    }
}
