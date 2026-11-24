using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.Models;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Replays the cached evaluation predictions with an arbitrary set of
    /// betting filters. This is deliberately separate from the trainer: no
    /// TensorFlow work happens here, so the page can re-price a filter change
    /// in milliseconds.
    /// </summary>
    public static class BettingSimulationEngine
    {
        public static BettingSimulationResult Simulate(
            BettingSimulationSnapshot snapshot,
            SimulatorFilterRequest filter)
        {
            if (snapshot is null)
            {
                throw new ArgumentNullException(nameof(snapshot));
            }

            filter ??= new SimulatorFilterRequest();

            decimal bankroll = Math.Max(0m, filter.StartingBankroll);
            decimal startingBankroll = bankroll;
            decimal totalStaked = 0m;
            decimal grossWinnings = 0m;
            decimal grossLosses = 0m;
            decimal peakBankroll = startingBankroll;
            decimal maxDrawdown = 0m;
            decimal maxDrawdownPercent = 0m;
            decimal expectedProfit = 0m;
            double expectedWins = 0d;
            int wins = 0;
            int longestWinStreak = 0;
            int longestLosingStreak = 0;
            int currentWinStreak = 0;
            int currentLosingStreak = 0;

            decimal commission = Math.Clamp(filter.CommissionPercent / 100m, 0m, 1m);
            decimal maxKelly = Math.Clamp(filter.MaxKellyPercent / 100m, 0m, 1m);
            decimal minEdge = filter.MinEdgePercent / 100m;
            decimal? maxEdge = filter.MaxEdgePercent.HasValue
                ? filter.MaxEdgePercent.Value / 100m
                : null;
            decimal minAiProbability = Math.Clamp(filter.MinAiProbabilityPercent / 100m, 0m, 1m);
            decimal? maxAiProbability = filter.MaxAiProbabilityPercent.HasValue
                ? Math.Clamp(filter.MaxAiProbabilityPercent.Value / 100m, 0m, 1m)
                : null;
            decimal slippage = Math.Clamp(filter.PriceSlippagePercent / 100m, 0m, 0.90m);
            decimal maxStakePercent = Math.Clamp(filter.MaxStakePercent, 0m, 100m) / 100m;
            decimal maxDailyStakePercent = Math.Clamp(filter.MaxDailyStakePercent, 0m, 100m) / 100m;
            decimal maxDailyLossPercent = Math.Clamp(filter.MaxDailyLossPercent, 0m, 100m) / 100m;

            bool isBaselineStrategy =
                filter.SelectionStrategy is SimulatorSelectionStrategy.MarketFavourite
                    or SimulatorSelectionStrategy.ModelTopPick;

            var bets = new List<SimulatorBet>();
            var equity = new List<SimulatorEquityPoint>
            {
                new()
                {
                    BetNumber = 0,
                    Bankroll = startingBankroll,
                    Profit = 0m
                }
            };

            int racesConsidered = 0;
            int racesPassedRaceFilter = 0;
            int racesSkippedLowSpread = 0;
            int racesWithCandidate = 0;
            int racesWithStake = 0;
            int unpricedRunnerRows = 0;
            bool stoppedForBankroll = false;

            DateTime? currentDay = null;
            decimal dayStartBankroll = startingBankroll;
            decimal dayStaked = 0m;
            decimal dayProfit = 0m;

            if (bankroll > 0m)
            {
                foreach (var race in GetRaces(snapshot)
                             .OrderBy(race => race.RaceDate ?? DateTime.MaxValue)
                             .ThenBy(race => race.RaceId))
                {
                    if (bankroll <= 0m)
                    {
                        stoppedForBankroll = true;
                        break;
                    }

                    racesConsidered++;
                    var rows = race.Runners;
                    if (rows.Count == 0)
                    {
                        continue;
                    }

                    if (!PassesRaceFilter(race, rows.Count, filter))
                    {
                        continue;
                    }

                    racesPassedRaceFilter++;

                    double rawMin = rows.Min(row => row.RawProbability);
                    double rawMax = rows.Max(row => row.RawProbability);
                    if (!isBaselineStrategy &&
                        rawMax - rawMin < (double)filter.MinProbabilitySpread)
                    {
                        // Match the live normaliser: do not turn an
                        // uninformative near-uniform vector into bets.
                        racesSkippedLowSpread++;
                        continue;
                    }

                    var candidates = new List<SimulatorCandidate>(rows.Count);
                    foreach (var row in rows)
                    {
                        var candidate = BuildCandidate(
                            row,
                            commission,
                            slippage,
                            maxKelly,
                            filter.PriceSource);
                        if (candidate is null)
                        {
                            if (ResolvePrice(row, filter.PriceSource) is null)
                            {
                                unpricedRunnerRows++;
                            }

                            continue;
                        }

                        if (!isBaselineStrategy)
                        {
                            if (candidate.Edge < (double)minEdge)
                            {
                                continue;
                            }

                            if (maxEdge.HasValue && candidate.Edge > (double)maxEdge.Value)
                            {
                                continue;
                            }

                            if ((decimal)candidate.Probability < minAiProbability ||
                                (maxAiProbability.HasValue &&
                                 (decimal)candidate.Probability > maxAiProbability.Value))
                            {
                                continue;
                            }

                            if (candidate.DecimalOdds < filter.MinOdds ||
                                (filter.MaxOdds.HasValue &&
                                 candidate.DecimalOdds > filter.MaxOdds.Value))
                            {
                                continue;
                            }
                        }

                        candidates.Add(candidate);
                    }

                    if (candidates.Count == 0)
                    {
                        continue;
                    }

                    racesWithCandidate++;

                    var ordered = OrderCandidates(candidates, filter.SelectionStrategy);
                    int raceBets = 0;
                    int betsToPlace = Math.Clamp(filter.BetsPerRace, 1, 10);

                    foreach (var candidate in ordered)
                    {
                        if (raceBets >= betsToPlace || bankroll <= 0m)
                        {
                            break;
                        }

                        var raceDay = race.RaceDate?.Date;
                        if (currentDay != raceDay)
                        {
                            currentDay = raceDay;
                            dayStartBankroll = bankroll;
                            dayStaked = 0m;
                            dayProfit = 0m;
                        }

                        if (maxDailyStakePercent > 0m &&
                            dayStaked >= dayStartBankroll * maxDailyStakePercent)
                        {
                            break;
                        }

                        if (maxDailyLossPercent > 0m &&
                            dayStartBankroll > 0m &&
                            dayProfit <= -dayStartBankroll * maxDailyLossPercent)
                        {
                            break;
                        }

                        var stake = CalculateStake(bankroll, candidate, filter, maxKelly);
                        if (stake <= 0m)
                        {
                            continue;
                        }

                        if (maxStakePercent > 0m)
                        {
                            stake = Math.Min(stake, bankroll * maxStakePercent);
                        }

                        if (filter.MaxStakeAmount.HasValue)
                        {
                            stake = Math.Min(stake, Math.Max(0m, filter.MaxStakeAmount.Value));
                        }

                        if (maxDailyStakePercent > 0m)
                        {
                            var remainingDaily = dayStartBankroll * maxDailyStakePercent - dayStaked;
                            stake = Math.Min(stake, Math.Max(0m, remainingDaily));
                        }

                        stake = Math.Min(stake, bankroll);
                        stake = decimal.Round(stake, 2, MidpointRounding.ToZero);
                        if (stake <= 0m || stake < filter.MinStake)
                        {
                            continue;
                        }

                        bankroll -= stake;
                        totalStaked += stake;
                        dayStaked += stake;

                        bool won = candidate.Row.Won;
                        decimal profit;
                        if (won)
                        {
                            wins++;
                            currentWinStreak++;
                            currentLosingStreak = 0;
                            longestWinStreak = Math.Max(longestWinStreak, currentWinStreak);
                            profit = stake * (candidate.EffectiveDecimalOdds - 1m);
                            grossWinnings += profit;
                            bankroll += stake * candidate.EffectiveDecimalOdds;
                        }
                        else
                        {
                            currentLosingStreak++;
                            currentWinStreak = 0;
                            longestLosingStreak = Math.Max(longestLosingStreak, currentLosingStreak);
                            profit = -stake;
                            grossLosses += stake;
                        }

                        dayProfit += profit;
                        raceBets++;

                        var betExpectedProfit = stake *
                            ((decimal)candidate.Probability * candidate.EffectiveDecimalOdds - 1m);
                        expectedProfit += betExpectedProfit;
                        expectedWins += candidate.Probability;

                        peakBankroll = Math.Max(peakBankroll, bankroll);
                        decimal drawdown = peakBankroll - bankroll;
                        if (drawdown > maxDrawdown)
                        {
                            maxDrawdown = drawdown;
                            maxDrawdownPercent = peakBankroll > 0m
                                ? drawdown / peakBankroll
                                : 0m;
                        }

                        bets.Add(new SimulatorBet
                        {
                            RaceId = candidate.Row.RaceId,
                            RaceDate = candidate.Row.RaceDate,
                            RaceTitle = candidate.Row.RaceTitle,
                            CourseName = candidate.Row.CourseName,
                            HorseName = candidate.Row.HorseName,
                            RaceType = candidate.Row.RaceType,
                            Class = candidate.Row.Class,
                            AgeRestriction = candidate.Row.AgeRestriction,
                            Surface = candidate.Row.Surface,
                            Going = candidate.Row.Going,
                            DistanceYards = candidate.Row.DistanceYards,
                            DistanceText = candidate.Row.DistanceText,
                            RunnerCount = candidate.Row.RunnerCount,
                            FinishPosition = candidate.Row.FinishPosition,
                            DecimalOdds = candidate.DecimalOdds,
                            EffectiveDecimalOdds = candidate.EffectiveDecimalOdds,
                            BetfairSp = candidate.Row.BetfairSp,
                            BetfairLastTradedPrice = candidate.Row.BetfairLastTradedPrice,
                            PriceSource = filter.PriceSource,
                            ClvPercent = CalculateClv(candidate.DecimalOdds, candidate.Row.BetfairSp),
                            AiProbability = candidate.Probability,
                            MarketProbability = candidate.MarketProbability,
                            Differential = candidate.Differential,
                            Edge = candidate.Edge,
                            Stake = stake,
                            Won = won,
                            Profit = profit,
                            ExpectedProfit = betExpectedProfit,
                            BankrollAfter = bankroll
                        });

                        equity.Add(new SimulatorEquityPoint
                        {
                            BetNumber = bets.Count,
                            RaceDate = candidate.Row.RaceDate,
                            Bankroll = bankroll,
                            Profit = bankroll - startingBankroll
                        });
                    }

                    if (raceBets > 0)
                    {
                        racesWithStake++;
                    }
                }
            }

            if (bankroll <= 0m && racesConsidered > 0)
            {
                stoppedForBankroll = true;
            }

            var statistics = CalculateStatistics(bets, filter);

            return new BettingSimulationResult
            {
                StartingBankroll = startingBankroll,
                EndingBankroll = bankroll,
                TotalStaked = totalStaked,
                BetCount = bets.Count,
                WinCount = wins,
                LongestWinStreak = longestWinStreak,
                LongestLosingStreak = longestLosingStreak,
                MaxDrawdown = maxDrawdown,
                MaxDrawdownPercent = maxDrawdownPercent,
                AverageEdge = bets.Count == 0 ? 0d : bets.Average(bet => bet.Edge),
                AverageOdds = bets.Count == 0 ? 0d : bets.Average(bet => (double)bet.DecimalOdds),
                AverageEffectiveOdds = bets.Count == 0
                    ? 0d
                    : bets.Average(bet => (double)bet.EffectiveDecimalOdds),
                AverageAiProbability = bets.Count == 0
                    ? 0d
                    : bets.Average(bet => bet.AiProbability),
                AverageMarketProbability = bets.Count == 0
                    ? 0d
                    : bets.Average(bet => bet.MarketProbability),
                AverageStake = bets.Count == 0 ? 0m : bets.Average(bet => bet.Stake),
                GrossWinnings = grossWinnings,
                GrossLosses = grossLosses,
                RacesConsidered = racesConsidered,
                RacesPassedRaceFilter = racesPassedRaceFilter,
                RacesSkippedLowSpread = racesSkippedLowSpread,
                RacesWithCandidate = racesWithCandidate,
                RacesWithStake = racesWithStake,
                UnpricedRunnerRows = unpricedRunnerRows,
                StoppedForBankroll = stoppedForBankroll,
                ExpectedProfit = expectedProfit,
                ExpectedWins = expectedWins,
                BreakEvenStrikeRate = CalculateBreakEvenStrikeRate(bets),
                RoiConfidenceLow = statistics.RoiLow,
                RoiConfidenceHigh = statistics.RoiHigh,
                ProfitConfidenceLow = statistics.ProfitLow,
                ProfitConfidenceHigh = statistics.ProfitHigh,
                ProbabilityOfProfit = statistics.ProbabilityOfProfit,
                StandardError = statistics.StandardError,
                TStatistic = statistics.TStatistic,
                BootstrapSamples = statistics.Samples,
                PriceSource = filter.PriceSource,
                SnapshotKey = filter.SnapshotKey,
                Bets = bets,
                EquityCurve = equity
            };
        }

        private static IEnumerable<SimulatorRace> GetRaces(BettingSimulationSnapshot snapshot)
        {
            if (snapshot.Races is { Count: > 0 })
            {
                return snapshot.Races;
            }

            return snapshot.Runners
                .GroupBy(row => row.RaceId)
                .Select(group =>
                {
                    var first = group.First();
                    return new SimulatorRace
                    {
                        RaceId = group.Key,
                        RaceDate = first.RaceDate,
                        RaceTitle = first.RaceTitle,
                        CourseName = first.CourseName,
                        RaceType = first.RaceType,
                        Class = first.Class,
                        AgeRestriction = first.AgeRestriction,
                        Surface = first.Surface,
                        Going = first.Going,
                        DistanceYards = first.DistanceYards,
                        DistanceText = first.DistanceText,
                        RunnerCount = first.RunnerCount,
                        Runners = group.ToList()
                    };
                });
        }

        private static bool PassesRaceFilter(
            SimulatorRace race,
            int rowCount,
            SimulatorFilterRequest filter)
        {
            int runnerCount = race.RunnerCount > 0 ? race.RunnerCount : rowCount;
            if (filter.MinRunners.HasValue && runnerCount < filter.MinRunners.Value)
            {
                return false;
            }

            if (filter.MaxRunners.HasValue && runnerCount > filter.MaxRunners.Value)
            {
                return false;
            }

            if (!IsAny(filter.RaceType) &&
                !string.Equals(Normalise(race.RaceType), Normalise(filter.RaceType), StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (!IsAny(filter.Surface) &&
                !string.Equals(Normalise(race.Surface), Normalise(filter.Surface), StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (!IsAny(filter.Going) &&
                !string.Equals(Normalise(race.Going), Normalise(filter.Going), StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (!IsAny(filter.AgeRestriction) &&
                !string.Equals(
                    Normalise(race.AgeRestriction),
                    Normalise(filter.AgeRestriction),
                    StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (filter.ClassMin.HasValue &&
                (!race.Class.HasValue || race.Class.Value < filter.ClassMin.Value))
            {
                return false;
            }

            if (filter.ClassMax.HasValue &&
                (!race.Class.HasValue || race.Class.Value > filter.ClassMax.Value))
            {
                return false;
            }

            var selectedCourses = (filter.Courses ?? new List<string>())
                .Where(course => !IsAny(course))
                .ToList();
            if (selectedCourses.Count > 0 &&
                !selectedCourses.Any(course => string.Equals(
                    course.Trim(),
                    Normalise(race.CourseName),
                    StringComparison.OrdinalIgnoreCase)))
            {
                return false;
            }

            if (!string.IsNullOrWhiteSpace(filter.CourseContains) &&
                (race.CourseName ?? string.Empty).IndexOf(
                    filter.CourseContains.Trim(),
                    StringComparison.OrdinalIgnoreCase) < 0)
            {
                return false;
            }

            if (filter.MinDistanceYards.HasValue &&
                (!race.DistanceYards.HasValue ||
                 race.DistanceYards.Value < filter.MinDistanceYards.Value))
            {
                return false;
            }

            if (filter.MaxDistanceYards.HasValue &&
                (!race.DistanceYards.HasValue ||
                 race.DistanceYards.Value > filter.MaxDistanceYards.Value))
            {
                return false;
            }

            if (filter.FromDate.HasValue &&
                (!race.RaceDate.HasValue || race.RaceDate.Value.Date < filter.FromDate.Value.Date))
            {
                return false;
            }

            if (filter.ToDate.HasValue &&
                (!race.RaceDate.HasValue || race.RaceDate.Value.Date > filter.ToDate.Value.Date))
            {
                return false;
            }

            return true;
        }

        private static SimulatorCandidate? BuildCandidate(
            SimulatorRunnerRow row,
            decimal commission,
            decimal slippage,
            decimal maxKelly,
            SimulatorPriceSource priceSource)
        {
            var selectedOdds = ResolvePrice(row, priceSource);
            if (!selectedOdds.HasValue || selectedOdds.Value <= 1m)
            {
                return null;
            }

            // Model a worse-than-quoted fill before commission. The selected
            // price shown on the bet is the executable price used here.
            var adjustedOdds = slippage <= 0m
                ? selectedOdds.Value
                : 1m + (selectedOdds.Value - 1m) * (1m - slippage);
            if (adjustedOdds <= 1m)
            {
                return null;
            }

            var effectiveOdds = BettingMath.CalculateEffectiveDecimalOdds(adjustedOdds, commission);
            if (effectiveOdds <= 1m)
            {
                return null;
            }

            double probability = Math.Clamp(row.Probability, 1e-9d, 1d - 1e-9d);
            double marketProbability = 1d / (double)effectiveOdds;
            if (!double.IsFinite(marketProbability) || marketProbability <= 0d)
            {
                return null;
            }

            return new SimulatorCandidate
            {
                Row = row,
                Probability = probability,
                MarketProbability = marketProbability,
                Differential = probability - marketProbability,
                Edge = probability * (double)effectiveOdds - 1d,
                DecimalOdds = adjustedOdds,
                EffectiveDecimalOdds = effectiveOdds,
                KellyFraction = BettingMath.CalculateKellyFraction(
                    probability,
                    (double)effectiveOdds,
                    maxKelly)
            };
        }

        private static IEnumerable<SimulatorCandidate> OrderCandidates(
            IEnumerable<SimulatorCandidate> candidates,
            SimulatorSelectionStrategy strategy)
        {
            return strategy switch
            {
                SimulatorSelectionStrategy.HighestEdge => candidates
                    .OrderByDescending(candidate => candidate.Edge)
                    .ThenByDescending(candidate => candidate.Probability),
                SimulatorSelectionStrategy.HighestKelly => candidates
                    .OrderByDescending(candidate => candidate.KellyFraction)
                    .ThenByDescending(candidate => candidate.Edge),
                SimulatorSelectionStrategy.ShortestPrice or SimulatorSelectionStrategy.MarketFavourite =>
                    candidates
                        .OrderBy(candidate => candidate.DecimalOdds)
                        .ThenByDescending(candidate => candidate.Probability),
                SimulatorSelectionStrategy.ModelTopPick => candidates
                    .OrderByDescending(candidate => candidate.Probability)
                    .ThenBy(candidate => candidate.DecimalOdds),
                _ => candidates
                    .OrderByDescending(candidate => candidate.Differential)
                    .ThenByDescending(candidate => candidate.Probability)
            };
        }

        private static decimal? ResolvePrice(SimulatorRunnerRow row, SimulatorPriceSource priceSource)
        {
            static decimal? Valid(decimal? price) => price is > 1m ? price : null;

            return priceSource switch
            {
                SimulatorPriceSource.BetfairSp =>
                    Valid(row.BetfairSp) ?? Valid(row.DecimalOdds),
                SimulatorPriceSource.LastTraded =>
                    Valid(row.BetfairLastTradedPrice) ?? Valid(row.DecimalOdds),
                SimulatorPriceSource.BestAvailable => new[]
                    {
                        Valid(row.DecimalOdds),
                        Valid(row.BetfairSp),
                        Valid(row.BetfairLastTradedPrice)
                    }
                    .Where(price => price.HasValue)
                    .Select(price => price!.Value)
                    .DefaultIfEmpty(0m)
                    .Max() is var best && best > 1m
                        ? best
                        : null,
                _ => Valid(row.DecimalOdds)
            };
        }

        private static decimal CalculateStake(
            decimal bankroll,
            SimulatorCandidate candidate,
            SimulatorFilterRequest filter,
            decimal maxKelly)
        {
            if (bankroll <= 0m)
            {
                return 0m;
            }

            decimal stake = filter.StakeMode switch
            {
                SimulatorStakeMode.Flat => Math.Max(0m, filter.FlatStake),
                SimulatorStakeMode.BankrollPercent =>
                    bankroll * Math.Clamp(filter.BankrollPercent, 0m, 100m) / 100m,
                _ => BettingMath.CalculateSequentialStake(
                    bankroll,
                    BettingMath.CalculateKellyFraction(
                        candidate.Probability,
                        (double)candidate.EffectiveDecimalOdds,
                        maxKelly))
            };

            if (stake <= 0m)
            {
                return 0m;
            }

            stake = Math.Min(stake, bankroll);
            stake = decimal.Round(stake, 2, MidpointRounding.ToZero);
            return stake;
        }

        private static double CalculateClv(decimal decimalOdds, decimal? betfairSp)
        {
            if (!betfairSp.HasValue || betfairSp.Value <= 1m || decimalOdds <= 1m)
            {
                return 0d;
            }

            return (double)((decimalOdds / betfairSp.Value) - 1m);
        }

        private static double CalculateBreakEvenStrikeRate(IReadOnlyList<SimulatorBet> bets)
        {
            if (bets.Count == 0)
            {
                return 0d;
            }

            decimal weightedReturn = 0m;
            decimal staked = 0m;
            foreach (var bet in bets)
            {
                weightedReturn += bet.Stake * bet.EffectiveDecimalOdds;
                staked += bet.Stake;
            }

            return weightedReturn <= 0m ? 0d : (double)(staked / weightedReturn);
        }

        private static SimulatorStatistics CalculateStatistics(
            IReadOnlyList<SimulatorBet> bets,
            SimulatorFilterRequest filter)
        {
            if (bets.Count == 0)
            {
                return new SimulatorStatistics();
            }

            int samples = Math.Clamp(filter.BootstrapSamples, 0, 10_000);
            decimal totalStake = bets.Sum(bet => bet.Stake);
            decimal totalProfit = bets.Sum(bet => bet.Profit);
            decimal mean = totalProfit / bets.Count;

            double variance = 0d;
            foreach (var bet in bets)
            {
                double difference = (double)(bet.Profit - mean);
                variance += difference * difference;
            }

            variance = bets.Count > 1 ? variance / (bets.Count - 1) : 0d;
            double standardError = Math.Sqrt(variance / bets.Count);
            double tStatistic = standardError > 0d ? (double)mean / standardError : 0d;

            if (samples <= 0)
            {
                return new SimulatorStatistics
                {
                    Samples = 0,
                    RoiLow = totalStake <= 0m ? 0d : (double)(totalProfit / totalStake),
                    RoiHigh = totalStake <= 0m ? 0d : (double)(totalProfit / totalStake),
                    ProfitLow = totalProfit,
                    ProfitHigh = totalProfit,
                    ProbabilityOfProfit = totalProfit > 0m ? 1d : 0d,
                    StandardError = standardError,
                    TStatistic = tStatistic
                };
            }

            // Resample races rather than individual bets so multiple bets in
            // the same race keep their correlation.
            var raceGroups = bets
                .GroupBy(bet => bet.RaceId)
                .Select(group => new
                {
                    Stake = group.Sum(bet => bet.Stake),
                    Profit = group.Sum(bet => bet.Profit)
                })
                .ToArray();

            var random = new Random(filter.BootstrapSeed);
            var rois = new double[samples];
            var profits = new decimal[samples];
            int profitable = 0;
            for (int sample = 0; sample < samples; sample++)
            {
                decimal sampleStake = 0m;
                decimal sampleProfit = 0m;
                for (int i = 0; i < raceGroups.Length; i++)
                {
                    var group = raceGroups[random.Next(raceGroups.Length)];
                    sampleStake += group.Stake;
                    sampleProfit += group.Profit;
                }

                profits[sample] = sampleProfit;
                rois[sample] = sampleStake <= 0m ? 0d : (double)(sampleProfit / sampleStake);
                if (sampleProfit > 0m)
                {
                    profitable++;
                }
            }

            Array.Sort(rois);
            Array.Sort(profits);

            return new SimulatorStatistics
            {
                Samples = samples,
                RoiLow = Percentile(rois, 0.025d),
                RoiHigh = Percentile(rois, 0.975d),
                ProfitLow = Percentile(profits, 0.025d),
                ProfitHigh = Percentile(profits, 0.975d),
                ProbabilityOfProfit = (double)profitable / samples,
                StandardError = standardError,
                TStatistic = tStatistic
            };
        }

        private static double Percentile(double[] sorted, double percentile)
        {
            if (sorted.Length == 0)
            {
                return 0d;
            }

            double position = Math.Clamp(percentile, 0d, 1d) * (sorted.Length - 1);
            int lower = (int)Math.Floor(position);
            int upper = (int)Math.Ceiling(position);
            if (lower == upper)
            {
                return sorted[lower];
            }

            double weight = position - lower;
            return sorted[lower] * (1d - weight) + sorted[upper] * weight;
        }

        private static decimal Percentile(decimal[] sorted, double percentile)
        {
            if (sorted.Length == 0)
            {
                return 0m;
            }

            double position = Math.Clamp(percentile, 0d, 1d) * (sorted.Length - 1);
            int lower = (int)Math.Floor(position);
            int upper = (int)Math.Ceiling(position);
            if (lower == upper)
            {
                return sorted[lower];
            }

            decimal weight = (decimal)(position - lower);
            return sorted[lower] * (1m - weight) + sorted[upper] * weight;
        }

        private static bool IsAny(string? value) =>
            string.IsNullOrWhiteSpace(value) ||
            string.Equals(value.Trim(), "Any", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value.Trim(), "All", StringComparison.OrdinalIgnoreCase);

        private static string Normalise(string? value) =>
            string.IsNullOrWhiteSpace(value) ? "Unknown" : value.Trim();

        private sealed class SimulatorCandidate
        {
            public SimulatorRunnerRow Row { get; init; } = null!;
            public double Probability { get; init; }
            public double MarketProbability { get; init; }
            public double Differential { get; init; }
            public double Edge { get; init; }
            public decimal DecimalOdds { get; init; }
            public decimal EffectiveDecimalOdds { get; init; }
            public decimal KellyFraction { get; init; }
        }

        private sealed class SimulatorStatistics
        {
            public int Samples { get; init; }
            public double RoiLow { get; init; }
            public double RoiHigh { get; init; }
            public decimal ProfitLow { get; init; }
            public decimal ProfitHigh { get; init; }
            public double ProbabilityOfProfit { get; init; }
            public double StandardError { get; init; }
            public double TStatistic { get; init; }
        }
    }
}
