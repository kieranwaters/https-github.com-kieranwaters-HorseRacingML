using System;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Provides helper methods for calculating bet sizing and odds
    /// related metrics that are shared between automated betting and
    /// offline simulations.
    /// </summary>
    public static class BettingMath
    {
        public static decimal CalculateAiDecimalOdds(double aiProbability)
        {
            if (aiProbability <= 0)
            {
                return 0m;
            }

            var inverted = 1.0 / aiProbability;
            if (double.IsPositiveInfinity(inverted))
            {
                return decimal.MaxValue;
            }

            if (!double.IsFinite(inverted) || inverted <= 0)
            {
                return 0m;
            }

            if (inverted > (double)decimal.MaxValue)
            {
                return decimal.MaxValue;
            }

            return (decimal)inverted;
        }

        public static decimal CalculateKellyFraction(double probability, double decimalOdds, decimal? maxFraction = null)
        {
            if (probability <= 0 || probability >= 1 || decimalOdds <= 1)
            {
                return 0m;
            }

            var b = decimalOdds - 1.0;
            if (Math.Abs(b) < double.Epsilon)
            {
                return 0m;
            }

            var q = 1.0 - probability;
            var fraction = (b * probability - q) / b;

            if (!double.IsFinite(fraction))
            {
                return 0m;
            }

            var result = (decimal)fraction;
            if (result < 0m)
            {
                result = 0m;
            }

            if (maxFraction.HasValue && result > maxFraction.Value)
            {
                result = maxFraction.Value;
            }

            if (result > 1m)
            {
                result = 1m;
            }

            return result;
        }

        public static decimal CalculateSequentialStake(decimal bankroll, decimal kellyFraction)
        {
            if (bankroll <= 0m || kellyFraction <= 0m)
            {
                return 0m;
            }

            var stake = bankroll * kellyFraction;
            if (stake <= 0m)
            {
                return 0m;
            }

            if (stake > bankroll)
            {
                stake = bankroll;
            }

            stake = decimal.Round(stake, 2, MidpointRounding.ToZero);
            if (stake < 0.01m)
            {
                return 0m;
            }

            return stake;
        }
        public static decimal CalculateLayKellyFraction(double probability, double layDecimalOdds, decimal? maxFraction = null)
        {
            if (probability <= 0 || probability >= 1 || layDecimalOdds <= 1)
            {
                return 0m;
            }

            var p = probability;
            var q = 1.0 - p;
            var b = layDecimalOdds;
            if (b <= 1.0)
            {
                return 0m;
            }

            var fraction = (q - (p / (b - 1.0))) / q;

            var result = (decimal)fraction;
            if (result <= 0m)
            {
                return 0m;
            }

            if (maxFraction.HasValue && result > maxFraction.Value)
            {
                result = maxFraction.Value;
            }

            if (result > 1m)
            {
                result = 1m;
            }

            return result;
        }

        public static decimal CalculateLayStake(decimal bankroll, decimal layKellyFraction, decimal layDecimalOdds)
        {
            if (bankroll <= 0m || layKellyFraction <= 0m || layDecimalOdds <= 1m)
            {
                return 0m;
            }

            var liability = bankroll * layKellyFraction;
            if (liability <= 0m)
            {
                return 0m;
            }

            var profitMultiple = layDecimalOdds - 1m;
            if (profitMultiple <= 0m)
            {
                return 0m;
            }

            var stake = liability / profitMultiple;
            if (stake <= 0m)
            {
                return 0m;
            }

            stake = decimal.Round(stake, 2, MidpointRounding.ToZero);
            if (stake <= 0m)
            {
                return 0m;
            }

            return stake;
        }
    }
}
