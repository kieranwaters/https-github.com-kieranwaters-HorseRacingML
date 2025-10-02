using System;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Services
{
    public enum MaxStakeMode
    {
        None,
        PercentageOfBankroll,
        FixedAmount
    }

    public record AutomationSettingsSnapshot(
        decimal KellyDampener,
        decimal? MaxKellyFraction,
        MaxStakeMode MaxStakeMode,
        decimal? MaxStakePercentOfBankroll,
        decimal? MaxStakeFixedAmount);

    public record AutomationSettingsUpdate(
        decimal KellyDampener,
        decimal? MaxKellyFraction,
        MaxStakeMode MaxStakeMode,
        decimal? MaxStakePercentOfBankroll,
        decimal? MaxStakeFixedAmount);

    public class AutomationSettingsService
    {
        private readonly object _syncRoot = new();
        private decimal _kellyDampener;
        private decimal? _maxKellyFraction;
        private MaxStakeMode _maxStakeMode;
        private decimal? _maxStakePercentOfBankroll;
        private decimal? _maxStakeFixedAmount;

        public AutomationSettingsService(IConfiguration configuration)
        {
            if (configuration == null)
            {
                throw new ArgumentNullException(nameof(configuration));
            }

            _kellyDampener = NormalizeKelly(configuration.GetValue<decimal?>("Betting:KellyDampener"));
            _maxKellyFraction = NormalizeFraction(configuration.GetValue<decimal?>("Betting:MaxKellyFraction"));
            _maxStakeMode = ParseMaxStakeMode(configuration["Betting:MaxStakeMode"]);
            _maxStakePercentOfBankroll = NormalizeFraction(configuration.GetValue<decimal?>("Betting:MaxStakePercent"));
            _maxStakeFixedAmount = NormalizeNonNegative(configuration.GetValue<decimal?>("Betting:MaxStakeFixedAmount"));
        }

        public AutomationSettingsSnapshot GetSnapshot()
        {
            lock (_syncRoot)
            {
                return new AutomationSettingsSnapshot(
                    _kellyDampener,
                    _maxKellyFraction,
                    _maxStakeMode,
                    _maxStakePercentOfBankroll,
                    _maxStakeFixedAmount);
            }
        }

        public AutomationSettingsSnapshot UpdateSettings(AutomationSettingsUpdate update)
        {
            if (update == null)
            {
                throw new ArgumentNullException(nameof(update));
            }

            lock (_syncRoot)
            {
                _kellyDampener = NormalizeKelly(update.KellyDampener);
                _maxKellyFraction = update.MaxKellyFraction.HasValue
                    ? NormalizeFraction(update.MaxKellyFraction)
                    : null;

                _maxStakeMode = update.MaxStakeMode;
                _maxStakePercentOfBankroll = update.MaxStakePercentOfBankroll.HasValue
                    ? NormalizeFraction(update.MaxStakePercentOfBankroll)
                    : null;
                _maxStakeFixedAmount = update.MaxStakeFixedAmount.HasValue
                    ? NormalizeNonNegative(update.MaxStakeFixedAmount)
                    : null;

                return new AutomationSettingsSnapshot(
                    _kellyDampener,
                    _maxKellyFraction,
                    _maxStakeMode,
                    _maxStakePercentOfBankroll,
                    _maxStakeFixedAmount);
            }
        }

        private static decimal NormalizeKelly(decimal? value)
        {
            if (!value.HasValue)
            {
                return 1m;
            }

            var sanitized = value.Value;
            if (sanitized < 0m)
            {
                sanitized = 0m;
            }

            if (sanitized > 1m)
            {
                sanitized = 1m;
            }

            return decimal.Round(sanitized, 3, MidpointRounding.ToZero);
        }

        private static decimal? NormalizeFraction(decimal? value)
        {
            if (!value.HasValue)
            {
                return null;
            }

            var sanitized = value.Value;
            if (sanitized < 0m)
            {
                sanitized = 0m;
            }

            if (sanitized > 1m)
            {
                sanitized = 1m;
            }

            return decimal.Round(sanitized, 4, MidpointRounding.ToZero);
        }

        private static decimal? NormalizeNonNegative(decimal? value)
        {
            if (!value.HasValue)
            {
                return null;
            }

            var sanitized = value.Value;
            if (sanitized < 0m)
            {
                sanitized = 0m;
            }

            return decimal.Round(sanitized, 2, MidpointRounding.ToZero);
        }

        private static MaxStakeMode ParseMaxStakeMode(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return MaxStakeMode.None;
            }

            if (Enum.TryParse<MaxStakeMode>(value, true, out var parsed))
            {
                return parsed;
            }

            return value.Trim().ToLowerInvariant() switch
            {
                "percent" or "percentage" or "percentageofbankroll" or "percentbankroll" => MaxStakeMode.PercentageOfBankroll,
                "fixed" or "amount" or "cash" or "fixedamount" => MaxStakeMode.FixedAmount,
                _ => MaxStakeMode.None
            };
        }
    }
}