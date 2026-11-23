using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// The opening, shortest and longest observed decimal prices for a runner.
    /// Tokens are the original fractional strings (for example "7/4" or
    /// "Evens") so they can be stored in the existing RunnerResult columns.
    /// </summary>
    internal readonly record struct MarketOddsRange(
        string? OpeningToken,
        string? LowToken,
        string? HighToken,
        int PointCount)
    {
        public bool HasRange => PointCount > 1 && LowToken is not null && HighToken is not null;
    }

    /// <summary>
    /// Parses the odds tokens used by Sporting Life. The result pages expose
    /// both the short "op ... tchd ..." comment and a longer
    /// betting.historical_odds array. The array is the important one for
    /// high/range features: the comment only carries one touched price.
    /// </summary>
    internal static class MarketOddsParser
    {
        private const string OddsTokenPattern =
            @"(?:evens?|evs|\d+\s*/\s*\d+[A-Za-z]*|\d+(?:\.\d+)?)";

        private static readonly Regex OddsTokenRegex = new(
            OddsTokenPattern,
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex OpeningTokenRegex = new(
            @"\bop\s+(?<token>" + OddsTokenPattern + @")",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex TouchedLowTokenRegex = new(
            @"\b(?:tchd|low)\s+(?<token>" + OddsTokenPattern + @")",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex TouchedHighTokenRegex = new(
            @"\bhigh\s+(?<token>" + OddsTokenPattern + @")",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex FavouriteSuffixRegex = new(
            @"(?:JF|CF|F|J)$",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex FractionRegex = new(
            @"^(?<numerator>\d+)\s*/\s*(?<denominator>\d+)$",
            RegexOptions.Compiled);

        /// <summary>
        /// Converts a fractional or decimal odds token to decimal odds
        /// (including the stake). "Evens"/"Evs" maps to 2.0.
        /// </summary>
        internal static decimal? Parse(string? text)
        {
            var token = NormalizeToken(text);
            if (string.IsNullOrWhiteSpace(token))
            {
                return null;
            }

            if (token.Equals("EVS", StringComparison.OrdinalIgnoreCase) ||
                token.Equals("EVENS", StringComparison.OrdinalIgnoreCase) ||
                token.Equals("EVEN", StringComparison.OrdinalIgnoreCase))
            {
                return 2m;
            }

            var fraction = FractionRegex.Match(token);
            if (fraction.Success &&
                decimal.TryParse(
                    fraction.Groups["numerator"].Value,
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out var numerator) &&
                decimal.TryParse(
                    fraction.Groups["denominator"].Value,
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out var denominator) &&
                denominator > 0m)
            {
                return 1m + (numerator / denominator);
            }

            if (decimal.TryParse(token, NumberStyles.Number, CultureInfo.InvariantCulture, out var decimalOdds) &&
                decimalOdds > 1m)
            {
                return decimalOdds;
            }

            return null;
        }

        /// <summary>
        /// Removes favourite markers and surrounding noise while preserving the
        /// fractional spelling used by the source.
        /// </summary>
        internal static string? NormalizeToken(string? text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return null;
            }

            var token = text.Trim().Replace('\u00A0', ' ');
            token = FavouriteSuffixRegex.Replace(token, string.Empty).Trim();
            if (token.Length == 0)
            {
                return null;
            }

            if (token.Equals("EVS", StringComparison.OrdinalIgnoreCase) ||
                token.Equals("EVENS", StringComparison.OrdinalIgnoreCase) ||
                token.Equals("EVEN", StringComparison.OrdinalIgnoreCase))
            {
                return "Evens";
            }

            var fraction = Regex.Match(token, @"\d+\s*/\s*\d+", RegexOptions.IgnoreCase);
            if (fraction.Success)
            {
                return Regex.Replace(fraction.Value, @"\s+", string.Empty);
            }

            if (decimal.TryParse(token, NumberStyles.Number, CultureInfo.InvariantCulture, out var decimalOdds) &&
                decimalOdds > 1m)
            {
                return decimalOdds.ToString(CultureInfo.InvariantCulture);
            }

            return null;
        }

        internal static string? ExtractOpeningToken(string? text)
        {
            var match = OpeningTokenRegex.Match(text ?? string.Empty);
            return match.Success ? NormalizeToken(match.Groups["token"].Value) : null;
        }

        internal static string? ExtractToken(string? text, string key)
        {
            if (string.IsNullOrWhiteSpace(text) || string.IsNullOrWhiteSpace(key))
            {
                return null;
            }

            if (key.Equals("op", StringComparison.OrdinalIgnoreCase))
            {
                return ExtractOpeningToken(text);
            }

            var match = Regex.Match(
                text,
                $@"\b{Regex.Escape(key)}\s+(?<token>{OddsTokenPattern})",
                RegexOptions.IgnoreCase);
            return match.Success ? NormalizeToken(match.Groups["token"].Value) : null;
        }

        internal static (string? low, string? high) ExtractTouchedTokens(string? text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return (null, null);
            }

            var lowMatch = TouchedLowTokenRegex.Match(text);
            var highMatch = TouchedHighTokenRegex.Match(text);
            return (
                lowMatch.Success ? NormalizeToken(lowMatch.Groups["token"].Value) : null,
                highMatch.Success ? NormalizeToken(highMatch.Groups["token"].Value) : null);
        }

        /// <summary>
        /// Builds the opening/shortest/longest range from the full historical
        /// odds array. The explicit "op" and "tchd" tokens are included as
        /// candidates because the array can be sampled or omit the exact
        /// touched point. High is only returned when at least two distinct
        /// prices were observed, so a single touched price cannot masquerade as
        /// a full range.
        /// </summary>
        internal static MarketOddsRange BuildRange(
            IEnumerable<string?>? historicalOdds,
            string? openingToken = null,
            string? touchedLowToken = null,
            string? touchedHighToken = null)
        {
            var points = new List<(string Token, decimal Price)>();
            if (historicalOdds is not null)
            {
                foreach (var raw in historicalOdds)
                {
                    AddPoint(points, raw);
                }
            }

            AddPoint(points, openingToken);
            AddPoint(points, touchedLowToken);
            AddPoint(points, touchedHighToken);

            // The opening/touched tokens usually also appear in the historical
            // series. Count distinct price levels so the point count describes
            // the observed range rather than the number of duplicate inputs.
            points = points
                .GroupBy(point => point.Price)
                .Select(group => group.First())
                .ToList();

            if (points.Count == 0)
            {
                return new MarketOddsRange(null, null, null, 0);
            }

            var opening = NormalizeToken(openingToken);
            if (opening is null || Parse(opening) is null)
            {
                opening = points[0].Token;
            }

            (string Token, decimal Price)? low = null;
            (string Token, decimal Price)? high = null;
            foreach (var point in points)
            {
                if (low is null || point.Price < low.Value.Price)
                {
                    low = point;
                }
                if (high is null || point.Price > high.Value.Price)
                {
                    high = point;
                }
            }

            var hasRange = low is not null &&
                           high is not null &&
                           high.Value.Price > low.Value.Price;
            return new MarketOddsRange(
                opening,
                low?.Token,
                hasRange ? high?.Token : null,
                points.Count);
        }

        private static void AddPoint(
            ICollection<(string Token, decimal Price)> points,
            string? raw)
        {
            var token = NormalizeToken(raw);
            if (token is null)
            {
                return;
            }

            var price = Parse(token);
            if (price.HasValue && price.Value > 1m)
            {
                points.Add((token, price.Value));
            }
        }
    }
}
