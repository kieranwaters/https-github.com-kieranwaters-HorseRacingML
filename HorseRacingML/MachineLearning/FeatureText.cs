using System;
using System.Text.RegularExpressions;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Canonical text values used by both the historical feature builder and the
    /// live scraper. Keeping these in one place prevents values such as
    /// "Allweather" and "All Weather" from occupying different one-hot slots.
    /// </summary>
    internal static class FeatureText
    {
        private static readonly Regex Whitespace = new("\\s+", RegexOptions.Compiled);
        private static readonly Regex NonAlphanumeric = new("[^a-z0-9]+", RegexOptions.Compiled);
        private static readonly Regex Bracketed = new("\\s*\\([^\\)]*\\)|\\s*\\[[^\\]]*\\]", RegexOptions.Compiled);

        public static string NormalizeCategory(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return "unknown";
            }

            var collapsed = Whitespace.Replace(value.Trim(), " ");
            return collapsed.ToLowerInvariant();
        }

        public static string NormalizeSurface(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return "unknown";
            }

            var normalized = NormalizeCategory(value);
            if (normalized.Contains("all weather") ||
                normalized.Contains("allweather") ||
                normalized.Contains("all-weather") ||
                normalized == "aw" ||
                normalized.Contains("polytrack") ||
                normalized.Contains("tapeta") ||
                normalized.Contains("fibresand"))
            {
                return "allweather";
            }

            if (normalized.Contains("dirt"))
            {
                return "dirt";
            }

            if (normalized.Contains("turf"))
            {
                return "turf";
            }

            return normalized;
        }

        public static string NormalizeHorseName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var withoutBracketed = Bracketed.Replace(value, " ");
            var normalized = NonAlphanumeric.Replace(withoutBracketed.Trim().ToLowerInvariant(), " ");
            return Whitespace.Replace(normalized, " ").Trim();
        }
    }
}
