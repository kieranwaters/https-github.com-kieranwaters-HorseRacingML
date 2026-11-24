using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using HorseRacingML.Models;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Shared horse-name and saddlecloth matching for the external result
    /// sources. Names are compared after removing a trailing country/sex
    /// bracket and normalising punctuation, with saddlecloth as a fallback.
    /// </summary>
    public static class RunnerMatcher
    {
        private static readonly Regex BracketSuffix = new(
            @"\s*\([^)]*\)\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        public static string NormalizeHorseName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            string withoutSuffix = BracketSuffix.Replace(value.Trim(), string.Empty);
            return RacingTvDataParser.NormalizeTitle(withoutSuffix);
        }

        public static RaceRunnerReference? FindRunner(
            IReadOnlyList<RaceRunnerReference> runners,
            int? clothNumber,
            string? horseName)
        {
            if (runners is null || runners.Count == 0)
            {
                return null;
            }

            string normalized = NormalizeHorseName(horseName);
            var nameMatches = normalized.Length == 0
                ? new List<RaceRunnerReference>()
                : runners
                    .Where(runner => NormalizeHorseName(runner.HorseName) == normalized)
                    .ToList();

            if (clothNumber.HasValue)
            {
                var byClothAndName = nameMatches.FirstOrDefault(
                    runner => runner.SaddleclothNumber.HasValue &&
                              runner.SaddleclothNumber.Value == clothNumber.Value);
                if (byClothAndName is not null)
                {
                    return byClothAndName;
                }
            }

            if (nameMatches.Count == 1)
            {
                return nameMatches[0];
            }

            if (clothNumber.HasValue)
            {
                var byCloth = runners.FirstOrDefault(
                    runner => runner.SaddleclothNumber.HasValue &&
                              runner.SaddleclothNumber.Value == clothNumber.Value);
                if (byCloth is not null)
                {
                    return byCloth;
                }
            }

            return null;
        }
    }
}
