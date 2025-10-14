using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.Models;

namespace HorseRacingML.Scraping
{
    internal static class RaceDayReportDeduplicator
    {
        public static List<RaceDayReport> ByMarketId(IEnumerable<RaceDayReport?>? races)
        {
            if (races == null)
            {
                return new List<RaceDayReport>();
            }

            var sourceList = races
                .Where(race => race != null)
                .Cast<RaceDayReport>()
                .ToList();

            if (sourceList.Count == 0)
            {
                return new List<RaceDayReport>();
            }

            var deduplicated = new List<RaceDayReport>(sourceList.Count);
            var seen = new Dictionary<string, (int Index, RaceDayReport Race)>(StringComparer.OrdinalIgnoreCase);

            for (var i = 0; i < sourceList.Count; i++)
            {
                var race = sourceList[i];
                var marketId = race.MarketId;

                if (string.IsNullOrWhiteSpace(marketId))
                {
                    deduplicated.Add(race);
                    continue;
                }

                if (!seen.TryGetValue(marketId, out var existing))
                {
                    seen[marketId] = (deduplicated.Count, race);
                    deduplicated.Add(race);
                    continue;
                }

                if (IsRaceMoreComplete(race, existing.Race))
                {
                    deduplicated[existing.Index] = race;
                    seen[marketId] = (existing.Index, race);
                }
            }

            return deduplicated;
        }

        private static bool IsRaceMoreComplete(RaceDayReport candidate, RaceDayReport existing)
        {
            if (candidate == null)
            {
                return false;
            }

            if (existing == null)
            {
                return true;
            }

            var candidateRunnerCount = candidate.Runners?.Count ?? 0;
            var existingRunnerCount = existing.Runners?.Count ?? 0;

            if (candidateRunnerCount != existingRunnerCount)
            {
                return candidateRunnerCount > existingRunnerCount;
            }

            var candidateScore = CalculateRunnerDataScore(candidate);
            var existingScore = CalculateRunnerDataScore(existing);

            if (candidateScore != existingScore)
            {
                return candidateScore > existingScore;
            }

            return false;
        }

        private static int CalculateRunnerDataScore(RaceDayReport race)
        {
            if (race?.Runners == null || race.Runners.Count == 0)
            {
                return 0;
            }

            var score = 0;
            foreach (var runner in race.Runners)
            {
                if (runner == null)
                {
                    continue;
                }

                if (runner.AiProbability.HasValue)
                {
                    score += 2;
                }

                if (runner.MarketDecimalOdds.HasValue)
                {
                    score++;
                }

                if (runner.LayDecimalOdds.HasValue)
                {
                    score++;
                }

                if (runner.SuggestedStake.HasValue || runner.LaySuggestedStake.HasValue)
                {
                    score++;
                }
            }

            return score;
        }
    }
}