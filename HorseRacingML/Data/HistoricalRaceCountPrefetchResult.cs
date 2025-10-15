using System.Collections.ObjectModel;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace HorseRacingML.Data
{
    public sealed class HistoricalRaceCountPrefetchResult
    {
        public static HistoricalRaceCountPrefetchResult Empty { get; } = new(
            new ReadOnlyDictionary<string, int>(new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)),
            new ReadOnlyDictionary<string, int>(new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)),
            matchedHorseIdCount: 0);

        public HistoricalRaceCountPrefetchResult(
            IReadOnlyDictionary<string, int> countsByCandidate,
            IReadOnlyDictionary<string, int> countsByOriginal,
            int matchedHorseIdCount)
        {
            CountsByCandidate = countsByCandidate ?? throw new ArgumentNullException(nameof(countsByCandidate));
            CountsByOriginal = countsByOriginal ?? throw new ArgumentNullException(nameof(countsByOriginal));
            MatchedHorseIdCount = matchedHorseIdCount;
            TotalHistoricalRaces = countsByOriginal.Values.Sum();
        }

        public IReadOnlyDictionary<string, int> CountsByCandidate { get; }

        public IReadOnlyDictionary<string, int> CountsByOriginal { get; }

        public int MatchedHorseCount => CountsByOriginal.Count;

        public int MatchedHorseIdCount { get; }

        public int TotalHistoricalRaces { get; }

        public bool IsEmpty => CountsByCandidate.Count == 0;

        public bool TryGetCountByCandidate(string? candidate, out int count)
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                count = 0;
                return false;
            }

            return CountsByCandidate.TryGetValue(candidate, out count);
        }
    }
}