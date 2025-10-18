using System;
using System.Collections.Generic;
using System.Linq;

namespace HorseRacingML.Models
{
    public class FeaturePopulationSummary
    {
        public static FeaturePopulationSummary Empty { get; } = new FeaturePopulationSummary();

        private int? _populatedCount;

        public int PopulatedCount
        {
            get
            {
                if (PopulatedKeys?.Count > 0)
                {
                    return PopulatedKeys.Count;
                }

                return _populatedCount ?? 0;
            }
            init => _populatedCount = value;
        }

        private int? _missingCount;

        public int MissingCount
        {
            get
            {
                if (MissingKeys?.Count > 0)
                {
                    return MissingKeys.Count;
                }

                return _missingCount ?? 0;
            }
            init => _missingCount = value;
        }

        public IReadOnlyList<string> PopulatedKeys { get; init; } = Array.Empty<string>();

        public IReadOnlyList<string> MissingKeys { get; init; } = Array.Empty<string>();

        private int? _activeEncodedDimensions;

        public int ActiveEncodedDimensions
        {
            get => _activeEncodedDimensions ?? 0;
            init => _activeEncodedDimensions = value;
        }

        private int? _totalEncodedDimensions;

        public int TotalEncodedDimensions
        {
            get => _totalEncodedDimensions ?? 0;
            init => _totalEncodedDimensions = value;
        }

        public int InactiveEncodedDimensions =>
            TotalEncodedDimensions > ActiveEncodedDimensions
                ? TotalEncodedDimensions - ActiveEncodedDimensions
                : 0;
        public int TotalTrackedKeys => PopulatedCount + MissingCount;

        public bool HasData => PopulatedCount > 0 || MissingCount > 0;

        public FeaturePopulationSummary WithSortedKeys()
        {
            if (!HasData)
            {
                return this;
            }

            var populated = PopulatedKeys?.Count > 0
                ? PopulatedKeys.OrderBy(k => k, StringComparer.OrdinalIgnoreCase).ToArray()
                : Array.Empty<string>();
            var missing = MissingKeys?.Count > 0
                ? MissingKeys.OrderBy(k => k, StringComparer.OrdinalIgnoreCase).ToArray()
                : Array.Empty<string>();

            return new FeaturePopulationSummary
            {
                PopulatedCount = populated.Length,
                MissingCount = missing.Length,
                PopulatedKeys = populated,
                MissingKeys = missing,
                ActiveEncodedDimensions = ActiveEncodedDimensions,
                TotalEncodedDimensions = TotalEncodedDimensions
            };
        }
    }
}