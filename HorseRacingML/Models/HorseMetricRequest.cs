using System;

namespace HorseRacingML.Data
{
    /// <summary>
    /// Represents a lookup request for horse performance metrics when batching repository queries.
    /// </summary>
    public readonly struct HorseMetricRequest : IEquatable<HorseMetricRequest>
    {
        public HorseMetricRequest(int? horseId, string? normalizedHorseName, DateTime? beforeDate, string? rawHorseName)
        {
            HorseId = horseId;
            NormalizedHorseName = string.IsNullOrWhiteSpace(normalizedHorseName)
                ? null
                : normalizedHorseName;
            RawHorseName = rawHorseName;
            BeforeDate = beforeDate?.Date;
        }

        public int? HorseId { get; }

        public string? NormalizedHorseName { get; }

        public string? RawHorseName { get; }

        public DateTime? BeforeDate { get; }

        public bool Equals(HorseMetricRequest other)
        {
            return HorseId == other.HorseId &&
                   string.Equals(NormalizedHorseName, other.NormalizedHorseName, StringComparison.OrdinalIgnoreCase) &&
                   Nullable.Equals(BeforeDate, other.BeforeDate);
        }

        public override bool Equals(object? obj)
        {
            return obj is HorseMetricRequest other && Equals(other);
        }

        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(HorseId.GetValueOrDefault());
            hash.Add(HorseId.HasValue);
            hash.Add(NormalizedHorseName ?? string.Empty, StringComparer.OrdinalIgnoreCase);
            hash.Add(BeforeDate?.Date ?? default);
            return hash.ToHashCode();
        }
    }
}