namespace HorseRacingML.Models
{
    /// <summary>
    /// Summarises historical distance information for a horse.
    /// </summary>
    /// <param name="LastDistanceYards">The distance in yards of the most recent recorded race.</param>
    /// <param name="AverageDistanceYards">The average distance in yards across all recorded races.</param>
    public readonly record struct HorseDistanceStats(int? LastDistanceYards, double? AverageDistanceYards);
}