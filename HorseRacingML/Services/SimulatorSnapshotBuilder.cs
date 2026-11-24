using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.ML;
using HorseRacingML.Models;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Converts trainer predictions into the compact, pre-grouped snapshot used
    /// by the betting simulator. Kept outside the controller so Test AI, Train
    /// AI and the background simulator job all publish the same shape.
    /// </summary>
    public static class SimulatorSnapshotBuilder
    {
        public static BettingSimulationSnapshot Build(
            HyperparameterTrainer.TrainingResult result,
            IReadOnlyList<float> predictions,
            IDictionary<int, RaceSummary> raceSummaries,
            MLParameter parameter,
            DateTime validationStart,
            DateTime validationEnd,
            int? months,
            int trainingRaceCount,
            int validationRaceCount,
            string source,
            bool usedSavedModel = false,
            string? snapshotKey = null)
        {
            if (result is null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            raceSummaries ??= new Dictionary<int, RaceSummary>();
            predictions ??= Array.Empty<float>();

            int count = Math.Min(
                result.ValidationExamples.Count,
                Math.Min(result.ValidationRaceIds.Count, predictions.Count));

            var rows = new List<SimulatorRunnerRow>(count);
            var indicesByRace = new Dictionary<int, List<int>>();
            for (int i = 0; i < count; i++)
            {
                int raceId = result.ValidationRaceIds[i];
                if (!indicesByRace.TryGetValue(raceId, out var indices))
                {
                    indices = new List<int>();
                    indicesByRace[raceId] = indices;
                }

                indices.Add(i);
            }

            var races = new List<SimulatorRace>(indicesByRace.Count);
            foreach (var (raceId, indices) in indicesByRace)
            {
                raceSummaries.TryGetValue(raceId, out var summary);

                double sum = 0d;
                foreach (int index in indices)
                {
                    double probability = predictions[index];
                    if (double.IsFinite(probability) && probability > 0d)
                    {
                        sum += probability;
                    }
                }

                string? raceType = ResolveRaceType(summary?.RaceType, summary?.Title, summary?.Surface);
                int runnerCount = summary?.RunnerCount is byte declared && declared > 0
                    ? declared
                    : indices.Count;

                var raceRows = new List<SimulatorRunnerRow>(indices.Count);
                foreach (int index in indices)
                {
                    var example = result.ValidationExamples[index];
                    double probability = predictions[index];
                    if (!double.IsFinite(probability) || probability < 0d)
                    {
                        probability = 0d;
                    }

                    var row = new SimulatorRunnerRow
                    {
                        RaceId = raceId,
                        RaceDate = summary?.RaceDate,
                        RaceTitle = summary?.Title,
                        CourseName = summary?.CourseName,
                        RaceType = raceType,
                        Class = summary?.Class,
                        AgeRestriction = summary?.AgeRestriction,
                        Surface = summary?.Surface,
                        Going = summary?.Going,
                        DistanceYards = summary?.DistanceYards,
                        DistanceText = summary?.DistanceText,
                        RunnerCount = runnerCount,
                        HorseName = example.HorseName,
                        FinishPosition = example.FinishPosition,
                        DecimalOdds = example.StartingPriceDecimal,
                        BetfairSp = example.BetfairSp,
                        BetfairLastTradedPrice = example.BetfairLastTradedPrice,
                        Probability = sum > 0d ? probability / sum : 0d,
                        RawProbability = probability,
                        Won = example.Label >= 0.5f
                    };

                    rows.Add(row);
                    raceRows.Add(row);
                }

                races.Add(new SimulatorRace
                {
                    RaceId = raceId,
                    RaceDate = summary?.RaceDate,
                    RaceTitle = summary?.Title,
                    CourseName = summary?.CourseName,
                    RaceType = raceType,
                    Class = summary?.Class,
                    AgeRestriction = summary?.AgeRestriction,
                    Surface = summary?.Surface,
                    Going = summary?.Going,
                    DistanceYards = summary?.DistanceYards,
                    DistanceText = summary?.DistanceText,
                    RunnerCount = runnerCount,
                    Runners = raceRows
                });
            }

            static List<string> DistinctValues(IEnumerable<string?> values) =>
                values
                    .Select(value => string.IsNullOrWhiteSpace(value) ? null : value!.Trim())
                    .Where(value => value is not null)
                    .Select(value => value!)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .OrderBy(value => value, StringComparer.OrdinalIgnoreCase)
                    .ToList();

            var pricedRunners = rows.Count(row =>
                row.DecimalOdds is > 1m || row.BetfairSp is > 1m ||
                row.BetfairLastTradedPrice is > 1m);
            var firstRaceDate = races
                .Where(race => race.RaceDate.HasValue)
                .Select(race => race.RaceDate)
                .Min();
            var lastRaceDate = races
                .Where(race => race.RaceDate.HasValue)
                .Select(race => race.RaceDate)
                .Max();

            return new BettingSimulationSnapshot
            {
                Summary = new BettingSimulationSnapshotSummary
                {
                    Key = snapshotKey ?? string.Empty,
                    CreatedUtc = DateTime.UtcNow,
                    Source = source,
                    ValidationStart = validationStart,
                    ValidationEnd = validationEnd,
                    FirstRaceDate = firstRaceDate,
                    LastRaceDate = lastRaceDate,
                    Months = months,
                    TrainingRaceCount = trainingRaceCount,
                    ValidationRaceCount = validationRaceCount,
                    RunnerCount = rows.Count,
                    PricedRunnerCount = pricedRunners,
                    UnpricedRunnerCount = rows.Count - pricedRunners,
                    Objective = parameter?.Objective ?? result.Objective,
                    ModelArtifact = parameter?.ModelArtifact ?? result.ModelArtifactFileName,
                    ModelHash = parameter?.ModelHash ?? result.ModelHash,
                    FeatureSchemaHash = parameter?.FeatureSchemaHash ?? result.FeatureSchemaHash,
                    UsedSavedModel = usedSavedModel,
                    ModelTrainingEnd = parameter?.TestEndDate ?? parameter?.TrainingEndDate,
                    BlendModelWeight = parameter?.BlendModelWeight ?? 1d,
                    BlendMarketWeight = parameter?.BlendMarketWeight ?? 0d,
                    RaceTypes = DistinctValues(rows.Select(row => row.RaceType)),
                    Surfaces = DistinctValues(rows.Select(row => row.Surface)),
                    Goings = DistinctValues(rows.Select(row => row.Going)),
                    Courses = DistinctValues(rows.Select(row => row.CourseName)),
                    AgeRestrictions = DistinctValues(rows.Select(row => row.AgeRestriction)),
                    HasBetfairPrices = rows.Any(row =>
                        row.BetfairSp is > 1m || row.BetfairLastTradedPrice is > 1m)
                },
                Runners = rows,
                Races = races
            };
        }

        /// <summary>
        /// The stored RaceType column is empty in the imported data, so derive a
        /// usable type from the race title for the ROI breakdowns.
        /// </summary>
        public static string ResolveRaceType(string? storedRaceType, string? title, string? surface)
        {
            if (!string.IsNullOrWhiteSpace(storedRaceType))
            {
                return storedRaceType!;
            }

            var value = title ?? string.Empty;
            if (value.Contains("Hurdle", StringComparison.OrdinalIgnoreCase))
            {
                return "Hurdle";
            }

            if (value.Contains("Chase", StringComparison.OrdinalIgnoreCase))
            {
                return "Chase";
            }

            if (value.Contains("NH Flat", StringComparison.OrdinalIgnoreCase) ||
                value.Contains("Flat Race", StringComparison.OrdinalIgnoreCase) ||
                value.Contains("Bumper", StringComparison.OrdinalIgnoreCase))
            {
                return "NH Flat";
            }

            if (value.Contains("Flat", StringComparison.OrdinalIgnoreCase) ||
                !string.IsNullOrWhiteSpace(surface))
            {
                return "Flat";
            }

            return "Unknown";
        }
    }
}
