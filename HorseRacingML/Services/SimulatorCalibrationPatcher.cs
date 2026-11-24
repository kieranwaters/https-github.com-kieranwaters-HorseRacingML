using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.ML;
using HorseRacingML.Models;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Applies an already-fitted field calibration to the cached simulator
    /// snapshot. The simulator normally reuses cached predictions, so a
    /// calibration applied after a training run would otherwise be invisible
    /// until the snapshot was rebuilt. This patches the existing probabilities
    /// with the same logit transform used by TrainedModelRuntime.
    /// </summary>
    public static class SimulatorCalibrationPatcher
    {
        public static string Apply(
            BettingSimulatorStore store,
            MLParameter parameter,
            TrainedModel model,
            Action<string>? log = null)
        {
            if (store is null) throw new ArgumentNullException(nameof(store));
            if (parameter is null) throw new ArgumentNullException(nameof(parameter));
            if (model is null) throw new ArgumentNullException(nameof(model));
            log ??= Console.WriteLine;

            var buckets = (model.FieldCalibration ?? new List<FieldCalibrationBucket>())
                .OrderBy(bucket => bucket.MaxFieldSize)
                .ToList();
            if (buckets.Count == 0)
            {
                throw new InvalidOperationException("The active model has no field-calibration buckets.");
            }

            var snapshot = store.Get()
                ?? throw new InvalidOperationException("No simulator snapshot is available.");
            if (string.Equals(
                    snapshot.Summary.ModelHash,
                    parameter.ModelHash,
                    StringComparison.OrdinalIgnoreCase))
            {
                log($"[simulator-cal] snapshot {snapshot.Summary.Key} already uses the calibrated model hash; nothing to do.");
                return snapshot.Summary.Key;
            }

            int races = 0;
            int runners = 0;
            if (snapshot.Races is { Count: > 0 })
            {
                foreach (var race in snapshot.Races)
                {
                    if (race?.Runners is null || race.Runners.Count < 2)
                    {
                        continue;
                    }

                    if (ApplyRace(race.Runners, buckets))
                    {
                        races++;
                        runners += race.Runners.Count;
                    }
                }

                snapshot.Runners = snapshot.Races
                    .Where(race => race?.Runners is not null)
                    .SelectMany(race => race!.Runners!)
                    .ToList();
            }
            else if (snapshot.Runners is { Count: > 0 })
            {
                foreach (var group in snapshot.Runners.GroupBy(runner => runner.RaceId))
                {
                    var raceRunners = group.ToList();
                    if (ApplyRace(raceRunners, buckets))
                    {
                        races++;
                        runners += raceRunners.Count;
                    }
                }
            }

            if (races == 0)
            {
                throw new InvalidOperationException("The simulator snapshot contained no calibratable races.");
            }

            var previousSource = snapshot.Summary.Source;
            snapshot.Summary.Source = string.IsNullOrWhiteSpace(previousSource)
                ? "Field-calibrated cached snapshot"
                : $"{previousSource} + field-calibrated";
            snapshot.Summary.ModelArtifact = parameter.ModelArtifact;
            snapshot.Summary.ModelHash = parameter.ModelHash;
            snapshot.Summary.FeatureSchemaHash = parameter.FeatureSchemaHash
                ?? snapshot.Summary.FeatureSchemaHash;
            snapshot.Summary.CreatedUtc = DateTime.UtcNow;

            var key = store.Set(snapshot);
            log(
                $"[simulator-cal] applied {buckets.Count} field bucket(s) to {races:N0} races / " +
                $"{runners:N0} runners; published snapshot {key}");
            return key;
        }

        private static bool ApplyRace(
            IReadOnlyList<SimulatorRunnerRow> raceRunners,
            IReadOnlyList<FieldCalibrationBucket> buckets)
        {
            var bucket = buckets.FirstOrDefault(candidate =>
                raceRunners.Count <= candidate.MaxFieldSize) ?? buckets[^1];
            if (bucket is null ||
                (Math.Abs(bucket.Intercept) < 1e-12 && Math.Abs(bucket.Slope - 1d) < 1e-12))
            {
                return false;
            }

            double sum = 0d;
            var calibrated = new double[raceRunners.Count];
            for (int i = 0; i < raceRunners.Count; i++)
            {
                var runner = raceRunners[i];
                double input = runner.RawProbability > 0d
                    ? runner.RawProbability
                    : runner.Probability;
                double p = Math.Clamp(input, 1e-6, 1d - 1e-6);
                double logit = Math.Log(p / (1d - p));
                double value = 1d / (1d + Math.Exp(-(bucket.Intercept + bucket.Slope * logit)));
                value = Math.Clamp(value, 1e-9, 1d - 1e-9);
                calibrated[i] = value;
                sum += value;
            }

            if (sum <= double.Epsilon || !double.IsFinite(sum))
            {
                return false;
            }

            for (int i = 0; i < raceRunners.Count; i++)
            {
                raceRunners[i].RawProbability = calibrated[i];
                raceRunners[i].Probability = calibrated[i] / sum;
            }

            return true;
        }
    }
}
