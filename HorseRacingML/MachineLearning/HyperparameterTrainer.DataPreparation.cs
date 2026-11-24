using Dapper;
using Microsoft.Data.SqlClient;
using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Globalization;
using System.Linq;
using System.Text;
using static HorseRacingML.ML.HyperparameterTrainer.TrainingDataset;
using static HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;

namespace HorseRacingML.ML
{
    public partial class HyperparameterTrainer
    {
        public class TrainingDataset
        {
            public TrainingDataset(
                List<RaceExample> trainingRaces,
                List<RaceExample> validationRaces,
                List<string> featureKeys,
                Dictionary<string, int> featureDimensions,
                Dictionary<string, Dictionary<string, int>> stringMaps,
                NormalizationParameters normalization)
            {
                TrainingRaces = trainingRaces ?? throw new ArgumentNullException(nameof(trainingRaces));
                ValidationRaces = validationRaces ?? throw new ArgumentNullException(nameof(validationRaces));
                FeatureKeys = featureKeys ?? throw new ArgumentNullException(nameof(featureKeys));
                FeatureDimensions = featureDimensions ?? throw new ArgumentNullException(nameof(featureDimensions));
                StringMaps = stringMaps ?? throw new ArgumentNullException(nameof(stringMaps));
                Normalization = normalization ?? throw new ArgumentNullException(nameof(normalization));

                Races = TrainingRaces.Concat(ValidationRaces).ToList();
                FeatureCount = featureDimensions.Values.Sum();
            }

            public List<RaceExample> TrainingRaces { get; }
            public List<RaceExample> ValidationRaces { get; }
            public List<RaceExample> Races { get; }
            public List<string> FeatureKeys { get; }
            public Dictionary<string, int> FeatureDimensions { get; }
            public Dictionary<string, Dictionary<string, int>> StringMaps { get; }
            public NormalizationParameters Normalization { get; }
            public int FeatureCount { get; }
            public int RowCount => Races.Sum(r => r.Runners.Count);
            public int TrainingRowCount => TrainingRaces.Sum(r => r.Runners.Count);
            public int ValidationRowCount => ValidationRaces.Sum(r => r.Runners.Count);

            public class PreparedDataset
            {
                public PreparedDataset(List<PreparedRace> races)
                {
                    Races = races ?? throw new ArgumentNullException(nameof(races));
                }

                public List<PreparedRace> Races { get; }
                public IEnumerable<Dictionary<string, object?>> Rows => Races.SelectMany(r => r.Rows);
                public int RowCount => Races.Sum(r => r.Rows.Count);

                internal static bool TryGetRequiredInt32(Dictionary<string, object?> row, string key, out int result)
                {
                    if (row.TryGetValue(key, out var value) &&
                        value is not null &&
                        TryConvertToInt32(value, out result))
                    {
                        return true;
                    }

                    result = 0;
                    return false;
                }

                public static HashSet<string> LoadColumnNames(SqlConnection conn, string tableName)
                {
                    if (conn is null)
                        throw new ArgumentNullException(nameof(conn));
                    if (string.IsNullOrWhiteSpace(tableName))
                        throw new ArgumentException("Table name must be provided.", nameof(tableName));

                    var columns = conn.Query<string>(
                        @"SELECT COLUMN_NAME
                  FROM INFORMATION_SCHEMA.COLUMNS
                  WHERE TABLE_NAME = @TableName",
                        new { TableName = tableName });

                    return new HashSet<string>(columns, StringComparer.OrdinalIgnoreCase);
                }

                internal static int GetRequiredInt32(Dictionary<string, object?> row, string key)
                {
                    if (!row.TryGetValue(key, out var value) || value is null)
                    {
                        throw new InvalidOperationException($"Missing required column '{key}'.");
                    }

                    if (TryConvertToInt32(value, out var result))
                    {
                        return result;
                    }

                    string typeName = value.GetType().FullName ?? value.GetType().Name;
                    string displayValue = value switch
                    {
                        byte[] bytes => $"0x{BitConverter.ToString(bytes).Replace("-", string.Empty)}",
                        _ => value.ToString() ?? string.Empty
                    };

                    throw new FormatException($"Column '{key}' with value '{displayValue}' of type '{typeName}' could not be converted to Int32.");
                }

                internal static bool TryConvertToInt32(object? value, out int result)
                {
                    switch (value)
                    {
                        case null:
                            result = 0;
                            return false;
                        case int i:
                            result = i;
                            return true;
                        case long l when l >= int.MinValue && l <= int.MaxValue:
                            result = (int)l;
                            return true;
                        case short s:
                            result = s;
                            return true;
                        case sbyte sb:
                            result = sb;
                            return true;
                        case byte b:
                            result = b;
                            return true;
                        case ushort us when us <= int.MaxValue:
                            result = us;
                            return true;
                        case uint ui when ui <= int.MaxValue:
                            result = (int)ui;
                            return true;
                        case decimal dec when dec >= int.MinValue && dec <= int.MaxValue:
                            result = (int)dec;
                            return true;
                        case double dbl when dbl >= int.MinValue && dbl <= int.MaxValue:
                            result = (int)dbl;
                            return true;
                        case float fl when fl >= int.MinValue && fl <= int.MaxValue:
                            result = (int)fl;
                            return true;
                        case SqlInt32 sqlInt when !sqlInt.IsNull:
                            result = sqlInt.Value;
                            return true;
                        case SqlInt16 sqlShort when !sqlShort.IsNull:
                            result = sqlShort.Value;
                            return true;
                        case SqlInt64 sqlLong when !sqlLong.IsNull && sqlLong.Value >= int.MinValue && sqlLong.Value <= int.MaxValue:
                            result = (int)sqlLong.Value;
                            return true;
                        case SqlDecimal sqlDecimal when !sqlDecimal.IsNull && sqlDecimal.Value >= int.MinValue && sqlDecimal.Value <= int.MaxValue:
                            result = decimal.ToInt32(sqlDecimal.Value);
                            return true;
                        case SqlDouble sqlDouble when !sqlDouble.IsNull && sqlDouble.Value >= int.MinValue && sqlDouble.Value <= int.MaxValue:
                            result = (int)sqlDouble.Value;
                            return true;
                        case SqlSingle sqlSingle when !sqlSingle.IsNull && sqlSingle.Value >= int.MinValue && sqlSingle.Value <= int.MaxValue:
                            result = (int)sqlSingle.Value;
                            return true;
                        case SqlMoney sqlMoney when !sqlMoney.IsNull && sqlMoney.Value >= int.MinValue && sqlMoney.Value <= int.MaxValue:
                            result = (int)sqlMoney.Value;
                            return true;
                        case SqlByte sqlByte when !sqlByte.IsNull:
                            result = sqlByte.Value;
                            return true;
                        case SqlBoolean sqlBool when !sqlBool.IsNull:
                            result = sqlBool.Value ? 1 : 0;
                            return true;
                        case SqlString sqlString when !sqlString.IsNull:
                            return TryConvertToInt32(sqlString.Value, out result);
                        case SqlChars sqlChars when !sqlChars.IsNull:
                            return TryConvertToInt32(sqlChars.Value is null ? null : new string(sqlChars.Value), out result);
                        case SqlBinary sqlBinary when !sqlBinary.IsNull:
                            return TryConvertFromBytes(sqlBinary.Value, out result);
                        case SqlBytes sqlBytes when !sqlBytes.IsNull:
                            return TryConvertFromBytes(sqlBytes.Value, out result);
                        case string str:
                            if (int.TryParse(str, NumberStyles.Integer, CultureInfo.InvariantCulture, out result))
                            {
                                return true;
                            }
                            if (int.TryParse(str, NumberStyles.Integer, CultureInfo.CurrentCulture, out result))
                            {
                                return true;
                            }

                            string trimmed = str.Trim();
                            if (trimmed.Length > 0)
                            {
                                var filtered = new string(trimmed.Where(c => char.IsDigit(c) || c == '-' || c == '+').ToArray());
                                if (!string.IsNullOrWhiteSpace(filtered) &&
                                    int.TryParse(filtered, NumberStyles.Integer, CultureInfo.InvariantCulture, out result))
                                {
                                    return true;
                                }

                                if (TryConvertFromBytes(Encoding.Unicode.GetBytes(trimmed), out result))
                                {
                                    return true;
                                }
                            }
                            break;
                        case byte[] bytes:
                            if (TryConvertFromBytes(bytes, out result))
                            {
                                return true;
                            }
                            break;
                        case ReadOnlyMemory<byte> memory:
                            if (TryConvertFromBytes(memory.Span, out result))
                            {
                                return true;
                            }
                            break;
                        case IConvertible convertible:
                            try
                            {
                                result = convertible.ToInt32(CultureInfo.InvariantCulture);
                                return true;
                            }
                            catch
                            {
                                break;
                            }
                    }

                    result = 0;
                    return false;
                }

                private static bool TryConvertFromBytes(byte[]? bytes, out int value)
                {
                    if (bytes is null)
                    {
                        value = 0;
                        return false;
                    }

                    return TryConvertFromBytes(bytes.AsSpan(), out value);
                }

                private static bool TryConvertFromBytes(ReadOnlySpan<byte> bytes, out int value)
                {
                    if (bytes.IsEmpty)
                    {
                        value = 0;
                        return false;
                    }

                    string asString = Encoding.UTF8.GetString(bytes).Trim('\0', ' ', '\t', '\r', '\n');
                    if (asString.Length > 0)
                    {
                        if (int.TryParse(asString, NumberStyles.Integer, CultureInfo.InvariantCulture, out value) ||
                            int.TryParse(asString, NumberStyles.Integer, CultureInfo.CurrentCulture, out value))
                        {
                            return true;
                        }
                    }

                    if (bytes.Length <= 4)
                    {
                        var padded = new byte[4];
                        bytes.CopyTo(padded);
                        value = BitConverter.ToInt32(padded, 0);
                        return true;
                    }

                    value = 0;
                    return false;
                }

                public class PreparedRace
                {
                    public PreparedRace(int raceId, List<Dictionary<string, object?>> rows)
                    {
                        RaceId = raceId;
                        Rows = rows;
                    }

                    public int RaceId { get; }
                    public List<Dictionary<string, object?>> Rows { get; }
                    public List<Dictionary<string, object?>> Runners => Rows;
                }
            }
        }

        public class RaceExample
        {
            public RaceExample(int raceId, DateTime raceDate, List<RunnerExample> runners)
            {
                RaceId = raceId;
                RaceDate = raceDate;
                Runners = runners;
            }

            public int RaceId { get; }
            public DateTime RaceDate { get; }
            public List<RunnerExample> Runners { get; }
        }

        public class RunnerExample
        {
            public RunnerExample(
                int raceId,
                float[] features,
                float label,
                int? horseId,
                string? horseName,
                decimal? startingPriceDecimal,
                float marketLogit = 0f,
                bool hasMarketPrice = false,
                float marketProbability = 0f,
                float marketLogProbability = 0f,
                int? finishPosition = null,
                int fieldSize = 0,
                decimal? betfairSp = null,
                decimal? betfairLastTradedPrice = null)
            {
                RaceId = raceId;
                Features = features;
                Label = label;
                HorseId = horseId;
                HorseName = horseName;
                StartingPriceDecimal = startingPriceDecimal;
                BetfairSp = betfairSp;
                BetfairLastTradedPrice = betfairLastTradedPrice;
                MarketLogit = marketLogit;
                HasMarketPrice = hasMarketPrice;
                MarketProbability = marketProbability;
                MarketLogProbability = marketLogProbability;
                FinishPosition = finishPosition;
                FieldSize = fieldSize;

                // Normalise the finishing position to [0, 1] with the winner at 1
                // and the last finisher at 0. The auxiliary finish head predicts
                // this score, so it learns to separate the whole field rather than
                // only the winner. Non-finishers (unseated, pulled up, etc.) and
                // inconsistent positions stay masked out of that loss.
                if (finishPosition.HasValue &&
                    fieldSize > 0 &&
                    finishPosition.Value >= 1 &&
                    finishPosition.Value <= fieldSize)
                {
                    HasFinishTarget = true;
                    FinishTarget = fieldSize == 1
                        ? 1f
                        : (float)(fieldSize - finishPosition.Value) / (fieldSize - 1);
                }
            }

            public int RaceId { get; }
            public float[] Features { get; }
            public float Label { get; }
            public int? HorseId { get; }
            public string? HorseName { get; }
            public decimal? StartingPriceDecimal { get; }
            /// <summary>
            /// Betfair starting price, or null when the market has no BSP.
            /// Stored separately from the industry SP so the simulator can
            /// model exchange execution and closing-line value.
            /// </summary>
            public decimal? BetfairSp { get; }
            /// <summary>
            /// Last traded price before the off, or null when unavailable.
            /// </summary>
            public decimal? BetfairLastTradedPrice { get; }
            /// <summary>
            /// Official finishing position, or null for a non-finisher / missing
            /// result. Only used by the auxiliary finish loss.
            /// </summary>
            public int? FinishPosition { get; }
            /// <summary>
            /// Number of declared runners in the race. Used to normalise the
            /// finishing position to the [0, 1] target space.
            /// </summary>
            public int FieldSize { get; }
            /// <summary>
            /// Normalised finish target: 1 for the winner, 0 for the last
            /// finisher. Meaningful only when <see cref="HasFinishTarget"/> is true.
            /// </summary>
            public float FinishTarget { get; }
            /// <summary>
            /// True when a valid finishing position was available. Rows without a
            /// finish are excluded from the auxiliary finish loss.
            /// </summary>
            public bool HasFinishTarget { get; }
            /// <summary>
            /// No-vig market log-odds known before the off. Used as an additive
            /// offset so the network only has to learn the residual edge over
            /// the market rather than reproducing the price from scratch.
            /// </summary>
            public float MarketLogit { get; }
            /// <summary>
            /// No-vig market probability for this runner, or 0 when the runner
            /// had no usable pre-off price. This is the probability space used
            /// by the market blend and the fair market baseline.
            /// </summary>
            public float MarketProbability { get; }
            /// <summary>
            /// Per-race centred log market probability used as the RaceSoftmax
            /// prior. Missing-price runners receive the mean log-probability of
            /// the priced runners (a neutral runner) rather than 0, which would
            /// be equivalent to giving them a 50% win chance.
            /// </summary>
            public float MarketLogProbability { get; }
            /// <summary>
            /// True when the runner had a usable pre-off price. Used to report
            /// deployment-relevant metrics on the priced subset separately from
            /// full-field metrics that include missing-market runners.
            /// </summary>
            public bool HasMarketPrice { get; }
        }

        private sealed class DatasetFeatureMetadata
        {
            public DatasetFeatureMetadata(
                List<string> featureKeys,
                Dictionary<string, int> featureDimensions,
                Dictionary<string, Dictionary<string, int>> stringMaps)
            {
                FeatureKeys = featureKeys;
                FeatureDimensions = featureDimensions;
                StringMaps = stringMaps;
                FeatureCount = featureDimensions.Values.Sum();
            }

            public List<string> FeatureKeys { get; }
            public Dictionary<string, int> FeatureDimensions { get; }
            public Dictionary<string, Dictionary<string, int>> StringMaps { get; }
            public int FeatureCount { get; }
        }

        private static bool TryConvertFromBytes(byte[]? bytes, out int value)
        {
            if (bytes is null)
            {
                value = 0;
                return false;
            }

            return TryConvertFromBytes(bytes.AsSpan(), out value);
        }

        private static bool TryConvertFromBytes(ReadOnlySpan<byte> bytes, out int value)
        {
            if (bytes.IsEmpty)
            {
                value = 0;
                return false;
            }

            string asString = Encoding.UTF8.GetString(bytes).Trim('\0', ' ', '\t', '\r', '\n');
            if (asString.Length > 0)
            {
                if (int.TryParse(asString, NumberStyles.Integer, CultureInfo.InvariantCulture, out value) ||
                    int.TryParse(asString, NumberStyles.Integer, CultureInfo.CurrentCulture, out value))
                {
                    return true;
                }
            }

            if (bytes.Length <= 4)
            {
                var padded = new byte[4];
                bytes.CopyTo(padded);
                value = BitConverter.ToInt32(padded, 0);
                return true;
            }

            value = 0;
            return false;
        }

        /// <summary>
        /// Writes one encoded feature directly into the destination vector. The
        /// previous implementation allocated a temporary array per key and copied
        /// it, producing tens of millions of short-lived arrays per full dataset.
        /// </summary>
        private static void EncodeFeatureInto(
            string key,
            object? value,
            int dim,
            IDictionary<string, Dictionary<string, int>> stringMaps,
            float[] destination,
            int offset)
        {
            int baseDim = stringMaps.TryGetValue(key, out var map)
                ? map.Count
                : 1;
            bool hasMissingIndicator = dim > baseDim;

            if (value == null)
            {
                if (hasMissingIndicator)
                {
                    destination[offset + baseDim] = 1f;
                }
                return;
            }

            switch (value)
            {
                case DateTime dt:
                    destination[offset] = (float)(dt - BaseDate).TotalDays;
                    return;
                case TimeSpan ts:
                    destination[offset] = (float)ts.TotalSeconds;
                    return;
                case string s:
                    EncodeStringInto(key, s, baseDim, stringMaps, destination, offset);
                    return;
                case bool b:
                    destination[offset] = b ? 1f : 0f;
                    return;
                default:
                    float number = Convert.ToSingle(value);
                    if (Math.Abs(number) > 1_000_000f)
                    {
                        number /= 1_000_000f;
                    }
                    destination[offset] = number;
                    return;
            }
        }

        private static void EncodeStringInto(
            string key,
            string value,
            int baseDim,
            IDictionary<string, Dictionary<string, int>> stringMaps,
            float[] destination,
            int offset)
        {
            var normalized = string.Equals(key, "Surface", StringComparison.OrdinalIgnoreCase)
                ? FeatureText.NormalizeSurface(value)
                : FeatureText.NormalizeCategory(value);

            if (stringMaps.TryGetValue(key, out var map) && map.TryGetValue(normalized, out var index))
            {
                destination[offset + index] = 1f;
            }
            else
            {
                // Reserve the last index for unknown categories.
                destination[offset + baseDim - 1] = 1f;
            }
        }

        private static readonly HashSet<string> ExcludedFeatureKeys = new(StringComparer.OrdinalIgnoreCase)
        {
            "FinishPos", "RaceId", "HorseId", "CourseId", "TrainerId", "JockeyId",
            "OutcomeCode", "DistanceBeatenText", "DistanceBeatenLengths", "DistanceBeatenKnown",
            "SP_Fraction", "SP_Decimal", "OpeningFraction", "TouchedHighFraction", "TouchedLowFraction",
            "MarketDecimalOdds", "MarketMoveEarlyPrice", "MarketMoveLowPrice", "MarketMoveHighPrice",
            "BetfairSelectionId", "BetfairSP", "BetfairPreOffHigh", "BetfairPreOffLow",
            "BetfairLastTradedPrice", "BetfairTradedVolume", "BetfairProjectedSP",
            "MarketAnchorPrice", "MarketAnchorCapturedUtc",
            "BetfairAvailableToBack", "BetfairAvailableToLay", "BetfairTotalMatched",
            "BetfairMarketId", "BetfairMarketName", "BetfairBspReconciled",
            "HorseName", "JockeyName", "TrainerName", "Title", "RaceMonth", "RaceDayOfWeek", "Season",
            "ActualOff", "ScheduledOff", "RaceDate", "CourseName", "DistanceText", "Status",
            "WeightText", "FavTag", "SaddleclothNumber", "Purse",
            "Comment", "EarlyPosition", "HeldUp", "Keen", "SlowStart",
            "CurrentSectionalCount", "CurrentSectionalLast1FSeconds", "CurrentSectionalLast2FSeconds",
            "CurrentSectionalLast3FSeconds", "CurrentSectionalLast1FPercentile",
            "CurrentSectionalLast2FPercentile", "CurrentSectionalLast3FPercentile",
            "CurrentSectionalEarlyPositionNorm", "CurrentSectionalClosingPositionNorm",
            "CurrentSectionalPaceStyle", "CurrentSectionalFinishSpeedPct",
            "CurrentSectionalTimeIndexScore", "CurrentSectionalMeetingAverageTimeIndex",
            "CurrentSectionalWinTimeVsPar", "SireName", "DamName", "DamSireName",
            "SourceHorseId", "SourceSlug"
        };

        private sealed class FeatureAccumulator
        {
            public bool HasMissing;
            public object? Sample;
            public HashSet<string>? DistinctStrings;
        }

        /// <summary>
        /// Builds the feature schema incrementally, one runner row at a time.
        /// The previous implementation kept every dictionary row in memory just
        /// to discover the keys and category values; this accumulator lets the
        /// caller throw each race away as soon as it has been observed.
        /// </summary>
        private sealed class FeatureMetadataAccumulator
        {
            private readonly HashSet<string> _keys = new(StringComparer.OrdinalIgnoreCase);
            private readonly Dictionary<string, FeatureAccumulator> _accumulators =
                new(StringComparer.OrdinalIgnoreCase);
            private readonly bool _marketBlind;
            private readonly FeatureBlocks _featureBlocks;
            private int _rowCount;

            public FeatureMetadataAccumulator(bool marketBlind, FeatureBlocks featureBlocks)
            {
                _marketBlind = marketBlind;
                _featureBlocks = featureBlocks;
            }

            public void ObserveRow(Dictionary<string, object?> row)
            {
                if (row is null)
                    throw new ArgumentNullException(nameof(row));

                foreach (var key in row.Keys)
                {
                    if (ExcludedFeatureKeys.Contains(key) ||
                        !IsFeatureEnabled(key, _featureBlocks) ||
                        (_marketBlind && IsMarketDerivedFeature(key)))
                    {
                        continue;
                    }

                    if (_keys.Add(key))
                    {
                        var accumulator = new FeatureAccumulator();
                        // A key first seen after the first row was absent from
                        // every earlier row, so it needs a missing indicator.
                        accumulator.HasMissing = _rowCount > 0;
                        _accumulators[key] = accumulator;
                    }
                }

                foreach (var key in _keys)
                {
                    var accumulator = _accumulators[key];
                    if (row.TryGetValue(key, out var value) && value is not null)
                    {
                        accumulator.Sample ??= value;
                        if (accumulator.Sample is string)
                        {
                            accumulator.DistinctStrings ??= new HashSet<string>(StringComparer.Ordinal);
                            var text = value as string ?? value.ToString() ?? string.Empty;
                            accumulator.DistinctStrings.Add(key == "Surface"
                                ? FeatureText.NormalizeSurface(text)
                                : FeatureText.NormalizeCategory(text));
                        }
                    }
                    else
                    {
                        accumulator.HasMissing = true;
                    }
                }

                _rowCount++;
            }

            public DatasetFeatureMetadata Build()
            {
                var featureDims = new Dictionary<string, int>();
                var stringMaps = new Dictionary<string, Dictionary<string, int>>();
                foreach (var key in _keys)
                {
                    var accumulator = _accumulators[key];
                    if (accumulator.Sample is null)
                    {
                        continue;
                    }

                    int baseDim;
                    if (accumulator.Sample is string)
                    {
                        var distinct = accumulator.DistinctStrings?.ToList() ?? new List<string>();
                        var map = new Dictionary<string, int>(StringComparer.Ordinal);
                        for (int i = 0; i < distinct.Count; i++)
                        {
                            map[distinct[i]] = i;
                        }
                        map["__unknown__"] = distinct.Count;
                        stringMaps[key] = map;
                        baseDim = map.Count;
                    }
                    else
                    {
                        baseDim = 1;
                    }

                    featureDims[key] = (key == "DaysSinceLastWin" || key == "RacesSinceLastWin")
                        ? baseDim // presence handled by HasLastWin
                        : baseDim + (accumulator.HasMissing ? 1 : 0);
                }

                AddFixedFeatureDimensions(featureDims, _marketBlind, _featureBlocks);
                var featureKeys = featureDims.Keys.ToList();

                return new DatasetFeatureMetadata(featureKeys, featureDims, stringMaps);
            }
        }

        private static bool IsMarketDerivedFeature(string key)
        {
            return key.StartsWith("Market", StringComparison.OrdinalIgnoreCase) ||
                   key.StartsWith("Betfair", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(key, "IsFavourite", StringComparison.OrdinalIgnoreCase);
        }

        private static void AddFixedFeatureDimensions(
            Dictionary<string, int> featureDims,
            bool marketBlind,
            FeatureBlocks featureBlocks)
        {
            bool enablePaceFeatures = featureBlocks.HasFlag(FeatureBlocks.Pace);
            featureDims["HasLastWin"] = 1;

            featureDims["TimeOfDaySin"] = 1;
            featureDims["TimeOfDayCos"] = 1;
            featureDims["DistanceChangeFromLast"] = 1;
            featureDims["DistanceRatioFromAverage"] = 1;
            featureDims["CareerStarts"] = 1;
            featureDims["LifetimeWinRate"] = 1;
            featureDims["DrawBias"] = 1;

            // History presence and sample-size features. A smoothed rate alone
            // cannot distinguish "no data" from "average", so the model needs
            // both the rate and the number of observations behind it.
            featureDims["HasHistory"] = 1;
            featureDims["IsFirstStart"] = 1;
            featureDims["HasSpeedHistory"] = 1;
            featureDims["HasRatingHistory"] = 1;
            featureDims["HasJockey"] = 1;
            featureDims["ClassStarts"] = 1;
            featureDims["GoingStarts"] = 1;
            featureDims["SurfaceStarts"] = 1;
            featureDims["CourseStarts"] = 1;
            featureDims["DistanceBucketStarts"] = 1;
            featureDims["GoingCourseStarts"] = 1;
            featureDims["GoingDistanceStarts"] = 1;
            featureDims["AgeRestrictionStarts"] = 1;
            featureDims["JockeyClassStarts"] = 1;
            featureDims["JockeySurfaceStarts"] = 1;
            featureDims["JockeyGoingStarts"] = 1;
            featureDims["JockeyDistanceBucketStarts"] = 1;
            featureDims["JockeyCourseStarts"] = 1;
            featureDims["JockeyGoingDistanceStarts"] = 1;

            // Within-race comparison features. A pointwise MLP otherwise has
            // to learn runner-vs-runner comparisons from absolute values.
            featureDims["RunnerCountInv"] = 1;
            if (!marketBlind)
            {
                featureDims["MarketProbRank"] = 1;
                featureDims["IsFavourite"] = 1;
                featureDims["MarketProbGapToNext"] = 1;
                featureDims["MarketProbGapToLeader"] = 1;
            }
            featureDims["WinRateLast5RankRelative"] = 1;
            featureDims["AvgNormPosLast5RankRelative"] = 1;
            featureDims["JockeyWinRateRankRelative"] = 1;
            featureDims["JockeyGoingDistanceWinRateRankRelative"] = 1;
            featureDims["CareerStartsRankRelative"] = 1;
            featureDims["AvgSpeedLast5RankRelative"] = 1;
            featureDims["ClassWinRateRankRelative"] = 1;
            featureDims["GoingWinRateRankRelative"] = 1;
            featureDims["DistanceBucketWinRateRankRelative"] = 1;
            featureDims["RecentImprovementRankRelative"] = 1;
            featureDims["SaddleclothMissing"] = 1;
            featureDims["SaddleclothRelative"] = 1;
            featureDims["SaddleclothDiffFromMean"] = 1;
            featureDims["JockeyGoingDistanceWinRate"] = 1;
            featureDims["JockeyGoingDistanceAvgNorm"] = 1;
            featureDims["LastJockeyGoingDistanceNormPos"] = 1;
            featureDims["ClassWinRate"] = 1;
            featureDims["ClassAvgNorm"] = 1;
            featureDims["LastClassNormPos"] = 1;
            featureDims["JockeyClassWinRate"] = 1;
            featureDims["JockeyClassAvgNorm"] = 1;
            featureDims["LastJockeyClassNormPos"] = 1;
            if (!marketBlind)
            {
                featureDims["MarketImpliedProb"] = 1;
                featureDims["MarketLogOdds"] = 1;
                featureDims["MarketProbRankRelative"] = 1;
                featureDims["MarketProbDiffFromMean"] = 1;
                featureDims["MarketProbMissing"] = 1;

                // Market context and market-vs-form disagreement. The offset uses
                // the market prior directly, while these features let the network
                // condition on how strong/concentrated the market is and where the
                // market disagrees with the form figures.
                featureDims["MarketLogit"] = 1;
                featureDims["MarketFavouriteProb"] = 1;
                featureDims["MarketProbEntropy"] = 1;
                featureDims["MarketHerfindahl"] = 1;
                featureDims["MarketProbStdDev"] = 1;
                featureDims["MarketTop2Mass"] = 1;
                featureDims["MarketProbGapTop2"] = 1;
                featureDims["MarketOverround"] = 1;
                featureDims["MarketProbMinusLifetimeWinRate"] = 1;
                featureDims["MarketLogitMinusFormLogit"] = 1;
                featureDims["MarketRankMinusWinRateRank"] = 1;
                featureDims["MarketRankMinusAvgNormRank"] = 1;
                featureDims["MarketRankMinusClassRank"] = 1;
                if (enablePaceFeatures)
                {
                    featureDims["MarketRankMinusFrontRunRank"] = 1;
                    featureDims["MarketRankMinusProminentRank"] = 1;
                }
            }

            if (enablePaceFeatures)
            {
                // Run-style features. These are derived from historical comments
                // only; the current race's comment is used after the race to update
                // history, never as a feature of the same race.
                featureDims["HasPaceHistory"] = 1;
                featureDims["LastRunStyle"] = 1;
                featureDims["LastRunStyleKnown"] = 1;
                foreach (var window in PaceWindows)
                {
                    featureDims[$"FrontRunRateLast{window}"] = 1;
                    featureDims[$"ProminentRateLast{window}"] = 1;
                    featureDims[$"ChasedRateLast{window}"] = 1;
                    featureDims[$"MidfieldRateLast{window}"] = 1;
                    featureDims[$"HeldUpRateLast{window}"] = 1;
                    featureDims[$"RearRateLast{window}"] = 1;
                    featureDims[$"KeenRateLast{window}"] = 1;
                    featureDims[$"SlowStartRateLast{window}"] = 1;
                    featureDims[$"AvgPaceScoreLast{window}"] = 1;
                    featureDims[$"PaceStartsLast{window}"] = 1;
                }

                featureDims["FrontRunnerCount"] = 1;
                featureDims["ProminentRunnerCount"] = 1;
                featureDims["HeldUpRunnerCount"] = 1;
                featureDims["KnownPaceRunnerCount"] = 1;
                featureDims["UnknownPaceRunnerCount"] = 1;
                featureDims["LoneSpeedFlag"] = 1;
                featureDims["FieldPaceContention"] = 1;
                featureDims["FieldFrontRunSum"] = 1;
                featureDims["FieldMaxFrontRunRate"] = 1;
                featureDims["FieldMinFrontRunRate"] = 1;
                featureDims["FieldAvgFrontRunRate"] = 1;
                featureDims["FieldProminentSum"] = 1;
                featureDims["FieldMaxProminentRate"] = 1;
                featureDims["FieldMinProminentRate"] = 1;
                featureDims["FieldAvgProminentRate"] = 1;
                featureDims["FieldHeldUpSum"] = 1;
                featureDims["FieldMaxHeldUpRate"] = 1;
                featureDims["FieldMinHeldUpRate"] = 1;
                featureDims["FieldAvgHeldUpRate"] = 1;
                featureDims["PacePressure"] = 1;
                featureDims["OwnFrontRunMinusFieldMean"] = 1;
                featureDims["IsFrontRunner"] = 1;
                featureDims["IsLoneFrontRunner"] = 1;
                featureDims["FrontRunRateRankRelative"] = 1;
                featureDims["ProminentRateRankRelative"] = 1;
                featureDims["HeldUpRateRankRelative"] = 1;
                featureDims["AvgPaceScoreRankRelative"] = 1;

                // Field max/min/mean for the existing form features. The rank
                // features already say where a runner sits; these add the magnitude
                // of the gap to the best and worst in the race.
                featureDims["FieldMaxAvgSpeedLast5"] = 1;
                featureDims["FieldMinAvgSpeedLast5"] = 1;
                featureDims["FieldMeanAvgSpeedLast5"] = 1;
                featureDims["FieldMaxWinRateLast5"] = 1;
                featureDims["FieldMinWinRateLast5"] = 1;
                featureDims["FieldMeanWinRateLast5"] = 1;
                featureDims["FieldMaxAvgNormPosLast5"] = 1;
                featureDims["FieldMinAvgNormPosLast5"] = 1;
                featureDims["FieldMeanAvgNormPosLast5"] = 1;
                featureDims["KnownSpeedRunnerCount"] = 1;
                featureDims["OwnAvgSpeedMinusFieldMean"] = 1;
                featureDims["OwnWinRateMinusFieldMean"] = 1;
                featureDims["OwnAvgNormPosMinusFieldMean"] = 1;
            }

            foreach (var window in PerformanceWindows)
            {
                featureDims[$"AvgRatingLast{window}"] = 1;
            }
            featureDims["RatingSlope"] = 1;

            if (featureBlocks.HasFlag(FeatureBlocks.TrainerForm))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.TrainerForm);
            }
            // MarketMovement and WeatherBalance columns are discovered from
            // the rows so the missing-indicator dimension is added when the
            // source data is incomplete. Forcing a fixed dimension would
            // overwrite that indicator and encode a missing value as a real
            // zero.
            if (featureBlocks.HasFlag(FeatureBlocks.RaceContext))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.RaceContext);
            }
            if (featureBlocks.HasFlag(FeatureBlocks.FormContext))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.FormContext);
            }
            if (featureBlocks.HasFlag(FeatureBlocks.PriceData))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.PriceData);
            }
            if (featureBlocks.HasFlag(FeatureBlocks.TrackGeometry))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.TrackGeometry);
            }
            if (featureBlocks.HasFlag(FeatureBlocks.DeepForm))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.DeepForm);
            }
            if (featureBlocks.HasFlag(FeatureBlocks.Sectionals))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.Sectionals);
            }
            if (featureBlocks.HasFlag(FeatureBlocks.Breeding))
            {
                AddFixedDimensions(featureDims, FeatureBlockCatalog.Breeding);
            }
        }

        private static void AddFixedDimensions(
            Dictionary<string, int> featureDims,
            IEnumerable<string> names)
        {
            foreach (var name in names)
            {
                featureDims[name] = 1;
            }
        }

        private DatasetFeatureMetadata BuildFeatureMetadata(
            PreparedDataset prepared,
            List<Dictionary<string, object?>> metadataRows,
            bool marketBlind)
        {
            if (prepared is null)
                throw new ArgumentNullException(nameof(prepared));
            if (metadataRows is null)
                throw new ArgumentNullException(nameof(metadataRows));

            var accumulator = new FeatureMetadataAccumulator(marketBlind, _featureBlocks);
            foreach (var row in metadataRows)
            {
                accumulator.ObserveRow(row);
            }

            return accumulator.Build();
        }

        /// <summary>
        /// Derives the feature schema directly from a streaming pass over the
        /// database. Only category values and missing indicators are retained;
        /// the runner dictionaries themselves are discarded race-by-race.
        /// </summary>
        private DatasetFeatureMetadata BuildFeatureMetadataFromDatabase(
            ISet<int>? includeRaceIds,
            ISet<int>? stateRaceWhitelist,
            bool includeIdentifiers,
            DateTime? minimumRaceDate,
            bool requireMarketPrice = false,
            bool marketBlind = false)
        {
            var accumulator = new FeatureMetadataAccumulator(marketBlind, _featureBlocks);
            ProcessPreparedRaces(
                includeRaceIds,
                stateRaceWhitelist,
                includeIdentifiers,
                minimumRaceDate,
                race =>
                {
                    foreach (var row in race.Rows)
                    {
                        accumulator.ObserveRow(row);
                    }
                },
                requireMarketPrice);

            return accumulator.Build();
        }

        internal static bool IsFeatureEnabled(string key, FeatureBlocks featureBlocks)
        {
            if (IsTrainerOrAgeFeature(key))
            {
                return featureBlocks.HasFlag(FeatureBlocks.TrainerForm);
            }

            if (key.StartsWith("MarketMove", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.MarketMovement);
            }

            if (key.StartsWith("Betfair", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.MarketLiquidity);
            }

            if (key.StartsWith("WeatherBalance", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.WeatherBalance);
            }

            if (key.StartsWith("RaceContext", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.RaceContext);
            }

            if (key.StartsWith("Form", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.FormContext);
            }

            if (IsPriceDataFeature(key))
            {
                return featureBlocks.HasFlag(FeatureBlocks.PriceData);
            }

            if (FeatureBlockCatalog.WeatherExtended.Contains(key, StringComparer.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.WeatherExtended);
            }

            if (key.StartsWith("TrackGeometry", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.TrackGeometry);
            }

            if (key.StartsWith("DeepForm", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.DeepForm);
            }

            if (key.StartsWith("Sectional", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.Sectionals);
            }

            if (key.StartsWith("Pedigree", StringComparison.OrdinalIgnoreCase))
            {
                return featureBlocks.HasFlag(FeatureBlocks.Breeding);
            }

            return IsLiveAvailableFeature(key);
        }

        private static bool IsPriceDataFeature(string key)
        {
            return key.Equals("OfficialRating", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("OfficialRatingRelative", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("OfficialRatingMissing", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("OfficialRatingVsFieldMax", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("OfficialRatingRankRelative", StringComparison.OrdinalIgnoreCase) ||
                   key.StartsWith("FieldRating", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("WeightLbs", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("WeightDiffFromMean", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("WeightMissing", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("IsTopWeight", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("IsBottomWeight", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("WeightChangeFromLast", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("PurseLevel", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("PurseLog", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("RatingDiffFromField", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("RatingMissing", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("RatingChangeFromLast", StringComparison.OrdinalIgnoreCase) ||
                   key.StartsWith("Headgear", StringComparison.OrdinalIgnoreCase) ||
                   key.StartsWith("Medication", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// True for every feature that is derived from the market price or the
        /// exchange ladder. Used to build a market-blind ensemble member from
        /// an already encoded dataset without a second database pass.
        /// </summary>
        internal static bool IsMarketFeatureKey(string key)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                return false;
            }

            return key.StartsWith("Market", StringComparison.OrdinalIgnoreCase) ||
                   key.StartsWith("Betfair", StringComparison.OrdinalIgnoreCase) ||
                   key.StartsWith("BackPrice", StringComparison.OrdinalIgnoreCase) ||
                   key.StartsWith("LayPrice", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("IsFavourite", StringComparison.OrdinalIgnoreCase);
        }

        internal static bool IsFeatureInBlock(string key, FeatureBlocks block)
        {
            return block switch
            {
                FeatureBlocks.TrainerForm => IsTrainerOrAgeFeature(key),
                FeatureBlocks.MarketMovement => key.StartsWith("MarketMove", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.WeatherBalance => key.StartsWith("WeatherBalance", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.RaceContext => key.StartsWith("RaceContext", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.FormContext => key.StartsWith("Form", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.PriceData => IsPriceDataFeature(key),
                FeatureBlocks.WeatherExtended => FeatureBlockCatalog.WeatherExtended.Contains(key, StringComparer.OrdinalIgnoreCase),
                FeatureBlocks.TrackGeometry => key.StartsWith("TrackGeometry", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.DeepForm => key.StartsWith("DeepForm", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.MarketLiquidity => key.StartsWith("Betfair", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.Sectionals => key.StartsWith("Sectional", StringComparison.OrdinalIgnoreCase),
                FeatureBlocks.Breeding => key.StartsWith("Pedigree", StringComparison.OrdinalIgnoreCase),
                _ => false
            };
        }

        internal static bool IsTrainerOrAgeFeature(string key)
        {
            // LastTrainer* features were already part of the pre-block schema.
            // They become meaningful as soon as the live path supplies a
            // trainer, so they are not gated here to keep the baseline
            // comparable with previously trained models.
            return key.StartsWith("Trainer", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("Age", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("AgeRelative", StringComparison.OrdinalIgnoreCase) ||
                   key.Equals("AgeProgression", StringComparison.OrdinalIgnoreCase);
        }

        internal static bool IsLiveAvailableFeature(string key)
        {
            // Betfair traded volume/BSP is imported from historical files but
            // there is no live capture yet. Marking it unavailable keeps a
            // model trained with MarketLiquidity out of live scoring until a
            // live Betfair API snapshot path exists.
            if (key.StartsWith("Betfair", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            // Trainer, age, weight and official rating are populated for live
            // rows from the horse's most recent result (see
            // BuildUpcomingRaceRows). Post-race speed and beaten-distance
            // features cannot be known before the off.
            if (key.StartsWith("Trainer", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            // Field-strength aggregates use the same live rating fallback, so
            // they remain available whenever the individual rating is.
            if (key.StartsWith("FieldRating", StringComparison.OrdinalIgnoreCase) ||
                key.Equals("OfficialRatingRelative", StringComparison.OrdinalIgnoreCase) ||
                key.Equals("OfficialRatingVsFieldMax", StringComparison.OrdinalIgnoreCase) ||
                key.Equals("OfficialRatingRankRelative", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            return key switch
            {
                "WinningTimeMs" => false,
                "RaceSpeed" => false,
                "RunnerSpeed" => false,
                "SpeedDiff" => false,
                "SpeedRatio" => false,
                "SpeedMissing" => false,
                "DistanceBeatenLengths" => false,
                "DistanceBeatenKnown" => false,
                "Age" => true,
                "WeightLbs" => true,
                "OfficialRating" => true,
                "PurseLevel" => true,
                "AgeRelative" => true,
                "WeightDiffFromMean" => true,
                "RatingDiffFromField" => true,
                "FieldRatingStdDev" => true,
                "IsTopWeight" => true,
                "IsBottomWeight" => true,
                "RatingMissing" => true,
                "WeightMissing" => true,
                "RatingChangeFromLast" => true,
                "WeightChangeFromLast" => true,
                "AgeProgression" => true,
                _ => true
            };
        }

        internal static float[] EncodeFeatureVector(
           IDictionary<string, object?> row,
           IList<string> featureKeys,
           IDictionary<string, int> featureDimensions,
           IDictionary<string, Dictionary<string, int>> stringMaps,
           int featureCount)
        {
            if (row is null)
                throw new ArgumentNullException(nameof(row));
            if (featureKeys is null)
                throw new ArgumentNullException(nameof(featureKeys));
            if (featureDimensions is null)
                throw new ArgumentNullException(nameof(featureDimensions));
            if (stringMaps is null)
                throw new ArgumentNullException(nameof(stringMaps));

            var features = new float[featureCount];
            int offset = 0;
            foreach (var key in featureKeys)
            {
                if (!featureDimensions.TryGetValue(key, out var dim))
                {
                    continue;
                }

                row.TryGetValue(key, out var value);
                EncodeFeatureInto(key, value, dim, stringMaps, features, offset);
                offset += dim;
            }

            return features;
        }

        private static RaceExample EncodeRace(
            PreparedRace preparedRace,
            List<string> featureKeys,
            Dictionary<string, int> featureDimensions,
            Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            if (preparedRace is null)
                throw new ArgumentNullException(nameof(preparedRace));
            if (featureKeys is null)
                throw new ArgumentNullException(nameof(featureKeys));
            if (featureDimensions is null)
                throw new ArgumentNullException(nameof(featureDimensions));
            if (stringMaps is null)
                throw new ArgumentNullException(nameof(stringMaps));

            int featureCount = featureDimensions.Values.Sum();
            int rowCount = preparedRace.Rows.Count;
            var marketProbabilities = new float[rowCount];
            var marketLogProbabilities = new float[rowCount];
            int pricedCount = 0;
            double logProbabilitySum = 0d;
            for (int i = 0; i < rowCount; i++)
            {
                var row = preparedRace.Rows[i];
                float probability = 0f;
                if (row.TryGetValue("MarketImpliedProb", out var marketProbObj) && marketProbObj is not null)
                {
                    try
                    {
                        probability = Convert.ToSingle(marketProbObj, CultureInfo.InvariantCulture);
                    }
                    catch
                    {
                        probability = 0f;
                    }
                }

                bool missing = row.TryGetValue("MarketProbMissing", out var marketMissing) &&
                               marketMissing is bool isMarketMissing &&
                               isMarketMissing;
                if (!missing && probability > 1e-6f && probability < 1f - 1e-6f)
                {
                    marketProbabilities[i] = probability;
                    marketLogProbabilities[i] = MathF.Log(probability);
                    logProbabilitySum += marketLogProbabilities[i];
                    pricedCount++;
                }
                else
                {
                    marketProbabilities[i] = 0f;
                    marketLogProbabilities[i] = float.NaN;
                }
            }

            // Missing-price runners are treated as an average priced runner in
            // log space. A zero offset would mean p=0.5, which is almost never a
            // neutral prior in a field of eight or more runners.
            float meanLogProbability = pricedCount > 0
                ? (float)(logProbabilitySum / pricedCount)
                : 0f;
            double allLogProbabilitySum = 0d;
            for (int i = 0; i < rowCount; i++)
            {
                if (!float.IsFinite(marketLogProbabilities[i]))
                {
                    marketLogProbabilities[i] = meanLogProbability;
                }
                allLogProbabilitySum += marketLogProbabilities[i];
            }

            // Centre per race. Softmax is invariant to a constant, so this only
            // keeps the learned offset scale numerically well behaved.
            float logProbabilityCentre = rowCount > 0
                ? (float)(allLogProbabilitySum / rowCount)
                : 0f;
            for (int i = 0; i < rowCount; i++)
            {
                marketLogProbabilities[i] -= logProbabilityCentre;
            }

            var runners = new List<RunnerExample>(rowCount);

            // Prefer the declared field size, which includes non-finishers.
            // Falling back to the encoded row count keeps synthetic datasets
            // and partial imports usable.
            int fieldSize = rowCount;
            if (rowCount > 0 &&
                preparedRace.Rows[0].TryGetValue("RunnerCount", out var runnerCountValue) &&
                runnerCountValue is not null &&
                PreparedDataset.TryConvertToInt32(runnerCountValue, out var declaredRunnerCount) &&
                declaredRunnerCount > 0)
            {
                fieldSize = declaredRunnerCount;
            }

            for (int i = 0; i < rowCount; i++)
            {
                var row = preparedRace.Rows[i];
                var features = EncodeFeatureVector(row, featureKeys, featureDimensions, stringMaps, featureCount);

                int? finishPosition = null;
                if (row.TryGetValue("FinishPos", out var finishValue) &&
                    finishValue is not null &&
                    PreparedDataset.TryConvertToInt32(finishValue, out var parsedFinishPosition) &&
                    parsedFinishPosition > 0)
                {
                    finishPosition = parsedFinishPosition;
                }

                float label = finishPosition == 1 ? 1f : 0f;

                int? horseId = null;
                if (row.TryGetValue("HorseId", out var horseIdValue) && horseIdValue is not null && PreparedDataset.TryConvertToInt32(horseIdValue, out var parsedHorseId))
                {
                    horseId = parsedHorseId;
                }

                string? horseName = null;
                if (row.TryGetValue("HorseName", out var horseNameValue) && horseNameValue is not null)
                {
                    horseName = horseNameValue.ToString();
                }

                var spDecimal = GetNullableDecimal(row, "SP_Decimal");
                var betfairSp = GetNullableDecimal(row, "BetfairSP");
                var betfairLastTraded = GetNullableDecimal(row, "BetfairLastTradedPrice");
                bool hasMarketPrice = marketProbabilities[i] > 0f;
                float marketLogit = marketProbabilities[i] > 0f
                    ? (float)Math.Log(marketProbabilities[i] / (1f - marketProbabilities[i]))
                    : 0f;

                runners.Add(new RunnerExample(
                    preparedRace.RaceId,
                    features,
                    label,
                    horseId,
                    horseName,
                    spDecimal,
                    marketLogit,
                    hasMarketPrice,
                    marketProbabilities[i],
                    marketLogProbabilities[i],
                    finishPosition,
                    fieldSize,
                    betfairSp,
                    betfairLastTraded));
            }

            DateTime raceDate = preparedRace.Rows.Count > 0 &&
                preparedRace.Rows[0].TryGetValue("RaceDate", out var raceDateValue) &&
                raceDateValue is DateTime parsedRaceDate
                    ? parsedRaceDate.Date
                    : DateTime.MinValue;

            return new RaceExample(preparedRace.RaceId, raceDate, runners);
        }

        private static List<RaceExample> EncodeRaces(
            IEnumerable<PreparedRace> races,
            List<string> featureKeys,
            Dictionary<string, int> featureDimensions,
            Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            if (races is null)
                throw new ArgumentNullException(nameof(races));

            var result = new List<RaceExample>();
            foreach (var preparedRace in races)
            {
                result.Add(EncodeRace(preparedRace, featureKeys, featureDimensions, stringMaps));
            }

            return result;
        }

        /// <summary>
        /// Encodes races while they stream out of the chronological database
        /// pass. Each encoded race is compact (float vectors plus a little
        /// metadata), so the large per-runner dictionaries can be collected as
        /// soon as the race has been encoded.
        /// </summary>
        private List<RaceExample> EncodeRacesFromDatabase(
            DatasetFeatureMetadata metadata,
            ISet<int>? includeRaceIds,
            ISet<int>? stateRaceWhitelist,
            bool includeIdentifiers,
            DateTime? minimumRaceDate,
            bool requireMarketPrice = false)
        {
            if (metadata is null)
                throw new ArgumentNullException(nameof(metadata));

            var result = new List<RaceExample>();
            ProcessPreparedRaces(
                includeRaceIds,
                stateRaceWhitelist,
                includeIdentifiers,
                minimumRaceDate,
                race =>
                {
                    result.Add(EncodeRace(
                        race,
                        metadata.FeatureKeys,
                        metadata.FeatureDimensions,
                        metadata.StringMaps));
                },
                requireMarketPrice);

            return result;
        }
        private static decimal? GetNullableDecimal(Dictionary<string, object?> row, string key)
        {
            if (!row.TryGetValue(key, out var value) || value is null)
            {
                return null;
            }

            return ConvertToDecimal(value);
        }

        private static decimal? ConvertToDecimal(object value)
        {
            try
            {
                return value switch
                {
                    decimal dec => dec,
                    double dbl when double.IsFinite(dbl) => (decimal)dbl,
                    float fl when float.IsFinite(fl) => (decimal)fl,
                    int i => i,
                    long l => l,
                    short s => s,
                    byte b => b,
                    uint ui => (decimal)ui,
                    ulong ul => Convert.ToDecimal(ul),
                    SqlDecimal sqlDec when !sqlDec.IsNull => sqlDec.Value,
                    SqlMoney sqlMoney when !sqlMoney.IsNull => sqlMoney.Value,
                    SqlDouble sqlDouble when !sqlDouble.IsNull && double.IsFinite(sqlDouble.Value) => (decimal)sqlDouble.Value,
                    SqlSingle sqlSingle when !sqlSingle.IsNull && float.IsFinite(sqlSingle.Value) => (decimal)sqlSingle.Value,
                    SqlInt32 sqlInt when !sqlInt.IsNull => sqlInt.Value,
                    SqlInt64 sqlLong when !sqlLong.IsNull => sqlLong.Value,
                    SqlInt16 sqlShort when !sqlShort.IsNull => sqlShort.Value,
                    SqlByte sqlByte when !sqlByte.IsNull => sqlByte.Value,
                    string s when decimal.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsed) => parsed,
                    string s2 when decimal.TryParse(s2, NumberStyles.Any, CultureInfo.CurrentCulture, out var parsedLocal) => parsedLocal,
                    _ => Convert.ToDecimal(value, CultureInfo.InvariantCulture)
                };
            }
            catch
            {
                return null;
            }
        }

    }
}