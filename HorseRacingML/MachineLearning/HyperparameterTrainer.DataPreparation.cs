using Dapper;
using Microsoft.Data.SqlClient;
using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Globalization;
using HorseRacingML.Models;
using System.Linq;
using System.Text;
using static HorseRacingML.ML.HyperparameterTrainer.TrainingDataset;
using static HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

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

            public class PreparedDataset
            {
                public PreparedDataset(List<PreparedRace> races)
                {
                    Races = races ?? throw new ArgumentNullException(nameof(races));
                }

                public List<PreparedRace> Races { get; }
                public IEnumerable<Dictionary<string, object?>> Rows => Races.SelectMany(r => r.Rows);
                public int RowCount => Races.Sum(r => r.Rows.Count);
                private static readonly Dictionary<string, string[]> ColumnAliases = new(StringComparer.OrdinalIgnoreCase)
                {
                    ["HorseId"] = new[] { "HorseID", "Horse_Id", "HorseIdentifier", "RunnerHorseId", "RunnerHorseID", "Id" },
                    ["RaceId"] = new[] { "RaceID" },
                };
                internal static bool TryGetRequiredInt32(Dictionary<string, object?> row, string key, out int result)
                {
                    if (row is null)
                    {
                        throw new ArgumentNullException(nameof(row));
                    }

                    if (TryGetValueWithAliases(row, key, out var value) &&
                        value is not null &&
                        TryConvertToInt32(value, out result))
                    {
                        row[key] = result;
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
                    if (TryGetRequiredInt32(row, key, out var result))
                    {
                        return result;
                    }

                    if (TryGetValueWithAliases(row, key, out var rawValue, requireNonNull: false))
                    {
                        if (rawValue is null)
                        {
                            throw new InvalidOperationException($"Missing required column '{key}'.");
                        }

                        if (TryConvertToInt32(rawValue, out result))
                        {
                            row[key] = result;
                            return result;
                        }

                        string typeName = rawValue.GetType().FullName ?? rawValue.GetType().Name;
                        string displayValue = rawValue switch
                        {
                            byte[] bytes => $"0x{BitConverter.ToString(bytes).Replace("-", string.Empty)}",
                            _ => rawValue.ToString() ?? string.Empty
                        };

                        throw new FormatException($"Column '{key}' with value '{displayValue}' of type '{typeName}' could not be converted to Int32.");
                    }

                    string message = $"Missing required column '{key}'.";
                    if (row is not null && row.Count > 0)
                    {
                        var available = string.Join(", ", row.Keys.OrderBy(k => k, StringComparer.OrdinalIgnoreCase));
                        message = string.Concat(message, " Available columns: ", available, ".");
                    }

                    throw new InvalidOperationException(message);
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
                public static bool TryGetValueWithAliases(
                    Dictionary<string, object?> row,
                    string key,
                    out object? value,
                    bool requireNonNull = true)
                {
                    if (row.TryGetValue(key, out value) && (!requireNonNull || value is not null))
                    {
                        return true;
                    }

                    if (ColumnAliases.TryGetValue(key, out var aliases))
                    {
                        foreach (var alias in aliases)
                        {
                            if (row.TryGetValue(alias, out var aliasValue) && (!requireNonNull || aliasValue is not null))
                            {
                                value = aliasValue;
                                if (aliasValue is not null)
                                {
                                    row[key] = aliasValue;
                                }
                                return true;
                            }
                        }
                    }

                    value = null;
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
            public RaceExample(int raceId, List<RunnerExample> runners)
            {
                RaceId = raceId;
                Runners = runners;
            }

            public int RaceId { get; }
            public List<RunnerExample> Runners { get; }
        }

        public class RunnerExample
        {
            public RunnerExample(int raceId, float[] features, float label, int? horseId, string? horseName, decimal? startingPriceDecimal)
            {
                RaceId = raceId;
                Features = features;
                Label = label;
                HorseId = horseId;
                HorseName = horseName;
                StartingPriceDecimal = startingPriceDecimal;
            }

            public int RaceId { get; }
            public float[] Features { get; }
            public float Label { get; }
            public int? HorseId { get; }
            public string? HorseName { get; }
            public decimal? StartingPriceDecimal { get; }
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

        private static float[] EncodeFeature(
            string key,
            object? value,
            int dim,
             IDictionary<string, Dictionary<string, int>> stringMaps)
        {
            // Determine the base dimension (excluding missing indicator) and whether an
            // additional slot is reserved for missing values.
            int baseDim = stringMaps.TryGetValue(key, out var map)
                ? map.Count
                : 1;
            bool hasMissingIndicator = dim > baseDim;

            if (value == null)
            {
                var arr = new float[dim];
                if (hasMissingIndicator)
                    arr[baseDim] = 1f;
                return arr;
            }

            float[] encoded = value switch
            {
                // Convert to days relative to a recent base date to avoid huge tick values
                DateTime dt => new[] { (float)(dt - BaseDate).TotalDays },
                // Seconds are a reasonable scale for durations
                TimeSpan ts => new[] { (float)ts.TotalSeconds },
                string s => EncodeString(key, s, baseDim, stringMaps),
                bool b => new[] { b ? 1f : 0f },
                // Scale down very large numeric values to keep them in a manageable range
                _ => EncodeNumeric(value, baseDim)
            };

            if (!hasMissingIndicator)
                return encoded;

            var result = new float[dim];
            Array.Copy(encoded, result, baseDim);
            return result;
        }

        private static float[] EncodeNumeric(object value, int dim)
        {
            float f = Convert.ToSingle(value);
            if (Math.Abs(f) > 1_000_000f)
                f /= 1_000_000f;
            var arr = new float[dim];
            arr[0] = f;
            return arr;
        }

        private static float[] EncodeString(
            string key,
            string s,
            int dim,
            IDictionary<string, Dictionary<string, int>> stringMaps)
        {
            var vec = new float[dim];
            if (!stringMaps.TryGetValue(key, out var map) || map.Count == 0)
            {
                return vec;
            }

            if (map.TryGetValue(s, out var idx) && idx >= 0 && idx < dim)
            {
                vec[idx] = 1f;
                return vec;
            }

            if (map.TryGetValue("__unknown__", out var unknownIdx) && unknownIdx >= 0 && unknownIdx < dim)
            {
                vec[unknownIdx] = 1f;
            }
            else if (dim > 0)
            {
                vec[dim - 1] = 1f;
            }

            return vec;
        }

        private DatasetFeatureMetadata BuildFeatureMetadata(
            PreparedDataset prepared,
            List<Dictionary<string, object?>> metadataRows)
        {
            if (prepared is null)
                throw new ArgumentNullException(nameof(prepared));
            if (metadataRows is null)
                throw new ArgumentNullException(nameof(metadataRows));

            var keys = prepared.Rows
                .SelectMany(r => r.Keys)
                .Distinct()
                .ToList();
            keys.Remove("FinishPos");
            keys.Remove("RaceId");
            keys.Remove("HorseId");
            keys.Remove("CourseId");
            keys.Remove("TrainerId");
            keys.Remove("JockeyId");
            keys.Remove("OutcomeCode");
            keys.Remove("DistanceBeatenText");
            keys.Remove("DistanceBeatenLengths");
            keys.Remove("DistanceBeatenKnown");
            keys.Remove("SP_Fraction");
            keys.Remove("SP_Decimal");
            keys.Remove("OpeningFraction");
            keys.Remove("TouchedHighFraction");
            keys.Remove("TouchedLowFraction");
            keys.Remove("HorseName");
            keys.Remove("JockeyName");
            keys.Remove("TrainerName");
            keys.Remove("Title");
            keys.Remove("RaceMonth");
            keys.Remove("RaceDayOfWeek");
            keys.Remove("Season");
            keys.Remove("ActualOff");
            keys.Remove("ScheduledOff");
            keys.Remove("RaceDate");
            keys.Remove("CourseName");
            keys.Remove("DistanceText");
            keys.Remove("Status");
            keys.Remove("WeightText");
            keys.Remove("FavTag");
            keys.Remove("SaddleclothNumber");
            keys.Remove("Purse");

            var featureDims = new Dictionary<string, int>();
            var stringMaps = new Dictionary<string, Dictionary<string, int>>();
            foreach (var k in keys)
            {
                var values = metadataRows
                    .Select(r => r.ContainsKey(k) ? r[k] : null)
                    .ToList();
                bool hasMissing = values.Count == 0 || values.Any(v => v == null);
                var nonNullValues = values.Where(v => v != null).ToList();
                object? sample = nonNullValues.Count > 0
                    ? nonNullValues[0]
                    : prepared.Rows
                        .Select(r => r.ContainsKey(k) ? r[k] : null)
                        .FirstOrDefault(v => v != null);
                if (sample == null)
                {
                    continue;
                }
                var firstValue = nonNullValues.Count > 0 ? nonNullValues[0]! : sample;
                int baseDim;
                if (firstValue is string)
                {
                    var distinct = nonNullValues.Count > 0
                        ? nonNullValues.Cast<string>().Distinct().ToList()
                        : new List<string>();
                    var map = distinct
                        .Select((v, idx) => new { v, idx })
                        .ToDictionary(x => x.v, x => x.idx);
                    map["__unknown__"] = distinct.Count;
                    stringMaps[k] = map;
                    baseDim = map.Count;
                }
                else
                {
                    baseDim = 1;
                }
                if (k == "DaysSinceLastWin" || k == "RacesSinceLastWin")
                {
                    featureDims[k] = baseDim; // presence handled by HasLastWin
                }
                else
                {
                    featureDims[k] = baseDim + (hasMissing ? 1 : 0);
                }
            }
            featureDims["HasLastWin"] = 1;

            featureDims["TimeOfDaySin"] = 1;
            featureDims["TimeOfDayCos"] = 1;
            featureDims["DistanceChangeFromLast"] = 1;
            featureDims["DistanceRatioFromAverage"] = 1;
            featureDims["CareerStarts"] = 1;
            featureDims["LifetimeWinRate"] = 1;
            featureDims["DrawBias"] = 1;
            featureDims["DistanceBeatenLengths"] = 1;
            featureDims["DistanceBeatenKnown"] = 1;
            featureDims["SaddleclothMissing"] = 1;
            featureDims["SaddleclothRelative"] = 1;
            featureDims["SaddleclothDiffFromMean"] = 1;
            featureDims["IsTopWeight"] = 1;
            featureDims["IsBottomWeight"] = 1;
            featureDims["JockeyGoingDistanceWinRate"] = 1;
            featureDims["JockeyGoingDistanceAvgNorm"] = 1;
            featureDims["LastJockeyGoingDistanceNormPos"] = 1;
            featureDims["TrainerJockeyCourseWinRate"] = 1;
            featureDims["ClassWinRate"] = 1;
            featureDims["ClassAvgNorm"] = 1;
            featureDims["LastClassNormPos"] = 1;
            featureDims["TrainerClassWinRate"] = 1;
            featureDims["TrainerClassAvgNorm"] = 1;
            featureDims["LastTrainerClassNormPos"] = 1;
            featureDims["JockeyClassWinRate"] = 1;
            featureDims["JockeyClassAvgNorm"] = 1;
            featureDims["LastJockeyClassNormPos"] = 1;
            foreach (var window in PerformanceWindows)
            {
                featureDims[$"AvgRatingLast{window}"] = 1;
            }
            featureDims["RatingSlope"] = 1;
            var featureKeys = featureDims.Keys.ToList();

            return new DatasetFeatureMetadata(featureKeys, featureDims, stringMaps);
        }
        internal static float[] EncodeFeatureVector(
           Dictionary<string, object?> row,
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
                var vec = EncodeFeature(key, value, dim, stringMaps);
                Array.Copy(vec, 0, features, offset, dim);
                offset += dim;
            }

            return features;
        }

        private static List<RaceExample> EncodeRaces(
            IEnumerable<PreparedRace> races,
            List<string> featureKeys,
            Dictionary<string, int> featureDimensions,
            Dictionary<string, Dictionary<string, int>> stringMaps)
        {
            if (races is null)
                throw new ArgumentNullException(nameof(races));
            if (featureKeys is null)
                throw new ArgumentNullException(nameof(featureKeys));
            if (featureDimensions is null)
                throw new ArgumentNullException(nameof(featureDimensions));
            if (stringMaps is null)
                throw new ArgumentNullException(nameof(stringMaps));

            int featureCount = featureDimensions.Values.Sum();
            var result = new List<RaceExample>();
            foreach (var preparedRace in races)
            {
                var runners = new List<RunnerExample>(preparedRace.Rows.Count);
                foreach (var row in preparedRace.Rows)
                {
                    var features = EncodeFeatureVector(row, featureKeys, featureDimensions, stringMaps, featureCount);

                    float label = row.TryGetValue("FinishPos", out var f) && PreparedDataset.TryConvertToInt32(f, out var finishPos) && finishPos == 1
                        ? 1f
                    : 0f;

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

                    runners.Add(new RunnerExample(preparedRace.RaceId, features, label, horseId, horseName, spDecimal));
                }

                result.Add(new RaceExample(preparedRace.RaceId, runners));
            }

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