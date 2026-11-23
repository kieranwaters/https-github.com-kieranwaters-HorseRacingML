using System.Data;
using System.Globalization;
using System.Text.RegularExpressions;
using Dapper;
using HorseRacingML.Services;
using Microsoft.Data.SqlClient;
using Microsoft.Data.Sqlite;

namespace HorseRacingML.Data
{
    /// <summary>
    /// Imports the pre-2016 UK/IRE results from the Kaggle "Horse Racing results -
    /// UK/Ireland 1988-2026" SQLite files into the application schema.
    ///
    /// Duplicate safety:
    ///  * only dates strictly before the earliest Sporting Life date are read;
    ///  * races are matched on (course, date, off, normalised title) before inserting;
    ///  * runner rows are matched on (RaceId, HorseId) and skipped when they exist;
    ///  * horses/trainers/jockeys are matched by cleaned name and reused.
    /// </summary>
    public sealed class KaggleRaceformImporter
    {
        private readonly string _connectionString;
        private readonly Action<string> _log;
        private readonly Dictionary<string, int> _courseIds = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _horseIds = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _trainerIds = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _jockeyIds = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _raceIds = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<long> _runnerKeys = new();
        private readonly List<RunnerDraft> _pending = new();
        private readonly HashSet<string> _pendingHorses = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _pendingTrainers = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _pendingJockeys = new(StringComparer.OrdinalIgnoreCase);
        private SqlConnection _sql = null!;
        private DateTime _cutoff;
        private int _races;
        private int _runners;
        private int _skippedRaces;

        private const int RunnerBatchSize = 40_000;

        public KaggleRaceformImporter(string connectionString, Action<string>? log = null)
        {
            _connectionString = connectionString;
            _log = log ?? Console.WriteLine;
        }

        public sealed class ImportSummary
        {
            public int Races { get; set; }
            public int Runners { get; set; }
            public int SkippedRaces { get; set; }
            public TimeSpan Elapsed { get; set; }
        }

        public ImportSummary Import(IEnumerable<string> sqliteFiles, DateTime? cutoffOverride = null)
        {
            var started = DateTime.UtcNow;
            using var sql = new SqlConnection(_connectionString);
            sql.Open();
            _sql = sql;

            _cutoff = cutoffOverride ?? GetCutoff(sql);
            _log($"[kaggle] cutoff = {_cutoff:yyyy-MM-dd} (only earlier dates will be imported)");

            LoadCaches(sql);
            _log($"[kaggle] loaded {_courseIds.Count} courses, {_horseIds.Count} horses, {_trainerIds.Count} trainers, {_jockeyIds.Count} jockeys, {_raceIds.Count} existing races, {_runnerKeys.Count} existing runners");

            foreach (var file in sqliteFiles)
            {
                if (!File.Exists(file))
                {
                    _log($"[kaggle] missing file: {file}");
                    continue;
                }
                ImportFile(file);
                Flush(sql);
            }

            Flush(sql);
            var elapsed = DateTime.UtcNow - started;
            _log($"[kaggle] finished: {_races} races, {_runners} runners, {_skippedRaces} skipped races in {elapsed:hh\\:mm\\:ss}");
            return new ImportSummary { Races = _races, Runners = _runners, SkippedRaces = _skippedRaces, Elapsed = elapsed };
        }

        private DateTime GetCutoff(SqlConnection sql)
        {
            var earliest = sql.QuerySingleOrDefault<DateTime?>(
                @"SELECT MIN(r.RaceDate)
                  FROM Race r
                  WHERE EXISTS (SELECT 1 FROM RunnerResult rr WHERE rr.RaceId = r.RaceId)");
            return earliest ?? DateTime.Today;
        }

        private void LoadCaches(SqlConnection sql)
        {
            foreach (var row in sql.Query<IdName>("SELECT CourseId AS Id, Name FROM Course"))
            {
                _courseIds.TryAdd(CleanName(row.Name), row.Id);
            }
            foreach (var row in sql.Query<IdName>("SELECT HorseId AS Id, Name FROM Horse"))
            {
                _horseIds.TryAdd(CleanName(row.Name), row.Id);
            }
            foreach (var row in sql.Query<IdName>("SELECT TrainerId AS Id, Name FROM Trainer"))
            {
                _trainerIds.TryAdd(CleanName(row.Name), row.Id);
            }
            foreach (var row in sql.Query<IdName>("SELECT JockeyId AS Id, Name FROM Jockey"))
            {
                _jockeyIds.TryAdd(CleanName(row.Name), row.Id);
            }

            foreach (var row in sql.Query<ExistingRace>(
                @"SELECT RaceId, CourseId, RaceDate, ScheduledOff, Title
                  FROM Race
                  WHERE RaceDate < @Cutoff", new { Cutoff = _cutoff }))
            {
                _raceIds[RaceKey(row.CourseId, row.RaceDate, row.ScheduledOff, row.Title)] = row.RaceId;
            }

            foreach (var row in sql.Query<ExistingRunner>(
                @"SELECT rr.RaceId, rr.HorseId
                  FROM RunnerResult rr
                  INNER JOIN Race r ON r.RaceId = rr.RaceId
                  WHERE r.RaceDate < @Cutoff", new { Cutoff = _cutoff }))
            {
                _runnerKeys.Add(RunnerKey(row.RaceId, row.HorseId));
            }
        }

        private void ImportFile(string path)
        {
            _log($"[kaggle] reading {Path.GetFileName(path)}");
            using var lite = new SqliteConnection($"Data Source={path};Mode=ReadOnly");
            lite.Open();
            using var command = lite.CreateCommand();
            command.CommandText = @"
SELECT date, course, race_id, off, race_name, type, class, age_band, dist, going,
       ran, num, pos, draw, ovr_btn, btn, horse, age, wgt, time, sp, jockey, trainer, comment
FROM data
ORDER BY date, course, off, race_id, pos";
            using var reader = command.ExecuteReader();

            long? currentRaceId = null;
            var group = new List<SqliteRow>();
            var rowsRead = 0;

            while (reader.Read())
            {
                var row = ReadRow(reader);
                rowsRead++;
                if (row == null) continue;
                if (currentRaceId.HasValue && row.RaceId != currentRaceId.Value)
                {
                    ProcessRace(group);
                    group.Clear();
                }
                currentRaceId = row.RaceId;
                group.Add(row);
            }
            if (group.Count > 0) ProcessRace(group);
            _log($"[kaggle] {Path.GetFileName(path)}: {rowsRead} rows read");
        }

        private void ProcessRace(List<SqliteRow> rows)
        {
            if (rows.Count == 0) return;
            var first = rows[0];
            if (first.Date >= _cutoff)
            {
                _skippedRaces++;
                return;
            }

            var course = CleanCourse(first.Course);
            if (course == null)
            {
                _skippedRaces++;
                return;
            }

            var courseId = EnsureCourse(course);
            var title = (first.RaceName ?? string.Empty).Trim();
            var key = RaceKey(courseId, first.Date, first.Off, title);
            if (!_raceIds.TryGetValue(key, out var raceId))
            {
                raceId = InsertRace(courseId, first);
                _raceIds[key] = raceId;
                _races++;
            }

            foreach (var row in rows)
            {
                var horse = CleanName(row.Horse);
                if (string.IsNullOrWhiteSpace(horse)) continue;
                var trainer = CleanName(row.Trainer);
                var jockey = CleanName(row.Jockey);
                var (finishPos, outcome) = ParsePosition(row.Pos);
                var (spFraction, spDecimal, favTag) = ParseSp(row.Sp);
                var (beatenText, beatenLengths) = ParseBeaten(row.Btn, row.OvrBtn);
                var weightText = row.Weight ?? string.Empty;
                var weightLbs = ParseWeightLbs(weightText);

                _pending.Add(new RunnerDraft
                {
                    RaceId = raceId,
                    SourceRunnerId = $"{row.RaceId}:{row.Num}",
                    Horse = horse,
                    Trainer = trainer,
                    Jockey = jockey,
                    Saddlecloth = ToByte(row.Num),
                    Draw = ToByte(row.Draw),
                    Age = ToByte(row.Age),
                    WeightLbs = weightLbs,
                    WeightText = weightText,
                    FinishPos = finishPos,
                    OutcomeCode = outcome,
                    DistanceBeatenText = beatenText,
                    DistanceBeatenLengths = beatenLengths,
                    SpFraction = spFraction,
                    SpDecimal = spDecimal,
                    FavTag = favTag,
                    Comment = row.Comment
                });
                _pendingHorses.Add(horse);
                if (!string.IsNullOrWhiteSpace(trainer)) _pendingTrainers.Add(trainer);
                if (!string.IsNullOrWhiteSpace(jockey)) _pendingJockeys.Add(jockey);
            }

            if (_pending.Count >= RunnerBatchSize)
            {
                Flush(_sql);
            }
        }

        private void Flush(SqlConnection sql)
        {
            if (_pending.Count == 0) return;

            EnsureLookup(sql, "Horse", "HorseId", _pendingHorses, _horseIds);
            EnsureLookup(sql, "Trainer", "TrainerId", _pendingTrainers, _trainerIds);
            EnsureLookup(sql, "Jockey", "JockeyId", _pendingJockeys, _jockeyIds);

            var table = new DataTable();
            table.Columns.Add("RaceId", typeof(int));
            table.Columns.Add("HorseId", typeof(int));
            table.Columns.Add("TrainerId", typeof(short));
            table.Columns.Add("JockeyId", typeof(short));
            table.Columns.Add("SaddleclothNumber", typeof(byte));
            table.Columns.Add("Draw", typeof(byte));
            table.Columns.Add("Age", typeof(byte));
            table.Columns.Add("WeightLbs", typeof(byte));
            table.Columns.Add("WeightText", typeof(string));
            table.Columns.Add("FinishPos", typeof(short));
            table.Columns.Add("OutcomeCode", typeof(string));
            table.Columns.Add("DistanceBeatenText", typeof(string));
            table.Columns.Add("DistanceBeatenLengths", typeof(decimal));
            table.Columns.Add("SP_Fraction", typeof(string));
            table.Columns.Add("SP_Decimal", typeof(decimal));
            table.Columns.Add("FavTag", typeof(string));
            table.Columns.Add("OpeningFraction", typeof(string));
            table.Columns.Add("TouchedHighFraction", typeof(string));
            table.Columns.Add("TouchedLowFraction", typeof(string));
            table.Columns.Add("Comment", typeof(string));
            table.Columns.Add("Source", typeof(string));
            table.Columns.Add("SourceRunnerId", typeof(string));

            var inserted = 0;
            var insertedRunStyles = new List<RunStyleInput>();
            foreach (var draft in _pending)
            {
                if (!_horseIds.TryGetValue(draft.Horse, out var horseId)) continue;
                var key = RunnerKey(draft.RaceId, horseId);
                if (!_runnerKeys.Add(key)) continue;

                short? trainerId = _trainerIds.TryGetValue(draft.Trainer, out var t) ? (short)t : null;
                short? jockeyId = _jockeyIds.TryGetValue(draft.Jockey, out var j) ? (short)j : null;

                table.Rows.Add(
                    draft.RaceId,
                    horseId,
                    trainerId ?? (object)DBNull.Value,
                    jockeyId ?? (object)DBNull.Value,
                    draft.Saddlecloth ?? (object)DBNull.Value,
                    draft.Draw ?? (object)DBNull.Value,
                    draft.Age ?? (object)DBNull.Value,
                    draft.WeightLbs ?? (object)DBNull.Value,
                    Truncate(draft.WeightText, 64) ?? (object)DBNull.Value,
                    draft.FinishPos ?? (object)DBNull.Value,
                    Truncate(draft.OutcomeCode, 32) ?? (object)DBNull.Value,
                    Truncate(draft.DistanceBeatenText, 64) ?? (object)DBNull.Value,
                    draft.DistanceBeatenLengths ?? (object)DBNull.Value,
                    Truncate(draft.SpFraction, 32) ?? (object)DBNull.Value,
                    draft.SpDecimal ?? (object)DBNull.Value,
                    Truncate(draft.FavTag, 8) ?? (object)DBNull.Value,
                    (object)DBNull.Value,
                    (object)DBNull.Value,
                    (object)DBNull.Value,
                    draft.Comment ?? (object)DBNull.Value,
                    "Kaggle",
                    draft.SourceRunnerId);
                insertedRunStyles.Add(new RunStyleInput(draft.RaceId, horseId, draft.Comment));
                inserted++;
            }

            if (table.Rows.Count > 0)
            {
                using var bulk = new SqlBulkCopy(sql)
                {
                    DestinationTableName = "RunnerResult",
                    BatchSize = 10_000,
                    BulkCopyTimeout = 600
                };
                foreach (DataColumn column in table.Columns)
                {
                    bulk.ColumnMappings.Add(column.ColumnName, column.ColumnName);
                }
                bulk.WriteToServer(table);
                RunStyleStore.Upsert(sql, insertedRunStyles);
            }

            _runners += inserted;
            _pending.Clear();
            _pendingHorses.Clear();
            _pendingTrainers.Clear();
            _pendingJockeys.Clear();

            if (_races % 500 == 0 || _pending.Count == 0)
            {
                _log($"[kaggle] progress: {_races} races, {_runners} runners");
            }
        }

        private int EnsureCourse(string name)
        {
            if (_courseIds.TryGetValue(name, out var id)) return id;

            double? latitude = null;
            double? longitude = null;
            string? timeZoneId = null;
            if (CourseLocationCatalog.TryMatch(name, out var location))
            {
                latitude = location.Latitude;
                longitude = location.Longitude;
                timeZoneId = location.TimeZoneId;
            }

            id = _sql.QuerySingle<int>(
                @"INSERT INTO Course(Name, Latitude, Longitude, TimeZoneId)
                  OUTPUT INSERTED.CourseId
                  VALUES(@Name, @Latitude, @Longitude, @TimeZoneId)",
                new
                {
                    Name = Truncate(name, 256),
                    Latitude = latitude,
                    Longitude = longitude,
                    TimeZoneId = timeZoneId
                });
            _courseIds[name] = id;
            return id;
        }

        private int InsertRace(int courseId, SqliteRow row)
        {
            var distanceText = (row.Distance ?? string.Empty).Trim();
            var going = NormalizeGoing(row.Going);
            var isAw = IsAllWeather(going);
            var classValue = ParseClass(row.Class);

            return _sql.QuerySingle<int>(
                @"INSERT INTO Race
                  (CourseId, RaceDate, ScheduledOff, Title, RaceType, Class, AgeRestriction,
                   Surface, Going, DistanceYards, DistanceText, RunnerCount, Status,
                   WinningTimeMs, WinningTimeText, Source, SourceRaceId)
                  OUTPUT INSERTED.RaceId
                  VALUES
                  (@CourseId, @RaceDate, @ScheduledOff, @Title, @RaceType, @Class, @AgeRestriction,
                   @Surface, @Going, @DistanceYards, @DistanceText, @RunnerCount, @Status,
                   @WinningTimeMs, @WinningTimeText, @Source, @SourceRaceId)",
                new
                {
                    CourseId = (short)courseId,
                    RaceDate = row.Date,
                    ScheduledOff = row.Off,
                    Title = Truncate((row.RaceName ?? string.Empty).Trim(), 512),
                    RaceType = Truncate(row.Type, 128),
                    Class = classValue,
                    AgeRestriction = Truncate(row.AgeBand, 64),
                    Surface = isAw ? "Allweather" : "Turf",
                    Going = going,
                    DistanceYards = (short?)ParseDistanceYards(distanceText),
                    DistanceText = Truncate(distanceText, 64),
                    RunnerCount = ToByte(row.Ran),
                    Status = "Result",
                    WinningTimeMs = ParseWinningTimeMs(row.Time),
                    WinningTimeText = Truncate(row.Time, 64),
                    Source = "Kaggle",
                    SourceRaceId = row.RaceId.ToString(CultureInfo.InvariantCulture)
                });
        }

        private void EnsureLookup(SqlConnection sql, string table, string idColumn, IEnumerable<string> names, Dictionary<string, int> cache)
        {
            var missing = names
                .Where(n => !string.IsNullOrWhiteSpace(n) && !cache.ContainsKey(n))
                .Select(n => Truncate(n, 256)!)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (missing.Count == 0) return;

            sql.Execute("CREATE TABLE #NewNames (Name NVARCHAR(256) COLLATE DATABASE_DEFAULT NOT NULL)");
            var data = new DataTable();
            data.Columns.Add("Name", typeof(string));
            foreach (var name in missing) data.Rows.Add(name);
            using (var bulk = new SqlBulkCopy(sql) { DestinationTableName = "#NewNames", BulkCopyTimeout = 600 })
            {
                bulk.WriteToServer(data);
            }

            sql.Execute($"INSERT INTO {table}(Name) SELECT DISTINCT n.Name FROM #NewNames n WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE t.Name = n.Name)");
            foreach (var row in sql.Query<IdName>($"SELECT t.{idColumn} AS Id, t.Name FROM {table} t INNER JOIN #NewNames n ON n.Name = t.Name"))
            {
                cache.TryAdd(CleanName(row.Name), row.Id);
            }
            sql.Execute("DROP TABLE #NewNames");
        }

        private static SqliteRow? ReadRow(SqliteDataReader reader)
        {
            try
            {
                var date = ParseDate(reader["date"]);
                if (date == null) return null;
                return new SqliteRow
                {
                    Date = date.Value.Date,
                    RaceId = Convert.ToInt64(reader["race_id"], CultureInfo.InvariantCulture),
                    Course = AsString(reader["course"]),
                    Off = ParseTime(AsString(reader["off"])),
                    RaceName = AsString(reader["race_name"]),
                    Type = AsString(reader["type"]),
                    Class = AsString(reader["class"]),
                    AgeBand = AsString(reader["age_band"]),
                    Distance = AsString(reader["dist"]),
                    Going = AsString(reader["going"]),
                    Ran = AsInt(reader["ran"]),
                    Num = AsInt(reader["num"]),
                    Pos = reader["pos"],
                    Draw = AsInt(reader["draw"]),
                    OvrBtn = reader["ovr_btn"],
                    Btn = reader["btn"],
                    Horse = AsString(reader["horse"]),
                    Age = AsInt(reader["age"]),
                    Weight = AsString(reader["wgt"]),
                    Time = AsString(reader["time"]),
                    Sp = AsString(reader["sp"]),
                    Jockey = AsString(reader["jockey"]),
                    Trainer = AsString(reader["trainer"]),
                    Comment = AsString(reader["comment"])
                };
            }
            catch
            {
                return null;
            }
        }

        private static readonly Regex BracketTextRegex = new(@"\s*\([^\)]*\)|\s*\[[^\]]*\]", RegexOptions.Compiled);
        private static string CleanName(string? value)
        {
            if (string.IsNullOrWhiteSpace(value)) return string.Empty;
            var cleaned = BracketTextRegex.Replace(value, string.Empty);
            cleaned = Regex.Replace(cleaned, @"\s+", " ").Trim();
            return cleaned;
        }

        private static string? CleanCourse(string? course)
        {
            if (string.IsNullOrWhiteSpace(course)) return null;
            var value = course.Trim();
            var country = Regex.Match(value, @"\(([A-Za-z]{2,3})\)\s*$");
            if (country.Success)
            {
                var code = country.Groups[1].Value.ToUpperInvariant();
                if (code is not ("IRE" or "GB" or "UK")) return null;
                value = value[..country.Index].Trim();
            }
            value = Regex.Replace(value, @"\s*\(AW(T)?\)\s*$", string.Empty, RegexOptions.IgnoreCase).Trim();
            value = value switch
            {
                "Epsom" => "Epsom Downs",
                "Chelmsford" => "Chelmsford City",
                "Newmarket (July)" => "Newmarket",
                "Wexford (RH)" => "Wexford",
                _ => value
            };
            return AllowedCourses.Contains(value) ? value : null;
        }

        private static readonly HashSet<string> AllowedCourses = new(StringComparer.OrdinalIgnoreCase)
        {
            "Aintree", "Ascot", "Ayr", "Ballinrobe", "Bangor-on-Dee", "Bath", "Bellewstown",
            "Beverley", "Brighton", "Carlisle", "Cartmel", "Catterick", "Chelmsford City",
            "Cheltenham", "Chepstow", "Chester", "Clonmel", "Cork", "Curragh", "Doncaster",
            "Down Royal", "Downpatrick", "Dundalk", "Epsom Downs", "Exeter", "Fairyhouse",
            "Fakenham", "Ffos Las", "Folkestone", "Fontwell", "Galway", "Goodwood", "Gowran Park",
            "Great Leighs", "Hamilton", "Haydock", "Hereford", "Hexham", "Huntingdon", "Kelso",
            "Kempton", "Kilbeggan", "Killarney", "Laytown", "Leicester", "Leopardstown",
            "Limerick", "Lingfield", "Listowel", "Ludlow", "Mallow", "Market Rasen",
            "Musselburgh", "Naas", "Navan", "Newbury", "Newcastle", "Newmarket", "Newton Abbot",
            "Nottingham", "Perth", "Phoenix Park", "Plumpton", "Pontefract", "Punchestown",
            "Redcar", "Ripon", "Roscommon", "Salisbury", "Sandown", "Sedgefield", "Sligo",
            "Southwell", "Stratford", "Taunton", "Thirsk", "Thurles", "Tipperary", "Towcester",
            "Tralee", "Tramore", "Uttoxeter", "Warwick", "Wetherby", "Wexford", "Wincanton",
            "Windsor", "Wolverhampton", "Worcester", "Yarmouth", "York"
        };

        private static DateTime? ParseDate(object? value)
        {
            if (value == null || value == DBNull.Value) return null;
            if (value is DateTime date) return date.Date;
            if (value is string text)
            {
                return DateTime.TryParse(text, CultureInfo.InvariantCulture, DateTimeStyles.None, out var parsed)
                    ? parsed.Date
                    : null;
            }
            if (value is long number)
            {
                if (number > 10_000_000)
                {
                    return DateTime.TryParseExact(number.ToString(CultureInfo.InvariantCulture), "yyyyMMdd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var compact)
                        ? compact.Date
                        : null;
                }
                return DateTime.FromOADate(number).Date;
            }
            if (value is double real) return DateTime.FromOADate(real).Date;
            return null;
        }

        private static TimeSpan? ParseTime(string? value)
        {
            if (string.IsNullOrWhiteSpace(value)) return null;
            return TimeSpan.TryParse(value.Trim(), CultureInfo.InvariantCulture, out var time) ? time : null;
        }

        private static string? AsString(object? value)
            => value == null || value == DBNull.Value ? null : Convert.ToString(value, CultureInfo.InvariantCulture)?.Trim();

        private static int? AsInt(object? value)
        {
            if (value == null || value == DBNull.Value) return null;
            if (value is long number) return (int)number;
            if (value is int integer) return integer;
            if (int.TryParse(Convert.ToString(value, CultureInfo.InvariantCulture), out var parsed)) return parsed;
            return null;
        }

        private static byte? ToByte(int? value)
            => value.HasValue && value.Value > 0 && value.Value <= byte.MaxValue ? (byte)value.Value : null;

        private static (short? finishPos, string? outcome) ParsePosition(object? raw)
        {
            if (raw == null || raw == DBNull.Value) return (null, null);
            var text = Convert.ToString(raw, CultureInfo.InvariantCulture)?.Trim();
            if (string.IsNullOrWhiteSpace(text)) return (null, null);
            if (short.TryParse(text, out var position) && position > 0 && position <= short.MaxValue)
            {
                return (position, null);
            }
            var normalized = new string(text.Where(char.IsLetter).ToArray()).ToUpperInvariant();
            if (string.IsNullOrEmpty(normalized)) return (null, null);
            normalized = normalized switch
            {
                "PULLEDUP" => "PU",
                "UNSEATEDRIDER" => "UR",
                "BROUGHTDOWN" => "BD",
                "REFUSED" => "REF",
                "RANOUT" => "RO",
                "DISQUALIFIED" => "DSQ",
                "SLIPPEDUP" => "SU",
                "NONRUNNER" => "NR",
                "WITHDRAWN" => "WD",
                _ => normalized
            };
            return (null, normalized);
        }

        private static (string? fraction, decimal? value, string? favTag) ParseSp(string? sp)
        {
            if (string.IsNullOrWhiteSpace(sp)) return (null, null, null);
            var text = sp.Trim().ToUpperInvariant().Replace(" ", string.Empty);
            if (text is "-" or "–" or "—" or "N/A" or "NA" or "0") return (null, null, null);

            string? fav = null;
            if (text.EndsWith("JF", StringComparison.Ordinal)) { fav = "JF"; text = text[..^2]; }
            else if (text.EndsWith("CF", StringComparison.Ordinal)) { fav = "CF"; text = text[..^2]; }
            else if (text.EndsWith("F", StringComparison.Ordinal)) { fav = "F"; text = text[..^1]; }
            else if (text.EndsWith("C", StringComparison.Ordinal)) { fav = "CF"; text = text[..^1]; }
            else if (text.EndsWith("J", StringComparison.Ordinal)) { fav = "JF"; text = text[..^1]; }

            decimal? value = null;
            if (text is "EVS" or "EVENS")
            {
                value = 2.0m;
            }
            else if (text.Contains('/'))
            {
                var parts = text.Split('/');
                if (parts.Length == 2 && decimal.TryParse(parts[0], NumberStyles.Number, CultureInfo.InvariantCulture, out var numerator)
                    && decimal.TryParse(parts[1], NumberStyles.Number, CultureInfo.InvariantCulture, out var denominator) && denominator != 0)
                {
                    value = Math.Round(1m + numerator / denominator, 3);
                }
            }
            else if (decimal.TryParse(text, NumberStyles.Number, CultureInfo.InvariantCulture, out var decimalValue))
            {
                value = decimalValue;
            }

            return (sp.Trim(), value, fav);
        }

        private static (string? text, decimal? lengths) ParseBeaten(object? btn, object? ovrBtn)
        {
            var raw = btn ?? ovrBtn;
            if (raw == null || raw == DBNull.Value) return (null, null);
            var text = Convert.ToString(raw, CultureInfo.InvariantCulture)?.Trim();
            if (string.IsNullOrWhiteSpace(text) || text is "-" or "–" or "—") return (null, null);

            var lower = text.ToLowerInvariant();
            if (lower is "nk" or "neck") return (text, 0.3m);
            if (lower is "hd" or "head") return (text, 0.2m);
            if (lower is "shd" or "short head" or "shorthead" or "s.h" or "sh") return (text, 0.1m);
            if (lower is "nse" or "nose") return (text, 0.05m);

            var mixed = Regex.Match(text, @"^(\d+)\s+(\d+)/(\d+)$");
            if (mixed.Success && int.TryParse(mixed.Groups[3].Value, out var den) && den != 0)
            {
                return (text, decimal.Parse(mixed.Groups[1].Value, CultureInfo.InvariantCulture)
                    + decimal.Parse(mixed.Groups[2].Value, CultureInfo.InvariantCulture) / den);
            }
            var fraction = Regex.Match(text, @"^(\d+)/(\d+)$");
            if (fraction.Success && int.TryParse(fraction.Groups[2].Value, out var fden) && fden != 0)
            {
                return (text, decimal.Round(decimal.Parse(fraction.Groups[1].Value, CultureInfo.InvariantCulture) / fden, 2));
            }
            if (decimal.TryParse(text, NumberStyles.Number, CultureInfo.InvariantCulture, out var numeric))
            {
                return (text, numeric);
            }
            return (text, null);
        }

        private static byte? ParseWeightLbs(string? weight)
        {
            if (string.IsNullOrWhiteSpace(weight)) return null;
            var match = Regex.Match(weight, @"(\d{1,2})\s*-\s*(\d{1,2})");
            if (!match.Success) match = Regex.Match(weight, @"(\d{1,2})\s*st\s*(\d{1,2})");
            if (!match.Success) return null;
            var total = int.Parse(match.Groups[1].Value) * 14 + int.Parse(match.Groups[2].Value);
            return total >= byte.MinValue && total <= byte.MaxValue ? (byte)total : null;
        }

        private static int ParseDistanceYards(string? distance)
        {
            if (string.IsNullOrWhiteSpace(distance)) return 0;
            var text = distance
                .Replace("¼", ".25")
                .Replace("½", ".5")
                .Replace("¾", ".75");
            // A bare fraction such as "3m¾f" became "3m.75f"; give it a leading zero.
            text = Regex.Replace(text, @"(?<![0-9])\.(\d+)", "0.$1");

            var yards = 0d;
            foreach (Match match in Regex.Matches(text, @"(\d+(?:\.\d+)?)\s*m", RegexOptions.IgnoreCase))
            {
                yards += double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) * 1760;
            }
            foreach (Match match in Regex.Matches(text, @"(\d+(?:\.\d+)?)\s*f", RegexOptions.IgnoreCase))
            {
                yards += double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) * 220;
            }
            foreach (Match match in Regex.Matches(text, @"(\d+(?:\.\d+)?)\s*y", RegexOptions.IgnoreCase))
            {
                yards += double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
            }
            return (int)Math.Round(yards);
        }

        private static int? ParseWinningTimeMs(string? time)
        {
            if (string.IsNullOrWhiteSpace(time)) return null;
            var text = time.Trim().ToLowerInvariant();
            var clock = Regex.Match(text, @"^(?:(\d+):)?(\d+(?:\.\d+)?)$");
            if (clock.Success)
            {
                var minutes = clock.Groups[1].Success ? int.Parse(clock.Groups[1].Value) : 0;
                var seconds = double.Parse(clock.Groups[2].Value, CultureInfo.InvariantCulture);
                return (int)Math.Round(minutes * 60_000 + seconds * 1000);
            }
            var verbose = Regex.Match(text, @"(\d+)\s*m\s*(\d+(?:\.\d+)?)\s*s");
            if (verbose.Success)
            {
                return (int)Math.Round(int.Parse(verbose.Groups[1].Value) * 60_000 + double.Parse(verbose.Groups[2].Value, CultureInfo.InvariantCulture) * 1000);
            }
            var secondsOnly = Regex.Match(text, @"^(\d+(?:\.\d+)?)s$");
            if (secondsOnly.Success)
            {
                return (int)Math.Round(double.Parse(secondsOnly.Groups[1].Value, CultureInfo.InvariantCulture) * 1000);
            }
            return null;
        }

        private static byte? ParseClass(string? raceClass)
        {
            if (string.IsNullOrWhiteSpace(raceClass)) return null;
            var match = Regex.Match(raceClass, @"\d");
            return match.Success ? byte.Parse(match.Value) : null;
        }

        private static string? NormalizeGoing(string? going)
        {
            if (string.IsNullOrWhiteSpace(going)) return going;
            var value = Regex.Replace(going, @"[,;]?\s*in places.*$", string.Empty, RegexOptions.IgnoreCase);
            value = Regex.Replace(value, @"\s+on\s+.*$", string.Empty, RegexOptions.IgnoreCase).Trim();
            return value.Length > 30 ? value[..30] : value;
        }

        private static bool IsAllWeather(string? going)
        {
            if (string.IsNullOrWhiteSpace(going)) return false;
            var value = going.ToLowerInvariant();
            return value.Contains("standard") || value.Contains("polytrack") || value.Contains("tapeta")
                || value.Contains("fibresand") || value.Contains("all-weather") || value.Contains("allweather");
        }

        private static string? Truncate(string? value, int length)
            => string.IsNullOrEmpty(value) || value.Length <= length ? value : value[..length];

        private static string RaceKey(int courseId, DateTime date, TimeSpan? off, string? title)
            => string.Create(CultureInfo.InvariantCulture, $"{(short)courseId}|{date:yyyyMMdd}|{(int)(off ?? TimeSpan.Zero).TotalMinutes}|{NormalizeTitle(title)}");

        private static string NormalizeTitle(string? title)
            => Regex.Replace((title ?? string.Empty).Trim().ToLowerInvariant(), @"\s+", " ");

        private static long RunnerKey(int raceId, int horseId) => ((long)raceId << 32) | (uint)horseId;

        private sealed class SqliteRow
        {
            public DateTime Date { get; set; }
            public long RaceId { get; set; }
            public string? Course { get; set; }
            public TimeSpan? Off { get; set; }
            public string? RaceName { get; set; }
            public string? Type { get; set; }
            public string? Class { get; set; }
            public string? AgeBand { get; set; }
            public string? Distance { get; set; }
            public string? Going { get; set; }
            public int? Ran { get; set; }
            public int? Num { get; set; }
            public object? Pos { get; set; }
            public int? Draw { get; set; }
            public object? OvrBtn { get; set; }
            public object? Btn { get; set; }
            public string? Horse { get; set; }
            public int? Age { get; set; }
            public string? Weight { get; set; }
            public string? Time { get; set; }
            public string? Sp { get; set; }
            public string? Jockey { get; set; }
            public string? Trainer { get; set; }
            public string? Comment { get; set; }
        }

        private sealed class RunnerDraft
        {
            public int RaceId { get; set; }
            public string SourceRunnerId { get; set; } = string.Empty;
            public string Horse { get; set; } = string.Empty;
            public string Trainer { get; set; } = string.Empty;
            public string Jockey { get; set; } = string.Empty;
            public byte? Saddlecloth { get; set; }
            public byte? Draw { get; set; }
            public byte? Age { get; set; }
            public byte? WeightLbs { get; set; }
            public string? WeightText { get; set; }
            public short? FinishPos { get; set; }
            public string? OutcomeCode { get; set; }
            public string? DistanceBeatenText { get; set; }
            public decimal? DistanceBeatenLengths { get; set; }
            public string? SpFraction { get; set; }
            public decimal? SpDecimal { get; set; }
            public string? FavTag { get; set; }
            public string? Comment { get; set; }
        }

        private sealed class IdName
        {
            public int Id { get; set; }
            public string Name { get; set; } = string.Empty;
        }

        private sealed class ExistingRace
        {
            public int RaceId { get; set; }
            public short CourseId { get; set; }
            public DateTime RaceDate { get; set; }
            public TimeSpan? ScheduledOff { get; set; }
            public string? Title { get; set; }
        }

        private sealed class ExistingRunner
        {
            public int RaceId { get; set; }
            public int HorseId { get; set; }
        }
    }
}
