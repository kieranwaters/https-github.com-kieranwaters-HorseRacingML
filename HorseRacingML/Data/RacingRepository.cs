using System.Data;
using System;
using Dapper;
using Microsoft.Data.SqlClient;
using HorseRacingML.Models;
using Microsoft.Extensions.Hosting;
using System.Diagnostics;
using System.Net.Sockets;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading;
using System.Globalization;
using System.Text;

namespace HorseRacingML.Data
{
    /// <summary>
    /// Simple repository for inserting parsed race data into SQL Server.
    /// </summary>
    public class RacingRepository : IRacingRepository
    {
        private readonly string _connectionString;
        private static readonly Regex BracketTextRegex =
            new Regex("\\s*\\([^\\)]*\\)|\\s*\\[[^\\]]*\\]", RegexOptions.Compiled);
        private const int GoingMaxLength = 30;
        private static readonly object SchemaLock = new();
        private static bool _raceScreenTableEnsured;
        private static bool _runnerFlowTableEnsured;
        private static bool _upcomingRaceTableEnsured;
        private readonly AsyncLocal<DayReportScopeState?> _dayReportScope = new();
        private IDbConnection OpenConnection()
        {
            const int maxAttempts = 3;
            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                var conn = new SqlConnection(_connectionString);
                try
                {
                    conn.Open();
                    return conn;
                }
                catch (SqlException ex) when (ex.Message.Contains("Timeout expired") && attempt < maxAttempts - 1)
                {
                    conn.Dispose();
                    Thread.Sleep(500);
                }
            }

            // Last attempt - let any exception bubble up to the caller
            var finalConn = new SqlConnection(_connectionString);
            finalConn.Open();
            return finalConn;
        }
        public void InsertRunnerFlow(RunnerFlow flow)
        {
            if (flow is null)
            {
                throw new ArgumentNullException(nameof(flow));
            }

            InsertRunnerFlows(new[] { flow });
        }
        private (IDbConnection Connection, DayReportScopeState? Scope, bool OwnsConnection) GetScopedConnection()
        {
            var scope = _dayReportScope.Value;
            if (scope != null)
            {
                return (scope.Connection, scope, false);
            }

            var connection = OpenConnection();
            return (connection, null, true);
        }

        private void EnsureRaceScreenTable(IDbConnection connection, DayReportScopeState? scope)
        {
            if (scope != null)
            {
                if (!scope.RaceScreenEnsured)
                {
                    EnsureRaceScreenTableExists(connection);
                    scope.RaceScreenEnsured = true;
                }

                return;
            }

            EnsureRaceScreenTableExists(connection);
        }

        private void EnsureRunnerFlowTable(IDbConnection connection, DayReportScopeState? scope)
        {
            if (scope != null)
            {
                if (!scope.RunnerFlowEnsured)
                {
                    EnsureRunnerFlowTableExists(connection);
                    scope.RunnerFlowEnsured = true;
                }

                return;
            }

            EnsureRunnerFlowTableExists(connection);
        }

        private void EnsureUpcomingRaceTable(IDbConnection connection, DayReportScopeState? scope)
        {
            if (scope != null)
            {
                if (!scope.UpcomingRaceEnsured)
                {
                    EnsureUpcomingRaceTableExists(connection);
                    scope.UpcomingRaceEnsured = true;
                }

                return;
            }

            EnsureUpcomingRaceTableExists(connection);
        }

        public IDisposable BeginDayReportScope()
        {
            var scopeState = new DayReportScopeState(OpenConnection());
            var previous = _dayReportScope.Value;
            _dayReportScope.Value = scopeState;
            return new DayReportScope(this, previous, scopeState);
        }

        private sealed class DayReportScopeState
        {
            public DayReportScopeState(IDbConnection connection)
            {
                Connection = connection ?? throw new ArgumentNullException(nameof(connection));
            }

            public IDbConnection Connection { get; }
            public bool RaceScreenEnsured { get; set; }
            public bool RunnerFlowEnsured { get; set; }
            public bool UpcomingRaceEnsured { get; set; }
        }

        private sealed class DayReportScope : IDisposable
        {
            private readonly RacingRepository _repository;
            private readonly DayReportScopeState? _previous;
            private readonly DayReportScopeState _current;
            private bool _disposed;

            public DayReportScope(RacingRepository repository, DayReportScopeState? previous, DayReportScopeState current)
            {
                _repository = repository;
                _previous = previous;
                _current = current;
            }

            public void Dispose()
            {
                if (_disposed)
                {
                    return;
                }

                _disposed = true;

                try
                {
                    _current.Connection.Dispose();
                }
                finally
                {
                    _repository._dayReportScope.Value = _previous;
                }
            }
        }
        public void InsertRunnerFlows(IEnumerable<RunnerFlow> flows)
        {
            if (flows is null)
            {
                throw new ArgumentNullException(nameof(flows));
            }

            var flowList = flows.ToList();
            if (flowList.Any(f => f is null))
            {
                throw new ArgumentException("Flow collection cannot contain null entries.", nameof(flows));
            }
            if (flowList.Count == 0)
            {
                return;
            }

            const string header = @"INSERT INTO RunnerFlow(MarketId, UpcomingRaceId, RaceDate, ScheduledOff, VenueName, VenueCountry, RaceTitle, RaceDetails, RaceType, Surface, Going, DistanceYards, DistanceText, RunnerCount, BackBookPercentage, LayBookPercentage, ClothNumber, Draw, HorseName, JockeyName, BackPrice1, BackPrice2, BackPrice3, LayPrice1, LayPrice2, LayPrice3, AiOdds, Age, WeightLbs, WeightText, TrainerName)
VALUES";
      

            var sqlBuilder = new StringBuilder(header.Length + flowList.Count * 128);
            sqlBuilder.Append(header);

            var parameters = new DynamicParameters();

            for (var i = 0; i < flowList.Count; i++)
            {
                var flow = flowList[i];
                var suffix = i.ToString(CultureInfo.InvariantCulture);

                sqlBuilder.Append("(@MarketId").Append(suffix)
                    .Append(", @UpcomingRaceId").Append(suffix)
                    .Append(", @RaceDate").Append(suffix)
                    .Append(", @ScheduledOff").Append(suffix)
                    .Append(", @VenueName").Append(suffix)
                    .Append(", @VenueCountry").Append(suffix)
                    .Append(", @RaceTitle").Append(suffix)
                    .Append(", @RaceDetails").Append(suffix)
                    .Append(", @RaceType").Append(suffix)
                    .Append(", @Surface").Append(suffix)
                    .Append(", @Going").Append(suffix)
                    .Append(", @DistanceYards").Append(suffix)
                    .Append(", @DistanceText").Append(suffix)
                    .Append(", @RunnerCount").Append(suffix)
                    .Append(", @BackBookPercentage").Append(suffix)
                    .Append(", @LayBookPercentage").Append(suffix)
                    .Append(", @ClothNumber").Append(suffix)
                    .Append(", @Draw").Append(suffix)
                    .Append(", @HorseName").Append(suffix)
                    .Append(", @JockeyName").Append(suffix)
                    .Append(", @BackPrice1").Append(suffix)
                    .Append(", @BackPrice2").Append(suffix)
                    .Append(", @BackPrice3").Append(suffix)
                    .Append(", @LayPrice1").Append(suffix)
                    .Append(", @LayPrice2").Append(suffix)
                    .Append(", @LayPrice3").Append(suffix)
                    .Append(", @AiOdds").Append(suffix)
                    .Append(", @Age").Append(suffix)
                    .Append(", @WeightLbs").Append(suffix)
                    .Append(", @WeightText").Append(suffix)
                    .Append(", @TrainerName").Append(suffix)
                    .Append(')');

                if (i < flowList.Count - 1)
                {
                    sqlBuilder.Append(", ");
                }

                parameters.Add($"MarketId{suffix}", flow.MarketId);
                parameters.Add($"UpcomingRaceId{suffix}", flow.UpcomingRaceId);
                parameters.Add($"RaceDate{suffix}", flow.RaceDate);
                parameters.Add($"ScheduledOff{suffix}", flow.ScheduledOff);
                parameters.Add($"VenueName{suffix}", flow.VenueName);
                parameters.Add($"VenueCountry{suffix}", flow.VenueCountry);
                parameters.Add($"RaceTitle{suffix}", flow.RaceTitle);
                parameters.Add($"RaceDetails{suffix}", flow.RaceDetails);
                parameters.Add($"RaceType{suffix}", flow.RaceType);
                parameters.Add($"Surface{suffix}", flow.Surface);
                parameters.Add($"Going{suffix}", flow.Going);
                parameters.Add($"DistanceYards{suffix}", flow.DistanceYards);
                parameters.Add($"DistanceText{suffix}", flow.DistanceText);
                parameters.Add($"RunnerCount{suffix}", flow.RunnerCount);
                parameters.Add($"BackBookPercentage{suffix}", flow.BackBookPercentage);
                parameters.Add($"LayBookPercentage{suffix}", flow.LayBookPercentage);
                parameters.Add($"ClothNumber{suffix}", flow.ClothNumber);
                parameters.Add($"Draw{suffix}", flow.Draw);
                parameters.Add($"HorseName{suffix}", flow.HorseName);
                parameters.Add($"JockeyName{suffix}", flow.JockeyName);
                parameters.Add($"BackPrice1{suffix}", flow.BackPrice1);
                parameters.Add($"BackPrice2{suffix}", flow.BackPrice2);
                parameters.Add($"BackPrice3{suffix}", flow.BackPrice3);
                parameters.Add($"LayPrice1{suffix}", flow.LayPrice1);
                parameters.Add($"LayPrice2{suffix}", flow.LayPrice2);
                parameters.Add($"LayPrice3{suffix}", flow.LayPrice3);
                parameters.Add($"AiOdds{suffix}", flow.AiOdds);
                parameters.Add($"Age{suffix}", flow.Age);
                parameters.Add($"WeightLbs{suffix}", flow.WeightLbs);
                parameters.Add($"WeightText{suffix}", flow.WeightText);
                parameters.Add($"TrainerName{suffix}", flow.TrainerName);
            }

            var (connection, scope, ownsConnection) = GetScopedConnection();
            IDbTransaction? transaction = null;
            try
            {
                EnsureRunnerFlowTable(connection, scope);
                transaction = connection.BeginTransaction();
                connection.Execute(sqlBuilder.ToString(), parameters, transaction);
                transaction.Commit();
            }
            catch
            {
                try
                {
                    transaction?.Rollback();
                }
                catch
                {
                    // Ignore rollback failures.
                }

                throw;
            }
            finally
            {
                transaction?.Dispose();
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }
        public int UpsertUpcomingRace(UpcomingRace race)
        {
            if (race is null)
            {
                throw new ArgumentNullException(nameof(race));
            }

            StripBracketedText(race);
            race.Going = NormalizeGoing(race.Going);

            const string sql = @"
SET NOCOUNT ON;

DECLARE @ExistingId INT;

SELECT TOP (1) @ExistingId = UpcomingRaceId
FROM UpcomingRaces
WHERE MarketId = @MarketId;

IF @ExistingId IS NULL AND @RaceDate IS NOT NULL
BEGIN
    SELECT TOP (1) @ExistingId = UpcomingRaceId
    FROM UpcomingRaces
    WHERE RaceDate = @RaceDate
      AND ISNULL(LTRIM(RTRIM(VenueName)), '') = ISNULL(LTRIM(RTRIM(@VenueName)), '')
      AND ISNULL(ScheduledOff, '00:00:00') = ISNULL(@ScheduledOff, '00:00:00');
END;

IF @ExistingId IS NULL
BEGIN
    INSERT INTO UpcomingRaces
    (
        MarketId,
        RaceDate,
        ScheduledOff,
        VenueName,
        VenueCountry,
        Title,
        RaceDetails,
        RaceType,
        Surface,
        Going,
        DistanceYards,
        DistanceText,
        RunnerCount,
        BackBookPercentage,
        LayBookPercentage
    )
    VALUES
    (
        @MarketId,
        @RaceDate,
        @ScheduledOff,
        @VenueName,
        @VenueCountry,
        @Title,
        @RaceDetails,
        @RaceType,
        @Surface,
        @Going,
        @DistanceYards,
        @DistanceText,
        @RunnerCount,
        @BackBookPercentage,
        @LayBookPercentage
    );

    SELECT CAST(SCOPE_IDENTITY() AS INT);
END
ELSE
BEGIN
    UPDATE UpcomingRaces
    SET RaceDate = COALESCE(@RaceDate, RaceDate),
        ScheduledOff = @ScheduledOff,
        VenueName = @VenueName,
        VenueCountry = @VenueCountry,
        Title = @Title,
        RaceDetails = @RaceDetails,
        RaceType = @RaceType,
        Surface = @Surface,
        Going = @Going,
        DistanceYards = @DistanceYards,
        DistanceText = @DistanceText,
        RunnerCount = @RunnerCount,
        BackBookPercentage = @BackBookPercentage,
        LayBookPercentage = @LayBookPercentage,
        LastUpdatedUtc = SYSUTCDATETIME()
    WHERE UpcomingRaceId = @ExistingId;

    SELECT @ExistingId;
END;";

            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                EnsureUpcomingRaceTable(connection, scope);
                return connection.QuerySingle<int>(sql, race);
            }
            finally
            {
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }
        private static void EnsureUpcomingRaceTableExists(IDbConnection conn)
        {
            if (_upcomingRaceTableEnsured)
            {
                return;
            }

            lock (SchemaLock)
            {
                if (_upcomingRaceTableEnsured)
                {
                    return;
                }

                const string sql = @"
IF OBJECT_ID(N'dbo.UpcomingRaces', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.UpcomingRaces
    (
        UpcomingRaceId      INT           IDENTITY(1,1) PRIMARY KEY,
        MarketId            NVARCHAR(32)  NOT NULL,
        RaceDate            DATE          NOT NULL,
        ScheduledOff        TIME(0)       NULL,
        VenueName           NVARCHAR(256) NULL,
        VenueCountry        NVARCHAR(128) NULL,
        Title               NVARCHAR(512) NULL,
        RaceDetails         NVARCHAR(MAX) NULL,
        RaceType            NVARCHAR(128) NULL,
        Surface             NVARCHAR(64)  NULL,
        Going               NVARCHAR(30)  NULL,
        DistanceYards       SMALLINT      NULL,
        DistanceText        NVARCHAR(64)  NULL,
        RunnerCount         TINYINT       NULL,
        BackBookPercentage  DECIMAL(9,2)  NULL,
        LayBookPercentage   DECIMAL(9,2)  NULL,
        CreatedUtc          DATETIME2(0)  NOT NULL DEFAULT SYSUTCDATETIME(),
        LastUpdatedUtc      DATETIME2(0)  NOT NULL DEFAULT SYSUTCDATETIME()
    );
END;
IF COL_LENGTH(N'dbo.UpcomingRaces', 'Class') IS NOT NULL
BEGIN
    ALTER TABLE dbo.UpcomingRaces DROP COLUMN Class;
END;

IF COL_LENGTH(N'dbo.UpcomingRaces', 'AgeRestriction') IS NOT NULL
BEGIN
    ALTER TABLE dbo.UpcomingRaces DROP COLUMN AgeRestriction;
END;
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_UpcomingRaces_MarketId' AND object_id = OBJECT_ID(N'dbo.UpcomingRaces'))
BEGIN
    CREATE UNIQUE INDEX IX_UpcomingRaces_MarketId ON dbo.UpcomingRaces(MarketId);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_UpcomingRaces_RaceDateVenue' AND object_id = OBJECT_ID(N'dbo.UpcomingRaces'))
BEGIN
    CREATE INDEX IX_UpcomingRaces_RaceDateVenue ON dbo.UpcomingRaces(RaceDate, VenueName);
END;";

                conn.Execute(sql);
                _upcomingRaceTableEnsured = true;
            }
        }
        public int InsertRace(Race race)
        {
            StripBracketedText(race);
            race.Going = NormalizeGoing(race.Going);
            const string sql = @"
DECLARE @NormalizedTitle NVARCHAR(512) = LTRIM(RTRIM(ISNULL(@Title,'')));
DECLARE @ExistingId INT;

SELECT TOP 1 @ExistingId = RaceId
FROM Race
WHERE CourseId=@CourseId
  AND RaceDate=@RaceDate
  AND ScheduledOff=@ScheduledOff
  AND (
        @NormalizedTitle = ''
        OR LTRIM(RTRIM(ISNULL(Title,''))) = @NormalizedTitle
        OR LTRIM(RTRIM(ISNULL(Title,''))) = ''
      )
ORDER BY RaceId;

IF @ExistingId IS NOT NULL
BEGIN
    UPDATE Race
    SET Title = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(Title,'')))) = 0 AND LEN(@NormalizedTitle) > 0 THEN @Title ELSE Title END,
        RaceType = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(RaceType,'')))) = 0 AND LEN(LTRIM(RTRIM(ISNULL(@RaceType,'')))) > 0 THEN @RaceType ELSE RaceType END,
        Class = COALESCE(Class, @Class),
        AgeRestriction = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(AgeRestriction,'')))) = 0 AND LEN(LTRIM(RTRIM(ISNULL(@AgeRestriction,'')))) > 0 THEN @AgeRestriction ELSE AgeRestriction END,
        Surface = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(Surface,'')))) = 0 AND LEN(LTRIM(RTRIM(ISNULL(@Surface,'')))) > 0 THEN @Surface ELSE Surface END,
        Going = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(Going,'')))) = 0 AND LEN(LTRIM(RTRIM(ISNULL(@Going,'')))) > 0 THEN @Going ELSE Going END,
        DistanceYards = CASE WHEN ISNULL(DistanceYards, 0) = 0 AND @DistanceYards > 0 THEN @DistanceYards ELSE DistanceYards END,
        DistanceText = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(DistanceText,'')))) = 0 AND LEN(LTRIM(RTRIM(ISNULL(@DistanceText,'')))) > 0 THEN @DistanceText ELSE DistanceText END,
        RunnerCount = CASE WHEN RunnerCount IS NULL AND @RunnerCount IS NOT NULL THEN @RunnerCount ELSE RunnerCount END,
        Status = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(Status,'')))) = 0 AND LEN(LTRIM(RTRIM(ISNULL(@Status,'')))) > 0 THEN @Status ELSE Status END,
        WinningTimeMs = COALESCE(WinningTimeMs, @WinningTimeMs),
        WinningTimeText = CASE WHEN LEN(LTRIM(RTRIM(ISNULL(WinningTimeText,'')))) = 0 AND LEN(LTRIM(RTRIM(ISNULL(@WinningTimeText,'')))) > 0 THEN @WinningTimeText ELSE WinningTimeText END,
        ActualOff = COALESCE(ActualOff, @ActualOff)
    WHERE RaceId=@ExistingId;

    SELECT @ExistingId;
END
ELSE
BEGIN
    INSERT INTO Race(CourseId, RaceDate, ScheduledOff, ActualOff, Title, RaceType, Class, AgeRestriction, Surface, Going, DistanceYards, DistanceText, RunnerCount, Status, WinningTimeMs, WinningTimeText)
    VALUES(@CourseId, @RaceDate, @ScheduledOff, @ActualOff, @Title, @RaceType, @Class, @AgeRestriction, @Surface, @Going, @DistanceYards, @DistanceText, @RunnerCount, @Status, @WinningTimeMs, @WinningTimeText);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, race);
        }

        public void BulkInsertRunnerResults(IEnumerable<RunnerResult> results)
        {
            var list = results?.ToList();
            if (list == null || list.Count == 0) return;

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

            foreach (var r in list)
            {
                StripBracketedText(r);
                table.Rows.Add(
                    r.RaceId,
                    r.HorseId,
                    r.TrainerId ?? (object)DBNull.Value,
                    r.JockeyId ?? (object)DBNull.Value,
                    r.SaddleclothNumber ?? (object)DBNull.Value,
                    r.Draw ?? (object)DBNull.Value,
                    r.Age ?? (object)DBNull.Value,
                    r.WeightLbs ?? (object)DBNull.Value,
                    r.WeightText ?? (object)DBNull.Value,
                    r.FinishPos ?? (object)DBNull.Value,
                    r.OutcomeCode ?? (object)DBNull.Value,
                    r.DistanceBeatenText ?? (object)DBNull.Value,
                    r.DistanceBeatenLengths ?? (object)DBNull.Value,
                    r.SP_Fraction ?? (object)DBNull.Value,
                    r.SP_Decimal ?? (object)DBNull.Value,
                    r.FavTag ?? (object)DBNull.Value,
                    r.OpeningFraction ?? (object)DBNull.Value,
                    r.TouchedHighFraction ?? (object)DBNull.Value,
                    r.TouchedLowFraction ?? (object)DBNull.Value,
                    r.Comment ?? (object)DBNull.Value
                );
            }

            using var conn = new SqlConnection(_connectionString);
            conn.Open();
            using var bulk = new SqlBulkCopy(conn)
            {
                DestinationTableName = "RunnerResult"
            };

            foreach (DataColumn col in table.Columns)
            {
                bulk.ColumnMappings.Add(col.ColumnName, col.ColumnName);
            }

            bulk.WriteToServer(table);
        }
        public int? GetHistoricalRaceCountByHorseName(string? horseName)
        {
            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return null;
            }

            const string sql = @"
SELECT COUNT(*)
FROM RunnerResult rr
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names;";

            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, new { Names = candidates.ToArray() });
        }

        public HistoricalRaceCountPrefetchResult GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
        {
            if (horseNames == null)
            {
                throw new ArgumentNullException(nameof(horseNames));
            }

            var originalNames = new HashSet<string>(horseNames.Where(n => !string.IsNullOrWhiteSpace(n)), StringComparer.OrdinalIgnoreCase);
            if (originalNames.Count == 0)
            {
                return HistoricalRaceCountPrefetchResult.Empty;
            }

            // Map every generated candidate back to the original horse name so we can
            // attribute query results correctly and later expose all candidate spellings.
            var candidateMap = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
            foreach (var original in originalNames)
            {
                foreach (var candidate in BuildHistoricalNameCandidates(original))
                {
                    if (!candidateMap.TryGetValue(candidate, out var originals))
                    {
                        originals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                        candidateMap[candidate] = originals;
                    }

                    originals.Add(original);
                }
            }

            if (candidateMap.Count == 0)
            {
                return HistoricalRaceCountPrefetchResult.Empty;
            }

            using var conn = OpenConnection();
            var horseIdsByOriginal = new Dictionary<string, HashSet<int>>(StringComparer.OrdinalIgnoreCase);

            static void AddHorseId(Dictionary<string, HashSet<int>> lookup, string original, int horseId)
            {
                if (horseId <= 0)
                {
                    return;
                }

                if (!lookup.TryGetValue(original, out var ids))
                {
                    ids = new HashSet<int>();
                    lookup[original] = ids;
                }

                ids.Add(horseId);
            }

            const string horseSql = @"
SELECT Name,
       HorseId
FROM Horse
WHERE Name IN @Names;";

            var candidateList = candidateMap.Keys.ToArray();
            if (candidateList.Length > 0)
            {
                foreach (var (Name, HorseId) in conn.Query<(string Name, int HorseId)>(horseSql, new { Names = candidateList }))
                {
                    if (!candidateMap.TryGetValue(Name, out var originals) || originals == null)
                    {
                        continue;
                    }

                    foreach (var original in originals)
                    {
                        AddHorseId(horseIdsByOriginal, original, HorseId);
                    }
                }
            }

            var unmatchedOriginals = originalNames
                .Where(name => !horseIdsByOriginal.ContainsKey(name))
                .ToList();

            if (unmatchedOriginals.Count > 0)
            {
                var normalizedMap = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
                foreach (var original in unmatchedOriginals)
                {
                    foreach (var candidate in BuildHistoricalNameCandidates(original))
                    {
                        var normalized = NormalizeHistoricalNameKey(candidate);
                        if (string.IsNullOrEmpty(normalized))
                        {
                            continue;
                        }

                        if (!normalizedMap.TryGetValue(normalized, out var originals))
                        {
                            originals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                            normalizedMap[normalized] = originals;
                        }

                        originals.Add(original);
                    }
                }

                if (normalizedMap.Count > 0)
                {
                    const string normalizedHorseSql = @"
SELECT lookup.Normalized,
       h.HorseId
FROM Horse h
CROSS APPLY (
    SELECT Normalized = LOWER(
        REPLACE(
            REPLACE(
                REPLACE(
                    REPLACE(
                        REPLACE(
                            REPLACE(
                                REPLACE(
                                    REPLACE(
                                        REPLACE(
                                            REPLACE(h.Name, ' ', ''),
                                            '-', ''
                                        ),
                                        CHAR(39), ''
                                    ),
                                    NCHAR(8217), ''
                                ),
                                '.', ''
                            ),
                            ',', ''
                        ),
                        '&', 'and'
                    ),
                    '(', ''
                ),
                ')', ''
            ),
            '/', ''
        )
    )
) AS lookup
WHERE lookup.Normalized IN @Names;";

                    var normalizedKeys = normalizedMap.Keys.ToArray();
                    foreach (var (Normalized, HorseId) in conn.Query<(string Normalized, int HorseId)>(normalizedHorseSql, new { Names = normalizedKeys }))
                    {
                        if (!normalizedMap.TryGetValue(Normalized, out var originals) || originals == null)
                        {
                            continue;
                        }

                        foreach (var original in originals)
                        {
                            AddHorseId(horseIdsByOriginal, original, HorseId);
                        }
                    }
                }
            }

            if (horseIdsByOriginal.Count == 0)
            {
                return HistoricalRaceCountPrefetchResult.Empty;
            }

            var distinctHorseIds = horseIdsByOriginal
                .SelectMany(kvp => kvp.Value)
                .Where(id => id > 0)
                .Distinct()
                .ToArray();

            if (distinctHorseIds.Length == 0)
            {
                return HistoricalRaceCountPrefetchResult.Empty;
            }

            const string countSql = @"
SELECT rr.HorseId,
       COUNT(*) AS RaceCount,
       SUM(CASE WHEN rr.FinishPos = 1 THEN 1 ELSE 0 END) AS WinCount
FROM RunnerResult rr
WHERE rr.HorseId IN @HorseIds
GROUP BY rr.HorseId;";

            var countsByHorseId = new Dictionary<int, int>();
            var winsByHorseId = new Dictionary<int, int>();
            foreach (var row in conn.Query(countSql, new { HorseIds = distinctHorseIds }))
            {
                var horseId = (int)row.HorseId;
                var count = (int)row.RaceCount;
                var wins = (int)row.WinCount;
                countsByHorseId[horseId] = count;
                winsByHorseId[horseId] = wins;
            }

            var countsByOriginal = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var winsByOriginal = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var (original, horseIds) in horseIdsByOriginal)
            {
                if (horseIds == null || horseIds.Count == 0)
                {
                    continue;
                }

                var total = 0;
                var totalWins = 0;
                foreach (var horseId in horseIds)
                {
                    if (countsByHorseId.TryGetValue(horseId, out var count))
                    {
                        total += count;
                    }
                    if (winsByHorseId.TryGetValue(horseId, out var wins))
                    {
                        totalWins += wins;
                    }
                }

                countsByOriginal[original] = total;
                winsByOriginal[original] = totalWins;
            }

            if (countsByOriginal.Count == 0)
            {
                return HistoricalRaceCountPrefetchResult.Empty;
            }

            // Expose counts for every candidate spelling so callers can resolve matches
            // using their preferred representation.
            var results = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var winResults = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var (original, count) in countsByOriginal)
            {
                winsByOriginal.TryGetValue(original, out var wins);
                foreach (var candidate in BuildHistoricalNameCandidates(original))
                {
                    if (!results.ContainsKey(candidate))
                    {
                        results[candidate] = count;
                    }
                    if (!winResults.ContainsKey(candidate))
                    {
                        winResults[candidate] = wins;
                    }
                }
            }

            return new HistoricalRaceCountPrefetchResult(
                new ReadOnlyDictionary<string, int>(results),
                new ReadOnlyDictionary<string, int>(countsByOriginal),
                new ReadOnlyDictionary<string, int>(winResults),
                new ReadOnlyDictionary<string, int>(winsByOriginal),
                matchedHorseIdCount: distinctHorseIds.Length);
        }
        public int GetHistoricalWinCountByHorseName(string? horseName)
        {
            if (string.IsNullOrWhiteSpace(horseName))
            {
                return 0;
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return 0;
            }

            const string sql = @"
SELECT SUM(CASE WHEN rr.FinishPos = 1 THEN 1 ELSE 0 END)
FROM RunnerResult rr
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names;";

            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, new { Names = candidates.ToArray() });
        }
        internal static IReadOnlyCollection<string> BuildHistoricalNameCandidates(string? horseName)
        {
            var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            if (string.IsNullOrWhiteSpace(horseName))
            {
                return candidates;
            }

            var trimmed = horseName.Trim();
            if (!string.IsNullOrEmpty(trimmed))
            {
                candidates.Add(trimmed);
            }

            var normalized = RemoveBracketedText(trimmed);
            if (!string.IsNullOrEmpty(normalized))
            {
                candidates.Add(normalized);
            }
            var key = NormalizeHistoricalNameKey(trimmed);
            if (!string.IsNullOrEmpty(key))
            {
                candidates.Add(key);
            }
            if (!string.IsNullOrEmpty(normalized))
            {
                var tokens = normalized
                    .Split(new[] { ' ', '\t', '\u00A0' }, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length > 1)
                {
                    var surname = tokens[^1].Trim();
                    if (!string.IsNullOrEmpty(surname))
                    {
                        static char? TryGetInitial(string token)
                        {
                            foreach (var c in token)
                            {
                                if (char.IsLetter(c))
                                {
                                    return char.ToUpperInvariant(c);
                                }
                            }

                            return null;
                        }

                        var initials = new List<char>();
                        for (int i = 0; i < tokens.Length - 1; i++)
                        {
                            var initial = TryGetInitial(tokens[i]);
                            if (initial.HasValue)
                            {
                                initials.Add(initial.Value);
                            }
                        }

                        if (initials.Count > 0)
                        {
                            var primaryInitial = initials[0].ToString();
                            candidates.Add($"{primaryInitial} {surname}");
                            candidates.Add($"{primaryInitial}{surname}");

                            if (initials.Count > 1)
                            {
                                var combined = new string(initials.ToArray());
                                candidates.Add($"{combined} {surname}");
                                candidates.Add($"{combined}{surname}");
                            }
                        }
                    }
                }
            }
            return candidates;
        }
        internal static string NormalizeHistoricalNameKey(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var withoutBracketed = RemoveBracketedText(value).Trim();
            if (withoutBracketed.Length == 0)
            {
                return string.Empty;
            }

            var normalized = withoutBracketed.Normalize(NormalizationForm.FormD);
            var builder = new StringBuilder(normalized.Length);

            foreach (var c in normalized)
            {
                var category = CharUnicodeInfo.GetUnicodeCategory(c);
                if (category == UnicodeCategory.NonSpacingMark ||
                    category == UnicodeCategory.SpacingCombiningMark ||
                    category == UnicodeCategory.EnclosingMark)
                {
                    continue;
                }

                if (char.IsLetterOrDigit(c))
                {
                    builder.Append(char.ToLowerInvariant(c));
                    continue;
                }

                if (char.IsWhiteSpace(c))
                {
                    continue;
                }

                if (c == '&')
                {
                    builder.Append("and");
                }
            }

            return builder.ToString();
        }
        public UpcomingRace? GetUpcomingRaceByMarketId(string? marketId)
        {
            if (string.IsNullOrWhiteSpace(marketId))
            {
                return null;
            }

            const string sql = @"SELECT UpcomingRaceId,
                                           MarketId,
                                           RaceDate,
                                           ScheduledOff,
                                           VenueName,
                                           VenueCountry,
                                           Title,
                                           RaceDetails,
                                           RaceType,
                                           Surface,
                                           Going,
                                           DistanceYards,
                                           DistanceText,
                                           RunnerCount,
                                           BackBookPercentage,
                                           LayBookPercentage
                                    FROM UpcomingRaces
                                    WHERE MarketId = @MarketId";

            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                EnsureUpcomingRaceTable(connection, scope);
                return connection.QueryFirstOrDefault<UpcomingRace>(sql, new { MarketId = marketId.Trim() });
            }
            finally
            {
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }
        

        public static string NormalizeLookupKey(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var lower = value.Trim().ToLowerInvariant();
            lower = Regex.Replace(lower, "[^a-z0-9]+", " ");
            lower = Regex.Replace(lower, "\\s+", " ").Trim();
            return lower;
        }
        public DateTime GetLatestRaceDate()
        {
            using var conn = OpenConnection();
            const string sql = "SELECT MAX(RaceDate) FROM Race WHERE RaceDate IS NOT NULL AND RaceDate <= CAST(GETDATE() AS date)";
            var latest = conn.QuerySingleOrDefault<DateTime?>(sql);
            // When no races exist yet, return a past date so scraping can begin
            return latest ?? new DateTime(2000, 1, 1);
        }
        public (double includingJoint, double excludingJoint, double logLoss) GetFavouriteAccuracy()
        {
            using var conn = OpenConnection();

            const string totalIncludingSql = @"SELECT COUNT(DISTINCT RaceId) FROM RunnerResult WHERE FavTag IN ('F','JF','CF')";
            const string winsIncludingSql = @"SELECT COUNT(DISTINCT RaceId) FROM RunnerResult WHERE FinishPos = 1 AND FavTag IN ('F','JF','CF')";

            int totalIncluding = conn.QuerySingle<int>(totalIncludingSql);
            int winsIncluding = conn.QuerySingle<int>(winsIncludingSql);

            const string totalExcludingSql = @"SELECT COUNT(*) FROM (
    SELECT RaceId
    FROM RunnerResult
    GROUP BY RaceId
    HAVING SUM(CASE WHEN FavTag='F' THEN 1 ELSE 0 END)=1 AND SUM(CASE WHEN FavTag IN ('JF','CF') THEN 1 ELSE 0 END)=0
) t";

            const string winsExcludingSql = @"SELECT COUNT(*) FROM (
    SELECT RaceId
    FROM RunnerResult
    GROUP BY RaceId
    HAVING SUM(CASE WHEN FavTag='F' THEN 1 ELSE 0 END)=1 AND SUM(CASE WHEN FavTag IN ('JF','CF') THEN 1 ELSE 0 END)=0
       AND MAX(CASE WHEN FinishPos=1 AND FavTag='F' THEN 1 ELSE 0 END)=1
) t";

            int totalExcluding = conn.QuerySingle<int>(totalExcludingSql);
            int winsExcluding = conn.QuerySingle<int>(winsExcludingSql);

            double includingJoint = totalIncluding == 0 ? 0 : (double)winsIncluding / totalIncluding;
            double excludingJoint = totalExcluding == 0 ? 0 : (double)winsExcluding / totalExcluding;

            const string logLossSql = @"SELECT SP_Decimal AS SP, CASE WHEN FinishPos = 1 THEN 1 ELSE 0 END AS Won FROM RunnerResult WHERE FavTag IN ('F','JF','CF') AND SP_Decimal IS NOT NULL AND SP_Decimal > 0";
            var rows = conn.Query<(double SP, int Won)>(logLossSql);

            double logLossSum = 0;
            int count = 0;
            const double epsilon = 1e-15;
            foreach (var row in rows)
            {
                double p = 1.0 / row.SP;
                p = Math.Max(Math.Min(p, 1 - epsilon), epsilon);
                double y = row.Won;
                double loss = -(y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
                logLossSum += loss;
                count++;
            }
            double logLoss = count == 0 ? 0 : logLossSum / count;

            return (includingJoint, excludingJoint, logLoss);
        }
        public void InsertRaceScreen(RaceScreen screen)
        {
            if (screen == null)
            {
                throw new ArgumentNullException(nameof(screen));
            }

            screen.Going = NormalizeGoing(screen.Going);
            const string sql = @"
INSERT INTO RaceScreen(MarketId, RaceDate, OffTime, Title, VenueName, VenueCountry, EventDateText, RaceDetails, RaceType, Going, BackBookPercentage, LayBookPercentage, RaceUrl)
VALUES(@MarketId, @RaceDate, @OffTime, @Title, @VenueName, @VenueCountry, @EventDateText, @RaceDetails, @RaceType, @Going, @BackBookPercentage, @LayBookPercentage, @RaceUrl);";
            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                EnsureRaceScreenTable(connection, scope);
                connection.Execute(sql, screen);
            }
            finally
            {
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }
        public void ClearDayReportTables()
        {
            var (connection, scope, ownsConnection) = GetScopedConnection();
            IDbTransaction? transaction = null;

            try
            {
                EnsureRaceScreenTable(connection, scope);
                EnsureRunnerFlowTable(connection, scope);
                EnsureUpcomingRaceTable(connection, scope);

                transaction = connection.BeginTransaction();

                ExecuteTruncateOrDelete(connection, transaction, "RunnerFlow");
                ExecuteTruncateOrDelete(connection, transaction, "RaceScreen");
                ExecuteTruncateOrDelete(connection, transaction, "UpcomingRaces");

                transaction.Commit();
            }
            catch
            {
                try
                {
                    transaction?.Rollback();
                }
                catch
                {
                    // Ignore rollback failures.
                }

                throw;
            }
            finally
            {
                transaction?.Dispose();

                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }

        private static void ExecuteTruncateOrDelete(IDbConnection connection, IDbTransaction transaction, string tableName)
        {
            if (connection is null)
            {
                throw new ArgumentNullException(nameof(connection));
            }

            if (transaction is null)
            {
                throw new ArgumentNullException(nameof(transaction));
            }

            if (string.IsNullOrWhiteSpace(tableName))
            {
                throw new ArgumentException("Table name must be provided.", nameof(tableName));
            }

            try
            {
                connection.Execute($"TRUNCATE TABLE [{tableName}];", transaction: transaction);
            }
            catch (SqlException ex) when (IsTruncateNotAllowed(ex))
            {
                connection.Execute($"DELETE FROM [{tableName}];", transaction: transaction);
            }
        }

        private static bool IsTruncateNotAllowed(SqlException ex)
        {
            // 4712: Cannot truncate table because it is being referenced by a FOREIGN KEY constraint.
            return ex.Number == 4712;
        }
        private static string? NormalizeGoing(string? going)
        {
            if (string.IsNullOrWhiteSpace(going)) return going;

            var g = Regex.Replace(going, @"[,;]?\s*in places.*$", string.Empty, RegexOptions.IgnoreCase);
            g = Regex.Replace(g, @"\s+on\s+.*$", string.Empty, RegexOptions.IgnoreCase);
            g = g.Trim();

            if (g.Length > GoingMaxLength)
            {
                g = g.Substring(0, GoingMaxLength);
            }
            return g;
        }
        private static void EnsureRaceScreenTableExists(IDbConnection conn)
        {
            if (_raceScreenTableEnsured) return;

            lock (SchemaLock)
            {
                if (_raceScreenTableEnsured) return;

                const string sql = @"
IF OBJECT_ID(N'dbo.RaceScreen', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.RaceScreen
    (
        RaceScreenId     BIGINT        IDENTITY(1,1) PRIMARY KEY,
        MarketId         NVARCHAR(32)  NULL,
        RaceDate         DATE          NULL,
        OffTime          TIME(0)       NULL,
        Title            NVARCHAR(512) NULL,
        VenueName        NVARCHAR(256) NULL,
        VenueCountry     NVARCHAR(128) NULL,
        EventDateText    NVARCHAR(128) NULL,
        RaceDetails      NVARCHAR(MAX) NULL,
        RaceType         NVARCHAR(128) NULL,
        Going            NVARCHAR(30)  NULL,
        BackBookPercentage DECIMAL(9,2) NULL,
        LayBookPercentage  DECIMAL(9,2) NULL,
        RaceUrl           NVARCHAR(1024) NULL
    );
END";

                conn.Execute(sql);
                const string addColumnSql = @"
IF COL_LENGTH('dbo.RaceScreen', 'RaceUrl') IS NULL
BEGIN
    ALTER TABLE dbo.RaceScreen ADD RaceUrl NVARCHAR(1024) NULL;
END";

                conn.Execute(addColumnSql);
                const string addRaceTypeSql = @"
IF COL_LENGTH('dbo.RaceScreen', 'RaceType') IS NULL
BEGIN
    ALTER TABLE dbo.RaceScreen ADD RaceType NVARCHAR(128) NULL;
END";

                conn.Execute(addRaceTypeSql);
                const string addGoingSql = @"
IF COL_LENGTH('dbo.RaceScreen', 'Going') IS NULL
BEGIN
    ALTER TABLE dbo.RaceScreen ADD Going NVARCHAR(30) NULL;
END";

                conn.Execute(addGoingSql);
                _raceScreenTableEnsured = true;
            }
        }
        private static void EnsureRunnerFlowTableExists(IDbConnection conn)
        {
            if (_runnerFlowTableEnsured) return;

            lock (SchemaLock)
            {
                if (_runnerFlowTableEnsured) return;

                const string sql = @"
IF OBJECT_ID(N'dbo.RunnerFlow', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.RunnerFlow
    (
        RunnerFlowId       BIGINT        IDENTITY(1,1) PRIMARY KEY,
        MarketId           NVARCHAR(32)  NULL,
        UpcomingRaceId     INT           NULL,
        RaceDate           DATE          NULL,
        ScheduledOff       TIME(0)       NULL,
        VenueName          NVARCHAR(256) NULL,
        VenueCountry       NVARCHAR(128) NULL,
        RaceTitle          NVARCHAR(512) NULL,
        RaceDetails        NVARCHAR(MAX) NULL,
        RaceType           NVARCHAR(128) NULL,
        Class              TINYINT       NULL,
        Surface            NVARCHAR(64)  NULL,
        Going              NVARCHAR(30)  NULL,
        DistanceYards      SMALLINT      NULL,
        DistanceText       NVARCHAR(64)  NULL,
        RunnerCount        TINYINT       NULL,
        BackBookPercentage DECIMAL(9,2)  NULL,
        LayBookPercentage  DECIMAL(9,2)  NULL,
        ClothNumber        TINYINT       NULL,
        Draw               TINYINT       NULL,
        HorseName          NVARCHAR(256) NULL,
        JockeyName         NVARCHAR(256) NULL,
        BackPrice1         DECIMAL(9,2)  NULL,
        BackPrice2         DECIMAL(9,2)  NULL,
        BackPrice3         DECIMAL(9,2)  NULL,
        LayPrice1          DECIMAL(9,2)  NULL,
        LayPrice2          DECIMAL(9,2)  NULL,
        LayPrice3          DECIMAL(9,2)  NULL,
        AiOdds             FLOAT         NULL,
        Age                TINYINT       NULL,
        WeightLbs          TINYINT       NULL,
        WeightText         NVARCHAR(32)  NULL,
        TrainerName        NVARCHAR(256) NULL
    );
END
IF COL_LENGTH(N'dbo.RunnerFlow', 'HorseName') IS NULL
BEGIN
    IF COL_LENGTH(N'dbo.RunnerFlow', 'RunnerName') IS NOT NULL
    BEGIN
        EXEC sp_rename 'dbo.RunnerFlow.RunnerName', 'HorseName', 'COLUMN';
    END
    ELSE
    BEGIN
        ALTER TABLE dbo.RunnerFlow ADD HorseName NVARCHAR(256) NULL;
    END
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'AiOdds') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD AiOdds FLOAT NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'Age') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD Age TINYINT NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'WeightLbs') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD WeightLbs TINYINT NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'WeightText') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD WeightText NVARCHAR(32) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'TrainerName') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD TrainerName NVARCHAR(256) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'UpcomingRaceId') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD UpcomingRaceId INT NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'RaceDate') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD RaceDate DATE NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'ScheduledOff') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD ScheduledOff TIME(0) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'VenueName') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD VenueName NVARCHAR(256) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'VenueCountry') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD VenueCountry NVARCHAR(128) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'RaceTitle') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD RaceTitle NVARCHAR(512) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'RaceDetails') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD RaceDetails NVARCHAR(MAX) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'RaceType') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD RaceType NVARCHAR(128) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'Class') IS NOT NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow DROP COLUMN Class;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'Surface') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD Surface NVARCHAR(64) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'Going') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD Going NVARCHAR(30) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'DistanceYards') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD DistanceYards SMALLINT NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'DistanceText') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD DistanceText NVARCHAR(64) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'RunnerCount') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD RunnerCount TINYINT NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'BackBookPercentage') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD BackBookPercentage DECIMAL(9,2) NULL;
END

IF COL_LENGTH(N'dbo.RunnerFlow', 'LayBookPercentage') IS NULL
BEGIN
    ALTER TABLE dbo.RunnerFlow ADD LayBookPercentage DECIMAL(9,2) NULL;
END";

                conn.Execute(sql);
                _runnerFlowTableEnsured = true;
            }
        }

        private static string RemoveBracketedText(string input)
            => string.IsNullOrWhiteSpace(input) ? input : BracketTextRegex.Replace(input, string.Empty).Trim();

        private static void StripBracketedText<T>(T obj)
        {
            if (obj == null) return;

            var stringProps = typeof(T).GetProperties()
                .Where(p => p.PropertyType == typeof(string) && p.CanRead && p.CanWrite);

            foreach (var prop in stringProps)
            {
                var value = prop.GetValue(obj) as string;
                if (!string.IsNullOrWhiteSpace(value))
                    prop.SetValue(obj, RemoveBracketedText(value));
            }
        }
        public RacingRepository(string connectionString)
        {
            _connectionString = connectionString;
        }

        public short InsertCourse(Course course)
        {
            StripBracketedText(course);
            const string sql = @"
IF EXISTS (SELECT 1 FROM Course WHERE Name = @Name AND ISNULL(Country,'') = ISNULL(@Country,''))
    SELECT TOP 1 CourseId FROM Course WHERE Name = @Name AND ISNULL(Country,'') = ISNULL(@Country,'') ORDER BY CourseId;
ELSE
BEGIN
    INSERT INTO Course(Name, Country) VALUES(@Name, @Country);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<short>(sql, course);
        }
        public int InsertMLParameter(MLParameter param)
        {
            const string sql = @"
INSERT INTO MLParameters(RunDate, Units, Dropout, Layers, LearningRate, TrainAccuracy, ValidationAccuracy, ValidationLoss, Epochs, BatchSize, Folds, TrainLoss, TrainBrier, Fold, ValidationBrier)
VALUES(@RunDate, @Units, @Dropout, @Layers, @LearningRate, @TrainAccuracy, @ValidationAccuracy, @ValidationLoss, @Epochs, @BatchSize, @Folds, @TrainLoss, @TrainBrier, @Fold, @ValidationBrier);
SELECT CAST(SCOPE_IDENTITY() as int);";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, param);
        }
        public List<int> GetRaceIdsBetweenDates(DateTime startInclusive, DateTime endInclusive)
        {
            if (endInclusive < startInclusive)
            {
                throw new ArgumentException("The end date must be on or after the start date.", nameof(endInclusive));
            }

            using var conn = OpenConnection();
            const string sql = @"SELECT RaceId
FROM Race
WHERE RaceDate >= @Start AND RaceDate <= @End
ORDER BY RaceDate, RaceId";
            return conn.Query<int>(sql, new { Start = startInclusive, End = endInclusive }).ToList();
        }

        public List<int> GetRaceIdsOutsideRange(DateTime startInclusive, DateTime endInclusive)
        {
            if (endInclusive < startInclusive)
            {
                throw new ArgumentException("The end date must be on or after the start date.", nameof(endInclusive));
            }

            using var conn = OpenConnection();
            const string sql = @"SELECT RaceId
FROM Race
WHERE RaceDate < @Start OR RaceDate > @End
ORDER BY RaceDate, RaceId";
            return conn.Query<int>(sql, new { Start = startInclusive, End = endInclusive }).ToList();
        }
        public IDictionary<int, RaceSummary> GetRaceSummaries(IEnumerable<int> raceIds)
        {
            if (raceIds is null)
            {
                throw new ArgumentNullException(nameof(raceIds));
            }

            var distinctIds = raceIds
                .Where(id => id > 0)
                .Distinct()
                .ToList();

            if (distinctIds.Count == 0)
            {
                return new Dictionary<int, RaceSummary>();
            }

            var result = new Dictionary<int, RaceSummary>();
            const int chunkSize = 2000; // stay below SQL parameter limit
            const string sql = @"SELECT r.RaceId, r.RaceDate, r.Title, c.Name AS CourseName
FROM Race r
LEFT JOIN Course c ON c.CourseId = r.CourseId
WHERE r.RaceId IN @Ids";

            using var conn = OpenConnection();

            for (int offset = 0; offset < distinctIds.Count; offset += chunkSize)
            {
                var chunk = distinctIds.Skip(offset).Take(chunkSize).ToArray();
                if (chunk.Length == 0)
                {
                    continue;
                }

                foreach (var summary in conn.Query<RaceSummary>(sql, new { Ids = chunk }))
                {
                    result[summary.RaceId] = summary;
                }
            }

            return result;
        }
        private sealed class HorseDistanceRow
        {
            public int? DistanceYards { get; set; }
            public double? AverageDistance { get; set; }
        }

        public HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName)
        {
            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return null;
            }

            const string sql = @"
WITH Distances AS (
    SELECT r.DistanceYards,
           r.RaceDate,
           rr.RunnerResultId,
           AVG(CAST(r.DistanceYards AS float)) OVER () AS AverageDistance
    FROM RunnerResult rr
    INNER JOIN Race r ON r.RaceId = rr.RaceId
    INNER JOIN Horse h ON h.HorseId = rr.HorseId
    WHERE h.Name IN @Names AND r.DistanceYards IS NOT NULL
)
SELECT TOP (1)
       DistanceYards,
       AverageDistance
FROM Distances
ORDER BY RaceDate DESC, RunnerResultId DESC;";

            using var conn = OpenConnection();
            var row = conn.QueryFirstOrDefault<HorseDistanceRow>(sql, new { Names = candidates.ToArray() });
            if (row == null)
            {
                return null;
            }

            return new HorseDistanceStats(row.DistanceYards, row.AverageDistance);
        }
        public int? GetLastWinningTimeMilliseconds(string? horseName, int? horseId, DateTime? beforeDate)
        {
            if (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName))
            {
                return null;
            }

            var cutoffDate = beforeDate?.Date;

            const string sqlById = @"SELECT TOP (1)
    r.WinningTimeMs
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND r.WinningTimeMs IS NOT NULL
  AND r.WinningTimeMs > 0
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (1)
    r.WinningTimeMs
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
  AND r.WinningTimeMs IS NOT NULL
  AND r.WinningTimeMs > 0
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.QueryFirstOrDefault<int?>(sqlById, new
                {
                    HorseId = horseId.Value,
                    BeforeDate = cutoffDate
                });

                if (byId.HasValue)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return null;
            }

            return conn.QueryFirstOrDefault<int?>(sqlByName, new
            {
                Names = candidates.ToArray(),
                BeforeDate = cutoffDate
            });
        }
        public byte? GetMostRecentRaceClass(string? horseName, int? horseId)
        {
            if (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName))
            {
                return null;
            }

            const string sqlById = @"SELECT TOP (1)
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND r.Class IS NOT NULL
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (1)
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
  AND r.Class IS NOT NULL
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.QueryFirstOrDefault<byte?>(sqlById, new
                {
                    HorseId = horseId.Value
                });

                if (byId.HasValue)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return null;
            }

            return conn.QueryFirstOrDefault<byte?>(sqlByName, new
            {
                Names = candidates.ToArray()
            });
        }
        public IReadOnlyList<RaceClassRating> GetHistoricalRaceClassRatings(string? horseName, int? horseId, int maxCount)
        {
            if (maxCount <= 0 || (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName)))
            {
                return Array.Empty<RaceClassRating>();
            }

            const string sqlById = @"SELECT TOP (@Limit)
    r.RaceDate,
    rr.OfficialRating,
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (@Limit)
    r.RaceDate,
    rr.OfficialRating,
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.Query<RaceClassRating>(sqlById, new
                {
                    HorseId = horseId.Value,
                    Limit = maxCount
                }).ToList();

                if (byId.Count > 0)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return Array.Empty<RaceClassRating>();
            }

            return conn.Query<RaceClassRating>(sqlByName, new
            {
                Names = candidates.ToArray(),
                Limit = maxCount
            }).ToList();
        }
        public IReadOnlyList<RunnerResult> GetLastSavedResults(string? raceTitle, DateTime? raceDate)
        {
            if (!raceDate.HasValue)
            {
                return Array.Empty<RunnerResult>();
            }

            var trimmedTitle = string.IsNullOrWhiteSpace(raceTitle)
                ? null
                : raceTitle.Trim();

            const string sql = @"
SELECT rr.RunnerResultId,
       rr.RaceId,
       rr.HorseId,
       rr.TrainerId,
       rr.JockeyId,
       rr.SaddleclothNumber,
       rr.Draw,
       rr.Age,
       rr.WeightLbs,
       rr.WeightText,
       rr.FinishPos,
       rr.OutcomeCode,
       rr.DistanceBeatenText,
       rr.DistanceBeatenLengths,
       rr.SP_Fraction,
       rr.SP_Decimal,
       rr.FavTag,
       rr.OpeningFraction,
       rr.TouchedHighFraction,
       rr.TouchedLowFraction,
       rr.Comment,
       rr.Going,
       rr.Surface,
       r.CourseId,
       r.DistanceYards
FROM Race r
INNER JOIN RunnerResult rr ON rr.RaceId = r.RaceId
WHERE r.RaceDate = @RaceDate
  AND (@Title IS NULL OR LTRIM(RTRIM(r.Title)) = @Title)
ORDER BY rr.RunnerResultId DESC;";

            using var conn = OpenConnection();
            var rows = conn.Query<RunnerResult>(sql, new
            {
                RaceDate = raceDate.Value.Date,
                Title = trimmedTitle
            })?.ToList();

            return rows != null && rows.Count > 0
                ? rows
                : Array.Empty<RunnerResult>();
        }
        public int? GetWinningTimeMilliseconds(int raceId)
        {
            if (raceId <= 0)
            {
                return null;
            }

            const string sql = "SELECT WinningTimeMs FROM Race WHERE RaceId = @RaceId";

            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                return connection.QueryFirstOrDefault<int?>(sql, new { RaceId = raceId });
            }
            finally
            {
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }
        public int? GetRaceIdByRunnerResult(int runnerResultId)
        {
            if (runnerResultId <= 0)
            {
                return null;
            }

            const string sql = "SELECT RaceId FROM RunnerResult WHERE RunnerResultId = @RunnerResultId";

            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                return connection.QueryFirstOrDefault<int?>(sql, new { RunnerResultId = runnerResultId });
            }
            finally
            {
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }

        public decimal? GetDistanceBeatenLengths(int runnerResultId)
        {
            if (runnerResultId <= 0)
            {
                return null;
            }

            const string sql = "SELECT DistanceBeatenLengths FROM RunnerResult WHERE RunnerResultId = @RunnerResultId";

            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                return connection.QueryFirstOrDefault<decimal?>(sql, new { RunnerResultId = runnerResultId });
            }
            finally
            {
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }
        public decimal? GetLastDistanceBeatenLengths(string? horseName, int? horseId, DateTime? beforeDate)
        {
            if (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName))
            {
                return null;
            }

            var cutoffDate = beforeDate?.Date;

            const string sqlById = @"SELECT TOP (1)
    rr.DistanceBeatenLengths
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND rr.DistanceBeatenLengths IS NOT NULL
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (1)
    rr.DistanceBeatenLengths
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
  AND rr.DistanceBeatenLengths IS NOT NULL
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.QueryFirstOrDefault<decimal?>(sqlById, new
                {
                    HorseId = horseId.Value,
                    BeforeDate = cutoffDate
                });

                if (byId.HasValue)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return null;
            }

            return conn.QueryFirstOrDefault<decimal?>(sqlByName, new
            {
                Names = candidates.ToArray(),
                BeforeDate = cutoffDate
            });
        }
        public int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate)
        {
            if (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName))
            {
                return null;
            }

            var cutoffDate = beforeDate?.Date;

            const string sqlById = @"SELECT TOP (1)
       r.DistanceYards
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND r.DistanceYards IS NOT NULL
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (1)
       r.DistanceYards
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
  AND r.DistanceYards IS NOT NULL
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.QueryFirstOrDefault<int?>(sqlById, new
                {
                    HorseId = horseId.Value,
                    BeforeDate = cutoffDate
                });

                if (byId.HasValue)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return null;
            }

            return conn.QueryFirstOrDefault<int?>(sqlByName, new
            {
                Names = candidates.ToArray(),
                BeforeDate = cutoffDate
            });
        }
        public (int Wins, int Starts)? GetRecentHorseWinStats(string? horseName, int? horseId, DateTime? beforeDate, int windowSize)
        {
            if (windowSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSize));
            }

            if (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName))
            {
                return null;
            }

            var cutoffDate = beforeDate?.Date;

            const string sqlById = @"SELECT TOP (@Window)
    rr.FinishPos
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (@Window)
    rr.FinishPos
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.Query<short?>(sqlById, new
                {
                    HorseId = horseId.Value,
                    BeforeDate = cutoffDate,
                    Window = windowSize
                }).ToList();

                var stats = ComputeWinStats(byId);
                if (stats.HasValue)
                {
                    return stats;
                }
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return null;
            }

            var byName = conn.Query<short?>(sqlByName, new
            {
                Names = candidates.ToArray(),
                BeforeDate = cutoffDate,
                Window = windowSize
            }).ToList();

            return ComputeWinStats(byName);
        }

        private static (int Wins, int Starts)? ComputeWinStats(IReadOnlyCollection<short?> finishes)
        {
            if (finishes == null || finishes.Count == 0)
            {
                return null;
            }

            var starts = 0;
            var wins = 0;

            foreach (var finish in finishes)
            {
                starts++;
                if (finish.HasValue && finish.Value == 1)
                {
                    wins++;
                }
            }

            return starts > 0 ? (wins, starts) : null;
        }
        public MLParameter? GetBestMLParameter()
        {
            const string sql = @"SELECT TOP (1)
    Id,
    RunDate,
    Units,
    Dropout,
    Layers,
    LearningRate,
    TrainAccuracy,
    ValidationAccuracy,
    ValidationLoss,
    Epochs,
    BatchSize,
    Folds,
    TrainLoss,
    TrainBrier,
    Fold,
    ValidationBrier
FROM MLParameters
WHERE Units > 0
  AND Layers > 0
  AND Epochs > 0
  AND BatchSize > 0
  AND LearningRate > 0
ORDER BY ISNULL(ValidationAccuracy, 0) DESC, RunDate DESC";

            using var conn = OpenConnection();
            return conn.QueryFirstOrDefault<MLParameter>(sql);
        }
        public UpcomingRace? FindUpcomingRace(DateTime raceDate, string? title, string? venueName)
        {
            const string sql = @"SELECT UpcomingRaceId,
                                           MarketId,
                                           RaceDate,
                                           ScheduledOff,
                                           VenueName,
                                           VenueCountry,
                                           Title,
                                           RaceDetails,
                                           RaceType,
                                           Surface,
                                           Going,
                                           DistanceYards,
                                           DistanceText,
                                           RunnerCount,
                                           BackBookPercentage,
                                           LayBookPercentage
                                    FROM UpcomingRaces
                                    WHERE RaceDate = @RaceDate";

            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                EnsureUpcomingRaceTable(connection, scope);
                var candidates = connection.Query<UpcomingRace>(sql, new { RaceDate = raceDate.Date }).ToList();
                if (candidates.Count == 0)
                {
                    return null;
                }

                var normalizedTitle = NormalizeLookupKey(title);
                var normalizedVenue = NormalizeLookupKey(venueName);

                var (best, bestScore, titleAligned, venueAligned) = SelectBestUpcomingRaceCandidate(
                    candidates,
                    normalizedTitle,
                    normalizedVenue);

                if (best is null)
                {
                    return null;
                }

                if (bestScore > 0)
                {
                    return null;
                }

                var hasTitle = !string.IsNullOrEmpty(normalizedTitle);
                var hasVenue = !string.IsNullOrEmpty(normalizedVenue);
                if ((hasTitle || hasVenue) && !titleAligned && !venueAligned)
                {
                    return null;
                }

                return best;
            }
            finally
            {
                if (ownsConnection)
                {
                    connection.Dispose();
                }
            }
        }


        internal static (UpcomingRace? Candidate, int Score, bool TitleAligned, bool VenueAligned) SelectBestUpcomingRaceCandidate(
            IEnumerable<UpcomingRace> candidates,
            string normalizedTitle,
            string normalizedVenue)
        {
            UpcomingRace? best = null;
            int bestScore = int.MaxValue;
            bool bestTitleAligned = false;
            bool bestVenueAligned = false;

            foreach (var candidate in candidates)
            {
                var (score, titleAligned, venueAligned) = ScoreUpcomingCandidate(candidate, normalizedTitle, normalizedVenue);

                if (score < bestScore ||
                    (score == bestScore && (best == null || candidate.UpcomingRaceId < best.UpcomingRaceId)))
                {
                    best = candidate;
                    bestScore = score;
                    bestTitleAligned = titleAligned;
                    bestVenueAligned = venueAligned;
                }
            }

            return (best, bestScore, bestTitleAligned, bestVenueAligned);
        }

        private static (int Score, bool TitleAligned, bool VenueAligned) ScoreUpcomingCandidate(
            UpcomingRace candidate,
            string normalizedTitle,
            string normalizedVenue)
        {
            int score = 0;
            bool titleAligned = false;
            bool venueAligned = false;

            var candidateTitle = NormalizeLookupKey(candidate.Title);
            if (!string.IsNullOrEmpty(normalizedTitle))
            {
                if (candidateTitle == normalizedTitle)
                {
                    score -= 4;
                    titleAligned = true;
                }
                else if (!string.IsNullOrEmpty(candidateTitle) &&
                         (candidateTitle.Contains(normalizedTitle) || normalizedTitle.Contains(candidateTitle)))
                {
                    score -= 2;
                    titleAligned = true;
                }
                else
                {
                    score += 4;
                }
            }

            var candidateVenue = NormalizeLookupKey(candidate.VenueName);
            if (!string.IsNullOrEmpty(normalizedVenue))
            {
                if (candidateVenue == normalizedVenue)
                {
                    score -= 2;
                    venueAligned = true;
                }
                else if (!string.IsNullOrEmpty(candidateVenue) &&
                         (candidateVenue.Contains(normalizedVenue) || normalizedVenue.Contains(candidateVenue)))
                {
                    score -= 1;
                    venueAligned = true;
                }
                else
                {
                    score += 2;
                }
            }

            return (score, titleAligned, venueAligned);
        }
        public short InsertTrainer(Trainer trainer)
        {
            StripBracketedText(trainer);
            const string sql = @"
IF EXISTS (SELECT 1 FROM Trainer WHERE Name=@Name)
    SELECT TOP 1 TrainerId FROM Trainer WHERE Name=@Name ORDER BY TrainerId;
ELSE
BEGIN
    INSERT INTO Trainer(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<short>(sql, trainer);
        }

        public short InsertJockey(Jockey jockey)
        {
            StripBracketedText(jockey);
            const string sql = @"
IF EXISTS (SELECT 1 FROM Jockey WHERE Name=@Name)
    SELECT TOP 1 JockeyId FROM Jockey WHERE Name=@Name ORDER BY JockeyId;
ELSE
BEGIN
    INSERT INTO Jockey(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<short>(sql, jockey); 
        }

        public int InsertHorse(Horse horse)
        {
            StripBracketedText(horse);
            const string sql = @"
IF EXISTS (SELECT 1 FROM Horse WHERE Name=@Name)
    SELECT TOP 1 HorseId FROM Horse WHERE Name=@Name ORDER BY HorseId;
ELSE
BEGIN
    INSERT INTO Horse(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, horse);
        }
    }
}