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
        private const float MsPerLength = 200f;
        private static readonly object SchemaLock = new();
        private static bool _raceScreenTableEnsured;
        private static bool _runnerFlowTableEnsured;
        private static bool _upcomingRaceTableEnsured;
        private readonly AsyncLocal<DayReportScopeState?> _dayReportScope = new();
        private static bool _raceTableSchemaEnsured;
        private static bool _runnerResultUniquenessEnsured;
        private static bool _horsePerformanceIndexesEnsured;
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
        private void EnsureHorsePerformanceIndexes()
        {
            if (_horsePerformanceIndexesEnsured)
            {
                return;
            }

            lock (SchemaLock)
            {
                if (_horsePerformanceIndexesEnsured)
                {
                    return;
                }

                using var connection = OpenConnection();

                const string sql = @"
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunnerResult_HorseId_RaceId' AND object_id = OBJECT_ID(N'dbo.RunnerResult'))
BEGIN
    CREATE INDEX IX_RunnerResult_HorseId_RaceId ON dbo.RunnerResult(HorseId, RaceId)
        INCLUDE (RunnerResultId, FinishPos, DistanceBeatenLengths);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Race_RaceDate' AND object_id = OBJECT_ID(N'dbo.Race'))
BEGIN
    CREATE INDEX IX_Race_RaceDate ON dbo.Race(RaceDate, RaceId)
        INCLUDE (DistanceYards, WinningTimeMs);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Horse_Name' AND object_id = OBJECT_ID(N'dbo.Horse'))
BEGIN
    CREATE INDEX IX_Horse_Name ON dbo.Horse(Name, HorseId);
END;";

                connection.Execute(sql);
                _horsePerformanceIndexesEnsured = true;
            }
        }

        private static string BuildHorseIdRequestCte(
            IReadOnlyList<HorseMetricRequest> requests,
            DynamicParameters parameters)
        {
            var builder = new StringBuilder();
            builder.AppendLine("WITH Request(RequestIndex, HorseId, BeforeDate) AS (");

            for (var i = 0; i < requests.Count; i++)
            {
                var request = requests[i];
                var prefix = i == 0 ? "    SELECT " : "    UNION ALL SELECT ";
                builder.Append(prefix)
                    .Append(i)
                    .Append(", @HorseId")
                    .Append(i)
                    .Append(", @BeforeDate")
                    .Append(i)
                    .AppendLine();

                parameters.Add($"HorseId{i}", request.HorseId);
                parameters.Add($"BeforeDate{i}", request.BeforeDate);
            }

            builder.AppendLine(")");
            return builder.ToString();
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

            var normalizedMarketIds = flowList
                .Select(flow => flow?.MarketId?.Trim())
                .Where(marketId => !string.IsNullOrEmpty(marketId))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            const string header = @"INSERT INTO RunnerFlow(MarketId, UpcomingRaceId, RaceDate, ScheduledOff, VenueName, VenueCountry, RaceTitle, RaceDetails, RaceType, Surface, Going, DistanceYards, DistanceText, RunnerCount, BackBookPercentage, LayBookPercentage, ClothNumber, Draw, HorseName, JockeyName, BackPrice1, BackPrice2, BackPrice3, LayPrice1, LayPrice2, LayPrice3, AiOdds, Age, WeightLbs, WeightText, TrainerName)
VALUES";


            var sqlBuilder = new StringBuilder(header.Length + flowList.Count * 128);
            sqlBuilder.Append(header);

            var parameters = new DynamicParameters();

            for (var i = 0; i < flowList.Count; i++)
            {
                var flow = flowList[i];
                var suffix = i.ToString(CultureInfo.InvariantCulture);
                var marketId = flow.MarketId?.Trim();

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

                parameters.Add($"MarketId{suffix}", marketId);
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

                if (normalizedMarketIds.Count > 0)
                {
                    const string deleteSql = "DELETE FROM RunnerFlow WHERE MarketId = @MarketId;";
                    foreach (var marketId in normalizedMarketIds)
                    {
                        connection.Execute(deleteSql, new { MarketId = marketId }, transaction);
                    }
                }

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
        public int InsertRace(Race race)
        {
            if (race is null)
            {
                throw new ArgumentNullException(nameof(race));
            }

            StripBracketedText(race);
            race.Going = NormalizeGoing(race.Going);
            race.DistanceText ??= string.Empty;
            EnsureRaceTableSchema();
            const string sql = @"SET NOCOUNT ON;

DECLARE @ExistingId INT;

SELECT TOP (1) @ExistingId = RaceId
FROM Race
WHERE (Rid IS NOT NULL AND Rid = @Rid) OR
      (Rid IS NULL AND CourseId = @CourseId
  AND RaceDate = @RaceDate
  AND (@ScheduledOff IS NULL OR ISNULL(ScheduledOff, '00:00:00') = @ScheduledOff)
  AND ISNULL(LTRIM(RTRIM(Title)), '') = ISNULL(LTRIM(RTRIM(@Title)), ''));

IF @ExistingId IS NULL
BEGIN
    INSERT INTO Race
        (Rid, CourseId, RaceDate, ScheduledOff, ActualOff, Title, RaceType, Class, AgeRestriction, Surface, Going, DistanceYards, DistanceText, RunnerCount, Status, WinningTimeMs, WinningTimeText, PrizeMoney)
    VALUES
        (@Rid, @CourseId, @RaceDate, @ScheduledOff, @ActualOff, @Title, @RaceType, @Class, @AgeRestriction, @Surface, @Going, @DistanceYards, @DistanceText, @RunnerCount, @Status, @WinningTimeMs, @WinningTimeText, @PrizeMoney);

    SELECT CAST(SCOPE_IDENTITY() as int);
END
ELSE
BEGIN
    UPDATE Race
    SET ActualOff = @ActualOff,
        RaceType = @RaceType,
        Class = @Class,
        AgeRestriction = @AgeRestriction,
        Surface = @Surface,
        Going = @Going,
        DistanceYards = @DistanceYards,
        DistanceText = @DistanceText,
        RunnerCount = @RunnerCount,
        Status = @Status,
        WinningTimeMs = @WinningTimeMs,
        WinningTimeText = @WinningTimeText,
        PrizeMoney = @PrizeMoney,
        Rid = COALESCE(@Rid, Rid)
    WHERE RaceId = @ExistingId;
    SELECT @ExistingId;
END;";

            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, race);
        }

        public Dictionary<int, int> GetRaceIdsByRids(IEnumerable<int> rids)
        {
            if (rids == null) return new Dictionary<int, int>();
            var distinctRids = rids.Distinct().ToList();
            if (distinctRids.Count == 0) return new Dictionary<int, int>();

            const string sql = "SELECT Rid, RaceId FROM Race WHERE Rid IN @Rids";
            var result = new Dictionary<int, int>();

            using var conn = OpenConnection();
            foreach (var chunk in Chunk(distinctRids, 1000))
            {
                foreach (var row in conn.Query<(int Rid, int RaceId)>(sql, new { Rids = chunk }))
                {
                    result[row.Rid] = row.RaceId;
                }
            }
            return result;
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
        public void BulkInsertRunnerResults(IEnumerable<RunnerResult> results)
        {
            var list = results?.ToList();
            if (list == null || list.Count == 0)
            {
                return;
            }

            var deduped = new Dictionary<(int RaceId, int HorseId), RunnerResult>();
            foreach (var result in list)
            {
                if (result == null)
                {
                    continue;
                }

                StripBracketedText(result);
                deduped[(result.RaceId, result.HorseId)] = result;
            }

            if (deduped.Count == 0)
            {
                return;
            }

            Console.WriteLine($"[Repository] Merging {deduped.Count} runner results into database...");
            using var conn = (SqlConnection)OpenConnection();

            const string createTempTableSql = @"
CREATE TABLE #RunnerResultStage (
    RaceId INT,
    Rid INT NULL,
    HorseId INT,
    TrainerId SMALLINT NULL,
    JockeyId SMALLINT NULL,
    SaddleclothNumber TINYINT NULL,
    Draw TINYINT NULL,
    Age TINYINT NULL,
    WeightLbs TINYINT NULL,
    WeightText NVARCHAR(32) NULL,
    FinishPos SMALLINT NULL,
    OutcomeCode NVARCHAR(32) NULL,
    DistanceBeatenText NVARCHAR(32) NULL,
    DistanceBeatenLengths DECIMAL(4, 2) NULL,
    SP_Fraction NVARCHAR(32) NULL,
    SP_Decimal DECIMAL(8, 3) NULL,
    FavTag NVARCHAR(32) NULL,
    OpeningFraction NVARCHAR(32) NULL,
    TouchedHighFraction NVARCHAR(32) NULL,
    TouchedLowFraction NVARCHAR(32) NULL,
    Comment NVARCHAR(MAX) NULL,
    IsPlace BIT NOT NULL
)";
            conn.Execute(createTempTableSql);

            using (var bulkCopy = new SqlBulkCopy(conn))
            {
                bulkCopy.DestinationTableName = "#RunnerResultStage";
                bulkCopy.BulkCopyTimeout = 300;

                var table = new DataTable();
                table.Columns.Add("RaceId", typeof(int));
                table.Columns.Add("Rid", typeof(int));
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
                table.Columns.Add("IsPlace", typeof(bool));

                foreach (var item in deduped.Values)
                {
                    // Apply safety clamping logic to ensure SqlBulkCopy doesn't fail
                    // DistanceBeatenLengths DECIMAL(4, 2) -> Max 99.99
                    object distVal = DBNull.Value;
                    if (item.DistanceBeatenLengths.HasValue)
                    {
                        var d = item.DistanceBeatenLengths.Value;
                        if (d > 99.99m) d = 99.99m;
                        else if (d < -99.99m) d = -99.99m;
                        distVal = d;
                    }

                    // SP_Decimal DECIMAL(8, 3) -> Max 99999.999
                    object spVal = DBNull.Value;
                    if (item.SP_Decimal.HasValue)
                    {
                        var s = item.SP_Decimal.Value;
                        if (s > 99999.999m) s = 99999.999m;
                        else if (s < -99999.999m) s = -99999.999m;
                        spVal = s;
                    }

                    // WeightLbs TINYINT -> Max 255
                    // Note: item.WeightLbs is already byte?, so logically it fits, 
                    // but if it was populated from raw unchecked code it might be good to ensure.
                    // (Since it is byte in C#, it is already < 256, so no extra clamp needed here for the type check)

                    table.Rows.Add(
                        item.RaceId,
                        item.Rid ?? (object)DBNull.Value,
                        item.HorseId,
                        item.TrainerId ?? (object)DBNull.Value,
                        item.JockeyId ?? (object)DBNull.Value,
                        item.SaddleclothNumber ?? (object)DBNull.Value,
                        item.Draw ?? (object)DBNull.Value,
                        item.Age ?? (object)DBNull.Value,
                        item.WeightLbs ?? (object)DBNull.Value,
                        item.WeightText,
                        item.FinishPos ?? (object)DBNull.Value,
                        item.OutcomeCode,
                        item.DistanceBeatenText,
                        distVal,
                        item.SP_Fraction,
                        spVal,
                        item.FavTag,
                        item.OpeningFraction,
                        item.TouchedHighFraction,
                        item.TouchedLowFraction,
                        item.Comment,
                        item.IsPlace
                    );
                }

                bulkCopy.WriteToServer(table);
            }

            const string mergeSql = @"
MERGE INTO dbo.RunnerResult AS target
USING #RunnerResultStage AS source
ON target.RaceId = source.RaceId AND target.HorseId = source.HorseId
WHEN MATCHED THEN
    UPDATE SET
        Rid = source.Rid,
        TrainerId = source.TrainerId,
        JockeyId = source.JockeyId,
        SaddleclothNumber = source.SaddleclothNumber,
        Draw = source.Draw,
        Age = source.Age,
        WeightLbs = source.WeightLbs,
        WeightText = source.WeightText,
        FinishPos = source.FinishPos,
        OutcomeCode = source.OutcomeCode,
        DistanceBeatenText = source.DistanceBeatenText,
        DistanceBeatenLengths = source.DistanceBeatenLengths,
        SP_Fraction = source.SP_Fraction,
        SP_Decimal = source.SP_Decimal,
        FavTag = source.FavTag,
        OpeningFraction = source.OpeningFraction,
        TouchedHighFraction = source.TouchedHighFraction,
        TouchedLowFraction = source.TouchedLowFraction,
        Comment = source.Comment,
        IsPlace = source.IsPlace
WHEN NOT MATCHED BY TARGET THEN
    INSERT (
        RaceId,
        Rid,
        HorseId,
        TrainerId,
        JockeyId,
        SaddleclothNumber,
        Draw,
        Age,
        WeightLbs,
        WeightText,
        FinishPos,
        OutcomeCode,
        DistanceBeatenText,
        DistanceBeatenLengths,
        SP_Fraction,
        SP_Decimal,
        FavTag,
        OpeningFraction,
        TouchedHighFraction,
        TouchedLowFraction,
        Comment,
        IsPlace
    )
    VALUES (
        source.RaceId,
        source.Rid,
        source.HorseId,
        source.TrainerId,
        source.JockeyId,
        source.SaddleclothNumber,
        source.Draw,
        source.Age,
        source.WeightLbs,
        source.WeightText,
        source.FinishPos,
        source.OutcomeCode,
        source.DistanceBeatenText,
        source.DistanceBeatenLengths,
        source.SP_Fraction,
        source.SP_Decimal,
        source.FavTag,
        source.OpeningFraction,
        source.TouchedHighFraction,
        source.TouchedLowFraction,
        source.Comment,
        source.IsPlace
    );
DROP TABLE #RunnerResultStage;";

            EnsureRunnerResultUniqueness(conn);
            try
            {
                conn.Execute(mergeSql, commandTimeout: 300);
                Console.WriteLine($"[Repository] Successfully merged {deduped.Count} results.");
            }
            
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Repository] Error merging runner results: {ex.Message}");
                throw;
            }
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

            var candidateList = candidateMap.Keys.ToList();
            if (candidateList.Count > 0)
            {
                foreach (var chunk in Chunk(candidateList, 1000))
                {
                    foreach (var (Name, HorseId) in conn.Query<(string Name, int HorseId)>(horseSql, new { Names = chunk }))
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

                    var normalizedKeys = normalizedMap.Keys.ToList();
                    foreach (var chunk in Chunk(normalizedKeys, 1000))
                    {
                        foreach (var (Normalized, HorseId) in conn.Query<(string Normalized, int HorseId)>(normalizedHorseSql, new { Names = chunk }))
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
            foreach (var chunk in Chunk(distinctHorseIds, 1000))
            {
                foreach (var row in conn.Query(countSql, new { HorseIds = chunk }))
                {
                    var horseId = (int)row.HorseId;
                    var count = (int)row.RaceCount;
                    var wins = (int)row.WinCount;
                    countsByHorseId[horseId] = count;
                    winsByHorseId[horseId] = wins;
                }
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

            const string upsertSql = @"
IF @MarketId IS NULL
BEGIN
    INSERT INTO RaceScreen(MarketId, RaceDate, OffTime, Title, VenueName, VenueCountry, EventDateText, RaceDetails, RaceType, Going, BackBookPercentage, LayBookPercentage, RaceUrl)
    VALUES(@MarketId, @RaceDate, @OffTime, @Title, @VenueName, @VenueCountry, @EventDateText, @RaceDetails, @RaceType, @Going, @BackBookPercentage, @LayBookPercentage, @RaceUrl);
END
ELSE
BEGIN
    DECLARE @ExistingId BIGINT;
    SELECT TOP 1 @ExistingId = RaceScreenId
    FROM RaceScreen
    WHERE MarketId = @MarketId;

    IF @ExistingId IS NULL
    BEGIN
        INSERT INTO RaceScreen(MarketId, RaceDate, OffTime, Title, VenueName, VenueCountry, EventDateText, RaceDetails, RaceType, Going, BackBookPercentage, LayBookPercentage, RaceUrl)
        VALUES(@MarketId, @RaceDate, @OffTime, @Title, @VenueName, @VenueCountry, @EventDateText, @RaceDetails, @RaceType, @Going, @BackBookPercentage, @LayBookPercentage, @RaceUrl);
    END
    ELSE
    BEGIN
        UPDATE RaceScreen
        SET RaceDate = @RaceDate,
            OffTime = @OffTime,
            Title = @Title,
            VenueName = @VenueName,
            VenueCountry = @VenueCountry,
            EventDateText = @EventDateText,
            RaceDetails = @RaceDetails,
            RaceType = @RaceType,
            Going = @Going,
            BackBookPercentage = @BackBookPercentage,
            LayBookPercentage = @LayBookPercentage,
            RaceUrl = @RaceUrl
        WHERE RaceScreenId = @ExistingId;
    END
END";

            var (connection, scope, ownsConnection) = GetScopedConnection();
            try
            {
                EnsureRaceScreenTable(connection, scope);
                connection.Execute(upsertSql, screen);
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
        private void EnsureRaceTableSchema()
        {
            if (_raceTableSchemaEnsured) return;

            lock (SchemaLock)
            {
                if (_raceTableSchemaEnsured) return;

                using var conn = OpenConnection();
                const string sql = @"
IF COL_LENGTH('dbo.Race', 'PrizeMoney') IS NULL
BEGIN
   ALTER TABLE dbo.Race ADD PrizeMoney DECIMAL(18, 2) NULL;
END;

IF COL_LENGTH('dbo.Race', 'Rid') IS NULL
BEGIN
   ALTER TABLE dbo.Race ADD Rid INT NULL;
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Race_Rid' AND object_id = OBJECT_ID('dbo.Race'))
BEGIN
   EXEC('CREATE INDEX IX_Race_Rid ON dbo.Race(Rid) WHERE Rid IS NOT NULL');
END;

IF COL_LENGTH('dbo.RunnerResult', 'Rid') IS NULL
BEGIN
   ALTER TABLE dbo.RunnerResult ADD Rid INT NULL;
END;

IF COL_LENGTH('dbo.RunnerResult', 'IsPlace') IS NULL
BEGIN
   ALTER TABLE dbo.RunnerResult ADD IsPlace BIT NOT NULL DEFAULT 0;
END;

ALTER TABLE dbo.Race ALTER COLUMN ScheduledOff TIME(7) NULL;
";
                conn.Execute(sql);
                _raceTableSchemaEnsured = true;
            }
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
                const string dedupeSql = @"
WITH Ranked AS (
    SELECT RaceScreenId,
           ROW_NUMBER() OVER (PARTITION BY MarketId ORDER BY RaceScreenId DESC) AS RowNum
    FROM dbo.RaceScreen
    WHERE MarketId IS NOT NULL
)
DELETE FROM dbo.RaceScreen
WHERE RaceScreenId IN (
    SELECT RaceScreenId
    FROM Ranked
    WHERE RowNum > 1
);";

                conn.Execute(dedupeSql);

                const string addMarketIdIndexSql = @"
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RaceScreen_MarketId' AND object_id = OBJECT_ID(N'dbo.RaceScreen'))
BEGIN
    CREATE UNIQUE INDEX IX_RaceScreen_MarketId ON dbo.RaceScreen(MarketId) WHERE MarketId IS NOT NULL;
END";

                conn.Execute(addMarketIdIndexSql);
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
        private static void EnsureRunnerResultUniqueness(IDbConnection conn)
        {
            if (_runnerResultUniquenessEnsured) return;

            lock (SchemaLock)
            {
                if (_runnerResultUniquenessEnsured) return;

                const string dedupeSql = @"
IF OBJECT_ID(N'dbo.RunnerResult', 'U') IS NOT NULL
BEGIN
    WITH Ranked AS (
        SELECT RunnerResultId,
               ROW_NUMBER() OVER (PARTITION BY RaceId, HorseId ORDER BY RunnerResultId DESC) AS RowNum
        FROM dbo.RunnerResult
    )
    DELETE FROM dbo.RunnerResult
    WHERE RunnerResultId IN (
        SELECT RunnerResultId
        FROM Ranked
        WHERE RowNum > 1
    );
END";

                conn.Execute(dedupeSql);

                const string indexSql = @"
IF OBJECT_ID(N'dbo.RunnerResult', 'U') IS NOT NULL
BEGIN
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunnerResult_RaceId_HorseId' AND object_id = OBJECT_ID(N'dbo.RunnerResult'))
    BEGIN
        CREATE UNIQUE INDEX IX_RunnerResult_RaceId_HorseId ON dbo.RunnerResult(RaceId, HorseId);
    END
END";

                conn.Execute(indexSql);
                _runnerResultUniquenessEnsured = true;
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
IF COL_LENGTH('dbo.MLParameters', 'ModelType') IS NULL
BEGIN
    ALTER TABLE dbo.MLParameters ADD ModelType INT NOT NULL DEFAULT 0;
END;


IF COL_LENGTH('dbo.MLParameters', 'LgbmLearningRate') IS NULL
BEGIN
    ALTER TABLE dbo.MLParameters ADD LgbmLearningRate FLOAT NULL;
END;

IF COL_LENGTH('dbo.MLParameters', 'LgbmEpochs') IS NULL
BEGIN
    ALTER TABLE dbo.MLParameters ADD LgbmEpochs INT NULL;
END;

INSERT INTO MLParameters(RunDate, Units, Dropout, Layers, LearningRate, TrainAccuracy, ValidationAccuracy, ValidationLoss, Epochs, BatchSize, Folds, TrainLoss, TrainBrier, Fold, ValidationBrier, TrainFocalLoss, ValidationFocalLoss, ModelType, LgbmLeaves, LgbmMinDataInLeaf, LgbmMaxDepth, LgbmLearningRate, LgbmEpochs)
VALUES(@RunDate, @Units, @Dropout, @Layers, @LearningRate, @TrainAccuracy, @ValidationAccuracy, @ValidationLoss, @Epochs, @BatchSize, @Folds, @TrainLoss, @TrainBrier, @Fold, @ValidationBrier, @TrainFocalLoss, @ValidationFocalLoss, @ModelType, @LgbmLeaves, @LgbmMinDataInLeaf, @LgbmMaxDepth, @LgbmLearningRate, @LgbmEpochs);
SELECT CAST(SCOPE_IDENTITY() as int);";
            using var conn = OpenConnection();

            // Map nulls to defaults for existing NOT NULL columns to preserve schema compatibility
            var parameters = new DynamicParameters(param);
            if (!param.Units.HasValue) parameters.Add("Units", 0);
            if (!param.Dropout.HasValue) parameters.Add("Dropout", 0);
            if (!param.Layers.HasValue) parameters.Add("Layers", 0);
            if (!param.BatchSize.HasValue) parameters.Add("BatchSize", 0);

            return conn.QuerySingle<int>(sql, parameters);
        }
        public List<int> GetRaceIdsBetweenDates(DateTime startInclusive, DateTime endInclusive, string country = null)
        {
            if (endInclusive < startInclusive)
            {
                throw new ArgumentException("The end date must be on or after the start date.", nameof(endInclusive));
            }

            using var conn = OpenConnection();
            var sql = new StringBuilder(@"SELECT r.RaceId FROM Race r");
            if (!string.IsNullOrEmpty(country))
            {
                sql.Append(" INNER JOIN Course c ON r.CourseId = c.CourseId WHERE c.Country = @Country AND");
            }
            else
            {
                sql.Append(" WHERE");
            }
            sql.Append(" r.RaceDate >= @Start AND r.RaceDate <= @End ORDER BY r.RaceDate, r.RaceId");
            return conn.Query<int>(sql.ToString(), new { Start = startInclusive, End = endInclusive, Country = country }).ToList();
        }

        public List<int> GetRaceIdsOutsideRange(DateTime startInclusive, DateTime endInclusive, string country = null)
        {
            if (endInclusive < startInclusive)
            {
                throw new ArgumentException("The end date must be on or after the start date.", nameof(endInclusive));
            }

            using var conn = OpenConnection();
            var sql = new StringBuilder(@"SELECT r.RaceId FROM Race r");
            if (!string.IsNullOrEmpty(country))
            {
                sql.Append(" INNER JOIN Course c ON r.CourseId = c.CourseId WHERE c.Country = @Country AND");
            }
            else
            {
                sql.Append(" WHERE");
            }
            sql.Append(" (r.RaceDate < @Start OR r.RaceDate > @End) ORDER BY r.RaceDate, r.RaceId");
            return conn.Query<int>(sql.ToString(), new { Start = startInclusive, End = endInclusive, Country = country }).ToList();
        }

        public IEnumerable<string> GetCountries()
        {
            using var conn = OpenConnection();
            const string sql = @"SELECT DISTINCT Country FROM Course WHERE Country IS NOT NULL ORDER BY Country";
            return conn.Query<string>(sql);
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
        public IDictionary<int, RaceFeatureBackfill> GetRaceFeatureBackfills(IEnumerable<int> raceIds)
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
                return new Dictionary<int, RaceFeatureBackfill>();
            }

            var result = new Dictionary<int, RaceFeatureBackfill>();

            using var conn = OpenConnection();

            var raceColumns = LoadColumnNames(conn, "Race");
            bool hasRaceType = raceColumns.Contains("RaceType");
            bool hasSurface = raceColumns.Contains("Surface");
            bool hasGoing = raceColumns.Contains("Going");
            bool hasDistanceYards = raceColumns.Contains("DistanceYards");
            bool hasDistanceText = raceColumns.Contains("DistanceText");
            bool hasRunnerCount = raceColumns.Contains("RunnerCount");

            var selectParts = new List<string>
            {
                "r.RaceId"
            };

            selectParts.Add(hasRaceType
                ? "r.RaceType AS RaceType"
                : "CAST(NULL AS nvarchar(128)) AS RaceType");
            selectParts.Add(hasSurface
                ? "r.Surface AS Surface"
                : "CAST(NULL AS nvarchar(64)) AS Surface");
            selectParts.Add(hasGoing
                ? "r.Going AS Going"
                : "CAST(NULL AS nvarchar(30)) AS Going");
            selectParts.Add(hasDistanceYards
                ? "r.DistanceYards AS DistanceYards"
                : "CAST(NULL AS int) AS DistanceYards");
            selectParts.Add(hasDistanceText
                ? "r.DistanceText AS DistanceText"
                : "CAST(NULL AS nvarchar(64)) AS DistanceText");
            selectParts.Add(hasRunnerCount
                ? "r.RunnerCount AS RunnerCount"
                : "CAST(NULL AS int) AS RunnerCount");

            string sql = $"SELECT {string.Join(", ", selectParts)} FROM Race r WHERE r.RaceId IN @Ids";

            const int chunkSize = 2000;
            for (int offset = 0; offset < distinctIds.Count; offset += chunkSize)
            {
                var chunk = distinctIds.Skip(offset).Take(chunkSize).ToArray();
                if (chunk.Length == 0)
                {
                    continue;
                }

                foreach (var row in conn.Query<RaceFeatureBackfill>(sql, new { Ids = chunk }))
                {
                    if (row == null)
                    {
                        continue;
                    }

                    result[row.RaceId] = row;
                }
            }

            return result;
        }

        private static HashSet<string> LoadColumnNames(IDbConnection connection, string tableName)
        {
            if (connection is null)
            {
                throw new ArgumentNullException(nameof(connection));
            }

            if (string.IsNullOrWhiteSpace(tableName))
            {
                throw new ArgumentException("Table name must be provided.", nameof(tableName));
            }

            const string sql = @"SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = @Table";

            var names = connection.Query<string>(sql, new { Table = tableName });
            return new HashSet<string>(names ?? Array.Empty<string>(), StringComparer.OrdinalIgnoreCase);
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
        public IReadOnlyList<HorseHistoricalRaceSummary> GetRecentHorseResults(string? horseName, int? horseId, int maxCount)
        {
            if (maxCount <= 0 || (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName)))
            {
                return Array.Empty<HorseHistoricalRaceSummary>();
            }

            const string sqlById = @"SELECT TOP (@Limit)
    r.RaceDate,
    r.Title AS RaceTitle,
    rr.FinishPos AS FinishPosition,
    rr.OutcomeCode,
    r.RunnerCount
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (@Limit)
    r.RaceDate,
    r.Title AS RaceTitle,
    rr.FinishPos AS FinishPosition,
    rr.OutcomeCode,
    r.RunnerCount
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.Query<HorseHistoricalRaceSummary>(sqlById, new
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
                return Array.Empty<HorseHistoricalRaceSummary>();
            }

            return conn.Query<HorseHistoricalRaceSummary>(sqlByName, new
            {
                Names = candidates.ToArray(),
                Limit = maxCount
            }).ToList();
        }
        public IReadOnlyList<ParticipantHistoricalRaceSummary> GetRecentTrainerResults(
            string? trainerName,
            int? trainerId,
            int maxCount)
        {
            if (maxCount <= 0 || (!trainerId.HasValue && string.IsNullOrWhiteSpace(trainerName)))
            {
                return Array.Empty<ParticipantHistoricalRaceSummary>();
            }

            const string sqlById = @"SELECT TOP (@Limit)
    r.RaceDate,
    r.Title AS RaceTitle,
    rr.FinishPos AS FinishPosition,
    rr.OutcomeCode,
    r.RunnerCount
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.TrainerId = @TrainerId
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (@Limit)
    r.RaceDate,
    r.Title AS RaceTitle,
    rr.FinishPos AS FinishPosition,
    rr.OutcomeCode,
    r.RunnerCount
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Trainer t ON t.TrainerId = rr.TrainerId
WHERE t.Name IN @Names
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (trainerId.HasValue && trainerId.Value > 0)
            {
                var byId = conn.Query<ParticipantHistoricalRaceSummary>(sqlById, new
                {
                    TrainerId = trainerId.Value,
                    Limit = maxCount
                }).ToList();

                if (byId.Count > 0)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(trainerName);
            if (candidates.Count == 0)
            {
                return Array.Empty<ParticipantHistoricalRaceSummary>();
            }

            return conn.Query<ParticipantHistoricalRaceSummary>(sqlByName, new
            {
                Names = candidates.ToArray(),
                Limit = maxCount
            }).ToList();
        }

        public IReadOnlyList<ParticipantHistoricalRaceSummary> GetRecentJockeyResults(
            string? jockeyName,
            int? jockeyId,
            int maxCount)
        {
            if (maxCount <= 0 || (!jockeyId.HasValue && string.IsNullOrWhiteSpace(jockeyName)))
            {
                return Array.Empty<ParticipantHistoricalRaceSummary>();
            }

            const string sqlById = @"SELECT TOP (@Limit)
    r.RaceDate,
    r.Title AS RaceTitle,
    rr.FinishPos AS FinishPosition,
    rr.OutcomeCode,
    r.RunnerCount
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.JockeyId = @JockeyId
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (@Limit)
    r.RaceDate,
    r.Title AS RaceTitle,
    rr.FinishPos AS FinishPosition,
    rr.OutcomeCode,
    r.RunnerCount
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Jockey j ON j.JockeyId = rr.JockeyId
WHERE j.Name IN @Names
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (jockeyId.HasValue && jockeyId.Value > 0)
            {
                var byId = conn.Query<ParticipantHistoricalRaceSummary>(sqlById, new
                {
                    JockeyId = jockeyId.Value,
                    Limit = maxCount
                }).ToList();

                if (byId.Count > 0)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(jockeyName);
            if (candidates.Count == 0)
            {
                return Array.Empty<ParticipantHistoricalRaceSummary>();
            }

            return conn.Query<ParticipantHistoricalRaceSummary>(sqlByName, new
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
        public byte? GetMostRecentRaceClassForHorseJockey(
            string? horseName,
            int? horseId,
            string? jockeyName,
            int? jockeyId)
        {
            bool hasHorseIdentifier = horseId.HasValue && horseId.Value > 0;
            bool hasHorseName = !string.IsNullOrWhiteSpace(horseName);
            bool hasJockeyIdentifier = jockeyId.HasValue && jockeyId.Value > 0;
            bool hasJockeyName = !string.IsNullOrWhiteSpace(jockeyName);

            if ((!hasHorseIdentifier && !hasHorseName) || (!hasJockeyIdentifier && !hasJockeyName))
            {
                return null;
            }

            const string sqlByIds = @"SELECT TOP (1)
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND rr.JockeyId = @JockeyId
  AND r.Class IS NOT NULL
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByHorseIdJockeyNames = @"SELECT TOP (1)
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Jockey j ON j.JockeyId = rr.JockeyId
WHERE rr.HorseId = @HorseId
  AND j.Name IN @Names
  AND r.Class IS NOT NULL
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByHorseNamesJockeyId = @"SELECT TOP (1)
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
  AND rr.JockeyId = @JockeyId
  AND r.Class IS NOT NULL
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByNames = @"SELECT TOP (1)
    r.Class
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
INNER JOIN Jockey j ON j.JockeyId = rr.JockeyId
WHERE h.Name IN @HorseNames
  AND j.Name IN @JockeyNames
  AND r.Class IS NOT NULL
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (hasHorseIdentifier && hasJockeyIdentifier)
            {
                var byIds = conn.QueryFirstOrDefault<byte?>(sqlByIds, new
                {
                    HorseId = horseId!.Value,
                    JockeyId = jockeyId!.Value
                });

                if (byIds.HasValue)
                {
                    return byIds;
                }
            }

            var horseCandidates = hasHorseName
                ? BuildHistoricalNameCandidates(horseName)
                : Array.Empty<string>();

            var jockeyCandidates = hasJockeyName
                ? BuildHistoricalNameCandidates(jockeyName)
                : Array.Empty<string>();

            if (hasHorseIdentifier && jockeyCandidates.Count > 0)
            {
                var byHorseIdAndNames = conn.QueryFirstOrDefault<byte?>(sqlByHorseIdJockeyNames, new
                {
                    HorseId = horseId!.Value,
                    Names = jockeyCandidates.ToArray()
                });

                if (byHorseIdAndNames.HasValue)
                {
                    return byHorseIdAndNames;
                }
            }

            if (hasJockeyIdentifier && horseCandidates.Count > 0)
            {
                var byNamesAndJockeyId = conn.QueryFirstOrDefault<byte?>(sqlByHorseNamesJockeyId, new
                {
                    Names = horseCandidates.ToArray(),
                    JockeyId = jockeyId!.Value
                });

                if (byNamesAndJockeyId.HasValue)
                {
                    return byNamesAndJockeyId;
                }
            }

            if (horseCandidates.Count == 0 || jockeyCandidates.Count == 0)
            {
                return null;
            }

            return conn.QueryFirstOrDefault<byte?>(sqlByNames, new
            {
                HorseNames = horseCandidates.ToArray(),
                JockeyNames = jockeyCandidates.ToArray()
            });
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
        public IReadOnlyDictionary<HorseMetricRequest, (int Wins, int Starts)?> GetRecentHorseWinStatsBatch(
            IEnumerable<HorseMetricRequest> requests,
            int windowSize)
        {
            if (requests is null)
            {
                throw new ArgumentNullException(nameof(requests));
            }

            if (windowSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSize));
            }

            var deduped = requests
                .Where(r => r.HorseId.HasValue || !string.IsNullOrWhiteSpace(r.NormalizedHorseName))
                .Distinct()
                .ToList();

            var results = new Dictionary<HorseMetricRequest, (int Wins, int Starts)?>(deduped.Count);
            if (deduped.Count == 0)
            {
                return results;
            }

            var horseIdRequests = deduped
                .Where(r => r.HorseId.HasValue && r.HorseId.Value > 0)
                .ToList();

            if (horseIdRequests.Count > 0)
            {
                EnsureHorsePerformanceIndexes();
                using var connection = OpenConnection();
                var parameters = new DynamicParameters();
                parameters.Add("Window", windowSize);
                var cte = BuildHorseIdRequestCte(horseIdRequests, parameters);

                var sql = $@"{cte}
SELECT req.RequestIndex,
       aggregated.Wins,
       aggregated.Starts
FROM Request req
OUTER APPLY (
    SELECT SUM(CASE WHEN innerResult.FinishPos = 1 THEN 1 ELSE 0 END) AS Wins,
           COUNT(innerResult.FinishPos) AS Starts
    FROM (
        SELECT TOP (@Window) rr.FinishPos
        FROM RunnerResult rr
        INNER JOIN Race r ON r.RaceId = rr.RaceId
        WHERE rr.HorseId = req.HorseId
          AND (req.BeforeDate IS NULL OR r.RaceDate < req.BeforeDate)
        ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC
    ) innerResult
) aggregated;";

                var rows = connection.Query<RequestWinRateRow>(sql, parameters).ToList();
                foreach (var row in rows)
                {
                    if (row.RequestIndex < 0 || row.RequestIndex >= horseIdRequests.Count)
                    {
                        continue;
                    }

                    var request = horseIdRequests[row.RequestIndex];
                    if (row.Starts.HasValue && row.Starts.Value > 0)
                    {
                        results[request] = (row.Wins ?? 0, row.Starts.Value);
                    }
                    else
                    {
                        results[request] = null;
                    }
                }

                for (var i = 0; i < horseIdRequests.Count; i++)
                {
                    var request = horseIdRequests[i];
                    if (!results.ContainsKey(request))
                    {
                        results[request] = null;
                    }
                }
            }

            foreach (var request in deduped)
            {
                if (results.ContainsKey(request))
                {
                    continue;
                }

                var stats = GetRecentHorseWinStats(request.RawHorseName, request.HorseId, request.BeforeDate, windowSize);
                results[request] = stats;
            }

            return results;
        }
        public IReadOnlyList<HorseSpeedEntry> GetRecentHorseSpeedEntries(
            string? horseName,
            int? horseId,
            DateTime? beforeDate,
            int windowSize)
        {
            if (windowSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSize));
            }

            if (!horseId.HasValue && string.IsNullOrWhiteSpace(horseName))
            {
                return Array.Empty<HorseSpeedEntry>();
            }

            var cutoffDate = beforeDate?.Date;

            const string sqlById = @"SELECT TOP (@Window)
    r.RaceDate,
    r.DistanceYards,
    r.WinningTimeMs AS WinningTimeMilliseconds,
    rr.DistanceBeatenLengths
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            const string sqlByName = @"SELECT TOP (@Window)
    r.RaceDate,
    r.DistanceYards,
    r.WinningTimeMs AS WinningTimeMilliseconds,
    rr.DistanceBeatenLengths
FROM RunnerResult rr
INNER JOIN Race r ON r.RaceId = rr.RaceId
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
  AND (@BeforeDate IS NULL OR r.RaceDate < @BeforeDate)
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC;";

            using var conn = OpenConnection();

            if (horseId.HasValue && horseId.Value > 0)
            {
                var byId = conn.Query<HorseSpeedEntry>(sqlById, new
                {
                    HorseId = horseId.Value,
                    BeforeDate = cutoffDate,
                    Window = windowSize
                }).ToList();

                if (byId.Count > 0)
                {
                    return byId;
                }
            }

            var candidates = BuildHistoricalNameCandidates(horseName);
            if (candidates.Count == 0)
            {
                return Array.Empty<HorseSpeedEntry>();
            }

            var byName = conn.Query<HorseSpeedEntry>(sqlByName, new
            {
                Names = candidates.ToArray(),
                BeforeDate = cutoffDate,
                Window = windowSize
            }).ToList();

            return byName.Count > 0
                ? byName
                : Array.Empty<HorseSpeedEntry>();
        }
        public IReadOnlyDictionary<HorseMetricRequest, float?> GetRecentHorseAverageSpeedsBatch(
            IEnumerable<HorseMetricRequest> requests,
            int windowSize)
        {
            if (requests is null)
            {
                throw new ArgumentNullException(nameof(requests));
            }

            if (windowSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSize));
            }

            var deduped = requests
                .Where(r => r.HorseId.HasValue || !string.IsNullOrWhiteSpace(r.NormalizedHorseName))
                .Distinct()
                .ToList();

            var results = new Dictionary<HorseMetricRequest, float?>(deduped.Count);
            if (deduped.Count == 0)
            {
                return results;
            }

            var horseIdRequests = deduped
                .Where(r => r.HorseId.HasValue && r.HorseId.Value > 0)
                .ToList();

            if (horseIdRequests.Count > 0)
            {
                EnsureHorsePerformanceIndexes();
                using var connection = OpenConnection();
                var parameters = new DynamicParameters();
                parameters.Add("Window", windowSize);
                parameters.Add("MsPerLength", MsPerLength);
                var cte = BuildHorseIdRequestCte(horseIdRequests, parameters);

                var sql = $@"{cte}
SELECT req.RequestIndex,
       CASE WHEN aggregated.SpeedCount > 0 THEN aggregated.SpeedSum / aggregated.SpeedCount ELSE NULL END AS AverageSpeed
FROM Request req
OUTER APPLY (
    SELECT SUM(speed.SpeedValue) AS SpeedSum,
           COUNT(speed.SpeedValue) AS SpeedCount
    FROM (
        SELECT TOP (@Window)
               CASE
                   WHEN r.DistanceYards IS NULL OR r.DistanceYards <= 0 THEN NULL
                   WHEN r.WinningTimeMs IS NULL OR r.WinningTimeMs <= 0 THEN NULL
                   WHEN rr.DistanceBeatenLengths IS NOT NULL
                       THEN CAST(r.DistanceYards AS float) /
                            NULLIF(CAST(r.WinningTimeMs AS float) + CAST(rr.DistanceBeatenLengths AS float) * @MsPerLength, 0)
                   ELSE CAST(r.DistanceYards AS float) / NULLIF(CAST(r.WinningTimeMs AS float), 0)
               END AS SpeedValue
        FROM RunnerResult rr
        INNER JOIN Race r ON r.RaceId = rr.RaceId
        WHERE rr.HorseId = req.HorseId
          AND (req.BeforeDate IS NULL OR r.RaceDate < req.BeforeDate)
        ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC
    ) speed
    WHERE speed.SpeedValue IS NOT NULL
) aggregated;";

                var rows = connection.Query<RequestAverageSpeedRow>(sql, parameters).ToList();
                foreach (var row in rows)
                {
                    if (row.RequestIndex < 0 || row.RequestIndex >= horseIdRequests.Count)
                    {
                        continue;
                    }

                    var request = horseIdRequests[row.RequestIndex];
                    if (row.AverageSpeed.HasValue)
                    {
                        results[request] = (float)row.AverageSpeed.Value;
                    }
                    else
                    {
                        results[request] = null;
                    }
                }

                for (var i = 0; i < horseIdRequests.Count; i++)
                {
                    var request = horseIdRequests[i];
                    if (!results.ContainsKey(request))
                    {
                        results[request] = null;
                    }
                }
            }

            foreach (var request in deduped)
            {
                if (results.ContainsKey(request))
                {
                    continue;
                }

                var entries = GetRecentHorseSpeedEntries(request.RawHorseName, request.HorseId, request.BeforeDate, windowSize);
                if (entries == null || entries.Count == 0)
                {
                    results[request] = null;
                    continue;
                }

                float speedSum = 0f;
                int speedCount = 0;
                foreach (var entry in entries)
                {
                    if (!entry.DistanceYards.HasValue || entry.DistanceYards.Value <= 0)
                    {
                        continue;
                    }

                    if (!entry.WinningTimeMilliseconds.HasValue || entry.WinningTimeMilliseconds.Value <= 0)
                    {
                        continue;
                    }

                    var runnerTime = (float)entry.WinningTimeMilliseconds.Value;
                    if (entry.DistanceBeatenLengths.HasValue)
                    {
                        runnerTime += (float)entry.DistanceBeatenLengths.Value * MsPerLength;
                    }

                    if (runnerTime <= 0f)
                    {
                        continue;
                    }

                    var speed = entry.DistanceYards.Value / runnerTime;
                    if (!float.IsNaN(speed) && !float.IsInfinity(speed) && speed > 0f)
                    {
                        speedSum += speed;
                        speedCount++;
                    }
                }

                results[request] = speedCount > 0 ? speedSum / speedCount : (float?)null;
            }

            return results;
        }

        public IReadOnlyDictionary<HorseMetricRequest, int?> GetLastRaceDistancesBatch(IEnumerable<HorseMetricRequest> requests)
        {
            if (requests is null)
            {
                throw new ArgumentNullException(nameof(requests));
            }

            var deduped = requests
                .Where(r => r.HorseId.HasValue || !string.IsNullOrWhiteSpace(r.NormalizedHorseName))
                .Distinct()
                .ToList();

            var results = new Dictionary<HorseMetricRequest, int?>(deduped.Count);
            if (deduped.Count == 0)
            {
                return results;
            }

            var horseIdRequests = deduped
                .Where(r => r.HorseId.HasValue && r.HorseId.Value > 0)
                .ToList();

            if (horseIdRequests.Count > 0)
            {
                EnsureHorsePerformanceIndexes();
                using var connection = OpenConnection();
                var parameters = new DynamicParameters();
                var cte = BuildHorseIdRequestCte(horseIdRequests, parameters);

                var sql = $@"{cte}
SELECT req.RequestIndex,
       distance.DistanceYards
FROM Request req
OUTER APPLY (
    SELECT TOP (1) r.DistanceYards
    FROM RunnerResult rr
    INNER JOIN Race r ON r.RaceId = rr.RaceId
    WHERE rr.HorseId = req.HorseId
      AND r.DistanceYards IS NOT NULL
      AND (req.BeforeDate IS NULL OR r.RaceDate < req.BeforeDate)
    ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC
) distance;";

                var rows = connection.Query<RequestLastDistanceRow>(sql, parameters).ToList();
                foreach (var row in rows)
                {
                    if (row.RequestIndex < 0 || row.RequestIndex >= horseIdRequests.Count)
                    {
                        continue;
                    }

                    var request = horseIdRequests[row.RequestIndex];
                    results[request] = row.DistanceYards;
                }

                for (var i = 0; i < horseIdRequests.Count; i++)
                {
                    var request = horseIdRequests[i];
                    if (!results.ContainsKey(request))
                    {
                        results[request] = null;
                    }
                }
            }

            foreach (var request in deduped)
            {
                if (results.ContainsKey(request))
                {
                    continue;
                }

                var distance = GetLastRaceDistance(request.RawHorseName, request.HorseId, request.BeforeDate);
                results[request] = distance;
            }

            return results;
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
        public MLParameter? GetMostRecentMLParameter()
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
AND Fold IS NULL
  AND (Folds IS NULL OR Folds <= 1)
ORDER BY RunDate DESC, Id DESC";

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
        private sealed class RequestWinRateRow
        {
            public int RequestIndex { get; set; }
            public int? Wins { get; set; }
            public int? Starts { get; set; }
        }

        private sealed class RequestAverageSpeedRow
        {
            public int RequestIndex { get; set; }
            public double? AverageSpeed { get; set; }
        }

        private sealed class RequestLastDistanceRow
        {
            public int RequestIndex { get; set; }
            public int? DistanceYards { get; set; }
        }
        private static IEnumerable<IReadOnlyCollection<T>> Chunk<T>(IReadOnlyCollection<T> collection, int size)
        {
            if (collection is null)
            {
                throw new ArgumentNullException(nameof(collection));
            }

            if (size <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "Chunk size must be positive.");
            }

            if (collection.Count == 0)
            {
                yield break;
            }

            var offset = 0;
            while (offset < collection.Count)
            {
                var chunk = collection.Skip(offset).Take(size).ToList();
                offset += chunk.Count;
                yield return chunk;
            }
        }
    }
}