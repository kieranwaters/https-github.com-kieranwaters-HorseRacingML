using System.Data;
using System;
using Dapper;
using Microsoft.Data.SqlClient;
using HorseRacingML.Models;
using HorseRacingML.ML;
using HorseRacingML.Services;
using Microsoft.Extensions.Hosting;
using System.Diagnostics;
using System.Net.Sockets;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Threading;

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
        private static bool _goingSnapshotTableEnsured;
        private static bool _placedBetTableEnsured;
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
        Class,
        AgeRestriction,
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
        @Class,
        @AgeRestriction,
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
        Class = @Class,
        AgeRestriction = @AgeRestriction,
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

            using var conn = OpenConnection();
            EnsureUpcomingRaceTableExists(conn);
            EnsureGoingSnapshotTableExists(conn);
            var upcomingRaceId = conn.QuerySingle<int>(sql, race);

            // UpcomingRaces.Going is an upsert, so without this append-only log
            // an earlier going reading is lost every time the market is rescraped.
            if (!string.IsNullOrWhiteSpace(race.Going))
            {
                conn.Execute(
                    @"IF NOT EXISTS
                      (
                          SELECT 1
                          FROM dbo.GoingSnapshot
                          WHERE MarketId = @MarketId
                            AND Going = @Going
                            AND CapturedUtc >= DATEADD(HOUR, -1, SYSUTCDATETIME())
                      )
                      INSERT INTO dbo.GoingSnapshot
                          (UpcomingRaceId, MarketId, VenueName, RaceDate, Going, Surface)
                      VALUES
                          (@UpcomingRaceId, @MarketId, @VenueName, @RaceDate, @Going, @Surface);",
                    new
                    {
                        UpcomingRaceId = upcomingRaceId,
                        MarketId = race.MarketId,
                        VenueName = race.VenueName,
                        RaceDate = race.RaceDate,
                        Going = race.Going,
                        Surface = race.Surface
                    });
            }

            return upcomingRaceId;
        }

        private static void EnsureGoingSnapshotTableExists(IDbConnection conn)
        {
            if (_goingSnapshotTableEnsured)
            {
                return;
            }

            lock (SchemaLock)
            {
                if (_goingSnapshotTableEnsured)
                {
                    return;
                }

                const string sql = @"
IF OBJECT_ID(N'dbo.GoingSnapshot', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.GoingSnapshot
    (
        GoingSnapshotId BIGINT        IDENTITY(1,1) NOT NULL CONSTRAINT PK_GoingSnapshot PRIMARY KEY,
        UpcomingRaceId  INT           NULL,
        MarketId        NVARCHAR(64)  NULL,
        VenueName       NVARCHAR(256) NULL,
        RaceDate        DATE          NULL,
        Going           NVARCHAR(30)  NULL,
        Surface         NVARCHAR(64)  NULL,
        CapturedUtc     DATETIME2(0)  NOT NULL
            CONSTRAINT DF_GoingSnapshot_CapturedUtc DEFAULT SYSUTCDATETIME()
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_GoingSnapshot_MarketId_CapturedUtc' AND object_id = OBJECT_ID(N'dbo.GoingSnapshot', N'U'))
BEGIN
    CREATE INDEX IX_GoingSnapshot_MarketId_CapturedUtc ON dbo.GoingSnapshot(MarketId, CapturedUtc);
END;";

                conn.Execute(sql);
                _goingSnapshotTableEnsured = true;
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
        Class               TINYINT       NULL,
        AgeRestriction      NVARCHAR(64)  NULL,
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
            race.SourceRaceId = TruncateForColumn(race.SourceRaceId, 128);
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
        Purse = COALESCE(Purse, @Purse),
        ActualOff = COALESCE(ActualOff, @ActualOff),
        Source = COALESCE(Source, @Source),
        SourceRaceId = COALESCE(SourceRaceId, @SourceRaceId)
    WHERE RaceId=@ExistingId;

    SELECT @ExistingId;
END
ELSE
BEGIN
    INSERT INTO Race(CourseId, RaceDate, ScheduledOff, ActualOff, Title, RaceType, Class, AgeRestriction, Surface, Going, DistanceYards, DistanceText, RunnerCount, Status, WinningTimeMs, WinningTimeText, Purse, Source, SourceRaceId)
    VALUES(@CourseId, @RaceDate, @ScheduledOff, @ActualOff, @Title, @RaceType, @Class, @AgeRestriction, @Surface, @Going, @DistanceYards, @DistanceText, @RunnerCount, @Status, @WinningTimeMs, @WinningTimeText, @Purse, @Source, @SourceRaceId);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, race);
        }

        public void BulkInsertRunnerResults(IEnumerable<RunnerResult> results)
        {
            var list = results?.ToList();
            if (list == null || list.Count == 0) return;

            // Re-scraping a day used to append the same (RaceId, HorseId) rows
            // again. That corrupts rolling history and duplicates winners in the
            // training set, so filter both within the batch and against the DB.
            list = list
                .GroupBy(r => (r.RaceId, r.HorseId))
                .Select(g => g.First())
                .ToList();

            var raceIds = list.Select(r => r.RaceId).Distinct().ToArray();
            var existing = new HashSet<(int RaceId, int HorseId)>();
            if (raceIds.Length > 0)
            {
                using var lookupConn = new SqlConnection(_connectionString);
                lookupConn.Open();
                foreach (var row in lookupConn.Query<(int RaceId, int HorseId)>(
                    "SELECT RaceId, HorseId FROM RunnerResult WHERE RaceId IN @RaceIds",
                    new { RaceIds = raceIds }))
                {
                    existing.Add((row.RaceId, row.HorseId));
                }
            }

            list = list
                .Where(r => !existing.Contains((r.RaceId, r.HorseId)))
                .ToList();

            if (list.Count == 0)
            {
                return;
            }

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
            table.Columns.Add("OfficialRating", typeof(short));
            table.Columns.Add("Headgear", typeof(string));
            table.Columns.Add("Medication", typeof(string));
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
                    r.OfficialRating ?? (object)DBNull.Value,
                    TruncateForColumn(r.Headgear, 128) ?? (object)DBNull.Value,
                    TruncateForColumn(r.Medication, 128) ?? (object)DBNull.Value,
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
                    r.Comment ?? (object)DBNull.Value,
                    r.Source ?? (object)DBNull.Value,
                    TruncateForColumn(r.SourceRunnerId, 128) ?? (object)DBNull.Value
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

            // Classify the comments once, at ingest. Existing rows are handled
            // by BackfillRunStyles; this keeps new results in sync without the
            // training pass ever reading the wide Comment column.
            RunStyleStore.Upsert(
                conn,
                list.Select(r => new RunStyleInput(r.RaceId, r.HorseId, r.Comment)));
        }
        /// <summary>
        /// Updates the price-data and market-movement columns on existing
        /// results without touching the rest of the row. Used by the
        /// resumable backfills after a schema extension, so re-scraping a
        /// completed date cannot duplicate runners or overwrite the result
        /// fields.
        /// </summary>
        public void BulkUpdatePriceData(IEnumerable<RunnerResult> results)
            => BulkUpdateResultMetadata(results, includePriceData: true, includeMarketMovement: true);

        /// <summary>
        /// Updates only OpeningFraction/TouchedHighFraction/TouchedLowFraction
        /// on existing results. Used by the market-movement backfill so it
        /// cannot overwrite newer price-data values.
        /// </summary>
        public void BulkUpdateMarketMovement(IEnumerable<RunnerResult> results)
            => BulkUpdateResultMetadata(results, includePriceData: false, includeMarketMovement: true);

        private void BulkUpdateResultMetadata(
            IEnumerable<RunnerResult> results,
            bool includePriceData,
            bool includeMarketMovement)
        {
            var list = results?
                .Where(r =>
                    (includePriceData &&
                     (r.OfficialRating.HasValue ||
                      !string.IsNullOrWhiteSpace(r.Headgear) ||
                      !string.IsNullOrWhiteSpace(r.Medication))) ||
                    (includeMarketMovement &&
                     (!string.IsNullOrWhiteSpace(r.OpeningFraction) ||
                      !string.IsNullOrWhiteSpace(r.TouchedHighFraction) ||
                      !string.IsNullOrWhiteSpace(r.TouchedLowFraction))))
                .ToList();
            if (list is null || list.Count == 0)
            {
                return;
            }

            var table = new DataTable();
            table.Columns.Add("RaceId", typeof(int));
            table.Columns.Add("HorseId", typeof(int));
            if (includePriceData)
            {
                table.Columns.Add("OfficialRating", typeof(short));
                table.Columns.Add("Headgear", typeof(string));
                table.Columns.Add("Medication", typeof(string));
            }
            if (includeMarketMovement)
            {
                table.Columns.Add("OpeningFraction", typeof(string));
                table.Columns.Add("TouchedHighFraction", typeof(string));
                table.Columns.Add("TouchedLowFraction", typeof(string));
            }

            foreach (var r in list)
            {
                var values = new List<object?>
                {
                    r.RaceId,
                    r.HorseId
                };
                if (includePriceData)
                {
                    values.Add(r.OfficialRating ?? (object)DBNull.Value);
                    values.Add(TruncateForColumn(r.Headgear, 128) ?? (object)DBNull.Value);
                    values.Add(TruncateForColumn(r.Medication, 128) ?? (object)DBNull.Value);
                }
                if (includeMarketMovement)
                {
                    values.Add(TruncateForColumn(r.OpeningFraction, 32) ?? (object)DBNull.Value);
                    values.Add(TruncateForColumn(r.TouchedHighFraction, 32) ?? (object)DBNull.Value);
                    values.Add(TruncateForColumn(r.TouchedLowFraction, 32) ?? (object)DBNull.Value);
                }

                table.Rows.Add(values.ToArray());
            }

            var columnTypes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["OfficialRating"] = "SMALLINT",
                ["Headgear"] = "NVARCHAR(128)",
                ["Medication"] = "NVARCHAR(128)",
                ["OpeningFraction"] = "NVARCHAR(32)",
                ["TouchedHighFraction"] = "NVARCHAR(32)",
                ["TouchedLowFraction"] = "NVARCHAR(32)"
            };
            var metadataColumns = table.Columns
                .Cast<DataColumn>()
                .Where(column => !column.ColumnName.Equals("RaceId", StringComparison.OrdinalIgnoreCase) &&
                                 !column.ColumnName.Equals("HorseId", StringComparison.OrdinalIgnoreCase))
                .Select(column => column.ColumnName)
                .ToList();
            if (metadataColumns.Count == 0)
            {
                return;
            }

            var createColumns = string.Join(
                ",\n    ",
                metadataColumns.Select(name => $"{name} {columnTypes[name]} NULL"));
            var updateSet = string.Join(
                ",\n    ",
                metadataColumns.Select(name => $"{name} = COALESCE(m.{name}, rr.{name})"));

            using var conn = new SqlConnection(_connectionString);
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = $@"
CREATE TABLE #ResultMetadata
(
    RaceId INT NOT NULL,
    HorseId INT NOT NULL,
    {createColumns}
);";
            cmd.ExecuteNonQuery();

            using (var bulk = new SqlBulkCopy(conn)
            {
                DestinationTableName = "#ResultMetadata",
                BatchSize = 5000
            })
            {
                foreach (DataColumn col in table.Columns)
                {
                    bulk.ColumnMappings.Add(col.ColumnName, col.ColumnName);
                }
                bulk.WriteToServer(table);
            }

            cmd.CommandText = $@"
CREATE INDEX IX_ResultMetadata ON #ResultMetadata(RaceId, HorseId);
UPDATE rr
SET {updateSet}
FROM dbo.RunnerResult rr
JOIN #ResultMetadata m ON m.RaceId = rr.RaceId AND m.HorseId = rr.HorseId;
DROP TABLE #ResultMetadata;";
            cmd.ExecuteNonQuery();
        }

        /// <summary>
        /// Closing-line value for recent bets. The closing price is the last
        /// pre-off RunnerFlow snapshot for the same market/selection.
        /// </summary>
        public IReadOnlyList<ClvRow> GetClvRows(int limit = 200)
        {
            using var conn = OpenConnection();
            EnsureRunnerFlowTableExists(conn);
            EnsurePlacedBetTableExists(conn);

            const string sql = @"
WITH last_price AS
(
    SELECT rf.MarketId,
           rf.SelectionId,
           rf.BackPrice1,
           ROW_NUMBER() OVER (
               PARTITION BY rf.MarketId, rf.SelectionId
               ORDER BY rf.CapturedUtc DESC, rf.RunnerFlowId DESC) AS rn
    FROM dbo.RunnerFlow rf
    WHERE rf.BackPrice1 IS NOT NULL
)
SELECT TOP (@Limit)
       pb.MarketId,
       pb.SelectionId,
       pb.HorseName,
       pb.EffectiveOdds AS BetOdds,
       lp.BackPrice1 AS ClosingOdds,
       CASE WHEN lp.BackPrice1 > 0
            THEN (CAST(pb.EffectiveOdds AS FLOAT) / CAST(lp.BackPrice1 AS FLOAT)) - 1
       END AS Clv,
       pb.PlacedUtc
FROM dbo.PlacedBet pb
LEFT JOIN last_price lp
       ON lp.MarketId = pb.MarketId
      AND lp.SelectionId = pb.SelectionId
      AND lp.rn = 1
ORDER BY pb.PlacedUtc DESC";

            return conn.Query<ClvRow>(
                sql,
                new { Limit = Math.Clamp(limit, 1, 5000) }).ToList();
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
                                           Class,
                                           AgeRestriction,
                                           Surface,
                                           Going,
                                           DistanceYards,
                                           DistanceText,
                                           RunnerCount,
                                           BackBookPercentage,
                                           LayBookPercentage
                                    FROM UpcomingRaces
                                    WHERE MarketId = @MarketId";

            using var conn = OpenConnection();
            EnsureUpcomingRaceTableExists(conn);
            return conn.QueryFirstOrDefault<UpcomingRace>(sql, new { MarketId = marketId.Trim() });
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
        public DateTime GetEarliestRaceDate()
        {
            using var conn = OpenConnection();
            const string sql = "SELECT MIN(r.RaceDate) FROM Race r WHERE r.RaceDate IS NOT NULL AND r.RaceDate <= CAST(GETDATE() AS date) AND EXISTS (SELECT 1 FROM RunnerResult rr WHERE rr.RaceId = r.RaceId)";
            var earliest = conn.QuerySingleOrDefault<DateTime?>(sql);
            // When no races exist yet, start from tomorrow so we scrape backwards from today
            return earliest ?? DateTime.Today.AddDays(1);
        }

        /// <summary>
        /// Earliest race date in an inclusive range, used to record the start of a
        /// training window on MLParameters rows.
        /// </summary>
        public DateTime? GetEarliestRaceDateBetween(DateTime startInclusive, DateTime endInclusive)
        {
            using var conn = OpenConnection();
            const string sql = "SELECT MIN(RaceDate) FROM Race WHERE RaceDate >= @Start AND RaceDate <= @End";
            return conn.QuerySingleOrDefault<DateTime?>(sql, new { Start = startInclusive.Date, End = endInclusive.Date });
        }
        public DateTime GetLatestRaceDate()
        {
            using var conn = OpenConnection();
            const string sql = "SELECT MAX(r.RaceDate) FROM Race r WHERE r.RaceDate IS NOT NULL AND r.RaceDate <= CAST(GETDATE() AS date) AND EXISTS (SELECT 1 FROM RunnerResult rr WHERE rr.RaceId = r.RaceId)";
            var latest = conn.QuerySingleOrDefault<DateTime?>(sql);
            // When no races exist yet, return a past date so scraping can begin
            return latest ?? new DateTime(2000, 1, 1);
        }

        /// <summary>
        /// Dates in the range where every race already has at least one runner result.
        /// Used to resume a long backfill without re-downloading completed days.
        /// </summary>
        public HashSet<DateTime> GetDatesWithCompleteResults(DateTime minInclusive, DateTime maxInclusive)
        {
            if (maxInclusive < minInclusive) return new HashSet<DateTime>();
            const string sql = @"
SELECT RaceDate
FROM (
    SELECT r.RaceDate,
           CASE WHEN EXISTS (SELECT 1 FROM RunnerResult rr WHERE rr.RaceId = r.RaceId) THEN 1 ELSE 0 END AS HasResults
    FROM Race r
    WHERE r.RaceDate >= @Min AND r.RaceDate <= @Max
) t
GROUP BY RaceDate
HAVING MIN(HasResults) = 1";
            using var conn = OpenConnection();
            return conn.Query<DateTime>(sql, new { Min = minInclusive.Date, Max = maxInclusive.Date }).ToHashSet();
        }
        /// <summary>
        /// Classifies historical comments that do not yet have a current-version
        /// run-style row. Safe to stop and restart; already-classified rows are
        /// skipped unless <see cref="RunStyleParser.Version"/> is increased.
        /// </summary>
        public int BackfillRunStyles(int batchSize = 20_000, int? maxRows = null, Action<string>? log = null, CancellationToken cancellationToken = default)
        {
            batchSize = Math.Max(1, batchSize);
            int limit = maxRows.HasValue ? Math.Max(0, maxRows.Value) : int.MaxValue;
            if (limit == 0)
            {
                return 0;
            }

            using var readConnection = new SqlConnection(_connectionString);
            using var writeConnection = new SqlConnection(_connectionString);
            readConnection.Open();
            writeConnection.Open();
            RunStyleStore.EnsureTable(writeConnection);

            const string sql = @"
SELECT TOP (@MaxRows) rr.RaceId, rr.HorseId, rr.Comment
FROM RunnerResult rr
LEFT JOIN dbo.RunnerRunStyle rs
       ON rs.RaceId = rr.RaceId AND rs.HorseId = rr.HorseId
WHERE rr.Comment IS NOT NULL
  AND LEN(LTRIM(RTRIM(rr.Comment))) > 0
  AND (rs.RaceId IS NULL OR rs.Version < @Version)
ORDER BY rr.RunnerResultId";

            var batch = new List<RunStyleInput>(batchSize);
            int classified = 0;

            foreach (var row in readConnection.Query<RunStyleSourceRow>(
                         sql,
                         new { MaxRows = limit, Version = RunStyleParser.Version },
                         commandTimeout: 6000,
                         buffered: false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                batch.Add(new RunStyleInput(row.RaceId, row.HorseId, row.Comment));
                if (batch.Count >= batchSize)
                {
                    classified += RunStyleStore.Upsert(writeConnection, batch);
                    batch.Clear();
                    log?.Invoke($"[run-style] {classified} rows classified");
                }
            }

            if (batch.Count > 0)
            {
                classified += RunStyleStore.Upsert(writeConnection, batch);
            }

            log?.Invoke($"[run-style] finished: {classified} rows classified");
            return classified;
        }

        private sealed class RunStyleSourceRow
        {
            public int RaceId { get; set; }
            public int HorseId { get; set; }
            public string? Comment { get; set; }
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
            const string sql = @"
INSERT INTO RaceScreen(MarketId, RaceDate, OffTime, Title, VenueName, VenueCountry, EventDateText, RaceDetails, BackBookPercentage, LayBookPercentage)
VALUES(@MarketId, @RaceDate, @OffTime, @Title, @VenueName, @VenueCountry, @EventDateText, @RaceDetails, @BackBookPercentage, @LayBookPercentage);";
            using var conn = OpenConnection();
            EnsureRaceScreenTableExists(conn);
            conn.Execute(sql, screen);
        }

        public void InsertRunnerFlow(RunnerFlow flow)
        {
            const string sql = @"
INSERT INTO RunnerFlow(MarketId, SelectionId, ClothNumber, Draw, HorseName, JockeyName, BackPrice1, BackPrice2, BackPrice3, LayPrice1, LayPrice2, LayPrice3, CapturedUtc, AiOdds)
VALUES(@MarketId, @SelectionId, @ClothNumber, @Draw, @HorseName, @JockeyName, @BackPrice1, @BackPrice2, @BackPrice3, @LayPrice1, @LayPrice2, @LayPrice3, @CapturedUtc, @AiOdds);";
            using var conn = OpenConnection();
            EnsureRunnerFlowTableExists(conn);
            conn.Execute(sql, flow);
        }

        public bool HasPlacedBet(string marketId)
        {
            if (string.IsNullOrWhiteSpace(marketId))
            {
                return false;
            }

            using var conn = OpenConnection();
            EnsurePlacedBetTableExists(conn);
            return conn.QuerySingleOrDefault<int?>(
                "SELECT TOP (1) 1 FROM dbo.PlacedBet WHERE MarketId = @MarketId",
                new { MarketId = marketId.Trim() }).HasValue;
        }

        public void InsertPlacedBet(PlacedBet bet)
        {
            if (bet is null)
            {
                throw new ArgumentNullException(nameof(bet));
            }

            const string sql = @"
INSERT INTO dbo.PlacedBet
    (MarketId, SelectionId, HorseName, RequestedOdds, EffectiveOdds, Stake, AiProbability, MarketProbability, Differential, PlacedUtc)
VALUES
    (@MarketId, @SelectionId, @HorseName, @RequestedOdds, @EffectiveOdds, @Stake, @AiProbability, @MarketProbability, @Differential, @PlacedUtc);";

            using var conn = OpenConnection();
            EnsurePlacedBetTableExists(conn);
            conn.Execute(sql, new
            {
                bet.MarketId,
                bet.SelectionId,
                bet.HorseName,
                bet.RequestedOdds,
                bet.EffectiveOdds,
                bet.Stake,
                bet.AiProbability,
                bet.MarketProbability,
                bet.Differential,
                PlacedUtc = bet.PlacedUtc == default ? DateTime.UtcNow : bet.PlacedUtc
            });
        }

        private static string? TruncateForColumn(string? value, int maxLength)
            => string.IsNullOrEmpty(value) || value.Length <= maxLength
                ? value
                : value.Substring(0, maxLength);

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
        BackBookPercentage DECIMAL(9,2) NULL,
        LayBookPercentage  DECIMAL(9,2) NULL
    );
END";

                conn.Execute(sql);
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
        RunnerFlowId BIGINT        IDENTITY(1,1) PRIMARY KEY,
        MarketId     NVARCHAR(32)  NULL,
        SelectionId  NVARCHAR(32)  NULL,
        ClothNumber  TINYINT       NULL,
        Draw         TINYINT       NULL,
        HorseName    NVARCHAR(256) NULL,
        JockeyName   NVARCHAR(256) NULL,
        BackPrice1   DECIMAL(9,2)  NULL,
        BackPrice2   DECIMAL(9,2)  NULL,
        BackPrice3   DECIMAL(9,2)  NULL,
        LayPrice1    DECIMAL(9,2)  NULL,
        LayPrice2    DECIMAL(9,2)  NULL,
        LayPrice3    DECIMAL(9,2)  NULL,
        CapturedUtc  DATETIME2(0)  NOT NULL CONSTRAINT DF_RunnerFlow_CapturedUtc DEFAULT SYSUTCDATETIME(),
        AiOdds       FLOAT         NULL
    );
END
IF COL_LENGTH(N'dbo.RunnerFlow', N'CapturedUtc') IS NULL
    ALTER TABLE dbo.RunnerFlow ADD CapturedUtc DATETIME2(0) NOT NULL
        CONSTRAINT DF_RunnerFlow_CapturedUtc DEFAULT SYSUTCDATETIME();
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_RunnerFlow_MarketSelectionPrice' AND object_id = OBJECT_ID(N'dbo.RunnerFlow', N'U'))
BEGIN
    CREATE INDEX IX_RunnerFlow_MarketSelectionPrice
        ON dbo.RunnerFlow(MarketId, SelectionId, RunnerFlowId)
        INCLUDE (HorseName, BackPrice1);
END";

                conn.Execute(sql);
                _runnerFlowTableEnsured = true;
            }
        }

        private static void EnsurePlacedBetTableExists(IDbConnection conn)
        {
            if (_placedBetTableEnsured) return;

            lock (SchemaLock)
            {
                if (_placedBetTableEnsured) return;

                const string sql = @"
IF OBJECT_ID(N'dbo.PlacedBet', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.PlacedBet
    (
        PlacedBetId       BIGINT        IDENTITY(1,1) PRIMARY KEY,
        MarketId          NVARCHAR(32)  NOT NULL,
        SelectionId       NVARCHAR(32)  NULL,
        HorseName         NVARCHAR(256) NULL,
        RequestedOdds     DECIMAL(9,2)  NOT NULL,
        EffectiveOdds     DECIMAL(9,2)  NOT NULL,
        Stake             DECIMAL(18,2) NOT NULL,
        AiProbability     FLOAT         NOT NULL,
        MarketProbability FLOAT         NOT NULL,
        Differential      FLOAT         NOT NULL,
        PlacedUtc         DATETIME2(0)  NOT NULL
    );

    CREATE INDEX IX_PlacedBet_MarketId ON dbo.PlacedBet(MarketId);
END";

                conn.Execute(sql);
                _placedBetTableEnsured = true;
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
        
        public void InsertRunnerResults(IEnumerable<RunnerResult> results)
        {
            var list = results.ToList();
            foreach (var r in list)
            {
                StripBracketedText(r);
            }
            const string sql = @"
IF NOT EXISTS (SELECT 1 FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId)
    INSERT INTO RunnerResult(RaceId, HorseId, TrainerId, JockeyId, SaddleclothNumber, Draw, Age, WeightLbs, WeightText, OfficialRating, Headgear, Medication, FinishPos, OutcomeCode, DistanceBeatenText, DistanceBeatenLengths, SP_Fraction, SP_Decimal, FavTag, OpeningFraction, TouchedHighFraction, TouchedLowFraction, Comment, Source, SourceRunnerId)
    VALUES(@RaceId, @HorseId, @TrainerId, @JockeyId, @SaddleclothNumber, @Draw, @Age, @WeightLbs, @WeightText, @OfficialRating, @Headgear, @Medication, @FinishPos, @OutcomeCode, @DistanceBeatenText, @DistanceBeatenLengths, @SP_Fraction, @SP_Decimal, @FavTag, @OpeningFraction, @TouchedHighFraction, @TouchedLowFraction, @Comment, @Source, @SourceRunnerId);";
            using var conn = OpenConnection();
            conn.Execute(sql, list);
        }
        public RacingRepository(string connectionString)
        {
            _connectionString = connectionString;
        }

        public short InsertCourse(Course course)
        {
            StripBracketedText(course);
            if (CourseLocationCatalog.TryMatch(course.Name, out var location))
            {
                course.Latitude ??= location.Latitude;
                course.Longitude ??= location.Longitude;
                course.TimeZoneId ??= location.TimeZoneId;
            }

            const string sql = @"
IF EXISTS (SELECT 1 FROM Course WHERE Name = @Name AND ISNULL(Country,'') = ISNULL(@Country,''))
BEGIN
    UPDATE Course
    SET Latitude = COALESCE(Latitude, @Latitude),
        Longitude = COALESCE(Longitude, @Longitude),
        TimeZoneId = COALESCE(TimeZoneId, @TimeZoneId)
    WHERE Name = @Name AND ISNULL(Country,'') = ISNULL(@Country,'');

    SELECT TOP 1 CourseId FROM Course WHERE Name = @Name AND ISNULL(Country,'') = ISNULL(@Country,'') ORDER BY CourseId;
END
ELSE
BEGIN
    INSERT INTO Course(Name, Country, Latitude, Longitude, TimeZoneId)
    VALUES(@Name, @Country, @Latitude, @Longitude, @TimeZoneId);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<short>(sql, course);
        }
        public int InsertMLParameter(MLParameter param)
        {
            const string sql = @"
INSERT INTO MLParameters(
    RunDate, EarliestTrainingDate, Units, Dropout, Layers, LearningRate,
    TrainAccuracy, ValidationAccuracy, ValidationLoss, Epochs, BatchSize, Folds,
    TrainLoss, TrainBrier, Fold, ValidationBrier,
    Objective, Seed, BestEpoch, MarketOffsetScale, MarketOffsetMode, MarketBlind,
    BlendModelWeight, BlendMarketWeight, NormalizationMode, FeatureBlocks, RaceBatchSize,
    TrainingStartDate, TrainingEndDate, DevStartDate, DevEndDate, TestStartDate, TestEndDate,
    TrainingRaceCount, DevRaceCount, TestRaceCount, TrainingRowCount, DevRowCount, TestRowCount,
    DevAccuracy, DevLoss, DevBrier, DevRaceLogLoss, DevTop3Accuracy,
    DevMarketRaceLogLoss, DevMarketTop1Accuracy, DevMarketTop3Accuracy,
    DevTemperature, DevMarketLogLoss, DevMarketBrier,
    DevFairMarketLogLoss, DevFairMarketBrier, DevBlendLogLoss, DevBlendBrier, DevBlendRaceLogLoss,
    TestAccuracy, TestLoss, TestBrier, TestRaceLogLoss, TestTop3Accuracy,
    TestMarketRaceLogLoss, TestMarketTop1Accuracy, TestMarketTop3Accuracy,
    TestTemperature, TestMarketLogLoss, TestMarketBrier,
    TestFairMarketLogLoss, TestFairMarketBrier, TestBlendLogLoss, TestBlendBrier, TestBlendRaceLogLoss,
    WinRateAlpha, WinRateBeta, SelectedOn, ModelArtifact, ModelHash, FeatureSchemaHash, EnsembleSize, IsActive)
VALUES(
    @RunDate, @EarliestTrainingDate, @Units, @Dropout, @Layers, @LearningRate,
    @TrainAccuracy, @ValidationAccuracy, @ValidationLoss, @Epochs, @BatchSize, @Folds,
    @TrainLoss, @TrainBrier, @Fold, @ValidationBrier,
    @Objective, @Seed, @BestEpoch, @MarketOffsetScale, @MarketOffsetMode, @MarketBlind,
    @BlendModelWeight, @BlendMarketWeight, @NormalizationMode, @FeatureBlocks, @RaceBatchSize,
    @TrainingStartDate, @TrainingEndDate, @DevStartDate, @DevEndDate, @TestStartDate, @TestEndDate,
    @TrainingRaceCount, @DevRaceCount, @TestRaceCount, @TrainingRowCount, @DevRowCount, @TestRowCount,
    @DevAccuracy, @DevLoss, @DevBrier, @DevRaceLogLoss, @DevTop3Accuracy,
    @DevMarketRaceLogLoss, @DevMarketTop1Accuracy, @DevMarketTop3Accuracy,
    @DevTemperature, @DevMarketLogLoss, @DevMarketBrier,
    @DevFairMarketLogLoss, @DevFairMarketBrier, @DevBlendLogLoss, @DevBlendBrier, @DevBlendRaceLogLoss,
    @TestAccuracy, @TestLoss, @TestBrier, @TestRaceLogLoss, @TestTop3Accuracy,
    @TestMarketRaceLogLoss, @TestMarketTop1Accuracy, @TestMarketTop3Accuracy,
    @TestTemperature, @TestMarketLogLoss, @TestMarketBrier,
    @TestFairMarketLogLoss, @TestFairMarketBrier, @TestBlendLogLoss, @TestBlendBrier, @TestBlendRaceLogLoss,
    @WinRateAlpha, @WinRateBeta, @SelectedOn, @ModelArtifact, @ModelHash, @FeatureSchemaHash, @EnsembleSize, ISNULL(@IsActive, 0));
SELECT CAST(SCOPE_IDENTITY() as int);";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, param);
        }

        /// <summary>
        /// Promotes one MLParameters row to the live model and records the
        /// artifact that contains its weights. The update is atomic so there can
        /// never be two active rows.
        /// </summary>
        public void ActivateMLParameter(
            int id,
            string? modelArtifact,
            string? modelHash,
            string? featureSchemaHash)
        {
            if (id <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(id));
            }

            const string sql = @"
SET NOCOUNT ON;
UPDATE MLParameters SET IsActive = 0 WHERE ISNULL(IsActive, 0) = 1;
UPDATE MLParameters
SET IsActive = 1,
    ModelArtifact = COALESCE(@ModelArtifact, ModelArtifact),
    ModelHash = COALESCE(@ModelHash, ModelHash),
    FeatureSchemaHash = COALESCE(@FeatureSchemaHash, FeatureSchemaHash)
WHERE Id = @Id;";

            using var conn = OpenConnection();
            using var tx = conn.BeginTransaction();
            conn.Execute(sql, new { Id = id, ModelArtifact = modelArtifact, ModelHash = modelHash, FeatureSchemaHash = featureSchemaHash }, tx);
            tx.Commit();
        }

        public MLParameter? GetMLParameterById(int id)
        {
            using var conn = OpenConnection();
            return conn.QueryFirstOrDefault<MLParameter>(
                "SELECT * FROM MLParameters WHERE Id = @Id",
                new { Id = id });
        }

        public MLParameter? GetActiveMLParameter()
        {
            const string sql = @"
SELECT TOP (1)
    Id, RunDate, EarliestTrainingDate, Units, Dropout, Layers, LearningRate,
    TrainAccuracy, ValidationAccuracy, ValidationLoss, Epochs, BatchSize, Folds,
    TrainLoss, TrainBrier, Fold, ValidationBrier,
    Objective, Seed, BestEpoch, MarketOffsetScale, MarketOffsetMode, MarketBlind,
    BlendModelWeight, BlendMarketWeight, NormalizationMode, FeatureBlocks, RaceBatchSize,
    TrainingStartDate, TrainingEndDate, DevStartDate, DevEndDate, TestStartDate, TestEndDate,
    TrainingRaceCount, DevRaceCount, TestRaceCount, TrainingRowCount, DevRowCount, TestRowCount,
    DevAccuracy, DevLoss, DevBrier, DevRaceLogLoss, DevTop3Accuracy,
    DevMarketRaceLogLoss, DevMarketTop1Accuracy, DevMarketTop3Accuracy,
    DevTemperature, DevMarketLogLoss, DevMarketBrier,
    DevFairMarketLogLoss, DevFairMarketBrier, DevBlendLogLoss, DevBlendBrier, DevBlendRaceLogLoss,
    TestAccuracy, TestLoss, TestBrier, TestRaceLogLoss, TestTop3Accuracy,
    TestMarketRaceLogLoss, TestMarketTop1Accuracy, TestMarketTop3Accuracy,
    TestTemperature, TestMarketLogLoss, TestMarketBrier,
    TestFairMarketLogLoss, TestFairMarketBrier, TestBlendLogLoss, TestBlendBrier, TestBlendRaceLogLoss,
    WinRateAlpha, WinRateBeta, SelectedOn, ModelArtifact, ModelHash, FeatureSchemaHash, EnsembleSize, IsActive
FROM MLParameters
WHERE ISNULL(IsActive, 0) = 1
ORDER BY RunDate DESC, Id DESC;";

            using var conn = OpenConnection();
            return conn.QueryFirstOrDefault<MLParameter>(sql);
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

        /// <summary>
        /// Race IDs in the date range that have a usable pre-off market: at least
        /// two runners carry an opening price or a T-N Betfair anchor. Training
        /// on these races keeps the market features present in most rows,
        /// matching the live pipeline where every runner has a current back
        /// price.
        /// </summary>
        public DateTime? GetRaceDate(int raceId)
        {
            using var conn = OpenConnection();
            const string sql = "SELECT RaceDate FROM Race WHERE RaceId = @RaceId";
            return conn.QuerySingleOrDefault<DateTime?>(sql, new { RaceId = raceId });
        }
        public List<int> GetRaceIdsWithOpeningPrice(DateTime startInclusive, DateTime endInclusive)
        {
            if (endInclusive < startInclusive)
            {
                throw new ArgumentException("The end date must be on or after the start date.", nameof(endInclusive));
            }

            using var conn = OpenConnection();
            const string sql = @"
SELECT r.RaceId
FROM Race r
JOIN RunnerResult rr ON rr.RaceId = r.RaceId
WHERE r.RaceDate >= @Start
  AND r.RaceDate <= @End
  AND (
        (rr.OpeningFraction IS NOT NULL AND LTRIM(RTRIM(rr.OpeningFraction)) <> '')
        OR (rr.MarketAnchorPrice IS NOT NULL AND rr.MarketAnchorPrice > 1)
      )
GROUP BY r.RaceId
HAVING COUNT(*) >= 2
ORDER BY MIN(r.RaceDate), r.RaceId";
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
            const string sql = @"SELECT r.RaceId, r.RaceDate, r.Title, c.Name AS CourseName,
       r.RaceType, r.Class, r.AgeRestriction, r.Surface, r.Going,
       r.DistanceYards, r.DistanceText, r.RunnerCount
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


        public MLParameter? GetBestMLParameter()
        {
            const string sql = @"SELECT TOP (1)
    Id,
    RunDate,
    EarliestTrainingDate,
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
    ValidationBrier,
    Objective,
    Seed,
    BestEpoch,
    MarketOffsetScale,
    MarketOffsetMode,
    MarketBlind,
    BlendModelWeight,
    BlendMarketWeight,
    NormalizationMode,
    RaceBatchSize,
    TrainingStartDate,
    TrainingEndDate,
    DevStartDate,
    DevEndDate,
    TestStartDate,
    TestEndDate,
    TrainingRaceCount,
    DevRaceCount,
    TestRaceCount,
    TrainingRowCount,
    DevRowCount,
    TestRowCount,
    DevAccuracy,
    DevLoss,
    DevBrier,
    DevRaceLogLoss,
    DevTop3Accuracy,
    DevMarketRaceLogLoss,
    DevMarketTop1Accuracy,
    DevMarketTop3Accuracy,
    DevTemperature,
    DevMarketLogLoss,
    DevMarketBrier,
    DevFairMarketLogLoss,
    DevFairMarketBrier,
    DevBlendLogLoss,
    DevBlendBrier,
    DevBlendRaceLogLoss,
    TestAccuracy,
    TestLoss,
    TestBrier,
    TestRaceLogLoss,
    TestTop3Accuracy,
    TestMarketRaceLogLoss,
    TestMarketTop1Accuracy,
    TestMarketTop3Accuracy,
    TestTemperature,
    TestMarketLogLoss,
    TestMarketBrier,
    TestFairMarketLogLoss,
    TestFairMarketBrier,
    TestBlendLogLoss,
    TestBlendBrier,
    TestBlendRaceLogLoss,
    WinRateAlpha,
    WinRateBeta,
    SelectedOn,
    ModelArtifact,
    ModelHash,
    FeatureSchemaHash,
    EnsembleSize,
    IsActive
FROM MLParameters
WHERE Units > 0
  AND Layers > 0
  AND Epochs > 0
  AND BatchSize > 0
  AND LearningRate > 0
  AND Fold IS NULL
  -- Baseline rows are reference points, not trainable hyperparameter
  -- candidates. Keep them out of automatic selection.
  AND ISNULL(SelectedOn, '') <> 'Baseline'
  AND ISNULL(SelectedOn, '') <> 'MarketRejected'
  AND COALESCE(DevBrier, ValidationBrier) IS NOT NULL
  AND COALESCE(DevBrier, ValidationBrier) > 0
-- Selection is model-only: legacy rows marked Rejected were rejected by the
-- old post-model blend gate, so they remain eligible here.
ORDER BY CASE WHEN ISNULL(IsActive, 0) = 1 THEN 0 ELSE 1 END,
         ISNULL(DevRaceLogLoss, 1E9) ASC,
         COALESCE(DevBrier, ValidationBrier) ASC,
         ISNULL(COALESCE(DevLoss, ValidationLoss), 1E9) ASC,
         ISNULL(DevBlendLogLoss, 1E9) ASC,
         -- The final row is retrained on train+dev and carries the
         -- persisted weights. Prefer it over the dev-only candidate when
         -- their dev metrics tie.
         CASE WHEN SelectedOn = 'Test' THEN 0 ELSE 1 END,
         ISNULL(TrainBrier, 1E9) ASC,
         RunDate DESC";

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
                                           Class,
                                           AgeRestriction,
                                           Surface,
                                           Going,
                                           DistanceYards,
                                           DistanceText,
                                           RunnerCount,
                                           BackBookPercentage,
                                           LayBookPercentage
                                    FROM UpcomingRaces
                                    WHERE RaceDate = @RaceDate";

            using var conn = OpenConnection();
            var candidates = conn.Query<UpcomingRace>(sql, new { RaceDate = raceDate.Date }).ToList();
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

        /// <summary>
        /// Resolves (creating where needed) horse/trainer/jockey rows in a single
        /// connection. Used by the results scraper so a race with 14 runners costs one
        /// connection instead of ~42.
        /// </summary>
        public (Dictionary<string, int> Horses, Dictionary<string, int> Trainers, Dictionary<string, int> Jockeys) EnsureReferenceData(
            IEnumerable<string> horseNames,
            IEnumerable<string> trainerNames,
            IEnumerable<string> jockeyNames)
        {
            const string horseSql = @"
IF EXISTS (SELECT 1 FROM Horse WHERE Name=@Name)
    SELECT TOP 1 HorseId FROM Horse WHERE Name=@Name ORDER BY HorseId;
ELSE
BEGIN
    INSERT INTO Horse(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            const string trainerSql = @"
IF EXISTS (SELECT 1 FROM Trainer WHERE Name=@Name)
    SELECT TOP 1 TrainerId FROM Trainer WHERE Name=@Name ORDER BY TrainerId;
ELSE
BEGIN
    INSERT INTO Trainer(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            const string jockeySql = @"
IF EXISTS (SELECT 1 FROM Jockey WHERE Name=@Name)
    SELECT TOP 1 JockeyId FROM Jockey WHERE Name=@Name ORDER BY JockeyId;
ELSE
BEGIN
    INSERT INTO Jockey(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";

            static IEnumerable<string> Clean(IEnumerable<string>? names) =>
                (names ?? Enumerable.Empty<string>())
                    .Where(n => !string.IsNullOrWhiteSpace(n))
                    .Select(RemoveBracketedText)
                    .Where(n => !string.IsNullOrWhiteSpace(n))
                    .Distinct(StringComparer.OrdinalIgnoreCase);

            var horses = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var trainers = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var jockeys = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            using var conn = OpenConnection();
            // Serializable so two scraper workers cannot both INSERT the same name.
            using var tx = conn.BeginTransaction(System.Data.IsolationLevel.Serializable);
            foreach (var name in Clean(horseNames)) horses[name] = conn.QuerySingle<int>(horseSql, new { Name = name }, tx);
            foreach (var name in Clean(trainerNames)) trainers[name] = (short)conn.QuerySingle<int>(trainerSql, new { Name = name }, tx);
            foreach (var name in Clean(jockeyNames)) jockeys[name] = (short)conn.QuerySingle<int>(jockeySql, new { Name = name }, tx);
            tx.Commit();

            return (horses, trainers, jockeys);
        }

        public int InsertRunnerResult(RunnerResult result)
        {
            StripBracketedText(result);
            result.SourceRunnerId = TruncateForColumn(result.SourceRunnerId, 128);
            const string sql = @"
IF EXISTS (SELECT 1 FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId)
    SELECT TOP 1 RunnerResultId FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId ORDER BY RunnerResultId;
ELSE
BEGIN
    INSERT INTO RunnerResult(RaceId, HorseId, TrainerId, JockeyId, SaddleclothNumber, Draw, Age, WeightLbs, WeightText, OfficialRating, Headgear, Medication, FinishPos, OutcomeCode, DistanceBeatenText, DistanceBeatenLengths, SP_Fraction, SP_Decimal, FavTag, OpeningFraction, TouchedHighFraction, TouchedLowFraction, Comment, Source, SourceRunnerId)
    VALUES(@RaceId, @HorseId, @TrainerId, @JockeyId, @SaddleclothNumber, @Draw, @Age, @WeightLbs, @WeightText, @OfficialRating, @Headgear, @Medication, @FinishPos, @OutcomeCode, @DistanceBeatenText, @DistanceBeatenLengths, @SP_Fraction, @SP_Decimal, @FavTag, @OpeningFraction, @TouchedHighFraction, @TouchedLowFraction, @Comment, @Source, @SourceRunnerId);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            var sqlConnection = (SqlConnection)conn;
            var runnerResultId = sqlConnection.QuerySingle<int>(sql, result);
            RunStyleStore.Upsert(
                sqlConnection,
                new[] { new RunStyleInput(result.RaceId, result.HorseId, result.Comment) });
            return runnerResultId;
        }
    }
}