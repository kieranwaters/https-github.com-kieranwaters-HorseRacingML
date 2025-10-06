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

            const string header = @"INSERT INTO RunnerFlow(MarketId, SelectionId, ClothNumber, Draw, HorseName, JockeyName, BackPrice1, BackPrice2, BackPrice3, LayPrice1, LayPrice2, LayPrice3, AiOdds)
VALUES";

            var sqlBuilder = new StringBuilder(header.Length + flowList.Count * 128);
            sqlBuilder.Append(header);

            var parameters = new DynamicParameters();

            for (var i = 0; i < flowList.Count; i++)
            {
                var flow = flowList[i];
                var suffix = i.ToString(CultureInfo.InvariantCulture);

                sqlBuilder.Append("(@MarketId").Append(suffix)
                    .Append(", @SelectionId").Append(suffix)
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
                    .Append(')');

                if (i < flowList.Count - 1)
                {
                    sqlBuilder.Append(", ");
                }

                parameters.Add($"MarketId{suffix}", flow.MarketId);
                parameters.Add($"SelectionId{suffix}", flow.SelectionId);
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
            }

            using var conn = OpenConnection();
            EnsureRunnerFlowTableExists(conn);

            using var transaction = conn.BeginTransaction();
            conn.Execute(sqlBuilder.ToString(), parameters, transaction);
            transaction.Commit();
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
            return conn.QuerySingle<int>(sql, race);
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

        public IReadOnlyDictionary<string, int> GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames)
        {
            if (horseNames == null)
            {
                throw new ArgumentNullException(nameof(horseNames));
            }

            var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var name in horseNames)
            {
                foreach (var candidate in BuildHistoricalNameCandidates(name))
                {
                    candidates.Add(candidate);
                }
            }

            if (candidates.Count == 0)
            {
                return new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            }

            const string sql = @"
SELECT h.Name,
       COUNT(*) AS RaceCount
FROM RunnerResult rr
INNER JOIN Horse h ON h.HorseId = rr.HorseId
WHERE h.Name IN @Names
GROUP BY h.Name;";

            using var conn = OpenConnection();
            var rows = conn.Query(sql, new { Names = candidates.ToArray() });

            var results = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var row in rows)
            {
                var name = (string)row.Name;
                var count = (int)row.RaceCount;
                results[name] = count;
            }

            return results;
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

            return candidates;
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
            const string sql = "SELECT MIN(RaceDate) FROM Race WHERE RaceDate IS NOT NULL AND RaceDate <= CAST(GETDATE() AS date)";
            var earliest = conn.QuerySingleOrDefault<DateTime?>(sql);
            // When no races exist yet, start from tomorrow so we scrape backwards from today
            return earliest ?? DateTime.Today.AddDays(1);
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
            using var conn = OpenConnection();
            EnsureRaceScreenTableExists(conn);
            conn.Execute(sql, screen);
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
RaceType         NVARCHAR(128) NULL,
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
        public void ClearDayReportTables()
        {
            using var conn = OpenConnection();

            EnsureRaceScreenTableExists(conn);
            EnsureRunnerFlowTableExists(conn);
            EnsureUpcomingRaceTableExists(conn);

            using var transaction = conn.BeginTransaction();

            conn.Execute("DELETE FROM RunnerFlow;", transaction: transaction);
            conn.Execute("DELETE FROM RaceScreen;", transaction: transaction);
            conn.Execute("DELETE FROM UpcomingRaces;", transaction: transaction);

            transaction.Commit();
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
        AiOdds       FLOAT         NULL
    );
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
        
        public void InsertRunnerResults(IEnumerable<RunnerResult> results)
        {
            var list = results.ToList();
            foreach (var r in list)
            {
                StripBracketedText(r);
            }
            const string sql = @"
IF NOT EXISTS (SELECT 1 FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId)
    INSERT INTO RunnerResult(RaceId, HorseId, TrainerId, JockeyId, SaddleclothNumber, Draw, Age, WeightLbs, WeightText, FinishPos, OutcomeCode, DistanceBeatenText, DistanceBeatenLengths, SP_Fraction, SP_Decimal, FavTag, OpeningFraction, TouchedHighFraction, TouchedLowFraction, Comment)
    VALUES(@RaceId, @HorseId, @TrainerId, @JockeyId, @SaddleclothNumber, @Draw, @Age, @WeightLbs, @WeightText, @FinishPos, @OutcomeCode, @DistanceBeatenText, @DistanceBeatenLengths, @SP_Fraction, @SP_Decimal, @FavTag, @OpeningFraction, @TouchedHighFraction, @TouchedLowFraction, @Comment);";
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

        public int InsertRunnerResult(RunnerResult result)
        {
            StripBracketedText(result);
            const string sql = @"
IF EXISTS (SELECT 1 FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId)
    SELECT TOP 1 RunnerResultId FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId ORDER BY RunnerResultId;
ELSE
BEGIN
    INSERT INTO RunnerResult(RaceId, HorseId, TrainerId, JockeyId, SaddleclothNumber, Draw, Age, WeightLbs, WeightText, FinishPos, OutcomeCode, DistanceBeatenText, DistanceBeatenLengths, SP_Fraction, SP_Decimal, FavTag, OpeningFraction, TouchedHighFraction, TouchedLowFraction, Comment)
    VALUES(@RaceId, @HorseId, @TrainerId, @JockeyId, @SaddleclothNumber, @Draw, @Age, @WeightLbs, @WeightText, @FinishPos, @OutcomeCode, @DistanceBeatenText, @DistanceBeatenLengths, @SP_Fraction, @SP_Decimal, @FavTag, @OpeningFraction, @TouchedHighFraction, @TouchedLowFraction, @Comment);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, result);
        }
    }
}