using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Dapper;
using Microsoft.Data.SqlClient;

namespace HorseRacingML.Data
{
    public sealed class BetfairCourseRow
    {
        public short CourseId { get; set; }
        public string Name { get; set; } = string.Empty;
    }

    public sealed class BetfairRaceCandidate
    {
        public int RaceId { get; set; }
        public TimeSpan? ScheduledOff { get; set; }
        public string? Title { get; set; }
    }

    public sealed class BetfairRunnerRow
    {
        public int RunnerResultId { get; set; }
        public int HorseId { get; set; }
        public string HorseName { get; set; } = string.Empty;
        public byte? SaddleclothNumber { get; set; }
    }

    public sealed class BetfairMarketUpdate
    {
        public int RaceId { get; set; }
        public string MarketId { get; set; } = string.Empty;
        public string? MarketName { get; set; }
        public decimal? TotalMatched { get; set; }
        public bool BspReconciled { get; set; }
        public List<BetfairRunnerUpdate> Runners { get; } = new();
    }

    public sealed class BetfairRunnerUpdate
    {
        public int RunnerResultId { get; set; }
        public int HorseId { get; set; }
        public long? SelectionId { get; set; }
        public decimal? Bsp { get; set; }
        public decimal? PreOffHigh { get; set; }
        public decimal? PreOffLow { get; set; }
        public decimal? LastTradedPrice { get; set; }
        public decimal? TradedVolume { get; set; }
        public decimal? ProjectedSp { get; set; }
        public decimal? AvailableToBack { get; set; }
        public decimal? AvailableToLay { get; set; }
    }

    /// <summary>
    /// Lookups and bulk updates used by the Betfair historical stream
    /// importer. Kept separate from RacingRepository so the importer can run
    /// without the scraper/trainer dependencies.
    /// </summary>
    public sealed class BetfairHistoricalRepository
    {
        private readonly string _connectionString;

        public BetfairHistoricalRepository(string connectionString)
        {
            _connectionString = string.IsNullOrWhiteSpace(connectionString)
                ? throw new ArgumentException(
                    "A database connection string is required.",
                    nameof(connectionString))
                : connectionString;
        }

        private SqlConnection OpenConnection()
        {
            var connection = new SqlConnection(_connectionString);
            connection.Open();
            return connection;
        }

        public IReadOnlyList<BetfairCourseRow> GetCourses()
        {
            using var connection = OpenConnection();
            return connection.Query<BetfairCourseRow>(
                "SELECT CourseId, Name FROM dbo.Course ORDER BY CourseId").ToList();
        }

        public IReadOnlyList<BetfairRaceCandidate> GetRaces(short courseId, DateTime date)
        {
            using var connection = OpenConnection();
            return connection.Query<BetfairRaceCandidate>(
                @"SELECT RaceId, ScheduledOff, Title
                  FROM dbo.Race
                  WHERE CourseId = @CourseId AND RaceDate = @RaceDate",
                new { CourseId = courseId, RaceDate = date.Date }).ToList();
        }

        public IReadOnlyList<BetfairRunnerRow> GetRunners(int raceId)
        {
            using var connection = OpenConnection();
            return connection.Query<BetfairRunnerRow>(
                @"SELECT rr.RunnerResultId, rr.HorseId, h.Name AS HorseName,
                         rr.SaddleclothNumber
                  FROM dbo.RunnerResult rr
                  JOIN dbo.Horse h ON h.HorseId = rr.HorseId
                  WHERE rr.RaceId = @RaceId
                  ORDER BY rr.RunnerResultId",
                new { RaceId = raceId }).ToList();
        }

        /// <summary>
        /// Updates the matched races and runners in one transaction. Re-importing
        /// the same file is idempotent; the most recent import wins.
        /// </summary>
        public int SaveBatch(IReadOnlyList<BetfairMarketUpdate> markets)
        {
            if (markets is null || markets.Count == 0)
            {
                return 0;
            }

            var raceTable = new DataTable();
            raceTable.Columns.Add("RaceId", typeof(int));
            raceTable.Columns.Add("MarketId", typeof(string));
            raceTable.Columns.Add("MarketName", typeof(string));
            raceTable.Columns.Add("TotalMatched", typeof(decimal));
            raceTable.Columns.Add("BspReconciled", typeof(bool));

            var runnerTable = new DataTable();
            runnerTable.Columns.Add("RunnerResultId", typeof(int));
            runnerTable.Columns.Add("HorseId", typeof(int));
            runnerTable.Columns.Add("SelectionId", typeof(long));
            runnerTable.Columns.Add("Bsp", typeof(decimal));
            runnerTable.Columns.Add("PreOffHigh", typeof(decimal));
            runnerTable.Columns.Add("PreOffLow", typeof(decimal));
            runnerTable.Columns.Add("LastTradedPrice", typeof(decimal));
            runnerTable.Columns.Add("TradedVolume", typeof(decimal));
            runnerTable.Columns.Add("ProjectedSp", typeof(decimal));
            runnerTable.Columns.Add("AvailableToBack", typeof(decimal));
            runnerTable.Columns.Add("AvailableToLay", typeof(decimal));

            foreach (var market in markets)
            {
                raceTable.Rows.Add(
                    market.RaceId,
                    market.MarketId,
                    (object?)market.MarketName ?? DBNull.Value,
                    (object?)market.TotalMatched ?? DBNull.Value,
                    market.BspReconciled);

                foreach (var runner in market.Runners)
                {
                    runnerTable.Rows.Add(
                        runner.RunnerResultId,
                        runner.HorseId,
                        (object?)runner.SelectionId ?? DBNull.Value,
                        (object?)runner.Bsp ?? DBNull.Value,
                        (object?)runner.PreOffHigh ?? DBNull.Value,
                        (object?)runner.PreOffLow ?? DBNull.Value,
                        (object?)runner.LastTradedPrice ?? DBNull.Value,
                        (object?)runner.TradedVolume ?? DBNull.Value,
                        (object?)runner.ProjectedSp ?? DBNull.Value,
                        (object?)runner.AvailableToBack ?? DBNull.Value,
                        (object?)runner.AvailableToLay ?? DBNull.Value);
                }
            }

            if (raceTable.Rows.Count == 0)
            {
                return 0;
            }

            using var connection = OpenConnection();
            using var transaction = connection.BeginTransaction();
            try
            {
                using (var command = connection.CreateCommand())
                {
                    command.Transaction = transaction;
                    command.CommandText = @"
CREATE TABLE #BetfairRace
(
    RaceId INT NOT NULL,
    MarketId NVARCHAR(32) NULL,
    MarketName NVARCHAR(512) NULL,
    TotalMatched DECIMAL(18,2) NULL,
    BspReconciled BIT NOT NULL
);
CREATE TABLE #BetfairRunner
(
    RunnerResultId INT NOT NULL,
    HorseId INT NOT NULL,
    SelectionId BIGINT NULL,
    Bsp DECIMAL(9,2) NULL,
    PreOffHigh DECIMAL(9,2) NULL,
    PreOffLow DECIMAL(9,2) NULL,
    LastTradedPrice DECIMAL(9,2) NULL,
    TradedVolume DECIMAL(18,2) NULL,
    ProjectedSp DECIMAL(9,2) NULL,
    AvailableToBack DECIMAL(18,2) NULL,
    AvailableToLay DECIMAL(18,2) NULL
);";
                    command.ExecuteNonQuery();
                }

                using (var bulk = new SqlBulkCopy(connection, SqlBulkCopyOptions.Default, transaction)
                {
                    DestinationTableName = "#BetfairRace",
                    BatchSize = 5000
                })
                {
                    foreach (DataColumn column in raceTable.Columns)
                    {
                        bulk.ColumnMappings.Add(column.ColumnName, column.ColumnName);
                    }
                    bulk.WriteToServer(raceTable);
                }

                if (runnerTable.Rows.Count > 0)
                {
                    using var bulk = new SqlBulkCopy(connection, SqlBulkCopyOptions.Default, transaction)
                    {
                        DestinationTableName = "#BetfairRunner",
                        BatchSize = 5000
                    };
                    foreach (DataColumn column in runnerTable.Columns)
                    {
                        bulk.ColumnMappings.Add(column.ColumnName, column.ColumnName);
                    }
                    bulk.WriteToServer(runnerTable);
                }

                using (var command = connection.CreateCommand())
                {
                    command.Transaction = transaction;
                    command.CommandText = @"
CREATE INDEX IX_BetfairRace ON #BetfairRace(RaceId);
CREATE INDEX IX_BetfairRunner ON #BetfairRunner(RunnerResultId);

UPDATE r
SET BetfairMarketId = COALESCE(b.MarketId, r.BetfairMarketId),
    BetfairMarketName = COALESCE(b.MarketName, r.BetfairMarketName),
    BetfairTotalMatched = COALESCE(b.TotalMatched, r.BetfairTotalMatched),
    BetfairBspReconciled = CASE WHEN b.BspReconciled = 1 THEN 1 ELSE r.BetfairBspReconciled END
FROM dbo.Race r
JOIN #BetfairRace b ON b.RaceId = r.RaceId;

UPDATE rr
SET BetfairSelectionId = COALESCE(b.SelectionId, rr.BetfairSelectionId),
    BetfairSP = COALESCE(b.Bsp, rr.BetfairSP),
    BetfairPreOffHigh = COALESCE(b.PreOffHigh, rr.BetfairPreOffHigh),
    BetfairPreOffLow = COALESCE(b.PreOffLow, rr.BetfairPreOffLow),
    BetfairLastTradedPrice = COALESCE(b.LastTradedPrice, rr.BetfairLastTradedPrice),
    BetfairTradedVolume = COALESCE(b.TradedVolume, rr.BetfairTradedVolume),
    BetfairProjectedSP = COALESCE(b.ProjectedSp, rr.BetfairProjectedSP),
    BetfairAvailableToBack = COALESCE(b.AvailableToBack, rr.BetfairAvailableToBack),
    BetfairAvailableToLay = COALESCE(b.AvailableToLay, rr.BetfairAvailableToLay)
FROM dbo.RunnerResult rr
JOIN #BetfairRunner b
  ON b.RunnerResultId = rr.RunnerResultId
 AND b.HorseId = rr.HorseId;

DROP TABLE #BetfairRunner;
DROP TABLE #BetfairRace;";
                    var updated = command.ExecuteNonQuery();
                    transaction.Commit();
                    return updated;
                }
            }
            catch
            {
                transaction.Rollback();
                throw;
            }
        }
    }
}
