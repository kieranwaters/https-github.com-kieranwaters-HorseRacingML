using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Dapper;
using HorseRacingML.ML;
using Microsoft.Data.SqlClient;

namespace HorseRacingML.Data
{
    internal readonly record struct RunStyleInput(int RaceId, int HorseId, string? Comment);

    /// <summary>
    /// Persists the parsed run style next to <see cref="RunnerResult"/>. The
    /// comments are wide NVARCHAR(MAX) values, so the training query must not
    /// read them on every replay. This side table keeps the chronological
    /// feature pass narrow and lets the rules be re-run by bumping
    /// <see cref="RunStyleParser.Version"/>.
    ///
    /// The table is deliberately FK-free: it is derived and fully rebuildable,
    /// and avoiding FKs means creating/backfilling it cannot block on the
    /// schema locks held by a long-running training read of Race/RunnerResult.
    /// </summary>
    internal static class RunStyleStore
    {
        public static void EnsureTable(SqlConnection connection)
        {
            if (connection is null)
            {
                throw new ArgumentNullException(nameof(connection));
            }

            connection.Execute(@"
IF OBJECT_ID(N'dbo.RunnerRunStyle', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RunnerRunStyle
    (
        RaceId        INT          NOT NULL,
        HorseId       INT          NOT NULL,
        EarlyPosition TINYINT      NOT NULL,
        HeldUp        BIT          NOT NULL,
        Keen          BIT          NOT NULL,
        SlowStart     BIT          NOT NULL,
        Version       TINYINT      NOT NULL,
        ClassifiedUtc DATETIME2(0) NOT NULL CONSTRAINT DF_RunnerRunStyle_ClassifiedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_RunnerRunStyle PRIMARY KEY (RaceId, HorseId),
        CONSTRAINT CK_RunnerRunStyle_EarlyPosition CHECK (EarlyPosition BETWEEN 0 AND 5)
    );
END;", commandTimeout: 120);
        }

        public static int Upsert(SqlConnection connection, IEnumerable<RunStyleInput> inputs)
        {
            if (connection is null)
            {
                throw new ArgumentNullException(nameof(connection));
            }

            var records = inputs?
                .Where(input => input.RaceId > 0 && input.HorseId > 0 &&
                                !string.IsNullOrWhiteSpace(input.Comment))
                .GroupBy(input => (input.RaceId, input.HorseId))
                .Select(group => group.Last())
                .ToList();

            if (records is null || records.Count == 0)
            {
                return 0;
            }

            EnsureTable(connection);

            var table = new DataTable();
            table.Columns.Add("RaceId", typeof(int));
            table.Columns.Add("HorseId", typeof(int));
            table.Columns.Add("EarlyPosition", typeof(byte));
            table.Columns.Add("HeldUp", typeof(bool));
            table.Columns.Add("Keen", typeof(bool));
            table.Columns.Add("SlowStart", typeof(bool));
            table.Columns.Add("Version", typeof(byte));

            foreach (var record in records)
            {
                var info = RunStyleParser.Parse(record.Comment);
                table.Rows.Add(
                    record.RaceId,
                    record.HorseId,
                    (byte)info.Position,
                    info.HeldUp,
                    info.Keen,
                    info.SlowStart,
                    RunStyleParser.Version);
            }

            connection.Execute(@"
IF OBJECT_ID(N'tempdb..#RunStyleStage') IS NOT NULL
    DROP TABLE #RunStyleStage;

CREATE TABLE #RunStyleStage
(
    RaceId        INT     NOT NULL,
    HorseId       INT     NOT NULL,
    EarlyPosition TINYINT NOT NULL,
    HeldUp        BIT     NOT NULL,
    Keen          BIT     NOT NULL,
    SlowStart     BIT     NOT NULL,
    Version       TINYINT NOT NULL
);", commandTimeout: 600);

            using (var bulk = new SqlBulkCopy(connection)
            {
                DestinationTableName = "#RunStyleStage",
                BatchSize = 10_000,
                BulkCopyTimeout = 600
            })
            {
                foreach (DataColumn column in table.Columns)
                {
                    bulk.ColumnMappings.Add(column.ColumnName, column.ColumnName);
                }

                bulk.WriteToServer(table);
            }

            connection.Execute(@"
MERGE dbo.RunnerRunStyle AS target
USING (SELECT RaceId, HorseId, EarlyPosition, HeldUp, Keen, SlowStart, Version
       FROM #RunStyleStage) AS source
ON target.RaceId = source.RaceId AND target.HorseId = source.HorseId
WHEN MATCHED AND target.Version < source.Version THEN
    UPDATE SET
        EarlyPosition = source.EarlyPosition,
        HeldUp = source.HeldUp,
        Keen = source.Keen,
        SlowStart = source.SlowStart,
        Version = source.Version,
        ClassifiedUtc = SYSUTCDATETIME()
WHEN NOT MATCHED THEN
    INSERT (RaceId, HorseId, EarlyPosition, HeldUp, Keen, SlowStart, Version)
    VALUES (source.RaceId, source.HorseId, source.EarlyPosition, source.HeldUp,
            source.Keen, source.SlowStart, source.Version);

DROP TABLE #RunStyleStage;", commandTimeout: 600);

            return records.Count;
        }
    }
}
