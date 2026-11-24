using System;
using System.Collections.Generic;
using System.Linq;
using Dapper;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;

namespace HorseRacingML.Data
{
    /// <summary>
    /// Persistence for the two Racing TV derived tables. The schema is created
    /// by Data/Schema.sql during startup; these methods only read/write it.
    /// </summary>
    public sealed class RacingTvRepository
    {
        private readonly string _connectionString;

        public RacingTvRepository(string connectionString)
        {
            _connectionString = connectionString
                ?? throw new ArgumentNullException(nameof(connectionString));
        }

        public IReadOnlyList<RaceBackfillCandidate> GetRacesForDate(DateTime date)
        {
            using var connection = new SqlConnection(_connectionString);
            return connection.Query<RaceBackfillCandidate>(
                @"SELECT r.RaceId,
                         r.CourseId,
                         c.Name AS CourseName,
                         c.TimeZoneId,
                         r.ScheduledOff,
                         r.Title,
                         r.RunnerCount
                  FROM dbo.Race r
                  JOIN dbo.Course c ON c.CourseId = r.CourseId
                  WHERE r.RaceDate = @Date
                  ORDER BY r.ScheduledOff, r.RaceId",
                new { Date = date.Date }).ToList();
        }

        public IReadOnlyList<RaceRunnerReference> GetRunnersForRace(int raceId)
        {
            using var connection = new SqlConnection(_connectionString);
            return connection.Query<RaceRunnerReference>(
                @"SELECT rr.RaceId,
                         rr.HorseId,
                         h.Name AS HorseName,
                         rr.SaddleclothNumber,
                         rr.FinishPos
                  FROM dbo.RunnerResult rr
                  JOIN dbo.Horse h ON h.HorseId = rr.HorseId
                  WHERE rr.RaceId = @RaceId",
                new { RaceId = raceId }).ToList();
        }

        public IReadOnlyDictionary<int, (int Rows, int Finishers)> GetSectionalCoverage(DateTime date)
        {
            using var connection = new SqlConnection(_connectionString);
            var rows = connection.Query<(int RaceId, int Rows, int Finishers)>(
                @"SELECT rr.RaceId,
                         COUNT(DISTINCT rs.HorseId) AS [Rows],
                         COUNT(DISTINCT CASE WHEN rr.FinishPos IS NOT NULL THEN rr.HorseId END) AS Finishers
                  FROM dbo.RunnerResult rr
                  JOIN dbo.Race r ON r.RaceId = rr.RaceId
                  LEFT JOIN dbo.RunnerSectional rs ON rs.RaceId = rr.RaceId
                  WHERE r.RaceDate = @Date
                  GROUP BY rr.RaceId",
                new { Date = date.Date });

            return rows.ToDictionary(row => row.RaceId, row => (row.Rows, row.Finishers));
        }

        public IReadOnlyDictionary<int, (int Runners, int WithPedigree)> GetPedigreeCoverage(DateTime date)
        {
            using var connection = new SqlConnection(_connectionString);
            var rows = connection.Query<(int RaceId, int Runners, int WithPedigree)>(
                @"SELECT rr.RaceId,
                         COUNT(*) AS Runners,
                         COUNT(hp.HorseId) AS WithPedigree
                  FROM dbo.RunnerResult rr
                  JOIN dbo.Race r ON r.RaceId = rr.RaceId
                  LEFT JOIN dbo.HorsePedigree hp ON hp.HorseId = rr.HorseId
                  WHERE r.RaceDate = @Date
                  GROUP BY rr.RaceId",
                new { Date = date.Date });

            return rows.ToDictionary(row => row.RaceId, row => (row.Runners, row.WithPedigree));
        }

        public int UpsertHorsePedigree(IEnumerable<HorsePedigree> rows)
        {
            var list = (rows ?? Enumerable.Empty<HorsePedigree>())
                .Where(row => row is not null && row.HorseId > 0)
                .GroupBy(row => row.HorseId)
                .Select(group => group.Last())
                .ToList();
            if (list.Count == 0)
            {
                return 0;
            }

            using var connection = new SqlConnection(_connectionString);
            connection.Open();
            using var transaction = connection.BeginTransaction();
            const string sql = @"
MERGE dbo.HorsePedigree AS target
USING (SELECT @HorseId AS HorseId) AS source
ON target.HorseId = source.HorseId
WHEN MATCHED THEN UPDATE SET
    SireName = @SireName,
    DamName = @DamName,
    DamSireName = @DamSireName,
    Sex = @Sex,
    Colour = @Colour,
    FoalingDate = @FoalingDate,
    SourceHorseId = @SourceHorseId,
    SourceSlug = @SourceSlug,
    Source = @Source,
    FetchedUtc = @FetchedUtc
WHEN NOT MATCHED THEN INSERT
    (HorseId, SireName, DamName, DamSireName, Sex, Colour, FoalingDate,
     SourceHorseId, SourceSlug, Source, FetchedUtc)
VALUES
    (@HorseId, @SireName, @DamName, @DamSireName, @Sex, @Colour, @FoalingDate,
     @SourceHorseId, @SourceSlug, @Source, @FetchedUtc);";

            int affected = 0;
            foreach (var row in list)
            {
                affected += connection.Execute(sql, row, transaction);
            }

            transaction.Commit();
            return affected;
        }

        public int UpsertRunnerSectionals(IEnumerable<RunnerSectional> rows, bool overwrite = true)
        {
            var list = (rows ?? Enumerable.Empty<RunnerSectional>())
                .Where(row => row is not null && row.RaceId > 0 && row.HorseId > 0)
                .GroupBy(row => (row.RaceId, row.HorseId))
                .Select(group => group.Last())
                .ToList();
            if (list.Count == 0)
            {
                return 0;
            }

            using var connection = new SqlConnection(_connectionString);
            connection.Open();
            using var transaction = connection.BeginTransaction();
            string updateClause = overwrite
                ? @"WHEN MATCHED THEN UPDATE SET
    FinishPos = @FinishPos,
    ClothNumber = @ClothNumber,
    SectionCount = @SectionCount,
    Last1FSeconds = @Last1FSeconds,
    Last2FSeconds = @Last2FSeconds,
    Last3FSeconds = @Last3FSeconds,
    Last1FPercentile = @Last1FPercentile,
    Last2FPercentile = @Last2FPercentile,
    Last3FPercentile = @Last3FPercentile,
    EarlyPositionNorm = @EarlyPositionNorm,
    ClosingPositionNorm = @ClosingPositionNorm,
    PaceStyle = @PaceStyle,
    FinishSpeedPct = @FinishSpeedPct,
    TimeIndexScore = @TimeIndexScore,
    MeetingAverageTimeIndex = @MeetingAverageTimeIndex,
    WinTimeVsPar = @WinTimeVsPar,
    SourceRaceId = @SourceRaceId,
    RawJson = @RawJson,
    Source = @Source,
    FetchedUtc = @FetchedUtc
"
                : string.Empty;
            string sql = $@"
MERGE dbo.RunnerSectional AS target
USING (SELECT @RaceId AS RaceId, @HorseId AS HorseId) AS source
ON target.RaceId = source.RaceId AND target.HorseId = source.HorseId
{updateClause}WHEN NOT MATCHED THEN INSERT
    (RaceId, HorseId, FinishPos, ClothNumber, SectionCount,
     Last1FSeconds, Last2FSeconds, Last3FSeconds,
     Last1FPercentile, Last2FPercentile, Last3FPercentile,
     EarlyPositionNorm, ClosingPositionNorm, PaceStyle,
     FinishSpeedPct, TimeIndexScore, MeetingAverageTimeIndex, WinTimeVsPar,
     SourceRaceId, RawJson, Source, FetchedUtc)
VALUES
    (@RaceId, @HorseId, @FinishPos, @ClothNumber, @SectionCount,
     @Last1FSeconds, @Last2FSeconds, @Last3FSeconds,
     @Last1FPercentile, @Last2FPercentile, @Last3FPercentile,
     @EarlyPositionNorm, @ClosingPositionNorm, @PaceStyle,
     @FinishSpeedPct, @TimeIndexScore, @MeetingAverageTimeIndex, @WinTimeVsPar,
     @SourceRaceId, @RawJson, @Source, @FetchedUtc);";

            int affected = 0;
            foreach (var row in list)
            {
                affected += connection.Execute(sql, row, transaction);
            }

            transaction.Commit();
            return affected;
        }
    }
}
