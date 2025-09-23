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

namespace HorseRacingML.Data
{
    /// <summary>
    /// Simple repository for inserting parsed race data into SQL Server.
    /// </summary>
    public class RacingRepository
    {
        private readonly string _connectionString;
        private static readonly Regex BracketTextRegex =
            new Regex("\\s*\\([^\\)]*\\)|\\s*\\[[^\\]]*\\]", RegexOptions.Compiled);
        private const int GoingMaxLength = 30;
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
        public int? FindRaceId(DateTime raceDate, string? title, string? venueName)
        {
            const string sql = @"SELECT r.RaceId, r.Title, r.RaceDate, c.Name AS CourseName
FROM Race r
LEFT JOIN Course c ON r.CourseId = c.CourseId
WHERE CAST(r.RaceDate AS date) = @RaceDate";

            using var conn = OpenConnection();
            var candidates = conn.Query<(int RaceId, string Title, DateTime RaceDate, string? CourseName)>(sql, new { RaceDate = raceDate.Date }).ToList();
            if (candidates.Count == 0)
            {
                return null;
            }

            var normalizedTitle = NormalizeLookupKey(title);
            var normalizedVenue = NormalizeLookupKey(venueName);

            int? bestId = null;
            int bestScore = int.MaxValue;

            foreach (var candidate in candidates)
            {
                int score = 0;
                var candidateTitle = NormalizeLookupKey(candidate.Title);
                if (!string.IsNullOrEmpty(normalizedTitle))
                {
                    if (candidateTitle == normalizedTitle)
                    {
                        score -= 4;
                    }
                    else if (!string.IsNullOrEmpty(candidateTitle) && (candidateTitle.Contains(normalizedTitle) || normalizedTitle.Contains(candidateTitle)))
                    {
                        score -= 2;
                    }
                    else
                    {
                        score += 4;
                    }
                }

                var candidateVenue = NormalizeLookupKey(candidate.CourseName);
                if (!string.IsNullOrEmpty(normalizedVenue))
                {
                    if (candidateVenue == normalizedVenue)
                    {
                        score -= 2;
                    }
                    else if (!string.IsNullOrEmpty(candidateVenue) && (candidateVenue.Contains(normalizedVenue) || normalizedVenue.Contains(candidateVenue)))
                    {
                        score -= 1;
                    }
                    else
                    {
                        score += 2;
                    }
                }

                if (score < bestScore || (score == bestScore && (!bestId.HasValue || candidate.RaceId < bestId.Value)))
                {
                    bestScore = score;
                    bestId = candidate.RaceId;
                }
            }

            return bestId;
        }

        private static string NormalizeLookupKey(string? value)
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
            const string sql = @"
INSERT INTO RaceScreen(MarketId, RaceDate, OffTime, Title, VenueName, VenueCountry, EventDateText, RaceDetails, BackBookPercentage, LayBookPercentage)
VALUES(@MarketId, @RaceDate, @OffTime, @Title, @VenueName, @VenueCountry, @EventDateText, @RaceDetails, @BackBookPercentage, @LayBookPercentage);";
            using var conn = OpenConnection();
            conn.Execute(sql, screen);
        }

        public void InsertRunnerFlow(RunnerFlow flow)
        {
            const string sql = @"
INSERT INTO RunnerFlow(MarketId, SelectionId, ClothNumber, Draw, HorseName, JockeyName, BackPrice1, BackPrice2, BackPrice3, LayPrice1, LayPrice2, LayPrice3, AiOdds)
VALUES(@MarketId, @SelectionId, @ClothNumber, @Draw, @HorseName, @JockeyName, @BackPrice1, @BackPrice2, @BackPrice3, @LayPrice1, @LayPrice2, @LayPrice3, @AiOdds);";
            using var conn = OpenConnection();
            conn.Execute(sql, flow);
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

        public int InsertRace(Race race)
        {
            StripBracketedText(race);
            race.Going = NormalizeGoing(race.Going);
            const string sql = @"
IF EXISTS (SELECT 1 FROM Race WHERE CourseId=@CourseId AND RaceDate=@RaceDate AND ScheduledOff=@ScheduledOff AND Title=@Title)
    SELECT TOP 1 RaceId FROM Race WHERE CourseId=@CourseId AND RaceDate=@RaceDate AND ScheduledOff=@ScheduledOff AND Title=@Title ORDER BY RaceId;
ELSE
BEGIN
    INSERT INTO Race(CourseId, RaceDate, ScheduledOff, ActualOff, Title, RaceType, Class, AgeRestriction, Surface, Going, DistanceYards, DistanceText, RunnerCount, Status, WinningTimeMs, WinningTimeText)
    VALUES(@CourseId, @RaceDate, @ScheduledOff, @ActualOff, @Title, @RaceType, @Class, @AgeRestriction, @Surface, @Going, @DistanceYards, @DistanceText, @RunnerCount, @Status, @WinningTimeMs, @WinningTimeText);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, race);
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