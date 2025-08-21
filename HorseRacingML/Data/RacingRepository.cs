using System.Data;
using Dapper;
using Microsoft.Data.SqlClient;
using HorseRacingML.Models;
using Microsoft.Extensions.Hosting;
using System.Diagnostics;
using System.Net.Sockets;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;

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
        private IDbConnection OpenConnection() => new SqlConnection(_connectionString);

        public int InsertCourse(Course course)
        {
            StripBracketedText(course);
            const string sql = @"
IF EXISTS (SELECT CourseId FROM Course WHERE Name = @Name AND ISNULL(Country,'') = ISNULL(@Country,''))
    SELECT CourseId FROM Course WHERE Name = @Name AND ISNULL(Country,'') = ISNULL(@Country,'');
ELSE
BEGIN
    INSERT INTO Course(Name, Country) VALUES(@Name, @Country);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, course);
        }

        public int InsertRace(Race race)
        {
            StripBracketedText(race);
            const string sql = @"
IF EXISTS (SELECT RaceId FROM Race WHERE CourseId=@CourseId AND RaceDate=@RaceDate AND ScheduledOff=@ScheduledOff AND Title=@Title)
    SELECT RaceId FROM Race WHERE CourseId=@CourseId AND RaceDate=@RaceDate AND ScheduledOff=@ScheduledOff AND Title=@Title;
ELSE
BEGIN
    INSERT INTO Race(CourseId, RaceDate, ScheduledOff, ActualOff, Title, RaceType, Class, AgeRestriction, Surface, Going, DistanceYards, DistanceText, RunnerCount, Status, WinningTimeMs, WinningTimeText)
    VALUES(@CourseId, @RaceDate, @ScheduledOff, @ActualOff, @Title, @RaceType, @Class, @AgeRestriction, @Surface, @Going, @DistanceYards, @DistanceText, @RunnerCount, @Status, @WinningTimeMs, @WinningTimeText);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, race);
        }

        public int InsertTrainer(Trainer trainer)
        {
            StripBracketedText(trainer);
            const string sql = @"
IF EXISTS (SELECT TrainerId FROM Trainer WHERE Name=@Name)
    SELECT TrainerId FROM Trainer WHERE Name=@Name;
ELSE
BEGIN
    INSERT INTO Trainer(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, trainer);
        }

        public int InsertJockey(Jockey jockey)
        {
            StripBracketedText(jockey);
            const string sql = @"
IF EXISTS (SELECT JockeyId FROM Jockey WHERE Name=@Name)
    SELECT JockeyId FROM Jockey WHERE Name=@Name;
ELSE
BEGIN
    INSERT INTO Jockey(Name) VALUES(@Name);
    SELECT CAST(SCOPE_IDENTITY() as int);
END";
            using var conn = OpenConnection();
            return conn.QuerySingle<int>(sql, jockey);
        }

        public int InsertHorse(Horse horse)
        {
            StripBracketedText(horse);
            const string sql = @"
IF EXISTS (SELECT HorseId FROM Horse WHERE Name=@Name)
    SELECT HorseId FROM Horse WHERE Name=@Name;
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
IF EXISTS (SELECT RunnerResultId FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId)
    SELECT RunnerResultId FROM RunnerResult WHERE RaceId=@RaceId AND HorseId=@HorseId;
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