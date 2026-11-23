using System;
using System.Collections.Generic;
using System.Linq;
using Dapper;
using HorseRacingML.Data;
using HorseRacingML.ML;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace HorseRacingML.Tests
{
    public class RacingTvIntegrationTests
    {
        [Trait("Category", "Integration")]
        [Fact]
        public void SectionalAndPedigreeJoinsLoadIntoTheEncodedDataset()
        {
            var baseConnectionString = Environment.GetEnvironmentVariable("HORSERACING_TEST_CONNECTION");
            if (string.IsNullOrWhiteSpace(baseConnectionString))
            {
                return;
            }

            var builder = new SqlConnectionStringBuilder(baseConnectionString)
            {
                InitialCatalog = $"HorseRacingML_RtvTest_{Guid.NewGuid():N}"
            };
            var connectionString = builder.ConnectionString;

            try
            {
                DatabaseBootstrapper.EnsureDatabase(connectionString);
                SeedRacingTvData(connectionString);

                var configuration = new ConfigurationBuilder()
                    .AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["ConnectionStrings:HorseRacingDb"] = connectionString,
                        ["Training:FeatureBlocks"] = "Sectionals,Breeding"
                    })
                    .Build();

                var trainer = new HyperparameterTrainer(configuration);
                var dataset = trainer.LoadEncodedTrainingDataset(
                    includeIdentifiers: false,
                    minimumRaceDate: null,
                    requireMarketPrice: false);

                Assert.Equal(2, dataset.TrainingRaces.Count);
                Assert.Contains("SectionalHasHistory", dataset.FeatureKeys);
                Assert.Contains("PedigreeSireStarts", dataset.FeatureKeys);
                Assert.Contains("PedigreeKnown", dataset.FeatureKeys);
                Assert.Contains("SectionalAvgFinishSpeedPctLast5", dataset.FeatureKeys);
            }
            finally
            {
                DropDatabase(builder);
            }
        }

        private static void SeedRacingTvData(string connectionString)
        {
            using var connection = new SqlConnection(connectionString);
            connection.Open();
            using var transaction = connection.BeginTransaction();

            int courseId = connection.ExecuteScalar<int>(
                "INSERT INTO dbo.Course (Name) VALUES ('Test Course'); SELECT SCOPE_IDENTITY();",
                transaction: transaction);
            int firstRaceId = InsertRace(connection, transaction, courseId, new DateTime(2024, 1, 1), "12:00");
            int secondRaceId = InsertRace(connection, transaction, courseId, new DateTime(2024, 1, 2), "12:00");
            int firstHorseId = InsertHorse(connection, transaction, "Horse One");
            int secondHorseId = InsertHorse(connection, transaction, "Horse Two");

            InsertRunner(connection, transaction, firstRaceId, firstHorseId, 1, 1);
            InsertRunner(connection, transaction, firstRaceId, secondHorseId, 2, 2);
            InsertRunner(connection, transaction, secondRaceId, firstHorseId, 1, 1);
            InsertRunner(connection, transaction, secondRaceId, secondHorseId, 2, 2);

            connection.Execute(
                @"INSERT INTO dbo.HorsePedigree
                    (HorseId, SireName, DamName, DamSireName, Source, FetchedUtc)
                  VALUES
                    (@FirstHorseId, 'Galileo', 'Kind', 'Danehill', 'RacingTV', SYSUTCDATETIME()),
                    (@SecondHorseId, 'Galileo', 'Other Mare', 'Sadler''s Wells', 'RacingTV', SYSUTCDATETIME());",
                new { FirstHorseId = firstHorseId, SecondHorseId = secondHorseId },
                transaction);

            connection.Execute(
                @"INSERT INTO dbo.RunnerSectional
                    (RaceId, HorseId, FinishPos, SectionCount, Last3FSeconds, Last3FPercentile,
                     FinishSpeedPct, EarlyPositionNorm, ClosingPositionNorm, PaceStyle, Source, FetchedUtc)
                  VALUES
                    (@RaceId, @HorseId, 1, 6, 34.5, 0.9, 108.0, 0.3, 0.9, 0.6, 'RacingTV', SYSUTCDATETIME());",
                new { RaceId = firstRaceId, HorseId = firstHorseId },
                transaction);

            transaction.Commit();
        }

        private static int InsertRace(
            SqlConnection connection,
            SqlTransaction transaction,
            int courseId,
            DateTime date,
            string offTime)
        {
            return connection.ExecuteScalar<int>(
                @"INSERT INTO dbo.Race
                    (CourseId, RaceDate, ScheduledOff, Title, RaceType, Surface, Going,
                     DistanceYards, DistanceText, RunnerCount, Status)
                  VALUES
                    (@CourseId, @RaceDate, @ScheduledOff, 'Test Race', 'Flat', 'Turf', 'Good',
                     1760, '1m', 2, 'Result');
                  SELECT SCOPE_IDENTITY();",
                new
                {
                    CourseId = courseId,
                    RaceDate = date,
                    ScheduledOff = TimeSpan.Parse(offTime)
                },
                transaction);
        }

        private static int InsertHorse(SqlConnection connection, SqlTransaction transaction, string name)
        {
            return connection.ExecuteScalar<int>(
                "INSERT INTO dbo.Horse (Name) VALUES (@Name); SELECT SCOPE_IDENTITY();",
                new { Name = name },
                transaction);
        }

        private static void InsertRunner(
            SqlConnection connection,
            SqlTransaction transaction,
            int raceId,
            int horseId,
            int finishPos,
            int clothNumber)
        {
            connection.Execute(
                @"INSERT INTO dbo.RunnerResult
                    (RaceId, HorseId, SaddleclothNumber, FinishPos)
                  VALUES (@RaceId, @HorseId, @ClothNumber, @FinishPos);",
                new { RaceId = raceId, HorseId = horseId, ClothNumber = clothNumber, FinishPos = finishPos },
                transaction);
        }

        private static void DropDatabase(SqlConnectionStringBuilder builder)
        {
            var masterBuilder = new SqlConnectionStringBuilder(builder.ConnectionString)
            {
                InitialCatalog = "master"
            };
            using var connection = new SqlConnection(masterBuilder.ConnectionString);
            connection.Open();
            string databaseName = builder.InitialCatalog.Replace("]", "]]");
            connection.Execute(
                $@"IF DB_ID(N'{builder.InitialCatalog.Replace("'", "''")}') IS NOT NULL
                   BEGIN
                       ALTER DATABASE [{databaseName}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
                       DROP DATABASE [{databaseName}];
                   END;");
        }
    }
}
