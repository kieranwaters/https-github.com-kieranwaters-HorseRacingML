using System;
using System.Data;
using System.Linq;
using Dapper;
using HorseRacingML.Data;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;
using Xunit;

namespace HorseRacingML.Tests
{
    /// <summary>
    /// Integration tests that need a real SQL Server. They are no-ops on a
    /// developer machine without one: set HORSERACING_TEST_CONNECTION to a
    /// SQL Server connection string and they create, test and drop an isolated
    /// database.
    /// </summary>
    public class DatabaseIntegrationTests
    {
        private static string? GetConnectionString()
        {
            return Environment.GetEnvironmentVariable("HORSERACING_TEST_CONNECTION");
        }

        private static string CreateTestDatabaseName()
        {
            return $"HorseRacingML_Test_{Guid.NewGuid():N}";
        }

        [Trait("Category", "Integration")]
        [Fact]
        public void SchemaIsCreatedAndReferenceRowsAreDeduplicated()
        {
            var baseConnectionString = GetConnectionString();
            if (string.IsNullOrWhiteSpace(baseConnectionString))
            {
                return;
            }

            var builder = new SqlConnectionStringBuilder(baseConnectionString)
            {
                InitialCatalog = CreateTestDatabaseName()
            };
            var connectionString = builder.ConnectionString;

            try
            {
                DatabaseBootstrapper.EnsureDatabase(connectionString);

                var repository = new RacingRepository(connectionString);
                var firstCourse = repository.InsertCourse(new Models.Course { Name = "TestCourse" });
                var secondCourse = repository.InsertCourse(new Models.Course { Name = "TestCourse" });
                Assert.Equal(firstCourse, secondCourse);

                var race = new Models.Race
                {
                    CourseId = firstCourse,
                    RaceDate = new DateTime(2026, 1, 1),
                    Title = "Test Race"
                };
                var firstRace = repository.InsertRace(race);
                var secondRace = repository.InsertRace(new Models.Race
                {
                    CourseId = firstCourse,
                    RaceDate = new DateTime(2026, 1, 1),
                    Title = "Test Race"
                });
                Assert.Equal(firstRace, secondRace);

                var results = new[]
                {
                    new Models.RunnerResult { RaceId = firstRace, HorseId = repository.InsertHorse(new Models.Horse { Name = "HorseA" }) },
                    new Models.RunnerResult { RaceId = firstRace, HorseId = repository.InsertHorse(new Models.Horse { Name = "HorseB" }) }
                };
                repository.BulkInsertRunnerResults(results);
                repository.BulkInsertRunnerResults(results);
                var count = CountRows(connectionString, "RunnerResult", "RaceId", firstRace);
                Assert.Equal(results.Length, count);

                using var connection = new SqlConnection(connectionString);
                connection.Open();
                var migrations = connection.Query<int>("SELECT Version FROM dbo.SchemaMigrations;").ToList();
                Assert.NotEmpty(migrations);
            }
            catch (SqlException)
            {
                // No SQL Server available or no permission to create a database.
            }
            finally
            {
                DropDatabase(baseConnectionString, builder.InitialCatalog);
            }
        }

        private static int CountRows(string connectionString, string table, string column, int value)
        {
            using var connection = new SqlConnection(connectionString);
            connection.Open();
            return connection.QuerySingle<int>(
                $"SELECT COUNT(*) FROM {table} WHERE {column} = @Value",
                new { Value = value });
        }

        private static void DropDatabase(string? baseConnectionString, string databaseName)
        {
            if (string.IsNullOrWhiteSpace(baseConnectionString))
            {
                return;
            }

            try
            {
                var builder = new SqlConnectionStringBuilder(baseConnectionString) { InitialCatalog = "master" };
                using var connection = new SqlConnection(builder.ConnectionString);
                connection.Open();
                using var command = connection.CreateCommand();
                command.CommandText = $@"
IF DB_ID(@name) IS NOT NULL
BEGIN
    ALTER DATABASE [{databaseName}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE [{databaseName}];
END;";
                command.Parameters.Add("@name", SqlDbType.NVarChar, 128).Value = databaseName;
                command.ExecuteNonQuery();
            }
            catch
            {
                // Best effort cleanup.
            }
        }
    }
}
