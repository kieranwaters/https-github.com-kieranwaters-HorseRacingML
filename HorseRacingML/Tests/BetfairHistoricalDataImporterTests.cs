using System;
using System.IO;
using System.Linq;
using System.Text;
using Dapper;
using HorseRacingML.Data;
using HorseRacingML.Models;
using ICSharpCode.SharpZipLib.BZip2;
using Microsoft.Data.SqlClient;
using Xunit;

namespace HorseRacingML.Tests
{
    public class BetfairHistoricalDataImporterTests
    {
        private const string SyntheticStream = """
{"op":"mcm","clk":"1","pt":1704117540000,"mc":[{"id":"1.123","marketDefinition":{"bspMarket":true,"marketType":"WIN","numberOfWinners":1,"marketTime":"2024-01-01T14:00:00.000Z","suspendTime":"2024-01-01T14:00:00.000Z","bspReconciled":false,"inPlay":false,"venue":"Ascot","countryCode":"GB","timezone":"Europe/London","name":"14:00 Ascot","runners":[{"id":111,"name":"1. Alpha","status":"ACTIVE","sortPriority":1},{"id":222,"name":"2. Beta","status":"ACTIVE","sortPriority":2}]},"rc":[]}]}
{"op":"mcm","clk":"2","pt":1704117570000,"mc":[{"id":"1.123","rc":[{"id":111,"ltp":3.0,"trd":[[3.0,100]],"spn":3.2,"batb":[[0,3.0,50]],"batl":[[0,3.2,40]]},{"id":222,"ltp":4.0,"trd":[[4.0,200]],"spn":4.2}]}]}
{"op":"mcm","clk":"3","pt":1704117590000,"mc":[{"id":"1.123","rc":[{"id":111,"ltp":2.5,"trd":[[2.5,50]],"spn":2.7}]}]}
{"op":"mcm","clk":"4","pt":1704117720000,"mc":[{"id":"1.123","marketDefinition":{"bspReconciled":true,"inPlay":true,"runners":[{"id":111,"bsp":2.6},{"id":222,"bsp":4.1}]},"rc":[{"id":111,"ltp":1.5}]}]}
""";

        [Fact]
        public void ParsesPreOffPricesVolumeBspAndProjectedSp()
        {
            var repository = new BetfairHistoricalRepository(
                "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;");
            var importer = new BetfairHistoricalDataImporter(repository);

            using var stream = new MemoryStream(Encoding.UTF8.GetBytes(SyntheticStream));
            var parsed = importer.ParseMarkets(stream, "synthetic.json");

            Assert.Equal(4, parsed.LinesRead);
            var market = Assert.Single(parsed.Markets);
            Assert.Equal("1.123", market.MarketId);
            Assert.Equal("14:00 Ascot", market.MarketName);
            Assert.Equal("Ascot", market.Venue);
            Assert.True(market.BspReconciled);
            Assert.Equal(350m, market.TotalMatched);

            var alpha = Assert.Single(market.Runners, runner => runner.SelectionId == 111);
            Assert.Equal(3.0m, alpha.PreOffHigh);
            Assert.Equal(2.5m, alpha.PreOffLow);
            Assert.Equal(2.5m, alpha.LastTradedPrice);
            Assert.Equal(150m, alpha.TradedVolume);
            Assert.Equal(2.7m, alpha.ProjectedSp);
            Assert.Equal(2.6m, alpha.Bsp);
            Assert.Equal(50m, alpha.AvailableToBack);
            Assert.Equal(40m, alpha.AvailableToLay);

            var beta = Assert.Single(market.Runners, runner => runner.SelectionId == 222);
            Assert.Equal(4.0m, beta.PreOffHigh);
            Assert.Equal(4.0m, beta.PreOffLow);
            Assert.Equal(200m, beta.TradedVolume);
            Assert.Equal(4.2m, beta.ProjectedSp);
            Assert.Equal(4.1m, beta.Bsp);
        }

        [Fact]
        public void ParsesBzip2CompressedStream()
        {
            var repository = new BetfairHistoricalRepository(
                "Server=(local);Database=HorseRacingMLTest;Trusted_Connection=True;");
            var importer = new BetfairHistoricalDataImporter(repository);

            using var compressed = new MemoryStream();
            using (var bzipOut = new BZip2OutputStream(compressed) { IsStreamOwner = false })
            {
                var bytes = Encoding.UTF8.GetBytes(SyntheticStream);
                bzipOut.Write(bytes, 0, bytes.Length);
            }

            compressed.Position = 0;
            using var bzipIn = new BZip2InputStream(compressed);
            var parsed = importer.ParseMarkets(bzipIn, "synthetic.bz2");

            Assert.Single(parsed.Markets);
            Assert.Equal(2, parsed.Markets[0].Runners.Count);
        }

        [Trait("Category", "Integration")]
        [Fact]
        public void ImportMatchesRaceAndRunnerAndWritesBetfairColumns()
        {
            var baseConnectionString = Environment.GetEnvironmentVariable("HORSERACING_TEST_CONNECTION");
            if (string.IsNullOrWhiteSpace(baseConnectionString))
            {
                return;
            }

            var databaseName = $"HorseRacingML_BetfairTest_{Guid.NewGuid():N}";
            var builder = new SqlConnectionStringBuilder(baseConnectionString)
            {
                InitialCatalog = databaseName
            };
            string? tempFile = null;

            try
            {
                DatabaseBootstrapper.EnsureDatabase(builder.ConnectionString);

                var repository = new RacingRepository(builder.ConnectionString);
                var courseId = repository.InsertCourse(new Course { Name = "Ascot" });
                var raceId = repository.InsertRace(new Race
                {
                    CourseId = courseId,
                    RaceDate = new DateTime(2024, 1, 1),
                    ScheduledOff = new TimeSpan(14, 0, 0),
                    Title = "Test Race",
                    DistanceYards = 1760,
                    DistanceText = "1m",
                    Surface = "Turf",
                    Going = "Good",
                    RunnerCount = 2
                });

                var alphaId = repository.InsertHorse(new Horse { Name = "Alpha" });
                var betaId = repository.InsertHorse(new Horse { Name = "Beta" });
                repository.BulkInsertRunnerResults(new[]
                {
                    new RunnerResult { RaceId = raceId, HorseId = alphaId },
                    new RunnerResult { RaceId = raceId, HorseId = betaId }
                });

                tempFile = Path.Combine(
                    Path.GetTempPath(),
                    $"betfair-importer-{Guid.NewGuid():N}.json");
                File.WriteAllText(tempFile, SyntheticStream);

                var importer = new BetfairHistoricalDataImporter(
                    new BetfairHistoricalRepository(builder.ConnectionString));
                var summary = importer.ImportPath(tempFile);

                Assert.Equal(1, summary.MarketsMatched);
                Assert.Equal(2, summary.RunnersMatched);

                using var connection = new SqlConnection(builder.ConnectionString);
                connection.Open();
                var race = connection.QuerySingle<ImportedRaceRow>(
                    @"SELECT BetfairMarketId, BetfairTotalMatched, BetfairBspReconciled
                      FROM dbo.Race WHERE RaceId = @RaceId",
                    new { RaceId = raceId });
                Assert.Equal("1.123", race.BetfairMarketId);
                Assert.Equal(350m, race.BetfairTotalMatched);
                Assert.True(race.BetfairBspReconciled);

                var runners = connection.Query<ImportedRunnerRow>(
                    @"SELECT h.Name, rr.BetfairSP, rr.BetfairPreOffHigh, rr.BetfairPreOffLow,
                             rr.BetfairTradedVolume
                      FROM dbo.RunnerResult rr
                      JOIN dbo.Horse h ON h.HorseId = rr.HorseId
                      WHERE rr.RaceId = @RaceId
                      ORDER BY h.Name",
                    new { RaceId = raceId }).ToList();

                var alpha = Assert.Single(runners, runner => runner.Name == "Alpha");
                Assert.Equal(2.6m, alpha.BetfairSP);
                Assert.Equal(3.0m, alpha.BetfairPreOffHigh);
                Assert.Equal(2.5m, alpha.BetfairPreOffLow);
                Assert.Equal(150m, alpha.BetfairTradedVolume);

                var beta = Assert.Single(runners, runner => runner.Name == "Beta");
                Assert.Equal(4.1m, beta.BetfairSP);
                Assert.Equal(200m, beta.BetfairTradedVolume);
            }
            catch (SqlException)
            {
                // No SQL Server available or no permission to create a database.
            }
            finally
            {
                if (tempFile is not null)
                {
                    try { File.Delete(tempFile); } catch { }
                }

                DropDatabase(baseConnectionString, databaseName);
            }
        }

        private sealed class ImportedRaceRow
        {
            public string? BetfairMarketId { get; set; }
            public decimal? BetfairTotalMatched { get; set; }
            public bool? BetfairBspReconciled { get; set; }
        }

        private sealed class ImportedRunnerRow
        {
            public string Name { get; set; } = string.Empty;
            public decimal? BetfairSP { get; set; }
            public decimal? BetfairPreOffHigh { get; set; }
            public decimal? BetfairPreOffLow { get; set; }
            public decimal? BetfairTradedVolume { get; set; }
        }

        private static void DropDatabase(string? baseConnectionString, string databaseName)
        {
            if (string.IsNullOrWhiteSpace(baseConnectionString))
            {
                return;
            }

            try
            {
                var master = new SqlConnectionStringBuilder(baseConnectionString)
                {
                    InitialCatalog = "master"
                };
                using var connection = new SqlConnection(master.ConnectionString);
                connection.Open();
                using var command = connection.CreateCommand();
                command.CommandText = $@"
IF DB_ID(N'{databaseName}') IS NOT NULL
BEGIN
    ALTER DATABASE [{databaseName}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE [{databaseName}];
END;";
                command.ExecuteNonQuery();
            }
            catch
            {
                // Best-effort cleanup.
            }
        }

        [Theory]
        [InlineData("1. Alpha", "alpha")]
        [InlineData("1 Alpha", "alpha")]
        [InlineData("2. Alpha (IRE)", "alpha")]
        [InlineData("Alpha's Star", "alpha s star")]
        [InlineData("  ALPHA   STAR ", "alpha star")]
        public void NormalizesBetfairAndSportingLifeNames(string input, string expected)
        {
            Assert.Equal(expected, BetfairHistoricalDataImporter.NormalizeName(input));
        }
    }
}
