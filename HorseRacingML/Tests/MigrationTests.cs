using System.Linq;
using HorseRacingML.Data;
using Xunit;

namespace HorseRacingML.Tests
{
    public class MigrationTests
    {
        [Fact]
        public void EmbeddedMigrationsAreDiscoverableAndHaveUniqueIncreasingVersions()
        {
            var resources = DatabaseBootstrapper.GetMigrationResourceNames();

            Assert.NotEmpty(resources);

            var versions = resources
                .Select(resource =>
                {
                    Assert.True(
                        DatabaseBootstrapper.TryGetMigrationVersion(resource, out var version),
                        $"Could not parse a migration version from '{resource}'.");
                    return version;
                })
                .ToList();

            Assert.Equal(versions.Distinct().Count(), versions.Count);
            Assert.Equal(versions.OrderBy(version => version), versions);
            Assert.Contains(1, versions);
            Assert.Contains(2, versions);
            Assert.Contains(3, versions);
            Assert.Contains(4, versions);
        }

        [Fact]
        public void MigrationVersionIsParsedFromTheFileNameNotTheNamespace()
        {
            Assert.True(
                DatabaseBootstrapper.TryGetMigrationVersion(
                    "HorseRacingML.Data.Migrations.042_Example.sql",
                    out var version));

            Assert.Equal(42, version);
            Assert.Equal("042_Example.sql", DatabaseBootstrapper.GetMigrationFileName(
                "HorseRacingML.Data.Migrations.042_Example.sql"));
        }
    }
}
