using System.Data;
using System.Text.RegularExpressions;
using Microsoft.Data.SqlClient;

namespace HorseRacingML.Data
{
    /// <summary>
    /// Creates the configured SQL Server database (if it does not exist yet) and
    /// applies the embedded schema so the application can run against a brand new
    /// local server without any manual setup.
    /// </summary>
    public static class DatabaseBootstrapper
    {
        public static void EnsureDatabase(string connectionString)
        {
            var builder = new SqlConnectionStringBuilder(connectionString);
            var databaseName = builder.InitialCatalog;
            if (string.IsNullOrWhiteSpace(databaseName))
            {
                throw new InvalidOperationException(
                    "The 'HorseRacingDb' connection string must include a database name (Initial Catalog or Database).");
            }

            CreateDatabaseIfMissing(connectionString, databaseName);

            using var connection = OpenWithRetry(connectionString);
            ExecuteScript(connection, LoadSchemaScript());
            ApplyMigrations(connection);
        }

        private static void CreateDatabaseIfMissing(string connectionString, string databaseName)
        {
            var masterBuilder = new SqlConnectionStringBuilder(connectionString) { InitialCatalog = "master" };
            using var master = new SqlConnection(masterBuilder.ConnectionString);
            master.Open();

            using (var exists = master.CreateCommand())
            {
                exists.CommandText = "SELECT DB_ID(@name);";
                exists.Parameters.Add("@name", SqlDbType.NVarChar, 128).Value = databaseName;

                var result = exists.ExecuteScalar();
                if (result is not null && result is not DBNull)
                {
                    return;
                }
            }

            var escapedName = databaseName.Replace("]", "]]");
            using var create = master.CreateCommand();
            create.CommandText = $"CREATE DATABASE [{escapedName}];";
            create.CommandTimeout = 120;
            create.ExecuteNonQuery();
        }

        private static SqlConnection OpenWithRetry(string connectionString)
        {
            SqlException? lastError = null;

            // A freshly created LocalDB database can take a moment to come online.
            for (var attempt = 0; attempt < 5; attempt++)
            {
                var connection = new SqlConnection(connectionString);
                try
                {
                    connection.Open();
                    return connection;
                }
                catch (SqlException ex)
                {
                    connection.Dispose();
                    lastError = ex;
                    Thread.Sleep(500);
                }
            }

            throw new InvalidOperationException(
                "Could not connect to the HorseRacingDb database. Check the 'HorseRacingDb' connection string in appsettings.json.",
                lastError);
        }

        private static string LoadSchemaScript()
        {
            var assembly = typeof(DatabaseBootstrapper).Assembly;
            var resourceName = assembly.GetManifestResourceNames()
                .FirstOrDefault(name => name.EndsWith("Data.Schema.sql", StringComparison.OrdinalIgnoreCase));

            if (resourceName is null)
            {
                throw new InvalidOperationException(
                    "The embedded database schema (Data/Schema.sql) could not be found.");
            }

            using var stream = assembly.GetManifestResourceStream(resourceName)
                ?? throw new InvalidOperationException(
                    "The embedded database schema (Data/Schema.sql) could not be opened.");
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        private static void ApplyMigrations(SqlConnection connection)
        {
            using (var create = connection.CreateCommand())
            {
                create.CommandText = @"
IF OBJECT_ID(N'dbo.SchemaMigrations', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.SchemaMigrations
    (
        Version    INT           NOT NULL CONSTRAINT PK_SchemaMigrations PRIMARY KEY,
        Name       NVARCHAR(260) NOT NULL,
        AppliedUtc DATETIME2(0)  NOT NULL CONSTRAINT DF_SchemaMigrations_AppliedUtc DEFAULT SYSUTCDATETIME()
    );
END;";
                create.CommandTimeout = 120;
                create.ExecuteNonQuery();
            }

            var applied = new HashSet<int>();
            using (var read = connection.CreateCommand())
            {
                read.CommandText = "SELECT Version FROM dbo.SchemaMigrations;";
                using var reader = read.ExecuteReader();
                while (reader.Read())
                {
                    applied.Add(reader.GetInt32(0));
                }
            }

            var assembly = typeof(DatabaseBootstrapper).Assembly;
            var migrations = assembly.GetManifestResourceNames()
                .Where(name => name.Contains(".Data.Migrations.", StringComparison.OrdinalIgnoreCase) &&
                               name.EndsWith(".sql", StringComparison.OrdinalIgnoreCase))
                .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
                .ToList();

            foreach (var resourceName in migrations)
            {
                if (!TryGetMigrationVersion(resourceName, out var version) || applied.Contains(version))
                {
                    continue;
                }

                var fileName = GetMigrationFileName(resourceName);

                using var tx = connection.BeginTransaction();
                try
                {
                    ExecuteScript(connection, LoadEmbeddedScript(resourceName), tx);

                    using (var record = connection.CreateCommand())
                    {
                        record.Transaction = tx;
                        record.CommandText = @"
INSERT INTO dbo.SchemaMigrations (Version, Name)
VALUES (@Version, @Name);";
                        record.Parameters.Add("@Version", SqlDbType.Int).Value = version;
                        record.Parameters.Add("@Name", SqlDbType.NVarChar, 260).Value = fileName;
                        record.ExecuteNonQuery();
                    }

                    tx.Commit();
                }
                catch
                {
                    tx.Rollback();
                    throw;
                }
            }
        }

        internal static string GetMigrationFileName(string resourceName)
        {
            const string marker = ".Data.Migrations.";
            var markerIndex = resourceName.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            return markerIndex >= 0
                ? resourceName[(markerIndex + marker.Length)..]
                : resourceName;
        }

        internal static bool TryGetMigrationVersion(string resourceName, out int version)
        {
            var fileName = GetMigrationFileName(resourceName);
            var versionText = fileName.Split('_', 2)[0];
            return int.TryParse(versionText, out version);
        }

        internal static IReadOnlyList<string> GetMigrationResourceNames()
        {
            return typeof(DatabaseBootstrapper).Assembly
                .GetManifestResourceNames()
                .Where(name => name.Contains(".Data.Migrations.", StringComparison.OrdinalIgnoreCase) &&
                               name.EndsWith(".sql", StringComparison.OrdinalIgnoreCase))
                .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        private static string LoadEmbeddedScript(string resourceName)
        {
            var assembly = typeof(DatabaseBootstrapper).Assembly;
            using var stream = assembly.GetManifestResourceStream(resourceName)
                ?? throw new InvalidOperationException($"The embedded database migration '{resourceName}' could not be opened.");
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        private static void ExecuteScript(
            SqlConnection connection,
            string script,
            SqlTransaction? transaction = null)
        {
            // sqlcmd-style GO separators are used in the script for readability;
            // SqlCommand only accepts a single batch at a time.
            foreach (var batch in Regex.Split(script, @"^\s*GO\s*$", RegexOptions.Multiline | RegexOptions.IgnoreCase))
            {
                if (string.IsNullOrWhiteSpace(batch))
                {
                    continue;
                }

                using var command = connection.CreateCommand();
                command.Transaction = transaction;
                command.CommandText = batch;
                command.CommandTimeout = 120;
                command.ExecuteNonQuery();
            }
        }
    }
}
