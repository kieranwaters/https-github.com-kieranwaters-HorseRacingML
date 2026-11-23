-- Provenance lets a row be traced back to its source and makes re-imports
-- idempotent at the database level rather than only in application memory.
IF COL_LENGTH(N'dbo.Race', N'Source') IS NULL
    ALTER TABLE dbo.Race ADD Source NVARCHAR(32) NULL;
IF COL_LENGTH(N'dbo.Race', N'SourceRaceId') IS NULL
    ALTER TABLE dbo.Race ADD SourceRaceId NVARCHAR(128) NULL;
IF COL_LENGTH(N'dbo.Race', N'IngestedUtc') IS NULL
    ALTER TABLE dbo.Race ADD IngestedUtc DATETIME2(0) NOT NULL
        CONSTRAINT DF_Race_IngestedUtc DEFAULT SYSUTCDATETIME();

IF COL_LENGTH(N'dbo.RunnerResult', N'Source') IS NULL
    ALTER TABLE dbo.RunnerResult ADD Source NVARCHAR(32) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'SourceRunnerId') IS NULL
    ALTER TABLE dbo.RunnerResult ADD SourceRunnerId NVARCHAR(128) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'IngestedUtc') IS NULL
    ALTER TABLE dbo.RunnerResult ADD IngestedUtc DATETIME2(0) NOT NULL
        CONSTRAINT DF_RunnerResult_IngestedUtc DEFAULT SYSUTCDATETIME();
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Race_SourceRaceId' AND object_id = OBJECT_ID(N'dbo.Race', N'U'))
BEGIN
    CREATE INDEX IX_Race_SourceRaceId ON dbo.Race(Source, SourceRaceId);
END;
GO
