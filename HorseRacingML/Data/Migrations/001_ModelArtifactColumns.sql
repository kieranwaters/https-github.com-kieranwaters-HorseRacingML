-- Versioned model artifacts. Safe to run repeatedly and on a fresh database.
IF COL_LENGTH(N'dbo.MLParameters', N'ModelArtifact') IS NULL
    ALTER TABLE dbo.MLParameters ADD ModelArtifact NVARCHAR(260) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'ModelHash') IS NULL
    ALTER TABLE dbo.MLParameters ADD ModelHash NVARCHAR(64) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'FeatureSchemaHash') IS NULL
    ALTER TABLE dbo.MLParameters ADD FeatureSchemaHash NVARCHAR(64) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'IsActive') IS NULL
    ALTER TABLE dbo.MLParameters ADD IsActive BIT NOT NULL
        CONSTRAINT DF_MLParameters_IsActive DEFAULT 0;
GO
