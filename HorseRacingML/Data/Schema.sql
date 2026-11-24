/*
    HorseRacingML database schema.

    The application used to rely on a SQL Server database that was created
    outside of source control, so the schema was never part of the repository.
    This script recreates the tables and indexes that the code in
    HorseRacingML/Data and HorseRacingML/MachineLearning expect.

    The script is idempotent: it only creates objects that are missing and never
    drops or alters existing data. It is also executed automatically on startup
    (see Data/DatabaseBootstrapper.cs), so a fresh clone can create its own
    local database.

    To run it by hand against a database:

        sqlcmd -S (localdb)\MSSQLLocalDB -d HorseRacingML -i Data\Schema.sql

    or open it in SQL Server Management Studio and execute it against the
    HorseRacingML database.
*/

SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

/* ------------------------------------------------------------------------- */
/* Reference data                                                            */
/* ------------------------------------------------------------------------- */

IF OBJECT_ID(N'dbo.Course', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Course
    (
        CourseId SMALLINT      IDENTITY(1,1) NOT NULL CONSTRAINT PK_Course PRIMARY KEY,
        Name     NVARCHAR(256) NOT NULL,
        Country  NVARCHAR(128) NULL,
        Latitude FLOAT         NULL,
        Longitude FLOAT        NULL,
        TimeZoneId NVARCHAR(64) NULL
    );
END;
GO

IF COL_LENGTH(N'dbo.Course', N'Latitude') IS NULL
    ALTER TABLE dbo.Course ADD Latitude FLOAT NULL;
IF COL_LENGTH(N'dbo.Course', N'Longitude') IS NULL
    ALTER TABLE dbo.Course ADD Longitude FLOAT NULL;
IF COL_LENGTH(N'dbo.Course', N'TimeZoneId') IS NULL
    ALTER TABLE dbo.Course ADD TimeZoneId NVARCHAR(64) NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Course_Name' AND object_id = OBJECT_ID(N'dbo.Course', N'U'))
BEGIN
    CREATE INDEX IX_Course_Name ON dbo.Course(Name);
END;
GO

IF OBJECT_ID(N'dbo.Trainer', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Trainer
    (
        TrainerId SMALLINT      IDENTITY(1,1) NOT NULL CONSTRAINT PK_Trainer PRIMARY KEY,
        Name      NVARCHAR(256) NOT NULL
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Trainer_Name' AND object_id = OBJECT_ID(N'dbo.Trainer', N'U'))
BEGIN
    CREATE INDEX IX_Trainer_Name ON dbo.Trainer(Name);
END;
GO

IF OBJECT_ID(N'dbo.Jockey', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Jockey
    (
        JockeyId SMALLINT      IDENTITY(1,1) NOT NULL CONSTRAINT PK_Jockey PRIMARY KEY,
        Name     NVARCHAR(256) NOT NULL
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Jockey_Name' AND object_id = OBJECT_ID(N'dbo.Jockey', N'U'))
BEGIN
    CREATE INDEX IX_Jockey_Name ON dbo.Jockey(Name);
END;
GO

IF OBJECT_ID(N'dbo.Horse', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Horse
    (
        HorseId INT           IDENTITY(1,1) NOT NULL CONSTRAINT PK_Horse PRIMARY KEY,
        Name    NVARCHAR(256) NOT NULL
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Horse_Name' AND object_id = OBJECT_ID(N'dbo.Horse', N'U'))
BEGIN
    CREATE INDEX IX_Horse_Name ON dbo.Horse(Name);
END;
GO

/* ------------------------------------------------------------------------- */
/* Race results                                                              */
/* ------------------------------------------------------------------------- */

IF OBJECT_ID(N'dbo.Race', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Race
    (
        RaceId          INT           IDENTITY(1,1) NOT NULL CONSTRAINT PK_Race PRIMARY KEY,
        CourseId        SMALLINT      NOT NULL,
        RaceDate        DATE          NOT NULL,
        ScheduledOff    TIME(0)       NULL,
        ActualOff       TIME(0)       NULL,
        Title           NVARCHAR(512) NULL,
        RaceType        NVARCHAR(128) NULL,
        Class           TINYINT       NULL,
        AgeRestriction  NVARCHAR(64)  NULL,
        Surface         NVARCHAR(64)  NULL,
        Going           NVARCHAR(30)  NULL,
        DistanceYards   SMALLINT      NULL,
        DistanceText    NVARCHAR(64)  NULL,
        RunnerCount     TINYINT       NULL,
        Status          NVARCHAR(64)  NULL,
        WinningTimeMs   INT           NULL,
        WinningTimeText NVARCHAR(64)  NULL,
        Purse           DECIMAL(12,2) NULL,
        Source          NVARCHAR(32)  NULL,
        SourceRaceId    NVARCHAR(128) NULL,
        BetfairMarketId NVARCHAR(32)  NULL,
        BetfairMarketName NVARCHAR(512) NULL,
        BetfairTotalMatched DECIMAL(18,2) NULL,
        BetfairBspReconciled BIT       NULL,
        IngestedUtc     DATETIME2(0)  NOT NULL CONSTRAINT DF_Race_IngestedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_Race_Course FOREIGN KEY (CourseId) REFERENCES dbo.Course(CourseId)
    );
END;
GO

IF COL_LENGTH(N'dbo.Race', N'Purse') IS NULL
    ALTER TABLE dbo.Race ADD Purse DECIMAL(12,2) NULL;
GO

IF COL_LENGTH(N'dbo.Race', N'BetfairMarketId') IS NULL
    ALTER TABLE dbo.Race ADD BetfairMarketId NVARCHAR(32) NULL;
IF COL_LENGTH(N'dbo.Race', N'BetfairMarketName') IS NULL
    ALTER TABLE dbo.Race ADD BetfairMarketName NVARCHAR(512) NULL;
IF COL_LENGTH(N'dbo.Race', N'BetfairTotalMatched') IS NULL
    ALTER TABLE dbo.Race ADD BetfairTotalMatched DECIMAL(18,2) NULL;
IF COL_LENGTH(N'dbo.Race', N'BetfairBspReconciled') IS NULL
    ALTER TABLE dbo.Race ADD BetfairBspReconciled BIT NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Race_RaceDate' AND object_id = OBJECT_ID(N'dbo.Race', N'U'))
BEGIN
    CREATE INDEX IX_Race_RaceDate ON dbo.Race(RaceDate);
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Race_CourseId_RaceDate' AND object_id = OBJECT_ID(N'dbo.Race', N'U'))
BEGIN
    CREATE INDEX IX_Race_CourseId_RaceDate ON dbo.Race(CourseId, RaceDate);
END;
GO

IF OBJECT_ID(N'dbo.RunnerResult', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RunnerResult
    (
        RunnerResultId        INT           IDENTITY(1,1) NOT NULL CONSTRAINT PK_RunnerResult PRIMARY KEY,
        RaceId                INT           NOT NULL,
        HorseId               INT           NOT NULL,
        TrainerId             SMALLINT      NULL,
        JockeyId              SMALLINT      NULL,
        SaddleclothNumber     TINYINT       NULL,
        Draw                  TINYINT       NULL,
        Age                   TINYINT       NULL,
        WeightLbs             TINYINT       NULL,
        WeightText            NVARCHAR(64)  NULL,
        OfficialRating        SMALLINT      NULL,
        Headgear              NVARCHAR(128) NULL,
        Medication            NVARCHAR(128) NULL,
        FinishPos             SMALLINT      NULL,
        OutcomeCode           NVARCHAR(32)  NULL,
        DistanceBeatenText    NVARCHAR(64)  NULL,
        DistanceBeatenLengths DECIMAL(9,2)  NULL,
        SP_Fraction           NVARCHAR(32)  NULL,
        SP_Decimal            DECIMAL(9,2)  NULL,
        FavTag                NVARCHAR(8)   NULL,
        OpeningFraction       NVARCHAR(32)  NULL,
        TouchedHighFraction   NVARCHAR(32)  NULL,
        TouchedLowFraction    NVARCHAR(32)  NULL,
        BetfairSelectionId    BIGINT        NULL,
        BetfairSP             DECIMAL(9,2)  NULL,
        BetfairPreOffHigh     DECIMAL(9,2)  NULL,
        BetfairPreOffLow      DECIMAL(9,2)  NULL,
        BetfairLastTradedPrice DECIMAL(9,2) NULL,
        MarketAnchorPrice     DECIMAL(9,2)  NULL,
        MarketAnchorCapturedUtc DATETIME2(0) NULL,
        BetfairTradedVolume   DECIMAL(18,2) NULL,
        BetfairProjectedSP    DECIMAL(9,2)  NULL,
        BetfairAvailableToBack DECIMAL(18,2) NULL,
        BetfairAvailableToLay DECIMAL(18,2) NULL,
        Comment               NVARCHAR(MAX) NULL,
        Source                NVARCHAR(32)  NULL,
        SourceRunnerId        NVARCHAR(128) NULL,
        IngestedUtc           DATETIME2(0)  NOT NULL CONSTRAINT DF_RunnerResult_IngestedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_RunnerResult_Race    FOREIGN KEY (RaceId)    REFERENCES dbo.Race(RaceId),
        CONSTRAINT FK_RunnerResult_Horse   FOREIGN KEY (HorseId)   REFERENCES dbo.Horse(HorseId),
        CONSTRAINT FK_RunnerResult_Trainer FOREIGN KEY (TrainerId) REFERENCES dbo.Trainer(TrainerId),
        CONSTRAINT FK_RunnerResult_Jockey  FOREIGN KEY (JockeyId)  REFERENCES dbo.Jockey(JockeyId)
    );
END;
GO

IF COL_LENGTH(N'dbo.RunnerResult', N'OfficialRating') IS NULL
    ALTER TABLE dbo.RunnerResult ADD OfficialRating SMALLINT NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'Headgear') IS NULL
    ALTER TABLE dbo.RunnerResult ADD Headgear NVARCHAR(128) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'Medication') IS NULL
    ALTER TABLE dbo.RunnerResult ADD Medication NVARCHAR(128) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairSelectionId') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairSelectionId BIGINT NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairSP') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairSP DECIMAL(9,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairPreOffHigh') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairPreOffHigh DECIMAL(9,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairPreOffLow') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairPreOffLow DECIMAL(9,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairLastTradedPrice') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairLastTradedPrice DECIMAL(9,2) NULL;
/* Timestamped pre-off market anchor: the last traded price at or before the
   configured cutoff (default five minutes before the off). Historical feature
   engineering uses this instead of the opening show so training and live
   scoring read the market at the same point. */
IF COL_LENGTH(N'dbo.RunnerResult', N'MarketAnchorPrice') IS NULL
    ALTER TABLE dbo.RunnerResult ADD MarketAnchorPrice DECIMAL(9,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'MarketAnchorCapturedUtc') IS NULL
    ALTER TABLE dbo.RunnerResult ADD MarketAnchorCapturedUtc DATETIME2(0) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairTradedVolume') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairTradedVolume DECIMAL(18,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairProjectedSP') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairProjectedSP DECIMAL(9,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairAvailableToBack') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairAvailableToBack DECIMAL(18,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairAvailableToLay') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairAvailableToLay DECIMAL(18,2) NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_RunnerResult_RaceId' AND object_id = OBJECT_ID(N'dbo.RunnerResult', N'U'))
BEGIN
    CREATE INDEX IX_RunnerResult_RaceId ON dbo.RunnerResult(RaceId);
END;
GO

/* Covering index for the chronological feature-building scan. It lets SQL Server
   satisfy the join and projection from one index instead of doing a key lookup
   for every runner row. */
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_RunnerResult_RaceId_Covering' AND object_id = OBJECT_ID(N'dbo.RunnerResult', N'U'))
BEGIN
    CREATE INDEX IX_RunnerResult_RaceId_Covering
        ON dbo.RunnerResult(RaceId)
        INCLUDE (HorseId, TrainerId, JockeyId, SaddleclothNumber, Draw, Age, WeightLbs,
                 FinishPos, OutcomeCode, DistanceBeatenText, SP_Decimal, OpeningFraction);
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_RunnerResult_HorseId' AND object_id = OBJECT_ID(N'dbo.RunnerResult', N'U'))
BEGIN
    CREATE INDEX IX_RunnerResult_HorseId ON dbo.RunnerResult(HorseId);
END;
GO

/* Provenance columns added by migration 002. */
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

/* ------------------------------------------------------------------------- */
/* Derived run style                                                         */
/* ------------------------------------------------------------------------- */

/* Parsed early-race position from RunnerResult.Comment. Stored separately so
   the chronological feature pass can join a narrow TINYINT/BIT row instead of
   reading the wide comment text for every runner. Version allows the parser
   rules to be re-run without rebuilding RunnerResult. */
IF OBJECT_ID(N'dbo.RunnerRunStyle', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RunnerRunStyle
    (
        RaceId        INT          NOT NULL,
        HorseId       INT          NOT NULL,
        EarlyPosition TINYINT      NOT NULL,
        HeldUp        BIT          NOT NULL,
        Keen          BIT          NOT NULL,
        SlowStart     BIT          NOT NULL,
        Version       TINYINT      NOT NULL,
        ClassifiedUtc DATETIME2(0) NOT NULL CONSTRAINT DF_RunnerRunStyle_ClassifiedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_RunnerRunStyle PRIMARY KEY (RaceId, HorseId),
        CONSTRAINT CK_RunnerRunStyle_EarlyPosition CHECK (EarlyPosition BETWEEN 0 AND 5)
    );
END;
GO

/* ------------------------------------------------------------------------- */
/* Sectionals and breeding (Racing TV / RaceiQ)                              */
/* ------------------------------------------------------------------------- */

/* Sire/dam/damsire as published by the Racing TV racecard. The model does
   not use the raw names; they key the rolling progeny statistics built by
   the Breeding feature block. */
IF OBJECT_ID(N'dbo.HorsePedigree', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.HorsePedigree
    (
        HorseId          INT           NOT NULL CONSTRAINT PK_HorsePedigree PRIMARY KEY,
        SireName         NVARCHAR(256) NULL,
        DamName          NVARCHAR(256) NULL,
        DamSireName      NVARCHAR(256) NULL,
        Sex              NVARCHAR(32)  NULL,
        Colour           NVARCHAR(64)  NULL,
        FoalingDate      DATE          NULL,
        SourceHorseId    INT           NULL,
        SourceSlug       NVARCHAR(256) NULL,
        Source           NVARCHAR(32)  NOT NULL CONSTRAINT DF_HorsePedigree_Source DEFAULT 'RacingTV',
        FetchedUtc       DATETIME2(0)  NOT NULL CONSTRAINT DF_HorsePedigree_FetchedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_HorsePedigree_Horse FOREIGN KEY (HorseId) REFERENCES dbo.Horse(HorseId)
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_HorsePedigree_SireName' AND object_id = OBJECT_ID(N'dbo.HorsePedigree', N'U'))
BEGIN
    CREATE INDEX IX_HorsePedigree_SireName ON dbo.HorsePedigree(SireName);
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_HorsePedigree_DamName' AND object_id = OBJECT_ID(N'dbo.HorsePedigree', N'U'))
BEGIN
    CREATE INDEX IX_HorsePedigree_DamName ON dbo.HorsePedigree(DamName);
END;
GO

/* One summary row per runner and race. The raw furlong splits stay in
   RawJson for audit/future features; the typed columns are what the
   chronological feature pass reads. */
IF OBJECT_ID(N'dbo.RunnerSectional', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RunnerSectional
    (
        RaceId                    INT           NOT NULL,
        HorseId                   INT           NOT NULL,
        FinishPos                 SMALLINT      NULL,
        ClothNumber               TINYINT       NULL,
        SectionCount              TINYINT       NULL,
        Last1FSeconds             FLOAT         NULL,
        Last2FSeconds             FLOAT         NULL,
        Last3FSeconds             FLOAT         NULL,
        Last1FPercentile          FLOAT         NULL,
        Last2FPercentile          FLOAT         NULL,
        Last3FPercentile          FLOAT         NULL,
        EarlyPositionNorm         FLOAT         NULL,
        ClosingPositionNorm       FLOAT         NULL,
        PaceStyle                 FLOAT         NULL,
        FinishSpeedPct            FLOAT         NULL,
        TimeIndexScore            FLOAT         NULL,
        MeetingAverageTimeIndex   FLOAT         NULL,
        WinTimeVsPar              FLOAT         NULL,
        SourceRaceId              NVARCHAR(64)  NULL,
        RawJson                   NVARCHAR(MAX) NULL,
        Source                    NVARCHAR(32)  NOT NULL CONSTRAINT DF_RunnerSectional_Source DEFAULT 'RacingTV',
        FetchedUtc                DATETIME2(0)  NOT NULL CONSTRAINT DF_RunnerSectional_FetchedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_RunnerSectional PRIMARY KEY (RaceId, HorseId),
        CONSTRAINT FK_RunnerSectional_Race FOREIGN KEY (RaceId) REFERENCES dbo.Race(RaceId),
        CONSTRAINT FK_RunnerSectional_Horse FOREIGN KEY (HorseId) REFERENCES dbo.Horse(HorseId)
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_RunnerSectional_HorseId' AND object_id = OBJECT_ID(N'dbo.RunnerSectional', N'U'))
BEGIN
    CREATE INDEX IX_RunnerSectional_HorseId ON dbo.RunnerSectional(HorseId, RaceId);
END;
GO

/* ------------------------------------------------------------------------- */
/* Machine learning                                                          */
/* ------------------------------------------------------------------------- */

IF OBJECT_ID(N'dbo.MLParameters', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MLParameters
    (
        Id                 INT           IDENTITY(1,1) NOT NULL CONSTRAINT PK_MLParameters PRIMARY KEY,
        RunDate            DATETIME2(0)  NOT NULL CONSTRAINT DF_MLParameters_RunDate DEFAULT SYSUTCDATETIME(),
        EarliestTrainingDate DATE       NULL,
        Units              INT           NOT NULL,
        Dropout            FLOAT         NOT NULL,
        Layers             INT           NOT NULL,
        LearningRate       FLOAT         NOT NULL,
        Epochs             INT           NOT NULL,
        BatchSize          INT           NOT NULL,
        Folds              INT           NOT NULL,
        TrainAccuracy      FLOAT         NULL,
        ValidationAccuracy FLOAT         NULL,
        ValidationLoss     FLOAT         NULL,
        TrainLoss          FLOAT         NULL,
        TrainBrier         FLOAT         NULL,
        Fold               INT           NULL,
        ValidationBrier    FLOAT         NULL,
        ModelArtifact      NVARCHAR(260) NULL,
        ModelHash          NVARCHAR(64)  NULL,
        FeatureSchemaHash  NVARCHAR(64)  NULL,
        IsActive           BIT           NOT NULL CONSTRAINT DF_MLParameters_IsActive DEFAULT 0
    );
END;
GO

IF COL_LENGTH(N'dbo.MLParameters', N'EarliestTrainingDate') IS NULL
BEGIN
    ALTER TABLE dbo.MLParameters ADD EarliestTrainingDate DATE NULL;
END;
GO

/* Evaluation/objective metadata added for the chronological dev/test split,
   race-softmax objective, market offset and seed-aware training. */
IF COL_LENGTH(N'dbo.MLParameters', N'Objective') IS NULL
    ALTER TABLE dbo.MLParameters ADD Objective NVARCHAR(32) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'Seed') IS NULL
    ALTER TABLE dbo.MLParameters ADD Seed INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'BestEpoch') IS NULL
    ALTER TABLE dbo.MLParameters ADD BestEpoch INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'MarketOffsetScale') IS NULL
    ALTER TABLE dbo.MLParameters ADD MarketOffsetScale FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'MarketOffsetMode') IS NULL
    ALTER TABLE dbo.MLParameters ADD MarketOffsetMode NVARCHAR(16) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'MarketBlind') IS NULL
    ALTER TABLE dbo.MLParameters ADD MarketBlind BIT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'BlendModelWeight') IS NULL
    ALTER TABLE dbo.MLParameters ADD BlendModelWeight FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'BlendMarketWeight') IS NULL
    ALTER TABLE dbo.MLParameters ADD BlendMarketWeight FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'NormalizationMode') IS NULL
    ALTER TABLE dbo.MLParameters ADD NormalizationMode NVARCHAR(64) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'FeatureBlocks') IS NULL
    ALTER TABLE dbo.MLParameters ADD FeatureBlocks NVARCHAR(128) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'RaceBatchSize') IS NULL
    ALTER TABLE dbo.MLParameters ADD RaceBatchSize INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TrainingStartDate') IS NULL
    ALTER TABLE dbo.MLParameters ADD TrainingStartDate DATE NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TrainingEndDate') IS NULL
    ALTER TABLE dbo.MLParameters ADD TrainingEndDate DATE NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevStartDate') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevStartDate DATE NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevEndDate') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevEndDate DATE NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestStartDate') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestStartDate DATE NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestEndDate') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestEndDate DATE NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TrainingRaceCount') IS NULL
    ALTER TABLE dbo.MLParameters ADD TrainingRaceCount INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevRaceCount') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevRaceCount INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestRaceCount') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestRaceCount INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TrainingRowCount') IS NULL
    ALTER TABLE dbo.MLParameters ADD TrainingRowCount INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevRowCount') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevRowCount INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestRowCount') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestRowCount INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevAccuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevAccuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevRaceLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevRaceLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevTop3Accuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevTop3Accuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevMarketRaceLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevMarketRaceLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevMarketTop1Accuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevMarketTop1Accuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevMarketTop3Accuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevMarketTop3Accuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevTemperature') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevTemperature FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevMarketLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevMarketLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevMarketBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevMarketBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevFairMarketLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevFairMarketLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevFairMarketBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevFairMarketBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevBlendLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevBlendLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevBlendBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevBlendBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'DevBlendRaceLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD DevBlendRaceLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestAccuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestAccuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestRaceLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestRaceLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestTop3Accuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestTop3Accuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestMarketRaceLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestMarketRaceLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestMarketTop1Accuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestMarketTop1Accuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestMarketTop3Accuracy') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestMarketTop3Accuracy FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestTemperature') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestTemperature FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestMarketLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestMarketLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestMarketBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestMarketBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestFairMarketLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestFairMarketLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestFairMarketBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestFairMarketBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestBlendLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestBlendLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestBlendBrier') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestBlendBrier FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'TestBlendRaceLogLoss') IS NULL
    ALTER TABLE dbo.MLParameters ADD TestBlendRaceLogLoss FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'WinRateAlpha') IS NULL
    ALTER TABLE dbo.MLParameters ADD WinRateAlpha FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'WinRateBeta') IS NULL
    ALTER TABLE dbo.MLParameters ADD WinRateBeta FLOAT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'SelectedOn') IS NULL
    ALTER TABLE dbo.MLParameters ADD SelectedOn NVARCHAR(16) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'ModelArtifact') IS NULL
    ALTER TABLE dbo.MLParameters ADD ModelArtifact NVARCHAR(260) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'ModelHash') IS NULL
    ALTER TABLE dbo.MLParameters ADD ModelHash NVARCHAR(64) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'FeatureSchemaHash') IS NULL
    ALTER TABLE dbo.MLParameters ADD FeatureSchemaHash NVARCHAR(64) NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'EnsembleSize') IS NULL
    ALTER TABLE dbo.MLParameters ADD EnsembleSize INT NULL;
IF COL_LENGTH(N'dbo.MLParameters', N'IsActive') IS NULL
    ALTER TABLE dbo.MLParameters ADD IsActive BIT NOT NULL CONSTRAINT DF_MLParameters_IsActive DEFAULT 0;
GO

/* ------------------------------------------------------------------------- */
/* Live market data (matching the tables the repository also creates lazily) */
/* ------------------------------------------------------------------------- */

IF OBJECT_ID(N'dbo.UpcomingRaces', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.UpcomingRaces
    (
        UpcomingRaceId     INT           IDENTITY(1,1) PRIMARY KEY,
        MarketId           NVARCHAR(32)  NOT NULL,
        RaceDate           DATE          NOT NULL,
        ScheduledOff       TIME(0)       NULL,
        VenueName          NVARCHAR(256) NULL,
        VenueCountry       NVARCHAR(128) NULL,
        Title              NVARCHAR(512) NULL,
        RaceDetails        NVARCHAR(MAX) NULL,
        RaceType           NVARCHAR(128) NULL,
        Class              TINYINT       NULL,
        AgeRestriction     NVARCHAR(64)  NULL,
        Surface            NVARCHAR(64)  NULL,
        Going              NVARCHAR(30)  NULL,
        DistanceYards      SMALLINT      NULL,
        DistanceText       NVARCHAR(64)  NULL,
        RunnerCount        TINYINT       NULL,
        BackBookPercentage DECIMAL(9,2)  NULL,
        LayBookPercentage  DECIMAL(9,2)  NULL,
        CreatedUtc         DATETIME2(0)  NOT NULL DEFAULT SYSUTCDATETIME(),
        LastUpdatedUtc     DATETIME2(0)  NOT NULL DEFAULT SYSUTCDATETIME()
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_UpcomingRaces_MarketId' AND object_id = OBJECT_ID(N'dbo.UpcomingRaces', N'U'))
BEGIN
    CREATE UNIQUE INDEX IX_UpcomingRaces_MarketId ON dbo.UpcomingRaces(MarketId);
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_UpcomingRaces_RaceDateVenue' AND object_id = OBJECT_ID(N'dbo.UpcomingRaces', N'U'))
BEGIN
    CREATE INDEX IX_UpcomingRaces_RaceDateVenue ON dbo.UpcomingRaces(RaceDate, VenueName);
END;
GO

IF OBJECT_ID(N'dbo.RaceScreen', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RaceScreen
    (
        RaceScreenId       BIGINT        IDENTITY(1,1) PRIMARY KEY,
        MarketId           NVARCHAR(32)  NULL,
        RaceDate           DATE          NULL,
        OffTime            TIME(0)       NULL,
        Title              NVARCHAR(512) NULL,
        VenueName          NVARCHAR(256) NULL,
        VenueCountry       NVARCHAR(128) NULL,
        EventDateText      NVARCHAR(128) NULL,
        RaceDetails        NVARCHAR(MAX) NULL,
        BackBookPercentage DECIMAL(9,2)  NULL,
        LayBookPercentage  DECIMAL(9,2)  NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.RunnerFlow', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RunnerFlow
    (
        RunnerFlowId BIGINT        IDENTITY(1,1) PRIMARY KEY,
        MarketId     NVARCHAR(32)  NULL,
        SelectionId  NVARCHAR(32)  NULL,
        ClothNumber  TINYINT       NULL,
        Draw         TINYINT       NULL,
        HorseName    NVARCHAR(256) NULL,
        JockeyName   NVARCHAR(256) NULL,
        BackPrice1   DECIMAL(9,2)  NULL,
        BackPrice2   DECIMAL(9,2)  NULL,
        BackPrice3   DECIMAL(9,2)  NULL,
        LayPrice1    DECIMAL(9,2)  NULL,
        LayPrice2    DECIMAL(9,2)  NULL,
        LayPrice3    DECIMAL(9,2)  NULL,
        CapturedUtc  DATETIME2(0)  NOT NULL CONSTRAINT DF_RunnerFlow_CapturedUtc DEFAULT SYSUTCDATETIME(),
        AiOdds       FLOAT         NULL
    );
END;
GO

IF COL_LENGTH(N'dbo.RunnerFlow', N'CapturedUtc') IS NULL
    ALTER TABLE dbo.RunnerFlow ADD CapturedUtc DATETIME2(0) NOT NULL
        CONSTRAINT DF_RunnerFlow_CapturedUtc DEFAULT SYSUTCDATETIME();
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_RunnerFlow_MarketSelectionPrice' AND object_id = OBJECT_ID(N'dbo.RunnerFlow', N'U'))
BEGIN
    CREATE INDEX IX_RunnerFlow_MarketSelectionPrice
        ON dbo.RunnerFlow(MarketId, SelectionId, RunnerFlowId)
        INCLUDE (HorseName, BackPrice1);
END;
GO

IF OBJECT_ID(N'dbo.PlacedBet', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.PlacedBet
    (
        PlacedBetId       BIGINT        IDENTITY(1,1) PRIMARY KEY,
        MarketId          NVARCHAR(32)  NOT NULL,
        SelectionId       NVARCHAR(32)  NULL,
        HorseName         NVARCHAR(256) NULL,
        RequestedOdds     DECIMAL(9,2)  NOT NULL,
        EffectiveOdds     DECIMAL(9,2)  NOT NULL,
        Stake             DECIMAL(18,2) NOT NULL,
        AiProbability     FLOAT         NOT NULL,
        MarketProbability FLOAT         NOT NULL,
        Differential      FLOAT         NOT NULL,
        PlacedUtc         DATETIME2(0)  NOT NULL
    );

    CREATE INDEX IX_PlacedBet_MarketId ON dbo.PlacedBet(MarketId);
END;
GO

/* ------------------------------------------------------------------------- */
/* Weather and going history                                                 */
/* ------------------------------------------------------------------------- */

IF OBJECT_ID(N'dbo.CourseWeatherDaily', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.CourseWeatherDaily
    (
        CourseId         SMALLINT     NOT NULL,
        WeatherDate      DATE         NOT NULL,
        PrecipitationMm  FLOAT        NULL,
        RainMm           FLOAT        NULL,
        TempMaxC         FLOAT        NULL,
        TempMinC         FLOAT        NULL,
        WindMaxKph       FLOAT        NULL,
        WindGustMaxKph   FLOAT        NULL,
        WindDirectionDominant FLOAT   NULL,
        RelativeHumidityMean  FLOAT   NULL,
        DewPointMean          FLOAT   NULL,
        PrecipitationHours    FLOAT   NULL,
        SnowfallCm            FLOAT   NULL,
        SoilMoisture0To7      FLOAT   NULL,
        SoilMoisture0To28     FLOAT   NULL,
        Et0Mm            FLOAT        NULL,
        Source           NVARCHAR(32) NULL,
        FetchedUtc       DATETIME2(0) NOT NULL
            CONSTRAINT DF_CourseWeatherDaily_FetchedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_CourseWeatherDaily PRIMARY KEY (CourseId, WeatherDate),
        CONSTRAINT FK_CourseWeatherDaily_Course FOREIGN KEY (CourseId)
            REFERENCES dbo.Course(CourseId)
    );
END;
GO

IF COL_LENGTH(N'dbo.CourseWeatherDaily', N'WindDirectionDominant') IS NULL
    ALTER TABLE dbo.CourseWeatherDaily ADD WindDirectionDominant FLOAT NULL;
IF COL_LENGTH(N'dbo.CourseWeatherDaily', N'RelativeHumidityMean') IS NULL
    ALTER TABLE dbo.CourseWeatherDaily ADD RelativeHumidityMean FLOAT NULL;
IF COL_LENGTH(N'dbo.CourseWeatherDaily', N'DewPointMean') IS NULL
    ALTER TABLE dbo.CourseWeatherDaily ADD DewPointMean FLOAT NULL;
IF COL_LENGTH(N'dbo.CourseWeatherDaily', N'PrecipitationHours') IS NULL
    ALTER TABLE dbo.CourseWeatherDaily ADD PrecipitationHours FLOAT NULL;
IF COL_LENGTH(N'dbo.CourseWeatherDaily', N'SnowfallCm') IS NULL
    ALTER TABLE dbo.CourseWeatherDaily ADD SnowfallCm FLOAT NULL;
IF COL_LENGTH(N'dbo.CourseWeatherDaily', N'SoilMoisture0To7') IS NULL
    ALTER TABLE dbo.CourseWeatherDaily ADD SoilMoisture0To7 FLOAT NULL;
IF COL_LENGTH(N'dbo.CourseWeatherDaily', N'SoilMoisture0To28') IS NULL
    ALTER TABLE dbo.CourseWeatherDaily ADD SoilMoisture0To28 FLOAT NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_CourseWeatherDaily_Date' AND object_id = OBJECT_ID(N'dbo.CourseWeatherDaily', N'U'))
BEGIN
    CREATE INDEX IX_CourseWeatherDaily_Date ON dbo.CourseWeatherDaily(WeatherDate);
END;
GO

IF OBJECT_ID(N'dbo.RaceWeather', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RaceWeather
    (
        RaceId                INT          NOT NULL,
        CourseId              SMALLINT     NOT NULL,
        WeatherDate           DATE         NOT NULL,
        RainPrev1dMm          FLOAT        NULL,
        RainPrev3dMm          FLOAT        NULL,
        RainPrev7dMm          FLOAT        NULL,
        RainPrev14dMm         FLOAT        NULL,
        RainRaceDayMm         FLOAT        NULL,
        WeatherTempMaxC       FLOAT        NULL,
        WeatherTempMinC       FLOAT        NULL,
        WeatherWindMaxKph     FLOAT        NULL,
        WeatherWindDirectionDegrees FLOAT NULL,
        WeatherHumidity       FLOAT        NULL,
        WeatherDewPointC      FLOAT        NULL,
        WeatherRainHoursPrev1 FLOAT        NULL,
        WeatherFrostDaysPrev7 INT          NULL,
        WeatherSoilMoisture0To7 FLOAT      NULL,
        WeatherSoilMoisture0To28 FLOAT     NULL,
        WeatherRainAnomaly14dMm FLOAT      NULL,
        WeatherSoilMoistureAnomaly FLOAT   NULL,
        WeatherEt0Prev7dMm    FLOAT        NULL,
        DaysSinceLastMeeting  INT          NULL,
        LastGoingAtCourse     NVARCHAR(30) NULL,
        WeatherGoingIndex     FLOAT        NULL,
        WeatherSoftRisk       FLOAT        NULL,
        Source                NVARCHAR(32) NULL,
        FetchedUtc            DATETIME2(0) NOT NULL
            CONSTRAINT DF_RaceWeather_FetchedUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_RaceWeather PRIMARY KEY (RaceId)
    );
END;
GO

IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherWindDirectionDegrees') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherWindDirectionDegrees FLOAT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherHumidity') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherHumidity FLOAT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherDewPointC') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherDewPointC FLOAT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherRainHoursPrev1') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherRainHoursPrev1 FLOAT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherFrostDaysPrev7') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherFrostDaysPrev7 INT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherSoilMoisture0To7') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherSoilMoisture0To7 FLOAT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherSoilMoisture0To28') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherSoilMoisture0To28 FLOAT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherRainAnomaly14dMm') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherRainAnomaly14dMm FLOAT NULL;
IF COL_LENGTH(N'dbo.RaceWeather', N'WeatherSoilMoistureAnomaly') IS NULL
    ALTER TABLE dbo.RaceWeather ADD WeatherSoilMoistureAnomaly FLOAT NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_RaceWeather_CourseId_WeatherDate' AND object_id = OBJECT_ID(N'dbo.RaceWeather', N'U'))
BEGIN
    CREATE INDEX IX_RaceWeather_CourseId_WeatherDate ON dbo.RaceWeather(CourseId, WeatherDate);
END;
GO

IF OBJECT_ID(N'dbo.GoingSnapshot', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.GoingSnapshot
    (
        GoingSnapshotId BIGINT        IDENTITY(1,1) NOT NULL CONSTRAINT PK_GoingSnapshot PRIMARY KEY,
        UpcomingRaceId  INT           NULL,
        MarketId        NVARCHAR(64)  NULL,
        VenueName       NVARCHAR(256) NULL,
        RaceDate        DATE          NULL,
        Going           NVARCHAR(30)  NULL,
        Surface         NVARCHAR(64)  NULL,
        CapturedUtc     DATETIME2(0)  NOT NULL
            CONSTRAINT DF_GoingSnapshot_CapturedUtc DEFAULT SYSUTCDATETIME()
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_GoingSnapshot_MarketId_CapturedUtc' AND object_id = OBJECT_ID(N'dbo.GoingSnapshot', N'U'))
BEGIN
    CREATE INDEX IX_GoingSnapshot_MarketId_CapturedUtc ON dbo.GoingSnapshot(MarketId, CapturedUtc);
END;
GO
