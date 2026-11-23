/*
    Betfair Exchange market data imported from Historical Data / Stream API
    files.

    The free BASIC package provides market definition + last traded price at
    one-minute intervals and each runner's BSP. The ADVANCED/PRO packages add
    traded volume and the available-to-back/lay ladder; those columns stay
    NULL when only BASIC data was imported.

    BSP is stored for evaluation and CLV work but is deliberately not used as
    a pre-off model feature: it is only known at the off. Liquidity/range
    columns are pre-off values and are safe to use as features.
*/

IF COL_LENGTH(N'dbo.Race', N'BetfairMarketId') IS NULL
    ALTER TABLE dbo.Race ADD BetfairMarketId NVARCHAR(32) NULL;
IF COL_LENGTH(N'dbo.Race', N'BetfairMarketName') IS NULL
    ALTER TABLE dbo.Race ADD BetfairMarketName NVARCHAR(512) NULL;
IF COL_LENGTH(N'dbo.Race', N'BetfairTotalMatched') IS NULL
    ALTER TABLE dbo.Race ADD BetfairTotalMatched DECIMAL(18,2) NULL;
IF COL_LENGTH(N'dbo.Race', N'BetfairBspReconciled') IS NULL
    ALTER TABLE dbo.Race ADD BetfairBspReconciled BIT NULL;
GO

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
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairTradedVolume') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairTradedVolume DECIMAL(18,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairProjectedSP') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairProjectedSP DECIMAL(9,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairAvailableToBack') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairAvailableToBack DECIMAL(18,2) NULL;
IF COL_LENGTH(N'dbo.RunnerResult', N'BetfairAvailableToLay') IS NULL
    ALTER TABLE dbo.RunnerResult ADD BetfairAvailableToLay DECIMAL(18,2) NULL;
GO

