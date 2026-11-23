/*
    Weather integration for the Going features.

    Course coordinates are required before a forecast can be requested.
    CourseWeatherDaily stores the raw daily values returned by Open-Meteo so a
    feature rebuild never needs to call the API again. RaceWeather is the
    denormalised, per-race feature row the model reads; it intentionally only
    contains weather observed before the race, never race-day totals that would
    leak the result of the race itself.

    GoingSnapshot records the going value that was visible at each scrape. The
    existing UpcomingRaces.Going column is an upsert and therefore overwrites
    earlier values, which makes it impossible to learn how going changes in
    the hours before the off.
*/

IF COL_LENGTH(N'dbo.Course', N'Latitude') IS NULL
    ALTER TABLE dbo.Course ADD Latitude FLOAT NULL;
IF COL_LENGTH(N'dbo.Course', N'Longitude') IS NULL
    ALTER TABLE dbo.Course ADD Longitude FLOAT NULL;
IF COL_LENGTH(N'dbo.Course', N'TimeZoneId') IS NULL
    ALTER TABLE dbo.Course ADD TimeZoneId NVARCHAR(64) NULL;
GO

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
