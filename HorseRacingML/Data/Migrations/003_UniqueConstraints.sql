/*
    Natural-key uniqueness for the reference and result tables.

    Existing databases may contain duplicates created by the old
    check-then-insert code. This migration merges duplicates onto the
    lowest id before creating the indexes, so the application can rely on
    the database instead of in-memory caches.

    Race is deliberately not given a natural-key unique index here: its
    identity is (CourseId, RaceDate, ScheduledOff, Title) and a bad merge is
    more damaging than a duplicate race. RunnerResult and the reference
    tables are the critical history inputs for the rolling features.
*/

-- Course duplicates.
;WITH duplicate AS
(
    SELECT CourseId, MIN(CourseId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Course
)
UPDATE r
SET CourseId = duplicate.KeepId
FROM dbo.Race r
INNER JOIN duplicate ON duplicate.CourseId = r.CourseId
WHERE duplicate.CourseId <> duplicate.KeepId;

;WITH duplicate AS
(
    SELECT CourseId, MIN(CourseId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Course
)
DELETE c
FROM dbo.Course c
INNER JOIN duplicate ON duplicate.CourseId = c.CourseId
WHERE duplicate.CourseId <> duplicate.KeepId;

-- Trainer duplicates.
;WITH duplicate AS
(
    SELECT TrainerId, MIN(TrainerId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Trainer
)
UPDATE rr
SET TrainerId = duplicate.KeepId
FROM dbo.RunnerResult rr
INNER JOIN duplicate ON duplicate.TrainerId = rr.TrainerId
WHERE duplicate.TrainerId <> duplicate.KeepId;

;WITH duplicate AS
(
    SELECT TrainerId, MIN(TrainerId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Trainer
)
DELETE t
FROM dbo.Trainer t
INNER JOIN duplicate ON duplicate.TrainerId = t.TrainerId
WHERE duplicate.TrainerId <> duplicate.KeepId;

-- Jockey duplicates.
;WITH duplicate AS
(
    SELECT JockeyId, MIN(JockeyId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Jockey
)
UPDATE rr
SET JockeyId = duplicate.KeepId
FROM dbo.RunnerResult rr
INNER JOIN duplicate ON duplicate.JockeyId = rr.JockeyId
WHERE duplicate.JockeyId <> duplicate.KeepId;

;WITH duplicate AS
(
    SELECT JockeyId, MIN(JockeyId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Jockey
)
DELETE j
FROM dbo.Jockey j
INNER JOIN duplicate ON duplicate.JockeyId = j.JockeyId
WHERE duplicate.JockeyId <> duplicate.KeepId;

-- Horse duplicates.
;WITH duplicate AS
(
    SELECT HorseId, MIN(HorseId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Horse
)
UPDATE rr
SET HorseId = duplicate.KeepId
FROM dbo.RunnerResult rr
INNER JOIN duplicate ON duplicate.HorseId = rr.HorseId
WHERE duplicate.HorseId <> duplicate.KeepId;

;WITH duplicate AS
(
    SELECT HorseId, MIN(HorseId) OVER (PARTITION BY Name) AS KeepId
    FROM dbo.Horse
)
DELETE h
FROM dbo.Horse h
INNER JOIN duplicate ON duplicate.HorseId = h.HorseId
WHERE duplicate.HorseId <> duplicate.KeepId;

-- RunnerResult duplicates by natural key.
;WITH duplicate AS
(
    SELECT RunnerResultId,
           ROW_NUMBER() OVER (PARTITION BY RaceId, HorseId ORDER BY RunnerResultId) AS RowNumber
    FROM dbo.RunnerResult
)
DELETE rr
FROM dbo.RunnerResult rr
INNER JOIN duplicate ON duplicate.RunnerResultId = rr.RunnerResultId
WHERE duplicate.RowNumber > 1;

-- UpcomingRaces duplicates by market.
;WITH duplicate AS
(
    SELECT UpcomingRaceId,
           ROW_NUMBER() OVER (PARTITION BY MarketId ORDER BY UpcomingRaceId) AS RowNumber
    FROM dbo.UpcomingRaces
)
DELETE u
FROM dbo.UpcomingRaces u
INNER JOIN duplicate ON duplicate.UpcomingRaceId = u.UpcomingRaceId
WHERE duplicate.RowNumber > 1;

-- PlacedBet is intentionally one row per market in this application.
;WITH duplicate AS
(
    SELECT PlacedBetId,
           ROW_NUMBER() OVER (PARTITION BY MarketId ORDER BY PlacedBetId) AS RowNumber
    FROM dbo.PlacedBet
)
DELETE p
FROM dbo.PlacedBet p
INNER JOIN duplicate ON duplicate.PlacedBetId = p.PlacedBetId
WHERE duplicate.RowNumber > 1;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_UpcomingRaces_MarketId' AND object_id = OBJECT_ID(N'dbo.UpcomingRaces', N'U'))
    CREATE UNIQUE INDEX IX_UpcomingRaces_MarketId ON dbo.UpcomingRaces(MarketId);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'UX_Course_Name' AND object_id = OBJECT_ID(N'dbo.Course', N'U'))
    CREATE UNIQUE INDEX UX_Course_Name ON dbo.Course(Name);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'UX_Trainer_Name' AND object_id = OBJECT_ID(N'dbo.Trainer', N'U'))
    CREATE UNIQUE INDEX UX_Trainer_Name ON dbo.Trainer(Name);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'UX_Jockey_Name' AND object_id = OBJECT_ID(N'dbo.Jockey', N'U'))
    CREATE UNIQUE INDEX UX_Jockey_Name ON dbo.Jockey(Name);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'UX_Horse_Name' AND object_id = OBJECT_ID(N'dbo.Horse', N'U'))
    CREATE UNIQUE INDEX UX_Horse_Name ON dbo.Horse(Name);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'UX_RunnerResult_RaceId_HorseId' AND object_id = OBJECT_ID(N'dbo.RunnerResult', N'U'))
    CREATE UNIQUE INDEX UX_RunnerResult_RaceId_HorseId ON dbo.RunnerResult(RaceId, HorseId);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'UX_PlacedBet_MarketId' AND object_id = OBJECT_ID(N'dbo.PlacedBet', N'U'))
    CREATE UNIQUE INDEX UX_PlacedBet_MarketId ON dbo.PlacedBet(MarketId);
GO
