using System;
using System.Collections.Generic;
using HorseRacingML.Models;

namespace HorseRacingML.Data
{
    public interface IRacingRepository
    {
        void ClearDayReportTables();
        void InsertRaceScreen(RaceScreen screen);
        void InsertRunnerFlow(RunnerFlow flow);
        void InsertRunnerFlows(IEnumerable<RunnerFlow> flows);
    
        int UpsertUpcomingRace(UpcomingRace race);
        UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName);
        UpcomingRace? GetUpcomingRaceByMarketId(string? marketId);
        int? GetHistoricalRaceCountByHorseName(string? horseName);
        HistoricalRaceCountPrefetchResult GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames);
        int GetHistoricalWinCountByHorseName(string? horseName);
        HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName);
        int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate);
        IReadOnlyList<RunnerResult> GetLastSavedResults(string? raceTitle, DateTime? raceDate);
        int? GetWinningTimeMilliseconds(int raceId);
        int? GetRaceIdByRunnerResult(int runnerResultId);
        decimal? GetDistanceBeatenLengths(int runnerResultId);
        decimal? GetLastDistanceBeatenLengths(string? horseName, int? horseId, DateTime? beforeDate);
        int? GetLastWinningTimeMilliseconds(string? horseName, int? horseId, DateTime? beforeDate);
        byte? GetMostRecentRaceClass(string? horseName, int? horseId);
    }
}