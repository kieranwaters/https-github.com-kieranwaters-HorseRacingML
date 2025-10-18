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
        HorseDistanceStats? GetHorseDistanceStatsByHorseName(string? horseName);
        int? GetLastRaceDistance(string? horseName, int? horseId, DateTime? beforeDate);
        int? GetWinningTimeMilliseconds(int raceId);
        int? GetRaceIdByRunnerResult(int runnerResultId);
        decimal? GetDistanceBeatenLengths(int runnerResultId);
    }
}