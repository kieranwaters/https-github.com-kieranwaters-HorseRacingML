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
        int UpsertUpcomingRace(UpcomingRace race);
        UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName);
        UpcomingRace? GetUpcomingRaceByMarketId(string? marketId);
        int? GetHistoricalRaceCountByHorseName(string? horseName);
        IReadOnlyDictionary<string, int> GetHistoricalRaceCountsByHorseNames(IEnumerable<string> horseNames);
    }
}