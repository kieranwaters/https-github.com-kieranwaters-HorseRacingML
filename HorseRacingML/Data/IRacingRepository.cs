using System;
using System.Collections.Generic;
using HorseRacingML.Models;

namespace HorseRacingML.Data
{
    public interface IRacingRepository
    {
        void InsertRaceScreen(RaceScreen screen);
        void InsertRunnerFlow(RunnerFlow flow);
        int UpsertUpcomingRace(UpcomingRace race);
        int? FindRaceId(DateTime raceDate, string? raceTitle, string? venueName);
        UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName);
        UpcomingRace? GetUpcomingRaceByMarketId(string? marketId);
    }
}