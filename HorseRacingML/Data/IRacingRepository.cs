using System;
using System.Collections.Generic;
using System.Threading;
using HorseRacingML.Models;

namespace HorseRacingML.Data
{
    public interface IRacingRepository
    {
        void InsertRaceScreen(RaceScreen screen);
        void InsertRunnerFlow(RunnerFlow flow);
        int UpsertUpcomingRace(UpcomingRace race);
        UpcomingRace? FindUpcomingRace(DateTime raceDate, string? raceTitle, string? venueName);
        UpcomingRace? GetUpcomingRaceByMarketId(string? marketId);

        // Defaults keep lightweight test/fake repositories source-compatible.
        bool HasPlacedBet(string marketId) => false;
        void InsertPlacedBet(PlacedBet bet) { }
        int BackfillRunStyles(int batchSize = 20_000, int? maxRows = null, Action<string>? log = null, CancellationToken cancellationToken = default) => 0;
    }
}