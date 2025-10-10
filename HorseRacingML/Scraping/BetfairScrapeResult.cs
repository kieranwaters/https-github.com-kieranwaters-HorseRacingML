using System.Collections.Generic;
using HorseRacingML.Models;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Represents the aggregate output of a Betfair scraping pass, containing
    /// both the race day reports used for reporting and the bet recommendations
    /// generated during the session.
    /// </summary>
    public sealed class BetfairScrapeResult
    {
        /// <summary>
        /// Gets the collection of race reports captured during the scrape.
        /// </summary>
        public List<RaceDayReport> Races { get; } = new();

        /// <summary>
        /// Gets the collection of bet recommendations produced during the scrape.
        /// </summary>
        public List<BetRecommendation> Recommendations { get; } = new();
    }
}