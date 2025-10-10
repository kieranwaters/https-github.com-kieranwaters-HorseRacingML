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
        public BetfairScrapeResult()
        {
            Races = new List<RaceDayReport>();
            Recommendations = new List<BetRecommendation>();
        }

        /// <summary>
        /// Gets the collection of race reports captured during the scrape.
        /// </summary>
        public IList<RaceDayReport> Races { get; }

        /// <summary>
        /// Gets the collection of bet recommendations produced during the scrape.
        /// </summary>
        public IList<BetRecommendation> Recommendations { get; }
    }
}