using HorseRacingML.Data;
using HorseRacingML.Services;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Backwards-compatible shim that preserves the previous RaceDataScraper type name.
    /// </summary>
    public class RaceDataScraper : RaceResultsScraper
    {
        public RaceDataScraper(RacingRepository repo, ScrapingStatusService status)
            : base(repo, status)
        {
        }
    }
}