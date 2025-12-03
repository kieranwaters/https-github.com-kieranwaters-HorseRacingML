using HorseRacingML.Data;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Backwards-compatible shim that preserves the previous RaceDataScraper type name.
    /// </summary>
    public class RaceDataScraper : RaceResultsScraper
    {
        public RaceDataScraper(RacingRepository repo, ScrapingStatusService status, IConfiguration config)
            : base(repo, status, config)
        {
        }
    }
}