using System;
using System.Threading.Tasks;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class ScrapeRaceResultsJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.ScrapeRaceResults;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            var scraper = context.Services.GetRequiredService<RaceResultsScraper>();
            var status = context.Services.GetRequiredService<ScrapingStatusService>();

            status.Update($"Scraping started at {DateTime.Now:G}");
            context.Report("Scraping started", 0);
            await Task.Run(
                () => scraper.ScrapeFromTodayBackwards(context.CancellationToken),
                context.CancellationToken);
            status.Update($"Scraping completed at {DateTime.Now:G}");
            context.Report("Scraping completed", 100);
        }
    }
}
