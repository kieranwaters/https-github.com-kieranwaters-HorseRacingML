using System;
using System.Threading.Tasks;
using HorseRacingML.Controllers;
using HorseRacingML.Scraping;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class AutomateBetsJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.AutomateBets;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            var controller = ActivatorUtilities.CreateInstance<HomeController>(context.Services);
            await Task.Run(() => controller.RunAutomateBetsWork(context), context.CancellationToken);
        }
    }
}
