using System;
using System.Threading.Tasks;
using HorseRacingML.Data;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class BackfillRunStylesJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.BackfillRunStyles;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            var repository = context.Services.GetRequiredService<RacingRepository>();
            int? maxRows = null;
            if (context.Payload.TryGetValue("maxRows", out var maxRowsText) &&
                int.TryParse(maxRowsText, out var parsed) && parsed > 0)
            {
                maxRows = parsed;
            }

            context.Report("Run-style backfill started", 0);
            await Task.Run(() =>
            {
                var classified = repository.BackfillRunStyles(
                    maxRows: maxRows,
                    log: message => context.Report(message),
                    cancellationToken: context.CancellationToken);
                context.Report($"Run-style backfill finished: {classified} rows classified", 100);
            }, context.CancellationToken);
        }
    }
}
