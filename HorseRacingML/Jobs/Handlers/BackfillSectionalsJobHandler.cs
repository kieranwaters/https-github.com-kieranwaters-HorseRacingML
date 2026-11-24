using System;
using System.Threading.Tasks;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class BackfillSectionalsJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.BackfillSectionals;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            var backfill = context.Services.GetRequiredService<RacingTvBackfillService>();
            var configuration = context.Services.GetRequiredService<IConfiguration>();
            var (fromDate, toDate, maxDays, force) = RacingTvJobOptions.Parse(
                context,
                configuration.GetValue<DateTime?>("RacingTv:SectionalBackfillFrom")
                    ?? new DateTime(2023, 3, 1));

            context.Report($"Sectional backfill started ({fromDate:yyyy-MM-dd} to {toDate:yyyy-MM-dd})", 0);
            var summary = await backfill.BackfillAsync(
                fromDate,
                toDate,
                includeSectionals: true,
                includeBreeding: false,
                force,
                maxDays,
                message => context.Report(message),
                context.CancellationToken);

            context.Report(
                $"Sectional backfill finished: {summary.SectionalRowsUpserted} rows for " +
                $"{summary.SectionalRacesUpserted} races, {summary.RacesWithoutSourceData} without source data",
                100);
        }
    }

    public sealed class BackfillBreedingJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.BackfillBreeding;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            var backfill = context.Services.GetRequiredService<RacingTvBackfillService>();
            var configuration = context.Services.GetRequiredService<IConfiguration>();
            var (fromDate, toDate, maxDays, force) = RacingTvJobOptions.Parse(
                context,
                configuration.GetValue<DateTime?>("RacingTv:BreedingBackfillFrom")
                    ?? new DateTime(2009, 1, 1));

            context.Report($"Breeding backfill started ({fromDate:yyyy-MM-dd} to {toDate:yyyy-MM-dd})", 0);
            var summary = await backfill.BackfillAsync(
                fromDate,
                toDate,
                includeSectionals: false,
                includeBreeding: true,
                force,
                maxDays,
                message => context.Report(message),
                context.CancellationToken);

            context.Report(
                $"Breeding backfill finished: {summary.PedigreeRowsUpserted} pedigree rows",
                100);
        }
    }

    internal static class RacingTvJobOptions
    {
        public static (DateTime From, DateTime To, int? MaxDays, bool Force) Parse(
            JobExecutionContext context,
            DateTime defaultFrom)
        {
            DateTime fromDate = defaultFrom.Date;
            if (context.Payload.TryGetValue("from", out var fromText) &&
                DateTime.TryParse(fromText, out var parsedFrom))
            {
                fromDate = parsedFrom.Date;
            }

            DateTime toDate = DateTime.Today;
            if (context.Payload.TryGetValue("to", out var toText) &&
                DateTime.TryParse(toText, out var parsedTo))
            {
                toDate = parsedTo.Date;
            }

            int? maxDays = null;
            if (context.Payload.TryGetValue("maxDays", out var maxText) &&
                int.TryParse(maxText, out var parsedMax) &&
                parsedMax > 0)
            {
                maxDays = parsedMax;
            }

            bool force = context.Payload.TryGetValue("force", out var forceText) &&
                         bool.TryParse(forceText, out var parsedForce) &&
                         parsedForce;
            return (fromDate, toDate, maxDays, force);
        }
    }
}
