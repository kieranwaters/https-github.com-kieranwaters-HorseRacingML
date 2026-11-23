using System;
using System.Threading.Tasks;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class BackfillWeatherJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.BackfillWeather;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            var weather = context.Services.GetRequiredService<WeatherBackfillService>();
            var configuration = context.Services.GetRequiredService<IConfiguration>();

            var defaultFrom = configuration.GetValue<DateTime?>("Weather:BackfillFrom")
                ?? new DateTime(2010, 1, 1);

            DateTime fromDate = defaultFrom;
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

            int? maxCourses = null;
            if (context.Payload.TryGetValue("maxCourses", out var maxText) &&
                int.TryParse(maxText, out var parsedMax) &&
                parsedMax > 0)
            {
                maxCourses = parsedMax;
            }

            var mode = WeatherBackfillMode.Normal;
            if (context.Payload.TryGetValue("mode", out var modeText))
            {
                if (string.Equals(modeText, nameof(WeatherBackfillMode.Refresh), StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(modeText, "force", StringComparison.OrdinalIgnoreCase))
                {
                    mode = WeatherBackfillMode.Refresh;
                }
                else if (string.Equals(modeText, nameof(WeatherBackfillMode.CacheOnly), StringComparison.OrdinalIgnoreCase) ||
                         string.Equals(modeText, "cache", StringComparison.OrdinalIgnoreCase))
                {
                    mode = WeatherBackfillMode.CacheOnly;
                }
            }

            var action = mode == WeatherBackfillMode.CacheOnly ? "Weather feature rebuild" : "Weather backfill";
            context.Report($"{action} started ({fromDate:yyyy-MM-dd} to {toDate:yyyy-MM-dd}, {mode})", 0);
            var summary = await Task.Run(
                () => weather.Backfill(
                    fromDate,
                    toDate,
                    maxCourses,
                    message => context.Report(message),
                    context.CancellationToken,
                    mode),
                context.CancellationToken);

            var message = $"{action} finished: {summary.RaceRowsUpserted} race rows, " +
                          $"{summary.WeatherDaysUpserted} weather days, {summary.CoursesSkipped} skipped";
            if (summary.Warnings.Count > 0)
            {
                message += $" | {summary.Warnings.Count} warning(s): " + string.Join(" | ", summary.Warnings);
            }

            context.Report(message, 100);
        }
    }
}
