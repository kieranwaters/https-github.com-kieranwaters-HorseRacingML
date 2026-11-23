using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using HorseRacingML.Jobs;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace HorseRacingML.Services
{
    /// <summary>
    /// Periodically runs the existing market scan during racing hours so
    /// RunnerFlow receives pre-off price snapshots. The scan does not place
    /// bets; it only records the flows and reports value opportunities.
    /// Disabled by default because it needs Betfair credentials and Chrome.
    /// </summary>
    public sealed class MarketSnapshotScheduler : BackgroundService
    {
        private readonly BackgroundJobQueue _queue;
        private readonly IConfiguration _configuration;
        private readonly ILogger<MarketSnapshotScheduler> _logger;

        public MarketSnapshotScheduler(
            BackgroundJobQueue queue,
            IConfiguration configuration,
            ILogger<MarketSnapshotScheduler> logger)
        {
            _queue = queue;
            _configuration = configuration;
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (!(_configuration.GetValue<bool?>("Betting:EnableSnapshotScheduler") ?? false))
            {
                return;
            }

            int intervalMinutes = Math.Clamp(
                _configuration.GetValue<int?>("Betting:SnapshotIntervalMinutes") ?? 10,
                1,
                120);
            int startHour = Math.Clamp(
                _configuration.GetValue<int?>("Betting:SnapshotStartHour") ?? 11,
                0,
                23);
            int endHour = Math.Clamp(
                _configuration.GetValue<int?>("Betting:SnapshotEndHour") ?? 22,
                startHour + 1,
                24);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    var now = DateTime.Now;
                    if (now.Hour >= startHour && now.Hour < endHour)
                    {
                        var queued = _queue.TryEnqueue(
                            BackgroundJobKind.AutomateBets,
                            "Market snapshot",
                            new Dictionary<string, string?>());
                        if (queued)
                        {
                            _logger.LogInformation(
                                "Queued market snapshot scan at {Time:HH:mm}.",
                                now);
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Market snapshot scheduler failed to queue a scan.");
                }

                try
                {
                    await Task.Delay(TimeSpan.FromMinutes(intervalMinutes), stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }
    }
}
