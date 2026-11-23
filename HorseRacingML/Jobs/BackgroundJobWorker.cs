using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace HorseRacingML.Jobs
{
    public sealed class BackgroundJobWorker : BackgroundService
    {
        private readonly BackgroundJobQueue _queue;
        private readonly BackgroundJobStore _store;
        private readonly IServiceScopeFactory _scopeFactory;
        private readonly ILogger<BackgroundJobWorker> _logger;

        public BackgroundJobWorker(
            BackgroundJobQueue queue,
            BackgroundJobStore store,
            IServiceScopeFactory scopeFactory,
            ILogger<BackgroundJobWorker> logger)
        {
            _queue = queue;
            _store = store;
            _scopeFactory = scopeFactory;
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            await foreach (var request in _queue.Reader.ReadAllAsync(stoppingToken))
            {
                var record = _store.Get(request.Id);
                if (record is null)
                {
                    continue;
                }

                using var cts = CancellationTokenSource.CreateLinkedTokenSource(
                    record.CancellationTokenSource?.Token ?? CancellationToken.None,
                    stoppingToken);

                _store.MarkRunning(request.Id);
                _logger.LogInformation("Background job {JobId} ({Kind}) started.", request.Id, request.Kind);

                using var scope = _scopeFactory.CreateScope();
                var context = new JobExecutionContext(
                    record,
                    scope.ServiceProvider,
                    cts.Token,
                    (message, progress) => _store.Report(request.Id, message, progress),
                    request.Payload);

                try
                {
                    var handler = scope.ServiceProvider
                        .GetServices<IBackgroundJobHandler>()
                        .FirstOrDefault(item => item.Kind == request.Kind);

                    if (handler is null)
                    {
                        throw new InvalidOperationException(
                            $"No background job handler is registered for {request.Kind}.");
                    }

                    await handler.ExecuteAsync(context);
                    if (cts.Token.IsCancellationRequested)
                    {
                        _store.MarkCancelled(request.Id);
                    }
                    else
                    {
                        _store.MarkSucceeded(request.Id, record.Message ?? "Completed");
                    }
                }
                catch (OperationCanceledException)
                {
                    _store.MarkCancelled(request.Id);
                }
                catch (Exception) when (cts.Token.IsCancellationRequested)
                {
                    // A handler may wrap cancellation in an AggregateException
                    // (for example the scraper's parallel day workers).
                    _store.MarkCancelled(request.Id);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Background job {JobId} ({Kind}) failed.", request.Id, request.Kind);
                    _store.MarkFailed(request.Id, ex.Message);
                }
                finally
                {
                    _store.Release(request.Id);
                    _logger.LogInformation("Background job {JobId} ({Kind}) finished.", request.Id, request.Kind);
                }
            }
        }
    }
}
