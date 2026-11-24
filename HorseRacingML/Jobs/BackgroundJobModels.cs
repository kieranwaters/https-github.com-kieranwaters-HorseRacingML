using System;
using System.Collections.Concurrent;
using System.Threading;

namespace HorseRacingML.Jobs
{
    public enum BackgroundJobKind
    {
        ScrapeRaceResults,
        BackfillRunStyles,
        BackfillWeather,
        BackfillSectionals,
        BackfillBreeding,
        TrainAI,
        CustomTraining,
        SimulatorPrepare,
        AutomateBets
    }

    public enum BackgroundJobStatus
    {
        Queued,
        Running,
        Succeeded,
        Failed,
        Cancelled
    }

    public sealed class BackgroundJobRecord
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public BackgroundJobKind Kind { get; set; }
        public string DisplayName { get; set; } = string.Empty;
        public BackgroundJobStatus Status { get; set; } = BackgroundJobStatus.Queued;
        public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
        public DateTime? StartedUtc { get; set; }
        public DateTime? CompletedUtc { get; set; }
        public string? Message { get; set; }
        public int? Progress { get; set; }
        public string? Error { get; set; }
        public CancellationTokenSource? CancellationTokenSource { get; set; }
    }

    public sealed record BackgroundJobRequest(
        Guid Id,
        BackgroundJobKind Kind,
        string DisplayName,
        IReadOnlyDictionary<string, string?> Payload);

    public sealed class JobExecutionContext
    {
        public JobExecutionContext(
            BackgroundJobRecord record,
            IServiceProvider services,
            CancellationToken cancellationToken,
            Action<string, int?> report,
            IReadOnlyDictionary<string, string?> payload)
        {
            Record = record;
            Services = services;
            CancellationToken = cancellationToken;
            _report = report;
            Payload = payload;
        }

        public IReadOnlyDictionary<string, string?> Payload { get; }

        public BackgroundJobRecord Record { get; }
        public IServiceProvider Services { get; }
        public CancellationToken CancellationToken { get; }
        private readonly Action<string, int?> _report;

        public void Report(string message, int? progress = null)
        {
            _report(message, progress);
        }
    }
}
