using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace HorseRacingML.Jobs
{
    public sealed class BackgroundJobStore
    {
        private readonly ConcurrentDictionary<Guid, BackgroundJobRecord> _jobs = new();
        private readonly ConcurrentDictionary<BackgroundJobKind, Guid> _active = new();

        /// <summary>
        /// Creates a queued job and reserves its kind atomically. Returns null
        /// when a job of the same kind is already queued or running.
        /// </summary>
        public BackgroundJobRecord? TryCreate(BackgroundJobKind kind, string displayName)
        {
            var record = new BackgroundJobRecord
            {
                Kind = kind,
                DisplayName = displayName,
                Status = BackgroundJobStatus.Queued,
                CancellationTokenSource = new CancellationTokenSource()
            };

            if (!_active.TryAdd(kind, record.Id))
            {
                record.CancellationTokenSource.Dispose();
                return null;
            }

            if (!_jobs.TryAdd(record.Id, record))
            {
                _active.TryRemove(kind, out _);
                record.CancellationTokenSource.Dispose();
                return null;
            }

            return record;
        }

        public bool HasActiveOrQueued(BackgroundJobKind kind)
        {
            return _active.ContainsKey(kind);
        }

        public void MarkRunning(Guid id)
        {
            if (_jobs.TryGetValue(id, out var record))
            {
                record.Status = BackgroundJobStatus.Running;
                record.StartedUtc = DateTime.UtcNow;
                _active[record.Kind] = id;
            }
        }

        public void Report(Guid id, string? message, int? progress)
        {
            if (_jobs.TryGetValue(id, out var record))
            {
                record.Message = message;
                record.Progress = progress;
            }
        }

        public void MarkSucceeded(Guid id, string? message = null)
        {
            if (_jobs.TryGetValue(id, out var record))
            {
                record.Status = BackgroundJobStatus.Succeeded;
                record.Message = message ?? record.Message;
                record.CompletedUtc = DateTime.UtcNow;
            }
        }

        public void MarkFailed(Guid id, string? error)
        {
            if (_jobs.TryGetValue(id, out var record))
            {
                record.Status = BackgroundJobStatus.Failed;
                record.Error = error;
                record.CompletedUtc = DateTime.UtcNow;
            }
        }

        public void MarkCancelled(Guid id)
        {
            if (_jobs.TryGetValue(id, out var record))
            {
                record.Status = BackgroundJobStatus.Cancelled;
                record.CompletedUtc = DateTime.UtcNow;
            }
        }

        public void Release(Guid id)
        {
            if (_jobs.TryGetValue(id, out var record))
            {
                _active.TryRemove(record.Kind, out _);
                record.CancellationTokenSource?.Dispose();
                record.CancellationTokenSource = null;
            }
        }

        public void RequestCancel(Guid id)
        {
            if (!_jobs.TryGetValue(id, out var record) ||
                (record.Status != BackgroundJobStatus.Queued &&
                 record.Status != BackgroundJobStatus.Running))
            {
                return;
            }

            try
            {
                record.CancellationTokenSource?.Cancel();
            }
            catch (ObjectDisposedException)
            {
                // The job finished between the status check and the cancel call.
            }
        }

        public BackgroundJobRecord? Get(Guid id)
        {
            _jobs.TryGetValue(id, out var record);
            return record;
        }

        public IReadOnlyList<BackgroundJobRecord> List()
        {
            return _jobs.Values.OrderByDescending(job => job.CreatedUtc).ToList();
        }
    }
}
