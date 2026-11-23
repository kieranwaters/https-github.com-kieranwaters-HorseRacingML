using System.Collections.Generic;
using System.Linq;
using HorseRacingML.Jobs;
using Xunit;

namespace HorseRacingML.Tests
{
    public class BackgroundJobTests
    {
        [Fact]
        public void DuplicateJobKindsAreRejectedUntilTheRunningJobIsReleased()
        {
            var store = new BackgroundJobStore();

            var first = store.TryCreate(BackgroundJobKind.ScrapeRaceResults, "first");
            Assert.NotNull(first);

            var duplicate = store.TryCreate(BackgroundJobKind.ScrapeRaceResults, "duplicate");
            Assert.Null(duplicate);

            store.MarkRunning(first!.Id);
            Assert.Null(store.TryCreate(BackgroundJobKind.ScrapeRaceResults, "still duplicate"));

            store.MarkSucceeded(first.Id, "done");
            store.Release(first.Id);

            var next = store.TryCreate(BackgroundJobKind.ScrapeRaceResults, "next");
            Assert.NotNull(next);
        }

        [Fact]
        public void ProgressAndCancellationAreVisibleOnTheJobRecord()
        {
            var store = new BackgroundJobStore();
            var job = store.TryCreate(BackgroundJobKind.TrainAI, "train");
            Assert.NotNull(job);

            store.MarkRunning(job!.Id);
            store.Report(job.Id, "encoding features", 40);
            Assert.Equal(BackgroundJobStatus.Running, job.Status);
            Assert.Equal("encoding features", job.Message);
            Assert.Equal(40, job.Progress);

            store.RequestCancel(job.Id);
            Assert.True(job.CancellationTokenSource!.IsCancellationRequested);

            store.MarkCancelled(job.Id);
            Assert.Equal(BackgroundJobStatus.Cancelled, job.Status);
            Assert.NotNull(job.CompletedUtc);
        }

        [Fact]
        public void QueueCarriesThePayloadAndOnlyAcceptsOneJobPerKind()
        {
            var store = new BackgroundJobStore();
            var queue = new BackgroundJobQueue(store);
            var payload = new Dictionary<string, string?> { ["maxRows"] = "1000" };

            Assert.True(queue.TryEnqueue(BackgroundJobKind.BackfillRunStyles, "backfill", payload));
            Assert.False(queue.TryEnqueue(BackgroundJobKind.BackfillRunStyles, "backfill again", payload));

            Assert.True(queue.Reader.TryRead(out var request));
            Assert.Equal(BackgroundJobKind.BackfillRunStyles, request.Kind);
            Assert.Equal("1000", request.Payload["maxRows"]);

            var jobs = store.List();
            Assert.Single(jobs);
            Assert.Equal(BackgroundJobStatus.Queued, jobs.Single().Status);
        }
    }
}
