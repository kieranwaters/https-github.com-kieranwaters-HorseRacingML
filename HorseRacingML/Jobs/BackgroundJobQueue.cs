using System;
using System.Collections.Generic;
using System.Threading.Channels;

namespace HorseRacingML.Jobs
{
    public sealed class BackgroundJobQueue
    {
        private readonly Channel<BackgroundJobRequest> _channel =
            Channel.CreateUnbounded<BackgroundJobRequest>(
                new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        private readonly BackgroundJobStore _store;

        public BackgroundJobQueue(BackgroundJobStore store)
        {
            _store = store;
        }

        public bool TryEnqueue(
            BackgroundJobKind kind,
            string displayName,
            IReadOnlyDictionary<string, string?> payload)
        {
            var record = _store.TryCreate(kind, displayName);
            if (record is null)
            {
                return false;
            }

            var request = new BackgroundJobRequest(record.Id, kind, displayName, payload);
            if (!_channel.Writer.TryWrite(request))
            {
                _store.MarkFailed(record.Id, "The background job queue is unavailable.");
                _store.Release(record.Id);
                return false;
            }

            return true;
        }

        public ChannelReader<BackgroundJobRequest> Reader => _channel.Reader;
    }
}
