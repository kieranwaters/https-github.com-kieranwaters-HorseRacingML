using System;
using System.Text.Json;
using System.Threading.Tasks;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class SimulatorPrepareJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.SimulatorPrepare;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            context.Payload.TryGetValue("requestJson", out var requestJson);
            var request = string.IsNullOrWhiteSpace(requestJson)
                ? new SimulatorRunRequest()
                : (JsonSerializer.Deserialize<SimulatorRunRequest>(requestJson) ?? new SimulatorRunRequest());

            var service = context.Services.GetRequiredService<SimulatorSnapshotService>();
            var (success, error, _) = await service.PrepareAndStoreAsync(
                request,
                context.Report,
                context.CancellationToken);

            if (!success)
            {
                throw new InvalidOperationException(error ?? "Simulator snapshot preparation failed.");
            }
        }
    }
}
