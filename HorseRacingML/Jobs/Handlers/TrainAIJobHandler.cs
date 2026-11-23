using System;
using System.Text.Json;
using System.Threading.Tasks;
using HorseRacingML.Controllers;
using HorseRacingML.Models;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class TrainAIJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.TrainAI;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            context.Payload.TryGetValue("batchJson", out var batchJson);
            var batch = string.IsNullOrWhiteSpace(batchJson)
                ? new MLParameterBatch()
                : (JsonSerializer.Deserialize<MLParameterBatch>(batchJson) ?? new MLParameterBatch());

            var controller = ActivatorUtilities.CreateInstance<HyperparameterController>(context.Services);
            context.Report("Training started", 0);
            await controller.RunTrainAI(batch, context.CancellationToken);
            context.Report("Training finished", 100);
        }
    }
}
