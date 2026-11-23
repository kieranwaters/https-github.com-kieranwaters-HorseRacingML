using System;
using System.Text.Json;
using System.Threading.Tasks;
using HorseRacingML.Controllers;
using HorseRacingML.Models;
using Microsoft.Extensions.DependencyInjection;

namespace HorseRacingML.Jobs
{
    public sealed class CustomTrainingJobHandler : IBackgroundJobHandler
    {
        public BackgroundJobKind Kind => BackgroundJobKind.CustomTraining;

        public async Task ExecuteAsync(JobExecutionContext context)
        {
            context.Payload.TryGetValue("batchJson", out var batchJson);
            var batch = string.IsNullOrWhiteSpace(batchJson)
                ? new MLParameterBatch()
                : (JsonSerializer.Deserialize<MLParameterBatch>(batchJson) ?? new MLParameterBatch());

            var controller = ActivatorUtilities.CreateInstance<HyperparameterController>(context.Services);
            context.Report("Custom training started", 0);
            await controller.RunCustomTraining(batch, context.CancellationToken);
            context.Report("Custom training finished", 100);
        }
    }
}
