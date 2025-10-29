using HorseRacingML.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Globalization;

namespace HorseRacingML.ML
{
    public static class WinnerPredictor
    {
        private static readonly MLContext Ml = new MLContext(seed: 1);
        private static PredictionEngine<RunnerFeatures, RunnerPrediction>? _engine;
        private static readonly string ModelPath = Path.Combine(AppContext.BaseDirectory, "winnerModel.zip");
        private static readonly string MetricsPath = Path.Combine(AppContext.BaseDirectory, "winnerModel.metrics.json");
        private static readonly string StringMapPath = Path.Combine(AppContext.BaseDirectory, "string_maps.json");
        private class RunnerFeatures
        {
            public float Odds { get; set; }
            public float Weight { get; set; }
            public float WeightMissing { get; set; }
            public float Draw { get; set; }
            public float DrawMissing { get; set; }
            public float Age { get; set; }
            public float AgeMissing { get; set; }
            public float Going { get; set; }
            public float Surface { get; set; }
            public float Course { get; set; }
            public float Distance { get; set; }
            public float TimeOfDaySin { get; set; }
            public float TimeOfDayCos { get; set; }
            public float DistanceChangeFromLast { get; set; }
            public float DistanceRatioFromAverage { get; set; }
            public float CareerStarts { get; set; }
            public float LifetimeWinRate { get; set; }
            public float DrawBias { get; set; }
            public float DistanceBeatenLengths { get; set; }
            public float DistanceBeatenKnown { get; set; }
            public bool Label { get; set; }
        }
        private class ModelMetrics
        {
            public double Accuracy { get; set; }
            public double AreaUnderRocCurve { get; set; }
            public double LogLoss { get; set; }
        }
        private class RunnerPrediction
        {
            public bool PredictedLabel { get; set; }
            public float Probability { get; set; }
        }
    }
}