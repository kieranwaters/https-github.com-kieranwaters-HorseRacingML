using Dapper;
using HorseRacingML.Models;
using Microsoft.Data.SqlClient;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using PreparedDataset = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset;
using PreparedRace = HorseRacingML.ML.HyperparameterTrainer.TrainingDataset.PreparedDataset.PreparedRace;

namespace HorseRacingML.ML
{
    public partial class HyperparameterTrainer
    {
        private static readonly DateTime BaseDate = new DateTime(2005, 1, 1);
        private const int PastRaceCount = 5;
        // Maximum history to keep per horse; must cover largest window
        private const int HistoryLength = 120;
        // Windows (in races) for which performance metrics will be generated
        private static readonly int[] PerformanceWindows =
            { 1, 3, 5, 10, 15, 20, 25, 30, 50, 100 };
        private const int TrainerJockeyRecentStarts = 50;
        private const int TrainerJockeyRecentDays = 180;
        private const float TypicalRestDays = 30f;
        private const float MsPerLength = 200f;
        private static readonly Regex HorseNameBracketTextRegex =
            new Regex("\\s*\\([^\\)]*\\)|\\s*\\[[^\\]]*\\]", RegexOptions.Compiled);
        private static readonly Regex HorseNameWhitespaceRegex = new Regex("\\s+", RegexOptions.Compiled);

        private class RollingStat
        {
            public int Starts;
            public int Wins;
            public Queue<(DateTime date, bool win)> Recent = new();
        }

        private readonly record struct HistoryEntry(
            DateTime Date,
            float NormFinish,
            short? Finish,
            string Going,
            string Surface,
            int CourseId,
            string Bucket,
            int RaceClass,
            float Speed,
            float SpeedDiff,
            int Age,
            bool Won,
            float Rating,
            float Weight);

        private sealed class FeatureEngineeringState
        {
            private readonly HyperparameterTrainer _trainer;
            private readonly ISet<string>? _identifierKeys;
            private readonly Dictionary<int, List<HistoryEntry>> _horseHistory = new();
            private readonly Dictionary<int, RollingStat> _trainerStats = new();
            private readonly Dictionary<int, RollingStat> _jockeyStats = new();
            private readonly Dictionary<(int trainerId, int jockeyId), (int starts, int wins)> _trainerJockeyStats = new();
            private readonly Dictionary<(int trainerId, int jockeyId, string surface), (int starts, int wins)> _trainerJockeySurfaceStats = new();
            private readonly Dictionary<(int trainerId, int courseId), (int starts, int wins)> _trainerCourseStats = new();
            private readonly Dictionary<(int jockeyId, int courseId), (int starts, int wins)> _jockeyCourseStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _surfaceStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _goingStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _goingCourseStats = new();
            private readonly Dictionary<int, Dictionary<int, (int starts, int wins, float sumNorm, float lastNorm)>> _courseStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _ageStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _distanceBucketStats = new();
            private readonly Dictionary<(int horseId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)> _horseClassStats = new();
            private readonly Dictionary<(int trainerId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)> _trainerClassStats = new();
            private readonly Dictionary<(int jockeyId, int classVal), (int starts, int wins, float sumNorm, float lastNorm)> _jockeyClassStats = new();
            private readonly Dictionary<(int horseId, string going, string bucket), (int starts, int wins, float sumNorm, float lastNorm)> _goingDistanceStats = new();
            private readonly Dictionary<(int courseId, string bucket, int draw), (int starts, int wins)> _drawStats = new();
            private readonly Dictionary<(int courseId, string bucket), (int starts, int wins)> _drawBaselineStats = new();
            private readonly Dictionary<int, (double sum, int count)> _horseDistanceAll = new();
            private readonly Dictionary<int, (double sum, int count)> _horseDistanceWins = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _trainerSurfaceStats = new();
            private readonly Dictionary<int, int> _horseLastDistance = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _trainerGoingStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _trainerDistanceStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _jockeySurfaceStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _jockeyGoingStats = new();
            private readonly Dictionary<int, Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)>> _jockeyDistanceStats = new();
            private readonly Dictionary<(int jockeyId, string going, string bucket), (int starts, int wins, float sumNorm, float lastNorm)> _jockeyGoingDistanceStats = new();

            public FeatureEngineeringState(HyperparameterTrainer trainer, ISet<string>? identifierKeys = null)
            {
                _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
                _identifierKeys = identifierKeys;
            }

            public void ProcessRace(List<Dictionary<string, object?>> rows, bool includeRace, bool updateState)
            {
                if (rows is null)
                {
                    throw new ArgumentNullException(nameof(rows));
                }
                if (rows.Count == 0 || (!includeRace && !updateState))
                {
                    return;
                }
                rows.RemoveAll(row =>
                {
                    if (!PreparedDataset.TryGetRequiredInt32(row, "HorseId", out _))
                    {
                        return true;
                    }

                    if (!PreparedDataset.TryGetRequiredInt32(row, "RaceId", out _))
                    {
                        return true;
                    }

                    return false;
                });

                if (rows.Count == 0 || (!includeRace && !updateState))
                {
                    return;
                }
                foreach (var runnerRow in rows)
                {
                    bool distanceKnown = false;
                    float beatenLengths = 0f;
                    if (runnerRow.TryGetValue("DistanceBeatenLengths", out var lenObj) && lenObj != null)
                    {
                        beatenLengths = Convert.ToSingle(lenObj);
                        distanceKnown = true;
                    }

                    if (runnerRow.TryGetValue("DistanceBeatenText", out var txtObj) && txtObj is string txt)
                    {
                        var parsed = ParseDistanceBeaten(txt);
                        if (parsed.HasValue)
                        {
                            beatenLengths = parsed.Value;
                            distanceKnown = true;
                        }
                    }

                    runnerRow["DistanceBeatenKnown"] = distanceKnown;
                    runnerRow["DistanceBeatenLengths"] = beatenLengths;


                    var raceStat = ComputeRaceStats(rows);

                    foreach (var row in rows)
                    {
                        int horseId = PreparedDataset.GetRequiredInt32(row, "HorseId");
                        DateTime date = (DateTime)row["RaceDate"];
                        if (TryGetTimeOfDay(row, out var timeOfDay))
                        {
                            float minutes = (float)timeOfDay.TotalMinutes;
                            float timeAngle = 2f * MathF.PI * minutes / (24f * 60f);
                            row["TimeOfDaySin"] = MathF.Sin(timeAngle);
                            row["TimeOfDayCos"] = MathF.Cos(timeAngle);
                        }
                        else
                        {
                            row["TimeOfDaySin"] = 0f;
                            row["TimeOfDayCos"] = 0f;

                            int month = date.Month;
                            float monthAngle = 2f * MathF.PI * month / 12f;
                            row["RaceMonthSin"] = MathF.Sin(monthAngle);
                            row["RaceMonthCos"] = MathF.Cos(monthAngle);

                            int dayOfWeek = (int)date.DayOfWeek;
                            float dowAngle = 2f * MathF.PI * dayOfWeek / 7f;
                            row["RaceDayOfWeekSin"] = MathF.Sin(dowAngle);
                            row["RaceDayOfWeekCos"] = MathF.Cos(dowAngle);
                            row["IsWeekend"] = date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday;
                            int season = (month % 12) / 3;
                            float seasonAngle = 2f * MathF.PI * season / 4f;
                            row["SeasonSin"] = MathF.Sin(seasonAngle);
                            row["SeasonCos"] = MathF.Cos(seasonAngle);

                            row.Remove("RaceMonth");
                            row.Remove("RaceDayOfWeek");
                            row.Remove("Season");
                            row.Remove("ActualOff");
                            row.Remove("ScheduledOff");
                            row.Remove("RaceDate");
                            short? finish = row["FinishPos"] != null ? (short?)Convert.ToInt16(row["FinishPos"]) : null;
                            int raceId = PreparedDataset.GetRequiredInt32(row, "RaceId");
                            int runnerCount = row.TryGetValue("RunnerCount", out var runnerCountObj) &&
                                                PreparedDataset.TryConvertToInt32(runnerCountObj, out var runnerValue)
                                ? runnerValue
                            : raceStat.RunnerCount;
                            int? trainerId = row.TryGetValue("TrainerId", out var tObj) &&
                                              PreparedDataset.TryConvertToInt32(tObj, out var trainerValue)
                                ? trainerValue
                                : (int?)null;
                            int? jockeyId = row.TryGetValue("JockeyId", out var jObj) &&
                                             PreparedDataset.TryConvertToInt32(jObj, out var jockeyValue)
                                ? jockeyValue
                            : (int?)null;

                            bool drawMissing = row["Draw"] == null;
                            int draw = 0;
                            if (!drawMissing)
                            {
                                if (!PreparedDataset.TryConvertToInt32(row["Draw"], out draw))
                                {
                                    drawMissing = true;
                                }
                            }
                            row["DrawMissing"] = drawMissing;

                            float runnerSpeed = 0f;

                            bool weightMissing = row["WeightLbs"] == null;
                            float weight = weightMissing ? 0f : Convert.ToSingle(row["WeightLbs"]);
                            row["WeightMissing"] = weightMissing;

                            bool ratingMissing = !(row.TryGetValue("OfficialRating", out var ratingObj) && ratingObj != null);
                            float rating = ratingMissing ? raceStat.AvgRating : Convert.ToSingle(ratingObj);
                            row["RatingMissing"] = ratingMissing;
                            row["RelativeDraw"] = runnerCount > 0 ? (float)draw / runnerCount : 0f;
                            int saddlecloth = 0;
                            bool saddleclothMissing = !(row.TryGetValue("SaddleclothNumber", out var saddleclothObj) &&
                                                         PreparedDataset.TryConvertToInt32(saddleclothObj, out saddlecloth));
                            row["SaddleclothMissing"] = saddleclothMissing;
                            row["SaddleclothRelative"] = !saddleclothMissing && runnerCount > 0 ? (float)saddlecloth / runnerCount : 0f;
                            row["SaddleclothDiffFromMean"] = !saddleclothMissing && raceStat.HasSaddleclothStats
                                ? saddlecloth - raceStat.AvgSaddlecloth
                                : 0f;
                            row["WeightDiffFromMean"] = weight - raceStat.AvgWeight;
                            bool hasWeightStats = raceStat.HasWeightStats;
                            row["IsTopWeight"] = !weightMissing && hasWeightStats && Math.Abs(weight - raceStat.MaxWeight) < 0.001f;
                            row["IsBottomWeight"] = !weightMissing && hasWeightStats && Math.Abs(weight - raceStat.MinWeight) < 0.001f;
                            row["RatingDiffFromField"] = rating - raceStat.AvgRating;
                            row["FieldRatingStdDev"] = raceStat.StdRating;
                            row["PurseLevel"] = raceStat.TotalPurse;
                            int age = row.TryGetValue("Age", out var ageObj) && PreparedDataset.TryConvertToInt32(ageObj, out var ageValue)
                                ? ageValue
                                : 0;
                            row["AgeRelative"] = age - raceStat.AvgAge;
                            int classVal = row.TryGetValue("Class", out var classObj) && PreparedDataset.TryConvertToInt32(classObj, out var classValue)
                                ? classValue
                            : 0;
                            var horseClassKey = (horseId, classVal);
                            if (!_horseClassStats.TryGetValue(horseClassKey, out var horseClassStat))
                                horseClassStat = (0, 0, 0f, 0f);
                            row["ClassWinRate"] = _trainer.SmoothedWinRate(horseClassStat.wins, horseClassStat.starts);
                            row["ClassAvgNorm"] = horseClassStat.starts > 0
                                ? horseClassStat.sumNorm / horseClassStat.starts
                                : 0f;
                            row["LastClassNormPos"] = horseClassStat.lastNorm;

                            (int trainerId, int classVal) trainerClassKey = default;
                            (int starts, int wins, float sumNorm, float lastNorm) trainerClassStat = default;
                            bool hasTrainerClass = false;
                            if (trainerId.HasValue)
                            {
                                trainerClassKey = (trainerId.Value, classVal);
                                if (!_trainerClassStats.TryGetValue(trainerClassKey, out trainerClassStat))
                                    trainerClassStat = (0, 0, 0f, 0f);
                                row["TrainerClassWinRate"] = _trainer.SmoothedWinRate(trainerClassStat.wins, trainerClassStat.starts);
                                row["TrainerClassAvgNorm"] = trainerClassStat.starts > 0
                                    ? trainerClassStat.sumNorm / trainerClassStat.starts
                                    : 0f;
                                row["LastTrainerClassNormPos"] = trainerClassStat.lastNorm;
                                hasTrainerClass = true;
                            }
                            else
                            {
                                row["TrainerClassWinRate"] = 0f;
                                row["TrainerClassAvgNorm"] = 0f;
                                row["LastTrainerClassNormPos"] = 0f;
                            }

                            (int jockeyId, int classVal) jockeyClassKey = default;
                            (int starts, int wins, float sumNorm, float lastNorm) jockeyClassStat = default;
                            bool hasJockeyClass = false;
                            if (jockeyId.HasValue)
                            {
                                jockeyClassKey = (jockeyId.Value, classVal);
                                if (!_jockeyClassStats.TryGetValue(jockeyClassKey, out jockeyClassStat))
                                    jockeyClassStat = (0, 0, 0f, 0f);
                                row["JockeyClassWinRate"] = _trainer.SmoothedWinRate(jockeyClassStat.wins, jockeyClassStat.starts);
                                row["JockeyClassAvgNorm"] = jockeyClassStat.starts > 0
                                                    ? jockeyClassStat.sumNorm / jockeyClassStat.starts
                                                    : 0f;
                                row["LastJockeyClassNormPos"] = jockeyClassStat.lastNorm;
                                hasJockeyClass = true;
                            }
                            else
                            {
                                row["JockeyClassWinRate"] = 0f;
                                row["JockeyClassAvgNorm"] = 0f;
                                row["LastJockeyClassNormPos"] = 0f;
                            }
                            if (!_horseHistory.TryGetValue(horseId, out var history))
                            {
                                history = new List<HistoryEntry>();
                                _horseHistory[horseId] = history;
                            }
                            row["RatingChangeFromLast"] = history.Count > 0 ? rating - history[^1].Rating : 0f;
                            row["WeightChangeFromLast"] = history.Count > 0 ? weight - history[^1].Weight : 0f;
                            row["AgeProgression"] = history.Count > 0 ? age - history[^1].Age : 0f;
                            row["ClassChangeFromLast"] = history.Count > 0 ? classVal - history[^1].RaceClass : 0;
                            row["DaysSinceLastRace"] = history.Count > 0 ? (float)(date - history[^1].Date).TotalDays : 0f;
                            row["LastFinishPos"] = history.Count > 0 ? history[^1].Finish ?? 0 : 0;
                            int lastWinIdx = history.FindLastIndex(h => h.Won);
                            bool hasLastWin = lastWinIdx >= 0;
                            row["HasLastWin"] = hasLastWin;
                            if (hasLastWin)
                            {
                                row["DaysSinceLastWin"] = (float?)(date - history[lastWinIdx].Date).TotalDays;
                                row["RacesSinceLastWin"] = history.Count - 1 - lastWinIdx;
                            }
                            else
                            {
                                row["DaysSinceLastWin"] = null;
                                row["RacesSinceLastWin"] = null;
                            }
                            var daysSinceLast = (float)row["DaysSinceLastRace"];
                            row["LayoffShort"] = daysSinceLast < 30f;
                            row["LayoffMedium"] = daysSinceLast >= 30f && daysSinceLast <= 90f;
                            row["LayoffLong"] = daysSinceLast > 90f;
                            row["LayoffNormalized"] = daysSinceLast / TypicalRestDays;
                            for (int i = 0; i < PastRaceCount; i++)
                            {
                                var key = $"Last{i + 1}NormPos";
                                row[key] = i < history.Count
                                     ? history[history.Count - 1 - i].NormFinish
                                    : 0f;
                            }
                            int normCount = Math.Min(PastRaceCount, history.Count);
                            if (normCount > 1)
                            {
                                // Oldest first for regression
                                var recentNorms = history
                                    .GetRange(history.Count - normCount, normCount)
                                    .Select(h => h.NormFinish)
                                    .ToList();

                                float xMean = (normCount - 1) / 2f;
                                float yMean = recentNorms.Average();
                                float num = 0f, den = 0f;
                                for (int j = 0; j < recentNorms.Count; j++)
                                {
                                    float x = j;
                                    float y = recentNorms[j];
                                    num += (x - xMean) * (y - yMean);
                                    den += (x - xMean) * (x - xMean);
                                }
                                row["NormPosSlope"] = den != 0f ? num / den : 0f;
                            }
                            else
                            {
                                row["NormPosSlope"] = 0f;
                            }
                            int speedCount = Math.Min(PastRaceCount, history.Count);
                            if (speedCount > 0)
                            {
                                var recentSpeeds = history
                                    .GetRange(history.Count - speedCount, speedCount)
                                    .Select(h => h.Speed)
                                    .ToList();
                                float speedMean = recentSpeeds.Average();
                                float variance = 0f;
                                foreach (var s in recentSpeeds)
                                {
                                    float diff = s - speedMean;
                                    variance += diff * diff;
                                }
                                row["SpeedStdDev"] = (float)Math.Sqrt(variance / recentSpeeds.Count);
                                if (speedCount > 1)
                                {
                                    float xMean = (speedCount - 1) / 2f;
                                    float num = 0f, den = 0f;
                                    for (int j = 0; j < recentSpeeds.Count; j++)
                                    {
                                        float x = j;
                                        float y = recentSpeeds[j];
                                        num += (x - xMean) * (y - speedMean);
                                        den += (x - xMean) * (x - xMean);
                                    }
                                    row["SpeedSlope"] = den != 0f ? num / den : 0f;
                                }
                                else
                                {
                                    row["SpeedSlope"] = 0f;
                                }
                            }
                            else
                            {
                                row["SpeedSlope"] = 0f;
                                row["SpeedStdDev"] = 0f;
                            }
                            int ratingCount = Math.Min(PastRaceCount, history.Count);
                            if (ratingCount > 0)
                            {
                                var recentRatings = history
                                    .GetRange(history.Count - ratingCount, ratingCount)
                                    .Select(h => h.Rating)
                                    .ToList();
                                if (ratingCount > 1)
                                {
                                    float ratingMean = recentRatings.Average();
                                    float xMean = (ratingCount - 1) / 2f;
                                    float num = 0f, den = 0f;
                                    for (int j = 0; j < recentRatings.Count; j++)
                                    {
                                        float x = j;
                                        float y = recentRatings[j];
                                        num += (x - xMean) * (y - ratingMean);
                                        den += (x - xMean) * (x - xMean);
                                    }
                                    row["RatingSlope"] = den != 0f ? num / den : 0f;
                                }
                                else
                                {
                                    row["RatingSlope"] = 0f;
                                }
                            }
                            else
                            {
                                row["RatingSlope"] = 0f;
                            }
                            row["RecentImprovement"] =
                                    (float)row["Last1NormPos"] - (float)row[$"Last{PastRaceCount}NormPos"];
                            int careerStarts = history.Count;
                            int careerWins = history.Count(h => h.Won);
                            row["CareerStarts"] = careerStarts;
                            row["LifetimeWinRate"] = _trainer.SmoothedWinRate(careerWins, careerStarts);
                            foreach (var window in PerformanceWindows)
                            {
                                int count = Math.Min(window, history.Count);

                                // Ensure `recent` is available regardless of branch to avoid scope issues.
                                List<HistoryEntry> recent;
                                if (count > 0)
                                {
                                    recent = history.GetRange(history.Count - count, count);
                                    int wins = recent.Count(h => h.Finish == 1);

                                    row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(wins, count);
                                    row[$"AvgNormPosLast{window}"] = recent.Sum(h => h.NormFinish) / count;
                                    row[$"AvgSpeedLast{window}"] = recent.Sum(h => h.Speed) / count;
                                    row[$"AvgSpeedDiffLast{window}"] = recent.Sum(h => h.SpeedDiff) / count;
                                    row[$"AvgRatingLast{window}"] = recent.Sum(h => h.Rating) / count;

                                }
                                else
                                {
                                    recent = new();
                                    row[$"WinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"AvgNormPosLast{window}"] = 0f;
                                    row[$"AvgSpeedLast{window}"] = 0f;
                                    row[$"AvgSpeedDiffLast{window}"] = 0f;
                                    row[$"AvgRatingLast{window}"] = 0f;

                                }
                            }
                            // Going performance
                            string going = row["Going"] as string ?? "Unknown";
                            if (!_goingStats.TryGetValue(horseId, out var gDict))
                            {
                                gDict = new();
                                _goingStats[horseId] = gDict;
                            }
                            if (!gDict.TryGetValue(going, out var gStats))
                                gStats = (0, 0, 0f, 0f);
                            row["GoingWinRate"] = _trainer.SmoothedWinRate(gStats.wins, gStats.starts);
                            row["GoingAvgNorm"] = gStats.starts > 0 ? gStats.sumNorm / gStats.starts : 0f;
                            row["LastGoingNormPos"] = gStats.lastNorm;
                            var goingFeatureKey = going.Replace(" ", "");
                            row[$"LayoffNormalized_{goingFeatureKey}"] = (float)row["LayoffNormalized"];
                            string surface = row["Surface"] as string ?? "Unknown";
                            if (!_surfaceStats.TryGetValue(horseId, out var sDict))
                            {
                                sDict = new();
                                _surfaceStats[horseId] = sDict;
                            }
                            if (!sDict.TryGetValue(surface, out var sStats))
                                sStats = (0, 0, 0f, 0f);
                            row["SurfaceWinRate"] = _trainer.SmoothedWinRate(sStats.wins, sStats.starts);
                            row["SurfaceAvgNorm"] = sStats.starts > 0 ? sStats.sumNorm / sStats.starts : 0f;
                            row["LastSurfaceNormPos"] = sStats.lastNorm;
                            // Going + Course preference
                            int courseId = row.TryGetValue("CourseId", out var courseObj) && PreparedDataset.TryConvertToInt32(courseObj, out var courseIdValue)
                                ? courseIdValue
                            : 0;
                            if (!_courseStats.TryGetValue(horseId, out var cDict))
                            {
                                cDict = new();
                                _courseStats[horseId] = cDict;
                            }
                            if (!cDict.TryGetValue(courseId, out var cStats))
                                cStats = (0, 0, 0f, 0f);
                            row["CourseWinRate"] = _trainer.SmoothedWinRate(cStats.wins, cStats.starts);
                            row["LastCourseNormPos"] = cStats.lastNorm;
                            string gcKey = going + "_" + courseId;
                            if (!_goingCourseStats.TryGetValue(horseId, out var gcDict))
                            {
                                gcDict = new();
                                _goingCourseStats[horseId] = gcDict;
                            }
                            if (!gcDict.TryGetValue(gcKey, out var gcStats))
                                gcStats = (0, 0, 0f, 0f);
                            row["GoingCourseWinRate"] = _trainer.SmoothedWinRate(gcStats.wins, gcStats.starts);
                            row["GoingCourseAvgNorm"] = gcStats.starts > 0 ? gcStats.sumNorm / gcStats.starts : 0f;
                            row["LastGoingCourseNormPos"] = gcStats.lastNorm;
                            // Age restriction context
                            string ageRes = row["AgeRestriction"] as string ?? "Unknown";
                            if (!_ageStats.TryGetValue(horseId, out var aDict))
                            {
                                aDict = new();
                                _ageStats[horseId] = aDict;
                            }
                            if (!aDict.TryGetValue(ageRes, out var aStats))
                                aStats = (0, 0, 0f, 0f);
                            row["AgeRestrictionWinRate"] = _trainer.SmoothedWinRate(aStats.wins, aStats.starts);
                            row["LastAgeRestrictionNormPos"] = aStats.lastNorm;

                            // Distance specialization
                            int distanceYards = row.TryGetValue("DistanceYards", out var distanceObj) && PreparedDataset.TryConvertToInt32(distanceObj, out var distanceValue)
                                            ? distanceValue
                                            : 0;
                            int? winningMs = row.TryGetValue("WinningTimeMs", out var winningObj) && PreparedDataset.TryConvertToInt32(winningObj, out var winningValue)
                                ? winningValue
                            : (int?)null;
                            bool speedMissing = !(winningMs.HasValue && winningMs.Value > 0);
                            float raceSpeed = speedMissing
                                ? 0f
                                : distanceYards / (float)winningMs.Value;
                            row["RaceSpeed"] = raceSpeed;
                            if (!speedMissing)
                            {
                                float beaten = row["DistanceBeatenLengths"] != null ? Convert.ToSingle(row["DistanceBeatenLengths"]) : 0f;
                                float runnerTime = winningMs.Value + beaten * MsPerLength;
                                runnerSpeed = runnerTime > 0f ? distanceYards / runnerTime : 0f;
                                row["RunnerSpeed"] = runnerSpeed;
                            }
                            else
                            {
                                runnerSpeed = 0f;
                                row["RunnerSpeed"] = 0f;
                            }
                            float speedDiff = runnerSpeed - raceSpeed;
                            row["SpeedMissing"] = speedMissing;
                            row["SpeedDiff"] = speedDiff;
                            row["SpeedRatio"] = raceSpeed != 0f ? runnerSpeed / raceSpeed : 0f;
                            string bucket = DistanceBucket(distanceYards);
                            row["DistanceBucket"] = bucket;
                            row[$"RelativeDraw_{bucket}"] = (float)row["RelativeDraw"];
                            var drawKey = (courseId, bucket, draw);
                            _drawStats.TryGetValue(drawKey, out var drawStat);
                            var baseKey = (courseId, bucket);
                            _drawBaselineStats.TryGetValue(baseKey, out var baseStat);
                            row["DrawBias"] = _trainer.SmoothedWinRate(drawStat.wins, drawStat.starts) -
                                             _trainer.SmoothedWinRate(baseStat.wins, baseStat.starts);
                            if (!_distanceBucketStats.TryGetValue(horseId, out var dDict))
                            {
                                dDict = new();
                                _distanceBucketStats[horseId] = dDict;
                            }
                            if (!dDict.TryGetValue(bucket, out var dStats))
                                dStats = (0, 0, 0f, 0f);
                            row["DistanceBucketWinRate"] = _trainer.SmoothedWinRate(dStats.wins, dStats.starts);
                            row["LastDistanceBucketNormPos"] = dStats.lastNorm;
                            var gdKey = (horseId, going, bucket);
                            if (!_goingDistanceStats.TryGetValue(gdKey, out var gdStats))
                                gdStats = (0, 0, 0f, 0f);
                            row["GoingDistanceWinRate"] = _trainer.SmoothedWinRate(gdStats.wins, gdStats.starts);
                            row["GoingDistanceAvgNorm"] = gdStats.starts > 0 ? gdStats.sumNorm / gdStats.starts : 0f;
                            row["LastGoingDistanceNormPos"] = gdStats.lastNorm;
                            foreach (var window in PerformanceWindows)
                            {
                                int count = Math.Min(window, history.Count);
                                if (count > 0)
                                {
                                    var recent = history.GetRange(history.Count - count, count);

                                    var goingRecent = recent.Where(h => h.Going == going).ToList();
                                    row[$"GoingWinRateLast{window}"] = _trainer.SmoothedWinRate(goingRecent.Count(h => h.Finish == 1), goingRecent.Count);
                                    row[$"GoingAvgNormLast{window}"] = goingRecent.Count > 0
                                        ? goingRecent.Sum(h => h.NormFinish) / goingRecent.Count
                                        : 0f;
                                    var surfaceRecent = recent.Where(h => h.Surface == surface).ToList();
                                    row[$"SurfaceWinRateLast{window}"] = _trainer.SmoothedWinRate(surfaceRecent.Count(h => h.Finish == 1), surfaceRecent.Count);
                                    row[$"SurfaceAvgNormLast{window}"] = surfaceRecent.Count > 0
                                        ? surfaceRecent.Sum(h => h.NormFinish) / surfaceRecent.Count
                                        : 0f;
                                    var courseRecent = recent.Where(h => h.CourseId == courseId).ToList();
                                    row[$"CourseWinRateLast{window}"] = _trainer.SmoothedWinRate(courseRecent.Count(h => h.Finish == 1), courseRecent.Count);
                                    row[$"CourseAvgNormLast{window}"] = courseRecent.Count > 0
                                        ? courseRecent.Sum(h => h.NormFinish) / courseRecent.Count
                                        : 0f;

                                    var bucketRecent = recent.Where(h => h.Bucket == bucket).ToList();
                                    row[$"DistanceBucketWinRateLast{window}"] = _trainer.SmoothedWinRate(bucketRecent.Count(h => h.Finish == 1), bucketRecent.Count);
                                    row[$"DistanceBucketAvgNormLast{window}"] = bucketRecent.Count > 0
                                        ? bucketRecent.Sum(h => h.NormFinish) / bucketRecent.Count
                                        : 0f;
                                }
                                else
                                {
                                    row[$"GoingWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"GoingAvgNormLast{window}"] = 0f;
                                    row[$"SurfaceWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"SurfaceAvgNormLast{window}"] = 0f;
                                    row[$"CourseWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"CourseAvgNormLast{window}"] = 0f;
                                    row[$"DistanceBucketWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"DistanceBucketAvgNormLast{window}"] = 0f;
                                }
                            }
                            // Preferred distance deviation
                            _horseDistanceAll.TryGetValue(horseId, out var allDist);
                            _horseDistanceWins.TryGetValue(horseId, out var winDist);
                            double avgDist = allDist.count > 0 ? allDist.sum / allDist.count : distanceYards;
                            row["DistanceRatioFromAverage"] = avgDist > 0 ? distanceYards / (float)avgDist : 1f;
                            if (_horseLastDistance.TryGetValue(horseId, out var lastDist))
                                row["DistanceChangeFromLast"] = distanceYards - lastDist;
                            else
                                row["DistanceChangeFromLast"] = 0f;
                            double pref = winDist.count > 0 ? winDist.sum / winDist.count : (allDist.count > 0 ? allDist.sum / allDist.count : distanceYards);
                            row["DistanceFromPreferred"] = (float)Math.Abs(distanceYards - pref);

                            // Trainer statistics
                            if (trainerId.HasValue)
                            {
                                if (!_trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict))
                                {
                                    tsDict = new();
                                    _trainerSurfaceStats[trainerId.Value] = tsDict;
                                }
                                if (!tsDict.TryGetValue(surface, out var tsStats))
                                    tsStats = (0, 0, 0f, 0f);
                                row["TrainerSurfaceWinRate"] = _trainer.SmoothedWinRate(tsStats.wins, tsStats.starts);
                                row["TrainerSurfaceAvgNorm"] = tsStats.starts > 0 ? tsStats.sumNorm / tsStats.starts : 0f;
                                row["LastTrainerSurfaceNormPos"] = tsStats.lastNorm;

                                if (!_trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict))
                                {
                                    tgDict = new();
                                    _trainerGoingStats[trainerId.Value] = tgDict;
                                }
                                if (!tgDict.TryGetValue(going, out var tgStats))
                                    tgStats = (0, 0, 0f, 0f);
                                row["TrainerGoingWinRate"] = _trainer.SmoothedWinRate(tgStats.wins, tgStats.starts);
                                row["TrainerGoingAvgNorm"] = tgStats.starts > 0 ? tgStats.sumNorm / tgStats.starts : 0f;
                                row["LastTrainerGoingNormPos"] = tgStats.lastNorm;

                                if (!_trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict))
                                {
                                    tdDict = new();
                                    _trainerDistanceStats[trainerId.Value] = tdDict;
                                }
                                if (!tdDict.TryGetValue(bucket, out var tdStats))
                                    tdStats = (0, 0, 0f, 0f);
                                row["TrainerDistanceBucketWinRate"] = _trainer.SmoothedWinRate(tdStats.wins, tdStats.starts);
                                row["TrainerDistanceBucketAvgNorm"] = tdStats.starts > 0 ? tdStats.sumNorm / tdStats.starts : 0f;
                                row["LastTrainerDistanceBucketNormPos"] = tdStats.lastNorm;
                                if (!_trainerStats.TryGetValue(trainerId.Value, out var trainerStat))
                                    trainerStat = new RollingStat();
                                while (trainerStat.Recent.Count > 0 &&
                                                        (date - trainerStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                                    trainerStat.Recent.Dequeue();

                                var tRecent = trainerStat.Recent.ToList();
                                int tCount = Math.Min(tRecent.Count, TrainerJockeyRecentStarts);
                                int tWinsRecent = tRecent.Skip(tRecent.Count - tCount).Count(r => r.win);
                                row[$"TrainerWinRateLast{TrainerJockeyRecentStarts}"] = _trainer.SmoothedWinRate(tWinsRecent, tCount);
                                row["TrainerWinRateRecentDays"] = _trainer.SmoothedWinRate(tRecent.Count(r => r.win), tRecent.Count);
                                row["TrainerWinRate"] = _trainer.SmoothedWinRate(trainerStat.Wins, trainerStat.Starts);

                                if (updateState)
                                {
                                    trainerStat.Starts++;
                                    bool tWin = finish.HasValue && finish.Value == 1;
                                    if (tWin) trainerStat.Wins++;
                                    trainerStat.Recent.Enqueue((date, tWin));
                                    while (trainerStat.Recent.Count > TrainerJockeyRecentStarts)
                                        trainerStat.Recent.Dequeue();
                                    _trainerStats[trainerId.Value] = trainerStat;
                                }
                            }
                            else
                            {
                                row["TrainerWinRate"] = 0f;
                                row[$"TrainerWinRateLast{TrainerJockeyRecentStarts}"] = 0f;
                                row["TrainerWinRateRecentDays"] = 0f;
                                row["TrainerSurfaceWinRate"] = 0f;
                                row["TrainerSurfaceAvgNorm"] = 0f;
                                row["LastTrainerSurfaceNormPos"] = 0f;
                                row["TrainerGoingWinRate"] = 0f;
                                row["TrainerGoingAvgNorm"] = 0f;
                                row["LastTrainerGoingNormPos"] = 0f;
                                row["TrainerDistanceBucketWinRate"] = 0f;
                                row["TrainerDistanceBucketAvgNorm"] = 0f;
                                row["LastTrainerDistanceBucketNormPos"] = 0f;
                            }
                            if (jockeyId.HasValue)
                            {
                                if (!_jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict))
                                {
                                    jsDict = new();
                                    _jockeySurfaceStats[jockeyId.Value] = jsDict;
                                }
                                if (!jsDict.TryGetValue(surface, out var jsStats))
                                    jsStats = (0, 0, 0f, 0f);
                                row["JockeySurfaceWinRate"] = _trainer.SmoothedWinRate(jsStats.wins, jsStats.starts);
                                row["JockeySurfaceAvgNorm"] = jsStats.starts > 0 ? jsStats.sumNorm / jsStats.starts : 0f;
                                row["LastJockeySurfaceNormPos"] = jsStats.lastNorm;

                                if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict))
                                {
                                    jgDict = new();
                                    _jockeyGoingStats[jockeyId.Value] = jgDict;
                                }
                                if (!jgDict.TryGetValue(going, out var jgStats))
                                    jgStats = (0, 0, 0f, 0f);
                                row["JockeyGoingWinRate"] = _trainer.SmoothedWinRate(jgStats.wins, jgStats.starts);
                                row["JockeyGoingAvgNorm"] = jgStats.starts > 0 ? jgStats.sumNorm / jgStats.starts : 0f;
                                row["LastJockeyGoingNormPos"] = jgStats.lastNorm;

                                if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict))
                                {
                                    jdDict = new();
                                    _jockeyDistanceStats[jockeyId.Value] = jdDict;
                                }
                                if (!jdDict.TryGetValue(bucket, out var jdStats))
                                    jdStats = (0, 0, 0f, 0f);
                                row["JockeyDistanceBucketWinRate"] = _trainer.SmoothedWinRate(jdStats.wins, jdStats.starts);
                                row["JockeyDistanceBucketAvgNorm"] = jdStats.starts > 0 ? jdStats.sumNorm / jdStats.starts : 0f;
                                row["LastJockeyDistanceBucketNormPos"] = jdStats.lastNorm;
                                var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                                if (!_jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat))
                                    jockeyGoingDistanceStat = (0, 0, 0f, 0f);
                                row["JockeyGoingDistanceWinRate"] = _trainer.SmoothedWinRate(jockeyGoingDistanceStat.wins, jockeyGoingDistanceStat.starts);
                                row["JockeyGoingDistanceAvgNorm"] = jockeyGoingDistanceStat.starts > 0
                                    ? jockeyGoingDistanceStat.sumNorm / jockeyGoingDistanceStat.starts
                                    : 0f;
                                row["LastJockeyGoingDistanceNormPos"] = jockeyGoingDistanceStat.lastNorm;
                                if (!_jockeyStats.TryGetValue(jockeyId.Value, out var jockeyStat))
                                    jockeyStat = new RollingStat();

                                while (jockeyStat.Recent.Count > 0 &&
                                       (date - jockeyStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                                    jockeyStat.Recent.Dequeue();

                                var jRecent = jockeyStat.Recent.ToList();
                                int jCount = Math.Min(jRecent.Count, TrainerJockeyRecentStarts);
                                int jWinsRecent = jRecent.Skip(jRecent.Count - jCount).Count(r => r.win);
                                row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = _trainer.SmoothedWinRate(jWinsRecent, jCount);
                                row["JockeyWinRateRecentDays"] = _trainer.SmoothedWinRate(jRecent.Count(r => r.win), jRecent.Count);
                                row["JockeyWinRate"] = _trainer.SmoothedWinRate(jockeyStat.Wins, jockeyStat.Starts);

                                if (updateState)
                                {
                                    jockeyStat.Starts++;
                                    bool jWin = finish.HasValue && finish.Value == 1;
                                    if (jWin) jockeyStat.Wins++;
                                    jockeyStat.Recent.Enqueue((date, jWin));
                                    while (jockeyStat.Recent.Count > TrainerJockeyRecentStarts)
                                        jockeyStat.Recent.Dequeue();
                                    _jockeyStats[jockeyId.Value] = jockeyStat;
                                }
                            }
                            else
                            {
                                row["JockeyWinRate"] = 0f;
                                row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = 0f;
                                row["JockeyWinRateRecentDays"] = 0f;
                                row["JockeySurfaceWinRate"] = 0f;
                                row["JockeySurfaceAvgNorm"] = 0f;
                                row["LastJockeySurfaceNormPos"] = 0f;
                                row["JockeyGoingWinRate"] = 0f;
                                row["JockeyGoingAvgNorm"] = 0f;
                                row["LastJockeyGoingNormPos"] = 0f;
                                row["JockeyDistanceBucketWinRate"] = 0f;
                                row["JockeyDistanceBucketAvgNorm"] = 0f;
                                row["LastJockeyDistanceBucketNormPos"] = 0f;
                                row["JockeyGoingDistanceWinRate"] = 0f;
                                row["JockeyGoingDistanceAvgNorm"] = 0f;
                                row["LastJockeyGoingDistanceNormPos"] = 0f;
                            }
                            if (trainerId.HasValue)
                            {
                                var tcKey = (trainerId.Value, courseId);
                                _trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                                row["TrainerCourseWinRate"] = _trainer.SmoothedWinRate(tcStat.wins, tcStat.starts);
                            }
                            else
                            {
                                row["TrainerCourseWinRate"] = 0f;
                            }
                            if (jockeyId.HasValue)
                            {
                                var jcKey = (jockeyId.Value, courseId);
                                if (!_jockeyCourseStats.TryGetValue(jcKey, out var jcStat))
                                {
                                    jcStat = (0, 0);
                                }
                                row["JockeyCourseWinRate"] = _trainer.SmoothedWinRate(jcStat.wins, jcStat.starts);
                            }
                            else
                            {
                                row["JockeyCourseWinRate"] = 0f;
                            }
                            if (trainerId.HasValue && jockeyId.HasValue)
                            {
                                var pairKey = (trainerId.Value, jockeyId.Value);
                                if (!_trainerJockeyStats.TryGetValue(pairKey, out var pairStat))
                                    pairStat = (0, 0);
                                row["TrainerJockeyWinRate"] = _trainer.SmoothedWinRate(pairStat.wins, pairStat.starts);
                                var trainerJockeySurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                                if (!_trainerJockeySurfaceStats.TryGetValue(trainerJockeySurfaceKey, out var pairSurfaceStat))
                                    pairSurfaceStat = (0, 0);
                                row["TrainerJockeySurfaceWinRate"] =
                                    _trainer.SmoothedWinRate(pairSurfaceStat.wins, pairSurfaceStat.starts);
                            }
                            else
                            {
                                row["TrainerJockeyWinRate"] = 0f;
                                row["TrainerJockeySurfaceWinRate"] = 0f;
                                row["TrainerJockeyCourseWinRate"] = 0f;
                            }
                            if (updateState)
                            {
                                // Compute normalized finish
                                float normFinish = (finish.HasValue && runnerCount > 1)
                                ? (runnerCount - finish.Value) / (float)(runnerCount - 1)
                            : 0f;
                                history.Add(new HistoryEntry(
                                    date,
                                    normFinish,
                                    finish,
                                    going,
                                    surface,
                                    courseId,
                                    bucket,
                                    classVal,
                                    runnerSpeed,
                                    speedDiff,
                                    age,
                                    finish.HasValue && finish.Value == 1,
                                    rating,
                                    weight));
                                horseClassStat.starts++;
                                horseClassStat.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) horseClassStat.wins++;
                                horseClassStat.lastNorm = normFinish;
                                _horseClassStats[horseClassKey] = horseClassStat;

                                if (hasTrainerClass)
                                {
                                    trainerClassStat.starts++;
                                    trainerClassStat.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) trainerClassStat.wins++;
                                    trainerClassStat.lastNorm = normFinish;
                                    _trainerClassStats[trainerClassKey] = trainerClassStat;
                                }

                                if (hasJockeyClass)
                                {
                                    jockeyClassStat.starts++;
                                    jockeyClassStat.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) jockeyClassStat.wins++;
                                    jockeyClassStat.lastNorm = normFinish;
                                    _jockeyClassStats[jockeyClassKey] = jockeyClassStat;
                                }
                                if (history.Count > HistoryLength)
                                    history.RemoveAt(0);
                                sStats.starts++;
                                sStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) sStats.wins++;
                                sStats.lastNorm = normFinish;
                                sDict[surface] = sStats;
                                // Update going stats after race
                                gStats.starts++;
                                gStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) gStats.wins++;
                                gStats.lastNorm = normFinish;
                                gDict[going] = gStats;

                                gcStats.starts++;
                                gcStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) gcStats.wins++;
                                gcStats.lastNorm = normFinish;
                                gcDict[gcKey] = gcStats;
                                cStats.starts++;
                                cStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) cStats.wins++;
                                cStats.lastNorm = normFinish;
                                cDict[courseId] = cStats;
                                aStats.starts++;
                                aStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) aStats.wins++;
                                aStats.lastNorm = normFinish;
                                aDict[ageRes] = aStats;

                                dStats.starts++;
                                dStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) dStats.wins++;
                                dStats.lastNorm = normFinish;
                                dDict[bucket] = dStats;
                                var updateKey = (horseId, going, bucket);
                                _goingDistanceStats.TryGetValue(updateKey, out var updateStats);
                                updateStats.starts++;
                                updateStats.sumNorm += normFinish;
                                if (finish.HasValue && finish.Value == 1) updateStats.wins++;
                                updateStats.lastNorm = normFinish;
                                _goingDistanceStats[updateKey] = updateStats;
                                drawStat.starts++;
                                if (finish.HasValue && finish.Value == 1) drawStat.wins++;
                                _drawStats[drawKey] = drawStat;
                                baseStat.starts++;
                                if (finish.HasValue && finish.Value == 1) baseStat.wins++;
                                _drawBaselineStats[baseKey] = baseStat;
                                allDist.sum += distanceYards;
                                allDist.count++;
                                _horseDistanceAll[horseId] = allDist;
                                _horseLastDistance[horseId] = distanceYards;
                                if (finish.HasValue && finish.Value == 1)
                                {
                                    winDist.sum += distanceYards;
                                    winDist.count++;
                                }
                                _horseDistanceWins[horseId] = winDist;
                                if (trainerId.HasValue)
                                {
                                    if (!_trainerSurfaceStats.TryGetValue(trainerId.Value, out var tsDict)) tsDict = new();
                                    if (!tsDict.TryGetValue(surface, out var tsStats)) tsStats = (0, 0, 0f, 0f);
                                    tsStats.starts++; tsStats.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) tsStats.wins++;
                                    tsStats.lastNorm = normFinish;
                                    tsDict[surface] = tsStats;

                                    if (!_trainerGoingStats.TryGetValue(trainerId.Value, out var tgDict)) tgDict = new();
                                    if (!tgDict.TryGetValue(going, out var tgStats)) tgStats = (0, 0, 0f, 0f);
                                    tgStats.starts++; tgStats.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) tgStats.wins++;
                                    tgStats.lastNorm = normFinish;
                                    tgDict[going] = tgStats;

                                    if (!_trainerDistanceStats.TryGetValue(trainerId.Value, out var tdDict)) tdDict = new();
                                    if (!tdDict.TryGetValue(bucket, out var tdStats)) tdStats = (0, 0, 0f, 0f);
                                    tdStats.starts++; tdStats.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) tdStats.wins++;
                                    tdStats.lastNorm = normFinish;
                                    tdDict[bucket] = tdStats;
                                    var tcKey = (trainerId.Value, courseId);
                                    _trainerCourseStats.TryGetValue(tcKey, out var tcStat);
                                    tcStat.starts++;
                                    if (finish.HasValue && finish.Value == 1) tcStat.wins++;
                                    _trainerCourseStats[tcKey] = tcStat;
                                }
                                if (jockeyId.HasValue)
                                {
                                    if (!_jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jsDict)) jsDict = new();
                                    if (!jsDict.TryGetValue(surface, out var jsStats)) jsStats = (0, 0, 0f, 0f);
                                    jsStats.starts++; jsStats.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) jsStats.wins++;
                                    jsStats.lastNorm = normFinish;
                                    jsDict[surface] = jsStats;

                                    if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict)) jgDict = new();
                                    if (!jgDict.TryGetValue(going, out var jgStats)) jgStats = (0, 0, 0f, 0f);
                                    jgStats.starts++; jgStats.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) jgStats.wins++;
                                    jgStats.lastNorm = normFinish;
                                    jgDict[going] = jgStats;

                                    if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict)) jdDict = new();
                                    if (!jdDict.TryGetValue(bucket, out var jdStats)) jdStats = (0, 0, 0f, 0f);
                                    jdStats.starts++; jdStats.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) jdStats.wins++;
                                    jdStats.lastNorm = normFinish;
                                    jdDict[bucket] = jdStats;
                                    var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                                    _jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat);
                                    jockeyGoingDistanceStat.starts++;
                                    jockeyGoingDistanceStat.sumNorm += normFinish;
                                    if (finish.HasValue && finish.Value == 1) jockeyGoingDistanceStat.wins++;
                                    jockeyGoingDistanceStat.lastNorm = normFinish;
                                    _jockeyGoingDistanceStats[jockeyGoingDistanceKey] = jockeyGoingDistanceStat;
                                    var jcKey = (jockeyId.Value, courseId);
                                    if (!_jockeyCourseStats.TryGetValue(jcKey, out var jcStat))
                                    {
                                        jcStat = (0, 0);
                                    }
                                    jcStat.starts++;
                                    if (finish.HasValue && finish.Value == 1)
                                    {
                                        jcStat.wins++;
                                    }
                                    _jockeyCourseStats[jcKey] = jcStat;
                                }
                                if (trainerId.HasValue && jockeyId.HasValue)
                                {
                                    var pairKey = (trainerId.Value, jockeyId.Value);
                                    _trainerJockeyStats.TryGetValue(pairKey, out var pairStat);
                                    pairStat.starts++;
                                    if (finish.HasValue && finish.Value == 1)
                                    {
                                        pairStat.wins++;
                                    }
                                    _trainerJockeyStats[pairKey] = pairStat;
                                    var pairSurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                                    _trainerJockeySurfaceStats.TryGetValue(pairSurfaceKey, out var pairSurfaceStat);
                                    pairSurfaceStat.starts++;
                                    if (finish.HasValue && finish.Value == 1) pairSurfaceStat.wins++;
                                    _trainerJockeySurfaceStats[pairSurfaceKey] = pairSurfaceStat;
                                }
                            }
                        }
                    }
                }
                if (includeRace)
                {
                    float raceAvgSpeed = rows
                                            .Select(r => r.ContainsKey("AvgSpeedLast5") && r["AvgSpeedLast5"] != null ? Convert.ToSingle(r["AvgSpeedLast5"]) : 0f)
                                            .DefaultIfEmpty(0f)
                                            .Average();
                    float raceAvgWinRate = rows
                        .Select(r => r.ContainsKey("WinRateLast5") && r["WinRateLast5"] != null ? Convert.ToSingle(r["WinRateLast5"]) : 0f)
                        .DefaultIfEmpty(0f)
                        .Average();

                    foreach (var raceRow in rows)
                    {
                        raceRow["RaceAvgSpeedLast5"] = raceAvgSpeed;
                        raceRow["RaceAvgWinRateLast5"] = raceAvgWinRate;
                        TrimRunnerRow(raceRow, _identifierKeys);
                    }
                }
            }
            private static bool TryGetTimeOfDay(Dictionary<string, object?> row, out TimeSpan timeOfDay)
            {
                if (TryReadTimeValue(row, "ActualOff", out timeOfDay) ||
                    TryReadTimeValue(row, "ScheduledOff", out timeOfDay))
                {
                    return true;
                }

                if (row.TryGetValue("RaceDate", out var dateObj) && dateObj != null)
                {
                    switch (dateObj)
                    {
                        case DateTime dt:
                            timeOfDay = dt.TimeOfDay;
                            return true;
                        case DateTimeOffset dto:
                            timeOfDay = dto.TimeOfDay;
                            return true;
                    }
                }

                timeOfDay = default;
                return false;
            }

            public static bool TryReadTimeValue(Dictionary<string, object?> row, string key, out TimeSpan time)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    time = default;
                    return false;
                }

                switch (value)
                {
                    case TimeSpan ts:
                        time = ts;
                        return true;
                    case DateTime dt:
                        time = dt.TimeOfDay;
                        return true;
                    case DateTimeOffset dto:
                        time = dto.TimeOfDay;
                        return true;
                    case string s when TimeSpan.TryParse(s, CultureInfo.InvariantCulture, out var parsed):
                        time = parsed;
                        return true;
                    default:
                        time = default;
                        return false;
                }
            }
            private static string DistanceBucket(int yards)
                => yards < 1760 ? "Sprint" : yards < 2640 ? "Middle" : "Long";
            private static string NormalizeHorseNameForLookup(string horseName)
            {
                if (string.IsNullOrWhiteSpace(horseName))
                {
                    return string.Empty;
                }

                var trimmed = horseName.Trim();
                var withoutBracketed = HorseNameBracketTextRegex.Replace(trimmed, " ");
                var collapsed = HorseNameWhitespaceRegex.Replace(withoutBracketed, " ").Trim();

                return string.IsNullOrEmpty(collapsed) ? trimmed : collapsed;
            }
            private int ResolveHorseId(SqlConnection conn, string horseName)
            {
                const string sql = "SELECT TOP (1) HorseId FROM Horse WHERE Name = @Name ORDER BY HorseId";
                var normalizedName = NormalizeHorseNameForLookup(horseName);
                var existing = conn.QuerySingleOrDefault<int?>(sql, new { Name = normalizedName });
                if (existing.HasValue)
                {
                    return existing.Value;
                }

                var syntheticSeed = string.IsNullOrWhiteSpace(normalizedName) ? horseName : normalizedName;
                return GenerateSyntheticId("horse:" + syntheticSeed);
            }

            private RaceStats ComputeRaceStats(List<Dictionary<string, object?>> rows)
            {
                int cnt = rows.Count;
                var drawValues = rows.Where(r => r["Draw"] != null).Select(r => Convert.ToSingle(r["Draw"])).ToList();
                float avgDraw = drawValues.Count > 0 ? drawValues.Average() : 0f;
                var weightValues = rows.Where(r => r["WeightLbs"] != null).Select(r => Convert.ToSingle(r["WeightLbs"])).ToList();
                float avgWeight = weightValues.Count > 0 ? weightValues.Average() : 0f;
                float minWeight = weightValues.Count > 0 ? weightValues.Min() : 0f;
                float maxWeight = weightValues.Count > 0 ? weightValues.Max() : 0f;
                var saddleclothValues = rows.Where(r => r.TryGetValue("SaddleclothNumber", out var scObj) && scObj != null).Select(r => Convert.ToSingle(r["SaddleclothNumber"])).ToList();
                float avgSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Average() : 0f;
                float minSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Min() : 0f;
                float maxSaddlecloth = saddleclothValues.Count > 0 ? saddleclothValues.Max() : 0f;
                float avgAge = rows.Where(r => r["Age"] != null).Select(r => Convert.ToSingle(r["Age"])).DefaultIfEmpty(0f).Average();
                var ratingValues = rows.Where(r => r.TryGetValue("OfficialRating", out var orObj) && orObj != null).Select(r => Convert.ToSingle(r["OfficialRating"])).ToList();
                float avgRating = ratingValues.DefaultIfEmpty(0f).Average();
                float stdRating = 0f;
                if (ratingValues.Count > 0)
                {
                    float variance = ratingValues.Select(r => (r - avgRating) * (r - avgRating)).Average();
                    stdRating = (float)Math.Sqrt(variance);
                }
                float purse = rows.Select(r => r.TryGetValue("Purse", out var pObj) && pObj != null ? Convert.ToSingle(pObj) : 0f).FirstOrDefault();
                float avgSaddle = saddleclothValues.Count > 0 ? saddleclothValues.Average() : 0f;
                float minSaddle = saddleclothValues.Count > 0 ? saddleclothValues.Min() : 0f;
                float maxSaddle = saddleclothValues.Count > 0 ? saddleclothValues.Max() : 0f;

                return new RaceStats(
                    cnt,
                    avgDraw,
                    avgWeight,
                    minWeight,
                    maxWeight,
                    avgAge,
                    avgRating,
                    stdRating,
                    purse,
                    saddleclothValues.Count > 0,
                    avgSaddle,
                    minSaddle,
                    maxSaddle,
                    weightValues.Count > 0);
            }

            private readonly record struct RaceStats(
                       int RunnerCount,
                       float AvgDraw,
                       float AvgWeight,
                       float MinWeight,
                       float MaxWeight,
                       float AvgAge,
                       float AvgRating,
                       float StdRating,
                       float TotalPurse,
                       bool HasSaddleclothStats,
                       float AvgSaddlecloth,
                       float MinSaddlecloth,
                       float MaxSaddlecloth,
                       bool HasWeightStats);
        }

        private static float? ParseDistanceBeaten(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return null;
            text = text.Trim().ToLowerInvariant();
            var map = new Dictionary<string, float>
            {
                {"nse", 0.05f},
                {"nose", 0.05f},
                {"shd", 0.1f},
                {"sht-hd", 0.1f},
                {"hd", 0.2f},
                {"snk", 0.25f},
                {"nk", 0.3f},
                {"dist", 30f}
            };
            if (map.TryGetValue(text, out var val))
                return val;
            text = text.Replace("¼", ".25").Replace("½", ".5").Replace("¾", ".75");
            double total = 0;
            foreach (var part in text.Split(new[] { ' ', '+' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (double.TryParse(part, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out var num))
                {
                    total += num;
                }
                else if (part.Contains('/'))
                {
                    var frac = part.Split('/');
                    if (frac.Length == 2 &&
                        double.TryParse(frac[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) &&
                        double.TryParse(frac[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var d) &&
                        d != 0)
                    {
                        total += n / d;
                    }
                }
            }
            return total > 0 ? (float)total : (float?)null;
        }

        private float SmoothedWinRate(int wins, int starts)
            => (wins + _winRateAlpha) / (starts + _winRateBeta);


        private static object? NormalizeDbValue(object? value)
        {
            return value == null || value is DBNull ? null : value;
        }

        private static void Shuffle<T>(IList<T> list, Random random)
        {
            if (list is null)
                throw new ArgumentNullException(nameof(list));
            if (random is null)
                throw new ArgumentNullException(nameof(random));

            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        private static readonly HashSet<string> NonTrainingKeys = new HashSet<string>
        {
                "HorseId",
                "CourseId",
                "TrainerId",
                "JockeyId",
                "OutcomeCode",
                "DistanceBeatenText",
                "SP_Fraction",
                "SP_Decimal",
                "OpeningFraction",
                "TouchedHighFraction",
                "TouchedLowFraction",
                "HorseName",
                "JockeyName",
                "TrainerName",
                "Title",
                "RaceMonth",
                "RaceDayOfWeek",
                "Season",
                "ActualOff",
                "ScheduledOff",
                "RaceDate",
                "CourseName",
                "DistanceText",
                "Status",
                "WeightText",
                "FavTag",
                "SaddleclothNumber",
                "Purse"
        };



        private static void TrimRunnerRow(Dictionary<string, object?> row, ISet<string>? preserveKeys = null)
        {
            if (row is null)
                throw new ArgumentNullException(nameof(row));

            foreach (var key in NonTrainingKeys)
            {
                if (preserveKeys != null && preserveKeys.Contains(key))
                {
                    continue;
                }

                row.Remove(key);
            }
        }

        public virtual PreparedDataset PrepareDataset(
                ISet<int>? includeRaceIds = null,
                ISet<int>? stateRaceWhitelist = null,
                bool includeIdentifiers = false)
        {
            using var conn = new SqlConnection(_connectionString);
            conn.Open();

            var raceColumns = PreparedDataset.LoadColumnNames(conn, "Race");
            string scheduledOffColumn = raceColumns.Contains("ScheduledOff")
            ? "r.ScheduledOff AS ScheduledOff"
            : "CAST(NULL AS time(0)) AS ScheduledOff";
            string actualOffColumn = raceColumns.Contains("ActualOff")
                ? "r.ActualOff AS ActualOff"
                : "CAST(NULL AS time(0)) AS ActualOff";

            var sql = $@"SELECT c.Name AS CourseName,
                                   h.Name AS HorseName,
                                   j.Name AS JockeyName,
                                   t.Name AS TrainerName,
                                   r.RaceId,
                                   r.CourseId,
                                   r.RaceDate,
                                   {scheduledOffColumn},
                                   {actualOffColumn},
                                   r.Title,
                                   r.RaceType,
                                   r.Class,
                                   r.AgeRestriction,
                                   r.Surface,
                                   r.Going,
                                   r.DistanceYards,
                                   r.DistanceText,
                                   r.RunnerCount,
                                   r.Status,
                                   r.WinningTimeMs,
                                   rr.HorseId,
                                   rr.TrainerId,
                                   rr.JockeyId,
                                   rr.SaddleclothNumber,
                                   rr.Draw,
                                   rr.Age,
                                   rr.WeightLbs,
                                   rr.WeightText,
                                   rr.FinishPos,
                                   rr.OutcomeCode,
                                   rr.DistanceBeatenText,
                                   rr.SP_Fraction,
                                   rr.SP_Decimal,
                                   rr.FavTag,
                                   rr.OpeningFraction,
                                   rr.TouchedHighFraction,
                                   rr.TouchedLowFraction
                            FROM Race r
                            JOIN Course c ON r.CourseId = c.CourseId
                            JOIN RunnerResult rr ON r.RaceId = rr.RaceId
                            LEFT JOIN Horse h ON rr.HorseId = h.HorseId
                            LEFT JOIN Trainer t ON rr.TrainerId = t.TrainerId
                            LEFT JOIN Jockey j ON rr.JockeyId = j.JockeyId
                            ORDER BY r.RaceDate, r.RaceId, rr.RunnerResultId";

            ISet<string>? identifierKeys = null;
            if (includeIdentifiers)
            {
                identifierKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "HorseName",
                    "HorseId",
                    "TrainerName",
                    "TrainerId",
                    "JockeyName",
                    "JockeyId",
                    "SaddleclothNumber",
                    "Draw"
                };
            }

            var featureState = new FeatureEngineeringState(this, identifierKeys);
            var races = new List<PreparedRace>();
            var rnd = new Random();
            var currentRows = new List<Dictionary<string, object?>>();
            int? currentRaceId = null;
            bool ShouldInclude(int raceId) => includeRaceIds is null || includeRaceIds.Contains(raceId);
            bool ShouldUpdate(int raceId) => stateRaceWhitelist is null || stateRaceWhitelist.Contains(raceId);

            foreach (var record in conn.Query(sql, commandTimeout: 6000, buffered: false))
            {
                var source = (IDictionary<string, object?>)record;
                var row = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                foreach (var kvp in source)
                {
                    row[kvp.Key] = NormalizeDbValue(kvp.Value);
                }

                if (!PreparedDataset.TryGetRequiredInt32(row, "RaceId", out var raceId))
                {
                    continue;
                }

                if (currentRaceId.HasValue && raceId != currentRaceId.Value)
                {
                    Shuffle(currentRows, rnd);
                    int previousRaceId = currentRaceId.Value;
                    bool include = ShouldInclude(previousRaceId);
                    bool update = ShouldUpdate(previousRaceId);
                    if (include || update)
                    {
                        featureState.ProcessRace(currentRows, include, update);
                        if (include)
                        {
                            races.Add(new PreparedRace(previousRaceId, currentRows));
                        }
                    }
                    currentRows = new List<Dictionary<string, object?>>();
                }

                currentRows.Add(row);
                currentRaceId = raceId;
            }

            if (currentRaceId.HasValue && currentRows.Count > 0)
            {
                Shuffle(currentRows, rnd);
                int finalRaceId = currentRaceId.Value;
                bool include = ShouldInclude(finalRaceId);
                bool update = ShouldUpdate(finalRaceId);
                if (include || update)
                {
                    featureState.ProcessRace(currentRows, include, update);
                    if (include)
                    {
                        races.Add(new PreparedRace(finalRaceId, currentRows));
                    }
                }
            }

            return new PreparedDataset(races);
        }
        public virtual PreparedRace? PrepareUpcomingRace(UpcomingRace upcoming, IReadOnlyList<RunnerFlow> flows)
        {
            if (upcoming is null)
                throw new ArgumentNullException(nameof(upcoming));
            if (flows is null)
                throw new ArgumentNullException(nameof(flows));
            if (flows.Count == 0)
            {
                return null;
            }

            using var conn = new SqlConnection(_connectionString);
            conn.Open();

            var raceColumns = PreparedDataset.LoadColumnNames(conn, "Race");
            var runnerColumns = PreparedDataset.LoadColumnNames(conn, "RunnerResult");
            string scheduledOffColumn = raceColumns.Contains("ScheduledOff")
                ? "r.ScheduledOff AS ScheduledOff"
                : "CAST(NULL AS time(0)) AS ScheduledOff";
            string actualOffColumn = raceColumns.Contains("ActualOff")
                ? "r.ActualOff AS ActualOff"
                : "CAST(NULL AS time(0)) AS ActualOff";
            string purseColumn = raceColumns.Contains("Purse")
                ? "r.Purse AS Purse"
                : "CAST(NULL AS decimal(18, 2)) AS Purse";
            string officialRatingColumn = runnerColumns.Contains("OfficialRating")
                ? "rr.OfficialRating AS OfficialRating"
                : "CAST(NULL AS smallint) AS OfficialRating";
            string weightTextColumn = runnerColumns.Contains("WeightText")
                ? "rr.WeightText AS WeightText"
                : "CAST(NULL AS nvarchar(50)) AS WeightText";
            string weightLbsColumn = runnerColumns.Contains("WeightLbs")
                ? "rr.WeightLbs AS WeightLbs"
                : "CAST(NULL AS smallint) AS WeightLbs";
            string ageColumn = runnerColumns.Contains("Age")
                ? "rr.Age AS Age"
                : "CAST(NULL AS smallint) AS Age";
            var sql = $@"SELECT c.Name AS CourseName,
                                   h.Name AS HorseName,
                                   j.Name AS JockeyName,
                                   t.Name AS TrainerName,
                                   r.RaceId,
                                   r.CourseId,
                                   r.RaceDate,
                                   {scheduledOffColumn},
                                   {actualOffColumn},
                                   r.Title,
                                   r.RaceType,
                                   r.Class,
                                   r.AgeRestriction,
                                   r.Surface,
                                   r.Going,
                                   r.DistanceYards,
                                   r.DistanceText,
                                   r.RunnerCount,
                                   r.Status,
                                   r.WinningTimeMs,
                                    {purseColumn},
                                   rr.HorseId,
                                   rr.TrainerId,
                                   rr.JockeyId,
                                   rr.SaddleclothNumber,
                                   rr.Draw,
                                   {ageColumn},
                                   {weightLbsColumn},
                                   {weightTextColumn},
                                   {officialRatingColumn},
                                   rr.FinishPos,
                                   rr.OutcomeCode,
                                   rr.DistanceBeatenText,
                                   rr.SP_Fraction,
                                   rr.SP_Decimal,
                                   rr.FavTag,
                                   rr.OpeningFraction,
                                   rr.TouchedHighFraction,
                                   rr.TouchedLowFraction
                            FROM Race r
                            JOIN Course c ON r.CourseId = c.CourseId
                            JOIN RunnerResult rr ON r.RaceId = rr.RaceId
                            LEFT JOIN Horse h ON rr.HorseId = h.HorseId
                            LEFT JOIN Trainer t ON rr.TrainerId = t.TrainerId
                            LEFT JOIN Jockey j ON rr.JockeyId = j.JockeyId
                            WHERE r.RaceDate < @TargetDate
                            ORDER BY r.RaceDate, r.RaceId, rr.RunnerResultId";

            var identifierKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "HorseName",
                "HorseId",
                "TrainerName",
                "TrainerId",
                "JockeyName",
                "JockeyId",
                "SaddleclothNumber",
                "Draw",
                "SelectionId"
            };

            var featureState = new FeatureEngineeringState(this, identifierKeys);

            var currentRows = new List<Dictionary<string, object?>>();
            int? currentRaceId = null;
            foreach (var record in conn.Query(sql, new { TargetDate = upcoming.RaceDate.Date }, commandTimeout: 6000, buffered: false))
            {
                var source = (IDictionary<string, object?>)record;
                var row = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                foreach (var kvp in source)
                {
                    row[kvp.Key] = NormalizeDbValue(kvp.Value);
                }

                if (!PreparedDataset.TryGetRequiredInt32(row, "RaceId", out var raceId))
                {
                    continue;
                }

                if (currentRaceId.HasValue && raceId != currentRaceId.Value)
                {
                    if (currentRows.Count > 0)
                    {
                        featureState.ProcessRace(currentRows, includeRace: false, updateState: true);
                    }
                    currentRows = new List<Dictionary<string, object?>>();
                }

                currentRows.Add(row);
                currentRaceId = raceId;
            }

            if (currentRaceId.HasValue && currentRows.Count > 0)
            {
                featureState.ProcessRace(currentRows, includeRace: false, updateState: true);
            }

            var syntheticRaceId = CreateSyntheticRaceId(upcoming);
            var syntheticRows = BuildUpcomingRaceRows(conn, upcoming, flows, syntheticRaceId, runnerColumns);
            if (syntheticRows.Count == 0)
            {
                return null;
            }

            featureState.ProcessRace(syntheticRows, includeRace: true, updateState: false);
            return new PreparedRace(syntheticRaceId, syntheticRows);
        }

        private static int CreateSyntheticRaceId(UpcomingRace upcoming)
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + upcoming.RaceDate.GetHashCode();
                if (!string.IsNullOrWhiteSpace(upcoming.MarketId))
                {
                    hash = hash * 31 + StringComparer.OrdinalIgnoreCase.GetHashCode(upcoming.MarketId.Trim());
                }
                if (!string.IsNullOrWhiteSpace(upcoming.Title))
                {
                    hash = hash * 31 + StringComparer.OrdinalIgnoreCase.GetHashCode(upcoming.Title.Trim());
                }

                return unchecked((int)(0x80000000 | ((uint)hash & 0x7FFFFFFF)));
            }
        }

        private List<Dictionary<string, object?>> BuildUpcomingRaceRows(
            SqlConnection conn,
            UpcomingRace upcoming,
            IReadOnlyList<RunnerFlow> flows,
            int raceId,
            IReadOnlyCollection<string> runnerColumns)
        {
            var rows = new List<Dictionary<string, object?>>(flows.Count);
            var (courseId, courseName) = ResolveCourse(conn, upcoming);
            int runnerCount = flows.Count;

            foreach (var flow in flows)
            {
                if (flow == null)
                {
                    continue;
                }

                var horseName = flow.HorseName;
                if (string.IsNullOrWhiteSpace(horseName))
                {
                    continue;
                }

                var row = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
                {
                    ["RaceId"] = raceId,
                    ["CourseId"] = courseId,
                    ["CourseName"] = courseName,
                    ["RaceDate"] = upcoming.RaceDate,
                    ["ScheduledOff"] = upcoming.ScheduledOff,
                    ["ActualOff"] = null,
                    ["Title"] = upcoming.Title,
                    ["RaceType"] = upcoming.RaceType,
                    ["Class"] = upcoming.Class,
                    ["AgeRestriction"] = upcoming.AgeRestriction,
                    ["Surface"] = upcoming.Surface,
                    ["Going"] = upcoming.Going,
                    ["DistanceYards"] = upcoming.DistanceYards,
                    ["DistanceText"] = upcoming.DistanceText,
                    ["RunnerCount"] = upcoming.RunnerCount ?? Math.Min(runnerCount, byte.MaxValue),
                    ["Status"] = null,
                    ["WinningTimeMs"] = null,
                    ["HorseName"] = horseName,
                    ["SelectionId"] = flow.SelectionId,
                    ["JockeyName"] = flow.JockeyName,
                    ["SaddleclothNumber"] = flow.ClothNumber,
                    ["Draw"] = flow.Draw,
                    ["Age"] = null,
                    ["WeightLbs"] = null,
                    ["WeightText"] = null,
                    ["OfficialRating"] = null,
                    ["FinishPos"] = null,
                    ["OutcomeCode"] = null,
                    ["DistanceBeatenText"] = null,
                    ["SP_Fraction"] = null,
                    ["SP_Decimal"] = null,
                    ["FavTag"] = null,
                    ["OpeningFraction"] = null,
                    ["TouchedHighFraction"] = null,
                    ["TouchedLowFraction"] = null,
                    ["BackPrice1"] = flow.BackPrice1,
                    ["BackPrice2"] = flow.BackPrice2,
                    ["BackPrice3"] = flow.BackPrice3,
                    ["LayPrice1"] = flow.LayPrice1,
                    ["LayPrice2"] = flow.LayPrice2,
                    ["LayPrice3"] = flow.LayPrice3
                };

                // Values that depend on historical lookups are populated below when data is available.
                row["Purse"] = null;
                row["TrainerId"] = null;
                row["TrainerName"] = null;

                var horseId = ResolveHorseId(conn, horseName);
                row["HorseId"] = horseId;

                if (!string.IsNullOrWhiteSpace(flow.JockeyName))
                {
                    var jockeyId = ResolveJockeyId(conn, flow.JockeyName!);
                    if (jockeyId.HasValue)
                    {
                        row["JockeyId"] = jockeyId.Value;
                    }
                }

                rows.Add(row);
            }
            PopulateRunnerDefaults(conn, upcoming, runnerColumns, row, horseId);
            return rows;
        }
        private void PopulateRunnerDefaults(
            SqlConnection conn,
            UpcomingRace upcoming,
            IReadOnlyCollection<string> runnerColumns,
            Dictionary<string, object?> row,
            int horseId)
        {
            if (runnerColumns == null)
            {
                return;
            }

            string weightTextColumn = runnerColumns.Contains("WeightText")
                ? "rr.WeightText AS WeightText"
                : "CAST(NULL AS nvarchar(50)) AS WeightText";
            string officialRatingColumn = runnerColumns.Contains("OfficialRating")
                ? "rr.OfficialRating AS OfficialRating"
                : "CAST(NULL AS smallint) AS OfficialRating";
            string ageColumn = runnerColumns.Contains("Age")
                ? "rr.Age AS Age"
                : "CAST(NULL AS smallint) AS Age";
            string weightLbsColumn = runnerColumns.Contains("WeightLbs")
                ? "rr.WeightLbs AS WeightLbs"
                : "CAST(NULL AS smallint) AS WeightLbs";

            const string trainerNameSelect = "t.Name AS TrainerName";

            string sql = $@"SELECT TOP (1)
                                        rr.TrainerId,
                                        {trainerNameSelect},
                                        {ageColumn},
                                        {weightLbsColumn},
                                        {weightTextColumn},
                                        {officialRatingColumn}
                                  FROM RunnerResult rr
                                  JOIN Race r ON r.RaceId = rr.RaceId
                                  LEFT JOIN Trainer t ON rr.TrainerId = t.TrainerId
                                  WHERE rr.HorseId = @HorseId AND r.RaceDate < @TargetDate
                                  ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC";

            var snapshot = conn.QuerySingleOrDefault(sql, new
            {
                HorseId = horseId,
                TargetDate = upcoming.RaceDate.Date
            });

            if (snapshot == null)
            {
                return;
            }

            var dict = (IDictionary<string, object?>)snapshot;

            if (row["TrainerId"] == null && dict.TryGetValue("TrainerId", out var trainerIdObj) && trainerIdObj != null)
            {
                row["TrainerId"] = Convert.ToInt32(trainerIdObj);
            }

            if (row["TrainerName"] == null &&
                dict.TryGetValue("TrainerName", out var trainerNameObj) && trainerNameObj is string trainerName)
            {
                row["TrainerName"] = trainerName;
            }

            if (row["Age"] == null && dict.TryGetValue("Age", out var ageObj) && ageObj != null)
            {
                row["Age"] = Convert.ToInt32(ageObj);
            }

            if (row["WeightLbs"] == null && dict.TryGetValue("WeightLbs", out var weightObj) && weightObj != null)
            {
                row["WeightLbs"] = Convert.ToInt32(weightObj);
            }

            if (row["WeightText"] == null && dict.TryGetValue("WeightText", out var weightTextObj) && weightTextObj is string weightText)
            {
                row["WeightText"] = weightText;
            }

            if (dict.TryGetValue("OfficialRating", out var ratingObj) && ratingObj != null)
            {
                row["OfficialRating"] = Convert.ToInt32(ratingObj);
            }
        }

        private (int CourseId, string? CourseName) ResolveCourse(SqlConnection conn, UpcomingRace upcoming)
        {
            if (!string.IsNullOrWhiteSpace(upcoming.VenueName))
            {
                const string exactSql = "SELECT TOP (1) CourseId, Name FROM Course WHERE Name = @Name ORDER BY CourseId";
                var exact = conn.QuerySingleOrDefault<(int CourseId, string Name)?>(exactSql, new { Name = upcoming.VenueName });
                if (exact.HasValue)
                {
                    return (exact.Value.CourseId, exact.Value.Name);
                }

                const string courseSql = "SELECT CourseId, Name FROM Course";
                var allCourses = conn.Query<(int CourseId, string Name)>(courseSql).ToList();
                var normalizedVenue = NormalizeLookupKey(upcoming.VenueName);
                (int CourseId, string Name)? best = null;
                int bestScore = int.MaxValue;
                foreach (var course in allCourses)
                {
                    var candidate = NormalizeLookupKey(course.Name);
                    int score = 0;
                    if (candidate == normalizedVenue)
                    {
                        score -= 3;
                    }
                    else if (!string.IsNullOrEmpty(candidate) &&
                             (candidate.Contains(normalizedVenue) || normalizedVenue.Contains(candidate)))
                    {
                        score -= 1;
                    }
                    else
                    {
                        score += 1;
                    }

                    if (score < bestScore || (score == bestScore && (!best.HasValue || course.CourseId < best.Value.CourseId)))
                    {
                        best = course;
                        bestScore = score;
                    }
                }

                if (best.HasValue)
                {
                    return (best.Value.CourseId, best.Value.Name);
                }
            }
            Console.WriteLine($"\t\tNo match found in Course.Name for venue '{upcoming.VenueName ?? "<null>"}'; using synthetic course metadata.");
            int syntheticCourseId = GenerateSyntheticId("course:" + (upcoming.VenueName ?? upcoming.MarketId ?? string.Empty));
            return (syntheticCourseId, upcoming.VenueName);
        }

        private int ResolveHorseId(SqlConnection conn, string horseName)
        {
            const string sql = "SELECT TOP (1) HorseId FROM Horse WHERE Name = @Name ORDER BY HorseId";
            var existing = conn.QuerySingleOrDefault<int?>(sql, new { Name = horseName });
            if (existing.HasValue)
            {
                return existing.Value;
            }
            Console.WriteLine($"\t\tNo match found in Horse.Name for '{horseName}'; using synthetic horse identifier.");
            return GenerateSyntheticId("horse:" + horseName);
        }

        private int? ResolveJockeyId(SqlConnection conn, string jockeyName)
        {
            const string sql = "SELECT TOP (1) JockeyId FROM Jockey WHERE Name = @Name ORDER BY JockeyId";
            var existing = conn.QuerySingleOrDefault<int?>(sql, new { Name = jockeyName });
            if (existing.HasValue)
            {
                return existing.Value;
            }
            Console.WriteLine($"\t\tNo match found in Jockey.Name for '{jockeyName}'; jockey history will be unavailable.");
            return null;
        }

        private static int GenerateSyntheticId(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return int.MaxValue;
            }

            unchecked
            {
                int hash = 17;
                hash = hash * 31 + StringComparer.OrdinalIgnoreCase.GetHashCode(value.Trim());
                return 0x60000000 | (hash & 0x0FFFFFFF);
            }
        }

        private static string NormalizeLookupKey(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var lower = value.Trim().ToLowerInvariant();
            lower = System.Text.RegularExpressions.Regex.Replace(lower, "[^a-z0-9]+", " ");
            lower = System.Text.RegularExpressions.Regex.Replace(lower, "\\s+", " ").Trim();
            return lower;
        }
    }
}