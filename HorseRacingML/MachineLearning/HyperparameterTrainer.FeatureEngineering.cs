using Dapper;
using HorseRacingML.Models;
using HorseRacingML.Services;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
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
        private static readonly int[] PaceWindows = { 1, 3, 5, 10 };
        private const int PacePriorStarts = 3;
        private const float FrontRunPrior = 0.10f;
        private const float ProminentPrior = 0.25f;
        private const float ChasedPrior = 0.16f;
        private const float MidfieldPrior = 0.16f;
        private const float HeldUpPrior = 0.12f;
        private const float RearPrior = 0.16f;
        private const float KeenPrior = 0.06f;
        private const float SlowStartPrior = 0.05f;
        private const float FrontRunnerThreshold = 0.40f;
        private const float ProminentRunnerThreshold = 0.50f;
        // Normalised finish position runs from 0 (last) to 1 (winner), so 0.5
        // is the neutral "no information" value. Encoding missing history as 0
        // made a debutant look like a horse that had finished last every time.
        private const float NeutralNormPosition = 0.5f;
        private static readonly HashSet<string> LiveIdentifierKeys = new(StringComparer.OrdinalIgnoreCase)
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
            float Weight,
            RunStyleInfo RunStyle,
            bool Completed,
            int JockeyId,
            int TrainerId,
            int DistanceYards,
            string Headgear,
            string Medication,
            bool HasSectional,
            float SectionalFinishSpeedPct,
            float SectionalLast3FPercentile,
            float SectionalLast3FSeconds,
            float SectionalEarlyPositionNorm,
            float SectionalClosingPositionNorm,
            float SectionalPaceStyle);

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
            private readonly Dictionary<(int courseId, string bucket, string going, int draw), (int starts, int wins)> _drawGoingStats = new();
            private readonly Dictionary<(int courseId, string bucket, string going), (int starts, int wins)> _drawGoingBaselineStats = new();
            private readonly Dictionary<int, (DateTime lastDate, int meetingDay, int raceIndex)> _courseMeetingState = new();
            private readonly Dictionary<(int horseId, int opponentId), (int starts, int wins, float sumNorm)> _headToHead = new();
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
            private readonly Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)> _sireStats = new(StringComparer.Ordinal);
            private readonly Dictionary<(string key, string surface), (int starts, int wins, float sumNorm, float lastNorm)> _sireSurfaceStats = new();
            private readonly Dictionary<(string key, string going), (int starts, int wins, float sumNorm, float lastNorm)> _sireGoingStats = new();
            private readonly Dictionary<(string key, string bucket), (int starts, int wins, float sumNorm, float lastNorm)> _sireDistanceStats = new();
            private readonly Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)> _damStats = new(StringComparer.Ordinal);
            private readonly Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)> _damSireStats = new(StringComparer.Ordinal);

            public FeatureEngineeringState(HyperparameterTrainer trainer, ISet<string>? identifierKeys = null)
            {
                _trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
                _identifierKeys = identifierKeys;
            }

            private readonly record struct RunnerUpdate(
                int HorseId,
                DateTime Date,
                short? Finish,
                float NormFinish,
                bool Won,
                string Going,
                string Surface,
                int CourseId,
                string Bucket,
                int ClassVal,
                string AgeRestriction,
                float Speed,
                float SpeedDiff,
                int Age,
                float Rating,
                float Weight,
                int DistanceYards,
                int Draw,
                int? TrainerId,
                int? JockeyId,
                RunStyleInfo RunStyle,
                string Headgear,
                string Medication,
                bool HasSectional,
                float SectionalFinishSpeedPct,
                float SectionalLast3FPercentile,
                float SectionalLast3FSeconds,
                float SectionalEarlyPositionNorm,
                float SectionalClosingPositionNorm,
                float SectionalPaceStyle,
                string SireKey,
                string DamKey,
                string DamSireKey);

            private readonly record struct SectionalResultInfo(
                bool HasSectional,
                float FinishSpeedPct,
                float Last3FPercentile,
                float Last3FSeconds,
                float EarlyPositionNorm,
                float ClosingPositionNorm,
                float PaceStyle)
            {
                public static readonly SectionalResultInfo Empty = new(false, 0f, 0.5f, 0f, 0.5f, 0.5f, 0f);
            }

            private readonly record struct RunnerPedigreeInfo(
                string SireKey,
                string DamKey,
                string DamSireKey)
            {
                public static readonly RunnerPedigreeInfo Empty = new(string.Empty, string.Empty, string.Empty);
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

                    if (row.TryGetValue("OutcomeCode", out var outcome) && outcome is string outcomeCode)
                    {
                        var code = outcomeCode.Trim();
                        if (code.Equals("NR", StringComparison.OrdinalIgnoreCase) ||
                            code.Equals("WD", StringComparison.OrdinalIgnoreCase))
                        {
                            return true;
                        }
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
                }

                var raceStat = ComputeRaceStats(rows);

                // Meeting context: the going, draw bias and pace bias can move
                // across the days of a meeting as the ground is used. Day 1/2/3
                // and the race index within the meeting are known before the off.
                // The columns are always built so a saved model that needs a
                // block can be scored regardless of the live process's current
                // Training:FeatureBlocks setting. The block switch only decides
                // which columns enter the trained schema.
                bool raceContextEnabled = true;
                int meetingCourseId = 0;
                DateTime meetingDate = default;
                int meetingDay = 1;
                int meetingRaceIndex = 1;
                if (raceContextEnabled && rows.Count > 0)
                {
                    var firstRow = rows[0];
                    if (firstRow.TryGetValue("CourseId", out var courseObj) &&
                        PreparedDataset.TryConvertToInt32(courseObj, out var parsedCourseId))
                    {
                        meetingCourseId = parsedCourseId;
                    }
                    if (firstRow.TryGetValue("RaceDate", out var dateObj) && dateObj is DateTime parsedDate)
                    {
                        meetingDate = parsedDate.Date;
                    }

                    if (meetingCourseId != 0 &&
                        meetingDate != default &&
                        _courseMeetingState.TryGetValue(meetingCourseId, out var meetingState))
                    {
                        if (meetingState.lastDate == meetingDate)
                        {
                            meetingDay = meetingState.meetingDay;
                            meetingRaceIndex = meetingState.raceIndex + 1;
                        }
                        else if (meetingState.lastDate == meetingDate.AddDays(-1))
                        {
                            meetingDay = meetingState.meetingDay + 1;
                            meetingRaceIndex = 1;
                        }
                    }
                }

                void UpdateMeetingState()
                {
                    if (updateState &&
                        raceContextEnabled &&
                        meetingCourseId != 0 &&
                        meetingDate != default)
                    {
                        _courseMeetingState[meetingCourseId] = (meetingDate, meetingDay, meetingRaceIndex);
                    }
                }

                // Pre-off market features. Historical rows use the timestamped
                // Betfair anchor (T-N, normally five minutes before the off)
                // when the importer stored one, and fall back to the opening
                // show otherwise. Live rows use the back price at the same
                // anchor point, falling back to the latest captured snapshot.
                // The final SP is deliberately not used.
                int rowCount = rows.Count;
                var marketPrices = new float[rowCount];
                var marketProbabilities = new float[rowCount];
                var marketRanks = new int[rowCount];
                int pricedRunners = 0;
                float overround = 0f;
                for (int i = 0; i < rowCount; i++)
                {
                    var marketRow = rows[i];
                    float price = ReadHistoricalMarketPrice(marketRow, "MarketAnchorPrice");
                    if (price <= 1f &&
                        marketRow.TryGetValue("MarketDecimalOdds", out var livePrice) &&
                        livePrice != null)
                    {
                        float live = Convert.ToSingle(livePrice, CultureInfo.InvariantCulture);
                        if (live > 1f && float.IsFinite(live))
                        {
                            price = live;
                        }
                    }
                    if (price <= 1f &&
                        marketRow.TryGetValue("OpeningFraction", out var opening) &&
                        opening is string openingText)
                    {
                        price = ParseMarketPrice(openingText);
                    }

                    if (price > 1f && float.IsFinite(price))
                    {
                        marketPrices[i] = price;
                        overround += 1f / price;
                        pricedRunners++;
                    }
                }

                if (overround > 0f)
                {
                    for (int i = 0; i < rowCount; i++)
                    {
                        marketProbabilities[i] = marketPrices[i] > 1f
                            ? (1f / marketPrices[i]) / overround
                            : 0f;
                    }

                    var ranked = Enumerable.Range(0, rowCount)
                        .Where(i => marketPrices[i] > 1f)
                        .OrderByDescending(i => marketProbabilities[i])
                        .ToList();
                    for (int rank = 0; rank < ranked.Count; rank++)
                    {
                        marketRanks[ranked[rank]] = rank + 1;
                    }
                }

                bool marketUsable = pricedRunners >= 2 && overround > 0f;
                float marketMean = marketUsable ? 1f / pricedRunners : 0f;

                // Market movement: historical rows use the opening show and the
                // touched low; live rows use the first and lowest snapshots
                // recorded in RunnerFlow. Both describe the same thing: how far
                // the price has shortened since it was first seen.
                bool marketMovementEnabled = true;

                float marketFavouriteProb = 0f;
                float marketProbEntropy = 0f;
                float marketHerfindahl = 0f;
                float marketProbStdDev = 0f;
                float marketTop2Mass = 0f;
                float marketProbGapTop2 = 0f;
                float marketOverround = marketUsable ? Math.Max(0f, overround - 1f) : 0f;
                if (marketUsable)
                {
                    var pricedProbabilities = new List<float>(pricedRunners);
                    for (int i = 0; i < rowCount; i++)
                    {
                        if (marketProbabilities[i] > 0f)
                        {
                            pricedProbabilities.Add(marketProbabilities[i]);
                        }
                    }

                    if (pricedProbabilities.Count > 0)
                    {
                        pricedProbabilities.Sort((left, right) => right.CompareTo(left));
                        marketFavouriteProb = pricedProbabilities[0];
                        marketTop2Mass = pricedProbabilities.Take(2).Sum();
                        marketProbGapTop2 = pricedProbabilities.Count > 1
                            ? pricedProbabilities[0] - pricedProbabilities[1]
                            : pricedProbabilities[0];

                        double mean = pricedProbabilities.Average();
                        double variance = 0;
                        double entropy = 0;
                        double herfindahl = 0;
                        foreach (var probability in pricedProbabilities)
                        {
                            double diff = probability - mean;
                            variance += diff * diff;
                            if (probability > 0)
                            {
                                entropy -= probability * Math.Log(probability);
                            }
                            herfindahl += probability * probability;
                        }

                        marketProbStdDev = (float)Math.Sqrt(variance / pricedProbabilities.Count);
                        marketHerfindahl = (float)herfindahl;
                        marketProbEntropy = pricedProbabilities.Count > 1
                            ? (float)(entropy / Math.Log(pricedProbabilities.Count))
                            : 0f;
                    }
                }

                void ProcessRows(List<RunnerUpdate> updates)
                {
                    for (int rowIndex = 0; rowIndex < rows.Count; rowIndex++)
                    {
                        var row = rows[rowIndex];
                        AddWeatherBalanceFeatures(row, enabled: true);
                        AddBetfairLiquidityFeatures(row);
                        if (raceContextEnabled)
                        {
                            row["RaceContextMeetingDay"] = meetingDay;
                            row["RaceContextMeetingRaceIndex"] = meetingRaceIndex;
                        }
                        row["MarketImpliedProb"] = marketUsable ? marketProbabilities[rowIndex] : 0f;
                        row["MarketLogOdds"] = marketPrices[rowIndex] > 1f ? MathF.Log(marketPrices[rowIndex]) : 0f;
                        row["MarketProbRankRelative"] = marketUsable && marketRanks[rowIndex] > 0 ? (float)marketRanks[rowIndex] / rowCount : 0f;
                        row["MarketProbDiffFromMean"] = marketMean > 0f ? marketProbabilities[rowIndex] - marketMean : 0f;
                        row["MarketProbMissing"] = !marketUsable || marketPrices[rowIndex] <= 1f;
                        float rowMarketProbability = marketUsable ? marketProbabilities[rowIndex] : 0f;
                        row["MarketLogit"] = rowMarketProbability > 0f && rowMarketProbability < 1f
                            ? Logit(rowMarketProbability)
                            : 0f;
                        row["MarketFavouriteProb"] = marketFavouriteProb;
                        row["MarketProbEntropy"] = marketProbEntropy;
                        row["MarketHerfindahl"] = marketHerfindahl;
                        row["MarketProbStdDev"] = marketProbStdDev;
                        row["MarketTop2Mass"] = marketTop2Mass;
                        row["MarketProbGapTop2"] = marketProbGapTop2;
                        row["MarketOverround"] = marketOverround;
                        if (marketMovementEnabled)
                        {
                            float earlyPrice = ReadHistoricalMarketPrice(row, "OpeningFraction");
                            if (row.TryGetValue("MarketMoveEarlyPrice", out var earlyValue) &&
                                earlyValue is not null)
                            {
                                earlyPrice = Convert.ToSingle(earlyValue, CultureInfo.InvariantCulture);
                            }

                            // Prefer the live snapshot range (already limited to
                            // the T-N cutoff by ResolveFlowPriceRange). For
                            // historical rows the only safe movement is opening
                            // -> anchor; the full-day touched high/low includes
                            // prices observed after the live decision time, so
                            // it is opt-in for ablation only.
                            float anchorPrice = ReadHistoricalMarketPrice(row, "MarketAnchorPrice");
                            float touchedLow = 0f;
                            float touchedHigh = 0f;
                            if (anchorPrice > 1f && earlyPrice > 1f)
                            {
                                // The only two prices known at the decision
                                // time are the opening show and the T-N anchor.
                                // Treat them as a two-point movement range so
                                // backed/drifted flags remain meaningful.
                                touchedLow = Math.Min(earlyPrice, anchorPrice);
                                touchedHigh = Math.Max(earlyPrice, anchorPrice);
                            }

                            if (row.TryGetValue("MarketMoveLowPrice", out var lowValue) &&
                                lowValue is not null)
                            {
                                touchedLow = Convert.ToSingle(lowValue, CultureInfo.InvariantCulture);
                            }
                            if (row.TryGetValue("MarketMoveHighPrice", out var highValue) &&
                                highValue is not null)
                            {
                                touchedHigh = Convert.ToSingle(highValue, CultureInfo.InvariantCulture);
                            }
                            else if (touchedLow <= 1f && _trainer.UseFullDayMarketMovement)
                            {
                                touchedLow = ReadHistoricalMarketPrice(row, "TouchedLowFraction");
                                touchedHigh = ReadHistoricalMarketPrice(row, "TouchedHighFraction");
                            }

                            AddMarketMovementFeatures(
                                row,
                                earlyPrice,
                                touchedLow,
                                touchedHigh,
                                marketUsable && marketRanks[rowIndex] > 0
                                    ? (float)marketRanks[rowIndex] / rowCount
                                    : (float?)null);
                        }
                        int horseId = PreparedDataset.GetRequiredInt32(row, "HorseId");
                        DateTime date = (DateTime)row["RaceDate"];
                        var paceInfo = _trainer.EnablePaceFeatures
                            ? ReadRunStyleInfo(row)
                            : RunStyleInfo.Unknown;
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
                        }

                        // Calendar features are always available. Previously they
                        // were only written when the scheduled off time was
                        // missing, which made them effectively dead features.
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

                        // Common feature block: this must run for every runner,
                        // whether or not the time-of-day fields were available.
                        short? finish = row["FinishPos"] != null ? (short?)Convert.ToInt16(row["FinishPos"]) : null;
                            int raceId = PreparedDataset.GetRequiredInt32(row, "RaceId");
                            int runnerCount = row.TryGetValue("RunnerCount", out var runnerCountObj) &&
                                                PreparedDataset.TryConvertToInt32(runnerCountObj, out var runnerValue)
                                ? runnerValue
                            : raceStat.RunnerCount;
                            row["RunnerCountInv"] = runnerCount > 0 ? 1f / runnerCount : 0f;
                            int? trainerId = row.TryGetValue("TrainerId", out var tObj) &&
                                              PreparedDataset.TryConvertToInt32(tObj, out var trainerValue)
                                ? trainerValue
                                : (int?)null;
                            int? jockeyId = row.TryGetValue("JockeyId", out var jObj) &&
                                             PreparedDataset.TryConvertToInt32(jObj, out var jockeyValue)
                                ? jockeyValue
                            : (int?)null;
                            row["HasJockey"] = jockeyId.HasValue;

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
                            row["OfficialRating"] = ratingMissing ? null : (float?)Convert.ToSingle(ratingObj);
                            row["OfficialRatingRelative"] = rating - raceStat.AvgRating;
                            row["OfficialRatingMissing"] = ratingMissing;
                            row["PurseLog"] = MathF.Log(1f + Math.Max(0f, raceStat.TotalPurse));
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
                            // Field-strength aggregates. These are constant within
                            // a race, so together with OfficialRatingRankRelative
                            // and OfficialRatingVsFieldMax they tell the network
                            // how strong the opposition is, not just how this
                            // runner compares with the average.
                            row["FieldRatingMean"] = raceStat.AvgRating;
                            row["FieldRatingMax"] = raceStat.MaxRating;
                            row["FieldRatingMin"] = raceStat.MinRating;
                            row["FieldRatingRange"] = raceStat.MaxRating - raceStat.MinRating;
                            row["FieldRatingTop3Mean"] = raceStat.Top3RatingMean;
                            row["FieldRatingKnownCount"] = raceStat.KnownRatingCount;
                            row["FieldRatingCoverage"] = raceStat.RunnerCount > 0
                                ? (float)raceStat.KnownRatingCount / raceStat.RunnerCount
                                : 0f;
                            row["OfficialRatingVsFieldMax"] = ratingMissing || raceStat.KnownRatingCount == 0
                                ? 0f
                                : rating - raceStat.MaxRating;
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
                            row["ClassStarts"] = horseClassStat.starts;
                            row["ClassWinRate"] = _trainer.SmoothedWinRate(horseClassStat.wins, horseClassStat.starts);
                            row["ClassAvgNorm"] = horseClassStat.starts > 0
                                ? horseClassStat.sumNorm / horseClassStat.starts
                                : NeutralNormPosition;
                            row["LastClassNormPos"] = horseClassStat.starts > 0
                                ? horseClassStat.lastNorm
                                : NeutralNormPosition;

                            (int trainerId, int classVal) trainerClassKey = default;
                            (int starts, int wins, float sumNorm, float lastNorm) trainerClassStat = default;
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
                            }
                            else
                            {
                                row["TrainerClassWinRate"] = 0f;
                                row["TrainerClassAvgNorm"] = 0f;
                                row["LastTrainerClassNormPos"] = 0f;
                            }

                            (int jockeyId, int classVal) jockeyClassKey = default;
                            (int starts, int wins, float sumNorm, float lastNorm) jockeyClassStat = default;
                            if (jockeyId.HasValue)
                            {
                                jockeyClassKey = (jockeyId.Value, classVal);
                                if (!_jockeyClassStats.TryGetValue(jockeyClassKey, out jockeyClassStat))
                                    jockeyClassStat = (0, 0, 0f, 0f);
                                row["JockeyClassStarts"] = jockeyClassStat.starts;
                                row["JockeyClassWinRate"] = _trainer.SmoothedWinRate(jockeyClassStat.wins, jockeyClassStat.starts);
                                row["JockeyClassAvgNorm"] = jockeyClassStat.starts > 0
                                                    ? jockeyClassStat.sumNorm / jockeyClassStat.starts
                                                    : NeutralNormPosition;
                                row["LastJockeyClassNormPos"] = jockeyClassStat.starts > 0
                                                    ? jockeyClassStat.lastNorm
                                                    : NeutralNormPosition;
                            }
                            else
                            {
                                row["JockeyClassStarts"] = 0;
                                row["JockeyClassWinRate"] = 0f;
                                row["JockeyClassAvgNorm"] = NeutralNormPosition;
                                row["LastJockeyClassNormPos"] = NeutralNormPosition;
                            }
                            if (!_horseHistory.TryGetValue(horseId, out var history))
                            {
                                history = new List<HistoryEntry>();
                                _horseHistory[horseId] = history;
                            }
                            bool hasHistory = history.Count > 0;
                            row["HasHistory"] = hasHistory;
                            row["IsFirstStart"] = !hasHistory;
                            string currentHeadgear = RowString(row, "Headgear");
                            row["HeadgearFlag"] = !string.IsNullOrEmpty(currentHeadgear);
                            row["HeadgearFirstTime"] = hasHistory &&
                                !string.IsNullOrEmpty(currentHeadgear) &&
                                string.IsNullOrEmpty(history[^1].Headgear);
                            row["MedicationFlag"] = !string.IsNullOrEmpty(RowString(row, "Medication"));
                            row["HasSpeedHistory"] = history.Any(h => h.Speed > 0f);
                            row["HasRatingHistory"] = history.Any(h => h.Rating > 0f);
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
                            row["LayoffShort"] = hasHistory && daysSinceLast < 30f;
                            row["LayoffMedium"] = hasHistory && daysSinceLast >= 30f && daysSinceLast <= 90f;
                            row["LayoffLong"] = hasHistory && daysSinceLast > 90f;
                            row["LayoffNormalized"] = daysSinceLast / TypicalRestDays;
                            for (int i = 0; i < PastRaceCount; i++)
                            {
                                var key = $"Last{i + 1}NormPos";
                                row[key] = i < history.Count
                                     ? history[history.Count - 1 - i].NormFinish
                                    : NeutralNormPosition;
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
                            row["RecentImprovement"] = history.Count >= PastRaceCount
                                    ? (float)row["Last1NormPos"] - (float)row[$"Last{PastRaceCount}NormPos"]
                                    : 0f;
                            int careerStarts = history.Count;
                            int careerWins = history.Count(h => h.Won);
                            row["CareerStarts"] = careerStarts;
                            row["IsFirstStart"] = careerStarts == 0;
                            row["HasHistory"] = careerStarts > 0;
                            row["LifetimeWinRate"] = _trainer.SmoothedWinRate(careerWins, careerStarts);

                            // Run style is post-race information, so only prior runs
                            // may contribute. paceInfo belongs to the current race
                            // and is carried into RunnerUpdate for the next race,
                            // never written into this row's own feature vector.
                            if (_trainer.EnablePaceFeatures)
                            {
                                AddRollingPaceFeatures(row, history);
                                row["LastRunStyle"] = history.Count > 0
                                    ? (float)(byte)history[^1].RunStyle.Position
                                    : 0f;
                                row["LastRunStyleKnown"] = history.Count > 0 &&
                                                           history[^1].RunStyle.HasPosition;
                            }

                            bool marketMissingForDisagreement = row.TryGetValue("MarketProbMissing", out var marketMissingValue) &&
                                                                marketMissingValue is bool marketMissingFlag &&
                                                                marketMissingFlag;
                            if (!marketMissingForDisagreement)
                            {
                                float disagreementMarketProbability = Convert.ToSingle(row["MarketImpliedProb"]);
                                float lifetimeWinRate = Convert.ToSingle(row["LifetimeWinRate"]);
                                row["MarketProbMinusLifetimeWinRate"] = disagreementMarketProbability - lifetimeWinRate;
                                row["MarketLogitMinusFormLogit"] = Convert.ToSingle(row["MarketLogit"]) - Logit(lifetimeWinRate);
                            }
                            else
                            {
                                row["MarketProbMinusLifetimeWinRate"] = 0f;
                                row["MarketLogitMinusFormLogit"] = 0f;
                            }
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
                                    row[$"AvgNormPosLast{window}"] = NeutralNormPosition;
                                    row[$"AvgSpeedLast{window}"] = 0f;
                                    row[$"AvgSpeedDiffLast{window}"] = 0f;
                                    row[$"AvgRatingLast{window}"] = 0f;

                                }
                            }
                            // Going performance
                            string going = FeatureText.NormalizeCategory(row["Going"] as string);
                            row["Going"] = going;

                            // Ordinal going and the gap between the official
                            // reading and the weather-implied going. A positive
                            // delta means the track is softer than the weather
                            // alone would suggest (watering, drainage, covers).
                            var goingOrdinal = WeatherFeatureCalculator.GoingIndex(going);
                            row["GoingIndex"] = goingOrdinal;
                            if (goingOrdinal.HasValue &&
                                row.TryGetValue("WeatherGoingIndex", out var weatherGoingObj) &&
                                weatherGoingObj is not null)
                            {
                                row["GoingWeatherDelta"] =
                                    goingOrdinal.Value - Convert.ToDouble(weatherGoingObj, CultureInfo.InvariantCulture);
                            }
                            else
                            {
                                row["GoingWeatherDelta"] = null;
                            }

                            if (!_goingStats.TryGetValue(horseId, out var gDict))
                            {
                                gDict = new();
                                _goingStats[horseId] = gDict;
                            }
                            if (!gDict.TryGetValue(going, out var gStats))
                                gStats = (0, 0, 0f, 0f);
                            row["GoingStarts"] = gStats.starts;
                            row["GoingWinRate"] = _trainer.SmoothedWinRate(gStats.wins, gStats.starts);
                            row["GoingAvgNorm"] = gStats.starts > 0 ? gStats.sumNorm / gStats.starts : NeutralNormPosition;
                            row["LastGoingNormPos"] = gStats.starts > 0 ? gStats.lastNorm : NeutralNormPosition;
                            var goingFeatureKey = going.Replace(" ", "");
                            row[$"LayoffNormalized_{goingFeatureKey}"] = (float)row["LayoffNormalized"];
                            string surface = FeatureText.NormalizeSurface(row["Surface"] as string);
                            row["Surface"] = surface;
                            if (!_surfaceStats.TryGetValue(horseId, out var sDict))
                            {
                                sDict = new();
                                _surfaceStats[horseId] = sDict;
                            }
                            if (!sDict.TryGetValue(surface, out var sStats))
                                sStats = (0, 0, 0f, 0f);
                            row["SurfaceStarts"] = sStats.starts;
                            row["SurfaceWinRate"] = _trainer.SmoothedWinRate(sStats.wins, sStats.starts);
                            row["SurfaceAvgNorm"] = sStats.starts > 0 ? sStats.sumNorm / sStats.starts : NeutralNormPosition;
                            row["LastSurfaceNormPos"] = sStats.starts > 0 ? sStats.lastNorm : NeutralNormPosition;
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
                            row["CourseStarts"] = cStats.starts;
                            row["CourseWinRate"] = _trainer.SmoothedWinRate(cStats.wins, cStats.starts);
                            row["LastCourseNormPos"] = cStats.starts > 0 ? cStats.lastNorm : NeutralNormPosition;
                            string gcKey = going + "_" + courseId;
                            if (!_goingCourseStats.TryGetValue(horseId, out var gcDict))
                            {
                                gcDict = new();
                                _goingCourseStats[horseId] = gcDict;
                            }
                            if (!gcDict.TryGetValue(gcKey, out var gcStats))
                                gcStats = (0, 0, 0f, 0f);
                            row["GoingCourseStarts"] = gcStats.starts;
                            row["GoingCourseWinRate"] = _trainer.SmoothedWinRate(gcStats.wins, gcStats.starts);
                            row["GoingCourseAvgNorm"] = gcStats.starts > 0 ? gcStats.sumNorm / gcStats.starts : NeutralNormPosition;
                            row["LastGoingCourseNormPos"] = gcStats.starts > 0 ? gcStats.lastNorm : NeutralNormPosition;
                            // Age restriction context
                            string ageRes = FeatureText.NormalizeCategory(row["AgeRestriction"] as string);
                            row["AgeRestriction"] = ageRes;
                            if (!_ageStats.TryGetValue(horseId, out var aDict))
                            {
                                aDict = new();
                                _ageStats[horseId] = aDict;
                            }
                            if (!aDict.TryGetValue(ageRes, out var aStats))
                                aStats = (0, 0, 0f, 0f);
                            row["AgeRestrictionStarts"] = aStats.starts;
                            row["AgeRestrictionWinRate"] = _trainer.SmoothedWinRate(aStats.wins, aStats.starts);
                            row["LastAgeRestrictionNormPos"] = aStats.starts > 0 ? aStats.lastNorm : NeutralNormPosition;

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
                            if (raceContextEnabled)
                            {
                                if (courseId > 0)
                                {
                                    var courseCode = courseId.ToString(CultureInfo.InvariantCulture);
                                    row["RaceContextCourse"] = courseCode;
                                    row["RaceContextCourseDistance"] = courseCode + "_" + bucket;
                                }
                                else
                                {
                                    row["RaceContextCourse"] = null;
                                    row["RaceContextCourseDistance"] = null;
                                }

                                var drawGoingKey = (courseId, bucket, going, draw);
                                _drawGoingStats.TryGetValue(drawGoingKey, out var drawGoingStat);
                                var drawGoingBaselineKey = (courseId, bucket, going);
                                _drawGoingBaselineStats.TryGetValue(drawGoingBaselineKey, out var drawGoingBaselineStat);
                                row["RaceContextDrawBiasGoing"] =
                                    _trainer.SmoothedWinRate(drawGoingStat.wins, drawGoingStat.starts) -
                                    _trainer.SmoothedWinRate(drawGoingBaselineStat.wins, drawGoingBaselineStat.starts);
                            }
                            AddFormContextFeatures(
                                row,
                                history,
                                daysSinceLast,
                                jockeyId,
                                trainerId,
                                going,
                                courseId,
                                bucket,
                                distanceYards,
                                classVal);
                            AddTrackGeometryFeatures(
                                row,
                                RowString(row, "CourseName"),
                                RowString(row, "Title"),
                                date);
                            AddDeepFormFeatures(
                                row,
                                history,
                                horseId,
                                rows,
                                daysSinceLast);

                            // Sectional times and pedigree are post-race / pre-race
                            // information respectively. Only a horse's earlier
                            // sectionals enter these features; the current race's
                            // splits are read for the state update below and are
                            // never written into this row's feature vector.
                            var sectionalResult = ReadSectionalResultInfo(row);
                            AddSectionalHistoryFeatures(row, history);
                            var pedigreeInfo = ReadPedigreeInfo(row);
                            AddPedigreeFeatures(row, pedigreeInfo, surface, going, bucket);

                            if (!_distanceBucketStats.TryGetValue(horseId, out var dDict))
                            {
                                dDict = new();
                                _distanceBucketStats[horseId] = dDict;
                            }
                            if (!dDict.TryGetValue(bucket, out var dStats))
                                dStats = (0, 0, 0f, 0f);
                            row["DistanceBucketStarts"] = dStats.starts;
                            row["DistanceBucketWinRate"] = _trainer.SmoothedWinRate(dStats.wins, dStats.starts);
                            row["LastDistanceBucketNormPos"] = dStats.starts > 0 ? dStats.lastNorm : NeutralNormPosition;
                            var gdKey = (horseId, going, bucket);
                            if (!_goingDistanceStats.TryGetValue(gdKey, out var gdStats))
                                gdStats = (0, 0, 0f, 0f);
                            row["GoingDistanceStarts"] = gdStats.starts;
                            row["GoingDistanceWinRate"] = _trainer.SmoothedWinRate(gdStats.wins, gdStats.starts);
                            row["GoingDistanceAvgNorm"] = gdStats.starts > 0 ? gdStats.sumNorm / gdStats.starts : NeutralNormPosition;
                            row["LastGoingDistanceNormPos"] = gdStats.starts > 0 ? gdStats.lastNorm : NeutralNormPosition;
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
                                        : NeutralNormPosition;
                                    var surfaceRecent = recent.Where(h => h.Surface == surface).ToList();
                                    row[$"SurfaceWinRateLast{window}"] = _trainer.SmoothedWinRate(surfaceRecent.Count(h => h.Finish == 1), surfaceRecent.Count);
                                    row[$"SurfaceAvgNormLast{window}"] = surfaceRecent.Count > 0
                                        ? surfaceRecent.Sum(h => h.NormFinish) / surfaceRecent.Count
                                        : NeutralNormPosition;
                                    var courseRecent = recent.Where(h => h.CourseId == courseId).ToList();
                                    row[$"CourseWinRateLast{window}"] = _trainer.SmoothedWinRate(courseRecent.Count(h => h.Finish == 1), courseRecent.Count);
                                    row[$"CourseAvgNormLast{window}"] = courseRecent.Count > 0
                                        ? courseRecent.Sum(h => h.NormFinish) / courseRecent.Count
                                        : NeutralNormPosition;

                                    var bucketRecent = recent.Where(h => h.Bucket == bucket).ToList();
                                    row[$"DistanceBucketWinRateLast{window}"] = _trainer.SmoothedWinRate(bucketRecent.Count(h => h.Finish == 1), bucketRecent.Count);
                                    row[$"DistanceBucketAvgNormLast{window}"] = bucketRecent.Count > 0
                                        ? bucketRecent.Sum(h => h.NormFinish) / bucketRecent.Count
                                        : NeutralNormPosition;
                                }
                                else
                                {
                                    row[$"GoingWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"GoingAvgNormLast{window}"] = NeutralNormPosition;
                                    row[$"SurfaceWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"SurfaceAvgNormLast{window}"] = NeutralNormPosition;
                                    row[$"CourseWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"CourseAvgNormLast{window}"] = NeutralNormPosition;
                                    row[$"DistanceBucketWinRateLast{window}"] = _trainer.SmoothedWinRate(0, 0);
                                    row[$"DistanceBucketAvgNormLast{window}"] = NeutralNormPosition;
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
                                row["JockeySurfaceStarts"] = jsStats.starts;
                                row["JockeySurfaceWinRate"] = _trainer.SmoothedWinRate(jsStats.wins, jsStats.starts);
                                row["JockeySurfaceAvgNorm"] = jsStats.starts > 0 ? jsStats.sumNorm / jsStats.starts : NeutralNormPosition;
                                row["LastJockeySurfaceNormPos"] = jsStats.starts > 0 ? jsStats.lastNorm : NeutralNormPosition;

                                if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jgDict))
                                {
                                    jgDict = new();
                                    _jockeyGoingStats[jockeyId.Value] = jgDict;
                                }
                                if (!jgDict.TryGetValue(going, out var jgStats))
                                    jgStats = (0, 0, 0f, 0f);
                                row["JockeyGoingStarts"] = jgStats.starts;
                                row["JockeyGoingWinRate"] = _trainer.SmoothedWinRate(jgStats.wins, jgStats.starts);
                                row["JockeyGoingAvgNorm"] = jgStats.starts > 0 ? jgStats.sumNorm / jgStats.starts : NeutralNormPosition;
                                row["LastJockeyGoingNormPos"] = jgStats.starts > 0 ? jgStats.lastNorm : NeutralNormPosition;

                                if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jdDict))
                                {
                                    jdDict = new();
                                    _jockeyDistanceStats[jockeyId.Value] = jdDict;
                                }
                                if (!jdDict.TryGetValue(bucket, out var jdStats))
                                    jdStats = (0, 0, 0f, 0f);
                                row["JockeyDistanceBucketStarts"] = jdStats.starts;
                                row["JockeyDistanceBucketWinRate"] = _trainer.SmoothedWinRate(jdStats.wins, jdStats.starts);
                                row["JockeyDistanceBucketAvgNorm"] = jdStats.starts > 0 ? jdStats.sumNorm / jdStats.starts : NeutralNormPosition;
                                row["LastJockeyDistanceBucketNormPos"] = jdStats.starts > 0 ? jdStats.lastNorm : NeutralNormPosition;
                                var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                                if (!_jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStat))
                                    jockeyGoingDistanceStat = (0, 0, 0f, 0f);
                                row["JockeyGoingDistanceStarts"] = jockeyGoingDistanceStat.starts;
                                row["JockeyGoingDistanceWinRate"] = _trainer.SmoothedWinRate(jockeyGoingDistanceStat.wins, jockeyGoingDistanceStat.starts);
                                row["JockeyGoingDistanceAvgNorm"] = jockeyGoingDistanceStat.starts > 0
                                    ? jockeyGoingDistanceStat.sumNorm / jockeyGoingDistanceStat.starts
                                    : NeutralNormPosition;
                                row["LastJockeyGoingDistanceNormPos"] = jockeyGoingDistanceStat.starts > 0
                                    ? jockeyGoingDistanceStat.lastNorm
                                    : NeutralNormPosition;
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
                            }
                            else
                            {
                                row["JockeyWinRate"] = 0f;
                                row[$"JockeyWinRateLast{TrainerJockeyRecentStarts}"] = 0f;
                                row["JockeyWinRateRecentDays"] = 0f;
                                row["JockeySurfaceStarts"] = 0;
                                row["JockeySurfaceWinRate"] = 0f;
                                row["JockeySurfaceAvgNorm"] = NeutralNormPosition;
                                row["LastJockeySurfaceNormPos"] = NeutralNormPosition;
                                row["JockeyGoingStarts"] = 0;
                                row["JockeyGoingWinRate"] = 0f;
                                row["JockeyGoingAvgNorm"] = NeutralNormPosition;
                                row["LastJockeyGoingNormPos"] = NeutralNormPosition;
                                row["JockeyDistanceBucketStarts"] = 0;
                                row["JockeyDistanceBucketWinRate"] = 0f;
                                row["JockeyDistanceBucketAvgNorm"] = NeutralNormPosition;
                                row["LastJockeyDistanceBucketNormPos"] = NeutralNormPosition;
                                row["JockeyGoingDistanceStarts"] = 0;
                                row["JockeyGoingDistanceWinRate"] = 0f;
                                row["JockeyGoingDistanceAvgNorm"] = NeutralNormPosition;
                                row["LastJockeyGoingDistanceNormPos"] = NeutralNormPosition;
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
                                row["JockeyCourseStarts"] = jcStat.starts;
                                row["JockeyCourseWinRate"] = _trainer.SmoothedWinRate(jcStat.wins, jcStat.starts);
                            }
                            else
                            {
                                row["JockeyCourseStarts"] = 0;
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
                            }
                            float normFinish = (finish.HasValue && runnerCount > 1)
                                ? (runnerCount - finish.Value) / (float)(runnerCount - 1)
                                : 0f;

                            updates.Add(new RunnerUpdate(
                                horseId,
                                date,
                                finish,
                                normFinish,
                                finish.HasValue && finish.Value == 1,
                                going,
                                surface,
                                courseId,
                                bucket,
                                classVal,
                                ageRes,
                                runnerSpeed,
                                speedDiff,
                                age,
                                rating,
                                weight,
                                distanceYards,
                                draw,
                                trainerId,
                                jockeyId,
                                paceInfo,
                                RowString(row, "Headgear"),
                                RowString(row, "Medication"),
                                sectionalResult.HasSectional,
                                sectionalResult.FinishSpeedPct,
                                sectionalResult.Last3FPercentile,
                                sectionalResult.Last3FSeconds,
                                sectionalResult.EarlyPositionNorm,
                                sectionalResult.ClosingPositionNorm,
                                sectionalResult.PaceStyle,
                                pedigreeInfo.SireKey,
                                pedigreeInfo.DamKey,
                                pedigreeInfo.DamSireKey));
                        }
                    }
                var updates = new List<RunnerUpdate>(rows.Count);
                ProcessRows(updates);

                if (!includeRace)
                {
                    if (updateState)
                    {
                        ApplyUpdates(updates);
                        UpdateMeetingState();
                    }
                    return;
                }

                // Relative features are only needed for encoded races, not for
                // races that merely advance the rolling state.
                AddRaceRelativeFeatures(
                    rows,
                    marketProbabilities,
                    marketRanks,
                    marketUsable,
                    rowCount,
                    _trainer.EnablePaceFeatures);

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
                }

                if (updateState)
                {
                    // Features were computed above from the state as it existed
                    // before this race. Apply the state changes only after every
                    // runner has been encoded, so a result can never leak into its
                    // own features (and no second feature pass is needed).
                    ApplyUpdates(updates);
                    UpdateMeetingState();
                }

                foreach (var raceRow in rows)
                {
                    TrimRunnerRow(raceRow, _identifierKeys);
                }
            }

            /// <summary>
            /// Adds features that describe a runner relative to the rest of its
            /// race. A pointwise MLP otherwise has to learn comparisons such as
            /// "is this the favourite?" or "is this the best recent form in the
            /// field?" from absolute values, which is much harder.
            /// </summary>
            private static void AddRaceRelativeFeatures(
                List<Dictionary<string, object?>> rows,
                float[] marketProbabilities,
                int[] marketRanks,
                bool marketUsable,
                int rowCount,
                bool enablePaceFeatures)
            {
                for (int i = 0; i < rows.Count; i++)
                {
                    rows[i]["MarketProbRank"] = marketUsable && marketRanks[i] > 0 ? marketRanks[i] : 0;
                    rows[i]["IsFavourite"] = marketUsable && marketRanks[i] == 1;
                    rows[i]["MarketProbGapToNext"] = 0f;
                    rows[i]["MarketProbGapToLeader"] = 0f;
                }

                if (marketUsable && rowCount > 0)
                {
                    var priced = Enumerable.Range(0, rows.Count)
                        .Where(i => marketProbabilities[i] > 0f)
                        .OrderByDescending(i => marketProbabilities[i])
                        .ToList();

                    for (int rank = 0; rank < priced.Count; rank++)
                    {
                        int index = priced[rank];
                        float probability = marketProbabilities[index];
                        float gapToNext = rank + 1 < priced.Count
                            ? probability - marketProbabilities[priced[rank + 1]]
                            : 0f;
                        float gapToLeader = probability - marketProbabilities[priced[0]];
                        rows[index]["MarketProbGapToNext"] = Math.Max(0f, gapToNext);
                        rows[index]["MarketProbGapToLeader"] = Math.Max(0f, gapToLeader);
                    }
                }

                if (enablePaceFeatures)
                {
                    AddFieldFormFeatures(rows);
                    AddPaceFieldFeatures(rows);
                }

                AddRankRelative(rows, "WinRateLast5", "WinRateLast5RankRelative");
                AddRankRelative(rows, "AvgNormPosLast5", "AvgNormPosLast5RankRelative");
                if (enablePaceFeatures)
                {
                    AddRankRelative(rows, "FrontRunRateLast5", "FrontRunRateRankRelative");
                    AddRankRelative(rows, "ProminentRateLast5", "ProminentRateRankRelative");
                    AddRankRelative(rows, "HeldUpRateLast5", "HeldUpRateRankRelative");
                    AddRankRelative(rows, "AvgPaceScoreLast5", "AvgPaceScoreRankRelative");
                }
                AddRankRelative(rows, "JockeyWinRate", "JockeyWinRateRankRelative");
                AddRankRelative(rows, "JockeyGoingDistanceWinRate", "JockeyGoingDistanceWinRateRankRelative");
                AddRankRelative(rows, "CareerStarts", "CareerStartsRankRelative");
                AddRankRelative(rows, "AvgSpeedLast5", "AvgSpeedLast5RankRelative");
                AddRankRelative(rows, "OfficialRating", "OfficialRatingRankRelative");
                AddRankRelative(rows, "ClassWinRate", "ClassWinRateRankRelative");
                AddRankRelative(rows, "GoingWinRate", "GoingWinRateRankRelative");
                AddRankRelative(rows, "DistanceBucketWinRate", "DistanceBucketWinRateRankRelative");
                AddRankRelative(rows, "RecentImprovement", "RecentImprovementRankRelative");

                AddSectionalFieldFeatures(rows);
                AddPedigreeFieldFeatures(rows);
                AddRankRelative(rows, "SectionalAvgFinishSpeedPctLast5", "SectionalFinishSpeedRankRelative");
                AddRankRelative(rows, "SectionalAvgLast3FPercentileLast5", "SectionalLast3FPercentileRankRelative");
                AddRankRelative(rows, "SectionalAvgClosingRankNormLast5", "SectionalClosingRankRankRelative");
                AddRankRelative(rows, "PedigreeSireWinRate", "PedigreeSireWinRateRankRelative");
                AddRankRelative(rows, "PedigreeDamWinRate", "PedigreeDamWinRateRankRelative");

                // Positive values mean the market ranks the runner worse than
                // the form figures do (a potential value signal); negative
                // values mean the market likes it more than the form does.
                AddRankDifference(rows, "MarketProbRankRelative", "WinRateLast5RankRelative", "MarketRankMinusWinRateRank");
                AddRankDifference(rows, "MarketProbRankRelative", "AvgNormPosLast5RankRelative", "MarketRankMinusAvgNormRank");
                AddRankDifference(rows, "MarketProbRankRelative", "ClassWinRateRankRelative", "MarketRankMinusClassRank");
                if (enablePaceFeatures)
                {
                    AddRankDifference(rows, "MarketProbRankRelative", "FrontRunRateRankRelative", "MarketRankMinusFrontRunRank");
                    AddRankDifference(rows, "MarketProbRankRelative", "ProminentRateRankRelative", "MarketRankMinusProminentRank");
                }
            }

            private static void AddFieldFormFeatures(List<Dictionary<string, object?>> rows)
            {
                int count = rows.Count;
                var speeds = new List<float>(count);
                var winRates = new List<float>(count);
                var normPositions = new List<float>(count);

                foreach (var row in rows)
                {
                    if (RowBool(row, "HasSpeedHistory"))
                    {
                        speeds.Add(RowFloat(row, "AvgSpeedLast5"));
                    }

                    winRates.Add(RowFloat(row, "WinRateLast5"));
                    normPositions.Add(RowFloat(row, "AvgNormPosLast5", NeutralNormPosition));
                }

                float maxSpeed = speeds.Count > 0 ? speeds.Max() : 0f;
                float minSpeed = speeds.Count > 0 ? speeds.Min() : 0f;
                float meanSpeed = speeds.Count > 0 ? speeds.Average() : 0f;
                float maxWinRate = winRates.Count > 0 ? winRates.Max() : 0f;
                float minWinRate = winRates.Count > 0 ? winRates.Min() : 0f;
                float meanWinRate = winRates.Count > 0 ? winRates.Average() : 0f;
                float maxNorm = normPositions.Count > 0 ? normPositions.Max() : NeutralNormPosition;
                float minNorm = normPositions.Count > 0 ? normPositions.Min() : NeutralNormPosition;
                float meanNorm = normPositions.Count > 0 ? normPositions.Average() : NeutralNormPosition;

                foreach (var row in rows)
                {
                    row["FieldMaxAvgSpeedLast5"] = maxSpeed;
                    row["FieldMinAvgSpeedLast5"] = minSpeed;
                    row["FieldMeanAvgSpeedLast5"] = meanSpeed;
                    row["FieldMaxWinRateLast5"] = maxWinRate;
                    row["FieldMinWinRateLast5"] = minWinRate;
                    row["FieldMeanWinRateLast5"] = meanWinRate;
                    row["FieldMaxAvgNormPosLast5"] = maxNorm;
                    row["FieldMinAvgNormPosLast5"] = minNorm;
                    row["FieldMeanAvgNormPosLast5"] = meanNorm;
                    row["KnownSpeedRunnerCount"] = speeds.Count;

                    row["OwnAvgSpeedMinusFieldMean"] = RowBool(row, "HasSpeedHistory")
                        ? RowFloat(row, "AvgSpeedLast5") - meanSpeed
                        : 0f;
                    row["OwnWinRateMinusFieldMean"] = RowFloat(row, "WinRateLast5") - meanWinRate;
                    row["OwnAvgNormPosMinusFieldMean"] =
                        RowFloat(row, "AvgNormPosLast5", NeutralNormPosition) - meanNorm;
                }
            }

            private static void AddPaceFieldFeatures(List<Dictionary<string, object?>> rows)
            {
                int count = rows.Count;
                var front = new float[count];
                var prominent = new float[count];
                var heldUp = new float[count];
                var known = new bool[count];

                for (int i = 0; i < count; i++)
                {
                    known[i] = RowFloat(rows[i], "PaceStartsLast5") > 0f;
                    front[i] = RowFloat(rows[i], "FrontRunRateLast5", FrontRunPrior);
                    prominent[i] = RowFloat(rows[i], "ProminentRateLast5", ProminentPrior);
                    heldUp[i] = RowFloat(rows[i], "HeldUpRateLast5", HeldUpPrior);
                }

                var knownIndices = Enumerable.Range(0, count).Where(i => known[i]).ToList();
                int frontCount = knownIndices.Count(i => front[i] >= FrontRunnerThreshold);
                int prominentCount = knownIndices.Count(i => prominent[i] >= ProminentRunnerThreshold);
                int heldUpCount = knownIndices.Count(i => heldUp[i] >= 0.5f);
                float frontSum = knownIndices.Sum(i => front[i]);
                float prominentSum = knownIndices.Sum(i => prominent[i]);
                float heldUpSum = knownIndices.Sum(i => heldUp[i]);
                float maxFront = knownIndices.Count > 0 ? knownIndices.Max(i => front[i]) : 0f;
                float minFront = knownIndices.Count > 0 ? knownIndices.Min(i => front[i]) : 0f;
                float meanFront = knownIndices.Count > 0 ? frontSum / knownIndices.Count : 0f;
                float maxProminent = knownIndices.Count > 0 ? knownIndices.Max(i => prominent[i]) : 0f;
                float minProminent = knownIndices.Count > 0 ? knownIndices.Min(i => prominent[i]) : 0f;
                float meanProminent = knownIndices.Count > 0 ? prominentSum / knownIndices.Count : 0f;
                float maxHeldUp = knownIndices.Count > 0 ? knownIndices.Max(i => heldUp[i]) : 0f;
                float minHeldUp = knownIndices.Count > 0 ? knownIndices.Min(i => heldUp[i]) : 0f;
                float meanHeldUp = knownIndices.Count > 0 ? heldUpSum / knownIndices.Count : 0f;

                for (int i = 0; i < count; i++)
                {
                    bool ownKnown = known[i];
                    float ownFront = front[i];

                    rows[i]["FrontRunnerCount"] = frontCount;
                    rows[i]["ProminentRunnerCount"] = prominentCount;
                    rows[i]["HeldUpRunnerCount"] = heldUpCount;
                    rows[i]["KnownPaceRunnerCount"] = knownIndices.Count;
                    rows[i]["UnknownPaceRunnerCount"] = count - knownIndices.Count;
                    rows[i]["LoneSpeedFlag"] = frontCount == 1;
                    rows[i]["FieldPaceContention"] = knownIndices.Count > 0
                        ? (float)frontCount / knownIndices.Count
                        : 0f;
                    rows[i]["FieldFrontRunSum"] = frontSum;
                    rows[i]["FieldMaxFrontRunRate"] = maxFront;
                    rows[i]["FieldMinFrontRunRate"] = minFront;
                    rows[i]["FieldAvgFrontRunRate"] = meanFront;
                    rows[i]["FieldProminentSum"] = prominentSum;
                    rows[i]["FieldMaxProminentRate"] = maxProminent;
                    rows[i]["FieldMinProminentRate"] = minProminent;
                    rows[i]["FieldAvgProminentRate"] = meanProminent;
                    rows[i]["FieldHeldUpSum"] = heldUpSum;
                    rows[i]["FieldMaxHeldUpRate"] = maxHeldUp;
                    rows[i]["FieldMinHeldUpRate"] = minHeldUp;
                    rows[i]["FieldAvgHeldUpRate"] = meanHeldUp;
                    rows[i]["PacePressure"] = ownKnown ? frontSum - ownFront : frontSum;
                    rows[i]["OwnFrontRunMinusFieldMean"] = ownKnown ? ownFront - meanFront : 0f;
                    rows[i]["IsFrontRunner"] = ownKnown && ownFront >= FrontRunnerThreshold;
                    rows[i]["IsLoneFrontRunner"] = ownKnown &&
                                                   ownFront >= FrontRunnerThreshold &&
                                                   frontCount == 1;
                }
            }

            private static void AddSectionalFieldFeatures(List<Dictionary<string, object?>> rows)
            {
                var known = rows.Where(row => RowBool(row, "SectionalHasHistory")).ToList();
                int knownCount = known.Count;

                var finishSpeeds = known.Select(row => RowFloat(row, "SectionalAvgFinishSpeedPctLast5")).ToList();
                var percentiles = known.Select(row => RowFloat(row, "SectionalAvgLast3FPercentileLast5", 0.5f)).ToList();
                var closingRanks = known.Select(row => RowFloat(row, "SectionalAvgClosingRankNormLast5", 0.5f)).ToList();
                var paceStyles = known.Select(row => RowFloat(row, "SectionalAvgPaceStyleLast5")).ToList();

                float finishSpeedMean = finishSpeeds.Count > 0 ? finishSpeeds.Average() : 0f;
                float finishSpeedMax = finishSpeeds.Count > 0 ? finishSpeeds.Max() : 0f;
                float finishSpeedMin = finishSpeeds.Count > 0 ? finishSpeeds.Min() : 0f;
                float percentileMean = percentiles.Count > 0 ? percentiles.Average() : 0.5f;
                float percentileMax = percentiles.Count > 0 ? percentiles.Max() : 0.5f;
                float percentileMin = percentiles.Count > 0 ? percentiles.Min() : 0.5f;
                float closingMean = closingRanks.Count > 0 ? closingRanks.Average() : 0.5f;
                float paceStyleMean = paceStyles.Count > 0 ? paceStyles.Average() : 0f;

                foreach (var row in rows)
                {
                    bool ownKnown = RowBool(row, "SectionalHasHistory");
                    float ownFinishSpeed = RowFloat(row, "SectionalAvgFinishSpeedPctLast5");
                    float ownPercentile = RowFloat(row, "SectionalAvgLast3FPercentileLast5", 0.5f);
                    float ownClosing = RowFloat(row, "SectionalAvgClosingRankNormLast5", 0.5f);
                    float ownPaceStyle = RowFloat(row, "SectionalAvgPaceStyleLast5");

                    row["SectionalFieldKnownCount"] = knownCount;
                    row["SectionalFieldAvgFinishSpeedPct"] = finishSpeedMean;
                    row["SectionalFieldMaxFinishSpeedPct"] = finishSpeedMax;
                    row["SectionalFieldMinFinishSpeedPct"] = finishSpeedMin;
                    row["SectionalFieldAvgLast3FPercentile"] = percentileMean;
                    row["SectionalFieldMaxLast3FPercentile"] = percentileMax;
                    row["SectionalFieldMinLast3FPercentile"] = percentileMin;
                    row["SectionalFieldAvgClosingRankNorm"] = closingMean;
                    row["SectionalFieldAvgPaceStyle"] = paceStyleMean;
                    row["SectionalOwnFinishSpeedMinusFieldMean"] = ownKnown
                        ? ownFinishSpeed - finishSpeedMean
                        : 0f;
                    row["SectionalOwnLast3FPercentileMinusFieldMean"] = ownKnown
                        ? ownPercentile - percentileMean
                        : 0f;
                    row["SectionalOwnClosingRankMinusFieldMean"] = ownKnown
                        ? ownClosing - closingMean
                        : 0f;
                    row["SectionalOwnPaceStyleMinusFieldMean"] = ownKnown
                        ? ownPaceStyle - paceStyleMean
                        : 0f;
                }
            }

            private static void AddPedigreeFieldFeatures(List<Dictionary<string, object?>> rows)
            {
                var known = rows.Where(row => RowBool(row, "PedigreeKnown")).ToList();
                int knownCount = known.Count;

                var sireRates = known
                    .Select(row => RowNullableFloat(row, "PedigreeSireWinRate"))
                    .Where(value => value.HasValue)
                    .Select(value => value!.Value)
                    .ToList();
                var damRates = known
                    .Select(row => RowNullableFloat(row, "PedigreeDamWinRate"))
                    .Where(value => value.HasValue)
                    .Select(value => value!.Value)
                    .ToList();

                float sireMean = sireRates.Count > 0
                    ? sireRates.Average()
                    : 0f;
                float damMean = damRates.Count > 0
                    ? damRates.Average()
                    : 0f;

                foreach (var row in rows)
                {
                    float? ownSire = RowNullableFloat(row, "PedigreeSireWinRate");
                    float? ownDam = RowNullableFloat(row, "PedigreeDamWinRate");
                    row["PedigreeFieldKnownCount"] = knownCount;
                    row["PedigreeFieldSireWinRateMean"] = sireMean;
                    row["PedigreeFieldDamWinRateMean"] = damMean;
                    row["PedigreeOwnSireWinRateMinusFieldMean"] = ownSire.HasValue
                        ? ownSire.Value - sireMean
                        : 0f;
                    row["PedigreeOwnDamWinRateMinusFieldMean"] = ownDam.HasValue
                        ? ownDam.Value - damMean
                        : 0f;
                }
            }

            private static void AddRankDifference(
                List<Dictionary<string, object?>> rows,
                string leftKey,
                string rightKey,
                string destinationKey)
            {
                foreach (var row in rows)
                {
                    if (row.TryGetValue("MarketProbMissing", out var marketMissing) &&
                        marketMissing is bool isMissing &&
                        isMissing)
                    {
                        row[destinationKey] = 0f;
                        continue;
                    }

                    if (!row.TryGetValue(leftKey, out var leftValue) ||
                        !row.TryGetValue(rightKey, out var rightValue) ||
                        leftValue is null ||
                        rightValue is null)
                    {
                        row[destinationKey] = 0f;
                        continue;
                    }

                    try
                    {
                        row[destinationKey] = Convert.ToSingle(leftValue, CultureInfo.InvariantCulture) -
                                             Convert.ToSingle(rightValue, CultureInfo.InvariantCulture);
                    }
                    catch
                    {
                        row[destinationKey] = 0f;
                    }
                }
            }

            private static void AddRankRelative(
                List<Dictionary<string, object?>> rows,
                string sourceKey,
                string destinationKey)
            {
                int count = rows.Count;
                for (int i = 0; i < count; i++)
                {
                    rows[i][destinationKey] = NeutralNormPosition;
                }

                var values = new List<(int Index, float Value)>(count);
                for (int i = 0; i < count; i++)
                {
                    if (!rows[i].TryGetValue(sourceKey, out var raw) || raw is null)
                    {
                        continue;
                    }

                    try
                    {
                        float value = Convert.ToSingle(raw, CultureInfo.InvariantCulture);
                        if (float.IsFinite(value))
                        {
                            values.Add((i, value));
                        }
                    }
                    catch
                    {
                        // Leave the neutral value in place.
                    }
                }

                if (values.Count <= 1)
                {
                    return;
                }

                values.Sort((left, right) =>
                {
                    int comparison = right.Value.CompareTo(left.Value);
                    return comparison != 0 ? comparison : left.Index.CompareTo(right.Index);
                });
                int n = values.Count;
                for (int rank = 0; rank < n; rank++)
                {
                    rows[values[rank].Index][destinationKey] = (rank + 1f) / n;
                }
            }

            /// <summary>
            /// Form-context features that the flat rolling averages miss:
            /// non-finishers are currently encoded as "last", a long layoff means
            /// different things for different horses, and a jockey/trainer change
            /// is a signal in its own right.
            /// </summary>
            private void AddFormContextFeatures(
                Dictionary<string, object?> row,
                List<HistoryEntry> history,
                float daysSinceLastRace,
                int? jockeyId,
                int? trainerId,
                string going,
                int courseId,
                string bucket,
                int distanceYards,
                int classVal)
            {
                bool hasHistory = history.Count > 0;
                bool hasCompletedHistory = history.Any(h => h.Completed);
                row["FormHasCompletedHistory"] = hasCompletedHistory;
                row["FormCompletedStarts"] = history.Count(h => h.Completed);
                row["FormLastRunNonFinisher"] = hasHistory && !history[^1].Completed;
                row["FormLastRunWasWin"] = hasHistory && history[^1].Won;
                row["FormLastRunPlaced"] = hasHistory &&
                    history[^1].Finish.HasValue &&
                    history[^1].Finish!.Value <= 3;

                int last5 = Math.Min(5, history.Count);
                int last10 = Math.Min(10, history.Count);
                int nonFinishes5 = last5 > 0
                    ? history.Skip(history.Count - last5).Count(h => !h.Completed)
                    : 0;
                int nonFinishes10 = last10 > 0
                    ? history.Skip(history.Count - last10).Count(h => !h.Completed)
                    : 0;
                row["FormNonFinishRateLast5"] = last5 > 0 ? (float)nonFinishes5 / last5 : 0f;
                row["FormNonFinishRateLast10"] = last10 > 0 ? (float)nonFinishes10 / last10 : 0f;

                var completedRecent = history
                    .Where(h => h.Completed)
                    .TakeLast(5)
                    .ToList();
                row["FormAvgNormPosCompletedLast5"] = completedRecent.Count > 0
                    ? completedRecent.Sum(h => h.NormFinish) / completedRecent.Count
                    : NeutralNormPosition;

                if (history.Count >= 2)
                {
                    double totalGap = 0d;
                    int gapCount = 0;
                    for (int i = 1; i < history.Count; i++)
                    {
                        double gap = (history[i].Date - history[i - 1].Date).TotalDays;
                        if (gap > 0d)
                        {
                            totalGap += gap;
                            gapCount++;
                        }
                    }

                    float averageGap = gapCount > 0 ? (float)(totalGap / gapCount) : 0f;
                    row["FormAvgDaysBetweenRuns"] = averageGap;
                    row["FormLayoffVsTypical"] = averageGap > 0f
                        ? Math.Clamp(daysSinceLastRace / averageGap, 0f, 10f)
                        : 1f;
                }
                else
                {
                    row["FormAvgDaysBetweenRuns"] = 0f;
                    row["FormLayoffVsTypical"] = 1f;
                }

                row["FormIsSecondCareerStart"] = history.Count == 1;

                int lastJockeyId = hasHistory ? history[^1].JockeyId : 0;
                bool jockeyChanged = jockeyId.HasValue &&
                    lastJockeyId > 0 &&
                    jockeyId.Value != lastJockeyId;
                row["FormJockeyChanged"] = jockeyChanged;
                float currentJockeyRate = jockeyId.HasValue &&
                    _jockeyStats.TryGetValue(jockeyId.Value, out var currentJockeyStat)
                        ? _trainer.SmoothedWinRate(currentJockeyStat.Wins, currentJockeyStat.Starts)
                        : 0.5f;
                float previousJockeyRate = lastJockeyId > 0 &&
                    _jockeyStats.TryGetValue(lastJockeyId, out var previousJockeyStat)
                        ? _trainer.SmoothedWinRate(previousJockeyStat.Wins, previousJockeyStat.Starts)
                        : 0.5f;
                row["FormJockeyWinRateDelta"] = jockeyChanged
                    ? currentJockeyRate - previousJockeyRate
                    : 0f;

                int lastTrainerId = hasHistory ? history[^1].TrainerId : 0;
                bool trainerChanged = trainerId.HasValue &&
                    lastTrainerId > 0 &&
                    trainerId.Value != lastTrainerId;
                row["FormTrainerChanged"] = trainerChanged;
                float currentTrainerRate = trainerId.HasValue &&
                    _trainerStats.TryGetValue(trainerId.Value, out var currentTrainerStat)
                        ? _trainer.SmoothedWinRate(currentTrainerStat.Wins, currentTrainerStat.Starts)
                        : 0.5f;
                float previousTrainerRate = lastTrainerId > 0 &&
                    _trainerStats.TryGetValue(lastTrainerId, out var previousTrainerStat)
                        ? _trainer.SmoothedWinRate(previousTrainerStat.Wins, previousTrainerStat.Starts)
                        : 0.5f;
                row["FormTrainerWinRateDelta"] = trainerChanged
                    ? currentTrainerRate - previousTrainerRate
                    : 0f;

                row["FormLastRunSameGoing"] = hasHistory && history[^1].Going == going;
                row["FormLastRunSameCourse"] = hasHistory && history[^1].CourseId == courseId;
                row["FormLastRunSameDistance"] = hasHistory && history[^1].DistanceYards == distanceYards;

                int lastClass = hasHistory ? history[^1].RaceClass : 0;
                row["FormClassDrop"] = hasHistory && classVal > 0 && lastClass > 0 && classVal > lastClass;
                row["FormClassRise"] = hasHistory && classVal > 0 && lastClass > 0 && classVal < lastClass;

                var wonClasses = history
                    .Where(h => h.Won && h.RaceClass > 0)
                    .Select(h => h.RaceClass)
                    .ToList();
                int bestClassWon = wonClasses.Count > 0 ? wonClasses.Min() : 0;
                row["FormBestClassWon"] = bestClassWon;
                row["FormClassVsBestWon"] = bestClassWon > 0 && classVal > 0
                    ? classVal - bestClassWon
                    : 0;
            }

            private static void AddTrackGeometryFeatures(
                Dictionary<string, object?> row,
                string courseName,
                string title,
                DateTime raceDate)
            {
                bool known = CourseGeometryCatalog.TryGet(courseName, out var geometry);
                row["TrackGeometryKnown"] = known;
                row["TrackGeometryLeftHanded"] = known && geometry.LeftHanded;
                row["TrackGeometryRightHanded"] = known && geometry.RightHanded;
                row["TrackGeometryStraight"] = known && geometry.StraightCourse;
                row["TrackGeometryRunInNormalized"] = known
                    ? Math.Clamp(geometry.RunInYards / 300.0, 0.0, 1.0)
                    : 0.0;
                row["TrackGeometryFestivalRace"] =
                    CourseGeometryCatalog.LooksLikeFestivalRace(title);
                row["TrackGeometryFestivalMeeting"] =
                    known && geometry.FestivalMonths.Contains(raceDate.Month);
            }

            private void AddDeepFormFeatures(
                Dictionary<string, object?> row,
                List<HistoryEntry> history,
                int horseId,
                List<Dictionary<string, object?>> rows,
                float daysSinceLastRace)
            {
                bool secondUpAfterLayoff = false;
                if (history.Count >= 2)
                {
                    double gap = (history[^1].Date - history[^2].Date).TotalDays;
                    secondUpAfterLayoff = gap >= 60d;
                }
                row["DeepFormSecondUpAfterLayoff"] = secondUpAfterLayoff;

                float peakSpeed = history.Count > 0 ? history.Max(h => h.Speed) : 0f;
                var recent = history
                    .Where(h => h.Speed > 0f)
                    .TakeLast(5)
                    .ToList();
                float recentAverage = recent.Count > 0
                    ? recent.Average(h => h.Speed)
                    : 0f;
                float lastSpeed = history.Count > 0 ? history[^1].Speed : 0f;
                row["DeepFormPeakSpeedRatio"] = peakSpeed > 0f
                    ? recentAverage / peakSpeed
                    : 0f;
                row["DeepFormLastRunSpeedRatio"] = peakSpeed > 0f
                    ? lastSpeed / peakSpeed
                    : 0f;
                row["DeepFormBounceFlag"] = history.Count >= 2 &&
                    peakSpeed > 0f &&
                    lastSpeed >= peakSpeed * 0.98f &&
                    daysSinceLastRace <= 28f;

                int starts = 0;
                int wins = 0;
                float sumNorm = 0f;
                var seenOpponents = new HashSet<int>();
                foreach (var opponentRow in rows)
                {
                    if (!PreparedDataset.TryGetRequiredInt32(
                            opponentRow,
                            "HorseId",
                            out var opponentId) ||
                        opponentId == horseId ||
                        !seenOpponents.Add(opponentId))
                    {
                        continue;
                    }

                    if (_headToHead.TryGetValue((horseId, opponentId), out var h2h))
                    {
                        starts += h2h.starts;
                        wins += h2h.wins;
                        sumNorm += h2h.sumNorm;
                    }
                }

                row["DeepFormH2HStarts"] = starts;
                row["DeepFormH2HWins"] = wins;
                row["DeepFormH2HWinRate"] = _trainer.SmoothedWinRate(wins, starts);
                row["DeepFormH2HNormPos"] = starts > 0
                    ? sumNorm / starts
                    : NeutralNormPosition;
            }

            private void UpdateHeadToHead(
                int horseId,
                int opponentId,
                bool won,
                float normFinish)
            {
                var key = (horseId, opponentId);
                _headToHead.TryGetValue(key, out var stats);
                stats.starts++;
                if (won)
                {
                    stats.wins++;
                }
                stats.sumNorm += normFinish;
                _headToHead[key] = stats;
            }

            private static void AddRollingPaceFeatures(
                Dictionary<string, object?> row,
                List<HistoryEntry> history)
            {
                bool hasPaceHistory = history.Any(h => h.RunStyle.HasPosition);
                row["HasPaceHistory"] = hasPaceHistory;

                foreach (var window in PaceWindows)
                {
                    int count = Math.Min(window, history.Count);
                    if (count <= 0)
                    {
                        row[$"FrontRunRateLast{window}"] = FrontRunPrior;
                        row[$"ProminentRateLast{window}"] = ProminentPrior;
                        row[$"ChasedRateLast{window}"] = ChasedPrior;
                        row[$"MidfieldRateLast{window}"] = MidfieldPrior;
                        row[$"HeldUpRateLast{window}"] = HeldUpPrior;
                        row[$"RearRateLast{window}"] = RearPrior;
                        row[$"KeenRateLast{window}"] = KeenPrior;
                        row[$"SlowStartRateLast{window}"] = SlowStartPrior;
                        row[$"AvgPaceScoreLast{window}"] = 0.5f;
                        row[$"PaceStartsLast{window}"] = 0;
                        continue;
                    }

                    var recent = history.GetRange(history.Count - count, count);
                    var known = recent.Where(h => h.RunStyle.HasPosition).ToList();
                    int starts = known.Count;
                    int led = known.Count(h => h.RunStyle.Position == EarlyPosition.Led);
                    int prominent = known.Count(h =>
                        h.RunStyle.Position == EarlyPosition.Led ||
                        h.RunStyle.Position == EarlyPosition.Pressed);
                    int chased = known.Count(h => h.RunStyle.Position == EarlyPosition.Chased);
                    int midfield = known.Count(h => h.RunStyle.Position == EarlyPosition.Midfield);
                    int rear = known.Count(h => h.RunStyle.Position == EarlyPosition.Rear);
                    int heldUp = known.Count(h => h.RunStyle.HeldUp);
                    int keen = known.Count(h => h.RunStyle.Keen);
                    int slowStart = known.Count(h => h.RunStyle.SlowStart);

                    row[$"FrontRunRateLast{window}"] = PaceRate(led, starts, FrontRunPrior);
                    row[$"ProminentRateLast{window}"] = PaceRate(prominent, starts, ProminentPrior);
                    row[$"ChasedRateLast{window}"] = PaceRate(chased, starts, ChasedPrior);
                    row[$"MidfieldRateLast{window}"] = PaceRate(midfield, starts, MidfieldPrior);
                    row[$"HeldUpRateLast{window}"] = PaceRate(heldUp, starts, HeldUpPrior);
                    row[$"RearRateLast{window}"] = PaceRate(rear, starts, RearPrior);
                    row[$"KeenRateLast{window}"] = PaceRate(keen, starts, KeenPrior);
                    row[$"SlowStartRateLast{window}"] = PaceRate(slowStart, starts, SlowStartPrior);
                    row[$"AvgPaceScoreLast{window}"] = starts > 0
                        ? known.Sum(h => h.RunStyle.PaceScore) / starts
                        : 0.5f;
                    row[$"PaceStartsLast{window}"] = starts;
                }
            }

            private static float PaceRate(int successes, int starts, float prior)
                => (successes + prior * PacePriorStarts) / (starts + PacePriorStarts);

            private static RunStyleInfo ReadRunStyleInfo(Dictionary<string, object?> row)
            {
                if (row.TryGetValue("EarlyPosition", out var positionValue) &&
                    positionValue is not null &&
                    PreparedDataset.TryConvertToInt32(positionValue, out var position) &&
                    position >= 0 &&
                    position <= 5)
                {
                    return new RunStyleInfo(
                        (EarlyPosition)position,
                        RowBool(row, "HeldUp"),
                        RowBool(row, "Keen"),
                        RowBool(row, "SlowStart"));
                }

                if (row.TryGetValue("Comment", out var commentValue) && commentValue is string comment)
                {
                    return RunStyleParser.Parse(comment);
                }

                return RunStyleInfo.Unknown;
            }

            private static bool RowBool(Dictionary<string, object?> row, string key)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    return false;
                }

                if (value is bool boolean)
                {
                    return boolean;
                }

                return PreparedDataset.TryConvertToInt32(value, out var numeric) && numeric != 0;
            }

            private static string RowString(Dictionary<string, object?> row, string key)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    return string.Empty;
                }

                return value.ToString()?.Trim() ?? string.Empty;
            }

            private static float RowFloat(Dictionary<string, object?> row, string key, float fallback = 0f)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    return fallback;
                }

                try
                {
                    return Convert.ToSingle(value, CultureInfo.InvariantCulture);
                }
                catch
                {
                    return fallback;
                }
            }

            private static float? RowNullableFloat(Dictionary<string, object?> row, string key)
            {
                if (!row.TryGetValue(key, out var value) || value is null)
                {
                    return null;
                }

                try
                {
                    float parsed = Convert.ToSingle(value, CultureInfo.InvariantCulture);
                    return float.IsFinite(parsed) ? parsed : (float?)null;
                }
                catch
                {
                    return null;
                }
            }

            private static SectionalResultInfo ReadSectionalResultInfo(Dictionary<string, object?> row)
            {
                int sectionCount = (int)RowFloat(row, "CurrentSectionalCount");
                if (sectionCount <= 0)
                {
                    return SectionalResultInfo.Empty;
                }

                float? finishSpeedPct = RowNullableFloat(row, "CurrentSectionalFinishSpeedPct");
                float? last3FPercentile = RowNullableFloat(row, "CurrentSectionalLast3FPercentile");
                float? last3FSeconds = RowNullableFloat(row, "CurrentSectionalLast3FSeconds");
                if (!finishSpeedPct.HasValue && !last3FPercentile.HasValue && !last3FSeconds.HasValue)
                {
                    return SectionalResultInfo.Empty;
                }

                return new SectionalResultInfo(
                    true,
                    finishSpeedPct ?? 0f,
                    last3FPercentile ?? 0.5f,
                    last3FSeconds ?? 0f,
                    RowNullableFloat(row, "CurrentSectionalEarlyPositionNorm") ?? 0.5f,
                    RowNullableFloat(row, "CurrentSectionalClosingPositionNorm") ?? 0.5f,
                    RowNullableFloat(row, "CurrentSectionalPaceStyle") ?? 0f);
            }

            private static RunnerPedigreeInfo ReadPedigreeInfo(Dictionary<string, object?> row)
            {
                string sireKey = NormalizePedigreeKey(RowString(row, "SireName"));
                string damNameKey = NormalizePedigreeKey(RowString(row, "DamName"));
                string damSireKey = NormalizePedigreeKey(RowString(row, "DamSireName"));
                string damKey = string.IsNullOrEmpty(damNameKey)
                    ? string.Empty
                    : damNameKey + "|" + damSireKey;

                return new RunnerPedigreeInfo(sireKey, damKey, damSireKey);
            }

            private static string NormalizePedigreeKey(string? value)
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    return string.Empty;
                }

                var normalized = FeatureText.NormalizeHorseName(value);
                if (string.IsNullOrEmpty(normalized))
                {
                    return string.Empty;
                }

                var builder = new System.Text.StringBuilder(normalized.Length);
                foreach (char character in normalized)
                {
                    if (char.IsLetterOrDigit(character) || character == ' ')
                    {
                        builder.Append(character);
                    }
                }

                return builder.ToString().Trim();
            }

            private static void AddSectionalHistoryFeatures(
                Dictionary<string, object?> row,
                List<HistoryEntry> history)
            {
                var sectionalHistory = history.Where(entry => entry.HasSectional).ToList();
                int count = sectionalHistory.Count;

                row["SectionalHasHistory"] = count > 0;
                row["SectionalStarts"] = count;
                if (count == 0)
                {
                    row["SectionalLastFinishSpeedPct"] = 0f;
                    row["SectionalAvgFinishSpeedPctLast3"] = 0f;
                    row["SectionalAvgFinishSpeedPctLast5"] = 0f;
                    row["SectionalBestFinishSpeedPctLast5"] = 0f;
                    row["SectionalFinishSpeedTrendLast5"] = 0f;
                    row["SectionalLast3FSecondsLast"] = 0f;
                    row["SectionalAvgLast3FSecondsLast5"] = 0f;
                    row["SectionalLast3FPercentileLast"] = 0.5f;
                    row["SectionalAvgLast3FPercentileLast3"] = 0.5f;
                    row["SectionalAvgLast3FPercentileLast5"] = 0.5f;
                    row["SectionalBestLast3FPercentileLast5"] = 0.5f;
                    row["SectionalClosingRankNormLast"] = 0.5f;
                    row["SectionalAvgClosingRankNormLast5"] = 0.5f;
                    row["SectionalEarlyRankNormLast"] = 0.5f;
                    row["SectionalAvgEarlyRankNormLast5"] = 0.5f;
                    row["SectionalPaceStyleLast"] = 0f;
                    row["SectionalAvgPaceStyleLast5"] = 0f;

                    // Null sources are skipped by AddRankRelative, so unknown
                    // sectional histories stay neutral rather than being ranked
                    // as the worst in the field.
                    row["SectionalFinishSpeedRankRelative"] = null;
                    row["SectionalLast3FPercentileRankRelative"] = null;
                    row["SectionalClosingRankRankRelative"] = null;
                    return;
                }

                var last = sectionalHistory[^1];
                var recent3 = sectionalHistory.Skip(Math.Max(0, count - 3)).ToList();
                var recent5 = sectionalHistory.Skip(Math.Max(0, count - 5)).ToList();
                float avgFinishSpeed5 = recent5.Average(entry => entry.SectionalFinishSpeedPct);
                float avgPercentile5 = recent5.Average(entry => entry.SectionalLast3FPercentile);

                row["SectionalLastFinishSpeedPct"] = last.SectionalFinishSpeedPct;
                row["SectionalAvgFinishSpeedPctLast3"] = recent3.Average(entry => entry.SectionalFinishSpeedPct);
                row["SectionalAvgFinishSpeedPctLast5"] = avgFinishSpeed5;
                row["SectionalBestFinishSpeedPctLast5"] = recent5.Max(entry => entry.SectionalFinishSpeedPct);
                row["SectionalFinishSpeedTrendLast5"] = last.SectionalFinishSpeedPct - avgFinishSpeed5;
                row["SectionalLast3FSecondsLast"] = last.SectionalLast3FSeconds;
                row["SectionalAvgLast3FSecondsLast5"] = recent5.Average(entry => entry.SectionalLast3FSeconds);
                row["SectionalLast3FPercentileLast"] = last.SectionalLast3FPercentile;
                row["SectionalAvgLast3FPercentileLast3"] = recent3.Average(entry => entry.SectionalLast3FPercentile);
                row["SectionalAvgLast3FPercentileLast5"] = avgPercentile5;
                row["SectionalBestLast3FPercentileLast5"] = recent5.Max(entry => entry.SectionalLast3FPercentile);
                row["SectionalClosingRankNormLast"] = last.SectionalClosingPositionNorm;
                row["SectionalAvgClosingRankNormLast5"] = recent5.Average(entry => entry.SectionalClosingPositionNorm);
                row["SectionalEarlyRankNormLast"] = last.SectionalEarlyPositionNorm;
                row["SectionalAvgEarlyRankNormLast5"] = recent5.Average(entry => entry.SectionalEarlyPositionNorm);
                row["SectionalPaceStyleLast"] = last.SectionalPaceStyle;
                row["SectionalAvgPaceStyleLast5"] = recent5.Average(entry => entry.SectionalPaceStyle);
            }

            private void AddPedigreeFeatures(
                Dictionary<string, object?> row,
                RunnerPedigreeInfo pedigree,
                string surface,
                string going,
                string bucket)
            {
                bool hasSire = !string.IsNullOrEmpty(pedigree.SireKey);
                bool hasDam = !string.IsNullOrEmpty(pedigree.DamKey);
                bool hasDamSire = !string.IsNullOrEmpty(pedigree.DamSireKey);
                row["PedigreeKnown"] = hasSire || hasDam;

                _sireStats.TryGetValue(pedigree.SireKey, out var sireStat);
                _sireSurfaceStats.TryGetValue((pedigree.SireKey, surface), out var sireSurfaceStat);
                _sireGoingStats.TryGetValue((pedigree.SireKey, going), out var sireGoingStat);
                _sireDistanceStats.TryGetValue((pedigree.SireKey, bucket), out var sireDistanceStat);
                _damStats.TryGetValue(pedigree.DamKey, out var damStat);
                _damSireStats.TryGetValue(pedigree.DamSireKey, out var damSireStat);

                if (hasSire)
                {
                    row["PedigreeSireStarts"] = sireStat.starts;
                    row["PedigreeSireWinRate"] = _trainer.SmoothedWinRate(sireStat.wins, sireStat.starts);
                    row["PedigreeSireAvgNormPos"] = sireStat.starts > 0
                        ? sireStat.sumNorm / sireStat.starts
                        : NeutralNormPosition;
                    row["PedigreeSireLastNormPos"] = sireStat.starts > 0
                        ? sireStat.lastNorm
                        : NeutralNormPosition;
                    row["PedigreeSireSurfaceStarts"] = sireSurfaceStat.starts;
                    row["PedigreeSireSurfaceWinRate"] = _trainer.SmoothedWinRate(
                        sireSurfaceStat.wins,
                        sireSurfaceStat.starts);
                    row["PedigreeSireGoingStarts"] = sireGoingStat.starts;
                    row["PedigreeSireGoingWinRate"] = _trainer.SmoothedWinRate(
                        sireGoingStat.wins,
                        sireGoingStat.starts);
                    row["PedigreeSireDistanceStarts"] = sireDistanceStat.starts;
                    row["PedigreeSireDistanceWinRate"] = _trainer.SmoothedWinRate(
                        sireDistanceStat.wins,
                        sireDistanceStat.starts);
                }
                else
                {
                    row["PedigreeSireStarts"] = 0;
                    row["PedigreeSireWinRate"] = null;
                    row["PedigreeSireAvgNormPos"] = NeutralNormPosition;
                    row["PedigreeSireLastNormPos"] = NeutralNormPosition;
                    row["PedigreeSireSurfaceStarts"] = 0;
                    row["PedigreeSireSurfaceWinRate"] = 0f;
                    row["PedigreeSireGoingStarts"] = 0;
                    row["PedigreeSireGoingWinRate"] = 0f;
                    row["PedigreeSireDistanceStarts"] = 0;
                    row["PedigreeSireDistanceWinRate"] = 0f;
                }

                if (hasDam)
                {
                    row["PedigreeDamStarts"] = damStat.starts;
                    row["PedigreeDamWinRate"] = _trainer.SmoothedWinRate(damStat.wins, damStat.starts);
                    row["PedigreeDamAvgNormPos"] = damStat.starts > 0
                        ? damStat.sumNorm / damStat.starts
                        : NeutralNormPosition;
                    row["PedigreeDamLastNormPos"] = damStat.starts > 0
                        ? damStat.lastNorm
                        : NeutralNormPosition;
                }
                else
                {
                    row["PedigreeDamStarts"] = 0;
                    row["PedigreeDamWinRate"] = null;
                    row["PedigreeDamAvgNormPos"] = NeutralNormPosition;
                    row["PedigreeDamLastNormPos"] = NeutralNormPosition;
                }

                if (hasDamSire)
                {
                    row["PedigreeDamSireStarts"] = damSireStat.starts;
                    row["PedigreeDamSireWinRate"] = _trainer.SmoothedWinRate(
                        damSireStat.wins,
                        damSireStat.starts);
                    row["PedigreeDamSireAvgNormPos"] = damSireStat.starts > 0
                        ? damSireStat.sumNorm / damSireStat.starts
                        : NeutralNormPosition;
                }
                else
                {
                    row["PedigreeDamSireStarts"] = 0;
                    row["PedigreeDamSireWinRate"] = 0f;
                    row["PedigreeDamSireAvgNormPos"] = NeutralNormPosition;
                }
            }

            private static void AddPedigreeStat(
                Dictionary<string, (int starts, int wins, float sumNorm, float lastNorm)> stats,
                string key,
                bool won,
                float normFinish)
            {
                if (string.IsNullOrEmpty(key))
                {
                    return;
                }

                stats.TryGetValue(key, out var stat);
                stat.starts++;
                stat.sumNorm += normFinish;
                if (won)
                {
                    stat.wins++;
                }
                stat.lastNorm = normFinish;
                stats[key] = stat;
            }

            private static void AddPedigreeStat(
                Dictionary<(string key, string bucket), (int starts, int wins, float sumNorm, float lastNorm)> stats,
                string key,
                string bucket,
                bool won,
                float normFinish)
            {
                if (string.IsNullOrEmpty(key))
                {
                    return;
                }

                stats.TryGetValue((key, bucket), out var stat);
                stat.starts++;
                stat.sumNorm += normFinish;
                if (won)
                {
                    stat.wins++;
                }
                stat.lastNorm = normFinish;
                stats[(key, bucket)] = stat;
            }

            /// <summary>
            /// Applies the post-race state changes that were captured while the
            /// features were computed. Keeping this separate means the feature loop
            /// runs once per race instead of twice, and no row snapshot/restore is
            /// needed to prevent the current result leaking into its own features.
            /// </summary>
            private void ApplyUpdates(List<RunnerUpdate> updates)
            {
                foreach (var update in updates)
                {
                    int horseId = update.HorseId;
                    DateTime date = update.Date;
                    short? finish = update.Finish;
                    float normFinish = update.NormFinish;
                    bool won = update.Won;
                    int classVal = update.ClassVal;
                    string going = update.Going;
                    string surface = update.Surface;
                    int courseId = update.CourseId;
                    string bucket = update.Bucket;
                    int distanceYards = update.DistanceYards;
                    int? trainerId = update.TrainerId;
                    int? jockeyId = update.JockeyId;

                    var horseClassKey = (horseId, classVal);
                    _horseClassStats.TryGetValue(horseClassKey, out var horseClassStat);
                    horseClassStat.starts++;
                    horseClassStat.sumNorm += normFinish;
                    if (won) horseClassStat.wins++;
                    horseClassStat.lastNorm = normFinish;
                    _horseClassStats[horseClassKey] = horseClassStat;

                    if (trainerId.HasValue)
                    {
                        var trainerClassKey = (trainerId.Value, classVal);
                        _trainerClassStats.TryGetValue(trainerClassKey, out var trainerClassStat);
                        trainerClassStat.starts++;
                        trainerClassStat.sumNorm += normFinish;
                        if (won) trainerClassStat.wins++;
                        trainerClassStat.lastNorm = normFinish;
                        _trainerClassStats[trainerClassKey] = trainerClassStat;
                    }

                    if (jockeyId.HasValue)
                    {
                        var jockeyClassKey = (jockeyId.Value, classVal);
                        _jockeyClassStats.TryGetValue(jockeyClassKey, out var jockeyClassStat);
                        jockeyClassStat.starts++;
                        jockeyClassStat.sumNorm += normFinish;
                        if (won) jockeyClassStat.wins++;
                        jockeyClassStat.lastNorm = normFinish;
                        _jockeyClassStats[jockeyClassKey] = jockeyClassStat;
                    }

                    if (!_horseHistory.TryGetValue(horseId, out var history))
                    {
                        history = new List<HistoryEntry>();
                        _horseHistory[horseId] = history;
                    }

                    history.Add(new HistoryEntry(
                        date,
                        normFinish,
                        finish,
                        going,
                        surface,
                        courseId,
                        bucket,
                        classVal,
                        update.Speed,
                        update.SpeedDiff,
                        update.Age,
                        won,
                        update.Rating,
                        update.Weight,
                        update.RunStyle,
                        finish.HasValue,
                        jockeyId ?? 0,
                        trainerId ?? 0,
                        distanceYards,
                        update.Headgear,
                        update.Medication,
                        update.HasSectional,
                        update.SectionalFinishSpeedPct,
                        update.SectionalLast3FPercentile,
                        update.SectionalLast3FSeconds,
                        update.SectionalEarlyPositionNorm,
                        update.SectionalClosingPositionNorm,
                        update.SectionalPaceStyle));
                    if (history.Count > HistoryLength)
                    {
                        history.RemoveAt(0);
                    }

                    if (!_surfaceStats.TryGetValue(horseId, out var surfaceDict))
                    {
                        surfaceDict = new();
                        _surfaceStats[horseId] = surfaceDict;
                    }
                    if (!surfaceDict.TryGetValue(surface, out var surfaceStats))
                    {
                        surfaceStats = (0, 0, 0f, 0f);
                    }
                    surfaceStats.starts++;
                    surfaceStats.sumNorm += normFinish;
                    if (won) surfaceStats.wins++;
                    surfaceStats.lastNorm = normFinish;
                    surfaceDict[surface] = surfaceStats;

                    if (!_goingStats.TryGetValue(horseId, out var goingDict))
                    {
                        goingDict = new();
                        _goingStats[horseId] = goingDict;
                    }
                    if (!goingDict.TryGetValue(going, out var goingStats))
                    {
                        goingStats = (0, 0, 0f, 0f);
                    }
                    goingStats.starts++;
                    goingStats.sumNorm += normFinish;
                    if (won) goingStats.wins++;
                    goingStats.lastNorm = normFinish;
                    goingDict[going] = goingStats;

                    string goingCourseKey = going + "_" + courseId;
                    if (!_goingCourseStats.TryGetValue(horseId, out var goingCourseDict))
                    {
                        goingCourseDict = new();
                        _goingCourseStats[horseId] = goingCourseDict;
                    }
                    if (!goingCourseDict.TryGetValue(goingCourseKey, out var goingCourseStats))
                    {
                        goingCourseStats = (0, 0, 0f, 0f);
                    }
                    goingCourseStats.starts++;
                    goingCourseStats.sumNorm += normFinish;
                    if (won) goingCourseStats.wins++;
                    goingCourseStats.lastNorm = normFinish;
                    goingCourseDict[goingCourseKey] = goingCourseStats;

                    if (!_courseStats.TryGetValue(horseId, out var courseDict))
                    {
                        courseDict = new();
                        _courseStats[horseId] = courseDict;
                    }
                    if (!courseDict.TryGetValue(courseId, out var courseStats))
                    {
                        courseStats = (0, 0, 0f, 0f);
                    }
                    courseStats.starts++;
                    courseStats.sumNorm += normFinish;
                    if (won) courseStats.wins++;
                    courseStats.lastNorm = normFinish;
                    courseDict[courseId] = courseStats;

                    if (!_ageStats.TryGetValue(horseId, out var ageDict))
                    {
                        ageDict = new();
                        _ageStats[horseId] = ageDict;
                    }
                    if (!ageDict.TryGetValue(update.AgeRestriction, out var ageStats))
                    {
                        ageStats = (0, 0, 0f, 0f);
                    }
                    ageStats.starts++;
                    ageStats.sumNorm += normFinish;
                    if (won) ageStats.wins++;
                    ageStats.lastNorm = normFinish;
                    ageDict[update.AgeRestriction] = ageStats;

                    if (!_distanceBucketStats.TryGetValue(horseId, out var distanceBucketDict))
                    {
                        distanceBucketDict = new();
                        _distanceBucketStats[horseId] = distanceBucketDict;
                    }
                    if (!distanceBucketDict.TryGetValue(bucket, out var distanceBucketStats))
                    {
                        distanceBucketStats = (0, 0, 0f, 0f);
                    }
                    distanceBucketStats.starts++;
                    distanceBucketStats.sumNorm += normFinish;
                    if (won) distanceBucketStats.wins++;
                    distanceBucketStats.lastNorm = normFinish;
                    distanceBucketDict[bucket] = distanceBucketStats;

                    var goingDistanceKey = (horseId, going, bucket);
                    _goingDistanceStats.TryGetValue(goingDistanceKey, out var goingDistanceStats);
                    goingDistanceStats.starts++;
                    goingDistanceStats.sumNorm += normFinish;
                    if (won) goingDistanceStats.wins++;
                    goingDistanceStats.lastNorm = normFinish;
                    _goingDistanceStats[goingDistanceKey] = goingDistanceStats;

                    var drawKey = (courseId, bucket, update.Draw);
                    _drawStats.TryGetValue(drawKey, out var drawStats);
                    drawStats.starts++;
                    if (won) drawStats.wins++;
                    _drawStats[drawKey] = drawStats;

                    var drawBaselineKey = (courseId, bucket);
                    _drawBaselineStats.TryGetValue(drawBaselineKey, out var drawBaselineStats);
                    drawBaselineStats.starts++;
                    if (won) drawBaselineStats.wins++;
                    _drawBaselineStats[drawBaselineKey] = drawBaselineStats;

                    var drawGoingKey = (courseId, bucket, going, update.Draw);
                    _drawGoingStats.TryGetValue(drawGoingKey, out var drawGoingStats);
                    drawGoingStats.starts++;
                    if (won) drawGoingStats.wins++;
                    _drawGoingStats[drawGoingKey] = drawGoingStats;

                    var drawGoingBaselineKey = (courseId, bucket, going);
                    _drawGoingBaselineStats.TryGetValue(drawGoingBaselineKey, out var drawGoingBaselineStats);
                    drawGoingBaselineStats.starts++;
                    if (won) drawGoingBaselineStats.wins++;
                    _drawGoingBaselineStats[drawGoingBaselineKey] = drawGoingBaselineStats;

                    _horseDistanceAll.TryGetValue(horseId, out var allDistances);
                    allDistances.sum += distanceYards;
                    allDistances.count++;
                    _horseDistanceAll[horseId] = allDistances;
                    _horseLastDistance[horseId] = distanceYards;

                    _horseDistanceWins.TryGetValue(horseId, out var winningDistances);
                    if (won)
                    {
                        winningDistances.sum += distanceYards;
                        winningDistances.count++;
                    }
                    _horseDistanceWins[horseId] = winningDistances;

                    // Sire/dam progeny statistics are updated only after the
                    // race, exactly like trainer and jockey form. The current
                    // runner's pedigree identifies which family pools to
                    // advance; it is never read back into the same race.
                    if (!string.IsNullOrEmpty(update.SireKey))
                    {
                        AddPedigreeStat(_sireStats, update.SireKey, won, normFinish);
                        AddPedigreeStat(_sireSurfaceStats, update.SireKey, surface, won, normFinish);
                        AddPedigreeStat(_sireGoingStats, update.SireKey, going, won, normFinish);
                        AddPedigreeStat(_sireDistanceStats, update.SireKey, bucket, won, normFinish);
                    }
                    if (!string.IsNullOrEmpty(update.DamKey))
                    {
                        AddPedigreeStat(_damStats, update.DamKey, won, normFinish);
                    }
                    if (!string.IsNullOrEmpty(update.DamSireKey))
                    {
                        AddPedigreeStat(_damSireStats, update.DamSireKey, won, normFinish);
                    }

                    if (trainerId.HasValue)
                    {
                        if (!_trainerSurfaceStats.TryGetValue(trainerId.Value, out var trainerSurfaceDict))
                        {
                            trainerSurfaceDict = new();
                            _trainerSurfaceStats[trainerId.Value] = trainerSurfaceDict;
                        }
                        if (!trainerSurfaceDict.TryGetValue(surface, out var trainerSurfaceStats))
                        {
                            trainerSurfaceStats = (0, 0, 0f, 0f);
                        }
                        trainerSurfaceStats.starts++;
                        trainerSurfaceStats.sumNorm += normFinish;
                        if (won) trainerSurfaceStats.wins++;
                        trainerSurfaceStats.lastNorm = normFinish;
                        trainerSurfaceDict[surface] = trainerSurfaceStats;

                        if (!_trainerGoingStats.TryGetValue(trainerId.Value, out var trainerGoingDict))
                        {
                            trainerGoingDict = new();
                            _trainerGoingStats[trainerId.Value] = trainerGoingDict;
                        }
                        if (!trainerGoingDict.TryGetValue(going, out var trainerGoingStats))
                        {
                            trainerGoingStats = (0, 0, 0f, 0f);
                        }
                        trainerGoingStats.starts++;
                        trainerGoingStats.sumNorm += normFinish;
                        if (won) trainerGoingStats.wins++;
                        trainerGoingStats.lastNorm = normFinish;
                        trainerGoingDict[going] = trainerGoingStats;

                        if (!_trainerDistanceStats.TryGetValue(trainerId.Value, out var trainerDistanceDict))
                        {
                            trainerDistanceDict = new();
                            _trainerDistanceStats[trainerId.Value] = trainerDistanceDict;
                        }
                        if (!trainerDistanceDict.TryGetValue(bucket, out var trainerDistanceStats))
                        {
                            trainerDistanceStats = (0, 0, 0f, 0f);
                        }
                        trainerDistanceStats.starts++;
                        trainerDistanceStats.sumNorm += normFinish;
                        if (won) trainerDistanceStats.wins++;
                        trainerDistanceStats.lastNorm = normFinish;
                        trainerDistanceDict[bucket] = trainerDistanceStats;

                        var trainerCourseKey = (trainerId.Value, courseId);
                        _trainerCourseStats.TryGetValue(trainerCourseKey, out var trainerCourseStats);
                        trainerCourseStats.starts++;
                        if (won) trainerCourseStats.wins++;
                        _trainerCourseStats[trainerCourseKey] = trainerCourseStats;

                        if (!_trainerStats.TryGetValue(trainerId.Value, out var trainerStat))
                        {
                            trainerStat = new RollingStat();
                        }
                        while (trainerStat.Recent.Count > 0 &&
                               (date - trainerStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                        {
                            trainerStat.Recent.Dequeue();
                        }
                        trainerStat.Starts++;
                        if (won) trainerStat.Wins++;
                        trainerStat.Recent.Enqueue((date, won));
                        while (trainerStat.Recent.Count > TrainerJockeyRecentStarts)
                        {
                            trainerStat.Recent.Dequeue();
                        }
                        _trainerStats[trainerId.Value] = trainerStat;
                    }

                    if (jockeyId.HasValue)
                    {
                        if (!_jockeySurfaceStats.TryGetValue(jockeyId.Value, out var jockeySurfaceDict))
                        {
                            jockeySurfaceDict = new();
                            _jockeySurfaceStats[jockeyId.Value] = jockeySurfaceDict;
                        }
                        if (!jockeySurfaceDict.TryGetValue(surface, out var jockeySurfaceStats))
                        {
                            jockeySurfaceStats = (0, 0, 0f, 0f);
                        }
                        jockeySurfaceStats.starts++;
                        jockeySurfaceStats.sumNorm += normFinish;
                        if (won) jockeySurfaceStats.wins++;
                        jockeySurfaceStats.lastNorm = normFinish;
                        jockeySurfaceDict[surface] = jockeySurfaceStats;

                        if (!_jockeyGoingStats.TryGetValue(jockeyId.Value, out var jockeyGoingDict))
                        {
                            jockeyGoingDict = new();
                            _jockeyGoingStats[jockeyId.Value] = jockeyGoingDict;
                        }
                        if (!jockeyGoingDict.TryGetValue(going, out var jockeyGoingStats))
                        {
                            jockeyGoingStats = (0, 0, 0f, 0f);
                        }
                        jockeyGoingStats.starts++;
                        jockeyGoingStats.sumNorm += normFinish;
                        if (won) jockeyGoingStats.wins++;
                        jockeyGoingStats.lastNorm = normFinish;
                        jockeyGoingDict[going] = jockeyGoingStats;

                        if (!_jockeyDistanceStats.TryGetValue(jockeyId.Value, out var jockeyDistanceDict))
                        {
                            jockeyDistanceDict = new();
                            _jockeyDistanceStats[jockeyId.Value] = jockeyDistanceDict;
                        }
                        if (!jockeyDistanceDict.TryGetValue(bucket, out var jockeyDistanceStats))
                        {
                            jockeyDistanceStats = (0, 0, 0f, 0f);
                        }
                        jockeyDistanceStats.starts++;
                        jockeyDistanceStats.sumNorm += normFinish;
                        if (won) jockeyDistanceStats.wins++;
                        jockeyDistanceStats.lastNorm = normFinish;
                        jockeyDistanceDict[bucket] = jockeyDistanceStats;

                        var jockeyGoingDistanceKey = (jockeyId.Value, going, bucket);
                        _jockeyGoingDistanceStats.TryGetValue(jockeyGoingDistanceKey, out var jockeyGoingDistanceStats);
                        jockeyGoingDistanceStats.starts++;
                        jockeyGoingDistanceStats.sumNorm += normFinish;
                        if (won) jockeyGoingDistanceStats.wins++;
                        jockeyGoingDistanceStats.lastNorm = normFinish;
                        _jockeyGoingDistanceStats[jockeyGoingDistanceKey] = jockeyGoingDistanceStats;

                        var jockeyCourseKey = (jockeyId.Value, courseId);
                        _jockeyCourseStats.TryGetValue(jockeyCourseKey, out var jockeyCourseStats);
                        jockeyCourseStats.starts++;
                        if (won) jockeyCourseStats.wins++;
                        _jockeyCourseStats[jockeyCourseKey] = jockeyCourseStats;

                        if (!_jockeyStats.TryGetValue(jockeyId.Value, out var jockeyStat))
                        {
                            jockeyStat = new RollingStat();
                        }
                        while (jockeyStat.Recent.Count > 0 &&
                               (date - jockeyStat.Recent.Peek().date).TotalDays > TrainerJockeyRecentDays)
                        {
                            jockeyStat.Recent.Dequeue();
                        }
                        jockeyStat.Starts++;
                        if (won) jockeyStat.Wins++;
                        jockeyStat.Recent.Enqueue((date, won));
                        while (jockeyStat.Recent.Count > TrainerJockeyRecentStarts)
                        {
                            jockeyStat.Recent.Dequeue();
                        }
                        _jockeyStats[jockeyId.Value] = jockeyStat;
                    }

                    if (trainerId.HasValue && jockeyId.HasValue)
                    {
                        var pairKey = (trainerId.Value, jockeyId.Value);
                        _trainerJockeyStats.TryGetValue(pairKey, out var pairStats);
                        pairStats.starts++;
                        if (won) pairStats.wins++;
                        _trainerJockeyStats[pairKey] = pairStats;

                        var pairSurfaceKey = (trainerId.Value, jockeyId.Value, surface);
                        _trainerJockeySurfaceStats.TryGetValue(pairSurfaceKey, out var pairSurfaceStats);
                        pairSurfaceStats.starts++;
                        if (won) pairSurfaceStats.wins++;
                        _trainerJockeySurfaceStats[pairSurfaceKey] = pairSurfaceStats;
                    }
                }

                for (int i = 0; i < updates.Count; i++)
                {
                    var left = updates[i];
                    for (int j = i + 1; j < updates.Count; j++)
                    {
                        var right = updates[j];
                        UpdateHeadToHead(left.HorseId, right.HorseId, left.Won, left.NormFinish);
                        UpdateHeadToHead(right.HorseId, left.HorseId, right.Won, right.NormFinish);
                    }
                }
            }

            private static bool TryGetTimeOfDay(Dictionary<string, object?> row, out TimeSpan timeOfDay)
            {
                if (TryReadTimeValue(row, "ScheduledOff", out timeOfDay) ||
                    TryReadTimeValue(row, "ActualOff", out timeOfDay))
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
                // Field strength: the average official rating measures the
                // class of the race, while the top three capture how much
                // quality is concentrated at the head of the field.
                float minRating = ratingValues.Count > 0 ? ratingValues.Min() : 0f;
                float maxRating = ratingValues.Count > 0 ? ratingValues.Max() : 0f;
                float top3RatingMean = ratingValues.Count > 0
                    ? ratingValues.OrderByDescending(r => r).Take(3).Average()
                    : 0f;
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
                    minRating,
                    maxRating,
                    top3RatingMean,
                    ratingValues.Count,
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
                       float MinRating,
                       float MaxRating,
                       float Top3RatingMean,
                       int KnownRatingCount,
                       float TotalPurse,
                       bool HasSaddleclothStats,
                       float AvgSaddlecloth,
                       float MinSaddlecloth,
                       float MaxSaddlecloth,
                       bool HasWeightStats);
        }

        private static bool HasPreOffMarketPrice(Dictionary<string, object?> row)
        {
            return row.TryGetValue("MarketProbMissing", out var missing) &&
                   missing is bool isMissing &&
                   !isMissing;
        }

        private static bool HasPricedWinner(List<Dictionary<string, object?>> rows)
        {
            foreach (var row in rows)
            {
                if (row.TryGetValue("FinishPos", out var value) &&
                    value is not null &&
                    PreparedDataset.TryConvertToInt32(value, out var finishPosition) &&
                    finishPosition == 1)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool HasUsableResult(List<Dictionary<string, object?>> rows)
        {
            if (rows is null || rows.Count == 0)
            {
                return false;
            }

            foreach (var row in rows)
            {
                if (row.TryGetValue("FinishPos", out var value) &&
                    value != null &&
                    PreparedDataset.TryConvertToInt32(value, out var finishPosition) &&
                    finishPosition > 0)
                {
                    return true;
                }
            }

            return false;
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

        private static float ParseMarketPrice(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return 0f;
            }

            var trimmed = text.Trim()
                .Replace("JF", string.Empty, StringComparison.OrdinalIgnoreCase)
                .Replace("CF", string.Empty, StringComparison.OrdinalIgnoreCase)
                .Trim();

            if (trimmed.Equals("EVS", StringComparison.OrdinalIgnoreCase) ||
                trimmed.Equals("EVENS", StringComparison.OrdinalIgnoreCase))
            {
                return 2f;
            }

            if (float.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out var decimalPrice) &&
                decimalPrice > 1f)
            {
                return decimalPrice;
            }

            var match = Regex.Match(trimmed, @"(\d+)\s*/\s*(\d+)");
            if (match.Success &&
                int.TryParse(match.Groups[1].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var numerator) &&
                int.TryParse(match.Groups[2].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var denominator) &&
                denominator > 0)
            {
                return 1f + (float)numerator / denominator;
            }

            return 0f;
        }

        private float SmoothedWinRate(int wins, int starts)
            => (wins + _winRateAlpha) / (starts + _winRateBeta);

        private static float Logit(float probability)
        {
            probability = Math.Clamp(probability, 1e-6f, 1f - 1e-6f);
            return MathF.Log(probability / (1f - probability));
        }


        private static object? NormalizeDbValue(object? value)
        {
            return value == null || value is DBNull ? null : value;
        }

        private const string RaceWeatherSelect = @"rw.RainPrev1dMm,
                                   rw.RainPrev3dMm,
                                   rw.RainPrev7dMm,
                                   rw.RainPrev14dMm,
                                   rw.RainRaceDayMm,
                                   rw.WeatherTempMaxC,
                                   rw.WeatherTempMinC,
                                   rw.WeatherWindMaxKph,
                                   rw.WeatherWindDirectionDegrees,
                                   rw.WeatherHumidity,
                                   rw.WeatherDewPointC,
                                   rw.WeatherRainHoursPrev1,
                                   rw.WeatherFrostDaysPrev7,
                                   rw.WeatherSoilMoisture0To7,
                                   rw.WeatherSoilMoisture0To28,
                                   rw.WeatherRainAnomaly14dMm,
                                   rw.WeatherSoilMoistureAnomaly,
                                   rw.WeatherEt0Prev7dMm,
                                   rw.DaysSinceLastMeeting,
                                   rw.LastGoingAtCourse,
                                   rw.WeatherGoingIndex,
                                   rw.WeatherSoftRisk";

        private static string BuildRaceWeatherSelect(SqlConnection conn)
        {
            if (!TableExists(conn, "dbo.RaceWeather"))
            {
                return RaceWeatherNullSelect;
            }

            var columns = PreparedDataset.LoadColumnNames(conn, "RaceWeather");
            string Col(string name, string sqlType) => columns.Contains(name)
                ? $"rw.{name}"
                : $"CAST(NULL AS {sqlType}) AS {name}";

            return $@"rw.RainPrev1dMm,
                                   rw.RainPrev3dMm,
                                   rw.RainPrev7dMm,
                                   rw.RainPrev14dMm,
                                   rw.RainRaceDayMm,
                                   rw.WeatherTempMaxC,
                                   rw.WeatherTempMinC,
                                   rw.WeatherWindMaxKph,
                                   {Col("WeatherWindDirectionDegrees", "FLOAT")},
                                   {Col("WeatherHumidity", "FLOAT")},
                                   {Col("WeatherDewPointC", "FLOAT")},
                                   {Col("WeatherRainHoursPrev1", "FLOAT")},
                                   {Col("WeatherFrostDaysPrev7", "INT")},
                                   {Col("WeatherSoilMoisture0To7", "FLOAT")},
                                   {Col("WeatherSoilMoisture0To28", "FLOAT")},
                                   {Col("WeatherRainAnomaly14dMm", "FLOAT")},
                                   {Col("WeatherSoilMoistureAnomaly", "FLOAT")},
                                   rw.WeatherEt0Prev7dMm,
                                   rw.DaysSinceLastMeeting,
                                   rw.LastGoingAtCourse,
                                   rw.WeatherGoingIndex,
                                   rw.WeatherSoftRisk";
        }

        private const string RaceWeatherNullSelect = @"CAST(NULL AS FLOAT) AS RainPrev1dMm,
                                   CAST(NULL AS FLOAT) AS RainPrev3dMm,
                                   CAST(NULL AS FLOAT) AS RainPrev7dMm,
                                   CAST(NULL AS FLOAT) AS RainPrev14dMm,
                                   CAST(NULL AS FLOAT) AS RainRaceDayMm,
                                   CAST(NULL AS FLOAT) AS WeatherTempMaxC,
                                   CAST(NULL AS FLOAT) AS WeatherTempMinC,
                                   CAST(NULL AS FLOAT) AS WeatherWindMaxKph,
                                   CAST(NULL AS FLOAT) AS WeatherWindDirectionDegrees,
                                   CAST(NULL AS FLOAT) AS WeatherHumidity,
                                   CAST(NULL AS FLOAT) AS WeatherDewPointC,
                                   CAST(NULL AS FLOAT) AS WeatherRainHoursPrev1,
                                   CAST(NULL AS INT) AS WeatherFrostDaysPrev7,
                                   CAST(NULL AS FLOAT) AS WeatherSoilMoisture0To7,
                                   CAST(NULL AS FLOAT) AS WeatherSoilMoisture0To28,
                                   CAST(NULL AS FLOAT) AS WeatherRainAnomaly14dMm,
                                   CAST(NULL AS FLOAT) AS WeatherSoilMoistureAnomaly,
                                   CAST(NULL AS FLOAT) AS WeatherEt0Prev7dMm,
                                   CAST(NULL AS INT) AS DaysSinceLastMeeting,
                                   CAST(NULL AS NVARCHAR(30)) AS LastGoingAtCourse,
                                   CAST(NULL AS FLOAT) AS WeatherGoingIndex,
                                   CAST(NULL AS FLOAT) AS WeatherSoftRisk";

        private static bool TableExists(SqlConnection connection, string qualifiedName)
        {
            return connection.ExecuteScalar<int>(
                "SELECT CASE WHEN OBJECT_ID(@Name, N'U') IS NULL THEN 0 ELSE 1 END;",
                new { Name = qualifiedName }) == 1;
        }

        private RaceWeatherFeatures? TryBuildUpcomingWeatherFeatures(
            SqlConnection connection,
            int courseId,
            DateTime raceDate,
            TimeSpan? offTime,
            string? surface)
        {
            if (!_weatherEnabled ||
                _weatherRepository is null ||
                _weatherClient is null ||
                courseId <= 0 ||
                courseId > short.MaxValue)
            {
                return null;
            }

            try
            {
                var date = raceDate.Date;
                var startDate = date.AddDays(-_weatherLookbackDays);
                var weather = _weatherRepository
                    .GetWeather((short)courseId, startDate, date)
                    .ToList();

                var today = DateTime.Today;
                var hasFreshRaceDay = weather.Any(day =>
                    day.WeatherDate.Date == date &&
                    (date < today ||
                     (day.Source == "forecast" &&
                      day.FetchedUtc.HasValue &&
                      day.FetchedUtc.Value >= DateTime.UtcNow.AddHours(-2))));
                var needsFetch = weather.Count == 0 ||
                                 !hasFreshRaceDay ||
                                 weather.All(day => !day.PrecipitationMm.HasValue && !day.TempMaxC.HasValue);
                if (needsFetch && !IsWeatherFetchBlocked() && date <= today.AddDays(16))
                {
                    var course = connection.QuerySingleOrDefault<Course>(
                        @"SELECT TOP (1) CourseId, Name, Latitude, Longitude, TimeZoneId
                          FROM dbo.Course WHERE CourseId = @CourseId",
                        new { CourseId = (short)courseId });
                    if (course?.Latitude is null || course.Longitude is null)
                    {
                        // Unmatched tracks keep null weather features rather than
                        // guessing a location; rerunning the backfill applies any
                        // new catalogue entries.
                        return null;
                    }

                    var fetched = new List<CourseWeatherDay>();
                    if (startDate <= today.AddDays(-5))
                    {
                        var archiveEnd = date < today.AddDays(-5) ? date : today.AddDays(-5);
                        fetched.AddRange(_weatherClient.GetArchive(
                            (short)courseId,
                            course.Latitude.Value,
                            course.Longitude.Value,
                            course.TimeZoneId,
                            startDate,
                            archiveEnd));
                    }

                    if (date >= today.AddDays(-14))
                    {
                        fetched.AddRange(_weatherClient.GetForecast(
                            (short)courseId,
                            course.Latitude.Value,
                            course.Longitude.Value,
                            course.TimeZoneId));
                    }

                    var refreshed = fetched
                        .Where(day => day.WeatherDate.Date >= startDate && day.WeatherDate.Date <= date)
                        .GroupBy(day => day.WeatherDate.Date)
                        .Select(group => group.First())
                        .OrderBy(day => day.WeatherDate)
                        .ToList();

                    if (refreshed.Count > 0)
                    {
                        weather = refreshed;
                        _weatherRepository.UpsertWeather(weather);
                    }
                }

                var dayMap = weather
                    .GroupBy(day => day.WeatherDate.Date)
                    .ToDictionary(group => group.Key, group => group.First());
                var lastRace = _weatherRepository.GetLastRaceOnOrBefore(
                    (short)courseId,
                    date,
                    offTime);
                var daysSinceLastMeeting = lastRace is null
                    ? (int?)null
                    : Math.Max(0, (date - lastRace.RaceDate.Date).Days);
                var (rainNormal14d, soilMoistureNormal) =
                    _weatherRepository.GetWeatherNormals((short)courseId, date);

                return WeatherFeatureCalculator.Compute(
                    0,
                    (short)courseId,
                    date,
                    offTime,
                    surface,
                    lastRace?.Going,
                    daysSinceLastMeeting,
                    dayMap,
                    includeRaceDayRain: date >= today,
                    source: weather.Any(day => day.Source == "archive") ? "archive" : "forecast",
                    rainNormal14d: rainNormal14d,
                    soilMoistureNormal: soilMoistureNormal);
            }
            catch (Exception ex)
            {
                // A network outage must not add a 20-second timeout to every
                // live market. Keep serving cached features and retry later.
                BlockWeatherFetch();
                Console.Error.WriteLine(
                    $"\tWeather features unavailable for course {courseId} on {raceDate:yyyy-MM-dd}: {ex.Message}");
                return null;
            }
        }

        private bool IsWeatherFetchBlocked()
        {
            lock (_weatherFetchLock)
            {
                return DateTime.UtcNow < _weatherFetchDisabledUntilUtc;
            }
        }

        private void BlockWeatherFetch()
        {
            lock (_weatherFetchLock)
            {
                _weatherFetchDisabledUntilUtc = DateTime.UtcNow.AddMinutes(10);
            }
        }

        private static void ApplyWeatherFeatures(
            Dictionary<string, object?> row,
            RaceWeatherFeatures? features)
        {
            row["RainPrev1dMm"] = features?.RainPrev1dMm;
            row["RainPrev3dMm"] = features?.RainPrev3dMm;
            row["RainPrev7dMm"] = features?.RainPrev7dMm;
            row["RainPrev14dMm"] = features?.RainPrev14dMm;
            row["RainRaceDayMm"] = features?.RainRaceDayMm;
            row["WeatherTempMaxC"] = features?.WeatherTempMaxC;
            row["WeatherTempMinC"] = features?.WeatherTempMinC;
            row["WeatherWindMaxKph"] = features?.WeatherWindMaxKph;
            row["WeatherWindDirectionDegrees"] = features?.WeatherWindDirectionDegrees;
            row["WeatherHumidity"] = features?.WeatherHumidity;
            row["WeatherDewPointC"] = features?.WeatherDewPointC;
            row["WeatherRainHoursPrev1"] = features?.WeatherRainHoursPrev1;
            row["WeatherFrostDaysPrev7"] = features?.WeatherFrostDaysPrev7;
            row["WeatherSoilMoisture"] = features?.WeatherSoilMoisture0To7;
            row["WeatherSoilMoistureDeep"] = features?.WeatherSoilMoisture0To28;
            row["WeatherRainAnomaly14d"] = features?.WeatherRainAnomaly14dMm;
            row["WeatherSoilMoistureAnomaly"] = features?.WeatherSoilMoistureAnomaly;

            if (features?.WeatherWindDirectionDegrees is double degrees && double.IsFinite(degrees))
            {
                double radians = degrees * Math.PI / 180d;
                row["WeatherWindDirectionSin"] = Math.Sin(radians);
                row["WeatherWindDirectionCos"] = Math.Cos(radians);
            }
            else
            {
                row["WeatherWindDirectionSin"] = null;
                row["WeatherWindDirectionCos"] = null;
            }

            row["WeatherEt0Prev7dMm"] = features?.WeatherEt0Prev7dMm;
            row["DaysSinceLastMeeting"] = features?.DaysSinceLastMeeting;
            row["LastGoingAtCourse"] = features?.LastGoingAtCourse;
            row["WeatherGoingIndex"] = features?.WeatherGoingIndex;
            row["WeatherSoftRisk"] = features?.WeatherSoftRisk;
        }

        /// <summary>
        /// Rain and evapotranspiration are more informative together than either
        /// alone: 10 mm of rain after a dry week changes the surface much more
        /// than the same rain after a wet week. These features are derived from
        /// columns that already exist, so they are available for the full
        /// historical backfill and for live forecasts without another API call.
        /// </summary>
        private static void AddWeatherBalanceFeatures(
            Dictionary<string, object?> row,
            bool enabled)
        {
            if (!enabled)
            {
                return;
            }

            float? rain1 = ReadNullableFloat(row, "RainPrev1dMm");
            float? rain3 = ReadNullableFloat(row, "RainPrev3dMm");
            float? rain7 = ReadNullableFloat(row, "RainPrev7dMm");
            float? et07 = ReadNullableFloat(row, "WeatherEt0Prev7dMm");

            if (rain7.HasValue && et07.HasValue)
            {
                row["WeatherBalance7dMm"] = rain7.Value - et07.Value;
                row["WeatherBalanceRainEt0Ratio7"] = (rain7.Value + 1f) / (et07.Value + 1f);
                row["WeatherBalanceDryness7"] = et07.Value - rain7.Value;
            }

            if (rain3.HasValue && et07.HasValue)
            {
                row["WeatherBalance3dMm"] = rain3.Value - et07.Value * (3f / 7f);
            }

            if (rain3.HasValue && rain7.HasValue)
            {
                row["WeatherBalanceRainAccel7"] = rain3.Value - rain7.Value * (3f / 7f);
            }

            if (rain1.HasValue && rain7.HasValue)
            {
                row["WeatherBalanceRain1dShare"] = rain1.Value / (rain7.Value + 0.1f);
            }
        }

        private static float ReadHistoricalMarketPrice(Dictionary<string, object?> row, string key)
        {
            if (!row.TryGetValue(key, out var value) || value is null)
            {
                return 0f;
            }

            try
            {
                float price = value is string text
                    ? ParseMarketPrice(text)
                    : Convert.ToSingle(value, CultureInfo.InvariantCulture);
                return price > 1f && float.IsFinite(price) ? price : 0f;
            }
            catch
            {
                return 0f;
            }
        }

        private static float? ReadNullableFloat(Dictionary<string, object?> row, string key)
        {
            if (!row.TryGetValue(key, out var value) || value is null)
            {
                return null;
            }

            try
            {
                var parsed = Convert.ToSingle(value, CultureInfo.InvariantCulture);
                return float.IsFinite(parsed) ? parsed : null;
            }
            catch
            {
                return null;
            }
        }

        private static void AddBetfairLiquidityFeatures(Dictionary<string, object?> row)
        {
            float marketMatched = ReadNullableFloat(row, "BetfairTotalMatched") ?? 0f;
            float runnerVolume = ReadNullableFloat(row, "BetfairTradedVolume") ?? 0f;
            float preOffHigh = ReadNullableFloat(row, "BetfairPreOffHigh") ?? 0f;
            float preOffLow = ReadNullableFloat(row, "BetfairPreOffLow") ?? 0f;
            float projectedSp = ReadNullableFloat(row, "BetfairProjectedSP") ?? 0f;
            float availableBack = ReadNullableFloat(row, "BetfairAvailableToBack") ?? 0f;
            float availableLay = ReadNullableFloat(row, "BetfairAvailableToLay") ?? 0f;

            bool hasLiquidity = marketMatched > 0f || runnerVolume > 0f;
            row["BetfairMarketMatchedLog"] = marketMatched > 0f
                ? MathF.Log(1f + marketMatched)
                : 0f;
            row["BetfairRunnerVolumeLog"] = runnerVolume > 0f
                ? MathF.Log(1f + runnerVolume)
                : 0f;
            row["BetfairRunnerVolumeShare"] = marketMatched > 0f && runnerVolume > 0f
                ? Math.Clamp(runnerVolume / marketMatched, 0f, 1f)
                : 0f;
            row["BetfairLiquidityMissing"] = !hasLiquidity;

            bool hasRange = preOffHigh > 1f &&
                            preOffLow > 1f &&
                            preOffHigh > preOffLow &&
                            float.IsFinite(preOffHigh) &&
                            float.IsFinite(preOffLow);
            row["BetfairPreOffRangeLog"] = hasRange
                ? MathF.Log(preOffHigh / preOffLow)
                : 0f;
            row["BetfairExchangeRangeMissing"] = !hasRange;

            bool hasProjectedSp = projectedSp > 1f && float.IsFinite(projectedSp);
            row["BetfairProjectedSPLog"] = hasProjectedSp
                ? MathF.Log(projectedSp)
                : 0f;
            row["BetfairProjectedSPMissing"] = !hasProjectedSp;

            row["BetfairAvailableBackLog"] = availableBack > 0f
                ? MathF.Log(1f + availableBack)
                : 0f;
            row["BetfairAvailableLayLog"] = availableLay > 0f
                ? MathF.Log(1f + availableLay)
                : 0f;
            float totalAvailable = availableBack + availableLay;
            row["BetfairBookPressure"] = totalAvailable > 0f
                ? (availableBack - availableLay) / totalAvailable
                : 0f;
        }

        private static void AddMarketMovementFeatures(
            Dictionary<string, object?> row,
            float openingPrice,
            float touchedLow,
            float touchedHigh,
            float? openingRankRelative)
        {
            bool openingMissing = !(openingPrice > 1f && float.IsFinite(openingPrice));
            bool lowMissing = !(touchedLow > 1f && float.IsFinite(touchedLow));
            // A high equal to the low is not a range. Treat it as missing so a
            // single touched price cannot look like a zero-width but genuine
            // market range.
            bool highMissing = !(touchedHigh > 1f && float.IsFinite(touchedHigh)) ||
                               lowMissing ||
                               touchedHigh <= touchedLow;

            row["MarketMoveOpeningMissing"] = openingMissing;
            row["MarketMoveLowMissing"] = lowMissing;
            row["MarketMoveHighMissing"] = highMissing;
            row["MarketMoveOpeningRankRelative"] = openingRankRelative;

            // Neutral defaults keep the feature schema stable. The explicit
            // missing flags tell the model which values are real.
            row["MarketMoveOpeningToLowLog"] = 0f;
            row["MarketMoveOpeningToHighLog"] = 0f;
            row["MarketMoveOpeningMinusLow"] = 0f;
            row["MarketMoveShortenedFlag"] = false;
            row["MarketMoveRangeLog"] = 0f;
            row["MarketMoveOpeningInRange"] = 0.5f;
            row["MarketMoveBackedFlag"] = false;
            row["MarketMoveDriftFlag"] = false;

            if (openingMissing)
            {
                return;
            }

            if (!lowMissing)
            {
                row["MarketMoveOpeningToLowLog"] = MathF.Log(openingPrice / touchedLow);
                row["MarketMoveOpeningMinusLow"] = openingPrice - touchedLow;
                row["MarketMoveShortenedFlag"] = openingPrice > touchedLow * 1.01f;
            }

            if (highMissing)
            {
                return;
            }

            float range = touchedHigh - touchedLow;
            float openingInRange = Math.Clamp((openingPrice - touchedLow) / range, 0f, 1f);
            row["MarketMoveOpeningToHighLog"] = MathF.Log(touchedHigh / openingPrice);
            row["MarketMoveRangeLog"] = MathF.Log(touchedHigh / touchedLow);
            row["MarketMoveOpeningInRange"] = openingInRange;
            row["MarketMoveBackedFlag"] = openingInRange >= 0.5f;
            row["MarketMoveDriftFlag"] = openingInRange < 0.5f;
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
                "MarketDecimalOdds",
                "MarketMoveEarlyPrice",
                "MarketMoveLowPrice",
                "MarketMoveHighPrice",
                "TouchedHighFraction",
                "TouchedLowFraction",
                "BetfairSelectionId",
                "BetfairSP",
                "BetfairPreOffHigh",
                "BetfairPreOffLow",
                "BetfairLastTradedPrice",
                "MarketAnchorPrice",
                "MarketAnchorCapturedUtc",
                "BetfairTradedVolume",
                "BetfairProjectedSP",
                "BetfairAvailableToBack",
                "BetfairAvailableToLay",
                "BetfairTotalMatched",
                "BetfairMarketId",
                "BetfairMarketName",
                "BetfairBspReconciled",
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
                "Purse",
                "Comment",
                "EarlyPosition",
                "HeldUp",
                "Keen",
                "SlowStart",
                "CurrentSectionalCount",
                "CurrentSectionalLast1FSeconds",
                "CurrentSectionalLast2FSeconds",
                "CurrentSectionalLast3FSeconds",
                "CurrentSectionalLast1FPercentile",
                "CurrentSectionalLast2FPercentile",
                "CurrentSectionalLast3FPercentile",
                "CurrentSectionalEarlyPositionNorm",
                "CurrentSectionalClosingPositionNorm",
                "CurrentSectionalPaceStyle",
                "CurrentSectionalFinishSpeedPct",
                "CurrentSectionalTimeIndexScore",
                "CurrentSectionalMeetingAverageTimeIndex",
                "CurrentSectionalWinTimeVsPar",
                "SireName",
                "DamName",
                "DamSireName",
                "SourceHorseId",
                "SourceSlug"
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

        /// <summary>
        /// Runs the chronological feature-engineering pass and hands each usable
        /// race to <paramref name="onIncludedRace"/>. The callback receives the
        /// only race rows that are alive at that moment, so callers can either
        /// retain them (the historical <see cref="PrepareDataset"/> behaviour) or
        /// encode them and let the dictionaries be collected race-by-race.
        /// </summary>
        private void ProcessPreparedRaces(
                ISet<int>? includeRaceIds,
                ISet<int>? stateRaceWhitelist,
                bool includeIdentifiers,
                DateTime? minimumRaceDate,
                Action<PreparedRace> onIncludedRace,
                bool requireMarketPrice = false)
        {
            if (onIncludedRace is null)
                throw new ArgumentNullException(nameof(onIncludedRace));

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
                : "CAST(NULL AS DECIMAL(12,2)) AS Purse";
            string officialRatingColumn = runnerColumns.Contains("OfficialRating")
                ? "rr.OfficialRating AS OfficialRating"
                : "CAST(NULL AS SMALLINT) AS OfficialRating";
            string headgearColumn = runnerColumns.Contains("Headgear")
                ? "rr.Headgear AS Headgear"
                : "CAST(NULL AS NVARCHAR(128)) AS Headgear";
            string medicationColumn = runnerColumns.Contains("Medication")
                ? "rr.Medication AS Medication"
                : "CAST(NULL AS NVARCHAR(128)) AS Medication";
            string betfairTotalMatchedColumn = raceColumns.Contains("BetfairTotalMatched")
                ? "r.BetfairTotalMatched AS BetfairTotalMatched"
                : "CAST(NULL AS DECIMAL(18,2)) AS BetfairTotalMatched";
            string betfairTradedVolumeColumn = runnerColumns.Contains("BetfairTradedVolume")
                ? "rr.BetfairTradedVolume AS BetfairTradedVolume"
                : "CAST(NULL AS DECIMAL(18,2)) AS BetfairTradedVolume";
            string betfairPreOffHighColumn = runnerColumns.Contains("BetfairPreOffHigh")
                ? "rr.BetfairPreOffHigh AS BetfairPreOffHigh"
                : "CAST(NULL AS DECIMAL(9,2)) AS BetfairPreOffHigh";
            string betfairPreOffLowColumn = runnerColumns.Contains("BetfairPreOffLow")
                ? "rr.BetfairPreOffLow AS BetfairPreOffLow"
                : "CAST(NULL AS DECIMAL(9,2)) AS BetfairPreOffLow";
            string marketAnchorPriceColumn = runnerColumns.Contains("MarketAnchorPrice")
                ? "rr.MarketAnchorPrice AS MarketAnchorPrice"
                : "CAST(NULL AS DECIMAL(9,2)) AS MarketAnchorPrice";
            string marketAnchorCapturedColumn = runnerColumns.Contains("MarketAnchorCapturedUtc")
                ? "rr.MarketAnchorCapturedUtc AS MarketAnchorCapturedUtc"
                : "CAST(NULL AS DATETIME2(0)) AS MarketAnchorCapturedUtc";
            string betfairProjectedSpColumn = runnerColumns.Contains("BetfairProjectedSP")
                ? "rr.BetfairProjectedSP AS BetfairProjectedSP"
                : "CAST(NULL AS DECIMAL(9,2)) AS BetfairProjectedSP";
            string betfairAvailableToBackColumn = runnerColumns.Contains("BetfairAvailableToBack")
                ? "rr.BetfairAvailableToBack AS BetfairAvailableToBack"
                : "CAST(NULL AS DECIMAL(18,2)) AS BetfairAvailableToBack";
            string betfairAvailableToLayColumn = runnerColumns.Contains("BetfairAvailableToLay")
                ? "rr.BetfairAvailableToLay AS BetfairAvailableToLay"
                : "CAST(NULL AS DECIMAL(18,2)) AS BetfairAvailableToLay";
            bool hasRaceWeather = TableExists(conn, "dbo.RaceWeather");
            string raceWeatherSelect = BuildRaceWeatherSelect(conn);
            string raceWeatherJoin = hasRaceWeather
                ? "LEFT JOIN dbo.RaceWeather rw ON rw.RaceId = r.RaceId"
                : string.Empty;
            bool hasRunnerSectional = TableExists(conn, "dbo.RunnerSectional");
            string sectionalSelect = hasRunnerSectional
                ? "sec.SectionCount AS CurrentSectionalCount, " +
                  "sec.Last1FSeconds AS CurrentSectionalLast1FSeconds, " +
                  "sec.Last2FSeconds AS CurrentSectionalLast2FSeconds, " +
                  "sec.Last3FSeconds AS CurrentSectionalLast3FSeconds, " +
                  "sec.Last1FPercentile AS CurrentSectionalLast1FPercentile, " +
                  "sec.Last2FPercentile AS CurrentSectionalLast2FPercentile, " +
                  "sec.Last3FPercentile AS CurrentSectionalLast3FPercentile, " +
                  "sec.EarlyPositionNorm AS CurrentSectionalEarlyPositionNorm, " +
                  "sec.ClosingPositionNorm AS CurrentSectionalClosingPositionNorm, " +
                  "sec.PaceStyle AS CurrentSectionalPaceStyle, " +
                  "sec.FinishSpeedPct AS CurrentSectionalFinishSpeedPct, " +
                  "sec.TimeIndexScore AS CurrentSectionalTimeIndexScore, " +
                  "sec.MeetingAverageTimeIndex AS CurrentSectionalMeetingAverageTimeIndex, " +
                  "sec.WinTimeVsPar AS CurrentSectionalWinTimeVsPar"
                : "CAST(NULL AS TINYINT) AS CurrentSectionalCount, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast1FSeconds, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast2FSeconds, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast3FSeconds, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast1FPercentile, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast2FPercentile, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast3FPercentile, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalEarlyPositionNorm, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalClosingPositionNorm, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalPaceStyle, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalFinishSpeedPct, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalTimeIndexScore, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalMeetingAverageTimeIndex, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalWinTimeVsPar";
            string sectionalJoin = hasRunnerSectional
                ? "LEFT JOIN dbo.RunnerSectional sec ON sec.RaceId = r.RaceId AND sec.HorseId = rr.HorseId"
                : string.Empty;
            bool hasPedigree = TableExists(conn, "dbo.HorsePedigree");
            string pedigreeSelect = hasPedigree
                ? "hp.SireName AS SireName, hp.DamName AS DamName, hp.DamSireName AS DamSireName, " +
                  "hp.SourceHorseId AS SourceHorseId, hp.SourceSlug AS SourceSlug"
                : "CAST(NULL AS NVARCHAR(256)) AS SireName, " +
                  "CAST(NULL AS NVARCHAR(256)) AS DamName, " +
                  "CAST(NULL AS NVARCHAR(256)) AS DamSireName, " +
                  "CAST(NULL AS INT) AS SourceHorseId, " +
                  "CAST(NULL AS NVARCHAR(256)) AS SourceSlug";
            string pedigreeJoin = hasPedigree
                ? "LEFT JOIN dbo.HorsePedigree hp ON hp.HorseId = rr.HorseId"
                : string.Empty;

            var sql = $@"SELECT h.Name AS HorseName,
                                   r.RaceId,
                                   r.CourseId,
                                   c.Name AS CourseName,
                                   r.Title,
                                   r.RaceDate,
                                   {scheduledOffColumn},
                                   {actualOffColumn},
                                   r.RaceType,
                                   r.Class,
                                   r.AgeRestriction,
                                   r.Surface,
                                   r.Going,
                                   r.DistanceYards,
                                   r.RunnerCount,
                                   r.WinningTimeMs,
                                   {purseColumn},
                                   rr.HorseId,
                                   rr.TrainerId,
                                   rr.JockeyId,
                                   rr.SaddleclothNumber,
                                   rr.Draw,
                                   rr.Age,
                                   rr.WeightLbs,
                                   {officialRatingColumn},
                                   {headgearColumn},
                                   {medicationColumn},
                                   rr.FinishPos,
                                   rr.OutcomeCode,
                                   rr.DistanceBeatenText,
                                   rr.SP_Decimal,
                                   rr.BetfairSP,
                                   rr.BetfairLastTradedPrice,
                                   rr.OpeningFraction,
                                   rr.TouchedHighFraction,
                                   rr.TouchedLowFraction,
                                   {betfairTotalMatchedColumn},
                                   {betfairTradedVolumeColumn},
                                   {betfairPreOffHighColumn},
                                   {betfairPreOffLowColumn},
                                   {marketAnchorPriceColumn},
                                   {marketAnchorCapturedColumn},
                                   {betfairProjectedSpColumn},
                                   {betfairAvailableToBackColumn},
                                   {betfairAvailableToLayColumn},
                                   rs.EarlyPosition,
                                   rs.HeldUp,
                                   rs.Keen,
                                   rs.SlowStart,
                                   {sectionalSelect},
                                   {pedigreeSelect},
                                   {raceWeatherSelect}
                            FROM Race r
                            JOIN RunnerResult rr ON r.RaceId = rr.RaceId
                            LEFT JOIN Horse h ON rr.HorseId = h.HorseId
                            LEFT JOIN dbo.Course c ON c.CourseId = r.CourseId
                            LEFT JOIN dbo.RunnerRunStyle rs ON rs.RaceId = r.RaceId AND rs.HorseId = rr.HorseId
                            {sectionalJoin}
                            {pedigreeJoin}
                            {raceWeatherJoin}
                            WHERE (@MinDate IS NULL OR r.RaceDate >= @MinDate)
                            ORDER BY r.RaceDate, ScheduledOff, r.RaceId, rr.RunnerResultId
                            OPTION (FORCE ORDER)";

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
                    "Draw",
                    "SP_Decimal",
                    "BetfairSP",
                    "BetfairLastTradedPrice"
                };
            }

            var featureState = new FeatureEngineeringState(this, identifierKeys);
            // Feature values must be reproducible for a given Training:Seed.
            // The previous unseeded Random made tied ranks and within-race row
            // order depend on process timing.
            var rnd = new Random(_seed);
            var currentRows = new List<Dictionary<string, object?>>();
            int? currentRaceId = null;
            bool ShouldInclude(int raceId) => includeRaceIds is null || includeRaceIds.Contains(raceId);
            bool ShouldUpdate(int raceId) => stateRaceWhitelist is null || stateRaceWhitelist.Contains(raceId);
            void EmitIncludedRace(int raceId, List<Dictionary<string, object?>> rows)
            {
                if (requireMarketPrice)
                {
                    // Keep the state pass complete (every runner above was used
                    // to advance the rolling statistics), but only hand priced
                    // runners to the training/evaluation dataset.
                    rows.RemoveAll(row => !HasPreOffMarketPrice(row));

                    // If the winner has no opening price, the remaining rows have
                    // no positive label. Such a race cannot teach the model (or
                    // produce a meaningful winner-accuracy score), so skip it.
                    if (!HasPricedWinner(rows))
                    {
                        return;
                    }
                }

                if (rows.Count > 0)
                {
                    onIncludedRace(new PreparedRace(raceId, rows));
                }
            }

            foreach (var record in conn.Query(sql, new { MinDate = minimumRaceDate }, commandTimeout: 6000, buffered: false))
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
                    if ((include || update) && HasUsableResult(currentRows))
                    {
                        featureState.ProcessRace(currentRows, include, update);
                        if (include)
                        {
                            EmitIncludedRace(previousRaceId, currentRows);
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
                if ((include || update) && HasUsableResult(currentRows))
                {
                    featureState.ProcessRace(currentRows, include, update);
                    if (include)
                    {
                        EmitIncludedRace(finalRaceId, currentRows);
                    }
                }
            }
        }

        public virtual PreparedDataset PrepareDataset(
                ISet<int>? includeRaceIds = null,
                ISet<int>? stateRaceWhitelist = null,
                bool includeIdentifiers = false,
                DateTime? minimumRaceDate = null)
        {
            var races = new List<PreparedRace>();
            ProcessPreparedRaces(
                includeRaceIds,
                stateRaceWhitelist,
                includeIdentifiers,
                minimumRaceDate,
                races.Add);
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

            var syntheticRaceId = CreateSyntheticRaceId(upcoming);
            var syntheticRows = BuildUpcomingRaceRows(conn, upcoming, flows, syntheticRaceId);
            if (syntheticRows.Count == 0)
            {
                return null;
            }

            lock (_liveStateLock)
            {
                var featureState = GetOrCreateLiveFeatureState(conn, upcoming);
                featureState.ProcessRace(syntheticRows, includeRace: true, updateState: false);
            }

            return new PreparedRace(syntheticRaceId, syntheticRows);
        }

        /// <summary>
        /// Reuses one historical feature state across every market in a live
        /// session. Previously each market replayed the whole history database,
        /// which is the dominant cost when a day report covers dozens of races.
        /// </summary>
        private FeatureEngineeringState GetOrCreateLiveFeatureState(SqlConnection conn, UpcomingRace upcoming)
        {
            // Never reuse a snapshot for a historical replay: it could contain
            // results from after the target race.
            if (upcoming.RaceDate.Date < DateTime.Today)
            {
                return BuildLiveFeatureState(conn, upcoming.RaceDate.Date, upcoming.ScheduledOff);
            }

            lock (_liveStateLock)
            {
                if (_liveState is null ||
                    (DateTime.UtcNow - _liveStateBuiltUtc).TotalMinutes >= _liveStateCacheMinutes)
                {
                    _liveState = BuildLiveFeatureState(conn, DateTime.Today, DateTime.Now.TimeOfDay);
                    _liveStateBuiltUtc = DateTime.UtcNow;
                }

                return _liveState;
            }
        }

        private FeatureEngineeringState BuildLiveFeatureState(SqlConnection conn, DateTime targetDate, TimeSpan? targetOff)
        {
            var raceColumns = PreparedDataset.LoadColumnNames(conn, "Race");
            string scheduledOffColumn = raceColumns.Contains("ScheduledOff")
                ? "r.ScheduledOff AS ScheduledOff"
                : "CAST(NULL AS time(0)) AS ScheduledOff";
            string actualOffColumn = raceColumns.Contains("ActualOff")
                ? "r.ActualOff AS ActualOff"
                : "CAST(NULL AS time(0)) AS ActualOff";
            string scheduledOffExpression = raceColumns.Contains("ScheduledOff")
                ? "r.ScheduledOff"
                : "CAST(NULL AS time(0))";
            bool hasRaceWeather = TableExists(conn, "dbo.RaceWeather");
            string raceWeatherSelect = BuildRaceWeatherSelect(conn);
            string raceWeatherJoin = hasRaceWeather
                ? "LEFT JOIN dbo.RaceWeather rw ON rw.RaceId = r.RaceId"
                : string.Empty;
            bool hasRunnerSectional = TableExists(conn, "dbo.RunnerSectional");
            string sectionalSelect = hasRunnerSectional
                ? "sec.SectionCount AS CurrentSectionalCount, " +
                  "sec.Last1FSeconds AS CurrentSectionalLast1FSeconds, " +
                  "sec.Last2FSeconds AS CurrentSectionalLast2FSeconds, " +
                  "sec.Last3FSeconds AS CurrentSectionalLast3FSeconds, " +
                  "sec.Last1FPercentile AS CurrentSectionalLast1FPercentile, " +
                  "sec.Last2FPercentile AS CurrentSectionalLast2FPercentile, " +
                  "sec.Last3FPercentile AS CurrentSectionalLast3FPercentile, " +
                  "sec.EarlyPositionNorm AS CurrentSectionalEarlyPositionNorm, " +
                  "sec.ClosingPositionNorm AS CurrentSectionalClosingPositionNorm, " +
                  "sec.PaceStyle AS CurrentSectionalPaceStyle, " +
                  "sec.FinishSpeedPct AS CurrentSectionalFinishSpeedPct, " +
                  "sec.TimeIndexScore AS CurrentSectionalTimeIndexScore, " +
                  "sec.MeetingAverageTimeIndex AS CurrentSectionalMeetingAverageTimeIndex, " +
                  "sec.WinTimeVsPar AS CurrentSectionalWinTimeVsPar"
                : "CAST(NULL AS TINYINT) AS CurrentSectionalCount, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast1FSeconds, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast2FSeconds, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast3FSeconds, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast1FPercentile, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast2FPercentile, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalLast3FPercentile, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalEarlyPositionNorm, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalClosingPositionNorm, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalPaceStyle, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalFinishSpeedPct, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalTimeIndexScore, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalMeetingAverageTimeIndex, " +
                  "CAST(NULL AS FLOAT) AS CurrentSectionalWinTimeVsPar";
            string sectionalJoin = hasRunnerSectional
                ? "LEFT JOIN dbo.RunnerSectional sec ON sec.RaceId = r.RaceId AND sec.HorseId = rr.HorseId"
                : string.Empty;
            bool hasPedigree = TableExists(conn, "dbo.HorsePedigree");
            string pedigreeSelect = hasPedigree
                ? "hp.SireName AS SireName, hp.DamName AS DamName, hp.DamSireName AS DamSireName, " +
                  "hp.SourceHorseId AS SourceHorseId, hp.SourceSlug AS SourceSlug"
                : "CAST(NULL AS NVARCHAR(256)) AS SireName, " +
                  "CAST(NULL AS NVARCHAR(256)) AS DamName, " +
                  "CAST(NULL AS NVARCHAR(256)) AS DamSireName, " +
                  "CAST(NULL AS INT) AS SourceHorseId, " +
                  "CAST(NULL AS NVARCHAR(256)) AS SourceSlug";
            string pedigreeJoin = hasPedigree
                ? "LEFT JOIN dbo.HorsePedigree hp ON hp.HorseId = rr.HorseId"
                : string.Empty;

            var sql = $@"SELECT r.RaceId,
                                   r.CourseId,
                                   r.RaceDate,
                                   {scheduledOffColumn},
                                   {actualOffColumn},
                                   r.RaceType,
                                   r.Class,
                                   r.AgeRestriction,
                                   r.Surface,
                                   r.Going,
                                   r.DistanceYards,
                                   r.RunnerCount,
                                   r.WinningTimeMs,
                                   rr.HorseId,
                                   rr.TrainerId,
                                   rr.JockeyId,
                                   rr.SaddleclothNumber,
                                   rr.Draw,
                                   rr.Age,
                                   rr.WeightLbs,
                                   rr.FinishPos,
                                   rr.OutcomeCode,
                                   rr.DistanceBeatenText,
                                   rr.SP_Decimal,
                                   rs.EarlyPosition,
                                   rs.HeldUp,
                                   rs.Keen,
                                   rs.SlowStart,
                                   {sectionalSelect},
                                   {pedigreeSelect},
                                   {raceWeatherSelect}
                            FROM Race r
                            JOIN RunnerResult rr ON r.RaceId = rr.RaceId
                            LEFT JOIN dbo.RunnerRunStyle rs ON rs.RaceId = r.RaceId AND rs.HorseId = rr.HorseId
                            {sectionalJoin}
                            {pedigreeJoin}
                            {raceWeatherJoin}
                            WHERE r.RaceDate < @TargetDate
                               OR (r.RaceDate = @TargetDate
                                   AND @TargetOff IS NOT NULL
                                   AND {scheduledOffExpression} IS NOT NULL
                                   AND {scheduledOffExpression} < @TargetOff)
                            ORDER BY r.RaceDate, ScheduledOff, r.RaceId, rr.RunnerResultId";

            var featureState = new FeatureEngineeringState(this, LiveIdentifierKeys);
            var currentRows = new List<Dictionary<string, object?>>();
            int? currentRaceId = null;
            foreach (var record in conn.Query(sql, new { TargetDate = targetDate.Date, TargetOff = targetOff }, commandTimeout: 6000, buffered: false))
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
                    if (currentRows.Count > 0 && HasUsableResult(currentRows))
                    {
                        featureState.ProcessRace(currentRows, includeRace: false, updateState: true);
                    }
                    currentRows = new List<Dictionary<string, object?>>();
                }

                currentRows.Add(row);
                currentRaceId = raceId;
            }

            if (currentRaceId.HasValue && currentRows.Count > 0 && HasUsableResult(currentRows))
            {
                featureState.ProcessRace(currentRows, includeRace: false, updateState: true);
            }

            return featureState;
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
            int raceId)
        {
            var rows = new List<Dictionary<string, object?>>(flows.Count);
            var (courseId, courseName, courseTimeZoneId) = ResolveCourse(conn, upcoming);
            var weatherFeatures = TryBuildUpcomingWeatherFeatures(
                conn,
                courseId,
                upcoming.RaceDate,
                upcoming.ScheduledOff,
                upcoming.Surface);
            int runnerCount = flows.Count;
            int marketAnchorMinutes = _marketAnchorMinutes;
            var anchorCutoffUtc = ResolveMarketAnchorCutoffUtc(
                upcoming,
                courseTimeZoneId,
                marketAnchorMinutes);

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
                    ["MarketDecimalOdds"] = flow.BackPrice1,
                    ["LayPrice1"] = flow.LayPrice1,
                    ["LayPrice2"] = flow.LayPrice2,
                    ["LayPrice3"] = flow.LayPrice3
                };

                // Values that depend on historical lookups are populated below when data is available.
                row["Purse"] = null;
                var purseMatch = Regex.Match(
                    upcoming.RaceDetails ?? string.Empty,
                    @"£\s*([\d,]+)");
                if (purseMatch.Success &&
                    decimal.TryParse(
                        purseMatch.Groups[1].Value.Replace(",", string.Empty),
                        NumberStyles.Number,
                        CultureInfo.InvariantCulture,
                        out var parsedPurse))
                {
                    row["Purse"] = parsedPurse;
                }
                row["TrainerId"] = null;
                row["TrainerName"] = null;
                ApplyWeatherFeatures(row, weatherFeatures);
                AddWeatherBalanceFeatures(row, enabled: true);

                var horseId = ResolveHorseId(conn, horseName);
                row["HorseId"] = horseId;
                var pedigree = ResolvePedigreeInfo(conn, horseId, horseName);
                row["SireName"] = pedigree?.SireName;
                row["DamName"] = pedigree?.DamName;
                row["DamSireName"] = pedigree?.DamSireName;
                row["SourceHorseId"] = pedigree?.SourceHorseId;
                row["SourceSlug"] = pedigree?.SourceSlug;

                // Live rows previously left trainer and age null, which meant
                // every Trainer* feature had to be dropped from the schema and
                // the Age features were unavailable. The previous run is a
                // reliable proxy: the trainer is usually unchanged and the
                // official age increments on 1 January, so age at the last run
                // plus the year difference is exact.
                var lastRun = ResolveLastRunInfo(conn, horseId, upcoming.RaceDate);
                if (lastRun is not null)
                {
                    if (lastRun.TrainerId.HasValue)
                    {
                        row["TrainerId"] = lastRun.TrainerId.Value;
                    }
                    if (lastRun.Age.HasValue)
                    {
                        row["Age"] = lastRun.Age.Value +
                            Math.Max(0, upcoming.RaceDate.Year - lastRun.RaceDate.Year);
                    }
                    if (lastRun.OfficialRating.HasValue)
                    {
                        row["OfficialRating"] = lastRun.OfficialRating.Value;
                    }
                    if (lastRun.WeightLbs.HasValue)
                    {
                        row["WeightLbs"] = lastRun.WeightLbs.Value;
                    }
                    if (!string.IsNullOrWhiteSpace(lastRun.Headgear))
                    {
                        row["Headgear"] = lastRun.Headgear;
                    }
                    if (!string.IsNullOrWhiteSpace(lastRun.Medication))
                    {
                        row["Medication"] = lastRun.Medication;
                    }
                }

                if (!string.IsNullOrWhiteSpace(flow.JockeyName))
                {
                    var jockeyId = ResolveJockeyId(conn, flow.JockeyName!);
                    if (jockeyId.HasValue)
                    {
                        row["JockeyId"] = jockeyId.Value;
                    }
                }

                var priceRange = ResolveFlowPriceRange(conn, flow, anchorCutoffUtc);
                if (priceRange is not null)
                {
                    if (priceRange.EarlyPrice.HasValue)
                    {
                        row["MarketMoveEarlyPrice"] = priceRange.EarlyPrice.Value;
                    }
                    if (priceRange.LowPrice.HasValue)
                    {
                        row["MarketMoveLowPrice"] = priceRange.LowPrice.Value;
                    }
                    if (priceRange.HighPrice.HasValue)
                    {
                        row["MarketMoveHighPrice"] = priceRange.HighPrice.Value;
                    }
                    if (priceRange.MarketPrice.HasValue && priceRange.MarketPrice.Value > 1m)
                    {
                        row["MarketDecimalOdds"] = priceRange.MarketPrice.Value;
                    }
                    if (priceRange.AnchorPrice.HasValue)
                    {
                        row["MarketAnchorPrice"] = priceRange.AnchorPrice.Value;
                        row["MarketAnchorCapturedUtc"] = priceRange.AnchorCapturedUtc;
                    }
                }

                rows.Add(row);
            }

            return rows;
        }

        private sealed class LastRunInfo
        {
            public int? TrainerId { get; set; }
            public byte? Age { get; set; }
            public short? OfficialRating { get; set; }
            public byte? WeightLbs { get; set; }
            public string? Headgear { get; set; }
            public string? Medication { get; set; }
            public DateTime RaceDate { get; set; }
        }

        private sealed class FlowPriceRange
        {
            public decimal? EarlyPrice { get; set; }
            public decimal? LowPrice { get; set; }
            public decimal? HighPrice { get; set; }
            public decimal? EarliestPrice { get; set; }
            public decimal? AnchorPrice { get; set; }
            public DateTime? AnchorCapturedUtc { get; set; }
            public decimal? MarketPrice { get; set; }
        }

        private static FlowPriceRange? ResolveFlowPriceRange(
            SqlConnection conn,
            RunnerFlow flow,
            DateTime? anchorCutoffUtc)
        {
            decimal? current = flow.BackPrice1 is > 1m ? flow.BackPrice1 : null;
            if (!TableExists(conn, "dbo.RunnerFlow"))
            {
                return current.HasValue
                    ? new FlowPriceRange
                    {
                        EarlyPrice = current,
                        LowPrice = current,
                        HighPrice = current,
                        EarliestPrice = current,
                        MarketPrice = current
                    }
                    : null;
            }

            const string sql = @"
SELECT
    (SELECT TOP (1) rf.BackPrice1
       FROM dbo.RunnerFlow rf
      WHERE rf.MarketId = @MarketId
        AND ((@SelectionId IS NOT NULL AND rf.SelectionId = @SelectionId)
             OR (@SelectionId IS NULL AND rf.HorseName = @HorseName))
        AND rf.BackPrice1 > 1
      ORDER BY rf.RunnerFlowId ASC) AS EarlyPrice,
    (SELECT MIN(rf.BackPrice1)
       FROM dbo.RunnerFlow rf
      WHERE rf.MarketId = @MarketId
        AND ((@SelectionId IS NOT NULL AND rf.SelectionId = @SelectionId)
             OR (@SelectionId IS NULL AND rf.HorseName = @HorseName))
        AND rf.BackPrice1 > 1
        AND (@AnchorCutoffUtc IS NULL OR rf.CapturedUtc <= @AnchorCutoffUtc)) AS LowPrice,
    (SELECT MAX(rf.BackPrice1)
       FROM dbo.RunnerFlow rf
      WHERE rf.MarketId = @MarketId
        AND ((@SelectionId IS NOT NULL AND rf.SelectionId = @SelectionId)
             OR (@SelectionId IS NULL AND rf.HorseName = @HorseName))
        AND rf.BackPrice1 > 1
        AND (@AnchorCutoffUtc IS NULL OR rf.CapturedUtc <= @AnchorCutoffUtc)) AS HighPrice,
    (SELECT TOP (1) rf.BackPrice1
       FROM dbo.RunnerFlow rf
      WHERE rf.MarketId = @MarketId
        AND ((@SelectionId IS NOT NULL AND rf.SelectionId = @SelectionId)
             OR (@SelectionId IS NULL AND rf.HorseName = @HorseName))
        AND rf.BackPrice1 > 1
        AND @AnchorCutoffUtc IS NOT NULL
        AND rf.CapturedUtc <= @AnchorCutoffUtc
      ORDER BY rf.CapturedUtc DESC, rf.RunnerFlowId DESC) AS AnchorPrice,
    (SELECT TOP (1) rf.CapturedUtc
       FROM dbo.RunnerFlow rf
      WHERE rf.MarketId = @MarketId
        AND ((@SelectionId IS NOT NULL AND rf.SelectionId = @SelectionId)
             OR (@SelectionId IS NULL AND rf.HorseName = @HorseName))
        AND rf.BackPrice1 > 1
        AND @AnchorCutoffUtc IS NOT NULL
        AND rf.CapturedUtc <= @AnchorCutoffUtc
      ORDER BY rf.CapturedUtc DESC, rf.RunnerFlowId DESC) AS AnchorCapturedUtc,
    (SELECT TOP (1) rf.BackPrice1
       FROM dbo.RunnerFlow rf
      WHERE rf.MarketId = @MarketId
        AND ((@SelectionId IS NOT NULL AND rf.SelectionId = @SelectionId)
             OR (@SelectionId IS NULL AND rf.HorseName = @HorseName))
        AND rf.BackPrice1 > 1
      ORDER BY rf.CapturedUtc ASC, rf.RunnerFlowId ASC) AS EarliestPrice";

            var range = conn.QuerySingleOrDefault<FlowPriceRange>(
                sql,
                new
                {
                    MarketId = flow.MarketId,
                    SelectionId = flow.SelectionId,
                    HorseName = flow.HorseName,
                    AnchorCutoffUtc = anchorCutoffUtc
                });

            decimal? anchor = range?.AnchorPrice is > 1m ? range.AnchorPrice : null;
            DateTime? anchorCaptured = anchor.HasValue ? range?.AnchorCapturedUtc : null;
            decimal? earliest = range?.EarliestPrice is > 1m ? range.EarliestPrice : null;
            decimal? early = range?.EarlyPrice is > 1m ? range.EarlyPrice : current;
            decimal? low = range?.LowPrice is > 1m ? range.LowPrice : current;
            decimal? high = range?.HighPrice is > 1m ? range.HighPrice : current;
            if (current.HasValue)
            {
                low = !low.HasValue || current.Value < low.Value ? current : low;
                high = !high.HasValue || current.Value > high.Value ? current : high;
            }

            // The latest snapshot at or before T-N is the aligned price. If
            // the race is still more than N minutes away, or no earlier
            // snapshot exists, use the current quote; the earliest stored
            // snapshot is the last resort.
            decimal? marketPrice = anchor ?? current ?? earliest;
            if (!marketPrice.HasValue &&
                !early.HasValue &&
                !low.HasValue &&
                !high.HasValue)
            {
                return null;
            }

            return new FlowPriceRange
            {
                EarlyPrice = early,
                LowPrice = low,
                HighPrice = high,
                EarliestPrice = earliest,
                AnchorPrice = anchor,
                AnchorCapturedUtc = anchorCaptured,
                MarketPrice = marketPrice
            };
        }

        /// <summary>
        /// Converts the local race date and scheduled off time to a UTC cutoff
        /// of T-N. The course time zone is used when known; otherwise the
        /// process time zone is the best available approximation for the
        /// locally displayed racecard time.
        /// </summary>
        private static DateTime? ResolveMarketAnchorCutoffUtc(
            UpcomingRace upcoming,
            string? courseTimeZoneId,
            int marketAnchorMinutes)
        {
            if (!upcoming.ScheduledOff.HasValue)
            {
                return null;
            }

            var localOff = DateTime.SpecifyKind(
                upcoming.RaceDate.Date + upcoming.ScheduledOff.Value,
                DateTimeKind.Unspecified);
            var timeZone = ResolveTimeZone(courseTimeZoneId) ?? TimeZoneInfo.Local;
            DateTime offUtc;
            try
            {
                offUtc = TimeZoneInfo.ConvertTimeToUtc(localOff, timeZone);
            }
            catch (ArgumentException)
            {
                offUtc = DateTime.SpecifyKind(localOff, DateTimeKind.Utc);
            }

            return offUtc.AddMinutes(-Math.Max(0, marketAnchorMinutes));
        }

        private static TimeZoneInfo? ResolveTimeZone(string? timeZoneId)
        {
            if (string.IsNullOrWhiteSpace(timeZoneId))
            {
                return null;
            }

            if (TimeZoneInfo.TryFindSystemTimeZoneById(timeZoneId, out var timeZone))
            {
                return timeZone;
            }

            // Windows installations without ICU only understand Windows IDs.
            if (timeZoneId.Equals("Europe/London", StringComparison.OrdinalIgnoreCase) ||
                timeZoneId.Equals("Europe/Dublin", StringComparison.OrdinalIgnoreCase))
            {
                if (TimeZoneInfo.TryFindSystemTimeZoneById("GMT Standard Time", out timeZone))
                {
                    return timeZone;
                }
            }

            return null;
        }

        private sealed class PedigreeLookup
        {
            public int HorseId { get; set; }
            public string? SireName { get; set; }
            public string? DamName { get; set; }
            public string? DamSireName { get; set; }
            public int? SourceHorseId { get; set; }
            public string? SourceSlug { get; set; }
        }

        private PedigreeLookup? ResolvePedigreeInfo(
            SqlConnection conn,
            int horseId,
            string horseName)
        {
            if (horseId <= 0 || !TableExists(conn, "dbo.HorsePedigree"))
            {
                return null;
            }

            const string sql = @"
SELECT HorseId, SireName, DamName, DamSireName, SourceHorseId, SourceSlug
FROM dbo.HorsePedigree
WHERE HorseId = @HorseId";
            var existing = conn.QueryFirstOrDefault<PedigreeLookup>(sql, new { HorseId = horseId });
            if (existing is not null)
            {
                return existing;
            }

            var fetched = TryFetchPedigreeFromRacingTv(horseId, horseName);
            if (fetched is null)
            {
                return null;
            }

            return new PedigreeLookup
            {
                HorseId = fetched.HorseId,
                SireName = fetched.SireName,
                DamName = fetched.DamName,
                DamSireName = fetched.DamSireName,
                SourceHorseId = fetched.SourceHorseId,
                SourceSlug = fetched.SourceSlug
            };
        }

        private static LastRunInfo? ResolveLastRunInfo(
            SqlConnection conn,
            int horseId,
            DateTime raceDate)
        {
            const string sql = @"
SELECT TOP (1) rr.TrainerId, rr.Age, rr.OfficialRating, rr.WeightLbs,
       rr.Headgear, rr.Medication, r.RaceDate
FROM dbo.RunnerResult rr
JOIN dbo.Race r ON r.RaceId = rr.RaceId
WHERE rr.HorseId = @HorseId
  AND r.RaceDate < @RaceDate
ORDER BY r.RaceDate DESC, rr.RunnerResultId DESC";

            return conn.QueryFirstOrDefault<LastRunInfo>(
                sql,
                new { HorseId = horseId, RaceDate = raceDate.Date });
        }

        private (int CourseId, string? CourseName, string? TimeZoneId) ResolveCourse(
            SqlConnection conn,
            UpcomingRace upcoming)
        {
            if (!string.IsNullOrWhiteSpace(upcoming.VenueName))
            {
                const string exactSql = "SELECT TOP (1) CourseId, Name, TimeZoneId FROM Course WHERE Name = @Name ORDER BY CourseId";
                var exact = conn.QuerySingleOrDefault<(int CourseId, string Name, string? TimeZoneId)?>(
                    exactSql,
                    new { Name = upcoming.VenueName });
                if (exact.HasValue)
                {
                    return (exact.Value.CourseId, exact.Value.Name, exact.Value.TimeZoneId);
                }

                const string courseSql = "SELECT CourseId, Name, TimeZoneId FROM Course";
                var allCourses = conn.Query<(int CourseId, string Name, string? TimeZoneId)>(courseSql).ToList();
                var normalizedVenue = NormalizeLookupKey(upcoming.VenueName);
                (int CourseId, string Name, string? TimeZoneId)? best = null;
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
                    return (best.Value.CourseId, best.Value.Name, best.Value.TimeZoneId);
                }
            }
            Console.WriteLine($"\t\tNo match found in Course.Name for venue '{upcoming.VenueName ?? "<null>"}'; using synthetic course metadata.");
            int syntheticCourseId = GenerateSyntheticId("course:" + (upcoming.VenueName ?? upcoming.MarketId ?? string.Empty));
            return (syntheticCourseId, upcoming.VenueName, null);
        }

        private int ResolveHorseId(SqlConnection conn, string horseName)
        {
            const string exactSql = "SELECT TOP (1) HorseId FROM Horse WHERE Name = @Name ORDER BY HorseId";
            var existing = conn.QuerySingleOrDefault<int?>(exactSql, new { Name = horseName });
            if (existing.HasValue)
            {
                return existing.Value;
            }

            var normalized = FeatureText.NormalizeHorseName(horseName);
            if (string.IsNullOrEmpty(normalized))
            {
                return GenerateSyntheticId("horse:" + horseName);
            }

            var tokens = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var patterns = new List<string>();
            if (tokens.Length > 0)
            {
                patterns.Add(tokens[0] + "%");
            }
            if (tokens.Length > 1)
            {
                patterns.Add("%" + tokens[^1] + "%");
            }

            const string candidateSql = "SELECT HorseId, Name FROM Horse WHERE Name LIKE @Pattern";
            foreach (var pattern in patterns.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                var candidates = conn.Query<(int HorseId, string Name)>(candidateSql, new { Pattern = pattern }).ToList();
                var matches = candidates
                    .Where(c => FeatureText.NormalizeHorseName(c.Name) == normalized)
                    .Select(c => c.HorseId)
                    .Distinct()
                    .OrderBy(id => id)
                    .ToList();

                if (matches.Count > 0)
                {
                    return matches[0];
                }
            }

            Console.WriteLine($"\t\tNo normalized match found in Horse.Name for '{horseName}'; using synthetic horse identifier.");
            return GenerateSyntheticId("horse:" + normalized);
        }

        private int? ResolveJockeyId(SqlConnection conn, string jockeyName)
        {
            const string exactSql = "SELECT TOP (1) JockeyId FROM Jockey WHERE Name = @Name ORDER BY JockeyId";
            var existing = conn.QuerySingleOrDefault<int?>(exactSql, new { Name = jockeyName });
            if (existing.HasValue)
            {
                return existing.Value;
            }

            var normalized = FeatureText.NormalizeHorseName(jockeyName);
            if (string.IsNullOrEmpty(normalized))
            {
                return null;
            }

            var tokens = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0)
            {
                return null;
            }

            var surname = tokens[^1];
            const string candidateSql = "SELECT JockeyId, Name FROM Jockey WHERE Name LIKE @Pattern";
            var candidates = conn
                .Query<(int JockeyId, string Name)>(candidateSql, new { Pattern = "%" + surname + "%" })
                .ToList();

            var matches = candidates
                .Where(c =>
                {
                    var candidate = FeatureText.NormalizeHorseName(c.Name);
                    return candidate == normalized ||
                           candidate.EndsWith(" " + surname, StringComparison.Ordinal);
                })
                .Select(c => c.JockeyId)
                .Distinct()
                .OrderBy(id => id)
                .ToList();

            if (matches.Count == 1)
            {
                return matches[0];
            }

            if (matches.Count > 1)
            {
                Console.WriteLine($"\t\tAmbiguous Jockey.Name match for '{jockeyName}'; jockey history will be unavailable.");
                return null;
            }

            Console.WriteLine($"\t\tNo normalized match found in Jockey.Name for '{jockeyName}'; jockey history will be unavailable.");
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
