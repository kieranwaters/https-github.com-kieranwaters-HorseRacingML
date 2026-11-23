using System;
using System.Collections.Generic;
using System.Linq;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Independently switchable blocks of model features. <see cref="Default"/>
    /// is the schema used when no explicit Training:FeatureBlocks setting is
    /// present; the remaining blocks can still be added or removed individually.
    /// </summary>
    [Flags]
    public enum FeatureBlocks
    {
        None = 0,
        Pace = 1 << 0,
        TrainerForm = 1 << 1,
        MarketMovement = 1 << 2,
        WeatherBalance = 1 << 3,
        RaceContext = 1 << 4,
        FormContext = 1 << 5,
        PriceData = 1 << 6,
        WeatherExtended = 1 << 7,
        TrackGeometry = 1 << 8,
        DeepForm = 1 << 9,
        MarketLiquidity = 1 << 10,
        Sectionals = 1 << 11,
        Breeding = 1 << 12,
        Default = Pace | MarketMovement | PriceData | RaceContext | TrackGeometry | Sectionals | Breeding,
    }

    public static class FeatureBlockNames
    {
        public static FeatureBlocks Parse(string? value, FeatureBlocks fallback = FeatureBlocks.Default)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return fallback;
            }

            var blocks = FeatureBlocks.None;
            var names = value.Split(
                new[] { ',', ';', '|', ' ', '\t' },
                StringSplitOptions.RemoveEmptyEntries);

            foreach (var rawName in names)
            {
                var name = rawName.Trim();
                if (name.Equals("none", StringComparison.OrdinalIgnoreCase))
                {
                    return FeatureBlocks.None;
                }

                if (name.Equals("default", StringComparison.OrdinalIgnoreCase))
                {
                    return FeatureBlocks.Default;
                }

                if (name.Equals("all", StringComparison.OrdinalIgnoreCase))
                {
                    return FeatureBlocks.Pace
                        | FeatureBlocks.TrainerForm
                        | FeatureBlocks.MarketMovement
                        | FeatureBlocks.WeatherBalance
                        | FeatureBlocks.RaceContext
                        | FeatureBlocks.FormContext
                        | FeatureBlocks.PriceData
                        | FeatureBlocks.WeatherExtended
                        | FeatureBlocks.TrackGeometry
                        | FeatureBlocks.DeepForm
                        | FeatureBlocks.MarketLiquidity
                        | FeatureBlocks.Sectionals
                        | FeatureBlocks.Breeding;
                }

                blocks |= name.ToLowerInvariant() switch
                {
                    "pace" or "runstyle" => FeatureBlocks.Pace,
                    "trainer" or "trainerform" or "trainerage" or "age" => FeatureBlocks.TrainerForm,
                    "market" or "marketmovement" or "pricemovement" => FeatureBlocks.MarketMovement,
                    "weather" or "weatherbalance" => FeatureBlocks.WeatherBalance,
                    "race" or "racecontext" or "course" => FeatureBlocks.RaceContext,
                    "pricedata" or "rating" or "ratings" or "weight" or "purse" or "headgear" => FeatureBlocks.PriceData,
                    "weatherextended" or "soil" or "soilmoisture" or "climate" or "wind" => FeatureBlocks.WeatherExtended,
                    "track" or "trackgeometry" or "geometry" or "festival" => FeatureBlocks.TrackGeometry,
                    "deepform" or "headtohead" or "h2h" or "peak" or "bounce" or "secondup" => FeatureBlocks.DeepForm,
                    "marketliquidity" or "liquidity" or "betfair" or "betfairsp" or "bsp" or "volume" or "tradedvolume" => FeatureBlocks.MarketLiquidity,
                    "sectionals" or "sectional" or "sectionaltimes" or "raceiq" or "splits" => FeatureBlocks.Sectionals,
                    "breeding" or "pedigree" or "sire" or "dam" or "damsire" => FeatureBlocks.Breeding,
                    "form" or "formcontext" => FeatureBlocks.FormContext,
                    _ => FeatureBlocks.None
                };
            }

            return blocks;
        }

        public static string Format(FeatureBlocks blocks)
        {
            if (blocks == FeatureBlocks.None)
            {
                return "None";
            }

            var names = new List<string>();
            if (blocks.HasFlag(FeatureBlocks.Pace)) names.Add("Pace");
            if (blocks.HasFlag(FeatureBlocks.TrainerForm)) names.Add("TrainerForm");
            if (blocks.HasFlag(FeatureBlocks.MarketMovement)) names.Add("MarketMovement");
            if (blocks.HasFlag(FeatureBlocks.WeatherBalance)) names.Add("WeatherBalance");
            if (blocks.HasFlag(FeatureBlocks.RaceContext)) names.Add("RaceContext");
            if (blocks.HasFlag(FeatureBlocks.FormContext)) names.Add("FormContext");
            if (blocks.HasFlag(FeatureBlocks.PriceData)) names.Add("PriceData");
            if (blocks.HasFlag(FeatureBlocks.WeatherExtended)) names.Add("WeatherExtended");
            if (blocks.HasFlag(FeatureBlocks.TrackGeometry)) names.Add("TrackGeometry");
            if (blocks.HasFlag(FeatureBlocks.DeepForm)) names.Add("DeepForm");
            if (blocks.HasFlag(FeatureBlocks.MarketLiquidity)) names.Add("MarketLiquidity");
            if (blocks.HasFlag(FeatureBlocks.Sectionals)) names.Add("Sectionals");
            if (blocks.HasFlag(FeatureBlocks.Breeding)) names.Add("Breeding");
            return string.Join(",", names);
        }

        public static IEnumerable<FeatureBlocks> IndividualBlocks()
        {
            yield return FeatureBlocks.Pace;
            yield return FeatureBlocks.TrainerForm;
            yield return FeatureBlocks.MarketMovement;
            yield return FeatureBlocks.WeatherBalance;
            yield return FeatureBlocks.RaceContext;
            yield return FeatureBlocks.FormContext;
            yield return FeatureBlocks.PriceData;
            yield return FeatureBlocks.WeatherExtended;
            yield return FeatureBlocks.TrackGeometry;
            yield return FeatureBlocks.DeepForm;
            yield return FeatureBlocks.MarketLiquidity;
            yield return FeatureBlocks.Sectionals;
            yield return FeatureBlocks.Breeding;
        }
    }

    /// <summary>
    /// Stable feature names for each block. Keeping the list in one
    /// place means the schema builder, live feature writer and benchmark
    /// report cannot drift apart.
    /// </summary>
    public static class FeatureBlockCatalog
    {
        public static readonly string[] TrainerForm =
        {
            "TrainerWinRate",
            "TrainerWinRateRecentDays",
            "TrainerWinRateLast50",
            "TrainerSurfaceWinRate",
            "TrainerSurfaceAvgNorm",
            "LastTrainerSurfaceNormPos",
            "TrainerGoingWinRate",
            "TrainerGoingAvgNorm",
            "LastTrainerGoingNormPos",
            "TrainerDistanceBucketWinRate",
            "TrainerDistanceBucketAvgNorm",
            "LastTrainerDistanceBucketNormPos",
            "TrainerCourseWinRate",
            "TrainerClassWinRate",
            "TrainerClassAvgNorm",
            "LastTrainerClassNormPos",
            "TrainerJockeyWinRate",
            "TrainerJockeySurfaceWinRate",
            "Age",
            "AgeRelative",
            "AgeProgression"
        };

        // The Sporting Life historical_odds series now populates both sides of
        // the pre-off range, so the block carries drift, range position and
        // explicit presence flags.
        public static readonly string[] MarketMovement =
        {
            "MarketMoveOpeningToLowLog",
            "MarketMoveOpeningToHighLog",
            "MarketMoveOpeningMinusLow",
            "MarketMoveShortenedFlag",
            "MarketMoveOpeningRankRelative",
            "MarketMoveRangeLog",
            "MarketMoveOpeningInRange",
            "MarketMoveBackedFlag",
            "MarketMoveDriftFlag",
            "MarketMoveOpeningMissing",
            "MarketMoveLowMissing",
            "MarketMoveHighMissing"
        };

        public static readonly string[] WeatherBalance =
        {
            "WeatherBalance7dMm",
            "WeatherBalance3dMm",
            "WeatherBalanceRainEt0Ratio7",
            "WeatherBalanceRainAccel7",
            "WeatherBalanceDryness7",
            "WeatherBalanceRain1dShare"
        };

        public static readonly string[] RaceContext =
        {
            "RaceContextDrawBiasGoing",
            "RaceContextMeetingDay",
            "RaceContextMeetingRaceIndex"
        };

        // String/categorical members are discovered by the metadata
        // accumulator because they need a category map.
        public static readonly string[] RaceContextCategorical =
        {
            "RaceContextCourse",
            "RaceContextCourseDistance"
        };

        public static readonly string[] PriceData =
        {
            "OfficialRating",
            "OfficialRatingRelative",
            "OfficialRatingMissing",
            "RatingDiffFromField",
            "OfficialRatingVsFieldMax",
            "OfficialRatingRankRelative",
            "FieldRatingMean",
            "FieldRatingMax",
            "FieldRatingMin",
            "FieldRatingRange",
            "FieldRatingStdDev",
            "FieldRatingTop3Mean",
            "FieldRatingKnownCount",
            "FieldRatingCoverage",
            "WeightLbs",
            "WeightDiffFromMean",
            "WeightMissing",
            "IsTopWeight",
            "IsBottomWeight",
            "WeightChangeFromLast",
            "PurseLevel",
            "PurseLog",
            "HeadgearFlag",
            "HeadgearFirstTime",
            "MedicationFlag"
        };

        public static readonly string[] WeatherExtended =
        {
            "WeatherSoilMoisture",
            "WeatherSoilMoistureDeep",
            "WeatherSoilMoistureAnomaly",
            "WeatherRainAnomaly14d",
            "WeatherWindDirectionDegrees",
            "WeatherWindDirectionSin",
            "WeatherWindDirectionCos",
            "WeatherHumidity",
            "WeatherDewPointC",
            "WeatherFrostDaysPrev7",
            "WeatherRainHoursPrev1"
        };

        public static readonly string[] TrackGeometry =
        {
            "TrackGeometryKnown",
            "TrackGeometryLeftHanded",
            "TrackGeometryRightHanded",
            "TrackGeometryStraight",
            "TrackGeometryRunInNormalized",
            "TrackGeometryFestivalRace",
            "TrackGeometryFestivalMeeting"
        };

        public static readonly string[] DeepForm =
        {
            "DeepFormSecondUpAfterLayoff",
            "DeepFormPeakSpeedRatio",
            "DeepFormLastRunSpeedRatio",
            "DeepFormBounceFlag",
            "DeepFormH2HStarts",
            "DeepFormH2HWins",
            "DeepFormH2HWinRate",
            "DeepFormH2HNormPos"
        };

        // BSP itself is stored on RunnerResult but deliberately excluded from
        // the model: it is only known at the off. These are the pre-off
        // liquidity and exchange price-range features.
        public static readonly string[] MarketLiquidity =
        {
            "BetfairMarketMatchedLog",
            "BetfairRunnerVolumeLog",
            "BetfairRunnerVolumeShare",
            "BetfairPreOffRangeLog",
            "BetfairProjectedSPLog",
            "BetfairAvailableBackLog",
            "BetfairAvailableLayLog",
            "BetfairBookPressure",
            "BetfairLiquidityMissing",
            "BetfairExchangeRangeMissing",
            "BetfairProjectedSPMissing"
        };

        // Sectional times are post-race information. Only a horse's prior
        // sectionals enter the feature vector; the current race's splits are
        // used exclusively to advance the rolling state.
        public static readonly string[] Sectionals =
        {
            "SectionalHasHistory",
            "SectionalStarts",
            "SectionalLastFinishSpeedPct",
            "SectionalAvgFinishSpeedPctLast3",
            "SectionalAvgFinishSpeedPctLast5",
            "SectionalBestFinishSpeedPctLast5",
            "SectionalFinishSpeedTrendLast5",
            "SectionalLast3FSecondsLast",
            "SectionalAvgLast3FSecondsLast5",
            "SectionalLast3FPercentileLast",
            "SectionalAvgLast3FPercentileLast3",
            "SectionalAvgLast3FPercentileLast5",
            "SectionalBestLast3FPercentileLast5",
            "SectionalClosingRankNormLast",
            "SectionalAvgClosingRankNormLast5",
            "SectionalEarlyRankNormLast",
            "SectionalAvgEarlyRankNormLast5",
            "SectionalPaceStyleLast",
            "SectionalAvgPaceStyleLast5",
            "SectionalFieldKnownCount",
            "SectionalFieldAvgFinishSpeedPct",
            "SectionalFieldMaxFinishSpeedPct",
            "SectionalFieldMinFinishSpeedPct",
            "SectionalOwnFinishSpeedMinusFieldMean",
            "SectionalFieldAvgLast3FPercentile",
            "SectionalFieldMaxLast3FPercentile",
            "SectionalFieldMinLast3FPercentile",
            "SectionalOwnLast3FPercentileMinusFieldMean",
            "SectionalFieldAvgClosingRankNorm",
            "SectionalOwnClosingRankMinusFieldMean",
            "SectionalFieldAvgPaceStyle",
            "SectionalOwnPaceStyleMinusFieldMean",
            "SectionalFinishSpeedRankRelative",
            "SectionalLast3FPercentileRankRelative",
            "SectionalClosingRankRankRelative"
        };

        // The raw sire/dam/damsire identifiers are excluded from the feature
        // schema; these are the rolling progeny statistics built from them.
        public static readonly string[] Breeding =
        {
            "PedigreeKnown",
            "PedigreeSireStarts",
            "PedigreeSireWinRate",
            "PedigreeSireAvgNormPos",
            "PedigreeSireLastNormPos",
            "PedigreeSireSurfaceStarts",
            "PedigreeSireSurfaceWinRate",
            "PedigreeSireGoingStarts",
            "PedigreeSireGoingWinRate",
            "PedigreeSireDistanceStarts",
            "PedigreeSireDistanceWinRate",
            "PedigreeDamStarts",
            "PedigreeDamWinRate",
            "PedigreeDamAvgNormPos",
            "PedigreeDamLastNormPos",
            "PedigreeDamSireStarts",
            "PedigreeDamSireWinRate",
            "PedigreeDamSireAvgNormPos",
            "PedigreeSireWinRateRankRelative",
            "PedigreeDamWinRateRankRelative",
            "PedigreeFieldKnownCount",
            "PedigreeFieldSireWinRateMean",
            "PedigreeOwnSireWinRateMinusFieldMean",
            "PedigreeFieldDamWinRateMean",
            "PedigreeOwnDamWinRateMinusFieldMean"
        };

        public static readonly string[] FormContext =
        {
            "FormLastRunNonFinisher",
            "FormNonFinishRateLast5",
            "FormNonFinishRateLast10",
            "FormCompletedStarts",
            "FormAvgNormPosCompletedLast5",
            "FormHasCompletedHistory",
            "FormAvgDaysBetweenRuns",
            "FormLayoffVsTypical",
            "FormIsSecondCareerStart",
            "FormJockeyChanged",
            "FormJockeyWinRateDelta",
            "FormTrainerChanged",
            "FormTrainerWinRateDelta",
            "FormLastRunSameGoing",
            "FormLastRunSameCourse",
            "FormLastRunSameDistance",
            "FormClassDrop",
            "FormClassRise",
            "FormBestClassWon",
            "FormClassVsBestWon",
            "FormLastRunWasWin",
            "FormLastRunPlaced"
        };
    }
}
