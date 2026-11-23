using System;
using System.Collections.Generic;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Static track geometry for the courses in the stored history. Values are
    /// deliberately coarse (handedness, whether the course has a straight
    /// sprint track, approximate run-in and the months of its main festival).
    /// They are pre-race facts, so they are safe to use live.
    /// </summary>
    public readonly record struct CourseGeometry(
        bool LeftHanded,
        bool RightHanded,
        bool StraightCourse,
        double RunInYards,
        IReadOnlySet<int> FestivalMonths);

    public static class CourseGeometryCatalog
    {
        private static readonly Dictionary<string, CourseGeometry> Courses =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ["ascot"] = Geo(false, true, true, 220, 6, 10),
                ["aintree"] = Geo(true, false, false, 250, 4),
                ["ayr"] = Geo(true, false, true, 200, 9),
                ["bath"] = Geo(true, false, true, 180, 0),
                ["beverley"] = Geo(true, false, true, 200, 0),
                ["brighton"] = Geo(true, false, true, 180, 0),
                ["carlisle"] = Geo(true, false, true, 200, 0),
                ["catterick"] = Geo(true, false, true, 180, 0),
                ["cheltenham"] = Geo(true, false, false, 220, 3),
                ["chepstow"] = Geo(true, false, false, 220, 12),
                ["chester"] = Geo(true, false, false, 150, 5),
                ["doncaster"] = Geo(true, false, true, 220, 9),
                ["epsom"] = Geo(true, false, true, 200, 6),
                ["exeter"] = Geo(true, false, false, 220, 0),
                ["folkestone"] = Geo(true, false, false, 200, 0),
                ["fontwell"] = Geo(true, false, false, 200, 0),
                ["goodwood"] = Geo(false, true, true, 220, 7, 8),
                ["hamilton"] = Geo(false, true, true, 200, 0),
                ["haydock"] = Geo(true, false, true, 220, 5, 6),
                ["hereford"] = Geo(true, false, false, 200, 0),
                ["kempton"] = Geo(false, true, true, 200, 12),
                ["kelso"] = Geo(true, false, false, 220, 0),
                ["leicester"] = Geo(true, false, true, 200, 0),
                ["lingfield"] = Geo(true, false, true, 200, 0),
                ["ludlow"] = Geo(true, false, false, 200, 0),
                ["market rasen"] = Geo(true, false, false, 200, 0),
                ["musselburgh"] = Geo(false, true, true, 200, 0),
                ["newbury"] = Geo(true, false, true, 220, 2, 5, 9),
                ["newcastle"] = Geo(true, false, true, 220, 0),
                ["newmarket"] = Geo(false, true, true, 220, 5, 10),
                ["nottingham"] = Geo(true, false, true, 200, 0),
                ["perth"] = Geo(false, true, false, 220, 4),
                ["pontefract"] = Geo(true, false, true, 200, 0),
                ["redcar"] = Geo(true, false, true, 200, 0),
                ["ripon"] = Geo(false, true, true, 200, 0),
                ["salisbury"] = Geo(true, false, true, 220, 0),
                ["sandown"] = Geo(false, true, true, 220, 4, 7),
                ["sedgefield"] = Geo(true, false, false, 200, 0),
                ["southwell"] = Geo(true, false, false, 200, 0),
                ["thirsk"] = Geo(true, false, true, 200, 0),
                ["towcester"] = Geo(true, false, false, 200, 0),
                ["uttoxeter"] = Geo(true, false, false, 200, 0),
                ["warwick"] = Geo(true, false, false, 200, 0),
                ["wetherby"] = Geo(true, false, false, 220, 0),
                ["wincanton"] = Geo(true, false, false, 200, 0),
                ["windsor"] = Geo(true, false, true, 200, 6),
                ["wolverhampton"] = Geo(true, false, false, 200, 0),
                ["worcester"] = Geo(true, false, false, 220, 0),
                ["yarmouth"] = Geo(true, false, true, 200, 0),
                ["york"] = Geo(true, false, true, 220, 8),
                ["curragh"] = Geo(false, true, true, 220, 6, 7),
                ["leopardstown"] = Geo(true, false, true, 220, 12),
                ["punchestown"] = Geo(false, true, false, 220, 4, 5),
                ["fairyhouse"] = Geo(false, true, false, 220, 4),
                ["galway"] = Geo(false, true, false, 180, 7, 8),
                ["listowel"] = Geo(false, true, false, 200, 9),
                ["cork"] = Geo(false, true, true, 220, 0),
                ["naas"] = Geo(false, true, true, 220, 0),
                ["navan"] = Geo(false, true, false, 220, 0),
                ["dundalk"] = Geo(true, false, false, 200, 0),
                ["down royal"] = Geo(false, true, false, 220, 11),
                ["downpatrick"] = Geo(false, true, false, 200, 0),
                ["limerick"] = Geo(false, true, false, 220, 12),
                ["tipperary"] = Geo(false, true, false, 220, 0),
                ["thurles"] = Geo(false, true, false, 220, 0),
                ["clonmel"] = Geo(false, true, false, 200, 0),
                ["wexford"] = Geo(false, true, false, 200, 0),
                ["sligo"] = Geo(false, true, false, 200, 0),
                ["roscommon"] = Geo(false, true, false, 200, 0),
                ["ballinrobe"] = Geo(false, true, false, 200, 0),
                ["bellewstown"] = Geo(false, true, false, 200, 0),
                ["tramore"] = Geo(false, true, false, 200, 8),
                ["killarney"] = Geo(false, true, false, 200, 5, 7),
                ["laytown"] = Geo(true, false, true, 100, 9)
            };

        private static readonly string[] FestivalKeywords =
        {
            "festival", "gold cup", "grand national", "derby", "guineas",
            "st leger", "ebor", "glorious", "royal ascot", "champion",
            "christmas", "easter", "welsh national", "triumph", "world hurdle",
            "queen mother", "king george", "arc", "poule"
        };

        public static bool TryGet(string? courseName, out CourseGeometry geometry)
        {
            geometry = default;
            if (string.IsNullOrWhiteSpace(courseName))
            {
                return false;
            }

            return Courses.TryGetValue(courseName.Trim(), out geometry);
        }

        public static bool LooksLikeFestivalRace(string? title)
        {
            if (string.IsNullOrWhiteSpace(title))
            {
                return false;
            }

            foreach (var keyword in FestivalKeywords)
            {
                if (title.Contains(keyword, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        private static CourseGeometry Geo(
            bool left,
            bool right,
            bool straight,
            double runIn,
            params int[] festivalMonths)
            => new(left, right, straight, runIn, new HashSet<int>(festivalMonths));
    }
}
