using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace HorseRacingML.Services
{
    public sealed record CourseLocation(double Latitude, double Longitude, string TimeZoneId);

    /// <summary>
    /// Approximate racecourse coordinates used to request a weather grid cell.
    /// Weather reanalysis is delivered on a 9-11 km grid, so a few hundred metres
    /// of error in these coordinates is immaterial. Names are matched with the
    /// same bracketed-text stripping and fuzzy matching the race resolver uses.
    /// </summary>
    public static class CourseLocationCatalog
    {
        private static readonly Regex NonAlphanumeric = new("[^a-z0-9]+", RegexOptions.Compiled);
        private static readonly Regex Whitespace = new("\\s+", RegexOptions.Compiled);
        private static readonly string[] Suffixes =
        {
            " racecourse",
            " race course",
            " race track",
            " racetrack"
        };

        private static readonly Dictionary<string, CourseLocation> Locations = Build();

        public static bool TryMatch(string? courseName, out CourseLocation location)
        {
            location = default!;
            var key = Normalize(courseName);
            if (key.Length == 0)
            {
                return false;
            }

            if (Locations.TryGetValue(key, out var exact))
            {
                location = exact;
                return true;
            }

            foreach (var suffix in Suffixes)
            {
                if (key.EndsWith(suffix, StringComparison.Ordinal))
                {
                    var trimmed = key[..^suffix.Length].Trim();
                    if (Locations.TryGetValue(trimmed, out var suffixed))
                    {
                        location = suffixed;
                        return true;
                    }
                }
            }

            // Fallback for values such as "Ascot Racecourse" or
            // "Newmarket (July Course)" that the exact/alias lookup missed.
            string? bestKey = null;
            foreach (var candidate in Locations.Keys)
            {
                if (candidate.Length < 4)
                {
                    continue;
                }

                // Only allow a prefix match so a foreign track such as
                // "New York" cannot be mistaken for York in the UK.
                if (!string.Equals(key, candidate, StringComparison.Ordinal) &&
                    !key.StartsWith(candidate + " ", StringComparison.Ordinal))
                {
                    continue;
                }

                if (bestKey is null || candidate.Length > bestKey.Length)
                {
                    bestKey = candidate;
                }
            }

            if (bestKey is not null)
            {
                location = Locations[bestKey];
                return true;
            }

            return false;
        }

        internal static string Normalize(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var withoutBrackets = Regex.Replace(value, "\\s*\\([^\\)]*\\)|\\s*\\[[^\\]]*\\]", " ");
            var normalized = NonAlphanumeric.Replace(withoutBrackets.Trim().ToLowerInvariant(), " ");
            return Whitespace.Replace(normalized, " ").Trim();
        }

        private static Dictionary<string, CourseLocation> Build()
        {
            var map = new Dictionary<string, CourseLocation>(StringComparer.Ordinal);
            const string uk = "Europe/London";
            const string ire = "Europe/Dublin";

            static CourseLocation Gb(double lat, double lon) => new(lat, lon, uk);
            static CourseLocation Ie(double lat, double lon) => new(lat, lon, ire);

            void Add(string key, CourseLocation location)
            {
                map[key] = location;
            }

            void AddAliases(CourseLocation location, params string[] keys)
            {
                foreach (var key in keys)
                {
                    Add(key, location);
                }
            }

            // Great Britain
            AddAliases(Gb(53.476, -2.943), "aintree");
            AddAliases(Gb(51.410, -0.678), "ascot", "royal ascot");
            AddAliases(Gb(55.463, -4.628), "ayr");
            AddAliases(Gb(53.003, -2.914), "bangor on dee", "bangor");
            AddAliases(Gb(51.405, -2.383), "bath");
            AddAliases(Gb(53.857, -0.440), "beverley");
            AddAliases(Gb(50.835, -0.147), "brighton");
            AddAliases(Gb(54.930, -2.930), "carlisle");
            AddAliases(Gb(54.201, -2.951), "cartmel");
            AddAliases(Gb(54.373, -1.641), "catterick");
            AddAliases(Gb(51.744, 0.470), "chelmsford city", "chelmsford");
            AddAliases(Gb(51.919, -2.056), "cheltenham");
            AddAliases(Gb(51.642, -2.689), "chepstow");
            AddAliases(Gb(53.187, -2.872), "chester");
            AddAliases(Gb(53.522, -1.096), "doncaster");
            AddAliases(Gb(51.306, -0.256), "epsom downs", "epsom");
            AddAliases(Gb(50.718, -3.520), "exeter");
            AddAliases(Gb(52.838, 0.858), "fakenham");
            AddAliases(Gb(51.090, 1.100), "folkestone");
            AddAliases(Gb(51.735, -4.245), "ffos las");
            AddAliases(Gb(50.855, -0.660), "fontwell park", "fontwell");
            AddAliases(Gb(50.897, -0.750), "goodwood");
            AddAliases(Gb(55.773, -4.050), "hamilton park", "hamilton");
            AddAliases(Gb(53.478, -2.630), "haydock park", "haydock");
            AddAliases(Gb(52.070, -2.730), "hereford");
            AddAliases(Gb(54.970, -2.100), "hexham");
            AddAliases(Gb(52.330, -0.200), "huntingdon");
            AddAliases(Gb(55.630, -2.430), "kelso");
            AddAliases(Gb(51.418, -0.389), "kempton park", "kempton");
            AddAliases(Gb(52.620, -1.100), "leicester");
            AddAliases(Gb(51.170, -0.080), "lingfield park", "lingfield");
            AddAliases(Gb(52.370, -2.720), "ludlow");
            AddAliases(Gb(53.390, -0.350), "market rasen");
            AddAliases(Gb(55.940, -3.050), "musselburgh");
            AddAliases(Gb(51.400, -1.320), "newbury");
            AddAliases(Gb(55.000, -1.610), "newcastle");
            AddAliases(Gb(52.240, 0.400), "newmarket", "newmarket july", "newmarket rowley", "rowley mile", "july course");
            AddAliases(Gb(50.530, -3.610), "newton abbot");
            AddAliases(Gb(52.930, -1.150), "nottingham");
            AddAliases(Gb(56.390, -3.430), "perth");
            AddAliases(Gb(50.930, -0.100), "plumpton");
            AddAliases(Gb(53.690, -1.310), "pontefract");
            AddAliases(Gb(54.610, -1.070), "redcar");
            AddAliases(Gb(54.130, -1.520), "ripon");
            AddAliases(Gb(51.070, -1.790), "salisbury");
            AddAliases(Gb(51.370, -0.370), "sandown park", "sandown");
            AddAliases(Gb(54.650, -1.450), "sedgefield");
            AddAliases(Gb(53.080, -0.990), "southwell");
            AddAliases(Gb(52.190, -1.710), "stratford upon avon", "stratford");
            AddAliases(Gb(51.020, -3.100), "taunton");
            AddAliases(Gb(54.230, -1.340), "thirsk");
            AddAliases(Gb(52.130, -0.990), "towcester");
            AddAliases(Gb(52.900, -1.860), "uttoxeter");
            AddAliases(Gb(52.270, -1.590), "warwick");
            AddAliases(Gb(53.930, -1.390), "wetherby");
            AddAliases(Gb(51.060, -2.410), "wincanton");
            AddAliases(Gb(51.480, -0.630), "windsor");
            AddAliases(Gb(52.590, -2.130), "wolverhampton");
            AddAliases(Gb(52.190, -2.220), "worcester");
            AddAliases(Gb(52.610, 1.730), "yarmouth", "great yarmouth");
            AddAliases(Gb(53.960, -1.090), "york");

            // Ireland
            AddAliases(Ie(53.630, -9.220), "ballinrobe");
            AddAliases(Ie(53.640, -6.350), "bellewstown");
            AddAliases(Ie(52.350, -7.700), "clonmel");
            AddAliases(Ie(51.880, -8.470), "cork");
            AddAliases(Ie(53.170, -6.840), "curragh");
            AddAliases(Ie(54.540, -6.170), "down royal");
            AddAliases(Ie(54.330, -5.710), "downpatrick");
            AddAliases(Ie(54.000, -6.400), "dundalk");
            AddAliases(Ie(53.480, -6.460), "fairyhouse");
            AddAliases(Ie(53.270, -9.040), "galway", "ballybrit");
            AddAliases(Ie(52.630, -7.070), "gowran park", "gowran");
            AddAliases(Ie(53.370, -7.500), "kilbeggan");
            AddAliases(Ie(52.060, -9.500), "killarney");
            AddAliases(Ie(53.680, -6.240), "laytown");
            AddAliases(Ie(53.270, -6.200), "leopardstown");
            AddAliases(Ie(52.650, -8.630), "limerick");
            AddAliases(Ie(52.450, -9.480), "listowel");
            AddAliases(Ie(52.133, -8.633), "mallow");
            AddAliases(Ie(53.220, -6.650), "naas");
            AddAliases(Ie(53.650, -6.680), "navan");
            AddAliases(Ie(53.360, -6.330), "phoenix park");
            AddAliases(Ie(53.180, -6.630), "punchestown");
            AddAliases(Ie(53.630, -8.190), "roscommon");
            AddAliases(Ie(54.270, -8.470), "sligo");
            AddAliases(Ie(52.680, -7.820), "thurles");
            AddAliases(Ie(52.470, -8.160), "tipperary");
            AddAliases(Ie(52.160, -7.150), "tramore");
            AddAliases(Ie(52.270, -9.700), "tralee");
            AddAliases(Ie(52.340, -6.460), "wexford");

            return map;
        }
    }
}
