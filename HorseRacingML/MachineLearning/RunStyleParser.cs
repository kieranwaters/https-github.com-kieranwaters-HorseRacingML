using System;
using System.Text.RegularExpressions;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Early-race position derived from a post-race comment. The values are
    /// ordinal because a front-runner is closer to a prominent runner than to a
    /// hold-up horse; <see cref="Unknown"/> is deliberately not part of that
    /// ordering and must be treated as missing, not as "rear".
    /// </summary>
    public enum EarlyPosition : byte
    {
        Unknown = 0,
        Led = 1,
        Pressed = 2,
        Chased = 3,
        Midfield = 4,
        Rear = 5
    }

    /// <summary>
    /// Parsed run-style information for one past run. Position and the
    /// hold-up/keen/slow-start flags are kept separate because "held up in
    /// touch" and "held up in rear" are different positions but the same
    /// tactical style.
    /// </summary>
    public readonly record struct RunStyleInfo(
        EarlyPosition Position,
        bool HeldUp,
        bool Keen,
        bool SlowStart)
    {
        public static RunStyleInfo Unknown => new(EarlyPosition.Unknown, false, false, false);

        public bool HasPosition => Position != EarlyPosition.Unknown;

        /// <summary>
        /// Continuous 0..1 score used by the rolling average. Unknown is neutral
        /// at the point of use; callers must exclude it from the sample count.
        /// </summary>
        public float PaceScore => Position switch
        {
            EarlyPosition.Led => 1.0f,
            EarlyPosition.Pressed => 0.75f,
            EarlyPosition.Chased => 0.5f,
            EarlyPosition.Midfield => 0.3f,
            EarlyPosition.Rear => 0.1f,
            _ => 0.5f
        };
    }

    /// <summary>
    /// Deterministic keyword classifier for the race comments collected from
    /// Sporting Life and the historical Kaggle results. The vocabulary is small
    /// and formulaic, so a rule set is more auditable than a model and can be
    /// versioned with <see cref="Version"/> when the rules change.
    ///
    /// The parser only reads the early-position clause. It deliberately ignores
    /// outcome words such as "weakened", "headway", "ran on" or "no extra"; those
    /// describe what happened later in the same race and would leak the result.
    /// </summary>
    public static class RunStyleParser
    {
        public const byte Version = 1;

        private const RegexOptions Options =
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.CultureInvariant;

        private static readonly Regex BracketText =
            new(@"\s*\([^)]*\)|\s*\[[^\]]*\]", Options);

        private static readonly Regex BettingToken =
            new(@"\b(?:op|tchd|low|high)\s+\d+(?:\s*[/-]\s*\d+)?\b", Options);

        private static readonly Regex NonBreakingSpace =
            new(@"[\u00A0\u2007\u202F]", RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex Whitespace =
            new(@"\s+", RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex ClauseSplit =
            new(@"\s*[,;]\s*|\s+-\s+|\s*[\u2013\u2014]\s*", RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex TypoTracked =
            new(@"\btraked\b", Options);

        private static readonly Regex TypoLed =
            new(@"(^|[\s,;\-])ed(?=[\s,;\-])", Options);

        private static readonly Regex KeenToPost =
            new(@"\bkeen\s+(?:to|going\s+to)\s+post\b", Options);

        private static readonly Regex LedToPost =
            new(@"\b(?:led|lead)\s+(?:in|to|away\s+to)\s+(?:the\s+)?(?:start|post)\b|\bhad\s+to\s+be\s+led\b|\bled\s+up\b", Options);

        private static readonly Regex LateLed =
            new(@"\b(?:inside\s+final|final\s+(?:furlong|stride|\d+\s*(?:f|yds?|yards?))|close\s+home|near\s+finish|last\s+\d+)\b", Options);

        private static readonly Regex Led =
            new(@"\b(?:led|make\s+all|made\s+all|make\s+virtually\s+all|made\s+virtually\s+all|makeall|madeall|set\s+(?:the\s+)?(?:pace|a\s+(?:strong|steady|modest|furious|fast|good|sedate|slow)\s+pace|(?:strong|steady|modest|furious|fast|good|sedate|slow)\s+pace)|took\s+them\s+along|went\s+to\s+the\s+front|front\s+rank|front[- ]running|made\s+(?:the\s+)?running|disputed\s+lead|shared\s+lead|overall\s+leader|leader\s+stands\s+side)\b", Options);

        private static readonly Regex Pressed =
            new(@"\b(?:pressed|pressing|with\s+(?:the\s+)?leader|joined\s+leader|prominent(?:ly)?|close(?:[-\s]+up|[-\s]+(?:second|third|fourth|fifth|2nd|3rd|4th|5th))|tracked\s+leader|tracked\s+winner|tracking\s+leader|chased\s+leader|chased\s+winner|chased\s+clear\s+winner|chasing\s+leader|chasing\s+winner|handy|raced\s+handy|on\s+the\s+pace|upsides\s+leader|raced\s+in\s+(?:second|2nd))\b", Options);

        private static readonly Regex Chased =
            new(@"\b(?:chased\s+(?:the\s+)?(?:clear\s+)?leaders|chased\s+leading|tracked\s+(?:the\s+)?leaders|tracked\s+leading|tracked\s+front\s+(?:pair|two|trio)|tracked\s+leading\s+(?:pair|trio|group)|tracking\s+leaders|chasing\s+(?:the\s+)?leaders|chasing\s+leading|stalked|(?:in|raced\s+in|raced)\s+(?:3rd|third|4th|fourth|5th|fifth|6th|sixth))\b", Options);

        private static readonly Regex Midfield =
            new(@"\b(?:mid[- ]division|midfield|in\s+touch|middle\s+of\s+the\s+field|in\s+the\s+middle|centre|center|(?:in|raced\s+in|raced)\s+(?:7th|seventh|8th|eighth|9th|ninth|10th|tenth|11th|eleventh))\b", Options);

        private static readonly Regex RearStrong =
            new(@"\b(?:in\s+rear|towards\s+rear|rear\s+of|rear\s+mid|rear[- ]division|at\s+the\s+rear)\b", Options);

        private static readonly Regex HeldUp =
            new(@"\b(?:held\s+up|held-up|waited\s+(?:with|in)|settled\s+(?:behind|in\s+rear|towards\s+rear)|steadied\s+rear|dropped\s+(?:in|out))\b", Options);

        private static readonly Regex Rear =
            new(@"\b(?:rear|behind|last|detached|tailed\s+off|lost\s+touch|well\s+adrift|adrift)\b", Options);

        private static readonly Regex Keen =
            new(@"\b(?:keen|keenly|pulled\s+(?:very\s+)?hard|pulling\s+hard|raced\s+freely|took\s+keen\s+hold|took\s+(?:a\s+)?(?:strong\s+)?hold|strong\s+hold|over[- ]racing|lit\s+up)\b", Options);

        private static readonly Regex SlowStart =
            new(@"\b(?:slowly\s+away|dwelt|missed\s+break|awkwardly\s+away|steadied\s+start|reared|left\s+at\s+start|very\s+slowly|slowly\s+into\s+stride)\b", Options);

        public static RunStyleInfo Parse(string? comment)
        {
            if (string.IsNullOrWhiteSpace(comment))
            {
                return RunStyleInfo.Unknown;
            }

            var text = Normalize(comment);
            if (text.Length == 0)
            {
                return RunStyleInfo.Unknown;
            }

            // Flags are independent of position and are safe to read from the
            // whole comment. "Keen to post" is a pre-race behaviour, not a
            // running style, so remove it before testing.
            var keenText = KeenToPost.Replace(text, " ");
            bool heldUp = HeldUp.IsMatch(text);
            bool keen = Keen.IsMatch(keenText);
            bool slowStart = SlowStart.IsMatch(text);

            // "Led in start" / "had to be led" describe the preliminaries, not
            // the race. Remove them before looking for a front-running clause.
            var positionText = LedToPost.Replace(text, " ");

            foreach (var rawClause in ClauseSplit.Split(positionText))
            {
                var clause = rawClause.Trim();
                if (clause.Length == 0)
                {
                    continue;
                }

                if (Led.IsMatch(clause))
                {
                    // "Led inside final furlong" is a late move, not an early
                    // front-run. The first clause is normally the early position,
                    // but this guards against the occasional exception.
                    if (!LateLed.IsMatch(clause))
                    {
                        return new RunStyleInfo(EarlyPosition.Led, heldUp, keen, slowStart);
                    }

                    continue;
                }

                if (Pressed.IsMatch(clause))
                {
                    return new RunStyleInfo(EarlyPosition.Pressed, heldUp, keen, slowStart);
                }

                if (Chased.IsMatch(clause))
                {
                    return new RunStyleInfo(EarlyPosition.Chased, heldUp, keen, slowStart);
                }

                if (RearStrong.IsMatch(clause))
                {
                    return new RunStyleInfo(EarlyPosition.Rear, heldUp, keen, slowStart);
                }

                if (Midfield.IsMatch(clause))
                {
                    return new RunStyleInfo(EarlyPosition.Midfield, heldUp, keen, slowStart);
                }

                if (HeldUp.IsMatch(clause))
                {
                    return new RunStyleInfo(EarlyPosition.Rear, heldUp, keen, slowStart);
                }

                if (Rear.IsMatch(clause))
                {
                    return new RunStyleInfo(EarlyPosition.Rear, heldUp, keen, slowStart);
                }
            }

            return new RunStyleInfo(EarlyPosition.Unknown, heldUp, keen, slowStart);
        }

        private static string Normalize(string comment)
        {
            var text = BracketText.Replace(comment, " ");
            text = BettingToken.Replace(text, " ");
            text = NonBreakingSpace.Replace(text, " ");
            text = text.Replace('\r', ' ').Replace('\n', ' ').Trim().ToLowerInvariant();
            text = Whitespace.Replace(text, " ");
            text = text.Replace("mid-divison", "mid-division");
            text = text.Replace("mid devision", "mid-division");
            text = TypoTracked.Replace(text, "tracked");
            text = TypoLed.Replace(text, "$1led");
            return text.Trim();
        }
    }
}
