using HorseRacingML.ML;
using Xunit;

namespace HorseRacingML.Tests
{
    public class RunStyleParserTests
    {
        [Theory]
        [InlineData("Led, ridden entering straight, headed 2f out, weakened", EarlyPosition.Led)]
        [InlineData("Made all, ridden out", EarlyPosition.Led)]
        [InlineData("Soon led, headed 3f out", EarlyPosition.Led)]
        [InlineData("Disputed lead, kept on well", EarlyPosition.Led)]
        [InlineData("Overall leader stands side, ridden over 1f out", EarlyPosition.Led)]
        [InlineData("Prominent, ridden 3f out", EarlyPosition.Pressed)]
        [InlineData("Tracked leader, ridden over 1f out", EarlyPosition.Pressed)]
        [InlineData("Close up, ridden 2f out", EarlyPosition.Pressed)]
        [InlineData("Chased winner, ridden over 1f out", EarlyPosition.Pressed)]
        [InlineData("Chased leaders, ridden 2f out", EarlyPosition.Chased)]
        [InlineData("Tracked leaders, 7th halfway", EarlyPosition.Chased)]
        [InlineData("Mid-division, pushed along", EarlyPosition.Midfield)]
        [InlineData("In touch, ridden 2f out", EarlyPosition.Midfield)]
        [InlineData("Rear of mid-division, 9th halfway", EarlyPosition.Rear)]
        [InlineData("Held up in rear, headway 2f out", EarlyPosition.Rear)]
        [InlineData("Held up in touch, headway 3f out", EarlyPosition.Midfield)]
        [InlineData("Rear, 12th halfway", EarlyPosition.Rear)]
        [InlineData("Towards rear, pushed along halfway", EarlyPosition.Rear)]
        [InlineData("Led in start, towards rear", EarlyPosition.Rear)]
        [InlineData("Reluctant to line up and had to be led in start, towards rear", EarlyPosition.Rear)]
        [InlineData("Led inside final furlong", EarlyPosition.Unknown)]
        [InlineData("Led final 2f", EarlyPosition.Unknown)]
        [InlineData("Held up, led inside final furlong", EarlyPosition.Rear)]
        [InlineData("Slowly away, outpaced", EarlyPosition.Unknown)]
        [InlineData("Mid-divison, pushed along", EarlyPosition.Midfield)]
        [InlineData("Traked leaders, ridden 2f out", EarlyPosition.Chased)]
        [InlineData("Ed, mistake 5th", EarlyPosition.Led)]
        [InlineData("Led (op 7/2 tchd 4/1), ridden out", EarlyPosition.Led)]
        public void ClassifiesEarlyPosition(string comment, EarlyPosition expected)
        {
            Assert.Equal(expected, RunStyleParser.Parse(comment).Position);
        }

        [Fact]
        public void ParsesHeldUpKeenAndSlowStartFlags()
        {
            var heldUp = RunStyleParser.Parse("Held up in rear, headway 2f out");
            Assert.True(heldUp.HeldUp);
            Assert.False(heldUp.Keen);
            Assert.False(heldUp.SlowStart);

            var keen = RunStyleParser.Parse("Took keen hold, led");
            Assert.Equal(EarlyPosition.Led, keen.Position);
            Assert.True(keen.Keen);

            var slow = RunStyleParser.Parse("Slowly away, always behind");
            Assert.Equal(EarlyPosition.Rear, slow.Position);
            Assert.True(slow.SlowStart);

            var keenToPost = RunStyleParser.Parse("Keen to post, prominent");
            Assert.Equal(EarlyPosition.Pressed, keenToPost.Position);
            Assert.False(keenToPost.Keen);
        }

        [Fact]
        public void UnknownCommentsReturnNeutralUnknown()
        {
            var empty = RunStyleParser.Parse(null);
            Assert.Equal(EarlyPosition.Unknown, empty.Position);
            Assert.False(empty.HasPosition);
            Assert.Equal(0.5f, empty.PaceScore);

            var uninformative = RunStyleParser.Parse("Outpaced");
            Assert.Equal(EarlyPosition.Unknown, uninformative.Position);
            Assert.False(uninformative.HasPosition);
        }
    }
}
