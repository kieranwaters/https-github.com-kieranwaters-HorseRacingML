using System;
using System.Collections.Generic;
using System.Linq;
using HorseRacingML.Models;
using HorseRacingML.Scraping;
using Xunit;

namespace HorseRacingML.Tests
{
    public class AtTheRacesSectionalTests
    {
        [Fact]
        public void ResultsPageLinksExposeTheRaceIdAndLocalTime()
        {
            const string html = @"
<article class=""panel-content--medium"">
  <h3><a href=""/racecard/Southwell/28-March-2024/1930"">19:30 - Test Handicap</a></h3>
  <a href=""/form/horse/Silver-Trumpet/GB/3530129?raceid=1424731"">Silver Trumpet</a>
</article>
<article class=""panel-content--medium"">
  <h3><a href=""/racecard/Chelmsford-City/05-November-2026/1815"">18:15 - Another Race</a></h3>
  <span data-video-raceid=""1629820""></span>
</article>";

            var links = AtTheRacesSectionalParser.ParseResultsPage(html);

            Assert.Equal(2, links.Count);
            var southwell = links.Single(link => link.RaceId == 1424731);
            Assert.Equal("Southwell", southwell.Course);
            Assert.Equal(new DateTime(2024, 3, 28), southwell.LocalDate);
            Assert.Equal(new TimeSpan(19, 30, 0), southwell.LocalTime);
            Assert.Equal("/racecard/Southwell/28-March-2024/1930", southwell.PagePath);

            var chelmsford = links.Single(link => link.RaceId == 1629820);
            Assert.Equal("Chelmsford City", chelmsford.Course);
            Assert.Equal(new TimeSpan(18, 15, 0), chelmsford.LocalTime);
        }

        [Fact]
        public void SectionalPageBuildsLastFurlongFeaturesAndPercentiles()
        {
            const string sectionals = @"
<div class=""card-header"">
  <div class=""card-header__th card-cell card-cell--sectional-header"">Start-2f</div>
  <div class=""card-header__th card-cell card-cell--sectional-header"">2f-1f</div>
  <div class=""card-header__th card-cell card-cell--sectional-header"">1f-Finish</div>
  <div class=""card-header__th card-cell card-cell--sectional-header"">Finish</div>
</div>
<div class=""card-entry"" data-officialplacing=""1"">
  <h2 class=""h7 horse__details flush"">
    <span class=""p--small visible--inline-block"">3.</span>
    <a class=""horse__link"" href=""#"">Alpha</a>
  </h2>
  <div class=""card-cell card-cell--sectional-time"">
    <div class=""card-sectional card-sectional--even"">
      <div class=""card-sectional__inner"" title=""Running Total: 12.00""><span class=""visible"">12.00</span></div>
    </div>
  </div>
  <div class=""card-cell card-cell--sectional-time"">
    <div class=""card-sectional card-sectional--even-to-fast"">
      <div class=""card-sectional__inner"" title=""Running Total: 24.00""><span class=""visible"">12.00</span><span class=""text-pill"">Front</span></div>
    </div>
  </div>
  <div class=""card-cell card-cell--sectional-time"">
    <div class=""card-sectional card-sectional--fast"">
      <div class=""card-sectional__inner"" title=""Running Total: 36.00""><span class=""visible"">12.00</span></div>
    </div>
  </div>
  <div class=""card-cell card-cell--sectional-time card-cell--sectional-time-3-sectionals"">
    <div class=""card-sectional card-sectional--finish""><div class=""card-sectional__inner"">36.00</div></div>
  </div>
</div>
<div class=""card-entry"" data-officialplacing=""2"">
  <h2 class=""h7 horse__details flush"">
    <span class=""p--small visible--inline-block"">4.</span>
    <a class=""horse__link"" href=""#"">Beta</a>
  </h2>
  <div class=""card-cell card-cell--sectional-time"">
    <div class=""card-sectional card-sectional--even"">
      <div class=""card-sectional__inner"" title=""Running Total: 12.50""><span class=""visible"">12.50</span></div>
    </div>
  </div>
  <div class=""card-cell card-cell--sectional-time"">
    <div class=""card-sectional card-sectional--slow"">
      <div class=""card-sectional__inner"" title=""Running Total: 25.00""><span class=""visible"">12.50</span></div>
    </div>
  </div>
  <div class=""card-cell card-cell--sectional-time"">
    <div class=""card-sectional card-sectional--slow"">
      <div class=""card-sectional__inner"" title=""Running Total: 37.50""><span class=""visible"">12.50</span></div>
    </div>
  </div>
  <div class=""card-cell card-cell--sectional-time card-cell--sectional-time-3-sectionals"">
    <div class=""card-sectional card-sectional--finish""><div class=""card-sectional__inner"">37.50</div></div>
  </div>
</div>";

            const string tools = @"
<div class=""card-entry"">
  <a class=""horse__link"" href=""#"">Alpha</a>
  <div class=""card-cell card-cell--horse-finish"">101.50</div>
</div>
<div class=""card-entry"">
  <a class=""horse__link"" href=""#"">Beta</a>
  <div class=""card-cell card-cell--horse-finish"">99.00</div>
</div>";

            var sourceRunners = AtTheRacesSectionalParser.ParseSectionalPage(sectionals);
            Assert.Equal(2, sourceRunners.Count);

            var percentages = AtTheRacesSectionalParser.ParseFinishPercentages(tools);
            var databaseRunners = new List<RaceRunnerReference>
            {
                new() { RaceId = 10, HorseId = 1, HorseName = "Alpha", SaddleclothNumber = 3 },
                new() { RaceId = 10, HorseId = 2, HorseName = "Beta", SaddleclothNumber = 4 }
            };

            var rows = AtTheRacesSectionalParser.BuildRunnerSectionals(
                sourceRunners,
                databaseRunners,
                raceId: 10,
                atrRaceId: 1424731,
                percentages);

            Assert.Equal(2, rows.Count);
            var alpha = rows.Single(row => row.HorseId == 1);
            Assert.Equal((byte?)3, alpha.SectionCount);
            Assert.Equal(12f, alpha.Last1FSeconds!.Value, 2);
            Assert.Equal(24f, alpha.Last2FSeconds!.Value, 2);
            Assert.Equal(36f, alpha.Last3FSeconds!.Value, 2);
            Assert.Equal(1f, alpha.Last1FPercentile!.Value, 2);
            Assert.Equal(1f, alpha.Last3FPercentile!.Value, 2);
            Assert.Equal(1f, alpha.EarlyPositionNorm!.Value, 2);
            Assert.Equal(1f, alpha.ClosingPositionNorm!.Value, 2);
            Assert.Equal(0f, alpha.PaceStyle!.Value, 2);
            Assert.Equal(101.5f, alpha.FinishSpeedPct!.Value, 2);
            Assert.Equal("AtTheRaces", alpha.Source);
            Assert.Equal("1424731", alpha.SourceRaceId);
            Assert.Contains("AtTheRaces", alpha.RawJson);

            var beta = rows.Single(row => row.HorseId == 2);
            Assert.Equal(0f, beta.Last1FPercentile!.Value, 2);
            Assert.Equal(0f, beta.Last3FPercentile!.Value, 2);
            Assert.Equal(0f, beta.ClosingPositionNorm!.Value, 2);
            Assert.Equal(99f, beta.FinishSpeedPct!.Value, 2);
        }

        [Fact]
        public void AtTheRacesCourseNamesMatchTheStoredNames()
        {
            Assert.True(RacingTvDataParser.CourseNamesMatch("Chelmsford City", "Chelmsford City"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Lingfield", "Lingfield Park"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Epsom Downs", "Epsom"));
            Assert.True(RacingTvDataParser.CourseNamesMatch("Catterick", "Catterick Bridge"));
            Assert.False(RacingTvDataParser.CourseNamesMatch("Southwell", "Newcastle"));
        }
    }
}
