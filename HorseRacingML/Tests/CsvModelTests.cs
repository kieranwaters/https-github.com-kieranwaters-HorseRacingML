using CsvHelper;
using CsvHelper.Configuration;
using HorseRacingML.Models;
using System.Globalization;
using System.IO;
using System.Linq;
using Xunit;

namespace HorseRacingML.Tests
{
    public class CsvModelTests
    {
        [Fact]
        public void RunnerCsvModel_MapsHeadersCorrectly()
        {
            // Arrange
            var csvContent = @"rid,horseName,age,saddle,decimalPrice,isFav,trainerName,jockeyName,position,positionL,dist,weightSt,weightLb,overWeight,outHandicap,headGear,RPR,TR,OR,father,mother,gfather,runners,margin,weight,res_win,res_place
1,Horse Name,5,1,2.5,1,Trainer Name,Jockey Name,1,0,0,10,5,0,0,H,100,100,100,Father,Mother,Grandfather,10,0,145,1,1";

            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };

            using var reader = new StringReader(csvContent);
            using var csv = new CsvReader(reader, config);

            // Act
            var records = csv.GetRecords<RunnerCsvModel>().ToList();

            // Assert
            Assert.Single(records);
            var record = records[0];
            Assert.Equal(1, record.Rid);
            Assert.Equal("Horse Name", record.HorseName);
            Assert.Equal(1, record.ResPlace);
            Assert.Equal(10, record.WeightSt);
            Assert.Equal(5, record.WeightLb);
        }
    }
}
