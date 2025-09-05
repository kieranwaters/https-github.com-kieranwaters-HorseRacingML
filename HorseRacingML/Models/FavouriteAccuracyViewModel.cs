using System;

namespace HorseRacingML.Models
{
    public class FavouriteAccuracyViewModel
    {
        public double IncludingJoint { get; set; }
        public double ExcludingJoint { get; set; }
        public double LogLoss { get; set; }
    }
}