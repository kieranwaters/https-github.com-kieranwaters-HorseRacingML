using System.Collections.Generic;

namespace HorseRacingML.Models
{
    public class MLParameterBatch
    {
        public List<MLParameter> Parameters { get; set; } = new List<MLParameter>();

        /// <summary>
        /// Optional single-split mode for the Custom page. When set, each
        /// parameter is trained on every race before this date and validated
        /// once on races from this date onward, instead of running every
        /// walk-forward fold.
        /// </summary>
        public DateTime? ValidationStartDate { get; set; }
    }
}