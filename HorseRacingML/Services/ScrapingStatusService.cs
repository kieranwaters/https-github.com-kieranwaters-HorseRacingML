using HorseRacingML.Models;
using HorseRacingML.Scraping;
using System;
using System.Collections.Generic;
using System.Linq;

namespace HorseRacingML.Services
{
    public class ScrapingStatusService
    {
        private string _message = "Idle";
        private readonly object _lock = new object();
        private readonly List<RaceDayReport> _results = new List<RaceDayReport>();

        public int TotalRaces { get; private set; }
        public int ProcessedRaces { get; private set; }
        public bool IsComplete { get; private set; }
        public DateTime? LastUpdated { get; private set; }

        public string Message
        {
            get
            {
                lock (_lock)
                {
                    return _message;
                }
            }
        }

        public void Update(string message)
        {
            lock (_lock)
            {
                _message = message;
                LastUpdated = DateTime.UtcNow;
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                _message = "Starting...";
                _results.Clear();
                TotalRaces = 0;
                ProcessedRaces = 0;
                IsComplete = false;
                LastUpdated = DateTime.UtcNow;
            }
        }

        public void SetTotal(int total)
        {
            lock (_lock)
            {
                TotalRaces = total;
                LastUpdated = DateTime.UtcNow;
            }
        }

        public void AddResults(IEnumerable<RaceDayReport> newResults)
        {
            lock (_lock)
            {
                if (newResults != null)
                {
                    _results.AddRange(newResults);
                    ProcessedRaces = _results.Count;
                }
                LastUpdated = DateTime.UtcNow;
            }
        }

        public void MarkComplete()
        {
            lock (_lock)
            {
                IsComplete = true;
                _message = "Completed";
                LastUpdated = DateTime.UtcNow;
            }
        }

        public List<RaceDayReport> GetSnapshot()
        {
            lock (_lock)
            {
                return new List<RaceDayReport>(_results);
            }
        }
    }
}
