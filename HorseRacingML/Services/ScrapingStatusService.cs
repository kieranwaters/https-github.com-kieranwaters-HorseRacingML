using System;
using System.Threading;

namespace HorseRacingML.Services
{
    public class ScrapingStatusService
    {
        private string _message = "Idle";
        private readonly object _lock = new object();

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
            }
        }
    }
}