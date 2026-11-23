using System.Threading.Tasks;

namespace HorseRacingML.Jobs
{
    public interface IBackgroundJobHandler
    {
        BackgroundJobKind Kind { get; }
        Task ExecuteAsync(JobExecutionContext context);
    }
}
