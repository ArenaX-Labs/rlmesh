using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace RLMesh.Unity
{
    public static class QuitPreflight
    {
        // Process exit is best effort, unlike domain unload. Return to the
        // ORIGINAL quit request, preserving its exit code and normal save prompts.
        public static bool TryShutdown(TimeSpan timeout, Func<TimeSpan, bool> shutdown,
            Func<IReadOnlyList<string>> pending, Action<string> report)
        {
            if (timeout < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(timeout));
            var attempt = Task.Run(() => shutdown(timeout));
            try
            {
                if (attempt.Wait(timeout) && attempt.GetAwaiter().GetResult()) return true;
            }
            catch (Exception error) { report("RLMesh quit shutdown failed: " + error); }
            _ = attempt.ContinueWith(task => { _ = task.Exception; }, TaskContinuationOptions.OnlyOnFaulted);
            report("RLMesh quit shutdown incomplete; process exit may proceed. Pending: " + string.Join("; ", pending()));
            return false;
        }
    }
}
