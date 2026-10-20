using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace RLMesh.Unity
{
    // Main-thread state machine. The barrier is always called on a worker;
    // Poll is nonblocking and the caller keeps pumping its normal main loop.
    public sealed class ShutdownPreflight
    {
        private readonly Func<TimeSpan, bool> shutdown;
        private readonly Func<IReadOnlyList<string>> pending;
        private readonly Action acquire;
        private readonly Action release;
        private readonly Action<string> report;
        private Task<bool> attempt;
        public bool IsLocked { get; private set; }
        public bool IsRunning => attempt != null;
        public bool Completed { get; private set; }
        private bool used;
        public bool NeedsUnloadPreflight => used && !Completed;

        public ShutdownPreflight(Func<TimeSpan, bool> shutdown, Func<IReadOnlyList<string>> pending,
            Action acquire, Action release, Action<string> report)
        {
            this.shutdown = shutdown; this.pending = pending;
            this.acquire = acquire; this.release = release; this.report = report;
        }

        public void Hold()
        {
            if (Completed) throw new InvalidOperationException("RLMesh shutdown is terminal; reload the domain before serving again");
            if (!IsLocked) { acquire(); IsLocked = true; used = true; }
        }

        public void Request(TimeSpan timeout)
        {
            if (timeout < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(timeout));
            if (Completed || attempt != null) return;
            Hold();
            attempt = Task.Run(() => shutdown(timeout));
        }

        public void Poll()
        {
            if (attempt == null || !attempt.IsCompleted) return;
            var finished = attempt;
            attempt = null;
            bool safe;
            try { safe = finished.GetAwaiter().GetResult(); }
            catch (Exception error) { Report("Shutdown threw: " + error); return; }
            if (!safe) { Report("Shutdown returned false"); return; }
            Completed = true;
            if (IsLocked) { release(); IsLocked = false; }
        }

        public void DiagnoseUnload()
        {
            if (NeedsUnloadPreflight) Report("Unload began before preflight completed; unsafe unload may crash");
        }

        // Only for an explicitly confirmed, unsafe Editor escape hatch.
        public void ForceUnlock()
        {
            if (IsLocked) { release(); IsLocked = false; }
        }

        private void Report(string reason)
        {
            report("RLMesh: " + reason + (IsLocked ? ". Reload remains locked." : ". Reload lock is not held.") + " Release blocked environment code, then use RLMesh > Shutdown and Reload to retry. Pending: " + string.Join("; ", pending()));
        }
    }
}
