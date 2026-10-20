using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;
using RLMesh.Ext;

namespace RLMesh.Unity
{
    public static class UnityEnvHost
    {
        private static readonly List<MainThreadDispatcher> dispatchers = new List<MainThreadDispatcher>();
        private static readonly List<Task> stops = new List<Task>();
        private static readonly object gate = new object();
        private static int mainThread;
        private static readonly List<string> diagnostics = new List<string>();
        public static event Action Serving;

        public static void InitializeMainThread()
        {
            int current = Thread.CurrentThread.ManagedThreadId;
            Interlocked.CompareExchange(ref mainThread, current, 0);
            AssertMainThreadIfInitialized();
        }

        internal static void AssertMainThreadIfInitialized()
        {
            if (Volatile.Read(ref mainThread) != 0 && mainThread != Thread.CurrentThread.ManagedThreadId)
                throw new InvalidOperationException("RLMesh Unity host operations must run on the initialized main thread");
        }

        public static MainThreadDispatcher BeginServe()
        {
            AssertMainThreadIfInitialized();
            if (mainThread == 0) throw new InvalidOperationException("Initialize the Unity main thread before serving");
            Serving?.Invoke(); // Editor takes the reload lock BEFORE native construction.
            var dispatcher = new MainThreadDispatcher(Lifetime.ShuttingDown);
            dispatchers.Add(dispatcher);
            return dispatcher;
        }

        public static void Pump()
        {
            AssertMainThreadIfInitialized();
            foreach (var dispatcher in dispatchers.ToArray()) dispatcher.Pump();
        }

        public static IReadOnlyList<string> Pending()
        {
            var pending = new List<string>(Lifetime.Pending());
            lock (gate)
            {
                ObserveStops();
                pending.AddRange(diagnostics);
                foreach (var stop in stops)
                    if (!stop.IsCompleted) pending.Add("Unity component disposal worker still running");
            }
            return pending;
        }

        internal static Task Stop(MainThreadDispatcher dispatcher, EnvServer server)
        {
            AssertMainThreadIfInitialized();
            dispatcher?.Dispose();
            dispatchers.Remove(dispatcher);
            if (server == null) return Task.CompletedTask;
            if (Lifetime.State == LifetimeState.Completed) { server.Dispose(); return Task.CompletedTask; }
            // Dispose can join foreign callbacks: never do that in Unity's tick.
            lock (gate)
            {
                ObserveStops();
                var stop = Task.Run(() => server.Dispose());
                stops.Add(stop);
                return stop;
            }
        }

        public static bool Shutdown(TimeSpan timeout)
        {
            var clock = Stopwatch.StartNew();
            if (!Lifetime.Shutdown(timeout)) return false;
            while (true)
            {
                lock (gate)
                {
                    ObserveStops();
                    if (stops.Count == 0) return true;
                }
                if (clock.Elapsed >= timeout) return false;
                Thread.Sleep(1);
            }
        }

        private static void ObserveStops()
        {
            stops.RemoveAll(stop =>
            {
                if (!stop.IsCompleted) return false;
                if (stop.IsFaulted) diagnostics.Add("Unity component disposal failure: " + stop.Exception);
                if (stop.IsCanceled) diagnostics.Add("Unity component disposal was cancelled");
                return true;
            });
        }

        // Unit tests inject a failed disposal without replacing native cleanup.
        internal static void TrackStop(Task stop) { lock (gate) stops.Add(stop); }
    }
}
