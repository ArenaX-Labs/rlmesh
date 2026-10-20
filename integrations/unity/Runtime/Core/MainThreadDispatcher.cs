using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Internal;

namespace RLMesh.Unity
{
    // No Unity references: also compiled and exercised against netstandard2.1.
    public sealed class MainThreadDispatcher : IDisposable
    {
        private readonly int mainThread = Thread.CurrentThread.ManagedThreadId;
        private readonly object gate = new object();
        private readonly Queue<Job> queue = new Queue<Job>();
        private readonly CancellationToken shuttingDown;
        private readonly CancellationTokenRegistration registration;
        private bool closed;

        public MainThreadDispatcher(CancellationToken shuttingDown)
        {
            UnityEnvHost.AssertMainThreadIfInitialized();
            this.shuttingDown = shuttingDown;
            registration = shuttingDown.Register(Close);
        }

        public int PendingCount { get { lock (gate) return queue.Count; } }
        public bool IsMainThread => Thread.CurrentThread.ManagedThreadId == mainThread;

        public T Call<T>(Func<T> callback)
        {
            if (callback == null) throw new ArgumentNullException(nameof(callback));
            Job<T> job;
            lock (gate)
            {
                // Admission and enqueue share a lock with Close. The token check
                // also refuses calls before its registration has obtained gate.
                if (closed || shuttingDown.IsCancellationRequested) throw Ended();
                job = new Job<T>(callback);
                if (!IsMainThread) queue.Enqueue(job);
            }
            // Never hold gate across authored code: callbacks may re-enter Call.
            if (IsMainThread) job.Run();
            return job.Result.GetAwaiter().GetResult();
        }

        public void Pump()
        {
            if (!IsMainThread) throw new InvalidOperationException("Pump must run on the Unity main thread");
            // Limit this tick to requests already queued. A busy client must not
            // starve the Editor loop or its reload coordinator.
            int count = PendingCount;
            for (int i = 0; i < count; i++)
            {
                Job job;
                lock (gate)
                {
                    if (queue.Count == 0) return;
                    job = queue.Dequeue();
                    if (closed || shuttingDown.IsCancellationRequested) { job.Fail(Ended()); continue; }
                }
                job.Run();
            }
        }

        public void Close()
        {
            lock (gate)
            {
                closed = true;
                while (queue.Count != 0) queue.Dequeue().Fail(Ended());
            }
        }

        public void Dispose() { Close(); registration.Dispose(); }
        private static RuntimeError Ended() => new RuntimeError("the Unity environment is shutting down");

        private abstract class Job
        {
            public abstract void Run();
            public abstract void Fail(Exception error);
        }

        private sealed class Job<T> : Job
        {
            private readonly Func<T> callback;
            private readonly CallbackContext.Snapshot context;
            private readonly TaskCompletionSource<T> completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            public Job(Func<T> callback) { this.callback = callback; context = CallbackContext.Capture(); }
            public Task<T> Result => completion.Task;
            public override void Fail(Exception error) { completion.TrySetException(error); }
            public override void Run()
            {
                // The native lane stays blocked in Call until this synchronous
                // scope exits. Never hand a captured context to fire-and-forget work.
                T result = default(T);
                Exception failure = null;
                using (context.Enter())
                {
                    try { result = callback(); }
                    catch (Exception error) { failure = error; }
                }
                if (failure != null) Fail(failure);
                else completion.TrySetResult(result);
            }
        }
    }
}
