using System;
using System.Collections.Generic;
using System.Threading;

namespace RLMesh.Internal;

// No .NET pool thread blocks in C. An independent queued operation always gets
// a worker even when every other native operation is stuck in authored code.
internal sealed class TrackedPool
{
    private readonly object sync = new();
    private readonly Queue<Action> queue = new();
    private readonly List<Thread> threads = new();
    private readonly string name;
    private readonly Func<object, bool> wait;
    private int idle;
    private bool stopping;
    internal TrackedPool(string name, Func<object, bool>? wait = null)
    {
        this.name = name;
        this.wait = wait ?? (s => Monitor.Wait(s, TimeSpan.FromSeconds(30)));
    }
    internal int Workers { get { lock (sync) return threads.FindAll(t => t.IsAlive).Count; } }
    private void Start()
    {
        var thread = new Thread(Work) { IsBackground = true, Name = name };
        threads.Add(thread);
        idle++;
        try { thread.Start(); }
        catch { idle--; threads.Remove(thread); throw; }
    }
    internal void Enqueue(Action action)
    {
        lock (sync)
        {
            if (stopping) throw new InvalidOperationException("RLMesh worker pool has stopped");
            // Start before publishing the action. If thread creation fails the
            // caller still owns its references, and no future worker may run it.
            if (idle == 0) Start();
            queue.Enqueue(action);
            Monitor.Pulse(sync);
        }
    }
    private void Work()
    {
        CallbackContext.LibraryThread = true;
        while (true)
        {
            Action action;
            lock (sync)
            {
                while (queue.Count == 0)
                {
                    if (stopping) { idle--; return; }
                    if (!wait(sync) && queue.Count == 0) { idle--; return; }
                }
                idle--;
                action = queue.Dequeue();
                if (queue.Count > 0 && idle == 0 && !stopping)
                {
                    try { Start(); }
                    catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
                }
            }
            try { action(); }
            catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
            finally { action = null!; } // An idle worker must not root a completed operation's wrapper.
            lock (sync) idle++;
        }
    }
    // Called only by the tracked barrier coordinator, never by its caller.
    internal void StopAndJoin()
    {
        Thread[] snapshot;
        lock (sync) { stopping = true; Monitor.PulseAll(sync); snapshot = threads.ToArray(); }
        foreach (var thread in snapshot) thread.Join();
    }
}

internal static class LibraryWorkers
{
    private static readonly TrackedPool Pool = new("RLMesh operation");
    internal static Action? BeforeAdoption;
    internal static void Enqueue(Action operation) => Pool.Enqueue(operation);
    internal static void StopAndJoin() => Pool.StopAndJoin();
    internal static int Workers => Pool.Workers;
}
