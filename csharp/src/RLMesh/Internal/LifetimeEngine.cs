using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using RLMesh.Ext;
using RLMesh.Interop;
#if NET10_0_OR_GREATER
using System.Runtime.Loader;
#endif

namespace RLMesh.Internal;

internal static class LifetimeEngine
{
    private static readonly object Sync = new();
    private static readonly CancellationTokenSource NotificationSource = new();
    private static Thread? coordinator;
    private static int state;
    private static string? refusal;
    private static bool quiescent;
    private static readonly ConcurrentDictionary<Thread, string> Teardowns = new();
#if NET10_0_OR_GREATER
    private static readonly AssemblyLoadContext? Context;
#endif
    static LifetimeEngine()
    {
        AppDomain.CurrentDomain.ProcessExit += ProcessExit;
        AppDomain.CurrentDomain.DomainUnload += DomainUnload;
#if NET10_0_OR_GREATER
        Context = AssemblyLoadContext.GetLoadContext(typeof(LifetimeEngine).Assembly);
        if (Context?.IsCollectible == true) Context.Unloading += Unloading;
#endif
    }
    internal static void EnsureInitialized() { }
    internal static CancellationToken Notification => NotificationSource.Token;
    internal static LifetimeState State => (LifetimeState)Volatile.Read(ref state);
    internal static bool Shutdown(TimeSpan timeout)
    {
        if (timeout < TimeSpan.Zero) throw new ValueError("timeout must be nonnegative and finite");
        var elapsed = Stopwatch.StartNew();
        if (State == LifetimeState.Completed) return true;
        if (CallbackContext.Depth > 0 || CallbackContext.LibraryThread)
        {
            Volatile.Write(ref refusal, CallbackContext.Depth > 0 ? "Shutdown refused inside a callback" : "Shutdown refused on a library thread (including notification)");
            return false;
        }
        Thread thread;
        lock (Sync)
        {
            if (State == LifetimeState.Completed) return true;
            refusal = null;
            // Failed native teardown attempts remain diagnostic until a caller
            // retries Shutdown; do not spin up failing threads in a hot loop.
            foreach (var record in FailedTeardowns.Keys)
                if (FailedTeardowns.TryRemove(record, out _)) Seen.TryRemove(record, out _);
            if (coordinator == null || !coordinator.IsAlive && !quiescent)
            {
                Admission.Close();
                Volatile.Write(ref state, (int)LifetimeState.ShuttingDown);
                coordinator = new Thread(Drain) { IsBackground = true, Name = "RLMesh barrier" };
                coordinator.Start();
            }
            thread = coordinator;
        }
        // Joining the coordinator also joins every thread and native return it
        // covers. No user code, native free, notification or GC runs here.
        while (thread.IsAlive)
        {
            TimeSpan remaining = timeout - elapsed.Elapsed;
            if (remaining <= TimeSpan.Zero) break;
            thread.Join(remaining > TimeSpan.FromMilliseconds(100) ? TimeSpan.FromMilliseconds(100) : remaining);
        }
        lock (Sync)
        {
            if (ReferenceEquals(thread, coordinator) && !thread.IsAlive && quiescent)
            {
                refusal = null;
                Volatile.Write(ref state, (int)LifetimeState.Completed);
                return true;
            }
            if (State != LifetimeState.Completed) Volatile.Write(ref state, (int)LifetimeState.Incomplete);
            return State == LifetimeState.Completed;
        }
    }
    private static Thread Start(List<Thread> threads, string name, Action action)
    {
        var thread = new Thread(() =>
        {
            CallbackContext.LibraryThread = true;
            try { action(); }
            catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
        }) { IsBackground = true, Name = name };
        threads.Add(thread);
        Teardowns.TryAdd(thread, name);
        try { thread.Start(); }
        catch { threads.Remove(thread); Teardowns.TryRemove(thread, out _); throw; }
        return thread;
    }
    internal static Action? BeforeDrain;
    internal static Action? CollectFinalizers;
    private static Thread? collector;
    private static bool notificationsStarted;
    internal static Action<NativeHandle>? BeforeTeardown;
    private static readonly ConcurrentDictionary<NativeHandle, byte> Seen = new();
    private static readonly ConcurrentDictionary<NativeHandle, byte> FailedTeardowns = new();
    private static void Drain()
    {
        CallbackContext.LibraryThread = true;
        try
        {
            BeforeDrain?.Invoke();
            var threads = new List<Thread>(Teardowns.Keys);
            if (!notificationsStarted)
            {
                Start(threads, "RLMesh notification", () => NotificationSource.Cancel());
                Start(threads, "RLMesh cancellation", CancellationBridge.CancelAll);
                notificationsStarted = true;
            }
            var collections = Stopwatch.StartNew();
            while (true)
            {
                foreach (var record in NativeHandle.StrongTable.Keys)
                {
                    if (!Seen.TryAdd(record, 0)) continue;
                    try
                    {
                        Start(threads, "RLMesh teardown " + record.Kind, () =>
                        {
                            try { BeforeTeardown?.Invoke(record); Teardown(record); }
                            catch { FailedTeardowns.TryAdd(record, 0); throw; }
                        });
                    }
                    catch { Seen.TryRemove(record, out _); throw; }
                }
                // A blocked unrelated finalizer must not stop teardown of
                // other records. Keep collection on a separately tracked thread.
                if (!NativeHandle.StrongTable.IsEmpty && (collector == null || !collector.IsAlive) &&
                    collections.Elapsed >= TimeSpan.FromSeconds(1))
                {
                    collector = Start(threads, "RLMesh finalizer collection", () =>
                    {
                        if (CollectFinalizers != null) CollectFinalizers();
                        else { GC.Collect(); GC.WaitForPendingFinalizers(); }
                    });
                    collections.Restart();
                }
                bool joined = true;
                foreach (var thread in threads)
                    if (!thread.Join(0))
                    {
                        // Collection is only needed to discover unreachable
                        // owners while strong records remain. An unrelated
                        // finalizer is not a native callback-bearing root.
                        if (thread != collector || !NativeHandle.StrongTable.IsEmpty) joined = false;
                    }
                    else Teardowns.TryRemove(thread, out _);
                if (Admission.Count == 0 && NativeHandle.StrongTable.IsEmpty && ManagedRoots.Live.IsEmpty &&
                    Volatile.Read(ref CallbackContext.ActiveFrames) == 0 && CleanupWorkers.Records.IsEmpty &&
                    CancellationBridge.Count == 0 && ManagedRoots.ReturnsJoined() && joined)
                {
                    LibraryWorkers.StopAndJoin();
                    CleanupWorkers.StopAndJoin();
                    // Stops are after admission zero and roots released. No new
                    // callback-bearing record or worker request can now appear.
                    if (Admission.Count == 0 && NativeHandle.StrongTable.IsEmpty && ManagedRoots.Live.IsEmpty &&
                        CallbackContext.ActiveFrames == 0 && CleanupWorkers.Records.IsEmpty && ManagedRoots.ReturnsJoined())
                    {
                        Callbacks.ReleaseResources();
                        RemoveDiagnostics();
                        lock (Sync) quiescent = true;
                        return;
                    }
                }
                Thread.Sleep(10);
            }
        }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
    }
    private static unsafe void Teardown(NativeHandle record)
    {
        if (record.Kind != HandleKind.EnvServer) { record.CloseOwner(0, false, anyGeneration: true); return; }
        HandleLease lease;
        try { lease = record.Acquire(teardown: true); }
        catch (ObjectDisposedException) { return; } // A free already owns it.
        using (lease)
        {
            record.CloseOwner(0, false, anyGeneration: true);
            Volatile.Write(ref record.WaitingIn, "rlmesh_env_server_shutdown");
            Status.Check(Native.rlmesh_env_server_shutdown(lease.Server));
        }
    }
    internal static IReadOnlyList<string> Pending()
    {
        var pending = new List<string>();
        string? refused = Volatile.Read(ref refusal);
        if (refused != null) pending.Add(refused);
        foreach (var operation in Admission.Operations.Values) pending.Add("operation: " + operation);
        if (Admission.Count != 0) pending.Add("admitted operations: " + Admission.Count);
        foreach (var record in NativeHandle.StrongTable.Keys)
            pending.Add(record.Kind + ": " + NativeHandle.State(record.Word) + " (" + (Volatile.Read(ref record.WaitingIn) ?? "outstanding reference or pending owner close") + ")");
        foreach (var root in ManagedRoots.Live.Values) pending.Add("managed root: " + root.Kind);
        if (!ManagedRoots.ReturnsJoined()) pending.Add("free_payload/free_user_data native return is not joined: " + ManagedRoots.AwaitingReturn.Count);
        foreach (string thread in Teardowns.Values) pending.Add("teardown thread: " + thread);
        if (CallbackContext.ActiveFrames != 0) pending.Add("active trampoline frames: " + CallbackContext.ActiveFrames);
        if (!CleanupWorkers.Records.IsEmpty) pending.Add("queued/claimed cleanup records: " + CleanupWorkers.Records.Count);
        foreach (var failure in NativeHandle.CleanupFailures)
        {
            string message;
            try { message = failure.Message; } catch { message = "message unavailable"; }
            pending.Add("cleanup failure: " + failure.GetType().Name + ": " + message);
        }
        if (LibraryWorkers.Workers != 0) pending.Add("operation workers: " + LibraryWorkers.Workers);
        if (CleanupWorkers.Workers != 0) pending.Add("cleanup workers: " + CleanupWorkers.Workers);
        if (coordinator != null && !coordinator.IsAlive && !quiescent)
            pending.Add("barrier coordinator stopped; retry Shutdown to resume teardown");
        if (pending.Count == 0 && coordinator?.IsAlive == true)
            pending.Add("barrier coordinator: worker joins");
        return pending.AsReadOnly();
    }
    private static void RemoveDiagnostics()
    {
        AppDomain.CurrentDomain.ProcessExit -= ProcessExit;
        AppDomain.CurrentDomain.DomainUnload -= DomainUnload;
#if NET10_0_OR_GREATER
        if (Context?.IsCollectible == true) Context.Unloading -= Unloading;
#endif
    }
    internal static void Diagnose()
    {
        if (State == LifetimeState.Completed) return;
        Console.Error.WriteLine("RLMesh unload is unsafe and may crash; preflight Lifetime.Shutdown must complete before unload. " + string.Join("; ", Pending()));
        Shutdown(TimeSpan.FromSeconds(1));
    }
    private static void ProcessExit(object? sender, EventArgs args)
    {
        try
        {
            if (!Shutdown(TimeSpan.FromSeconds(5)))
                Console.Error.WriteLine("RLMesh process exit did not complete shutdown: " + string.Join("; ", Pending()));
        }
        catch { /* Exit diagnostics cannot repair or prevent process exit. */ }
    }
    private static void DomainUnload(object? sender, EventArgs args) => Diagnose();
#if NET10_0_OR_GREATER
    private static void Unloading(AssemblyLoadContext context) => Diagnose();
#endif
}
