using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Ext;
using RLMesh.Internal;
using Xunit;

namespace RLMesh.Tests;

public sealed class BarrierTests
{
    [Theory]
    [InlineData("success")]
    [InlineData("blocked")]
    [InlineData("constructor")]
    [InlineData("adoption")]
    [InlineData("refusal")]
    [InlineData("bridge")]
    [InlineData("diagnostic")]
    [InlineData("abandoned-close")]
    [InlineData("collector")]
    [InlineData("restart")]
    [InlineData("teardown-retry")]
    public void CollectibleContextCompletesItsPreflightThenUnloads(string scenario)
    {
        WeakReference context = RunIsolated(scenario);
        var deadline = Stopwatch.StartNew();
        while (context.IsAlive && deadline.Elapsed < TimeSpan.FromSeconds(30))
        {
            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); Thread.Sleep(20);
        }
        Assert.False(context.IsAlive, "the completed context must have no worker, event subscription or native callback root");
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference RunIsolated(string scenario)
    {
        var context = new FixtureContext();
        var assembly = context.LoadFromAssemblyPath(typeof(BarrierTests).Assembly.Location);
        try { assembly.GetType(typeof(BarrierFixtures).FullName!)!.GetMethod(nameof(BarrierFixtures.Run))!.Invoke(null, new object[] { scenario }); }
        catch (TargetInvocationException e) { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException!).Throw(); }
        var weak = new WeakReference(context);
        context.Unload();
        return weak;
    }
    private sealed class FixtureContext() : AssemblyLoadContext(isCollectible: true)
    {
        protected override Assembly? Load(AssemblyName name)
        {
            // xUnit's comparer caches hold Type instances. Keep those caches in
            // this context too, rather than rooting fixture types in the host.
            if (name.Name == "RLMesh" || name.Name!.StartsWith("xunit", StringComparison.Ordinal))
            {
                string path = Path.Combine(Path.GetDirectoryName(typeof(Env).Assembly.Location)!, name.Name + ".dll");
                if (File.Exists(path)) return LoadFromAssemblyPath(path);
            }
            return null;
        }
    }
}

// Invoked in a fresh collectible context: shutdown is terminal, including when
// a fixture fails. No test-only feature is added to the public library surface.
public static class BarrierFixtures
{
    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(20);
    public static void Run(string scenario)
    {
        Assert.Equal(LifetimeState.Open, Lifetime.State);
        Assert.Throws<ValueError>(() => Lifetime.Shutdown(TimeSpan.FromMilliseconds(-1)));
        Assert.Throws<ValueError>(() => Lifetime.Shutdown(Timeout.InfiniteTimeSpan));
        switch (scenario)
        {
            case "success": Success(); break;
            case "blocked": Blocked(); break;
            case "constructor": Constructor(); break;
            case "adoption": Adoption(); break;
            case "refusal": Refusal(); break;
            case "bridge": Bridge(); break;
            case "diagnostic": Diagnostic(); break;
            case "abandoned-close": AbandonedClose(); break;
            case "collector": Collector(); break;
            case "restart": Restart(); break;
            case "teardown-retry": TeardownRetry(); break;
            default: throw new ArgumentException(scenario);
        }
        Assert.True(Lifetime.Shutdown(Deadline), string.Join("; ", Lifetime.Pending()));
        Assert.Equal(LifetimeState.Completed, Lifetime.State);
        Assert.True(Lifetime.Shutdown(TimeSpan.Zero));
        CallbackContext.Enter();
        try { Assert.True(Lifetime.Shutdown(TimeSpan.Zero)); }
        finally { CallbackContext.Exit(); }
        bool wasLibraryThread = CallbackContext.LibraryThread;
        try { CallbackContext.LibraryThread = true; Assert.True(Lifetime.Shutdown(TimeSpan.Zero)); }
        finally { CallbackContext.LibraryThread = wasLibraryThread; }
        Assert.Throws<ObjectDisposedException>(() => Value.None);
        Assert.Throws<ObjectDisposedException>(() => new EnvServer(new FoundationTests.Counter()));
        Assert.Empty(ManagedRoots.Live);
        Assert.True(ManagedRoots.ReturnsJoined());
        Assert.Empty(NativeHandle.StrongTable);
        Assert.Empty(CleanupWorkers.Records);
        Assert.Equal(0, CallbackContext.ActiveFrames);
        Assert.Equal(0, Admission.Count);
        Assert.Equal(0, LibraryWorkers.Workers);
        Assert.Equal(0, CleanupWorkers.Workers);
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AbandonWrapper() => _ = Env.For(new FoundationTests.Counter());
    private static void Success()
    {
        AbandonWrapper();
        var authored = new FoundationTests.Counter { Error = new RecoverableEnvironmentException("once") };
        var server = new EnvServer(authored); server.Start();
        using var client = new NativeClient(server.Address);
        Assert.Throws<RecoverableEnvironmentException>(() => client.Run());
        authored.Error = null; client.Run();
        Assert.True(Lifetime.Shutdown(Deadline), string.Join("; ", Lifetime.Pending()));
        int calls = authored.Steps + authored.Closes;
        long frames = CallbackContext.EnteredFrames;
        Assert.True(client.RawResetFails()); // Raw C client, after managed admission has closed.
        Thread.Sleep(50);
        Assert.Equal(calls, authored.Steps + authored.Closes);
        Assert.Equal(frames, CallbackContext.EnteredFrames);
        Assert.True(Lifetime.WaitFreed(server, TimeSpan.Zero));
        server.Dispose(); // Exempt even after the terminal gate closes.
    }
    private static void Blocked()
    {
        var env = new CancellationTests.BlockingStep(); var server = new EnvServer(env); server.Start();
        using var client = new NativeClient(server.Address);
        var run = Task.Run(() => { try { client.Run(); } catch (Exception) { } });
        Assert.True(env.Entered.Wait(Deadline));
        var other = new EnvServer(new FoundationTests.Counter()); other.Start();
        var clock = Stopwatch.StartNew();
        Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(100)));
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(2), "caller exceeded the absolute deadline plus scheduling margin");
        Assert.Equal(LifetimeState.Incomplete, Lifetime.State);
        Assert.Contains(Lifetime.Pending(), s => s.Contains("EnvServer", StringComparison.Ordinal));
        Assert.Contains(Lifetime.Pending(), s => s.Contains("trampoline", StringComparison.Ordinal));
        Assert.True(Lifetime.WaitFreed(other, Deadline)); // Independent teardown progresses.
        Assert.False(Lifetime.WaitFreed(server, TimeSpan.Zero));
        env.Release.Set(); run.GetAwaiter().GetResult();
        Assert.True(Lifetime.Shutdown(Deadline), string.Join("; ", Lifetime.Pending()));
        Assert.True(Lifetime.WaitFreed(server, TimeSpan.Zero));
    }
    private static void Constructor()
    {
        var authored = new FoundationTests.Counter();
        using var entered = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        NativeHandle.BeforePublication = record => { if (record.Kind == HandleKind.Env) { entered.Set(); release.Wait(); } };
        var constructor = Task.Run(() => new EnvServer(authored));
        try
        {
            Assert.True(entered.Wait(Deadline)); Assert.Empty(NativeHandle.StrongTable);
            Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(50)));
            Assert.Contains(Lifetime.Pending(), s => s.Contains("EnvServer constructor", StringComparison.Ordinal));
            Assert.True(Admission.Count > 0);
        }
        finally { NativeHandle.BeforePublication = null; release.Set(); }
        EnvServer server = constructor.GetAwaiter().GetResult();
        Assert.True(Lifetime.Shutdown(Deadline), string.Join("; ", Lifetime.Pending()));
        Assert.True(Lifetime.WaitFreed(server, TimeSpan.Zero));
    }
    private static void Adoption()
    {
        using var entered = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        var server = new EnvServer(new FoundationTests.Counter()); server.Start();
        LibraryWorkers.BeforeAdoption = () => { entered.Set(); release.Wait(); };
        Task worker = server.ShutdownAsync();
        try
        {
            Assert.True(entered.Wait(Deadline)); Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(50)));
            Assert.Contains(Lifetime.Pending(), s => s.Contains("ShutdownAsync", StringComparison.Ordinal));
            Assert.False(Lifetime.WaitFreed(server, TimeSpan.Zero));
        }
        finally { LibraryWorkers.BeforeAdoption = null; release.Set(); }
        worker.GetAwaiter().GetResult();
    }
    private sealed class RefusingEnv : FoundationTests.Counter
    {
        public override StepResult Step(Value? action)
        {
            Assert.False(Lifetime.Shutdown(TimeSpan.FromSeconds(10)));
            Assert.Equal(LifetimeState.Open, Lifetime.State);
            Assert.Contains(Lifetime.Pending(), s => s.Contains("callback", StringComparison.Ordinal));
            return base.Step(action);
        }
    }
    private static void Refusal()
    {
        var server = new EnvServer(new RefusingEnv()); server.Start();
        using var client = new NativeClient(server.Address); client.Run();
        using var done = new ManualResetEventSlim();
        Exception? failure = null;
        LibraryWorkers.Enqueue(() =>
        {
            try { Assert.False(Lifetime.Shutdown(Deadline)); Assert.Equal(LifetimeState.Open, Lifetime.State); }
            catch (Exception e) { failure = e; }
            finally { done.Set(); }
        });
        Assert.True(done.Wait(Deadline)); Assert.Null(failure);
        bool notified = false;
        using var registration = Lifetime.ShuttingDown.Register(() =>
        {
            notified = true;
            Assert.False(Lifetime.Shutdown(Deadline));
            Assert.Contains(Lifetime.Pending(), s => s.Contains("notification", StringComparison.Ordinal));
        });
        // Ordinary .NET pool threads are callers, not the library's own pool.
        Assert.True(Task.Run(() => Lifetime.Shutdown(Deadline)).GetAwaiter().GetResult());
        Assert.True(notified);
    }
    private static void Bridge()
    {
        var env = new CancellationTests.BlockingStep(); var server = new EnvServer(env); server.Start();
        using var client = new NativeClient(server.Address);
        Exception? error = null;
        var thread = new Thread(() => { try { client.Run(); } catch (Exception e) { error = e; } }); thread.Start();
        Assert.True(env.Entered.Wait(Deadline));
        Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(100)));
        Assert.True(thread.Join(Deadline)); Assert.IsType<CancelledError>(error);
        Assert.False(((CancelledError)error!).CancellationToken.IsCancellationRequested);
        env.Release.Set();
    }
    private static void AbandonedClose()
    {
        var env = new CancellationTests.BlockingClose();
        var server = new EnvServer(env); // Ready close has the native 5s timeout.
        var stop = server.ShutdownAsync();
        Task? dispose = null;
        try
        {
            Assert.True(env.Entered.Wait(Deadline));
            var error = Assert.Throws<RLMeshException>(() => stop.GetAwaiter().GetResult());
            Assert.Contains("close timed out", error.Message);
            dispose = Task.Run(server.Dispose);
            Assert.True(SpinWait.SpinUntil(() => Volatile.Read(ref server.Record.WaitingIn) == "rlmesh_env_server_free", Deadline));
            Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(100)));
            Assert.Contains(Lifetime.Pending(), s => s.Contains("rlmesh_env_server_free", StringComparison.Ordinal));
            Assert.False(server.Record.JoinedNativeReturn.Returned);
            Assert.False(Lifetime.WaitFreed(server, TimeSpan.Zero));
        }
        finally { env.Release.Set(); }
        dispose!.GetAwaiter().GetResult();
        Assert.True(Lifetime.Shutdown(Deadline), string.Join("; ", Lifetime.Pending()));
        Assert.True(server.Record.JoinedNativeReturn.Returned);
    }
    private static void Collector()
    {
        using var entered = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        using var publishing = new ManualResetEventSlim(); using var publish = new ManualResetEventSlim();
        using var freeing = new ManualResetEventSlim(); using var free = new ManualResetEventSlim();
        var first = Env.For(new FoundationTests.Counter());
        NativeHandle.BeforeFree = record => { if (ReferenceEquals(record, first.Record)) { freeing.Set(); free.Wait(); } };
        NativeHandle.BeforePublication = record => { if (!ReferenceEquals(record, first.Record)) { publishing.Set(); publish.Wait(); } };
        LifetimeEngine.CollectFinalizers = () => { entered.Set(); release.Wait(); };
        var constructing = Task.Run(() => Env.For(new FoundationTests.Counter()));
        try
        {
            Assert.True(publishing.Wait(Deadline));
            Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(50)));
            Assert.True(freeing.Wait(Deadline)); Assert.True(entered.Wait(Deadline));
            Assert.Contains(Lifetime.Pending(), s => s.Contains("finalizer collection", StringComparison.Ordinal));
            free.Set(); publish.Set();
            var late = constructing.GetAwaiter().GetResult();
            Assert.True(Lifetime.WaitFreed(late, Deadline)); // Coordinator progresses while finalizers are stuck.
            Assert.True(Lifetime.WaitFreed(first, Deadline));
        }
        finally
        {
            NativeHandle.BeforeFree = null; NativeHandle.BeforePublication = null;
            free.Set(); publish.Set(); release.Set(); LifetimeEngine.CollectFinalizers = null;
        }
    }
    private static void Restart()
    {
        LifetimeEngine.BeforeDrain = () => throw new InvalidOperationException("injected drain failure");
        try
        {
            Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(100)));
            Assert.Contains(Lifetime.Pending(), s => s.Contains("coordinator stopped", StringComparison.Ordinal));
        }
        finally { LifetimeEngine.BeforeDrain = null; }
        Assert.True(Lifetime.Shutdown(Deadline), string.Join("; ", Lifetime.Pending()));
    }
    private static void TeardownRetry()
    {
        var env = Env.For(new FoundationTests.Counter());
        int attempts = 0;
        using var attempted = new ManualResetEventSlim();
        LifetimeEngine.BeforeTeardown = record =>
        {
            if (ReferenceEquals(record, env.Record))
            {
                Interlocked.Increment(ref attempts);
                attempted.Set();
                throw new InvalidOperationException("injected record teardown failure");
            }
        };
        try
        {
            Assert.False(Lifetime.Shutdown(TimeSpan.FromMilliseconds(100)));
            Assert.True(attempted.Wait(Deadline));
            Assert.True(SpinWait.SpinUntil(() => !NativeHandle.CleanupFailures.IsEmpty, Deadline));
            Assert.Equal(1, Volatile.Read(ref attempts));
            Assert.False(env.Record.IsFreed);
            Assert.Contains(Lifetime.Pending(), s => s.Contains("injected record teardown failure", StringComparison.Ordinal));
        }
        finally { LifetimeEngine.BeforeTeardown = null; }
        Assert.True(Lifetime.Shutdown(Deadline), string.Join("; ", Lifetime.Pending()));
        Assert.True(env.Record.IsFreed);
    }
    private static void Diagnostic()
    {
        AbandonWrapper();
        var original = Console.Error;
        using var text = new StringWriter();
        try { Console.SetError(text); LifetimeEngine.Diagnose(); }
        finally { Console.SetError(original); }
        Assert.Contains("unload is unsafe and may crash", text.ToString());
        Assert.Contains("Env", text.ToString());
    }
}
