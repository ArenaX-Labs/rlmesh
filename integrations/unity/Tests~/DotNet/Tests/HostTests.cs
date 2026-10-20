using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Ext;
using RLMesh.Internal;
using RLMesh.Spaces;
using RLMesh.Tests;
using Xunit;

namespace RLMesh.Unity.Tests;

public sealed class HostTests
{
    [Theory]
    [InlineData("threads")]
    [InlineData("notification")]
    [InlineData("stop")]
    [InlineData("dispose")]
    [InlineData("refusal")]
    [InlineData("result")]
    [InlineData("faulted-stop")]
    [InlineData("wrong-thread")]
    public void RealServerHostLifecycle(string scenario)
    {
        var weak = Isolated(scenario);
        Assert.True(SpinWait.SpinUntil(() => { GC.Collect(); GC.WaitForPendingFinalizers(); return !weak.IsAlive; }, TimeSpan.FromSeconds(15)), "host context remained rooted");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference Isolated(string scenario)
    {
        var context = new FixtureContext();
        var assembly = context.LoadFromAssemblyPath(typeof(HostTests).Assembly.Location);
        try { assembly.GetType(typeof(HostFixtures).FullName).GetMethod(nameof(HostFixtures.Run)).Invoke(null, new object[] { scenario }); }
        catch (TargetInvocationException error) { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error.InnerException).Throw(); }
        var weak = new WeakReference(context);
        context.Unload();
        return weak;
    }

    private sealed class FixtureContext() : AssemblyLoadContext(isCollectible: true)
    {
        protected override Assembly Load(AssemblyName name)
        {
            if (name.Name == "RLMesh" || name.Name == "RLMesh.Unity" || name.Name.StartsWith("xunit", StringComparison.Ordinal))
            {
                string path = Path.Combine(Path.GetDirectoryName(typeof(HostTests).Assembly.Location), name.Name + ".dll");
                if (File.Exists(path)) return LoadFromAssemblyPath(path);
            }
            return null;
        }
    }
}

public static class HostFixtures
{
    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(15);
    public static void Run(string scenario)
    {
        UnityEnvHost.InitializeMainThread();
        int main = Thread.CurrentThread.ManagedThreadId;
        int locks = 0;
        UnityEnvHost.Serving += () => { Assert.Equal(main, Thread.CurrentThread.ManagedThreadId); locks++; };
        if (scenario == "wrong-thread")
        {
            Exception error = null;
            var worker = new Thread(() => error = Record.Exception(() => new MainThreadDispatcher(CancellationToken.None)));
            worker.Start();
            Assert.True(worker.Join(Deadline));
            Assert.IsType<InvalidOperationException>(error);
        }
        else if (scenario == "faulted-stop")
        {
            UnityEnvHost.TrackStop(Task.FromException(new InvalidOperationException("disposal probe")));
            Assert.True(Task.Run(() => UnityEnvHost.Shutdown(Deadline)).GetAwaiter().GetResult());
            Assert.Contains(UnityEnvHost.Pending(), text => text.Contains("disposal probe"));
            Assert.True(UnityEnvHost.Shutdown(TimeSpan.Zero)); // fault is diagnostic, never strands a retry.
        }
        else
        {
            var dispatcher = UnityEnvHost.BeginServe();
            var callers = new ConcurrentDictionary<int, byte>();
            var servers = new List<EnvServer>();
            var clients = new List<NativeClient>();
            var drivers = new List<Thread>();
            var envs = new List<SceneEnv>();
            int count = scenario == "threads" ? 3 : 1;
            for (int i = 0; i < count; i++)
            {
                var env = new SceneEnv(main, () => locks > 0);
                var served = new MainThreadEnv(dispatcher, env);
                var server = new EnvServer(new LaneProbe(served, callers));
                servers.Add(server); envs.Add(env);
                env.StepBody = () =>
                {
                    Assert.True(CallbackContext.Depth > 0);
                    Assert.Same(server.Record, CallbackContext.HostedBy);
                    if (scenario == "dispose") server.Dispose();
                    if (scenario == "refusal")
                    {
                        var clock = Stopwatch.StartNew();
                        Assert.False(Lifetime.Shutdown(TimeSpan.FromSeconds(5)));
                        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1));
                        Assert.Equal(LifetimeState.Open, Lifetime.State);
                    }
                    if (scenario == "result")
                    {
                        var stop = Task.Run(() => UnityEnvHost.Shutdown(TimeSpan.FromMilliseconds(100)));
                        Assert.True(SpinWait.SpinUntil(() => Lifetime.ShuttingDown.IsCancellationRequested, Deadline));
                        Value value = 7;
                        Assert.Equal(7, value.AsInt());
                        Assert.False(stop.GetAwaiter().GetResult());
                    }
                };
                server.Start();
                var client = new NativeClient(server.Address); clients.Add(client);
                var driver = new Thread(() => { try { client.Run(); } catch (Exception) { env.ClientFailed = true; } }) { IsBackground = true };
                drivers.Add(driver); driver.Start();
            }
            Assert.True(SpinWait.SpinUntil(() => dispatcher.PendingCount == count, Deadline));
            if (scenario == "threads") Assert.True(callers.Count >= 3, "several native lanes must be blocked concurrently");
            if (scenario == "notification")
            {
                Assert.True(Task.Run(() => UnityEnvHost.Shutdown(Deadline)).GetAwaiter().GetResult());
                Assert.True(Lifetime.ShuttingDown.IsCancellationRequested);
                Assert.Equal(0, dispatcher.PendingCount);
                Assert.Throws<RuntimeError>(() => dispatcher.Call(() => 1));
            }
            if (scenario == "stop")
            {
                var clock = Stopwatch.StartNew();
                UnityEnvHost.Stop(dispatcher, servers[0]);
                Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1));
            }
            var timer = Stopwatch.StartNew();
            while (drivers.Exists(thread => thread.IsAlive))
            {
                UnityEnvHost.Pump();
                Assert.Equal(0, CallbackContext.Depth);
                Assert.Null(CallbackContext.HostedBy);
                Assert.False(CallbackContext.LibraryThread);
                Assert.True(timer.Elapsed < Deadline, "dispatched callback deadlocked");
                Thread.Sleep(1);
            }
            foreach (var client in clients) client.Dispose();
            foreach (var env in envs)
            {
                Assert.Null(env.Failure);
                if (scenario == "threads" || scenario == "refusal") Assert.False(env.ClientFailed);
            }
            Assert.True(Task.Run(() => UnityEnvHost.Shutdown(Deadline)).GetAwaiter().GetResult(), string.Join("; ", UnityEnvHost.Pending()));
            foreach (var server in servers) UnityEnvHost.Stop(dispatcher, server);
        }
        Assert.True(Task.Run(() => UnityEnvHost.Shutdown(Deadline)).GetAwaiter().GetResult(), string.Join("; ", UnityEnvHost.Pending()));
    }

    private sealed class LaneProbe(IEnv env, ConcurrentDictionary<int, byte> callers) : IEnv
    {
        public Space ObservationSpace => env.ObservationSpace;
        public Space ActionSpace => env.ActionSpace;
        public ResetResult Reset(long? seed, Metadata options) { callers.TryAdd(Thread.CurrentThread.ManagedThreadId, 0); return env.Reset(seed, options); }
        public StepResult Step(Value action) => env.Step(action);
    }

    private sealed class SceneEnv(int main, Func<bool> locked) : IEnv
    {
        internal Action StepBody;
        internal Exception Failure;
        internal bool ClientFailed;
        private void AssertMain() { Assert.Equal(main, Thread.CurrentThread.ManagedThreadId); Assert.True(locked()); }
        public Space ObservationSpace { get { AssertMain(); return new Discrete(2); } }
        public Space ActionSpace { get { AssertMain(); return new Discrete(3); } }
        public ResetResult Reset(long? seed, Metadata options) { AssertMain(); return new ResetResult(0, new Metadata()); }
        public StepResult Step(Value action)
        {
            AssertMain();
            try { StepBody(); }
            catch (Exception error) { Failure = error; throw; }
            return new StepResult(1, -1, true, false, new Metadata());
        }
    }
}
