using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using RLMesh.Spaces;
using Xunit;

[assembly: CollectionBehavior(DisableTestParallelization = true)]
namespace RLMesh.Unity.Tests;

public sealed class DispatcherTests
{
    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(30);

    [Fact]
    public void TestsLoadTheNetstandardCoreAndFoundationAssemblies()
    {
        foreach (var type in new[] { typeof(MainThreadDispatcher), typeof(IEnv) })
        {
            var target = (System.Runtime.Versioning.TargetFrameworkAttribute)Attribute.GetCustomAttribute(type.Assembly, typeof(System.Runtime.Versioning.TargetFrameworkAttribute));
            Assert.Equal(".NETStandard,Version=v2.1", target.FrameworkName);
        }
        string name = typeof(MainThreadDispatcher).Assembly.GetName().Name;
        Assert.Equal("RLMesh.Unity", name);
        var friends = typeof(IEnv).Assembly.GetCustomAttributes(typeof(InternalsVisibleToAttribute), false).Cast<InternalsVisibleToAttribute>();
        Assert.Contains(friends, friend => friend.AssemblyName == name);
        Assert.Contains(friends, friend => friend.AssemblyName == "RLMesh.Unity.Tests");
        string root = System.Environment.GetEnvironmentVariable("RLMESH_TEST_ROOT");
        using var asmdef = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "integrations/unity/Runtime/RLMesh.Unity.asmdef")));
        Assert.Equal(name, asmdef.RootElement.GetProperty("name").GetString());
    }

    [Fact]
    public void MainThreadCallsRunInlineAndCanReenter()
    {
        using var dispatcher = new MainThreadDispatcher(CancellationToken.None);
        int thread = Thread.CurrentThread.ManagedThreadId;
        Assert.Equal(thread, dispatcher.Call(() => dispatcher.Call(() => Thread.CurrentThread.ManagedThreadId)));
        Assert.Equal(0, dispatcher.PendingCount);
    }

    [Fact]
    public async Task WorkerBlocksUntilPumpAndReentryRunsInline()
    {
        using var dispatcher = new MainThreadDispatcher(CancellationToken.None);
        int thread = Thread.CurrentThread.ManagedThreadId;
        var call = Task.Run(() => dispatcher.Call(() => dispatcher.Call(() => Thread.CurrentThread.ManagedThreadId)));
        AwaitQueued(dispatcher);
        Assert.False(call.IsCompleted);
        dispatcher.Pump();
        Assert.Equal(thread, await call.WaitAsync(Deadline));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ShutdownFailsPendingAndFutureCalls(bool fromLifetime)
    {
        using var cancellation = new CancellationTokenSource();
        using var dispatcher = new MainThreadDispatcher(cancellation.Token);
        bool ran = false;
        var call = Task.Run(() => Record.Exception(() => dispatcher.Call(() => { ran = true; return 1; })));
        AwaitQueued(dispatcher);
        if (fromLifetime) cancellation.Cancel(); else dispatcher.Close();
        Assert.IsType<RuntimeError>(await call.WaitAsync(Deadline));
        Assert.Throws<RuntimeError>(() => dispatcher.Call(() => 1));
        Assert.False(ran);
    }

    [Fact]
    public async Task ShutdownRacingAdmissionNeverStrandsARequest()
    {
        for (int i = 0; i < 100; i++)
        {
            using var cancellation = new CancellationTokenSource();
            using var dispatcher = new MainThreadDispatcher(cancellation.Token);
            var call = Task.Run(() => Record.Exception(() => dispatcher.Call(() => 1)));
            cancellation.Cancel();
            Assert.IsType<RuntimeError>(await call.WaitAsync(Deadline));
        }
    }

    [Fact]
    public async Task CallbackCanCloseDispatcherAndFailTheNextQueuedRequest()
    {
        using var dispatcher = new MainThreadDispatcher(CancellationToken.None);
        var first = Task.Run(() => dispatcher.Call(() => { dispatcher.Close(); return 7; }));
        AwaitQueued(dispatcher);
        var second = Task.Run(() => Record.Exception(() => dispatcher.Call(() => 8)));
        Assert.True(SpinWait.SpinUntil(() => dispatcher.PendingCount == 2, Deadline));
        dispatcher.Pump();
        Assert.Equal(7, await first.WaitAsync(Deadline));
        Assert.IsType<RuntimeError>(await second.WaitAsync(Deadline));
    }

    [Fact]
    public async Task AuthoredExceptionsPassThroughUnchanged()
    {
        using var dispatcher = new MainThreadDispatcher(CancellationToken.None);
        var recoverable = new RecoverableEnvironmentException("retry");
        Assert.Same(recoverable, Assert.Throws<RecoverableEnvironmentException>(() => dispatcher.Call<int>(() => throw recoverable)));
        var fatal = new EnvironmentException("fatal");
        Assert.Same(fatal, Assert.Throws<EnvironmentException>(() => dispatcher.Call<int>(() => throw fatal)));
        var original = new InvalidOperationException("broken scene");
        var worker = Task.Run(() => Record.Exception(() => dispatcher.Call<int>(() => throw original)));
        AwaitQueued(dispatcher);
        dispatcher.Pump();
        Assert.Same(original, await worker.WaitAsync(Deadline));
        foreach (Exception error in new Exception[] { new TypeError("type"), new ValueError("value"), new RuntimeError("runtime") })
        {
            using var inline = new MainThreadDispatcher(CancellationToken.None);
            Assert.Same(error, Record.Exception(() => inline.Call<int>(() => throw error)));
        }
    }

    [Fact]
    public async Task PumpRefusesTheWrongThread()
    {
        using var dispatcher = new MainThreadDispatcher(CancellationToken.None);
        var result = Task.Run(() => Record.Exception(dispatcher.Pump));
        Assert.IsType<InvalidOperationException>(await result.WaitAsync(Deadline));
    }

    [Fact]
    public void HostJsonNamesUnityAndTheActualBackend()
    {
        Assert.Equal("{\"language\":\"csharp\",\"framework_versions\":{\"unity\":\"6000\",\"mono\":\"6\"}}", HostRuntime.Json("6000", false, "6"));
        Assert.Contains("\"il2cpp\":\"6000\"", HostRuntime.Json("6000", true, "6000"));
        Assert.Contains("\\\"\\u000a", HostRuntime.Json("\"\n", false, "6"));
    }

    [Fact]
    public async Task PythonDrivesDispatcherEnvOnAFakeMainLoop()
    {
        using var dispatcher = new MainThreadDispatcher(CancellationToken.None);
        var env = new Counter();
        using var server = new EnvServer(new MainThreadEnv(dispatcher, env), new EnvServerOptions
        {
            Ext = new Ext.EnvServerOptions { HostRuntimeJson = HostRuntime.Json("fake-main-loop", false, "test") }
        });
        server.Start();
        string root = System.Environment.GetEnvironmentVariable("RLMESH_TEST_ROOT") ?? throw new Exception("run through test:csharp");
        var start = new ProcessStartInfo("uv") { WorkingDirectory = root, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (string arg in new[] { "run", "--no-sync", "python", Path.Combine(root, "integrations/unity/Tests~/client.py"), "--address", server.Address, "--episodes", "3", "--counter" }) start.ArgumentList.Add(arg);
        using var client = Process.Start(start);
        var stdout = client.StandardOutput.ReadToEndAsync();
        var stderr = client.StandardError.ReadToEndAsync();
        var clock = Stopwatch.StartNew();
        try
        {
            while (!client.WaitForExit(1))
            {
                dispatcher.Pump();
                if (clock.Elapsed > Deadline) { client.Kill(true); throw new TimeoutException("Python client did not finish"); }
            }
            Assert.True(client.ExitCode == 0, await stdout + await stderr);
            Assert.Equal(3, env.Resets);
            Assert.Equal(9, env.Steps);
        }
        finally { dispatcher.Close(); }
    }

    private static void AwaitQueued(MainThreadDispatcher dispatcher) => Assert.True(SpinWait.SpinUntil(() => dispatcher.PendingCount > 0, Deadline));

    private sealed class Counter : IEnv
    {
        private readonly int thread = Thread.CurrentThread.ManagedThreadId;
        public int Resets, Steps;
        private int position;
        private void OnMain() { Assert.Equal(thread, Thread.CurrentThread.ManagedThreadId); }
        public Space ObservationSpace { get { OnMain(); return new Discrete(10); } }
        public Space ActionSpace { get { OnMain(); return new Discrete(3); } }
        public ResetResult Reset(long? seed, Metadata options)
        {
            OnMain(); Resets++; position = 0;
            return new ResetResult(0, new Metadata());
        }
        public StepResult Step(Value action)
        {
            OnMain(); Steps++; position++;
            return new StepResult(position, -position, position == 3, false, new Metadata());
        }
    }
}
