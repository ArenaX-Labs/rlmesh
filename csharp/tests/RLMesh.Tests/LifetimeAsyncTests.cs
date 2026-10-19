using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Internal;
using RLMesh.Interop;
using Xunit;

namespace RLMesh.Tests;

public sealed class LifetimeAsyncTests
{
    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(20);

    [Fact]
    public async Task FailedRootCleanupStillFreesAndRetiresTheEnvironment()
    {
        var authored = new FoundationTests.Counter(); var env = Env.For(authored);
        IntPtr root = default;
        var failure = new InvalidOperationException("cleanup failure");
        Callbacks.CleanupHook = data => { root = data; throw failure; };
        try { env.Dispose(); Assert.True(env.WaitFreed(Deadline)); }
        finally { Callbacks.CleanupHook = null; }
        Assert.Contains(failure, NativeHandle.CleanupFailures);
        Assert.Contains(NativeHandle.CleanupFailures, e => e.Message == "environment root was not released by native free");
        // The intentionally failed callback leaked its root; reclaim the test fixture.
        GCHandle.FromIntPtr(root).Free();
        var rebuilt = Task.Run(() => Env.For(authored));
        using var next = await rebuilt.WaitAsync(Deadline);
        Assert.NotSame(env.Record, next.Record);
    }
    [Fact]
    public async Task DisposeSignalsShutdownWhileWaitHoldsAReference()
    {
        var server = new EnvServer(new FoundationTests.Counter()); server.Start();
        var waiter = Task.Run(() => server.Wait());
        Assert.True(SpinWait.SpinUntil(() => NativeHandle.Count(server.Record.Word) > 0, Deadline));
        server.Dispose();
        Assert.True(await waiter.WaitAsync(Deadline));
        Assert.True(server.WaitFreed(Deadline));
    }
    [Fact]
    public async Task PoolThreadDisposePermitsImmediateRebind()
    {
        await Task.Run(() =>
        {
            var first = new EnvServer(new FoundationTests.Counter()); first.Start(); string address = first.Address;
            first.Dispose();
            Assert.True(first.Record.IsFreed);
            using var next = new EnvServer(new FoundationTests.Counter(), new() { Address = address });
            next.Start();
        }).WaitAsync(Deadline);
    }
}
