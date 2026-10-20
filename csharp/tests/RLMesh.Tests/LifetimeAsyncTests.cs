using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
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
    public void RootSlotIsUntrackedBeforeItCanBeReused()
    {
        var root = ManagedRoots.Allocate(new object(), "ordering fixture");
        IntPtr pointer = GCHandle.ToIntPtr(root);
        bool observed = false;
        ManagedRoots.BeforeHandleFree = released =>
        {
            if (released != pointer) return;
            observed = true;
            Assert.False(ManagedRoots.Live.ContainsKey(pointer));
        };
        try { ManagedRoots.Free(root); Assert.True(observed); }
        finally { ManagedRoots.BeforeHandleFree = null; }
        Parallel.For(0, 2000, _ =>
        {
            var allocated = ManagedRoots.Allocate(new object(), "concurrent fixture");
            Assert.True(ManagedRoots.Live.ContainsKey(GCHandle.ToIntPtr(allocated)));
            ManagedRoots.Free(allocated);
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ShutdownAsyncDoesNotRequirePumpingTheCallersContext(bool cancel)
    {
        var env = new CancellationTests.BlockingClose();
        using var server = new EnvServer(env); server.Start();
        using var source = new CancellationTokenSource();
        var previous = SynchronizationContext.Current;
        var context = new NonPumpingContext();
        Task shutdown;
        try { SynchronizationContext.SetSynchronizationContext(context); shutdown = server.ShutdownAsync(source.Token); }
        finally { SynchronizationContext.SetSynchronizationContext(previous); }
        try
        {
            Assert.True(env.Entered.Wait(Deadline));
            if (cancel) source.Cancel(); else env.Release.Set();
            Assert.True(SpinWait.SpinUntil(() => shutdown.IsCompleted, Deadline));
            if (cancel) await Assert.ThrowsAsync<RLMesh.Ext.CancelledError>(() => shutdown);
            else await shutdown;
            Assert.Equal(0, context.Posts);
        }
        finally { env.Release.Set(); }
    }
    private sealed class NonPumpingContext : SynchronizationContext
    {
        internal int Posts;
        public override void Post(SendOrPostCallback callback, object? state) => Interlocked.Increment(ref Posts);
    }

    [Fact]
    public void EnvironmentRecordFreeIsNotEvidenceOfJoiningAForeignReturn()
    {
        var env = Env.For(new FoundationTests.Counter());
        var payload = ManagedRoots.Allocate(new object(), "unjoined foreign payload", env.Record);
        var retired = ManagedRoots.FreeFromNative(payload);
        try
        {
            env.Dispose(); Assert.True(env.WaitFreed(Deadline));
            Assert.False(retired.Joined);
        }
        finally { ManagedRoots.AwaitingReturn.TryRemove(retired, out _); }
    }

    [Fact]
    public void ACompletedServerJoinCannotCoverALaterForeignCallback()
    {
        var server = new EnvServer(new FoundationTests.Counter());
        server.Dispose(); Assert.True(server.WaitFreed(Deadline));
        Assert.True(server.Record.JoinedNativeReturn.Returned);
        var payload = ManagedRoots.Allocate(new object(), "late foreign payload", server.Record);
        var retired = ManagedRoots.FreeFromNative(payload);
        try { Assert.False(retired.Joined); }
        finally { ManagedRoots.AwaitingReturn.TryRemove(retired, out _); }
    }

    [Fact]
    public async Task IdleOperationWorkerDoesNotRootACompletedServersOwner()
    {
        var (weak, record) = await StopAndForget();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        Assert.False(weak.TryGetTarget(out _));
        Assert.True(record.WaitFreed(Deadline));
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static async Task<(WeakReference<EnvServer>, NativeHandle)> StopAndForget()
    {
        var server = new EnvServer(new FoundationTests.Counter()); server.Start();
        await server.ShutdownAsync();
        return (new WeakReference<EnvServer>(server), server.Record);
    }

    [Fact]
    public void FreedPayloadStillCountsUntilItsNativeCallReturns()
    {
        var root = ManagedRoots.Allocate(new object(), "error payload");
        ManagedRoots.ReleasedRoot pending;
        using (var nativeReturn = new NativeReturnScope())
        {
            pending = ManagedRoots.FreeFromNative(root);
            Assert.False(pending.Joined);
        }
        Assert.True(pending.Joined);
        ManagedRoots.ReturnsJoined();
        Assert.False(ManagedRoots.AwaitingReturn.ContainsKey(pending));
    }

    [Fact]
    public void UnknownPayloadReturnRemainsDiagnostic()
    {
        var root = ManagedRoots.Allocate(new object(), "error payload");
        var pending = ManagedRoots.FreeFromNative(root);
        try
        {
            Assert.False(ManagedRoots.ReturnsJoined());
            Assert.Contains(RLMesh.Ext.Lifetime.Pending(), item => item.Contains("free_payload", StringComparison.Ordinal));
        }
        finally
        {
            // This test invoked the release directly and has returned from it;
            // no native frame exists. The production barrier never assumes it.
            ManagedRoots.AwaitingReturn.TryRemove(pending, out _);
        }
    }

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
        ManagedRoots.Free(GCHandle.FromIntPtr(root));
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
