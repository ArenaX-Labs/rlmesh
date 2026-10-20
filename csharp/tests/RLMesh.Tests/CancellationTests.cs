using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Ext;
using RLMesh.Internal;
using Xunit;

namespace RLMesh.Tests;

public sealed class CancellationTests
{
    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(20);
    internal sealed class BlockingStep : FoundationTests.Counter
    {
        internal readonly ManualResetEventSlim Entered = new(false), Release = new(false);
        public override StepResult Step(Value? action)
        {
            Entered.Set(); Release.Wait();
            // These constructors must also work after the terminal gate closes.
            using var tensor = new Tensor(new byte[] { 1 }, new long[] { 1 }, DType.Uint8);
            return base.Step(action);
        }
    }
    internal sealed class BlockingClose : IEnv, IClose
    {
        private readonly FoundationTests.Counter env = new();
        internal readonly ManualResetEventSlim Entered = new(false), Release = new(false);
        internal int Thread;
        public Spaces.Space ObservationSpace => env.ObservationSpace;
        public Spaces.Space ActionSpace => env.ActionSpace;
        public ResetResult Reset(long? seed, Metadata? options) => env.Reset(seed, options);
        public StepResult Step(Value? action) => env.Step(action);
        public void Close() { Thread = System.Environment.CurrentManagedThreadId; Entered.Set(); Release.Wait(); }
    }
    [Fact]
    public async Task ShutdownCancellationStopsOnlyTheWaitAndPreservesExceptionIdentity()
    {
        var env = new BlockingClose(); using var server = new EnvServer(env); server.Start();
        int caller = System.Environment.CurrentManagedThreadId;
        using var source = new CancellationTokenSource();
        Task shutdown = server.ShutdownAsync(source.Token);
        Assert.True(env.Entered.Wait(Deadline)); Assert.NotEqual(caller, env.Thread);
        source.Cancel();
        var error = await Assert.ThrowsAsync<CancelledError>(() => shutdown.WaitAsync(Deadline));
        Assert.Equal(source.Token, error.CancellationToken); Assert.True(shutdown.IsCanceled);
        Assert.Same(error, await Assert.ThrowsAsync<CancelledError>(() => shutdown));
        Assert.False(server.Record.IsFreed);
        env.Release.Set();
        await server.ShutdownAsync().WaitAsync(Deadline);
    }
    [Fact]
    public async Task AlreadyCancelledShutdownDoesNotTouchTheServer()
    {
        var env = new FoundationTests.Counter(); using var server = new EnvServer(env); server.Start();
        using var source = new CancellationTokenSource(); source.Cancel();
        var error = await Assert.ThrowsAsync<CancelledError>(() => server.ShutdownAsync(source.Token));
        Assert.Equal(source.Token, error.CancellationToken);
        Assert.False(server.Wait(TimeSpan.Zero)); Assert.Equal(0, env.Closes);
    }
    [Fact]
    public async Task DisposeDuringShutdownKeepsItsNativeReferenceUntilDrainReturns()
    {
        var env = new BlockingClose(); var server = new EnvServer(env); server.Start();
        Task shutdown = server.ShutdownAsync(); Assert.True(env.Entered.Wait(Deadline));
        var dispose = Task.Run(server.Dispose);
        Assert.False(server.WaitFreed(TimeSpan.Zero));
        env.Release.Set();
        await Task.WhenAll(shutdown, dispose).WaitAsync(Deadline);
        Assert.True(server.WaitFreed(Deadline));
    }
    [Fact]
    public async Task NativeRunHonoursThePerOperationTokenWhileStepIsBlocked()
    {
        var env = new BlockingStep(); using var server = new EnvServer(env); server.Start();
        using var client = new NativeClient(server.Address); using var source = new CancellationTokenSource();
        var trace = new ConcurrentQueue<string>(); CancellationBridge.Trace = trace.Enqueue;
        try
        {
            Task run = Task.Run(() => client.Run(source.Token));
            Assert.True(env.Entered.Wait(Deadline)); source.Cancel();
            var error = await Assert.ThrowsAsync<CancelledError>(() => run.WaitAsync(Deadline));
            Assert.Equal(source.Token, error.CancellationToken);
            Assert.Equal(new[] { "cancel", "call returned", "registration disposed", "token freed" }, trace.ToArray());
        }
        finally { CancellationBridge.Trace = null; env.Release.Set(); }
    }
    [Fact]
    public async Task ExplicitRemoteStepHonoursThePerOperationTokenWhileStepIsBlocked()
    {
        var env = new BlockingStep(); using var server = new EnvServer(env); server.Start();
        using var client = new NativeClient(server.Address); using var source = new CancellationTokenSource();
        Assert.False(client.RawResetFails());
        var trace = new ConcurrentQueue<string>(); CancellationBridge.Trace = trace.Enqueue;
        try
        {
            Task step = Task.Run(() => client.Step(source.Token));
            Assert.True(env.Entered.Wait(Deadline)); source.Cancel();
            var error = await Assert.ThrowsAsync<CancelledError>(() => step.WaitAsync(Deadline));
            Assert.Equal(source.Token, error.CancellationToken);
            Assert.Equal(new[] { "cancel", "call returned", "registration disposed", "token freed" }, trace.ToArray());
        }
        finally { CancellationBridge.Trace = null; env.Release.Set(); }
    }
    [Fact]
    public async Task CancellationBetweenRegistrationAndNativeAdmissionIsNotLost()
    {
        var env = new FoundationTests.Counter(); using var server = new EnvServer(env); server.Start();
        using var client = new NativeClient(server.Address); using var source = new CancellationTokenSource();
        using var registered = new ManualResetEventSlim(); using var proceed = new ManualResetEventSlim();
        CancellationBridge.BeforeCall = () => { registered.Set(); proceed.Wait(); };
        try
        {
            Task run = Task.Run(() => client.Run(source.Token));
            Assert.True(registered.Wait(Deadline)); source.Cancel(); proceed.Set();
            await Assert.ThrowsAsync<CancelledError>(() => run.WaitAsync(Deadline));
            Assert.Equal(0, env.Resets); Assert.Equal(0, env.Steps);
        }
        finally { CancellationBridge.BeforeCall = null; proceed.Set(); }
        int starts = 0;
        Assert.Throws<CancelledError>(() => InvokeBridge(source.Token, () => starts++));
        Assert.Equal(0, starts);
    }
    [Fact]
    public async Task RegistrationDisposalJoinsAnInFlightCancelBeforeFree()
    {
        using var source = new CancellationTokenSource(); using var cancelEntered = new ManualResetEventSlim();
        using var releaseCancel = new ManualResetEventSlim(); using var bodyEntered = new ManualResetEventSlim();
        using var complete = new ManualResetEventSlim();
        var trace = new ConcurrentQueue<string>(); CancellationBridge.Trace = trace.Enqueue;
        CancellationBridge.BeforeCancel = () => { cancelEntered.Set(); releaseCancel.Wait(); };
        try
        {
            Task run = Task.Run(() => InvokeBridge(source.Token, () => { bodyEntered.Set(); complete.Wait(); }));
            Assert.True(bodyEntered.Wait(Deadline));
            Task cancel = Task.Run(source.Cancel); Assert.True(cancelEntered.Wait(Deadline));
            complete.Set();
            Assert.True(SpinWait.SpinUntil(() => trace.Contains("call returned"), Deadline));
            Assert.DoesNotContain("registration disposed", trace); Assert.DoesNotContain("token freed", trace);
            releaseCancel.Set(); await Task.WhenAll(run, cancel).WaitAsync(Deadline);
            Assert.Equal(new[] { "call returned", "cancel", "registration disposed", "token freed" }, trace.ToArray());
            source.Cancel(); // No remaining registration can touch the freed reference.
        }
        finally { releaseCancel.Set(); complete.Set(); CancellationBridge.BeforeCancel = null; CancellationBridge.Trace = null; }
    }
    [Fact]
    public async Task CompletionRacesAndWorkerReuseDoNotCarryCancellationForward()
    {
        using var server = new EnvServer(new FoundationTests.Counter()); server.Start();
        using var client = new NativeClient(server.Address);
        var done = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        LibraryWorkers.Enqueue(() =>
        {
            try
            {
                for (int i = 0; i < 30; i++)
                {
                    using var source = new CancellationTokenSource();
                    var cancel = new Thread(source.Cancel); cancel.Start();
                    try { client.Run(source.Token); } catch (CancelledError) { }
                    cancel.Join();
                    client.Run(); // The same library worker always completes a fresh operation.
                }
                done.SetResult(true);
            }
            catch (Exception e) { done.SetException(e); }
        });
        await done.Task.WaitAsync(Deadline);
    }
    [Fact]
    public void CancellationCallbackContainsFailures()
    {
        using var source = new CancellationTokenSource();
        var failure = new Exception("cancel callback failure");
        CancellationBridge.BeforeCancel = () => throw failure;
        try { InvokeBridge(source.Token, source.Cancel); }
        finally { CancellationBridge.BeforeCancel = null; }
        Assert.Contains(failure, NativeHandle.CleanupFailures);
    }
    [Fact]
    public void CompletionUnregistersBeforeALateCancellation()
    {
        using var source = new CancellationTokenSource();
        var trace = new ConcurrentQueue<string>(); CancellationBridge.Trace = trace.Enqueue;
        try
        {
            InvokeBridge(source.Token, () => { });
            source.Cancel();
            Assert.Equal(new[] { "call returned", "registration disposed", "token freed" }, trace.ToArray());
        }
        finally { CancellationBridge.Trace = null; }
    }
    [Fact]
    public async Task ShutdownCompletionRacingCancellationAlwaysDrainsOnce()
    {
        for (int i = 0; i < 20; i++)
        {
            var env = new FoundationTests.Counter(); using var server = new EnvServer(env); server.Start();
            using var source = new CancellationTokenSource();
            Task shutdown = server.ShutdownAsync(source.Token);
            Task cancel = Task.Run(source.Cancel);
            try { await shutdown.WaitAsync(Deadline); }
            catch (CancelledError error) { Assert.Equal(source.Token, error.CancellationToken); Assert.True(shutdown.IsCanceled); }
            await cancel.WaitAsync(Deadline);
            await server.ShutdownAsync().WaitAsync(Deadline);
            Assert.Equal(1, env.Closes);
        }
    }
    private static unsafe void InvokeBridge(CancellationToken token, Action action) =>
        CancellationBridge.Run(token, _ => { action(); return 0; });
}
