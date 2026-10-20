using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Ext;
using RLMesh.Internal;
using RLMesh.Spaces;
using Xunit;

namespace RLMesh.Tests;

// Each case runs inside BarrierTests' fresh collectible context, over a real
// native server/client. A dedicated managed thread stands in for the engine.
internal static class DispatchContextFixtures
{
    internal static void Run(string scenario)
    {
        using var queue = new BlockingCollection<Action>();
        Exception? failure = null;
        var host = new Thread(() =>
        {
            foreach (var job in queue.GetConsumingEnumerable())
            {
                job();
                Assert.Equal(0, CallbackContext.Depth);
                Assert.Null(CallbackContext.HostedBy);
                Assert.False(CallbackContext.LibraryThread);
            }
        }) { IsBackground = true };
        host.Start();
        var env = new DispatchedEnv(queue);
        var server = new EnvServer(env);
        env.StepBody = () =>
        {
            Assert.True(CallbackContext.Depth > 0);
            Assert.Same(server.Record, CallbackContext.HostedBy);
            var clock = Stopwatch.StartNew();
            if (scenario == "dispose")
            {
                server.Dispose();
                Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1), "Dispose waited for its own dispatched callback");
            }
            else if (scenario == "refusal")
            {
                Assert.False(Lifetime.Shutdown(TimeSpan.FromSeconds(5)));
                Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1), "callback shutdown must refuse immediately");
                Assert.Equal(LifetimeState.Open, Lifetime.State);
            }
            else
            {
                var shutdown = Task.Run(() => Lifetime.Shutdown(TimeSpan.FromMilliseconds(100)));
                Assert.True(SpinWait.SpinUntil(() => Lifetime.ShuttingDown.IsCancellationRequested, TimeSpan.FromSeconds(5)));
                Value value = 7;
                Assert.Equal(7, value.AsInt()); // callback admission survives the closed gate.
                Assert.False(shutdown.GetAwaiter().GetResult());
            }
            return new StepResult(1, 1, true, false, new Metadata());
        };
        server.Start();
        var client = new NativeClient(server.Address);
        var driver = new Thread(() => { try { client.Run(); } catch (Exception e) { failure = e; } }) { IsBackground = true };
        driver.Start();
        Assert.True(driver.Join(TimeSpan.FromSeconds(10)), "dispatched callback deadlocked");
        queue.CompleteAdding();
        Assert.True(host.Join(TimeSpan.FromSeconds(5)));
        client.Dispose();
        // Shutdown may cancel the remote run; assertion failures in the callback
        // must never be confused with that expected terminal cancellation.
        Assert.Null(env.Failure);
        if (scenario == "refusal") Assert.Null(failure);
        Assert.True(Lifetime.Shutdown(TimeSpan.FromSeconds(10)), string.Join("; ", Lifetime.Pending()));
        server.Dispose();
    }

    private sealed class DispatchedEnv(BlockingCollection<Action> queue) : IEnv
    {
        internal Func<StepResult> StepBody = null!;
        internal Exception? Failure;
        public Space ObservationSpace => new Discrete(2);
        public Space ActionSpace => new Discrete(2);
        public ResetResult Reset(long? seed, Metadata? options) => Dispatch(() => new ResetResult(0, new Metadata()));
        public StepResult Step(Value? action) => Dispatch(StepBody);
        private T Dispatch<T>(Func<T> callback)
        {
            var context = CallbackContext.Capture();
            var done = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            queue.Add(() =>
            {
                T result = default!;
                Exception? failure = null;
                using (context.Enter())
                {
                    try { result = callback(); }
                    catch (Exception error) { Failure = failure = error; }
                }
                if (failure != null) done.SetException(failure);
                else done.SetResult(result);
            });
            return done.Task.GetAwaiter().GetResult();
        }
    }
}
