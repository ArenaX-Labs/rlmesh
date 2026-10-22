using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Internal;
using RLMesh.Interop;
using RLMesh.Spaces;
using Xunit;

[assembly: CollectionBehavior(DisableTestParallelization = true)]
namespace RLMesh.Tests;

public sealed unsafe class FoundationTests
{
    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(20);
    internal static string Root => System.Environment.GetEnvironmentVariable("RLMESH_TEST_ROOT") ?? throw new InvalidOperationException("run mise run test:csharp");

    internal class Counter : IEnv, IClose
    {
        public readonly Space Obs = new Discrete(100);
        public readonly Space Act = new Discrete(2);
        internal int Steps, Closes, Resets;
        internal Exception? Error;
        internal readonly HashSet<int> Threads = new();
        public virtual Space ObservationSpace => Obs;
        public Space ActionSpace => Act;
        public virtual ResetResult Reset(long? seed, Metadata? options) { Interlocked.Increment(ref Resets); Steps = 0; return new(0, new()); }
        public virtual StepResult Step(Value? action)
        {
            lock (Threads) Threads.Add(System.Environment.CurrentManagedThreadId);
            if (Error != null) throw Error;
            return new(Interlocked.Increment(ref Steps), 1, Steps >= 2, false, new());
        }
        public void Close() { Interlocked.Increment(ref Closes); }
    }

    [Fact]
    public void ValuesHaveExactKindsAndStrictReads()
    {
        Value[] ints = { (sbyte)-1, (byte)1, (short)-1, (ushort)1, -1, 1U, -1L };
        Assert.All(ints, v => Assert.Equal(ValueKind.Int, v.Kind));
        Value[] floats = { 1F, 1D };
        Assert.All(floats, v => Assert.Equal(ValueKind.Float, v.Kind));
        Value b = true, i = 1, f = 1D;
        Assert.False(b == i); Assert.False(i == f);
        Assert.Throws<TypeError>(() => i.AsFloat());
        Assert.Throws<TypeError>(() => b.AsInt());
        Assert.Throws<TypeError>(() => i.Count);
        Assert.Throws<TypeError>(() => i.TryGet(-1, out _));
        Assert.True(Value.None.IsNone);
        Assert.Equal(long.MaxValue, Value.FromUInt64(long.MaxValue).AsInt());
        Assert.Throws<ValueError>(() => Value.FromUInt64(ulong.MaxValue));
        Value text = "a\0b";
        Assert.Equal("a\0b", text.AsText());
        Assert.Throws<ValueError>(() => { Value invalid = "\uD800"; });
        Assert.Equal(new byte[] { 0, 1, 255 }, Value.Bytes(new byte[] { 0, 1, 255 }).AsBytes());
        var list = Value.List(1, "a"); var tuple = Value.Tuple(1, "a");
        Assert.Equal(2, list.Count); Assert.Equal("a", list[1].AsText());
        Assert.False(list.TryGet(2, out _)); Assert.False(list.TryGet(-1, out _));
        Assert.Equal("index", Assert.Throws<ArgumentOutOfRangeException>(() => list[3]).ParamName);
        Assert.Equal(2, list.AsList().Count); Assert.Equal(2, tuple.AsTuple().Count);
        Assert.Throws<TypeError>(() => tuple.AsList());
        Assert.False(list == tuple);
        Value nan = double.NaN;
        Assert.False(nan.Equals(nan));
        Assert.Equal(Value.List(1, 2).GetHashCode(), Value.List(1, 2).GetHashCode());
    }
    [Fact]
    public void IntegerIndexingChecksKindBeforeBounds()
    {
        foreach (var dict in new[] { Value.Dict(new Metadata()), Value.Dict(new Metadata { ["a"] = 1 }) })
            foreach (int index in new[] { -1, 0, 1, int.MaxValue })
            {
                Assert.Throws<TypeError>(() => dict.TryGet(index, out _));
                Assert.Throws<TypeError>(() => dict[index]);
            }
        foreach (var sequence in new[] { Value.List(1), Value.Tuple(1), Value.List(), Value.Tuple() })
            foreach (int index in new[] { -1, sequence.Count, int.MaxValue })
            {
                Assert.False(sequence.TryGet(index, out _));
                Assert.Equal("index", Assert.Throws<ArgumentOutOfRangeException>(() => sequence[index]).ParamName);
            }
    }
    [Fact]
    public void EmptyOwnedTextIsValid()
    {
        Assert.Equal("", Utf8.OwnedText(default));
        Value text = "";
        Assert.Equal("", text.AsText());
    }
    [Fact]
    public void PinsKeepTensorStorageAliveAfterLeaseDisposal()
    {
        using var tensor = new Tensor(new byte[] { 7, 8 }, new long[] { 2 }, DType.Uint8);
        var lease = tensor.Buffer();
        var memory = lease.Memory;
        var first = memory.Pin(); var second = memory.Slice(1).Pin();
        tensor.Dispose(); lease.Dispose(); Collect();
        Assert.Equal(OwnershipState.Closing, NativeHandle.State(lease.Record.Word));
        Assert.Equal(2, NativeHandle.Count(lease.Record.Word));
        Assert.Equal(7, *(byte*)first.Pointer); Assert.Equal(8, *(byte*)second.Pointer);
        Assert.Throws<ObjectDisposedException>(() => memory.Pin());
        first.Dispose(); Collect();
        Assert.Equal(8, *(byte*)second.Pointer);
        second.Dispose();
        Assert.True(lease.Record.WaitFreed(Deadline));
    }
    [Fact]
    public void MetadataPreservesInsertionOrderAndIgnoresItForEquality()
    {
        var a = new Metadata { ["b"] = 2, ["a\0"] = 1, [""] = true };
        var b = new Metadata { [""] = true, ["a\0"] = 1, ["b"] = 2 };
        Assert.True(a == b); Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.Equal(new[] { "b", "a\0", "" }, a.Keys);
        a["b"] = 3;
        Assert.Equal(new[] { "b", "a\0", "" }, a.Select(p => p.Key));
        Assert.Throws<ArgumentException>(() => a.Add("b", 0));
        Value native = a;
        Assert.True(a == native.AsDict());
        a.Remove("b"); a.Add("b", 3);
        Assert.Equal(new[] { "a\0", "", "b" }, a.Keys);
        Assert.True(((ICollection<KeyValuePair<string, Value>>)a).Remove(new("b", 3)));
        ((ICollection<KeyValuePair<string, Value>>)a).Clear();
        Assert.Empty(a);
    }
    [Fact]
    public void TensorLeaseSurvivesTensorDisposalAndCollection()
    {
        var tensor = new Tensor(new byte[] { 1, 2, 3 }, new long[] { 3 }, DType.Uint8);
        Assert.Equal(new long[] { 3 }, tensor.Shape); Assert.Equal(DType.Uint8, tensor.Dtype);
        Assert.Equal(new byte[] { 1, 2, 3 }, tensor.Tobytes());
        Value value = tensor;
        using var fromValue = value.AsTensor();
        var lease = tensor.Buffer(); ReadOnlyMemory<byte> memory = lease.Memory;
        tensor.Dispose(); tensor.Dispose();
        Assert.Throws<ObjectDisposedException>(() => tensor.Tobytes());
        Collect();
        Assert.Equal(new byte[] { 1, 2, 3 }, lease.Span.ToArray());
        Assert.Equal(3, memory.Span[2]);
        lease.Dispose();
        Assert.Throws<ObjectDisposedException>(() => lease.Memory);
        Assert.Throws<ObjectDisposedException>(() => memory.Span.Length);
        Assert.Throws<ValueError>(() => new Tensor(Array.Empty<byte>(), new long[] { -1 }, DType.Bool));
        Assert.Throws<ValueError>(() => new Tensor(new byte[1], new long[] { 3 }, DType.Float32));
        var abandoned = MakeLease();
        Collect();
        Assert.Equal(42, abandoned.Span[0]); abandoned.Dispose();
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static TensorLease<byte> MakeLease() => new Tensor(new byte[] { 42 }, new long[] { 1 }, DType.Uint8).Buffer();
    [Fact]
    public void ResultsRejectNullAndDefaultResultsBecomeTypeErrors()
    {
        Assert.Throws<ArgumentNullException>(() => new ResetResult(null!, new()));
        Assert.Throws<ArgumentNullException>(() => new StepResult(0, 0, false, false, null!));
        var (obs, info) = new ResetResult(1, new()); Assert.Equal(1, obs.AsInt()); Assert.Empty(info);
        var (o, r, term, trunc, inf) = new StepResult(2, 3, true, false, new());
        Assert.Equal(2, o.AsInt()); Assert.Equal(3, r); Assert.True(term); Assert.False(trunc); Assert.Empty(inf);
        using var env = Env.For(new DefaultResultEnv());
        using var session = new LocalSession(env);
        Assert.Throws<TypeError>(() => session.Reset());
    }
    private sealed class DefaultResultEnv : Counter { public override ResetResult Reset(long? seed, Metadata? options) => default; }

    [Fact]
    public void BoundArmsAndBoolPathsRoundTripThroughC()
    {
        (Bound low, Bound high, DType dtype, RLMeshBoundType type)[] cases =
        {
            (false, true, DType.Bool, RLMeshBoundType.RLMESH_BOUND_TYPE_BOOL),
            (-2L, 5L, DType.Int64, RLMeshBoundType.RLMESH_BOUND_TYPE_INT),
            (0UL, ulong.MaxValue, DType.Uint64, RLMeshBoundType.RLMESH_BOUND_TYPE_UINT),
            (-1.5, 2.5, DType.Float64, RLMeshBoundType.RLMESH_BOUND_TYPE_FLOAT),
        };
        foreach (var test in cases)
        {
            var box = new Box(test.low, test.high, new long[] { 1 }, new() { Dtype = test.dtype });
            using var call = box.Handle.Acquire(box);
            RLMeshBound low = default, high = default; UIntPtr length = default;
            Status.Check(Native.rlmesh_box_low(call.Space, &low, (UIntPtr)1, &length));
            Status.Check(Native.rlmesh_box_high(call.Space, &high, (UIntPtr)1, &length));
            Assert.Equal(test.type, low.type_); Assert.Equal(test.type, high.type_);
            Assert.Equal(test.low.NativeValue.value.u, low.value.u); Assert.Equal(test.high.NativeValue.value.u, high.value.u);
            bool has = false; RLMeshBoundsKind kind = default;
            Status.Check(Native.rlmesh_box_bounds_kind(call.Space, &has, &kind)); Assert.True(has);
        }
        Bound[] conversions = { true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1U, 1L, 1UL, 1F, 1D };
        Assert.Equal(11, conversions.Length);
        Assert.Throws<ValueError>(() => { Bound invalid = double.NaN; });
        var discrete = new Discrete(3, new() { Start = 2, Dtype = DType.Int32 });
        using var dc = discrete.Handle.Acquire(discrete);
        long n = 0, start = 0; Status.Check(Native.rlmesh_discrete_n(dc.Space, &n)); Status.Check(Native.rlmesh_discrete_start(dc.Space, &start));
        Assert.Equal(3, n); Assert.Equal(2, start);
        Value yes = true, no = false;
        Assert.True(yes.AsBool()); Assert.False(no.AsBool());
        using var yesCall = yes.Handle.Acquire(yes); using var noCall = no.Handle.Acquire(no);
        Assert.Equal((byte)1, Native.rlmesh_value_equals(yesCall.Value, yesCall.Value));
        Assert.Equal((byte)0, Native.rlmesh_value_equals(yesCall.Value, noCall.Value));
    }

    [Fact]
    public void ForIsUniqueConcurrentAndReentrantBuildIsRefused()
    {
        var authored = new Counter();
        var handles = new Env[32];
        Parallel.For(0, handles.Length, i => handles[i] = Env.For(authored));
        Assert.All(handles, h => Assert.Same(handles[0], h));
        handles[0].Dispose(); Assert.True(handles[0].WaitFreed(Deadline));
        using var next = Env.For(authored); Assert.NotSame(handles[0].Record, next.Record);
        Assert.Throws<RuntimeError>(() => Env.For(new ReentrantEnv()));
    }
    private sealed class ReentrantEnv : Counter
    {
        public override Space ObservationSpace { get { Env.For(this); return Obs; } }
    }
    [Fact]
    public void LibraryOwnerAndClosingBorrowCannotProduceAnotherHandle()
    {
        var authored = new Counter();
        var server = new EnvServer(authored);
        using (server)
        {
            Assert.Throws<RuntimeError>(() => Env.For(authored));
            server.Start(); Assert.False(server.Wait(TimeSpan.Zero));
            server.Shutdown(); Assert.True(server.Wait(TimeSpan.Zero));
            Assert.Equal(1, authored.Closes);
        }
        Assert.True(server.WaitFreed(Deadline));
        using var env = Env.For(authored);
        using var borrowed = new EnvServer(env);
        Assert.Same(env, Env.For(authored));
        env.Dispose(); Assert.Throws<RuntimeError>(() => Env.For(authored));
        Assert.False(env.WaitFreed(TimeSpan.Zero));
        borrowed.Dispose(); Assert.True(env.WaitFreed(Deadline));
        using var next = Env.For(authored);
        Assert.NotSame(env.Record, next.Record);
    }
    [Fact]
    public void RegistryAdoptionWinsAgainstAnOldFinalizer()
    {
        var authored = new Counter();
        using var entered = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        NativeHandle.BeforeFinalize = token => { if (token.Record.Kind == HandleKind.Env) { entered.Set(); release.Wait(Deadline); } };
        try
        {
            var (weak, record) = UnreachableEnv(authored);
            GC.Collect(); Assert.True(entered.Wait(Deadline));
            Assert.False(weak.TryGetTarget(out _));
            using var adopted = Env.For(authored);
            Assert.Same(record, adopted.Record); Assert.Equal(1U, NativeHandle.Generation(record.Word));
            release.Set(); GC.WaitForPendingFinalizers();
            Assert.Equal(OwnershipState.Idle, NativeHandle.State(record.Word));
        }
        finally { release.Set(); NativeHandle.BeforeFinalize = null; }
    }
    [Fact]
    public void FinalizationRetiresBeforeRebuildingAndFinalizesUnreachableServers()
    {
        var authored = new Counter();
        var (weak, record) = UnreachableEnv(authored);
        Collect(); Assert.False(weak.TryGetTarget(out _)); Assert.True(record.WaitFreed(Deadline));
        using var next = Env.For(authored); Assert.NotSame(record, next.Record);
        var other = new Counter();
        var (serverWeak, serverRecord) = UnreachableServer(other);
        Collect(); Assert.False(serverWeak.TryGetTarget(out _)); Assert.True(serverRecord.WaitFreed(Deadline));
        using var after = Env.For(other);
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (WeakReference<Env>, NativeHandle) UnreachableEnv(IEnv authored)
    { var env = Env.For(authored); return (new(env), env.Record); }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (WeakReference<EnvServer>, NativeHandle) UnreachableServer(IEnv authored)
    { var server = new EnvServer(authored); server.Start(); return (new(server), server.Record); }

    [Fact]
    public void OwnershipTransitionsAndConcurrentDisposeAreSafe()
    {
        using var tensor = new Tensor(new byte[1], new long[] { 1 }, DType.Bool);
        var record = tensor.Handle;
        var a = record.Acquire(tensor); var b = record.Acquire(tensor);
        Assert.Equal(2, NativeHandle.Count(record.Word));
        tensor.Dispose(); Assert.Equal(OwnershipState.Closing, NativeHandle.State(record.Word));
        Assert.Throws<ObjectDisposedException>(() => record.Acquire());
        a.Dispose(); Assert.Equal(1, NativeHandle.Count(record.Word));
        b.Dispose(); Assert.True(record.WaitFreed(Deadline));
        var authored = new Counter(); using var env = Env.For(authored);
        using var started = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        var call = new Thread(() => { using var lease = env.Record.Acquire(env); started.Set(); release.Wait(Deadline); });
        call.Start();
        Assert.True(started.Wait(Deadline)); env.Dispose(); Assert.False(env.WaitFreed(TimeSpan.Zero));
        release.Set(); Assert.True(call.Join(Deadline)); Assert.True(env.WaitFreed(Deadline));
        using var transferred = new Tensor(new byte[1], new long[] { 1 }, DType.Bool);
        Assert.True(transferred.Handle.TryReserveTransfer()); Assert.False(transferred.Handle.TryReserveTransfer());
        Assert.Throws<ObjectDisposedException>(() => transferred.Handle.Acquire());
        transferred.Handle.FinishTransfer(false);
        using (transferred.Handle.Acquire()) Assert.False(transferred.Handle.TryReserveTransfer());
        Assert.True(transferred.Handle.TryAdopt(out uint generation));
        transferred.Handle.CloseOwner(0, true); Assert.False(transferred.Handle.IsFreed);
        transferred.Handle.CloseOwner(generation, true); Assert.True(transferred.Handle.IsFreed);
    }
    [Fact]
    public void CleanupWorkersProgressIndependentlyAndWaitFreedNeverBlocksInCallbacks()
    {
        using var held = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        var first = Env.For(new Counter()); var second = Env.For(new Counter());
        NativeHandle.BeforeFree = record => { if (ReferenceEquals(record, first.Record)) { held.Set(); release.Wait(Deadline); } };
        try
        {
            CallbackContext.Enter();
            try
            {
                first.Dispose();
                var sw = Stopwatch.StartNew(); Assert.False(first.WaitFreed(Deadline)); Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1));
                second.Dispose();
            }
            finally { CallbackContext.Exit(); }
            Assert.True(held.Wait(Deadline)); Assert.True(second.WaitFreed(Deadline)); Assert.False(first.WaitFreed(TimeSpan.Zero));
            release.Set(); Assert.True(first.WaitFreed(Deadline));
        }
        finally { release.Set(); NativeHandle.BeforeFree = null; }
    }

    [Fact]
    public void WorkerRechecksQueueAfterTimedWait()
    {
        using var done = new ManualResetEventSlim();
        int waitingThread = 0, runningThread = 0;
        CleanupWorkers.WaitForWork = sync =>
        {
            CleanupWorkers.WaitForWork = null;
            waitingThread = System.Environment.CurrentManagedThreadId;
            // Model an enqueuer winning Sync after timeout but before the
            // worker reacquires it: idle still includes the retiring worker.
            CleanupWorkers.Enqueue(new CleanupRecord(() =>
            {
                runningThread = System.Environment.CurrentManagedThreadId;
                done.Set();
            }));
            return false;
        };
        try
        {
            CleanupWorkers.Enqueue(new CleanupRecord(() => { }));
            Assert.True(done.Wait(Deadline));
            Assert.Equal(waitingThread, runningThread);
        }
        finally { CleanupWorkers.WaitForWork = null; }
    }
    [Fact]
    public void AfterFreeFailureDoesNotStrandTheRecord()
    {
        var failure = new InvalidOperationException("after free");
        var record = new NativeHandle(IntPtr.Zero, HandleKind.Env, () => throw failure);
        record.CloseOwner(0, true);
        Assert.True(record.WaitFreed(Deadline));
        Assert.False(NativeHandle.StrongTable.ContainsKey(record));
        Assert.Contains(failure, NativeHandle.CleanupFailures);
    }
    [Fact]
    public void ExhaustedGenerationRetiresInsteadOfWrapping()
    {
        var record = new NativeHandle(IntPtr.Zero, HandleKind.Env);
        typeof(NativeHandle).GetField("word", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(record, unchecked((long)uint.MaxValue << 32));
        Assert.False(record.TryAdopt(out _));
        Assert.True(record.WaitFreed(Deadline));
        Assert.Equal(uint.MaxValue, NativeHandle.Generation(record.Word));
    }
    [Fact]
    public void CallbackDisposalDefersBlockingShutdown()
    {
        using var release = new ManualResetEventSlim();
        using var returned = new ManualResetEventSlim();
        var server = new EnvServer(new BlockingCloseCounter(release));
        var callback = new Thread(() =>
        {
            CallbackContext.Enter();
            try { server.Dispose(); returned.Set(); }
            finally { CallbackContext.Exit(); }
        });
        callback.Start();
        try { Assert.True(returned.Wait(TimeSpan.FromSeconds(2))); }
        finally { release.Set(); Assert.True(callback.Join(Deadline)); }
        Assert.True(server.WaitFreed(Deadline));
    }
    private sealed class BlockingCloseCounter(ManualResetEventSlim release) : Counter, IClose
    {
        void IClose.Close() { release.Wait(Deadline); }
    }
    [Fact]
    public void OwnerTokenFailureReleasesServerBeforeEnvironmentBorrow()
    {
        var authored = new Counter(); NativeHandle? record = null;
        OwnerToken.BeforeCreate = value =>
        {
            if (value.Kind == HandleKind.EnvServer) { record = value; throw new OutOfMemoryException("test"); }
        };
        try { Assert.Throws<OutOfMemoryException>(() => new EnvServer(authored)); }
        finally { OwnerToken.BeforeCreate = null; }
        Assert.NotNull(record); Assert.True(record.WaitFreed(Deadline));
        Assert.Equal(1, authored.Closes);
        using var next = Env.For(authored);
    }

    [Fact]
    public void LocalTrampolinesRethrowEveryExceptionByIdentity()
    {
        Exception[] errors = {
            new RLMeshException("library"), new ProtocolException("protocol"), new EnvironmentException("env"),
            new RecoverableEnvironmentException("again"), new ValueError("value"), new TypeError("type"),
            new RuntimeError("runtime"), new OSError("os"), new ConnectionError("connect"), new TimeoutError("time"),
            new InvalidOperationException("managed"), new InvalidOperationException("bad\0\uD800text"), new ThrowingMessageException(),
        };
        foreach (Exception error in errors)
        {
            var authored = new Counter { Error = error }; using var env = Env.For(authored); using var session = new LocalSession(env);
            session.Reset();
            Exception actual = Assert.ThrowsAny<Exception>(() => session.Step());
            Assert.Same(error, actual); Assert.Contains(nameof(Counter.Step), actual.StackTrace);
        }
        Assert.Equal(0, CallbackContext.ActiveFrames);
    }
    private sealed class ThrowingMessageException : Exception { public override string Message => throw new InvalidOperationException("getter failed"); }

    [Fact]
    public void PythonDrivesServedEnvDuringGcAndSeesWireErrorClasses()
    {
        var authored = new Counter();
        using var server = new EnvServer(authored); server.Start();
        using var stop = new CancellationTokenSource();
        var collector = new Thread(() => { while (!stop.IsCancellationRequested) { Collect(); Thread.Sleep(2); } });
        collector.Start();
        try
        {
            Python(server.Address, "normal");
            Assert.True(authored.Steps >= 2); Assert.True(authored.Resets >= 3);
            Assert.NotEmpty(authored.Threads);
        }
        finally { stop.Cancel(); Assert.True(collector.Join(Deadline)); }
        server.Shutdown(); Assert.True(server.Wait(TimeSpan.Zero));
        // A builtin class names the scenario: Python raises it too.
        (Exception error, string text, string scenario)[] failures = {
            (new RLMeshException("library"), "library", "plain"),
            (new ProtocolException("protocol"), "protocol", "plain"),
            (new EnvironmentException("env"), "env", "plain"),
            (new RecoverableEnvironmentException("again"), "again", "recoverable"),
            (new ValueError("value"), "value", "ValueError"),
            (new TypeError("type"), "type", "TypeError"),
            (new RuntimeError("runtime"), "runtime", "RuntimeError"),
            (new OSError("os"), "os", "OSError"),
            (new ConnectionError("connect"), "connect", "ConnectionError"),
            (new TimeoutError("time"), "time", "TimeoutError"),
            (new InvalidOperationException("managed"), "managed", "plain"),
            (new ValueError("bad\0\uD800text"), "bad\uFFFD\uFFFDtext", "ValueError"),
            (new ThrowingMessageException(), "ThrowingMessageException: message unavailable", "plain"),
        };
        foreach (var failure in failures)
        {
            using var faulty = new EnvServer(new Counter { Error = failure.error }); faulty.Start();
            Python(faulty.Address, failure.scenario, failure.text);
        }
    }
    private sealed class ImageStateEnv : IEnv
    {
        public Space ObservationSpace { get; } = new Dict(new KeyValuePair<string, Space>[]
        {
            new("state", new Box(0, 10, new long[] { 2 }, new() { Dtype = DType.Uint8 })),
            new("image", new Box(0, 255, new long[] { 2, 2, 1 }, new() { Dtype = DType.Uint8 })),
        });
        public Space ActionSpace { get; } = new Discrete(2);
        private static Value Observation() => Value.Dict(new Metadata
        {
            ["image"] = new Tensor(new byte[] { 1, 2, 3, 4 }, new long[] { 2, 2, 1 }, DType.Uint8),
            ["state"] = new Tensor(new byte[] { 5, 6 }, new long[] { 2 }, DType.Uint8),
        });
        public ResetResult Reset(long? seed, Metadata? options) => new(Observation(), new() { ["z"] = 9, ["a"] = 8 });
        public StepResult Step(Value? action) => new(Observation(), 1, true, false, new() { ["z"] = 9, ["a"] = 8 });
    }
    [Fact]
    public void DictObservationsAndOrderedValuesRoundTripThroughPython()
    {
        var authored = new ImageStateEnv();
        var dict = Assert.IsType<Dict>(authored.ObservationSpace);
        Assert.Equal(2, dict.Count);
        Assert.Equal(new[] { "image", "state" }, dict.Keys);
        Assert.Equal(dict.Keys, dict.Select(pair => pair.Key));
        Assert.Equal(dict.Values, dict.Select(pair => pair.Value));
        Assert.True(dict.ContainsKey("image"));
        Assert.True(dict.TryGetValue("image", out var image));
        Assert.Same(dict["image"], image); Assert.IsType<Box>(image);
        Assert.False(dict.TryGetValue("absent", out _));
        Assert.Throws<KeyNotFoundException>(() => dict["absent"]);
        Assert.Throws<ValueError>(() => new Dict(new[] { new KeyValuePair<string, Space>("x", image), new KeyValuePair<string, Space>("x", image) }));
        Value ordered = Value.Dict(new Metadata { ["z"] = 1, ["a"] = 2 });
        Assert.Equal(new[] { "z", "a" }, ordered.AsDict().Keys);
        using var server = new EnvServer(authored); server.Start();
        Python(server.Address, "dict");
    }
    private static void Python(string address, string scenario, string expected = "")
    {
        ProcessResult result = Process("uv", new[] { "run", "--no-sync", "python", Path.Combine(Root, "csharp/tests/python_client.py"), address, scenario, expected });
        Assert.True(result.Code == 0, result.Output);
    }
    [Fact]
    public void BindFailuresReleaseLibraryOwnerAndOptionsAreValidated()
    {
        // A v1 port in use fails the bind at construction.
        using var live = new EnvServer(new Counter(), new() { Address = "tcp://127.0.0.1:0" });
        var authored = new Counter();
        Assert.Throws<ConnectionError>(() => new EnvServer(authored, new() { Address = live.Address }));
        using var recovered = Env.For(authored);
        Assert.Throws<ValueError>(() => new EnvServer(new Counter(), new() { Address = "tcp://127.0.0.1:0\0bad" }));
        Assert.Throws<ValueError>(() => new EnvServer(new Counter(), new() { Ext = new() { HostRuntimeJson = "\uD800" } }));
        using var valid = new EnvServer(new Counter());
        Assert.Throws<ValueError>(() => valid.Wait(TimeSpan.FromTicks(-1)));
        valid.Dispose(); Assert.Throws<ObjectDisposedException>(() => valid.Address);
        Assert.Throws<ObjectDisposedException>(() => valid.Start());
    }
    [Fact]
    public void ClosePolicyAndNoRenderAndDefaultHostRuntimeAreCorrect()
    {
        var authored = new Counter();
        using (var server = new EnvServer(authored, new() { CloseEnvOnShutdown = false })) { server.Start(); server.Shutdown(); }
        Assert.Equal(0, authored.Closes);
        var json = EnvServer.HostRuntime(); Assert.Contains("\"language\":\"csharp\"", json); Assert.Contains("framework_versions", json); Assert.DoesNotContain("toolchain", json);
        Assert.Equal(IntPtr.Zero, Callbacks.EnvVtable(true).render);
        Assert.NotEqual(IntPtr.Zero, Callbacks.EnvVtable(true).close);
        Assert.Equal(IntPtr.Zero, Callbacks.EnvVtable(false).close);
        using var described = new EnvServer(new Counter());
        using var call = described.Record.Acquire(described);
        RLMeshBytes text = default;
        Status.Check(Native.rlmesh_ext_env_server_describe_json(call.Server, &text));
        using var document = JsonDocument.Parse(Utf8.OwnedText(text));
        var runtime = document.RootElement.GetProperty("runtime");
        Assert.Equal("csharp", runtime.GetProperty("language").GetString());
        Assert.True(runtime.GetProperty("framework_versions").TryGetProperty("dotnet", out _));
        Assert.False(runtime.TryGetProperty("toolchain", out _));
    }

    [Fact]
    public void RetiringForWaitsForNativeFreeAndNeverBuildsTwoLiveHandles()
    {
        var authored = new CountingSpaces();
        var (_, record) = UnreachableEnv(authored);
        using var entered = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        NativeHandle.BeforeFree = current => { if (ReferenceEquals(current, record)) { entered.Set(); release.Wait(Deadline); } };
        Env? replacement = null; Exception? failure = null;
        var thread = new Thread(() => { try { replacement = Env.For(authored); } catch (Exception error) { failure = error; } });
        try
        {
            GC.Collect(); GC.WaitForPendingFinalizers(); Assert.True(entered.Wait(Deadline));
            thread.Start(); Assert.False(thread.Join(TimeSpan.FromMilliseconds(30)));
            Assert.Equal(1, authored.SpaceReads);
            release.Set(); Assert.True(thread.Join(Deadline));
            Assert.Null(failure); Assert.NotNull(replacement);
            Assert.True(record.IsFreed); Assert.NotSame(record, replacement.Record);
            Assert.Equal(2, authored.SpaceReads);
        }
        finally { release.Set(); NativeHandle.BeforeFree = null; replacement?.Dispose(); }
    }
    private sealed class CountingSpaces : Counter
    {
        internal int SpaceReads;
        public override Space ObservationSpace { get { Interlocked.Increment(ref SpaceReads); return Obs; } }
    }
    [Fact]
    public void ConcurrentServedCallbacksAreRootedOnNativeThreads()
    {
        using var entered = new CountdownEvent(2); using var release = new ManualResetEventSlim();
        var a = new BlockingCounter(entered, release); var b = new BlockingCounter(entered, release);
        using var one = new EnvServer(a); using var two = new EnvServer(b); one.Start(); two.Start();
        var results = new ProcessResult[2];
        string first = one.Address, second = two.Address;
        var t1 = new Thread(() => results[0] = Process("uv", new[] { "run", "--no-sync", "python", Path.Combine(Root, "csharp/tests/python_client.py"), first, "normal" }));
        var t2 = new Thread(() => results[1] = Process("uv", new[] { "run", "--no-sync", "python", Path.Combine(Root, "csharp/tests/python_client.py"), second, "normal" }));
        t1.Start(); t2.Start();
        try
        {
            Assert.True(entered.Wait(Deadline)); Assert.True(CallbackContext.ActiveFrames >= 2);
            Collect();
        }
        finally { release.Set(); Assert.True(t1.Join(Deadline)); Assert.True(t2.Join(Deadline)); }
        Assert.All(results, result => Assert.True(result.Code == 0, result.Output));
    }
    private sealed class BlockingCounter(CountdownEvent entered, ManualResetEventSlim release) : Counter
    {
        private int first;
        public override StepResult Step(Value? action)
        {
            if (Interlocked.Exchange(ref first, 1) == 0) { entered.Signal(); release.Wait(Deadline); }
            return base.Step(action);
        }
    }
    [Fact]
    public void CleanupFailuresAreContainedAndPreserveTheLastError()
    {
        // Create a real last-error slot before entering a void cleanup callback.
        Value scalar = 1; using var call = scalar.Handle.Acquire(scalar);
        double output = 0;
        Assert.NotEqual(RLMeshStatus.RLMESH_OK, Native.rlmesh_value_as_float(call.Value, &output));
        string before = Utf8.Borrowed(Native.rlmesh_last_error_message());
        var data = GCHandle.Alloc(new object());
        var table = Callbacks.EnvVtable(false);
        var cleanup = Marshal.GetDelegateForFunctionPointer<Callbacks.FreeFn>(table.free_user_data);
        var expected = new InvalidOperationException("cleanup test failure");
        Callbacks.CleanupHook = pointer => { if (pointer == GCHandle.ToIntPtr(data)) throw expected; };
        try
        {
            cleanup(GCHandle.ToIntPtr(data));
            Assert.Equal(before, Utf8.Borrowed(Native.rlmesh_last_error_message()));
            Assert.Contains(expected, NativeHandle.CleanupFailures);
            Assert.Equal(0, CallbackContext.ActiveFrames);
            Assert.True(data.IsAllocated);
        }
        finally { Callbacks.CleanupHook = null; data.Free(); }
        while (NativeHandle.CleanupFailures.TryDequeue(out _)) { }
    }
    [Fact]
    public void EveryNativeLoadExceptionIsReportedWithLibraryAndCause()
    {
        foreach (var error in new Exception[] { new DllNotFoundException("missing"), new EntryPointNotFoundException("symbol"), new BadImageFormatException("architecture") })
        {
            Assert.True(NativeGate.IsLoadFailure(error));
            var wrapped = NativeGate.LoadError(error, "fixture");
            Assert.Contains("rlmesh_capi", wrapped.Message); Assert.Contains(error.Message, wrapped.Message); Assert.Same(error, wrapped.InnerException);
            wrapped = Assert.Throws<RuntimeError>(() => NativeGate.Continue((Action)(() => throw error), "worker fixture"));
            Assert.Contains("worker fixture", wrapped.Message); Assert.Same(error, wrapped.InnerException);
        }
    }

    [Fact]
    public void FreedServerReleasesItsManagedBorrowEvenIfWrapperStaysAlive()
    {
        var (server, weak, record) = BorrowedServer();
        server.Dispose(); Assert.True(server.WaitFreed(Deadline));
        Collect(); Assert.False(weak.TryGetTarget(out _)); Assert.True(record.WaitFreed(Deadline));
        GC.KeepAlive(server);
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (EnvServer, WeakReference<Env>, NativeHandle) BorrowedServer()
    {
        var env = Env.For(new Counter());
        return (new EnvServer(env), new WeakReference<Env>(env), env.Record);
    }
    [Fact]
    public void ExplicitDisposeOnAnOrdinaryThreadCompletesItsNativeFree()
    {
        using var env = Env.For(new Counter()); var server = new EnvServer(env);
        int freeThread = 0, callerThread = 0; bool freedBeforeReturn = false;
        NativeHandle.BeforeFree = record => { if (ReferenceEquals(record, server.Record)) freeThread = System.Environment.CurrentManagedThreadId; };
        try
        {
            var thread = new Thread(() => { callerThread = System.Environment.CurrentManagedThreadId; server.Dispose(); freedBeforeReturn = server.Record.IsFreed; });
            thread.Start(); Assert.True(thread.Join(Deadline));
            Assert.True(freedBeforeReturn); Assert.Equal(callerThread, freeThread);
        }
        finally { NativeHandle.BeforeFree = null; server.Dispose(); }
    }

    [Fact]
    public void PublicEnumsMatchTheNativeVocabulary()
    {
        foreach (DType value in Enum.GetValues<DType>())
            Assert.Equal((uint)Enum.Parse<RLMeshDType>("RLMESH_DTYPE_" + value.ToString().ToUpperInvariant()), (uint)value);
        foreach (ValueKind value in Enum.GetValues<ValueKind>())
            Assert.Equal((uint)Enum.Parse<RLMeshValueKind>("RLMESH_VALUE_" + value.ToString().ToUpperInvariant()), (uint)value);
        Assert.Equal(12, Enum.GetValues<DType>().Length); Assert.Equal(10, Enum.GetValues<ValueKind>().Length);
    }

    private sealed class LocalSession : IDisposable
    {
        private readonly HandleLease borrow;
        private readonly RLMeshSession* session;
        internal LocalSession(Env env)
        {
            borrow = env.Record.Acquire(env);
            var options = new RLMeshSessionOptions { struct_size = (UIntPtr)sizeof(RLMeshSessionOptions), execution_horizon = 1 };
            RLMeshSession* output = null;
            try
            {
                Status.Check(Native.rlmesh_session_new(new RLMeshModelTarget { kind = RLMeshModelTargetKind.RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE }, Native.rlmesh_env_target(borrow.Env), &options, &output));
                session = output;
            }
            catch { borrow.Dispose(); throw; }
        }
        internal void Reset()
        {
            RLMeshValue* obs = null; RLMeshValue* info = null;
            try { Status.Check(Native.rlmesh_session_reset(session, null, null, &obs, &info)); }
            finally { Native.rlmesh_value_free(obs); Native.rlmesh_value_free(info); }
        }
        internal void Step()
        {
            Value action = 0; using var call = action.Handle.Acquire(action);
            RLMeshValue* obs = null; RLMeshValue* info = null; double reward = 0; bool terminated = false, truncated = false;
            try { Status.Check(Native.rlmesh_session_step(session, call.Value, &obs, &reward, &terminated, &truncated, &info)); }
            finally { Native.rlmesh_value_free(obs); Native.rlmesh_value_free(info); }
        }
        public void Dispose() { Native.rlmesh_session_free(session); borrow.Dispose(); }
    }
    private static void Collect() { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); }
    internal readonly record struct ProcessResult(int Code, string Output);
    internal static ProcessResult Process(string command, IEnumerable<string> args, string? directory = null)
    {
        var start = new ProcessStartInfo(command) { WorkingDirectory = directory ?? Root, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (string arg in args) start.ArgumentList.Add(arg);
        using var process = System.Diagnostics.Process.Start(start)!;
        var output = process.StandardOutput.ReadToEndAsync(); var error = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(120000)) { process.Kill(true); throw new InvalidOperationException("test process timed out: " + command); }
        return new(process.ExitCode, output.GetAwaiter().GetResult() + error.GetAwaiter().GetResult());
    }
}
