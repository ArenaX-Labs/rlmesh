using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using RLMesh.Interop;

namespace RLMesh.Internal;

internal static class HandleMetrics
{
    internal static long FinalizableConstructions;
    internal static long LiveHandles;
}

internal enum HandleKind { Value, Tensor, Space, Env, EnvServer }
internal enum OwnershipState { Idle, InUse, Transferring, Transferred, Closing, Freeing, Freed }

// Inert records are their own SafeHandle. Strong records have no finalizer;
// their finalizable owner token is reachable only from the wrapper.
internal sealed unsafe class NativeHandle : SafeHandle
{
    private long word;
    private readonly bool strong;
    private Action? afterFree;
    internal readonly HandleKind Kind;
    internal string? WaitingIn;
    internal NativeHandle? HostedBy;
    // Completed only after server_free returns, which joins native actor and
    // abandoned close threads. Env free alone is never a thread join.
    internal readonly NativeReturnScope JoinedNativeReturn = new(attached: false);
    internal CleanupRecord? Cleanup;
    private ManualResetEventSlim? completion;
    internal bool IsFreed => State(Word) == OwnershipState.Freed;
    private ManualResetEventSlim Completion
    {
        get
        {
            var current = Volatile.Read(ref completion);
            if (current == null)
            {
                var created = new ManualResetEventSlim(false);
                current = Interlocked.CompareExchange(ref completion, created, null) ?? created;
                if (!ReferenceEquals(current, created)) created.Dispose();
            }
            if (IsFreed) current.Set();
            return current;
        }
    }
    internal static readonly ConcurrentDictionary<NativeHandle, byte> StrongTable = new();
    internal static readonly ConcurrentQueue<Exception> CleanupFailures = new();
    internal static Action<NativeHandle>? BeforeFree;
    internal static Action<OwnerToken>? BeforeFinalize;
    internal static Action<NativeHandle>? BeforePublication;

    internal NativeHandle(IntPtr pointer, HandleKind kind, Action? afterFree = null, bool finalizable = true) : base(IntPtr.Zero, finalizable && kind is not (HandleKind.Env or HandleKind.EnvServer))
    {
        if (finalizable && kind is not (HandleKind.Env or HandleKind.EnvServer))
            Interlocked.Increment(ref HandleMetrics.FinalizableConstructions);
        Interlocked.Increment(ref HandleMetrics.LiveHandles);
        SetHandle(pointer);
        Kind = kind;
        strong = kind is HandleKind.Env or HandleKind.EnvServer;
        this.afterFree = afterFree;
        if (strong) Admission.Publish(this);
    }
    internal void Publish()
    {
        BeforePublication?.Invoke(this);
        StrongTable.TryAdd(this, 0);
        // A failing constructor may already have freed it, including racing a
        // finalizer. Pair this check with Destroy's removal after native return.
        if (IsFreed) StrongTable.TryRemove(this, out _);
    }
    public override bool IsInvalid => handle == IntPtr.Zero;
    internal long Word => Volatile.Read(ref word);
    internal static OwnershipState State(long value) => (OwnershipState)(value & 7);
    internal static int Count(long value) => (int)((value >> 3) & 0x1fffffff);
    internal static uint Generation(long value) => (uint)((ulong)value >> 32);
    private static long Pack(OwnershipState state, int count, uint generation) => unchecked((long)((ulong)generation << 32)) | ((long)count << 3) | (long)state;
    private bool Change(long before, OwnershipState state, int count, uint generation) => Interlocked.CompareExchange(ref word, Pack(state, count, generation), before) == before;

    internal HandleLease Acquire(object? keepAlive = null, bool teardown = false)
    {
        NativeGate.Ensure();
        while (true)
        {
            long value = Word;
            if (State(value) is not (OwnershipState.Idle or OwnershipState.InUse) && !(teardown && State(value) == OwnershipState.Closing)) throw new ObjectDisposedException(Kind.ToString());
            int count = Count(value);
            if (count == 0x1fffffff) throw new RuntimeError("too many outstanding native references");
            if (!Change(value, State(value) == OwnershipState.Closing ? OwnershipState.Closing : OwnershipState.InUse, count + 1, Generation(value))) continue;
            bool added = false;
            try
            {
                DangerousAddRef(ref added);
                return new HandleLease(this, keepAlive);
            }
            catch
            {
                if (added) DangerousRelease();
                ReleaseReference();
                throw;
            }
        }
    }
    internal void Release()
    {
        try { ReleaseReference(); }
        finally { DangerousRelease(); }
    }
    private void ReleaseReference()
    {
        while (true)
        {
            long value = Word;
            var state = State(value);
            int count = Count(value);
            if (state is not (OwnershipState.InUse or OwnershipState.Closing) || count == 0)
                throw new InvalidOperationException("unbalanced native reference");
            var next = count > 1 ? state : state == OwnershipState.Closing ? OwnershipState.Freeing : OwnershipState.Idle;
            if (!Change(value, next, count - 1, Generation(value))) continue;
            if (next == OwnershipState.Freeing) ScheduleFree(false);
            return;
        }
    }
    internal bool TryAdopt(out uint generation)
    {
        while (true)
        {
            long value = Word;
            generation = Generation(value);
            if (State(value) is not (OwnershipState.Idle or OwnershipState.InUse)) return false;
            // Never wrap: an ancient token must not become current again.
            if (generation == uint.MaxValue) { CloseOwner(generation, false); return false; }
            generation++;
            if (Change(value, State(value), Count(value), generation)) return true;
        }
    }
    internal bool TryReserveTransfer()
    {
        long value = Word;
        return State(value) == OwnershipState.Idle && Change(value, OwnershipState.Transferring, 0, Generation(value));
    }
    internal void FinishTransfer(bool committed)
    {
        long value = Word;
        if (State(value) != OwnershipState.Transferring || !Change(value, committed ? OwnershipState.Transferred : OwnershipState.Idle, 0, Generation(value)))
            throw new InvalidOperationException("invalid transfer transition");
        if (committed) { Interlocked.Decrement(ref HandleMetrics.LiveHandles); SetHandleAsInvalid(); }
    }
    internal void CloseOwner(uint generation, bool explicitDispose, bool anyGeneration = false)
    {
        while (true)
        {
            long value = Word;
            if (!anyGeneration && generation != Generation(value)) return;
            if (State(value) is not (OwnershipState.Idle or OwnershipState.InUse)) return;
            var next = Count(value) == 0 ? OwnershipState.Freeing : OwnershipState.Closing;
            if (!Change(value, next, Count(value), Generation(value))) continue;
            if (next == OwnershipState.Freeing) ScheduleFree(explicitDispose);
            return;
        }
    }
    private void ScheduleFree(bool explicitDispose)
    {
        if (!strong || (explicitDispose && CallbackContext.Depth == 0 && !CallbackContext.LibraryThread)) Destroy();
        else
        {
            var cleanup = new CleanupRecord(Destroy);
            Volatile.Write(ref Cleanup, cleanup);
            CleanupWorkers.Enqueue(cleanup);
        }
    }
    private void Destroy()
    {
        using var nativeReturn = new NativeReturnScope();
        try
        {
            Volatile.Write(ref WaitingIn, Kind == HandleKind.EnvServer ? "rlmesh_env_server_free" : "rlmesh_env_free");
            BeforeFree?.Invoke(this);
            switch (Kind)
            {
                case HandleKind.Value: Native.rlmesh_value_free((RLMeshValue*)handle); break;
                case HandleKind.Tensor: Native.rlmesh_tensor_free((RLMeshTensor*)handle); break;
                case HandleKind.Space: Native.rlmesh_space_free((RLMeshSpace*)handle); break;
                case HandleKind.Env: Native.rlmesh_env_free((RLMeshEnv*)handle); break;
                case HandleKind.EnvServer: Native.rlmesh_env_server_free((RLMeshEnvServer*)handle); JoinedNativeReturn.CompleteJoin(); break;
            }
            // Native free has returned. Managed cleanup diagnostics must never
            // strand the record (or its registry tombstone) in Freeing.
            try { Interlocked.Exchange(ref afterFree, null)?.Invoke(); }
            finally
            {
                Interlocked.Decrement(ref HandleMetrics.LiveHandles);
                SetHandleAsInvalid();
                long value = Word;
                if (!Change(value, OwnershipState.Freed, 0, Generation(value)))
                    CleanupFailures.Enqueue(new InvalidOperationException("invalid free transition"));
                StrongTable.TryRemove(this, out _);
                Volatile.Write(ref WaitingIn, null);
                Volatile.Read(ref completion)?.Set();
            }
        }
        catch (Exception e) { CleanupFailures.Enqueue(e); }
    }
    internal bool WaitFreed(TimeSpan timeout)
    {
        if (timeout < TimeSpan.Zero) throw new ValueError("timeout must be nonnegative");
        if (CallbackContext.Depth > 0) return IsFreed;
        var deadline = System.Diagnostics.Stopwatch.StartNew();
        while (!IsFreed)
        {
            var remaining = timeout - deadline.Elapsed;
            if (remaining <= TimeSpan.Zero) return IsFreed;
            Completion.Wait(remaining > TimeSpan.FromSeconds(1) ? TimeSpan.FromSeconds(1) : remaining);
        }
        return true;
    }
    protected override bool ReleaseHandle()
    {
        try { CloseOwner(0, false); }
        catch (Exception e) { CleanupFailures.Enqueue(e); }
        return true;
    }
}

internal readonly unsafe struct HandleLease : IDisposable
{
    private readonly NativeHandle record;
    private readonly object? keepAlive;
    internal HandleLease(NativeHandle record, object? keepAlive) { this.record = record; this.keepAlive = keepAlive; }
    internal IntPtr Pointer => record.DangerousGetHandle();
    internal RLMeshValue* Value => (RLMeshValue*)Pointer;
    internal RLMeshTensor* Tensor => (RLMeshTensor*)Pointer;
    internal RLMeshSpace* Space => (RLMeshSpace*)Pointer;
    internal RLMeshEnv* Env => (RLMeshEnv*)Pointer;
    internal RLMeshEnvServer* Server => (RLMeshEnvServer*)Pointer;
    public void Dispose()
    {
        try { record.Release(); }
        finally { GC.KeepAlive(keepAlive); }
    }
}

internal sealed class OwnerToken : SafeHandle
{
    internal readonly NativeHandle Record;
    internal readonly uint Generation;
    internal static Action<NativeHandle>? BeforeCreate;
    internal OwnerToken(NativeHandle record, uint generation) : base(IntPtr.Zero, true)
    {
        Interlocked.Increment(ref HandleMetrics.FinalizableConstructions);
        Record = record;
        Generation = generation;
        BeforeCreate?.Invoke(record);
        SetHandle(new IntPtr(1));
    }
    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void CloseExplicit()
    {
        try { Record.CloseOwner(Generation, true); }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
        finally { SetHandleAsInvalid(); Dispose(); }
    }
    protected override bool ReleaseHandle()
    {
        try
        {
            NativeHandle.BeforeFinalize?.Invoke(this);
            Record.CloseOwner(Generation, false);
        }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
        return true;
    }
}

internal static class CallbackContext
{
    [ThreadStatic] internal static int Depth;
    [ThreadStatic] private static bool libraryThread;
    [ThreadStatic] internal static NativeHandle? HostedBy;
    internal static bool LibraryThread
    {
        get => libraryThread;
        set => libraryThread = value;
    }
    internal static long ActiveFrames;
    internal static long EnteredFrames;
    internal static void Enter() { Interlocked.Increment(ref ActiveFrames); Interlocked.Increment(ref EnteredFrames); Depth++; }
    internal static void Exit() { Depth--; Interlocked.Decrement(ref ActiveFrames); }
}

internal sealed class CleanupRecord
{
    private int claim;
    private Action? free;
    internal CleanupRecord(Action free) { this.free = free; }
    internal bool TryRun()
    {
        if (Interlocked.CompareExchange(ref claim, 1, 0) != 0) return false;
        try { Interlocked.Exchange(ref free, null)!(); }
        finally { Volatile.Write(ref claim, 2); CleanupWorkers.Records.TryRemove(this, out _); }
        return true;
    }
    internal int Claim => Volatile.Read(ref claim);
}

internal static class CleanupWorkers
{
    internal static readonly ConcurrentDictionary<CleanupRecord, byte> Records = new();
    internal static Func<object, bool>? WaitForWork;
    private static readonly TrackedPool Pool = new("RLMesh cleanup", sync => WaitForWork?.Invoke(sync) ?? Monitor.Wait(sync, TimeSpan.FromSeconds(30)));
    internal static void Enqueue(CleanupRecord record)
    {
        Records.TryAdd(record, 0);
        Pool.Enqueue(() =>
        {
            record.TryRun();
            if (record.Claim == 2) Records.TryRemove(record, out _);
        });
    }
    internal static void StopAndJoin() => Pool.StopAndJoin();
    internal static int Workers => Pool.Workers;
}
