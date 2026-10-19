using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using RLMesh.Interop;

namespace RLMesh.Internal;

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

    internal NativeHandle(IntPtr pointer, HandleKind kind, Action? afterFree = null, bool finalizable = true) : base(IntPtr.Zero, finalizable && kind is not (HandleKind.Env or HandleKind.EnvServer))
    {
        SetHandle(pointer);
        Kind = kind;
        strong = kind is HandleKind.Env or HandleKind.EnvServer;
        this.afterFree = afterFree;
        if (strong) StrongTable.TryAdd(this, 0);
    }
    public override bool IsInvalid => handle == IntPtr.Zero;
    internal long Word => Volatile.Read(ref word);
    internal static OwnershipState State(long value) => (OwnershipState)(value & 7);
    internal static int Count(long value) => (int)((value >> 3) & 0x1fffffff);
    internal static uint Generation(long value) => (uint)((ulong)value >> 32);
    private static long Pack(OwnershipState state, int count, uint generation) => unchecked((long)((ulong)generation << 32)) | ((long)count << 3) | (long)state;
    private bool Change(long before, OwnershipState state, int count, uint generation) => Interlocked.CompareExchange(ref word, Pack(state, count, generation), before) == before;

    internal HandleLease Acquire(object? keepAlive = null)
    {
        NativeGate.Ensure();
        while (true)
        {
            long value = Word;
            if (State(value) is not (OwnershipState.Idle or OwnershipState.InUse)) throw new ObjectDisposedException(Kind.ToString());
            int count = Count(value);
            if (count == 0x1fffffff) throw new RuntimeError("too many outstanding native references");
            if (!Change(value, OwnershipState.InUse, count + 1, Generation(value))) continue;
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
        if (committed) SetHandleAsInvalid();
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
        try
        {
            BeforeFree?.Invoke(this);
            switch (Kind)
            {
                case HandleKind.Value: Native.rlmesh_value_free((RLMeshValue*)handle); break;
                case HandleKind.Tensor: Native.rlmesh_tensor_free((RLMeshTensor*)handle); break;
                case HandleKind.Space: Native.rlmesh_space_free((RLMeshSpace*)handle); break;
                case HandleKind.Env: Native.rlmesh_env_free((RLMeshEnv*)handle); break;
                case HandleKind.EnvServer: Native.rlmesh_env_server_free((RLMeshEnvServer*)handle); break;
            }
            // Native free has returned. Managed cleanup diagnostics must never
            // strand the record (or its registry tombstone) in Freeing.
            try { Interlocked.Exchange(ref afterFree, null)?.Invoke(); }
            finally
            {
                SetHandleAsInvalid();
                long value = Word;
                if (!Change(value, OwnershipState.Freed, 0, Generation(value)))
                    CleanupFailures.Enqueue(new InvalidOperationException("invalid free transition"));
                StrongTable.TryRemove(this, out _);
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
    internal static bool LibraryThread
    {
        get => libraryThread;
        set => libraryThread = value;
    }
    internal static long ActiveFrames;
    internal static void Enter() { Interlocked.Increment(ref ActiveFrames); Depth++; }
    internal static void Exit() { Depth--; Interlocked.Decrement(ref ActiveFrames); }
}

internal sealed class CleanupRecord
{
    private int claim;
    private readonly Action free;
    internal CleanupRecord(Action free) { this.free = free; }
    internal bool TryRun()
    {
        if (Interlocked.CompareExchange(ref claim, 1, 0) != 0) return false;
        try { free(); }
        finally { Volatile.Write(ref claim, 2); }
        return true;
    }
    internal int Claim => Volatile.Read(ref claim);
}

internal static class CleanupWorkers
{
    private static readonly object Sync = new();
    private static readonly Queue<CleanupRecord> Queue = new();
    private static int idle;
    internal static Func<object, bool>? WaitForWork;
    internal static void Enqueue(CleanupRecord record)
    {
        lock (Sync)
        {
            Queue.Enqueue(record);
            if (idle == 0)
            {
                idle++;
                new Thread(Work) { IsBackground = true, Name = "RLMesh cleanup" }.Start();
            }
            Monitor.Pulse(Sync);
        }
    }
    private static void Work()
    {
        CallbackContext.LibraryThread = true;
        while (true)
        {
            CleanupRecord record;
            lock (Sync)
            {
                while (Queue.Count == 0)
                    if (!(WaitForWork?.Invoke(Sync) ?? Monitor.Wait(Sync, TimeSpan.FromSeconds(30))) && Queue.Count == 0)
                    { idle--; return; }
                idle--;
                record = Queue.Dequeue();
                // Ensure queued records can progress even if this free blocks.
                if (Queue.Count > 0 && idle == 0)
                {
                    idle++;
                    new Thread(Work) { IsBackground = true, Name = "RLMesh cleanup" }.Start();
                }
            }
            record.TryRun();
            lock (Sync) idle++;
        }
    }
}
