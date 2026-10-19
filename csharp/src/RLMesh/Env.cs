using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh;

/// <summary>Experimental. The unique native handle for an authored environment.</summary>
public sealed unsafe class Env : IDisposable
{
    internal readonly NativeHandle Record;
    private readonly OwnerToken owner;
    private Env(NativeHandle record, uint generation) { Record = record; owner = new(record, generation); }
    private static readonly object Sync = new();
    private static readonly Dictionary<IEnv, Entry> Registry = new(new IdentityComparer());
    private sealed class IdentityComparer : IEqualityComparer<IEnv>
    {
        public bool Equals(IEnv? x, IEnv? y) => ReferenceEquals(x, y);
        public int GetHashCode(IEnv obj) => RuntimeHelpers.GetHashCode(obj);
    }
    internal sealed class Entry
    {
        internal readonly int BuildingThread = System.Environment.CurrentManagedThreadId;
        internal readonly ManualResetEventSlim Built = new(false);
        internal NativeHandle? Record;
        internal WeakReference<Env>? Wrapper;
        internal bool LibraryOwned;
        internal bool RootReleased;
    }
    internal sealed class Root
    {
        internal readonly IEnv Authored;
        internal readonly Entry Entry;
        internal Root(IEnv authored, Entry entry) { Authored = authored; Entry = entry; }
    }
    public static Env For(IEnv env) => Resolve(env, false, out _);
    internal static Env ForServer(IEnv env, out bool owns) => Resolve(env, true, out owns);
    private static Env Resolve(IEnv env, bool server, out bool owns)
    {
        if (env == null) throw new ArgumentNullException(nameof(env));
        NativeGate.Ensure();
        owns = false;
        while (true)
        {
            Entry entry;
            NativeHandle? retiring = null;
            bool build = false;
            lock (Sync)
            {
                if (!Registry.TryGetValue(env, out entry!))
                {
                    entry = new Entry();
                    Registry.Add(env, entry);
                    build = true;
                }
                else if (entry.Record == null)
                {
                    if (entry.BuildingThread == System.Environment.CurrentManagedThreadId)
                        throw new RuntimeError("Env.For was re-entered from the environment's space getter");
                }
                else
                {
                    var record = entry.Record;
                    var state = NativeHandle.State(record.Word);
                    if (state is OwnershipState.Idle or OwnershipState.InUse)
                    {
                        if (entry.LibraryOwned) throw new RuntimeError("the library is still using this environment's handle");
                        if (entry.Wrapper != null && entry.Wrapper.TryGetTarget(out var current)) return current;
                        if (record.TryAdopt(out uint generation))
                        {
                            var adopted = new Env(record, generation);
                            entry.LibraryOwned = server;
                            entry.Wrapper = server ? null : new WeakReference<Env>(adopted);
                            owns = server;
                            return adopted;
                        }
                    }
                    state = NativeHandle.State(record.Word);
                    if (state == OwnershipState.Closing || state is OwnershipState.Transferred or OwnershipState.Transferring)
                        throw new RuntimeError("the library is still using this environment's handle");
                    retiring = record;
                }
            }
            if (build)
            {
                try
                {
                    Env wrapper = Build(env, entry);
                    lock (Sync)
                    {
                        entry.Record = wrapper.Record;
                        entry.LibraryOwned = server;
                        entry.Wrapper = server ? null : new WeakReference<Env>(wrapper);
                    }
                    entry.Built.Set();
                    owns = server;
                    return wrapper;
                }
                catch
                {
                    lock (Sync) { if (Registry.TryGetValue(env, out var current) && ReferenceEquals(current, entry)) Registry.Remove(env); }
                    entry.Built.Set();
                    throw;
                }
            }
            if (retiring != null)
            {
                if (CallbackContext.Depth == 0 && !CallbackContext.LibraryThread) Volatile.Read(ref retiring.Cleanup)?.TryRun();
                // There are no borrowers in Freeing: native env free runs no authored close.
                // No registry lock is held while waiting for its cleanup worker.
                while (!retiring.WaitFreed(TimeSpan.FromSeconds(1)))
                    if (CallbackContext.Depth > 0) Thread.Yield();
            }
            else entry.Built.Wait();
        }
    }
    private static Env Build(IEnv env, Entry entry) => NativeGate.Call(() =>
    {
        var observation = env.ObservationSpace ?? throw new TypeError("ObservationSpace cannot be null");
        var action = env.ActionSpace ?? throw new TypeError("ActionSpace cannot be null");
        using var obs = observation.Handle.Acquire(observation);
        using var act = action.Handle.Acquire(action);
        var attributes = Initializers.EnvAttributes();
        attributes.observation_space = obs.Space;
        attributes.action_space = act.Space;
        var vtable = Callbacks.EnvVtable(env is IClose);
        var root = new Root(env, entry);
        var data = GCHandle.Alloc(root);
        RLMeshEnv* output = null;
        try
        {
            Status.Check(Native.rlmesh_env_new(&vtable, (void*)GCHandle.ToIntPtr(data), &attributes, &output));
        }
        catch { data.Free(); throw; }
        var record = new NativeHandle((IntPtr)output, HandleKind.Env, () => Retire(env, entry));
        return new Env(record, 0);
    });
    internal static void RootReleased(Root root)
    {
        lock (Sync)
        {
            // Keep a retiring entry until rlmesh_env_free has actually returned;
            // otherwise another For could create a handle in the free callback.
            root.Entry.RootReleased = true;
        }
    }
    private static void Retire(IEnv env, Entry entry)
    {
        lock (Sync)
        {
            try
            {
                if (!entry.RootReleased)
                    NativeHandle.CleanupFailures.Enqueue(new InvalidOperationException("environment root was not released by native free"));
            }
            finally
            {
                if (Registry.TryGetValue(env, out var current) && ReferenceEquals(current, entry)) Registry.Remove(env);
            }
        }
    }
    public void Dispose()
    {
        try { owner.CloseExplicit(); }
        finally { GC.KeepAlive(this); }
    }
    internal bool WaitFreed(TimeSpan timeout) => Record.WaitFreed(timeout);
}
