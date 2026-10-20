using System;
using System.Collections.Concurrent;
using System.Threading;
using RLMesh.Interop;

namespace RLMesh.Internal;

// One native reference per operation. No ambient token and no reuse between
// calls. Registration disposal joins Cancel before the native reference dies.
internal sealed unsafe class CancellationBridge : IDisposable
{
    internal unsafe delegate T NativeOperation<T>(RLMeshExtCancelToken* token);
    private static readonly ConcurrentDictionary<CancellationBridge, byte> Active = new();
    private readonly object sync = new();
    private RLMeshExtCancelToken* token;
    private CancellationTokenRegistration registration;
    internal static Action<string>? Trace;
    internal static Action? BeforeCancel;
    internal static Action? BeforeCall;
    private CancellationBridge(CancellationToken cancellationToken)
    {
        NativeGate.Ensure();
        RLMeshExtCancelToken* created = null;
        Status.Check(Native.rlmesh_ext_cancel_token_new(&created));
        token = created;
        try
        {
            Active.TryAdd(this, 0);
            registration = cancellationToken.Register(CancelRegistered, this);
            // Admission can precede publication of this token. Pair this check
            // with CancelAll's snapshot so shutdown cannot miss that interval.
            if (!Admission.Open) Cancel();
        }
        catch { Active.TryRemove(this, out _); Native.rlmesh_ext_cancel_token_free(token); token = null; throw; }
    }
    private static void CancelRegistered(object? state)
    {
        try { ((CancellationBridge)state!).Cancel(); }
        catch { /* No failure may escape onto the cancelling caller's thread. */ }
    }
    private void Cancel()
    {
        try
        {
            lock (sync)
            {
                if (token == null) return;
                BeforeCancel?.Invoke();
                NativeGate.Continue(() => Status.Check(Native.rlmesh_ext_cancel_token_cancel(token)), "rlmesh_ext_cancel_token_cancel");
                Trace?.Invoke("cancel");
            }
        }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
    }
    internal static void CancelAll() { foreach (var bridge in Active.Keys) bridge.Cancel(); }
    internal static int Count => Active.Count;
    internal static T Run<T>(CancellationToken cancellationToken, NativeOperation<T> call)
    {
        if (cancellationToken.IsCancellationRequested) throw new Ext.CancelledError(cancellationToken);
        using var operation = Admission.Enter("cancellation bridge");
        using var bridge = new CancellationBridge(cancellationToken);
        try { BeforeCall?.Invoke(); return NativeGate.Continue(() => call(bridge.token), "cancellation bridge"); }
        finally { Trace?.Invoke("call returned"); }
    }
    public void Dispose()
    {
        registration.Dispose();
        Trace?.Invoke("registration disposed");
        lock (sync)
        {
            if (token == null) return;
            NativeGate.Continue(() => Native.rlmesh_ext_cancel_token_free(token), "rlmesh_ext_cancel_token_free");
            token = null;
            Active.TryRemove(this, out _);
            Trace?.Invoke("token freed");
        }
    }
}
