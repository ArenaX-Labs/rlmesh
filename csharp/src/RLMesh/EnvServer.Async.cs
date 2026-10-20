using System;
using System.Threading;
using System.Threading.Tasks;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh;

public sealed partial class EnvServer
{
    /// <summary>Experimental. Drain on a library worker; cancellation stops only the wait.</summary>
    public async Task ShutdownAsync(CancellationToken cancellationToken = default)
    {
        if (cancellationToken.IsCancellationRequested) throw new Ext.CancelledError(cancellationToken);
        var operation = Admission.Enter("EnvServer.ShutdownAsync", asynchronous: true);
        HandleLease call = default;
        bool acquired = false;
        var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            call = Record.Acquire(this); acquired = true;
            LibraryWorkers.Enqueue(() =>
            {
                Exception? failure = null;
                try { ShutdownNative(call); LibraryWorkers.BeforeAdoption?.Invoke(); }
                catch (Exception e) { failure = e; }
                finally
                {
                    try { call.Dispose(); }
                    catch (Exception e) { failure ??= e; }
                    finally { call = default; operation.Dispose(); }
                }
                if (failure == null) completion.TrySetResult(true);
                else completion.TrySetException(failure);
            });
        }
        catch
        {
            if (acquired) call.Dispose();
            operation.Dispose();
            throw;
        }
        var cancelled = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        using (cancellationToken.Register(() => { try { cancelled.TrySetResult(true); } catch { } }))
        {
            if (await Task.WhenAny(completion.Task, cancelled.Task).ConfigureAwait(false) == cancelled.Task)
            {
                // Observe a later native error even though the caller stopped waiting.
                _ = completion.Task.ContinueWith(t => { _ = t.Exception; }, CancellationToken.None,
                    TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
                throw new Ext.CancelledError(cancellationToken);
            }
            await completion.Task.ConfigureAwait(false);
        }
    }
    private static unsafe void ShutdownNative(HandleLease call) =>
        NativeGate.Continue(() => Status.Check(Native.rlmesh_env_server_shutdown(call.Server)), "rlmesh_env_server_shutdown");
}
