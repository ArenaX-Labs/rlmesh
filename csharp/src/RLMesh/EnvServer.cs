using System;
using System.Runtime.InteropServices;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh;

/// <summary>Experimental. Options for serving one authored environment.</summary>
public sealed class EnvServerOptions
{
    public string? Address { get; set; }
    public bool CloseEnvOnShutdown { get; set; } = true;
    public Ext.EnvServerOptions? Ext { get; set; }
}

/// <summary>Experimental. Binds at construction and serves on native library threads.</summary>
public sealed unsafe partial class EnvServer : IDisposable
{
    internal readonly NativeHandle Record;
    private readonly OwnerToken owner;
    public EnvServer(Env env, EnvServerOptions? options = null)
    {
        using var operation = Admission.Enter("EnvServer constructor");
        (Record, owner) = Create(env, options, false);
    }
    public EnvServer(IEnv env, EnvServerOptions? options = null)
    {
        using var operation = Admission.Enter("EnvServer constructor");
        Env wrapper = Env.ForServer(env, out bool owns);
        (Record, owner) = Create(wrapper, options, owns);
    }
    private static (NativeHandle, OwnerToken) Create(Env env, EnvServerOptions? options, bool owns)
    {
        NativeHandle? record = null;
        if (env == null) throw new ArgumentNullException(nameof(env));
        HandleLease borrow = default;
        bool acquired = false;
        try
        {
            record = NativeGate.Call(() =>
            {
                // Encode before acquiring; an invalid option must still release
                // the library-owned env through the constructor's outer catch.
                byte[] address = Utf8.Encode(options?.Address ?? "tcp://127.0.0.1:0", true);
                byte[] host = Utf8.Encode(options?.Ext?.HostRuntimeJson ?? HostRuntime(), true);
                borrow = env.Record.Acquire(env);
                acquired = true;
                RLMeshEnvServer* output = null;
                fixed (byte* addressData = address)
                fixed (byte* hostData = host)
                {
                    var ext = Initializers.ExtEnvServerOptions((sbyte*)hostData);
                    var init = Initializers.EnvServerOptions();
                    init.address = (sbyte*)addressData;
                    init.close_env_on_shutdown = (options?.CloseEnvOnShutdown ?? true) ? (byte)1 : (byte)0;
                    init.ext = &ext;
                    RLMeshEnv* nativeEnv = borrow.Env;
                    Status.Check(Native.rlmesh_env_server_new(&nativeEnv, (UIntPtr)1, &init, &output));
                }
                var created = new NativeHandle((IntPtr)output, HandleKind.EnvServer, () =>
                {
                    // The borrow and wrapper remain rooted through native free.
                    System.Threading.Interlocked.CompareExchange(ref env.Record.HostedBy, null, record);
                    borrow.Dispose();
                    if (owns) env.Dispose();
                });
                System.Threading.Volatile.Write(ref env.Record.HostedBy, created);
                return created;
            });
            return (record, new OwnerToken(record, 0));
        }
        catch
        {
            // Once the record owns the borrow, release the server first. This
            // also covers OwnerToken allocation failure after native creation.
            if (record != null) record.CloseOwner(0, true);
            else
            {
                if (acquired) borrow.Dispose();
                if (owns) env.Dispose();
            }
            throw;
        }
    }
    internal static string HostRuntime()
    {
        // FrameworkDescription is available on both targets and distinguishes Mono
        // without reflecting over the author's types or probing Type.GetType.
        string runtime = RuntimeInformation.FrameworkDescription.StartsWith("Mono", StringComparison.OrdinalIgnoreCase) ? "mono" : "dotnet";
        return "{\"language\":\"csharp\",\"framework_versions\":{\"" + runtime + "\":\"" + System.Environment.Version + "\"}}";
    }
    public string Address => NativeGate.Call(() =>
    {
        using var call = Record.Acquire(this);
        RLMeshBytes bytes = default; Status.Check(Native.rlmesh_env_server_address(call.Server, &bytes));
        return Utf8.OwnedText(bytes);
    });
    public void Start() => NativeGate.Call(() => { using var call = Record.Acquire(this); Status.Check(Native.rlmesh_env_server_start(call.Server)); });
    public bool Wait(TimeSpan? timeout = null) => NativeGate.Call(() =>
    {
        if (timeout < TimeSpan.Zero) throw new ValueError("timeout must be nonnegative");
        using var call = Record.Acquire(this);
        double seconds = timeout?.TotalSeconds ?? 0;
        bool stopped = false;
        Status.Check(Native.rlmesh_env_server_wait(call.Server, timeout.HasValue ? &seconds : null, &stopped));
        return stopped;
    });
    public void Shutdown() => NativeGate.Call(() => { using var call = Record.Acquire(this); Status.Check(Native.rlmesh_env_server_shutdown(call.Server)); });
    public void Dispose()
    {
        HandleLease call = default;
        bool acquired = false, deferred = false;
        try
        {
            // A blocking Wait holds a counted reference, so free cannot signal
            // shutdown until Wait returns. Protect that signal with its own lease.
            call = Record.Acquire(this); acquired = true;
            if (CallbackContext.Depth > 0 || CallbackContext.LibraryThread)
            {
                var cleanup = new CleanupRecord(() =>
                {
                    try { Status.Check(Native.rlmesh_env_server_shutdown(call.Server)); }
                    catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
                    finally { call.Dispose(); }
                });
                owner.CloseExplicit();
                CleanupWorkers.Enqueue(cleanup);
                deferred = true;
            }
            else Status.Check(Native.rlmesh_env_server_shutdown(call.Server));
        }
        catch (ObjectDisposedException) { }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
        finally
        {
            if (acquired && !deferred) call.Dispose();
            owner.CloseExplicit();
            GC.KeepAlive(this);
        }
    }
    internal bool WaitFreed(TimeSpan timeout) => Record.WaitFreed(timeout);
}
