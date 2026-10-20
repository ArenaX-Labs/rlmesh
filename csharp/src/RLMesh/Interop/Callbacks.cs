using System;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using AOT;
using RLMesh.Internal;

namespace RLMesh.Interop;

internal static unsafe class Callbacks
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate RLMeshStatus ResetFn(IntPtr data, long* seed, RLMeshValue* options, RLMeshValue** observation, RLMeshValue** info);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate RLMeshStatus StepFn(IntPtr data, RLMeshValue* action, RLMeshValue** observation, double* reward, byte* terminated, byte* truncated, RLMeshValue** info);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate RLMeshStatus CloseFn(IntPtr data);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void FreeFn(IntPtr data);

    private static readonly ResetFn ResetDelegate = Reset;
    private static readonly StepFn StepDelegate = Step;
    private static readonly CloseFn CloseDelegate = Close;
    private static readonly FreeFn FreeEnvDelegate = FreeEnv;
    private static readonly FreeFn FreePayloadDelegate = FreePayload;
    internal static readonly IntPtr PayloadTag = Marshal.AllocHGlobal(1);
    private static readonly IntPtr FallbackText = AllocateFallback();
    private static readonly IntPtr FreePayloadPointer = Marshal.GetFunctionPointerForDelegate(FreePayloadDelegate);
    internal static Action<IntPtr>? CleanupHook;
    internal static void ReleaseResources()
    {
        // Called once, after every native holder and callback frame is joined.
        Marshal.FreeHGlobal(PayloadTag);
        Marshal.FreeHGlobal(FallbackText);
    }

    private static IntPtr AllocateFallback()
    {
        byte[] bytes = Encoding.UTF8.GetBytes("C# callback failed and its error could not be reported\0");
        IntPtr pointer = Marshal.AllocHGlobal(bytes.Length);
        Marshal.Copy(bytes, 0, pointer, bytes.Length);
        return pointer;
    }
    internal static RLMeshEnvVtable EnvVtable(bool close)
    {
        var table = Initializers.EnvVtable();
        table.reset = Marshal.GetFunctionPointerForDelegate(ResetDelegate);
        table.step = Marshal.GetFunctionPointerForDelegate(StepDelegate);
        if (close) table.close = Marshal.GetFunctionPointerForDelegate(CloseDelegate);
        table.free_user_data = Marshal.GetFunctionPointerForDelegate(FreeEnvDelegate);
        return table;
    }
    private static Env.Root Root(IntPtr data) => (Env.Root)GCHandle.FromIntPtr(data).Target!;
    private static NativeHandle? Host(Env.Root root)
    {
        var record = Volatile.Read(ref root.Entry.Record);
        return record == null ? null : Volatile.Read(ref record.HostedBy);
    }
    private static void Write(Value? observation, Metadata? info, RLMeshValue** outObservation, RLMeshValue** outInfo)
    {
        if (observation is null || info is null) throw new TypeError("environment returned a default result or null observation/info");
        // Copy outputs while holding both records. C owns any output already
        // written even if a subsequent conversion or the callback fails.
        using var obs = observation.Handle.Acquire(observation);
        if (info.Count == 0)
        {
            *outObservation = Native.rlmesh_value_clone(obs.Value);
            *outInfo = null; // The ABI defines NULL info as an empty dictionary.
            return;
        }
        Value metadata = Value.Dict(info);
        using var inf = metadata.Handle.Acquire(metadata);
        *outObservation = Native.rlmesh_value_clone(obs.Value);
        *outInfo = Native.rlmesh_value_clone(inf.Value);
    }
    [MonoPInvokeCallback(typeof(ResetFn))]
    private static RLMeshStatus Reset(IntPtr data, long* seed, RLMeshValue* options, RLMeshValue** observation, RLMeshValue** info)
    {
        NativeHandle? previousHost = null;
        bool entered = false;
        try
        {
            CallbackContext.Enter(); entered = true;
            previousHost = CallbackContext.HostedBy;
            *observation = null; *info = null;
            var root = Root(data);
            CallbackContext.HostedBy = Host(root);
            Value? input = options == null ? null : Value.Borrow(options);
            Metadata? metadata = input?.AsDict();
            var result = root.Authored.Reset(seed == null ? null : *seed, metadata);
            Write(result.Observation, result.Info, observation, info);
            GC.KeepAlive(input);
            return RLMeshStatus.RLMESH_OK;
        }
        catch (Exception e) { return Report(e); }
        finally { if (entered) { CallbackContext.HostedBy = previousHost; CallbackContext.Exit(); } }
    }
    [MonoPInvokeCallback(typeof(StepFn))]
    private static RLMeshStatus Step(IntPtr data, RLMeshValue* action, RLMeshValue** observation, double* reward, byte* terminated, byte* truncated, RLMeshValue** info)
    {
        NativeHandle? previousHost = null;
        bool entered = false;
        try
        {
            CallbackContext.Enter(); entered = true;
            previousHost = CallbackContext.HostedBy;
            *observation = null; *info = null;
            var root = Root(data);
            CallbackContext.HostedBy = Host(root);
            Value? input = action == null ? null : Value.Borrow(action);
            var result = root.Authored.Step(input);
            Write(result.Observation, result.Info, observation, info);
            *reward = result.Reward;
            *terminated = result.Terminated ? (byte)1 : (byte)0;
            *truncated = result.Truncated ? (byte)1 : (byte)0;
            GC.KeepAlive(input);
            return RLMeshStatus.RLMESH_OK;
        }
        catch (Exception e) { return Report(e); }
        finally { if (entered) { CallbackContext.HostedBy = previousHost; CallbackContext.Exit(); } }
    }
    [MonoPInvokeCallback(typeof(CloseFn))]
    private static RLMeshStatus Close(IntPtr data)
    {
        NativeHandle? previousHost = null;
        bool entered = false;
        try
        {
            CallbackContext.Enter(); entered = true;
            previousHost = CallbackContext.HostedBy;
            var root = Root(data);
            CallbackContext.HostedBy = Host(root);
            ((IClose)root.Authored).Close();
            return RLMeshStatus.RLMESH_OK;
        }
        catch (Exception e) { return Report(e); }
        finally { if (entered) { CallbackContext.HostedBy = previousHost; CallbackContext.Exit(); } }
    }
    [MonoPInvokeCallback(typeof(FreeFn))]
    private static void FreeEnv(IntPtr data)
    {
        bool entered = false;
        try
        {
            CallbackContext.Enter(); entered = true;
            CleanupHook?.Invoke(data);
            var handle = GCHandle.FromIntPtr(data);
            var root = (Env.Root)handle.Target!;
            Env.RootReleased(root);
            ManagedRoots.FreeFromNative(handle, root.Entry.Record);
        }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
        finally { if (entered) CallbackContext.Exit(); }
    }
    [MonoPInvokeCallback(typeof(FreeFn))]
    private static void FreePayload(IntPtr data)
    {
        bool entered = false;
        try
        {
            CallbackContext.Enter(); entered = true;
            CleanupHook?.Invoke(data);
            ManagedRoots.FreeFromNative(GCHandle.FromIntPtr(data));
        }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
        finally { if (entered) CallbackContext.Exit(); }
    }
    private static RLMeshStatus Report(Exception exception)
    {
        try
        {
            // Attach first so a text failure preserves the original exception.
            GCHandle payload = ManagedRoots.Allocate(ExceptionDispatchInfo.Capture(exception), "error payload", CallbackContext.HostedBy);
            try { Native.rlmesh_ext_callback_set_tagged_error_payload((void*)GCHandle.ToIntPtr(payload), FreePayloadPointer, (void*)PayloadTag); }
            catch { ManagedRoots.Free(payload); throw; }
            string text;
            try { text = exception is EnvironmentException environment ? environment.RawMessage : exception.Message; }
            catch { text = exception.GetType().Name + ": message unavailable"; }
            // Replacement UTF-8 repairs unpaired surrogates. Repair NUL first.
            byte[] bytes = Encoding.UTF8.GetBytes(text.Replace('\0', '\uFFFD') + "\0");
            fixed (byte* message = bytes)
            {
                var cls = Class(exception);
                if (cls == RLMeshErrorClass.RLMESH_ERROR_CLASS_INVALID)
                    Native.rlmesh_callback_set_error((sbyte*)message, 0);
                else Native.rlmesh_callback_set_error_class(cls, (sbyte*)message);
            }
        }
        catch
        {
            try { Native.rlmesh_callback_set_error((sbyte*)FallbackText, 0); }
            catch { /* A manually replaced library cannot unwind through C either. */ }
        }
        return RLMeshStatus.RLMESH_ERR_ENVIRONMENT;
    }
    private static RLMeshErrorClass Class(Exception e) => e switch
    {
        RecoverableEnvironmentException => RLMeshErrorClass.RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION,
        EnvironmentException => RLMeshErrorClass.RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION,
        ProtocolException => RLMeshErrorClass.RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION,
        RLMeshException => RLMeshErrorClass.RLMESH_ERROR_CLASS_RLMESH_EXCEPTION,
        ValueError => RLMeshErrorClass.RLMESH_ERROR_CLASS_VALUE_ERROR,
        TypeError => RLMeshErrorClass.RLMESH_ERROR_CLASS_TYPE_ERROR,
        RuntimeError => RLMeshErrorClass.RLMESH_ERROR_CLASS_RUNTIME_ERROR,
        TimeoutError => RLMeshErrorClass.RLMESH_ERROR_CLASS_TIMEOUT_ERROR,
        ConnectionError => RLMeshErrorClass.RLMESH_ERROR_CLASS_CONNECTION_ERROR,
        OSError => RLMeshErrorClass.RLMESH_ERROR_CLASS_OS_ERROR,
        _ => RLMeshErrorClass.RLMESH_ERROR_CLASS_INVALID,
    };
}
