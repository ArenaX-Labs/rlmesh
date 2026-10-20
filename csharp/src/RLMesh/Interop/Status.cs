using System;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace RLMesh.Interop;

internal static unsafe class Status
{
    internal static void Check(RLMeshStatus status, CancellationToken cancellationToken = default)
    {
        if (status == RLMeshStatus.RLMESH_OK) return;
        // Copy TLS text first, on this thread. These getters do not clear TLS.
        string message = Utf8.Borrowed(Native.rlmesh_last_error_message());
        string code = Utf8.Borrowed(Native.rlmesh_last_error_code_str());
        var cls = Native.rlmesh_last_error_class();
        bool recoverable = Native.rlmesh_last_error_is_recoverable() != 0;
        void* payload = null;
        IntPtr deleter = default;
        void* tag = null;
        var take = Native.rlmesh_ext_last_error_take_payload_owned(&payload, &deleter, &tag);
        if (take == RLMeshStatus.RLMESH_OK && payload != null)
        {
            if ((IntPtr)tag == Callbacks.PayloadTag)
            {
                var handle = GCHandle.FromIntPtr((IntPtr)payload);
                ExceptionDispatchInfo? exception;
                try { exception = handle.Target as ExceptionDispatchInfo; }
                finally { RLMesh.Internal.ManagedRoots.Free(handle); }
                if (exception != null) exception.Throw();
            }
            else if (deleter != IntPtr.Zero)
                Marshal.GetDelegateForFunctionPointer<Callbacks.FreeFn>(deleter)((IntPtr)payload);
        }
        string prefix = "[" + code + "] ";
        string raw = message.StartsWith(prefix, StringComparison.Ordinal) ? message.Substring(prefix.Length) : message;
        if (status == RLMeshStatus.RLMESH_ERR_CANCELLED) throw new Ext.CancelledError(cancellationToken);
        throw cls switch
        {
            RLMeshErrorClass.RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION => new ProtocolException(message),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION when recoverable => new RecoverableEnvironmentException(message, raw),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION => new EnvironmentException(message, raw),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION => new RecoverableEnvironmentException(message, raw),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_VALUE_ERROR => new ValueError(message),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_TYPE_ERROR => new TypeError(message),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_RUNTIME_ERROR => new RuntimeError(message),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_OS_ERROR => new OSError(message),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_CONNECTION_ERROR => new ConnectionError(message),
            RLMeshErrorClass.RLMESH_ERROR_CLASS_TIMEOUT_ERROR => new TimeoutError(message),
            _ => new RLMeshException(message),
        };
    }
}
