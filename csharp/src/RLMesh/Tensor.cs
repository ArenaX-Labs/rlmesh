using System;
using System.Buffers;
using System.Collections.Generic;
using System.Threading;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh;

/// <summary>Experimental. An immutable, copied tensor with shared native storage.</summary>
public sealed unsafe class Tensor : IDisposable
{
    internal readonly NativeHandle Handle;
    internal Tensor(RLMeshTensor* owned) { Handle = new((IntPtr)owned, HandleKind.Tensor); }
    public Tensor(ReadOnlySpan<byte> buffer, IReadOnlyList<long> shape, DType dtype)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        var dimensions = new UIntPtr[shape.Count];
        for (int i = 0; i < dimensions.Length; i++)
        {
            if (shape[i] < 0) throw new ValueError("tensor dimensions must be nonnegative");
            dimensions[i] = new UIntPtr((ulong)shape[i]);
        }
        byte[] bytes = buffer.ToArray();
        Handle = NativeGate.Call(() =>
        {
            RLMeshTensor* output = null;
            fixed (byte* data = bytes)
            fixed (UIntPtr* dims = dimensions)
                Status.Check(Native.rlmesh_tensor_new(data, (UIntPtr)bytes.Length, dims, (UIntPtr)dimensions.Length, (RLMeshDType)dtype, &output));
            return new NativeHandle((IntPtr)output, HandleKind.Tensor);
        });
    }
    public IReadOnlyList<long> Shape => NativeGate.Call(() =>
    {
        using var call = Handle.Acquire(this);
        UIntPtr n = default; Status.Check(Native.rlmesh_tensor_shape(call.Tensor, null, UIntPtr.Zero, &n));
        var dimensions = new UIntPtr[Utf8.Length(n)];
        fixed (UIntPtr* data = dimensions) Status.Check(Native.rlmesh_tensor_shape(call.Tensor, data, n, &n));
        var result = new long[dimensions.Length];
        for (int i = 0; i < result.Length; i++)
        {
            ulong size = dimensions[i].ToUInt64();
            if (size > long.MaxValue) throw new ValueError("tensor dimension exceeds int64");
            result[i] = (long)size;
        }
        return Array.AsReadOnly(result);
    });
    public DType Dtype => NativeGate.Call(() => { using var call = Handle.Acquire(this); return (DType)Native.rlmesh_tensor_dtype(call.Tensor); });
    public byte[] Tobytes() => NativeGate.Call(() =>
    {
        using var call = Handle.Acquire(this);
        UIntPtr length = default; Status.Check(Native.rlmesh_tensor_nbytes(call.Tensor, &length));
        _ = Utf8.Length(length);
        RLMeshBytes bytes = default; Status.Check(Native.rlmesh_tensor_tobytes(call.Tensor, &bytes));
        return Utf8.OwnedBytes(bytes);
    });
    public TensorLease<byte> Buffer() => NativeGate.Call(() =>
    {
        using var call = Handle.Acquire(this);
        RLMeshTensor* clone = Native.rlmesh_tensor_clone(call.Tensor);
        return new TensorLease<byte>(clone);
    });
    public void Dispose()
    {
        try { Handle.CloseOwner(0, true); Handle.Dispose(); }
        catch (Exception e) { NativeHandle.CleanupFailures.Enqueue(e); }
        finally { GC.KeepAlive(this); }
    }
}

/// <summary>Experimental. Owns a native reference until disposed. Spans must not be used after disposal.</summary>
public sealed unsafe class TensorLease<T> : IDisposable where T : unmanaged
{
    private readonly NativeHandle handle;
    internal NativeHandle Record => handle;
    private readonly LeaseMemory memory;
    private readonly IntPtr pointer;
    private readonly int length;
    private int disposed;
    internal TensorLease(RLMeshTensor* owned)
    {
        // No finalizer: a leaked lease intentionally leaks its native reference,
        // so a span can never outlive storage merely through collection.
        handle = new NativeHandle((IntPtr)owned, HandleKind.Tensor, finalizable: false);
        try
        {
            using var call = handle.Acquire(this);
            void* data = null; UIntPtr bytes = default;
            Status.Check(Native.rlmesh_tensor_buffer(call.Tensor, &data, &bytes));
            pointer = (IntPtr)data;
            length = Utf8.Length(bytes) / sizeof(T);
            memory = new LeaseMemory(this);
        }
        catch { handle.CloseOwner(0, true); throw; }
    }
    public ReadOnlySpan<T> Span
    {
        get { Ensure(); return new ReadOnlySpan<T>((void*)pointer, length); }
    }
    public ReadOnlyMemory<T> Memory
    {
        get { Ensure(); return memory.Memory; }
    }
    private void Ensure() { if (Volatile.Read(ref disposed) != 0) throw new ObjectDisposedException(nameof(TensorLease<T>)); }
    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        handle.CloseOwner(0, true);
        handle.Dispose();
        GC.KeepAlive(this);
    }
    private sealed class LeaseMemory : MemoryManager<T>
    {
        private readonly TensorLease<T> lease;
        private readonly object sync = new();
        private readonly Stack<HandleLease> pins = new();
        internal LeaseMemory(TensorLease<T> lease) { this.lease = lease; }
        public override Span<T> GetSpan() { lease.Ensure(); return new Span<T>((void*)lease.pointer, lease.length); }
        public override MemoryHandle Pin(int elementIndex = 0)
        {
            lease.Ensure();
            if (elementIndex < 0 || elementIndex > lease.length) throw new ArgumentOutOfRangeException(nameof(elementIndex));
            lock (sync)
            {
                // Acquire races safely with CloseOwner. Each pin keeps storage
                // alive even after the lease itself is explicitly disposed.
                var pin = lease.handle.Acquire(lease);
                try { pins.Push(pin); }
                catch { pin.Dispose(); throw; }
                return new MemoryHandle((T*)lease.pointer + elementIndex, default, this);
            }
        }
        public override void Unpin()
        {
            HandleLease pin;
            lock (sync) pin = pins.Pop();
            pin.Dispose();
        }
        protected override void Dispose(bool disposing) { }
    }
}
