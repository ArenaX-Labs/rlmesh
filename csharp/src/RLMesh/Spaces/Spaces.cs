using System;
using System.Collections.Generic;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh.Spaces;

/// <summary>Experimental. An environment space. External subclasses are unsupported.</summary>
public class Space
{
    internal readonly NativeHandle Handle;
    internal Space(NativeHandle handle) { Handle = handle; }
}

/// <summary>Experimental. A scalar Box bound, retaining its numeric domain.</summary>
public readonly struct Bound
{
    internal readonly RLMeshBound NativeValue;
    private Bound(RLMeshBound value) { NativeValue = value; }
    private static Bound Signed(long v) => new(new RLMeshBound { type_ = RLMeshBoundType.RLMESH_BOUND_TYPE_INT, value = new() { i = v } });
    private static Bound Unsigned(ulong v) => new(new RLMeshBound { type_ = RLMeshBoundType.RLMESH_BOUND_TYPE_UINT, value = new() { u = v } });
    private static Bound Floating(double v)
    {
        if (double.IsNaN(v)) throw new ValueError("Box bounds cannot be NaN");
        return new(new RLMeshBound { type_ = RLMeshBoundType.RLMESH_BOUND_TYPE_FLOAT, value = new() { f = v } });
    }
    public static implicit operator Bound(bool v) => new(new RLMeshBound { type_ = RLMeshBoundType.RLMESH_BOUND_TYPE_BOOL, value = new() { b = v ? (byte)1 : (byte)0 } });
    public static implicit operator Bound(sbyte v) => Signed(v);
    public static implicit operator Bound(short v) => Signed(v);
    public static implicit operator Bound(int v) => Signed(v);
    public static implicit operator Bound(long v) => Signed(v);
    public static implicit operator Bound(byte v) => Unsigned(v);
    public static implicit operator Bound(ushort v) => Unsigned(v);
    public static implicit operator Bound(uint v) => Unsigned(v);
    public static implicit operator Bound(ulong v) => Unsigned(v);
    public static implicit operator Bound(float v) => Floating(v);
    public static implicit operator Bound(double v) => Floating(v);
}

/// <summary>Experimental. Box construction options.</summary>
public sealed class BoxOptions
{
    public DType Dtype { get; set; } = DType.Float32;
}

/// <summary>Experimental. A bounded tensor observation or action space.</summary>
public sealed unsafe class Box : Space
{
    internal Box(NativeHandle handle) : base(handle) { }
    public Box(Bound low, Bound high, IReadOnlyList<long> shape, BoxOptions? options = null)
        : base(Create(low, high, shape, options)) { }
    private static NativeHandle Create(Bound low, Bound high, IReadOnlyList<long> shape, BoxOptions? options)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        var dimensions = new long[shape.Count];
        for (int i = 0; i < dimensions.Length; i++) dimensions[i] = shape[i];
        return NativeGate.Call(() =>
        {
            var init = Initializers.BoxOptions();
            init.dtype = (RLMeshDType)(options?.Dtype ?? DType.Float32);
            RLMeshSpace* output = null;
            fixed (long* data = dimensions)
                Status.Check(Native.rlmesh_box_new(low.NativeValue, high.NativeValue, data, (UIntPtr)dimensions.Length, &init, &output));
            return new NativeHandle((IntPtr)output, HandleKind.Space);
        });
    }
}

/// <summary>Experimental. Discrete construction options.</summary>
public sealed class DiscreteOptions
{
    public long Start { get; set; }
    public DType Dtype { get; set; } = DType.Int64;
}

/// <summary>Experimental. A finite integer observation or action space.</summary>
public sealed unsafe class Discrete : Space
{
    internal Discrete(NativeHandle handle) : base(handle) { }
    public Discrete(long n, DiscreteOptions? options = null) : base(NativeGate.Call(() =>
    {
        var init = Initializers.DiscreteOptions();
        init.start = options?.Start ?? 0;
        init.dtype = (RLMeshDType)(options?.Dtype ?? DType.Int64);
        RLMeshSpace* output = null;
        Status.Check(Native.rlmesh_discrete_new(n, &init, &output));
        return new NativeHandle((IntPtr)output, HandleKind.Space);
    })) { }
}
