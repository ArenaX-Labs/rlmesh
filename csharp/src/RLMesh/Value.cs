using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh;

/// <summary>Experimental. The kind of a value exchanged with an environment.</summary>
public enum ValueKind { None = 1, Bool = 2, Int = 3, Float = 4, Text = 5, Bytes = 6, Tensor = 7, List = 8, Tuple = 9, Dict = 10 }

/// <summary>Experimental. An immutable native value. Reads and equality are kind-strict.</summary>
public sealed unsafe class Value : IEquatable<Value>
{
    internal readonly NativeHandle Handle;
    internal Value(RLMeshValue* owned) { Handle = new((IntPtr)owned, HandleKind.Value); }
    internal static Value Borrow(RLMeshValue* borrowed) => new(Native.rlmesh_value_clone(borrowed));
    public static Value None => NativeGate.Call(() => { RLMeshValue* p = null; Status.Check(Native.rlmesh_value_none(&p)); return new Value(p); });
    public static implicit operator Value(bool v) => NativeGate.Call(() => { RLMeshValue* p = null; Status.Check(Native.rlmesh_value_bool(v ? (byte)1 : (byte)0, &p)); return new Value(p); });
    public static implicit operator Value(sbyte v) => Integer(v);
    public static implicit operator Value(byte v) => Integer(v);
    public static implicit operator Value(short v) => Integer(v);
    public static implicit operator Value(ushort v) => Integer(v);
    public static implicit operator Value(int v) => Integer(v);
    public static implicit operator Value(uint v) => Integer(v);
    public static implicit operator Value(long v) => Integer(v);
    public static implicit operator Value(float v) => Floating(v);
    public static implicit operator Value(double v) => Floating(v);
    [Obsolete("char is not a Value; convert explicitly to string or a signed integer", true)]
    public static implicit operator Value(char v) => throw new TypeError("char is not a Value");
    [Obsolete("use Value.FromUInt64 for a checked integer conversion", true)]
    public static implicit operator Value(ulong v) => throw new TypeError("ulong requires Value.FromUInt64");
    private static Value Integer(long v) => NativeGate.Call(() => { RLMeshValue* p = null; Status.Check(Native.rlmesh_value_int(v, &p)); return new Value(p); });
    private static Value Floating(double v) => NativeGate.Call(() => { RLMeshValue* p = null; Status.Check(Native.rlmesh_value_float(v, &p)); return new Value(p); });
    public static Value FromUInt64(ulong v)
    {
        if (v > long.MaxValue) throw new ValueError("integer exceeds int64");
        return Integer((long)v);
    }
    public static implicit operator Value(string v)
    {
        byte[] text = Utf8.Encode(v);
        return NativeGate.Call(() =>
        {
            RLMeshValue* p = null;
            fixed (byte* data = text)
            {
                // A non-null pointer distinguishes an empty string from a missing view.
                byte empty = 0;
                Status.Check(Native.rlmesh_value_text(new RLMeshStr { data = (sbyte*)(text.Length == 0 ? &empty : data), len = (UIntPtr)text.Length }, &p));
            }
            return new Value(p);
        });
    }
    public static implicit operator Value(Tensor v)
    {
        if (v == null) throw new ArgumentNullException(nameof(v));
        return NativeGate.Call(() => { using var call = v.Handle.Acquire(v); RLMeshValue* p = null; Status.Check(Native.rlmesh_value_tensor(call.Tensor, &p)); return new Value(p); });
    }
    public static implicit operator Value(Metadata v) => Dict(v);
    public static Value Bytes(ReadOnlySpan<byte> data)
    {
        var bytes = data.ToArray();
        return NativeGate.Call(() => { RLMeshValue* p = null; fixed (byte* buffer = bytes) Status.Check(Native.rlmesh_value_bytes(buffer, (UIntPtr)bytes.Length, &p)); return new Value(p); });
    }
    public static Value List(params Value[] items) => Sequence(items, false);
    public static Value List(IEnumerable<Value> items) => Sequence(items, false);
    public static Value Tuple(params Value[] items) => Sequence(items, true);
    public static Value Tuple(IEnumerable<Value> items) => Sequence(items, true);
    private static Value Sequence(IEnumerable<Value> items, bool tuple)
    {
        if (items == null) throw new ArgumentNullException(nameof(items));
        var values = new List<Value>(items);
        return NativeGate.Call(() =>
        {
            var calls = new List<HandleLease>(values.Count);
            try
            {
                var pointers = new IntPtr[values.Count];
                for (int i = 0; i < pointers.Length; i++)
                {
                    if (values[i] is null) throw new ArgumentNullException(nameof(items));
                    var call = values[i].Handle.Acquire(values[i]); calls.Add(call); pointers[i] = call.Pointer;
                }
                RLMeshValue* p = null;
                fixed (IntPtr* data = pointers)
                    Status.Check(tuple ? Native.rlmesh_value_tuple((RLMeshValue**)data, (UIntPtr)pointers.Length, &p) : Native.rlmesh_value_list((RLMeshValue**)data, (UIntPtr)pointers.Length, &p));
                return new Value(p);
            }
            finally { for (int i = calls.Count - 1; i >= 0; i--) calls[i].Dispose(); }
        });
    }
    public static Value Dict(Metadata items)
    {
        if (items == null) throw new ArgumentNullException(nameof(items));
        return NativeGate.Call(() =>
        {
            var pins = new List<GCHandle>();
            var calls = new List<HandleLease>();
            try
            {
                var keys = new RLMeshStr[items.Count];
                var values = new IntPtr[items.Count];
                int index = 0;
                foreach (var pair in items)
                {
                    // Reserve a sentinel byte even for an empty length-bearing key.
                    byte[] key = Utf8.Encode(pair.Key + "\0");
                    var pin = GCHandle.Alloc(key, GCHandleType.Pinned); pins.Add(pin);
                    keys[index] = new() { data = (sbyte*)pin.AddrOfPinnedObject(), len = (UIntPtr)(key.Length - 1) };
                    var call = pair.Value.Handle.Acquire(pair.Value); calls.Add(call); values[index++] = call.Pointer;
                }
                RLMeshValue* p = null;
                fixed (RLMeshStr* keyData = keys)
                fixed (IntPtr* valueData = values)
                    Status.Check(Native.rlmesh_value_dict(keyData, (RLMeshValue**)valueData, (UIntPtr)values.Length, &p));
                return new Value(p);
            }
            finally
            {
                for (int i = calls.Count - 1; i >= 0; i--) calls[i].Dispose();
                foreach (var pin in pins) pin.Free();
            }
        });
    }
    public ValueKind Kind => NativeGate.Call(() => { using var call = Handle.Acquire(this); return (ValueKind)Native.rlmesh_value_kind(call.Value); });
    public bool IsNone => Kind == ValueKind.None;
    public bool AsBool() => NativeGate.Call(() => { using var call = Handle.Acquire(this); bool v = false; Status.Check(Native.rlmesh_value_as_bool(call.Value, &v)); return v; });
    public long AsInt() => NativeGate.Call(() => { using var call = Handle.Acquire(this); long v = 0; Status.Check(Native.rlmesh_value_as_int(call.Value, &v)); return v; });
    public double AsFloat() => NativeGate.Call(() => { using var call = Handle.Acquire(this); double v = 0; Status.Check(Native.rlmesh_value_as_float(call.Value, &v)); return v; });
    public string AsText() => NativeGate.Call(() => { using var call = Handle.Acquire(this); RLMeshStr v = default; Status.Check(Native.rlmesh_value_as_text(call.Value, &v)); return Utf8.Borrowed(v); });
    public byte[] AsBytes() => NativeGate.Call(() => { using var call = Handle.Acquire(this); byte* data = null; UIntPtr length = default; Status.Check(Native.rlmesh_value_as_bytes(call.Value, &data, &length)); return Utf8.Copy(data, length); });
    public Tensor AsTensor() => NativeGate.Call(() => { using var call = Handle.Acquire(this); RLMeshTensor* p = null; Status.Check(Native.rlmesh_value_as_tensor(call.Value, &p)); return new Tensor(p); });
    public IReadOnlyList<Value> AsList() => ReadSequence(ValueKind.List);
    public IReadOnlyList<Value> AsTuple() => ReadSequence(ValueKind.Tuple);
    private IReadOnlyList<Value> ReadSequence(ValueKind kind) => NativeGate.Call(() =>
    {
        using var call = Handle.Acquire(this);
        if ((ValueKind)Native.rlmesh_value_kind(call.Value) != kind) throw new TypeError("value has the wrong sequence kind");
        UIntPtr length = default; Status.Check(Native.rlmesh_value_len(call.Value, &length));
        var result = new Value[Utf8.Length(length)];
        for (int i = 0; i < result.Length; i++) { RLMeshValue* p = null; Status.Check(Native.rlmesh_value_get(call.Value, (UIntPtr)i, &p)); result[i] = Borrow(p); }
        return Array.AsReadOnly(result);
    });
    public Metadata AsDict() => NativeGate.Call(() =>
    {
        using var call = Handle.Acquire(this);
        if (Native.rlmesh_value_kind(call.Value) != RLMeshValueKind.RLMESH_VALUE_DICT) throw new TypeError("value is not a dict");
        UIntPtr length = default; Status.Check(Native.rlmesh_value_len(call.Value, &length));
        var result = new Metadata();
        for (int i = 0; i < Utf8.Length(length); i++)
        {
            RLMeshStr key = default; Status.Check(Native.rlmesh_value_dict_key_at(call.Value, (UIntPtr)i, &key));
            string text = Utf8.Borrowed(key);
            RLMeshValue* p = null; Status.Check(Native.rlmesh_value_dict_value_at(call.Value, (UIntPtr)i, &p));
            result.Add(text, Borrow(p));
        }
        return result;
    });
    public int Count => NativeGate.Call(() => { using var call = Handle.Acquire(this); UIntPtr n = default; Status.Check(Native.rlmesh_value_len(call.Value, &n)); return Utf8.Length(n); });
    public Value this[int index] => TryGet(index, out var item) ? item! : throw new ArgumentOutOfRangeException(nameof(index));
    public bool TryGet(int index, out Value? item)
    {
        Value? result = NativeGate.Call(() =>
        {
            using var call = Handle.Acquire(this);
            // Dictionaries have a length too, but never support integer indexing.
            var kind = Native.rlmesh_value_kind(call.Value);
            if (kind is not (RLMeshValueKind.RLMESH_VALUE_LIST or RLMeshValueKind.RLMESH_VALUE_TUPLE))
                throw new TypeError("integer indexing requires a list or tuple");
            UIntPtr length = default; Status.Check(Native.rlmesh_value_len(call.Value, &length));
            if (index < 0 || (ulong)index >= length.ToUInt64()) return null;
            RLMeshValue* p = null; Status.Check(Native.rlmesh_value_get(call.Value, (UIntPtr)index, &p));
            return p == null ? null : Borrow(p);
        });
        item = result;
        return result is not null;
    }
    public bool Equals(Value? other)
    {
        if (other is null) return false;
        return NativeGate.Call(() => { using var left = Handle.Acquire(this); using var right = other.Handle.Acquire(other); return Native.rlmesh_value_equals(left.Value, right.Value) != 0; });
    }
    public override bool Equals(object? obj) => obj is Value other && Equals(other);
    public override int GetHashCode()
    {
        var kind = Kind;
        int length = kind is ValueKind.List or ValueKind.Tuple or ValueKind.Dict ? Count : 0;
        return ((int)kind * 397) ^ length;
    }
    public static bool operator ==(Value? left, Value? right) => left is null ? right is null : left.Equals(right);
    public static bool operator !=(Value? left, Value? right) => !(left == right);
}
