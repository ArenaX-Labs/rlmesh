using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh.Spaces;

/// <summary>Experimental. Named child spaces in the native spec's declaration order.</summary>
public sealed unsafe class Dict : Space, IReadOnlyDictionary<string, Space>
{
    private readonly List<KeyValuePair<string, Space>> entries = new();
    private readonly Dictionary<string, Space> children = new(StringComparer.Ordinal);
    public Dict(IEnumerable<KeyValuePair<string, Space>> spaces) : this(Prepare(spaces)) { }
    private Dict(Dict prepared) : base(prepared.Handle) { entries = prepared.entries; children = prepared.children; }
    private static Dict Prepare(IEnumerable<KeyValuePair<string, Space>> spaces) =>
        NativeGate.Call(() => new Dict(Create(spaces)), "Dict constructor");
    internal Dict(NativeHandle handle) : base(handle)
    {
        try
        {
            NativeGate.Call(() =>
            {
                using var call = handle.Acquire(this);
                UIntPtr count = default; Status.Check(Native.rlmesh_dict_len(call.Space, &count));
                for (int i = 0; i < Utf8.Length(count); i++)
                {
                    RLMeshStr key = default; RLMeshSpace* child = null;
                    Status.Check(Native.rlmesh_dict_key_at(call.Space, (UIntPtr)i, &key));
                    Status.Check(Native.rlmesh_dict_value_at(call.Space, (UIntPtr)i, &child));
                    string name = Utf8.Borrowed(key);
                    var owned = new NativeHandle((IntPtr)Native.rlmesh_space_clone(child), HandleKind.Space);
                    Space space = Native.rlmesh_space_kind(child) switch
                    {
                        RLMeshSpaceKind.RLMESH_SPACE_KIND_BOX => new Box(owned),
                        RLMeshSpaceKind.RLMESH_SPACE_KIND_DISCRETE => new Discrete(owned),
                        RLMeshSpaceKind.RLMESH_SPACE_KIND_DICT => new Dict(owned),
                        _ => new Space(owned),
                    };
                    children.Add(name, space); entries.Add(new(name, space));
                }
            });
        }
        catch { handle.CloseOwner(0, true); throw; }
    }
    private static NativeHandle Create(IEnumerable<KeyValuePair<string, Space>> spaces)
    {
        if (spaces == null) throw new ArgumentNullException(nameof(spaces));
        var pairs = new List<KeyValuePair<string, Space>>(spaces);
        return NativeGate.Call(() =>
        {
            var keys = new RLMeshStr[pairs.Count]; var specs = new IntPtr[pairs.Count];
            var pins = new List<GCHandle>(); var borrows = new List<HandleLease>();
            try
            {
                for (int i = 0; i < pairs.Count; i++)
                {
                    if (pairs[i].Value == null) throw new ArgumentNullException(nameof(spaces));
                    byte[] text = Utf8.Encode(pairs[i].Key);
                    var pin = GCHandle.Alloc(text, GCHandleType.Pinned); pins.Add(pin);
                    keys[i] = new RLMeshStr { data = (sbyte*)pin.AddrOfPinnedObject(), len = (UIntPtr)text.Length };
                    var borrow = pairs[i].Value.Handle.Acquire(pairs[i].Value); borrows.Add(borrow);
                    specs[i] = (IntPtr)Native.rlmesh_space_spec(borrow.Space);
                }
                RLMeshSpace* output = null;
                fixed (RLMeshStr* keyData = keys)
                fixed (IntPtr* specData = specs)
                    Status.Check(Native.rlmesh_dict_new(keyData, (RLMeshSpaceSpec**)specData, (UIntPtr)pairs.Count, &output));
                return new NativeHandle((IntPtr)output, HandleKind.Space);
            }
            finally
            {
                foreach (var borrow in borrows) borrow.Dispose();
                foreach (var pin in pins) pin.Free();
            }
        });
    }
    public int Count => entries.Count;
    public Space this[string key] => children[key];
    public bool TryGetValue(string key, out Space value) => children.TryGetValue(key, out value!);
    public bool ContainsKey(string key) => children.ContainsKey(key);
    public IEnumerable<string> Keys { get { foreach (var pair in entries) yield return pair.Key; } }
    public IEnumerable<Space> Values { get { foreach (var pair in entries) yield return pair.Value; } }
    public IEnumerator<KeyValuePair<string, Space>> GetEnumerator() => entries.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
