using System;
using System.Text;

namespace RLMesh.Interop;

internal static unsafe class Utf8
{
    private static readonly UTF8Encoding Strict = new(false, true);
    internal static byte[] Encode(string text, bool terminated = false)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        if (terminated && text.IndexOf('\0') >= 0) throw new ValueError("text contains U+0000");
        try { return Strict.GetBytes(terminated ? text + "\0" : text); }
        catch (EncoderFallbackException e) { throw new ValueError("text contains an unpaired surrogate", e); }
    }
    internal static string Borrowed(sbyte* text)
    {
        if (text == null) return "";
        int length = 0;
        while (text[length] != 0) length++;
        return Encoding.UTF8.GetString((byte*)text, length);
    }
    internal static string Borrowed(RLMeshStr text) => Encoding.UTF8.GetString((byte*)text.data, Length(text.len));
    internal static int Length(UIntPtr size)
    {
        if (size.ToUInt64() > int.MaxValue) throw new ValueError("data exceeds the .NET collection length limit");
        return (int)size.ToUInt64();
    }
    internal static byte[] Copy(void* data, UIntPtr length) => new ReadOnlySpan<byte>(data, Length(length)).ToArray();
    internal static string OwnedText(RLMeshBytes bytes)
    {
        try { return bytes.len == UIntPtr.Zero ? "" : Encoding.UTF8.GetString(bytes.data, Length(bytes.len)); }
        finally { Native.rlmesh_bytes_free(bytes); }
    }
    internal static byte[] OwnedBytes(RLMeshBytes bytes)
    {
        try { return Copy(bytes.data, bytes.len); }
        finally { Native.rlmesh_bytes_free(bytes); }
    }
}
