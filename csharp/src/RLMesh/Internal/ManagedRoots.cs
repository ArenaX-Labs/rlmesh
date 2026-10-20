using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace RLMesh.Internal;

internal static class ManagedRoots
{
    internal sealed class RootInfo
    {
        internal readonly string Kind;
        internal readonly NativeHandle? Server;
        internal RootInfo(string kind, NativeHandle? server) { Kind = kind; Server = server; }
    }
    internal sealed class ReleasedRoot
    {
        private readonly NativeReturnScope? nativeReturn;
        internal ReleasedRoot(NativeHandle? server)
        {
            // Only the surrounding native return or the server's explicit
            // native join covers a foreign frame. Env/record Freed does not.
            var host = server?.Kind == HandleKind.EnvServer ? server : server?.HostedBy;
            var join = host?.JoinedNativeReturn;
            // A callback arriving after that join cannot have been covered by it.
            nativeReturn = NativeReturnScope.Current ?? (join?.Returned == false ? join : null);
        }
        internal bool Joined => nativeReturn?.Returned ?? false;
    }

    internal static readonly ConcurrentDictionary<IntPtr, RootInfo> Live = new();
    internal static readonly ConcurrentDictionary<ReleasedRoot, byte> AwaitingReturn = new();
    internal static Action<IntPtr>? BeforeHandleFree;
    internal static GCHandle Allocate(object target, string kind, NativeHandle? server = null)
    {
        var root = GCHandle.Alloc(target);
        try
        {
            if (!Live.TryAdd(GCHandle.ToIntPtr(root), new RootInfo(kind, server)))
                throw new InvalidOperationException("duplicate live GCHandle slot");
            return root;
        }
        catch { root.Free(); throw; }
    }
    internal static void Free(GCHandle root)
    {
        IntPtr pointer = GCHandle.ToIntPtr(root);
        if (!Live.TryRemove(pointer, out _)) throw new InvalidOperationException("untracked GCHandle release");
        try { BeforeHandleFree?.Invoke(pointer); }
        finally { root.Free(); }
    }
    internal static ReleasedRoot FreeFromNative(GCHandle root, NativeHandle? coveredBy = null)
    {
        Live.TryGetValue(GCHandle.ToIntPtr(root), out var info);
        var released = new ReleasedRoot(coveredBy ?? info?.Server);
        AwaitingReturn.TryAdd(released, 0);
        Free(root);
        return released;
    }
    internal static bool ReturnsJoined()
    {
        foreach (var root in AwaitingReturn.Keys)
            if (root.Joined) AwaitingReturn.TryRemove(root, out _);
        return AwaitingReturn.IsEmpty;
    }
}
