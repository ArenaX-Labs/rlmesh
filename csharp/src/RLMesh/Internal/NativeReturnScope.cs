using System;
using System.Threading;

namespace RLMesh.Internal;

// A cleanup callback's final instruction precedes its return through the native
// frame. Keep that return accounted for separately from the trampoline count.
internal sealed class NativeReturnScope : IDisposable
{
    [ThreadStatic] internal static NativeReturnScope? Current;
    private readonly NativeReturnScope? previous;
    private readonly bool attached;
    private int returned;
    internal bool Returned => Volatile.Read(ref returned) != 0;
    internal NativeReturnScope(bool attached = true) { this.attached = attached; if (attached) { previous = Current; Current = this; } }
    internal void CompleteJoin() => Volatile.Write(ref returned, 1);
    public void Dispose() { if (attached) Current = previous; CompleteJoin(); }
}
