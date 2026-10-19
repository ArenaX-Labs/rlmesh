# RLMesh for C# — Experimental

RLMesh for C# supports serving an authored environment to a client. It targets
`netstandard2.1` and `net10.0` on 64-bit hosts. Its public API is consumable
from C# 9. Experimental APIs may change or disappear.

The SDK is pinned in `global.json`. The C# tasks install it on demand.
From the repository root:

```sh
mise run check:csharp
```

Make the matching native library available to the platform loader as
`rlmesh_capi` (`librlmesh_capi.so` on Linux, `rlmesh_capi.dll` on Windows,
`librlmesh_capi.dylib` on macOS). The source package does not distribute
native binaries. The first native call checks ABI generation 5 and the
header's package version, currently `0.1.0`. Loader and version failures
throw `RLMesh.RuntimeError` with the cause.

```csharp
using System;
using RLMesh;
using RLMesh.Spaces;

sealed class Counter : IEnv, IClose
{
    private int steps;
    public Space ObservationSpace { get; } = new Discrete(100);
    public Space ActionSpace { get; } = new Discrete(2);

    public ResetResult Reset(long? seed, Metadata? options)
    {
        steps = 0;
        return new ResetResult(0, new Metadata());
    }

    public StepResult Step(Value? action)
    {
        steps++;
        return new StepResult(steps, 1.0, steps == 2, false, new Metadata());
    }

    public void Close() { }
}

static class Program
{
    static void Main()
    {
        using var server = new EnvServer(new Counter());
        Console.WriteLine(server.Address);
        server.Start();
        Console.ReadLine();
        server.Shutdown();
    }
}
```

Construction binds the address and can throw `ConnectionError`. The default
address is `tcp://127.0.0.1:0`; pass `EnvServerOptions.Address` to choose one.
`Wait(TimeSpan.Zero)` polls; `Shutdown()` drains and closes the environment
once if it implements `IClose` and `CloseEnvOnShutdown` is true (the default).
`Dispose()` releases resources and never throws. After disposal, native
operations throw `ObjectDisposedException`. Dispose servers before borrowed
`Env` wrappers.

An engine can start the server and poll from its tick. Callbacks execute on
native library threads. Dispatch to an engine's main thread in the integration
when necessary; avoid blocking that main thread on a call waiting for the same
callback. Shutdown is synchronous. Runtime reload and unload are unsupported;
keep the managed runtime alive until all servers have been disposed.

`Env.For(obj)` returns the unique caller-owned wrapper for that object. A
server constructed from `IEnv` borrows an existing caller wrapper or owns a new
handle until it is disposed. `Env.For` refuses a library-owned or disposed but
still borrowed handle with `RuntimeError`. Collection can retire an unborrowed
handle; a subsequent `For` safely adopts or rebuilds it.

Values have strict kinds: integers, floats and booleans are distinct even for
equality. `char` and `ulong` implicit conversions to `Value` are compile errors;
use a string for text, and `Value.FromUInt64` for an integer checked against
`long.MaxValue`. `Metadata` preserves insertion order and compares without
regard to order. Its count-based hash agrees with equality; do not mutate it
while using it as a dictionary key. Text values and metadata keys may contain
NUL; NUL-terminated arguments such as addresses refuse NUL and unpaired
surrogates. Callback error text is sanitized for transport; local rethrows
preserve the original exception object and stack.

Tensor construction copies bytes. `Buffer()` returns an owning lease:

```csharp
using var tensor = new Tensor(new byte[] { 1, 2 }, new long[] { 2 }, DType.Uint8);
using var lease = tensor.Buffer();
ReadOnlySpan<byte> bytes = lease.Span;
```

Disposing the tensor does not invalidate an open lease. Dispose the lease after
its last span or memory use; a span must never be used after lease disposal.
A pin acquired through `Memory.Pin()` keeps storage alive until the pin is
disposed, including after lease disposal. A lease intentionally has no finalizer, so abandoning it leaks its native
reference instead of freeing memory beneath an outstanding span.

The server reports `language: "csharp"` and the detected `dotnet` or `mono`
version under `framework_versions`. Set
`Ext.EnvServerOptions.HostRuntimeJson` to replace this advisory information.
The detection does not distinguish IL2CPP from Mono.

`Dict` accepts an enumeration of string/space pairs. Construction copies child
specs and sorts keys; reads and `foreach` then follow that declaration order,
with `foreach` yielding `KeyValuePair<string, Space>`. Child wrappers are cached.
For example:

```csharp
var observationSpace = new Dict(new System.Collections.Generic.KeyValuePair<string, Space>[]
{
    new("image", new Box(0, 255, new long[] { 64, 64, 3 }, new BoxOptions { Dtype = DType.Uint8 })),
    new("state", new Box(-1, 1, new long[] { 3 })),
});
```

Space getters must return promptly. A getter must not wait for another thread
that calls `Env.For` on the same environment: that call waits for the first
construction to finish. Same-thread reentry throws `RuntimeError`.

For source development and tests, see [the maintainer guide](../docs/csharp.md).
