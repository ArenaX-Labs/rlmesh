# RLMesh for C# — Experimental

RLMesh for C# supports serving an authored environment to a client. It targets
`netstandard2.1` and `net10.0` on 64-bit hosts. Its public API is consumable
from C# 9. Experimental APIs may change or disappear.

## Install

Download `RLMesh.<version>.nupkg` from the matching
[GitHub Release](https://github.com/ArenaX-Labs/rlmesh/releases) into a local feed,
then install it in a .NET application:

```sh
dotnet add package RLMesh --version <version> --source /path/to/feed
```

The package includes `netstandard2.1` and `net10.0` assemblies plus Linux x64,
Windows x64 and universal macOS arm64/x64 native libraries. NuGet selects the
native asset automatically. The `.snupkg` is available for portable symbols.
nuget.org publication activates once the maintainer configures trusted publishing;
GitHub Release packages remain available as a local feed. When the version is
available on nuget.org, omit `--source /path/to/feed`.

For Unity 2021.3 or later using .NET Standard 2.1, download
`com.rlmesh.rlmesh-<version>.tgz` and select **Add package from tarball** in Package
Manager. The prebuilt DLL is auto-referenced, and native importers select desktop
Editor/Standalone by OS and CPU. 32-bit, WebGL, iOS and Android are unsupported.
No asmdef is needed: an asmdef defines compilation of sources; this package
supplies a prebuilt assembly with its PluginImporter enabled and auto-referenced.

For a git URL installation, unpack the generated `package/` directory into a
versioned distribution repository and use Package Manager's **Add package from
git URL**, for example `https://your-host/your-team/rlmesh-upm.git#v<version>`.
A subdirectory distribution uses `?path=/path/to/package#v<version>`. Pin the
revision and retain all `.meta` files. The framework source repository does not
commit generated binaries, so its source URL alone is not an installable UPM
package. You can also unpack a release tarball and use **Add package from disk**
with its `package.json`.

Source development uses the SDK pinned in `global.json`; `mise run check:csharp`
installs it on demand. Build the native library and make `rlmesh_capi`
(`librlmesh_capi.so`, `rlmesh_capi.dll` or `librlmesh_capi.dylib`) available to the
platform loader when consuming a source build.

The first native call checks ABI generation 6 and the
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
address is `:0/env`, the env `env` in a session of its own on an OS-assigned
local port; pass `EnvServerOptions.Address` (`host:port/session/name`) to choose
one. Python dials the printed address: `rlmesh.numpy.RemoteEnv(address)`.
`Wait(TimeSpan.Zero)` polls; `Shutdown()` drains and closes the environment
once if it implements `IClose` and `CloseEnvOnShutdown` is true (the default).
`Dispose()` releases resources and never throws. After disposal, native
operations throw `ObjectDisposedException`. Dispose servers before borrowed
`Env` wrappers.

An engine can start the server and poll from its tick. Callbacks execute on
native library threads. Dispatch to an engine's main thread in the integration
when necessary; avoid blocking that main thread on a call waiting for the same
callback. `ShutdownAsync(cancellationToken)` drains on a library worker. Cancelling
its token stops waiting with `RLMesh.Ext.CancelledError`; the drain continues.
An already-cancelled token leaves the server untouched.

## Shutdown before unloading

Before an engine host unloads its managed runtime or a collectible context, call
`RLMesh.Ext.Lifetime.Shutdown(TimeSpan.FromSeconds(5))` outside a callback.
Unload only when it returns `true`. If it returns `false`, do not unload: inspect
`Lifetime.Pending()`, release any blocked environment code, and retry. Shutdown
is terminal; later native operations throw `ObjectDisposedException`.
`Lifetime.ShuttingDown` lets a host fail queued main-thread dispatch requests
before the drain waits for them. Its registrations run on a library thread, so
marshal any engine work to the appropriate thread.

`Lifetime.WaitFreed(envOrServer, timeout)` waits for native release after disposal.
The unload and process-exit hooks provide diagnostics and best-effort cleanup;
they do not replace the host’s preflight.

## Authoring values

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
