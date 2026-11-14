# RLMesh for Unity — Experimental

**Experimental; Unity support not yet verified on IL2CPP.** Do not claim Unity
support until all three [Unity release gates](../../docs/unity-release-checklist.md)
pass. This package has licence-free .NET tests; this development host has no
Unity installation or licence. The verification Editor is pinned in
[`ProjectVersion.txt`](ProjectVersion.txt).

`com.rlmesh.unity` serves a scene through `EnvServer.Start()`. Python, Rust,
C and C++ clients can drive it over the same RLMesh wire as the Unreal plugin.
The core package is a separate dependency, `com.rlmesh.rlmesh`, containing the
prebuilt `Runtime/RLMesh.dll` (netstandard2.1) and D-40 native plugins.
This integration does not build or distribute that package.

## Install

Install the matching **core** UPM artifact first: in Package Manager choose
**Install package from tarball** and select `com.rlmesh.rlmesh` from the same
RLMesh release. A directory install also works: its `package.json` must be at
the package root, beside `Runtime/RLMesh.dll` and `Runtime/Plugins/`.

Then choose **Install package from git URL**:

```text
https://github.com/ArenaX-Labs/rlmesh.git?path=/integrations/unity#csharp/unity
```

Pin a release tag or commit instead of the development branch for a reproducible
install. Both packages must have matching versions (currently `0.1.0`). The core
native plugins must be enabled for the Editor and the matching Standalone OS.
Only desktop 64-bit targets are in scope: Linux x64, Windows x64 and macOS
(universal). Mobile and WebGL are unsupported. A platform's release gate must
pass before support for it is claimed; the supplied player gate targets Linux.

## Try Reach Target

Import **Reach Target** from this package's Samples tab. Attach
`RLMeshReachTargetBehaviour` to a GameObject, then press Play. The log shows
`Serving Unity-v0 on 127.0.0.1:7447/env` (set `EnvId` to change the label): the env
`env` in a session the player hosts on port 7447.

The observation is a float32 tensor of shape `[1]`: the offset along world X to
a target chosen on reset. `Discrete(3)` actions move backward, stay, or move
forward. Reward is minus the remaining distance; reaching within half a step
terminates, and `MaxSteps` truncates. This follows the Unreal Reach Target
example's spaces and motion; Unity reports distance directly as requested,
where Unreal normalizes its reward by `TargetRange`.

```python
import rlmesh

print(rlmesh.run(rlmesh.RANDOM_SAMPLE, "127.0.0.1:7447/env", episodes=3))
```

## Serve your own environment

Subclass `RLMeshEnvBehaviour` and override the four environment members:

```csharp
using RLMesh;
using RLMesh.Spaces;
using RLMesh.Unity;

public sealed class CounterBehaviour : RLMeshEnvBehaviour
{
    private int steps;
    protected override Space ObservationSpace => new Discrete(101);
    protected override Space ActionSpace => new Discrete(3);
    protected override ResetResult ResetEnv(long? seed, Metadata options)
    {
        steps = 0;
        return new ResetResult(0, new Metadata());
    }
    protected override StepResult StepEnv(Value action)
    {
        steps++;
        return new StepResult(steps, 1, steps >= 100, false, new Metadata());
    }
}
```

`Start` starts serving. If overriding `Start`, call `base.Start()` after your
initialization; call `base.OnEnable()`, `base.Update()` and `base.OnDisable()` in overrides too.
`Address` selects the session participant address; `:0/env` requests an
ephemeral port. `GetServedAddress()` returns the actual address while serving.
`EnvId` is the log label and reset info's `env_id`: the core v0.2 authoring API
has no environment-ID interface. `StartServing()` / `StopServing()` are explicit
main-thread controls. A component starts serving again when re-enabled. While
serving, the integration enables `Application.runInBackground`, restoring its
previous value after the last component stops. `Update` polls server completion
and reports unexpected serving failures.

The integration sets `Ext.EnvServerOptions.HostRuntimeJson` to
`{"language":"csharp","framework_versions":{"unity":"<Application.unityVersion>","mono":"<runtime version>"}}`,
or `il2cpp` instead of `mono` in IL2CPP players (the IL2CPP version is the Unity
version it ships with). The Editor always reports Mono, even when the selected
player backend is IL2CPP. Under Mono, `System.Environment.Version` is the CLR
compatibility version (often `4.0.30319.42000`), not a Mono release version. This
is describe's host runtime, not native handshake `PeerInfo`.

## Threads

The core's R-28/D-38 rules remain the same: synchronous local operations and
factory callbacks run on their caller; served environments run on library
threads, potentially several at once. The core does not capture a
`SynchronizationContext` or marshal calls. This integration marshals all four
authored members onto the Unity main thread.

`MainThreadDispatcher.Call` runs inline when already on that thread, including
re-entry from a callback. Otherwise it blocks the caller until `Update` or the
Editor pump executes it. Never block the Unity main thread on an RLMesh call
that needs another main-thread callback. Keep callbacks short and space getters
prompt; no dispatcher can keep the Editor responsive while authored main-thread
code is itself blocked.

Throw `RecoverableEnvironmentException` for a failure after which reset may
recover. Every authored exception passes through unchanged, including its
identity, stack and RLMesh class; the core trampoline handles wire classification.
Closing the dispatcher or starting `Ext.Lifetime` shutdown fails pending and
future calls with `RuntimeError`. The shutdown check and enqueue share one lock, preventing a
request from being stranded behind shutdown. An already running callback must
return before the barrier can complete. Synchronous dispatch explicitly carries
the core callback depth and host identity, then restores the main thread's prior
context before releasing the native lane. This preserves callback disposal,
shutdown refusal and permission to finish constructing results during shutdown.

The host captures the main thread during Editor/runtime initialization and
rejects host operations or dispatcher construction from other threads. In a
plain .NET fake host, construct the dispatcher on the pumping thread; host registry
users must first call `UnityEnvHost.InitializeMainThread()` there.
For a manually created `MainThreadEnv`/`EnvServer`, close the dispatcher **before**
disposing the server on the main thread. Otherwise a pending callback can wait
for the same main thread that is waiting for disposal.

## Reload and quitting

Before any native serving begins, the Editor takes
`EditorApplication.LockReloadAssemblies()`. It keeps that lock even if every
component is disabled. Compilation, entry into Play Mode with domain reload
enabled, or **RLMesh > Shutdown and Reload** start `Ext.Lifetime.Shutdown(timeout)` on a worker. The Editor keeps
pumping the dispatcher and polls completion; no blocking native join runs on
the main thread. Disabling a component also drains its server on a worker.

Only a `true` barrier result, followed by completion of component disposal
workers, releases the reload lock. On `false` or an exception, the lock stays
held, `Pending()` is logged, and the Editor tells you to release blocked code
and retry **RLMesh > Shutdown and Reload**. Shutdown is terminal: reload the
domain before serving again after terminal shutdown. Ordinary Play Mode exit
only stops the components; entering with domain reload disabled does not run the
terminal barrier or force a reload. The menu deliberately requests a fresh domain.

**RLMesh > Force unlock reload** is an explicitly confirmed, unsafe escape hatch.
It warns that active callbacks can crash an ensuing reload and never claims the
barrier completed. A persisted SessionState flag recovers an orphaned native lock
count if Unity unloads the old domain despite the lock.

`beforeAssemblyReload` logs unsafe unload and attempts a one-second worker
shutdown as damage limitation; this cannot make an already-started unload safe.
`Application.quitting` is diagnostic. Normal Editor/player quitting attempts a
five-second worker barrier, logs pending work on failure, then accepts the original
quit request. The Editor offers **Quit anyway** or cancel after failure. This
preserves normal save prompts, other quit handlers and player exit codes; it never
vetoes quitting forever. A player retains a persistent pump after the served
GameObject ends. Process exit does not require domain-unload safety.

The core grants internal access to assembly `RLMesh.Unity` (also the Unity asmdef
name and the .NET fixture assembly name), with no public core API additions. These
are ordinary compiled calls, not reflection; `Runtime/link.xml` preserves the core
and its callback/context machinery under IL2CPP High stripping. Only the actual
IL2CPP release gate can establish Unity support.

## Verification

`mise run test:csharp` compiles the Unity-free core as C# 9/netstandard2.1 and
runs dispatcher, reload-coordinator and Python-to-.NET end-to-end tests without
Unity. `test:affected` selects these for any `integrations/unity/**` change.

`Tests~/Project` contains EditMode reload tests, a PlayMode component test and
an IL2CPP smoke build script. `Tests~/prepare.py` consumes a separately built
core UPM directory or tarball, stages the integration/sample and builds the
native client used to wedge a real served callback. It does not package the
core. Run it through `mise run dev:heavy -- …`.

Follow the [release checklist](../../docs/unity-release-checklist.md) for the
three Unity gates and dispatch-only GameCI workflow. Passing .NET tests or
NativeAOT does not establish Mono Editor or IL2CPP support.
