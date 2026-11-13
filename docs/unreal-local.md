# Unreal Engine: local RLMesh setup

How to build the C/C++ library from this checkout, install the Unreal plugin into a project you already have, and confirm the native stack. Env component details and Python demos live in [`integrations/unreal/RLMesh/README.md`](../integrations/unreal/RLMesh/README.md). The C++ API is in [`crates/rlmesh-capi/README-cpp.md`](../crates/rlmesh-capi/README-cpp.md).

The plugin is standalone: four drop-in components (Reach Target, Chase Point, Avoid Hazards, Collect Coins). Add a component, Press Play, connect with `--address` / `RLMESH_UNREAL_ADDR`.

## Engine versions

The plugin targets **Unreal Engine 5** (Win64 and Linux). [`.uplugin`](../integrations/unreal/RLMesh/RLMesh.uplugin) does not pin a minor version.

| Engine | Status |
|--------|--------|
| 5.6.x | Intended / compatible (API surface) |
| 5.7.x | Intended / compatible (API surface) |
| 5.8 | Expected; not engine-tested in-repo |

In-repo checks do **not** run UnrealBuildTool. They cover the game-thread helper and native gate via `mise run test:cxx`.

## 1. Build the C/C++ package

From the repo root (use `mise run dev:heavy -- …` on a shared machine):

```bash
mise run release:cxx:package
```

On Windows that produces:

`target/capi-pkg/rlmesh-<version>-x86_64-pc-windows-msvc.tar.gz`

Linux host packages are `…-x86_64-unknown-linux-gnu.tar.gz`. You can also download the matching tarball from [GitHub releases](https://github.com/ArenaX-Labs/rlmesh/releases).

On Windows, set `RLMESH_RELEASE_BUILD=1` when packaging if this checkout’s Python and the plugin’s `rlmesh_capi` must share a sealed workflow edition (see [local-dev](local-dev.md)).

## 2. Unpack into the plugin ThirdParty tree

For developing against this checkout:

```bash
tar -xzf target/capi-pkg/rlmesh-<version>-x86_64-pc-windows-msvc.tar.gz \
  --strip-components=1 \
  -C integrations/unreal/RLMesh/Source/ThirdParty/rlmesh
```

That folder must then contain `include/` (with `rlmesh.hpp`), `lib/` (`rlmesh_capi.lib` on Windows), and on Windows `bin/rlmesh_capi.dll`. Those binaries are gitignored.

For a consumer Unreal project, unpack into `YourProject/Plugins/RLMesh/Source/ThirdParty/rlmesh/` the same way after copying the plugin (next step).

## 3. Install the plugin in your Unreal project

1. Copy `integrations/unreal/RLMesh/` into `YourProject/Plugins/RLMesh/` (or symlink it).
2. Ensure ThirdParty is unpacked (step 2).
3. Enable **RLMesh** under Edit → Plugins.
4. In your **game** module’s `.Build.cs`, add `"RLMeshUnreal"` to `PublicDependencyModuleNames` and set `bEnableExceptions = true` (required by `rlmesh.hpp`).

The runtime module is named `RLMeshUnreal`, not `RLMesh`, because Unreal’s `<MODULE>_API` export would collide with the C library’s `RLMESH_API`.

## 4. Host identity (optional, for stable Serving keys)

Loopback invites include `#keys` from a session **HostIdentity**. Generate one static identity once (with `mise run setup:python`):

```bash
python integrations/unreal/demos/reach_target/app.py --init-identity
```

That writes `%LOCALAPPDATA%/rlmesh/unreal_host_identity` (or the platform equivalent) and `integrations/unreal/.env`. Rebuild or relaunch the editor so LogRLMesh shows `Loaded RLMESH_HOST_IDENTITY` at startup. Without it, Serving `#keys` change each Play.

## 5. Verify without the Unreal Editor

```bash
mise run test:cxx
```

On Windows this runs the C ABI smokes, the Unreal `RLMeshGameThreadEnv` unit test, and the Gymnasium `native_gate` against both the C++ gate and the Unreal game-thread gate. The full Unix C++ example matrix remains CI’s job.

## Related

- Plugin README (components, Serving, demos): [`integrations/unreal/RLMesh/README.md`](../integrations/unreal/RLMesh/README.md)
- C++ guide: [`crates/rlmesh-capi/README-cpp.md`](../crates/rlmesh-capi/README-cpp.md)
- Testing: [`docs/testing.md`](testing.md)
- Packaging: `mise run release:cxx:package`
