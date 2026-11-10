# Unreal Engine: local RLMesh setup

How to build the C/C++ library from this checkout, prepare the Unreal plugin, and drive a served env from Python. This guide does **not** create an Unreal project; you drop the plugin into a project you already have.

The short plugin install notes live in [`integrations/unreal/RLMesh/README.md`](../integrations/unreal/RLMesh/README.md). The C++ API is in [`crates/rlmesh-capi/README-cpp.md`](../crates/rlmesh-capi/README-cpp.md). PoC status: [Unreal Gym PoC](../vault/rlmesh/integrations/unreal/Unreal%20Gym%20PoC.md).

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

## 4. Try Reach Target (happy path)

### One-time

```bash
python integrations/unreal/reach_target_app.py --init-identity
```

Writes `%LOCALAPPDATA%/rlmesh/unreal_host_identity` and `integrations/unreal/.env`. Rebuild the plugin once so LogRLMesh shows `Loaded RLMESH_HOST_IDENTITY` at editor start.

In the demo project (`RLmeshSim`), wire the level (editor Python console, or `restart_editor.py --python-script`):

```text
py "G:/_UnrealProjects/RLmeshSim/Scripts/setup_reach_target_scene.py"
```

### Each run

1. Press **Play** in Unreal. LogRLMesh: `Serving Unreal-ReachTarget-v0 on 127.0.0.1:7447#…/env` (keys match `.env`).
2. From the rlmesh checkout (`mise run setup:python`):

```bash
python integrations/unreal/reach_target_app.py --verbose
```

Or open a Gymnasium-style client (still one EnvServer — no second server):

```bash
python integrations/unreal/reach_target_gym.py
```

**Which actor moves:** the **owner** of `URLMeshReachTargetComponent` (in RLmeshSim, the Agent `ARLMeshReachDemoActor`). Level props without the component do not move. If the owner root was **Static**, BeginPlay upgrades it to **Movable** so steps can change transforms.

## PoC contract (Reach Target)

| Item | Contract |
|------|----------|
| Claim | **RLMesh env in Unreal (Gym-style API)** — `reset` / `step` / spaces over `EnvServer`. Not a registered Gymnasium env id; `reach_target_gym.py` is a thin `rlmesh.numpy.RemoteEnv` helper. |
| Env id | `Unreal-ReachTarget-v0` |
| Observation | `Box(-inf, inf, shape=(3,), float32)` — remaining world XYZ gap to the goal |
| Action | `Box([-1, 1]^3)` — direction; normalized × `StepSize` (default 50). Near-zero stays. Discrete `0/1/2` still accepted as ±X / stay. |
| Reward | `-distance / max(TargetRange, ‖TargetOffset‖)` |
| Terminated | distance ≤ `StepSize / 2` |
| Truncated | `steps >= MaxSteps` (default 200) |
| `info` | `agent_xyz`, `target_xyz` (float32[3]), `distance` (float), `steps` (int) |
| Address | `127.0.0.1:7447#<keys>/env` from static `RLMESH_HOST_IDENTITY` |
| Edition | Plugin `rlmesh_capi` must match this Python’s workflow edition (Windows: `RLMESH_RELEASE_BUILD=1`) |
| Timing | Each step waits ≥ one game-thread frame; wall-clock follows PIE frame rate |

Demo checklist: fixed `MaxSteps` on the component, optional lower frame time / fixed timestep in the project for shorter wall-clock episodes.

## 5. Verify without the Unreal Editor

```bash
mise run test:cxx
```

On Windows this runs the C ABI smokes, the Unreal `RLMeshGameThreadEnv` unit test, and the Gymnasium `native_gate` against both the C++ gate and the Unreal game-thread gate. The full Unix C++ example matrix remains CI’s job.

Dirty-tree Python and C builds only negotiate when both speak the same sealed edition; on Windows the task defaults `RLMESH_RELEASE_BUILD=1` for that cohort (see [local-dev](local-dev.md)).

## Related

- Plugin README: [`integrations/unreal/RLMesh/README.md`](../integrations/unreal/RLMesh/README.md)
- C++ guide: [`crates/rlmesh-capi/README-cpp.md`](../crates/rlmesh-capi/README-cpp.md)
- Testing: [`docs/testing.md`](testing.md)
- Packaging: `mise run release:cxx:package`
