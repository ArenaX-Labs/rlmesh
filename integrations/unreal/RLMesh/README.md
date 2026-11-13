# RLMesh for Unreal Engine

An Unreal Engine 5 plugin that serves game state as an [RLMesh](https://github.com/ArenaX-Labs/rlmesh) environment. While the game plays, any RLMesh model, in Python, Rust, C or C++, on this machine or across the network, can connect and call `reset` and `step`.

The plugin is **standalone**: drop it into any Unreal project with the ThirdParty C package. It does not depend on a sample project (or any other repo). Sample Unreal projects may depend on this plugin and on the Python package from this repository. Add a component, Press Play, connect to Serving.

It runs on Win64 and Linux and uses the prebuilt RLMesh C/C++ library. The `.uplugin` does not pin an engine minor: it is intended for UE **5.6.x** and **5.7.x**, and is expected on **5.8**, but in-repo checks do not run UnrealBuildTool.

## Install

1. Copy this `RLMesh` folder into your project's `Plugins/` folder.
2. Download the RLMesh C/C++ package for your platform from the [releases](https://github.com/ArenaX-Labs/rlmesh/releases) (`rlmesh-<version>-x86_64-pc-windows-msvc.tar.gz` or `rlmesh-<version>-x86_64-unknown-linux-gnu.tar.gz`) and unpack it into `Plugins/RLMesh/Source/ThirdParty/rlmesh/`:

   ```bash
   tar -xzf rlmesh-<version>-<platform>.tar.gz --strip-components=1 -C Plugins/RLMesh/Source/ThirdParty/rlmesh
   ```

   That folder then holds `include/`, `lib/` and, on Windows, `bin/`. The build stages the library beside your game binary.

3. Enable the plugin (Edit > Plugins > RLMesh), and add `"RLMeshUnreal"` to `PublicDependencyModuleNames` in your game module's `.Build.cs`. Set `bEnableExceptions = true` in that `.Build.cs` too: RLMesh reports failures as C++ exceptions.

## Try the example

Add an **RLMesh Reach Target** component to any actor and press Play. The log shows `Serving … on 127.0.0.1:7447#…/env`: the env `env` in a session the game hosts on port 7447. **Steps move that component’s owner** (if its root was Static, BeginPlay upgrades it to Movable). Each episode moves the owner in world XYZ toward a target: a hidden random offset in a cube by default, or set **Target Actor** to a level actor. Observation is the remaining XYZ gap (`Box[3]`); action is `Box([-1, 1]^3)` × **Step Size**; reward is minus normalized Euclidean distance. Reset/step `info` also carries `agent_xyz`, `target_xyz`, `distance`, and `steps`.

**Chase Point** (`URLMeshChasePointComponent`, env id `Unreal-ChasePoint-v0`) is the planar sibling: obs/action `Box[2]` (XY), Z locked, optional **Goal Actor**. The goal wanders on the floor for **Goal Move Steps** (not toward the agent, inside **Goal Wander Radius**), then freezes.

**Avoid Hazards** (`URLMeshAvoidHazardsComponent`, env id `Unreal-AvoidHazards-v0`): XY reach with `NumHazards` cubes; obs `Box[2+2N]`; collision terminates (`hit_hazard`). Hazards spawn at Play.

**Collect Coins** (`URLMeshCollectCoinsComponent`, env id `Unreal-CollectCoins-v0`): XY collect `NumCoins` gold spheres; obs `Box[3]` (nearest coin + coins left); terminates when all collected (`success`). Coins spawn at Play.

This is an **RLMesh env in Unreal (Gym-style API)** — not a Gymnasium registry id. Drive it with `rlmesh.run` or `rlmesh.numpy.RemoteEnv`. Optional Python helpers live under `integrations/unreal/demos/` in this repo: Press Play with the matching component, then pass the Serving address (`--address` / `RLMESH_UNREAL_ADDR`).

Loopback invites include `#keys` from a session **HostIdentity**. Generate one static identity once (from this checkout, with `mise run setup:python`):

```bash
python integrations/unreal/demos/reach_target/app.py --init-identity   # once per machine
# rebuild the plugin once, open any Unreal project with the plugin, add a component, Press Play
python integrations/unreal/demos/reach_target/app.py                  # uses integrations/unreal/.env
```

The plugin loads that identity into `RLMESH_HOST_IDENTITY` at module startup (and again at BeginPlay), so Serving `#keys` stay stable across Play.

On Windows, the plugin’s ThirdParty `rlmesh_capi` and this Python must share a workflow edition — build the C package with `RLMESH_RELEASE_BUILD=1`.

## Serve your own env

Subclass `URLMeshEnvComponent` in C++ and give it the four members of any RLMesh env:

```cpp
#include "RLMeshEnvComponent.h"

#include "MyEnvComponent.generated.h"

UCLASS(ClassGroup = (Game), meta = (BlueprintSpawnableComponent))
class UMyEnvComponent : public URLMeshEnvComponent
{
	GENERATED_BODY()

protected:
	rlmesh::spaces::Space ObservationSpace() const override { return rlmesh::spaces::Box(-1.0, 1.0, {3}); }
	rlmesh::spaces::Space ActionSpace() const override { return rlmesh::spaces::Discrete(4); }

	rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed,
								 const std::optional<rlmesh::InfoDict>& Options) override
	{
		// Put the world back to a start state (Seed / Options optional).
		return {Observe(), /*info=*/{}};
	}

	rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action) override
	{
		// Apply (*Action) to the world on this game thread.
		return {Observe(), Reward, bTerminated, bTruncated, /*info=*/{}};
	}
};
```

`ResetEnv` and `StepEnv` run on the game thread, from the component's tick, so they may read and change actors freely. A step takes at least one engine tick; raise **Serving Max FPS** (default 120) or pass reset option `serving_max_fps` / env `RLMESH_UNREAL_MAX_FPS` so lockstep can run faster than the editor default. Set **Address** on the component to serve elsewhere (port 0 picks a free port, which `GetServedAddress` returns) and **Env Id** to name the env.

Throw `rlmesh::RecoverableEnvironmentException` from `ResetEnv` or `StepEnv` to fail one request while the client keeps its session; any other exception ends the session. The component catches every RLMesh exception at the Unreal boundary and logs it under `LogRLMesh`.

See the [C++ guide](../../../crates/rlmesh-capi/README-cpp.md) for spaces, values and the rest of the API.

## Layout

```
RLMesh.uplugin
Source/RLMeshUnreal/              the RLMeshUnreal runtime module
  Public/RLMeshEnvComponent.h     serve an env from an actor component
  Public/RLMeshReachTargetComponent.h   3D reach example env
  Public/RLMeshChasePointComponent.h    2D chase-point example env
  Public/RLMeshAvoidHazardsComponent.h  2D avoid-hazards example env
  Public/RLMeshCollectCoinsComponent.h  2D collect-coins example env
  Public/RLMeshGameThreadEnv.h    hands server calls to the game thread (no Unreal headers)
  Public/RLMeshIncludes.h         includes rlmesh.hpp as a third-party header
Source/ThirdParty/rlmesh/         the unpacked RLMesh package (not committed)
Tests/                            a standalone test of RLMeshGameThreadEnv.h
```

The module is named `RLMeshUnreal` rather than `RLMesh` because Unreal defines a `<MODULE>_API` export macro per module, and `RLMESH_API` is already the RLMesh C library's own.
