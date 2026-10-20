# Unity v0.2 release checklist

**Status: Experimental; Unity support not yet verified on IL2CPP.** Unity support
must not be claimed until all three D-44 gates below pass on the release
artifacts. There is no Unity installation or licence on the implementation host;
the .NET tests cannot substitute for these gates.

Use Linux x64 with Unity Hub, a valid Editor licence, and the **Linux Build
Support (IL2CPP)** module. The only Editor version pin is
[`integrations/unity/ProjectVersion.txt`](../integrations/unity/ProjectVersion.txt);
the preparation script copies it into the test project's `ProjectSettings`.
The test framework is pinned to `1.4.6` in the project manifest. Record the exact
Editor version, OS, scripting backend, git commit and core UPM SHA-256 with the
logs and XML results. Other desktop targets need their own verification before
support is claimed.

## Prepare

Obtain the separately built `com.rlmesh.rlmesh` UPM package for the same revision
and release cohort. Its D-40 layout must contain `Runtime/RLMesh.dll` and
`Runtime/Plugins/x86_64/linux/librlmesh_capi.so`, with their importers. Do not use
a NuGet package or a dev-cohort native library with a release-cohort Python
wheel. Install the repo's pinned tools; preparation uses pinned Zig 0.13.0.

From the repository root:

```sh
mise run dev:heavy -- python integrations/unity/Tests~/prepare.py \
  --core-package /absolute/path/to/com.rlmesh.rlmesh.tgz
export UNITY_EDITOR=/absolute/path/to/Unity/Editor/Unity
export UNITY_PROJECT="$PWD/integrations/unity/Tests~/Project"
mkdir -p "$HOME/.cache/rlmesh-unity/results"
```

`--core-package` also accepts an unpacked package directory. Preparation copies
the existing core, integration and Reach Target sample into ignored project
folders, then compiles `Assets/Tools/step-client` against the supplied native
library using `zig c++ -target x86_64-linux-gnu.2.17` with static C++ runtimes,
so it can execute inside the Ubuntu-based GameCI Editor container. Nothing here
builds the core UPM package.

## Gate 1: Mono Editor reload

```sh
mise run dev:heavy -- "$UNITY_EDITOR" -batchmode -nographics \
  -projectPath "$UNITY_PROJECT" -runTests -testPlatform EditMode \
  -testFilter RLMesh.Unity.Verification.ReloadTests.ServeReloadServeAgain \
  -testResults "$HOME/.cache/rlmesh-unity/results/reload.xml" \
  -logFile "$HOME/.cache/rlmesh-unity/results/reload.log"
```

Expect a passing test and `Unity Editor reload OK: served before and after domain
reload`. A native client completes a real episode through `MainThreadEnv` before
and after reload, asserting callbacks ran on the Editor main thread. It finishes
the real terminal barrier,
asserts the reload lock is released, reloads the domain, verifies a changed
domain identity and open lifetime, then serves again. There must be no unsafe
unload diagnostics, crash or hang. Keep both log and XML.

## Gate 2: failed barrier keeps reload deferred

```sh
mise run dev:heavy -- "$UNITY_EDITOR" -batchmode -nographics \
  -projectPath "$UNITY_PROJECT" -runTests -testPlatform EditMode \
  -testFilter RLMesh.Unity.Verification.ReloadTests.FailedBarrierDefersReloadAndRetryReleasesIt \
  -testResults "$HOME/.cache/rlmesh-unity/results/failed-barrier.xml" \
  -logFile "$HOME/.cache/rlmesh-unity/results/failed-barrier.log"
```

The native client drives a deliberately wedged served `IEnv.Step` on a native
thread, leaving the Editor free to tick. A 100 ms barrier attempt must return
`false`, log `Pending()`, and keep the actual assembly-reload lock held. The test
requests a real script reload and observes 30 responsive Editor ticks with the
same domain identity. After it releases the callback, retry must complete and
the pending reload must run. Expect the passing test and
`Unity failed-barrier reload OK: deferred, responsive, released, reloaded`.
No crash, reload while blocked, silent unlock or Editor hang is acceptable.

Also repeat interactively with script compilation, Enter/Exit Play Mode (domain
reload enabled and disabled), the menu retry and normal quitting. While the
test env is blocked, verify the UI still repaints, the diagnostic names pending
native work, and reload-requiring paths stay deferred. Exiting Play Mode must
remain a normal scene teardown; entry with domain reload disabled must preserve
the domain and open lifetime. A failed quit barrier must allow **Quit anyway**
(with normal save prompts) and an explicit **Force unlock reload** must warn of
unsafe unload. Quitting the process is allowed even when shutdown is incomplete. A callback that blocks
the Unity main thread itself freezes the Editor; this fixture deliberately
wedges the native callback to test the reload barrier while the UI can run.
Record interactive results; a flag-only coordinator test is insufficient.

Run the PlayMode example test too:

```sh
mise run dev:heavy -- "$UNITY_EDITOR" -batchmode -nographics \
  -projectPath "$UNITY_PROJECT" -runTests -testPlatform PlayMode \
  -testResults "$HOME/.cache/rlmesh-unity/results/playmode.xml" \
  -logFile "$HOME/.cache/rlmesh-unity/results/playmode.log"
```

## Gate 3: built Linux IL2CPP player against Python

```sh
mise run dev:heavy -- "$UNITY_EDITOR" -batchmode -nographics -quit \
  -projectPath "$UNITY_PROJECT" -buildTarget StandaloneLinux64 \
  -executeMethod RLMesh.Unity.Verification.BuildSmoke.Build \
  -rlmeshBuildPath "$PWD/build/unity/Smoke.x86_64" \
  -logFile "$HOME/.cache/rlmesh-unity/results/il2cpp-build.log"
```

The build method creates a minimal `Assets/Smoke.unity` scene, explicitly selects
IL2CPP, .NET Standard compatibility and high managed stripping, then builds
`StandaloneLinux64`. Verify the build log says IL2CPP, and the output contains
the core native plugin and the player's IL2CPP data, rather than a Mono build.

Use the matching Python release wheel, or build a release-cohort non-editable
test environment from this revision:

```sh
RLMESH_RELEASE_BUILD=1 UV_PROJECT_ENVIRONMENT=target/unity-python \
  mise run dev:heavy -- uv sync --locked --no-editable --no-default-groups --group csharp
UV_PROJECT_ENVIRONMENT=target/unity-python uv run --no-sync python \
  integrations/unity/Tests~/player_smoke.py \
  --player build/unity/Smoke.x86_64 --episodes 3
```

Expect `Unity client OK: 3 episodes`, `Unity IL2CPP smoke OK: 3 episodes` and
`Unity IL2CPP player gate passed`, with player exit code 0. The Python client
checks the offset observation, Discrete actions, negative-distance rewards and
episode boundaries. It signals completion only after receiving the last
response; the player exits via the terminal shutdown preflight. Startup and
shutdown have bounded timeouts. Retain the build, logs and client output.

## Dispatch-only GameCI

Add repository secrets `UNITY_LICENSE`, `UNITY_EMAIL` and `UNITY_PASSWORD`, then
dispatch **Unity release gates** (`.github/workflows/unity.yml`) on the integration
revision with `core_package_url` (HTTPS UPM tarball) and its
`core_package_sha256`. The URL is fetched without authentication; use a release
asset or a short-lived artifact URL and never put credentials in workflow input.
If any of `UNITY_LICENSE`, `UNITY_EMAIL` or `UNITY_PASSWORD` is absent, the
workflow fails before checkout or Unity execution.

The user approved this dispatch-only GameCI mechanism and its
`depot-ubuntu-24.04-8` runner on 2026-10-20. The workflow has only
`workflow_dispatch`, one job, no matrix, runner
`depot-ubuntu-24.04-8`, timeout 45 minutes and concurrency
`unity-${{ github.ref }}` with `cancel-in-progress: true`. The pinned GameCI test
runner executes EditMode and PlayMode; the pinned builder builds Linux IL2CPP;
both actions use the same immutable Linux IL2CPP Editor image digest, and the
test runner pins GameCI CLI `v0.1.72`. Update the digest when changing the Editor
version pin. `checks: write` allows the runner to publish test results;
the built player is then driven by Python. It does not run on pushes, pull
requests or schedules. Download `unity-release-gates` evidence and record all
three gate results before changing the Experimental support wording.
