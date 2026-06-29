# Changelog

All notable changes to RLMesh are documented here. This changelog tracks the `rlmesh` Python package on PyPI. The Rust crates are internal implementation detail and currently carry no separate user stability promise.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-rc.2] - Unreleased

The pre-1.0 breaking wave: rc.2 freezes the wire contract and finalizes the authoring API. It does not interoperate with rc.1, so upgrade every environment, model, and runtime together.

### Breaking changes

- Observation and action values now travel as spec-directed bytes, and protocol generation (`rlmesh-wire-v1`) is matched by exact equality. An rc.1 peer cannot connect to an rc.2 peer.
- Roles are required on `ImageTag`, `StateTag`, `TextTag`, `Image`, and `Text`, passed as the first argument: `ImageTag(IMAGE_PRIMARY)`, `Text(INSTRUCTION)`.
- Adapter spec classes use their final names. Rename the old input and action classes in your imports, for example `ImageInput` to `Image`, `StateComponent` to `Concat`, `ActionComponent` to `Actuator`, `ActionLayout` to `Action`.
- `action_horizon=` on `run()` and `session()` is now `execution_horizon=`, and `execute_horizon` is gone from `Action`. The horizon is a runtime setting, not part of the spec.
- `predict_chunk(self, observation)` no longer takes a horizon; return your native chunk and the runtime replays it one action per step. An autoregressive head may add `execution_horizon: int = 1` to decode exactly that many.
- `on_reset` is removed from `Model` and `Session`. Use `on_episode_end`, which fires at every episode boundary on both the local and served paths; a subclass `reset()` is wired to it.
- Sandbox config is grouped. Pass `build=SandboxBuild(...)` and `runtime=SandboxRuntime(...)` instead of flat kwargs: `SandboxEnv("CartPole-v1", build=SandboxBuild(packages=["gymnasium==1.3.0"]))`.

### Added

- Action chunking and batched prediction in the runtime. Implement the most general of four corners and the runtime derives the rest: `predict`, `predict_chunk`, `predict_batch`, `predict_chunk_batch`. The runtime owns chunk replay and the execution horizon, so one action still reaches the environment per step.
- `session()` and `Session` for manual `reset`/`predict`/`step` control alongside `run()`, plus `Session.reader()` and `Session.read()` for read-only, role-addressed views of an observation through the model's adapter.
- PyTorch and JAX environments served with framework tensors: `EnvServer(env, framework="torch", device="cuda:0")`, with `rlmesh.torch` and `rlmesh.jax` factory, model, and sandbox classes. A model's `device` moves observations before `predict`.
- Declared construction parameters: `EnvFactory.params = ParamSpec(Param(...))` validates `make()` arguments and adds a `Vector` type. `enumerate_variants()` lists a factory's concrete sub-environments.
- `describe()`, `describe_json()`, and `python -m rlmesh.describe` emit a JSON metadata envelope for a factory or model without constructing it.
- More adapter conversions: rotation accept-sets, image `fit`, `normalize` ranges, channel validation, optional cameras, per-actuator `clip`, role-less actuators, and `adapter.advisories()` for data-loss notes. A newer environment contract also parses against an older core and fails only when a model input needs the missing piece.
- A live debug viewer: pass `view="terminal"`, `view="http:9000"`, or `view="both"` to `run()`/`session()`, or configure `rlmesh.View(...)`. It is best-effort and never breaks an eval.
- Sampling and ergonomic helpers on the space wrappers.

### Changed

- The adapter engine and chunk replay moved into the Rust core for vectorized, stateful serving. The Python authoring path is unchanged.
- `Actuator` `scale`, `invert`, and `threshold` apply on both sides and compose in order, so a model can bridge its output convention at the boundary.

### Fixed

- Sandbox startup waits longer before timing out, so slow environments such as LIBERO task suites start reliably.
- Debug viewer fixes for SSH terminals and observation-fed frames.

[0.1.0-rc.2]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0-rc.2

## [0.1.0-rc.1] - 2026-06-17

First release candidate for 0.1.0. RLMesh connects models to environments across process, dependency, and machine boundaries with a Gymnasium-style API.

### Added

- Serve Gymnasium-style environments and drive them with `reset`, `step`, `render`, and `close` over local or remote gRPC transports.
- DLPack-native `Tensor` transport with zero-copy NumPy, Torch, and JAX backends (#3).
- Run served environments locally or rebuild them identically in an isolated sandbox (`SandboxEnv`) (#8).
- Evaluate models locally, against a remote server, or inside a sandbox (`Model`, `RemoteModel`, `SandboxModel`) (#11).
- Tag-driven IO adapters that resolve environment tags against model specs at runtime (#9).
- Negotiated workflow editions content-pinned to the `2026.06` edition spec (#2).
- Per-lane `NEXT_STEP` autoreset contract for vector environments (#7).

### Changed

- Hardened the public Python API, space wrappers, and transport for the stable surface (#5).

[0.1.0-rc.1]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0-rc.1
