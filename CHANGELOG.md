# Changelog

All notable changes to RLMesh are documented here. This changelog tracks the `rlmesh` Python package on PyPI. The Rust crates are internal implementation detail and currently carry no separate user stability promise.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

The first release. RLMesh connects models to environments across process, dependency, and machine boundaries with a Gymnasium-style API. This release freezes the `rlmesh-wire-v1` protocol generation and seals the `2026.06` workflow edition; everything after 0.1.0 stays compatible with peers built from it.

### Added

- Serve Gymnasium-style environments and drive them with `reset`, `step`, `render`, and `close` over local or remote gRPC transports: `EnvServer`, `RemoteEnv`, `RemoteVectorEnv`, and `python -m rlmesh.serve` as a universal container entrypoint.
- Evaluate models locally, against a remote server, or inside a sandbox: `Model`, `RemoteModel`, and `SandboxModel`, with `run()` for whole evals and `session()`/`Session` for manual `reset`/`predict`/`step` control.
- Tag-driven IO adapters that resolve environment tags against model specs at runtime. Environments tag what they emit (`ImageTag`, `StateTag`, `TextTag`, `Split`); models declare what they consume (`Image`, `State`, `Text`, `Concat`) and produce (`Action`, `Actuator`). Conversions cover rotation accept-sets, image `fit` and `normalize` ranges, channel validation, optional cameras, frame stacking, per-actuator `clip`/`scale`/`invert`/`threshold`, and role-less actuators; `adapter.explain()` prints the chosen transforms and `adapter.advisories()` surfaces severity-tiered data-loss notes (`info` for benign hints, `caution` when the adapter substituted or fabricated model-visible data).
- Action chunking and batched prediction in the runtime. Implement the most general of four corners and the runtime derives the rest: `predict`, `predict_chunk`, `predict_batch`, `predict_chunk_batch`. The runtime owns chunk replay and the execution horizon, so one action still reaches the environment per step.
- Run observability: pass `hooks=RunHooks(...)` to `run()`/`Session.run` for per-step `StepEvent`s (action, reward, timing, lazy role-addressed reads) and episode/run boundaries, cap episodes with `max_episode_steps`/`max_episode_seconds`, and read per-episode timing off `EpisodeResult`.
- Read-only observation inspection: `Session.reader()`/`Session.read()` extract any role from a raw observation through the model's adapter, and `Session.observation_roles()`/`EnvTags.observation_roles` list the roles an env declares.
- PyTorch and JAX end to end: environments served with framework tensors (`EnvServer(env, framework="torch", device="cuda:0")`), `rlmesh.torch` and `rlmesh.jax` factory, model, and sandbox classes, and DLPack-native `Tensor` transport with zero-copy NumPy, Torch, and JAX backends.
- Isolated sandboxes: rebuild an environment identically in a container (`SandboxEnv` with grouped `SandboxBuild`/`SandboxRuntime` config, including `SandboxRuntime.user` for writable bind mounts) or run a prebuilt image. A bare tagless image name resolves against local Docker images and never silently falls into a source build.
- Declared construction parameters: `EnvFactory.params = ParamSpec(Param(...))` validates `make()` arguments, `enumerate_variants()` lists a factory's concrete sub-environments, and `describe()`/`describe_json()`/`python -m rlmesh._describe` emit a JSON metadata envelope without constructing anything.
- A live debug viewer: pass `view="terminal"`, `view="http:9000"`, or `view="both"` to `run()`/`session()`, or configure `rlmesh.View(...)`. It is best-effort and never breaks an eval.
- `rlmesh.sanitize_metadata()` coerces a third-party sim's rich `info` objects into wire-safe metadata, and connection failures name the address and the likely fix instead of a bare transport error.
- Client timeouts: `RemoteEnv`, `RemoteVectorEnv`, and `RemoteModel` accept `connect_timeout_seconds` and `request_timeout_seconds`, so a hung peer fails with a `TimeoutError` instead of blocking forever.
- Fail-loud argument handling: a parameter that cannot be honored (`instruction=` on a served model, an unknown `vectorization_mode`, a conflicting CLI flag) raises with a directive error instead of being silently ignored.
- Self-explanatory predict failures: when a model's `predict` raises, the error is annotated with the adapter-assembled input signature (key → dtype/shape), both in local `run()` loops and across the wire from a served model, so a shape mismatch names what the model was actually handed.
- Build identity: `rlmesh.__build__` and `rlmesh version` report the commit-stamped workflow edition, so two builds sharing a package version stay distinguishable.
- Negotiated workflow editions content-pinned to the sealed `2026.06` edition spec, exact-match `rlmesh-wire-v1` protocol generation, and a per-lane `NEXT_STEP` autoreset contract for vector environments.

[0.1.0]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0
