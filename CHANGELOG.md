# Changelog

All notable changes to RLMesh are documented here. This changelog tracks the `rlmesh` Python package on PyPI. The Rust crates are internal implementation detail and currently carry no separate user stability promise.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

The first release. RLMesh connects models to environments across process, dependency, and machine boundaries with a Gymnasium-style API. This release freezes the `rlmesh-wire-v1` protocol generation and seals the `2026.06` workflow edition; everything after 0.1.0 stays compatible with peers built from it.

### Added

- Serve Gymnasium-style environments and drive them with `reset`, `step`, `render`, and `close` over local or remote gRPC transports: `EnvServer`, `RemoteEnv`, `RemoteVectorEnv`, and `python -m rlmesh.serve` as a universal container entrypoint.
- Evaluate models locally, against a remote server, or inside a sandbox: `Model`, `RemoteModel`, and `SandboxModel`, with `run()` for whole evals and `session()`/`Session` for manual `reset`/`predict`/`step` control.
- Tag-driven IO adapters that resolve environment tags against model specs at runtime. Environments tag what they emit (`ImageTag`, `StateTag`, `TextTag`, `Split`); models declare what they consume (`Image`, `State`, `Text`, `Concat`) and produce (`Action`, `Actuator`). Conversions cover rotation accept-sets, custom rotation encodings (`CustomEncoding`, wire-portable in published specs), image `fit` and `normalize` ranges, channel validation, optional cameras, frame stacking, per-actuator `clip`/`scale`/`invert`/`threshold`, and role-less actuators; `adapter.explain()` prints the chosen transforms and `adapter.advisories()` surfaces severity-tiered data-loss notes (`info` for benign hints, `caution` when the adapter substituted or fabricated model-visible data).
- Action chunking and batched prediction in the runtime. Implement the most general of four corners and the runtime derives the rest: `predict`, `predict_chunk`, `predict_batch`, `predict_chunk_batch`. The runtime owns chunk replay and the execution horizon, so one action still reaches the environment per step. Any unbatched corner may accept a trailing `context` argument carrying the episode's `episode_id` and `episode_seed`, locally and across the wire. Grouped predict requests spanning several environments fuse into one batched forward pass when the model's batched corner is the corner a direct predict would run, so grouping never changes model behavior; `Model.allow_fusion = False` opts out a batched forward that assumes fixed-size, single-environment batches.
- Seeded evals: `run(seeds=[...])` gives each episode its own seed (and sets the episode count), threading it to the environment's `reset(seed=)` and to the model as episode context; `EpisodeResult.seed` records what each episode ran with.
- Run observability: pass `hooks=RunHooks(...)` to `run()`/`Session.run` for per-step `StepEvent`s (action, reward, timing, lazy role-addressed reads) and episode/run boundaries, cap episodes with `max_episode_steps`/`max_episode_seconds`, and read per-episode timing off `EpisodeResult`.
- Read-only observation inspection: `Session.reader()`/`Session.read()` extract any role from a raw observation through the model's adapter, and `Session.observation_roles()`/`EnvTags.observation_roles` list the roles an env declares.
- PyTorch and JAX end to end: environments served with framework tensors (`EnvServer(env, framework="torch", device="cuda:0")`), `rlmesh.torch` and `rlmesh.jax` factory, model, and sandbox classes, and DLPack-native `Tensor` transport with zero-copy NumPy, Torch, and JAX backends.
- Isolated sandboxes: rebuild an environment identically in a container (`SandboxEnv` with grouped `SandboxBuild`/`SandboxRuntime` config, including `SandboxRuntime.user` for writable bind mounts) or run a prebuilt image. A bare tagless image name resolves against local Docker images and never silently falls into a source build.
- Declared construction parameters: `EnvFactory.params = ParamSpec(Param(...))` validates `make()` arguments, `enumerate_variants()` lists a factory's concrete sub-environments, and `describe()`/`describe_json()`/`python -m rlmesh._describe` emit a JSON metadata envelope without constructing anything; a model envelope reports the predict `corners` the class actually defines, and `python -m rlmesh._describe --check IMAGE` validates a built image's rlmesh labels locally, the way the platform probe will, before pushing.
- A live debug viewer: pass `view="terminal"`, `view="http:9000"`, or `view="both"` to `run()`/`session()`, or configure `rlmesh.View(...)`. It is best-effort and never breaks an eval.
- Record eval runs to a portable bundle for upload: `rlmesh.Recorder` accumulates per-episode metrics with `add()` (from any `RunResult`) or `capture()` (live `RunHooks` on the session path) and writes an `rlmesh.result.v1` folder or zip with `export()`. On the session path it records the same sources the live viewer offers -- the env's `render()` frame plus every declared image role -- each to its own AV1 `.mp4`, encoded in process (pure Rust, no ffmpeg or extra dependency; `Recorder(fps=, quality=)` set playback rate and size/fidelity); env-produced video files are carried into the bundle by path. Capture is best-effort: a source that fails to read or encode is dropped with a warning and never aborts the eval.
- `rlmesh.sanitize_metadata()` coerces a third-party sim's rich `info` objects into wire-safe metadata, and connection failures name the address and the likely fix instead of a bare transport error.
- Client timeouts: `RemoteEnv`, `RemoteVectorEnv`, and `RemoteModel` accept `connect_timeout_seconds` and `request_timeout_seconds`, so a hung peer fails with a `TimeoutError` instead of blocking forever.
- Fail-loud argument handling: a parameter that cannot be honored (`instruction=` on a served model, an unknown `vectorization_mode`, a conflicting CLI flag) raises with a directive error instead of being silently ignored.
- Self-explanatory predict failures: when a model's `predict` raises, the error is annotated with the adapter-assembled input signature (key → dtype/shape), both in local `run()` loops and across the wire from a served model, so a shape mismatch names what the model was actually handed.
- Managed-platform sign-in from the CLI: `rlmesh login` (device flow), `rlmesh logout`, `rlmesh whoami`, and `rlmesh registry login` (registers the bundled `docker-credential-rlmesh` helper, so docker fetches a fresh short-lived token per pull/push), with named profiles (`rlmesh profile use`/`list`/`remove`, `--profile`, `RLMESH_PROFILE`) and credentials held in the OS keychain.
- Build identity: `rlmesh.__build__` and `rlmesh version` report the commit-stamped workflow edition, so two builds sharing a package version stay distinguishable.
- Runtime hook events `ObservationEmittedEvent` and `ActionReceivedEvent` carry `raw_observation` / `raw_action` alongside the transformed payload, so hooks can record both the pre- and post-transform leaves per step.
- `StepCompletedEvent` and reset `ObservationEmittedEvent`s carry the env's per-step / reset `infos` map, so hooks see dense rewards, success flags, and task diagnostics as they happen rather than only `final_info` at episode end. Under `NEXT_STEP` autoreset the roll step's `infos` land on the new episode's first `ObservationEmittedEvent`, not on the old episode's `StepCompletedEvent`.
- `EpisodeStartedEvent` and `EpisodeCompletedEvent` carry the episode's `seed` (`None` when the runtime sent no per-episode seed, e.g. an env-owned `NEXT_STEP` autoreset), so hooks can record it from episode start.
- Gym vector envs served with `num_envs > 1` report each autoreset episode's seed as `info["seed"]`, derived from the lane's last explicit seed, on both the registry `make_vec` path and the factory fan-out path.
- Negotiated workflow editions content-pinned to the sealed `2026.06` edition spec, exact-match `rlmesh-wire-v1` protocol generation, and a per-lane `NEXT_STEP` autoreset contract for vector environments.

## [0.1.0-rc.6] - 2026-08-26

### Added

- Runtime hook events `ObservationEmittedEvent` and `ActionReceivedEvent` carry `raw_observation` / `raw_action` alongside the transformed payload, so hooks can record both the pre- and post-transform leaves per step.
- `StepCompletedEvent` and reset `ObservationEmittedEvent`s carry the env's per-step / reset `infos` map, so hooks see dense rewards, success flags, and task diagnostics as they happen rather than only `final_info` at episode end. Under `NEXT_STEP` autoreset the roll step's `infos` land on the new episode's first `ObservationEmittedEvent`, not on the old episode's `StepCompletedEvent`.
- `EpisodeStartedEvent` and `EpisodeCompletedEvent` carry the episode's `seed` (`None` when the runtime sent no per-episode seed, e.g. an env-owned `NEXT_STEP` autoreset), so hooks can record it from episode start.
- Gym vector envs served with `num_envs > 1` report each autoreset episode's seed as `info["seed"]`, derived from the lane's last explicit seed, on both the registry `make_vec` path and the factory fan-out path.

## [0.1.0-rc.5] - 2026-08-20

### Added

- `python -m rlmesh._describe --check IMAGE` validates a built image's rlmesh labels locally, the way the platform probe will, before pushing: the describe envelope must parse cleanly with no `error` badges, and packaged checkpoint declarations must be well-formed (DNS-label names, a `uri` per entry, a single default). Failures exit nonzero, so it can gate a push in CI.
- `RLMESH_ALLOW_REMOTE_SHUTDOWN=1` opts a served environment into honoring the remote `shutdown` RPC, so an orchestrator that queues env workloads can tell a finished worker to exit and free its slot. The variable can only enable remote shutdown, never disable a programmatic `allow_remote_shutdown=True`.
- Driver telemetry records each loop iteration's wall clock (`runner.round`), bracketing the full predict -> step -> transform body, so subtracting the per-op rows yields the driver's own residual overhead.

### Changed

- Batched leaf-slab encode and decode on the wire parallelizes past 64KB, speeding up large batched observation and action payloads.

## [0.1.0-rc.4] - 2026-08-19

### Added

- Seeded evals. `run(seeds=[...])` gives each episode its own seed and sets the episode count unless `max_episodes` is given; the seed reaches the environment's `reset(seed=)` and rides to the model as episode context, locally and across the wire. Any unbatched predict corner (`predict`, `predict_chunk`) may accept a trailing `context` argument carrying `episode_id` and `episode_seed`; batched corners never receive episode context. `EpisodeResult.seed` records what each episode ran with.
- Record eval runs to a portable bundle for upload: `rlmesh.Recorder` accumulates per-episode metrics with `add()` (from any `RunResult`) or `capture()` (live `RunHooks` on the session path) and writes an `rlmesh.result.v1` folder or zip with `export()`. On the session path it records the env's `render()` frame plus every declared image role, each to its own AV1 `.mp4` encoded in process (pure Rust, no ffmpeg or extra dependency; `Recorder(fps=, quality=)` set playback rate and size/fidelity). Capture is best-effort: a source that fails to read or encode is dropped with a warning and never aborts the eval.
- Managed-platform sign-in from the CLI: `rlmesh login` (RFC 8628 device flow), `rlmesh logout`, `rlmesh whoami` (with server-side identity verification), and `rlmesh registry login`, which registers the bundled `docker-credential-rlmesh` helper for the platform registry so docker requests a fresh short-lived token on every pull and push instead of storing a static password. AWS-style named profiles (`rlmesh profile use`/`list`/`remove`, `--profile`, `RLMESH_PROFILE`) select platform and identity; credentials live in the OS keychain with a 0600 file fallback, keyed by profile so two profiles can hit one platform as different users.
- Custom rotation encodings survive spec publication: a `CustomEncoding` arm serializes to the wire as a `module:callable` reference, or as a `<local>` marker for an in-process callable -- the published spec stays showable and validatable, while execution stays host-side.
- Model describe envelopes report `corners`: the predict corners the class actually defines, introspected from the code that ships rather than declared, so a packaging claim like batching support can be checked against it.
- Step telemetry records hook transform round-trips, so time spent in `RunHooks` transforms is visible per step.

### Changed

- `RunResult.success_rate` warns once when no episode in the run reported a task-outcome signal (Gymnasium `info["is_success"]`/`info["success"]`): the rate is then the `terminated` fallback, which counts any terminal state as a success.

## [0.1.0-rc.3] - 2026-07-06

### Added

- Grouped predicts fuse across environments. When the control plane groups predict requests, a model with a batched corner (`predict_batch` or `predict_chunk_batch`) serves the whole group in one forward pass. Fusion engages only when that batched corner is exactly the corner a direct predict would run, so grouping never changes which of a model's predict methods executes or how it chunks; a fused failure keeps the model error's recoverable flag and annotates each group with its own input signature.
- `Model.allow_fusion` (default `True`). Set it to `False` when a batched forward assumes fixed-size, single-environment batches, such as a shape-pinned `jit` trace or per-batch statistics; grouped predicts then run one route at a time.
- Adapter advisories carry a severity tier: `Advisory.severity` is `"info"` for benign hints and `"caution"` when the adapter substituted or fabricated model-visible data, the tier an eval harness may want to hard-fail on.

### Fixed

- Batched and chunked prediction now works in local `run()` evals. `run()` drives the same native runtime loop as a served model, so `predict_batch`, `predict_chunk`, and `predict_chunk_batch` activate locally instead of only on the served wire path.

[0.1.0]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0
[0.1.0-rc.6]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0-rc.6
[0.1.0-rc.5]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0-rc.5
[0.1.0-rc.4]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0-rc.4
[0.1.0-rc.3]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0-rc.3
