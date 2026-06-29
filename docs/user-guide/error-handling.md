# Error handling

Most of what can go wrong in an eval surfaces at one of two seams: adapter resolution (a model spec that does not line up with an env's tags and spaces) or the transport (a remote env or served model that is unreachable, slow, or fails mid-episode). RLMesh reports each as a distinct exception so you can tell a wiring mistake from a runtime fault and recover the right way.

This page maps the exceptions {func}`~rlmesh.adapters.resolve`, `serve`, `predict`, and the eval loop ({func}`~rlmesh.run` / {func}`~rlmesh.session`) raise, what causes each, and how a run behaves when a connection drops or a model crashes part-way through an episode. For misaligned adapters specifically, {doc}`debugging` covers `describe()` and the `read`/`reader` inspection path.

## The exception families

Two families cover almost everything.

Resolution failures raise {exc}`~rlmesh.adapters.AdapterResolutionError`, a subclass of `ValueError`. It comes from {func}`~rlmesh.adapters.resolve`, {func}`~rlmesh.adapters.resolve_from_contract`, {func}`~rlmesh.adapters.tag`, the read API, and the adapter resolution that `run`/`session` do on connect. It always points at the offending leaf and what it expected, and it fires before a single step runs.

Runtime failures come from the native core and reach Python through a small exception hierarchy plus a few standard built-ins. The native module defines `RLMeshException` (a subclass of `RuntimeError`) as the base, with `ProtocolException` and `EnvironmentException` beneath it. An environment that reports a fault while serving a request raises `EnvironmentException`. Transport faults, timeouts, and bad arguments map to the standard `ConnectionError`, `TimeoutError`, and `ValueError` instead, so ordinary `except` clauses catch them without importing anything RLMesh-specific.

```{note}
`RLMeshException`, `ProtocolException`, and `EnvironmentException` live in the
native module (`rlmesh._rlmesh`). `EnvironmentException` is the one the env path
raises today; `ProtocolException` is reserved for protocol-level faults, and the
current boundary surfaces generation/handshake mismatches as `RuntimeError`
rather than that type. Catch `RLMeshException` to cover the whole family at once.
```

## What each call raises

### `resolve()` and `resolve_from_contract()`

Both raise {exc}`~rlmesh.adapters.AdapterResolutionError` when the model spec cannot be bridged to the env's tags and spaces: a required role with no `optional` fallback, a declared channel mismatch, an upscale without `allow_upscale`, an aspect mismatch without `fit`, an unsupported `resample` or `dtype`, an impossible encoding conversion, an unknown field on a known leaf, or a join-time class/width/encoding/range disagreement. {func}`~rlmesh.adapters.resolve_from_contract` adds two of its own: the contract carries no adapter tags, or those tags are not serializable JSON.

```python
import rlmesh.adapters as adapt

try:
    adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
except adapt.AdapterResolutionError as exc:
    print(exc)  # names the leaf and what it expected
```

A spec that references a `Custom` input by `entrypoint=` also raises here unless you pass `resolve(..., trust_entrypoints=True)`. The conversion-policy table in {doc}`adapters/reference` decides which conversions apply silently, warn, or fail.

### `run()`, `session()`, and `Session.predict()`

The eval loop resolves the adapter on the first connect, so a spec/env mismatch raises {exc}`~rlmesh.adapters.AdapterResolutionError` before any episode begins. A few more checks run at the same point:

- Pointing {func}`~rlmesh.run` at a vector endpoint (`num_envs > 1`) raises `ValueError`. The per-episode loop reads scalar reward and termination, so a vector env is rejected up front rather than crashing deep in the step loop. Serve it as a single env, or drive it with `RemoteVectorEnv` directly.
- An env that publishes adapter tags paired with a model whose `spec` is `None` raises {exc}`~rlmesh.adapters.AdapterResolutionError`: pass a `ModelSpec`, or `rlmesh.NO_ADAPTER` if the model adapts its own observations.
- A target that is neither an env, an `EnvFactory`, a remote handle, nor an address string raises `TypeError`.

Once running, `predict` and `step` surface whatever the env or model raises. For a local model, `predict` runs in-process, so a bug in your prediction function propagates as its own native Python exception. For a served model, the server maps a handler that declines a request to `RuntimeError` (`"model error: ..."`).

### `serve()`

{class}`~rlmesh.EnvServer` validates published tags against the env's spaces at construction (for a scalar env), so a bad tag raises {exc}`~rlmesh.adapters.AdapterResolutionError` at startup instead of when the first model connects. Asking for `device=` on a numpy env, or combining an explicit `address` with `host`/`port`/`path`, raises `ValueError`. A bind that fails surfaces as `RuntimeError` (`"server error: ..."`).

A served model resolves its adapter once per env, at the configure step rather than at connect, so a spec/env mismatch fails route configuration loudly with {exc}`~rlmesh.adapters.AdapterResolutionError` rather than predicting wrongly.

## Symptom, cause, fix

| Symptom                                                        | Cause                                                           | Fix                                                                          |
| -------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `AdapterResolutionError: ... no usable counterpart`            | a model input or actuator role the env never tags               | tag the role on the env, mark the input `optional`, or drop it from the spec |
| `AdapterResolutionError: env contract carries no adapter tags` | resolving from a contract on an untagged env                    | serve with `tags=` or {func}`~rlmesh.adapters.tag`; see {doc}`adapters`      |
| `AdapterResolutionError: ... has spec=None`                    | a tagged env paired with a model that declares no spec          | pass `spec=<ModelSpec>`, or `rlmesh.NO_ADAPTER` to opt out of adaptation     |
| `ValueError: ... reports num_envs=...`                         | {func}`~rlmesh.run` aimed at a vector endpoint                  | serve a single env, or use `RemoteVectorEnv` for the vector endpoint         |
| `ValueError: Endpoint ... serves N environments`               | `RemoteEnv` connected to a multi-env endpoint                   | connect with `RemoteVectorEnv` instead (see {doc}`remote-clients`)           |
| `ConnectionError`                                              | the env/model endpoint is unreachable or the connection dropped | confirm the address and that the server is up; re-dial a fresh client        |
| `TimeoutError`                                                 | a connect or request exceeded its deadline                      | check the endpoint is serving; retry the operation                           |
| `EnvironmentException`                                         | the env reported NotReady, Busy, Internal, Crashed, or Closed   | reset the env, or inspect the env-side logs for the underlying fault         |
| `RuntimeError: model error: ...`                               | a served model handler declined the request                     | check the model's prediction code against the obs it actually receives       |
| `ValueError: device=... requires a torch/jax ...`              | `device=` on a numpy env/model                                  | drop `device=`, or set `framework="torch"` / `"jax"`                         |
| `ImportError: Failed to import _rlmesh native module`          | the compiled extension is missing                               | reinstall the wheel for your platform                                        |

## Connection loss and crashes mid-episode

The eval loop does not retry a failed step. When the env connection drops or a served model fails while an episode is in flight, the call (`predict` or `step`) raises, and {func}`~rlmesh.run` propagates it. Before the exception leaves `run`, the loop still runs cleanup: it ends the open episode (firing a stateful model's `on_episode_end`), calls `on_close`, and closes the session. The in-progress episode is discarded, and `run` raises rather than returning a partial {class}`~rlmesh.RunResult`.

```{mermaid}
flowchart TD
    A[run starts an episode] --> B[predict, then step]
    B -->|ok, not done| B
    B -->|env or model fault| E[predict/step raises]
    B -->|terminated or truncated| C[record EpisodeResult]
    C -->|more episodes| A
    C -->|done| R[return RunResult]
    E --> F[end open episode: on_episode_end, on_close, close]
    F --> G[re-raise the exception]
```

A few consequences follow from this.

A transport fault during an established run is reported as `ConnectionError`, not as a recoverable retry. RLMesh classifies transport conditions internally (a server that is still binding is treated as transient), but the public Python clients do not reconnect a dropped session for you. To recover, construct a fresh client and start a new run.

A served model handler that raises becomes a `RuntimeError` carrying the handler's message. The env stays up, so you can fix the model and re-dial without restarting the env server.

```{caution}
Per-step request timeouts and connect timeouts exist on the native client but are
not exposed through `RemoteEnv` / `RemoteVectorEnv` or the `run`/`session` loop
today. A step against an env that hangs will block. Bound it at the env: serve
with `ServeOptions(idle_timeout_seconds=...)` so an idle server stops on its own,
and supervise the process. See {doc}`streaming` for the session lifecycle.
```

## Recovering cleanly

Wrap a run in a normal `try`/`except` and decide per family. Resolution errors are wiring bugs you fix in the spec or tags; transport errors call for a re-dial; an `EnvironmentException` usually means resetting or restarting the env.

```python
import rlmesh
import rlmesh.adapters as adapt

try:
    result = model.run(env, seeds=range(100))
except adapt.AdapterResolutionError as exc:
    raise SystemExit(f"adapter mismatch, fix the spec or tags: {exc}")
except (ConnectionError, TimeoutError) as exc:
    ...  # re-dial a fresh client and retry the run
except rlmesh._rlmesh.EnvironmentException as exc:
    ...  # the env reported a fault; reset or restart it
```

Using `session()` as a context manager (or `run()`, which closes for you) guarantees the connection and any managed model are released even when an episode raises, so the next attempt starts clean.

## Where next

- {doc}`debugging` — `describe()`, the `read`/`reader` inspection path, and join advisories for a misaligned adapter.
- {doc}`adapters` and {doc}`adapters/reference` — the resolution rules and the conversion policy behind every {exc}`~rlmesh.adapters.AdapterResolutionError`.
- {doc}`evaluation` — how {func}`~rlmesh.run` and {func}`~rlmesh.session` drive episodes and what a {class}`~rlmesh.RunResult` reports.
- {doc}`serving-environments` and {doc}`remote-clients` — readiness, health, and the client side of the transport.
