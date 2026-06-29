# Editions

A workflow edition is a named behavioral contract for RLMesh workflow semantics. The base edition (`YYYY.MM`) identifies one spec document in this section; prerelease and local builds append a cohort suffix so moving builds fail closed unless both sides are from the same cohort. Exactly one edition governs a session, chosen during handshake.

```{note}
In 0.1.0-rc.1 the active cohort is `2026.06-0.1.0-rc.1`; the bare `2026.06` edition seals at the final 0.1.0.
```

Editions answer a different question than the protocol generation. The protocol generation (`rlmesh-wire-v1`) names the wire shape: which services, messages, and fields exist. The edition names what a conforming interaction over that shape _means_: lifecycle, ordering, episode accounting, and error semantics.

## Negotiation

The client sends every edition it can operate under in `HandshakeRequest.supported_workflow_editions`. The server intersects that offer with its own supported set and selects the highest mutual edition. A matching suffixed cohort wins over its sealed fallback; if prerelease cohorts differ, peers can only interoperate through a sealed edition that both sides explicitly advertise. The selection is returned in `HandshakeResponse.selected_workflow_edition`. The runtime currently supports a single edition and refuses any other; making the selected edition drive runtime behavior is on the roadmap (see {doc}`../compatibility`).

- An empty intersection means `compatible = false`. The response lists the server's supported editions for diagnostics, but there is no second round trip because the client's offer was already complete.
- Servers accept only editions they explicitly support. A server never accepts an unknown edition on the assumption that it is probably compatible; forward compatibility lives in the client's offer set, not in server leniency.

## Edition vs. Capability vs. Bug Fix

Most development never touches the edition:

- A change to the meaning of an existing, conforming interaction mints a new edition. This is rare, and breaking semantic changes batch into at most one new edition per release.
- A new addition that is ignorable or detectable, such as a new RPC, a new field, or an opt-in behavior, is a capability or a plain feature. No edition.
- An implementation that deviates from the governing spec document has a bug. Fixing it needs no edition.

## Lifecycle: Provisional, Then Sealed

An edition is **provisional** while no stable release has shipped it: prerelease builds use exact release-cohort names such as `2026.06-0.1.0-rc.1`, and local source builds use exact `dev.<git>` cohort names. This prevents accidental interoperability between moving builds that have not had stable-release scrutiny. The first stable release that ships an edition **seals** the bare `YYYY.MM` name permanently: the spec document becomes immutable (enforced by checksum), and any later semantic change mints a new edition.

```{mermaid}
stateDiagram-v2
    [*] --> Provisional: prerelease / local cohort
    Provisional --> Sealed: first stable release seals the bare YYYY.MM name
    Sealed --> Sealed: spec immutable, valid indefinitely
```

`2026.06` uses provisional cohorts through the 0.1 beta and release-candidate series and seals at v0.1.0. After sealing it remains valid indefinitely; a new edition is minted only by a deliberate semantic redesign, never on a schedule.

## Support Window

Sealing freezes an edition's spec by checksum. The support window is a forward-compatibility guarantee on the roadmap (see {doc}`../compatibility`), binding at 1.0, not today: every later release keeps offering and accepting a sealed edition, including betas for a later edition, and sealed editions are never pruned. A provisional cohort, which no stable release has sealed, may change or be dropped and interoperates only with the same cohort unless both sides implement and advertise a sealed fallback.

## Enforcement

`rlmesh.toml` records the base edition, current official release cohort, supported editions, and each edition's `status` (`provisional` or `sealed`) plus `spec` path. A sealed edition also records `sealed_in` and `spec_sha256`. `scripts/check_rlmesh_policy.py` verifies sealed spec checksums, rejects provisional editions in stable releases, and checks that prerelease cohorts match the workspace SemVer. Local dev cohorts are generated at build time and are not committed to the manifest.

## 2026.06

- **Status:** provisional through prerelease cohorts; seals as `2026.06` when v0.1.0 ships
- **Protocol generation:** `rlmesh-wire-v1`

This document is the behavioral contract identified by the workflow edition base `2026.06`. When a handshake selects `2026.06` or an exact prerelease/dev cohort for this base, both peers commit to the semantics below for the rest of the session.

The protobuf files for `rlmesh-wire-v1` define the wire shape; this document defines what conforming use of that shape means. Where an implementation and this document disagree, the implementation has a bug. A change that alters the meaning of an interaction described here mints a new edition; it does not amend this one once sealed.

### Session Establishment

A session is one successful `Handshake` followed by one `Join` stream. Conforming clients do not open `Join` before completing a handshake with `compatible = true`.

- The client states its wire protocol in `protocol_generation` and **declares** every edition it can operate under in `supported_workflow_editions`. The handshake only _declares_ editions; it does not select one.
- The server replies `compatible = true` when the protocol generations are compatible. Generation is the only handshake-level decision (a hard, full-restart break). The **edition is not decided at the handshake**: only the runtime sees every participant, so it reconciles the session edition (the floor across env, model, and runtime, recorded on `ResolveAdapter`/`ConfigureEnv`). A generation-compatible peer that shares no edition handshakes fine and then fails at the runtime's floor.
- On `compatible = false`, `error_message` explains the (generation) failure and `supported_workflow_editions` lists the server's editions for diagnostics. The session ends; there is no renegotiation round.
- Capability maps are advisory in both directions: a present key means the named feature is available, an absent key means it is not. Capabilities gate optional features; they never change the meaning of the interactions defined here.
- The environment handshake response carries the `EnvContract` (spaces, metadata, render mode, `num_envs`) whenever `compatible = true`. The contract is fixed for the life of the session.

### Environment Workflow (`rlmesh.env.v1.EnvService`)

#### Ordering

Requests on a `Join` stream are processed strictly in arrival order, one at a time. Every request produces exactly one response, carrying the originating `request_id`. There is no reordering and no silent dropping: a request the server cannot satisfy produces an in-band `EnvError` response.

#### Vectorization

The served environment is a fixed-width vector of `num_envs` sub-environments, established by the handshake contract. Observations and actions are batched `SpaceValue`s covering all sub-environments; `rewards`, `terminated_mask`, and `truncated_mask` carry one entry per sub-environment, in index order (mask bytes are per-sub-environment flags; nonzero means set).

#### Reset

`Reset` (re)starts all sub-environments and must precede the first `Step` of a session. `seeds` is either empty (server defaults apply) or carries one seed per sub-environment. `ResetRequest.episode_ids` carries the authoritative episode ids the runtime minted for the lanes being started; the env adopts them (it never mints) and tags its tracked episodes with them. The `ResetResponse` carries the initial batched observation and `infos`; it has no episode-id fields, since ids are reported back on `Step` in `completed_episodes`.

#### Step

`Step` applies one batched action to all sub-environments. `StepRequest.episode_ids` carries the runtime's authoritative per-lane ids (the env adopts a rolled id when it autoresets a lane under NEXT_STEP). The response carries the next batched observation, per-sub-environment rewards and termination/truncation masks, shared `infos`, and `completed_episodes` metadata for episodes that ended on this step. The env reports lifecycle via `completed_episodes`, and each `EpisodeMetadata` carries the runtime-minted `episode_id` (alongside its `env_index`) that the env adopted at the episode's start; the runtime remains the sole id authority, the env only echoes the ids it was handed.

#### Episode Accounting

A tracked episode per sub-environment begins at `Reset`. Each `Step` accrues to it. When a sub-environment reports terminated or truncated, its episode completes: metadata (id, seed, step count, cumulative reward, termination cause, timing, `final_info`) is delivered once in that response's `completed_episodes`. The id in that metadata is the runtime-minted id the env adopted at the episode's start.

The edition itself does not restart sub-environments on termination. Whether a terminated sub-environment continues to accept steps is the served environment's autoreset behavior, conveyed through observations and `infos`; only an explicit `Reset` re-establishes tracked episodes.

#### Render and Close

`Render` returns a PNG frame when the environment supports the contract's render mode, and no frame otherwise. `Close` ends the session: the server responds once (with metadata for episodes it finalizes) and then closes the `Join` stream. No further requests on that stream are answered.

#### Timeouts and Errors

A positive `timeout_ms` on a request is a server-enforced deadline; expiry produces an in-band `EnvError` with code `TIMEOUT`. Errors are reported in-band as `EnvError` responses with a code, message, and `is_recoverable`: a recoverable error leaves the session usable for further requests; a non-recoverable error means the client must abandon the session.

#### Shutdown

`Shutdown` is a unary request to terminate the endpoint itself, distinct from closing a session. The server may refuse it (`accepted = false`), and endpoints may disable remote shutdown entirely.

### Model Workflow (`rlmesh.model.v1.ModelService`)

The model service reverses the dialing direction: the runtime connects to a served model participant. Session establishment is as above (without an environment contract in the response).

#### Adapter Lifecycle

The model-facing protocol is keyed by two ids: `env_id` (a connected env container, UUIDv7, minted by the runtime) and `episode_id` (one rollout, UUIDv7, minted fresh on every reset). There is no positional lane/slot on the wire. An adapter is the resolved model-side context for one `env_id` (one adapter per env, no sharing), carried in `AdapterContext`.

Four ops, all keyed by `env_id`:

- **`ResolveAdapter`** must precede the first `Predict` for an env and fixes that env's `EnvContract` (idempotent upsert). A `Predict` on an unresolved env produces a `ModelError` with code `NOT_CONFIGURED`.
- **`Predict`** infers through the resolved adapter (below).
- **`ResetAdapter`** drops per-episode adapter state (frame-stack buffers): with `episode_ids` set, exactly those keys; empty, all of the env's episode state. The adapter stays resolved. Because UUIDv7 ids never repeat, a missed/late `ResetAdapter` only leaks memory; it can never alias a new episode, so it is GC, not a correctness gate.
- **`ReleaseAdapter`** removes the adapter entirely (implies reset-all). `Close` requests graceful shutdown of the whole participant.

#### Predict

A `Predict` request carries the `AdapterContext` (`env_id`, `session_id`, `request_id`), a batched observation encoded per the env's observation space, and an ordered `episode_ids` vector, the self-describing batch: row `i` of the observation belongs to `episode_ids[i]`, decided per tick. The model keys all per-episode state by `episode_id`, never by position-across-ticks; it lazy-seeds an episode's state on its first appearance and evicts it on `ResetAdapter`. The response mirrors the request's context and carries an `actions` list whose first frame (`actions[0]`) is this step's action batch (one row per `episode_id`), inheriting the request's `episode_ids` order.

`actions[1..]`, when present, are pre-split future-step action frames for open-loop action chunking. The runtime buffers them and applies one per step **without** re-calling the model, then re-plans when the buffer drains or any lane's episode rolls. This is scheduling data (out of scope; see below), not a change to the per-step contract: `actions[0]` still carries exactly one action row per `episode_id`, and a chunk-unaware runtime that uses only `actions[0]` re-plans every step and stays correct. How many frames a model emits is governed by `execution_horizon`, which the runtime pins on `ResolveAdapter` (the replay horizon `h`; `0`/`1` = no chunking). The horizon is a runtime decision, not part of the model spec.

#### Ordering and Errors

Every request on a `Join` stream is answered exactly once, mirroring `request_id`; failures are in-band `ModelError` responses, and `is_recoverable` has the same meaning as on the environment service.

A served model **may** pipeline requests: process them concurrently and emit responses in completion order rather than strict arrival order. This is a scheduling choice (out of scope for the edition; see below), never a wire change. Responses still mirror `request_id`, so a client demuxes them by id. A server that pipelines advertises the `rlmesh.model.concurrent_predict.v1` capability at handshake; its absence means responses arrive in arrival order. **Per-env ordering is always preserved**: for a given `env_id`, the model applies `ResolveAdapter`, `Predict`, `ResetAdapter`, and `ReleaseAdapter` in the order the client sent them. A whole-session `Close` drains after every outstanding request. Requests for _different_ envs may complete in either order.

### Value Conformance

When a handshake selects `2026.06`, both peers agree on how observation and action values are checked against the spaces declared in the `EnvContract`. A value's dtype is always coerced to its declared dtype before transport, so a delivered value's dtype always equals the space the peer negotiated; a peer never receives a per-message dtype. A conformance warning may accompany a delivered value; a warning never withholds or alters the value beyond the coercion already applied.

#### Structural conformance

Regardless of the validation policy, a value is rejected when:

- a `Dict` value is missing a declared key;
- a `Box` or `MultiDiscrete` value has the wrong rank or shape, a `MultiBinary` value the wrong shape, or a `Tuple` value the wrong arity;
- a `Discrete` element is outside its domain, or a `MultiDiscrete` element is outside its own per-element domain;
- any element of a numeric value is `NaN`. `NaN` is never a member of any space, so it is rejected even when other elements are merely out of bounds.

The recoverability of a structural rejection depends on which side produced the value. A rejected action is a recoverable `EnvError` (`InvalidAction`): the action is not delivered, but the session stays usable. A rejected observation is non-recoverable (`Internal`), because the serving side produced a value that violates its own contract. In both cases the offending value is never delivered.

#### Range conformance

For `Box` bounds and `Text` charset and length, the serving side carries a validation policy:

- **warn** (the default): the deviation is delivered and reported as a conformance warning.
- **strict**: the deviation is rejected, like a structural deviation.
- **off**: the range, charset, and length checks are skipped; structural conformance still applies.

A `Box` element is in range when it satisfies its declared bounds; an infinite or absent bound imposes no constraint on that side, so `+/-inf` is in range against a matching infinite bound. A `Text` value conforms when its length is within `[min_length, max_length]` (counted in characters) and, when the charset is non-empty, every character is in it. Observations and actions share one policy and both default to **warn**.

#### Enforcement

The serving side validates each observation it produces before transport and each action it receives before delivering it to the environment. Structural deviations are rejected regardless of policy; range deviations follow the policy.

#### Dtype conformance

A value is coerced to its declared dtype before transport:

- a value carried as a native RLMesh tensor must already have the declared dtype;
- a value supplied as a host array or sequence (for example NumPy) is coerced to the declared dtype, except that a float supplied for an integer dtype is rejected unless every element is finite, integral, and representable in the target integer dtype's range, with no silent truncation and no out-of-range wraparound;
- a floating-point value supplied for a floating-point dtype is coerced to the declared dtype (a narrowing such as `float64` to `float32` may lose precision).

#### Conformance warnings

A conformance warning is reported in-band under the reserved `rlmesh.conformance.warning` key in the info map returned by `reset` and `step`, at most once per `(deviation kind, value path)` per session. Conformance warnings are advisory and never make a session non-recoverable; the path format is advisory.

### Out of Scope

This edition does not constrain transport security or authentication, client retry or reconnection policy, scheduling and batching strategy, performance, or the meaning of individual capability names. Those evolve freely without minting an edition.

## Where next

- {doc}`../compatibility` covers the support-window and runtime-enforcement roadmap.
- {doc}`../user-guide/adapters` covers how value conformance shows up when authoring env and model adapters.
