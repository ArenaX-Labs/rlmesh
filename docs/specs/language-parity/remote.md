# Remote clients

Part of the [language parity spec](../language-parity.md). Status: Draft —
reviewed round 5. Draft: C# column pending review. Rules are in [rules](rules.md); paths use its
abbreviations (`py/`, `pyrs/`, `rs/`, `capi/`, `h`, `hpp`), plus `grpc/` =
`crates/rlmesh-grpc/src/`.

Python reference: `rlmesh.RemoteEnv`, `rlmesh.RemoteVectorEnv`,
`rlmesh.RemoteModel` (Stable, `py/_native.py:25,36,46`), with bodies in
`py/_client/` (`_remote_base.py`, `_remote_env.py`, `_remote_vector_env.py`,
`_remote_model.py`, `_endpoint.py`) and the native core in `pyrs/client.rs`
and `pyrs/model.rs:990-1050`. This area owns the three clients, their options
structs and client timeouts (R-27). `Session`, `SessionOptions`, `EnvTarget`
and `ModelTarget` are [Running](running.md)'s; `Transport` and `VectorSeed`
are [Env](env.md)'s; `Space`, `SpaceSpec` and batched values are
[Spaces](spaces.md)'; `EnvContract` is Env's; editions are [Core](core.md)'s.

## Shape

Connect, drive one episode by hand, then bind a served model to the same env.

```python
with rlmesh.RemoteEnv("tcp://127.0.0.1:5555", request_timeout_seconds=5.0) as env:
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(policy(obs))
    frame = env.render()
    model = rlmesh.RemoteModel("tcp://127.0.0.1:6000")
    with model.session(env) as s:
        obs, info = s.reset(seed=1)

with rlmesh.RemoteVectorEnv("tcp://127.0.0.1:5556") as venv:
    obs, info = venv.reset(seed=[0, 1])
    obs, rewards, terminated, truncated, info = venv.step(actions)
```

```rust
let mut env = RemoteEnv::new(RemoteEnvOptions {
    address: Some("tcp://127.0.0.1:5555".into()),
    request_timeout: Some(Duration::from_secs(5)),
    ..Default::default()
})?;
let (obs, info) = env.reset(Some(0), None)?;
let (obs, reward, terminated, truncated, info) = env.step(policy(&obs))?;
let frame: Option<Value> = env.render(None)?;

let model = RemoteModel::new(RemoteModelOptions::default().address("tcp://127.0.0.1:6000"))?;
let mut s = model.session(&mut env, SessionOptions::default())?;
let (obs, info) = s.reset(Some(1), None)?;
drop(s);
env.close()?;

let mut venv = RemoteVectorEnv::new(RemoteEnvOptions::default().address("tcp://127.0.0.1:5556"))?;
let (obs, info) = venv.reset(Some(vec![0, 1].into()), None)?;
let (obs, rewards, terminated, truncated, info) = venv.step(actions)?;

// Async twins (R-16): same parameters, same borrows.
let mut aenv = RemoteEnv::new_async(RemoteEnvOptions::default().address("tcp://127.0.0.1:5555")).await?;
let (obs, info) = aenv.reset_async(Some(0), None).await?;
```

```cpp
rlmesh::RemoteEnv env({.address = "tcp://127.0.0.1:5555", .request_timeout = std::chrono::seconds(5)});
auto [obs, info] = env.reset(0);
auto [obs2, reward, terminated, truncated, info2] = env.step(policy(obs));
std::optional<rlmesh::Value> frame = env.render();

rlmesh::RemoteModel model({.address = "tcp://127.0.0.1:6000"});
auto s = model.session(env);

rlmesh::RemoteVectorEnv venv({.address = "tcp://127.0.0.1:5556"});
auto [vobs, vinfo] = venv.reset(rlmesh::VectorSeed{0, 1});
```

```c
RLMeshRemoteEnvOptions eo = RLMESH_REMOTE_ENV_OPTIONS_INIT;
eo.address = "tcp://127.0.0.1:5555";
eo.has_request_timeout_seconds = true;
eo.request_timeout_seconds = 5.0;
RLMeshRemoteEnv* env = NULL;
if (rlmesh_remote_env_new(&eo, &env) != RLMESH_OK) { /* rlmesh_last_error_message() */ }

int64_t seed = 0;
RLMeshValue *obs = NULL, *info = NULL;
rlmesh_remote_env_reset(env, &seed, NULL, &obs, &info);
double reward;
bool terminated, truncated;
RLMeshValue *obs2 = NULL, *info2 = NULL;
rlmesh_remote_env_step(env, action, &obs2, &reward, &terminated, &truncated, &info2);

rlmesh_remote_env_close(env); /* observe close errors */
rlmesh_remote_env_free(env);  /* always required */
```

```csharp
using var env = new RemoteEnv(new RemoteEnvOptions
{
    Address = "tcp://127.0.0.1:5555",
    RequestTimeout = TimeSpan.FromSeconds(5),
});
var (obs, info) = env.Reset(0);
var (obs2, reward, terminated, truncated, info2) = env.Step(Policy(obs));
Value? frame = env.Render();

var model = new RemoteModel(new RemoteModelOptions { Address = "tcp://127.0.0.1:6000" });
using (var s = model.Session(env))
{
    var (sobs, sinfo) = s.Reset(1);
}

using var venv = new RemoteVectorEnv(new RemoteEnvOptions { Address = "tcp://127.0.0.1:5556" });
var (vobs, vinfo) = venv.Reset(new long[] { 0, 1 });
var (vobs2, rewards, vterminated, vtruncated, vinfo2) = venv.Step(actions);

// Async twins (R-16): the constructor's twin is a static factory (RMT-24).
using var aenv = await RemoteEnv.ConnectAsync(new RemoteEnvOptions { Address = "tcp://127.0.0.1:5555" });
var (aobs, ainfo) = await aenv.ResetAsync(0, cancellationToken: token);
```

`address` is an options field, not a positional parameter, because R-6 puts
every defaulted parameter in the options struct and R-7 does not list the
client constructors (RMT-2).

## Parity table

`Option<Duration>` / `std::optional<std::chrono::duration<double>>` /
`has_*_seconds` + `double` per R-3. "same" means the `RemoteEnv` cell with
`remote_vector_env` / `RemoteVectorEnv` substituted.

### Constructor keywords: `RemoteEnvOptions`, `RemoteModelOptions`

`RemoteEnv` and `RemoteVectorEnv` share `RemoteEnvOptions` (identical
keywords). `RemoteModelOptions` has the same fields without
`workflow_edition`. C layout and `_INIT` are below the tables.

| Python keyword (default)                      | Rust field                              | C field                                                            | C++ field                                                      | C# property (RMT-24)                                                           |
| --------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `address: str \| None = None`                 | `address: Option<String>`               | `const char* address` (NULL = None)                                | `std::optional<std::string> address`                           | `string? Address { get; set; }`                                                |
| `host: str \| None = None`                    | `host: Option<String>`                  | `const char* host`                                                 | `std::optional<std::string> host`                              | `string? Host { get; set; }`                                                   |
| `port: int \| None = None`                    | `port: Option<u16>` (RMT-11)            | `bool has_port; uint16_t port`                                     | `std::optional<uint16_t> port`                                 | `ushort? Port { get; set; }` (RMT-11)                                          |
| `path: str \| None = None`                    | `path: Option<String>`                  | `const char* path`                                                 | `std::optional<std::string> path`                              | `string? Path { get; set; }`                                                   |
| `transport: Transport \| None = None`         | `transport: Option<Transport>`          | `bool has_transport; RLMeshTransport transport`                    | `std::optional<Transport> transport`                           | `Transport? Transport { get; set; }`                                           |
| `connect_timeout_seconds = None`              | `connect_timeout: Option<Duration>`     | `bool has_connect_timeout_seconds; double connect_timeout_seconds` | `std::optional<std::chrono::duration<double>> connect_timeout` | `TimeSpan? ConnectTimeout { get; set; }` (RMT-4: `TimeSpan.Zero` = no timeout) |
| `request_timeout_seconds = None`              | `request_timeout: Option<Duration>`     | `bool has_request_timeout_seconds; double request_timeout_seconds` | `std::optional<std::chrono::duration<double>> request_timeout` | `TimeSpan? RequestTimeout { get; set; }` (RMT-4)                               |
| `workflow_edition: str \| None = None`        | `workflow_edition: Option<String>`      | `const char* workflow_edition`                                     | `std::optional<std::string> workflow_edition`                  | `string? WorkflowEdition { get; set; }`                                        |
| `api_key: str \| None = None`                 | `api_key: Option<String>` (RMT-22)      | `const char* api_key` (appended after ABI 5)                       | `std::optional<std::string> api_key`                           | `string? ApiKey { get; set; }` (RMT-22)                                        |
| `tls_ca_file: str \| PathLike \| None = None` | `tls_ca_file: Option<PathBuf>` (RMT-22) | `const char* tls_ca_file` (appended after ABI 5)                   | `std::optional<std::string> tls_ca_file`                       | `string? TlsCaFile { get; set; }` (RMT-22)                                     |

- Address grammar (RMT-22): the scheme picks the transport. `host:port`,
  `tcp://host:port` and `http://host:port` are plaintext HTTP/2;
  `unix:///path` is a Unix socket (env and model clients alike);
  `https://host[:port]` is TLS with ALPN `h2`, port 443 by default, and a
  path other than empty or `/`, a query, a fragment or userinfo is
  ValueError. `transport` stays `tcp` / `unix`; there is no TLS value.
- `api_key` / `tls_ca_file` (RMT-22): the key travels as
  `authorization: Bearer <key>`; `None` reads `RLMESH_API_KEY` for an
  `https://` address only, `""` sends none. `tls_ca_file` trusts only that
  PEM bundle; without it a TLS dial trusts the host's trust store alone, and
  only when that store yields no roots, the bundled web PKI roots alone (the
  two sets are never combined). Every address an env or model client dials,
  including a local `Model`'s address target, resolves its key this way.
- When credentials are read (RMT-22): constructors check `api_key`,
  `RLMESH_API_KEY` and `tls_ca_file` (ValueError early); every dial resolves
  them again, so a rotated key or CA bundle reaches the next
  `RemoteModel.session` in every language.
- Address helpers: one facade implementation of `normalize_connect_address`
  (`py/_client/_endpoint.py:56-127`) for every language. `address` cannot be
  empty or combined with helpers; `path` excludes `host`/`port`;
  `transport=unix` needs `path` and is refused on Windows; `path` with
  `transport=tcp` is refused (RMT-P4; Python today silently yields
  `unix://path`, `_endpoint.py:82-89`); `host`/`port` need `port`; `host`
  defaults to `127.0.0.1`; with neither `address` nor `host`/`port` (or only
  `transport=tcp`) → `address or host/port is required`
  (`_endpoint.py:90-91`), so the default options, `rlmesh_*_new(NULL, &out)`
  and C++ `RemoteEnv()` / `RemoteVectorEnv()` / `RemoteModel()` fail. All
  ValueError.
- Setters: `RemoteEnvOptions` and `RemoteModelOptions` derive `Default`
  and provide one R-8 setter per field (`address`, `host`, `path`,
  `workflow_edition` take `impl Into<String>`; `port` takes `u16`;
  `transport` takes `Transport`; `connect_timeout`, `request_timeout` take
  `impl Into<Duration>`), so the Shape's
  `RemoteModelOptions::default().address("tcp://…")` equals the struct
  literal.
- C#: `public sealed class RemoteEnvOptions` and `RemoteModelOptions`
  (R-6, D-22), every property `null` by default; `new RemoteEnv(new
RemoteEnvOptions { Address = "tcp://…" })` replaces Rust's setters (R-8).
  The defaults fail as C++'s `RemoteEnv()` does. `TimeSpan.Zero` passes
  `has_* = true, 0.0`, which RMT-4 reads as no timeout; a negative
  `TimeSpan`, `Timeout.InfiniteTimeSpan` included, is ValueError (D-35), so
  "no timeout" is `null` or `TimeSpan.Zero`, never `InfiniteTimeSpan`.
  `TlsCaFile` is a `string` path, as in C++ (no `FileInfo` overload, which
  would be an extra). A string property holding U+0000 is ValueError before
  the C call (D-36). The C struct is filled from the options at the call
  with the generated `sizeof`, which covers `api_key` and `tls_ca_file`
  (R-19's C# bullet).
- Timeouts: RMT-4. Editions: an empty or all-whitespace `workflow_edition`
  is not a declaration and resolution continues (Core, D-7;
  `py/_editions.py:139`); an edition this build cannot run is ValueError.

### `RemoteEnv`

| Python                                                        | Rust                                                                                                             | C                                                                                                                                                                                                        | C++                                                                                                                                    | C#                                                                                                                                                                                                      |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RemoteEnv(address=None, **keywords)`                         | `RemoteEnv::new(options: RemoteEnvOptions) -> Result<RemoteEnv>`; `new_async`                                    | `RLMeshStatus rlmesh_remote_env_new(const RLMeshRemoteEnvOptions* options, RLMeshRemoteEnv** out)`                                                                                                       | `explicit RemoteEnv(const RemoteEnvOptions& options = {})`                                                                             | `public RemoteEnv(RemoteEnvOptions? options = null)` (connects); `public static Task<RemoteEnv> ConnectAsync(RemoteEnvOptions? options = null, CancellationToken cancellationToken = default)` (RMT-24) |
| `.address -> str`                                             | `fn address(&self) -> &str` (canonical connect string, see Address)                                              | `RLMeshStatus rlmesh_remote_env_address(const RLMeshRemoteEnv*, RLMeshBytes* out)`                                                                                                                       | `std::string address() const`                                                                                                          | `string Address { get; }`                                                                                                                                                                               |
| `.env_id -> str`                                              | `fn env_id(&self) -> &str`                                                                                       | `RLMeshStatus rlmesh_remote_env_env_id(const RLMeshRemoteEnv*, RLMeshBytes* out)`                                                                                                                        | `std::string env_id() const`                                                                                                           | `string EnvId { get; }`                                                                                                                                                                                 |
| `.selected_workflow_edition -> str`                           | `fn selected_workflow_edition(&self) -> &str`                                                                    | `RLMeshStatus rlmesh_remote_env_selected_workflow_edition(const RLMeshRemoteEnv*, RLMeshBytes* out)`                                                                                                     | `std::string selected_workflow_edition() const`                                                                                        | `string SelectedWorkflowEdition { get; }` (read through C each time, RMT-20)                                                                                                                            |
| `.env_contract -> EnvContract`                                | `fn env_contract(&self) -> &EnvContract`                                                                         | `const RLMeshEnvContract* rlmesh_remote_env_env_contract(const RLMeshRemoteEnv*)` (borrowed)                                                                                                             | `const specs::EnvContract& env_contract() const`                                                                                       | `Specs.EnvContract EnvContract { get; }` (cached, own reference via `rlmesh_env_contract_clone`)                                                                                                        |
| `.render_mode -> str \| None`                                 | `fn render_mode(&self) -> Option<&str>`                                                                          | `RLMeshStatus rlmesh_remote_env_render_mode(const RLMeshRemoteEnv*, bool* out_has, RLMeshBytes* out)` (as `rlmesh_env_contract_render_mode`; `out` untouched when `*out_has` is false)                   | `std::optional<std::string> render_mode() const`                                                                                       | `string? RenderMode { get; }`                                                                                                                                                                           |
| `.metadata -> Metadata`                                       | `fn metadata(&self) -> &Metadata` (empty when absent)                                                            | `const RLMeshValue* rlmesh_remote_env_metadata(const RLMeshRemoteEnv*)` (borrowed dict)                                                                                                                  | `const Metadata& metadata() const`                                                                                                     | `Metadata Metadata { get; }` (empty when absent; RMT-25)                                                                                                                                                |
| `.observation_space` / `.action_space -> Space`               | `fn observation_space(&self) -> &Space`, `action_space` (cached, RMT-9)                                          | `const RLMeshSpace* rlmesh_remote_env_observation_space(const RLMeshRemoteEnv*)`, `_action_space` (borrowed, cached)                                                                                     | `const Space& observation_space() const`, `action_space`                                                                               | `Spaces.Space ObservationSpace { get; }`, `ActionSpace` (cached, same object every read, RMT-9)                                                                                                         |
| `.observation_space_spec` / `.action_space_spec`              | `fn observation_space_spec(&self) -> &SpaceSpec`, `action_space_spec`                                            | `const RLMeshSpaceSpec* rlmesh_remote_env_observation_space_spec(const RLMeshRemoteEnv*)`, `_action_space_spec`                                                                                          | `const SpaceSpec& observation_space_spec() const`, `action_space_spec`                                                                 | `Spaces.SpaceSpec ObservationSpaceSpec { get; }`, `ActionSpaceSpec`                                                                                                                                     |
| `.reset(*, seed=None, options=None) -> (obs, info)`           | `fn reset(&mut self, seed: Option<i64>, options: Option<&InfoDict>) -> Result<(Value, InfoDict)>`; `reset_async` | `RLMeshStatus rlmesh_remote_env_reset(RLMeshRemoteEnv*, const int64_t* seed, const RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info)`                                         | `std::tuple<Value, InfoDict> reset(std::optional<int64_t> seed = std::nullopt, const std::optional<InfoDict>& options = std::nullopt)` | `ResetResult Reset(long? seed = null, Metadata? options = null)` (R-7.2); `ResetAsync`                                                                                                                  |
| `.step(action) -> (obs, reward, terminated, truncated, info)` | `fn step(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)>`; `step_async`                  | `RLMeshStatus rlmesh_remote_env_step(RLMeshRemoteEnv*, const RLMeshValue* action, RLMeshValue** out_observation, double* out_reward, bool* out_terminated, bool* out_truncated, RLMeshValue** out_info)` | `std::tuple<Value, double, bool, bool, InfoDict> step(const Value& action)`                                                            | `StepResult Step(Value action)`; `StepAsync`                                                                                                                                                            |
| `.render(*, env_index=0) -> frame \| None`                    | `fn render(&mut self, env_index: Option<u64>) -> Result<Option<Value>>` (None = 0); `render_async`               | `RLMeshStatus rlmesh_remote_env_render(RLMeshRemoteEnv*, uint64_t env_index, RLMeshValue** out_frame)` (NULL on OK = None)                                                                               | `std::optional<Value> render(uint64_t env_index = 0)`                                                                                  | `Value? Render(ulong envIndex = 0)` (R-7.3); `RenderAsync`                                                                                                                                              |
| `.close() -> None`                                            | `fn close(&mut self) -> Result<()>`; `close_async`                                                               | `RLMeshStatus rlmesh_remote_env_close(RLMeshRemoteEnv*)`                                                                                                                                                 | `void close()`                                                                                                                         | `void Close()`; `CloseAsync`                                                                                                                                                                            |
| `.shutdown(reason='owner shutdown') -> bool`                  | `fn shutdown(&mut self, reason: Option<&str>) -> Result<bool>`; `shutdown_async`                                 | `RLMeshStatus rlmesh_remote_env_shutdown(RLMeshRemoteEnv*, const char* reason /* NULL = default */, bool* out_accepted)`                                                                                 | `bool shutdown(std::string_view reason = "owner shutdown")`                                                                            | `bool Shutdown(string? reason = null)` (R-7.4; `null` = `"owner shutdown"`); `ShutdownAsync`                                                                                                            |
| `with RemoteEnv(...) as env` (`__exit__` → `close()`)         | `Drop` (RMT-8)                                                                                                   | `void rlmesh_remote_env_free(RLMeshRemoteEnv*)` (RMT-8)                                                                                                                                                  | `~RemoteEnv()` wraps `_free`; move-only (R-11)                                                                                         | `IDisposable`: `Dispose()` = RMT-8; finalizer frees on an R-28 cleanup worker                                                                                                                           |
| KeyboardInterrupt in a blocked call                           | `fn cancel_handle(&self) -> ext::CancelToken`; drop the `_async` future (R-18)                                   | `RLMeshStatus rlmesh_remote_env_cancel_handle(const RLMeshRemoteEnv*, RLMeshExtCancelToken** out)` (new reference, RMT-6)                                                                                | `ext::CancelToken cancel_handle() const`                                                                                               | `Ext.CancelToken CancelHandle()`; a twin's `CancellationToken` (R-18)                                                                                                                                   |

Behaviour, all languages:

- **Construction connects** (RMT-1): dial plus handshake. The Join stream
  (the env's exclusive session slot) opens lazily on the first `reset`,
  `step` or `render` (each calls `ensure_join_stream`,
  `grpc/env/client.rs:490,521,552,679`), never on `close` (a local-only close
  when no Join is open, `:585-593`) or `shutdown` (a unary RPC, `:620-650`),
  so an idle client locks nobody out; a `step` or `render` before any
  `reset` takes the slot too. Errors: RMT-12.
- **Address**: `RemoteEnv` and `RemoteVectorEnv` report the canonical
  connect string of the dialed endpoint (`ConnectAddress` display:
  trimmed, a scheme-less `host:port` as `tcp://host:port`, a unix path
  normalized; `pyrs/client.rs:90,105`, `rs/address.rs:51-65`,
  `grpc/helpers/address.rs:65-118`), unlike `RemoteModel.address`, which is
  the normalized input verbatim.
- **Arity**: refuses `num_envs != 1` with ValueError ("Use RemoteVectorEnv
  instead", `pyrs/client.rs:112-116`).
- **Info**: `info` always carries the runtime-reserved `episode_ids`
  (one-element list) and, on `step`, `completed_episodes` (1 on the step that
  ends an episode, else 0). An env key of the same name wins, logged once
  (RMT-18; `pyrs/client.rs:147-171,517,560-564`).
- **Render** (RMT-16): the facade decodes the wire PNG to a uint8 `Tensor`
  value of shape `[H,W]`, `[H,W,3]` or `[H,W,4]` (other formats → RGBA,
  `pyrs/client.rs:800-829`); an env without a frame gives None.
- **Lane index**: `env_index` (and `ext::reset_lanes`' `env_indices`) is
  `u64` / `uint64_t`, a lane offset into the remote env like `num_envs`'s
  count (R-5), not an in-memory position. A value above `u32::MAX` (the wire
  carries `uint32`,
  `crates/rlmesh-proto/proto/rlmesh/env/v1/interaction.proto:39,72`) is
  ValueError before any I/O (RMT-P11); a lane at or beyond `num_envs` is
  the env's to refuse, as today.
- **Bounds**: `request_timeout` bounds `reset`, `step`, `render`, `shutdown`;
  `close` per RMT-4. `reset`, `step` and `render` also send it on the wire
  as `timeout_ms`, so the server enforces it too; an expired request is
  TimeoutError in every language, never the server's `TIMEOUT` reply
  (RMT-21). An accepted `shutdown` closes the handle locally
  (`grpc/env/client.rs:646-648`).
- **C#**: accessors are properties (R-4); the contract, spaces and specs
  are wrapped once at construction over new native references
  (`rlmesh_env_contract_clone`, `rlmesh_space_clone`,
  `rlmesh_space_spec_clone`), so they stay valid after the client is
  disposed, unlike C's borrowed pointers. `SelectedWorkflowEdition` is
  read through C on every get, so it reports a `Session`'s pin (RMT-20).
  An overlapping I/O call fails with the RuntimeError class from C;
  accessors never do. A call after `Dispose()` is `ObjectDisposedException`
  (D-41); after `Close()` it reaches C and is RuntimeError (RMT-13). The
  Rust-only `ext::reset_lanes` (RMT-14) has no C# form, as it has no C or
  C++ form.
- **Concurrency**: one I/O call (`reset`, `step`, `render`, `close`,
  `shutdown`) at a time per handle (`py/_client/_remote_env.py` docstring;
  `pyrs/client.rs:31-38`). Rust `&mut self` makes it compile-time; a C/C++
  overlapping or re-entrant I/O call fails with R-21's `RuntimeError` class,
  never UB. Accessors (`address`, `env_id`, `selected_workflow_edition`,
  `env_contract`, `render_mode`, `metadata`, the spaces and specs,
  `num_envs`) and `cancel_handle` may overlap any call from any thread and
  never fail with the overlap error: the handle keeps their state outside the
  per-call guard, immutable after construction except the selected edition
  (RMT-20; RMT-17, RMT-P5). Rust `RemoteEnv: Send + Sync`.
- **Closed handle**: `close` is idempotent; any other I/O call after `close`
  or an accepted `shutdown` is R-21's `RuntimeError` (RMT-13). Accessors keep
  working.
- **C ownership**: outputs are written only on `RLMESH_OK`, NULL otherwise,
  and freed by the caller (`rlmesh_value_free`, `rlmesh_bytes_free`).
  Borrowed accessor results live until `_free`, including after `_close`.
  `_free` is required after every successful `_new`, including after a failed
  `_close`; freeing during a call is the caller's bug.

### `RemoteVectorEnv`

| Python                                                                               | Rust                                                                                                                    | C                                                                                                                                                                                                                                                                  | C++                                                                                                                                       | C#                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RemoteVectorEnv(address=None, **keywords)`                                          | `RemoteVectorEnv::new(options: RemoteEnvOptions) -> Result<RemoteVectorEnv>`; `new_async`                               | `RLMeshStatus rlmesh_remote_vector_env_new(const RLMeshRemoteEnvOptions* options, RLMeshRemoteVectorEnv** out)`                                                                                                                                                    | `explicit RemoteVectorEnv(const RemoteEnvOptions& options = {})`                                                                          | `public RemoteVectorEnv(RemoteEnvOptions? options = null)`; `static Task<RemoteVectorEnv> ConnectAsync(RemoteEnvOptions? options = null, CancellationToken cancellationToken = default)` (RMT-24) |
| `.num_envs -> int`                                                                   | `fn num_envs(&self) -> u64`                                                                                             | `uint64_t rlmesh_remote_vector_env_num_envs(const RLMeshRemoteVectorEnv*)`                                                                                                                                                                                         | `uint64_t num_envs() const`                                                                                                               | `ulong NumEnvs { get; }`                                                                                                                                                                          |
| `.single_observation_space` / `.single_action_space`                                 | `fn single_observation_space(&self) -> &Space`, `single_action_space`                                                   | `const RLMeshSpace* rlmesh_remote_vector_env_single_observation_space(const RLMeshRemoteVectorEnv*)`, `_single_action_space`                                                                                                                                       | `const Space& single_observation_space() const`, `single_action_space`                                                                    | `Spaces.Space SingleObservationSpace { get; }`, `SingleActionSpace`                                                                                                                               |
| `.observation_space` / `.action_space` (alias of `single_*`)                         | same names; return the same object as `single_*` (RMT-9)                                                                | `rlmesh_remote_vector_env_observation_space`, `_action_space` (same pointer as `single_*`)                                                                                                                                                                         | same names, same object                                                                                                                   | same names; `ReferenceEquals` the `Single*` object (RMT-9)                                                                                                                                        |
| `.reset(*, seed: int \| list[int] \| None = None, options=None)`                     | `fn reset(&mut self, seed: Option<VectorSeed>, options: Option<&InfoDict>) -> Result<(Value, InfoDict)>`; `reset_async` | `RLMeshStatus rlmesh_remote_vector_env_reset(RLMeshRemoteVectorEnv*, const int64_t* seed, size_t seed_len, const RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info)`                                                                     | `std::tuple<Value, InfoDict> reset(std::optional<VectorSeed> seed = std::nullopt, const std::optional<InfoDict>& options = std::nullopt)` | `ResetResult Reset(VectorSeed? seed = null, Metadata? options = null)`; `ResetAsync`                                                                                                              |
| `.step(actions) -> (obs, rewards, terminated, truncated, info)`                      | `fn step(&mut self, actions: Value) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)>`; `step_async`         | `RLMeshStatus rlmesh_remote_vector_env_step(RLMeshRemoteVectorEnv*, const RLMeshValue* actions, RLMeshValue** out_observation, double* out_rewards, bool* out_terminated, bool* out_truncated, size_t capacity, size_t* out_len, RLMeshValue** out_info)` (RMT-15) | `std::tuple<Value, std::vector<double>, std::vector<bool>, std::vector<bool>, InfoDict> step(const Value& actions)`                       | `VectorStepResult Step(Value actions)` (Env's; `IReadOnlyList<double> Rewards`, `IReadOnlyList<bool> Terminated`, `Truncated`; RMT-15); `StepAsync`                                               |
| `.address`, `.env_id`, `.selected_workflow_edition`, `.env_contract`, `.render_mode` | same as `RemoteEnv`                                                                                                     | same (`rlmesh_remote_vector_env_*`)                                                                                                                                                                                                                                | same                                                                                                                                      | same as `RemoteEnv`                                                                                                                                                                               |
| `.metadata`                                                                          | `fn metadata(&self) -> &Metadata`; `autoreset_mode` stays the wire string (RMT-10)                                      | same; wire string                                                                                                                                                                                                                                                  | same; wire string                                                                                                                         | `Metadata Metadata { get; }`; wire string                                                                                                                                                         |
| `.observation_space_spec` / `.action_space_spec`                                     | same as `RemoteEnv` (single-env specs)                                                                                  | same                                                                                                                                                                                                                                                               | same                                                                                                                                      | same as `RemoteEnv`                                                                                                                                                                               |
| `.render(*, env_index=0)`, `.close()`, `.shutdown(reason)`                           | same as `RemoteEnv`                                                                                                     | same                                                                                                                                                                                                                                                               | same                                                                                                                                      | same as `RemoteEnv`                                                                                                                                                                               |
| `with` (`__exit__` → `close()`)                                                      | `Drop` (RMT-8)                                                                                                          | `rlmesh_remote_vector_env_free` (RMT-8)                                                                                                                                                                                                                            | `~RemoteVectorEnv()`; move-only                                                                                                           | `IDisposable` (RMT-8)                                                                                                                                                                             |
| KeyboardInterrupt in a blocked call                                                  | `cancel_handle(&self) -> ext::CancelToken`                                                                              | `RLMeshStatus rlmesh_remote_vector_env_cancel_handle(const RLMeshRemoteVectorEnv*, RLMeshExtCancelToken** out)`                                                                                                                                                    | `ext::CancelToken cancel_handle() const`                                                                                                  | `Ext.CancelToken CancelHandle()`                                                                                                                                                                  |

Behaviour beyond `RemoteEnv`'s:

- **Arity**: refuses `num_envs < 2` with ValueError
  (`pyrs/client.rs:117-121`; the Python-side check at
  `py/_client/_remote_vector_env.py:57-65` is dead and goes).
- **Batched values** (Spaces owns the layout): observations and actions are
  one `Value` in Python's batched layout
  (`pyrs/spaces/value_codec/batch.rs:215-260`). Python's float64/bool
  `[num_envs]` tensors for rewards and flags become `Vec<f64>` / `Vec<bool>`,
  `std::vector`, and C caller arrays (RMT-15).
- **C arrays**: R-20's caller-buffer form, with one `capacity` and one
  `out_len` shared by the three arrays (they always have `num_envs`
  elements). `capacity` below `num_envs` is `RLMESH_ERR_INVALID_ARGUMENT`,
  checked before any I/O (`num_envs` is known from the handshake): the env
  is not stepped and nothing is written; on success the first `num_envs` elements of each array
  are written and `*out_len = num_envs`.
- **Seeds**: `VectorSeed` (Env) is one int, a per-lane list, or None. Python
  sends an int as a one-element list (`_remote_vector_env.py:126-132`), so
  C's `&seed, 1` equals Python's `seed=0`. C NULL = None; non-NULL with
  `seed_len == 0` = `[]` (R-13).
- **Info**: `episode_ids` (one per lane) and, on `step`,
  `completed_episodes`: the int count of lanes whose episode ended this step
  (0 when none), not a list, as on `RemoteEnv` (`pyrs/client.rs:610,646-651`;
  `_remote_vector_env.py:143-146`).

### `RemoteModel`

| Python                                                                                                              | Rust                                                                                                                       | C                                                                                                                                                   | C++                                                                        | C#                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RemoteModel(address=None, *, host, port, path, transport, connect_timeout_seconds, request_timeout_seconds)`       | `RemoteModel::new(options: RemoteModelOptions) -> Result<RemoteModel>` (no I/O, no twin)                                   | `RLMeshStatus rlmesh_remote_model_new(const RLMeshRemoteModelOptions* options, RLMeshRemoteModel** out)`                                            | `explicit RemoteModel(const RemoteModelOptions& options = {})`             | `public RemoteModel(RemoteModelOptions? options = null)` (no I/O, no twin)                                                                                                                   |
| handle sharing                                                                                                      | `Clone` shares; `Send + Sync`                                                                                              | `RLMeshStatus rlmesh_remote_model_clone(const RLMeshRemoteModel*, RLMeshRemoteModel** out)`; `rlmesh_remote_model_free`                             | copy shares                                                                | a reference shares (R-11); no `Clone()`; `IDisposable`                                                                                                                                       |
| `.address -> str`                                                                                                   | `fn address(&self) -> &str` (normalized, not canonicalized)                                                                | `RLMeshStatus rlmesh_remote_model_address(const RLMeshRemoteModel*, RLMeshBytes* out)`                                                              | `std::string address() const`                                              | `string Address { get; }`                                                                                                                                                                    |
| `__repr__` → `RemoteModel(address='…')` (`_remote_model.py:146-147`)                                                | `Display`, `Debug`                                                                                                         | `RLMeshStatus rlmesh_remote_model_repr(const RLMeshRemoteModel*, RLMeshBytes* out)`                                                                 | `operator<<`, `to_string()`                                                | `ToString()` = `RemoteModel(address='…')`                                                                                                                                                    |
| `.session(env, *, instruction, close_env, trust_entrypoints, execution_horizon, view, workflow_edition) -> Session` | `fn session<'a>(&'a self, env: impl Into<EnvTarget<'a>>, options: SessionOptions) -> Result<Session<'a>>`; `session_async` | `RLMeshStatus rlmesh_remote_model_session(const RLMeshRemoteModel*, RLMeshEnvTarget env, const RLMeshSessionOptions* options, RLMeshSession** out)` | `Session session(EnvTarget env, const SessionOptions& options = {}) const` | `Session Session(EnvTarget env, SessionOptions? options = null)`; `Task<Session> SessionAsync(EnvTarget env, SessionOptions? options = null, CancellationToken cancellationToken = default)` |

`RemoteModel.session` parameters map onto Running's `SessionOptions` fields
of the same name. Only the remote-specific behaviour is listed:

| Python parameter         | Behaviour in every language                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `env: EnvTarget`         | An address, `&EnvServer` (C `rlmesh_env_server_target`, C++ `const EnvServer&`), `&mut RemoteEnv` or `&mut RemoteVectorEnv` (R-17). For an `EnvServer` the model is bound with the server's `env_contract` and no env offer, so the env tier stands in as this build (`pyrs/model.rs:1016-1024`); the session dials the server's `address` with `RemoteEnvOptions { workflow_edition: <the call's workflow_edition>, ..Default::default() }` on its first env use and owns and closes that connection (`_server.py:303-311`; `_connect.py:52-62,91-94,172-179`; `_eval.py:886-897`; RMT-P12). A local env, or an `EnvFactory` whose env the session builds now and closes, is stepped on the caller's side with no `EnvServer`; the model is bound with the contract an `EnvServer` would publish for it and no env offer (RMT-23). A local vector env is TypeError: `a served model drives a single env (RemoteModel is single-lane); got a local vector env: pass a single env, or serve the vector env and run a local model`. The model drives one lane: a `RemoteVectorEnv` (whose arity check guarantees `num_envs >= 2`) or an `EnvServer` whose `env_contract` reports `num_envs > 1` is ValueError from `session`, in every language, after the contract-without-spaces check and before the model is dialed (and before the `EnvServer`'s address is), with today's text `RemoteModel drives a single env, but the env contract reports num_envs={n}; use num_envs=1 (this client sends one observation per predict on lane 0)`; a contract reporting 0 lanes is read as 1 (`rs/model/remote.rs:107-119`; RMT-P13). An address target is dialed as a `RemoteEnv`, whose arity check (RMT-12) already refuses more lanes. Other lane rules are Running's. |
| `instruction=None`       | Set → ValueError (`_remote_model.py:109-113`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `close_env=False`        | With `close_env`, closing a connected session calls `shutdown()` with its default reason on the env: on the caller's `RemoteEnv` / `RemoteVectorEnv`, or on the session's own connection for an address or `EnvServer` target. An accepted shutdown closes that handle locally; a declined one (`allow_remote_shutdown` off, `shutdown` returns false) leaves a borrowed handle open. The session then closes any connection it dialed, even if `shutdown` failed, and returns that error. A session that never connected shuts nothing down. The caller still frees borrowed handles (R-17). (`_eval.py:1546-1566`; `_connect.py:255-272`.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `trust_entrypoints=None` | Set → ValueError (`_remote_model.py:114-118`). C: `has_trust_entrypoints == false`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `execution_horizon=1`    | Checked before any dial: outside `1..=1024` (`MAX_EXECUTION_HORIZON`, `_eval.py:70,775-781`) → ValueError (Running's RUN-P6; remote facts in RMT-P1).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `view=None`              | As Running's `view`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `workflow_edition=None`  | Declares the runtime tier for the session (`_remote_model.py:125`); the env leg is pinned to the three-way floor before the first reset (`_eval.py:897-902`). For an address or `EnvServer` target (the two connections the session dials itself) it is also the dialed env's `RemoteEnvOptions.workflow_edition`, so the env handshake uses the call's declaration ahead of `RLMESH_WORKFLOW_EDITION`, as Python's docstring states (`_remote_model.py:104-107`; RMT-P12). A caller's `RemoteEnv` / `RemoteVectorEnv` is already connected and keeps its own handshake.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

- `new` runs the address helpers, parses the resulting address
  (`ConnectAddress::parse`, R-22's `invalid address: …` text) and validates
  both timeouts (RMT-4), all ValueError, and stores the address verbatim, so
  `"127.0.0.1:5555"` stays scheme-less (`_remote_model.py:62-71`). It does
  no I/O: the model is dialed at `session`. Python parses the address and
  validates the timeouts only at `session` today (`rs/model/remote.rs:204`;
  `pyrs/model.rs:1008-1011`), after an address target's env is dialed;
  RMT-P6 moves both checks to construction, as `RemoteEnv` does
  (`pyrs/client.rs:90-91,721`).
- `session` dials the model with the handle's timeouts. A dial failure is
  ConnectionError, a handshake refusal ProtocolException (P-3), an expired
  `connect_timeout` TimeoutError. A target whose own `env_contract` lacks
  spaces (an `EnvServer` or other local handle; never a connected
  `RemoteEnv`, which RMT-12 refuses at construction) is RuntimeError, R-21's
  "contract without spaces" misuse, raised before the model is dialed. Today
  it comes from reading the contract's `observation_space` /
  `action_space` getter (`pyrs/model.rs:1149,1155` →
  `pyrs/spaces/spec_view.rs:143-155,583-592`: "remote environment contract
  missing observation_space"); the `ok_or_else` checks at
  `pyrs/model.rs:1002-1007` are unreachable, because
  `native_env_contract_from_py` always returns both spaces (`:1166-1172`),
  and go when the check moves into the facade.
- Model calls: `request_timeout` bounds each `predict` and the session's
  route close over the model (RMT-4, model paragraph).
- An address target is dialed with `RemoteEnvOptions { workflow_edition:
<the call's workflow_edition>, ..Default::default() }`, not the model's
  timeouts (`_remote_model.py:161-171`; RMT-P12), before the model; the session
  owns and closes it. A `session` call that fails after dialing it (model
  dial, handshake, timeout) closes that connection with RMT-4's bounded
  close before returning the original error (RMT-P7; Python today leaves it
  to GC detach, `_remote_model.py:126-141`).
- The session borrows the model (`&'a RemoteModel`, R-17); C and C++ free the
  model after the session; a C# `Session` holds a reference on it (R-17's
  C# bullet), so disposing the `RemoteModel` first is harmless. Concurrent `session` calls on one handle are
  allowed.

### Bridges and deferred

| Python                                                                                            | Rust, C, C++                                        | C#                           |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------- |
| `rlmesh.numpy.RemoteEnv` / `RemoteVectorEnv` / `RemoteModel` (Stable)                             | none: NumPy bridge (R-24)                           | none (R-24)                  |
| `rlmesh.torch.Remote*`, `rlmesh.jax.Remote*` (Experimental)                                       | none: Torch / JAX bridges (R-24)                    | none (R-24)                  |
| `SandboxEnv`, `SandboxVectorEnv`, `SandboxModel`, `SandboxRuntime`, `SandboxBuild` (Experimental) | deferred; R-7.2-R-7.4 already fix their call shapes | deferred, as R-7.2-R-7.4 fix |

Deferral covers every form that names a sandbox handle: Running's
`ModelTarget` `From<&'a SandboxModel>` / `rlmesh_sandbox_model_target` /
`SandboxModel&` and its sandbox `EnvTarget` forms ship with the clients, not
before (R-7 note, R-17). Until then `EnvTarget` and `ModelTarget` have no
sandbox variant in any language.

Sandbox infrastructure failures (Docker missing, pull failed, container
exited) raise builtin `RuntimeError`, R-21's misuse class, today
(`py/_sandbox/session.py:397-406,426-445,517-530`). They move to a
non-misuse class before sandbox parity lands (RMT-P8, deferred; R-21 scope
note); the sandbox client deadlines produce `Error::Timeout` with R-22's
text (P-8).

### C options structs

```c
typedef struct RLMeshRemoteEnvOptions {
  size_t struct_size;
  const char* address;          /* NULL = None */
  const char* host;             /* NULL = None */
  bool has_port;
  uint16_t port;
  const char* path;             /* NULL = None */
  bool has_transport;
  RLMeshTransport transport;    /* Env; INVALID = 0 */
  bool has_connect_timeout_seconds;
  double connect_timeout_seconds; /* RMT-4 */
  bool has_request_timeout_seconds;
  double request_timeout_seconds; /* RMT-4 */
  const char* workflow_edition; /* NULL, "" or whitespace = not declared */
  /* Appended after ABI 5 (RMT-22): read only when struct_size covers them.
   * _reserved (0) starts exactly at the ABI-5 sizeof on every target. */
  uint64_t _reserved;
  const char* api_key;          /* NULL = RLMESH_API_KEY for https://; "" = none */
  const char* tls_ca_file;      /* NULL = system roots */
} RLMeshRemoteEnvOptions;
#define RLMESH_REMOTE_ENV_OPTIONS_MIN_SIZE offsetof(RLMeshRemoteEnvOptions, _reserved)
#define RLMESH_REMOTE_ENV_OPTIONS_INIT \
  {sizeof(RLMeshRemoteEnvOptions), NULL, NULL, false, 0, NULL, false, RLMESH_TRANSPORT_INVALID, false, 0.0, false, 0.0, NULL, 0, NULL, NULL}

typedef struct RLMeshRemoteModelOptions {
  size_t struct_size;
  const char* address;
  const char* host;
  bool has_port;
  uint16_t port;
  const char* path;
  bool has_transport;
  RLMeshTransport transport;
  bool has_connect_timeout_seconds;
  double connect_timeout_seconds;
  bool has_request_timeout_seconds;
  double request_timeout_seconds;
  const char* api_key;          /* appended after ABI 5, as above */
  const char* tls_ca_file;
} RLMeshRemoteModelOptions;
#define RLMESH_REMOTE_MODEL_OPTIONS_MIN_SIZE offsetof(RLMeshRemoteModelOptions, api_key)
#define RLMESH_REMOTE_MODEL_OPTIONS_INIT \
  {sizeof(RLMeshRemoteModelOptions), NULL, NULL, false, 0, NULL, false, RLMESH_TRANSPORT_INVALID, false, 0.0, false, 0.0, NULL, NULL}
```

Both are new in ABI 5. `_MIN_SIZE` stays frozen at their ABI-5 `sizeof` on
every target: the env options' `uint64_t _reserved` starts there (a 32-bit
target that aligns `double` to 8 pads the ABI-5 struct after
`workflow_edition`, and `_reserved` keeps the appended fields out of that
padding), and the model options end on a `double`, with no tail padding. A
caller built against ABI 5 keeps working: `struct_size == _MIN_SIZE` reads
no appended field, a `struct_size` that stops between appended fields reads
the missing ones as NULL, and one that ends inside a field is ValueError
(`… struct_size N ends inside a field`). Otherwise
sizing follows R-19 and a NULL options pointer means `_INIT`.

## Python changes

- **RMT-P1 Validate `execution_horizon` in `RemoteModel.session`.** The
  change, its ordering (before either dial) and its changelog entry are
  Running's RUN-P6; this item only records the remote-specific facts. Today
  the value goes straight to `PyModelClient`'s `u32`
  (`py/_client/_remote_model.py:128-136`; `pyrs/model.rs:994,1049`;
  `rs/model/remote.rs:274-275`), because `remote_session` builds the
  `Session` with the default horizon 1 (`_remote_model.py:207-214`), so
  `_eval.py:775-781` never sees the real value. A negative value raises
  OverflowError; 0 is pinned on the wire, where it means no chunking, so it
  silently runs as 1
  (`crates/rlmesh-proto/proto/rlmesh/model/v1/interaction.proto:56-61`); > 1024 is refused only by
  the served engine at route resolution, on the first `predict`, as a model
  error (`rs/model/engine.rs:1272-1281,1332`).
- **RMT-P2 Calls on a closed client raise `RuntimeError`** (RMT-13). Today a
  `reset`/`step`/`render`/`shutdown` after `close()` raises
  `ConnectionError("not connected")` (`grpc/env/client.rs:624-626` for
  `shutdown`, `:871-876` (`ensure_ready`) for the rest, mapped
  through `rs/error.rs:262`). R-21 files a closed handle under builtin
  `RuntimeError`. Breaking for `except ConnectionError` around a closed
  client; settled with RMT-13 (user).
- **RMT-P3 Render decode failure raises `ProtocolException`** instead of
  builtin `RuntimeError` (`pyrs/client.rs:792-797`): an undecodable frame is a
  malformed message (R-21). Subclass of `RuntimeError`; catches keep working.
- Handshake and edition refusals at construction, including a peer contract
  that lacks spaces or does not decode, raise `ProtocolException` (P-3,
  RMT-12). Today they arrive as `RuntimeError` through `Error::Internal`
  (`grpc/env/client.rs:372-425,885-900` → `rs/error.rs:205-209,248`;
  `rs/env/client.rs:104-107`).
- **RMT-P4 `path` with `transport='tcp'` raises ValueError.** Today it
  silently connects to `unix://path`, ignoring the explicit transport
  (`_endpoint.py:82-89,116-119`). D-16: contradictory helpers are refused,
  as `transport='unix'` with `host`/`port` already is. Compatibility: a call
  that used to connect over a unix socket now raises.
- **RMT-P5 Accessors never raise "Already borrowed".** Today `env_id`,
  `selected_workflow_edition` and `RemoteVectorEnv.num_envs` (not cached,
  `_remote_vector_env.py:79-82`) go through `&self` pyclass methods, so while
  another thread holds the client in a call they raise pyo3's
  `RuntimeError` (`pyrs/client.rs:382-395,470-474,582-584`), whereas `address` and
  `env_contract`, cached Python attributes, work. Target: every accessor
  overlaps any call (RemoteEnv Concurrency); the extension reads them from
  state outside the per-call borrow. Compatibility: a call that raised now
  succeeds.
- **RMT-P6 `RemoteModel` validates its address and timeouts at
  construction.** Today `RemoteModel.__init__` runs only the address helpers
  (`normalize_connect_address`, which does not parse,
  `_endpoint.py:56-120`) and stores the timeouts raw
  (`_remote_model.py:62-71`), so `RemoteModel("bogus://x")` and
  `RemoteModel(addr, connect_timeout_seconds=-1)` succeed and every
  `session` then raises ValueError (`rs/model/remote.rs:204`;
  `pyrs/model.rs:1008-1011`), for an address target after its env is
  dialed, while `RemoteEnv` parses the address and checks both timeouts at
  construction (`pyrs/client.rs:90-91,721`). Target: an address that does
  not parse (`invalid address: …`, R-22) or a negative or non-finite
  timeout is ValueError from `RemoteModel(...)`, `RemoteModel::new`,
  `rlmesh_remote_model_new` and the C++ constructor (RMT-4, RMT-12). D-16:
  the deferral is accidental. Compatibility: the ValueError moves from
  `session` to the constructor.
- **RMT-P7 A failed `RemoteModel.session` closes the env it dialed.** Today
  an address target is dialed into a `RemoteEnv` before the model, and if
  the model dial or handshake then fails the fresh `RemoteEnv` is never
  closed and is left to GC detach (`_remote_model.py:126-141`). Target: it
  is closed with RMT-4's bounded close before the original error propagates;
  a close error is logged, not raised. D-16. Compatibility: no API change;
  the env server sees the connection close promptly.
- **RMT-P8 (deferred, with sandbox parity) Sandbox infrastructure failures
  leave the misuse class.** Docker missing, a failed pull, a container that
  fails to start, exits or publishes no port raise builtin `RuntimeError`
  (`py/_sandbox/session.py:397-406,426-445,517-530`). Target: `RLMeshException`
  for a local infrastructure failure, `ConnectionError` when the container
  cannot be reached; builtin `RuntimeError` stays misuse only (R-21 scope
  note). Lands before the sandbox clients get native forms. Compatibility:
  `RLMeshException` subclasses `RuntimeError`, so those catches keep working;
  the `ConnectionError` cases no longer match `except RuntimeError`.
- **RMT-P9 Ctrl-C abandons a blocked served-model call.** Today the model
  client's `predict` and route close block without polling signals
  (`pyrs/model.rs:1093-1135`), so KeyboardInterrupt arrives only after the
  RPC returns, unlike the env client (`pyrs/client.rs:201-262`). Target:
  they poll as the env client does, and a KeyboardInterrupt abandons the
  call as RMT-6 describes (R-18: one mechanism). Compatibility: Ctrl-C during
  a served-model step returns promptly.
- **RMT-P10 An expired env request is always `TimeoutError`.** Today
  `reset`/`step`/`render` send the request timeout as `timeout_ms`,
  truncated to whole milliseconds (`pyrs/client.rs:194-198,258-298`), and
  the env server enforces it: on expiry it drains the op, then replies
  `TIMEOUT` (`grpc/env/server.rs:40-90`). The client checks its own deadline
  only on the 100 ms signal-poll tick (`pyrs/client.rs:25,217-232`), so a
  reply that lands between the deadline and the next tick is returned: a
  late success comes back, and a server `TIMEOUT` raises builtin
  `TimeoutError("[TIMEOUT] env.step timed out after 5000ms")`
  (`pyrs/types.rs:70-71`), which P-4 alone would turn into
  `RecoverableEnvironmentException`. Target (RMT-21): `timeout_ms` is
  rounded up to whole milliseconds, and a reply handled at or after the
  client's deadline is discarded, the call raising `TimeoutError` with
  R-22's text. D-16: which deadline wins is accidental. Compatibility: the
  class stays `TimeoutError`; a call that finishes up to 100 ms past its
  deadline now raises instead of returning.
- **RMT-P11 `render(env_index=…)` above `u32::MAX` raises ValueError.**
  Today the extension takes `usize` (`pyrs/client.rs:435-439`), so a
  negative index is OverflowError (kept, R-21) and a value above `u32::MAX`
  is truncated by `i as u32` (`grpc/wire/value/interaction.rs:78-83`),
  silently rendering another lane. Target: ValueError before any I/O
  (RemoteEnv Lane index). D-16. Compatibility: a call that rendered the
  wrong lane now raises.
- **RMT-P12 `RemoteModel.session(address or EnvServer, workflow_edition=…)`
  declares the dialed env's edition.** Today `dial_env_address` builds `RemoteEnv(env)`
  without it (`_remote_model.py:125-126,161-171`), so the env handshake
  resolves `RLMESH_WORKFLOW_EDITION` or the project instead
  (`py/_editions.py:139-151`) and is refused when that names an edition the
  env cannot run, even when the call names one it can, although the
  docstring says the call overrides it (`_remote_model.py:104-107`). An
  `EnvServer` target drops the value the same way: `remote_session` builds
  the `Session` without `workflow_edition` (`_remote_model.py:207-215`), so
  `Session._workflow_edition` stays None (`_eval.py:767,807`) and the lazy
  dial `connect_env(..., self._workflow_edition)` builds
  `RemoteEnv(address, workflow_edition=None)` (`_eval.py:890-892`;
  `_connect.py:91-94,172-179`), whereas `rlmesh.session(model, env_server,
workflow_edition=X)` passes X to the same dial. Target: the address target
  is dialed with `workflow_edition=` the call's value, and `remote_session`
  passes the call's `workflow_edition` into `Session._create`, so the
  `EnvServer` dial carries it too. D-16: the docstring states the intent.
  Compatibility: such a session now connects; without a call value nothing
  changes.
- **RMT-P13 `RemoteModel.session` refuses a multi-lane target with
  ValueError.** Today `RemoteModel.session(remote_vector_env)`, or an
  `EnvServer` whose contract has `num_envs > 1`, raises builtin RuntimeError
  (RLMeshException after P-2) from `session`: `PyModelClient` passes the
  contract to `connect_inner`, whose `single_lane_contract` returns
  `Error::Internal` before the model is dialed (`rs/model/remote.rs:107-119,217`;
  `pyrs/model.rs:1001,1027-1033`; `pyrs/types.rs:52`). A `RemoteVectorEnv`
  target therefore always fails. Target: ValueError with the same text,
  raised at the same point (RemoteModel `env` row). D-16: the env's arity is
  an argument value, not misuse (R-21). Compatibility: `except RuntimeError`
  no longer matches.
- **RMT-P14 `api_key`, `tls_ca_file`, `https://` and local envs
  (RMT-22, RMT-23).** `RemoteEnv`, `RemoteVectorEnv` and `RemoteModel` gain
  the keywords `api_key: str | None = None` and
  `tls_ca_file: str | os.PathLike[str] | None = None`, threaded through
  `PyEnvClient`, `PyVectorEnvClient`, `PyModelClient` and
  `check_remote_model_options`; `https://` addresses dial TLS and
  `unix://` addresses reach a served model. `RemoteModel.session` (and
  `rlmesh.run` with a `RemoteModel`) accepts a plain local env or an
  `EnvFactory`, bound through the native `local_env_contract` and stepped
  on the caller's thread with the numpy bridge (or the env's own
  `_bridge`); a local vector env is TypeError. Compatibility: a local env
  target no longer raises TypeError; `test_token_auth_surface_stays_removed`
  is replaced by a test that the key never reaches repr or error text.
- Timeout texts follow P-8: `remote environment call timed out after {:.3}s`
  (`pyrs/client.rs:243-246`), `remote environment connect timed out after
{:.3}s` (`pyrs/client.rs:726-730`), `remote model connect timed out after
{:.3}s` (`pyrs/model.rs:1038-1041`) and `remote model request timed out
after {:.3}s` (`pyrs/model.rs:1127-1130`, every `predict` and route close
  over a `RemoteModel`) all become R-22's `operation timed out after {:?}`,
  as does the server deadline's `[TIMEOUT] env.step timed out after
5000ms` that can reach a caller today (RMT-P10). The class stays
  `TimeoutError`.
- Moves without behaviour change: address normalization, timeout parsing,
  the close bound and detach fallback, reserved info keys, PNG decoding, the
  arity checks and the "could not connect to an env at …; is the EnvServer
  running?" / "could not connect to a model at …" texts
  (`_remote_base.py:95-99`, `_remote_model.py:137-141`) move into the
  facade (R-22), and Python calls it. The handshake, the arity checks and
  the edition refusal already run in the native constructor
  (`pyrs/client.rs:94-122`), and `handshake()` is now a local conversion
  with no I/O (`pyrs/client.rs:181-183`), so the try/close around it at
  `_remote_base.py:101-117`, its never-raised "connected to an env at …, but
  the handshake failed" text and the `_post_handshake` arity `ValueError`s
  (`_remote_env.py:54-63`, `_remote_vector_env.py:57-65`) are dead and are
  removed. `autoreset_mode` enum restoration stays in Python (RMT-10).

## Current differences

Rust (`rs/env/client.rs`, `rs/model/remote.rs`, `grpc/`):

- Clients are async-only (`rs/env/client.rs:59-406,420-536`; G-9): today's
  methods become the `_async` twins; add blocking primaries that poll the
  borrowed future on the caller (R-17). A sync call inside a runtime returns
  `Error::Runtime` naming the twin (R-16).
- The grpc client spawns its response pump and channel worker with ambient
  `tokio::spawn` (`grpc/env/stream.rs:65-70`, `grpc/model/stream.rs:30-34`,
  `grpc/env/client.rs:679-708`); spawn them on the library runtime handle.
  That alone does not let twins run on any executor: the caller-polled
  futures also create tokio timers and I/O resources (`open_join`'s sleep and
  deadline, `grpc/env/client.rs:724,740`; the request deadline; tonic's
  connector), which panic when first polled outside a tokio context. Each
  `_async` twin therefore wraps its future so every poll runs inside the
  library runtime's context (`Handle::enter` around `poll`), registering
  timers and sockets with the library runtime's driver whichever executor
  polls it (R-17).
- Constructors `connect`, `connect_with_token`, `connect_to`,
  `connect_declaring` (`:59-90,420-444`) → `new(RemoteEnvOptions)` /
  `new_async`. `RemoteEnv` cannot declare an edition today (`:420-444`).
- `RemoteEnv` wraps the public `RemoteVectorEnv` (`:414-416`), and the Python
  extension builds `RemoteEnv` on `RemoteVectorEnv` (`pyrs/client.rs:57-60`):
  extract one any-arity core as a `pub` type in the `#[doc(hidden)]`,
  semver-exempt `rlmesh::binding` module (R-25), `binding::EnvClientCore`,
  which the extension uses directly and which `RemoteEnv` and
  `RemoteVectorEnv` each wrap.
- Arity: `RemoteVectorEnv` accepts `num_envs == 1` (`:93-132`); `RemoteEnv`
  refuses `> 1` with `Error::Internal` (`:437`) → `Error::InvalidArgument`.
- Errors: handshake failures are `Error::Internal` (`rs/error.rs:205-209,248`)
  → `Error::Protocol`. That includes an env contract without spaces, which
  the grpc handshake refuses first as `ProtocolError::HandshakeFailed`
  (`grpc/env/client.rs:404,885-900`), and a contract that does not decode,
  today `Error::Internal("invalid spaces spec from remote env")`
  (`:104-107`). The facade's own `validate_env_contract` (`:108,539-551`)
  and the extension's `require_contract_space` (`pyrs/client.rs:106-111,
752-758`) are unreachable and go. A half-connected client is dropped
  before the error returns; that releases nothing, because no Join is open
  yet (`pyrs/client.rs:94-122`).
- No timeouts in the facade; only per-request `timeout_ms`
  (`crates/rlmesh-spaces/src/request.rs:8`), which the caller fills. Port
  `optional_timeout`, `connect_remote_env`, `run_rpc`'s deadline and the
  `timeout_ms` the extension sends on `reset`/`step`/`render`
  (`pyrs/client.rs:194-198,201-298,715-750`; `pyrs/model.rs:1008-1045`),
  with RMT-21's rounding and deadline-wins rule, and drop them from the
  extension.
- `close` waits unbounded, has no detach fallback and returns
  `VectorCloseResult` / `CloseResult` (`:381-392,528-531`) → `Result<()>` with
  RMT-4's bound and fallback (`pyrs/client.rs:30,307-332`). Final-episode
  metadata leaves the public surface (Python has none).
- `Drop` only drops the stream, which detaches → RMT-8.
- `shutdown` requires a reason (`:399,534`) → `Option<&str>`.
- `RemoteEnv::reset`/`step` take `ResetRequest`/`StepRequest` and return
  `episode_id` as a struct field, dropping ids from steps (`:479-519`);
  vector calls use per-lane `Vec<SpaceValue>` (`:259-271,309-314`) → R-15
  tuples, batched `Value`, `Vec<f64>`/`Vec<bool>`, reserved info keys,
  `Option<VectorSeed>`.
- `render` returns PNG bytes in `RenderResult` (`:368,523`) → `Option<Value>`;
  move `decode_render_bytes` (`pyrs/client.rs:800-829`) into the facade and
  add `image` (png feature) to `crates/rlmesh/Cargo.toml`.
- Missing: `selected_workflow_edition` on `RemoteEnv` (vector only, `:139`),
  cached spaces, `*_space_spec`, `render_mode`, `metadata`, `cancel_handle`.
- `RemoteModel` is an eager per-env client, `connect(address, env_contract)`
  (`rs/model/remote.rs:135-195`), with a hand loop `reset` / `predict` /
  `close` (`:314,344,446`) and an unchecked `set_execution_horizon` (`:274`) →
  `RemoteModel::new` (no I/O), `session` / `session_async`; the per-env
  client moves to `binding` as Session's engine, `binding::ModelClient`.
  `single_lane_contract` (`rs/model/remote.rs:107-119`) moves with it and
  returns `Error::InvalidArgument` instead of `Error::Internal`, still
  before the model dial (RMT-P13).
- Fencing (RMT-5, RMT-14): `connect_with_token` (`:66,425`;
  `rs/model/remote.rs:141`) → the `api_key` option (RMT-22); the inherent
  `session_offer` and `pin_workflow_edition` on both clients
  (`:146,170,458,464`) → methods of `binding::EnvClientCore`, so none stays
  on a shared type (R-25, E-2 item 4); `RemoteModel::ceiling`
  (`rs/model/remote.rs:286`) → a method of `binding::ModelClient`;
  `detach` (`:182,469`) → private;
  partial reset (`VectorResetRequest.env_indices`, `:204-253`) →
  `ext::reset_lanes`; the lane conversion moves with it and takes `u64`, so
  an index above `u32::MAX` is `Error::InvalidArgument` before any I/O
  (today `Vec<i32>` and a negative lane is `Error::Environment` with
  `InvalidAction`, `rs/env/types.rs:55`, `:207-224`).
- A token a header cannot carry (`token_header_problem`,
  `grpc/helpers/auth.rs:49-59`) is refused today only on the first
  authorized request, after the dial, as `ClientError::InvalidToken`
  (`grpc/error.rs:324-327`; `grpc/env/client.rs:789-793`), which
  `rs/error.rs:262` maps to `Error::Connection` → RMT-22: `api_key` is
  checked at construction, before any dial, `Error::InvalidArgument`; map
  `ClientError::InvalidToken` to `Error::InvalidArgument` too.
- The selected edition is a plain `String` overwritten by
  `pin_workflow_edition(&mut self)` (`rs/env/client.rs:45,146-150`); the grpc
  client already keeps the pinned spelling in `Shared::pinned_workflow_edition`,
  a `Mutex<String>` every clone reads (`grpc/env/client.rs:122-125,433,
457-460`). Both stay, written by the one pin; the C handle reads the
  latter through `binding::EditionCell` (RMT-20).
- Error text drops the `connection error:` style prefixes (R-22).

C (`h`, `capi/`):

- No client exists; only `rlmesh_model_run_local` takes an env address
  (`h:478`; G-8). Add `RLMeshRemoteEnv`, `RLMeshRemoteVectorEnv`,
  `RLMeshRemoteModel`, their functions, `RLMeshRemoteEnvOptions` /
  `RLMeshRemoteModelOptions` with `_INIT` and `_MIN_SIZE` (R-19).
- The contract type is `RLMeshContract` (`h:229`) → `RLMeshEnvContract`
  (R-27, G-28).

C++ (`hpp`):

- No client classes (G-8). Add `rlmesh::RemoteEnv`, `RemoteVectorEnv`
  (move-only) and `RemoteModel` (copy shares), each wrapping the C handle
  one to one, with exceptions per R-21 (D-5).

C# (`cs/`):

- Design record (D-45): v0.2 builds no C# remote clients (its tests drive
  the C remote client through the internal raw layer). The column above
  records how `RemoteEnv`, `RemoteVectorEnv` and `RemoteModel` would wrap
  the C handles one to one as C++ does; it is not a backlog.
- Twins' cancellation (R-18): `ResetAsync` and `StepAsync` pass their
  per-operation token through X-12's `rlmesh_ext_remote_env_reset` /
  `_step` (and the vector forms). The other twins (`RenderAsync`,
  `CloseAsync`, `ShutdownAsync`, `ConnectAsync`, `RemoteModel`'s) have no
  per-call input: their `CancellationToken` is checked before scheduling,
  and after that the call runs to its client timeout. Firing the handle's
  shared token instead would reach the handle's next call. v2's
  per-operation context removes the limit (Lessons 3).
- TLS, `api_key` and `RLMESH_API_KEY` resolution all happen in the C
  library's dialer (RMT-22): the C# layer passes `Address`, `ApiKey` and
  `TlsCaFile` through and uses no .NET networking (`SslStream`,
  `HttpClient`, `X509Store`), so trust, ALPN and error texts are the same
  as every other language's.
- `RLMESH_API_KEY` is read by the C library from the C process environment.
  On .NET (CoreCLR) for Unix, `Environment.SetEnvironmentVariable` changes
  only the runtime's own copy and never reaches it; on Windows and Mono it
  does. C# callers that set the key in-process pass `ApiKey` (RMT-26).

## Decisions

| ID     | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Status                  |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| RMT-1  | Construction connects: Rust `RemoteEnv::new` / `new_async`, C `rlmesh_remote_env_new`, C++ constructor; same for `RemoteVectorEnv`. Python's constructor connects, and `new` matches `Model::new`; `connect` would be a second verb for the same act.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Settled                 |
| RMT-2  | `address` is a `RemoteEnvOptions` / `RemoteModelOptions` field, because R-6 sends every defaulted parameter to the options struct and R-7 is closed. Proposed rule change: add `RemoteEnv`, `RemoteVectorEnv`, `RemoteModel` constructors' `address` as R-7.8 (`RemoteEnv::new(Some(addr), opts)`, C `const char* address`, C++ `RemoteEnv("tcp://…")`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | User (accepted default) |
| RMT-3  | One `RemoteEnvOptions` for `RemoteEnv` and `RemoteVectorEnv` (identical Python keywords); `RemoteModelOptions` is the same minus `workflow_edition`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Settled                 |
| RMT-4  | Timeouts: None means no timeout. Zero also means no timeout, as Python reads it (`pyrs/client.rs:737-748`); C carries it as `has_*` true with 0.0, a value, so R-14's no-sentinel rule holds. Negative or non-finite is ValueError. `close` is bounded by `request_timeout` after that normalization, else a 5 s grace (`pyrs/client.rs:30,307-332`); on timeout or a lost connection it detaches locally and succeeds; other errors return. **Model client:** `RemoteModelOptions` timeouts take the same normalization and are validated at `new` (RMT-P6). `connect_timeout` bounds the model dial at `session`. `request_timeout` bounds each `predict` of a session over the model and the session's route close, with TimeoutError (R-22 text, P-8); zero or None means no bound. There is no 5 s grace and no detach fallback: with no bound the route close waits for the server, and on expiry the error is returned while the session still releases its other resources, as in Python (`pyrs/model.rs:1093-1135`; `_eval.py:1535-1544`). An expired, cancelled (token, R-18) or dropped (`_async` future) `predict` is abandoned as RMT-6 describes for env calls: the server finishes it and the reply is discarded (RMT-P9 for Python's Ctrl-C). The env wire deadline (`timeout_ms`) and the resulting error class are RMT-21. Alternative: refuse 0 as `ServeOptions` does (a Python change).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | User (accepted default) |
| RMT-5  | **Superseded by RMT-22 and removed.** A client sends its key through the Stable `api_key` option (RMT-22); the `ext` token constructors below are gone. A server still sets the token it expects through `ext::EnvServerOptions` / `ext::ModelServeOptions`. Original decision: endpoint tokens stay out of the shared surface (Python sends `""`, `pyrs/client.rs:722`, `pyrs/model.rs:1029`): Rust `ext::remote_env_with_token(options: RemoteEnvOptions, token: &str) -> Result<RemoteEnv>` and `remote_env_with_token_async`, `ext::remote_vector_env_with_token(options: RemoteEnvOptions, token: &str) -> Result<RemoteVectorEnv>` and `remote_vector_env_with_token_async`, `ext::remote_model_with_token(options: RemoteModelOptions, token: &str) -> Result<RemoteModel>` (no I/O, no twin; the token is sent when `session` dials); C `RLMeshStatus rlmesh_ext_remote_env_new_with_token(const RLMeshRemoteEnvOptions* options, const char* token, RLMeshRemoteEnv** out)`, `RLMeshStatus rlmesh_ext_remote_vector_env_new_with_token(const RLMeshRemoteEnvOptions* options, const char* token, RLMeshRemoteVectorEnv** out)`, `RLMeshStatus rlmesh_ext_remote_model_new_with_token(const RLMeshRemoteModelOptions* options, const char* token, RLMeshRemoteModel** out)` (`token` non-NULL, else `RLMESH_ERR_INVALID_ARGUMENT`; `""` = no token, as Python sends); C++ `RemoteEnv rlmesh::ext::remote_env_with_token(const RemoteEnvOptions& options, std::string_view token)`, `RemoteVectorEnv rlmesh::ext::remote_vector_env_with_token(const RemoteEnvOptions&, std::string_view)`, `RemoteModel rlmesh::ext::remote_model_with_token(const RemoteModelOptions&, std::string_view)`. A non-empty token that `token_header_problem` refuses (surrounding whitespace, a control or non-ASCII character; `grpc/helpers/auth.rs:49-59`) fails the constructor with ValueError (`Error::InvalidArgument`, `RLMESH_ERR_INVALID_ARGUMENT`, `rlmesh::ValueError`) before any dial, with the grpc problem text, which never includes the token; the model form checks it in `remote_model_with_token`, not at `session`. Otherwise each behaves as its plain constructor. Alternatives: a Stable `token` keyword in Python, or clients reading `RLMESH_*_ENDPOINT_TOKEN`.                                                                                                                                   | User (accepted default) |
| RMT-6  | Cancellation (R-18): `cancel_handle()` returns the handle's token; how a token fires and re-arms is R-18's, the same for every token. Remote specifics: the call is abandoned: the server finishes it and the reply is discarded, the episode is not ended, later calls queue behind it (`pyrs/client.rs:201-262`; `grpc/env/client.rs:692-697`); a fatal reply that lands later still ends the session, as in Python. A `request_timeout` abandons a call the same way. C returns a new token reference, freed with `rlmesh_ext_cancel_token_free`, so a canceller thread may outlive the env. Dials are bounded by `connect_timeout`, not cancellable, as Python's connect does not poll signals (`pyrs/client.rs:715-735`); a client constructor has no token to fire (its `cancel_handle()` exists only after it returns), and dropping a `new_async` / `session_async` future abandons its dial at once (R-18 Dropped futures). A token fired during a dial inside a call that carries it (`RemoteModel.session`'s address-target env dial and model dial, through `SessionOptions.cancel`; a `Session`'s lazy dial of an address or `EnvServer` target at its first env use) counts as fired on that call (R-18): when the dial returns, successful or not, the call fails with R-21's cancellation class after RMT-4's bounded close of every connection the call dialed (as RMT-P7), and the fire is consumed, not left armed for the next call. Python matches: a Ctrl-C during the dial raises KeyboardInterrupt as soon as the dial returns to the interpreter.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Settled                 |
| RMT-7  | Async twins: `new_async`, `reset_async`, `step_async`, `render_async`, `close_async`, `shutdown_async`, `RemoteModel::session_async`, and the fenced `ext::reset_lanes_async` (RMT-14), polled on the caller's executor inside the library runtime's context (R-16, R-17; Current differences). Dropping a twin's future is a cancel (R-18). Accessors have no twins; `RemoteModel::new` does no I/O and has none.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Settled                 |
| RMT-8  | Implicit teardown (Rust `Drop`, C `_free` of an unclosed handle, C++ destructor) runs `close`'s bounded close with its fallback, logging errors; inside an async runtime Rust hands the connection to a detached library task that runs the same close (R-17). Python's `with` is the counterpart; an unclosed Python client collected by GC only detaches (`pyrs/client.rs:477-481`), which no native scope exit needs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Settled                 |
| RMT-9  | Spaces are built once per handle and cached; every call returns the same object, so `seed` then `sample` keeps RNG state, and a vector env's `observation_space` is the same object as `single_observation_space` (`_remote_vector_env.py:84-106`). The RNG lives in the `Space`, synchronized so the client stays `Send + Sync` (Spaces).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Settled                 |
| RMT-10 | Vector `metadata["autoreset_mode"]` stays the wire string natively. Python's restoration of Gymnasium's `AutoresetMode` enum (`_remote_vector_env.py:67-77,186-206`) is a Gymnasium bridge (R-24).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Settled                 |
| RMT-11 | `port` is `u16` / `uint16_t`: Python refuses values outside `0..=65535` (`_endpoint.py:122-127`), as R-5 requires for a narrower type. Python's TypeError for a non-int port is a compile error natively.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Settled                 |
| RMT-12 | Construction errors: invalid address, helpers, timeout, edition, arity or `api_key` (RMT-22) → ValueError; dial failure → ConnectionError; handshake or edition refusal, including a peer contract that lacks spaces or does not decode (a malformed handshake), → ProtocolException (P-3) in every language; expired `connect_timeout` → TimeoutError. The half-connected client is dropped before the error returns, which releases nothing because the Join is not open yet. RuntimeError for a contract without spaces applies only to `RemoteModel.session` given a local handle (an `EnvServer`) whose contract lacks them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Settled                 |
| RMT-13 | I/O on a closed `RemoteEnv` / `RemoteVectorEnv` is R-21's `RuntimeError` in every language (RMT-P2); `close` stays idempotent. Alternative: keep today's ConnectionError.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | User (accepted default) |
| RMT-14 | Rust-only remote plumbing goes to `binding`, never as an inherent or trait member of a shared type (R-25, E-2 item 4); a Python user drives it through `Session`. `binding::EnvClientCore`, the any-arity core both env clients wrap, carries `fn pin_workflow_edition(&mut self, edition: &str)`, `fn session_offer(&self) -> &rlmesh_proto::SessionOffer` and `fn edition_cell(&self) -> binding::EditionCell` (RMT-20). The C ABI reaches a client's core through the free functions `binding::env_core(env: &RemoteEnv) -> &binding::EnvClientCore` and `binding::vector_env_core(env: &RemoteVectorEnv) -> &binding::EnvClientCore`; the facade's own `Session` reaches it crate-privately, and the Python extension owns its core directly. `fn ceiling(&self) -> &PeerCeiling` is a method of the per-env model client, `binding::ModelClient`; the shared `RemoteModel` has none. `detach` becomes private, reachable only as `close`'s fallback. Partial reset is the only user-facing extra, Rust only: `ext::reset_lanes(env: &mut RemoteVectorEnv, env_indices: &[u64], seed: Option<VectorSeed>, options: Option<&InfoDict>) -> Result<(Value, InfoDict)>` and `ext::reset_lanes_async` (same parameters, R-16). It resets the listed lanes (a per-lane seed list aligns with `env_indices`; an empty slice is a full reset, as on the wire; an index above `u32::MAX` is ValueError before any I/O, RemoteEnv Lane index) and returns the batched observation of all `num_envs` lanes and `info` with every lane's `episode_ids` (`rs/env/client.rs:204-270`); other errors and concurrency as `reset`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Settled                 |
| RMT-15 | Vector rewards and flags: `Vec<f64>` / `Vec<bool>`, `std::vector`, C caller arrays in R-20's (`out`, `cap`, `out_len`) form, the three arrays sharing one `capacity` and one `out_len` placed after them; a `capacity` below `num_envs` is refused before any I/O, so the env is not stepped; batched observations and actions stay one `Value`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Settled                 |
| RMT-16 | `render` returns the decoded uint8 frame as a `Value`, or None; a frame that does not decode is ProtocolException (RMT-P3).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Settled                 |
| RMT-17 | C text accessors return `RLMeshStatus` with an `RLMeshBytes` out (R-5), the optional `render_mode` with `bool* out_has` as Env's `rlmesh_env_contract_render_mode` (R-13: `RLMeshBytes` reports `""` as `data == NULL`, `capi/codec.rs:16-26`, so NULL cannot mean None); object accessors (contract, spaces, specs, metadata) and `num_envs` return borrowed pointers or values directly, valid until `_free`. They have no status to report an overlap, so every accessor is safe to call during any call from any thread (RemoteEnv Concurrency).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Settled                 |
| RMT-18 | An env info key that shadows `episode_ids` / `completed_episodes` keeps the env's value and is logged once per client through the library log, as Python's extension does (`pyrs/client.rs:147-171`). It is not an R-23 warning in any language.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Settled                 |
| RMT-19 | `path` with `transport=tcp` is ValueError in every language, as `transport=unix` with `host`/`port` is: contradictory helpers are refused, not resolved by silently preferring one (D-16; RMT-P4).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Settled                 |
| RMT-20 | Selected edition: the value is the handshake selection until `binding::EnvClientCore::pin_workflow_edition(&mut self, edition: &str)` replaces it (RMT-14); a `Session` pins on every connect before its first reset (`_eval.py:897-903`), so pins repeat and the last one is reported. Rust's `fn selected_workflow_edition(&self) -> &str` returns the core's plain `String`; `&mut self` already excludes a concurrent pin. The same pin call also writes the cell the grpc client opens each Join with (`Shared::pinned_workflow_edition`, a `Mutex<String>` shared by every clone, `grpc/env/client.rs:122-125,457-460`), which the handshake set to the same selection (`:433`). `binding::EditionCell` is a `Clone + Send + Sync + 'static` handle on that cell with `fn get(&self) -> String`. The C handle takes one at `_new` (`binding::env_core(&env).edition_cell()`, `vector_env_core` for the vector env), keeps no copy of its own, and `rlmesh_remote_*_selected_workflow_edition` copies `get()` into its `RLMeshBytes`; so a C or C++ accessor may overlap a `Session` pinning through the borrowed `&mut RemoteEnv`, never sees the overlap error, and reports the pin as soon as it is made, as Python does. A pin while a Join is open updates the reported value at once and takes effect on the wire when the next Join opens (the edition is the Join's first message; `grpc/env/client.rs:457-460,759`); it is not refused.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Settled                 |
| RMT-21 | Env request deadline on the wire: the facade sends `request_timeout` as `timeout_ms` on every `reset`, `step`, `render` and `ext::reset_lanes`, as Python does (`pyrs/client.rs:194-198,258-298`), rounded up to whole milliseconds (0 when None or zero, which the server reads as no deadline); `shutdown` and `close` carry none. The client arms its deadline before sending the request, and a reply handled at or after that deadline is discarded as an abandoned call's (RMT-6): the call fails with TimeoutError (R-22 text, P-8). The server's deadline starts no earlier and is no shorter, and on expiry it drains the op before replying `TIMEOUT` (`grpc/env/server.rs:40-90`), so that reply never reaches a caller whose `request_timeout` is set: an expired request is TimeoutError in every language, never RecoverableEnvironmentException. A `TIMEOUT` reply that does reach a caller (a server's own deadline the client did not set) is RecoverableEnvironmentException per D-11 / P-4. Python changes in RMT-P10.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Settled                 |
| RMT-22 | TLS and API keys are shared Stable client options (supersedes RMT-5). One `address` argument; the scheme picks the transport (`https://` = TLS over HTTP/2 with ALPN `h2`, default port 443, no path), and `transport` keeps `tcp` / `unix`. `api_key` (Python `str \| None`, Rust `Option<String>`, C `const char*`, C++ `std::optional<std::string>`) is sent as `authorization: Bearer <key>`; `None` reads `RLMESH_API_KEY`, attached only to an `https://` endpoint, never to plaintext TCP or a Unix socket; `""` sends none. A key `token_header_problem` refuses is ValueError before any dial: `the api_key cannot be sent in the authorization header: it {problem}` (`RLMESH_API_KEY cannot be sent …` for the variable). An explicit key bound for plaintext TCP to a host other than loopback warns once, `UserWarning`: `api_key is sent unencrypted to {address}; use an https:// address to keep it private`. `tls_ca_file` (Python `str \| os.PathLike[str] \| None`, Rust `Option<PathBuf>`, C `const char*`, C++ `std::optional<std::string>`) trusts only that PEM bundle (like `curl --cacert`); unreadable, empty, holding a certificate that does not parse, or given for an address that is not `https://`, is ValueError before any dial. Without it the host's trust store is trusted alone, and only when it yields no roots the bundled web PKI roots alone. A failed TLS handshake is ConnectionError with one text from the shared dialer: `TLS handshake with host:port failed: {reason}` inside the usual `could not connect to …` wrapper (R-22). A key the server rejects (gRPC `Unauthenticated`) is ConnectionError, one text in every language that never echoes the key: `could not connect to a model at '{address}': the server rejected the API key; pass the key it expects as api_key (or set RLMESH_API_KEY for an https:// address)` (`an env` for an env client). The key never appears in error text, `Debug` / `repr` output or logs. Constructors check the key, `RLMESH_API_KEY` and the CA bundle (ValueError early, no network I/O) and every dial resolves them again, so `RemoteModel.session` picks up a rotated key in every language; `RemoteEnv` / `RemoteVectorEnv` resolve before their dial. The plaintext warning is once per address per process. A refused address never echoes its userinfo, query, fragment or path. Python changes in RMT-P14. | User                    |
| RMT-24 | C# client construction (RMT-1): the constructor connects synchronously, as C++'s does. C# constructors cannot be `async`, so Rust's `new_async` is `public static Task<RemoteEnv> ConnectAsync(RemoteEnvOptions? options = null, CancellationToken cancellationToken = default)` (and on `RemoteVectorEnv`), the .NET factory idiom. The dial is not cancellable (RMT-6): an already-cancelled token fails before any I/O; a token fired during the dial fails the twin with `Ext.CancelledError` when the dial returns, after RMT-4's bounded close of the fresh client, as RMT-6 states for a dial inside a call. `RemoteModel` has no twin (no I/O).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pending (C#)            |
| RMT-25 | C# client accessors are properties over values captured once at construction (contract, spaces, specs, metadata, address, env id, `NumEnvs`, render mode), each wrapped over its own native reference or copy, so they outlive the client; only `SelectedWorkflowEdition` reads C on every get (RMT-20). `Metadata` returns the same cached object each read; mutating it changes that C# object only, never the env, as a Python caller mutating the cached dict does.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pending (C#)            |
| RMT-26 | C# `RLMESH_API_KEY`: the C library reads the process environment (`getenv`); .NET's `Environment.SetEnvironmentVariable` does not reliably reach it (on Unix .NET keeps its own copy of the environment block). C# documents that an in-process key goes in `ApiKey`; the variable works when set before the process starts. No C# re-implementation of the lookup, which would split RMT-22's one resolution path.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Pending (C#)            |
| RMT-23 | A served model drives a plain local env (or an `EnvFactory`'s env) with no `EnvServer`: `RemoteModel.session(env)` and `rlmesh.run(RemoteModel(...), env)` derive the env's contract as an `EnvServer` would publish it, before the model is dialed, and step the env on the caller's side; a factory's env is built at `session` and closed with the session (or at once when the dial fails). A local vector env stays TypeError (RemoteModel is single-lane). Python changes in RMT-P14.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | User                    |

### Review notes

- Applied: r1-remote 1 (`&'a mut RemoteEnv` / `&'a mut RemoteVectorEnv` in
  `EnvTarget` and `Session<'a>`, R-17); 2 (abandon, not end, the episode;
  abort-on-drop for spawned tasks; cancellation never calls `detach`; RMT-6,
  R-18); 3 (handshake → ProtocolException, transport → ConnectionError;
  RMT-12, P-3); 4 (RMT-P1); 5 (C++ `const std::optional<InfoDict>&`
  options); 6 (RMT-10).
- Applied reviewer defaults: tokens fenced under `ext` (RMT-5, pending
  user); `new` / `new_async` and C++ constructors (RMT-1); zero timeout = no
  timeout, normalized before the 5 s close fallback (RMT-4, user);
  cached spaces with a synchronized RNG and vector alias identity (RMT-9);
  bounded close outside a runtime, scheduled inside (RMT-8).
- Superseded from round 0: "a cancel ends the in-flight episode" (wrong, see
  RMT-6); "sessions do not borrow the model" (R-17 borrows it); the open
  question on space mutability (Spaces owns it, RMT-9); the Drop review note
  (R-17).
- Consistency pass: `num_envs` is `u64` / `uint64_t` (R-5: counts), as on
  `EnvContract`; `EnvContract` links to Env, its owner.
- Rejected review: none. Each r1-remote finding checked out against the cited
  source.
- Applied (r2): 1 (contract without spaces or undecodable at construction →
  ProtocolException; RMT-12, Current differences); 2 (`&EnvServer` target);
  3 (accessors overlap any call; RMT-17, RMT-P5); 4 (RMT-P1 wording); 5
  (RMT-6 defers token semantics to R-18); 6 (`render_mode` `out_has`); 7
  (timeout texts under P-8); 8 (dead handshake path, citations); 9 (RMT-P4,
  RMT-19); 10 (sandbox targets deferred); 11 (`completed_episodes` count); 12
  (`Handle::enter` around twin polls); 13 (`out_len`; RMT-15); 14 (exact `ext` signatures and `_async` twins; RMT-5, RMT-7, RMT-14).
- Rejected review (r2): none. Each r2 finding checked out against the cited
  source.
- Cross-file (r2): rules.md R-18 — state the token semantics for every
  token (`cancel_handle()` and `cancel` fields alike): one `cancel()` fails
  exactly one call, the one in flight or, when idle, the next call made with
  that token, then the token re-arms, as one Ctrl-C raises one
  KeyboardInterrupt. RMT-6 now cites R-18 for this.
- Cross-file (r2): rules.md P-8 — name all three Python client timeout texts
  it replaces: `remote environment call timed out after {:.3}s`
  (`pyrs/client.rs:243-246`), `remote environment connect timed out after
{:.3}s` (`pyrs/client.rs:727-730`), `remote model connect timed out after
{:.3}s` (`pyrs/model.rs:1038-1041`).
- Cross-file (r2): running.md — mark the sandbox `ModelTarget` forms
  (`From<&'a SandboxModel>`, `rlmesh_sandbox_model_target(...)`,
  `SandboxModel&`) and the sandbox `EnvTarget` forms ("sandbox handles",
  "sandbox `_target`s") as deferred with the sandbox clients (Remote, R-17
  note), not part of the first parity release.
- Cross-file (r2): running.md — `EnvTarget`'s `From<&'a EnvServer>` /
  `rlmesh_env_server_target` / `EnvServer&` is accepted by
  `RemoteModel.session` too (dialed at first env use, owned by the session,
  no env offer); give `rlmesh_env_server_target` a full signature
  (`RLMeshEnvTarget rlmesh_env_server_target(const RLMeshEnvServer*)`).
- Applied (r3): 1 (`close_env` row: owner `shutdown()` with its default
  reason, declined leaves a borrowed handle open, dialed connection closed
  after); 2 (RMT-4 model-client paragraph; RMT-P9); 3 (fourth timeout text;
  already in rules.md P-8); 4 (RMT-P5 lists `num_envs`); 5
  (`binding::EnvClientCore` is `pub`); 6 (malformed token is ValueError
  before any dial; RMT-5, RMT-12, Current differences); 7 (RMT-P8, deferred
  sandbox failure class; Bridges and deferred); 8 (RMT-P1 defers to RUN-P6,
  citation `:128-136`); 9 (RMT-P6, timeouts validated at `RemoteModel::new`);
  10 (RMT-20: `&mut self` pin, `&str` accessor, C mutex copy, pin semantics
  while a Join is open); 11 (RMT-P7, failed `session` closes the dialed
  env); 12 (Address bullet: canonical connect string for the env clients);
  13 (`RLMeshStatus` return types); 14 (Errors citations `:104-107`,
  `:108,539-551`). Also applied rules.md's Cross-file (r3) notes for this
  file: C token type `RLMeshExtCancelToken`, `rlmesh_ext_cancel_token_free`,
  and RMT-6 no longer cites Cross-file (r2).
- Rejected review (r3): none. Each r3 finding checked out against the cited
  source.
- Cross-file (r3): running.md — RUN-P6: cite `_remote_model.py:128-136`
  (today `126-134`) and note that Remote's RMT-P1 carries the remote-specific
  facts (u32/OverflowError, 0 pinned as no chunking, > 1024 refused at the
  first `predict`) and has no changelog entry of its own.
- Applied (r4): 1 (RMT-21: `request_timeout` sent as `timeout_ms`, rounded
  up, on `reset`/`step`/`render`/`reset_lanes`; a reply at or after the
  client deadline is discarded, so an expired request is TimeoutError in
  every language and D-11 still holds for any `TIMEOUT` that reaches a
  caller; RMT-P10, Bounds, Current differences); 2 (RMT-20:
  `binding::EditionCell` over the grpc `Shared::pinned_workflow_edition`,
  read by the C accessor, no C copy); 3 (RMT-P12: an address target is
  dialed with the call's `workflow_edition`, as the docstring says); 4 (C
  vector `step` capacity checked before any I/O; RMT-15); 5 (R-8 setters on
  both options structs); 6 (`env_index` / `env_indices` are `u64` /
  `uint64_t`, above `u32::MAX` ValueError; Lane index, RMT-P11, which also
  fixes today's silent `as u32` truncation); 7 (RMT-14: binding items are
  methods of `binding::EnvClientCore` / `binding::ModelClient`, reached
  through `binding::env_core` / `vector_env_core`); 8 (RMT-P2 citation
  `:624-626`, `:871-876`); 9 (RMT-P1 citation, model/v1 proto).
- Rejected review (r4): none. Each r4 finding checked out against the cited
  source.
- Cross-file (r4): rules.md E-3 — add to the Errors family: a scripted env
  op slower than a client's `request_timeout` fails with the TimeoutError
  class and R-22 text, never `RecoverableEnvironmentException`, in every
  language, including when the op finishes just after the deadline
  (RMT-21).
- Cross-file (r4): rules.md P-8 — also list the env server deadline's
  `[TIMEOUT] env.step timed out after {N}ms`, which reaches a Python caller
  as builtin `TimeoutError` today when the reply beats the client's 100 ms
  poll tick (`grpc/env/server.rs:40-90`, `pyrs/types.rs:70-71`); it becomes
  R-22's text with the TimeoutError class (Remote RMT-21, RMT-P10), not a
  P-4 `RecoverableEnvironmentException`.
- Applied (r5): 1 (an `EnvServer` target is dialed with the call's
  `workflow_edition`, as an address target is; `env` and `workflow_edition`
  rows, RMT-P12 now has `remote_session` pass it into `Session._create`); 2
  (contract-without-spaces RuntimeError cited at `pyrs/model.rs:1149,1155` →
  `pyrs/spaces/spec_view.rs:583-592`, raised before the model dial; the
  `:1002-1007` checks are unreachable and go); 3 (the Join opens on the first
  `reset`, `step` or `render`, never on `close` or `shutdown`); 4 (RMT-6: a
  token fired during an uncancellable dial fails the enclosing call when the
  dial returns, after closing what it dialed, and is consumed); 5 (Shape's
  async twin uses an explicit options value and a new binding, `aenv`).
- Rejected review (r5): none. Each r5 finding checked out against the cited
  source.
- Cross-file (r5): running.md — "Served-model targets" bullet: the session
  dials an `EnvServer`'s address with the call's `workflow_edition` (as for
  an address target; Remote RMT-P12), not with no declaration.
- Cross-file (r5): rules.md E-3 — Cancellation family: a `cancel()` fired
  while `RemoteModel.session` is dialing fails that `session` call with the
  cancellation class once the dial returns, leaves no connection open, and
  the next call on the token succeeds (Remote RMT-6).
- Applied (r6): 1 and 5 (`env` row: C++ `const EnvServer&`, as Running's
  `EnvTarget`); 2 (`env` row: a multi-lane target is ValueError from
  `session` before the model dial; RMT-P13; Rust Current differences moves
  `single_lane_contract` into `binding::ModelClient` as
  `Error::InvalidArgument`); 3 (`RemoteModel::new` parses the address at
  construction in every language; RMT-P6 now covers the address, so RMT-12
  reads correctly for `RemoteModel`); 4 (Fencing bullet: `ext::reset_lanes`
  lane conversion takes `u64`, above `u32::MAX` is `Error::InvalidArgument`,
  today `Error::Environment(InvalidAction)`); 6 (Address helpers: no address
  and no `host`/`port` is ValueError, so default options fail; RMT-11
  citation `_endpoint.py:122-127`).
- Rejected review (r6): none. Each r6 finding checked out against the cited
  source.
- Cross-file (r6): running.md — Validation order bullet ("a vector env on the
  session loop … `py/_models/_resolve.py:74-86`"): note that for a
  `RemoteModel` (including `rlmesh.session(remote_model, env)`, which calls
  `RemoteModel.session`, `py/_models/base.py:1701-1719`) the multi-lane
  refusal is Remote's: ValueError from `session` before the model is dialed,
  with Remote's text (RMT-P13), not `reject_vector_env`'s at connect.
