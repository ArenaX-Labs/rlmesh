# rlmesh-capi

Experimental C ABI for RLMesh, plus a header-only C++20 wrapper, for both sides
of the wire:

- **Models.** A C or C++ program implements `predict`, and the RLMesh runtime
  does the rest: connect to an environment, decode observations, encode actions,
  run episodes, or serve the model as a gRPC `ModelService` endpoint.
- **Environments.** A C or C++ simulator implements `reset` / `step` (and
  optionally `render` / `close`), declares its spaces and adapter tags, and is
  served as a gRPC `EnvService` endpoint that any RLMesh model — Python, Rust,
  C++ — drives.

Status: alpha. The ABI (`RLMESH_ABI_VERSION`) will break before 1.0.

Ordinary output slots need not be initialized: successful writes never read or
free their previous contents. Free any previously owned result before reusing
its slot. Sized output structs still require their `_INIT` / `struct_size`.
Input bool fields must contain a 0 or 1 byte; malformed representations return
`RLMESH_ERR_INVALID_ARGUMENT` before the library reads a typed struct.

## ABI 5 values and tensors

ABI 5 replaces the space-specific value projection with immutable core
`RLMeshValue` handles: None, Bool, Int, Float, Text, Bytes, Tensor, List,
Tuple and Dict. A Discrete action is an Int; a Box observation and a
MultiBinary or MultiDiscrete sample are Tensors. Constructors return a
status and fill an owned handle. `rlmesh_value_clone` shares a reference;
`rlmesh_value_free` releases it. Composite constructors retain references to
children, so callers still free their own handles after success or failure.
Dicts preserve insertion order, permit empty and embedded-NUL keys, and
replace repeated keys in place. Space sampling uses the facade Value conversion,
so sampled Dicts follow their spec's declaration order (including a remote
spec whose keys are not sorted). Space membership uses facade acceptance:
extra Dict keys are ignored and a one-element integer Tensor can be a
Discrete member; missing required keys and out-of-range values are refused.
Metadata and InfoDict use the same Dict kind.

Text and dict keys use length-bearing `RLMeshStr`. Reads of another kind
return `RLMESH_ERR_TYPE`; numeric reads are strict. Missing keys and indices
succeed with NULL, while present empty strings have non-NULL data. Borrowed
children and bytes remain valid while a handle sharing their parent lives.

`RLMeshTensor` is now opaque. `rlmesh_tensor_new` copies a buffer using the
`RLMeshDType` enum; shape and strides use `(out, cap, out_len)` buffers,
with NULL/0 for a length query and no partial writes for a short buffer.
Strides are in bytes. Clone, Value conversion and contiguous reshape share
read-only storage; `rlmesh_tensor_copy` makes independent storage. An
`rlmesh_value_as_tensor` result is an owned shared reference to free with
`rlmesh_tensor_free`. The C++ compatibility Tensor shares that owned handle
on copy, so its data, shape and strides survive the source Value and other
Tensor copies. The old tensor POD, DLPack dtype triple, tensor release
function and package-semver ABI accessors are removed. Use
`rlmesh_build_info` for package identity. Package SONAME generation is read
from `RLMESH_ABI_VERSION`, now 5.

`RLMeshEnvContract` replaces `RLMeshContract`. The library builds it; callers
can retain it with `rlmesh_env_contract_clone` (or the `env` / `spec` aliases)
and free it with `rlmesh_env_contract_free`. Its id, spaces, optional render
mode, lane count, typed metadata, `to_dict` and repr follow the facade.
`rlmesh_trial_index` takes a Value dict and an optional lane, using Python's
integer/float conversion and per-lane validation directly from the facade.

The C env/model/run and remote interfaces below use the facade surface.
The fenced compatibility sections document the old paths still needed by
the C++ wrapper. C++ receives only compatibility changes;
its full parity API is a separate wave.

`rlmesh_tensor_from_dlpack` and `rlmesh_tensor_to_dlpack` use versioned
`struct DLManagedTensorVersioned` descriptors through the Rust facade.
Import copies CPU data and calls the producer's deleter only on success;
on error the caller still owns the descriptor. Export shares read-only
storage by default; `RLMeshToDlpackOptions options = RLMESH_TO_DLPACK_OPTIONS_INIT`
with `options.copy = true` exports independent writable storage. Pass NULL
for default options. The only supported `dl_device` is `(kDLCPU, 0)`, also
reported by `rlmesh_tensor_dlpack_device`. Options require `struct_size`,
zero reserved fields and zero unknown trailing bytes. All failed exchange
calls set a provided output pointer to NULL; an invalid output cannot consume
an import. An exported descriptor owns its data and metadata after the
source tensor is freed; release it exactly once with `exported->deleter(exported)`.
See [`examples/c_dlpack.c`](examples/c_dlpack.c) for imports, exports and ownership.

`rlmesh.h` embeds the complete, unchanged upstream DLPack v1.0 header under
`DLPACK_DLPACK_H_`, with its Apache-2.0 license in `include/dlpack.LICENSE`.
It supports either include order with DLPack 1.x headers and rejects a
pre-1.0 DLPack header included first. The native API has no legacy descriptor,
stream or Python capsule negotiation parameters. The C++ DLPack wrapper is
deferred to the C++20 wave.

## Describe

`rlmesh_model_vtable_describe` / `_describe_json` describe static policy
fields without constructing a Model or invoking any slot, including load and
loaded_native_chunk. They use the same declaration checks as Model construction.
`rlmesh_model_describe` / `_describe_json` describe an instance, keeping the
static native_chunk and declarations while reading the model's current spec.
`rlmesh_env_factory_describe` / `_describe_json` use the facade's representative
branch builds, badges, discriminants and variant catalogs. Factory hooks are
exclusive; each representative env is closed and freed before return.

Use `RLMESH_DESCRIBE_OPTIONS_INIT` or `RLMESH_DESCRIBE_JSON_OPTIONS_INIT`;
NULL means INIT. `has_kind` cross-checks ENV or MODEL, and generated_at is
validated by the facade. Describe returns an owned Value dict; describe_json
returns owned bytes. JSON options can borrow scalar lanes (only lane zero is
read) or a native vector env in place of the default build. A present empty
lane list, NULL lane, or both served forms fails with InvalidArgument. The
borrowed env is never closed. An env leased by a server uses the shorthand below.

C identity is `native:<id>` (UnknownEnv-v1 for a failed/unnamed default build)
or `native:model`, with the C runtime and declared parameter order. Factory
serving publishes that same native identity. A NULL workflow_edition resolves
and records an edition refusal; a non-NULL value bypasses resolution and
validation, including an explicit empty string.

`rlmesh_ext_env_server_describe_json` (COR-16) reads the facade's cached
served-env envelope with the server's resolved edition and host declaration,
including when publication is disabled. It remains readable after start and
shutdown, never rebuilds an env, and never re-resolves the edition. It adds no
public Rust extra. `examples/c_describe.c` exercises the family and sizing rules.

## ABI 5 sizing and space names

Every shared sized input uses its ABI-5 sizeof as MIN_SIZE: NULL options use
INIT, below-minimum sizes fail with InvalidArgument, and nonzero unknown tail
bytes fail with TypeError naming the struct and byte offset. Current fields
are all within that minimum; future appended fields need whole-field checks.
Sized outputs preserve the caller's struct_size and leave unknown fields at
the caller's INIT values.
`rlmesh_space_dtype` reads a Space handle; `rlmesh_space_spec_dtype` reads a
spec. Space and SpaceSpec use distinct free functions. Box, Discrete and Dict
have typed constructors and readers; Experimental MultiBinary, MultiDiscrete,
Text and Tuple now construct Space handles too. Every temporary legacy export
and callback record has been removed. `examples/c_abi5.c` covers strict sizes
and output boundaries.

## Stable adapters

`RLMeshEnvTags` and `RLMeshModelSpec` are immutable facade values. Build them
from typed `RLMeshObsNode` / `RLMeshInputNode` trees and sized leaves, or use
the dict / JSON / metadata constructors. Constructors copy input data; clone
shares the immutable lent tree. Dict keys and text fill use `RLMeshStr`, so
embedded NULs are preserved. Authored order is kept for access and observation
roles; serialized objects use the facade's canonical sorted JSON. Canonical
reads can change a state leaf's representation, as in Python's codec.

Every sized leaf and options record requires its `RLMESH_*_INIT` minimum,
zero reserved bytes and zero unknown input tails. Frozen node arrays retain
their stride; Action components and Split fields are arrays of leaf pointers.
The C fixture asserts that every sized adapter record has explicit padding.
The Zig gate compiles those assertions for both 32-bit x86 and ARM, including
their different double alignments.
Tolerant vocabulary values use `OTHER` plus text; closed enums refuse it.
Image stride is expanded when a spec consumes the leaf, and lent images carry
offsets rather than stride sugar.

Transform handles retain shared callable identity across clones, leaves and
specs. A successful `_new` takes `user_data`; its optional `free_user_data`
runs once when the last reference drops. The callback must support concurrent
calls, receives a borrowed Value, and transfers an owned output even on error.
A failed constructor leaves state with its caller. Callback errors preserve
their recorded error class.

`rlmesh_adapters_tag` / `_tag_vector` validate through the facade before
replacing the env wrapper, leaving the original handle unchanged on failure.
`rlmesh_adapters_resolve` / `_resolve_from_contract` return a shared Adapter
that executes the facade engine, serializing per-step calls and explain.
`rlmesh_adapter_new` implements AdapterBase with a C vtable; Adapter-only
history/advisory queries on that handle fail with TypeError. `wrap_predict`
returns an owned reusable callable that keeps its adapter and callback state
alive until freed.

Attach a spec through `RLMeshModelOptions.spec` or the policy vtable. The
model retains it. `RLMESH_SPEC_KIND_NO_ADAPTER` explicitly bypasses adaptation;
a tagged env with no model spec otherwise fails with AdapterResolutionError.
Trusted custom-input entrypoints resolve through the fenced
`rlmesh_ext_register_entrypoint` registry. Custom-encoding string arms remain
describe-only and are refused by resolve. See `examples/c_adapters.c`.

## Remote clients

`rlmesh_remote_env_new` and `rlmesh_remote_vector_env_new` connect through
facade clients; `rlmesh_remote_model_new` validates without dialing. Use the
corresponding options INIT macro (NULL means INIT), with address or the
host/port/path/transport helpers. Timeout seconds use the facade's zero/None,
finite and sign rules. Token-bearing constructors are fenced under `rlmesh_ext_`.

Remote envs expose cached spaces, specs, metadata and contract pointers until
free, including after close. All immutable accessors and cancel_handle work
while I/O or a session borrows the client; selected_workflow_edition reads the
facade's shared edition cell, including session pins. Overlapping I/O is a
RuntimeError. A Session or run target borrows the client exclusively and
returns it afterward; freeing during a call is the caller's error.

Reset and step return owned Value handles in tuple order. A vector step uses
one capacity/out_len for reward and termination arrays, and refuses a short
buffer before I/O without changing any output. Render delegates PNG decoding
to the facade's Tensor conversion. Close is idempotent; other I/O after close
fails with RuntimeError. Free is required even after a failed close.
RemoteModel clones share the facade handle and can create concurrent sessions
against addresses, EnvServers and remote env targets.

## Running and sessions

`rlmesh_run` and `rlmesh_model_run` accept frozen by-value model/env targets
and `RLMESH_RUN_OPTIONS_INIT`. Options use Python's `episodes`, `seeds`,
`max_episode_steps`, `max_episode_seconds`, execution settings and trial base;
`has_*` flags distinguish an explicit zero from absence. NULL seeds means
None; a non-NULL pointer with length zero means an empty list. Both the native
Model loop and RANDOM_SAMPLE session loop delegate to the facade.

RunHooks are owned vtable handles, borrowed exclusively for a run. Callbacks
run on the calling thread, receive lent sized EpisodeResult/StepEvent structs,
and may clone RunResult or RunContext handles to retain them. Hook errors
follow the facade's first-error rule and still trigger on_run_end once.
Results, telemetry, equality, repr and aggregates use facade result types.

`rlmesh_session_new` / `rlmesh_model_session` create a lazy facade Session.
Reset, predict, step, run, context access, cancellation and close keep the
facade's state machine. A session leases local envs/factories until free;
its borrowed model and server handles must outlive it. Within its own hooks,
only the RunContext members contract/num_envs/read/frame_sources can query the session; stateful
re-entry fails with RuntimeError. Closing/freeing a session never closes its
model. Cancellation tokens are retained independently from the caller's token
handle. Free drains local callbacks before releasing the lease.

Create tokens with `rlmesh_ext_cancel_token_new(&token)` and share them with
`rlmesh_ext_cancel_token_clone(token, &copy)`; both return a status and an owned
reference. `rlmesh_ext_cancel_token_cancel(token)` returns a status and fires
the same token through any clone. Free references independently. Repeated
fires coalesce, and a consumed cancellation re-arms the shared token (R-18).

`RunOptions.instruction` and `SessionOptions.instruction` copy the optional
text and use the facade's input injection. Read items copy a semantic role or
an Image/State/Text leaf. Session readers retain a facade adapter independently
of the session and return Value dicts; roles keep declaration order. Direct
session/context/event reads return the selected value. An authored RunContext
vtable must provide all four protocol slots; slot outputs transfer ownership,
including on failure.

FrameSources owns its roles and labels. A library-made render source borrows
its session/context and preserves the facade's closed, busy and hook-thread
checks. A local context reaches render only during its hook dispatch; direct
Session frame sources can render while the session is idle. Session-made sources
retain weak state: render returns RuntimeError after the session is freed, without
prolonging its env lease. Context-made sources retain the context, including
sources transferred from C callbacks. Missing optional outputs set `out_has=false` and
leave the value/bytes output untouched. Equality compares callable sources as
specified, and repr uses the facade formatter.

RunResult constructors copy typed Advisory records. Returned advisory messages
are borrowed while a shared result handle lives. Advisory lists from the engine
are available on real runs and sessions. See `examples/c_reads.c`.

## View and Recorder

`RLMeshView view = RLMESH_VIEW_INIT` supplies Python's defaults: terminal,
port 8008, 30 fps, JPEG quality 75, no selected source, hold or stepping rate.
`rlmesh_view_parse` accepts Python's shorthand (`term`, `tty`, `http:PORT`,
`both`), ignoring surrounding whitespace and ASCII case. Initialize its
output with `RLMESH_VIEW_INIT`; the parser sets `source` to NULL, preserves
the caller's `struct_size`, and leaves unknown output bytes untouched.
`rlmesh_view_validate`, `_equals` and `_repr` use the facade. Invalid enum
discriminants and undersized structs are refused.
Seconds holds accept every finite nonnegative value, saturating overflow to
Rust's `Duration::MAX`; negative, NaN and infinite values are refused.

Set `RunOptions.view` or `SessionOptions.view` to a View pointer; the consuming
call copies its fields and source string. A local `rlmesh_model_run` refuses
a set view as Python does; use a session for local model viewing. The facade
owns terminal/HTTP rendering, pacing, hold, skip and quit behavior. C++ View
and Recorder wrappers remain part of the separate C++20 wave.

`rlmesh_recorder_new(NULL, &recorder)` uses 30 fps, quality 60 and a generated
result-set id. `rlmesh_recorder_add` records completed runs;
`rlmesh_recorder_capture` returns an ordinary owned `RLMeshRunHooks` handle
for runs, sessions or manual `rlmesh_run_hooks_on_*` dispatch. Captures retain
their explicit context and the recorder's shared state until hooks are freed.
NULL `cameras` discovers sources; a non-NULL list with length zero records
metrics only and permits vector envs. NULL `video_info_keys` selects Python's
default pair; a non-NULL empty list disables video-path discovery. Camera and
video-key strings, task/model/env labels and config Dict values are copied.
The facade writes AV1 media and delivers best-effort capture warnings.

Calls on a recorder serialize internally. `rlmesh_recorder_close` is
idempotent and removes staged media. `rlmesh_recorder_free` closes with errors
swallowed, as required by the C API; retained capture hooks can still record
metrics, while further media staging needs an open recorder. Folder and ZIP
exports use the facade's writer. Refused targets and filesystem failures
return `RLMESH_ERR_OS`; exporting removed staged media after close returns
`RLMESH_ERR_RUNTIME` before writing. Metric-only exports still work after close.

WorkloadRecord, EpisodeRecord and MediaRef are immutable snapshot handles.
Constructors copy their inputs, `_get` returns an independently owned snapshot,
and `_free` releases each handle. Scalar/length reads of NULL return zero;
optional accessors return false for None, and out-of-range `_get` calls return
`RLMESH_ERR_INVALID_ARGUMENT`. Record construction follows Python's dataclasses
without adding numeric validation. Typed Dict conversion, equality, repr and
workload aggregates come from the facade. Every new options type has
`RLMESH_*_INIT`, `_MIN_SIZE` and the ABI-5 size/tail checks; NULL options use
the defaults, while `{0}` is refused. See `examples/c_recorder.c` for ownership,
real capture, exports, manual dispatch and sized-input tests.

## Env authoring and servers

`rlmesh_env_new(vtable, user_data, attributes, out)` and
`rlmesh_vector_env_new` wrap the facade's Gymnasium-shaped protocols. Reset
and step use Value handles and tuple-ordered outputs; info is a Value dict,
not JSON. Vtables include `free_user_data`; ownership transfers only on a
successful constructor. Free releases state without calling close. Close is
idempotent, and callbacks run serially for each handle.

`rlmesh_env_server_new` borrows a list of scalar handles;
`rlmesh_env_server_new_vector` borrows a native vector handle. Construction
binds and derives the contract through the facade. `start`, `serve`, `wait`
and `shutdown` keep its state machine, validation and thread behavior. The
server holds an exclusive lease on each env until free; duplicate or busy
handles fail with RuntimeError. Server free drains C callbacks even when a
facade shutdown deadline has expired.

Use `RLMESH_ENV_VTABLE_INIT`, `RLMESH_ENV_ATTRIBUTES_INIT`,
`RLMESH_VECTOR_ENV_VTABLE_INIT`, `RLMESH_VECTOR_ENV_ATTRIBUTES_INIT`,
`RLMESH_ENV_SERVER_OPTIONS_INIT` and `RLMESH_SERVE_OPTIONS_INIT`.
All new structs have `struct_size` and `_MIN_SIZE`; zero initialization is
refused, and nonzero unknown trailing bytes fail with TypeError. Serve
options use seconds and explicit `has_*` fields, including for zero.
`examples/c_env_api.c` exercises the C API; C++ uses these same handles
through detected Env/VectorEnv adapters and an owning EnvServer.

`RLMeshEnvFactory` takes a sized vtable and copied attributes (tags,
TagParam discriminants, reset options and workflow edition). All hooks use
the facade's branch checks, default binding, stamping and serving pipeline.
Hooks are exclusive, including re-entry from a callback; attribute getters
can run concurrently. Factory free releases state without calling close.
`rlmesh_env_factory_make` returns one owned scalar or vector handle, with
typed metadata stamped by the facade. A Variant catalog uses an owned sink;
Variant fields are mutable and mapping fields are Value dicts. TagParam and
Variant lists lend items until the list is freed. A reusable
`RLMeshExtCancelToken` can cancel factory serving from another thread.
`examples/c_factory.c` covers these contracts, including live serving and
cancellation. Factory serving drains all C callbacks before it returns.

## Policy and Model

`RLMeshModelVtable` defines any of the four prediction corners. NULL slots
are absent, and `context_corners` marks only the slots that receive a
`RLMeshPredictContext`. `rlmesh_model_new` delegates declarations, loading,
corner synthesis, serialization and the episode store to the facade Model.
`rlmesh_model_from_config` checks declared load keys before calling load.
The loaded native chunk is read once. Context state uses borrowed
`RLMeshEpisodeState` handles with owned `void*` values and deleters.

Use `RLMESH_MODEL_VTABLE_INIT`, `RLMESH_MODEL_OPTIONS_INIT` and
`RLMESH_MODEL_SERVE_OPTIONS_INIT`; zero initialization is refused. On a
constructor error neither policy nor hook state is adopted, no close hook
runs, and the output is NULL. Clone shares the model. Close is terminal,
serialized and idempotent; the last C model free drains callbacks and releases
policy and hook state once. Getter data remains valid while any C clone
lives. Serve uses the shared ServeOptions and reusable cancellation token.
An already-fired token refuses serving without closing the model.

`examples/c_policy.c` covers the C API. C++ detects the four Policy corners
and shares the same Model/run/session implementation.

## Layout

- `include/rlmesh.h` — the C ABI (C11). Hand-authored; the single header authority.
- `include/rlmesh.hpp` — RAII C++ wrapper over the C ABI (no exceptions, `Result<T>`).
- `examples/c_model.c`, `examples/cpp_model.cpp` — a zero-action model in each language.
- `examples/cpp_env.cpp` — a tagged C++ env (a point reaching a target, with a camera); `--describe` prints its image-label envelope.
- `examples/c_env.c` — a minimal C env printing its describe envelope.
- `examples/cpp_surface.cpp` — compile-only: names every wrapper type so a broken template fails CI.
- `examples/consumer/` — a CMake project consuming the packaged library via `find_package(rlmesh)`.
- `src/` — the Rust side: `extern "C"` projections over the core `rlmesh` crate.

## Quick demo

Serve any RLMesh environment, then point the C++ model at it:

```bash
# terminal 1: the Python quickstart env (or `cargo run -p rlmesh --example serve_env`)
uv run python examples/python/quickstart/serve.py --address 127.0.0.1:5555

# terminal 2: build the library and the C++ example, run 3 episodes
cargo build -p rlmesh-capi --lib
zig c++ -std=c++20 -I crates/rlmesh-capi/include crates/rlmesh-capi/examples/cpp_model.cpp \
  -L target/debug -lrlmesh_capi -Wl,-rpath,"$PWD/target/debug" -o /tmp/cpp_model
/tmp/cpp_model tcp://127.0.0.1:5555 3
```

A C++ policy takes a typed observation and returns a typed action. To sample
from the remote environment's action space:

```cpp
#include <rlmesh.hpp>

int main() try {
  rlmesh::RemoteEnv env({.address = "tcp://127.0.0.1:5555"});
  auto space = env.action_space();
  rlmesh::Model model([space](const rlmesh::Value&) { return space.sample(); });
  auto result = model.run(env, {.episodes = 2, .seeds = {0, 1}});
  auto session = model.session(env);
  auto [observation, info] = session.reset(0);
  auto [next, reward, terminated, truncated, next_info] =
      session.step(session.predict(observation));
} catch (const std::exception& error) {
  return 1;
}
```

See [`cpp_model.cpp`](examples/cpp_model.cpp), [`cpp_env.cpp`](examples/cpp_env.cpp),
and the corresponding [`c_model.c`](examples/c_model.c) and
[`c_env.c`](examples/c_env.c) for complete consumers. ABI 5 removes every
`rlmesh_ext_legacy_*` export; C and C++ now use the same facade APIs.

## Using it from CMake

Each GitHub Release from 0.1.1 on carries a prebuilt package,
`rlmesh-<version>-x86_64-unknown-linux-gnu.tar.gz` (Linux x86_64, glibc 2.34 or
newer), and its `.sha256`. It holds both headers, the shared and static
libraries, and CMake and pkg-config metadata, so no Rust toolchain is needed.

Fetch it at configure time:

```cmake
include(FetchContent)
FetchContent_Declare(rlmesh
  URL https://github.com/ArenaX-Labs/rlmesh/releases/download/v0.1.1/rlmesh-0.1.1-x86_64-unknown-linux-gnu.tar.gz
  URL_HASH SHA256=<the release's .sha256>
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(rlmesh)

target_link_libraries(my_sim PRIVATE rlmesh::rlmesh)
```

Or unpack it anywhere and use `find_package`:

```cmake
find_package(rlmesh CONFIG REQUIRED)  # -Drlmesh_DIR=<unpacked>/lib/cmake/rlmesh
target_link_libraries(my_sim PRIVATE rlmesh::rlmesh)
```

`rlmesh::rlmesh` links the shared library, which CMake's build rpath finds;
install your binary beside it or set an install rpath. `rlmesh::rlmesh_static`
is for fully static targets with no other Rust library linked in. The package
version is the ABI generation (`RLMESH_ABI_VERSION`), so
`find_package(rlmesh 5)` refuses a package of another ABI generation; call
`rlmesh_abi_check()` at startup to catch a header and library from different
releases.

## C++20

Include `rlmesh.hpp` and enable C++20 and exceptions. Fallible calls throw the
R-21 exception class: `ValueError`, `TypeError`, `RuntimeError`, `OSError`
(with `ConnectionError` and `TimeoutError` subclasses), or the
`RLMeshException` family. `ext::CancelledError` has no `std::exception` base,
so handle it explicitly. An exception thrown by an authored policy, env,
adapter, hook, or warning handler is rethrown unchanged when its error returns
in process. Remote errors preserve their documented class, code, text and
recoverability.

`Model` detects `predict`, `predict_chunk`, `predict_batch` and
`predict_chunk_batch`, independently detecting each corner's context and
execution-horizon parameters. Callables work directly. Optional policy members
include `load`, `on_episode_end`, `close`, and static spec/load/edition/chunk
declarations. `Model::from_config<Policy>(kwargs)` applies load declarations.
`types::PredictContext` exposes episode ids, optional seeds, predict indices,
and a typed `EpisodeState`. Batch contexts use `std::span`.

`Env` and `VectorEnv` detect Gymnasium-shaped methods and space getters.
`EnvServer` owns the env handles passed to it. Supply a `std::shared_ptr<E>`
when the host also needs the simulation. `EnvFactory<F>` uses CRTP to capture
static declarations, and `make_env` applies factory stamping. Factories cannot
be copied or moved; their C handles are built on first use. A foreground
`serve()` runs lane zero on the caller; `start()` runs callbacks on library
threads. `shutdown()` reports errors; destruction drains and swallows cleanup
errors.

`Value`, `Tensor`, spaces, `Model`, `RemoteModel`, and adapters share their
underlying objects on copy. `Tensor::copy()` explicitly copies data. Env,
remote-env, session, server and recorder handles are move-only. Runs and
sessions borrow targets; keep their model and env handles alive until the
session is destroyed. A run never closes its model, and closes an env only
when `close_env` is set. Call `close()` explicitly to observe cleanup errors.

Options are aggregates with Python defaults and support designated
initializers. Optional scalars use `std::optional`; `OptionalList<T>` preserves
None versus an empty list. Durations use `std::chrono::duration<double>` and
drop the `_seconds` suffix. Reset and step return tuples for structured
bindings. `ext::CancelToken` copies share a reusable cancellation source.

`adapters` (also `adapt`) exposes typed spec/tag trees, tolerant vocabularies,
transforms, resolution, history and custom `AdapterBase` subclasses. Use
`std::shared_ptr` for custom adapters; `wrap_predict` retains ownership.
`Session` exposes readers, frame sources and retained run contexts. `RunHooks`,
`RunResult`, `EpisodeResult`, `View`, and `Recorder` use the same names and
fields as Python. Tensor DLPack descriptors use `ext::DlpackPtr`; imports
consume ownership only on success, and exports retain their storage.

The old `Result<T>`, `RLMESH_TRY`, aborting accessors, `ValueRef`, `SpaceRef`,
worker `Batch`/`Request`, `Environment`, and combined `EnvConfig` are removed.
`-fno-exceptions` is unsupported. The executable fixtures in
[`cpp_parity.cpp`](examples/cpp_parity.cpp) cover the new surface.

## Host runtime in describe

The C++ server supplies its host declaration through ABI 5
`RLMeshExtEnvServerOptions.host_runtime_json`. In C, initializing that record
with `RLMESH_EXT_ENV_SERVER_OPTIONS_INIT` does the same. The header detects the
translation unit's language, standard and compiler. Shared model/factory
Describe reports C's native runtime; served env Describe includes the host's.
Use one C++ standard across a binary's translation units.

The optional C host JSON accepts `language`, `language_version`,
`toolchain` (`name` and `major.minor` version), and `framework_versions`.
Invalid declarations fail creation. The library owns component, package,
OS/architecture and protocol/edition identity. C++ supplies this declaration
automatically; its fenced server options expose token, peer metadata and
publication settings.

## Tasks

- `mise run dev:heavy -- mise run test:cxx` — compile the C11 and C++20 headers,
  run GCC and Clang C++ parity fixtures, and exercise C/C++ over real gRPC.
  The C header also compiles as C++98; C++ requires exceptions.
- `mise run test:cxx:pkg` — build installed-package consumers through pkg-config,
  CMake `find_package`, and FetchContent (a separate packaging gate).
- `mise run fmt:cxx` — format the headers and native examples.
