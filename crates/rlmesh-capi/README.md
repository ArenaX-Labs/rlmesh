# rlmesh-capi

Experimental C ABI for RLMesh, plus a header-only C++17 wrapper, for both sides
of the wire:

- **Models.** A C or C++ program implements `predict`, and the RLMesh runtime
  does the rest: connect to an environment, decode observations, encode actions,
  run episodes, or serve the model as a gRPC `ModelService` endpoint.
- **Environments.** A C or C++ simulator implements `reset` / `step` (and
  optionally `render` / `close`), declares its spaces and adapter tags, and is
  served as a gRPC `EnvService` endpoint that any RLMesh model — Python, Rust,
  C++ — drives.

Status: alpha. The ABI (`RLMESH_ABI_VERSION`) will break before 1.0.

## ABI 5 values and tensors

ABI 5 replaces the space-specific value projection with immutable core
`RLMeshValue` handles: None, Bool, Int, Float, Text, Bytes, Tensor, List,
Tuple and Dict. A Discrete action is an Int; a Box observation and a
MultiBinary or MultiDiscrete sample are Tensors. Constructors return a
status and fill an owned handle. `rlmesh_value_clone` shares a reference;
`rlmesh_value_free` releases it. Composite constructors retain references to
children, so callers still free their own handles after success or failure.
Dicts preserve insertion order, permit empty and embedded-NUL keys, and
replace repeated keys in place. Metadata and InfoDict use the same Dict kind.

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
`rlmesh_tensor_free`. The old tensor POD, DLPack dtype triple, tensor release
function and package-semver ABI accessors are removed. Use
`rlmesh_build_info` for package identity. Package SONAME generation is read
from `RLMESH_ABI_VERSION`, now 5.

The rest of the C parity migration is ongoing: the legacy env/model/run
interfaces below have not yet become the full facade surface. In particular,
the facade on this branch has no native DLPack import/export methods, and
C does not expose them. C++ receives only compatibility changes for the new C
value and tensor calls; its full parity API is a separate wave.

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
zig c++ -std=c++17 -I crates/rlmesh-capi/include crates/rlmesh-capi/examples/cpp_model.cpp \
  -L target/debug -lrlmesh_capi -Wl,-rpath,"$PWD/target/debug" -o /tmp/cpp_model
/tmp/cpp_model tcp://127.0.0.1:5555 3
```

The whole model, in C++:

```cpp
#include <cstdio>
#include <rlmesh.hpp>

int main() {
  // One action per predict: the neutral value of whatever the route's action
  // space turns out to be (Box, Discrete, Text, Multi*, Dict, Tuple).
  auto model = rlmesh::Model::from_predict([](const rlmesh::Request& request) {
    return rlmesh::zeros_for(request.action_space());
  });
  if (!model) return 1;
  model->on_episode_end([](std::string_view, std::string_view id) { /* ... */ })
      .on_close([] { /* ... */ });

  rlmesh::RunOptions options;
  options.max_episodes = 3;
  options.seeded = true;
  options.base_seed = 7;
  auto report = model->run_local("tcp://127.0.0.1:5555", options);
  if (!report) {
    std::fprintf(stderr, "run failed: %s\n", report.error().message().c_str());
    return 1;
  }
  std::printf("%lld episodes, mean reward %.2f\n",
              static_cast<long long>(report->total_episodes), report->mean_reward);
}
```

And in C:

```c
#include <rlmesh.h>

static int predict(void* ud, const RLMeshObservation* obs, RLMeshValue** out) {
  for (size_t i = 0; i < obs->num_envs; ++i) {
    RLMeshStatus status = rlmesh_value_int(0, &out[i]);
    if (status != RLMESH_OK) return status;
  }
  return RLMESH_OK;
}

int main(void) {
  RLMeshModelVtable vt = {.struct_size = sizeof vt, .predict = predict};
  RLMeshModel* model;
  rlmesh_model_new(&vt, NULL, &model);
  RLMeshStatus rc = rlmesh_model_run_local(model, "tcp://127.0.0.1:5555", NULL, NULL);
  rlmesh_model_free(model);
  return rc == RLMESH_OK ? 0 : 1;
}
```

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

## Model contract

`predict` receives `num_envs` decoded observation values (one per sub-env) plus
routing metadata (session, env, request ids; one `RLMeshEpisode` per row). It
writes one owned action value per row into `out_actions` and returns `RLMESH_OK`,
or returns nonzero after `rlmesh_callback_set_error(...)` to decline. The runtime
validates each action against the route's action space before it reaches the
wire: a structural mismatch fails the step, a Box-bounds overshoot is left to the
environment's own policy.

Each `RLMeshEpisode` row carries the episode's `id` and reset `seed` (when
`seeded`), plus the same per-predict context the Python SDK stamps on a predict:
`predict_index`, the re-plan ordinal within the episode (0 on the first predict
under that id, then +1 per predict), and `predict_seed`, a reproducible mix of
the episode seed and that ordinal (`rlmesh.predict_seed` in the SDK; meaningful
only when `seeded`). The capi counts these per episode id and drops the counter
when the episode ends, holding at most 65,536 live episodes as the Python SDK
does. It never evicts one: a predict that would start a new episode past that
fails (`RLMESH_ERR_INTERNAL`, where Python raises `RuntimeError`) before the
callback runs.

Optional hooks: `on_episode_end(env_id, episode_id)` when the runtime drops an
episode (`episode_id == NULL` means every episode of that env), and `on_close`
once per handle: when a serve stops, or at `rlmesh_model_free`. A run borrows
the model and never closes it, as Python's `Model.run` does. Callbacks run on a worker thread; `user_data` must be safe to
use from a thread other than the one that created it, and a callback must not
re-enter its own model handle. `rlmesh_model_run_local` fills an optional
`RLMeshRunReport`; `rlmesh_model_cancel` stops a blocking run/serve from another
thread.

`RLMeshRunOptions` bounds and seeds a run: `max_episodes` (the exact episode
count, as in the Python SDK's `run(episodes=...)`; 0 runs one episode, or one per
seed when `episode_seeds` is given, never "until the env ends"), `base_seed` (when
`seeded`) or explicit `episode_seeds`, the per-episode step/time caps,
`execution_horizon`, and `trial_index_base` (when `trial_indexed`) — the first
trial ordinal the episodes walk, delivered as `reset(options={"trial_index": k})`
to an environment that declares that reset option.

## Environment contract

`rlmesh_env_new(vtable, config, user_data, &env)` takes the callbacks and an
`RLMeshEnvConfig`: the observation and action spaces (built with
`rlmesh_box_new`, `rlmesh_discrete_new` or `rlmesh_dict_new` and lent with
`rlmesh_space_spec(space)`), an id, the adapter `EnvTags` as JSON, the reset
options the env understands (`trial_index`), a render mode, and extra metadata.
The tags are validated against the spaces right there, the same publish-time
check Python's `adapters.tag()` runs, so a typo fails `rlmesh_env_new` rather
than a model's resolve. The tag JSON grammar (leaf kinds, fields, defaults, and
validation) is specified in [`docs/specs/env_tags.v1.md`](../../docs/specs/env_tags.v1.md).

Then `rlmesh_env_bind` (learn the address, e.g. for port 0) and
`rlmesh_env_serve` (blocks). `rlmesh_env_cancel` from any thread drains the
server, runs `close` once, and lets `serve` return `RLMESH_OK`.

When `rlmesh_env_serve` (or `rlmesh_env_free`) returns, no callback is running
and none will run again, so the callback state behind `user_data` can be freed
right after. Serve waits for a callback still in flight, such as a `step` the
`drain_timeout_ms` gave up on, and for every lane's `close`, however long it
takes: a callback that never returns blocks serve. `close_timeout_ms` does not
apply to an env.

`rlmesh_env_describe_json` (C++: `EnvServer::describe_json()`) returns the env's
`rlmesh.describe.v1` envelope: the spaces, the tags, and the edition handshake,
the same artifact `rlmesh describe` emits for a Python env. `rlmesh_env_bind`
puts it on that endpoint's handshake too (each env bound in a process reports
its own), so the managed platform can read an image without a label; bake it
as the image's `dev.rlmesh.describe` label to describe the image before it
runs.

An image that serves several named scenarios (each picked by the make kwargs or
environment variables a `dev.rlmesh.package` scenario sets) lists them in
`RLMeshEnvConfig.variants_catalog_json` (C++: `EnvConfig::variants_catalog_json`),
a JSON array of `{"id", "params", "metadata"}` with unique ids. The envelope
carries it as `variants.catalog`, as `enumerate_variants` does for a Python env,
so the platform can match the label's scenarios to it.

By default the env is served as one lane: every callback runs on one dedicated
thread, one call at a time. `reset` gets the seed and trial index (and every
reset option as JSON); `step` borrows the action and writes an owned
observation, the reward, the terminated/truncated flags and an optional info
JSON object; `render` writes an owned uint8 `[H, W, 3]` image that the capi
PNG-encodes. A callback fails a request by returning nonzero after
`rlmesh_callback_set_error`; that client's session ends and the env keeps
serving.

In C++, subclass `rlmesh::Environment` and hand it to `rlmesh::EnvServer`:

```cpp
class Reach : public rlmesh::Environment {
  rlmesh::Result<rlmesh::ResetOutput> reset(const rlmesh::ResetArgs& args) override;
  rlmesh::Result<rlmesh::StepOutput> step(std::optional<rlmesh::ValueRef> action) override;
};

rlmesh::EnvConfig config;
config.observation_space = rlmesh::Space::box<float>({3}, -1, 1).unwrap();
config.action_space = rlmesh::Space::box<float>({3}, -1, 1).unwrap();
config.adapter_tags_json = R"({"observation": {...}, "action": {...}})";
auto server = rlmesh::EnvServer::create(std::make_unique<Reach>(), config);
std::printf("listening on %s\n", server->bind("0.0.0.0:50051")->c_str());
server->serve();
```

Any model drives it, for example from Python:
`rlmesh.numpy.Model(predict, spec=SPEC).run("127.0.0.1:50051", episodes=3)`.
`examples/chrono/` is a full Project Chrono simulation served this way.

### Lanes and the foreground thread

`rlmesh_env_new_lanes(vtable, config, user_data, num_lanes, &env)` serves
`num_lanes` independent simulations as the lanes of one `num_envs = num_lanes`
endpoint, for a model that batches across them. Lane `i` passes `user_data[i]`
to its callbacks; every lane shares the vtable and config (one contract) but
runs on its own thread, concurrently with the others, so each `user_data` must
be its own simulation. `num_lanes == 0` is rejected.

A simulation bound to the thread that created it (a GL or Vulkan context, Isaac
Sim) sets `RLMeshEnvConfig.foreground = true`: `rlmesh_env_serve` then runs the
server on a helper thread and every callback, `close` included, on the thread
that called it, until the server stops. `rlmesh_env_cancel` works as before,
and PNG encoding stays off that thread. A foreground env serves exactly one
lane: `rlmesh_env_new_lanes` rejects `foreground` with more than one.

In C++, `EnvServer::create` also takes a
`std::vector<std::unique_ptr<Environment>>`, one lane per element, and
`EnvConfig::foreground` selects the foreground thread.

### In C++

`Model::from_predict` takes a single-env policy — `Result<Value>(const Request&)`,
where `Request` carries `observation()` (a `std::optional<ValueRef>`, absent when
the route sends none), `episode()` (an `Episode`: `id`, optional `seed`,
`predict_index`, optional `predict_seed`), and `action_space()` /
`observation_space()` as `SpaceRef`. `Model::from_predict_batch` takes the whole `Batch` and returns one
action per row. `zeros_for(SpaceRef)` builds the neutral action for _any_ space,
so a policy never has to switch on the kind to get started.

`ValueRef` reads all seven value kinds and `SpaceRef` walks a space spec (bounds,
`n`/`start`, text limits + charset, `nvec`, dict/tuple children by key or by
index); `Value`'s constructors build all seven, honoring the C constructors'
all-or-nothing ownership.

Every fallible call returns `Result<T>` (`Status` is `Result<void>`): a capi call
has already recorded its message on the thread, and `Error::from_last(status)`
snapshots it. Nothing throws — `value()` / `unwrap()` abort instead, and the
header compiles under `-fno-exceptions`. `RLMESH_TRY(expr)` propagates an error
out of a `Result`-returning function.

`run_local` takes a `RunOptions` (the C `RLMeshRunOptions` field for field, with
`episode_seeds` as a `std::vector`) and returns a `RunReport`; `serve` takes a
`ServeOptions` (owned token, `std::chrono` timeouts). `Model::cancel()` is the one member callable while
`run_local` / `serve` blocks — including from another thread; a cancelled
`serve` returns cleanly, a cancelled `run_local` fails with
`Error::is_cancelled()`. Never destroy a `Model` from inside its own callback.

## Host runtime in describe

`rlmesh.hpp` fills `EnvConfig::host_runtime_json` automatically from the host's
translation unit: `language: "c++"`, the standard (`"17"`, `"20"`, `"23"`, etc.)
from `__cplusplus` (or `_MSVC_LANG`), and the compiler under
`toolchain: {"name": "gcc", "version": "16.2"}`. Names are `gcc`, `clang`, or
`msvc`; versions use `major.minor`, including MSVC (`_MSC_VER` 1944 becomes
`"19.44"`). Detection does not populate `framework_versions`; that map is for
frameworks the host explicitly declares, such as Torch.
Clang is detected before its GNU compatibility macros. An unknown compiler is
omitted. This reports the compiler that built the host, rather than Rust's
compiler that built the ABI library.

The C++ `EnvConfig` default is inline. A binary that mixes C++ standards across
translation units has differing inline definitions (an ODR violation); the
reported standard may be whichever definition the linker keeps, depending on
link order. Use one standard throughout the binary, or explicitly declare the
runtime in each config instead of relying on that default.

In C, initialize the config with the header helper, then fill its spaces:

```c
RLMeshEnvConfig config = rlmesh_env_config_init();
config.observation_space = observation;
config.action_space = action;
```

The helper sets `struct_size`, zeroes other fields, and declares `"c"` with the
standard from `__STDC_VERSION__` (`"11"`, `"17"`, etc.). When that macro is
absent (including MSVC's default C mode), `language_version` is omitted.
It also detects C++ when the C header is included from a C++ translation unit.
`RLMESH_HOST_RUNTIME_JSON` exposes the detected declaration as a `const char*`.

For an explicit declaration, set `RLMeshEnvConfig.host_runtime_json` (C++:
`EnvConfig::host_runtime_json`) to a JSON object like:

```json
{
  "language": "c++",
  "language_version": "20",
  "toolchain": { "name": "clang", "version": "19.1" },
  "framework_versions": { "torch": "2.1" }
}
```

Only these four optional keys are accepted: `language` and `language_version`
are strings; `framework_versions` maps framework names to version strings;
`toolchain` is an object containing exactly the string fields `name` and
`version`, with the compiler names and `major.minor` format above. All strings
and framework names are trimmed, then rejected if empty. Framework names that
collide after trimming are rejected. The declaration is copied at creation and
remains on the envelope after bind, including its per-server handshake copy. Component,
package version, OS/architecture, and protocol/edition fields remain owned by
the library. Invalid declarations fail creation with `RLMESH_ERR_INVALID_ARGUMENT`.

`NULL` (C++: an empty string) preserves the legacy runtime exactly: `"c"` with
no language version or compiler. Hosts built against older headers also keep
that output. A new header with an old library silently emits the old `"c"`
output too: the old
library ignores the appended field. The field is guarded by `struct_size`, so
For the legacy env config below, C callers that zero-initialize a config
can adopt `rlmesh_env_config_init()` to opt into detection. Native models do
not currently emit a describe envelope and have no model config to extend.

## Tasks

- `mise run test:cxx` — compile the headers as C11 and C++17 under `-Werror`
  (the C header also compiles as C++98 under `-pedantic`),
  build the examples, run each model end-to-end against a live env, and drive
  the C++ env with a model (`e2e_env_harness`), and check C11/C17 and C++17/C++20
  describe runtimes and byte-for-byte legacy defaults.
- `mise run test:cxx:pkg` — package (`release:cxx:package`) and build the C++
  model against the installed package via pkg-config, CMake `find_package`, and
  CMake `FetchContent` from the tarball URL, plus the native env describe
  runtimes through the CMake consumer.
- `mise run fmt:cxx` — clang-format the headers and examples.
