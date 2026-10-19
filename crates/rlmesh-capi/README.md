# RLMesh C API

RLMesh connects models to environments for evaluation. Serve a Gymnasium-style environment in one process and drive it with a model from another, on the same machine or across the network. `rlmesh.h` is the C API: a C11 header over a prebuilt shared library, for simulators and policies written in C, or for any language that binds to C.

RLMesh aims to feel familiar in whichever language you use it from. Python, Rust, C and C++ share the same nouns and constructs (`Env`, `Space`, `Value`, `Model`, `run`, `Session`, `EnvServer`, `RemoteEnv`), each written in that language's own conventions and with helpers that only make sense there. What you learn here carries over, and every language talks to every other over the same wire.

Writing C++? Use [`rlmesh.hpp`](README-cpp.md), the C++17 API over this header, with RAII handles and exceptions.

## Install

### Prebuilt package

Each [GitHub Release](https://github.com/ArenaX-Labs/rlmesh/releases) carries `rlmesh-<version>-x86_64-unknown-linux-gnu.tar.gz` (Linux x86_64, glibc 2.34 or newer) and its `.sha256`. It holds `rlmesh.h` and `rlmesh.hpp`, the shared library (`librlmesh_capi.so.5`) and a static archive, plus CMake and pkg-config metadata. No Rust toolchain is needed.

With CMake, fetch it at configure time:

```cmake
include(FetchContent)
FetchContent_Declare(rlmesh
  URL https://github.com/ArenaX-Labs/rlmesh/releases/download/v<version>/rlmesh-<version>-x86_64-unknown-linux-gnu.tar.gz
  URL_HASH SHA256=<the release's .sha256>
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(rlmesh)

target_link_libraries(my_sim PRIVATE rlmesh::rlmesh)
```

Or unpack it anywhere and use `find_package`:

```cmake
find_package(rlmesh 5 CONFIG REQUIRED)  # -Drlmesh_DIR=<unpacked>/lib/cmake/rlmesh
target_link_libraries(my_sim PRIVATE rlmesh::rlmesh)
```

Without CMake, use pkg-config:

```bash
export PKG_CONFIG_PATH=<unpacked>/lib/pkgconfig
cc -std=c11 serve.c $(pkg-config --cflags --libs rlmesh) -o serve
```

`rlmesh::rlmesh` links the shared library; install your binary beside it, set an rpath, or add `<unpacked>/lib` to `LD_LIBRARY_PATH`. `rlmesh::rlmesh_static` is for fully static targets that link no other Rust library. The CMake package version is the ABI generation, so `find_package(rlmesh 5)` refuses a package built for another generation.

### Windows (MSVC)

Releases also carry `rlmesh-<version>-x86_64-pc-windows-msvc.tar.gz` and its `.sha256`, built with the MSVC toolchain against the UCRT (Windows 10 or later). It holds the headers, `bin/rlmesh_capi.dll` (with its `.pdb`), the import library `lib/rlmesh_capi.lib` and the CMake metadata; there is no static library or pkg-config file. Windows 10 and later unpack it with the built-in `tar -xf`.

The CMake snippets above work unchanged with the Windows tarball's URL. `rlmesh::rlmesh` links the import library; copy `bin/rlmesh_capi.dll` beside your `.exe`, for example with a post-build step:

```cmake
add_custom_command(TARGET my_sim POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:rlmesh::rlmesh> $<TARGET_FILE_DIR:my_sim>)
```

Without CMake, add `<unpacked>\include` to the include path and link `<unpacked>\lib\rlmesh_capi.lib`:

```bat
cl /std:c11 /W4 /I <unpacked>\include serve.c /link <unpacked>\lib\rlmesh_capi.lib
cl /std:c++17 /EHsc /W4 /permissive- /I <unpacked>\include serve.cpp /link <unpacked>\lib\rlmesh_capi.lib
```

The headers declare every function `__declspec(dllimport)` on Windows. For Unreal Engine, link the import library from the consuming module's `.Build.cs` (`PublicAdditionalLibraries`), stage the DLL with `RuntimeDependencies`, and set `bEnableExceptions = true`, which `rlmesh.hpp` needs.

### From source

From a checkout of this repository (needs Rust and `protoc`):

```bash
cargo build -p rlmesh-capi --release   # target/release/librlmesh_capi.{so,a}
cc -std=c11 -I crates/rlmesh-capi/include serve.c \
  -L target/release -lrlmesh_capi -Wl,-rpath,"$PWD/target/release" -lpthread -ldl -lm -o serve
```

`mise run release:cxx:package` assembles the same package as a release for the host platform, under `target/capi-pkg/`. `mise run release:cxx:package --target windows` cross-builds the Windows package from Linux or macOS with [cargo-xwin](https://github.com/rust-cross/cargo-xwin), which needs `clang` and downloads the MSVC CRT and Windows SDK, accepting Microsoft's license for them.

## Quickstart

Serve an environment. The env is a vtable of callbacks plus your own state pointer:

```c
/* serve.c: a three-step counter env on 127.0.0.1:5555 */
#include <rlmesh.h>
#include <stdio.h>

typedef struct Counter {
  int64_t steps;
} Counter;

static RLMeshStatus counter_reset(void* self, const int64_t* seed, const RLMeshValue* options,
                                  RLMeshValue** out_observation, RLMeshValue** out_info) {
  (void)seed;
  (void)options;
  (void)out_info; /* left NULL: an empty info dict */
  ((Counter*)self)->steps = 0;
  return rlmesh_value_int(0, out_observation);
}

static RLMeshStatus counter_step(void* self, const RLMeshValue* action,
                                 RLMeshValue** out_observation, double* out_reward,
                                 bool* out_terminated, bool* out_truncated,
                                 RLMeshValue** out_info) {
  Counter* counter = (Counter*)self;
  (void)action;
  (void)out_info;
  counter->steps += 1;
  *out_reward = 1.0;
  *out_terminated = counter->steps >= 3;
  *out_truncated = false;
  return rlmesh_value_int(counter->steps, out_observation);
}

int main(void) {
  static Counter counter;
  RLMeshSpace* observation_space = NULL;
  RLMeshSpace* action_space = NULL;
  RLMeshEnv* env = NULL;
  RLMeshEnvServer* server = NULL;
  RLMeshEnvVtable vtable = RLMESH_ENV_VTABLE_INIT;
  RLMeshEnvAttributes attributes = RLMESH_ENV_ATTRIBUTES_INIT;
  RLMeshEnvServerOptions options = RLMESH_ENV_SERVER_OPTIONS_INIT;
  RLMeshStatus status;

  vtable.reset = counter_reset;
  vtable.step = counter_step;
  options.address = "127.0.0.1:5555";

  status = rlmesh_discrete_new(4, NULL, &observation_space);
  if (status == RLMESH_OK) status = rlmesh_discrete_new(2, NULL, &action_space);
  if (status == RLMESH_OK) {
    attributes.observation_space = observation_space;
    attributes.action_space = action_space;
    attributes.id = "Counter-v0";
    status = rlmesh_env_new(&vtable, &counter, &attributes, &env);
  }
  if (status == RLMESH_OK) status = rlmesh_env_server_new(&env, 1, &options, &server);
  if (status == RLMESH_OK) {
    puts("serving on 127.0.0.1:5555");
    status = rlmesh_env_server_serve(server); /* blocks until shutdown or Ctrl-C */
  }
  if (status != RLMESH_OK && status != RLMESH_ERR_CANCELLED) {
    fprintf(stderr, "error: %s\n", rlmesh_last_error_message());
  }
  rlmesh_env_server_free(server); /* free the server before the env it serves */
  rlmesh_env_free(env);
  rlmesh_space_free(action_space);
  rlmesh_space_free(observation_space);
  return status == RLMESH_OK || status == RLMESH_ERR_CANCELLED ? 0 : 1;
}
```

Run a model against it from another process. A model is a vtable too; the smallest one fills in `predict`:

```c
/* run.c: run a constant policy for three episodes */
#include <rlmesh.h>
#include <stdio.h>

static RLMeshStatus predict(void* self, const RLMeshValue* observation,
                            const RLMeshPredictContext* context, RLMeshValue** out_action) {
  (void)self;
  (void)observation;
  (void)context; /* NULL unless requested through context_corners */
  return rlmesh_value_int(0, out_action);
}

int main(void) {
  RLMeshModelVtable vtable = RLMESH_MODEL_VTABLE_INIT;
  RLMeshRunOptions options = RLMESH_RUN_OPTIONS_INIT;
  RLMeshModel* model = NULL;
  RLMeshRunResult* result = NULL;
  RLMeshStatus status;

  vtable.predict = predict;
  options.has_episodes = true;
  options.episodes = 3;

  status = rlmesh_model_new(&vtable, NULL, NULL, &model);
  if (status == RLMESH_OK) {
    status =
        rlmesh_model_run(model, rlmesh_address_target("127.0.0.1:5555"), &options, &result);
  }
  if (status != RLMESH_OK) {
    fprintf(stderr, "error: %s\n", rlmesh_last_error_message());
  } else {
    printf("%zu episodes, %lld steps, mean reward %.1f\n", rlmesh_run_result_num_episodes(result),
           (long long)rlmesh_run_result_total_steps(result), rlmesh_run_result_mean_reward(result));
  }
  rlmesh_run_result_free(result);
  rlmesh_model_free(model);
  return status == RLMESH_OK ? 0 : 1;
}
```

```bash
cc -std=c11 serve.c $(pkg-config --cflags --libs rlmesh) -o serve && ./serve
cc -std=c11 run.c $(pkg-config --cflags --libs rlmesh) -o run && ./run
# 3 episodes, 9 steps, mean reward 3.0
```

Neither side cares what language the other is written in: the same model can drive an env served from Python, Rust or C++, and the Python, Rust and C++ clients can drive this env.

## How the C API works

These rules hold across the whole header, so they are worth reading once.

**Handles and ownership.** Every object is an opaque handle (`RLMeshValue*`, `RLMeshSpace*`, `RLMeshModel*`, ...) created by a `rlmesh_<type>_new` (or similar) call that writes it to an `out` parameter. You own what you create and release it with `rlmesh_<type>_free`; every free function accepts NULL. Handles that hold resources also have `rlmesh_<type>_close`, which is idempotent and reports cleanup errors; free is still required after close. Pointers returned by accessors (a remote env's `action_space`, a dict entry from `rlmesh_value_dict_get`) are borrowed and stay valid while a handle sharing their parent lives.

**Outputs.** An out slot need not be initialized, and a successful call never reads or frees what it held, so free a previous result before reusing the slot. Variable-length text comes back as `RLMeshBytes` (free with `rlmesh_bytes_free`). Numeric arrays use `(out, cap, out_len)` buffers: pass `NULL, 0` to query the length, and a short buffer writes nothing.

**Strings.** Inputs that may hold any text, such as dict keys and text values, are length-bearing `RLMeshStr`; `rlmesh_str("...")` wraps a NUL-terminated string. Plain `const char*` options (addresses, ids) are NUL-terminated UTF-8, and NULL means "not set".

**Options structs and `_INIT`.** Every options struct, vtable, attributes record and out-struct starts with `struct_size` and has an `RLMESH_<NAME>_INIT` macro holding the defaults. Start from the macro and set the fields you need, or pass NULL for all defaults. An optional number is a value plus a `has_` flag (`has_episodes`, `has_port`, `has_max_episode_seconds`), so an explicit zero is distinct from "not set". A seed list is a pointer plus a length; a NULL pointer means no seeds, while a non-NULL pointer with length zero is an explicit empty list.

**User data.** Constructors that take callbacks also take a `void* user_data`, passed back to every callback. The library takes ownership of it only when the constructor returns `RLMESH_OK`; the vtable's `free_user_data` (NULL for none) then runs once, after the last library reference is gone.

**Threads.** Error details are per thread (see [Errors](#errors)). Callbacks may run on library threads, so their state must be safe to use from another thread; an env's callbacks run one at a time.

## Core concepts

### Values and tensors

`RLMeshValue` is an immutable, shareable value of one of ten kinds: None, Bool, Int, Float, Text, Bytes, Tensor, List, Tuple and Dict. Observations, actions and info dicts are all values: a Discrete action is an Int, a Box observation a Tensor, a Dict observation a Dict. Dicts keep insertion order. Composite constructors retain their children, so you still free your own handles:

```c
RLMeshValue* x = NULL;
RLMeshValue* label = NULL;
RLMeshValue* observation = NULL;
const RLMeshValue* field = NULL;
double value = 0.0;

rlmesh_value_float(0.5, &x);
rlmesh_value_text(rlmesh_str("left"), &label);
{
  RLMeshStr keys[] = {rlmesh_str("x"), rlmesh_str("label")};
  const RLMeshValue* values[] = {x, label};
  rlmesh_value_dict(keys, values, 2, &observation); /* retains x and label */
}
rlmesh_value_free(x);
rlmesh_value_free(label);

/* Borrowed; NULL (with RLMESH_OK) when the key is absent. */
if (rlmesh_value_dict_get(observation, rlmesh_str("x"), &field) == RLMESH_OK && field != NULL &&
    rlmesh_value_as_float(field, &value) == RLMESH_OK) {
  printf("x = %f\n", value);
}
rlmesh_value_free(observation);
```

Reads are strict: `rlmesh_value_as_float` on an Int returns `RLMESH_ERR_TYPE`. `rlmesh_value_kind` tells you what you hold.

`RLMeshTensor` is an immutable n-dimensional array. `rlmesh_tensor_new(buffer, len, shape, ndim, dtype, &out)` copies a buffer; `rlmesh_tensor_buffer` reads one back; strides are in bytes. Tensors exchange data with other libraries through DLPack: `rlmesh_tensor_from_dlpack` imports a `DLManagedTensorVersioned` (the DLPack 1.0 header is bundled in `rlmesh.h`) and `rlmesh_tensor_to_dlpack` exports one.

### Spaces

`RLMeshSpace` is a space with its own random sampler. `rlmesh_box_new`, `rlmesh_discrete_new` and `rlmesh_dict_new` build the common kinds (`RLMeshBox`, `RLMeshDiscrete` and `RLMeshDict` are the same handle type), and MultiBinary, MultiDiscrete, Text and Tuple constructors are available as Experimental. A space seeds, samples and checks membership; `rlmesh_space_spec` lends its immutable `RLMeshSpaceSpec`:

```c
const int64_t shape[] = {3};
RLMeshSpace* box = NULL;
RLMeshValue* sample = NULL;
const uint64_t seed = 7;

rlmesh_box_new(rlmesh_bound_float(-1.0), rlmesh_bound_float(1.0), shape, 1, NULL, &box);
rlmesh_space_seed(box, &seed, NULL);
rlmesh_space_sample(box, &sample); /* a float32 Tensor of shape [3] */
if (rlmesh_space_contains(box, sample)) puts("sample is in the box");
rlmesh_value_free(sample);
rlmesh_space_free(box);
```

### Environments and servers

An env is an `RLMeshEnvVtable` (`reset` and `step` are required; `render`, `close` and `free_user_data` are optional) plus `RLMeshEnvAttributes` naming its spaces, id, render mode and metadata. `reset` and `step` exchange values in Gymnasium's order, and a NULL info means an empty dict. `rlmesh_vector_env_new` builds a natively batched env from an `RLMeshVectorEnvVtable`.

`rlmesh_env_server_new` serves one or more envs (several copies serve concurrent sessions) and `rlmesh_env_server_new_vector` a vector env. The server binds at construction; `rlmesh_env_server_address` reports where. `rlmesh_env_server_serve` blocks on the calling thread; `rlmesh_env_server_start` serves in the background, after which `rlmesh_env_server_wait` and `rlmesh_env_server_shutdown` control it. The server leases its envs until it is freed, so free the server first.

`RLMeshEnvFactory` (`rlmesh_env_factory_new`) builds envs on demand, with discriminant parameters (`RLMeshTagParam`), a catalog of named variants, and `rlmesh_env_factory_serve`. See [`examples/c_env_api.c`](examples/c_env_api.c) and [`examples/c_factory.c`](examples/c_factory.c).

`rlmesh_ext_env_factory_start(factory, address, options, &server)` prepares the factory, makes and stamps its envs, publishes the describe envelope, and starts serving in the background without installing signal handlers. Control the returned server with `rlmesh_env_server_wait`, `_shutdown` and `_free`. It owns its envs; the factory may be reused or freed immediately after start returns. A start overlapping a session borrow or another factory hook returns `RLMESH_ERR_RUNTIME`. Failed starts close and free all envs made so far and leave the factory reusable. Start subscribes to the cancellation token after factory admission and carries that subscription through preparation and serving; the caller may free its token handle after start returns. Cancelling drains and closes the envs; whichever of wait or shutdown first observes the stop reports cancellation, and later calls succeed. A shutdown without a firing succeeds. A firing during drain or close after another stop has begun cannot change its result.

Both factory serve calls accept `RLMeshEnvFactoryServeOptions.ext`, pointing to an `RLMeshExtEnvFactoryServeOptions` initialized with `RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_INIT`. Its `host_runtime_json` declares optional `language`, `language_version`, `framework_versions` and `toolchain` host fields for the handshake PeerInfo and its published describe envelope. Unknown fields are refused. A supplied toolchain must name `gcc`, `clang` or `msvc` and a `major.minor` version. NULL keeps the legacy C runtime. Older ABI-5 options allocations are accepted; appended fields default to zero.

### Models

An `RLMeshModelVtable` fills in any of four prediction corners: `predict` (one observation to one action), `predict_batch`, `predict_chunk` (one observation to several actions) and `predict_chunk_batch`. Optional slots cover `load`, `on_episode_end` and `close`, and fields declare a `spec` for adapters, a `workflow_edition`, `load_keys` and a static `native_chunk`. `rlmesh_model_new` builds a model and `rlmesh_model_from_config` builds one from a config dict passed to `load`.

A corner whose bit is set in `context_corners` receives an `RLMeshPredictContext`: the episode id, its seed, the predict index, a derived per-predict seed and an `RLMeshEpisodeState` for per-episode scratch data (`rlmesh_episode_state_get` / `_set` / `_remove`):

```c
static RLMeshStatus predict(void* self, const RLMeshValue* observation,
                            const RLMeshPredictContext* context, RLMeshValue** out_action) {
  (void)self;
  (void)observation;
  printf("episode %s, predict %llu\n", context->episode_id,
         (unsigned long long)context->predict_index);
  return rlmesh_value_int((int64_t)(context->predict_index % 2), out_action);
}

/* ... */
vtable.predict = predict;
vtable.context_corners = RLMESH_CORNER_PREDICT;
```

`rlmesh_model_serve` serves a model as an endpoint other runs can use. Set `RLMeshExtModelServeOptions.no_signal_handlers` through the serve options' `ext` pointer when the host owns signal handling. The call still blocks and can be stopped with its cancellation token or `rlmesh_model_close`. The default false installs and restores SIGINT/SIGTERM handlers. See [`examples/c_policy.c`](examples/c_policy.c).

### Running and sessions

`rlmesh_model_run` drives whole episodes and returns an `RLMeshRunResult` (`rlmesh_run_result_num_episodes`, `_total_steps`, `_mean_reward`, `_success_rate`, `_episodes_get` for each `RLMeshEpisodeResult`, plus telemetry). The env argument is an `RLMeshEnvTarget`, made from an address (`rlmesh_address_target`), a running server (`rlmesh_env_server_target`), a remote client (`rlmesh_remote_env_target`), or a local env, vector env or factory (`rlmesh_env_target`, served on a private loopback for the run). `rlmesh_run` also takes an `RLMeshModelTarget`: a model (`rlmesh_model_target`), a remote model, or `RLMESH_RANDOM_SAMPLE`, a baseline that samples the action space. A run borrows its targets: it never closes the model and closes the env only when `close_env` is set.

`RLMeshRunOptions` covers `episodes`, `seeds`, `max_episode_steps`, `max_episode_seconds`, `hooks`, `instruction`, `close_env` and more. `RLMeshRunHooks` (built with `rlmesh_run_hooks_new` from an `RLMeshRunHooksVtable`) observe a run on the calling thread: run start and end, episode start and end, and every step.

For step-by-step control, open a session. Nothing connects until first use, and closing a session never closes the model:

```c
RLMeshSession* session = NULL;
RLMeshValue* observation = NULL;
RLMeshValue* info = NULL;
const int64_t seed = 0;
bool done = false;

if (rlmesh_model_session(model, rlmesh_address_target("127.0.0.1:5555"), NULL, &session) ==
        RLMESH_OK &&
    rlmesh_session_reset(session, &seed, NULL, &observation, &info) == RLMESH_OK) {
  while (!done) {
    RLMeshValue* action = NULL;
    RLMeshValue* next = NULL;
    RLMeshValue* step_info = NULL;
    double reward = 0.0;
    bool terminated = false, truncated = false;
    if (rlmesh_session_predict(session, observation, &action) != RLMESH_OK) break;
    RLMeshStatus status = rlmesh_session_step(session, action, &next, &reward, &terminated,
                                              &truncated, &step_info);
    rlmesh_value_free(action);
    if (status != RLMESH_OK) break;
    rlmesh_value_free(observation);
    rlmesh_value_free(step_info);
    observation = next;
    done = terminated || truncated;
  }
}
rlmesh_value_free(observation);
rlmesh_value_free(info);
if (session != NULL) rlmesh_session_close(session);
rlmesh_session_free(session);
```

`rlmesh_session_run` runs whole episodes inside an open session. See [`examples/c_run.c`](examples/c_run.c).

### Remote clients

`rlmesh_remote_env_new` connects to a served env so you can step it directly, Gymnasium style; `rlmesh_remote_vector_env_new` does the same for a vector env. `rlmesh_remote_model_new` points at a served model; it validates its options without connecting, and runs and sessions connect as they need to. Addresses are `host:port`, `tcp://host:port` or `unix:///path/to.sock`, or set `host`, `port`, `path` and `transport` separately in the options.

```c
RLMeshRemoteEnvOptions options = RLMESH_REMOTE_ENV_OPTIONS_INIT;
RLMeshRemoteEnv* env = NULL;
RLMeshValue* observation = NULL;
RLMeshValue* info = NULL;
RLMeshValue* action = NULL;
double reward = 0.0;
bool terminated = false, truncated = false;

options.address = "127.0.0.1:5555";
options.has_request_timeout_seconds = true;
options.request_timeout_seconds = 5.0;
if (rlmesh_remote_env_new(&options, &env) == RLMESH_OK &&
    rlmesh_remote_env_reset(env, NULL, NULL, &observation, &info) == RLMESH_OK &&
    rlmesh_space_sample(rlmesh_remote_env_action_space(env), &action) == RLMESH_OK) {
  rlmesh_value_free(observation);
  rlmesh_value_free(info);
  rlmesh_remote_env_step(env, action, &observation, &reward, &terminated, &truncated, &info);
}
rlmesh_value_free(action);
rlmesh_value_free(observation);
rlmesh_value_free(info);
if (env != NULL) rlmesh_remote_env_close(env);
rlmesh_remote_env_free(env);
```

See [`examples/c_remote.c`](examples/c_remote.c).

### Adapters

When a model and an env describe their observations and actions differently (camera names, state layouts, action components), adapters bridge them declaratively. An env publishes `RLMeshEnvTags` (the `tags` field of the server options, or `rlmesh_adapters_tag`), a model declares an `RLMeshModelSpec`, and RLMesh resolves an adapter for each route. Both are built from typed node trees or from their dict and JSON forms. See [`examples/c_adapters.c`](examples/c_adapters.c).

### View, Recorder and describe

`RLMeshView` (`RLMESH_VIEW_INIT`, or `rlmesh_view_parse` with `"term"` or `"http:8008"`) shows live frames when set on session options. `rlmesh_recorder_new` collects run results or captures live runs through the hooks from `rlmesh_recorder_capture`, and exports them with media as a folder or ZIP bundle ([`examples/c_recorder.c`](examples/c_recorder.c)). `rlmesh_model_describe` and `rlmesh_env_factory_describe` report declarations ([`examples/c_describe.c`](examples/c_describe.c)), and `rlmesh_build_info` the library's version, protocol generation and workflow editions.

## Errors

Every fallible call returns an `RLMeshStatus`: `RLMESH_OK`, or a code for the kind of failure (`RLMESH_ERR_INVALID_ARGUMENT`, `RLMESH_ERR_TYPE`, `RLMESH_ERR_ENVIRONMENT`, `RLMESH_ERR_MODEL`, `RLMESH_ERR_CONNECTION`, `RLMESH_ERR_TIMEOUT`, `RLMESH_ERR_PROTOCOL`, `RLMESH_ERR_RUNTIME`, `RLMESH_ERR_CANCELLED`, ...). After a failure, the calling thread can read the details until its next RLMesh call:

- `rlmesh_last_error_message()`: the message, the same text every RLMesh language reports for that failure.
- `rlmesh_last_error_class()`: the exact class, finer than the status (for example `RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION`); `rlmesh_error_class_is_subclass` tests family membership and `rlmesh_error_class_name` names it.
- `rlmesh_last_error_is_recoverable()` and `rlmesh_last_error_code()` for environment and model failures.

```c
RLMeshStatus status = rlmesh_model_run(model, target, &options, &result);
if (status != RLMESH_OK) {
  RLMeshErrorClass cls = rlmesh_last_error_class();
  if (rlmesh_error_class_is_subclass(cls, RLMESH_ERROR_CLASS_OS_ERROR)) {
    fprintf(stderr, "could not reach the env: %s\n", rlmesh_last_error_message());
  } else {
    fprintf(stderr, "%s: %s\n", rlmesh_error_class_name(cls), rlmesh_last_error_message());
  }
}
```

A callback fails by setting the error and returning nonzero. `rlmesh_callback_set_error(message, recoverable)` reports an environment or model failure; a recoverable env error fails one request while the client keeps its session. `rlmesh_callback_set_error_class` reports an exact class instead, such as `RLMESH_ERROR_CLASS_VALUE_ERROR`:

```c
static RLMeshStatus arm_step(void* self, const RLMeshValue* action, RLMeshValue** out_observation,
                             double* out_reward, bool* out_terminated, bool* out_truncated,
                             RLMeshValue** out_info) {
  (void)self;
  (void)action;
  (void)out_observation;
  (void)out_reward;
  (void)out_terminated;
  (void)out_truncated;
  (void)out_info;
  rlmesh_callback_set_error("target out of reach", true);
  return RLMESH_ERR_ENVIRONMENT;
}
```

## Workflow editions

Every served env and model runs at a workflow edition, which pins what a session means across upgrades. Declare one (the `workflow_edition` field of `RLMeshServeOptions`, the model vtable, or the run and session options, or the `RLMESH_WORKFLOW_EDITION` environment variable); a participant that declares none floats to the newest edition and warns once. `rlmesh_current_workflow_edition()` returns the edition to declare. See the [documentation](https://rlmesh.dev/docs/).

## C-only extras

Additions that exist only for native callers carry the `rlmesh_ext_` prefix (types `RLMeshExt...`):

- Cancellation: `rlmesh_ext_cancel_token_new`, `_clone`, `_cancel` and `_free`. Put a token in the `cancel` field of run, session or serve options and cancel it from any thread; the call in flight returns `RLMESH_ERR_CANCELLED`. Remote clients and sessions expose their token through `_cancel_handle`.
- Bearer-token endpoints: `RLMeshExtEnvServerOptions` and `RLMeshExtModelServeOptions` (through the `ext` field of the serve options) set the token a server expects; clients send it as `api_key`.
- `rlmesh_ext_env_server_describe_json` for a served env's describe envelope, `rlmesh_ext_register_entrypoint` for trusted custom adapter transforms, and helpers such as `rlmesh_ext_dtype_size`.

## Compatibility

The C API is **Experimental**. Binary compatibility is tracked by the ABI generation, currently 5: `RLMESH_ABI_VERSION`, the `librlmesh_capi.so.5` SONAME and the CMake package version all carry it. An incompatible change (a struct layout, an enum value, a signature or a removed symbol) bumps the generation, which can happen in any minor release; a binary built for another generation will not load. Within a generation, structs grow only by appending optional fields whose zero bytes mean the default, so code compiled against an older header keeps working. Call `rlmesh_abi_check()` at startup if you load the library with `dlopen` or otherwise bypass the SONAME. `RLMESH_VERSION` is the header's package version and `rlmesh_build_info()` reports the linked library's. See the [compatibility policy](https://rlmesh.dev/docs/releases/compatibility/).

The header is C11 and also compiles warning-free as C++98, C++17 and C++20. It bundles the upstream DLPack 1.0 header (license in [`include/dlpack.LICENSE`](include/dlpack.LICENSE)). Creating a model or env handle ignores `SIGPIPE` on POSIX unless your program installed its own handler, so a peer that disconnects mid-write cannot kill the process.

## C++17

[`rlmesh.hpp`](include/rlmesh.hpp) is a header-only C++17 API over this header, with RAII handles, exceptions, `std::optional`, `std::chrono` and aggregate options with portable `with_<field>` setters (designated initializers are a C++20 convenience). It has its own guide: [RLMesh C++ API](README-cpp.md).

## Examples

[`examples/`](examples) holds runnable programs, all built and run by `mise run test:cxx`:

- [`c_model.c`](examples/c_model.c): a model that reads the env's action space and runs episodes against a remote env.
- [`c_env.c`](examples/c_env.c): a minimal env and its served describe envelope.
- Per-area programs: `c_spaces.c`, `c_dlpack.c`, `c_env_api.c`, `c_factory.c`, `c_policy.c`, `c_run.c`, `c_reads.c`, `c_remote.c`, `c_adapters.c`, `c_describe.c`, `c_recorder.c`, `c_core.c` and `c_abi5.c` (struct sizing rules).
- [`consumer/`](examples/consumer): a CMake project that builds against the installed package.

## Development

- `mise run dev:heavy -- mise run test:cxx`: compile the headers as C11, C++98, C++17 and C++20, build and run every example, and exercise C and C++ models and envs over real gRPC. `CC` and `CXX` override the compilers; locally, the default checks both zig’s bundled Clang and GCC when installed, at both C++ standards.
- `mise run test:cxx:pkg`: build the package, then build `examples/consumer` against it through pkg-config, `find_package` and FetchContent.
- `mise run fmt:cxx`: format the headers and examples.

## Links

- Documentation: https://rlmesh.dev/docs/
- C++ API: [README-cpp.md](README-cpp.md)
- Rust crate: https://crates.io/crates/rlmesh
- Python package: https://pypi.org/project/rlmesh/

## Per-call cancellation

`rlmesh_ext_session_reset`, `rlmesh_ext_session_predict`,
`rlmesh_ext_session_step`, `rlmesh_ext_remote_env_reset`,
`rlmesh_ext_remote_env_step`, `rlmesh_ext_remote_vector_env_reset` and
`rlmesh_ext_remote_vector_env_step` take a borrowed
`const RLMeshExtCancelToken* cancel` immediately before the first output.
Pass NULL for the ordinary call's behavior. The vector step retains its
`capacity` argument after the numeric output buffers.

Fire the token from another thread to cancel the operation with
`RLMESH_ERR_CANCELLED`; the handle stays usable. An idle firing fails the next
admitted call using that token once. Busy, closed or overlapping calls consume
no firing. Each admitted call watches both this token and its handle token,
including lazy session factory creation. The caller can free the token after
the call returns. One token can cancel multiple calls and a started server.

Cancellation is checked synchronously after native result processing and each
subscription is detached under its token lock. A firing before that completion
boundary reports cancellation even if the polling watcher has not run. With an
idle token, a firing after completion remains pending for its next call.
Vector-step actions are validated before subscribing; malformed actions leave
an idle firing pending.
