# rlmesh-capi

Experimental C ABI for RLMesh, plus a header-only C++17 wrapper. A C or C++
program implements `predict`, and the RLMesh runtime does the rest: connect to an
environment, decode observations, encode actions, run episodes, or serve the
model as a gRPC `ModelService` endpoint.

Status: alpha. The ABI (`RLMESH_ABI_VERSION`) will break before 1.0.

## Layout

- `include/rlmesh.h` — the C ABI (C11). Hand-authored; the single header authority.
- `include/rlmesh.hpp` — RAII C++ wrapper over the C ABI (no exceptions, `Result<T>`).
- `examples/c_model.c`, `examples/cpp_model.cpp` — a zero-action model in each language.
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

static int predict(void* ud, const RlmeshObservation* obs, RlmeshValue** out) {
  for (size_t i = 0; i < obs->num_envs; ++i) out[i] = rlmesh_value_discrete(0);
  return RLMESH_OK;
}

int main(void) {
  RlmeshModelVtable vt = {.struct_size = sizeof vt, .predict = predict};
  RlmeshModel* model;
  rlmesh_model_new(&vt, NULL, &model);
  RlmeshStatus rc = rlmesh_model_run_local(model, "tcp://127.0.0.1:5555", NULL, NULL);
  rlmesh_model_free(model);
  return rc == RLMESH_OK ? 0 : 1;
}
```

## Model contract

`predict` receives `num_envs` decoded observation values (one per sub-env) plus
routing metadata (session, env, request ids; per-row episode id and seed). It
writes one owned action value per row into `out_actions` and returns `RLMESH_OK`,
or returns nonzero after `rlmesh_callback_set_error(...)` to decline. The runtime
validates each action against the route's action space before it reaches the
wire: a structural mismatch fails the step, a Box-bounds overshoot is left to the
environment's own policy.

Optional hooks: `on_episode_end(env_id, episode_id)` when the runtime drops an
episode (`episode_id == NULL` means every episode of that env), and `on_close`
once at shutdown. Callbacks run on a worker thread; `user_data` must be safe to
use from a thread other than the one that created it, and a callback must not
re-enter its own model handle. `rlmesh_model_run_local` fills an optional
`RlmeshRunReport`; `rlmesh_model_cancel` stops a blocking run/serve from another
thread.

### In C++

`Model::from_predict` takes a single-env policy — `Result<Value>(const Request&)`,
where `Request` carries `observation()` (a `std::optional<ValueRef>`, absent when
the route sends none), `episode()`, and `action_space()` / `observation_space()`
as `SpaceRef`. `Model::from_predict_batch` takes the whole `Batch` and returns one
action per row. `zeros_for(SpaceRef)` builds the neutral action for _any_ space,
so a policy never has to switch on the kind to get started.

`ValueRef` reads all seven value kinds and `SpaceRef` walks a space spec (bounds,
`n`/`start`, text limits, `nvec`, dict/tuple children); `Value`'s constructors
build all seven, honoring the C constructors' all-or-nothing ownership.

Every fallible call returns `Result<T>` (`Status` is `Result<void>`): a capi call
has already recorded its message on the thread, and `Error::from_last(status)`
snapshots it. Nothing throws — `value()` / `unwrap()` abort instead, and the
header compiles under `-fno-exceptions`. `RLMESH_TRY(expr)` propagates an error
out of a `Result`-returning function.

`run_local` returns a `RunReport`; `serve` takes a `ServeOptions` (owned token,
`std::chrono` timeouts). `Model::cancel()` is the one member callable while
`run_local` / `serve` blocks — including from another thread. Never destroy a
`Model` from inside its own callback.

## Tasks

- `mise run test:cxx` — compile the headers as C11 and C++17 under `-Werror`,
  build both examples, run each end-to-end against a live env.
- `mise run test:cxx:pkg` — package (`release:cxx:package`) and build the C++
  example against the installed package via pkg-config and CMake `find_package`.
- `mise run fmt:cxx` — clang-format the headers and examples.
