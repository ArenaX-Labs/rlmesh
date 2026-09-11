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
#include <rlmesh.hpp>

int main() {
  auto model = rlmesh::Model::from_predict([](const rlmesh::Observation& obs) {
    return rlmesh::Value::discrete(0);           // one action per predict
  });
  rlmesh::RunOptions options;
  options.max_episodes = 3;
  return model.value().run_local("tcp://127.0.0.1:5555", options) ? 0 : 1;
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

## Tasks

- `mise run test:cxx` — compile the headers as C11 and C++17 under `-Werror`,
  build both examples, run each end-to-end against a live env.
- `mise run test:cxx:pkg` — package (`release:cxx:package`) and build the C++
  example against the installed package via pkg-config and CMake `find_package`.
- `mise run fmt:cxx` — clang-format the headers and examples.
