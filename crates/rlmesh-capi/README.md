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

## Layout

- `include/rlmesh.h` — the C ABI (C11). Hand-authored; the single header authority.
- `include/rlmesh.hpp` — RAII C++ wrapper over the C ABI (no exceptions, `Result<T>`).
- `examples/c_model.c`, `examples/cpp_model.cpp` — a zero-action model in each language.
- `examples/cpp_env.cpp` — a tagged C++ env (a point reaching a target, with a camera).
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
routing metadata (session, env, request ids; one `RlmeshEpisode` per row). It
writes one owned action value per row into `out_actions` and returns `RLMESH_OK`,
or returns nonzero after `rlmesh_callback_set_error(...)` to decline. The runtime
validates each action against the route's action space before it reaches the
wire: a structural mismatch fails the step, a Box-bounds overshoot is left to the
environment's own policy.

Each `RlmeshEpisode` row carries the episode's `id` and reset `seed` (when
`seeded`), plus the same per-predict context the Python SDK stamps on a predict:
`predict_index`, the re-plan ordinal within the episode (0 on the first predict
under that id, then +1 per predict), and `predict_seed`, a reproducible mix of
the episode seed and that ordinal (`rlmesh.predict_seed` in the SDK; meaningful
only when `seeded`). The capi counts these per episode id and drops the counter
when the episode ends, holding at most 4096 live episodes — past that the least
recently predicted one is evicted through `on_episode_end`.

Optional hooks: `on_episode_end(env_id, episode_id)` when the runtime drops an
episode (`episode_id == NULL` means every episode of that env), and `on_close`
once at shutdown. Callbacks run on a worker thread; `user_data` must be safe to
use from a thread other than the one that created it, and a callback must not
re-enter its own model handle. `rlmesh_model_run_local` fills an optional
`RlmeshRunReport`; `rlmesh_model_cancel` stops a blocking run/serve from another
thread.

`RlmeshRunOptions` bounds and seeds a run: `max_episodes`, `base_seed` (when
`seeded`) or explicit `episode_seeds`, the per-episode step/time caps,
`execution_horizon`, and `trial_index_base` (when `trial_indexed`) — the first
trial ordinal the episodes walk, delivered as `reset(options={"trial_index": k})`
to an environment that declares that reset option.

## Environment contract

`rlmesh_env_new(vtable, config, user_data, &env)` takes the callbacks and an
`RlmeshEnvConfig`: the observation and action spaces (built with the
`rlmesh_space_*` builders), an id, the adapter `EnvTags` as JSON, the reset
options the env understands (`trial_index`), a render mode, and extra metadata.
The tags are validated against the spaces right there, the same publish-time
check Python's `adapters.tag()` runs, so a typo fails `rlmesh_env_new` rather
than a model's resolve. The tag JSON grammar (leaf kinds, fields, defaults, and
validation) is specified in [`docs/specs/env_tags.v1.md`](../../docs/specs/env_tags.v1.md).

Then `rlmesh_env_bind` (learn the address, e.g. for port 0) and
`rlmesh_env_serve` (blocks). `rlmesh_env_cancel` from any thread drains the
server, runs `close` once, and lets `serve` return `RLMESH_OK`.

`rlmesh_env_describe_json` (C++: `EnvServer::describe_json()`) returns the env's
`rlmesh.describe.v1` envelope: the spaces, the tags, and the edition handshake,
the same artifact `rlmesh describe` emits for a Python env. `rlmesh_env_bind`
puts it on the handshake too, so the managed platform can read an image without
a label; bake it as the image's `dev.rlmesh.describe` label to describe the
image before it runs.

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
Sim) sets `RlmeshEnvConfig.foreground = true`: `rlmesh_env_serve` then runs the
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

`run_local` takes a `RunOptions` (the C `RlmeshRunOptions` field for field, with
`episode_seeds` as a `std::vector`) and returns a `RunReport`; `serve` takes a
`ServeOptions` (owned token, `std::chrono` timeouts). `Model::cancel()` is the one member callable while
`run_local` / `serve` blocks — including from another thread; a cancelled
`serve` returns cleanly, a cancelled `run_local` fails with
`Error::is_cancelled()`. Never destroy a `Model` from inside its own callback.

## Tasks

- `mise run test:cxx` — compile the headers as C11 and C++17 under `-Werror`,
  build the examples, run each model end-to-end against a live env, and drive
  the C++ env with a model (`e2e_env_harness`).
- `mise run test:cxx:pkg` — package (`release:cxx:package`) and build the C++
  example against the installed package via pkg-config and CMake `find_package`.
- `mise run fmt:cxx` — clang-format the headers and examples.
