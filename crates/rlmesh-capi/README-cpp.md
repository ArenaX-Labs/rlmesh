# RLMesh C++ API

RLMesh connects models to environments for evaluation. Serve a Gymnasium-style environment in one process and drive it with a model from another, on the same machine or across the network. `rlmesh.hpp` is the C++ API: a header-only library over the RLMesh C ABI that lets you write environments and policies as ordinary C++ classes or lambdas, serve them over gRPC, and run evaluations against anything else that speaks RLMesh.

RLMesh aims to feel familiar in whichever language you use it from. Python, Rust, C and C++ share the same nouns and constructs (`Env`, `Space`, `Value`, `Model`, `run`, `Session`, `EnvServer`, `RemoteEnv`), each written in that language's own conventions and with helpers that only make sense there. What you learn here carries over, and every language talks to every other over the same wire.

The header requires C++17 and exceptions (it stops with `#error` otherwise), and is tested at C++17 and C++20. Every RLMesh entity has one token-identical definition in C++17 and C++20, so translation units built at either standard can share wrapper types (ODR). Standard or feature macros never select RLMesh definitions or members. Apart from the top-level version/exceptions guards, the sole exception is `std::ranges::enable_borrowed_range` and `std::ranges::enable_view` specializations for `rlmesh::Span<T>`, guarded by `#if defined(__cpp_lib_ranges)`; these add no RLMesh members. RTTI is not needed.

## Install

`rlmesh.hpp` ships next to `rlmesh.h` in the same package and links the same library, so installation is the same as for C. The short version, with CMake and a prebuilt release (Linux x86_64, glibc 2.34 or newer):

```cmake
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(rlmesh
  URL https://github.com/ArenaX-Labs/rlmesh/releases/download/v<version>/rlmesh-<version>-x86_64-unknown-linux-gnu.tar.gz
  URL_HASH SHA256=<the release's .sha256>
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(rlmesh)

target_link_libraries(my_sim PRIVATE rlmesh::rlmesh)
```

`find_package(rlmesh 5 CONFIG REQUIRED)` against an unpacked package and pkg-config (`c++ -std=c++17 sim.cpp $(pkg-config --cflags --libs rlmesh)`) work too. See [Install in the C guide](README.md#install) for every option, including building from source.

## Quickstart

Serve an environment. Any class with Gymnasium-shaped `observation_space`, `action_space`, `reset` and `step` members is an env:

```cpp
// serve.cpp: a three-step counter env on 127.0.0.1:5555
#include <cstdio>
#include <rlmesh.hpp>

// Three steps per episode; the observation is the step count.
class Counter {
  int64_t steps_ = 0;

 public:
  rlmesh::spaces::Space observation_space() const { return rlmesh::spaces::Discrete(4); }
  rlmesh::spaces::Space action_space() const { return rlmesh::spaces::Discrete(2); }
  std::optional<std::string> id() const { return "Counter-v0"; }

  rlmesh::ResetResult reset(std::optional<int64_t> /*seed*/,
                            const std::optional<rlmesh::InfoDict>& /*options*/) {
    steps_ = 0;
    return {rlmesh::Value(0), {}};
  }

  rlmesh::StepResult step(std::optional<rlmesh::Value> /*action*/) {
    ++steps_;
    return {rlmesh::Value(steps_), 1.0, steps_ >= 3, false, {}};
  }
};

int main() try {
  auto options = rlmesh::EnvServerOptions{}.with_address("127.0.0.1:5555");
  rlmesh::EnvServer server(Counter{}, options);
  std::printf("serving on %s\n", server.address().c_str());
  server.serve();  // blocks until shutdown; Ctrl-C throws rlmesh::ext::CancelledError
  return 0;
} catch (const rlmesh::ext::CancelledError&) {
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "error: %s\n", error.what());
  return 1;
}
```

Run a model against it from another process. A lambda from observation to action is the smallest policy:

```cpp
// run.cpp: run a constant policy for three episodes
#include <cstdio>
#include <rlmesh.hpp>

int main() try {
  rlmesh::Model model([](const rlmesh::Value& /*observation*/) { return rlmesh::Value(0); });
  auto options = rlmesh::RunOptions{}.with_episodes(3);
  rlmesh::RunResult result = model.run("127.0.0.1:5555", options);
  std::printf("%zu episodes, %lld steps, mean reward %.1f\n", result.num_episodes(),
              static_cast<long long>(result.total_steps()), result.mean_reward());
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "error: %s\n", error.what());
  return 1;
}
```

```bash
c++ -std=c++17 serve.cpp $(pkg-config --cflags --libs rlmesh) -o serve && ./serve
c++ -std=c++17 run.cpp $(pkg-config --cflags --libs rlmesh) -o run && ./run
# 3 episodes, 9 steps, mean reward 3.0
```

Neither side cares what language the other is written in: the same model can drive an env served from Python, Rust or C, and the Python, Rust and C clients can drive this env.

## C++ conventions

**Errors are exceptions.** Every failing call throws; there are no status codes to check. See [Errors](#errors).

**Handles manage themselves.** Destructors release library objects without throwing. Values, tensors, spaces, `Model`, `RemoteModel` and adapters are cheap to copy: copies share the same underlying object (`Tensor::copy()` makes an independent copy of the data). Envs, remote envs, sessions, servers and recorders are move-only. Call `close()` explicitly when you want to observe cleanup errors; destruction swallows them.

**Options are plain structs.** Every operation that takes options has an options struct (`RunOptions`, `SessionOptions`, `EnvServerOptions`, `RemoteEnvOptions`, `ModelOptions`, ...) whose members default to the documented defaults. The portable spelling is `RunOptions{}.with_episodes(2).with_seeds({0, 1})`: each `with_<field>` setter returns the same object and works on both named options and temporaries. You can also assign members directly, or pass nothing for all defaults. With C++20 you can also write the struct inline with designated initializers, as in `model.run(env, {.episodes = 3})`.

**Standard vocabulary types.** Optional values are `std::optional`; `rlmesh::OptionalList<T>` keeps "not set" distinct from an empty list (as in `RunOptions::seeds`). Durations are `std::chrono` durations (`rlmesh::Duration` is `std::chrono::duration<double>`), so a time limit is `options.max_episode = std::chrono::seconds(30)`. Text arguments take `std::string_view`, and tensor bytes come back as `rlmesh::Span<const std::byte>`.

**Borrowed contiguous views.** `rlmesh::Span<T>` provides `data`, `size`, `begin`, `end`, `operator[]` and `empty`. It accepts a pointer and count, or any contiguous range with `std::data` and `std::size` whose element pointer converts to `T*` (including vectors, `std::array`, C arrays and user-included C++20 `std::span`). The same constructor template serves both standards; the header does not include `<span>`. In C++20 it is a borrowed ranges view, converts to `std::span` through that type's range constructor, and supports policies taking `std::span<const Value>` and `std::span<const PredictContext>`. Named contiguous ranges of environments also construct `ServedEnv`; temporaries are rejected to protect the borrowed handles. Batch policies take `rlmesh::Span<const Value>`, and `BatchPredictContext` is a `Span<const PredictContext>`. Keep the backing storage alive while a view is used.

**Unreal and other hosts.** The header saves and restores common Unreal, X11 and Windows macros, including `check`, `None`, `Bool`, `min` and `max`, and compiles without RTTI. When a host defines an object-like macro such as `None` or `Bool`, undefine it in your code before spelling the matching scoped enumerator (`ValueKind::None`, `DType::Bool`). In Unreal, enable exceptions for the consuming module with `bEnableExceptions = true` in its `.Build.cs`.

**Results unpack with structured bindings.** `reset` returns `rlmesh::ResetResult` (`std::tuple<Value, InfoDict>`) and `step` returns `rlmesh::StepResult` (`std::tuple<Value, double, bool, bool, InfoDict>`), so `auto [observation, reward, terminated, truncated, info] = env.step(action);` reads like Gymnasium.

**Duck-typed authoring.** You do not inherit from a base class to write an env or a policy. The header detects the members a class or lambda has: `predict`, `predict_batch`, `predict_chunk` and `predict_chunk_batch` for a policy, with or without a context parameter; `reset`, `step`, optional `render`, `close`, `id`, `render_mode` and `metadata` for an env. A class without the required members does not compile.

**`rlmesh::ext`** holds native-only extras: `ext::CancelToken` for cancellation, `ext::CancelledError`, bearer-token options (`ext::EnvServerOptions`, `ext::ModelServeOptions`), DLPack ownership (`ext::DlpackPtr`), `ext::env_server_describe_json` and `ext::register_entrypoint`.

## Core concepts

### Spaces and values

`rlmesh::spaces` has `Box`, `Discrete` and `Dict` (plus the Experimental `MultiBinary`, `MultiDiscrete`, `Text` and `Tuple`), all usable as a `Space`. A space seeds, samples and checks membership; `SpaceSpec` is its immutable description, and a space converts to its spec where one is expected.

Observations, actions and info entries are `rlmesh::Value`s: one of None, Bool, Int, Float, Text, Bytes, Tensor, List, Tuple and Dict. A Discrete action is an Int, a Box observation a `Tensor`, a Dict observation a Dict. Values construct implicitly from `bool`, integers, floating point, strings, tensors and `rlmesh::Metadata`, the insertion-ordered map that `InfoDict` and `Kwargs` alias. Reads are strict: `as_float()` on an Int throws `rlmesh::TypeError`.

```cpp
using namespace rlmesh::spaces;

Dict space({{"pos", Box(-1.0, 1.0, {3})}, {"gripper", Discrete(2)}});
space.seed(7);
rlmesh::Value sample = space.sample();  // a Dict value
bool inside = space.contains(sample);

std::vector<uint8_t> pixels(2 * 2 * 3);
rlmesh::Value observation = rlmesh::Metadata{
    {"camera", rlmesh::Tensor(pixels.data(), pixels.size(), {2, 2, 3}, rlmesh::DType::Uint8)},
    {"step", 3}};
int64_t step = observation.as_dict().at("step").as_int();
```

`rlmesh::Tensor` exchanges data with other libraries through DLPack: `to_dlpack` exports a versioned DLPack tensor as an `ext::DlpackPtr`, and `from_dlpack` imports one.

### Environments and servers

An env class needs `observation_space()`, `action_space()`, `reset(seed, options)` and `step(action)`, as in the [quickstart](#quickstart). `render()`, `close()`, `id()`, `render_mode()` and `metadata()` are optional. A natively batched env provides `num_envs()`, `single_observation_space()`, `single_action_space()` and a `step` returning `rlmesh::VectorStepResult`, and is served as a vector env automatically.

`rlmesh::EnvServer` takes the env by value. To keep reaching the simulation from your own code, wrap a `std::shared_ptr` to it in `rlmesh::Env` and serve that; to serve several independent copies, pass a `std::vector<rlmesh::Env>`. The server binds at construction, so `address()` is known before it serves. `serve()` blocks on the calling thread; `start()` serves in the background, after which `wait(timeout)` and `shutdown()` control it. Destroying a server shuts it down and drains it.

Inside `step`, throw `rlmesh::RecoverableEnvironmentException("...")` to fail one request while the client keeps its session; any other exception ends the session.

To build envs on demand, derive from `rlmesh::EnvFactory<YourFactory>` and write `rlmesh::MadeEnv make(const rlmesh::Kwargs&)`. Optional static members declare discriminant parameters (`tag_params`), a catalog of named variants (`enumerate_variants`) and reset options. `factory.serve(address)` serves it.

### Models and policies

`rlmesh::Model` wraps a policy: a lambda or any class with a prediction member. A policy writes one or more of four corners: `predict` (one observation to one action), `predict_batch`, `predict_chunk` (one observation to a chunk of actions) and `predict_chunk_batch`. A corner that also takes a `const rlmesh::PredictContext&` receives the episode id, its seed, the predict index, a derived per-predict seed and `state`, a typed per-episode scratch store:

```cpp
// Alternates actions, counting predicts per episode.
struct Alternate {
  rlmesh::Value predict(const rlmesh::Value& /*observation*/,
                        const rlmesh::PredictContext& context) {
    int64_t& count = context.state.get_or_emplace<int64_t>("predicts", 0);
    ++count;
    return rlmesh::Value(count % 2);
  }

  void on_episode_end(std::string_view episode_id) {
    std::printf("episode %.*s ended\n", static_cast<int>(episode_id.size()), episode_id.data());
  }
};

rlmesh::Model model{Alternate{}};
model.serve("127.0.0.1:50051");  // a model endpoint any RLMesh run can use (blocks)
```

Optional members cover `load(const rlmesh::Kwargs&)` (with `rlmesh::Model::from_config<Policy>(config)` to build from a config map), `on_episode_end`, `close`, and static declarations for an adapter `spec`, `load_keys`, `native_chunk` and `workflow_edition`. `rlmesh::ModelOptions` adds hooks and a spec for a lambda.

### Running and sessions

`model.run(env, options)` and the free function `rlmesh::run(model, env, options)` drive whole episodes and return an `rlmesh::RunResult`: `episodes` (each an `EpisodeResult`), `telemetry`, and `num_episodes()`, `total_steps()`, `mean_reward()` and `success_rate()`. The env argument is an `rlmesh::EnvTarget`, which converts from an address string, a running `EnvServer`, a `RemoteEnv` or `RemoteVectorEnv`, or a local `rlmesh::Env`, `rlmesh::VectorEnv` or factory (served on a private loopback for the run). `rlmesh::run` also takes a `RemoteModel` or `rlmesh::RANDOM_SAMPLE`, a baseline that samples the action space. A run borrows its model and env: it never closes the model, and closes the env only when `close_env` is set.

Derive from `rlmesh::RunHooks` to observe a run as it happens; every member has an empty default:

```cpp
struct PrintEpisodes : rlmesh::RunHooks {
  void on_episode_end(const rlmesh::EpisodeResult& result) override {
    std::printf("episode %lld: reward %.1f\n", static_cast<long long>(result.index), result.reward);
  }
};

PrintEpisodes hooks;
rlmesh::RunOptions options;
options.seeds = {0, 1, 2};
options.hooks = &hooks;
model.run("127.0.0.1:5555", options);
```

For step-by-step control, open a `Session`. Nothing connects until first use, and closing a session never closes the model:

```cpp
rlmesh::Session session = model.session("127.0.0.1:5555");
auto [observation, info] = session.reset(0);
bool done = false;
while (!done) {
  auto [next, reward, terminated, truncated, step_info] =
      session.step(session.predict(observation));
  observation = next;
  done = terminated || truncated;
}
session.close();
```

`session.run(options)` runs whole episodes inside an open session. Keep the model and env alive while a session uses them.

### Remote clients

`rlmesh::RemoteEnv` and `RemoteVectorEnv` connect at construction and step a served env directly, Gymnasium style. `rlmesh::RemoteModel` points at a served model; it validates its options without connecting, and runs and sessions connect as they need to. Addresses are `host:port`, `tcp://host:port` or `unix:///path/to.sock`, or set `host`, `port`, `path` and `transport` separately.

```cpp
rlmesh::RemoteEnvOptions connection;
connection.address = "127.0.0.1:5555";
connection.request_timeout = std::chrono::seconds(5);
rlmesh::RemoteEnv env(connection);

auto [observation, info] = env.reset(0);
auto [next, reward, terminated, truncated, step_info] = env.step(env.action_space().sample());
env.close();

rlmesh::RunResult baseline = rlmesh::run(rlmesh::RANDOM_SAMPLE, "127.0.0.1:5555");
```

### Adapters

When a model and an env describe their observations and actions differently (camera names, state layouts, action components), `rlmesh::adapters` bridges them declaratively. An env publishes `adapters::EnvTags` (the `tags` member of `EnvServerOptions`, or `adapters::tag`), a model declares an `adapters::ModelSpec` (a static `spec` or `ModelOptions::spec`), and RLMesh resolves an adapter for each route. `rlmesh::NO_ADAPTER` opts a model out. [`examples/cpp_env.cpp`](examples/cpp_env.cpp) tags a served env.

### View, Recorder and describe

`rlmesh::View` shows live frames in the terminal or a local web page when set on `SessionOptions::view`. `rlmesh::Recorder` collects run results or captures live runs through the hooks returned by `capture`, and exports them with media as a folder or ZIP bundle. `rlmesh::describe` and `describe_json` report a model's or factory's declarations, and `rlmesh::build_info()` the library's version, protocol generation and workflow editions.

## Errors

Failures throw a class from one hierarchy, so you can catch as broadly or narrowly as you like:

| Class                                                                                           | Base                    | Thrown for                                         |
| ----------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------- |
| `rlmesh::ValueError` (and `AdapterResolutionError`)                                             | `std::invalid_argument` | an invalid argument or option                      |
| `rlmesh::TypeError`                                                                             | `std::logic_error`      | the wrong kind of value                            |
| `rlmesh::RuntimeError`                                                                          | `std::runtime_error`    | misuse, such as overlapping calls on one handle    |
| `rlmesh::OSError`, `ConnectionError`, `TimeoutError`                                            | `std::runtime_error`    | filesystem, transport and deadline failures        |
| `rlmesh::RLMeshException` and its `ProtocolException`, `EnvironmentException`, `ModelException` | `rlmesh::RuntimeError`  | handshake failures and failures reported by a peer |
| `rlmesh::ext::CancelledError`                                                                   | none                    | cancellation (Ctrl-C or an `ext::CancelToken`)     |

`EnvironmentException` (with `RecoverableEnvironmentException`) and `ModelException` carry `code()` and `is_recoverable()`. `what()` is the message every RLMesh language reports for the same failure, and an exception your own env, policy or hook throws comes back to you unchanged when the failure stays in process. `ext::CancelledError` deliberately derives from no standard exception, so `catch (const std::exception&)` does not swallow a cancellation.

```cpp
try {
  model.run("127.0.0.1:5555");
} catch (const rlmesh::OSError& error) {  // ConnectionError, TimeoutError
  std::fprintf(stderr, "could not reach the env: %s\n", error.what());
} catch (const rlmesh::EnvironmentException& error) {
  std::fprintf(stderr, "env failed (%s): %s\n", std::string(rlmesh::to_string(error.code())).c_str(),
               error.what());
}
```

## Workflow editions

Every served env and model runs at a workflow edition, which pins what a session means across upgrades. Declare one (a static `workflow_edition` on a policy or factory, `ServeOptions::workflow_edition`, the `workflow_edition` member of the run and session options, or the `RLMESH_WORKFLOW_EDITION` environment variable); a participant that declares none floats to the newest edition and warns once. See the [documentation](https://rlmesh.dev/docs/).

## Compatibility

The C++ API is **Experimental**. It is header-only and compiled into your program, and it changes together with the C ABI it wraps (generation 5, `RLMESH_ABI_VERSION`); a binary built against another generation will not load. See [Compatibility in the C guide](README.md#compatibility) and the [compatibility policy](https://rlmesh.dev/docs/releases/compatibility/). C++17 and C++20 translation units may share wrapper types in one binary. A served env reports its host language and compiler in its describe envelope; it omits the optional `language_version` because the binary may mix standards.

## Examples

[`examples/`](examples) holds runnable programs, built and run by `mise run test:cxx`:

- [`cpp_model.cpp`](examples/cpp_model.cpp): a context-reading model that samples the env's action space and runs seeded episodes.
- [`cpp_env.cpp`](examples/cpp_env.cpp): a reaching env with a Dict observation, a camera tensor and adapter tags.
- [`cpp_parity.cpp`](examples/cpp_parity.cpp): factories, policies with every corner, sessions, hooks and results.
- `cpp_spaces.cpp`, `cpp_errors.cpp`, `cpp_runtime.cpp`, `cpp_payload.cpp` and more cover one area each.

## Links

- Documentation: https://rlmesh.dev/docs/
- C API: [README.md](README.md)
- Rust crate: https://crates.io/crates/rlmesh
- Python package: https://pypi.org/project/rlmesh/
