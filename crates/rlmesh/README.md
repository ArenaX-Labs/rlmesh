# rlmesh

RLMesh connects models to environments for evaluation. Serve a Gymnasium-style environment in one process and drive it with a model from another, on the same machine or across the network. This crate is the Rust API: write environments and policies as ordinary Rust types, serve them over gRPC, and run evaluations against anything else that speaks RLMesh.

RLMesh aims to feel familiar in whichever language you use it from. Python, Rust, C and C++ share the same nouns and constructs (`Env`, `Space`, `Value`, `Model`, `run`, `Binding`, `EnvServer`, `RemoteEnv`), each written in that language's own conventions and with helpers that only make sense there. What you learn here carries over, and every language talks to every other over the same wire.

## Install

```toml
[dependencies]
rlmesh = "0.1"
```

The build generates gRPC stubs, so it needs `protoc` on `PATH` (`apt install protobuf-compiler` or `brew install protobuf`). The minimum supported Rust version is the crate's `rust-version`.

## Quickstart

An environment implements `Env` in Gymnasium shape, and an `EnvServer` hosts it:

```rust,no_run
use rlmesh::prelude::*;
use rlmesh::spaces::{Discrete, DiscreteOptions};
use rlmesh::{EnvServer, EnvServerOptions};

/// Three steps per episode; the observation is the step count.
struct Counter {
    observation_space: Space,
    action_space: Space,
    steps: i64,
}

impl Counter {
    fn new() -> rlmesh::Result<Self> {
        Ok(Counter {
            observation_space: Discrete::new(4, DiscreteOptions::default())?.into(),
            action_space: Discrete::new(2, DiscreteOptions::default())?.into(),
            steps: 0,
        })
    }
}

impl Env for Counter {
    fn observation_space(&self) -> Space {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        self.steps = 0;
        Ok((Value::Int(0), InfoDict::new()))
    }

    fn step(
        &mut self,
        _action: Option<Value>,
    ) -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)> {
        self.steps += 1;
        let terminated = self.steps >= 3;
        Ok((Value::Int(self.steps), 1.0, terminated, false, InfoDict::new()))
    }
}

fn main() -> rlmesh::Result<()> {
    let server = EnvServer::new(
        Counter::new()?,
        EnvServerOptions {
            address: Some("127.0.0.1:5555".into()),
            ..Default::default()
        },
    )?;
    println!("serving on {}", server.address());
    server.serve() // blocks; Ctrl-C drains, closes the env and returns Err(Error::Cancelled)
}
```

In another process, wrap a policy in a `Model` and run it against the address:

```rust,no_run
use rlmesh::prelude::*;
use rlmesh::{Model, ModelOptions, RunOptions};

fn main() -> rlmesh::Result<()> {
    // A closure is the smallest policy: observation in, action out.
    let model = Model::new(|_observation: Value| Ok(Value::Int(0)), ModelOptions::default())?;
    let result = model.run(
        "127.0.0.1:5555",
        RunOptions {
            episodes: Some(3),
            ..Default::default()
        },
    )?;
    println!(
        "{} episodes, {} steps, mean reward {:.1}",
        result.num_episodes(),
        result.total_steps(),
        result.mean_reward()
    );
    model.close()
}
```

The repository ships the same pair as runnable examples:

```bash
cargo run -p rlmesh --example serve_env    # serve a counter env on 127.0.0.1:5555
cargo run -p rlmesh --example run_model    # run a model against it
```

The server does not care what language its client is written in: the Python, C and C++ clients can drive this env, and a Rust model can drive an env served from any of them.

## Core concepts

### Environments

`Env` is one environment: `observation_space`, `action_space`, `reset(seed, options)` returning `(observation, info)` and `step(action)` returning `(observation, reward, terminated, truncated, info)`. `render`, `close`, `render_mode`, `metadata` and `id` have defaults you override when you need them. `VectorEnv` is a natively batched environment that steps `num_envs` sub-environments together.

`EnvServer::new` serves one env, `EnvServer::lanes` several independent copies and `EnvServer::vector` a `VectorEnv`. The server binds at construction, so `address()` is known before it serves. `serve()` blocks on the calling thread; `start()` serves in the background, after which `wait(timeout)` and `shutdown()` control it. Authoring is synchronous: each env lane runs on its own thread for the life of the server.

`EnvFactory` builds envs on demand, with optional discriminant parameters (`TagParam`) and a catalog of named `Variant`s. Implement `make` and serve it through `EnvFactoryExt`:

```rust,ignore
#[derive(Default)]
struct CounterFactory;

impl EnvFactory for CounterFactory {
    fn make(&mut self, _kwargs: &Kwargs) -> rlmesh::Result<MadeEnv> {
        Ok(MadeEnv::env(Counter::new()?))
    }
}

CounterFactory.serve("127.0.0.1:5555", EnvFactoryServeOptions::default())?;
```

### Spaces and values

[`rlmesh::spaces`](https://docs.rs/rlmesh/latest/rlmesh/spaces/index.html) has `Box`, `Discrete` and `Dict` as typed views of a shared `Space` handle (a peer's Experimental `MultiBinary`, `MultiDiscrete`, `Text` and `Tuple` spaces work through the same `Space` handle). A space seeds, samples and checks membership; a `SpaceSpec` is its immutable description. The module is kept out of the prelude because `spaces::Box` would shadow `std`'s `Box`.

Observations, actions and info entries are `Value`s: an enum of `None`, `Bool`, `Int`, `Float`, `Text`, `Bytes`, `Tensor`, `List`, `Tuple` and `Dict`. A Discrete action is an `Int`, a Box observation a `Tensor`, a Dict observation a `Dict`. `Metadata` (aliased as `InfoDict` and `Kwargs`) is an insertion-ordered string-keyed map. Reads are strict: `as_int` refuses a `Bool` and `as_float` an `Int`.

```rust
use rlmesh::prelude::*;
use rlmesh::spaces::{Box, BoxOptions, Dict, Discrete, DiscreteOptions, SpaceSpec};
use rlmesh::{DType, Tensor};

fn main() -> rlmesh::Result<()> {
    let pos = Box::new(-1.0, 1.0, [3], BoxOptions::default())?;
    let gripper = Discrete::new(2, DiscreteOptions::default())?;
    let space = Dict::new([
        ("pos", SpaceSpec::from(&pos)),
        ("gripper", SpaceSpec::from(&gripper)),
    ])?;
    space.seed(Some(7));
    let sample = space.sample()?; // a Value::Dict
    assert!(space.contains(&sample));

    let pixels = Tensor::new(&[0u8; 12], &[2, 2, 3], DType::Uint8)?;
    let observation: Value = [("camera", Value::from(pixels)), ("step", Value::from(3))]
        .into_iter()
        .collect::<Metadata>()
        .into();
    let step = observation.as_dict()?.get("step").map(Value::as_int).transpose()?;
    assert_eq!(step, Some(3));
    Ok(())
}
```

Tensors exchange data with other libraries through DLPack: `Tensor::to_dlpack` exports a versioned DLPack tensor and the `unsafe` `Tensor::from_dlpack` imports one.

### Models and policies

A `Model` wraps a `Policy`. Any `FnMut(Value) -> rlmesh::Result<Value>` closure is a policy; for more, implement the trait. A policy writes one or more of four prediction corners (`predict`, `predict_batch`, `predict_chunk`, `predict_chunk_batch`) and declares them in the `CORNERS` const (`Corners::PREDICT` by default). A corner that needs the per-episode `PredictContext`, with its episode id, seeds and typed scratch state, is also listed in `CONTEXT` and written as its `_with_context` twin:

```rust,no_run
use rlmesh::prelude::*;
use rlmesh::{Model, ModelOptions, ModelServeOptions};

/// Alternates actions, counting predicts per episode.
#[derive(Default)]
struct Alternate;

impl Policy for Alternate {
    const CONTEXT: Corners = Corners::PREDICT;

    fn predict_with_context(
        &mut self,
        _observation: Value,
        context: &mut PredictContext<'_>,
    ) -> rlmesh::Result<Value> {
        let count = context.state.get_or_insert_with("predicts", || 0i64)?;
        *count += 1;
        Ok(Value::Int(*count % 2))
    }

    fn on_episode_end(&mut self, episode_id: &str) -> rlmesh::Result<()> {
        println!("episode {episode_id} ended");
        Ok(())
    }
}

fn main() -> rlmesh::Result<()> {
    let model = Model::new(Alternate, ModelOptions::default())?;
    // Serve it as a model endpoint that any RLMesh run can use.
    model.serve("127.0.0.1:50051", ModelServeOptions::default())
}
```

`Policy` also covers `load` (with `LOAD_KEYS` and `Model::from_config` to build from a config map), `close`, a `spec()` for adapters and a `workflow_edition()`. `rlmesh::with_context` turns a closure that takes the context into a policy.

### Running and binding

`Model::run` and the free function `rlmesh::run` drive whole episodes and return a `RunResult` with per-episode results and telemetry. The env argument is anything that converts into an `EnvTarget`: an address, a running `&EnvServer`, a `&mut RemoteEnv`, or a local `&mut` env (served on a private loopback for the run; use `EnvTarget::vector` and `EnvTarget::factory` for vector envs and factories). `rlmesh::run` also accepts `RANDOM_SAMPLE`, a baseline that samples the action space, or a `&RemoteModel`: a model served elsewhere, reached by its participant address (`"lab:7447/openpi"`). A run borrows its model and env: it never closes the model, and closes the env only when `close_env` is set.

`RunHooks` observe a run as it happens; every method has a default:

```rust,ignore
use rlmesh::{EpisodeResult, RunHooks, RunOptions};

struct PrintEpisodes;

impl RunHooks for PrintEpisodes {
    fn on_episode_end(&mut self, result: &EpisodeResult) -> rlmesh::Result<()> {
        println!("episode {}: reward {}", result.index, result.reward);
        Ok(())
    }
}

let mut hooks = PrintEpisodes;
model.run(
    "127.0.0.1:5555",
    RunOptions {
        seeds: Some(vec![0, 1, 2]),
        hooks: Some(&mut hooks),
        ..Default::default()
    },
)?;
```

For step-by-step control, bind the model to one env, as Python's `Model.bind` does. Binding connects (a local env or factory is served in this process and runs only on the thread that calls the binding); `step(None)` steps under the model's action, and closing the binding never closes the model:

```rust,ignore
use rlmesh::BindOptions;

let mut binding = model.bind("127.0.0.1:7447/counter", BindOptions::default())?;
let (_observation, _info) = binding.reset(Some(0), None)?;
loop {
    let (_next, _reward, terminated, truncated, _info) = binding.step(None)?;
    if terminated || truncated {
        break;
    }
}
binding.close()?;
```

`rlmesh::bind(model, env, options)` also binds a `&RemoteModel` or `RANDOM_SAMPLE`. A binding is sync only; it has no async twin.

### Remote clients

`RemoteEnv` and `RemoteVectorEnv` connect to a served env and step it directly, Gymnasium style. `RemoteModel` points at a model served elsewhere by its participant address; it validates its options without dialing, and a run or a binding reaches it.

```rust,ignore
use rlmesh::{RemoteEnv, RemoteEnvOptions, RemoteModel, RemoteModelOptions, RunOptions};

let mut env = RemoteEnv::new(RemoteEnvOptions::default().address("127.0.0.1:7447/counter"))?;
let (_observation, _info) = env.reset(Some(0), None)?;
let action = env.action_space().sample()?;
let (_observation, reward, terminated, truncated, _info) = env.step(action)?;
env.close()?;

let model = RemoteModel::new(RemoteModelOptions::default().address("127.0.0.1:7447/policy"))?;
let result = rlmesh::run(&model, "127.0.0.1:7447/counter", RunOptions::default())?;
```

Addresses are session participant addresses (`host:port/name`, `host:port/session/name`, or `:0/name` for a session of its own on an OS-assigned local port). The v1 forms (`host:port`, `tcp://host:port`, `unix:///path/to.sock`, and `host`, `port`, `path` and `transport` set separately) still connect a remote env until v1 is removed; a run and a binding refuse them.

### Adapters

When a model and an env describe their observations and actions differently (camera names, state layouts, action components), [`rlmesh::adapters`](https://docs.rs/rlmesh/latest/rlmesh/adapters/index.html) bridges them declaratively. An env publishes `EnvTags` (the `tags` field of `EnvServerOptions`, or `adapters::tag`), a model declares a `ModelSpec` (`ModelOptions::spec` or `Policy::spec`), and RLMesh resolves an adapter for each route. `NO_ADAPTER` opts a model out.

### View and Recorder

`View` configures the live viewer (`View::parse("term")`, `View::parse("http:8008")`). `Recorder` collects run results, or captures live runs through the hooks returned by `Recorder::capture`, and exports them with media as a folder or ZIP bundle.

`describe` and `describe_json` report a model's or factory's declarations, and `build_info()` reports the library's version, protocol generation and workflow editions.

## Rust conventions

**Options structs with `Default`.** Optional parameters live in options structs (`EnvServerOptions`, `RunOptions`, `BindOptions`, `RemoteEnvOptions`, ...) whose `Default` holds the defaults. Fill in the fields you need and finish with `..Default::default()`. New fields can be added in a minor release, so a full struct literal without `..` is not supported. The remote options also have builder methods (`RemoteEnvOptions::default().address(..).request_timeout(..)`), and durations are `std::time::Duration`.

**Blocking calls with `_async` twins.** Every call that does I/O blocks by default and has an async twin: `run_async`, `Model::run_async`, `RemoteEnv::new_async`, `EnvServer::serve_async` and so on. Use the twin from async code: a blocking call made inside an async task fails with the `RuntimeError` class and names the twin to call. Dropping an async run cancels it.

**`rlmesh::prelude`** imports the authoring traits and the types their signatures name: `Env`, `VectorEnv`, `Policy`, `EnvFactory`, `Space`, `Value`, `InfoDict`, `Error` and friends.

**`rlmesh::ext`** holds Rust-only extras that sit beside the shared surface:

- `ext::CancelToken`: a cloneable cancellation handle. Pass it in `RunOptions::cancel` (or the serve and factory options) and call `cancel()` from anywhere; the call in flight fails with `Error::Cancelled`. `RemoteEnv::cancel_handle` returns the token a handle already uses.
- `ext::TensorExt` (zero-copy `from_vec`, `zeros`, `stack`, `unstack`, storage and stride access) and `ext::SpaceExt::contains_detail` (why a value is not in a space).
- Bearer-token endpoints: `ext::EnvServerOptions` and `ext::ModelServeOptions` (through the `ext` field of the serve options) set the token a server expects; clients send it as `api_key`.
- Raw DLPack types in `ext::dlpack`, and `ext::register_entrypoint` for custom adapter transforms.

## Errors

Fallible calls return `rlmesh::Result<T>`, whose `Error` has one variant per failure kind: `Environment` and `Model` (each carrying a code, a message and a recoverable flag), `Protocol`, `InvalidArgument`, `Type`, `Runtime`, `Os`, `Connection`, `Timeout`, `Cancelled` and `Internal`. `Error::class()` returns an `ErrorClass` whose `is_subclass_of` groups related kinds (a `ConnectionError` and a `TimeoutError` are both an `OSError`), and `Display` is the message every RLMesh language reports for the same failure.

```rust,ignore
use rlmesh::{Error, ErrorClass, RunOptions};

match model.run("127.0.0.1:5555", RunOptions::default()) {
    Ok(result) => println!("mean reward {}", result.mean_reward()),
    Err(err) if err.class().is_subclass_of(ErrorClass::OSError) => {
        eprintln!("could not reach the env: {err}");
    }
    Err(Error::Environment(failure)) => {
        eprintln!("env failed ({}): {}", failure.code.as_str(), failure.message);
    }
    Err(err) => return Err(err),
}
```

Inside an env, return `Error::environment_recoverable("...")` to fail one request while the client keeps its session; any other error ends the session. A policy reports failures with `Error::model` and `Error::model_recoverable`.

## Workflow editions

Every served env and model runs at a workflow edition, which pins what a session means across upgrades. Declare one (`Policy::workflow_edition`, `ServeOptions::workflow_edition`, the `workflow_edition` field of the run and session options, or the `RLMESH_WORKFLOW_EDITION` environment variable); a participant that declares none floats to the newest edition and warns once. See [workflow editions](https://rlmesh.dev/docs/).

## Stability

This crate is **Preview**: it is intended to become stable, but it may still change in a `0.x` minor release, and every breaking change comes with a migration note in the [changelog](https://github.com/ArenaX-Labs/rlmesh/blob/main/CHANGELOG.md). `rlmesh::ext` is part of the crate's surface. `rlmesh::binding` is hidden plumbing for RLMesh's own bindings and carries no compatibility promise; do not depend on it. The other `rlmesh-*` crates on crates.io are internal implementation details. See the [compatibility policy](https://rlmesh.dev/docs/releases/compatibility/).

## Links

- Documentation: https://rlmesh.dev/docs/
- API reference: https://docs.rs/rlmesh
- Repository: https://github.com/ArenaX-Labs/rlmesh
- C and C++: [`rlmesh.h` and `rlmesh.hpp`](https://github.com/ArenaX-Labs/rlmesh/tree/main/crates/rlmesh-capi)
- Python package: https://pypi.org/project/rlmesh/

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
