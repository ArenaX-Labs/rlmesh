# Env authoring and serving

Part of the [language parity spec](../language-parity.md). Status: Draft —
reviewed round 6. Rules, error classes and paths are in [rules](rules.md);
this file cites them by ID.

Owns (R-27): the Gymnasium env shape (`Env`, `VectorEnv`, `VectorSeed`),
`EnvFactory` (with Rust's `EnvFactoryExt` and C++'s `EnvFactoryBase`),
`EnvFactoryServeOptions`, `MadeEnv`, `TagParam`, `Variant`, `VariantOptions`,
`EnvServer`, `EnvServerOptions`, `ServeOptions`, `Transport`, `specs.EnvContract`,
`ServedEnv` (Core's `describe_json(served_env=)` argument), `trial_index` and
the serve signal rules. Links: `EnvTags`, `tag`,
`ObservationRoles`, `ENV_METADATA_KEY`, `ENV_BRANCH_METADATA_KEY`
([Adapters](adapters.md)); `Value`, `Metadata` / `InfoDict` / `Kwargs`,
`ENV_RESET_OPTIONS_KEY`, `describe`, editions ([Core](core.md)); `Space`
([Spaces](spaces.md)); envs as run and session inputs ([Running](running.md)).

## Shape

Author an env in Gymnasium shape, serve it, and serve a factory.

```python
class Cartpole:
    observation_space = Box(-1.0, 1.0, (4,))
    action_space = Discrete(2)
    def reset(self, *, seed=None, options=None): return obs, {}
    def step(self, action): return obs, 1.0, False, False, {}
    def close(self): pass

with rlmesh.EnvServer(Cartpole(), "tcp://127.0.0.1:0") as server:
    server.start()
    server.wait(10)

class Factory(rlmesh.EnvFactory):
    tags = TAGS
    reset_options = ("trial_index",)
    def make(self, **kwargs): return Cartpole()

Factory().serve("tcp://0.0.0.0:7000", num_envs=4)
```

```rust
struct Cartpole { observation_space: Space, action_space: Space }

impl Cartpole {
    fn new() -> Result<Self> {
        Ok(Self {
            observation_space: spaces::Box::new(-1.0, 1.0, [4], BoxOptions::default())?.into(),
            action_space: Discrete::new(2, DiscreteOptions::default())?.into(),
        })
    }
}

impl rlmesh::Env for Cartpole {
    fn observation_space(&self) -> Space { self.observation_space.clone() }
    fn action_space(&self) -> Space { self.action_space.clone() }
    fn reset(&mut self, seed: Option<i64>, options: Option<&InfoDict>) -> Result<(Value, InfoDict)> { .. }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> { .. }
}

let server = EnvServer::new(
    Cartpole::new()?,
    EnvServerOptions { address: Some("tcp://127.0.0.1:0".into()), ..Default::default() },
)?;
server.start()?;
server.wait(Some(Duration::from_secs(10)))?;
server.shutdown()?; // the `with` exit; Drop would run it but discard its error (R-17)

#[derive(Default)]
struct Factory;
impl rlmesh::EnvFactory for Factory {
    fn tags() -> Option<EnvTags> { Some(tags()) }
    fn reset_options() -> Vec<String> { vec!["trial_index".into()] }
    fn make(&mut self, _: &Kwargs) -> Result<MadeEnv> { Ok(MadeEnv::env(Cartpole::new()?)) }
}
Factory.serve("tcp://0.0.0.0:7000", EnvFactoryServeOptions { num_envs: 4, ..Default::default() })?;
```

```cpp
struct Cartpole {  // detected, no base class
  rlmesh::spaces::Space observation_space() const;
  rlmesh::spaces::Space action_space() const;
  std::tuple<rlmesh::Value, rlmesh::InfoDict> reset(std::optional<int64_t> seed,
                                                    const std::optional<rlmesh::InfoDict>& options);
  std::tuple<rlmesh::Value, double, bool, bool, rlmesh::InfoDict> step(std::optional<rlmesh::Value> action);
};

rlmesh::EnvServer server(Cartpole{}, {.address = "tcp://127.0.0.1:0"});
server.start();
server.wait(std::chrono::seconds(10));
server.shutdown();  // the `with` exit; ~EnvServer() would run it but swallow its error

struct Factory : rlmesh::EnvFactory<Factory> {  // CRTP: statics are class attributes; non-copyable, non-movable
  static std::optional<rlmesh::adapters::EnvTags> tags() { return make_tags(); }
  static std::vector<std::string> reset_options() { return {"trial_index"}; }
  rlmesh::MadeEnv make(const rlmesh::Kwargs&) { return rlmesh::Env(Cartpole{}); }
};
Factory{}.serve("tcp://0.0.0.0:7000", {.num_envs = 4});
```

```c
RLMeshEnvVtable vt = RLMESH_ENV_VTABLE_INIT;
vt.reset = cp_reset; /* (ud, const int64_t* seed, const RLMeshValue* options, RLMeshValue** obs, RLMeshValue** info) */
vt.step = cp_step;   /* (ud, const RLMeshValue* action, RLMeshValue** obs, double*, bool*, bool*, RLMeshValue** info) */
RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
attrs.observation_space = obs_space;
attrs.action_space = act_space;
RLMeshEnv* env = NULL;
rlmesh_env_new(&vt, state, &attrs, &env);

RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
so.address = "tcp://127.0.0.1:0";
RLMeshEnvServer* server = NULL;
rlmesh_env_server_new(&env, 1, &so, &server);
rlmesh_env_server_start(server);
double timeout = 10.0;  /* seconds */
bool stopped = false;
rlmesh_env_server_wait(server, &timeout, &stopped);
rlmesh_env_server_shutdown(server); /* the `with` exit; _free would run it but drop its status */
rlmesh_env_server_free(server); /* the server borrowed env */
rlmesh_env_free(env);

RLMeshEnvFactoryVtable fvt = RLMESH_ENV_FACTORY_VTABLE_INIT;
fvt.make = factory_make; /* (ud, const RLMeshValue* kwargs, RLMeshEnv** out_env, RLMeshVectorEnv** out_vector_env) */
RLMeshEnvFactoryAttributes fa = RLMESH_ENV_FACTORY_ATTRIBUTES_INIT;
fa.tags = tags;
RLMeshEnvFactory* factory = NULL;
rlmesh_env_factory_new(&fvt, NULL, &fa, &factory);
RLMeshEnvFactoryServeOptions fso = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
fso.num_envs = 4;
rlmesh_env_factory_serve(factory, "tcp://0.0.0.0:7000", &fso);
```

- **The author writes Gymnasium shape; the library derives the contract.** No
  author builds an `EnvContract`, request or result struct (ENV-2).
- **Python bases stay bases, duck types stay duck types.** Python envs are
  duck-typed, so C++ detects members (`rlmesh::Env` wraps any matching class)
  and Rust uses a trait. Python's `EnvFactory` is a base class with class
  attributes, so C++ uses a CRTP base whose statics are the class attributes
  and Rust a trait with associated functions (ENV-14, ENV-20).
- **Sync only.** Authoring is sync in every language; Rust's async twins are
  `EnvServer::serve_async` / `wait_async` / `shutdown_async` and
  `EnvFactoryExt::serve_async` (ENV-1, R-16). `serve_async` runs every lane on
  library threads, as `start()` does, installs no signal handlers and
  resolves when the server stops, with the serve error; only sync `serve()`
  keeps lane 0 on the caller (ENV-29; R-16's `serve_async` exception).
  `EnvFactoryExt::serve_async` borrows the factory, so its pre-serve hooks
  (branch check, `prepare`, `make_env`) run on the polling thread inside the
  first poll and block it until the server is built (ENV-29). Dropping any of
  these futures is ENV-32's cancellation: a dropped serve stops its server.

## Parity table

Signatures are targets. `Result<T>` is `rlmesh::Result<T>`; C functions
return `RLMeshStatus` unless shown. Defaults equal Python's (R-6). `—` = no
counterpart needed (cited rule).

### Env shape (`types.EnvLike`; the protocol object is an R-24 bridge)

| Python                                         | Rust                                                                                               | C                                                                                                                                                                  | C++ (detected member)                                                                                       |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| an object with `reset` / `step` / `close`      | `trait Env`; blanket impls for `Box<E>`, `Box<dyn Env + Send>`, `Arc<Mutex<E>>`                    | `RLMeshEnvVtable` + `RLMeshEnvAttributes` → `rlmesh_env_new(const RLMeshEnvVtable*, void* user_data, const RLMeshEnvAttributes*, RLMeshEnv** out)`; `rlmesh_env_close`, `rlmesh_env_free` | `rlmesh::Env` (move-only handle) built from any class with the members below, or a `std::shared_ptr` to one |
| `observation_space`                            | `fn observation_space(&self) -> Space`                                                             | `attrs.observation_space` (`const RLMeshSpace*`, required, copied; NULL is `RLMESH_ERR_TYPE`, ENV-P4)                                                              | `Space observation_space() const`                                                                           |
| `action_space`                                 | `fn action_space(&self) -> Space`                                                                  | `attrs.action_space` (required, copied; NULL is `RLMESH_ERR_TYPE`)                                                                                                 | `Space action_space() const`                                                                                |
| `reset(*, seed=None, options=None)`            | `fn reset(&mut self, seed: Option<i64>, options: Option<&InfoDict>) -> Result<(Value, InfoDict)>`  | `RLMeshStatus (*reset)(void*, const int64_t* seed, const RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info)`                             | `std::tuple<Value, InfoDict> reset(std::optional<int64_t> seed, const std::optional<InfoDict>& options)`    |
| `step(action)`                                 | `fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)>` (ENV-3) | `RLMeshStatus (*step)(void*, const RLMeshValue* action /* NULL = None */, RLMeshValue** out_observation, double* out_reward, bool* out_terminated, bool* out_truncated, RLMeshValue** out_info)` | `std::tuple<Value, double, bool, bool, InfoDict> step(std::optional<Value> action)`                         |
| `render()` (read by the server)                | `fn render(&mut self) -> Result<Option<Value>>`, default `Ok(None)` (ENV-5)                        | `RLMeshStatus (*render)(void*, RLMeshValue** out_frame)`; NULL slot or NULL frame = None                                                                          | optional `std::optional<Value> render()`                                                                    |
| `close()`                                      | `fn close(&mut self) -> Result<()>`, default no-op (ENV-25)                                        | `RLMeshStatus (*close)(void*)`, NULL = no-op                                                                                                                       | optional `void close()`                                                                                     |
| `render_mode` (read by the server)             | `fn render_mode(&self) -> Option<String>`, default `None`                                          | `attrs.render_mode` (NULL = None)                                                                                                                                  | optional `std::optional<std::string> render_mode() const`                                                   |
| `metadata` (read by the server)                | `fn metadata(&self) -> Option<Metadata>`, default `None`                                           | `attrs.metadata` (`const RLMeshValue*` DICT, NULL = None)                                                                                                          | optional `std::optional<Metadata> metadata() const`                                                         |
| `spec.id` (read by the server)                 | `fn id(&self) -> Option<String>`; None = `"UnknownEnv-v1"` (ENV-6)                                 | `attrs.id` (NULL = `"UnknownEnv-v1"`)                                                                                                                              | optional `std::optional<std::string> id() const`                                                            |
| `vt.free_user_data`                            | —                                                                                                  | `void (*free_user_data)(void*)` (R-20)                                                                                                                             | —                                                                                                           |
| raise `RecoverableEnvironmentException(msg)`   | `Err(Error::environment_recoverable(msg))`                                                         | nonzero after `rlmesh_callback_set_error(msg, true)` or `rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION, msg)` (R-12) | `throw rlmesh::RecoverableEnvironmentException(msg)`                                                        |
| raise anything else                            | any other `Err`: ends the client session, code `INTERNAL` (R-21)                                   | any other nonzero: `rlmesh_callback_set_error(msg, false)`, `rlmesh_callback_set_error_class` with any other class (code `INTERNAL`, not recoverable, R-12), or no error set | any other exception                                                                                         |
| (the text the client receives)                 | `"{op} failed: {message}"` (ENV-35)                                                                | same                                                                                                                                                               | same                                                                                                        |

### Vector env shape (`types.VectorEnvLike`)

| Python                                       | Rust                                                                                                     | C                                                                                                                                                                                       | C++ (detected member)                                                                                    |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| an object with `single_observation_space`    | `trait VectorEnv`; same blanket impls                                                                    | `RLMeshVectorEnvVtable` + `RLMeshVectorEnvAttributes` → `rlmesh_vector_env_new(vtable, user_data, &attrs, &out)`; `_close`, `_free`                                                     | `rlmesh::VectorEnv` handle from any matching class                                                       |
| `num_envs`                                   | `fn num_envs(&self) -> u64`                                                                              | `attrs.num_envs` (`uint64_t`)                                                                                                                                                            | `uint64_t num_envs() const`                                                                              |
| `single_observation_space`, `single_action_space` | `fn single_observation_space(&self) -> Space`, `fn single_action_space(&self) -> Space`             | `attrs.single_observation_space`, `attrs.single_action_space`                                                                                                                           | `Space single_observation_space() const`, `single_action_space() const`                                 |
| `reset(*, seed=None, options=None)` | `fn reset(&mut self, seed: Option<VectorSeed>, options: Option<&InfoDict>) -> Result<(Value, InfoDict)>` | `RLMeshStatus (*reset)(void*, const int64_t* seed, size_t seed_len, const RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info)`; NULL = None, `seed_len == 1` = one int | `std::tuple<Value, InfoDict> reset(const std::optional<VectorSeed>& seed, const std::optional<InfoDict>& options)` |
| `step(actions)`                              | `fn step(&mut self, actions: Value) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)>`        | `RLMeshStatus (*step)(void*, const RLMeshValue* actions, RLMeshValue** out_observation, double* out_rewards, bool* out_terminated, bool* out_truncated, RLMeshValue** out_info)`; arrays are `num_envs` long, library-allocated | `std::tuple<Value, std::vector<double>, std::vector<bool>, std::vector<bool>, InfoDict> step(const Value& actions)` |
| `render`, `close`, `render_mode`, `metadata`, `spec.id` | as `Env`                                                                                      | as `RLMeshEnvVtable` / attributes                                                                                                                                                       | as `Env`                                                                                                 |
| seed `None \| int \| list[int]` | `enum VectorSeed { One(i64), Each(Vec<i64>) }`, `From<i64>`, `From<Vec<i64>>`, `From<&[i64]>`; None is `Option::None` (ENV-4) | pointer + `seed_len` | `class VectorSeed` from `int64_t`, `std::vector<int64_t>`, `std::initializer_list<int64_t>`; None is `std::nullopt` |

Batched layout (`actions` passed in, observations returned) is the one a
default Python vector env sees, the numpy path of `ValueBackend::Auto`
(`pyrs/spaces/value_codec/batch.rs:283-311`), per space kind:

- Box, MultiBinary, MultiDiscrete: one stacked `Tensor` of shape
  `[num_envs, *space.shape]` in the space's dtype.
- Discrete: one stacked int64 `Tensor` of shape `[num_envs]` (Python's
  `np.array` of the lane ints), not a `List`.
- Dict: a `Dict` whose every child is batched by these rules; Tuple: a
  `Tuple` likewise (`batch.rs:39-70`).
- Every other kind (Text, Sequence, Graph, …): a `List` of `num_envs`
  per-lane values.

Returned observations may also give a stacked kind as a `List` of per-lane
values, as Python accepts any sequence (`batch.rs:106-205`). Python's
framework-bridged (neutral) path, which hands Discrete over as a list
(`batch.rs:218-260`), is a Python-only bridge (R-24).

### `EnvFactory`

| Python                                                       | Rust                                                                                                                      | C                                                                                                                                                       | C++ (`struct F : rlmesh::EnvFactory<F>`)                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `class EnvFactory(ABC)`                                      | `trait EnvFactory: Sized`; provided `describe` / `serve` / `make_env` on sealed `EnvFactoryExt` (blanket)                | `RLMeshEnvFactoryVtable` + `RLMeshEnvFactoryAttributes` → `rlmesh_env_factory_new(vtable, user_data, &attrs /* NULL = INIT */, &out)`; `rlmesh_env_factory_free` ([C declarations](#c-declarations)) | CRTP base `rlmesh::EnvFactory<F> : EnvFactoryBase` (ENV-20); statics hide the base defaults |
| `tags: ClassVar[EnvTags \| None] = None`                     | `fn tags() -> Option<EnvTags>` (associated, default `None`)                                                               | `attrs.tags` (`const RLMeshEnvTags*`, copied, NULL = None); `rlmesh_env_factory_tags(f, RLMeshEnvTags** out)`                                           | `static std::optional<adapters::EnvTags> tags()`                                       |
| `params: ClassVar[ParamSpec \| None]` (Experimental type)    | deferred until `params` is Stable                                                                                         | deferred                                                                                                                                                | deferred                                                                               |
| `tag_params: ClassVar[tuple[str, ...]] = ()`                 | `fn tag_params() -> Vec<TagParam>` (default empty; ENV-15)                                                                | `attrs.tag_params` (`const RLMeshTagParam* const*`), `attrs.tag_params_len`; `rlmesh_env_factory_tag_params(f, RLMeshTagParamList** out)`               | `static std::vector<TagParam> tag_params()`                                            |
| (from `params` + `make` defaults)                            | `struct TagParam { pub name: String, pub default_: Value, pub choices: Vec<Value> }` (ENV-15)                              | `RLMeshTagParam { size_t struct_size; const char* name; const RLMeshValue* default_; const RLMeshValue* choices /* LIST */; }` + `_INIT`; an input in `attrs`, a lent struct (R-19) from `rlmesh_tag_param_list_get` | `struct TagParam { std::string name; Value default_; std::vector<Value> choices; }`    |
| `reset_options: ClassVar[tuple[str, ...]] = ()`              | `fn reset_options() -> Vec<String>`                                                                                       | `attrs.reset_options` (`const char* const*`), `attrs.reset_options_len`; `rlmesh_env_factory_reset_options(f, RLMeshValue** out /* LIST */)`              | `static std::vector<std::string> reset_options()`                                      |
| `workflow_edition: ClassVar[str \| None] = None`             | `fn workflow_edition() -> Option<String>`                                                                                 | `attrs.workflow_edition` (NULL = None); `rlmesh_env_factory_workflow_edition(f, bool* out_has, RLMeshBytes* out)`                                       | `static std::optional<std::string> workflow_edition()`                                 |
| `prepare()`                                                  | `fn prepare(&mut self) -> Result<()>` (default no-op)                                                                     | `RLMeshStatus (*prepare)(void*)` (NULL = no-op); `rlmesh_env_factory_prepare(f)`                                                                        | optional `void prepare()`                                                              |
| `make(**kwargs)` (abstract; the call stamps, ENV-17)         | `fn make(&mut self, kwargs: &Kwargs) -> Result<MadeEnv>` (authored, unstamped); `EnvFactoryExt::make_env(&mut self, kwargs: &Kwargs) -> Result<MadeEnv>` (stamps) | `RLMeshStatus (*make)(void*, const RLMeshValue* kwargs, RLMeshEnv** out_env, RLMeshVectorEnv** out_vector_env)` (exactly one set, else `RLMESH_ERR_TYPE`, as Python refuses a non-env; a NULL slot is `RLMESH_ERR_TYPE` at `rlmesh_env_factory_new`, as Python refuses an abstract `make`); `rlmesh_env_factory_make(f, kwargs /* DICT; NULL = {} */, &env, &vector_env)` stamps | `MadeEnv make(const Kwargs&)` (authored); `MadeEnv make_env(const Kwargs& = {})` (stamps) |
| return value of `make`                                       | `enum MadeEnv { Env(Box<dyn Env + Send>), Vector(Box<dyn VectorEnv + Send>) }`, `MadeEnv::env(e)`, `MadeEnv::vector(v)`  | two out-params                                                                                                                                          | `using MadeEnv = std::variant<Env, VectorEnv>`                                         |
| `@classmethod tags_for(**params)`                            | `fn tags_for(params: &Kwargs) -> Result<Option<EnvTags>>` (associated; default `Ok(Self::tags())`)                        | `RLMeshStatus (*tags_for)(void*, const RLMeshValue* params, RLMeshEnvTags** out)` (NULL = `tags`); `rlmesh_env_factory_tags_for(f, params /* DICT; NULL = {} */, RLMeshEnvTags** out /* NULL = None */)` | optional `static std::optional<adapters::EnvTags> tags_for(const Kwargs&)`             |
| `@classmethod enumerate_variants()` (optional, duck-typed)   | `fn enumerate_variants() -> Result<Option<Vec<Variant>>>` (associated; `Ok(None)` = not defined)                          | `RLMeshStatus (*enumerate_variants)(void*, RLMeshVariantSink*)` + `rlmesh_variant_sink_push` ([C declarations](#c-declarations)); NULL = not defined; `rlmesh_env_factory_enumerate_variants(f, RLMeshVariantList** out /* NULL = not defined */)` | optional `static std::vector<Variant> enumerate_variants()`                            |
| `@classmethod describe() -> dict` (`@final` intent) | `EnvFactoryExt::describe() -> Result<Metadata> where Self: Default`; instance form is Core's `rlmesh::describe(&mut f)` (ENV-19) | `rlmesh_env_factory_describe(RLMeshEnvFactory*, const RLMeshDescribeOptions* /* NULL */, RLMeshValue** out)` ([Core](core.md)) | `static Metadata describe()` (non-virtual; `F` default-constructible) |
| `close()`                                                    | `fn close(&mut self) -> Result<()>` (default no-op)                                                                       | `RLMeshStatus (*close)(void*)`; `rlmesh_env_factory_close(f)`; `_free` does not close                                                                   | optional `void close()`                                                                |
| `@final serve(address, *, num_envs=1, workflow_edition=None, **make_kwargs)` | `EnvFactoryExt::serve(&mut self, address: &str, options: EnvFactoryServeOptions) -> Result<()>` + `serve_async` | `rlmesh_env_factory_serve(RLMeshEnvFactory*, const char* address, const RLMeshEnvFactoryServeOptions*)`                                          | `void serve(std::string_view address, const EnvFactoryServeOptions& = {})` (non-virtual) |
| `num_envs=1`                                                 | `EnvFactoryServeOptions::num_envs: i64` (1; `<= 1` serves one env, Python's `max(num_envs, 1)`)                           | `int64_t num_envs` (INIT 1)                                                                                                                             | `int64_t num_envs = 1`                                                                 |
| `workflow_edition=None`                                      | `workflow_edition: Option<String>`                                                                                        | `const char* workflow_edition` (NULL, `""`, blank = None)                                                                                               | `std::optional<std::string> workflow_edition`                                          |
| `**make_kwargs`                                              | `make_kwargs: Kwargs`                                                                                                     | `const RLMeshValue* make_kwargs` (DICT, NULL = `{}`)                                                                                                    | `Kwargs make_kwargs`                                                                   |
| (KeyboardInterrupt, R-18; ENV-33)                            | `cancel: Option<ext::CancelToken>`                                                                                        | `RLMeshExtCancelToken* cancel` (NULL = none, borrowed for the call)                                                                                     | `std::optional<ext::CancelToken> cancel`                                               |
| `vectorization_mode=`, `framework=`, `device=`               | — (R-24)                                                                                                                  | —                                                                                                                                                       | —                                                                                      |

C++ factories wrap C one to one (D-2, ENV-20):

```cpp
class EnvFactoryBase {                       // non-template; what EnvTarget and describe take
 public:
  EnvFactoryBase(const EnvFactoryBase&) = delete;
  EnvFactoryBase& operator=(const EnvFactoryBase&) = delete;
  EnvFactoryBase(EnvFactoryBase&&) = delete; // the handle's user_data is this object
  RLMeshEnvFactory* handle();                // built on first call; throws TypeError (branch table)
  void serve(std::string_view address, const EnvFactoryServeOptions& = {});
  MadeEnv make_env(const Kwargs& = {});      // stamps (ENV-17)
 protected:
  using Build = RLMeshEnvFactory* (*)(EnvFactoryBase&);  // rlmesh_env_factory_new with F's trampolines
  explicit EnvFactoryBase(Build build) noexcept;
  ~EnvFactoryBase();                         // rlmesh_env_factory_free; never calls close (as C)
};
template <class F> class EnvFactory : public EnvFactoryBase {
 public:
  EnvFactory() noexcept;                     // public, non-explicit: see below
  // implicit public destructor; default statics (tags() = nullopt, …); trampolines over F
};
```

The handle is built lazily, on the first `handle()` call (from `serve`,
`make_env`, `describe` or `EnvTarget` construction), because
`F` is not yet constructed inside the base constructor. It is built by
`rlmesh_env_factory_new` with `user_data = static_cast<F*>(this)`,
`free_user_data = NULL` (the C++ object owns itself), attributes copied from
`F`'s statics, and `noexcept` trampolines that call `F`'s members and
statics and capture exceptions (R-12). A factory is therefore neither
copyable nor movable: hold it in place or behind a `std::unique_ptr`.
`Factory{}.serve(…)` is valid because a prvalue is not moved, and because
`EnvFactory<F>`'s default constructor and destructor are public and the
constructor is not `explicit`: `Factory{}` is aggregate initialization, which
copy-list-initializes the `EnvFactory<Factory>` base from `{}` and potentially
invokes its destructor in the caller's context ([dcl.init.aggr]), so a
protected or `explicit` default constructor would make it ill-formed. That
constructor passes `F`'s build function to the protected `EnvFactoryBase`
constructor.

### `Variant`

| Python                                          | Rust                                                                                                                | C                                                                                                                                  | C++                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `Variant(id, params, /, **metadata)`            | `struct Variant { pub id: String, pub params: Kwargs, pub metadata: Kwargs }`; `Variant::new(id: impl Into<String>, params: Kwargs, options: VariantOptions) -> Variant` | `rlmesh_variant_new(const char* id, const RLMeshValue* params /* DICT */, const RLMeshVariantOptions* /* NULL = INIT */, RLMeshVariant** out)`; `_free` | `struct Variant { std::string id; Kwargs params; Kwargs metadata = {}; }` (aggregate; designated initializers) |
| `**metadata` (R-6)                              | `#[derive(Clone, Debug, Default)] struct VariantOptions { pub metadata: Kwargs }`                                   | `RLMeshVariantOptions { size_t struct_size; const RLMeshValue* metadata /* DICT, NULL = {} */; }` + `RLMESH_VARIANT_OPTIONS_INIT` | the `metadata` member's default                                          |
| `.id`, `.params`, `.metadata`                   | public fields                                                                                                       | `rlmesh_variant_id` → `RLMeshBytes`, `rlmesh_variant_params`, `rlmesh_variant_metadata` → `const RLMeshValue*` DICT (borrowed until the next setter or `_free`); `rlmesh_variant_set_id` / `_set_params` / `_set_metadata` (copied; non-DICT → `RLMESH_ERR_INVALID_ARGUMENT`) | members                                                                   |
| params not a mapping → `ValueError`             | type-checked                                                                                                        | non-DICT `params` → `RLMESH_ERR_INVALID_ARGUMENT`                                                                                  | type-checked                                                              |
| `__repr__`, `__eq__` (no `__hash__`)            | `Display`, `Debug`, `PartialEq`                                                                                     | `rlmesh_variant_repr`, `rlmesh_variant_equals`                                                                                     | `operator<<`, `to_string()`, `operator==`                                 |

### `specs.EnvContract`

Read-only; built only by the library (ENV-2).

| Python                                  | Rust (`rlmesh::specs::EnvContract`) | C (`RLMeshEnvContract`)                                         | C++ (`rlmesh::specs::EnvContract`)        |
| --------------------------------------- | ----------------------------------- | --------------------------------------------------------------- | ----------------------------------------- |
| `.id: str`                              | `fn id(&self) -> &str`              | `rlmesh_env_contract_id` → `RLMeshBytes`                        | `std::string id() const`                  |
| `.observation_space: SpaceSpec`         | `fn observation_space(&self) -> &SpaceSpec` | `rlmesh_env_contract_observation_space` (borrowed)     | `SpaceSpec observation_space() const`     |
| `.action_space: SpaceSpec`              | `fn action_space(&self) -> &SpaceSpec` | `rlmesh_env_contract_action_space` (borrowed)                | `SpaceSpec action_space() const`          |
| `.render_mode: str \| None`             | `fn render_mode(&self) -> Option<&str>` | `rlmesh_env_contract_render_mode(c, bool* out_has, RLMeshBytes* out)` | `std::optional<std::string> render_mode() const` |
| `.num_envs: int`                        | `fn num_envs(&self) -> u64`         | `uint64_t rlmesh_env_contract_num_envs`                         | `uint64_t num_envs() const`               |
| `.metadata: dict \| None`               | `fn metadata(&self) -> Option<&Metadata>` | `rlmesh_env_contract_metadata` (borrowed, NULL = None)    | `std::optional<Metadata> metadata() const`|
| `.env`, `.spec` (aliases of self)       | `fn env(&self) -> EnvContract`, `fn spec(&self) -> EnvContract` (copies; ENV-22) | `rlmesh_env_contract_env`, `_spec` → new reference | `EnvContract env() const`, `spec() const` |
| `.to_dict()`                            | `fn to_dict(&self) -> Metadata`     | `rlmesh_env_contract_to_dict(c, RLMeshValue** out)`             | `Metadata to_dict() const`                |
| `__repr__` (no `__eq__`)                | `Display`, `Debug`                  | `rlmesh_env_contract_repr`                                      | `operator<<`, `to_string()`               |
| handle                                  | `Clone` (value)                     | `rlmesh_env_contract_clone`, `_free`                            | copyable                                  |

### `ServedEnv` (Core's `describe_json(served_env=)`)

Python accepts an env, a list of lanes (lane 0 is read) or a vector env, and
reads its spaces as is without closing it (`py/_describe.py:398-432`). A
vector env reports `single_*` spaces plus `num_envs`. This section is the one
definition; [Core](core.md)'s `DescribeJsonOptions.served_env` row links here
(COR-23).

| Python                                     | Rust                                                                                                                                                        | C (`RLMeshDescribeJsonOptions` fields, borrowed for the call)                                       | C++                                                                                     |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| the type                                   | `#[non_exhaustive] pub enum ServedEnv<'a> { Env(&'a dyn Env), Vector(&'a dyn VectorEnv), Lanes(Vec<&'a dyn Env>) }`                                          | the three fields below                                                                             | `class ServedEnv` (non-owning view; the viewed envs must outlive it)                    |
| `served_env` an env                        | `ServedEnv::env(env: &'a impl Env) -> ServedEnv<'a>`; `impl<'a> From<&'a MadeEnv> for ServedEnv<'a>` (either variant)                                        | `const RLMeshEnv* const* served_env; size_t served_env_len;` with one lane                        | `ServedEnv(const Env&)` (implicit)                                                      |
| `served_env` a vector env                  | `ServedEnv::vector(env: &'a impl VectorEnv) -> ServedEnv<'a>` (no blanket `From<&E>`: it would overlap for a type implementing both traits, as in Running's `EnvTarget::vector`) | `const RLMeshVectorEnv* served_vector_env` (NULL = not a vector env); setting it and `served_env` is `RLMESH_ERR_INVALID_ARGUMENT` | `ServedEnv(const VectorEnv&)` (implicit)                                                |
| `served_env` a list of lanes (lane 0 read) | `ServedEnv::lanes<E: Env>(lanes: &'a [E]) -> ServedEnv<'a>`                                                                                                  | `served_env` + `served_env_len` lanes (R-13: NULL = None, non-NULL with 0 = `[]`)                  | `ServedEnv(std::span<const Env> lanes)`; `ServedEnv(const std::vector<Env>& lanes)` (implicit, so `{.served_env = lanes}` compiles: vector → span → `ServedEnv` would be two user-defined conversions); `ServedEnv(std::vector<Env>&&) = delete` (a view of a temporary would dangle) |
| `served_env=[]` → `ValueError` (COR-P5)    | `Lanes` empty → `Error::InvalidArgument` at the describe call                                                                                               | `served_env_len == 0` → `RLMESH_ERR_INVALID_ARGUMENT`                                              | empty span or vector → `rlmesh::ValueError`                                             |

`ServedEnv` reads only `&self` members (spaces, `num_envs`), so describing
never mutates or closes the env. In C and C++ a served `EnvServer` is
described through Core's fenced COR-16 shorthand; Rust has no such extra and
describes the env with `ServedEnv` before moving it into the server (COR-16,
COR-23).

### `EnvServer`

| Python                                                         | Rust                                                                                                                                         | C                                                                                                                                                         | C++                                                                                                    |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `EnvServer(env, address=None, *, …)`, `env` an env             | `EnvServer::new<E: Env + Send + 'static>(env: E, options: EnvServerOptions) -> Result<EnvServer>`                                            | `rlmesh_env_server_new(RLMeshEnv* const* envs, size_t envs_len, const RLMeshEnvServerOptions*, RLMeshEnvServer** out)` with `envs_len == 1`               | `EnvServer(Env env, EnvServerOptions = {})`; template overload for any detected class                  |
| … `env` a sequence of envs (lanes)                             | `EnvServer::lanes<E: Env + Send + 'static>(envs: Vec<E>, options) -> Result<EnvServer>` (`E = Box<dyn Env + Send>` mixes types)              | same call, `envs_len` lanes                                                                                                                               | `EnvServer(std::vector<Env> lanes, EnvServerOptions = {})`                                             |
| … `env` a vector env                                           | `EnvServer::vector<V: VectorEnv + Send + 'static>(env: V, options) -> Result<EnvServer>`                                                      | `rlmesh_env_server_new_vector(RLMeshVectorEnv*, const RLMeshEnvServerOptions*, RLMeshEnvServer** out)`                                                   | `EnvServer(VectorEnv env, EnvServerOptions = {})`                                                      |
| keywords after `env`                                           | `struct EnvServerOptions` (manual `Default`)                                                                                                 | `RLMeshEnvServerOptions` + `RLMESH_ENV_SERVER_OPTIONS_INIT`                                                                                               | `struct EnvServerOptions`                                                                              |
| `address=None` (default `tcp://127.0.0.1:0`)                   | `address: Option<String>` (R-6: it has a default)                                                                                            | `const char* address` (NULL = None)                                                                                                                       | `std::optional<std::string> address`                                                                   |
| `host`, `port`, `path`, `transport`                            | `host: Option<String>`, `port: Option<u16>` (`_client/_endpoint.py:125` refuses outside 0..65535), `path: Option<String>`, `transport: Option<Transport>` | `const char* host`, `bool has_port; uint16_t port`, `const char* path`, `bool has_transport; RLMeshTransport transport`                       | `std::optional<…>` fields                                                                              |
| `Transport = Literal["tcp", "unix"]`                           | `#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] enum Transport { Tcp, Unix }`, `as_str()` and `Display` = `"tcp"` / `"unix"` (R-9)         | `RLMeshTransport { RLMESH_TRANSPORT_INVALID = 0, _TCP = 1, _UNIX = 2 }`, `rlmesh_transport_str`                                                          | `enum class Transport { Tcp = 1, Unix = 2 }` (values of the C enum, as R-23's `WarningCategory`; `unix` is a GNU macro), free `std::string to_string(Transport)` = `"tcp"` / `"unix"` (R-9) |
| `options: ServeOptions \| None = None`                         | `options: Option<ServeOptions>`                                                                                                              | `const RLMeshServeOptions* options` (NULL = None)                                                                                                        | `std::optional<ServeOptions> options`                                                                  |
| `tags: EnvTags \| None = None`                                 | `tags: Option<EnvTags>`                                                                                                                      | `const RLMeshEnvTags* tags` (NULL = None)                                                                                                                 | `std::optional<adapters::EnvTags> tags`                                                                |
| `close_env_on_shutdown=True`                                   | `close_env_on_shutdown: bool` (true)                                                                                                         | `bool close_env_on_shutdown` (INIT true)                                                                                                                  | `bool close_env_on_shutdown = true`                                                                    |
| `framework=`, `device=`                                        | — (R-24)                                                                                                                                     | —                                                                                                                                                         | —                                                                                                      |
| (language extras)                                              | `ext: Option<ext::EnvServerOptions>` (ENV-12); `#[derive(Clone, Debug, Default)] pub struct ext::EnvServerOptions { pub token: Option<String>, pub peer_info_extra: BTreeMap<String, String>, pub publish_describe: bool }` | `const RLMeshExtEnvServerOptions* ext` (NULL = none; [C declarations](#c-declarations))                                                                  | `std::optional<ext::EnvServerOptions> ext`; `struct ext::EnvServerOptions { std::optional<std::string> token; std::map<std::string, std::string> peer_info_extra; bool publish_describe = false; }` (no `host_runtime_json`: the wrapper supplies it, ENV-12) |
| `.address: str`                                                | `fn address(&self) -> &str`                                                                                                                  | `rlmesh_env_server_address(const RLMeshEnvServer*, RLMeshBytes* out)` (a copy)                                                                           | `std::string address() const`                                                                          |
| `.env_contract: EnvContract`                                   | `fn env_contract(&self) -> &EnvContract`                                                                                                     | `const RLMeshEnvContract* rlmesh_env_server_env_contract(const RLMeshEnvServer*)` (borrowed until `_free`)                                                | `const specs::EnvContract& env_contract() const`                                                       |
| `.serve()`                                                     | `fn serve(&self) -> Result<()>` + `serve_async`                                                                                              | `rlmesh_env_server_serve(RLMeshEnvServer*)`                                                                                                               | `void serve()`                                                                                         |
| `.start()`                                                     | `fn start(&self) -> Result<()>`                                                                                                              | `rlmesh_env_server_start`                                                                                                                                 | `void start()`                                                                                         |
| `.wait(timeout=None) -> bool` (R-7.5)                          | `fn wait(&self, timeout: Option<Duration>) -> Result<bool>` + `wait_async`                                                                   | `rlmesh_env_server_wait(RLMeshEnvServer*, const double* timeout /* seconds; NULL = None */, bool* out_stopped)`                                           | `bool wait(std::optional<std::chrono::duration<double>> timeout = std::nullopt)`                       |
| `.shutdown()`                                                  | `fn shutdown(&self) -> Result<()>` + `shutdown_async`                                                                                        | `rlmesh_env_server_shutdown`                                                                                                                              | `void shutdown()`                                                                                      |
| `__enter__` / `__exit__`, finalizer                            | `Drop`: outside a runtime `shutdown()`; inside one, triggers the drain and detaches the rest (ENV-10, R-17). Drop, like Python's finalizer, discards the serve or close error that `__exit__` raises; call `shutdown()` to observe it | `rlmesh_env_server_free` = `shutdown()`, then free; its status is lost, so call `rlmesh_env_server_shutdown` first to observe it | `~EnvServer()` = `shutdown()`, swallowing its error; call `shutdown()` before scope exit to observe it |
| `__repr__` (`EnvServer(address='…')`)                          | `Display`, `Debug`                                                                                                                           | `rlmesh_env_server_repr`                                                                                                                                  | `operator<<`, `to_string()`                                                                            |

### `ServeOptions` (also `ModelServeOptions.options`)

| Python                                       | Rust (`struct ServeOptions`, `Default`)    | C (`RLMeshServeOptions`, `RLMESH_SERVE_OPTIONS_INIT`)            | C++ (`struct ServeOptions`)                                          |
| -------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| `allow_remote_shutdown=False`                | `allow_remote_shutdown: bool`              | `bool allow_remote_shutdown`                                     | `bool allow_remote_shutdown = false`                                 |
| `idle_timeout_seconds=None`                  | `idle_timeout: Option<Duration>`           | `bool has_idle_timeout_seconds; double idle_timeout_seconds`     | `std::optional<std::chrono::duration<double>> idle_timeout`          |
| `drain_timeout_seconds=None` (None = 5 s cap, ENV-7) | `drain_timeout: Option<Duration>`  | `has_drain_timeout_seconds`, `drain_timeout_seconds`             | `std::optional<…> drain_timeout`                                     |
| `close_timeout_seconds=None` (None = 5 s cap, ENV-7) | `close_timeout: Option<Duration>`  | `has_close_timeout_seconds`, `close_timeout_seconds`             | `std::optional<…> close_timeout`                                     |
| `workflow_edition=None`                      | `workflow_edition: Option<String>`         | `const char* workflow_edition` (NULL, `""`, blank = None)        | `std::optional<std::string> workflow_edition`                        |
| `compress_responses=False`                   | `compress_responses: bool`                 | `bool compress_responses`                                        | `bool compress_responses = false`                                    |
| getters of the same names                    | public fields                              | fields                                                           | members                                                              |
| validation, `ValueError`                     | timeouts `> 0` and finite (R-14); blank edition = None; else a retained edition; checked by the consuming call (R-6), `Error::InvalidArgument` | same, `RLMESH_ERR_INVALID_ARGUMENT`                         | same, `rlmesh::ValueError`                                           |
| no `__repr__` / `__eq__`                     | `Debug` only                               | —                                                                | —                                                                    |

### Free functions, constants and bridges

| Python                                                                                   | Rust                                                                                       | C                                                                                                         | C++                                                                                                  |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `trial_index(options, *, lane=None) -> int \| None` (R-7.7)                              | `rlmesh::trial_index(options: Option<&InfoDict>, lane: Option<usize>) -> Result<Option<i64>>` (ENV-21) | `rlmesh_trial_index(const RLMeshValue* options, const size_t* lane, bool* out_has, int64_t* out)` | `std::optional<int64_t> trial_index(const std::optional<InfoDict>& options, std::optional<size_t> lane = std::nullopt)` |
| `ENV_RESET_OPTIONS_KEY`                                                                  | [Core](core.md)                                                                            | Core                                                                                                      | Core                                                                                                 |
| `adapters.EnvTags`, `ObservationRoles`, `tag`, `ENV_METADATA_KEY`, `ENV_BRANCH_METADATA_KEY` | [Adapters](adapters.md)                                                                | Adapters                                                                                                  | Adapters                                                                                             |
| `sanitize_metadata(info)`                                                                | — (D-9)                                                                                    | —                                                                                                         | —                                                                                                    |
| `types.EnvLike`, `VectorEnvLike`, `HasAddress`, `HasEnvContract`, `SupportsMake`, `SupportsResetStep`, `SpaceLike` | — (R-24; the shape they describe is mapped above)                  | —                                                                                                         | —                                                                                                    |
| `RLMESH_EXPECTED_ENV_BRANCH` (env var)                                                   | same variable, same check                                                                  | same                                                                                                      | same                                                                                                 |
| `RLMESH_VALIDATION_POLICY` (env var; served value validation)                            | same variable, same check                                                                  | same                                                                                                      | same                                                                                                 |

`lane` is a position, so it is `usize`; Python refuses a negative lane with
ValueError (`py/_authoring.py:287`).

### Behaviour shared by every language

**`EnvServer` construction**, in order (`py/_server.py:174-301`):

1. Lanes: at least one (`ValueError`), all scalar (`TypeError`). An env
   without `reset`, `step` or its two spaces is `TypeError` in every language:
   C refuses a NULL `reset` / `step` slot at `rlmesh_env_new` /
   `rlmesh_vector_env_new` and a NULL space attribute there too
   (`pyrs/server/py_environment.rs:221-262`; ENV-P4 for Python's missing
   spaces, checked before the one-lane view of step 2 and the tagging of
   step 3). Rust traits and C++ detection cannot omit them.
2. A vector env with `num_envs == 1` is served as a single scalar lane
   (`py/_server.py:207-211`) through the one-lane view below; any other
   `num_envs` below 2 (0) is `ValueError`
   (`pyrs/server/py_environment.rs:396-408`).
3. `tags` merge into each env's metadata through `adapters.tag`: validated
   against the spaces for a scalar env, deferred for a vector env. Without
   `tags`, tags already in a scalar env's metadata are re-validated.
4. `RLMESH_EXPECTED_ENV_BRANCH`, when set and non-empty, must equal
   `metadata[ENV_BRANCH_METADATA_KEY]` after JSON parsing, else `ValueError`
   (`py/_server.py:75-105`).
5. The address is normalized (`py/_client/_endpoint.py:15-53,99-127`); each
   refusal is `ValueError`. An empty `address` is refused, and `address`
   excludes the helpers. `path` excludes `host` and `port`; `Unix` excludes
   `host` and `port` and needs `path`; `path` with `Tcp` is refused (ENV-P3,
   as Remote's RMT-P4); a Unix address on Windows is refused. Then `path` or
   `Unix` gives `unix://<path>`; any of host, port, `Tcp` gives
   `tcp://<host or 127.0.0.1>:<port or 0>`; no helper at all gives the default
   `tcp://127.0.0.1:0`.
6. The workflow edition is resolved (Core, D-7): an explicit
   `options.workflow_edition` is the call's value (COR-6 rung 1), adopted
   verbatim ahead of `RLMESH_WORKFLOW_EDITION`; otherwise
   `RLMESH_WORKFLOW_EDITION`; Python alone then reads the project manifest
   (`py/_server.py:283-290`; D-7). A served env is an authored participant,
   so a server that resolves to no edition (it floats) emits
   `WorkflowEditionWarning` per COR-7 (R-23), unless
   `RLMESH_WORKFLOW_EDITION` is set but blank or a caller resolved it first
   (`py/_editions.py:139-154,251-273`).
7. Each lane's contract is derived: id, spaces, `render_mode`, metadata
   (tags included) and the autoreset mode, which comes from
   `metadata["autoreset_mode"]` (ENV-27; `pyrs/server/conversion.rs:43-109`).
   Every lane's whole derived contract must be equal to lane 0's, else
   `ValueError` (`pyrs/server/py_environment.rs:280-288,382-390`).
8. The endpoint token is resolved (ENV-12): `RLMESH_ENV_ENDPOINT_TOKEN`, when
   set, replaces `ext.token` (which is then not checked); a set variable that
   is empty, whitespace-only, not Unicode or cannot travel in an HTTP header
   fails construction before bind with the `RLMeshException` /
   `Error::Internal` class, as Python's does (P-2;
   `rs/serve_options.rs:111-169`, `pyrs/server/py_env_server.rs:147-153`).
   With the variable unset, an `ext.token` that cannot travel in an HTTP
   header is an invalid options field, so it fails before bind with the
   `ValueError` class (`Error::InvalidArgument`,
   `RLMESH_ERR_INVALID_ARGUMENT`, `rlmesh::ValueError`; R-21). Python has no
   `ext.token`, so R-21 alone gives its class; today both sources are
   `Error::Server` (`rs/serve_options.rs:149-153`).
9. The server binds, so `address` and `env_contract` are readable before
   serving. A bind failure (address in use, a live Unix socket that
   `remove_stale_socket` refuses to unlink) is the `ConnectionError` class
   (`Error::Connection`, `RLMESH_ERR_CONNECTION`, `rlmesh::ConnectionError`;
   `pyrs/server/py_env_server.rs:181-188,215-228`).

**One-lane vector env** (step 2; `py/_single_lane.py:22-125`). The server
wraps the vector env in a scalar view and every later step treats it as a
scalar env:

- Before step 3, the metadata key `autoreset_mode` is read by ENV-27's rule
  (absent, `"NextStep"` or `"Disabled"` accepted; anything else, `"SameStep"`
  included, is `ValueError`; Python change ENV-P5) and then removed, so the
  scalar lane's derived autoreset mode (step 7) is Disabled. The scalar
  lane's metadata is the vector env's metadata without that key, and an empty
  `Dict` when the vector env has none (Python's `dict(metadata or {})`,
  `py/_single_lane.py:32,43`). Spaces are the
  vector env's `single_*` spaces; `num_envs` is hidden (the lane reports 1);
  `render_mode` and `id` pass through.
- `reset(seed, options)` calls the vector `reset` with `VectorSeed::One(seed)`
  (None stays None) and the same options.
- `step(action)` passes the action as a one-lane batch by Python's
  `_batch1` rule (`py/_single_lane.py:55-58,87-96`), which is not the
  stacked layout above for Discrete: Box / MultiBinary / MultiDiscrete: the
  `Tensor` with a leading axis of 1; Dict / Tuple: per child; every other
  kind, Discrete included: a one-item `List` holding the action (Discrete a
  `List` of one `Int`, because the server decodes a Discrete action to a
  Python `int`, which has no `shape`, `pyrs/spaces/value_codec/codec.rs:49-51`).
  A None action for a non-composite single space is the one-item batch
  `Value::List(vec![Value::None])` (C a one-item LIST holding a NONE value; C++
  `Value::list({Value::none()})`), so the vector `step` never receives None
  (ENV-3). For a Dict or Tuple single space a None action fails the step as
  an env failure that is not recoverable (code `INTERNAL`, the "raise
  anything else" row), as Python's `_batch1` raises `TypeError` indexing None
  inside the step callback; a Dict space with no children instead passes an
  empty `Dict`. It returns element 0 of `rewards`, `terminated` and
  `truncated`.
- Lane 0 of an observation walks the single space: Dict and Tuple per child,
  any other kind the first item of a `List` or the first slice along axis 0
  of a `Tensor`.
- Lane 0 of an info dict, per key: a key starting with `_` is dropped; a key
  whose `_<key>` mask has a false element 0 is dropped; a `Dict` value is
  projected recursively; a `List`, `Tuple` or `Tensor` of length 1 (axis 0)
  becomes its element 0 (Python's `_info0` takes any sized non-`str`,
  non-`bytes` value of length 1, `py/_single_lane.py:117-122`); any other value (`Text` and `Bytes` included) is kept as is.
- `render()` returns element 0 of the vector env's frames (a `List`'s first
  item or a `Tensor`'s first slice), or None when it returns None.
- `close()` closes the vector env.

**Served value validation** (`pyrs/server/py_environment.rs:45-60,330-360,585-623`,
`rs/env/wire.rs:34-45,80-100`; `docs/compatibility.md`, "Value conformance").
Every server, in every language, checks each reset and step observation and
each step action (per lane for a vector env, against the single spaces)
against the contract's spaces. `RLMESH_VALIDATION_POLICY`, read at
construction, is `strict` or `off` (trimmed, case-insensitive); unset or any
other value is `warn`.

- A structural deviation (shape, dtype, arity, domain, a missing key, NaN) is
  always rejected, whatever the policy: an action with wire code
  `INVALID_ACTION`, an observation with `INTERNAL`.
- A range deviation (Box bounds, Text charset or length) under `warn` is
  delivered and reported in that reset or step's info under
  `rlmesh.conformance.warning` (the edition's key) as a `List` of
  `{kind, path, detail}` entries, each (kind, path) once per served env (each
  lane of a lane list; a vector env once for all its lanes) for the server's
  life; under `strict` it is rejected as a structural
  one; under `off` it is not checked.
- A `None` step action is not checked (ENV-3).

**Authored env failure text** (ENV-35). When an authored `reset`, `step`,
`render` or `close` fails, the client receives code `INTERNAL` (R-21) with
the message `"{op} failed: {message}"`, `{op}` being `reset`, `step`,
`render` or `close`, so its error reads `[INTERNAL] step failed: boom`
(R-22). `{message}` is, for an Environment-class error, the message the
author gave it, without R-22's `[CODE] ` prefix (Rust
`EnvironmentError.message`, the C text passed to `rlmesh_callback_set_error`
/ `_class`, the C++ and Python constructor's message); for any other error,
its R-22 text (Python `str(exc)`; a C callback that set no error, R-12's
`callback failed with status <n>`). No language puts a class name in it,
since natives carry none (R-22's reason; ENV-P6). A close failure during a
server stop is the serve failure `serve failed: close failed: {message}`
(state machine below).

**Server state machine** (`pyrs/server/py_env_server.rs:370-404,495-572`).
Misuse is builtin `RuntimeError` (`Error::Runtime`, R-21).

| State                      | `serve()`                                              | `start()`                   | `wait(timeout)`                                                  | `shutdown()`                                                                  |
| -------------------------- | ------------------------------------------------------ | --------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| built (bound, not started) | blocks; lane 0 (or the vector env) on the caller       | serves from library threads | error "wait() called before start()"                             | releases the listener; closes the envs if `close_env_on_shutdown`, bounded by `close_timeout` |
| serving in the foreground  | error                                                  | error                       | error                                                            | triggers the drain, moves the state to stopped at once, returns (the drain and close finish inside `serve()`) |
| running in the background  | error                                                  | error                       | `false` on timeout; the call that observes the stop joins drain and close, moves the state to stopped and returns the serve error (ENV-P1 class), once, else `true`; 0 allowed | triggers the drain, joins drain and close, moves the state to stopped, returns the serve error unless a `wait()` already returned it |
| stopped                    | error                                                  | error                       | `true`                                                           | no-op                                                                         |

- `wait(timeout)` validates `timeout` first, in every state: `< 0` or
  non-finite is `ValueError` (`Error::InvalidArgument`,
  `RLMESH_ERR_INVALID_ARGUMENT`, `rlmesh::ValueError`), so `wait(-1)` on a
  built server is `ValueError`, not the "before start()" error
  (`pyrs/server/py_env_server.rs:363-367,478-485`). The column above applies
  to a valid timeout.
- Once `shutdown()` has moved a foreground server to stopped, `wait()`
  returns `true` and `serve()` / `start()` fail "after stopped" while the
  drain may still be running inside `serve()` (`pyrs/server/py_env_server.rs:379-387`).
- Called from inside one of the server's own env callbacks, `shutdown()` only
  triggers the drain and returns. Under `serve()` it moves the state to
  stopped, as from any thread; under `start()` the state stays running in the
  background, so the owner's `wait()` or `shutdown()` observes the stop and
  returns the serve error. Python does this today only under `serve()`; under
  `start()` it joins its own serve thread, which waits on this callback, and
  fails after both 5 s caps (ENV-P2).
- C `rlmesh_env_server_wait` sets `*out_stopped = true` when it returns the
  serve error (the server has stopped), and leaves it unset on any other
  error status.
- Threads: callbacks for one env never overlap, and each lane's (or the
  vector env's) callbacks, `close` included, run on one thread for the
  server's life: under `serve()`, lane 0 (or the vector env) on the calling
  thread and every other lane on its own dedicated plain library thread;
  under `start()` and `serve_async`, every lane (or the vector env) on its
  own dedicated plain library thread. Never on tokio's blocking pool or
  inside a task, so a callback may make sync rlmesh calls (R-16
  "Library-run callbacks") and a thread-bound context (GL, OSMesa) stays
  valid (`pyrs/server/py_env_server.rs:589-647`, `h:559-561`; Python's
  vector env under `start()` changes, ENV-P7).
- A serve failure (including a close past `close_timeout`) is
  `RLMeshException` / `Error::Internal`, from `serve()`, `wait()` and
  `shutdown()` alike, with the one text `serve failed: <message>`, where
  `<message>` is the failure's own text (`close timed out after 5000ms`,
  ENV-35's `close failed: …`) (`pyrs/server/py_env_server.rs:774-780`;
  Python change ENV-P1).
- Signals: ENV-11.

**`EnvFactory` branch table.** Python checks it at class creation
(`py/_authoring.py:381-455`); natively it runs when the factory first enters
the library (`rlmesh_env_factory_new`; the C++ base's first library call;
every Rust `EnvFactoryExt` and Running entry point), before any `prepare` /
`make`. Each failure is `TypeError`:

- every `TagParam` has non-empty `choices` and a `default_` among them;
- at most 8 branches (product of the choices); index 0 is the default branch;
- `tags_for` overridden only alongside `tag_params` (ENV-16);
- no mix of `EnvTags` and None across branches;
- the default branch's `tags_for` equals `tags`.

**Stamping** (`make_env`, `rlmesh_env_factory_make`; `py/_authoring.py:110-149`):
bind the kwargs with each `TagParam.default_` applied; a discriminant value not
in its choices is `ValueError` before `make` runs; call `make` with those bound
kwargs, so every missing discriminant arrives set to its `TagParam.default_`
(Rust `&Kwargs`, the C `kwargs` DICT and C++ `const Kwargs&` alike) and the
body sees the value the stamp records, as Python's signature defaults
guarantee (`py/_authoring.py:110-121,347-358`); tag the env with
`tags_for(branch)` (or `tags`) without validation; stamp
`metadata[ENV_BRANCH_METADATA_KEY] = branch` and
`metadata[ENV_RESET_OPTIONS_KEY] = reset_options` when non-empty.

**`EnvFactory.serve`** (`py/serve.py:423-533`, `py/_bootstrap/loaders.py:403-447`):

0. a `cancel` token already fired fails the call at once with R-21's
   cancellation class, before the branch check, `prepare` or any `make`
   (ENV-33);
1. run the branch-table check, `prepare()`;
2. `make_env(make_kwargs)` once, or `num_envs` times as lanes (each must be
   scalar);
3. resolve the edition (COR-6): `RLMESH_WORKFLOW_EDITION`, then
   `options.workflow_edition`, then `workflow_edition()`; Python alone then
   reads `pyproject.toml` (D-7). A set-but-blank variable stops resolution and
   blank values declare nothing, as COR-6 states (`py/_editions.py:139-155`);
4. build an `EnvServer` with `tags = tags()` only when `tag_params` is empty;
   a branched factory passes no tags, so the stamped branch tags stand
   (`py/serve.py:478-492`; ENV-18);
5. publish the describe envelope (built from the served env) and the startup
   marks on the handshake `PeerInfo.extra`; a describe failure prints one
   stderr line and serving continues (`py/serve.py:68-128`);
6. print `RLMesh serving env on <address> (startup: …)` and `serve()` in the
   foreground. Firing `cancel` while it serves acts as SIGINT (ENV-11): drain,
   close the envs, then the cancellation class; a cancel fired during steps
   1-5 takes effect once the server is built. The factory is not closed
   (Python's `serve` never closes it).

**`EnvFactory.describe`** is best-effort, as Python's
(`py/_describe.py:387-445`): construction, `prepare`, `make` or space failures
become `env_spec: {error, error_type}` badges and the rest of the envelope
ships. `error_type` is the R-21 class name (`ErrorClass::name()`). The
representative env is closed after reading its spaces.

**Render frames** (ENV-5): the frame is any numeric image `Value`, cast to
uint8; shapes `[H, W, 3]`, `[H, W, 4]`, `[3, H, W]`, `[4, H, W]`, `[H, W]` are
PNG-encoded; any other shape yields no frame
(`pyrs/server/conversion.rs:253-289`).

### C declarations

```c
/* Handles declared by this area (opaque). */
typedef struct RLMeshEnv RLMeshEnv;
typedef struct RLMeshVectorEnv RLMeshVectorEnv;
typedef struct RLMeshEnvServer RLMeshEnvServer;
typedef struct RLMeshEnvContract RLMeshEnvContract;
typedef struct RLMeshEnvFactory RLMeshEnvFactory;
typedef struct RLMeshVariant RLMeshVariant;
typedef struct RLMeshVariantSink RLMeshVariantSink;
typedef struct RLMeshVariantList RLMeshVariantList;
typedef struct RLMeshTagParamList RLMeshTagParamList;

typedef enum RLMeshTransport {
  RLMESH_TRANSPORT_INVALID = 0, RLMESH_TRANSPORT_TCP = 1, RLMESH_TRANSPORT_UNIX = 2
} RLMeshTransport;

typedef struct RLMeshServeOptions {
  size_t struct_size;
  bool allow_remote_shutdown;                                    /* false */
  bool has_idle_timeout_seconds;  double idle_timeout_seconds;   /* None; > 0, finite */
  bool has_drain_timeout_seconds; double drain_timeout_seconds;  /* None = 5 s cap */
  bool has_close_timeout_seconds; double close_timeout_seconds;  /* None = 5 s cap */
  const char* workflow_edition;                                  /* NULL/""/blank = None */
  bool compress_responses;                                       /* false */
} RLMeshServeOptions;
#define RLMESH_SERVE_OPTIONS_INIT {sizeof(RLMeshServeOptions), false, false, 0.0, false, 0.0, false, 0.0, NULL, false}

typedef struct RLMeshExtEnvServerOptions {      /* fenced (R-25) */
  size_t struct_size;
  const char* token;                            /* NULL = none; RLMESH_ENV_ENDPOINT_TOKEN overrides when set */
  const RLMeshValue* peer_info_extra;           /* str -> str DICT; NULL = none; non-TEXT entry: RLMESH_ERR_INVALID_ARGUMENT */
  const char* host_runtime_json;                /* NULL = legacy runtime (language "c", no version or toolchain), as today */
  bool publish_describe;                        /* false; ENV-13 */
} RLMeshExtEnvServerOptions;
/* _INIT supplies the caller's translation unit's RLMESH_HOST_RUNTIME_JSON. On MSVC that
 * macro is a function call, so this initializer is valid only for automatic storage. */
#define RLMESH_EXT_ENV_SERVER_OPTIONS_INIT {sizeof(RLMeshExtEnvServerOptions), NULL, NULL, RLMESH_HOST_RUNTIME_JSON, false}

typedef struct RLMeshEnvServerOptions {
  size_t struct_size;
  const char* address;                          /* NULL = None */
  const char* host;
  bool has_port; uint16_t port;
  const char* path;
  bool has_transport; RLMeshTransport transport;  /* has_transport with INVALID is refused */
  const RLMeshServeOptions* options;            /* NULL = None */
  const RLMeshEnvTags* tags;                    /* NULL = None */
  bool close_env_on_shutdown;                   /* true */
  const RLMeshExtEnvServerOptions* ext;         /* NULL = none: no token or extra, no publication, legacy runtime */
} RLMeshEnvServerOptions;
#define RLMESH_ENV_SERVER_OPTIONS_INIT {sizeof(RLMeshEnvServerOptions), NULL, NULL, false, 0, NULL, false, RLMESH_TRANSPORT_INVALID, NULL, NULL, true, NULL}

typedef struct RLMeshEnvFactoryServeOptions {
  size_t struct_size;
  int64_t num_envs;                 /* 1; <= 1 serves one env */
  const char* workflow_edition;     /* NULL/""/blank = None */
  const RLMeshValue* make_kwargs;   /* DICT; NULL = {} */
  RLMeshExtCancelToken* cancel;     /* NULL = none, borrowed for the call (R-18, ENV-33) */
} RLMeshEnvFactoryServeOptions;
#define RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT {sizeof(RLMeshEnvFactoryServeOptions), 1, NULL, NULL, NULL}

typedef struct RLMeshEnvAttributes {            /* deep-copied at rlmesh_env_new */
  size_t struct_size;
  const RLMeshSpace* observation_space;         /* required */
  const RLMeshSpace* action_space;              /* required */
  const char* render_mode;                      /* NULL = None */
  const RLMeshValue* metadata;                  /* DICT; NULL = None */
  const char* id;                               /* NULL = "UnknownEnv-v1" */
} RLMeshEnvAttributes;
#define RLMESH_ENV_ATTRIBUTES_INIT {sizeof(RLMeshEnvAttributes), NULL, NULL, NULL, NULL, NULL}

typedef struct RLMeshVectorEnvAttributes {      /* deep-copied at rlmesh_vector_env_new */
  size_t struct_size;
  uint64_t num_envs;                            /* 0 is refused at server construction (step 2) */
  const RLMeshSpace* single_observation_space;  /* required */
  const RLMeshSpace* single_action_space;       /* required */
  const char* render_mode;                      /* NULL = None */
  const RLMeshValue* metadata;                  /* DICT; NULL = None */
  const char* id;                               /* NULL = "UnknownEnv-v1" */
} RLMeshVectorEnvAttributes;
#define RLMESH_VECTOR_ENV_ATTRIBUTES_INIT {sizeof(RLMeshVectorEnvAttributes), 0, NULL, NULL, NULL, NULL, NULL}

typedef struct RLMeshEnvVtable {
  size_t struct_size;
  RLMeshStatus (*reset)(void* user_data, const int64_t* seed /* NULL = None */, const RLMeshValue* options /* NULL = None */,
                        RLMeshValue** out_observation, RLMeshValue** out_info);              /* required */
  RLMeshStatus (*step)(void* user_data, const RLMeshValue* action /* NULL = None */, RLMeshValue** out_observation,
                       double* out_reward, bool* out_terminated, bool* out_truncated, RLMeshValue** out_info);  /* required */
  RLMeshStatus (*render)(void* user_data, RLMeshValue** out_frame);  /* NULL = None */
  RLMeshStatus (*close)(void* user_data);                            /* NULL = no-op */
  void (*free_user_data)(void* user_data);                           /* NULL = none */
} RLMeshEnvVtable;
#define RLMESH_ENV_VTABLE_INIT {sizeof(RLMeshEnvVtable), NULL, NULL, NULL, NULL, NULL}

typedef struct RLMeshVectorEnvVtable {
  size_t struct_size;
  RLMeshStatus (*reset)(void* user_data, const int64_t* seed, size_t seed_len /* NULL = None; 1 = one int */,
                        const RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info);  /* required */
  RLMeshStatus (*step)(void* user_data, const RLMeshValue* actions, RLMeshValue** out_observation,
                       double* out_rewards, bool* out_terminated, bool* out_truncated,  /* num_envs long, library-allocated */
                       RLMeshValue** out_info);                                         /* required */
  RLMeshStatus (*render)(void* user_data, RLMeshValue** out_frame);  /* NULL = None */
  RLMeshStatus (*close)(void* user_data);                            /* NULL = no-op */
  void (*free_user_data)(void* user_data);                           /* NULL = none */
} RLMeshVectorEnvVtable;
#define RLMESH_VECTOR_ENV_VTABLE_INIT {sizeof(RLMeshVectorEnvVtable), NULL, NULL, NULL, NULL, NULL}

typedef struct RLMeshTagParam {
  size_t struct_size;
  const char* name;                             /* required */
  const RLMeshValue* default_;                  /* required; one of choices */
  const RLMeshValue* choices;                   /* LIST, non-empty */
} RLMeshTagParam;
#define RLMESH_TAG_PARAM_INIT {sizeof(RLMeshTagParam), NULL, NULL, NULL}

typedef struct RLMeshEnvFactoryAttributes {     /* deep-copied at rlmesh_env_factory_new */
  size_t struct_size;
  const RLMeshEnvTags* tags;                                  /* NULL = None */
  const RLMeshTagParam* const* tag_params; size_t tag_params_len;  /* () */
  const char* const* reset_options; size_t reset_options_len;      /* () */
  const char* workflow_edition;                               /* NULL = None */
} RLMeshEnvFactoryAttributes;
#define RLMESH_ENV_FACTORY_ATTRIBUTES_INIT {sizeof(RLMeshEnvFactoryAttributes), NULL, NULL, 0, NULL, 0, NULL}

typedef struct RLMeshEnvFactoryVtable {
  size_t struct_size;
  RLMeshStatus (*prepare)(void* user_data);                          /* NULL = no-op */
  RLMeshStatus (*make)(void* user_data, const RLMeshValue* kwargs /* DICT, discriminants bound */,
                       RLMeshEnv** out_env, RLMeshVectorEnv** out_vector_env);  /* required; exactly one set */
  RLMeshStatus (*tags_for)(void* user_data, const RLMeshValue* params, RLMeshEnvTags** out);  /* NULL = tags */
  RLMeshStatus (*enumerate_variants)(void* user_data, RLMeshVariantSink* sink);              /* NULL = not defined */
  RLMeshStatus (*close)(void* user_data);                            /* NULL = no-op */
  void (*free_user_data)(void* user_data);                           /* NULL = none */
} RLMeshEnvFactoryVtable;
#define RLMESH_ENV_FACTORY_VTABLE_INIT {sizeof(RLMeshEnvFactoryVtable), NULL, NULL, NULL, NULL, NULL, NULL}

/* Env and vector env handles. Attributes: NULL = _INIT, whose NULL spaces are RLMESH_ERR_TYPE. */
RLMeshStatus rlmesh_env_new(const RLMeshEnvVtable* vtable, void* user_data,
                            const RLMeshEnvAttributes* attributes, RLMeshEnv** out);
RLMeshStatus rlmesh_env_close(RLMeshEnv* env);       /* at most once; later calls RLMESH_OK */
void rlmesh_env_free(RLMeshEnv* env);                 /* never calls close; NULL no-op */
RLMeshStatus rlmesh_vector_env_new(const RLMeshVectorEnvVtable* vtable, void* user_data,
                                   const RLMeshVectorEnvAttributes* attributes, RLMeshVectorEnv** out);
RLMeshStatus rlmesh_vector_env_close(RLMeshVectorEnv* env);
void rlmesh_vector_env_free(RLMeshVectorEnv* env);    /* never calls close; NULL no-op */

/* Server. envs_len == 0 is RLMESH_ERR_INVALID_ARGUMENT (envs may then be NULL). */
RLMeshStatus rlmesh_env_server_new(RLMeshEnv* const* envs, size_t envs_len,
                                   const RLMeshEnvServerOptions* options /* NULL = INIT */,
                                   RLMeshEnvServer** out);
RLMeshStatus rlmesh_env_server_new_vector(RLMeshVectorEnv* env,
                                          const RLMeshEnvServerOptions* options /* NULL = INIT */,
                                          RLMeshEnvServer** out);
RLMeshStatus rlmesh_env_server_address(const RLMeshEnvServer* server, RLMeshBytes* out);
const RLMeshEnvContract* rlmesh_env_server_env_contract(const RLMeshEnvServer* server);  /* borrowed until _free */
RLMeshStatus rlmesh_env_server_serve(RLMeshEnvServer* server);
RLMeshStatus rlmesh_env_server_start(RLMeshEnvServer* server);
RLMeshStatus rlmesh_env_server_wait(RLMeshEnvServer* server, const double* timeout /* seconds; NULL = None */,
                                    bool* out_stopped /* required */);
RLMeshStatus rlmesh_env_server_shutdown(RLMeshEnvServer* server);
RLMeshStatus rlmesh_env_server_repr(const RLMeshEnvServer* server, RLMeshBytes* out);
void rlmesh_env_server_free(RLMeshEnvServer* server);  /* shutdown(), then free; NULL no-op */

/* Transport spelling: static storage, "tcp" / "unix"; NULL for INVALID or an unknown value (R-9). */
const char* rlmesh_transport_str(RLMeshTransport transport);

/* EnvContract: an immutable value handle (R-11); borrowed returns live until the contract is freed. */
RLMeshStatus rlmesh_env_contract_id(const RLMeshEnvContract* contract, RLMeshBytes* out);
const RLMeshSpaceSpec* rlmesh_env_contract_observation_space(const RLMeshEnvContract* contract);
const RLMeshSpaceSpec* rlmesh_env_contract_action_space(const RLMeshEnvContract* contract);
RLMeshStatus rlmesh_env_contract_render_mode(const RLMeshEnvContract* contract, bool* out_has, RLMeshBytes* out);
uint64_t rlmesh_env_contract_num_envs(const RLMeshEnvContract* contract);        /* NULL -> 0 */
const RLMeshValue* rlmesh_env_contract_metadata(const RLMeshEnvContract* contract);  /* DICT; NULL = None */
RLMeshEnvContract* rlmesh_env_contract_env(const RLMeshEnvContract* contract);   /* new shared reference (ENV-22) */
RLMeshEnvContract* rlmesh_env_contract_spec(const RLMeshEnvContract* contract);  /* likewise */
RLMeshStatus rlmesh_env_contract_to_dict(const RLMeshEnvContract* contract, RLMeshValue** out);
RLMeshStatus rlmesh_env_contract_repr(const RLMeshEnvContract* contract, RLMeshBytes* out);
RLMeshEnvContract* rlmesh_env_contract_clone(const RLMeshEnvContract* contract); /* as rlmesh_value_clone */
void rlmesh_env_contract_free(RLMeshEnvContract* contract);

/* trial_index (ENV-21). lane: NULL = None. *out_has false = None. */
RLMeshStatus rlmesh_trial_index(const RLMeshValue* options /* DICT; NULL = None */, const size_t* lane,
                                bool* out_has, int64_t* out);

/* Factory. new runs the branch-table check; attributes NULL = INIT (R-20). */
RLMeshStatus rlmesh_env_factory_new(const RLMeshEnvFactoryVtable* vtable, void* user_data,
                                    const RLMeshEnvFactoryAttributes* attributes, RLMeshEnvFactory** out);
void rlmesh_env_factory_free(RLMeshEnvFactory* factory);   /* never calls close; NULL no-op */
RLMeshStatus rlmesh_env_factory_prepare(RLMeshEnvFactory* factory);
/* Stamps (ENV-17). On RLMESH_OK exactly one of *out_env / *out_vector_env is non-NULL (the
 * caller's) and the other is set to NULL; on any other status neither is written. */
RLMeshStatus rlmesh_env_factory_make(RLMeshEnvFactory* factory, const RLMeshValue* kwargs /* DICT; NULL = {} */,
                                     RLMeshEnv** out_env, RLMeshVectorEnv** out_vector_env);
RLMeshStatus rlmesh_env_factory_tags_for(RLMeshEnvFactory* factory, const RLMeshValue* params /* DICT; NULL = {} */,
                                         RLMeshEnvTags** out /* NULL = None */);
RLMeshStatus rlmesh_env_factory_enumerate_variants(RLMeshEnvFactory* factory,
                                                   RLMeshVariantList** out /* NULL = not defined */);
RLMeshStatus rlmesh_env_factory_close(RLMeshEnvFactory* factory);
RLMeshStatus rlmesh_env_factory_serve(RLMeshEnvFactory* factory, const char* address,
                                      const RLMeshEnvFactoryServeOptions* options /* NULL = INIT */);
/* rlmesh_env_factory_describe / _describe_json: Core (COR-10); rlmesh_ext_env_server_describe_json: COR-16. */

/* Factory class-attribute getters (Stable in Python); they read the copied attributes. */
RLMeshStatus rlmesh_env_factory_tags(const RLMeshEnvFactory*, RLMeshEnvTags** out /* NULL = None */);
RLMeshStatus rlmesh_env_factory_reset_options(const RLMeshEnvFactory*, RLMeshValue** out /* LIST of TEXT */);
RLMeshStatus rlmesh_env_factory_tag_params(const RLMeshEnvFactory*, RLMeshTagParamList** out);
RLMeshStatus rlmesh_env_factory_workflow_edition(const RLMeshEnvFactory*, bool* out_has, RLMeshBytes* out);

/* Lists (R-20): items are borrowed until the list is freed. */
size_t rlmesh_variant_list_len(const RLMeshVariantList*);
const RLMeshVariant* rlmesh_variant_list_get(const RLMeshVariantList*, size_t index);  /* NULL past the end */
void rlmesh_variant_list_free(RLMeshVariantList*);
size_t rlmesh_tag_param_list_len(const RLMeshTagParamList*);
/* A lent struct (R-19): struct_size is the library's own sizeof; read a field only if
 * offsetof(field) + sizeof(field) <= p->struct_size, else treat it as its _INIT value.
 * NULL past the end. */
const RLMeshTagParam* rlmesh_tag_param_list_get(const RLMeshTagParamList*, size_t index);
void rlmesh_tag_param_list_free(RLMeshTagParamList*);

/* Variant (a value handle; R-11). */
typedef struct RLMeshVariantOptions {
  size_t struct_size;
  const RLMeshValue* metadata;                  /* DICT, copied; NULL = {} (Python's **metadata) */
} RLMeshVariantOptions;
#define RLMESH_VARIANT_OPTIONS_INIT {sizeof(RLMeshVariantOptions), NULL}
RLMeshStatus rlmesh_variant_new(const char* id, const RLMeshValue* params /* DICT, copied */,
                                const RLMeshVariantOptions* options /* NULL = INIT */, RLMeshVariant** out);
void rlmesh_variant_free(RLMeshVariant*);
RLMeshStatus rlmesh_variant_id(const RLMeshVariant*, RLMeshBytes* out);
const RLMeshValue* rlmesh_variant_params(const RLMeshVariant*);    /* DICT; borrowed until the next setter or _free */
const RLMeshValue* rlmesh_variant_metadata(const RLMeshVariant*);  /* DICT; borrowed likewise */
RLMeshStatus rlmesh_variant_repr(const RLMeshVariant*, RLMeshBytes* out);
bool rlmesh_variant_equals(const RLMeshVariant*, const RLMeshVariant*);
/* Variant fields are mutable, as Python's slots are (py/_variants.py:39-49). */
RLMeshStatus rlmesh_variant_set_id(RLMeshVariant*, const char* id);
RLMeshStatus rlmesh_variant_set_params(RLMeshVariant*, const RLMeshValue* params /* DICT, copied */);
RLMeshStatus rlmesh_variant_set_metadata(RLMeshVariant*, const RLMeshValue* metadata /* DICT, copied */);

/* RLMeshVariantSink is lent to enumerate_variants for that callback only; never stored or freed. */
/* Takes `variant` only on RLMESH_OK; on any other status the caller still owns it
 * and frees it (R-20's ownership rule, applied to a pushed handle). */
RLMeshStatus rlmesh_variant_sink_push(RLMeshVariantSink* sink, RLMeshVariant* variant);
```

Each struct has `RLMESH_<NAME>_MIN_SIZE` and follows R-19. C handle rules go
into `rlmesh.h` as a comment block above each handle:

- **`RLMeshEnv` / `RLMeshVectorEnv`.** Vtable and attributes copied at `_new`;
  `user_data` kept by pointer and released through `free_user_data` at
  `_free`. A server borrows the handle: free the server first. An env handle
  is served by at most one server, in one lane, at a time: a handle that
  appears twice in `envs`, or one already being served or driven by a
  session or run, is refused with `RLMESH_ERR_RUNTIME`, so one `user_data`'s
  callbacks never run on two threads at once. `close` runs at
  most once per env, on the first of: the server stopping with
  `close_env_on_shutdown`, `rlmesh_env_close`, a session's or run's
  `close_env`; later closes return `RLMESH_OK`. `_free` never calls `close`,
  as Python's GC never closes an env, so a C++ `rlmesh::Env` (or one sharing a
  `std::shared_ptr<E>`) destroyed with its server leaves `close` to the
  caller when `close_env_on_shutdown` is false (ENV-26); `_free` only
  releases `user_data` through `free_user_data`. `rlmesh_env_close` on an env
  being served is `RLMESH_ERR_RUNTIME`. `user_data` must be safe to move
  between threads. Never call `_close` / `_free`, or `serve` / `wait` /
  `free` on the serving server, from a callback; `rlmesh_env_server_shutdown`
  is allowed there. Out-params are owned and taken whether or not the
  callback returns 0, as today (`h:531-545,573-581`); NULL `*out_info` = `{}`.
- **`RLMeshEnvServer`.** `_new` binds. `_address` writes a copy into the
  caller's `RLMeshBytes`; the `_env_contract` pointer is borrowed until
  `_free`. Calls that may overlap a running `serve` / `start`,
  from any thread: `_shutdown`, `_address`, `_env_contract`, `_wait` (after
  `start`). `_free` runs `shutdown` and returns only once no callback is
  running or can run, however long (ENV-7, ENV-10). It must not be called
  while `_serve` runs on another thread.
- **`RLMeshEnvFactory`.** Copied at `_new`, which runs the branch-table check.
  Hooks run on the thread that called the `rlmesh_env_factory_*` function, so
  `user_data` must be safe to move between threads. Calls that run hooks
  (`_prepare`, `_make`, `_tags_for`, `_enumerate_variants`, `_close`,
  `_serve`, Core's `_describe` / `_describe_json`, and a session or run
  holding the handle through `rlmesh_env_factory_target`) are exclusive: one
  started while another is in progress on the same handle, from any thread
  or from one of the factory's own hooks, fails with `RLMESH_ERR_RUNTIME`
  before any hook runs; `_serve` holds the handle until it returns (ENV-34).
  The attribute getters (`_tags`, `_reset_options`, `_tag_params`,
  `_workflow_edition`) read the copies and may be called at any time from any
  thread. `_free` must not overlap any call. The `make` callback's
  out-parameters follow the env callbacks' rule: owned and taken whether or
  not it returns 0; returning 0 with neither or both set is `RLMESH_ERR_TYPE`
  (Python's `TypeError` for a `make` that returns no env), and the library
  frees whatever was set. Envs from `rlmesh_env_factory_make` are the
  caller's. `_free` does not call `close`; Python never closes a factory for
  you.

## Python changes

| ID     | Change                                                                                                                                                                                                                                                       | Compatibility             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| ENV-P1 | A failed serve or close (`serve failed: …`, `close timed out after …ms`) raises `RLMeshException` instead of builtin `RuntimeError`, from `shutdown()` and from the `wait()` that observes the stop (`pyrs/server/py_env_server.rs:497-518,774-780`; foreground `serve()` likewise). Extends P-2: R-21 puts server-side failures under `RLMeshException`. Foreground `serve()` also gains the `serve failed: ` prefix that `wait()` and `shutdown()` already add: today it raises the raw message (`pyrs/server/py_env_server.rs:321`), so one failure has two texts. | Subclass of `RuntimeError`; foreground `serve()` failure text gains `serve failed: ` |
| ENV-P2 | `shutdown()` called from one of the server's own env callbacks under `start()` only triggers the drain and returns, as it already does under `serve()`; the owner's `wait()` / `shutdown()` then reports the stop. Today it joins the serve thread, which waits on that callback, and fails with `serve failed: close timed out` after the drain and close caps (`pyrs/server/py_env_server.rs:370-397,656-729`). Changelog: "fixed". | Fix; no success path changes |
| ENV-P3 | `EnvServer(path=…, transport="tcp")` raises `ValueError` instead of silently binding `unix://<path>` (`py/_client/_endpoint.py:41-48`), matching Remote's RMT-P4 for the connect side. Changelog: breaking for that contradictory combination only. | Breaking (refusal) |
| ENV-P4 | An env without its observation or action space (`observation_space` / `action_space`, or `single_*` for a vector env) raises `TypeError` ("env must have 'observation_space'"), as a missing `reset` / `step` / `close` already does. `EnvServer.__init__` runs this check on each lane (or the vector env, reading `single_*`) first, before `SingleLaneEnv` and before `adapters.tag`, so construction step 1 precedes steps 2 and 3. Today three paths leak other classes: the native `getattr` raises `AttributeError` (`pyrs/server/py_environment.rs:248-262`); `SingleLaneEnv.__init__` raises `AttributeError` reading `single_*` for a one-lane vector env (`py/_single_lane.py:41-42`); and with `tags=` given, or tags already in a scalar env's metadata, `adapters.tag`'s validation runs first and raises `AdapterResolutionError` (`py/_server.py:229-258`, `py/adapters/tag.py:46-52`). `AttributeError` has no R-21 row and is not one of the Python-only exceptions R-21 lists. | Breaking for `except AttributeError` / `except AdapterResolutionError` around `EnvServer(...)` with a spaceless env |
| ENV-P5 | A one-lane vector env's `autoreset_mode` metadata is checked by ENV-27's rule before `SingleLaneEnv` strips it: a present None, a non-text non-enum value, an unknown string or `"SameStep"` raises `ValueError`, as it would on any env. Today only an enum (or text) whose upper-cased name is `SAME_STEP` is refused and everything else, the string `"SameStep"` included, is stripped silently (`py/_single_lane.py:18-43`). The `SAME_STEP` message stays. D-16: an accident of `_mode_name`, not a concept. | Breaking (refusal) for invalid keys only |
| ENV-P6 | An authored env callback failure reaches the client as `"{op} failed: {message}"` (ENV-35): the message of an `EnvironmentException` (without a `[CODE] ` prefix) or `str(exc)` of any other exception, with no class name. Today the server formats the `PyErr`, `"{phase} failed: {ExcType}: {exc}"` (`step failed: RuntimeError: boom`, `pyrs/server/py_environment.rs:457`), whose type name natives cannot reproduce (R-22's reason). D-16: an accident of `PyErr`'s `Display`. | Text of the client's `EnvironmentException` changes |
| ENV-P7 | A vector env served under `start()` runs its callbacks on one dedicated plain thread for the server's life, as a lane already does (`Dispatch::Pinned`, `pyrs/server/py_environment.rs:373`), instead of tokio's blocking pool (`Dispatch::Pool`, `:297,442-443`), where a call may hop threads and, once R-16's refusal is in force, a sync rlmesh call from the callback is refused (`spawn_blocking` sets `try_id`). D-16: the pool is an implementation accident (the server state machine's Threads bullet). | Fix; a thread-bound vector env (GL context) starts working under `start()` |
| ENV-P8 | Foreground `serve()` leaves the process's SIGTERM disposition as it found it: the binding uses ENV-11's native installation, which restores it on return. Today the `tokio::signal` SIGTERM handler installed for the serve is never removed (`pyrs/server/py_env_server.rs:400-436,614`), so after one `serve()` a SIGTERM no longer ends the process. The SIGINT handling (`py/_server.py:327-342`) is unchanged. | Fix |
| ENV-P9 | `trial_index` returns None for an integral float outside `[-2^63, 2^63)`, as natives do (ENV-21), instead of `int(value)` past `i64` (`py/_authoring.py:294-302`, `trial_index(…{"trial_index": 1e19})` is `10000000000000000000` today). D-16: an accident of Python's unbounded `int`; natives return `i64`. | Breaking only for such floats |

Nothing else in this area changes Python: the target matches its shapes,
defaults, validation, state machine, SIGINT handling and describe
publication.

## Current differences

### Rust

- Replace the async request/result `Env` / `VectorEnv` (`Send + Sync`,
  required `env_contract`, `reset_subset`, `take_last_phases`;
  `rs/env.rs:37-158`, request types `crates/rlmesh-spaces/src/request.rs`)
  with the sync Gymnasium traits; the server derives the contract
  (`pyrs/server/py_environment.rs:239-287`). Python has no partial vector
  reset, so `reset_subset` goes.
- `render` returns an image `Value`, not PNG bytes
  (`crates/rlmesh-spaces/src/render.rs:7-22`).
- `EnvContract`: public fields become accessors; `render_mode` `""` → `None`;
  spaces stop being `Option`; add `env`, `spec`, `to_dict`; hide
  `autoreset_mode`, which the server now derives from
  `metadata["autoreset_mode"]` (ENV-27) instead of the author setting it;
  drop `PartialEq` (Python has no `__eq__`, R-10; construction step 7's lane
  comparison becomes crate-private) and `Default` (only the library builds a
  contract, ENV-2); move to `rlmesh::specs`
  (`crates/rlmesh-spaces/src/types.rs:219-230`, `rs/lib.rs:166`).
- `EnvServer<E>` builder + `VectorEnvServer` + `BoundEnvServer` (async
  `bind` / `serve`, `rs/env/server.rs:19-211`) collapse into one non-generic
  `EnvServer`: `new` / `lanes` / `vector`, std listeners, own runtime on own
  threads (as `pyrs/server/py_env_server.rs:47-55`), sync methods plus async
  twins, `Drop` as ENV-10 (outside a runtime `shutdown()`; inside one,
  trigger and detach). Today drop does not stop the server and
  `shutdown_trigger` (`:198`) does not join.
- Add the construction checks (expected branch, lane contract at
  construction rather than bind, one-lane vector, tag re-validation) and
  `close_env_on_shutdown`; today the close hook always runs
  (`rs/env/server.rs:202-210`).
- Keep served value validation in the rewritten server. Today it lives in
  `WireEnvAdapter` / `WireLaneAdapter` (`rs/env/wire.rs:34-140`), which every
  Rust and C server binds through (`rs/env/server.rs:67,118`,
  `capi/env.rs:936`); the new `EnvServer` applies the same checks to the
  values its sync lanes produce and receive. One change: range-warning
  deduplication moves from once per server (one `Conformance` per
  `WireLaneAdapter`, shared by all lanes, `rs/env/wire.rs:444-456`) to once
  per lane, as Python's per-`PyEnvironment` set
  (`pyrs/server/py_environment.rs:113,301,350`); a vector env stays once for
  all its lanes.
- An authored env failure is sent as ENV-35's `"{op} failed: {message}"`;
  today the lane sends the author's raw message.
- `ServeOptions`: None drain / close mean wait forever
  (`rs/serve_options.rs:28-34`) → 5 s cap (ENV-7); `token`,
  `predict_concurrency`, `peer_info_extra` (`:39-90`) leave it: the env-side
  `token` and `peer_info_extra` go to `ext::EnvServerOptions` (ENV-12), the
  model's `token` and `predict_concurrency` to [Model](model.md)'s
  `ext::ModelServeOptions` (MDL-20); drop
  `PartialEq, Eq`; validate in the consuming call (today accepted, refused at
  negotiation). `From<ServeOptions>` becomes `ServeOptions::to_grpc` (R-25).
  A configured token that cannot travel in an HTTP header is
  `Error::Server` today (`rs/serve_options.rs:149-153`) → `ext.token` fails
  with `Error::InvalidArgument` (C `RLMESH_ERR_INVALID_ARGUMENT`; step 8);
  the variable's failures keep the `Internal` class.
- Foreground `serve()` installs ENV-11's SIGINT / SIGTERM handling, saving
  the previous dispositions and restoring them on return (so not through
  `tokio::signal`, whose handlers cannot be removed); today no native server
  installs any (only the Python binding's foreground serve does,
  `pyrs/server/py_env_server.rs:431-436,613-615`, and never removes its
  SIGTERM handler, ENV-P8).
- Add `trial_index`, `EnvFactory`, `EnvFactoryExt`, `MadeEnv`, `TagParam`,
  `Variant`, `VariantOptions`, `VectorSeed`, `EnvServerOptions`,
  `EnvFactoryServeOptions`, `Transport`, `ServedEnv` (G-14).
  `TRIAL_INDEX_OPTION` (`rs/lib.rs:164`),
  `LaneEnv`, request/result types, `EndpointPhases` and `RenderFrame`
  (`rs/lib.rs:166`, obsolete once `render` returns a `Value`) move to
  `binding` (R-25); `VectorEnvServer` and `BoundEnvServer` are removed with
  the collapse above. `rlmesh::env` also exports `lane_skew_ns` (with
  `EndpointPhases`, `rs/env.rs:17`) and the `#[doc(hidden)] pub`
  `WireEnvAdapter` / `WireLaneAdapter` (`rs/env.rs:29-30`), which the Python
  binding serves through (`pyrs/server/py_env_server.rs:12`): all three move
  to `binding`, and the two adapters are removed once the binding serves
  through the new `EnvServer`.

### C

- Split `RLMeshEnv` (env + server: `rlmesh_env_new` / `_new_lanes` / `_bind` /
  `_serve` / `_cancel` / `_free`, `h:755-817`) into `RLMeshEnv`,
  `RLMeshVectorEnv` (new; vector envs are missing) and `RLMeshEnvServer`.
- Vtable: callbacks return `int` and `close` returns `void` (`h:582-588`) →
  `RLMeshStatus`; reset / step use `RLMeshResetArgs` with `options_json` and a
  separate `trial_index`, and `info_json` (`h:523-552`) → typed `RLMeshValue*`
  and tuple-order out-params; `trial_index` is read with `rlmesh_trial_index`.
  Add `free_user_data`. `rlmesh_env_new` takes the config before `user_data`
  (`h:760`) → R-20 order (X-6).
- `RLMeshEnvConfig` (`h:713-740`) → `RLMeshEnvAttributes`: `id` default
  `"env"` (`h:715`, `capi/env.rs:758`) → `"UnknownEnv-v1"`; `metadata_json`
  → typed; `adapter_tags_json` → `RLMeshEnvServerOptions.tags` or
  `rlmesh_adapters_tag`; `reset_options` and `variants_catalog_json` →
  `RLMeshEnvFactoryAttributes` / `enumerate_variants`, or
  `metadata[ENV_RESET_OPTIONS_KEY]` for a bare env; `host_runtime_json` and
  `rlmesh_env_config_init` (`h:733-753`) → `RLMeshExtEnvServerOptions`.
- Remove `foreground` (`h:722-727`): `serve()` runs lane 0 on the caller, so
  more than one lane with a foreground serve (`h:768-769`) is allowed.
- Render: today uint8 `[H, W, 3|4]` / `[H, W]` only, error otherwise
  (`capi/env.rs:352-378`) → ENV-5.
- `RLMeshServeOptions` (`h:485-493`): no `struct_size`, millisecond integers
  with 0 = unset, `token` and `predict_concurrency` inline,
  `close_timeout_ms` ignored by envs (`h:490`, `capi/env.rs:919`), no
  `compress_responses` → the struct above (X-1, X-4); `token` to ext; the
  model's `predict_concurrency` moves to [Model](model.md)'s ext.
- `rlmesh_env_free` runs each lane's `close` on a handle that never served
  (stopping a bound server, or closing on the caller for an unbound one,
  `h:811-816`) → `rlmesh_env_free` never closes; `close` runs only on server
  stop with `close_env_on_shutdown`, `rlmesh_env_close` or a session's
  `close_env` (ENV-26).
- `rlmesh_env_cancel` (`h:809`) is a terminal non-blocking trigger →
  `rlmesh_env_server_shutdown` with the state machine; add `start` and `wait`.
- `rlmesh_env_serve` installs no signal handling today → the foreground
  `rlmesh_env_server_serve` installs ENV-11's SIGINT / SIGTERM handling and
  restores the previous dispositions on return (C++ `serve()` inherits it).
- A failed callback sends its raw message, or `environment {op} failed` when
  it set none (`capi/env.rs:310-319`) → ENV-35's `"{op} failed: {message}"`,
  with R-12's `callback failed with status <n>` as the unset message.
- Range-warning deduplication is once per server today (the shared
  `WireLaneAdapter` `Conformance`, `rs/env/wire.rs:444-456`) → once per lane,
  as in Rust above.
- `rlmesh_env_bind` always publishes the describe envelope
  (`h:785-795`, `capi/env.rs:929-940`) → only factory serve, or
  `ext.publish_describe` (ENV-13). `rlmesh_env_describe_json` becomes the
  fenced
  `rlmesh_ext_env_server_describe_json(const RLMeshEnvServer*, RLMeshBytes* out)`
  on the server (COR-16); factories use `rlmesh_env_factory_describe` (typed).
- `RLMeshContract` (`h:229-233`) → `RLMeshEnvContract` with the accessors
  above (G-28).
- Add `RLMeshEnvFactory`, `RLMeshTagParam`, `RLMeshTagParamList`,
  `RLMeshVariant`, `RLMeshVariantOptions`, `RLMeshVariantSink`, `RLMeshVariantList`,
  `rlmesh_trial_index`, `RLMeshTransport` (G-14).

### C++

- C++17 `Result<T>` wrapper (`hpp:1-19`) → exceptions (D-5).
- `class Environment` (`hpp:1287`, a virtual base) → detected classes wrapped
  by `rlmesh::Env`; add `rlmesh::VectorEnv`. Remove `ResetArgs` /
  `ResetOutput` / `StepOutput` (`hpp:1259-1280`) and `EnvConfig`
  (`hpp:1303-1328`).
- `EnvServer::create` / `bind` / `describe_json` / `cancel`
  (`hpp:1332-1462`) → constructors plus `address`, `env_contract`, `serve`,
  `start`, `wait`, `shutdown`; destructor = `shutdown()`; `describe_json`
  becomes `rlmesh::ext::describe_json(const EnvServer&)` (COR-16).
- `ServeOptions` uses `std::chrono::milliseconds` with 0 = unset
  (`hpp:923-929`) → `std::optional<std::chrono::duration<double>>`; add
  `compress_responses`.
- Add `rlmesh::EnvFactoryBase`, `rlmesh::EnvFactory<F>`, `ServedEnv`,
  `MadeEnv`, `TagParam`, `Variant`,
  `VectorSeed`, `EnvServerOptions`, `EnvFactoryServeOptions`, `Transport`,
  `specs::EnvContract`, `trial_index` (G-14).

## Decisions

| ID     | Decision                                                                                                                                                                                                                                                                                                                                  | Status                 |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| ENV-1  | Authoring is sync in every language: no `AsyncEnv`. An env runs on its lane thread; Rust's async twins are only the server's and `EnvFactoryExt::serve_async` (D-1, R-16).                                                                                                                                                              | Settled                |
| ENV-2  | The author writes Gymnasium shape; the server reads the attributes once at construction and derives `EnvContract`. Nobody builds a contract, request or result.                                                                                                                                                                            | Settled                |
| ENV-3  | `step` receives an optional action: Python calls `step(None)` when a request carries none (`pyrs/server/py_environment.rs:641-647`); all languages keep that. Vector `step` always receives actions; the one-lane view passes a None as a one-item batch (construction section).                                                                                                                                       | Settled                |
| ENV-4  | `VectorSeed`: no seed → None, one wire seed → `One` (Python passes an int), two or more → `Each` (a list) (`pyrs/server/py_environment.rs:812-817`). `Each` with one element never arrives from the server.                                                                                                                                | Settled                |
| ENV-5  | Render frames follow Python's lenient rule (cast to uint8; HWC or CHW with 3 or 4 channels, or HW; other shapes give no frame). C's strict check changes. Rust authors return an image `Value`, not PNG.                                                                                                                                    | Settled (R-2)          |
| ENV-6  | An env with no id is `"UnknownEnv-v1"` everywhere (`pyrs/server/py_environment.rs:267-278`); C's `"env"` changes.                                                                                                                                                                                                                         | Settled (R-2)          |
| ENV-7  | A None drain or close timeout means a 5 s cap in every env server (`pyrs/server/py_env_server.rs:45,670-671,765`). C and C++ return from `serve` / `shutdown` at the cap with the error, but `rlmesh_env_server_free` and `rlmesh_env_free` still wait for an abandoned callback, because `user_data` may be freed after they return. Rust detaches it on the lane thread (R-17). | Settled                |
| ENV-8  | One `EnvServer`. Python duck-types the env kind, which a trait bound cannot, so Rust has `new` / `lanes` / `vector`, C `_new` / `_new_vector`, C++ three overloads. `address` has a default, so it is an options field (R-6).                                                                                                           | Settled                |
| ENV-9  | The server state machine and construction order are Python's (tables above).                                                                                                                                                                                                                                                             | Settled                |
| ENV-10 | Drop / destroy / free of a live server runs `shutdown()`, unbounded; drain and close are bounded by their own timeouts. Rust's borrow rules forbid dropping during `serve()`; in C and C++ it is a precondition. Rust exception (R-17, R-16): Drop inside an async execution context triggers the drain and detaches the join and close onto the server's own threads, whose runtime then shuts down in the background, so Drop neither blocks the executor nor panics (dropping a tokio `Runtime` there would); `shutdown_async()` is how to observe errors. | Settled                |
| ENV-11 | Signals, foreground `serve()` only (`py/_server.py:313-342`, `pyrs/server/py_env_server.rs:400-441`): SIGINT stops the server (drain, close) and `serve()` then fails with R-21's cancellation class. A second SIGINT: Python raises KeyboardInterrupt in whatever Python code is running, normally a stuck env call, and restores the previous handler only when `serve()` returns (`py/_server.py:327-342`); a native callback cannot be interrupted, so natively a second SIGINT while the stop is running restores the default SIGINT disposition and re-raises the signal (native decision). SIGTERM stops it gracefully and `serve()` returns success. `start()` installs nothing. Python off the main thread returns success after SIGINT; natively there is no main-thread distinction. `wait()` after `start()`: Python polls signals, so Ctrl-C raises KeyboardInterrupt from `wait()`, the state stays running, and the caller's `with` / `shutdown()` drains and closes the env (`pyrs/server/py_env_server.rs:525-537`). Natively `wait()` is not interruptible and nothing is installed, so SIGINT keeps the process's disposition (by default it ends the process without closing the env); a native caller that needs Python's behaviour installs its own handler that sets a flag, waits with a finite `timeout` in a loop, and calls `shutdown()` when the flag is set (native decision: `EnvServer.wait` takes no `cancel`, as R-7.5 keeps it positional and the server already has a stop path, `shutdown()`, callable from any thread). `EnvFactory.serve`, which returns no handle, takes R-18's `cancel` field instead (ENV-33). **Native installation scope.** A native foreground `serve()` saves the current SIGINT and SIGTERM dispositions on entry and restores both when it returns, by any path (success, error, cancellation), as Python restores its SIGINT handler in a `finally` (`py/_server.py:328-342`); so it does not use `tokio::signal`, whose handlers can never be removed (`pyrs/server/py_env_server.rs:400-436`; Python's own SIGTERM leak is ENV-P8). Concurrent foreground serves in one process, on different threads, env and model alike (MDL-20 serve signals follow this rule), share one process-wide, reference-counted installation: the first to enter saves and installs, a signal stops every foreground server then serving, the second-SIGINT rule above is process-wide, and the last to return restores. On Windows, which has no SIGTERM, the installation is a console control handler (`SetConsoleCtrlHandler`) added on entry and removed on return under the same counting: `CTRL_C_EVENT` acts as SIGINT (a second one removes the handler and returns `FALSE`, so the default handler ends the process, the counterpart of re-raising), and every other console event keeps its default. | Settled                |
| ENV-12 | Rust- and C-only serve fields stay, fenced: `ext::EnvServerOptions { token, peer_info_extra, publish_describe }` (C adds `host_runtime_json`: NULL, and a NULL `ext`, mean the legacy runtime, because the library cannot see the caller's header; `_INIT` and the C++ wrapper, compiled in the caller's translation unit, supply `RLMESH_HOST_RUNTIME_JSON`). Python's `ServeOptions` does not gain `token`; it reads only `RLMESH_ENV_ENDPOINT_TOKEN`. Precedence is today's (`rs/serve_options.rs:38-43,111-169`): the variable, when set, replaces `ext.token` (a deployment secret overrides code), and an invalid variable or token fails construction before bind with step 8's classes (`RLMeshException` for the variable, as Python; `ValueError` for `ext.token`, R-21). Fields (EnvServer table): Rust `token: Option<String>`, `peer_info_extra: BTreeMap<String, String>` (today's type, `rs/serve_options.rs:88`; empty = none), `publish_describe: bool`; C++ the same members as `std::optional<std::string>`, `std::map<std::string, std::string>`, `bool`; C a str → str DICT for `peer_info_extra`, where a non-DICT value or a non-TEXT entry fails construction before bind with `RLMESH_ERR_INVALID_ARGUMENT` (the `ValueError` class, an invalid options field). The C++ struct has no `host_runtime_json`: the wrapper always passes a C `ext` carrying `RLMESH_HOST_RUNTIME_JSON`, built from the C++ `ext`'s fields or, when it is `std::nullopt`, from `_INIT` (no token, no extra, no publication). Factory serve has no `ext` (R-25 fences only the server options' `ext`), so a C or C++ `EnvFactory.serve` reports C's runtime on its handshake and envelope (COR-25's C runtime), as a NULL `host_runtime_json` does; PeerInfo is advisory. | Settled (R-25)         |
| ENV-13 | Describe publication follows Python: `EnvFactory.serve` publishes the envelope on the handshake in every language; a bare `EnvServer` does not. Existing bare C images keep it by setting `ext.publish_describe = true` (an env-only envelope, as C publishes today) or by moving to `RLMeshEnvFactory`.                               | User (accepted default) |
| ENV-14 | `EnvFactory` class attributes and classmethods are type-level: Rust associated fns with no `self` (`tags`, `tag_params`, `reset_options`, `workflow_edition`, `tags_for`, `enumerate_variants`); C++ statics on the CRTP subclass; C attribute fields and callbacks that receive the factory's `user_data`, since C has no class/instance split. Rust uses fns, not associated consts, so the names keep Python's spelling (R-4). | Settled                |
| ENV-15 | `tag_params` carries `TagParam { name, default_, choices }`. Python reads the default from `make`'s signature and the choices from the Experimental `params`; natives have no signature reflection, so the branch table's inputs are declared with the names. Independent of `params`; Python is unchanged.                             | User (accepted default) |
| ENV-16 | Branch-table checks run when the factory first enters the library, with Python's `TypeError`. Rust cannot observe whether `tags_for` was overridden, so it skips that one check; C (non-NULL slot) and C++ (static function address) run it.                                                                                           | Settled                |
| ENV-17 | Stamping: Python's `make` wrapper stamps even on a direct call. C's `rlmesh_env_factory_make` stamps. Rust and C++ cannot wrap an override, so the authored `make` is unstamped and the provided `make_env` stamps; every library path (serve, run, session, describe) uses `make_env`.                                                   | Settled                |
| ENV-18 | `EnvFactory.serve` passes `tags` to the server only when `tag_params` is empty; a branched factory's stamped tags stand.                                                                                                                                                                                                                 | Settled                |
| ENV-19 | `describe` is best-effort (badges, never a failed call for env-spec problems). Python's classmethod builds `cls()`; the Rust type-level form needs `Self: Default` and C++ a default-constructible `F`. A factory without a no-argument constructor uses Core's instance form `rlmesh::describe(&mut f)` (Python's `describe(instance)`); C describes the handle. | Settled                |
| ENV-20 | C++ `EnvFactory<F>` is a CRTP base: static members dispatch to the subclass at compile time, which a virtual base cannot do for statics. It derives from the non-template, non-copyable, non-movable `EnvFactoryBase`, which owns the `RLMeshEnvFactory*` (built on first library use, `user_data` = the `F` object) and is what `EnvTarget` and describe take. Envs are detected classes (R-12), matching Python's duck typing. | Settled                |
| ENV-21 | `trial_index` returns a signed `i64` and follows `_as_ordinal` (`py/_authoring.py:294-302`): integers as is (negatives too), integral floats in `[-2^63, 2^63)` converted, bools, integral floats outside that range (`1e19`) and anything else None (Python change ENV-P9).                                                                                                                                                    | Settled                |
| ENV-22 | `EnvContract.env` and `.spec` are mirrored as copies of the contract; Python is not asked to deprecate them.                                                                                                                                                                                                                             | Settled                |
| ENV-23 | Rust envs that are `!Send` (a GL context): `rlmesh::ext::LocalEnvServer::new(env, options)` with `address`, `env_contract`, `serve` only (foreground; stopped by signals, idle timeout or remote shutdown). Every `EnvServer` constructor requires `Send` because `start()` moves lanes to library threads.                              | User (accepted default) |
| ENV-24 | `wait`'s parameter is `timeout` in every language: Rust and C++ type it as a duration (`Option<Duration>`, `std::optional<std::chrono::duration<double>>`), C as `const double* timeout` in seconds. R-4's only rename is the `_seconds` suffix Python already has; Python's `timeout` has none, so C adds none (E-2 rule 2). Revised r5; it said C `timeout_seconds`.                                                                                                                                                                                                    | Settled                |
| ENV-25 | Native `close` defaults to a no-op (NULL slot in C, absent member in C++), as `gymnasium.Env.close` does. Python refuses an object with no `close` (`pyrs/server/py_environment.rs:233-235`); every Gymnasium env inherits one.                                                                                                       | Settled                |
| ENV-26 | Ownership: Rust servers take envs by value (`Arc<Mutex<E>>` keeps access); C servers borrow env handles; C++ servers own the `rlmesh::Env` they are given (a `std::shared_ptr<E>` keeps access). `close_env_on_shutdown = false` is for envs the caller closes: in no language does releasing an env close it (Rust drop, `rlmesh_env_free`, `~Env`), as Python's GC never does, so destroying a C++ server, which frees its `rlmesh::Env` handles, leaves a shared env open. `close` runs only on server stop with `close_env_on_shutdown`, an explicit close, or a session's `close_env`. Revised r5: `rlmesh_env_free` no longer closes (Current differences, C).                                                                        | Settled                |
| ENV-27 | The autoreset mode is derived from the env's metadata key `autoreset_mode` for every env, scalar or vector, in every language (`pyrs/server/conversion.rs:43-109`): absent key or no metadata → Disabled; `"NextStep"` / `"Disabled"` accepted; a present None, a non-text value, an unknown string or `"SameStep"` (reserved, not honoured) → `ValueError`. Natives read the TEXT value of `metadata()`; Python also accepts a gymnasium `AutoresetMode` enum by its `.value` (R-24). It is part of the lane contract compared in construction step 7 and is not exposed on `EnvContract`. A self-autoresetting vector env declares `NextStep` this way. A one-lane vector env's key is checked by this rule and then removed, so its scalar lane is Disabled (ENV-31, ENV-P5). | Settled                |
| ENV-28 | An env handle is served by at most one server, in one lane, at a time (C handle rules). C++ handles built from one `std::shared_ptr<E>` share the object, not the handle; serving two of them at once is a caller precondition, as in C with one `user_data` behind two handles. Rust (by value; `Arc<Mutex<E>>` serializes shared use) and C++ (owning, move-only `rlmesh::Env`) cannot express it through one handle; C can, so the ABI refuses it with `RLMESH_ERR_RUNTIME`, using the "being served" state `rlmesh_env_close` already checks. | Settled                |
| ENV-29 | `EnvServer::serve_async` and `EnvFactoryExt::serve_async` run every lane on library threads (as `start()`), install no signal handlers and resolve when the server stops, with the serve error. Only sync `serve()` keeps lane 0 on the caller and installs ENV-11's handlers; running sync env callbacks inside the future would block the caller's executor. Exception: `EnvFactoryExt::serve_async(&mut self, …)` runs the branch check, `prepare()` and `make_env` on the polling thread inside its first poll, blocking it until the server is built, because the future borrows the factory and a borrowed `&mut F` cannot move to a library thread (that would need `F: Send + 'static` and taking the factory by value, which Python's instance method does not do). A caller that cannot block its executor builds the envs itself and uses `EnvServer::serve_async`. Dropping either future: ENV-32. | Settled                |
| ENV-30 | Served value validation is server behaviour, not author code, so every server in every language keeps it with today's rules and today's variable (`RLMESH_VALIDATION_POLICY`; construction section). It is part of the user contract (`docs/compatibility.md`, "Value conformance"), so the sync rewrite must not drop it with `WireEnvAdapter`. | Settled |
| ENV-31 | A one-lane vector env is served as a scalar lane in every language through Python's `SingleLaneEnv` projection (construction section): the wire's only vector signal is `num_envs`, and a scalar lane gets driver-owned resets, per-episode seeds and `trial_index`. | Settled |
| ENV-32 | Dropped futures (R-18). A future dropped before its first poll has done nothing and does nothing. Dropping `EnvServer::serve_async` afterwards acts as `shutdown()` on a foreground serve: it triggers the drain and moves the state to stopped at once; the drain and the close (per `close_env_on_shutdown`) finish detached on the server's own threads, as in ENV-10's in-runtime Drop, and the serve error is discarded. A later `wait()` returns `true`, `shutdown()` is a no-op, and the server's Drop joins the detached work outside a runtime and detaches it inside one (ENV-10). Dropping `wait_async` only stops waiting and leaves the state as it was, as a `wait()` interrupted by KeyboardInterrupt does (ENV-11); if it had observed the stop, the join finishes detached, the state is stopped and the serve error is discarded. Dropping `shutdown_async` leaves the drain it triggered to finish detached; the state is stopped and the serve error is discarded. Dropping `EnvFactoryExt::serve_async` after its first poll stops the server it built as a dropped `EnvServer::serve_async` does, closing the envs it made (factory serve keeps `close_env_on_shutdown` true); the factory is not closed. No new env call starts after the drop; callbacks in flight finish inside the drain. Model's dropped `serve_async` stops its server likewise ([Model](model.md) Serve). | Settled |
| ENV-33 | `EnvFactoryServeOptions` carries R-18's `cancel` (Rust `Option<ext::CancelToken>`, C `RLMeshExtCancelToken*` borrowed for the call, C++ `std::optional<ext::CancelToken>`), as Model's `ModelServeOptions::cancel`: `EnvFactory.serve` blocks and returns no handle, so a native caller on a worker thread has no other stop path than signals, idle timeout or remote shutdown. Firing it acts as SIGINT (ENV-11): drain, close the envs, then R-21's cancellation class; a token already fired fails the call at its start, before the branch check and `prepare`; a cancel fired during the pre-serve steps takes effect once the server is built. The factory is not closed. It is one of R-25's fenced `cancel` fields, listed under `[ext]` by E-2 rule 4. Python is unchanged (KeyboardInterrupt). | Settled |
| ENV-34 | C factory calls that run hooks are exclusive per handle and refuse an overlapping call with `RLMESH_ERR_RUNTIME` (C handle rules), matching Rust's `&mut self` on `make` / `prepare` / `close` / `serve` and ENV-28's refusal for env handles; serializing instead would deadlock a `_make` issued while `_serve` holds the handle. Python, under the GIL, does not refuse; a concurrent Python factory is not a supported pattern. C++ inherits the rule through `EnvFactoryBase`. | Settled |
| ENV-35 | An authored env failure reaches the client as code `INTERNAL` with the text `"{op} failed: {message}"` in every language (`{op}` the callback: `reset`, `step`, `render`, `close`; `{message}` the author's message for an Environment-class error, else the error's R-22 text; behaviour section). Python's `"{phase} failed: {ExcType}: …"` (`pyrs/server/py_environment.rs:457`) carries a class name natives cannot produce, and C sends the raw message (`capi/env.rs:310-319`); one text lets E-3's Errors family compare a scripted env failure across server languages (ENV-P6). | Settled |

### Review notes

- Applied r1-env 1: `TagParam` carries name, default and choices, so the
  branch-count, default-in-choices, mixed-tag and default-branch checks and
  describe's branch table run natively (ENV-15, ENV-16).
- Applied r1-env 2: ENV-18.
- Applied r1-env 3: ENV-21 (signed, integral floats, bools None).
- Applied r1-env 4: `tags_for` and `enumerate_variants` are type-level
  (ENV-14).
- Applied r1-env 5: CRTP base and detected envs (ENV-20).
- Applied r1-env 6, partly: describe is best-effort with `error` /
  `error_type` badges (ENV-19). Rejected review (r1-env 6, "optional
  representative constructor"): Python's own representative is `cls()`, which
  is `Default`; the instance form (`describe(instance)`,
  `py/_describe.py:469`) already covers other factories, so no new hook.
- Applied r1-env 7: ENV-4.
- Applied r1-env 8: ENV-11.
- Applied the r1-env defaults: 5 s cap (ENV-7), sync authoring (ENV-1),
  contract aliases (ENV-22), best-effort describe (ENV-19), minimal
  descriptor (ENV-15), `make_env` (ENV-17), lenient render (ENV-5), signals
  (ENV-11), fenced extras (ENV-12), `UnknownEnv-v1` (ENV-6), `timeout_seconds`
  (ENV-24), factory-only publication (ENV-13), `LocalEnvServer` (ENV-23),
  `step(None)` (ENV-3).
- Rejected review (round 0, "Drop/free should bound at 15 s like Python's
  finalizer"): the finalizer calls `shutdown()` first, which joins without a
  bound (`pyrs/server/py_env_server.rs:370-397`); the later `wait(15.0)` only
  returns true (ENV-10).
- Rejected review (round 0, partly, "drop `Send` and require it only on
  `start()`"): the traits drop `Send`, but a non-generic `EnvServer` cannot
  make `start()` depend on the env's `Send`; ENV-23 covers `!Send` envs.
- `EnvTags` equality and hashing are owned by [Adapters](adapters.md).
- Consistency pass: the Rust shape builds spaces with Spaces' fallible
  constructors; C++ `VectorSeed` excludes None, as Rust's does
  (`std::optional<VectorSeed>`); the factory describe signature and the
  served-env describe follow Core (COR-10, COR-16).
- Applied r2 (17 findings): factory-serve edition chain is COR-6 with
  Python's `pyproject.toml` rung marked Python-only; the state machine gives
  `wait()` the serve error once and moves a foreground server to stopped on
  `shutdown()` (ENV-P1 extended); autoreset derivation and whole-contract lane
  equality (ENV-27, step 7); token precedence as today (ENV-12, step 8);
  one server per C env handle (ENV-28); Rust Drop inside a runtime
  (ENV-10); `EnvFactoryBase` (ENV-20); batched layout per space kind;
  in-callback `shutdown()` under `start()` (ENV-P2); second SIGINT (ENV-11);
  address and vector `num_envs` refusals, `path` + `Tcp` refused (ENV-P3);
  `make` receives bound discriminant defaults; missing C structs, getters,
  list accessors and `Variant` setters; `ServedEnv` defined here;
  `serve_async` threading (ENV-29); `RenderFrame`, `VectorEnvServer` and
  `BoundEnvServer` placement.
- Rejected review (r2, partly, "`ServedEnv` `From<&EnvServer>`"): Python's
  `served_env` is an env, a lane list or a vector env, read for its spaces
  (`py/_describe.py:398-432`); a server is not one, and describing a served
  server is Core's fenced COR-16 shorthand.
- Cross-file (r2): running.md:143 — C++ `EnvTarget` is implicit from
  `EnvFactoryBase&` (not the class template `EnvFactory&`); C
  `rlmesh_env_target` / `rlmesh_vector_env_target` sessions and runs refuse a
  handle already being served or driven with `RLMESH_ERR_RUNTIME` and mark it
  driven while they hold it (ENV-28).
- Cross-file (r2): rules.md R-27 — add `ServedEnv` to Env's owner list.
- Applied r3 (all 16; findings 1 and 9 are one change): COR-16's
  `rlmesh_ext_env_server_describe_json` name; `ServedEnv` is core's lane form,
  defined here once (env, vector, lanes; C pointer + `served_env_len`; C++
  span; COR-P5's empty list); served value validation (ENV-30, env-var row);
  the one-lane view spelled out with `autoreset_mode` checked then removed
  (ENV-31, ENV-P5); `host_runtime_json` NULL = legacy runtime and the MSVC
  `_INIT` caveat (ENV-12); float warning (step 6) and bind
  `ConnectionError` (step 9); missing spaces and NULL `make` are `TypeError`
  (ENV-P4); `ServeOptions` heading and the token / `predict_concurrency`
  split (MDL-20); `VariantOptions` for `**metadata` (R-6); `EnvContract`
  drops `PartialEq` and `Default`; `lane_skew_ns` and the wire adapters go to
  `binding`; native signal installation listed and native `wait()` decided
  (ENV-11); `EnvFactoryExt::serve_async` pre-serve hooks block the poll
  (ENV-29); C forward typedefs, `RLMeshVariantSink` / `_push` ownership and
  variant getters declared; `rlmesh_tag_param_list_get` returns a lent struct
  (R-19, which now covers borrowed returns); `_server.py:174-301` citation.
- Rejected review (r3, finding 5, the R-19 half): R-19's "Lent structs"
  bullet already covers sized structs the library owns and lends as a
  borrowed return (rules.md R-19), so no rules change is needed; env.md only
  cites it.
- Cross-file (r3): core.md:161 — replace the restated `ServedEnv` signature
  with a link to [Env](env.md#servedenv-cores-describe_jsonserved_env); the
  definitions now match, so this is a de-duplication only.
- Cross-file (r3): rules.md R-27 (Env owner list) and R-25 prelude list —
  add `VariantOptions` beside `Variant`.
- Applied r4 (all 8; none rejected): the one-lane view's `step` follows
  `_batch1` (Discrete is a one-item `List`, a None action is
  `List([None])` for a non-composite space and an env failure for Dict /
  Tuple), so ENV-3 holds; `Tuple` info values and `{}` metadata in the view;
  `serve_async` cites R-16's exception; an invalid `ext.token` is the
  `ValueError` class (step 8, ENV-12); the Shape calls `shutdown()` and the
  `__exit__` row says Drop / destructor / free discard its error; C++
  `ServedEnv(const std::vector<Env>&)`; the C recoverable cell names
  `rlmesh_callback_set_error_class` (R-12).
- Applied r5 (all 9; one sub-point of finding 3 rejected below): `rlmesh_env_free`
  never closes, so a C++ server destroyed over a shared env leaves `close` to
  the caller (handle rules, ENV-26, Current differences C); dropped
  `_async` futures (ENV-32); `EnvFactoryServeOptions::cancel` (ENV-33, ENV-11
  reworded); `wait` validates its timeout in every state; C `wait`'s
  parameter is `timeout` (ENV-24); every exported env-area C function
  declared with full types, factory `make` outputs (neither or both is
  `RLMESH_ERR_TYPE`, taken on any return), NULL `kwargs` / `params` = `{}`,
  NULL factory attributes = `_INIT` (R-20); factory handle threading
  (ENV-34); `EnvFactory<F>`'s public non-explicit default constructor makes
  `Factory{}` valid; status and Owns list.
- Rejected review (r5, finding 3 part "list it under E-2 rule 4"): E-2 rule 4
  already exempts "R-18's `cancel` fields" generically (rules.md E-2), so the
  new field needs no rules change.
- Cross-file (r5): rules.md E-3 Cancellation — add: dropping a polled
  `EnvServer::serve_async` stops the server, closes the env with
  `close_env_on_shutdown` and does not block the dropping task; dropping
  `wait_async` leaves the server running; dropping a polled
  `EnvFactoryExt::serve_async` closes the envs it made; a factory serve whose
  `cancel` fires drains, closes the envs and fails with the cancellation
  class, and one with an already-fired token runs no `prepare` (ENV-32,
  ENV-33).
- Cross-file (r5): running.md (RUN-27 / `rlmesh_env_factory_target`) — a C
  session or run holding a factory through `rlmesh_env_factory_target` marks
  the handle busy while it holds it and refuses a busy handle with
  `RLMESH_ERR_RUNTIME` (ENV-34), as it does for env handles (ENV-28).
- Cross-file (r5): rules.md R-27 Env owner row — add C++ `EnvFactoryBase`
  (defined here, ENV-20).
- Cross-file (r2): core.md:126 — Rust and C++ `ServedEnv` are defined in
  Env; the C column adds `const RLMeshVectorEnv* served_vector_env` beside
  `served_env` (at most one set, else `RLMESH_ERR_INVALID_ARGUMENT`), since a
  vector env is a valid `served_env` in Python.
- Applied r6 (all 11; none rejected): authored env failure text
  `"{op} failed: {message}"` with no class name (ENV-35, ENV-P6, Current
  differences Rust and C); native signal installation saves and restores
  both dispositions, is shared and reference-counted across concurrent
  foreground serves, and has a Windows console-handler form (ENV-11;
  Python's leaked SIGTERM handler, ENV-P8); each lane or vector env runs on
  one dedicated plain thread, never the blocking pool (Threads bullet,
  ENV-P7); exact `ext::EnvServerOptions` fields in Rust and C++, C's
  non-TEXT `peer_info_extra` entry refused as the ValueError class, the C++
  wrapper always passing `RLMESH_HOST_RUNTIME_JSON` (ENV-12); per-lane
  range-warning deduplication listed as a Rust and C change; ENV-P4 checks
  spaces before `SingleLaneEnv` and `tag`; one `serve failed: <message>`
  text from `serve()`, `wait()` and `shutdown()` (ENV-P1); `Transport`
  carries C's numbers in C++ and R-9's text forms; factory serve reports C's
  runtime (ENV-12; the alternative, an `ext` on the factory serve options,
  would need an R-25 change); `trial_index` gives None for integral floats
  outside `i64` (ENV-21, ENV-P9); the `ServedEnv` closing paragraph follows
  COR-16 / COR-23.
- Cross-file (r6): core.md COR-25 — after "only a served env's envelope
  carries the C++ host runtime, through ENV-12's `host_runtime_json`", add
  that a C or C++ `EnvFactory.serve` reports C's runtime on its handshake
  and envelope, because factory serve options have no `ext` (ENV-12).
- Cross-file (r6): rules.md E-3 Errors — add: a scripted authored env
  failure (`step` raising / returning `boom`, not recoverable) reads
  `[INTERNAL] step failed: boom` at the client for a server in every
  language (ENV-35); and a failed serve reads `serve failed: <message>` from
  `serve()`, `wait()` and `shutdown()` (ENV-P1).
- Cross-file (r6): model.md Serve, Signals bullet — cite ENV-11's native
  installation scope: dispositions saved on entry and restored on return,
  one reference-counted installation shared with env serves, Windows
  console handler (MDL-20 inherits it).
