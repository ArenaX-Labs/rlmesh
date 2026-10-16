// Executable ABI 5 C++17 parity fixtures. Built with GCC and Clang by test:cxx.
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <rlmesh.hpp>
#include <thread>
#include <type_traits>
#include <unordered_set>
using namespace rlmesh;
using namespace rlmesh::spaces;
namespace {
static_assert(std::is_same_v<std::iterator_traits<Dict::key_iterator>::iterator_category,
                             std::random_access_iterator_tag>);
static_assert(!std::is_copy_constructible_v<Env>);
static_assert(!std::is_copy_constructible_v<Session>);
static_assert(!std::is_copy_constructible_v<RemoteEnv>);
static_assert(std::is_copy_constructible_v<Model>);
static_assert(std::is_copy_constructible_v<RemoteModel>);
static_assert(std::is_copy_constructible_v<ext::CancelToken>);
static_assert(std::is_nothrow_move_constructible_v<ext::CancelToken>);
static_assert(std::is_constructible_v<ServedEnv, Env&>);
static_assert(std::is_constructible_v<ServedEnv, VectorEnv&>);
static_assert(std::is_constructible_v<ServedEnv, rlmesh::Span<const Env>&>);
static_assert(!std::is_constructible_v<ServedEnv, Env&&>);
static_assert(!std::is_constructible_v<ServedEnv, const Env&&>);
static_assert(!std::is_constructible_v<ServedEnv, VectorEnv&&>);
static_assert(!std::is_constructible_v<ServedEnv, const VectorEnv&&>);
static_assert(!std::is_constructible_v<ServedEnv, rlmesh::Span<const Env>&&>);
static_assert(!std::is_constructible_v<ServedEnv, const rlmesh::Span<const Env>&&>);
static_assert(!std::is_constructible_v<ServedEnv, std::vector<Env>&&>);
static_assert(!std::is_constructible_v<ServedEnv, const std::vector<Env>&&>);
static_assert(!std::is_constructible_v<Value, std::nullptr_t>);
static_assert(!std::is_constructible_v<Value, char>);
static_assert(std::is_constructible_v<Value, unsigned char>);

struct Counter {
  int steps = 0, closes = 0;
  Space observation_space() const { return Discrete(10); }
  Space action_space() const { return Discrete(10); }
  std::optional<std::string> id() const { return "CppCounter-v0"; }
  ResetResult reset(std::optional<int64_t>, const std::optional<InfoDict>&) {
    steps = 0;
    return {0, {}};
  }
  StepResult step(std::optional<Value> action) {
    assert(action);
    ++steps;
    return {steps, 1.0, steps >= 2, false, {{"is_success", true}}};
  }
  std::optional<Value> render() { return Value(steps); }
  void close() { ++closes; }
};
struct Greedy {
  int* ends = nullptr;
  Value predict(const Value&) { return Value(0); }
  void on_episode_end() {
    if (ends) ++*ends;
  }
};
struct Factory : EnvFactory<Factory> {
  int builds = 0, prepares = 0;
  MadeEnv make(const Kwargs&) {
    ++builds;
    return Env(Counter{});
  }
  void prepare() { ++prepares; }
  static std::vector<std::string> reset_options() { return {"trial_index"}; }
  static std::optional<std::vector<Variant>> enumerate_variants() {
    return std::vector<Variant>{[&] {
      rlmesh::Variant value{};
      value.id = "default";
      value.params = {};
      return value;
    }()};
  }
};
struct Loaded {
  static constexpr std::array load_keys{types::LoadKey{"weight", true}};
  int64_t weight = 0;
  void load(const Kwargs& k) { weight = k.at("weight").as_int(); }
  Value predict(const Value&) { return weight; }
};
struct Hooks : RunHooks {
  int starts = 0, steps = 0, ends = 0;
  std::shared_ptr<RunContext> kept;
  void on_run_start(std::shared_ptr<RunContext> c) override {
    ++starts;
    kept = std::move(c);
  }
  void on_step(const StepEvent& e) override {
    ++steps;
    assert(e.reward == 1);
  }
  void on_run_end(const RunResult& r) override {
    ++ends;
    assert(r.num_episodes() == 2);
  }
};
struct OwnError : TypeError {
  int tag;
  explicit OwnError(int n) : TypeError("author exception"), tag(n) {}
};
void values_spaces() {
  Value zero(0);
  assert(zero.kind() == ValueKind::Int);
  assert(!(zero == Value(0.0)));
  assert(Value(uint8_t(7)).as_int() == 7);
  try {
    Value too_big(UINT64_MAX);
    assert(false);
  } catch (const ValueError&) {
  }
  Metadata m{{std::string("a\0b", 3), "text"}, {"n", 0}};
  m.insert_or_assign("n", 1);
  assert(m.begin()->first.size() == 3);
  assert(Value(m).as_dict() == m);
  assert(Value::tuple({1, 2}).as_tuple().size() == 2);
  assert(Value::list({1, 2})[1].as_int() == 2);
  std::array<std::byte, 2> bytes{std::byte{0}, std::byte{255}};
  assert(Value::bytes(bytes).as_bytes()[1] == std::byte{255});
  float data[] = {1, 2};
  Tensor t(data, sizeof(data), {2}, DType::Float32);
  Tensor shared = t;
  auto original = t.buffer().data();
  t = shared.reshape({1, 2});
  assert(t.buffer().data() == original);
  assert(t.copy().buffer().data() != original);
  assert(t.tobytes().size() == sizeof(data));
  auto dl = t.to_dlpack();
  assert(dl->flags & DLPACK_FLAG_BITMASK_READ_ONLY);
  auto imported = Tensor::from_dlpack(std::move(dl));
  assert(!dl);
  assert(imported.shape() == t.shape());
  auto writable = t.to_dlpack(rlmesh::ToDlpackOptions{}.with_copy(true));
  assert(!(writable->flags & DLPACK_FLAG_BITMASK_READ_ONLY));
  Dict dict({{"p", Box(-1, 1, {2})}, {"id", Discrete(3)}});
  assert(dict.end() - dict.begin() == 2);
  assert(std::distance(dict.begin(), dict.end()) == 2);
  assert(dict.begin() < dict.end() && dict.begin() <= dict.end());
  assert(dict.end() > dict.begin() && dict.end() >= dict.begin());
  assert(dict.begin() != dict.end() && dict.begin() == dict.begin());
  assert(dict.begin()[0] == "id");
  auto child = dict["id"];
  child.seed(42);
  auto first = child.sample();
  dict["id"].seed(42);
  assert(first == child.sample());
  assert(dict.contains(dict.sample()));
  assert(spaces::Text(4).contains(Value("ok")));
  assert(MultiBinary({2}).shape() == std::vector<int64_t>{2});
  assert(MultiDiscrete({2, 3}).nvec() == (std::vector<int64_t>{2, 3}));
  assert(Tuple({Discrete(2), spaces::Text(4)}).spaces().size() == 2);
}
void runs_sessions() {
  int closed = 0;
  {
    Model first([](const Value&) { return Value(0); },
                rlmesh::ModelOptions{}.with_on_close([&] { ++closed; }));
    {
      auto copy = first;
      auto moved = std::move(copy);
      assert(closed == 0);
    }
    assert(closed == 0);
  }
  assert(closed == 1);

  auto object = std::make_shared<Counter>();
  Env env(object);
  int ends = 0;
  Model model(Greedy{&ends});
  Model copy = model;
  Model moved = std::move(copy);
  Hooks hooks;
  auto result =
      model.run(env, rlmesh::RunOptions{}.with_episodes(2).with_seeds({0, 1}).with_hooks(&hooks));
  assert(result.num_episodes() == 2 && result.total_steps() == 4 && result.mean_reward() == 2 &&
         result.success_rate() == 1);
  assert(hooks.starts == 1 && hooks.steps == 4 && hooks.ends == 1);
  assert(object->closes == 0 && ends == 2);
  assert(moved.run(env).num_episodes() == 1);
  assert(model.run(env, rlmesh::RunOptions{}.with_seeds({4, 5})).num_episodes() == 2);
  assert(model.run(env, rlmesh::RunOptions{}.with_seeds(std::vector<int64_t>{})).num_episodes() ==
         0);
  try {
    model.run(env, rlmesh::RunOptions{}.with_episodes(2).with_seeds({0}));
    assert(false);
  } catch (const ValueError&) {
  }
  {
    auto s = rlmesh::session(model, env);
    auto [obs, info] = s.reset(0);
    assert(info.empty());
    auto [obs2, reward, terminated, truncated, info2] = s.step(s.predict(obs));
    (void)obs2;
    (void)info2;
    assert(reward == 1 && !terminated && !truncated);
    assert(s.contract().id() == "CppCounter-v0");
    assert(s.num_envs() == 1);
    auto token_copy = [&] {
      auto token = s.cancel_handle();
      auto copy = token;
      assert(copy.raw() != token.raw());
      return copy;
    }();  // The copied C handle survives the original handle's destruction.
    token_copy.cancel();
    token_copy.cancel();  // Pending fires coalesce.
    try {
      s.predict(obs);
      assert(false);
    } catch (const ext::CancelledError&) {
    }
    assert(s.predict(obs) == Value(0));
    ext::CancelToken assigned;
    assigned = token_copy;
    assert(assigned.raw() != token_copy.raw());
    auto* raw = assigned.raw();
    auto moved = std::move(assigned);
    assert(!assigned.raw() && moved.raw() == raw);
    assigned = std::move(moved);
    assert(!moved.raw() && assigned.raw() == raw);
    assigned.cancel();  // The cloned token observes the re-armed shared state.
    try {
      s.predict(obs);
      assert(false);
    } catch (const ext::CancelledError&) {
    }
    assert(s.predict(obs) == Value(0));
    try {
      moved.cancel();
      assert(false);
    } catch (const ValueError&) {
    }
    struct RefuseFree : RunHooks {
      Session& session;
      bool called = false;
      explicit RefuseFree(Session& s) : session(s) {}
      void on_step(const StepEvent&) override {
        rlmesh_session_free(session.raw());
        assert(rlmesh_last_error_status() == RLMESH_ERR_RUNTIME);
        called = true;
      }
    } refuse(s);
    assert(s.run(rlmesh::SessionRunOptions{}.with_hooks(&refuse)).num_episodes() == 1);
    assert(refuse.called && s.done());
  }
  assert(object->closes == 0);
  moved.close();
  try {
    model.run(env);
    assert(false);
  } catch (const RuntimeError&) {
  }
  env.close();
  assert(object->closes == 1);
  auto loaded = Model::from_config<Loaded>({{"weight", 3}});
  Env another(Counter{});
  auto s = loaded.session(another);
  auto [obs, info] = s.reset();
  (void)info;
  assert(s.predict(obs).as_int() == 3);
}
void missing_dict_action_keys() {
  struct DictCounter : Counter {
    Space action_space() const { return Dict({{"nested", Dict({{"choice", Discrete(3)}})}}); }
  };
  EnvServer server(DictCounter{}, rlmesh::EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
  server.start();
  RemoteEnv remote(rlmesh::RemoteEnvOptions{}.with_address(server.address()));
  remote.reset();
  Value missing(Metadata{{"nested", Value(Metadata{})}});
  try {
    remote.step(missing);
    assert(false);
  } catch (const ValueError& e) {
    assert(std::string(e.what()) == "$.nested: missing key 'choice'");
  }
  // The C entrypoint preserves the same class and text before any I/O.
  RLMeshValue *obs = nullptr, *info = nullptr;
  double reward = 0;
  bool terminated = false, truncated = false;
  assert(rlmesh_remote_env_step(remote.raw(), missing.raw(), &obs, &reward, &terminated, &truncated,
                                &info) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(rlmesh_last_error_class() == RLMESH_ERROR_CLASS_VALUE_ERROR);
  assert(std::string(rlmesh_last_error_message()) == "$.nested: missing key 'choice'");
  assert(!obs && !info);
  remote.close();
  server.shutdown();
}
void session_target_lifetimes() {
  Env env(Counter{});
  int closed = 0;
  auto make_model = [&] {
    return Model([](const Value&) { return 0; },
                 rlmesh::ModelOptions{}.with_on_close([&] { ++closed; }));
  };
  auto make_server = [] {
    EnvServer server(Counter{}, rlmesh::EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
    server.start();
    return server;
  };
  {
    auto s = make_model().session(env);
    assert(closed == 0);
    auto [obs, info] = s.reset();
    (void)info;
    assert(s.predict(obs) == Value(0));
    auto moved = std::move(s);
    assert(moved.step(0) == StepResult(1, 1.0, false, false, {{"is_success", true}}));
  }
  assert(closed == 1);
  {
    auto s = rlmesh::session(make_model(), make_server());
    auto [obs, info] = s.reset();
    (void)info;
    assert(s.predict(obs) == Value(0));
    assert(std::get<1>(s.step(0)) == 1);
    auto other = rlmesh::session(make_model(), make_server());
    s = std::move(other);
    assert(closed == 2);
    assert(s.run().total_steps() == 2);
  }
  assert(closed == 3);
  Model served([](const Value&) { return 0; });
  const auto address = [] {
    // Reserve an OS-selected TCP port, then release it for the model server.
    EnvServer reservation(Counter{}, rlmesh::EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
    return reservation.address();
  }();
  ext::CancelToken stop;
  auto server = std::async(std::launch::async, [&] {
    try {
      served.serve(address, rlmesh::ModelServeOptions{}.with_cancel(stop));
    } catch (const ext::CancelledError&) {
    }
  });
  try {
    if (server.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      server.get();
      throw RuntimeError("model server stopped before startup");
    }
    {
      auto s = rlmesh::session(RemoteModel(rlmesh::RemoteModelOptions{}.with_address(address)),
                               make_server());
      auto [obs, info] = s.reset();
      (void)info;
      assert(s.predict(obs) == Value(0));
      assert(std::get<1>(s.step(0)) == 1);
    }
    {
      auto s =
          RemoteModel(rlmesh::RemoteModelOptions{}.with_address(address)).session(make_server());
      assert(s.run().total_steps() == 2);
    }
  } catch (...) {
    stop.cancel();
    if (server.valid()) server.get();
    throw;
  }
  stop.cancel();
  server.get();
}
void exceptions() {
  Model policy([](const Value&) -> Value { throw OwnError(17); });
  Env env(Counter{});
  {
    auto s = policy.session(env);
    auto [obs, info] = s.reset();
    (void)info;
    try {
      s.predict(obs);
      assert(false);
    } catch (const OwnError& e) {
      assert(e.tag == 17);
    }
  }
  Model unknown([](const Value&) -> Value { throw 42; });
  {
    auto s = unknown.session(env);
    auto [obs, info] = s.reset();
    (void)info;
    try {
      s.predict(obs);
      assert(false);
    } catch (int n) {
      assert(n == 42);
    }
  }
  struct FailHook : RunHooks {
    void on_step(const StepEvent&) override { throw OwnError(23); }
  } hook;
  Model model(Greedy{});
  try {
    model.run(env, rlmesh::RunOptions{}.with_hooks(&hook));
    assert(false);
  } catch (const OwnError& e) {
    assert(e.tag == 23);
  }
  set_warning_handler([](const Warning&) { throw OwnError(29); });
  try {
    model.run(env, rlmesh::RunOptions{}.with_execution_horizon(2));
    assert(false);
  } catch (const OwnError& e) {
    assert(e.tag == 29);
  }
  set_warning_handler({});
  // One shared model, two calling threads: the exception follows the call.
  Model threaded([](const Value&, const PredictContext& c) -> Value {
    throw OwnError(static_cast<int>(*c.episode_seed));
  });
  auto work = [&](int n) {
    Env local(Counter{});
    auto s = threaded.session(local);
    auto [o, i] = s.reset(n);
    (void)i;
    try {
      s.predict(o);
      assert(false);
    } catch (const OwnError& e) {
      assert(e.tag == n);
    }
  };
  auto a = std::async(std::launch::async, work, 31);
  auto b = std::async(std::launch::async, work, 47);
  a.get();
  b.get();
}
void foreign_error_payloads() {
  static int freed = 0;
  static const unsigned char foreign_tag = 0;
  RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
  vt.predict = [](void*, const RLMeshValue*, const RLMeshPredictContext*, RLMeshValue**) {
    return RLMESH_OK;
  };
  vt.load = [](void* user, const RLMeshValue*) {
    auto kind = *static_cast<int*>(user);
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_VALUE_ERROR, "foreign payload");
    if (kind == 2) {
      auto* p = new rlmesh::detail::ExceptionPayload(std::make_exception_ptr(OwnError(99)));
      p->magic = 0;
      rlmesh_ext_callback_set_tagged_error_payload(
          p,
          [](void* value) {
            ++freed;
            delete static_cast<rlmesh::detail::ExceptionPayload*>(value);
          },
          &rlmesh::detail::exception_payload_tag);
    } else {
      auto* p = new unsigned char(7);  // Too small to inspect as a C++ envelope.
      auto release = [](void* value) {
        ++freed;
        delete static_cast<unsigned char*>(value);
        rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_RUNTIME_ERROR, "deleter reentered");
      };
      if (kind == 1)
        rlmesh_ext_callback_set_tagged_error_payload(p, release, &foreign_tag);
      else
        rlmesh_ext_callback_set_error_payload(p, release);
    }
    return RLMESH_ERR_INVALID_ARGUMENT;
  };
  for (int kind = 0; kind < 3; ++kind) {
    RLMeshModel* model = nullptr;
    auto status = rlmesh_model_new(&vt, &kind, nullptr, &model);
    assert(status == RLMESH_ERR_INVALID_ARGUMENT && !model);
    try {
      rlmesh::detail::check_status(status);
      assert(false);
    } catch (const ValueError& error) {
      assert(std::string(error.what()) == "foreign payload");
    }
    assert(freed == kind + 1);
  }
}
struct BatchCounter {
  std::optional<Metadata> metadata() const { return Metadata{{"autoreset_mode", "NextStep"}}; }
  uint64_t num_envs() const { return 2; }
  Space single_observation_space() const { return Discrete(10); }
  Space single_action_space() const { return Discrete(10); }
  Value observations() const {
    const int64_t values[] = {0, 0};
    return Tensor(values, sizeof(values), {2}, DType::Int64);
  }
  ResetResult reset(std::optional<VectorSeed>, const std::optional<InfoDict>&) {
    return {observations(), {}};
  }
  VectorStepResult step(std::optional<Value>) {
    return {observations(), {1, 2}, {true, true}, {false, false}, {}};
  }
};
struct ChunkPolicy {
  Value predict(const Value&) { return 0; }
  static constexpr std::optional<uint32_t> native_chunk = 2;
  int* calls;
  Value predict_chunk(const Value&, uint32_t horizon, const PredictContext& context) {
    ++*calls;
    auto& seen = context.state.get_or_emplace<int>("seen", 0);
    assert(seen == static_cast<int>(context.predict_index));
    ++seen;
    assert(context.state.get<int>("seen") && !context.state.get<double>("seen"));
    return Value::list(std::vector<Value>(horizon, 0));
  }
};
void vector_lane_count_is_fixed() {
  struct ChangingBatch : BatchCounter {
    uint64_t count = 2;
    bool grow_in_step = false;
    uint64_t num_envs() const { return count; }
    VectorStepResult step(std::optional<Value>) {
      if (grow_in_step) count = 3;
      return {observations(),
              std::vector<double>(count, 1),
              std::vector<bool>(count, true),
              std::vector<bool>(count, false),
              {}};
    }
  };
  for (bool during_step : {false, true}) {
    auto object = std::make_shared<ChangingBatch>();
    VectorEnv env(object);
    if (during_step)
      object->grow_in_step = true;
    else
      object->count = 3;
    Model model([](const Value&) { return BatchCounter{}.observations(); });
    bool rejected = false;
    try {
      model.run(env);
    } catch (const EnvironmentException& error) {
      assert(std::string(error.what()).find("num_envs changed") != std::string::npos);
      rejected = true;
    }
    assert(rejected);
  }
}
void policy_context_precedence() {
  struct BothForms {
    int predicts = 0, calls = 0;
    Value predict(const Value&) {
      ++predicts;
      return 1;
    }
    Value operator()(const Value&, const PredictContext&) {
      ++calls;
      return 2;
    }
  };
  rlmesh::detail::PolicyTable<BothForms> table;
  assert(table.table.context_corners == 0);
  auto policy = std::make_shared<BothForms>();
  Model model(policy);
  Env env(Counter{});
  auto session = model.session(env);
  auto [obs, info] = session.reset();
  (void)info;
  assert(session.predict(obs) == Value(1));
  assert(session.predict(obs) == Value(1));
  assert(policy->predicts == 2 && policy->calls == 0);
  struct CallableOnly {
    Value operator()(const Value&, const PredictContext&) { return 0; }
  };
  rlmesh::detail::PolicyTable<CallableOnly> callable;
  assert(callable.table.context_corners == RLMESH_CORNER_PREDICT);
}
void vector_and_chunks() {
  int calls = 0;
  Env env(Counter{});
  Model chunk(ChunkPolicy{&calls});
  assert(chunk.run(env, rlmesh::RunOptions{}.with_execution_horizon(2)).total_steps() == 2);
  assert(calls == 1);
  auto batch_policy = [](const Value&) { return BatchCounter{}.observations(); };
  Model scalar(batch_policy);
  VectorEnv vector(BatchCounter{});
  auto result = scalar.run(vector, rlmesh::RunOptions{}.with_episodes(2));
  assert(result.num_episodes() == 2);
  struct BatchPolicy {
    std::vector<Value> predict_batch(rlmesh::Span<const Value> observations,
                                     BatchPredictContext contexts) {
      assert(observations.size() == contexts.size());
      std::vector<Value> out;
      for (auto& context : contexts) {
        assert(context.predict_index == 0);
        context.state.insert_or_assign("batch", true);
        out.emplace_back(0);
      }
      return out;
    }
  };
  Model model(BatchPolicy{});
  EnvServer server(VectorEnv(BatchCounter{}),
                   rlmesh::EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
  server.start();
  {
    RemoteVectorEnv remote(rlmesh::RemoteEnvOptions{}.with_address(server.address()));
    assert(remote.num_envs() == 2);
    auto [obs, info] = remote.reset(VectorSeed(std::vector<int64_t>{0, 1}));
    (void)obs;
    (void)info;
    auto [next, rewards, terminated, truncated, step_info] =
        remote.step(BatchCounter{}.observations());
    (void)next;
    (void)step_info;
    assert(rewards.size() == 2 && terminated[0] && !truncated[1]);
    remote.close();
  }
  assert(model.run(server, rlmesh::RunOptions{}.with_episodes(2)).num_episodes() == 2);
  server.shutdown();
}
void factories_remote_describe() {
  struct LifecyclePolicy {
    int loads = 0, closes = 0, predictions = 0;
    void load() { ++loads; }
    void close() { ++closes; }
    Value predict(const Value&) {
      ++predictions;
      return 0;
    }
  } policy;
  assert(!describe(policy).empty());
  assert(policy.loads == 0 && policy.closes == 0 && policy.predictions == 0);
  Env description_env(Counter{});
  ServedEnv borrowed(description_env);
  assert(!describe_json(policy, rlmesh::DescribeJsonOptions{}.with_served_env(borrowed)).empty());
  std::vector<Env> lanes;
  lanes.emplace_back(Counter{});
  rlmesh::Span<const Env> span(lanes);
  ServedEnv borrowed_span(span);
  assert(
      !describe_json(policy, rlmesh::DescribeJsonOptions{}.with_served_env(borrowed_span)).empty());
  assert(!describe_json(policy).empty());
  assert(policy.loads == 0 && policy.closes == 0 && policy.predictions == 0);
  Factory factory;
  auto made = factory.make_env();
  assert(factory.builds == 1);
  Model model(Greedy{});
  assert(model.run(factory).num_episodes() == 1);
  auto description = describe<Greedy>();
  assert(!description.empty());
  assert(!describe(factory).empty());
  assert(!describe_json<Greedy>().empty());
  assert(!describe_json(factory).empty());
  auto v = [&] {
    Variant value{};
    value.id = "x";
    value.params = {{"n", 2}};
    value.metadata = {{"label", "variant"}};
    return value;
  }();
  assert(to_string(v).find("Variant") != std::string::npos);
  EnvServer server(
      Counter{},
      rlmesh::EnvServerOptions{}
          .with_address("tcp://127.0.0.1:0")
          .with_options(ServeOptions{}.with_allow_remote_shutdown(true).with_workflow_edition(
              std::string(current_workflow_edition()))));
  server.start();
  {
    RemoteEnv remote(rlmesh::RemoteEnvOptions{}.with_address(server.address()));
    assert(!remote.env_id().empty());
    assert(remote.env_contract().id() == "CppCounter-v0");
    assert(model.run(remote).total_steps() == 2);
    auto [o, i] = remote.reset(0);
    (void)i;
    auto [next, r, t, tr, info] = remote.step(0);
    (void)o;
    (void)next;
    (void)info;
    assert(r == 1 && !t && !tr);
    remote.close();
  }
  assert(!ext::env_server_describe_json(server).empty());
  assert(!server.wait(Duration(0)));
  server.shutdown();
  assert(server.wait(Duration(1)));
  RemoteModel rm(rlmesh::RemoteModelOptions{}.with_address("tcp://127.0.0.1:1"));
  auto rm_copy = rm;
  assert(rm_copy.address() == rm.address());
}
struct Identity : adapters::AdapterBase {
  Value transform_obs(const Value& v) override { return v; }
  Value transform_action(const Value& v) override { return v; }
};
void adapter_axis_presence() {
  using namespace adapters;
  State state{};
  state.role = JOINT_POS;
  state.dim = 1;
  Actuator actuator{};
  actuator.role = ACTION_JOINT_POS;
  actuator.dim = 1;
  ModelSpec absent(state, [&] {
    Action value{};
    value.components = {actuator};
    return value;
  }());
  const auto* input = static_cast<const RLMeshState*>(rlmesh_model_spec_input(absent.raw())->leaf);
  assert(!input->has_scale && !input->axis_scale && !input->has_offset && !input->axis_offset);
  const auto* output = rlmesh_model_spec_output(absent.raw())->components[0];
  assert(!output->has_scale && !output->axis_scale && !output->has_offset && !output->axis_offset);
  for (bool offset : {false, true}) {
    auto empty_state = state;
    (offset ? empty_state.offset : empty_state.scale) = std::vector<double>{};
    bool rejected = false;
    try {
      ModelSpec invalid(empty_state, [&] {
        Action value{};
        value.components = {actuator};
        return value;
      }());
    } catch (const ValueError&) {
      rejected = true;
    }
    assert(rejected);
    auto empty_actuator = actuator;
    (offset ? empty_actuator.offset : empty_actuator.scale) = std::vector<double>{};
    rejected = false;
    try {
      ModelSpec invalid(state, [&] {
        Action value{};
        value.components = {empty_actuator};
        return value;
      }());
    } catch (const ValueError&) {
      rejected = true;
    }
    assert(rejected);
  }
  state.scale = std::vector<double>{2};
  state.offset = std::vector<double>{1};
  actuator.scale = std::vector<double>{2};
  actuator.offset = std::vector<double>{1};
  ModelSpec present(state, [&] {
    Action value{};
    value.components = {actuator};
    return value;
  }());
  assert(present == ModelSpec(
                        [&] {
                          State value{};
                          value.role = JOINT_POS;
                          value.dim = 1;
                          value.scale = std::vector<double>{2};
                          value.offset = std::vector<double>{1};
                          return value;
                        }(),
                        [&] {
                          Action value{};
                          value.components = {actuator};
                          return value;
                        }()));
}
void adapter_callable_identity() {
  using namespace adapters;
  ObsTransform observation([](const Value& value) { return value; });
  RotationTransform rotation([](const Value& value) { return value; });
  Custom input{};
  input.transform = observation;
  CustomEncoding encoding{};
  encoding.base = RotationEncoding::EulerXyz;
  encoding.from_base = rotation;
  encoding.to_base = rotation;
  auto action = [&] {
    Action value{};
    value.components = {[&] {
      rlmesh::adapters::Actuator value{};
      value.role = ACTION_JOINT_POS;
      value.dim = 3;
      value.encoding = encoding;
      return value;
    }()};
    return value;
  }();
  detail_adapter::Arena arena;
  auto custom_action = action.components[0].raw(arena);
  assert(!custom_action.has_encoding && custom_action.custom_encoding);
  auto native_action =
      [&] {
        Actuator value{};
        value.role = ACTION_JOINT_POS;
        value.dim = 3;
        value.encoding = RotationEncoding::EulerXyz;
        return value;
      }()
          .raw(arena);
  assert(native_action.has_encoding && !native_action.custom_encoding);
  ModelSpec spec(input, action);
  Model model([](const Value&) { return 0; }, rlmesh::ModelOptions{}.with_spec(spec));
  const auto& reconstructed = std::get<ModelSpec>(*model.spec());
  auto raw_input = reconstructed.input().raw(arena);
  auto clone = Custom::from_raw(*static_cast<const RLMeshCustom*>(raw_input.leaf));
  assert(clone.transform->raw() != observation.raw());
  assert(clone == input);
  assert(std::unordered_set<Custom>{input}.count(clone));
  assert(reconstructed.input() == spec.input());
  assert(reconstructed.output() == action);
  const auto& cloned_encoding =
      std::get<CustomEncoding>(*reconstructed.output().components[0].encoding);
  assert(std::get<RotationTransform>(*cloned_encoding.from_base).raw() != rotation.raw());
  assert(cloned_encoding == encoding);
  assert(std::unordered_set<CustomEncoding>{encoding}.count(cloned_encoding));
  assert(ObsTransform([](const Value& value) { return value; }) != observation);
  assert(RotationTransform([](const Value& value) { return value; }) != rotation);
}
void adapter_dict_equality() {
  using namespace adapters;
  auto check = [](auto a, auto b) {
    using Node = decltype(a);
    auto first = Node::dict({{"a", a}, {"b", b}});
    auto reversed = Node::dict({{"b", b}, {"a", a}});
    assert(first == reversed && reversed == first);
    assert(Node::dict({{"outer", first}}) == Node::dict({{"outer", reversed}}));
    assert(first != Node::dict({{"a", b}, {"b", a}}));
    assert(first != Node::dict({{"a", a}, {"c", b}}));
    assert(first != Node::dict({{"a", a}}));
    assert(Node::tuple({a, b}) != Node::tuple({b, a}));
    assert(Node::dict({}) != Node::tuple({}));
  };
  check(InputNode([&] {
          adapters::Text value{};
          value.role = "a";
          return value;
        }()),
        InputNode([&] {
          adapters::Text value{};
          value.role = "b";
          return value;
        }()));
  check(ObsNode([&] {
          TextTag value{};
          value.role = "a";
          return value;
        }()),
        ObsNode([&] {
          TextTag value{};
          value.role = "b";
          return value;
        }()));
}
void adapter_hashes() {
  using namespace adapters;
  adapters::Text text{};
  text.role = "instruction";
  text.container = TextContainer::Str;
  text.fill = "";
  assert(std::unordered_set<adapters::Text>{text}.count(adapters::Text(text)));
  assert(!std::unordered_set<adapters::Text>{text}.count([&] {
    adapters::Text value{};
    value.role = "instruction";
    return value;
  }()));
  ImageTag image{};
  image.role = IMAGE_PRIMARY;
  image.layout = ImageLayout::Chw;
  image.upside_down = true;
  image.part = "left";
  assert(std::unordered_set<ImageTag>{image}.count(ImageTag(image)));
  assert(!std::unordered_set<ImageTag>{image}.count([&] {
    ImageTag value{};
    value.role = IMAGE_PRIMARY;
    return value;
  }()));
  assert(std::unordered_set<TextTag>{[&] {
           rlmesh::adapters::TextTag value{};
           value.role = "instruction";
           return value;
         }()}
             .count([&] {
               TextTag value{};
               value.role = "instruction";
               return value;
             }()));
  Custom entrypoint{};
  entrypoint.entrypoint = "example:transform";
  assert(std::unordered_set<Custom>{entrypoint}.count(Custom(entrypoint)));
  CustomEncoding encoding{};
  encoding.base = RotationEncoding::EulerXyz;
  encoding.from_base = "example:from";
  encoding.to_base = "example:to";
  assert(std::unordered_set<CustomEncoding>{encoding}.count(CustomEncoding(encoding)));
}
Value vector(float f) { return Tensor(&f, sizeof(f), {1}, DType::Float32); }
struct Tagged {
  std::optional<std::string> render_mode() const { return "rgb_array"; }
  Space observation_space() const { return Box(-10, 10, {1}); }
  Space action_space() const { return Box(-10, 10, {1}); }
  ResetResult reset(std::optional<int64_t>, const std::optional<InfoDict>&) {
    return {vector(2), {}};
  }
  StepResult step(std::optional<Value>) { return {vector(3), 1, true, false, {}}; }
  std::optional<Value> render() { return Value(1); }
};
void adapters_readers_recorder() {
  using namespace adapters;
  EnvTags tags(
      [&] {
        StateTag value{};
        value.role = JOINT_POS;
        return value;
      }(),
      [&] {
        Action value{};
        value.components = {[&] {
          rlmesh::adapters::Actuator value{};
          value.role = ACTION_JOINT_POS;
          value.dim = 1;
          return value;
        }()};
        return value;
      }());
  ModelSpec spec(
      [&] {
        State value{};
        value.role = JOINT_POS;
        value.dim = 1;
        return value;
      }(),
      [&] {
        Action value{};
        value.components = {[&] {
          rlmesh::adapters::Actuator value{};
          value.role = ACTION_JOINT_POS;
          value.dim = 1;
          return value;
        }()};
        return value;
      }());
  assert(to_string([&] {
           TextTag value{};
           value.role = "instruction";
           return value;
         }()) == "TextTag(role='instruction')");
  assert(to_string(Constant{}) == "Constant(dim=1, fill=0.0)");
  assert(to_string([&] {
           Image value{};
           value.role = IMAGE_PRIMARY;
           return value;
         }()
                       .with_stride(2))
             .find("offsets=(0,)") != std::string::npos);
  assert(std::hash<Advisory>{}({AdvisorySeverity::Info, "hello"}) ==
         std::hash<Advisory>{}({AdvisorySeverity::Info, "hello"}));
  assert(ModelSpec::from_dict(spec.to_dict()) == spec);
  assert(ModelSpec::from_metadata(spec.to_metadata()) == spec);
  assert(EnvTags::from_json(tags.to_json()) == tags);
  assert(tags.observation_roles().states.size() == 1);
  Box space(-10, 10, {1});
  ModelSpec missing(
      [&] {
        Custom value{};
        value.entrypoint = "cpp:unregistered:input";
        return value;
      }(),
      spec.output());
  try {
    resolve(tags, space.spec(), space.spec(), missing,
            rlmesh::adapters::ResolveOptions{}.with_trust_entrypoints(true));
    assert(false);
  } catch (const AdapterResolutionError& e) {
    assert(std::string(e.what()).find("unregistered custom input entrypoint") != std::string::npos);
  }
  auto adapter = resolve(tags, space.spec(), space.spec(), spec);
  auto wrapped = adapter.wrap_predict([](const Value& v) { return v; });
  assert(wrapped(vector(2)).as_tensor().size() == 1);
  auto identity = std::make_shared<Identity>();
  auto f = identity->wrap_predict([](const Value& v) { return v; });
  identity.reset();
  assert(f(3).as_int() == 3);
  Adapter custom(std::make_shared<Identity>());
  auto kept = static_cast<AdapterBase&>(custom).wrap_predict([](const Value& v) { return v; });
  assert(kept(4).as_int() == 4);
  assert(Tolerant<Frame>::parse("world").known() == Frame::World);
  assert(!Tolerant<Frame>::parse("future_frame").known());
  Env env(Tagged{});
  tag(env, tags);
  Model model([](const Value&) { return vector(0); }, rlmesh::ModelOptions{}.with_spec(spec));
  auto s = model.session(env);
  auto [o, i] = s.reset(0);
  (void)i;
  auto reader = s.reader({JOINT_POS});
  assert(reader(o).contains_key(JOINT_POS));
  assert(s.read(o, JOINT_POS).as_tensor().size() == 1);
  assert(s.observation_roles().states.size() == 1);
  auto sources = s.frame_sources();
  assert(sources.render && sources.render().as_int() == 1);
  Recorder recorder(rlmesh::RecorderOptions{}.with_result_set_id("cpp-parity"));
  auto capture = recorder.capture("policy", "env",
                                  rlmesh::RecorderCaptureOptions{}.with_session(s.run_context()));
  auto result = s.run(rlmesh::SessionRunOptions{}.with_hooks(capture.get()));
  assert(result.num_episodes() == 1);
  assert(recorder.workloads().size() == 1);
  auto record = recorder.add(result, "policy", "env");
  assert(record.total_steps() == 1);
  assert(record.to_dict().contains_key("episodes"));
  assert(recorder.result_set_id() == "cpp-parity");
  assert(recorder.to_dict(rlmesh::RecorderToDictOptions{}.with_recorded_at("2026-01-01T00:00:00Z"))
             .contains_key("workloads"));
  recorder.close();
  View view = View::parse("terminal");
  view.validate();
  assert(view.backend == ViewBackend::Terminal);
  try {
    [&] {
      View value{};
      value.fps = 0;
      return value;
    }()
        .validate();
    assert(false);
  } catch (const ValueError&) {
  }
}
}  // namespace
int main() try {
  values_spaces();
  vector_and_chunks();
  vector_lane_count_is_fixed();
  policy_context_precedence();
  runs_sessions();
  session_target_lifetimes();
  missing_dict_action_keys();
  exceptions();
  foreign_error_payloads();
  factories_remote_describe();
  adapter_axis_presence();
  adapter_callable_identity();
  adapter_dict_equality();
  adapter_hashes();
  adapters_readers_recorder();
  std::cout << "cpp_parity OK\n";
} catch (const std::exception& e) {
  std::cerr << e.what() << "\n";
  return 1;
}
