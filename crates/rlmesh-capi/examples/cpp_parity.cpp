// Executable ABI 5 C++20 parity fixtures. Built with GCC and Clang by test:cxx.
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <rlmesh.hpp>
#include <thread>
#include <type_traits>
using namespace rlmesh;
using namespace rlmesh::spaces;
namespace {
static_assert(std::random_access_iterator<Dict::key_iterator>);
static_assert(!std::is_copy_constructible_v<Env>);
static_assert(!std::is_copy_constructible_v<Session>);
static_assert(!std::is_copy_constructible_v<RemoteEnv>);
static_assert(std::is_copy_constructible_v<Model>);
static_assert(std::is_copy_constructible_v<RemoteModel>);
static_assert(std::is_copy_constructible_v<ext::CancelToken>);
static_assert(std::is_nothrow_move_constructible_v<ext::CancelToken>);
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
    return std::vector<Variant>{{.id = "default", .params = {}}};
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
  auto writable = t.to_dlpack({.copy = true});
  assert(!(writable->flags & DLPACK_FLAG_BITMASK_READ_ONLY));
  Dict dict({{"p", Box(-1, 1, {2})}, {"id", Discrete(3)}});
  assert(dict.end() - dict.begin() == 2);
  assert(dict.begin()[0] == "id");
  auto child = dict["id"];
  child.seed(42);
  auto first = child.sample();
  dict["id"].seed(42);
  assert(first == child.sample());
  assert(dict.contains(dict.sample()));
  assert(Text(4).contains(Value("ok")));
  assert(MultiBinary({2}).shape() == std::vector<int64_t>{2});
  assert(MultiDiscrete({2, 3}).nvec() == (std::vector<int64_t>{2, 3}));
  assert(Tuple({Discrete(2), Text(4)}).spaces().size() == 2);
}
void runs_sessions() {
  int closed = 0;
  {
    Model first([](const Value&) { return Value(0); }, {.on_close = [&] { ++closed; }});
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
  auto result = model.run(env, {.episodes = 2, .seeds = {0, 1}, .hooks = &hooks});
  assert(result.num_episodes() == 2 && result.total_steps() == 4 && result.mean_reward() == 2 &&
         result.success_rate() == 1);
  assert(hooks.starts == 1 && hooks.steps == 4 && hooks.ends == 1);
  assert(object->closes == 0 && ends == 2);
  assert(moved.run(env).num_episodes() == 1);
  assert(model.run(env, {.seeds = {4, 5}}).num_episodes() == 2);
  assert(model.run(env, {.seeds = std::vector<int64_t>{}}).num_episodes() == 0);
  try {
    model.run(env, {.episodes = 2, .seeds = {0}});
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
    assert(s.run({.hooks = &refuse}).num_episodes() == 1);
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
    model.run(env, {.hooks = &hook});
    assert(false);
  } catch (const OwnError& e) {
    assert(e.tag == 23);
  }
  set_warning_handler([](const Warning&) { throw OwnError(29); });
  try {
    model.run(env, {.execution_horizon = 2});
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
void vector_and_chunks() {
  int calls = 0;
  Env env(Counter{});
  Model chunk(ChunkPolicy{&calls});
  assert(chunk.run(env, {.execution_horizon = 2}).total_steps() == 2);
  assert(calls == 1);
  auto batch_policy = [](const Value&) { return BatchCounter{}.observations(); };
  Model scalar(batch_policy);
  VectorEnv vector(BatchCounter{});
  auto result = scalar.run(vector, {.episodes = 2});
  assert(result.num_episodes() == 2);
  struct BatchPolicy {
    std::vector<Value> predict_batch(std::span<const Value> observations,
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
  EnvServer server(VectorEnv(BatchCounter{}), {.address = "tcp://127.0.0.1:0"});
  server.start();
  {
    RemoteVectorEnv remote({.address = server.address()});
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
  assert(model.run(server, {.episodes = 2}).num_episodes() == 2);
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
  Variant v{.id = "x", .params = {{"n", 2}}, .metadata = {{"label", "variant"}}};
  assert(to_string(v).find("Variant") != std::string::npos);
  EnvServer server(Counter{}, {.address = "tcp://127.0.0.1:0",
                               .options = ServeOptions{
                                   .allow_remote_shutdown = true,
                                   .workflow_edition = std::string(current_workflow_edition())}});
  server.start();
  {
    RemoteEnv remote({.address = server.address()});
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
  RemoteModel rm({.address = "tcp://127.0.0.1:1"});
  auto rm_copy = rm;
  assert(rm_copy.address() == rm.address());
}
struct Identity : adapters::AdapterBase {
  Value transform_obs(const Value& v) override { return v; }
  Value transform_action(const Value& v) override { return v; }
};
void adapter_axis_presence() {
  using namespace adapters;
  State state{.role = JOINT_POS, .dim = 1};
  Actuator actuator{.role = ACTION_JOINT_POS, .dim = 1};
  ModelSpec absent(state, Action{.components = {actuator}});
  const auto* input = static_cast<const RLMeshState*>(rlmesh_model_spec_input(absent.raw())->leaf);
  assert(!input->has_scale && !input->axis_scale && !input->has_offset && !input->axis_offset);
  const auto* output = rlmesh_model_spec_output(absent.raw())->components[0];
  assert(!output->has_scale && !output->axis_scale && !output->has_offset && !output->axis_offset);
  for (bool offset : {false, true}) {
    auto empty_state = state;
    (offset ? empty_state.offset : empty_state.scale) = std::vector<double>{};
    bool rejected = false;
    try {
      ModelSpec invalid(empty_state, Action{.components = {actuator}});
    } catch (const ValueError&) {
      rejected = true;
    }
    assert(rejected);
    auto empty_actuator = actuator;
    (offset ? empty_actuator.offset : empty_actuator.scale) = std::vector<double>{};
    rejected = false;
    try {
      ModelSpec invalid(state, Action{.components = {empty_actuator}});
    } catch (const ValueError&) {
      rejected = true;
    }
    assert(rejected);
  }
  state.scale = std::vector<double>{2};
  state.offset = std::vector<double>{1};
  actuator.scale = std::vector<double>{2};
  actuator.offset = std::vector<double>{1};
  ModelSpec present(state, Action{.components = {actuator}});
  assert(present == ModelSpec(State{.role = JOINT_POS, .dim = 1,
                                   .scale = std::vector<double>{2},
                                   .offset = std::vector<double>{1}},
                              Action{.components = {actuator}}));
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
  EnvTags tags(StateTag{.role = JOINT_POS},
               Action{.components = {{.role = ACTION_JOINT_POS, .dim = 1}}});
  ModelSpec spec(State{.role = JOINT_POS, .dim = 1},
                 Action{.components = {{.role = ACTION_JOINT_POS, .dim = 1}}});
  assert(to_string(TextTag{.role = "instruction"}) == "TextTag(role='instruction')");
  assert(to_string(Constant{}) == "Constant(dim=1, fill=0.0)");
  assert(to_string(Image{.role = IMAGE_PRIMARY}.with_stride(2)).find("offsets=(0,)") !=
         std::string::npos);
  assert(std::hash<Advisory>{}({AdvisorySeverity::Info, "hello"}) ==
         std::hash<Advisory>{}({AdvisorySeverity::Info, "hello"}));
  assert(ModelSpec::from_dict(spec.to_dict()) == spec);
  assert(ModelSpec::from_metadata(spec.to_metadata()) == spec);
  assert(EnvTags::from_json(tags.to_json()) == tags);
  assert(tags.observation_roles().states.size() == 1);
  Box space(-10, 10, {1});
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
  Model model([](const Value&) { return vector(0); }, {.spec = spec});
  auto s = model.session(env);
  auto [o, i] = s.reset(0);
  (void)i;
  auto reader = s.reader({JOINT_POS});
  assert(reader(o).contains_key(JOINT_POS));
  assert(s.read(o, JOINT_POS).as_tensor().size() == 1);
  assert(s.observation_roles().states.size() == 1);
  auto sources = s.frame_sources();
  assert(sources.render && sources.render().as_int() == 1);
  Recorder recorder({.result_set_id = "cpp-parity"});
  auto capture = recorder.capture("policy", "env", {.session = s.run_context()});
  auto result = s.run({.hooks = capture.get()});
  assert(result.num_episodes() == 1);
  assert(recorder.workloads().size() == 1);
  auto record = recorder.add(result, "policy", "env");
  assert(record.total_steps() == 1);
  assert(record.to_dict().contains_key("episodes"));
  assert(recorder.result_set_id() == "cpp-parity");
  assert(recorder.to_dict({.recorded_at = "2026-01-01T00:00:00Z"}).contains_key("workloads"));
  recorder.close();
  View view = View::parse("terminal");
  view.validate();
  assert(view.backend == ViewBackend::Terminal);
  try {
    View{.fps = 0}.validate();
    assert(false);
  } catch (const ValueError&) {
  }
}
}  // namespace
int main() try {
  values_spaces();
  vector_and_chunks();
  runs_sessions();
  exceptions();
  factories_remote_describe();
  adapter_axis_presence();
  adapters_readers_recorder();
  std::cout << "cpp_parity OK\n";
} catch (const std::exception& e) {
  std::cerr << e.what() << "\n";
  return 1;
}
