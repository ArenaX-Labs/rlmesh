// The C++ gate program of python/rlmesh/tests/conformance/native_gate.py: the
// Gymnasium gate's echo env, echo model and clients, on v2 sessions.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <rlmesh.hpp>
#include <string>
#include <thread>

namespace {
using namespace rlmesh;
using namespace rlmesh::spaces;

struct Unsupported {
  std::string why;
};

Space f32_box(double low, double high, std::vector<int64_t> shape) { return Box(low, high, shape); }
Space typed_box(Bound low, Bound high, std::vector<int64_t> shape, DType dtype) {
  return Box(low, high, shape, BoxOptions{}.with_dtype(dtype));
}

// The gate's SPACES, by name.
Space gate_space(const std::string& row) {
  const double inf = INFINITY;
  if (row == "box_f32_bounded") return f32_box(-1.0, 2.0, {3});
  if (row == "box_f64_inf") return typed_box(-inf, inf, {2, 2}, DType::Float64);
  if (row == "box_mixed_bounds") return ext::box_elementwise({-1.0, -inf}, {inf, 3.0}, {2});
  if (row == "box_f16") return typed_box(-1.0, 1.0, {4}, DType::Float16);
  if (row == "box_i8") return typed_box(-5, 5, {3}, DType::Int8);
  if (row == "box_u8_image") return typed_box(0, 255, {6, 5, 3}, DType::Uint8);
  if (row == "box_u16") return typed_box(0, 60000, {2}, DType::Uint16);
  if (row == "box_i64") return typed_box(-(int64_t{1} << 40), int64_t{1} << 40, {2}, DType::Int64);
  if (row == "box_u64") return typed_box(0, int64_t{1} << 40, {2}, DType::Uint64);
  if (row == "box_i16") return typed_box(-300, 300, {3}, DType::Int16);
  if (row == "box_i32") return typed_box(-(1 << 20), 1 << 20, {3}, DType::Int32);
  if (row == "box_u32") return typed_box(0, 1 << 30, {2}, DType::Uint32);
  if (row == "box_bool") return typed_box(0, 1, {3}, DType::Bool);
  if (row == "discrete") return Discrete(4);
  if (row == "discrete_start") return Discrete(5, DiscreteOptions{}.with_start(-2));
  if (row == "multi_discrete") return MultiDiscrete({3, 5, 2});
  if (row == "multi_discrete_2d") throw Unsupported{"C++ MultiDiscrete takes a 1-D nvec"};
  if (row == "multi_discrete_start") throw Unsupported{"C++ MultiDiscrete takes no start"};
  if (row == "multi_binary") return MultiBinary({4});
  if (row == "multi_binary_2d") return MultiBinary({2, 3});
  if (row == "dict_flat") return Dict({{"pos", f32_box(-1, 1, {2})}, {"gripper", Discrete(2)}});
  if (row == "dict_nested")
    return Dict({{"robot", Dict({{"q", f32_box(-1, 1, {3})}, {"mode", Discrete(2)}})},
                 {"scene", Dict({{"img", typed_box(0, 255, {4, 4, 3}, DType::Uint8)}})}});
  if (row == "dict_keys_that_escape_alike")
    return Dict({{"Pos", Discrete(3)}, {"pos", Discrete(5)}});
  if (row == "dict_key_named_action") return Dict({{"action", f32_box(-1, 1, {1})}});
  if (row == "tuple") return Tuple({Discrete(2), f32_box(-1, 1, {2})});
  if (row == "tuple_nested")
    return Tuple({Dict({{"a", Discrete(2)}}), Tuple({MultiBinary({2}), f32_box(-1, 1, {2})})});
  if (row == "text") return Text(12, TextOptions{}.with_min_length(1).with_charset("abc xyz"));
  if (row == "scalar_box") return f32_box(-1, 1, {});
  return f32_box(-1, 1, {3});
}

// The gate's rich_info(t).
InfoDict rich_info(int64_t t) {
  const std::byte raw[] = {std::byte{0x00}, std::byte{0xff}};
  return {{"nested", Metadata{{"a", Metadata{{"b", t}, {"c", Value::list({1, 2.5, "x"})}}},
                              {"flag", t % 2 == 0}}},
          {"nan", NAN},
          {"inf", -INFINITY},
          {"bytes", Value::bytes(raw)},
          {"$dollar", 1},
          {"none", Value::none()},
          {"third", 1.0 / 3},
          {"is_success", t >= 3}};
}

class EchoEnv {
  std::string row_;
  Space space_;
  int64_t t_ = 0;

 public:
  explicit EchoEnv(std::string row) : row_(std::move(row)), space_(gate_space(row_)) {}
  Space observation_space() const { return space_; }
  Space action_space() const { return space_; }
  ResetResult reset(std::optional<int64_t>, const std::optional<InfoDict>& options) {
    if (row_ == "error_reset") throw ValueError("boom in reset");
    t_ = 0;
    InfoDict info;
    if (options) info = {{"options", *options}};
    return {space_.sample(), info};
  }
  StepResult step(std::optional<Value> action) {
    if (row_ == "error_step") throw RuntimeError("boom in step");
    ++t_;
    const bool done = t_ >= 3;
    InfoDict info = row_ == "info" ? rich_info(t_) : InfoDict{{"t", t_}};
    return {*action, static_cast<double>(t_), done, done && row_ == "both_done", info};
  }
  std::optional<Value> render() const {
    std::vector<uint8_t> frame(4 * 5 * 3, 0);
    for (size_t i = 0; i < frame.size(); i += 3) frame[i] = static_cast<uint8_t>(t_);
    return Tensor(frame.data(), frame.size(), {4, 5, 3}, DType::Uint8);
  }
};

void wait_for_eof() {
  while (std::fgetc(stdin) != EOF) {
  }
}

int serve_env(const std::string& row, const std::string& address) {
  EnvServer server(EchoEnv(row), EnvServerOptions{}.with_address(address));
  server.start();
  std::printf("listening on %s\n", server.address().c_str());
  std::fflush(stdout);
  wait_for_eof();
  server.shutdown();
  return 0;
}

Model echo_model() {
  return Model([](const Value& observation) { return observation; });
}

int serve_model(const std::string& address) {
  ext::CancelToken stop;
  std::thread([stop] {
    wait_for_eof();
    stop.cancel();
  }).detach();
  std::puts("serving");
  std::fflush(stdout);
  try {
    echo_model().serve(address, ModelServeOptions{}.with_cancel(stop));
  } catch (const ext::CancelledError&) {
  }
  return 0;
}

RunOptions two_episodes() { return RunOptions{}.with_episodes(2).with_seeds({0, 1}); }

int drive(const std::string& verb, const std::string& address, const char* model) {
  if (verb == "run") {
    std::printf("episodes=%zu\n", echo_model().run(address, two_episodes()).num_episodes());
    return 0;
  }
  if (verb == "remote-model") {
    RemoteModel remote(RemoteModelOptions{}.with_address(model));
    std::printf("episodes=%zu\n", rlmesh::run(remote, address, two_episodes()).num_episodes());
    return 0;
  }
  RemoteEnv env(RemoteEnvOptions{}.with_address(address));
  for (int64_t seed = 0; seed < 2; ++seed) {
    auto [observation, info] = env.reset(seed);
    (void)info;
    for (bool done = false; !done;) {
      auto [next, reward, terminated, truncated, step_info] = env.step(observation);
      (void)reward, (void)step_info;
      observation = next;
      done = terminated || truncated;
    }
  }
  env.close();
  std::puts("episodes=2");
  return 0;
}
}  // namespace

int main(int argc, char** argv) try {
  const std::string verb = argc > 1 ? argv[1] : "";
  if (verb == "env" && argc == 4) return serve_env(argv[2], argv[3]);
  if (verb == "model" && argc == 3) return serve_model(argv[2]);
  if ((verb == "remote-env" || verb == "run") && argc == 3) return drive(verb, argv[2], nullptr);
  if (verb == "remote-model" && argc == 4) return drive(verb, argv[3], argv[2]);
  std::fprintf(stderr, "usage: cpp_gate env|model|remote-env|run|remote-model ...\n");
  return 2;
} catch (const Unsupported& unsupported) {
  std::printf("unsupported: %s\n", unsupported.why.c_str());
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
