// A detected C++20 env with typed spaces, values and metadata.
#include <cmath>
#include <cstdio>
#include <rlmesh.hpp>
#include <thread>

namespace {
class PointReach {
  int step_ = 0;
  std::vector<float> pos_{0, 0, 0}, target_{0, 0.2f, 0.3f};
  bool announce_;
  rlmesh::Value camera() const {
    std::vector<uint8_t> pixels(8 * 8 * 3, 32);
    return rlmesh::Tensor(pixels.data(), pixels.size(), {8, 8, 3}, rlmesh::DType::Uint8);
  }
  rlmesh::Value observe() const {
    return rlmesh::Metadata{
        {"camera", camera()},
        {"eef_pos",
         rlmesh::Tensor(pos_.data(), pos_.size() * sizeof(float), {3}, rlmesh::DType::Float32)},
        {"target_pos", rlmesh::Tensor(target_.data(), target_.size() * sizeof(float), {3},
                                      rlmesh::DType::Float32)}};
  }

 public:
  explicit PointReach(bool announce) : announce_(announce) {}
  rlmesh::spaces::Space observation_space() const {
    using namespace rlmesh::spaces;
    return Dict({{"camera", Box(0, 255, {8, 8, 3}, {.dtype = rlmesh::DType::Uint8})},
                 {"eef_pos", Box(-INFINITY, INFINITY, {3})},
                 {"target_pos", Box(-INFINITY, INFINITY, {3})}});
  }
  rlmesh::spaces::Space action_space() const { return rlmesh::spaces::Box(-1, 1, {3}); }
  std::optional<std::string> id() const { return "PointReach-cpp-v0"; }
  std::optional<std::string> render_mode() const { return "rgb_array"; }
  std::optional<rlmesh::Metadata> metadata() const {
    return rlmesh::Metadata{
        {std::string(rlmesh::ENV_RESET_OPTIONS_KEY), rlmesh::Value::list({"trial_index"})}};
  }
  rlmesh::ResetResult reset(std::optional<int64_t> seed,
                            const std::optional<rlmesh::InfoDict>& options) {
    step_ = 0;
    pos_ = {0, 0, 0};
    auto trial = rlmesh::trial_index(options);
    target_[0] = 0.1f * static_cast<float>((seed.value_or(0) + trial.value_or(0)) % 7);
    if (announce_) {
      std::printf("reset seed=%lld trial=%lld\n", static_cast<long long>(seed.value_or(-1)),
                  static_cast<long long>(trial.value_or(-1)));
      std::fflush(stdout);
    }
    return {observe(), {{"target", "point"}}};
  }
  rlmesh::StepResult step(std::optional<rlmesh::Value> action) {
    if (!action) throw rlmesh::TypeError("step without an action");
    auto tensor = action->as_tensor();
    if (tensor.dtype() != rlmesh::DType::Float32 || tensor.size() != 3)
      throw rlmesh::TypeError("expected float32[3] action");
    const auto* delta = reinterpret_cast<const float*>(tensor.buffer().data());
    double distance = 0;
    for (size_t i = 0; i < 3; ++i) {
      pos_[i] += 0.1f * delta[i];
      distance += (pos_[i] - target_[i]) * (pos_[i] - target_[i]);
    }
    return {
        observe(), -std::sqrt(distance), ++step_ >= 2, false, {{"distance", std::sqrt(distance)}}};
  }
  std::optional<rlmesh::Value> render() { return camera(); }
  void close() {
    if (announce_) {
      std::puts("env closed");
      std::fflush(stdout);
    }
  }
};
}  // namespace
int main(int argc, char** argv) try {
  const std::string address = argc > 1 ? argv[1] : "tcp://127.0.0.1:5555";
  const bool describe = address == "--describe";
  using namespace rlmesh::adapters;
  EnvTags tags(ObsNode::dict({{"camera", ImageTag{.role = IMAGE_PRIMARY}},
                              {"eef_pos", StateTag{.role = EEF_POS}}}),
               Action{.components = {{.role = ACTION_DELTA_POS, .dim = 3}},
                      .clip = std::pair<double, double>{-1, 1}});
  rlmesh::EnvServer server(PointReach(!describe),
                           {.address = describe ? "tcp://127.0.0.1:0" : address, .tags = tags});
  if (describe) {
    std::puts(rlmesh::ext::env_server_describe_json(server).c_str());
    return 0;
  }
  server.start();
  std::printf("listening on %s\n", server.address().c_str());
  std::fflush(stdout);
  while (std::fgetc(stdin) != EOF) {
  }
  server.shutdown();
  return 0;
} catch (const rlmesh::ext::CancelledError&) {
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
