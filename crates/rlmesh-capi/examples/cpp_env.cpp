// A C++ environment served over RLMesh through the rlmesh.hpp wrapper: a point
// that moves toward a target, with a tiny camera, published with adapter tags
// so any tagged model resolves against it.
//
//   c++ -std=c++17 -I<include> cpp_env.cpp -lrlmesh_capi -o cpp_env
//   ./cpp_env [bind-address]        # default 127.0.0.1:5555; port 0 picks one
//
// Prints `listening on <address>` once bound. Serves until stdin closes (or a
// remote shutdown), then closes the env and exits 0.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <rlmesh.hpp>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kSteps = 2;   // steps per episode
constexpr int64_t kPx = 8;  // camera side, in pixels

// Adapter tags: the roles a model's spec resolves against. Structure (shapes,
// dtypes) comes from the spaces; the tags carry only the semantics.
constexpr const char* kTags = R"({
  "observation": {
    "camera": {"type": "image", "role": "image/primary", "layout": "hwc"},
    "eef_pos": {"type": "state", "role": "proprio/eef_pos"}
  },
  "action": {"components": [{"role": "action/delta_eef_pos", "dim": 3}], "clip": [-1.0, 1.0]}
})";

class PointReach : public rlmesh::Environment {
 public:
  rlmesh::Result<rlmesh::ResetOutput> reset(const rlmesh::ResetArgs& args) override {
    step_ = 0;
    pos_ = {0.0f, 0.0f, 0.0f};
    // Seed and trial index pick the target, so a seeded run is reproducible.
    const int64_t salt = args.seed.value_or(0) + args.trial_index.value_or(0);
    target_ = {0.1f * static_cast<float>(salt % 7), 0.2f, 0.3f};
    std::printf("reset seed=%lld trial=%lld\n", static_cast<long long>(args.seed.value_or(-1)),
                static_cast<long long>(args.trial_index.value_or(-1)));
    std::fflush(stdout);
    auto observation = observe();
    if (!observation) return observation.error();
    return rlmesh::ResetOutput{observation.unwrap(), R"({"target": "point"})"};
  }

  rlmesh::Result<rlmesh::StepOutput> step(std::optional<rlmesh::ValueRef> action) override {
    if (!action) return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "step without an action");
    auto tensor = action->as_tensor();
    if (!tensor) return tensor.error();
    const float* delta = tensor->as<float>();
    if (delta == nullptr || tensor->numel() != 3) {
      return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "expected a contiguous float32[3] action");
    }
    for (int i = 0; i < 3; ++i) pos_[i] += 0.1f * delta[i];
    ++step_;
    auto observation = observe();
    if (!observation) return observation.error();
    return rlmesh::StepOutput{observation.unwrap(), -distance(), /*terminated=*/step_ >= kSteps,
                              /*truncated=*/false,
                              "{\"distance\": " + std::to_string(distance()) + "}"};
  }

  rlmesh::Result<std::optional<rlmesh::Value>> render() override {
    auto frame = camera();
    if (!frame) return frame.error();
    return std::optional<rlmesh::Value>(frame.unwrap());
  }

  void close() override {
    std::printf("env closed\n");
    std::fflush(stdout);
  }

 private:
  float distance() const {
    float sum = 0;
    for (int i = 0; i < 3; ++i) sum += (pos_[i] - target_[i]) * (pos_[i] - target_[i]);
    return std::sqrt(sum);
  }

  rlmesh::Result<rlmesh::Value> camera() const {
    std::vector<uint8_t> pixels(kPx * kPx * 3, 32);
    auto plot = [&](const std::vector<float>& p, uint8_t r, uint8_t g, uint8_t b) {
      const int64_t x = std::lround((p[0] + 1.0f) * 0.5f * (kPx - 1));
      const int64_t y = std::lround((p[1] + 1.0f) * 0.5f * (kPx - 1));
      if (x < 0 || y < 0 || x >= kPx || y >= kPx) return;
      uint8_t* px = &pixels[static_cast<size_t>((y * kPx + x) * 3)];
      px[0] = r, px[1] = g, px[2] = b;
    };
    plot(target_, 220, 40, 40);
    plot(pos_, 40, 200, 80);
    return rlmesh::Value::box(pixels, {kPx, kPx, 3});
  }

  rlmesh::Result<rlmesh::Value> observe() const {
    auto image = camera();
    if (!image) return image.error();
    auto eef = rlmesh::Value::box(pos_, {3});
    if (!eef) return eef.error();
    auto target = rlmesh::Value::box(target_, {3});
    if (!target) return target.error();
    std::vector<std::pair<std::string, rlmesh::Value>> entries;
    entries.emplace_back("camera", image.unwrap());
    entries.emplace_back("eef_pos", eef.unwrap());
    entries.emplace_back("target_pos", target.unwrap());
    return rlmesh::Value::dict(std::move(entries));
  }

  int step_ = 0;
  std::vector<float> pos_{0.0f, 0.0f, 0.0f};
  std::vector<float> target_{0.0f, 0.0f, 0.0f};
};

rlmesh::Result<rlmesh::EnvConfig> config() {
  const double inf = INFINITY;
  std::vector<std::pair<std::string, rlmesh::Space>> fields;
  auto camera = rlmesh::Space::box<uint8_t>({kPx, kPx, 3}, 0, 255);
  if (!camera) return camera.error();
  auto eef = rlmesh::Space::box<float>({3}, -inf, inf);
  if (!eef) return eef.error();
  auto target = rlmesh::Space::box<float>({3}, -inf, inf);
  if (!target) return target.error();
  fields.emplace_back("camera", camera.unwrap());
  fields.emplace_back("eef_pos", eef.unwrap());
  fields.emplace_back("target_pos", target.unwrap());
  auto observation = rlmesh::Space::dict(std::move(fields));
  if (!observation) return observation.error();
  auto action = rlmesh::Space::box<float>({3}, -1.0, 1.0);
  if (!action) return action.error();

  rlmesh::EnvConfig config;
  config.id = "PointReach-cpp-v0";
  config.observation_space = observation.unwrap();
  config.action_space = action.unwrap();
  config.adapter_tags_json = kTags;
  config.reset_options = {"trial_index"};
  config.render_mode = "rgb_array";
  return config;
}

/// A composite the capi refuses must leave its children with the wrapper, which
/// frees them exactly once (a double free here aborts the smoke), and an
/// integer bound past the dtype must be refused rather than clamped.
bool rejected_spaces_are_refused_cleanly() {
  std::vector<std::pair<std::string, rlmesh::Space>> fields;
  auto ok = rlmesh::Space::discrete(2);
  auto empty_key = rlmesh::Space::discrete(3);
  if (!ok || !empty_key) return false;
  fields.emplace_back("ok", ok.unwrap());
  fields.emplace_back("", empty_key.unwrap());
  if (rlmesh::Space::dict(std::move(fields))) return false;
  return !rlmesh::Space::box<int64_t>({2}, 1e30, 1e31);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "127.0.0.1:5555";
  if (!rejected_spaces_are_refused_cleanly()) {
    std::fprintf(stderr, "an invalid space was not refused\n");
    return 1;
  }
  auto env_config = config();
  if (!env_config) {
    std::fprintf(stderr, "invalid env config: %s\n", env_config.error().message().c_str());
    return 1;
  }
  auto server = rlmesh::EnvServer::create(std::make_unique<PointReach>(), *env_config);
  if (!server) {
    std::fprintf(stderr, "failed to create env: %s\n", server.error().message().c_str());
    return 1;
  }
  auto bound = server->bind(address);
  if (!bound) {
    std::fprintf(stderr, "failed to bind: %s\n", bound.error().message().c_str());
    return 1;
  }
  std::printf("listening on %s\n", bound->c_str());
  std::fflush(stdout);

  // Stop when stdin closes: the parent process (or a terminal Ctrl-D) owns our
  // lifetime. cancel() is the one call that is safe while serve() blocks.
  rlmesh::EnvServer* handle = &*server;
  std::thread watcher([handle] {
    while (std::fgetc(stdin) != EOF) {
    }
    handle->cancel();
  });
  watcher.detach();

  rlmesh::Status served = server->serve();
  if (!served) {
    std::fprintf(stderr, "serve failed: %s\n", served.error().message().c_str());
    return 1;
  }
  return 0;
}
