// A C++ model for the Chrono env: no Python on either side of the wire. It
// reads the tool and target positions out of the observation and steers the
// tool straight at the target.
//
//   chrono_reach_model [ADDRESS] [EPISODES]     # default 127.0.0.1:50051, 4
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <rlmesh.hpp>
#include <string>
#include <vector>

namespace {

// The gap (metres) at which the action saturates.
constexpr float kStepMetres = 0.03f;

rlmesh::Result<std::vector<float>> read_vec3(const rlmesh::ValueRef& obs, const char* key) {
  std::optional<rlmesh::ValueRef> field = obs.get(key);
  if (!field) return rlmesh::Error(RLMESH_ERR_MODEL, std::string("observation has no ") + key);
  auto tensor = field->as_tensor();
  if (!tensor) return tensor.error();
  const float* data = tensor->as<float>();
  if (data == nullptr || tensor->numel() != 3) {
    return rlmesh::Error(RLMESH_ERR_MODEL, std::string(key) + " is not a float32[3]");
  }
  return std::vector<float>(data, data + 3);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "127.0.0.1:50051";
  rlmesh::RunOptions options;
  options.max_episodes = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 4;
  options.seeded = true;
  options.base_seed = 7;

  auto model = rlmesh::Model::from_predict(
      [](const rlmesh::Request& request) -> rlmesh::Result<rlmesh::Value> {
        std::optional<rlmesh::ValueRef> obs = request.observation();
        if (!obs) return rlmesh::Error(RLMESH_ERR_MODEL, "no observation");
        auto eef = read_vec3(*obs, "eef_pos");
        if (!eef) return eef.error();
        auto target = read_vec3(*obs, "target_pos");
        if (!target) return target.error();
        std::vector<float> action(3);
        for (int i = 0; i < 3; ++i) {
          action[i] = std::clamp(((*target)[i] - (*eef)[i]) / kStepMetres, -1.0f, 1.0f);
        }
        return rlmesh::Value::box(action, {3});
      });
  if (!model) {
    std::fprintf(stderr, "failed to create model: %s\n", model.error().message().c_str());
    return 1;
  }
  model->on_episode_end([](std::string_view, std::string_view id) {
    if (!id.empty()) std::printf("episode %.*s done\n", static_cast<int>(id.size()), id.data());
  });

  std::printf("C++ model driving %s for %llu episodes\n", address.c_str(),
              static_cast<unsigned long long>(options.max_episodes));
  auto report = model->run_local(address, options);
  if (!report) {
    std::fprintf(stderr, "run failed: %s\n", report.error().message().c_str());
    return 1;
  }
  std::printf("episodes=%lld steps=%lld terminated=%lld truncated=%lld mean_reward=%.2f\n",
              static_cast<long long>(report->total_episodes),
              static_cast<long long>(report->total_steps),
              static_cast<long long>(report->terminated_episodes),
              static_cast<long long>(report->truncated_episodes), report->mean_reward);
  return 0;
}
