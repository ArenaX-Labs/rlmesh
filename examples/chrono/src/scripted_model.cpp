// Native C++17 policy driving the Chrono env.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <rlmesh.hpp>
namespace {
std::vector<float> read_vec3(const rlmesh::Value& observation, const char* key) {
  const auto value = observation.as_dict().at(key).as_tensor();
  if (value.dtype() != rlmesh::DType::Float32 || value.size() != 3)
    throw rlmesh::TypeError(std::string(key) + " must be float32[3]");
  const auto* data = reinterpret_cast<const float*>(value.buffer().data());
  return {data, data + 3};
}
}  // namespace
int main(int argc, char** argv) try {
  const std::string address = argc > 1 ? argv[1] : "tcp://127.0.0.1:50051";
  const uint64_t episodes = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 4;
  rlmesh::Model model(
      [](const rlmesh::Value& observation) -> rlmesh::Value {
        const auto eef = read_vec3(observation, "eef_pos"),
                   target = read_vec3(observation, "target_pos");
        std::vector<float> action(3);
        for (size_t i = 0; i < 3; ++i)
          action[i] = std::clamp((target[i] - eef[i]) / 0.03f, -1.0f, 1.0f);
        return rlmesh::Tensor(action.data(), action.size() * sizeof(float), {3},
                              rlmesh::DType::Float32);
      },
      rlmesh::ModelOptions{}.with_on_episode_end([](std::string_view id) {
        std::printf("episode %.*s done\n", static_cast<int>(id.size()), id.data());
      }));
  std::vector<int64_t> seeds;
  for (uint64_t i = 0; i < episodes; ++i) seeds.push_back(static_cast<int64_t>(7 + i));
  auto result =
      model.run(address, rlmesh::RunOptions{}.with_episodes(episodes).with_seeds(std::move(seeds)));
  std::printf("episodes=%zu steps=%lld mean_reward=%.2f\n", result.num_episodes(),
              static_cast<long long>(result.total_steps()), result.mean_reward());
  return 0;
} catch (const rlmesh::ext::CancelledError&) {
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
