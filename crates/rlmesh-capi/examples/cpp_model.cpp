// A C++ model driving a remote RLMesh environment through the rlmesh.hpp wrapper.
//
//   c++ -std=c++17 -I<include> cpp_model.cpp -lrlmesh_capi -o cpp_model
//   ./cpp_model tcp://127.0.0.1:5555 [episodes]
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <rlmesh.hpp>
#include <string>
#include <string_view>

namespace {

// Log what arrived: which predict of the episode this is (plus its per-predict
// seed when the episode was seeded), then the observation, whatever kind the
// observation space happens to be.
void log_observation(const rlmesh::Request& request) {
  const rlmesh::Episode& episode = request.episode();
  std::printf("episode %.*s: predict %llu", static_cast<int>(episode.id.size()), episode.id.data(),
              static_cast<unsigned long long>(episode.predict_index));
  if (episode.predict_seed) {
    std::printf(" seed %lld", static_cast<long long>(*episode.predict_seed));
  }
  if (std::optional<rlmesh::ValueRef> observation = request.observation()) {
    if (auto n = observation->as_discrete()) {
      std::printf(" obs %lld", static_cast<long long>(*n));
    } else if (auto tensor = observation->as_tensor()) {
      std::printf(" obs %d-D tensor, %zu elements", tensor->ndim(), tensor->numel());
    }
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "tcp://127.0.0.1:5555";
  rlmesh::RunOptions options;
  options.seeded = true;  // deterministic env resets
  options.base_seed = 7;
  if (argc > 2) options.max_episodes = std::strtoull(argv[2], nullptr, 10);

  // One action per predict: the neutral value of whatever the route's action
  // space turns out to be — Box, Discrete, Text, Multi*, Dict or Tuple.
  auto model = rlmesh::Model::from_predict([](const rlmesh::Request& request) {
    log_observation(request);
    return rlmesh::zeros_for(request.action_space());
  });
  if (!model) {
    std::fprintf(stderr, "failed to create model: %s\n", model.error().message().c_str());
    return 1;
  }
  model
      ->on_episode_end([](std::string_view, std::string_view episode_id) {
        std::string id = episode_id.empty() ? std::string("(all)") : std::string(episode_id);
        std::printf("episode end: %s\n", id.c_str());
      })
      .on_close([] { std::printf("model closed\n"); });

  std::printf("connecting to %s ...\n", address.c_str());
  auto report = model->run_local(address, options);
  if (!report) {
    // This example never cancels, but a host that does (a Ctrl-C handler calling
    // Model::cancel from another thread) lands here: a cancelled run is a clean
    // stop with no report, not a failure.
    if (report.error().is_cancelled()) return 0;
    std::fprintf(stderr, "run failed: %s\n", report.error().message().c_str());
    return 1;
  }
  std::printf("run report: episodes=%lld steps=%lld reward=%.1f mean=%.1f\n",
              static_cast<long long>(report->total_episodes),
              static_cast<long long>(report->total_steps), report->total_reward,
              report->mean_reward);
  return 0;
}
