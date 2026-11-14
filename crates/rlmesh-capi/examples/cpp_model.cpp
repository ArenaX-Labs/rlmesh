// c++ -std=c++17 -I<include> cpp_model.cpp -lrlmesh_capi -o cpp_model
#include <cstdio>
#include <cstdlib>
#include <rlmesh.hpp>

int main(int argc, char** argv) try {
  const std::string address = argc > 1 ? argv[1] : "localhost:7447/env";
  const uint64_t episodes = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 3;
  // The run drives the env itself: keep its action space, release the lease.
  auto action_space =
      rlmesh::RemoteEnv(rlmesh::RemoteEnvOptions{}.with_address(address)).action_space();
  rlmesh::Model model(
      [action_space](const rlmesh::Value&, const rlmesh::PredictContext& context) {
        std::printf("episode %.*s: predict %llu", static_cast<int>(context.episode_id.size()),
                    context.episode_id.data(),
                    static_cast<unsigned long long>(context.predict_index));
        if (context.predict_seed)
          std::printf(" seed %lld", static_cast<long long>(*context.predict_seed));
        std::puts("");
        return action_space.sample();
      },
      rlmesh::ModelOptions{}.with_on_episode_end([](std::string_view id) {
        std::printf("episode end: %.*s\n", static_cast<int>(id.size()), id.data());
      }));
  std::vector<int64_t> seeds;
  for (uint64_t i = 0; i < episodes; ++i) seeds.push_back(static_cast<int64_t>(7 + i));
  auto result =
      model.run(address, rlmesh::RunOptions{}.with_episodes(episodes).with_seeds(std::move(seeds)));
  std::printf("run report: episodes=%zu steps=%lld mean=%.1f\n", result.num_episodes(),
              static_cast<long long>(result.total_steps()), result.mean_reward());
  return 0;
} catch (const rlmesh::ext::CancelledError&) {
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
