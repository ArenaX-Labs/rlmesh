#include <chrono>
#include <iostream>
#include <rlmesh.hpp>
#include <string>

struct Counter {
  float data[4] = {};
  rlmesh::Value observation =
      rlmesh::Tensor(data, sizeof(data), {4}, rlmesh::DType::Float32);
  rlmesh::Metadata info;
  long steps = 0;
  rlmesh::spaces::Space observation_space() {
    return rlmesh::spaces::Box(-1.f, 1.f, {4});
  }
  rlmesh::spaces::Space action_space() { return rlmesh::spaces::Discrete(2); }
  rlmesh::ResetResult reset(std::optional<int64_t>,
                            std::optional<rlmesh::Metadata>) {
    steps = 0;
    return {observation, info};
  }
  rlmesh::StepResult step(std::optional<rlmesh::Value>) {
    ++steps;
    return {observation, 0, false, false, info};
  }
};
int main() {
  rlmesh::Env env(Counter{});
  auto binding = rlmesh::bind(rlmesh::RANDOM_SAMPLE, env);
  binding.reset();
  rlmesh::Value action = 0;
  for (int i = 0; i < 1000; ++i)
    binding.step(action);
  std::cout << "ready" << std::endl;
  std::string line;
  while (std::getline(std::cin, line)) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; ++i)
      binding.step(action);
    auto nanos = std::chrono::duration<double, std::nano>(
                     std::chrono::steady_clock::now() - start)
                     .count();
    std::cout << nanos / 10000 << std::endl;
  }
}
