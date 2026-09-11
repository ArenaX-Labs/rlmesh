// A C++ model driving a remote RLMesh environment through the rlmesh.hpp wrapper.
//
//   c++ -std=c++17 -I<include> cpp_model.cpp -lrlmesh_capi -o cpp_model
//   ./cpp_model tcp://127.0.0.1:5555 [episodes]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <rlmesh.hpp>
#include <string>

namespace {

// Always act with zero: a scalar 0 for Discrete, a zero tensor for Box.
rlmesh::Result<rlmesh::Value> zero_policy(const rlmesh::Observation& obs) {
  const RlmeshSpaceSpec* action = obs.action_space();
  if (action == nullptr) return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "no action space");
  switch (rlmesh_space_type(action)) {
    case 1:
      return rlmesh::Value::zeros(action);
    case 2:
      return rlmesh::Value::discrete(0);
    default:
      return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "example handles Box/Discrete only");
  }
}

void report(std::string_view what, const rlmesh::Error& error) {
  std::fprintf(stderr, "%.*s: %.*s\n", static_cast<int>(what.size()), what.data(),
               static_cast<int>(error.message().size()), error.message().data());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "tcp://127.0.0.1:5555";
  rlmesh::RunOptions options;
  if (argc > 2) options.max_episodes = std::strtoull(argv[2], nullptr, 10);

  auto model = rlmesh::Model::from_predict([](const rlmesh::Observation& obs) {
    if (obs.has_values()) {
      if (auto n = obs.at(0).as_discrete()) {
        std::printf("episode %.*s: obs %lld\n", static_cast<int>(obs.episode_id(0).size()),
                    obs.episode_id(0).data(), static_cast<long long>(n.value()));
      } else if (auto tensor = obs.at(0).as_tensor()) {
        std::printf("obs: %d-D tensor, %zu elements\n", tensor.value().ndim(),
                    tensor.value().numel());
      }
    }
    return zero_policy(obs);
  });
  if (!model) {
    report("failed to create model", model.error());
    return 1;
  }

  std::printf("connecting to %s ...\n", address.c_str());
  if (auto result = model.value().run_local(address, options); !result) {
    report("run failed", result.error());
    return 1;
  }
  return 0;
}
