// A C++ model driving a remote RLMesh environment (connection shape #1).
//
//   c++ -std=c++17 -I<include> model.cpp -lrlmesh_capi -o cpp_model
//   ./cpp_model tcp://127.0.0.1:50051
//
// The policy reads the observation, then emits a zero action sized to the
// environment's action space — a placeholder for real control code.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <rlmesh.hpp>
#include <string>
#include <vector>

namespace {

// A trivial "zero policy": build an all-zeros action matching the action space.
rlmesh::Result<rlmesh::Value> zero_policy(const rlmesh::Observation& obs) {
  const RlmeshSpaceSpec* action = obs.action_space();
  if (action == nullptr) {
    return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "no action space", false);
  }

  switch (rlmesh_space_type(action)) {
    case 1: {  // Box: zeros of the right shape/dtype.
      RlmeshDType dtype = rlmesh_space_dtype(action);
      size_t ndim = rlmesh_space_ndim(action);
      std::vector<int64_t> shape(ndim);
      if (RlmeshStatus s = rlmesh_space_copy_shape(action, shape.data(), ndim); s != RLMESH_OK) {
        return rlmesh::Error::from_last(s);
      }
      size_t numel = 1;
      for (int64_t dim : shape) numel *= static_cast<size_t>(dim);
      std::vector<uint8_t> zeros(numel * rlmesh_dtype_size(dtype), 0);
      return rlmesh::Value::box(zeros.data(), dtype, std::move(shape));
    }
    case 2:  // Discrete: action 0.
      return rlmesh::Value::discrete(0);
    default:
      return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "example handles Box/Discrete only", false);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "tcp://127.0.0.1:50051";
  // Optional `<address> <episodes>`: run N episodes then exit (used by the e2e
  // harness). Absent → run until the environment ends.
  std::optional<uint64_t> episodes;
  if (argc > 2) episodes = std::strtoull(argv[2], nullptr, 10);

  auto model = rlmesh::Model::from_predict([](const rlmesh::Observation& obs) {
    // A real policy would decode and run inference; we just demonstrate access:
    if (auto batch = obs.decode(); batch && batch.value().size() > 0) {
      if (auto tensor = batch.value().tensor_at(0)) {
        std::printf("obs: %d-D tensor, %zu elements\n", tensor.value().ndim(),
                    tensor.value().numel());
      }
    }
    return zero_policy(obs);
  });
  if (!model) {
    std::fprintf(stderr, "failed to create model: %.*s\n",
                 static_cast<int>(model.error().message().size()), model.error().message().data());
    return 1;
  }

  std::printf("connecting to %s ...\n", address.c_str());
  auto result =
      episodes ? model.value().run_local(address, *episodes) : model.value().run_local(address);
  if (!result) {
    std::fprintf(stderr, "run failed: %.*s\n", static_cast<int>(result.error().message().size()),
                 result.error().message().data());
    return 1;
  }
  return 0;
}
