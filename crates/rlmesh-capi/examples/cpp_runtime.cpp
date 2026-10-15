// Describe compatibility through the C++ wrapper and the C header in C++.
#include <cstdio>
#include <rlmesh.hpp>

namespace {
class UnusedEnv : public rlmesh::Environment {
 public:
  rlmesh::Result<rlmesh::ResetOutput> reset(const rlmesh::ResetArgs&) override {
    return rlmesh::Error(RLMESH_ERR_ENVIRONMENT, "describe never calls reset");
  }
  rlmesh::Result<rlmesh::StepOutput> step(std::optional<rlmesh::ValueRef>) override {
    return rlmesh::Error(RLMESH_ERR_ENVIRONMENT, "describe never calls step");
  }
};

int reset(void*, const RLMeshExtLegacyResetArgs*, RLMeshExtLegacyResetResult*) { return 1; }
int step(void*, const RLMeshValue*, RLMeshExtLegacyStepResult*) { return 1; }

rlmesh::Result<std::string> describe(const RLMeshExtLegacyEnvConfig& config) {
  RLMeshExtLegacyEnvVtable vtable{sizeof(RLMeshExtLegacyEnvVtable), reset, step, nullptr, nullptr};
  RLMeshExtLegacyEnv* env = nullptr;
  void* user_data = nullptr;
  RLMeshStatus status = rlmesh_ext_legacy_env_new_lanes(&vtable, &config, &user_data, 1, &env);
  if (status != RLMESH_OK) return rlmesh::Error(status, rlmesh_last_error_message());
  RLMeshBytes bytes{};
  status = rlmesh_ext_legacy_env_describe_json(env, &bytes);
  rlmesh_ext_legacy_env_free(env);
  if (status != RLMESH_OK) return rlmesh::Error(status, rlmesh_last_error_message());
  std::string text(reinterpret_cast<const char*>(bytes.data), bytes.len);
  rlmesh_bytes_free(bytes);
  return text;
}
}  // namespace

int main() {
  auto observation = rlmesh::Space::discrete(2);
  auto action = rlmesh::Space::discrete(2);
  if (!observation || !action) return 1;
  RLMeshExtLegacyEnvConfig raw = rlmesh_ext_legacy_env_config_init();
  raw.observation_space = observation->raw();
  raw.action_space = action->raw();
  raw.struct_size = offsetof(RLMeshExtLegacyEnvConfig, host_runtime_json);
  auto old = describe(raw);
  if (old || old.error().code() != RLMESH_ERR_INVALID_ARGUMENT) return 1;
  raw.struct_size = sizeof(RLMeshExtLegacyEnvConfig);
  raw.host_runtime_json = nullptr;
  auto null_runtime = describe(raw);

  rlmesh::EnvConfig config;
  config.observation_space = observation.unwrap();
  config.action_space = action.unwrap();
  config.host_runtime_json.clear();
  auto server = rlmesh::EnvServer::create(std::make_unique<UnusedEnv>(), config);
  if (!null_runtime || !server) return 1;
  auto empty_runtime = server->describe_json();
  if (!empty_runtime || *null_runtime != *empty_runtime) {
    std::fprintf(stderr, "legacy describe output changed\n");
    return 1;
  }
  std::puts("C++ empty and NULL declarations match byte-for-byte; ABI-4 prefix refused");
  return 0;
}
