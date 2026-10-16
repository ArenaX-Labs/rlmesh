/* A minimal C environment's image-label describe envelope. The config helper
 * detects the C standard and compiler in this translation unit. */
#include <rlmesh.h>
#include <stdio.h>

static int reset(void* data, const RLMeshExtLegacyResetArgs* args,
                 RLMeshExtLegacyResetResult* out) {
  (void)data;
  (void)args;
  rlmesh_value_int(0, &out->observation);
  return out->observation ? 0 : 1;
}

static int step(void* data, const RLMeshValue* action, RLMeshExtLegacyStepResult* out) {
  (void)data;
  (void)action;
  rlmesh_value_int(0, &out->observation);
  out->terminated = true;
  return out->observation ? 0 : 1;
}

int main(void) {
  RLMeshExtLegacyEnvConfig config = rlmesh_ext_legacy_env_config_init();
  config.id = "CExample-v0";
  RLMeshDiscrete* observation = NULL;
  RLMeshDiscrete* action = NULL;
  if (rlmesh_discrete_new(2, NULL, &observation) != RLMESH_OK ||
      rlmesh_discrete_new(2, NULL, &action) != RLMESH_OK) {
    fprintf(stderr, "space failed: %s\n", rlmesh_last_error_message());
    return 1;
  }
  /* The config borrows each space's spec; the env copies it. */
  config.observation_space = rlmesh_space_spec(observation);
  config.action_space = rlmesh_space_spec(action);
  RLMeshExtLegacyEnvVtable vtable = {sizeof(RLMeshExtLegacyEnvVtable), reset, step, NULL, NULL};
  RLMeshExtLegacyEnv* env = NULL;
  void* user_data = NULL;
  RLMeshStatus status = rlmesh_ext_legacy_env_new_lanes(&vtable, &config, &user_data, 1, &env);
  rlmesh_space_free(observation);
  rlmesh_space_free(action);
  if (status != RLMESH_OK) {
    fprintf(stderr, "create failed: %s\n", rlmesh_last_error_message());
    return 1;
  }
  RLMeshBytes description = {NULL, 0, 0};
  status = rlmesh_ext_legacy_env_describe_json(env, &description);
  if (status == RLMESH_OK) {
    fwrite(description.data, 1, description.len, stdout);
    putchar('\n');
    rlmesh_bytes_free(description);
  } else {
    fprintf(stderr, "describe failed: %s\n", rlmesh_last_error_message());
  }
  rlmesh_ext_legacy_env_free(env);
  return status == RLMESH_OK ? 0 : 1;
}
