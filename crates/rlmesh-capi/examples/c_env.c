/* Minimal ABI 5 environment describe with the translation unit's host runtime. */
#include <rlmesh.h>
#include <stdio.h>
static RLMeshStatus reset(void* u, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  (void)u;
  (void)seed;
  (void)options;
  (void)info;
  return rlmesh_value_int(0, obs);
}
static RLMeshStatus step(void* u, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                         bool* term, bool* trunc, RLMeshValue** info) {
  (void)u;
  (void)action;
  (void)reward;
  (void)trunc;
  (void)info;
  *term = true;
  return rlmesh_value_int(0, obs);
}
int main(void) {
  RLMeshSpace* space = NULL;
  if (rlmesh_discrete_new(2, NULL, &space) != RLMESH_OK) return 1;
  RLMeshEnvVtable vt = RLMESH_ENV_VTABLE_INIT;
  vt.reset = reset;
  vt.step = step;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = space;
  attrs.action_space = space;
  attrs.id = "CExample-v0";
  RLMeshEnv* env = NULL;
  if (rlmesh_env_new(&vt, NULL, &attrs, &env) != RLMESH_OK) return 1;
  RLMeshExtEnvServerOptions ext = RLMESH_EXT_ENV_SERVER_OPTIONS_INIT;
  RLMeshEnvServerOptions options = RLMESH_ENV_SERVER_OPTIONS_INIT;
  options.ext = &ext;
  RLMeshEnvServer* server = NULL;
  RLMeshStatus status = rlmesh_env_server_new(&env, 1, &options, &server);
  if (status == RLMESH_OK) {
    RLMeshBytes description = {NULL, 0, 0};
    status = rlmesh_ext_env_server_describe_json(server, &description);
    if (status == RLMESH_OK) printf("%.*s\n", (int)description.len, (const char*)description.data);
    rlmesh_bytes_free(description);
  }
  if (status != RLMESH_OK) fprintf(stderr, "%s\n", rlmesh_last_error_message());
  rlmesh_env_server_free(server);
  rlmesh_env_free(env);
  rlmesh_space_free(space);
  return status == RLMESH_OK ? 0 : 1;
}
