/* ABI 5 Env/VectorEnv/EnvServer: typed values, ownership and sized options. */
#include <math.h>
#include <rlmesh.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(x)                                              \
  do {                                                        \
    if (!(x)) {                                               \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      ++failures;                                             \
    }                                                         \
  } while (0)
typedef struct State {
  int closes;
  int frees;
} State;
static RLMeshStatus reset(void* ud, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  (void)ud;
  (void)seed;
  (void)options;
  (void)info;
  return rlmesh_value_int(0, obs);
}
static RLMeshStatus step(void* ud, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                         bool* terminated, bool* truncated, RLMeshValue** info) {
  (void)ud;
  (void)action;
  (void)info;
  *reward = 1.0;
  *terminated = true;
  *truncated = false;
  return rlmesh_value_int(1, obs);
}
static RLMeshStatus close_env(void* ud) {
  ++((State*)ud)->closes;
  return RLMESH_OK;
}
static RLMeshStatus fail_close(void* ud) {
  ++((State*)ud)->closes;
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION, "forced close failure");
  return RLMESH_ERR_ENVIRONMENT;
}
static void free_state(void* ud) { ++((State*)ud)->frees; }
static RLMeshStatus vector_reset(void* ud, const int64_t* seed, size_t seed_len,
                                 const RLMeshValue* options, RLMeshValue** obs,
                                 RLMeshValue** info) {
  (void)seed_len;
  return reset(ud, seed, options, obs, info);
}
static RLMeshStatus vector_step(void* ud, const RLMeshValue* action, RLMeshValue** obs,
                                double* rewards, bool* terminated, bool* truncated,
                                RLMeshValue** info) {
  return step(ud, action, obs, rewards, terminated, truncated, info);
}
static void wait_error_outputs(const RLMeshEnvVtable* vtable, const RLMeshEnvAttributes* attrs) {
  State state = {0, 0};
  RLMeshEnvVtable vt = *vtable;
  vt.close = fail_close;
  RLMeshEnv* env = NULL;
  CHECK(rlmesh_env_new(&vt, &state, attrs, &env) == RLMESH_OK);
  RLMeshServeOptions serve = RLMESH_SERVE_OPTIONS_INIT;
  serve.workflow_edition = rlmesh_current_workflow_edition();
  serve.has_idle_timeout_seconds = true;
  serve.idle_timeout_seconds = 0.01;
  RLMeshEnvServerOptions options = RLMESH_ENV_SERVER_OPTIONS_INIT;
  options.options = &serve;
  RLMeshEnvServer* server = NULL;
  CHECK(rlmesh_env_server_new(&env, 1, &options, &server) == RLMESH_OK);
  /* Misuse leaves either caller-supplied sentinel unchanged. */
  bool stopped = false;
  CHECK(rlmesh_env_server_wait(server, NULL, &stopped) == RLMESH_ERR_RUNTIME && !stopped);
  stopped = true;
  CHECK(rlmesh_env_server_wait(server, NULL, &stopped) == RLMESH_ERR_RUNTIME && stopped);
  CHECK(rlmesh_env_server_wait(NULL, NULL, &stopped) == RLMESH_ERR_INVALID_ARGUMENT && stopped);
  double invalid_timeout = -1;
  CHECK(rlmesh_env_server_wait(server, &invalid_timeout, &stopped) == RLMESH_ERR_INVALID_ARGUMENT &&
        stopped);
  CHECK(rlmesh_env_server_wait(server, NULL, NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(rlmesh_env_server_start(server) == RLMESH_OK);
  /* Idle shutdown invokes the failing close on the background serve thread. */
  stopped = false;
  double timeout = 10;
  CHECK(rlmesh_env_server_wait(server, &timeout, &stopped) == RLMESH_ERR_INTERNAL && stopped);
  CHECK(rlmesh_last_error_class() == RLMESH_ERROR_CLASS_RLMESH_EXCEPTION);
  CHECK(strstr(rlmesh_last_error_message(), "serve failed:") != NULL);
  CHECK(strstr(rlmesh_last_error_message(), "forced close failure") != NULL);
  CHECK(state.closes == 1);
  /* Validation errors still leave output untouched on a stopped server. */
  stopped = false;
  invalid_timeout = NAN;
  CHECK(rlmesh_env_server_wait(server, &invalid_timeout, &stopped) == RLMESH_ERR_INVALID_ARGUMENT &&
        !stopped);
  timeout = 0;
  CHECK(rlmesh_env_server_wait(server, &timeout, &stopped) == RLMESH_OK && stopped);
  CHECK(rlmesh_env_server_shutdown(server) == RLMESH_OK); /* serve error consumed once */
  rlmesh_env_server_free(server);
  rlmesh_env_free(env);
  CHECK(state.closes == 1 && state.frees == 1);
}
int main(void) {
  State state = {0, 0};
  RLMeshDiscrete* space = NULL;
  CHECK(rlmesh_discrete_new(4, NULL, &space) == RLMESH_OK);
  RLMeshEnvVtable vt = RLMESH_ENV_VTABLE_INIT;
  vt.reset = reset;
  vt.step = step;
  vt.close = close_env;
  vt.free_user_data = free_state;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = space;
  attrs.action_space = space;
  attrs.id = "CFacade-v1";
  wait_error_outputs(&vt, &attrs);
  RLMeshEnv* env = NULL;
  CHECK(rlmesh_env_new(&vt, &state, NULL, &env) == RLMESH_ERR_TYPE && state.frees == 0);
  size_t saved = vt.struct_size;
  vt.struct_size = 0;
  CHECK(rlmesh_env_new(&vt, &state, &attrs, &env) == RLMESH_ERR_INVALID_ARGUMENT &&
        state.frees == 0);
  vt.struct_size = saved;
  CHECK(rlmesh_env_new(&vt, &state, &attrs, &env) == RLMESH_OK);
  RLMeshServeOptions serve = RLMESH_SERVE_OPTIONS_INIT;
  serve.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshEnvServerOptions opts = RLMESH_ENV_SERVER_OPTIONS_INIT;
  opts.options = &serve;
  RLMeshEnvServer* server = NULL;
  RLMeshEnv* duplicate[] = {env, env};
  CHECK(rlmesh_env_server_new(duplicate, 2, &opts, &server) == RLMESH_ERR_RUNTIME);
  CHECK(rlmesh_env_server_new(&env, 1, &opts, &server) == RLMESH_OK);
  CHECK(rlmesh_env_close(env) == RLMESH_ERR_RUNTIME);
  RLMeshEnvServer* other = NULL;
  CHECK(rlmesh_env_server_new(&env, 1, &opts, &other) == RLMESH_ERR_RUNTIME);
  const RLMeshEnvContract* contract = rlmesh_env_server_env_contract(server);
  CHECK(rlmesh_env_contract_num_envs(contract) == 1);
  RLMeshBytes address = {NULL, 0, 0};
  CHECK(rlmesh_env_server_address(server, &address) == RLMESH_OK);
  CHECK(address.len > 6 && memcmp(address.data, "tcp://", 6) == 0);
  rlmesh_bytes_free(address);
  CHECK(rlmesh_env_server_start(server) == RLMESH_OK);
  bool stopped = true;
  double timeout = 0.0;
  CHECK(rlmesh_env_server_wait(server, &timeout, &stopped) == RLMESH_OK && !stopped);
  CHECK(rlmesh_env_server_shutdown(server) == RLMESH_OK);
  rlmesh_env_server_free(server);
  CHECK(state.closes == 1 && state.frees == 0);
  CHECK(rlmesh_env_close(env) == RLMESH_OK && state.closes == 1);
  rlmesh_env_free(env);
  CHECK(state.frees == 1);

  State not_closed = {0, 0};
  env = NULL;
  CHECK(rlmesh_env_new(&vt, &not_closed, &attrs, &env) == RLMESH_OK);
  opts.close_env_on_shutdown = false;
  CHECK(rlmesh_env_server_new(&env, 1, &opts, &server) == RLMESH_OK);
  rlmesh_env_server_free(server);
  rlmesh_env_free(env);
  CHECK(not_closed.closes == 0 && not_closed.frees == 1);

  State vstate = {0, 0};
  RLMeshVectorEnvVtable vvt = RLMESH_VECTOR_ENV_VTABLE_INIT;
  vvt.reset = vector_reset;
  vvt.step = vector_step;
  vvt.close = close_env;
  vvt.free_user_data = free_state;
  RLMeshVectorEnvAttributes va = RLMESH_VECTOR_ENV_ATTRIBUTES_INIT;
  va.num_envs = 2;
  va.single_observation_space = space;
  va.single_action_space = space;
  RLMeshVectorEnv* vector = NULL;
  CHECK(rlmesh_vector_env_new(&vvt, &vstate, &va, &vector) == RLMESH_OK);
  opts.close_env_on_shutdown = true;
  CHECK(rlmesh_env_server_new_vector(vector, &opts, &server) == RLMESH_OK);
  CHECK(rlmesh_env_contract_num_envs(rlmesh_env_server_env_contract(server)) == 2);
  CHECK(rlmesh_vector_env_close(vector) == RLMESH_ERR_RUNTIME);
  rlmesh_env_server_free(server);
  rlmesh_vector_env_free(vector);
  CHECK(vstate.closes == 1 && vstate.frees == 1);

  struct Extended {
    RLMeshEnvAttributes attrs;
    unsigned char tail[8];
  } extended = {RLMESH_ENV_ATTRIBUTES_INIT, {0}};
  extended.attrs = attrs;
  extended.attrs.struct_size = sizeof(extended);
  State extended_state = {0, 0};
  CHECK(rlmesh_env_new(&vt, &extended_state, &extended.attrs, &env) == RLMESH_OK);
  rlmesh_env_free(env);
  extended.tail[3] = 1;
  CHECK(rlmesh_env_new(&vt, &state, &extended.attrs, &env) == RLMESH_ERR_TYPE);
  CHECK(strstr(rlmesh_last_error_message(), "RLMeshEnvAttributes") != NULL);
  serve.has_idle_timeout_seconds = true;
  serve.idle_timeout_seconds = NAN;
  State invalid = {0, 0};
  CHECK(rlmesh_env_new(&vt, &invalid, &attrs, &env) == RLMESH_OK);
  CHECK(rlmesh_env_server_new(&env, 1, &opts, &server) == RLMESH_ERR_INVALID_ARGUMENT);
  rlmesh_env_free(env);
  rlmesh_space_free(space);
  if (failures) return 1;
  puts("c_env_api OK");
  return 0;
}
