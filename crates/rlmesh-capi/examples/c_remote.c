/* ABI 5 remote clients: real gRPC, cache identity, borrows and vector buffers. */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <math.h>
#include <pthread.h>
#include <rlmesh.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
static atomic_int failures;
#define CHECK(x)                                                                                \
  do {                                                                                          \
    if (!(x)) {                                                                                 \
      fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, rlmesh_last_error_message()); \
      atomic_fetch_add(&failures, 1);                                                           \
    }                                                                                           \
  } while (0)
typedef struct State {
  atomic_int steps;
  RLMeshRemoteEnv* remote;
} State;
static RLMeshStatus reset(void* ud, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  State* s = (State*)ud;
  (void)seed;
  (void)options;
  (void)info;
  if (s->remote) {
    CHECK(rlmesh_remote_env_close(s->remote) == RLMESH_ERR_RUNTIME);
    CHECK(rlmesh_value_kind(rlmesh_remote_env_metadata(s->remote)) == RLMESH_VALUE_DICT);
    RLMeshBytes edition = {NULL, 0, 0};
    CHECK(rlmesh_remote_env_selected_workflow_edition(s->remote, &edition) == RLMESH_OK &&
          edition.len > 0);
    rlmesh_bytes_free(edition);
  }
  return rlmesh_value_int(0, obs);
}
static RLMeshStatus step(void* ud, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                         bool* terminated, bool* truncated, RLMeshValue** info) {
  atomic_fetch_add(&((State*)ud)->steps, 1);
  (void)action;
  (void)info;
  *reward = 2;
  *terminated = true;
  *truncated = false;
  return rlmesh_value_int(1, obs);
}
static RLMeshStatus predict(void* ud, const RLMeshValue* obs, const RLMeshPredictContext* context,
                            RLMeshValue** out) {
  (void)ud;
  (void)obs;
  CHECK(context != NULL);
  return rlmesh_value_int(0, out);
}
typedef struct Serving {
  RLMeshModel* model;
  RLMeshExtCancelToken* cancel;
  char address[80];
  RLMeshStatus status;
} Serving;
static void* serve_model(void* arg) {
  Serving* s = (Serving*)arg;
  RLMeshModelServeOptions o = RLMESH_MODEL_SERVE_OPTIONS_INIT;
  o.cancel = s->cancel;
  s->status = rlmesh_model_serve(s->model, s->address, &o);
  return NULL;
}
static char* text(RLMeshBytes bytes) {
  char* s = (char*)malloc(bytes.len + 1);
  memcpy(s, bytes.data, bytes.len);
  s[bytes.len] = 0;
  rlmesh_bytes_free(bytes);
  return s;
}
int main(void) {
  RLMeshRemoteEnv* remote = NULL;
  RLMeshRemoteVectorEnv* vector = NULL;
  RLMeshRemoteModel* rm = NULL;
  CHECK(rlmesh_remote_env_new(NULL, &remote) == RLMESH_ERR_INVALID_ARGUMENT && remote == NULL);
  CHECK(rlmesh_remote_model_new(NULL, &rm) == RLMESH_ERR_INVALID_ARGUMENT && rm == NULL);
  RLMeshRemoteEnvOptions eo = RLMESH_REMOTE_ENV_OPTIONS_INIT;
  eo.address = "tcp://127.0.0.1:1";
  eo.has_connect_timeout_seconds = true;
  eo.connect_timeout_seconds = NAN;
  CHECK(rlmesh_remote_env_new(&eo, &remote) == RLMESH_ERR_INVALID_ARGUMENT && remote == NULL);
  eo.connect_timeout_seconds = 0;
  eo.has_transport = true;
  eo.transport = RLMESH_TRANSPORT_TCP;
  CHECK(rlmesh_remote_env_new(&eo, &remote) == RLMESH_ERR_INVALID_ARGUMENT);
  eo.has_transport = false;
  CHECK(rlmesh_ext_remote_env_new_with_token(&eo, " hidden-endpoint-token", &remote) ==
        RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(strstr(rlmesh_last_error_message(), "hidden-endpoint-token") == NULL);
  RLMeshRemoteModelOptions mo = RLMESH_REMOTE_MODEL_OPTIONS_INIT;
  mo.address = "127.0.0.1:1";
  CHECK(rlmesh_remote_model_new(&mo, &rm) == RLMESH_OK);
  RLMeshBytes bytes = {NULL, 0, 0};
  CHECK(rlmesh_remote_model_address(rm, &bytes) == RLMESH_OK);
  char* address = text(bytes);
  CHECK(strcmp(address, mo.address) == 0);
  free(address);
  rlmesh_remote_model_free(rm);
  RLMeshDiscrete* space = NULL;
  CHECK(rlmesh_discrete_new(3, NULL, &space) == RLMESH_OK);
  RLMeshEnvVtable ev = RLMESH_ENV_VTABLE_INIT;
  ev.reset = reset;
  ev.step = step;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = space;
  attrs.action_space = space;
  attrs.id = "C-Remote-v1";
  State state = {0, NULL};
  RLMeshEnv* env = NULL;
  CHECK(rlmesh_env_new(&ev, &state, &attrs, &env) == RLMESH_OK);
  RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
  so.address = "tcp://127.0.0.1:0";
  so.close_env_on_shutdown = false;
  RLMeshServeOptions serve = RLMESH_SERVE_OPTIONS_INIT;
  serve.allow_remote_shutdown = true;
  so.options = &serve;
  RLMeshEnvServer* server = NULL;
  CHECK(rlmesh_env_server_new(&env, 1, &so, &server) == RLMESH_OK);
  CHECK(rlmesh_env_server_start(server) == RLMESH_OK);
  CHECK(rlmesh_env_server_address(server, &bytes) == RLMESH_OK);
  address = text(bytes);
  eo.address = address;
  eo.workflow_edition = rlmesh_current_workflow_edition();
  CHECK(rlmesh_remote_env_new(&eo, &remote) == RLMESH_OK);
  state.remote = remote;
  CHECK(rlmesh_remote_vector_env_new(&eo, &vector) == RLMESH_ERR_INVALID_ARGUMENT &&
        vector == NULL);
  const RLMeshSpace* cached = rlmesh_remote_env_observation_space(remote);
  CHECK(cached == rlmesh_remote_env_observation_space(remote));
  CHECK(rlmesh_env_contract_num_envs(rlmesh_remote_env_env_contract(remote)) == 1);
  bool has = true;
  RLMeshBytes untouched = {NULL, 43, 0};
  CHECK(rlmesh_remote_env_render_mode(remote, &has, &untouched) == RLMESH_OK && !has &&
        untouched.len == 43);
  RLMeshExtCancelToken* token = NULL;
  CHECK(rlmesh_remote_env_cancel_handle(remote, &token) == RLMESH_OK);
  rlmesh_ext_cancel_token_free(token);
  RLMeshValue *obs = NULL, *info = NULL, *action = NULL;
  CHECK(rlmesh_remote_env_reset(remote, NULL, NULL, &obs, &info) == RLMESH_OK);
  const RLMeshValue* ids = NULL;
  CHECK(rlmesh_value_dict_get(info, rlmesh_str("episode_ids"), &ids) == RLMESH_OK && ids);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  CHECK(rlmesh_value_int(0, &action) == RLMESH_OK);
  double reward = 0;
  bool terminated = false, truncated = true;
  CHECK(rlmesh_remote_env_step(remote, action, &obs, &reward, &terminated, &truncated, &info) ==
            RLMESH_OK &&
        reward == 2 && terminated && !truncated);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  CHECK(rlmesh_remote_env_render(remote, UINT64_MAX, &obs) == RLMESH_ERR_INVALID_ARGUMENT &&
        obs == NULL);
  RLMeshSession* session = NULL;
  CHECK(rlmesh_session_new(RLMESH_RANDOM_SAMPLE, rlmesh_remote_env_target(remote), NULL,
                           &session) == RLMESH_OK);
  CHECK(rlmesh_remote_env_reset(remote, NULL, NULL, &obs, &info) == RLMESH_ERR_RUNTIME);
  CHECK(rlmesh_remote_env_observation_space(remote) == cached);
  CHECK(rlmesh_session_reset(session, NULL, NULL, &obs, &info) == RLMESH_OK);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  CHECK(rlmesh_session_close(session) == RLMESH_OK);
  rlmesh_session_free(session);
  CHECK(rlmesh_remote_env_reset(remote, NULL, NULL, &obs, &info) == RLMESH_OK);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  CHECK(rlmesh_remote_env_close(remote) == RLMESH_OK);
  CHECK(rlmesh_remote_env_close(remote) == RLMESH_OK);
  CHECK(rlmesh_remote_env_reset(remote, NULL, NULL, &obs, &info) == RLMESH_ERR_RUNTIME &&
        obs == NULL && info == NULL);
  CHECK(cached == rlmesh_remote_env_observation_space(remote));
  state.remote = NULL;
  rlmesh_remote_env_free(remote);
  /* RemoteModel keeps normalized input, clones, and sessions dial the facade. */
  RLMeshModelVtable mv = RLMESH_MODEL_VTABLE_INIT;
  mv.predict = predict;
  mv.context_corners = RLMESH_CORNER_PREDICT;
  mv.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshModel* model = NULL;
  CHECK(rlmesh_model_new(&mv, NULL, NULL, &model) == RLMESH_OK);
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(sock >= 0);
  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(bind(sock, (struct sockaddr*)&sa, sizeof(sa)) == 0);
  socklen_t salen = sizeof(sa);
  CHECK(getsockname(sock, (struct sockaddr*)&sa, &salen) == 0);
  close(sock);
  Serving work = {model, NULL, {0}, RLMESH_OK};
  CHECK(rlmesh_ext_cancel_token_new(&work.cancel) == RLMESH_OK);
  snprintf(work.address, sizeof(work.address), "tcp://127.0.0.1:%u", (unsigned)ntohs(sa.sin_port));
  pthread_t thread;
  CHECK(pthread_create(&thread, NULL, serve_model, &work) == 0);
  mo.address = work.address;
  mo.has_connect_timeout_seconds = true;
  mo.connect_timeout_seconds = 0.1;
  CHECK(rlmesh_remote_model_new(&mo, &rm) == RLMESH_OK);
  RLMeshRemoteModel* clone = NULL;
  CHECK(rlmesh_remote_model_clone(rm, &clone) == RLMESH_OK);
  rlmesh_remote_model_free(rm);
  rm = clone;
  RLMeshEnv* local = NULL;
  CHECK(rlmesh_env_new(&ev, &state, &attrs, &local) == RLMESH_OK);
  CHECK(rlmesh_remote_model_session(rm, rlmesh_env_target(local), NULL, &session) ==
            RLMESH_ERR_TYPE &&
        session == NULL);
  rlmesh_env_free(local);
  struct timespec delay = {0, 1000000};
  RLMeshStatus connected = RLMESH_ERR_CONNECTION;
  for (int i = 0; i < 500 && connected == RLMESH_ERR_CONNECTION; ++i) {
    connected = rlmesh_remote_model_session(rm, rlmesh_env_server_target(server), NULL, &session);
    if (connected == RLMESH_ERR_CONNECTION) nanosleep(&delay, NULL);
  }
  CHECK(connected == RLMESH_OK && session);
  RLMeshRunResult* result = NULL;
  CHECK(rlmesh_session_run(session, NULL, &result) == RLMESH_OK);
  CHECK(rlmesh_run_result_num_episodes(result) == 1 && rlmesh_run_result_total_steps(result) == 1);
  rlmesh_run_result_free(result);
  CHECK(rlmesh_session_close(session) == RLMESH_OK);
  rlmesh_session_free(session);
  rlmesh_remote_model_free(rm);
  CHECK(rlmesh_ext_cancel_token_cancel(work.cancel) == RLMESH_OK);
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(work.status == RLMESH_ERR_CANCELLED);
  rlmesh_ext_cancel_token_free(work.cancel);
  rlmesh_model_free(model);
  free(address);
  CHECK(rlmesh_env_server_shutdown(server) == RLMESH_OK);
  rlmesh_env_server_free(server);
  /* Two scalar authored lanes form the vector client. */
  State second = {0, NULL};
  RLMeshEnv* env2 = NULL;
  CHECK(rlmesh_env_new(&ev, &second, &attrs, &env2) == RLMESH_OK);
  RLMeshEnv* lanes[] = {env, env2};
  CHECK(rlmesh_env_server_new(lanes, 2, &so, &server) == RLMESH_OK);
  CHECK(rlmesh_env_server_start(server) == RLMESH_OK);
  CHECK(rlmesh_env_server_address(server, &bytes) == RLMESH_OK);
  address = text(bytes);
  eo.address = address;
  CHECK(rlmesh_remote_env_new(&eo, &remote) == RLMESH_ERR_INVALID_ARGUMENT && remote == NULL);
  CHECK(rlmesh_remote_vector_env_new(&eo, &vector) == RLMESH_OK);
  CHECK(rlmesh_remote_vector_env_num_envs(vector) == 2);
  CHECK(rlmesh_remote_vector_env_single_action_space(vector) ==
        rlmesh_remote_vector_env_action_space(vector));
  CHECK(rlmesh_remote_vector_env_reset(vector, NULL, 0, NULL, &obs, &info) == RLMESH_OK);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  int64_t data[] = {0, 1};
  size_t shape[] = {2};
  RLMeshTensor* tensor = NULL;
  RLMeshValue* actions = NULL;
  CHECK(rlmesh_tensor_new(data, sizeof(data), shape, 1, RLMESH_DTYPE_INT64, &tensor) == RLMESH_OK);
  CHECK(rlmesh_value_tensor(tensor, &actions) == RLMESH_OK);
  rlmesh_tensor_free(tensor);
  double rewards[] = {-1, -1};
  bool terms[] = {false, false}, truncs[] = {true, true};
  size_t len = 99;
  obs = action;
  info = action;
  int before = atomic_load(&state.steps) + atomic_load(&second.steps);
  CHECK(rlmesh_remote_vector_env_step(vector, actions, &obs, rewards, terms, truncs, 1, &len,
                                      &info) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(obs == action && info == action && len == 99 && rewards[0] == -1 &&
        before == atomic_load(&state.steps) + atomic_load(&second.steps));
  CHECK(rlmesh_remote_vector_env_step(vector, actions, &obs, rewards, terms, truncs, 2, &len,
                                      &info) == RLMESH_OK &&
        len == 2 && rewards[0] == 2 && rewards[1] == 2 && terms[0] && terms[1]);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  rlmesh_value_free(actions);
  rlmesh_value_free(action);
  bool accepted = false;
  CHECK(rlmesh_remote_vector_env_shutdown(vector, NULL, &accepted) == RLMESH_OK && accepted);
  CHECK(rlmesh_remote_vector_env_render(vector, 0, &obs) == RLMESH_ERR_RUNTIME);
  CHECK(rlmesh_remote_vector_env_close(vector) == RLMESH_OK);
  rlmesh_remote_vector_env_free(vector);
  rlmesh_env_server_free(server);
  free(address);
  rlmesh_env_free(env2);
  rlmesh_env_free(env);
  rlmesh_space_free(space);
  if (atomic_load(&failures) == 0) puts("C remote facade OK");
  return atomic_load(&failures) ? 1 : 0;
}
