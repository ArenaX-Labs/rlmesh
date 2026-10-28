/* Factory start ownership/failures, factory host identity, and model serve signals. */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <pthread.h>
#include <rlmesh.h>
#include <signal.h>
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
      ++failures;                                                                               \
    }                                                                                           \
  } while (0)

typedef struct State {
  RLMeshSpace* space;
  RLMeshExtCancelToken* cancel;
  atomic_int makes, env_closes, env_frees, factory_frees, prepares;
  int fail_make, fail_stamp, cancel_make, fail_after_make, fail_close;
  bool both_outputs, cancel_close;
  RLMeshEnv* returned_env;
  RLMeshEnvFactory* reentrant;
} State;
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
  (void)info;
  *reward = 2;
  *term = true;
  *trunc = false;
  return rlmesh_value_int(1, obs);
}
static RLMeshStatus env_close(void* u) {
  State* s = (State*)u;
  ++s->env_closes;
  if (s->cancel_close) {
    CHECK(rlmesh_ext_cancel_token_cancel(s->cancel) == RLMESH_OK);
    struct timespec delay = {0, 60000000};
    nanosleep(&delay, NULL);
  }
  if (s->fail_close) {
    rlmesh_callback_set_error("close failure during cancellation", false);
    return RLMESH_ERR_ENVIRONMENT;
  }
  return RLMESH_OK;
}
static void env_free(void* u) { ++((State*)u)->env_frees; }
static void factory_free(void* u) { ++((State*)u)->factory_frees; }
static RLMeshStatus prepare(void* u) {
  State* s = (State*)u;
  ++s->prepares;
  if (s->reentrant) {
    RLMeshEnvServer* out = (RLMeshEnvServer*)1;
    CHECK(rlmesh_ext_env_factory_start(s->reentrant, ":0/env", NULL, &out) == RLMESH_ERR_RUNTIME &&
          !out);
  }
  return RLMESH_OK;
}
static RLMeshStatus tags_for(void* u, const RLMeshValue* p, RLMeshEnvTags** tags) {
  State* s = (State*)u;
  (void)p;
  (void)tags;
  if (s->fail_stamp && s->makes >= s->fail_stamp) {
    rlmesh_callback_set_error("stamp failure", false);
    return RLMESH_ERR_ENVIRONMENT;
  }
  return RLMESH_OK;
}
static RLMeshStatus vector_reset(void* u, const int64_t* seeds, size_t len,
                                 const RLMeshValue* options, RLMeshValue** obs,
                                 RLMeshValue** info) {
  (void)len;
  return reset(u, seeds, options, obs, info);
}
static RLMeshStatus vector_step(void* u, const RLMeshValue* actions, RLMeshValue** obs,
                                double* rewards, bool* terms, bool* truncs, RLMeshValue** info) {
  return step(u, actions, obs, rewards, terms, truncs, info);
}
static RLMeshStatus make(void* u, const RLMeshValue* k, RLMeshEnv** env, RLMeshVectorEnv** v) {
  State* s = (State*)u;
  (void)k;
  int n = ++s->makes;
  if (n == s->fail_make) {
    rlmesh_callback_set_error("make failure", false);
    return RLMESH_ERR_ENVIRONMENT;
  }
  RLMeshEnvVtable vt = RLMESH_ENV_VTABLE_INIT;
  vt.reset = reset;
  vt.step = step;
  vt.close = env_close;
  vt.free_user_data = env_free;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = s->space;
  attrs.action_space = s->space;
  attrs.id = "Start-v1";
  RLMeshStatus result;
  if (s->returned_env) {
    *env = s->returned_env;
    s->returned_env = NULL;
    result = RLMESH_OK;
  } else
    result = rlmesh_env_new(&vt, s, &attrs, env);
  if (s->both_outputs) {
    RLMeshVectorEnvVtable vv = RLMESH_VECTOR_ENV_VTABLE_INIT;
    vv.reset = vector_reset;
    vv.step = vector_step;
    vv.close = env_close;
    vv.free_user_data = env_free;
    RLMeshVectorEnvAttributes va = RLMESH_VECTOR_ENV_ATTRIBUTES_INIT;
    va.num_envs = 2;
    va.single_observation_space = va.single_action_space = s->space;
    CHECK(rlmesh_vector_env_new(&vv, s, &va, v) == RLMESH_OK);
  }
  if (n == s->cancel_make) CHECK(rlmesh_ext_cancel_token_cancel(s->cancel) == RLMESH_OK);
  if (n == s->fail_after_make) {
    rlmesh_callback_set_error("make returned env and error", false);
    return RLMESH_ERR_ENVIRONMENT;
  }
  return result;
}
static RLMeshEnvFactory* factory(State* s) {
  RLMeshEnvFactoryVtable vt = RLMESH_ENV_FACTORY_VTABLE_INIT;
  vt.make = make;
  vt.prepare = prepare;
  vt.tags_for = tags_for;
  vt.free_user_data = factory_free;
  RLMeshValue *one = NULL, *choices = NULL;
  CHECK(rlmesh_value_int(1, &one) == RLMESH_OK);
  const RLMeshValue* values[] = {one};
  CHECK(rlmesh_value_list(values, 1, &choices) == RLMESH_OK);
  RLMeshTagParam param = RLMESH_TAG_PARAM_INIT;
  param.name = "task";
  param.default_ = one;
  param.choices = choices;
  const RLMeshTagParam* params[] = {&param};
  RLMeshEnvFactoryAttributes attrs = RLMESH_ENV_FACTORY_ATTRIBUTES_INIT;
  attrs.tag_params = params;
  attrs.tag_params_len = 1;
  attrs.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshEnvFactory* f = NULL;
  CHECK(rlmesh_env_factory_new(&vt, s, &attrs, &f) == RLMESH_OK);
  rlmesh_value_free(one);
  rlmesh_value_free(choices);
  return f;
}
/* Steps one episode of a binding. */
static void step_bound(RLMeshBinding* binding) {
  RLMeshValue *obs = NULL, *info = NULL;
  CHECK(rlmesh_binding_reset(binding, NULL, NULL, &obs, &info) == RLMESH_OK);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  double reward = 0;
  bool term = false, trunc = false;
  CHECK(rlmesh_binding_step(binding, NULL, &obs, &reward, &term, &trunc, &info) == RLMESH_OK);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
}
static void drive(RLMeshEnvServer* server) {
  RLMeshRunResult* result = NULL;
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_server_target(server), NULL, &result) ==
        RLMESH_OK);
  CHECK(rlmesh_run_result_num_episodes(result) == 1 && rlmesh_run_result_total_steps(result) == 1);
  rlmesh_run_result_free(result);
}
static void stop(RLMeshEnvServer* server) {
  CHECK(rlmesh_env_server_shutdown(server) == RLMESH_OK);
  double timeout = 2;
  bool stopped = false;
  CHECK(rlmesh_env_server_wait(server, &timeout, &stopped) == RLMESH_OK && stopped);
  rlmesh_env_server_free(server);
}
/* Keep a live listener to force a bind failure after all envs have been made. */
static int occupied(char* address, size_t cap) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(bind(fd, (struct sockaddr*)&sa, sizeof(sa)) == 0 && listen(fd, 1) == 0);
  socklen_t len = sizeof(sa);
  CHECK(getsockname(fd, (struct sockaddr*)&sa, &len) == 0);
  snprintf(address, cap, "tcp://127.0.0.1:%u", (unsigned)ntohs(sa.sin_port));
  return fd;
}
/* The real old allocation, rather than a new struct with a shorter size. */
typedef struct OldFactoryOptions {
  size_t struct_size;
  int64_t num_envs;
  const char* workflow_edition;
  const RLMeshValue* make_kwargs;
  RLMeshExtCancelToken* cancel;
} OldFactoryOptions;
typedef struct OldModelOptions {
  size_t struct_size;
  const char* token;
  size_t predict_concurrency;
  bool has_predict_concurrency;
  uint8_t reserved[sizeof(size_t) - 1];
} OldModelOptions;
_Static_assert(RLMESH_ENV_FACTORY_SERVE_OPTIONS_MIN_SIZE == sizeof(OldFactoryOptions),
               "factory ABI5 minimum");
_Static_assert(RLMESH_EXT_MODEL_SERVE_OPTIONS_MIN_SIZE == sizeof(OldModelOptions),
               "model ABI5 minimum");
_Static_assert(RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_MIN_SIZE ==
                   sizeof(RLMeshExtEnvFactoryServeOptions),
               "new ext minimum");
_Static_assert(offsetof(RLMeshEnvFactoryServeOptions, ext) == sizeof(OldFactoryOptions) + 8,
               "factory appended pointer");
#if UINTPTR_MAX == UINT32_MAX
_Static_assert(sizeof(RLMeshEnvFactoryServeOptions) ==
                   offsetof(RLMeshEnvFactoryServeOptions, _reserved1) + 4,
               "factory explicit 32-bit tail");
#else
_Static_assert(sizeof(RLMeshEnvFactoryServeOptions) ==
                   offsetof(RLMeshEnvFactoryServeOptions, ext) + sizeof(void*),
               "factory no tail padding");
#endif
_Static_assert(sizeof(RLMeshExtModelServeOptions) == sizeof(OldModelOptions) + sizeof(size_t),
               "model no tail padding");
#if SIZE_MAX == UINT64_MAX
_Static_assert(sizeof(OldFactoryOptions) == 40 && sizeof(RLMeshEnvFactoryServeOptions) == 56,
               "factory 64-bit layout");
_Static_assert(sizeof(OldModelOptions) == 32 && sizeof(RLMeshExtModelServeOptions) == 40,
               "model 64-bit layout");
_Static_assert(sizeof(RLMeshExtEnvFactoryServeOptions) == 16, "host ext 64-bit layout");
#endif

static void wait_cancelled(RLMeshEnvServer* server) {
  double timeout = 5;
  bool stopped = false;
  CHECK(rlmesh_env_server_wait(server, &timeout, &stopped) == RLMESH_ERR_CANCELLED && stopped);
  stopped = false;
  CHECK(rlmesh_env_server_wait(server, &timeout, &stopped) == RLMESH_OK && stopped);
}
typedef struct ForegroundDriver {
  State* state;
  int made;
  RLMeshEnvServer* server;
  RLMeshExtCancelToken* cancel;
} ForegroundDriver;
static void* foreground_driver(void* u) {
  ForegroundDriver* d = (ForegroundDriver*)u;
  struct timespec delay = {0, 1000000};
  int attempts = 0;
  while (d->state->makes <= d->made && ++attempts < 5000) nanosleep(&delay, NULL);
  CHECK(d->state->makes > d->made);
  drive(d->server);
  CHECK(rlmesh_ext_cancel_token_cancel(d->cancel) == RLMESH_OK);
  return NULL;
}
static void rejected_output_tests(State* original) {
  State s = {0};
  s.space = original->space;
  s.cancel = original->cancel;
  RLMeshEnvFactory* f = factory(&s);
  RLMeshEnvServer* server = NULL;
  s.both_outputs = true;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", NULL, &server) == RLMESH_ERR_TYPE && !server);
  CHECK(s.env_closes == 2 && s.env_frees == 2);
  s.both_outputs = false;
  s.fail_after_make = s.makes + 1;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", NULL, &server) == RLMESH_ERR_ENVIRONMENT &&
        !server);
  CHECK(s.env_closes == 3 && s.env_frees == 3);
  /* Foreground and binding borrows retain their existing free-only behaviour. */
  s.fail_after_make = s.makes + 1;
  CHECK(rlmesh_env_factory_serve(f, ":0/env", NULL) == RLMESH_ERR_ENVIRONMENT);
  CHECK(s.env_closes == 3 && s.env_frees == 4);
  s.fail_after_make = s.makes + 1;
  RLMeshBinding* binding = NULL;
  CHECK(rlmesh_bind(RLMESH_RANDOM_SAMPLE, rlmesh_env_factory_target(f), NULL, &binding) ==
            RLMESH_ERR_ENVIRONMENT &&
        !binding);
  CHECK(s.env_closes == 3 && s.env_frees == 5);
  s.fail_after_make = 0;
  RLMeshEnv* env = NULL;
  RLMeshVectorEnv* vector = NULL;
  CHECK(rlmesh_env_factory_make(f, NULL, &env, &vector) == RLMESH_OK && env && !vector);
  RLMeshEnvServerOptions opts = RLMESH_ENV_SERVER_OPTIONS_INIT;
  opts.address = ":0/env";
  RLMeshEnvServer* owner = NULL;
  CHECK(rlmesh_env_server_new(&env, 1, &opts, &owner) == RLMESH_OK);
  CHECK(rlmesh_env_server_start(owner) == RLMESH_OK);
  s.returned_env = env; /* Transfers the handle while the server holds its lease. */
  s.fail_after_make = s.makes + 1;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", NULL, &server) == RLMESH_ERR_ENVIRONMENT &&
        !server);
  CHECK(s.env_closes == 3 && s.env_frees == 5);
  drive(owner);
  stop(owner);
  CHECK(s.env_closes == 4 && s.env_frees == 6);
  rlmesh_env_factory_free(f);
}
static void start_tests(State* s) {
  RLMeshEnvFactory* f = factory(s);
  RLMeshEnvFactoryServeOptions o = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
  RLMeshExtEnvFactoryServeOptions ext = RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_INIT;
  ext.host_runtime_json =
      "{\"language\":\"csharp\",\"language_version\":\"9.0\",\"framework_versions\":{\"unity\":"
      "\"6\"}}";
  o.ext = &ext;
  RLMeshEnvServer *a = NULL, *b = NULL;
  struct sigaction before_int, before_term, after;
  CHECK(sigaction(SIGINT, NULL, &before_int) == 0 && sigaction(SIGTERM, NULL, &before_term) == 0);
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
  CHECK(sigaction(SIGINT, NULL, &after) == 0 && after.sa_handler == before_int.sa_handler);
  CHECK(sigaction(SIGTERM, NULL, &after) == 0 && after.sa_handler == before_term.sa_handler);
  RLMeshBytes json = {NULL, 0, 0};
  CHECK(rlmesh_ext_env_server_describe_json(a, &json) == RLMESH_OK);
  char* text = (char*)calloc(json.len + 1, 1);
  memcpy(text, json.data, json.len);
  CHECK(strstr(text, "csharp") && strstr(text, "unity"));
  free(text);
  rlmesh_bytes_free(json);
  /* The second start must not steal the first server's drain. */
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &b) == RLMESH_OK);
  drive(a);
  drive(b);
  stop(a);
  drive(b);
  stop(b);
  CHECK(s->env_closes == 2 && s->env_frees == 2);
  /* Hook reentry refuses without disturbing the outer start or its scope. */
  s->reentrant = f;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", NULL, &a) == RLMESH_OK);
  s->reentrant = NULL;
  RLMeshEnv* remade = NULL;
  RLMeshVectorEnv* vector = NULL;
  CHECK(rlmesh_env_factory_make(f, NULL, &remade, &vector) == RLMESH_OK && remade && !vector);
  CHECK(rlmesh_env_close(remade) == RLMESH_OK);
  rlmesh_env_free(remade);
  RLMeshExtCancelToken* foreground_cancel = NULL;
  CHECK(rlmesh_ext_cancel_token_new(&foreground_cancel) == RLMESH_OK);
  ForegroundDriver driver = {s, s->makes, a, foreground_cancel};
  pthread_t driver_thread;
  CHECK(pthread_create(&driver_thread, NULL, foreground_driver, &driver) == 0);
  RLMeshEnvFactoryServeOptions foreground = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
  foreground.cancel = foreground_cancel;
  CHECK(rlmesh_env_factory_serve(f, ":0/env", &foreground) == RLMESH_ERR_CANCELLED);
  CHECK(pthread_join(driver_thread, NULL) == 0);
  rlmesh_ext_cancel_token_free(foreground_cancel);
  drive(a);
  rlmesh_env_server_free(a); /* Free alone must stop and close its owned envs. */
  CHECK(s->env_closes == s->makes && s->env_frees == s->makes);
  /* A binding's admission must refuse without replacing its gate slot. */
  RLMeshBinding* binding = NULL;
  CHECK(rlmesh_bind(RLMESH_RANDOM_SAMPLE, rlmesh_env_factory_target(f), NULL, &binding) ==
        RLMESH_OK);
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_RUNTIME && !a);
  step_bound(binding);
  rlmesh_binding_free(binding);
  CHECK(s->env_closes == s->makes && s->env_frees == s->makes);
  /* No invalid input may consume this firing, including a host extension. */
  CHECK(rlmesh_ext_cancel_token_cancel(s->cancel) == RLMESH_OK);
  o.cancel = s->cancel;
  const char* invalid[] = {"[]", "{\"unexpected\":1}",
                           "{\"toolchain\":{\"name\":\"mono\",\"version\":\"1.0\"}}"};
  int prepares = s->prepares;
  for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); ++i) {
    ext.host_runtime_json = invalid[i];
    CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_INVALID_ARGUMENT && !a);
    CHECK(s->prepares == prepares);
  }
  ext.host_runtime_json = "{}";
  o._reserved = 1;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_INVALID_ARGUMENT && !a);
  o._reserved = 0;
  o.struct_size = RLMESH_ENV_FACTORY_SERVE_OPTIONS_MIN_SIZE + 1;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_INVALID_ARGUMENT && !a);
  o.struct_size = sizeof(o);
  ext.struct_size = 0;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_INVALID_ARGUMENT && !a);
  ext.struct_size = sizeof(ext);
  o.workflow_edition = "invalid-edition";
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_INVALID_ARGUMENT && !a);
  o.workflow_edition = NULL;
  CHECK(rlmesh_ext_env_factory_start(f, "tcp://127.0.0.1:not-a-port", &o, &a) ==
            RLMESH_ERR_INVALID_ARGUMENT &&
        !a);
  RLMeshValue *bad = NULL, *bad_kwargs = NULL;
  CHECK(rlmesh_value_int(99, &bad) == RLMESH_OK);
  RLMeshStr key = rlmesh_str("task");
  const RLMeshValue* values[] = {bad};
  CHECK(rlmesh_value_dict(&key, values, 1, &bad_kwargs) == RLMESH_OK);
  o.make_kwargs = bad_kwargs;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_INVALID_ARGUMENT && !a);
  o.make_kwargs = NULL;
  rlmesh_value_free(bad_kwargs);
  rlmesh_value_free(bad);
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_CANCELLED && !a);
  CHECK(s->prepares == prepares);
  /* Every partial build is drained exactly once, and admission is released. */
  int made = s->makes;
  for (int failure = 0; failure < 6; ++failure) {
    int before = s->env_closes;
    o.num_envs = 3;
    s->fail_make = failure == 0 ? s->makes + 2 : 0;
    s->fail_stamp = failure == 1 ? s->makes + 2 : 0;
    s->cancel_make = (failure == 3 || failure == 5) ? s->makes + 2 : 0;
    s->fail_close = failure == 5;
    s->fail_after_make = failure == 4 ? s->makes + 2 : 0;
    char address[80];
    int fd = failure == 2 ? occupied(address, sizeof(address)) : -1;
    RLMeshStatus status = rlmesh_ext_env_factory_start(f, fd < 0 ? ":0/env" : address, &o, &a);
    CHECK(status == ((failure == 3 || failure == 5) ? RLMESH_ERR_CANCELLED
                     : failure == 2                 ? RLMESH_ERR_CONNECTION
                                                    : RLMESH_ERR_ENVIRONMENT) &&
          !a);
    if (fd >= 0) close(fd);
    CHECK(s->env_closes == before + (failure == 0 ? 1 : (failure == 1 || failure == 4) ? 2 : 3));
    CHECK(s->env_closes == s->env_frees);
    s->fail_make = s->fail_stamp = s->cancel_make = s->fail_after_make = s->fail_close = 0;
    o.num_envs = 1;
    CHECK(rlmesh_env_factory_prepare(f) == RLMESH_OK);
    CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
    drive(a);
    stop(a);
    CHECK(s->env_closes == s->env_frees);
  }
  CHECK(s->makes > made);
  /* shutdown consumes the same background result as wait, exactly once. */
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
  CHECK(rlmesh_ext_cancel_token_cancel(s->cancel) == RLMESH_OK);
  CHECK(rlmesh_env_server_shutdown(a) == RLMESH_ERR_CANCELLED);
  CHECK(rlmesh_env_server_shutdown(a) == RLMESH_OK);
  double stopped_timeout = 1;
  bool stopped = false;
  CHECK(rlmesh_env_server_wait(a, &stopped_timeout, &stopped) == RLMESH_OK && stopped);
  rlmesh_env_server_free(a);
  CHECK(s->env_closes == s->env_frees);
  /* A firing during slow close loses to an earlier plain shutdown. */
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
  s->cancel_close = true;
  stop(a);
  s->cancel_close = false;
  CHECK(s->env_closes == s->env_frees);
  /* With no firing, the caller's own shutdown is successful. */
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
  stop(a);
  CHECK(s->env_closes == s->env_frees);
  /* Preparation must observe a firing even when another server is already
   * subscribed, so cancel() does not set the token's idle pending flag. */
  int closed_before = s->env_closes;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
  s->cancel_make = s->makes + 2;
  o.num_envs = 3;
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &b) == RLMESH_ERR_CANCELLED && !b);
  s->cancel_make = 0;
  o.num_envs = 1;
  wait_cancelled(a);
  stop(a);
  CHECK(s->env_closes == closed_before + 4 && s->env_closes == s->env_frees);
  CHECK(rlmesh_env_factory_prepare(f) == RLMESH_OK);
  /* A shared token broadcasts to both live subscriptions, rather than one
   * server consuming an idle firing intended for both. */
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &b) == RLMESH_OK);
  CHECK(rlmesh_ext_cancel_token_cancel(s->cancel) == RLMESH_OK);
  wait_cancelled(a);
  wait_cancelled(b);
  stop(a);
  stop(b);
  /* Every failed preparation and terminal serve released its subscription:
   * an idle firing must now fail the next start before any prepare hook. */
  prepares = s->prepares;
  CHECK(rlmesh_ext_cancel_token_cancel(s->cancel) == RLMESH_OK);
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_ERR_CANCELLED && !a);
  CHECK(s->prepares == prepares);
  OldFactoryOptions old = {sizeof(old), 1, NULL, NULL, NULL};
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", (const RLMeshEnvFactoryServeOptions*)&old, &a) ==
        RLMESH_OK);
  drive(a);
  stop(a);
  CHECK(rlmesh_ext_env_factory_start(f, ":0/env", &o, &a) == RLMESH_OK);
  /* The server owns both its env drain and its token subscription. */
  RLMeshExtCancelToken* alias = NULL;
  CHECK(rlmesh_ext_cancel_token_clone(s->cancel, &alias) == RLMESH_OK);
  rlmesh_env_factory_free(f);
  CHECK(s->factory_frees == 1);
  rlmesh_ext_cancel_token_free(s->cancel);
  s->cancel = NULL;
  drive(a);
  CHECK(rlmesh_ext_cancel_token_cancel(alias) == RLMESH_OK);
  rlmesh_ext_cancel_token_free(alias);
  wait_cancelled(a);
  stop(a);
  CHECK(s->env_closes == s->env_frees);
}

static volatile sig_atomic_t host_ints;
static void host_signal(int sig) {
  if (sig == SIGINT) ++host_ints;
}
/* The participant address the model serves at, once its host session is up. */
static char served_at[256];
static atomic_bool hosting;
/* A run through the served model returns only once its serve has joined, past
 * where it installs (or skips) its signal handlers. */
static void wait_served(State* s) {
  struct timespec poll = {0, 10000000};
  while (!atomic_load(&hosting)) nanosleep(&poll, NULL);
  RLMeshModel* served = NULL;
  CHECK(rlmesh_model_from_address(served_at, &served) == RLMESH_OK);
  RLMeshEnv* env = NULL;
  RLMeshVectorEnv* vector = NULL;
  CHECK(make(s, NULL, &env, &vector) == RLMESH_OK);
  RLMeshRunOptions o = RLMESH_RUN_OPTIONS_INIT;
  o.close_env = true;
  RLMeshRunResult* result = NULL;
  CHECK(rlmesh_model_run(served, rlmesh_env_target(env), &o, &result) == RLMESH_OK);
  rlmesh_run_result_free(result);
  rlmesh_env_free(env);
  rlmesh_model_free(served);
}
typedef struct Commands {
  RLMeshExtCancelToken* cancel;
  bool no_handlers;
  State* model; /* set when a served model is probed */
} Commands;
static void* commands(void* u) {
  Commands* c = (Commands*)u;
  int command;
  while ((command = getchar()) != EOF) {
    if (command == 'i') {
      if (c->model) wait_served(c->model);
      struct sigaction action;
      CHECK(sigaction(SIGINT, NULL, &action) == 0);
      CHECK((action.sa_handler == host_signal) == c->no_handlers);
      CHECK(sigaction(SIGTERM, NULL, &action) == 0);
      CHECK((action.sa_handler == host_signal) == c->no_handlers);
      CHECK(raise(SIGINT) == 0);
      CHECK(host_ints == (c->no_handlers ? 1 : 0));
      puts("checked signals");
      fflush(stdout);
      if (!c->no_handlers) return NULL; /* SIGINT alone must stop the serve. */
    } else if (command == 'q')
      break;
  }
  CHECK(rlmesh_ext_cancel_token_cancel(c->cancel) == RLMESH_OK);
  return NULL;
}
static RLMeshStatus predict(void* u, const RLMeshValue* obs, const RLMeshPredictContext* c,
                            RLMeshValue** action) {
  (void)u;
  (void)obs;
  (void)c;
  return rlmesh_value_int(0, action);
}
int main(int argc, char** argv) {
  State state = {0};
  CHECK(rlmesh_discrete_new(4, NULL, &state.space) == RLMESH_OK);
  CHECK(rlmesh_ext_cancel_token_new(&state.cancel) == RLMESH_OK);
  if (argc == 1) {
    rejected_output_tests(&state);
    start_tests(&state);
  } else {
    CHECK(argc == 3);
    struct sigaction action = {0};
    action.sa_handler = host_signal;
    sigemptyset(&action.sa_mask);
    CHECK(sigaction(SIGINT, &action, NULL) == 0 && sigaction(SIGTERM, &action, NULL) == 0);
    bool is_factory = strstr(argv[1], "foreground") || strstr(argv[1], "start");
    bool background = strstr(argv[1], "start") || !strcmp(argv[1], "env-server");
    bool is_model = !is_factory && strcmp(argv[1], "env-server");
    Commands c = {state.cancel, !strcmp(argv[1], "no-signals") || background,
                  is_model ? &state : NULL};
    pthread_t reader;
    CHECK(pthread_create(&reader, NULL, commands, &c) == 0);
    if (is_factory) {
      RLMeshEnvFactory* f = factory(&state);
      RLMeshEnvFactoryServeOptions opts = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
      RLMeshExtEnvFactoryServeOptions ext = RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_INIT;
      ext.host_runtime_json =
          "{\"language\":\"csharp\",\"language_version\":\"9.0\",\"framework_versions\":{\"unity\":"
          "\"6\"}}";
      opts.ext = strstr(argv[1], "legacy") ? NULL : &ext;
      opts.cancel = state.cancel;
      if (!background) {
        CHECK(rlmesh_env_factory_serve(f, argv[2], &opts) == RLMESH_ERR_CANCELLED);
        rlmesh_env_factory_free(f);
      } else {
        RLMeshEnvServer* server = NULL;
        CHECK(rlmesh_ext_env_factory_start(f, argv[2], &opts, &server) == RLMESH_OK);
        rlmesh_env_factory_free(f);
        CHECK(state.factory_frees == 1);
        CHECK(pthread_join(reader, NULL) == 0);
        wait_cancelled(server);
        stop(server);
      }
    } else if (!strcmp(argv[1], "env-server")) {
      RLMeshEnv* env = NULL;
      RLMeshVectorEnv* vector = NULL;
      CHECK(make(&state, NULL, &env, &vector) == RLMESH_OK);
      RLMeshEnvServerOptions opts = RLMESH_ENV_SERVER_OPTIONS_INIT;
      RLMeshExtEnvServerOptions ext = RLMESH_EXT_ENV_SERVER_OPTIONS_INIT;
      ext.publish_describe = true;
      ext.host_runtime_json =
          "{\"language\":\"csharp\",\"language_version\":\"9.0\",\"framework_versions\":{\"unity\":"
          "\"6\"}}";
      opts.ext = &ext;
      opts.address = argv[2];
      RLMeshEnvServer* server = NULL;
      CHECK(rlmesh_env_server_new(&env, 1, &opts, &server) == RLMESH_OK);
      CHECK(rlmesh_env_server_start(server) == RLMESH_OK);
      CHECK(pthread_join(reader, NULL) == 0);
      stop(server);
      rlmesh_env_free(env);
    } else {
      /* The model joins a session this process hosts on argv[2]. */
      RLMeshEnv* host_env = NULL;
      RLMeshVectorEnv* host_vector = NULL;
      CHECK(make(&state, NULL, &host_env, &host_vector) == RLMESH_OK);
      RLMeshEnvServerOptions host_opts = RLMESH_ENV_SERVER_OPTIONS_INIT;
      host_opts.address = argv[2];
      RLMeshEnvServer* host = NULL;
      CHECK(rlmesh_env_server_new(&host_env, 1, &host_opts, &host) == RLMESH_OK);
      CHECK(rlmesh_env_server_start(host) == RLMESH_OK);
      RLMeshBytes bytes;
      CHECK(rlmesh_env_server_address(host, &bytes) == RLMESH_OK);
      size_t session = bytes.len;
      while (session && bytes.data[session - 1] != '/') --session;
      snprintf(served_at, sizeof(served_at), "%.*spolicy", (int)session, (const char*)bytes.data);
      rlmesh_bytes_free(bytes);
      atomic_store(&hosting, true);
      RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
      vt.predict = predict;
      vt.workflow_edition = rlmesh_current_workflow_edition();
      RLMeshModel* model = NULL;
      CHECK(rlmesh_model_new(&vt, NULL, NULL, &model) == RLMESH_OK);
      RLMeshModelServeOptions opts = RLMESH_MODEL_SERVE_OPTIONS_INIT;
      RLMeshExtModelServeOptions ext = RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT;
      ext.no_signal_handlers = c.no_handlers;
      OldModelOptions old = {sizeof(old), NULL, 0, false, {0}};
      opts.ext = !strcmp(argv[1], "old-signals") ? (const RLMeshExtModelServeOptions*)&old : &ext;
      opts.cancel = state.cancel;
      RLMeshExtModelServeOptions invalid = RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT;
      ((unsigned char*)&invalid)[offsetof(RLMeshExtModelServeOptions, no_signal_handlers)] = 2;
      opts.ext = &invalid;
      CHECK(rlmesh_model_serve(model, served_at, &opts) == RLMESH_ERR_INVALID_ARGUMENT);
      invalid.struct_size = RLMESH_EXT_MODEL_SERVE_OPTIONS_MIN_SIZE + 2;
      CHECK(rlmesh_model_serve(model, served_at, &opts) == RLMESH_ERR_INVALID_ARGUMENT);
      opts.ext = !strcmp(argv[1], "old-signals") ? (const RLMeshExtModelServeOptions*)&old : &ext;
      CHECK(rlmesh_model_serve(model, served_at, &opts) == RLMESH_ERR_CANCELLED);
      rlmesh_model_free(model);
      stop(host);
      rlmesh_env_free(host_env);
    }
    if (!background) CHECK(pthread_join(reader, NULL) == 0);
    CHECK(sigaction(SIGINT, NULL, &action) == 0 && action.sa_handler == host_signal);
    CHECK(sigaction(SIGTERM, NULL, &action) == 0 && action.sa_handler == host_signal);
    CHECK(state.env_closes == state.env_frees);
  }
  rlmesh_ext_cancel_token_free(state.cancel);
  rlmesh_space_free(state.space);
  if (!failures) puts("c_serving OK");
  return failures ? 1 : 0;
}
