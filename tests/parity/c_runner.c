#define _POSIX_C_SOURCE 200809L
/* Shared E-3 script: native C authoring callbacks and facade calls only. */
// Always evaluate checks and calls, including release consumers built with
// NDEBUG.
#define REQUIRE(...)                                                           \
  do {                                                                         \
    if (!(__VA_ARGS__))                                                        \
      abort();                                                                 \
  } while (0)
#include <pthread.h>
#include <rlmesh.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define OK(call)                                                               \
  do {                                                                         \
    RLMeshStatus status_ = (call);                                             \
    if (status_ != RLMESH_OK) {                                                \
      fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,            \
              rlmesh_last_error_message());                                    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

typedef struct Script {
  int steps, env_closes, model_closes, episode_ends, run_ends, partial;
  int cancel_step, starts, warnings, prepares;
  bool cancel_in_make;
  RLMeshSpace *space;
  RLMeshExtCancelToken *cancel;
  RLMeshErrorClass failure;
  const char *failure_text;
  int64_t reset_seed, reset_trial;
  pthread_t caller;
  bool warning_fail, model_recoverable, wrong_warning_thread;
  char category[64];
} Script;

static void quoted(const char *text) {
  putchar('"');
  if (text)
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
      if (*p == '"' || *p == '\\') {
        putchar('\\');
        putchar(*p);
      } else if (*p < 32)
        printf("\\u%04x", *p);
      else
        putchar(*p);
    }
  putchar('"');
}
/* The last error's text with every `value` (a port, a socket path) masked. */
static char *masked_message(const char *value, const char *mask) {
  const char *text = rlmesh_last_error_message();
  size_t len = strlen(text), vlen = value ? strlen(value) : 0;
  char *out = (char *)malloc(len * (strlen(mask ? mask : "") + 1) + 1);
  REQUIRE(out != NULL);
  char *w = out;
  for (const char *p = text; *p;) {
    if (vlen && !strncmp(p, value, vlen)) {
      w += sprintf(w, "%s", mask);
      p += vlen;
    } else
      *w++ = *p++;
  }
  *w = 0;
  return out;
}
static void error_fields_masked(const char *value, const char *mask);
static void error_fields(void) { error_fields_masked(NULL, NULL); }
static void error_fields_masked(const char *value, const char *mask) {
  RLMeshErrorClass cls = rlmesh_last_error_class();
  bool recoverable = rlmesh_last_error_is_recoverable();
  const char *code = rlmesh_last_error_code_str();
  printf("\"error\":");
  quoted(rlmesh_error_class_name(cls));
  printf(",\"text\":");
  char *message = masked_message(value, mask);
  quoted(message);
  free(message);
  printf(",\"code\":");
  quoted(code);
  printf(",\"recoverable\":%s,\"runtime_base\":%s,\"os_base\":%s,\"value_"
         "base\":%s",
         recoverable ? "true" : "false",
         rlmesh_error_class_is_subclass(cls, RLMESH_ERROR_CLASS_RUNTIME_ERROR)
             ? "true"
             : "false",
         rlmesh_error_class_is_subclass(cls, RLMESH_ERROR_CLASS_OS_ERROR)
             ? "true"
             : "false",
         rlmesh_error_class_is_subclass(cls, RLMESH_ERROR_CLASS_VALUE_ERROR)
             ? "true"
             : "false");
  printf(
      ",\"rlmesh_base\":%s,\"environment_base\":%s,\"self_base\":%s",
      rlmesh_error_class_is_subclass(cls, RLMESH_ERROR_CLASS_RLMESH_EXCEPTION)
          ? "true"
          : "false",
      rlmesh_error_class_is_subclass(cls,
                                     RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION)
          ? "true"
          : "false",
      rlmesh_error_class_is_subclass(cls, cls) ? "true" : "false");
}
static RLMeshStatus reset(void *ud, const int64_t *seed,
                          const RLMeshValue *options, RLMeshValue **obs,
                          RLMeshValue **info) {
  Script *s = (Script *)ud;
  s->reset_seed = seed ? *seed : -1;
  bool has_trial = false;
  OK(rlmesh_trial_index(options, NULL, &has_trial, &s->reset_trial));
  if (!has_trial)
    s->reset_trial = -1;
  (void)info;
  ++s->starts;
  s->steps = 0;
  return rlmesh_value_int(0, obs);
}
static RLMeshStatus step(void *ud, const RLMeshValue *action, RLMeshValue **obs,
                         double *reward, bool *terminated, bool *truncated,
                         RLMeshValue **info) {
  Script *s = (Script *)ud;
  (void)action;
  (void)info;
  ++s->starts;
  ++s->steps;
  if (s->cancel_step) {
    s->cancel_step = 0;
    OK(rlmesh_ext_cancel_token_cancel(s->cancel));
  }
  *reward = 1;
  *terminated = s->steps >= 2;
  *truncated = false;
  return rlmesh_value_int(s->steps, obs);
}
static RLMeshStatus close_env(void *ud) {
  ++((Script *)ud)->env_closes;
  return RLMESH_OK;
}
static RLMeshStatus close_model(void *ud) {
  ++((Script *)ud)->model_closes;
  return RLMESH_OK;
}
static RLMeshStatus episode_end(void *ud, const char *id) {
  (void)id;
  ++((Script *)ud)->episode_ends;
  return RLMESH_OK;
}
static RLMeshStatus predict(void *ud, const RLMeshValue *obs,
                            const RLMeshPredictContext *ctx,
                            RLMeshValue **action) {
  Script *s = (Script *)ud;
  (void)obs;
  (void)ctx;
  ++s->starts;
  if (s->failure) {
    if (s->model_recoverable) {
      rlmesh_callback_set_error("scripted failure", true);
      return RLMESH_ERR_MODEL;
    }
    rlmesh_callback_set_error_class(
        s->failure, s->failure_text ? s->failure_text
                    : s->failure == RLMESH_ERROR_CLASS_TIMEOUT_ERROR
                        ? "authored timeout: checkpoint shard violet"
                    : s->failure == RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT
                        ? "cancelled"
                        : "scripted failure");
    return RLMESH_ERR_INTERNAL;
  }
  return rlmesh_value_int(0, action);
}
static RLMeshStatus chunk(void *ud, const RLMeshValue *obs, uint32_t horizon,
                          const RLMeshPredictContext *ctx, RLMeshValue **out) {
  (void)horizon;
  RLMeshValue *one = NULL;
  OK(predict(ud, obs, ctx, &one));
  const RLMeshValue *values[] = {one};
  RLMeshStatus status = rlmesh_value_list(values, 1, out);
  rlmesh_value_free(one);
  return status;
}
static RLMeshStatus factory_prepare(void *ud) {
  ++((Script *)ud)->prepares;
  return RLMESH_OK;
}
static RLMeshStatus factory_make(void *ud, const RLMeshValue *kwargs,
                                 RLMeshEnv **out, RLMeshVectorEnv **vector) {
  Script *s = (Script *)ud;
  (void)kwargs;
  (void)vector;
  if (s->cancel_in_make) {
    s->cancel_in_make = false;
    OK(rlmesh_ext_cancel_token_cancel(s->cancel));
  }
  RLMeshEnvVtable ev = RLMESH_ENV_VTABLE_INIT;
  ev.reset = reset;
  ev.step = step;
  ev.close = close_env;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = attrs.action_space = s->space;
  return rlmesh_env_new(&ev, s, &attrs, out);
}
static bool future_field(const void *value, size_t size, size_t offset) {
  bool result = false;
  if (offset + sizeof(result) <= size)
    memcpy(&result, (const char *)value + offset, sizeof(result));
  return result;
}
static RLMeshStatus lent_step(void *ud, const RLMeshStepEvent *event) {
  Script *s = (Script *)ud;
  ++s->starts;
  struct FutureEvent {
    RLMeshStepEvent known;
    bool enabled;
  };
  REQUIRE(event->struct_size == sizeof(*event));
  REQUIRE(!future_field(event, event->struct_size,
                        offsetof(struct FutureEvent, enabled)));
  return RLMESH_OK;
}
static RLMeshStatus run_end(void *ud, const RLMeshRunResult *result) {
  Script *s = (Script *)ud;
  ++s->run_ends;
  s->partial = (int)rlmesh_run_result_num_episodes(result);
  return RLMESH_OK;
}
static RLMeshStatus fail_hook(void *ud, const RLMeshStepEvent *event) {
  (void)ud;
  REQUIRE(event->struct_size == sizeof(*event));
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TYPE_ERROR, "hook abort");
  return RLMESH_ERR_TYPE;
}
static RLMeshStatus warning(void *ud, RLMeshWarningCategory category,
                            const char *message) {
  Script *s = (Script *)ud;
  (void)message;
  s->wrong_warning_thread |= !pthread_equal(s->caller, pthread_self());
  ++s->warnings;
  snprintf(s->category, sizeof(s->category), "%s",
           rlmesh_warning_category_name(category));
  if (s->warning_fail) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TYPE_ERROR,
                                    "warning abort");
    return RLMESH_ERR_TYPE;
  }
  return RLMESH_OK;
}
static RLMeshErrorClass find_class(const char *name) {
  if (!strcmp(name, "NotImplementedError"))
    name = "RuntimeError";
  if (!strcmp(name, "FileExistsError") || !strcmp(name, "PermissionError"))
    name = "OSError";
  if (!strcmp(name, "ParamError"))
    name = "ValueError";
  if (!strcmp(name, "ModelRecoverable"))
    name = "ModelException";
  for (int i = 1; i <= 13; ++i) {
    const char *candidate = rlmesh_error_class_name((RLMeshErrorClass)i);
    if (candidate && strcmp(name, candidate) == 0)
      return (RLMeshErrorClass)i;
  }
  abort();
}
typedef struct Serving {
  RLMeshModel *model;
  const char *address;
  RLMeshExtCancelToken *cancel;
  const char *token;
} Serving;
static void *serve_model(void *arg) {
  Serving *serving = (Serving *)arg;
  RLMeshExtModelServeOptions extra = RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT;
  extra.token = serving->token;
  RLMeshModelServeOptions options = RLMESH_MODEL_SERVE_OPTIONS_INIT;
  options.cancel = serving->cancel;
  options.ext = &extra;
  (void)rlmesh_model_serve(serving->model, serving->address, &options);
  return NULL;
}
static void wait_for_socket(const char *path) {
  for (int i = 0; i < 1000; ++i) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof addr);
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof addr.sun_path, "%s", path);
    int connected = connect(fd, (struct sockaddr *)&addr, sizeof addr);
    close(fd);
    if (connected == 0)
      return;
    struct timespec pause = {0, 10000000};
    nanosleep(&pause, NULL);
  }
  abort();
}
/* A served model reached with an api_key, over TLS, or driving a local env. */
static void remote_auth(const char *variant, RLMeshEnv *env, RLMeshModel *model,
                        const RLMeshEnvVtable *ev, Script *s,
                        const RLMeshEnvAttributes *attrs) {
  RLMeshRunResult *result = NULL;
  if (!strcmp(variant, "invalid_header")) {
    RLMeshRemoteModelOptions mo = RLMESH_REMOTE_MODEL_OPTIONS_INIT;
    mo.address = "https://localhost:1";
    mo.api_key = "bad\nkey";
    RLMeshRemoteModel *remote = NULL;
    REQUIRE(rlmesh_remote_model_new(&mo, &remote) != RLMESH_OK);
    error_fields();
    return;
  }
  if (!strcmp(variant, "https_to_plaintext") ||
      !strcmp(variant, "local_model_https_env")) {
    Script other = *s;
    RLMeshEnv *plain = NULL;
    OK(rlmesh_env_new(ev, &other, attrs, &plain));
    RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
    so.address = "tcp://127.0.0.1:0";
    RLMeshEnvServer *server = NULL;
    OK(rlmesh_env_server_new(&plain, 1, &so, &server));
    OK(rlmesh_env_server_start(server));
    RLMeshBytes address = {NULL, 0, 0};
    OK(rlmesh_env_server_address(server, &address));
    char port[16] = {0}, url[64];
    size_t colon = address.len;
    while (colon > 0 && ((const char *)address.data)[colon - 1] != ':')
      --colon;
    REQUIRE(colon > 0 && address.len - colon < sizeof port);
    memcpy(port, (const char *)address.data + colon, address.len - colon);
    snprintf(url, sizeof url, "https://127.0.0.1:%s", port);
    if (!strcmp(variant, "local_model_https_env")) {
      REQUIRE(rlmesh_model_run(model, rlmesh_address_target(url), NULL,
                               &result) != RLMESH_OK);
      error_fields_masked(port, "PORT");
    } else {
      RLMeshRemoteModelOptions mo = RLMESH_REMOTE_MODEL_OPTIONS_INIT;
      mo.address = url;
      RLMeshRemoteModel *remote = NULL;
      OK(rlmesh_remote_model_new(&mo, &remote));
      REQUIRE(rlmesh_run(rlmesh_remote_model_target(remote),
                         rlmesh_env_target(env), NULL, &result) != RLMESH_OK);
      error_fields_masked(port, "PORT");
      rlmesh_remote_model_free(remote);
    }
    OK(rlmesh_env_server_shutdown(server));
    rlmesh_env_server_free(server);
    rlmesh_bytes_free(address);
    rlmesh_env_free(plain);
    return;
  }
  char dir[] = "/tmp/rlmesh-parity-XXXXXX";
  REQUIRE(mkdtemp(dir) != NULL);
  char path[128], address[160];
  snprintf(path, sizeof path, "%s/model.sock", dir);
  snprintf(address, sizeof address, "unix://%s", path);
  Serving serving = {model, address, NULL, NULL};
  OK(rlmesh_ext_cancel_token_new(&serving.cancel));
  bool keyed = !strcmp(variant, "api_key_rejected") ||
               !strcmp(variant, "api_key_accepted");
  if (keyed)
    serving.token = "right-key";
  pthread_t thread;
  REQUIRE(pthread_create(&thread, NULL, serve_model, &serving) == 0);
  wait_for_socket(path);
  RLMeshRemoteModelOptions mo = RLMESH_REMOTE_MODEL_OPTIONS_INIT;
  mo.address = address;
  if (!strcmp(variant, "api_key_rejected"))
    mo.api_key = "wrong-key";
  else if (!strcmp(variant, "api_key_accepted"))
    mo.api_key = "right-key";
  RLMeshRemoteModel *remote = NULL;
  OK(rlmesh_remote_model_new(&mo, &remote));
  if (rlmesh_run(rlmesh_remote_model_target(remote), rlmesh_env_target(env),
                 NULL, &result) != RLMESH_OK)
    error_fields_masked(path, "SOCKET");
  else {
    printf("\"episodes\":%llu,\"steps\":%llu,\"env_closes\":%d",
           (unsigned long long)rlmesh_run_result_num_episodes(result),
           (unsigned long long)rlmesh_run_result_total_steps(result),
           s->env_closes);
    rlmesh_run_result_free(result);
  }
  rlmesh_remote_model_free(remote);
  OK(rlmesh_ext_cancel_token_cancel(serving.cancel));
  REQUIRE(pthread_join(thread, NULL) == 0);
  rlmesh_ext_cancel_token_free(serving.cancel);
  unlink(path);
  rmdir(dir);
}

int main(int argc, char **argv) {
  REQUIRE(argc == 6);
  const char *op = argv[1];
  const char *variant = argv[2];
  Script s = {0};
  s.caller = pthread_self();
  RLMeshDiscrete *space = NULL;
  OK(rlmesh_discrete_new(3, NULL, &space));
  s.space = space;
  RLMeshEnvVtable ev = RLMESH_ENV_VTABLE_INIT;
  ev.reset = reset;
  ev.step = step;
  ev.close = close_env;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = space;
  attrs.action_space = space;
  RLMeshDict *child_dict = NULL, *action_dict = NULL;
  if (!strcmp(op, "dict_error")) {
    RLMeshStr child_key = {"k", 1}, path_key = {"path", 4};
    const RLMeshSpaceSpec *child_spec = rlmesh_space_spec(space);
    OK(rlmesh_dict_new(&child_key, &child_spec, 1, &child_dict));
    const RLMeshSpaceSpec *path_spec = rlmesh_space_spec(child_dict);
    OK(rlmesh_dict_new(&path_key, &path_spec, 1, &action_dict));
    attrs.action_space = action_dict;
  }
  RLMeshValue *trial_name = NULL, *reset_options = NULL, *metadata = NULL;
  OK(rlmesh_value_text((RLMeshStr){"trial_index", 11}, &trial_name));
  const RLMeshValue *trial_names[] = {trial_name};
  OK(rlmesh_value_list(trial_names, 1, &reset_options));
  RLMeshStr meta_key = {RLMESH_ENV_RESET_OPTIONS_KEY,
                        sizeof(RLMESH_ENV_RESET_OPTIONS_KEY) - 1};
  const RLMeshValue *meta_values[] = {reset_options};
  OK(rlmesh_value_dict(&meta_key, meta_values, 1, &metadata));
  attrs.metadata = metadata;
  RLMeshEnv *env = NULL;
  OK(rlmesh_env_new(&ev, &s, &attrs, &env));
  rlmesh_value_free(trial_name);
  rlmesh_value_free(reset_options);
  rlmesh_value_free(metadata);
  attrs.metadata = NULL;
  RLMeshModelVtable mv = RLMESH_MODEL_VTABLE_INIT;
  mv.predict = predict;
  if (!strcmp(op, "warning") && !strncmp(variant, "runtime", 7))
    mv.predict_chunk = chunk;
  mv.on_episode_end = episode_end;
  mv.close = close_model;
  RLMeshModel *model = NULL;
  OK(rlmesh_model_new(&mv, &s, NULL, &model));
  RLMeshRunResult *result = NULL;
  RLMeshRunOptions opts = RLMESH_RUN_OPTIONS_INIT;
  int64_t seeds[128];
  size_t n = 0;
  if (strcmp(argv[4], "none")) {
    const char *p = argv[4];
    while (*p) {
      REQUIRE(n < 128);
      char *next;
      seeds[n++] = strtoll(p, &next, 10);
      REQUIRE(next != p);
      p = *next ? next + 1 : next;
    }
    opts.seeds = seeds;
    opts.seeds_len = n;
  }
  if (atoi(argv[3]) >= 0) {
    opts.has_episodes = true;
    opts.episodes = (uint64_t)atoi(argv[3]);
  }
  opts.close_env = atoi(argv[5]) != 0;
  putchar('{');
  if (!strcmp(op, "dict_error")) {
    RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
    so.address = "tcp://127.0.0.1:0";
    RLMeshEnvServer *server = NULL;
    OK(rlmesh_env_server_new(&env, 1, &so, &server));
    OK(rlmesh_env_server_start(server));
    RLMeshBytes address = {0};
    OK(rlmesh_env_server_address(server, &address));
    char *text = calloc(address.len + 1, 1);
    memcpy(text, address.data, address.len);
    RLMeshRemoteEnvOptions ro = RLMESH_REMOTE_ENV_OPTIONS_INIT;
    ro.address = text;
    RLMeshRemoteEnv *remote = NULL;
    OK(rlmesh_remote_env_new(&ro, &remote));
    RLMeshValue *obs = NULL, *info = NULL, *empty = NULL, *action = NULL;
    OK(rlmesh_remote_env_reset(remote, NULL, NULL, &obs, &info));
    OK(rlmesh_value_dict(NULL, NULL, 0, &empty));
    RLMeshStr key = {"path", 4};
    const RLMeshValue *values[] = {empty};
    OK(rlmesh_value_dict(&key, values, 1, &action));
    double reward = 0;
    bool term = false, trunc = false;
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    obs = NULL;
    info = NULL;
    RLMeshStatus status = rlmesh_remote_env_step(remote, action, &obs, &reward,
                                                 &term, &trunc, &info);
    REQUIRE(status != RLMESH_OK);
    error_fields();
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    rlmesh_value_free(empty);
    rlmesh_value_free(action);
    rlmesh_remote_env_free(remote);
    OK(rlmesh_env_server_shutdown(server));
    rlmesh_env_server_free(server);
    rlmesh_bytes_free(address);
    free(text);
  } else if (!strcmp(op, "reset_order")) {
    RLMeshSession *session = NULL;
    OK(rlmesh_model_session(model, rlmesh_env_target(env), NULL, &session));
    int64_t seed = 37, trial = 11, value = -1;
    RLMeshValue *obs = NULL, *info = NULL;
    OK(rlmesh_session_reset(session, &seed, &trial, &obs, &info));
    OK(rlmesh_value_as_int(obs, &value));
    size_t info_len = 0;
    if (info)
      OK(rlmesh_value_len(info, &info_len));
    printf("\"seed\":%lld,\"trial_index\":%lld,\"observation\":%lld,\"info_"
           "empty\":%s",
           (long long)s.reset_seed, (long long)s.reset_trial, (long long)value,
           info_len == 0 ? "true" : "false");
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    rlmesh_session_free(session);
  } else if (!strcmp(op, "config_error")) {
    RLMeshLoadKey key = {"checkpoint", true, {0}};
    mv.load_keys = &key;
    mv.load_keys_len = 1;
    RLMeshValue *config = NULL, *value = NULL;
    if (!strcmp(variant, "unknown")) {
      OK(rlmesh_value_int(1, &value));
      RLMeshStr keys[] = {{"checkpoint", 10}, {"scripted", 8}};
      const RLMeshValue *values[] = {value, value};
      OK(rlmesh_value_dict(keys, values, 2, &config));
    } else {
      OK(rlmesh_value_dict(NULL, NULL, 0, &config));
    }
    RLMeshModel *invalid = NULL;
    RLMeshStatus status = rlmesh_model_from_config(&mv, &s, config, &invalid);
    REQUIRE(status != RLMESH_OK);
    error_fields();
    rlmesh_value_free(value);
    rlmesh_value_free(config);
    rlmesh_model_free(invalid);
  } else if (!strcmp(op, "validation")) {
    RLMeshStatus status;
    if (!strncmp(variant, "horizon", 7)) {
      opts.execution_horizon = !strcmp(variant, "horizon_zero") ? 0 : 1025;
      status = rlmesh_model_run(model, rlmesh_env_target(env), &opts, &result);
    } else {
      RLMeshView view = RLMESH_VIEW_INIT;
      if (!strcmp(variant, "view_port"))
        view.port = 0;
      if (!strcmp(variant, "view_fps"))
        view.fps = 0;
      if (!strcmp(variant, "view_quality"))
        view.quality = 0;
      if (!strcmp(variant, "view_step_hz")) {
        view.has_step_hz = true;
        view.step_hz = 0;
      }
      status = rlmesh_view_validate(&view);
    }
    REQUIRE(status != RLMESH_OK);
    error_fields();
  } else if (!strcmp(op, "run")) {
    RLMeshStatus status =
        rlmesh_model_run(model, rlmesh_env_target(env), &opts, &result);
    if (status)
      error_fields();
    else {
      printf("\"episodes\":%llu,\"steps\":%llu,\"reward\":%.17g,\"env_closes\":"
             "%d,\"model_closes\":%d",
             (unsigned long long)rlmesh_run_result_num_episodes(result),
             (unsigned long long)rlmesh_run_result_total_steps(result),
             rlmesh_run_result_mean_reward(result), s.env_closes,
             s.model_closes);
      rlmesh_run_result_free(result);
      result = NULL;
      RLMeshValue *value = NULL;
      OK(rlmesh_value_int(0, &value));
      /* A second run verifies both borrowed handles (unless close_env closed
       * it). */
      if (!opts.close_env) {
        OK(rlmesh_model_run(model, rlmesh_env_target(env), NULL, &result));
        rlmesh_run_result_free(result);
        result = NULL;
      } else {
        RLMeshEnv *fresh = NULL;
        OK(rlmesh_env_new(&ev, &s, &attrs, &fresh));
        OK(rlmesh_model_run(model, rlmesh_env_target(fresh), NULL, &result));
        rlmesh_run_result_free(result);
        result = NULL;
        rlmesh_env_free(fresh);
      }
      rlmesh_value_free(value);
      printf(",\"reusable\":true");
    }
  } else if (!strcmp(op, "busy_session")) {
    RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
    so.address = "tcp://127.0.0.1:0";
    RLMeshEnvServer *server = NULL;
    OK(rlmesh_env_server_new(&env, 1, &so, &server));
    OK(rlmesh_env_server_start(server));
    RLMeshSession *first = NULL, *second = NULL;
    RLMeshValue *obs = NULL, *info = NULL, *action = NULL;
    double reward;
    bool term, trunc;
    OK(rlmesh_model_session(model, rlmesh_env_server_target(server), NULL,
                            &first));
    OK(rlmesh_session_reset(first, NULL, NULL, &obs, &info));
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    obs = info = NULL;
    RLMeshStatus status = rlmesh_model_session(
        model, rlmesh_env_server_target(server), NULL, &second);
    if (!status)
      status = rlmesh_session_reset(second, NULL, NULL, &obs, &info);
    REQUIRE(status);
    error_fields();
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    obs = info = NULL;
    rlmesh_session_free(second);
    OK(rlmesh_value_int(0, &action));
    OK(rlmesh_session_step(first, action, &obs, &reward, &term, &trunc, &info));
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    rlmesh_value_free(action);
    rlmesh_session_free(first);
    printf(",\"reusable\":true");
    OK(rlmesh_env_server_shutdown(server));
    rlmesh_env_server_free(server);
  } else if (!strcmp(op, "remote_auth")) {
    remote_auth(variant, env, model, &ev, &s, &attrs);
  } else if (!strcmp(op, "busy_remote")) {
    RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
    so.address = ":0/env";
    RLMeshEnvServer *server = NULL;
    OK(rlmesh_env_server_new(&env, 1, &so, &server));
    OK(rlmesh_env_server_start(server));
    RLMeshBytes address = {NULL, 0, 0};
    OK(rlmesh_env_server_address(server, &address));
    char *text = (char *)malloc(address.len + 1);
    memcpy(text, address.data, address.len);
    text[address.len] = 0;
    RLMeshRemoteEnvOptions ro = RLMESH_REMOTE_ENV_OPTIONS_INIT;
    ro.address = text;
    RLMeshRemoteEnv *remote = NULL;
    OK(rlmesh_remote_env_new(&ro, &remote));
    RLMeshValue *obs = NULL, *info = NULL, *action = NULL;
    double reward;
    bool term, trunc;
    OK(rlmesh_value_int(0, &action));
    OK(rlmesh_remote_env_reset(remote, NULL, NULL, &obs, &info));
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    OK(rlmesh_remote_env_step(remote, action, &obs, &reward, &term, &trunc,
                              &info));
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    RLMeshStatus status = rlmesh_model_run(
        model, rlmesh_remote_env_target(remote), NULL, &result);
    REQUIRE(status != RLMESH_OK);
    error_fields();
    OK(rlmesh_remote_env_step(remote, action, &obs, &reward, &term, &trunc,
                              &info));
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    rlmesh_value_free(action);
    printf(",\"reusable\":true");
    OK(rlmesh_remote_env_close(remote));
    rlmesh_remote_env_free(remote);
    OK(rlmesh_env_server_shutdown(server));
    rlmesh_env_server_free(server);
    rlmesh_bytes_free(address);
    free(text);
  } else if (!strcmp(op, "warning")) {
    s.warning_fail = !strcmp(variant, "fail") ||
                     !strcmp(variant, "runtime_fail") ||
                     !strcmp(variant, "recorder");
    if (!strcmp(variant, "workflow"))
      REQUIRE(unsetenv("RLMESH_WORKFLOW_EDITION") == 0);
    OK(rlmesh_set_warning_handler(warning, &s, NULL));
    opts.execution_horizon =
        !strcmp(variant, "workflow") || !strcmp(variant, "recorder") ? 1 : 2;
    RLMeshRecorder *recorder = NULL;
    RLMeshRunHooks *capture = NULL;
    if (!strcmp(variant, "recorder")) {
      OK(rlmesh_recorder_new(NULL, &recorder));
      RLMeshRecorderCaptureOptions capture_opts =
          RLMESH_RECORDER_CAPTURE_OPTIONS_INIT;
      const char *camera = "absent";
      capture_opts.cameras = &camera;
      capture_opts.cameras_len = 1;
      OK(rlmesh_recorder_capture(recorder, "model", "env", &capture_opts,
                                 &capture));
      opts.hooks = capture;
    }
    RLMeshStatus status =
        rlmesh_model_run(model, rlmesh_env_target(env), &opts, &result);
    if (status) {
      error_fields();
      putchar(',');
    }
    printf("\"warnings\":%d,\"calling_thread\":%s,\"category\":", s.warnings,
           s.wrong_warning_thread ? "false" : "true");
    quoted(s.category);
    OK(rlmesh_set_warning_handler(NULL, NULL, NULL));
    rlmesh_run_result_free(result);
    result = NULL;
    rlmesh_run_hooks_free(capture);
    rlmesh_recorder_free(recorder);
  } else if (!strcmp(op, "factory_start")) {
    RLMeshEnvFactoryVtable fv = RLMESH_ENV_FACTORY_VTABLE_INIT;
    fv.prepare = factory_prepare;
    fv.make = factory_make;
    RLMeshEnvFactory *factory = NULL;
    OK(rlmesh_env_factory_new(&fv, &s, NULL, &factory));
    RLMeshEnvServer *server = NULL;
    RLMeshEnvFactoryServeOptions fo = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
    RLMeshExtCancelToken *token = NULL;
    if (!strcmp(variant, "cancel")) {
      OK(rlmesh_ext_cancel_token_new(&token));
      fo.cancel = token;
    }
    OK(rlmesh_ext_env_factory_start(factory, "tcp://127.0.0.1:0", &fo,
                                    &server));
    rlmesh_env_factory_free(factory);
    OK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_server_target(server), NULL,
                  &result));
    printf("\"episodes\":%llu",
           (unsigned long long)rlmesh_run_result_num_episodes(result));
    if (token) {
      OK(rlmesh_ext_cancel_token_cancel(token));
      double timeout = 5;
      bool stopped = false;
      REQUIRE(rlmesh_env_server_wait(server, &timeout, &stopped) ==
                  RLMESH_ERR_CANCELLED &&
              stopped);
      OK(rlmesh_env_server_wait(server, &timeout, &stopped));
      printf(",\"cancelled\":true");
    }
    OK(rlmesh_env_server_shutdown(server));
    rlmesh_env_server_free(server);
    printf(",\"prepares\":%d,\"env_closes\":%d", s.prepares, s.env_closes);
    rlmesh_ext_cancel_token_free(token);
  } else if (!strcmp(op, "factory_cancel") ||
             (!strcmp(op, "drop_serve") &&
              (!strcmp(variant, "factory_cancel") ||
               !strcmp(variant, "factory_idle")))) {
    OK(rlmesh_ext_cancel_token_new(&s.cancel));
    s.cancel_in_make =
        !strcmp(variant, "serving") || !strcmp(variant, "factory_cancel");
    if (!s.cancel_in_make)
      OK(rlmesh_ext_cancel_token_cancel(s.cancel));
    RLMeshEnvFactoryVtable fv = RLMESH_ENV_FACTORY_VTABLE_INIT;
    fv.prepare = factory_prepare;
    fv.make = factory_make;
    RLMeshEnvFactoryAttributes fa = RLMESH_ENV_FACTORY_ATTRIBUTES_INIT;
    RLMeshEnvFactory *factory = NULL;
    OK(rlmesh_env_factory_new(&fv, &s, &fa, &factory));
    RLMeshEnvFactoryServeOptions fo = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
    fo.cancel = s.cancel;
    REQUIRE(rlmesh_env_factory_serve(factory, "tcp://127.0.0.1:0", &fo) ==
            RLMESH_ERR_CANCELLED);
    error_fields();
    printf(",\"env_closes\":%d,\"prepares\":%d", s.env_closes, s.prepares);
    rlmesh_env_factory_free(factory);
    rlmesh_ext_cancel_token_free(s.cancel);
  } else if (!strcmp(op, "sizing")) {
    RLMeshRunOptions zero = {0};
    if (!strcmp(variant, "zero")) {
      REQUIRE(rlmesh_model_run(model, rlmesh_env_target(env), &zero, &result));
      error_fields();
    } else if (!strcmp(variant, "minimum")) {
      opts.struct_size = RLMESH_RUN_OPTIONS_MIN_SIZE;
      OK(rlmesh_model_run(model, rlmesh_env_target(env), &opts, &result));
      printf("\"episodes\":%llu",
             (unsigned long long)rlmesh_run_result_num_episodes(result));
    } else if (!strcmp(variant, "abi5_remote")) {
      /* An ABI-5 caller's struct_size stops before the appended api_key and
       * tls_ca_file: they read as NULL. A size inside one is refused. */
      RLMeshRemoteModelOptions mo = RLMESH_REMOTE_MODEL_OPTIONS_INIT;
      mo.address = "127.0.0.1:1";
      mo.api_key = "never-read\n";
      mo.struct_size = RLMESH_REMOTE_MODEL_OPTIONS_MIN_SIZE;
      RLMeshRemoteModel *remote = NULL;
      OK(rlmesh_remote_model_new(&mo, &remote));
      rlmesh_remote_model_free(remote);
      RLMeshRemoteEnvOptions eo = RLMESH_REMOTE_ENV_OPTIONS_INIT;
      eo.struct_size = RLMESH_REMOTE_ENV_OPTIONS_MIN_SIZE + 1;
      RLMeshRemoteEnv *renv = NULL;
      REQUIRE(rlmesh_remote_env_new(&eo, &renv) != RLMESH_OK);
      /* The size differs per target; the text masks it. */
      char inside[32];
      snprintf(inside, sizeof inside, "%zu",
               (size_t)(RLMESH_REMOTE_ENV_OPTIONS_MIN_SIZE + 1));
      error_fields_masked(inside, "MIN_SIZE+1");
    } else if (!strcmp(variant, "out")) {
      OK(rlmesh_model_run(model, rlmesh_env_target(env), NULL, &result));
      struct ExtendedOut {
        RLMeshEpisodeResult known;
        bool future_enabled;
        uint8_t padding[7];
      } extended = {RLMESH_EPISODE_RESULT_INIT, false, {0}};
      extended.known.struct_size = sizeof(extended);
      OK(rlmesh_run_result_episodes_get(result, 0, &extended.known));
      REQUIRE(!extended.future_enabled && extended.known.steps == 2);
      printf("\"unknown_default\":true");
    } else if (!strcmp(variant, "lent")) {
      RLMeshRunHooksVtable hv = RLMESH_RUN_HOOKS_VTABLE_INIT;
      hv.on_step = lent_step;
      RLMeshRunHooks *hooks = NULL;
      OK(rlmesh_run_hooks_new(&hv, &s, &hooks));
      opts.hooks = hooks;
      OK(rlmesh_model_run(model, rlmesh_env_target(env), &opts, &result));
      rlmesh_run_hooks_free(hooks);
      RLMeshState state = RLMESH_STATE_INIT;
      state.role = RLMESH_ADAPTERS_JOINT_POS;
      state.has_dim = true;
      state.dim = 1;
      RLMeshInputNode input = rlmesh_input_node_state(&state);
      RLMeshAction action_spec = RLMESH_ACTION_INIT;
      RLMeshActuator actuator = RLMESH_ACTUATOR_INIT;
      actuator.role = RLMESH_ADAPTERS_ACTION_JOINT_POS;
      actuator.dim = 1;
      const RLMeshActuator *components[] = {&actuator};
      action_spec.components = components;
      action_spec.components_len = 1;
      RLMeshModelSpec *spec = NULL;
      OK(rlmesh_model_spec_new(&input, &action_spec, &spec));
      const RLMeshState *lent =
          (const RLMeshState *)rlmesh_model_spec_input(spec)->leaf;
      struct FutureState {
        RLMeshState known;
        bool enabled;
      };
      REQUIRE(lent->struct_size == sizeof(*lent));
      REQUIRE(!future_field(lent, lent->struct_size,
                            offsetof(struct FutureState, enabled)));
      rlmesh_model_spec_free(spec);
      printf("\"library_size\":true,\"unknown_default\":true");
    } else {
      struct Extended {
        RLMeshRunOptions known;
        uint64_t tail;
      } extended = {RLMESH_RUN_OPTIONS_INIT, 0};
      extended.known.struct_size = sizeof(extended);
      extended.tail = !strcmp(variant, "nonzero_tail") ? 1 : 0;
      RLMeshStatus status = rlmesh_model_run(model, rlmesh_env_target(env),
                                             &extended.known, &result);
      if (status)
        error_fields();
      else
        printf("\"episodes\":%llu",
               (unsigned long long)rlmesh_run_result_num_episodes(result));
    }
    rlmesh_run_result_free(result);
    result = NULL;
  } else {
    OK(rlmesh_ext_cancel_token_new(&s.cancel));
    RLMeshSessionOptions so = RLMESH_SESSION_OPTIONS_INIT;
    so.cancel = s.cancel;
    RLMeshSession *session = NULL;
    OK(rlmesh_model_session(model, rlmesh_env_target(env), &so, &session));
    RLMeshValue *obs = NULL, *info = NULL, *action = NULL;
    double reward;
    bool term, trunc;
    OK(rlmesh_value_int(0, &action));
    if (strcmp(op, "cancel_run") && strcmp(op, "hook_error")) {
      OK(rlmesh_session_reset(session, NULL, NULL, &obs, &info));
      rlmesh_value_free(info);
      info = NULL;
    }
    if (!strcmp(op, "borrow")) {
      rlmesh_session_free(session);
      session = NULL;
      printf("\"env_closes\":%d,\"model_closes\":%d,\"episode_ends\":%d",
             s.env_closes, s.model_closes, s.episode_ends);
      OK(rlmesh_model_run(model, rlmesh_env_target(env), NULL, &result));
      rlmesh_run_result_free(result);
      result = NULL;
      printf(",\"reusable\":true");
    } else if (!strcmp(op, "error")) {
      s.failure = find_class(variant);
      s.model_recoverable = !strcmp(variant, "ModelRecoverable");
      RLMeshValue *prediction = NULL;
      REQUIRE(rlmesh_session_predict(session, obs, &prediction));
      error_fields();
      REQUIRE(!prediction);
    } else if (!strcmp(op, "cancel_idle")) {
      int failures = 0;
      int cancelled_call_starts = 0;
      for (int i = 0; i < 2; ++i) {
        int before = s.starts;
        OK(rlmesh_ext_cancel_token_cancel(s.cancel));
        OK(rlmesh_ext_cancel_token_cancel(s.cancel));
        RLMeshValue *prediction = NULL;
        REQUIRE(rlmesh_session_predict(session, obs, &prediction) ==
                RLMESH_ERR_CANCELLED);
        ++failures;
        cancelled_call_starts += s.starts - before;
        if (!i)
          error_fields();
        OK(rlmesh_session_predict(session, obs, &prediction));
        rlmesh_value_free(prediction);
      }
      printf(",\"failures\":%d,\"cancelled_call_starts\":%d,\"reusable\":true",
             failures, cancelled_call_starts);
    } else if (!strcmp(op, "cancel_step") || !strcmp(op, "cancel_run") ||
               !strcmp(op, "hook_error")) {
      s.cancel_step = strcmp(op, "hook_error") != 0;
      RLMeshStatus status;
      RLMeshRunHooksVtable hv = RLMESH_RUN_HOOKS_VTABLE_INIT;
      hv.on_run_end = run_end;
      if (!strcmp(op, "hook_error"))
        hv.on_step = fail_hook;
      RLMeshRunHooks *hooks = NULL;
      OK(rlmesh_run_hooks_new(&hv, &s, &hooks));
      if (!strcmp(op, "cancel_step")) {
        rlmesh_value_free(obs);
        obs = NULL;
        status = rlmesh_session_step(session, action, &obs, &reward, &term,
                                     &trunc, &info);
      } else {
        RLMeshSessionRunOptions run = RLMESH_SESSION_RUN_OPTIONS_INIT;
        run.hooks = hooks;
        status = rlmesh_session_run(session, &run, &result);
      }
      REQUIRE(status);
      error_fields();
      printf(",\"episode_ends_before_reset\":%d", s.episode_ends);
      if (!strcmp(op, "cancel_step")) {
        OK(rlmesh_session_reset(session, NULL, NULL, &obs, &info));
      }
      printf(",\"episode_ends\":%d,\"run_ends\":%d,\"partial_episodes\":%d",
             s.episode_ends, s.run_ends, s.partial);
      rlmesh_value_free(obs);
      rlmesh_value_free(info);
      obs = info = NULL;
      OK(rlmesh_session_step(session, action, &obs, &reward, &term, &trunc,
                             &info));
      printf(",\"reusable\":true");
      rlmesh_run_hooks_free(hooks);
    } else {
      fprintf(stderr, "unknown operation %s\n", op);
      abort();
    }
    rlmesh_value_free(obs);
    rlmesh_value_free(info);
    rlmesh_value_free(action);
    rlmesh_session_free(session);
    rlmesh_ext_cancel_token_free(s.cancel);
  }
  puts("}");
  rlmesh_model_free(model);
  rlmesh_env_free(env);
  rlmesh_space_free(space);
  rlmesh_space_free(child_dict);
  rlmesh_space_free(action_dict);
  return 0;
}
