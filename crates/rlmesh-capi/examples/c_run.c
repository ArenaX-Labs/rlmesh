/* ABI 5: both facade loops, hook errors, contexts, budgets and result ownership. */
#include <math.h>
#include <rlmesh.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int failures;
#define CHECK(x)                                                                                \
  do {                                                                                          \
    if (!(x)) {                                                                                 \
      fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, rlmesh_last_error_message()); \
      ++failures;                                                                               \
    }                                                                                           \
  } while (0)
typedef struct State {
  int resets, steps, closes, starts, ends, episode_ends, hook_steps, frees;
  bool abort_step;
  RLMeshSession* session;
  RLMeshRunHooks* hooks;
  RLMeshRunContext* context;
  RLMeshRunResult* retained;
} State;
static RLMeshStatus reset(void* ud, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  State* s = (State*)ud;
  ++s->resets;
  (void)seed;
  (void)info;
  bool has = false;
  int64_t trial = 0;
  CHECK(rlmesh_trial_index(options, NULL, &has, &trial) == RLMESH_OK && has);
  return rlmesh_value_int(0, obs);
}
static RLMeshStatus step(void* ud, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                         bool* terminated, bool* truncated, RLMeshValue** info) {
  ++((State*)ud)->steps;
  (void)action;
  *reward = 3.0;
  *terminated = true;
  *truncated = false;
  RLMeshValue* success = NULL;
  CHECK(rlmesh_value_bool(true, &success) == RLMESH_OK);
  const RLMeshValue* vals[] = {success};
  RLMeshStr key = rlmesh_str("success");
  CHECK(rlmesh_value_dict(&key, vals, 1, info) == RLMESH_OK);
  rlmesh_value_free(success);
  return rlmesh_value_int(1, obs);
}
static RLMeshStatus close_env(void* ud) {
  ++((State*)ud)->closes;
  return RLMESH_OK;
}
static int state_frees, replaced_state_frees;
static void old_state_free(void* value) {
  ++replaced_state_frees;
  free(value);
}
static void new_state_free(void* value) {
  ++state_frees;
  free(value);
}
static RLMeshStatus predict(void* ud, const RLMeshValue* obs, const RLMeshPredictContext* ctx,
                            RLMeshValue** action) {
  (void)ud;
  (void)obs;
  CHECK(ctx != NULL && ctx->predict_index == 0);
  int* value = malloc(sizeof(*value));
  CHECK(value != NULL);
  *value = 42;
  int before = state_frees;
  CHECK(rlmesh_episode_state_set(ctx->state, "owned", value, old_state_free) == RLMESH_OK);
  CHECK(rlmesh_episode_state_set(ctx->state, "owned", value, new_state_free) == RLMESH_OK);
  CHECK(replaced_state_frees == 0 && state_frees == before);
  void* read = NULL;
  CHECK(rlmesh_episode_state_get(ctx->state, "owned", &read) && read == value);
  CHECK(*(int*)read == 42);
  CHECK(rlmesh_episode_state_remove(ctx->state, "owned"));
  CHECK(replaced_state_frees == 0 && state_frees == before + 1);
  return rlmesh_value_int(0, action);
}
static RLMeshStatus start(void* ud, const RLMeshRunContext* ctx) {
  State* s = (State*)ud;
  ++s->starts;
  rlmesh_run_context_free(s->context);
  s->context = rlmesh_run_context_clone(ctx);
  uint64_t n = 0;
  CHECK(rlmesh_run_context_num_envs(ctx, &n) == RLMESH_OK && n == 1);
  CHECK(rlmesh_run_hooks_on_run_start(s->hooks, ctx) == RLMESH_ERR_RUNTIME);
  if (s->session) {
    bool done = false;
    CHECK(rlmesh_session_done(s->session, &done) == RLMESH_ERR_RUNTIME);
    CHECK(rlmesh_session_num_envs(s->session, &n) == RLMESH_OK && n == 1);
    RLMeshEnvContract* contract = NULL;
    CHECK(rlmesh_session_contract(s->session, &contract) == RLMESH_OK && contract);
    rlmesh_env_contract_free(contract);
    RLMeshBytes edition = {NULL, 0, 0};
    CHECK(rlmesh_session_selected_workflow_edition(s->session, &edition) == RLMESH_ERR_RUNTIME);
    CHECK(rlmesh_session_close(s->session) == RLMESH_ERR_RUNTIME);
  }
  return RLMESH_OK;
}
static RLMeshStatus on_step(void* ud, const RLMeshStepEvent* event) {
  State* s = (State*)ud;
  ++s->hook_steps;
  CHECK(event->struct_size == sizeof(*event) && event->step == 0 && event->terminated);
  CHECK(rlmesh_value_kind(event->action) == RLMESH_VALUE_INT);
  CHECK(rlmesh_value_kind(event->info) == RLMESH_VALUE_DICT);
  CHECK(event->context != NULL);
  bool equal = false;
  CHECK(rlmesh_step_event_equals(event, event, &equal) == RLMESH_OK && equal);
  if (s->abort_step) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TYPE_ERROR, "hook abort");
    return RLMESH_ERR_TYPE;
  }
  return RLMESH_OK;
}
static RLMeshStatus episode_end(void* ud, const RLMeshEpisodeResult* e) {
  State* s = (State*)ud;
  ++s->episode_ends;
  CHECK(e->struct_size == sizeof(*e) && e->steps == 1 && e->reward == 3.0);
  return RLMESH_OK;
}
static RLMeshStatus end(void* ud, const RLMeshRunResult* r) {
  State* s = (State*)ud;
  ++s->ends;
  rlmesh_run_result_free(s->retained);
  s->retained = rlmesh_run_result_clone(r);
  if (s->abort_step) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_RUNTIME_ERROR, "end error loses");
    return RLMESH_ERR_RUNTIME;
  }
  return RLMESH_OK;
}
static void free_hooks(void* ud) { ++((State*)ud)->frees; }
static RLMeshStatus poison_error(void* user, const RLMeshRunContext* context) {
  (void)user;
  (void)context;
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_OS_ERROR, "unrelated stale error");
  return RLMESH_OK;
}
static RLMeshStatus silent_hook_failure(void* user, int64_t episode, const int64_t* seed) {
  (void)user;
  (void)episode;
  (void)seed;
  CHECK(rlmesh_last_error_message() == NULL);
  return RLMESH_ERR_MODEL;
}
int main(void) {
  State s = {0};
  RLMeshDiscrete* space = NULL;
  CHECK(rlmesh_discrete_new(2, NULL, &space) == RLMESH_OK);
  RLMeshEnvVtable ev = RLMESH_ENV_VTABLE_INIT;
  ev.reset = reset;
  ev.step = step;
  ev.close = close_env;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = space;
  attrs.action_space = space;
  attrs.id = "C-Run-v1";
  RLMeshValue* reset_option = NULL;
  RLMeshValue* metadata = NULL;
  CHECK(rlmesh_value_text(rlmesh_str("trial_index"), &reset_option) == RLMESH_OK);
  RLMeshStr reset_key = rlmesh_str(RLMESH_ENV_RESET_OPTIONS_KEY);
  const RLMeshValue* reset_vals[] = {reset_option};
  CHECK(rlmesh_value_dict(&reset_key, reset_vals, 1, &metadata) == RLMESH_OK);
  attrs.metadata = metadata;
  RLMeshEnv* env = NULL;
  CHECK(rlmesh_env_new(&ev, &s, &attrs, &env) == RLMESH_OK);
  rlmesh_space_free(space);
  rlmesh_value_free(metadata);
  rlmesh_value_free(reset_option);
  RLMeshModelVtable mv = RLMESH_MODEL_VTABLE_INIT;
  mv.predict = predict;
  mv.context_corners = RLMESH_CORNER_PREDICT;
  RLMeshModel* model = NULL;
  CHECK(rlmesh_model_new(&mv, NULL, NULL, &model) == RLMESH_OK);
  RLMeshRunHooksVtable hv = RLMESH_RUN_HOOKS_VTABLE_INIT;
  hv.on_run_start = start;
  hv.on_step = on_step;
  hv.on_episode_end = episode_end;
  hv.on_run_end = end;
  hv.free_user_data = free_hooks;
  CHECK(rlmesh_run_hooks_new(&hv, &s, &s.hooks) == RLMESH_OK);
  RLMeshRunOptions o = RLMESH_RUN_OPTIONS_INIT;
  int64_t seeds[] = {-7, 12};
  o.seeds = seeds;
  o.seeds_len = 2;
  o.hooks = s.hooks;
  o.trial_index_base = 19;
  RLMeshRunResult* result = NULL;
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &o, &result) == RLMESH_OK);
  CHECK(s.starts == 1 && s.ends == 1 && s.episode_ends == 2 && s.hook_steps == 2 && s.closes == 0);
  CHECK(rlmesh_run_result_num_episodes(result) == 2 && rlmesh_run_result_total_steps(result) == 2 &&
        rlmesh_run_result_mean_reward(result) == 3.0);
  double rate = 0.0;
  CHECK(rlmesh_run_result_success_rate(result, &rate) && rate == 1.0);
  RLMeshEpisodeResult e = RLMESH_EPISODE_RESULT_INIT;
  CHECK(rlmesh_run_result_episodes_get(result, 1, &e) == RLMESH_OK);
  CHECK(e.index == 1 && e.has_seed && e.seed == 12 && e.has_trial && e.trial == 20);
  e.struct_size = 0;
  CHECK(rlmesh_run_result_episodes_get(result, 0, &e) == RLMESH_ERR_INVALID_ARGUMENT);
  e.struct_size = sizeof(e);
  CHECK(rlmesh_run_result_episodes_get(result, 2, &e) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(rlmesh_run_result_equals(result, s.retained));
  rlmesh_run_result_free(result);
  o.hooks = NULL;
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &o, &result) == RLMESH_OK);
  CHECK(rlmesh_run_result_num_episodes(result) == 2);
  rlmesh_run_result_free(result);
  o.has_episodes = true;
  o.episodes = 1;
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &o, &result) ==
            RLMESH_ERR_INVALID_ARGUMENT &&
        result == NULL);
  o.seeds = NULL;
  o.seeds_len = 0;
  o.has_max_episode_seconds = true;
  o.max_episode_seconds = NAN;
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &o, &result) ==
            RLMESH_ERR_INVALID_ARGUMENT &&
        result == NULL);
  o.max_episode_seconds = INFINITY;
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &o, &result) == RLMESH_OK);
  rlmesh_run_result_free(result);
  struct Extended {
    RLMeshRunOptions options;
    uint64_t tail;
  } ex = {RLMESH_RUN_OPTIONS_INIT, 1};
  ex.options.struct_size = sizeof(ex);
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &ex.options, &result) ==
        RLMESH_ERR_TYPE);
  ex.tail = 0;
  ex.options.has_episodes = true;
  ex.options.episodes = 0;
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &ex.options, &result) ==
            RLMESH_OK &&
        rlmesh_run_result_num_episodes(result) == 0);
  rlmesh_run_result_free(result);
  RLMeshRunOptions zero = {0};
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &zero, &result) ==
        RLMESH_ERR_INVALID_ARGUMENT);
  RLMeshEnvTarget invalid = {RLMESH_ENV_TARGET_KIND_INVALID, NULL, NULL};
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, invalid, NULL, &result) == RLMESH_ERR_INVALID_ARGUMENT);
  o.hooks = s.hooks;
  s.abort_step = true;
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &o, &result) == RLMESH_ERR_TYPE &&
        result == NULL);
  CHECK(strcmp(rlmesh_last_error_message(), "hook abort") == 0 && s.ends == 2);
  s.abort_step = false;
  RLMeshSessionOptions so = RLMESH_SESSION_OPTIONS_INIT;
  so.workflow_edition = rlmesh_current_workflow_edition();
  CHECK(rlmesh_model_session(model, rlmesh_env_target(env), &so, &s.session) == RLMESH_OK);
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &o, &result) == RLMESH_ERR_RUNTIME);
  int64_t seed = 2, trial = 7;
  RLMeshValue *obs = NULL, *info = NULL, *action = NULL;
  CHECK(rlmesh_session_reset(s.session, &seed, &trial, &obs, &info) == RLMESH_OK);
  bool done = true;
  CHECK(rlmesh_session_done(s.session, &done) == RLMESH_OK && !done);
  CHECK(rlmesh_session_predict(s.session, obs, &action) == RLMESH_OK);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  double reward = 0;
  bool terminated = false, truncated = true;
  CHECK(rlmesh_session_step(s.session, action, &obs, &reward, &terminated, &truncated, &info) ==
        RLMESH_OK);
  CHECK(reward == 3.0 && terminated && !truncated);
  rlmesh_value_free(obs);
  rlmesh_value_free(info);
  rlmesh_value_free(action);
  CHECK(rlmesh_session_done(s.session, &done) == RLMESH_OK && done);
  RLMeshSessionRunOptions sr = RLMESH_SESSION_RUN_OPTIONS_INIT;
  sr.hooks = s.hooks;
  CHECK(rlmesh_session_run(s.session, &sr, &result) == RLMESH_OK);
  CHECK(rlmesh_run_result_num_episodes(result) == 1);
  rlmesh_run_result_free(result);
  RLMeshRunContext* session_context = NULL;
  CHECK(rlmesh_session_run_context(s.session, &session_context) == RLMESH_OK);
  CHECK(rlmesh_session_close(s.session) == RLMESH_OK && s.closes == 0);
  uint64_t lanes = 0;
  CHECK(rlmesh_run_context_num_envs(session_context, &lanes) == RLMESH_ERR_RUNTIME);
  rlmesh_run_context_free(session_context);
  CHECK(rlmesh_session_close(s.session) == RLMESH_OK);
  CHECK(rlmesh_session_reset(s.session, NULL, NULL, &obs, &info) == RLMESH_ERR_RUNTIME);
  rlmesh_session_free(s.session);
  s.session = NULL;
  RLMeshExtCancelToken* cancel = NULL;
  CHECK(rlmesh_ext_cancel_token_new(&cancel) == RLMESH_OK);
  so.cancel = cancel;
  CHECK(rlmesh_session_new(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &so, &s.session) ==
        RLMESH_OK);
  CHECK(rlmesh_ext_cancel_token_cancel(cancel) == RLMESH_OK);
  rlmesh_ext_cancel_token_free(cancel);
  CHECK(rlmesh_session_reset(s.session, NULL, NULL, &obs, &info) == RLMESH_ERR_CANCELLED);
  rlmesh_session_free(s.session);
  s.session = NULL;
  RLMeshRunHooksVtable stale_vt = RLMESH_RUN_HOOKS_VTABLE_INIT;
  stale_vt.on_run_start = poison_error;
  stale_vt.on_episode_start = silent_hook_failure;
  RLMeshRunHooks* stale_hooks = NULL;
  CHECK(rlmesh_run_hooks_new(&stale_vt, NULL, &stale_hooks) == RLMESH_OK);
  o.hooks = stale_hooks;
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &o, &result) ==
        RLMESH_ERR_INTERNAL);
  CHECK(result == NULL && rlmesh_last_error_class() == RLMESH_ERROR_CLASS_RLMESH_EXCEPTION);
  CHECK(strcmp(rlmesh_last_error_message(), "hook callback failed") == 0);
  rlmesh_run_hooks_free(stale_hooks);
  o.close_env = true;
  o.hooks = NULL;
  CHECK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &o, &result) == RLMESH_OK &&
        s.closes == 1);
  rlmesh_run_result_free(result);
  RLMeshTelemetryRow* row = NULL;
  CHECK(rlmesh_telemetry_row_new("env.step", "env", "rpc.total", "ms", 2, 1, 1, 1, 1, &row) ==
        RLMESH_OK);
  const RLMeshTelemetryRow* rows[] = {row};
  const RLMeshEpisodeResult* episodes[] = {&e};
  RLMeshRunResultOptions ro = RLMESH_RUN_RESULT_OPTIONS_INIT;
  ro.episodes = episodes;
  ro.episodes_len = 1;
  ro.telemetry = rows;
  ro.telemetry_len = 1;
  CHECK(rlmesh_run_result_new(&ro, &result) == RLMESH_OK);
  rlmesh_telemetry_row_free(row);
  row = NULL;
  CHECK(rlmesh_run_result_telemetry_get(result, 0, &row) == RLMESH_OK &&
        rlmesh_telemetry_row_count(row) == 2);
  rlmesh_telemetry_row_free(row);
  RLMeshBytes text = {NULL, 0, 0};
  CHECK(rlmesh_run_result_format_telemetry(result, &text) == RLMESH_OK && text.len > 0);
  rlmesh_bytes_free(text);
  rlmesh_run_result_free(result);
  rlmesh_run_result_free(s.retained);
  rlmesh_run_context_free(s.context);
  rlmesh_run_hooks_free(s.hooks);
  CHECK(s.frees == 1);
  rlmesh_model_free(model);
  rlmesh_env_free(env);
  if (failures == 0) puts("C facade run OK");
  return failures ? 1 : 0;
}
