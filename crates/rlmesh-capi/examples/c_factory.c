/* ABI 5 EnvFactory + TagParam + Variant against the shared facade. */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <rlmesh.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
static atomic_int failures;
#define CHECK(x)                                              \
  do {                                                        \
    if (!(x)) {                                               \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      ++failures;                                             \
    }                                                         \
  } while (0)
typedef struct State {
  RLMeshSpace* space;
  RLMeshEnvFactory* factory;
  atomic_int makes, prepares, closes, frees, env_closes, env_frees, reentry;
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
  (void)reward;
  (void)truncated;
  (void)info;
  *terminated = true;
  return rlmesh_value_int(1, obs);
}
static RLMeshStatus env_close(void* ud) {
  ++((State*)ud)->env_closes;
  return RLMESH_OK;
}
static void env_free(void* ud) { ++((State*)ud)->env_frees; }
static RLMeshStatus prepare(void* ud) {
  ++((State*)ud)->prepares;
  return RLMESH_OK;
}
static RLMeshStatus close_factory(void* ud) {
  ++((State*)ud)->closes;
  return RLMESH_OK;
}
static void free_factory(void* ud) { ++((State*)ud)->frees; }
static RLMeshStatus make(void* ud, const RLMeshValue* kwargs, RLMeshEnv** env,
                         RLMeshVectorEnv** vector) {
  State* s = (State*)ud;
  (void)vector;
  ++s->makes;
  const RLMeshValue* task = NULL;
  int64_t n = 0;
  CHECK(rlmesh_value_dict_get(kwargs, rlmesh_str("task"), &task) == RLMESH_OK);
  CHECK(rlmesh_value_as_int(task, &n) == RLMESH_OK && (n == 1 || n == 2));
  if (s->factory) {
    CHECK(rlmesh_env_factory_prepare(s->factory) == RLMESH_ERR_RUNTIME);
    ++s->reentry;
    RLMeshValue* options = NULL;
    CHECK(rlmesh_env_factory_reset_options(s->factory, &options) == RLMESH_OK);
    rlmesh_value_free(options);
  }
  RLMeshEnvVtable vt = RLMESH_ENV_VTABLE_INIT;
  vt.reset = reset;
  vt.step = step;
  vt.close = env_close;
  vt.free_user_data = env_free;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = s->space;
  attrs.action_space = s->space;
  attrs.id = "Factory-v1";
  return rlmesh_env_new(&vt, s, &attrs, env);
}
static RLMeshStatus enumerate(void* ud, RLMeshVariantSink* sink) {
  (void)ud;
  RLMeshValue* params = NULL;
  RLMeshStatus status = rlmesh_value_dict(NULL, NULL, 0, &params);
  if (status != RLMESH_OK) return status;
  RLMeshVariant* v = NULL;
  status = rlmesh_variant_new("default", params, NULL, &v);
  rlmesh_value_free(params);
  if (status != RLMESH_OK) return status;
  status = rlmesh_variant_sink_push(sink, v);
  if (status != RLMESH_OK) rlmesh_variant_free(v);
  return status;
}
typedef struct Serving {
  State* state;
  RLMeshExtCancelToken* cancel;
  RLMeshStatus result;
} Serving;
static void* serve_factory(void* data) {
  Serving* work = (Serving*)data;
  RLMeshEnvFactoryServeOptions opts = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
  opts.cancel = work->cancel;
  work->result = rlmesh_env_factory_serve(work->state->factory, "tcp://127.0.0.1:0", &opts);
  return NULL;
}
int main(void) {
  State state = {0};
  CHECK(rlmesh_discrete_new(4, NULL, &state.space) == RLMESH_OK);
  RLMeshValue *one = NULL, *two = NULL, *choices = NULL;
  CHECK(rlmesh_value_int(1, &one) == RLMESH_OK);
  CHECK(rlmesh_value_int(2, &two) == RLMESH_OK);
  const RLMeshValue* values[] = {one, two};
  CHECK(rlmesh_value_list(values, 2, &choices) == RLMESH_OK);
  RLMeshTagParam param = RLMESH_TAG_PARAM_INIT;
  param.name = "task";
  param.default_ = one;
  param.choices = choices;
  const RLMeshTagParam* params[] = {&param};
  const char* resets[] = {"trial_index"};
  RLMeshEnvFactoryAttributes attrs = RLMESH_ENV_FACTORY_ATTRIBUTES_INIT;
  attrs.tag_params = params;
  attrs.tag_params_len = 1;
  attrs.reset_options = resets;
  attrs.reset_options_len = 1;
  attrs.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshEnvFactoryVtable vt = RLMESH_ENV_FACTORY_VTABLE_INIT;
  vt.prepare = prepare;
  vt.make = make;
  vt.enumerate_variants = enumerate;
  vt.close = close_factory;
  vt.free_user_data = free_factory;
  RLMeshValue* empty = NULL;
  CHECK(rlmesh_value_list(NULL, 0, &empty) == RLMESH_OK);
  param.choices = empty;
  CHECK(rlmesh_env_factory_new(&vt, &state, &attrs, &state.factory) == RLMESH_ERR_TYPE &&
        state.frees == 0);
  param.choices = choices;
  CHECK(rlmesh_env_factory_new(&vt, &state, &attrs, &state.factory) == RLMESH_OK);
  rlmesh_value_free(one);
  rlmesh_value_free(two);
  rlmesh_value_free(choices);
  rlmesh_value_free(empty);
  RLMeshTagParamList* copied = NULL;
  CHECK(rlmesh_env_factory_tag_params(state.factory, &copied) == RLMESH_OK);
  CHECK(rlmesh_tag_param_list_len(copied) == 1);
  const RLMeshTagParam* lent = rlmesh_tag_param_list_get(copied, 0);
  CHECK(lent && lent->struct_size == sizeof(*lent) && strcmp(lent->name, "task") == 0);
  int64_t n = 0;
  CHECK(rlmesh_value_as_int(lent->default_, &n) == RLMESH_OK && n == 1);
  CHECK(rlmesh_tag_param_list_get(copied, 1) == NULL);
  rlmesh_tag_param_list_free(copied);
  CHECK(rlmesh_env_factory_prepare(state.factory) == RLMESH_OK && state.prepares == 1);
  RLMeshEnv* env = NULL;
  RLMeshVectorEnv* vector = NULL;
  CHECK(rlmesh_env_factory_make(state.factory, NULL, &env, &vector) == RLMESH_OK && env && !vector);
  CHECK(state.makes == 1 && state.reentry == 1);
  RLMeshEnvServer* server = NULL;
  RLMeshServeOptions serve = RLMESH_SERVE_OPTIONS_INIT;
  serve.workflow_edition = attrs.workflow_edition;
  RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
  so.options = &serve;
  CHECK(rlmesh_env_server_new(&env, 1, &so, &server) == RLMESH_OK);
  const RLMeshValue* metadata =
      rlmesh_env_contract_metadata(rlmesh_env_server_env_contract(server));
  const RLMeshValue* options = NULL;
  CHECK(rlmesh_value_dict_get(metadata, rlmesh_str(RLMESH_ENV_RESET_OPTIONS_KEY), &options) ==
            RLMESH_OK &&
        options);
  CHECK(rlmesh_value_kind(options) == RLMESH_VALUE_LIST);
  CHECK(rlmesh_value_dict_get(metadata, rlmesh_str("rlmesh.adapters.v1.env_branch"), &options) ==
            RLMESH_OK &&
        options);
  rlmesh_env_server_free(server);
  rlmesh_env_free(env);
  CHECK(state.env_closes == 1 && state.env_frees == 1);
  RLMeshValue* wrong = NULL;
  CHECK(rlmesh_value_int(99, &wrong) == RLMESH_OK);
  RLMeshStr key = rlmesh_str("task");
  const RLMeshValue* wrong_values[] = {wrong};
  RLMeshValue* kwargs = NULL;
  CHECK(rlmesh_value_dict(&key, wrong_values, 1, &kwargs) == RLMESH_OK);
  rlmesh_value_free(wrong);
  env = NULL;
  vector = NULL;
  CHECK(rlmesh_env_factory_make(state.factory, kwargs, &env, &vector) ==
            RLMESH_ERR_INVALID_ARGUMENT &&
        !env && !vector);
  CHECK(state.makes == 1);
  rlmesh_value_free(kwargs);
  RLMeshVariantList* variants = NULL;
  CHECK(rlmesh_env_factory_enumerate_variants(state.factory, &variants) == RLMESH_OK);
  CHECK(rlmesh_variant_list_len(variants) == 1);
  const RLMeshVariant* variant = rlmesh_variant_list_get(variants, 0);
  CHECK(variant != NULL);
  RLMeshBytes id = {NULL, 0, 0};
  CHECK(rlmesh_variant_id(variant, &id) == RLMESH_OK && id.len == 7);
  rlmesh_bytes_free(id);
  CHECK(rlmesh_variant_list_get(variants, 1) == NULL);
  rlmesh_variant_list_free(variants);
  RLMeshValue* map = NULL;
  CHECK(rlmesh_value_dict(NULL, NULL, 0, &map) == RLMESH_OK);
  RLMeshVariant *a = NULL, *b = NULL;
  CHECK(rlmesh_variant_new("a", map, NULL, &a) == RLMESH_OK);
  CHECK(rlmesh_variant_new("a", map, NULL, &b) == RLMESH_OK);
  CHECK(rlmesh_variant_equals(a, b));
  CHECK(rlmesh_variant_set_id(a, "b") == RLMESH_OK);
  CHECK(!rlmesh_variant_equals(a, b));
  RLMeshValue* scalar = NULL;
  CHECK(rlmesh_value_bool(true, &scalar) == RLMESH_OK);
  CHECK(rlmesh_variant_set_params(a, scalar) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(rlmesh_variant_set_metadata(a, map) == RLMESH_OK);
  rlmesh_value_free(scalar);
  rlmesh_value_free(map);
  rlmesh_variant_free(a);
  rlmesh_variant_free(b);
  RLMeshExtCancelToken* cancel = rlmesh_ext_cancel_token_new();
  rlmesh_ext_cancel_token_cancel(cancel);
  RLMeshEnvFactoryServeOptions fso = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
  fso.cancel = cancel;
  CHECK(rlmesh_env_factory_serve(state.factory, "tcp://127.0.0.1:0", &fso) == RLMESH_ERR_CANCELLED);
  CHECK(state.prepares == 1 && state.makes == 1);
  Serving work = {&state, cancel, RLMESH_OK};
  pthread_t thread;
  CHECK(pthread_create(&thread, NULL, serve_factory, &work) == 0);
  struct timespec delay = {0, 1000000};
  for (int i = 0; i < 5000 && atomic_load(&state.makes) < 2; ++i) nanosleep(&delay, NULL);
  CHECK(atomic_load(&state.makes) >= 2);
  CHECK(rlmesh_env_factory_prepare(state.factory) == RLMESH_ERR_RUNTIME);
  rlmesh_ext_cancel_token_cancel(cancel);
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(work.result == RLMESH_ERR_CANCELLED);
  CHECK(state.env_closes == state.makes && state.env_frees == state.makes);
  rlmesh_ext_cancel_token_free(cancel);
  CHECK(rlmesh_env_factory_close(state.factory) == RLMESH_OK && state.closes == 1);
  rlmesh_env_factory_free(state.factory);
  CHECK(state.frees == 1 && state.closes == 1);
  rlmesh_space_free(state.space);
  if (failures) return 1;
  puts("c_factory OK");
  return 0;
}
