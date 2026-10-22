/* ABI 5 Model construction, load ordering, ownership, contexts and serving. */
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
  int loads, closes, frees, loaded_reads;
} State;
static RLMeshStatus predict(void* ud, const RLMeshValue* obs, const RLMeshPredictContext* ctx,
                            RLMeshValue** out) {
  (void)ud;
  CHECK(ctx == NULL);
  *out = rlmesh_value_clone(obs);
  return *out ? RLMESH_OK : RLMESH_ERR_MODEL;
}
static RLMeshStatus load(void* ud, const RLMeshValue* config) {
  State* s = (State*)ud;
  ++s->loads;
  CHECK(rlmesh_value_kind(config) == RLMESH_VALUE_DICT);
  return RLMESH_OK;
}
static RLMeshStatus close_model(void* ud) {
  ++((State*)ud)->closes;
  return RLMESH_OK;
}
static void free_model(void* ud) { ++((State*)ud)->frees; }
static RLMeshStatus loaded(void* ud, bool* has, uint32_t* n) {
  ++((State*)ud)->loaded_reads;
  *has = false;
  *n = 0;
  return RLMESH_OK;
}
static RLMeshStatus decline_load(void* ud, const RLMeshValue* config) {
  (void)ud;
  (void)config;
  rlmesh_callback_set_error("load failed", false);
  return RLMESH_ERR_MODEL;
}
static RLMeshStatus decline_adapter_load(void* ud, const RLMeshValue* config) {
  (void)ud;
  (void)config;
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR,
                                  "adapter unavailable");
  return RLMESH_ERR_INVALID_ARGUMENT;
}
typedef struct PayloadState {
  int typed, frees;
} PayloadState;
static const char payload_tag;
static void free_payload(void* p) {
  ++((PayloadState*)p)->frees;
  (void)rlmesh_last_error_class(); /* owner deleters may reenter the library */
}
static RLMeshStatus payload_load(void* ud, const RLMeshValue* config) {
  (void)config;
  PayloadState* s = (PayloadState*)ud;
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_VALUE_ERROR, "foreign payload");
  if (s->typed)
    rlmesh_ext_callback_set_tagged_error_payload(s, free_payload, &payload_tag);
  else
    rlmesh_ext_callback_set_error_payload(s, free_payload);
  return RLMESH_ERR_INVALID_ARGUMENT;
}
static void payload_ownership(void) {
  for (int typed = 0; typed < 2; ++typed) {
    PayloadState state = {typed, 0};
    RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
    vt.predict = predict;
    vt.load = payload_load;
    RLMeshModel* model = NULL;
    CHECK(rlmesh_model_new(&vt, &state, NULL, &model) == RLMESH_ERR_INVALID_ARGUMENT && !model);
    void* payload = NULL;
    void (*release)(void*) = NULL;
    const void* tag = NULL;
    CHECK(rlmesh_ext_last_error_take_payload_owned(NULL, &release, &tag) ==
          RLMESH_ERR_INVALID_ARGUMENT);
    CHECK(state.frees == 0);
    CHECK(rlmesh_ext_last_error_take_payload_owned(&payload, &release, &tag) == RLMESH_OK);
    CHECK(payload == &state && release == free_payload && tag == (typed ? &payload_tag : NULL));
    CHECK(state.frees == 0);
    release(payload);
    CHECK(state.frees == 1);
    CHECK(rlmesh_ext_last_error_take_payload_owned(&payload, &release, &tag) == RLMESH_OK);
    CHECK(!payload && !release && !tag && state.frees == 1);
  }
}
int main(void) {
  payload_ownership();
  State state = {0};
  RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
  vt.predict = predict;
  vt.load = load;
  vt.close = close_model;
  vt.free_user_data = free_model;
  vt.loaded_native_chunk = loaded;
  vt.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshModel* model = NULL;
  vt.context_corners = RLMESH_CORNER_PREDICT_BATCH;
  CHECK(rlmesh_model_new(&vt, &state, NULL, &model) == RLMESH_ERR_TYPE && model == NULL);
  CHECK(state.loads == 0 && state.frees == 0 && state.closes == 0);
  vt.context_corners = 0;
  vt.has_native_chunk = true;
  vt.native_chunk = 1;
  CHECK(rlmesh_model_new(&vt, &state, NULL, &model) == RLMESH_ERR_TYPE && state.loads == 0);
  vt.has_native_chunk = false;
  CHECK(rlmesh_model_new(&vt, &state, NULL, &model) == RLMESH_OK && model);
  CHECK(state.loads == 1 && state.loaded_reads == 1 && state.frees == 0);
  CHECK(rlmesh_model_allow_fusion(model));
  bool has = true;
  uint32_t chunk = 99;
  rlmesh_model_native_chunk(model, &has, &chunk);
  CHECK(!has);
  RLMeshSpecArg spec = {RLMESH_SPEC_KIND_INVALID, NULL};
  CHECK(rlmesh_model_spec(model, &spec) == RLMESH_OK && spec.kind == RLMESH_SPEC_KIND_NONE);
  RLMeshBytes edition = {NULL, 0, 0};
  CHECK(rlmesh_model_workflow_edition(model, &has, &edition) == RLMESH_OK && has);
  rlmesh_bytes_free(edition);
  RLMeshModel* clone = NULL;
  CHECK(rlmesh_model_clone(model, &clone) == RLMESH_OK);
  rlmesh_model_free(model);
  CHECK(state.closes == 0 && state.frees == 0);
  RLMeshExtCancelToken* cancel = NULL;
  CHECK(rlmesh_ext_cancel_token_new(&cancel) == RLMESH_OK);
  CHECK(rlmesh_ext_cancel_token_cancel(cancel) == RLMESH_OK);
  RLMeshModelServeOptions so = RLMESH_MODEL_SERVE_OPTIONS_INIT;
  so.cancel = cancel;
  CHECK(rlmesh_model_serve(clone, ":0/env", &so) == RLMESH_ERR_CANCELLED && state.closes == 0);
  rlmesh_ext_cancel_token_free(cancel);
  CHECK(rlmesh_model_close(clone) == RLMESH_OK && state.closes == 1);
  CHECK(rlmesh_model_close(clone) == RLMESH_OK && state.closes == 1);
  CHECK(rlmesh_model_serve(clone, ":0/env", NULL) == RLMESH_ERR_RUNTIME);
  CHECK(state.frees == 0);
  rlmesh_model_free(clone);
  CHECK(state.frees == 1);

  State failure = {0};
  vt.load = decline_load;
  CHECK(rlmesh_model_new(&vt, &failure, NULL, &model) == RLMESH_ERR_MODEL && model == NULL);
  CHECK(failure.frees == 0 && failure.closes == 0);
  vt.load = decline_adapter_load;
  CHECK(rlmesh_model_new(&vt, &failure, NULL, &model) == RLMESH_ERR_INVALID_ARGUMENT &&
        model == NULL);
  CHECK(rlmesh_last_error_class() == RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR);
  CHECK(strcmp(rlmesh_last_error_message(), "adapter unavailable") == 0);
  CHECK(failure.frees == 0 && failure.closes == 0);
  vt.load = load;
  RLMeshValue* config = NULL;
  RLMeshValue* value = NULL;
  CHECK(rlmesh_value_int(1, &value) == RLMESH_OK);
  RLMeshStr key = rlmesh_str("weights");
  const RLMeshValue* values[] = {value};
  CHECK(rlmesh_value_dict(&key, values, 1, &config) == RLMESH_OK);
  rlmesh_value_free(value);
  RLMeshLoadKey load_key = {"weights", true, {0}};
  vt.load_keys = &load_key;
  vt.load_keys_len = 1;
  CHECK(rlmesh_model_from_config(&vt, &failure, config, &model) == RLMESH_OK);
  rlmesh_model_free(model);
  CHECK(failure.loads == 1 && failure.closes == 1 && failure.frees == 1);
  rlmesh_value_free(config);
  State missing = {0};
  CHECK(rlmesh_value_dict(NULL, NULL, 0, &config) == RLMESH_OK);
  CHECK(rlmesh_model_from_config(&vt, &missing, config, &model) == RLMESH_ERR_INVALID_ARGUMENT &&
        missing.loads == 0 && missing.frees == 0);
  rlmesh_value_free(config);
  vt.load_keys = NULL;
  vt.load_keys_len = 0;
  State policy = {0}, hook = {0};
  RLMeshModelOptions options = RLMESH_MODEL_OPTIONS_INIT;
  options.on_close = close_model;
  options.hook_user_data = &hook;
  options.free_hook_user_data = free_model;
  CHECK(rlmesh_model_new(&vt, &policy, &options, &model) == RLMESH_OK);
  rlmesh_model_free(model);
  CHECK(policy.closes == 0 && policy.frees == 1 && hook.closes == 1 && hook.frees == 1);
  CHECK(rlmesh_predict_seed(42, 1) == rlmesh_predict_seed(42, 1));
  if (failures) return 1;
  puts("c_policy OK");
  return 0;
}
