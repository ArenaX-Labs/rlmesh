/* ABI 5 Describe: static declarations, factory branches and server snapshots. */
#include <rlmesh.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int failures, loads, loaded_reads, model_closes, model_frees, predicts;
#define CHECK(x)                                                                                \
  do {                                                                                          \
    if (!(x)) {                                                                                 \
      fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, rlmesh_last_error_message()); \
      ++failures;                                                                               \
    }                                                                                           \
  } while (0)
static const RLMeshValue* get(const RLMeshValue* v, const char* k) {
  const RLMeshValue* out = NULL;
  CHECK(rlmesh_value_dict_get(v, rlmesh_str(k), &out) == RLMESH_OK && out);
  return out;
}
static const RLMeshValue* item(const RLMeshValue* v, size_t i) {
  const RLMeshValue* out = NULL;
  CHECK(rlmesh_value_get(v, i, &out) == RLMESH_OK && out);
  return out;
}
static bool text_eq(const RLMeshValue* v, const char* expected) {
  RLMeshStr text = {NULL, 0};
  return rlmesh_value_as_text(v, &text) == RLMESH_OK && text.len == strlen(expected) &&
         memcmp(text.data, expected, text.len) == 0;
}
static bool has(const RLMeshBytes* bytes, const char* needle) {
  char* text = (char*)malloc(bytes->len + 1);
  CHECK(text != NULL);
  if (!text) return false;
  memcpy(text, bytes->data, bytes->len);
  text[bytes->len] = 0;
  bool yes = strstr(text, needle) != NULL;
  free(text);
  return yes;
}
static RLMeshStatus predict(void* u, const RLMeshValue* obs, const RLMeshPredictContext* c,
                            RLMeshValue** out) {
  (void)u;
  (void)obs;
  (void)c;
  (void)out;
  ++predicts;
  return RLMESH_ERR_MODEL;
}
static RLMeshStatus chunk(void* u, const RLMeshValue* obs, uint32_t h,
                          const RLMeshPredictContext* c, RLMeshValue** out) {
  (void)h;
  return predict(u, obs, c, out);
}
static RLMeshStatus load(void* u, const RLMeshValue* config) {
  (void)u;
  (void)config;
  ++loads;
  return RLMESH_OK;
}
static RLMeshStatus loaded(void* u, bool* has_chunk, uint32_t* n) {
  (void)u;
  ++loaded_reads;
  *has_chunk = true;
  *n = 5;
  return RLMESH_OK;
}
static RLMeshStatus close_model(void* u) {
  (void)u;
  ++model_closes;
  return RLMESH_OK;
}
static void free_model(void* u) {
  (void)u;
  ++model_frees;
}
static void models(void) {
  RLMeshModelVtable v = RLMESH_MODEL_VTABLE_INIT;
  v.predict = predict;
  v.predict_chunk = chunk;
  v.load = load;
  v.loaded_native_chunk = loaded;
  v.close = close_model;
  v.free_user_data = free_model;
  v.has_native_chunk = true;
  v.native_chunk = 3;
  v.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshLoadKey keys[] = {{"weights", true, {0}}, {"device", false, {0}}};
  v.load_keys = keys;
  v.load_keys_len = 2;
  RLMeshDescribeOptions o = RLMESH_DESCRIBE_OPTIONS_INIT;
  o.has_kind = true;
  o.kind = RLMESH_DESCRIBE_KIND_MODEL;
  o.generated_at = "2026-10-15T12:00:00Z";
  RLMeshValue* desc = NULL;
  CHECK(rlmesh_model_vtable_describe(&v, &o, &desc) == RLMESH_OK);
  CHECK(loads == 0 && loaded_reads == 0 && predicts == 0 && model_closes == 0 && model_frees == 0);
  CHECK(text_eq(get(desc, "kind"), "model"));
  CHECK(text_eq(get(get(desc, "target"), "qualname"), "native:model"));
  CHECK(text_eq(get(get(desc, "runtime"), "component"), "rlmesh-capi"));
  CHECK(text_eq(get(get(desc, "runtime"), "language"), "c"));
  CHECK(text_eq(get(desc, "generated_at"), o.generated_at));
  const RLMeshValue* tier = get(get(desc, "params"), "signature_tier");
  CHECK(text_eq(get(item(tier, 0), "name"), "weights"));
  CHECK(text_eq(get(item(tier, 1), "name"), "device"));
  bool required = false;
  CHECK(rlmesh_value_as_bool(get(item(tier, 0), "required"), &required) == RLMESH_OK && required);
  rlmesh_value_free(desc);
  RLMeshDescribeJsonOptions jo = RLMESH_DESCRIBE_JSON_OPTIONS_INIT;
  RLMeshBytes json = {NULL, 0, 0};
  CHECK(rlmesh_model_vtable_describe_json(&v, &jo, &json) == RLMESH_OK &&
        has(&json, "\"native_chunk\":3"));
  rlmesh_bytes_free(json);
  RLMeshDescribeJsonOptions json_zero = {0};
  CHECK(rlmesh_model_vtable_describe_json(&v, &json_zero, &json) == RLMESH_ERR_INVALID_ARGUMENT &&
        json.data == NULL);
  struct JsonTail {
    RLMeshDescribeJsonOptions o;
    size_t extra;
  } jt = {RLMESH_DESCRIBE_JSON_OPTIONS_INIT, 1};
  jt.o.struct_size = sizeof(jt);
  CHECK(rlmesh_model_vtable_describe_json(&v, &jt.o, &json) == RLMESH_ERR_TYPE &&
        json.data == NULL);
  jt.extra = 0;
  CHECK(rlmesh_model_vtable_describe_json(&v, &jt.o, &json) == RLMESH_OK);
  rlmesh_bytes_free(json);
  o.kind = RLMESH_DESCRIBE_KIND_ENV;
  CHECK(rlmesh_model_vtable_describe(&v, &o, &desc) == RLMESH_ERR_INVALID_ARGUMENT && desc == NULL);
  o.kind = RLMESH_DESCRIBE_KIND_INVALID;
  CHECK(rlmesh_model_vtable_describe(&v, &o, &desc) == RLMESH_ERR_INVALID_ARGUMENT);
  o.has_kind = false;
  o.generated_at = "yesterday";
  CHECK(rlmesh_model_vtable_describe(&v, &o, &desc) == RLMESH_ERR_INVALID_ARGUMENT);
  o.generated_at = NULL;
  RLMeshDescribeOptions zero = {0};
  CHECK(rlmesh_model_vtable_describe(&v, &zero, &desc) == RLMESH_ERR_INVALID_ARGUMENT);
  struct Extra {
    RLMeshDescribeOptions o;
    size_t tail;
  } ext = {RLMESH_DESCRIBE_OPTIONS_INIT, 1};
  ext.o.struct_size = sizeof(ext);
  CHECK(rlmesh_model_vtable_describe(&v, &ext.o, &desc) == RLMESH_ERR_TYPE &&
        strstr(rlmesh_last_error_message(), "RLMeshDescribeOptions") != NULL);
  ext.tail = 0;
  CHECK(rlmesh_model_vtable_describe(&v, &ext.o, &desc) == RLMESH_OK);
  rlmesh_value_free(desc);
  /* Type form shares declaration validation with Model construction, never load. */
  RLMeshModel* model = NULL;
  uint32_t saved = v.context_corners;
  v.context_corners = RLMESH_CORNER_PREDICT_BATCH;
  CHECK(rlmesh_model_vtable_describe(&v, NULL, &desc) == RLMESH_ERR_TYPE);
  CHECK(rlmesh_model_new(&v, NULL, NULL, &model) == RLMESH_ERR_TYPE);
  v.context_corners = saved;
  v.predict_chunk = NULL;
  CHECK(rlmesh_model_vtable_describe(&v, NULL, &desc) == RLMESH_ERR_TYPE);
  CHECK(rlmesh_model_new(&v, NULL, NULL, &model) == RLMESH_ERR_TYPE);
  v.predict_chunk = chunk;
  v.native_chunk = 0;
  CHECK(rlmesh_model_vtable_describe(&v, NULL, &desc) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(rlmesh_model_new(&v, NULL, NULL, &model) == RLMESH_ERR_INVALID_ARGUMENT);
  v.native_chunk = 3;
  CHECK(loads == 0 && loaded_reads == 0 && model_frees == 0 && model_closes == 0);
  /* A refused declaration edition is recorded, rather than failing describe. */
  v.workflow_edition = "2099.01";
  CHECK(rlmesh_model_vtable_describe(&v, NULL, &desc) == RLMESH_OK);
  CHECK(get(get(desc, "runtime"), "workflow_edition_error") != NULL);
  rlmesh_value_free(desc);
  jo.workflow_edition = "arbitrary-not-validated";
  CHECK(rlmesh_model_vtable_describe_json(&v, &jo, &json) == RLMESH_OK &&
        has(&json, "arbitrary-not-validated") && !has(&json, "workflow_edition_error"));
  rlmesh_bytes_free(json);
  jo.workflow_edition = "";
  CHECK(rlmesh_model_vtable_describe_json(&v, &jo, &json) == RLMESH_OK &&
        !has(&json, "workflow_edition_error"));
  rlmesh_bytes_free(json);
  v.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshValue *weights = NULL, *config = NULL;
  CHECK(rlmesh_value_text(rlmesh_str("checkpoint"), &weights) == RLMESH_OK);
  RLMeshStr name = rlmesh_str("weights");
  const RLMeshValue* values[] = {weights};
  CHECK(rlmesh_value_dict(&name, values, 1, &config) == RLMESH_OK);
  CHECK(rlmesh_model_from_config(&v, NULL, config, &model) == RLMESH_OK);
  rlmesh_value_free(config);
  rlmesh_value_free(weights);
  CHECK(loads == 1 && loaded_reads == 1);
  CHECK(rlmesh_model_describe(model, NULL, &desc) == RLMESH_OK);
  int64_t n = 0;
  CHECK(rlmesh_value_as_int(get(desc, "native_chunk"), &n) == RLMESH_OK && n == 3);
  rlmesh_value_free(desc);
  CHECK(loads == 1 && loaded_reads == 1 && predicts == 0);
  CHECK(rlmesh_model_close(model) == RLMESH_OK);
  CHECK(rlmesh_model_describe_json(model, NULL, &json) == RLMESH_OK && has(&json, "native:model"));
  rlmesh_bytes_free(json);
  rlmesh_model_free(model);
  CHECK(model_closes == 1 && model_frees == 1);
}
typedef struct Factory {
  int prepares, makes, closes, frees;
  bool fail;
  RLMeshEnvFactory* handle;
} Factory;
static RLMeshStatus reset(void* u, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  (void)u;
  (void)seed;
  (void)options;
  (void)obs;
  (void)info;
  CHECK(false);
  return RLMESH_ERR_ENVIRONMENT;
}
static RLMeshStatus step(void* u, const RLMeshValue* a, RLMeshValue** o, double* r, bool* t,
                         bool* x, RLMeshValue** i) {
  (void)u;
  (void)a;
  (void)o;
  (void)r;
  (void)t;
  (void)x;
  (void)i;
  CHECK(false);
  return RLMESH_ERR_ENVIRONMENT;
}
static RLMeshStatus close_env(void* u) {
  ++((Factory*)u)->closes;
  return RLMESH_OK;
}
static void free_env(void* u) { ++((Factory*)u)->frees; }
static RLMeshStatus env(Factory* f, int64_t n, const char* id, RLMeshEnv** out) {
  RLMeshDiscrete* space = NULL;
  CHECK(rlmesh_discrete_new(n, NULL, &space) == RLMESH_OK);
  RLMeshEnvVtable v = RLMESH_ENV_VTABLE_INIT;
  v.reset = reset;
  v.step = step;
  v.close = close_env;
  v.free_user_data = free_env;
  RLMeshEnvAttributes a = RLMESH_ENV_ATTRIBUTES_INIT;
  a.observation_space = space;
  a.action_space = space;
  a.id = id;
  RLMeshStatus s = rlmesh_env_new(&v, f, &a, out);
  rlmesh_space_free(space);
  return s;
}
static RLMeshStatus prepare(void* u) {
  Factory* f = (Factory*)u;
  ++f->prepares;
  if (f->fail) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TYPE_ERROR, "prepare badge");
    return RLMESH_ERR_TYPE;
  }
  return RLMESH_OK;
}
static RLMeshStatus make(void* u, const RLMeshValue* kwargs, RLMeshEnv** out,
                         RLMeshVectorEnv** vector) {
  Factory* f = (Factory*)u;
  (void)vector;
  ++f->makes;
  int64_t n = 0;
  CHECK(rlmesh_value_as_int(get(kwargs, "size"), &n) == RLMESH_OK);
  if (f->handle) {
    RLMeshValue* nested = NULL;
    CHECK(rlmesh_env_factory_describe(f->handle, NULL, &nested) == RLMESH_ERR_RUNTIME &&
          nested == NULL);
  }
  return env(f, n, "DescribeFactory-v1", out);
}
static void factories(void) {
  Factory f = {0};
  RLMeshValue *one = NULL, *two = NULL, *choices = NULL;
  CHECK(rlmesh_value_int(1, &one) == RLMESH_OK);
  CHECK(rlmesh_value_int(2, &two) == RLMESH_OK);
  const RLMeshValue* values[] = {one, two};
  CHECK(rlmesh_value_list(values, 2, &choices) == RLMESH_OK);
  RLMeshTagParam p = RLMESH_TAG_PARAM_INIT;
  p.name = "size";
  p.default_ = one;
  p.choices = choices;
  const RLMeshTagParam* params[] = {&p};
  RLMeshEnvFactoryAttributes a = RLMESH_ENV_FACTORY_ATTRIBUTES_INIT;
  a.tag_params = params;
  a.tag_params_len = 1;
  a.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshEnvFactoryVtable v = RLMESH_ENV_FACTORY_VTABLE_INIT;
  v.prepare = prepare;
  v.make = make;
  CHECK(rlmesh_env_factory_new(&v, &f, &a, &f.handle) == RLMESH_OK);
  rlmesh_value_free(choices);
  rlmesh_value_free(one);
  rlmesh_value_free(two);
  RLMeshValue* desc = NULL;
  CHECK(rlmesh_env_factory_describe(f.handle, NULL, &desc) == RLMESH_OK);
  CHECK(f.prepares == 2 && f.makes == 2 && f.closes == 2 && f.frees == 2);
  CHECK(text_eq(get(get(desc, "target"), "qualname"), "native:DescribeFactory-v1"));
  size_t len = 0;
  CHECK(rlmesh_value_len(get(get(desc, "env_contracts"), "branches"), &len) == RLMESH_OK &&
        len == 2);
  CHECK(text_eq(get(item(get(get(desc, "params"), "signature_tier"), 0), "name"), "size"));
  rlmesh_value_free(desc);
  RLMeshEnv* existing = NULL;
  CHECK(env(&f, 7, "Existing-v1", &existing) == RLMESH_OK);
  const RLMeshEnv* lanes[] = {existing};
  RLMeshDescribeJsonOptions o = RLMESH_DESCRIBE_JSON_OPTIONS_INIT;
  o.served_env = lanes;
  o.served_env_len = 1;
  RLMeshBytes json = {NULL, 0, 0};
  int made = f.makes, closed = f.closes;
  CHECK(rlmesh_env_factory_describe_json(f.handle, &o, &json) == RLMESH_OK &&
        has(&json, "native:Existing-v1") && has(&json, "\"n\":7"));
  rlmesh_bytes_free(json);
  CHECK(f.makes == made + 1 && f.closes == closed + 1);
  o.served_env_len = 0;
  CHECK(rlmesh_env_factory_describe_json(f.handle, &o, &json) == RLMESH_ERR_INVALID_ARGUMENT &&
        json.data == NULL);
  o.served_env = NULL;
  o.served_env_len = 1;
  CHECK(rlmesh_env_factory_describe_json(f.handle, &o, &json) == RLMESH_ERR_INVALID_ARGUMENT);
  o.served_env_len = 0;
  RLMeshDiscrete* space = NULL;
  CHECK(rlmesh_discrete_new(9, NULL, &space) == RLMESH_OK);
  RLMeshVectorEnvVtable vv = RLMESH_VECTOR_ENV_VTABLE_INIT;
  RLMeshVectorEnv* vector = NULL;
  /* Dummy callbacks are never invoked by describe; match their vector signatures below. */
  extern RLMeshStatus vector_reset(void*, const int64_t*, size_t, const RLMeshValue*, RLMeshValue**,
                                   RLMeshValue**);
  extern RLMeshStatus vector_step(void*, const RLMeshValue*, RLMeshValue**, double*, bool*, bool*,
                                  RLMeshValue**);
  vv.reset = vector_reset;
  vv.step = vector_step;
  RLMeshVectorEnvAttributes va = RLMESH_VECTOR_ENV_ATTRIBUTES_INIT;
  va.num_envs = 3;
  va.single_observation_space = space;
  va.single_action_space = space;
  va.id = "VectorDescribe-v1";
  CHECK(rlmesh_vector_env_new(&vv, NULL, &va, &vector) == RLMESH_OK);
  rlmesh_space_free(space);
  o.served_vector_env = vector;
  o.served_env = lanes;
  o.served_env_len = 1;
  CHECK(rlmesh_env_factory_describe_json(f.handle, &o, &json) == RLMESH_ERR_INVALID_ARGUMENT);
  o.served_env = NULL;
  o.served_env_len = 0;
  CHECK(rlmesh_env_factory_describe_json(f.handle, &o, &json) == RLMESH_OK &&
        has(&json, "native:VectorDescribe-v1") && has(&json, "\"num_envs\":3"));
  rlmesh_bytes_free(json);
  RLMeshServeOptions serve = RLMESH_SERVE_OPTIONS_INIT;
  serve.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshEnvServerOptions so = RLMESH_ENV_SERVER_OPTIONS_INIT;
  so.address = "tcp://127.0.0.1:0";
  so.options = &serve;
  so.close_env_on_shutdown = false;
  RLMeshExtEnvServerOptions ext = RLMESH_EXT_ENV_SERVER_OPTIONS_INIT;
  ext.publish_describe = false;
  ext.host_runtime_json = "{\"language\":\"c++\",\"language_version\":\"20\"}";
  so.ext = &ext;
  RLMeshEnvServer* server = NULL;
  CHECK(rlmesh_env_server_new_vector(vector, &so, &server) == RLMESH_OK);
  RLMeshBytes before = {NULL, 0, 0};
  CHECK(rlmesh_ext_env_server_describe_json(server, &before) == RLMESH_OK &&
        has(&before, "native:VectorDescribe-v1") && has(&before, "\"num_envs\":3") &&
        has(&before, "\"language\":\"c++\""));
  CHECK(rlmesh_env_server_start(server) == RLMESH_OK);
  CHECK(rlmesh_env_server_shutdown(server) == RLMESH_OK);
  CHECK(rlmesh_ext_env_server_describe_json(server, &json) == RLMESH_OK && before.len == json.len &&
        memcmp(before.data, json.data, json.len) == 0);
  rlmesh_bytes_free(before);
  rlmesh_bytes_free(json);
  rlmesh_env_server_free(server);
  rlmesh_vector_env_free(vector);
  CHECK(rlmesh_env_server_new(&existing, 1, &so, &server) == RLMESH_OK);
  CHECK(rlmesh_ext_env_server_describe_json(server, &json) == RLMESH_OK &&
        has(&json, "native:Existing-v1") && !has(&json, "num_envs"));
  rlmesh_bytes_free(json);
  rlmesh_env_server_free(server);
  CHECK(f.closes == f.frees);
  rlmesh_env_free(existing);
  f.fail = true;
  CHECK(rlmesh_env_factory_describe(f.handle, NULL, &desc) == RLMESH_OK);
  CHECK(text_eq(get(get(desc, "target"), "qualname"), "native:UnknownEnv-v1"));
  CHECK(text_eq(get(get(desc, "env_spec"), "error_type"), "TypeError"));
  rlmesh_value_free(desc);
  rlmesh_env_factory_free(f.handle);
}
RLMeshStatus vector_reset(void* u, const int64_t* s, size_t n, const RLMeshValue* o,
                          RLMeshValue** v, RLMeshValue** i) {
  (void)n;
  return reset(u, s, o, v, i);
}
RLMeshStatus vector_step(void* u, const RLMeshValue* a, RLMeshValue** o, double* r, bool* t,
                         bool* x, RLMeshValue** i) {
  return step(u, a, o, r, t, x, i);
}
int main(void) {
  models();
  factories();
  if (!failures) puts("C facade describe OK");
  return failures ? 1 : 0;
}
