/* Stable adapters: typed authoring, immutable borrows, engine execution and ownership. */
#include <rlmesh.h>
#include <stdio.h>
#include <string.h>
static int failures;
#define CHECK(x)                                                                                \
  do {                                                                                          \
    if (!(x)) {                                                                                 \
      fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, rlmesh_last_error_message()); \
      ++failures;                                                                               \
    }                                                                                           \
  } while (0)
static RLMeshValue* vector(double number) {
  const size_t shape[] = {1};
  RLMeshTensor* tensor = NULL;
  RLMeshValue* value = NULL;
  CHECK(rlmesh_tensor_new(&number, sizeof(number), shape, 1, RLMESH_DTYPE_FLOAT64, &tensor) ==
        RLMESH_OK);
  CHECK(rlmesh_value_tensor(tensor, &value) == RLMESH_OK);
  rlmesh_tensor_free(tensor);
  return value;
}
static RLMeshStatus identity(void* user, const RLMeshValue* value, RLMeshValue** out) {
  if (user) ++*(int*)user;
  *out = rlmesh_value_clone(value);
  return *out ? RLMESH_OK : RLMESH_ERR_TYPE;
}
static void counted_free(void* user) { ++*(int*)user; }
static RLMeshStatus fail_transform(void* user, const RLMeshValue* value, RLMeshValue** out) {
  (void)user;
  *out = rlmesh_value_clone(value);
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_OS_ERROR, "transform failed");
  return RLMESH_ERR_OS;
}
static RLMeshStatus reset(void* user, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  (void)user;
  (void)seed;
  (void)options;
  (void)info;
  *obs = vector(2.5);
  return RLMESH_OK;
}
static RLMeshStatus step(void* user, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                         bool* terminated, bool* truncated, RLMeshValue** info) {
  (void)user;
  (void)action;
  (void)info;
  *obs = vector(3.5);
  *reward = 1.;
  *terminated = true;
  *truncated = false;
  return RLMESH_OK;
}
static RLMeshStatus predict(void* user, const RLMeshValue* obs, const RLMeshPredictContext* context,
                            RLMeshValue** out) {
  (void)context;
  return identity(user, obs, out);
}
static void layouts(void);
int main(void) {
  layouts();
  bool equal = false;
  CHECK(rlmesh_adapters_parts_len() == 7 &&
        strcmp(rlmesh_adapters_parts_get(0), RLMESH_ADAPTERS_LEFT_ARM) == 0 &&
        !rlmesh_adapters_parts_get(99));
  uint32_t dims = 99;
  RLMeshRotationEncoding encoding = RLMESH_ROTATION_ENCODING_OTHER;
  CHECK(rlmesh_rotation_encoding_dims(encoding, &dims) == RLMESH_ERR_INVALID_ARGUMENT &&
        dims == 99);
  CHECK(rlmesh_adapters_rotation_dims_get(0, &encoding, &dims) == RLMESH_OK &&
        encoding == RLMESH_ROTATION_ENCODING_AXIS_ANGLE && dims == 3);
  CHECK(rlmesh_adapters_rotation_dims_get(99, &encoding, &dims) == RLMESH_OK &&
        encoding == RLMESH_ROTATION_ENCODING_INVALID && dims == 0);
  CHECK(!rlmesh_rotation_encoding_str(RLMESH_ROTATION_ENCODING_OTHER));
  RLMeshActuator actuator = RLMESH_ACTUATOR_INIT;
  actuator.role = RLMESH_ADAPTERS_ACTION_JOINT_POS;
  actuator.dim = 1;
  const RLMeshActuator* components[] = {&actuator};
  RLMeshAction action = RLMESH_ACTION_INIT;
  action.components = components;
  action.components_len = 1;
  CHECK(rlmesh_action_dim(&action, &dims) == RLMESH_OK && dims == 1);
  RLMeshState state = RLMESH_STATE_INIT;
  state.role = RLMESH_ADAPTERS_JOINT_POS;
  state.has_dim = true;
  state.dim = 1;
  RLMeshInputNode input = rlmesh_input_node_state(&state);
  RLMeshModelSpec* spec = NULL;
  CHECK(rlmesh_model_spec_new(&input, &action, &spec) == RLMESH_OK && spec);
  const RLMeshState* lent = (const RLMeshState*)rlmesh_model_spec_input(spec)->leaf;
  CHECK(lent->struct_size == sizeof(*lent) && lent->dtype == RLMESH_DTYPE_FLOAT32 &&
        strcmp(lent->role, state.role) == 0);
  RLMeshModelSpec* same = rlmesh_model_spec_clone(spec);
  CHECK(rlmesh_model_spec_equals(spec, same));
  rlmesh_model_spec_free(same);
  RLMeshBytes json = {0};
  CHECK(rlmesh_model_spec_to_json(spec, &json) == RLMESH_OK);
  char text[4096];
  CHECK(json.len < sizeof(text));
  memcpy(text, json.data, json.len);
  text[json.len] = 0;
  rlmesh_bytes_free(json);
  CHECK(strstr(text, "\"dtype\":\"float32\"") != NULL && !strstr(text, ": "));
  RLMeshModelSpec* parsed = NULL;
  CHECK(rlmesh_model_spec_from_json(text, &parsed) == RLMESH_OK &&
        rlmesh_model_spec_equals(spec, parsed));
  rlmesh_model_spec_free(parsed);
  RLMeshValue* metadata = NULL;
  CHECK(rlmesh_model_spec_to_metadata(spec, &metadata) == RLMESH_OK);
  CHECK(rlmesh_model_spec_from_metadata(metadata, &parsed) == RLMESH_OK &&
        rlmesh_model_spec_equals(spec, parsed));
  rlmesh_model_spec_free(parsed);
  rlmesh_value_free(metadata);
  CHECK(rlmesh_value_dict(NULL, NULL, 0, &metadata) == RLMESH_OK);
  parsed = (RLMeshModelSpec*)1;
  CHECK(rlmesh_model_spec_from_metadata(metadata, &parsed) == RLMESH_OK && !parsed);
  rlmesh_value_free(metadata);
  /* Sized leaves cover sugar, binary-safe text, composites and closed enums. */
  RLMeshImage image = RLMESH_IMAGE_INIT;
  image.role = RLMESH_ADAPTERS_IMAGE_PRIMARY;
  CHECK(rlmesh_image_with_size(&image, 4) == RLMESH_OK && image.height == 4 && image.width == 4);
  CHECK(rlmesh_image_with_size(&image, 8) == RLMESH_ERR_INVALID_ARGUMENT);
  image.stack = 2;
  image.has_stride = true;
  image.stride = 2;
  RLMeshInputNode image_node = rlmesh_input_node_image(&image);
  RLMeshModelSpec* image_spec = NULL;
  CHECK(rlmesh_model_spec_new(&image_node, &action, &image_spec) == RLMESH_OK);
  const RLMeshImage* image_view = (const RLMeshImage*)rlmesh_model_spec_input(image_spec)->leaf;
  CHECK(!image_view->has_stride && image_view->offsets_len == 2 && image_view->offsets[0] == -2 &&
        image_view->offsets[1] == 0);
  rlmesh_model_spec_free(image_spec);
  image.stride = 0;
  CHECK(rlmesh_model_spec_new(&image_node, &action, &image_spec) == RLMESH_ERR_INVALID_ARGUMENT &&
        !image_spec);
  RLMeshAdaptersText leaf_text = RLMESH_ADAPTERS_TEXT_INIT;
  leaf_text.role = RLMESH_ADAPTERS_INSTRUCTION;
  leaf_text.has_fill = true;
  leaf_text.fill = (RLMeshStr){"a\0b", 3};
  RLMeshInputNode text_node = rlmesh_input_node_text(&leaf_text);
  CHECK(rlmesh_model_spec_new(&text_node, &action, &image_spec) == RLMESH_OK);
  const RLMeshAdaptersText* text_view =
      (const RLMeshAdaptersText*)rlmesh_model_spec_input(image_spec)->leaf;
  CHECK(text_view->fill.len == 3 && text_view->fill.data[1] == 0);
  rlmesh_model_spec_free(image_spec);
  RLMeshConstant constant = RLMESH_CONSTANT_INIT;
  RLMeshConcatPart parts[] = {rlmesh_concat_part_state(&state),
                              rlmesh_concat_part_constant(&constant)};
  RLMeshConcat concat = RLMESH_CONCAT_INIT;
  concat.parts = parts;
  concat.parts_len = 2;
  RLMeshInputNode concat_node = rlmesh_input_node_concat(&concat);
  CHECK(rlmesh_model_spec_new(&concat_node, &action, &image_spec) == RLMESH_OK &&
        rlmesh_model_spec_input(image_spec)->kind == RLMESH_INPUT_NODE_KIND_CONCAT);
  rlmesh_model_spec_free(image_spec);
  RLMeshState previous = RLMESH_STATE_INIT;
  CHECK(rlmesh_adapters_previous(actuator.role, NULL, &previous) == RLMESH_OK &&
        previous.source == RLMESH_STATE_SOURCE_ACTION && previous.role == actuator.role);
  const double matrix[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  double rotation_values[6];
  RLMeshRotation rotation = RLMESH_ROTATION_INIT;
  CHECK(rlmesh_rotation_from_matrix(matrix, rotation_values, &rotation) == RLMESH_OK &&
        rotation.encoding == RLMESH_ROTATION_ENCODING_ROT6D && rotation.value == rotation_values &&
        rotation.value_len == 6 && rotation_values[0] == 1 && rotation_values[4] == 1);
  RLMeshCustomEncoding encoding_spec = RLMESH_CUSTOM_ENCODING_INIT;
  encoding_spec.base = RLMESH_ROTATION_ENCODING_ROT6D;
  CHECK(rlmesh_custom_encoding_width(&encoding_spec, &dims) == RLMESH_OK && dims == 6);
  CHECK(rlmesh_custom_encoding_is_entrypoint(&encoding_spec, &equal) == RLMESH_OK && !equal);
  encoding_spec.to_base.entrypoint = "module:rotation";
  CHECK(rlmesh_custom_encoding_is_entrypoint(&encoding_spec, &equal) == RLMESH_OK && equal);
  encoding_spec.base = RLMESH_ROTATION_ENCODING_OTHER;
  CHECK(rlmesh_custom_encoding_width(&encoding_spec, &dims) == RLMESH_ERR_INVALID_ARGUMENT);
  actuator.has_encoding = true;
  actuator.encoding = RLMESH_ROTATION_ENCODING_OTHER;
  CHECK(rlmesh_model_spec_new(&input, &action, &image_spec) == RLMESH_ERR_INVALID_ARGUMENT &&
        !image_spec);
  actuator.has_encoding = false;
  RLMeshAdvisory advisory = RLMESH_ADVISORY_INIT;
  advisory.severity = RLMESH_ADVISORY_SEVERITY_CAUTION;
  advisory.message = "note";
  CHECK(rlmesh_advisory_equals(&advisory, &advisory, &equal) == RLMESH_OK && equal);
  CHECK(rlmesh_advisory_repr(&advisory, &json) == RLMESH_OK);
  rlmesh_bytes_free(json);
  advisory.message = NULL;
  CHECK(rlmesh_advisory_equals(&advisory, &advisory, &equal) == RLMESH_ERR_TYPE);
  /* Tolerant spelling and binary-safe dict keys survive typed and JSON round trips. */
  state.dtype = RLMESH_DTYPE_OTHER;
  state.dtype_other = "future-float";
  const char key_bytes[] = {'z', 0, 'x'};
  RLMeshStr keys[] = {{key_bytes, 3}, {"a", 1}};
  RLMeshInputNode children[] = {input, input};
  RLMeshInputNode dict = rlmesh_input_node_dict(keys, children, 2);
  RLMeshModelSpec* future = NULL;
  CHECK(rlmesh_model_spec_new(&dict, &action, &future) == RLMESH_OK);
  const RLMeshInputNode* tree = rlmesh_model_spec_input(future);
  CHECK(tree->keys[0].len == 3 && memcmp(tree->keys[0].data, key_bytes, 3) == 0);
  lent = (const RLMeshState*)tree->children[0].leaf;
  CHECK(lent->dtype == RLMESH_DTYPE_OTHER && strcmp(lent->dtype_other, "future-float") == 0);
  CHECK(rlmesh_model_spec_to_json(future, &json) == RLMESH_OK);
  rlmesh_bytes_free(json);
  rlmesh_model_spec_free(future);
  keys[1] = keys[0];
  CHECK(rlmesh_model_spec_new(&dict, &action, &future) == RLMESH_ERR_INVALID_ARGUMENT && !future);
  state.dtype = RLMESH_DTYPE_FLOAT32;
  state.dtype_other = NULL;
  RLMeshStateTag tag = RLMESH_STATE_TAG_INIT;
  tag.role = RLMESH_ADAPTERS_JOINT_POS;
  RLMeshObsNode obs = rlmesh_obs_node_state_tag(&tag);
  RLMeshEnvTags* tags = NULL;
  CHECK(rlmesh_env_tags_new(&obs, &action, &tags) == RLMESH_OK);
  RLMeshObservationRoles roles = RLMESH_OBSERVATION_ROLES_INIT;
  CHECK(rlmesh_env_tags_observation_roles(tags, &roles) == RLMESH_OK && roles.states_len == 1 &&
        strcmp(roles.states[0], state.role) == 0);

  CHECK(rlmesh_observation_roles_equals(&roles, &roles, &equal) == RLMESH_OK && equal);
  CHECK(rlmesh_observation_roles_repr(&roles, &json) == RLMESH_OK);
  rlmesh_bytes_free(json);
  RLMeshBoxOptions box_options = RLMESH_BOX_OPTIONS_INIT;
  box_options.dtype = RLMESH_DTYPE_FLOAT64;
  const int64_t shape[] = {1};
  RLMeshBox* space = NULL;
  CHECK(rlmesh_box_new(rlmesh_bound_float(-10.), rlmesh_bound_float(10.), shape, 1, &box_options,
                       &space) == RLMESH_OK);
  RLMeshAdapter* adapter = NULL;
  CHECK(rlmesh_adapters_resolve(tags, rlmesh_space_spec(space), rlmesh_space_spec(space), spec,
                                NULL, &adapter) == RLMESH_OK);
  RLMeshValue* raw = vector(2.5);
  RLMeshValue* converted = NULL;
  CHECK(rlmesh_adapter_transform_obs(adapter, raw, &converted) == RLMESH_OK &&
        rlmesh_value_kind(converted) == RLMESH_VALUE_TENSOR);
  rlmesh_value_free(converted);
  CHECK(rlmesh_adapter_observe(adapter, raw) == RLMESH_OK &&
        rlmesh_adapter_reset(adapter) == RLMESH_OK);
  CHECK(rlmesh_adapter_transform_action(adapter, raw, &converted) == RLMESH_OK);
  rlmesh_value_free(converted);
  size_t count = 99;
  CHECK(rlmesh_adapter_history_keys_len(adapter, &count) == RLMESH_OK && count == 0);
  CHECK(rlmesh_adapter_history_windows_len(adapter, &count) == RLMESH_OK && count == 0);
  CHECK(rlmesh_adapter_explain(adapter, &json) == RLMESH_OK);
  rlmesh_bytes_free(json);
  int calls = 0;
  RLMeshWrappedPredict* wrapped = NULL;
  CHECK(rlmesh_adapter_wrap_predict(adapter, identity, &calls, counted_free, &wrapped) ==
        RLMESH_OK);
  rlmesh_adapter_free(adapter);
  CHECK(rlmesh_wrapped_predict_call(wrapped, raw, &converted) == RLMESH_OK);
  rlmesh_value_free(converted);
  CHECK(rlmesh_wrapped_predict_call(wrapped, raw, &converted) == RLMESH_OK);
  rlmesh_value_free(converted);
  CHECK(calls == 2);
  rlmesh_wrapped_predict_free(wrapped);
  CHECK(calls == 3);
  /* Invalid R-19 input, zero/nonzero future tails, and no ownership on refusal. */
  RLMeshState bad = {0};
  RLMeshInputNode bad_node = rlmesh_input_node_state(&bad);
  CHECK(rlmesh_model_spec_new(&bad_node, &action, &future) == RLMESH_ERR_INVALID_ARGUMENT &&
        !future);
  struct FutureState {
    RLMeshState current;
    uint64_t tail;
  } extended = {RLMESH_STATE_INIT, 0};
  extended.current.role = state.role;
  extended.current.struct_size = sizeof(extended);
  bad_node = rlmesh_input_node_state(&extended.current);
  CHECK(rlmesh_model_spec_new(&bad_node, &action, &future) == RLMESH_OK);
  rlmesh_model_spec_free(future);
  extended.tail = 1;
  CHECK(rlmesh_model_spec_new(&bad_node, &action, &future) == RLMESH_ERR_TYPE && !future);
  RLMeshAdapterVtable vt = RLMESH_ADAPTER_VTABLE_INIT;
  int frees = 0;
  vt.free_user_data = counted_free;
  CHECK(rlmesh_adapter_new(&vt, &frees, &adapter) == RLMESH_ERR_TYPE && frees == 0);
  vt.transform_obs = identity;
  vt.transform_action = identity;
  CHECK(rlmesh_adapter_new(&vt, &frees, &adapter) == RLMESH_OK);
  CHECK(rlmesh_adapter_advisories_len(adapter, &count) == RLMESH_ERR_TYPE);
  CHECK(rlmesh_adapter_explain(adapter, &json) == RLMESH_OK && json.len == 28);
  rlmesh_bytes_free(json);
  RLMeshAdapter* alias = rlmesh_adapter_clone(adapter);
  rlmesh_adapter_free(adapter);
  CHECK(frees == 0);
  CHECK(rlmesh_adapter_transform_obs(alias, raw, &converted) == RLMESH_OK);
  rlmesh_value_free(converted);
  rlmesh_adapter_free(alias);
  CHECK(frees == 2);
  vt.transform_obs = fail_transform;
  vt.free_user_data = NULL;
  CHECK(rlmesh_adapter_new(&vt, NULL, &adapter) == RLMESH_OK);
  CHECK(rlmesh_adapter_transform_obs(adapter, raw, &converted) == RLMESH_ERR_OS && !converted &&
        rlmesh_last_error_class() == RLMESH_ERROR_CLASS_OS_ERROR);
  rlmesh_adapter_free(adapter);
  /* Transform handles share identity; specs retain them after the caller frees. */
  int transform_frees = 0;
  RLMeshObsTransform* transform = NULL;
  CHECK(rlmesh_obs_transform_new(NULL, &transform_frees, counted_free, &transform) ==
            RLMESH_ERR_TYPE &&
        transform_frees == 0);
  CHECK(rlmesh_obs_transform_new(identity, &transform_frees, counted_free, &transform) ==
        RLMESH_OK);
  RLMeshCustom custom = RLMESH_CUSTOM_INIT;
  custom.transform = transform;
  RLMeshInputNode custom_input = rlmesh_input_node_custom(&custom);
  RLMeshModelSpec* custom_spec = NULL;
  CHECK(rlmesh_model_spec_new(&custom_input, &action, &custom_spec) == RLMESH_OK);
  rlmesh_obs_transform_free(transform);
  CHECK(transform_frees == 0);
  CHECK(rlmesh_model_spec_to_dict(custom_spec, &converted) == RLMESH_ERR_INVALID_ARGUMENT &&
        !converted);
  CHECK(rlmesh_adapters_resolve(tags, rlmesh_space_spec(space), rlmesh_space_spec(space),
                                custom_spec, NULL, &alias) == RLMESH_OK);
  CHECK(rlmesh_adapter_transform_obs(alias, raw, &converted) == RLMESH_OK);
  CHECK(rlmesh_value_kind(converted) == RLMESH_VALUE_DICT);
  rlmesh_value_free(converted);
  rlmesh_model_spec_free(custom_spec);
  CHECK(transform_frees == 1);
  rlmesh_adapter_free(alias);
  CHECK(transform_frees == 2);
  /* Real model attachment and tag mutate the original env; NO_ADAPTER bypasses. */
  RLMeshEnvVtable ev = RLMESH_ENV_VTABLE_INIT;
  ev.reset = reset;
  ev.step = step;
  RLMeshEnvAttributes attributes = RLMESH_ENV_ATTRIBUTES_INIT;
  attributes.observation_space = space;
  attributes.action_space = space;
  RLMeshEnv* env = NULL;
  CHECK(rlmesh_env_new(&ev, NULL, &attributes, &env) == RLMESH_OK);
  CHECK(rlmesh_adapters_tag(env, tags, NULL) == RLMESH_OK);
  RLMeshModelVtable policy = RLMESH_MODEL_VTABLE_INIT;
  policy.predict = predict;
  policy.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshModelOptions mo = RLMESH_MODEL_OPTIONS_INIT;
  mo.spec = (RLMeshSpecArg){RLMESH_SPEC_KIND_MODEL_SPEC, spec};
  RLMeshModel* model = NULL;
  CHECK(rlmesh_model_new(&policy, NULL, &mo, &model) == RLMESH_OK);
  RLMeshSpecArg attached = {RLMESH_SPEC_KIND_INVALID, NULL};
  CHECK(rlmesh_model_spec(model, &attached) == RLMESH_OK &&
        attached.kind == RLMESH_SPEC_KIND_MODEL_SPEC &&
        rlmesh_model_spec_equals(spec, attached.spec));
  RLMeshRunResult* result = NULL;
  RLMeshRunOptions run = RLMESH_RUN_OPTIONS_INIT;
  run.workflow_edition = rlmesh_current_workflow_edition();
  run.close_env = false;
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &run, &result) == RLMESH_OK);
  rlmesh_run_result_free(result);
  rlmesh_model_free(model);
  mo.spec = (RLMeshSpecArg){RLMESH_SPEC_KIND_NONE, NULL};
  CHECK(rlmesh_model_new(&policy, NULL, &mo, &model) == RLMESH_OK);
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &run, &result) ==
            RLMESH_ERR_INVALID_ARGUMENT &&
        rlmesh_last_error_class() == RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR);
  rlmesh_model_free(model);
  mo.spec = (RLMeshSpecArg){RLMESH_SPEC_KIND_NO_ADAPTER, NULL};
  CHECK(rlmesh_model_new(&policy, NULL, &mo, &model) == RLMESH_OK);
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &run, &result) == RLMESH_OK);
  rlmesh_run_result_free(result);
  rlmesh_model_free(model);
  rlmesh_env_free(env);
  rlmesh_space_free(space);
  rlmesh_env_tags_free(tags);
  rlmesh_model_spec_free(spec);
  rlmesh_value_free(raw);
  if (failures) return 1;
  puts("c_adapters OK");
  return 0;
}

static void layouts(void) {
#if UINTPTR_MAX > UINT32_MAX
  CHECK(sizeof(RLMeshRotation) ==
        sizeof(((RLMeshRotation*)0)->struct_size) + sizeof(((RLMeshRotation*)0)->encoding) +
            sizeof(((RLMeshRotation*)0)->_reserved) + sizeof(((RLMeshRotation*)0)->value) +
            sizeof(((RLMeshRotation*)0)->value_len));
  CHECK(sizeof(RLMeshCustomEncoding) ==
        sizeof(((RLMeshCustomEncoding*)0)->struct_size) + sizeof(((RLMeshCustomEncoding*)0)->base) +
            sizeof(((RLMeshCustomEncoding*)0)->_reserved0) +
            sizeof(((RLMeshCustomEncoding*)0)->from_base) +
            sizeof(((RLMeshCustomEncoding*)0)->to_base) + sizeof(((RLMeshCustomEncoding*)0)->name));
  CHECK(sizeof(RLMeshImage) ==
        sizeof(((RLMeshImage*)0)->struct_size) + sizeof(((RLMeshImage*)0)->role) +
            sizeof(((RLMeshImage*)0)->has_height) + sizeof(((RLMeshImage*)0)->_reserved0) +
            sizeof(((RLMeshImage*)0)->height) + sizeof(((RLMeshImage*)0)->has_width) +
            sizeof(((RLMeshImage*)0)->_reserved1) + sizeof(((RLMeshImage*)0)->width) +
            sizeof(((RLMeshImage*)0)->layout) + sizeof(((RLMeshImage*)0)->has_channels) +
            sizeof(((RLMeshImage*)0)->_reserved2) + sizeof(((RLMeshImage*)0)->channels) +
            sizeof(((RLMeshImage*)0)->dtype) + sizeof(((RLMeshImage*)0)->dtype_other) +
            sizeof(((RLMeshImage*)0)->normalize) + sizeof(((RLMeshImage*)0)->_reserved3) +
            sizeof(((RLMeshImage*)0)->normalize_range) + sizeof(((RLMeshImage*)0)->lead_dims) +
            sizeof(((RLMeshImage*)0)->upside_down) + sizeof(((RLMeshImage*)0)->_reserved4) +
            sizeof(((RLMeshImage*)0)->resample) + sizeof(((RLMeshImage*)0)->_reserved5) +
            sizeof(((RLMeshImage*)0)->resample_other) + sizeof(((RLMeshImage*)0)->allow_upscale) +
            sizeof(((RLMeshImage*)0)->_reserved6) + sizeof(((RLMeshImage*)0)->fit) +
            sizeof(((RLMeshImage*)0)->fit_len) + sizeof(((RLMeshImage*)0)->fit_other) +
            sizeof(((RLMeshImage*)0)->optional) + sizeof(((RLMeshImage*)0)->has_fill) +
            sizeof(((RLMeshImage*)0)->fill) + sizeof(((RLMeshImage*)0)->_reserved7) +
            sizeof(((RLMeshImage*)0)->stack) + sizeof(((RLMeshImage*)0)->has_crop) +
            sizeof(((RLMeshImage*)0)->_reserved8) + sizeof(((RLMeshImage*)0)->crop) +
            sizeof(((RLMeshImage*)0)->has_crop_area) + sizeof(((RLMeshImage*)0)->_reserved9) +
            sizeof(((RLMeshImage*)0)->crop_area) + sizeof(((RLMeshImage*)0)->crop_mode) +
            sizeof(((RLMeshImage*)0)->_reserved10) + sizeof(((RLMeshImage*)0)->crop_mode_other) +
            sizeof(((RLMeshImage*)0)->has_jpeg_quality) + sizeof(((RLMeshImage*)0)->jpeg_quality) +
            sizeof(((RLMeshImage*)0)->_reserved11) + sizeof(((RLMeshImage*)0)->channel_order) +
            sizeof(((RLMeshImage*)0)->channel_order_other) + sizeof(((RLMeshImage*)0)->has_render) +
            sizeof(((RLMeshImage*)0)->_reserved12) + sizeof(((RLMeshImage*)0)->render) +
            sizeof(((RLMeshImage*)0)->_reserved13) + sizeof(((RLMeshImage*)0)->offsets) +
            sizeof(((RLMeshImage*)0)->offsets_len) + sizeof(((RLMeshImage*)0)->stack_pad) +
            sizeof(((RLMeshImage*)0)->_reserved14) + sizeof(((RLMeshImage*)0)->part) +
            sizeof(((RLMeshImage*)0)->has_stride) + sizeof(((RLMeshImage*)0)->_reserved15) +
            sizeof(((RLMeshImage*)0)->stride));
  CHECK(sizeof(RLMeshState) ==
        sizeof(((RLMeshState*)0)->struct_size) + sizeof(((RLMeshState*)0)->role) +
            sizeof(((RLMeshState*)0)->encoding) + sizeof(((RLMeshState*)0)->encoding_len) +
            sizeof(((RLMeshState*)0)->encoding_other) + sizeof(((RLMeshState*)0)->custom_encoding) +
            sizeof(((RLMeshState*)0)->has_dim) + sizeof(((RLMeshState*)0)->_reserved0) +
            sizeof(((RLMeshState*)0)->dim) + sizeof(((RLMeshState*)0)->has_index) +
            sizeof(((RLMeshState*)0)->_reserved1) + sizeof(((RLMeshState*)0)->index) +
            sizeof(((RLMeshState*)0)->optional) + sizeof(((RLMeshState*)0)->has_range) +
            sizeof(((RLMeshState*)0)->_reserved2) + sizeof(((RLMeshState*)0)->range) +
            sizeof(((RLMeshState*)0)->has_pad_to) + sizeof(((RLMeshState*)0)->_reserved3) +
            sizeof(((RLMeshState*)0)->pad_to) + sizeof(((RLMeshState*)0)->dtype) +
            sizeof(((RLMeshState*)0)->_reserved4) + sizeof(((RLMeshState*)0)->dtype_other) +
            sizeof(((RLMeshState*)0)->reshape) + sizeof(((RLMeshState*)0)->reshape_len) +
            sizeof(((RLMeshState*)0)->container) + sizeof(((RLMeshState*)0)->_reserved5) +
            sizeof(((RLMeshState*)0)->fill) + sizeof(((RLMeshState*)0)->post_rotate) +
            sizeof(((RLMeshState*)0)->has_scale) + sizeof(((RLMeshState*)0)->_reserved6) +
            sizeof(((RLMeshState*)0)->scale) + sizeof(((RLMeshState*)0)->axis_scale) +
            sizeof(((RLMeshState*)0)->axis_scale_len) + sizeof(((RLMeshState*)0)->has_offset) +
            sizeof(((RLMeshState*)0)->_reserved7) + sizeof(((RLMeshState*)0)->offset) +
            sizeof(((RLMeshState*)0)->axis_offset) + sizeof(((RLMeshState*)0)->axis_offset_len) +
            sizeof(((RLMeshState*)0)->has_frame) + sizeof(((RLMeshState*)0)->_reserved8) +
            sizeof(((RLMeshState*)0)->frame) + sizeof(((RLMeshState*)0)->frame_other) +
            sizeof(((RLMeshState*)0)->provenance) + sizeof(((RLMeshState*)0)->provenance_len) +
            sizeof(((RLMeshState*)0)->provenance_other) + sizeof(((RLMeshState*)0)->part) +
            sizeof(((RLMeshState*)0)->labels) + sizeof(((RLMeshState*)0)->labels_len) +
            sizeof(((RLMeshState*)0)->source) + sizeof(((RLMeshState*)0)->has_clip) +
            sizeof(((RLMeshState*)0)->_reserved9) + sizeof(((RLMeshState*)0)->clip));
  CHECK(sizeof(RLMeshConcat) ==
        sizeof(((RLMeshConcat*)0)->struct_size) + sizeof(((RLMeshConcat*)0)->parts) +
            sizeof(((RLMeshConcat*)0)->parts_len) + sizeof(((RLMeshConcat*)0)->has_pad_to) +
            sizeof(((RLMeshConcat*)0)->_reserved0) + sizeof(((RLMeshConcat*)0)->pad_to) +
            sizeof(((RLMeshConcat*)0)->dtype) + sizeof(((RLMeshConcat*)0)->_reserved1) +
            sizeof(((RLMeshConcat*)0)->dtype_other) + sizeof(((RLMeshConcat*)0)->reshape) +
            sizeof(((RLMeshConcat*)0)->reshape_len) + sizeof(((RLMeshConcat*)0)->container) +
            sizeof(((RLMeshConcat*)0)->has_clip) + sizeof(((RLMeshConcat*)0)->_reserved2) +
            sizeof(((RLMeshConcat*)0)->clip));
  CHECK(sizeof(RLMeshConstant) ==
        sizeof(((RLMeshConstant*)0)->struct_size) + sizeof(((RLMeshConstant*)0)->dim) +
            sizeof(((RLMeshConstant*)0)->_reserved0) + sizeof(((RLMeshConstant*)0)->fill));
  CHECK(sizeof(RLMeshAdaptersText) ==
        sizeof(((RLMeshAdaptersText*)0)->struct_size) + sizeof(((RLMeshAdaptersText*)0)->role) +
            sizeof(((RLMeshAdaptersText*)0)->container) +
            sizeof(((RLMeshAdaptersText*)0)->has_fill) +
            sizeof(((RLMeshAdaptersText*)0)->_reserved0) + sizeof(((RLMeshAdaptersText*)0)->fill));
  CHECK(sizeof(RLMeshCustom) == sizeof(((RLMeshCustom*)0)->struct_size) +
                                    sizeof(((RLMeshCustom*)0)->transform) +
                                    sizeof(((RLMeshCustom*)0)->entrypoint));
  CHECK(
      sizeof(RLMeshActuator) ==
      sizeof(((RLMeshActuator*)0)->struct_size) + sizeof(((RLMeshActuator*)0)->role) +
          sizeof(((RLMeshActuator*)0)->dim) + sizeof(((RLMeshActuator*)0)->has_encoding) +
          sizeof(((RLMeshActuator*)0)->_reserved0) + sizeof(((RLMeshActuator*)0)->encoding) +
          sizeof(((RLMeshActuator*)0)->_reserved1) + sizeof(((RLMeshActuator*)0)->custom_encoding) +
          sizeof(((RLMeshActuator*)0)->has_range) + sizeof(((RLMeshActuator*)0)->_reserved2) +
          sizeof(((RLMeshActuator*)0)->range) + sizeof(((RLMeshActuator*)0)->binary) +
          sizeof(((RLMeshActuator*)0)->has_scale) + sizeof(((RLMeshActuator*)0)->_reserved3) +
          sizeof(((RLMeshActuator*)0)->scale) + sizeof(((RLMeshActuator*)0)->axis_scale) +
          sizeof(((RLMeshActuator*)0)->axis_scale_len) + sizeof(((RLMeshActuator*)0)->invert) +
          sizeof(((RLMeshActuator*)0)->has_threshold) + sizeof(((RLMeshActuator*)0)->_reserved4) +
          sizeof(((RLMeshActuator*)0)->threshold) + sizeof(((RLMeshActuator*)0)->clip) +
          sizeof(((RLMeshActuator*)0)->_reserved5) + sizeof(((RLMeshActuator*)0)->fill) +
          sizeof(((RLMeshActuator*)0)->optional) + sizeof(((RLMeshActuator*)0)->has_frame) +
          sizeof(((RLMeshActuator*)0)->_reserved6) + sizeof(((RLMeshActuator*)0)->frame) +
          sizeof(((RLMeshActuator*)0)->frame_other) + sizeof(((RLMeshActuator*)0)->has_reference) +
          sizeof(((RLMeshActuator*)0)->_reserved7) + sizeof(((RLMeshActuator*)0)->reference) +
          sizeof(((RLMeshActuator*)0)->reference_other) + sizeof(((RLMeshActuator*)0)->part) +
          sizeof(((RLMeshActuator*)0)->has_offset) + sizeof(((RLMeshActuator*)0)->_reserved8) +
          sizeof(((RLMeshActuator*)0)->offset) + sizeof(((RLMeshActuator*)0)->axis_offset) +
          sizeof(((RLMeshActuator*)0)->axis_offset_len) + sizeof(((RLMeshActuator*)0)->labels) +
          sizeof(((RLMeshActuator*)0)->labels_len));
  CHECK(sizeof(RLMeshAction) ==
        sizeof(((RLMeshAction*)0)->struct_size) + sizeof(((RLMeshAction*)0)->components) +
            sizeof(((RLMeshAction*)0)->components_len) + sizeof(((RLMeshAction*)0)->has_clip) +
            sizeof(((RLMeshAction*)0)->_reserved0) + sizeof(((RLMeshAction*)0)->clip));
  CHECK(sizeof(RLMeshImageTag) ==
        sizeof(((RLMeshImageTag*)0)->struct_size) + sizeof(((RLMeshImageTag*)0)->role) +
            sizeof(((RLMeshImageTag*)0)->layout) + sizeof(((RLMeshImageTag*)0)->upside_down) +
            sizeof(((RLMeshImageTag*)0)->_reserved0) + sizeof(((RLMeshImageTag*)0)->part));
  CHECK(sizeof(RLMeshStateTag) ==
        sizeof(((RLMeshStateTag*)0)->struct_size) + sizeof(((RLMeshStateTag*)0)->role) +
            sizeof(((RLMeshStateTag*)0)->encoding) + sizeof(((RLMeshStateTag*)0)->encoding_len) +
            sizeof(((RLMeshStateTag*)0)->encoding_other) + sizeof(((RLMeshStateTag*)0)->has_range) +
            sizeof(((RLMeshStateTag*)0)->_reserved0) + sizeof(((RLMeshStateTag*)0)->range) +
            sizeof(((RLMeshStateTag*)0)->has_frame) + sizeof(((RLMeshStateTag*)0)->_reserved1) +
            sizeof(((RLMeshStateTag*)0)->frame) + sizeof(((RLMeshStateTag*)0)->frame_other) +
            sizeof(((RLMeshStateTag*)0)->has_provenance) +
            sizeof(((RLMeshStateTag*)0)->_reserved2) + sizeof(((RLMeshStateTag*)0)->provenance) +
            sizeof(((RLMeshStateTag*)0)->provenance_other) + sizeof(((RLMeshStateTag*)0)->part) +
            sizeof(((RLMeshStateTag*)0)->labels) + sizeof(((RLMeshStateTag*)0)->labels_len));
  CHECK(sizeof(RLMeshTextTag) ==
        sizeof(((RLMeshTextTag*)0)->struct_size) + sizeof(((RLMeshTextTag*)0)->role));
  CHECK(sizeof(RLMeshField) ==
        sizeof(((RLMeshField*)0)->struct_size) + sizeof(((RLMeshField*)0)->role) +
            sizeof(((RLMeshField*)0)->dim) + sizeof(((RLMeshField*)0)->_reserved0) +
            sizeof(((RLMeshField*)0)->encoding) + sizeof(((RLMeshField*)0)->encoding_len) +
            sizeof(((RLMeshField*)0)->encoding_other) + sizeof(((RLMeshField*)0)->has_range) +
            sizeof(((RLMeshField*)0)->_reserved1) + sizeof(((RLMeshField*)0)->range) +
            sizeof(((RLMeshField*)0)->has_frame) + sizeof(((RLMeshField*)0)->_reserved2) +
            sizeof(((RLMeshField*)0)->frame) + sizeof(((RLMeshField*)0)->frame_other) +
            sizeof(((RLMeshField*)0)->has_provenance) + sizeof(((RLMeshField*)0)->_reserved3) +
            sizeof(((RLMeshField*)0)->provenance) + sizeof(((RLMeshField*)0)->provenance_other) +
            sizeof(((RLMeshField*)0)->part) + sizeof(((RLMeshField*)0)->labels) +
            sizeof(((RLMeshField*)0)->labels_len));
  CHECK(sizeof(RLMeshSplit) == sizeof(((RLMeshSplit*)0)->struct_size) +
                                   sizeof(((RLMeshSplit*)0)->fields) +
                                   sizeof(((RLMeshSplit*)0)->fields_len));
  CHECK(sizeof(RLMeshPreviousOptions) == sizeof(((RLMeshPreviousOptions*)0)->struct_size) +
                                             sizeof(((RLMeshPreviousOptions*)0)->part) +
                                             sizeof(((RLMeshPreviousOptions*)0)->fill));
  CHECK(sizeof(RLMeshObservationRoles) == sizeof(((RLMeshObservationRoles*)0)->struct_size) +
                                              sizeof(((RLMeshObservationRoles*)0)->images) +
                                              sizeof(((RLMeshObservationRoles*)0)->images_len) +
                                              sizeof(((RLMeshObservationRoles*)0)->states) +
                                              sizeof(((RLMeshObservationRoles*)0)->states_len) +
                                              sizeof(((RLMeshObservationRoles*)0)->texts) +
                                              sizeof(((RLMeshObservationRoles*)0)->texts_len));
  CHECK(sizeof(RLMeshAdvisory) ==
        sizeof(((RLMeshAdvisory*)0)->struct_size) + sizeof(((RLMeshAdvisory*)0)->severity) +
            sizeof(((RLMeshAdvisory*)0)->_reserved) + sizeof(((RLMeshAdvisory*)0)->message));
  CHECK(sizeof(RLMeshTagOptions) == sizeof(((RLMeshTagOptions*)0)->struct_size) +
                                        sizeof(((RLMeshTagOptions*)0)->validate) +
                                        sizeof(((RLMeshTagOptions*)0)->_reserved0) +
                                        sizeof(((RLMeshTagOptions*)0)->_reserved1) +
                                        sizeof(((RLMeshTagOptions*)0)->_reserved2));
  CHECK(sizeof(RLMeshResolveOptions) == sizeof(((RLMeshResolveOptions*)0)->struct_size) +
                                            sizeof(((RLMeshResolveOptions*)0)->trust_entrypoints) +
                                            sizeof(((RLMeshResolveOptions*)0)->check_inverse) +
                                            sizeof(((RLMeshResolveOptions*)0)->_reserved0) +
                                            sizeof(((RLMeshResolveOptions*)0)->_reserved1));
#else
  CHECK(sizeof(RLMeshRotation) ==
        sizeof(((RLMeshRotation*)0)->struct_size) + sizeof(((RLMeshRotation*)0)->encoding) +
            sizeof(((RLMeshRotation*)0)->_reserved) + sizeof(((RLMeshRotation*)0)->value) +
            sizeof(((RLMeshRotation*)0)->value_len));
  CHECK(sizeof(RLMeshCustomEncoding) ==
        sizeof(((RLMeshCustomEncoding*)0)->struct_size) + sizeof(((RLMeshCustomEncoding*)0)->base) +
            sizeof(((RLMeshCustomEncoding*)0)->from_base) +
            sizeof(((RLMeshCustomEncoding*)0)->to_base) + sizeof(((RLMeshCustomEncoding*)0)->name));
  CHECK(sizeof(RLMeshImage) ==
        sizeof(((RLMeshImage*)0)->struct_size) + sizeof(((RLMeshImage*)0)->role) +
            sizeof(((RLMeshImage*)0)->has_height) + sizeof(((RLMeshImage*)0)->_reserved0) +
            sizeof(((RLMeshImage*)0)->height) + sizeof(((RLMeshImage*)0)->has_width) +
            sizeof(((RLMeshImage*)0)->_reserved1) + sizeof(((RLMeshImage*)0)->width) +
            sizeof(((RLMeshImage*)0)->layout) + sizeof(((RLMeshImage*)0)->has_channels) +
            sizeof(((RLMeshImage*)0)->_reserved2) + sizeof(((RLMeshImage*)0)->channels) +
            sizeof(((RLMeshImage*)0)->dtype) + sizeof(((RLMeshImage*)0)->dtype_other) +
            sizeof(((RLMeshImage*)0)->normalize) + sizeof(((RLMeshImage*)0)->normalize_range) +
            sizeof(((RLMeshImage*)0)->lead_dims) + sizeof(((RLMeshImage*)0)->upside_down) +
            sizeof(((RLMeshImage*)0)->_reserved3) + sizeof(((RLMeshImage*)0)->resample) +
            sizeof(((RLMeshImage*)0)->resample_other) + sizeof(((RLMeshImage*)0)->allow_upscale) +
            sizeof(((RLMeshImage*)0)->_reserved4) + sizeof(((RLMeshImage*)0)->fit) +
            sizeof(((RLMeshImage*)0)->fit_len) + sizeof(((RLMeshImage*)0)->fit_other) +
            sizeof(((RLMeshImage*)0)->optional) + sizeof(((RLMeshImage*)0)->has_fill) +
            sizeof(((RLMeshImage*)0)->fill) + sizeof(((RLMeshImage*)0)->_reserved5) +
            sizeof(((RLMeshImage*)0)->stack) + sizeof(((RLMeshImage*)0)->has_crop) +
            sizeof(((RLMeshImage*)0)->_reserved6) + sizeof(((RLMeshImage*)0)->crop) +
            sizeof(((RLMeshImage*)0)->has_crop_area) + sizeof(((RLMeshImage*)0)->_reserved7) +
            sizeof(((RLMeshImage*)0)->crop_area) + sizeof(((RLMeshImage*)0)->crop_mode) +
            sizeof(((RLMeshImage*)0)->crop_mode_other) +
            sizeof(((RLMeshImage*)0)->has_jpeg_quality) + sizeof(((RLMeshImage*)0)->jpeg_quality) +
            sizeof(((RLMeshImage*)0)->_reserved8) + sizeof(((RLMeshImage*)0)->channel_order) +
            sizeof(((RLMeshImage*)0)->channel_order_other) + sizeof(((RLMeshImage*)0)->has_render) +
            sizeof(((RLMeshImage*)0)->_reserved9) + sizeof(((RLMeshImage*)0)->render) +
            sizeof(((RLMeshImage*)0)->offsets) + sizeof(((RLMeshImage*)0)->offsets_len) +
            sizeof(((RLMeshImage*)0)->stack_pad) + sizeof(((RLMeshImage*)0)->part) +
            sizeof(((RLMeshImage*)0)->has_stride) + sizeof(((RLMeshImage*)0)->_reserved10) +
            sizeof(((RLMeshImage*)0)->stride));
  CHECK(sizeof(RLMeshState) ==
        sizeof(((RLMeshState*)0)->struct_size) + sizeof(((RLMeshState*)0)->role) +
            sizeof(((RLMeshState*)0)->encoding) + sizeof(((RLMeshState*)0)->encoding_len) +
            sizeof(((RLMeshState*)0)->encoding_other) + sizeof(((RLMeshState*)0)->custom_encoding) +
            sizeof(((RLMeshState*)0)->has_dim) + sizeof(((RLMeshState*)0)->_reserved0) +
            sizeof(((RLMeshState*)0)->dim) + sizeof(((RLMeshState*)0)->has_index) +
            sizeof(((RLMeshState*)0)->_reserved1) + sizeof(((RLMeshState*)0)->index) +
            sizeof(((RLMeshState*)0)->optional) + sizeof(((RLMeshState*)0)->has_range) +
            sizeof(((RLMeshState*)0)->_reserved2) + sizeof(((RLMeshState*)0)->range) +
            sizeof(((RLMeshState*)0)->has_pad_to) + sizeof(((RLMeshState*)0)->_reserved3) +
            sizeof(((RLMeshState*)0)->pad_to) + sizeof(((RLMeshState*)0)->dtype) +
            sizeof(((RLMeshState*)0)->dtype_other) + sizeof(((RLMeshState*)0)->reshape) +
            sizeof(((RLMeshState*)0)->reshape_len) + sizeof(((RLMeshState*)0)->container) +
            sizeof(((RLMeshState*)0)->fill) + sizeof(((RLMeshState*)0)->post_rotate) +
            sizeof(((RLMeshState*)0)->has_scale) + sizeof(((RLMeshState*)0)->_reserved4) +
            sizeof(((RLMeshState*)0)->scale) + sizeof(((RLMeshState*)0)->axis_scale) +
            sizeof(((RLMeshState*)0)->axis_scale_len) + sizeof(((RLMeshState*)0)->has_offset) +
            sizeof(((RLMeshState*)0)->_reserved5) + sizeof(((RLMeshState*)0)->offset) +
            sizeof(((RLMeshState*)0)->axis_offset) + sizeof(((RLMeshState*)0)->axis_offset_len) +
            sizeof(((RLMeshState*)0)->has_frame) + sizeof(((RLMeshState*)0)->_reserved6) +
            sizeof(((RLMeshState*)0)->frame) + sizeof(((RLMeshState*)0)->frame_other) +
            sizeof(((RLMeshState*)0)->provenance) + sizeof(((RLMeshState*)0)->provenance_len) +
            sizeof(((RLMeshState*)0)->provenance_other) + sizeof(((RLMeshState*)0)->part) +
            sizeof(((RLMeshState*)0)->labels) + sizeof(((RLMeshState*)0)->labels_len) +
            sizeof(((RLMeshState*)0)->source) + sizeof(((RLMeshState*)0)->has_clip) +
            sizeof(((RLMeshState*)0)->_reserved7) + sizeof(((RLMeshState*)0)->clip));
  CHECK(sizeof(RLMeshConcat) ==
        sizeof(((RLMeshConcat*)0)->struct_size) + sizeof(((RLMeshConcat*)0)->parts) +
            sizeof(((RLMeshConcat*)0)->parts_len) + sizeof(((RLMeshConcat*)0)->has_pad_to) +
            sizeof(((RLMeshConcat*)0)->_reserved0) + sizeof(((RLMeshConcat*)0)->pad_to) +
            sizeof(((RLMeshConcat*)0)->dtype) + sizeof(((RLMeshConcat*)0)->dtype_other) +
            sizeof(((RLMeshConcat*)0)->reshape) + sizeof(((RLMeshConcat*)0)->reshape_len) +
            sizeof(((RLMeshConcat*)0)->container) + sizeof(((RLMeshConcat*)0)->has_clip) +
            sizeof(((RLMeshConcat*)0)->_reserved1) + sizeof(((RLMeshConcat*)0)->clip));
  CHECK(sizeof(RLMeshConstant) == sizeof(((RLMeshConstant*)0)->struct_size) +
                                      sizeof(((RLMeshConstant*)0)->dim) +
                                      sizeof(((RLMeshConstant*)0)->fill));
  CHECK(sizeof(RLMeshAdaptersText) ==
        sizeof(((RLMeshAdaptersText*)0)->struct_size) + sizeof(((RLMeshAdaptersText*)0)->role) +
            sizeof(((RLMeshAdaptersText*)0)->container) +
            sizeof(((RLMeshAdaptersText*)0)->has_fill) +
            sizeof(((RLMeshAdaptersText*)0)->_reserved0) + sizeof(((RLMeshAdaptersText*)0)->fill));
  CHECK(sizeof(RLMeshCustom) == sizeof(((RLMeshCustom*)0)->struct_size) +
                                    sizeof(((RLMeshCustom*)0)->transform) +
                                    sizeof(((RLMeshCustom*)0)->entrypoint));
  CHECK(sizeof(RLMeshActuator) ==
        sizeof(((RLMeshActuator*)0)->struct_size) + sizeof(((RLMeshActuator*)0)->role) +
            sizeof(((RLMeshActuator*)0)->dim) + sizeof(((RLMeshActuator*)0)->has_encoding) +
            sizeof(((RLMeshActuator*)0)->_reserved0) + sizeof(((RLMeshActuator*)0)->encoding) +
            sizeof(((RLMeshActuator*)0)->custom_encoding) +
            sizeof(((RLMeshActuator*)0)->has_range) + sizeof(((RLMeshActuator*)0)->_reserved1) +
            sizeof(((RLMeshActuator*)0)->range) + sizeof(((RLMeshActuator*)0)->binary) +
            sizeof(((RLMeshActuator*)0)->has_scale) + sizeof(((RLMeshActuator*)0)->_reserved2) +
            sizeof(((RLMeshActuator*)0)->scale) + sizeof(((RLMeshActuator*)0)->axis_scale) +
            sizeof(((RLMeshActuator*)0)->axis_scale_len) + sizeof(((RLMeshActuator*)0)->invert) +
            sizeof(((RLMeshActuator*)0)->has_threshold) + sizeof(((RLMeshActuator*)0)->_reserved3) +
            sizeof(((RLMeshActuator*)0)->threshold) + sizeof(((RLMeshActuator*)0)->clip) +
            sizeof(((RLMeshActuator*)0)->_reserved4) + sizeof(((RLMeshActuator*)0)->fill) +
            sizeof(((RLMeshActuator*)0)->optional) + sizeof(((RLMeshActuator*)0)->has_frame) +
            sizeof(((RLMeshActuator*)0)->_reserved5) + sizeof(((RLMeshActuator*)0)->frame) +
            sizeof(((RLMeshActuator*)0)->frame_other) +
            sizeof(((RLMeshActuator*)0)->has_reference) + sizeof(((RLMeshActuator*)0)->_reserved6) +
            sizeof(((RLMeshActuator*)0)->reference) +
            sizeof(((RLMeshActuator*)0)->reference_other) + sizeof(((RLMeshActuator*)0)->part) +
            sizeof(((RLMeshActuator*)0)->has_offset) + sizeof(((RLMeshActuator*)0)->_reserved7) +
            sizeof(((RLMeshActuator*)0)->offset) + sizeof(((RLMeshActuator*)0)->axis_offset) +
            sizeof(((RLMeshActuator*)0)->axis_offset_len) + sizeof(((RLMeshActuator*)0)->labels) +
            sizeof(((RLMeshActuator*)0)->labels_len));
  CHECK(sizeof(RLMeshAction) ==
        sizeof(((RLMeshAction*)0)->struct_size) + sizeof(((RLMeshAction*)0)->components) +
            sizeof(((RLMeshAction*)0)->components_len) + sizeof(((RLMeshAction*)0)->has_clip) +
            sizeof(((RLMeshAction*)0)->_reserved0) + sizeof(((RLMeshAction*)0)->clip));
  CHECK(sizeof(RLMeshImageTag) ==
        sizeof(((RLMeshImageTag*)0)->struct_size) + sizeof(((RLMeshImageTag*)0)->role) +
            sizeof(((RLMeshImageTag*)0)->layout) + sizeof(((RLMeshImageTag*)0)->upside_down) +
            sizeof(((RLMeshImageTag*)0)->_reserved0) + sizeof(((RLMeshImageTag*)0)->part));
  CHECK(sizeof(RLMeshStateTag) ==
        sizeof(((RLMeshStateTag*)0)->struct_size) + sizeof(((RLMeshStateTag*)0)->role) +
            sizeof(((RLMeshStateTag*)0)->encoding) + sizeof(((RLMeshStateTag*)0)->encoding_len) +
            sizeof(((RLMeshStateTag*)0)->encoding_other) + sizeof(((RLMeshStateTag*)0)->has_range) +
            sizeof(((RLMeshStateTag*)0)->_reserved0) + sizeof(((RLMeshStateTag*)0)->range) +
            sizeof(((RLMeshStateTag*)0)->has_frame) + sizeof(((RLMeshStateTag*)0)->_reserved1) +
            sizeof(((RLMeshStateTag*)0)->frame) + sizeof(((RLMeshStateTag*)0)->frame_other) +
            sizeof(((RLMeshStateTag*)0)->has_provenance) +
            sizeof(((RLMeshStateTag*)0)->_reserved2) + sizeof(((RLMeshStateTag*)0)->provenance) +
            sizeof(((RLMeshStateTag*)0)->provenance_other) + sizeof(((RLMeshStateTag*)0)->part) +
            sizeof(((RLMeshStateTag*)0)->labels) + sizeof(((RLMeshStateTag*)0)->labels_len));
  CHECK(sizeof(RLMeshTextTag) ==
        sizeof(((RLMeshTextTag*)0)->struct_size) + sizeof(((RLMeshTextTag*)0)->role));
  CHECK(sizeof(RLMeshField) ==
        sizeof(((RLMeshField*)0)->struct_size) + sizeof(((RLMeshField*)0)->role) +
            sizeof(((RLMeshField*)0)->dim) + sizeof(((RLMeshField*)0)->encoding) +
            sizeof(((RLMeshField*)0)->encoding_len) + sizeof(((RLMeshField*)0)->encoding_other) +
            sizeof(((RLMeshField*)0)->has_range) + sizeof(((RLMeshField*)0)->_reserved0) +
            sizeof(((RLMeshField*)0)->range) + sizeof(((RLMeshField*)0)->has_frame) +
            sizeof(((RLMeshField*)0)->_reserved1) + sizeof(((RLMeshField*)0)->frame) +
            sizeof(((RLMeshField*)0)->frame_other) + sizeof(((RLMeshField*)0)->has_provenance) +
            sizeof(((RLMeshField*)0)->_reserved2) + sizeof(((RLMeshField*)0)->provenance) +
            sizeof(((RLMeshField*)0)->provenance_other) + sizeof(((RLMeshField*)0)->part) +
            sizeof(((RLMeshField*)0)->labels) + sizeof(((RLMeshField*)0)->labels_len));
  CHECK(sizeof(RLMeshSplit) == sizeof(((RLMeshSplit*)0)->struct_size) +
                                   sizeof(((RLMeshSplit*)0)->fields) +
                                   sizeof(((RLMeshSplit*)0)->fields_len));
  CHECK(sizeof(RLMeshPreviousOptions) == sizeof(((RLMeshPreviousOptions*)0)->struct_size) +
                                             sizeof(((RLMeshPreviousOptions*)0)->part) +
                                             sizeof(((RLMeshPreviousOptions*)0)->fill));
  CHECK(sizeof(RLMeshObservationRoles) == sizeof(((RLMeshObservationRoles*)0)->struct_size) +
                                              sizeof(((RLMeshObservationRoles*)0)->images) +
                                              sizeof(((RLMeshObservationRoles*)0)->images_len) +
                                              sizeof(((RLMeshObservationRoles*)0)->states) +
                                              sizeof(((RLMeshObservationRoles*)0)->states_len) +
                                              sizeof(((RLMeshObservationRoles*)0)->texts) +
                                              sizeof(((RLMeshObservationRoles*)0)->texts_len));
  CHECK(sizeof(RLMeshAdvisory) ==
        sizeof(((RLMeshAdvisory*)0)->struct_size) + sizeof(((RLMeshAdvisory*)0)->severity) +
            sizeof(((RLMeshAdvisory*)0)->_reserved) + sizeof(((RLMeshAdvisory*)0)->message));
  CHECK(sizeof(RLMeshTagOptions) == sizeof(((RLMeshTagOptions*)0)->struct_size) +
                                        sizeof(((RLMeshTagOptions*)0)->validate) +
                                        sizeof(((RLMeshTagOptions*)0)->_reserved0) +
                                        sizeof(((RLMeshTagOptions*)0)->_reserved1) +
                                        sizeof(((RLMeshTagOptions*)0)->_reserved2));
  CHECK(sizeof(RLMeshResolveOptions) == sizeof(((RLMeshResolveOptions*)0)->struct_size) +
                                            sizeof(((RLMeshResolveOptions*)0)->trust_entrypoints) +
                                            sizeof(((RLMeshResolveOptions*)0)->check_inverse) +
                                            sizeof(((RLMeshResolveOptions*)0)->_reserved0) +
                                            sizeof(((RLMeshResolveOptions*)0)->_reserved1));
#endif
}
