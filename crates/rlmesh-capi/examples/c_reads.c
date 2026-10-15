/* Adapter readers, instruction injection, retained contexts and render reach. */
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
typedef struct State {
  int step, predicts, reads, frees;
  bool render_fail;
  bool counted_sources;
  int sources_calls;
  RLMeshFrameSources* inspect_source;
  RLMeshRunContext* kept;
  RLMeshFrameSources* kept_sources;
  RLMeshReadItem* item;
  RLMeshSession* session;
  RLMeshFrameSources* transfer_source;
} State;
static RLMeshValue* vector(float n) {
  const size_t shape[] = {1};
  RLMeshTensor* t = NULL;
  RLMeshValue* v = NULL;
  CHECK(rlmesh_tensor_new(&n, sizeof(n), shape, 1, RLMESH_DTYPE_FLOAT32, &t) == RLMESH_OK);
  CHECK(rlmesh_value_tensor(t, &v) == RLMESH_OK);
  rlmesh_tensor_free(t);
  return v;
}
static RLMeshStatus reset(void* user, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  (void)seed;
  (void)options;
  (void)info;
  ((State*)user)->step = 0;
  *obs = vector(2);
  return RLMESH_OK;
}
static RLMeshStatus step(void* user, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                         bool* term, bool* trunc, RLMeshValue** info) {
  (void)action;
  (void)info;
  ++((State*)user)->step;
  *obs = vector(3);
  *reward = 1;
  *term = true;
  *trunc = false;
  return RLMESH_OK;
}
static RLMeshStatus render(void* user, RLMeshValue** out) {
  if (((State*)user)->inspect_source)
    CHECK(rlmesh_frame_sources_roles_len(((State*)user)->inspect_source) == 2);
  if (((State*)user)->render_fail) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION, "render failed");
    return RLMESH_ERR_ENVIRONMENT;
  }
  return rlmesh_value_int(((State*)user)->step, out);
}
static RLMeshStatus predict(void* user, const RLMeshValue* obs, const RLMeshPredictContext* context,
                            RLMeshValue** out) {
  (void)context;
  State* s = (State*)user;
  ++s->predicts;
  const RLMeshValue* v = NULL;
  CHECK(rlmesh_value_dict_get(obs, rlmesh_str("instruction"), &v) == RLMESH_OK);
  RLMeshStr text = {NULL, 0};
  CHECK(rlmesh_value_as_text(v, &text) == RLMESH_OK && text.len == 8 &&
        memcmp(text.data, "override", 8) == 0);
  *out = vector(0);
  return RLMESH_OK;
}
static RLMeshStatus start(void* user, const RLMeshRunContext* ctx) {
  State* s = (State*)user;
  rlmesh_run_context_free(s->kept);
  s->kept = rlmesh_run_context_clone(ctx);
  return RLMESH_OK;
}
static RLMeshStatus on_step(void* user, const RLMeshStepEvent* event) {
  State* s = (State*)user;
  RLMeshValue* out = NULL;
  CHECK(rlmesh_step_event_read(event, s->item, &out) == RLMESH_OK &&
        rlmesh_value_kind(out) == RLMESH_VALUE_TENSOR);
  rlmesh_value_free(out);
  ++s->reads;
  RLMeshFrameSources* sources = NULL;
  CHECK(rlmesh_run_context_frame_sources(event->context, &sources) == RLMESH_OK);
  bool has = false;
  out = NULL;
  CHECK(rlmesh_frame_sources_render(sources, &out, &has) == RLMESH_OK && has);
  int64_t n = -1;
  CHECK(rlmesh_value_as_int(out, &n) == RLMESH_OK && n == 1);
  rlmesh_value_free(out);
  rlmesh_frame_sources_free(s->kept_sources);
  s->kept_sources = sources;
  if (s->session) {
    RLMeshReader* forbidden = NULL;
    const RLMeshReadItem* items[] = {s->item};
    CHECK(rlmesh_session_reader(s->session, items, 1, &forbidden) == RLMESH_ERR_RUNTIME &&
          !forbidden);
    RLMeshObservationRoles roles = RLMESH_OBSERVATION_ROLES_INIT;
    CHECK(rlmesh_session_observation_roles(s->session, &roles) == RLMESH_ERR_RUNTIME);
    out = NULL;
    CHECK(rlmesh_session_read(s->session, event->observation, s->item, &out) == RLMESH_OK);
    rlmesh_value_free(out);
    CHECK(rlmesh_session_frame_sources(s->session, &sources) == RLMESH_OK);
    rlmesh_frame_sources_free(sources);
  }
  return RLMESH_OK;
}
static void free_state(void* user) { ++((State*)user)->frees; }
static RLMeshStatus context_contract(void* user, RLMeshEnvContract** out) {
  (void)user;
  *out = NULL;
  return RLMESH_OK;
}
static RLMeshStatus context_num(void* user, uint64_t* out) {
  (void)user;
  *out = 1;
  return RLMESH_OK;
}
static RLMeshStatus context_read(void* user, const RLMeshValue* observation,
                                 const RLMeshReadItem* item, RLMeshValue** out) {
  (void)user;
  (void)item;
  *out = rlmesh_value_clone(observation);
  return RLMESH_OK;
}
static RLMeshStatus context_sources(void* user, RLMeshFrameSources** out) {
  State* state = (State*)user;
  ++state->sources_calls;
  if (state->transfer_source) {
    *out = state->transfer_source;
    state->transfer_source = NULL;
    return RLMESH_OK;
  }
  if (state->counted_sources)
    return rlmesh_frame_sources_new(NULL, 0, "render", render, user, free_state, out);
  return rlmesh_frame_sources_new(NULL, 0, NULL, NULL, NULL, NULL, out);
}
int main(void) {
  State state = {0};
  RLMeshReadItem* item = NULL;
  CHECK(rlmesh_read_item_role(RLMESH_ADAPTERS_GRIPPER_POS, &item) == RLMESH_OK);
  state.item = item;
  RLMeshRunContextVtable context_vt = RLMESH_RUN_CONTEXT_VTABLE_INIT;
  context_vt.free_user_data = free_state;
  RLMeshRunContext* custom = NULL;
  CHECK(rlmesh_run_context_new(&context_vt, &state, &custom) == RLMESH_ERR_INVALID_ARGUMENT &&
        state.frees == 0);
  context_vt.contract = context_contract;
  context_vt.num_envs = context_num;
  context_vt.read = context_read;
  context_vt.frame_sources = context_sources;
  CHECK(rlmesh_run_context_new(&context_vt, &state, &custom) == RLMESH_OK);
  RLMeshRunContext* custom_clone = rlmesh_run_context_clone(custom);
  rlmesh_run_context_free(custom);
  CHECK(state.frees == 0);
  RLMeshValue* raw = vector(2);
  RLMeshValue* out = NULL;
  CHECK(rlmesh_run_context_read(custom_clone, raw, item, &out) == RLMESH_OK &&
        rlmesh_value_equals(raw, out));
  rlmesh_value_free(out);
  RLMeshFrameSources* empty = NULL;
  CHECK(rlmesh_run_context_frame_sources(custom_clone, &empty) == RLMESH_OK &&
        rlmesh_frame_sources_labels_len(empty) == 0);
  bool has = true;
  out = (RLMeshValue*)1;
  CHECK(rlmesh_frame_sources_render(empty, &out, &has) == RLMESH_OK && !has &&
        out == (RLMeshValue*)1);
  RLMeshBytes bytes = {(uint8_t*)1, 9, 0};
  CHECK(rlmesh_frame_sources_render_label(empty, &bytes, &has) == RLMESH_OK && !has &&
        bytes.data == (uint8_t*)1 && bytes.len == 9);
  rlmesh_frame_sources_free(empty);
  rlmesh_run_context_free(custom_clone);
  CHECK(state.frees == 1);
  State source_state = {0};
  source_state.counted_sources = true;
  CHECK(rlmesh_run_context_new(&context_vt, &source_state, &custom) == RLMESH_OK);
  RLMeshFrameSources* authored_sources = NULL;
  CHECK(rlmesh_run_context_frame_sources(custom, &authored_sources) == RLMESH_OK &&
        source_state.sources_calls == 1 && source_state.frees == 0);
  for (int i = 0; i < 2; ++i) {
    out = NULL;
    CHECK(rlmesh_frame_sources_render(authored_sources, &out, &has) == RLMESH_OK && has &&
          source_state.sources_calls == 1 && source_state.frees == 0);
    rlmesh_value_free(out);
  }
  rlmesh_frame_sources_free(authored_sources);
  CHECK(source_state.frees == 1);
  rlmesh_run_context_free(custom);
  CHECK(source_state.frees == 2);
  const char* roles[] = {"render", "image"};
  RLMeshFrameSources* caller = NULL;
  RLMeshFrameSources* caller_clone = NULL;
  CHECK(rlmesh_frame_sources_new(roles, 2, "render()", render, &state, free_state, &caller) ==
        RLMESH_OK);
  CHECK(rlmesh_frame_sources_new(roles, 2, "render()", render, &state, free_state, &caller_clone) ==
        RLMESH_OK);
  state.inspect_source = caller;
  out = NULL;
  CHECK(rlmesh_frame_sources_render(caller, &out, &has) == RLMESH_OK && has);
  rlmesh_value_free(out);
  state.inspect_source = NULL;
  CHECK(rlmesh_frame_sources_equals(caller, caller_clone));
  CHECK(rlmesh_frame_sources_labels_len(caller) == 3);
  bytes = (RLMeshBytes){0};
  CHECK(rlmesh_frame_sources_default_label(caller, &bytes, &has) == RLMESH_OK && has &&
        bytes.len == 6 && memcmp(bytes.data, "render", 6) == 0);
  rlmesh_bytes_free(bytes);
  CHECK(rlmesh_frame_sources_repr(caller, &bytes) == RLMESH_OK);
  rlmesh_bytes_free(bytes);
  rlmesh_frame_sources_free(caller);
  rlmesh_frame_sources_free(caller_clone);
  CHECK(state.frees == 3);
  /* Result constructors copy advisory inputs; lending survives shared clones. */
  char message[] = "copied";
  RLMeshAdvisory note = RLMESH_ADVISORY_INIT;
  note.severity = RLMESH_ADVISORY_SEVERITY_INFO;
  note.message = message;
  const RLMeshAdvisory* notes[] = {&note};
  RLMeshRunResultOptions ro = RLMESH_RUN_RESULT_OPTIONS_INIT;
  ro.advisories = notes;
  ro.advisories_len = 1;
  RLMeshRunResult* result = NULL;
  CHECK(rlmesh_run_result_new(&ro, &result) == RLMESH_OK);
  message[0] = 'X';
  RLMeshRunResult* result_clone = rlmesh_run_result_clone(result);
  rlmesh_run_result_free(result);
  RLMeshAdvisory copied = RLMESH_ADVISORY_INIT;
  CHECK(rlmesh_run_result_advisories_get(result_clone, 0, &copied) == RLMESH_OK &&
        strcmp(copied.message, "copied") == 0);
  CHECK(rlmesh_run_result_advisories_len(result_clone) == 1);
  struct FutureAdvisory {
    RLMeshAdvisory head;
    uint64_t tail;
  } future_note = {RLMESH_ADVISORY_INIT, 123};
  future_note.head.struct_size = sizeof(future_note);
  CHECK(rlmesh_run_result_advisories_get(result_clone, 0, &future_note.head) == RLMESH_OK &&
        future_note.tail == 123 && future_note.head.struct_size == sizeof(future_note));
  rlmesh_run_result_free(result_clone);
  const int64_t shape[] = {1};
  RLMeshBox* space = NULL;
  CHECK(rlmesh_box_new(rlmesh_bound_float(-10), rlmesh_bound_float(10), shape, 1, NULL, &space) ==
        RLMESH_OK);
  RLMeshActuator actuator = RLMESH_ACTUATOR_INIT;
  actuator.role = RLMESH_ADAPTERS_ACTION_GRIPPER;
  actuator.dim = 1;
  const RLMeshActuator* components[] = {&actuator};
  RLMeshAction action = RLMESH_ACTION_INIT;
  action.components = components;
  action.components_len = 1;
  RLMeshStateTag tag = RLMESH_STATE_TAG_INIT;
  tag.role = RLMESH_ADAPTERS_GRIPPER_POS;
  RLMeshObsNode obs_tag = rlmesh_obs_node_state_tag(&tag);
  RLMeshEnvTags* tags = NULL;
  CHECK(rlmesh_env_tags_new(&obs_tag, &action, &tags) == RLMESH_OK);
  RLMeshState input_state = RLMESH_STATE_INIT;
  input_state.role = tag.role;
  RLMeshAdaptersText input_text = RLMESH_ADAPTERS_TEXT_INIT;
  input_text.role = RLMESH_ADAPTERS_INSTRUCTION;
  input_text.has_fill = true;
  input_text.fill = rlmesh_str("fallback");
  RLMeshImage input_image = RLMESH_IMAGE_INIT;
  input_image.role = RLMESH_ADAPTERS_IMAGE_PRIMARY;
  input_image.optional = true;
  input_image.has_height = input_image.has_width = input_image.has_channels = true;
  input_image.height = input_image.width = 1;
  input_image.channels = 3;
  RLMeshInputNode children[] = {rlmesh_input_node_state(&input_state),
                                rlmesh_input_node_text(&input_text),
                                rlmesh_input_node_image(&input_image)};
  RLMeshStr keys[] = {rlmesh_str("state"), rlmesh_str("instruction"), rlmesh_str("camera")};
  RLMeshInputNode input = rlmesh_input_node_dict(keys, children, 3);
  RLMeshModelSpec* spec = NULL;
  CHECK(rlmesh_model_spec_new(&input, &action, &spec) == RLMESH_OK);
  RLMeshEnvVtable ev = RLMESH_ENV_VTABLE_INIT;
  ev.reset = reset;
  ev.step = step;
  ev.render = render;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = space;
  attrs.action_space = space;
  attrs.render_mode = "rgb_array";
  RLMeshEnv* env = NULL;
  CHECK(rlmesh_env_new(&ev, &state, &attrs, &env) == RLMESH_OK &&
        rlmesh_adapters_tag(env, tags, NULL) == RLMESH_OK);
  RLMeshModelVtable policy = RLMESH_MODEL_VTABLE_INIT;
  policy.predict = predict;
  policy.workflow_edition = rlmesh_current_workflow_edition();
  RLMeshModelOptions mo = RLMESH_MODEL_OPTIONS_INIT;
  mo.spec = (RLMeshSpecArg){RLMESH_SPEC_KIND_MODEL_SPEC, spec};
  RLMeshModel* model = NULL;
  CHECK(rlmesh_model_new(&policy, &state, &mo, &model) == RLMESH_OK);
  RLMeshRunHooksVtable hv = RLMESH_RUN_HOOKS_VTABLE_INIT;
  hv.on_run_start = start;
  hv.on_step = on_step;
  RLMeshRunHooks* hooks = NULL;
  CHECK(rlmesh_run_hooks_new(&hv, &state, &hooks) == RLMESH_OK);
  RLMeshRunOptions run = RLMESH_RUN_OPTIONS_INIT;
  run.hooks = hooks;
  run.instruction = "override";
  run.workflow_edition = rlmesh_current_workflow_edition();
  run.close_env = false;
  CHECK(rlmesh_model_run(model, rlmesh_env_target(env), &run, &result) == RLMESH_OK);
  CHECK(state.reads == 1 && state.predicts == 1 && rlmesh_run_result_advisories_len(result) > 0);
  copied = (RLMeshAdvisory)RLMESH_ADVISORY_INIT;
  CHECK(rlmesh_run_result_advisories_get(result, 0, &copied) == RLMESH_OK && copied.message);
  rlmesh_run_result_free(result);
  out = NULL;
  CHECK(rlmesh_run_context_read(state.kept, raw, item, &out) == RLMESH_OK);
  rlmesh_value_free(out);
  out = NULL;
  CHECK(rlmesh_frame_sources_render(state.kept_sources, &out, &has) == RLMESH_ERR_RUNTIME && !out);
  rlmesh_frame_sources_free(state.kept_sources);
  state.kept_sources = NULL;
  CHECK(rlmesh_run_context_frame_sources(state.kept, &empty) == RLMESH_OK &&
        rlmesh_frame_sources_labels_len(empty) == 0);
  rlmesh_frame_sources_free(empty);
  RLMeshSessionOptions session_options = RLMESH_SESSION_OPTIONS_INIT;
  char instruction[] = "override";
  session_options.instruction = instruction;
  session_options.workflow_edition = rlmesh_current_workflow_edition();
  CHECK(rlmesh_model_session(model, rlmesh_env_target(env), &session_options, &state.session) ==
        RLMESH_OK);
  instruction[0] = 'X'; /* Session copied it at creation. */
  RLMeshValue* observation = NULL;
  RLMeshValue* info = NULL;
  CHECK(rlmesh_session_reset(state.session, NULL, NULL, &observation, &info) == RLMESH_OK);
  rlmesh_value_free(info);
  RLMeshEnvContract* contract = NULL;
  CHECK(rlmesh_session_contract(state.session, &contract) == RLMESH_OK);
  RLMeshAdapter* resolved = NULL;
  CHECK(rlmesh_adapters_resolve_from_contract(contract, spec, NULL, &resolved) == RLMESH_OK);
  rlmesh_env_contract_free(contract);
  const char* direct_roles[] = {"state"};
  RLMeshReader* direct_reader = NULL;
  CHECK(rlmesh_reader_new(resolved, direct_roles, 1, &direct_reader) == RLMESH_OK);
  rlmesh_adapter_free(resolved);
  out = NULL;
  CHECK(rlmesh_reader_call(direct_reader, observation, &out) == RLMESH_OK &&
        rlmesh_value_kind(out) == RLMESH_VALUE_DICT);
  rlmesh_value_free(out);
  rlmesh_reader_free(direct_reader);
  const RLMeshReadItem* items[] = {item};
  RLMeshReader* reader = NULL;
  CHECK(rlmesh_session_reader(state.session, items, 1, &reader) == RLMESH_OK &&
        rlmesh_reader_roles_len(reader) == 1);
  CHECK(rlmesh_reader_roles_get(reader, 0, &bytes) == RLMESH_OK);
  rlmesh_bytes_free(bytes);
  CHECK(rlmesh_reader_call(reader, observation, &out) == RLMESH_OK &&
        rlmesh_value_kind(out) == RLMESH_VALUE_DICT);
  rlmesh_value_free(out);
  RLMeshObservationRoles observed = RLMESH_OBSERVATION_ROLES_INIT;
  CHECK(rlmesh_session_observation_roles(state.session, &observed) == RLMESH_OK &&
        observed.states_len == 1 && strcmp(observed.states[0], tag.role) == 0);
  RLMeshFrameSources* direct = NULL;
  CHECK(rlmesh_session_frame_sources(state.session, &direct) == RLMESH_OK);
  State transfer_state = {0};
  CHECK(rlmesh_session_frame_sources(state.session, &transfer_state.transfer_source) == RLMESH_OK);
  RLMeshRunContext* transfer_context = NULL;
  CHECK(rlmesh_run_context_new(&context_vt, &transfer_state, &transfer_context) == RLMESH_OK);
  RLMeshFrameSources* transferred = NULL;
  CHECK(rlmesh_run_context_frame_sources(transfer_context, &transferred) == RLMESH_OK);
  CHECK(transfer_state.transfer_source == NULL);
  rlmesh_run_context_free(transfer_context); /* transferred source retains the context */
  out = NULL;
  CHECK(rlmesh_frame_sources_render(direct, &out, &has) == RLMESH_OK && has);
  rlmesh_value_free(out);
  state.render_fail = true;
  out = NULL;
  CHECK(rlmesh_frame_sources_render(direct, &out, &has) == RLMESH_ERR_ENVIRONMENT && !out &&
        strcmp(rlmesh_last_error_message(), "[INTERNAL] render failed") == 0);
  state.render_fail = false;
  RLMeshSessionRunOptions sr = RLMESH_SESSION_RUN_OPTIONS_INIT;
  sr.hooks = hooks;
  CHECK(rlmesh_session_run(state.session, &sr, &result) == RLMESH_OK && state.reads == 2);
  rlmesh_run_result_free(result);
  CHECK(rlmesh_session_close(state.session) == RLMESH_OK);
  out = NULL;
  CHECK(rlmesh_frame_sources_render(direct, &out, &has) == RLMESH_ERR_RUNTIME && !out);
  CHECK(rlmesh_run_context_read(state.kept, raw, item, &out) == RLMESH_ERR_RUNTIME);
  CHECK(rlmesh_run_context_frame_sources(state.kept, &empty) == RLMESH_ERR_RUNTIME && !empty);
  CHECK(strcmp(observed.states[0], tag.role) == 0);
  CHECK(rlmesh_reader_call(reader, observation, &out) == RLMESH_OK);
  rlmesh_value_free(out);
  rlmesh_value_free(observation);
  rlmesh_reader_free(reader);
  rlmesh_frame_sources_free(state.kept_sources);
  rlmesh_run_context_free(state.kept);
  rlmesh_session_free(state.session);
  state.session = NULL;
  out = NULL;
  CHECK(rlmesh_frame_sources_render(direct, &out, &has) == RLMESH_ERR_RUNTIME && !out);
  CHECK(strstr(rlmesh_last_error_message(), "session has been freed") != NULL);
  CHECK(rlmesh_frame_sources_render(transferred, &out, &has) == RLMESH_ERR_RUNTIME && !out);
  CHECK(strstr(rlmesh_last_error_message(), "session has been freed") != NULL);
  CHECK(transfer_state.frees == 0);
  rlmesh_frame_sources_free(transferred);
  CHECK(transfer_state.frees == 1);
  rlmesh_frame_sources_free(direct);
  rlmesh_run_hooks_free(hooks);
  rlmesh_model_free(model);
  rlmesh_env_free(env);
  rlmesh_env_tags_free(tags);
  rlmesh_model_spec_free(spec);
  rlmesh_space_free(space);
  rlmesh_read_item_free(item);
  rlmesh_value_free(raw);
  if (failures) return 1;
  puts("c_reads OK");
  return 0;
}
