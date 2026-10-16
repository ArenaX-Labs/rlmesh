/* ABI 5 View / Recorder: facade semantics, owned snapshots and capture hooks. */
#include <assert.h>
#include <float.h>
#include <math.h>
#include <rlmesh.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(RLMeshView) == 2 * sizeof(size_t) + 40, "View has no padding");
_Static_assert(sizeof(RLMeshRecorderOptions) == 2 * sizeof(size_t) + 16, "Recorder options layout");
_Static_assert(sizeof(RLMeshRecorderCaptureOptions) == 9 * sizeof(size_t),
               "Capture options layout");

#define OK(call)                                                                                  \
  do {                                                                                            \
    RLMeshStatus s_ = (call);                                                                     \
    if (s_ != RLMESH_OK) {                                                                        \
      fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, rlmesh_last_error_message()); \
      assert(s_ == RLMESH_OK);                                                                    \
    }                                                                                             \
  } while (0)
static void bytes_equal(RLMeshBytes bytes, const char* text) {
  assert(bytes.len == strlen(text) && memcmp(bytes.data, text, bytes.len) == 0);
  rlmesh_bytes_free(bytes);
}
static const RLMeshValue* field(const RLMeshValue* dict, const char* key) {
  const RLMeshValue* value = NULL;
  OK(rlmesh_value_dict_get(dict, rlmesh_str(key), &value));
  assert(value);
  return value;
}
static void text_equal(const RLMeshValue* value, const char* text) {
  RLMeshStr str = {NULL, 0};
  OK(rlmesh_value_as_text(value, &str));
  assert(str.len == strlen(text) && memcmp(str.data, text, str.len) == 0);
}
static void view_values(void) {
  RLMeshView view = RLMESH_VIEW_INIT;
  OK(rlmesh_view_validate(&view));
  assert(view.fps == 30 && view.quality == 75 && view.port == 8008 && !view.source);
  assert(view.backend == RLMESH_VIEW_BACKEND_TERMINAL && view.format == RLMESH_VIEW_FORMAT_JPEG);
  assert(view.hold == RLMESH_VIEW_HOLD_OFF && !view.has_step_hz);
  const char* valid[] = {"", " terminal ", "TERM", "tty", "both", "http", "http:", " HTTP:9000 "};
  for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); ++i) {
    OK(rlmesh_view_parse(valid[i], &view));
    OK(rlmesh_view_validate(&view));
    assert(!view.source);
  }
  assert(view.backend == RLMESH_VIEW_BACKEND_HTTP && view.port == 9000);
  RLMeshView other = RLMESH_VIEW_INIT;
  OK(rlmesh_view_parse("http:9000", &other));
  bool equal = false;
  OK(rlmesh_view_equals(&view, &other, &equal));
  assert(equal);
  other.quality = 76;
  OK(rlmesh_view_equals(&view, &other, &equal));
  assert(!equal);
  RLMeshView hold = RLMESH_VIEW_INIT;
  hold.hold = RLMESH_VIEW_HOLD_SECONDS;
  hold.hold_seconds = DBL_MAX;
  OK(rlmesh_view_validate(&hold));
  RLMeshView saturated = hold;
  saturated.hold_seconds = ldexp(1.0, 64); /* also beyond Duration::MAX */
  OK(rlmesh_view_equals(&hold, &saturated, &equal));
  assert(equal);
  saturated.hold_seconds = 0.0;
  OK(rlmesh_view_equals(&hold, &saturated, &equal));
  assert(!equal);
  const double invalid_holds[] = {INFINITY, -INFINITY, NAN, -1.0};
  for (size_t i = 0; i < sizeof(invalid_holds) / sizeof(invalid_holds[0]); ++i) {
    hold.hold_seconds = invalid_holds[i];
    assert(rlmesh_view_validate(&hold) == RLMESH_ERR_INVALID_ARGUMENT);
    assert(rlmesh_last_error_class() == RLMESH_ERROR_CLASS_VALUE_ERROR);
  }
  RLMeshBytes repr = {0};
  OK(rlmesh_view_repr(&view, &repr));
  assert(repr.len > 20);
  rlmesh_bytes_free(repr);
  const char* invalid[] = {"http:0", "http:65536", "http:+1", "http: 1", "foo"};
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
    assert(rlmesh_view_parse(invalid[i], &view) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(rlmesh_view_parse(NULL, &view) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(rlmesh_view_parse("http", NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  for (int i = 0; i < 8; ++i) {
    RLMeshView v = RLMESH_VIEW_INIT;
    switch (i) {
      case 0:
        v.backend = RLMESH_VIEW_BACKEND_INVALID;
        break;
      case 1:
        v.format = RLMESH_VIEW_FORMAT_INVALID;
        break;
      case 2:
        v.hold = RLMESH_VIEW_HOLD_INVALID;
        break;
      case 3:
        v.fps = 0;
        break;
      case 4:
        v.port = 0;
        break;
      case 5:
        v.quality = 101;
        break;
      case 6:
        v.has_step_hz = true;
        v.step_hz = NAN;
        break;
      case 7:
        v.hold = RLMESH_VIEW_HOLD_SECONDS;
        v.hold_seconds = -1;
        break;
    }
    assert(rlmesh_view_validate(&v) == RLMESH_ERR_INVALID_ARGUMENT);
    if (i < 3) assert(rlmesh_view_equals(&v, &view, &equal) == RLMESH_ERR_INVALID_ARGUMENT);
  }
  RLMeshView zero = {0};
  assert(rlmesh_view_validate(&zero) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(rlmesh_view_parse("term", &zero) == RLMESH_ERR_INVALID_ARGUMENT);
  struct Future {
    RLMeshView known;
    unsigned char tail[8];
  } future = {RLMESH_VIEW_INIT, {0}};
  future.known.struct_size = sizeof(future);
  memset(future.tail, 0xA5, sizeof(future.tail));
  OK(rlmesh_view_parse("both", &future.known));
  assert(future.known.struct_size == sizeof(future) && future.known.source == NULL);
  for (size_t i = 0; i < sizeof(future.tail); ++i) assert(future.tail[i] == 0xA5);
  assert(rlmesh_view_validate(&future.known) == RLMESH_ERR_TYPE);
  memset(future.tail, 0, sizeof(future.tail));
  OK(rlmesh_view_validate(&future.known));
  future.known.struct_size = RLMESH_VIEW_MIN_SIZE - 1;
  assert(rlmesh_view_validate(&future.known) == RLMESH_ERR_INVALID_ARGUMENT);
}

static void record_snapshots(void) {
  RLMeshMediaRefOptions mo = RLMESH_MEDIA_REF_OPTIONS_INIT;
  mo.has_frame_count = true;
  mo.frame_count = 0;
  mo.has_width = true;
  mo.width = -2;
  mo.has_fps = true;
  mo.fps = -1; /* dataclasses impose no extra validation */
  RLMeshMediaRef *media = NULL, *copy = NULL;
  OK(rlmesh_media_ref_new("cam", "video", "m.mp4", "mp4", &mo, &media));
  OK(rlmesh_media_ref_new("cam", "video", "m.mp4", "mp4", &mo, &copy));
  assert(rlmesh_media_ref_equals(media, copy));
  int64_t n = 99;
  double fps = 0;
  assert(rlmesh_media_ref_frame_count(media, &n) && n == 0);
  assert(rlmesh_media_ref_width(media, &n) && n == -2);
  assert(!rlmesh_media_ref_height(media, &n));
  assert(rlmesh_media_ref_fps(media, &fps) && fps == -1);
  RLMeshBytes bytes = {0};
  OK(rlmesh_media_ref_camera(media, &bytes));
  bytes_equal(bytes, "cam");
  OK(rlmesh_media_ref_kind(media, &bytes));
  bytes_equal(bytes, "video");
  OK(rlmesh_media_ref_path(media, &bytes));
  bytes_equal(bytes, "m.mp4");
  OK(rlmesh_media_ref_format(media, &bytes));
  bytes_equal(bytes, "mp4");
  OK(rlmesh_media_ref_repr(media, &bytes));
  assert(bytes.len > 10);
  rlmesh_bytes_free(bytes);
  RLMeshValue* dict = NULL;
  OK(rlmesh_media_ref_to_dict(media, &dict));
  text_equal(field(dict, "camera"), "cam");
  rlmesh_value_free(dict);
  RLMeshEpisodeRecordOptions eo = RLMESH_EPISODE_RECORD_OPTIONS_INIT;
  const RLMeshMediaRef* media_list[] = {media};
  eo.media = media_list;
  eo.media_len = 1;
  eo.has_success = true;
  eo.success = true;
  eo.duration_s = .125;
  int64_t seed = -7;
  RLMeshEpisodeRecord* episode = NULL;
  OK(rlmesh_episode_record_new(2, &seed, 3, 4.5, true, false, &eo, &episode));
  rlmesh_media_ref_free(media); /* child data was retained */
  media = NULL;
  OK(rlmesh_episode_record_media_get(episode, 0, &media));
  assert(rlmesh_media_ref_equals(media, copy));
  assert(rlmesh_episode_record_index(episode) == 2 && rlmesh_episode_record_steps(episode) == 3);
  assert(rlmesh_episode_record_seed(episode, &n) && n == -7);
  assert(rlmesh_episode_record_reward(episode) == 4.5);
  assert(rlmesh_episode_record_terminated(episode) && !rlmesh_episode_record_truncated(episode));
  bool success = false;
  assert(rlmesh_episode_record_success(episode, &success) && success);
  assert(rlmesh_episode_record_duration_s(episode) == .125 &&
         rlmesh_episode_record_included_in_metrics(episode));
  assert(rlmesh_episode_record_media_len(episode) == 1);
  RLMeshMediaRef* absent = media;
  assert(rlmesh_episode_record_media_get(episode, 1, &absent) == RLMESH_ERR_INVALID_ARGUMENT &&
         !absent);
  RLMeshWorkloadRecordOptions wo = RLMESH_WORKLOAD_RECORD_OPTIONS_INIT;
  const RLMeshEpisodeRecord* episodes[] = {episode};
  wo.episodes = episodes;
  wo.episodes_len = 1;
  RLMeshWorkloadRecord* workload = NULL;
  char model[] = "model";
  OK(rlmesh_workload_record_new(model, "env", "task", &wo, &workload));
  memset(model, 'x', 5);
  rlmesh_episode_record_free(episode);
  OK(rlmesh_workload_record_model(workload, &bytes));
  bytes_equal(bytes, "model");
  OK(rlmesh_workload_record_env(workload, &bytes));
  bytes_equal(bytes, "env");
  OK(rlmesh_workload_record_task(workload, &bytes));
  bytes_equal(bytes, "task");
  OK(rlmesh_workload_record_config(workload, &dict));
  assert(rlmesh_value_kind(dict) == RLMESH_VALUE_DICT);
  rlmesh_value_free(dict);
  assert(rlmesh_workload_record_mean_reward(workload) == 4.5 &&
         rlmesh_workload_record_total_steps(workload) == 3);
  double rate = 0;
  assert(rlmesh_workload_record_success_rate(workload, &rate) && rate == 1);
  OK(rlmesh_workload_record_to_dict(workload, &dict));
  text_equal(field(dict, "model"), "model");
  rlmesh_value_free(dict);
  OK(rlmesh_workload_record_episodes_get(workload, 0, &episode));
  assert(rlmesh_episode_record_equals(episode, episode));
  OK(rlmesh_episode_record_to_dict(episode, &dict));
  rlmesh_value_free(dict);
  OK(rlmesh_episode_record_repr(episode, &bytes));
  rlmesh_bytes_free(bytes);
  OK(rlmesh_workload_record_repr(workload, &bytes));
  rlmesh_bytes_free(bytes);
  assert(rlmesh_workload_record_equals(workload, workload));
  rlmesh_workload_record_free(workload);
  assert(rlmesh_episode_record_reward(episode) == 4.5); /* independent snapshot */
  rlmesh_episode_record_free(episode);
  rlmesh_media_ref_free(media);
  rlmesh_media_ref_free(copy);
  assert(rlmesh_episode_record_steps(NULL) == 0 && !rlmesh_episode_record_success(NULL, &success));
  assert(rlmesh_workload_record_episodes_len(NULL) == 0 &&
         !rlmesh_workload_record_equals(NULL, NULL));
}

typedef struct State {
  int renders;
  int context_frees;
  uint64_t lanes;
} State;
static RLMeshStatus reset(void* user, const int64_t* seed, const RLMeshValue* options,
                          RLMeshValue** obs, RLMeshValue** info) {
  (void)user;
  (void)seed;
  (void)options;
  (void)info;
  return rlmesh_value_int(0, obs);
}
static RLMeshStatus step(void* user, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                         bool* term, bool* trunc, RLMeshValue** info) {
  (void)user;
  (void)action;
  (void)info;
  *reward = 2;
  *term = true;
  *trunc = false;
  return rlmesh_value_int(0, obs);
}
static RLMeshStatus render(void* user, RLMeshValue** out) {
  ++((State*)user)->renders;
  unsigned char pixels[16 * 16 * 3];
  memset(pixels, 128, sizeof(pixels));
  size_t shape[] = {16, 16, 3};
  RLMeshTensor* tensor = NULL;
  RLMeshStatus status =
      rlmesh_tensor_new(pixels, sizeof(pixels), shape, 3, RLMESH_DTYPE_UINT8, &tensor);
  if (status == RLMESH_OK) status = rlmesh_value_tensor(tensor, out);
  rlmesh_tensor_free(tensor);
  return status;
}
static RLMeshStatus predict(void* user, const RLMeshValue* obs, const RLMeshPredictContext* context,
                            RLMeshValue** out) {
  (void)user;
  (void)obs;
  (void)context;
  return rlmesh_value_int(0, out);
}
static RLMeshStatus contract(void* user, RLMeshEnvContract** out) {
  (void)user;
  *out = NULL;
  return RLMESH_OK;
}
static RLMeshStatus lanes(void* user, uint64_t* out) {
  *out = ((State*)user)->lanes;
  return RLMESH_OK;
}
static RLMeshStatus read_value(void* user, const RLMeshValue* obs, const RLMeshReadItem* item,
                               RLMeshValue** out) {
  (void)user;
  (void)item;
  *out = rlmesh_value_clone(obs);
  return RLMESH_OK;
}
static RLMeshStatus sources(void* user, RLMeshFrameSources** out) {
  (void)user;
  return rlmesh_frame_sources_new(NULL, 0, NULL, NULL, NULL, NULL, out);
}
static void context_free(void* user) { ++((State*)user)->context_frees; }
static RLMeshStatus warning_error(void* user, RLMeshWarningCategory category, const char* message) {
  (void)category;
  if (!strstr(message, "recorder:")) return RLMESH_OK;
  ++*(int*)user;
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_RUNTIME_ERROR, "warnings are errors");
  return RLMESH_ERR_RUNTIME;
}

/* Every sized input refuses {0}/short prefixes, accepts a zero future tail,
 * and refuses each nonzero unknown byte without returning a partial handle. */
#define SIZED_INPUT(Type, Init, Handle, call, release)     \
  do {                                                     \
    struct {                                               \
      Type known;                                          \
      unsigned char tail[8];                               \
    } future_ = {Init, {0}};                               \
    Type* opts = &future_.known;                           \
    Handle* out = NULL;                                    \
    opts->struct_size = 0;                                 \
    assert((call) == RLMESH_ERR_INVALID_ARGUMENT && !out); \
    opts->struct_size = sizeof(Type) - 1;                  \
    assert((call) == RLMESH_ERR_INVALID_ARGUMENT && !out); \
    opts->struct_size = sizeof(future_);                   \
    OK(call);                                              \
    release(out);                                          \
    out = NULL;                                            \
    for (size_t i_ = 0; i_ < sizeof(future_.tail); ++i_) { \
      future_.tail[i_] = 1;                                \
      assert((call) == RLMESH_ERR_TYPE && !out);           \
      future_.tail[i_] = 0;                                \
    }                                                      \
  } while (0)

static void option_boundaries(const char* root) {
  SIZED_INPUT(RLMeshRecorderOptions, RLMESH_RECORDER_OPTIONS_INIT, RLMeshRecorder,
              rlmesh_recorder_new(opts, &out), rlmesh_recorder_free);
  RLMeshRecorder* rec = NULL;
  OK(rlmesh_recorder_new(NULL, &rec));
  RLMeshRunResult* result = NULL;
  OK(rlmesh_run_result_new(NULL, &result));
  SIZED_INPUT(RLMeshRecorderAddOptions, RLMESH_RECORDER_ADD_OPTIONS_INIT, RLMeshWorkloadRecord,
              rlmesh_recorder_add(rec, result, "m", "e", opts, &out), rlmesh_workload_record_free);
  SIZED_INPUT(RLMeshRecorderCaptureOptions, RLMESH_RECORDER_CAPTURE_OPTIONS_INIT, RLMeshRunHooks,
              rlmesh_recorder_capture(rec, "m", "e", opts, &out), rlmesh_run_hooks_free);
  SIZED_INPUT(RLMeshRecorderToDictOptions, RLMESH_RECORDER_TO_DICT_OPTIONS_INIT, RLMeshValue,
              rlmesh_recorder_to_dict(rec, opts, &out), rlmesh_value_free);
  SIZED_INPUT(RLMeshMediaRefOptions, RLMESH_MEDIA_REF_OPTIONS_INIT, RLMeshMediaRef,
              rlmesh_media_ref_new("c", "video", "x", "mp4", opts, &out), rlmesh_media_ref_free);
  SIZED_INPUT(RLMeshEpisodeRecordOptions, RLMESH_EPISODE_RECORD_OPTIONS_INIT, RLMeshEpisodeRecord,
              rlmesh_episode_record_new(0, NULL, 0, 0, false, false, opts, &out),
              rlmesh_episode_record_free);
  SIZED_INPUT(RLMeshWorkloadRecordOptions, RLMESH_WORKLOAD_RECORD_OPTIONS_INIT,
              RLMeshWorkloadRecord, rlmesh_workload_record_new("m", "e", "t", opts, &out),
              rlmesh_workload_record_free);
  char path[1024];
  snprintf(path, sizeof(path), "%s/option-bundle", root);
  RLMeshBytes bytes = {0};
  struct {
    RLMeshRecorderExportOptions known;
    unsigned char tail[8];
  } future = {RLMESH_RECORDER_EXPORT_OPTIONS_INIT, {0}};
  future.known.struct_size = 0;
  assert(rlmesh_recorder_export(rec, path, &future.known, &bytes) == RLMESH_ERR_INVALID_ARGUMENT);
  future.known.struct_size = sizeof(RLMeshRecorderExportOptions) - 1;
  assert(rlmesh_recorder_export(rec, path, &future.known, &bytes) == RLMESH_ERR_INVALID_ARGUMENT);
  future.known.struct_size = sizeof(future);
  for (size_t i = 0; i < sizeof(future.tail); ++i) {
    future.tail[i] = 1;
    assert(rlmesh_recorder_export(rec, path, &future.known, &bytes) == RLMESH_ERR_TYPE);
    future.tail[i] = 0;
  }
  assert(rlmesh_recorder_export(rec, path, &future.known, NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  OK(rlmesh_recorder_export(rec, path, &future.known, &bytes));
  bytes_equal(bytes, path);
  RLMeshRecorderOptions ctor = RLMESH_RECORDER_OPTIONS_INIT;
  RLMeshRecorder* bad = NULL;
  ctor.fps = 0;
  assert(rlmesh_recorder_new(&ctor, &bad) == RLMESH_ERR_INVALID_ARGUMENT && !bad);
  ctor.fps = 30;
  ctor.quality = 0;
  assert(rlmesh_recorder_new(&ctor, &bad) == RLMESH_ERR_INVALID_ARGUMENT);
  ctor.quality = 60;
  ctor._reserved[6] = 1;
  assert(rlmesh_recorder_new(&ctor, &bad) == RLMESH_ERR_TYPE);
  RLMeshValue* scalar = NULL;
  OK(rlmesh_value_int(1, &scalar));
  RLMeshRecorderAddOptions add = RLMESH_RECORDER_ADD_OPTIONS_INIT;
  add.config = scalar;
  RLMeshWorkloadRecord* workload = NULL;
  assert(rlmesh_recorder_add(rec, result, "m", "e", &add, &workload) == RLMESH_ERR_TYPE &&
         !workload);
  add.config = NULL;
  add._reserved[0] = 1;
  assert(rlmesh_recorder_add(rec, result, "m", "e", &add, &workload) == RLMESH_ERR_TYPE);
  RLMeshRecorderCaptureOptions cap = RLMESH_RECORDER_CAPTURE_OPTIONS_INIT;
  cap.config = scalar;
  RLMeshRunHooks* hooks = NULL;
  assert(rlmesh_recorder_capture(rec, "m", "e", &cap, &hooks) == RLMESH_ERR_TYPE && !hooks);
  cap.config = NULL;
  cap.cameras_len = 1;
  assert(rlmesh_recorder_capture(rec, "m", "e", &cap, &hooks) == RLMESH_ERR_INVALID_ARGUMENT);
  const char* null_string = NULL;
  cap.cameras = &null_string;
  assert(rlmesh_recorder_capture(rec, "m", "e", &cap, &hooks) == RLMESH_ERR_INVALID_ARGUMENT);
  cap.cameras_len = 0;
  cap.video_info_keys_len = 1;
  assert(rlmesh_recorder_capture(rec, "m", "e", &cap, &hooks) == RLMESH_ERR_INVALID_ARGUMENT);
  cap.video_info_keys = &null_string;
  assert(rlmesh_recorder_capture(rec, "m", "e", &cap, &hooks) == RLMESH_ERR_INVALID_ARGUMENT);
  cap.video_info_keys_len = 0;
  cap._reserved[0] = 1;
  assert(rlmesh_recorder_capture(rec, "m", "e", &cap, &hooks) == RLMESH_ERR_TYPE);
  future.known._reserved[0] = 1;
  assert(rlmesh_recorder_export(rec, path, &future.known, &bytes) == RLMESH_ERR_TYPE);
  size_t before = rlmesh_recorder_workloads_len(rec);
  assert(rlmesh_recorder_capture(rec, "m", "e", NULL, NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(rlmesh_recorder_workloads_len(rec) == before);
  RLMeshEpisodeRecordOptions ep = RLMESH_EPISODE_RECORD_OPTIONS_INIT;
  ep.media_len = 1;
  RLMeshEpisodeRecord* episode = NULL;
  assert(rlmesh_episode_record_new(0, NULL, 0, 0, false, false, &ep, &episode) ==
             RLMESH_ERR_INVALID_ARGUMENT &&
         !episode);
  const RLMeshMediaRef* null_media = NULL;
  ep.media = &null_media;
  assert(rlmesh_episode_record_new(0, NULL, 0, 0, false, false, &ep, &episode) ==
         RLMESH_ERR_INVALID_ARGUMENT);
  RLMeshWorkloadRecordOptions work = RLMESH_WORKLOAD_RECORD_OPTIONS_INIT;
  work.config = scalar;
  assert(rlmesh_workload_record_new("m", "e", "t", &work, &workload) == RLMESH_ERR_TYPE);
  work.config = NULL;
  work.episodes_len = 1;
  assert(rlmesh_workload_record_new("m", "e", "t", &work, &workload) ==
         RLMESH_ERR_INVALID_ARGUMENT);
  const RLMeshEpisodeRecord* null_episode = NULL;
  work.episodes = &null_episode;
  assert(rlmesh_workload_record_new("m", "e", "t", &work, &workload) ==
         RLMESH_ERR_INVALID_ARGUMENT);
  RLMeshMediaRef* media = NULL;
  assert(rlmesh_media_ref_new(NULL, "v", "p", "f", NULL, &media) == RLMESH_ERR_INVALID_ARGUMENT &&
         !media);
  assert(rlmesh_workload_record_new("m", "e", NULL, NULL, &workload) ==
         RLMESH_ERR_INVALID_ARGUMENT);
  rlmesh_value_free(scalar);
  rlmesh_run_result_free(result);
  rlmesh_recorder_free(rec);
  rlmesh_recorder_free(NULL);
  rlmesh_media_ref_free(NULL);
  rlmesh_episode_record_free(NULL);
  rlmesh_workload_record_free(NULL);
}

static void recorder_and_runs(const char* root) {
  State state = {0, 0, 1};
  RLMeshDiscrete* space = NULL;
  OK(rlmesh_discrete_new(2, NULL, &space));
  RLMeshEnvVtable ev = RLMESH_ENV_VTABLE_INIT;
  ev.reset = reset;
  ev.step = step;
  ev.render = render;
  RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
  attrs.observation_space = space;
  attrs.action_space = space;
  attrs.render_mode = "rgb_array";
  RLMeshEnv* env = NULL;
  OK(rlmesh_env_new(&ev, &state, &attrs, &env));
  rlmesh_space_free(space);
  RLMeshModelVtable mv = RLMESH_MODEL_VTABLE_INIT;
  mv.predict = predict;
  RLMeshModel* model = NULL;
  OK(rlmesh_model_new(&mv, NULL, NULL, &model));
  RLMeshView view = RLMESH_VIEW_INIT;
  RLMeshRunOptions ro = RLMESH_RUN_OPTIONS_INIT;
  ro.has_episodes = true;
  ro.episodes = 1;
  ro.view = &view;
  RLMeshRunResult* result = NULL;
  assert(rlmesh_model_run(model, rlmesh_env_target(env), &ro, &result) == RLMESH_ERR_TYPE &&
         !result);
  view.hold = RLMESH_VIEW_HOLD_SECONDS;
  view.hold_seconds = DBL_MAX;
  /* Conversion succeeds: the facade then rejects local Model.run viewing.
   * No viewer is opened and the saturated hold is never actually waited on. */
  assert(rlmesh_model_run(model, rlmesh_env_target(env), &ro, &result) == RLMESH_ERR_TYPE &&
         !result);
  const double invalid_holds[] = {INFINITY, NAN, -1.0};
  for (size_t i = 0; i < sizeof(invalid_holds) / sizeof(invalid_holds[0]); ++i) {
    view.hold_seconds = invalid_holds[i];
    assert(rlmesh_model_run(model, rlmesh_env_target(env), &ro, &result) ==
               RLMESH_ERR_INVALID_ARGUMENT &&
           !result);
  }
  view.hold = RLMESH_VIEW_HOLD_OFF;
  view.hold_seconds = 0.0;
  view.fps = 0;
  assert(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &ro, &result) ==
             RLMESH_ERR_INVALID_ARGUMENT &&
         !result);
  view.fps = 30;
  RLMeshSessionOptions so = RLMESH_SESSION_OPTIONS_INIT;
  so.view = &view;
  RLMeshSession* session = NULL;
  OK(rlmesh_session_new(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &so, &session));
  view.fps = 0; /* session copied its View */
  RLMeshSessionRunOptions sro = RLMESH_SESSION_RUN_OPTIONS_INIT;
  sro.has_episodes = true;
  sro.episodes = 0;
  OK(rlmesh_session_run(session, &sro, &result));
  rlmesh_run_result_free(result);
  rlmesh_session_free(session);
  ro.view = NULL;
  OK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &ro, &result));
  RLMeshRecorderOptions options = RLMESH_RECORDER_OPTIONS_INIT;
  options.result_set_id = "fixed";
  RLMeshRecorder* recorder = NULL;
  OK(rlmesh_recorder_new(&options, &recorder));
  RLMeshBytes bytes = {0};
  OK(rlmesh_recorder_result_set_id(recorder, &bytes));
  bytes_equal(bytes, "fixed");
  RLMeshWorkloadRecord *first = NULL, *snapshot = NULL;
  OK(rlmesh_recorder_add(recorder, result, "model", "env", NULL, &first));
  assert(rlmesh_workload_record_episodes_len(first) == 1 &&
         rlmesh_workload_record_mean_reward(first) == 2);
  OK(rlmesh_recorder_workloads_get(recorder, 0, &snapshot));
  assert(rlmesh_workload_record_equals(first, snapshot));
  rlmesh_workload_record_free(snapshot);
  rlmesh_run_result_free(result);
  RLMeshRunHooks* capture = NULL;
  OK(rlmesh_recorder_capture(recorder, "live", "env", NULL, &capture));
  ro.hooks = capture;
  OK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &ro, &result));
  assert(state.renders > 0);
  rlmesh_run_result_free(result);
  rlmesh_run_hooks_free(capture);
  assert(rlmesh_recorder_workloads_len(recorder) == 2);
  OK(rlmesh_recorder_workloads_get(recorder, 1, &snapshot));
  RLMeshEpisodeRecord* episode = NULL;
  OK(rlmesh_workload_record_episodes_get(snapshot, 0, &episode));
  assert(rlmesh_episode_record_media_len(episode) == 1); /* real AV1 media */
  RLMeshMediaRef* media = NULL;
  OK(rlmesh_episode_record_media_get(episode, 0, &media));
  int64_t count = 0;
  assert(rlmesh_media_ref_frame_count(media, &count) && count == 1);
  rlmesh_media_ref_free(media);
  rlmesh_episode_record_free(episode);
  rlmesh_workload_record_free(snapshot);
  char folder[1024], zip[1024], closed[1024];
  snprintf(folder, sizeof(folder), "%s/recorded", root);
  snprintf(zip, sizeof(zip), "%s/recorded.zip", root);
  snprintf(closed, sizeof(closed), "%s/closed", root);
  RLMeshRecorderExportOptions ex = RLMESH_RECORDER_EXPORT_OPTIONS_INIT;
  ex.recorded_at = "stamp";
  OK(rlmesh_recorder_export(recorder, folder, &ex, &bytes));
  bytes_equal(bytes, folder);
  /* Exporting to a recognized bundle replaces its owned media and manifest. */
  OK(rlmesh_recorder_export(recorder, folder, &ex, &bytes));
  bytes_equal(bytes, folder);
  char refused[1024];
  snprintf(refused, sizeof(refused), "%s/not-a-directory", root);
  FILE* refused_file = fopen(refused, "wb");
  assert(refused_file);
  fputs("keep", refused_file);
  fclose(refused_file);
  assert(rlmesh_recorder_export(recorder, refused, &ex, &bytes) == RLMESH_ERR_OS);
  refused_file = fopen(refused, "rb");
  assert(refused_file && fgetc(refused_file) == 'k');
  fclose(refused_file);
  OK(rlmesh_recorder_export(recorder, zip, &ex, &bytes));
  bytes_equal(bytes, zip);
  FILE* f = fopen(zip, "rb");
  assert(f);
  assert(fgetc(f) == 'P' && fgetc(f) == 'K');
  fclose(f);
  RLMeshRecorderToDictOptions td = RLMESH_RECORDER_TO_DICT_OPTIONS_INIT;
  td.recorded_at = "stamp";
  RLMeshValue* dict = NULL;
  OK(rlmesh_recorder_to_dict(recorder, &td, &dict));
  text_equal(field(dict, "schema"), RLMESH_RECORDER_SCHEMA);
  text_equal(field(dict, "recordedAt"), "stamp");
  rlmesh_value_free(dict);
  OK(rlmesh_recorder_close(recorder));
  OK(rlmesh_recorder_close(recorder));
  assert(rlmesh_recorder_export(recorder, closed, NULL, &bytes) == RLMESH_ERR_RUNTIME);
  f = fopen(closed, "rb");
  assert(!f); /* refused before writing */
  rlmesh_recorder_free(recorder);
  assert(rlmesh_workload_record_episodes_len(first) == 1);
  rlmesh_workload_record_free(first);

  /* Capture retains explicit contexts and works after recorder_free. */
  RLMeshRunContextVtable cv = RLMESH_RUN_CONTEXT_VTABLE_INIT;
  cv.contract = contract;
  cv.num_envs = lanes;
  cv.read = read_value;
  cv.frame_sources = sources;
  cv.free_user_data = context_free;
  RLMeshRunContext* context = NULL;
  OK(rlmesh_run_context_new(&cv, &state, &context));
  OK(rlmesh_recorder_new(NULL, &recorder));
  RLMeshRecorderCaptureOptions co = RLMESH_RECORDER_CAPTURE_OPTIONS_INIT;
  co.session = context;
  OK(rlmesh_recorder_capture(recorder, "manual", "env", &co, &capture));
  state.lanes = 2;
  assert(rlmesh_run_hooks_on_episode_start(capture, 0, NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(strstr(rlmesh_last_error_message(), "recorder captures frames from a single env"));
  rlmesh_run_hooks_free(capture);
  const char* empty = NULL;
  co.cameras = &empty;
  co.cameras_len = 0;
  OK(rlmesh_recorder_capture(recorder, "metrics", "env", &co, &capture));
  OK(rlmesh_run_hooks_on_run_start(capture, context));
  rlmesh_run_context_free(context);
  assert(state.context_frees == 0);
  rlmesh_recorder_free(recorder);
  OK(rlmesh_run_hooks_on_episode_start(capture, 0, NULL));
  RLMeshValue *observation = NULL, *info = NULL;
  OK(rlmesh_value_int(0, &observation));
  OK(rlmesh_value_dict(NULL, NULL, 0, &info));
  RLMeshStepEvent event = RLMESH_STEP_EVENT_INIT;
  event.observation = observation;
  event.action = observation;
  event.info = info;
  OK(rlmesh_run_hooks_on_step(capture, &event));
  rlmesh_value_free(observation);
  rlmesh_value_free(info);
  RLMeshEpisodeResult er = RLMESH_EPISODE_RESULT_INIT;
  er.steps = 1;
  er.reward = 7;
  OK(rlmesh_run_hooks_on_episode_end(capture, &er));
  RLMeshRunResultOptions rro = RLMESH_RUN_RESULT_OPTIONS_INIT;
  OK(rlmesh_run_result_new(&rro, &result));
  OK(rlmesh_run_hooks_on_run_end(capture, result));
  rlmesh_run_hooks_free(capture);
  assert(state.context_frees == 1);
  rlmesh_run_result_free(result);

  /* Missing-frame warnings must not abort even when the handler raises. */
  OK(rlmesh_recorder_new(NULL, &recorder));
  const char* missing = "absent";
  co.session = NULL;
  co.cameras = &missing;
  co.cameras_len = 1;
  OK(rlmesh_recorder_capture(recorder, "warning", "env", &co, &capture));
  int warnings = 0;
  OK(rlmesh_set_warning_handler(warning_error, &warnings, NULL));
  ro.hooks = capture;
  OK(rlmesh_run(RLMESH_RANDOM_SAMPLE, rlmesh_env_target(env), &ro, &result));
  assert(warnings == 1);
  OK(rlmesh_set_warning_handler(NULL, NULL, NULL));
  rlmesh_run_result_free(result);
  rlmesh_run_hooks_free(capture);
  rlmesh_recorder_free(recorder);
  /* With no staged media, closing still permits metric-only export. */
  OK(rlmesh_recorder_new(NULL, &recorder));
  OK(rlmesh_recorder_close(recorder));
  OK(rlmesh_recorder_export(recorder, closed, NULL, &bytes));
  bytes_equal(bytes, closed);
  rlmesh_recorder_free(recorder);
  rlmesh_model_free(model);
  rlmesh_env_free(env);
}

int main(int argc, char** argv) {
  assert(argc == 2 && rlmesh_abi_check());
  view_values();
  record_snapshots();
  option_boundaries(argv[1]);
  recorder_and_runs(argv[1]);
  puts("C View/Recorder OK");
  return 0;
}
