/* Full ABI-5 input minima, unknown tails and unchanged extended out fields. */
#include <rlmesh.h>
#include <stdio.h>
static int failures;
#define CHECK(x)                                              \
  do {                                                        \
    if (!(x)) {                                               \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      ++failures;                                             \
    }                                                         \
  } while (0)
static int predict(void* u, const RLMeshExtLegacyObservation* o, RLMeshValue** out) {
  (void)u;
  (void)o;
  (void)out;
  return RLMESH_OK;
}
int main(void) {
  CHECK(rlmesh_abi_check() && RLMESH_ABI_VERSION == 5);
  RLMeshExtLegacyModelVtable zero = {0};
  RLMeshExtLegacyModel* model = NULL;
  CHECK(rlmesh_ext_legacy_model_new(&zero, NULL, &model) == RLMESH_ERR_INVALID_ARGUMENT &&
        model == NULL);
  struct VtableTail {
    RLMeshExtLegacyModelVtable vt;
    size_t tail;
  } vt = {RLMESH_EXT_LEGACY_MODEL_VTABLE_INIT, 0};
  vt.vt.struct_size = sizeof(vt);
  vt.vt.predict = predict;
  vt.tail = 1;
  CHECK(rlmesh_ext_legacy_model_new(&vt.vt, NULL, &model) == RLMESH_ERR_TYPE && model == NULL);
  vt.tail = 0;
  CHECK(rlmesh_ext_legacy_model_new(&vt.vt, NULL, &model) == RLMESH_OK);
  RLMeshExtLegacyRunOptions run = {0};
  CHECK(rlmesh_ext_legacy_model_run_local(model, "tcp://127.0.0.1:1", &run, NULL) ==
        RLMESH_ERR_INVALID_ARGUMENT);
  struct RunTail {
    RLMeshExtLegacyRunOptions options;
    size_t tail;
  } rt = {RLMESH_EXT_LEGACY_RUN_OPTIONS_INIT, 1};
  rt.options.struct_size = sizeof(rt);
  CHECK(rlmesh_ext_legacy_model_run_local(model, "tcp://127.0.0.1:1", &rt.options, NULL) ==
        RLMESH_ERR_TYPE);
  RLMeshExtLegacyServeOptions serve = {0};
  CHECK(rlmesh_ext_legacy_model_serve(model, "tcp://127.0.0.1:0", &serve) ==
        RLMESH_ERR_INVALID_ARGUMENT);
  rlmesh_ext_legacy_model_free(model);
  RLMeshEpisodeResult episode = RLMESH_EPISODE_RESULT_INIT;
  episode.steps = 4;
  episode.reward = 2;
  const RLMeshEpisodeResult* episodes[] = {&episode};
  RLMeshRunResultOptions options = RLMESH_RUN_RESULT_OPTIONS_INIT;
  options.episodes = episodes;
  options.episodes_len = 1;
  RLMeshRunResult* result = NULL;
  CHECK(rlmesh_run_result_new(&options, &result) == RLMESH_OK);
  struct EpisodeTail {
    RLMeshEpisodeResult episode;
    size_t tail;
  } filled = {RLMESH_EPISODE_RESULT_INIT, 0};
  filled.episode.struct_size = sizeof(filled);
  CHECK(rlmesh_run_result_episodes_get(result, 0, &filled.episode) == RLMESH_OK);
  CHECK(filled.episode.struct_size == sizeof(filled) && filled.episode.steps == 4 &&
        filled.tail == 0);
  filled.tail = 1;
  filled.episode.steps = 77;
  CHECK(rlmesh_run_result_episodes_get(result, 0, &filled.episode) == RLMESH_ERR_TYPE &&
        filled.episode.steps == 77);
  filled.tail = 0;
  filled.episode.struct_size = RLMESH_EPISODE_RESULT_MIN_SIZE - 1;
  CHECK(rlmesh_run_result_episodes_get(result, 0, &filled.episode) == RLMESH_ERR_INVALID_ARGUMENT &&
        filled.episode.steps == 77);
  rlmesh_run_result_free(result);
  if (!failures) puts("C ABI 5 sizing OK");
  return failures ? 1 : 0;
}
