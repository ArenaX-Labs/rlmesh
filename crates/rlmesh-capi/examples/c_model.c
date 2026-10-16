/* A pure-C model driving a remote RLMesh environment — rlmesh.h as valid C11,
 * the C ABI without the C++ wrapper.
 *
 *   zig cc -std=c11 -I<include> c_model.c -lrlmesh_capi -o c_model
 *   ./c_model tcp://127.0.0.1:5555 [episodes]
 */
#include <rlmesh.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A zero action shaped like the route's action space (Box or Discrete). */
static RLMeshValue* zero_action(const RLMeshSpaceSpec* action) {
  switch (rlmesh_space_spec_kind(action)) {
    case RLMESH_SPACE_KIND_BOX: {
      RLMeshDType dtype = rlmesh_space_spec_dtype(action);
      size_t ndim = 0;
      if (rlmesh_space_spec_shape(action, NULL, 0, &ndim) != RLMESH_OK) return NULL;
      int64_t shape[16];
      size_t dims[16];
      if (ndim > 16) {
        rlmesh_callback_set_error("action rank exceeds 16", false);
        return NULL;
      }
      if (rlmesh_space_spec_shape(action, shape, ndim, &ndim) != RLMESH_OK) {
        return NULL; /* last-error already set */
      }
      size_t numel = 1;
      for (size_t i = 0; i < ndim; ++i) {
        dims[i] = (size_t)shape[i];
        numel *= dims[i];
      }
      size_t nbytes = numel * rlmesh_ext_dtype_size(dtype);
      uint8_t* zeros = (uint8_t*)calloc(nbytes ? nbytes : 1, 1);
      RLMeshTensor* tensor = NULL;
      RLMeshValue* value = NULL;
      RLMeshStatus status = rlmesh_tensor_new(zeros, nbytes, dims, ndim, dtype, &tensor);
      if (status == RLMESH_OK) status = rlmesh_value_tensor(tensor, &value);
      if (status != RLMESH_OK) {
        free(zeros);
        return NULL;
      }
      rlmesh_tensor_free(tensor);
      free(zeros);
      return value;
    }
    case RLMESH_SPACE_KIND_DISCRETE: {
      RLMeshValue* value = NULL;
      rlmesh_value_int(0, &value);
      return value;
    }
    default:
      rlmesh_callback_set_error("c_model handles Box/Discrete only", false);
      return NULL;
  }
}

static RLMeshStatus predict(void* user_data, const RLMeshValue* observation,
                            const RLMeshPredictContext* context, RLMeshValue** output) {
  (void)observation;
  printf("episode %s: predict %llu", context->episode_id,
         (unsigned long long)context->predict_index);
  if (context->has_predict_seed) printf(" seed %lld", (long long)context->predict_seed);
  puts("");
  *output = zero_action((const RLMeshSpaceSpec*)user_data);
  return *output ? RLMESH_OK : RLMESH_ERR_MODEL;
}
int main(int argc, char** argv) {
  const char* address = argc > 1 ? argv[1] : "tcp://127.0.0.1:5555";
  const uint64_t count = argc > 2 ? strtoull(argv[2], NULL, 10) : 3;
  RLMeshRemoteEnvOptions connection = RLMESH_REMOTE_ENV_OPTIONS_INIT;
  connection.address = address;
  RLMeshRemoteEnv* env = NULL;
  if (rlmesh_remote_env_new(&connection, &env) != RLMESH_OK) goto fail;
  RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
  vt.predict = predict;
  vt.context_corners = RLMESH_CORNER_PREDICT;
  RLMeshModel* model = NULL;
  if (rlmesh_model_new(&vt, (void*)rlmesh_remote_env_action_space_spec(env), NULL, &model) !=
      RLMESH_OK)
    goto fail;
  RLMeshRunOptions options = RLMESH_RUN_OPTIONS_INIT;
  options.has_episodes = true;
  options.episodes = count;
  int64_t* seeds = (int64_t*)calloc(count ? count : 1, sizeof(int64_t));
  for (uint64_t i = 0; i < count; ++i) seeds[i] = (int64_t)(7 + i);
  options.seeds = seeds;
  options.seeds_len = (size_t)count;
  RLMeshRunResult* result = NULL;
  RLMeshStatus status = rlmesh_model_run(model, rlmesh_remote_env_target(env), &options, &result);
  free(seeds);
  if (status != RLMESH_OK) {
    fprintf(stderr, "%s\n", rlmesh_last_error_message());
    rlmesh_model_free(model);
    rlmesh_remote_env_free(env);
    return 1;
  }
  printf("run report: episodes=%zu steps=%lld mean=%.1f\n", rlmesh_run_result_num_episodes(result),
         (long long)rlmesh_run_result_total_steps(result), rlmesh_run_result_mean_reward(result));
  rlmesh_run_result_free(result);
  rlmesh_model_free(model);
  rlmesh_remote_env_free(env);
  return 0;
fail:
  fprintf(stderr, "%s\n", rlmesh_last_error_message());
  return 1;
}
