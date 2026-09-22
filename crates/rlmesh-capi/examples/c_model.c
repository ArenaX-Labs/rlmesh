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
static RlmeshValue* zero_action(const RlmeshSpaceSpec* action) {
  switch (rlmesh_space_type(action)) {
    case RLMESH_VALUE_BOX: {
      RlmeshDType dtype = rlmesh_space_dtype(action);
      size_t ndim = rlmesh_space_ndim(action);
      int64_t shape[16];
      if (ndim > 16) {
        rlmesh_callback_set_error("action rank exceeds 16", false);
        return NULL;
      }
      if (rlmesh_space_copy_shape(action, shape, ndim) != RLMESH_OK) {
        return NULL; /* last-error already set */
      }
      size_t numel = 1;
      for (size_t i = 0; i < ndim; ++i) {
        numel *= (size_t)shape[i];
      }
      size_t nbytes = numel * rlmesh_dtype_size(dtype);
      uint8_t* zeros = (uint8_t*)calloc(nbytes ? nbytes : 1, 1);
      RlmeshTensor tensor = {0};
      tensor.data = zeros;
      tensor.ndim = (int32_t)ndim;
      tensor.shape = shape;
      tensor.dtype = dtype;
      tensor.device_type = RLMESH_DEVICE_CPU;
      RlmeshValue* value = rlmesh_value_box(&tensor); /* copies */
      free(zeros);
      return value;
    }
    case RLMESH_VALUE_DISCRETE:
      return rlmesh_value_discrete(0);
    default:
      rlmesh_callback_set_error("c_model handles Box/Discrete only", false);
      return NULL;
  }
}

static int predict(void* user_data, const RlmeshObservation* obs, RlmeshValue** out_actions) {
  (void)user_data;
  const RlmeshSpaceSpec* action =
      obs->contract ? rlmesh_contract_action_space(obs->contract) : NULL;
  if (action == NULL) {
    rlmesh_callback_set_error("no action space on contract", false);
    return RLMESH_ERR_INVALID_VALUE;
  }
  for (size_t i = 0; i < obs->num_envs; ++i) {
    /* Each row names its episode and how many times this episode has been
     * predicted; a seeded episode also carries a per-predict sampling seed. */
    const RlmeshEpisode* episode = &obs->episodes[i];
    printf("episode %s: predict %llu", episode->id, (unsigned long long)episode->predict_index);
    if (episode->seeded) {
      printf(" seed %lld", (long long)episode->predict_seed);
    }
    if (obs->observations != NULL) {
      int64_t n = 0;
      if (rlmesh_value_as_discrete(obs->observations[i], &n) == RLMESH_OK) {
        printf(" obs %lld", (long long)n);
      }
    }
    printf("\n");
    out_actions[i] = zero_action(action);
    if (out_actions[i] == NULL) {
      return RLMESH_ERR_INVALID_VALUE; /* the capi frees rows already written */
    }
  }
  return RLMESH_OK;
}

static void on_episode_end(void* user_data, const char* env_id, const char* episode_id) {
  (void)user_data;
  (void)env_id;
  printf("episode end: %s\n", episode_id ? episode_id : "(all)");
}

static void on_close(void* user_data) {
  (void)user_data;
  printf("model closed\n");
}

int main(int argc, char** argv) {
  const char* address = argc > 1 ? argv[1] : "tcp://127.0.0.1:5555";
  RlmeshRunOptions options = {0};
  options.seeded = true; /* deterministic env resets */
  options.base_seed = 7;
  if (argc > 2) {
    options.max_episodes = strtoull(argv[2], NULL, 10);
  }

  RlmeshModelVtable vtable = {0};
  vtable.struct_size = sizeof(vtable);
  vtable.predict = predict;
  vtable.on_episode_end = on_episode_end;
  vtable.on_close = on_close;

  RlmeshModel* model = NULL;
  if (rlmesh_model_new(&vtable, NULL, &model) != RLMESH_OK) {
    fprintf(stderr, "failed to create model: %s\n", rlmesh_last_error_message());
    return 1;
  }

  printf("connecting to %s ...\n", address);
  RlmeshRunReport report = {0};
  RlmeshStatus status = rlmesh_model_run_local(model, address, &options, &report);
  if (status != RLMESH_OK) {
    /* Read the message BEFORE the next capi call on this thread -- including
     * rlmesh_model_free -- which invalidates it. */
    fprintf(stderr, "run failed: %s\n", rlmesh_last_error_message());
    rlmesh_model_free(model);
    return 1;
  }
  rlmesh_model_free(model);

  printf("run report: episodes=%lld steps=%lld reward=%.1f mean=%.1f\n",
         (long long)report.total_episodes, (long long)report.total_steps, report.total_reward,
         report.mean_reward);
  return 0;
}
