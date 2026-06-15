/* A pure-C model driving a remote RLMesh environment — rlmesh.h as valid C11,
 * the C ABI without the C++ wrapper.
 *
 *   zig cc -std=c11 -I<include> c_model.c -lrlmesh_capi -o c_model
 *   ./c_model tcp://127.0.0.1:50051 [episodes]
 */
#include <rlmesh.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int predict(void* user_data, const RlmeshObservation* obs, RlmeshBytes* out_action) {
  (void)user_data;
  const RlmeshSpaceSpec* action =
      obs->contract ? rlmesh_contract_action_space(obs->contract) : NULL;
  if (action == NULL) {
    rlmesh_callback_set_error("no action space on contract", false);
    return RLMESH_ERR_INVALID_VALUE;
  }

  RlmeshValue* value = NULL;
  switch (rlmesh_space_type(action)) {
    case 1: {
      RlmeshDType dtype = rlmesh_space_dtype(action);
      size_t ndim = rlmesh_space_ndim(action);
      int64_t shape[16];
      if (ndim > 16) {
        rlmesh_callback_set_error("action rank exceeds 16", false);
        return RLMESH_ERR_INVALID_VALUE;
      }
      if (rlmesh_space_copy_shape(action, shape, ndim) != RLMESH_OK) {
        return RLMESH_ERR_INVALID_VALUE; /* last-error already set */
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
      value = rlmesh_value_box(&tensor);
      free(zeros);
      break;
    }
    case 2:
      value = rlmesh_value_discrete(0);
      break;
    default:
      rlmesh_callback_set_error("c_model handles Box/Discrete only", false);
      return RLMESH_ERR_INVALID_VALUE;
  }
  if (value == NULL) {
    return RLMESH_ERR_INVALID_VALUE; /* rlmesh_value_box set the error */
  }

  const RlmeshValue* values[1] = {value};
  RlmeshStatus status = rlmesh_encode_batch(values, 1, action, out_action);
  rlmesh_value_free(value);
  return status;
}

int main(int argc, char** argv) {
  const char* address = argc > 1 ? argv[1] : "tcp://127.0.0.1:50051";

  RlmeshModelVtable vtable = {0};
  vtable.struct_size = sizeof(vtable);
  vtable.predict = predict;

  RlmeshModel* model = NULL;
  if (rlmesh_model_new(&vtable, NULL, &model) != RLMESH_OK) {
    fprintf(stderr, "failed to create model: %s\n", rlmesh_last_error_message());
    return 1;
  }

  printf("connecting to %s ...\n", address);
  RlmeshStatus status;
  if (argc > 2) {
    status = rlmesh_model_run_local_for_episodes(model, address, strtoull(argv[2], NULL, 10));
  } else {
    status = rlmesh_model_run_local(model, address);
  }
  rlmesh_model_free(model);

  if (status != RLMESH_OK) {
    fprintf(stderr, "run failed: %s\n", rlmesh_last_error_message());
    return 1;
  }
  return 0;
}
