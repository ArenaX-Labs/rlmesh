/* ABI 5: versioned CPU DLPack and success-only ownership transfer. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "rlmesh.h"

_Static_assert(RLMESH_TO_DLPACK_OPTIONS_MIN_SIZE == sizeof(size_t) + 16, "no option padding");
_Static_assert(offsetof(RLMeshToDlpackOptions, copy) == sizeof(size_t), "copy offset");
_Static_assert(offsetof(RLMeshToDlpackOptions, has_dl_device) == sizeof(size_t) + 1,
               "device flag offset");
_Static_assert(offsetof(RLMeshToDlpackOptions, _reserved0) == sizeof(size_t) + 2,
               "reserved offset");
_Static_assert(offsetof(RLMeshToDlpackOptions, dl_device_type) == sizeof(size_t) + 4,
               "device type offset");
_Static_assert(offsetof(RLMeshToDlpackOptions, dl_device_id) == sizeof(size_t) + 8,
               "device id offset");
_Static_assert(offsetof(RLMeshToDlpackOptions, _reserved1) == sizeof(size_t) + 12,
               "reserved tail offset");

typedef struct Producer {
  struct DLManagedTensorVersioned managed;
  int64_t shape[2];
  int64_t strides[2];
  unsigned char data[6];
  int* deletes;
} Producer;

static void delete_producer(struct DLManagedTensorVersioned* managed) {
  Producer* producer = (Producer*)managed;
  assert(managed->manager_ctx == producer);
  ++*producer->deletes;
  memset(producer->data, 0, sizeof(producer->data));
  free(producer);
}

static Producer* producer_new(int* deletes) {
  Producer* producer = (Producer*)calloc(1, sizeof(Producer));
  assert(producer);
  producer->deletes = deletes;
  producer->shape[0] = 2;
  producer->shape[1] = 3;
  producer->strides[0] = 3;
  producer->strides[1] = 1;
  for (size_t i = 0; i < 6; ++i) producer->data[i] = (unsigned char)(i + 1);
  producer->managed.version.major = 1;
  producer->managed.manager_ctx = producer;
  producer->managed.deleter = delete_producer;
  producer->managed.dl_tensor.data = producer->data;
  producer->managed.dl_tensor.device.device_type = kDLCPU;
  producer->managed.dl_tensor.device.device_id = 0;
  producer->managed.dl_tensor.dtype.code = kDLUInt;
  producer->managed.dl_tensor.dtype.bits = 8;
  producer->managed.dl_tensor.dtype.lanes = 1;
  producer->managed.dl_tensor.ndim = 2;
  producer->managed.dl_tensor.shape = producer->shape;
  producer->managed.dl_tensor.strides = producer->strides;
  return producer;
}

static void assert_bytes(const RLMeshTensor* tensor, const unsigned char* expected, size_t size) {
  RLMeshBytes bytes = {0};
  assert(rlmesh_tensor_tobytes(tensor, &bytes) == RLMESH_OK);
  assert(bytes.len == size);
  if (size) assert(memcmp(bytes.data, expected, size) == 0);
  rlmesh_bytes_free(bytes);
}

static void imports_copy_and_consume_once(void) {
  const unsigned char expected[] = {1, 2, 3, 4, 5, 6};
  for (int layout = 0; layout < 4; ++layout) {
    int deletes = 0;
    Producer* producer = producer_new(&deletes);
    if (layout == 1) {
      producer->strides[1] = -1;
      producer->managed.dl_tensor.byte_offset = 2;
    } else if (layout == 2) {
      producer->strides[0] = 0;
    } else if (layout == 3) {
      producer->managed.dl_tensor.strides = NULL;
    }
    RLMeshTensor* imported = NULL;
    assert(rlmesh_tensor_from_dlpack(&producer->managed, &imported) == RLMESH_OK);
    assert(deletes == 1); /* producer, its data and its shape are already gone */
    if (layout == 1) {
      const unsigned char reversed[] = {3, 2, 1, 6, 5, 4};
      assert_bytes(imported, reversed, sizeof(reversed));
    } else if (layout == 2) {
      const unsigned char broadcast[] = {1, 2, 3, 1, 2, 3};
      assert_bytes(imported, broadcast, sizeof(broadcast));
    } else {
      assert_bytes(imported, expected, sizeof(expected));
    }
    rlmesh_tensor_free(imported);
    assert(deletes == 1);
  }
}

static void failed_imports_keep_producer(void) {
  for (int failure = 0; failure < 9; ++failure) {
    int deletes = 0;
    Producer* producer = producer_new(&deletes);
    switch (failure) {
      case 0:
        producer->managed.version.major = 2;
        break;
      case 1:
        producer->managed.dl_tensor.device.device_type = kDLCUDA;
        break;
      case 2:
        producer->managed.dl_tensor.dtype.lanes = 2;
        break;
      case 3:
        producer->managed.dl_tensor.ndim = -1;
        break;
      case 4:
        producer->managed.dl_tensor.shape = NULL;
        break;
      case 5:
        producer->managed.dl_tensor.data = NULL;
        break;
      case 6:
        producer->shape[0] = -1;
        break;
      case 7:
        producer->strides[0] = INT64_MAX;
        break;
      case 8:
        producer->managed.dl_tensor.byte_offset = UINT64_MAX;
        break;
    }
    RLMeshTensor* imported = (RLMeshTensor*)producer;
    assert(rlmesh_tensor_from_dlpack(&producer->managed, &imported) == RLMESH_ERR_INVALID_ARGUMENT);
    assert(imported == NULL && deletes == 0);
    /* Failed calls can repeat without consuming the producer. */
    assert(rlmesh_tensor_from_dlpack(&producer->managed, &imported) == RLMESH_ERR_INVALID_ARGUMENT);
    assert(deletes == 0);
    producer->managed.deleter(&producer->managed);
    assert(deletes == 1);
  }
  int deletes = 0;
  Producer* producer = producer_new(&deletes);
  assert(rlmesh_tensor_from_dlpack(&producer->managed, NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(deletes == 0); /* invalid out must be checked before consuming obj */
  RLMeshTensor* imported = NULL;
  assert(rlmesh_tensor_from_dlpack(&producer->managed, &imported) == RLMESH_OK);
  assert(deletes == 1);
  rlmesh_tensor_free(imported);
  assert(deletes == 1);
  assert(rlmesh_tensor_from_dlpack(NULL, &imported) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(imported == NULL);
}

static void exports_own_storage(void) {
  unsigned char data[] = {1, 2, 3, 4};
  const size_t shape[] = {2, 2};
  RLMeshTensor* tensor = NULL;
  assert(rlmesh_tensor_new(data, sizeof(data), shape, 2, RLMESH_DTYPE_UINT8, &tensor) == RLMESH_OK);
  int32_t device_type = -1, device_id = -1;
  assert(rlmesh_tensor_dlpack_device(tensor, &device_type, &device_id) == RLMESH_OK);
  assert(device_type == kDLCPU && device_id == 0);
  assert(rlmesh_tensor_dlpack_device(tensor, NULL, &device_id) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(rlmesh_tensor_dlpack_device(tensor, &device_type, NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(rlmesh_tensor_dlpack_device(NULL, &device_type, &device_id) ==
         RLMESH_ERR_INVALID_ARGUMENT);
  struct DLManagedTensorVersioned *shared = NULL, *copy = NULL;
  RLMeshToDlpackOptions options = RLMESH_TO_DLPACK_OPTIONS_INIT;
  assert(rlmesh_tensor_to_dlpack(tensor, NULL, &shared) == RLMESH_OK);
  assert(shared->version.major == 1 && shared->version.minor == 0);
  assert(shared->flags == DLPACK_FLAG_BITMASK_READ_ONLY);
  assert(shared->dl_tensor.dtype.code == kDLUInt && shared->dl_tensor.dtype.bits == 8);
  assert(shared->dl_tensor.dtype.lanes == 1);
  assert(shared->dl_tensor.ndim == 2 && shared->dl_tensor.shape[0] == 2);
  assert(shared->dl_tensor.strides[0] == 2 && shared->dl_tensor.strides[1] == 1);
  const void* buffer = NULL;
  size_t buffer_len = 0;
  assert(rlmesh_tensor_buffer(tensor, &buffer, &buffer_len) == RLMESH_OK);
  assert(shared->dl_tensor.data == buffer && buffer_len == 4);
  options.copy = true;
  options.has_dl_device = true;
  options.dl_device_type = kDLCPU;
  options.dl_device_id = 0;
  assert(rlmesh_tensor_to_dlpack(tensor, &options, &copy) == RLMESH_OK);
  assert(copy->flags == DLPACK_FLAG_BITMASK_IS_COPIED);
  assert(copy->dl_tensor.data != buffer);
  ((unsigned char*)copy->dl_tensor.data)[0] = 42;
  assert_bytes(tensor, data, sizeof(data)); /* writable copy is isolated */
  rlmesh_tensor_free(tensor);
  memset(data, 0, sizeof(data));
  assert(memcmp(shared->dl_tensor.data, "\1\2\3\4", 4) == 0);
  assert(shared->dl_tensor.shape[1] == 2); /* metadata also survives */
  assert(((unsigned char*)copy->dl_tensor.data)[0] == 42);
  shared->deleter(shared); /* release exactly once through its own deleter */
  RLMeshTensor* imported = NULL;
  assert(rlmesh_tensor_from_dlpack(copy, &imported) == RLMESH_OK); /* consumes copy */
  const unsigned char expected[] = {42, 2, 3, 4};
  assert_bytes(imported, expected, sizeof(expected));
  rlmesh_tensor_free(imported);
}

static void export_options_obey_sizing(void) {
  const size_t shape[] = {0};
  RLMeshTensor* tensor = NULL;
  assert(rlmesh_tensor_new(NULL, 0, shape, 1, RLMESH_DTYPE_BOOL, &tensor) == RLMESH_OK);
  struct DLManagedTensorVersioned dummy = {0};
  struct DLManagedTensorVersioned* out = &dummy;
  RLMeshToDlpackOptions options = {0};
  assert(rlmesh_tensor_to_dlpack(tensor, &options, &out) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(out == NULL);
  options.struct_size = RLMESH_TO_DLPACK_OPTIONS_MIN_SIZE - 1;
  assert(rlmesh_tensor_to_dlpack(tensor, &options, &out) == RLMESH_ERR_INVALID_ARGUMENT);
  RLMeshToDlpackOptions init = RLMESH_TO_DLPACK_OPTIONS_INIT;
  options = init;
  assert(!init.copy && !init.has_dl_device && !init._reserved0 && !init._reserved1);
  for (int failure = 0; failure < 4; ++failure) {
    options = init;
    switch (failure) {
      case 0:
        options._reserved0 = 1;
        break;
      case 1:
        options._reserved1 = 1;
        break;
      case 2:
        options.has_dl_device = true;
        options.dl_device_type = kDLCUDA;
        break;
      case 3:
        options.has_dl_device = true;
        options.dl_device_type = kDLCPU;
        options.dl_device_id = 1;
        break;
    }
    out = &dummy;
    assert(rlmesh_tensor_to_dlpack(tensor, &options, &out) ==
           (failure < 2 ? RLMESH_ERR_TYPE : RLMESH_ERR_INVALID_ARGUMENT));
    assert(out == NULL);
  }
  struct Future {
    RLMeshToDlpackOptions known;
    unsigned char tail[8];
  } future = {RLMESH_TO_DLPACK_OPTIONS_INIT, {0}};
  future.known.struct_size = sizeof(future);
  assert(rlmesh_tensor_to_dlpack(tensor, &future.known, &out) == RLMESH_OK);
  out->deleter(out);
  for (size_t i = 0; i < sizeof(future.tail); ++i) {
    future.tail[i] = 1;
    out = &dummy;
    assert(rlmesh_tensor_to_dlpack(tensor, &future.known, &out) == RLMESH_ERR_TYPE);
    assert(out == NULL);
    future.tail[i] = 0;
  }
  assert(rlmesh_tensor_to_dlpack(NULL, NULL, &out) == RLMESH_ERR_INVALID_ARGUMENT);
  assert(out == NULL);
  assert(rlmesh_tensor_to_dlpack(tensor, NULL, NULL) == RLMESH_ERR_INVALID_ARGUMENT);
  rlmesh_tensor_free(tensor);
}

int main(void) {
  assert(rlmesh_abi_check());
  imports_copy_and_consume_once();
  failed_imports_keep_producer();
  exports_own_storage();
  export_options_obey_sizing();
  return 0;
}
