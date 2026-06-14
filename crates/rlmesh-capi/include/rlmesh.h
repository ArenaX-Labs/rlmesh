/* RLMesh C ABI — experimental, v1 model path (a C/C++ model driving a remote
 * environment). Hand-authored; cbindgen-verifiable (a header-drift CI gate is a
 * follow-up). */
#ifndef RLMESH_H
#define RLMESH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version (must match the crate version; rlmesh_abi_version_* return these
 * at runtime so a plugin can refuse a too-old host). */
#define RLMESH_ABI_VERSION_MAJOR 0
#define RLMESH_ABI_VERSION_MINOR 1
#define RLMESH_ABI_VERSION_PATCH 0

uint32_t rlmesh_abi_version_major(void);
uint32_t rlmesh_abi_version_minor(void);
uint32_t rlmesh_abi_version_patch(void);

/* ---- status + errors ---------------------------------------------------- */

typedef enum RlmeshStatus {
  RLMESH_OK = 0,
  RLMESH_ERR_INVALID_ARGUMENT = 1,
  RLMESH_ERR_INVALID_VALUE = 2,
  RLMESH_ERR_ENVIRONMENT = 3,
  RLMESH_ERR_MODEL = 4,
  RLMESH_ERR_TRANSPORT = 5,
  RLMESH_ERR_TIMEOUT = 6,
  RLMESH_ERR_PANIC = 7,
  RLMESH_ERR_INTERNAL = 99
} RlmeshStatus;

/* Most recent failing call's message on THIS thread (valid until the next
 * RLMesh call on this thread; NULL if none). Read only after a nonzero status. */
const char* rlmesh_last_error_message(void);
int rlmesh_last_error_is_recoverable(void);

/* ---- dtype + tensor ----------------------------------------------------- */

/* DLPack (code, bits, lanes): code is DLDataTypeCode (int=0, uint=1, float=2,
 * bfloat=4, bool=6); lanes is always 1. */
typedef struct RlmeshDType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} RlmeshDType;

#ifdef __cplusplus
#define RLMESH_DTYPE_INIT(c, b, l) \
  RlmeshDType { (uint8_t)(c), (uint8_t)(b), (uint16_t)(l) }
#else
#define RLMESH_DTYPE_INIT(c, b, l) \
  (RlmeshDType) { (uint8_t)(c), (uint8_t)(b), (uint16_t)(l) }
#endif
#define RLMESH_F32 RLMESH_DTYPE_INIT(2, 32, 1)
#define RLMESH_F64 RLMESH_DTYPE_INIT(2, 64, 1)
#define RLMESH_I32 RLMESH_DTYPE_INIT(0, 32, 1)
#define RLMESH_I64 RLMESH_DTYPE_INIT(0, 64, 1)
#define RLMESH_U8 RLMESH_DTYPE_INIT(1, 8, 1)
#define RLMESH_BOOL RLMESH_DTYPE_INIT(6, 8, 1)

#define RLMESH_DEVICE_CPU 1
#define RLMESH_TENSOR_FLAG_READ_ONLY ((uint64_t)1)

/* Element byte size, or 0 if unsupported (or lanes != 1). */
size_t rlmesh_dtype_size(RlmeshDType dtype);

/* A DLPack-shaped tensor view. `strides` is in element counts (NULL = row-major
 * contiguous); `data` points at element 0. A tensor returned by value is a
 * borrowed view (`deleter == NULL`) valid only while its source value lives. */
typedef struct RlmeshTensor {
  void* data;
  int32_t ndim;
  const int64_t* shape;
  const int64_t* strides;
  RlmeshDType dtype;
  int32_t device_type;
  int32_t device_id;
  uint64_t flags;
  void* manager_ctx;
  void (*deleter)(struct RlmeshTensor* self);
} RlmeshTensor;

/* Release a tensor's backing resource (`manager_ctx` only; never frees `self`).
 * A no-op for a borrowed view. */
void rlmesh_tensor_release(RlmeshTensor* tensor);

/* ---- values (a SpaceValue projection) ----------------------------------- */

typedef struct RlmeshValue RlmeshValue;

typedef enum RlmeshValueKind {
  RLMESH_VALUE_BOX = 1,
  RLMESH_VALUE_DISCRETE = 2,
  RLMESH_VALUE_MULTI_BINARY = 3,
  RLMESH_VALUE_MULTI_DISCRETE = 4,
  RLMESH_VALUE_TEXT = 5,
  RLMESH_VALUE_DICT = 10,
  RLMESH_VALUE_TUPLE = 11
} RlmeshValueKind;

RlmeshValueKind rlmesh_value_kind(const RlmeshValue* value);

/* Box: borrowed tensor view (valid while `value` lives) / copy-construct. */
RlmeshStatus rlmesh_value_as_tensor(const RlmeshValue* value, RlmeshTensor* out);
RlmeshValue* rlmesh_value_box(const RlmeshTensor* tensor); /* contiguous only */

/* Discrete. */
RlmeshValue* rlmesh_value_discrete(int64_t value);
RlmeshStatus rlmesh_value_as_discrete(const RlmeshValue* value, int64_t* out);

/* Text: `len` UTF-8 bytes (not NUL-terminated). */
RlmeshValue* rlmesh_value_text(const char* data, size_t len);
RlmeshStatus rlmesh_value_as_text(const RlmeshValue* value, const char** out_ptr, size_t* out_len);

/* MultiBinary / MultiDiscrete: constructed from / copied into a caller buffer. */
RlmeshValue* rlmesh_value_multi_discrete(const int64_t* data, size_t n);
RlmeshValue* rlmesh_value_multi_binary(const uint8_t* data, size_t n);
size_t rlmesh_value_array_len(const RlmeshValue* value);
RlmeshStatus rlmesh_value_copy_multi_discrete(const RlmeshValue* value, int64_t* out, size_t cap);
RlmeshStatus rlmesh_value_copy_multi_binary(const RlmeshValue* value, uint8_t* out, size_t cap);

/* Dict / Tuple: borrowed children (valid while `value` lives); the
 * constructors take ownership of (and free) each child value. */
size_t rlmesh_value_len(const RlmeshValue* value);
const RlmeshValue* rlmesh_value_tuple_get(const RlmeshValue* value, size_t index);
const RlmeshValue* rlmesh_value_dict_get(const RlmeshValue* value, const char* key);
RlmeshValue* rlmesh_value_tuple(RlmeshValue* const* children, size_t n);
RlmeshValue* rlmesh_value_dict(const char* const* keys, RlmeshValue* const* values, size_t n);

/* Free an owned value (from a constructor or rlmesh_decode_batch). Not for a
 * borrowed child (*_get) or a tensor view. */
void rlmesh_value_free(RlmeshValue* value);

/* ---- spaces + contract (read-only; builders are env-side, not yet here) -- */

typedef struct RlmeshSpaceSpec RlmeshSpaceSpec;
typedef struct RlmeshContract RlmeshContract;

const RlmeshSpaceSpec* rlmesh_contract_observation_space(const RlmeshContract* contract);
const RlmeshSpaceSpec* rlmesh_contract_action_space(const RlmeshContract* contract);
uint32_t rlmesh_contract_num_envs(const RlmeshContract* contract);

/* Space introspection (so a model can size an action to the action space).
 * `rlmesh_space_type` returns a RlmeshValueKind discriminant (0 if unknown). */
int32_t rlmesh_space_type(const RlmeshSpaceSpec* spec);
RlmeshDType rlmesh_space_dtype(const RlmeshSpaceSpec* spec);
size_t rlmesh_space_ndim(const RlmeshSpaceSpec* spec);
RlmeshStatus rlmesh_space_copy_shape(const RlmeshSpaceSpec* spec, int64_t* out, size_t cap);

/* ---- codec -------------------------------------------------------------- */

/* An owned buffer produced by the capi (its `cap` lets the capi reclaim the
 * allocation). Free with rlmesh_bytes_free. A buffer handed BACK to the capi —
 * e.g. a predict callback's `out_action` — MUST be one produced by
 * rlmesh_encode_batch; the capi frees it with its own allocator (never malloc). */
typedef struct RlmeshBytes {
  uint8_t* data;
  size_t len;
  size_t cap;
} RlmeshBytes;

void rlmesh_bytes_free(RlmeshBytes bytes);

/* Decode a payload against `spec` into `*out_values` (length `*out_n`, one per
 * sub-env); free with rlmesh_values_free. NULL/empty payload -> zero values. */
RlmeshStatus rlmesh_decode_batch(const uint8_t* data, size_t len, const RlmeshSpaceSpec* spec,
                                 RlmeshValue*** out_values, size_t* out_n);

/* Encode `n` values against `spec` into `out` (validated with contains). */
RlmeshStatus rlmesh_encode_batch(const RlmeshValue* const* values, size_t n,
                                 const RlmeshSpaceSpec* spec, RlmeshBytes* out);

void rlmesh_values_free(RlmeshValue** values, size_t n);

/* ---- model -------------------------------------------------------------- */

typedef struct RlmeshRouteSlot {
  const char* episode_id;
  int32_t env_index;
  int64_t step;
  bool reset;
} RlmeshRouteSlot;

typedef struct RlmeshObservation {
  const uint8_t* observation;
  size_t observation_len;
  const RlmeshContract* contract;
  uint32_t num_envs;
  const char* session_id;
  const char* route_id;
  const char* request_id;
  const RlmeshRouteSlot* slots;
  size_t num_slots;
} RlmeshObservation;

/* Set this call's error message + recoverability before returning nonzero. */
void rlmesh_callback_set_error(const char* message, bool recoverable);

/* The model callback vtable. Set struct_size = sizeof(RlmeshModelVtable); fields
 * beyond that are ignored (append-only). `predict` is required. Callbacks run on
 * a worker thread, so `user_data` must be thread-migration-safe. */
typedef struct RlmeshModelVtable {
  size_t struct_size;
  RlmeshStatus (*predict)(void* user_data, const RlmeshObservation* obs, RlmeshBytes* out_action);
  void (*on_lane_reset)(void* user_data, const char* episode_id, int32_t env_index);
  void (*on_episode_end)(void* user_data, const char* episode_id, int32_t env_index);
  void (*on_close)(void* user_data);
} RlmeshModelVtable;

typedef struct RlmeshModel RlmeshModel;

RlmeshStatus rlmesh_model_new(const RlmeshModelVtable* vtable, void* user_data, RlmeshModel** out);
RlmeshStatus rlmesh_model_run_local(RlmeshModel* model, const char* env_address, const char* token);
RlmeshStatus rlmesh_model_run_local_for_episodes(RlmeshModel* model, const char* env_address,
                                                 const char* token, uint64_t max_episodes);
void rlmesh_model_free(RlmeshModel* model);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RLMESH_H */
