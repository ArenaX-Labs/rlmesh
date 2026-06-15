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

/* Binary ABI generation — bumped ONLY on a binary-incompatible change (a
 * repr(C) layout/enum-discriminant change, an extern "C" signature retype, or a
 * symbol removal). Decoupled from the package semver below, which can't express
 * an ABI break. Appending a struct_size-guarded vtable field is NOT a break. */
#define RLMESH_ABI_VERSION 1

uint32_t rlmesh_abi_version(void);

/* Nonzero when the linked library's ABI generation matches the one this header
 * was compiled against. SONAME-linked consumers are already gated by the loader
 * (librlmesh_capi.so.N); this is for dlopen / raw-path consumers that bypass it. */
static inline int rlmesh_abi_check(void) { return RLMESH_ABI_VERSION == rlmesh_abi_version(); }

/* Package (marketing) semver — informational only. Do NOT gate ABI
 * compatibility on these; use RLMESH_ABI_VERSION / rlmesh_abi_check(). */
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

/* ---- adapters (experimental) -------------------------------------------- */

/* Resolve the env's tags (env_tags_json; see rlmesh_contract_adapter_tags_json)
 * against this model's spec (model_spec_json) into an opaque plan. Specs are the
 * frozen v1 JSON wire format; observation/action_space are borrowed contract
 * spaces, not retained. trust_entrypoints allows custom-input entrypoint strings
 * (the C caller vets them). On RLMESH_OK *out_plan owns a plan; free it with
 * rlmesh_adapter_plan_free. Per-step apply is not yet exposed. */
typedef struct RlmeshAdapterPlan RlmeshAdapterPlan;

RlmeshStatus rlmesh_adapter_resolve(const char* env_tags_json,
                                    const RlmeshSpaceSpec* observation_space,
                                    const RlmeshSpaceSpec* action_space,
                                    const char* model_spec_json, bool trust_entrypoints,
                                    RlmeshAdapterPlan** out_plan);
void rlmesh_adapter_plan_free(RlmeshAdapterPlan* plan);
/* Human-readable summary (UTF-8) into out; free with rlmesh_bytes_free. */
RlmeshStatus rlmesh_adapter_plan_describe(const RlmeshAdapterPlan* plan, RlmeshBytes* out);
/* Top-level observation keys the plan reads, as a JSON array of strings into
 * out; free with rlmesh_bytes_free. */
RlmeshStatus rlmesh_adapter_plan_referenced_obs_keys(const RlmeshAdapterPlan* plan,
                                                     RlmeshBytes* out);
/* The env's EnvTags as JSON into out (ready for rlmesh_adapter_resolve); free
 * with rlmesh_bytes_free. Empty buffer (RLMESH_OK) when the env is untagged. */
RlmeshStatus rlmesh_contract_adapter_tags_json(const RlmeshContract* contract, RlmeshBytes* out);

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
 * a worker thread, so `user_data` must be thread-migration-safe.
 * `predict` returns a plain int (0 == RLMESH_OK, nonzero declines) so an
 * out-of-range value from a C author stays well-defined. */
typedef struct RlmeshModelVtable {
  size_t struct_size;
  int (*predict)(void* user_data, const RlmeshObservation* obs, RlmeshBytes* out_action);
  void (*on_lane_reset)(void* user_data, const char* episode_id, int32_t env_index);
  void (*on_episode_end)(void* user_data, const char* episode_id, int32_t env_index);
  void (*on_close)(void* user_data);
} RlmeshModelVtable;

typedef struct RlmeshModel RlmeshModel;

RlmeshStatus rlmesh_model_new(const RlmeshModelVtable* vtable, void* user_data, RlmeshModel** out);
RlmeshStatus rlmesh_model_run_local(RlmeshModel* model, const char* env_address);
RlmeshStatus rlmesh_model_run_local_for_episodes(RlmeshModel* model, const char* env_address,
                                                 uint64_t max_episodes);

/* Serve options for rlmesh_model_serve. Pass NULL for all defaults (no auth, no
 * remote shutdown, no timeouts — serves until the process is killed). A 0 timeout
 * / concurrency means "unset". */
typedef struct RlmeshServeOptions {
  const char* token;          /* NULL/"" disables auth */
  bool allow_remote_shutdown; /* honor a client-issued shutdown request */
  uint64_t idle_timeout_ms;   /* 0 = never idle-shutdown */
  uint64_t drain_timeout_ms;  /* 0 = unset */
  uint64_t close_timeout_ms;  /* 0 = unset */
  size_t predict_concurrency; /* 0 = default */
} RlmeshServeOptions;

/* Serve the model as a ModelService endpoint at `bind_address` (tcp://host:port
 * or unix:///path). Blocking — returns when the server stops (a remote shutdown
 * request or an idle timeout). The same vtable backs every predict, exactly as
 * rlmesh_model_run_local. `options` may be NULL for defaults. */
RlmeshStatus rlmesh_model_serve(RlmeshModel* model, const char* bind_address,
                                const RlmeshServeOptions* options);

void rlmesh_model_free(RlmeshModel* model);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RLMESH_H */
