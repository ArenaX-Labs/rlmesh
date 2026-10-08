/* RLMesh C ABI — experimental. Both v1 paths: a C/C++ model driving a remote
 * environment, and a C/C++ environment served to remote models. Hand-authored;
 * cbindgen-verifiable (a header-drift CI gate is a follow-up). */
#ifndef RLMESH_H
#define RLMESH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Symbol visibility. A consumer needs nothing on ELF/Mach-O; on Windows it gets
 * __declspec(dllimport) unless it defines RLMESH_STATIC (static link). The
 * library's own build defines RLMESH_BUILD_DLL. */
#if defined(_WIN32) && !defined(RLMESH_STATIC)
#ifdef RLMESH_BUILD_DLL
#define RLMESH_API __declspec(dllexport)
#else
#define RLMESH_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define RLMESH_API __attribute__((visibility("default")))
#else
#define RLMESH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Binary ABI generation — bumped ONLY on a binary-incompatible change (a
 * repr(C) layout/enum-discriminant change, an extern "C" signature retype, or a
 * symbol removal). Decoupled from the package semver below, which can't express
 * an ABI break. Appending a struct_size-guarded vtable field is NOT a break. */
#define RLMESH_ABI_VERSION 4

RLMESH_API uint32_t rlmesh_abi_version(void);

/* Nonzero when the linked library's ABI generation matches the one this header
 * was compiled against. SONAME-linked consumers are already gated by the loader
 * (librlmesh_capi.so.N); this is for dlopen / raw-path consumers that bypass it. */
static inline int rlmesh_abi_check(void) { return RLMESH_ABI_VERSION == rlmesh_abi_version(); }

/* Package (marketing) semver — informational only. Do NOT gate ABI
 * compatibility on these; use RLMESH_ABI_VERSION / rlmesh_abi_check(). */
#define RLMESH_ABI_VERSION_MAJOR 0
#define RLMESH_ABI_VERSION_MINOR 1
#define RLMESH_ABI_VERSION_PATCH 0

RLMESH_API uint32_t rlmesh_abi_version_major(void);
RLMESH_API uint32_t rlmesh_abi_version_minor(void);
RLMESH_API uint32_t rlmesh_abi_version_patch(void);

/* Signals: creating a model or env handle sets SIGPIPE to ignored (POSIX), unless
 * the host already installed a handler for it. The transport's socket writes
 * cannot opt out of SIGPIPE, and its default action would kill the host when a
 * peer disconnects mid-write; with it ignored, that write fails with EPIPE. */

/* ---- status + errors ---------------------------------------------------- */

typedef enum RLMeshStatus {
  RLMESH_OK = 0,
  RLMESH_ERR_INVALID_ARGUMENT = 1,
  RLMESH_ERR_INVALID_VALUE = 2,
  RLMESH_ERR_ENVIRONMENT = 3,
  RLMESH_ERR_MODEL = 4,
  RLMESH_ERR_TRANSPORT = 5,
  RLMESH_ERR_TIMEOUT = 6,
  RLMESH_ERR_PANIC = 7,
  RLMESH_ERR_CANCELLED = 8,
  RLMESH_ERR_INTERNAL = 99
} RLMeshStatus;

/* Most recent failing call's message on THIS thread (valid until the next
 * RLMesh call on this thread; NULL if none). Read only after a nonzero status. */
RLMESH_API const char* rlmesh_last_error_message(void);
RLMESH_API int rlmesh_last_error_is_recoverable(void);

/* The status that same call recorded — RLMESH_OK when none is. Redundant after
 * a status-returning call (it returned the same code); it is the only status
 * channel the pointer-returning calls have, since they report failure by
 * returning NULL. */
RLMESH_API RLMeshStatus rlmesh_last_error_status(void);

/* ---- dtype + tensor ----------------------------------------------------- */

/* DLPack (code, bits, lanes): code is DLDataTypeCode (int=0, uint=1, float=2,
 * bfloat=4, bool=6); lanes is always 1. */
typedef struct RLMeshDType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} RLMeshDType;

#ifdef __cplusplus
#define RLMESH_DTYPE_INIT(c, b, l) \
  RLMeshDType { (uint8_t)(c), (uint8_t)(b), (uint16_t)(l) }
#else
#define RLMESH_DTYPE_INIT(c, b, l) \
  (RLMeshDType) { (uint8_t)(c), (uint8_t)(b), (uint16_t)(l) }
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
RLMESH_API size_t rlmesh_dtype_size(RLMeshDType dtype);

/* A DLPack-shaped tensor view. `strides` is in element counts (NULL = row-major
 * contiguous); `data` points at element 0. A tensor returned by value is a
 * borrowed view (`deleter == NULL`) valid only while its source value lives.
 *
 * `data` is const: a borrowed view aliases library-owned memory, so writing
 * through it is undefined. A tensor you fill in for rlmesh_value_box is only
 * read, so the same const pointer serves both directions. */
typedef struct RLMeshTensor {
  const void* data;
  int32_t ndim;
  const int64_t* shape;
  const int64_t* strides;
  RLMeshDType dtype;
  int32_t device_type;
  int32_t device_id;
  uint64_t flags;
  void* manager_ctx;
  void (*deleter)(struct RLMeshTensor* self);
} RLMeshTensor;

/* Release a tensor's backing resource (`manager_ctx` only; never frees `self`).
 * A no-op for a borrowed view. */
RLMESH_API void rlmesh_tensor_release(RLMeshTensor* tensor);

/* ---- values (a SpaceValue projection) ----------------------------------- */

typedef struct RLMeshValue RLMeshValue;

typedef enum RLMeshValueKind {
  RLMESH_VALUE_INVALID = 0, /* NULL handle, or a space of unspecified kind */
  RLMESH_VALUE_BOX = 1,
  RLMESH_VALUE_DISCRETE = 2,
  RLMESH_VALUE_MULTI_BINARY = 3,
  RLMESH_VALUE_MULTI_DISCRETE = 4,
  RLMESH_VALUE_TEXT = 5,
  RLMESH_VALUE_DICT = 10,
  RLMESH_VALUE_TUPLE = 11
} RLMeshValueKind;

/* Accessor convention across the value and space surfaces: a fallible read
 * returns RLMeshStatus and writes through an out-param (detail in
 * rlmesh_last_error_message()); a kind accessor returns RLMESH_VALUE_INVALID;
 * a borrow accessor returns a pointer, NULL when there is no such child. A
 * length is never a sentinel — 0 is a legal length — so it travels by status. */

/* RLMESH_VALUE_INVALID when `value` is NULL. */
RLMESH_API RLMeshValueKind rlmesh_value_kind(const RLMeshValue* value);

/* Box: borrowed tensor view (valid while `value` lives) / copy-construct. */
RLMESH_API RLMeshStatus rlmesh_value_as_tensor(const RLMeshValue* value, RLMeshTensor* out);
RLMESH_API RLMeshValue* rlmesh_value_box(const RLMeshTensor* tensor); /* contiguous only */

/* Discrete. */
RLMESH_API RLMeshValue* rlmesh_value_discrete(int64_t value);
RLMESH_API RLMeshStatus rlmesh_value_as_discrete(const RLMeshValue* value, int64_t* out);

/* Text: `len` UTF-8 bytes (not NUL-terminated), in either direction. The
 * accessor borrows `*out_len` bytes that are NOT NUL-terminated and stay valid
 * only while `value` lives. */
RLMESH_API RLMeshValue* rlmesh_value_text(const char* data, size_t len);
RLMESH_API RLMeshStatus rlmesh_value_as_text(const RLMeshValue* value, const char** out_ptr,
                                             size_t* out_len);

/* MultiBinary / MultiDiscrete: constructed from / copied into a caller buffer.
 * A copy into a NULL `out` is OK when the value is empty, an error otherwise.
 * The element count is the space's shape flattened row-major: `n` for the usual
 * MultiBinary([n]) / MultiDiscrete(nvec), the product of the dims for a
 * multi-dimensional one (both spaces allow any rank >= 1). MultiBinary carries
 * no `n` of its own — see rlmesh_space_copy_shape below. */
RLMESH_API RLMeshValue* rlmesh_value_multi_discrete(const int64_t* data, size_t n);
RLMESH_API RLMeshValue* rlmesh_value_multi_binary(const uint8_t* data, size_t n);
/* Element count into `*out`; RLMESH_ERR_INVALID_VALUE for any other kind. */
RLMESH_API RLMeshStatus rlmesh_value_array_len(const RLMeshValue* value, size_t* out);
RLMESH_API RLMeshStatus rlmesh_value_copy_multi_discrete(const RLMeshValue* value, int64_t* out,
                                                         size_t cap);
RLMESH_API RLMeshStatus rlmesh_value_copy_multi_binary(const RLMeshValue* value, uint8_t* out,
                                                       size_t cap);

/* Dict / Tuple: borrowed children (valid while `value` lives). */

/* Child count into `*out`; RLMESH_ERR_INVALID_VALUE for any other kind. */
RLMESH_API RLMeshStatus rlmesh_value_len(const RLMeshValue* value, size_t* out);
/* NULL when `value` is not that kind, or the index/key is absent. */
RLMESH_API const RLMeshValue* rlmesh_value_tuple_get(const RLMeshValue* value, size_t index);
RLMESH_API const RLMeshValue* rlmesh_value_dict_get(const RLMeshValue* value, const char* key);
/* The `index`-th dict key in sorted order: `*out_len` UTF-8 bytes, NOT
 * NUL-terminated, valid while `value` lives. Pair with rlmesh_value_len to
 * iterate a dict (its keys are otherwise undiscoverable from C).
 * NOTE: a dict VALUE indexes in sorted key order, a dict SPACE in declaration
 * order (rlmesh_space_dict_key) — the two can disagree, so match a value child
 * to its space by KEY (rlmesh_space_dict_get), never by index. */
RLMESH_API RLMeshStatus rlmesh_value_dict_key(const RLMeshValue* value, size_t index,
                                              const char** out_ptr, size_t* out_len);
/* The `index`-th dict child, in the SAME sorted-key order as
 * rlmesh_value_dict_key — so key(i) names get_at(i). Iterating a dict this way
 * needs no NUL-terminated copy of each key. NULL when `value` is not a Dict or
 * `index` is out of range. */
RLMESH_API const RLMeshValue* rlmesh_value_dict_get_at(const RLMeshValue* value, size_t index);

/* The composite constructors take ownership of (and free) each child value on
 * success. On failure (NULL return) they take ownership of NOTHING: every child
 * is still the caller's to free. rlmesh_value_dict additionally requires unique
 * keys — a duplicate is RLMESH_ERR_INVALID_ARGUMENT, not a silent last-wins. */
RLMESH_API RLMeshValue* rlmesh_value_tuple(RLMeshValue* const* children, size_t n);
RLMESH_API RLMeshValue* rlmesh_value_dict(const char* const* keys, RLMeshValue* const* values,
                                          size_t n);

/* Free an owned value (from a constructor). Not for a borrowed child (*_get), a
 * predict observation row, or a tensor view. */
RLMESH_API void rlmesh_value_free(RLMeshValue* value);

/* ---- spaces + contract ------------------------------------------------- */

typedef struct RLMeshSpaceSpec RLMeshSpaceSpec;
typedef struct RLMeshContract RLMeshContract;

RLMESH_API const RLMeshSpaceSpec* rlmesh_contract_observation_space(const RLMeshContract* contract);
RLMESH_API const RLMeshSpaceSpec* rlmesh_contract_action_space(const RLMeshContract* contract);
RLMESH_API uint32_t rlmesh_contract_num_envs(const RLMeshContract* contract);

/* Space introspection — enough to build a valid zero (or random) action for any
 * space: shape + dtype + bounds for Box, n/start for Discrete, the length limits
 * for Text, the per-element category counts for MultiDiscrete, and a child walk
 * for Dict/Tuple. RLMESH_VALUE_INVALID when `spec` is NULL or unspecified. */
RLMESH_API RLMeshValueKind rlmesh_space_type(const RLMeshSpaceSpec* spec);
RLMESH_API RLMeshDType rlmesh_space_dtype(const RLMeshSpaceSpec* spec);
/* Rank; 0 for a scalar shape or a NULL spec. */
RLMESH_API size_t rlmesh_space_ndim(const RLMeshSpaceSpec* spec);
/* Shape into `out` (capacity `cap`); a NULL `out` is OK for a rank-0 space. */
RLMESH_API RLMeshStatus rlmesh_space_copy_shape(const RLMeshSpaceSpec* spec, int64_t* out,
                                                size_t cap);

/* Composite walk: child count into `*out`, then children by index (Tuple) or by
 * key (Dict). The borrow accessors return NULL for the wrong kind or an absent
 * index/key; children stay valid while `spec` lives. */
RLMESH_API RLMeshStatus rlmesh_space_len(const RLMeshSpaceSpec* spec, size_t* out);
RLMESH_API const RLMeshSpaceSpec* rlmesh_space_tuple_get(const RLMeshSpaceSpec* spec, size_t index);
RLMESH_API const RLMeshSpaceSpec* rlmesh_space_dict_get(const RLMeshSpaceSpec* spec,
                                                        const char* key);
/* The `index`-th dict key (declaration order, parallel to the children, and the
 * order the wire encodes the dict's leaves in): `*out_len` UTF-8 bytes, NOT
 * NUL-terminated, valid while `spec` lives. This is NOT the sorted order
 * rlmesh_value_dict_key uses — pair a space child with a value child by key. */
RLMESH_API RLMeshStatus rlmesh_space_dict_key(const RLMeshSpaceSpec* spec, size_t index,
                                              const char** out_ptr, size_t* out_len);
/* The `index`-th dict child, in the SAME declaration order as
 * rlmesh_space_dict_key — so key(i) names get_at(i). NULL when `spec` is not a
 * Dict or `index` is out of range. */
RLMESH_API const RLMeshSpaceSpec* rlmesh_space_dict_get_at(const RLMeshSpaceSpec* spec,
                                                           size_t index);

/* Leaf parameters. `rlmesh_space_box_bounds` reports the `index`-th element's
 * inclusive bounds in row-major order (a uniform bound broadcasts; an undeclared
 * one reads as -inf/+inf); `index` must be below the element count.
 * A Discrete space's valid values are `start ..= start + n - 1`. For the paired
 * out-params, either may be NULL to skip it. */
RLMESH_API RLMeshStatus rlmesh_space_box_bounds(const RLMeshSpaceSpec* spec, size_t index,
                                                double* out_low, double* out_high);
RLMESH_API RLMeshStatus rlmesh_space_discrete_n(const RLMeshSpaceSpec* spec, int64_t* out_n,
                                                int64_t* out_start);
RLMESH_API RLMeshStatus rlmesh_space_text_length(const RLMeshSpaceSpec* spec, int64_t* out_min,
                                                 int64_t* out_max);
/* A Text space's allowed characters: `*out_len` UTF-8 bytes, NOT NUL-terminated,
 * valid while `spec` lives. An EMPTY charset means any character is allowed. */
RLMESH_API RLMeshStatus rlmesh_space_text_charset(const RLMeshSpaceSpec* spec, const char** out_ptr,
                                                  size_t* out_len);
/* One category count per element of the shape, row-major (capacity `cap`).
 * A MultiBinary space has no such table and no `n` field of its own: its
 * dimensions live entirely in the shape (rlmesh_space_ndim /
 * rlmesh_space_copy_shape), so MultiBinary([n]) reads as shape `[n]` and a
 * value for it carries exactly `n` elements — the shape's row-major element
 * count, for a multi-dimensional MultiBinary too. */
RLMESH_API RLMeshStatus rlmesh_space_copy_nvec(const RLMeshSpaceSpec* spec, int64_t* out,
                                               size_t cap);

/* Space builders (the env-authoring side). Each returns an OWNED space, or NULL
 * on error (detail in rlmesh_last_error_message()); free with rlmesh_space_free
 * unless a composite builder adopted it. A float Box takes +-INFINITY for an
 * unbounded side; an integer-dtype Box needs finite whole-number bounds the
 * dtype can represent (none negative for an unsigned dtype): an out-of-range
 * bound is RLMESH_ERR_INVALID_ARGUMENT, never clamped. A double holds every
 * integer only up to 2^53: a larger 64-bit bound is the double's exact value. */
RLMESH_API RLMeshSpaceSpec* rlmesh_space_box(RLMeshDType dtype, const int64_t* shape, size_t ndim,
                                             double low, double high);
/* Per-element bounds: `low` / `high` each hold the shape's element count. */
RLMESH_API RLMeshSpaceSpec* rlmesh_space_box_elementwise(RLMeshDType dtype, const int64_t* shape,
                                                         size_t ndim, const double* low,
                                                         const double* high);
RLMESH_API RLMeshSpaceSpec* rlmesh_space_discrete(int64_t n, int64_t start);
RLMESH_API RLMeshSpaceSpec* rlmesh_space_multi_binary(const int64_t* shape, size_t ndim);
RLMESH_API RLMeshSpaceSpec* rlmesh_space_multi_discrete(const int64_t* nvec, size_t n);
/* `charset` NULL or "" allows any character. */
RLMESH_API RLMeshSpaceSpec* rlmesh_space_text(int64_t min_length, int64_t max_length,
                                              const char* charset);
/* The composite builders take ownership of every child on success and of
 * NOTHING on failure (NULL), whatever the reason, the same rule as
 * rlmesh_value_dict. Dict keys must be non-empty and unique. */
RLMESH_API RLMeshSpaceSpec* rlmesh_space_dict(const char* const* keys,
                                              RLMeshSpaceSpec* const* children, size_t n);
RLMESH_API RLMeshSpaceSpec* rlmesh_space_tuple(RLMeshSpaceSpec* const* children, size_t n);
/* Free an owned space. Not for a borrowed contract space or child. */
RLMESH_API void rlmesh_space_free(RLMeshSpaceSpec* spec);

/* ---- bytes -------------------------------------------------------------- */

/* An owned buffer produced by the capi (its `cap` lets the capi reclaim the
 * allocation). Free with rlmesh_bytes_free. An EMPTY buffer is normalized to
 * `{NULL, 0, 0}` — `data` is never a non-NULL zero-length pointer, so a
 * consumer may branch on `data` as well as on `len` — and rlmesh_bytes_free
 * accepts that form. */
typedef struct RLMeshBytes {
  uint8_t* data;
  size_t len;
  size_t cap;
} RLMeshBytes;

RLMESH_API void rlmesh_bytes_free(RLMeshBytes bytes);

/* ---- adapters (experimental) -------------------------------------------- */

/* Resolve the env's tags (env_tags_json; see rlmesh_contract_adapter_tags_json)
 * against this model's spec (model_spec_json) into an opaque plan. Specs are the
 * frozen v1 JSON wire format; observation/action_space are borrowed contract
 * spaces, not retained. trust_entrypoints allows custom-input entrypoint strings
 * (the C caller vets them). On RLMESH_OK *out_plan owns a plan; free it with
 * rlmesh_adapter_plan_free. Per-step apply is not yet exposed. */
typedef struct RLMeshAdapterPlan RLMeshAdapterPlan;

RLMESH_API RLMeshStatus rlmesh_adapter_resolve(const char* env_tags_json,
                                               const RLMeshSpaceSpec* observation_space,
                                               const RLMeshSpaceSpec* action_space,
                                               const char* model_spec_json, bool trust_entrypoints,
                                               RLMeshAdapterPlan** out_plan);
RLMESH_API void rlmesh_adapter_plan_free(RLMeshAdapterPlan* plan);
/* Human-readable summary (UTF-8) into out; free with rlmesh_bytes_free. */
RLMESH_API RLMeshStatus rlmesh_adapter_plan_describe(const RLMeshAdapterPlan* plan,
                                                     RLMeshBytes* out);
/* Top-level observation keys the plan reads, as a JSON array of strings into
 * out; free with rlmesh_bytes_free. */
RLMESH_API RLMeshStatus rlmesh_adapter_plan_referenced_obs_keys(const RLMeshAdapterPlan* plan,
                                                                RLMeshBytes* out);
/* The env's EnvTags as JSON into out (ready for rlmesh_adapter_resolve); free
 * with rlmesh_bytes_free. Empty buffer (RLMESH_OK) when the env is untagged. */
RLMESH_API RLMeshStatus rlmesh_contract_adapter_tags_json(const RLMeshContract* contract,
                                                          RLMeshBytes* out);

/* ---- model -------------------------------------------------------------- */

/* Error convention for every call below: on a nonzero RLMeshStatus (or a NULL
 * return from a pointer-returning call) the failing call has ALREADY recorded
 * its message on this thread, so a caller can simply propagate the status and
 * let the outermost frame read rlmesh_last_error_message(). Never set your own
 * message for a capi call that already failed. */

/* One row's episode identity. `id` is runtime-minted and never repeats, so a
 * stateful model keys per-episode state by it (no positional lane concept).
 *
 * `predict_index` is the episode's re-plan ordinal: 0 on the first predict
 * under `id`, then +1 per predict until on_episode_end drops the episode (the
 * capi counts it per episode id, the way the Python SDK does; the runtime does
 * not send it). `predict_seed` mixes the episode's reset seed with that ordinal
 * (rlmesh.predict_seed in the SDK) so a stochastic policy can seed each forward
 * reproducibly under interleaving; it is meaningful only when `seeded`, 0
 * otherwise. The capi holds at most 4096 live episodes: past that the least
 * recently predicted one is evicted through on_episode_end. */
typedef struct RLMeshEpisode {
  const char* id;
  bool seeded; /* whether `seed` carries an explicit reset seed */
  int64_t seed;
  uint64_t predict_index; /* re-plan ordinal within the episode, from 0 */
  int64_t predict_seed;   /* predict_seed(seed, predict_index); only when `seeded` */
} RLMeshEpisode;

/* What a predict callback receives. Every pointer is valid only for the
 * duration of the call. Row i of `observations` belongs to `episodes[i]`;
 * `episodes` always has exactly num_envs entries. */
typedef struct RLMeshObservation {
  /* num_envs decoded values, or NULL when this request carries no observation
   * or the contract declares no observation space (both are legal routes). */
  const RLMeshValue* const* observations;
  size_t num_envs;
  /* Spaces/metadata for the route. Never NULL on a predict the runtime
   * delivers: a route pins a contract (with an action space) before its first
   * predict. */
  const RLMeshContract* contract;
  const char* session_id;
  const char* env_id;
  const char* request_id;
  const RLMeshEpisode* episodes; /* num_envs entries */
} RLMeshObservation;

/* Set this call's error message + recoverability before returning nonzero (any
 * callback: a model's predict or an env's reset / step / render). */
RLMESH_API void rlmesh_callback_set_error(const char* message, bool recoverable);

/* The model callback vtable. Set struct_size = sizeof(RLMeshModelVtable); fields
 * beyond that are ignored (append-only). `predict` is required. The vtable is
 * COPIED into the model at rlmesh_model_new, so the caller's struct need not
 * outlive the model; `user_data` is kept by pointer and must.
 *
 * Callbacks run on a worker thread, so `user_data` must be thread-migration-safe.
 * A callback must not re-enter the model handle that invoked it (no
 * rlmesh_model_run_local / _serve / _free on it from inside a callback);
 * rlmesh_model_cancel is the one exception and is safe from anywhere.
 *
 * `predict` writes one OWNED action value per row into `out_actions` (num_envs
 * slots, pre-zeroed; the capi takes ownership) and returns 0 == RLMESH_OK, or
 * nonzero to decline. On a nonzero return the capi frees any rows already
 * written. The return is a plain int so an out-of-range value stays defined.
 * Actions are validated against the route's action space by the runtime: a
 * structural mismatch (wrong kind/shape/dtype) fails the step; a value merely
 * outside a Box's bounds is left to the environment's own policy.
 *
 * `on_episode_end` fires when the runtime drops an episode; `episode_id` NULL
 * means every episode of `env_id`. `on_close` fires once at shutdown. */
typedef struct RLMeshModelVtable {
  size_t struct_size;
  int (*predict)(void* user_data, const RLMeshObservation* obs, RLMeshValue** out_actions);
  void (*on_episode_end)(void* user_data, const char* env_id, const char* episode_id);
  void (*on_close)(void* user_data);
} RLMeshModelVtable;

/* An owned model handle. It is not a concurrency primitive: run at most one
 * rlmesh_model_run_local / rlmesh_model_serve on a handle at a time, and free it
 * only from a thread that is not executing it. rlmesh_model_cancel is the only
 * call that may overlap a running one, from any thread. */
typedef struct RLMeshModel RLMeshModel;

RLMESH_API RLMeshStatus rlmesh_model_new(const RLMeshModelVtable* vtable, void* user_data,
                                         RLMeshModel** out);

/* Run options for rlmesh_model_run_local. NULL == defaults (run until the env
 * ends, unseeded, no caps). Every field is "unset" at 0 / NULL. */
typedef struct RLMeshRunOptions {
  uint64_t max_episodes;        /* 0 = until the env ends */
  bool seeded;                  /* whether base_seed is set */
  int64_t base_seed;            /* seed for episode 0; later episodes derive from it */
  int64_t max_episode_steps;    /* truncate an episode after this many steps; 0 = unset */
  double max_episode_seconds;   /* truncate an episode after this long; 0 = unset */
  uint32_t execution_horizon;   /* actions per predicted chunk to execute; 0/1 = no chunking */
  bool close_env;               /* ask the env to close when the run ends */
  const int64_t* episode_seeds; /* explicit per-episode seeds (overrides base_seed); NULL = unset */
  size_t num_episode_seeds;     /* length of episode_seeds; 0 = unset */
  bool trial_indexed;           /* whether trial_index_base is set */
  uint64_t trial_index_base;    /* first trial ordinal the episodes walk (base, base+1, ...);
                                   delivered as reset(options={"trial_index": k}) to an env
                                   that declares that reset option */
} RLMeshRunOptions;

/* What a finished run reports. Plain scalars: copy what you need. */
typedef struct RLMeshRunReport {
  int64_t total_episodes;      /* episodes completed during the run */
  int64_t total_steps;         /* env steps across the whole run */
  double total_reward;         /* summed episode reward */
  double mean_reward;          /* mean episode reward (0 if none completed) */
  int64_t terminated_episodes; /* completed episodes that ended terminated */
  int64_t truncated_episodes;  /* completed episodes that hit a step/time cap */
} RLMeshRunReport;

/* Drive the model against the env at `env_address` (tcp://host:port,
 * host:port, or unix:///path). Blocking — returns when the run ends.
 * `out_report` may be NULL; it is written only on RLMESH_OK. */
RLMESH_API RLMeshStatus rlmesh_model_run_local(RLMeshModel* model, const char* env_address,
                                               const RLMeshRunOptions* options,
                                               RLMeshRunReport* out_report);

/* Serve options for rlmesh_model_serve. Pass NULL for all defaults (no auth, no
 * remote shutdown, no timeouts — serves until the process is killed). A 0 timeout
 * / concurrency means "unset". */
typedef struct RLMeshServeOptions {
  const char* token;            /* NULL/"" disables auth */
  bool allow_remote_shutdown;   /* honor a client-issued shutdown request */
  uint64_t idle_timeout_ms;     /* 0 = never idle-shutdown */
  uint64_t drain_timeout_ms;    /* 0 = unset */
  uint64_t close_timeout_ms;    /* 0 = unset */
  size_t predict_concurrency;   /* 0 = default */
  const char* workflow_edition; /* workflow edition this server declares; NULL/"" = none */
} RLMeshServeOptions;

/* Serve the model as a ModelService endpoint at `bind_address` (tcp://host:port
 * or unix:///path). Blocking — returns when the server stops: a remote shutdown
 * request, an idle timeout, or rlmesh_model_cancel. The same vtable backs every
 * predict, exactly as rlmesh_model_run_local. `options` may be NULL for
 * defaults. */
RLMESH_API RLMeshStatus rlmesh_model_serve(RLMeshModel* model, const char* bind_address,
                                           const RLMeshServeOptions* options);

/* Stop a blocking rlmesh_model_run_local / rlmesh_model_serve on `model`. Call
 * it from another thread (a signal handler's worker, a UI thread); it returns
 * immediately and the blocked call unwinds shortly after. NULL is a no-op.
 *
 * Cancellation is terminal for the handle: a cancelled model refuses further
 * runs. A cancelled serve returns RLMESH_OK after running on_close -- once, and
 * only after any predict still inside the C callback has returned; a cancelled
 * run_local (this one, or any later one on the handle) returns
 * RLMESH_ERR_CANCELLED, since it has no report to give. */
RLMESH_API void rlmesh_model_cancel(RLMeshModel* model);

/* Free a model handle. NULL is a no-op. Must NOT be called from inside one of
 * this model's own callbacks (it drops the runtime the callback is running on);
 * doing so is contained rather than fatal, but leaves the handle undefined. */
RLMESH_API void rlmesh_model_free(RLMeshModel* model);

/* ---- environment ------------------------------------------------------- */

/* What `reset` receives; pointers are valid for the call only. `trial_index`
 * arrives only when the env declared the "trial_index" reset option. */
typedef struct RLMeshResetArgs {
  bool seeded; /* whether `seed` carries an explicit reset seed */
  int64_t seed;
  bool has_trial_index; /* whether `trial_index` is set */
  int64_t trial_index;
  const char* options_json; /* every reset option as a JSON object; NULL = none */
} RLMeshResetArgs;

/* What `reset` / `step` write. `observation` is OWNED (the capi takes it).
 * `info_json` is a borrowed JSON object (NULL = none) that must stay valid
 * until the callback's next call on this env (e.g. a std::string member). */
typedef struct RLMeshResetResult {
  RLMeshValue* observation;
  const char* info_json;
} RLMeshResetResult;

typedef struct RLMeshStepResult {
  RLMeshValue* observation;
  double reward;
  bool terminated;
  bool truncated;
  const char* info_json;
} RLMeshStepResult;

/* The environment callback vtable. Set struct_size = sizeof(RLMeshEnvVtable);
 * fields beyond it are ignored (append-only). `reset` and `step` are required.
 * Copied in at rlmesh_env_new; `user_data` is kept by pointer and must outlive
 * the env.
 *
 * Each lane runs its callbacks on ONE dedicated thread (not the one that
 * created the env), one call at a time; the lanes of a multi-lane env
 * (rlmesh_env_new_lanes) run concurrently, each with its own `user_data`. A
 * foreground env (RLMeshEnvConfig.foreground) runs them on the thread blocked
 * in rlmesh_env_serve instead. `reset` / `step` /
 * `render` return 0 == RLMESH_OK, or nonzero after rlmesh_callback_set_error
 * to fail that request. With `recoverable` set, only the request fails and the
 * client's session stays usable; otherwise the session ends (the client's next
 * reset opens a new one, and the env keeps serving). `step` gets the action
 * borrowed (NULL when the request carries none). `render` writes an OWNED uint8
 * image value of shape [H, W, 3], [H, W, 4] or [H, W] (or leaves it NULL for no
 * frame); the capi PNG-encodes it. `close` fires once, when the server stops. */
typedef struct RLMeshEnvVtable {
  size_t struct_size;
  int (*reset)(void* user_data, const RLMeshResetArgs* args, RLMeshResetResult* out);
  int (*step)(void* user_data, const RLMeshValue* action, RLMeshStepResult* out);
  int (*render)(void* user_data, RLMeshValue** out_frame);
  void (*close)(void* user_data);
} RLMeshEnvVtable;

/* What the env declares. Set struct_size = sizeof(RLMeshEnvConfig); fields
 * beyond it read as unset. Borrowed for the rlmesh_env_new call only (the
 * spaces are cloned). `adapter_tags_json` is the env's adapter EnvTags (v1
 * JSON), validated against the spaces right there: an unknown field or a tag
 * that does not fit its space fails rlmesh_env_new. The JSON grammar is in
 * docs/specs/env_tags.v1.md. */
typedef struct RLMeshEnvConfig {
  size_t struct_size;
  const char* id;                           /* contract id; NULL/"" = "env" */
  const RLMeshSpaceSpec* observation_space; /* required */
  const RLMeshSpaceSpec* action_space;      /* required */
  const char* adapter_tags_json;            /* NULL = untagged */
  const char* const* reset_options;         /* e.g. {"trial_index"}; NULL = none */
  size_t num_reset_options;
  const char* render_mode;   /* "rgb_array" when `render` produces frames */
  const char* metadata_json; /* extra contract metadata (JSON object); NULL = none */
  /* Run every callback (close included) on the thread that calls
   * rlmesh_env_serve, which then serves from a helper thread: for a simulation
   * bound to the thread that created it (a GL / Vulkan context, Isaac Sim).
   * One lane only. false = callbacks run on a dedicated lane thread. */
  bool foreground;
} RLMeshEnvConfig;

/* An owned env handle. Lifecycle: rlmesh_env_new -> rlmesh_env_bind ->
 * rlmesh_env_serve (blocks) -> rlmesh_env_free. rlmesh_env_cancel is the only
 * call that may overlap a running serve, from any thread. */
typedef struct RLMeshEnv RLMeshEnv;

RLMESH_API RLMeshStatus rlmesh_env_new(const RLMeshEnvVtable* vtable, const RLMeshEnvConfig* config,
                                       void* user_data, RLMeshEnv** out);

/* Serve `num_lanes` independent simulations as the lanes of one endpoint
 * (num_envs = num_lanes): lane i passes `user_data[i]` to its callbacks. Every
 * lane shares `vtable` and `config` (one contract) but runs on its own thread,
 * concurrently with the others, so each `user_data` must be its own simulation.
 * `user_data` (the array) is borrowed for the call; each entry must outlive the
 * env. num_lanes == 0 and a foreground config with num_lanes > 1 fail with
 * RLMESH_ERR_INVALID_ARGUMENT. */
RLMESH_API RLMeshStatus rlmesh_env_new_lanes(const RLMeshEnvVtable* vtable,
                                             const RLMeshEnvConfig* config, void* const* user_data,
                                             size_t num_lanes, RLMeshEnv** out);

/* Bind to `bind_address` (tcp://host:port, host:port or unix:///path) without
 * serving yet; once per handle. `out_address` (may be NULL) receives the
 * resolved address, e.g. the OS-assigned port for port 0 (UTF-8, not
 * NUL-terminated; free with rlmesh_bytes_free). `options` may be NULL;
 * `predict_concurrency` does not apply to an env. */
RLMESH_API RLMeshStatus rlmesh_env_bind(RLMeshEnv* env, const char* bind_address,
                                        const RLMeshServeOptions* options,
                                        RLMeshBytes* out_address);

/* The env's describe envelope (rlmesh.describe.v1: target "native:<id>", the
 * spaces as env_spec, the published adapter tags as env_tags, and the runtime
 * edition handshake) as UTF-8 JSON into `out` (not NUL-terminated; free with
 * rlmesh_bytes_free). Valid before and after bind; after bind it declares the
 * workflow edition the server was bound with. rlmesh_env_bind also puts it on
 * the handshake PeerInfo.extra under "rlmesh.describe.v1" (process-wide: with
 * several envs in one process the last bound wins), which is where the managed
 * platform reads it for an image without a baked `dev.rlmesh.describe` label.
 * Bake that label with this output to describe the image before it runs. */
RLMESH_API RLMeshStatus rlmesh_env_describe_json(const RLMeshEnv* env, RLMeshBytes* out);

/* Serve the bound env until a remote shutdown, an idle timeout, or
 * rlmesh_env_cancel. Blocking. `close` runs once per lane before this returns.
 * A foreground env runs every callback on the calling thread until then. */
RLMESH_API RLMeshStatus rlmesh_env_serve(RLMeshEnv* env);

/* Stop a blocking rlmesh_env_serve from another thread: it drains, closes the
 * env and returns RLMESH_OK. Terminal for the handle. NULL is a no-op. */
RLMESH_API void rlmesh_env_cancel(RLMeshEnv* env);

/* Free an env handle; not from inside its own callbacks. NULL is a no-op. */
RLMESH_API void rlmesh_env_free(RLMeshEnv* env);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RLMESH_H */
