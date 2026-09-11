/* RLMesh C ABI — experimental, v1 model path (a C/C++ model driving a remote
 * environment). Hand-authored; cbindgen-verifiable (a header-drift CI gate is a
 * follow-up). */
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
#define RLMESH_ABI_VERSION 1

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
RLMESH_API const char* rlmesh_last_error_message(void);
RLMESH_API int rlmesh_last_error_is_recoverable(void);

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
RLMESH_API size_t rlmesh_dtype_size(RlmeshDType dtype);

/* A DLPack-shaped tensor view. `strides` is in element counts (NULL = row-major
 * contiguous); `data` points at element 0. A tensor returned by value is a
 * borrowed view (`deleter == NULL`) valid only while its source value lives.
 *
 * `data` is const: a borrowed view aliases library-owned memory, so writing
 * through it is undefined. A tensor you fill in for rlmesh_value_box is only
 * read, so the same const pointer serves both directions. */
typedef struct RlmeshTensor {
  const void* data;
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
RLMESH_API void rlmesh_tensor_release(RlmeshTensor* tensor);

/* ---- values (a SpaceValue projection) ----------------------------------- */

typedef struct RlmeshValue RlmeshValue;

typedef enum RlmeshValueKind {
  RLMESH_VALUE_INVALID = 0, /* NULL handle, or a space of unspecified kind */
  RLMESH_VALUE_BOX = 1,
  RLMESH_VALUE_DISCRETE = 2,
  RLMESH_VALUE_MULTI_BINARY = 3,
  RLMESH_VALUE_MULTI_DISCRETE = 4,
  RLMESH_VALUE_TEXT = 5,
  RLMESH_VALUE_DICT = 10,
  RLMESH_VALUE_TUPLE = 11
} RlmeshValueKind;

/* Accessor convention across the value and space surfaces: a fallible read
 * returns RlmeshStatus and writes through an out-param (detail in
 * rlmesh_last_error_message()); a kind accessor returns RLMESH_VALUE_INVALID;
 * a borrow accessor returns a pointer, NULL when there is no such child. A
 * length is never a sentinel — 0 is a legal length — so it travels by status. */

/* RLMESH_VALUE_INVALID when `value` is NULL. */
RLMESH_API RlmeshValueKind rlmesh_value_kind(const RlmeshValue* value);

/* Box: borrowed tensor view (valid while `value` lives) / copy-construct. */
RLMESH_API RlmeshStatus rlmesh_value_as_tensor(const RlmeshValue* value, RlmeshTensor* out);
RLMESH_API RlmeshValue* rlmesh_value_box(const RlmeshTensor* tensor); /* contiguous only */

/* Discrete. */
RLMESH_API RlmeshValue* rlmesh_value_discrete(int64_t value);
RLMESH_API RlmeshStatus rlmesh_value_as_discrete(const RlmeshValue* value, int64_t* out);

/* Text: `len` UTF-8 bytes (not NUL-terminated), in either direction. The
 * accessor borrows `*out_len` bytes that are NOT NUL-terminated and stay valid
 * only while `value` lives. */
RLMESH_API RlmeshValue* rlmesh_value_text(const char* data, size_t len);
RLMESH_API RlmeshStatus rlmesh_value_as_text(const RlmeshValue* value, const char** out_ptr,
                                             size_t* out_len);

/* MultiBinary / MultiDiscrete: constructed from / copied into a caller buffer.
 * A copy into a NULL `out` is OK when the value is empty, an error otherwise. */
RLMESH_API RlmeshValue* rlmesh_value_multi_discrete(const int64_t* data, size_t n);
RLMESH_API RlmeshValue* rlmesh_value_multi_binary(const uint8_t* data, size_t n);
/* Element count into `*out`; RLMESH_ERR_INVALID_VALUE for any other kind. */
RLMESH_API RlmeshStatus rlmesh_value_array_len(const RlmeshValue* value, size_t* out);
RLMESH_API RlmeshStatus rlmesh_value_copy_multi_discrete(const RlmeshValue* value, int64_t* out,
                                                         size_t cap);
RLMESH_API RlmeshStatus rlmesh_value_copy_multi_binary(const RlmeshValue* value, uint8_t* out,
                                                       size_t cap);

/* Dict / Tuple: borrowed children (valid while `value` lives). */

/* Child count into `*out`; RLMESH_ERR_INVALID_VALUE for any other kind. */
RLMESH_API RlmeshStatus rlmesh_value_len(const RlmeshValue* value, size_t* out);
/* NULL when `value` is not that kind, or the index/key is absent. */
RLMESH_API const RlmeshValue* rlmesh_value_tuple_get(const RlmeshValue* value, size_t index);
RLMESH_API const RlmeshValue* rlmesh_value_dict_get(const RlmeshValue* value, const char* key);
/* The `index`-th dict key in sorted order: `*out_len` UTF-8 bytes, NOT
 * NUL-terminated, valid while `value` lives. Pair with rlmesh_value_len to
 * iterate a dict (its keys are otherwise undiscoverable from C). */
RLMESH_API RlmeshStatus rlmesh_value_dict_key(const RlmeshValue* value, size_t index,
                                              const char** out_ptr, size_t* out_len);

/* The composite constructors take ownership of (and free) each child value on
 * success. On failure (NULL return) they take ownership of NOTHING: every child
 * is still the caller's to free. rlmesh_value_dict additionally requires unique
 * keys — a duplicate is RLMESH_ERR_INVALID_ARGUMENT, not a silent last-wins. */
RLMESH_API RlmeshValue* rlmesh_value_tuple(RlmeshValue* const* children, size_t n);
RLMESH_API RlmeshValue* rlmesh_value_dict(const char* const* keys, RlmeshValue* const* values,
                                          size_t n);

/* Free an owned value (from a constructor). Not for a borrowed child (*_get), a
 * predict observation row, or a tensor view. */
RLMESH_API void rlmesh_value_free(RlmeshValue* value);

/* ---- spaces + contract (read-only; builders are env-side, not yet here) -- */

typedef struct RlmeshSpaceSpec RlmeshSpaceSpec;
typedef struct RlmeshContract RlmeshContract;

RLMESH_API const RlmeshSpaceSpec* rlmesh_contract_observation_space(const RlmeshContract* contract);
RLMESH_API const RlmeshSpaceSpec* rlmesh_contract_action_space(const RlmeshContract* contract);
RLMESH_API uint32_t rlmesh_contract_num_envs(const RlmeshContract* contract);

/* Space introspection — enough to build a valid zero (or random) action for any
 * space: shape + dtype + bounds for Box, n/start for Discrete, the length limits
 * for Text, the per-element category counts for MultiDiscrete, and a child walk
 * for Dict/Tuple. RLMESH_VALUE_INVALID when `spec` is NULL or unspecified. */
RLMESH_API RlmeshValueKind rlmesh_space_type(const RlmeshSpaceSpec* spec);
RLMESH_API RlmeshDType rlmesh_space_dtype(const RlmeshSpaceSpec* spec);
/* Rank; 0 for a scalar shape or a NULL spec. */
RLMESH_API size_t rlmesh_space_ndim(const RlmeshSpaceSpec* spec);
/* Shape into `out` (capacity `cap`); a NULL `out` is OK for a rank-0 space. */
RLMESH_API RlmeshStatus rlmesh_space_copy_shape(const RlmeshSpaceSpec* spec, int64_t* out,
                                                size_t cap);

/* Composite walk: child count into `*out`, then children by index (Tuple) or by
 * key (Dict). The borrow accessors return NULL for the wrong kind or an absent
 * index/key; children stay valid while `spec` lives. */
RLMESH_API RlmeshStatus rlmesh_space_len(const RlmeshSpaceSpec* spec, size_t* out);
RLMESH_API const RlmeshSpaceSpec* rlmesh_space_tuple_get(const RlmeshSpaceSpec* spec, size_t index);
RLMESH_API const RlmeshSpaceSpec* rlmesh_space_dict_get(const RlmeshSpaceSpec* spec,
                                                        const char* key);
/* The `index`-th dict key (declaration order, parallel to the children):
 * `*out_len` UTF-8 bytes, NOT NUL-terminated, valid while `spec` lives. */
RLMESH_API RlmeshStatus rlmesh_space_dict_key(const RlmeshSpaceSpec* spec, size_t index,
                                              const char** out_ptr, size_t* out_len);

/* Leaf parameters. `rlmesh_space_box_bounds` reports the `index`-th element's
 * inclusive bounds in row-major order (a uniform bound broadcasts; an undeclared
 * one reads as -inf/+inf); `index` must be below the element count.
 * A Discrete space's valid values are `start ..= start + n - 1`. For the paired
 * out-params, either may be NULL to skip it. */
RLMESH_API RlmeshStatus rlmesh_space_box_bounds(const RlmeshSpaceSpec* spec, size_t index,
                                                double* out_low, double* out_high);
RLMESH_API RlmeshStatus rlmesh_space_discrete_n(const RlmeshSpaceSpec* spec, int64_t* out_n,
                                                int64_t* out_start);
RLMESH_API RlmeshStatus rlmesh_space_text_length(const RlmeshSpaceSpec* spec, int64_t* out_min,
                                                 int64_t* out_max);
/* One category count per element of the shape, row-major (capacity `cap`). */
RLMESH_API RlmeshStatus rlmesh_space_copy_nvec(const RlmeshSpaceSpec* spec, int64_t* out,
                                               size_t cap);

/* ---- bytes -------------------------------------------------------------- */

/* An owned buffer produced by the capi (its `cap` lets the capi reclaim the
 * allocation). Free with rlmesh_bytes_free. */
typedef struct RlmeshBytes {
  uint8_t* data;
  size_t len;
  size_t cap;
} RlmeshBytes;

RLMESH_API void rlmesh_bytes_free(RlmeshBytes bytes);

/* ---- adapters (experimental) -------------------------------------------- */

/* Resolve the env's tags (env_tags_json; see rlmesh_contract_adapter_tags_json)
 * against this model's spec (model_spec_json) into an opaque plan. Specs are the
 * frozen v1 JSON wire format; observation/action_space are borrowed contract
 * spaces, not retained. trust_entrypoints allows custom-input entrypoint strings
 * (the C caller vets them). On RLMESH_OK *out_plan owns a plan; free it with
 * rlmesh_adapter_plan_free. Per-step apply is not yet exposed. */
typedef struct RlmeshAdapterPlan RlmeshAdapterPlan;

RLMESH_API RlmeshStatus rlmesh_adapter_resolve(const char* env_tags_json,
                                               const RlmeshSpaceSpec* observation_space,
                                               const RlmeshSpaceSpec* action_space,
                                               const char* model_spec_json, bool trust_entrypoints,
                                               RlmeshAdapterPlan** out_plan);
RLMESH_API void rlmesh_adapter_plan_free(RlmeshAdapterPlan* plan);
/* Human-readable summary (UTF-8) into out; free with rlmesh_bytes_free. */
RLMESH_API RlmeshStatus rlmesh_adapter_plan_describe(const RlmeshAdapterPlan* plan,
                                                     RlmeshBytes* out);
/* Top-level observation keys the plan reads, as a JSON array of strings into
 * out; free with rlmesh_bytes_free. */
RLMESH_API RlmeshStatus rlmesh_adapter_plan_referenced_obs_keys(const RlmeshAdapterPlan* plan,
                                                                RlmeshBytes* out);
/* The env's EnvTags as JSON into out (ready for rlmesh_adapter_resolve); free
 * with rlmesh_bytes_free. Empty buffer (RLMESH_OK) when the env is untagged. */
RLMESH_API RlmeshStatus rlmesh_contract_adapter_tags_json(const RlmeshContract* contract,
                                                          RlmeshBytes* out);

/* ---- model -------------------------------------------------------------- */

/* Error convention for every call below: on a nonzero RlmeshStatus (or a NULL
 * return from a pointer-returning call) the failing call has ALREADY recorded
 * its message on this thread, so a caller can simply propagate the status and
 * let the outermost frame read rlmesh_last_error_message(). Never set your own
 * message for a capi call that already failed. */

/* One row's episode identity. `id` is runtime-minted and never repeats, so a
 * stateful model keys per-episode state by it (no positional lane concept). */
typedef struct RlmeshEpisode {
  const char* id;
  bool seeded; /* whether `seed` carries an explicit reset seed */
  int64_t seed;
} RlmeshEpisode;

/* What a predict callback receives. Every pointer is valid only for the
 * duration of the call. Row i of `observations` belongs to `episodes[i]`;
 * `episodes` always has exactly num_envs entries. */
typedef struct RlmeshObservation {
  /* num_envs decoded values, or NULL when this request carries no observation
   * or the contract declares no observation space (both are legal routes). */
  const RlmeshValue* const* observations;
  size_t num_envs;
  /* Spaces/metadata for the route. Never NULL on a predict the runtime
   * delivers: a route pins a contract (with an action space) before its first
   * predict. */
  const RlmeshContract* contract;
  const char* session_id;
  const char* env_id;
  const char* request_id;
  const RlmeshEpisode* episodes; /* num_envs entries */
} RlmeshObservation;

/* Set this call's error message + recoverability before returning nonzero. */
RLMESH_API void rlmesh_callback_set_error(const char* message, bool recoverable);

/* The model callback vtable. Set struct_size = sizeof(RlmeshModelVtable); fields
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
typedef struct RlmeshModelVtable {
  size_t struct_size;
  int (*predict)(void* user_data, const RlmeshObservation* obs, RlmeshValue** out_actions);
  void (*on_episode_end)(void* user_data, const char* env_id, const char* episode_id);
  void (*on_close)(void* user_data);
} RlmeshModelVtable;

/* An owned model handle. It is not a concurrency primitive: run at most one
 * rlmesh_model_run_local / rlmesh_model_serve on a handle at a time, and free it
 * only from a thread that is not executing it. rlmesh_model_cancel is the only
 * call that may overlap a running one, from any thread. */
typedef struct RlmeshModel RlmeshModel;

RLMESH_API RlmeshStatus rlmesh_model_new(const RlmeshModelVtable* vtable, void* user_data,
                                         RlmeshModel** out);

/* Run options for rlmesh_model_run_local. NULL == defaults (run until the env
 * ends, unseeded, no caps). Every field is "unset" at 0 / NULL. */
typedef struct RlmeshRunOptions {
  uint64_t max_episodes;        /* 0 = until the env ends */
  bool seeded;                  /* whether base_seed is set */
  int64_t base_seed;            /* seed for episode 0; later episodes derive from it */
  int64_t max_episode_steps;    /* truncate an episode after this many steps; 0 = unset */
  double max_episode_seconds;   /* truncate an episode after this long; 0 = unset */
  uint32_t execution_horizon;   /* actions per predicted chunk to execute; 0/1 = no chunking */
  bool close_env;               /* ask the env to close when the run ends */
  const int64_t* episode_seeds; /* explicit per-episode seeds (overrides base_seed); NULL = unset */
  size_t num_episode_seeds;     /* length of episode_seeds; 0 = unset */
} RlmeshRunOptions;

/* What a finished run reports. Plain scalars: copy what you need. */
typedef struct RlmeshRunReport {
  int64_t total_episodes;      /* episodes completed during the run */
  int64_t total_steps;         /* env steps across the whole run */
  double total_reward;         /* summed episode reward */
  double mean_reward;          /* mean episode reward (0 if none completed) */
  int64_t terminated_episodes; /* completed episodes that ended terminated */
  int64_t truncated_episodes;  /* completed episodes that hit a step/time cap */
} RlmeshRunReport;

/* Drive the model against the env at `env_address` (tcp://host:port,
 * host:port, or unix:///path). Blocking — returns when the run ends.
 * `out_report` may be NULL; it is written only on RLMESH_OK. */
RLMESH_API RlmeshStatus rlmesh_model_run_local(RlmeshModel* model, const char* env_address,
                                               const RlmeshRunOptions* options,
                                               RlmeshRunReport* out_report);

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
 * or unix:///path). Blocking — returns when the server stops: a remote shutdown
 * request, an idle timeout, or rlmesh_model_cancel. The same vtable backs every
 * predict, exactly as rlmesh_model_run_local. `options` may be NULL for
 * defaults. */
RLMESH_API RlmeshStatus rlmesh_model_serve(RlmeshModel* model, const char* bind_address,
                                           const RlmeshServeOptions* options);

/* Stop a blocking rlmesh_model_run_local / rlmesh_model_serve on `model`. Call
 * it from another thread (a signal handler's worker, a UI thread); it returns
 * immediately and the blocked call unwinds shortly after. NULL is a no-op.
 *
 * Cancellation is terminal for the handle: a cancelled model refuses further
 * runs. A cancelled serve returns RLMESH_OK after running on_close; a cancelled
 * run_local returns an error naming the cancellation (it has no report to give). */
RLMESH_API void rlmesh_model_cancel(RlmeshModel* model);

/* Free a model handle. NULL is a no-op. Must NOT be called from inside one of
 * this model's own callbacks (it drops the runtime the callback is running on);
 * doing so is contained rather than fatal, but leaves the handle undefined. */
RLMESH_API void rlmesh_model_free(RlmeshModel* model);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RLMESH_H */
