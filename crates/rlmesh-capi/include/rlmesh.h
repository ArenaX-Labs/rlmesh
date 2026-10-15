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
 * an ABI break. Appending a struct_size-guarded vtable/config field is NOT a
 * break. */
#define RLMESH_ABI_VERSION 5

RLMESH_API uint32_t rlmesh_abi_version(void);

/* Nonzero when the linked library's ABI generation matches the one this header
 * was compiled against. SONAME-linked consumers are already gated by the loader
 * (librlmesh_capi.so.N); this is for dlopen / raw-path consumers that bypass it. */
static inline int rlmesh_abi_check(void) { return RLMESH_ABI_VERSION == rlmesh_abi_version(); }

/* This header's package version, SemVer (Python's __version__ is the PEP 440
 * spelling of the same release). rlmesh_build_info() reports the linked
 * library's. */
#define RLMESH_VERSION "0.1.0"

/* Signals: creating a model or env handle sets SIGPIPE to ignored (POSIX), unless
 * the host already installed a handler for it. The transport's socket writes
 * cannot opt out of SIGPIPE, and its default action would kill the host when a
 * peer disconnects mid-write; with it ignored, that write fails with EPIPE. */

/* ---- status + errors ---------------------------------------------------- */

/* Each nonzero status is one exception class of the Python package; a class
 * without a status of its own (RECOVERABLE_ENVIRONMENT_EXCEPTION,
 * ADAPTER_RESOLUTION_ERROR) returns its parent's, and rlmesh_last_error_class()
 * reports the exact class. Values 0-8 and 99 keep their original numbers. */
typedef enum RLMeshStatus {
  RLMESH_OK = 0,
  RLMESH_ERR_INVALID_ARGUMENT = 1, /* ValueError, AdapterResolutionError */
  RLMESH_ERR_INVALID_VALUE = 2,    /* deprecated: never returned; see RLMESH_ERR_TYPE */
  RLMESH_ERR_ENVIRONMENT = 3,      /* (Recoverable)EnvironmentException */
  RLMESH_ERR_MODEL = 4,            /* ModelException */
  RLMESH_ERR_CONNECTION = 5,       /* ConnectionError */
  /* Deprecated alias of RLMESH_ERR_CONNECTION. */
  RLMESH_ERR_TRANSPORT = RLMESH_ERR_CONNECTION,
  RLMESH_ERR_TIMEOUT = 6,   /* TimeoutError */
  RLMESH_ERR_PANIC = 7,     /* a Rust panic; class RLMeshException */
  RLMESH_ERR_CANCELLED = 8, /* KeyboardInterrupt */
  RLMESH_ERR_PROTOCOL = 9,  /* ProtocolException */
  RLMESH_ERR_TYPE = 10,     /* TypeError */
  RLMESH_ERR_RUNTIME = 11,  /* RuntimeError: misuse, local refusals */
  RLMESH_ERR_OS = 12,       /* OSError */
  RLMESH_ERR_INTERNAL = 99  /* RLMeshException */
} RLMeshStatus;

/* The exact class of an error, spelled as Python spells it. */
typedef enum RLMeshErrorClass {
  RLMESH_ERROR_CLASS_INVALID = 0,
  RLMESH_ERROR_CLASS_RLMESH_EXCEPTION = 1,
  RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION = 2,
  RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION = 3,
  RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION = 4,
  RLMESH_ERROR_CLASS_MODEL_EXCEPTION = 5,
  RLMESH_ERROR_CLASS_VALUE_ERROR = 6,
  RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR = 7,
  RLMESH_ERROR_CLASS_TYPE_ERROR = 8,
  RLMESH_ERROR_CLASS_RUNTIME_ERROR = 9,
  RLMESH_ERROR_CLASS_OS_ERROR = 10,
  RLMESH_ERROR_CLASS_CONNECTION_ERROR = 11,
  RLMESH_ERROR_CLASS_TIMEOUT_ERROR = 12,
  RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT = 13
} RLMeshErrorClass;

/* An environment error's code (the wire numbers). */
typedef enum RLMeshErrorCode {
  RLMESH_ERROR_CODE_UNSPECIFIED = 0,
  RLMESH_ERROR_CODE_TIMEOUT = 1,
  RLMESH_ERROR_CODE_INVALID_ACTION = 2,
  RLMESH_ERROR_CODE_NOT_READY = 3,
  RLMESH_ERROR_CODE_BUSY = 4,
  RLMESH_ERROR_CODE_UNSUPPORTED = 5,
  RLMESH_ERROR_CODE_INTERNAL = 10,
  RLMESH_ERROR_CODE_CRASHED = 11,
  RLMESH_ERROR_CODE_CANCELLED = 12,
  RLMESH_ERROR_CODE_CLOSED = 13
} RLMeshErrorCode;

/* A model error's code (the wire numbers). */
typedef enum RLMeshModelErrorCode {
  RLMESH_MODEL_ERROR_CODE_UNSPECIFIED = 0,
  RLMESH_MODEL_ERROR_CODE_INVALID_REQUEST = 2,
  RLMESH_MODEL_ERROR_CODE_NOT_CONFIGURED = 3,
  RLMESH_MODEL_ERROR_CODE_BUSY = 4,
  RLMESH_MODEL_ERROR_CODE_UNSUPPORTED = 5,
  RLMESH_MODEL_ERROR_CODE_INTERNAL = 10,
  RLMESH_MODEL_ERROR_CODE_CANCELLED = 12,
  RLMESH_MODEL_ERROR_CODE_CLOSED = 13
} RLMeshModelErrorCode;

/* Most recent failing call's message on THIS thread (valid until the next
 * RLMesh call on this thread; NULL if none). Read only after a nonzero status.
 * The message is Python's str(exc): "[CODE] message" for an environment or
 * model error, "operation timed out after 5s" for a timeout. The recoverable
 * flag is nonzero only for a recoverable environment or model error. */
RLMESH_API const char* rlmesh_last_error_message(void);
RLMESH_API int rlmesh_last_error_is_recoverable(void);

/* The status that same call recorded — RLMESH_OK when none is. Redundant after
 * a status-returning call (it returned the same code); it is the only status
 * channel the pointer-returning calls have, since they report failure by
 * returning NULL. */
RLMESH_API RLMeshStatus rlmesh_last_error_status(void);

/* The exact class that same call recorded; RLMESH_ERROR_CLASS_INVALID when
 * none is. */
RLMESH_API RLMeshErrorClass rlmesh_last_error_class(void);

/* The last error's code: writes the wire number of an RLMeshErrorCode (an
 * environment class) or an RLMeshModelErrorCode (the model class) to `*out`
 * and returns true; returns false with `*out` untouched for any other class.
 * rlmesh_last_error_code_str() is the code's text ("BUSY"), or NULL with no
 * code; valid until the next RLMesh call on this thread. */
RLMESH_API bool rlmesh_last_error_code(int32_t* out);
RLMESH_API const char* rlmesh_last_error_code_str(void);

/* Static text of a code ("INVALID_ACTION"), or NULL for an unknown value. */
RLMESH_API const char* rlmesh_error_code_str(RLMeshErrorCode code);
RLMESH_API const char* rlmesh_model_error_code_str(RLMeshModelErrorCode code);

/* Python's name of a class ("ConnectionError"), static; NULL for INVALID or an
 * unknown value. rlmesh_error_class_is_subclass is Python's
 * issubclass(cls, base): reflexive, following Python's inheritance, so
 * CONNECTION_ERROR is a subclass of OS_ERROR and the builtins are not
 * subclasses of RLMESH_EXCEPTION. */
RLMESH_API const char* rlmesh_error_class_name(RLMeshErrorClass cls);
RLMESH_API bool rlmesh_error_class_is_subclass(RLMeshErrorClass cls, RLMeshErrorClass base);

/* ---- build info, editions, warnings -------------------------------------- */

/* Where a build's cohort came from: Python's BuildInfo.build_source. */
typedef enum RLMeshBuildSource {
  RLMESH_BUILD_SOURCE_INVALID = 0,
  RLMESH_BUILD_SOURCE_RELEASE = 1, /* "release" */
  RLMESH_BUILD_SOURCE_PACKAGE = 2, /* "package" */
  RLMESH_BUILD_SOURCE_GIT = 3      /* "git" */
} RLMeshBuildSource;

/* Python's spelling ("release"), static; NULL for INVALID or an unknown value. */
RLMESH_API const char* rlmesh_build_source_str(RLMeshBuildSource source);

/* The linked library's identity: Python's rlmesh.build_info(). Every string is
 * static, NUL-terminated UTF-8 with process lifetime; nothing is freed.
 * supported_workflow_editions lists supported_workflow_editions_len editions,
 * newest first; git is NULL when the build was not stamped from git. */
typedef struct RLMeshBuildInfo {
  size_t struct_size;
  const char* version;
  const char* protocol_generation;
  const char* workflow_edition;      /* the exact spelling this build advertises */
  const char* workflow_edition_base; /* its bare YYYY.MM base */
  const char* const* supported_workflow_editions;
  size_t supported_workflow_editions_len;
  const char* build_cohort;
  RLMeshBuildSource build_source;
  uint32_t _reserved;
  const char* git;
} RLMeshBuildInfo;
#define RLMESH_BUILD_INFO_INIT                                         \
  {sizeof(RLMeshBuildInfo),     NULL, NULL, NULL, NULL, NULL, 0, NULL, \
   RLMESH_BUILD_SOURCE_INVALID, 0,    NULL}
#define RLMESH_BUILD_INFO_MIN_SIZE (offsetof(RLMeshBuildInfo, git) + sizeof(const char*))

/* Fill `*out` (initialized with RLMESH_BUILD_INFO_INIT). */
RLMESH_API RLMeshStatus rlmesh_build_info(RLMeshBuildInfo* out);
struct RLMeshBytes; /* defined below */
/* Python's repr(build_info()) into `out` (UTF-8, not NUL-terminated; free with
 * rlmesh_bytes_free). */
RLMESH_API RLMeshStatus rlmesh_build_info_repr(struct RLMeshBytes* out);

/* The bare YYYY.MM workflow edition this build runs at (static): the value to
 * declare. A served model or env resolves its edition from
 * RLMeshExtLegacyServeOptions.workflow_edition, else the RLMESH_WORKFLOW_EDITION
 * environment variable (set but blank: float deliberately, quietly); one that
 * declares neither floats to the newest edition and warns once per process. */
RLMESH_API const char* rlmesh_current_workflow_edition(void);

/* The describe envelope's metadata key and schema version, and the reset
 * option key an env declares its reset options under. */
#define RLMESH_DESCRIBE_METADATA_KEY "rlmesh.describe.v1"
#define RLMESH_DESCRIBE_SCHEMA_VERSION 1
#define RLMESH_ENV_RESET_OPTIONS_KEY "rlmesh.env.v1.reset_options"

/* A warning's category: Python's warning classes. */
typedef enum RLMeshWarningCategory {
  RLMESH_WARNING_CATEGORY_INVALID = 0,
  RLMESH_WARNING_CATEGORY_USER_WARNING = 1,
  RLMESH_WARNING_CATEGORY_RUNTIME_WARNING = 2,
  RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING = 3 /* a USER_WARNING */
} RLMeshWarningCategory;

/* Python's class name ("UserWarning"), static; NULL for INVALID or an unknown
 * value. rlmesh_warning_category_is_subclass is Python's issubclass. */
RLMESH_API const char* rlmesh_warning_category_name(RLMeshWarningCategory category);
RLMESH_API bool rlmesh_warning_category_is_subclass(RLMeshWarningCategory cls,
                                                    RLMeshWarningCategory base);

/* A warning handler: `message` is NUL-terminated UTF-8, borrowed for the call.
 * Return RLMESH_OK, or nonzero (after rlmesh_callback_set_error for the text)
 * to fail the call that warned with that error. Called on the thread that
 * warned; it must be thread-safe. */
typedef RLMeshStatus (*RLMeshWarningHandlerFn)(void* user_data, RLMeshWarningCategory category,
                                               const char* message);

/* Install the process-wide warning handler; a NULL `fn` restores the default,
 * one line to stderr ("rlmesh: <Category>: <message>") per distinct warning.
 * A NULL `fn` with a non-NULL `user_data` or `free_user_data` is
 * RLMESH_ERR_INVALID_ARGUMENT and keeps the installed handler. On RLMESH_OK the
 * library owns `user_data` and calls `free_user_data` (NULL = none) once,
 * after the handler is replaced and every call of it has returned. */
RLMESH_API RLMeshStatus rlmesh_set_warning_handler(RLMeshWarningHandlerFn fn, void* user_data,
                                                   void (*free_user_data)(void*));

/* ---- dtype + tensor ----------------------------------------------------- */

/* An element dtype: Python's dtype strings. The numbers are the wire's.
 * RLMESH_DTYPE_OTHER exists only for Adapters' tolerant dtype fields; every
 * spaces input refuses it, as it refuses INVALID and unknown values, and no
 * spaces output returns it. */
typedef enum RLMeshDType {
  RLMESH_DTYPE_INVALID = 0,
  RLMESH_DTYPE_BOOL = 1,
  RLMESH_DTYPE_UINT8 = 2,
  RLMESH_DTYPE_UINT16 = 3,
  RLMESH_DTYPE_UINT32 = 4,
  RLMESH_DTYPE_UINT64 = 5,
  RLMESH_DTYPE_INT8 = 6,
  RLMESH_DTYPE_INT16 = 7,
  RLMESH_DTYPE_INT32 = 8,
  RLMESH_DTYPE_INT64 = 9,
  RLMESH_DTYPE_FLOAT16 = 10,
  RLMESH_DTYPE_FLOAT32 = 11,
  RLMESH_DTYPE_FLOAT64 = 12,
  RLMESH_DTYPE_OTHER = 1000
} RLMeshDType;

/* Python's spelling ("float32"), static storage; NULL for INVALID, OTHER or an
 * unknown value. */
RLMESH_API const char* rlmesh_dtype_str(RLMeshDType dtype);
/* Byte size of one element; 0 for INVALID, OTHER or an unknown value. */
RLMESH_API size_t rlmesh_ext_dtype_size(RLMeshDType dtype);

/* A space's kind (Python's `kind`); output only. UNKNOWN is a spec with no
 * payload and what a NULL handle reads. */
typedef enum RLMeshSpaceKind {
  RLMESH_SPACE_KIND_UNKNOWN = 0,
  RLMESH_SPACE_KIND_BOX = 1,
  RLMESH_SPACE_KIND_DISCRETE = 2,
  RLMESH_SPACE_KIND_MULTI_BINARY = 3,
  RLMESH_SPACE_KIND_MULTI_DISCRETE = 4,
  RLMESH_SPACE_KIND_TEXT = 5,
  RLMESH_SPACE_KIND_DICT = 10,
  RLMESH_SPACE_KIND_TUPLE = 11
} RLMeshSpaceKind;

typedef struct RLMeshStr {
  const char* data;
  size_t len;
} RLMeshStr;

/* Wrap a NUL-terminated string (NULL gives the empty text). */
static inline RLMeshStr rlmesh_str(const char* s) {
  RLMeshStr out;
  size_t len = 0;
  if (s != NULL) {
    while (s[len] != '\0') {
      ++len;
    }
  }
  out.data = s;
  out.len = len;
  return out;
}

typedef enum RLMeshDevice { RLMESH_DEVICE_INVALID = 0, RLMESH_DEVICE_CPU = 1 } RLMeshDevice;
RLMESH_API const char* rlmesh_device_str(RLMeshDevice device);

/* Opaque immutable tensor. Construction copies; clone and reshape share;
 * copy creates independent storage. Strides are in bytes. */
typedef struct RLMeshTensor RLMeshTensor;
RLMESH_API RLMeshStatus rlmesh_tensor_new(const void* buffer, size_t buffer_len,
                                          const size_t* shape, size_t ndim, RLMeshDType dtype,
                                          RLMeshTensor** out);
RLMESH_API RLMeshTensor* rlmesh_tensor_clone(const RLMeshTensor* tensor);
RLMESH_API void rlmesh_tensor_free(RLMeshTensor* tensor);
RLMESH_API RLMeshDType rlmesh_tensor_dtype(const RLMeshTensor* tensor);
RLMESH_API RLMeshDevice rlmesh_tensor_device(const RLMeshTensor* tensor);
RLMESH_API RLMeshStatus rlmesh_tensor_shape(const RLMeshTensor* tensor, size_t* out, size_t cap,
                                            size_t* out_len);
RLMESH_API RLMeshStatus rlmesh_tensor_strides(const RLMeshTensor* tensor, size_t* out, size_t cap,
                                              size_t* out_len);
RLMESH_API RLMeshStatus rlmesh_tensor_ndim(const RLMeshTensor* tensor, size_t* out);
RLMESH_API RLMeshStatus rlmesh_tensor_size(const RLMeshTensor* tensor, size_t* out);
RLMESH_API RLMeshStatus rlmesh_tensor_nbytes(const RLMeshTensor* tensor, size_t* out);
RLMESH_API RLMeshStatus rlmesh_tensor_buffer(const RLMeshTensor* tensor, const void** out_data,
                                             size_t* out_len);
RLMESH_API bool rlmesh_tensor_is_contiguous(const RLMeshTensor* tensor);
RLMESH_API RLMeshStatus rlmesh_tensor_reshape(const RLMeshTensor* tensor, const int64_t* shape,
                                              size_t ndim, RLMeshTensor** out);
RLMESH_API RLMeshStatus rlmesh_tensor_copy(const RLMeshTensor* tensor, RLMeshTensor** out);
RLMESH_API RLMeshStatus rlmesh_tensor_tobytes(const RLMeshTensor* tensor, struct RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_tensor_repr(const RLMeshTensor* tensor, struct RLMeshBytes* out);

/* Core Value, Metadata and InfoDict. Handles are immutable; clones share.
 * A dict preserves insertion order, replacing duplicate keys in place.
 * Composite constructors retain children: callers still free their handles.
 * Borrowed reads live as long as the value. Missing keys/indices succeed with
 * NULL (or RLMeshStr {NULL, 0}); a present empty string has non-NULL data.
 * A read of the wrong kind returns RLMESH_ERR_TYPE. */
typedef struct RLMeshValue RLMeshValue;
typedef enum RLMeshValueKind {
  RLMESH_VALUE_INVALID = 0,
  RLMESH_VALUE_NONE = 1,
  RLMESH_VALUE_BOOL = 2,
  RLMESH_VALUE_INT = 3,
  RLMESH_VALUE_FLOAT = 4,
  RLMESH_VALUE_TEXT = 5,
  RLMESH_VALUE_BYTES = 6,
  RLMESH_VALUE_TENSOR = 7,
  RLMESH_VALUE_LIST = 8,
  RLMESH_VALUE_TUPLE = 9,
  RLMESH_VALUE_DICT = 10
} RLMeshValueKind;
RLMESH_API RLMeshValueKind rlmesh_value_kind(const RLMeshValue* value);
RLMESH_API RLMeshValue* rlmesh_value_clone(const RLMeshValue* value);
RLMESH_API bool rlmesh_value_equals(const RLMeshValue* a, const RLMeshValue* b);
RLMESH_API void rlmesh_value_free(RLMeshValue* value);
RLMESH_API RLMeshStatus rlmesh_value_none(RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_bool(bool v, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_int(int64_t v, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_float(double v, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_text(RLMeshStr v, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_bytes(const uint8_t* data, size_t len, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_tensor(const RLMeshTensor* tensor, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_list(const RLMeshValue* const* items, size_t n,
                                          RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_tuple(const RLMeshValue* const* items, size_t n,
                                           RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_dict(const RLMeshStr* keys, const RLMeshValue* const* values,
                                          size_t n, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_as_bool(const RLMeshValue* value, bool* out);
RLMESH_API RLMeshStatus rlmesh_value_as_int(const RLMeshValue* value, int64_t* out);
RLMESH_API RLMeshStatus rlmesh_value_as_float(const RLMeshValue* value, double* out);
RLMESH_API RLMeshStatus rlmesh_value_as_text(const RLMeshValue* value, RLMeshStr* out);
RLMESH_API RLMeshStatus rlmesh_value_as_bytes(const RLMeshValue* value, const uint8_t** out_data,
                                              size_t* out_len);
/* Unlike other reads, as_tensor returns an owned shared reference. */
RLMESH_API RLMeshStatus rlmesh_value_as_tensor(const RLMeshValue* value, RLMeshTensor** out);
RLMESH_API RLMeshStatus rlmesh_value_len(const RLMeshValue* value, size_t* out);
RLMESH_API RLMeshStatus rlmesh_value_get(const RLMeshValue* value, size_t i,
                                         const RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_dict_key_at(const RLMeshValue* value, size_t i,
                                                 RLMeshStr* out);
RLMESH_API RLMeshStatus rlmesh_value_dict_value_at(const RLMeshValue* value, size_t i,
                                                   const RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_dict_get(const RLMeshValue* value, RLMeshStr key,
                                              const RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_value_dict_contains_key(const RLMeshValue* value, RLMeshStr key,
                                                       bool* out);

/* ---- spaces + contract ------------------------------------------------- */

typedef struct RLMeshSpaceSpec RLMeshSpaceSpec;
typedef struct RLMeshEnvContract RLMeshEnvContract;

RLMESH_API const RLMeshSpaceSpec* rlmesh_env_contract_observation_space(
    const RLMeshEnvContract* contract);
RLMESH_API const RLMeshSpaceSpec* rlmesh_env_contract_action_space(
    const RLMeshEnvContract* contract);
RLMESH_API uint64_t rlmesh_env_contract_num_envs(const RLMeshEnvContract* contract);

/* The full immutable contract surface. Borrowed spaces/metadata stay valid
 * while any contract reference lives. Text outputs are owned RLMeshBytes. */
RLMESH_API RLMeshStatus rlmesh_env_contract_id(const RLMeshEnvContract* contract,
                                               struct RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_env_contract_render_mode(const RLMeshEnvContract* contract,
                                                        bool* out_has, struct RLMeshBytes* out);
RLMESH_API const RLMeshValue* rlmesh_env_contract_metadata(const RLMeshEnvContract* contract);
RLMESH_API RLMeshEnvContract* rlmesh_env_contract_clone(const RLMeshEnvContract* contract);
RLMESH_API RLMeshEnvContract* rlmesh_env_contract_env(const RLMeshEnvContract* contract);
RLMESH_API RLMeshEnvContract* rlmesh_env_contract_spec(const RLMeshEnvContract* contract);
RLMESH_API void rlmesh_env_contract_free(RLMeshEnvContract* contract);
RLMESH_API RLMeshStatus rlmesh_env_contract_to_dict(const RLMeshEnvContract* contract,
                                                    RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_env_contract_repr(const RLMeshEnvContract* contract,
                                                 struct RLMeshBytes* out);
/* Typed reset options; NULL options/lane mean None. */
RLMESH_API RLMeshStatus rlmesh_trial_index(const RLMeshValue* options, const size_t* lane,
                                           bool* out_has, int64_t* out);

/* Space introspection — enough to build a valid zero (or random) action for any
 * space: shape + dtype + bounds for Box, n/start for Discrete, the length limits
 * for Text, the per-element category counts for MultiDiscrete, and a child walk
 * for Dict/Tuple. RLMESH_SPACE_KIND_UNKNOWN when `spec` is NULL or unspecified. */
RLMESH_API RLMeshSpaceKind rlmesh_space_type(const RLMeshSpaceSpec* spec);
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
 * NUL-terminated, valid while `spec` lives. A Value dict preserves insertion order;
 * pair a space child with a value child by key. */
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

/* Deprecated space builders: use rlmesh_box_new, rlmesh_discrete_new,
 * rlmesh_dict_new and rlmesh_ext_box_elementwise_new below, and lend their spec
 * with rlmesh_space_spec(). Each returns an OWNED spec, or NULL on error (detail
 * in rlmesh_last_error_message()); free it with rlmesh_space_spec_free unless a
 * composite builder adopted it. A float Box takes +-INFINITY for an
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
 * rlmesh_value_dict. Dict keys must be non-empty and unique; the built space
 * stores them sorted by key, whatever order they are passed in. */
RLMESH_API RLMeshSpaceSpec* rlmesh_space_dict(const char* const* keys,
                                              RLMeshSpaceSpec* const* children, size_t n);
RLMESH_API RLMeshSpaceSpec* rlmesh_space_tuple(RLMeshSpaceSpec* const* children, size_t n);
/* Free an owned spec (a builder's or rlmesh_space_spec_clone's). Not for a
 * borrowed contract space, child or rlmesh_space_spec() result. NULL is a no-op. */
RLMESH_API void rlmesh_space_spec_free(RLMeshSpaceSpec* spec);

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

/* ---- spaces (Python's rlmesh.spaces) ------------------------------------ */

/* UTF-8 text with its length. Inputs may hold NUL; `data` may be NULL only when
 * `len` is 0. Outputs are borrowed from their parent and NOT NUL-terminated. */
/* How a Box stores its bounds (Python's `bounds_kind`). */
typedef enum RLMeshBoundsKind {
  RLMESH_BOUNDS_KIND_INVALID = 0,
  RLMESH_BOUNDS_KIND_UNBOUNDED = 1,
  RLMESH_BOUNDS_KIND_UNIFORM = 2,
  RLMESH_BOUNDS_KIND_ELEMENTWISE = 3,
  RLMESH_BOUNDS_KIND_TYPED_UNIFORM = 4,
  RLMESH_BOUNDS_KIND_TYPED_ELEMENTWISE = 5
} RLMeshBoundsKind;

/* Python's spellings ("multi_binary", "typed_uniform"), static storage; NULL
 * for an unknown value (and for RLMESH_BOUNDS_KIND_INVALID). */
RLMESH_API const char* rlmesh_space_kind_str(RLMeshSpaceKind kind);
RLMESH_API const char* rlmesh_bounds_kind_str(RLMeshBoundsKind kind);

/* One Box bound, exact in its domain: read `value` per `type_`. INVALID is
 * refused on input. Build one with the rlmesh_bound_* helpers. */
typedef enum RLMeshBoundType {
  RLMESH_BOUND_TYPE_INVALID = 0,
  RLMESH_BOUND_TYPE_BOOL = 1,
  RLMESH_BOUND_TYPE_INT = 2,
  RLMESH_BOUND_TYPE_UINT = 3,
  RLMESH_BOUND_TYPE_FLOAT = 4
} RLMeshBoundType;

typedef struct RLMeshBound {
  RLMeshBoundType type_;
  union {
    bool b;
    int64_t i;
    uint64_t u;
    double f; /* +-INFINITY is Python's "inf"; NaN is refused */
  } value;
} RLMeshBound;

static inline RLMeshBound rlmesh_bound_bool(bool v) {
  RLMeshBound out;
  out.value.u = 0;
  out.type_ = RLMESH_BOUND_TYPE_BOOL;
  out.value.b = v;
  return out;
}
static inline RLMeshBound rlmesh_bound_int(int64_t v) {
  RLMeshBound out;
  out.type_ = RLMESH_BOUND_TYPE_INT;
  out.value.i = v;
  return out;
}
static inline RLMeshBound rlmesh_bound_uint(uint64_t v) {
  RLMeshBound out;
  out.type_ = RLMESH_BOUND_TYPE_UINT;
  out.value.u = v;
  return out;
}
static inline RLMeshBound rlmesh_bound_float(double v) {
  RLMeshBound out;
  out.type_ = RLMESH_BOUND_TYPE_FLOAT;
  out.value.f = v;
  return out;
}

/* A space: a spec plus its sampler. RLMeshBox, RLMeshDiscrete and RLMeshDict
 * are the same handle (pass a Dict to rlmesh_space_* without a cast); clone and
 * free every kind with rlmesh_space_clone / rlmesh_space_free. A typed accessor
 * (rlmesh_box_low, rlmesh_discrete_n, rlmesh_dict_get, ...) on a handle of
 * another kind returns RLMESH_ERR_TYPE; check rlmesh_space_kind first. */
typedef struct RLMeshSpace RLMeshSpace;
typedef RLMeshSpace RLMeshBox;
typedef RLMeshSpace RLMeshDiscrete;
typedef RLMeshSpace RLMeshDict;

/* Buffer rule for every (out, cap, out_len) call below: *out_len always gets
 * the full length; out == NULL with cap == 0 is a length query (RLMESH_OK); a
 * short cap returns RLMESH_ERR_INVALID_ARGUMENT and writes nothing. */

/* SpaceSpec: immutable data. An owned copy is freed with rlmesh_space_spec_free. */
RLMESH_API RLMeshSpaceSpec* rlmesh_space_spec_clone(const RLMeshSpaceSpec* spec);
RLMESH_API RLMeshSpaceKind rlmesh_space_spec_kind(const RLMeshSpaceSpec* spec);
RLMESH_API RLMeshStatus rlmesh_space_spec_shape(const RLMeshSpaceSpec* spec, int64_t* out,
                                                size_t cap, size_t* out_len);
/* A Dict reads RLMESH_DTYPE_FLOAT32, as in Python; INVALID for NULL. */
RLMESH_API RLMeshDType rlmesh_space_spec_dtype(const RLMeshSpaceSpec* spec);
/* A fresh space with its own sampler; never fails for a valid spec. */
RLMESH_API RLMeshStatus rlmesh_space_spec_to_space(const RLMeshSpaceSpec* spec, RLMeshSpace** out);
/* Gymnasium-style text, Python's repr; free with rlmesh_bytes_free. */
RLMESH_API RLMeshStatus rlmesh_space_spec_repr(const RLMeshSpaceSpec* spec, RLMeshBytes* out);
/* Python's space_from_spec: refuses a spec with no payload. */
RLMESH_API RLMeshStatus rlmesh_spaces_space_from_spec(const RLMeshSpaceSpec* spec,
                                                      RLMeshSpace** out);

/* Space(spec), clone (shares the sampler), free. rlmesh_space_free also frees
 * an owned spec, as it did before RLMeshSpace existed; new code frees specs with
 * rlmesh_space_spec_free. */
RLMESH_API RLMeshStatus rlmesh_space_new(const RLMeshSpaceSpec* spec, RLMeshSpace** out);
RLMESH_API RLMeshSpace* rlmesh_space_clone(const RLMeshSpace* space);
RLMESH_API void rlmesh_space_free(RLMeshSpace* space);
/* The space's spec, borrowed while any handle sharing the space lives. */
RLMESH_API const RLMeshSpaceSpec* rlmesh_space_spec(const RLMeshSpace* space);
RLMESH_API RLMeshSpaceKind rlmesh_space_kind(const RLMeshSpace* space);
RLMESH_API RLMeshStatus rlmesh_space_shape(const RLMeshSpace* space, int64_t* out, size_t cap,
                                           size_t* out_len);
/* seed(seed): `seed` NULL draws one from OS entropy; the seed applied goes to
 * *out_seed (may be NULL). Restarts this space's stream only (a Dict's children
 * keep their own). */
RLMESH_API RLMeshStatus rlmesh_space_seed(const RLMeshSpace* space, const uint64_t* seed,
                                          uint64_t* out_seed);
/* An owned value; free with rlmesh_value_free. */
RLMESH_API RLMeshStatus rlmesh_space_sample(const RLMeshSpace* space, RLMeshValue** out);
/* Membership; never fails (false for NULL). */
RLMESH_API bool rlmesh_space_contains(const RLMeshSpace* space, const RLMeshValue* value);
/* Python's == (normalized specs; a Dict's key order does not matter). */
RLMESH_API bool rlmesh_space_equals(const RLMeshSpace* a, const RLMeshSpace* b);
RLMESH_API RLMeshStatus rlmesh_space_repr(const RLMeshSpace* space, RLMeshBytes* out);

/* Box(low, high, shape, dtype). (-inf, inf) is unbounded for any dtype; on an
 * integer or bool dtype finite bounds are stored exactly and must be whole
 * numbers (0.5 is RLMESH_ERR_INVALID_ARGUMENT); NaN is refused. */
typedef struct RLMeshBoxOptions {
  size_t struct_size;
  RLMeshDType dtype; /* RLMESH_DTYPE_FLOAT32 */
  uint32_t _reserved;
} RLMeshBoxOptions;
#define RLMESH_BOX_OPTIONS_INIT {sizeof(RLMeshBoxOptions), RLMESH_DTYPE_FLOAT32, 0}
#define RLMESH_BOX_OPTIONS_MIN_SIZE (offsetof(RLMeshBoxOptions, _reserved) + sizeof(uint32_t))

RLMESH_API RLMeshStatus rlmesh_box_new(RLMeshBound low, RLMeshBound high, const int64_t* shape,
                                       size_t ndim,
                                       const RLMeshBoxOptions* options /* NULL = INIT */,
                                       RLMeshBox** out);
/* Box(spec): a spec of another kind is RLMESH_ERR_INVALID_ARGUMENT. */
RLMESH_API RLMeshStatus rlmesh_box_from_spec(const RLMeshSpaceSpec* spec, RLMeshBox** out);
/* low / high: each entry's type_ is the domain it is stored in (BOOL, UINT or
 * INT for integer bounds, FLOAT otherwise). Whether a bound is None, uniform or
 * per element comes from rlmesh_box_bounds_kind, never from the entry count. */
RLMESH_API RLMeshStatus rlmesh_box_low(const RLMeshBox* box, RLMeshBound* out, size_t cap,
                                       size_t* out_len);
RLMESH_API RLMeshStatus rlmesh_box_high(const RLMeshBox* box, RLMeshBound* out, size_t cap,
                                        size_t* out_len);
/* *out_has is false when the spec carries no bounds. */
RLMESH_API RLMeshStatus rlmesh_box_bounds_kind(const RLMeshBox* box, bool* out_has,
                                               RLMeshBoundsKind* out);
/* C-only extra: a Box with one bound pair per element (`bounds_len` must equal
 * the shape's element count). */
RLMESH_API RLMeshStatus rlmesh_ext_box_elementwise_new(const RLMeshBound* low,
                                                       const RLMeshBound* high, size_t bounds_len,
                                                       const int64_t* shape, size_t ndim,
                                                       const RLMeshBoxOptions* options,
                                                       RLMeshBox** out);

/* Discrete(n, start, dtype): dtype is INT64, INT32 or UINT8. */
typedef struct RLMeshDiscreteOptions {
  size_t struct_size;
  int64_t start;     /* 0 */
  RLMeshDType dtype; /* RLMESH_DTYPE_INT64 */
  uint32_t _reserved;
} RLMeshDiscreteOptions;
#define RLMESH_DISCRETE_OPTIONS_INIT {sizeof(RLMeshDiscreteOptions), 0, RLMESH_DTYPE_INT64, 0}
#define RLMESH_DISCRETE_OPTIONS_MIN_SIZE \
  (offsetof(RLMeshDiscreteOptions, _reserved) + sizeof(uint32_t))

RLMESH_API RLMeshStatus rlmesh_discrete_new(int64_t n,
                                            const RLMeshDiscreteOptions* options /* NULL = INIT */,
                                            RLMeshDiscrete** out);
RLMESH_API RLMeshStatus rlmesh_discrete_from_spec(const RLMeshSpaceSpec* spec,
                                                  RLMeshDiscrete** out);
RLMESH_API RLMeshStatus rlmesh_discrete_n(const RLMeshDiscrete* space, int64_t* out);
RLMESH_API RLMeshStatus rlmesh_discrete_start(const RLMeshDiscrete* space, int64_t* out);

/* Dict(spaces): copies each spec into a fresh child space; the keys sort at
 * construction, as Python's do. Duplicate or empty keys are
 * RLMESH_ERR_INVALID_ARGUMENT. Lend a space's spec with rlmesh_space_spec(). */
RLMESH_API RLMeshStatus rlmesh_dict_new(const RLMeshStr* keys, const RLMeshSpaceSpec* const* specs,
                                        size_t n, RLMeshDict** out);
RLMESH_API RLMeshStatus rlmesh_dict_from_spec(const RLMeshSpaceSpec* spec, RLMeshDict** out);
RLMESH_API RLMeshStatus rlmesh_dict_len(const RLMeshDict* dict, size_t* out);
/* Declaration order; both borrowed while any handle sharing the Dict lives. A
 * child is the same space (and sampler) on every read. An index past the end
 * reads as absent: RLMESH_OK with *out = {NULL, 0} / NULL (a present key's
 * `data` is never NULL). */
RLMESH_API RLMeshStatus rlmesh_dict_key_at(const RLMeshDict* dict, size_t i, RLMeshStr* out);
RLMESH_API RLMeshStatus rlmesh_dict_value_at(const RLMeshDict* dict, size_t i,
                                             const RLMeshSpace** out);
/* d[key], borrowed; *out = NULL (and RLMESH_OK) when absent. */
RLMESH_API RLMeshStatus rlmesh_dict_get(const RLMeshDict* dict, RLMeshStr key,
                                        const RLMeshSpace** out);
/* key in d, into *out; RLMESH_ERR_TYPE on another kind. */
RLMESH_API RLMeshStatus rlmesh_dict_contains_key(const RLMeshDict* dict, RLMeshStr key, bool* out);

/* ---- adapters (experimental) -------------------------------------------- */

/* Resolve the env's tags (env_tags_json; see rlmesh_env_contract_adapter_tags_json)
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
RLMESH_API RLMeshStatus rlmesh_env_contract_adapter_tags_json(const RLMeshEnvContract* contract,
                                                              RLMeshBytes* out);

/* ---- Env authoring and serving (facade) ---------------------------------- */

typedef struct RLMeshEnv RLMeshEnv;
typedef struct RLMeshVectorEnv RLMeshVectorEnv;
typedef struct RLMeshEnvServer RLMeshEnvServer;
typedef struct RLMeshEnvTags RLMeshEnvTags;
/* Adapter authoring belongs to its separate wave. Returned tags are owned. */
RLMESH_API void rlmesh_env_tags_free(RLMeshEnvTags* tags);
typedef enum RLMeshTransport {
  RLMESH_TRANSPORT_INVALID = 0,
  RLMESH_TRANSPORT_TCP = 1,
  RLMESH_TRANSPORT_UNIX = 2
} RLMeshTransport;
RLMESH_API const char* rlmesh_transport_str(RLMeshTransport transport);

/* Every sized input requires at least MIN_SIZE. NULL options means INIT;
 * unknown trailing bytes must be zero. {0} is refused. Explicit reserved
 * fields keep the layout free of implicit padding for later append-only fields. */
typedef struct RLMeshServeOptions {
  size_t struct_size;
  bool allow_remote_shutdown;
  bool has_idle_timeout_seconds;
  bool has_drain_timeout_seconds;
  bool has_close_timeout_seconds;
  bool compress_responses;
  uint8_t _reserved[3];
  double idle_timeout_seconds;
  double drain_timeout_seconds;
  double close_timeout_seconds;
  const char* workflow_edition;
} RLMeshServeOptions;
#define RLMESH_SERVE_OPTIONS_INIT \
  {sizeof(RLMeshServeOptions), false, false, false, false, false, {0, 0, 0}, 0.0, 0.0, 0.0, NULL}
#define RLMESH_SERVE_OPTIONS_MIN_SIZE sizeof(RLMeshServeOptions)

typedef struct RLMeshExtEnvServerOptions {
  size_t struct_size;
  const char* token;
  const RLMeshValue* peer_info_extra;
  const char* host_runtime_json;
  bool publish_describe;
  uint8_t _reserved[sizeof(size_t) - 1];
} RLMeshExtEnvServerOptions;
#define RLMESH_EXT_ENV_SERVER_OPTIONS_INIT \
  {sizeof(RLMeshExtEnvServerOptions), NULL, NULL, RLMESH_HOST_RUNTIME_JSON, false, {0}}
#define RLMESH_EXT_ENV_SERVER_OPTIONS_MIN_SIZE sizeof(RLMeshExtEnvServerOptions)

typedef struct RLMeshEnvServerOptions {
  size_t struct_size;
  const char* address;
  const char* host;
  const char* path;
  const RLMeshServeOptions* options;
  const RLMeshEnvTags* tags;
  const RLMeshExtEnvServerOptions* ext;
  uint16_t port;
  bool has_port;
  bool has_transport;
  RLMeshTransport transport;
  bool close_env_on_shutdown;
  uint8_t _reserved[7];
} RLMeshEnvServerOptions;
#define RLMESH_ENV_SERVER_OPTIONS_INIT                                                  \
  {sizeof(RLMeshEnvServerOptions), NULL, NULL, NULL, NULL, NULL, NULL, 0, false, false, \
   RLMESH_TRANSPORT_INVALID,       true, {0}}
#define RLMESH_ENV_SERVER_OPTIONS_MIN_SIZE sizeof(RLMeshEnvServerOptions)

typedef struct RLMeshEnvAttributes {
  size_t struct_size;
  const RLMeshSpace* observation_space;
  const RLMeshSpace* action_space;
  const char* render_mode;
  const RLMeshValue* metadata;
  const char* id;
} RLMeshEnvAttributes;
#define RLMESH_ENV_ATTRIBUTES_INIT {sizeof(RLMeshEnvAttributes), NULL, NULL, NULL, NULL, NULL}
#define RLMESH_ENV_ATTRIBUTES_MIN_SIZE sizeof(RLMeshEnvAttributes)

typedef struct RLMeshVectorEnvAttributes {
  size_t struct_size;
  uint64_t num_envs;
  const RLMeshSpace* single_observation_space;
  const RLMeshSpace* single_action_space;
  const char* render_mode;
  const RLMeshValue* metadata;
  const char* id;
} RLMeshVectorEnvAttributes;
#define RLMESH_VECTOR_ENV_ATTRIBUTES_INIT \
  {sizeof(RLMeshVectorEnvAttributes), 0, NULL, NULL, NULL, NULL, NULL}
#define RLMESH_VECTOR_ENV_ATTRIBUTES_MIN_SIZE sizeof(RLMeshVectorEnvAttributes)

typedef struct RLMeshEnvVtable {
  size_t struct_size;
  RLMeshStatus (*reset)(void*, const int64_t* seed, const RLMeshValue* options,
                        RLMeshValue** out_observation, RLMeshValue** out_info);
  RLMeshStatus (*step)(void*, const RLMeshValue* action, RLMeshValue** out_observation,
                       double* out_reward, bool* out_terminated, bool* out_truncated,
                       RLMeshValue** out_info);
  RLMeshStatus (*render)(void*, RLMeshValue** out_frame);
  RLMeshStatus (*close)(void*);
  void (*free_user_data)(void*);
} RLMeshEnvVtable;
#define RLMESH_ENV_VTABLE_INIT {sizeof(RLMeshEnvVtable), NULL, NULL, NULL, NULL, NULL}
#define RLMESH_ENV_VTABLE_MIN_SIZE sizeof(RLMeshEnvVtable)

typedef struct RLMeshVectorEnvVtable {
  size_t struct_size;
  RLMeshStatus (*reset)(void*, const int64_t* seed, size_t seed_len, const RLMeshValue* options,
                        RLMeshValue** out_observation, RLMeshValue** out_info);
  RLMeshStatus (*step)(void*, const RLMeshValue* actions, RLMeshValue** out_observation,
                       double* out_rewards, bool* out_terminated, bool* out_truncated,
                       RLMeshValue** out_info);
  RLMeshStatus (*render)(void*, RLMeshValue** out_frame);
  RLMeshStatus (*close)(void*);
  void (*free_user_data)(void*);
} RLMeshVectorEnvVtable;
#define RLMESH_VECTOR_ENV_VTABLE_INIT {sizeof(RLMeshVectorEnvVtable), NULL, NULL, NULL, NULL, NULL}
#define RLMESH_VECTOR_ENV_VTABLE_MIN_SIZE sizeof(RLMeshVectorEnvVtable)

/* Env handles copy vtables/attributes. State transfers only on success and
 * free_user_data runs once after the last library reference. Free never closes;
 * close runs at most once. A handle may be borrowed by one server/run/session
 * at a time. Free a borrowing server before freeing the env. Close on a busy
 * env fails with RLMESH_ERR_RUNTIME. Callback outputs are taken even on failure;
 * NULL info means {}. Callbacks and state must be safe to move between threads. */
RLMESH_API RLMeshStatus rlmesh_env_new(const RLMeshEnvVtable*, void* user_data,
                                       const RLMeshEnvAttributes*, RLMeshEnv** out);
RLMESH_API RLMeshStatus rlmesh_vector_env_new(const RLMeshVectorEnvVtable*, void* user_data,
                                              const RLMeshVectorEnvAttributes*,
                                              RLMeshVectorEnv** out);
RLMESH_API RLMeshStatus rlmesh_env_close(RLMeshEnv*);
RLMESH_API RLMeshStatus rlmesh_vector_env_close(RLMeshVectorEnv*);
RLMESH_API void rlmesh_env_free(RLMeshEnv*);
RLMESH_API void rlmesh_vector_env_free(RLMeshVectorEnv*);

/* New binds. start runs callbacks on library threads; serve keeps lane 0 on
 * the caller. shutdown is allowed from a callback. Free shuts down and drains
 * all C callbacks, including any still running past the facade's timeout. */
RLMESH_API RLMeshStatus rlmesh_env_server_new(RLMeshEnv* const* envs, size_t envs_len,
                                              const RLMeshEnvServerOptions*, RLMeshEnvServer** out);
RLMESH_API RLMeshStatus rlmesh_env_server_new_vector(RLMeshVectorEnv*,
                                                     const RLMeshEnvServerOptions*,
                                                     RLMeshEnvServer** out);
RLMESH_API RLMeshStatus rlmesh_env_server_address(const RLMeshEnvServer*, RLMeshBytes* out);
RLMESH_API const RLMeshEnvContract* rlmesh_env_server_env_contract(const RLMeshEnvServer*);
RLMESH_API RLMeshStatus rlmesh_env_server_start(RLMeshEnvServer*);
RLMESH_API RLMeshStatus rlmesh_env_server_serve(RLMeshEnvServer*);
RLMESH_API RLMeshStatus rlmesh_env_server_wait(RLMeshEnvServer*, const double* timeout,
                                               bool* out_stopped);
RLMESH_API RLMeshStatus rlmesh_env_server_shutdown(RLMeshEnvServer*);
RLMESH_API RLMeshStatus rlmesh_env_server_repr(const RLMeshEnvServer*, RLMeshBytes* out);
RLMESH_API void rlmesh_env_server_free(RLMeshEnvServer*);

/* EnvFactory callbacks run on the calling thread and are exclusive. Attribute
 * getters are concurrent reads of copied attributes. Free never calls close.
 * Each make callback output is owned, even on failure; exactly one env kind
 * must be returned on success. Returned handles remain the caller's to free. */
typedef struct RLMeshEnvFactory RLMeshEnvFactory;
typedef struct RLMeshVariant RLMeshVariant;
typedef struct RLMeshVariantSink RLMeshVariantSink;
typedef struct RLMeshVariantList RLMeshVariantList;
typedef struct RLMeshTagParamList RLMeshTagParamList;
typedef struct RLMeshExtCancelToken RLMeshExtCancelToken;
RLMESH_API RLMeshExtCancelToken* rlmesh_ext_cancel_token_new(void);
RLMESH_API void rlmesh_ext_cancel_token_cancel(const RLMeshExtCancelToken*);
RLMESH_API void rlmesh_ext_cancel_token_free(RLMeshExtCancelToken*);

typedef struct RLMeshEnvFactoryServeOptions {
  size_t struct_size;
  int64_t num_envs;               /* 1; <= 1 serves one env */
  const char* workflow_edition;   /* NULL/""/blank = None */
  const RLMeshValue* make_kwargs; /* DICT; NULL = {} */
  RLMeshExtCancelToken* cancel;   /* NULL = none, borrowed for the call (R-18, ENV-33) */
} RLMeshEnvFactoryServeOptions;
#define RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT \
  {sizeof(RLMeshEnvFactoryServeOptions), 1, NULL, NULL, NULL}

typedef struct RLMeshTagParam {
  size_t struct_size;
  const char* name;            /* required */
  const RLMeshValue* default_; /* required; one of choices */
  const RLMeshValue* choices;  /* LIST, non-empty */
} RLMeshTagParam;
#define RLMESH_TAG_PARAM_INIT {sizeof(RLMeshTagParam), NULL, NULL, NULL}

typedef struct RLMeshEnvFactoryAttributes { /* deep-copied at rlmesh_env_factory_new */
  size_t struct_size;
  const RLMeshEnvTags* tags; /* NULL = None */
  const RLMeshTagParam* const* tag_params;
  size_t tag_params_len; /* () */
  const char* const* reset_options;
  size_t reset_options_len;     /* () */
  const char* workflow_edition; /* NULL = None */
} RLMeshEnvFactoryAttributes;
#define RLMESH_ENV_FACTORY_ATTRIBUTES_INIT \
  {sizeof(RLMeshEnvFactoryAttributes), NULL, NULL, 0, NULL, 0, NULL}

typedef struct RLMeshEnvFactoryVtable {
  size_t struct_size;
  RLMeshStatus (*prepare)(void* user_data); /* NULL = no-op */
  RLMeshStatus (*make)(void* user_data, const RLMeshValue* kwargs /* DICT, discriminants bound */,
                       RLMeshEnv** out_env,
                       RLMeshVectorEnv** out_vector_env); /* required; exactly one set */
  RLMeshStatus (*tags_for)(void* user_data, const RLMeshValue* params,
                           RLMeshEnvTags** out); /* NULL = tags */
  RLMeshStatus (*enumerate_variants)(void* user_data,
                                     RLMeshVariantSink* sink); /* NULL = not defined */
  RLMeshStatus (*close)(void* user_data);                      /* NULL = no-op */
  void (*free_user_data)(void* user_data);                     /* NULL = none */
} RLMeshEnvFactoryVtable;
#define RLMESH_ENV_FACTORY_VTABLE_INIT \
  {sizeof(RLMeshEnvFactoryVtable), NULL, NULL, NULL, NULL, NULL, NULL}

/* Factory. new runs the branch-table check; attributes NULL = INIT (R-20). */
RLMESH_API RLMeshStatus rlmesh_env_factory_new(const RLMeshEnvFactoryVtable* vtable,
                                               void* user_data,
                                               const RLMeshEnvFactoryAttributes* attributes,
                                               RLMeshEnvFactory** out);
RLMESH_API void rlmesh_env_factory_free(
    RLMeshEnvFactory* factory); /* never calls close; NULL no-op */
RLMESH_API RLMeshStatus rlmesh_env_factory_prepare(RLMeshEnvFactory* factory);
/* Stamps (ENV-17). On RLMESH_OK exactly one of *out_env / *out_vector_env is non-NULL (the
 * caller's) and the other is set to NULL; on any other status neither is written. */
RLMESH_API RLMeshStatus rlmesh_env_factory_make(RLMeshEnvFactory* factory,
                                                const RLMeshValue* kwargs /* DICT; NULL = {} */,
                                                RLMeshEnv** out_env,
                                                RLMeshVectorEnv** out_vector_env);
RLMESH_API RLMeshStatus rlmesh_env_factory_tags_for(RLMeshEnvFactory* factory,
                                                    const RLMeshValue* params /* DICT; NULL = {} */,
                                                    RLMeshEnvTags** out /* NULL = None */);
RLMESH_API RLMeshStatus rlmesh_env_factory_enumerate_variants(
    RLMeshEnvFactory* factory, RLMeshVariantList** out /* NULL = not defined */);
RLMESH_API RLMeshStatus rlmesh_env_factory_close(RLMeshEnvFactory* factory);
RLMESH_API RLMeshStatus
rlmesh_env_factory_serve(RLMeshEnvFactory* factory, const char* address,
                         const RLMeshEnvFactoryServeOptions* options /* NULL = INIT */);
/* rlmesh_env_factory_describe / _describe_json: Core (COR-10); rlmesh_ext_env_server_describe_json:
 * COR-16. */

/* Factory class-attribute getters (Stable in Python); they read the copied attributes. */
RLMESH_API RLMeshStatus rlmesh_env_factory_tags(const RLMeshEnvFactory*,
                                                RLMeshEnvTags** out /* NULL = None */);
RLMESH_API RLMeshStatus rlmesh_env_factory_reset_options(const RLMeshEnvFactory*,
                                                         RLMeshValue** out /* LIST of TEXT */);
RLMESH_API RLMeshStatus rlmesh_env_factory_tag_params(const RLMeshEnvFactory*,
                                                      RLMeshTagParamList** out);
RLMESH_API RLMeshStatus rlmesh_env_factory_workflow_edition(const RLMeshEnvFactory*, bool* out_has,
                                                            RLMeshBytes* out);

/* Lists (R-20): items are borrowed until the list is freed. */
RLMESH_API size_t rlmesh_variant_list_len(const RLMeshVariantList*);
RLMESH_API const RLMeshVariant* rlmesh_variant_list_get(const RLMeshVariantList*,
                                                        size_t index); /* NULL past the end */
RLMESH_API void rlmesh_variant_list_free(RLMeshVariantList*);
RLMESH_API size_t rlmesh_tag_param_list_len(const RLMeshTagParamList*);
/* A lent struct (R-19): struct_size is the library's own sizeof; read a field only if
 * offsetof(field) + sizeof(field) <= p->struct_size, else treat it as its _INIT value.
 * NULL past the end. */
RLMESH_API const RLMeshTagParam* rlmesh_tag_param_list_get(const RLMeshTagParamList*, size_t index);
RLMESH_API void rlmesh_tag_param_list_free(RLMeshTagParamList*);

/* Variant (a value handle; R-11). */
typedef struct RLMeshVariantOptions {
  size_t struct_size;
  const RLMeshValue* metadata; /* DICT, copied; NULL = {} (Python's **metadata) */
} RLMeshVariantOptions;
#define RLMESH_VARIANT_OPTIONS_INIT {sizeof(RLMeshVariantOptions), NULL}
RLMESH_API RLMeshStatus rlmesh_variant_new(const char* id,
                                           const RLMeshValue* params /* DICT, copied */,
                                           const RLMeshVariantOptions* options /* NULL = INIT */,
                                           RLMeshVariant** out);
RLMESH_API void rlmesh_variant_free(RLMeshVariant*);
RLMESH_API RLMeshStatus rlmesh_variant_id(const RLMeshVariant*, RLMeshBytes* out);
RLMESH_API const RLMeshValue* rlmesh_variant_params(
    const RLMeshVariant*); /* DICT; borrowed until the next setter or _free */
RLMESH_API const RLMeshValue* rlmesh_variant_metadata(
    const RLMeshVariant*); /* DICT; borrowed likewise */
RLMESH_API RLMeshStatus rlmesh_variant_repr(const RLMeshVariant*, RLMeshBytes* out);
RLMESH_API bool rlmesh_variant_equals(const RLMeshVariant*, const RLMeshVariant*);
/* Variant fields are mutable, as Python's slots are (py/_variants.py:39-49). */
RLMESH_API RLMeshStatus rlmesh_variant_set_id(RLMeshVariant*, const char* id);
RLMESH_API RLMeshStatus rlmesh_variant_set_params(RLMeshVariant*,
                                                  const RLMeshValue* params /* DICT, copied */);
RLMESH_API RLMeshStatus rlmesh_variant_set_metadata(RLMeshVariant*,
                                                    const RLMeshValue* metadata /* DICT, copied */);

/* RLMeshVariantSink is lent to enumerate_variants for that callback only; never stored or freed. */
/* Takes `variant` only on RLMESH_OK; on any other status the caller still owns it
 * and frees it (R-20's ownership rule, applied to a pushed handle). */
RLMESH_API RLMeshStatus rlmesh_variant_sink_push(RLMeshVariantSink* sink, RLMeshVariant* variant);

#define RLMESH_ENV_FACTORY_SERVE_OPTIONS_MIN_SIZE sizeof(RLMeshEnvFactoryServeOptions)

#define RLMESH_TAG_PARAM_MIN_SIZE sizeof(RLMeshTagParam)

#define RLMESH_ENV_FACTORY_ATTRIBUTES_MIN_SIZE sizeof(RLMeshEnvFactoryAttributes)

#define RLMESH_ENV_FACTORY_VTABLE_MIN_SIZE sizeof(RLMeshEnvFactoryVtable)

#define RLMESH_VARIANT_OPTIONS_MIN_SIZE sizeof(RLMeshVariantOptions)

/* ---- Policy / Model (facade) --------------------------------------------- */
typedef struct RLMeshModel RLMeshModel;
typedef struct RLMeshModelSpec RLMeshModelSpec;
typedef enum RLMeshSpecKind {
  RLMESH_SPEC_KIND_INVALID = 0,
  RLMESH_SPEC_KIND_NONE = 1,
  RLMESH_SPEC_KIND_MODEL_SPEC = 2,
  RLMESH_SPEC_KIND_NO_ADAPTER = 3
} RLMeshSpecKind;
typedef struct RLMeshSpecArg { /* by value, frozen per ABI generation (R-19) */
  RLMeshSpecKind kind;
  const RLMeshModelSpec* spec; /* MODEL_SPEC only; deep-copied by the consumer */
} RLMeshSpecArg;

#define RLMESH_CORNER_PREDICT 1u
#define RLMESH_CORNER_PREDICT_CHUNK 2u
#define RLMESH_CORNER_PREDICT_BATCH 4u
#define RLMESH_CORNER_PREDICT_CHUNK_BATCH 8u

typedef struct RLMeshEpisodeState RLMeshEpisodeState; /* borrowed, valid for the call */
/* true and *out = value when present (a stored NULL is a value); false when missing. */
RLMESH_API bool rlmesh_episode_state_get(const RLMeshEpisodeState*, const char* key, void** out);
/* R-20: on RLMESH_OK value (and free_value, NULL = none) pass to the state, and replacing
 * a key frees the old value first; on nonzero free_value is not called, the caller still
 * owns value, and the state is unchanged. */
RLMESH_API RLMeshStatus rlmesh_episode_state_set(RLMeshEpisodeState*, const char* key, void* value,
                                                 void (*free_value)(void* value));
RLMESH_API bool rlmesh_episode_state_remove(RLMeshEpisodeState*, const char* key);
RLMESH_API size_t rlmesh_episode_state_len(const RLMeshEpisodeState*);

typedef struct RLMeshPredictContext { /* struct_size = the library's sizeof */
  size_t struct_size;
  const char* episode_id;
  bool has_episode_seed;
  uint8_t _reserved0[7];
  int64_t episode_seed;
  uint64_t predict_index;
  bool has_predict_seed;
  uint8_t _reserved1[7];
  int64_t predict_seed;
  RLMeshEpisodeState* state;
} RLMeshPredictContext;
/* Lent (R-19): a reader treats any field struct_size does not cover as this value. */
#define RLMESH_PREDICT_CONTEXT_INIT \
  {sizeof(RLMeshPredictContext), "", false, {0}, 0, 0, false, {0}, 0, NULL}

typedef struct RLMeshLoadKey { /* by value, frozen per ABI generation (R-19) */
  const char* name;            /* nonempty, unique */
  bool required;
  uint8_t _reserved[sizeof(size_t) - 1]; /* no default in load */
} RLMeshLoadKey;

/* Corners write OWNED values into pre-zeroed out slots. On nonzero the library
 * frees rows already written. context(s) is NULL unless the corner's bit is set
 * in context_corners. */
typedef struct RLMeshModelVtable {
  size_t struct_size;
  RLMeshStatus (*predict)(void* user_data, const RLMeshValue* observation,
                          const RLMeshPredictContext* context, RLMeshValue** out_action);
  RLMeshStatus (*predict_chunk)(void* user_data, const RLMeshValue* observation,
                                uint32_t execution_horizon, const RLMeshPredictContext* context,
                                RLMeshValue** out_chunk);
  RLMeshStatus (*predict_batch)(void* user_data, const RLMeshValue* const* observations,
                                size_t count, const RLMeshPredictContext* const* contexts,
                                RLMeshValue** out_actions);
  RLMeshStatus (*predict_chunk_batch)(void* user_data, const RLMeshValue* const* observations,
                                      size_t count, uint32_t execution_horizon,
                                      const RLMeshPredictContext* const* contexts,
                                      RLMeshValue** out_chunks);
  RLMeshStatus (*on_episode_end)(void* user_data, const char* episode_id);
  RLMeshStatus (*close)(void* user_data);
  RLMeshStatus (*load)(void* user_data, const RLMeshValue* config);
  RLMeshStatus (*loaded_native_chunk)(void* user_data, bool* out_has, uint32_t* out);
  void (*free_user_data)(void* user_data);
  uint32_t context_corners; /* 0; bits of non-NULL slots only, else RLMESH_ERR_TYPE */
  uint32_t _reserved0;
  RLMeshSpecArg spec; /* {RLMESH_SPEC_KIND_NONE, NULL} */
  bool allow_fusion;  /* true */
  bool has_native_chunk;
  uint8_t _reserved1[2]; /* false */
  uint32_t native_chunk; /* static K for describe; 0 with has_native_chunk: ValueError class */
  const char* workflow_edition;   /* NULL = None */
  const RLMeshLoadKey* load_keys; /* NULL = None: every key reaches load (MDL-19); R-13 */
  size_t
      load_keys_len; /* entries, deep-copied; non-NULL with 0 = []; NULL with >0: RLMESH_ERR_TYPE */
} RLMeshModelVtable;
#define RLMESH_MODEL_VTABLE_INIT                                                               \
  {sizeof(RLMeshModelVtable),     NULL, NULL,  NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, \
   {RLMESH_SPEC_KIND_NONE, NULL}, true, false, {0},  0,    NULL, NULL, 0}

typedef struct RLMeshModelOptions {
  size_t struct_size;
  RLMeshSpecArg spec; /* NONE = vtable's */
  RLMeshStatus (*on_episode_end)(void* hook_user_data,
                                 const char* episode_id); /* NULL = vtable's */
  RLMeshStatus (*on_close)(void* hook_user_data);         /* NULL = vtable's close */
  void* hook_user_data;
  void (*free_hook_user_data)(void* hook_user_data);
  bool trust_entrypoints;
  uint8_t _reserved[sizeof(size_t) - 1]; /* false */
} RLMeshModelOptions;
#define RLMESH_MODEL_OPTIONS_INIT \
  {sizeof(RLMeshModelOptions), {RLMESH_SPEC_KIND_NONE, NULL}, NULL, NULL, NULL, NULL, false, {0}}

typedef struct RLMeshExtModelServeOptions {
  size_t struct_size;
  const char* token;
  size_t predict_concurrency;
  bool has_predict_concurrency;
  uint8_t _reserved[sizeof(size_t) - 1];
} RLMeshExtModelServeOptions;
#define RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT \
  {sizeof(RLMeshExtModelServeOptions), NULL, 0, false, {0}}

typedef struct RLMeshModelServeOptions {
  size_t struct_size;
  const RLMeshServeOptions* options;     /* NULL = None (Env) */
  RLMeshExtCancelToken* cancel;          /* NULL = none, borrowed (R-18) */
  const RLMeshExtModelServeOptions* ext; /* NULL = none */
} RLMeshModelServeOptions;
#define RLMESH_MODEL_SERVE_OPTIONS_INIT {sizeof(RLMeshModelServeOptions), NULL, NULL, NULL}

RLMESH_API RLMeshStatus
rlmesh_model_serve(RLMeshModel* model, const char* address,
                   const RLMeshModelServeOptions* options /* NULL = INIT */);

RLMESH_API RLMeshStatus rlmesh_model_new(const RLMeshModelVtable*, void* user_data,
                                         const RLMeshModelOptions*, RLMeshModel** out);
RLMESH_API RLMeshStatus rlmesh_model_from_config(const RLMeshModelVtable*, void* user_data,
                                                 const RLMeshValue* config, RLMeshModel** out);
RLMESH_API RLMeshStatus rlmesh_model_clone(const RLMeshModel*, RLMeshModel** out);
RLMESH_API RLMeshStatus rlmesh_model_close(RLMeshModel*);
RLMESH_API void rlmesh_model_free(RLMeshModel*);
RLMESH_API RLMeshStatus rlmesh_model_spec(const RLMeshModel*, RLMeshSpecArg* out);
RLMESH_API bool rlmesh_model_allow_fusion(const RLMeshModel*);
RLMESH_API void rlmesh_model_native_chunk(const RLMeshModel*, bool* out_has, uint32_t* out);
RLMESH_API RLMeshStatus rlmesh_model_workflow_edition(const RLMeshModel*, bool* out_has,
                                                      RLMeshBytes* out);
RLMESH_API int64_t rlmesh_predict_seed(int64_t episode_seed, uint64_t predict_index);

#define RLMESH_MODEL_VTABLE_MIN_SIZE sizeof(RLMeshModelVtable)

#define RLMESH_MODEL_OPTIONS_MIN_SIZE sizeof(RLMeshModelOptions)

#define RLMESH_PREDICT_CONTEXT_MIN_SIZE sizeof(RLMeshPredictContext)

#define RLMESH_MODEL_SERVE_OPTIONS_MIN_SIZE sizeof(RLMeshModelServeOptions)

#define RLMESH_EXT_MODEL_SERVE_OPTIONS_MIN_SIZE sizeof(RLMeshExtModelServeOptions)

/* ---- Running: facade targets, observers and results ---------------------- */
typedef struct RLMeshRunResult RLMeshRunResult;
typedef struct RLMeshTelemetryRow RLMeshTelemetryRow;
typedef struct RLMeshRunHooks RLMeshRunHooks;
typedef struct RLMeshRunContext RLMeshRunContext;
/* Read/frame-source and advisory authoring belongs to the adapters wave. */
typedef struct RLMeshAdvisory RLMeshAdvisory;
typedef enum RLMeshModelTargetKind {
  RLMESH_MODEL_TARGET_KIND_INVALID = 0,
  RLMESH_MODEL_TARGET_KIND_MODEL = 1,
  RLMESH_MODEL_TARGET_KIND_REMOTE_MODEL = 2,
  RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE = 3
} RLMeshModelTargetKind;
typedef enum RLMeshEnvTargetKind {
  RLMESH_ENV_TARGET_KIND_INVALID = 0,
  RLMESH_ENV_TARGET_KIND_ADDRESS = 1,
  RLMESH_ENV_TARGET_KIND_ENV_SERVER = 2,
  RLMESH_ENV_TARGET_KIND_REMOTE_ENV = 3,
  RLMESH_ENV_TARGET_KIND_REMOTE_VECTOR_ENV = 4,
  RLMESH_ENV_TARGET_KIND_ENV = 5,
  RLMESH_ENV_TARGET_KIND_VECTOR_ENV = 6,
  RLMESH_ENV_TARGET_KIND_ENV_FACTORY = 7
} RLMeshEnvTargetKind;
/* Frozen by-value layouts (RUN-27). Constructors borrow handles and addresses. */
typedef struct RLMeshModelTarget {
  RLMeshModelTargetKind kind;
  const void* handle;
} RLMeshModelTarget;
typedef struct RLMeshEnvTarget {
  RLMeshEnvTargetKind kind;
  const void* handle;
  const char* address;
} RLMeshEnvTarget;
#ifdef __cplusplus
#define RLMESH_RANDOM_SAMPLE (RLMeshModelTarget{RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE, NULL})
#else
#define RLMESH_RANDOM_SAMPLE ((RLMeshModelTarget){RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE, NULL})
#endif
RLMESH_API RLMeshModelTarget rlmesh_model_target(RLMeshModel*);
RLMESH_API RLMeshEnvTarget rlmesh_address_target(const char*);
RLMESH_API RLMeshEnvTarget rlmesh_env_server_target(const RLMeshEnvServer*);
RLMESH_API RLMeshEnvTarget rlmesh_env_target(const RLMeshEnv*);
RLMESH_API RLMeshEnvTarget rlmesh_vector_env_target(const RLMeshVectorEnv*);
RLMESH_API RLMeshEnvTarget rlmesh_env_factory_target(const RLMeshEnvFactory*);

typedef struct RLMeshEpisodeResult {
  size_t struct_size;
  int64_t index;
  bool has_seed;
  int64_t seed;
  int64_t steps;
  double reward;
  bool terminated;
  bool truncated;
  bool has_success;
  bool success;
  double duration_s;
  bool has_predict_ms;
  double predict_ms;
  bool has_step_ms;
  double step_ms;
  bool has_trial;
  int64_t trial;
} RLMeshEpisodeResult;
#define RLMESH_EPISODE_RESULT_MIN_SIZE sizeof(RLMeshEpisodeResult)
#define RLMESH_EPISODE_RESULT_INIT \
  {sizeof(RLMeshEpisodeResult),    \
   0,                              \
   false,                          \
   0,                              \
   0,                              \
   0.0,                            \
   false,                          \
   false,                          \
   false,                          \
   false,                          \
   0.0,                            \
   false,                          \
   0.0,                            \
   false,                          \
   0.0,                            \
   false,                          \
   0}
RLMESH_API RLMeshStatus rlmesh_episode_result_equals(const RLMeshEpisodeResult*,
                                                     const RLMeshEpisodeResult*, bool* out);
RLMESH_API RLMeshStatus rlmesh_episode_result_repr(const RLMeshEpisodeResult*, RLMeshBytes* out);
typedef struct RLMeshRunResultOptions {
  size_t struct_size;
  const RLMeshEpisodeResult* const* episodes;
  size_t episodes_len;
  const RLMeshTelemetryRow* const* telemetry;
  size_t telemetry_len;
  const RLMeshAdvisory* const* advisories;
  size_t advisories_len;
} RLMeshRunResultOptions;
#define RLMESH_RUN_RESULT_OPTIONS_MIN_SIZE sizeof(RLMeshRunResultOptions)
#define RLMESH_RUN_RESULT_OPTIONS_INIT {sizeof(RLMeshRunResultOptions), NULL, 0, NULL, 0, NULL, 0}
RLMESH_API RLMeshStatus rlmesh_run_result_new(const RLMeshRunResultOptions*, RLMeshRunResult** out);
RLMESH_API RLMeshRunResult* rlmesh_run_result_clone(const RLMeshRunResult*);
RLMESH_API void rlmesh_run_result_free(RLMeshRunResult*);
RLMESH_API size_t rlmesh_run_result_episodes_len(const RLMeshRunResult*);
RLMESH_API size_t rlmesh_run_result_num_episodes(const RLMeshRunResult*);
RLMESH_API int64_t rlmesh_run_result_total_steps(const RLMeshRunResult*);
RLMESH_API double rlmesh_run_result_mean_reward(const RLMeshRunResult*);
RLMESH_API bool rlmesh_run_result_success_rate(const RLMeshRunResult*, double* out);
RLMESH_API RLMeshStatus rlmesh_run_result_episodes_get(const RLMeshRunResult*, size_t,
                                                       RLMeshEpisodeResult* out);
RLMESH_API size_t rlmesh_run_result_telemetry_len(const RLMeshRunResult*);
RLMESH_API RLMeshStatus rlmesh_run_result_telemetry_get(const RLMeshRunResult*, size_t,
                                                        RLMeshTelemetryRow** out);
RLMESH_API RLMeshStatus rlmesh_run_result_format_telemetry(const RLMeshRunResult*,
                                                           RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_run_result_repr(const RLMeshRunResult*, RLMeshBytes* out);
RLMESH_API bool rlmesh_run_result_equals(const RLMeshRunResult*, const RLMeshRunResult*);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_new(const char*, const char*, const char*, const char*,
                                                 int64_t, double, double, double, double,
                                                 RLMeshTelemetryRow** out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_op(const RLMeshTelemetryRow*, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_component(const RLMeshTelemetryRow*, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_metric(const RLMeshTelemetryRow*, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_unit(const RLMeshTelemetryRow*, RLMeshBytes* out);
RLMESH_API int64_t rlmesh_telemetry_row_count(const RLMeshTelemetryRow*);
RLMESH_API double rlmesh_telemetry_row_avg(const RLMeshTelemetryRow*);
RLMESH_API double rlmesh_telemetry_row_p50(const RLMeshTelemetryRow*);
RLMESH_API double rlmesh_telemetry_row_p95(const RLMeshTelemetryRow*);
RLMESH_API double rlmesh_telemetry_row_p99(const RLMeshTelemetryRow*);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_repr(const RLMeshTelemetryRow*, RLMeshBytes* out);
RLMESH_API bool rlmesh_telemetry_row_equals(const RLMeshTelemetryRow*, const RLMeshTelemetryRow*);
RLMESH_API void rlmesh_telemetry_row_free(RLMeshTelemetryRow*);

/* Contexts are shared, retained independently from the hook invocation. The
 * adapters wave adds read/frame_sources to the facade and these APIs. */
typedef struct RLMeshRunContextVtable {
  size_t struct_size;
  RLMeshStatus (*contract)(void*, RLMeshEnvContract** out);
  RLMeshStatus (*num_envs)(void*, uint64_t* out);
  void (*free_user_data)(void*);
} RLMeshRunContextVtable;
#define RLMESH_RUN_CONTEXT_VTABLE_MIN_SIZE sizeof(RLMeshRunContextVtable)
#define RLMESH_RUN_CONTEXT_VTABLE_INIT {sizeof(RLMeshRunContextVtable), NULL, NULL, NULL}
RLMESH_API RLMeshStatus rlmesh_run_context_new(const RLMeshRunContextVtable*, void*,
                                               RLMeshRunContext** out);
RLMESH_API RLMeshRunContext* rlmesh_run_context_clone(const RLMeshRunContext*);
RLMESH_API void rlmesh_run_context_free(RLMeshRunContext*);
RLMESH_API RLMeshStatus rlmesh_run_context_contract(const RLMeshRunContext*,
                                                    RLMeshEnvContract** out);
RLMESH_API RLMeshStatus rlmesh_run_context_num_envs(const RLMeshRunContext*, uint64_t* out);
typedef struct RLMeshStepEvent {
  size_t struct_size;
  int64_t episode;
  bool has_seed;
  int64_t seed;
  int64_t step;
  const RLMeshValue* observation;
  const RLMeshValue* action;
  double reward;
  bool terminated;
  bool truncated;
  const RLMeshValue* info;
  double predict_ms;
  double step_ms;
  const RLMeshRunContext* context;
} RLMeshStepEvent;
#define RLMESH_STEP_EVENT_MIN_SIZE sizeof(RLMeshStepEvent)
#define RLMESH_STEP_EVENT_INIT \
  {sizeof(RLMeshStepEvent), 0, false, 0, 0, NULL, NULL, 0.0, false, false, NULL, 0.0, 0.0, NULL}
RLMESH_API RLMeshStatus rlmesh_step_event_equals(const RLMeshStepEvent*, const RLMeshStepEvent*,
                                                 bool* out);
RLMESH_API RLMeshStatus rlmesh_step_event_repr(const RLMeshStepEvent*, RLMeshBytes* out);
typedef struct RLMeshRunHooksVtable {
  size_t struct_size;
  RLMeshStatus (*on_run_start)(void*, const RLMeshRunContext*);
  RLMeshStatus (*on_episode_start)(void*, int64_t, const int64_t*);
  RLMeshStatus (*on_step)(void*, const RLMeshStepEvent*);
  RLMeshStatus (*on_episode_end)(void*, const RLMeshEpisodeResult*);
  RLMeshStatus (*on_run_end)(void*, const RLMeshRunResult*);
  void (*free_user_data)(void*);
} RLMeshRunHooksVtable;
#define RLMESH_RUN_HOOKS_VTABLE_MIN_SIZE sizeof(RLMeshRunHooksVtable)
#define RLMESH_RUN_HOOKS_VTABLE_INIT \
  {sizeof(RLMeshRunHooksVtable), NULL, NULL, NULL, NULL, NULL, NULL}
RLMESH_API RLMeshStatus rlmesh_run_hooks_new(const RLMeshRunHooksVtable*, void*,
                                             RLMeshRunHooks** out);
RLMESH_API void rlmesh_run_hooks_free(RLMeshRunHooks*);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_run_start(RLMeshRunHooks*, const RLMeshRunContext*);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_episode_start(RLMeshRunHooks*, int64_t, const int64_t*);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_step(RLMeshRunHooks*, const RLMeshStepEvent*);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_episode_end(RLMeshRunHooks*,
                                                        const RLMeshEpisodeResult*);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_run_end(RLMeshRunHooks*, const RLMeshRunResult*);

typedef struct RLMeshRunOptions {
  size_t struct_size;
  const int64_t* seeds;
  size_t seeds_len;
  RLMeshRunHooks* hooks;
  const char* workflow_edition;
  const RLMeshExtCancelToken* cancel;
  uint64_t episodes;
  uint64_t max_episode_steps;
  double max_episode_seconds;
  uint64_t prefetch_lead;
  uint64_t trial_index_base;
  uint32_t execution_horizon;
  bool has_episodes;
  bool has_max_episode_steps;
  bool has_max_episode_seconds;
  bool close_env;
  bool has_trust_entrypoints;
  bool trust_entrypoints;
  uint8_t _reserved[2];
} RLMeshRunOptions;
#define RLMESH_RUN_OPTIONS_MIN_SIZE sizeof(RLMeshRunOptions)
#define RLMESH_RUN_OPTIONS_INIT \
  {sizeof(RLMeshRunOptions),    \
   NULL,                        \
   0,                           \
   NULL,                        \
   NULL,                        \
   NULL,                        \
   0,                           \
   0,                           \
   0.0,                         \
   0,                           \
   0,                           \
   1,                           \
   false,                       \
   false,                       \
   false,                       \
   false,                       \
   false,                       \
   false,                       \
   {0, 0}}
/* NULL options means INIT; zero-initialized options are refused. Result remains
 * NULL on failure. Hooks/targets are borrowed until the call returns. Instruction,
 * view and advisory support arrive with adapters, matching the Rust facade. */
RLMESH_API RLMeshStatus rlmesh_run(RLMeshModelTarget, RLMeshEnvTarget, const RLMeshRunOptions*,
                                   RLMeshRunResult** out);
RLMESH_API RLMeshStatus rlmesh_model_run(RLMeshModel*, RLMeshEnvTarget, const RLMeshRunOptions*,
                                         RLMeshRunResult** out);

/* A Session borrows its model/server until free. Local envs/factories stay
 * exclusively leased until free; close never closes the model. */
typedef struct RLMeshSession RLMeshSession;
typedef struct RLMeshSessionOptions {
  size_t struct_size;
  const char* workflow_edition;
  const RLMeshExtCancelToken* cancel;
  uint32_t execution_horizon;
  bool close_env;
  bool has_trust_entrypoints;
  bool trust_entrypoints;
  uint8_t _reserved;
} RLMeshSessionOptions;
#define RLMESH_SESSION_OPTIONS_MIN_SIZE sizeof(RLMeshSessionOptions)
#define RLMESH_SESSION_OPTIONS_INIT \
  {sizeof(RLMeshSessionOptions), NULL, NULL, 1, false, false, false, 0}
typedef struct RLMeshSessionRunOptions {
  size_t struct_size;
  const int64_t* seeds;
  size_t seeds_len;
  RLMeshRunHooks* hooks;
  const RLMeshExtCancelToken* cancel;
  uint64_t episodes;
  uint64_t max_episode_steps;
  double max_episode_seconds;
  uint64_t trial_index_base;
  bool has_episodes;
  bool has_max_episode_steps;
  bool has_max_episode_seconds;
  uint8_t _reserved[5];
} RLMeshSessionRunOptions;
#define RLMESH_SESSION_RUN_OPTIONS_MIN_SIZE sizeof(RLMeshSessionRunOptions)
#define RLMESH_SESSION_RUN_OPTIONS_INIT \
  {sizeof(RLMeshSessionRunOptions),     \
   NULL,                                \
   0,                                   \
   NULL,                                \
   NULL,                                \
   0,                                   \
   0,                                   \
   0.0,                                 \
   0,                                   \
   false,                               \
   false,                               \
   false,                               \
   {0, 0, 0, 0, 0}}
RLMESH_API RLMeshStatus rlmesh_session_new(RLMeshModelTarget, RLMeshEnvTarget,
                                           const RLMeshSessionOptions*, RLMeshSession** out);
RLMESH_API RLMeshStatus rlmesh_model_session(RLMeshModel*, RLMeshEnvTarget,
                                             const RLMeshSessionOptions*, RLMeshSession** out);
RLMESH_API RLMeshStatus rlmesh_session_reset(RLMeshSession*, const int64_t* seed,
                                             const int64_t* trial_index,
                                             RLMeshValue** out_observation, RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_session_predict(RLMeshSession*, const RLMeshValue*,
                                               RLMeshValue** out_action);
RLMESH_API RLMeshStatus rlmesh_session_step(RLMeshSession*, const RLMeshValue*,
                                            RLMeshValue** out_observation, double* out_reward,
                                            bool* out_terminated, bool* out_truncated,
                                            RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_session_run(RLMeshSession*, const RLMeshSessionRunOptions*,
                                           RLMeshRunResult** out);
RLMESH_API RLMeshStatus rlmesh_session_done(const RLMeshSession*, bool* out);
RLMESH_API RLMeshStatus rlmesh_session_contract(RLMeshSession*, RLMeshEnvContract** out);
RLMESH_API RLMeshStatus rlmesh_session_num_envs(RLMeshSession*, uint64_t* out);
RLMESH_API RLMeshStatus rlmesh_session_selected_workflow_edition(RLMeshSession*, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_session_run_context(RLMeshSession*, RLMeshRunContext** out);
RLMESH_API RLMeshStatus rlmesh_session_cancel_handle(const RLMeshSession*,
                                                     RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_session_close(RLMeshSession*);
RLMESH_API void rlmesh_session_free(RLMeshSession*);

/* ---- Remote clients ----------------------------------------------------- */
typedef struct RLMeshRemoteEnv RLMeshRemoteEnv;
typedef struct RLMeshRemoteVectorEnv RLMeshRemoteVectorEnv;
typedef struct RLMeshRemoteModel RLMeshRemoteModel;
typedef struct RLMeshRemoteEnvOptions {
  size_t struct_size;
  const char* address;
  const char* host;
  bool has_port;
  uint16_t port;
  const char* path;
  bool has_transport;
  RLMeshTransport transport;
  bool has_connect_timeout_seconds;
  double connect_timeout_seconds;
  bool has_request_timeout_seconds;
  double request_timeout_seconds;
  const char* workflow_edition;
} RLMeshRemoteEnvOptions;
#define RLMESH_REMOTE_ENV_OPTIONS_MIN_SIZE sizeof(RLMeshRemoteEnvOptions)
#define RLMESH_REMOTE_ENV_OPTIONS_INIT                                   \
  {sizeof(RLMeshRemoteEnvOptions), NULL,  NULL, false, 0,   NULL, false, \
   RLMESH_TRANSPORT_INVALID,       false, 0.0,  false, 0.0, NULL}
typedef struct RLMeshRemoteModelOptions {
  size_t struct_size;
  const char* address;
  const char* host;
  bool has_port;
  uint16_t port;
  const char* path;
  bool has_transport;
  RLMeshTransport transport;
  bool has_connect_timeout_seconds;
  double connect_timeout_seconds;
  bool has_request_timeout_seconds;
  double request_timeout_seconds;
} RLMeshRemoteModelOptions;
#define RLMESH_REMOTE_MODEL_OPTIONS_MIN_SIZE sizeof(RLMeshRemoteModelOptions)
#define RLMESH_REMOTE_MODEL_OPTIONS_INIT                                  \
  {sizeof(RLMeshRemoteModelOptions), NULL,  NULL, false, 0,  NULL, false, \
   RLMESH_TRANSPORT_INVALID,         false, 0.0,  false, 0.0}
/* Env construction connects; Model construction only validates. Remote I/O
 * is exclusive, while immutable accessors/cancellation may overlap any call.
 * Accessor pointers remain borrowed until free, including after close. */
RLMESH_API RLMeshStatus rlmesh_remote_env_new(const RLMeshRemoteEnvOptions*, RLMeshRemoteEnv** out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_new(const RLMeshRemoteEnvOptions*,
                                                     RLMeshRemoteVectorEnv** out);
RLMESH_API RLMeshStatus rlmesh_remote_model_new(const RLMeshRemoteModelOptions*,
                                                RLMeshRemoteModel** out);
RLMESH_API RLMeshStatus rlmesh_ext_remote_env_new_with_token(const RLMeshRemoteEnvOptions*,
                                                             const char* token,
                                                             RLMeshRemoteEnv** out);
RLMESH_API RLMeshStatus rlmesh_ext_remote_vector_env_new_with_token(const RLMeshRemoteEnvOptions*,
                                                                    const char* token,
                                                                    RLMeshRemoteVectorEnv** out);
RLMESH_API RLMeshStatus rlmesh_ext_remote_model_new_with_token(const RLMeshRemoteModelOptions*,
                                                               const char* token,
                                                               RLMeshRemoteModel** out);
RLMESH_API RLMeshStatus rlmesh_remote_model_clone(const RLMeshRemoteModel*,
                                                  RLMeshRemoteModel** out);
RLMESH_API RLMeshStatus rlmesh_remote_model_address(const RLMeshRemoteModel*, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_model_repr(const RLMeshRemoteModel*, RLMeshBytes* out);
RLMESH_API void rlmesh_remote_model_free(RLMeshRemoteModel*);
RLMESH_API RLMeshModelTarget rlmesh_remote_model_target(const RLMeshRemoteModel*);
RLMESH_API RLMeshEnvTarget rlmesh_remote_env_target(RLMeshRemoteEnv*);
RLMESH_API RLMeshEnvTarget rlmesh_remote_vector_env_target(RLMeshRemoteVectorEnv*);
RLMESH_API RLMeshStatus rlmesh_remote_model_session(const RLMeshRemoteModel*, RLMeshEnvTarget,
                                                    const RLMeshSessionOptions*,
                                                    RLMeshSession** out);
RLMESH_API RLMeshStatus rlmesh_remote_env_address(const RLMeshRemoteEnv*, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_env_env_id(const RLMeshRemoteEnv*, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_env_selected_workflow_edition(const RLMeshRemoteEnv*,
                                                                    RLMeshBytes* out);
RLMESH_API const RLMeshEnvContract* rlmesh_remote_env_env_contract(const RLMeshRemoteEnv*);
RLMESH_API const RLMeshValue* rlmesh_remote_env_metadata(const RLMeshRemoteEnv*);
RLMESH_API const RLMeshSpace* rlmesh_remote_env_observation_space(const RLMeshRemoteEnv*);
RLMESH_API const RLMeshSpace* rlmesh_remote_env_action_space(const RLMeshRemoteEnv*);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_env_observation_space_spec(const RLMeshRemoteEnv*);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_env_action_space_spec(const RLMeshRemoteEnv*);
RLMESH_API RLMeshStatus rlmesh_remote_env_render_mode(const RLMeshRemoteEnv*, bool* out_has,
                                                      RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_env_cancel_handle(const RLMeshRemoteEnv*,
                                                        RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_remote_env_render(RLMeshRemoteEnv*, uint64_t env_index,
                                                 RLMeshValue** out_frame);
RLMESH_API RLMeshStatus rlmesh_remote_env_shutdown(RLMeshRemoteEnv*, const char* reason,
                                                   bool* out_accepted);
RLMESH_API RLMeshStatus rlmesh_remote_env_close(RLMeshRemoteEnv*);
RLMESH_API void rlmesh_remote_env_free(RLMeshRemoteEnv*);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_address(const RLMeshRemoteVectorEnv*,
                                                         RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_env_id(const RLMeshRemoteVectorEnv*,
                                                        RLMeshBytes* out);
RLMESH_API RLMeshStatus
rlmesh_remote_vector_env_selected_workflow_edition(const RLMeshRemoteVectorEnv*, RLMeshBytes* out);
RLMESH_API const RLMeshEnvContract* rlmesh_remote_vector_env_env_contract(
    const RLMeshRemoteVectorEnv*);
RLMESH_API const RLMeshValue* rlmesh_remote_vector_env_metadata(const RLMeshRemoteVectorEnv*);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_observation_space(
    const RLMeshRemoteVectorEnv*);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_action_space(const RLMeshRemoteVectorEnv*);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_vector_env_observation_space_spec(
    const RLMeshRemoteVectorEnv*);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_vector_env_action_space_spec(
    const RLMeshRemoteVectorEnv*);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_render_mode(const RLMeshRemoteVectorEnv*,
                                                             bool* out_has, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_cancel_handle(const RLMeshRemoteVectorEnv*,
                                                               RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_render(RLMeshRemoteVectorEnv*, uint64_t env_index,
                                                        RLMeshValue** out_frame);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_shutdown(RLMeshRemoteVectorEnv*,
                                                          const char* reason, bool* out_accepted);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_close(RLMeshRemoteVectorEnv*);
RLMESH_API void rlmesh_remote_vector_env_free(RLMeshRemoteVectorEnv*);
RLMESH_API uint64_t rlmesh_remote_vector_env_num_envs(const RLMeshRemoteVectorEnv*);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_single_observation_space(
    const RLMeshRemoteVectorEnv*);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_single_action_space(
    const RLMeshRemoteVectorEnv*);
RLMESH_API RLMeshStatus rlmesh_remote_env_reset(RLMeshRemoteEnv*, const int64_t* seed,
                                                const RLMeshValue* options,
                                                RLMeshValue** out_observation,
                                                RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_remote_env_step(RLMeshRemoteEnv*, const RLMeshValue*,
                                               RLMeshValue** out_observation, double* out_reward,
                                               bool* out_terminated, bool* out_truncated,
                                               RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_reset(RLMeshRemoteVectorEnv*, const int64_t* seed,
                                                       size_t seed_len, const RLMeshValue* options,
                                                       RLMeshValue** out_observation,
                                                       RLMeshValue** out_info);
/* A short capacity fails before I/O and leaves ALL outputs untouched. */
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_step(RLMeshRemoteVectorEnv*, const RLMeshValue*,
                                                      RLMeshValue** out_observation,
                                                      double* out_rewards, bool* out_terminated,
                                                      bool* out_truncated, size_t capacity,
                                                      size_t* out_len, RLMeshValue** out_info);

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
 * otherwise. The capi holds at most 65536 live episodes and never evicts one:
 * a predict that would start a new episode past that fails with
 * RLMESH_ERR_INTERNAL (Python raises RuntimeError) before the callback runs. */
typedef struct RLMeshExtLegacyEpisode {
  const char* id;
  bool seeded; /* whether `seed` carries an explicit reset seed */
  int64_t seed;
  uint64_t predict_index; /* re-plan ordinal within the episode, from 0 */
  int64_t predict_seed;   /* predict_seed(seed, predict_index); only when `seeded` */
} RLMeshExtLegacyEpisode;

/* What a predict callback receives. Every pointer is valid only for the
 * duration of the call. Row i of `observations` belongs to `episodes[i]`;
 * `episodes` always has exactly num_envs entries. */
typedef struct RLMeshExtLegacyObservation {
  /* num_envs decoded values, or NULL when this request carries no observation
   * or the contract declares no observation space (both are legal routes). */
  const RLMeshValue* const* observations;
  size_t num_envs;
  /* Spaces/metadata for the route. Never NULL on a predict the runtime
   * delivers: a route pins a contract (with an action space) before its first
   * predict. */
  const RLMeshEnvContract* contract;
  const char* session_id;
  const char* env_id;
  const char* request_id;
  const RLMeshExtLegacyEpisode* episodes; /* num_envs entries */
} RLMeshExtLegacyObservation;

/* Set this call's error message + recoverability before returning nonzero (any
 * callback: a model's predict or an env's reset / step / render). */
RLMESH_API void rlmesh_callback_set_error(const char* message, bool recoverable);

/* Set this call's error as an exact class instead, as a Python author raises
 * ValueError or TypeError from a hook: in process the failing call reports that
 * class, its status and `message`; across the wire the peer sees an environment
 * or model failure. An environment or model class carries code INTERNAL and is
 * recoverable only as RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION (a
 * recoverable model failure uses rlmesh_callback_set_error). INVALID or an
 * unknown value records RLMESH_ERROR_CLASS_RLMESH_EXCEPTION. The last call
 * before the callback returns wins. */
RLMESH_API void rlmesh_callback_set_error_class(RLMeshErrorClass error_class, const char* message);

/* The model callback vtable. Set struct_size = sizeof(RLMeshExtLegacyModelVtable); fields
 * beyond that are ignored (append-only). `predict` is required. The vtable is
 * COPIED into the model at rlmesh_ext_legacy_model_new, so the caller's struct need not
 * outlive the model; `user_data` is kept by pointer and must.
 *
 * Callbacks run on a worker thread, so `user_data` must be thread-migration-safe.
 * A callback must not re-enter the model handle that invoked it (no
 * rlmesh_ext_legacy_model_run_local / _serve / _free on it from inside a callback);
 * rlmesh_ext_legacy_model_cancel is the one exception and is safe from anywhere.
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
 * means every episode of `env_id`. `on_close` fires once per handle: when a
 * serve stops, or at rlmesh_ext_legacy_model_free; a run_local borrows the model and
 * never fires it. */
typedef struct RLMeshExtLegacyModelVtable {
  size_t struct_size;
  int (*predict)(void* user_data, const RLMeshExtLegacyObservation* obs, RLMeshValue** out_actions);
  void (*on_episode_end)(void* user_data, const char* env_id, const char* episode_id);
  void (*on_close)(void* user_data);
} RLMeshExtLegacyModelVtable;

/* An owned model handle. It is not a concurrency primitive: run at most one
 * rlmesh_ext_legacy_model_run_local / rlmesh_ext_legacy_model_serve on a handle at a time, and free
 * it only from a thread that is not executing it. rlmesh_ext_legacy_model_cancel is the only call
 * that may overlap a running one, from any thread. */
typedef struct RLMeshExtLegacyModel RLMeshExtLegacyModel;

RLMESH_API RLMeshStatus rlmesh_ext_legacy_model_new(const RLMeshExtLegacyModelVtable* vtable,
                                                    void* user_data, RLMeshExtLegacyModel** out);

/* Run options for rlmesh_ext_legacy_model_run_local. NULL == defaults (one episode,
 * unseeded, no caps). Every field is "unset" at 0 / NULL.
 *
 * Episode budget (Python's run(episodes=...)): max_episodes is the exact
 * episode count. 0 = unset: one episode, or one per seed when episode_seeds is
 * given. A non-zero count with episode_seeds must equal num_episode_seeds
 * (RLMESH_ERR_INVALID_ARGUMENT otherwise). There is no unbounded run.
 * TODO(docs/specs/language-parity/running.md): Python's episodes=0 (run
 * nothing) is not expressible here, since 0 means unset. */
typedef struct RLMeshExtLegacyRunOptions {
  uint64_t max_episodes;        /* exact episode count; 0 = unset (1, or the seed count) */
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
} RLMeshExtLegacyRunOptions;

/* What a finished run reports. Plain scalars: copy what you need. */
typedef struct RLMeshExtLegacyRunReport {
  int64_t total_episodes;      /* episodes completed during the run */
  int64_t total_steps;         /* env steps across the whole run */
  double total_reward;         /* summed episode reward */
  double mean_reward;          /* mean episode reward (0 if none completed) */
  int64_t terminated_episodes; /* completed episodes that ended terminated */
  int64_t truncated_episodes;  /* completed episodes that hit a step/time cap */
} RLMeshExtLegacyRunReport;

/* Drive the model against the env at `env_address` (tcp://host:port,
 * host:port, or unix:///path). Blocking — returns when the run ends.
 * `out_report` may be NULL; it is written only on RLMESH_OK. */
RLMESH_API RLMeshStatus rlmesh_ext_legacy_model_run_local(RLMeshExtLegacyModel* model,
                                                          const char* env_address,
                                                          const RLMeshExtLegacyRunOptions* options,
                                                          RLMeshExtLegacyRunReport* out_report);

/* Serve options for rlmesh_ext_legacy_model_serve. Pass NULL for all defaults (no auth, no
 * remote shutdown, no timeouts — serves until the process is killed). A 0 timeout
 * / concurrency means "unset". */
typedef struct RLMeshExtLegacyServeOptions {
  const char* token;            /* NULL/"" disables auth */
  bool allow_remote_shutdown;   /* honor a client-issued shutdown request */
  uint64_t idle_timeout_ms;     /* 0 = never idle-shutdown */
  uint64_t drain_timeout_ms;    /* 0 = unset */
  uint64_t close_timeout_ms;    /* 0 = unset; ignored by an env (see rlmesh_ext_legacy_env_serve) */
  size_t predict_concurrency;   /* 0 = default */
  const char* workflow_edition; /* workflow edition this server declares; NULL/"" = none */
} RLMeshExtLegacyServeOptions;

/* Serve the model as a ModelService endpoint at `bind_address` (tcp://host:port
 * or unix:///path). Blocking — returns when the server stops: a remote shutdown
 * request, an idle timeout, or rlmesh_ext_legacy_model_cancel. The same vtable backs every
 * predict, exactly as rlmesh_ext_legacy_model_run_local. `options` may be NULL for
 * defaults. */
RLMESH_API RLMeshStatus rlmesh_ext_legacy_model_serve(RLMeshExtLegacyModel* model,
                                                      const char* bind_address,
                                                      const RLMeshExtLegacyServeOptions* options);

/* Stop a blocking rlmesh_ext_legacy_model_run_local / rlmesh_ext_legacy_model_serve on `model`.
 * Call it from another thread (a signal handler's worker, a UI thread); it returns immediately and
 * the blocked call unwinds shortly after. NULL is a no-op.
 *
 * Cancellation is terminal for the handle: a cancelled model refuses further
 * runs. A cancelled serve returns RLMESH_OK after running on_close -- once, and
 * only after any predict still inside the C callback has returned; a cancelled
 * run_local (this one, or any later one on the handle) returns
 * RLMESH_ERR_CANCELLED, since it has no report to give. */
RLMESH_API void rlmesh_ext_legacy_model_cancel(RLMeshExtLegacyModel* model);

/* Free a model handle. NULL is a no-op. Freeing closes the model: on_close
 * fires here, on the calling thread, unless a serve already fired it
 * (rlmesh_ext_legacy_model_run_local borrows the model and never closes it, as Python's
 * Model.run does). Must NOT be called from inside one of
 * this model's own callbacks (it drops the runtime the callback is running on);
 * doing so is contained rather than fatal, but leaves the handle undefined. */
RLMESH_API void rlmesh_ext_legacy_model_free(RLMeshExtLegacyModel* model);

/* ---- environment ------------------------------------------------------- */

/* What `reset` receives; pointers are valid for the call only. `trial_index`
 * arrives only when the env declared the "trial_index" reset option. */
typedef struct RLMeshExtLegacyResetArgs {
  bool seeded; /* whether `seed` carries an explicit reset seed */
  int64_t seed;
  bool has_trial_index; /* whether `trial_index` is set */
  int64_t trial_index;
  const char* options_json; /* every reset option as a JSON object; NULL = none */
} RLMeshExtLegacyResetArgs;

/* What `reset` / `step` write. `observation` is OWNED: the capi takes and
 * frees whatever it holds when the callback returns, even when the callback
 * returns nonzero, so a callback that frees it on an error path must set it
 * back to NULL. It must be a value the callback owns (from a constructor),
 * never a borrowed one such as the step's action or a rlmesh_value_tuple_get /
 * rlmesh_value_dict_get child: freeing that is undefined behaviour. There is no
 * value copy function; build a fresh value with the constructors (e.g.
 * rlmesh_value_clone retains a value). `info_json` is a
 * borrowed JSON object (NULL = none) that must stay valid until the callback's
 * next call on this env (e.g. a std::string member). */
typedef struct RLMeshExtLegacyResetResult {
  RLMeshValue* observation;
  const char* info_json;
} RLMeshExtLegacyResetResult;

typedef struct RLMeshExtLegacyStepResult {
  RLMeshValue* observation;
  double reward;
  bool terminated;
  bool truncated;
  const char* info_json;
} RLMeshExtLegacyStepResult;

/* The environment callback vtable. Set struct_size = sizeof(RLMeshExtLegacyEnvVtable);
 * fields beyond it are ignored (append-only). `reset` and `step` are required.
 * Copied in at rlmesh_ext_legacy_env_new; `user_data` is kept by pointer and must outlive
 * the env.
 *
 * Each lane runs its callbacks on ONE dedicated thread (not the one that
 * created the env), one call at a time; the lanes of a multi-lane env
 * (rlmesh_ext_legacy_env_new_lanes) run concurrently, each with its own `user_data`. A
 * foreground env (RLMeshExtLegacyEnvConfig.foreground) runs them on the thread blocked
 * in rlmesh_ext_legacy_env_serve instead. `reset` / `step` /
 * `render` return 0 == RLMESH_OK, or nonzero after rlmesh_callback_set_error
 * to fail that request. With `recoverable` set, only the request fails and the
 * client's session stays usable; otherwise the session ends (the client's next
 * reset opens a new one, and the env keeps serving). The client reads
 * "[INTERNAL] {op} failed: {message}", the message the callback set, or
 * "callback failed with status <n>" when it set none. `step` gets the action
 * borrowed (NULL when the request carries none). `render` writes an OWNED
 * numeric image value (cast to uint8) of shape [H, W, 3], [H, W, 4],
 * [3, H, W], [4, H, W] or [H, W], or leaves it NULL for no frame; the capi
 * PNG-encodes it, and any other shape is no frame. `close` fires once per lane: when the server
 * stops, or in rlmesh_ext_legacy_env_free for an env that never served.
 *
 * Out-parameter ownership: the capi takes ownership of any value left in an
 * out-parameter (`out->observation`, `*out_frame`) when the callback returns,
 * WHETHER OR NOT it returns 0. A callback that fails after building a value
 * either leaves it for the capi to free or frees it itself and sets the field
 * back to NULL; freeing it without clearing the field is a double free. Never
 * return a borrowed value (the step's action, a rlmesh_value_tuple_get /
 * rlmesh_value_dict_get / rlmesh_value_dict_get_at child) as an observation or
 * frame: the capi frees it, which is undefined behaviour. Build an owned copy
 * with the constructors instead (no value copy function exists). */
typedef struct RLMeshExtLegacyEnvVtable {
  size_t struct_size;
  int (*reset)(void* user_data, const RLMeshExtLegacyResetArgs* args,
               RLMeshExtLegacyResetResult* out);
  int (*step)(void* user_data, const RLMeshValue* action, RLMeshExtLegacyStepResult* out);
  int (*render)(void* user_data, RLMeshValue** out_frame);
  void (*close)(void* user_data);
} RLMeshExtLegacyEnvVtable;

/* Runtime of the translation unit including this header, not of the Rust
 * library behind the ABI. Used by rlmesh_ext_legacy_env_config_init and rlmesh.hpp. */
#define RLMESH_DETAIL_STRINGIFY_IMPL(x) #x
#define RLMESH_DETAIL_STRINGIFY(x) RLMESH_DETAIL_STRINGIFY_IMPL(x)
#ifdef __cplusplus
#define RLMESH_DETAIL_HOST_LANGUAGE "c++"
#if defined(_MSVC_LANG) && _MSVC_LANG > __cplusplus
#define RLMESH_DETAIL_CXX_STANDARD _MSVC_LANG
#else
#define RLMESH_DETAIL_CXX_STANDARD __cplusplus
#endif
#if RLMESH_DETAIL_CXX_STANDARD > 202302L
#define RLMESH_DETAIL_HOST_VERSION "26"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '2', '6'
#elif RLMESH_DETAIL_CXX_STANDARD > 202002L
#define RLMESH_DETAIL_HOST_VERSION "23"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '2', '3'
#elif RLMESH_DETAIL_CXX_STANDARD >= 202002L
#define RLMESH_DETAIL_HOST_VERSION "20"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '2', '0'
#elif RLMESH_DETAIL_CXX_STANDARD >= 201703L
#define RLMESH_DETAIL_HOST_VERSION "17"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '1', '7'
#elif RLMESH_DETAIL_CXX_STANDARD >= 201402L
#define RLMESH_DETAIL_HOST_VERSION "14"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '1', '4'
#elif RLMESH_DETAIL_CXX_STANDARD >= 201103L
#define RLMESH_DETAIL_HOST_VERSION "11"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '1', '1'
#else
#define RLMESH_DETAIL_HOST_VERSION "98"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '9', '8'
#endif
#else
#define RLMESH_DETAIL_HOST_LANGUAGE "c"
#if defined(__STDC_VERSION__) && __STDC_VERSION__ > 201710L
#define RLMESH_DETAIL_HOST_VERSION "23"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '2', '3'
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201710L
#define RLMESH_DETAIL_HOST_VERSION "17"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '1', '7'
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define RLMESH_DETAIL_HOST_VERSION "11"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '1', '1'
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define RLMESH_DETAIL_HOST_VERSION "99"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '9', '9'
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199409L
#define RLMESH_DETAIL_HOST_VERSION "95"
#define RLMESH_DETAIL_HOST_VERSION_CHARS '9', '5'
#else
/* No standard macro: leave the language version undeclared. */
#endif
#endif

/* Clang defines GNU compatibility macros too, so check it first. */
#if defined(__clang_major__) && defined(__clang_minor__)
#define RLMESH_DETAIL_HOST_COMPILER                                            \
  ",\"toolchain\":{\"name\":\"clang\",\"version\":\"" RLMESH_DETAIL_STRINGIFY( \
      __clang_major__) "." RLMESH_DETAIL_STRINGIFY(__clang_minor__) "\"}"
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define RLMESH_DETAIL_HOST_COMPILER                                          \
  ",\"toolchain\":{\"name\":\"gcc\",\"version\":\"" RLMESH_DETAIL_STRINGIFY( \
      __GNUC__) "." RLMESH_DETAIL_STRINGIFY(__GNUC_MINOR__) "\"}"
#else
#define RLMESH_DETAIL_HOST_COMPILER ""
#endif

/* MSVC's packed _MSC_VER (e.g. 1944) becomes 19.44. An immutable
 * character array keeps this safe in C and C++98 without runtime initialization. */
#if defined(_MSC_VER) && !defined(__clang_major__) && !defined(__GNUC__)
static inline const char* rlmesh_detail_msvc_host_runtime(void) {
  /* clang-format off */
  static const char runtime[] = {
    '{', '"',
    'l', 'a', 'n', 'g', 'u', 'a', 'g', 'e', '"',
    ':', '"',
#ifdef __cplusplus
    'c', '+', '+',
#else
    'c',
#endif
    '"',
#ifdef RLMESH_DETAIL_HOST_VERSION
    ',', '"',
    'l', 'a', 'n', 'g', 'u', 'a', 'g', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '"',
    ':', '"',
    RLMESH_DETAIL_HOST_VERSION_CHARS, '"',
#endif
    ',', '"',
    't', 'o', 'o', 'l', 'c', 'h', 'a', 'i', 'n', '"',
    ':', '{', '"',
    'n', 'a', 'm', 'e', '"',
    ':', '"',
    'm', 's', 'v', 'c', '"',
    ',', '"',
    'v', 'e', 'r', 's', 'i', 'o', 'n', '"',
    ':', '"',
    '0' + (_MSC_VER / 1000) % 10, '0' + (_MSC_VER / 100) % 10, '.', '0' + (_MSC_VER / 10) % 10, '0' + _MSC_VER % 10,
    '"',
    '}', '}', '\0'
  };
  /* clang-format on */
  return runtime;
}
#define RLMESH_HOST_RUNTIME_JSON rlmesh_detail_msvc_host_runtime()
#else
#ifdef RLMESH_DETAIL_HOST_VERSION
#define RLMESH_DETAIL_HOST_STANDARD ",\"language_version\":\"" RLMESH_DETAIL_HOST_VERSION "\""
#else
#define RLMESH_DETAIL_HOST_STANDARD ""
#endif
#define RLMESH_HOST_RUNTIME_JSON                 \
  "{\"language\":\"" RLMESH_DETAIL_HOST_LANGUAGE \
  "\"" RLMESH_DETAIL_HOST_STANDARD RLMESH_DETAIL_HOST_COMPILER "}"
#endif

/* What the env declares. Set struct_size = sizeof(RLMeshExtLegacyEnvConfig); fields
 * beyond it read as unset. Borrowed for the rlmesh_ext_legacy_env_new call only (the
 * spaces are cloned). `adapter_tags_json` is the env's adapter EnvTags (v1
 * JSON), validated against the spaces right there: an unknown field or a tag
 * that does not fit its space fails rlmesh_ext_legacy_env_new. The JSON grammar is in
 * docs/specs/env_tags.v1.md. */
typedef struct RLMeshExtLegacyEnvConfig {
  size_t struct_size;
  const char* id;                           /* contract id; NULL/"" = "UnknownEnv-v1" */
  const RLMeshSpaceSpec* observation_space; /* required */
  const RLMeshSpaceSpec* action_space;      /* required */
  const char* adapter_tags_json;            /* NULL = untagged */
  const char* const* reset_options;         /* e.g. {"trial_index"}; NULL = none */
  size_t num_reset_options;
  const char* render_mode;   /* "rgb_array" when `render` produces frames */
  const char* metadata_json; /* extra contract metadata (JSON object); NULL = none */
  /* Run every callback (close included) on the thread that calls
   * rlmesh_ext_legacy_env_serve, which then serves from a helper thread: for a simulation
   * bound to the thread that created it (a GL / Vulkan context, Isaac Sim).
   * One lane only. false = callbacks run on a dedicated lane thread. */
  bool foreground;
  /* The named variants (scenarios) one image serves, as the describe
   * envelope's variants.catalog: a JSON array of {"id", "params", "metadata"},
   * e.g. [{"id": "viper/firm", "params": {"vehicle": "viper"}, "metadata":
   * {"name": "Viper on firm soil"}}]. Ids must be unique. NULL = none. */
  const char* variants_catalog_json;
  /* Optional host runtime: JSON object with language / language_version strings,
   * framework_versions (framework name -> version strings), and toolchain
   * {"name": "gcc"/"clang"/"msvc", "version": "major.minor"}. No other keys.
   * Strings and framework names are trimmed and must remain non-empty.
   * Copied at creation; NULL = legacy runtime (language "c", no version/compiler).
   * Use rlmesh_ext_legacy_env_config_init() to detect the host automatically. */
  const char* host_runtime_json;
} RLMeshExtLegacyEnvConfig;

/* Zero defaults plus struct_size and the host translation unit's runtime.
 * Fill the required spaces afterward; override host_runtime_json if needed. */
static inline RLMeshExtLegacyEnvConfig rlmesh_ext_legacy_env_config_init(void) {
#ifdef __cplusplus
  RLMeshExtLegacyEnvConfig config = RLMeshExtLegacyEnvConfig();
#else
  RLMeshExtLegacyEnvConfig config = {0};
#endif
  config.struct_size = sizeof(RLMeshExtLegacyEnvConfig);
  config.host_runtime_json = RLMESH_HOST_RUNTIME_JSON;
  return config;
}

/* An owned env handle. Lifecycle: rlmesh_ext_legacy_env_new -> rlmesh_ext_legacy_env_bind ->
 * rlmesh_ext_legacy_env_serve (blocks) -> rlmesh_ext_legacy_env_free. rlmesh_ext_legacy_env_cancel
 * is the only call that may overlap a running serve, from any thread. */
typedef struct RLMeshExtLegacyEnv RLMeshExtLegacyEnv;

RLMESH_API RLMeshStatus rlmesh_ext_legacy_env_new(const RLMeshExtLegacyEnvVtable* vtable,
                                                  const RLMeshExtLegacyEnvConfig* config,
                                                  void* user_data, RLMeshExtLegacyEnv** out);

/* Serve `num_lanes` independent simulations as the lanes of one endpoint
 * (num_envs = num_lanes): lane i passes `user_data[i]` to its callbacks. Every
 * lane shares `vtable` and `config` (one contract) but runs on its own thread,
 * concurrently with the others, so each `user_data` must be its own simulation.
 * `user_data` (the array) is borrowed for the call; each entry must outlive the
 * env. num_lanes == 0 and a foreground config with num_lanes > 1 fail with
 * RLMESH_ERR_INVALID_ARGUMENT. */
RLMESH_API RLMeshStatus rlmesh_ext_legacy_env_new_lanes(const RLMeshExtLegacyEnvVtable* vtable,
                                                        const RLMeshExtLegacyEnvConfig* config,
                                                        void* const* user_data, size_t num_lanes,
                                                        RLMeshExtLegacyEnv** out);

/* Bind to `bind_address` (tcp://host:port, host:port or unix:///path) without
 * serving yet; once per handle. `out_address` (may be NULL) receives the
 * resolved address, e.g. the OS-assigned port for port 0 (UTF-8, not
 * NUL-terminated; free with rlmesh_bytes_free). `options` may be NULL;
 * `predict_concurrency` and `close_timeout_ms` do not apply to an env. A failed
 * bind (an address in use, a bad RLMESH_ENV_ENDPOINT_TOKEN) leaves the handle
 * as it was: bind may be retried, and `close` still runs once per lane. */
RLMESH_API RLMeshStatus rlmesh_ext_legacy_env_bind(RLMeshExtLegacyEnv* env,
                                                   const char* bind_address,
                                                   const RLMeshExtLegacyServeOptions* options,
                                                   RLMeshBytes* out_address);

/* The env's describe envelope (rlmesh.describe.v1: target "native:<id>", the
 * spaces as env_spec, the published adapter tags as env_tags, and the runtime
 * edition handshake) as UTF-8 JSON into `out` (not NUL-terminated; free with
 * rlmesh_bytes_free). Valid before and after bind; after bind it declares the
 * workflow edition the server was bound with. rlmesh_ext_legacy_env_bind also puts it on
 * the handshake PeerInfo.extra under "rlmesh.describe.v1" of its own endpoint
 * (each env bound in one process reports its own; a failed bind publishes
 * nothing), which is where the managed
 * platform reads it for an image without a baked `dev.rlmesh.describe` label.
 * Bake that label with this output to describe the image before it runs. */
RLMESH_API RLMeshStatus rlmesh_ext_legacy_env_describe_json(const RLMeshExtLegacyEnv* env,
                                                            RLMeshBytes* out);

/* Serve the bound env until a remote shutdown, an idle timeout, or
 * rlmesh_ext_legacy_env_cancel. Blocking. `close` runs once per lane before this returns.
 * A foreground env runs every callback on the calling thread until then.
 *
 * When this returns, no callback is running and none will run again, so the
 * `user_data` may be freed. It waits for a callback still in flight (a step the
 * drain timeout gave up on) and for every `close`, however long: a callback
 * that never returns blocks this forever. `close_timeout_ms` does not apply. */
RLMESH_API RLMeshStatus rlmesh_ext_legacy_env_serve(RLMeshExtLegacyEnv* env);

/* Stop a blocking rlmesh_ext_legacy_env_serve from another thread: it drains, closes the
 * env and returns RLMESH_OK. Terminal for the handle. NULL is a no-op. */
RLMESH_API void rlmesh_ext_legacy_env_cancel(RLMeshExtLegacyEnv* env);

/* Free an env handle; not from inside its own callbacks. NULL is a no-op. Like
 * rlmesh_ext_legacy_env_serve, it returns only once no callback is running or can run.
 * For a handle that never served it runs each lane's `close` first: a bound
 * one stops its server (closing on the lane threads), an unbound one (a failed
 * bind included) closes on the calling thread. `close` runs once per lane over
 * the handle's life; `user_data` may be freed once this returns. */
RLMESH_API void rlmesh_ext_legacy_env_free(RLMeshExtLegacyEnv* env);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RLMESH_H */
