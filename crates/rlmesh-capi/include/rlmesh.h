/* RLMesh C ABI — experimental. Both v1 paths: a C/C++ model driving a remote
 * environment, and a C/C++ environment served to remote models. Hand-authored;
 * cbindgen-verifiable (a header-drift CI gate is a follow-up). */
#ifndef RLMESH_H
#define RLMESH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Vendored verbatim from dmlc/dlpack v1.0, include/dlpack/dlpack.h.
 * Copyright (c) 2017 by Contributors. Licensed under Apache-2.0;
 * see dlpack.LICENSE for the complete license and disclaimer.
 * https://github.com/dmlc/dlpack/tree/v1.0
 * Keep this upstream block unchanged, including its own include guard. */
/* clang-format off */
#if defined(__cplusplus) && __cplusplus < 201103L && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wc++11-extensions"
#if !defined(__clang__)
/* GCC diagnoses upstream's trailing enum commas as pedantic in C++98. */
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#endif
/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

/**
 * \brief Compatibility with C++
 */
#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

/*! \brief The current major version of dlpack */
#define DLPACK_MAJOR_VERSION 1

/*! \brief The current minor version of dlpack */
#define DLPACK_MINOR_VERSION 0

/*! \brief DLPACK_DLL prefix for windows */
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief The DLPack version.
 *
 * A change in major version indicates that we have changed the
 * data layout of the ABI - DLManagedTensorVersioned.
 *
 * A change in minor version indicates that we have added new
 * code, such as a new device type, but the ABI is kept the same.
 *
 * If an obtained DLPack tensor has a major version that disagrees
 * with the version number specified in this header file
 * (i.e. major != DLPACK_MAJOR_VERSION), the consumer must call the deleter
 * (and it is safe to do so). It is not safe to access any other fields
 * as the memory layout will have changed.
 *
 * In the case of a minor version mismatch, the tensor can be safely used as
 * long as the consumer knows how to interpret all fields. Minor version
 * updates indicate the addition of enumeration values.
 */
typedef struct {
  /*! \brief DLPack major version. */
  uint32_t major;
  /*! \brief DLPack minor version. */
  uint32_t minor;
} DLPackVersion;

/*!
 * \brief The device type in DLDevice.
 */
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLCUDA = 2,
  /*!
   * \brief Pinned CUDA CPU memory by cudaMallocHost
   */
  kDLCUDAHost = 3,
  /*! \brief OpenCL devices. */
  kDLOpenCL = 4,
  /*! \brief Vulkan buffer for next generation graphics. */
  kDLVulkan = 7,
  /*! \brief Metal for Apple GPU. */
  kDLMetal = 8,
  /*! \brief Verilog simulator buffer */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*!
   * \brief Pinned ROCm CPU memory allocated by hipMallocHost
   */
  kDLROCMHost = 11,
  /*!
   * \brief Reserved extension device type,
   * used for quickly test extension device
   * The semantics can differ depending on the implementation.
   */
  kDLExtDev = 12,
  /*!
   * \brief CUDA managed/unified memory allocated by cudaMallocManaged
   */
  kDLCUDAManaged = 13,
  /*!
   * \brief Unified shared memory allocated on a oneAPI non-partititioned
   * device. Call to oneAPI runtime is required to determine the device
   * type, the USM allocation type and the sycl context it is bound to.
   *
   */
  kDLOneAPI = 14,
  /*! \brief GPU support for next generation WebGPU standard. */
  kDLWebGPU = 15,
  /*! \brief Qualcomm Hexagon DSP */
  kDLHexagon = 16,
  /*! \brief Microsoft MAIA devices */
  kDLMAIA = 17,
} DLDeviceType;

/*!
 * \brief A Device for Tensor and operator.
 */
typedef struct {
  /*! \brief The device type used in the device. */
  DLDeviceType device_type;
  /*!
   * \brief The device index.
   * For vanilla CPU memory, pinned memory, or managed memory, this is set to 0.
   */
  int32_t device_id;
} DLDevice;

/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
  /*! \brief signed integer */
  kDLInt = 0U,
  /*! \brief unsigned integer */
  kDLUInt = 1U,
  /*! \brief IEEE floating point */
  kDLFloat = 2U,
  /*!
   * \brief Opaque handle type, reserved for testing purposes.
   * Frameworks need to agree on the handle data type for the exchange to be well-defined.
   */
  kDLOpaqueHandle = 3U,
  /*! \brief bfloat16 */
  kDLBfloat = 4U,
  /*!
   * \brief complex number
   * (C/C++/Python layout: compact struct per complex number)
   */
  kDLComplex = 5U,
  /*! \brief boolean */
  kDLBool = 6U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 * native endian-ness. An explicit error message should be raised when attempting to
 * export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes = 1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes = 4
 *   - int8: type_code = 0, bits = 8, lanes = 1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 *   - bool: type_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits)
 */
typedef struct {
  /*!
   * \brief Type code of base types.
   * We keep it uint8_t instead of DLDataTypeCode for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   * */
  uint8_t code;
  /*!
   * \brief Number of bits, common choices are 8, 16, 32.
   */
  uint8_t bits;
  /*! \brief Number of lanes in the type, used for vector types. */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief Plain C Tensor object, does not manage memory.
 */
typedef struct {
  /*!
   * \brief The data pointer points to the allocated data. This will be CUDA
   * device pointer or cl_mem handle in OpenCL. It may be opaque on some device
   * types. This pointer is always aligned to 256 bytes as in CUDA. The
   * `byte_offset` field should be used to point to the beginning of the data.
   *
   * Note that as of Nov 2021, multiply libraries (CuPy, PyTorch, TensorFlow,
   * TVM, perhaps others) do not adhere to this 256 byte aligment requirement
   * on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must be fixed
   * (after which this note will be updated); at the moment it is recommended
   * to not rely on the data pointer being correctly aligned.
   *
   * For given DLTensor, the size of memory required to store the contents of
   * data is calculated as follows:
   *
   * \code{.c}
   * static inline size_t GetDataSize(const DLTensor* t) {
   *   size_t size = 1;
   *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
   *     size *= t->shape[i];
   *   }
   *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
   *   return size;
   * }
   * \endcode
   *
   * Note that if the tensor is of size zero, then the data pointer should be
   * set to `NULL`.
   */
  void* data;
  /*! \brief The device of the tensor */
  DLDevice device;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief The data type of the pointer*/
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
  /*!
   * \brief strides of the tensor (in number of elements, not bytes)
   *  can be NULL, indicating tensor is compact and row-majored.
   */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief C Tensor object, manage memory of DLTensor. This data structure is
 *  intended to facilitate the borrowing of DLTensor by another framework. It is
 *  not meant to transfer the tensor. When the borrowing framework doesn't need
 *  the tensor, it should call the deleter to notify the host that the resource
 *  is no longer needed.
 *
 * \note This data structure is used as Legacy DLManagedTensor
 *       in DLPack exchange and is deprecated after DLPack v0.8
 *       Use DLManagedTensorVersioned instead.
 *       This data structure may get renamed or deleted in future versions.
 *
 * \sa DLManagedTensorVersioned
 */
typedef struct DLManagedTensor {
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
  /*! \brief the context of the original host framework of DLManagedTensor in
   *   which DLManagedTensor is used in the framework. It can also be NULL.
   */
  void * manager_ctx;
  /*!
   * \brief Destructor - this should be called
   * to destruct the manager_ctx  which backs the DLManagedTensor. It can be
   * NULL if there is no way for the caller to provide a reasonable destructor.
   * The destructor deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensor * self);
} DLManagedTensor;

// bit masks used in in the DLManagedTensorVersioned

/*! \brief bit mask to indicate that the tensor is read only. */
#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0UL)

/*!
 * \brief bit mask to indicate that the tensor is a copy made by the producer.
 *
 * If set, the tensor is considered solely owned throughout its lifetime by the
 * consumer, until the producer-provided deleter is invoked.
 */
#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1UL)

/*!
 * \brief A versioned and managed C Tensor object, manage memory of DLTensor.
 *
 * This data structure is intended to facilitate the borrowing of DLTensor by
 * another framework. It is not meant to transfer the tensor. When the borrowing
 * framework doesn't need the tensor, it should call the deleter to notify the
 * host that the resource is no longer needed.
 *
 * \note This is the current standard DLPack exchange data structure.
 */
struct DLManagedTensorVersioned {
  /*!
   * \brief The API and ABI version of the current managed Tensor
   */
  DLPackVersion version;
  /*!
   * \brief the context of the original host framework.
   *
   * Stores DLManagedTensorVersioned is used in the
   * framework. It can also be NULL.
   */
  void *manager_ctx;
  /*!
   * \brief Destructor.
   *
   * This should be called to destruct manager_ctx which holds the DLManagedTensorVersioned.
   * It can be NULL if there is no way for the caller to provide a reasonable
   * destructor. The destructor deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensorVersioned *self);
  /*!
   * \brief Additional bitmask flags information about the tensor.
   *
   * By default the flags should be set to 0.
   *
   * \note Future ABI changes should keep everything until this field
   *       stable, to ensure that deleter can be correctly called.
   *
   * \sa DLPACK_FLAG_BITMASK_READ_ONLY
   * \sa DLPACK_FLAG_BITMASK_IS_COPIED
   */
  uint64_t flags;
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
};

#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif  // DLPACK_DLPACK_H_
#if defined(__cplusplus) && __cplusplus < 201103L && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
/* clang-format on */
#if !defined(DLPACK_MAJOR_VERSION) || DLPACK_MAJOR_VERSION < 1
#error "RLMesh requires DLPack 1.0 or newer; include rlmesh.h before a pre-1.0 dlpack.h"
#endif

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
 * RLMeshServeOptions.workflow_edition, else the RLMESH_WORKFLOW_EDITION
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
                                                   void (*free_user_data)(void* user_data));

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

/* DLPack 1.x, versioned descriptors only. Options follow R-19: NULL = INIT;
 * struct_size below MIN_SIZE is refused; reserved/unknown bytes must be zero.
 * copy=false exports shared read-only storage; copy=true owns writable storage.
 * dl_device, when present, must be (kDLCPU, 0). No implicit padding. */
typedef struct RLMeshToDlpackOptions {
  size_t struct_size;
  bool copy;
  bool has_dl_device;
  uint16_t _reserved0;
  int32_t dl_device_type;
  int32_t dl_device_id;
  uint32_t _reserved1;
} RLMeshToDlpackOptions;
#define RLMESH_TO_DLPACK_OPTIONS_INIT {sizeof(RLMeshToDlpackOptions), false, false, 0, 0, 0, 0}
#define RLMESH_TO_DLPACK_OPTIONS_MIN_SIZE sizeof(RLMeshToDlpackOptions)
/* obj must be exclusively owned, with live readable shape/strides and CPU data
 * for its entire described layout. Import copies, then calls obj->deleter once
 * on success (if non-NULL). On error C still owns obj. A non-NULL out is required
 * and is set to NULL on error, including invalid descriptors. */
RLMESH_API RLMeshStatus rlmesh_tensor_from_dlpack(struct DLManagedTensorVersioned* obj,
                                                  RLMeshTensor** out);
/* On success *out owns the descriptor, its layout and storage, even after tensor
 * is freed. The caller releases it once through (*out)->deleter(*out). On error
 * *out is NULL. Never free a DLPack descriptor with rlmesh_tensor_free. */
RLMESH_API RLMeshStatus rlmesh_tensor_to_dlpack(const RLMeshTensor* tensor,
                                                const RLMeshToDlpackOptions* options,
                                                struct DLManagedTensorVersioned** out);
RLMESH_API RLMeshStatus rlmesh_tensor_dlpack_device(const RLMeshTensor* tensor, int32_t* out_type,
                                                    int32_t* out_id);

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

/* Free an owned immutable spec; NULL is a no-op. */
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

/* Space(spec), clone (shares the sampler), free. Free an owned spec with
 * rlmesh_space_spec_free; the two opaque handle types are distinct. */
RLMESH_API RLMeshStatus rlmesh_space_new(const RLMeshSpaceSpec* spec, RLMeshSpace** out);
RLMESH_API RLMeshSpace* rlmesh_space_clone(const RLMeshSpace* space);
RLMESH_API void rlmesh_space_free(RLMeshSpace* space);
/* The space's spec, borrowed while any handle sharing the space lives. */
RLMESH_API const RLMeshSpaceSpec* rlmesh_space_spec(const RLMeshSpace* space);
RLMESH_API RLMeshSpaceKind rlmesh_space_kind(const RLMeshSpace* space);
RLMESH_API RLMeshDType rlmesh_space_dtype(const RLMeshSpace* space);
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
RLMESH_API RLMeshStatus rlmesh_dict_new(const RLMeshStr* spaces_keys,
                                        const RLMeshSpaceSpec* const* spaces, size_t spaces_len,
                                        RLMeshDict** out);
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

/* Experimental spaces. Handles follow the same clone/free rules as Space. */
RLMESH_API RLMeshStatus rlmesh_multi_binary_new(const int64_t* shape, size_t shape_len,
                                                RLMeshSpace** out);
RLMESH_API RLMeshStatus rlmesh_multi_discrete_new(const int64_t* nvec, size_t nvec_len,
                                                  RLMeshDType dtype, RLMeshSpace** out);
RLMESH_API RLMeshStatus rlmesh_text_new(int64_t max_length, int64_t min_length, const char* charset,
                                        RLMeshSpace** out);
RLMESH_API RLMeshStatus rlmesh_tuple_new(const RLMeshSpaceSpec* const* spaces, size_t spaces_len,
                                         RLMeshSpace** out);
RLMESH_API RLMeshStatus rlmesh_multi_discrete_nvec(const RLMeshSpace* space, int64_t* out,
                                                   size_t capacity, size_t* out_len);
RLMESH_API RLMeshStatus rlmesh_text_limits(const RLMeshSpace* space, int64_t* out_max_length,
                                           int64_t* out_min_length, RLMeshStr* out_charset);
RLMESH_API RLMeshStatus rlmesh_tuple_len(const RLMeshSpace* space, size_t* out);
RLMESH_API RLMeshStatus rlmesh_tuple_get(const RLMeshSpace* space, size_t index, RLMeshSpace** out);

typedef struct RLMeshEnv RLMeshEnv;
typedef struct RLMeshVectorEnv RLMeshVectorEnv;
/* Stable adapters: facade-backed typed trees and shared callables. */
typedef struct RLMeshRotation RLMeshRotation;
typedef struct RLMeshCustomEncoding RLMeshCustomEncoding;
typedef struct RLMeshImage RLMeshImage;
typedef struct RLMeshState RLMeshState;
typedef struct RLMeshConcat RLMeshConcat;
typedef struct RLMeshConstant RLMeshConstant;
typedef struct RLMeshAdaptersText RLMeshAdaptersText;
typedef struct RLMeshCustom RLMeshCustom;
typedef struct RLMeshActuator RLMeshActuator;
typedef struct RLMeshAction RLMeshAction;
typedef struct RLMeshImageTag RLMeshImageTag;
typedef struct RLMeshStateTag RLMeshStateTag;
typedef struct RLMeshTextTag RLMeshTextTag;
typedef struct RLMeshField RLMeshField;
typedef struct RLMeshSplit RLMeshSplit;
typedef struct RLMeshPreviousOptions RLMeshPreviousOptions;
typedef struct RLMeshObservationRoles RLMeshObservationRoles;
typedef struct RLMeshAdvisory RLMeshAdvisory;
typedef struct RLMeshTagOptions RLMeshTagOptions;
typedef struct RLMeshResolveOptions RLMeshResolveOptions;
typedef struct RLMeshObsTransform RLMeshObsTransform;
typedef struct RLMeshRotationTransform RLMeshRotationTransform;
typedef struct RLMeshModelSpec RLMeshModelSpec;
typedef struct RLMeshEnvTags RLMeshEnvTags;
typedef struct RLMeshAdapter RLMeshAdapter;
typedef struct RLMeshWrappedPredict RLMeshWrappedPredict;
typedef enum RLMeshImageLayout {
  RLMESH_IMAGE_LAYOUT_INVALID = 0,
  RLMESH_IMAGE_LAYOUT_HWC = 1,
  RLMESH_IMAGE_LAYOUT_CHW = 2
} RLMeshImageLayout;
RLMESH_API const char* rlmesh_image_layout_str(RLMeshImageLayout value);
typedef enum RLMeshFitMode {
  RLMESH_FIT_MODE_INVALID = 0,
  RLMESH_FIT_MODE_STRETCH = 1,
  RLMESH_FIT_MODE_CROP = 2,
  RLMESH_FIT_MODE_PAD = 3,
  RLMESH_FIT_MODE_OTHER = 1000
} RLMeshFitMode;
RLMESH_API const char* rlmesh_fit_mode_str(RLMeshFitMode value);
typedef enum RLMeshFrame {
  RLMESH_FRAME_INVALID = 0,
  RLMESH_FRAME_WORLD = 1,
  RLMESH_FRAME_ROBOT_BASE = 2,
  RLMESH_FRAME_TOOL = 3,
  RLMESH_FRAME_OTHER = 1000
} RLMeshFrame;
RLMESH_API const char* rlmesh_frame_str(RLMeshFrame value);
typedef enum RLMeshReference {
  RLMESH_REFERENCE_INVALID = 0,
  RLMESH_REFERENCE_CURRENT = 1,
  RLMESH_REFERENCE_TARGET = 2,
  RLMESH_REFERENCE_OTHER = 1000
} RLMeshReference;
RLMESH_API const char* rlmesh_reference_str(RLMeshReference value);
typedef enum RLMeshProvenance {
  RLMESH_PROVENANCE_INVALID = 0,
  RLMESH_PROVENANCE_SENSED = 1,
  RLMESH_PROVENANCE_ESTIMATED = 2,
  RLMESH_PROVENANCE_PRIVILEGED = 3,
  RLMESH_PROVENANCE_OTHER = 1000
} RLMeshProvenance;
RLMESH_API const char* rlmesh_provenance_str(RLMeshProvenance value);
typedef enum RLMeshResample {
  RLMESH_RESAMPLE_INVALID = 0,
  RLMESH_RESAMPLE_BILINEAR = 1,
  RLMESH_RESAMPLE_BILINEAR_AA = 2,
  RLMESH_RESAMPLE_BICUBIC_AA = 3,
  RLMESH_RESAMPLE_LANCZOS3_AA = 4,
  RLMESH_RESAMPLE_AREA = 5,
  RLMESH_RESAMPLE_OTHER = 1000
} RLMeshResample;
RLMESH_API const char* rlmesh_resample_str(RLMeshResample value);
typedef enum RLMeshRotationEncoding {
  RLMESH_ROTATION_ENCODING_INVALID = 0,
  RLMESH_ROTATION_ENCODING_QUAT_XYZW = 1,
  RLMESH_ROTATION_ENCODING_QUAT_WXYZ = 2,
  RLMESH_ROTATION_ENCODING_AXIS_ANGLE = 3,
  RLMESH_ROTATION_ENCODING_ROT6D = 4,
  RLMESH_ROTATION_ENCODING_ROT6D_ROWMAJOR = 5,
  RLMESH_ROTATION_ENCODING_EULER_XYZ = 6,
  RLMESH_ROTATION_ENCODING_GRAVITY_XYZ = 7,
  RLMESH_ROTATION_ENCODING_OTHER = 1000
} RLMeshRotationEncoding;
RLMESH_API const char* rlmesh_rotation_encoding_str(RLMeshRotationEncoding value);
typedef enum RLMeshStackPad {
  RLMESH_STACK_PAD_INVALID = 0,
  RLMESH_STACK_PAD_FIRST = 1,
  RLMESH_STACK_PAD_BLACK = 2
} RLMeshStackPad;
RLMESH_API const char* rlmesh_stack_pad_str(RLMeshStackPad value);
typedef enum RLMeshCropMode {
  RLMESH_CROP_MODE_INVALID = 0,
  RLMESH_CROP_MODE_ZOOM = 1,
  RLMESH_CROP_MODE_SLICE = 2,
  RLMESH_CROP_MODE_OTHER = 1000
} RLMeshCropMode;
RLMESH_API const char* rlmesh_crop_mode_str(RLMeshCropMode value);
typedef enum RLMeshChannelOrder {
  RLMESH_CHANNEL_ORDER_INVALID = 0,
  RLMESH_CHANNEL_ORDER_RGB = 1,
  RLMESH_CHANNEL_ORDER_BGR = 2,
  RLMESH_CHANNEL_ORDER_OTHER = 1000
} RLMeshChannelOrder;
RLMESH_API const char* rlmesh_channel_order_str(RLMeshChannelOrder value);
typedef enum RLMeshStateContainer {
  RLMESH_STATE_CONTAINER_INVALID = 0,
  RLMESH_STATE_CONTAINER_ARRAY = 1,
  RLMESH_STATE_CONTAINER_LIST = 2
} RLMeshStateContainer;
RLMESH_API const char* rlmesh_state_container_str(RLMeshStateContainer value);
typedef enum RLMeshTextContainer {
  RLMESH_TEXT_CONTAINER_INVALID = 0,
  RLMESH_TEXT_CONTAINER_STR = 1,
  RLMESH_TEXT_CONTAINER_LIST = 2
} RLMeshTextContainer;
RLMESH_API const char* rlmesh_text_container_str(RLMeshTextContainer value);
typedef enum RLMeshStateSource {
  RLMESH_STATE_SOURCE_INVALID = 0,
  RLMESH_STATE_SOURCE_OBSERVATION = 1,
  RLMESH_STATE_SOURCE_ACTION = 2
} RLMeshStateSource;
RLMESH_API const char* rlmesh_state_source_str(RLMeshStateSource value);
typedef enum RLMeshAdvisorySeverity {
  RLMESH_ADVISORY_SEVERITY_INVALID = 0,
  RLMESH_ADVISORY_SEVERITY_INFO = 1,
  RLMESH_ADVISORY_SEVERITY_CAUTION = 2
} RLMeshAdvisorySeverity;
RLMESH_API const char* rlmesh_advisory_severity_str(RLMeshAdvisorySeverity value);
typedef enum RLMeshNormalize {
  RLMESH_NORMALIZE_INVALID = 0,
  RLMESH_NORMALIZE_OFF = 1,
  RLMESH_NORMALIZE_UNIT = 2,
  RLMESH_NORMALIZE_RANGE = 3
} RLMeshNormalize;
typedef enum RLMeshInputNodeKind {
  RLMESH_INPUT_NODE_KIND_INVALID = 0,
  RLMESH_INPUT_NODE_KIND_IMAGE = 1,  /* leaf: const RLMeshImage* */
  RLMESH_INPUT_NODE_KIND_STATE = 2,  /* leaf: const RLMeshState* */
  RLMESH_INPUT_NODE_KIND_CONCAT = 3, /* leaf: const RLMeshConcat* */
  RLMESH_INPUT_NODE_KIND_TEXT = 4,   /* leaf: const RLMeshAdaptersText* */
  RLMESH_INPUT_NODE_KIND_CUSTOM = 5, /* leaf: const RLMeshCustom* */
  RLMESH_INPUT_NODE_KIND_DICT = 6,   /* keys + children */
  RLMESH_INPUT_NODE_KIND_TUPLE = 7   /* children */
} RLMeshInputNodeKind;
typedef struct RLMeshInputNode {
  RLMeshInputNodeKind kind;
  const void* leaf;                       /* leaf kinds; NULL otherwise */
  const RLMeshStr* keys;                  /* DICT: children_len keys; NULL otherwise */
  const struct RLMeshInputNode* children; /* DICT, TUPLE; NULL otherwise */
  size_t children_len;
} RLMeshInputNode;

typedef enum RLMeshObsNodeKind {
  RLMESH_OBS_NODE_KIND_INVALID = 0,
  RLMESH_OBS_NODE_KIND_IMAGE_TAG = 1, /* leaf: const RLMeshImageTag* */
  RLMESH_OBS_NODE_KIND_STATE_TAG = 2, /* leaf: const RLMeshStateTag* */
  RLMESH_OBS_NODE_KIND_TEXT_TAG = 3,  /* leaf: const RLMeshTextTag* */
  RLMESH_OBS_NODE_KIND_SPLIT = 4,     /* leaf: const RLMeshSplit* */
  RLMESH_OBS_NODE_KIND_DICT = 5,
  RLMESH_OBS_NODE_KIND_TUPLE = 6
} RLMeshObsNodeKind;
typedef struct RLMeshObsNode {
  RLMeshObsNodeKind kind;
  const void* leaf;
  const RLMeshStr* keys;
  const struct RLMeshObsNode* children;
  size_t children_len;
} RLMeshObsNode;

typedef enum RLMeshConcatPartKind {
  RLMESH_CONCAT_PART_KIND_INVALID = 0,
  RLMESH_CONCAT_PART_KIND_ROLE = 1,    /* role */
  RLMESH_CONCAT_PART_KIND_STATE = 2,   /* leaf: const RLMeshState* */
  RLMESH_CONCAT_PART_KIND_CONSTANT = 3 /* leaf: const RLMeshConstant* */
} RLMeshConcatPartKind;
typedef struct RLMeshConcatPart {
  RLMeshConcatPartKind kind;
  const char* role; /* ROLE; NULL otherwise */
  const void* leaf; /* STATE, CONSTANT; NULL otherwise */
} RLMeshConcatPart;
RLMeshConcatPart rlmesh_concat_part_role(const char* role);
RLMeshConcatPart rlmesh_concat_part_state(const RLMeshState* leaf);
RLMeshConcatPart rlmesh_concat_part_constant(const RLMeshConstant* leaf);

typedef struct RLMeshEncodingArm {
  const RLMeshRotationTransform* transform;
  const char* entrypoint;
} RLMeshEncodingArm;
struct RLMeshRotation {
  size_t struct_size;
  RLMeshRotationEncoding encoding;
  uint32_t _reserved;
  const double* value;
  size_t value_len;
};
#define RLMESH_ROTATION_INIT {sizeof(RLMeshRotation), (RLMeshRotationEncoding)0, 0, NULL, 0}
#define RLMESH_ROTATION_MIN_SIZE sizeof(RLMeshRotation)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshCustomEncoding {
  size_t struct_size;
  RLMeshRotationEncoding base;
  uint8_t _reserved0[4];
  RLMeshEncodingArm from_base;
  RLMeshEncodingArm to_base;
  const char* name;
};
#define RLMESH_CUSTOM_ENCODING_INIT \
  {sizeof(RLMeshCustomEncoding),    \
   (RLMeshRotationEncoding)0,       \
   {0},                             \
   {NULL, NULL},                    \
   {NULL, NULL},                    \
   "custom"}
#else
struct RLMeshCustomEncoding {
  size_t struct_size;
  RLMeshRotationEncoding base;
  RLMeshEncodingArm from_base;
  RLMeshEncodingArm to_base;
  const char* name;
};
#define RLMESH_CUSTOM_ENCODING_INIT \
  {sizeof(RLMeshCustomEncoding), (RLMeshRotationEncoding)0, {NULL, NULL}, {NULL, NULL}, "custom"}
#endif
#define RLMESH_CUSTOM_ENCODING_MIN_SIZE sizeof(RLMeshCustomEncoding)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshImage {
  size_t struct_size;
  const char* role;
  bool has_height;
  uint8_t _reserved0[3];
  uint32_t height;
  bool has_width;
  uint8_t _reserved1[3];
  uint32_t width;
  RLMeshImageLayout layout;
  bool has_channels;
  uint8_t _reserved2[3];
  uint32_t channels;
  RLMeshDType dtype;
  const char* dtype_other;
  RLMeshNormalize normalize;
  uint8_t _reserved3[4];
  double normalize_range[2];
  uint32_t lead_dims;
  bool upside_down;
  uint8_t _reserved4[3];
  RLMeshResample resample;
  uint8_t _reserved5[4];
  const char* resample_other;
  bool allow_upscale;
  uint8_t _reserved6[7];
  const RLMeshFitMode* fit;
  size_t fit_len;
  const char* const* fit_other;
  bool optional;
  bool has_fill;
  uint8_t fill;
  uint8_t _reserved7[1];
  uint32_t stack;
  bool has_crop;
  uint8_t _reserved8[7];
  double crop;
  bool has_crop_area;
  uint8_t _reserved9[7];
  double crop_area;
  RLMeshCropMode crop_mode;
  uint8_t _reserved10[4];
  const char* crop_mode_other;
  bool has_jpeg_quality;
  uint8_t jpeg_quality;
  uint8_t _reserved11[2];
  RLMeshChannelOrder channel_order;
  const char* channel_order_other;
  bool has_render;
  uint8_t _reserved12[3];
  uint32_t render[2];
  uint8_t _reserved13[4];
  const int32_t* offsets;
  size_t offsets_len;
  RLMeshStackPad stack_pad;
  uint8_t _reserved14[4];
  const char* part;
  bool has_stride;
  uint8_t _reserved15[3];
  uint32_t stride;
};
#define RLMESH_IMAGE_INIT \
  {sizeof(RLMeshImage),   \
   NULL,                  \
   0,                     \
   {0},                   \
   0,                     \
   0,                     \
   {0},                   \
   0,                     \
   (RLMeshImageLayout)1,  \
   0,                     \
   {0},                   \
   0,                     \
   (RLMeshDType)2,        \
   NULL,                  \
   (RLMeshNormalize)1,    \
   {0},                   \
   {0, 0},                \
   0,                     \
   0,                     \
   {0},                   \
   (RLMeshResample)1,     \
   {0},                   \
   NULL,                  \
   0,                     \
   {0},                   \
   NULL,                  \
   0,                     \
   NULL,                  \
   0,                     \
   0,                     \
   0,                     \
   {0},                   \
   1,                     \
   0,                     \
   {0},                   \
   0,                     \
   0,                     \
   {0},                   \
   0,                     \
   (RLMeshCropMode)1,     \
   {0},                   \
   NULL,                  \
   0,                     \
   0,                     \
   {0},                   \
   (RLMeshChannelOrder)1, \
   NULL,                  \
   0,                     \
   {0},                   \
   {0, 0},                \
   {0},                   \
   NULL,                  \
   0,                     \
   (RLMeshStackPad)1,     \
   {0},                   \
   NULL,                  \
   0,                     \
   {0},                   \
   0}
#else
struct RLMeshImage {
  size_t struct_size;
  const char* role;
  bool has_height;
  uint8_t _reserved0[3];
  uint32_t height;
  bool has_width;
  uint8_t _reserved1[3];
  uint32_t width;
  RLMeshImageLayout layout;
  bool has_channels;
  uint8_t _reserved2[3];
  uint32_t channels;
  RLMeshDType dtype;
  const char* dtype_other;
  RLMeshNormalize normalize;
  double normalize_range[2];
  uint32_t lead_dims;
  bool upside_down;
  uint8_t _reserved3[3];
  RLMeshResample resample;
  const char* resample_other;
  bool allow_upscale;
  uint8_t _reserved4[3];
  const RLMeshFitMode* fit;
  size_t fit_len;
  const char* const* fit_other;
  bool optional;
  bool has_fill;
  uint8_t fill;
  uint8_t _reserved5[1];
  uint32_t stack;
  bool has_crop;
  uint8_t _reserved6[7];
  double crop;
  bool has_crop_area;
  uint8_t _reserved7[7];
  double crop_area;
  RLMeshCropMode crop_mode;
  const char* crop_mode_other;
  bool has_jpeg_quality;
  uint8_t jpeg_quality;
  uint8_t _reserved8[2];
  RLMeshChannelOrder channel_order;
  const char* channel_order_other;
  bool has_render;
  uint8_t _reserved9[3];
  uint32_t render[2];
  const int32_t* offsets;
  size_t offsets_len;
  RLMeshStackPad stack_pad;
  const char* part;
  bool has_stride;
  uint8_t _reserved10[3];
  uint32_t stride;
};
#define RLMESH_IMAGE_INIT \
  {sizeof(RLMeshImage),   \
   NULL,                  \
   0,                     \
   {0},                   \
   0,                     \
   0,                     \
   {0},                   \
   0,                     \
   (RLMeshImageLayout)1,  \
   0,                     \
   {0},                   \
   0,                     \
   (RLMeshDType)2,        \
   NULL,                  \
   (RLMeshNormalize)1,    \
   {0, 0},                \
   0,                     \
   0,                     \
   {0},                   \
   (RLMeshResample)1,     \
   NULL,                  \
   0,                     \
   {0},                   \
   NULL,                  \
   0,                     \
   NULL,                  \
   0,                     \
   0,                     \
   0,                     \
   {0},                   \
   1,                     \
   0,                     \
   {0},                   \
   0,                     \
   0,                     \
   {0},                   \
   0,                     \
   (RLMeshCropMode)1,     \
   NULL,                  \
   0,                     \
   0,                     \
   {0},                   \
   (RLMeshChannelOrder)1, \
   NULL,                  \
   0,                     \
   {0},                   \
   {0, 0},                \
   NULL,                  \
   0,                     \
   (RLMeshStackPad)1,     \
   NULL,                  \
   0,                     \
   {0},                   \
   0}
#endif
#define RLMESH_IMAGE_MIN_SIZE sizeof(RLMeshImage)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshState {
  size_t struct_size;
  const char* role;
  const RLMeshRotationEncoding* encoding;
  size_t encoding_len;
  const char* const* encoding_other;
  const RLMeshCustomEncoding* custom_encoding;
  bool has_dim;
  uint8_t _reserved0[3];
  uint32_t dim;
  bool has_index;
  uint8_t _reserved1[3];
  uint32_t index;
  bool optional;
  bool has_range;
  uint8_t _reserved2[6];
  double range[2];
  bool has_pad_to;
  uint8_t _reserved3[3];
  uint32_t pad_to;
  RLMeshDType dtype;
  uint8_t _reserved4[4];
  const char* dtype_other;
  const int64_t* reshape;
  size_t reshape_len;
  RLMeshStateContainer container;
  uint8_t _reserved5[4];
  double fill;
  const RLMeshRotation* post_rotate;
  bool has_scale;
  uint8_t _reserved6[7];
  double scale;
  const double* axis_scale;
  size_t axis_scale_len;
  bool has_offset;
  uint8_t _reserved7[7];
  double offset;
  const double* axis_offset;
  size_t axis_offset_len;
  bool has_frame;
  uint8_t _reserved8[3];
  RLMeshFrame frame;
  const char* frame_other;
  const RLMeshProvenance* provenance;
  size_t provenance_len;
  const char* const* provenance_other;
  const char* part;
  const char* const* labels;
  size_t labels_len;
  RLMeshStateSource source;
  bool has_clip;
  uint8_t _reserved9[3];
  double clip[2];
};
#define RLMESH_STATE_INIT   \
  {sizeof(RLMeshState),     \
   NULL,                    \
   NULL,                    \
   0,                       \
   NULL,                    \
   NULL,                    \
   0,                       \
   {0},                     \
   0,                       \
   0,                       \
   {0},                     \
   0,                       \
   0,                       \
   0,                       \
   {0},                     \
   {0, 0},                  \
   0,                       \
   {0},                     \
   0,                       \
   (RLMeshDType)11,         \
   {0},                     \
   NULL,                    \
   NULL,                    \
   0,                       \
   (RLMeshStateContainer)1, \
   {0},                     \
   0,                       \
   NULL,                    \
   0,                       \
   {0},                     \
   0,                       \
   NULL,                    \
   0,                       \
   0,                       \
   {0},                     \
   0,                       \
   NULL,                    \
   0,                       \
   0,                       \
   {0},                     \
   (RLMeshFrame)0,          \
   NULL,                    \
   NULL,                    \
   0,                       \
   NULL,                    \
   NULL,                    \
   NULL,                    \
   0,                       \
   (RLMeshStateSource)1,    \
   0,                       \
   {0},                     \
   {0, 0}}
#else
struct RLMeshState {
  size_t struct_size;
  const char* role;
  const RLMeshRotationEncoding* encoding;
  size_t encoding_len;
  const char* const* encoding_other;
  const RLMeshCustomEncoding* custom_encoding;
  bool has_dim;
  uint8_t _reserved0[3];
  uint32_t dim;
  bool has_index;
  uint8_t _reserved1[3];
  uint32_t index;
  bool optional;
  bool has_range;
  uint8_t _reserved2[6];
  double range[2];
  bool has_pad_to;
  uint8_t _reserved3[3];
  uint32_t pad_to;
  RLMeshDType dtype;
  const char* dtype_other;
  const int64_t* reshape;
  size_t reshape_len;
  RLMeshStateContainer container;
  uint8_t _reserved4[4];
  double fill;
  const RLMeshRotation* post_rotate;
  bool has_scale;
  uint8_t _reserved5[3];
  double scale;
  const double* axis_scale;
  size_t axis_scale_len;
  bool has_offset;
  uint8_t _reserved6[7];
  double offset;
  const double* axis_offset;
  size_t axis_offset_len;
  bool has_frame;
  uint8_t _reserved7[3];
  RLMeshFrame frame;
  const char* frame_other;
  const RLMeshProvenance* provenance;
  size_t provenance_len;
  const char* const* provenance_other;
  const char* part;
  const char* const* labels;
  size_t labels_len;
  RLMeshStateSource source;
  bool has_clip;
  uint8_t _reserved8[7];
  double clip[2];
};
#define RLMESH_STATE_INIT   \
  {sizeof(RLMeshState),     \
   NULL,                    \
   NULL,                    \
   0,                       \
   NULL,                    \
   NULL,                    \
   0,                       \
   {0},                     \
   0,                       \
   0,                       \
   {0},                     \
   0,                       \
   0,                       \
   0,                       \
   {0},                     \
   {0, 0},                  \
   0,                       \
   {0},                     \
   0,                       \
   (RLMeshDType)11,         \
   NULL,                    \
   NULL,                    \
   0,                       \
   (RLMeshStateContainer)1, \
   {0},                     \
   0,                       \
   NULL,                    \
   0,                       \
   {0},                     \
   0,                       \
   NULL,                    \
   0,                       \
   0,                       \
   {0},                     \
   0,                       \
   NULL,                    \
   0,                       \
   0,                       \
   {0},                     \
   (RLMeshFrame)0,          \
   NULL,                    \
   NULL,                    \
   0,                       \
   NULL,                    \
   NULL,                    \
   NULL,                    \
   0,                       \
   (RLMeshStateSource)1,    \
   0,                       \
   {0},                     \
   {0, 0}}
#endif
#define RLMESH_STATE_MIN_SIZE sizeof(RLMeshState)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshConcat {
  size_t struct_size;
  const RLMeshConcatPart* parts;
  size_t parts_len;
  bool has_pad_to;
  uint8_t _reserved0[3];
  uint32_t pad_to;
  RLMeshDType dtype;
  uint8_t _reserved1[4];
  const char* dtype_other;
  const int64_t* reshape;
  size_t reshape_len;
  RLMeshStateContainer container;
  bool has_clip;
  uint8_t _reserved2[3];
  double clip[2];
};
#define RLMESH_CONCAT_INIT                                                                 \
  {sizeof(RLMeshConcat),    NULL, 0,   0,     {0}, 0, (RLMeshDType)11, {0}, NULL, NULL, 0, \
   (RLMeshStateContainer)1, 0,    {0}, {0, 0}}
#else
struct RLMeshConcat {
  size_t struct_size;
  const RLMeshConcatPart* parts;
  size_t parts_len;
  bool has_pad_to;
  uint8_t _reserved0[3];
  uint32_t pad_to;
  RLMeshDType dtype;
  const char* dtype_other;
  const int64_t* reshape;
  size_t reshape_len;
  RLMeshStateContainer container;
  bool has_clip;
  uint8_t _reserved1[7];
  double clip[2];
};
#define RLMESH_CONCAT_INIT                                                            \
  {sizeof(RLMeshConcat),    NULL, 0,   0,     {0}, 0, (RLMeshDType)11, NULL, NULL, 0, \
   (RLMeshStateContainer)1, 0,    {0}, {0, 0}}
#endif
#define RLMESH_CONCAT_MIN_SIZE sizeof(RLMeshConcat)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshConstant {
  size_t struct_size;
  uint32_t dim;
  uint8_t _reserved0[4];
  double fill;
};
#define RLMESH_CONSTANT_INIT {sizeof(RLMeshConstant), 1, {0}, 0}
#else
struct RLMeshConstant {
  size_t struct_size;
  uint32_t dim;
  double fill;
};
#define RLMESH_CONSTANT_INIT {sizeof(RLMeshConstant), 1, 0}
#endif
#define RLMESH_CONSTANT_MIN_SIZE sizeof(RLMeshConstant)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshAdaptersText {
  size_t struct_size;
  const char* role;
  RLMeshTextContainer container;
  bool has_fill;
  uint8_t _reserved0[3];
  RLMeshStr fill;
};
#define RLMESH_ADAPTERS_TEXT_INIT \
  {sizeof(RLMeshAdaptersText), NULL, (RLMeshTextContainer)1, 0, {0}, {NULL, 0}}
#else
struct RLMeshAdaptersText {
  size_t struct_size;
  const char* role;
  RLMeshTextContainer container;
  bool has_fill;
  uint8_t _reserved0[3];
  RLMeshStr fill;
};
#define RLMESH_ADAPTERS_TEXT_INIT \
  {sizeof(RLMeshAdaptersText), NULL, (RLMeshTextContainer)1, 0, {0}, {NULL, 0}}
#endif
#define RLMESH_ADAPTERS_TEXT_MIN_SIZE sizeof(RLMeshAdaptersText)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshCustom {
  size_t struct_size;
  const RLMeshObsTransform* transform;
  const char* entrypoint;
};
#define RLMESH_CUSTOM_INIT {sizeof(RLMeshCustom), NULL, NULL}
#else
struct RLMeshCustom {
  size_t struct_size;
  const RLMeshObsTransform* transform;
  const char* entrypoint;
};
#define RLMESH_CUSTOM_INIT {sizeof(RLMeshCustom), NULL, NULL}
#endif
#define RLMESH_CUSTOM_MIN_SIZE sizeof(RLMeshCustom)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshActuator {
  size_t struct_size;
  const char* role;
  uint32_t dim;
  bool has_encoding;
  uint8_t _reserved0[3];
  RLMeshRotationEncoding encoding;
  uint8_t _reserved1[4];
  const RLMeshCustomEncoding* custom_encoding;
  bool has_range;
  uint8_t _reserved2[7];
  double range[2];
  bool binary;
  bool has_scale;
  uint8_t _reserved3[6];
  double scale;
  const double* axis_scale;
  size_t axis_scale_len;
  bool invert;
  bool has_threshold;
  uint8_t _reserved4[6];
  double threshold;
  bool clip;
  uint8_t _reserved5[7];
  double fill;
  bool optional;
  bool has_frame;
  uint8_t _reserved6[2];
  RLMeshFrame frame;
  const char* frame_other;
  bool has_reference;
  uint8_t _reserved7[3];
  RLMeshReference reference;
  const char* reference_other;
  const char* part;
  bool has_offset;
  uint8_t _reserved8[7];
  double offset;
  const double* axis_offset;
  size_t axis_offset_len;
  const char* const* labels;
  size_t labels_len;
};
#define RLMESH_ACTUATOR_INIT  \
  {sizeof(RLMeshActuator),    \
   NULL,                      \
   0,                         \
   0,                         \
   {0},                       \
   (RLMeshRotationEncoding)0, \
   {0},                       \
   NULL,                      \
   0,                         \
   {0},                       \
   {0, 0},                    \
   0,                         \
   0,                         \
   {0},                       \
   0,                         \
   NULL,                      \
   0,                         \
   0,                         \
   0,                         \
   {0},                       \
   0,                         \
   0,                         \
   {0},                       \
   0,                         \
   0,                         \
   0,                         \
   {0},                       \
   (RLMeshFrame)0,            \
   NULL,                      \
   0,                         \
   {0},                       \
   (RLMeshReference)0,        \
   NULL,                      \
   NULL,                      \
   0,                         \
   {0},                       \
   0,                         \
   NULL,                      \
   0,                         \
   NULL,                      \
   0}
#else
struct RLMeshActuator {
  size_t struct_size;
  const char* role;
  uint32_t dim;
  bool has_encoding;
  uint8_t _reserved0[3];
  RLMeshRotationEncoding encoding;
  const RLMeshCustomEncoding* custom_encoding;
  bool has_range;
  uint8_t _reserved1[7];
  double range[2];
  bool binary;
  bool has_scale;
  uint8_t _reserved2[6];
  double scale;
  const double* axis_scale;
  size_t axis_scale_len;
  bool invert;
  bool has_threshold;
  uint8_t _reserved3[6];
  double threshold;
  bool clip;
  uint8_t _reserved4[7];
  double fill;
  bool optional;
  bool has_frame;
  uint8_t _reserved5[2];
  RLMeshFrame frame;
  const char* frame_other;
  bool has_reference;
  uint8_t _reserved6[3];
  RLMeshReference reference;
  const char* reference_other;
  const char* part;
  bool has_offset;
  uint8_t _reserved7[3];
  double offset;
  const double* axis_offset;
  size_t axis_offset_len;
  const char* const* labels;
  size_t labels_len;
};
#define RLMESH_ACTUATOR_INIT  \
  {sizeof(RLMeshActuator),    \
   NULL,                      \
   0,                         \
   0,                         \
   {0},                       \
   (RLMeshRotationEncoding)0, \
   NULL,                      \
   0,                         \
   {0},                       \
   {0, 0},                    \
   0,                         \
   0,                         \
   {0},                       \
   0,                         \
   NULL,                      \
   0,                         \
   0,                         \
   0,                         \
   {0},                       \
   0,                         \
   0,                         \
   {0},                       \
   0,                         \
   0,                         \
   0,                         \
   {0},                       \
   (RLMeshFrame)0,            \
   NULL,                      \
   0,                         \
   {0},                       \
   (RLMeshReference)0,        \
   NULL,                      \
   NULL,                      \
   0,                         \
   {0},                       \
   0,                         \
   NULL,                      \
   0,                         \
   NULL,                      \
   0}
#endif
#define RLMESH_ACTUATOR_MIN_SIZE sizeof(RLMeshActuator)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshAction {
  size_t struct_size;
  const RLMeshActuator* const* components;
  size_t components_len;
  bool has_clip;
  uint8_t _reserved0[7];
  double clip[2];
};
#define RLMESH_ACTION_INIT {sizeof(RLMeshAction), NULL, 0, 0, {0}, {0, 0}}
#else
struct RLMeshAction {
  size_t struct_size;
  const RLMeshActuator* const* components;
  size_t components_len;
  bool has_clip;
  uint8_t _reserved0[3];
  double clip[2];
};
#define RLMESH_ACTION_INIT {sizeof(RLMeshAction), NULL, 0, 0, {0}, {0, 0}}
#endif
#define RLMESH_ACTION_MIN_SIZE sizeof(RLMeshAction)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshImageTag {
  size_t struct_size;
  const char* role;
  RLMeshImageLayout layout;
  bool upside_down;
  uint8_t _reserved0[3];
  const char* part;
};
#define RLMESH_IMAGE_TAG_INIT {sizeof(RLMeshImageTag), NULL, (RLMeshImageLayout)1, 0, {0}, NULL}
#else
struct RLMeshImageTag {
  size_t struct_size;
  const char* role;
  RLMeshImageLayout layout;
  bool upside_down;
  uint8_t _reserved0[3];
  const char* part;
};
#define RLMESH_IMAGE_TAG_INIT {sizeof(RLMeshImageTag), NULL, (RLMeshImageLayout)1, 0, {0}, NULL}
#endif
#define RLMESH_IMAGE_TAG_MIN_SIZE sizeof(RLMeshImageTag)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshStateTag {
  size_t struct_size;
  const char* role;
  const RLMeshRotationEncoding* encoding;
  size_t encoding_len;
  const char* const* encoding_other;
  bool has_range;
  uint8_t _reserved0[7];
  double range[2];
  bool has_frame;
  uint8_t _reserved1[3];
  RLMeshFrame frame;
  const char* frame_other;
  bool has_provenance;
  uint8_t _reserved2[3];
  RLMeshProvenance provenance;
  const char* provenance_other;
  const char* part;
  const char* const* labels;
  size_t labels_len;
};
#define RLMESH_STATE_TAG_INIT \
  {sizeof(RLMeshStateTag),    \
   NULL,                      \
   NULL,                      \
   0,                         \
   NULL,                      \
   0,                         \
   {0},                       \
   {0, 0},                    \
   0,                         \
   {0},                       \
   (RLMeshFrame)0,            \
   NULL,                      \
   0,                         \
   {0},                       \
   (RLMeshProvenance)0,       \
   NULL,                      \
   NULL,                      \
   NULL,                      \
   0}
#else
struct RLMeshStateTag {
  size_t struct_size;
  const char* role;
  const RLMeshRotationEncoding* encoding;
  size_t encoding_len;
  const char* const* encoding_other;
  bool has_range;
  uint8_t _reserved0[3];
  double range[2];
  bool has_frame;
  uint8_t _reserved1[3];
  RLMeshFrame frame;
  const char* frame_other;
  bool has_provenance;
  uint8_t _reserved2[3];
  RLMeshProvenance provenance;
  const char* provenance_other;
  const char* part;
  const char* const* labels;
  size_t labels_len;
  uint8_t _reserved3[4];
};
#define RLMESH_STATE_TAG_INIT \
  {sizeof(RLMeshStateTag),    \
   NULL,                      \
   NULL,                      \
   0,                         \
   NULL,                      \
   0,                         \
   {0},                       \
   {0, 0},                    \
   0,                         \
   {0},                       \
   (RLMeshFrame)0,            \
   NULL,                      \
   0,                         \
   {0},                       \
   (RLMeshProvenance)0,       \
   NULL,                      \
   NULL,                      \
   NULL,                      \
   0,                         \
   {0}}
#endif
#define RLMESH_STATE_TAG_MIN_SIZE sizeof(RLMeshStateTag)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshTextTag {
  size_t struct_size;
  const char* role;
};
#define RLMESH_TEXT_TAG_INIT {sizeof(RLMeshTextTag), NULL}
#else
struct RLMeshTextTag {
  size_t struct_size;
  const char* role;
};
#define RLMESH_TEXT_TAG_INIT {sizeof(RLMeshTextTag), NULL}
#endif
#define RLMESH_TEXT_TAG_MIN_SIZE sizeof(RLMeshTextTag)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshField {
  size_t struct_size;
  const char* role;
  uint32_t dim;
  uint8_t _reserved0[4];
  const RLMeshRotationEncoding* encoding;
  size_t encoding_len;
  const char* const* encoding_other;
  bool has_range;
  uint8_t _reserved1[7];
  double range[2];
  bool has_frame;
  uint8_t _reserved2[3];
  RLMeshFrame frame;
  const char* frame_other;
  bool has_provenance;
  uint8_t _reserved3[3];
  RLMeshProvenance provenance;
  const char* provenance_other;
  const char* part;
  const char* const* labels;
  size_t labels_len;
};
#define RLMESH_FIELD_INIT \
  {sizeof(RLMeshField),   \
   NULL,                  \
   0,                     \
   {0},                   \
   NULL,                  \
   0,                     \
   NULL,                  \
   0,                     \
   {0},                   \
   {0, 0},                \
   0,                     \
   {0},                   \
   (RLMeshFrame)0,        \
   NULL,                  \
   0,                     \
   {0},                   \
   (RLMeshProvenance)0,   \
   NULL,                  \
   NULL,                  \
   NULL,                  \
   0}
#else
struct RLMeshField {
  size_t struct_size;
  const char* role;
  uint32_t dim;
  const RLMeshRotationEncoding* encoding;
  size_t encoding_len;
  const char* const* encoding_other;
  bool has_range;
  uint8_t _reserved0[7];
  double range[2];
  bool has_frame;
  uint8_t _reserved1[3];
  RLMeshFrame frame;
  const char* frame_other;
  bool has_provenance;
  uint8_t _reserved2[3];
  RLMeshProvenance provenance;
  const char* provenance_other;
  const char* part;
  const char* const* labels;
  size_t labels_len;
  uint8_t _reserved3[4];
};
#define RLMESH_FIELD_INIT \
  {sizeof(RLMeshField),   \
   NULL,                  \
   0,                     \
   NULL,                  \
   0,                     \
   NULL,                  \
   0,                     \
   {0},                   \
   {0, 0},                \
   0,                     \
   {0},                   \
   (RLMeshFrame)0,        \
   NULL,                  \
   0,                     \
   {0},                   \
   (RLMeshProvenance)0,   \
   NULL,                  \
   NULL,                  \
   NULL,                  \
   0,                     \
   {0}}
#endif
#define RLMESH_FIELD_MIN_SIZE sizeof(RLMeshField)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshSplit {
  size_t struct_size;
  const RLMeshField* const* fields;
  size_t fields_len;
};
#define RLMESH_SPLIT_INIT {sizeof(RLMeshSplit), NULL, 0}
#else
struct RLMeshSplit {
  size_t struct_size;
  const RLMeshField* const* fields;
  size_t fields_len;
};
#define RLMESH_SPLIT_INIT {sizeof(RLMeshSplit), NULL, 0}
#endif
#define RLMESH_SPLIT_MIN_SIZE sizeof(RLMeshSplit)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshPreviousOptions {
  size_t struct_size;
  const char* part;
  double fill;
};
#define RLMESH_PREVIOUS_OPTIONS_INIT {sizeof(RLMeshPreviousOptions), NULL, 0}
#else
struct RLMeshPreviousOptions {
  size_t struct_size;
  const char* part;
  double fill;
};
#define RLMESH_PREVIOUS_OPTIONS_INIT {sizeof(RLMeshPreviousOptions), NULL, 0}
#endif
#define RLMESH_PREVIOUS_OPTIONS_MIN_SIZE sizeof(RLMeshPreviousOptions)
#if UINTPTR_MAX > UINT32_MAX
struct RLMeshObservationRoles {
  size_t struct_size;
  const char* const* images;
  size_t images_len;
  const char* const* states;
  size_t states_len;
  const char* const* texts;
  size_t texts_len;
};
#define RLMESH_OBSERVATION_ROLES_INIT {sizeof(RLMeshObservationRoles), NULL, 0, NULL, 0, NULL, 0}
#else
struct RLMeshObservationRoles {
  size_t struct_size;
  const char* const* images;
  size_t images_len;
  const char* const* states;
  size_t states_len;
  const char* const* texts;
  size_t texts_len;
};
#define RLMESH_OBSERVATION_ROLES_INIT {sizeof(RLMeshObservationRoles), NULL, 0, NULL, 0, NULL, 0}
#endif
#define RLMESH_OBSERVATION_ROLES_MIN_SIZE sizeof(RLMeshObservationRoles)
struct RLMeshAdvisory {
  size_t struct_size;
  RLMeshAdvisorySeverity severity;
  uint32_t _reserved;
  const char* message;
};
#define RLMESH_ADVISORY_INIT {sizeof(RLMeshAdvisory), (RLMeshAdvisorySeverity)0, 0, NULL}
#define RLMESH_ADVISORY_MIN_SIZE sizeof(RLMeshAdvisory)
struct RLMeshTagOptions {
  size_t struct_size;
  bool validate;
  uint8_t _reserved0;
  uint16_t _reserved1;
  uint32_t _reserved2;
};
#define RLMESH_TAG_OPTIONS_INIT {sizeof(RLMeshTagOptions), true, 0, 0, 0}
#define RLMESH_TAG_OPTIONS_MIN_SIZE sizeof(RLMeshTagOptions)
struct RLMeshResolveOptions {
  size_t struct_size;
  bool trust_entrypoints;
  bool check_inverse;
  uint16_t _reserved0;
  uint32_t _reserved1;
};
#define RLMESH_RESOLVE_OPTIONS_INIT {sizeof(RLMeshResolveOptions), 0, true, 0, 0}
#define RLMESH_RESOLVE_OPTIONS_MIN_SIZE sizeof(RLMeshResolveOptions)

#define RLMESH_ADAPTERS_IMAGE_PRIMARY "image/primary"
#define RLMESH_ADAPTERS_IMAGE_SECONDARY "image/secondary"
#define RLMESH_ADAPTERS_INSTRUCTION "text/instruction"
#define RLMESH_ADAPTERS_JOINT_POS "proprio/joint_pos"
#define RLMESH_ADAPTERS_JOINT_VEL "proprio/joint_vel"
#define RLMESH_ADAPTERS_ACTION_JOINT_POS "action/joint_pos"
#define RLMESH_ADAPTERS_ACTION_JOINT_VEL "action/joint_vel"
#define RLMESH_ADAPTERS_IMAGE_WRIST "image/wrist"
#define RLMESH_ADAPTERS_EEF_POS "proprio/eef_pos"
#define RLMESH_ADAPTERS_EEF_ROT "proprio/eef_rot"
#define RLMESH_ADAPTERS_GRIPPER_POS "proprio/gripper"
#define RLMESH_ADAPTERS_EEF_WRENCH "proprio/eef_wrench"
#define RLMESH_ADAPTERS_ACTION_DELTA_POS "action/delta_eef_pos"
#define RLMESH_ADAPTERS_ACTION_DELTA_ROT "action/delta_eef_rot"
#define RLMESH_ADAPTERS_ACTION_GRIPPER "action/gripper"
#define RLMESH_ADAPTERS_ACTION_EEF_POS "action/eef_pos"
#define RLMESH_ADAPTERS_ACTION_EEF_ROT "action/eef_rot"
#define RLMESH_ADAPTERS_BASE_ANG_VEL "proprio/base_ang_vel"
#define RLMESH_ADAPTERS_BASE_ROT "proprio/base_rot"
#define RLMESH_ADAPTERS_COMMAND_BASE_VEL "command/base_vel"
#define RLMESH_ADAPTERS_LEFT_ARM "left_arm"
#define RLMESH_ADAPTERS_RIGHT_ARM "right_arm"
#define RLMESH_ADAPTERS_HEAD "head"
#define RLMESH_ADAPTERS_TORSO "torso"
#define RLMESH_ADAPTERS_BASE "base"
#define RLMESH_ADAPTERS_LEFT_LEG "left_leg"
#define RLMESH_ADAPTERS_RIGHT_LEG "right_leg"
#define RLMESH_ADAPTERS_ENV_METADATA_KEY "rlmesh.adapters.v1.env_tags"
#define RLMESH_ADAPTERS_MODEL_METADATA_KEY "rlmesh.adapters.v1.model_spec"
#define RLMESH_ADAPTERS_ENV_BRANCH_METADATA_KEY "rlmesh.adapters.v1.env_branch"
RLMESH_API RLMeshModelSpec* rlmesh_model_spec_clone(const RLMeshModelSpec* value);
RLMESH_API void rlmesh_model_spec_free(RLMeshModelSpec* value);
RLMESH_API bool rlmesh_model_spec_equals(const RLMeshModelSpec* a, const RLMeshModelSpec* b);
RLMESH_API RLMeshStatus rlmesh_model_spec_repr(const RLMeshModelSpec* value, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_model_spec_to_json(const RLMeshModelSpec* value, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_model_spec_to_dict(const RLMeshModelSpec* value, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_model_spec_to_metadata(const RLMeshModelSpec* value,
                                                      RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_model_spec_from_dict(const RLMeshValue* data, RLMeshModelSpec** out);
RLMESH_API RLMeshStatus rlmesh_model_spec_from_metadata(const RLMeshValue* metadata,
                                                        RLMeshModelSpec** out);
RLMESH_API RLMeshStatus rlmesh_model_spec_from_json(const char* payload, RLMeshModelSpec** out);
RLMESH_API RLMeshEnvTags* rlmesh_env_tags_clone(const RLMeshEnvTags* value);
RLMESH_API void rlmesh_env_tags_free(RLMeshEnvTags* value);
RLMESH_API bool rlmesh_env_tags_equals(const RLMeshEnvTags* a, const RLMeshEnvTags* b);
RLMESH_API RLMeshStatus rlmesh_env_tags_repr(const RLMeshEnvTags* value, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_env_tags_to_json(const RLMeshEnvTags* value, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_env_tags_to_dict(const RLMeshEnvTags* value, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_env_tags_to_metadata(const RLMeshEnvTags* value, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_env_tags_from_dict(const RLMeshValue* data, RLMeshEnvTags** out);
RLMESH_API RLMeshStatus rlmesh_env_tags_from_metadata(const RLMeshValue* metadata,
                                                      RLMeshEnvTags** out);
RLMESH_API RLMeshStatus rlmesh_env_tags_from_json(const char* payload, RLMeshEnvTags** out);
/* All callables receive borrowed inputs and transfer their owned output Value,
 * even on failure. State transfers only on successful construction. Transforms
 * may be called concurrently; their clones share identity and user_data. */
typedef RLMeshStatus (*RLMeshObsTransformFn)(void* user_data, const RLMeshValue* value,
                                             RLMeshValue** out);
typedef RLMeshStatus (*RLMeshRotationTransformFn)(void* user_data, const RLMeshValue* value,
                                                  RLMeshValue** out);
typedef RLMeshStatus (*RLMeshPredictFn)(void* user_data, const RLMeshValue* value,
                                        RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_obs_transform_new(RLMeshObsTransformFn fn, void* user_data,
                                                 void (*free_user_data)(void* user_data),
                                                 RLMeshObsTransform** out);
RLMESH_API RLMeshObsTransform* rlmesh_obs_transform_clone(const RLMeshObsTransform* transform);
RLMESH_API void rlmesh_obs_transform_free(RLMeshObsTransform* transform);
/* Callable identity survives cloning; hashes are process-local. NULL equals NULL and hashes to 0.
 */
RLMESH_API bool rlmesh_obs_transform_equals(const RLMeshObsTransform* obs_transform,
                                            const RLMeshObsTransform* obs_transform_1);
RLMESH_API uint64_t rlmesh_obs_transform_hash(const RLMeshObsTransform* obs_transform);
RLMESH_API RLMeshStatus rlmesh_rotation_transform_new(RLMeshRotationTransformFn fn, void* user_data,
                                                      void (*free_user_data)(void* user_data),
                                                      RLMeshRotationTransform** out);
RLMESH_API RLMeshRotationTransform* rlmesh_rotation_transform_clone(
    const RLMeshRotationTransform* transform);
RLMESH_API void rlmesh_rotation_transform_free(RLMeshRotationTransform* transform);
/* Callable identity survives cloning; hashes are process-local. NULL equals NULL and hashes to 0.
 */
RLMESH_API bool rlmesh_rotation_transform_equals(
    const RLMeshRotationTransform* rotation_transform,
    const RLMeshRotationTransform* rotation_transform_1);
RLMESH_API uint64_t
rlmesh_rotation_transform_hash(const RLMeshRotationTransform* rotation_transform);
RLMESH_API RLMeshStatus rlmesh_model_spec_new(const RLMeshInputNode* input,
                                              const RLMeshAction* output, RLMeshModelSpec** out);
RLMESH_API const RLMeshInputNode* rlmesh_model_spec_input(const RLMeshModelSpec* spec);
RLMESH_API const RLMeshAction* rlmesh_model_spec_output(const RLMeshModelSpec* spec);
RLMESH_API RLMeshStatus rlmesh_env_tags_new(const RLMeshObsNode* observation,
                                            const RLMeshAction* action, RLMeshEnvTags** out);
RLMESH_API const RLMeshObsNode* rlmesh_env_tags_observation(const RLMeshEnvTags* tags);
RLMESH_API const RLMeshAction* rlmesh_env_tags_action(const RLMeshEnvTags* tags);
RLMESH_API RLMeshStatus rlmesh_env_tags_observation_roles(const RLMeshEnvTags* tags,
                                                          RLMeshObservationRoles* out);
RLMESH_API RLMeshStatus rlmesh_observation_roles_equals(const RLMeshObservationRoles* a,
                                                        const RLMeshObservationRoles* b, bool* out);
RLMESH_API RLMeshStatus rlmesh_observation_roles_repr(const RLMeshObservationRoles* value,
                                                      RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_advisory_equals(const RLMeshAdvisory* a, const RLMeshAdvisory* b,
                                               bool* out);
RLMESH_API RLMeshStatus rlmesh_advisory_repr(const RLMeshAdvisory* value, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_action_dim(const RLMeshAction* action, uint32_t* out);
RLMESH_API RLMeshStatus rlmesh_custom_encoding_is_entrypoint(const RLMeshCustomEncoding* encoding,
                                                             bool* out);
RLMESH_API RLMeshStatus rlmesh_custom_encoding_width(const RLMeshCustomEncoding* encoding,
                                                     uint32_t* out);
RLMESH_API RLMeshStatus rlmesh_rotation_encoding_dims(RLMeshRotationEncoding encoding,
                                                      uint32_t* out);
RLMESH_API size_t rlmesh_adapters_parts_len(void);
RLMESH_API const char* rlmesh_adapters_parts_get(size_t i);
RLMESH_API size_t rlmesh_adapters_rotation_dims_len(void);
RLMESH_API RLMeshStatus rlmesh_adapters_rotation_dims_get(size_t i,
                                                          RLMeshRotationEncoding* encoding,
                                                          uint32_t* dims);
RLMESH_API RLMeshStatus rlmesh_image_with_size(RLMeshImage* image, uint32_t size);
RLMESH_API RLMeshStatus rlmesh_rotation_from_matrix(const double rows[3][3], double value[6],
                                                    RLMeshRotation* out);
RLMESH_API RLMeshStatus rlmesh_adapters_previous(const char* role,
                                                 const RLMeshPreviousOptions* options,
                                                 RLMeshState* out);
RLMESH_API RLMeshInputNode rlmesh_input_node_image(const RLMeshImage* leaf);
RLMESH_API RLMeshInputNode rlmesh_input_node_state(const RLMeshState* leaf);
RLMESH_API RLMeshInputNode rlmesh_input_node_concat(const RLMeshConcat* leaf);
RLMESH_API RLMeshInputNode rlmesh_input_node_text(const RLMeshAdaptersText* leaf);
RLMESH_API RLMeshInputNode rlmesh_input_node_custom(const RLMeshCustom* leaf);
RLMESH_API RLMeshInputNode rlmesh_input_node_dict(const RLMeshStr* keys,
                                                  const RLMeshInputNode* children, size_t n);
RLMESH_API RLMeshInputNode rlmesh_input_node_tuple(const RLMeshInputNode* children, size_t n);
RLMESH_API RLMeshObsNode rlmesh_obs_node_image_tag(const RLMeshImageTag* leaf);
RLMESH_API RLMeshObsNode rlmesh_obs_node_state_tag(const RLMeshStateTag* leaf);
RLMESH_API RLMeshObsNode rlmesh_obs_node_text_tag(const RLMeshTextTag* leaf);
RLMESH_API RLMeshObsNode rlmesh_obs_node_split(const RLMeshSplit* leaf);
RLMESH_API RLMeshObsNode rlmesh_obs_node_dict(const RLMeshStr* keys, const RLMeshObsNode* children,
                                              size_t n);
RLMESH_API RLMeshObsNode rlmesh_obs_node_tuple(const RLMeshObsNode* children, size_t n);
/* AdapterBase: per-step calls and explain serialize across clones. */
typedef struct RLMeshAdapterVtable {
  size_t struct_size;
  RLMeshStatus (*transform_obs)(void* user_data, const RLMeshValue* raw_obs, RLMeshValue** out);
  RLMeshStatus (*transform_action)(void* user_data, const RLMeshValue* raw_action,
                                   RLMeshValue** out);
  RLMeshStatus (*reset)(void* user_data);
  RLMeshStatus (*observe)(void* user_data, const RLMeshValue* raw_obs);
  /* Borrowed UTF-8, copied before another slot runs; NULL slot uses the base text. */
  RLMeshStatus (*explain)(void* user_data, const char** out);
  void (*free_user_data)(void* user_data);
} RLMeshAdapterVtable;
#define RLMESH_ADAPTER_VTABLE_INIT {sizeof(RLMeshAdapterVtable), NULL, NULL, NULL, NULL, NULL, NULL}
#define RLMESH_ADAPTER_VTABLE_MIN_SIZE sizeof(RLMeshAdapterVtable)
RLMESH_API RLMeshStatus rlmesh_adapter_new(const RLMeshAdapterVtable* vtable, void* user_data,
                                           RLMeshAdapter** out);
RLMESH_API RLMeshAdapter* rlmesh_adapter_clone(const RLMeshAdapter* adapter);
RLMESH_API void rlmesh_adapter_free(RLMeshAdapter* adapter);
RLMESH_API RLMeshStatus rlmesh_adapters_resolve(const RLMeshEnvTags* env_tags,
                                                const RLMeshSpaceSpec* observation_space,
                                                const RLMeshSpaceSpec* action_space,
                                                const RLMeshModelSpec* model_spec,
                                                const RLMeshResolveOptions* options,
                                                RLMeshAdapter** out);
RLMESH_API RLMeshStatus rlmesh_adapters_resolve_from_contract(const RLMeshEnvContract* contract,
                                                              const RLMeshModelSpec* model_spec,
                                                              const RLMeshResolveOptions* options,
                                                              RLMeshAdapter** out);
RLMESH_API RLMeshStatus rlmesh_adapter_transform_obs(RLMeshAdapter* adapter,
                                                     const RLMeshValue* raw_obs, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_adapter_transform_action(RLMeshAdapter* adapter,
                                                        const RLMeshValue* raw_action,
                                                        RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_adapter_observe(RLMeshAdapter* adapter, const RLMeshValue* raw_obs);
RLMESH_API RLMeshStatus rlmesh_adapter_reset(RLMeshAdapter* adapter);
RLMESH_API RLMeshStatus rlmesh_adapter_explain(const RLMeshAdapter* adapter, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_adapter_advisories_len(const RLMeshAdapter* adapter, size_t* out);
RLMESH_API RLMeshStatus rlmesh_adapter_advisories_get(const RLMeshAdapter* adapter, size_t i,
                                                      RLMeshAdvisory* out);
RLMESH_API RLMeshStatus rlmesh_adapter_history_keys_len(const RLMeshAdapter* adapter, size_t* out);
RLMESH_API RLMeshStatus rlmesh_adapter_history_keys_get(const RLMeshAdapter* adapter, size_t i,
                                                        RLMeshStr* out);
RLMESH_API RLMeshStatus rlmesh_adapter_history_windows_len(const RLMeshAdapter* adapter,
                                                           size_t* out);
RLMESH_API RLMeshStatus rlmesh_adapter_history_windows_get(const RLMeshAdapter* adapter, size_t i,
                                                           RLMeshStr* placement, uint64_t* span,
                                                           uint64_t* frame_bytes);
RLMESH_API RLMeshStatus rlmesh_adapter_wrap_predict(RLMeshAdapter* adapter,
                                                    RLMeshPredictFn predict_fn, void* user_data,
                                                    void (*free_user_data)(void* user_data),
                                                    RLMeshWrappedPredict** out);
RLMESH_API RLMeshStatus rlmesh_wrapped_predict_call(RLMeshWrappedPredict* predict,
                                                    const RLMeshValue* raw_obs,
                                                    RLMeshValue** out_action);
RLMESH_API void rlmesh_wrapped_predict_free(RLMeshWrappedPredict* predict);
RLMESH_API RLMeshStatus rlmesh_adapters_tag(RLMeshEnv* env, const RLMeshEnvTags* tags,
                                            const RLMeshTagOptions* options);
RLMESH_API RLMeshStatus rlmesh_adapters_tag_vector(RLMeshVectorEnv* env, const RLMeshEnvTags* tags,
                                                   const RLMeshTagOptions* options);
RLMESH_API RLMeshStatus rlmesh_ext_register_entrypoint(const char* entrypoint,
                                                       const RLMeshObsTransform* transform);

/* Runtime of the translation unit including this header, not of the Rust
 * library behind the ABI. Used by the ABI 5 server options and rlmesh.hpp. */
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

/* ---- Env authoring and serving (facade) ---------------------------------- */

typedef struct RLMeshEnv RLMeshEnv;
typedef struct RLMeshVectorEnv RLMeshVectorEnv;
typedef struct RLMeshEnvServer RLMeshEnvServer;
typedef struct RLMeshEnvTags RLMeshEnvTags;
/* Returned tags are owned. */
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
  RLMeshStatus (*reset)(void* user_data, const int64_t* seed, const RLMeshValue* options,
                        RLMeshValue** out_observation, RLMeshValue** out_info);
  RLMeshStatus (*step)(void* user_data, const RLMeshValue* action, RLMeshValue** out_observation,
                       double* out_reward, bool* out_terminated, bool* out_truncated,
                       RLMeshValue** out_info);
  RLMeshStatus (*render)(void* user_data, RLMeshValue** out_frame);
  RLMeshStatus (*close)(void* user_data);
  void (*free_user_data)(void* user_data);
} RLMeshEnvVtable;
#define RLMESH_ENV_VTABLE_INIT {sizeof(RLMeshEnvVtable), NULL, NULL, NULL, NULL, NULL}
#define RLMESH_ENV_VTABLE_MIN_SIZE sizeof(RLMeshEnvVtable)

typedef struct RLMeshVectorEnvVtable {
  size_t struct_size;
  RLMeshStatus (*reset)(void* user_data, const int64_t* seed, size_t seed_len,
                        const RLMeshValue* options, RLMeshValue** out_observation,
                        RLMeshValue** out_info);
  RLMeshStatus (*step)(void* user_data, const RLMeshValue* actions, RLMeshValue** out_observation,
                       double* out_rewards, bool* out_terminated, bool* out_truncated,
                       RLMeshValue** out_info);
  RLMeshStatus (*render)(void* user_data, RLMeshValue** out_frame);
  RLMeshStatus (*close)(void* user_data);
  void (*free_user_data)(void* user_data);
} RLMeshVectorEnvVtable;
#define RLMESH_VECTOR_ENV_VTABLE_INIT {sizeof(RLMeshVectorEnvVtable), NULL, NULL, NULL, NULL, NULL}
#define RLMESH_VECTOR_ENV_VTABLE_MIN_SIZE sizeof(RLMeshVectorEnvVtable)

/* Env handles copy vtables/attributes. State transfers only on success and
 * free_user_data runs once after the last library reference. Free never closes;
 * close runs at most once. A handle may be borrowed by one server/run/session
 * at a time. Free a borrowing server before freeing the env. Close on a busy
 * env fails with RLMESH_ERR_RUNTIME. Callback outputs are taken even on failure;
 * NULL info means {}. Callbacks and state must be safe to move between threads. */
RLMESH_API RLMeshStatus rlmesh_env_new(const RLMeshEnvVtable* env_vtable, void* user_data,
                                       const RLMeshEnvAttributes* env_attributes, RLMeshEnv** out);
RLMESH_API RLMeshStatus rlmesh_vector_env_new(
    const RLMeshVectorEnvVtable* vector_env_vtable, void* user_data,
    const RLMeshVectorEnvAttributes* vector_env_attributes, RLMeshVectorEnv** out);
RLMESH_API RLMeshStatus rlmesh_env_close(RLMeshEnv* env);
RLMESH_API RLMeshStatus rlmesh_vector_env_close(RLMeshVectorEnv* vector_env);
RLMESH_API void rlmesh_env_free(RLMeshEnv* env);
RLMESH_API void rlmesh_vector_env_free(RLMeshVectorEnv* vector_env);

/* New binds. start runs callbacks on library threads; serve keeps lane 0 on
 * the caller. shutdown is allowed from a callback. Free shuts down and drains
 * all C callbacks, including any still running past the facade's timeout. */
RLMESH_API RLMeshStatus rlmesh_env_server_new(RLMeshEnv* const* env, size_t envs_len,
                                              const RLMeshEnvServerOptions* options,
                                              RLMeshEnvServer** out);
RLMESH_API RLMeshStatus rlmesh_env_server_new_vector(RLMeshVectorEnv* vector_env,
                                                     const RLMeshEnvServerOptions* options,
                                                     RLMeshEnvServer** out);
RLMESH_API RLMeshStatus rlmesh_env_server_address(const RLMeshEnvServer* env_server,
                                                  RLMeshBytes* out);
RLMESH_API const RLMeshEnvContract* rlmesh_env_server_env_contract(
    const RLMeshEnvServer* env_server);
RLMESH_API RLMeshStatus rlmesh_env_server_start(RLMeshEnvServer* env_server);
RLMESH_API RLMeshStatus rlmesh_env_server_serve(RLMeshEnvServer* env_server);
/* timeout is seconds (NULL = no deadline). A stopped server sets out_stopped
 * true even when returning its background serve error, which is returned once.
 * Timeout expiry sets false; misuse and invalid inputs leave it untouched. */
RLMESH_API RLMeshStatus rlmesh_env_server_wait(RLMeshEnvServer* env_server, const double* timeout,
                                               bool* out_stopped);
RLMESH_API RLMeshStatus rlmesh_env_server_shutdown(RLMeshEnvServer* env_server);
RLMESH_API RLMeshStatus rlmesh_env_server_repr(const RLMeshEnvServer* env_server, RLMeshBytes* out);
RLMESH_API void rlmesh_env_server_free(RLMeshEnvServer* env_server);

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
RLMESH_API RLMeshStatus rlmesh_ext_cancel_token_new(RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_ext_cancel_token_clone(const RLMeshExtCancelToken* ext_cancel_token,
                                                      RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_ext_cancel_token_cancel(RLMeshExtCancelToken* ext_cancel_token);
RLMESH_API void rlmesh_ext_cancel_token_free(RLMeshExtCancelToken* ext_cancel_token);

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
RLMESH_API RLMeshStatus rlmesh_env_factory_tags(const RLMeshEnvFactory* env_factory,
                                                RLMeshEnvTags** out /* NULL = None */);
RLMESH_API RLMeshStatus rlmesh_env_factory_reset_options(const RLMeshEnvFactory* env_factory,
                                                         RLMeshValue** out /* LIST of TEXT */);
RLMESH_API RLMeshStatus rlmesh_env_factory_tag_params(const RLMeshEnvFactory* env_factory,
                                                      RLMeshTagParamList** out);
RLMESH_API RLMeshStatus rlmesh_env_factory_workflow_edition(const RLMeshEnvFactory* env_factory,
                                                            bool* out_has, RLMeshBytes* out);

/* Lists (R-20): items are borrowed until the list is freed. */
RLMESH_API size_t rlmesh_variant_list_len(const RLMeshVariantList* variant_list);
RLMESH_API const RLMeshVariant* rlmesh_variant_list_get(const RLMeshVariantList* variant_list,
                                                        size_t index); /* NULL past the end */
RLMESH_API void rlmesh_variant_list_free(RLMeshVariantList* variant_list);
RLMESH_API size_t rlmesh_tag_param_list_len(const RLMeshTagParamList* tag_param_list);
/* A lent struct (R-19): struct_size is the library's own sizeof; read a field only if
 * offsetof(field) + sizeof(field) <= p->struct_size, else treat it as its _INIT value.
 * NULL past the end. */
RLMESH_API const RLMeshTagParam* rlmesh_tag_param_list_get(const RLMeshTagParamList* tag_param_list,
                                                           size_t index);
RLMESH_API void rlmesh_tag_param_list_free(RLMeshTagParamList* tag_param_list);

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
RLMESH_API void rlmesh_variant_free(RLMeshVariant* variant);
RLMESH_API RLMeshStatus rlmesh_variant_id(const RLMeshVariant* variant, RLMeshBytes* out);
RLMESH_API const RLMeshValue* rlmesh_variant_params(
    const RLMeshVariant* variant); /* DICT; borrowed until the next setter or _free */
RLMESH_API const RLMeshValue* rlmesh_variant_metadata(
    const RLMeshVariant* variant); /* DICT; borrowed likewise */
RLMESH_API RLMeshStatus rlmesh_variant_repr(const RLMeshVariant* variant, RLMeshBytes* out);
RLMESH_API bool rlmesh_variant_equals(const RLMeshVariant* variant, const RLMeshVariant* variant_1);
/* Variant fields are mutable, as Python's slots are (py/_variants.py:39-49). */
RLMESH_API RLMeshStatus rlmesh_variant_set_id(RLMeshVariant* variant, const char* id);
RLMESH_API RLMeshStatus rlmesh_variant_set_params(RLMeshVariant* variant,
                                                  const RLMeshValue* params /* DICT, copied */);
RLMESH_API RLMeshStatus rlmesh_variant_set_metadata(RLMeshVariant* variant,
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
RLMESH_API bool rlmesh_episode_state_get(const RLMeshEpisodeState* episode_state, const char* key,
                                         void** out);
/* R-20: on RLMESH_OK value (and free_value, NULL = none) pass to the state, and replacing
 * a key frees the old value first; on nonzero free_value is not called, the caller still
 * owns value, and the state is unchanged. Re-setting the same pointer keeps the
 * value alive and updates its free function without calling the previous one. */
RLMESH_API RLMeshStatus rlmesh_episode_state_set(RLMeshEpisodeState* episode_state, const char* key,
                                                 void* value, void (*free_value)(void* user_data));
RLMESH_API bool rlmesh_episode_state_remove(RLMeshEpisodeState* episode_state, const char* key);
RLMESH_API size_t rlmesh_episode_state_len(const RLMeshEpisodeState* episode_state);

/* One row's episode identity. `episode_id` is runtime-minted and never repeats,
 * so a stateful model keys per-episode state by it.
 *
 * `predict_index` is 0 on the first predict under the episode, then +1 per
 * predict until on_episode_end drops it. `predict_seed` mixes the episode's reset
 * seed with that ordinal (rlmesh_predict_seed) so a stochastic policy can seed
 * each forward reproducibly; has_predict_seed is false when the episode was reset
 * without a seed. A model holds at most 65536 live episodes and never evicts one:
 * a predict that would start one more fails with RLMESH_ERR_RUNTIME before the
 * callback runs. */
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
  RLMeshStatus (*load)(void* user_data, const RLMeshValue* kwargs);
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

RLMESH_API RLMeshStatus rlmesh_model_new(const RLMeshModelVtable* source, void* user_data,
                                         const RLMeshModelOptions* options, RLMeshModel** out);
RLMESH_API RLMeshStatus rlmesh_model_from_config(const RLMeshModelVtable* model_vtable,
                                                 void* user_data, const RLMeshValue* config,
                                                 RLMeshModel** out);
RLMESH_API RLMeshStatus rlmesh_model_clone(const RLMeshModel* model, RLMeshModel** out);
RLMESH_API RLMeshStatus rlmesh_model_close(RLMeshModel* model);
RLMESH_API void rlmesh_model_free(RLMeshModel* model);
RLMESH_API RLMeshStatus rlmesh_model_spec(const RLMeshModel* model, RLMeshSpecArg* out);
RLMESH_API bool rlmesh_model_allow_fusion(const RLMeshModel* model);
RLMESH_API void rlmesh_model_native_chunk(const RLMeshModel* model, bool* out_has, uint32_t* out);
RLMESH_API RLMeshStatus rlmesh_model_workflow_edition(const RLMeshModel* model, bool* out_has,
                                                      RLMeshBytes* out);
RLMESH_API int64_t rlmesh_predict_seed(int64_t episode_seed, uint64_t predict_index);

#define RLMESH_MODEL_VTABLE_MIN_SIZE sizeof(RLMeshModelVtable)

#define RLMESH_MODEL_OPTIONS_MIN_SIZE sizeof(RLMeshModelOptions)

#define RLMESH_PREDICT_CONTEXT_MIN_SIZE sizeof(RLMeshPredictContext)

#define RLMESH_MODEL_SERVE_OPTIONS_MIN_SIZE sizeof(RLMeshModelServeOptions)

#define RLMESH_EXT_MODEL_SERVE_OPTIONS_MIN_SIZE sizeof(RLMeshExtModelServeOptions)

/* ---- Describe (facade) -------------------------------------------------- */
typedef enum RLMeshDescribeKind {
  RLMESH_DESCRIBE_KIND_INVALID = 0,
  RLMESH_DESCRIBE_KIND_ENV = 1,
  RLMESH_DESCRIBE_KIND_MODEL = 2
} RLMeshDescribeKind;
typedef struct RLMeshDescribeOptions {
  size_t struct_size;
  const char* generated_at;
  RLMeshDescribeKind kind;
  bool has_kind;
  uint8_t _reserved[3];
} RLMeshDescribeOptions;
#define RLMESH_DESCRIBE_OPTIONS_MIN_SIZE sizeof(RLMeshDescribeOptions)
#define RLMESH_DESCRIBE_OPTIONS_INIT \
  {sizeof(RLMeshDescribeOptions), NULL, RLMESH_DESCRIBE_KIND_INVALID, false, {0, 0, 0}}
typedef struct RLMeshDescribeJsonOptions {
  size_t struct_size;
  const char* generated_at;
  const RLMeshEnv* const* served_env;
  size_t served_env_len;
  const RLMeshVectorEnv* served_vector_env;
  const char* workflow_edition;
  RLMeshDescribeKind kind;
  bool has_kind;
  uint8_t _reserved[3];
} RLMeshDescribeJsonOptions;
#define RLMESH_DESCRIBE_JSON_OPTIONS_MIN_SIZE sizeof(RLMeshDescribeJsonOptions)
#define RLMESH_DESCRIBE_JSON_OPTIONS_INIT                             \
  {sizeof(RLMeshDescribeJsonOptions), NULL,  NULL,     0, NULL, NULL, \
   RLMESH_DESCRIBE_KIND_INVALID,      false, {0, 0, 0}}
/* NULL options means INIT. Type forms call no vtable slot. Instance forms
 * use the facade's static declarations and model spec/representative factory
 * builds. The Value result is an owned Dict, JSON is owned RLMeshBytes.
 * served_env NULL/0 = None; non-NULL/0 is an invalid empty list. At most one
 * served form may be set. It replaces the default build and is never closed.
 * workflow_edition NULL resolves/records refusal; "" explicitly declares none. */
RLMESH_API RLMeshStatus rlmesh_model_vtable_describe(const RLMeshModelVtable* obj,
                                                     const RLMeshDescribeOptions* options,
                                                     RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_model_vtable_describe_json(const RLMeshModelVtable* obj,
                                                          const RLMeshDescribeJsonOptions* options,
                                                          RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_model_describe(const RLMeshModel* obj,
                                              const RLMeshDescribeOptions* options,
                                              RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_model_describe_json(const RLMeshModel* obj,
                                                   const RLMeshDescribeJsonOptions* options,
                                                   RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_env_factory_describe(RLMeshEnvFactory* obj,
                                                    const RLMeshDescribeOptions* options,
                                                    RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_env_factory_describe_json(RLMeshEnvFactory* obj,
                                                         const RLMeshDescribeJsonOptions* options,
                                                         RLMeshBytes* out);
/* Fenced served-env shorthand (COR-16): reads the server's resolved snapshot,
 * including its host declaration, before/after start and shutdown. */
RLMESH_API RLMeshStatus rlmesh_ext_env_server_describe_json(const RLMeshEnvServer* env_server,
                                                            RLMeshBytes* out);

/* ---- Running: facade targets, observers and results ---------------------- */
typedef struct RLMeshRunResult RLMeshRunResult;
typedef struct RLMeshTelemetryRow RLMeshTelemetryRow;
typedef struct RLMeshRunHooks RLMeshRunHooks;
typedef struct RLMeshRunContext RLMeshRunContext;
/* Typed advisories and read/frame-source APIs are declared below. */
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
RLMESH_API RLMeshModelTarget rlmesh_model_target(RLMeshModel* model);
RLMESH_API RLMeshEnvTarget rlmesh_address_target(const char* value);
RLMESH_API RLMeshEnvTarget rlmesh_env_server_target(const RLMeshEnvServer* env_server);
RLMESH_API RLMeshEnvTarget rlmesh_env_target(const RLMeshEnv* env);
RLMESH_API RLMeshEnvTarget rlmesh_vector_env_target(const RLMeshVectorEnv* vector_env);
RLMESH_API RLMeshEnvTarget rlmesh_env_factory_target(const RLMeshEnvFactory* env_factory);

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
RLMESH_API RLMeshStatus rlmesh_episode_result_equals(const RLMeshEpisodeResult* episode_result,
                                                     const RLMeshEpisodeResult* episode_result_1,
                                                     bool* out);
RLMESH_API RLMeshStatus rlmesh_episode_result_repr(const RLMeshEpisodeResult* episode_result,
                                                   RLMeshBytes* out);
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
RLMESH_API RLMeshStatus rlmesh_run_result_new(const RLMeshRunResultOptions* options,
                                              RLMeshRunResult** out);
RLMESH_API RLMeshRunResult* rlmesh_run_result_clone(const RLMeshRunResult* run_result);
RLMESH_API void rlmesh_run_result_free(RLMeshRunResult* run_result);
RLMESH_API size_t rlmesh_run_result_episodes_len(const RLMeshRunResult* run_result);
RLMESH_API size_t rlmesh_run_result_num_episodes(const RLMeshRunResult* run_result);
RLMESH_API int64_t rlmesh_run_result_total_steps(const RLMeshRunResult* run_result);
RLMESH_API double rlmesh_run_result_mean_reward(const RLMeshRunResult* run_result);
RLMESH_API bool rlmesh_run_result_success_rate(const RLMeshRunResult* run_result, double* out);
RLMESH_API RLMeshStatus rlmesh_run_result_episodes_get(const RLMeshRunResult* run_result,
                                                       size_t value, RLMeshEpisodeResult* out);
RLMESH_API size_t rlmesh_run_result_telemetry_len(const RLMeshRunResult* run_result);
RLMESH_API RLMeshStatus rlmesh_run_result_telemetry_get(const RLMeshRunResult* run_result,
                                                        size_t value, RLMeshTelemetryRow** out);
RLMESH_API RLMeshStatus rlmesh_run_result_format_telemetry(const RLMeshRunResult* run_result,
                                                           RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_run_result_repr(const RLMeshRunResult* run_result, RLMeshBytes* out);
RLMESH_API bool rlmesh_run_result_equals(const RLMeshRunResult* run_result,
                                         const RLMeshRunResult* run_result_1);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_new(const char* op, const char* component,
                                                 const char* metric, const char* unit,
                                                 int64_t count, double avg, double p50, double p95,
                                                 double p99, RLMeshTelemetryRow** out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_op(const RLMeshTelemetryRow* telemetry_row,
                                                RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_component(const RLMeshTelemetryRow* telemetry_row,
                                                       RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_metric(const RLMeshTelemetryRow* telemetry_row,
                                                    RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_unit(const RLMeshTelemetryRow* telemetry_row,
                                                  RLMeshBytes* out);
RLMESH_API int64_t rlmesh_telemetry_row_count(const RLMeshTelemetryRow* telemetry_row);
RLMESH_API double rlmesh_telemetry_row_avg(const RLMeshTelemetryRow* telemetry_row);
RLMESH_API double rlmesh_telemetry_row_p50(const RLMeshTelemetryRow* telemetry_row);
RLMESH_API double rlmesh_telemetry_row_p95(const RLMeshTelemetryRow* telemetry_row);
RLMESH_API double rlmesh_telemetry_row_p99(const RLMeshTelemetryRow* telemetry_row);
RLMESH_API RLMeshStatus rlmesh_telemetry_row_repr(const RLMeshTelemetryRow* telemetry_row,
                                                  RLMeshBytes* out);
RLMESH_API bool rlmesh_telemetry_row_equals(const RLMeshTelemetryRow* telemetry_row,
                                            const RLMeshTelemetryRow* telemetry_row_1);
RLMESH_API void rlmesh_telemetry_row_free(RLMeshTelemetryRow* telemetry_row);

/* Contexts are shared and retained independently from hook invocation. Every
 * protocol slot is required; NULL slots fail with INVALID_ARGUMENT. */
typedef struct RLMeshReadItem RLMeshReadItem;
typedef struct RLMeshReader RLMeshReader;
typedef struct RLMeshFrameSources RLMeshFrameSources;
typedef struct RLMeshRunContextVtable {
  size_t struct_size;
  RLMeshStatus (*contract)(void* user_data, RLMeshEnvContract** out);
  RLMeshStatus (*num_envs)(void* user_data, uint64_t* out);
  RLMeshStatus (*read)(void* user_data, const RLMeshValue* observation, const RLMeshReadItem* item,
                       RLMeshValue** out);
  RLMeshStatus (*frame_sources)(void* user_data, RLMeshFrameSources** out);
  void (*free_user_data)(void* user_data);
} RLMeshRunContextVtable;
#define RLMESH_RUN_CONTEXT_VTABLE_MIN_SIZE sizeof(RLMeshRunContextVtable)
#define RLMESH_RUN_CONTEXT_VTABLE_INIT \
  {sizeof(RLMeshRunContextVtable), NULL, NULL, NULL, NULL, NULL}
RLMESH_API RLMeshStatus rlmesh_run_context_new(const RLMeshRunContextVtable* run_context_vtable,
                                               void* user_data, RLMeshRunContext** out);
RLMESH_API RLMeshRunContext* rlmesh_run_context_clone(const RLMeshRunContext* run_context);
RLMESH_API void rlmesh_run_context_free(RLMeshRunContext* run_context);
RLMESH_API RLMeshStatus rlmesh_run_context_contract(const RLMeshRunContext* run_context,
                                                    RLMeshEnvContract** out);
RLMESH_API RLMeshStatus rlmesh_run_context_num_envs(const RLMeshRunContext* run_context,
                                                    uint64_t* out);
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
RLMESH_API RLMeshStatus rlmesh_step_event_equals(const RLMeshStepEvent* step_event,
                                                 const RLMeshStepEvent* step_event_1, bool* out);
RLMESH_API RLMeshStatus rlmesh_step_event_repr(const RLMeshStepEvent* step_event, RLMeshBytes* out);
typedef struct RLMeshRunHooksVtable {
  size_t struct_size;
  RLMeshStatus (*on_run_start)(void* user_data, const RLMeshRunContext* context);
  RLMeshStatus (*on_episode_start)(void* user_data, int64_t episode, const int64_t* seed);
  RLMeshStatus (*on_step)(void* user_data, const RLMeshStepEvent* event);
  RLMeshStatus (*on_episode_end)(void* user_data, const RLMeshEpisodeResult* result);
  RLMeshStatus (*on_run_end)(void* user_data, const RLMeshRunResult* result);
  void (*free_user_data)(void* user_data);
} RLMeshRunHooksVtable;
#define RLMESH_RUN_HOOKS_VTABLE_MIN_SIZE sizeof(RLMeshRunHooksVtable)
#define RLMESH_RUN_HOOKS_VTABLE_INIT \
  {sizeof(RLMeshRunHooksVtable), NULL, NULL, NULL, NULL, NULL, NULL}
RLMESH_API RLMeshStatus rlmesh_run_hooks_new(const RLMeshRunHooksVtable* run_hooks_vtable,
                                             void* user_data, RLMeshRunHooks** out);
RLMESH_API void rlmesh_run_hooks_free(RLMeshRunHooks* run_hooks);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_run_start(RLMeshRunHooks* hooks,
                                                      const RLMeshRunContext* context);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_episode_start(RLMeshRunHooks* hooks, int64_t episode,
                                                          const int64_t* seed);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_step(RLMeshRunHooks* hooks,
                                                 const RLMeshStepEvent* event);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_episode_end(RLMeshRunHooks* hooks,
                                                        const RLMeshEpisodeResult* result);
RLMESH_API RLMeshStatus rlmesh_run_hooks_on_run_end(RLMeshRunHooks* hooks,
                                                    const RLMeshRunResult* result);

/* Plain View configuration. Strings are borrowed here, copied by a consuming
 * run/session call. Parse writes no string allocation (source is NULL).
 * Initialize both inputs and parse outputs with RLMESH_VIEW_INIT. Seconds holds
 * require finite nonnegative values; overflow saturates to Duration::MAX. */
typedef enum RLMeshViewBackend {
  RLMESH_VIEW_BACKEND_INVALID = 0,
  RLMESH_VIEW_BACKEND_TERMINAL = 1,
  RLMESH_VIEW_BACKEND_HTTP = 2,
  RLMESH_VIEW_BACKEND_BOTH = 3
} RLMeshViewBackend;
typedef enum RLMeshViewFormat {
  RLMESH_VIEW_FORMAT_INVALID = 0,
  RLMESH_VIEW_FORMAT_JPEG = 1,
  RLMESH_VIEW_FORMAT_PNG = 2
} RLMeshViewFormat;
typedef enum RLMeshViewHold {
  RLMESH_VIEW_HOLD_INVALID = 0,
  RLMESH_VIEW_HOLD_OFF = 1,
  RLMESH_VIEW_HOLD_UNTIL_QUIT = 2,
  RLMESH_VIEW_HOLD_SECONDS = 3
} RLMeshViewHold;
typedef struct RLMeshView {
  size_t struct_size;
  const char* source;
  uint64_t fps;
  double hold_seconds;
  double step_hz;
  RLMeshViewBackend backend;
  RLMeshViewFormat format;
  RLMeshViewHold hold;
  uint16_t port;
  uint8_t quality;
  bool has_step_hz;
} RLMeshView;
#define RLMESH_VIEW_MIN_SIZE sizeof(RLMeshView)
#define RLMESH_VIEW_INIT            \
  {                                 \
      sizeof(RLMeshView),           \
      NULL,                         \
      30,                           \
      0.0,                          \
      0.0,                          \
      RLMESH_VIEW_BACKEND_TERMINAL, \
      RLMESH_VIEW_FORMAT_JPEG,      \
      RLMESH_VIEW_HOLD_OFF,         \
      8008,                         \
      75,                           \
      false}
RLMESH_API RLMeshStatus rlmesh_view_validate(const RLMeshView* view);
RLMESH_API RLMeshStatus rlmesh_view_parse(const char* text, RLMeshView* out);
RLMESH_API RLMeshStatus rlmesh_view_equals(const RLMeshView* view, const RLMeshView* view_1,
                                           bool* out);
RLMESH_API RLMeshStatus rlmesh_view_repr(const RLMeshView* view, RLMeshBytes* out);

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
  uint8_t _reserved[6];
  const char* instruction; /* NULL = None; copied. */
  const RLMeshView* view;  /* NULL = None; copied and validated by the facade. */
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
   {0, 0},                      \
   NULL,                        \
   NULL}
/* NULL options means INIT; zero-initialized options are refused. Result remains
 * NULL on failure. Hooks/targets are borrowed until the call returns. Instruction
 * is copied. A local Model.run refuses view; use a session for local viewing. */
RLMESH_API RLMeshStatus rlmesh_run(RLMeshModelTarget model, RLMeshEnvTarget env,
                                   const RLMeshRunOptions* options, RLMeshRunResult** out);
RLMESH_API RLMeshStatus rlmesh_model_run(RLMeshModel* model, RLMeshEnvTarget env_or_address,
                                         const RLMeshRunOptions* options, RLMeshRunResult** out);

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
  const char* instruction; /* NULL = None; copied. */
  const RLMeshView* view;  /* NULL = None; copied. */
} RLMeshSessionOptions;
#define RLMESH_SESSION_OPTIONS_MIN_SIZE sizeof(RLMeshSessionOptions)
#define RLMESH_SESSION_OPTIONS_INIT \
  {sizeof(RLMeshSessionOptions), NULL, NULL, 1, false, false, false, 0, NULL, NULL}
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
RLMESH_API RLMeshStatus rlmesh_session_new(RLMeshModelTarget model, RLMeshEnvTarget env,
                                           const RLMeshSessionOptions* options,
                                           RLMeshSession** out);
RLMESH_API RLMeshStatus rlmesh_model_session(RLMeshModel* model, RLMeshEnvTarget env_or_address,
                                             const RLMeshSessionOptions* options,
                                             RLMeshSession** out);
RLMESH_API RLMeshStatus rlmesh_session_reset(RLMeshSession* session, const int64_t* seed,
                                             const int64_t* trial_index,
                                             RLMeshValue** out_observation, RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_session_predict(RLMeshSession* session,
                                               const RLMeshValue* observation,
                                               RLMeshValue** out_action);
RLMESH_API RLMeshStatus rlmesh_session_step(RLMeshSession* session, const RLMeshValue* action,
                                            RLMeshValue** out_observation, double* out_reward,
                                            bool* out_terminated, bool* out_truncated,
                                            RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_session_run(RLMeshSession* session,
                                           const RLMeshSessionRunOptions* options,
                                           RLMeshRunResult** out);
RLMESH_API RLMeshStatus rlmesh_session_done(const RLMeshSession* session, bool* out);
RLMESH_API RLMeshStatus rlmesh_session_contract(RLMeshSession* session, RLMeshEnvContract** out);
RLMESH_API RLMeshStatus rlmesh_session_num_envs(RLMeshSession* session, uint64_t* out);
RLMESH_API RLMeshStatus rlmesh_session_selected_workflow_edition(RLMeshSession* session,
                                                                 RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_session_run_context(RLMeshSession* session, RLMeshRunContext** out);
RLMESH_API RLMeshStatus rlmesh_session_cancel_handle(const RLMeshSession* session,
                                                     RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_session_close(RLMeshSession* session);
/* NULL no-op. From the session's own run hook, refuses to release the handle
 * and records ERR_RUNTIME in last error; the caller must free it after the run. */
RLMESH_API void rlmesh_session_free(RLMeshSession* session);

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
RLMESH_API RLMeshStatus rlmesh_remote_env_new(const RLMeshRemoteEnvOptions* options,
                                              RLMeshRemoteEnv** out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_new(const RLMeshRemoteEnvOptions* options,
                                                     RLMeshRemoteVectorEnv** out);
RLMESH_API RLMeshStatus rlmesh_remote_model_new(const RLMeshRemoteModelOptions* options,
                                                RLMeshRemoteModel** out);
RLMESH_API RLMeshStatus rlmesh_ext_remote_env_new_with_token(const RLMeshRemoteEnvOptions* options,
                                                             const char* token,
                                                             RLMeshRemoteEnv** out);
RLMESH_API RLMeshStatus rlmesh_ext_remote_vector_env_new_with_token(
    const RLMeshRemoteEnvOptions* options, const char* token, RLMeshRemoteVectorEnv** out);
RLMESH_API RLMeshStatus rlmesh_ext_remote_model_new_with_token(
    const RLMeshRemoteModelOptions* options, const char* token, RLMeshRemoteModel** out);
RLMESH_API RLMeshStatus rlmesh_remote_model_clone(const RLMeshRemoteModel* remote_model,
                                                  RLMeshRemoteModel** out);
RLMESH_API RLMeshStatus rlmesh_remote_model_address(const RLMeshRemoteModel* remote_model,
                                                    RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_model_repr(const RLMeshRemoteModel* remote_model,
                                                 RLMeshBytes* out);
RLMESH_API void rlmesh_remote_model_free(RLMeshRemoteModel* remote_model);
RLMESH_API RLMeshModelTarget rlmesh_remote_model_target(const RLMeshRemoteModel* remote_model);
RLMESH_API RLMeshEnvTarget rlmesh_remote_env_target(RLMeshRemoteEnv* remote_env);
RLMESH_API RLMeshEnvTarget
rlmesh_remote_vector_env_target(RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API RLMeshStatus rlmesh_remote_model_session(const RLMeshRemoteModel* model,
                                                    RLMeshEnvTarget env,
                                                    const RLMeshSessionOptions* options,
                                                    RLMeshSession** out);
RLMESH_API RLMeshStatus rlmesh_remote_env_address(const RLMeshRemoteEnv* remote_env,
                                                  RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_env_env_id(const RLMeshRemoteEnv* remote_env,
                                                 RLMeshBytes* out);
RLMESH_API RLMeshStatus
rlmesh_remote_env_selected_workflow_edition(const RLMeshRemoteEnv* remote_env, RLMeshBytes* out);
RLMESH_API const RLMeshEnvContract* rlmesh_remote_env_env_contract(
    const RLMeshRemoteEnv* remote_env);
RLMESH_API const RLMeshValue* rlmesh_remote_env_metadata(const RLMeshRemoteEnv* remote_env);
RLMESH_API const RLMeshSpace* rlmesh_remote_env_observation_space(
    const RLMeshRemoteEnv* remote_env);
RLMESH_API const RLMeshSpace* rlmesh_remote_env_action_space(const RLMeshRemoteEnv* remote_env);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_env_observation_space_spec(
    const RLMeshRemoteEnv* remote_env);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_env_action_space_spec(
    const RLMeshRemoteEnv* remote_env);
RLMESH_API RLMeshStatus rlmesh_remote_env_render_mode(const RLMeshRemoteEnv* remote_env,
                                                      bool* out_has, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_env_cancel_handle(const RLMeshRemoteEnv* remote_env,
                                                        RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_remote_env_render(RLMeshRemoteEnv* remote_env, uint64_t env_index,
                                                 RLMeshValue** out_frame);
RLMESH_API RLMeshStatus rlmesh_remote_env_shutdown(RLMeshRemoteEnv* remote_env, const char* reason,
                                                   bool* out_accepted);
RLMESH_API RLMeshStatus rlmesh_remote_env_close(RLMeshRemoteEnv* remote_env);
RLMESH_API void rlmesh_remote_env_free(RLMeshRemoteEnv* remote_env);
RLMESH_API RLMeshStatus
rlmesh_remote_vector_env_address(const RLMeshRemoteVectorEnv* remote_vector_env, RLMeshBytes* out);
RLMESH_API RLMeshStatus
rlmesh_remote_vector_env_env_id(const RLMeshRemoteVectorEnv* remote_vector_env, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_selected_workflow_edition(
    const RLMeshRemoteVectorEnv* remote_vector_env, RLMeshBytes* out);
RLMESH_API const RLMeshEnvContract* rlmesh_remote_vector_env_env_contract(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API const RLMeshValue* rlmesh_remote_vector_env_metadata(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_observation_space(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_action_space(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_vector_env_observation_space_spec(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API const RLMeshSpaceSpec* rlmesh_remote_vector_env_action_space_spec(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_render_mode(
    const RLMeshRemoteVectorEnv* remote_vector_env, bool* out_has, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_cancel_handle(
    const RLMeshRemoteVectorEnv* remote_vector_env, RLMeshExtCancelToken** out);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_render(RLMeshRemoteVectorEnv* remote_vector_env,
                                                        uint64_t env_index,
                                                        RLMeshValue** out_frame);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_shutdown(RLMeshRemoteVectorEnv* remote_vector_env,
                                                          const char* reason, bool* out_accepted);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_close(RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API void rlmesh_remote_vector_env_free(RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API uint64_t
rlmesh_remote_vector_env_num_envs(const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_single_observation_space(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API const RLMeshSpace* rlmesh_remote_vector_env_single_action_space(
    const RLMeshRemoteVectorEnv* remote_vector_env);
RLMESH_API RLMeshStatus rlmesh_remote_env_reset(RLMeshRemoteEnv* remote_env, const int64_t* seed,
                                                const RLMeshValue* options,
                                                RLMeshValue** out_observation,
                                                RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_remote_env_step(RLMeshRemoteEnv* env, const RLMeshValue* action,
                                               RLMeshValue** out_observation, double* out_reward,
                                               bool* out_terminated, bool* out_truncated,
                                               RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_reset(RLMeshRemoteVectorEnv* remote_vector_env,
                                                       const int64_t* seed, size_t seed_len,
                                                       const RLMeshValue* options,
                                                       RLMeshValue** out_observation,
                                                       RLMeshValue** out_info);
/* A short capacity fails before I/O and leaves ALL outputs untouched. */
RLMESH_API RLMeshStatus rlmesh_remote_vector_env_step(RLMeshRemoteVectorEnv* env,
                                                      const RLMeshValue* actions,
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

/* Read items copy leaves. Readers retain a resolved adapter and own their roles. */
RLMESH_API RLMeshStatus rlmesh_read_item_role(const char* role, RLMeshReadItem** out);
RLMESH_API RLMeshStatus rlmesh_read_item_image(const RLMeshImage* image, RLMeshReadItem** out);
RLMESH_API RLMeshStatus rlmesh_read_item_state(const RLMeshState* state, RLMeshReadItem** out);
RLMESH_API RLMeshStatus rlmesh_read_item_text(const RLMeshAdaptersText* adapters_text,
                                              RLMeshReadItem** out);
RLMESH_API void rlmesh_read_item_free(RLMeshReadItem* read_item);
RLMESH_API RLMeshStatus rlmesh_reader_new(const RLMeshAdapter* adapter, const char* const* roles,
                                          size_t roles_len, RLMeshReader** out);
RLMESH_API void rlmesh_reader_free(RLMeshReader* reader);
RLMESH_API size_t rlmesh_reader_roles_len(const RLMeshReader* reader);
RLMESH_API RLMeshStatus rlmesh_reader_roles_get(const RLMeshReader* reader, size_t i,
                                                RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_reader_call(const RLMeshReader* reader, const RLMeshValue* value,
                                           RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_session_reader(RLMeshSession* session,
                                              const RLMeshReadItem* const* items, size_t items_len,
                                              RLMeshReader** out);
RLMESH_API RLMeshStatus rlmesh_session_read(RLMeshSession* session, const RLMeshValue* observation,
                                            const RLMeshReadItem* item, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_session_observation_roles(RLMeshSession* session,
                                                         RLMeshObservationRoles* out);
RLMESH_API RLMeshStatus rlmesh_run_context_read(const RLMeshRunContext* context,
                                                const RLMeshValue* observation,
                                                const RLMeshReadItem* item, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_step_event_read(const RLMeshStepEvent* step_event,
                                               const RLMeshReadItem* read_item, RLMeshValue** out);
/* Session-made sources retain weak state: render after session free returns
 * ERR_RUNTIME. Context-made sources retain the context. Facade closed/busy/
 * dispatch checks remain in force, including after a C callback transfers sources. */
typedef RLMeshStatus (*RLMeshRenderFn)(void* user_data, RLMeshValue** out);
RLMESH_API RLMeshStatus rlmesh_frame_sources_new(const char* const* roles, size_t roles_len,
                                                 const char* render_label, RLMeshRenderFn render,
                                                 void* user_data,
                                                 void (*free_user_data)(void* user_data),
                                                 RLMeshFrameSources** out);
RLMESH_API RLMeshStatus rlmesh_session_frame_sources(RLMeshSession* session,
                                                     RLMeshFrameSources** out);
RLMESH_API RLMeshStatus rlmesh_run_context_frame_sources(const RLMeshRunContext* run_context,
                                                         RLMeshFrameSources** out);
RLMESH_API size_t rlmesh_frame_sources_roles_len(const RLMeshFrameSources* frame_sources);
RLMESH_API RLMeshStatus rlmesh_frame_sources_roles_get(const RLMeshFrameSources* frame_sources,
                                                       size_t i, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_frame_sources_render_label(const RLMeshFrameSources* frame_sources,
                                                          RLMeshBytes* out, bool* out_has);
RLMESH_API RLMeshStatus rlmesh_frame_sources_render(const RLMeshFrameSources* frame_sources,
                                                    RLMeshValue** out, bool* out_has);
RLMESH_API size_t rlmesh_frame_sources_labels_len(const RLMeshFrameSources* frame_sources);
RLMESH_API RLMeshStatus rlmesh_frame_sources_labels_get(const RLMeshFrameSources* frame_sources,
                                                        size_t i, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_frame_sources_default_label(const RLMeshFrameSources* frame_sources,
                                                           RLMeshBytes* out, bool* out_has);
RLMESH_API bool rlmesh_frame_sources_equals(const RLMeshFrameSources* frame_sources,
                                            const RLMeshFrameSources* frame_sources_1);
RLMESH_API RLMeshStatus rlmesh_frame_sources_repr(const RLMeshFrameSources* frame_sources,
                                                  RLMeshBytes* out);
RLMESH_API void rlmesh_frame_sources_free(RLMeshFrameSources* frame_sources);
/* Binding introspection: independent of render_label, without invoking render. */
RLMESH_API bool rlmesh_ext_frame_sources_has_render(const RLMeshFrameSources* frame_sources);
RLMESH_API size_t rlmesh_run_result_advisories_len(const RLMeshRunResult* run_result);
RLMESH_API RLMeshStatus rlmesh_run_result_advisories_get(const RLMeshRunResult* run_result,
                                                         size_t i, RLMeshAdvisory* out);

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
/* Opaque host exception attached to this callback failure; ownership transfers.
 * Replacing, clearing or dropping an untaken payload calls free_payload once.
 * The error carries it across worker threads. The failing caller may take it. */
RLMESH_API void rlmesh_ext_callback_set_error_payload(void* payload,
                                                      void (*free_payload)(void* user_data));
RLMESH_API void* rlmesh_ext_last_error_take_payload(void);
/* Tagged ownership transfer: the library never reads tag or payload bytes.
 * The tag identifies a payload representation/owner domain. The legacy setter
 * records a NULL tag. Use the returned deleter for foreign representations.
 * On success, all outputs are initialized (NULL when empty); outputs must be
 * distinct writable slots. All three are required. A failing call does not
 * consume the payload. A successful take transfers the deleter obligation. */
RLMESH_API void rlmesh_ext_callback_set_tagged_error_payload(void* payload,
                                                             void (*free_payload)(void* user_data),
                                                             const void* tag);
RLMESH_API RLMeshStatus rlmesh_ext_last_error_take_payload_owned(
    void** out_payload, void (**out_free_payload)(void* user_data), const void** out_tag);

/* ---- Recorder and immutable recorder records ---------------------------- */
#define RLMESH_RECORDER_SCHEMA "rlmesh.result.v1"
typedef struct RLMeshRecorder RLMeshRecorder;
typedef struct RLMeshMediaRef RLMeshMediaRef;
typedef struct RLMeshEpisodeRecord RLMeshEpisodeRecord;
typedef struct RLMeshWorkloadRecord RLMeshWorkloadRecord;
/* Options are copied. NULL = INIT, undersized structs are refused, reserved and
 * unknown trailing bytes must be zero. configs are DICT Values, NULL = {}.
 * Calls on one recorder serialize. Free must not overlap another call.
 * Record snapshots own their data independently of recorders and input handles. */
typedef struct RLMeshRecorderOptions {
  size_t struct_size;
  const char* result_set_id;
  uint64_t fps;
  uint8_t quality;
  uint8_t _reserved[7];
} RLMeshRecorderOptions;
#define RLMESH_RECORDER_OPTIONS_MIN_SIZE sizeof(RLMeshRecorderOptions)
#define RLMESH_RECORDER_OPTIONS_INIT {sizeof(RLMeshRecorderOptions), NULL, 30, 60, {0}}
typedef struct RLMeshRecorderAddOptions {
  size_t struct_size;
  const char* task;
  const RLMeshValue* config;
  bool included_in_metrics;
  uint8_t _reserved[sizeof(size_t) - 1];
} RLMeshRecorderAddOptions;
#define RLMESH_RECORDER_ADD_OPTIONS_MIN_SIZE sizeof(RLMeshRecorderAddOptions)
#define RLMESH_RECORDER_ADD_OPTIONS_INIT {sizeof(RLMeshRecorderAddOptions), NULL, NULL, true, {0}}
typedef struct RLMeshRecorderCaptureOptions {
  size_t struct_size;
  const char* task;
  const RLMeshValue* config;
  const char* const* cameras;
  size_t cameras_len;
  const RLMeshRunContext* session;
  const char* const* video_info_keys;
  size_t video_info_keys_len;
  bool included_in_metrics;
  uint8_t _reserved[sizeof(size_t) - 1];
} RLMeshRecorderCaptureOptions;
#define RLMESH_RECORDER_CAPTURE_OPTIONS_MIN_SIZE sizeof(RLMeshRecorderCaptureOptions)
#define RLMESH_RECORDER_CAPTURE_OPTIONS_INIT \
  {sizeof(RLMeshRecorderCaptureOptions), NULL, NULL, NULL, 0, NULL, NULL, 0, true, {0}}
/* NULL cameras = discover all; non-NULL/0 = metrics only. NULL video_info_keys
 * = ["video_artifact_path", "video_url"]; non-NULL/0 = empty. A nonzero count
 * with a NULL list, or a NULL list element, is refused. Capture retains session
 * and recorder state through run_hooks_free, even after recorder_free.
 * Returned hooks work with run/session and all run_hooks_on_* calls. */
typedef struct RLMeshRecorderToDictOptions {
  size_t struct_size;
  const char* recorded_at;
} RLMeshRecorderToDictOptions;
#define RLMESH_RECORDER_TO_DICT_OPTIONS_MIN_SIZE sizeof(RLMeshRecorderToDictOptions)
#define RLMESH_RECORDER_TO_DICT_OPTIONS_INIT {sizeof(RLMeshRecorderToDictOptions), NULL}
typedef struct RLMeshRecorderExportOptions {
  size_t struct_size;
  const char* recorded_at;
  bool has_archive;
  bool archive;
  uint8_t _reserved[sizeof(size_t) - 2];
} RLMeshRecorderExportOptions;
#define RLMESH_RECORDER_EXPORT_OPTIONS_MIN_SIZE sizeof(RLMeshRecorderExportOptions)
#define RLMESH_RECORDER_EXPORT_OPTIONS_INIT \
  {sizeof(RLMeshRecorderExportOptions), NULL, false, false, {0}}
RLMESH_API RLMeshStatus rlmesh_recorder_new(const RLMeshRecorderOptions* options,
                                            RLMeshRecorder** out);
RLMESH_API RLMeshStatus rlmesh_recorder_add(RLMeshRecorder* recorder, const RLMeshRunResult* result,
                                            const char* model, const char* env,
                                            const RLMeshRecorderAddOptions* options,
                                            RLMeshWorkloadRecord** out);
RLMESH_API RLMeshStatus rlmesh_recorder_capture(RLMeshRecorder* recorder, const char* model,
                                                const char* env,
                                                const RLMeshRecorderCaptureOptions* options,
                                                RLMeshRunHooks** out);
RLMESH_API RLMeshStatus rlmesh_recorder_to_dict(const RLMeshRecorder* recorder,
                                                const RLMeshRecorderToDictOptions* options,
                                                RLMeshValue** out);
/* export returns owned UTF-8 path bytes. Refused targets and I/O fail
 * with ERR_OS. After close, exporting removed staged media fails ERR_RUNTIME;
 * exporting metrics without staged media still works. */
RLMESH_API RLMeshStatus rlmesh_recorder_export(const RLMeshRecorder* recorder, const char* path,
                                               const RLMeshRecorderExportOptions* options,
                                               RLMeshBytes* out_path);
RLMESH_API RLMeshStatus rlmesh_recorder_result_set_id(const RLMeshRecorder* recorder,
                                                      RLMeshBytes* out);
RLMESH_API size_t rlmesh_recorder_workloads_len(const RLMeshRecorder* recorder);
RLMESH_API RLMeshStatus rlmesh_recorder_workloads_get(const RLMeshRecorder* recorder, size_t i,
                                                      RLMeshWorkloadRecord** out);
RLMESH_API RLMeshStatus rlmesh_recorder_close(RLMeshRecorder* recorder);
/* Closes with errors swallowed; NULL no-op. Captures retain the shared state. */
RLMESH_API void rlmesh_recorder_free(RLMeshRecorder* recorder);

typedef struct RLMeshMediaRefOptions {
  size_t struct_size;
  bool has_frame_count;
  int64_t frame_count;
  bool has_width;
  int64_t width;
  bool has_height;
  int64_t height;
  bool has_fps;
  double fps;
} RLMeshMediaRefOptions;
#define RLMESH_MEDIA_REF_OPTIONS_MIN_SIZE sizeof(RLMeshMediaRefOptions)
#define RLMESH_MEDIA_REF_OPTIONS_INIT \
  {sizeof(RLMeshMediaRefOptions), false, 0, false, 0, false, 0, false, 0.0}
typedef struct RLMeshEpisodeRecordOptions {
  size_t struct_size;
  bool has_success;
  bool success;
  double duration_s;
  bool included_in_metrics;
  const RLMeshMediaRef* const* media;
  size_t media_len;
} RLMeshEpisodeRecordOptions;
#define RLMESH_EPISODE_RECORD_OPTIONS_MIN_SIZE sizeof(RLMeshEpisodeRecordOptions)
#define RLMESH_EPISODE_RECORD_OPTIONS_INIT \
  {sizeof(RLMeshEpisodeRecordOptions), false, false, 0.0, true, NULL, 0}
typedef struct RLMeshWorkloadRecordOptions {
  size_t struct_size;
  const RLMeshValue* config;
  const RLMeshEpisodeRecord* const* episodes;
  size_t episodes_len;
} RLMeshWorkloadRecordOptions;
#define RLMESH_WORKLOAD_RECORD_OPTIONS_MIN_SIZE sizeof(RLMeshWorkloadRecordOptions)
#define RLMESH_WORKLOAD_RECORD_OPTIONS_INIT {sizeof(RLMeshWorkloadRecordOptions), NULL, NULL, 0}
/* Required strings cannot be NULL. Records copy values without adding
 * validation beyond C representation (matching Python's dataclasses).
 * Child lists: NULL/0 = empty, NULL/nonzero or a NULL element is refused.
 * _get returns a new owned snapshot; i >= len is ERR_INVALID_ARGUMENT.
 * Scalars/lengths return zero on NULL. Optional reads return false for None.
 * _to_dict returns an owned DICT; _repr returns owned UTF-8 bytes. */
RLMESH_API RLMeshStatus rlmesh_media_ref_new(const char* camera, const char* kind, const char* path,
                                             const char* format,
                                             const RLMeshMediaRefOptions* options,
                                             RLMeshMediaRef** out);
RLMESH_API RLMeshStatus rlmesh_media_ref_camera(const RLMeshMediaRef* media_ref, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_media_ref_kind(const RLMeshMediaRef* media_ref, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_media_ref_path(const RLMeshMediaRef* media_ref, RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_media_ref_format(const RLMeshMediaRef* media_ref, RLMeshBytes* out);
RLMESH_API bool rlmesh_media_ref_frame_count(const RLMeshMediaRef* media_ref, int64_t* out);
RLMESH_API bool rlmesh_media_ref_width(const RLMeshMediaRef* media_ref, int64_t* out);
RLMESH_API bool rlmesh_media_ref_height(const RLMeshMediaRef* media_ref, int64_t* out);
RLMESH_API bool rlmesh_media_ref_fps(const RLMeshMediaRef* media_ref, double* out);
RLMESH_API RLMeshStatus rlmesh_media_ref_to_dict(const RLMeshMediaRef* media_ref,
                                                 RLMeshValue** out);
RLMESH_API bool rlmesh_media_ref_equals(const RLMeshMediaRef* media_ref,
                                        const RLMeshMediaRef* media_ref_1);
RLMESH_API RLMeshStatus rlmesh_media_ref_repr(const RLMeshMediaRef* media_ref, RLMeshBytes* out);
RLMESH_API void rlmesh_media_ref_free(RLMeshMediaRef* media_ref);
RLMESH_API RLMeshStatus rlmesh_episode_record_new(int64_t index, const int64_t* seed, int64_t steps,
                                                  double reward, bool terminated, bool truncated,
                                                  const RLMeshEpisodeRecordOptions* options,
                                                  RLMeshEpisodeRecord** out);
RLMESH_API int64_t rlmesh_episode_record_index(const RLMeshEpisodeRecord* episode_record);
RLMESH_API bool rlmesh_episode_record_seed(const RLMeshEpisodeRecord* episode_record, int64_t* out);
RLMESH_API int64_t rlmesh_episode_record_steps(const RLMeshEpisodeRecord* episode_record);
RLMESH_API double rlmesh_episode_record_reward(const RLMeshEpisodeRecord* episode_record);
RLMESH_API bool rlmesh_episode_record_terminated(const RLMeshEpisodeRecord* episode_record);
RLMESH_API bool rlmesh_episode_record_truncated(const RLMeshEpisodeRecord* episode_record);
RLMESH_API bool rlmesh_episode_record_success(const RLMeshEpisodeRecord* episode_record, bool* out);
RLMESH_API double rlmesh_episode_record_duration_s(const RLMeshEpisodeRecord* episode_record);
RLMESH_API bool rlmesh_episode_record_included_in_metrics(
    const RLMeshEpisodeRecord* episode_record);
RLMESH_API size_t rlmesh_episode_record_media_len(const RLMeshEpisodeRecord* episode_record);
RLMESH_API RLMeshStatus rlmesh_episode_record_media_get(const RLMeshEpisodeRecord* episode_record,
                                                        size_t i, RLMeshMediaRef** out);
RLMESH_API RLMeshStatus rlmesh_episode_record_to_dict(const RLMeshEpisodeRecord* episode_record,
                                                      RLMeshValue** out);
RLMESH_API bool rlmesh_episode_record_equals(const RLMeshEpisodeRecord* episode_record,
                                             const RLMeshEpisodeRecord* episode_record_1);
RLMESH_API RLMeshStatus rlmesh_episode_record_repr(const RLMeshEpisodeRecord* episode_record,
                                                   RLMeshBytes* out);
RLMESH_API void rlmesh_episode_record_free(RLMeshEpisodeRecord* episode_record);
RLMESH_API RLMeshStatus rlmesh_workload_record_new(const char* model, const char* env,
                                                   const char* task,
                                                   const RLMeshWorkloadRecordOptions* options,
                                                   RLMeshWorkloadRecord** out);
RLMESH_API RLMeshStatus rlmesh_workload_record_model(const RLMeshWorkloadRecord* workload_record,
                                                     RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_workload_record_env(const RLMeshWorkloadRecord* workload_record,
                                                   RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_workload_record_task(const RLMeshWorkloadRecord* workload_record,
                                                    RLMeshBytes* out);
RLMESH_API RLMeshStatus rlmesh_workload_record_config(const RLMeshWorkloadRecord* workload_record,
                                                      RLMeshValue** out);
RLMESH_API size_t rlmesh_workload_record_episodes_len(const RLMeshWorkloadRecord* workload_record);
RLMESH_API RLMeshStatus rlmesh_workload_record_episodes_get(
    const RLMeshWorkloadRecord* workload_record, size_t i, RLMeshEpisodeRecord** out);
RLMESH_API double rlmesh_workload_record_mean_reward(const RLMeshWorkloadRecord* workload_record);
RLMESH_API bool rlmesh_workload_record_success_rate(const RLMeshWorkloadRecord* workload_record,
                                                    double* out);
RLMESH_API int64_t rlmesh_workload_record_total_steps(const RLMeshWorkloadRecord* workload_record);
RLMESH_API RLMeshStatus rlmesh_workload_record_to_dict(const RLMeshWorkloadRecord* workload_record,
                                                       RLMeshValue** out);
RLMESH_API bool rlmesh_workload_record_equals(const RLMeshWorkloadRecord* workload_record,
                                              const RLMeshWorkloadRecord* workload_record_1);
RLMESH_API RLMeshStatus rlmesh_workload_record_repr(const RLMeshWorkloadRecord* workload_record,
                                                    RLMeshBytes* out);
RLMESH_API void rlmesh_workload_record_free(RLMeshWorkloadRecord* workload_record);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RLMESH_H */
