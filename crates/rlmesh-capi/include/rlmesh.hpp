// RLMesh C++ wrapper — experimental, header-only, C++17. A thin RAII layer over
// the C ABI (rlmesh.h): a `Model` driving or serving a policy, and an
// `EnvServer` serving an `Environment` you subclass.
//
// Error handling, in one place — there is exactly one model:
//   * A capi call reports failure by status (or a NULL return) and has ALREADY
//     recorded the detail on this thread. `Error::from_last(status)` snapshots
//     it; the wrapper never invents a message for a call that already failed.
//     The pointer-returning capi calls behind the `Value` constructors report
//     failure as a NULL return; their code comes from `rlmesh_last_error_status`,
//     so `code()` is exact there too.
//   * Every fallible wrapper call returns `Result<T>` (`Status` == `Result<void>`).
//     Check it (`if (!r) ... r.error()`), `*r` / `r->` to use the value, or
//     `RLMESH_TRY(expr)` to propagate. `value()` / `unwrap()` abort on an error
//     rather than throw — this header advertises no exceptions and compiles
//     under `-fno-exceptions`.
//   * A callback (predict, or an env's reset / step / render) declines by
//     returning an `Error` through its `Result`. The wrapper forwards it to
//     `rlmesh_callback_set_error` and returns its code.
//   * With exceptions enabled, the header also defines the exception classes of
//     the Python package, with Python's inheritance (`RLMeshException`,
//     `ModelException`, `ValueError`, ...; `rlmesh::ext::CancelledError` has no
//     `std::` base). `throw_error(error)` / `Result::value_or_throw()` raise the
//     class an `Error` belongs to, and a callback may throw
//     `ModelException(msg, recoverable)` / `(Recoverable)EnvironmentException(msg)`
//     to decline. Without exceptions they are left out and nothing else changes.
#ifndef RLMESH_HPP
#define RLMESH_HPP

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "rlmesh.h"

namespace rlmesh {

// ---- errors + results -------------------------------------------------------

namespace detail {

/// The class a status reports when nothing more specific is known.
inline RLMeshErrorClass class_for_status(RLMeshStatus status) {
  switch (status) {
    case RLMESH_OK:
      return RLMESH_ERROR_CLASS_INVALID;
    case RLMESH_ERR_INVALID_ARGUMENT:
      return RLMESH_ERROR_CLASS_VALUE_ERROR;
    case RLMESH_ERR_INVALID_VALUE:
    case RLMESH_ERR_TYPE:
      return RLMESH_ERROR_CLASS_TYPE_ERROR;
    case RLMESH_ERR_ENVIRONMENT:
      return RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION;
    case RLMESH_ERR_MODEL:
      return RLMESH_ERROR_CLASS_MODEL_EXCEPTION;
    case RLMESH_ERR_CONNECTION:
      return RLMESH_ERROR_CLASS_CONNECTION_ERROR;
    case RLMESH_ERR_TIMEOUT:
      return RLMESH_ERROR_CLASS_TIMEOUT_ERROR;
    case RLMESH_ERR_CANCELLED:
      return RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT;
    case RLMESH_ERR_PROTOCOL:
      return RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION;
    case RLMESH_ERR_RUNTIME:
      return RLMESH_ERROR_CLASS_RUNTIME_ERROR;
    case RLMESH_ERR_OS:
      return RLMESH_ERROR_CLASS_OS_ERROR;
    case RLMESH_ERR_PANIC:
    case RLMESH_ERR_INTERNAL:
    default:
      return RLMESH_ERROR_CLASS_RLMESH_EXCEPTION;
  }
}

}  // namespace detail

/// A failure: a status code plus an owned snapshot of the message, the exact
/// Python class (`error_class()`), and the environment or model error code
/// when the class carries one.
class Error {
 public:
  Error(RLMeshStatus code, std::string message, bool recoverable = false)
      : code_(code),
        message_(std::move(message)),
        recoverable_(recoverable),
        class_(recoverable && code == RLMESH_ERR_ENVIRONMENT
                   ? RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION
                   : detail::class_for_status(code)) {}

  /// Snapshot the thread-local last error after a nonzero status (or a NULL
  /// return from a pointer-returning capi call).
  static Error from_last(RLMeshStatus code) {
    const char* message = rlmesh_last_error_message();
    Error error(code, message ? std::string(message) : std::string(),
                rlmesh_last_error_is_recoverable() != 0);
    RLMeshErrorClass cls = rlmesh_last_error_class();
    if (cls != RLMESH_ERROR_CLASS_INVALID) error.class_ = cls;
    int32_t detail_code = 0;
    if (rlmesh_last_error_code(&detail_code)) error.detail_code_ = detail_code;
    return error;
  }

  RLMeshStatus code() const { return code_; }
  const std::string& message() const { return message_; }
  bool is_recoverable() const { return recoverable_; }
  /// The run/serve was stopped by `Model::cancel()` — a clean stop, not a fault.
  bool is_cancelled() const { return code_ == RLMESH_ERR_CANCELLED; }
  /// The exact class, as Python names it (`rlmesh_error_class_name`).
  RLMeshErrorClass error_class() const { return class_; }
  /// The environment (`RLMeshErrorCode`) or model (`RLMeshModelErrorCode`)
  /// code's wire number, when the class carries one.
  std::optional<int32_t> detail_code() const { return detail_code_; }

 private:
  RLMeshStatus code_;
  std::string message_;
  bool recoverable_;
  RLMeshErrorClass class_;
  std::optional<int32_t> detail_code_;
};

#if defined(__cpp_exceptions)

/// An environment error's code (Python's `EnvironmentException.code`).
enum class ErrorCode : int32_t {
  Unspecified = RLMESH_ERROR_CODE_UNSPECIFIED,
  Timeout = RLMESH_ERROR_CODE_TIMEOUT,
  InvalidAction = RLMESH_ERROR_CODE_INVALID_ACTION,
  NotReady = RLMESH_ERROR_CODE_NOT_READY,
  Busy = RLMESH_ERROR_CODE_BUSY,
  Unsupported = RLMESH_ERROR_CODE_UNSUPPORTED,
  Internal = RLMESH_ERROR_CODE_INTERNAL,
  Crashed = RLMESH_ERROR_CODE_CRASHED,
  Cancelled = RLMESH_ERROR_CODE_CANCELLED,
  Closed = RLMESH_ERROR_CODE_CLOSED,
};

/// A model error's code (Python's `ModelException.code`).
enum class ModelErrorCode : int32_t {
  Unspecified = RLMESH_MODEL_ERROR_CODE_UNSPECIFIED,
  InvalidRequest = RLMESH_MODEL_ERROR_CODE_INVALID_REQUEST,
  NotConfigured = RLMESH_MODEL_ERROR_CODE_NOT_CONFIGURED,
  Busy = RLMESH_MODEL_ERROR_CODE_BUSY,
  Unsupported = RLMESH_MODEL_ERROR_CODE_UNSUPPORTED,
  Internal = RLMESH_MODEL_ERROR_CODE_INTERNAL,
  Cancelled = RLMESH_MODEL_ERROR_CODE_CANCELLED,
  Closed = RLMESH_MODEL_ERROR_CODE_CLOSED,
};

/// The code's text ("INVALID_ACTION"); empty for an unknown value.
inline std::string_view to_string(ErrorCode code) {
  const char* text = rlmesh_error_code_str(static_cast<RLMeshErrorCode>(code));
  return text != nullptr ? std::string_view(text) : std::string_view();
}
inline std::string_view to_string(ModelErrorCode code) {
  const char* text = rlmesh_model_error_code_str(static_cast<RLMeshModelErrorCode>(code));
  return text != nullptr ? std::string_view(text) : std::string_view();
}

namespace detail {
/// Tag for the library's own constructors, which carry a code; the public
/// constructors take Python's authoring arguments.
struct FromLibrary {};
/// The wrapper's read of an authored exception's raw message (its `what()`
/// carries the `[CODE]` prefix the wire adds again).
struct ExceptionText;
}  // namespace detail

/// Python's builtin `RuntimeError`: misuse and local refusals.
class RuntimeError : public std::runtime_error {
 public:
  explicit RuntimeError(const std::string& message) : std::runtime_error(message) {}
};

/// `RLMeshException`: an internal or server-side failure. The base of the
/// rlmesh-specific classes; the builtins (`ValueError`, `OSError`, ...) are not
/// under it, as in Python.
class RLMeshException : public RuntimeError {
 public:
  explicit RLMeshException(const std::string& message) : RuntimeError(message) {}
};

/// `ProtocolException`: generation mismatch, refused edition, failed
/// handshake, malformed message.
class ProtocolException : public RLMeshException {
 public:
  explicit ProtocolException(const std::string& message) : RLMeshException(message) {}
};

/// `EnvironmentException`: the env reported a failure. Thrown from an env
/// callback, it fails the request and ends the session (code `INTERNAL`). An
/// authored one's `what()` is `[INTERNAL] message`, as Python's `str()`.
class EnvironmentException : public RLMeshException {
 public:
  explicit EnvironmentException(std::string message)
      : EnvironmentException(std::move(message), false, detail::FromLibrary{}) {}
  EnvironmentException(std::string message, ErrorCode code, bool is_recoverable,
                       detail::FromLibrary)
      : RLMeshException(message), code_(code), recoverable_(is_recoverable), raw_(message) {}

  ErrorCode code() const { return code_; }
  bool is_recoverable() const { return recoverable_; }

 protected:
  /// An authored exception: code `INTERNAL`, text `[INTERNAL] message`.
  EnvironmentException(std::string message, bool is_recoverable, detail::FromLibrary)
      : RLMeshException("[INTERNAL] " + message),
        code_(ErrorCode::Internal),
        recoverable_(is_recoverable),
        raw_(std::move(message)) {}

 private:
  friend struct detail::ExceptionText;
  ErrorCode code_;
  bool recoverable_;
  std::string raw_;
};

/// `RecoverableEnvironmentException`: the request failed, the session stays
/// usable. Throw it from an env callback to fail one request.
class RecoverableEnvironmentException : public EnvironmentException {
 public:
  explicit RecoverableEnvironmentException(std::string message)
      : EnvironmentException(std::move(message), true, detail::FromLibrary{}) {}
  RecoverableEnvironmentException(std::string message, ErrorCode code, detail::FromLibrary tag)
      : EnvironmentException(std::move(message), code, true, tag) {}
};

/// `ModelException`: the model reported a failure. Throw
/// `ModelException(msg, true)` from predict to fail one request as recoverable
/// (code `INTERNAL`); its `what()` is `[INTERNAL] msg`, as Python's `str()`.
class ModelException : public RLMeshException {
 public:
  explicit ModelException(std::string message, bool is_recoverable = false)
      : RLMeshException("[INTERNAL] " + message),
        code_(ModelErrorCode::Internal),
        recoverable_(is_recoverable),
        raw_(std::move(message)) {}
  ModelException(std::string message, ModelErrorCode code, bool is_recoverable, detail::FromLibrary)
      : RLMeshException(message), code_(code), recoverable_(is_recoverable), raw_(message) {}

  ModelErrorCode code() const { return code_; }
  bool is_recoverable() const { return recoverable_; }

 private:
  friend struct detail::ExceptionText;
  ModelErrorCode code_;
  bool recoverable_;
  std::string raw_;
};

namespace detail {
struct ExceptionText {
  static const std::string& raw(const EnvironmentException& error) { return error.raw_; }
  static const std::string& raw(const ModelException& error) { return error.raw_; }
};
}  // namespace detail

/// Python's `ValueError`: an invalid argument value, address or options field.
class ValueError : public std::invalid_argument {
 public:
  explicit ValueError(const std::string& message) : std::invalid_argument(message) {}
};

/// `AdapterResolutionError`: adapters cannot bridge env and model.
class AdapterResolutionError : public ValueError {
 public:
  explicit AdapterResolutionError(const std::string& message) : ValueError(message) {}
};

/// Python's `TypeError`: the wrong kind of value or object.
class TypeError : public std::logic_error {
 public:
  explicit TypeError(const std::string& message) : std::logic_error(message) {}
};

/// Python's `OSError`: a filesystem failure.
class OSError : public std::runtime_error {
 public:
  explicit OSError(const std::string& message) : std::runtime_error(message) {}
};

/// Python's `ConnectionError` (an `OSError`): a transport failure.
class ConnectionError : public OSError {
 public:
  explicit ConnectionError(const std::string& message) : OSError(message) {}
};

/// Python's `TimeoutError` (an `OSError`): a connect or request deadline.
class TimeoutError : public OSError {
 public:
  explicit TimeoutError(const std::string& message) : OSError(message) {}
};

namespace ext {

/// Cancellation (Python's `KeyboardInterrupt`). Derives from no `std::`
/// exception, so `catch (const std::exception&)` does not swallow a cancel.
class CancelledError {
 public:
  explicit CancelledError(std::string message = "cancelled") : message_(std::move(message)) {}
  const char* what() const noexcept { return message_.c_str(); }

 private:
  std::string message_;
};

}  // namespace ext

/// Throw the exception class `error` belongs to, with its message as `what()`.
[[noreturn]] inline void throw_error(const Error& error) {
  const std::string& message = error.message();
  const int32_t raw_code = error.detail_code().value_or(RLMESH_ERROR_CODE_INTERNAL);
  switch (error.error_class()) {
    case RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION:
      throw ProtocolException(message);
    case RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION:
      throw EnvironmentException(message, static_cast<ErrorCode>(raw_code), false,
                                 detail::FromLibrary{});
    case RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION:
      throw RecoverableEnvironmentException(message, static_cast<ErrorCode>(raw_code),
                                            detail::FromLibrary{});
    case RLMESH_ERROR_CLASS_MODEL_EXCEPTION:
      throw ModelException(message, static_cast<ModelErrorCode>(raw_code), error.is_recoverable(),
                           detail::FromLibrary{});
    case RLMESH_ERROR_CLASS_VALUE_ERROR:
      throw ValueError(message);
    case RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR:
      throw AdapterResolutionError(message);
    case RLMESH_ERROR_CLASS_TYPE_ERROR:
      throw TypeError(message);
    case RLMESH_ERROR_CLASS_RUNTIME_ERROR:
      throw RuntimeError(message);
    case RLMESH_ERROR_CLASS_OS_ERROR:
      throw OSError(message);
    case RLMESH_ERROR_CLASS_CONNECTION_ERROR:
      throw ConnectionError(message);
    case RLMESH_ERROR_CLASS_TIMEOUT_ERROR:
      throw TimeoutError(message);
    case RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT:
      throw ext::CancelledError(message);
    case RLMESH_ERROR_CLASS_INVALID:
    case RLMESH_ERROR_CLASS_RLMESH_EXCEPTION:
    default:
      throw RLMeshException(message);
  }
}

namespace detail {

/// Record the exception being handled as the running callback's error
/// (X-10): an exception of the Python classes keeps its class
/// (`rlmesh_callback_set_error_class`), as a Python author's `TypeError`
/// propagates as itself; any other exception is the callback kind's failure
/// with its `what()`. Returns the nonzero status the trampoline returns.
inline int record_current_exception(RLMeshStatus kind, const char* unknown) noexcept {
  RLMeshErrorClass cls = RLMESH_ERROR_CLASS_INVALID;
  std::string text;
  try {
    throw;
  } catch (const RecoverableEnvironmentException& e) {
    cls = RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION;
    text = ExceptionText::raw(e);
  } catch (const EnvironmentException& e) {
    cls = RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION;
    text = ExceptionText::raw(e);
  } catch (const ModelException& e) {
    cls = RLMESH_ERROR_CLASS_MODEL_EXCEPTION;
    text = ExceptionText::raw(e);
  } catch (const ProtocolException& e) {
    cls = RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION;
    text = e.what();
  } catch (const RLMeshException& e) {
    cls = RLMESH_ERROR_CLASS_RLMESH_EXCEPTION;
    text = e.what();
  } catch (const AdapterResolutionError& e) {
    cls = RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR;
    text = e.what();
  } catch (const ValueError& e) {
    cls = RLMESH_ERROR_CLASS_VALUE_ERROR;
    text = e.what();
  } catch (const TypeError& e) {
    cls = RLMESH_ERROR_CLASS_TYPE_ERROR;
    text = e.what();
  } catch (const RuntimeError& e) {
    cls = RLMESH_ERROR_CLASS_RUNTIME_ERROR;
    text = e.what();
  } catch (const TimeoutError& e) {
    cls = RLMESH_ERROR_CLASS_TIMEOUT_ERROR;
    text = e.what();
  } catch (const ConnectionError& e) {
    cls = RLMESH_ERROR_CLASS_CONNECTION_ERROR;
    text = e.what();
  } catch (const OSError& e) {
    cls = RLMESH_ERROR_CLASS_OS_ERROR;
    text = e.what();
  } catch (const ext::CancelledError& e) {
    cls = RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT;
    text = e.what();
  } catch (const std::exception& e) {
    rlmesh_callback_set_error(e.what(), false);
    return static_cast<int>(kind);
  } catch (...) {
    rlmesh_callback_set_error(unknown, false);
    return static_cast<int>(kind);
  }
  rlmesh_callback_set_error_class(cls, text.c_str());
  return static_cast<int>(kind);
}

}  // namespace detail

#endif  // __cpp_exceptions

namespace detail {

/// The one diagnostic an aborting accessor prints. No exceptions: a contract
/// violation is a bug in the caller, not a recoverable condition.
[[noreturn]] inline void die(const char* what, const Error& error) {
  std::fprintf(stderr, "rlmesh: %s: [%d] %s\n", what, static_cast<int>(error.code()),
               error.message().c_str());
  std::abort();
}

inline std::string_view sv(const char* text) {
  return text != nullptr ? std::string_view(text) : std::string_view();
}

}  // namespace detail

/// A value or an Error. `operator bool` / `has_value()` to test, `*` / `->` /
/// `value()` to read, `error()` for the failure. `value()` and `unwrap()` abort
/// (with the same diagnostic) when called on an error — they are assertions.
template <class T>
class [[nodiscard]] Result {
 public:
  Result(T value) : data_(std::move(value)) {}
  Result(Error error) : data_(std::move(error)) {}

  bool has_value() const { return std::holds_alternative<T>(data_); }
  explicit operator bool() const { return has_value(); }

  T& value() { return checked(); }
  const T& value() const { return checked(); }
  T& operator*() { return checked(); }
  const T& operator*() const { return checked(); }
  T* operator->() { return &checked(); }
  const T* operator->() const { return &checked(); }

  const Error& error() const {
    if (const Error* error = std::get_if<Error>(&data_)) return *error;
    detail::die("Result::error() on a success", Error(RLMESH_OK, "no error"));
  }

  /// The value, or `fallback` when this is an error (T must be copyable).
  template <class U>
  T value_or(U&& fallback) const {
    return has_value() ? **this : static_cast<T>(std::forward<U>(fallback));
  }

  /// Move the value out; aborts on an error.
  T unwrap() { return std::move(checked()); }

#if defined(__cpp_exceptions)
  /// Move the value out, or throw the error's exception class (`throw_error`).
  T value_or_throw() {
    if (!has_value()) throw_error(error());
    return std::move(checked());
  }
#endif

  /// Chain: `map` wraps `fn`'s return in a Result, `and_then` expects one.
  /// Both consume the Result (`std::move(result).map(...)` for an lvalue).
  template <class F>
  auto map(F&& fn) && -> Result<decltype(std::forward<F>(fn)(std::declval<T&&>()))> {
    if (!has_value()) return error();
    return std::forward<F>(fn)(std::move(**this));
  }
  template <class F>
  auto and_then(F&& fn) && -> decltype(std::forward<F>(fn)(std::declval<T&&>())) {
    if (!has_value()) return error();
    return std::forward<F>(fn)(std::move(**this));
  }

 private:
  T& checked() {
    if (T* value = std::get_if<T>(&data_)) return *value;
    detail::die("Result::value() on an error", error());
  }
  const T& checked() const {
    if (const T* value = std::get_if<T>(&data_)) return *value;
    detail::die("Result::value() on an error", error());
  }

  std::variant<T, Error> data_;
};

/// Success or an Error, with no value — the `Status` of a void-returning call.
template <>
class [[nodiscard]] Result<void> {
 public:
  Result() = default;
  Result(Error error) : error_(std::move(error)) {}

  bool has_value() const { return !error_.has_value(); }
  bool ok() const { return has_value(); }
  explicit operator bool() const { return has_value(); }

  const Error& error() const {
    if (error_.has_value()) return *error_;
    detail::die("Result::error() on a success", Error(RLMESH_OK, "no error"));
  }

  /// Abort unless this is a success.
  void value() const {
    if (!has_value()) detail::die("Result::value() on an error", error());
  }
  void unwrap() const { value(); }

#if defined(__cpp_exceptions)
  /// Throw the error's exception class (`throw_error`) unless this is a success.
  void value_or_throw() const {
    if (!has_value()) throw_error(error());
  }
#endif

  template <class F>
  auto and_then(F&& fn) const -> decltype(std::forward<F>(fn)()) {
    if (!has_value()) return error();
    return std::forward<F>(fn)();
  }

 private:
  std::optional<Error> error_;
};

using Status = Result<void>;
inline Status ok() { return Status(); }

// Evaluate a Result-returning expression, propagating its Error out of the
// enclosing (Result-returning) function and yielding the value otherwise:
//   int64_t n = RLMESH_TRY(observation.as_discrete());
// A GNU statement expression, so gcc / clang / zig c++ only.
#if defined(__GNUC__)
#define RLMESH_TRY(expr)                          \
  ({                                              \
    auto rlmesh_try_ = (expr);                    \
    if (!rlmesh_try_) return rlmesh_try_.error(); \
    std::move(*rlmesh_try_);                      \
  })
#endif

// ---- build info, editions, warnings -----------------------------------------

/// This header's package version, SemVer (`RLMESH_VERSION`); `build_info()`
/// reports the linked library's.
inline constexpr std::string_view VERSION = RLMESH_VERSION;

/// The describe envelope's metadata key and schema version, and the env reset
/// options key: Python's `rlmesh.DESCRIBE_METADATA_KEY`, ...
inline constexpr std::string_view DESCRIBE_METADATA_KEY = RLMESH_DESCRIBE_METADATA_KEY;
inline constexpr int64_t DESCRIBE_SCHEMA_VERSION = RLMESH_DESCRIBE_SCHEMA_VERSION;
inline constexpr std::string_view ENV_RESET_OPTIONS_KEY = RLMESH_ENV_RESET_OPTIONS_KEY;

/// Where a build's cohort came from (Python's `BuildInfo.build_source`).
enum class BuildSource : int32_t {
  Release = RLMESH_BUILD_SOURCE_RELEASE,
  Package = RLMESH_BUILD_SOURCE_PACKAGE,
  Git = RLMESH_BUILD_SOURCE_GIT,
};

/// Python's spelling (`"release"`).
inline std::string_view to_string(BuildSource source) {
  return detail::sv(rlmesh_build_source_str(static_cast<RLMeshBuildSource>(source)));
}

/// The linked library's identity: Python's `rlmesh.BuildInfo`. Every view is
/// static, with process lifetime.
struct BuildInfo {
  std::string_view version;
  std::string_view protocol_generation;
  std::string_view workflow_edition;
  std::string_view workflow_edition_base;
  std::vector<std::string_view> supported_workflow_editions;  ///< newest first
  std::string_view build_cohort;
  BuildSource build_source = BuildSource::Package;
  std::optional<std::string_view> git;
};

/// This library's build identity (`rlmesh.build_info()`).
inline BuildInfo build_info() {
  RLMeshBuildInfo raw = RLMESH_BUILD_INFO_INIT;
  RLMeshStatus status = rlmesh_build_info(&raw);
  if (status != RLMESH_OK) detail::die("rlmesh_build_info", Error::from_last(status));
  BuildInfo info;
  info.version = detail::sv(raw.version);
  info.protocol_generation = detail::sv(raw.protocol_generation);
  info.workflow_edition = detail::sv(raw.workflow_edition);
  info.workflow_edition_base = detail::sv(raw.workflow_edition_base);
  for (size_t i = 0; i < raw.supported_workflow_editions_len; ++i) {
    info.supported_workflow_editions.push_back(detail::sv(raw.supported_workflow_editions[i]));
  }
  info.build_cohort = detail::sv(raw.build_cohort);
  info.build_source = static_cast<BuildSource>(raw.build_source);
  if (raw.git != nullptr) info.git = detail::sv(raw.git);
  return info;
}

/// Python's `repr(build_info())` (`BuildInfo(version="0.1.0", ...)`), as the
/// library spells it. `info` is a `build_info()` result.
inline std::string to_string(const BuildInfo& info) {
  (void)info;  // the static identity: there is only the one
  RLMeshBytes bytes = {nullptr, 0, 0};
  RLMeshStatus status = rlmesh_build_info_repr(&bytes);
  if (status != RLMESH_OK) detail::die("rlmesh_build_info_repr", Error::from_last(status));
  std::string text(reinterpret_cast<const char*>(bytes.data), bytes.len);
  rlmesh_bytes_free(bytes);
  return text;
}

template <class Stream>
auto operator<<(Stream& out, const BuildInfo& info) -> decltype(out << std::string()) {
  return out << to_string(info);
}

/// The bare `YYYY.MM` workflow edition this build runs at: the value to declare.
inline std::string_view current_workflow_edition() {
  return detail::sv(rlmesh_current_workflow_edition());
}

/// A warning's category: Python's warning classes (values of the C enum).
enum class WarningCategory : int32_t {
  UserWarning = RLMESH_WARNING_CATEGORY_USER_WARNING,
  RuntimeWarning = RLMESH_WARNING_CATEGORY_RUNTIME_WARNING,
  WorkflowEditionWarning = RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING,
};

/// Python's class name (`"WorkflowEditionWarning"`).
inline std::string_view name(WarningCategory category) {
  return detail::sv(rlmesh_warning_category_name(static_cast<RLMeshWarningCategory>(category)));
}

/// Python's `issubclass(cls, base)`.
inline bool is_subclass_of(WarningCategory cls, WarningCategory base) {
  return rlmesh_warning_category_is_subclass(static_cast<RLMeshWarningCategory>(cls),
                                             static_cast<RLMeshWarningCategory>(base));
}

/// One warning.
struct Warning {
  WarningCategory category;
  std::string message;

  bool operator==(const Warning& other) const {
    return category == other.category && message == other.message;
  }
  bool operator!=(const Warning& other) const { return !(*this == other); }
};

/// `<Category>: <message>`.
template <class Stream>
auto operator<<(Stream& out, const Warning& warning) -> decltype(out << std::string()) {
  return out << std::string(name(warning.category)) + ": " + warning.message;
}

namespace detail {

using WarningHandler = std::function<void(const Warning&)>;

inline RLMeshStatus warning_trampoline(void* user_data, RLMeshWarningCategory category,
                                       const char* message) {
  auto* handler = static_cast<WarningHandler*>(user_data);
#if defined(__cpp_exceptions)
  try {
    (*handler)(Warning{static_cast<WarningCategory>(category), std::string(sv(message))});
  } catch (...) {
    return static_cast<RLMeshStatus>(
        record_current_exception(RLMESH_ERR_INTERNAL, "unknown C++ exception"));
  }
#else
  (*handler)(Warning{static_cast<WarningCategory>(category), std::string(sv(message))});
#endif
  return RLMESH_OK;
}

inline void free_warning_handler(void* user_data) {
  delete static_cast<WarningHandler*>(user_data);
}

}  // namespace detail

/// Install the process-wide warning handler; an empty `handler` restores the
/// default (stderr, each distinct warning once). A handler that throws fails
/// the call that warned, with the exception's `what()` as an `RLMeshException`.
inline void set_warning_handler(std::function<void(const Warning&)> handler) {
  if (!handler) {
    (void)rlmesh_set_warning_handler(nullptr, nullptr, nullptr);
    return;
  }
  auto* owned = new detail::WarningHandler(std::move(handler));
  RLMeshStatus status =
      rlmesh_set_warning_handler(&detail::warning_trampoline, owned, &detail::free_warning_handler);
  if (status != RLMESH_OK) {
    delete owned;
    detail::die("rlmesh_set_warning_handler", Error::from_last(status));
  }
}

// ---- dtype + tensor ---------------------------------------------------------

/// The RLMeshDType of a C++ element type (float, double, int32_t, int64_t,
/// uint8_t, bool).
template <class T>
constexpr RLMeshDType dtype_of() {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value ||
                    std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
                    std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                "rlmesh::dtype_of: unsupported element type");
  if constexpr (std::is_same<T, float>::value) {
    return RLMESH_DTYPE_FLOAT32;
  } else if constexpr (std::is_same<T, double>::value) {
    return RLMESH_DTYPE_FLOAT64;
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return RLMESH_DTYPE_INT32;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return RLMESH_DTYPE_INT64;
  } else if constexpr (std::is_same<T, bool>::value) {
    return RLMESH_DTYPE_BOOL;
  } else {
    return RLMESH_DTYPE_UINT8;
  }
}

/// Compatibility view over an owned ABI 5 tensor reference.
class Tensor {
 public:
  explicit Tensor(RLMeshTensor* raw) : raw_(raw, rlmesh_tensor_free) {
    size_t n = 0;
    rlmesh_tensor_ndim(raw, &n);
    std::vector<size_t> shape(n), strides(n);
    rlmesh_tensor_shape(raw, shape.data(), n, &n);
    rlmesh_tensor_strides(raw, strides.data(), n, &n);
    shape_.assign(shape.begin(), shape.end());
    size_t width = rlmesh_ext_dtype_size(rlmesh_tensor_dtype(raw));
    for (size_t stride : strides) strides_.push_back(static_cast<int64_t>(stride / width));
  }
  const void* data() const {
    const void* out = nullptr;
    size_t n = 0;
    rlmesh_tensor_buffer(raw_.get(), &out, &n);
    return out;
  }
  RLMeshDType dtype() const { return rlmesh_tensor_dtype(raw_.get()); }
  int32_t ndim() const { return static_cast<int32_t>(shape_.size()); }
  const int64_t* shape() const { return shape_.data(); }
  const int64_t* strides() const { return is_contiguous() ? nullptr : strides_.data(); }
  bool is_contiguous() const { return rlmesh_tensor_is_contiguous(raw_.get()); }
  size_t numel() const {
    size_t n = 0;
    rlmesh_tensor_size(raw_.get(), &n);
    return n;
  }
  template <class T>
  const T* as() const {
    return is_contiguous() ? static_cast<const T*>(data()) : nullptr;
  }

 private:
  std::shared_ptr<RLMeshTensor> raw_;
  std::vector<int64_t> shape_, strides_;
};

// ---- values -----------------------------------------------------------------

/// A borrowed value view (valid only while its owner lives — for an observation
/// row, the duration of the predict call). Reads every value kind.
class ValueRef {
 public:
  explicit ValueRef(const RLMeshValue* ptr) : ptr_(ptr) {}

  const RLMeshValue* raw() const { return ptr_; }
  RLMeshValueKind kind() const { return rlmesh_value_kind(ptr_); }

  Result<Tensor> as_tensor() const {
    RLMeshTensor* out = nullptr;
    if (RLMeshStatus status = rlmesh_value_as_tensor(ptr_, &out); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return Tensor(out);
  }

  Result<int64_t> as_discrete() const {
    int64_t out = 0;
    if (RLMeshStatus status = rlmesh_value_as_int(ptr_, &out); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// Borrowed UTF-8 bytes (NOT NUL-terminated), valid while the value lives.
  Result<std::string_view> as_text() const {
    RLMeshStr out{};
    if (RLMeshStatus status = rlmesh_value_as_text(ptr_, &out); status != RLMESH_OK)
      return Error::from_last(status);
    return std::string_view(out.data, out.len);
  }

  /// Compatibility reads for the tensor representation of array spaces.
  Result<size_t> array_len() const {
    auto t = as_tensor();
    if (!t) return t.error();
    return t->numel();
  }
  Result<std::vector<uint8_t>> as_multi_binary() const {
    auto t = as_tensor();
    if (!t) return t.error();
    if (!t->is_contiguous() ||
        (t->dtype() != RLMESH_DTYPE_INT8 && t->dtype() != RLMESH_DTYPE_UINT8 &&
         t->dtype() != RLMESH_DTYPE_BOOL))
      return Error(RLMESH_ERR_TYPE, "expected a contiguous byte tensor");
    auto p = t->as<uint8_t>();
    return std::vector<uint8_t>(p, p + t->numel());
  }
  Result<std::vector<int64_t>> as_multi_discrete() const {
    auto t = as_tensor();
    if (!t) return t.error();
    if (!t->is_contiguous() || t->dtype() != RLMESH_DTYPE_INT64)
      return Error(RLMESH_ERR_TYPE, "expected a contiguous int64 tensor");
    auto p = t->as<int64_t>();
    return std::vector<int64_t>(p, p + t->numel());
  }

  /// Child count of a Dict / Tuple value.
  Result<size_t> size() const {
    size_t out = 0;
    if (RLMeshStatus status = rlmesh_value_len(ptr_, &out); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// Tuple child `index`; nullopt when this is not a Tuple or `index` is absent.
  std::optional<ValueRef> at(size_t index) const {
    const RLMeshValue* child = nullptr;
    if (rlmesh_value_get(ptr_, index, &child) != RLMESH_OK) return std::nullopt;
    if (child == nullptr) return std::nullopt;
    return ValueRef(child);
  }

  /// Dict key `index`, in insertion order (borrowed, NOT NUL-terminated).
  Result<std::string_view> key(size_t index) const {
    RLMeshStr out{};
    if (RLMeshStatus status = rlmesh_value_dict_key_at(ptr_, index, &out); status != RLMESH_OK)
      return Error::from_last(status);
    return std::string_view(out.data ? out.data : "", out.len);
  }

  /// Dict child by key; nullopt when this is not a Dict or the key is absent.
  std::optional<ValueRef> get(const std::string& key) const {
    const RLMeshValue* child = nullptr;
    RLMeshStr name{key.data(), key.size()};
    if (rlmesh_value_dict_get(ptr_, name, &child) != RLMESH_OK) return std::nullopt;
    if (child == nullptr) return std::nullopt;
    return ValueRef(child);
  }

  /// Dict child `index`, in the same sorted key order as `key(index)`; nullopt
  /// when this is not a Dict or `index` is out of range.
  std::optional<ValueRef> at_key(size_t index) const {
    const RLMeshValue* child = nullptr;
    if (rlmesh_value_dict_value_at(ptr_, index, &child) != RLMESH_OK) return std::nullopt;
    if (child == nullptr) return std::nullopt;
    return ValueRef(child);
  }

  /// Every (key, child) of a Dict, in insertion order — NOT the declaration
  /// order `SpaceRef::items()` walks. Pair a value child with its space by key.
  Result<std::vector<std::pair<std::string_view, ValueRef>>> items() const {
    auto count = size();
    if (!count) return count.error();
    std::vector<std::pair<std::string_view, ValueRef>> out;
    out.reserve(*count);
    for (size_t i = 0; i < *count; ++i) {
      auto name = key(i);
      if (!name) return name.error();
      std::optional<ValueRef> child = at_key(i);
      if (!child) return Error(RLMESH_ERR_INTERNAL, "dict child missing for a reported key");
      out.emplace_back(*name, *child);
    }
    return out;
  }

 private:
  const RLMeshValue* ptr_;
};

/// An owned value (move-only). Every constructor returns a Result: a capi
/// constructor can fail, and the failure carries the reason.
class Value {
 public:
  explicit Value(RLMeshValue* ptr) : ptr_(ptr) {}
  Value(Value&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Value& operator=(Value&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;
  ~Value() { reset(); }

  const RLMeshValue* raw() const { return ptr_; }
  /// Give up ownership (the capi takes it back from a predict callback).
  RLMeshValue* release() {
    RLMeshValue* out = ptr_;
    ptr_ = nullptr;
    return out;
  }
  ValueRef ref() const { return ValueRef(ptr_); }

  static Result<Value> discrete(int64_t value) {
    RLMeshValue* out = nullptr;
    auto status = rlmesh_value_int(value, &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Value(out);
  }

  static Result<Value> text(std::string_view text) {
    RLMeshValue* out = nullptr;
    RLMeshStr view{text.data(), text.size()};
    auto status = rlmesh_value_text(view, &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Value(out);
  }

  /// Copy a contiguous row-major buffer into a Box value.
  static Result<Value> box(const void* data, RLMeshDType dtype, std::vector<int64_t> shape) {
    std::vector<size_t> dims;
    size_t nbytes = rlmesh_ext_dtype_size(dtype);
    for (int64_t dim : shape) {
      if (dim < 0) return Error(RLMESH_ERR_INVALID_ARGUMENT, "negative shape");
      dims.push_back(static_cast<size_t>(dim));
      nbytes *= static_cast<size_t>(dim);
    }
    RLMeshTensor* tensor = nullptr;
    auto status = rlmesh_tensor_new(data, nbytes, dims.data(), dims.size(), dtype, &tensor);
    if (status != RLMESH_OK) return Error::from_last(status);
    RLMeshValue* out = nullptr;
    status = rlmesh_value_tensor(tensor, &out);
    if (status != RLMESH_OK) {
      Error e = Error::from_last(status);
      rlmesh_tensor_free(tensor);
      return e;
    }
    rlmesh_tensor_free(tensor);
    return Value(out);
  }

  /// Box from typed elements; the dtype follows T (see `dtype_of`).
  template <class T>
  static Result<Value> box(const std::vector<T>& data, std::vector<int64_t> shape) {
    return box(data.data(), dtype_of<T>(), std::move(shape));
  }

  static Result<Value> multi_binary(const std::vector<uint8_t>& bits) {
    std::vector<uint8_t> normalized;
    for (uint8_t bit : bits) normalized.push_back(bit != 0);
    return box(normalized.data(), RLMESH_DTYPE_INT8, {static_cast<int64_t>(bits.size())});
  }

  static Result<Value> multi_discrete(const std::vector<int64_t>& values) {
    return box(values.data(), RLMESH_DTYPE_INT64, {static_cast<int64_t>(values.size())});
  }

  /// Composite constructors retain child references in ABI 5.
  static Result<Value> tuple(std::vector<Value> children) {
    std::vector<const RLMeshValue*> raw;
    for (auto& child : children) raw.push_back(child.raw());
    RLMeshValue* out = nullptr;
    auto status = rlmesh_value_tuple(raw.data(), raw.size(), &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Value(out);
  }
  static Result<Value> dict(std::vector<std::pair<std::string, Value>> entries) {
    std::vector<RLMeshStr> keys;
    std::vector<const RLMeshValue*> raw;
    for (auto& entry : entries) {
      keys.push_back({entry.first.data(), entry.first.size()});
      raw.push_back(entry.second.raw());
    }
    RLMeshValue* out = nullptr;
    auto status = rlmesh_value_dict(keys.data(), raw.data(), raw.size(), &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Value(out);
  }

 private:
  /// The capi's constructors return NULL, not a status, so the code comes from
  /// the thread-local slot the failing call just wrote.
  static Result<Value> adopt(RLMeshValue* ptr) {
    if (ptr == nullptr) return Error::from_last(rlmesh_last_error_status());
    return Value(ptr);
  }

  /// Free the children a failed composite constructor left with us. Snapshot
  /// the error first — the last-error slot is only good until the next capi
  /// call.
  static Error disown(const std::vector<RLMeshValue*>& children) {
    Error error = Error::from_last(rlmesh_last_error_status());
    for (RLMeshValue* child : children) rlmesh_value_free(child);
    return error;
  }

  void reset() {
    if (ptr_ != nullptr) {
      rlmesh_value_free(ptr_);
      ptr_ = nullptr;
    }
  }

  RLMeshValue* ptr_;
};

// ---- spaces -----------------------------------------------------------------

/// A borrowed space spec (valid while its contract lives) — the ValueRef of the
/// space side: every leaf parameter an encoder needs, plus a composite walk.
class SpaceRef {
 public:
  struct Bounds {
    double low;
    double high;
  };
  struct DiscreteRange {
    int64_t n;
    int64_t start;
  };
  struct TextLength {
    int64_t min;
    int64_t max;
  };

  explicit SpaceRef(const RLMeshSpaceSpec* ptr) : ptr_(ptr) {}

  const RLMeshSpaceSpec* raw() const { return ptr_; }
  bool valid() const { return ptr_ != nullptr; }
  RLMeshSpaceKind kind() const { return rlmesh_space_type(ptr_); }

  RLMeshDType dtype() const { return rlmesh_space_dtype(ptr_); }
  size_t ndim() const { return rlmesh_space_ndim(ptr_); }

  Result<std::vector<int64_t>> shape() const {
    std::vector<int64_t> out(ndim());
    if (RLMeshStatus status = rlmesh_space_copy_shape(ptr_, out.data(), out.size());
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// Element count of the shape — a Box space's numel, a MultiBinary's `n`.
  Result<size_t> numel() const {
    auto dims = shape();
    if (!dims) return dims.error();
    size_t n = 1;
    for (int64_t dim : *dims) n *= static_cast<size_t>(dim);
    return n;
  }

  /// Inclusive bounds of a Box space's `index`-th element (row-major); an
  /// undeclared bound reads as -inf / +inf.
  Result<Bounds> bounds(size_t index) const {
    Bounds out{0, 0};
    if (RLMeshStatus status = rlmesh_space_box_bounds(ptr_, index, &out.low, &out.high);
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// A Discrete space's category count and first value (`start ..= start+n-1`).
  Result<DiscreteRange> discrete() const {
    DiscreteRange out{0, 0};
    if (RLMeshStatus status = rlmesh_space_discrete_n(ptr_, &out.n, &out.start);
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// A Text space's length limits, in characters.
  Result<TextLength> text_length() const {
    TextLength out{0, 0};
    if (RLMeshStatus status = rlmesh_space_text_length(ptr_, &out.min, &out.max);
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// A Text space's allowed characters (borrowed, NOT NUL-terminated); empty
  /// means any character is allowed.
  Result<std::string_view> charset() const {
    const char* data = nullptr;
    size_t len = 0;
    if (RLMeshStatus status = rlmesh_space_text_charset(ptr_, &data, &len); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return std::string_view(data, len);
  }

  /// A MultiDiscrete space's per-element category counts (row-major).
  Result<std::vector<int64_t>> nvec() const {
    auto count = numel();
    if (!count) return count.error();
    std::vector<int64_t> out(*count);
    if (RLMeshStatus status = rlmesh_space_copy_nvec(ptr_, out.data(), out.size());
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// Child count of a Dict / Tuple space.
  Result<size_t> size() const {
    size_t out = 0;
    if (RLMeshStatus status = rlmesh_space_len(ptr_, &out); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// Tuple child `index`; nullopt when this is not a Tuple or `index` is absent.
  std::optional<SpaceRef> at(size_t index) const {
    const RLMeshSpaceSpec* child = rlmesh_space_tuple_get(ptr_, index);
    if (child == nullptr) return std::nullopt;
    return SpaceRef(child);
  }

  /// Dict key `index`, in declaration order (borrowed, NOT NUL-terminated).
  Result<std::string_view> key(size_t index) const {
    const char* data = nullptr;
    size_t len = 0;
    if (RLMeshStatus status = rlmesh_space_dict_key(ptr_, index, &data, &len);
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return std::string_view(data, len);
  }

  /// Dict child by key; nullopt when this is not a Dict or the key is absent.
  std::optional<SpaceRef> get(const std::string& key) const {
    const RLMeshSpaceSpec* child = rlmesh_space_dict_get(ptr_, key.c_str());
    if (child == nullptr) return std::nullopt;
    return SpaceRef(child);
  }

  /// Dict child `index`, in the same declaration order as `key(index)`; nullopt
  /// when this is not a Dict or `index` is out of range.
  std::optional<SpaceRef> at_key(size_t index) const {
    const RLMeshSpaceSpec* child = rlmesh_space_dict_get_at(ptr_, index);
    if (child == nullptr) return std::nullopt;
    return SpaceRef(child);
  }

  /// Every (key, child) of a Dict space, in declaration order (the order the
  /// wire encodes the dict's leaves in) — NOT the sorted order `ValueRef` uses.
  Result<std::vector<std::pair<std::string_view, SpaceRef>>> items() const {
    auto count = size();
    if (!count) return count.error();
    std::vector<std::pair<std::string_view, SpaceRef>> out;
    out.reserve(*count);
    for (size_t i = 0; i < *count; ++i) {
      auto name = key(i);
      if (!name) return name.error();
      std::optional<SpaceRef> child = at_key(i);
      if (!child) return Error(RLMESH_ERR_INTERNAL, "dict space child missing for a key");
      out.emplace_back(*name, *child);
    }
    return out;
  }

 private:
  const RLMeshSpaceSpec* ptr_;
};

namespace detail {

/// Store `scalar` into `dst` as `dtype`; false for a dtype we cannot encode.
inline bool store_scalar(uint8_t* dst, RLMeshDType dtype, double scalar) {
  auto put = [&](auto typed) {
    std::memcpy(dst, &typed, sizeof(typed));
    return true;
  };
  switch (dtype) {
    case RLMESH_DTYPE_INT8:
      return put(static_cast<int8_t>(scalar));
    case RLMESH_DTYPE_INT16:
      return put(static_cast<int16_t>(scalar));
    case RLMESH_DTYPE_INT32:
      return put(static_cast<int32_t>(scalar));
    case RLMESH_DTYPE_INT64:
      return put(static_cast<int64_t>(scalar));
    case RLMESH_DTYPE_UINT8:
      return put(static_cast<uint8_t>(scalar));
    case RLMESH_DTYPE_UINT16:
      return put(static_cast<uint16_t>(scalar));
    case RLMESH_DTYPE_UINT32:
      return put(static_cast<uint32_t>(scalar));
    case RLMESH_DTYPE_UINT64:
      return put(static_cast<uint64_t>(scalar));
    case RLMESH_DTYPE_FLOAT32:
      return put(static_cast<float>(scalar));
    case RLMESH_DTYPE_FLOAT64:
      return put(scalar);
    case RLMESH_DTYPE_BOOL:
      return put(static_cast<uint8_t>(scalar != 0.0));
    default:
      return false;
  }
}

}  // namespace detail

/// A neutral value for ANY space — the "do nothing" action a model can always
/// produce: Box zeros clamped into each element's bounds, Discrete `start`,
/// Text the shortest legal string (built from the charset when the space pins
/// one), MultiBinary / MultiDiscrete zeros — one per element of the shape, so
/// `n` for MultiBinary([n]) — and Dict / Tuple recursed.
inline Result<Value> zeros_for(SpaceRef space) {
  switch (space.kind()) {
    case RLMESH_SPACE_KIND_BOX: {
      RLMeshDType dtype = space.dtype();
      size_t stride = rlmesh_ext_dtype_size(dtype);
      if (stride == 0) return Error(RLMESH_ERR_TYPE, "unsupported Box dtype");
      auto shape = space.shape();
      if (!shape) return shape.error();
      size_t numel = 1;
      for (int64_t dim : *shape) numel *= static_cast<size_t>(dim);
      std::vector<uint8_t> bytes(numel * stride == 0 ? 1 : numel * stride, 0);
      for (size_t i = 0; i < numel; ++i) {
        auto range = space.bounds(i);
        if (!range) return range.error();
        double scalar = 0.0;
        if (range->low > scalar) scalar = range->low;
        if (range->high < scalar) scalar = range->high;
        if (scalar != 0.0 && !detail::store_scalar(bytes.data() + i * stride, dtype, scalar)) {
          return Error(RLMESH_ERR_TYPE, "unsupported Box dtype");
        }
      }
      return Value::box(bytes.data(), dtype, std::move(*shape));
    }
    case RLMESH_SPACE_KIND_DISCRETE: {
      auto range = space.discrete();
      if (!range) return range.error();
      return Value::discrete(range->start);
    }
    case RLMESH_SPACE_KIND_TEXT: {
      auto length = space.text_length();
      if (!length) return length.error();
      size_t min = length->min > 0 ? static_cast<size_t>(length->min) : 0;
      auto allowed = space.charset();
      if (!allowed) return allowed.error();
      // Lengths are counted in characters, so repeat one CHARACTER: the charset's
      // first (its bytes up to the next UTF-8 lead byte), or 'a' when the charset
      // is empty and every character is legal.
      std::string_view fill = "a";
      if (!allowed->empty()) {
        size_t bytes = 1;
        while (bytes < allowed->size() &&
               (static_cast<unsigned char>((*allowed)[bytes]) & 0xC0U) == 0x80U) {
          ++bytes;
        }
        fill = allowed->substr(0, bytes);
      }
      std::string text;
      text.reserve(min * fill.size());
      for (size_t i = 0; i < min; ++i) text.append(fill);
      return Value::text(text);
    }
    case RLMESH_SPACE_KIND_MULTI_BINARY: {
      auto count = space.numel();
      if (!count) return count.error();
      return Value::multi_binary(std::vector<uint8_t>(*count, 0));
    }
    case RLMESH_SPACE_KIND_MULTI_DISCRETE: {
      auto counts = space.nvec();
      if (!counts) return counts.error();
      return Value::multi_discrete(std::vector<int64_t>(counts->size(), 0));
    }
    case RLMESH_SPACE_KIND_TUPLE: {
      auto count = space.size();
      if (!count) return count.error();
      std::vector<Value> children;
      children.reserve(*count);
      for (size_t i = 0; i < *count; ++i) {
        std::optional<SpaceRef> child = space.at(i);
        if (!child) return Error(RLMESH_ERR_INTERNAL, "tuple space child missing");
        auto value = zeros_for(*child);
        if (!value) return value.error();
        children.push_back(std::move(*value));
      }
      return Value::tuple(std::move(children));
    }
    case RLMESH_SPACE_KIND_DICT: {
      auto fields = space.items();
      if (!fields) return fields.error();
      std::vector<std::pair<std::string, Value>> entries;
      entries.reserve(fields->size());
      for (auto& field : *fields) {
        auto value = zeros_for(field.second);
        if (!value) return value.error();
        entries.emplace_back(std::string(field.first), std::move(*value));
      }
      return Value::dict(std::move(entries));
    }
    case RLMESH_SPACE_KIND_UNKNOWN:
    default:
      return Error(RLMESH_ERR_INVALID_ARGUMENT, "no value fits an unspecified space");
  }
}

// ---- model ------------------------------------------------------------------

/// One row's episode identity. `id` is runtime-minted and never repeats, so a
/// stateful model keys per-episode state by it. `predict_index` is the
/// episode's re-plan ordinal (0 on its first predict, +1 per predict until the
/// episode ends) and `predict_seed` the reproducible per-predict seed derived
/// from the episode seed and that ordinal — nullopt for an unseeded episode.
struct Episode {
  std::string_view id;
  std::optional<int64_t> seed;
  uint64_t predict_index = 0;
  std::optional<int64_t> predict_seed;
};

/// What a batched predict receives: routing metadata plus one decoded
/// observation per sub-env, all borrowed for the duration of the call.
class Batch {
 public:
  explicit Batch(const RLMeshObservation* raw) : raw_(raw) {}

  /// Rows in this request (the route's `num_envs`).
  size_t size() const { return raw_->num_envs; }
  /// Whether the request carries observations at all — a contract with no
  /// observation space legitimately carries none.
  bool has_observations() const { return raw_->observations != nullptr; }

  /// Row `i`'s observation; nullopt when `i` is out of range or the row carries
  /// no observation. Never dereferences a null row.
  std::optional<ValueRef> at(size_t i) const {
    if (raw_->observations == nullptr || i >= raw_->num_envs) return std::nullopt;
    const RLMeshValue* row = raw_->observations[i];
    if (row == nullptr) return std::nullopt;
    return ValueRef(row);
  }

  /// Row `i`'s episode; an empty Episode when `i` is out of range.
  Episode episode(size_t i) const {
    if (raw_->episodes == nullptr || i >= raw_->num_envs) return Episode{};
    const RLMeshEpisode& row = raw_->episodes[i];
    return Episode{detail::sv(row.id), row.seeded ? std::optional<int64_t>(row.seed) : std::nullopt,
                   row.predict_index,
                   row.seeded ? std::optional<int64_t>(row.predict_seed) : std::nullopt};
  }

  std::string_view env_id() const { return detail::sv(raw_->env_id); }
  std::string_view session_id() const { return detail::sv(raw_->session_id); }
  std::string_view request_id() const { return detail::sv(raw_->request_id); }

  const RLMeshContract* contract() const { return raw_->contract; }
  SpaceRef action_space() const {
    return SpaceRef(raw_->contract ? rlmesh_contract_action_space(raw_->contract) : nullptr);
  }
  SpaceRef observation_space() const {
    return SpaceRef(raw_->contract ? rlmesh_contract_observation_space(raw_->contract) : nullptr);
  }

 private:
  const RLMeshObservation* raw_;
};

/// The single-env view a `Model::from_predict` policy receives: one
/// observation, its episode, and the route's spaces — no indexing.
class Request {
 public:
  explicit Request(const Batch& batch) : batch_(batch), episode_(batch.episode(0)) {}

  /// The observation; nullopt when this request carries none.
  std::optional<ValueRef> observation() const { return batch_.at(0); }
  const Episode& episode() const { return episode_; }

  SpaceRef action_space() const { return batch_.action_space(); }
  SpaceRef observation_space() const { return batch_.observation_space(); }
  std::string_view env_id() const { return batch_.env_id(); }
  std::string_view session_id() const { return batch_.session_id(); }
  std::string_view request_id() const { return batch_.request_id(); }
  /// The underlying batch, for a policy that wants the raw row.
  const Batch& batch() const { return batch_; }

 private:
  Batch batch_;
  Episode episode_;
};

/// Options for Model::run_local — the C RLMeshRunOptions, field for field.
/// Default-constructed == the C defaults (one episode, no caps).
struct RunOptions {
  /// The exact episode count, resolved like Python's run(episodes=...).
  /// 0 = unset: one episode, or one per seed when `episode_seeds` is given; a
  /// non-zero count with `episode_seeds` must match the seed count. There is no
  /// unbounded run, and 0 cannot mean "run nothing" (see rlmesh.h).
  uint64_t max_episodes = 0;
  /// Seed episode 0 with `base_seed`; later episodes derive from it.
  bool seeded = false;
  int64_t base_seed = 0;
  /// Truncate an episode after this many steps / this long (0 = unset).
  int64_t max_episode_steps = 0;
  double max_episode_seconds = 0;
  /// Actions per predicted chunk to execute (0/1 = no chunking).
  uint32_t execution_horizon = 0;
  /// Ask the env to close when the run ends.
  bool close_env = false;
  /// Explicit per-episode seeds (overriding `base_seed`), borrowed for the
  /// duration of the run_local call only.
  std::vector<int64_t> episode_seeds;
  /// Walk trial ordinals from `trial_index_base` (one per episode), delivered
  /// as `reset(options={"trial_index": k})` to an env that declares the option.
  bool trial_indexed = false;
  uint64_t trial_index_base = 0;
};

/// What a finished run reports (the C RLMeshRunReport).
struct RunReport {
  int64_t total_episodes = 0;
  int64_t total_steps = 0;
  double total_reward = 0;
  double mean_reward = 0;
  int64_t terminated_episodes = 0;
  int64_t truncated_episodes = 0;
};

/// Options for Model::serve — the C RLMeshServeOptions with owned/typed fields.
/// A zero duration or concurrency means "unset".
struct ServeOptions {
  /// Bearer token clients must present; empty disables auth.
  std::string token;
  bool allow_remote_shutdown = false;
  std::chrono::milliseconds idle_timeout{0};
  std::chrono::milliseconds drain_timeout{0};
  std::chrono::milliseconds close_timeout{0};
  size_t predict_concurrency = 0;
  /// Workflow edition this server declares (its WANT); empty declares none.
  /// Write the bare YYYY.MM base you authored against: it names the contract
  /// and selects whichever spelling of it both sides offer (a dev build's
  /// cohort included). A cohort spelling pins to that exact build. A value that
  /// admits nothing this build offers fails serve() with
  /// RLMESH_ERR_INVALID_ARGUMENT.
  std::string workflow_edition;
};

/// A model worker: bind a predict policy, then drive it against an environment
/// or serve it as a ModelService endpoint.
///
/// Callbacks run on a runtime worker thread, so whatever a policy captures must
/// be safe to use from a thread other than the one that built it, and a callback
/// must not re-enter its own Model — `cancel()` is the one exception.
class Model {
 public:
  /// Single-env policy: one Request -> one action Value.
  using PredictFn = std::function<Result<Value>(const Request&)>;
  /// Batched policy: the whole Batch -> one action per row (`batch.size()`).
  using PredictBatchFn = std::function<Result<std::vector<Value>>(const Batch&)>;
  /// Per-episode teardown; an empty `episode_id` means every episode of `env_id`.
  using EpisodeEndFn = std::function<void(std::string_view env_id, std::string_view episode_id)>;
  /// Shutdown hook, fired once.
  using CloseFn = std::function<void()>;

  static Result<Model> from_predict(PredictFn predict) {
    return create([predict = std::move(predict)](const Batch& batch) -> Result<std::vector<Value>> {
      if (batch.size() != 1) {
        return Error(RLMESH_ERR_MODEL, "from_predict is single-env; use from_predict_batch");
      }
      auto action = predict(Request(batch));
      if (!action) return action.error();
      std::vector<Value> out;
      out.push_back(std::move(*action));
      return out;
    });
  }

  static Result<Model> from_predict_batch(PredictBatchFn predict) {
    return create(std::move(predict));
  }

  /// Register an episode-teardown hook (optional; call before run/serve).
  Model& on_episode_end(EpisodeEndFn hook) {
    state_->on_episode_end = std::move(hook);
    return *this;
  }
  /// Register a shutdown hook (optional; call before run/serve).
  Model& on_close(CloseFn hook) {
    state_->on_close = std::move(hook);
    return *this;
  }

  // Move must null the source's raw handle (a default move only copies it,
  // double-freeing when the moved-from temporary is destroyed).
  Model(Model&& other) noexcept : model_(other.model_), state_(std::move(other.state_)) {
    other.model_ = nullptr;
  }
  Model& operator=(Model&& other) noexcept {
    if (this != &other) {
      if (model_ != nullptr) rlmesh_model_free(model_);
      model_ = other.model_;
      other.model_ = nullptr;
      state_ = std::move(other.state_);
    }
    return *this;
  }
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// Frees the handle, which closes the model: the on_close hook runs here
  /// unless a serve already ran it (run_local borrows the model and never
  /// closes it). NEVER destroy a Model from inside one of its own
  /// callbacks (predict / on_episode_end / on_close): that drops the runtime the
  /// callback is running on. Destroy it from a thread that is not executing it,
  /// after run_local / serve has returned.
  ~Model() {
    if (model_ != nullptr) rlmesh_model_free(model_);
  }

  /// Drive against a remote environment. Blocking; returns the run's report.
  Result<RunReport> run_local(std::string_view env_address, const RunOptions& options = {}) {
    std::string address(env_address);
    RLMeshRunOptions raw{};
    raw.max_episodes = options.max_episodes;
    raw.seeded = options.seeded;
    raw.base_seed = options.base_seed;
    raw.max_episode_steps = options.max_episode_steps;
    raw.max_episode_seconds = options.max_episode_seconds;
    raw.execution_horizon = options.execution_horizon;
    raw.close_env = options.close_env;
    raw.episode_seeds = options.episode_seeds.empty() ? nullptr : options.episode_seeds.data();
    raw.num_episode_seeds = options.episode_seeds.size();
    raw.trial_indexed = options.trial_indexed;
    raw.trial_index_base = options.trial_index_base;
    RLMeshRunReport report{};
    RLMeshStatus status = rlmesh_model_run_local(model_, address.c_str(), &raw, &report);
    if (status != RLMESH_OK) return Error::from_last(status);
    return RunReport{report.total_episodes, report.total_steps,         report.total_reward,
                     report.mean_reward,    report.terminated_episodes, report.truncated_episodes};
  }

  /// Serve as a ModelService endpoint. Blocking until a remote shutdown, an
  /// idle timeout, or `cancel()`.
  Status serve(std::string_view bind_address, const ServeOptions& options = {}) {
    std::string address(bind_address);
    RLMeshServeOptions raw{};
    raw.token = options.token.empty() ? nullptr : options.token.c_str();
    raw.allow_remote_shutdown = options.allow_remote_shutdown;
    raw.idle_timeout_ms = static_cast<uint64_t>(options.idle_timeout.count());
    raw.drain_timeout_ms = static_cast<uint64_t>(options.drain_timeout.count());
    raw.close_timeout_ms = static_cast<uint64_t>(options.close_timeout.count());
    raw.predict_concurrency = options.predict_concurrency;
    raw.workflow_edition =
        options.workflow_edition.empty() ? nullptr : options.workflow_edition.c_str();
    RLMeshStatus status = rlmesh_model_serve(model_, address.c_str(), &raw);
    if (status != RLMESH_OK) return Error::from_last(status);
    return ok();
  }

  /// Stop a blocking run_local / serve. Thread-safe and the one member callable
  /// while the model is running (including from inside a callback); terminal —
  /// a cancelled model refuses further runs.
  void cancel() { rlmesh_model_cancel(model_); }

 private:
  struct State {
    PredictBatchFn predict;
    EpisodeEndFn on_episode_end;
    CloseFn on_close;
  };

  Model(RLMeshModel* model, std::unique_ptr<State> state)
      : model_(model), state_(std::move(state)) {}

  static Result<Model> create(PredictBatchFn predict) {
    auto state = std::make_unique<State>();
    state->predict = std::move(predict);
    RLMeshModelVtable vtable{};
    vtable.struct_size = sizeof(RLMeshModelVtable);
    vtable.predict = &trampoline_predict;
    vtable.on_episode_end = &trampoline_episode_end;
    vtable.on_close = &trampoline_close;
    RLMeshModel* raw = nullptr;
    RLMeshStatus status = rlmesh_model_new(&vtable, state.get(), &raw);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Model(raw, std::move(state));
  }

  /// Report a declining callback through the capi's error channel. An Error
  /// carrying code 0 would read as success, so it becomes RLMESH_ERR_MODEL.
  static int fail(const Error& error) {
    rlmesh_callback_set_error(error.message().c_str(), error.is_recoverable());
    return static_cast<int>(error.code() == RLMESH_OK ? RLMESH_ERR_MODEL : error.code());
  }

  static int trampoline_predict(void* user_data, const RLMeshObservation* obs,
                                RLMeshValue** out_actions) noexcept {
    auto* state = static_cast<State*>(user_data);
    Batch batch(obs);
#if defined(__cpp_exceptions)
    try {
#endif
      Result<std::vector<Value>> actions = state->predict(batch);
      if (!actions) return fail(actions.error());
      if (actions->size() != batch.size()) {
        return fail(Error(RLMESH_ERR_MODEL, "predict returned " + std::to_string(actions->size()) +
                                                " actions for " + std::to_string(batch.size()) +
                                                " envs"));
      }
      for (size_t i = 0; i < actions->size(); ++i) out_actions[i] = (*actions)[i].release();
      return RLMESH_OK;
#if defined(__cpp_exceptions)
    } catch (const ModelException& error) {
      // Only a ModelException's own flag makes a model failure recoverable.
      return fail(
          Error(RLMESH_ERR_MODEL, detail::ExceptionText::raw(error), error.is_recoverable()));
    } catch (...) {
      return detail::record_current_exception(RLMESH_ERR_MODEL, "unknown C++ exception in predict");
    }
#endif
  }

  static void trampoline_episode_end(void* user_data, const char* env_id,
                                     const char* episode_id) noexcept {
    auto* state = static_cast<State*>(user_data);
    if (!state->on_episode_end) return;
#if defined(__cpp_exceptions)
    try {
#endif
      state->on_episode_end(detail::sv(env_id), detail::sv(episode_id));
#if defined(__cpp_exceptions)
    } catch (...) {
      // A teardown hook has no error channel; swallowing beats unwinding into Rust.
    }
#endif
  }

  static void trampoline_close(void* user_data) noexcept {
    auto* state = static_cast<State*>(user_data);
    if (!state->on_close) return;
#if defined(__cpp_exceptions)
    try {
#endif
      state->on_close();
#if defined(__cpp_exceptions)
    } catch (...) {
      // Same as on_episode_end: a shutdown hook has nowhere to report.
    }
#endif
  }

  RLMeshModel* model_ = nullptr;
  std::unique_ptr<State> state_;
};

// ---- spaces (env side) ------------------------------------------------------

/// An owned space spec (move-only), built env-side to declare an
/// `Environment`'s observation / action spaces. `ref()` reads it back.
class Space {
 public:
  explicit Space(RLMeshSpaceSpec* ptr) : ptr_(ptr) {}
  Space(Space&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Space& operator=(Space&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  Space(const Space&) = delete;
  Space& operator=(const Space&) = delete;
  ~Space() { reset(); }

  const RLMeshSpaceSpec* raw() const { return ptr_; }
  SpaceRef ref() const { return SpaceRef(ptr_); }
  /// Give up ownership (a composite builder adopts it).
  RLMeshSpaceSpec* release() {
    RLMeshSpaceSpec* out = ptr_;
    ptr_ = nullptr;
    return out;
  }

  /// A Box with one bound pair for every element. A float dtype takes
  /// +-INFINITY for an unbounded side; an integer one needs finite whole bounds
  /// the dtype can represent (out of range is an error, never clamped).
  static Result<Space> box(RLMeshDType dtype, const std::vector<int64_t>& shape, double low,
                           double high) {
    return adopt(rlmesh_space_box(dtype, shape.data(), shape.size(), low, high));
  }
  /// Box with the dtype following T (see `dtype_of`).
  template <class T>
  static Result<Space> box(const std::vector<int64_t>& shape, double low, double high) {
    return box(dtype_of<T>(), shape, low, high);
  }
  /// A Box with per-element bounds (`low` / `high` hold the element count each).
  static Result<Space> box(RLMeshDType dtype, const std::vector<int64_t>& shape,
                           const std::vector<double>& low, const std::vector<double>& high) {
    size_t numel = 1;
    for (int64_t dim : shape) numel *= static_cast<size_t>(dim < 0 ? 0 : dim);
    if (low.size() != numel || high.size() != numel) {
      return Error(RLMESH_ERR_INVALID_ARGUMENT, "Box bounds must hold one value per element");
    }
    return adopt(
        rlmesh_space_box_elementwise(dtype, shape.data(), shape.size(), low.data(), high.data()));
  }
  static Result<Space> discrete(int64_t n, int64_t start = 0) {
    return adopt(rlmesh_space_discrete(n, start));
  }
  static Result<Space> multi_binary(const std::vector<int64_t>& shape) {
    return adopt(rlmesh_space_multi_binary(shape.data(), shape.size()));
  }
  static Result<Space> multi_discrete(const std::vector<int64_t>& nvec) {
    return adopt(rlmesh_space_multi_discrete(nvec.data(), nvec.size()));
  }
  /// `charset` empty allows any character.
  static Result<Space> text(int64_t min_length, int64_t max_length,
                            const std::string& charset = {}) {
    return adopt(
        rlmesh_space_text(min_length, max_length, charset.empty() ? nullptr : charset.c_str()));
  }
  /// Dict from owned (key, space) entries, stored sorted by key; keys
  /// non-empty and unique.
  /// All-or-nothing ownership, as `Value::dict`.
  static Result<Space> dict(std::vector<std::pair<std::string, Space>> entries) {
    std::vector<const char*> keys;
    std::vector<RLMeshSpaceSpec*> raw;
    keys.reserve(entries.size());
    raw.reserve(entries.size());
    for (auto& entry : entries) {
      keys.push_back(entry.first.c_str());
      raw.push_back(entry.second.release());
    }
    RLMeshSpaceSpec* space = rlmesh_space_dict(keys.data(), raw.data(), raw.size());
    if (space == nullptr) return disown(raw);
    return Space(space);
  }
  static Result<Space> tuple(std::vector<Space> children) {
    std::vector<RLMeshSpaceSpec*> raw;
    raw.reserve(children.size());
    for (Space& child : children) raw.push_back(child.release());
    RLMeshSpaceSpec* space = rlmesh_space_tuple(raw.data(), raw.size());
    if (space == nullptr) return disown(raw);
    return Space(space);
  }

 private:
  static Result<Space> adopt(RLMeshSpaceSpec* ptr) {
    if (ptr == nullptr) return Error::from_last(rlmesh_last_error_status());
    return Space(ptr);
  }
  static Error disown(const std::vector<RLMeshSpaceSpec*>& children) {
    Error error = Error::from_last(rlmesh_last_error_status());
    for (RLMeshSpaceSpec* child : children) rlmesh_space_spec_free(child);
    return error;
  }
  void reset() {
    if (ptr_ != nullptr) {
      rlmesh_space_spec_free(ptr_);
      ptr_ = nullptr;
    }
  }

  RLMeshSpaceSpec* ptr_;
};

// ---- spaces (Python's rlmesh.spaces) ----------------------------------------

/// An element dtype: Python's dtype strings; C's `RLMeshDType` numbers.
enum class DType : int {
  Bool = RLMESH_DTYPE_BOOL,
  Uint8 = RLMESH_DTYPE_UINT8,
  Uint16 = RLMESH_DTYPE_UINT16,
  Uint32 = RLMESH_DTYPE_UINT32,
  Uint64 = RLMESH_DTYPE_UINT64,
  Int8 = RLMESH_DTYPE_INT8,
  Int16 = RLMESH_DTYPE_INT16,
  Int32 = RLMESH_DTYPE_INT32,
  Int64 = RLMESH_DTYPE_INT64,
  Float16 = RLMESH_DTYPE_FLOAT16,
  Float32 = RLMESH_DTYPE_FLOAT32,
  Float64 = RLMESH_DTYPE_FLOAT64,
};

/// Python's spelling ("float32"); empty for an unknown value.
inline std::string_view to_string(DType dtype) {
  return detail::sv(rlmesh_dtype_str(static_cast<RLMeshDType>(dtype)));
}

namespace spaces {

/// A space's kind (Python's `kind`).
enum class SpaceKind : int {
  Unknown = RLMESH_SPACE_KIND_UNKNOWN,
  Box = RLMESH_SPACE_KIND_BOX,
  Discrete = RLMESH_SPACE_KIND_DISCRETE,
  MultiBinary = RLMESH_SPACE_KIND_MULTI_BINARY,
  MultiDiscrete = RLMESH_SPACE_KIND_MULTI_DISCRETE,
  Text = RLMESH_SPACE_KIND_TEXT,
  Dict = RLMESH_SPACE_KIND_DICT,
  Tuple = RLMESH_SPACE_KIND_TUPLE,
};

/// How a Box stores its bounds (Python's `bounds_kind`).
enum class BoundsKind : int {
  Unbounded = RLMESH_BOUNDS_KIND_UNBOUNDED,
  Uniform = RLMESH_BOUNDS_KIND_UNIFORM,
  Elementwise = RLMESH_BOUNDS_KIND_ELEMENTWISE,
  TypedUniform = RLMESH_BOUNDS_KIND_TYPED_UNIFORM,
  TypedElementwise = RLMESH_BOUNDS_KIND_TYPED_ELEMENTWISE,
};

/// The C tag of a `Bound`.
enum class BoundType : int {
  Bool = RLMESH_BOUND_TYPE_BOOL,
  Int = RLMESH_BOUND_TYPE_INT,
  Uint = RLMESH_BOUND_TYPE_UINT,
  Float = RLMESH_BOUND_TYPE_FLOAT,
};

inline std::string_view to_string(SpaceKind kind) {
  return detail::sv(rlmesh_space_kind_str(static_cast<RLMeshSpaceKind>(kind)));
}
inline std::string_view to_string(BoundsKind kind) {
  return detail::sv(rlmesh_bounds_kind_str(static_cast<RLMeshBoundsKind>(kind)));
}

namespace detail {

[[noreturn]] inline void type_error(const char* what) {
#if defined(__cpp_exceptions)
  throw rlmesh::TypeError(what);
#else
  rlmesh::detail::die(what, Error(RLMESH_ERR_TYPE, what));
#endif
}

inline Error last(RLMeshStatus status) { return Error::from_last(status); }

inline std::string take_bytes(RLMeshBytes bytes) {
  std::string out(reinterpret_cast<const char*>(bytes.data), bytes.len);
  rlmesh_bytes_free(bytes);
  return out;
}

inline RLMeshStr str(std::string_view text) {
  RLMeshStr out;
  out.data = text.data();
  out.len = text.size();
  return out;
}

}  // namespace detail

/// One Box bound, exact in its domain: built implicitly from any arithmetic
/// value (`bool` is Bool, signed integers Int, unsigned Uint, floats Float).
class Bound {
 public:
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  Bound(T value) {  // NOLINT(google-explicit-constructor): Python takes a number
    if constexpr (std::is_same_v<T, bool>) {
      raw_ = rlmesh_bound_bool(value);
    } else if constexpr (std::is_floating_point_v<T>) {
      raw_ = rlmesh_bound_float(static_cast<double>(value));
    } else if constexpr (std::is_signed_v<T>) {
      raw_ = rlmesh_bound_int(static_cast<int64_t>(value));
    } else {
      raw_ = rlmesh_bound_uint(static_cast<uint64_t>(value));
    }
  }
  static Bound from_raw(RLMeshBound raw) { return Bound(raw, 0); }

  BoundType type() const { return static_cast<BoundType>(raw_.type_); }
  /// Each throws `rlmesh::TypeError` (aborts without exceptions) on another type.
  bool as_bool() const { return checked(BoundType::Bool).value.b; }
  int64_t as_int() const { return checked(BoundType::Int).value.i; }
  uint64_t as_uint() const { return checked(BoundType::Uint).value.u; }
  double as_float() const { return checked(BoundType::Float).value.f; }
  const RLMeshBound& raw() const { return raw_; }

 private:
  Bound(RLMeshBound raw, int) : raw_(raw) {}
  const RLMeshBound& checked(BoundType type) const {
    if (this->type() != type) detail::type_error("Bound holds another type");
    return raw_;
  }
  RLMeshBound raw_;
};

/// A Box's `low` / `high`: one bound for every element, or one per element.
using BoxBound = std::variant<Bound, std::vector<Bound>>;

/// `Box`'s keyword arguments.
struct BoxOptions {
  DType dtype = DType::Float32;
};

/// `Discrete`'s keyword arguments.
struct DiscreteOptions {
  int64_t start = 0;
  DType dtype = DType::Int64;
};

class Space;

/// Immutable data describing a space (Python's `SpaceSpec`).
class SpaceSpec {
 public:
  /// Copy a borrowed spec (a contract's, a space's, a child's).
  static SpaceSpec copy_of(const RLMeshSpaceSpec* spec) {
    return SpaceSpec(rlmesh_space_spec_clone(spec));
  }
  SpaceSpec(const SpaceSpec& other) : ptr_(rlmesh_space_spec_clone(other.ptr_)) {}
  SpaceSpec(SpaceSpec&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  SpaceSpec& operator=(SpaceSpec other) noexcept {
    std::swap(ptr_, other.ptr_);
    return *this;
  }
  ~SpaceSpec() { rlmesh_space_spec_free(ptr_); }

  const RLMeshSpaceSpec* raw() const { return ptr_; }
  SpaceKind kind() const { return static_cast<SpaceKind>(rlmesh_space_spec_kind(ptr_)); }
  std::vector<int64_t> shape() const {
    size_t len = 0;
    if (rlmesh_space_spec_shape(ptr_, nullptr, 0, &len) != RLMESH_OK) return {};
    std::vector<int64_t> out(len);
    if (rlmesh_space_spec_shape(ptr_, out.data(), out.size(), &len) != RLMESH_OK) return {};
    return out;
  }
  DType dtype() const { return static_cast<DType>(rlmesh_space_spec_dtype(ptr_)); }
  /// A fresh space with its own sampler.
  inline Result<Space> to_space() const;
  /// Gymnasium-style text, Python's `repr`.
  std::string to_string() const {
    RLMeshBytes bytes = {nullptr, 0, 0};
    if (rlmesh_space_spec_repr(ptr_, &bytes) != RLMESH_OK) return {};
    return detail::take_bytes(bytes);
  }

 private:
  explicit SpaceSpec(RLMeshSpaceSpec* ptr) : ptr_(ptr) {}
  RLMeshSpaceSpec* ptr_;
};

class Box;
class Discrete;
class Dict;

/// A space: a spec plus its sampler (Python's `Space`). Copies share the
/// handle, so two copies drive one RNG.
class Space {
 public:
  /// `Space(spec)`.
  static Result<Space> from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_space_new(spec.raw(), &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Space(out);
  }
  /// Adopt an owned handle.
  static Space adopt(RLMeshSpace* ptr) { return Space(ptr); }

  Space(const Space& other) : ptr_(rlmesh_space_clone(other.ptr_)) {}
  Space(Space&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Space& operator=(Space other) noexcept {
    std::swap(ptr_, other.ptr_);
    return *this;
  }
  ~Space() { rlmesh_space_free(ptr_); }

  const RLMeshSpace* raw() const { return ptr_; }
  SpaceSpec spec() const { return SpaceSpec::copy_of(rlmesh_space_spec(ptr_)); }
  /// A space converts to its spec (Dict's children are specs).
  operator SpaceSpec() const { return spec(); }  // NOLINT(google-explicit-constructor)
  SpaceKind kind() const { return static_cast<SpaceKind>(rlmesh_space_kind(ptr_)); }
  std::vector<int64_t> shape() const {
    size_t len = 0;
    if (rlmesh_space_shape(ptr_, nullptr, 0, &len) != RLMESH_OK) return {};
    std::vector<int64_t> out(len);
    if (rlmesh_space_shape(ptr_, out.data(), out.size(), &len) != RLMESH_OK) return {};
    return out;
  }
  DType dtype() const {
    return static_cast<DType>(rlmesh_space_spec_dtype(rlmesh_space_spec(ptr_)));
  }
  /// Restart the stream from `seed` (OS entropy when empty); returns the seed.
  uint64_t seed(std::optional<uint64_t> seed = std::nullopt) const {
    uint64_t used = 0;
    if (seed.has_value()) {
      uint64_t value = *seed;
      (void)rlmesh_space_seed(ptr_, &value, &used);
    } else {
      (void)rlmesh_space_seed(ptr_, nullptr, &used);
    }
    return used;
  }
  Result<Value> sample() const {
    RLMeshValue* out = nullptr;
    RLMeshStatus status = rlmesh_space_sample(ptr_, &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Value(out);
  }
  bool contains(const Value& value) const { return rlmesh_space_contains(ptr_, value.raw()); }
  bool contains(const ValueRef& value) const { return rlmesh_space_contains(ptr_, value.raw()); }
  bool operator==(const Space& other) const { return rlmesh_space_equals(ptr_, other.ptr_); }
  bool operator!=(const Space& other) const { return !(*this == other); }
  std::string to_string() const {
    RLMeshBytes bytes = {nullptr, 0, 0};
    if (rlmesh_space_repr(ptr_, &bytes) != RLMESH_OK) return {};
    return detail::take_bytes(bytes);
  }

  inline std::optional<Box> as_box() const;
  inline std::optional<Discrete> as_discrete() const;
  inline std::optional<Dict> as_dict() const;

 protected:
  explicit Space(RLMeshSpace* ptr) : ptr_(ptr) {}
  RLMeshSpace* ptr_;
};

inline Result<Space> SpaceSpec::to_space() const {
  RLMeshSpace* out = nullptr;
  RLMeshStatus status = rlmesh_space_spec_to_space(ptr_, &out);
  if (status != RLMESH_OK) return detail::last(status);
  return Space::adopt(out);
}

/// Python's `space_from_spec`: refuses a spec with no payload.
inline Result<Space> space_from_spec(const SpaceSpec& spec) {
  RLMeshSpace* out = nullptr;
  RLMeshStatus status = rlmesh_spaces_space_from_spec(spec.raw(), &out);
  if (status != RLMESH_OK) return detail::last(status);
  return Space::adopt(out);
}

namespace detail {
inline std::optional<BoxBound> read_bounds(const RLMeshSpace* box, bool high) {
  bool has = false;
  RLMeshBoundsKind kind = RLMESH_BOUNDS_KIND_INVALID;
  if (rlmesh_box_bounds_kind(box, &has, &kind) != RLMESH_OK || !has ||
      kind == RLMESH_BOUNDS_KIND_UNBOUNDED) {
    return std::nullopt;
  }
  auto read = high ? rlmesh_box_high : rlmesh_box_low;
  size_t len = 0;
  if (read(box, nullptr, 0, &len) != RLMESH_OK) return std::nullopt;
  std::vector<RLMeshBound> raw(len);
  if (read(box, raw.data(), raw.size(), &len) != RLMESH_OK) return std::nullopt;
  if (kind == RLMESH_BOUNDS_KIND_UNIFORM || kind == RLMESH_BOUNDS_KIND_TYPED_UNIFORM) {
    if (raw.empty()) return std::nullopt;
    return BoxBound(Bound::from_raw(raw[0]));
  }
  std::vector<Bound> out;
  out.reserve(raw.size());
  for (const RLMeshBound& bound : raw) out.push_back(Bound::from_raw(bound));
  return BoxBound(std::move(out));
}

inline RLMeshBoxOptions box_options(const BoxOptions& options) {
  RLMeshBoxOptions raw = RLMESH_BOX_OPTIONS_INIT;
  raw.dtype = static_cast<RLMeshDType>(options.dtype);
  return raw;
}
}  // namespace detail

/// A box of numbers in `[low, high]` (Python's `Box`).
class Box : public Space {
 public:
  /// `Box(low, high, shape, dtype)`.
  static Result<Box> create(Bound low, Bound high, const std::vector<int64_t>& shape,
                            BoxOptions options = {}) {
    RLMeshBoxOptions raw = detail::box_options(options);
    RLMeshSpace* out = nullptr;
    RLMeshStatus status =
        rlmesh_box_new(low.raw(), high.raw(), shape.data(), shape.size(), &raw, &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Box(out);
  }
  /// `Box(spec)`: another kind is a ValueError.
  static Result<Box> from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_box_from_spec(spec.raw(), &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Box(out);
  }
#if defined(__cpp_exceptions)
  /// Throwing forms of `create` / `from_spec`.
  Box(Bound low, Bound high, const std::vector<int64_t>& shape, BoxOptions options = {})
      : Box(create(low, high, shape, options).value_or_throw()) {}
  explicit Box(const SpaceSpec& spec) : Box(from_spec(spec).value_or_throw()) {}
#endif

  std::optional<BoxBound> low() const { return detail::read_bounds(ptr_, false); }
  std::optional<BoxBound> high() const { return detail::read_bounds(ptr_, true); }
  std::optional<BoundsKind> bounds_kind() const {
    bool has = false;
    RLMeshBoundsKind kind = RLMESH_BOUNDS_KIND_INVALID;
    if (rlmesh_box_bounds_kind(ptr_, &has, &kind) != RLMESH_OK || !has) return std::nullopt;
    return static_cast<BoundsKind>(kind);
  }

 private:
  friend class Space;
  explicit Box(RLMeshSpace* ptr) : Space(ptr) {}
};

/// The integers `start .. start + n` (Python's `Discrete`).
class Discrete : public Space {
 public:
  static Result<Discrete> create(int64_t n, DiscreteOptions options = {}) {
    RLMeshDiscreteOptions raw = RLMESH_DISCRETE_OPTIONS_INIT;
    raw.start = options.start;
    raw.dtype = static_cast<RLMeshDType>(options.dtype);
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_discrete_new(n, &raw, &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Discrete(out);
  }
  static Result<Discrete> from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_discrete_from_spec(spec.raw(), &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Discrete(out);
  }
#if defined(__cpp_exceptions)
  explicit Discrete(int64_t n, DiscreteOptions options = {})
      : Discrete(create(n, options).value_or_throw()) {}
  explicit Discrete(const SpaceSpec& spec) : Discrete(from_spec(spec).value_or_throw()) {}
#endif

  int64_t n() const {
    int64_t out = 0;
    (void)rlmesh_discrete_n(ptr_, &out);
    return out;
  }
  int64_t start() const {
    int64_t out = 0;
    (void)rlmesh_discrete_start(ptr_, &out);
    return out;
  }

 private:
  friend class Space;
  explicit Discrete(RLMeshSpace* ptr) : Space(ptr) {}
};

/// Named child spaces (Python's `Dict`), read in declaration order.
class Dict : public Space {
 public:
  /// `Dict(spaces)`: each spec is copied into a fresh child; keys sort.
  static Result<Dict> create(const std::vector<std::pair<std::string, SpaceSpec>>& spaces) {
    std::vector<RLMeshStr> keys;
    std::vector<const RLMeshSpaceSpec*> specs;
    keys.reserve(spaces.size());
    specs.reserve(spaces.size());
    for (const auto& entry : spaces) {
      keys.push_back(detail::str(entry.first));
      specs.push_back(entry.second.raw());
    }
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_dict_new(keys.data(), specs.data(), spaces.size(), &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Dict(out);
  }
  static Result<Dict> from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_dict_from_spec(spec.raw(), &out);
    if (status != RLMESH_OK) return detail::last(status);
    return Dict(out);
  }
#if defined(__cpp_exceptions)
  explicit Dict(const std::vector<std::pair<std::string, SpaceSpec>>& spaces)
      : Dict(create(spaces).value_or_throw()) {}
  explicit Dict(const SpaceSpec& spec) : Dict(from_spec(spec).value_or_throw()) {}
#endif

  size_t size() const {
    size_t out = 0;
    (void)rlmesh_dict_len(ptr_, &out);
    return out;
  }
  /// The `i`-th key, borrowed while any handle sharing the Dict lives.
  std::string_view key_at(size_t i) const {
    RLMeshStr out = {nullptr, 0};
    if (rlmesh_dict_key_at(ptr_, i, &out) != RLMESH_OK) return {};
    return std::string_view(out.data, out.len);
  }
  std::vector<std::string> keys() const {
    std::vector<std::string> out;
    for (size_t i = 0, n = size(); i < n; ++i) out.emplace_back(key_at(i));
    return out;
  }
  std::vector<Space> values() const {
    std::vector<Space> out;
    for (size_t i = 0, n = size(); i < n; ++i) {
      const RLMeshSpace* child = nullptr;
      if (rlmesh_dict_value_at(ptr_, i, &child) == RLMESH_OK) out.push_back(borrow(child));
    }
    return out;
  }
  std::vector<std::pair<std::string, Space>> items() const {
    std::vector<std::pair<std::string, Space>> out;
    std::vector<Space> children = values();
    for (size_t i = 0; i < children.size(); ++i) {
      out.emplace_back(std::string(key_at(i)), std::move(children[i]));
    }
    return out;
  }
  std::optional<Space> get(std::string_view key) const {
    const RLMeshSpace* child = nullptr;
    if (rlmesh_dict_get(ptr_, detail::str(key), &child) != RLMESH_OK || child == nullptr) {
      return std::nullopt;
    }
    return borrow(child);
  }
#if defined(__cpp_exceptions)
  /// `d[key]`; throws `std::out_of_range` when absent.
  Space operator[](std::string_view key) const {
    std::optional<Space> child = get(key);
    if (!child.has_value()) throw std::out_of_range("no Dict key " + std::string(key));
    return std::move(*child);
  }
#endif
  bool contains_key(std::string_view key) const {
    bool out = false;
    return rlmesh_dict_contains_key(ptr_, detail::str(key), &out) == RLMESH_OK && out;
  }

 private:
  friend class Space;
  explicit Dict(RLMeshSpace* ptr) : Space(ptr) {}
  /// A shared handle to a borrowed child (the same space on every read).
  static Space borrow(const RLMeshSpace* child) { return Space::adopt(rlmesh_space_clone(child)); }
};

inline std::optional<Box> Space::as_box() const {
  if (kind() != SpaceKind::Box) return std::nullopt;
  return Box(rlmesh_space_clone(ptr_));
}
inline std::optional<Discrete> Space::as_discrete() const {
  if (kind() != SpaceKind::Discrete) return std::nullopt;
  return Discrete(rlmesh_space_clone(ptr_));
}
inline std::optional<Dict> Space::as_dict() const {
  if (kind() != SpaceKind::Dict) return std::nullopt;
  return Dict(rlmesh_space_clone(ptr_));
}

}  // namespace spaces

namespace ext {

/// A Box with one bound pair per element (C++-only extra over
/// `rlmesh_ext_box_elementwise_new`).
inline Result<spaces::Box> box_elementwise(const std::vector<spaces::Bound>& low,
                                           const std::vector<spaces::Bound>& high,
                                           const std::vector<int64_t>& shape,
                                           spaces::BoxOptions options = {}) {
  if (low.size() != high.size()) {
    return Error(RLMESH_ERR_INVALID_ARGUMENT, "Box low and high need the same length");
  }
  std::vector<RLMeshBound> raw_low;
  std::vector<RLMeshBound> raw_high;
  for (const spaces::Bound& bound : low) raw_low.push_back(bound.raw());
  for (const spaces::Bound& bound : high) raw_high.push_back(bound.raw());
  RLMeshBoxOptions raw = spaces::detail::box_options(options);
  RLMeshSpace* out = nullptr;
  RLMeshStatus status = rlmesh_ext_box_elementwise_new(
      raw_low.data(), raw_high.data(), raw_low.size(), shape.data(), shape.size(), &raw, &out);
  if (status != RLMESH_OK) return spaces::detail::last(status);
  Result<spaces::Box> box =
      spaces::Box::from_spec(spaces::SpaceSpec::copy_of(rlmesh_space_spec(out)));
  rlmesh_space_free(out);
  return box;
}

}  // namespace ext

// ---- environment ------------------------------------------------------------

/// What `Environment::reset` receives.
struct ResetArgs {
  std::optional<int64_t> seed;
  /// Set only when the env declared the "trial_index" reset option.
  std::optional<int64_t> trial_index;
  /// Every reset option as a JSON object; empty when there are none.
  std::string_view options_json;
};

/// What `Environment::reset` returns. `info_json` is a JSON object or empty.
struct ResetOutput {
  Value observation;
  std::string info_json;
};

/// What `Environment::step` returns. `info_json` is a JSON object or empty.
struct StepOutput {
  Value observation;
  double reward = 0;
  bool terminated = false;
  bool truncated = false;
  std::string info_json;
};

/// A simulation served as one RLMesh env lane. Subclass it and hand it to
/// `EnvServer::create`. Every method runs on its lane's thread (not the thread
/// that built it), one call at a time; with `EnvConfig::foreground`, on the
/// thread that calls `EnvServer::serve` instead. An `Error` with `recoverable`
/// set fails only that request; any other error ends the client's session.
class Environment {
 public:
  virtual ~Environment() = default;
  virtual Result<ResetOutput> reset(const ResetArgs& args) = 0;
  /// `action` is borrowed for the call; nullopt when the request carries none.
  virtual Result<StepOutput> step(std::optional<ValueRef> action) = 0;
  /// A numeric image Value, cast to uint8 ([H, W, 3], [H, W, 4], [3, H, W],
  /// [4, H, W] or [H, W]; any other shape is no frame), or nullopt for no
  /// frame. Declare `render_mode = "rgb_array"` in the EnvConfig to advertise it.
  virtual Result<std::optional<Value>> render() { return std::optional<Value>(); }
  /// The server stopped, or the EnvServer was destroyed without serving
  /// (once).
  virtual void close() {}
};

/// What an env declares (the C RLMeshEnvConfig with owned fields). The spaces
/// are cloned at `EnvServer::create`, so this need not outlive it.
struct EnvConfig {
  std::string id;
  std::optional<Space> observation_space;
  std::optional<Space> action_space;
  /// Adapter EnvTags (the v1 JSON wire format); empty = untagged. Validated
  /// against the spaces at create(); grammar in docs/specs/env_tags.v1.md.
  std::string adapter_tags_json;
  /// Reset options the env understands, e.g. {"trial_index"}.
  std::vector<std::string> reset_options;
  /// "rgb_array" when `render()` produces frames.
  std::string render_mode;
  /// Extra contract metadata as a JSON object; empty = none.
  std::string metadata_json;
  /// Run every `Environment` method on the thread that calls
  /// `EnvServer::serve` (which then serves from a helper thread): for a
  /// simulation bound to the thread that created it (a GL / Vulkan context).
  /// One lane only.
  bool foreground = false;
  /// The named variants (scenarios) this image serves, as the describe
  /// envelope's `variants.catalog`: a JSON array of {"id", "params",
  /// "metadata"} with unique ids; empty = none.
  std::string variants_catalog_json;
  /// Host language, standard, and compiler detected by the header. Override
  /// with a runtime JSON object, or empty for the legacy C-only runtime.
  std::string host_runtime_json = RLMESH_HOST_RUNTIME_JSON;
};

/// Serves `Environment`s as an RLMesh EnvService endpoint:
/// `create` -> `bind` (learn the address) -> `serve` (blocks).
class EnvServer {
 public:
  static Result<EnvServer> create(std::unique_ptr<Environment> env, const EnvConfig& config) {
    std::vector<std::unique_ptr<Environment>> lanes;
    lanes.push_back(std::move(env));
    return create(std::move(lanes), config);
  }

  /// Serve each `Environment` as one lane of a `num_envs = lanes.size()`
  /// endpoint. Every lane shares `config` but runs on its own thread,
  /// concurrently with the others, so each must be an independent simulation.
  /// A foreground config takes exactly one lane.
  static Result<EnvServer> create(std::vector<std::unique_ptr<Environment>> lanes,
                                  const EnvConfig& config) {
    if (lanes.empty()) {
      return Error(RLMESH_ERR_INVALID_ARGUMENT, "EnvServer::create: no environments");
    }
    if (!config.observation_space || !config.action_space) {
      return Error(RLMESH_ERR_INVALID_ARGUMENT,
                   "EnvServer::create: observation_space and action_space are required");
    }
    std::vector<std::unique_ptr<State>> states;
    std::vector<void*> user_data;
    states.reserve(lanes.size());
    user_data.reserve(lanes.size());
    for (std::unique_ptr<Environment>& env : lanes) {
      if (!env) return Error(RLMESH_ERR_INVALID_ARGUMENT, "EnvServer::create: null environment");
      auto state = std::make_unique<State>();
      state->env = std::move(env);
      user_data.push_back(state.get());
      states.push_back(std::move(state));
    }
    std::vector<const char*> options;
    options.reserve(config.reset_options.size());
    for (const std::string& option : config.reset_options) options.push_back(option.c_str());
    auto opt = [](const std::string& text) { return text.empty() ? nullptr : text.c_str(); };

    RLMeshEnvConfig raw = rlmesh_env_config_init();
    raw.id = opt(config.id);
    raw.observation_space = config.observation_space->raw();
    raw.action_space = config.action_space->raw();
    raw.adapter_tags_json = opt(config.adapter_tags_json);
    raw.reset_options = options.empty() ? nullptr : options.data();
    raw.num_reset_options = options.size();
    raw.render_mode = opt(config.render_mode);
    raw.metadata_json = opt(config.metadata_json);
    raw.foreground = config.foreground;
    raw.variants_catalog_json = opt(config.variants_catalog_json);
    raw.host_runtime_json = opt(config.host_runtime_json);

    RLMeshEnvVtable vtable{};
    vtable.struct_size = sizeof(RLMeshEnvVtable);
    vtable.reset = &trampoline_reset;
    vtable.step = &trampoline_step;
    vtable.render = &trampoline_render;
    vtable.close = &trampoline_close;
    RLMeshEnv* handle = nullptr;
    RLMeshStatus status =
        rlmesh_env_new_lanes(&vtable, &raw, user_data.data(), user_data.size(), &handle);
    if (status != RLMESH_OK) return Error::from_last(status);
    return EnvServer(handle, std::move(states));
  }

  EnvServer(EnvServer&& other) noexcept : env_(other.env_), states_(std::move(other.states_)) {
    other.env_ = nullptr;
  }
  EnvServer& operator=(EnvServer&& other) noexcept {
    if (this != &other) {
      if (env_ != nullptr) rlmesh_env_free(env_);
      env_ = other.env_;
      other.env_ = nullptr;
      states_ = std::move(other.states_);
    }
    return *this;
  }
  EnvServer(const EnvServer&) = delete;
  EnvServer& operator=(const EnvServer&) = delete;
  /// Frees the handle; never from inside one of the env's own callbacks.
  ~EnvServer() {
    if (env_ != nullptr) rlmesh_env_free(env_);
  }

  /// Bind without serving; returns the resolved address (e.g. the port the OS
  /// picked for port 0). `predict_concurrency` and `close_timeout` do not
  /// apply to an env.
  Result<std::string> bind(std::string_view bind_address, const ServeOptions& options = {}) {
    std::string address(bind_address);
    RLMeshServeOptions raw{};
    raw.token = options.token.empty() ? nullptr : options.token.c_str();
    raw.allow_remote_shutdown = options.allow_remote_shutdown;
    raw.idle_timeout_ms = static_cast<uint64_t>(options.idle_timeout.count());
    raw.drain_timeout_ms = static_cast<uint64_t>(options.drain_timeout.count());
    raw.close_timeout_ms = static_cast<uint64_t>(options.close_timeout.count());
    raw.workflow_edition =
        options.workflow_edition.empty() ? nullptr : options.workflow_edition.c_str();
    RLMeshBytes resolved{};
    RLMeshStatus status = rlmesh_env_bind(env_, address.c_str(), &raw, &resolved);
    if (status != RLMESH_OK) return Error::from_last(status);
    std::string out(reinterpret_cast<const char*>(resolved.data), resolved.len);
    rlmesh_bytes_free(resolved);
    return out;
  }

  /// The env's describe envelope JSON (`rlmesh.describe.v1`): what to bake as
  /// the image's `dev.rlmesh.describe` label. After `bind` it declares the
  /// bound workflow edition; `bind` also publishes it on this endpoint's
  /// handshake.
  Result<std::string> describe_json() const {
    RLMeshBytes bytes{};
    RLMeshStatus status = rlmesh_env_describe_json(env_, &bytes);
    if (status != RLMESH_OK) return Error::from_last(status);
    std::string out(reinterpret_cast<const char*>(bytes.data), bytes.len);
    rlmesh_bytes_free(bytes);
    return out;
  }

  /// Serve until a remote shutdown, an idle timeout, or `cancel()`. Blocking;
  /// each lane's `Environment::close` runs before it returns. With
  /// `EnvConfig::foreground`, every `Environment` method runs on this thread.
  /// On return no `Environment` method is running or will run again: serve
  /// waits for an in-flight call (one the drain timeout gave up on) and for
  /// every `close`, however long it takes.
  Status serve() {
    RLMeshStatus status = rlmesh_env_serve(env_);
    if (status != RLMESH_OK) return Error::from_last(status);
    return ok();
  }

  /// Stop a blocking serve from any thread. Terminal.
  void cancel() { rlmesh_env_cancel(env_); }

 private:
  struct State {
    std::unique_ptr<Environment> env;
    // Borrowed by the capi until the next callback (the C info_json contract).
    std::string info_json;
  };

  EnvServer(RLMeshEnv* env, std::vector<std::unique_ptr<State>> states)
      : env_(env), states_(std::move(states)) {}

  static int fail(const Error& error) {
    rlmesh_callback_set_error(error.message().c_str(), error.is_recoverable());
    return static_cast<int>(error.code() == RLMESH_OK ? RLMESH_ERR_ENVIRONMENT : error.code());
  }

  static const char* keep_info(State* state, std::string info) {
    state->info_json = std::move(info);
    return state->info_json.empty() ? nullptr : state->info_json.c_str();
  }

  static int trampoline_reset(void* user_data, const RLMeshResetArgs* args,
                              RLMeshResetResult* out) noexcept {
    auto* state = static_cast<State*>(user_data);
    ResetArgs request;
    if (args->seeded) request.seed = args->seed;
    if (args->has_trial_index) request.trial_index = args->trial_index;
    request.options_json = detail::sv(args->options_json);
#if defined(__cpp_exceptions)
    try {
#endif
      Result<ResetOutput> result = state->env->reset(request);
      if (!result) return fail(result.error());
      out->observation = result->observation.release();
      out->info_json = keep_info(state, std::move(result->info_json));
      return RLMESH_OK;
#if defined(__cpp_exceptions)
    } catch (const EnvironmentException& error) {
      // Only an EnvironmentException's own flag makes an env failure recoverable.
      return fail(
          Error(RLMESH_ERR_ENVIRONMENT, detail::ExceptionText::raw(error), error.is_recoverable()));
    } catch (...) {
      return detail::record_current_exception(RLMESH_ERR_ENVIRONMENT,
                                              "unknown C++ exception in reset");
    }
#endif
  }

  static int trampoline_step(void* user_data, const RLMeshValue* action,
                             RLMeshStepResult* out) noexcept {
    auto* state = static_cast<State*>(user_data);
    std::optional<ValueRef> borrowed;
    if (action != nullptr) borrowed = ValueRef(action);
#if defined(__cpp_exceptions)
    try {
#endif
      Result<StepOutput> result = state->env->step(borrowed);
      if (!result) return fail(result.error());
      out->observation = result->observation.release();
      out->reward = result->reward;
      out->terminated = result->terminated;
      out->truncated = result->truncated;
      out->info_json = keep_info(state, std::move(result->info_json));
      return RLMESH_OK;
#if defined(__cpp_exceptions)
    } catch (const EnvironmentException& error) {
      // Only an EnvironmentException's own flag makes an env failure recoverable.
      return fail(
          Error(RLMESH_ERR_ENVIRONMENT, detail::ExceptionText::raw(error), error.is_recoverable()));
    } catch (...) {
      return detail::record_current_exception(RLMESH_ERR_ENVIRONMENT,
                                              "unknown C++ exception in step");
    }
#endif
  }

  static int trampoline_render(void* user_data, RLMeshValue** out_frame) noexcept {
    auto* state = static_cast<State*>(user_data);
#if defined(__cpp_exceptions)
    try {
#endif
      Result<std::optional<Value>> frame = state->env->render();
      if (!frame) return fail(frame.error());
      if (frame->has_value()) *out_frame = (*frame)->release();
      return RLMESH_OK;
#if defined(__cpp_exceptions)
    } catch (const EnvironmentException& error) {
      // Only an EnvironmentException's own flag makes an env failure recoverable.
      return fail(
          Error(RLMESH_ERR_ENVIRONMENT, detail::ExceptionText::raw(error), error.is_recoverable()));
    } catch (...) {
      return detail::record_current_exception(RLMESH_ERR_ENVIRONMENT,
                                              "unknown C++ exception in render");
    }
#endif
  }

  static void trampoline_close(void* user_data) noexcept {
    auto* state = static_cast<State*>(user_data);
#if defined(__cpp_exceptions)
    try {
#endif
      state->env->close();
#if defined(__cpp_exceptions)
    } catch (...) {
      // A shutdown hook has nowhere to report.
    }
#endif
  }

  RLMeshEnv* env_ = nullptr;
  std::vector<std::unique_ptr<State>> states_;  // one per lane
};

}  // namespace rlmesh

#endif  // RLMESH_HPP
