// RLMesh C++17 API. Header-only projection of C ABI 5.
// RLMesh uses the same nouns and constructs in every language, following each
// language's conventions.
// Copies of shared handles refer to the same library object. Call close() to
// observe cleanup errors; destructors release ownership without throwing.
#ifndef RLMESH_HPP
#define RLMESH_HPP
#if (defined(_MSVC_LANG) ? _MSVC_LANG : __cplusplus) < 201703L
#error "rlmesh.hpp requires C++17"
#endif
#if !defined(__cpp_exceptions) && !defined(_CPPUNWIND)
#error "rlmesh.hpp requires exceptions"
#endif
// Host headers (Unreal, X11, Windows) may define these common macro names.
// Save and undefine them through our includes and declarations, then restore
// them below. Callers must undefine object-like macros such as None or Bool
// themselves before spelling the matching scoped enumerators in their code.
#pragma push_macro("check")
#undef check
#pragma push_macro("verify")
#undef verify
#pragma push_macro("ensure")
#undef ensure
#pragma push_macro("checkf")
#undef checkf
#pragma push_macro("TEXT")
#undef TEXT
#pragma push_macro("None")
#undef None
#pragma push_macro("Bool")
#undef Bool
#pragma push_macro("True")
#undef True
#pragma push_macro("False")
#undef False
#pragma push_macro("Status")
#undef Status
#pragma push_macro("Success")
#undef Success
#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max
#pragma push_macro("ERROR")
#undef ERROR
#pragma push_macro("IN")
#undef IN
#pragma push_macro("OUT")
#undef OUT
#pragma push_macro("DELETE")
#undef DELETE
#include <algorithm>
#include <any>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "rlmesh.h"
namespace rlmesh {
/// A borrowed contiguous view. The caller keeps the underlying storage alive.
/// This type has the same representation in C++17 and C++20.
template <class T>
class Span;
namespace detail {
template <class T>
struct IsSpan : std::false_type {};
template <class T>
struct IsSpan<Span<T>> : std::true_type {};
}  // namespace detail
template <class T>
class Span {
  T* data_ = nullptr;
  size_t size_ = 0;

 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using iterator = T*;
  constexpr Span() noexcept = default;
  constexpr Span(T* data, size_t size) noexcept : data_(data), size_(size) {}
  template <class U, std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>, int> = 0>
  constexpr Span(Span<U> view) noexcept : data_(view.data()), size_(view.size()) {}
  template <class R,
            std::enable_if_t<!detail::IsSpan<std::remove_cv_t<std::remove_reference_t<R>>>::value &&
                                 std::is_convertible_v<decltype(std::data(std::declval<R&>())), T*>,
                             int> = 0,
            class = decltype(std::size(std::declval<R&>()))>
  constexpr Span(R&& values) noexcept(noexcept(std::data(values)) && noexcept(std::size(values)))
      : data_(std::data(values)), size_(std::size(values)) {}
  constexpr T* data() const noexcept { return data_; }
  constexpr size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return size_ ? data_ + size_ : data_; }
  constexpr T& operator[](size_t index) const noexcept { return data_[index]; }
};

}  // namespace rlmesh
#if defined(__cpp_lib_ranges)
// The only standard-dependent declarations: these std templates exist in C++20.
namespace std::ranges {
template <class T>
inline constexpr bool enable_borrowed_range<rlmesh::Span<T>> = true;
template <class T>
inline constexpr bool enable_view<rlmesh::Span<T>> = true;
}  // namespace std::ranges
#endif
namespace rlmesh {

namespace detail {
template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
template <template <class> class Expression, class T, class = void>
struct IsDetected : std::false_type {};
template <template <class> class Expression, class T>
struct IsDetected<Expression, T, std::void_t<Expression<T>>> : std::true_type {};
template <template <class> class Expression, class T>
inline constexpr bool is_detected = IsDetected<Expression, T>::value;
template <class T>
inline constexpr bool value_integer =
    std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, char> &&
    !std::is_same_v<T, wchar_t> && !std::is_same_v<T, char16_t> && !std::is_same_v<T, char32_t> &&
    !std::is_same_v<T, decltype(u8'a')>;
// Compare member types without accessing an incomplete enclosing class in
// a hidden friend's exception specification (also supported by libclang).
template <class... T>
inline constexpr bool nothrow_equal =
    (noexcept(std::declval<const T&>() == std::declval<const T&>()) && ...);
template <class T>
struct IsPolicyPointer : std::false_type {};
template <class T>
struct IsPolicyPointer<std::shared_ptr<T>> : std::true_type {};
template <class T>
struct IsPolicyPointer<std::unique_ptr<T>> : std::true_type {};
}  // namespace detail

/// An environment error's code (`EnvironmentException.code`).
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

/// A model error's code (`ModelException.code`).
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
/// constructors take a message and optional recoverability.
struct FromLibrary {};
/// The wrapper's read of an authored exception's raw message (its `what()`
/// carries the `[CODE]` prefix the wire adds again).
struct ExceptionText;
}  // namespace detail

/// `RuntimeError`: misuse and local refusals.
class RuntimeError : public std::runtime_error {
 public:
  explicit RuntimeError(const std::string& message) : std::runtime_error(message) {}
};

/// `RLMeshException`: an internal or server-side failure. The base of the
/// rlmesh-specific classes; the general error classes (`ValueError`, `OSError`,
/// ...) are not under it.
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
/// authored one's `what()` is `[INTERNAL] message`.
class EnvironmentException : public RLMeshException {
 public:
  explicit EnvironmentException(std::string message)
      : EnvironmentException(std::move(message), false, detail::FromLibrary{}) {}
  EnvironmentException(std::string message, ErrorCode code, bool is_recoverable,
                       detail::FromLibrary)
      : RLMeshException(message),
        code_(code),
        recoverable_(is_recoverable),
        raw_(message.substr(message.find("] ") == std::string::npos ? 0 : message.find("] ") + 2)) {
  }

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
  ErrorCode code_{};
  bool recoverable_{};
  std::string raw_{};
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

/// `ModelException`: the model reported a failure. Throw `ModelException(msg,
/// true)` from predict to fail one request as recoverable (code `INTERNAL`);
/// its `what()` is
/// `[INTERNAL] msg`.
class ModelException : public RLMeshException {
 public:
  explicit ModelException(std::string message, bool is_recoverable = false)
      : RLMeshException("[INTERNAL] " + message),
        code_(ModelErrorCode::Internal),
        recoverable_(is_recoverable),
        raw_(std::move(message)) {}
  ModelException(std::string message, ModelErrorCode code, bool is_recoverable, detail::FromLibrary)
      : RLMeshException(message),
        code_(code),
        recoverable_(is_recoverable),
        raw_(message.substr(message.find("] ") == std::string::npos ? 0 : message.find("] ") + 2)) {
  }

  ModelErrorCode code() const { return code_; }
  bool is_recoverable() const { return recoverable_; }

 private:
  friend struct detail::ExceptionText;
  ModelErrorCode code_{};
  bool recoverable_{};
  std::string raw_{};
};

namespace detail {
struct ExceptionText {
  static const std::string& raw(const EnvironmentException& error) { return error.raw_; }
  static const std::string& raw(const ModelException& error) { return error.raw_; }
};
}  // namespace detail

/// `ValueError`: an invalid argument value, address or options field.
class ValueError : public std::invalid_argument {
 public:
  explicit ValueError(const std::string& message) : std::invalid_argument(message) {}
};

/// `AdapterResolutionError`: adapters cannot bridge env and model.
class AdapterResolutionError : public ValueError {
 public:
  explicit AdapterResolutionError(const std::string& message) : ValueError(message) {}
};

/// `TypeError`: the wrong kind of value or object.
class TypeError : public std::logic_error {
 public:
  explicit TypeError(const std::string& message) : std::logic_error(message) {}
};

/// `OSError`: a filesystem failure.
class OSError : public std::runtime_error {
 public:
  explicit OSError(const std::string& message) : std::runtime_error(message) {}
};

/// `ConnectionError` (an `OSError`): a transport failure.
class ConnectionError : public OSError {
 public:
  explicit ConnectionError(const std::string& message) : OSError(message) {}
};

/// `TimeoutError` (an `OSError`): a connect or request deadline.
class TimeoutError : public OSError {
 public:
  explicit TimeoutError(const std::string& message) : OSError(message) {}
};

namespace ext {

/// Cancellation (the `KeyboardInterrupt` error class). Derives from no `std::`
/// exception, so `catch (const std::exception&)` does not swallow a cancel.
class CancelledError {
 public:
  explicit CancelledError(std::string message = "cancelled") : message_(std::move(message)) {}
  const char* what() const noexcept { return message_.c_str(); }

 private:
  std::string message_{};
};

}  // namespace ext

namespace detail {
inline std::string_view sv(const char* s) { return s ? std::string_view(s) : std::string_view{}; }
inline RLMeshStr str(std::string_view s) { return {s.data(), s.size()}; }
inline const char* ptr(const std::optional<std::string>& s) { return s ? s->c_str() : nullptr; }
template <class T>
const T* ptr(const std::optional<T>& v) {
  return v ? &*v : nullptr;
}
// Hidden visibility keeps this tag local to each shared-library/runtime domain.
#if defined(__GNUC__) || defined(__clang__)
[[gnu::visibility("hidden")]]
#endif
inline const unsigned char exception_payload_tag{};
#if defined(__GNUC__) || defined(__clang__)
struct [[gnu::visibility("hidden")]] ExceptionPayload {
#else
struct ExceptionPayload {
#endif
  static constexpr uint64_t MAGIC = 0x524c4d4350503230ULL;
  uint64_t magic = MAGIC;
  const void* tag = &exception_payload_tag;
  std::exception_ptr exception;
  explicit ExceptionPayload(std::exception_ptr e) noexcept : exception(std::move(e)) {}
};
struct PayloadOwner {
  void* pointer;
  void (*free)(void*);
  ~PayloadOwner() {
    if (pointer && free) free(pointer);
  }
};
#if defined(__GNUC__) || defined(__clang__)
[[gnu::visibility("hidden")]]
#endif
inline void free_exception_payload(void* p) noexcept {
  delete static_cast<ExceptionPayload*>(p);
}
#if defined(__GNUC__) || defined(__clang__)
[[gnu::visibility("hidden")]]
#endif
inline void check_status(RLMeshStatus status) {
  if (status == RLMESH_OK) return;
  const auto cls = rlmesh_last_error_class();
  const std::string message(sv(rlmesh_last_error_message()));
  const bool recoverable = rlmesh_last_error_is_recoverable() != 0;
  int32_t code = RLMESH_ERROR_CODE_INTERNAL;
  (void)rlmesh_last_error_code(&code);
  void* payload = nullptr;
  void (*free)(void*) = nullptr;
  const void* tag = nullptr;
  if (rlmesh_ext_last_error_take_payload_owned(&payload, &free, &tag) == RLMESH_OK) {
    PayloadOwner owned{payload, free};
    // Never inspect foreign payload bytes, even if their allocation is smaller
    // than this envelope. Their originating owner's deleter handles disposal.
    if (payload && tag == &exception_payload_tag) {
      const auto* envelope = static_cast<const ExceptionPayload*>(payload);
      if (envelope->magic == ExceptionPayload::MAGIC && envelope->tag == &exception_payload_tag &&
          envelope->exception)
        std::rethrow_exception(envelope->exception);
    }
  }
  switch (cls) {
    case RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION:
      throw ProtocolException(message);
    case RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION:
      throw EnvironmentException(message, static_cast<ErrorCode>(code), recoverable, FromLibrary{});
    case RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION:
      throw RecoverableEnvironmentException(message, static_cast<ErrorCode>(code), FromLibrary{});
    case RLMESH_ERROR_CLASS_MODEL_EXCEPTION:
      throw ModelException(message, static_cast<ModelErrorCode>(code), recoverable, FromLibrary{});
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
    default:
      throw RLMeshException(message);
  }
}
#if defined(__GNUC__) || defined(__clang__)
[[gnu::visibility("hidden")]]
#endif
inline RLMeshStatus record_exception(RLMeshStatus kind = RLMESH_ERR_INTERNAL) noexcept {
  const auto exception = std::current_exception();
  try {
    std::rethrow_exception(exception);
  } catch (const EnvironmentException& e) {
    rlmesh_callback_set_error_class(e.is_recoverable()
                                        ? RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION
                                        : RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION,
                                    ExceptionText::raw(e).c_str());
  } catch (const ModelException& e) {
    if (kind == RLMESH_ERR_MODEL)
      rlmesh_callback_set_error(ExceptionText::raw(e).c_str(), e.is_recoverable());
    else
      rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_MODEL_EXCEPTION,
                                      ExceptionText::raw(e).c_str());
  } catch (const ProtocolException& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION, e.what());
  } catch (const RLMeshException& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_RLMESH_EXCEPTION, e.what());
  } catch (const AdapterResolutionError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR, e.what());
  } catch (const ValueError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_VALUE_ERROR, e.what());
  } catch (const TypeError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TYPE_ERROR, e.what());
  } catch (const RuntimeError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_RUNTIME_ERROR, e.what());
  } catch (const TimeoutError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TIMEOUT_ERROR, e.what());
  } catch (const ConnectionError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_CONNECTION_ERROR, e.what());
  } catch (const OSError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_OS_ERROR, e.what());
  } catch (const ext::CancelledError& e) {
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT, e.what());
  } catch (const std::exception& e) {
    rlmesh_callback_set_error(e.what(), false);
  } catch (...) {
    rlmesh_callback_set_error("unknown C++ exception", false);
  }
  // Allocation failure must not escape the C boundary either.
  try {
    auto payload = std::make_unique<ExceptionPayload>(exception);
    rlmesh_ext_callback_set_tagged_error_payload(payload.get(), free_exception_payload,
                                                 &exception_payload_tag);
    payload.release();
  } catch (...) {
  }
  return kind;
}
template <class F>
RLMeshStatus callback(F&& f, RLMeshStatus kind = RLMESH_ERR_INTERNAL) noexcept {
  try {
    std::forward<F>(f)();
    return RLMESH_OK;
  } catch (...) {
    return record_exception(kind);
  }
}
template <class T, auto Free>
using Owned = std::unique_ptr<T, decltype(Free)>;
template <class T, auto Free>
Owned<T, Free> own(T* p) {
  return Owned<T, Free>(p, Free);
}
struct Bytes {
  RLMeshBytes value{};
  ~Bytes() { rlmesh_bytes_free(value); }
  std::string text() const {
    return value.len ? std::string(reinterpret_cast<const char*>(value.data), value.len)
                     : std::string{};
  }
};
template <class F, class... A>
std::string text(F f, A... a) {
  Bytes out{};
  check_status(f(a..., &out.value));
  return out.text();
}
inline std::string take_bytes(RLMeshBytes bytes) {
  Bytes owned{bytes};
  return owned.text();
}
template <class T>
void delete_state(void* p) {
  delete static_cast<T*>(p);
}
}  // namespace detail
inline constexpr std::string_view VERSION = RLMESH_VERSION;

/// The describe envelope's metadata key and schema version, and the env reset
/// options key.
inline constexpr std::string_view DESCRIBE_METADATA_KEY = RLMESH_DESCRIBE_METADATA_KEY;
inline constexpr int64_t DESCRIBE_SCHEMA_VERSION = RLMESH_DESCRIBE_SCHEMA_VERSION;
inline constexpr std::string_view ENV_RESET_OPTIONS_KEY = RLMESH_ENV_RESET_OPTIONS_KEY;

/// Where a build's cohort came from.
enum class BuildSource : int32_t {
  Release = RLMESH_BUILD_SOURCE_RELEASE,
  Package = RLMESH_BUILD_SOURCE_PACKAGE,
  Git = RLMESH_BUILD_SOURCE_GIT,
};

/// The canonical spelling (`"release"`).
inline std::string_view to_string(BuildSource source) {
  return detail::sv(rlmesh_build_source_str(static_cast<RLMeshBuildSource>(source)));
}

/// The linked library's identity. Every view is static, with process lifetime.
struct BuildInfo {
  std::string_view version{};
  std::string_view protocol_generation{};
  std::string_view workflow_edition{};
  std::string_view workflow_edition_base{};
  std::vector<std::string_view> supported_workflow_editions{};  ///< newest first
  std::string_view build_cohort{};
  BuildSource build_source = BuildSource::Package;
  std::optional<std::string_view> git{};
};

/// This library's build identity (`rlmesh.build_info()`).
inline BuildInfo build_info() {
  RLMeshBuildInfo raw = RLMESH_BUILD_INFO_INIT;
  RLMeshStatus status = rlmesh_build_info(&raw);
  detail::check_status(status);
  BuildInfo info{};
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

/// The build's text representation (`BuildInfo(version="0.1.0", ...)`), as the
/// library spells it. `info` is a `build_info()` result.
inline std::string to_string(const BuildInfo& info) {
  std::ostringstream out;
  out << "BuildInfo(version=" << std::quoted(std::string(info.version))
      << ", protocol_generation=" << std::quoted(std::string(info.protocol_generation))
      << ", workflow_edition=" << std::quoted(std::string(info.workflow_edition))
      << ", workflow_edition_base=" << std::quoted(std::string(info.workflow_edition_base))
      << ", build_cohort=" << std::quoted(std::string(info.build_cohort))
      << ", build_source=" << std::quoted(std::string(to_string(info.build_source))) << ", git=";
  if (info.git)
    out << std::quoted(std::string(*info.git));
  else
    out << "None";
  return out.str() + ")";
}

template <class Stream>
auto operator<<(Stream& out, const BuildInfo& info) -> decltype(out << std::string()) {
  return out << to_string(info);
}

/// The bare `YYYY.MM` workflow edition this build runs at: the value to
/// declare.
inline std::string_view current_workflow_edition() {
  return detail::sv(rlmesh_current_workflow_edition());
}

/// A warning's category (values of the C enum).
enum class WarningCategory : int32_t {
  UserWarning = RLMESH_WARNING_CATEGORY_USER_WARNING,
  RuntimeWarning = RLMESH_WARNING_CATEGORY_RUNTIME_WARNING,
  WorkflowEditionWarning = RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING,
};

/// The class name (`"WorkflowEditionWarning"`).
inline std::string_view name(WarningCategory category) {
  return detail::sv(rlmesh_warning_category_name(static_cast<RLMeshWarningCategory>(category)));
}

/// Test class inheritance, including identity.
inline bool is_subclass_of(WarningCategory cls, WarningCategory base) {
  return rlmesh_warning_category_is_subclass(static_cast<RLMeshWarningCategory>(cls),
                                             static_cast<RLMeshWarningCategory>(base));
}

/// One warning.
struct Warning {
  WarningCategory category{};
  std::string message{};

  friend bool operator==(const Warning& self, const Warning& other) noexcept(
      detail::nothrow_equal<decltype(category), decltype(message)>) {
    return self.category == other.category && self.message == other.message;
  }
  friend bool operator!=(const Warning& self,
                         const Warning& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};

/// `<Category>: <message>`.
template <class Stream>
auto operator<<(Stream& out, const Warning& warning) -> decltype(out << std::string()) {
  return out << std::string(name(warning.category)) + ": " + warning.message;
}

namespace detail {

using WarningHandler = std::function<void(const Warning&)>;

inline RLMeshStatus warning_trampoline(void* user_data, RLMeshWarningCategory category,
                                       const char* message) noexcept {
  auto* handler = static_cast<WarningHandler*>(user_data);
  try {
    (*handler)(Warning{static_cast<WarningCategory>(category), std::string(sv(message))});
  } catch (...) {
    return static_cast<RLMeshStatus>(record_exception(RLMESH_ERR_INTERNAL));
  }

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
    detail::check_status(status);
  }
}

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

/// The canonical spelling ("float32"); empty for an unknown value.
inline std::string_view to_string(DType dtype) {
  return detail::sv(rlmesh_dtype_str(static_cast<RLMeshDType>(dtype)));
}

enum class Device { Cpu = RLMESH_DEVICE_CPU };
inline std::string_view to_string(Device d) {
  return detail::sv(rlmesh_device_str(static_cast<RLMeshDevice>(d)));
}
namespace ext {
struct DlpackDeleter {
  void operator()(DLManagedTensorVersioned* p) const noexcept {
    if (p && p->deleter) p->deleter(p);
  }
};
using DlpackPtr = std::unique_ptr<DLManagedTensorVersioned, DlpackDeleter>;
class CancelToken {
  detail::Owned<RLMeshExtCancelToken, rlmesh_ext_cancel_token_free> handle_{
      nullptr, rlmesh_ext_cancel_token_free};

 public:
  CancelToken() {
    RLMeshExtCancelToken* p = nullptr;
    detail::check_status(rlmesh_ext_cancel_token_new(&p));
    handle_.reset(p);
  }
  explicit CancelToken(RLMeshExtCancelToken* p) : handle_(p, rlmesh_ext_cancel_token_free) {}
  CancelToken(const CancelToken& other) {
    if (other.raw()) {
      RLMeshExtCancelToken* p = nullptr;
      detail::check_status(rlmesh_ext_cancel_token_clone(other.raw(), &p));
      handle_.reset(p);
    }
  }
  CancelToken& operator=(const CancelToken& other) {
    if (this != &other) {
      CancelToken copy(other);
      handle_.swap(copy.handle_);
    }
    return *this;
  }
  CancelToken(CancelToken&&) noexcept = default;
  CancelToken& operator=(CancelToken&&) noexcept = default;
  RLMeshExtCancelToken* raw() const { return handle_.get(); }
  void cancel() const { detail::check_status(rlmesh_ext_cancel_token_cancel(raw())); }
};
}  // namespace ext
struct ToDlpackOptions {
  std::optional<std::pair<int32_t, int32_t>> dl_device{};
  bool copy = false;
  ToDlpackOptions& with_dl_device(std::optional<std::pair<int32_t, int32_t>> value) & {
    dl_device = std::move(value);
    return *this;
  }
  ToDlpackOptions with_dl_device(std::optional<std::pair<int32_t, int32_t>> value) && {
    dl_device = std::move(value);
    return std::move(*this);
  }
  ToDlpackOptions& with_copy(bool value) & {
    copy = std::move(value);
    return *this;
  }
  ToDlpackOptions with_copy(bool value) && {
    copy = std::move(value);
    return std::move(*this);
  }
};
class Tensor {
  std::shared_ptr<RLMeshTensor> handle_{};

 public:
  explicit Tensor(RLMeshTensor* p) : handle_(p, rlmesh_tensor_free) {}
  Tensor(const void* buffer, size_t buffer_len, std::vector<size_t> shape, DType dtype) {
    RLMeshTensor* p = nullptr;
    detail::check_status(rlmesh_tensor_new(buffer, buffer_len, shape.data(), shape.size(),
                                           static_cast<RLMeshDType>(dtype), &p));
    handle_ = {p, rlmesh_tensor_free};
  }
  const RLMeshTensor* raw() const { return handle_.get(); }
  DType dtype() const { return static_cast<DType>(rlmesh_tensor_dtype(raw())); }
  Device device() const { return static_cast<Device>(rlmesh_tensor_device(raw())); }
  std::vector<size_t> shape() const {
    size_t n = 0;
    detail::check_status(rlmesh_tensor_shape(raw(), nullptr, 0, &n));
    std::vector<size_t> v(n);
    detail::check_status(rlmesh_tensor_shape(raw(), v.data(), v.size(), &n));
    return v;
  }
  std::vector<size_t> strides() const {
    size_t n = 0;
    detail::check_status(rlmesh_tensor_strides(raw(), nullptr, 0, &n));
    std::vector<size_t> v(n);
    detail::check_status(rlmesh_tensor_strides(raw(), v.data(), v.size(), &n));
    return v;
  }
  size_t ndim() const {
    size_t n = 0;
    detail::check_status(rlmesh_tensor_ndim(raw(), &n));
    return n;
  }
  size_t size() const {
    size_t n = 0;
    detail::check_status(rlmesh_tensor_size(raw(), &n));
    return n;
  }
  size_t nbytes() const {
    size_t n = 0;
    detail::check_status(rlmesh_tensor_nbytes(raw(), &n));
    return n;
  }
  Span<const std::byte> buffer() const {
    const void* p = nullptr;
    size_t n = 0;
    detail::check_status(rlmesh_tensor_buffer(raw(), &p, &n));
    return {static_cast<const std::byte*>(p), n};
  }
  bool is_contiguous() const { return rlmesh_tensor_is_contiguous(raw()); }
  Tensor reshape(std::vector<int64_t> shape) const {
    RLMeshTensor* p = nullptr;
    detail::check_status(rlmesh_tensor_reshape(raw(), shape.data(), shape.size(), &p));
    return Tensor(p);
  }
  Tensor copy() const {
    RLMeshTensor* p = nullptr;
    detail::check_status(rlmesh_tensor_copy(raw(), &p));
    return Tensor(p);
  }
  std::vector<std::byte> tobytes() const {
    detail::Bytes b;
    detail::check_status(rlmesh_tensor_tobytes(raw(), &b.value));
    if (!b.value.len) return {};
    const auto* p = reinterpret_cast<const std::byte*>(b.value.data);
    return {p, p + b.value.len};
  }
  std::string to_string() const { return detail::text(rlmesh_tensor_repr, raw()); }
  static Tensor from_dlpack(ext::DlpackPtr&& obj) {
    RLMeshTensor* p = nullptr;
    detail::check_status(rlmesh_tensor_from_dlpack(obj.get(), &p));
    obj.release();
    return Tensor(p);
  }
  ext::DlpackPtr to_dlpack(ToDlpackOptions options = {}) const {
    RLMeshToDlpackOptions c = RLMESH_TO_DLPACK_OPTIONS_INIT;
    c.copy = options.copy;
    if (options.dl_device) {
      c.has_dl_device = true;
      c.dl_device_type = options.dl_device->first;
      c.dl_device_id = options.dl_device->second;
    }
    DLManagedTensorVersioned* p = nullptr;
    detail::check_status(rlmesh_tensor_to_dlpack(raw(), &c, &p));
    return ext::DlpackPtr(p);
  }
  std::pair<int32_t, int32_t> dlpack_device() const {
    int32_t t = 0, i = 0;
    detail::check_status(rlmesh_tensor_dlpack_device(raw(), &t, &i));
    return {t, i};
  }
};
inline std::string to_string(const Tensor& v) { return v.to_string(); }
inline std::ostream& operator<<(std::ostream& s, const Tensor& v) { return s << v.to_string(); }
class Metadata;
enum class ValueKind { None = 1, Bool, Int, Float, Text, Bytes, Tensor, List, Tuple, Dict };
class Value {
  std::shared_ptr<RLMeshValue> handle_{};
  static Value sequence(std::vector<Value> items, bool tuple) {
    std::vector<const RLMeshValue*> children;
    for (const auto& v : items) children.push_back(v.raw());
    RLMeshValue* p = nullptr;
    detail::check_status(
        (tuple ? rlmesh_value_tuple : rlmesh_value_list)(children.data(), children.size(), &p));
    return adopt(p);
  }

 public:
  static Value adopt(RLMeshValue* p) { return Value(p, 0); }
  static Value copy_of(const RLMeshValue* p) { return adopt(rlmesh_value_clone(p)); }
  Value() {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_value_none(&p));
    handle_ = {p, rlmesh_value_free};
  }
  Value(std::nullptr_t) = delete;
  Value(bool v) {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_value_bool(v, &p));
    handle_ = {p, rlmesh_value_free};
  }
  Value(char) = delete;
  Value(wchar_t) = delete;

  template <class C, std::enable_if_t<std::is_same_v<C, decltype(u8'a')>, int> = 0>
  Value(C) = delete;
  Value(char16_t) = delete;
  Value(char32_t) = delete;
  template <class I, std::enable_if_t<detail::value_integer<I>, int> = 0>
  Value(I v) {
    if constexpr (std::is_unsigned_v<I>)
      if (v > static_cast<std::make_unsigned_t<int64_t>>(INT64_MAX))
        throw ValueError("integer exceeds int64 range");
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_value_int(static_cast<int64_t>(v), &p));
    handle_ = {p, rlmesh_value_free};
  }
  template <class F, std::enable_if_t<std::is_floating_point_v<F>, int> = 0>
  Value(F v) {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_value_float(static_cast<double>(v), &p));
    handle_ = {p, rlmesh_value_free};
  }
  Value(const char* v) : Value(std::string_view(v ? v : "")) {
    if (!v) throw TypeError("text must not be null");
  }
  Value(std::string v) : Value(std::string_view(v)) {}
  Value(std::string_view v) {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_value_text(detail::str(v), &p));
    handle_ = {p, rlmesh_value_free};
  }
  Value(Tensor v) {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_value_tensor(v.raw(), &p));
    handle_ = {p, rlmesh_value_free};
  }
  Value(const Metadata&);
  static Value none() { return {}; }
  static Value bytes(Span<const std::byte> v) {
    RLMeshValue* p = nullptr;
    detail::check_status(
        rlmesh_value_bytes(reinterpret_cast<const uint8_t*>(v.data()), v.size(), &p));
    return adopt(p);
  }
  static Value list(std::vector<Value> items) { return sequence(std::move(items), false); }
  static Value tuple(std::vector<Value> items) { return sequence(std::move(items), true); }
  static Value dict(const Metadata&);
  const RLMeshValue* raw() const { return handle_.get(); }
  RLMeshValue* clone_raw() const { return rlmesh_value_clone(raw()); }
  ValueKind kind() const { return static_cast<ValueKind>(rlmesh_value_kind(raw())); }
  bool is_none() const { return kind() == ValueKind::None; }
  bool as_bool() const {
    bool v = false;
    detail::check_status(rlmesh_value_as_bool(raw(), &v));
    return v;
  }
  int64_t as_int() const {
    int64_t v = 0;
    detail::check_status(rlmesh_value_as_int(raw(), &v));
    return v;
  }
  double as_float() const {
    double v = 0;
    detail::check_status(rlmesh_value_as_float(raw(), &v));
    return v;
  }
  std::string_view as_text() const {
    RLMeshStr v{};
    detail::check_status(rlmesh_value_as_text(raw(), &v));
    return {v.data ? v.data : "", v.len};
  }
  Span<const std::byte> as_bytes() const {
    const uint8_t* p = nullptr;
    size_t n = 0;
    detail::check_status(rlmesh_value_as_bytes(raw(), &p, &n));
    return {reinterpret_cast<const std::byte*>(p), n};
  }
  Tensor as_tensor() const {
    RLMeshTensor* p = nullptr;
    detail::check_status(rlmesh_value_as_tensor(raw(), &p));
    return Tensor(p);
  }
  size_t size() const {
    size_t n = 0;
    detail::check_status(rlmesh_value_len(raw(), &n));
    return n;
  }
  Value operator[](size_t i) const {
    const RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_value_get(raw(), i, &p));
    if (!p) throw std::out_of_range("Value index");
    return copy_of(p);
  }
  std::vector<Value> as_list() const {
    if (kind() != ValueKind::List) throw TypeError("Value is not a list");
    std::vector<Value> v;
    for (size_t i = 0; i < size(); ++i) v.push_back((*this)[i]);
    return v;
  }
  std::vector<Value> as_tuple() const {
    if (kind() != ValueKind::Tuple) throw TypeError("Value is not a tuple");
    std::vector<Value> v;
    for (size_t i = 0; i < size(); ++i) v.push_back((*this)[i]);
    return v;
  }
  Metadata as_dict() const;

 private:
  Value(RLMeshValue* p, int) : handle_(p, rlmesh_value_free) {}

 public:
  friend bool operator==(const Value& self, const Value& v) {
    return rlmesh_value_equals(self.raw(), v.raw());
  }
  friend bool operator!=(const Value& self, const Value& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
class Metadata {
  std::vector<std::pair<std::string, Value>> entries_{};

 public:
  Metadata() = default;
  Metadata(std::initializer_list<std::pair<std::string, Value>> entries) {
    for (const auto& [k, v] : entries) insert_or_assign(k, v);
  }
  auto begin() const { return entries_.begin(); }
  auto end() const { return entries_.end(); }
  auto find(std::string_view k) const {
    return std::find_if(begin(), end(), [&](const auto& e) { return e.first == k; });
  }
  bool contains_key(std::string_view k) const { return find(k) != end(); }
  const Value& at(std::string_view k) const {
    auto i = find(k);
    if (i == end()) throw std::out_of_range("no Metadata key " + std::string(k));
    return i->second;
  }
  void insert_or_assign(std::string_view k, Value v) {
    auto i =
        std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.first == k; });
    if (i == entries_.end())
      entries_.emplace_back(k, std::move(v));
    else
      i->second = std::move(v);
  }
  bool erase(std::string_view k) {
    auto i =
        std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.first == k; });
    if (i == entries_.end()) return false;
    entries_.erase(i);
    return true;
  }
  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

  friend bool operator==(const Metadata& self, const Metadata& m) {
    if (self.size() != m.size()) return false;
    for (const auto& [k, v] : self.entries_) {
      auto i = m.find(k);
      if (i == m.end() || !(v == i->second)) return false;
    }
    return true;
  }
  friend bool operator!=(const Metadata& self,
                         const Metadata& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
using InfoDict = Metadata;
using Kwargs = Metadata;
inline Value::Value(const Metadata& v) {
  std::vector<RLMeshStr> keys{};
  std::vector<const RLMeshValue*> values{};
  for (const auto& [k, x] : v) {
    keys.push_back(detail::str(k));
    values.push_back(x.raw());
  }
  RLMeshValue* p = nullptr;
  detail::check_status(rlmesh_value_dict(keys.data(), values.data(), keys.size(), &p));
  handle_ = {p, rlmesh_value_free};
}
inline Value Value::dict(const Metadata& v) { return Value(v); }
inline Metadata Value::as_dict() const {
  if (kind() != ValueKind::Dict) throw TypeError("Value is not a dict");
  Metadata m{};
  for (size_t i = 0; i < size(); ++i) {
    RLMeshStr k{};
    const RLMeshValue* v = nullptr;
    detail::check_status(rlmesh_value_dict_key_at(raw(), i, &k));
    detail::check_status(rlmesh_value_dict_value_at(raw(), i, &v));
    m.insert_or_assign({k.data, k.len}, copy_of(v));
  }
  return m;
}
namespace detail {
inline InfoDict info(const RLMeshValue* v) { return v ? Value::copy_of(v).as_dict() : InfoDict{}; }
}  // namespace detail
// OptionalList preserves an absent value versus an explicitly empty list.
template <class T>
class OptionalList {
  std::optional<std::vector<T>> value_{};

 public:
  OptionalList() = default;
  OptionalList(std::nullopt_t) {}
  OptionalList(std::vector<T> value) : value_(std::move(value)) {}
  OptionalList(std::initializer_list<T> value) : value_(std::vector<T>(value)) {}
  bool has_value() const { return value_.has_value(); }
  explicit operator bool() const { return has_value(); }
  const std::vector<T>& operator*() const { return value_.value(); }
  const std::vector<T>* operator->() const { return &value_.value(); }

  friend constexpr bool operator==(const OptionalList& self, const OptionalList& other) noexcept(
      detail::nothrow_equal<decltype(value_)>) {
    return self.value_ == other.value_;
  }
  friend constexpr bool operator!=(const OptionalList& self,
                                   const OptionalList& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
namespace spaces {

/// A space's kind.
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

/// How a Box stores its bounds.
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
struct Adopt {};

[[noreturn]] inline void type_error(const char* what) { throw rlmesh::TypeError(what); }

[[noreturn]] inline void last(RLMeshStatus status) {
  rlmesh::detail::check_status(status);
  throw RLMeshException("missing C error");
}

inline std::string take_bytes(RLMeshBytes bytes) {
  std::string out(reinterpret_cast<const char*>(bytes.data), bytes.len);
  rlmesh_bytes_free(bytes);
  return out;
}

inline RLMeshStr str(std::string_view text) {
  RLMeshStr out{};
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
  Bound(T value) {  // NOLINT(google-explicit-constructor): accepts numeric bounds
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
  /// Each throws `rlmesh::TypeError` on another type.
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
  RLMeshBound raw_{};
};

/// A Box's `low` / `high`: one bound for every element, or one per element.
using BoxBound = std::variant<Bound, std::vector<Bound>>;

/// `Box`'s keyword arguments.
struct BoxOptions {
  DType dtype = DType::Float32;
  BoxOptions& with_dtype(DType value) & {
    dtype = std::move(value);
    return *this;
  }
  BoxOptions with_dtype(DType value) && {
    dtype = std::move(value);
    return std::move(*this);
  }
};

/// `Discrete`'s keyword arguments.
struct DiscreteOptions {
  int64_t start = 0;
  DType dtype = DType::Int64;
  DiscreteOptions& with_start(int64_t value) & {
    start = std::move(value);
    return *this;
  }
  DiscreteOptions with_start(int64_t value) && {
    start = std::move(value);
    return std::move(*this);
  }
  DiscreteOptions& with_dtype(DType value) & {
    dtype = std::move(value);
    return *this;
  }
  DiscreteOptions with_dtype(DType value) && {
    dtype = std::move(value);
    return std::move(*this);
  }
};

class Space;

/// Immutable data describing a space.
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
    rlmesh::detail::check_status(rlmesh_space_spec_shape(ptr_, nullptr, 0, &len));
    std::vector<int64_t> out(len);
    rlmesh::detail::check_status(rlmesh_space_spec_shape(ptr_, out.data(), out.size(), &len));
    return out;
  }
  DType dtype() const { return static_cast<DType>(rlmesh_space_spec_dtype(ptr_)); }
  /// A fresh space with its own sampler.
  inline Space to_space() const;
  /// Gymnasium-style text.
  std::string to_string() const {
    RLMeshBytes bytes = {nullptr, 0, 0};
    rlmesh::detail::check_status(rlmesh_space_spec_repr(ptr_, &bytes));
    return detail::take_bytes(bytes);
  }

 private:
  explicit SpaceSpec(RLMeshSpaceSpec* ptr) : ptr_(ptr) {}
  RLMeshSpaceSpec* ptr_{};
};

class Box;
class Discrete;
class Dict;

/// A space: a spec plus its sampler. Copies share the handle, so two copies
/// drive one RNG.
class Space {
 public:
  /// `Space(spec)`.
  static Space from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_space_new(spec.raw(), &out);
    if (status != RLMESH_OK) detail::last(status);
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
  explicit Space(const SpaceSpec& spec) : Space(from_spec(spec)) {}
  SpaceSpec spec() const { return SpaceSpec::copy_of(rlmesh_space_spec(ptr_)); }
  /// A space converts to its spec (Dict's children are specs).
  operator SpaceSpec() const { return spec(); }  // NOLINT(google-explicit-constructor)
  SpaceKind kind() const { return static_cast<SpaceKind>(rlmesh_space_kind(ptr_)); }
  std::vector<int64_t> shape() const {
    size_t len = 0;
    rlmesh::detail::check_status(rlmesh_space_shape(ptr_, nullptr, 0, &len));
    std::vector<int64_t> out(len);
    rlmesh::detail::check_status(rlmesh_space_shape(ptr_, out.data(), out.size(), &len));
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
      rlmesh::detail::check_status(rlmesh_space_seed(ptr_, &value, &used));
    } else {
      rlmesh::detail::check_status(rlmesh_space_seed(ptr_, nullptr, &used));
    }
    return used;
  }
  Value sample() const {
    RLMeshValue* out = nullptr;
    RLMeshStatus status = rlmesh_space_sample(ptr_, &out);
    if (status != RLMESH_OK) detail::last(status);
    return Value::adopt(out);
  }
  bool contains(const Value& value) const { return rlmesh_space_contains(ptr_, value.raw()); }

  std::string to_string() const {
    RLMeshBytes bytes = {nullptr, 0, 0};
    rlmesh::detail::check_status(rlmesh_space_repr(ptr_, &bytes));
    return detail::take_bytes(bytes);
  }

  inline std::optional<Box> as_box() const;
  inline std::optional<Discrete> as_discrete() const;
  inline std::optional<Dict> as_dict() const;

 protected:
  explicit Space(RLMeshSpace* ptr) : ptr_(ptr) {}
  RLMeshSpace* ptr_{};

 public:
  friend bool operator==(const Space& self, const Space& other) {
    return rlmesh_space_equals(self.ptr_, other.ptr_);
  }
  friend bool operator!=(const Space& self, const Space& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};

inline Space SpaceSpec::to_space() const {
  RLMeshSpace* out = nullptr;
  RLMeshStatus status = rlmesh_space_spec_to_space(ptr_, &out);
  if (status != RLMESH_OK) detail::last(status);
  return Space::adopt(out);
}

/// Build a space from a spec; refuses a spec with no payload.
inline Space space_from_spec(const SpaceSpec& spec) {
  RLMeshSpace* out = nullptr;
  RLMeshStatus status = rlmesh_spaces_space_from_spec(spec.raw(), &out);
  if (status != RLMESH_OK) detail::last(status);
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
  std::vector<Bound> out{};
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

/// A box of numbers in `[low, high]`.
class Box : public Space {
 public:
  /// `Box(low, high, shape, dtype)`.
  static Box create(Bound low, Bound high, const std::vector<int64_t>& shape,
                    BoxOptions options = {}) {
    RLMeshBoxOptions raw = detail::box_options(options);
    RLMeshSpace* out = nullptr;
    RLMeshStatus status =
        rlmesh_box_new(low.raw(), high.raw(), shape.data(), shape.size(), &raw, &out);
    if (status != RLMESH_OK) detail::last(status);
    return Box(out, detail::Adopt{});
  }
  /// `Box(spec)`: another kind is a ValueError.
  static Box from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_box_from_spec(spec.raw(), &out);
    if (status != RLMESH_OK) detail::last(status);
    return Box(out, detail::Adopt{});
  }
  /// Throwing forms of `create` / `from_spec`.
  Box(Bound low, Bound high, const std::vector<int64_t>& shape, BoxOptions options = {})
      : Box(create(low, high, shape, options)) {}
  explicit Box(const SpaceSpec& spec) : Box(from_spec(spec)) {}

  std::optional<BoxBound> low() const { return detail::read_bounds(ptr_, false); }
  std::optional<BoxBound> high() const { return detail::read_bounds(ptr_, true); }
  std::optional<BoundsKind> bounds_kind() const {
    bool has = false;
    RLMeshBoundsKind kind = RLMESH_BOUNDS_KIND_INVALID;
    rlmesh::detail::check_status(rlmesh_box_bounds_kind(ptr_, &has, &kind));
    if (!has) return std::nullopt;
    return static_cast<BoundsKind>(kind);
  }

 private:
  friend class Space;
  explicit Box(RLMeshSpace* ptr, detail::Adopt) : Space(ptr) {}
};

/// The integers `start .. start + n`.
class Discrete : public Space {
 public:
  static Discrete create(int64_t n, DiscreteOptions options = {}) {
    RLMeshDiscreteOptions raw = RLMESH_DISCRETE_OPTIONS_INIT;
    raw.start = options.start;
    raw.dtype = static_cast<RLMeshDType>(options.dtype);
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_discrete_new(n, &raw, &out);
    if (status != RLMESH_OK) detail::last(status);
    return Discrete(out, detail::Adopt{});
  }
  static Discrete from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_discrete_from_spec(spec.raw(), &out);
    if (status != RLMESH_OK) detail::last(status);
    return Discrete(out, detail::Adopt{});
  }
  explicit Discrete(int64_t n, DiscreteOptions options = {}) : Discrete(create(n, options)) {}
  explicit Discrete(const SpaceSpec& spec) : Discrete(from_spec(spec)) {}

  int64_t n() const {
    int64_t out = 0;
    rlmesh::detail::check_status(rlmesh_discrete_n(ptr_, &out));
    return out;
  }
  int64_t start() const {
    int64_t out = 0;
    rlmesh::detail::check_status(rlmesh_discrete_start(ptr_, &out));
    return out;
  }

 private:
  friend class Space;
  explicit Discrete(RLMeshSpace* ptr, detail::Adopt) : Space(ptr) {}
};

/// Named child spaces, read in declaration order.
class Dict : public Space {
 public:
  /// `Dict(spaces)`: each spec is copied into a fresh child; keys sort.
  static Dict create(const std::vector<std::pair<std::string, SpaceSpec>>& spaces) {
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
    if (status != RLMESH_OK) detail::last(status);
    return Dict(out, detail::Adopt{});
  }
  static Dict from_spec(const SpaceSpec& spec) {
    RLMeshSpace* out = nullptr;
    RLMeshStatus status = rlmesh_dict_from_spec(spec.raw(), &out);
    if (status != RLMESH_OK) detail::last(status);
    return Dict(out, detail::Adopt{});
  }
  explicit Dict(const std::vector<std::pair<std::string, SpaceSpec>>& spaces)
      : Dict(create(spaces)) {}
  explicit Dict(const SpaceSpec& spec) : Dict(from_spec(spec)) {}

  size_t size() const {
    size_t out = 0;
    rlmesh::detail::check_status(rlmesh_dict_len(ptr_, &out));
    return out;
  }
  /// The `i`-th key, borrowed while any handle sharing the Dict lives.
  std::string_view key_at(size_t i) const {
    RLMeshStr out = {nullptr, 0};
    rlmesh::detail::check_status(rlmesh_dict_key_at(ptr_, i, &out));
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
  /// The children in declaration order, sharing the Dict's child samplers.
  std::vector<std::pair<std::string, Space>> spaces() const { return items(); }
  std::optional<Space> get(std::string_view key) const {
    const RLMeshSpace* child = nullptr;
    rlmesh::detail::check_status(rlmesh_dict_get(ptr_, detail::str(key), &child));
    if (child == nullptr) {
      return std::nullopt;
    }
    return borrow(child);
  }
  /// `d[key]`; throws `std::out_of_range` when absent.
  Space operator[](std::string_view key) const {
    std::optional<Space> child = get(key);
    if (!child.has_value()) throw std::out_of_range("no Dict key " + std::string(key));
    return std::move(*child);
  }
  bool contains_key(std::string_view key) const {
    bool out = false;
    rlmesh::detail::check_status(rlmesh_dict_contains_key(ptr_, detail::str(key), &out));
    return out;
  }

  class key_iterator {
    const RLMeshDict* dict_ = nullptr;
    std::ptrdiff_t index_ = 0;

   public:
    using value_type = std::string_view;
    using reference = std::string_view;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept = std::random_access_iterator_tag;
    key_iterator() = default;
    key_iterator(const RLMeshDict* d, difference_type i) : dict_(d), index_(i) {}
    reference operator*() const {
      RLMeshStr s{};
      rlmesh::detail::check_status(rlmesh_dict_key_at(dict_, static_cast<size_t>(index_), &s));
      return {s.data ? s.data : "", s.len};
    }
    reference operator[](difference_type n) const { return *(*this + n); }
    key_iterator& operator++() {
      ++index_;
      return *this;
    }
    key_iterator operator++(int) {
      auto p = *this;
      ++*this;
      return p;
    }
    key_iterator& operator--() {
      --index_;
      return *this;
    }
    key_iterator operator--(int) {
      auto p = *this;
      --*this;
      return p;
    }
    key_iterator& operator+=(difference_type n) {
      index_ += n;
      return *this;
    }
    key_iterator& operator-=(difference_type n) {
      index_ -= n;
      return *this;
    }
    friend key_iterator operator+(key_iterator i, difference_type n) { return i += n; }
    friend key_iterator operator+(difference_type n, key_iterator i) { return i += n; }
    friend key_iterator operator-(key_iterator i, difference_type n) { return i -= n; }
    friend difference_type operator-(key_iterator a, key_iterator b) { return a.index_ - b.index_; }

    bool operator<(const key_iterator& other) const {
      return std::tie(dict_, index_) < std::tie(other.dict_, other.index_);
    }
    bool operator>(const key_iterator& other) const { return other < *this; }
    bool operator<=(const key_iterator& other) const { return !(other < *this); }
    bool operator>=(const key_iterator& other) const { return !(*this < other); }

    friend constexpr bool operator==(const key_iterator& self, const key_iterator& other) noexcept {
      return self.dict_ == other.dict_ && self.index_ == other.index_;
    }
    friend constexpr bool operator!=(const key_iterator& self,
                                     const key_iterator& other) noexcept(noexcept(self == other)) {
      return !(self == other);
    }
  };
  key_iterator begin() const { return {ptr_, 0}; }
  key_iterator end() const { return {ptr_, static_cast<std::ptrdiff_t>(size())}; }

 private:
  friend class Space;
  explicit Dict(RLMeshSpace* ptr, detail::Adopt) : Space(ptr) {}
  /// A shared handle to a borrowed child (the same space on every read).
  static Space borrow(const RLMeshSpace* child) { return Space::adopt(rlmesh_space_clone(child)); }
};

inline std::optional<Box> Space::as_box() const {
  if (kind() != SpaceKind::Box) return std::nullopt;
  return Box(rlmesh_space_clone(ptr_), detail::Adopt{});
}
inline std::optional<Discrete> Space::as_discrete() const {
  if (kind() != SpaceKind::Discrete) return std::nullopt;
  return Discrete(rlmesh_space_clone(ptr_), detail::Adopt{});
}
inline std::optional<Dict> Space::as_dict() const {
  if (kind() != SpaceKind::Dict) return std::nullopt;
  return Dict(rlmesh_space_clone(ptr_), detail::Adopt{});
}

class MultiBinary : public Space {
  static RLMeshSpace* make(const std::vector<int64_t>& shape) {
    RLMeshSpace* p = nullptr;
    rlmesh::detail::check_status(rlmesh_multi_binary_new(shape.data(), shape.size(), &p));
    return p;
  }

 public:
  explicit MultiBinary(std::vector<int64_t> shape) : Space(make(shape)) {}
};
struct MultiDiscreteOptions {
  DType dtype = DType::Int64;
  MultiDiscreteOptions& with_dtype(DType value) & {
    dtype = std::move(value);
    return *this;
  }
  MultiDiscreteOptions with_dtype(DType value) && {
    dtype = std::move(value);
    return std::move(*this);
  }
};
class MultiDiscrete : public Space {
  static RLMeshSpace* make(const std::vector<int64_t>& nvec, MultiDiscreteOptions options) {
    RLMeshSpace* p = nullptr;
    rlmesh::detail::check_status(rlmesh_multi_discrete_new(
        nvec.data(), nvec.size(), static_cast<RLMeshDType>(options.dtype), &p));
    return p;
  }

 public:
  explicit MultiDiscrete(std::vector<int64_t> nvec, MultiDiscreteOptions options = {})
      : Space(make(nvec, options)) {}
  std::vector<int64_t> nvec() const {
    size_t n = 0;
    rlmesh::detail::check_status(rlmesh_multi_discrete_nvec(raw(), nullptr, 0, &n));
    std::vector<int64_t> v(n);
    rlmesh::detail::check_status(rlmesh_multi_discrete_nvec(raw(), v.data(), v.size(), &n));
    return v;
  }
};
struct TextOptions {
  int64_t min_length = 1;
  std::optional<std::string> charset{};
  TextOptions& with_min_length(int64_t value) & {
    min_length = std::move(value);
    return *this;
  }
  TextOptions with_min_length(int64_t value) && {
    min_length = std::move(value);
    return std::move(*this);
  }
  TextOptions& with_charset(std::optional<std::string> value) & {
    charset = std::move(value);
    return *this;
  }
  TextOptions with_charset(std::optional<std::string> value) && {
    charset = std::move(value);
    return std::move(*this);
  }
};
class Text : public Space {
  static RLMeshSpace* make(int64_t max, TextOptions options) {
    RLMeshSpace* p = nullptr;
    rlmesh::detail::check_status(
        rlmesh_text_new(max, options.min_length, rlmesh::detail::ptr(options.charset), &p));
    return p;
  }

 public:
  explicit Text(int64_t max_length, TextOptions options = {}) : Space(make(max_length, options)) {}
  int64_t min_length() const {
    int64_t min = 0, max = 0;
    RLMeshStr chars{};
    rlmesh::detail::check_status(rlmesh_text_limits(raw(), &min, &max, &chars));
    return min;
  }
  int64_t max_length() const {
    int64_t min = 0, max = 0;
    RLMeshStr chars{};
    rlmesh::detail::check_status(rlmesh_text_limits(raw(), &min, &max, &chars));
    return max;
  }
  std::string charset() const {
    int64_t min = 0, max = 0;
    RLMeshStr chars{};
    rlmesh::detail::check_status(rlmesh_text_limits(raw(), &min, &max, &chars));
    return std::string(chars.data ? chars.data : "", chars.len);
  }
};
class Tuple : public Space {
  static RLMeshSpace* make(const std::vector<SpaceSpec>& spaces) {
    std::vector<const RLMeshSpaceSpec*> ptrs;
    for (auto& s : spaces) ptrs.push_back(s.raw());
    RLMeshSpace* p = nullptr;
    rlmesh::detail::check_status(rlmesh_tuple_new(ptrs.data(), ptrs.size(), &p));
    return p;
  }

 public:
  explicit Tuple(std::vector<SpaceSpec> spaces) : Space(make(spaces)) {}
  std::vector<Space> spaces() const {
    size_t n = 0;
    rlmesh::detail::check_status(rlmesh_tuple_len(raw(), &n));
    std::vector<Space> v;
    for (size_t i = 0; i < n; ++i) {
      RLMeshSpace* p = nullptr;
      rlmesh::detail::check_status(rlmesh_tuple_get(raw(), i, &p));
      v.push_back(Space::adopt(p));
    }
    return v;
  }
};
inline std::string to_string(const Space& v) { return v.to_string(); }
inline std::string to_string(const SpaceSpec& v) { return v.to_string(); }
inline std::ostream& operator<<(std::ostream& s, const Space& v) { return s << v.to_string(); }
inline std::ostream& operator<<(std::ostream& s, const SpaceSpec& v) { return s << v.to_string(); }
}  // namespace spaces

namespace ext {

/// A Box with one bound pair per element (C++-only extra over
/// `rlmesh_ext_box_elementwise_new`).
inline spaces::Box box_elementwise(const std::vector<spaces::Bound>& low,
                                   const std::vector<spaces::Bound>& high,
                                   const std::vector<int64_t>& shape,
                                   spaces::BoxOptions options = {}) {
  if (low.size() != high.size()) {
    throw ValueError("Box low and high need the same length");
  }
  std::vector<RLMeshBound> raw_low{};
  std::vector<RLMeshBound> raw_high{};
  for (const spaces::Bound& bound : low) raw_low.push_back(bound.raw());
  for (const spaces::Bound& bound : high) raw_high.push_back(bound.raw());
  RLMeshBoxOptions raw = spaces::detail::box_options(options);
  RLMeshSpace* out = nullptr;
  RLMeshStatus status = rlmesh_ext_box_elementwise_new(
      raw_low.data(), raw_high.data(), raw_low.size(), shape.data(), shape.size(), &raw, &out);
  if (status != RLMESH_OK) spaces::detail::last(status);
  auto owned = detail::own<RLMeshSpace, rlmesh_space_free>(out);
  return spaces::Box::from_spec(spaces::SpaceSpec::copy_of(rlmesh_space_spec(owned.get())));
}

}  // namespace ext

namespace specs {
using SpaceSpec = spaces::SpaceSpec;
class EnvContract {
  std::shared_ptr<RLMeshEnvContract> handle_{};

 public:
  explicit EnvContract(RLMeshEnvContract* p) : handle_(p, rlmesh_env_contract_free) {}
  static EnvContract copy_of(const RLMeshEnvContract* p) {
    return EnvContract(rlmesh_env_contract_clone(p));
  }
  const RLMeshEnvContract* raw() const { return handle_.get(); }
  std::string id() const { return detail::text(rlmesh_env_contract_id, raw()); }
  uint64_t num_envs() const { return rlmesh_env_contract_num_envs(raw()); }
  spaces::SpaceSpec observation_space() const {
    return spaces::SpaceSpec::copy_of(rlmesh_env_contract_observation_space(raw()));
  }
  spaces::SpaceSpec action_space() const {
    return spaces::SpaceSpec::copy_of(rlmesh_env_contract_action_space(raw()));
  }
  Metadata metadata() const { return detail::info(rlmesh_env_contract_metadata(raw())); }
  std::optional<std::string> render_mode() const {
    bool has = false;
    detail::Bytes b;
    detail::check_status(rlmesh_env_contract_render_mode(raw(), &has, &b.value));
    return has ? std::optional(b.text()) : std::nullopt;
  }
  EnvContract env() const { return EnvContract(rlmesh_env_contract_env(raw())); }
  EnvContract spec() const { return EnvContract(rlmesh_env_contract_spec(raw())); }
  Metadata to_dict() const {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_env_contract_to_dict(raw(), &p));
    return Value::adopt(p).as_dict();
  }
  std::string to_string() const { return detail::text(rlmesh_env_contract_repr, raw()); }
};
}  // namespace specs
inline std::optional<int64_t> trial_index(const std::optional<InfoDict>& options,
                                          std::optional<size_t> lane = {}) {
  std::optional<Value> v{};
  if (options) v = Value(*options);
  bool has = false;
  int64_t n = 0;
  detail::check_status(rlmesh_trial_index(v ? v->raw() : nullptr, detail::ptr(lane), &has, &n));
  return has ? std::optional(n) : std::nullopt;
}
namespace adapters {
enum class ImageLayout { Hwc = RLMESH_IMAGE_LAYOUT_HWC, Chw = RLMESH_IMAGE_LAYOUT_CHW };
inline std::string_view to_string(ImageLayout v) {
  return detail::sv(rlmesh_image_layout_str(static_cast<RLMeshImageLayout>(v)));
}
enum class FitMode {
  Stretch = RLMESH_FIT_MODE_STRETCH,
  Crop = RLMESH_FIT_MODE_CROP,
  Pad = RLMESH_FIT_MODE_PAD
};
inline std::string_view to_string(FitMode v) {
  return detail::sv(rlmesh_fit_mode_str(static_cast<RLMeshFitMode>(v)));
}
enum class Frame {
  World = RLMESH_FRAME_WORLD,
  RobotBase = RLMESH_FRAME_ROBOT_BASE,
  Tool = RLMESH_FRAME_TOOL
};
inline std::string_view to_string(Frame v) {
  return detail::sv(rlmesh_frame_str(static_cast<RLMeshFrame>(v)));
}
enum class Provenance {
  Sensed = RLMESH_PROVENANCE_SENSED,
  Estimated = RLMESH_PROVENANCE_ESTIMATED,
  Privileged = RLMESH_PROVENANCE_PRIVILEGED
};
inline std::string_view to_string(Provenance v) {
  return detail::sv(rlmesh_provenance_str(static_cast<RLMeshProvenance>(v)));
}
enum class Reference { Current = RLMESH_REFERENCE_CURRENT, Target = RLMESH_REFERENCE_TARGET };
inline std::string_view to_string(Reference v) {
  return detail::sv(rlmesh_reference_str(static_cast<RLMeshReference>(v)));
}
enum class Resample {
  Bilinear = RLMESH_RESAMPLE_BILINEAR,
  BilinearAa = RLMESH_RESAMPLE_BILINEAR_AA,
  BicubicAa = RLMESH_RESAMPLE_BICUBIC_AA,
  Lanczos3Aa = RLMESH_RESAMPLE_LANCZOS3_AA,
  Area = RLMESH_RESAMPLE_AREA
};
inline std::string_view to_string(Resample v) {
  return detail::sv(rlmesh_resample_str(static_cast<RLMeshResample>(v)));
}
enum class RotationEncoding {
  QuatXyzw = RLMESH_ROTATION_ENCODING_QUAT_XYZW,
  QuatWxyz = RLMESH_ROTATION_ENCODING_QUAT_WXYZ,
  AxisAngle = RLMESH_ROTATION_ENCODING_AXIS_ANGLE,
  Rot6D = RLMESH_ROTATION_ENCODING_ROT6D,
  Rot6DRowmajor = RLMESH_ROTATION_ENCODING_ROT6D_ROWMAJOR,
  EulerXyz = RLMESH_ROTATION_ENCODING_EULER_XYZ,
  GravityXyz = RLMESH_ROTATION_ENCODING_GRAVITY_XYZ
};
inline std::string_view to_string(RotationEncoding v) {
  return detail::sv(rlmesh_rotation_encoding_str(static_cast<RLMeshRotationEncoding>(v)));
}
enum class StackPad { First = RLMESH_STACK_PAD_FIRST, Black = RLMESH_STACK_PAD_BLACK };
inline std::string_view to_string(StackPad v) {
  return detail::sv(rlmesh_stack_pad_str(static_cast<RLMeshStackPad>(v)));
}
enum class CropMode { Zoom = RLMESH_CROP_MODE_ZOOM, Slice = RLMESH_CROP_MODE_SLICE };
inline std::string_view to_string(CropMode v) {
  return detail::sv(rlmesh_crop_mode_str(static_cast<RLMeshCropMode>(v)));
}
enum class ChannelOrder { Rgb = RLMESH_CHANNEL_ORDER_RGB, Bgr = RLMESH_CHANNEL_ORDER_BGR };
inline std::string_view to_string(ChannelOrder v) {
  return detail::sv(rlmesh_channel_order_str(static_cast<RLMeshChannelOrder>(v)));
}
enum class StateContainer {
  Array = RLMESH_STATE_CONTAINER_ARRAY,
  List = RLMESH_STATE_CONTAINER_LIST
};
inline std::string_view to_string(StateContainer v) {
  return detail::sv(rlmesh_state_container_str(static_cast<RLMeshStateContainer>(v)));
}
enum class TextContainer { Str = RLMESH_TEXT_CONTAINER_STR, List = RLMESH_TEXT_CONTAINER_LIST };
inline std::string_view to_string(TextContainer v) {
  return detail::sv(rlmesh_text_container_str(static_cast<RLMeshTextContainer>(v)));
}
enum class StateSource {
  Observation = RLMESH_STATE_SOURCE_OBSERVATION,
  Action = RLMESH_STATE_SOURCE_ACTION
};
inline std::string_view to_string(StateSource v) {
  return detail::sv(rlmesh_state_source_str(static_cast<RLMeshStateSource>(v)));
}
enum class AdvisorySeverity {
  Info = RLMESH_ADVISORY_SEVERITY_INFO,
  Caution = RLMESH_ADVISORY_SEVERITY_CAUTION
};
inline std::string_view to_string(AdvisorySeverity v) {
  return detail::sv(rlmesh_advisory_severity_str(static_cast<RLMeshAdvisorySeverity>(v)));
}
template <class E>
class Tolerant {
  std::variant<E, std::string> value_{};

 public:
  Tolerant(E e) : value_(e) {}
  static Tolerant parse(std::string_view s) { return Tolerant(std::string(s)); }
  std::optional<E> known() const {
    if (auto p = std::get_if<E>(&value_)) return *p;
    for (int i = 1; i <= 16; ++i)
      if (adapters_string(static_cast<E>(i)) == std::get<std::string>(value_))
        return static_cast<E>(i);
    return {};
  }
  std::string to_string() const {
    if (auto p = std::get_if<E>(&value_)) return std::string(adapters_string(*p));
    return std::get<std::string>(value_);
  }

 private:
  explicit Tolerant(std::string s) : value_(std::move(s)) {}
  static std::string_view adapters_string(E e) {
    using adapters::to_string;
    using rlmesh::to_string;
    return to_string(e);
  }

 public:
  friend bool operator==(const Tolerant& self, const Tolerant& v) {
    return self.to_string() == v.to_string();
  }
  friend bool operator!=(const Tolerant& self,
                         const Tolerant& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
namespace detail_adapter {
struct Arena {
  std::vector<std::shared_ptr<void>> values{};
  template <class T>
  const T* keep(T value) {
    auto p = std::make_shared<T>(std::move(value));
    const auto* r = p.get();
    values.push_back(std::move(p));
    return r;
  }
  template <class T>
  const T* array(std::vector<T> v) {
    auto p = keep(std::move(v));
    return p->empty() ? nullptr : p->data();
  }
  const char* text(std::string s) { return keep(std::move(s))->c_str(); }
  const char* const* strings(const std::vector<std::string>& v) {
    std::vector<const char*> p;
    for (auto& s : v) p.push_back(text(s));
    return array(std::move(p));
  }
};
template <class Node>
bool dict_equal(const std::vector<std::pair<std::string, Node>>& left,
                const std::vector<std::pair<std::string, Node>>& right) {
  return left.size() == right.size() &&
         std::all_of(left.begin(), left.end(), [&](const auto& entry) {
           return std::find(right.begin(), right.end(), entry) != right.end();
         });
}
template <class E, class C>
C tolerant(const Tolerant<E>& v, const char*& other, Arena& a) {
  auto k = v.known();
  if (k) return static_cast<C>(*k);
  other = a.text(v.to_string());
  return static_cast<C>(1000);
}
template <class E, class C>
Tolerant<E> tolerant(C v, const char* other) {
  if (static_cast<int>(v) == 1000) return Tolerant<E>::parse(detail::sv(other));
  return Tolerant<E>(static_cast<E>(v));
}
}  // namespace detail_adapter
class ObsTransform {
  std::shared_ptr<RLMeshObsTransform> handle_{};

 public:
  explicit ObsTransform(RLMeshObsTransform* p) : handle_(p, rlmesh_obs_transform_free) {}
  template <class F, std::enable_if_t<std::is_invocable_v<const F&, const Value&>, int> = 0>
  ObsTransform(F f) {
    auto state = std::make_unique<F>(std::move(f));
    RLMeshObsTransform* p = nullptr;
    detail::check_status(rlmesh_obs_transform_new(
        [](void* u, const RLMeshValue* v, RLMeshValue** out) noexcept {
          return detail::callback(
              [&] { *out = (*static_cast<const F*>(u))(Value::copy_of(v)).clone_raw(); });
        },
        state.get(), detail::delete_state<F>, &p));
    state.release();
    handle_ = {p, rlmesh_obs_transform_free};
  }
  const RLMeshObsTransform* raw() const { return handle_.get(); }

  friend bool operator==(const ObsTransform& self, const ObsTransform& v) {
    return rlmesh_obs_transform_equals(self.raw(), v.raw());
  }
  friend bool operator!=(const ObsTransform& self,
                         const ObsTransform& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
class RotationTransform {
  std::shared_ptr<RLMeshRotationTransform> handle_{};

 public:
  explicit RotationTransform(RLMeshRotationTransform* p)
      : handle_(p, rlmesh_rotation_transform_free) {}
  template <class F, std::enable_if_t<std::is_invocable_v<const F&, const Value&>, int> = 0>
  RotationTransform(F f) {
    auto state = std::make_unique<F>(std::move(f));
    RLMeshRotationTransform* p = nullptr;
    detail::check_status(rlmesh_rotation_transform_new(
        [](void* u, const RLMeshValue* v, RLMeshValue** out) noexcept {
          return detail::callback(
              [&] { *out = (*static_cast<const F*>(u))(Value::copy_of(v)).clone_raw(); });
        },
        state.get(), detail::delete_state<F>, &p));
    state.release();
    handle_ = {p, rlmesh_rotation_transform_free};
  }
  const RLMeshRotationTransform* raw() const { return handle_.get(); }

  friend bool operator==(const RotationTransform& self, const RotationTransform& v) {
    return rlmesh_rotation_transform_equals(self.raw(), v.raw());
  }
  friend bool operator!=(const RotationTransform& self,
                         const RotationTransform& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
using EncodingArm = std::variant<RotationTransform, std::string>;
struct Rotation {
  RotationEncoding encoding{};
  std::vector<double> value{};
  RLMeshRotation raw(detail_adapter::Arena&) const {
    RLMeshRotation c = RLMESH_ROTATION_INIT;
    c.encoding = static_cast<RLMeshRotationEncoding>(encoding);
    c.value = value.data();
    c.value_len = value.size();
    return c;
  }
  static Rotation from_raw(const RLMeshRotation& c) {
    return {static_cast<RotationEncoding>(c.encoding), {c.value, c.value + c.value_len}};
  }
  static Rotation from_matrix(const std::array<std::array<double, 3>, 3>& rows) {
    double r[3][3];
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j) r[i][j] = rows[i][j];
    double values[6];
    RLMeshRotation c = RLMESH_ROTATION_INIT;
    detail::check_status(rlmesh_rotation_from_matrix(r, values, &c));
    return from_raw(c);
  }

  friend bool operator==(const Rotation& self, const Rotation& other) noexcept(
      detail::nothrow_equal<decltype(encoding), decltype(value)>) {
    return self.encoding == other.encoding && self.value == other.value;
  }
  friend bool operator!=(const Rotation& self,
                         const Rotation& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct CustomEncoding {
  RotationEncoding base{};
  std::optional<EncodingArm> from_base{};
  std::optional<EncodingArm> to_base{};
  std::string name = "custom";
  RLMeshCustomEncoding raw(detail_adapter::Arena&) const;
  static CustomEncoding from_raw(const RLMeshCustomEncoding&);
  bool is_entrypoint() const {
    detail_adapter::Arena a;
    auto c = raw(a);
    bool v = false;
    detail::check_status(rlmesh_custom_encoding_is_entrypoint(&c, &v));
    return v;
  }
  uint32_t width() const {
    detail_adapter::Arena a;
    auto c = raw(a);
    uint32_t v = 0;
    detail::check_status(rlmesh_custom_encoding_width(&c, &v));
    return v;
  }

  friend bool operator==(const CustomEncoding& self, const CustomEncoding& other) noexcept(
      detail::nothrow_equal<decltype(base), decltype(from_base), decltype(to_base),
                            decltype(name)>) {
    return self.base == other.base && self.from_base == other.from_base &&
           self.to_base == other.to_base && self.name == other.name;
  }
  friend bool operator!=(const CustomEncoding& self,
                         const CustomEncoding& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline RLMeshCustomEncoding CustomEncoding::raw(detail_adapter::Arena&) const {
  RLMeshCustomEncoding c = RLMESH_CUSTOM_ENCODING_INIT;
  c.base = static_cast<RLMeshRotationEncoding>(base);
  c.name = name.c_str();
  auto arm = [](const std::optional<EncodingArm>& value) {
    RLMeshEncodingArm empty{};
    if (!value) return empty;
    const auto& v = *value;
    RLMeshEncodingArm r{};
    if (auto p = std::get_if<RotationTransform>(&v))
      r.transform = p->raw();
    else
      r.entrypoint = std::get<std::string>(v).c_str();
    return r;
  };
  c.from_base = arm(from_base);
  c.to_base = arm(to_base);
  return c;
}
inline CustomEncoding CustomEncoding::from_raw(const RLMeshCustomEncoding& c) {
  auto arm = [](const RLMeshEncodingArm& v) -> std::optional<EncodingArm> {
    if (v.transform) return RotationTransform(rlmesh_rotation_transform_clone(v.transform));
    if (v.entrypoint) return std::string(v.entrypoint);
    return {};
  };
  return {static_cast<RotationEncoding>(c.base), arm(c.from_base), arm(c.to_base),
          std::string(detail::sv(c.name))};
}
using AxisValue = std::variant<double, std::vector<double>>;
using ActionEncoding = std::variant<RotationEncoding, CustomEncoding>;
using StackSpec = std::vector<int32_t>;
struct Image {
  std::string role{};
  std::optional<uint32_t> height{};
  std::optional<uint32_t> width{};
  ImageLayout layout = ImageLayout::Hwc;
  std::optional<uint32_t> channels{};
  Tolerant<DType> dtype = DType::Uint8;
  std::variant<bool, std::pair<double, double>> normalize = false;
  uint32_t lead_dims = 0;
  bool upside_down = false;
  Tolerant<Resample> resample = Resample::Bilinear;
  bool allow_upscale = false;
  OptionalList<Tolerant<FitMode>> fit{};
  bool optional = false;
  std::optional<uint8_t> fill{};
  uint32_t stack = 1;
  std::optional<double> crop{};
  std::optional<double> crop_area{};
  Tolerant<CropMode> crop_mode = CropMode::Zoom;
  std::optional<uint8_t> jpeg_quality{};
  Tolerant<ChannelOrder> channel_order = ChannelOrder::Rgb;
  std::optional<std::pair<uint32_t, uint32_t>> render{};
  OptionalList<int32_t> offsets{};
  StackPad stack_pad = StackPad::First;
  std::optional<std::string> part{};
  RLMeshImage raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshImage c = RLMESH_IMAGE_INIT;
    c.role = (role).c_str();
    if (height) {
      c.has_height = true;
      c.height = *height;
    }
    if (width) {
      c.has_width = true;
      c.width = *width;
    }
    c.layout = static_cast<RLMeshImageLayout>(layout);
    if (channels) {
      c.has_channels = true;
      c.channels = *channels;
    }
    c.dtype = detail_adapter::tolerant<DType, RLMeshDType>(dtype, c.dtype_other, a);
    if (auto b = std::get_if<bool>(&normalize))
      c.normalize = *b ? RLMESH_NORMALIZE_UNIT : RLMESH_NORMALIZE_OFF;
    else {
      c.normalize = RLMESH_NORMALIZE_RANGE;
      auto v = std::get<std::pair<double, double>>(normalize);
      c.normalize_range[0] = v.first;
      c.normalize_range[1] = v.second;
    }
    c.lead_dims = lead_dims;
    c.upside_down = upside_down;
    c.resample = detail_adapter::tolerant<Resample, RLMeshResample>(resample, c.resample_other, a);
    c.allow_upscale = allow_upscale;
    if (fit) {
      std::vector<RLMeshFitMode> vals;
      std::vector<const char*> others;
      for (const auto& e : *fit) {
        const char* other = nullptr;
        vals.push_back(detail_adapter::tolerant<FitMode, RLMeshFitMode>(e, other, a));
        others.push_back(other);
      }
      c.fit = a.array(std::move(vals));
      c.fit_other = a.array(std::move(others));
      c.fit_len = fit->size();
      if (!c.fit) c.fit = a.keep(static_cast<RLMeshFitMode>(0));
    }
    c.optional = optional;
    if (fill) {
      c.has_fill = true;
      c.fill = *fill;
    }
    c.stack = stack;
    if (crop) {
      c.has_crop = true;
      c.crop = *crop;
    }
    if (crop_area) {
      c.has_crop_area = true;
      c.crop_area = *crop_area;
    }
    c.crop_mode =
        detail_adapter::tolerant<CropMode, RLMeshCropMode>(crop_mode, c.crop_mode_other, a);
    if (jpeg_quality) {
      c.has_jpeg_quality = true;
      c.jpeg_quality = *jpeg_quality;
    }
    c.channel_order = detail_adapter::tolerant<ChannelOrder, RLMeshChannelOrder>(
        channel_order, c.channel_order_other, a);
    if (render) {
      c.has_render = true;
      c.render[0] = render->first;
      c.render[1] = render->second;
    }
    if (offsets) {
      c.offsets = a.array(*offsets);
      c.offsets_len = offsets->size();
      if (!c.offsets) c.offsets = a.keep(int32_t{});
    }
    c.stack_pad = static_cast<RLMeshStackPad>(stack_pad);
    if (part) {
      c.part = (*part).c_str();
    }
    return c;
  }
  static Image from_raw(const RLMeshImage& c) {
    Image v;
    v.role = std::string(detail::sv(c.role));
    if (c.has_height) v.height = c.height;
    if (c.has_width) v.width = c.width;
    v.layout = static_cast<ImageLayout>(c.layout);
    if (c.has_channels) v.channels = c.channels;
    v.dtype = detail_adapter::tolerant<DType>(c.dtype, c.dtype_other);
    if (c.normalize == RLMESH_NORMALIZE_RANGE)
      v.normalize = std::pair<double, double>{c.normalize_range[0], c.normalize_range[1]};
    else
      v.normalize = (c.normalize == RLMESH_NORMALIZE_UNIT);
    v.lead_dims = c.lead_dims;
    v.upside_down = c.upside_down;
    v.resample = detail_adapter::tolerant<Resample>(c.resample, c.resample_other);
    v.allow_upscale = c.allow_upscale;
    if (c.fit) {
      std::vector<Tolerant<FitMode>> vals;
      for (size_t i = 0; i < c.fit_len; ++i)
        vals.push_back(
            detail_adapter::tolerant<FitMode>(c.fit[i], c.fit_other ? c.fit_other[i] : nullptr));
      v.fit = std::move(vals);
    }
    v.optional = c.optional;
    if (c.has_fill) v.fill = c.fill;
    v.stack = c.stack;
    if (c.has_crop) v.crop = c.crop;
    if (c.has_crop_area) v.crop_area = c.crop_area;
    v.crop_mode = detail_adapter::tolerant<CropMode>(c.crop_mode, c.crop_mode_other);
    if (c.has_jpeg_quality) v.jpeg_quality = c.jpeg_quality;
    v.channel_order =
        detail_adapter::tolerant<ChannelOrder>(c.channel_order, c.channel_order_other);
    if (c.has_render) v.render = std::pair<uint32_t, uint32_t>{c.render[0], c.render[1]};
    if (c.offsets) v.offsets = std::vector<int32_t>(c.offsets, c.offsets + c.offsets_len);
    v.stack_pad = static_cast<StackPad>(c.stack_pad);
    if (c.part) v.part = std::string(detail::sv(c.part));
    return v;
  }
  Image with_size(uint32_t size) const {
    auto v = *this;
    detail_adapter::Arena a;
    auto c = v.raw(a);
    detail::check_status(rlmesh_image_with_size(&c, size));
    v.height = c.height;
    v.width = c.width;
    return v;
  }
  Image with_stride(uint32_t stride) const {
    if (stride == 0 || offsets)
      throw ValueError("Image: stride must be >= 1 and cannot combine with offsets");
    auto value = *this;
    std::vector<int32_t> expanded;
    for (uint32_t i = 0; i < stack; ++i) {
      const int64_t offset = -static_cast<int64_t>(stack - 1 - i) * stride;
      if (offset < std::numeric_limits<int32_t>::min())
        throw ValueError("Image stride overflows offsets");
      expanded.push_back(static_cast<int32_t>(offset));
    }
    value.offsets = std::move(expanded);
    return value;
  }

  friend bool operator==(const Image& self, const Image& other) noexcept(
      detail::nothrow_equal<
          decltype(role), decltype(height), decltype(width), decltype(layout), decltype(channels),
          decltype(dtype), decltype(normalize), decltype(lead_dims), decltype(upside_down),
          decltype(resample), decltype(allow_upscale), decltype(fit), decltype(optional),
          decltype(fill), decltype(stack), decltype(crop), decltype(crop_area), decltype(crop_mode),
          decltype(jpeg_quality), decltype(channel_order), decltype(render), decltype(offsets),
          decltype(stack_pad), decltype(part)>) {
    return self.role == other.role && self.height == other.height && self.width == other.width &&
           self.layout == other.layout && self.channels == other.channels &&
           self.dtype == other.dtype && self.normalize == other.normalize &&
           self.lead_dims == other.lead_dims && self.upside_down == other.upside_down &&
           self.resample == other.resample && self.allow_upscale == other.allow_upscale &&
           self.fit == other.fit && self.optional == other.optional && self.fill == other.fill &&
           self.stack == other.stack && self.crop == other.crop &&
           self.crop_area == other.crop_area && self.crop_mode == other.crop_mode &&
           self.jpeg_quality == other.jpeg_quality && self.channel_order == other.channel_order &&
           self.render == other.render && self.offsets == other.offsets &&
           self.stack_pad == other.stack_pad && self.part == other.part;
  }
  friend bool operator!=(const Image& self, const Image& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct State {
  std::string role{};
  std::optional<std::variant<std::vector<Tolerant<RotationEncoding>>, CustomEncoding>> encoding{};
  std::optional<uint32_t> dim{};
  std::optional<uint32_t> index{};
  bool optional = false;
  std::optional<std::pair<double, double>> range{};
  std::optional<uint32_t> pad_to{};
  Tolerant<DType> dtype = DType::Float32;
  OptionalList<int64_t> reshape{};
  StateContainer container = StateContainer::Array;
  double fill = 0.0;
  std::optional<Rotation> post_rotate{};
  std::optional<AxisValue> scale{};
  std::optional<AxisValue> offset{};
  std::optional<Tolerant<Frame>> frame{};
  OptionalList<Tolerant<Provenance>> provenance{};
  std::optional<std::string> part{};
  OptionalList<std::string> labels{};
  StateSource source = StateSource::Observation;
  std::optional<std::pair<double, double>> clip{};
  std::optional<std::string> polarity{};
  RLMeshState raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshState c = RLMESH_STATE_INIT;
    c.role = (role).c_str();
    if (encoding) {
      if (auto custom = std::get_if<CustomEncoding>(&*encoding))
        c.custom_encoding = a.keep(custom->raw(a));
      else {
        const auto& native = std::get<std::vector<Tolerant<RotationEncoding>>>(*encoding);
        std::vector<RLMeshRotationEncoding> values;
        std::vector<const char*> others;
        for (const auto& e : native) {
          const char* other = nullptr;
          values.push_back(
              detail_adapter::tolerant<RotationEncoding, RLMeshRotationEncoding>(e, other, a));
          others.push_back(other);
        }
        c.encoding = a.array(std::move(values));
        c.encoding_other = a.array(std::move(others));
        c.encoding_len = native.size();
        if (!c.encoding) c.encoding = a.keep(static_cast<RLMeshRotationEncoding>(0));
      }
    }
    if (dim) {
      c.has_dim = true;
      c.dim = *dim;
    }
    if (index) {
      c.has_index = true;
      c.index = *index;
    }
    c.optional = optional;
    if (range) {
      c.has_range = true;
      c.range[0] = range->first;
      c.range[1] = range->second;
    }
    if (pad_to) {
      c.has_pad_to = true;
      c.pad_to = *pad_to;
    }
    c.dtype = detail_adapter::tolerant<DType, RLMeshDType>(dtype, c.dtype_other, a);
    if (reshape) {
      c.reshape = a.array(*reshape);
      c.reshape_len = reshape->size();
      if (!c.reshape) c.reshape = a.keep(int64_t{});
    }
    c.container = static_cast<RLMeshStateContainer>(container);
    c.fill = fill;
    if (post_rotate) {
      c.post_rotate = a.keep(post_rotate->raw(a));
    }
    if (scale) {
      if (auto p = std::get_if<double>(&*scale)) {
        c.has_scale = true;
        c.scale = *p;
      } else {
        const auto& v = std::get<std::vector<double>>(*scale);
        c.axis_scale = a.array(v);
        if (!c.axis_scale) c.axis_scale = a.keep(double{});
        c.axis_scale_len = v.size();
      }
    }
    if (offset) {
      if (auto p = std::get_if<double>(&*offset)) {
        c.has_offset = true;
        c.offset = *p;
      } else {
        const auto& v = std::get<std::vector<double>>(*offset);
        c.axis_offset = a.array(v);
        if (!c.axis_offset) c.axis_offset = a.keep(double{});
        c.axis_offset_len = v.size();
      }
    }
    if (frame) {
      c.has_frame = true;
      c.frame = detail_adapter::tolerant<Frame, RLMeshFrame>(*frame, c.frame_other, a);
    }
    if (provenance) {
      std::vector<RLMeshProvenance> vals;
      std::vector<const char*> others;
      for (const auto& e : *provenance) {
        const char* other = nullptr;
        vals.push_back(detail_adapter::tolerant<Provenance, RLMeshProvenance>(e, other, a));
        others.push_back(other);
      }
      c.provenance = a.array(std::move(vals));
      c.provenance_other = a.array(std::move(others));
      c.provenance_len = provenance->size();
      if (!c.provenance) c.provenance = a.keep(static_cast<RLMeshProvenance>(0));
    }
    if (part) {
      c.part = (*part).c_str();
    }
    if (labels) {
      c.labels = a.strings(*labels);
      c.labels_len = labels->size();
      if (!c.labels) c.labels = a.keep(static_cast<const char*>(nullptr));
    }
    c.source = static_cast<RLMeshStateSource>(source);
    if (clip) {
      c.has_clip = true;
      c.clip[0] = clip->first;
      c.clip[1] = clip->second;
    }
    c.polarity = detail::ptr(polarity);
    return c;
  }
  static State from_raw(const RLMeshState& c) {
    State v;
    v.role = std::string(detail::sv(c.role));
    if (c.custom_encoding)
      v.encoding = CustomEncoding::from_raw(*c.custom_encoding);
    else if (c.encoding) {
      std::vector<Tolerant<RotationEncoding>> values;
      for (size_t i = 0; i < c.encoding_len; ++i)
        values.push_back(detail_adapter::tolerant<RotationEncoding>(
            c.encoding[i], c.encoding_other ? c.encoding_other[i] : nullptr));
      v.encoding = std::move(values);
    }
    if (c.has_dim) v.dim = c.dim;
    if (c.has_index) v.index = c.index;
    v.optional = c.optional;
    if (c.has_range) v.range = std::pair<double, double>{c.range[0], c.range[1]};
    if (c.has_pad_to) v.pad_to = c.pad_to;
    v.dtype = detail_adapter::tolerant<DType>(c.dtype, c.dtype_other);
    if (c.reshape) v.reshape = std::vector<int64_t>(c.reshape, c.reshape + c.reshape_len);
    v.container = static_cast<StateContainer>(c.container);
    v.fill = c.fill;
    if (c.post_rotate) v.post_rotate = Rotation::from_raw(*c.post_rotate);
    if (c.has_scale)
      v.scale = c.scale;
    else if (c.axis_scale)
      v.scale = std::vector<double>(c.axis_scale, c.axis_scale + c.axis_scale_len);
    if (c.has_offset)
      v.offset = c.offset;
    else if (c.axis_offset)
      v.offset = std::vector<double>(c.axis_offset, c.axis_offset + c.axis_offset_len);
    if (c.has_frame) v.frame = detail_adapter::tolerant<Frame>(c.frame, c.frame_other);
    if (c.provenance) {
      std::vector<Tolerant<Provenance>> vals;
      for (size_t i = 0; i < c.provenance_len; ++i)
        vals.push_back(detail_adapter::tolerant<Provenance>(
            c.provenance[i], c.provenance_other ? c.provenance_other[i] : nullptr));
      v.provenance = std::move(vals);
    }
    if (c.part) v.part = std::string(detail::sv(c.part));
    if (c.labels) {
      std::vector<std::string> vals;
      for (size_t i = 0; i < c.labels_len; ++i) vals.emplace_back(c.labels[i]);
      v.labels = std::move(vals);
    }
    v.source = static_cast<StateSource>(c.source);
    if (c.has_clip) v.clip = std::pair<double, double>{c.clip[0], c.clip[1]};
    if (c.polarity) v.polarity = std::string(c.polarity);
    return v;
  }

  friend bool operator==(const State& self, const State& other) noexcept(
      detail::nothrow_equal<decltype(role), decltype(encoding), decltype(dim), decltype(index),
                            decltype(optional), decltype(range), decltype(pad_to), decltype(dtype),
                            decltype(reshape), decltype(container), decltype(fill),
                            decltype(post_rotate), decltype(scale), decltype(offset),
                            decltype(frame), decltype(provenance), decltype(part), decltype(labels),
                            decltype(source), decltype(clip), decltype(polarity)>) {
    return self.role == other.role && self.encoding == other.encoding && self.dim == other.dim &&
           self.index == other.index && self.optional == other.optional &&
           self.range == other.range && self.pad_to == other.pad_to && self.dtype == other.dtype &&
           self.reshape == other.reshape && self.container == other.container &&
           self.fill == other.fill && self.post_rotate == other.post_rotate &&
           self.scale == other.scale && self.offset == other.offset && self.frame == other.frame &&
           self.provenance == other.provenance && self.part == other.part &&
           self.labels == other.labels && self.source == other.source && self.clip == other.clip &&
           self.polarity == other.polarity;
  }
  friend bool operator!=(const State& self, const State& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Constant {
  uint32_t dim = 1;
  double fill = 0.0;
  RLMeshConstant raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshConstant c = RLMESH_CONSTANT_INIT;
    c.dim = dim;
    c.fill = fill;
    return c;
  }
  static Constant from_raw(const RLMeshConstant& c) {
    Constant v;
    v.dim = c.dim;
    v.fill = c.fill;
    return v;
  }

  friend constexpr bool operator==(const Constant& self, const Constant& other) noexcept(
      detail::nothrow_equal<decltype(dim), decltype(fill)>) {
    return self.dim == other.dim && self.fill == other.fill;
  }
  friend constexpr bool operator!=(const Constant& self,
                                   const Constant& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Text {
  std::string role{};
  TextContainer container = TextContainer::Str;
  std::optional<std::string> fill{};
  RLMeshAdaptersText raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshAdaptersText c = RLMESH_ADAPTERS_TEXT_INIT;
    c.role = (role).c_str();
    c.container = static_cast<RLMeshTextContainer>(container);
    if (fill) {
      c.has_fill = true;
      c.fill = detail::str(*fill);
    }
    return c;
  }
  static Text from_raw(const RLMeshAdaptersText& c) {
    Text v;
    v.role = std::string(detail::sv(c.role));
    v.container = static_cast<TextContainer>(c.container);
    if (c.has_fill) v.fill = std::string(c.fill.data ? c.fill.data : "", c.fill.len);
    return v;
  }

  friend bool operator==(const Text& self, const Text& other) noexcept(
      detail::nothrow_equal<decltype(role), decltype(container), decltype(fill)>) {
    return self.role == other.role && self.container == other.container && self.fill == other.fill;
  }
  friend bool operator!=(const Text& self, const Text& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Custom {
  std::optional<ObsTransform> transform{};
  std::optional<std::string> entrypoint{};
  RLMeshCustom raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshCustom c = RLMESH_CUSTOM_INIT;
    if (transform) {
      c.transform = transform->raw();
    }
    if (entrypoint) {
      c.entrypoint = (*entrypoint).c_str();
    }
    return c;
  }
  static Custom from_raw(const RLMeshCustom& c) {
    Custom v;
    if (c.transform) v.transform = ObsTransform(rlmesh_obs_transform_clone(c.transform));
    if (c.entrypoint) v.entrypoint = std::string(detail::sv(c.entrypoint));
    return v;
  }

  friend bool operator==(const Custom& self, const Custom& other) noexcept(
      detail::nothrow_equal<decltype(transform), decltype(entrypoint)>) {
    return self.transform == other.transform && self.entrypoint == other.entrypoint;
  }
  friend bool operator!=(const Custom& self,
                         const Custom& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Actuator {
  std::optional<std::string> role{};
  uint32_t dim = 0;
  std::optional<ActionEncoding> encoding{};
  std::optional<std::pair<double, double>> range{};
  bool binary = false;
  std::optional<AxisValue> scale{};
  bool invert = false;
  std::optional<double> threshold{};
  bool clip = false;
  double fill = 0.0;
  bool optional = false;
  std::optional<Tolerant<Frame>> frame{};
  std::optional<Tolerant<Reference>> reference{};
  std::optional<std::string> part{};
  std::optional<AxisValue> offset{};
  OptionalList<std::string> labels{};
  std::optional<std::string> polarity{};
  std::optional<double> unit{};
  RLMeshActuator raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshActuator c = RLMESH_ACTUATOR_INIT;
    c.role = detail::ptr(role);
    c.dim = dim;
    if (encoding) {
      if (auto p = std::get_if<RotationEncoding>(&*encoding)) {
        c.has_encoding = true;
        c.encoding = static_cast<RLMeshRotationEncoding>(*p);
      } else {
        c.custom_encoding = a.keep(std::get<CustomEncoding>(*encoding).raw(a));
      }
    }
    if (range) {
      c.has_range = true;
      c.range[0] = range->first;
      c.range[1] = range->second;
    }
    c.binary = binary;
    if (scale) {
      if (auto p = std::get_if<double>(&*scale)) {
        c.has_scale = true;
        c.scale = *p;
      } else {
        const auto& v = std::get<std::vector<double>>(*scale);
        c.axis_scale = a.array(v);
        if (!c.axis_scale) c.axis_scale = a.keep(double{});
        c.axis_scale_len = v.size();
      }
    }
    c.invert = invert;
    if (threshold) {
      c.has_threshold = true;
      c.threshold = *threshold;
    }
    c.clip = clip;
    c.fill = fill;
    c.optional = optional;
    if (frame) {
      c.has_frame = true;
      c.frame = detail_adapter::tolerant<Frame, RLMeshFrame>(*frame, c.frame_other, a);
    }
    if (reference) {
      c.has_reference = true;
      c.reference =
          detail_adapter::tolerant<Reference, RLMeshReference>(*reference, c.reference_other, a);
    }
    if (part) {
      c.part = (*part).c_str();
    }
    if (offset) {
      if (auto p = std::get_if<double>(&*offset)) {
        c.has_offset = true;
        c.offset = *p;
      } else {
        const auto& v = std::get<std::vector<double>>(*offset);
        c.axis_offset = a.array(v);
        if (!c.axis_offset) c.axis_offset = a.keep(double{});
        c.axis_offset_len = v.size();
      }
    }
    if (labels) {
      c.labels = a.strings(*labels);
      c.labels_len = labels->size();
      if (!c.labels) c.labels = a.keep(static_cast<const char*>(nullptr));
    }
    c.polarity = detail::ptr(polarity);
    if (unit) {
      c.has_unit = true;
      c.unit = *unit;
    }
    return c;
  }
  static Actuator from_raw(const RLMeshActuator& c) {
    Actuator v;
    if (c.role) v.role = std::string(c.role);
    v.dim = c.dim;
    if (c.custom_encoding)
      v.encoding = CustomEncoding::from_raw(*c.custom_encoding);
    else if (c.has_encoding)
      v.encoding = static_cast<RotationEncoding>(c.encoding);
    if (c.has_range) v.range = std::pair<double, double>{c.range[0], c.range[1]};
    v.binary = c.binary;
    if (c.has_scale)
      v.scale = c.scale;
    else if (c.axis_scale)
      v.scale = std::vector<double>(c.axis_scale, c.axis_scale + c.axis_scale_len);
    v.invert = c.invert;
    if (c.has_threshold) v.threshold = c.threshold;
    v.clip = c.clip;
    v.fill = c.fill;
    v.optional = c.optional;
    if (c.has_frame) v.frame = detail_adapter::tolerant<Frame>(c.frame, c.frame_other);
    if (c.has_reference)
      v.reference = detail_adapter::tolerant<Reference>(c.reference, c.reference_other);
    if (c.part) v.part = std::string(detail::sv(c.part));
    if (c.has_offset)
      v.offset = c.offset;
    else if (c.axis_offset)
      v.offset = std::vector<double>(c.axis_offset, c.axis_offset + c.axis_offset_len);
    if (c.labels) {
      std::vector<std::string> vals;
      for (size_t i = 0; i < c.labels_len; ++i) vals.emplace_back(c.labels[i]);
      v.labels = std::move(vals);
    }
    if (c.polarity) v.polarity = std::string(c.polarity);
    if (c.has_unit) v.unit = c.unit;
    return v;
  }

  friend bool operator==(const Actuator& self, const Actuator& other) noexcept(
      detail::nothrow_equal<decltype(role), decltype(dim), decltype(encoding), decltype(range),
                            decltype(binary), decltype(scale), decltype(invert),
                            decltype(threshold), decltype(clip), decltype(fill), decltype(optional),
                            decltype(frame), decltype(reference), decltype(part), decltype(offset),
                            decltype(labels), decltype(polarity), decltype(unit)>) {
    return self.role == other.role && self.dim == other.dim && self.encoding == other.encoding &&
           self.range == other.range && self.binary == other.binary && self.scale == other.scale &&
           self.invert == other.invert && self.threshold == other.threshold &&
           self.clip == other.clip && self.fill == other.fill && self.optional == other.optional &&
           self.frame == other.frame && self.reference == other.reference &&
           self.part == other.part && self.offset == other.offset && self.labels == other.labels &&
           self.polarity == other.polarity && self.unit == other.unit;
  }
  friend bool operator!=(const Actuator& self,
                         const Actuator& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct ImageTag {
  std::string role{};
  ImageLayout layout = ImageLayout::Hwc;
  bool upside_down = false;
  std::optional<std::string> part{};
  RLMeshImageTag raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshImageTag c = RLMESH_IMAGE_TAG_INIT;
    c.role = (role).c_str();
    c.layout = static_cast<RLMeshImageLayout>(layout);
    c.upside_down = upside_down;
    if (part) {
      c.part = (*part).c_str();
    }
    return c;
  }
  static ImageTag from_raw(const RLMeshImageTag& c) {
    ImageTag v;
    v.role = std::string(detail::sv(c.role));
    v.layout = static_cast<ImageLayout>(c.layout);
    v.upside_down = c.upside_down;
    if (c.part) v.part = std::string(detail::sv(c.part));
    return v;
  }

  friend bool operator==(const ImageTag& self, const ImageTag& other) noexcept(
      detail::nothrow_equal<decltype(role), decltype(layout), decltype(upside_down),
                            decltype(part)>) {
    return self.role == other.role && self.layout == other.layout &&
           self.upside_down == other.upside_down && self.part == other.part;
  }
  friend bool operator!=(const ImageTag& self,
                         const ImageTag& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct StateTag {
  std::string role{};
  OptionalList<Tolerant<RotationEncoding>> encoding{};
  std::optional<std::pair<double, double>> range{};
  std::optional<Tolerant<Frame>> frame{};
  std::optional<Tolerant<Provenance>> provenance{};
  std::optional<std::string> part{};
  OptionalList<std::string> labels{};
  std::optional<std::string> polarity{};
  RLMeshStateTag raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshStateTag c = RLMESH_STATE_TAG_INIT;
    c.role = (role).c_str();
    if (encoding) {
      std::vector<RLMeshRotationEncoding> vals;
      std::vector<const char*> others;
      for (const auto& e : *encoding) {
        const char* other = nullptr;
        vals.push_back(
            detail_adapter::tolerant<RotationEncoding, RLMeshRotationEncoding>(e, other, a));
        others.push_back(other);
      }
      c.encoding = a.array(std::move(vals));
      c.encoding_other = a.array(std::move(others));
      c.encoding_len = encoding->size();
      if (!c.encoding) c.encoding = a.keep(static_cast<RLMeshRotationEncoding>(0));
    }
    if (range) {
      c.has_range = true;
      c.range[0] = range->first;
      c.range[1] = range->second;
    }
    if (frame) {
      c.has_frame = true;
      c.frame = detail_adapter::tolerant<Frame, RLMeshFrame>(*frame, c.frame_other, a);
    }
    if (provenance) {
      c.has_provenance = true;
      c.provenance = detail_adapter::tolerant<Provenance, RLMeshProvenance>(*provenance,
                                                                            c.provenance_other, a);
    }
    if (part) {
      c.part = (*part).c_str();
    }
    if (labels) {
      c.labels = a.strings(*labels);
      c.labels_len = labels->size();
      if (!c.labels) c.labels = a.keep(static_cast<const char*>(nullptr));
    }
    c.polarity = detail::ptr(polarity);
    return c;
  }
  static StateTag from_raw(const RLMeshStateTag& c) {
    StateTag v;
    v.role = std::string(detail::sv(c.role));
    if (c.encoding) {
      std::vector<Tolerant<RotationEncoding>> vals;
      for (size_t i = 0; i < c.encoding_len; ++i)
        vals.push_back(detail_adapter::tolerant<RotationEncoding>(
            c.encoding[i], c.encoding_other ? c.encoding_other[i] : nullptr));
      v.encoding = std::move(vals);
    }
    if (c.has_range) v.range = std::pair<double, double>{c.range[0], c.range[1]};
    if (c.has_frame) v.frame = detail_adapter::tolerant<Frame>(c.frame, c.frame_other);
    if (c.has_provenance)
      v.provenance = detail_adapter::tolerant<Provenance>(c.provenance, c.provenance_other);
    if (c.part) v.part = std::string(detail::sv(c.part));
    if (c.labels) {
      std::vector<std::string> vals;
      for (size_t i = 0; i < c.labels_len; ++i) vals.emplace_back(c.labels[i]);
      v.labels = std::move(vals);
    }
    if (c.polarity) v.polarity = std::string(c.polarity);
    return v;
  }

  friend bool operator==(const StateTag& self, const StateTag& other) noexcept(
      detail::nothrow_equal<decltype(role), decltype(encoding), decltype(range), decltype(frame),
                            decltype(provenance), decltype(part), decltype(labels),
                            decltype(polarity)>) {
    return self.role == other.role && self.encoding == other.encoding &&
           self.range == other.range && self.frame == other.frame &&
           self.provenance == other.provenance && self.part == other.part &&
           self.labels == other.labels && self.polarity == other.polarity;
  }
  friend bool operator!=(const StateTag& self,
                         const StateTag& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct TextTag {
  std::string role{};
  RLMeshTextTag raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshTextTag c = RLMESH_TEXT_TAG_INIT;
    c.role = (role).c_str();
    return c;
  }
  static TextTag from_raw(const RLMeshTextTag& c) {
    TextTag v;
    v.role = std::string(detail::sv(c.role));
    return v;
  }

  friend bool operator==(const TextTag& self,
                         const TextTag& other) noexcept(detail::nothrow_equal<decltype(role)>) {
    return self.role == other.role;
  }
  friend bool operator!=(const TextTag& self,
                         const TextTag& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Field {
  std::optional<std::string> role{};
  uint32_t dim = 0;
  OptionalList<Tolerant<RotationEncoding>> encoding{};
  std::optional<std::pair<double, double>> range{};
  std::optional<Tolerant<Frame>> frame{};
  std::optional<Tolerant<Provenance>> provenance{};
  std::optional<std::string> part{};
  OptionalList<std::string> labels{};
  std::optional<std::string> polarity{};
  RLMeshField raw(detail_adapter::Arena& a) const {
    (void)a;
    RLMeshField c = RLMESH_FIELD_INIT;
    c.role = detail::ptr(role);
    c.dim = dim;
    if (encoding) {
      std::vector<RLMeshRotationEncoding> vals;
      std::vector<const char*> others;
      for (const auto& e : *encoding) {
        const char* other = nullptr;
        vals.push_back(
            detail_adapter::tolerant<RotationEncoding, RLMeshRotationEncoding>(e, other, a));
        others.push_back(other);
      }
      c.encoding = a.array(std::move(vals));
      c.encoding_other = a.array(std::move(others));
      c.encoding_len = encoding->size();
      if (!c.encoding) c.encoding = a.keep(static_cast<RLMeshRotationEncoding>(0));
    }
    if (range) {
      c.has_range = true;
      c.range[0] = range->first;
      c.range[1] = range->second;
    }
    if (frame) {
      c.has_frame = true;
      c.frame = detail_adapter::tolerant<Frame, RLMeshFrame>(*frame, c.frame_other, a);
    }
    if (provenance) {
      c.has_provenance = true;
      c.provenance = detail_adapter::tolerant<Provenance, RLMeshProvenance>(*provenance,
                                                                            c.provenance_other, a);
    }
    if (part) {
      c.part = (*part).c_str();
    }
    if (labels) {
      c.labels = a.strings(*labels);
      c.labels_len = labels->size();
      if (!c.labels) c.labels = a.keep(static_cast<const char*>(nullptr));
    }
    c.polarity = detail::ptr(polarity);
    return c;
  }
  static Field from_raw(const RLMeshField& c) {
    Field v;
    if (c.role) v.role = std::string(c.role);
    v.dim = c.dim;
    if (c.encoding) {
      std::vector<Tolerant<RotationEncoding>> vals;
      for (size_t i = 0; i < c.encoding_len; ++i)
        vals.push_back(detail_adapter::tolerant<RotationEncoding>(
            c.encoding[i], c.encoding_other ? c.encoding_other[i] : nullptr));
      v.encoding = std::move(vals);
    }
    if (c.has_range) v.range = std::pair<double, double>{c.range[0], c.range[1]};
    if (c.has_frame) v.frame = detail_adapter::tolerant<Frame>(c.frame, c.frame_other);
    if (c.has_provenance)
      v.provenance = detail_adapter::tolerant<Provenance>(c.provenance, c.provenance_other);
    if (c.part) v.part = std::string(detail::sv(c.part));
    if (c.labels) {
      std::vector<std::string> vals;
      for (size_t i = 0; i < c.labels_len; ++i) vals.emplace_back(c.labels[i]);
      v.labels = std::move(vals);
    }
    if (c.polarity) v.polarity = std::string(c.polarity);
    return v;
  }

  friend bool operator==(const Field& self, const Field& other) noexcept(
      detail::nothrow_equal<decltype(role), decltype(dim), decltype(encoding), decltype(range),
                            decltype(frame), decltype(provenance), decltype(part), decltype(labels),
                            decltype(polarity)>) {
    return self.role == other.role && self.dim == other.dim && self.encoding == other.encoding &&
           self.range == other.range && self.frame == other.frame &&
           self.provenance == other.provenance && self.part == other.part &&
           self.labels == other.labels && self.polarity == other.polarity;
  }
  friend bool operator!=(const Field& self, const Field& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
using ConcatPart = std::variant<std::string, State, Constant>;
struct Concat {
  std::vector<ConcatPart> parts{};
  std::optional<uint32_t> pad_to{};
  Tolerant<DType> dtype = DType::Float32;
  OptionalList<int64_t> reshape{};
  StateContainer container = StateContainer::Array;
  std::optional<std::pair<double, double>> clip{};
  RLMeshConcat raw(detail_adapter::Arena& a) const {
    RLMeshConcat c = RLMESH_CONCAT_INIT;
    std::vector<RLMeshConcatPart> p;
    for (const auto& v : parts)
      std::visit(
          [&](const auto& x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::string>)
              p.push_back(rlmesh_concat_part_role(x.c_str()));
            else if constexpr (std::is_same_v<T, State>)
              p.push_back(rlmesh_concat_part_state(a.keep(x.raw(a))));
            else
              p.push_back(rlmesh_concat_part_constant(a.keep(x.raw(a))));
          },
          v);
    c.parts = a.array(std::move(p));
    c.parts_len = parts.size();
    if (pad_to) {
      c.has_pad_to = true;
      c.pad_to = *pad_to;
    }
    c.dtype = detail_adapter::tolerant<DType, RLMeshDType>(dtype, c.dtype_other, a);
    if (reshape) {
      c.reshape = a.array(*reshape);
      c.reshape_len = reshape->size();
      if (!c.reshape) c.reshape = a.keep(int64_t{});
    }
    c.container = static_cast<RLMeshStateContainer>(container);
    if (clip) {
      c.has_clip = true;
      c.clip[0] = clip->first;
      c.clip[1] = clip->second;
    }
    return c;
  }
  static Concat from_raw(const RLMeshConcat& c) {
    Concat v;
    for (size_t i = 0; i < c.parts_len; ++i) {
      auto& p = c.parts[i];
      if (p.kind == RLMESH_CONCAT_PART_KIND_ROLE)
        v.parts.emplace_back(std::string(detail::sv(p.role)));
      else if (p.kind == RLMESH_CONCAT_PART_KIND_STATE)
        v.parts.emplace_back(State::from_raw(*static_cast<const RLMeshState*>(p.leaf)));
      else
        v.parts.emplace_back(Constant::from_raw(*static_cast<const RLMeshConstant*>(p.leaf)));
    }
    if (c.has_pad_to) v.pad_to = c.pad_to;
    v.dtype = detail_adapter::tolerant<DType>(c.dtype, c.dtype_other);
    if (c.reshape) v.reshape = std::vector<int64_t>(c.reshape, c.reshape + c.reshape_len);
    v.container = static_cast<StateContainer>(c.container);
    if (c.has_clip) v.clip = std::pair<double, double>{c.clip[0], c.clip[1]};
    return v;
  }

  friend bool operator==(const Concat& self, const Concat& other) noexcept(
      detail::nothrow_equal<decltype(parts), decltype(pad_to), decltype(dtype), decltype(reshape),
                            decltype(container), decltype(clip)>) {
    return self.parts == other.parts && self.pad_to == other.pad_to && self.dtype == other.dtype &&
           self.reshape == other.reshape && self.container == other.container &&
           self.clip == other.clip;
  }
  friend bool operator!=(const Concat& self,
                         const Concat& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Action {
  std::vector<Actuator> components{};
  std::optional<std::pair<double, double>> clip{};
  RLMeshAction raw(detail_adapter::Arena& a) const {
    RLMeshAction c = RLMESH_ACTION_INIT;
    std::vector<const RLMeshActuator*> p;
    for (auto& x : components) p.push_back(a.keep(x.raw(a)));
    c.components = a.array(std::move(p));
    c.components_len = components.size();
    if (clip) {
      c.has_clip = true;
      c.clip[0] = clip->first;
      c.clip[1] = clip->second;
    }
    return c;
  }
  static Action from_raw(const RLMeshAction& c) {
    Action v;
    if (c.has_clip) v.clip = std::pair<double, double>{c.clip[0], c.clip[1]};
    for (size_t i = 0; i < c.components_len; ++i)
      v.components.push_back(Actuator::from_raw(*c.components[i]));
    return v;
  }
  uint32_t dim() const {
    detail_adapter::Arena a;
    auto c = raw(a);
    uint32_t n = 0;
    detail::check_status(rlmesh_action_dim(&c, &n));
    return n;
  }

  friend bool operator==(const Action& self, const Action& other) noexcept(
      detail::nothrow_equal<decltype(components), decltype(clip)>) {
    return self.components == other.components && self.clip == other.clip;
  }
  friend bool operator!=(const Action& self,
                         const Action& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Split {
  std::vector<Field> fields{};
  RLMeshSplit raw(detail_adapter::Arena& a) const {
    RLMeshSplit c = RLMESH_SPLIT_INIT;
    std::vector<const RLMeshField*> p;
    for (auto& x : fields) p.push_back(a.keep(x.raw(a)));
    c.fields = a.array(std::move(p));
    c.fields_len = fields.size();
    return c;
  }
  static Split from_raw(const RLMeshSplit& c) {
    Split v;
    for (size_t i = 0; i < c.fields_len; ++i) v.fields.push_back(Field::from_raw(*c.fields[i]));
    return v;
  }

  friend bool operator==(const Split& self,
                         const Split& other) noexcept(detail::nothrow_equal<decltype(fields)>) {
    return self.fields == other.fields;
  }
  friend bool operator!=(const Split& self, const Split& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct PreviousOptions {
  std::optional<std::string> part{};
  double fill = 0.0;
  PreviousOptions& with_part(std::optional<std::string> value) & {
    part = std::move(value);
    return *this;
  }
  PreviousOptions with_part(std::optional<std::string> value) && {
    part = std::move(value);
    return std::move(*this);
  }
  PreviousOptions& with_fill(double value) & {
    fill = std::move(value);
    return *this;
  }
  PreviousOptions with_fill(double value) && {
    fill = std::move(value);
    return std::move(*this);
  }
};
inline State Previous(std::string role, PreviousOptions options = {}) {
  RLMeshPreviousOptions c = RLMESH_PREVIOUS_OPTIONS_INIT;
  c.part = detail::ptr(options.part);
  c.fill = options.fill;
  RLMeshState s = RLMESH_STATE_INIT;
  detail::check_status(rlmesh_adapters_previous(role.c_str(), &c, &s));
  return State::from_raw(s);
}
using ModelLeaf = std::variant<Image, State, Concat, Text, Custom>;
using ObsLeaf = std::variant<ImageTag, StateTag, TextTag, Split>;
class InputNode {
  std::variant<ModelLeaf, std::vector<std::pair<std::string, InputNode>>, std::vector<InputNode>>
      value_;

 public:
  friend std::string to_string(const InputNode&);

  template <class T, std::enable_if_t<std::is_constructible_v<ModelLeaf, T>, int> = 0>
  InputNode(T v) : value_(ModelLeaf(std::move(v))) {}
  static InputNode dict(std::vector<std::pair<std::string, InputNode>> v) {
    return InputNode(std::move(v), 0);
  }
  static InputNode tuple(std::vector<InputNode> v) { return InputNode(std::move(v), 0); }
  RLMeshInputNode raw(detail_adapter::Arena& a) const {
    if (auto p = std::get_if<ModelLeaf>(&value_))
      return std::visit(
          [&](const auto& v) -> RLMeshInputNode {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Image>)
              return rlmesh_input_node_image(a.keep(v.raw(a)));
            else if constexpr (std::is_same_v<T, State>)
              return rlmesh_input_node_state(a.keep(v.raw(a)));
            else if constexpr (std::is_same_v<T, Concat>)
              return rlmesh_input_node_concat(a.keep(v.raw(a)));
            else if constexpr (std::is_same_v<T, Text>)
              return rlmesh_input_node_text(a.keep(v.raw(a)));
            else if constexpr (std::is_same_v<T, Custom>)
              return rlmesh_input_node_custom(a.keep(v.raw(a)));
          },
          *p);
    std::vector<RLMeshInputNode> children;
    std::vector<RLMeshStr> keys;
    if (auto p = std::get_if<std::vector<std::pair<std::string, InputNode>>>(&value_)) {
      for (auto& [k, v] : *p) {
        keys.push_back(detail::str(k));
        children.push_back(v.raw(a));
      }
      auto n = children.size();
      return rlmesh_input_node_dict(a.array(std::move(keys)), a.array(std::move(children)), n);
    }
    for (auto& v : std::get<std::vector<InputNode>>(value_)) children.push_back(v.raw(a));
    auto n = children.size();
    return rlmesh_input_node_tuple(a.array(std::move(children)), n);
  }
  static InputNode from_raw(const RLMeshInputNode& c) {
    switch (c.kind) {
      case RLMESH_INPUT_NODE_KIND_IMAGE:
        return Image::from_raw(*static_cast<const RLMeshImage*>(c.leaf));
      case RLMESH_INPUT_NODE_KIND_STATE:
        return State::from_raw(*static_cast<const RLMeshState*>(c.leaf));
      case RLMESH_INPUT_NODE_KIND_CONCAT:
        return Concat::from_raw(*static_cast<const RLMeshConcat*>(c.leaf));
      case RLMESH_INPUT_NODE_KIND_TEXT:
        return Text::from_raw(*static_cast<const RLMeshAdaptersText*>(c.leaf));
      case RLMESH_INPUT_NODE_KIND_CUSTOM:
        return Custom::from_raw(*static_cast<const RLMeshCustom*>(c.leaf));
      case RLMESH_INPUT_NODE_KIND_DICT: {
        std::vector<std::pair<std::string, InputNode>> v;
        for (size_t i = 0; i < c.children_len; ++i)
          v.emplace_back(std::string(c.keys[i].data, c.keys[i].len), from_raw(c.children[i]));
        return dict(std::move(v));
      }
      case RLMESH_INPUT_NODE_KIND_TUPLE: {
        std::vector<InputNode> v;
        for (size_t i = 0; i < c.children_len; ++i) v.push_back(from_raw(c.children[i]));
        return tuple(std::move(v));
      }
      default:
        throw TypeError("invalid adapter node");
    }
  }

 private:
  template <class T>
  InputNode(T v, int) : value_(std::move(v)) {}

 public:
  friend bool operator==(const InputNode& self, const InputNode& other) {
    if (auto left = std::get_if<1>(&self.value_)) {
      auto right = std::get_if<1>(&other.value_);
      return right && detail_adapter::dict_equal(*left, *right);
    }
    return self.value_ == other.value_;
  }
  friend bool operator!=(const InputNode& self,
                         const InputNode& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
class ObsNode {
  std::variant<ObsLeaf, std::vector<std::pair<std::string, ObsNode>>, std::vector<ObsNode>>
      value_{};

 public:
  friend std::string to_string(const ObsNode&);

  template <class T, std::enable_if_t<std::is_constructible_v<ObsLeaf, T>, int> = 0>
  ObsNode(T v) : value_(ObsLeaf(std::move(v))) {}
  static ObsNode dict(std::vector<std::pair<std::string, ObsNode>> v) {
    return ObsNode(std::move(v), 0);
  }
  static ObsNode tuple(std::vector<ObsNode> v) { return ObsNode(std::move(v), 0); }
  RLMeshObsNode raw(detail_adapter::Arena& a) const {
    if (auto p = std::get_if<ObsLeaf>(&value_))
      return std::visit(
          [&](const auto& v) -> RLMeshObsNode {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, ImageTag>)
              return rlmesh_obs_node_image_tag(a.keep(v.raw(a)));
            else if constexpr (std::is_same_v<T, StateTag>)
              return rlmesh_obs_node_state_tag(a.keep(v.raw(a)));
            else if constexpr (std::is_same_v<T, TextTag>)
              return rlmesh_obs_node_text_tag(a.keep(v.raw(a)));
            else if constexpr (std::is_same_v<T, Split>)
              return rlmesh_obs_node_split(a.keep(v.raw(a)));
          },
          *p);
    std::vector<RLMeshObsNode> children;
    std::vector<RLMeshStr> keys;
    if (auto p = std::get_if<std::vector<std::pair<std::string, ObsNode>>>(&value_)) {
      for (auto& [k, v] : *p) {
        keys.push_back(detail::str(k));
        children.push_back(v.raw(a));
      }
      auto n = children.size();
      return rlmesh_obs_node_dict(a.array(std::move(keys)), a.array(std::move(children)), n);
    }
    for (auto& v : std::get<std::vector<ObsNode>>(value_)) children.push_back(v.raw(a));
    auto n = children.size();
    return rlmesh_obs_node_tuple(a.array(std::move(children)), n);
  }
  static ObsNode from_raw(const RLMeshObsNode& c) {
    switch (c.kind) {
      case RLMESH_OBS_NODE_KIND_IMAGE_TAG:
        return ImageTag::from_raw(*static_cast<const RLMeshImageTag*>(c.leaf));
      case RLMESH_OBS_NODE_KIND_STATE_TAG:
        return StateTag::from_raw(*static_cast<const RLMeshStateTag*>(c.leaf));
      case RLMESH_OBS_NODE_KIND_TEXT_TAG:
        return TextTag::from_raw(*static_cast<const RLMeshTextTag*>(c.leaf));
      case RLMESH_OBS_NODE_KIND_SPLIT:
        return Split::from_raw(*static_cast<const RLMeshSplit*>(c.leaf));
      case RLMESH_OBS_NODE_KIND_DICT: {
        std::vector<std::pair<std::string, ObsNode>> v;
        for (size_t i = 0; i < c.children_len; ++i)
          v.emplace_back(std::string(c.keys[i].data, c.keys[i].len), from_raw(c.children[i]));
        return dict(std::move(v));
      }
      case RLMESH_OBS_NODE_KIND_TUPLE: {
        std::vector<ObsNode> v;
        for (size_t i = 0; i < c.children_len; ++i) v.push_back(from_raw(c.children[i]));
        return tuple(std::move(v));
      }
      default:
        throw TypeError("invalid adapter node");
    }
  }

 private:
  template <class T>
  ObsNode(T v, int) : value_(std::move(v)) {}

 public:
  friend bool operator==(const ObsNode& self, const ObsNode& other) {
    if (auto left = std::get_if<1>(&self.value_)) {
      auto right = std::get_if<1>(&other.value_);
      return right && detail_adapter::dict_equal(*left, *right);
    }
    return self.value_ == other.value_;
  }
  friend bool operator!=(const ObsNode& self,
                         const ObsNode& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct ObservationRoles {
  std::vector<std::string> images{}, states{}, texts{};

  static ObservationRoles from_raw(const RLMeshObservationRoles& c) {
    ObservationRoles v;
    for (size_t i = 0; i < c.images_len; ++i) v.images.emplace_back(c.images[i]);
    for (size_t i = 0; i < c.states_len; ++i) v.states.emplace_back(c.states[i]);
    for (size_t i = 0; i < c.texts_len; ++i) v.texts.emplace_back(c.texts[i]);
    return v;
  }

  friend bool operator==(const ObservationRoles& self, const ObservationRoles& other) noexcept(
      detail::nothrow_equal<decltype(images), decltype(states), decltype(texts)>) {
    return self.images == other.images && self.states == other.states && self.texts == other.texts;
  }
  friend bool operator!=(const ObservationRoles& self,
                         const ObservationRoles& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct Advisory {
  AdvisorySeverity severity = AdvisorySeverity::Info;
  std::string message{};

  RLMeshAdvisory raw() const {
    RLMeshAdvisory c = RLMESH_ADVISORY_INIT;
    c.severity = static_cast<RLMeshAdvisorySeverity>(severity);
    c.message = message.c_str();
    return c;
  }
  std::string repr() const {
    auto c = raw();
    return detail::text(rlmesh_advisory_repr, &c);
  }

  friend bool operator==(const Advisory& self, const Advisory& other) noexcept(
      detail::nothrow_equal<decltype(severity), decltype(message)>) {
    return self.severity == other.severity && self.message == other.message;
  }
  friend bool operator!=(const Advisory& self,
                         const Advisory& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const Advisory& a) { return a.message; }
inline std::ostream& operator<<(std::ostream& s, const Advisory& a) { return s << a.message; }
// How a model paces its actions: execution is the execution horizon a run
// uses when neither the run nor the model's load() names one (1..=1024).
struct Timing {
  std::optional<uint32_t> execution{};
  Timing& with_execution(std::optional<uint32_t> value) & {
    execution = value;
    return *this;
  }
  Timing with_execution(std::optional<uint32_t> value) && {
    execution = value;
    return std::move(*this);
  }
  friend bool operator==(const Timing& a, const Timing& b) noexcept {
    return a.execution == b.execution;
  }
  friend bool operator!=(const Timing& a, const Timing& b) noexcept { return !(a == b); }
};
class ModelSpec {
  std::shared_ptr<RLMeshModelSpec> handle_{};
  InputNode input_;
  Action output_{};

 public:
  explicit ModelSpec(RLMeshModelSpec* p)
      : handle_(p, rlmesh_model_spec_free),
        input_(InputNode::from_raw(*rlmesh_model_spec_input(p))),
        output_(Action::from_raw(*rlmesh_model_spec_output(p))) {}
  // allow names the projections (lossy fits) the spec accepts: drop_action,
  // select_state, rebind_camera, flip_gripper, convert_units.
  ModelSpec(InputNode input, Action output, std::vector<std::string> allow = {}, Timing timing = {})
      : input_(std::move(input)), output_(std::move(output)) {
    detail_adapter::Arena a;
    auto raw_input = input_.raw(a);
    auto raw_output = output_.raw(a);
    RLMeshModelSpec* p = nullptr;
    detail::check_status(rlmesh_model_spec_new(&raw_input, &raw_output, &p));
    handle_ = {p, rlmesh_model_spec_free};
    if (!allow.empty()) {
      std::vector<const char*> names;
      for (const auto& name : allow) names.push_back(name.c_str());
      RLMeshModelSpec* q = nullptr;
      detail::check_status(
          rlmesh_model_spec_with_allow(handle_.get(), names.data(), names.size(), &q));
      handle_ = {q, rlmesh_model_spec_free};
    }
    if (timing.execution) {
      RLMeshTiming c = RLMESH_TIMING_INIT;
      c.has_execution = true;
      c.execution = *timing.execution;
      RLMeshModelSpec* q = nullptr;
      detail::check_status(rlmesh_model_spec_with_timing(handle_.get(), &c, &q));
      handle_ = {q, rlmesh_model_spec_free};
    }
  }
  const RLMeshModelSpec* raw() const { return handle_.get(); }
  const InputNode& input() const { return input_; }
  const Action& output() const { return output_; }
  // How this spec paces its actions.
  Timing timing() const {
    RLMeshTiming c = RLMESH_TIMING_INIT;
    detail::check_status(rlmesh_model_spec_timing(raw(), &c));
    Timing t;
    if (c.has_execution) t.execution = c.execution;
    return t;
  }
  // The named projections this spec accepts.
  std::vector<std::string> allow() const {
    size_t n = 0;
    const char* const* items = rlmesh_model_spec_allow(raw(), &n);
    return std::vector<std::string>(items, items + n);
  }
  Metadata to_dict() const {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_model_spec_to_dict(raw(), &p));
    return Value::adopt(p).as_dict();
  }
  static ModelSpec from_dict(const Metadata& data) {
    Value value(data);
    RLMeshModelSpec* p = nullptr;
    detail::check_status(rlmesh_model_spec_from_dict(value.raw(), &p));
    return ModelSpec(p);
  }
  Metadata to_metadata() const {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_model_spec_to_metadata(raw(), &p));
    return Value::adopt(p).as_dict();
  }
  static std::optional<ModelSpec> from_metadata(const Metadata& metadata) {
    Value value(metadata);
    RLMeshModelSpec* p = nullptr;
    detail::check_status(rlmesh_model_spec_from_metadata(value.raw(), &p));
    if (!p) return {};
    return ModelSpec(p);
  }
  std::string to_json() const { return detail::text(rlmesh_model_spec_to_json, raw()); }
  static ModelSpec from_json(std::string_view payload) {
    RLMeshModelSpec* p = nullptr;
    detail::check_status(rlmesh_model_spec_from_json(std::string(payload).c_str(), &p));
    return ModelSpec(p);
  }

  std::string to_string() const { return detail::text(rlmesh_model_spec_repr, raw()); }

  friend bool operator==(const ModelSpec& self, const ModelSpec& v) {
    return rlmesh_model_spec_equals(self.raw(), v.raw());
  }
  friend bool operator!=(const ModelSpec& self,
                         const ModelSpec& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const ModelSpec& v) { return v.to_string(); }
inline std::ostream& operator<<(std::ostream& s, const ModelSpec& v) { return s << v.to_string(); }
class EnvTags {
  std::shared_ptr<RLMeshEnvTags> handle_{};
  ObsNode observation_;
  Action action_{};

 public:
  explicit EnvTags(RLMeshEnvTags* p)
      : handle_(p, rlmesh_env_tags_free),
        observation_(ObsNode::from_raw(*rlmesh_env_tags_observation(p))),
        action_(Action::from_raw(*rlmesh_env_tags_action(p))) {}
  EnvTags(ObsNode observation, Action action)
      : observation_(std::move(observation)), action_(std::move(action)) {
    detail_adapter::Arena a;
    auto raw_observation = observation_.raw(a);
    auto raw_action = action_.raw(a);
    RLMeshEnvTags* p = nullptr;
    detail::check_status(rlmesh_env_tags_new(&raw_observation, &raw_action, &p));
    handle_ = {p, rlmesh_env_tags_free};
  }
  const RLMeshEnvTags* raw() const { return handle_.get(); }
  const ObsNode& observation() const { return observation_; }
  const Action& action() const { return action_; }
  Metadata to_dict() const {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_env_tags_to_dict(raw(), &p));
    return Value::adopt(p).as_dict();
  }
  static EnvTags from_dict(const Metadata& data) {
    Value value(data);
    RLMeshEnvTags* p = nullptr;
    detail::check_status(rlmesh_env_tags_from_dict(value.raw(), &p));
    return EnvTags(p);
  }
  Metadata to_metadata() const {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_env_tags_to_metadata(raw(), &p));
    return Value::adopt(p).as_dict();
  }
  static std::optional<EnvTags> from_metadata(const Metadata& metadata) {
    Value value(metadata);
    RLMeshEnvTags* p = nullptr;
    detail::check_status(rlmesh_env_tags_from_metadata(value.raw(), &p));
    if (!p) return {};
    return EnvTags(p);
  }
  std::string to_json() const { return detail::text(rlmesh_env_tags_to_json, raw()); }
  static EnvTags from_json(std::string_view payload) {
    RLMeshEnvTags* p = nullptr;
    detail::check_status(rlmesh_env_tags_from_json(std::string(payload).c_str(), &p));
    return EnvTags(p);
  }

  std::string to_string() const { return detail::text(rlmesh_env_tags_repr, raw()); }
  ObservationRoles observation_roles() const {
    RLMeshObservationRoles c = RLMESH_OBSERVATION_ROLES_INIT;
    detail::check_status(rlmesh_env_tags_observation_roles(raw(), &c));
    return ObservationRoles::from_raw(c);
  }

  friend bool operator==(const EnvTags& self, const EnvTags& v) {
    return rlmesh_env_tags_equals(self.raw(), v.raw());
  }
  friend bool operator!=(const EnvTags& self,
                         const EnvTags& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const EnvTags& v) { return v.to_string(); }
inline std::ostream& operator<<(std::ostream& s, const EnvTags& v) { return s << v.to_string(); }
inline constexpr const char* IMAGE_PRIMARY = "image/primary";
inline constexpr const char* IMAGE_SECONDARY = "image/secondary";
inline constexpr const char* INSTRUCTION = "text/instruction";
inline constexpr const char* JOINT_POS = "proprio/joint_pos";
inline constexpr const char* JOINT_VEL = "proprio/joint_vel";
inline constexpr const char* ACTION_JOINT_POS = "action/joint_pos";
inline constexpr const char* ACTION_JOINT_VEL = "action/joint_vel";
inline constexpr const char* IMAGE_WRIST = "image/wrist";
inline constexpr const char* EEF_POS = "proprio/eef_pos";
inline constexpr const char* EEF_ROT = "proprio/eef_rot";
inline constexpr const char* GRIPPER_POS = "proprio/gripper";
inline constexpr const char* EEF_WRENCH = "proprio/eef_wrench";
inline constexpr const char* ACTION_DELTA_POS = "action/delta_eef_pos";
inline constexpr const char* ACTION_DELTA_ROT = "action/delta_eef_rot";
inline constexpr const char* ACTION_GRIPPER = "action/gripper";
inline constexpr const char* ACTION_EEF_POS = "action/eef_pos";
inline constexpr const char* ACTION_EEF_ROT = "action/eef_rot";
inline constexpr const char* BASE_ANG_VEL = "proprio/base_ang_vel";
inline constexpr const char* BASE_ROT = "proprio/base_rot";
inline constexpr const char* COMMAND_BASE_VEL = "command/base_vel";
inline constexpr const char* LEFT_ARM = "left_arm";
inline constexpr const char* RIGHT_ARM = "right_arm";
inline constexpr const char* HEAD = "head";
inline constexpr const char* TORSO = "torso";
inline constexpr const char* BASE = "base";
inline constexpr const char* LEFT_LEG = "left_leg";
inline constexpr const char* RIGHT_LEG = "right_leg";
inline constexpr const char* ENV_METADATA_KEY = "rlmesh.adapters.v1.env_tags";
inline constexpr const char* MODEL_METADATA_KEY = "rlmesh.adapters.v1.model_spec";
inline constexpr const char* ENV_BRANCH_METADATA_KEY = "rlmesh.adapters.v1.env_branch";
inline const std::vector<const char*>& PARTS() {
  static const auto values = [] {
    std::vector<const char*> v{};
    for (size_t i = 0; i < rlmesh_adapters_parts_len(); ++i)
      v.push_back(rlmesh_adapters_parts_get(i));
    return v;
  }();
  return values;
}
inline const std::vector<std::pair<RotationEncoding, uint32_t>>& ROTATION_DIMS() {
  static const auto values = [] {
    std::vector<std::pair<RotationEncoding, uint32_t>> v{};
    for (size_t i = 0; i < rlmesh_adapters_rotation_dims_len(); ++i) {
      RLMeshRotationEncoding e{};
      uint32_t n = 0;
      detail::check_status(rlmesh_adapters_rotation_dims_get(i, &e, &n));
      v.emplace_back(static_cast<RotationEncoding>(e), n);
    }
    return v;
  }();
  return values;
}
class AdapterBase : public std::enable_shared_from_this<AdapterBase> {
 public:
  virtual ~AdapterBase() = default;
  virtual Value transform_obs([[maybe_unused]] const Value& raw_obs) = 0;
  virtual Value transform_action([[maybe_unused]] const Value& raw_action) = 0;
  virtual void reset() {}
  virtual void observe([[maybe_unused]] const Value& raw_obs) {}
  virtual std::string explain() const { return "AdapterBase (custom adapter)"; }
  virtual std::function<Value(const Value&)> wrap_predict(
      std::function<Value(const Value&)> predict_fn) {
    std::shared_ptr<AdapterBase> self;
    try {
      self = shared_from_this();
    } catch (const std::bad_weak_ptr&) {
      throw RuntimeError("wrap_predict requires shared ownership");
    }
    return [self = std::move(self), predict_fn = std::move(predict_fn)](const Value& v) {
      self->observe(v);
      return self->transform_action(predict_fn(self->transform_obs(v)));
    };
  }
};
class Adapter : public AdapterBase {
  std::shared_ptr<RLMeshAdapter> handle_{};

 public:
  explicit Adapter(RLMeshAdapter* p) : handle_(p, rlmesh_adapter_free) {}
  explicit Adapter(std::shared_ptr<AdapterBase> p) {
    struct AdapterState {
      std::shared_ptr<AdapterBase> value;
      std::string explanation;
    };
    auto state = std::make_unique<AdapterState>();
    state->value = std::move(p);
    if (!state->value) throw TypeError("adapter must not be null");
    RLMeshAdapterVtable c = RLMESH_ADAPTER_VTABLE_INIT;
    c.transform_obs = [](void* u, const RLMeshValue* v, RLMeshValue** o) noexcept {
      return detail::callback([&] {
        *o = static_cast<AdapterState*>(u)->value->transform_obs(Value::copy_of(v)).clone_raw();
      });
    };
    c.transform_action = [](void* u, const RLMeshValue* v, RLMeshValue** o) noexcept {
      return detail::callback([&] {
        *o = static_cast<AdapterState*>(u)->value->transform_action(Value::copy_of(v)).clone_raw();
      });
    };
    c.reset = [](void* u) noexcept {
      return detail::callback([&] { static_cast<AdapterState*>(u)->value->reset(); });
    };
    c.observe = [](void* u, const RLMeshValue* v) noexcept {
      return detail::callback(
          [&] { static_cast<AdapterState*>(u)->value->observe(Value::copy_of(v)); });
    };
    c.explain = [](void* u, const char** o) noexcept {
      return detail::callback([&] {
        auto& s = *static_cast<AdapterState*>(u);
        s.explanation = s.value->explain();
        *o = s.explanation.c_str();
      });
    };
    c.free_user_data = detail::delete_state<AdapterState>;
    RLMeshAdapter* h = nullptr;
    detail::check_status(rlmesh_adapter_new(&c, state.get(), &h));
    state.release();
    handle_ = {h, rlmesh_adapter_free};
  }
  RLMeshAdapter* raw() const { return handle_.get(); }
  Value transform_obs(const Value& raw_obs) override {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_adapter_transform_obs(raw(), raw_obs.raw(), &p));
    return Value::adopt(p);
  }
  Value transform_action(const Value& raw_action) override {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_adapter_transform_action(raw(), raw_action.raw(), &p));
    return Value::adopt(p);
  }
  void observe(const Value& raw_obs) override {
    detail::check_status(rlmesh_adapter_observe(raw(), raw_obs.raw()));
  }
  void reset() override { detail::check_status(rlmesh_adapter_reset(raw())); }
  std::string explain() const override { return detail::text(rlmesh_adapter_explain, raw()); }
  std::vector<Advisory> advisories() const {
    size_t n = 0;
    detail::check_status(rlmesh_adapter_advisories_len(raw(), &n));
    std::vector<Advisory> v;
    for (size_t i = 0; i < n; ++i) {
      RLMeshAdvisory c = RLMESH_ADVISORY_INIT;
      detail::check_status(rlmesh_adapter_advisories_get(raw(), i, &c));
      v.push_back({static_cast<AdvisorySeverity>(c.severity), std::string(detail::sv(c.message))});
    }
    return v;
  }
  std::vector<std::string> history_keys() const {
    size_t n = 0;
    detail::check_status(rlmesh_adapter_history_keys_len(raw(), &n));
    std::vector<std::string> v;
    for (size_t i = 0; i < n; ++i) {
      RLMeshStr s{};
      detail::check_status(rlmesh_adapter_history_keys_get(raw(), i, &s));
      v.emplace_back(s.data, s.len);
    }
    return v;
  }
  std::vector<std::tuple<std::string, uint64_t, uint64_t>> history_windows() const {
    size_t n = 0;
    detail::check_status(rlmesh_adapter_history_windows_len(raw(), &n));
    std::vector<std::tuple<std::string, uint64_t, uint64_t>> v;
    for (size_t i = 0; i < n; ++i) {
      RLMeshStr s{};
      uint64_t span = 0, bytes = 0;
      detail::check_status(rlmesh_adapter_history_windows_get(raw(), i, &s, &span, &bytes));
      v.emplace_back(std::string(s.data, s.len), span, bytes);
    }
    return v;
  }
  std::function<Value(const Value&)> wrap_predict(
      std::function<Value(const Value&)> predict_fn) final {
    auto state = std::make_unique<std::function<Value(const Value&)>>(std::move(predict_fn));
    RLMeshWrappedPredict* p = nullptr;
    detail::check_status(rlmesh_adapter_wrap_predict(
        raw(),
        [](void* u, const RLMeshValue* v, RLMeshValue** o) noexcept {
          return detail::callback([&] {
            *o = (*static_cast<std::function<Value(const Value&)>*>(u))(Value::copy_of(v))
                     .clone_raw();
          });
        },
        state.get(), detail::delete_state<std::function<Value(const Value&)>>, &p));
    state.release();
    auto owned = std::shared_ptr<RLMeshWrappedPredict>(p, rlmesh_wrapped_predict_free);
    return [owned](const Value& v) {
      RLMeshValue* o = nullptr;
      detail::check_status(rlmesh_wrapped_predict_call(owned.get(), v.raw(), &o));
      return Value::adopt(o);
    };
  }
};
struct ResolveOptions {
  bool trust_entrypoints = false;
  bool check_inverse = true;
  ResolveOptions& with_trust_entrypoints(bool value) & {
    trust_entrypoints = std::move(value);
    return *this;
  }
  ResolveOptions with_trust_entrypoints(bool value) && {
    trust_entrypoints = std::move(value);
    return std::move(*this);
  }
  ResolveOptions& with_check_inverse(bool value) & {
    check_inverse = std::move(value);
    return *this;
  }
  ResolveOptions with_check_inverse(bool value) && {
    check_inverse = std::move(value);
    return std::move(*this);
  }
};
inline Adapter resolve(const EnvTags& env_tags, const spaces::SpaceSpec& observation_space,
                       const spaces::SpaceSpec& action_space, const ModelSpec& model_spec,
                       ResolveOptions options = {}) {
  RLMeshResolveOptions c = RLMESH_RESOLVE_OPTIONS_INIT;
  c.trust_entrypoints = options.trust_entrypoints;
  c.check_inverse = options.check_inverse;
  RLMeshAdapter* p = nullptr;
  detail::check_status(rlmesh_adapters_resolve(env_tags.raw(), observation_space.raw(),
                                               action_space.raw(), model_spec.raw(), &c, &p));
  return Adapter(p);
}
inline Adapter resolve_from_contract(const specs::EnvContract& contract,
                                     const ModelSpec& model_spec, ResolveOptions options = {}) {
  RLMeshResolveOptions c = RLMESH_RESOLVE_OPTIONS_INIT;
  c.trust_entrypoints = options.trust_entrypoints;
  c.check_inverse = options.check_inverse;
  RLMeshAdapter* p = nullptr;
  detail::check_status(
      rlmesh_adapters_resolve_from_contract(contract.raw(), model_spec.raw(), &c, &p));
  return Adapter(p);
}
using AdapterResolutionError = rlmesh::AdapterResolutionError;
}  // namespace adapters
namespace adapt = adapters;
struct NoAdapter {
  friend constexpr bool operator==(const NoAdapter&, const NoAdapter&) noexcept { return true; }
  friend constexpr bool operator!=(const NoAdapter& self,
                                   const NoAdapter& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline constexpr NoAdapter NO_ADAPTER{};
using SpecArg = std::variant<adapters::ModelSpec, NoAdapter>;
namespace detail {
inline RLMeshSpecArg spec_arg(const std::optional<SpecArg>& v) {
  if (!v) return {RLMESH_SPEC_KIND_NONE, nullptr};
  if (auto p = std::get_if<adapters::ModelSpec>(&*v))
    return {RLMESH_SPEC_KIND_MODEL_SPEC, p->raw()};
  return {RLMESH_SPEC_KIND_NO_ADAPTER, nullptr};
}
}  // namespace detail
using Duration = std::chrono::duration<double>;
using ResetResult = std::tuple<Value, InfoDict>;
using StepResult = std::tuple<Value, double, bool, bool, InfoDict>;
using VectorStepResult =
    std::tuple<Value, std::vector<double>, std::vector<bool>, std::vector<bool>, InfoDict>;
using VectorSeed = std::variant<int64_t, std::vector<int64_t>>;
namespace detail {
template <class T>
using id_expression = decltype((void)std::declval<T&>().id());
template <class T>
using render_mode_expression = decltype((void)std::declval<T&>().render_mode());
template <class T>
using metadata_expression = decltype((void)std::declval<T&>().metadata());
template <class T>
using render_expression = decltype((void)std::declval<T&>().render());
template <class T>
using close_expression = decltype((void)std::declval<T&>().close());
template <class T>
using prepare_expression = decltype((void)std::declval<T&>().prepare());
template <class T>
using tags_expression = decltype((void)T::tags());
template <class T>
using reset_options_expression = decltype((void)T::reset_options());
template <class T>
using workflow_edition_expression = decltype((void)T::workflow_edition());
template <class T>
using tag_params_expression = decltype((void)T::tag_params());
template <class T>
using spec_expression = decltype((void)T::spec());
template <class T>
using allow_fusion_expression = decltype(T::allow_fusion);
template <class T>
using native_chunk_expression = decltype(T::native_chunk);
template <class T>
using load_keys_expression = decltype(T::load_keys);
template <class T>
using reset_member_expression = decltype(&T::reset);
template <class T>
using reset_expression = decltype((void)std::declval<T&>().reset());
template <class T>
using reset_id_expression = decltype((void)std::declval<T&>().reset(std::string_view{}));
template <class T>
using load_member_expression = decltype(&T::load);
template <class T>
using load_expression = decltype((void)std::declval<T&>().load());
template <class T>
using load_kwargs_expression =
    decltype((void)std::declval<T&>().load(std::declval<const Kwargs&>()));
template <class T>
using on_episode_end_member_expression = decltype(&T::on_episode_end);
template <class T>
using on_episode_end_expression = decltype((void)std::declval<T&>().on_episode_end());
template <class T>
using on_episode_end_id_expression =
    decltype((void)std::declval<T&>().on_episode_end(std::string_view{}));
// Validation used a named lvalue; dispatch below deliberately tests a prvalue.
template <class T>
using on_episode_end_lvalue_expression =
    decltype((void)std::declval<T&>().on_episode_end(std::declval<std::string_view&>()));
template <class T>
using loaded_native_chunk_expression = decltype((void)std::declval<T&>().loaded_native_chunk());
template <class T>
using loaded_execution_horizon_expression =
    decltype((void)std::declval<T&>().loaded_execution_horizon());
template <class T>
using policy_reset_options_expression =
    decltype((void)std::declval<T&>().reset_options(std::declval<const InfoDict&>()));
template <class From, class To, class = void>
struct ConvertibleTo : std::false_type {};
template <class From, class To>
struct ConvertibleTo<From, To, std::void_t<decltype(static_cast<To>(std::declval<From>()))>>
    : std::is_convertible<From, To> {};
template <class From, class To>
inline constexpr bool convertible_to = ConvertibleTo<From, To>::value;
template <class E>
using ScalarEnv_expression_0 = decltype(std::declval<E&>().observation_space());
template <class E>
using ScalarEnv_expression_1 = decltype(std::declval<E&>().action_space());
template <class E>
using ScalarEnv_expression_2 = decltype(std::declval<E&>().reset(
    std::declval<std::optional<int64_t>&>(), std::declval<const std::optional<InfoDict>&>()));
template <class E>
using ScalarEnv_expression_3 =
    decltype(std::declval<E&>().step(std::declval<std::optional<Value>&>()));
template <class E, class = void>
struct ScalarEnv_trait : std::false_type {};
template <class E>
struct ScalarEnv_trait<E, std::void_t<ScalarEnv_expression_0<E>, ScalarEnv_expression_1<E>,
                                      ScalarEnv_expression_2<E>, ScalarEnv_expression_3<E>>>
    : std::bool_constant<convertible_to<ScalarEnv_expression_0<E>, spaces::Space> &&
                         convertible_to<ScalarEnv_expression_1<E>, spaces::Space> &&
                         std::is_same_v<ScalarEnv_expression_2<E>, ResetResult> &&
                         std::is_same_v<ScalarEnv_expression_3<E>, StepResult>> {};
template <class E>
inline constexpr bool ScalarEnv = ScalarEnv_trait<E>::value;
template <class E>
using BatchedEnv_expression_0 = decltype(std::declval<E&>().single_observation_space());
template <class E>
using BatchedEnv_expression_1 = decltype(std::declval<E&>().single_action_space());
template <class E>
using BatchedEnv_expression_2 = decltype(std::declval<E&>().num_envs());
template <class E>
using BatchedEnv_expression_3 = decltype(std::declval<E&>().reset(
    std::declval<std::optional<VectorSeed>&>(), std::declval<const std::optional<InfoDict>&>()));
template <class E>
using BatchedEnv_expression_4 =
    decltype(std::declval<E&>().step(std::declval<std::optional<Value>&>()));
template <class E, class = void>
struct BatchedEnv_trait : std::false_type {};
template <class E>
struct BatchedEnv_trait<E, std::void_t<BatchedEnv_expression_0<E>, BatchedEnv_expression_1<E>,
                                       BatchedEnv_expression_2<E>, BatchedEnv_expression_3<E>,
                                       BatchedEnv_expression_4<E>>>
    : std::bool_constant<convertible_to<BatchedEnv_expression_0<E>, spaces::Space> &&
                         convertible_to<BatchedEnv_expression_1<E>, spaces::Space> &&
                         convertible_to<BatchedEnv_expression_2<E>, uint64_t> &&
                         std::is_same_v<BatchedEnv_expression_3<E>, ResetResult> &&
                         std::is_same_v<BatchedEnv_expression_4<E>, VectorStepResult>> {};
template <class E>
inline constexpr bool BatchedEnv = BatchedEnv_trait<E>::value;
template <class N>
uint64_t checked_num_envs(N count) {
  if constexpr (std::is_signed_v<N>)
    if (count < 0) throw ValueError("num_envs must not be negative");
  return static_cast<uint64_t>(count);
}
inline std::optional<InfoDict> optional_info(const RLMeshValue* p) {
  if (p) return info(p);
  return {};
}
inline std::optional<Value> optional_value(const RLMeshValue* p) {
  if (p) return Value::copy_of(p);
  return {};
}
}  // namespace detail
class Env {
  detail::Owned<RLMeshEnv, rlmesh_env_free> handle_{nullptr, rlmesh_env_free};

 public:
  explicit Env(RLMeshEnv* p) : handle_(p, rlmesh_env_free) {}
  Env(Env&&) = default;
  Env& operator=(Env&&) = default;
  template <class E, std::enable_if_t<detail::ScalarEnv<detail::remove_cvref_t<E>>, int> = 0>
  explicit Env(E&& e) : Env(std::make_shared<detail::remove_cvref_t<E>>(std::forward<E>(e))) {}
  template <class E, std::enable_if_t<detail::ScalarEnv<E>, int> = 0>
  explicit Env(std::shared_ptr<E> source) {
    static_assert(detail::ScalarEnv<E>,
                  "rlmesh::Env: incompatible observation/action spaces or "
                  "reset/step signatures");
    if (!source) throw TypeError("env must not be null");
    auto state = std::make_unique<std::shared_ptr<E>>(std::move(source));
    auto& object = **state;
    auto observation_space = object.observation_space();
    auto action_space = object.action_space();
    RLMeshEnvAttributes attrs = RLMESH_ENV_ATTRIBUTES_INIT;
    attrs.observation_space = observation_space.raw();
    attrs.action_space = action_space.raw();
    std::optional<std::string> id, render_mode;
    std::optional<Metadata> metadata;
    std::optional<Value> metadata_value;
    if constexpr (detail::is_detected<detail::id_expression, E>) {
      id = object.id();
      attrs.id = detail::ptr(id);
    }
    if constexpr (detail::is_detected<detail::render_mode_expression, E>) {
      render_mode = object.render_mode();
      attrs.render_mode = detail::ptr(render_mode);
    }
    if constexpr (detail::is_detected<detail::metadata_expression, E>) {
      metadata = object.metadata();
      if (metadata) {
        metadata_value = Value(*metadata);
        attrs.metadata = metadata_value->raw();
      }
    }
    RLMeshEnvVtable vt = RLMESH_ENV_VTABLE_INIT;
    vt.reset = [](void* u, const int64_t* seed, const RLMeshValue* options, RLMeshValue** obs,
                  RLMeshValue** info) noexcept {
      return detail::callback(
          [&] {
            auto& e = **static_cast<std::shared_ptr<E>*>(u);
            auto [o, i] =
                e.reset(seed ? std::optional(*seed) : std::nullopt, detail::optional_info(options));
            *obs = o.clone_raw();
            *info = Value(i).clone_raw();
          },
          RLMESH_ERR_ENVIRONMENT);
    };
    vt.step = [](void* u, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                 bool* terminated, bool* truncated, RLMeshValue** info) noexcept {
      return detail::callback(
          [&] {
            auto& e = **static_cast<std::shared_ptr<E>*>(u);
            auto [o, r, t, tr, i] = e.step(detail::optional_value(action));
            *reward = r;
            *terminated = t;
            *truncated = tr;
            *obs = o.clone_raw();
            *info = Value(i).clone_raw();
          },
          RLMESH_ERR_ENVIRONMENT);
    };
    if constexpr (detail::is_detected<detail::render_expression, E>)
      vt.render = [](void* u, RLMeshValue** out) noexcept {
        return detail::callback(
            [&] {
              auto v = (**static_cast<std::shared_ptr<E>*>(u)).render();
              if (v) *out = v->clone_raw();
            },
            RLMESH_ERR_ENVIRONMENT);
      };
    if constexpr (detail::is_detected<detail::close_expression, E>)
      vt.close = [](void* u) noexcept {
        return detail::callback([&] { (**static_cast<std::shared_ptr<E>*>(u)).close(); },
                                RLMESH_ERR_ENVIRONMENT);
      };
    vt.free_user_data = detail::delete_state<std::shared_ptr<E>>;
    RLMeshEnv* p = nullptr;
    detail::check_status(rlmesh_env_new(&vt, state.get(), &attrs, &p));
    state.release();
    handle_.reset(p);
  }
  RLMeshEnv* raw() const { return handle_.get(); }
  RLMeshEnv* release() { return handle_.release(); }
  void close() { detail::check_status(rlmesh_env_close(raw())); }
};
class VectorEnv {
  detail::Owned<RLMeshVectorEnv, rlmesh_vector_env_free> handle_{nullptr, rlmesh_vector_env_free};

 public:
  explicit VectorEnv(RLMeshVectorEnv* p) : handle_(p, rlmesh_vector_env_free) {}
  VectorEnv(VectorEnv&&) = default;
  VectorEnv& operator=(VectorEnv&&) = default;
  template <class E, std::enable_if_t<detail::BatchedEnv<detail::remove_cvref_t<E>>, int> = 0>
  explicit VectorEnv(E&& e)
      : VectorEnv(std::make_shared<detail::remove_cvref_t<E>>(std::forward<E>(e))) {}
  template <class E, std::enable_if_t<detail::BatchedEnv<E>, int> = 0>
  explicit VectorEnv(std::shared_ptr<E> source) {
    static_assert(detail::BatchedEnv<E>,
                  "rlmesh::VectorEnv: incompatible observation/action spaces "
                  "or reset/step signatures");
    if (!source) throw TypeError("env must not be null");
    struct State {
      std::shared_ptr<E> object;
      uint64_t num_envs;
    };
    const auto num_envs = detail::checked_num_envs(source->num_envs());
    auto state = std::make_unique<State>(State{std::move(source), num_envs});
    auto& object = *state->object;
    auto observation_space = object.single_observation_space();
    auto action_space = object.single_action_space();
    RLMeshVectorEnvAttributes attrs = RLMESH_VECTOR_ENV_ATTRIBUTES_INIT;
    attrs.single_observation_space = observation_space.raw();
    attrs.single_action_space = action_space.raw();
    attrs.num_envs = num_envs;
    std::optional<std::string> id, render_mode;
    std::optional<Metadata> metadata;
    std::optional<Value> metadata_value;
    if constexpr (detail::is_detected<detail::id_expression, E>) {
      id = object.id();
      attrs.id = detail::ptr(id);
    }
    if constexpr (detail::is_detected<detail::render_mode_expression, E>) {
      render_mode = object.render_mode();
      attrs.render_mode = detail::ptr(render_mode);
    }
    if constexpr (detail::is_detected<detail::metadata_expression, E>) {
      metadata = object.metadata();
      if (metadata) {
        metadata_value = Value(*metadata);
        attrs.metadata = metadata_value->raw();
      }
    }
    RLMeshVectorEnvVtable vt = RLMESH_VECTOR_ENV_VTABLE_INIT;
    vt.reset = [](void* u, const int64_t* seed, size_t seed_len, const RLMeshValue* options,
                  RLMeshValue** obs, RLMeshValue** info) noexcept {
      return detail::callback(
          [&] {
            auto& e = *static_cast<State*>(u)->object;
            std::optional<VectorSeed> seeds;
            if (seed) {
              if (seed_len == 1)
                seeds = *seed;
              else
                seeds = std::vector<int64_t>(seed, seed + seed_len);
            }
            auto [o, i] = e.reset(seeds, detail::optional_info(options));
            *obs = o.clone_raw();
            *info = Value(i).clone_raw();
          },
          RLMESH_ERR_ENVIRONMENT);
    };
    vt.step = [](void* u, const RLMeshValue* action, RLMeshValue** obs, double* reward,
                 bool* terminated, bool* truncated, RLMeshValue** info) noexcept {
      return detail::callback(
          [&] {
            const auto& callback_state = *static_cast<State*>(u);
            auto& e = *callback_state.object;
            if (detail::checked_num_envs(e.num_envs()) != callback_state.num_envs)
              throw ValueError("vector num_envs changed after construction");
            auto [o, r, t, tr, i] = e.step(detail::optional_value(action));
            if (detail::checked_num_envs(e.num_envs()) != callback_state.num_envs)
              throw ValueError("vector num_envs changed during step");
            const auto n = static_cast<size_t>(callback_state.num_envs);
            if (r.size() != n || t.size() != n || tr.size() != n)
              throw ValueError("vector step result length must equal num_envs");
            for (size_t lane = 0; lane < n; ++lane) {
              reward[lane] = r[lane];
              terminated[lane] = t[lane];
              truncated[lane] = tr[lane];
            }
            *obs = o.clone_raw();
            *info = Value(i).clone_raw();
          },
          RLMESH_ERR_ENVIRONMENT);
    };
    if constexpr (detail::is_detected<detail::render_expression, E>)
      vt.render = [](void* u, RLMeshValue** out) noexcept {
        return detail::callback(
            [&] {
              auto v = (*static_cast<State*>(u)->object).render();
              if (v) *out = v->clone_raw();
            },
            RLMESH_ERR_ENVIRONMENT);
      };
    if constexpr (detail::is_detected<detail::close_expression, E>)
      vt.close = [](void* u) noexcept {
        return detail::callback([&] { (*static_cast<State*>(u)->object).close(); },
                                RLMESH_ERR_ENVIRONMENT);
      };
    vt.free_user_data = detail::delete_state<State>;
    RLMeshVectorEnv* p = nullptr;
    detail::check_status(rlmesh_vector_env_new(&vt, state.get(), &attrs, &p));
    state.release();
    handle_.reset(p);
  }
  RLMeshVectorEnv* raw() const { return handle_.get(); }
  RLMeshVectorEnv* release() { return handle_.release(); }
  void close() { detail::check_status(rlmesh_vector_env_close(raw())); }
};
using MadeEnv = std::variant<Env, VectorEnv>;
class ServedEnv {
  std::vector<const RLMeshEnv*> lanes_{};
  const RLMeshVectorEnv* vector_ = nullptr;
  void assign(Span<const Env> envs) {
    if (envs.empty()) throw ValueError("served_env must not be empty");
    for (auto& e : envs) lanes_.push_back(e.raw());
  }

 public:
  ServedEnv(const Env& e) : lanes_{e.raw()} {}
  ServedEnv(Env&&) = delete;
  ServedEnv(const Env&&) = delete;
  ServedEnv(const VectorEnv& e) : vector_(e.raw()) {}
  ServedEnv(VectorEnv&&) = delete;
  ServedEnv(const VectorEnv&&) = delete;
  template <class R, std::enable_if_t<std::is_constructible_v<Span<const Env>, R&>, int> = 0>
  ServedEnv(R& envs) {
    assign(Span<const Env>(envs));
  }
  template <class R, std::enable_if_t<!std::is_lvalue_reference_v<R> &&
                                          std::is_constructible_v<Span<const Env>, R>,
                                      int> = 0>
  ServedEnv(R&&) = delete;
  const auto& lanes() const { return lanes_; }
  const RLMeshVectorEnv* vector() const { return vector_; }
};
enum class Transport { Tcp = 1, Unix = 2 };
inline std::string to_string(Transport t) {
  return std::string(detail::sv(rlmesh_transport_str(static_cast<RLMeshTransport>(t))));
}
struct ServeOptions {
  bool allow_remote_shutdown = false;
  std::optional<Duration> idle_timeout{};
  std::optional<Duration> drain_timeout{};
  std::optional<Duration> close_timeout{};
  bool compress_responses = false;
  std::optional<std::string> workflow_edition{};
  RLMeshServeOptions raw() const {
    RLMeshServeOptions c = RLMESH_SERVE_OPTIONS_INIT;
    c.allow_remote_shutdown = allow_remote_shutdown;
    c.has_idle_timeout_seconds = idle_timeout.has_value();
    if (idle_timeout) c.idle_timeout_seconds = idle_timeout->count();
    c.has_drain_timeout_seconds = drain_timeout.has_value();
    if (drain_timeout) c.drain_timeout_seconds = drain_timeout->count();
    c.has_close_timeout_seconds = close_timeout.has_value();
    if (close_timeout) c.close_timeout_seconds = close_timeout->count();
    c.compress_responses = compress_responses;
    c.workflow_edition = detail::ptr(workflow_edition);
    return c;
  }
  ServeOptions& with_allow_remote_shutdown(bool value) & {
    allow_remote_shutdown = std::move(value);
    return *this;
  }
  ServeOptions with_allow_remote_shutdown(bool value) && {
    allow_remote_shutdown = std::move(value);
    return std::move(*this);
  }
  ServeOptions& with_idle_timeout(std::optional<Duration> value) & {
    idle_timeout = std::move(value);
    return *this;
  }
  ServeOptions with_idle_timeout(std::optional<Duration> value) && {
    idle_timeout = std::move(value);
    return std::move(*this);
  }
  ServeOptions& with_drain_timeout(std::optional<Duration> value) & {
    drain_timeout = std::move(value);
    return *this;
  }
  ServeOptions with_drain_timeout(std::optional<Duration> value) && {
    drain_timeout = std::move(value);
    return std::move(*this);
  }
  ServeOptions& with_close_timeout(std::optional<Duration> value) & {
    close_timeout = std::move(value);
    return *this;
  }
  ServeOptions with_close_timeout(std::optional<Duration> value) && {
    close_timeout = std::move(value);
    return std::move(*this);
  }
  ServeOptions& with_compress_responses(bool value) & {
    compress_responses = std::move(value);
    return *this;
  }
  ServeOptions with_compress_responses(bool value) && {
    compress_responses = std::move(value);
    return std::move(*this);
  }
  ServeOptions& with_workflow_edition(std::optional<std::string> value) & {
    workflow_edition = std::move(value);
    return *this;
  }
  ServeOptions with_workflow_edition(std::optional<std::string> value) && {
    workflow_edition = std::move(value);
    return std::move(*this);
  }
};
namespace ext {
struct EnvServerOptions {
  std::optional<std::string> token{};
  std::map<std::string, std::string> peer_info_extra{};
  bool publish_describe = false;
  EnvServerOptions& with_token(std::optional<std::string> value) & {
    token = std::move(value);
    return *this;
  }
  EnvServerOptions with_token(std::optional<std::string> value) && {
    token = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_peer_info_extra(std::map<std::string, std::string> value) & {
    peer_info_extra = std::move(value);
    return *this;
  }
  EnvServerOptions with_peer_info_extra(std::map<std::string, std::string> value) && {
    peer_info_extra = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_publish_describe(bool value) & {
    publish_describe = std::move(value);
    return *this;
  }
  EnvServerOptions with_publish_describe(bool value) && {
    publish_describe = std::move(value);
    return std::move(*this);
  }
};
struct ModelServeOptions {
  std::optional<std::string> token{};
  std::optional<size_t> predict_concurrency{};
  bool no_signal_handlers = false;
  ModelServeOptions& with_no_signal_handlers(bool value) & {
    no_signal_handlers = value;
    return *this;
  }
  ModelServeOptions with_no_signal_handlers(bool value) && {
    no_signal_handlers = value;
    return std::move(*this);
  }
  ModelServeOptions& with_token(std::optional<std::string> value) & {
    token = std::move(value);
    return *this;
  }
  ModelServeOptions with_token(std::optional<std::string> value) && {
    token = std::move(value);
    return std::move(*this);
  }
  ModelServeOptions& with_predict_concurrency(std::optional<size_t> value) & {
    predict_concurrency = std::move(value);
    return *this;
  }
  ModelServeOptions with_predict_concurrency(std::optional<size_t> value) && {
    predict_concurrency = std::move(value);
    return std::move(*this);
  }
};
}  // namespace ext
struct EnvServerOptions {
  std::optional<std::string> address{}, host{};
  std::optional<uint16_t> port{};
  std::optional<std::string> path{};
  std::optional<Transport> transport{};
  std::optional<ServeOptions> options{};
  std::optional<adapters::EnvTags> tags{};
  bool close_env_on_shutdown = true;
  std::optional<ext::EnvServerOptions> ext{};
  EnvServerOptions& with_address(std::optional<std::string> value) & {
    address = std::move(value);
    return *this;
  }
  EnvServerOptions with_address(std::optional<std::string> value) && {
    address = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_port(std::optional<uint16_t> value) & {
    port = std::move(value);
    return *this;
  }
  EnvServerOptions with_port(std::optional<uint16_t> value) && {
    port = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_path(std::optional<std::string> value) & {
    path = std::move(value);
    return *this;
  }
  EnvServerOptions with_path(std::optional<std::string> value) && {
    path = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_transport(std::optional<Transport> value) & {
    transport = std::move(value);
    return *this;
  }
  EnvServerOptions with_transport(std::optional<Transport> value) && {
    transport = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_options(std::optional<ServeOptions> value) & {
    options = std::move(value);
    return *this;
  }
  EnvServerOptions with_options(std::optional<ServeOptions> value) && {
    options = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_tags(std::optional<adapters::EnvTags> value) & {
    tags = std::move(value);
    return *this;
  }
  EnvServerOptions with_tags(std::optional<adapters::EnvTags> value) && {
    tags = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_close_env_on_shutdown(bool value) & {
    close_env_on_shutdown = std::move(value);
    return *this;
  }
  EnvServerOptions with_close_env_on_shutdown(bool value) && {
    close_env_on_shutdown = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_ext(std::optional<ext::EnvServerOptions> value) & {
    ext = std::move(value);
    return *this;
  }
  EnvServerOptions with_ext(std::optional<ext::EnvServerOptions> value) && {
    ext = std::move(value);
    return std::move(*this);
  }
  EnvServerOptions& with_host(std::optional<std::string> value) & {
    host = std::move(value);
    return *this;
  }
  EnvServerOptions with_host(std::optional<std::string> value) && {
    host = std::move(value);
    return std::move(*this);
  }
};
class EnvServer;
class EnvFactoryBase;
struct EnvFactoryServeOptions;
namespace ext {
EnvServer start(EnvFactoryBase& factory, std::string_view address,
                const EnvFactoryServeOptions& options);
}
class EnvServer {
  friend EnvServer ext::start(EnvFactoryBase&, std::string_view, const EnvFactoryServeOptions&);
  friend class EnvTarget;
  struct State {
    // The server handle is destroyed before its borrowed lanes.
    MadeEnv env;
    std::vector<Env> lanes{};
    detail::Owned<RLMeshEnvServer, rlmesh_env_server_free> handle{nullptr, rlmesh_env_server_free};
    std::optional<specs::EnvContract> contract{};
    explicit State(MadeEnv e, std::vector<Env> l = {}) : env(std::move(e)), lanes(std::move(l)) {}
  };
  std::shared_ptr<State> state_;
  void init(const EnvServerOptions& options) {
    RLMeshEnvServerOptions c = RLMESH_ENV_SERVER_OPTIONS_INIT;
    c.address = detail::ptr(options.address);
    c.host = detail::ptr(options.host);
    c.path = detail::ptr(options.path);
    c.has_port = options.port.has_value();
    if (options.port) c.port = *options.port;
    c.has_transport = options.transport.has_value();
    if (options.transport) c.transport = static_cast<RLMeshTransport>(*options.transport);
    c.close_env_on_shutdown = options.close_env_on_shutdown;
    if (options.tags) c.tags = options.tags->raw();
    std::optional<RLMeshServeOptions> so;
    if (options.options) {
      so = options.options->raw();
      c.options = &*so;
    }
    RLMeshExtEnvServerOptions extra = RLMESH_EXT_ENV_SERVER_OPTIONS_INIT;
    Metadata peer;
    if (options.ext) {
      extra.token = detail::ptr(options.ext->token);
      extra.publish_describe = options.ext->publish_describe;
      for (auto& [k, v] : options.ext->peer_info_extra) peer.insert_or_assign(k, v);
    }
    Value peer_value(peer);
    extra.peer_info_extra = peer_value.raw();
    c.ext = &extra;
    RLMeshEnvServer* p = nullptr;
    if (auto v = std::get_if<VectorEnv>(&state_->env))
      detail::check_status(rlmesh_env_server_new_vector(v->raw(), &c, &p));
    else {
      std::vector<RLMeshEnv*> lanes;
      if (state_->lanes.empty())
        lanes.push_back(std::get<Env>(state_->env).raw());
      else
        for (auto& e : state_->lanes) lanes.push_back(e.raw());
      detail::check_status(rlmesh_env_server_new(lanes.data(), lanes.size(), &c, &p));
    }
    state_->handle.reset(p);
    state_->contract = specs::EnvContract::copy_of(rlmesh_env_server_env_contract(p));
  }

  explicit EnvServer(detail::Owned<RLMeshEnvServer, rlmesh_env_server_free> server,
                     detail::FromLibrary)
      : state_(std::make_shared<State>(Env(static_cast<RLMeshEnv*>(nullptr)))) {
    state_->handle = std::move(server);
    state_->contract = specs::EnvContract::copy_of(rlmesh_env_server_env_contract(raw()));
  }

 public:
  explicit EnvServer(Env env, EnvServerOptions options = {})
      : state_(std::make_shared<State>(std::move(env))) {
    init(options);
  }
  explicit EnvServer(VectorEnv env, EnvServerOptions options = {})
      : state_(std::make_shared<State>(std::move(env))) {
    init(options);
  }
  explicit EnvServer(std::vector<Env> lanes, EnvServerOptions options = {})
      : state_(std::make_shared<State>(Env(static_cast<RLMeshEnv*>(nullptr)), std::move(lanes))) {
    if (state_->lanes.empty()) throw ValueError("env lanes must not be empty");
    init(options);
  }
  template <class E, std::enable_if_t<detail::ScalarEnv<detail::remove_cvref_t<E>>, int> = 0>
  explicit EnvServer(E&& env, EnvServerOptions options = {})
      : EnvServer(Env(std::forward<E>(env)), std::move(options)) {}
  template <class E, std::enable_if_t<detail::BatchedEnv<detail::remove_cvref_t<E>>, int> = 0>
  explicit EnvServer(E&& env, EnvServerOptions options = {})
      : EnvServer(VectorEnv(std::forward<E>(env)), std::move(options)) {}
  EnvServer(const EnvServer&) = delete;
  EnvServer& operator=(const EnvServer&) = delete;
  EnvServer(EnvServer&&) = default;
  EnvServer& operator=(EnvServer&&) = delete;
  RLMeshEnvServer* raw() const { return state_->handle.get(); }
  std::string address() const { return detail::text(rlmesh_env_server_address, raw()); }
  const specs::EnvContract& env_contract() const { return *state_->contract; }
  void start() { detail::check_status(rlmesh_env_server_start(raw())); }
  void serve() { detail::check_status(rlmesh_env_server_serve(raw())); }
  bool wait(std::optional<Duration> timeout = {}) {
    std::optional<double> t;
    if (timeout) t = timeout->count();
    bool stopped = false;
    detail::check_status(rlmesh_env_server_wait(raw(), detail::ptr(t), &stopped));
    return stopped;
  }
  void shutdown() { detail::check_status(rlmesh_env_server_shutdown(raw())); }
  std::string to_string() const { return detail::text(rlmesh_env_server_repr, raw()); }
};
namespace adapters {
struct TagOptions {
  bool validate = true;
  TagOptions& with_validate(bool value) & {
    validate = std::move(value);
    return *this;
  }
  TagOptions with_validate(bool value) && {
    validate = std::move(value);
    return std::move(*this);
  }
};
inline Env& tag(Env& env, const EnvTags& tags, TagOptions options = {}) {
  RLMeshTagOptions c = RLMESH_TAG_OPTIONS_INIT;
  c.validate = options.validate;
  detail::check_status(rlmesh_adapters_tag(env.raw(), tags.raw(), &c));
  return env;
}
inline VectorEnv& tag(VectorEnv& env, const EnvTags& tags, TagOptions options = {}) {
  RLMeshTagOptions c = RLMESH_TAG_OPTIONS_INIT;
  c.validate = options.validate;
  detail::check_status(rlmesh_adapters_tag_vector(env.raw(), tags.raw(), &c));
  return env;
}
}  // namespace adapters
struct Variant {
  std::string id{};
  Kwargs params{};
  Kwargs metadata = {};

  RLMeshVariant* raw() const {
    Value p(params), m(metadata);
    RLMeshVariantOptions c = RLMESH_VARIANT_OPTIONS_INIT;
    c.metadata = m.raw();
    RLMeshVariant* v = nullptr;
    detail::check_status(rlmesh_variant_new(id.c_str(), p.raw(), &c, &v));
    return v;
  }
  static Variant from_raw(const RLMeshVariant* v) {
    return {detail::text(rlmesh_variant_id, v), detail::info(rlmesh_variant_params(v)),
            detail::info(rlmesh_variant_metadata(v))};
  }

  friend bool operator==(const Variant& self, const Variant& other) noexcept(
      detail::nothrow_equal<decltype(id), decltype(params), decltype(metadata)>) {
    return self.id == other.id && self.params == other.params && self.metadata == other.metadata;
  }
  friend bool operator!=(const Variant& self,
                         const Variant& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const Variant& v) {
  auto p = detail::own<RLMeshVariant, rlmesh_variant_free>(v.raw());
  return detail::text(rlmesh_variant_repr, p.get());
}
inline std::ostream& operator<<(std::ostream& s, const Variant& v) { return s << to_string(v); }
struct TagParam {
  std::string name{};
  Value default_{};
  std::vector<Value> choices{};
};
struct EnvFactoryServeOptions {
  int64_t num_envs = 1;
  std::optional<std::string> workflow_edition{};
  Kwargs make_kwargs{};
  std::optional<ext::CancelToken> cancel{};
  EnvFactoryServeOptions& with_num_envs(int64_t value) & {
    num_envs = std::move(value);
    return *this;
  }
  EnvFactoryServeOptions with_num_envs(int64_t value) && {
    num_envs = std::move(value);
    return std::move(*this);
  }
  EnvFactoryServeOptions& with_workflow_edition(std::optional<std::string> value) & {
    workflow_edition = std::move(value);
    return *this;
  }
  EnvFactoryServeOptions with_workflow_edition(std::optional<std::string> value) && {
    workflow_edition = std::move(value);
    return std::move(*this);
  }
  EnvFactoryServeOptions& with_make_kwargs(Kwargs value) & {
    make_kwargs = std::move(value);
    return *this;
  }
  EnvFactoryServeOptions with_make_kwargs(Kwargs value) && {
    make_kwargs = std::move(value);
    return std::move(*this);
  }
  EnvFactoryServeOptions& with_cancel(std::optional<ext::CancelToken> value) & {
    cancel = std::move(value);
    return *this;
  }
  EnvFactoryServeOptions with_cancel(std::optional<ext::CancelToken> value) && {
    cancel = std::move(value);
    return std::move(*this);
  }
};
class EnvFactoryBase {
  using Build = RLMeshEnvFactory* (*)(EnvFactoryBase&);
  Build build_{};
  detail::Owned<RLMeshEnvFactory, rlmesh_env_factory_free> handle_{nullptr,
                                                                   rlmesh_env_factory_free};

 protected:
  explicit EnvFactoryBase(Build build) noexcept : build_(build) {}

 public:
  EnvFactoryBase(const EnvFactoryBase&) = delete;
  EnvFactoryBase& operator=(const EnvFactoryBase&) = delete;
  EnvFactoryBase(EnvFactoryBase&&) = delete;
  EnvFactoryBase& operator=(EnvFactoryBase&&) = delete;
  ~EnvFactoryBase() = default;
  RLMeshEnvFactory* handle() {
    if (!handle_) handle_.reset(build_(*this));
    return handle_.get();
  }
  MadeEnv make_env(const Kwargs& kwargs = {}) {
    Value k(kwargs);
    RLMeshEnv* e = nullptr;
    RLMeshVectorEnv* v = nullptr;
    detail::check_status(rlmesh_env_factory_make(handle(), k.raw(), &e, &v));
    if (e) return Env(e);
    return VectorEnv(v);
  }
  void serve(std::string_view address, const EnvFactoryServeOptions& options = {}) {
    Value k(options.make_kwargs);
    RLMeshEnvFactoryServeOptions c = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
    c.num_envs = options.num_envs;
    c.workflow_edition = detail::ptr(options.workflow_edition);
    c.make_kwargs = k.raw();
    if (options.cancel) c.cancel = options.cancel->raw();
    RLMeshExtEnvFactoryServeOptions extra = RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_INIT;
    c.ext = &extra;
    detail::check_status(rlmesh_env_factory_serve(handle(), std::string(address).c_str(), &c));
  }
};
namespace ext {
/// Prepare and start a factory server without signal handlers. The factory may
/// be reused or destroyed on return; the returned server owns its made envs.
inline EnvServer start(EnvFactoryBase& factory, std::string_view address,
                       const EnvFactoryServeOptions& options = {}) {
  Value k(options.make_kwargs);
  RLMeshEnvFactoryServeOptions c = RLMESH_ENV_FACTORY_SERVE_OPTIONS_INIT;
  c.num_envs = options.num_envs;
  c.workflow_edition = detail::ptr(options.workflow_edition);
  c.make_kwargs = k.raw();
  if (options.cancel) c.cancel = options.cancel->raw();
  RLMeshExtEnvFactoryServeOptions extra = RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_INIT;
  c.ext = &extra;
  RLMeshEnvServer* server = nullptr;
  detail::check_status(
      rlmesh_ext_env_factory_start(factory.handle(), std::string(address).c_str(), &c, &server));
  return EnvServer(detail::own<RLMeshEnvServer, rlmesh_env_server_free>(server),
                   detail::FromLibrary{});
}
}  // namespace ext

template <class F>
class EnvFactory : public EnvFactoryBase {
  static RLMeshEnvFactory* build(EnvFactoryBase& base) {
    RLMeshEnvFactoryVtable vt = RLMESH_ENV_FACTORY_VTABLE_INIT;
    vt.make = [](void* u, const RLMeshValue* k, RLMeshEnv** e, RLMeshVectorEnv** v) noexcept {
      return detail::callback([&] {
        auto env = static_cast<F*>(u)->make(detail::info(k));
        std::visit(
            [&](auto& x) {
              using T = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<T, Env>)
                *e = x.release();
              else
                *v = x.release();
            },
            env);
      });
    };
    if constexpr (detail::is_detected<detail::prepare_expression, F>)
      vt.prepare = [](void* u) noexcept {
        return detail::callback([&] { static_cast<F*>(u)->prepare(); });
      };
    if constexpr (detail::is_detected<detail::close_expression, F>)
      vt.close = [](void* u) noexcept {
        return detail::callback([&] { static_cast<F*>(u)->close(); });
      };
    if constexpr (&F::tags_for != &EnvFactory<F>::tags_for)
      vt.tags_for = [](void*, const RLMeshValue* k, RLMeshEnvTags** o) noexcept {
        return detail::callback([&] {
          auto t = F::tags_for(detail::info(k));
          if (t) *o = rlmesh_env_tags_clone(t->raw());
        });
      };
    if constexpr (&F::enumerate_variants != &EnvFactory<F>::enumerate_variants)
      vt.enumerate_variants = [](void*, RLMeshVariantSink* sink) noexcept {
        return detail::callback([&] {
          auto variants = F::enumerate_variants();
          if (variants)
            for (auto& v : *variants) {
              auto p = detail::own<RLMeshVariant, rlmesh_variant_free>(v.raw());
              detail::check_status(rlmesh_variant_sink_push(sink, p.get()));
              p.release();
            }
        });
      };
    RLMeshEnvFactoryAttributes c = RLMESH_ENV_FACTORY_ATTRIBUTES_INIT;
    std::optional<adapters::EnvTags> tags;
    if constexpr (detail::is_detected<detail::tags_expression, F>) {
      tags = F::tags();
      if (tags) c.tags = tags->raw();
    }
    std::vector<std::string> reset;
    std::vector<const char*> reset_ptrs;
    if constexpr (detail::is_detected<detail::reset_options_expression, F>) {
      reset = F::reset_options();
      for (auto& s : reset) reset_ptrs.push_back(s.c_str());
      c.reset_options = reset_ptrs.data();
      c.reset_options_len = reset_ptrs.size();
    }
    std::optional<std::string> edition;
    if constexpr (detail::is_detected<detail::workflow_edition_expression, F>) {
      edition = F::workflow_edition();
      c.workflow_edition = detail::ptr(edition);
    }
    std::vector<TagParam> params;
    std::vector<Value> choices;
    std::vector<RLMeshTagParam> records;
    std::vector<const RLMeshTagParam*> ptrs;
    if constexpr (detail::is_detected<detail::tag_params_expression, F>) {
      params = F::tag_params();
      choices.reserve(params.size());
      records.reserve(params.size());
      for (auto& p : params) {
        choices.push_back(Value::list(p.choices));
        RLMeshTagParam r = RLMESH_TAG_PARAM_INIT;
        r.name = p.name.c_str();
        r.default_ = p.default_.raw();
        r.choices = choices.back().raw();
        records.push_back(r);
      }
      for (auto& r : records) ptrs.push_back(&r);
      c.tag_params = ptrs.data();
      c.tag_params_len = ptrs.size();
    }
    RLMeshEnvFactory* p = nullptr;
    detail::check_status(rlmesh_env_factory_new(&vt, static_cast<F*>(&base), &c, &p));
    return p;
  }

 public:
  EnvFactory() noexcept : EnvFactoryBase(build) {}
  static std::optional<adapters::EnvTags> tags() { return {}; }
  static std::vector<TagParam> tag_params() { return {}; }
  static std::vector<std::string> reset_options() { return {}; }
  static std::optional<std::string> workflow_edition() { return {}; }
  static std::optional<adapters::EnvTags> tags_for([[maybe_unused]] const Kwargs& params) {
    return F::tags();
  }
  static std::optional<std::vector<Variant>> enumerate_variants() { return {}; }
  static Metadata describe();
  void prepare() {}
  void close() {}
};
namespace types {
using SpecArg = rlmesh::SpecArg;
struct LoadKey {
  std::string_view name{};
  bool required = false;
};
class EpisodeState {
  RLMeshEpisodeState* state_{};
  std::any* value(std::string_view key) const {
    void* p = nullptr;
    std::string k(key);
    return rlmesh_episode_state_get(state_, k.c_str(), &p) ? static_cast<std::any*>(p) : nullptr;
  }

 public:
  explicit EpisodeState(RLMeshEpisodeState* p) : state_(p) {}
  template <class T>
  T* get(std::string_view key) {
    auto p = value(key);
    return p ? std::any_cast<T>(p) : nullptr;
  }
  template <class T>
  const T* get(std::string_view key) const {
    auto p = value(key);
    return p ? std::any_cast<T>(p) : nullptr;
  }
  template <class T>
  void insert_or_assign(std::string_view key, T v) {
    auto p = std::make_unique<std::any>(std::move(v));
    std::string k(key);
    detail::check_status(
        rlmesh_episode_state_set(state_, k.c_str(), p.get(), detail::delete_state<std::any>));
    p.release();
  }
  template <class T, class... A>
  T& get_or_emplace(std::string_view key, A&&... args) {
    if (auto p = value(key)) {
      if (auto v = std::any_cast<T>(p)) return *v;
      throw TypeError("episode state key holds another type");
    }
    insert_or_assign(key, T(std::forward<A>(args)...));
    return *get<T>(key);
  }
  bool erase(std::string_view key) {
    return rlmesh_episode_state_remove(state_, std::string(key).c_str());
  }
  bool contains_key(std::string_view key) const { return value(key) != nullptr; }
  size_t size() const { return rlmesh_episode_state_len(state_); }
  bool empty() const { return size() == 0; }
};
struct PredictContext {
  std::string_view episode_id{};
  std::optional<int64_t> episode_seed{};
  uint64_t predict_index{};
  std::optional<int64_t> predict_seed{};
  EpisodeState& state;
};
using BatchPredictContext = Span<const PredictContext>;
}  // namespace types
using types::BatchPredictContext;
using types::EpisodeState;
using types::PredictContext;
inline int64_t predict_seed(int64_t episode_seed, uint64_t predict_index) {
  return rlmesh_predict_seed(episode_seed, predict_index);
}
class EpisodeEndHook {
  std::function<void(std::string_view)> function_{};

 public:
  EpisodeEndHook() = default;
  template <class F,
            std::enable_if_t<std::is_invocable_v<F&, std::string_view> || std::is_invocable_v<F&>,
                             int> = 0>
  EpisodeEndHook(F f) {
    function_ = [f = std::move(f)](std::string_view id) mutable {
      if constexpr (std::is_invocable_v<F&, std::string_view>)
        f(id);
      else
        f();
    };
  }
  explicit operator bool() const { return bool(function_); }
  void operator()(std::string_view id) const { function_(id); }
};
struct ModelOptions {
  std::optional<SpecArg> spec{};
  EpisodeEndHook on_episode_end{};
  std::function<void()> on_close{};
  bool trust_entrypoints = false;
  ModelOptions& with_on_close(std::function<void()> value) & {
    on_close = std::move(value);
    return *this;
  }
  ModelOptions with_on_close(std::function<void()> value) && {
    on_close = std::move(value);
    return std::move(*this);
  }
  ModelOptions& with_spec(std::optional<SpecArg> value) & {
    spec = std::move(value);
    return *this;
  }
  ModelOptions with_spec(std::optional<SpecArg> value) && {
    spec = std::move(value);
    return std::move(*this);
  }
  ModelOptions& with_on_episode_end(EpisodeEndHook value) & {
    on_episode_end = std::move(value);
    return *this;
  }
  ModelOptions with_on_episode_end(EpisodeEndHook value) && {
    on_episode_end = std::move(value);
    return std::move(*this);
  }
  ModelOptions& with_trust_entrypoints(bool value) & {
    trust_entrypoints = std::move(value);
    return *this;
  }
  ModelOptions with_trust_entrypoints(bool value) && {
    trust_entrypoints = std::move(value);
    return std::move(*this);
  }
};
struct ModelServeOptions {
  std::optional<ServeOptions> options{};
  std::optional<ext::CancelToken> cancel{};
  std::optional<ext::ModelServeOptions> ext{};
  // The participant the model joins the session at address as; nullopt reads
  // it off address.
  std::optional<std::string> name{};
  ModelServeOptions& with_name(std::optional<std::string> value) & {
    name = std::move(value);
    return *this;
  }
  ModelServeOptions with_name(std::optional<std::string> value) && {
    name = std::move(value);
    return std::move(*this);
  }
  ModelServeOptions& with_options(std::optional<ServeOptions> value) & {
    options = std::move(value);
    return *this;
  }
  ModelServeOptions with_options(std::optional<ServeOptions> value) && {
    options = std::move(value);
    return std::move(*this);
  }
  ModelServeOptions& with_cancel(std::optional<ext::CancelToken> value) & {
    cancel = std::move(value);
    return *this;
  }
  ModelServeOptions with_cancel(std::optional<ext::CancelToken> value) && {
    cancel = std::move(value);
    return std::move(*this);
  }
  ModelServeOptions& with_ext(std::optional<ext::ModelServeOptions> value) & {
    ext = std::move(value);
    return *this;
  }
  ModelServeOptions with_ext(std::optional<ext::ModelServeOptions> value) && {
    ext = std::move(value);
    return std::move(*this);
  }
};
namespace detail {
struct Contexts {
  std::vector<EpisodeState> states{};
  std::vector<PredictContext> contexts{};
  Contexts(const RLMeshPredictContext* const* p, size_t n) {
    states.reserve(n);
    contexts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto& c = *p[i];
      states.emplace_back(c.state);
      contexts.push_back(
          {sv(c.episode_id), c.has_episode_seed ? std::optional(c.episode_seed) : std::nullopt,
           c.predict_index, c.has_predict_seed ? std::optional(c.predict_seed) : std::nullopt,
           states.back()});
    }
  }
};
template <class P>
using predict_c_expression_0 = decltype(std::declval<P&>().predict(
    std::declval<const Value&>(), std::declval<const PredictContext&>()));
template <class P, class = void>
struct predict_c_trait : std::false_type {};
template <class P>
struct predict_c_trait<P, std::void_t<predict_c_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_c_expression_0<P>, Value>> {};
template <class P>
inline constexpr bool predict_c = predict_c_trait<P>::value;
template <class P>
using predict_plain_expression_0 =
    decltype(std::declval<P&>().predict(std::declval<const Value&>()));
template <class P, class = void>
struct predict_plain_trait : std::false_type {};
template <class P>
struct predict_plain_trait<P, std::void_t<predict_plain_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_plain_expression_0<P>, Value>> {};
template <class P>
inline constexpr bool predict_plain = predict_plain_trait<P>::value;
template <class P>
inline constexpr bool callable_c =
    std::is_invocable_r_v<Value, P&, const Value&, const PredictContext&>;
template <class P>
inline constexpr bool callable_plain = std::is_invocable_r_v<Value, P&, const Value&>;
template <class P>
inline constexpr bool has_predict =
    predict_c<P> || predict_plain<P> || callable_c<P> || callable_plain<P>;
template <class P>
inline constexpr bool context_predict = predict_c<P> || (!predict_plain<P> && callable_c<P>);
template <class P>
using predict_chunk_hc_expression_0 = decltype(std::declval<P&>().predict_chunk(
    std::declval<const Value&>(), std::declval<uint32_t&>(),
    std::declval<const PredictContext&>()));
template <class P, class = void>
struct predict_chunk_hc_trait : std::false_type {};
template <class P>
struct predict_chunk_hc_trait<P, std::void_t<predict_chunk_hc_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_chunk_hc_expression_0<P>, Value>> {};
template <class P>
inline constexpr bool predict_chunk_hc = predict_chunk_hc_trait<P>::value;
template <class P>
using predict_chunk_c_expression_0 = decltype(std::declval<P&>().predict_chunk(
    std::declval<const Value&>(), std::declval<const PredictContext&>()));
template <class P, class = void>
struct predict_chunk_c_trait : std::false_type {};
template <class P>
struct predict_chunk_c_trait<P, std::void_t<predict_chunk_c_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_chunk_c_expression_0<P>, Value>> {};
template <class P>
inline constexpr bool predict_chunk_c = predict_chunk_c_trait<P>::value;
template <class P>
using predict_chunk_h_expression_0 = decltype(std::declval<P&>().predict_chunk(
    std::declval<const Value&>(), std::declval<uint32_t&>()));
template <class P, class = void>
struct predict_chunk_h_trait : std::false_type {};
template <class P>
struct predict_chunk_h_trait<P, std::void_t<predict_chunk_h_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_chunk_h_expression_0<P>, Value>> {};
template <class P>
inline constexpr bool predict_chunk_h = predict_chunk_h_trait<P>::value;
template <class P>
using predict_chunk_plain_expression_0 =
    decltype(std::declval<P&>().predict_chunk(std::declval<const Value&>()));
template <class P, class = void>
struct predict_chunk_plain_trait : std::false_type {};
template <class P>
struct predict_chunk_plain_trait<P, std::void_t<predict_chunk_plain_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_chunk_plain_expression_0<P>, Value>> {};
template <class P>
inline constexpr bool predict_chunk_plain = predict_chunk_plain_trait<P>::value;
template <class P>
inline constexpr bool has_predict_chunk =
    predict_chunk_hc<P> || predict_chunk_c<P> || predict_chunk_h<P> || predict_chunk_plain<P>;
template <class P>
inline constexpr bool context_predict_chunk = predict_chunk_hc<P> || predict_chunk_c<P>;
template <class P>
using predict_batch_c_expression_0 = decltype(std::declval<P&>().predict_batch(
    std::declval<Span<const Value>&>(), std::declval<BatchPredictContext&>()));
template <class P, class = void>
struct predict_batch_c_trait : std::false_type {};
template <class P>
struct predict_batch_c_trait<P, std::void_t<predict_batch_c_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_batch_c_expression_0<P>, std::vector<Value>>> {};
template <class P>
inline constexpr bool predict_batch_c = predict_batch_c_trait<P>::value;
template <class P>
using predict_batch_plain_expression_0 =
    decltype(std::declval<P&>().predict_batch(std::declval<Span<const Value>&>()));
template <class P, class = void>
struct predict_batch_plain_trait : std::false_type {};
template <class P>
struct predict_batch_plain_trait<P, std::void_t<predict_batch_plain_expression_0<P>>>
    : std::bool_constant<std::is_same_v<predict_batch_plain_expression_0<P>, std::vector<Value>>> {
};
template <class P>
inline constexpr bool predict_batch_plain = predict_batch_plain_trait<P>::value;
template <class P>
inline constexpr bool has_predict_batch = predict_batch_c<P> || predict_batch_plain<P>;
template <class P>
inline constexpr bool context_predict_batch = predict_batch_c<P>;
template <class P>
using predict_chunk_batch_hc_expression_0 = decltype(std::declval<P&>().predict_chunk_batch(
    std::declval<Span<const Value>&>(), std::declval<uint32_t&>(),
    std::declval<BatchPredictContext&>()));
template <class P, class = void>
struct predict_chunk_batch_hc_trait : std::false_type {};
template <class P>
struct predict_chunk_batch_hc_trait<P, std::void_t<predict_chunk_batch_hc_expression_0<P>>>
    : std::bool_constant<
          std::is_same_v<predict_chunk_batch_hc_expression_0<P>, std::vector<Value>>> {};
template <class P>
inline constexpr bool predict_chunk_batch_hc = predict_chunk_batch_hc_trait<P>::value;
template <class P>
using predict_chunk_batch_c_expression_0 = decltype(std::declval<P&>().predict_chunk_batch(
    std::declval<Span<const Value>&>(), std::declval<BatchPredictContext&>()));
template <class P, class = void>
struct predict_chunk_batch_c_trait : std::false_type {};
template <class P>
struct predict_chunk_batch_c_trait<P, std::void_t<predict_chunk_batch_c_expression_0<P>>>
    : std::bool_constant<
          std::is_same_v<predict_chunk_batch_c_expression_0<P>, std::vector<Value>>> {};
template <class P>
inline constexpr bool predict_chunk_batch_c = predict_chunk_batch_c_trait<P>::value;
template <class P>
using predict_chunk_batch_h_expression_0 = decltype(std::declval<P&>().predict_chunk_batch(
    std::declval<Span<const Value>&>(), std::declval<uint32_t&>()));
template <class P, class = void>
struct predict_chunk_batch_h_trait : std::false_type {};
template <class P>
struct predict_chunk_batch_h_trait<P, std::void_t<predict_chunk_batch_h_expression_0<P>>>
    : std::bool_constant<
          std::is_same_v<predict_chunk_batch_h_expression_0<P>, std::vector<Value>>> {};
template <class P>
inline constexpr bool predict_chunk_batch_h = predict_chunk_batch_h_trait<P>::value;
template <class P>
using predict_chunk_batch_plain_expression_0 =
    decltype(std::declval<P&>().predict_chunk_batch(std::declval<Span<const Value>&>()));
template <class P, class = void>
struct predict_chunk_batch_plain_trait : std::false_type {};
template <class P>
struct predict_chunk_batch_plain_trait<P, std::void_t<predict_chunk_batch_plain_expression_0<P>>>
    : std::bool_constant<
          std::is_same_v<predict_chunk_batch_plain_expression_0<P>, std::vector<Value>>> {};
template <class P>
inline constexpr bool predict_chunk_batch_plain = predict_chunk_batch_plain_trait<P>::value;
template <class P>
inline constexpr bool has_predict_chunk_batch =
    predict_chunk_batch_hc<P> || predict_chunk_batch_c<P> || predict_chunk_batch_h<P> ||
    predict_chunk_batch_plain<P>;
template <class P>
inline constexpr bool context_predict_chunk_batch =
    predict_chunk_batch_hc<P> || predict_chunk_batch_c<P>;
template <class P>
struct PolicyTable {
  RLMeshModelVtable table = RLMESH_MODEL_VTABLE_INIT;
  std::optional<SpecArg> spec{};
  std::optional<std::string> edition{};
  std::vector<std::string> key_names{};
  std::vector<RLMeshLoadKey> keys{};
  RLMeshLoadKey empty_key{};
  PolicyTable() {
    static_assert(!(detail::is_detected<detail::reset_member_expression, P>) &&
                      !(detail::is_detected<detail::reset_expression, P>) &&
                      !(detail::is_detected<detail::reset_id_expression, P>),
                  "Policy.reset is unsupported; use on_episode_end");
    static_assert(has_predict<P> || has_predict_batch<P> || has_predict_chunk<P> ||
                      has_predict_chunk_batch<P>,
                  "rlmesh::Model: the policy type has none of "
                  "predict/predict_batch/predict_chunk/predict_chunk_batch");
    if constexpr (detail::is_detected<detail::native_chunk_expression, P>)
      static_assert(!P::native_chunk || has_predict_chunk<P> || has_predict_chunk_batch<P>,
                    "native_chunk requires a chunk corner");
    static_assert(!(detail::is_detected<detail::load_member_expression, P>) ||
                      detail::is_detected<detail::load_kwargs_expression, P> ||
                      detail::is_detected<detail::load_expression, P>,
                  "load must accept Kwargs or no arguments");
    static_assert(!(detail::is_detected<detail::on_episode_end_member_expression, P>) ||
                      detail::is_detected<detail::on_episode_end_lvalue_expression, P> ||
                      detail::is_detected<detail::on_episode_end_expression, P>,
                  "on_episode_end must accept episode_id or no arguments");
    static_assert(!(detail::is_detected<detail::load_expression, P>) ||
                      detail::is_detected<detail::load_kwargs_expression, P> ||
                      !(detail::is_detected<detail::load_keys_expression, P>),
                  "nullary load cannot declare load_keys");
    if constexpr (has_predict<P>) {
      if constexpr (context_predict<P>) table.context_corners |= RLMESH_CORNER_PREDICT;
      table.predict = [](void* u, const RLMeshValue* obs, const RLMeshPredictContext* cs,
                         RLMeshValue** output) noexcept {
        return detail::callback(
            [&] {
              auto& p = **static_cast<std::shared_ptr<P>*>(u);
              auto o = Value::copy_of(obs);
              auto invoke = [&]() -> Value {
                if constexpr (predict_c<P>) {
                  const RLMeshPredictContext* array[] = {cs};
                  Contexts ctx(array, 1);
                  const auto& c = ctx.contexts[0];
                  return p.predict(o, c);
                } else if constexpr (predict_plain<P>) {
                  return p.predict(o);
                } else if constexpr (callable_c<P>) {
                  const RLMeshPredictContext* array[] = {cs};
                  Contexts ctx(array, 1);
                  return p(o, ctx.contexts[0]);
                } else {
                  return p(o);
                }
              };
              auto result = invoke();
              *output = result.clone_raw();
            },
            RLMESH_ERR_MODEL);
      };
    }
    if constexpr (has_predict_chunk<P>) {
      if constexpr (context_predict_chunk<P>) table.context_corners |= RLMESH_CORNER_PREDICT_CHUNK;
      table.predict_chunk = [](void* u, const RLMeshValue* obs, uint32_t h,
                               const RLMeshPredictContext* cs, RLMeshValue** output) noexcept {
        return detail::callback(
            [&] {
              auto& p = **static_cast<std::shared_ptr<P>*>(u);
              auto o = Value::copy_of(obs);
              auto invoke = [&]() -> Value {
                if constexpr (predict_chunk_hc<P>) {
                  const RLMeshPredictContext* array[] = {cs};
                  Contexts ctx(array, 1);
                  const auto& c = ctx.contexts[0];
                  return p.predict_chunk(o, h, c);
                } else if constexpr (predict_chunk_c<P>) {
                  const RLMeshPredictContext* array[] = {cs};
                  Contexts ctx(array, 1);
                  const auto& c = ctx.contexts[0];
                  return p.predict_chunk(o, c);
                } else if constexpr (predict_chunk_h<P>) {
                  return p.predict_chunk(o, h);
                } else if constexpr (predict_chunk_plain<P>) {
                  return p.predict_chunk(o);
                }
              };
              auto result = invoke();
              *output = result.clone_raw();
            },
            RLMESH_ERR_MODEL);
      };
    }
    if constexpr (has_predict_batch<P>) {
      if constexpr (context_predict_batch<P>) table.context_corners |= RLMESH_CORNER_PREDICT_BATCH;
      table.predict_batch = [](void* u, const RLMeshValue* const* obs, size_t n,
                               const RLMeshPredictContext* const* cs,
                               RLMeshValue** output) noexcept {
        return detail::callback(
            [&] {
              auto& p = **static_cast<std::shared_ptr<P>*>(u);
              std::vector<Value> observations;
              observations.reserve(n);
              for (size_t i = 0; i < n; ++i) observations.push_back(Value::copy_of(obs[i]));
              Span<const Value> o(observations);
              auto invoke = [&]() -> std::vector<Value> {
                if constexpr (predict_batch_c<P>) {
                  Contexts ctx(cs, n);
                  BatchPredictContext c(ctx.contexts);
                  return p.predict_batch(o, c);
                } else if constexpr (predict_batch_plain<P>) {
                  return p.predict_batch(o);
                }
              };
              auto result = invoke();
              if (result.size() != n)
                throw ValueError("policy output length must equal batch size");
              for (size_t i = 0; i < n; ++i) output[i] = result[i].clone_raw();
            },
            RLMESH_ERR_MODEL);
      };
    }
    if constexpr (has_predict_chunk_batch<P>) {
      if constexpr (context_predict_chunk_batch<P>)
        table.context_corners |= RLMESH_CORNER_PREDICT_CHUNK_BATCH;
      table.predict_chunk_batch = [](void* u, const RLMeshValue* const* obs, size_t n, uint32_t h,
                                     const RLMeshPredictContext* const* cs,
                                     RLMeshValue** output) noexcept {
        return detail::callback(
            [&] {
              auto& p = **static_cast<std::shared_ptr<P>*>(u);
              std::vector<Value> observations;
              observations.reserve(n);
              for (size_t i = 0; i < n; ++i) observations.push_back(Value::copy_of(obs[i]));
              Span<const Value> o(observations);
              auto invoke = [&]() -> std::vector<Value> {
                if constexpr (predict_chunk_batch_hc<P>) {
                  Contexts ctx(cs, n);
                  BatchPredictContext c(ctx.contexts);
                  return p.predict_chunk_batch(o, h, c);
                } else if constexpr (predict_chunk_batch_c<P>) {
                  Contexts ctx(cs, n);
                  BatchPredictContext c(ctx.contexts);
                  return p.predict_chunk_batch(o, c);
                } else if constexpr (predict_chunk_batch_h<P>) {
                  return p.predict_chunk_batch(o, h);
                } else if constexpr (predict_chunk_batch_plain<P>) {
                  return p.predict_chunk_batch(o);
                }
              };
              auto result = invoke();
              if (result.size() != n)
                throw ValueError("policy output length must equal batch size");
              for (size_t i = 0; i < n; ++i) output[i] = result[i].clone_raw();
            },
            RLMESH_ERR_MODEL);
      };
    }
    if constexpr (detail::is_detected<detail::spec_expression, P>) {
      spec = P::spec();
      table.spec = spec_arg(spec);
    }
    if constexpr (detail::is_detected<detail::allow_fusion_expression, P>)
      table.allow_fusion = P::allow_fusion;
    if constexpr (detail::is_detected<detail::native_chunk_expression, P>) {
      table.has_native_chunk = P::native_chunk.has_value();
      if (P::native_chunk) table.native_chunk = *P::native_chunk;
    }
    if constexpr (detail::is_detected<detail::workflow_edition_expression, P>) {
      edition = P::workflow_edition();
      table.workflow_edition = ptr(edition);
    }
    if constexpr (detail::is_detected<detail::load_keys_expression, P>) {
      for (auto& k : P::load_keys) key_names.emplace_back(k.name);
      size_t i = 0;
      for (auto& k : P::load_keys) {
        RLMeshLoadKey c{};
        c.name = key_names[i++].c_str();
        c.required = k.required;
        keys.push_back(c);
      }
      table.load_keys = keys.empty() ? &empty_key : keys.data();
      table.load_keys_len = keys.size();
    } else if constexpr (detail::is_detected<detail::load_expression, P> &&
                         !detail::is_detected<detail::load_kwargs_expression, P>)
      table.load_keys = &empty_key;
    if constexpr (detail::is_detected<detail::load_kwargs_expression, P> ||
                  detail::is_detected<detail::load_expression, P>)
      table.load = [](void* u, const RLMeshValue* config) noexcept {
        return detail::callback(
            [&] {
              auto& p = **static_cast<std::shared_ptr<P>*>(u);
              if constexpr (detail::is_detected<detail::load_kwargs_expression, P>)
                p.load(info(config));
              else
                p.load();
            },
            RLMESH_ERR_MODEL);
      };
    if constexpr (detail::is_detected<detail::loaded_native_chunk_expression, P>)
      table.loaded_native_chunk = [](void* u, bool* has, uint32_t* value) noexcept {
        return detail::callback(
            [&] {
              auto v = (**static_cast<std::shared_ptr<P>*>(u)).loaded_native_chunk();
              *has = v.has_value();
              if (v) *value = *v;
            },
            RLMESH_ERR_MODEL);
      };
    if constexpr (detail::is_detected<detail::loaded_execution_horizon_expression, P>)
      table.loaded_execution_horizon = [](void* u, bool* has, uint32_t* value) noexcept {
        return detail::callback(
            [&] {
              std::optional<uint32_t> v =
                  (**static_cast<std::shared_ptr<P>*>(u)).loaded_execution_horizon();
              *has = v.has_value();
              if (v) *value = *v;
            },
            RLMESH_ERR_MODEL);
      };
    if constexpr (detail::is_detected<detail::policy_reset_options_expression, P>)
      table.reset_options = [](void* u, const RLMeshValue* episode, RLMeshValue** out) noexcept {
        return detail::callback(
            [&] {
              std::optional<InfoDict> options =
                  (**static_cast<std::shared_ptr<P>*>(u)).reset_options(info(episode));
              if (options) *out = Value(std::move(*options)).clone_raw();
            },
            RLMESH_ERR_MODEL);
      };
    if constexpr (detail::is_detected<detail::on_episode_end_id_expression, P> ||
                  detail::is_detected<detail::on_episode_end_expression, P>)
      table.on_episode_end = [](void* u, const char* id) noexcept {
        return detail::callback(
            [&] {
              auto& p = **static_cast<std::shared_ptr<P>*>(u);
              if constexpr (detail::is_detected<detail::on_episode_end_id_expression, P>)
                p.on_episode_end(sv(id));
              else
                p.on_episode_end();
            },
            RLMESH_ERR_MODEL);
      };
    if constexpr (detail::is_detected<detail::close_expression, P>)
      table.close = [](void* u) noexcept {
        return detail::callback([&] { (**static_cast<std::shared_ptr<P>*>(u)).close(); },
                                RLMESH_ERR_MODEL);
      };
    table.free_user_data = delete_state<std::shared_ptr<P>>;
  }
};
}  // namespace detail
template <class P>
inline constexpr bool Policy = detail::has_predict<P> || detail::has_predict_batch<P> ||
                               detail::has_predict_chunk<P> || detail::has_predict_chunk_batch<P>;
class EnvTarget;
class Binding;
struct BindOptions;
struct RunOptions;
struct RunResult;
class Model {
  friend class ModelTarget;
  std::shared_ptr<RLMeshModel> handle_{};
  std::optional<SpecArg> spec_{};
  void read_spec() {
    RLMeshSpecArg c{};
    detail::check_status(rlmesh_model_spec(raw(), &c));
    if (c.kind == RLMESH_SPEC_KIND_MODEL_SPEC)
      spec_ = adapters::ModelSpec(rlmesh_model_spec_clone(c.spec));
    else if (c.kind == RLMESH_SPEC_KIND_NO_ADAPTER)
      spec_ = NO_ADAPTER;
  }

 public:
  explicit Model(RLMeshModel* source) : handle_(source, rlmesh_model_free) { read_spec(); }
  template <class P, std::enable_if_t<!std::is_base_of_v<Model, detail::remove_cvref_t<P>> &&
                                          Policy<detail::remove_cvref_t<P>>,
                                      int> = 0>
  explicit Model(P&& source, ModelOptions options = {})
      : Model(std::make_shared<detail::remove_cvref_t<P>>(std::forward<P>(source)),
              std::move(options)) {}
  // Invalid policy objects get a deleted-constructor diagnostic without
  // participating as constructible handles or intercepting null pointer
  // constants.
  template <class P,
            std::enable_if_t<std::is_class_v<detail::remove_cvref_t<P>> &&
                                 !std::is_base_of_v<Model, detail::remove_cvref_t<P>> &&
                                 !Policy<detail::remove_cvref_t<P>> &&
                                 !detail::IsPolicyPointer<detail::remove_cvref_t<P>>::value &&
                                 !std::is_convertible_v<detail::remove_cvref_t<P>, RLMeshModel*>,
                             int> = 0>
  explicit Model(P&&, ModelOptions = {}) = delete;
  template <class P, std::enable_if_t<Policy<P>, int> = 0>
  explicit Model(std::unique_ptr<P> source, ModelOptions options = {})
      : Model(std::shared_ptr<P>(std::move(source)), std::move(options)) {}
  template <class P, std::enable_if_t<Policy<P>, int> = 0>
  explicit Model(std::shared_ptr<P> source, ModelOptions options = {}) {
    static_assert(Policy<P>,
                  "rlmesh::Model: the policy type has none of "
                  "predict/predict_batch/predict_chunk/predict_chunk_batch");
    if (!source) throw TypeError("policy must not be null");
    auto state = std::make_unique<std::shared_ptr<P>>(std::move(source));
    detail::PolicyTable<P> vt;
    RLMeshModelOptions c = RLMESH_MODEL_OPTIONS_INIT;
    c.spec = detail::spec_arg(options.spec);
    c.trust_entrypoints = options.trust_entrypoints;
    auto hooks = std::make_unique<ModelOptions>(std::move(options));
    c.hook_user_data = hooks.get();
    c.free_hook_user_data = detail::delete_state<ModelOptions>;
    if (hooks->on_episode_end)
      c.on_episode_end = [](void* u, const char* id) noexcept {
        return detail::callback(
            [&] { static_cast<ModelOptions*>(u)->on_episode_end(detail::sv(id)); },
            RLMESH_ERR_MODEL);
      };
    if (hooks->on_close)
      c.on_close = [](void* u) noexcept {
        return detail::callback([&] { static_cast<ModelOptions*>(u)->on_close(); },
                                RLMESH_ERR_MODEL);
      };
    RLMeshModel* h = nullptr;
    detail::check_status(rlmesh_model_new(&vt.table, state.get(), &c, &h));
    state.release();
    hooks.release();
    handle_ = {h, rlmesh_model_free};
    read_spec();
  }
  /// A model served elsewhere, reached by its participant address
  /// ("lab:7447/openpi"), as Python's Model(address).
  static Model from_address(std::string_view address) {
    RLMeshModel* h = nullptr;
    detail::check_status(rlmesh_model_from_address(std::string(address).c_str(), &h));
    return Model(h);
  }
  template <class P>
  static Model from_config(const Kwargs& config) {
    auto state = std::make_unique<std::shared_ptr<P>>(std::make_shared<P>());
    detail::PolicyTable<P> vt;
    Value c(config);
    RLMeshModel* h = nullptr;
    detail::check_status(rlmesh_model_from_config(&vt.table, state.get(), c.raw(), &h));
    state.release();
    return Model(h);
  }
  template <class P>
  static Metadata describe() {
    detail::PolicyTable<P> vt;
    RLMeshValue* v = nullptr;
    detail::check_status(rlmesh_model_vtable_describe(&vt.table, nullptr, &v));
    return Value::adopt(v).as_dict();
  }
  RLMeshModel* raw() const { return handle_.get(); }
  const std::optional<SpecArg>& spec() const { return spec_; }
  bool allow_fusion() const { return rlmesh_model_allow_fusion(raw()); }
  std::optional<uint32_t> native_chunk() const {
    bool has = false;
    uint32_t n = 0;
    rlmesh_model_native_chunk(raw(), &has, &n);
    return has ? std::optional(n) : std::nullopt;
  }
  std::optional<uint32_t> execution_horizon() const {
    bool has = false;
    uint32_t n = 0;
    rlmesh_model_execution_horizon(raw(), &has, &n);
    return has ? std::optional(n) : std::nullopt;
  }
  std::optional<std::string> workflow_edition() const {
    bool has = false;
    detail::Bytes b;
    detail::check_status(rlmesh_model_workflow_edition(raw(), &has, &b.value));
    return has ? std::optional(b.text()) : std::nullopt;
  }
  void close() { detail::check_status(rlmesh_model_close(raw())); }
  void serve(std::string_view address, const ModelServeOptions& options = {}) {
    RLMeshModelServeOptions c = RLMESH_MODEL_SERVE_OPTIONS_INIT;
    std::optional<RLMeshServeOptions> so;
    if (options.options) {
      so = options.options->raw();
      c.options = &*so;
    }
    if (options.cancel) c.cancel = options.cancel->raw();
    c.name = detail::ptr(options.name);
    RLMeshExtModelServeOptions extra = RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT;
    if (options.ext) {
      extra.token = detail::ptr(options.ext->token);
      extra.no_signal_handlers = options.ext->no_signal_handlers;
      extra.has_predict_concurrency = options.ext->predict_concurrency.has_value();
      if (options.ext->predict_concurrency)
        extra.predict_concurrency = *options.ext->predict_concurrency;
      c.ext = &extra;
    }
    detail::check_status(rlmesh_model_serve(raw(), std::string(address).c_str(), &c));
  }
  RunResult run([[maybe_unused]] EnvTarget env_or_address,
                [[maybe_unused]] const RunOptions& options);
  RunResult run([[maybe_unused]] EnvTarget env_or_address);
  Binding bind([[maybe_unused]] EnvTarget target, [[maybe_unused]] const BindOptions& options);
  Binding bind([[maybe_unused]] EnvTarget target);
};
struct RemoteEnvOptions {
  std::optional<std::string> address{}, host{};
  std::optional<uint16_t> port{};
  std::optional<std::string> path{};
  std::optional<Transport> transport{};
  std::optional<Duration> connect_timeout{}, request_timeout{};
  std::optional<std::string> workflow_edition{};
  /* Sent as `authorization: Bearer <key>`; unset reads RLMESH_API_KEY for an
   * https:// address only, "" sends none. Never printed. */
  std::optional<std::string> api_key{};
  /* A PEM bundle an https:// dial trusts instead of the system roots. */
  std::optional<std::string> tls_ca_file{};
  RLMeshRemoteEnvOptions raw() const {
    RLMeshRemoteEnvOptions c = RLMESH_REMOTE_ENV_OPTIONS_INIT;
    c.address = detail::ptr(address);
    c.host = detail::ptr(host);
    c.path = detail::ptr(path);
    c.has_port = port.has_value();
    if (port) c.port = *port;
    c.has_transport = transport.has_value();
    if (transport) c.transport = static_cast<RLMeshTransport>(*transport);
    c.has_connect_timeout_seconds = connect_timeout.has_value();
    if (connect_timeout) c.connect_timeout_seconds = connect_timeout->count();
    c.has_request_timeout_seconds = request_timeout.has_value();
    if (request_timeout) c.request_timeout_seconds = request_timeout->count();
    c.workflow_edition = detail::ptr(workflow_edition);
    c.api_key = detail::ptr(api_key);
    c.tls_ca_file = detail::ptr(tls_ca_file);
    return c;
  }
  RemoteEnvOptions& with_address(std::optional<std::string> value) & {
    address = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_address(std::optional<std::string> value) && {
    address = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_port(std::optional<uint16_t> value) & {
    port = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_port(std::optional<uint16_t> value) && {
    port = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_path(std::optional<std::string> value) & {
    path = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_path(std::optional<std::string> value) && {
    path = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_transport(std::optional<Transport> value) & {
    transport = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_transport(std::optional<Transport> value) && {
    transport = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_connect_timeout(std::optional<Duration> value) & {
    connect_timeout = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_connect_timeout(std::optional<Duration> value) && {
    connect_timeout = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_workflow_edition(std::optional<std::string> value) & {
    workflow_edition = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_workflow_edition(std::optional<std::string> value) && {
    workflow_edition = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_host(std::optional<std::string> value) & {
    host = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_host(std::optional<std::string> value) && {
    host = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_request_timeout(std::optional<Duration> value) & {
    request_timeout = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_request_timeout(std::optional<Duration> value) && {
    request_timeout = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_api_key(std::optional<std::string> value) & {
    api_key = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_api_key(std::optional<std::string> value) && {
    api_key = std::move(value);
    return std::move(*this);
  }
  RemoteEnvOptions& with_tls_ca_file(std::optional<std::string> value) & {
    tls_ca_file = std::move(value);
    return *this;
  }
  RemoteEnvOptions with_tls_ca_file(std::optional<std::string> value) && {
    tls_ca_file = std::move(value);
    return std::move(*this);
  }
};
namespace detail {
// Own every C output before shared_ptr adoption can allocate or throw.
inline std::pair<Value, Value> adopt_values(RLMeshValue* observation, RLMeshValue* info) {
  auto observation_owner = own<RLMeshValue, rlmesh_value_free>(observation);
  auto info_owner = own<RLMeshValue, rlmesh_value_free>(info);
  auto first = Value::adopt(observation_owner.release());
  auto second = Value::adopt(info_owner.release());
  return {std::move(first), std::move(second)};
}
}  // namespace detail
class RemoteEnv {
  detail::Owned<RLMeshRemoteEnv, rlmesh_remote_env_free> handle_{nullptr, rlmesh_remote_env_free};
  std::optional<specs::EnvContract> contract_{};
  std::optional<spaces::Space> observation_{}, action_{};
  std::optional<spaces::SpaceSpec> observation_spec_{}, action_spec_{};
  Metadata metadata_{};

 public:
  explicit RemoteEnv(const RemoteEnvOptions& options = {}) {
    auto c = options.raw();
    RLMeshRemoteEnv* p = nullptr;
    detail::check_status(rlmesh_remote_env_new(&c, &p));
    handle_.reset(p);
    contract_ = specs::EnvContract::copy_of(rlmesh_remote_env_env_contract(p));
    observation_ = spaces::Space::adopt(rlmesh_space_clone(rlmesh_remote_env_observation_space(p)));
    action_ = spaces::Space::adopt(rlmesh_space_clone(rlmesh_remote_env_action_space(p)));
    observation_spec_ = spaces::SpaceSpec::copy_of(rlmesh_remote_env_observation_space_spec(p));
    action_spec_ = spaces::SpaceSpec::copy_of(rlmesh_remote_env_action_space_spec(p));
    metadata_ = detail::info(rlmesh_remote_env_metadata(p));
  }
  RemoteEnv(RemoteEnv&&) = default;
  RemoteEnv& operator=(RemoteEnv&&) = default;
  RLMeshRemoteEnv* raw() const { return handle_.get(); }
  std::string address() const { return detail::text(rlmesh_remote_env_address, raw()); }
  std::string env_id() const { return detail::text(rlmesh_remote_env_env_id, raw()); }
  std::string selected_workflow_edition() const {
    return detail::text(rlmesh_remote_env_selected_workflow_edition, raw());
  }
  const specs::EnvContract& env_contract() const { return *contract_; }
  const spaces::Space& observation_space() const { return *observation_; }
  const spaces::Space& action_space() const { return *action_; }
  const spaces::SpaceSpec& observation_space_spec() const { return *observation_spec_; }
  const spaces::SpaceSpec& action_space_spec() const { return *action_spec_; }
  const Metadata& metadata() const { return metadata_; }
  std::optional<std::string> render_mode() const {
    bool has = false;
    detail::Bytes b;
    detail::check_status(rlmesh_remote_env_render_mode(raw(), &has, &b.value));
    return has ? std::optional(b.text()) : std::nullopt;
  }
  ext::CancelToken cancel_handle() const {
    RLMeshExtCancelToken* p = nullptr;
    detail::check_status(rlmesh_remote_env_cancel_handle(raw(), &p));
    return ext::CancelToken(p);
  }
  std::optional<Value> render(uint64_t env_index = 0) {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_remote_env_render(raw(), env_index, &p));
    if (p) return Value::adopt(p);
    return {};
  }
  bool shutdown(std::string_view reason = "owner shutdown") {
    bool stopped = false;
    detail::check_status(rlmesh_remote_env_shutdown(raw(), std::string(reason).c_str(), &stopped));
    return stopped;
  }
  void close() { detail::check_status(rlmesh_remote_env_close(raw())); }
  ResetResult reset(std::optional<int64_t> seed = {}, const std::optional<InfoDict>& options = {}) {
    std::optional<Value> v;
    if (options) v = Value(*options);
    RLMeshValue* o = nullptr;
    RLMeshValue* info = nullptr;
    detail::check_status(
        rlmesh_remote_env_reset(raw(), detail::ptr(seed), v ? v->raw() : nullptr, &o, &info));
    auto [observation, i] = detail::adopt_values(o, info);
    return {std::move(observation), i.as_dict()};
  }
  StepResult step(const Value& action) {
    RLMeshValue* o = nullptr;
    RLMeshValue* info = nullptr;
    double r = 0;
    bool t = false, tr = false;
    detail::check_status(rlmesh_remote_env_step(raw(), action.raw(), &o, &r, &t, &tr, &info));
    auto [observation, i] = detail::adopt_values(o, info);
    return {std::move(observation), r, t, tr, i.as_dict()};
  }
};
class RemoteVectorEnv {
  detail::Owned<RLMeshRemoteVectorEnv, rlmesh_remote_vector_env_free> handle_{
      nullptr, rlmesh_remote_vector_env_free};
  std::optional<specs::EnvContract> contract_{};
  std::optional<spaces::Space> observation_{}, action_{};
  std::optional<spaces::SpaceSpec> observation_spec_{}, action_spec_{};
  Metadata metadata_{};

 public:
  explicit RemoteVectorEnv(const RemoteEnvOptions& options = {}) {
    auto c = options.raw();
    RLMeshRemoteVectorEnv* p = nullptr;
    detail::check_status(rlmesh_remote_vector_env_new(&c, &p));
    handle_.reset(p);
    contract_ = specs::EnvContract::copy_of(rlmesh_remote_vector_env_env_contract(p));
    observation_ =
        spaces::Space::adopt(rlmesh_space_clone(rlmesh_remote_vector_env_observation_space(p)));
    action_ = spaces::Space::adopt(rlmesh_space_clone(rlmesh_remote_vector_env_action_space(p)));
    observation_spec_ =
        spaces::SpaceSpec::copy_of(rlmesh_remote_vector_env_observation_space_spec(p));
    action_spec_ = spaces::SpaceSpec::copy_of(rlmesh_remote_vector_env_action_space_spec(p));
    metadata_ = detail::info(rlmesh_remote_vector_env_metadata(p));
  }
  RemoteVectorEnv(RemoteVectorEnv&&) = default;
  RemoteVectorEnv& operator=(RemoteVectorEnv&&) = default;
  RLMeshRemoteVectorEnv* raw() const { return handle_.get(); }
  std::string address() const { return detail::text(rlmesh_remote_vector_env_address, raw()); }
  std::string env_id() const { return detail::text(rlmesh_remote_vector_env_env_id, raw()); }
  std::string selected_workflow_edition() const {
    return detail::text(rlmesh_remote_vector_env_selected_workflow_edition, raw());
  }
  const specs::EnvContract& env_contract() const { return *contract_; }
  const spaces::Space& observation_space() const { return *observation_; }
  const spaces::Space& action_space() const { return *action_; }
  const spaces::SpaceSpec& observation_space_spec() const { return *observation_spec_; }
  const spaces::SpaceSpec& action_space_spec() const { return *action_spec_; }
  const Metadata& metadata() const { return metadata_; }
  std::optional<std::string> render_mode() const {
    bool has = false;
    detail::Bytes b;
    detail::check_status(rlmesh_remote_vector_env_render_mode(raw(), &has, &b.value));
    return has ? std::optional(b.text()) : std::nullopt;
  }
  ext::CancelToken cancel_handle() const {
    RLMeshExtCancelToken* p = nullptr;
    detail::check_status(rlmesh_remote_vector_env_cancel_handle(raw(), &p));
    return ext::CancelToken(p);
  }
  std::optional<Value> render(uint64_t env_index = 0) {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_remote_vector_env_render(raw(), env_index, &p));
    if (p) return Value::adopt(p);
    return {};
  }
  bool shutdown(std::string_view reason = "owner shutdown") {
    bool stopped = false;
    detail::check_status(
        rlmesh_remote_vector_env_shutdown(raw(), std::string(reason).c_str(), &stopped));
    return stopped;
  }
  void close() { detail::check_status(rlmesh_remote_vector_env_close(raw())); }
  ResetResult reset(std::optional<VectorSeed> seed = {},
                    const std::optional<InfoDict>& options = {}) {
    std::optional<Value> v;
    if (options) v = Value(*options);
    RLMeshValue* o = nullptr;
    RLMeshValue* info = nullptr;
    std::vector<int64_t> seeds;
    if (seed) {
      if (auto p = std::get_if<int64_t>(&*seed))
        seeds.push_back(*p);
      else
        seeds = std::get<std::vector<int64_t>>(*seed);
    }
    int64_t empty = 0;
    detail::check_status(rlmesh_remote_vector_env_reset(
        raw(), seed ? (seeds.empty() ? &empty : seeds.data()) : nullptr, seeds.size(),
        v ? v->raw() : nullptr, &o, &info));
    auto [observation, i] = detail::adopt_values(o, info);
    return {std::move(observation), i.as_dict()};
  }
  uint64_t num_envs() const { return rlmesh_remote_vector_env_num_envs(raw()); }
  const spaces::Space& single_observation_space() const { return observation_space(); }
  const spaces::Space& single_action_space() const { return action_space(); }
  VectorStepResult step(const Value& actions) {
    const auto n = static_cast<size_t>(num_envs());
    std::vector<double> rewards(n);
    auto t = std::make_unique<bool[]>(n);
    auto tr = std::make_unique<bool[]>(n);
    RLMeshValue* o = nullptr;
    RLMeshValue* info = nullptr;
    size_t len = 0;
    detail::check_status(rlmesh_remote_vector_env_step(raw(), actions.raw(), &o, rewards.data(),
                                                       t.get(), tr.get(), n, &len, &info));
    auto [observation, i] = detail::adopt_values(o, info);
    return {std::move(observation), std::move(rewards), std::vector<bool>(t.get(), t.get() + len),
            std::vector<bool>(tr.get(), tr.get() + len), i.as_dict()};
  }
};
struct RandomSample {
  friend constexpr bool operator==(const RandomSample&, const RandomSample&) noexcept {
    return true;
  }
  friend constexpr bool operator!=(const RandomSample& self,
                                   const RandomSample& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline constexpr RandomSample RANDOM_SAMPLE{};
inline std::string to_string(RandomSample) { return "RANDOM_SAMPLE"; }
inline std::ostream& operator<<(std::ostream& s, RandomSample) { return s << "RANDOM_SAMPLE"; }
class ModelTarget {
  friend class Binding;
  RLMeshModelTarget target_{};
  std::shared_ptr<void> keepalive_{};

 public:
  ModelTarget(const Model& m) : target_(rlmesh_model_target(m.raw())), keepalive_(m.handle_) {}
  ModelTarget(RandomSample) : target_(RLMESH_RANDOM_SAMPLE) {}
  RLMeshModelTarget raw() const { return target_; }
};
class EnvTarget {
  friend class Binding;
  RLMeshEnvTarget target_{};
  std::shared_ptr<void> keepalive_{};
  std::string address_{};

 public:
  EnvTarget(const char* s) : EnvTarget(std::string_view(s)) {}
  EnvTarget(const std::string& s) : EnvTarget(std::string_view(s)) {}
  EnvTarget(std::string_view s) : target_(rlmesh_address_target(nullptr)), address_(s) {}
  EnvTarget(const EnvServer& s)
      : target_(rlmesh_env_server_target(s.raw())), keepalive_(s.state_) {}
  EnvTarget(RemoteEnv& e) : target_(rlmesh_remote_env_target(e.raw())) {}
  EnvTarget(RemoteVectorEnv& e) : target_(rlmesh_remote_vector_env_target(e.raw())) {}
  EnvTarget(Env& e) : target_(rlmesh_env_target(e.raw())) {}
  EnvTarget(VectorEnv& e) : target_(rlmesh_vector_env_target(e.raw())) {}
  EnvTarget(EnvFactoryBase& f) : target_(rlmesh_env_factory_target(f.handle())) {}
  RLMeshEnvTarget raw() const {
    auto c = target_;
    if (c.kind == RLMESH_ENV_TARGET_KIND_ADDRESS) c.address = address_.c_str();
    return c;
  }
};
struct EpisodeResult {
  int64_t index = 0;
  std::optional<int64_t> seed{};
  int64_t steps = 0;
  double reward = 0;
  bool terminated = false, truncated = false;
  std::optional<bool> success{};
  double duration_s = 0;
  std::optional<double> predict_ms{}, step_ms{};
  std::optional<int64_t> trial{};

  RLMeshEpisodeResult raw() const {
    RLMeshEpisodeResult c = RLMESH_EPISODE_RESULT_INIT;
    c.index = index;
    c.has_seed = seed.has_value();
    if (seed) c.seed = *seed;
    c.steps = steps;
    c.reward = reward;
    c.terminated = terminated;
    c.truncated = truncated;
    c.has_success = success.has_value();
    if (success) c.success = *success;
    c.duration_s = duration_s;
    c.has_predict_ms = predict_ms.has_value();
    if (predict_ms) c.predict_ms = *predict_ms;
    c.has_step_ms = step_ms.has_value();
    if (step_ms) c.step_ms = *step_ms;
    c.has_trial = trial.has_value();
    if (trial) c.trial = *trial;
    return c;
  }
  static EpisodeResult from_raw(const RLMeshEpisodeResult& c) {
    return {c.index,
            c.has_seed ? std::optional(c.seed) : std::nullopt,
            c.steps,
            c.reward,
            c.terminated,
            c.truncated,
            c.has_success ? std::optional(c.success) : std::nullopt,
            c.duration_s,
            c.has_predict_ms ? std::optional(c.predict_ms) : std::nullopt,
            c.has_step_ms ? std::optional(c.step_ms) : std::nullopt,
            c.has_trial ? std::optional(c.trial) : std::nullopt};
  }

  friend constexpr bool operator==(const EpisodeResult& self, const EpisodeResult& other) noexcept(
      detail::nothrow_equal<decltype(index), decltype(seed), decltype(steps), decltype(reward),
                            decltype(terminated), decltype(truncated), decltype(success),
                            decltype(duration_s), decltype(predict_ms), decltype(step_ms),
                            decltype(trial)>) {
    return self.index == other.index && self.seed == other.seed && self.steps == other.steps &&
           self.reward == other.reward && self.terminated == other.terminated &&
           self.truncated == other.truncated && self.success == other.success &&
           self.duration_s == other.duration_s && self.predict_ms == other.predict_ms &&
           self.step_ms == other.step_ms && self.trial == other.trial;
  }
  friend constexpr bool operator!=(const EpisodeResult& self,
                                   const EpisodeResult& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const EpisodeResult& v) {
  auto c = v.raw();
  return detail::text(rlmesh_episode_result_repr, &c);
}
struct TelemetryRow {
  std::string op{}, component{}, metric{}, unit{};
  int64_t count = 0;
  double avg = 0, p50 = 0, p95 = 0, p99 = 0;

  RLMeshTelemetryRow* raw() const {
    RLMeshTelemetryRow* p = nullptr;
    detail::check_status(rlmesh_telemetry_row_new(op.c_str(), component.c_str(), metric.c_str(),
                                                  unit.c_str(), count, avg, p50, p95, p99, &p));
    return p;
  }
  static TelemetryRow from_raw(const RLMeshTelemetryRow* c) {
    return {detail::text(rlmesh_telemetry_row_op, c),
            detail::text(rlmesh_telemetry_row_component, c),
            detail::text(rlmesh_telemetry_row_metric, c),
            detail::text(rlmesh_telemetry_row_unit, c),
            rlmesh_telemetry_row_count(c),
            rlmesh_telemetry_row_avg(c),
            rlmesh_telemetry_row_p50(c),
            rlmesh_telemetry_row_p95(c),
            rlmesh_telemetry_row_p99(c)};
  }

  friend bool operator==(const TelemetryRow& self, const TelemetryRow& other) noexcept(
      detail::nothrow_equal<decltype(op), decltype(component), decltype(metric), decltype(unit),
                            decltype(count), decltype(avg), decltype(p50), decltype(p95),
                            decltype(p99)>) {
    return self.op == other.op && self.component == other.component &&
           self.metric == other.metric && self.unit == other.unit && self.count == other.count &&
           self.avg == other.avg && self.p50 == other.p50 && self.p95 == other.p95 &&
           self.p99 == other.p99;
  }
  friend bool operator!=(const TelemetryRow& self,
                         const TelemetryRow& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const TelemetryRow& v) {
  auto p = detail::own<RLMeshTelemetryRow, rlmesh_telemetry_row_free>(v.raw());
  return detail::text(rlmesh_telemetry_row_repr, p.get());
}
// The labels of the one workload RunResult::save writes.
struct RunResultSaveOptions {
  std::string model = "model";
  std::string env = "env";
  std::optional<Metadata> config{};
  RunResultSaveOptions& with_model(std::string value) & {
    model = std::move(value);
    return *this;
  }
  RunResultSaveOptions with_model(std::string value) && {
    model = std::move(value);
    return std::move(*this);
  }
  RunResultSaveOptions& with_env(std::string value) & {
    env = std::move(value);
    return *this;
  }
  RunResultSaveOptions with_env(std::string value) && {
    env = std::move(value);
    return std::move(*this);
  }
  RunResultSaveOptions& with_config(std::optional<Metadata> value) & {
    config = std::move(value);
    return *this;
  }
  RunResultSaveOptions with_config(std::optional<Metadata> value) && {
    config = std::move(value);
    return std::move(*this);
  }
};
struct RunResult {
  std::vector<EpisodeResult> episodes{};
  std::vector<TelemetryRow> telemetry{};
  std::vector<adapters::Advisory> advisories{};

  size_t num_episodes() const { return episodes.size(); }
  int64_t total_steps() const {
    int64_t n = 0;
    for (auto& e : episodes) n += e.steps;
    return n;
  }
  double mean_reward() const {
    double n = 0;
    for (auto& e : episodes) n += e.reward;
    return episodes.empty() ? 0 : n / static_cast<double>(episodes.size());
  }
  std::optional<double> success_rate() const {
    if (episodes.empty()) return {};
    double successes = 0;
    for (const auto& e : episodes) {
      if (!e.success) return {};
      if (*e.success) ++successes;
    }
    return successes / static_cast<double>(episodes.size());
  }

  RLMeshRunResult* raw() const {
    std::vector<RLMeshEpisodeResult> ep;
    std::vector<const RLMeshEpisodeResult*> eps;
    for (auto& e : episodes) ep.push_back(e.raw());
    for (auto& e : ep) eps.push_back(&e);
    std::vector<detail::Owned<RLMeshTelemetryRow, rlmesh_telemetry_row_free>> ts;
    std::vector<const RLMeshTelemetryRow*> tp;
    for (auto& t : telemetry) {
      ts.push_back(detail::own<RLMeshTelemetryRow, rlmesh_telemetry_row_free>(t.raw()));
      tp.push_back(ts.back().get());
    }
    std::vector<RLMeshAdvisory> as;
    std::vector<const RLMeshAdvisory*> ap;
    for (auto& a : advisories) as.push_back(a.raw());
    for (auto& a : as) ap.push_back(&a);
    RLMeshRunResultOptions c = RLMESH_RUN_RESULT_OPTIONS_INIT;
    c.episodes = eps.data();
    c.episodes_len = eps.size();
    c.telemetry = tp.data();
    c.telemetry_len = tp.size();
    c.advisories = ap.data();
    c.advisories_len = ap.size();
    RLMeshRunResult* p = nullptr;
    detail::check_status(rlmesh_run_result_new(&c, &p));
    return p;
  }
  static RunResult from_raw(const RLMeshRunResult* p) {
    RunResult v;
    for (size_t i = 0; i < rlmesh_run_result_episodes_len(p); ++i) {
      RLMeshEpisodeResult c = RLMESH_EPISODE_RESULT_INIT;
      detail::check_status(rlmesh_run_result_episodes_get(p, i, &c));
      v.episodes.push_back(EpisodeResult::from_raw(c));
    }
    for (size_t i = 0; i < rlmesh_run_result_telemetry_len(p); ++i) {
      RLMeshTelemetryRow* c = nullptr;
      detail::check_status(rlmesh_run_result_telemetry_get(p, i, &c));
      auto owned = detail::own<RLMeshTelemetryRow, rlmesh_telemetry_row_free>(c);
      v.telemetry.push_back(TelemetryRow::from_raw(c));
    }
    for (size_t i = 0; i < rlmesh_run_result_advisories_len(p); ++i) {
      RLMeshAdvisory c = RLMESH_ADVISORY_INIT;
      detail::check_status(rlmesh_run_result_advisories_get(p, i, &c));
      v.advisories.push_back({static_cast<adapters::AdvisorySeverity>(c.severity),
                              std::string(detail::sv(c.message))});
    }
    return v;
  }
  static RunResult take(RLMeshRunResult* p) {
    auto owned = detail::own<RLMeshRunResult, rlmesh_run_result_free>(p);
    return from_raw(p);
  }
  std::string format_telemetry() const {
    auto p = detail::own<RLMeshRunResult, rlmesh_run_result_free>(raw());
    return detail::text(rlmesh_run_result_format_telemetry, p.get());
  }
  // Writes this result as an rlmesh.result.v1 bundle (a folder, or a zip when
  // path ends .zip) and returns its path.
  std::filesystem::path save(const std::filesystem::path& path,
                             const RunResultSaveOptions& options = {}) const {
    auto p = detail::own<RLMeshRunResult, rlmesh_run_result_free>(raw());
    Value cfg(options.config.value_or(Metadata{}));
    RLMeshRunResultSaveOptions c = RLMESH_RUN_RESULT_SAVE_OPTIONS_INIT;
    c.model = options.model.c_str();
    c.env = options.env.c_str();
    c.config = options.config ? cfg.raw() : nullptr;
    return detail::text(rlmesh_run_result_save, p.get(), path.string().c_str(), &c);
  }
  // The result a one-workload rlmesh.result.v1 bundle (folder or zip) holds.
  static RunResult load(const std::filesystem::path& path) {
    RLMeshRunResult* p = nullptr;
    detail::check_status(rlmesh_run_result_load(path.string().c_str(), &p));
    return take(p);
  }

  friend bool operator==(const RunResult& self, const RunResult& other) noexcept(
      detail::nothrow_equal<decltype(episodes), decltype(telemetry), decltype(advisories)>) {
    return self.episodes == other.episodes && self.telemetry == other.telemetry &&
           self.advisories == other.advisories;
  }
  friend bool operator!=(const RunResult& self,
                         const RunResult& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const RunResult& v) {
  auto p = detail::own<RLMeshRunResult, rlmesh_run_result_free>(v.raw());
  return detail::text(rlmesh_run_result_repr, p.get());
}
class ReadItem {
  std::shared_ptr<RLMeshReadItem> handle_{};

 public:
  explicit ReadItem(RLMeshReadItem* p) : handle_(p, rlmesh_read_item_free) {}
  ReadItem(RLMeshReadItem* p, bool) : handle_(p, [](RLMeshReadItem*) {}) {}
  ReadItem(const char* role) : ReadItem(std::string_view(role)) {}
  ReadItem(const std::string& role) : ReadItem(std::string_view(role)) {}
  ReadItem(std::string_view role) {
    RLMeshReadItem* p = nullptr;
    detail::check_status(rlmesh_read_item_role(std::string(role).c_str(), &p));
    handle_ = {p, rlmesh_read_item_free};
  }
  ReadItem(const adapters::Image& v) {
    adapters::detail_adapter::Arena a;
    auto c = v.raw(a);
    RLMeshReadItem* p = nullptr;
    detail::check_status(rlmesh_read_item_image(&c, &p));
    handle_ = {p, rlmesh_read_item_free};
  }
  ReadItem(const adapters::State& v) {
    adapters::detail_adapter::Arena a;
    auto c = v.raw(a);
    RLMeshReadItem* p = nullptr;
    detail::check_status(rlmesh_read_item_state(&c, &p));
    handle_ = {p, rlmesh_read_item_free};
  }
  ReadItem(const adapters::Text& v) {
    adapters::detail_adapter::Arena a;
    auto c = v.raw(a);
    RLMeshReadItem* p = nullptr;
    detail::check_status(rlmesh_read_item_text(&c, &p));
    handle_ = {p, rlmesh_read_item_free};
  }
  const RLMeshReadItem* raw() const { return handle_.get(); }
};
class Reader {
  std::shared_ptr<RLMeshReader> handle_{};
  std::vector<std::string> roles_{};

 public:
  explicit Reader(RLMeshReader* p) : handle_(p, rlmesh_reader_free) {
    for (size_t i = 0; i < rlmesh_reader_roles_len(p); ++i)
      roles_.push_back(detail::text(rlmesh_reader_roles_get, p, i));
  }
  Reader(adapters::Adapter adapter, std::vector<std::string> roles) : roles_(std::move(roles)) {
    std::vector<const char*> ptrs;
    for (auto& r : roles_) ptrs.push_back(r.c_str());
    RLMeshReader* p = nullptr;
    detail::check_status(rlmesh_reader_new(adapter.raw(), ptrs.data(), ptrs.size(), &p));
    handle_ = {p, rlmesh_reader_free};
  }
  const std::vector<std::string>& roles() const { return roles_; }
  Metadata operator()(const Value& v) const {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_reader_call(handle_.get(), v.raw(), &p));
    return Value::adopt(p).as_dict();
  }
};
template <class Signature, class EmptyError = ValueError>
class SharedFunction;
template <class R, class EmptyError, class... A>
class SharedFunction<R(A...), EmptyError> {
  std::shared_ptr<std::function<R(A...)>> function_{};

 public:
  SharedFunction() = default;
  template <class F, std::enable_if_t<!std::is_same_v<detail::remove_cvref_t<F>, SharedFunction> &&
                                          std::is_invocable_r_v<R, F&, A...>,
                                      int> = 0>
  SharedFunction(F f) : function_(std::make_shared<std::function<R(A...)>>(std::move(f))) {}
  explicit operator bool() const { return bool(function_); }
  R operator()(A... args) const {
    if (!function_) throw EmptyError("callable is empty");
    return (*function_)(std::forward<A>(args)...);
  }

  const void* identity() const { return function_.get(); }

  friend constexpr bool operator==(
      const SharedFunction& self,
      const SharedFunction& other) noexcept(noexcept(self.function_ == other.function_)) {
    return self.function_ == other.function_;
  }
  friend constexpr bool operator!=(const SharedFunction& self,
                                   const SharedFunction& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
using ReadFn = SharedFunction<Value(const ReadItem&)>;
using RenderFn = SharedFunction<Value()>;
struct FrameSources {
  std::vector<std::string> roles{};
  std::optional<std::string> render_label{};
  RenderFn render{};

  std::vector<std::string> labels() const {
    std::vector<std::string> v;
    if (render_label) v.push_back(*render_label);
    v.insert(v.end(), roles.begin(), roles.end());
    return v;
  }
  std::optional<std::string> default_label() const {
    if (!roles.empty()) return roles.front();
    return render_label;
  }
  RLMeshFrameSources* raw() const {
    std::vector<const char*> ptrs;
    for (auto& r : roles) ptrs.push_back(r.c_str());
    auto state = std::make_unique<RenderFn>(render);
    RLMeshFrameSources* p = nullptr;
    detail::check_status(rlmesh_frame_sources_new(ptrs.data(),ptrs.size(),detail::ptr(render_label),render?+[](void* u,RLMeshValue** o) noexcept { return detail::callback([&] { *o=(*static_cast<RenderFn*>(u))().clone_raw(); }); }:nullptr,state.get(),detail::delete_state<RenderFn>,&p));
    state.release();
    return p;
  }
  static FrameSources take(RLMeshFrameSources* p, std::shared_ptr<void> owner = {}) {
    auto handle = std::shared_ptr<RLMeshFrameSources>(p, rlmesh_frame_sources_free);
    FrameSources v;
    for (size_t i = 0; i < rlmesh_frame_sources_roles_len(p); ++i)
      v.roles.push_back(detail::text(rlmesh_frame_sources_roles_get, p, i));
    bool has = false;
    detail::Bytes b;
    detail::check_status(rlmesh_frame_sources_render_label(p, &b.value, &has));
    if (has) v.render_label = b.text();
    if (rlmesh_ext_frame_sources_has_render(p))
      v.render = [handle, owner] {
        (void)owner;
        RLMeshValue* frame = nullptr;
        bool has_frame = false;
        detail::check_status(rlmesh_frame_sources_render(handle.get(), &frame, &has_frame));
        if (!has_frame) return Value{};
        return Value::adopt(frame);
      };
    return v;
  }

  friend bool operator==(const FrameSources& self, const FrameSources& other) noexcept(
      detail::nothrow_equal<decltype(roles), decltype(render_label), decltype(render)>) {
    return self.roles == other.roles && self.render_label == other.render_label &&
           self.render == other.render;
  }
  friend bool operator!=(const FrameSources& self,
                         const FrameSources& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
class RunContext {
 public:
  virtual ~RunContext() = default;
  virtual std::optional<specs::EnvContract> contract() const = 0;
  virtual uint64_t num_envs() const = 0;
  virtual Value read([[maybe_unused]] const Value& observation,
                     [[maybe_unused]] const ReadItem& item) const = 0;
  virtual FrameSources frame_sources() const = 0;
};
namespace detail {
class NativeRunContext final : public RunContext {
  std::shared_ptr<RLMeshRunContext> handle_{};

 public:
  explicit NativeRunContext(RLMeshRunContext* p) : handle_(p, rlmesh_run_context_free) {}
  const RLMeshRunContext* raw() const { return handle_.get(); }
  std::optional<specs::EnvContract> contract() const override {
    RLMeshEnvContract* p = nullptr;
    check_status(rlmesh_run_context_contract(raw(), &p));
    if (!p) return {};
    return specs::EnvContract(p);
  }
  uint64_t num_envs() const override {
    uint64_t n = 0;
    check_status(rlmesh_run_context_num_envs(raw(), &n));
    return n;
  }
  Value read(const Value& v, const ReadItem& item) const override {
    RLMeshValue* p = nullptr;
    check_status(rlmesh_run_context_read(raw(), v.raw(), item.raw(), &p));
    return Value::adopt(p);
  }
  FrameSources frame_sources() const override {
    RLMeshFrameSources* p = nullptr;
    check_status(rlmesh_run_context_frame_sources(raw(), &p));
    return FrameSources::take(p, handle_);
  }
};
}  // namespace detail
struct StepEvent {
  int64_t episode = 0;
  std::optional<int64_t> seed{};
  int64_t step = 0;
  Value observation{};
  Value action{};
  double reward = 0;
  bool terminated = false, truncated = false;
  InfoDict info{};
  double predict_ms = 0, step_ms = 0;
  ReadFn read{};

  friend bool operator==(const StepEvent& self, const StepEvent& other) noexcept(
      detail::nothrow_equal<decltype(episode), decltype(seed), decltype(step),
                            decltype(observation), decltype(action), decltype(reward),
                            decltype(terminated), decltype(truncated), decltype(info),
                            decltype(predict_ms), decltype(step_ms), decltype(read)>) {
    return self.episode == other.episode && self.seed == other.seed && self.step == other.step &&
           self.observation == other.observation && self.action == other.action &&
           self.reward == other.reward && self.terminated == other.terminated &&
           self.truncated == other.truncated && self.info == other.info &&
           self.predict_ms == other.predict_ms && self.step_ms == other.step_ms &&
           self.read == other.read;
  }
  friend bool operator!=(const StepEvent& self,
                         const StepEvent& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
class RunHooks {
 public:
  virtual ~RunHooks() = default;
  virtual void on_run_start([[maybe_unused]] std::shared_ptr<RunContext> context) {}
  virtual void on_episode_start([[maybe_unused]] int64_t episode,
                                [[maybe_unused]] std::optional<int64_t> seed) {}
  virtual void on_step([[maybe_unused]] const StepEvent& event) {}
  virtual void on_episode_end([[maybe_unused]] const EpisodeResult& result) {}
  virtual void on_run_end([[maybe_unused]] const RunResult& result) {}
};
namespace detail {
inline Owned<RLMeshRunHooks, rlmesh_run_hooks_free> hooks(RunHooks* user_hooks) {
  if (!user_hooks) return own<RLMeshRunHooks, rlmesh_run_hooks_free>(nullptr);
  RLMeshRunHooksVtable c = RLMESH_RUN_HOOKS_VTABLE_INIT;
  c.on_run_start = [](void* u, const RLMeshRunContext* p) noexcept {
    return callback([&] {
      static_cast<RunHooks*>(u)->on_run_start(
          std::make_shared<NativeRunContext>(rlmesh_run_context_clone(p)));
    });
  };
  c.on_episode_start = [](void* u, int64_t e, const int64_t* seed) noexcept {
    return callback([&] {
      static_cast<RunHooks*>(u)->on_episode_start(e, seed ? std::optional(*seed) : std::nullopt);
    });
  };
  c.on_step = [](void* u, const RLMeshStepEvent* e) noexcept {
    return callback([&] {
      auto context = std::make_shared<NativeRunContext>(rlmesh_run_context_clone(e->context));
      auto observation = Value::copy_of(e->observation);
      StepEvent v{e->episode,
                  e->has_seed ? std::optional(e->seed) : std::nullopt,
                  e->step,
                  observation,
                  Value::copy_of(e->action),
                  e->reward,
                  e->terminated,
                  e->truncated,
                  info(e->info),
                  e->predict_ms,
                  e->step_ms,
                  [context, observation](const ReadItem& item) {
                    return context->read(observation, item);
                  }};
      static_cast<RunHooks*>(u)->on_step(v);
    });
  };
  c.on_episode_end = [](void* u, const RLMeshEpisodeResult* e) noexcept {
    return callback(
        [&] { static_cast<RunHooks*>(u)->on_episode_end(EpisodeResult::from_raw(*e)); });
  };
  c.on_run_end = [](void* u, const RLMeshRunResult* r) noexcept {
    return callback([&] { static_cast<RunHooks*>(u)->on_run_end(RunResult::from_raw(r)); });
  };
  RLMeshRunHooks* h = nullptr;
  check_status(rlmesh_run_hooks_new(&c, user_hooks, &h));
  return own<RLMeshRunHooks, rlmesh_run_hooks_free>(h);
}
}  // namespace detail
enum class ViewBackend { Terminal = 1, Http = 2, Both = 3 };
enum class ViewFormat { Jpeg = 1, Png = 2 };
struct View {
  ViewBackend backend = ViewBackend::Terminal;
  uint16_t port = 8008;
  uint64_t fps = 30;
  std::optional<std::string> source{};
  ViewFormat format = ViewFormat::Jpeg;
  uint8_t quality = 75;
  std::variant<bool, Duration> hold = false;
  std::optional<double> step_hz{};

  View& with_backend(ViewBackend value) & {
    backend = std::move(value);
    return *this;
  }
  View with_backend(ViewBackend value) && {
    backend = std::move(value);
    return std::move(*this);
  }
  View& with_port(uint16_t value) & {
    port = std::move(value);
    return *this;
  }
  View with_port(uint16_t value) && {
    port = std::move(value);
    return std::move(*this);
  }
  View& with_fps(uint64_t value) & {
    fps = std::move(value);
    return *this;
  }
  View with_fps(uint64_t value) && {
    fps = std::move(value);
    return std::move(*this);
  }
  View& with_source(std::optional<std::string> value) & {
    source = std::move(value);
    return *this;
  }
  View with_source(std::optional<std::string> value) && {
    source = std::move(value);
    return std::move(*this);
  }
  View& with_format(ViewFormat value) & {
    format = std::move(value);
    return *this;
  }
  View with_format(ViewFormat value) && {
    format = std::move(value);
    return std::move(*this);
  }
  View& with_quality(uint8_t value) & {
    quality = std::move(value);
    return *this;
  }
  View with_quality(uint8_t value) && {
    quality = std::move(value);
    return std::move(*this);
  }
  View& with_hold(std::variant<bool, Duration> value) & {
    hold = std::move(value);
    return *this;
  }
  View with_hold(std::variant<bool, Duration> value) && {
    hold = std::move(value);
    return std::move(*this);
  }
  View& with_step_hz(std::optional<double> value) & {
    step_hz = std::move(value);
    return *this;
  }
  View with_step_hz(std::optional<double> value) && {
    step_hz = std::move(value);
    return std::move(*this);
  }
  RLMeshView raw() const {
    RLMeshView c = RLMESH_VIEW_INIT;
    c.backend = static_cast<RLMeshViewBackend>(backend);
    c.port = port;
    c.fps = fps;
    c.source = detail::ptr(source);
    c.format = static_cast<RLMeshViewFormat>(format);
    c.quality = quality;
    if (auto b = std::get_if<bool>(&hold))
      c.hold = *b ? RLMESH_VIEW_HOLD_UNTIL_QUIT : RLMESH_VIEW_HOLD_OFF;
    else {
      c.hold = RLMESH_VIEW_HOLD_SECONDS;
      c.hold_seconds = std::get<Duration>(hold).count();
    }
    c.has_step_hz = step_hz.has_value();
    if (step_hz) c.step_hz = *step_hz;
    return c;
  }
  void validate() const {
    auto c = raw();
    detail::check_status(rlmesh_view_validate(&c));
  }
  static View parse(std::string_view text) {
    RLMeshView c = RLMESH_VIEW_INIT;
    detail::check_status(rlmesh_view_parse(std::string(text).c_str(), &c));
    View v;
    v.backend = static_cast<ViewBackend>(c.backend);
    v.port = c.port;
    v.fps = c.fps;
    if (c.source) v.source = c.source;
    v.format = static_cast<ViewFormat>(c.format);
    v.quality = c.quality;
    if (c.hold == RLMESH_VIEW_HOLD_SECONDS)
      v.hold = Duration(c.hold_seconds);
    else
      v.hold = c.hold == RLMESH_VIEW_HOLD_UNTIL_QUIT;
    if (c.has_step_hz) v.step_hz = c.step_hz;
    return v;
  }

  friend bool operator==(const View& self, const View& other) noexcept(
      detail::nothrow_equal<decltype(backend), decltype(port), decltype(fps), decltype(source),
                            decltype(format), decltype(quality), decltype(hold),
                            decltype(step_hz)>) {
    return self.backend == other.backend && self.port == other.port && self.fps == other.fps &&
           self.source == other.source && self.format == other.format &&
           self.quality == other.quality && self.hold == other.hold &&
           self.step_hz == other.step_hz;
  }
  friend bool operator!=(const View& self, const View& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
inline std::string to_string(const View& v) {
  auto c = v.raw();
  return detail::text(rlmesh_view_repr, &c);
}
struct RunOptions {
  std::optional<uint64_t> episodes{};
  OptionalList<int64_t> seeds{};
  std::optional<uint64_t> max_episode_steps{};
  std::optional<Duration> max_episode{};
  RunHooks* hooks = nullptr;
  std::optional<std::string> instruction{};
  bool close_env = false;
  std::optional<bool> trust_entrypoints{};
  // Unset: the model's own (its load()'s, else its spec's timing), else 1.
  std::optional<uint32_t> execution_horizon{};
  uint64_t prefetch_lead = 0;
  std::optional<View> view{};
  std::optional<std::string> workflow_edition{};
  uint64_t trial_index_base = 0;
  std::optional<ext::CancelToken> cancel{};
  RunOptions& with_episodes(std::optional<uint64_t> value) & {
    episodes = std::move(value);
    return *this;
  }
  RunOptions with_episodes(std::optional<uint64_t> value) && {
    episodes = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_seeds(OptionalList<int64_t> value) & {
    seeds = std::move(value);
    return *this;
  }
  RunOptions with_seeds(OptionalList<int64_t> value) && {
    seeds = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_max_episode_steps(std::optional<uint64_t> value) & {
    max_episode_steps = std::move(value);
    return *this;
  }
  RunOptions with_max_episode_steps(std::optional<uint64_t> value) && {
    max_episode_steps = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_max_episode(std::optional<Duration> value) & {
    max_episode = std::move(value);
    return *this;
  }
  RunOptions with_max_episode(std::optional<Duration> value) && {
    max_episode = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_hooks(RunHooks* value) & {
    hooks = std::move(value);
    return *this;
  }
  RunOptions with_hooks(RunHooks* value) && {
    hooks = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_instruction(std::optional<std::string> value) & {
    instruction = std::move(value);
    return *this;
  }
  RunOptions with_instruction(std::optional<std::string> value) && {
    instruction = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_close_env(bool value) & {
    close_env = std::move(value);
    return *this;
  }
  RunOptions with_close_env(bool value) && {
    close_env = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_trust_entrypoints(std::optional<bool> value) & {
    trust_entrypoints = std::move(value);
    return *this;
  }
  RunOptions with_trust_entrypoints(std::optional<bool> value) && {
    trust_entrypoints = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_execution_horizon(std::optional<uint32_t> value) & {
    execution_horizon = std::move(value);
    return *this;
  }
  RunOptions with_execution_horizon(std::optional<uint32_t> value) && {
    execution_horizon = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_prefetch_lead(uint64_t value) & {
    prefetch_lead = std::move(value);
    return *this;
  }
  RunOptions with_prefetch_lead(uint64_t value) && {
    prefetch_lead = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_view(std::optional<View> value) & {
    view = std::move(value);
    return *this;
  }
  RunOptions with_view(std::optional<View> value) && {
    view = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_workflow_edition(std::optional<std::string> value) & {
    workflow_edition = std::move(value);
    return *this;
  }
  RunOptions with_workflow_edition(std::optional<std::string> value) && {
    workflow_edition = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_trial_index_base(uint64_t value) & {
    trial_index_base = std::move(value);
    return *this;
  }
  RunOptions with_trial_index_base(uint64_t value) && {
    trial_index_base = std::move(value);
    return std::move(*this);
  }
  RunOptions& with_cancel(std::optional<ext::CancelToken> value) & {
    cancel = std::move(value);
    return *this;
  }
  RunOptions with_cancel(std::optional<ext::CancelToken> value) && {
    cancel = std::move(value);
    return std::move(*this);
  }
};
struct BindOptions {
  std::optional<View> view{};
  BindOptions& with_view(std::optional<View> value) & {
    view = std::move(value);
    return *this;
  }
  BindOptions with_view(std::optional<View> value) && {
    view = std::move(value);
    return std::move(*this);
  }
  bool close_env = false;
  // Unset: the model's own (its load()'s, else its spec's timing), else 1.
  std::optional<uint32_t> execution_horizon{};
  std::optional<std::string> workflow_edition{};
  // Borrowed for the binding's life: the binding is their run context, an
  // episode reset over before it ended counts as truncated, and on_run_end
  // fires at close.
  RunHooks* hooks = nullptr;

  BindOptions& with_close_env(bool value) & {
    close_env = value;
    return *this;
  }
  BindOptions with_close_env(bool value) && {
    close_env = value;
    return std::move(*this);
  }
  BindOptions& with_execution_horizon(std::optional<uint32_t> value) & {
    execution_horizon = value;
    return *this;
  }
  BindOptions with_execution_horizon(std::optional<uint32_t> value) && {
    execution_horizon = value;
    return std::move(*this);
  }
  BindOptions& with_workflow_edition(std::optional<std::string> value) & {
    workflow_edition = std::move(value);
    return *this;
  }
  BindOptions with_workflow_edition(std::optional<std::string> value) && {
    workflow_edition = std::move(value);
    return std::move(*this);
  }
  BindOptions& with_hooks(RunHooks* value) & {
    hooks = value;
    return *this;
  }
  BindOptions with_hooks(RunHooks* value) && {
    hooks = value;
    return std::move(*this);
  }
};
namespace detail {
template <class C, class O>
void budgets(C& c, const O& o, int64_t& empty) {
  c.has_episodes = o.episodes.has_value();
  if (o.episodes) c.episodes = *o.episodes;
  if (o.seeds) {
    c.seeds = o.seeds->empty() ? &empty : o.seeds->data();
    c.seeds_len = o.seeds->size();
  }
  c.has_max_episode_steps = o.max_episode_steps.has_value();
  if (o.max_episode_steps) c.max_episode_steps = *o.max_episode_steps;
  c.has_max_episode_seconds = o.max_episode.has_value();
  if (o.max_episode) c.max_episode_seconds = o.max_episode->count();
  c.trial_index_base = o.trial_index_base;
  if (o.cancel) c.cancel = o.cancel->raw();
}
template <class C, class O>
void run_options(C& c, const O& o) {
  c.instruction = ptr(o.instruction);
  c.close_env = o.close_env;
  c.has_trust_entrypoints = o.trust_entrypoints.has_value();
  if (o.trust_entrypoints) c.trust_entrypoints = *o.trust_entrypoints;
  c.has_execution_horizon = o.execution_horizon.has_value();
  c.execution_horizon = o.execution_horizon.value_or(0);
  c.workflow_edition = ptr(o.workflow_edition);
  if (o.cancel) c.cancel = o.cancel->raw();
}
}  // namespace detail
/// A model bound to one env and stepped by hand, as Python's Model.bind.
class Binding {
  // Release the C binding before the targets and hooks whose pointers it borrows.
  std::shared_ptr<void> model_keepalive_{}, env_keepalive_{}, hooks_keepalive_{};
  detail::Owned<RLMeshBinding, rlmesh_binding_free> handle_;

 public:
  explicit Binding(RLMeshBinding* p) : handle_(p, rlmesh_binding_free) {}
  Binding(RLMeshBinding* p, const ModelTarget& model, const EnvTarget& env) noexcept
      : model_keepalive_(model.keepalive_),
        env_keepalive_(env.keepalive_),
        handle_(p, rlmesh_binding_free) {}
  Binding(RLMeshBinding* p, const ModelTarget& model, const EnvTarget& env,
          std::shared_ptr<void> hooks) noexcept
      : model_keepalive_(model.keepalive_),
        env_keepalive_(env.keepalive_),
        hooks_keepalive_(std::move(hooks)),
        handle_(p, rlmesh_binding_free) {}
  Binding(Binding&&) noexcept = default;
  Binding& operator=(Binding&& other) noexcept {
    if (this != &other) {
      Binding incoming(std::move(other));
      model_keepalive_.swap(incoming.model_keepalive_);
      env_keepalive_.swap(incoming.env_keepalive_);
      hooks_keepalive_.swap(incoming.hooks_keepalive_);
      handle_.swap(incoming.handle_);
    }
    return *this;
  }
  RLMeshBinding* raw() const { return handle_.get(); }
  ResetResult reset(std::optional<int64_t> seed = {}, const std::optional<InfoDict>& options = {}) {
    std::optional<Value> v;
    if (options) v = Value(*options);
    RLMeshValue* o = nullptr;
    RLMeshValue* i = nullptr;
    detail::check_status(
        rlmesh_binding_reset(raw(), detail::ptr(seed), v ? v->raw() : nullptr, &o, &i));
    auto [obs, info] = detail::adopt_values(o, i);
    return {std::move(obs), info.as_dict()};
  }
  /// Steps under action, or the model's when it is unset.
  StepResult step(const std::optional<Value>& action = {}) {
    RLMeshValue* o = nullptr;
    RLMeshValue* i = nullptr;
    double r = 0;
    bool t = false, tr = false;
    detail::check_status(
        rlmesh_binding_step(raw(), action ? action->raw() : nullptr, &o, &r, &t, &tr, &i));
    auto [obs, info] = detail::adopt_values(o, i);
    return {std::move(obs), r, t, tr, info.as_dict()};
  }
  bool done() const {
    bool b = false;
    detail::check_status(rlmesh_binding_done(raw(), &b));
    return b;
  }
  specs::EnvContract contract() {
    RLMeshEnvContract* p = nullptr;
    detail::check_status(rlmesh_binding_contract(raw(), &p));
    return specs::EnvContract(p);
  }
  uint64_t num_envs() {
    uint64_t n = 0;
    detail::check_status(rlmesh_binding_num_envs(raw(), &n));
    return n;
  }
  std::string render_mode() { return detail::text(rlmesh_binding_render_mode, raw()); }
  Value render() {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_binding_render(raw(), &p));
    return Value::adopt(p);
  }
  FrameSources frame_sources() {
    RLMeshFrameSources* p = nullptr;
    detail::check_status(rlmesh_binding_frame_sources(raw(), &p));
    return FrameSources::take(p);
  }
  Value read(const Value& observation, const ReadItem& item) {
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_binding_read(raw(), observation.raw(), item.raw(), &p));
    return Value::adopt(p);
  }
  void close() { detail::check_status(rlmesh_binding_close(raw())); }
};
inline RunResult run(ModelTarget model, EnvTarget env, const RunOptions& options = {}) {
  RLMeshRunOptions c = RLMESH_RUN_OPTIONS_INIT;
  int64_t empty = 0;
  detail::budgets(c, options, empty);
  detail::run_options(c, options);
  c.prefetch_lead = options.prefetch_lead;
  auto hooks = detail::hooks(options.hooks);
  c.hooks = hooks.get();
  std::optional<RLMeshView> view{};
  if (options.view) {
    view = options.view->raw();
    c.view = &*view;
  }
  RLMeshRunResult* p = nullptr;
  detail::check_status(rlmesh_run(model.raw(), env.raw(), &c, &p));
  return RunResult::take(p);
}
inline RunResult Model::run(EnvTarget env_or_address, const RunOptions& options) {
  return rlmesh::run(*this, std::move(env_or_address), options);
}
inline RunResult Model::run(EnvTarget env_or_address) { return run(std::move(env_or_address), {}); }
/// Bind model (a Model or RANDOM_SAMPLE) to target; see Model::bind.
inline Binding bind(ModelTarget model, EnvTarget target, const BindOptions& options = {}) {
  RLMeshBindOptions c = RLMESH_BIND_OPTIONS_INIT;
  std::optional<RLMeshView> view{};
  if (options.view) {
    view = options.view->raw();
    c.view = &*view;
  }
  c.close_env = options.close_env;
  c.has_execution_horizon = options.execution_horizon.has_value();
  c.execution_horizon = options.execution_horizon.value_or(0);
  c.workflow_edition = detail::ptr(options.workflow_edition);
  std::shared_ptr<RLMeshRunHooks> hooks(detail::hooks(options.hooks).release(),
                                        rlmesh_run_hooks_free);
  c.hooks = hooks.get();
  RLMeshBinding* p = nullptr;
  detail::check_status(rlmesh_bind(model.raw(), target.raw(), &c, &p));
  return Binding(p, model, target, std::move(hooks));
}
inline Binding Model::bind(EnvTarget target, const BindOptions& options) {
  return rlmesh::bind(*this, std::move(target), options);
}
inline Binding Model::bind(EnvTarget target) { return bind(std::move(target), {}); }
namespace recorder {
inline constexpr std::string_view SCHEMA = RLMESH_RECORDER_SCHEMA;
}
struct MediaRef {
  std::string camera{}, kind{}, path{}, format{};
  std::optional<int64_t> frame_count{}, width{}, height{};
  std::optional<double> fps{};

  RLMeshMediaRef* raw() const {
    RLMeshMediaRefOptions c = RLMESH_MEDIA_REF_OPTIONS_INIT;
    c.has_frame_count = frame_count.has_value();
    if (frame_count) c.frame_count = *frame_count;
    c.has_width = width.has_value();
    if (width) c.width = *width;
    c.has_height = height.has_value();
    if (height) c.height = *height;
    c.has_fps = fps.has_value();
    if (fps) c.fps = *fps;
    RLMeshMediaRef* p = nullptr;
    detail::check_status(
        rlmesh_media_ref_new(camera.c_str(), kind.c_str(), path.c_str(), format.c_str(), &c, &p));
    return p;
  }
  static MediaRef from_raw(const RLMeshMediaRef* p) {
    MediaRef v{detail::text(rlmesh_media_ref_camera, p),
               detail::text(rlmesh_media_ref_kind, p),
               detail::text(rlmesh_media_ref_path, p),
               detail::text(rlmesh_media_ref_format, p),
               {},
               {},
               {},
               {}};
    int64_t n = 0;
    double f = 0;
    if (rlmesh_media_ref_frame_count(p, &n)) v.frame_count = n;
    if (rlmesh_media_ref_width(p, &n)) v.width = n;
    if (rlmesh_media_ref_height(p, &n)) v.height = n;
    if (rlmesh_media_ref_fps(p, &f)) v.fps = f;
    return v;
  }
  Metadata to_dict() const {
    auto p = detail::own<RLMeshMediaRef, rlmesh_media_ref_free>(raw());
    RLMeshValue* v = nullptr;
    detail::check_status(rlmesh_media_ref_to_dict(p.get(), &v));
    return Value::adopt(v).as_dict();
  }

  friend bool operator==(const MediaRef& self, const MediaRef& other) noexcept(
      detail::nothrow_equal<decltype(camera), decltype(kind), decltype(path), decltype(format),
                            decltype(frame_count), decltype(width), decltype(height),
                            decltype(fps)>) {
    return self.camera == other.camera && self.kind == other.kind && self.path == other.path &&
           self.format == other.format && self.frame_count == other.frame_count &&
           self.width == other.width && self.height == other.height && self.fps == other.fps;
  }
  friend bool operator!=(const MediaRef& self,
                         const MediaRef& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct EpisodeRecord {
  int64_t index = 0;
  std::optional<int64_t> seed{};
  int64_t steps = 0;
  double reward = 0;
  bool terminated = false, truncated = false;
  std::optional<bool> success{};
  double duration_s = 0;
  bool included_in_metrics = true;
  std::vector<MediaRef> media{};

  RLMeshEpisodeRecord* raw() const {
    RLMeshEpisodeRecordOptions c = RLMESH_EPISODE_RECORD_OPTIONS_INIT;
    c.has_success = success.has_value();
    if (success) c.success = *success;
    c.duration_s = duration_s;
    c.included_in_metrics = included_in_metrics;
    std::vector<detail::Owned<RLMeshMediaRef, rlmesh_media_ref_free>> owned;
    std::vector<const RLMeshMediaRef*> ptrs;
    for (auto& m : media) {
      owned.push_back(detail::own<RLMeshMediaRef, rlmesh_media_ref_free>(m.raw()));
      ptrs.push_back(owned.back().get());
    }
    c.media = ptrs.data();
    c.media_len = ptrs.size();
    RLMeshEpisodeRecord* p = nullptr;
    detail::check_status(rlmesh_episode_record_new(index, detail::ptr(seed), steps, reward,
                                                   terminated, truncated, &c, &p));
    return p;
  }
  static EpisodeRecord from_raw(const RLMeshEpisodeRecord* p) {
    EpisodeRecord v;
    v.index = rlmesh_episode_record_index(p);
    int64_t n = 0;
    if (rlmesh_episode_record_seed(p, &n)) v.seed = n;
    v.steps = rlmesh_episode_record_steps(p);
    v.reward = rlmesh_episode_record_reward(p);
    v.terminated = rlmesh_episode_record_terminated(p);
    v.truncated = rlmesh_episode_record_truncated(p);
    bool b = false;
    if (rlmesh_episode_record_success(p, &b)) v.success = b;
    v.duration_s = rlmesh_episode_record_duration_s(p);
    v.included_in_metrics = rlmesh_episode_record_included_in_metrics(p);
    for (size_t i = 0; i < rlmesh_episode_record_media_len(p); ++i) {
      RLMeshMediaRef* m = nullptr;
      detail::check_status(rlmesh_episode_record_media_get(p, i, &m));
      auto owned = detail::own<RLMeshMediaRef, rlmesh_media_ref_free>(m);
      v.media.push_back(MediaRef::from_raw(m));
    }
    return v;
  }
  Metadata to_dict() const {
    auto p = detail::own<RLMeshEpisodeRecord, rlmesh_episode_record_free>(raw());
    RLMeshValue* v = nullptr;
    detail::check_status(rlmesh_episode_record_to_dict(p.get(), &v));
    return Value::adopt(v).as_dict();
  }

  friend bool operator==(const EpisodeRecord& self, const EpisodeRecord& other) noexcept(
      detail::nothrow_equal<decltype(index), decltype(seed), decltype(steps), decltype(reward),
                            decltype(terminated), decltype(truncated), decltype(success),
                            decltype(duration_s), decltype(included_in_metrics), decltype(media)>) {
    return self.index == other.index && self.seed == other.seed && self.steps == other.steps &&
           self.reward == other.reward && self.terminated == other.terminated &&
           self.truncated == other.truncated && self.success == other.success &&
           self.duration_s == other.duration_s &&
           self.included_in_metrics == other.included_in_metrics && self.media == other.media;
  }
  friend bool operator!=(const EpisodeRecord& self,
                         const EpisodeRecord& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct WorkloadRecord {
  std::string model{}, env{}, task{};
  Metadata config{};
  std::vector<EpisodeRecord> episodes{};

  RLMeshWorkloadRecord* raw() const {
    RLMeshWorkloadRecordOptions c = RLMESH_WORKLOAD_RECORD_OPTIONS_INIT;
    Value cfg(config);
    c.config = cfg.raw();
    std::vector<detail::Owned<RLMeshEpisodeRecord, rlmesh_episode_record_free>> owned;
    std::vector<const RLMeshEpisodeRecord*> ptrs;
    for (auto& e : episodes) {
      owned.push_back(detail::own<RLMeshEpisodeRecord, rlmesh_episode_record_free>(e.raw()));
      ptrs.push_back(owned.back().get());
    }
    c.episodes = ptrs.data();
    c.episodes_len = ptrs.size();
    RLMeshWorkloadRecord* p = nullptr;
    detail::check_status(
        rlmesh_workload_record_new(model.c_str(), env.c_str(), task.c_str(), &c, &p));
    return p;
  }
  static WorkloadRecord from_raw(const RLMeshWorkloadRecord* p) {
    WorkloadRecord v;
    v.model = detail::text(rlmesh_workload_record_model, p);
    v.env = detail::text(rlmesh_workload_record_env, p);
    v.task = detail::text(rlmesh_workload_record_task, p);
    RLMeshValue* cfg = nullptr;
    detail::check_status(rlmesh_workload_record_config(p, &cfg));
    v.config = Value::adopt(cfg).as_dict();
    for (size_t i = 0; i < rlmesh_workload_record_episodes_len(p); ++i) {
      RLMeshEpisodeRecord* e = nullptr;
      detail::check_status(rlmesh_workload_record_episodes_get(p, i, &e));
      auto owned = detail::own<RLMeshEpisodeRecord, rlmesh_episode_record_free>(e);
      v.episodes.push_back(EpisodeRecord::from_raw(e));
    }
    return v;
  }
  Metadata to_dict() const {
    auto p = detail::own<RLMeshWorkloadRecord, rlmesh_workload_record_free>(raw());
    RLMeshValue* v = nullptr;
    detail::check_status(rlmesh_workload_record_to_dict(p.get(), &v));
    return Value::adopt(v).as_dict();
  }
  double mean_reward() const {
    auto p = detail::own<RLMeshWorkloadRecord, rlmesh_workload_record_free>(raw());
    return rlmesh_workload_record_mean_reward(p.get());
  }
  std::optional<double> success_rate() const {
    auto p = detail::own<RLMeshWorkloadRecord, rlmesh_workload_record_free>(raw());
    double v = 0;
    if (rlmesh_workload_record_success_rate(p.get(), &v)) return v;
    return {};
  }
  int64_t total_steps() const {
    auto p = detail::own<RLMeshWorkloadRecord, rlmesh_workload_record_free>(raw());
    return rlmesh_workload_record_total_steps(p.get());
  }

  friend bool operator==(const WorkloadRecord& self, const WorkloadRecord& other) noexcept(
      detail::nothrow_equal<decltype(model), decltype(env), decltype(task), decltype(config),
                            decltype(episodes)>) {
    return self.model == other.model && self.env == other.env && self.task == other.task &&
           self.config == other.config && self.episodes == other.episodes;
  }
  friend bool operator!=(const WorkloadRecord& self,
                         const WorkloadRecord& other) noexcept(noexcept(self == other)) {
    return !(self == other);
  }
};
struct RecorderOptions {
  std::optional<std::string> result_set_id{};
  uint64_t fps = 30;
  uint8_t quality = 60;
  RecorderOptions& with_result_set_id(std::optional<std::string> value) & {
    result_set_id = std::move(value);
    return *this;
  }
  RecorderOptions with_result_set_id(std::optional<std::string> value) && {
    result_set_id = std::move(value);
    return std::move(*this);
  }
  RecorderOptions& with_fps(uint64_t value) & {
    fps = std::move(value);
    return *this;
  }
  RecorderOptions with_fps(uint64_t value) && {
    fps = std::move(value);
    return std::move(*this);
  }
  RecorderOptions& with_quality(uint8_t value) & {
    quality = std::move(value);
    return *this;
  }
  RecorderOptions with_quality(uint8_t value) && {
    quality = std::move(value);
    return std::move(*this);
  }
};
struct RecorderAddOptions {
  std::optional<std::string> task{};
  std::optional<Metadata> config{};
  bool included_in_metrics = true;
  RecorderAddOptions& with_task(std::optional<std::string> value) & {
    task = std::move(value);
    return *this;
  }
  RecorderAddOptions with_task(std::optional<std::string> value) && {
    task = std::move(value);
    return std::move(*this);
  }
  RecorderAddOptions& with_config(std::optional<Metadata> value) & {
    config = std::move(value);
    return *this;
  }
  RecorderAddOptions with_config(std::optional<Metadata> value) && {
    config = std::move(value);
    return std::move(*this);
  }
  RecorderAddOptions& with_included_in_metrics(bool value) & {
    included_in_metrics = std::move(value);
    return *this;
  }
  RecorderAddOptions with_included_in_metrics(bool value) && {
    included_in_metrics = std::move(value);
    return std::move(*this);
  }
};
struct RecorderCaptureOptions {
  std::optional<std::string> task{};
  std::optional<Metadata> config{};
  OptionalList<std::string> cameras{};
  std::shared_ptr<RunContext> binding{};
  std::vector<std::string> video_info_keys = {"video_artifact_path", "video_url"};
  bool included_in_metrics = true;
  RecorderCaptureOptions& with_task(std::optional<std::string> value) & {
    task = std::move(value);
    return *this;
  }
  RecorderCaptureOptions with_task(std::optional<std::string> value) && {
    task = std::move(value);
    return std::move(*this);
  }
  RecorderCaptureOptions& with_config(std::optional<Metadata> value) & {
    config = std::move(value);
    return *this;
  }
  RecorderCaptureOptions with_config(std::optional<Metadata> value) && {
    config = std::move(value);
    return std::move(*this);
  }
  RecorderCaptureOptions& with_cameras(OptionalList<std::string> value) & {
    cameras = std::move(value);
    return *this;
  }
  RecorderCaptureOptions with_cameras(OptionalList<std::string> value) && {
    cameras = std::move(value);
    return std::move(*this);
  }
  RecorderCaptureOptions& with_binding(std::shared_ptr<RunContext> value) & {
    binding = std::move(value);
    return *this;
  }
  RecorderCaptureOptions with_binding(std::shared_ptr<RunContext> value) && {
    binding = std::move(value);
    return std::move(*this);
  }
  RecorderCaptureOptions& with_video_info_keys(std::vector<std::string> value) & {
    video_info_keys = std::move(value);
    return *this;
  }
  RecorderCaptureOptions with_video_info_keys(std::vector<std::string> value) && {
    video_info_keys = std::move(value);
    return std::move(*this);
  }
  RecorderCaptureOptions& with_included_in_metrics(bool value) & {
    included_in_metrics = std::move(value);
    return *this;
  }
  RecorderCaptureOptions with_included_in_metrics(bool value) && {
    included_in_metrics = std::move(value);
    return std::move(*this);
  }
};
struct RecorderToDictOptions {
  std::optional<std::string> recorded_at{};
  RecorderToDictOptions& with_recorded_at(std::optional<std::string> value) & {
    recorded_at = std::move(value);
    return *this;
  }
  RecorderToDictOptions with_recorded_at(std::optional<std::string> value) && {
    recorded_at = std::move(value);
    return std::move(*this);
  }
};
struct RecorderExportOptions {
  std::optional<bool> archive{};
  std::optional<std::string> recorded_at{};
  RecorderExportOptions& with_archive(std::optional<bool> value) & {
    archive = std::move(value);
    return *this;
  }
  RecorderExportOptions with_archive(std::optional<bool> value) && {
    archive = std::move(value);
    return std::move(*this);
  }
  RecorderExportOptions& with_recorded_at(std::optional<std::string> value) & {
    recorded_at = std::move(value);
    return *this;
  }
  RecorderExportOptions with_recorded_at(std::optional<std::string> value) && {
    recorded_at = std::move(value);
    return std::move(*this);
  }
};
namespace detail {
class NativeRunHooks final : public RunHooks {
  Owned<RLMeshRunHooks, rlmesh_run_hooks_free> handle_;

 public:
  explicit NativeRunHooks(RLMeshRunHooks* p) : handle_(p, rlmesh_run_hooks_free) {}
  void on_run_start(std::shared_ptr<RunContext> c) override;
  void on_episode_start(int64_t e, std::optional<int64_t> s) override {
    check_status(rlmesh_run_hooks_on_episode_start(handle_.get(), e, ptr(s)));
  }
  void on_step(const StepEvent& e) override;
  void on_episode_end(const EpisodeResult& e) override {
    auto c = e.raw();
    check_status(rlmesh_run_hooks_on_episode_end(handle_.get(), &c));
  }
  void on_run_end(const RunResult& r) override {
    auto p = own<RLMeshRunResult, rlmesh_run_result_free>(r.raw());
    check_status(rlmesh_run_hooks_on_run_end(handle_.get(), p.get()));
  }
};
inline Owned<RLMeshRunContext, rlmesh_run_context_free> context(std::shared_ptr<RunContext> ctx) {
  auto state = std::make_unique<std::shared_ptr<RunContext>>(std::move(ctx));
  RLMeshRunContextVtable c = RLMESH_RUN_CONTEXT_VTABLE_INIT;
  c.contract = [](void* u, RLMeshEnvContract** o) noexcept {
    return callback([&] {
      auto v = (*static_cast<std::shared_ptr<RunContext>*>(u))->contract();
      if (v) *o = rlmesh_env_contract_clone(v->raw());
    });
  };
  c.num_envs = [](void* u, uint64_t* o) noexcept {
    return callback([&] { *o = (*static_cast<std::shared_ptr<RunContext>*>(u))->num_envs(); });
  };
  // ReadItem is immutable; retain it by borrowing for this synchronous call.
  c.read = [](void* u, const RLMeshValue* v, const RLMeshReadItem* i, RLMeshValue** o) noexcept {
    return callback([&] {
      ReadItem item(const_cast<RLMeshReadItem*>(i), false);
      *o = (*static_cast<std::shared_ptr<RunContext>*>(u))
               ->read(Value::copy_of(v), item)
               .clone_raw();
    });
  };
  c.frame_sources = [](void* u, RLMeshFrameSources** o) noexcept {
    return callback(
        [&] { *o = (*static_cast<std::shared_ptr<RunContext>*>(u))->frame_sources().raw(); });
  };
  c.free_user_data = delete_state<std::shared_ptr<RunContext>>;
  RLMeshRunContext* p = nullptr;
  check_status(rlmesh_run_context_new(&c, state.get(), &p));
  state.release();
  return own<RLMeshRunContext, rlmesh_run_context_free>(p);
}
inline void NativeRunHooks::on_run_start(std::shared_ptr<RunContext> c) {
  auto p = context(std::move(c));
  check_status(rlmesh_run_hooks_on_run_start(handle_.get(), p.get()));
}
inline void NativeRunHooks::on_step(const StepEvent& e) {
  class EventContext final : public RunContext {
    ReadFn read_;

   public:
    explicit EventContext(ReadFn r) : read_(std::move(r)) {}
    std::optional<specs::EnvContract> contract() const override { return {}; }
    uint64_t num_envs() const override { return 1; }
    Value read(const Value&, const ReadItem& i) const override { return read_(i); }
    FrameSources frame_sources() const override { return {}; }
  };
  auto ctx = context(std::make_shared<EventContext>(e.read));
  Value info_value(e.info);
  RLMeshStepEvent c = RLMESH_STEP_EVENT_INIT;
  c.episode = e.episode;
  c.has_seed = e.seed.has_value();
  if (e.seed) c.seed = *e.seed;
  c.step = e.step;
  c.observation = e.observation.raw();
  c.action = e.action.raw();
  c.reward = e.reward;
  c.terminated = e.terminated;
  c.truncated = e.truncated;
  c.info = info_value.raw();
  c.predict_ms = e.predict_ms;
  c.step_ms = e.step_ms;
  c.context = ctx.get();
  check_status(rlmesh_run_hooks_on_step(handle_.get(), &c));
}
}  // namespace detail
class Recorder {
  detail::Owned<RLMeshRecorder, rlmesh_recorder_free> handle_{nullptr, rlmesh_recorder_free};

 public:
  explicit Recorder(const RecorderOptions& options = {}) {
    RLMeshRecorderOptions c = RLMESH_RECORDER_OPTIONS_INIT;
    c.result_set_id = detail::ptr(options.result_set_id);
    c.fps = options.fps;
    c.quality = options.quality;
    RLMeshRecorder* p = nullptr;
    detail::check_status(rlmesh_recorder_new(&c, &p));
    handle_.reset(p);
  }
  Recorder(Recorder&&) = default;
  Recorder& operator=(Recorder&&) = default;
  WorkloadRecord add(const RunResult& result, std::string_view model, std::string_view env,
                     const RecorderAddOptions& options = {}) {
    auto r = detail::own<RLMeshRunResult, rlmesh_run_result_free>(result.raw());
    Value cfg(options.config.value_or(Metadata{}));
    RLMeshRecorderAddOptions c = RLMESH_RECORDER_ADD_OPTIONS_INIT;
    c.task = detail::ptr(options.task);
    c.config = options.config ? cfg.raw() : nullptr;
    c.included_in_metrics = options.included_in_metrics;
    RLMeshWorkloadRecord* p = nullptr;
    detail::check_status(rlmesh_recorder_add(handle_.get(), r.get(), std::string(model).c_str(),
                                             std::string(env).c_str(), &c, &p));
    auto owned = detail::own<RLMeshWorkloadRecord, rlmesh_workload_record_free>(p);
    return WorkloadRecord::from_raw(p);
  }
  std::unique_ptr<RunHooks> capture(std::string_view model, std::string_view env,
                                    const RecorderCaptureOptions& options = {}) {
    Value cfg(options.config.value_or(Metadata{}));
    RLMeshRecorderCaptureOptions c = RLMESH_RECORDER_CAPTURE_OPTIONS_INIT;
    c.task = detail::ptr(options.task);
    c.config = options.config ? cfg.raw() : nullptr;
    c.included_in_metrics = options.included_in_metrics;
    std::vector<const char*> cameras, keys;
    const char* empty = nullptr;
    if (options.cameras) {
      for (auto& s : *options.cameras) cameras.push_back(s.c_str());
      c.cameras = cameras.empty() ? &empty : cameras.data();
      c.cameras_len = cameras.size();
    }
    for (auto& s : options.video_info_keys) keys.push_back(s.c_str());
    c.video_info_keys = keys.empty() ? &empty : keys.data();
    c.video_info_keys_len = keys.size();
    auto ctx = detail::own<RLMeshRunContext, rlmesh_run_context_free>(nullptr);
    if (options.binding) {
      ctx = detail::context(options.binding);
      c.binding = ctx.get();
    }
    RLMeshRunHooks* p = nullptr;
    detail::check_status(rlmesh_recorder_capture(handle_.get(), std::string(model).c_str(),
                                                 std::string(env).c_str(), &c, &p));
    return std::make_unique<detail::NativeRunHooks>(p);
  }
  Metadata to_dict(const RecorderToDictOptions& options = {}) const {
    RLMeshRecorderToDictOptions c = RLMESH_RECORDER_TO_DICT_OPTIONS_INIT;
    c.recorded_at = detail::ptr(options.recorded_at);
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_recorder_to_dict(handle_.get(), &c, &p));
    return Value::adopt(p).as_dict();
  }
  std::filesystem::path export_(const std::filesystem::path& path,
                                const RecorderExportOptions& options = {}) const {
    RLMeshRecorderExportOptions c = RLMESH_RECORDER_EXPORT_OPTIONS_INIT;
    c.recorded_at = detail::ptr(options.recorded_at);
    c.has_archive = options.archive.has_value();
    if (options.archive) c.archive = *options.archive;
    return detail::text(rlmesh_recorder_export, handle_.get(), path.string().c_str(), &c);
  }
  std::string result_set_id() const {
    return detail::text(rlmesh_recorder_result_set_id, handle_.get());
  }
  std::vector<WorkloadRecord> workloads() const {
    std::vector<WorkloadRecord> v;
    for (size_t i = 0; i < rlmesh_recorder_workloads_len(handle_.get()); ++i) {
      RLMeshWorkloadRecord* p = nullptr;
      detail::check_status(rlmesh_recorder_workloads_get(handle_.get(), i, &p));
      auto owned = detail::own<RLMeshWorkloadRecord, rlmesh_workload_record_free>(p);
      v.push_back(WorkloadRecord::from_raw(p));
    }
    return v;
  }
  void close() { detail::check_status(rlmesh_recorder_close(handle_.get())); }
};
inline std::string to_string(const MediaRef& v) {
  auto p = detail::own<RLMeshMediaRef, rlmesh_media_ref_free>(v.raw());
  return detail::text(rlmesh_media_ref_repr, p.get());
}
inline std::string to_string(const EpisodeRecord& v) {
  auto p = detail::own<RLMeshEpisodeRecord, rlmesh_episode_record_free>(v.raw());
  return detail::text(rlmesh_episode_record_repr, p.get());
}
inline std::string to_string(const WorkloadRecord& v) {
  auto p = detail::own<RLMeshWorkloadRecord, rlmesh_workload_record_free>(v.raw());
  return detail::text(rlmesh_workload_record_repr, p.get());
}
enum class DescribeKind { Env = 1, Model = 2 };
struct DescribeOptions {
  std::optional<DescribeKind> kind{};
  std::optional<std::string> generated_at{};
  RLMeshDescribeOptions raw() const {
    RLMeshDescribeOptions c = RLMESH_DESCRIBE_OPTIONS_INIT;
    c.has_kind = kind.has_value();
    if (kind) c.kind = static_cast<RLMeshDescribeKind>(*kind);
    c.generated_at = detail::ptr(generated_at);
    return c;
  }
  DescribeOptions& with_kind(std::optional<DescribeKind> value) & {
    kind = std::move(value);
    return *this;
  }
  DescribeOptions with_kind(std::optional<DescribeKind> value) && {
    kind = std::move(value);
    return std::move(*this);
  }
  DescribeOptions& with_generated_at(std::optional<std::string> value) & {
    generated_at = std::move(value);
    return *this;
  }
  DescribeOptions with_generated_at(std::optional<std::string> value) && {
    generated_at = std::move(value);
    return std::move(*this);
  }
};
struct DescribeJsonOptions {
  std::optional<DescribeKind> kind{};
  std::optional<std::string> generated_at{};
  std::optional<ServedEnv> served_env{};
  std::optional<std::string> workflow_edition{};
  RLMeshDescribeJsonOptions raw() const {
    RLMeshDescribeJsonOptions c = RLMESH_DESCRIBE_JSON_OPTIONS_INIT;
    c.has_kind = kind.has_value();
    if (kind) c.kind = static_cast<RLMeshDescribeKind>(*kind);
    c.generated_at = detail::ptr(generated_at);
    c.workflow_edition = detail::ptr(workflow_edition);
    if (served_env) {
      c.served_env = served_env->lanes().data();
      c.served_env_len = served_env->lanes().size();
      c.served_vector_env = served_env->vector();
    }
    return c;
  }
  DescribeJsonOptions& with_kind(std::optional<DescribeKind> value) & {
    kind = std::move(value);
    return *this;
  }
  DescribeJsonOptions with_kind(std::optional<DescribeKind> value) && {
    kind = std::move(value);
    return std::move(*this);
  }
  DescribeJsonOptions& with_generated_at(std::optional<std::string> value) & {
    generated_at = std::move(value);
    return *this;
  }
  DescribeJsonOptions with_generated_at(std::optional<std::string> value) && {
    generated_at = std::move(value);
    return std::move(*this);
  }
  DescribeJsonOptions& with_served_env(std::optional<ServedEnv> value) & {
    served_env = std::move(value);
    return *this;
  }
  DescribeJsonOptions with_served_env(std::optional<ServedEnv> value) && {
    served_env = std::move(value);
    return std::move(*this);
  }
  DescribeJsonOptions& with_workflow_edition(std::optional<std::string> value) & {
    workflow_edition = std::move(value);
    return *this;
  }
  DescribeJsonOptions with_workflow_edition(std::optional<std::string> value) && {
    workflow_edition = std::move(value);
    return std::move(*this);
  }
};
inline Metadata describe(Model& obj, const DescribeOptions& options = {}) {
  auto c = options.raw();
  RLMeshValue* p = nullptr;
  detail::check_status(rlmesh_model_describe(obj.raw(), &c, &p));
  return Value::adopt(p).as_dict();
}
inline Metadata describe(EnvFactoryBase& obj, const DescribeOptions& options = {}) {
  auto c = options.raw();
  RLMeshValue* p = nullptr;
  detail::check_status(rlmesh_env_factory_describe(obj.handle(), &c, &p));
  return Value::adopt(p).as_dict();
}
inline std::string describe_json(Model& obj, const DescribeJsonOptions& options = {}) {
  auto c = options.raw();
  return detail::text(rlmesh_model_describe_json, obj.raw(), &c);
}
inline std::string describe_json(EnvFactoryBase& obj, const DescribeJsonOptions& options = {}) {
  auto c = options.raw();
  return detail::text(rlmesh_env_factory_describe_json, obj.handle(), &c);
}
template <class T>
Metadata describe(const DescribeOptions& options = {}) {
  if constexpr (std::is_base_of_v<EnvFactoryBase, T> &&
                std::is_convertible_v<const T*, const EnvFactoryBase*>) {
    T f;
    return describe(f, options);
  } else {
    detail::PolicyTable<T> vt;
    auto c = options.raw();
    RLMeshValue* p = nullptr;
    detail::check_status(rlmesh_model_vtable_describe(&vt.table, &c, &p));
    return Value::adopt(p).as_dict();
  }
}
template <class T>
std::string describe_json(const DescribeJsonOptions& options = {}) {
  if constexpr (std::is_base_of_v<EnvFactoryBase, T> &&
                std::is_convertible_v<const T*, const EnvFactoryBase*>) {
    T f;
    return describe_json(f, options);
  } else {
    detail::PolicyTable<T> vt;
    auto c = options.raw();
    return detail::text(rlmesh_model_vtable_describe_json, &vt.table, &c);
  }
}
template <class P, std::enable_if_t<Policy<P> && (!std::is_same_v<P, Model>), int> = 0>
Metadata describe([[maybe_unused]] P& obj, const DescribeOptions& options = {}) {
  detail::PolicyTable<P> vt;
  auto c = options.raw();
  RLMeshValue* p = nullptr;
  detail::check_status(rlmesh_model_vtable_describe(&vt.table, &c, &p));
  return Value::adopt(p).as_dict();
}
template <class P, std::enable_if_t<Policy<P> && (!std::is_same_v<P, Model>), int> = 0>
std::string describe_json([[maybe_unused]] P& obj, const DescribeJsonOptions& options = {}) {
  detail::PolicyTable<P> vt;
  auto c = options.raw();
  return detail::text(rlmesh_model_vtable_describe_json, &vt.table, &c);
}
inline std::string to_string(NoAdapter) { return "NO_ADAPTER"; }
inline std::ostream& operator<<(std::ostream& s, NoAdapter) { return s << "NO_ADAPTER"; }
inline std::string to_string(const specs::EnvContract& v) { return v.to_string(); }
inline std::ostream& operator<<(std::ostream& s, const specs::EnvContract& v) {
  return s << v.to_string();
}
inline std::string to_string(const StepEvent& e) {
  Value info(e.info);
  RLMeshStepEvent c = RLMESH_STEP_EVENT_INIT;
  c.episode = e.episode;
  c.has_seed = e.seed.has_value();
  if (e.seed) c.seed = *e.seed;
  c.step = e.step;
  c.observation = e.observation.raw();
  c.action = e.action.raw();
  c.reward = e.reward;
  c.terminated = e.terminated;
  c.truncated = e.truncated;
  c.info = info.raw();
  c.predict_ms = e.predict_ms;
  c.step_ms = e.step_ms;
  return detail::text(rlmesh_step_event_repr, &c);
}
inline std::ostream& operator<<(std::ostream& s, const StepEvent& e) { return s << to_string(e); }
template <class F>
Metadata EnvFactory<F>::describe() {
  return rlmesh::describe<F>();
}
inline std::string_view to_string(ViewBackend v) {
  switch (v) {
    case ViewBackend::Terminal:
      return "terminal";
    case ViewBackend::Http:
      return "http";
    case ViewBackend::Both:
      return "both";
  }
  return {};
}
inline std::string_view to_string(ViewFormat v) {
  switch (v) {
    case ViewFormat::Jpeg:
      return "jpeg";
    case ViewFormat::Png:
      return "png";
  }
  return {};
}

namespace ext {
inline ResetResult reset(RemoteEnv& handle, const CancelToken& cancel,
                         std::optional<int64_t> seed = {},
                         const std::optional<InfoDict>& options = {}) {
  std::optional<Value> v;
  if (options) v = Value(*options);
  RLMeshValue* o = nullptr;
  RLMeshValue* info = nullptr;
  detail::check_status(rlmesh_ext_remote_env_reset(
      handle.raw(), detail::ptr(seed), v ? v->raw() : nullptr, cancel.raw(), &o, &info));
  auto [observation, i] = detail::adopt_values(o, info);
  return {std::move(observation), i.as_dict()};
}
inline StepResult step(RemoteEnv& handle, const CancelToken& cancel, const Value& action) {
  RLMeshValue* o = nullptr;
  RLMeshValue* info = nullptr;
  double r = 0;
  bool t = false, tr = false;
  detail::check_status(
      rlmesh_ext_remote_env_step(handle.raw(), action.raw(), cancel.raw(), &o, &r, &t, &tr, &info));
  auto [observation, i] = detail::adopt_values(o, info);
  return {std::move(observation), r, t, tr, i.as_dict()};
}
inline ResetResult reset(RemoteVectorEnv& handle, const CancelToken& cancel,
                         std::optional<VectorSeed> seed = {},
                         const std::optional<InfoDict>& options = {}) {
  std::optional<Value> v;
  if (options) v = Value(*options);
  RLMeshValue* o = nullptr;
  RLMeshValue* info = nullptr;
  std::vector<int64_t> seeds;
  if (seed) {
    if (auto p = std::get_if<int64_t>(&*seed))
      seeds.push_back(*p);
    else
      seeds = std::get<std::vector<int64_t>>(*seed);
  }
  int64_t empty = 0;
  detail::check_status(rlmesh_ext_remote_vector_env_reset(
      handle.raw(), seed ? (seeds.empty() ? &empty : seeds.data()) : nullptr, seeds.size(),
      v ? v->raw() : nullptr, cancel.raw(), &o, &info));
  auto [observation, i] = detail::adopt_values(o, info);
  return {std::move(observation), i.as_dict()};
}
inline VectorStepResult step(RemoteVectorEnv& handle, const CancelToken& cancel,
                             const Value& actions) {
  const auto n = static_cast<size_t>(handle.num_envs());
  std::vector<double> rewards(n);
  auto t = std::make_unique<bool[]>(n);
  auto tr = std::make_unique<bool[]>(n);
  RLMeshValue* o = nullptr;
  RLMeshValue* info = nullptr;
  size_t len = 0;
  detail::check_status(rlmesh_ext_remote_vector_env_step(handle.raw(), actions.raw(), cancel.raw(),
                                                         &o, rewards.data(), t.get(), tr.get(), n,
                                                         &len, &info));
  auto [observation, i] = detail::adopt_values(o, info);
  return {std::move(observation), std::move(rewards), std::vector<bool>(t.get(), t.get() + len),
          std::vector<bool>(tr.get(), tr.get() + len), i.as_dict()};
}
}  // namespace ext
namespace ext {
inline std::string env_server_describe_json(const EnvServer& server) {
  return detail::text(rlmesh_ext_env_server_describe_json, server.raw());
}
inline bool unregister_entrypoint(std::string_view name,
                                  const adapters::ObsTransform* expected = nullptr) {
  if (name.find('\0') != std::string_view::npos)
    throw ValueError("entrypoint contains an embedded NUL");
  bool removed = false;
  detail::check_status(rlmesh_ext_unregister_entrypoint(
      std::string(name).c_str(), expected ? expected->raw() : nullptr, &removed));
  return removed;
}
inline void register_entrypoint(std::string_view entrypoint,
                                const adapters::ObsTransform& transform) {
  if (entrypoint.find('\0') != std::string_view::npos)
    throw ValueError("entrypoint contains an embedded NUL");
  detail::check_status(
      rlmesh_ext_register_entrypoint(std::string(entrypoint).c_str(), transform.raw()));
}
}  // namespace ext
inline std::ostream& operator<<(std::ostream& s, const EpisodeResult& v) {
  return s << to_string(v);
}
inline std::ostream& operator<<(std::ostream& s, const TelemetryRow& v) {
  return s << to_string(v);
}
inline std::ostream& operator<<(std::ostream& s, const RunResult& v) { return s << to_string(v); }
inline std::string to_string(const FrameSources& v) {
  auto p = detail::own<RLMeshFrameSources, rlmesh_frame_sources_free>(v.raw());
  return detail::text(rlmesh_frame_sources_repr, p.get());
}
inline std::ostream& operator<<(std::ostream& s, const FrameSources& v) {
  return s << to_string(v);
}
inline std::ostream& operator<<(std::ostream& s, const View& v) { return s << to_string(v); }
inline std::ostream& operator<<(std::ostream& s, const MediaRef& v) { return s << to_string(v); }
inline std::ostream& operator<<(std::ostream& s, const EpisodeRecord& v) {
  return s << to_string(v);
}
inline std::ostream& operator<<(std::ostream& s, const WorkloadRecord& v) {
  return s << to_string(v);
}
inline std::ostream& operator<<(std::ostream& s, const EnvServer& v) { return s << v.to_string(); }
namespace adapters {
inline std::string to_string(const Image&);
inline std::string to_string(const State&);
inline std::string to_string(const Constant&);
inline std::string to_string(const Text&);
inline std::string to_string(const Custom&);
inline std::string to_string(const Actuator&);
inline std::string to_string(const ImageTag&);
inline std::string to_string(const StateTag&);
inline std::string to_string(const TextTag&);
inline std::string to_string(const Field&);
inline std::string to_string(const Rotation&);
inline std::string to_string(const CustomEncoding&);
inline std::string to_string(const Concat&);
inline std::string to_string(const Action&);
inline std::string to_string(const Split&);
inline std::string to_string(const ObservationRoles&);
inline std::string to_string(const InputNode&);
inline std::string to_string(const ObsNode&);
namespace detail_repr {
inline std::string quote(std::string_view text) {
  char q = text.find('\'') != text.npos && text.find('"') == text.npos ? '"' : '\'';
  std::string out(1, q);
  constexpr char hex[] = "0123456789abcdef";
  for (char character : text) {
    const auto c = static_cast<unsigned char>(character);
    if (c == static_cast<unsigned char>(q) || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c == '\n')
      out += "\\n";
    else if (c == '\r')
      out += "\\r";
    else if (c == '\t')
      out += "\\t";
    else if (c < 32 || c == 127) {
      out += "\\x";
      out += hex[c >> 4];
      out += hex[c & 15];
    } else
      out += static_cast<char>(c);
  }
  out += q;
  return out;
}
inline std::string repr(const std::string& s) { return quote(s); }
inline std::string repr(bool v) { return v ? "True" : "False"; }
template <class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
std::string repr(T v) {
  return std::to_string(v);
}
inline std::string repr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  char buffer[128];
  auto [end, ec] = std::to_chars(buffer, buffer + sizeof(buffer), v, std::chars_format::general);
  if (ec != std::errc{}) throw ValueError("cannot format float");
  std::string out(buffer, end);
  if (out.find_first_of(".e") == out.npos) out += ".0";
  return out;
}
template <class E, std::enable_if_t<std::is_enum_v<E>, int> = 0>
std::string repr(E v) {
  return quote(to_string(v));
}
template <class E>
std::string repr(const Tolerant<E>& v) {
  return quote(v.to_string());
}
template <class T, std::enable_if_t<!std::is_enum_v<T>, int> = 0,
          class = decltype(to_string(std::declval<const T&>()))>
std::string repr(const T& v) {
  return to_string(v);
}
template <class T>
std::string repr(const std::optional<T>&);
template <class T>
std::string repr(const std::vector<T>&);
template <class... T>
std::string repr(const std::variant<T...>&);
template <class A, class B>
std::string repr(const std::pair<A, B>&);
template <class T>
std::string repr(const OptionalList<T>&);
inline std::string repr(const ObsTransform& v) {
  std::ostringstream s;
  s << "ObsTransform(" << v.raw() << ")";
  return s.str();
}
inline std::string repr(const RotationTransform& v) {
  std::ostringstream s;
  s << "RotationTransform(" << v.raw() << ")";
  return s.str();
}
template <class T>
std::string repr(const std::optional<T>& v) {
  return v ? repr(*v) : "None";
}
template <class T>
std::string repr(const std::vector<T>& v) {
  std::string out = "(";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out += ", ";
    out += repr(v[i]);
  }
  if (v.size() == 1) out += ",";
  return out + ")";
}
template <class... T>
std::string repr(const std::variant<T...>& v) {
  return std::visit([](const auto& x) { return repr(x); }, v);
}
template <class A, class B>
std::string repr(const std::pair<A, B>& v) {
  return "(" + repr(v.first) + ", " + repr(v.second) + ")";
}
template <class T>
std::string repr(const OptionalList<T>& v) {
  return v ? repr(*v) : "None";
}
template <class T>
std::string accept(const OptionalList<T>& v) {
  return v && v->size() == 1 ? repr((*v)[0]) : repr(v);
}
inline std::string state_encoding(const State& s) {
  if (!s.encoding) return "None";
  if (auto p = std::get_if<std::vector<Tolerant<RotationEncoding>>>(&*s.encoding);
      p && p->size() == 1)
    return repr((*p)[0]);
  return repr(*s.encoding);
}
inline std::string dataclass(
    std::string_view name, std::initializer_list<std::pair<std::string_view, std::string>> fields) {
  std::string out(name);
  out += '(';
  bool first = true;
  for (auto& [key, value] : fields) {
    if (!first) out += ", ";
    first = false;
    out += key;
    out += '=';
    out += value;
  }
  return out + ")";
}
}  // namespace detail_repr
inline std::string to_string(const Image& v) {
  return detail_repr::dataclass("Image", {{"role", detail_repr::repr(v.role)},
                                          {"height", detail_repr::repr(v.height)},
                                          {"width", detail_repr::repr(v.width)},
                                          {"layout", detail_repr::repr(v.layout)},
                                          {"channels", detail_repr::repr(v.channels)},
                                          {"dtype", detail_repr::repr(v.dtype)},
                                          {"normalize", detail_repr::repr(v.normalize)},
                                          {"lead_dims", detail_repr::repr(v.lead_dims)},
                                          {"upside_down", detail_repr::repr(v.upside_down)},
                                          {"resample", detail_repr::repr(v.resample)},
                                          {"allow_upscale", detail_repr::repr(v.allow_upscale)},
                                          {"fit", detail_repr::accept(v.fit)},
                                          {"optional", detail_repr::repr(v.optional)},
                                          {"fill", detail_repr::repr(v.fill)},
                                          {"stack", detail_repr::repr(v.stack)},
                                          {"crop", detail_repr::repr(v.crop)},
                                          {"crop_area", detail_repr::repr(v.crop_area)},
                                          {"crop_mode", detail_repr::repr(v.crop_mode)},
                                          {"jpeg_quality", detail_repr::repr(v.jpeg_quality)},
                                          {"channel_order", detail_repr::repr(v.channel_order)},
                                          {"render", detail_repr::repr(v.render)},
                                          {"offsets", detail_repr::repr(v.offsets)},
                                          {"stack_pad", detail_repr::repr(v.stack_pad)},
                                          {"part", detail_repr::repr(v.part)}});
}
inline std::ostream& operator<<(std::ostream& s, const Image& v) { return s << to_string(v); }
inline std::string to_string(const State& v) {
  return detail_repr::dataclass("State", {{"role", detail_repr::repr(v.role)},
                                          {"encoding", detail_repr::state_encoding(v)},
                                          {"dim", detail_repr::repr(v.dim)},
                                          {"index", detail_repr::repr(v.index)},
                                          {"optional", detail_repr::repr(v.optional)},
                                          {"range", detail_repr::repr(v.range)},
                                          {"pad_to", detail_repr::repr(v.pad_to)},
                                          {"dtype", detail_repr::repr(v.dtype)},
                                          {"reshape", detail_repr::repr(v.reshape)},
                                          {"container", detail_repr::repr(v.container)},
                                          {"fill", detail_repr::repr(v.fill)},
                                          {"post_rotate", detail_repr::repr(v.post_rotate)},
                                          {"scale", detail_repr::repr(v.scale)},
                                          {"offset", detail_repr::repr(v.offset)},
                                          {"frame", detail_repr::repr(v.frame)},
                                          {"provenance", detail_repr::accept(v.provenance)},
                                          {"part", detail_repr::repr(v.part)},
                                          {"labels", detail_repr::repr(v.labels)},
                                          {"source", detail_repr::repr(v.source)},
                                          {"clip", detail_repr::repr(v.clip)}});
}
inline std::ostream& operator<<(std::ostream& s, const State& v) { return s << to_string(v); }
inline std::string to_string(const Constant& v) {
  return detail_repr::dataclass(
      "Constant", {{"dim", detail_repr::repr(v.dim)}, {"fill", detail_repr::repr(v.fill)}});
}
inline std::ostream& operator<<(std::ostream& s, const Constant& v) { return s << to_string(v); }
inline std::string to_string(const Text& v) {
  return detail_repr::dataclass("Text", {{"role", detail_repr::repr(v.role)},
                                         {"container", detail_repr::repr(v.container)},
                                         {"fill", detail_repr::repr(v.fill)}});
}
inline std::ostream& operator<<(std::ostream& s, const Text& v) { return s << to_string(v); }
inline std::string to_string(const Custom& v) {
  return detail_repr::dataclass("Custom", {{"transform", detail_repr::repr(v.transform)},
                                           {"entrypoint", detail_repr::repr(v.entrypoint)}});
}
inline std::ostream& operator<<(std::ostream& s, const Custom& v) { return s << to_string(v); }
inline std::string to_string(const Actuator& v) {
  return detail_repr::dataclass("Actuator", {{"role", detail_repr::repr(v.role)},
                                             {"dim", detail_repr::repr(v.dim)},
                                             {"encoding", detail_repr::repr(v.encoding)},
                                             {"range", detail_repr::repr(v.range)},
                                             {"binary", detail_repr::repr(v.binary)},
                                             {"scale", detail_repr::repr(v.scale)},
                                             {"invert", detail_repr::repr(v.invert)},
                                             {"threshold", detail_repr::repr(v.threshold)},
                                             {"clip", detail_repr::repr(v.clip)},
                                             {"fill", detail_repr::repr(v.fill)},
                                             {"optional", detail_repr::repr(v.optional)},
                                             {"frame", detail_repr::repr(v.frame)},
                                             {"reference", detail_repr::repr(v.reference)},
                                             {"part", detail_repr::repr(v.part)},
                                             {"offset", detail_repr::repr(v.offset)},
                                             {"labels", detail_repr::repr(v.labels)}});
}
inline std::ostream& operator<<(std::ostream& s, const Actuator& v) { return s << to_string(v); }
inline std::string to_string(const ImageTag& v) {
  return detail_repr::dataclass("ImageTag", {{"role", detail_repr::repr(v.role)},
                                             {"layout", detail_repr::repr(v.layout)},
                                             {"upside_down", detail_repr::repr(v.upside_down)},
                                             {"part", detail_repr::repr(v.part)}});
}
inline std::ostream& operator<<(std::ostream& s, const ImageTag& v) { return s << to_string(v); }
inline std::string to_string(const StateTag& v) {
  return detail_repr::dataclass("StateTag", {{"role", detail_repr::repr(v.role)},
                                             {"encoding", detail_repr::accept(v.encoding)},
                                             {"range", detail_repr::repr(v.range)},
                                             {"frame", detail_repr::repr(v.frame)},
                                             {"provenance", detail_repr::repr(v.provenance)},
                                             {"part", detail_repr::repr(v.part)},
                                             {"labels", detail_repr::repr(v.labels)}});
}
inline std::ostream& operator<<(std::ostream& s, const StateTag& v) { return s << to_string(v); }
inline std::string to_string(const TextTag& v) {
  return detail_repr::dataclass("TextTag", {{"role", detail_repr::repr(v.role)}});
}
inline std::ostream& operator<<(std::ostream& s, const TextTag& v) { return s << to_string(v); }
inline std::string to_string(const Field& v) {
  return detail_repr::dataclass("Field", {{"role", detail_repr::repr(v.role)},
                                          {"dim", detail_repr::repr(v.dim)},
                                          {"encoding", detail_repr::accept(v.encoding)},
                                          {"range", detail_repr::repr(v.range)},
                                          {"frame", detail_repr::repr(v.frame)},
                                          {"provenance", detail_repr::repr(v.provenance)},
                                          {"part", detail_repr::repr(v.part)},
                                          {"labels", detail_repr::repr(v.labels)}});
}
inline std::ostream& operator<<(std::ostream& s, const Field& v) { return s << to_string(v); }
inline std::string to_string(const Rotation& v) {
  return detail_repr::dataclass("Rotation", {{"encoding", detail_repr::repr(v.encoding)},
                                             {"value", detail_repr::repr(v.value)}});
}
inline std::ostream& operator<<(std::ostream& s, const Rotation& v) { return s << to_string(v); }
inline std::string to_string(const CustomEncoding& v) {
  return detail_repr::dataclass("CustomEncoding", {{"base", detail_repr::repr(v.base)},
                                                   {"from_base", detail_repr::repr(v.from_base)},
                                                   {"to_base", detail_repr::repr(v.to_base)},
                                                   {"name", detail_repr::repr(v.name)}});
}
inline std::ostream& operator<<(std::ostream& s, const CustomEncoding& v) {
  return s << to_string(v);
}
inline std::string to_string(const Concat& v) {
  return detail_repr::dataclass("Concat", {{"parts", detail_repr::repr(v.parts)},
                                           {"pad_to", detail_repr::repr(v.pad_to)},
                                           {"dtype", detail_repr::repr(v.dtype)},
                                           {"reshape", detail_repr::repr(v.reshape)},
                                           {"container", detail_repr::repr(v.container)},
                                           {"clip", detail_repr::repr(v.clip)}});
}
inline std::ostream& operator<<(std::ostream& s, const Concat& v) { return s << to_string(v); }
inline std::string to_string(const Action& v) {
  return detail_repr::dataclass("Action", {{"components", detail_repr::repr(v.components)},
                                           {"clip", detail_repr::repr(v.clip)}});
}
inline std::ostream& operator<<(std::ostream& s, const Action& v) { return s << to_string(v); }
inline std::string to_string(const Split& v) {
  return detail_repr::dataclass("Split", {{"fields", detail_repr::repr(v.fields)}});
}
inline std::ostream& operator<<(std::ostream& s, const Split& v) { return s << to_string(v); }
inline std::string to_string(const ObservationRoles& v) {
  return detail_repr::dataclass("ObservationRoles", {{"images", detail_repr::repr(v.images)},
                                                     {"states", detail_repr::repr(v.states)},
                                                     {"texts", detail_repr::repr(v.texts)}});
}
inline std::ostream& operator<<(std::ostream& s, const ObservationRoles& v) {
  return s << to_string(v);
}
inline std::string to_string(const InputNode& n) {
  if (auto leaf = std::get_if<ModelLeaf>(&n.value_)) return detail_repr::repr(*leaf);
  if (auto tuple = std::get_if<std::vector<InputNode>>(&n.value_)) return detail_repr::repr(*tuple);
  std::string out = "{";
  bool first = true;
  for (const auto& [k, v] : std::get<std::vector<std::pair<std::string, InputNode>>>(n.value_)) {
    if (!first) out += ", ";
    first = false;
    out += detail_repr::quote(k) + ": " + to_string(v);
  }
  return out + "}";
}
inline std::ostream& operator<<(std::ostream& s, const InputNode& v) { return s << to_string(v); }
inline std::string to_string(const ObsNode& n) {
  if (auto leaf = std::get_if<ObsLeaf>(&n.value_)) return detail_repr::repr(*leaf);
  if (auto tuple = std::get_if<std::vector<ObsNode>>(&n.value_)) return detail_repr::repr(*tuple);
  std::string out = "{";
  bool first = true;
  for (const auto& [k, v] : std::get<std::vector<std::pair<std::string, ObsNode>>>(n.value_)) {
    if (!first) out += ", ";
    first = false;
    out += detail_repr::quote(k) + ": " + to_string(v);
  }
  return out + "}";
}
inline std::ostream& operator<<(std::ostream& s, const ObsNode& v) { return s << to_string(v); }
}  // namespace adapters
}  // namespace rlmesh
namespace rlmesh::detail {
template <class... T>
size_t hash_fields(const T&... fields) {
  size_t result = 0;
  ((result ^= std::hash<T>{}(fields) + 0x9e3779b9U + (result << 6) + (result >> 2)), ...);
  return result;
}
}  // namespace rlmesh::detail
namespace std {
template <>
struct hash<rlmesh::adapters::Text> {
  size_t operator()(const rlmesh::adapters::Text& v) const {
    return rlmesh::detail::hash_fields(v.role, v.container, v.fill);
  }
};
template <>
struct hash<rlmesh::adapters::Custom> {
  size_t operator()(const rlmesh::adapters::Custom& v) const {
    return rlmesh::detail::hash_fields(
        v.transform.has_value(),
        v.transform ? rlmesh_obs_transform_hash(v.transform->raw()) : uint64_t{0}, v.entrypoint);
  }
};
template <>
struct hash<rlmesh::adapters::CustomEncoding> {
  size_t operator()(const rlmesh::adapters::CustomEncoding& v) const {
    auto arm = [](const optional<rlmesh::adapters::EncodingArm>& value) -> size_t {
      if (!value) return 0;
      if (auto transform = get_if<rlmesh::adapters::RotationTransform>(&*value))
        return rlmesh::detail::hash_fields(1, rlmesh_rotation_transform_hash(transform->raw()));
      return rlmesh::detail::hash_fields(2, get<string>(*value));
    };
    return rlmesh::detail::hash_fields(v.base, arm(v.from_base), arm(v.to_base), v.name);
  }
};
template <>
struct hash<rlmesh::adapters::ImageTag> {
  size_t operator()(const rlmesh::adapters::ImageTag& v) const {
    return rlmesh::detail::hash_fields(v.role, v.layout, v.upside_down, v.part);
  }
};
template <>
struct hash<rlmesh::adapters::TextTag> {
  size_t operator()(const rlmesh::adapters::TextTag& v) const { return hash<string>{}(v.role); }
};
template <class E>
struct hash<rlmesh::adapters::Tolerant<E>> {
  size_t operator()(const rlmesh::adapters::Tolerant<E>& v) const {
    return hash<string>{}(v.to_string());
  }
};
template <>
struct hash<rlmesh::adapters::Advisory> {
  size_t operator()(const rlmesh::adapters::Advisory& v) const {
    return hash<string>{}(v.message) ^ (static_cast<size_t>(v.severity) << 1);
  }
};
template <>
struct hash<rlmesh::adapters::ObservationRoles> {
  size_t operator()(const rlmesh::adapters::ObservationRoles& v) const {
    size_t h = 0;
    for (const auto* list : {&v.images, &v.states, &v.texts}) {
      h = h * 31 + list->size();
      for (auto& s : *list) h = h * 31 + hash<string>{}(s);
    }
    return h;
  }
};
template <>
struct hash<rlmesh::RenderFn> {
  size_t operator()(const rlmesh::RenderFn& v) const { return hash<const void*>{}(v.identity()); }
};
template <>
struct hash<rlmesh::FrameSources> {
  size_t operator()(const rlmesh::FrameSources& v) const {
    size_t h = hash<const void*>{}(v.render.identity());
    for (auto& s : v.roles) h = h * 31 + hash<string>{}(s);
    if (v.render_label) h = h * 31 + hash<string>{}(*v.render_label);
    return h;
  }
};
}  // namespace std
#pragma pop_macro("DELETE")
#pragma pop_macro("OUT")
#pragma pop_macro("IN")
#pragma pop_macro("ERROR")
#pragma pop_macro("max")
#pragma pop_macro("min")
#pragma pop_macro("Success")
#pragma pop_macro("Status")
#pragma pop_macro("False")
#pragma pop_macro("True")
#pragma pop_macro("Bool")
#pragma pop_macro("None")
#pragma pop_macro("TEXT")
#pragma pop_macro("checkf")
#pragma pop_macro("ensure")
#pragma pop_macro("verify")
#pragma pop_macro("check")
#endif  // RLMESH_HPP
