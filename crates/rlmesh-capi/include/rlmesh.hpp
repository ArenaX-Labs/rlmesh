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
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "rlmesh.h"

namespace rlmesh {

/// A failure: a status code plus an owned message snapshot.
class Error {
 public:
  Error(RLMeshStatus code, std::string message, bool recoverable = false)
      : code_(code), message_(std::move(message)), recoverable_(recoverable) {}

  /// Snapshot the thread-local last error after a nonzero status (or a NULL
  /// return from a pointer-returning capi call).
  static Error from_last(RLMeshStatus code) {
    const char* message = rlmesh_last_error_message();
    return Error(code, message ? std::string(message) : std::string(),
                 rlmesh_last_error_is_recoverable() != 0);
  }

  RLMeshStatus code() const { return code_; }
  const std::string& message() const { return message_; }
  bool is_recoverable() const { return recoverable_; }
  /// The run/serve was stopped by `Model::cancel()` — a clean stop, not a fault.
  bool is_cancelled() const { return code_ == RLMESH_ERR_CANCELLED; }

 private:
  RLMeshStatus code_;
  std::string message_;
  bool recoverable_;
};

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

/// The RLMeshDType of a C++ element type (float, double, int32_t, int64_t,
/// uint8_t, bool).
template <class T>
constexpr RLMeshDType dtype_of() {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value ||
                    std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
                    std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value,
                "rlmesh::dtype_of: unsupported element type");
  if constexpr (std::is_same<T, float>::value) {
    return RLMESH_F32;
  } else if constexpr (std::is_same<T, double>::value) {
    return RLMESH_F64;
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return RLMESH_I32;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return RLMESH_I64;
  } else if constexpr (std::is_same<T, bool>::value) {
    return RLMESH_BOOL;
  } else {
    return RLMESH_U8;
  }
}

/// A borrowed, DLPack-shaped tensor view (valid while its source value lives).
class Tensor {
 public:
  explicit Tensor(const RLMeshTensor& raw) : raw_(raw) {}

  const void* data() const { return raw_.data; }
  RLMeshDType dtype() const { return raw_.dtype; }
  int32_t ndim() const { return raw_.ndim; }
  const int64_t* shape() const { return raw_.shape; }
  const int64_t* strides() const { return raw_.strides; }
  bool is_contiguous() const { return raw_.strides == nullptr; }
  size_t numel() const {
    size_t n = 1;
    for (int32_t i = 0; i < raw_.ndim; ++i) n *= static_cast<size_t>(raw_.shape[i]);
    return n;
  }
  /// Flat element pointer for `[0, numel())` — valid only when `is_contiguous()`;
  /// nullptr for a strided view (walk `shape()`/`strides()` instead).
  template <class T>
  const T* as() const {
    return is_contiguous() ? static_cast<const T*>(raw_.data) : nullptr;
  }

 private:
  RLMeshTensor raw_;
};

/// A borrowed value view (valid only while its owner lives — for an observation
/// row, the duration of the predict call). Reads every value kind.
class ValueRef {
 public:
  explicit ValueRef(const RLMeshValue* ptr) : ptr_(ptr) {}

  const RLMeshValue* raw() const { return ptr_; }
  RLMeshValueKind kind() const { return rlmesh_value_kind(ptr_); }

  Result<Tensor> as_tensor() const {
    RLMeshTensor out{};
    if (RLMeshStatus status = rlmesh_value_as_tensor(ptr_, &out); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return Tensor(out);
  }

  Result<int64_t> as_discrete() const {
    int64_t out = 0;
    if (RLMeshStatus status = rlmesh_value_as_discrete(ptr_, &out); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  /// Borrowed UTF-8 bytes (NOT NUL-terminated), valid while the value lives.
  Result<std::string_view> as_text() const {
    const char* data = nullptr;
    size_t len = 0;
    if (RLMeshStatus status = rlmesh_value_as_text(ptr_, &data, &len); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return std::string_view(data, len);
  }

  /// Element count of a MultiBinary / MultiDiscrete value.
  Result<size_t> array_len() const {
    size_t out = 0;
    if (RLMeshStatus status = rlmesh_value_array_len(ptr_, &out); status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  Result<std::vector<uint8_t>> as_multi_binary() const {
    auto len = array_len();
    if (!len) return len.error();
    std::vector<uint8_t> out(*len);
    if (RLMeshStatus status = rlmesh_value_copy_multi_binary(ptr_, out.data(), out.size());
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
  }

  Result<std::vector<int64_t>> as_multi_discrete() const {
    auto len = array_len();
    if (!len) return len.error();
    std::vector<int64_t> out(*len);
    if (RLMeshStatus status = rlmesh_value_copy_multi_discrete(ptr_, out.data(), out.size());
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return out;
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
    const RLMeshValue* child = rlmesh_value_tuple_get(ptr_, index);
    if (child == nullptr) return std::nullopt;
    return ValueRef(child);
  }

  /// Dict key `index`, in sorted key order (borrowed, NOT NUL-terminated).
  Result<std::string_view> key(size_t index) const {
    const char* data = nullptr;
    size_t len = 0;
    if (RLMeshStatus status = rlmesh_value_dict_key(ptr_, index, &data, &len);
        status != RLMESH_OK) {
      return Error::from_last(status);
    }
    return std::string_view(data, len);
  }

  /// Dict child by key; nullopt when this is not a Dict or the key is absent.
  std::optional<ValueRef> get(const std::string& key) const {
    const RLMeshValue* child = rlmesh_value_dict_get(ptr_, key.c_str());
    if (child == nullptr) return std::nullopt;
    return ValueRef(child);
  }

  /// Dict child `index`, in the same sorted key order as `key(index)`; nullopt
  /// when this is not a Dict or `index` is out of range.
  std::optional<ValueRef> at_key(size_t index) const {
    const RLMeshValue* child = rlmesh_value_dict_get_at(ptr_, index);
    if (child == nullptr) return std::nullopt;
    return ValueRef(child);
  }

  /// Every (key, child) of a Dict, in sorted key order — NOT the declaration
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
      if (!child) return Error(RLMESH_ERR_INVALID_VALUE, "dict child missing for a reported key");
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

  static Result<Value> discrete(int64_t value) { return adopt(rlmesh_value_discrete(value)); }

  static Result<Value> text(std::string_view text) {
    return adopt(rlmesh_value_text(text.data(), text.size()));
  }

  /// Copy a contiguous row-major buffer into a Box value.
  static Result<Value> box(const void* data, RLMeshDType dtype, std::vector<int64_t> shape) {
    RLMeshTensor tensor{};
    tensor.data = data;
    tensor.ndim = static_cast<int32_t>(shape.size());
    tensor.shape = shape.data();
    tensor.strides = nullptr;
    tensor.dtype = dtype;
    tensor.device_type = RLMESH_DEVICE_CPU;
    return adopt(rlmesh_value_box(&tensor));
  }

  /// Box from typed elements; the dtype follows T (see `dtype_of`).
  template <class T>
  static Result<Value> box(const std::vector<T>& data, std::vector<int64_t> shape) {
    return box(data.data(), dtype_of<T>(), std::move(shape));
  }

  static Result<Value> multi_binary(const std::vector<uint8_t>& bits) {
    return adopt(rlmesh_value_multi_binary(bits.data(), bits.size()));
  }

  static Result<Value> multi_discrete(const std::vector<int64_t>& values) {
    return adopt(rlmesh_value_multi_discrete(values.data(), values.size()));
  }

  /// Tuple from owned children. On success the capi adopts every child; on
  /// failure it adopts none, so this frees them all before returning.
  static Result<Value> tuple(std::vector<Value> children) {
    std::vector<RLMeshValue*> raw;
    raw.reserve(children.size());
    for (Value& child : children) raw.push_back(child.release());
    RLMeshValue* value = rlmesh_value_tuple(raw.data(), raw.size());
    if (value == nullptr) return disown(raw);
    return Value(value);
  }

  /// Dict from owned (key, value) entries; keys must be unique. Same
  /// all-or-nothing ownership as `tuple`.
  static Result<Value> dict(std::vector<std::pair<std::string, Value>> entries) {
    std::vector<const char*> keys;
    std::vector<RLMeshValue*> raw;
    keys.reserve(entries.size());
    raw.reserve(entries.size());
    for (auto& entry : entries) {
      keys.push_back(entry.first.c_str());
      raw.push_back(entry.second.release());
    }
    RLMeshValue* value = rlmesh_value_dict(keys.data(), raw.data(), raw.size());
    if (value == nullptr) return disown(raw);
    return Value(value);
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
  RLMeshValueKind kind() const { return rlmesh_space_type(ptr_); }

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
      if (!child) return Error(RLMESH_ERR_INVALID_VALUE, "dict space child missing for a key");
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
  if (dtype.lanes != 1) return false;
  switch (dtype.code) {
    case 0:  // int
      switch (dtype.bits) {
        case 8:
          return put(static_cast<int8_t>(scalar));
        case 16:
          return put(static_cast<int16_t>(scalar));
        case 32:
          return put(static_cast<int32_t>(scalar));
        case 64:
          return put(static_cast<int64_t>(scalar));
        default:
          return false;
      }
    case 1:  // uint
      switch (dtype.bits) {
        case 8:
          return put(static_cast<uint8_t>(scalar));
        case 16:
          return put(static_cast<uint16_t>(scalar));
        case 32:
          return put(static_cast<uint32_t>(scalar));
        case 64:
          return put(static_cast<uint64_t>(scalar));
        default:
          return false;
      }
    case 2:  // float
      switch (dtype.bits) {
        case 32:
          return put(static_cast<float>(scalar));
        case 64:
          return put(scalar);
        default:
          return false;
      }
    case 6:  // bool
      return dtype.bits == 8 && put(static_cast<uint8_t>(scalar != 0.0 ? 1 : 0));
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
    case RLMESH_VALUE_BOX: {
      RLMeshDType dtype = space.dtype();
      size_t stride = rlmesh_dtype_size(dtype);
      if (stride == 0) return Error(RLMESH_ERR_INVALID_VALUE, "unsupported Box dtype");
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
          return Error(RLMESH_ERR_INVALID_VALUE, "unsupported Box dtype");
        }
      }
      return Value::box(bytes.data(), dtype, std::move(*shape));
    }
    case RLMESH_VALUE_DISCRETE: {
      auto range = space.discrete();
      if (!range) return range.error();
      return Value::discrete(range->start);
    }
    case RLMESH_VALUE_TEXT: {
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
    case RLMESH_VALUE_MULTI_BINARY: {
      auto count = space.numel();
      if (!count) return count.error();
      return Value::multi_binary(std::vector<uint8_t>(*count, 0));
    }
    case RLMESH_VALUE_MULTI_DISCRETE: {
      auto counts = space.nvec();
      if (!counts) return counts.error();
      return Value::multi_discrete(std::vector<int64_t>(counts->size(), 0));
    }
    case RLMESH_VALUE_TUPLE: {
      auto count = space.size();
      if (!count) return count.error();
      std::vector<Value> children;
      children.reserve(*count);
      for (size_t i = 0; i < *count; ++i) {
        std::optional<SpaceRef> child = space.at(i);
        if (!child) return Error(RLMESH_ERR_INVALID_VALUE, "tuple space child missing");
        auto value = zeros_for(*child);
        if (!value) return value.error();
        children.push_back(std::move(*value));
      }
      return Value::tuple(std::move(children));
    }
    case RLMESH_VALUE_DICT: {
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
    case RLMESH_VALUE_INVALID:
    default:
      return Error(RLMESH_ERR_INVALID_VALUE, "no value fits an unspecified space");
  }
}

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
/// Default-constructed == the C defaults (run until the env ends, no caps).
struct RunOptions {
  /// Stop after this many episodes (0 = until the env ends).
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

  /// Frees the handle. NEVER destroy a Model from inside one of its own
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
    } catch (const std::exception& error) {
      return fail(Error(RLMESH_ERR_MODEL, error.what()));
    } catch (...) {
      return fail(Error(RLMESH_ERR_MODEL, "unknown C++ exception in predict"));
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
    for (RLMeshSpaceSpec* child : children) rlmesh_space_free(child);
    return error;
  }
  void reset() {
    if (ptr_ != nullptr) {
      rlmesh_space_free(ptr_);
      ptr_ = nullptr;
    }
  }

  RLMeshSpaceSpec* ptr_;
};

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
  /// A uint8 image Value ([H, W, 3], [H, W, 4] or [H, W]), or nullopt for no
  /// frame. Declare `render_mode = "rgb_array"` in the EnvConfig to advertise it.
  virtual Result<std::optional<Value>> render() { return std::optional<Value>(); }
  /// The server stopped (once).
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

    RLMeshEnvConfig raw{};
    raw.struct_size = sizeof(RLMeshEnvConfig);
    raw.id = opt(config.id);
    raw.observation_space = config.observation_space->raw();
    raw.action_space = config.action_space->raw();
    raw.adapter_tags_json = opt(config.adapter_tags_json);
    raw.reset_options = options.empty() ? nullptr : options.data();
    raw.num_reset_options = options.size();
    raw.render_mode = opt(config.render_mode);
    raw.metadata_json = opt(config.metadata_json);
    raw.foreground = config.foreground;

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
    } catch (const std::exception& error) {
      return fail(Error(RLMESH_ERR_ENVIRONMENT, error.what()));
    } catch (...) {
      return fail(Error(RLMESH_ERR_ENVIRONMENT, "unknown C++ exception in reset"));
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
    } catch (const std::exception& error) {
      return fail(Error(RLMESH_ERR_ENVIRONMENT, error.what()));
    } catch (...) {
      return fail(Error(RLMESH_ERR_ENVIRONMENT, "unknown C++ exception in step"));
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
    } catch (const std::exception& error) {
      return fail(Error(RLMESH_ERR_ENVIRONMENT, error.what()));
    } catch (...) {
      return fail(Error(RLMESH_ERR_ENVIRONMENT, "unknown C++ exception in render"));
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
