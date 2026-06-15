// RLMesh C++ wrapper — experimental, header-only, C++17, v1 model path. A thin
// RAII layer over the C ABI (rlmesh.h): no exceptions by default, errors via
// Result<T, Error>; unwrap() aborts (an escape hatch, not the idiom).
#ifndef RLMESH_HPP
#define RLMESH_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "rlmesh.h"

namespace rlmesh {

/// A failure: a status code plus an owned message snapshot.
class Error {
 public:
  Error(RlmeshStatus code, std::string message, bool recoverable)
      : code_(code), message_(std::move(message)), recoverable_(recoverable) {}

  /// Snapshot the thread-local last error after a nonzero status.
  static Error from_last(RlmeshStatus code) {
    const char* message = rlmesh_last_error_message();
    return Error(code, message ? std::string(message) : std::string(),
                 rlmesh_last_error_is_recoverable() != 0);
  }

  RlmeshStatus code() const { return code_; }
  std::string_view message() const { return message_; }
  bool is_recoverable() const { return recoverable_; }

 private:
  RlmeshStatus code_;
  std::string message_;
  bool recoverable_;
};

/// A value or an Error. Check `operator bool` / `error()`; `unwrap()` aborts on
/// error and exists only for assertions (never the default path).
template <class T>
class [[nodiscard]] Result {
 public:
  Result(T value) : data_(std::move(value)) {}
  Result(Error error) : data_(std::move(error)) {}

  explicit operator bool() const { return std::holds_alternative<T>(data_); }
  T& value() { return std::get<T>(data_); }
  const T& value() const { return std::get<T>(data_); }
  Error& error() { return std::get<Error>(data_); }
  const Error& error() const { return std::get<Error>(data_); }

  T unwrap() {
    if (!*this) {
      auto message = error().message();
      std::fprintf(stderr, "rlmesh: unwrap() on error: %.*s\n", static_cast<int>(message.size()),
                   message.data());
      std::abort();
    }
    return std::move(value());
  }

 private:
  std::variant<T, Error> data_;
};

using Status = Result<std::monostate>;
inline Status ok() { return Status(std::monostate{}); }

/// A borrowed, DLPack-shaped tensor view (valid while its source value lives).
class Tensor {
 public:
  explicit Tensor(const RlmeshTensor& raw) : raw_(raw) {}

  const void* data() const { return raw_.data; }
  RlmeshDType dtype() const { return raw_.dtype; }
  int32_t ndim() const { return raw_.ndim; }
  const int64_t* shape() const { return raw_.shape; }
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
  RlmeshTensor raw_;
};

/// An owned value (move-only).
class Value {
 public:
  explicit Value(RlmeshValue* ptr) : ptr_(ptr) {}
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

  const RlmeshValue* get() const { return ptr_; }
  RlmeshValueKind kind() const { return rlmesh_value_kind(ptr_); }

  static Value discrete(int64_t value) { return Value(rlmesh_value_discrete(value)); }

  /// Copy a contiguous buffer into a Box value.
  static Result<Value> box(const void* data, RlmeshDType dtype, std::vector<int64_t> shape) {
    RlmeshTensor tensor{};
    tensor.data = const_cast<void*>(data);
    tensor.ndim = static_cast<int32_t>(shape.size());
    tensor.shape = shape.data();
    tensor.strides = nullptr;
    tensor.dtype = dtype;
    tensor.device_type = RLMESH_DEVICE_CPU;
    RlmeshValue* value = rlmesh_value_box(&tensor);
    if (value == nullptr) return Error::from_last(RLMESH_ERR_INVALID_VALUE);
    return Value(value);
  }

  Result<Tensor> as_tensor() const {
    RlmeshTensor out{};
    RlmeshStatus status = rlmesh_value_as_tensor(ptr_, &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Tensor(out);
  }

 private:
  void reset() {
    if (ptr_ != nullptr) {
      rlmesh_value_free(ptr_);
      ptr_ = nullptr;
    }
  }
  RlmeshValue* ptr_;
};

/// A decoded batch of borrowed value views (frees the whole batch on destruction).
class Batch {
 public:
  Batch(RlmeshValue** values, size_t size) : values_(values), size_(size) {}
  Batch(Batch&& other) noexcept : values_(other.values_), size_(other.size_) {
    other.values_ = nullptr;
    other.size_ = 0;
  }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;
  ~Batch() {
    if (values_ != nullptr) rlmesh_values_free(values_, size_);
  }

  size_t size() const { return size_; }
  /// Borrowed view of element `i` (do not free).
  const RlmeshValue* at(size_t i) const { return values_[i]; }
  Result<Tensor> tensor_at(size_t i) const {
    RlmeshTensor out{};
    RlmeshStatus status = rlmesh_value_as_tensor(values_[i], &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Tensor(out);
  }

 private:
  RlmeshValue** values_;
  size_t size_;
};

/// What a predict callback receives: routing plus the encoded observation, with
/// helpers to decode it against the contract's observation space.
class Observation {
 public:
  explicit Observation(const RlmeshObservation* raw) : raw_(raw) {}

  uint32_t num_envs() const { return raw_->num_envs; }
  const RlmeshContract* contract() const { return raw_->contract; }
  const RlmeshSpaceSpec* action_space() const {
    return raw_->contract ? rlmesh_contract_action_space(raw_->contract) : nullptr;
  }

  /// Decode the observation payload into one value per sub-env.
  Result<Batch> decode() const {
    const RlmeshSpaceSpec* space =
        raw_->contract ? rlmesh_contract_observation_space(raw_->contract) : nullptr;
    if (space == nullptr) {
      return Error(RLMESH_ERR_INVALID_VALUE, "no observation space on contract", false);
    }
    RlmeshValue** values = nullptr;
    size_t count = 0;
    RlmeshStatus status =
        rlmesh_decode_batch(raw_->observation, raw_->observation_len, space, &values, &count);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Batch(values, count);
  }

 private:
  const RlmeshObservation* raw_;
};

/// A model worker: bind a predict policy, then drive it against an environment.
class Model {
 public:
  /// `predict` maps an Observation to one action value per sub-env (one Value
  /// for a single env). Lifecycle hooks are optional.
  using PredictFn = std::function<Result<Value>(const Observation&)>;

  static Result<Model> from_predict(PredictFn predict) {
    auto state = std::make_unique<State>();
    state->predict = std::move(predict);
    RlmeshModelVtable vtable{};
    vtable.struct_size = sizeof(RlmeshModelVtable);
    vtable.predict = &trampoline_predict;
    RlmeshModel* raw = nullptr;
    RlmeshStatus status = rlmesh_model_new(&vtable, state.get(), &raw);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Model(raw, std::move(state));
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
  ~Model() {
    if (model_ != nullptr) rlmesh_model_free(model_);
  }

  /// Drive against a remote environment until it ends. Blocking.
  Status run_local(std::string_view env_address) {
    std::string address(env_address);
    RlmeshStatus status = rlmesh_model_run_local(model_, address.c_str());
    if (status != RLMESH_OK) return Error::from_last(status);
    return ok();
  }

  /// Drive against a remote environment for `max_episodes`. Blocking.
  Status run_local(std::string_view env_address, uint64_t max_episodes) {
    std::string address(env_address);
    RlmeshStatus status =
        rlmesh_model_run_local_for_episodes(model_, address.c_str(), max_episodes);
    if (status != RLMESH_OK) return Error::from_last(status);
    return ok();
  }

 private:
  struct State {
    PredictFn predict;
  };

  Model(RlmeshModel* model, std::unique_ptr<State> state)
      : model_(model), state_(std::move(state)) {}

  static int trampoline_predict(void* user_data, const RlmeshObservation* obs,
                                RlmeshBytes* out_action) noexcept {
    auto* state = static_cast<State*>(user_data);
    Observation observation(obs);
#if defined(__cpp_exceptions)
    try {
#endif
      // The functional predict returns ONE action value; that only round-trips a
      // whole batch for num_envs == 1. Fail loudly rather than desync the wire
      // for a vectorized non-Box action space (spec amendment A5).
      if (observation.num_envs() > 1) {
        rlmesh_callback_set_error(
            "from_predict supports single-env only (num_envs == 1); use the vector path", false);
        return RLMESH_ERR_MODEL;
      }
      Result<Value> action = state->predict(observation);
      if (!action) {
        std::string message(action.error().message());
        rlmesh_callback_set_error(message.c_str(), action.error().is_recoverable());
        return action.error().code();
      }
      const RlmeshSpaceSpec* space = observation.action_space();
      if (space == nullptr) {
        rlmesh_callback_set_error("no action space on contract", false);
        return RLMESH_ERR_INVALID_VALUE;
      }
      const RlmeshValue* value = action.value().get();
      return rlmesh_encode_batch(&value, 1, space, out_action);
#if defined(__cpp_exceptions)
    } catch (const std::exception& error) {
      rlmesh_callback_set_error(error.what(), false);
      return RLMESH_ERR_MODEL;
    } catch (...) {
      rlmesh_callback_set_error("unknown C++ exception in predict", false);
      return RLMESH_ERR_MODEL;
    }
#endif
  }

  RlmeshModel* model_ = nullptr;
  std::unique_ptr<State> state_;
};

}  // namespace rlmesh

#endif  // RLMESH_HPP
