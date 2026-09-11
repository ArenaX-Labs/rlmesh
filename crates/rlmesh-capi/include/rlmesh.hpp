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
  Error(RlmeshStatus code, std::string message, bool recoverable = false)
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

/// A borrowed value view (valid for the duration of the predict call).
class ValueRef {
 public:
  explicit ValueRef(const RlmeshValue* ptr) : ptr_(ptr) {}

  const RlmeshValue* get() const { return ptr_; }
  RlmeshValueKind kind() const { return rlmesh_value_kind(ptr_); }

  Result<Tensor> as_tensor() const {
    RlmeshTensor out{};
    RlmeshStatus status = rlmesh_value_as_tensor(ptr_, &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Tensor(out);
  }
  Result<int64_t> as_discrete() const {
    int64_t out = 0;
    RlmeshStatus status = rlmesh_value_as_discrete(ptr_, &out);
    if (status != RLMESH_OK) return Error::from_last(status);
    return out;
  }
  Result<std::string_view> as_text() const {
    const char* data = nullptr;
    size_t len = 0;
    RlmeshStatus status = rlmesh_value_as_text(ptr_, &data, &len);
    if (status != RLMESH_OK) return Error::from_last(status);
    return std::string_view(data, len);
  }

 private:
  const RlmeshValue* ptr_;
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
  /// Give up ownership (the capi takes it back from a predict callback).
  RlmeshValue* release() {
    RlmeshValue* out = ptr_;
    ptr_ = nullptr;
    return out;
  }
  ValueRef ref() const { return ValueRef(ptr_); }

  static Value discrete(int64_t value) { return Value(rlmesh_value_discrete(value)); }

  static Result<Value> text(std::string_view text) {
    RlmeshValue* value = rlmesh_value_text(text.data(), text.size());
    if (value == nullptr) return Error::from_last(RLMESH_ERR_INVALID_VALUE);
    return Value(value);
  }

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

  /// A zero-filled Box value shaped like `space` (a Box space spec).
  static Result<Value> zeros(const RlmeshSpaceSpec* space) {
    if (space == nullptr || rlmesh_space_type(space) != 1) {
      return Error(RLMESH_ERR_INVALID_VALUE, "zeros() needs a Box space");
    }
    RlmeshDType dtype = rlmesh_space_dtype(space);
    std::vector<int64_t> shape(rlmesh_space_ndim(space));
    if (RlmeshStatus s = rlmesh_space_copy_shape(space, shape.data(), shape.size());
        s != RLMESH_OK) {
      return Error::from_last(s);
    }
    size_t numel = 1;
    for (int64_t dim : shape) numel *= static_cast<size_t>(dim);
    std::vector<uint8_t> bytes(numel * rlmesh_dtype_size(dtype), 0);
    return box(bytes.data(), dtype, std::move(shape));
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

/// What a predict callback receives: routing plus one decoded observation per
/// sub-env (borrowed for the duration of the call).
class Observation {
 public:
  explicit Observation(const RlmeshObservation* raw) : raw_(raw) {}

  size_t num_envs() const { return raw_->num_envs; }
  /// The decoded observation of row `i`; `has_values()` is false when absent.
  bool has_values() const { return raw_->observations != nullptr; }
  ValueRef at(size_t i) const { return ValueRef(raw_->observations[i]); }
  const RlmeshEpisode& episode(size_t i) const { return raw_->episodes[i]; }
  std::string_view episode_id(size_t i) const { return raw_->episodes[i].id; }
  std::string_view env_id() const { return raw_->env_id; }
  std::string_view session_id() const { return raw_->session_id; }

  const RlmeshContract* contract() const { return raw_->contract; }
  const RlmeshSpaceSpec* action_space() const {
    return raw_->contract ? rlmesh_contract_action_space(raw_->contract) : nullptr;
  }
  const RlmeshSpaceSpec* observation_space() const {
    return raw_->contract ? rlmesh_contract_observation_space(raw_->contract) : nullptr;
  }

 private:
  const RlmeshObservation* raw_;
};

/// Options for Model::run_local. Zero-initialized == defaults.
struct RunOptions {
  /// Stop after this many episodes (0 = until the env ends).
  uint64_t max_episodes = 0;
  /// Seed for episode 0 when set; later episodes derive from it.
  bool seeded = false;
  int64_t base_seed = 0;
};

/// A model worker: bind a predict policy, then drive it against an environment
/// or serve it.
class Model {
 public:
  /// Single-env policy: one Observation row → one action Value.
  using PredictFn = std::function<Result<Value>(const Observation&)>;
  /// Batched policy: every row at once → one action per row (`num_envs`).
  using PredictBatchFn = std::function<Result<std::vector<Value>>(const Observation&)>;
  /// Per-episode teardown; `episode_id` empty means every episode of `env_id`.
  using EpisodeEndFn = std::function<void(std::string_view env_id, std::string_view episode_id)>;

  static Result<Model> from_predict(PredictFn predict) {
    return create(
        [predict = std::move(predict)](const Observation& obs) -> Result<std::vector<Value>> {
          if (obs.num_envs() != 1) {
            return Error(RLMESH_ERR_MODEL, "from_predict is single-env; use from_predict_batch");
          }
          auto action = predict(obs);
          if (!action) return action.error();
          std::vector<Value> out;
          out.push_back(std::move(action.value()));
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

  /// Drive against a remote environment. Blocking.
  Status run_local(std::string_view env_address, RunOptions options = {}) {
    std::string address(env_address);
    RlmeshRunOptions raw{};
    raw.max_episodes = options.max_episodes;
    raw.seeded = options.seeded;
    raw.base_seed = options.base_seed;
    RlmeshStatus status = rlmesh_model_run_local(model_, address.c_str(), &raw);
    if (status != RLMESH_OK) return Error::from_last(status);
    return ok();
  }

  /// Serve as a ModelService endpoint. Blocking; `options` nullptr == defaults.
  Status serve(std::string_view bind_address, const RlmeshServeOptions* options = nullptr) {
    std::string address(bind_address);
    RlmeshStatus status = rlmesh_model_serve(model_, address.c_str(), options);
    if (status != RLMESH_OK) return Error::from_last(status);
    return ok();
  }

 private:
  struct State {
    PredictBatchFn predict;
    EpisodeEndFn on_episode_end;
  };

  Model(RlmeshModel* model, std::unique_ptr<State> state)
      : model_(model), state_(std::move(state)) {}

  static Result<Model> create(PredictBatchFn predict) {
    auto state = std::make_unique<State>();
    state->predict = std::move(predict);
    RlmeshModelVtable vtable{};
    vtable.struct_size = sizeof(RlmeshModelVtable);
    vtable.predict = &trampoline_predict;
    vtable.on_episode_end = &trampoline_episode_end;
    RlmeshModel* raw = nullptr;
    RlmeshStatus status = rlmesh_model_new(&vtable, state.get(), &raw);
    if (status != RLMESH_OK) return Error::from_last(status);
    return Model(raw, std::move(state));
  }

  static int fail(const Error& error) {
    std::string message(error.message());
    rlmesh_callback_set_error(message.c_str(), error.is_recoverable());
    return error.code();
  }

  static int trampoline_predict(void* user_data, const RlmeshObservation* obs,
                                RlmeshValue** out_actions) noexcept {
    auto* state = static_cast<State*>(user_data);
    Observation observation(obs);
#if defined(__cpp_exceptions)
    try {
#endif
      Result<std::vector<Value>> actions = state->predict(observation);
      if (!actions) return fail(actions.error());
      if (actions.value().size() != observation.num_envs()) {
        return fail(Error(RLMESH_ERR_MODEL,
                          "predict returned " + std::to_string(actions.value().size()) +
                              " actions for " + std::to_string(observation.num_envs()) + " envs"));
      }
      for (size_t i = 0; i < actions.value().size(); ++i)
        out_actions[i] = actions.value()[i].release();
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
      state->on_episode_end(env_id ? env_id : "", episode_id ? episode_id : "");
#if defined(__cpp_exceptions)
    } catch (...) {
      // A teardown hook has no error channel; swallowing beats unwinding into Rust.
    }
#endif
  }

  RlmeshModel* model_ = nullptr;
  std::unique_ptr<State> state_;
};

}  // namespace rlmesh

#endif  // RLMESH_HPP
