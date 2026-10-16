// Regression coverage for the independent C++17 wrapper reviews.
#include <cassert>
#include <rlmesh.hpp>
#if __cplusplus >= 202002L
#include <ranges>
#include <span>
#endif

using namespace rlmesh;

struct NotAPolicy {};
struct ReviewPolicy {
  Value predict(const Value& v) { return v; }
};
struct PointerConvertiblePolicy : ReviewPolicy {
  operator RLMeshModel*() const { return nullptr; }
};
struct DerivedModel : Model {
  using Model::Model;
  // Even a policy-like derived handle must copy as a handle.
  Value predict(const Value& v) { return v; }
};
static_assert(!std::is_constructible_v<Model, int>);
static_assert(!std::is_constructible_v<Model, NotAPolicy>);
static_assert(!std::is_constructible_v<Model, std::unique_ptr<NotAPolicy>>);
static_assert(!std::is_constructible_v<Model, std::shared_ptr<NotAPolicy>>);
static_assert(!std::is_constructible_v<Model, std::shared_ptr<int>>);
static_assert(!std::is_constructible_v<Env, std::shared_ptr<int>>);
static_assert(!std::is_constructible_v<VectorEnv, std::shared_ptr<int>>);
static_assert(std::is_constructible_v<Model, std::shared_ptr<ReviewPolicy>>);
static_assert(std::is_constructible_v<Model, std::unique_ptr<ReviewPolicy>>);
static_assert(std::is_constructible_v<Model, DerivedModel&>);
static_assert(std::is_constructible_v<Value, int> && std::is_constructible_v<Value, unsigned>);
static_assert(!std::is_constructible_v<Value, char>);
#if __cplusplus >= 202002L
static_assert(!std::is_constructible_v<Value, decltype(u8'a')>);
#endif

struct ReviewScalar {
  spaces::Space observation_space() { return spaces::Discrete(2); }
  spaces::Space action_space() { return observation_space(); }
  ResetResult reset(std::optional<int64_t>, const std::optional<InfoDict>&) { return {0, {}}; }
  StepResult step(std::optional<Value>) { return {0, 0, true, false, {}}; }
};
static_assert(!std::is_constructible_v<VectorEnv, std::shared_ptr<ReviewScalar>>);
struct ReviewVector {
  int count = 1;
  int num_envs() { return count; }
  spaces::Space single_observation_space() { return spaces::Discrete(2); }
  spaces::Space single_action_space() { return single_observation_space(); }
  ResetResult reset(std::optional<VectorSeed>, const std::optional<InfoDict>&) { return {0, {}}; }
  VectorStepResult step(std::optional<Value>) { return {0, {0}, {true}, {false}, {}}; }
};
struct BadCount {
  operator int() const { return 1; }
  explicit operator uint64_t() const = delete;
};
struct BadVector : ReviewVector {
  BadCount num_envs() { return {}; }
};
static_assert(std::is_convertible_v<BadCount, uint64_t>);
static_assert(!detail::BatchedEnv<BadVector>);
static_assert(!std::is_constructible_v<VectorEnv, std::shared_ptr<BadVector>>);

struct Undestroyable {
  ~Undestroyable() = delete;
};
struct BadOptional {
  Undestroyable id();
  Undestroyable render_mode();
  Undestroyable metadata();
  Undestroyable render();
  Undestroyable close();
  Undestroyable prepare();
  static Undestroyable tags();
  static Undestroyable reset_options();
  static Undestroyable workflow_edition();
  static Undestroyable tag_params();
  static Undestroyable spec();
  Undestroyable reset();
  Undestroyable reset(std::string_view);
  Undestroyable load();
  Undestroyable load(const Kwargs&);
  Undestroyable on_episode_end();
  Undestroyable on_episode_end(std::string_view);
  Undestroyable loaded_native_chunk();
};
#define ABSENT(name) static_assert(!detail::is_detected<detail::name##_expression, BadOptional>)
ABSENT(id);
ABSENT(render_mode);
ABSENT(metadata);
ABSENT(render);
ABSENT(close);
ABSENT(prepare);
ABSENT(tags);
ABSENT(reset_options);
ABSENT(workflow_edition);
ABSENT(tag_params);
ABSENT(spec);
ABSENT(reset);
ABSENT(reset_id);
ABSENT(load);
ABSENT(load_kwargs);
ABSENT(on_episode_end);
ABSENT(on_episode_end_id);
ABSENT(on_episode_end_lvalue);
ABSENT(loaded_native_chunk);
#undef ABSENT
struct LvalueEndPolicy : ReviewPolicy {
  void on_episode_end(std::string_view&) {}
};
static_assert(detail::is_detected<detail::on_episode_end_lvalue_expression, LvalueEndPolicy>);
static_assert(!detail::is_detected<detail::on_episode_end_id_expression, LvalueEndPolicy>);

static_assert(adapters::Constant{} == adapters::Constant{});
static_assert(!(adapters::Constant{} != adapters::Constant{}));
static_assert(noexcept(adapters::Constant{} == adapters::Constant{}));
static_assert(noexcept(adapters::Constant{} != adapters::Constant{}));
static_assert(NO_ADAPTER == NO_ADAPTER);
static_assert(noexcept(NO_ADAPTER == NO_ADAPTER));
static_assert(RandomSample{} == RandomSample{});
static_assert(noexcept(RandomSample{} != RandomSample{}));
static_assert(EpisodeResult{} == EpisodeResult{});

static_assert(noexcept(std::declval<const adapters::TextTag&>() ==
                       std::declval<const adapters::TextTag&>()));
static_assert(std::is_same_v<decltype(RunOptions{}.with_episodes(3)), RunOptions>);
static_assert(std::is_same_v<decltype(std::declval<RunOptions&>().with_episodes(3)), RunOptions&>);
static_assert(!std::is_constructible_v<ServedEnv, std::vector<Env>>);
static_assert(!std::is_constructible_v<ServedEnv, std::array<Env, 1>>);
static_assert(!std::is_constructible_v<ServedEnv, Span<const Env>>);
static_assert(std::is_constructible_v<ServedEnv, const std::array<Env, 1>&>);

// Compile-only: zero must still select the raw-pointer overload.
Model null_model_overload() { return Model(0); }

#if __cplusplus >= 202002L
struct StdSpanPolicy {
  std::vector<Value> predict_batch(std::span<const Value> values, std::span<const PredictContext>) {
    return {values.begin(), values.end()};
  }
};
static_assert(Policy<StdSpanPolicy> && detail::predict_batch_c<StdSpanPolicy>);
static_assert(std::ranges::view<Span<int>>);
static_assert(std::ranges::borrowed_range<Span<int>>);
static_assert(!std::is_constructible_v<ServedEnv, std::span<const Env>>);
#endif

int main() {
  Value v(1);
  assert(1 == v && v == 1 && !(1 != v) && !(v != 1));
  assert(2 != v && v != 2);
  Value text("text");
  assert("text" == text && text == "text" && !("text" != text));
  adapters::Tolerant<DType> tolerant(DType::Uint8);
  assert(DType::Uint8 == tolerant && tolerant == DType::Uint8);
  assert(!(DType::Uint8 != tolerant) && DType::Int8 != tolerant);
  adapters::Text leaf{};
  adapters::InputNode input(leaf);
  assert(leaf == input && input == leaf && !(leaf != input));
  adapters::TextTag tag{};
  adapters::ObsNode obs(tag);
  assert(tag == obs && obs == tag && !(tag != obs));
  Model pointer_like(PointerConvertiblePolicy{});
  DerivedModel derived(ReviewPolicy{});
  Model copy(derived);
  assert(copy.raw() == derived.raw());
  Model lvalue_end(LvalueEndPolicy{});
  Env scalar(ReviewScalar{});
  std::vector<Env> lanes;
  lanes.push_back(std::move(scalar));
  ServedEnv vector_borrow(lanes);
  std::array<Env, 1> array{Env(ReviewScalar{})};
  const auto& named_array = array;
  ServedEnv array_borrow(named_array);
  assert(vector_borrow.lanes()[0] == lanes[0].raw());
  assert(array_borrow.lanes()[0] == array[0].raw());
  auto captured = std::make_shared<int>(42);
  std::weak_ptr<int> weak = captured;
  {
    const auto& options = ModelOptions{}.with_on_close([captured] {});
    captured.reset();
    assert(!weak.expired());
    options.on_close();
    const auto& run = RunOptions{}.with_episodes(3);
    assert(run.episodes == 3);
  }
  assert(weak.expired());
  const auto view = View{}
                        .with_backend(ViewBackend::Http)
                        .with_port(8009)
                        .with_fps(20)
                        .with_source("camera")
                        .with_format(ViewFormat::Png)
                        .with_quality(90)
                        .with_hold(Duration{1})
                        .with_step_hz(10);
  assert(view.backend == ViewBackend::Http && view.port == 8009 && view.fps == 20 &&
         *view.source == "camera" && view.format == ViewFormat::Png && view.quality == 90 &&
         std::get<Duration>(view.hold).count() == 1 && *view.step_hz == 10);
  View mutable_view;
  assert(&mutable_view.with_fps(10) == &mutable_view);
  try {
    VectorEnv negative(ReviewVector{-1});
    assert(false);
  } catch (const ValueError& error) {
    assert(std::string(error.what()).find("num_envs must not be negative") != std::string::npos);
  }
  VectorEnv positive(ReviewVector{});
  Value floating(1.0L);
  assert(floating.as_float() == 1.0);
#if __cplusplus >= 202002L
  Model batch(StdSpanPolicy{});  // Instantiate the dispatch, not only its detector.
  std::span<const Env> named_span(array);
  ServedEnv span_borrow(named_span);
  assert(span_borrow.lanes()[0] == array[0].raw());
  const std::byte storage[] = {std::byte{1}, std::byte{2}};
  const auto bytes = Value::bytes(storage);
  std::span<const std::byte> b = bytes.as_bytes();
  assert(b.size() == 2);
  auto transformed = bytes.as_bytes() |
                     std::views::transform([](std::byte x) { return std::to_integer<int>(x) + 1; });
  assert(*transformed.begin() == 2);
#endif
}
