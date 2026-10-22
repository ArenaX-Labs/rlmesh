// Compile-only Shapes and optional authoring signatures. Runtime cases live in
// cpp_parity.cpp. This translation unit also instantiates every policy corner.
#include <rlmesh.hpp>
#include <unordered_set>

void hashable_leaves() {
  std::unordered_set<rlmesh::adapters::Text> text;
  std::unordered_set<rlmesh::adapters::Custom> custom;
  std::unordered_set<rlmesh::adapters::CustomEncoding> encodings;
  std::unordered_set<rlmesh::adapters::ImageTag> images;
  std::unordered_set<rlmesh::adapters::TextTag> text_tags;
  (void)text;
  (void)custom;
  (void)encodings;
  (void)images;
  (void)text_tags;
}

rlmesh::adapters::ModelSpec model_spec();
struct Planner {
  static rlmesh::types::SpecArg spec() { return model_spec(); }
  static constexpr std::optional<uint32_t> native_chunk = 8;
  static constexpr std::array load_keys{rlmesh::types::LoadKey{"checkpoint"}};
  void load(const rlmesh::Kwargs&) {}
  rlmesh::Value predict(const rlmesh::Value&) { return 0; }
  rlmesh::Value predict_chunk(const rlmesh::Value&, uint32_t horizon,
                              const rlmesh::types::PredictContext& context) {
    ++context.state.get_or_emplace<uint64_t>("plans");
    return rlmesh::Value::list(std::vector<rlmesh::Value>(horizon, 0));
  }
};
void planner_shape(const rlmesh::Kwargs& config) {
  auto model = rlmesh::Model::from_config<Planner>(config);
  model.serve("tcp://0.0.0.0:50051");
}

struct Greedy {
  rlmesh::Value predict(const rlmesh::Value& observation) {
    (void)observation;
    return rlmesh::Value(0);
  }
};
void shapes(rlmesh::Env& env, rlmesh::SpecArg spec) {
  auto model = rlmesh::Model(Greedy{}, rlmesh::ModelOptions{}.with_spec(spec));
  auto result = model.run(env, rlmesh::RunOptions{}.with_episodes(2).with_seeds({0, 1}));
  auto b = model.bind(env);
  auto [obs, info] = b.reset(0);
  auto [obs2, reward, terminated, truncated, info2] = b.step();
  (void)result;
  (void)obs2;
  (void)reward;
  (void)terminated;
  (void)truncated;
  (void)info;
  (void)info2;
}
struct MixedCorners {
  static constexpr std::optional<uint32_t> native_chunk = 2;
  rlmesh::Value predict(const rlmesh::Value&, const rlmesh::PredictContext&) { return 0; }
  rlmesh::Value predict_chunk(const rlmesh::Value&) { return rlmesh::Value::list({0, 0}); }
  std::vector<rlmesh::Value> predict_batch(rlmesh::Span<const rlmesh::Value> values) {
    return std::vector<rlmesh::Value>(values.size(), 0);
  }
  std::vector<rlmesh::Value> predict_chunk_batch(rlmesh::Span<const rlmesh::Value> values,
                                                 uint32_t h, rlmesh::BatchPredictContext) {
    return std::vector<rlmesh::Value>(values.size(),
                                      rlmesh::Value::list(std::vector<rlmesh::Value>(h, 0)));
  }
};
struct HorizonContext {
  rlmesh::Value predict_chunk(const rlmesh::Value&, uint32_t h, const rlmesh::PredictContext&) {
    return rlmesh::Value::list(std::vector<rlmesh::Value>(h, 0));
  }
};
struct ContextOnly {
  rlmesh::Value predict_chunk(const rlmesh::Value&, const rlmesh::PredictContext&) {
    return rlmesh::Value::list({0});
  }
};
struct HorizonOnly {
  rlmesh::Value predict_chunk(const rlmesh::Value&, uint32_t h) {
    return rlmesh::Value::list(std::vector<rlmesh::Value>(h, 0));
  }
};
struct BatchOnly {
  std::vector<rlmesh::Value> predict_batch(rlmesh::Span<const rlmesh::Value> v,
                                           rlmesh::BatchPredictContext) {
    return std::vector<rlmesh::Value>(v.size(), 0);
  }
};
struct PlainChunkBatch {
  std::vector<rlmesh::Value> predict_chunk_batch(rlmesh::Span<const rlmesh::Value> v) {
    return std::vector<rlmesh::Value>(v.size(), rlmesh::Value::list({0}));
  }
};
void instantiate_corners() {
  rlmesh::Model a(MixedCorners{}), b(HorizonContext{}), c(ContextOnly{}), d(HorizonOnly{}),
      e(BatchOnly{}), f(PlainChunkBatch{});
}

// Requires-expression parameters are lvalues: retain that detection behavior.
struct BatchReferences {
  std::vector<rlmesh::Value> predict_batch(rlmesh::Span<const rlmesh::Value>& values,
                                           rlmesh::BatchPredictContext&) {
    return std::vector<rlmesh::Value>(values.size(), 0);
  }
  std::vector<rlmesh::Value> predict_chunk_batch(rlmesh::Span<const rlmesh::Value>& values,
                                                 uint32_t& horizon) {
    return std::vector<rlmesh::Value>(values.size(),
                                      rlmesh::Value::list(std::vector<rlmesh::Value>(horizon, 0)));
  }
};
struct WrongReturn {
  int predict(const rlmesh::Value&) { return 0; }
};
static_assert(!rlmesh::Policy<WrongReturn>);
static_assert(!rlmesh::Policy<int>);
static_assert(rlmesh::Policy<BatchReferences>);
rlmesh::Value function_policy(const rlmesh::Value& value) { return value; }
void callable_shapes() {
  rlmesh::Model reference_policy(BatchReferences{});
  rlmesh::Model pointer_policy(&function_policy);
}
