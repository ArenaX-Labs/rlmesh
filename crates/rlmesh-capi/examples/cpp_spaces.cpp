// The spaces C++ wrapper: shapes, typed bounds, downcasts and error classes.
// Exits nonzero on the first failed check.
#include <cmath>
#include <cstdio>
#include <limits>
#include <rlmesh.hpp>

namespace {

int failures = 0;

void check(bool ok, const char* what, int line) {
  if (!ok) {
    std::fprintf(stderr, "cpp_spaces.cpp:%d: %s failed\n", line, what);
    ++failures;
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

}  // namespace

int main() {
  using namespace rlmesh::spaces;

  // The Shape example (throwing constructors).
  auto obs = Dict({{"pos", Box(-1.0, 1.0, {3})}, {"id", Discrete(4, {.start = 1})}});
  CHECK(obs.seed(7) == 7);
  rlmesh::Value x = obs.sample();
  CHECK(obs.contains(x));
  CHECK(obs.keys() == (std::vector<std::string>{"id", "pos"}));
  CHECK(obs.size() == 2);
  const auto children = obs.spaces();
  CHECK(children.size() == obs.size());
  for (size_t i = 0; i < children.size(); ++i) {
    CHECK(children[i].first == obs.keys()[i]);
    CHECK(children[i].second == obs[children[i].first]);
  }
  CHECK(obs.contains_key("id") && !obs.contains_key("nope"));
  CHECK(!obs.get("nope").has_value());

  std::optional<Box> pos = obs["pos"].as_box();
  CHECK(pos.has_value());
  std::optional<BoxBound> low = pos->low();
  CHECK(low.has_value() && std::get<Bound>(*low).as_float() == -1.0);
  CHECK(pos->bounds_kind() == BoundsKind::Uniform);
  CHECK(pos->to_string() == "Box(-1.0, 1.0, (3,), float32)");
  CHECK(pos->shape() == (std::vector<int64_t>{3}));
  CHECK(pos->dtype() == rlmesh::DType::Float32);
  CHECK(rlmesh::to_string(rlmesh::DType::Float32) == "float32");
  CHECK(to_string(SpaceKind::MultiBinary) == "multi_binary");

  // Children are shared: the same space on every read.
  CHECK(obs["pos"] == *pos);
  CHECK(!obs.as_box().has_value());

  // Typed bounds read back exactly.
  Box wide =
      Box::create(uint64_t{0}, std::numeric_limits<uint64_t>::max(), {2}, {rlmesh::DType::Uint64});
  CHECK(std::get<Bound>(*wide.high()).as_uint() == std::numeric_limits<uint64_t>::max());
  CHECK(wide.bounds_kind() == BoundsKind::TypedUniform);
  bool threw_type = false;
  try {
    (void)std::get<Bound>(*wide.high()).as_int();
  } catch (const rlmesh::TypeError&) {
    threw_type = true;
  }
  CHECK(threw_type);

  // Error classes: fractional integer bound, wrong-kind spec.
  bool threw_value = false;
  try {
    Box bad(0.5, 3.0, {2}, {rlmesh::DType::Int32});
  } catch (const rlmesh::ValueError&) {
    threw_value = true;
  }
  CHECK(threw_value);
  try {
    Discrete bad(pos->spec());
    CHECK(false);
  } catch (const rlmesh::ValueError&) {
  }

  // Seeding reproduces; copies share the sampler.
  Discrete d(1000);
  Discrete copy = d;
  d.seed(3);
  rlmesh::Value a = d.sample();
  copy.seed(3);
  rlmesh::Value b = d.sample();
  CHECK(a.as_int() == b.as_int());
  CHECK(d.n() == 1000 && d.start() == 0);

  // Per-element bounds (C++-only extra).
  Box per = rlmesh::ext::box_elementwise({0, -5}, {1, 5}, {2}, {rlmesh::DType::Int64});
  CHECK(std::get<std::vector<Bound>>(*per.low())[1].as_int() == -5);

  // Tensor copies retain storage after the Value and the first view die.
  std::optional<rlmesh::Tensor> retained, another;
  {
    const float data[] = {1.25f, -3.5f};
    rlmesh::Value source = rlmesh::Tensor(data, sizeof(data), {2}, rlmesh::DType::Float32);
    auto tensor = source.as_tensor();
    retained = tensor;
    another = tensor;
    *retained = *another;
  }
  CHECK(retained->buffer().data() == another->buffer().data());
  another.reset();
  CHECK(retained->dtype() == rlmesh::DType::Float32 && retained->ndim() == 1);
  CHECK(retained->shape()[0] == 2 && retained->size() == 2);
  CHECK(reinterpret_cast<const float*>(retained->buffer().data())[0] == 1.25f &&
        reinterpret_cast<const float*>(retained->buffer().data())[1] == -3.5f);

  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("cpp_spaces OK\n");
  return 0;
}
