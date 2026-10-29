// The echo env of python/rlmesh/tests/conformance/native_gate.py, for the
// C++ gate program and the Unreal game thread's.
#pragma once
#include <cmath>
#include <cstdio>
#include <rlmesh.hpp>
#include <string>

namespace gate {
using namespace rlmesh;
using namespace rlmesh::spaces;

struct Unsupported {
  std::string why;
};

inline Space f32_box(double low, double high, std::vector<int64_t> shape) {
  return Box(low, high, shape);
}
inline Space typed_box(Bound low, Bound high, std::vector<int64_t> shape, DType dtype) {
  return Box(low, high, shape, BoxOptions{}.with_dtype(dtype));
}

// The gate's SPACES, by name.
inline Space gate_space(const std::string& row) {
  const double inf = INFINITY;
  if (row == "box_f32_bounded") return f32_box(-1.0, 2.0, {3});
  if (row == "box_f64_inf") return typed_box(-inf, inf, {2, 2}, DType::Float64);
  if (row == "box_mixed_bounds") return ext::box_elementwise({-1.0, -inf}, {inf, 3.0}, {2});
  if (row == "box_f16") return typed_box(-1.0, 1.0, {4}, DType::Float16);
  if (row == "box_i8") return typed_box(-5, 5, {3}, DType::Int8);
  if (row == "box_u8_image") return typed_box(0, 255, {6, 5, 3}, DType::Uint8);
  if (row == "box_u16") return typed_box(0, 60000, {2}, DType::Uint16);
  if (row == "box_i64") return typed_box(-(int64_t{1} << 40), int64_t{1} << 40, {2}, DType::Int64);
  if (row == "box_u64") return typed_box(0, int64_t{1} << 40, {2}, DType::Uint64);
  if (row == "box_i16") return typed_box(-300, 300, {3}, DType::Int16);
  if (row == "box_i32") return typed_box(-(1 << 20), 1 << 20, {3}, DType::Int32);
  if (row == "box_u32") return typed_box(0, 1 << 30, {2}, DType::Uint32);
  if (row == "box_bool") return typed_box(0, 1, {3}, DType::Bool);
  if (row == "discrete") return Discrete(4);
  if (row == "discrete_start") return Discrete(5, DiscreteOptions{}.with_start(-2));
  if (row == "multi_discrete") return MultiDiscrete({3, 5, 2});
  if (row == "multi_discrete_2d") throw Unsupported{"C++ MultiDiscrete takes a 1-D nvec"};
  if (row == "multi_discrete_start") throw Unsupported{"C++ MultiDiscrete takes no start"};
  if (row == "multi_binary") return MultiBinary({4});
  if (row == "multi_binary_2d") return MultiBinary({2, 3});
  if (row == "dict_flat") return Dict({{"pos", f32_box(-1, 1, {2})}, {"gripper", Discrete(2)}});
  if (row == "dict_nested")
    return Dict({{"robot", Dict({{"q", f32_box(-1, 1, {3})}, {"mode", Discrete(2)}})},
                 {"scene", Dict({{"img", typed_box(0, 255, {4, 4, 3}, DType::Uint8)}})}});
  if (row == "dict_keys_that_escape_alike")
    return Dict({{"Pos", Discrete(3)}, {"pos", Discrete(5)}});
  if (row == "dict_key_named_action") return Dict({{"action", f32_box(-1, 1, {1})}});
  if (row == "tuple") return Tuple({Discrete(2), f32_box(-1, 1, {2})});
  if (row == "tuple_nested")
    return Tuple({Dict({{"a", Discrete(2)}}), Tuple({MultiBinary({2}), f32_box(-1, 1, {2})})});
  if (row == "text") return Text(12, TextOptions{}.with_min_length(1).with_charset("abc xyz"));
  if (row == "scalar_box") return f32_box(-1, 1, {});
  return f32_box(-1, 1, {3});
}

// The gate's rich_info(t).
inline InfoDict rich_info(int64_t t) {
  const std::byte raw[] = {std::byte{0x00}, std::byte{0xff}};
  return {{"nested", Metadata{{"a", Metadata{{"b", t}, {"c", Value::list({1, 2.5, "x"})}}},
                              {"flag", t % 2 == 0}}},
          {"nan", NAN},
          {"inf", -INFINITY},
          {"bytes", Value::bytes(raw)},
          {"$dollar", 1},
          {"none", Value::none()},
          {"third", 1.0 / 3},
          {"is_success", t >= 3}};
}

class EchoEnv {
  std::string row_;
  Space space_;
  int64_t t_ = 0;

 public:
  explicit EchoEnv(std::string row) : row_(std::move(row)), space_(gate_space(row_)) {}
  Space observation_space() const { return space_; }
  Space action_space() const { return space_; }
  ResetResult reset(std::optional<int64_t>, const std::optional<InfoDict>& options) {
    if (row_ == "error_reset") throw ValueError("boom in reset");
    t_ = 0;
    InfoDict info;
    if (options) info = {{"options", *options}};
    return {space_.sample(), info};
  }
  StepResult step(std::optional<Value> action) {
    if (row_ == "error_step") throw RuntimeError("boom in step");
    ++t_;
    const bool done = t_ >= 3;
    InfoDict info = row_ == "info" ? rich_info(t_) : InfoDict{{"t", t_}};
    return {*action, static_cast<double>(t_), done, done && row_ == "both_done", info};
  }
  std::optional<Value> render() const {
    std::vector<uint8_t> frame(4 * 5 * 3, 0);
    for (size_t i = 0; i < frame.size(); i += 3) frame[i] = static_cast<uint8_t>(t_);
    return Tensor(frame.data(), frame.size(), {4, 5, 3}, DType::Uint8);
  }
};

inline void wait_for_eof() {
  while (std::fgetc(stdin) != EOF) {
  }
}

}  // namespace gate
