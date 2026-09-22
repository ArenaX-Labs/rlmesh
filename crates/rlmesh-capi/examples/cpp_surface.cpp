// Compile-only check that the whole rlmesh.hpp surface still instantiates: every
// public type and member is named here, so a template that stops compiling fails
// `mise run test:cxx` instead of a downstream build. Compiled to an object, never
// linked or run — the calls below are not meant to succeed at runtime.
#include <chrono>
#include <cstdint>
#include <optional>
#include <rlmesh.hpp>
#include <string>
#include <utility>
#include <vector>

namespace surface {

// Error / Result / Status ergonomics, including RLMESH_TRY.
rlmesh::Result<int64_t> results(rlmesh::ValueRef value) {
  rlmesh::Error error(RLMESH_ERR_MODEL, "boom", true);
  (void)error.code();
  (void)error.message();
  (void)error.is_recoverable();
  (void)error.is_cancelled();
  (void)rlmesh::Error::from_last(RLMESH_ERR_INTERNAL).code();

  rlmesh::Status status = rlmesh::ok();
  if (!status || !status.ok() || !status.has_value()) return status.error();
  status.value();
  status.unwrap();
  rlmesh::Status chained = status.and_then([] { return rlmesh::ok(); });
  if (!chained) return chained.error();

  rlmesh::Result<int64_t> discrete = value.as_discrete();
  (void)discrete.value_or(int64_t{0});
  (void)discrete.has_value();
  rlmesh::Result<int64_t> mapped = value.as_discrete().map([](int64_t n) { return n + 1; });
  if (!mapped) return mapped.error();
  rlmesh::Result<int64_t> doubled =
      std::move(discrete).and_then([](int64_t n) -> rlmesh::Result<int64_t> { return n * 2; });
  if (!doubled) return doubled.error();
  int64_t unwrapped = std::move(mapped).unwrap();
  int64_t tried = RLMESH_TRY(value.as_discrete());
  return unwrapped + *doubled + tried;
}

// Every ValueRef reader, plus the Tensor view.
rlmesh::Status read_value(rlmesh::ValueRef value) {
  (void)value.raw();
  (void)value.kind();
  auto tensor = value.as_tensor();
  if (!tensor) return tensor.error();
  (void)tensor->data();
  (void)tensor->dtype();
  (void)tensor->ndim();
  (void)tensor->shape();
  (void)tensor->strides();
  (void)tensor->is_contiguous();
  (void)tensor->numel();
  (void)tensor->as<float>();

  auto discrete = value.as_discrete();
  if (!discrete) return discrete.error();
  auto text = value.as_text();
  if (!text) return text.error();
  auto bits = value.as_multi_binary();
  if (!bits) return bits.error();
  auto nvec = value.as_multi_discrete();
  if (!nvec) return nvec.error();
  auto array_len = value.array_len();
  if (!array_len) return array_len.error();
  auto size = value.size();
  if (!size) return size.error();
  (void)value.at(0);
  auto key = value.key(0);
  if (!key) return key.error();
  (void)value.get("field");
  (void)value.at_key(0);
  auto items = value.items();
  if (!items) return items.error();
  return rlmesh::ok();
}

// Every owned-value constructor, including the typed Box convenience.
rlmesh::Result<rlmesh::Value> build_value() {
  std::vector<rlmesh::Value> children;
  auto f32 = rlmesh::Value::box(std::vector<float>{0.0F}, {1});
  if (!f32) return f32.error();
  children.push_back(std::move(*f32));
  auto f64 = rlmesh::Value::box(std::vector<double>{0.0}, {1});
  if (!f64) return f64.error();
  children.push_back(std::move(*f64));
  auto i32 = rlmesh::Value::box(std::vector<int32_t>{0}, {1});
  if (!i32) return i32.error();
  children.push_back(std::move(*i32));
  auto i64 = rlmesh::Value::box(std::vector<int64_t>{0}, {1});
  if (!i64) return i64.error();
  children.push_back(std::move(*i64));
  auto u8 = rlmesh::Value::box(std::vector<uint8_t>{0}, {1});
  if (!u8) return u8.error();
  children.push_back(std::move(*u8));
  const std::vector<uint8_t> flags(2, 0);
  auto boolean = rlmesh::Value::box(flags.data(), rlmesh::dtype_of<bool>(), {2});
  if (!boolean) return boolean.error();
  children.push_back(std::move(*boolean));
  auto tuple = rlmesh::Value::tuple(std::move(children));
  if (!tuple) return tuple.error();

  std::vector<std::pair<std::string, rlmesh::Value>> entries;
  auto discrete = rlmesh::Value::discrete(1);
  if (!discrete) return discrete.error();
  entries.emplace_back("discrete", std::move(*discrete));
  auto text = rlmesh::Value::text("hello");
  if (!text) return text.error();
  entries.emplace_back("text", std::move(*text));
  auto bits = rlmesh::Value::multi_binary({0, 1});
  if (!bits) return bits.error();
  entries.emplace_back("bits", std::move(*bits));
  auto nvec = rlmesh::Value::multi_discrete({0, 1});
  if (!nvec) return nvec.error();
  entries.emplace_back("nvec", std::move(*nvec));
  entries.emplace_back("tuple", std::move(*tuple));

  auto dict = rlmesh::Value::dict(std::move(entries));
  if (!dict) return dict.error();
  (void)dict->raw();
  (void)dict->ref().kind();
  rlmesh::Value moved = std::move(*dict);
  return rlmesh::Value(moved.release());
}

// Every SpaceRef accessor, plus zeros_for over an arbitrary space.
rlmesh::Status read_space(rlmesh::SpaceRef space) {
  (void)space.raw();
  (void)space.valid();
  (void)space.kind();
  (void)space.dtype();
  (void)space.ndim();
  auto shape = space.shape();
  if (!shape) return shape.error();
  auto numel = space.numel();
  if (!numel) return numel.error();
  auto bounds = space.bounds(0);
  if (!bounds) return bounds.error();
  (void)bounds->low;
  (void)bounds->high;
  auto discrete = space.discrete();
  if (!discrete) return discrete.error();
  (void)discrete->n;
  (void)discrete->start;
  auto length = space.text_length();
  if (!length) return length.error();
  (void)length->min;
  (void)length->max;
  auto charset = space.charset();
  if (!charset) return charset.error();
  auto nvec = space.nvec();
  if (!nvec) return nvec.error();
  auto size = space.size();
  if (!size) return size.error();
  (void)space.at(0);
  auto key = space.key(0);
  if (!key) return key.error();
  (void)space.get("field");
  (void)space.at_key(0);
  auto items = space.items();
  if (!items) return items.error();
  auto zeros = rlmesh::zeros_for(space);
  if (!zeros) return zeros.error();
  return rlmesh::ok();
}

// The batched policy shape: Batch indexing and per-row episodes.
rlmesh::Result<rlmesh::Model> batched_model() {
  return rlmesh::Model::from_predict_batch(
      [](const rlmesh::Batch& batch) -> rlmesh::Result<std::vector<rlmesh::Value>> {
        (void)batch.contract();
        (void)batch.has_observations();
        (void)batch.env_id();
        (void)batch.session_id();
        (void)batch.request_id();
        (void)batch.observation_space();
        std::vector<rlmesh::Value> actions;
        for (size_t i = 0; i < batch.size(); ++i) {
          rlmesh::Episode episode = batch.episode(i);
          (void)episode.id;
          (void)episode.seed;
          (void)episode.predict_index;
          (void)episode.predict_seed;
          if (std::optional<rlmesh::ValueRef> row = batch.at(i)) (void)row->kind();
          auto action = rlmesh::zeros_for(batch.action_space());
          if (!action) return action.error();
          actions.push_back(std::move(*action));
        }
        return actions;
      });
}

// The single-env policy plus the whole Model lifecycle.
rlmesh::Status drive(const std::string& address) {
  auto model = rlmesh::Model::from_predict([](const rlmesh::Request& request) {
    (void)request.env_id();
    (void)request.session_id();
    (void)request.request_id();
    (void)request.episode().id;
    (void)request.episode().seed;
    (void)request.episode().predict_index;
    (void)request.episode().predict_seed;
    (void)request.observation();
    (void)request.observation_space();
    (void)request.batch().size();
    return rlmesh::zeros_for(request.action_space());
  });
  if (!model) return model.error();
  model->on_episode_end([](std::string_view, std::string_view) {}).on_close([] {});

  rlmesh::RunOptions options;
  options.max_episodes = 2;
  options.seeded = true;
  options.base_seed = 7;
  options.max_episode_steps = 10;
  options.max_episode_seconds = 1.5;
  options.execution_horizon = 1;
  options.close_env = true;
  options.episode_seeds = {1, 2};
  options.trial_indexed = true;
  options.trial_index_base = 100;
  auto report = model->run_local(address, options);
  if (!report) return report.error();
  rlmesh::RunReport copied = *report;
  (void)copied.total_episodes;
  (void)copied.total_steps;
  (void)copied.total_reward;
  (void)copied.mean_reward;
  (void)copied.terminated_episodes;
  (void)copied.truncated_episodes;

  rlmesh::ServeOptions serve_options;
  serve_options.token = "secret";
  serve_options.allow_remote_shutdown = true;
  serve_options.idle_timeout = std::chrono::seconds(30);
  serve_options.drain_timeout = std::chrono::milliseconds(500);
  serve_options.close_timeout = std::chrono::seconds(5);
  serve_options.predict_concurrency = 4;
  // The edition this model was authored against: the bare base, kept until its
  // author deliberately moves it (it selects a source build's own
  // "2026.06-dev.<git>" spelling as readily as the sealed name).
  serve_options.workflow_edition = "2026.06";

  rlmesh::Model moved = std::move(*model);
  moved.cancel();
  return moved.serve(address, serve_options);
}

}  // namespace surface
