#include "cpp_mixed.hpp"

rlmesh::RunOptions mixed_options(rlmesh::RunOptions options) {
  return std::move(options).with_episodes(2).with_seeds({0, 1});
}
rlmesh::Span<const int> mixed_view(rlmesh::Span<const int> values) { return values; }
rlmesh::adapters::ImageTag mixed_tag(rlmesh::adapters::ImageTag tag) {
  tag.part = "left";
  return tag;
}
rlmesh::Model mixed_model() {
  return rlmesh::Model([](const rlmesh::Value& value) { return value; });
}
size_t mixed_run_options_size() { return sizeof(rlmesh::RunOptions); }
