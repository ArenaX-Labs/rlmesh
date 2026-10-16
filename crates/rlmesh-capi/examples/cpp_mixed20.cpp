#include <cassert>
#include <span>

#include "cpp_mixed.hpp"

int main() {
  assert(mixed_run_options_size() == sizeof(rlmesh::RunOptions));
  auto options = mixed_options(rlmesh::RunOptions{}.with_max_episode_steps(5));
  assert(options.episodes == 2 && *options.max_episode_steps == 5);
  assert(*options.seeds == std::vector<int64_t>({0, 1}));
  const int values[] = {3, 4};
  auto view = mixed_view(rlmesh::Span<const int>(std::span<const int>(values)));
  assert(view.data() == values && view.size() == 2 && view[1] == 4);
  rlmesh::adapters::ImageTag tag{.role = "camera"};
  auto result = mixed_tag(tag);
  tag.part = "left";
  assert(tag == result && !(tag != result));
  auto model = mixed_model();
  assert(!rlmesh::describe_json(model).empty());
}
