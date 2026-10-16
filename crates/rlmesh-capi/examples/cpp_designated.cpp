#include <span>
// C++20 convenience syntax; the options remain aggregates at both standards.
#include <cassert>
#include <rlmesh.hpp>

int main() {
  const rlmesh::RunOptions options{.episodes = 2, .seeds = {0, 1}};
  assert(options.episodes == 2 && options.seeds->size() == 2);
  std::array<int, 2> values{3, 4};
  rlmesh::Span<const int> view = std::span<int, 2>(values);
  assert(view.data() == values.data() && view[1] == 4);
}
