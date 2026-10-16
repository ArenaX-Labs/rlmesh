// Borrowing and mutable/const conversions use the same public view at both standards.
#include <cassert>
#include <rlmesh.hpp>

static_assert(!std::is_constructible_v<rlmesh::Span<int>, const std::vector<int>&>);
static_assert(!std::is_constructible_v<rlmesh::Span<int>, std::array<const int, 2>&>);
static_assert(!std::is_constructible_v<rlmesh::Span<int>, rlmesh::Span<const int>>);
static_assert(!std::is_constructible_v<rlmesh::Span<int>, std::vector<double>&>);
static_assert(std::is_trivially_copyable_v<rlmesh::Span<int>>);

int main() {
  rlmesh::Span<int> empty;
  assert(empty.empty() && empty.begin() == empty.end() && empty.data() == nullptr);
  int values[] = {1, 2, 3};
  rlmesh::Span<int> writable(values);
  writable[1] = 7;
  rlmesh::Span<const int> readonly(writable);
  assert(readonly.size() == 3 && readonly[1] == 7 && readonly.data() == values);
  assert(std::distance(readonly.begin(), readonly.end()) == 3);
  rlmesh::Span<const int> tail(values + 1, 2);
  assert(tail[0] == 7 && tail.end() == values + 3);
  std::array<int, 2> array{4, 5};
  rlmesh::Span<int> array_view(array);
  array_view[0] = 6;
  const auto& const_array = array;
  assert(rlmesh::Span<const int>(const_array)[0] == 6);
  std::vector<int> vector{8, 9};
  rlmesh::Span<int> vector_view(vector);
  vector_view[1] = 10;
  const auto& const_vector = vector;
  assert(rlmesh::Span<const int>(const_vector)[1] == 10);
  const int const_values[] = {11, 12};
  assert(rlmesh::Span<const int>(const_values)[1] == 12);
}
