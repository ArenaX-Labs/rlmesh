// Inject a failure after C construction, before a C++ Box can own the result.
#define rlmesh_ext_box_elementwise_new test_box_new
#define rlmesh_space_free test_space_free
#define rlmesh_space_spec test_space_spec
#define rlmesh_space_spec_clone test_spec_clone
#include <cassert>
#include <rlmesh.hpp>
namespace {
int frees = 0;
unsigned char space, spec;
}  // namespace
extern "C" RLMeshStatus test_box_new(const RLMeshBound*, const RLMeshBound*, size_t, const int64_t*,
                                     size_t, const RLMeshBoxOptions*, RLMeshSpace** out) {
  *out = reinterpret_cast<RLMeshSpace*>(&space);
  return RLMESH_OK;
}
extern "C" void test_space_free(RLMeshSpace* p) {
  assert(p == reinterpret_cast<RLMeshSpace*>(&space));
  ++frees;
}
extern "C" const RLMeshSpaceSpec* test_space_spec(const RLMeshSpace*) {
  return reinterpret_cast<const RLMeshSpaceSpec*>(&spec);
}
extern "C" RLMeshSpaceSpec* test_spec_clone(const RLMeshSpaceSpec*) { throw std::bad_alloc(); }
int main() {
  try {
    rlmesh::ext::box_elementwise({0}, {1}, {1});
    assert(false);
  } catch (const std::bad_alloc&) {
  }
  assert(frees == 1);
}
