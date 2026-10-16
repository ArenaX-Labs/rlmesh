// Count C reads to catch eager initialization, and fail the first lazy attempt.
#define rlmesh_adapters_parts_len test_parts_len
#define rlmesh_adapters_parts_get test_parts_get
#define rlmesh_adapters_rotation_dims_len test_dims_len
#define rlmesh_adapters_rotation_dims_get test_dims_get
#include <cassert>
#include <rlmesh.hpp>
namespace {
int parts_reads = 0, dims_reads = 0;
bool fail_dims = true;
}  // namespace
extern "C" size_t test_parts_len() {
  ++parts_reads;
  return 1;
}
extern "C" const char* test_parts_get(size_t) { return "left_arm"; }
extern "C" size_t test_dims_len() {
  ++dims_reads;
  return 1;
}
extern "C" RLMeshStatus test_dims_get(size_t, RLMeshRotationEncoding* encoding, uint32_t* width) {
  if (fail_dims) {
    fail_dims = false;
    rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_VALUE_ERROR, "lazy table failure");
    return RLMESH_ERR_INVALID_ARGUMENT;
  }
  *encoding = RLMESH_ROTATION_ENCODING_EULER_XYZ;
  *width = 3;
  return RLMESH_OK;
}
int main() {
  assert(parts_reads == 0 && dims_reads == 0);
  const auto& parts = rlmesh::adapters::PARTS();
  assert(parts.size() == 1 && std::string(parts[0]) == "left_arm");
  auto reads = parts_reads;
  assert(&rlmesh::adapters::PARTS() == &parts && parts_reads == reads);
  try {
    rlmesh::adapters::ROTATION_DIMS();
    assert(false);
  } catch (const rlmesh::ValueError& error) {
    assert(std::string(error.what()) == "lazy table failure");
  }
  const auto& dims = rlmesh::adapters::ROTATION_DIMS();
  assert(dims.size() == 1 && dims[0].first == rlmesh::adapters::RotationEncoding::EulerXyz &&
         dims[0].second == 3);
  reads = dims_reads;
  assert(&rlmesh::adapters::ROTATION_DIMS() == &dims && dims_reads == reads);
}
