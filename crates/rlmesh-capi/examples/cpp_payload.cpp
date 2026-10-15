#include <cassert>
#include <rlmesh.hpp>
extern "C" const void* cpp_payload_plugin_tag();
extern "C" int cpp_payload_plugin_frees();
extern "C" RLMeshStatus cpp_payload_plugin_load(void*, const RLMeshValue*);
int main() {
  assert(cpp_payload_plugin_tag() != &rlmesh::detail::exception_payload_tag);
  RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
  vt.predict = [](void*, const RLMeshValue*, const RLMeshPredictContext*, RLMeshValue**) {
    return RLMESH_OK;
  };
  vt.load = cpp_payload_plugin_load;
  RLMeshModel* model = nullptr;
  auto status = rlmesh_model_new(&vt, nullptr, nullptr, &model);
  assert(status == RLMESH_ERR_TYPE && !model);
  try {
    rlmesh::detail::check(status);
    assert(false);
  } catch (const rlmesh::TypeError& error) {
    assert(std::string(error.what()) == "plugin exception");
  }
  assert(cpp_payload_plugin_frees() == 1);
  vt.load = [](void*, const RLMeshValue*) noexcept {
    return rlmesh::detail::callback([] { throw std::logic_error("main exception"); },
                                    RLMESH_ERR_MODEL);
  };
  status = rlmesh_model_new(&vt, nullptr, nullptr, &model);
  assert(status == RLMESH_ERR_MODEL && !model);
  try {
    rlmesh::detail::check(status);
    assert(false);
  } catch (const std::logic_error& error) {
    assert(std::string(error.what()) == "main exception");
  }
  assert(cpp_payload_plugin_frees() == 1);
}
