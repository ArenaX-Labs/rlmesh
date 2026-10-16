// A distinct shared-library/runtime domain must own its own exception payload.
#include <rlmesh.hpp>
namespace {
int frees = 0;
}
extern "C" const void* cpp_payload_plugin_tag() { return &rlmesh::detail::exception_payload_tag; }
extern "C" int cpp_payload_plugin_frees() { return frees; }
extern "C" RLMeshStatus cpp_payload_plugin_load(void*, const RLMeshValue*) {
  auto* payload = new rlmesh::detail::ExceptionPayload(
      std::make_exception_ptr(std::logic_error("plugin exception")));
  rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TYPE_ERROR, "plugin exception");
  rlmesh_ext_callback_set_tagged_error_payload(
      payload,
      [](void* p) {
        ++frees;
        rlmesh::detail::free_exception_payload(p);
      },
      &rlmesh::detail::exception_payload_tag);
  return RLMESH_ERR_TYPE;
}
