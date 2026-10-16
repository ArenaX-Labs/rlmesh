// The C++ host declaration goes through ABI 5 server options.
#include <cassert>
#include <rlmesh.hpp>
int main() {
  auto text = std::string(RLMESH_HOST_RUNTIME_JSON);
  assert(text.find("\"language\":\"c++\"") != std::string::npos);
#if (defined(_MSVC_LANG) ? _MSVC_LANG : __cplusplus) >= 202002L
  assert(text.find("\"language_version\":\"20\"") != std::string::npos);
#else
  assert(text.find("\"language_version\":\"17\"") != std::string::npos);
#endif
}
