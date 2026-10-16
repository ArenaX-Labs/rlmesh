// The C++ host declaration goes through ABI 5 server options.
#include <cassert>
#include <rlmesh.hpp>
int main() {
  auto text = std::string(RLMESH_HOST_RUNTIME_JSON);
  assert(text.find("\"language\":\"c++\"") != std::string::npos);
  assert(text.find("\"language_version\"") == std::string::npos);
}
