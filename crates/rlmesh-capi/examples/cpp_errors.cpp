// Exception hierarchy and propagation through C ABI worker boundaries.
#include <cassert>
#include <rlmesh.hpp>
#include <type_traits>

static_assert(std::is_base_of_v<rlmesh::RuntimeError, rlmesh::RLMeshException>);
static_assert(std::is_base_of_v<rlmesh::RLMeshException, rlmesh::ProtocolException>);
static_assert(std::is_base_of_v<rlmesh::RLMeshException, rlmesh::EnvironmentException>);
static_assert(
    std::is_base_of_v<rlmesh::EnvironmentException, rlmesh::RecoverableEnvironmentException>);
static_assert(std::is_base_of_v<rlmesh::RLMeshException, rlmesh::ModelException>);
static_assert(std::is_base_of_v<rlmesh::ValueError, rlmesh::AdapterResolutionError>);
static_assert(std::is_base_of_v<rlmesh::OSError, rlmesh::ConnectionError>);
static_assert(std::is_base_of_v<rlmesh::OSError, rlmesh::TimeoutError>);
static_assert(!std::is_base_of_v<std::exception, rlmesh::ext::CancelledError>);
static_assert(!std::is_base_of_v<rlmesh::RLMeshException, rlmesh::ValueError>);
static_assert(!std::is_base_of_v<rlmesh::RuntimeError, rlmesh::OSError>);
static_assert(std::is_base_of_v<std::logic_error, rlmesh::TypeError>);

int main() {
  try {
    rlmesh::Value("text").as_int();
    assert(false);
  } catch (const rlmesh::TypeError&) {
  }
  try {
    rlmesh::spaces::Discrete invalid(0);
    assert(false);
  } catch (const rlmesh::ValueError&) {
  }
  try {
    throw rlmesh::RecoverableEnvironmentException("retry");
  } catch (const rlmesh::EnvironmentException& e) {
    assert(e.is_recoverable());
    assert(e.code() == rlmesh::ErrorCode::Internal);
    assert(std::string(e.what()) == "[INTERNAL] retry");
  }
  try {
    throw rlmesh::ext::CancelledError();
  } catch (const std::exception&) {
    assert(false);
  } catch (const rlmesh::ext::CancelledError& e) {
    assert(std::string(e.what()) == "cancelled");
  }
}
