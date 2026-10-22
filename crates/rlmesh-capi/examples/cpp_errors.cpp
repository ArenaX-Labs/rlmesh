// Exception hierarchy and propagation through C ABI worker boundaries.
#include <cassert>
#include <cstdlib>
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

// C++ callbacks attach their exception payload at the C boundary. Exercise
// the actual server wire formatter, not only an in-process exception rethrow.
struct RecoverableEnv {
  rlmesh::spaces::Space observation_space() const { return rlmesh::spaces::Discrete(2); }
  rlmesh::spaces::Space action_space() const { return rlmesh::spaces::Discrete(2); }
  rlmesh::ResetResult reset(std::optional<int64_t>, const std::optional<rlmesh::InfoDict>&) {
    return {0, {}};
  }
  rlmesh::StepResult step(std::optional<rlmesh::Value>) {
    throw rlmesh::RecoverableEnvironmentException("raw C++ retry text");
  }
};

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
  rlmesh::EnvServer server(RecoverableEnv{},
                           rlmesh::EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
  server.start();
  // TODO: dial a session address once the C API serves envs on sessions (the
  // parity step); until then Python 0.2 reaches a v1 endpoint only through the
  // v1 escape.
  const std::string probe = R"PY(uv run --no-sync python -c '
import sys
import rlmesh
import rlmesh.numpy
from rlmesh._upgrade import v1_endpoints
with v1_endpoints():
    env = rlmesh.numpy.RemoteEnv(sys.argv[1])
with env:
    env.reset()
    try:
        env.step(0)
    except rlmesh.RecoverableEnvironmentException as error:
        assert error.code == "INTERNAL", error.code
        assert error.is_recoverable
        assert error.args[0] == "step failed: raw C++ retry text", error.args
        assert str(error) == "[INTERNAL] step failed: raw C++ retry text", str(error)
    else:
        raise AssertionError("expected recoverable C++ environment failure")
' )PY" + server.address();
  const int result = std::system(probe.c_str());
  assert(result == 0);
  server.shutdown();
}
