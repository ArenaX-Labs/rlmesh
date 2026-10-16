// Compile with hostile host macros still active, warnings as errors, and no RTTI.
#define check(expr) ((expr) ? 1 : 0)
#define verify(expr) check(expr)
#define ensure(expr) check(expr)
#define checkf(expr, ...) check(expr)
#define TEXT(text) L##text
#define None 0L
#define Bool int
#define True 1
#define False 0
#define Status int
#define Success 0
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define ERROR 0
#define IN
#define OUT
#define DELETE 0
#include <rlmesh.hpp>

static_assert(check(True) && verify(True) && ensure(True) && checkf(True, "ok"));
static_assert(None == 0 && False == 0 && Success == 0 && ERROR == 0 && DELETE == 0);
static_assert(min(2, 3) == 2 && max(2, 3) == 3);
static_assert(sizeof(TEXT("x")) == 2 * sizeof(wchar_t));
using HostBool = Bool;
using HostStatus = Status;
IN OUT void host_annotations();

// Instantiate policy corners with the macros in force as well as parsing them.
#include "cpp_surface.cpp"

namespace {
struct HostEnv {
  rlmesh::spaces::Space observation_space() { return rlmesh::spaces::Discrete(2); }
  rlmesh::spaces::Space action_space() { return observation_space(); }
  rlmesh::ResetResult reset(std::optional<int64_t>, const std::optional<rlmesh::InfoDict>&) {
    return {0, {}};
  }
  rlmesh::StepResult step(std::optional<rlmesh::Value>) { return {0, 1, true, false, {}}; }
  std::optional<rlmesh::Value> render() { return {}; }
  void close() {}
};
struct HostVectorEnv {
  rlmesh::spaces::Space single_observation_space() { return rlmesh::spaces::Discrete(2); }
  rlmesh::spaces::Space single_action_space() { return single_observation_space(); }
  int num_envs() { return 1; }
  rlmesh::ResetResult reset(std::optional<rlmesh::VectorSeed>,
                            const std::optional<rlmesh::InfoDict>&) {
    return {0, {}};
  }
  rlmesh::VectorStepResult step(std::optional<rlmesh::Value>) {
    return {0, {1}, {true}, {false}, {}};
  }
  void close() {}
};
}  // namespace
void hostile_api() {
  rlmesh::Value floating(1.0L);
  rlmesh::Env scalar(HostEnv{});
  rlmesh::VectorEnv vector(HostVectorEnv{});
  rlmesh::Model policy(Greedy{});
  auto options = rlmesh::RunOptions{}.with_episodes(2).with_seeds({0, 1});
  (void)policy.run(scalar, options);
  (void)policy.run(vector, options);
}
