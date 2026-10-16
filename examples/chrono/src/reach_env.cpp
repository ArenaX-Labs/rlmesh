#include "reach_env.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "chrono/ChVersion.h"

using chrono::ChCoordsysd;
using chrono::ChVector3d;
using chrono::ChVectorDynamic;

namespace chrono_reach {

namespace {

// Link lengths of the robot in Chrono's industrial demo: shoulder height H,
// then biceps, forearm and wrist-to-flange (metres).
constexpr std::array<double, 4> kLengths = {0.180, 0.250, 0.317, 0.258};
constexpr const char* kInstruction = "move the gripper to the red sphere";

// The semantics a model's adapter spec resolves against. Structure (shapes,
// dtypes) comes from the spaces; the tags carry only meaning. `target_pos` has
// no registered role, so it gets an `x/` (deliberately non-standard) one.
constexpr const char* kTags = R"({
  "observation": {
    "image": {"type": "image", "role": "image/primary", "layout": "hwc"},
    "joint_pos": {"type": "state", "role": "proprio/joint_pos"},
    "joint_vel": {"type": "state", "role": "proprio/joint_vel"},
    "eef_pos": {"type": "state", "role": "proprio/eef_pos"},
    "target_pos": {"type": "state", "role": "x/target_pos"},
    "instruction": {"type": "text", "role": "text/instruction"}
  },
  "action": {
    "components": [{"role": "action/delta_eef_pos", "dim": 3}],
    "clip": [-1.0, 1.0]
  }
})";

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

scene::Vec3 rt(const ChVector3d& v) { return {v.x(), v.y(), v.z()}; }

std::vector<float> to_floats(const ChVectorDynamic<>& v) {
  std::vector<float> out(static_cast<size_t>(v.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i)
    out[static_cast<size_t>(i)] = static_cast<float>(v[i]);
  return out;
}

std::vector<float> to_floats(const ChVector3d& v) {
  return {static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z())};
}

bool finite(const ChVectorDynamic<>& v) {
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (!std::isfinite(v[i])) return false;
  }
  return v.size() > 0;
}

}  // namespace

IndustrialReach::IndustrialReach(Options options, std::unique_ptr<Renderer> renderer)
    : options_(std::move(options)), renderer_(std::move(renderer)) {}

void IndustrialReach::build_world() {
  system_ = std::make_unique<chrono::ChSystemNSC>();
  system_->SetGravitationalAcceleration(ChVector3d(0, -9.81, 0));

  auto floor = chrono_types::make_shared<chrono::ChBody>();
  floor->SetFixed(true);
  system_->Add(floor);

  robot_ =
      chrono_types::make_shared<chrono::industrial::IndustrialRobot6dof>(system_.get(), kLengths);
  // Analytic inverse kinematics, set up from the robot's joint frames at home.
  std::array<ChCoordsysd, 7> joints;
  const auto markers = robot_->GetMarkers();
  for (size_t i = 0; i < joints.size() && i < markers.size(); ++i) {
    joints[i] = markers[i]->GetAbsCoordsys();
  }
  kinematics_ = std::make_unique<chrono::industrial::IndustrialKinematics6dofSpherical>(
      joints, std::array<double, 2>{0, chrono::CH_PI_2});

  const ChCoordsysd tcp = robot_->GetMarkerTCP()->GetAbsCoordsys();
  home_ = tcp.pos;
  tool_rotation_ = tcp.rot;
  command_ = home_;
}

ChVectorDynamic<> IndustrialReach::solve_ik(const ChVector3d& tcp) const {
  return kinematics_->GetIK(ChCoordsysd(tcp, tool_rotation_));
}

bool IndustrialReach::reachable(const ChVector3d& tcp) const {
  if (tcp.y() < 0.06) return false;                // keep the tool off the floor
  if ((tcp - home_).Length() > 0.4) return false;  // stay in the comfortable workspace
  // The analytic IK clamps an out-of-reach target to finite angles, so check
  // the round trip: the joints it returns must put the tool where we asked,
  // with some reach to spare past the target.
  for (const double stretch : {1.0, 1.08}) {
    const ChVector3d probe = home_ + (tcp - home_) * stretch;
    const ChVectorDynamic<> joints = solve_ik(probe);
    if (!finite(joints)) return false;
    if ((kinematics_->GetFK(joints).pos - probe).Length() > 1e-3) return false;
  }
  return true;
}

ChVector3d IndustrialReach::tcp_position() const {
  return robot_->GetMarkerTCP()->GetAbsCoordsys().pos;
}

double IndustrialReach::distance() const { return (tcp_position() - target_).Length(); }

rlmesh::ResetResult IndustrialReach::reset(std::optional<int64_t> seed,
                                           const std::optional<rlmesh::InfoDict>& options) {
  const auto trial = rlmesh::trial_index(options);
  if (seed) {
    rng_.seed(static_cast<uint64_t>(*seed));
  } else if (trial) {
    rng_.seed(static_cast<uint64_t>(*trial) * 7919u + 17u);
  }
  build_world();
  steps_ = 0;
  success_ = false;
  trail_.clear();
  ++episodes_;

  // A target the arm can reach with its tool held level, away from home.
  std::uniform_real_distribution<double> spread(-0.28, 0.28);
  std::uniform_real_distribution<double> lift(-0.18, 0.12);
  target_ = home_ + ChVector3d(0.1, 0, 0);
  for (int attempt = 0; attempt < 64; ++attempt) {
    const ChVector3d candidate = home_ + ChVector3d(spread(rng_), lift(rng_), spread(rng_));
    if ((candidate - home_).Length() > 0.12 && reachable(candidate)) {
      target_ = candidate;
      break;
    }
  }
  // Settle the arm at home under gravity before the first observation.
  for (int i = 0; i < 20; ++i) {
    robot_->SetSetpoints(solve_ik(command_), system_->GetChTime());
    system_->DoStepDynamics(options_.physics_dt);
  }

  std::printf("episode %d: seed=%lld trial=%lld target=(%.3f, %.3f, %.3f)\n", episodes_,
              static_cast<long long>(seed.value_or(-1)), static_cast<long long>(trial.value_or(-1)),
              target_.x(), target_.y(), target_.z());
  std::fflush(stdout);

  auto observation = observe();
  return {std::move(observation),
          {{"target", rlmesh::Value::list({target_.x(), target_.y(), target_.z()})}}};
}

rlmesh::StepResult IndustrialReach::step(std::optional<rlmesh::Value> action) {
  if (!system_) throw rlmesh::EnvironmentException("step before reset");
  if (!action) throw rlmesh::TypeError("step without an action");
  auto tensor = action->as_tensor();
  const float* delta = reinterpret_cast<const float*>(tensor.buffer().data());
  if (tensor.dtype() != rlmesh::DType::Float32 || tensor.size() != 3) {
    throw rlmesh::TypeError("expected a contiguous float32[3] action");
  }

  // Move the tool command, then let Chrono drive the joints there: the IK
  // setpoints are interpolated across the control period's physics substeps.
  ChVector3d next = command_;
  for (int i = 0; i < 3; ++i) {
    const double a = std::clamp(static_cast<double>(delta[i]), -1.0, 1.0);
    next[i] += a * options_.max_delta;
  }
  if (!reachable(next)) next = command_;  // refuse to leave the workspace
  const int substeps =
      std::max(1, static_cast<int>(std::lround(options_.control_dt / options_.physics_dt)));
  double ik_ms = 0, physics_ms = 0;
  for (int i = 1; i <= substeps; ++i) {
    const double blend = static_cast<double>(i) / substeps;
    const ChVector3d waypoint = command_ + (next - command_) * blend;
    const auto ik_start = Clock::now();
    robot_->SetSetpoints(solve_ik(waypoint), system_->GetChTime());
    ik_ms += ms_since(ik_start);
    const auto physics_start = Clock::now();
    system_->DoStepDynamics(options_.physics_dt);
    physics_ms += ms_since(physics_start);
  }
  command_ = next;
  ++steps_;
  trail_.push_back(tcp_position());
  if (trail_.size() > 40) trail_.pop_front();

  const double dist = distance();
  success_ = dist < options_.success_radius;
  const bool truncated = !success_ && steps_ >= options_.max_steps;
  const auto observe_start = Clock::now();
  auto observation = observe();
  const double observe_ms = ms_since(observe_start);
  // Where this step's wall time went, so a driver can profile the env.
  return {std::move(observation),
          -dist + (success_ ? 1.0 : 0.0),
          success_,
          truncated,
          {{"is_success", success_},
           {"distance", dist},
           {"sim_time", system_->GetChTime()},
           {"substeps", substeps},
           {"physics_ms", physics_ms},
           {"ik_ms", ik_ms},
           {"observe_ms", observe_ms}}};
}

std::optional<rlmesh::Value> IndustrialReach::render() {
  if (!system_) return std::optional<rlmesh::Value>();
  auto frame = camera(options_.render_size);
  return std::optional<rlmesh::Value>(frame);
}

void IndustrialReach::close() {
  // stderr: close also runs for `--describe`, whose stdout is the label JSON.
  std::fprintf(stderr, "closing after %d episodes\n", episodes_);
  robot_.reset();
  system_.reset();
}

rlmesh::Value IndustrialReach::camera(int size) const {
  scene::Scene scene;
  const scene::Vec3 steel(0.93, 0.55, 0.12);  // industrial orange
  const scene::Vec3 joint(0.18, 0.19, 0.22);

  // The pedestal, then a capsule per link between consecutive joint frames.
  scene.capsules.push_back({{0, 0.0, 0}, {0, 0.03, 0}, 0.11, joint});
  std::vector<ChVector3d> chain;
  for (const auto& marker : robot_->GetMarkers()) chain.push_back(marker->GetAbsCoordsys().pos);
  chain.push_back(tcp_position());
  const double radii[] = {0.075, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025};
  for (size_t i = 0; i + 1 < chain.size(); ++i) {
    const double r = radii[std::min<size_t>(i, 7)];
    if ((chain[i + 1] - chain[i]).Length() > 1e-6) {
      scene.capsules.push_back({rt(chain[i]), rt(chain[i + 1]), r, steel});
    }
    scene.spheres.push_back({rt(chain[i]), r * 1.15, joint});
  }
  scene.spheres.push_back({rt(tcp_position()), 0.022, {0.95, 0.95, 0.95}});

  // The target turns green once reached; the tool's recent path trails behind.
  const scene::Vec3 target_color =
      success_ ? scene::Vec3(0.2, 0.85, 0.3) : scene::Vec3(0.9, 0.12, 0.12);
  scene.spheres.push_back({rt(target_), 0.03, target_color});
  for (size_t i = 0; i < trail_.size(); ++i) {
    if (i % 2 == 0) scene.spheres.push_back({rt(trail_[i]), 0.006, {0.98, 0.85, 0.2}});
  }
  try {
    const auto pixels = renderer_->render(scene, scene::Camera{}, size, size);
    return rlmesh::Tensor(pixels.data(), pixels.size(),
                          {static_cast<size_t>(size), static_cast<size_t>(size), 3},
                          rlmesh::DType::Uint8);
  } catch (const std::exception& error) {
    throw rlmesh::EnvironmentException(std::string("camera: ") + error.what());
  }
}

rlmesh::Value IndustrialReach::observe() const {
  auto image = camera(options_.image_size);
  const auto tensor = [](const std::vector<float>& data) -> rlmesh::Value {
    return rlmesh::Tensor(data.data(), data.size() * sizeof(float), {data.size()},
                          rlmesh::DType::Float32);
  };
  auto joint_pos = tensor(to_floats(robot_->GetMotorsPos()));
  auto joint_vel = tensor(to_floats(robot_->GetMotorsPosDt()));
  auto joint_torque = tensor(to_floats(robot_->GetMotorsForce()));
  auto eef = tensor(to_floats(tcp_position()));
  auto target = tensor(to_floats(target_));
  auto instruction = rlmesh::Value(kInstruction);

  std::vector<std::pair<std::string, rlmesh::Value>> entries;
  entries.emplace_back("image", image);
  entries.emplace_back("joint_pos", joint_pos);
  entries.emplace_back("joint_vel", joint_vel);
  entries.emplace_back("joint_torque", joint_torque);
  entries.emplace_back("eef_pos", eef);
  entries.emplace_back("target_pos", target);
  entries.emplace_back("instruction", instruction);
  rlmesh::Metadata metadata;
  for (auto& [key, value] : entries) metadata.insert_or_assign(key, std::move(value));
  return metadata;
}

rlmesh::spaces::Space IndustrialReach::observation_space() const {
  using namespace rlmesh::spaces;
  const double inf = INFINITY;
  const int64_t size = options_.image_size;
  return Dict(
      {{"image", Box(0, 255, {size, size, 3}, BoxOptions{}.with_dtype(rlmesh::DType::Uint8))},
       {"joint_pos", Box(-inf, inf, {6})},
       {"joint_vel", Box(-inf, inf, {6})},
       {"joint_torque", Box(-inf, inf, {6})},
       {"eef_pos", Box(-inf, inf, {3})},
       {"target_pos", Box(-inf, inf, {3})},
       {"instruction", Text(128)}});
}
rlmesh::spaces::Space IndustrialReach::action_space() const {
  return rlmesh::spaces::Box(-1, 1, {3});
}
rlmesh::adapters::EnvTags IndustrialReach::tags() {
  return rlmesh::adapters::EnvTags::from_json(kTags);
}
std::optional<rlmesh::Metadata> IndustrialReach::metadata() const {
  return rlmesh::Metadata{
      {"simulator", std::string("Project Chrono ") + CHRONO_VERSION},
      {"robot", "IndustrialRobot6dof"},
      {"renderer", options_.renderer},
      {"max_episode_steps", options_.max_steps},
      {std::string(rlmesh::ENV_RESET_OPTIONS_KEY), rlmesh::Value::list({"trial_index"})}};
}
}  // namespace chrono_reach
