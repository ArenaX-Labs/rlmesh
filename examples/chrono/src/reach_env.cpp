#include "reach_env.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "chrono/ChVersion.h"
#include "raytracer.h"

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

raytracer::Vec3 rt(const ChVector3d& v) { return {v.x(), v.y(), v.z()}; }

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

IndustrialReach::IndustrialReach(Options options) : options_(options) {}

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

rlmesh::Result<rlmesh::ResetOutput> IndustrialReach::reset(const rlmesh::ResetArgs& args) {
  if (args.seed) {
    rng_.seed(static_cast<uint64_t>(*args.seed));
  } else if (args.trial_index) {
    rng_.seed(static_cast<uint64_t>(*args.trial_index) * 7919u + 17u);
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
              static_cast<long long>(args.seed.value_or(-1)),
              static_cast<long long>(args.trial_index.value_or(-1)), target_.x(), target_.y(),
              target_.z());
  std::fflush(stdout);

  auto observation = observe();
  if (!observation) return observation.error();
  std::ostringstream info;
  info << "{\"target\": [" << target_.x() << ", " << target_.y() << ", " << target_.z() << "]}";
  return rlmesh::ResetOutput{observation.unwrap(), info.str()};
}

rlmesh::Result<rlmesh::StepOutput> IndustrialReach::step(std::optional<rlmesh::ValueRef> action) {
  if (!system_) return rlmesh::Error(RLMESH_ERR_ENVIRONMENT, "step before reset");
  if (!action) return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "step without an action");
  auto tensor = action->as_tensor();
  if (!tensor) return tensor.error();
  const float* delta = tensor->as<float>();
  if (delta == nullptr || tensor->numel() != 3) {
    return rlmesh::Error(RLMESH_ERR_INVALID_VALUE, "expected a contiguous float32[3] action");
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
  for (int i = 1; i <= substeps; ++i) {
    const double blend = static_cast<double>(i) / substeps;
    const ChVector3d waypoint = command_ + (next - command_) * blend;
    robot_->SetSetpoints(solve_ik(waypoint), system_->GetChTime());
    system_->DoStepDynamics(options_.physics_dt);
  }
  command_ = next;
  ++steps_;
  trail_.push_back(tcp_position());
  if (trail_.size() > 40) trail_.pop_front();

  const double dist = distance();
  success_ = dist < options_.success_radius;
  const bool truncated = !success_ && steps_ >= options_.max_steps;
  auto observation = observe();
  if (!observation) return observation.error();
  std::ostringstream info;
  info << "{\"is_success\": " << (success_ ? "true" : "false") << ", \"distance\": " << dist
       << ", \"sim_time\": " << system_->GetChTime() << "}";
  return rlmesh::StepOutput{observation.unwrap(), -dist + (success_ ? 1.0 : 0.0), success_,
                            truncated, info.str()};
}

rlmesh::Result<std::optional<rlmesh::Value>> IndustrialReach::render() {
  if (!system_) return std::optional<rlmesh::Value>();
  const int size = options_.render_size;
  auto frame = rlmesh::Value::box(camera(size), {size, size, 3});
  if (!frame) return frame.error();
  return std::optional<rlmesh::Value>(frame.unwrap());
}

void IndustrialReach::close() {
  std::printf("closing after %d episodes\n", episodes_);
  std::fflush(stdout);
  robot_.reset();
  system_.reset();
}

std::vector<uint8_t> IndustrialReach::camera(int size) const {
  raytracer::Scene scene;
  const raytracer::Vec3 steel(0.93, 0.55, 0.12);  // industrial orange
  const raytracer::Vec3 joint(0.18, 0.19, 0.22);

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
  const raytracer::Vec3 target_color =
      success_ ? raytracer::Vec3(0.2, 0.85, 0.3) : raytracer::Vec3(0.9, 0.12, 0.12);
  scene.spheres.push_back({rt(target_), 0.03, target_color});
  for (size_t i = 0; i < trail_.size(); ++i) {
    if (i % 2 == 0) scene.spheres.push_back({rt(trail_[i]), 0.006, {0.98, 0.85, 0.2}});
  }
  return raytracer::render(scene, raytracer::Camera{}, size, size);
}

rlmesh::Result<rlmesh::Value> IndustrialReach::observe() const {
  const int size = options_.image_size;
  auto image = rlmesh::Value::box(camera(size), {size, size, 3});
  if (!image) return image.error();
  auto joint_pos = rlmesh::Value::box(to_floats(robot_->GetMotorsPos()), {6});
  if (!joint_pos) return joint_pos.error();
  auto joint_vel = rlmesh::Value::box(to_floats(robot_->GetMotorsPosDt()), {6});
  if (!joint_vel) return joint_vel.error();
  auto joint_torque = rlmesh::Value::box(to_floats(robot_->GetMotorsForce()), {6});
  if (!joint_torque) return joint_torque.error();
  auto eef = rlmesh::Value::box(to_floats(tcp_position()), {3});
  if (!eef) return eef.error();
  auto target = rlmesh::Value::box(to_floats(target_), {3});
  if (!target) return target.error();
  auto instruction = rlmesh::Value::text(kInstruction);
  if (!instruction) return instruction.error();

  std::vector<std::pair<std::string, rlmesh::Value>> entries;
  entries.emplace_back("image", image.unwrap());
  entries.emplace_back("joint_pos", joint_pos.unwrap());
  entries.emplace_back("joint_vel", joint_vel.unwrap());
  entries.emplace_back("joint_torque", joint_torque.unwrap());
  entries.emplace_back("eef_pos", eef.unwrap());
  entries.emplace_back("target_pos", target.unwrap());
  entries.emplace_back("instruction", instruction.unwrap());
  return rlmesh::Value::dict(std::move(entries));
}

rlmesh::Result<rlmesh::EnvConfig> IndustrialReach::config(const Options& options) {
  const double inf = INFINITY;
  const int64_t size = options.image_size;
  std::vector<std::pair<std::string, rlmesh::Space>> fields;
  auto add = [&](const char* key, rlmesh::Result<rlmesh::Space> space) -> rlmesh::Status {
    if (!space) return space.error();
    fields.emplace_back(key, space.unwrap());
    return rlmesh::ok();
  };
  for (rlmesh::Status status : {
           add("image", rlmesh::Space::box<uint8_t>({size, size, 3}, 0, 255)),
           add("joint_pos", rlmesh::Space::box<float>({6}, -inf, inf)),
           add("joint_vel", rlmesh::Space::box<float>({6}, -inf, inf)),
           add("joint_torque", rlmesh::Space::box<float>({6}, -inf, inf)),
           add("eef_pos", rlmesh::Space::box<float>({3}, -inf, inf)),
           add("target_pos", rlmesh::Space::box<float>({3}, -inf, inf)),
           add("instruction", rlmesh::Space::text(1, 128)),
       }) {
    if (!status) return status.error();
  }
  auto observation = rlmesh::Space::dict(std::move(fields));
  if (!observation) return observation.error();
  auto action = rlmesh::Space::box<float>({3}, -1.0, 1.0);
  if (!action) return action.error();

  rlmesh::EnvConfig config;
  config.id = "ChronoIndustrialReach-v0";
  config.observation_space = observation.unwrap();
  config.action_space = action.unwrap();
  config.adapter_tags_json = kTags;
  config.reset_options = {"trial_index"};
  config.render_mode = "rgb_array";
  std::ostringstream metadata;
  metadata << R"({"simulator": "Project Chrono )" << CHRONO_VERSION
           << R"(", "robot": "IndustrialRobot6dof", "max_episode_steps": )" << options.max_steps
           << "}";
  config.metadata_json = metadata.str();
  return config;
}

}  // namespace chrono_reach
