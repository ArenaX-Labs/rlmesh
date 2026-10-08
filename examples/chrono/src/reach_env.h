// IndustrialReach: Chrono's 6-DOF industrial robot reaching a target, served as
// an RLMesh environment.
#pragma once

#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <rlmesh.hpp>
#include <string>
#include <vector>

#include "chrono/physics/ChSystemNSC.h"
#include "chrono_models/robot/industrial/IndustrialKinematics6dofSpherical.h"
#include "chrono_models/robot/industrial/IndustrialRobot6dof.h"

namespace chrono_reach {

struct Options {
  int image_size = 256;       // observation camera, square
  int render_size = 480;      // render() frames for viewers, square
  int max_steps = 120;        // the env truncates an episode here
  double control_dt = 0.05;   // seconds of simulation per env step
  double physics_dt = 0.005;  // Chrono integration step
  double max_delta = 0.02;    // metres the TCP command moves per unit action
  double success_radius = 0.025;
};

// One lane of the env: owns a Chrono system for the current episode.
class IndustrialReach : public rlmesh::Environment {
 public:
  explicit IndustrialReach(Options options);

  rlmesh::Result<rlmesh::ResetOutput> reset(const rlmesh::ResetArgs& args) override;
  rlmesh::Result<rlmesh::StepOutput> step(std::optional<rlmesh::ValueRef> action) override;
  rlmesh::Result<std::optional<rlmesh::Value>> render() override;
  void close() override;

  // The spaces, adapter tags and reset options this env serves with.
  static rlmesh::Result<rlmesh::EnvConfig> config(const Options& options);

 private:
  void build_world();
  chrono::ChVectorDynamic<> solve_ik(const chrono::ChVector3d& tcp) const;
  bool reachable(const chrono::ChVector3d& tcp) const;
  chrono::ChVector3d tcp_position() const;
  double distance() const;
  rlmesh::Result<rlmesh::Value> observe() const;
  std::vector<uint8_t> camera(int size) const;

  Options options_;
  std::unique_ptr<chrono::ChSystemNSC> system_;
  std::shared_ptr<chrono::industrial::IndustrialRobot6dof> robot_;
  std::unique_ptr<chrono::industrial::IndustrialKinematics6dofSpherical> kinematics_;
  chrono::ChQuaterniond tool_rotation_;
  chrono::ChVector3d home_;
  chrono::ChVector3d command_;
  chrono::ChVector3d target_;
  std::deque<chrono::ChVector3d> trail_;
  std::mt19937_64 rng_{0};
  int steps_ = 0;
  int episodes_ = 0;
  bool success_ = false;
};

}  // namespace chrono_reach
