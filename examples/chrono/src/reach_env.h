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
#include "renderer.h"

namespace chrono_reach {

struct Options {
  int image_size = 256;       // observation camera, square
  int render_size = 480;      // render() frames for viewers, square
  int max_steps = 120;        // the env truncates an episode here
  double control_dt = 0.05;   // seconds of simulation per env step
  double physics_dt = 0.005;  // Chrono integration step
  double max_delta = 0.02;    // metres the TCP command moves per unit action
  double success_radius = 0.025;
  std::string renderer = "raytrace";  // the camera backend: "raytrace" or "vulkan"
};

// One lane of the env: owns a Chrono system for the current episode.
class IndustrialReach {
 public:
  // `renderer` draws both the observation image and render() frames.
  IndustrialReach(Options options, std::unique_ptr<Renderer> renderer);

  rlmesh::ResetResult reset(std::optional<int64_t> seed,
                            const std::optional<rlmesh::InfoDict>& options);
  rlmesh::StepResult step(std::optional<rlmesh::Value> action);
  std::optional<rlmesh::Value> render();
  void close();

  // The spaces, adapter tags and reset options this env serves with.
  rlmesh::spaces::Space observation_space() const;
  rlmesh::spaces::Space action_space() const;
  std::optional<std::string> id() const { return "ChronoIndustrialReach-v0"; }
  std::optional<std::string> render_mode() const { return "rgb_array"; }
  std::optional<rlmesh::Metadata> metadata() const;
  static rlmesh::adapters::EnvTags tags();

 private:
  void build_world();
  chrono::ChVectorDynamic<> solve_ik(const chrono::ChVector3d& tcp) const;
  bool reachable(const chrono::ChVector3d& tcp) const;
  chrono::ChVector3d tcp_position() const;
  double distance() const;
  rlmesh::Value observe() const;
  rlmesh::Value camera(int size) const;

  Options options_;
  std::unique_ptr<Renderer> renderer_;
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
