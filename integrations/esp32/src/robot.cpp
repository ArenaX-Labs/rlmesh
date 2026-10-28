// rlmesh::Robot: the device robot over zenoh-pico, on the board's clock and
// randomness.

#include <stdlib.h>

#include "RLMeshESP32.h"
#include "rlmesh_zenoh.h"

namespace rlmesh {

bool Robot::begin(const char* address, const char* name, const char* robotJson,
                  uint32_t heartbeatMs) {
  end();
  rlmesh_device_robot_config_t config = {address, name, robotJson, heartbeatMs, &esp32::fillRandom};
  if (rlmesh_device_robot_new(&config, &robot_) != RLMESH_DEVICE_OK) {
    robot_ = nullptr;
    return false;
  }
  zenoh_ = static_cast<rlmesh_zenoh_t*>(malloc(sizeof(rlmesh_zenoh_t)));
  if (zenoh_ == nullptr || rlmesh_zenoh_open(zenoh_, robot_, &esp32::nowUs) != 0) {
    free(zenoh_);
    zenoh_ = nullptr;
    end();
    return false;
  }
  return true;
}

void Robot::poll() {
  if (zenoh_ != nullptr) rlmesh_zenoh_poll(zenoh_);
}

void Robot::end() {
  if (zenoh_ != nullptr) {
    rlmesh_zenoh_close(zenoh_);
    free(zenoh_);
    zenoh_ = nullptr;
  }
  rlmesh_device_robot_free(robot_);
  robot_ = nullptr;
}

int Robot::part(const char* path) const {
  return robot_ != nullptr ? rlmesh_device_robot_part(robot_, path) : -1;
}

bool Robot::tick(int part, rlmesh_device_tick_t& out) {
  if (robot_ == nullptr || part < 0 || part >= RLMESH_DEVICE_NO_PART) return false;
  return rlmesh_device_robot_tick(robot_, uint8_t(part), esp32::nowUs(), &out) == 1;
}

void Robot::handoverDone(int part) {
  if (robot_ != nullptr && part >= 0 && part < RLMESH_DEVICE_NO_PART) {
    rlmesh_device_robot_handover_done(robot_, uint8_t(part));
  }
}

int Robot::publish(int part, const void* value, size_t len) {
  if (robot_ == nullptr || part < 0 || part >= RLMESH_DEVICE_NO_PART) {
    return RLMESH_DEVICE_ERR_ARG;
  }
  uint64_t now = esp32::nowUs();
  return rlmesh_device_robot_publish(robot_, uint8_t(part), now, static_cast<const uint8_t*>(value),
                                     len, now);
}

rlmesh_device_robot_status_t Robot::status() const {
  rlmesh_device_robot_status_t s = {};
  if (robot_ != nullptr) rlmesh_device_robot_status(robot_, &s);
  return s;
}

uint32_t Robot::transportFailures() const { return zenoh_ != nullptr ? zenoh_->failed : 0; }

}  // namespace rlmesh
