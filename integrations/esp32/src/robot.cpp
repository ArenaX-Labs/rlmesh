// rlmesh::Robot: the device robot over zenoh-pico, on the board's clock and
// randomness.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RLMeshESP32.h"
#include "rlmesh_zenoh.h"

namespace rlmesh {

namespace {

// RFC 4648 base32, lowercase, unpadded: 52 characters for 32 bytes.
// Temporary: the address is parsed in Rust once session-core has it.
bool unbase32(const char* text, size_t len, uint8_t out[32]) {
  static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz234567";
  if (len != 52) return false;
  uint32_t acc = 0, bits = 0;
  size_t at = 0;
  for (size_t i = 0; i < len; ++i) {
    const char* p = text[i] == 0 ? nullptr : strchr(kAlphabet, text[i]);
    if (p == nullptr) return false;
    acc = (acc << 5) | uint32_t(p - kAlphabet);
    bits += 5;
    if (bits >= 8) {
      bits -= 8;
      out[at++] = uint8_t(acc >> bits);
    }
  }
  return at == 32;
}

}  // namespace

bool Robot::begin(const char* address, const char* name, const char* robotJson,
                  uint32_t heartbeatMs) {
  end();
  const char* hash = address ? strchr(address, '#') : nullptr;
  const char* dot = hash ? strchr(hash, '.') : nullptr;
  uint8_t secret[32], hostKey[32];
  if (dot == nullptr || !unbase32(hash + 1, size_t(dot - hash - 1), secret) ||
      !unbase32(dot + 1, strlen(dot + 1), hostKey)) {
    return false;
  }
  char connect[80];
  int n = snprintf(connect, sizeof connect, "tcp/%.*s", int(hash - address), address);
  if (n <= 0 || size_t(n) >= sizeof connect) return false;
  rlmesh_device_robot_config_t config = {
      "local", "default", name, secret, hostKey, robotJson, heartbeatMs, &esp32::fillRandom,
  };
  bool ok = rlmesh_device_robot_new(&config, &robot_) == RLMESH_DEVICE_OK;
  memset(secret, 0, sizeof secret);
  if (!ok) return false;
  zenoh_ = static_cast<rlmesh_zenoh_t*>(malloc(sizeof(rlmesh_zenoh_t)));
  if (zenoh_ == nullptr || rlmesh_zenoh_open(zenoh_, connect, robot_, &esp32::nowUs) != 0) {
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

uint32_t Robot::transportFailures() const { return zenoh_ != nullptr ? zenoh_->failed : 0; }

rlmesh_device_robot_status_t Robot::status() const {
  rlmesh_device_robot_status_t s = {};
  if (robot_ != nullptr) rlmesh_device_robot_status(robot_, &s);
  return s;
}

}  // namespace rlmesh
