// RLMeshESP32: an ESP32 on Wi-Fi as an RLMesh robot. The session's
// protocol and crypto are RLMesh's Rust core, linked as a precompiled
// library (`rlmesh_device.h`); this wraps it with the board's clock and
// randomness, and zenoh-pico (`rlmesh_zenoh.h`) for its network.

#ifndef RLMESH_ESP32_H_
#define RLMESH_ESP32_H_

#include <sdkconfig.h>

#if !defined(CONFIG_IDF_TARGET_ESP32) && !defined(CONFIG_IDF_TARGET_ESP32C6)
#error "RLMeshESP32 runs on the classic ESP32 (ESP32-WROOM-32) and the ESP32-C6 so far"
#endif

#include "rlmesh_device.h"

// The transport (rlmesh_zenoh.h): opaque here, so a sketch never sees
// zenoh-pico's headers.
struct rlmesh_zenoh_t;

namespace rlmesh {
namespace esp32 {

// The board's microsecond clock (esp_timer).
uint64_t nowUs();

// The hardware RNG. It is cryptographic only with the radio on (Wi-Fi or
// Bluetooth): start Wi-Fi before anything draws keys.
int fillRandom(uint8_t* buf, size_t len);

// What the session's crypto costs on this board (rlmesh_device_timing):
// about a second, blocking. Run it from setup(), never from a control loop.
inline int timing(rlmesh_device_timing_t& out) {
  return rlmesh_device_timing(&nowUs, &fillRandom, &out);
}

}  // namespace esp32

// The board as a robot in a session, over Wi-Fi. Start Wi-Fi first.
class Robot {
 public:
  Robot() = default;
  Robot(const Robot&) = delete;
  Robot& operator=(const Robot&) = delete;
  ~Robot() { end(); }

  // Joins the session at `address`, as `rlmesh host` prints it
  // ("192.168.1.20:7447#secret.hostkey", its default session), as robot
  // `name` with the map `robotJson` (authored JSON, kept by the caller
  // only for this call). Returns false on a bad address or no session to
  // the host; joining itself goes on in poll().
  bool begin(const char* address, const char* name, const char* robotJson,
             uint32_t heartbeatMs = 50);
  // Runs the robot: call it every loop().
  void poll();
  void end();

  rlmesh_device_robot_status_t status() const;
  bool running() const { return status().phase == 4; }
  // Actions the transport failed to carry out.
  uint32_t transportFailures() const;

 private:
  rlmesh_device_robot_t* robot_ = nullptr;
  struct rlmesh_zenoh_t* zenoh_ = nullptr;
};

}  // namespace rlmesh

#endif  // RLMESH_ESP32_H_
