// RLMeshESP32: an ESP32 on Wi-Fi as an RLMesh robot. The session's crypto
// and the robot edge are RLMesh's Rust core, linked as a precompiled
// library (`rlmesh_device.h`); this wraps it with the board's clock and
// randomness.

#ifndef RLMESH_ESP32_H_
#define RLMESH_ESP32_H_

#include <sdkconfig.h>

#if !defined(CONFIG_IDF_TARGET_ESP32) && !defined(CONFIG_IDF_TARGET_ESP32C6)
#error "RLMeshESP32 runs on the classic ESP32 (ESP32-WROOM-32) and the ESP32-C6 so far"
#endif

#include "rlmesh_device.h"

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
}  // namespace rlmesh

#endif  // RLMESH_ESP32_H_
