// The ESP32's clock and randomness, for the Rust core's callbacks.

#include <esp_random.h>
#include <esp_timer.h>

#include "RLMeshESP32.h"

namespace rlmesh {
namespace esp32 {

uint64_t nowUs() { return uint64_t(esp_timer_get_time()); }

int fillRandom(uint8_t* buf, size_t len) {
  esp_fill_random(buf, len);
  return 0;
}

}  // namespace esp32
}  // namespace rlmesh
