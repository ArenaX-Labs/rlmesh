#include <Arduino.h>
#include <esp_random.h>
#include <esp_timer.h>

#include "RLMesh.h"

__attribute__((weak)) size_t getArduinoLoopTaskStackSize() { return 12 * 1024; }

namespace rlmesh {
namespace esp32 {

uint64_t nowUs() { return uint64_t(esp_timer_get_time()); }

int fillRandom(uint8_t* buf, size_t len) {
  esp_fill_random(buf, len);
  return 0;
}

}  // namespace esp32
}  // namespace rlmesh
