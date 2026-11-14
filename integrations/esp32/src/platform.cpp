// The ESP32's clock and randomness, for the Rust core's callbacks.

#include <Arduino.h>
#include <esp_random.h>
#include <esp_timer.h>

#include "RLMesh.h"

// The loop task's stack, in place of Arduino's 8 KB: checking one host
// signature takes 8.6 KB. Weak, so a sketch's own SET_LOOP_TASK_STACK_SIZE
// still wins. test/check_stack.py holds this size against the firmware.
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
