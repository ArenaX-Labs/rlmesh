// What the board runtime needs from the MCU: its watchdog, the e-stop
// latch's raw storage, a per-boot value and the reset cause (spec §6.3–6.4).
// One implementation per architecture (`platform_avr.cpp`,
// `platform_renesas.cpp`); any other fails to build instead of running
// without a watchdog.

#ifndef RLMESH_TETHER_PLATFORM_H_
#define RLMESH_TETHER_PLATFORM_H_

#include <stdint.h>

#if !(defined(ARDUINO_ARCH_AVR) && defined(EEPM1)) && !defined(ARDUINO_ARCH_RENESAS_UNO)
#error "RLMeshTether runs on AVR (Uno, Nano, Mega, Leonardo) and the UNO R4 (Minima, WiFi) so far"
#endif

namespace rlmesh {
namespace tether {
namespace platform {

// The hardware watchdog's period: a sketch that doesn't reach poll() or
// tether.delay() for this long resets the MCU.
const uint16_t kWatchdogMs = 500;

// Starts the hardware watchdog (once), and feeds it.
void watchdogArm();
void watchdogFeed();

// The stored pin latch: set, or unreadable. Raw program-only writes, never
// the core's EEPROM API (§6.4).
bool latchLoad();
void latchStore();
void latchClear();

// Differs from the previous boot's value.
uint16_t boot();
// `Status::reset`: 0 power-on, 1 external, 2 brown-out, 3 watchdog, 4 other.
uint8_t resetCause();
// `Hello::arch`.
uint8_t arch();

}  // namespace platform
}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_PLATFORM_H_
