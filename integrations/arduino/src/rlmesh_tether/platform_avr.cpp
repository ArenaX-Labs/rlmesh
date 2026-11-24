#if defined(ARDUINO_ARCH_AVR)

#include <avr/eeprom.h>
#include <avr/interrupt.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <avr/wdt.h>

#include "platform.h"

namespace {

uint8_t* const kLatch = reinterpret_cast<uint8_t*>(E2END);
uint8_t* const kBootCounter = reinterpret_cast<uint8_t*>(E2END - 4);

uint8_t savedMcusr __attribute__((section(".noinit")));
uint8_t savedR2 __attribute__((section(".noinit")));

// Programs one EEPROM byte in `mode` (EEPM bits): write-only can only clear bits and erase-only
// only sets them, so neither has an erased moment that a brown-out could leave behind.
void program(uint8_t* addr, uint8_t value, uint8_t mode) {
  eeprom_busy_wait();
  uint8_t sreg = SREG;
  cli();
  EEAR = reinterpret_cast<uintptr_t>(addr);
  EEDR = value;
  EECR = mode;
  EECR |= _BV(EEMPE);
  EECR |= _BV(EEPE);
  SREG = sreg;
  eeprom_busy_wait();
  EECR = 0;
}

}  // namespace

// Before anything else runs: r2 is still the bootloader's.
extern "C" void rlmeshTetherSaveR2() __attribute__((naked, used, section(".init0")));
void rlmeshTetherSaveR2() { __asm__ __volatile__("sts %0, r2\n" : "=m"(savedR2) :); }

extern "C" void rlmeshTetherInit3() __attribute__((naked, used, section(".init3")));
// After a watchdog reset the watchdog stays on at its shortest period, so it is stopped before the
// core's setup can outlast it.
void rlmeshTetherInit3() {
  savedMcusr = MCUSR;
  MCUSR = 0;
  wdt_disable();
}

namespace rlmesh {
namespace tether {
namespace platform {

static_assert(kWatchdogMs == 500, "WDTO_500MS");

void watchdogArm() { wdt_enable(WDTO_500MS); }

void watchdogFeed() { wdt_reset(); }

bool latchLoad() { return eeprom_read_byte(kLatch) != 0xFF; }

void latchStore() { program(kLatch, 0x00, _BV(EEPM1)); }

void latchClear() { program(kLatch, 0xFF, _BV(EEPM0)); }

uint16_t boot() {
  uint8_t c[4];
  uint16_t sum = 0;
  for (uint8_t i = 0; i < 4; ++i) {
    c[i] = eeprom_read_byte(kBootCounter + i);
    sum = uint16_t(sum + c[i]);
  }
  uint8_t i = sum & 3;
  uint8_t next = uint8_t(c[i] + 1);
  eeprom_update_byte(kBootCounter + i, next);
  return uint16_t(sum - c[i] + next);
}

uint8_t resetCause() {
  uint8_t f = savedMcusr;
#if FLASHEND <= 0xFFFF
  uint8_t major = pgm_read_byte(FLASHEND);
  if (f == 0 && major >= 8 && major != 0xFF) f = savedR2;
#endif
  if (f & _BV(WDRF)) return 3;
  if (f & _BV(PORF)) return 0;
  if (f & _BV(BORF)) return 2;
  if (f & _BV(EXTRF)) return 1;
  return 4;
}

uint8_t arch() { return 1; }

}  // namespace platform
}  // namespace tether
}  // namespace rlmesh

#endif
