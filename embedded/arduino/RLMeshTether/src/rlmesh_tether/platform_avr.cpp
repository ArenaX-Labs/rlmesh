// Classic AVR (ATmega328P, 32U4, 2560): the watchdog, and the last five
// bytes of EEPROM: four for the boot counter, then the e-stop latch.

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

// MCUSR at reset, and what Optiboot 8 passes in r2 once it has cleared
// MCUSR. Optiboot 4.4, the Uno's and the Nano's stock bootloader, clears it
// and passes nothing, so the cause is unknown there. (Marking a watchdog
// reset from the WDT interrupt would need interrupt-then-reset mode, which
// never resets a sketch stuck with interrupts off.)
uint8_t savedMcusr __attribute__((section(".noinit")));
uint8_t savedR2 __attribute__((section(".noinit")));

// Programs one EEPROM byte in `mode` (EEPM bits): write-only can only clear
// bits and erase-only only sets them, so neither has an erased moment that
// a brown-out could leave behind.
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
  // Back to atomic erase-and-write, which avr-libc's writes assume.
  EECR = 0;
}

}  // namespace

// Before anything else runs: r2 is still the bootloader's.
extern "C" void rlmeshTetherSaveR2() __attribute__((naked, used, section(".init0")));
void rlmeshTetherSaveR2() { __asm__ __volatile__("sts %0, r2\n" : "=m"(savedR2) :); }

// After a watchdog reset the watchdog stays on at its shortest period, so it
// is stopped before the core's setup can outlast it.
extern "C" void rlmeshTetherInit3() __attribute__((naked, used, section(".init3")));
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

// Erased is 0xFF; anything else, a torn write included, is latched.
bool latchLoad() { return eeprom_read_byte(kLatch) != 0xFF; }

void latchStore() { program(kLatch, 0x00, _BV(EEPM1)); }

void latchClear() { program(kLatch, 0xFF, _BV(EEPM0)); }

// The four counter bytes' sum, after incrementing the byte the sum picks, so
// the writes rotate. Each boot changes the sum (by +1, or by -255 when a
// byte wraps).
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

// MCUSR when no bootloader cleared it, else r2 from Optiboot 8 or later,
// which keeps its version in the last word of flash (major in the high
// byte); otherwise unknown ("other").
uint8_t resetCause() {
  uint8_t f = savedMcusr;
  uint8_t major = pgm_read_byte(FLASHEND);
  if (f == 0 && major >= 8 && major != 0xFF) f = savedR2;
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

#endif  // ARDUINO_ARCH_AVR
