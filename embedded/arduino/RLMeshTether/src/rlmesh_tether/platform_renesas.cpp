// Renesas RA4M1 (UNO R4 Minima and WiFi): the core's WDT, the e-stop latch
// in the last 1 KB block of data flash, and a boot value in RAM that
// survives warm resets.
//
// Erased data flash on the RA reads back undefined, so "clear" is the
// flash's blank check, not a value read (§6.4). The block is the last of
// the 8 KB the core's EEPROM emulation also uses: a sketch with an
// `estopPin` shouldn't use EEPROM.h.

#if defined(ARDUINO_ARCH_RENESAS_UNO)

#include <Arduino.h>
#include <WDT.h>
#include <r_flash_lp.h>

#include "platform.h"

namespace {

const uint32_t kLatch = BSP_FEATURE_FLASH_DATA_FLASH_START + 0x1C00;

// The previous boot's value while the RAM survives (a watchdog, software or
// pin reset); at power-on the magic is gone and the value is the RAM's
// noise.
struct BootRecord {
  uint32_t magic;
  uint16_t value;
};
BootRecord record __attribute__((section(".noinit")));
const uint32_t kMagic = 0x524C4D54;  // "RLMT"

uint8_t cause = 0xFF;

// One flash operation: open, run, close, as the core's own drivers do, so
// the data flash isn't held open between them.
template <class F>
fsp_err_t withFlash(F op) {
  flash_lp_instance_ctrl_t ctrl = {};
  flash_cfg_t cfg = {};
  cfg.data_flash_bgo = false;
  cfg.ipl = BSP_IRQ_DISABLED;
  cfg.irq = FSP_INVALID_VECTOR;
  cfg.err_ipl = BSP_IRQ_DISABLED;
  cfg.err_irq = FSP_INVALID_VECTOR;
  fsp_err_t err = R_FLASH_LP_Open(&ctrl, &cfg);
  if (err != FSP_SUCCESS) return err;
  err = op(&ctrl);
  R_FLASH_LP_Close(&ctrl);
  return err;
}

}  // namespace

namespace rlmesh {
namespace tether {
namespace platform {

void watchdogArm() { WDT.begin(kWatchdogMs); }

void watchdogFeed() { WDT.refresh(); }

// Blank is clear; anything else, a failed check included, is latched.
bool latchLoad() {
  flash_result_t result = FLASH_RESULT_NOT_BLANK;
  fsp_err_t err = withFlash(
      [&](flash_lp_instance_ctrl_t* c) { return R_FLASH_LP_BlankCheck(c, kLatch, 1, &result); });
  return err != FSP_SUCCESS || result != FLASH_RESULT_BLANK;
}

void latchStore() {
  // From RAM: the flash sequencer reads the source while it programs.
  uint8_t zero = 0;
  withFlash([&](flash_lp_instance_ctrl_t* c) {
    return R_FLASH_LP_Write(c, reinterpret_cast<uint32_t>(&zero), kLatch, 1);
  });
}

void latchClear() {
  withFlash([](flash_lp_instance_ctrl_t* c) { return R_FLASH_LP_Erase(c, kLatch, 1); });
}

uint16_t boot() {
  if (record.magic == kMagic) {
    record.value = uint16_t(record.value + 1);
  } else {
    // Power-on: SRAM noise, mixed with how long the core took to get here.
    uint32_t x = record.magic ^ (uint32_t(record.value) << 16) ^ micros();
    x ^= x >> 16;
    x *= 0x45D9F3Bu;
    x ^= x >> 16;
    record.magic = kMagic;
    record.value = uint16_t(x);
  }
  return record.value;
}

uint8_t resetCause() {
  if (cause != 0xFF) return cause;
  uint16_t r1 = R_SYSTEM->RSTSR1;
  uint8_t r0 = R_SYSTEM->RSTSR0;
  bool warm = R_SYSTEM->RSTSR2_b.CWSF;
  if (r1 & (R_SYSTEM_RSTSR1_IWDTRF_Msk | R_SYSTEM_RSTSR1_WDTRF_Msk)) {
    cause = 3;
  } else if (!warm || (r0 & R_SYSTEM_RSTSR0_PORF_Msk)) {
    cause = 0;
  } else if (r0 & (R_SYSTEM_RSTSR0_LVD0RF_Msk | R_SYSTEM_RSTSR0_LVD1RF_Msk |
                   R_SYSTEM_RSTSR0_LVD2RF_Msk)) {
    cause = 2;
  } else if (r1 == 0 && r0 == 0) {
    cause = 1;  // the RES pin sets no flag of its own
  } else {
    cause = 4;
  }
  // Flags clear by writing 0 after reading 1; CWSF is set by writing 1.
  R_SYSTEM->RSTSR1 = 0;
  R_SYSTEM->RSTSR0 = 0;
  R_SYSTEM->RSTSR2_b.CWSF = 1;
  return cause;
}

uint8_t arch() { return 2; }

}  // namespace platform
}  // namespace tether
}  // namespace rlmesh

#endif  // ARDUINO_ARCH_RENESAS_UNO
