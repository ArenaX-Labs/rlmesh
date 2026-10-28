// Renesas RA4M1 (UNO R4 Minima and WiFi): the core's WDT, the e-stop latch
// in the last 1 KB block of data flash, and a boot value in the VBATT
// backup registers, which survive every reset but a power cycle (the DFU
// bootloader that runs at each reset reuses all of the sketch's RAM).
//
// Erased data flash on the RA reads back undefined, so "clear" is the
// flash's blank check, not a value read (§6.4). The core's data-flash
// driver (EEPROM.h's emulation, KVStore, a filesystem) manages all 8 KB and
// may erase the block, so with it linked the latch reads as set and can't
// be cleared.

#if defined(ARDUINO_ARCH_RENESAS_UNO)

#include <Arduino.h>
#include <WDT.h>
#include <r_flash_lp.h>

#include "platform.h"

namespace {

const uint32_t kLatch = BSP_FEATURE_FLASH_DATA_FLASH_START + 0x1C00;

// The boot record: the last 6 backup registers, a magic then the value.
const uint16_t kRecord = 506;
const uint8_t kMagic[4] = {'R', 'L', 'M', 'T'};

// RAM noise, mixed into the value at power-on.
uint32_t noise __attribute__((section(".noinit")));

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

// The core's data-flash driver, if the sketch links it (EEPROM.h goes
// through it too): a weak reference to its accessor, at namespace scope so
// it has external linkage.
void rlmeshTetherDataFlash() __asm__("_ZN20DataFlashBlockDevice11getInstanceEv")
    __attribute__((weak));

namespace {

// Through a volatile pointer, so the compiler can't assume a declared
// function's address is never null.
bool eepromLinked() {
  void (*volatile f)() = &rlmeshTetherDataFlash;
  return f != nullptr;
}

}  // namespace

namespace rlmesh {
namespace tether {
namespace platform {

void watchdogArm() { WDT.begin(kWatchdogMs); }

void watchdogFeed() { WDT.refresh(); }

// Blank is clear; anything else, a failed check included, is latched.
bool latchLoad() {
  if (eepromLinked()) return true;
  flash_result_t result = FLASH_RESULT_NOT_BLANK;
  fsp_err_t err = withFlash(
      [&](flash_lp_instance_ctrl_t* c) { return R_FLASH_LP_BlankCheck(c, kLatch, 1, &result); });
  return err != FSP_SUCCESS || result != FLASH_RESULT_BLANK;
}

void latchStore() {
  if (eepromLinked()) return;
  // From RAM: the flash sequencer reads the source while it programs.
  uint8_t zero = 0;
  withFlash([&](flash_lp_instance_ctrl_t* c) {
    return R_FLASH_LP_Write(c, reinterpret_cast<uint32_t>(&zero), kLatch, 1);
  });
}

void latchClear() {
  if (eepromLinked()) return;
  withFlash([](flash_lp_instance_ctrl_t* c) { return R_FLASH_LP_Erase(c, kLatch, 1); });
}

// The backup registers are behind PRC1, as the core's own double-tap
// magic in VBTBKR[0..3] is: every access is inside the unlock.
uint16_t boot() {
  R_BSP_RegisterProtectDisable(BSP_REG_PROTECT_OM_LPC_BATT);
  volatile uint8_t* r = &R_SYSTEM->VBTBKR[kRecord];
  bool kept = r[0] == kMagic[0] && r[1] == kMagic[1] && r[2] == kMagic[2] && r[3] == kMagic[3];
  uint16_t value = uint16_t(r[4] | (r[5] << 8));
  if (kept) {
    value = uint16_t(value + 1);
  } else {
    // Power-on: the registers' and the RAM's noise, and how long the core
    // took to get here.
    uint32_t x = noise ^ (uint32_t(value) << 16) ^ micros();
    x ^= x >> 16;
    x *= 0x45D9F3Bu;
    x ^= x >> 16;
    value = uint16_t(x);
    for (uint8_t i = 0; i < 4; ++i) r[i] = kMagic[i];
  }
  r[4] = uint8_t(value);
  r[5] = uint8_t(value >> 8);
  R_BSP_RegisterProtectEnable(BSP_REG_PROTECT_OM_LPC_BATT);
  return value;
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
