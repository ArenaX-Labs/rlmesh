# RLMesh for ESP32

Two servos, Wi-Fi, and a policy on your PC. Install **one Arduino library**,
include `<RLMesh.h>`, and use the `rlmesh::` C++ API. The library contains
RLMesh's Rust device runtime and a private copy of
[zenoh-pico](https://github.com/eclipse-zenoh/zenoh-pico); no second library
or `-DZENOH_*` flags are needed.

## Board quickstart

1. Install Espressif's **esp32** Arduino core 3.3.12 and build the library
   ZIP for your chip. The 0.2 alpha publishes no prebuilt board library, so
   it is built from this repository: `mise run setup:esp32`, then
   `mise run build:esp32` writes, under `target/esp32/dist/`, `RLMesh-esp32.zip` (classic ESP32 /
   WROOM-32), `RLMesh-esp32s3.zip` and `RLMesh-esp32c6.zip` (see
   [Building and verifying from source](#building-and-verifying-from-source) for the toolchains it
   needs). In Arduino IDE, choose **Sketch → Include Library → Add .ZIP
   Library**. Replace an older RLMesh installation when changing chips.
2. Open **File → Examples → RLMesh → Robot**. Set the Wi-Fi credentials,
   select your board and **Minimal SPIFFS** partition scheme, and upload.
   Connect pan to GPIO 18 and tilt to GPIO 19, adjusting pins for your board.
   Power the servos separately from the board's 3.3 V supply, with common ground.
3. Start `rlmesh host` on your PC. With the board running and Serial Monitor
   closed, provision it once over USB:

   ```sh
   rlmesh devices provision --port /dev/ttyUSB0 --name eye-3
   ```

4. The board remembers its device key and home in NVS and joins the host's
   session with its own key after every reboot. Run your policy on the PC
   (`Policy.bind()` with no robot drives every eye at once); `Robot` applies
   pan/tilt targets at 50 Hz.

[Real-time sessions](../../docs/realtime.md) walks through the whole setup: the host, two eyes, a camera and one policy driving both.

Every eye runs the same firmware. USB provisioning assigns its robot name
(`eye-3`, `eye-4`, …), stored with the home's key, id and addresses in NVS.
Names are 1–128 bytes of `[A-Za-z0-9._-]`; the whole record must fit 252 bytes.
Provision again with a new `--name` to rename a board: it keeps its device key,
and the host moves its pin if the board still owns the old name.
Old provisioning records are refused: provision again after upgrading.

The sketch must run before the first provisioning exchange. Later uploads
keep NVS unless you explicitly erase all flash. A full erase makes a new device
key; revoke a lost board with `rlmesh devices revoke`.

## Flash layout

Each example includes `partitions.csv` for 4 MB flash: two **1,966,080-byte**
app slots, 128 KiB SPIFFS, NVS, OTA metadata, and a core-dump partition.
The spare app slot is retained; this library adds no OTA updater. Select
**Minimal SPIFFS** so Arduino's size check agrees with the bundled layout.
Keep `partitions.csv` when copying an example into another sketch folder.
`build:esp32` prints measured sizes and explains a partition overflow.

## C++ loop

After Wi-Fi starts, call `provisioning.begin()` and then
`robot.begin(provisioning, kMap)`. The stored record supplies the robot name;
no name is compiled into the provisioned path.

```cpp
#include <RLMesh.h>

rlmesh::Robot robot;
// After Wi-Fi starts, begin provisioning and the robot as in examples/Robot.
// Call robot.poll() every loop; at the part's control rate:
void controlTick(int eyes) {
  rlmesh_device_tick_t tick;
  switch (robot.tick(eyes, tick)) {
    case rlmesh::TickResult::Ready:
      // tick.has_action: dispatch tick.value in the part's dense layout.
      // tick.safe: run the declared safe behaviour; Robot declares "hold".
      // tick.handover: switch control modes, then call handoverDone(eyes).
      break;
    case rlmesh::TickResult::Idle:
    case rlmesh::TickResult::NotReady:
    case rlmesh::TickResult::InvalidArgument:
      break;
  }
}
```

`publish()` returns `RLMESH_DEVICE_ERR_NOT_READY` until the robot runs and its
stream key arrives. `robot.status()` reports the phase, fault, mode, accepted
chunks, and heartbeat echoes. `examples/Timing` measures the crypto costs
on the board; build sizes and a host test do not establish real-board timing.

An optional normally closed e-stop button in `Robot` latches the board's
stop. The example clears it only after BOOT is held for two seconds with
the e-stop released. Disarm and arm the session before driving again.
A controller's e-stop also reaches the board through the session transport.

The board tries its stored home addresses in order, authenticating the host
by the key it was provisioned with. Four consecutive TCP ports from the
host's base port must be reachable.

## Building and verifying from source

```sh
mise run setup:esp32
mise run dev:heavy -- mise exec arduino-cli@1.5.1 -- mise run build:esp32
mise run dev:heavy -- mise run test:esp32:host
```

`setup:esp32` installs pinned Arduino CLI, the Arduino core, Espressif's Rust
fork, and nightly Cargo. The C6 uses upstream Rust; classic ESP32 and S3 use
Espressif's Xtensa targets. Without the fork, local builds explicitly skip
Xtensa; CI and releases set `RLMESH_ESP32_REQUIRE_XTENSA=1` to fail instead.
Other chips are unsupported; the C3 lacks the atomics the runtime needs.

Pass `ARDUINO_CLI=/path/to/arduino-cli` if the binary is off PATH, and
`ESP_EXPORT=/path/to/export-esp.sh` for a nonstandard toolchain installation.
The task stages archives in `src/<chip>/`, compiles both examples without
transport flags, and writes installable ZIPs in `target/esp32/dist/`.
It checks that private transport symbols are local and firmware `memcpy`
comes from ESP-IDF's IRAM or ROM. ZIPs include zenoh-pico's license and notice.

`test:esp32:host` compiles the same C transport on Linux. A real loopback
session host and controller send two distinct action targets; the board loop
asserts their values, applies them, and publishes them back for the controller
to verify. It also checks joining, heartbeat echoes, and transport failures.
No hardware is needed. For a board without Wi-Fi, see [RLMeshTether](../arduino).
