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

## The sketch

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
and refused chunks, and heartbeat echoes; `robot.lastRefusal()` says why the
last chunk was refused. `examples/Timing` measures the crypto costs on the
board; build sizes and a host test do not establish real-board timing.

The library runs on Arduino's loop task and sets that task's stack to 12 KB
in place of the default 8 KB, because checking one host signature alone needs
8.6 KB. A sketch needs no setting for this. `SET_LOOP_TASK_STACK_SIZE` in a
sketch replaces the library's value, so keep it at 12 KB or more, and add what
the sketch's own `loop()` needs below `robot.poll()`. Call `robot.begin()` and
`robot.poll()` from the loop task, or from a task with a stack at least as big.

An optional normally closed e-stop button in `Robot` latches the board's
stop. The example clears it only after BOOT is held for two seconds with
the e-stop released. Disarm and arm the session before driving again.
A controller's e-stop also reaches the board through the session transport.

The board tries its stored home addresses in order, authenticating the host
by the key it was provisioned with. Four consecutive TCP ports from the
host's base port must be reachable. When replies stop for five seconds the
board closes its links and joins again, backing off to one attempt every 30
seconds for as long as it runs: a host restart or a Wi-Fi drop never needs a
reset.

## Keep the loop on time

Commands that arrive while the loop is away from `robot.poll()` wait, and a
chunk that waits past its life is lost. Three settings decide how long the
loop is away.

**Give `Serial` a transmit buffer.** Arduino-ESP32 gives `Serial` none by
default: a print waits for the port, 87 microseconds a character at 115200
baud, and the loop waits with it. A status report of 900 characters holds the
loop for about 75 ms, longer than a chunk lives at 30 Hz with four actions.
Set a buffer before `Serial.begin()`, as the `Robot` example does:

```cpp
Serial.setTxBufferSize(2048);
Serial.begin(115200);
```

**Turn Wi-Fi power save off.** In power save (ESP-IDF's modem sleep) a
board's frames wait for the access point's next beacon, about 100 ms. Call
`WiFi.setSleep(false)` after `WiFi.mode(WIFI_STA)`, before joining.
Arduino-ESP32 3.x keeps the setting and applies it again each time the
station starts. It is on by default when nothing sets it, ESP-IDF needs it on
while Bluetooth runs beside Wi-Fi, and anything else that calls
`esp_wifi_set_ps` turns it back on. The library changes no radio setting.

**Choose the network tasks' core, if you need to.** The transport's tasks
(signature checks, sends, the datagram socket's reads) and zenoh-pico's run
on whichever core the scheduler picks. That is the default, `-1`. Build the
library with `RLMESH_TASK_CORE=0 mise run build:esp32` (or `1`) to pin them
all to one core instead; any other value is refused. Core 0 is where Wi-Fi
and lwIP run, and every send runs lwIP inline in the sending task.

## UDP for the realtime parts

`rlmesh host` offers its datagram path in the handshake by default
(`--board-data udp`). The board then takes its parts' action chunks, and
sends their states and `_applied` entries, over UDP to the host's base port
plus one (UDP 7448 by default). That port must be reachable from the board
for the faster path: open it in the host's firewall. When it is not, the
host moves the board to TCP, as below. Everything else stays on TCP. A lost Wi-Fi frame then costs that one message instead of
stalling the stream behind it. Datagrams are sealed and replay-checked as the
session's messages are, and the board probes the host about once a second so
the host follows it to a new address.

The host decides when the path is broken: with the board's heartbeats
arriving, one end sent datagrams for two seconds (at least three) and the
other opened none. It then moves the board's realtime parts back to TCP, and
onto UDP again, without a new attach, once the board's probes arrive and its
retry time has come (5 s, doubling up to 2 minutes while the path keeps
breaking). `rlmesh host --board-data udp-only` never moves a board back: a
setting for measuring the path, where loss is simply loss.

`robot.useDatagram(false)`, after `begin()` and before the first `poll()`,
keeps a board on TCP whatever the host offers; the default follows the host.
Restart the host with `--board-data tcp` to put every board back on TCP:
they switch as they re-attach, with no reflash. A board built from a library
before this path takes no part in it and stays on TCP.

## Reading the board's reports

The `Robot` example prints these every 5 s while the robot runs, and
`printRefusals` and `printDropped` when their counts change. Each takes any
`Print` (`Serial`). The numbers below are an example.

`robot.printChunkTiming(Serial, part)`, for one commanded part this attach:

```text
  chunks received 1500 accepted 1497 lost 3 late max 41.2 ms gap max 118.0 ms, by ms: <1 1210 1-2 230 2-4 40 32-64 17
```

`lost` counts gaps in the chunks' sequence numbers. The buckets count
accepted chunks by how late they arrived against their first slot, in
milliseconds, and `late max` is the latest. `gap max` is the longest time
between two accepted chunks.

`robot.printDatagram(Serial)` prints three lines:

```text
  wifi power save off, channel 6 HT20, rssi -58, listen interval 3; on the beacon: sends 0.04 of 250, receives 0.06 of 250
  loop: longest between polls 12.4 ms, 0 over 50 ms
  realtime parts over udp: sent 3000, received 1500, lost 2, reordered 0, unopened 0, fallbacks 0, returns 0, on udp 300 s, on tcp 0 s
```

The first line is the radio. The two "on the beacon" figures say how closely
the board's sends, and the messages it received, lined up with the access
point's 102.4 ms beacon since the last print: 0 not at all, 1 every one, and
about 0.05 for unrelated timing. Received messages that line up, with power
save off, mean the access point holds frames for the board as if it slept;
sends that line up mean the board's own send path waits on the radio.

The second line is the loop since the last print: the longest time between
two calls to `poll()`, and how many were over 50 ms. Anything near a chunk's
life here is the sketch's own work, usually unbuffered prints.

The third line is the datagram path this attach: datagrams sent and received,
received ones lost or out of order, ones that did not open, the host's moves
back to TCP (`fallbacks`) and onto UDP again (`returns`), and the seconds on
each. It reads `realtime parts over tcp` when the host offered no path.

`robot.printDropped(Serial)` says what the board did not take or send, over
every attach:

```text
  dropped: views too_large 1; unsent 0, send queue replaced 12 dropped 0
  part 0 published: state 1500 of 1500 at every sample, applied 1500 of 1500 at every sample
```

The first line counts what it received and dropped, by what (`views`,
`replies`, `chunks`, `other`) and why (`unopened`, `replayed`, `malformed`,
`too_large`, `no_memory`, `stale`), or `none`. `unsent` counts heartbeats and
samples it skipped because the heap had no room. `send queue replaced` counts
samples a newer one on the same key replaced while the link was slower than
the board publishes, and `dropped` publications dropped from a full send
queue. Each part's line gives the state and `_applied` samples sent of those
offered, and the rate its delivery plan allows.

`robot.printRefusals(Serial)` counts refused chunks by reason, this attach
and then over every attach, as
`  refused 4 chunks, last stale_token: stale_token 4; 9 over 3 attaches: stale_token 7 expired 2`.

The host sees the same figures from its side: `host/health` (the cockpit,
recordings) carries the board's counters and its own end's, and `rlmesh
host`'s health log and Python's `Host.health()` the seconds on each path and
the moves.

## Building and verifying from source

```sh
mise run setup:esp32
mise run dev:heavy -- mise run build:esp32
mise run dev:heavy -- mise run test:esp32:host
```

`setup:esp32` installs pinned Arduino CLI, the Arduino core, Espressif's Rust
fork, and nightly Cargo. `build:zenoh-pico` fetches zenoh-pico 1.10.1, the
version of the host's Zenoh, into `target/zenoh-pico`; the build patches it
(`transport/patch_zenoh_pico.py`) and stops on any other version. The C6 uses upstream Rust; classic ESP32 and S3 use
Espressif's Xtensa targets. Without the fork, local builds explicitly skip
Xtensa; CI and releases set `RLMESH_ESP32_REQUIRE_XTENSA=1` to fail instead.
Other chips are unsupported; the C3 lacks the atomics the runtime needs.

Pass `ARDUINO_CLI=/path/to/arduino-cli` to use another Arduino CLI, and
`ESP_EXPORT=/path/to/export-esp.sh` for a nonstandard toolchain installation.
The task stages archives in `src/<chip>/`, compiles both examples without
transport flags, and writes installable ZIPs in `target/esp32/dist/`.
It checks that private transport symbols are local and firmware `memcpy`
comes from ESP-IDF's IRAM or ROM, and `test/check_stack.py` holds the firmware's
stack frames and deepest call chains to stated budgets. ZIPs include
zenoh-pico's license and notice.

`test:esp32:host` compiles the same C transport on Linux. A real loopback
session host and controller send two distinct action targets; the board loop
asserts their values, applies them, and publishes them back for the controller
to verify. It also checks joining, heartbeat echoes, and transport failures.
No hardware is needed. For a board without Wi-Fi, see [RLMeshTether](../arduino).
