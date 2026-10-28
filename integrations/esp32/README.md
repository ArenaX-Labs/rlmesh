# RLMeshESP32

An ESP32 on Wi-Fi joins an RLMesh session as a robot, with no computer in
between. The session's protocol and crypto are RLMesh's Rust core
(`crates/rlmesh-device`), linked into the sketch as a precompiled
library. This library adds the board's clock and randomness, and its
network: [zenoh-pico](https://github.com/eclipse-zenoh/zenoh-pico) at the
version of the host's Zenoh.

> **Work in progress.** A board joins a session, declares its parts and
> heartbeats (`examples/Robot`), all sealed. Commands and samples come
> next.

For a board without a network (a Nano, Uno or UNO R4 on a USB cable), see
[RLMeshTether](../arduino).

## Boards

On the Espressif Arduino core 3.x:

| Board                                  | Rust toolchain                            |
| -------------------------------------- | ----------------------------------------- |
| ESP32-C6 (RISC-V)                      | Upstream stable. CI checks that it builds |
| Classic ESP32 (ESP32-WROOM-32, Xtensa) | Espressif's fork. Built locally only      |

Any other board fails to build. The ESP32-C3 lacks the atomics the core needs. The ESP32-S3 would need its own build.

## Building

The precompiled library isn't checked in. `mise run build:esp32` builds
it for each board and stages it into `src/`. It needs the `esp32:esp32`
Arduino core for that core's binutils. It also fetches zenoh-pico and
assembles it as an Arduino library in `target/zenoh-pico/arduino`, with its
platform defines passed to the compile (`-DZENOH_ARDUINO_ESP32
-DZENOH_COMPILER_GCC -DZENOH_C_STANDARD=11`). The classic ESP32 also needs
Espressif's Rust toolchain (`espup install`) and a nightly toolchain
(`rustup toolchain install nightly`); without them it's skipped.

```sh
mise run build:esp32
```

It also compiles each example with `arduino-cli` when that's available
(pass `ARDUINO_CLI` for a binary off `PATH`). The firmware keeps its own
`memcpy`, `memset` and math in IRAM or ROM. The task fails if the archive
exports any C name but its own, or if the linked `memcpy` runs from flash.

## Joining a session

On the computer, run a host on its LAN address:

```sh
rlmesh host --listen 192.168.1.20:7447
```

It prints the session's address with its key (`192.168.1.20:7447#…`). Put
that, your Wi-Fi and the robot's map in `examples/Robot`, flash it, and
open the Serial Monitor at 115200:

```cpp
rlmesh::Robot robot;

void setup() {
  WiFi.begin(ssid, password);  // first: the RNG needs the radio on
  // ... wait for WL_CONNECTED
  robot.begin(address, "eyes", robotJson);
}

void loop() {
  robot.poll();  // every loop
}
```

- **Joining:**
  - the board reads the host's signed state;
  - it runs the handshake with the address's key;
  - it attaches and declares its parts, sealed;
  - it heartbeats.

  `robot.status()` says where it is, and why it last started over.

- **Starting over:** any failure (no reply, a refusal, no heartbeat echo for 3 s) starts over from the handshake after a second.
- **The address's key is a secret.** Anyone holding it can join the session.
- **Ports:** the board connects to the host's first port only; the host listens on four.

The Rust side of this (`crates/rlmesh-device`'s `robot`) is tested against
a real host. `mise run test:esp32:host` runs this library's transport
(`src/rlmesh_zenoh.c`) on Linux against `rlmesh host`, as the board runs
it.

## Timing

`examples/Timing` prints the mean cost of each operation a robot runs:

| Operation                               | When                  |
| --------------------------------------- | --------------------- |
| Handshake msg 1 and msg 2               | Each attach           |
| Verify a signed authority state (1 KiB) | Each authority change |
| Open a chunk (256 B), signed or sealed  | Each chunk            |
| Seal a sample (64 B), signed            | Each sample           |

It then sums one 100 Hz control tick (a chunk and a sample) as a share of
a core. Flash it, and open the Serial Monitor at 115200.

## Notes

- **Randomness.** The ESP32's hardware RNG is cryptographic only with the
  radio on. Start Wi-Fi before anything draws keys.
- **Flash.** `examples/Robot` is about 1.3 MB (1.4 MB on the C6), past the
  default app partition. Choose a larger one (Tools → Partition Scheme,
  "Huge APP"). The crypto alone adds about 205 KB to a Wi-Fi sketch (189 KB
  on the C6).
- **Panics.** A fault in the Rust core calls `abort()`, so the board
  restarts, which stops its PWM outputs.
