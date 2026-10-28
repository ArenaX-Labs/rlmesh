# RLMeshESP32

An ESP32 on Wi-Fi joins an RLMesh session as a robot, with no computer in
between. The session's handshake, envelopes and robot edge are RLMesh's
Rust core (`crates/rlmesh-device`), linked into the sketch as a
precompiled library; this library wraps it with the board's clock and
randomness.

> **Work in progress.** So far the library measures what the session's
> crypto costs on the board (`examples/Timing`). Joining a session comes
> with the session's handshake and sealed traffic.

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
Arduino core for that core's binutils. The classic ESP32 also needs
Espressif's Rust toolchain (`espup install`) and a nightly toolchain
(`rustup toolchain install nightly`); without them it's skipped.

```sh
mise run build:esp32
```

It also compiles each example with `arduino-cli` when that's available
(pass `ARDUINO_CLI` for a binary off `PATH`). The firmware keeps its own
`memcpy`, `memset` and math in IRAM or ROM. The task fails if the archive
exports any C name but its own, or if the linked `memcpy` runs from flash.

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
- **Flash.** The crypto adds about 205 KB to a Wi-Fi sketch (189 KB on the C6). With the
  default partition table that leaves little room. Choose a larger app
  partition (Tools → Partition Scheme, such as "Huge APP").
- **Panics.** A fault in the Rust core calls `abort()`, so the board
  restarts, which stops its PWM outputs.
