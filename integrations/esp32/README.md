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

The classic ESP32 (ESP32-WROOM-32), on the Espressif Arduino core 3.x.
Any other board fails to build. The ESP32-S3 and the RISC-V parts (C3, C6)
need their own build of the library.

## Building

The precompiled library isn't checked in. Build it once with Espressif's
Rust toolchain (`espup install`), which stages it into `src/`:

```sh
mise run build:esp32
```

The task also compiles each example with `arduino-cli` (pass
`ARDUINO_CLI` for a binary off `PATH`) when the `esp32:esp32` core is
installed.

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
- **Flash.** The crypto adds about 230 KB to a Wi-Fi sketch. With the
  default partition table that leaves little room. Choose a larger app
  partition (Tools → Partition Scheme, such as "Huge APP").
- **Panics.** A fault in the Rust core calls `abort()`, so the board
  restarts, which stops its PWM outputs.
