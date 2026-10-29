# RLMeshESP32

An ESP32 on Wi-Fi joins an RLMesh session as a robot, with no computer in
between. The session's protocol and crypto are RLMesh's Rust core
(`crates/rlmesh-device`), linked into the sketch as a precompiled
library. This library adds the board's clock and randomness, and its
network: [zenoh-pico](https://github.com/eclipse-zenoh/zenoh-pico) at the
version of the host's Zenoh.

> **Work in progress.** A board waits in its home's lobby, joins the scene
> runs it is assigned, takes commands and publishes samples, all sealed or
> signed per part (`examples/Robot`), and latches every member's e-stop.
> Lockstep and serving history come next.

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

A board joins its **home**, the `rlmesh daemon` it was provisioned for, as
its own device. Provision it once over USB, on the daemon's computer, with
the board running a sketch that answers (`examples/Robot`) and the Serial
Monitor closed:

```sh
rlmesh devices provision --port /dev/ttyUSB0 --name eyes
```

- **The device key:** the board makes it on first boot, after Wi-Fi starts (the RNG needs the radio), and keeps it in NVS.
- **The exchange:** provisioning reads the public key and checks the name compiled into the sketch, then writes the home's key, id and addresses to NVS (tether spec §14). It enrolls the key with the daemon and pins the robot to it.
- **Afterwards:** the board waits in its home's lobby, at once and on every boot after. Provisioning again keeps the device; a full flash erase makes a new one.

### Runs

A provisioned board waits in its home's **lobby** until a scene run assigns
it, then joins the run, declares its parts, and comes back when the run
ends (scene design §3):

```sh
rlmesh lobby                       # eyes  (from eyes)
rlmesh run eyes.toml --arm manual  # r-dtqh: hosting, assigning, declaring, arming
rlmesh arm r-dtqh
rlmesh stop r-dtqh                 # eyes: returned
```

- **In the lobby** the board is attached and heartbeating, and asks for an order every 10 s; it never declares there.
- **Leaving a run:** when the run closes it detaches and goes back to the lobby. If the run's host loses it (a Wi-Fi drop, a reset), it goes back to the lobby too, and the home sends the same join again, so the run continues once someone arms it again. If another board takes its name, it stays safe and stops until reset.
- **A run hosted on another machine** (a satellite daemon): the board connects to the run's addresses, in order, and handshakes with the run's own host key; if none answers, the run shows the failure and the board goes back to the lobby at home.
- **Finding home:** the board tries each stored address in turn, then backs off (1 s doubling to 30 s). While none answers it asks mDNS for `_rlmesh._tcp` and keeps only the daemon whose TXT `id` is the one it was provisioned with; a wrong answer only fails the handshake.
- **Its map** must be the same on every join: a rejoin with another one is refused.
- `robot.status()` shows `in_run`, phase 6 while it waits in the lobby, `left` (why it last left a run) and `home_unreachable`.
- **At rest:** anyone with the board in hand can read its key from flash; revoke the device (`rlmesh devices revoke`) if one goes missing.

For a one-off host instead, run one on the computer's LAN address and put
the address it prints in `kAddress`:

```sh
rlmesh host --insecure --listen 192.168.1.20:7447
```

`--insecure` lets anyone who can reach those ports try the host; the
handshake and sealing still decide who joins. A host on a LAN address takes
the board's plain TCP links (a loopback host gates its links with a key
zenoh-pico can't present).

Put your Wi-Fi and the robot's map in `examples/Robot`, flash it, and open
the Serial Monitor at 115200:

```cpp
rlmesh::Robot robot;
int eyes;

void setup() {
  WiFi.begin(ssid, password);  // first: the RNG needs the radio on
  // ... wait for WL_CONNECTED
  robot.begin(address, "eyes", robotJson);
  eyes = robot.part("eyes");
}

void loop() {
  robot.poll();  // every loop
  // At the part's control rate:
  rlmesh_device_tick_t t;
  if (robot.tick(eyes, t) && t.has_action) {
    // t.value: the action, in the part's dense command layout
  }
  robot.publish(eyes, state);  // its dense sample layout
}
```

- **Joining:**
  - the board reads the host's signed state;
  - it runs the handshake with the address's key;
  - it attaches and declares its parts, sealed;
  - it takes the keys the host delivers.

  `robot.status()` says where it is, and why it last started over.

- **Commands:** `tick()` is the edge of session-core, as on any robot: an action under the lease, a hold, or the part's declared safe behaviour (`t.safe`) when the lease, the host or the commands lapse. It tells the host when it goes safe on its own account, and publishes what it applied.
- **Samples:** `publish()` returns `RLMESH_DEVICE_ERR_NOT_READY` until the robot runs and its stream key has arrived.
- **E-stop:** the board latches every member's e-stop signal at once, with or without the host. `robot.estop()` latches its own (an e-stop pin: `kEstopPin` in the example). Only `robot.clearEstop()`, a local action with the button released (the example: hold BOOT for 2 s), clears it; the host then stays in ESTOP until disarmed and armed. A latch carries into the board's next session.
- **Starting over:** any failure (no reply, a refusal, nothing from the host for 5 s) starts over from the handshake after a second.
- **The address's key is a secret.** Anyone holding it can join the session.
- **Ports:** the host listens on four consecutive ports, one per traffic class. The board opens a session to each one it uses (control, and its parts' classes), when it first uses it.

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
- **Flash.** `examples/Robot` is about 1.5 MB (1.6 MB on the C6), past the
  default app partition. Choose a larger one (Tools → Partition Scheme,
  "Huge APP"). The crypto alone adds about 205 KB to a Wi-Fi sketch (189 KB
  on the C6).
- **Panics.** A fault in the Rust core calls `abort()`, so the board
  restarts, which stops its PWM outputs.
