# RLMeshTether

A board without a network (an Arduino Nano, Uno, Mega or UNO R4, over a
USB cable) speaks framed messages over a serial link to a computer, and the
computer is the RLMesh robot for it. This library is the board's side.

> **Use `tether.print()`, never `Serial.print()`.** On a Nano the tether
> and `Serial` share one UART, and printed text corrupts frames.

The board runtime, `RLMeshTether<Map>`, does the board's half of the spec
(§5–6), with no switch to turn any of it off:

- **Discovery:** `HELLO` at boot and every 500 ms until attached; the map
  is served from flash on request.
- **Attach** checks the board id and boot. Each attach brings a new
  generation, and frames from any other generation are dropped.
- **Drive permits:** a `COMMAND` drives its part for its `ttl_ms` and no
  longer. When the permit runs out, the part goes to its declared safe
  behaviour until the next command.
- **Link watchdog:** with no valid frame for `watchdog_ms` (default 250),
  every part goes safe and the board says `HELLO` again.
- **Hardware watchdog:** armed on the first `poll()`, so a slow `setup()` is
  fine. A sketch that doesn't reach `poll()` or `tether.delay()` for 500 ms
  resets the MCU, and the next `STATUS` says so.
- **E-stop pin (optional):** latched in RAM and in storage that survives a
  reset. It is cleared only from the computer the board is plugged into,
  with the button released.

`rlmesh tether gen` writes the typed header for your map. `examples/Eyes`
has one written by hand in the same shape until the generator lands.

The codec underneath (CRC, COBS framing, frames and messages) is byte for
byte the same as the Rust crate `rlmesh-tether`, checked against the same
conformance vectors.

## Boards

| Board                          | Watchdog | E-stop latch                         | Boot value              |
| ------------------------------ | -------- | ------------------------------------ | ----------------------- |
| AVR: Uno, Nano, Mega, Leonardo | WDT      | last EEPROM byte, program-only write | EEPROM counter, 4 bytes |
| UNO R4 Minima, R4 WiFi         | `WDT.h`  | data flash 0x40101C00, blank check   | VBATT backup registers  |

Any other board fails to build instead of running without a watchdog.

- **AVR:** the library uses the last 5 bytes of EEPROM.
- **R4:** the boot value uses the last 6 VBATT backup registers.
- **R4 with an `estopPin`: don't use `EEPROM.h`.** Its emulation manages all
  of the data flash, and could erase the latch. With it linked, the latch
  reads as set and never clears, so the board stays e-stopped.
- **Reset cause:** read at startup. A stock Uno or Nano bootloader (Optiboot
  4.4) clears it before the sketch runs, so those boards report "other";
  Optiboot 8 passes it on.

## The sketch

```cpp
#include <RLMeshTether.h>
#include <Servo.h>

#include "eyes_map.h"  // generated: Part, EyesCmd, EyesState, EyesMap

Servo pan, tilt;
RLMeshTether<EyesMap> tether(Serial);

void onEyes(const EyesCmd& cmd, bool hold) {
  pan.write(90 + int(cmd.pos[0] * 57.2958f));
  tilt.write(90 + int(cmd.pos[1] * 57.2958f));
}

void onSafe(Part part, SafeBehaviour how, bool estop) {
  // Put `part` in its declared safe state (`how`): a `stop` part would
  // detach() its servos here.
}

void setup() {
  pan.attach(9);
  tilt.attach(10);
  Serial.begin(115200);
  tether.onCommand(EYES, onEyes);
  tether.begin(onSafe, /*watchdogMs=*/250);  // last
}

void loop() {
  tether.poll();
  tether.every(20, [] { tether.publish(EYES, EyesState{{0.1f, 0.2f}}); });
}
```

- **Callbacks** run from `poll()`, with a budget of 2 ms for all of them in
  one `poll()`.
- **`onSafe`** is called once per part at `begin()`, then on each change:
  safe, e-stop, or released.
- **Pauses:** in `loop()`, use `tether.delay(ms)`, which keeps polling,
  and `tether.every(ms, fn)` (one lambda per call site).
- **Never wait inside a callback.** There, `poll()` and `tether.delay()` do
  nothing, not even feed the watchdog, so a callback that waits more than
  500 ms resets the board.
- **`RLMeshTether<Map, RxMax>`:** `RxMax` (16–256, default 64) is the
  largest frame the board accepts. It must fit the largest command's dense
  value plus 9.

### E-stop wiring

```cpp
tether.estopPin(2, /*activeLow=*/false);  // before begin()
```

- **Wiring:** a normally-closed button between the pin and GND, with
  `activeLow` false. The pin's pull-up is on, so a cut wire reads as
  pressed.
- **Buttons:** prefer one that stays pressed. A momentary one also works,
  because the latch is stored.
- **First flash:** the stored latch may read as set. Clear it with
  `rlmesh tether clear-estop`.

### Wiring that resets safely

- **A reset must leave the board safe:** pins float or go low until
  `setup()` runs.
- **Hobby servos:** power them separately, not from the board's 5 V, or
  they brown it out.
- **Motor drivers (L298N and similar):** pull the enable pin down.
- **ESCs:** arm only from the sketch.

## Design

- **C++11, no heap, no exceptions, no RTTI, no C++ standard library.** It
  builds with avr-gcc (`-std=gnu++11`) and with any host g++ or clang.
- **The runtime is written against a small HAL** (`board.h`): a clock, the
  serial port, the watchdog, the e-stop pin and its storage. The same code
  runs on a board (`arduino.h`, `platform_*.cpp`) and in the host tests.
- **No transmit buffer.** A frame's CRC is computed first, then the COBS
  encoder reads ahead in place across the header, the value and the CRC, and
  writes each byte straight to a sink.
- **One receive buffer**, sized by the caller: the board's `rx_max` (16 to
  256 decoded bytes).
- **Values may live in flash.** `Bytes::progmem(p, n)` reads with
  `pgm_read_byte` where the platform needs it (AVR, and ESP8266, whose flash
  takes word loads only), so the map is sent from `PROGMEM`. On ARM, ESP32
  and RISC-V boards flash is ordinary memory and reads directly.
- **No Arduino header in the codec.** A sink is anything with
  `write(uint8_t)`: `sinkTo(Serial)` on Arduino, or your own UART driver.

## The codec alone

```cpp
#include <RLMeshTether.h>
using namespace rlmesh::tether;

CobsDecoderBuffer<64> rx;  // rx_max = 64
uint8_t seq = 0;

int16_t commandLen(const void*, uint8_t part) { return part == 0 ? 8 : -1; }

void loop() {
  while (Serial.available()) {
    if (rx.push(Serial.read()) != CobsResult::Frame) continue;
    Frame f;
    Msg m;
    DenseLen lens = {commandLen, nullptr};
    if (parseFrame(rx.data(), rx.length(), f) != FrameError::Ok) continue;
    if (decodeForBoard(f, lens, m) != Error::Ok) continue;
    if (m.type == ty::kPing) {
      Status s = {/* boot */ 1, m.ping.gen, status_flags::kAttached, 0, 0, 0, micros()};
      writeStatus(s, seq++, sinkTo(Serial));
    }
  }
}
```

| Header                  | What                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `rlmesh_tether/bytes.h` | `Bytes` (RAM or flash) and `Sink`, `sinkTo(...)`                   |
| `rlmesh_tether/crc.h`   | CRC-16/CCITT-FALSE, CRC-32/ISO-HDLC, bitwise                       |
| `rlmesh_tether/cobs.h`  | `cobsEncode` over pieces, `CobsDecoder` byte at a time             |
| `rlmesh_tether/frame.h` | `parseFrame`, `writeFrame`                                         |
| `rlmesh_tether/msg.h`   | the messages, `decode`/`decodeForBoard`, `write`/`writeStatus`/... |
| `rlmesh_tether/board.h` | the board runtime, `Board<Hal>`                                    |

Decoding checks what the Rust crate checks, with the same errors: direction,
unknown types, `HELLO`'s magic and version (before its length), exact
lengths, reserved bits, `rx_max` 16–256, `watchdog_ms` 20–5000, a UTF-8
`name` of at most 32 bytes, and unknown parts. Writing refuses what decoding
would, and writes nothing then.

Dense ↔ aligned conversion is not here: the bridge does it, and the board
only ever sees dense values.

## Tests

`mise run test:tether:cpp` runs everything below.

**Host build.** It compiles the library as a board would: `-nostdinc++`,
no exceptions or RTTI, and no symbols beyond `memcpy`/`memset`.

**Under GCC and zig's clang, it runs:**

- every vector in `crates/rlmesh-tether/tests/vectors.json`;
- the board runtime's tests (`test/board_test.cpp`) against a fake clock,
  port and storage: attach, permits, the generation fence, the link
  watchdog, a stalled sketch, a poll() from a callback, a flood of frames,
  the e-stop and its clear, a latch that
  survives a reset, counters, notes and the clock wrapping.

**Sanitizers.** Both runners are built with whichever ones the toolchain
can prove work (ASan and UBSan, else UBSan in trap mode), and the task
prints which ran.

**When the tools are there:**

- with zig, it compiles the codec for an ATmega328P, where `int` is 16 bits;
- with `arduino-cli` and its cores, it builds `examples/Eyes` for a Nano and
  an R4 Minima and fails on any warning from the library. `ARDUINO_CLI`
  names a binary off `PATH`.

```bash
mise run test:tether:cpp
CXX=clang++ mise run test:tether:cpp   # one compiler, host tests only
```

## Licence

MIT or Apache-2.0, at your option.
