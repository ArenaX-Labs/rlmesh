# RLMeshTether

A board without a network (an Arduino Nano, an Uno, any small MCU with a
UART) speaks framed messages over a serial link to a computer, and the
computer is the RLMesh robot for it. This library is the board's side.

**This release is the codec only:** CRC, COBS framing, frames and the
tether's messages, byte for byte the same as the Rust crate `rlmesh-tether`
and checked against the same conformance vectors. The board runtime (the
`RLMeshTether<Map>` class with `poll()`, drive permits, the link watchdog,
e-stop and the `HELLO` loop) comes in the next release, on top of this API.

## Design

- **C++11, no heap, no exceptions, no RTTI, no C++ standard library.** It
  builds with avr-gcc (`-std=gnu++11`) and with any host g++ or clang.
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

## Use

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

Decoding checks what the Rust crate checks, with the same errors: direction,
unknown types, `HELLO`'s magic and version (before its length), exact
lengths, reserved bits, `rx_max` 16–256, `watchdog_ms` 20–5000, a UTF-8
`name` of at most 32 bytes, and unknown parts. Writing refuses what decoding
would, and writes nothing then.

Dense ↔ aligned conversion is not here: the bridge does it, and the board
only ever sees dense values.

## Tests

The host runner compiles the codec as a board would (`-nostdinc++`, no
exceptions or RTTI, no symbols beyond `memcpy`/`memset`), then runs every
vector in `crates/rlmesh-tether/tests/vectors.json` against it, under GCC and
zig's clang. The runner is built with whichever sanitizers the toolchain can
prove work (ASan and UBSan, else UBSan in trap mode), and the task prints
which ran. With zig available it also compiles the codec for an ATmega328P,
where `int` is 16 bits.

```bash
mise run test:tether:cpp
CXX=clang++ mise run test:tether:cpp   # one compiler
```

## Licence

MIT or Apache-2.0, at your option.
