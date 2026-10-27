// Byte runs and byte sinks: what the codec reads from and writes to.
//
// No Arduino header here: a sink is any `write(uint8_t)` (an Arduino
// `Stream` or `Print`, or your own UART driver), and a run of bytes may sit
// in program memory, which AVR and ESP8266 read with `pgm_read_byte`.

#ifndef RLMESH_TETHER_BYTES_H_
#define RLMESH_TETHER_BYTES_H_

#include <stddef.h>
#include <stdint.h>

#if defined(__AVR__)
#include <avr/pgmspace.h>
#elif defined(ARDUINO_ARCH_ESP8266) || defined(ESP8266)
#include <pgmspace.h>
#endif

namespace rlmesh {
namespace tether {

// A run of bytes in RAM or, with `flash`, in program memory (`PROGMEM`).
// Flash is read with `pgm_read_byte` wherever the platform defines it: AVR
// has a separate address space, and ESP8266 maps flash for word loads only,
// so a plain byte load faults there. Elsewhere (ARM, ESP32, RISC-V, the
// host) flash is ordinary addressable memory and both read alike.
struct Bytes {
  const uint8_t* data;
  uint16_t len;
  bool flash;

  static Bytes ram(const uint8_t* data, uint16_t len) {
    Bytes b = {data, len, false};
    return b;
  }

  static Bytes progmem(const uint8_t* data, uint16_t len) {
    Bytes b = {data, len, true};
    return b;
  }

  uint8_t at(uint16_t i) const {
#if defined(pgm_read_byte)
    if (flash) return pgm_read_byte(data + i);
#endif
    return data[i];
  }
};

// Where encoded bytes go, one at a time: a function and its context.
struct Sink {
  void (*put)(void* ctx, uint8_t byte);
  void* ctx;

  void operator()(uint8_t byte) const { put(ctx, byte); }
};

template <class T>
struct WriteTo {
  static void put(void* ctx, uint8_t byte) { static_cast<T*>(ctx)->write(byte); }
};

// A sink that calls `target.write(byte)`: `sinkTo(Serial)` on Arduino.
template <class T>
Sink sinkTo(T& target) {
  Sink s = {&WriteTo<T>::put, &target};
  return s;
}

}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_BYTES_H_
