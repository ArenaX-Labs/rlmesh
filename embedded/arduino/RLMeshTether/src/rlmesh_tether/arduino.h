// `RLMeshTether<Map>`: the board runtime on an Arduino `Stream`, typed by
// the header `rlmesh tether gen` writes (spec §9.2).
//
// The generated `Map` provides:
//
//   enum Part : uint8_t { ... };           // authored order
//   enum SafeBehaviour : uint8_t { ... };  // the declared `safe` names
//   static const uint8_t kParts;
//   static const rlmesh::tether::BoardSpec& spec();

#ifndef RLMESH_TETHER_ARDUINO_H_
#define RLMESH_TETHER_ARDUINO_H_

#include <Arduino.h>
#include <string.h>

#include "board.h"
#include "platform.h"

namespace rlmesh {
namespace tether {

// The board's clock, serial port and pins.
class ArduinoHal {
 public:
  explicit ArduinoHal(Stream& serial) : serial_(serial) {}

  uint32_t millis() { return ::millis(); }
  uint32_t micros() { return ::micros(); }
  int read() { return serial_.read(); }
  void write(uint8_t byte) { serial_.write(byte); }
  void watchdogArm() { platform::watchdogArm(); }
  void watchdogFeed() { platform::watchdogFeed(); }

  void estopPin(uint8_t pin, bool activeLow) {
    pin_ = pin;
    activeLow_ = activeLow;
    pinMode(pin, INPUT_PULLUP);
  }
  bool estopConfigured() { return pin_ != kNoPin; }
  bool estopActive() { return (digitalRead(pin_) == LOW) == activeLow_; }
  bool latchLoad() { return platform::latchLoad(); }
  void latchStore() { platform::latchStore(); }
  void latchClear() { platform::latchClear(); }
  uint16_t boot() { return platform::boot(); }
  uint8_t resetCause() { return platform::resetCause(); }
  uint8_t arch() { return platform::arch(); }

 private:
  static const uint8_t kNoPin = 0xFF;
  Stream& serial_;
  uint8_t pin_ = kNoPin;
  bool activeLow_ = true;
};

// `RxMax` is the largest frame the board accepts (`HELLO.rx_max`): at
// least 16, and enough for the largest `COMMAND` (its dense value + 9).
template <class Map, uint16_t RxMax = 64>
class RLMeshTether {
  static_assert(RxMax >= kRxMaxMin && RxMax <= kRxMaxMax, "RxMax is 16..=256");

 public:
  typedef typename Map::Part Part;
  typedef typename Map::SafeBehaviour SafeBehaviour;
  typedef void (*SafeHandler)(Part part, SafeBehaviour how, bool estop);
  typedef void (*RawCommandHandler)(const uint8_t* value, uint16_t len, bool hold);

  explicit RLMeshTether(Stream& serial)
      : hal_(serial), board_(hal_, Map::spec(), slots_, rx_, RxMax) {}

  RLMeshTether(const RLMeshTether&) = delete;
  RLMeshTether& operator=(const RLMeshTether&) = delete;

  // An optional e-stop input, before begin(). The pin has its pull-up on.
  // A normally-closed button between the pin and GND, with `activeLow`
  // false, reads a cut wire as pressed.
  void estopPin(uint8_t pin, bool activeLow = true) { hal_.estopPin(pin, activeLow); }

  // Runs `fn` with each command for `part`. `T` is the part's generated
  // command struct; false when its size isn't the part's dense command.
  template <class T>
  bool onCommand(Part part, void (*fn)(const T& command, bool hold)) {
    if (!fits(part, sizeof(T))) return false;
    handlers_[part].fn = reinterpret_cast<void (*)()>(fn);
    handlers_[part].call = &Typed<T>::call;
    return true;
  }

  // The untyped form: the dense bytes as they arrived.
  bool onCommand(Part part, RawCommandHandler fn) {
    if (part >= Map::kParts || Map::spec().parts[part].commandLen < 0) return false;
    handlers_[part].fn = reinterpret_cast<void (*)()>(fn);
    handlers_[part].call = &raw;
    return true;
  }

  // Starts the tether: every part safe, then the first `HELLO`. Call it
  // last in setup(); the hardware watchdog arms on the first poll().
  void begin(SafeHandler onSafe, uint16_t watchdogMs = kDefaultWatchdogMs) {
    onSafe_ = onSafe;
    board_.begin(&safe, &command, this, watchdogMs);
  }

  // Call from loop() as often as possible: every callback runs from here,
  // with a budget of 2 ms for all of them.
  void poll() { board_.poll(); }

  // delay() that keeps the tether running.
  void delay(uint32_t ms) {
    uint32_t start = ::millis();
    do {
      poll();
    } while (::millis() - start < ms);
  }

  // Runs `fn` every `ms`, from loop(). Each call site needs its own lambda:
  // the timer lives with the lambda's type.
  template <class F>
  void every(uint32_t ms, F fn) {
    static uint32_t last = 0;
    static bool ran = false;
    uint32_t now = ::millis();
    if (ran && now - last < ms) return;
    ran = true;
    last = now;
    fn();
  }

  // Publishes `part`'s sample; false unless attached and `T` is the part's
  // dense sample size.
  template <class T>
  bool publish(Part part, const T& sample) {
    return board_.publish(part, reinterpret_cast<const uint8_t*>(&sample), sizeof(T));
  }

  // A `NOTE` to the computer: use this, not Serial.print(), which shares
  // the tether's port. At most 48 bytes and 2 a second.
  bool print(const char* text) {
    return board_.note(0, 0, reinterpret_cast<const uint8_t*>(text), uint16_t(strlen(text)));
  }

  bool attached() const { return board_.attached(); }
  bool estopLatched() const { return board_.estopLatched(); }

 private:
  struct Handler {
    void (*fn)();
    void (*call)(void (*fn)(), const uint8_t* value, uint16_t len, bool hold);
  };

  template <class T>
  struct Typed {
    static void call(void (*fn)(), const uint8_t* value, uint16_t len, bool hold) {
      // Copied out: the receive buffer may not be aligned for `T`.
      T command;
      memcpy(&command, value, len);
      reinterpret_cast<void (*)(const T&, bool)>(fn)(command, hold);
    }
  };

  static void raw(void (*fn)(), const uint8_t* value, uint16_t len, bool hold) {
    reinterpret_cast<RawCommandHandler>(fn)(value, len, hold);
  }

  static bool fits(Part part, size_t size) {
    return part < Map::kParts && Map::spec().parts[part].commandLen == int16_t(size);
  }

  static void safe(void* ctx, uint8_t part, uint8_t how, bool estop) {
    RLMeshTether& t = *static_cast<RLMeshTether*>(ctx);
    if (t.onSafe_) t.onSafe_(Part(part), SafeBehaviour(how), estop);
  }

  static void command(void* ctx, uint8_t part, const uint8_t* value, uint16_t len, bool hold) {
    Handler& h = static_cast<RLMeshTether*>(ctx)->handlers_[part];
    if (h.call) h.call(h.fn, value, len, hold);
  }

  ArduinoHal hal_;
  PartSlot slots_[Map::kParts];
  Handler handlers_[Map::kParts] = {};
  uint8_t rx_[RxMax];
  SafeHandler onSafe_ = nullptr;
  Board<ArduinoHal> board_;
};

}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_ARDUINO_H_
