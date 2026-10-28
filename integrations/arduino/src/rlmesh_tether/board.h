// The board's runtime (spec §5–6): the `HELLO` loop, the map, attach and
// its generation fence, drive permits, the link watchdog, the hardware
// watchdog and the pin e-stop.
//
// It is written against a `Hal`, so the same code runs on a board
// (`arduino.h`) and in the host tests. A `Hal` provides:
//
//   uint32_t millis(); uint32_t micros();
//   int read();                 // the next received byte, or -1
//   void write(uint8_t byte);
//   void watchdogArm();         // the MCU's watchdog, from the first poll()
//   void watchdogFeed();
//   bool estopConfigured();     // a sketch called estopPin()
//   bool estopActive();         // the pin, now
//   bool latchLoad();           // the stored latch; a read fault is latched
//   void latchStore();          // a program-only write (§6.4)
//   void latchClear();          // the erase
//   uint16_t boot();            // differs from the previous boot
//   uint8_t resetCause();       // `Status::reset`
//   uint8_t arch();             // `Hello::arch`

#ifndef RLMESH_TETHER_BOARD_H_
#define RLMESH_TETHER_BOARD_H_

#include "cobs.h"
#include "msg.h"

namespace rlmesh {
namespace tether {

// This library's version, in `HELLO`.
const uint16_t kLibVersion = 0x0200;
// How often an unattached board says `HELLO`.
const uint16_t kHelloMs = 500;
// The longest gap between `STATUS` frames while attached (§4.1).
const uint16_t kStatusMaxMs = 100;
// `NOTE` rate: at most this many per second; extras are dropped.
const uint8_t kNotesPerSecond = 2;
// The link watchdog's default (§6.2).
const uint16_t kDefaultWatchdogMs = 250;
// The most frames one poll() acts on, so a flood can't hold it.
const uint8_t kFramesPerPoll = 8;

// One part, as the generator describes it.
struct PartSpec {
  // The dense command length, or -1 when the part takes no commands.
  int16_t commandLen;
  // The dense sample length, or -1 when it publishes none.
  int16_t sampleLen;
  // Its declared safe behaviour: the generated `SafeBehaviour`.
  uint8_t safe;
};

// What `rlmesh tether gen` compiles into a sketch.
struct BoardSpec {
  // The minified map, usually in `PROGMEM`.
  Bytes map;
  // CRC-32/ISO-HDLC of `map`.
  uint32_t mapCrc;
  // 8 bytes.
  Bytes boardId;
  // UTF-8, at most `kMaxName` bytes.
  Bytes name;
  // In authored order: `part` indexes it.
  const PartSpec* parts;
  uint8_t partCount;
};

// Where a part stands.
enum class PartMode : uint8_t {
  // A permit is running.
  Driving,
  // Safe for now: no permit, a `SAFE`, a trip or a detach. The next valid
  // `COMMAND` resumes it.
  Safe,
  // Safe under the pin latch, until a local clear.
  Estop,
};

// A part's permit; the sketch's wrapper owns one per part.
struct PartSlot {
  uint32_t until;
  PartMode mode;
};

// `onCommand` for every part: the value is the part's dense command.
typedef void (*CommandFn)(void* ctx, uint8_t part, const uint8_t* value, uint16_t len, bool hold);
// `onSafe`: `how` is the part's `PartSpec::safe`.
typedef void (*SafeFn)(void* ctx, uint8_t part, uint8_t how, bool estop);

template <class Hal>
class Board {
 public:
  // `slots` holds `spec.partCount` entries, and `rx` `rxMax` bytes
  // (`kRxMaxMin`..=`kRxMaxMax`).
  Board(Hal& hal, const BoardSpec& spec, PartSlot* slots, uint8_t* rx, uint16_t rxMax)
      : hal_(hal), spec_(spec), slots_(slots), rx_(rx, rxMax), rxMax_(rxMax) {}

  // Puts every part in its safe state (`onSafe` once each), restores the
  // stored e-stop latch, and sends the first `HELLO`. `watchdogMs` is
  // clamped to 20–5000.
  void begin(SafeFn onSafe, CommandFn onCommand, void* ctx, uint16_t watchdogMs) {
    onSafe_ = onSafe;
    onCommand_ = onCommand;
    ctx_ = ctx;
    watchdogMs_ = watchdogMs < kWatchdogMsMin   ? kWatchdogMsMin
                  : watchdogMs > kWatchdogMsMax ? kWatchdogMsMax
                                                : watchdogMs;
    boot_ = hal_.boot();
    reset_ = hal_.resetCause();
    // A sketch without a pin ignores storage: stale bytes from another
    // sketch can't latch a board that has no button.
    estop_ = hal_.estopConfigured();
    latched_ = estop_ && hal_.latchLoad();
    input_ = estop_ && hal_.estopActive();
    for (uint8_t p = 0; p < spec_.partCount; ++p) {
      slots_[p].mode = latched_ || input_ ? PartMode::Estop : PartMode::Safe;
      onSafe_(ctx_, p, spec_.parts[p].safe, slots_[p].mode == PartMode::Estop);
    }
    // An active input is a latch, whatever was stored.
    bool store = input_ && !latched_;
    latched_ = latched_ || input_;
    started_ = true;
    // The `HELLO`'s flag reports the latch before it's stored, as `STATUS`
    // would once attached.
    sendHello(hal_.millis());
    if (store) hal_.latchStore();
  }

  // Runs the board: call it from `loop()` at least every couple of
  // milliseconds. Arms the hardware watchdog on the first call.
  //
  // From inside a callback it does nothing, not even feed the watchdog: a
  // frame acted on there could safe a part the outer callback then drives.
  // A callback that waits in `tether.delay()` past the hardware watchdog
  // resets the board.
  void poll() {
    if (!started_ || polling_) return;
    polling_ = true;
    if (!armed_) {
      hal_.watchdogArm();
      armed_ = true;
    }
    hal_.watchdogFeed();
    // Timers before each frame: a sketch that stalled past a permit or the
    // link watchdog stops before anything buffered meanwhile is acted on,
    // and a stream of frames can't hold a permit or the pin unchecked.
    for (uint8_t frames = 0; frames < kFramesPerPoll; ++frames) {
      timers(hal_.millis());
      if (!receiveFrame()) break;
    }
    uint32_t now = hal_.millis();
    if (!attached_ && reached(now, nextHello_)) sendHello(now);
    if ((attached_ || latched_) && reached(now, nextStatus_)) sendStatus(now);
    polling_ = false;
  }

  // Sends `part`'s dense sample, timestamped now. Only while attached, and
  // only at the part's declared length.
  bool publish(uint8_t part, const uint8_t* value, uint16_t len) {
    if (!attached_ || part >= spec_.partCount || spec_.parts[part].sampleLen != int16_t(len)) {
      return false;
    }
    State m = {part, hal_.micros(), Bytes::ram(value, len)};
    return writeState(m, seq_++, sinkTo(hal_)) == Error::Ok;
  }

  // Sends a `NOTE` (truncated to `kMaxNote` bytes), at most
  // `kNotesPerSecond` a second.
  bool note(uint8_t level, uint8_t code, const uint8_t* text, uint16_t len) {
    if (!started_) return false;
    uint32_t now = hal_.millis();
    // The older of the last two notes must be a second old.
    if (notes_ >= kNotesPerSecond && now - noteAt_[0] < 1000) return false;
    if (notes_ < kNotesPerSecond) ++notes_;
    noteAt_[0] = noteAt_[1];
    noteAt_[1] = now;
    Note m = {level, code, Bytes::ram(text, len > kMaxNote ? kMaxNote : len)};
    return writeNote(m, seq_++, sinkTo(hal_)) == Error::Ok;
  }

  bool attached() const { return attached_; }
  // The current generation, or the last one since boot.
  uint8_t gen() const { return gen_; }
  bool estopLatched() const { return latched_; }
  uint16_t lost() const { return lost_; }
  uint16_t bad() const { return bad_; }
  uint16_t watchdogMs() const { return watchdogMs_; }
  uint16_t bootId() const { return boot_; }

 private:
  static bool reached(uint32_t now, uint32_t at) { return int32_t(now - at) >= 0; }

  static int16_t commandLen(const void* ctx, uint8_t part) {
    const BoardSpec& s = *static_cast<const BoardSpec*>(ctx);
    return part < s.partCount ? s.parts[part].commandLen : -1;
  }

  static void bump(uint16_t& counter) {
    if (counter != 0xFFFF) ++counter;
  }

  void timers(uint32_t now) {
    sampleEstop(now);
    if (attached_ && reached(now, lastValid_ + watchdogMs_)) {
      safeAll(PartMode::Safe);
      attached_ = false;
      sendHello(now);
    }
    for (uint8_t p = 0; p < spec_.partCount; ++p) {
      if (slots_[p].mode == PartMode::Driving && reached(now, slots_[p].until)) {
        enterSafe(p, PartMode::Safe);
      }
    }
  }

  // Reads up to the end of one frame, valid or not, and acts on it. False
  // when the input ran dry first.
  bool receiveFrame() {
    for (int b = hal_.read(); b >= 0; b = hal_.read()) {
      switch (rx_.push(static_cast<uint8_t>(b))) {
        case CobsResult::None:
          continue;
        case CobsResult::Overflow:
        case CobsResult::Truncated:
          bump(bad_);
          return true;
        case CobsResult::Frame:
          receive();
          return true;
      }
    }
    return false;
  }

  void receive() {
    Frame f;
    Msg m;
    DenseLen lens = {&Board::commandLen, &spec_};
    if (parseFrame(rx_.data(), rx_.length(), f) != FrameError::Ok ||
        decodeForBoard(f, lens, m) != Error::Ok) {
      bump(bad_);
      return;
    }
    handle(m, f.seq, hal_.millis());
  }

  void handle(const Msg& m, uint8_t seq, uint32_t now) {
    switch (m.type) {
      case ty::kGetMap:
        if (m.getMap.boot == boot_) sendMap(m.getMap);
        return;
      case ty::kAttach:
        attach(m.attach, seq, now);
        return;
      default:
        break;
    }
    // Everything else carries `gen`, and is ignored before attach.
    if (!attached_) return;
    uint8_t gen = m.type == ty::kCommand ? m.command.gen
                  : m.type == ty::kSafe  ? m.safe.gen
                  : m.type == ty::kPing  ? m.ping.gen
                                         : m.detach.gen;
    if (gen != gen_) {
      bump(bad_);
      return;
    }
    countLost(seq);
    lastValid_ = now;
    switch (m.type) {
      case ty::kCommand:
        command(m.command, now);
        return;
      case ty::kSafe:
        for (uint8_t p = 0; p < spec_.partCount; ++p) {
          if (m.safe.part == kEveryPart || m.safe.part == p) enterSafe(p, PartMode::Safe);
        }
        return;
      case ty::kDetach:
        safeAll(PartMode::Safe);
        attached_ = false;
        sendHello(now);
        return;
      default:  // PING feeds the watchdog, and that's all.
        return;
    }
  }

  void attach(const Attach& a, uint8_t seq, uint32_t now) {
    if (a.boot != boot_ || a.gen == 0) return;
    for (uint8_t i = 0; i < 8; ++i) {
      if (a.boardId[i] != spec_.boardId.at(i)) return;
    }
    if (a.gen == gen_) {
      // A retry of the current attach: confirm it again. Unattached, the
      // last `gen` never comes back (a late retry from before a trip).
      if (!attached_) return;
    } else {
      if (attached_) safeAll(PartMode::Safe);
      gen_ = a.gen;
      attached_ = true;
      // The pin is read again here: it may have been pressed since.
      if ((a.flags & attach_flags::kClearEstop) && latched_ && !input_ && !hal_.estopActive()) {
        // Storage first, then RAM, then `STATUS` (below). Storage that
        // still reads as set (a failed erase, or none usable) keeps it.
        hal_.latchClear();
        latched_ = hal_.latchLoad();
        // The erase stalls the MCU: time the link from after it.
        now = hal_.millis();
        for (uint8_t p = 0; !latched_ && p < spec_.partCount; ++p) {
          if (slots_[p].mode == PartMode::Estop) {
            slots_[p].mode = PartMode::Safe;
            onSafe_(ctx_, p, spec_.parts[p].safe, false);
          }
        }
      }
    }
    // The bridge's `seq` restarts at each `ATTACH`.
    expected_ = uint8_t(seq + 1);
    lastValid_ = now;
    sendStatus(now);
  }

  void command(const Command& c, uint32_t now) {
    // A pin latch drops every command; `ttl_ms` is clamped to
    // [1, watchdog_ms].
    if (latched_) return;
    uint16_t ttl = c.ttlMs == 0 ? 1 : c.ttlMs > watchdogMs_ ? watchdogMs_ : c.ttlMs;
    PartSlot& s = slots_[c.part];
    s.mode = PartMode::Driving;
    s.until = now + ttl;
    // The value is in the receive buffer, which the next byte overwrites:
    // it is read before poll() reads on.
    onCommand_(ctx_, c.part, c.value.data, c.value.len, (c.flags & command_flags::kHold) != 0);
  }

  void countLost(uint8_t seq) {
    uint8_t gap = uint8_t(seq - expected_);
    lost_ = uint16_t(lost_ + gap) < lost_ ? 0xFFFF : uint16_t(lost_ + gap);
    expected_ = uint8_t(seq + 1);
  }

  // Latches on an active input, and reports changes of the input.
  void sampleEstop(uint32_t now) {
    if (!estop_) return;
    bool input = hal_.estopActive();
    if (input == input_) return;
    input_ = input;
    if (input && !latched_) {
      // RAM and the parts, then `STATUS`, then storage (§6.4): a reset
      // before the store loses it, but the host already has the e-stop.
      latched_ = true;
      safeAll(PartMode::Estop);
      sendStatus(now);
      hal_.latchStore();
      return;
    }
    sendStatus(now);
  }

  void safeAll(PartMode why) {
    for (uint8_t p = 0; p < spec_.partCount; ++p) enterSafe(p, why);
  }

  // `onSafe` on each change: driving to safe, and into or out of the latch.
  // A transient `SAFE` never takes a part out of `Estop`.
  void enterSafe(uint8_t p, PartMode why) {
    PartMode& mode = slots_[p].mode;
    if (mode == why || mode == PartMode::Estop) return;
    mode = why;
    onSafe_(ctx_, p, spec_.parts[p].safe, why == PartMode::Estop);
  }

  void sendHello(uint32_t now) {
    Hello h;
    h.libVersion = kLibVersion;
    h.arch = hal_.arch();
    for (uint8_t i = 0; i < 8; ++i) h.boardId[i] = spec_.boardId.at(i);
    h.boot = boot_;
    h.gen = gen_;
    h.flags = latched_ ? hello_flags::kEstop : 0;
    h.mapLen = spec_.map.len;
    h.mapCrc = spec_.mapCrc;
    h.watchdogMs = watchdogMs_;
    h.rxMax = rxMax_;
    h.name = spec_.name;
    writeHello(h, seq_++, sinkTo(hal_));
    nextHello_ = now + kHelloMs;
  }

  void sendMap(const GetMap& g) {
    uint16_t len = spec_.map.len;
    uint16_t n = g.offset >= len ? 0 : uint16_t(len - g.offset);
    if (n > g.max) n = g.max;
    Bytes data = spec_.map;
    data.data += g.offset < len ? g.offset : len;
    data.len = n;
    MapSlice m = {boot_, g.offset, data};
    writeMap(m, seq_++, sinkTo(hal_));
  }

  void sendStatus(uint32_t now) {
    uint8_t flags = 0;
    if (attached_) flags |= status_flags::kAttached;
    if (latched_) flags |= status_flags::kEstopLatched;
    if (input_) flags |= status_flags::kEstopInput;
    Status s = {boot_, gen_, flags, reset_, lost_, bad_, hal_.micros()};
    writeStatus(s, seq_++, sinkTo(hal_));
    uint16_t period = watchdogMs_ / 3 < kStatusMaxMs ? uint16_t(watchdogMs_ / 3) : kStatusMaxMs;
    nextStatus_ = now + period;
  }

  Hal& hal_;
  const BoardSpec& spec_;
  PartSlot* slots_;
  CobsDecoder rx_;
  uint16_t rxMax_;
  SafeFn onSafe_ = nullptr;
  CommandFn onCommand_ = nullptr;
  void* ctx_ = nullptr;
  uint16_t watchdogMs_ = kDefaultWatchdogMs;
  uint16_t boot_ = 0;
  uint8_t reset_ = 0;
  uint32_t lastValid_ = 0;
  uint32_t nextHello_ = 0;
  uint32_t nextStatus_ = 0;
  uint32_t noteAt_[kNotesPerSecond] = {0, 0};
  uint16_t lost_ = 0;
  uint16_t bad_ = 0;
  uint8_t notes_ = 0;
  uint8_t gen_ = 0;
  uint8_t seq_ = 0;
  uint8_t expected_ = 0;
  bool started_ = false;
  bool polling_ = false;
  bool armed_ = false;
  bool attached_ = false;
  bool estop_ = false;
  bool latched_ = false;
  bool input_ = false;
};

}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_BOARD_H_
