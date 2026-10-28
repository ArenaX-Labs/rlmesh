// Runs the board runtime (`board.h`) against a fake clock, serial port,
// watchdog and e-stop storage, as the bridge would drive it: attach, drive,
// permits, the link watchdog, the generation fence and the pin e-stop
// (spec §5–6).
//
//   board_test

#include <stdio.h>
#include <string.h>

#include <deque>
#include <string>
#include <vector>

#include "../src/RLMeshTether.h"

using namespace rlmesh::tether;

namespace {

int checks = 0;
int failures = 0;
const char* scenario = "";

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++checks;                                                                \
    if (!(cond)) {                                                           \
      ++failures;                                                            \
      fprintf(stderr, "FAIL %s (line %d): %s\n", scenario, __LINE__, #cond); \
    }                                                                        \
  } while (0)

typedef std::vector<uint8_t> Buf;

// Two parts: `eyes` (two f32 commands, two f32 samples) and `lid` (a
// sample only).
const uint8_t kMap[] = "{\"eyes\":{\"kind\":\"joints\"},\"lid\":{\"kind\":\"sensor\"}}";
const uint8_t kBoardId[8] = {1, 2, 3, 4, 5, 6, 7, 8};
const uint8_t kName[] = "eyes";
const PartSpec kParts[] = {{8, 8, 1}, {-1, 1, 0}};
const BoardSpec kSpec = {
    Bytes::ram(kMap, sizeof kMap - 1),
    0xC0FFEE11,
    Bytes::ram(kBoardId, 8),
    Bytes::ram(kName, 4),
    kParts,
    2,
};
const uint16_t kBoot = 0x1234;

struct FakeHal {
  uint32_t ms = 1000;
  std::deque<uint8_t> rx;
  Buf tx;
  bool armed = false;
  int feeds = 0;
  bool estopPin = true;
  bool input = false;
  bool stored = false;
  // Milliseconds the clock moves per byte read (a slow link, or slow callbacks).
  uint32_t msPerRead = 0;
  // Presses the e-stop after this many more reads (when positive).
  int pressAfterReads = 0;
  std::vector<std::string> events;

  uint32_t millis() { return ms; }
  uint32_t micros() { return ms * 1000; }
  int read() {
    if (rx.empty()) return -1;
    ms += msPerRead;
    if (pressAfterReads > 0 && --pressAfterReads == 0) input = true;
    uint8_t b = rx.front();
    rx.pop_front();
    return b;
  }
  void write(uint8_t b) {
    tx.push_back(b);
    if (b == 0) events.push_back("tx");
  }
  void watchdogArm() { armed = true; }
  void watchdogFeed() { ++feeds; }
  bool estopConfigured() { return estopPin; }
  bool estopActive() { return input; }
  bool latchLoad() { return stored; }
  void latchStore() {
    stored = true;
    events.push_back("store");
  }
  void latchClear() {
    stored = false;
    events.push_back("clear");
  }
  uint16_t boot() { return kBoot; }
  uint8_t resetCause() { return 3; }
  uint8_t arch() { return 1; }
};

// A board frame, kept with its own copy of the payload.
struct Out {
  uint8_t type;
  uint8_t seq;
  Buf payload;

  Msg msg() const {
    static const DenseLen lens = {
        [](const void*, uint8_t part) -> int16_t { return part < 2 ? kParts[part].sampleLen : -1; },
        nullptr};
    Frame f = {type, seq, payload.data(), uint16_t(payload.size())};
    Msg m;
    Error e = decodeForBridge(f, lens, m);
    CHECK(e == Error::Ok);
    return m;
  }
};

struct Rig {
  FakeHal hal;
  PartSlot slots[2];
  uint8_t rx[64];
  Board<FakeHal> board{hal, kSpec, slots, rx, 64};
  std::vector<std::string> calls;
  Buf lastValue;
  // The clock at the last onSafe.
  uint32_t safeAt = 0;
  // Run inside onCommand, as a sketch's handler would.
  void (*inCommand)(Rig&) = nullptr;
  uint8_t seq = 0;
  uint8_t gen = 0;

  static void onSafe(void* ctx, uint8_t part, uint8_t how, bool estop) {
    Rig& r = *static_cast<Rig*>(ctx);
    char s[48];
    snprintf(s, sizeof s, "safe %u how=%u%s", part, how, estop ? " estop" : "");
    r.calls.push_back(s);
    r.hal.events.push_back(s);
    r.safeAt = r.hal.ms;
  }
  static void onCommand(void* ctx, uint8_t part, const uint8_t* value, uint16_t len, bool hold) {
    Rig& r = *static_cast<Rig*>(ctx);
    char s[48];
    snprintf(s, sizeof s, "cmd %u%s", part, hold ? " hold" : "");
    r.calls.push_back(s);
    r.lastValue.assign(value, value + len);
    if (r.inCommand) r.inCommand(r);
  }

  explicit Rig(bool begin = true) {
    if (begin) start();
  }
  void start(uint16_t watchdogMs = 250) { board.begin(&onSafe, &onCommand, this, watchdogMs); }

  // Everything the board has sent since the last call.
  std::vector<Out> sent() {
    std::vector<Out> out;
    CobsDecoderBuffer<256> d;
    for (uint8_t b : hal.tx) {
      if (d.push(b) != CobsResult::Frame) continue;
      Frame f;
      CHECK(parseFrame(d.data(), d.length(), f) == FrameError::Ok);
      out.push_back(Out{f.type, f.seq, Buf(f.payload, f.payload + f.len)});
    }
    hal.tx.clear();
    return out;
  }
  std::vector<Out> sentOf(uint8_t type) {
    std::vector<Out> all = sent(), some;
    for (const Out& o : all) {
      if (o.type == type) some.push_back(o);
    }
    return some;
  }

  void feed(const Buf& wire) { hal.rx.insert(hal.rx.end(), wire.begin(), wire.end()); }
  template <class M>
  void send(Error (*encode)(const M&, uint8_t, Sink), const M& m) {
    CHECK(encode(m, seq++, sinkTo(*this)) == Error::Ok);
  }
  // `sinkTo(rig)` lands bridge bytes in the board's receive queue.
  void write(uint8_t b) { hal.rx.push_back(b); }

  void attach(uint8_t g, uint8_t flags = 0, uint16_t boot = kBoot) {
    Attach a;
    for (int i = 0; i < 8; ++i) a.boardId[i] = kBoardId[i];
    a.boot = boot;
    a.gen = g;
    a.flags = flags;
    seq = 0;
    send(writeAttach, a);
    gen = g;
  }
  void command(uint8_t part, uint16_t ttl, float a = 0.5f, float b = -0.25f, bool hold = false,
               uint8_t g = 0) {
    uint8_t v[8];
    memcpy(v, &a, 4);
    memcpy(v + 4, &b, 4);
    Command c = {g ? g : gen, part, uint8_t(hold ? command_flags::kHold : 0), ttl,
                 Bytes::ram(v, 8)};
    send(writeCommand, c);
  }
  void safe(uint8_t part) {
    Safe s = {gen, part};
    send(writeSafe, s);
  }
  void ping(uint8_t g = 0) {
    Ping p = {g ? g : gen};
    send(writePing, p);
  }
  void detach() {
    Detach d = {gen};
    send(writeDetach, d);
  }
  void getMap(uint16_t offset, uint8_t max, uint16_t boot = kBoot) {
    GetMap g = {boot, offset, max};
    send(writeGetMap, g);
  }

  void poll() { board.poll(); }
  // Advances the clock by `ms`, polling each millisecond.
  void run(uint32_t ms) {
    for (uint32_t i = 0; i < ms; ++i) {
      ++hal.ms;
      board.poll();
    }
  }
  // Attaches with `g` and drops what the board said.
  void attached(uint8_t g = 7) {
    attach(g);
    poll();
    CHECK(board.attached());
    sent();
    calls.clear();
  }
};

Status statusOf(const Out& o) { return o.msg().status; }

void beginSafesEveryPartAndSaysHello() {
  scenario = "begin";
  Rig r;
  CHECK(r.calls.size() == 2);
  CHECK(r.calls[0] == "safe 0 how=1" && r.calls[1] == "safe 1 how=0");
  std::vector<Out> out = r.sent();
  CHECK(out.size() == 1 && out[0].type == ty::kHello && out[0].seq == 0);
  Hello h = out[0].msg().hello;
  CHECK(h.boot == kBoot && h.gen == 0 && h.flags == 0 && h.arch == 1);
  CHECK(h.mapLen == sizeof kMap - 1 && h.mapCrc == 0xC0FFEE11);
  CHECK(h.watchdogMs == 250 && h.rxMax == 64 && h.libVersion == kLibVersion);
  CHECK(h.name.len == 4 && memcmp(h.name.data, "eyes", 4) == 0);
  CHECK(memcmp(h.boardId, kBoardId, 8) == 0);
  CHECK(!r.hal.armed);
  r.poll();
  CHECK(r.hal.armed && r.hal.feeds == 1);
  // watchdog_ms is clamped into 20..=5000.
  Rig low(false), high(false);
  low.start(5);
  high.start(9000);
  CHECK(low.board.watchdogMs() == 20 && high.board.watchdogMs() == 5000);
}

void helloRepeatsAndOnlyMapAndAttachAreAnswered() {
  scenario = "unattached";
  Rig r;
  r.sent();
  r.run(499);
  CHECK(r.sent().empty());
  r.run(1);
  std::vector<Out> out = r.sent();
  CHECK(out.size() == 1 && out[0].type == ty::kHello && out[0].seq == 1);
  // Before attach: PING, COMMAND, SAFE and DETACH are ignored, not counted.
  r.gen = 9;
  r.ping();
  r.command(0, 50);
  r.safe(0xFF);
  r.detach();
  r.poll();
  CHECK(r.sent().empty() && r.calls.size() == 2 && r.board.bad() == 0);
}

void theMapIsServedInSlicesFencedByBoot() {
  scenario = "map";
  Rig r;
  r.sent();
  const uint16_t len = sizeof kMap - 1;
  r.getMap(0, 10);
  r.getMap(10, 248);
  r.getMap(len, 5);
  r.getMap(0, 10, kBoot + 1);  // another boot: ignored
  r.poll();
  std::vector<Out> out = r.sent();
  CHECK(out.size() == 3);
  MapSlice a = out[0].msg().map, b = out[1].msg().map, c = out[2].msg().map;
  CHECK(a.boot == kBoot && a.offset == 0 && a.data.len == 10);
  CHECK(memcmp(a.data.data, kMap, 10) == 0);
  CHECK(b.offset == 10 && b.data.len == len - 10 && memcmp(b.data.data, kMap + 10, len - 10) == 0);
  CHECK(c.offset == len && c.data.len == 0);
  CHECK(!r.board.attached());
}

void attachChecksIdBootAndGen() {
  scenario = "attach";
  Rig r;
  r.sent();
  r.attach(0);  // gen 0 is always refused
  r.attach(5, 0, kBoot + 1);
  Attach other;
  for (int i = 0; i < 8; ++i) other.boardId[i] = uint8_t(9 - i);
  other.boot = kBoot;
  other.gen = 5;
  other.flags = 0;
  r.send(writeAttach, other);
  r.poll();
  CHECK(!r.board.attached() && r.sentOf(ty::kStatus).empty());
  r.attach(5);
  r.poll();
  std::vector<Out> st = r.sentOf(ty::kStatus);
  CHECK(r.board.attached() && st.size() == 1);
  Status s = statusOf(st[0]);
  CHECK(s.boot == kBoot && s.gen == 5 && s.flags == status_flags::kAttached && s.reset == 3);
  CHECK(s.tUs == r.hal.ms * 1000);
  // A retry of the same gen is confirmed again and feeds the watchdog.
  r.run(200);
  r.sent();
  r.attach(5);
  r.run(100);
  CHECK(r.board.attached());
  CHECK(statusOf(r.sentOf(ty::kStatus).front()).gen == 5);
  // A new gen while attached: every part safes first (none were driving),
  // then the board adopts it.
  r.calls.clear();
  r.command(0, 200);
  r.poll();
  r.attach(6);
  r.poll();
  CHECK(r.board.gen() == 6 && r.calls.size() == 2 && r.calls[1] == "safe 0 how=1");
}

void commandsArePermitsThatExpire() {
  scenario = "permits";
  Rig r;
  r.attached();
  r.command(0, 60, 1.5f, -0.5f, true);
  r.poll();
  CHECK(r.calls.size() == 1 && r.calls[0] == "cmd 0 hold");
  float a, b;
  memcpy(&a, r.lastValue.data(), 4);
  memcpy(&b, r.lastValue.data() + 4, 4);
  CHECK(r.lastValue.size() == 8 && a == 1.5f && b == -0.5f);
  // PING keeps the link up but never extends a permit.
  for (int i = 0; i < 5; ++i) {
    r.run(11);
    r.ping();
  }
  r.run(4);
  CHECK(r.calls.size() == 1);
  r.run(1);
  CHECK(r.calls.size() == 2 && r.calls[1] == "safe 0 how=1");
  CHECK(r.board.attached());
  // The next command resumes the part.
  r.command(0, 60);
  r.poll();
  CHECK(r.calls.back() == "cmd 0");
  // ttl is clamped to [1, watchdog_ms].
  r.calls.clear();
  r.command(0, 0);
  r.poll();
  r.run(1);
  CHECK(r.calls.size() == 2 && r.calls[1] == "safe 0 how=1");
  r.calls.clear();
  r.command(0, 60000);
  r.poll();  // received now: the permit is clamped to 250 ms from here
  for (int i = 0; i < 24; ++i) {
    r.run(10);
    r.ping();
  }
  r.run(9);
  CHECK(r.calls.size() == 1);
  r.run(1);
  CHECK(r.calls.size() == 2 && r.calls[1] == "safe 0 how=1");
}

void safeIsTransientAndReportedOnce() {
  scenario = "safe";
  Rig r;
  r.attached();
  r.command(0, 200);
  r.safe(0);
  r.safe(0);
  r.safe(kEveryPart);
  r.poll();
  CHECK(r.calls.size() == 2 && r.calls[1] == "safe 0 how=1");
  r.command(0, 200);
  r.poll();
  CHECK(r.calls.back() == "cmd 0");
  // A part beyond the map is ignored.
  r.safe(9);
  r.poll();
  CHECK(r.calls.size() == 3);
}

void theGenerationFenceDropsAndCounts() {
  scenario = "fence";
  Rig r;
  r.attached(7);
  r.command(0, 200, 0.5f, 0.5f, false, 8);
  r.poll();
  CHECK(r.calls.empty() && r.board.bad() == 1);
  // A stale-gen PING doesn't feed the link watchdog.
  for (int i = 0; i < 25; ++i) {
    r.run(10);
    r.ping(6);
  }
  r.run(1);
  CHECK(!r.board.attached());
  CHECK(r.board.bad() >= 25);
}

void theLinkWatchdogTripsAndRefusesTheOldGen() {
  scenario = "link watchdog";
  Rig r;
  r.attached(7);
  r.command(0, 200);
  r.poll();
  r.sent();
  r.calls.clear();
  r.run(249);
  CHECK(r.board.attached());
  r.run(1);
  CHECK(!r.board.attached());
  CHECK(r.calls.size() == 1 && r.calls[0] == "safe 0 how=1");
  std::vector<Out> out = r.sent();
  CHECK(!out.empty() && out.back().type == ty::kHello && out.back().msg().hello.gen == 7);
  // A late retry under the old gen can't re-attach; a fresh one can.
  r.attach(7);
  r.poll();
  CHECK(!r.board.attached());
  r.attach(8);
  r.poll();
  CHECK(r.board.attached() && r.board.gen() == 8);
}

void aStalledSketchTripsBeforeActingOnBufferedFrames() {
  scenario = "stall";
  Rig r;
  r.attached(7);
  r.command(0, 200);
  r.poll();
  r.calls.clear();
  // The sketch blocks for 300 ms while a command arrives.
  r.hal.ms += 300;
  r.command(0, 200);
  r.poll();
  CHECK(!r.board.attached());
  CHECK(r.calls.size() == 1 && r.calls[0] == "safe 0 how=1");
}

void detachSafesAndSaysHelloAtOnce() {
  scenario = "detach";
  Rig r;
  r.attached(7);
  r.command(0, 200);
  r.detach();
  r.poll();
  CHECK(!r.board.attached() && r.calls.back() == "safe 0 how=1");
  std::vector<Out> out = r.sent();
  CHECK(!out.empty() && out.back().type == ty::kHello);
}

void statusIsPeriodicWhileAttached() {
  scenario = "status timing";
  Rig r(false);
  r.start(60);  // period min(60 / 3, 100) = 20 ms
  r.attached(7);
  int seen = 0;
  for (int i = 0; i < 10; ++i) {
    r.run(10);
    r.ping();
    seen += int(r.sentOf(ty::kStatus).size());
  }
  CHECK(seen == 5);
  Rig slow;  // 250 ms: period 83 ms
  slow.attached(7);
  slow.run(83);
  CHECK(slow.sentOf(ty::kStatus).size() == 1);
}

void lostAndBadAreCounted() {
  scenario = "counters";
  Rig r;
  r.attached(7);  // ATTACH was seq 0
  r.ping();       // 1
  r.seq = 4;
  r.ping();  // 4: 2 lost
  r.poll();
  CHECK(r.board.lost() == 2);
  // A bad CRC, an empty frame, a truncated block, an overflow and a frame
  // of the wrong length are each one `bad`.
  Buf crc;
  Ping p = {7};
  struct Collect {
    Buf* b;
    void write(uint8_t x) { b->push_back(x); }
  } c = {&crc};
  writePing(p, 9, sinkTo(c));
  crc[2] ^= 0x40;
  r.feed(crc);
  r.feed({0x01, 0x00});
  r.feed({0x05, 0xAA, 0x00});
  Buf big(80, 0x11);
  big[0] = 0x51;  // one block of 80 bytes: more than rx_max
  big.push_back(0);
  r.feed(big);
  Buf len = {0x85, 0, 7, 7};  // PING with two payload bytes
  uint16_t crc16v = crc16(len.data(), uint16_t(len.size()));
  len.push_back(uint8_t(crc16v));
  len.push_back(uint8_t(crc16v >> 8));
  Buf wire;
  struct Out2 {
    Buf* b;
    void write(uint8_t x) { b->push_back(x); }
  } o = {&wire};
  Bytes piece = Bytes::ram(len.data(), uint16_t(len.size()));
  cobsEncode(&piece, 1, sinkTo(o));
  wire.push_back(0);
  r.feed(wire);
  r.poll();
  CHECK(r.board.bad() == 5);
  CHECK(r.board.attached());
  r.run(100);
  std::vector<Out> st = r.sentOf(ty::kStatus);
  CHECK(!st.empty() && statusOf(st.back()).bad == 5 && statusOf(st.back()).lost == 2);
}

void thePinLatchesStoresAfterStatusAndDropsCommands() {
  scenario = "e-stop";
  Rig r;
  r.attached(7);
  r.command(0, 200);
  r.poll();
  r.calls.clear();
  r.hal.events.clear();
  r.hal.input = true;
  r.poll();
  // Safe, then STATUS, then the store.
  CHECK(r.hal.events.size() == 4);
  CHECK(r.hal.events[0] == "safe 0 how=1 estop" && r.hal.events[1] == "safe 1 how=0 estop");
  CHECK(r.hal.events[2] == "tx" && r.hal.events[3] == "store");
  std::vector<Out> st = r.sentOf(ty::kStatus);
  CHECK(st.size() == 1 && statusOf(st[0]).flags == 0x07);
  // Commands are dropped but still feed the link; SAFE changes nothing.
  r.calls.clear();
  for (int i = 0; i < 30; ++i) {
    r.command(0, 200);
    r.safe(kEveryPart);
    r.run(10);
  }
  CHECK(r.calls.empty() && r.board.attached());
  // Releasing the input reports bit 2 but keeps the latch.
  r.sent();
  r.hal.input = false;
  r.poll();
  st = r.sentOf(ty::kStatus);
  CHECK(st.size() == 1 && statusOf(st[0]).flags == 0x03 && r.board.estopLatched());
  // "Clear" needs a new gen.
  r.attach(7, attach_flags::kClearEstop);
  r.poll();
  CHECK(r.board.estopLatched());
  // ...and the input released.
  r.hal.input = true;
  r.poll();
  r.attach(8, attach_flags::kClearEstop);
  r.poll();
  CHECK(r.board.estopLatched() && r.hal.stored);
  r.hal.input = false;
  r.poll();
  r.sent();
  r.calls.clear();
  r.hal.events.clear();
  r.attach(9, attach_flags::kClearEstop);
  r.poll();
  CHECK(!r.board.estopLatched() && !r.hal.stored);
  CHECK(r.hal.events.size() == 4 && r.hal.events[0] == "clear");
  CHECK(r.hal.events[1] == "safe 0 how=1" && r.hal.events[3] == "tx");
  st = r.sentOf(ty::kStatus);
  CHECK(st.size() == 1 && statusOf(st[0]).flags == status_flags::kAttached);
  r.command(0, 200);
  r.poll();
  CHECK(r.calls.back() == "cmd 0");
}

void theLatchSurvivesAReset() {
  scenario = "latch at boot";
  Rig r(false);
  r.hal.stored = true;
  r.start();
  CHECK(r.calls[0] == "safe 0 how=1 estop" && r.calls[1] == "safe 1 how=0 estop");
  std::vector<Out> out = r.sent();
  CHECK(out.size() == 1 && out[0].msg().hello.flags == hello_flags::kEstop);
  // A pressed button at boot latches and stores, whatever was stored.
  Rig pressed(false);
  pressed.hal.input = true;
  pressed.start();
  CHECK(pressed.board.estopLatched() && pressed.hal.stored);
  // No pin: storage is ignored.
  Rig none(false);
  none.hal.estopPin = false;
  none.hal.stored = true;
  none.start();
  CHECK(!none.board.estopLatched() && none.calls[0] == "safe 0 how=1");
  none.attached(3);
  none.command(0, 100);
  none.poll();
  CHECK(none.calls.back() == "cmd 0");
}

void publishAndNotes() {
  scenario = "publish";
  Rig r;
  uint8_t v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  CHECK(!r.board.publish(0, v, 8));  // not attached
  r.attached(7);
  CHECK(!r.board.publish(0, v, 7));
  CHECK(!r.board.publish(2, v, 1));
  CHECK(r.board.publish(0, v, 8) && r.board.publish(1, v, 1));
  std::vector<Out> out = r.sentOf(ty::kState);
  CHECK(out.size() == 2);
  State s = out[0].msg().state;
  CHECK(s.part == 0 && s.tUs == r.hal.ms * 1000 && s.value.len == 8);
  scenario = "notes";
  const char* longText = "0123456789012345678901234567890123456789012345678901234567890";
  CHECK(r.board.note(0, 1, reinterpret_cast<const uint8_t*>(longText), 61));
  CHECK(r.board.note(0, 2, reinterpret_cast<const uint8_t*>("b"), 1));
  CHECK(!r.board.note(0, 3, reinterpret_cast<const uint8_t*>("c"), 1));
  out = r.sentOf(ty::kNote);
  CHECK(out.size() == 2 && out[0].msg().note.text.len == kMaxNote);
  r.hal.ms += 999;
  CHECK(!r.board.note(0, 4, reinterpret_cast<const uint8_t*>("d"), 1));
  r.hal.ms += 1;
  CHECK(r.board.note(0, 5, reinterpret_cast<const uint8_t*>("e"), 1));
}

void seqCountsEveryBoardFrame() {
  scenario = "seq";
  Rig r;
  r.attached(7);
  uint8_t v[1] = {1};
  for (int i = 0; i < 300; ++i) r.board.publish(1, v, 1);
  std::vector<Out> out = r.sent();
  bool consecutive = true;
  for (size_t i = 1; i < out.size(); ++i) {
    consecutive = consecutive && out[i].seq == uint8_t(out[i - 1].seq + 1);
  }
  CHECK(out.size() == 300 && consecutive);
}

void timersSurviveTheClockWrapping() {
  scenario = "wrap";
  Rig r(false);
  r.hal.ms = 0xFFFFFF00u;
  r.start();
  r.attached(7);
  r.command(0, 200);
  r.poll();
  for (int i = 0; i < 40; ++i) {
    r.run(10);
    r.ping();
  }
  CHECK(r.board.attached() && r.hal.ms < 1000);
  CHECK(r.calls.size() == 2 && r.calls[1] == "safe 0 how=1");
}

void aPollFromACallbackDoesNothing() {
  scenario = "re-entry";
  Rig r;
  r.attached(7);
  // The handler waits with tether.delay(): it polls, while a DETACH waits.
  r.inCommand = [](Rig& g) {
    int feeds = g.hal.feeds;
    for (int i = 0; i < 5; ++i) {
      ++g.hal.ms;
      g.board.poll();
    }
    CHECK(g.hal.feeds == feeds);  // not even the watchdog
  };
  r.command(0, 200);
  r.detach();
  r.poll();
  // The DETACH is acted on after the handler returned, never inside it.
  CHECK(r.calls.size() == 2 && r.calls[0] == "cmd 0" && r.calls[1] == "safe 0 how=1");
  CHECK(!r.board.attached());
}

void aFloodCannotHoldAPermitOrThePin() {
  scenario = "flood";
  Rig r;
  r.attached(7);
  r.command(0, 30);
  r.poll();
  uint32_t granted = r.hal.ms;
  r.calls.clear();
  // PINGs arriving as fast as they are read, a millisecond a byte (7 bytes
  // a frame): one poll() expires the permit at the first frame past 30 ms,
  // and returns after kFramesPerPoll frames.
  r.hal.msPerRead = 1;
  for (int i = 0; i < 400; ++i) r.ping();
  r.poll();
  CHECK(r.calls.size() == 1 && r.calls[0] == "safe 0 how=1");
  CHECK(r.safeAt == granted + 35);
  CHECK(r.hal.rx.size() == (400 - 8) * 7);  // kFramesPerPoll
  // The pin is sampled between frames too.
  r.hal.pressAfterReads = 10;
  r.poll();
  CHECK(r.board.estopLatched() && r.hal.stored);
}

void theClearReadsThePinAgain() {
  scenario = "clear re-reads";
  Rig r;
  r.attached(7);
  r.hal.input = true;
  r.poll();
  r.hal.input = false;
  r.poll();
  // The button goes down while the clearing ATTACH is being read, after
  // poll() sampled it: the clear reads the pin itself and refuses.
  r.attach(8, attach_flags::kClearEstop);
  r.hal.pressAfterReads = 3;
  r.poll();
  CHECK(r.board.attached() && r.board.gen() == 8);
  CHECK(r.board.estopLatched() && r.hal.stored);
  for (const std::string& e : r.hal.events) CHECK(e != "clear");
}

void notesSurviveALongSilence() {
  scenario = "note wrap";
  Rig r;
  const uint8_t t[] = {'x'};
  CHECK(r.board.note(0, 0, t, 1) && r.board.note(0, 0, t, 1));
  r.hal.ms += 0x80000000u;  // 24.8 days without a note
  CHECK(r.board.note(0, 0, t, 1));
}

}  // namespace

int main() {
  beginSafesEveryPartAndSaysHello();
  helloRepeatsAndOnlyMapAndAttachAreAnswered();
  theMapIsServedInSlicesFencedByBoot();
  attachChecksIdBootAndGen();
  commandsArePermitsThatExpire();
  safeIsTransientAndReportedOnce();
  theGenerationFenceDropsAndCounts();
  theLinkWatchdogTripsAndRefusesTheOldGen();
  aStalledSketchTripsBeforeActingOnBufferedFrames();
  detachSafesAndSaysHelloAtOnce();
  statusIsPeriodicWhileAttached();
  lostAndBadAreCounted();
  thePinLatchesStoresAfterStatusAndDropsCommands();
  theLatchSurvivesAReset();
  publishAndNotes();
  seqCountsEveryBoardFrame();
  timersSurviveTheClockWrapping();
  aPollFromACallbackDoesNothing();
  aFloodCannotHoldAPermitOrThePin();
  theClearReadsThePinAgain();
  notesSurviveALongSilence();
  if (failures != 0) {
    fprintf(stderr, "%d of %d board checks failed\n", failures, checks);
    return 1;
  }
  printf("ok: %d board checks passed\n", checks);
  return 0;
}
