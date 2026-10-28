// The messages (spec §4): typed views over a frame's payload, and their
// encoding. Lengths are exact; reserved bits must be zero.
//
// `STATE` and `COMMAND` carry a part's dense value, whose length only the
// receiver knows (from the map), so decoding takes it as a lookup.

#ifndef RLMESH_TETHER_MSG_H_
#define RLMESH_TETHER_MSG_H_

#include "bytes.h"
#include "frame.h"

namespace rlmesh {
namespace tether {

// The protocol version this library speaks.
const uint8_t kVersion = 1;
// The largest `MAP` slice.
const uint8_t kMaxMapSlice = 248;
// The longest board name.
const uint8_t kMaxName = 32;
// The longest `NOTE` text.
const uint8_t kMaxNote = 48;
// `part` meaning every part (`SAFE`).
const uint8_t kEveryPart = 0xFF;
// `HELLO.rx_max` bounds: room for `ATTACH` (16 bytes decoded), at most a
// whole frame.
const uint16_t kRxMaxMin = 16;
const uint16_t kRxMaxMax = 256;
// `HELLO.watchdog_ms` bounds (§6.2).
const uint16_t kWatchdogMsMin = 20;
const uint16_t kWatchdogMsMax = 5000;

// Message types.
namespace ty {
// Board → bridge.
const uint8_t kHello = 0x01;
const uint8_t kMap = 0x02;
const uint8_t kState = 0x03;
const uint8_t kStatus = 0x04;
const uint8_t kNote = 0x05;
// Bridge → board.
const uint8_t kGetMap = 0x81;
const uint8_t kAttach = 0x82;
const uint8_t kCommand = 0x83;
const uint8_t kSafe = 0x84;
const uint8_t kPing = 0x85;
const uint8_t kDetach = 0x86;
}  // namespace ty

namespace hello_flags {
// The e-stop pin latch is set.
const uint8_t kEstop = 1 << 0;
const uint8_t kAll = kEstop;
}  // namespace hello_flags

namespace status_flags {
const uint8_t kAttached = 1 << 0;
// E-stop latched by the board's pin.
const uint8_t kEstopLatched = 1 << 1;
// E-stop input active.
const uint8_t kEstopInput = 1 << 2;
const uint8_t kAll = kAttached | kEstopLatched | kEstopInput;
}  // namespace status_flags

namespace attach_flags {
// Clear the pin e-stop latch (with the input released).
const uint8_t kClearEstop = 1 << 0;
const uint8_t kAll = kClearEstop;
}  // namespace attach_flags

namespace command_flags {
// A repeat of the last target (edge Hold).
const uint8_t kHold = 1 << 0;
const uint8_t kAll = kHold;
}  // namespace command_flags

// The board's identification.
struct Hello {
  uint16_t libVersion;
  // 0 unknown, 1 AVR, 2 ARM Cortex-M, 3 RISC-V, 4 Xtensa.
  uint8_t arch;
  uint8_t boardId[8];
  // Differs between boots.
  uint16_t boot;
  // The last attach generation since boot, 0 if none.
  uint8_t gen;
  // `hello_flags`.
  uint8_t flags;
  uint16_t mapLen;
  // CRC-32/ISO-HDLC of the map.
  uint32_t mapCrc;
  uint16_t watchdogMs;
  // The largest decoded frame the board accepts.
  uint16_t rxMax;
  // UTF-8, at most `kMaxName` bytes.
  Bytes name;
};

// A slice of the map from `offset`; empty past the end.
struct MapSlice {
  uint16_t boot;
  uint16_t offset;
  Bytes data;
};

// One part's dense sample.
struct State {
  uint8_t part;
  // `micros()` at capture.
  uint32_t tUs;
  Bytes value;
};

// The board's liveness report.
struct Status {
  uint16_t boot;
  uint8_t gen;
  // `status_flags`.
  uint8_t flags;
  // 0 power-on, 1 external, 2 brown-out, 3 watchdog, 4 other.
  uint8_t reset;
  // Frames lost, from `seq` gaps.
  uint16_t lost;
  // Frames dropped as invalid.
  uint16_t bad;
  // `micros()` just before transmit.
  uint32_t tUs;
};

// A log line: 0 info, 1 warn, 2 error.
struct Note {
  uint8_t level;
  uint8_t code;
  Bytes text;
};

struct GetMap {
  uint16_t boot;
  uint16_t offset;
  // 1..=`kMaxMapSlice`.
  uint8_t max;
};

struct Attach {
  uint8_t boardId[8];
  uint16_t boot;
  // The new generation.
  uint8_t gen;
  // `attach_flags`.
  uint8_t flags;
};

// A drive permit with its target.
struct Command {
  uint8_t gen;
  uint8_t part;
  // `command_flags`.
  uint8_t flags;
  // The permit's life from receipt.
  uint16_t ttlMs;
  Bytes value;
};

struct Safe {
  uint8_t gen;
  // A part, or `kEveryPart`.
  uint8_t part;
};

struct Ping {
  uint8_t gen;
};

struct Detach {
  uint8_t gen;
};

// One message; `type` (a `ty::k…`) says which member holds it. Decoded
// messages borrow their variable parts from the frame.
struct Msg {
  uint8_t type;
  // A `HELLO`'s version byte, with `Error::Version`.
  uint8_t version;
  union {
    Hello hello;
    MapSlice map;
    State state;
    Status status;
    Note note;
    GetMap getMap;
    Attach attach;
    Command command;
    Safe safe;
    Ping ping;
    Detach detach;
  };
};

// Why a payload isn't a valid message, or a message can't be written.
enum class Error : uint8_t {
  Ok,
  UnknownType,
  // A `HELLO` without the magic.
  Magic,
  // A `HELLO` of another protocol version (checked before its length).
  Version,
  // Not the message's exact length.
  Length,
  // A reserved bit is set.
  Reserved,
  // A field out of range (`max` 0, a name too long, `rx_max` or
  // `watchdog_ms` outside its bounds).
  Range,
  // `STATE` or `COMMAND` for a part the receiver doesn't know.
  UnknownPart,
  // A type that travels the other way (§3.2).
  Direction,
  // `HELLO.name` isn't UTF-8.
  Utf8,
};

// Which end is receiving: it decides which types are valid.
enum class Receiver : uint8_t {
  // The computer: board → bridge types (below 0x80).
  Bridge,
  // The board: bridge → board types (0x80 and up).
  Board,
};

// A part's dense value length: the command on the board, the sample on the
// bridge. `lookup` returns a negative length for an unknown part.
struct DenseLen {
  int16_t (*lookup)(const void* ctx, uint8_t part);
  const void* ctx;
};

// Decodes `frame`'s payload as `receiver` gets it.
Error decode(const Frame& frame, Receiver receiver, DenseLen lens, Msg& out);

// `decode` for one end, so a board links only the bridge → board half.
Error decodeForBoard(const Frame& frame, DenseLen lens, Msg& out);
Error decodeForBridge(const Frame& frame, DenseLen lens, Msg& out);

// Writes `msg` as one frame with `seq` to `out` (COBS-encoded, with its
// delimiter). Nothing is written when a field is out of range or the
// payload doesn't fit a frame; that's the error.
Error write(const Msg& msg, uint8_t seq, Sink out);

// `write` for one message type.
Error writeHello(const Hello& m, uint8_t seq, Sink out);
Error writeMap(const MapSlice& m, uint8_t seq, Sink out);
Error writeState(const State& m, uint8_t seq, Sink out);
Error writeStatus(const Status& m, uint8_t seq, Sink out);
Error writeNote(const Note& m, uint8_t seq, Sink out);
Error writeGetMap(const GetMap& m, uint8_t seq, Sink out);
Error writeAttach(const Attach& m, uint8_t seq, Sink out);
Error writeCommand(const Command& m, uint8_t seq, Sink out);
Error writeSafe(const Safe& m, uint8_t seq, Sink out);
Error writePing(const Ping& m, uint8_t seq, Sink out);
Error writeDetach(const Detach& m, uint8_t seq, Sink out);

// Whether `bytes` are well-formed UTF-8 (no overlong forms, surrogates or
// code points past U+10FFFF).
bool validUtf8(const Bytes& bytes);

}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_MSG_H_
