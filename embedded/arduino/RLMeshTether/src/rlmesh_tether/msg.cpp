#include "msg.h"

namespace rlmesh {
namespace tether {

namespace {

const uint8_t kMagic[4] = {'R', 'L', 'M', 'T'};

uint16_t u16At(const uint8_t* p, uint16_t i) {
  return static_cast<uint16_t>(p[i] | static_cast<uint16_t>(static_cast<uint16_t>(p[i + 1]) << 8));
}

uint32_t u32At(const uint8_t* p, uint16_t i) {
  return static_cast<uint32_t>(u16At(p, i)) | (static_cast<uint32_t>(u16At(p, i + 2)) << 16);
}

void putU16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}

void putU32(uint8_t* p, uint32_t v) {
  putU16(p, static_cast<uint16_t>(v));
  putU16(p + 2, static_cast<uint16_t>(v >> 16));
}

void copy8(uint8_t* to, const uint8_t* from) {
  for (uint8_t i = 0; i < 8; ++i) to[i] = from[i];
}

bool reserved(uint8_t flags, uint8_t all) { return (flags & ~all) != 0; }

// The payload is `head` fixed bytes followed by exactly the part's dense
// value, and the part is known.
Error headAnd(const Frame& f, uint16_t head, uint8_t part, DenseLen lens) {
  const int16_t len = lens.lookup(lens.ctx, part);
  if (len < 0) return Error::UnknownPart;
  return f.len >= head && f.len - head == static_cast<uint16_t>(len) ? Error::Ok : Error::Length;
}

Bytes tailOf(const Frame& f, uint16_t from) {
  return Bytes::ram(f.payload + from, static_cast<uint16_t>(f.len - from));
}

// Writes `head[..n] | tail` as one frame, or refuses a payload over a frame.
Error finish(uint8_t type, uint8_t seq, const uint8_t* head, uint8_t n, const Bytes& tail,
             Sink out) {
  if (tail.len > kMaxPayload - n) return Error::Length;
  const Bytes payload[2] = {Bytes::ram(head, n), tail};
  writeFrame(type, seq, payload, 2, out);
  return Error::Ok;
}

const Bytes kNone = {nullptr, 0, false};

}  // namespace

bool validUtf8(const Bytes& s) {
  uint16_t i = 0;
  while (i < s.len) {
    const uint8_t b = s.at(i++);
    if (b < 0x80) continue;
    uint8_t need;
    uint8_t lo = 0x80;
    uint8_t hi = 0xBF;
    if (b >= 0xC2 && b <= 0xDF) {
      need = 1;
    } else if (b == 0xE0) {
      need = 2;
      lo = 0xA0;  // no overlong forms
    } else if (b == 0xED) {
      need = 2;
      hi = 0x9F;  // no surrogates
    } else if (b >= 0xE1 && b <= 0xEF) {
      need = 2;
    } else if (b == 0xF0) {
      need = 3;
      lo = 0x90;  // no overlong forms
    } else if (b >= 0xF1 && b <= 0xF3) {
      need = 3;
    } else if (b == 0xF4) {
      need = 3;
      hi = 0x8F;  // nothing past U+10FFFF
    } else {
      return false;
    }
    for (; need > 0; --need) {
      if (i >= s.len) return false;
      const uint8_t c = s.at(i++);
      if (c < lo || c > hi) return false;
      lo = 0x80;
      hi = 0xBF;
    }
  }
  return true;
}

Error decodeForBridge(const Frame& f, DenseLen lens, Msg& out) {
  if (f.type & 0x80) return Error::Direction;
  const uint8_t* p = f.payload;
  const uint16_t n = f.len;
  out.type = f.type;
  switch (f.type) {
    case ty::kHello: {
      if (n < 4) return Error::Magic;
      for (uint8_t i = 0; i < 4; ++i) {
        if (p[i] != kMagic[i]) return Error::Magic;
      }
      if (n < 5) return Error::Length;
      if (p[4] != kVersion) {
        out.version = p[4];
        return Error::Version;
      }
      out.version = kVersion;
      if (n < 31) return Error::Length;
      const uint8_t nameLen = p[30];
      if (nameLen > kMaxName) return Error::Range;
      if (n != 31 + nameLen) return Error::Length;
      Hello& h = out.hello;
      h.flags = p[19];
      if (reserved(h.flags, hello_flags::kAll)) return Error::Reserved;
      h.watchdogMs = u16At(p, 26);
      h.rxMax = u16At(p, 28);
      if (h.watchdogMs < kWatchdogMsMin || h.watchdogMs > kWatchdogMsMax || h.rxMax < kRxMaxMin ||
          h.rxMax > kRxMaxMax) {
        return Error::Range;
      }
      h.name = tailOf(f, 31);
      if (!validUtf8(h.name)) return Error::Utf8;
      h.libVersion = u16At(p, 5);
      h.arch = p[7];
      copy8(h.boardId, p + 8);
      h.boot = u16At(p, 16);
      h.gen = p[18];
      h.mapLen = u16At(p, 20);
      h.mapCrc = u32At(p, 22);
      return Error::Ok;
    }
    case ty::kMap:
      if (n < 4 || n > 4 + kMaxMapSlice) return Error::Length;
      out.map.boot = u16At(p, 0);
      out.map.offset = u16At(p, 2);
      out.map.data = tailOf(f, 4);
      return Error::Ok;
    case ty::kState: {
      if (n < 1) return Error::Length;
      const Error e = headAnd(f, 5, p[0], lens);
      if (e != Error::Ok) return e;
      out.state.part = p[0];
      out.state.tUs = u32At(p, 1);
      out.state.value = tailOf(f, 5);
      return Error::Ok;
    }
    case ty::kStatus:
      if (n != 13) return Error::Length;
      if (reserved(p[3], status_flags::kAll)) return Error::Reserved;
      out.status.boot = u16At(p, 0);
      out.status.gen = p[2];
      out.status.flags = p[3];
      out.status.reset = p[4];
      out.status.lost = u16At(p, 5);
      out.status.bad = u16At(p, 7);
      out.status.tUs = u32At(p, 9);
      return Error::Ok;
    case ty::kNote:
      if (n < 2 || n > 2 + kMaxNote) return Error::Length;
      out.note.level = p[0];
      out.note.code = p[1];
      out.note.text = tailOf(f, 2);
      return Error::Ok;
    default:
      return Error::UnknownType;
  }
}

Error decodeForBoard(const Frame& f, DenseLen lens, Msg& out) {
  if (!(f.type & 0x80)) return Error::Direction;
  const uint8_t* p = f.payload;
  const uint16_t n = f.len;
  out.type = f.type;
  switch (f.type) {
    case ty::kGetMap:
      if (n != 5) return Error::Length;
      if (p[4] == 0 || p[4] > kMaxMapSlice) return Error::Range;
      out.getMap.boot = u16At(p, 0);
      out.getMap.offset = u16At(p, 2);
      out.getMap.max = p[4];
      return Error::Ok;
    case ty::kAttach:
      if (n != 12) return Error::Length;
      if (reserved(p[11], attach_flags::kAll)) return Error::Reserved;
      copy8(out.attach.boardId, p);
      out.attach.boot = u16At(p, 8);
      out.attach.gen = p[10];
      out.attach.flags = p[11];
      return Error::Ok;
    case ty::kCommand: {
      if (n < 5) return Error::Length;
      const Error e = headAnd(f, 5, p[1], lens);
      if (e != Error::Ok) return e;
      if (reserved(p[2], command_flags::kAll)) return Error::Reserved;
      out.command.gen = p[0];
      out.command.part = p[1];
      out.command.flags = p[2];
      out.command.ttlMs = u16At(p, 3);
      out.command.value = tailOf(f, 5);
      return Error::Ok;
    }
    case ty::kSafe:
      if (n != 2) return Error::Length;
      out.safe.gen = p[0];
      out.safe.part = p[1];
      return Error::Ok;
    case ty::kPing:
      if (n != 1) return Error::Length;
      out.ping.gen = p[0];
      return Error::Ok;
    case ty::kDetach:
      if (n != 1) return Error::Length;
      out.detach.gen = p[0];
      return Error::Ok;
    default:
      return Error::UnknownType;
  }
}

Error decode(const Frame& frame, Receiver receiver, DenseLen lens, Msg& out) {
  return receiver == Receiver::Board ? decodeForBoard(frame, lens, out)
                                     : decodeForBridge(frame, lens, out);
}

Error writeHello(const Hello& m, uint8_t seq, Sink out) {
  // Decode's order after the name length (which frames the payload, so
  // decode checks it first): reserved bits, ranges, UTF-8.
  if (reserved(m.flags, hello_flags::kAll)) return Error::Reserved;
  if (m.name.len > kMaxName || m.watchdogMs < kWatchdogMsMin || m.watchdogMs > kWatchdogMsMax ||
      m.rxMax < kRxMaxMin || m.rxMax > kRxMaxMax) {
    return Error::Range;
  }
  if (!validUtf8(m.name)) return Error::Utf8;
  uint8_t head[31];
  for (uint8_t i = 0; i < 4; ++i) head[i] = kMagic[i];
  head[4] = kVersion;
  putU16(head + 5, m.libVersion);
  head[7] = m.arch;
  copy8(head + 8, m.boardId);
  putU16(head + 16, m.boot);
  head[18] = m.gen;
  head[19] = m.flags;
  putU16(head + 20, m.mapLen);
  putU32(head + 22, m.mapCrc);
  putU16(head + 26, m.watchdogMs);
  putU16(head + 28, m.rxMax);
  head[30] = static_cast<uint8_t>(m.name.len);
  return finish(ty::kHello, seq, head, 31, m.name, out);
}

Error writeMap(const MapSlice& m, uint8_t seq, Sink out) {
  if (m.data.len > kMaxMapSlice) return Error::Length;
  uint8_t head[4];
  putU16(head, m.boot);
  putU16(head + 2, m.offset);
  return finish(ty::kMap, seq, head, 4, m.data, out);
}

Error writeState(const State& m, uint8_t seq, Sink out) {
  uint8_t head[5];
  head[0] = m.part;
  putU32(head + 1, m.tUs);
  return finish(ty::kState, seq, head, 5, m.value, out);
}

Error writeStatus(const Status& m, uint8_t seq, Sink out) {
  if (reserved(m.flags, status_flags::kAll)) return Error::Reserved;
  uint8_t head[13];
  putU16(head, m.boot);
  head[2] = m.gen;
  head[3] = m.flags;
  head[4] = m.reset;
  putU16(head + 5, m.lost);
  putU16(head + 7, m.bad);
  putU32(head + 9, m.tUs);
  return finish(ty::kStatus, seq, head, 13, kNone, out);
}

Error writeNote(const Note& m, uint8_t seq, Sink out) {
  if (m.text.len > kMaxNote) return Error::Length;
  const uint8_t head[2] = {m.level, m.code};
  return finish(ty::kNote, seq, head, 2, m.text, out);
}

Error writeGetMap(const GetMap& m, uint8_t seq, Sink out) {
  if (m.max == 0 || m.max > kMaxMapSlice) return Error::Range;
  uint8_t head[5];
  putU16(head, m.boot);
  putU16(head + 2, m.offset);
  head[4] = m.max;
  return finish(ty::kGetMap, seq, head, 5, kNone, out);
}

Error writeAttach(const Attach& m, uint8_t seq, Sink out) {
  if (reserved(m.flags, attach_flags::kAll)) return Error::Reserved;
  uint8_t head[12];
  copy8(head, m.boardId);
  putU16(head + 8, m.boot);
  head[10] = m.gen;
  head[11] = m.flags;
  return finish(ty::kAttach, seq, head, 12, kNone, out);
}

Error writeCommand(const Command& m, uint8_t seq, Sink out) {
  if (reserved(m.flags, command_flags::kAll)) return Error::Reserved;
  uint8_t head[5];
  head[0] = m.gen;
  head[1] = m.part;
  head[2] = m.flags;
  putU16(head + 3, m.ttlMs);
  return finish(ty::kCommand, seq, head, 5, m.value, out);
}

Error writeSafe(const Safe& m, uint8_t seq, Sink out) {
  const uint8_t head[2] = {m.gen, m.part};
  return finish(ty::kSafe, seq, head, 2, kNone, out);
}

Error writePing(const Ping& m, uint8_t seq, Sink out) {
  return finish(ty::kPing, seq, &m.gen, 1, kNone, out);
}

Error writeDetach(const Detach& m, uint8_t seq, Sink out) {
  return finish(ty::kDetach, seq, &m.gen, 1, kNone, out);
}

Error write(const Msg& m, uint8_t seq, Sink out) {
  switch (m.type) {
    case ty::kHello:
      return writeHello(m.hello, seq, out);
    case ty::kMap:
      return writeMap(m.map, seq, out);
    case ty::kState:
      return writeState(m.state, seq, out);
    case ty::kStatus:
      return writeStatus(m.status, seq, out);
    case ty::kNote:
      return writeNote(m.note, seq, out);
    case ty::kGetMap:
      return writeGetMap(m.getMap, seq, out);
    case ty::kAttach:
      return writeAttach(m.attach, seq, out);
    case ty::kCommand:
      return writeCommand(m.command, seq, out);
    case ty::kSafe:
      return writeSafe(m.safe, seq, out);
    case ty::kPing:
      return writePing(m.ping, seq, out);
    case ty::kDetach:
      return writeDetach(m.detach, seq, out);
    default:
      return Error::UnknownType;
  }
}

}  // namespace tether
}  // namespace rlmesh
