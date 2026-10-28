// Runs the conformance vectors (`crates/rlmesh-tether/tests/vectors.json`,
// schema in `crates/rlmesh-tether/tests/README.md`) against the codec on
// the host: every encoding byte for byte, every decode result and error.
//
//   vectors_test path/to/vectors.json

#include <stdio.h>

#include <string>
#include <vector>

#include "../src/RLMeshTether.h"
#include "json.h"

using namespace rlmesh::tether;

namespace {

typedef std::vector<uint8_t> Buf;

int checks = 0;
int failures = 0;

void check(bool ok, const std::string& name, const std::string& what) {
  ++checks;
  if (ok) return;
  ++failures;
  fprintf(stderr, "FAIL %s: %s\n", name.c_str(), what.c_str());
}

void same(const std::string& got, const std::string& want, const std::string& name) {
  check(got == want, name, "got " + got + ", want " + want);
}

std::string hex(const uint8_t* p, size_t n) {
  static const char digits[] = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += digits[p[i] >> 4];
    s += digits[p[i] & 15];
  }
  return s;
}

std::string hex(const Buf& b) { return hex(b.data(), b.size()); }

std::string hex(const Bytes& b) {
  Buf copy;
  for (uint16_t i = 0; i < b.len; ++i) copy.push_back(b.at(i));
  return hex(copy);
}

[[noreturn]] void bad(const char* what, const std::string& text) {
  fprintf(stderr, "vectors.json: %s: \"%s\"\n", what, text.c_str());
  exit(2);
}

int nibble(char c, const std::string& text) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  bad("not lowercase hex", text);
}

Buf unhex(const json::Value& v) {
  const std::string& s = v.string;
  if (v.kind != json::Value::kString) bad("bytes are not a hex string", s);
  if (s.size() % 2 != 0) bad("odd-length hex", s);
  Buf out;
  for (size_t i = 0; i < s.size(); i += 2)
    out.push_back(static_cast<uint8_t>(nibble(s[i], s) << 4 | nibble(s[i + 1], s)));
  return out;
}

uint64_t num(const json::Value& v) { return v.integer; }

std::string str(uint64_t n) { return std::to_string(static_cast<unsigned long long>(n)); }

// A sink target, through `sinkTo` as an Arduino `Stream` would be.
struct Collect {
  Buf bytes;
  size_t write(uint8_t b) {
    bytes.push_back(b);
    return 1;
  }
};

const char* errorName(Error e) {
  switch (e) {
    case Error::Ok:
      return "ok";
    case Error::UnknownType:
      return "unknown_type";
    case Error::Magic:
      return "magic";
    case Error::Version:
      return "version";
    case Error::Length:
      return "length";
    case Error::Reserved:
      return "reserved";
    case Error::Range:
      return "range";
    case Error::UnknownPart:
      return "unknown_part";
    case Error::Direction:
      return "direction";
    case Error::Utf8:
      return "utf8";
  }
  return "?";
}

struct TypeName {
  const char* name;
  uint8_t type;
};

const TypeName kTypes[] = {
    {"HELLO", ty::kHello},   {"MAP", ty::kMap},         {"STATE", ty::kState},
    {"STATUS", ty::kStatus}, {"NOTE", ty::kNote},       {"GET_MAP", ty::kGetMap},
    {"ATTACH", ty::kAttach}, {"COMMAND", ty::kCommand}, {"SAFE", ty::kSafe},
    {"PING", ty::kPing},     {"DETACH", ty::kDetach},
};

uint8_t typeOf(const std::string& name) {
  for (const TypeName& t : kTypes) {
    if (name == t.name) return t.type;
  }
  fprintf(stderr, "unknown message type %s\n", name.c_str());
  exit(2);
}

// Dense lengths from a vector's `dense_len`; parts not listed are unknown.
struct Lens {
  int16_t len[256];
};

int16_t lookupLen(const void* ctx, uint8_t part) {
  return static_cast<const Lens*>(ctx)->len[part];
}

Lens lensOf(const json::Value& v) {
  Lens l;
  for (int16_t& x : l.len) x = -1;
  for (const auto& m : v["dense_len"].members) {
    char* end = nullptr;
    const long part = strtol(m.first.c_str(), &end, 10);
    if (m.first.empty() || *end != '\0' || part < 0 || part > 255) bad("bad part index", m.first);
    if (m.second.kind != json::Value::kInt || num(m.second) > kMaxPayload)
      bad("bad dense length for part", m.first);
    l.len[part] = static_cast<int16_t>(num(m.second));
  }
  return l;
}

Receiver receiverFor(uint8_t type) { return (type & 0x80) ? Receiver::Board : Receiver::Bridge; }

void boardId(uint8_t* to, const json::Value& v) {
  Buf b = unhex(v);
  for (size_t i = 0; i < 8 && i < b.size(); ++i) to[i] = b[i];
}

// The message a vector describes; `store` holds its variable bytes.
Msg msgOf(const json::Value& v, Buf& store) {
  Msg m;
  m.type = typeOf(v["type"].string);
  m.version = kVersion;
  auto bytes = [&](const char* key) {
    store = unhex(v[key]);
    return Bytes::ram(store.data(), static_cast<uint16_t>(store.size()));
  };
  switch (m.type) {
    case ty::kHello:
      m.hello.libVersion = static_cast<uint16_t>(num(v["lib_version"]));
      m.hello.arch = static_cast<uint8_t>(num(v["arch"]));
      boardId(m.hello.boardId, v["board_id"]);
      m.hello.boot = static_cast<uint16_t>(num(v["boot"]));
      m.hello.gen = static_cast<uint8_t>(num(v["gen"]));
      m.hello.flags = static_cast<uint8_t>(num(v["flags"]));
      m.hello.mapLen = static_cast<uint16_t>(num(v["map_len"]));
      m.hello.mapCrc = static_cast<uint32_t>(num(v["map_crc"]));
      m.hello.watchdogMs = static_cast<uint16_t>(num(v["watchdog_ms"]));
      m.hello.rxMax = static_cast<uint16_t>(num(v["rx_max"]));
      m.hello.name = bytes("name");
      break;
    case ty::kMap:
      m.map.boot = static_cast<uint16_t>(num(v["boot"]));
      m.map.offset = static_cast<uint16_t>(num(v["offset"]));
      m.map.data = bytes("data");
      break;
    case ty::kState:
      m.state.part = static_cast<uint8_t>(num(v["part"]));
      m.state.tUs = static_cast<uint32_t>(num(v["t_us"]));
      m.state.value = bytes("value");
      break;
    case ty::kStatus:
      m.status.boot = static_cast<uint16_t>(num(v["boot"]));
      m.status.gen = static_cast<uint8_t>(num(v["gen"]));
      m.status.flags = static_cast<uint8_t>(num(v["flags"]));
      m.status.reset = static_cast<uint8_t>(num(v["reset"]));
      m.status.lost = static_cast<uint16_t>(num(v["lost"]));
      m.status.bad = static_cast<uint16_t>(num(v["bad"]));
      m.status.tUs = static_cast<uint32_t>(num(v["t_us"]));
      break;
    case ty::kNote:
      m.note.level = static_cast<uint8_t>(num(v["level"]));
      m.note.code = static_cast<uint8_t>(num(v["code"]));
      m.note.text = bytes("text");
      break;
    case ty::kGetMap:
      m.getMap.boot = static_cast<uint16_t>(num(v["boot"]));
      m.getMap.offset = static_cast<uint16_t>(num(v["offset"]));
      m.getMap.max = static_cast<uint8_t>(num(v["max"]));
      break;
    case ty::kAttach:
      boardId(m.attach.boardId, v["board_id"]);
      m.attach.boot = static_cast<uint16_t>(num(v["boot"]));
      m.attach.gen = static_cast<uint8_t>(num(v["gen"]));
      m.attach.flags = static_cast<uint8_t>(num(v["flags"]));
      break;
    case ty::kCommand:
      m.command.gen = static_cast<uint8_t>(num(v["gen"]));
      m.command.part = static_cast<uint8_t>(num(v["part"]));
      m.command.flags = static_cast<uint8_t>(num(v["flags"]));
      m.command.ttlMs = static_cast<uint16_t>(num(v["ttl_ms"]));
      m.command.value = bytes("value");
      break;
    case ty::kSafe:
      m.safe.gen = static_cast<uint8_t>(num(v["gen"]));
      m.safe.part = static_cast<uint8_t>(num(v["part"]));
      break;
    case ty::kPing:
      m.ping.gen = static_cast<uint8_t>(num(v["gen"]));
      break;
    case ty::kDetach:
      m.detach.gen = static_cast<uint8_t>(num(v["gen"]));
      break;
  }
  return m;
}

// Every field of `m`, for comparing a decoded message with its vector.
std::string describe(const Msg& m) {
  std::string s = str(m.type);
  auto f = [&](const char* k, uint64_t v) { s += std::string(" ") + k + "=" + str(v); };
  auto b = [&](const char* k, const Bytes& v) { s += std::string(" ") + k + "=" + hex(v); };
  switch (m.type) {
    case ty::kHello:
      f("lib_version", m.hello.libVersion);
      f("arch", m.hello.arch);
      s += " board_id=" + hex(m.hello.boardId, 8);
      f("boot", m.hello.boot);
      f("gen", m.hello.gen);
      f("flags", m.hello.flags);
      f("map_len", m.hello.mapLen);
      f("map_crc", m.hello.mapCrc);
      f("watchdog_ms", m.hello.watchdogMs);
      f("rx_max", m.hello.rxMax);
      b("name", m.hello.name);
      break;
    case ty::kMap:
      f("boot", m.map.boot);
      f("offset", m.map.offset);
      b("data", m.map.data);
      break;
    case ty::kState:
      f("part", m.state.part);
      f("t_us", m.state.tUs);
      b("value", m.state.value);
      break;
    case ty::kStatus:
      f("boot", m.status.boot);
      f("gen", m.status.gen);
      f("flags", m.status.flags);
      f("reset", m.status.reset);
      f("lost", m.status.lost);
      f("bad", m.status.bad);
      f("t_us", m.status.tUs);
      break;
    case ty::kNote:
      f("level", m.note.level);
      f("code", m.note.code);
      b("text", m.note.text);
      break;
    case ty::kGetMap:
      f("boot", m.getMap.boot);
      f("offset", m.getMap.offset);
      f("max", m.getMap.max);
      break;
    case ty::kAttach:
      s += " board_id=" + hex(m.attach.boardId, 8);
      f("boot", m.attach.boot);
      f("gen", m.attach.gen);
      f("flags", m.attach.flags);
      break;
    case ty::kCommand:
      f("gen", m.command.gen);
      f("part", m.command.part);
      f("flags", m.command.flags);
      f("ttl_ms", m.command.ttlMs);
      b("value", m.command.value);
      break;
    case ty::kSafe:
      f("gen", m.safe.gen);
      f("part", m.safe.part);
      break;
    case ty::kPing:
      f("gen", m.ping.gen);
      break;
    case ty::kDetach:
      f("gen", m.detach.gen);
      break;
  }
  return s;
}

Buf encode(const Bytes* pieces, uint8_t count) {
  Collect c;
  cobsEncode(pieces, count, sinkTo(c));
  return c.bytes;
}

// Feeds `wire` to a decoder of `rxMax` bytes; each result as the vectors
// write it, through `frame` for the decoded bytes.
template <class F>
std::vector<std::string> run(const Buf& wire, size_t rxMax, F frame) {
  Buf buffer(rxMax);
  CobsDecoder d(buffer.data(), static_cast<uint16_t>(rxMax));
  std::vector<std::string> out;
  for (uint8_t b : wire) {
    switch (d.push(b)) {
      case CobsResult::None:
        break;
      case CobsResult::Frame:
        out.push_back(frame(d.data(), d.length()));
        break;
      case CobsResult::Overflow:
        out.push_back("error overflow");
        break;
      case CobsResult::Truncated:
        out.push_back("error truncated");
        break;
    }
  }
  return out;
}

std::string cobsOk(const uint8_t* p, uint16_t n) { return "ok " + hex(p, n); }

std::string frameOk(const uint8_t* p, uint16_t n) {
  Frame f;
  switch (parseFrame(p, n, f)) {
    case FrameError::Ok:
      return "frame " + str(f.type) + " " + str(f.seq) + " " + hex(f.payload, f.len);
    case FrameError::Size:
      return "error size";
    case FrameError::Crc:
      return "error crc";
  }
  return "?";
}

std::string join(const std::vector<std::string>& xs) {
  std::string s = "[";
  for (const std::string& x : xs) s += (s.size() > 1 ? ", " : "") + x;
  return s + "]";
}

std::vector<std::string> wantResults(const json::Value& v) {
  std::vector<std::string> out;
  for (const json::Value& r : v["results"].items) {
    if (r.has("ok")) {
      out.push_back("ok " + r["ok"].string);
    } else if (r.has("frame")) {
      const json::Value& f = r["frame"];
      out.push_back("frame " + str(num(f["type"])) + " " + str(num(f["seq"])) + " " +
                    f["payload"].string);
    } else {
      out.push_back("error " + r["error"].string);
    }
  }
  return out;
}

void crcVectors(const json::Value& all) {
  for (const json::Value& v : all["crc"].items) {
    Buf data = unhex(v["data"]);
    const std::string& name = v["name"].string;
    same(str(crc16(data.data(), data.size())), str(num(v["crc16"])), name + " crc16");
    same(str(crc32(data.data(), data.size())), str(num(v["crc32"])), name + " crc32");
    const Bytes b = Bytes::ram(data.data(), static_cast<uint16_t>(data.size()));
    same(str(crc16Update(kCrc16Init, b)), str(num(v["crc16"])), name + " crc16 running");
  }
}

void cobsVectors(const json::Value& all) {
  for (const json::Value& v : all["cobs_encode"].items) {
    const std::string& name = v["name"].string;
    Buf decoded = unhex(v["decoded"]);
    const std::string want = v["encoded"].string;
    const uint16_t n = static_cast<uint16_t>(decoded.size());
    const Bytes whole = Bytes::ram(decoded.data(), n);
    same(hex(encode(&whole, 1)), want, name);
    // Read ahead across pieces: every split point, with an empty piece.
    for (uint16_t at = 0; at <= n; ++at) {
      const Bytes pieces[3] = {Bytes::ram(decoded.data(), at), Bytes::ram(nullptr, 0),
                               Bytes::progmem(decoded.data() + at, static_cast<uint16_t>(n - at))};
      const std::string got = hex(encode(pieces, 3));
      if (got != want) same(got, want, name + " split at " + str(at));
    }
    Buf wire = unhex(v["encoded"]);
    wire.push_back(0);
    same(join(run(wire, 512, cobsOk)), join({"ok " + v["decoded"].string}), name + " decodes back");
  }
  for (const json::Value& v : all["cobs_decode"].items) {
    same(join(run(unhex(v["wire"]), num(v["rx_max"]), cobsOk)), join(wantResults(v)),
         v["name"].string);
  }
}

void frameVectors(const json::Value& all) {
  for (const json::Value& v : all["frames"].items) {
    same(join(run(unhex(v["wire"]), num(v["rx_max"]), frameOk)), join(wantResults(v)),
         v["name"].string);
  }
}

void messageVectors(const json::Value& all) {
  for (const json::Value& v : all["messages"].items) {
    const std::string& name = v["name"].string;
    const uint8_t seq = static_cast<uint8_t>(num(v["seq"]));
    Buf store;
    const Msg want = msgOf(v["msg"], store);
    // Encode from the fields.
    Collect c;
    const Error e = write(want, seq, sinkTo(c));
    same(errorName(e), "ok", name + " writes");
    same(hex(c.bytes), v["wire"].string, name + " wire");
    // Decode from the wire.
    Buf wire = unhex(v["wire"]);
    CobsDecoderBuffer<kMaxDecoded> d;
    unsigned frames = 0;
    for (uint8_t b : wire) {
      if (d.push(b) != CobsResult::Frame) continue;
      ++frames;
      Frame f;
      check(parseFrame(d.data(), d.length(), f) == FrameError::Ok, name, "parses");
      same(str(f.seq), str(seq), name + " seq");
      const Lens lens = lensOf(v);
      const DenseLen lookup = {lookupLen, &lens};
      Msg got;
      same(errorName(decode(f, receiverFor(f.type), lookup, got)), "ok", name + " decodes");
      same(describe(got), describe(want), name + " fields");
    }
    same(str(frames), "1", name + " frames");
  }
}

void rejectionVectors(const json::Value& all) {
  for (const json::Value& v : all["rejections"].items) {
    const std::string& name = v["name"].string;
    Buf payload = unhex(v["payload"]);
    const Frame f = {static_cast<uint8_t>(num(v["type"])), 0, payload.data(),
                     static_cast<uint16_t>(payload.size())};
    const Receiver r = v["receiver"].string == "board" ? Receiver::Board : Receiver::Bridge;
    const Lens lens = lensOf(v);
    const DenseLen lookup = {lookupLen, &lens};
    Msg got;
    const Error e = decode(f, r, lookup, got);
    same(errorName(e), v["error"].string, name);
    if (v.has("version")) same(str(got.version), str(num(v["version"])), name + " version");
  }
  for (const json::Value& v : all["write_rejections"].items) {
    const std::string& name = v["name"].string;
    Buf store;
    const Msg m = msgOf(v["msg"], store);
    Collect c;
    same(errorName(write(m, 0, sinkTo(c))), v["error"].string, name);
    same(str(c.bytes.size()), "0", name + ": nothing is written on an error");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s vectors.json\n", argv[0]);
    return 2;
  }
  FILE* file = fopen(argv[1], "rb");
  if (!file) {
    perror(argv[1]);
    return 2;
  }
  std::string text;
  char chunk[4096];
  size_t n;
  while ((n = fread(chunk, 1, sizeof chunk, file)) > 0) text.append(chunk, n);
  fclose(file);
  const json::Value all = json::Parser(text).parse();
  if (num(all["schema"]) != 1) {
    fprintf(stderr, "unknown vectors schema\n");
    return 2;
  }
  // Every section must be present and non-empty, and nothing unknown: a
  // section renamed on the Rust side must fail here, not drop coverage.
  static const char* const sections[] = {"crc",
                                         "cobs_encode",
                                         "cobs_decode",
                                         "frames",
                                         "messages",
                                         "rejections",
                                         "write_rejections",
                                         "pack",
                                         "provision",
                                         "provision_rejections",
                                         "provision_write_rejections"};
  for (const char* name : sections) {
    if (all[name].kind != json::Value::kArray || all[name].items.empty()) {
      fprintf(stderr, "vectors section %s is missing or empty\n", name);
      return 2;
    }
  }
  for (const auto& member : all.members) {
    bool known = member.first == "schema";
    for (const char* name : sections) known = known || member.first == name;
    if (!known) {
      fprintf(stderr, "unknown vectors section %s\n", member.first.c_str());
      return 2;
    }
  }
  crcVectors(all);
  cobsVectors(all);
  frameVectors(all);
  messageVectors(all);
  rejectionVectors(all);
  // Dense <-> aligned conversion happens on the bridge, never on the board;
  // provisioning (spec §14) is a networked board's, in Rust.
  const size_t skipped = all["pack"].items.size();
  const size_t provision = all["provision"].items.size() +
                           all["provision_rejections"].items.size() +
                           all["provision_write_rejections"].items.size();
  if (failures) {
    fprintf(stderr, "%d of %d checks failed\n", failures, checks);
    return 1;
  }
  printf(
      "ok: %d checks passed (%zu pack vectors are the bridge's and %zu provisioning vectors a "
      "networked board's, skipped)\n",
      checks, skipped, provision);
  return 0;
}
