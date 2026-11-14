// The ESP32's C++/Rust boundary against the shared USB provisioning vectors.
#include <stdio.h>

#include <string>
#include <vector>

#include "../src/RLMeshTether.h"
#include "json.h"
#include "rlmesh_device.h"

using namespace rlmesh::tether;
typedef std::vector<uint8_t> Buf;

static int checks = 0;
static void check(bool ok, const std::string& name) {
  ++checks;
  if (!ok) {
    fprintf(stderr, "FAIL provisioning: %s\n", name.c_str());
    exit(1);
  }
}

static Buf unhex(const json::Value& v) {
  check(v.kind == json::Value::kString && v.string.size() % 2 == 0, "hex string");
  Buf out;
  for (size_t i = 0; i < v.string.size(); i += 2)
    out.push_back(static_cast<uint8_t>(strtoul(v.string.substr(i, 2).c_str(), nullptr, 16)));
  return out;
}

static void bytes(Buf& out, const json::Value& v, bool length) {
  Buf b = unhex(v);
  if (length) out.push_back(static_cast<uint8_t>(b.size()));
  out.insert(out.end(), b.begin(), b.end());
}

// Independent construction in C++ pins the record layout, including its version.
static Buf record(const json::Value& msg) {
  Buf out = {2, static_cast<uint8_t>(msg["flags"].integer)};
  bytes(out, msg["host_key"], false);
  bytes(out, msg["host_id"], true);
  bytes(out, msg["name"], true);
  out.push_back(static_cast<uint8_t>(msg["addresses"].items.size()));
  for (const auto& address : msg["addresses"].items) bytes(out, address, true);
  return out;
}

struct Collect {
  Buf bytes;
  size_t write(uint8_t b) {
    bytes.push_back(b);
    return 1;
  }
};

static Buf framed(const Buf& payload, uint8_t seq) {
  Collect out;
  out.bytes.push_back(0);
  const Bytes part = Bytes::ram(payload.data(), static_cast<uint16_t>(payload.size()));
  writeFrame(0x91, seq, &part, 1, sinkTo(out));
  return out.bytes;
}

static void store(const Buf& wire, const Buf& payload, const std::string& name) {
  uint8_t key[32] = {7};
  rlmesh_device_provision_t* responder = nullptr;
  check(rlmesh_device_provision_new(key, 4, nullptr, 0, &responder) == RLMESH_DEVICE_OK, name);
  int stores = 0;
  for (uint8_t byte : wire) {
    const uint8_t* data = nullptr;
    size_t len = 0;
    int step = rlmesh_device_provision_push(responder, byte, &data, &len);
    if (step == RLMESH_DEVICE_PROVISION_NONE) continue;
    check(step == RLMESH_DEVICE_PROVISION_STORE, name);
    check(Buf(data, data + len) == payload, name + ": NVS blob");
    ++stores;
    check(rlmesh_device_provision_stored(responder, 1, &data, &len) == RLMESH_DEVICE_OK, name);
    CobsDecoderBuffer<256> decoder;
    bool confirmed = false;
    for (size_t i = 0; i < len; ++i) {
      const auto result = decoder.push(data[i]);
      if (result != CobsResult::Frame) continue;
      Frame frame;
      check(parseFrame(decoder.data(), decoder.length(), frame) == FrameError::Ok, name);
      check(frame.type == 0x11 && frame.len == 5 && frame.payload[0] == 0, name);
      uint32_t crc = 0;
      for (uint8_t j = 0; j < 4; ++j) crc |= uint32_t(frame.payload[j + 1]) << (8 * j);
      check(crc == crc32(payload.data(), payload.size()), name + ": record CRC");
      confirmed = true;
    }
    check(confirmed, name + ": confirmation");
  }
  check(stores == 1, name + ": one atomic store");
  rlmesh_device_provision_free(responder);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  FILE* file = fopen(argv[1], "rb");
  if (!file) return 2;
  std::string text;
  char chunk[4096];
  size_t n;
  while ((n = fread(chunk, 1, sizeof chunk, file)) > 0) text.append(chunk, n);
  fclose(file);
  const json::Value all = json::Parser(text).parse();
  check(all["schema"].integer == 1, "schema");
  int records = 0;
  for (const auto& v : all["provision"].items) {
    if (v["msg"]["type"].string != "PROVISION") continue;
    Buf payload = record(v["msg"]);
    check(rlmesh_device_record_valid(payload.data(), payload.size()) == 1, v["name"].string);
    Buf wire = framed(payload, static_cast<uint8_t>(v["seq"].integer));
    check(wire == unhex(v["wire"]), v["name"].string + ": C++ encoding");
    store(wire, payload, v["name"].string);
    ++records;
  }
  check(records > 0, "record vectors present");
  int rejected = 0;
  for (const auto& v : all["provision_rejections"].items) {
    if (v["type"].integer != 0x91 || v["receiver"].string != "board") continue;
    const Buf payload = unhex(v["payload"]);
    check(rlmesh_device_record_valid(payload.data(), payload.size()) == 0, v["name"].string);
    ++rejected;
  }
  for (const auto& v : all["provision_write_rejections"].items) {
    if (v["msg"]["type"].string != "PROVISION") continue;
    const Buf payload = record(v["msg"]);
    check(rlmesh_device_record_valid(payload.data(), payload.size()) == 0, v["name"].string);
    ++rejected;
  }
  check(rejected > 0, "rejection vectors present");
  printf("ok: %d ESP32 provisioning checks passed\n", checks);
}
