#include "frame.h"

#include "cobs.h"
#include "crc.h"

namespace rlmesh {
namespace tether {

FrameError parseFrame(const uint8_t* decoded, uint16_t len, Frame& out) {
  if (len < kMinDecoded || len > kMaxDecoded) return FrameError::Size;
  const uint16_t body = len - 2;
  const uint16_t want = static_cast<uint16_t>(
      decoded[body] | static_cast<uint16_t>(static_cast<uint16_t>(decoded[body + 1]) << 8));
  if (crc16(decoded, body) != want) return FrameError::Crc;
  out.type = decoded[0];
  out.seq = decoded[1];
  out.payload = decoded + 2;
  out.len = body - 2;
  return FrameError::Ok;
}

bool writeFrame(uint8_t type, uint8_t seq, const Bytes* payload, uint8_t count, Sink out) {
  if (count > kMaxPayloadPieces) return false;
  uint16_t len = 0;
  for (uint8_t i = 0; i < count; ++i) {
    if (payload[i].len > kMaxPayload - len) return false;
    len += payload[i].len;
  }
  if (len == 0) return false;
  const uint8_t head[2] = {type, seq};
  uint16_t crc = crc16Update(crc16Update(kCrc16Init, type), seq);
  for (uint8_t i = 0; i < count; ++i) crc = crc16Update(crc, payload[i]);
  const uint8_t tail[2] = {static_cast<uint8_t>(crc), static_cast<uint8_t>(crc >> 8)};
  Bytes parts[kMaxPayloadPieces + 2];
  parts[0] = Bytes::ram(head, 2);
  for (uint8_t i = 0; i < count; ++i) parts[i + 1] = payload[i];
  parts[count + 1] = Bytes::ram(tail, 2);
  cobsEncode(parts, static_cast<uint8_t>(count + 2), out);
  out(0);
  return true;
}

}  // namespace tether
}  // namespace rlmesh
