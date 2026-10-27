// Frames: `type: u8 | seq: u8 | payload | crc: u16 LE`, COBS-encoded and
// ended by 0x00 (spec §3). The CRC covers `type | seq | payload`.

#ifndef RLMESH_TETHER_FRAME_H_
#define RLMESH_TETHER_FRAME_H_

#include "bytes.h"

namespace rlmesh {
namespace tether {

// The largest decoded frame: header, payload and CRC.
const uint16_t kMaxDecoded = 256;
// The largest encoded frame, without its delimiter.
const uint16_t kMaxEncoded = kMaxDecoded + kMaxDecoded / 254 + 1;
// The smallest decoded frame: header, a one-byte payload and CRC.
const uint16_t kMinDecoded = 5;
// The largest payload.
const uint16_t kMaxPayload = kMaxDecoded - 4;
// The most pieces `writeFrame` reads a payload from.
const uint8_t kMaxPayloadPieces = 4;

// One decoded, CRC-checked frame, borrowing the decoder's buffer.
struct Frame {
  // Below 0x80 board → bridge, from 0x80 bridge → board.
  uint8_t type;
  // The sender's frame counter, for loss accounting only.
  uint8_t seq;
  const uint8_t* payload;
  uint16_t len;
};

// Why a decoded frame was refused.
enum class FrameError : uint8_t {
  Ok,
  // Shorter than `kMinDecoded` or longer than `kMaxDecoded`.
  Size,
  // The CRC doesn't match.
  Crc,
};

// Checks and splits a COBS-decoded frame into `out`.
FrameError parseFrame(const uint8_t* decoded, uint16_t len, Frame& out);

// Writes one frame to `out`: CRC first, then COBS straight from the header,
// the payload pieces and the CRC, then the 0x00 delimiter. Writes nothing
// and returns false when the payload is empty, longer than `kMaxPayload`,
// or in more than `kMaxPayloadPieces` pieces.
bool writeFrame(uint8_t type, uint8_t seq, const Bytes* payload, uint8_t count, Sink out);

}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_FRAME_H_
