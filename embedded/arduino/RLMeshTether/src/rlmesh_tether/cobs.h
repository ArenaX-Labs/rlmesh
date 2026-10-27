// Consistent Overhead Byte Stuffing: a frame's bytes with every 0x00
// removed, so 0x00 can delimit frames on the wire.
//
// `cobsEncode` reads its source as a chain of pieces and looks ahead in
// place, so a frame is encoded straight from its header, value and CRC:
// no transmit buffer. `CobsDecoder` takes one byte at a time into a buffer
// the caller sizes (the board's `rx_max`).

#ifndef RLMESH_TETHER_COBS_H_
#define RLMESH_TETHER_COBS_H_

#include "bytes.h"

namespace rlmesh {
namespace tether {

// The largest encoding of `len` bytes, without the delimiter.
constexpr size_t cobsMaxEncoded(size_t len) { return len + len / 254 + 1; }

// Writes the COBS encoding of `pieces` (read as one sequence) to `out`,
// without the trailing 0x00.
//
// The form is pinned (spec §3.1): no code byte follows a final 254-byte
// block. `CobsDecoder` also accepts the variant that appends 0x01 there.
void cobsEncode(const Bytes* pieces, uint8_t count, Sink out);

// What a byte pushed into a `CobsDecoder` completed.
enum class CobsResult : uint8_t {
  // Nothing yet, or an empty frame (two delimiters in a row).
  None,
  // A frame: `data()` and `length()` hold it until the next push.
  Frame,
  // More than the buffer holds arrived before the delimiter; everything up
  // to and including it was discarded.
  Overflow,
  // The delimiter came inside a block: the frame was cut short.
  Truncated,
};

// A byte-at-a-time decoder into a caller's buffer. Every 0x00 ends the
// current frame, valid or not, and the next byte starts a new one, so a bad
// frame never costs the frame after it.
class CobsDecoder {
 public:
  CobsDecoder(uint8_t* buffer, uint16_t capacity) : buf_(buffer), cap_(capacity), done_(0) {
    reset();
  }

  // Feeds one wire byte.
  CobsResult push(uint8_t byte);

  // Drops any partial frame.
  void reset();

  // The last frame (after `CobsResult::Frame`).
  const uint8_t* data() const { return buf_; }
  uint16_t length() const { return done_; }

 private:
  bool put(uint8_t byte);

  uint8_t* buf_;
  uint16_t cap_;
  uint16_t len_;
  // The finished frame's length.
  uint16_t done_;
  // Data bytes left in the current block.
  uint8_t left_;
  // The current block was shorter than 254 bytes, so a zero follows it
  // unless the frame ends first.
  bool zero_;
  // A byte of this frame has arrived.
  bool started_;
  bool overflow_;
};

// A `CobsDecoder` with its own buffer of `N` bytes.
template <uint16_t N>
class CobsDecoderBuffer : public CobsDecoder {
 public:
  CobsDecoderBuffer() : CobsDecoder(storage_, N) {}

 private:
  uint8_t storage_[N];
};

}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_COBS_H_
