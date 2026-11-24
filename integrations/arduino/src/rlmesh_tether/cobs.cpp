#include "cobs.h"

namespace rlmesh {
namespace tether {

namespace {

struct Cursor {
  const Bytes* pieces;
  uint8_t count;
  uint8_t piece;
  uint16_t at;

  bool more() {
    while (piece < count && at >= pieces[piece].len) {
      ++piece;
      at = 0;
    }
    return piece < count;
  }

  uint8_t peek() const { return pieces[piece].at(at); }
};

}  // namespace

void cobsEncode(const Bytes* pieces, uint8_t count, Sink out) {
  Cursor c = {pieces, count, 0, 0};
  for (;;) {
    Cursor ahead = c;
    uint8_t run = 0;
    while (run < 254 && ahead.more() && ahead.peek() != 0) {
      ++ahead.at;
      ++run;
    }
    out(static_cast<uint8_t>(run + 1));
    for (uint8_t k = 0; k < run; ++k) {
      c.more();
      out(c.peek());
      ++c.at;
    }
    if (!c.more()) break;
    if (run < 254) ++c.at;
  }
}

void CobsDecoder::reset() {
  len_ = 0;
  left_ = 0;
  zero_ = false;
  started_ = false;
  overflow_ = false;
}

bool CobsDecoder::put(uint8_t byte) {
  if (len_ == cap_) {
    overflow_ = true;
    return false;
  }
  buf_[len_++] = byte;
  return true;
}

CobsResult CobsDecoder::push(uint8_t byte) {
  if (byte == 0) {
    const uint16_t len = len_;
    const uint8_t left = left_;
    const bool started = started_;
    const bool overflow = overflow_;
    reset();
    if (overflow || (started && left != 0)) {
      done_ = 0;
      return overflow ? CobsResult::Overflow : CobsResult::Truncated;
    }
    if (!started) return CobsResult::None;
    done_ = len;
    return CobsResult::Frame;
  }
  if (overflow_) return CobsResult::None;
  started_ = true;
  if (left_ == 0) {
    if (zero_ && !put(0)) return CobsResult::None;
    left_ = static_cast<uint8_t>(byte - 1);
    zero_ = byte != 0xFF;
  } else {
    --left_;
    put(byte);
  }
  return CobsResult::None;
}

}  // namespace tether
}  // namespace rlmesh
