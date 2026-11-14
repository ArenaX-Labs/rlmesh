// CRC-16/CCITT-FALSE for frames (polynomial 0x1021, initial value 0xFFFF,
// no reflection, no final XOR) and CRC-32/ISO-HDLC for the map. Bitwise, so
// a board carries no table.

#ifndef RLMESH_TETHER_CRC_H_
#define RLMESH_TETHER_CRC_H_

#include "bytes.h"

namespace rlmesh {
namespace tether {

// The CRC-16's initial value.
const uint16_t kCrc16Init = 0xFFFF;

// Folds one byte into a running CRC-16 (start from `kCrc16Init`).
uint16_t crc16Update(uint16_t crc, uint8_t byte);

// Folds `bytes` into a running CRC-16.
uint16_t crc16Update(uint16_t crc, const Bytes& bytes);

// The CRC-16 of `len` bytes.
uint16_t crc16(const uint8_t* data, size_t len);

// CRC-32/ISO-HDLC (the zlib CRC): reflected polynomial 0xEDB88320, initial
// value and final XOR 0xFFFFFFFF. `HELLO.map_crc` uses it.
uint32_t crc32(const uint8_t* data, size_t len);

}  // namespace tether
}  // namespace rlmesh

#endif  // RLMESH_TETHER_CRC_H_
