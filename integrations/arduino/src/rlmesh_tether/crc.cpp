#include "crc.h"

namespace rlmesh {
namespace tether {

uint16_t crc16Update(uint16_t crc, uint8_t byte) {
  crc = static_cast<uint16_t>(crc ^ static_cast<uint16_t>(static_cast<uint16_t>(byte) << 8));
  for (uint8_t i = 0; i < 8; ++i) {
    crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                         : static_cast<uint16_t>(crc << 1);
  }
  return crc;
}

uint16_t crc16Update(uint16_t crc, const Bytes& bytes) {
  for (uint16_t i = 0; i < bytes.len; ++i) crc = crc16Update(crc, bytes.at(i));
  return crc;
}

uint16_t crc16(const uint8_t* data, size_t len) {
  uint16_t crc = kCrc16Init;
  for (size_t i = 0; i < len; ++i) crc = crc16Update(crc, data[i]);
  return crc;
}

uint32_t crc32(const uint8_t* data, size_t len) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (uint8_t k = 0; k < 8; ++k) crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320UL : crc >> 1;
  }
  return ~crc;
}

}  // namespace tether
}  // namespace rlmesh
