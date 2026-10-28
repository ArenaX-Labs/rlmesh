// RLMeshTether: a board without a network speaks framed messages over a
// serial link to a computer, and the computer is the RLMesh robot for it
// (spec `tether-spec.md`).
//
// The codec both ends share (CRC, COBS, frames, messages) and the board
// runtime (`board.h`) need no Arduino header, no heap and no transmit
// buffer. In a sketch, `RLMeshTether<Map>` (`arduino.h`) runs the runtime on
// a `Stream`, typed by the header `rlmesh tether gen` writes.

#ifndef RLMESH_TETHER_H_
#define RLMESH_TETHER_H_

#include "rlmesh_tether/board.h"
#include "rlmesh_tether/bytes.h"
#include "rlmesh_tether/cobs.h"
#include "rlmesh_tether/crc.h"
#include "rlmesh_tether/frame.h"
#include "rlmesh_tether/msg.h"

#if defined(ARDUINO)
#include "rlmesh_tether/arduino.h"
using rlmesh::tether::RLMeshTether;
#endif

#endif  // RLMESH_TETHER_H_
