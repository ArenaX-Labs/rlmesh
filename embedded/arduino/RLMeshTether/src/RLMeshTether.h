// RLMeshTether: a board without a network speaks framed messages over a
// serial link to a computer, and the computer is the RLMesh robot for it
// (spec `tether-spec.md`).
//
// This release is the codec both ends share: CRC (`crc.h`), COBS
// (`cobs.h`), frames (`frame.h`) and messages (`msg.h`). It needs no
// Arduino header, no heap and no transmit buffer.

#ifndef RLMESH_TETHER_H_
#define RLMESH_TETHER_H_

#include "rlmesh_tether/bytes.h"
#include "rlmesh_tether/cobs.h"
#include "rlmesh_tether/crc.h"
#include "rlmesh_tether/frame.h"
#include "rlmesh_tether/msg.h"

#endif  // RLMESH_TETHER_H_
