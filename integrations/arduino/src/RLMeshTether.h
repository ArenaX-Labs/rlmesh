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

#endif
