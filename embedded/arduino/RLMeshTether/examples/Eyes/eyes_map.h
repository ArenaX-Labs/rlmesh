// What `rlmesh tether gen eyes.json` writes for the map below (spec §9.2),
// by hand until the generator lands. Regenerate it for your own board: a
// `BOARD_ID` must be unique to one board.
//
// {"eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"],
//           "hz": 50, "command": "position", "safe": "hold",
//           "limits": {"pos": [[-0.6, 0.6], [-0.4, 0.4]]}}}

#ifndef EYES_MAP_H_
#define EYES_MAP_H_

#include <RLMeshTether.h>

enum Part : uint8_t { EYES = 0 };
enum SafeBehaviour : uint8_t { SAFE_HOLD = 0 };
const SafeBehaviour EYES_SAFE = SAFE_HOLD;

// Dense, in field order: `pos` for `pan` and `tilt`.
struct EyesCmd {
  float pos[2];
};
struct EyesState {
  float pos[2];
};
static_assert(sizeof(EyesCmd) == 8, "EyesCmd is 8 dense bytes");
static_assert(sizeof(EyesState) == 8, "EyesState is 8 dense bytes");

struct EyesMap {
  typedef ::Part Part;
  typedef ::SafeBehaviour SafeBehaviour;
  static const uint8_t kParts = 1;

  static const rlmesh::tether::BoardSpec& spec() {
    static const uint8_t map[] PROGMEM =
        "{\"eyes\":{\"kind\":\"joints\",\"names\":[\"pan\",\"tilt\"],\"fields\":[\"pos\"],"
        "\"hz\":50,\"command\":\"position\",\"safe\":\"hold\","
        "\"limits\":{\"pos\":[[-0.6,0.6],[-0.4,0.4]]}}}";
    static const uint8_t boardId[8] PROGMEM = {0x3e, 0x91, 0x0c, 0x5a, 0xd2, 0x47, 0x88, 0x1f};
    static const uint8_t name[] PROGMEM = "eyes";
    static const rlmesh::tether::PartSpec parts[kParts] = {{8, 8, EYES_SAFE}};
    static const rlmesh::tether::BoardSpec s = {
        rlmesh::tether::Bytes::progmem(map, sizeof map - 1),
        0x1609BA1Bu,  // CRC-32 of the map
        rlmesh::tether::Bytes::progmem(boardId, 8),
        rlmesh::tether::Bytes::progmem(name, sizeof name - 1),
        parts,
        kParts,
    };
    return s;
  }
};

#endif  // EYES_MAP_H_
