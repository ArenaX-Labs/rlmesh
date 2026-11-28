#include <cstdio>
#include <cstring>

#include "rlmesh_map.h"

int main() {
  rlmesh::detail::MapWriter map;
  rlmesh::detail::write(map, "hand", rlmesh::Gripper{.hz = 20, .safe = rlmesh::Safe::Stop});
  rlmesh::detail::write(map, "eyes",
                        rlmesh::Joints<2>{
                            .names = {"pan", "tilt"},
                            .hz = 50,
                            .command = rlmesh::Command::Position,
                            .safe = rlmesh::Safe::Hold,
                            .limits = {{-0.6, 0.6}, {-0.4, 0.4}},
                        });
  rlmesh::detail::write(map, "led",
                        rlmesh::Custom<std::array<float, 3>>{
                            .kind = "light/led",
                            .field = "rgb",
                            .hz = 10,
                            .command = "set",
                        });
  const char* want =
      "{\"hand\":{\"kind\":\"gripper\",\"hz\":20.0000,\"command\":\"position\",\"safe\":\"stop\"},"
      "\"eyes\":{\"kind\":\"joints\",\"names\":[\"pan\",\"tilt\"],\"fields\":[\"pos\"],"
      "\"hz\":50.0000,\"command\":\"position\",\"safe\":\"hold\","
      "\"limits\":{\"pos\":[[-0.6000,0.6000],[-0.4000,0.4000]]}},"
      "\"led\":{\"kind\":\"light/led\",\"hz\":10.0000,\"fields\":{\"rgb\":[\"float32\",[3]]},"
      "\"command\":\"set\",\"command_fields\":{\"rgb\":[\"float32\",[3]]},\"safe\":\"hold\"}}";
  String got = map.json();
  if (strcmp(got.c_str(), want) != 0) {
    fprintf(stderr, "map:\n got  %s\n want %s\n", got.c_str(), want);
    return 1;
  }
  return 0;
}
