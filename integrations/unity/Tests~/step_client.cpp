#include "rlmesh.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  try {
    rlmesh::Model model([](const rlmesh::Value &) { return rlmesh::Value(0); });
    model.run(argv[1], rlmesh::RunOptions{}.with_episodes(1));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
