// Serve Chrono's industrial robot as an RLMesh environment, natively in C++.
//
//   chrono_reach_env [--address HOST:PORT] [--image-size N] [--max-steps N]
//                    [--renderer raytrace|vulkan]
//   chrono_reach_env --describe [--image-size N] [--max-steps N]
//
// --describe prints the env's describe envelope (its spaces, tags, and edition
// handshake) and exits, without binding: bake it as the image's
// dev.rlmesh.describe label (see README.md).
//
// --renderer picks the camera backend for both the observation image and
// render() frames: the CPU ray tracer (default), or an offscreen Vulkan
// rasterizer on the first GPU (or lavapipe) when built with CHRONO_REACH_VULKAN.
//
// The address defaults to $RLMESH_ADDRESS, else 0.0.0.0:50051 (the managed
// platform's convention). $RLMESH_ENV_ENDPOINT_TOKEN, when set, is required on
// every request. SIGINT / SIGTERM drain the server and close the env.
#include <pthread.h>
#include <signal.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "chrono/ChVersion.h"
#include "reach_env.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: chrono_reach_env [--address HOST:PORT] [--image-size N] [--max-steps N]\n"
               "                        [--renderer raytrace|vulkan]\n"
               "       chrono_reach_env --describe [--image-size N] [--max-steps N]\n");
}

}  // namespace

int main(int argc, char** argv) try {
  const char* env_address = std::getenv("RLMESH_ADDRESS");
  std::string address = env_address != nullptr ? env_address : "0.0.0.0:50051";
  chrono_reach::Options options;
  bool describe = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--address" && has_value) {
      address = argv[++i];
    } else if (arg == "--describe") {
      describe = true;
    } else if (arg == "--image-size" && has_value) {
      options.image_size = std::atoi(argv[++i]);
    } else if (arg == "--max-steps" && has_value) {
      options.max_steps = std::atoi(argv[++i]);
    } else if (arg == "--renderer" && has_value) {
      options.renderer = argv[++i];
    } else {
      usage();
      return 2;
    }
  }

  if (options.renderer != "raytrace" && options.renderer != "vulkan") {
    usage();
    return 2;
  }

  // --describe only reads the spaces, so it never needs a GPU.
  std::unique_ptr<chrono_reach::Renderer> renderer;
  try {
    renderer = options.renderer == "vulkan" && !describe ? chrono_reach::make_vulkan_renderer()
                                                         : chrono_reach::make_raytracer();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "failed to start the %s renderer: %s\n", options.renderer.c_str(),
                 error.what());
    return 1;
  }
  const std::string renderer_name = renderer->describe();

  rlmesh::EnvServer server(chrono_reach::IndustrialReach(options, std::move(renderer)),
                           rlmesh::EnvServerOptions{}
                               .with_address(describe ? "tcp://127.0.0.1:0" : address)
                               .with_tags(chrono_reach::IndustrialReach::tags()));
  if (describe) {
    std::puts(rlmesh::ext::env_server_describe_json(server).c_str());
    return 0;
  }
  std::printf("Project Chrono %s IndustrialRobot6dof env (native C++)\n", CHRONO_VERSION);
  std::printf("camera: %s\n", renderer_name.c_str());
  std::printf("listening on %s\n", server.address().c_str());
  std::fflush(stdout);
  server.serve();
  return 0;
} catch (const rlmesh::ext::CancelledError&) {
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
