// Serve Chrono's industrial robot as an RLMesh environment, natively in C++.
//
//   chrono_reach_env [--address HOST:PORT] [--image-size N] [--max-steps N]
//
// The address defaults to $RLMESH_ADDRESS, else 0.0.0.0:50051 (the managed
// platform's convention). $RLMESH_ENV_ENDPOINT_TOKEN, when set, is required on
// every request. SIGINT / SIGTERM drain the server and close the env.
#include <pthread.h>
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "chrono/ChVersion.h"
#include "reach_env.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: chrono_reach_env [--address HOST:PORT] [--image-size N] [--max-steps N]\n");
}

}  // namespace

int main(int argc, char** argv) {
  // Block the stop signals before any thread exists, so every thread the
  // runtime spawns inherits the mask and only the waiter below receives them.
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  sigaddset(&stop_signals, SIGINT);
  sigaddset(&stop_signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);

  const char* env_address = std::getenv("RLMESH_ADDRESS");
  std::string address = env_address != nullptr ? env_address : "0.0.0.0:50051";
  chrono_reach::Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--address" && has_value) {
      address = argv[++i];
    } else if (arg == "--image-size" && has_value) {
      options.image_size = std::atoi(argv[++i]);
    } else if (arg == "--max-steps" && has_value) {
      options.max_steps = std::atoi(argv[++i]);
    } else {
      usage();
      return 2;
    }
  }

  auto config = chrono_reach::IndustrialReach::config(options);
  if (!config) {
    std::fprintf(stderr, "invalid env config: %s\n", config.error().message().c_str());
    return 1;
  }
  auto server =
      rlmesh::EnvServer::create(std::make_unique<chrono_reach::IndustrialReach>(options), *config);
  if (!server) {
    std::fprintf(stderr, "failed to create env: %s\n", server.error().message().c_str());
    return 1;
  }
  auto bound = server->bind(address);
  if (!bound) {
    std::fprintf(stderr, "failed to bind %s: %s\n", address.c_str(),
                 bound.error().message().c_str());
    return 1;
  }
  std::printf("Project Chrono %s IndustrialRobot6dof env (native C++)\n", CHRONO_VERSION);
  std::printf("listening on %s\n", bound->c_str());
  std::fflush(stdout);

  rlmesh::EnvServer* handle = &*server;
  std::thread waiter([handle, stop_signals] {
    int signal = 0;
    sigwait(&stop_signals, &signal);
    std::printf("received %s, shutting down\n", signal == SIGINT ? "SIGINT" : "SIGTERM");
    std::fflush(stdout);
    handle->cancel();
  });
  waiter.detach();

  rlmesh::Status served = server->serve();
  if (!served) {
    std::fprintf(stderr, "serve failed: %s\n", served.error().message().c_str());
    return 1;
  }
  return 0;
}
