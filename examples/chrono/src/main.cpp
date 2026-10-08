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

// The stop signals, plus the signal that wakes the waiter when serve returns
// on its own. main blocks them all before any thread exists.
sigset_t waited_signals() {
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  sigaddset(&set, SIGUSR1);
  return set;
}

// Waits for SIGINT / SIGTERM and cancels the server. Destroying it wakes the
// thread and joins it, so it never outlives the server it cancels.
class SignalWaiter {
 public:
  explicit SignalWaiter(rlmesh::EnvServer& server) : thread_([this, &server] { wait(server); }) {}
  ~SignalWaiter() {
    done_.store(true);
    // A signal sent to the thread itself, so it reaches this waiter's sigwait
    // and no other. Harmless if the thread already returned.
    pthread_kill(thread_.native_handle(), SIGUSR1);
    thread_.join();
  }
  SignalWaiter(const SignalWaiter&) = delete;
  SignalWaiter& operator=(const SignalWaiter&) = delete;

 private:
  void wait(rlmesh::EnvServer& server) {
    const sigset_t set = waited_signals();
    for (;;) {
      int signal = 0;
      if (sigwait(&set, &signal) != 0) continue;
      if (signal == SIGUSR1) {
        if (done_.load()) return;
        continue;  // a stray SIGUSR1 from outside: not ours
      }
      std::printf("received %s, shutting down\n", signal == SIGINT ? "SIGINT" : "SIGTERM");
      std::fflush(stdout);
      server.cancel();
      return;
    }
  }

  std::atomic<bool> done_{false};
  std::thread thread_;
};

}  // namespace

int main(int argc, char** argv) {
  // Block the stop signals before any thread exists, so every thread the
  // runtime spawns inherits the mask and only the waiter below receives them.
  const sigset_t signals = waited_signals();
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);

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

  auto config = chrono_reach::IndustrialReach::config(options);
  if (!config) {
    std::fprintf(stderr, "invalid env config: %s\n", config.error().message().c_str());
    return 1;
  }
  auto server = rlmesh::EnvServer::create(
      std::make_unique<chrono_reach::IndustrialReach>(options, std::move(renderer)), *config);
  if (!server) {
    std::fprintf(stderr, "failed to create env: %s\n", server.error().message().c_str());
    return 1;
  }
  if (describe) {
    auto envelope = server->describe_json();
    if (!envelope) {
      std::fprintf(stderr, "failed to describe: %s\n", envelope.error().message().c_str());
      return 1;
    }
    std::printf("%s\n", envelope->c_str());
    return 0;
  }
  auto bound = server->bind(address);
  if (!bound) {
    std::fprintf(stderr, "failed to bind %s: %s\n", address.c_str(),
                 bound.error().message().c_str());
    return 1;
  }
  std::printf("Project Chrono %s IndustrialRobot6dof env (native C++)\n", CHRONO_VERSION);
  std::printf("camera: %s\n", renderer_name.c_str());
  std::printf("listening on %s\n", bound->c_str());
  std::fflush(stdout);

  rlmesh::Status served = [&] {
    SignalWaiter waiter(*server);
    return server->serve();
  }();  // the waiter is joined here, on every way serve returns
  if (!served) {
    std::fprintf(stderr, "serve failed: %s\n", served.error().message().c_str());
    return 1;
  }
  return 0;
}
