// The C++ gate program of python/rlmesh/tests/conformance/native_gate.py: the
// Gymnasium gate's echo env, echo model and clients, on v2 sessions.
#include "cpp_gate.hpp"

#include <thread>

namespace {
using namespace gate;

int serve_env(const std::string& row, const std::string& address) {
  EnvServer server(EchoEnv(row), EnvServerOptions{}.with_address(address));
  server.start();
  std::printf("listening on %s\n", server.address().c_str());
  std::fflush(stdout);
  wait_for_eof();
  server.shutdown();
  return 0;
}

Model echo_model() {
  return Model([](const Value& observation) { return observation; });
}

int serve_model(const std::string& address) {
  ext::CancelToken stop;
  std::thread([stop] {
    wait_for_eof();
    stop.cancel();
  }).detach();
  std::puts("serving");
  std::fflush(stdout);
  try {
    echo_model().serve(address, ModelServeOptions{}.with_cancel(stop));
  } catch (const ext::CancelledError&) {
  }
  return 0;
}

RunOptions two_episodes() { return RunOptions{}.with_episodes(2).with_seeds({0, 1}); }

int drive(const std::string& verb, const std::string& address, const char* model) {
  if (verb == "run") {
    std::printf("episodes=%zu\n", echo_model().run(address, two_episodes()).num_episodes());
    return 0;
  }
  if (verb == "remote-model") {
    std::printf("episodes=%zu\n",
                Model::from_address(model).run(address, two_episodes()).num_episodes());
    return 0;
  }
  RemoteEnv env(RemoteEnvOptions{}.with_address(address));
  for (int64_t seed = 0; seed < 2; ++seed) {
    auto [observation, info] = env.reset(seed);
    (void)info;
    for (bool done = false; !done;) {
      auto [next, reward, terminated, truncated, step_info] = env.step(observation);
      (void)reward, (void)step_info;
      observation = next;
      done = terminated || truncated;
    }
  }
  env.close();
  std::puts("episodes=2");
  return 0;
}
}  // namespace

int main(int argc, char** argv) try {
  const std::string verb = argc > 1 ? argv[1] : "";
  if (verb == "env" && argc == 4) return serve_env(argv[2], argv[3]);
  if (verb == "model" && argc == 3) return serve_model(argv[2]);
  if ((verb == "remote-env" || verb == "run") && argc == 3) return drive(verb, argv[2], nullptr);
  if (verb == "remote-model" && argc == 4) return drive(verb, argv[3], argv[2]);
  std::fprintf(stderr, "usage: cpp_gate env|model|remote-env|run|remote-model ...\n");
  return 2;
} catch (const Unsupported& unsupported) {
  std::printf("unsupported: %s\n", unsupported.why.c_str());
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
