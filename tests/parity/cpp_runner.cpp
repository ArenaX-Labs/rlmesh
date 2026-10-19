// C++17 consumer of the same two-step script as the other three runners.
// Always evaluate checks and calls, including release consumers built with
// NDEBUG.
#define REQUIRE(...)                                                           \
  do {                                                                         \
    if (!(__VA_ARGS__))                                                        \
      abort();                                                                 \
  } while (0)
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <rlmesh.hpp>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
// Observe the real ABI boundary, then forward to the linked library. Recorder
// stores both missing and empty config as {}, so a roundtrip alone cannot catch
// a wrapper accidentally converting an absent optional into an empty dict.
namespace {
bool observe_recorder_config = false;
std::vector<bool> add_configs, capture_configs;
} // namespace
extern "C" RLMeshStatus
rlmesh_recorder_add(RLMeshRecorder *recorder, const RLMeshRunResult *result,
                    const char *model, const char *env,
                    const RLMeshRecorderAddOptions *options,
                    RLMeshWorkloadRecord **out) {
  if (observe_recorder_config)
    add_configs.push_back(options && options->config);
  static auto forward = reinterpret_cast<decltype(&rlmesh_recorder_add)>(
      dlsym(RTLD_NEXT, "rlmesh_recorder_add"));
  REQUIRE(forward);
  return forward(recorder, result, model, env, options, out);
}
extern "C" RLMeshStatus rlmesh_recorder_capture(
    RLMeshRecorder *recorder, const char *model, const char *env,
    const RLMeshRecorderCaptureOptions *options, RLMeshRunHooks **out) {
  if (observe_recorder_config)
    capture_configs.push_back(options && options->config);
  static auto forward = reinterpret_cast<decltype(&rlmesh_recorder_capture)>(
      dlsym(RTLD_NEXT, "rlmesh_recorder_capture"));
  REQUIRE(forward);
  return forward(recorder, model, env, options, out);
}
using namespace rlmesh;
using rlmesh::spaces::Space;
namespace {
void quote(const std::string &text) {
  std::cout << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\')
      std::cout << '\\' << c;
    else if (c < 32) {
      const char *hex = "0123456789abcdef";
      std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
    } else
      std::cout << c;
  }
  std::cout << '"';
}
struct Script {
  int steps = 0, env_closes = 0, model_closes = 0, ends = 0, starts = 0;
  int64_t reset_seed = -1, reset_trial = -1;
  bool dict_action = false;
  bool cancel_step = false;
  std::optional<ext::CancelToken> cancel;
};
struct Counter {
  std::shared_ptr<Script> s;
  std::optional<Metadata> metadata() const {
    return Metadata{
        {std::string(ENV_RESET_OPTIONS_KEY), Value::list({"trial_index"})}};
  }
  Space observation_space() const { return spaces::Discrete(3); }
  Space action_space() const {
    if (s->dict_action) {
      auto child = spaces::Dict({{"k", spaces::Discrete(3).spec()}});
      return spaces::Dict({{"path", child.spec()}});
    }
    return spaces::Discrete(3);
  }
  ResetResult reset(std::optional<int64_t> seed,
                    const std::optional<InfoDict> &options) {
    s->reset_seed = seed.value_or(-1);
    s->reset_trial = trial_index(options).value_or(-1);
    ++s->starts;
    s->steps = 0;
    return {0, {}};
  }
  StepResult step(std::optional<Value>) {
    ++s->starts;
    ++s->steps;
    if (s->cancel_step) {
      s->cancel_step = false;
      s->cancel->cancel();
    }
    return {s->steps, 1.0, s->steps >= 2, false, {}};
  }
  void close() { ++s->env_closes; }
};
[[noreturn]] void fail(const std::string &name) {
#define ERROR(cls)                                                             \
  if (name == #cls)                                                            \
  throw cls("scripted failure")
  ERROR(RLMeshException);
  ERROR(ProtocolException);
  ERROR(EnvironmentException);
  ERROR(RecoverableEnvironmentException);
  ERROR(ModelException);
  ERROR(ValueError);
  ERROR(AdapterResolutionError);
  ERROR(TypeError);
  ERROR(RuntimeError);
  ERROR(OSError);
  ERROR(ConnectionError);
#undef ERROR
  if (name == "TimeoutError")
    throw TimeoutError("authored timeout: checkpoint shard violet");
  if (name == "KeyboardInterrupt")
    throw ext::CancelledError();
  if (name == "NotImplementedError")
    throw RuntimeError("scripted failure");
  if (name == "FileExistsError" || name == "PermissionError")
    throw OSError("scripted failure");
  if (name == "ParamError")
    throw ValueError("scripted failure");
  if (name == "ModelRecoverable")
    throw ModelException("scripted failure", true);
  throw std::logic_error("unknown fixture error");
}
struct PolicyScript {
  std::shared_ptr<Script> s;
  std::string failure;

  Value predict(const Value &) {
    ++s->starts;
    if (!failure.empty())
      fail(failure);
    return 0;
  }
  void on_episode_end() { ++s->ends; }
  void close() { ++s->model_closes; }
};
template <class T> void error(const T &e, const char *name) {
  std::cout << "\"error\":";
  quote(name);
  std::cout << ",\"text\":";
  quote(e.what());
  std::string code;
  bool recoverable = false;
  if constexpr (std::is_base_of_v<EnvironmentException, T> ||
                std::is_base_of_v<ModelException, T>) {
    code = std::string(to_string(e.code()));
    recoverable = e.is_recoverable();
  }
  std::cout << ",\"code\":";
  quote(code);
  std::cout << ",\"recoverable\":" << recoverable << ",\"runtime_base\":"
            << std::is_base_of_v<RuntimeError, T> << ",\"os_base\":"
            << std::is_base_of_v<OSError, T> << ",\"value_base\":"
            << std::is_base_of_v<ValueError, T> << ",\"rlmesh_base\":"
            << std::is_base_of_v<RLMeshException, T> << ",\"environment_base\":"
            << std::is_base_of_v<EnvironmentException, T>;
}
void current_error() {
  try {
    throw;
  } catch (const RecoverableEnvironmentException &e) {
    error(e, "RecoverableEnvironmentException");
  } catch (const EnvironmentException &e) {
    error(e, "EnvironmentException");
  } catch (const ModelException &e) {
    error(e, "ModelException");
  } catch (const ProtocolException &e) {
    error(e, "ProtocolException");
  } catch (const RLMeshException &e) {
    error(e, "RLMeshException");
  } catch (const AdapterResolutionError &e) {
    error(e, "AdapterResolutionError");
  } catch (const ValueError &e) {
    error(e, "ValueError");
  } catch (const TypeError &e) {
    error(e, "TypeError");
  } catch (const RuntimeError &e) {
    error(e, "RuntimeError");
  } catch (const ConnectionError &e) {
    error(e, "ConnectionError");
  } catch (const TimeoutError &e) {
    error(e, "TimeoutError");
  } catch (const OSError &e) {
    error(e, "OSError");
  } catch (const ext::CancelledError &e) {
    error(e, "KeyboardInterrupt");
  }
}

// The current exception's record, with every `value` (a port, a socket
// path) in its text masked.
void current_error_masked(const std::string &value, const std::string &mask) {
  std::ostringstream captured;
  auto *previous = std::cout.rdbuf(captured.rdbuf());
  current_error();
  std::cout.rdbuf(previous);
  std::string text = captured.str();
  for (size_t at = text.find(value); !value.empty() && at != std::string::npos;
       at = text.find(value, at + mask.size()))
    text.replace(at, value.size(), mask);
  std::cout << text;
}
void wait_for_socket(const std::string &path) {
  for (int i = 0; i < 1000; ++i) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof addr.sun_path, "%s", path.c_str());
    int connected =
        connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof addr);
    ::close(fd);
    if (connected == 0)
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::abort();
}
// A served model reached with an api_key, over TLS, or driving a local env.
void remote_auth(const std::string &variant, Env &env, Model &model,
                 const std::shared_ptr<Script> &state) {
  if (variant == "invalid_header") {
    try {
      RemoteModel(RemoteModelOptions{}
                      .with_address("https://localhost:1")
                      .with_api_key("bad\nkey"));
      REQUIRE(false);
    } catch (...) {
      current_error();
    }
    return;
  }
  if (variant == "local_model_https_env") {
    EnvServer server(Counter{std::make_shared<Script>()},
                     EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
    server.start();
    const std::string address = server.address();
    const std::string port = address.substr(address.rfind(':') + 1);
    try {
      model.run("https://127.0.0.1:" + port);
      REQUIRE(false);
    } catch (...) {
      current_error_masked(port, "PORT");
    }
    server.shutdown();
    return;
  }
  if (variant == "https_to_plaintext") {
    EnvServer server(Counter{std::make_shared<Script>()},
                     EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
    server.start();
    const std::string address = server.address();
    const std::string port = address.substr(address.rfind(':') + 1);
    RemoteModel remote(
        RemoteModelOptions{}.with_address("https://127.0.0.1:" + port));
    try {
      run(remote, env);
      REQUIRE(false);
    } catch (...) {
      current_error_masked(port, "PORT");
    }
    server.shutdown();
    return;
  }
  char dir[] = "/tmp/rlmesh-parity-XXXXXX";
  REQUIRE(mkdtemp(dir) != nullptr);
  const std::string path = std::string(dir) + "/model.sock";
  const std::string address = "unix://" + path;
  const bool keyed =
      variant == "api_key_rejected" || variant == "api_key_accepted";
  ext::CancelToken cancel;
  ModelServeOptions serve_options;
  serve_options.cancel = cancel;
  if (keyed)
    serve_options.ext = ext::ModelServeOptions{}.with_token("right-key");
  std::thread serving([&] {
    try {
      model.serve(address, serve_options);
    } catch (...) {
    }
  });
  wait_for_socket(path);
  auto options = RemoteModelOptions{}.with_address(address);
  if (variant == "api_key_rejected")
    options.api_key = "wrong-key";
  else if (variant == "api_key_accepted")
    options.api_key = "right-key";
  RemoteModel remote(options);
  try {
    auto result = run(remote, env);
    std::cout << "\"episodes\":" << result.num_episodes()
              << ",\"steps\":" << result.total_steps()
              << ",\"env_closes\":" << state->env_closes;
  } catch (...) {
    current_error_masked(path, "SOCKET");
  }
  cancel.cancel();
  serving.join();
  ::unlink(path.c_str());
  ::rmdir(dir);
}

struct KeyedPolicy {
  inline static const std::vector<types::LoadKey> load_keys{
      {"checkpoint", true}};
  void load(const Kwargs &) { std::abort(); }
  Value predict(const Value &) { return 0; }
};
struct ChunkScript : PolicyScript {
  Value predict_chunk(const Value &, uint32_t) { return Value::list({0}); }
};
struct Hooks : RunHooks {
  int ends = 0, partial = -1;
  bool fail = false;
  void on_step(const StepEvent &) override {
    if (fail)
      throw TypeError("hook abort");
  }
  void on_run_end(const RunResult &result) override {
    ++ends;
    partial = static_cast<int>(result.num_episodes());
  }
};
struct OwnError : TypeError {
  int tag;
  explicit OwnError(int n) : TypeError("author exception"), tag(n) {}
};
void own_exception(const std::string &variant) {
  if (variant == "threads") {
    Model model([](const Value &, const PredictContext &context) -> Value {
      throw OwnError(static_cast<int>(*context.episode_seed));
    });
    auto work = [&](int tag) {
      Env env(Counter{std::make_shared<Script>()});
      auto session = model.session(env);
      auto [obs, info] = session.reset(tag);
      (void)info;
      try {
        session.predict(obs);
        REQUIRE(false);
      } catch (const OwnError &e) {
        REQUIRE(e.tag == tag);
      }
    };
    std::thread a(work, 31), b(work, 47);
    a.join();
    b.join();
  } else {
    Env env(Counter{std::make_shared<Script>()});
    Model model([](const Value &) -> Value { throw OwnError(17); });
    if (variant == "hook") {
      struct Hook : RunHooks {
        void on_step(const StepEvent &) override { throw OwnError(23); }
      } hook;
      Model zero([](const Value &) { return 0; });
      try {
        zero.run(env, RunOptions{}.with_hooks(&hook));
        REQUIRE(false);
      } catch (const OwnError &e) {
        REQUIRE(e.tag == 23);
      }
    } else {
      auto session = model.session(env);
      auto [obs, info] = session.reset();
      (void)info;
      try {
        session.predict(obs);
        REQUIRE(false);
      } catch (const OwnError &e) {
        REQUIRE(e.tag == 17);
      }
    }
  }
  std::cout << "\"preserved\":true";
}
} // namespace
int main(int argc, char **argv) {
  REQUIRE(argc == 6);
  std::cout << std::boolalpha << '{';
  std::string op = argv[1], variant = argv[2];
  auto state = std::make_shared<Script>();
  state->dict_action = op == "dict_error";
  Env env(Counter{state});
  Model model(PolicyScript{state, op == "error" ? variant : ""});
  try {
    if (op == "dict_error") {
      EnvServer server(Counter{state},
                       EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
      server.start();
      RemoteEnv remote(RemoteEnvOptions{}.with_address(server.address()));
      remote.reset(0);
      remote.step(Value(Metadata{{"path", Value(Metadata{})}}));
      std::abort();
    } else if (op == "recorder_metadata") {
      observe_recorder_config = true;
      Recorder recorder;
      RecorderAddOptions absent;
      auto empty = RecorderAddOptions{}.with_config(Metadata{});
      auto populated =
          RecorderAddOptions{}.with_config(Metadata{{"marker", "violet"}});
      REQUIRE(!absent.config.has_value());
      REQUIRE(empty.config.has_value() && empty.config->empty());
      const auto result = model.run(env);
      auto a = recorder.add(result, "model", "env", absent);
      auto b = recorder.add(result, "model", "env", empty);
      auto c = recorder.add(result, "model", "env", populated);
      auto absent_capture =
          recorder.capture("model", "env", RecorderCaptureOptions{});
      auto empty_options = RecorderCaptureOptions{}.with_config(Metadata{});
      REQUIRE(empty_options.config.has_value());
      auto empty_capture = recorder.capture("model", "env", empty_options);
      model.run(env, RunOptions{}.with_hooks(absent_capture.get()));
      model.run(env, RunOptions{}.with_hooks(empty_capture.get()));
      auto rows = recorder.workloads();
      observe_recorder_config = false;
      const bool abi_presence =
          add_configs == std::vector<bool>{false, true, true} &&
          capture_configs == std::vector<bool>{false, true};
      std::cout << "\"abi_presence\":" << abi_presence << ',';
      std::cout << "\"absent\":" << !absent.config.has_value()
                << ",\"empty\":" << empty.config.has_value()
                << ",\"roundtrip\":"
                << (a.config.empty() && b.config.empty() &&
                    c.config.at("marker").as_text() == "violet" &&
                    rows.size() == 5 && rows[3].config.empty() &&
                    rows[4].config.empty());
    } else if (op == "reset_order") {
      auto session = model.session(env);
      auto [observation, info] = session.reset(37, 11);
      std::cout << "\"seed\":" << state->reset_seed
                << ",\"trial_index\":" << state->reset_trial
                << ",\"observation\":" << observation.as_int()
                << ",\"info_empty\":" << info.empty();
    } else if (op == "config_error") {
      Kwargs config;
      if (variant == "unknown")
        config = {{"checkpoint", 1}, {"scripted", 2}};
      Model::from_config<KeyedPolicy>(config);
      std::abort();
    } else if (op == "validation") {
      if (variant.find("horizon") == 0) {
        model.run(env, RunOptions{}.with_execution_horizon(
                           variant == "horizon_zero" ? 0 : 1025));
      } else {
        View view;
        if (variant == "view_port")
          view.port = 0;
        if (variant == "view_fps")
          view.fps = 0;
        if (variant == "view_quality")
          view.quality = 0;
        if (variant == "view_step_hz")
          view.step_hz = 0;
        view.validate();
      }
      std::abort();
    } else if (op == "run") {
      RunOptions opts;
      opts.close_env = std::stoi(argv[5]) != 0;
      if (std::stoi(argv[3]) >= 0)
        opts.episodes = std::stoi(argv[3]);
      if (std::string(argv[4]) != "none") {
        std::vector<int64_t> seeds;
        std::istringstream input(argv[4]);
        std::string item;
        while (std::getline(input, item, ','))
          seeds.push_back(std::stoll(item));
        opts.seeds = std::move(seeds);
      }
      auto result = model.run(env, opts);
      std::cout << "\"episodes\":" << result.num_episodes()
                << ",\"steps\":" << result.total_steps()
                << ",\"reward\":" << result.mean_reward()
                << ",\"env_closes\":" << state->env_closes
                << ",\"model_closes\":" << state->model_closes;
      Env fresh(Counter{std::make_shared<Script>()});
      model.run(fresh);
      std::cout << ",\"reusable\":true";
    } else if (op == "busy_session") {
      EnvServer server(Counter{state},
                       EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
      server.start();
      {
        auto first = model.session(server);
        first.reset(0);
        try {
          auto second = model.session(server);
          second.reset(0);
          REQUIRE(false);
        } catch (...) {
          current_error();
        }
        first.step(0);
        std::cout << ",\"reusable\":true";
      }
      server.shutdown();
    } else if (op == "remote_auth") {
      remote_auth(variant, env, model, state);
    } else if (op == "busy_remote") {
      EnvServer server(Counter{state},
                       EnvServerOptions{}.with_address("tcp://127.0.0.1:0"));
      server.start();
      RemoteEnv remote(RemoteEnvOptions{}.with_address(server.address()));
      remote.reset(0);
      remote.step(0);
      try {
        model.run(remote);
        REQUIRE(false);
      } catch (...) {
        current_error();
      }
      remote.step(0);
      std::cout << ",\"reusable\":true";
      remote.close();
      server.shutdown();
    } else if (op == "borrow") {
      {
        auto session = model.session(env);
        session.reset(0);
      }
      std::cout << "\"env_closes\":" << state->env_closes
                << ",\"model_closes\":" << state->model_closes
                << ",\"episode_ends\":" << state->ends;
      model.run(env);
      std::cout << ",\"reusable\":true";
    } else if (op == "own_exception") {
      own_exception(variant);
    } else if (op == "factory_start") {
      struct Factory : EnvFactory<Factory> {
        std::shared_ptr<Script> state;
        int prepares = 0;
        explicit Factory(std::shared_ptr<Script> s) : state(std::move(s)) {}
        void prepare() { ++prepares; }
        MadeEnv make(const Kwargs &) { return Env(Counter{state}); }
      };
      int prepares = 0;
      ext::CancelToken token;
      EnvFactoryServeOptions options;
      if (variant == "cancel")
        options.cancel = token;
      auto server = [&] {
        Factory factory(state);
        auto started = ext::start(factory, "tcp://127.0.0.1:0", options);
        prepares = factory.prepares;
        return started;
      }();
      auto result = rlmesh::run(RANDOM_SAMPLE, server);
      if (variant == "cancel") {
        token.cancel();
        bool cancelled = false;
        try {
          server.wait(Duration(5));
        } catch (const ext::CancelledError &) {
          cancelled = true;
        }
        REQUIRE(cancelled && server.wait(Duration(0)));
        std::cout << "\"cancelled\":true,";
      }
      server.shutdown();
      std::cout << "\"episodes\":" << result.num_episodes()
                << ",\"prepares\":" << prepares
                << ",\"env_closes\":" << state->env_closes;
    } else if (op == "factory_cancel" ||
               (op == "drop_serve" &&
                (variant == "factory_cancel" || variant == "factory_idle"))) {
      struct Factory : EnvFactory<Factory> {
        std::shared_ptr<Script> state;
        ext::CancelToken token;
        bool fire;
        int prepares = 0;
        Factory(std::shared_ptr<Script> s, ext::CancelToken t, bool f)
            : state(std::move(s)), token(std::move(t)), fire(f) {}
        void prepare() { ++prepares; }
        MadeEnv make(const Kwargs &) {
          if (fire) {
            fire = false;
            token.cancel();
          }
          return Env(Counter{state});
        }
      };
      ext::CancelToken token;
      const bool serving = variant == "serving" || variant == "factory_cancel";
      Factory factory(state, token, serving);
      if (!serving)
        token.cancel();
      try {
        factory.serve("tcp://127.0.0.1:0",
                      EnvFactoryServeOptions{}.with_cancel(token));
        REQUIRE(false);
      } catch (const ext::CancelledError &e) {
        error(e, "KeyboardInterrupt");
      }
      std::cout << ",\"env_closes\":" << state->env_closes
                << ",\"prepares\":" << factory.prepares;
    } else if (op == "warning") {
      const auto caller = std::this_thread::get_id();
      int warnings = 0;
      bool calling_thread = true;
      std::string category;
      if (variant == "workflow")
        REQUIRE(unsetenv("RLMESH_WORKFLOW_EDITION") == 0);
      set_warning_handler([&](const Warning &warning) {
        calling_thread &= std::this_thread::get_id() == caller;
        ++warnings;
        category = std::string(name(warning.category));
        if (variant == "fail" || variant == "runtime_fail" ||
            variant == "recorder")
          throw TypeError("warning abort");
      });
      try {
        if (variant == "workflow")
          model.run(env);
        else if (variant == "recorder") {
          Recorder recorder;
          auto capture = recorder.capture(
              "model", "env",
              RecorderCaptureOptions{}.with_cameras({"absent"}));
          model.run(env, RunOptions{}.with_hooks(capture.get()));
        } else if (variant.find("runtime") == 0) {
          Model chunk(ChunkScript{{state, ""}});
          chunk.run(env, RunOptions{}.with_execution_horizon(2));
        } else
          model.run(env, RunOptions{}.with_execution_horizon(2));
      } catch (...) {
        current_error();
        std::cout << ',';
      }
      set_warning_handler({});
      std::cout << "\"warnings\":" << warnings << ",\"category\":";
      quote(category);
      std::cout << ",\"calling_thread\":" << calling_thread;
    } else {
      auto token = ext::CancelToken();
      state->cancel = token;
      state->cancel_step = op == "cancel_step" || op == "cancel_run";
      auto session = model.session(env, SessionOptions{}.with_cancel(token));
      Value obs(0);
      if (op != "cancel_run" && op != "hook_error")
        obs = std::get<0>(session.reset(0));
      if (op == "borrow") {
        session.close();
        std::cout << "\"env_closes\":" << state->env_closes
                  << ",\"model_closes\":" << state->model_closes
                  << ",\"episode_ends\":" << state->ends;
        model.run(env);
        std::cout << ",\"reusable\":true";
      } else if (op == "error") {
        session.predict(obs);
        REQUIRE(false);
      } else if (op == "cancel_idle") {
        int failures = 0;
        int cancelled_call_starts = 0;
        for (int i = 0; i < 2; ++i) {
          int before = state->starts;
          token.cancel();
          token.cancel();
          try {
            session.predict(obs);
            REQUIRE(false);
          } catch (const ext::CancelledError &e) {
            ++failures;
            if (!i)
              error(e, "KeyboardInterrupt");
          }
          cancelled_call_starts += state->starts - before;
          session.predict(obs);
        }
        std::cout << ",\"failures\":" << failures
                  << ",\"cancelled_call_starts\":" << cancelled_call_starts
                  << ",\"reusable\":true";
      } else if (op == "cancel_step" || op == "cancel_run" ||
                 op == "hook_error") {
        Hooks hooks;
        hooks.fail = op == "hook_error";
        try {
          if (op == "cancel_step")
            session.step(0);
          else
            session.run(SessionRunOptions{}.with_hooks(&hooks));
          REQUIRE(false);
        } catch (...) {
          current_error();
        }
        std::cout << ",\"episode_ends_before_reset\":" << state->ends;
        if (op == "cancel_step")
          session.reset(0);
        std::cout << ",\"episode_ends\":" << state->ends
                  << ",\"run_ends\":" << hooks.ends
                  << ",\"partial_episodes\":" << hooks.partial;
        session.step(0);
        std::cout << ",\"reusable\":true";
      } else {
        throw std::logic_error("unknown operation " + op);
      }
    }
  } catch (...) {
    current_error();
  }
  std::cout << "}\n";
}
