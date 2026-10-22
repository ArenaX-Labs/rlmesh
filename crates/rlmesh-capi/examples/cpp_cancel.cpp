// X-12: explicit per-operation cancellation through the C and C++ surfaces.
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <rlmesh.hpp>
#include <thread>
#define CHECK(condition)                                                   \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::abort();                                                        \
    }                                                                      \
  } while (false)
using namespace rlmesh;
using namespace std::chrono_literals;
namespace {
struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false, released = false;
  void block() {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    changed.notify_all();
    CHECK(changed.wait_for(lock, 5s, [&] { return released; }));
  }
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    CHECK(changed.wait_for(lock, 5s, [&] { return entered; }));
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    changed.notify_all();
  }
};
struct Script {
  Gate* gate = nullptr;
  bool fail = false;
  ext::CancelToken* fire = nullptr;
  int resets = 0, steps = 0;
  spaces::Space observation_space() { return spaces::Discrete(10); }
  spaces::Space action_space() { return spaces::Discrete(10); }
  ResetResult reset(std::optional<int64_t>, const std::optional<InfoDict>&) {
    ++resets;
    if (fail) throw ValueError("scripted failure");
    return {0, {}};
  }
  StepResult step(std::optional<Value>) {
    ++steps;
    if (gate) gate->block();
    if (fire) fire->cancel();
    return {1, 1., false, false, {}};
  }
};
template <class F>
void cancelled(F&& f) {
  bool caught = false;
  try {
    f();
  } catch (const ext::CancelledError&) {
    caught = true;
  }
  CHECK(caught);
}
struct Factory : EnvFactory<Factory> {
  ext::CancelToken* fire = nullptr;
  bool fail = false;
  Session* reenter = nullptr;
  bool busy_seen = false;
  int builds = 0;
  MadeEnv make(const Kwargs&) {
    ++builds;
    if (reenter) {
      try {
        reenter->reset();
        CHECK(false);
      } catch (const RuntimeError&) {
        busy_seen = true;
      }
    }
    if (fire) fire->cancel();  // Deterministic admission-to-watcher test hook.
    if (fail) throw ValueError("make failed");
    return Env(Script{});
  }
};
void session_calls() {
  auto script = std::make_shared<Script>();
  Env env(script);
  Model model([](const Value&) { return Value(0); });
  auto session = model.session(env);
  ext::CancelToken token, other;
  auto obs = std::get<0>(ext::reset(session, token));
  CHECK(ext::predict(session, token, obs) == Value(0));
  ext::step(session, token, Value(0));
  token.cancel();  // Finished calls must have unsubscribed, despite live handles.
  session.step(Value(0));
  ext::step(session, other, Value(0));
  const auto steps = script->steps;
  cancelled([&] { ext::step(session, token, Value(0)); });
  CHECK(script->steps == steps);
  ext::step(session, token, Value(0));
  // An unrelated failure releases the subscription as well.
  script->fail = true;
  try {
    ext::reset(session, token);
    CHECK(false);
  } catch (const ValueError&) {
  }
  script->fail = false;
  token.cancel();
  cancelled([&] { ext::reset(session, token); });
  ext::reset(session, token);
  // A single reused worker carries no cancellation into its following NULL call.
  Gate gate;
  script->gate = &gate;
  auto worker = std::async(std::launch::async, [&] {
    cancelled([&] { ext::step(session, token, Value(0)); });
    session.step(Value(0));
  });
  gate.wait();
  ext::CancelToken refused;
  refused.cancel();
  RLMeshValue *out = nullptr, *info = nullptr;
  double reward = 0;
  bool terminated = false, truncated = false;
  Value action(0);
  CHECK(rlmesh_ext_session_step(session.raw(), action.raw(), refused.raw(), &out, &reward,
                                &terminated, &truncated, &info) == RLMESH_ERR_RUNTIME);
  token.cancel();
  gate.release();
  worker.get();
  cancelled([&] { ext::step(session, refused, action); });
  ext::step(session, refused, action);
  // Refusal with an unfired token cannot attach that token to the running call.
  Gate normal;
  script->gate = &normal;
  auto first = std::async(std::launch::async, [&] { session.step(action); });
  normal.wait();
  CHECK(rlmesh_ext_session_step(session.raw(), action.raw(), refused.raw(), &out, &reward,
                                &terminated, &truncated, &info) == RLMESH_ERR_RUNTIME);
  refused.cancel();
  normal.release();
  first.get();
  cancelled([&] { ext::step(session, refused, action); });
  script->gate = nullptr;
  // NULL is the ordinary call, and handle cancellation still composes.
  CHECK(rlmesh_ext_session_reset(session.raw(), nullptr, nullptr, nullptr, &out, &info) ==
        RLMESH_OK);
  rlmesh_value_free(out);
  rlmesh_value_free(info);
  auto handle = session.cancel_handle();
  handle.cancel();
  cancelled([&] { ext::predict(session, other, obs); });
  ext::predict(session, other, obs);
}
void lazy_make() {
  ext::CancelToken token;
  Factory factory;
  factory.fire = &token;
  auto s = rlmesh::session(RANDOM_SAMPLE, factory);
  CHECK(factory.builds == 0);
  token.cancel();
  cancelled([&] { ext::reset(s, token); });
  CHECK(factory.builds == 0);                // Idle firing fails before any make.
  cancelled([&] { ext::reset(s, token); });  // Firing inside make survives watcher creation.
  CHECK(factory.builds == 1);
  ext::reset(s, token);
  Factory failing;
  failing.fail = true;
  auto failed = rlmesh::session(RANDOM_SAMPLE, failing);
  try {
    ext::reset(failed, token);
    CHECK(false);
  } catch (const ValueError&) {
  }
  token.cancel();
  failing.fail = false;
  cancelled([&] { ext::reset(failed, token); });
  CHECK(failing.builds == 1);  // Exceptional make released its subscription.
  ext::reset(failed, token);
  Factory cancelled_make;
  cancelled_make.fire = &token;
  cancelled_make.fail = true;
  auto interrupted = rlmesh::session(RANDOM_SAMPLE, cancelled_make);
  cancelled([&] { ext::reset(interrupted, token); });
  cancelled_make.fire = nullptr;
  cancelled_make.fail = false;
  ext::reset(interrupted, token);
  Factory reentrant;
  auto recursive = rlmesh::session(RANDOM_SAMPLE, reentrant);
  reentrant.reenter = &recursive;
  recursive.reset();
  CHECK(reentrant.busy_seen && reentrant.builds == 1);
}
void remote_calls() {
  Gate gate;
  auto script = std::make_shared<Script>();
  script->gate = &gate;
  EnvServer server(Env(script), EnvServerOptions{}.with_address(":0/env"));
  server.start();
  RemoteEnv remote(RemoteEnvOptions{}.with_address(server.address()));
  ext::CancelToken token, refused;
  ext::reset(remote, token, 1, InfoDict{});
  auto call = std::async(std::launch::async,
                         [&] { cancelled([&] { ext::step(remote, token, Value(0)); }); });
  gate.wait();
  Value action(0);
  RLMeshValue *out = nullptr, *info = nullptr;
  double reward = 0;
  bool t = false, tr = false;
  refused.cancel();
  CHECK(rlmesh_ext_remote_env_step(remote.raw(), action.raw(), refused.raw(), &out, &reward, &t,
                                   &tr, &info) == RLMESH_ERR_RUNTIME);
  token.cancel();
  // Remote cancellation returns while the server's callback remains blocked.
  CHECK(call.wait_for(2s) == std::future_status::ready);
  call.get();
  gate.release();
  cancelled([&] { ext::step(remote, refused, action); });
  ext::step(remote, refused, action);
  token.cancel();
  remote.reset();
  cancelled([&] { ext::reset(remote, token); });
  ext::reset(remote, token);
  remote.cancel_handle().cancel();
  cancelled([&] { ext::step(remote, token, action); });
  ext::step(remote, token, action);
  remote.close();
  token.cancel();
  try {
    ext::reset(remote, token);
    CHECK(false);
  } catch (const RuntimeError&) {
  }
  server.shutdown();
  // A refused closed call leaves the firing pending for another handle.
  EnvServer next(Env(Script{}), EnvServerOptions{}.with_address(":0/env"));
  next.start();
  RemoteEnv fresh(RemoteEnvOptions{}.with_address(next.address()));
  cancelled([&] { ext::reset(fresh, token); });
  ext::reset(fresh, token);
  fresh.close();
  next.shutdown();
}
struct VectorScript {
  Gate* gate = nullptr;
  ext::CancelToken* fire = nullptr;
  int num_envs() { return 2; }
  spaces::Space single_observation_space() { return spaces::Discrete(10); }
  spaces::Space single_action_space() { return spaces::Discrete(10); }
  Value batch() {
    const int64_t values[] = {0, 0};
    return Value(Tensor(values, sizeof(values), {2}, DType::Int64));
  }
  ResetResult reset(std::optional<VectorSeed>, const std::optional<InfoDict>&) {
    return {batch(), {}};
  }
  VectorStepResult step(std::optional<Value>) {
    if (gate) gate->block();
    if (fire) fire->cancel();
    return {batch(), {1, 1}, {false, false}, {false, false}, {}};
  }
};
void vector_calls() {
  Gate gate;
  EnvServer server(VectorEnv(VectorScript{&gate}), EnvServerOptions{}.with_address(":0/env"));
  server.start();
  RemoteVectorEnv remote(RemoteEnvOptions{}.with_address(server.address()));
  ext::CancelToken token;
  auto actions =
      std::get<0>(ext::reset(remote, token, VectorSeed(std::vector<int64_t>{1, 2}), InfoDict{}));
  // Capacity refusal touches no outputs and consumes no firing.
  token.cancel();
  RLMeshValue *out = const_cast<RLMeshValue*>(actions.raw()), *info = out;
  double reward = 42;
  bool t = true, tr = true;
  size_t len = 99;
  CHECK(rlmesh_ext_remote_vector_env_step(remote.raw(), actions.raw(), token.raw(), &out, &reward,
                                          &t, &tr, 1, &len, &info) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(out == actions.raw() && info == actions.raw() && reward == 42 && t && tr && len == 99);
  cancelled([&] { ext::step(remote, token, actions); });
  auto call = std::async(std::launch::async,
                         [&] { cancelled([&] { ext::step(remote, token, actions); }); });
  gate.wait();
  token.cancel();
  CHECK(call.wait_for(2s) == std::future_status::ready);
  call.get();
  gate.release();
  ext::reset(remote, token);
  auto result = ext::step(remote, token, actions);
  CHECK(std::get<1>(result).size() == 2);
  token.cancel();
  remote.reset();
  cancelled([&] { ext::reset(remote, token); });
  ext::reset(remote, token);
  remote.close();
  server.shutdown();
}
void immediate_completion() {
  for (bool handle_token : {false, true}) {
    ext::CancelToken per_call;
    ext::CancelToken* fire_predict = nullptr;
    auto script = std::make_shared<Script>();
    Env env(script);
    Model model([&fire_predict](const Value&) {
      if (fire_predict) fire_predict->cancel();
      return Value(0);
    });
    auto session = model.session(env);
    auto observation = std::get<0>(session.reset());
    auto token = handle_token ? session.cancel_handle() : per_call;
    fire_predict = &token;
    cancelled([&] {
      if (handle_token)
        session.predict(observation);
      else
        ext::predict(session, token, observation);
    });
    fire_predict = nullptr;
    session.predict(observation);
    script->fire = &token;
    cancelled([&] {
      if (handle_token)
        session.step(Value(0));
      else
        ext::step(session, token, Value(0));
    });
    script->fire = nullptr;
    session.step(Value(0));

    auto remote_script = std::make_shared<Script>();
    EnvServer server(Env(remote_script), EnvServerOptions{}.with_address(":0/env"));
    server.start();
    RemoteEnv remote(RemoteEnvOptions{}.with_address(server.address()));
    remote.reset();
    auto remote_token = handle_token ? remote.cancel_handle() : per_call;
    remote_script->fire = &remote_token;
    cancelled([&] {
      if (handle_token)
        remote.step(Value(0));
      else
        ext::step(remote, remote_token, Value(0));
    });
    remote_script->fire = nullptr;
    remote.step(Value(0));
    remote.close();
    server.shutdown();

    auto vector_script = std::make_shared<VectorScript>();
    EnvServer vector_server(VectorEnv(vector_script), EnvServerOptions{}.with_address(":0/env"));
    vector_server.start();
    RemoteVectorEnv vector(RemoteEnvOptions{}.with_address(vector_server.address()));
    auto actions = std::get<0>(vector.reset());
    auto vector_token = handle_token ? vector.cancel_handle() : per_call;
    vector_script->fire = &vector_token;
    cancelled([&] {
      if (handle_token)
        vector.step(actions);
      else
        ext::step(vector, vector_token, actions);
    });
    vector_script->fire = nullptr;
    vector.step(actions);
    vector.close();
    vector_server.shutdown();
  }
}
void shared_server_token() {
  // X-13 deferred test: one firing reaches the server and an X-12 call.
  ext::CancelToken token;
  Factory factory;
  auto server = ext::start(factory, ":0/env", EnvFactoryServeOptions{}.with_cancel(token));
  Gate gate;
  auto script = std::make_shared<Script>();
  script->gate = &gate;
  Env env(script);
  auto session = rlmesh::session(RANDOM_SAMPLE, env);
  session.reset();
  auto call = std::async(std::launch::async,
                         [&] { cancelled([&] { ext::step(session, token, Value(0)); }); });
  gate.wait();
  token.cancel();
  gate.release();
  call.get();
  cancelled([&] { server.wait(2s); });
  // Both have released their subscription; the next idle firing is pending.
  token.cancel();
  cancelled([&] { ext::reset(session, token); });
  ext::reset(session, token);
}
}  // namespace
int main() {
  session_calls();
  lazy_make();
  remote_calls();
  vector_calls();
  immediate_completion();
  shared_server_token();
}
