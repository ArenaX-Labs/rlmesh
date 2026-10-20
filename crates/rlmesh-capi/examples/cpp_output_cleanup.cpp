// Fail shared_ptr allocation after C returned two owned Value outputs.
#define rlmesh_ext_session_reset test_ext_reset
#define rlmesh_ext_session_step test_ext_step
#define rlmesh_session_reset test_session_reset
#define rlmesh_session_step test_session_step
#define rlmesh_session_free test_session_free
#define rlmesh_value_free test_value_free
#include <cstdlib>
#include <new>
#include <rlmesh.hpp>
#define CHECK(condition)            \
  do {                              \
    if (!(condition)) std::abort(); \
  } while (false)
namespace {
int fail_after = -1, observation_frees = 0, info_frees = 0;
unsigned char session_storage, observation_storage, info_storage, action_storage;
RLMeshValue* observation() { return reinterpret_cast<RLMeshValue*>(&observation_storage); }
RLMeshValue* info() { return reinterpret_cast<RLMeshValue*>(&info_storage); }
}  // namespace
void* operator new(std::size_t size) {
  if (fail_after == 0) {
    fail_after = -1;
    throw std::bad_alloc();
  }
  if (fail_after > 0) --fail_after;
  if (void* p = std::malloc(size)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
extern "C" void test_value_free(RLMeshValue* p) {
  if (p == observation())
    ++observation_frees;
  else if (p == info())
    ++info_frees;
}
extern "C" void test_session_free(RLMeshSession*) {}
extern "C" RLMeshStatus test_session_reset(RLMeshSession*, const int64_t*, const int64_t*,
                                           RLMeshValue** o, RLMeshValue** i) {
  *o = observation();
  *i = info();
  return RLMESH_OK;
}
extern "C" RLMeshStatus test_session_step(RLMeshSession*, const RLMeshValue*, RLMeshValue** o,
                                          double* reward, bool* term, bool* trunc,
                                          RLMeshValue** i) {
  *reward = 0;
  *term = false;
  *trunc = false;
  *o = observation();
  *i = info();
  return RLMESH_OK;
}
extern "C" RLMeshStatus test_ext_reset(RLMeshSession* session, const int64_t* seed,
                                       const int64_t* trial, const RLMeshExtCancelToken*,
                                       RLMeshValue** o, RLMeshValue** i) {
  return test_session_reset(session, seed, trial, o, i);
}
extern "C" RLMeshStatus test_ext_step(RLMeshSession* session, const RLMeshValue* action,
                                      const RLMeshExtCancelToken*, RLMeshValue** o, double* reward,
                                      bool* term, bool* trunc, RLMeshValue** i) {
  return test_session_step(session, action, o, reward, term, trunc, i);
}
int main() {
  rlmesh::Session session(reinterpret_cast<RLMeshSession*>(&session_storage));
  auto action = rlmesh::Value::adopt(reinterpret_cast<RLMeshValue*>(&action_storage));
  rlmesh::ext::CancelToken token;
  for (int allocation : {0, 1}) {
    for (int operation : {0, 1, 2, 3}) {
      observation_frees = info_frees = 0;
      fail_after = allocation;
      bool threw = false;
      try {
        if (operation == 0)
          session.reset();
        else if (operation == 1)
          session.step(action);
        else if (operation == 2)
          rlmesh::ext::reset(session, token);
        else
          rlmesh::ext::step(session, token, action);
      } catch (const std::bad_alloc&) {
        threw = true;
      }
      fail_after = -1;
      CHECK(threw && observation_frees == 1 && info_frees == 1);
    }
  }
}
