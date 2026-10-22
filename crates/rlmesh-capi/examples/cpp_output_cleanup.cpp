// Fail shared_ptr allocation after C returned two owned Value outputs.
#define rlmesh_binding_reset test_binding_reset
#define rlmesh_binding_step test_binding_step
#define rlmesh_binding_free test_binding_free
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
unsigned char binding_storage, observation_storage, info_storage, action_storage;
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
extern "C" void test_binding_free(RLMeshBinding*) {}
extern "C" RLMeshStatus test_binding_reset(RLMeshBinding*, const int64_t*, const RLMeshValue*,
                                           RLMeshValue** o, RLMeshValue** i) {
  *o = observation();
  *i = info();
  return RLMESH_OK;
}
extern "C" RLMeshStatus test_binding_step(RLMeshBinding*, const RLMeshValue*, RLMeshValue** o,
                                          double* reward, bool* term, bool* trunc,
                                          RLMeshValue** i) {
  *reward = 0;
  *term = false;
  *trunc = false;
  *o = observation();
  *i = info();
  return RLMESH_OK;
}
int main() {
  rlmesh::Binding binding(reinterpret_cast<RLMeshBinding*>(&binding_storage));
  auto action = rlmesh::Value::adopt(reinterpret_cast<RLMeshValue*>(&action_storage));
  for (int allocation : {0, 1}) {
    for (int operation : {0, 1, 2}) {
      observation_frees = info_frees = 0;
      fail_after = allocation;
      bool threw = false;
      try {
        if (operation == 0)
          binding.reset();
        else if (operation == 1)
          binding.step(action);
        else
          binding.step();
      } catch (const std::bad_alloc&) {
        threw = true;
      }
      fail_after = -1;
      CHECK(threw && observation_frees == 1 && info_frees == 1);
    }
  }
}
