/* The spaces C API, Python's rlmesh.spaces: build, seed, sample and check a
 * Dict of a Box and a Discrete, read typed bounds back, and hit the error
 * paths. Exits nonzero on the first failed check. */
#include <math.h>
#include <rlmesh.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      fprintf(stderr, "%s:%d: CHECK(%s) failed (last error: %s)\n",    \
              __FILE__, __LINE__, #cond, rlmesh_last_error_message()); \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

static int str_eq(RLMeshStr s, const char* expected) {
  return s.len == strlen(expected) && memcmp(s.data, expected, s.len) == 0;
}

int main(void) {
  /* The spec's Shape example. */
  RLMeshBox* pos = NULL;
  const int64_t shape[] = {3};
  CHECK(rlmesh_box_new(rlmesh_bound_float(-1.0), rlmesh_bound_float(1.0), shape, 1, NULL, &pos) ==
        RLMESH_OK);
  RLMeshDiscreteOptions dopt = RLMESH_DISCRETE_OPTIONS_INIT;
  dopt.start = 1;
  RLMeshDiscrete* id = NULL;
  CHECK(rlmesh_discrete_new(4, &dopt, &id) == RLMESH_OK);

  const RLMeshStr keys[] = {rlmesh_str("pos"), rlmesh_str("id")};
  const RLMeshSpaceSpec* specs[] = {rlmesh_space_spec(pos), rlmesh_space_spec(id)};
  RLMeshDict* obs = NULL;
  CHECK(rlmesh_dict_new(keys, specs, 2, &obs) == RLMESH_OK);
  CHECK(rlmesh_space_kind(obs) == RLMESH_SPACE_KIND_DICT);

  uint64_t seed = 7, used = 0;
  CHECK(rlmesh_space_seed(obs, &seed, &used) == RLMESH_OK && used == 7);
  RLMeshValue* x = NULL;
  CHECK(rlmesh_space_sample(obs, &x) == RLMESH_OK);
  CHECK(rlmesh_space_contains(obs, x));
  CHECK(!rlmesh_space_contains(pos, x));
  CHECK(!rlmesh_space_contains(NULL, x));
  rlmesh_value_free(x);

  /* Same seed, same sample, from a clone that shares the sampler. */
  RLMeshSpace* again = rlmesh_space_clone(obs);
  RLMeshValue* a = NULL;
  RLMeshValue* b = NULL;
  CHECK(rlmesh_space_seed(again, &seed, NULL) == RLMESH_OK);
  CHECK(rlmesh_space_sample(obs, &a) == RLMESH_OK);
  CHECK(rlmesh_space_seed(obs, &seed, NULL) == RLMESH_OK);
  CHECK(rlmesh_space_sample(again, &b) == RLMESH_OK);
  CHECK(rlmesh_value_kind(a) == rlmesh_value_kind(b));
  rlmesh_value_free(a);
  rlmesh_value_free(b);
  rlmesh_space_free(again);

  /* Keys sort at construction; children are borrowed. */
  size_t len = 0;
  CHECK(rlmesh_dict_len(obs, &len) == RLMESH_OK && len == 2);
  RLMeshStr key = {NULL, 0};
  CHECK(rlmesh_dict_key_at(obs, 0, &key) == RLMESH_OK && str_eq(key, "id"));
  const RLMeshSpace* child = NULL;
  CHECK(rlmesh_dict_value_at(obs, 1, &child) == RLMESH_OK);
  CHECK(rlmesh_space_kind(child) == RLMESH_SPACE_KIND_BOX);
  CHECK(rlmesh_dict_get(obs, rlmesh_str("missing"), &child) == RLMESH_OK && child == NULL);
  CHECK(rlmesh_dict_get(obs, rlmesh_str("pos"), &child) == RLMESH_OK && child != NULL);
  CHECK(rlmesh_space_equals(child, pos));
  bool present = false;
  CHECK(rlmesh_dict_contains_key(obs, rlmesh_str("id"), &present) == RLMESH_OK && present);
  CHECK(rlmesh_dict_contains_key(obs, rlmesh_str("nope"), &present) == RLMESH_OK && !present);
  CHECK(rlmesh_dict_contains_key(pos, rlmesh_str("id"), &present) == RLMESH_ERR_TYPE);
  /* Past the end reads as absent. */
  CHECK(rlmesh_dict_key_at(obs, 2, &key) == RLMESH_OK && key.data == NULL && key.len == 0);
  CHECK(rlmesh_dict_value_at(obs, 2, &child) == RLMESH_OK && child == NULL);

  /* Typed bounds read back in the dtype's domain, by the buffer rule. */
  RLMeshBox* u = NULL;
  RLMeshBoxOptions bopt = RLMESH_BOX_OPTIONS_INIT;
  bopt.dtype = RLMESH_DTYPE_UINT64;
  CHECK(rlmesh_box_new(rlmesh_bound_uint(0), rlmesh_bound_uint(UINT64_MAX), shape, 1, &bopt, &u) ==
        RLMESH_OK);
  bool has = false;
  RLMeshBoundsKind kind = RLMESH_BOUNDS_KIND_INVALID;
  CHECK(rlmesh_box_bounds_kind(u, &has, &kind) == RLMESH_OK && has &&
        kind == RLMESH_BOUNDS_KIND_TYPED_UNIFORM);
  CHECK(strcmp(rlmesh_bounds_kind_str(kind), "typed_uniform") == 0);
  size_t n = 0;
  CHECK(rlmesh_box_high(u, NULL, 0, &n) == RLMESH_OK && n == 1);
  RLMeshBound high;
  CHECK(rlmesh_box_high(u, &high, 1, &n) == RLMESH_OK);
  CHECK(high.type_ == RLMESH_BOUND_TYPE_UINT && high.value.u == UINT64_MAX);
  CHECK(rlmesh_space_spec_dtype(rlmesh_space_spec(u)) == RLMESH_DTYPE_UINT64);
  CHECK(strcmp(rlmesh_dtype_str(RLMESH_DTYPE_UINT64), "uint64") == 0);
  CHECK(rlmesh_dtype_str(RLMESH_DTYPE_OTHER) == NULL);
  CHECK(rlmesh_ext_dtype_size(RLMESH_DTYPE_FLOAT64) == 8);
  int64_t dims[2];
  CHECK(rlmesh_space_shape(u, dims, 0, &n) == RLMESH_ERR_INVALID_ARGUMENT && n == 1);
  CHECK(rlmesh_space_shape(u, dims, 2, &n) == RLMESH_OK && dims[0] == 3);
  rlmesh_space_free(u);

  /* Errors: fractional integer bounds, wrong-kind spec, typed accessor. */
  RLMeshBox* bad = NULL;
  bopt.dtype = RLMESH_DTYPE_INT32;
  CHECK(rlmesh_box_new(rlmesh_bound_float(0.5), rlmesh_bound_float(3.0), shape, 1, &bopt, &bad) ==
        RLMESH_ERR_INVALID_ARGUMENT);
  bopt.dtype = RLMESH_DTYPE_OTHER;
  CHECK(rlmesh_box_new(rlmesh_bound_float(0.0), rlmesh_bound_float(1.0), shape, 1, &bopt, &bad) ==
        RLMESH_ERR_INVALID_ARGUMENT);
  RLMeshDiscrete* wrong = NULL;
  CHECK(rlmesh_discrete_from_spec(rlmesh_space_spec(pos), &wrong) == RLMESH_ERR_INVALID_ARGUMENT);
  int64_t value = 0;
  CHECK(rlmesh_discrete_n(pos, &value) == RLMESH_ERR_TYPE);
  CHECK(rlmesh_discrete_start(id, &value) == RLMESH_OK && value == 1);

  /* Unbounded on any dtype; repr is Python's. */
  RLMeshBox* open = NULL;
  bopt.dtype = RLMESH_DTYPE_INT64;
  CHECK(rlmesh_box_new(rlmesh_bound_float(-INFINITY), rlmesh_bound_float(INFINITY), shape, 1,
                       &bopt, &open) == RLMESH_OK);
  CHECK(rlmesh_box_low(open, NULL, 0, &n) == RLMESH_OK && n == 0);
  RLMeshBytes text = {NULL, 0, 0};
  CHECK(rlmesh_space_repr(pos, &text) == RLMESH_OK);
  CHECK(text.len == strlen("Box(-1.0, 1.0, (3,), float32)") &&
        memcmp(text.data, "Box(-1.0, 1.0, (3,), float32)", text.len) == 0);
  rlmesh_bytes_free(text);
  rlmesh_space_free(open);

  /* Per-element bounds (C-only extra). */
  const RLMeshBound lows[] = {rlmesh_bound_int(0), rlmesh_bound_int(-5), rlmesh_bound_int(2)};
  const RLMeshBound highs[] = {rlmesh_bound_int(1), rlmesh_bound_int(5), rlmesh_bound_int(9)};
  RLMeshBox* per = NULL;
  CHECK(rlmesh_ext_box_elementwise_new(lows, highs, 3, shape, 1, &bopt, &per) == RLMESH_OK);
  RLMeshBound got[3];
  CHECK(rlmesh_box_low(per, got, 3, &n) == RLMESH_OK && n == 3 && got[1].value.i == -5);
  rlmesh_space_free(per);

  /* An ABI 4 binary frees a builder's spec with rlmesh_space_free. */
  RLMeshSpaceSpec* legacy = rlmesh_space_discrete(3, 0);
  RLMeshSpace* from_legacy = NULL;
  CHECK(rlmesh_space_new(legacy, &from_legacy) == RLMESH_OK);
  rlmesh_space_free((RLMeshSpace*)legacy);
  rlmesh_space_free(from_legacy);

  rlmesh_space_free(obs);
  rlmesh_space_free(id);
  rlmesh_space_free(pos);
  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("c_spaces OK\n");
  return 0;
}
