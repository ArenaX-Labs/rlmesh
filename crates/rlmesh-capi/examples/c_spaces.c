/* The spaces C API, Python's rlmesh.spaces: build, seed, sample and check a
 * Dict of a Box and a Discrete, read typed bounds back, and hit the error
 * paths. Exits nonzero on the first failed check. */
#include <math.h>
#include <rlmesh.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                                            \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      fprintf(stderr, "%s:%d: CHECK(%s) failed (last error: %s)\n", __FILE__, __LINE__, #cond, \
              rlmesh_last_error_message());                                                    \
      ++failures;                                                                              \
    }                                                                                          \
  } while (0)

static int str_eq(RLMeshStr s, const char* expected) {
  return s.len == strlen(expected) && memcmp(s.data, expected, s.len) == 0;
}

/* SPC-5 accepts facade Values that the strict wire validator rejects. */
static void check_contains_acceptance(void) {
  RLMeshDiscreteOptions options = RLMESH_DISCRETE_OPTIONS_INIT;
  options.start = 2;
  RLMeshDiscrete* discrete = NULL;
  CHECK(rlmesh_discrete_new(4, &options, &discrete) == RLMESH_OK);
  int64_t ints[] = {3, 4};
  const size_t one[] = {1}, two[] = {2};
  RLMeshTensor* tensor = NULL;
  RLMeshValue* value = NULL;
  CHECK(rlmesh_tensor_new(ints, sizeof(ints[0]), one, 1, RLMESH_DTYPE_INT64, &tensor) == RLMESH_OK);
  CHECK(rlmesh_value_tensor(tensor, &value) == RLMESH_OK);
  CHECK(rlmesh_space_contains(discrete, value));
  rlmesh_value_free(value);
  rlmesh_tensor_free(tensor);
  ints[0] = 9;
  CHECK(rlmesh_tensor_new(ints, sizeof(ints[0]), one, 1, RLMESH_DTYPE_INT64, &tensor) == RLMESH_OK);
  CHECK(rlmesh_value_tensor(tensor, &value) == RLMESH_OK);
  CHECK(!rlmesh_space_contains(discrete, value));
  rlmesh_value_free(value);
  rlmesh_tensor_free(tensor);
  ints[0] = 3;
  CHECK(rlmesh_tensor_new(ints, sizeof(ints), two, 1, RLMESH_DTYPE_INT64, &tensor) == RLMESH_OK);
  CHECK(rlmesh_value_tensor(tensor, &value) == RLMESH_OK);
  CHECK(!rlmesh_space_contains(discrete, value));
  rlmesh_value_free(value);
  rlmesh_tensor_free(tensor);

  RLMeshStr keys[] = {rlmesh_str("required"), rlmesh_str("extra")};
  const RLMeshSpaceSpec* specs[] = {rlmesh_space_spec(discrete)};
  RLMeshDict* dict = NULL;
  CHECK(rlmesh_dict_new(keys, specs, 1, &dict) == RLMESH_OK);
  RLMeshValue *member = NULL, *extra = NULL;
  CHECK(rlmesh_value_int(3, &member) == RLMESH_OK);
  CHECK(rlmesh_value_text(rlmesh_str("ignored extra value"), &extra) == RLMESH_OK);
  const RLMeshValue* values[] = {member, extra};
  CHECK(rlmesh_value_dict(keys, values, 2, &value) == RLMESH_OK);
  CHECK(rlmesh_space_contains(dict, value));
  rlmesh_value_free(value);
  CHECK(rlmesh_value_dict(keys + 1, values + 1, 1, &value) == RLMESH_OK);
  CHECK(!rlmesh_space_contains(dict, value));
  rlmesh_value_free(value);
  rlmesh_value_free(member);
  rlmesh_value_free(extra);
  rlmesh_space_free(dict);
  rlmesh_space_free(discrete);
}

/* SPC-10/11/13: sampling follows the spec and keys carry their byte lengths. */
static void check_dict_key_bytes(void) {
  const char embedded[] = {'k', '\0', 'v'};
  const RLMeshStr keys[] = {{embedded, sizeof(embedded)}, {"k", 1}};
  RLMeshDiscrete* leaf = NULL;
  CHECK(rlmesh_discrete_new(8, NULL, &leaf) == RLMESH_OK);
  const RLMeshSpaceSpec* specs[] = {rlmesh_space_spec(leaf), rlmesh_space_spec(leaf)};
  RLMeshDict* dict = NULL;
  CHECK(rlmesh_dict_new(keys, specs, 2, &dict) == RLMESH_OK);
  const RLMeshSpace* child = NULL;
  CHECK(rlmesh_dict_get(dict, keys[0], &child) == RLMESH_OK && child != NULL);
  CHECK(rlmesh_dict_get(dict, keys[1], &child) == RLMESH_OK && child != NULL);
  /* An explicit length distinguishes longer keys sharing the same prefix. */
  const char missing_bytes[] = "k\0not-the-same";
  const RLMeshStr missing = {missing_bytes, sizeof(missing_bytes) - 1};
  CHECK(rlmesh_dict_get(dict, missing, &child) == RLMESH_OK && child == NULL);
  RLMeshValue* sampled = NULL;
  CHECK(rlmesh_space_sample(dict, &sampled) == RLMESH_OK);
  CHECK(rlmesh_space_contains(dict, sampled));
  for (size_t i = 0; i < 2; ++i) {
    RLMeshStr declared = {NULL, 0}, actual = {NULL, 0};
    CHECK(rlmesh_dict_key_at(dict, i, &declared) == RLMESH_OK);
    CHECK(rlmesh_value_dict_key_at(sampled, i, &actual) == RLMESH_OK);
    CHECK(actual.len == declared.len && memcmp(actual.data, declared.data, actual.len) == 0);
  }
  const RLMeshValue* borrowed = NULL;
  CHECK(rlmesh_value_dict_get(sampled, keys[0], &borrowed) == RLMESH_OK && borrowed != NULL);
  CHECK(rlmesh_value_kind(borrowed) == RLMESH_VALUE_INT);
  rlmesh_value_free(sampled);

  RLMeshValue *first = NULL, *second = NULL, *value = NULL;
  CHECK(rlmesh_value_int(1, &first) == RLMESH_OK);
  CHECK(rlmesh_value_int(2, &second) == RLMESH_OK);
  const RLMeshValue* values[] = {first, second};
  CHECK(rlmesh_value_dict(keys, values, 2, &value) == RLMESH_OK);
  rlmesh_value_free(first);
  rlmesh_value_free(second);
  RLMeshStr key = {NULL, 0};
  CHECK(rlmesh_value_dict_key_at(value, 0, &key) == RLMESH_OK && key.len == sizeof(embedded) &&
        memcmp(key.data, embedded, sizeof(embedded)) == 0);
  int64_t number = 0;
  CHECK(rlmesh_value_dict_get(value, keys[0], &borrowed) == RLMESH_OK && borrowed != NULL);
  CHECK(rlmesh_value_as_int(borrowed, &number) == RLMESH_OK && number == 1);
  CHECK(rlmesh_value_dict_get(value, keys[1], &borrowed) == RLMESH_OK && borrowed != NULL);
  CHECK(rlmesh_value_as_int(borrowed, &number) == RLMESH_OK && number == 2);
  CHECK(rlmesh_value_dict_get(value, missing, &borrowed) == RLMESH_OK && borrowed == NULL);
  bool present = false;
  CHECK(rlmesh_value_dict_contains_key(value, keys[0], &present) == RLMESH_OK && present);
  rlmesh_value_free(value);
  rlmesh_space_free(dict);
  rlmesh_space_free(leaf);
}

/* An extracted/cloned tensor owns storage independently from its source Value. */
static void check_tensor_reference_lifetime(void) {
  const int64_t elements[] = {7, 11};
  const size_t shape[] = {2};
  RLMeshTensor *original = NULL, *extracted = NULL, *copy = NULL;
  RLMeshValue* value = NULL;
  CHECK(rlmesh_tensor_new(elements, sizeof(elements), shape, 1, RLMESH_DTYPE_INT64, &original) ==
        RLMESH_OK);
  CHECK(rlmesh_value_tensor(original, &value) == RLMESH_OK);
  CHECK(rlmesh_value_as_tensor(value, &extracted) == RLMESH_OK);
  copy = rlmesh_tensor_clone(extracted);
  CHECK(copy != NULL);
  rlmesh_tensor_free(original);
  rlmesh_value_free(value);
  rlmesh_tensor_free(extracted);
  const void* buffer = NULL;
  size_t nbytes = 0;
  CHECK(rlmesh_tensor_buffer(copy, &buffer, &nbytes) == RLMESH_OK && nbytes == sizeof(elements));
  CHECK(memcmp(buffer, elements, sizeof(elements)) == 0);
  rlmesh_tensor_free(copy);
}

int main(void) {
  check_contains_acceptance();
  check_dict_key_bytes();
  check_tensor_reference_lifetime();
  /* The spec's Shape example. */
  RLMeshBox* pos = NULL;
  const int64_t shape[] = {3};
  CHECK(rlmesh_box_new(rlmesh_bound_float(-1.0), rlmesh_bound_float(1.0), shape, 1, NULL, &pos) ==
        RLMESH_OK);
  CHECK(rlmesh_space_dtype(pos) == RLMESH_DTYPE_FLOAT32);
  CHECK(rlmesh_space_dtype(NULL) == RLMESH_DTYPE_INVALID);
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
  CHECK(rlmesh_box_new(rlmesh_bound_float(-INFINITY), rlmesh_bound_float(INFINITY), shape, 1, &bopt,
                       &open) == RLMESH_OK);
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

  /* Space and SpaceSpec have distinct ABI-5 ownership/free functions. */
  RLMeshSpaceSpec* cloned = rlmesh_space_spec_clone(rlmesh_space_spec(id));
  RLMeshSpace* from_cloned = NULL;
  CHECK(rlmesh_space_new(cloned, &from_cloned) == RLMESH_OK);
  rlmesh_space_spec_free(cloned);
  rlmesh_space_free(from_cloned);

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
