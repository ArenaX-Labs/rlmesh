/* Core C API: build info, the current workflow edition, the describe and reset
 * constants, and the warning handler. Exits nonzero on the first failed
 * check. */
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

static int freed = 0;

static RLMeshStatus ignore_warning(void* user_data, RLMeshWarningCategory category,
                                   const char* message) {
  (void)user_data;
  (void)category;
  (void)message;
  return RLMESH_OK;
}

static void count_free(void* user_data) {
  (void)user_data;
  ++freed;
}

static void check_values(void) {
  RLMeshValue* one = NULL;
  RLMeshValue* flag = NULL;
  RLMeshValue* dict = NULL;
  CHECK(rlmesh_value_int(1, &one) == RLMESH_OK);
  CHECK(rlmesh_value_bool(true, &flag) == RLMESH_OK);
  int64_t number = 42;
  CHECK(rlmesh_value_as_int(flag, &number) == RLMESH_ERR_TYPE && number == 42);
  const RLMeshValue* values[] = {one, flag, flag};
  RLMeshStr keys[] = {rlmesh_str("z"), rlmesh_str(""), rlmesh_str("z")};
  CHECK(rlmesh_value_dict(keys, values, 3, &dict) == RLMESH_OK);
  rlmesh_value_free(one);
  rlmesh_value_free(flag);
  RLMeshValue* retained = rlmesh_value_clone(dict);
  rlmesh_value_free(dict);
  size_t n = 0;
  CHECK(rlmesh_value_len(retained, &n) == RLMESH_OK && n == 2);
  RLMeshStr key = {NULL, 0};
  CHECK(rlmesh_value_dict_key_at(retained, 0, &key) == RLMESH_OK && key.len == 1 &&
        key.data[0] == 'z');
  CHECK(rlmesh_value_dict_key_at(retained, 1, &key) == RLMESH_OK && key.len == 0 &&
        key.data != NULL);
  const RLMeshValue* child = NULL;
  CHECK(rlmesh_value_dict_get(retained, rlmesh_str("z"), &child) == RLMESH_OK);
  CHECK(rlmesh_value_kind(child) == RLMESH_VALUE_BOOL);
  CHECK(rlmesh_value_dict_get(retained, rlmesh_str("absent"), &child) == RLMESH_OK &&
        child == NULL);
  rlmesh_value_free(retained);

  RLMeshValue* ordinal = NULL;
  RLMeshValue* options = NULL;
  CHECK(rlmesh_value_int(-3, &ordinal) == RLMESH_OK);
  RLMeshStr trial_key = rlmesh_str("trial_index");
  const RLMeshValue* trial_values[] = {ordinal};
  CHECK(rlmesh_value_dict(&trial_key, trial_values, 1, &options) == RLMESH_OK);
  rlmesh_value_free(ordinal);
  bool has_trial = false;
  int64_t trial = 0;
  CHECK(rlmesh_trial_index(options, NULL, &has_trial, &trial) == RLMESH_OK && has_trial &&
        trial == -3);
  const size_t lane = 1;
  CHECK(rlmesh_trial_index(options, &lane, &has_trial, &trial) == RLMESH_ERR_INVALID_ARGUMENT);
  rlmesh_value_free(options);

  const float data[] = {1, 2, 3, 4};
  const size_t shape[] = {2, 2};
  RLMeshTensor* tensor = NULL;
  CHECK(rlmesh_tensor_new(data, sizeof(data), shape, 2, RLMESH_DTYPE_FLOAT32, &tensor) ==
        RLMESH_OK);
  size_t strides[2] = {99, 99};
  CHECK(rlmesh_tensor_strides(tensor, NULL, 0, &n) == RLMESH_OK && n == 2);
  CHECK(rlmesh_tensor_strides(tensor, strides, 1, &n) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(strides[0] == 99 && strides[1] == 99 && n == 2);
  CHECK(rlmesh_tensor_strides(tensor, strides, 2, &n) == RLMESH_OK && strides[0] == 8 &&
        strides[1] == 4);
  RLMeshValue* value = NULL;
  CHECK(rlmesh_value_tensor(tensor, &value) == RLMESH_OK);
  rlmesh_tensor_free(tensor);
  CHECK(rlmesh_value_as_tensor(value, &tensor) == RLMESH_OK);
  rlmesh_value_free(value);
  const void* buffer = NULL;
  CHECK(rlmesh_tensor_buffer(tensor, &buffer, &n) == RLMESH_OK && n == sizeof(data));
  CHECK(memcmp(buffer, data, sizeof(data)) == 0);
  rlmesh_tensor_free(tensor);
}

int main(void) {
  CHECK(rlmesh_abi_version() == 5 && rlmesh_abi_check());
  check_values();
  RLMeshBuildInfo bi = RLMESH_BUILD_INFO_INIT;
  CHECK(rlmesh_build_info(&bi) == RLMESH_OK);
  CHECK(strcmp(bi.version, RLMESH_VERSION) == 0);
  CHECK(strcmp(bi.protocol_generation, "rlmesh-wire-v1") == 0);
  CHECK(strcmp(bi.workflow_edition_base, rlmesh_current_workflow_edition()) == 0);
  CHECK(strncmp(bi.workflow_edition, bi.workflow_edition_base, strlen(bi.workflow_edition_base)) ==
        0);
  CHECK(bi.supported_workflow_editions_len >= 1);
  CHECK(strcmp(bi.supported_workflow_editions[0], bi.workflow_edition) == 0);
  CHECK(rlmesh_build_source_str(bi.build_source) != NULL);
  CHECK(rlmesh_build_source_str(RLMESH_BUILD_SOURCE_INVALID) == NULL);
  CHECK((bi.git != NULL) == (strncmp(bi.build_cohort, "dev.", 4) == 0));
  printf("%s %s\n", bi.version, rlmesh_current_workflow_edition());

  RLMeshBytes repr = {NULL, 0, 0};
  CHECK(rlmesh_build_info_repr(&repr) == RLMESH_OK);
  CHECK(repr.len > 10 && memcmp(repr.data, "BuildInfo(", 10) == 0);
  rlmesh_bytes_free(repr);

  /* R-19: a struct_size below the minimum is refused, a larger one with a
   * nonzero tail too. */
  RLMeshBuildInfo small = RLMESH_BUILD_INFO_INIT;
  small.struct_size = RLMESH_BUILD_INFO_MIN_SIZE - 1;
  CHECK(rlmesh_build_info(&small) == RLMESH_ERR_INVALID_ARGUMENT);
  struct {
    RLMeshBuildInfo info;
    uint64_t tail;
  } large = {RLMESH_BUILD_INFO_INIT, 1};
  large.info.struct_size = sizeof(large);
  CHECK(rlmesh_build_info(&large.info) == RLMESH_ERR_TYPE);
  large.tail = 0;
  CHECK(rlmesh_build_info(&large.info) == RLMESH_OK);

  CHECK(strcmp(RLMESH_DESCRIBE_METADATA_KEY, "rlmesh.describe.v1") == 0);
  CHECK(RLMESH_DESCRIBE_SCHEMA_VERSION == 1);
  CHECK(strcmp(RLMESH_ENV_RESET_OPTIONS_KEY, "rlmesh.env.v1.reset_options") == 0);

  CHECK(strcmp(rlmesh_warning_category_name(RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING),
               "WorkflowEditionWarning") == 0);
  CHECK(rlmesh_warning_category_is_subclass(RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING,
                                            RLMESH_WARNING_CATEGORY_USER_WARNING));
  CHECK(!rlmesh_warning_category_is_subclass(RLMESH_WARNING_CATEGORY_RUNTIME_WARNING,
                                             RLMESH_WARNING_CATEGORY_USER_WARNING));
  CHECK(rlmesh_set_warning_handler(ignore_warning, NULL, count_free) == RLMESH_OK);
  CHECK(rlmesh_set_warning_handler(NULL, NULL, count_free) == RLMESH_ERR_INVALID_ARGUMENT);
  CHECK(freed == 0);
  CHECK(rlmesh_set_warning_handler(NULL, NULL, NULL) == RLMESH_OK);
  CHECK(freed == 1);

  if (failures != 0) {
    fprintf(stderr, "c_core: %d check(s) failed\n", failures);
    return 1;
  }
  printf("c_core OK\n");
  return 0;
}
