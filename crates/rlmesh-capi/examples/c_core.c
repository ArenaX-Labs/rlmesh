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

int main(void) {
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
