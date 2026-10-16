// Core C++ wrapper: build info, the current workflow edition, the describe and
// reset constants, and the warning handler. Exits nonzero on the first failed
// check.
#include <cstdio>
#include <iostream>
#include <rlmesh.hpp>
#include <sstream>

namespace {

int failures = 0;

void check(bool ok, const char* what, int line) {
  if (!ok) {
    std::fprintf(stderr, "cpp_core.cpp:%d: %s failed\n", line, what);
    ++failures;
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

}  // namespace

int main() {
  // The Shape example.
  std::cout << rlmesh::build_info().version << " " << rlmesh::current_workflow_edition() << "\n";

  rlmesh::BuildInfo info = rlmesh::build_info();
  CHECK(info.version == rlmesh::VERSION);
  CHECK(info.protocol_generation == "rlmesh-wire-v1");
  CHECK(info.workflow_edition_base == rlmesh::current_workflow_edition());
  CHECK(!info.supported_workflow_editions.empty());
  CHECK(info.supported_workflow_editions.front() == info.workflow_edition);
  CHECK(!rlmesh::to_string(info.build_source).empty());
  CHECK(info.git.has_value() == (info.build_cohort.substr(0, 4) == "dev."));
  std::ostringstream repr;
  repr << info;
  CHECK(repr.str().rfind("BuildInfo(version=\"", 0) == 0);
  CHECK(repr.str() == rlmesh::to_string(info));

  CHECK(rlmesh::DESCRIBE_METADATA_KEY == "rlmesh.describe.v1");
  CHECK(rlmesh::DESCRIBE_SCHEMA_VERSION == 1);
  CHECK(rlmesh::ENV_RESET_OPTIONS_KEY == "rlmesh.env.v1.reset_options");

  using rlmesh::WarningCategory;
  CHECK(rlmesh::name(WarningCategory::WorkflowEditionWarning) == "WorkflowEditionWarning");
  CHECK(rlmesh::is_subclass_of(WarningCategory::WorkflowEditionWarning,
                               WarningCategory::UserWarning));
  CHECK(!rlmesh::is_subclass_of(WarningCategory::UserWarning, WarningCategory::RuntimeWarning));
  rlmesh::Warning warning{WarningCategory::RuntimeWarning, "short chunk"};
  std::ostringstream text;
  text << warning;
  CHECK(text.str() == "RuntimeWarning: short chunk");
  rlmesh::set_warning_handler([](const rlmesh::Warning&) {});
  rlmesh::set_warning_handler(nullptr);

  if (failures != 0) {
    std::fprintf(stderr, "cpp_core: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("cpp_core OK\n");
  return 0;
}
