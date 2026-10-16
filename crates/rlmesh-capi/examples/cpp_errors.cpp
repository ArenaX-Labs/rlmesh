// Runtime check that the C++ exception classes copy the Python package's
// inheritance and that a capi failure throws the class it reports. Exits
// nonzero (with the failed check) on any mismatch.
#include <cstdio>
#include <exception>
#include <rlmesh.hpp>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "cpp_errors: FAILED: %s\n", what);
    ++failures;
  }
}

// Which handlers catch the exception `error` throws.
struct Caught {
  bool rlmesh_exception = false;
  bool environment = false;
  bool runtime_error = false;
  bool os_error = false;
  bool value_error = false;
  bool std_exception = false;
  bool cancelled = false;
  std::string what;
};

Caught catch_all(const rlmesh::Error& error) {
  Caught caught;
  try {
    rlmesh::throw_error(error);
  } catch (const rlmesh::EnvironmentException& e) {
    caught.environment = caught.rlmesh_exception = caught.runtime_error = true;
    caught.std_exception = true;
    caught.what = e.what();
  } catch (const rlmesh::RLMeshException& e) {
    caught.rlmesh_exception = caught.runtime_error = caught.std_exception = true;
    caught.what = e.what();
  } catch (const rlmesh::RuntimeError& e) {
    caught.runtime_error = caught.std_exception = true;
    caught.what = e.what();
  } catch (const rlmesh::OSError& e) {
    caught.os_error = caught.std_exception = true;
    caught.what = e.what();
  } catch (const rlmesh::ValueError& e) {
    caught.value_error = caught.std_exception = true;
    caught.what = e.what();
  } catch (const std::exception& e) {
    caught.std_exception = true;
    caught.what = e.what();
  } catch (const rlmesh::ext::CancelledError& e) {
    caught.cancelled = true;
    caught.what = e.what();
  }
  return caught;
}

}  // namespace

int main() {
  // A wrong-kind read through the C ABI is TypeError: not a ValueError, not an
  // RLMeshException.
  RLMeshValue* text = rlmesh_value_text("x", 1);
  int64_t out = 0;
  RLMeshStatus status = rlmesh_value_as_discrete(text, &out);
  check(status == RLMESH_ERR_TYPE, "wrong-kind read returns RLMESH_ERR_TYPE");
  rlmesh::Error type_error = rlmesh::Error::from_last(status);
  check(type_error.error_class() == RLMESH_ERROR_CLASS_TYPE_ERROR, "class is TYPE_ERROR");
  check(std::string(rlmesh_error_class_name(type_error.error_class())) == "TypeError",
        "class name is TypeError");
  try {
    rlmesh::throw_error(type_error);
  } catch (const rlmesh::TypeError& e) {
    check(std::string(e.what()) == type_error.message(), "what() is the last error message");
  } catch (...) {
    check(false, "a wrong-kind read throws rlmesh::TypeError");
  }
  rlmesh_value_free(text);

  // The Environment family: a recoverable error is caught by every ancestor.
  Caught recoverable = catch_all(rlmesh::Error(RLMESH_ERR_ENVIRONMENT, "[BUSY] later", true));
  check(
      recoverable.environment && recoverable.rlmesh_exception && recoverable.runtime_error,
      "RecoverableEnvironmentException is an EnvironmentException, RLMeshException, RuntimeError");
  check(recoverable.what == "[BUSY] later", "what() is the message");
  try {
    rlmesh::throw_error(rlmesh::Error(RLMESH_ERR_ENVIRONMENT, "x", true));
  } catch (const rlmesh::RecoverableEnvironmentException& e) {
    check(e.is_recoverable(), "recoverable flag survives");
  } catch (...) {
    check(false, "a recoverable env error throws RecoverableEnvironmentException");
  }

  // ModelException carries its authoring flag and code INTERNAL.
  rlmesh::ModelException authored("declined", true);
  check(authored.is_recoverable(), "ModelException(msg, true) is recoverable");
  check(authored.code() == rlmesh::ModelErrorCode::Internal, "authored code is INTERNAL");
  check(rlmesh::to_string(authored.code()) == "INTERNAL", "to_string(ModelErrorCode)");
  check(rlmesh::to_string(rlmesh::ErrorCode::InvalidAction) == "INVALID_ACTION",
        "to_string(ErrorCode)");
  check(!rlmesh::ModelException("x").is_recoverable(), "ModelException defaults to permanent");

  // The builtins stay outside RLMeshException, as in Python.
  Caught misuse = catch_all(rlmesh::Error(RLMESH_ERR_RUNTIME, "closed"));
  check(misuse.runtime_error && !misuse.rlmesh_exception, "RuntimeError is not an RLMeshException");
  Caught connection = catch_all(rlmesh::Error(RLMESH_ERR_CONNECTION, "reset"));
  check(connection.os_error, "ConnectionError is an OSError");
  Caught timeout = catch_all(rlmesh::Error(RLMESH_ERR_TIMEOUT, "operation timed out after 5s"));
  check(timeout.os_error, "TimeoutError is an OSError");
  Caught value = catch_all(rlmesh::Error(RLMESH_ERR_INVALID_ARGUMENT, "bad seed"));
  check(value.value_error && !value.rlmesh_exception, "ValueError is outside RLMeshException");
  Caught protocol = catch_all(rlmesh::Error(RLMESH_ERR_PROTOCOL, "bad frame"));
  check(protocol.rlmesh_exception && !protocol.environment,
        "ProtocolException is an RLMeshException");
  Caught internal = catch_all(rlmesh::Error(RLMESH_ERR_PANIC, "boom"));
  check(internal.rlmesh_exception, "a panic throws RLMeshException");

  // A cancel is not a std::exception, as KeyboardInterrupt is not an Exception.
  Caught cancel = catch_all(rlmesh::Error(RLMESH_ERR_CANCELLED, "cancelled"));
  check(cancel.cancelled && !cancel.std_exception, "CancelledError has no std:: base");

  // Result::value_or_throw.
  rlmesh::Status failed = rlmesh::Error(RLMESH_ERR_TYPE, "kind");
  try {
    failed.value_or_throw();
    check(false, "value_or_throw throws on an error");
  } catch (const rlmesh::TypeError&) {
  }

  // The C class functions follow Python's inheritance.
  check(rlmesh_error_class_is_subclass(RLMESH_ERROR_CLASS_CONNECTION_ERROR,
                                       RLMESH_ERROR_CLASS_OS_ERROR),
        "C: ConnectionError is an OSError");
  check(!rlmesh_error_class_is_subclass(RLMESH_ERROR_CLASS_VALUE_ERROR,
                                        RLMESH_ERROR_CLASS_RLMESH_EXCEPTION),
        "C: ValueError is not an RLMeshException");

  if (failures != 0) return 1;
  std::printf("cpp_errors: OK\n");
  return 0;
}
