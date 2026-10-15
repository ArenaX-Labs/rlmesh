# Core: values, errors, editions, describe and build info

Part of the [language parity spec](../language-parity.md). Status: Draft —
reviewed round 4. [Rules](rules.md) are authoritative; paths use its
abbreviations (`py/`, `pyrs/`, `rs/`, `capi/`, `h`, `hpp`).

Core owns `Value`, `Metadata`, the error detail fields, workflow editions,
describe, build info and `ENV_RESET_OPTIONS_KEY` (R-27). Error classes and
statuses are R-21, error text R-22, warnings R-23.

## Shape

```python
try:
    obs, reward, terminated, truncated, info = env.step(action)
except rlmesh.RecoverableEnvironmentException as e:
    log(e.code)  # "BUSY"
print(rlmesh.build_info().version, rlmesh.current_workflow_edition())
envelope = rlmesh.describe_json(MyModel, generated_at="2026-10-15T00:00:00Z")
```

```rust
match env.step(action) {
    Ok((obs, reward, terminated, truncated, info)) => { /* … */ }
    Err(Error::Environment(e)) if e.is_recoverable => log(e.code.as_str()),
    Err(e) => return Err(e),
}
println!("{} {}", rlmesh::build_info().version, rlmesh::current_workflow_edition());
let envelope = rlmesh::describe_json(
    Describable::model::<MyPolicy>(),
    DescribeJsonOptions { generated_at: Some("2026-10-15T00:00:00Z".into()), ..Default::default() },
)?;
```

```cpp
try {
  auto [obs, reward, terminated, truncated, info] = env.step(action);
} catch (const rlmesh::RecoverableEnvironmentException& e) {
  log(rlmesh::to_string(e.code()));
}
std::cout << rlmesh::build_info().version << rlmesh::current_workflow_edition();
std::string envelope = rlmesh::describe_json<MyPolicy>({.generated_at = "2026-10-15T00:00:00Z"});
```

```c
RLMeshStatus st = /* any call, e.g. a RemoteEnv step */;
if (st != RLMESH_OK &&
    rlmesh_last_error_class() == RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION)
  log(rlmesh_last_error_code_str()); /* "BUSY" */

RLMeshBuildInfo bi = RLMESH_BUILD_INFO_INIT;
rlmesh_build_info(&bi);
printf("%s %s\n", bi.version, rlmesh_current_workflow_edition());

RLMeshDescribeJsonOptions o = RLMESH_DESCRIBE_JSON_OPTIONS_INIT;
o.generated_at = "2026-10-15T00:00:00Z";
RLMeshBytes envelope;
rlmesh_model_vtable_describe_json(&vt, &o, &envelope); /* vt: MyPolicy's RLMeshModelVtable */
```

## Parity table

### Values and metadata

| Python                                                                 | Rust                                                                                                                                                                                                   | C                                                                                                                                                                                                                                                                         | C++                                                                                                        |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `rlmesh.types.Value`                                                   | `rlmesh::Value` (also `rlmesh::types::Value`): `#[non_exhaustive] enum Value { None, Bool(bool), Int(i64), Float(f64), Text(String), Bytes(Vec<u8>), Tensor(Tensor), List(Vec<Value>), Tuple(Vec<Value>), Dict(Metadata) }`; `Clone`, `PartialEq` (COR-24), `Debug`; constructors and accessors in [Value API](#value-api) | `RLMeshValue*` (immutable, R-11); `RLMeshValueKind rlmesh_value_kind(const RLMeshValue*)`: `RLMESH_VALUE_INVALID = 0` (also a NULL handle), `NONE`, `BOOL`, `INT`, `FLOAT`, `TEXT`, `BYTES`, `TENSOR`, `LIST`, `TUPLE`, `DICT` (1-10); `RLMeshValue* rlmesh_value_clone(const RLMeshValue*)` (new shared reference, as `rlmesh_tensor_clone`), `bool rlmesh_value_equals(const RLMeshValue*, const RLMeshValue*)` (COR-24), `void rlmesh_value_free(RLMeshValue*)`; [Value API](#value-api) | `class rlmesh::Value` (copy shares, R-11); `ValueKind kind() const`; `operator==` (COR-24); [Value API](#value-api) |
| `rlmesh.types.PrimitiveValue`                                          | the `None` … `Bytes` arms of `Value`                                                                                                                                                                   | the `NONE` … `BYTES` kinds                                                                                                                                                                                                                                                | the same `Value` kinds                                                                                     |
| `rlmesh.types.Metadata`, `InfoDict` | `pub struct Metadata`, an insertion-ordered `String` → `Value` map (COR-2): `new()`, `get(&self, key: &str) -> Option<&Value>`, `get_mut`, `contains_key(&self, key: &str) -> bool`, `insert(&mut self, key: impl Into<String>, value: impl Into<Value>) -> Option<Value>`, `remove(&mut self, key: &str) -> Option<Value>`, `iter()`, `keys()`, `values()`, `len()`, `is_empty()`, `FromIterator<(K, V)>`, `IntoIterator`; `#[derive(Clone, Debug, Default)]` plus a hand-written order-insensitive `PartialEq`; on the wire, COR-26; `pub type InfoDict = Metadata`; `pub type Kwargs = Metadata` | `RLMeshValue*` of kind `DICT`, immutable: built with `rlmesh_value_dict`, read with `rlmesh_value_len`, `rlmesh_value_dict_key_at`, `rlmesh_value_dict_value_at` (insertion order), `rlmesh_value_dict_get`, `rlmesh_value_dict_contains_key` ([Value API](#value-api)) | `class Metadata`, an insertion-ordered map: constructor from `std::initializer_list<std::pair<std::string, Value>>`, `find`, `at` (throws `std::out_of_range`), `contains_key`, `insert_or_assign`, `erase`, `size`, `empty`, `begin`/`end` in insertion order; order-insensitive `operator==`; `using InfoDict = Metadata`; `using Kwargs = Metadata` |
| `key in d`, `d[k] = v`, `del d[k]`, `dict(pairs)` | `contains_key` (named as Spaces' `Dict::contains_key`); `insert` on an existing key replaces the value in place and keeps its position; `remove` shifts, keeping the order of the remaining keys (never a swap-remove); `FromIterator` with a repeated key keeps the first position and the last value | `rlmesh_value_dict_contains_key`; no mutation (build a new DICT); `rlmesh_value_dict` with a repeated key as `dict(pairs)` (COR-21) | `contains_key`; `insert_or_assign` keeps an existing key's position; `erase` keeps the order of the rest; the initializer-list constructor as `dict(pairs)` |
| a wrong-kind read (Python `TypeError`)                                 | `Error::Type`                                                                                                                                                                                          | `RLMESH_ERR_TYPE`                                                                                                                                                                                                                                                         | `rlmesh::TypeError`                                                                                        |

### Value API

Python builds and reads values with its own objects; the native forms below
are the constructors and checked reads of each kind. A read of another kind
fails with the `TypeError` class (`Error::Type`, `RLMESH_ERR_TYPE`,
`rlmesh::TypeError`); reads are strict, so `as_float` refuses an `Int` and
`as_int` refuses a `Bool` (COR-1). C constructors return `RLMeshStatus` and
write a new owned value to `out`; composite constructors take new shared
references to their children, so the caller still frees its own. C reads
borrow from the value and stay valid while it lives. An index past the end
reads as absent, as SPC-27 does for a missing key: Rust `None`, C
`RLMESH_OK` with `*out = NULL` for a handle output and `*out = {NULL, 0}`
for an `RLMeshStr` output (`rlmesh_value_dict_key_at`), C++
`std::out_of_range`. A present `RLMeshStr` output always has non-NULL
`data`, the empty key and empty text included, so `data == NULL` alone means
absent.

| Python kind | Rust | C | C++ |
| ----------- | ---- | - | --- |
| `None` | `Value::None`; `fn is_none(&self) -> bool` | `rlmesh_value_none(RLMeshValue** out)` | `Value()`, `static Value none()`; `Value(std::nullptr_t) = delete`; `bool is_none() const` |
| `bool` | `Value::Bool(b)`, `From<bool>`; `fn as_bool(&self) -> Result<bool>` | `rlmesh_value_bool(bool v, RLMeshValue** out)`; `rlmesh_value_as_bool(const RLMeshValue*, bool* out)` | `Value(bool)`; `bool as_bool() const` |
| `int` | `Value::Int(i)`, `From<i64>`, `From<i32>`; `fn as_int(&self) -> Result<i64>` | `rlmesh_value_int(int64_t v, RLMeshValue** out)`; `rlmesh_value_as_int(const RLMeshValue*, int64_t* out)` | `template <std::integral I> Value(I)` for every integer type except `bool`, `char`, `wchar_t`, `char8_t`, `char16_t` and `char32_t` (those overloads are deleted), so `Value(0)` is `Int` and `signed char` / `unsigned char` (`std::int8_t` / `std::uint8_t`) build `Int`, as Python accepts any `int`; an unsigned value above `INT64_MAX` throws `rlmesh::ValueError`; `int64_t as_int() const` |
| `float` | `Value::Float(f)`, `From<f64>`, `From<f32>`; `fn as_float(&self) -> Result<f64>` | `rlmesh_value_float(double v, RLMeshValue** out)`; `rlmesh_value_as_float(const RLMeshValue*, double* out)` | `template <std::floating_point F> Value(F)`; `double as_float() const` |
| `str` | `Value::Text(s)`, `From<&str>`, `From<String>`; `fn as_text(&self) -> Result<&str>` | `rlmesh_value_text(RLMeshStr v, RLMeshValue** out)` (copies; invalid UTF-8 → `RLMESH_ERR_INVALID_ARGUMENT`); `rlmesh_value_as_text(const RLMeshValue*, RLMeshStr* out)` (borrowed, SPC-13) | `Value(const char*)`, `Value(std::string_view)`, `Value(std::string)`; `std::string_view as_text() const` |
| `bytes` | `Value::Bytes(b)`, `From<Vec<u8>>`, `From<&[u8]>`; `fn as_bytes(&self) -> Result<&[u8]>` | `rlmesh_value_bytes(const uint8_t* data, size_t len, RLMeshValue** out)` (copies; `data` NULL only when `len` is 0); `rlmesh_value_as_bytes(const RLMeshValue*, const uint8_t** out_data, size_t* out_len)` (borrowed) | `static Value bytes(std::span<const std::byte>)`; `std::span<const std::byte> as_bytes() const` |
| `Tensor` | `Value::Tensor(t)`, `From<Tensor>`; `fn as_tensor(&self) -> Result<&Tensor>` | `rlmesh_value_tensor(const RLMeshTensor*, RLMeshValue** out)` (shares); `rlmesh_value_as_tensor(const RLMeshValue*, RLMeshTensor** out)` (new shared reference, [Spaces](spaces.md)) | `Value(Tensor)`; `Tensor as_tensor() const` |
| `list` | `Value::List(v)`, `From<Vec<Value>>`; `fn as_list(&self) -> Result<&[Value]>` | `rlmesh_value_list(const RLMeshValue* const* items, size_t n, RLMeshValue** out)` (`items` NULL only when `n` is 0) | `static Value list(std::vector<Value>)`; `std::vector<Value> as_list() const` |
| `tuple` | `Value::Tuple(v)`; `fn as_tuple(&self) -> Result<&[Value]>` | `rlmesh_value_tuple(const RLMeshValue* const* items, size_t n, RLMeshValue** out)` | `static Value tuple(std::vector<Value>)`; `std::vector<Value> as_tuple() const` |
| `dict` | `Value::Dict(m)`, `From<Metadata>`; `fn as_dict(&self) -> Result<&Metadata>` | `rlmesh_value_dict(const RLMeshStr* keys, const RLMeshValue* const* values, size_t n, RLMeshValue** out)` (COR-21); `rlmesh_value_dict_key_at(const RLMeshValue*, size_t i, RLMeshStr* out)` (`{NULL, 0}` past the end), `rlmesh_value_dict_value_at(const RLMeshValue*, size_t i, const RLMeshValue** out)` (insertion order), `rlmesh_value_dict_get(const RLMeshValue*, RLMeshStr key, const RLMeshValue** out)` (`*out = NULL` when absent), `rlmesh_value_dict_contains_key(const RLMeshValue*, RLMeshStr key, bool* out)` | `Value(Metadata)`, `static Value dict(Metadata)`; `Metadata as_dict() const` |
| `len(v)`, `v[i]` (list, tuple, dict) | `as_list()?.len()`, `as_list()?.get(i)` (and tuple, dict) | `rlmesh_value_len(const RLMeshValue*, size_t* out)`; `rlmesh_value_get(const RLMeshValue*, size_t i, const RLMeshValue** out)` (list or tuple item, borrowed) | `size_t size() const`; `Value operator[](size_t) const` (list or tuple) |
| `type(v)` | `fn kind(&self) -> ValueKind`; `#[non_exhaustive] enum ValueKind { None, Bool, Int, Float, Text, Bytes, Tensor, List, Tuple, Dict }` | `rlmesh_value_kind` | `ValueKind kind() const`; `enum class ValueKind { None = 1, Bool, Int, Float, Text, Bytes, Tensor, List, Tuple, Dict }` (namespace `rlmesh`, C numbers) |

C++ overload resolution: `Value(0)` and `Value(42L)` pick the integral
template (exact match) over `Value(bool)` and `Value(const char*)`
(conversions); `Value(true)` picks `Value(bool)`; `Value("x")` picks
`Value(const char*)` over `Value(bool)`; `Value('a')` and `Value(u'a')` do
not compile (`char` and the `charN_t` types are text units, not integers);
`Value(std::uint8_t{3})` is `Int(3)` (a byte is a number, as Python's
`int`).
`Value(std::nullptr_t) = delete;` makes `Value(nullptr)` fail to compile
(it would otherwise pick `Value(const char*)` and build text from a null
pointer); None is `Value()` or `Value::none()`.

Platform macros: Xlib defines object-like `None` and `Bool` (`X11/X.h`,
`X11/Xlib.h`), which a Linux rendering env often includes (through GL,
GLFW or a simulator viewer) before `rlmesh.hpp`. `rlmesh.hpp` wraps its
declarations in `#pragma push_macro("None")` / `#pragma push_macro("Bool")`
plus `#undef`, restored with `pop_macro` at its end (GCC, Clang and MSVC
all support the pragma), so the header compiles in either include order.
A translation unit that still has the macros defined cannot spell
`ValueKind::None` / `ValueKind::Bool` (or `DType::Bool`) itself; it uses
`is_none()` / `as_bool()`, or `#undef`s the macro after its X11 includes.
The C names (`RLMESH_VALUE_NONE`) are unaffected. Whether R-4 instead
renames such enumerators is rules' call (Cross-file (r5) note).

### Errors

Classes, bases and statuses are R-21; this table adds the members.

| Python                                                           | Rust                                                                                                                                                                                                    | C                                                                                                                                                                     | C++                                                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `RLMeshException` | exact class: `Error::Internal(String)`; `except RLMeshException`: `e.class().is_subclass_of(ErrorClass::RLMeshException)` (reflexive, as `issubclass`), true for Internal, Protocol, Environment and Model only | exact class: `RLMESH_ERR_INTERNAL` (99), C-only `RLMESH_ERR_PANIC` (7); `except RLMeshException`: `rlmesh_error_class_is_subclass(rlmesh_last_error_class(), RLMESH_ERROR_CLASS_RLMESH_EXCEPTION)`, true for statuses 99, 7, 9, 3, 4 only | `class RLMeshException : public RuntimeError`; the builtins are not under it (R-21) |
| `ProtocolException`                                              | `Error::Protocol(String)`                                                                                                                                                                               | `RLMESH_ERR_PROTOCOL` (9)                                                                                                                                             | `class ProtocolException : public RLMeshException`                                                      |
| `EnvironmentException` | `Error::Environment(EnvironmentError)`; author: `fn Error::environment(message: impl Into<String>) -> Error` (code `Internal`) | `RLMESH_ERR_ENVIRONMENT` (3); author: `rlmesh_callback_set_error(message, false)` in an env callback | `class EnvironmentException : public RLMeshException`; `explicit EnvironmentException(std::string message)` (code `Internal`, COR-20) |
| `EnvironmentException.code: str` | `EnvironmentError::code: ErrorCode`; `#[non_exhaustive] enum ErrorCode { Unspecified, Timeout, InvalidAction, NotReady, Busy, Unsupported, Internal, Crashed, Cancelled, Closed }`; `fn as_str(self) -> &'static str`, `Display` | `bool rlmesh_last_error_code(int32_t* out)` (writes the wire number of `RLMeshErrorCode` and returns true; returns false with `*out` untouched when the class has no code, COR-3); `const char* rlmesh_last_error_code_str(void)` (NULL when the class has no code); `typedef enum RLMeshErrorCode { RLMESH_ERROR_CODE_UNSPECIFIED = 0, RLMESH_ERROR_CODE_TIMEOUT = 1, RLMESH_ERROR_CODE_INVALID_ACTION = 2, RLMESH_ERROR_CODE_NOT_READY = 3, RLMESH_ERROR_CODE_BUSY = 4, RLMESH_ERROR_CODE_UNSUPPORTED = 5, RLMESH_ERROR_CODE_INTERNAL = 10, RLMESH_ERROR_CODE_CRASHED = 11, RLMESH_ERROR_CODE_CANCELLED = 12, RLMESH_ERROR_CODE_CLOSED = 13 } RLMeshErrorCode;` (the wire `EnvErrorCode` numbers, COR-3); `const char* rlmesh_error_code_str(RLMeshErrorCode)` (Python's spelling, `"UNSPECIFIED"` for 0; NULL for a number the enum does not name) | `enum class ErrorCode : std::int32_t { Unspecified = 0, Timeout = 1, InvalidAction = 2, NotReady = 3, Busy = 4, Unsupported = 5, Internal = 10, Crashed = 11, Cancelled = 12, Closed = 13 };` (C's numbers); `ErrorCode code() const`; `std::string_view to_string(ErrorCode)` |
| `EnvironmentException.is_recoverable: bool`                      | `EnvironmentError::is_recoverable: bool`                                                                                                                                                                | `int rlmesh_last_error_is_recoverable(void)`                                                                                                                          | `bool is_recoverable() const`                                                                           |
| `RecoverableEnvironmentException` | `Error::Environment(e)` with `e.is_recoverable`; author: `fn Error::environment_recoverable(message: impl Into<String>) -> Error` (code `Internal`) | status 3, `rlmesh_last_error_is_recoverable() != 0`; author: `rlmesh_callback_set_error(message, true)` in an env callback | `class RecoverableEnvironmentException : public EnvironmentException`; `explicit RecoverableEnvironmentException(std::string message)` (code `Internal`) |
| `RecoverableEnvironmentException.code`, `.is_recoverable`        | the same `EnvironmentError` fields                                                                                                                                                                      | the same functions                                                                                                                                                    | inherited `code()`, `is_recoverable()`                                                                  |
| `ModelException` (new, D-10); author: `ModelException(message, *, is_recoverable=False)` (COR-22) | `Error::Model(ModelError)`; author: `fn Error::model(message: impl Into<String>) -> Error`, `fn Error::model_recoverable(message: impl Into<String>) -> Error` (code `Internal`) | `RLMESH_ERR_MODEL` (4); author: `rlmesh_callback_set_error(message, recoverable)` in a model callback | `class ModelException : public RLMeshException`; `explicit ModelException(std::string message, bool is_recoverable = false)` (code `Internal`, COR-20) |
| `ModelException.code: str` | `ModelError::code: ModelErrorCode`; `#[non_exhaustive] enum ModelErrorCode { Unspecified, InvalidRequest, NotConfigured, Busy, Unsupported, Internal, Cancelled, Closed }`; `as_str`, `Display` | `rlmesh_last_error_code(&code)` as `RLMeshModelErrorCode`; `typedef enum RLMeshModelErrorCode { RLMESH_MODEL_ERROR_CODE_UNSPECIFIED = 0, RLMESH_MODEL_ERROR_CODE_INVALID_REQUEST = 2, RLMESH_MODEL_ERROR_CODE_NOT_CONFIGURED = 3, RLMESH_MODEL_ERROR_CODE_BUSY = 4, RLMESH_MODEL_ERROR_CODE_UNSUPPORTED = 5, RLMESH_MODEL_ERROR_CODE_INTERNAL = 10, RLMESH_MODEL_ERROR_CODE_CANCELLED = 12, RLMESH_MODEL_ERROR_CODE_CLOSED = 13 } RLMeshModelErrorCode;` (the wire `ModelErrorCode` numbers, COR-3); `rlmesh_last_error_code_str()`; `const char* rlmesh_model_error_code_str(RLMeshModelErrorCode)` (as `rlmesh_error_code_str`) | `enum class ModelErrorCode : std::int32_t { Unspecified = 0, InvalidRequest = 2, NotConfigured = 3, Busy = 4, Unsupported = 5, Internal = 10, Cancelled = 12, Closed = 13 };` (C's numbers); `ModelErrorCode code() const`; `std::string_view to_string(ModelErrorCode)` |
| `ModelException.is_recoverable: bool`                            | `ModelError::is_recoverable: bool`                                                                                                                                                                      | `rlmesh_last_error_is_recoverable()`                                                                                                                                  | `bool is_recoverable() const`                                                                           |
| `ValueError`; `AdapterResolutionError` detail | `Error::InvalidArgument(InvalidArgumentError)`; `struct InvalidArgumentError { pub message: String, pub adapter_resolution: Option<adapters::AdapterResolutionError> }` ([Adapters](adapters.md)) | `RLMESH_ERR_INVALID_ARGUMENT` (1); `RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR`; no detail beyond the message and the class (ADP-17) | `rlmesh::ValueError`, `rlmesh::AdapterResolutionError` (R-21) |
| `TypeError`, `RuntimeError`, `OSError`, `ConnectionError`, `TimeoutError`, `KeyboardInterrupt` | `Error::Type(String)`, `Error::Runtime(String)`, `Error::Os(String)`, `Error::Connection(String)`, `Error::Timeout(Duration)`, `Error::Cancelled` (R-21) | statuses 10, 11, 12, 5, 6, 8 (R-21) | R-21 classes |
| `str(exc)`                                                       | `Display` (R-22)                                                                                                                                                                                        | `const char* rlmesh_last_error_message(void)`                                                                                                                         | `what()`                                                                                                |
| `type(exc)`, `isinstance(exc, C)`                                | `fn Error::class(&self) -> ErrorClass`; `enum ErrorClass` (the 13 R-21 classes, Python names); `fn name(self) -> &'static str`; `fn is_subclass_of(self, base: ErrorClass) -> bool`                     | `RLMeshErrorClass rlmesh_last_error_class(void)` (`INVALID = 0`, then 1-13 in R-21 order); `const char* rlmesh_error_class_name(RLMeshErrorClass)`; `bool rlmesh_error_class_is_subclass(RLMeshErrorClass cls, RLMeshErrorClass base)` | the class tree; `catch`                                                                                 |
| —                                                                | —                                                                                                                                                                                                       | `RLMESH_ERR_INVALID_VALUE` (2): deprecated, never returned (R-21); `RLMESH_ERR_TRANSPORT`: deprecated alias of 5; `RLMESH_ERR_PANIC` (7): class `RLMeshException`    | —                                                                                                       |

Authoring, beyond the Environment and Model rows: a C callback raises any
other R-21 class (`ValueError`, `TypeError`, `RuntimeError`, …) with
`rlmesh_callback_set_error_class(cls, message)` (R-12), as a Python author
raises it from a hook; a C++ callback throws the R-21 class, and the
wrapper's trampoline records it and rethrows the author's own exception
object at the failing call through R-12's error payload channel (X-10). An
authored Environment or Model error has R-22's `[CODE] message` text from
the moment it is raised, in process too: `Error::environment("bad")`,
`rlmesh_callback_set_error("bad", false)` in an env callback,
`rlmesh::EnvironmentException("bad")` and Python
`EnvironmentException("bad")` all read `[INTERNAL] bad` (COR-5, COR-P8).
On the wire the error carries the raw `bad`, and the receiving client adds
the `[INTERNAL] ` prefix once, so the remote caller also reads
`[INTERNAL] bad` (COR-27).

### Workflow editions

Every `workflow_edition=` keyword is a field of its call's options struct;
the struct's owner names it. Python `str | None` is Rust `Option<String>`, C
`const char*` (NULL = None), C++ `std::optional<std::string>`.

| Python                                                                                 | Rust                                                                                       | C                                                                                                            | C++                                                                                    |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `current_workflow_edition() -> str`                                                    | `fn current_workflow_edition() -> &'static str`                                            | `const char* rlmesh_current_workflow_edition(void)` (static)                                                 | `std::string_view current_workflow_edition()`                                          |
| `WorkflowEditionWarning` (`UserWarning`)                                               | `WarningCategory::WorkflowEditionWarning` (R-23)                                           | `RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING` (R-23); `rlmesh_warning_category_name`, `rlmesh_warning_category_is_subclass(cls, RLMESH_WARNING_CATEGORY_USER_WARNING)` is true (R-23) | `WarningCategory::WorkflowEditionWarning` (R-23)                                       |
| `RLMESH_WORKFLOW_EDITION` (environment variable)                                       | read by the resolver (COR-6)                                                               | same                                                                                                         | same                                                                                   |
| `run(…, workflow_edition=None)`, `Model.run(…)`                                        | `RunOptions::workflow_edition` ([Running](running.md))                                     | `RLMeshRunOptions.workflow_edition`                                                                          | `RunOptions::workflow_edition`                                                         |
| `session(…)`, `Model.session(…)`, `RemoteModel.session(…, workflow_edition=None)`      | `SessionOptions::workflow_edition` ([Running](running.md))                                 | `RLMeshSessionOptions.workflow_edition`                                                                      | `SessionOptions::workflow_edition`                                                     |
| `RemoteEnv(…, workflow_edition=None)`, `RemoteVectorEnv(…)`                            | `RemoteEnvOptions::workflow_edition` ([Remote](remote.md))                                 | `RLMeshRemoteEnvOptions.workflow_edition`                                                                    | `RemoteEnvOptions::workflow_edition`                                                   |
| `ServeOptions(workflow_edition=None)`, `ServeOptions.workflow_edition: Optional[str]`  | `ServeOptions::workflow_edition: Option<String>` ([Env](env.md); COR-8)                    | `RLMeshServeOptions.workflow_edition`                                                                        | `ServeOptions::workflow_edition`                                                       |
| `EnvFactory.serve(…, workflow_edition=None)`                                           | `EnvFactoryServeOptions::workflow_edition` ([Env](env.md))                                 | `RLMeshEnvFactoryServeOptions.workflow_edition`                                                              | `EnvFactoryServeOptions::workflow_edition`                                             |
| `Model.workflow_edition`, `EnvFactory.workflow_edition` (`ClassVar[str \| None] = None`) | `fn workflow_edition() -> Option<String>`, associated, default `None`, on `Policy` and `EnvFactory` ([Model](model.md), [Env](env.md); ENV-14) | `const char* workflow_edition` in the vtable / factory attributes (NULL = None) | `static std::optional<std::string> workflow_edition()` (absent = None) |
| `Session.selected_workflow_edition: str` | `fn selected_workflow_edition(&mut self) -> Result<String>` + `_async` (connects; [Running](running.md)) | `RLMeshStatus rlmesh_session_selected_workflow_edition(RLMeshSession*, RLMeshBytes* out)` | `std::string selected_workflow_edition()` |
| `RemoteEnv.selected_workflow_edition: str` | `fn selected_workflow_edition(&self) -> &str` ([Remote](remote.md)) | `RLMeshStatus rlmesh_remote_env_selected_workflow_edition(const RLMeshRemoteEnv*, RLMeshBytes* out)` | `std::string selected_workflow_edition() const` |
| `RemoteVectorEnv.selected_workflow_edition: str` | `fn selected_workflow_edition(&self) -> &str` ([Remote](remote.md)) | `rlmesh_remote_vector_env_selected_workflow_edition(const RLMeshRemoteVectorEnv*, RLMeshBytes* out)` | `std::string selected_workflow_edition() const` |

The Experimental sandbox clients' `selected_workflow_edition` follows Remote.

### Describe

| Python                                                                                                         | Rust                                                                                                                             | C                                                                                                                                                                                      | C++                                                                                                                                         |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `describe(obj, *, kind=None, generated_at=None) -> dict[str, Any]`                                             | `fn describe<'a>(obj: impl Into<Describable<'a>>, options: DescribeOptions) -> Result<Metadata>`                                 | `RLMeshStatus rlmesh_model_describe(const RLMeshModel*, const RLMeshDescribeOptions*, RLMeshValue** out)`; `rlmesh_env_factory_describe(RLMeshEnvFactory*, const RLMeshDescribeOptions*, RLMeshValue** out)` | `template <class T> Metadata describe(const DescribeOptions& = {})`; `template <class T> Metadata describe(T& obj, const DescribeOptions& = {})` |
| `describe_json(obj, *, kind=None, generated_at=None, served_env=None, workflow_edition=None) -> str`           | `fn describe_json<'a>(obj: impl Into<Describable<'a>>, options: DescribeJsonOptions<'a>) -> Result<String>`                      | `RLMeshStatus rlmesh_model_describe_json(const RLMeshModel*, const RLMeshDescribeJsonOptions*, RLMeshBytes* out)`; `RLMeshStatus rlmesh_env_factory_describe_json(RLMeshEnvFactory*, const RLMeshDescribeJsonOptions*, RLMeshBytes* out)`     | `template <class T> std::string describe_json(const DescribeJsonOptions& = {})`; `template <class T> std::string describe_json(T& obj, const DescribeJsonOptions& = {})`                                 |
| keyword sets                                                                                                   | `struct DescribeOptions { kind, generated_at }`; `struct DescribeJsonOptions<'a> { kind, generated_at, served_env, workflow_edition }` (COR-11) | `RLMeshDescribeOptions`, `RLMeshDescribeJsonOptions` (`struct_size` first; `RLMESH_*_INIT`, `_MIN_SIZE`; R-19)                                                                        | `struct DescribeOptions`, `struct DescribeJsonOptions`                                                                                      |
| `kind: str \| None` (`"env"`, `"model"`)                                                                        | `kind: Option<DescribeKind>`; `enum DescribeKind { Env, Model }`, `as_str()` = `"env"` / `"model"` (COR-14)                      | `bool has_kind; RLMeshDescribeKind kind;` (`INVALID = 0`, `ENV = 1`, `MODEL = 2`)                                                                                                      | `std::optional<DescribeKind> kind`                                                                                                          |
| `generated_at: str \| None` (RFC 3339)                                                                          | `generated_at: Option<String>`                                                                                                   | `const char* generated_at`                                                                                                                                                             | `std::optional<std::string> generated_at`                                                                                                   |
| `served_env: object \| None`: an env, a vector env, or a list of lane envs (lane 0 is read; `py/_describe.py:411-426`) | `served_env: Option<ServedEnv<'a>>`; `ServedEnv` is [Env](env.md#servedenv-cores-describe_jsonserved_env)'s (R-27; COR-23) | `const RLMeshEnv* const* served_env; size_t served_env_len;` (lanes; one lane = an env; R-13) and `const RLMeshVectorEnv* served_vector_env;` (NULL = not a vector env; setting both is `RLMESH_ERR_INVALID_ARGUMENT`); borrowed for the call; defined in [Env](env.md#servedenv-cores-describe_jsonserved_env) | `std::optional<ServedEnv> served_env`; `ServedEnv` is Env's |
| `workflow_edition: str \| None` (tri-state, COR-12)                                                             | `workflow_edition: Option<String>`                                                                                               | `const char* workflow_edition` (NULL ≠ `""`)                                                                                                                                           | `std::optional<std::string> workflow_edition`                                                                                               |
| `obj` a factory or model class                                                                                 | `Describable::model::<P: Policy>()`; `Describable::env_factory::<F: EnvFactory + Default>()` (COR-10)                           | a model type: `RLMeshStatus rlmesh_model_vtable_describe(const RLMeshModelVtable*, const RLMeshDescribeOptions*, RLMeshValue** out)`, `RLMeshStatus rlmesh_model_vtable_describe_json(const RLMeshModelVtable*, const RLMeshDescribeJsonOptions*, RLMeshBytes* out)` (static fields only, no slot called, COR-10); an env factory type: — (C cannot construct one; describe a handle, ENV-19) | `describe<T>()`, `describe_json<T>()` (COR-10)                                                                                                       |
| `obj` a factory or model instance | `impl<'a> From<&'a Model> for Describable<'a>` (the wrapped policy's type-level items, COR-10, COR-P3); `impl<'a, F: EnvFactory> From<&'a mut F> for Describable<'a>` | `const RLMeshModel*`, `RLMeshEnvFactory*` | `describe(obj)`, `describe_json(obj)` |
| `obj` a bare make or predict callable, with `kind`                                                             | the `Model` or factory built from the closure ([Model](model.md), [Env](env.md))                                                 | the handle built from a vtable                                                                                                                                                         | the wrapper built from the callable                                                                                                         |
| `obj` a `"module:Class"` string                                                                                | — (COR-15)                                                                                                                       | —                                                                                                                                                                                      | —                                                                                                                                           |
| `EnvFactory.describe()`, `Model.describe()`                                                                    | owned by [Env](env.md) and [Model](model.md): `describe(<the type>, kind)`                                                      | `rlmesh_env_factory_describe(f, NULL, out)`, `rlmesh_model_vtable_describe(&vt, NULL, out)` (COR-10)                                                                                                     | owned by Env and Model                                                                                                                      |
| envelope `target`, `params`, `runtime`, badge `error_type` (reflected: `module:qualname`, the `make` / `load` signature, `collect_peer_info()`, `type(exc).__name__`) | `target.qualname` `rust:<type_name>`; `params` from `LOAD_KEYS` / `tag_params`; Rust PeerInfo; R-21 class name (COR-25) | `native:<id>` / `native:model`; `params` from `load_keys` / `tag_params`; C PeerInfo; `rlmesh_error_class_name` (COR-25) | as C (COR-25) |
| `DESCRIBE_METADATA_KEY` (`"rlmesh.describe.v1"`)                                                               | `pub const DESCRIBE_METADATA_KEY: &str`                                                                                          | `#define RLMESH_DESCRIBE_METADATA_KEY "rlmesh.describe.v1"`                                                                                                                            | `inline constexpr std::string_view DESCRIBE_METADATA_KEY`                                                                                   |
| `DESCRIBE_SCHEMA_VERSION` (`1`)                                                                                | `pub const DESCRIBE_SCHEMA_VERSION: i64 = 1` (COR-19)                                                                            | `#define RLMESH_DESCRIBE_SCHEMA_VERSION 1`                                                                                                                                             | `inline constexpr int64_t DESCRIBE_SCHEMA_VERSION = 1`                                                                                      |
| `ENV_RESET_OPTIONS_KEY` (`"rlmesh.env.v1.reset_options"`)                                                      | `pub const ENV_RESET_OPTIONS_KEY: &str`                                                                                          | `#define RLMESH_ENV_RESET_OPTIONS_KEY "rlmesh.env.v1.reset_options"`                                                                                                                   | `inline constexpr std::string_view ENV_RESET_OPTIONS_KEY`                                                                                   |

### Build info and version

| Python                                                             | Rust                                                                                                     | C                                                                                                                                                  | C++                                                                       |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `build_info() -> BuildInfo`                                        | `fn build_info() -> BuildInfo`                                                                           | `RLMeshStatus rlmesh_build_info(RLMeshBuildInfo* out)` (out-struct, R-19)                                                                          | `BuildInfo build_info()`                                                  |
| `BuildInfo` (frozen; `__repr__`)                                   | `#[non_exhaustive] #[derive(Clone, Copy, Debug)] pub struct BuildInfo`; `Display` = Python's repr        | `typedef struct RLMeshBuildInfo { size_t struct_size; … } RLMeshBuildInfo;` `RLMESH_BUILD_INFO_INIT`, `RLMESH_BUILD_INFO_MIN_SIZE`; `RLMeshStatus rlmesh_build_info_repr(RLMeshBytes* out)` | `struct BuildInfo`; `operator<<`, `std::string to_string(const BuildInfo&)` |
| `BuildInfo.version: str`                                           | `pub version: &'static str`                                                                              | `const char* version`                                                                                                                              | `std::string_view version`                                                |
| `BuildInfo.protocol_generation: str`                               | `pub protocol_generation: &'static str`                                                                  | `const char* protocol_generation`                                                                                                                  | `std::string_view protocol_generation`                                    |
| `BuildInfo.workflow_edition: str`                                  | `pub workflow_edition: &'static str`                                                                     | `const char* workflow_edition`                                                                                                                     | `std::string_view workflow_edition`                                       |
| `BuildInfo.workflow_edition_base: str`                             | `pub workflow_edition_base: &'static str`                                                                | `const char* workflow_edition_base`                                                                                                                | `std::string_view workflow_edition_base`                                  |
| `BuildInfo.supported_workflow_editions: list[str]` (newest first)  | `pub supported_workflow_editions: &'static [&'static str]`                                               | `const char* const* supported_workflow_editions; size_t supported_workflow_editions_len;`                                                          | `std::vector<std::string_view> supported_workflow_editions`               |
| `BuildInfo.build_cohort: str`                                      | `pub build_cohort: &'static str`                                                                         | `const char* build_cohort`                                                                                                                         | `std::string_view build_cohort`                                           |
| `BuildInfo.build_source: str` (`release`, `package`, `git`)        | `pub build_source: BuildSource`; `enum BuildSource { Release, Package, Git }`, `as_str()`, `Display`     | `RLMeshBuildSource build_source` (`INVALID = 0`, `RELEASE = 1`, `PACKAGE = 2`, `GIT = 3`); `const char* rlmesh_build_source_str(RLMeshBuildSource)` | `BuildSource build_source`; `std::string_view to_string(BuildSource)`     |
| `BuildInfo.git: Optional[str]`                                     | `pub git: Option<&'static str>`                                                                          | `const char* git` (NULL = None)                                                                                                                    | `std::optional<std::string_view> git`                                     |
| `__version__` (PEP 440, `0.1.0rc12`)                               | `pub const VERSION: &str` (SemVer, `0.1.0-rc.12`; COR-18)                                                | `#define RLMESH_VERSION "0.1.0-rc.12"`                                                                                                             | `inline constexpr std::string_view VERSION`                               |
| `__build__` (deprecated)                                           | — (D-8)                                                                                                  | —                                                                                                                                                  | —                                                                         |
| `sanitize_metadata(info)`                                          | — (D-9)                                                                                                  | —                                                                                                                                                  | —                                                                         |
| `NO_ADAPTER`, `RANDOM_SAMPLE`                                      | [Model](model.md), [Running](running.md)                                                                 | same                                                                                                                                               | same                                                                      |
| `rlmesh.params` (`Param`, `ParamSpec`, `Vector`, `params.*Error`); `rlmesh.platform` (`PlatformError`) | deferred (Experimental, R-21)                                     | deferred                                                                                                                                           | deferred                                                                  |

## Python changes

P-1 to P-5 (error classes, codes, `ModelException`), P-6 (resolver in the
facade), P-7 (`sanitize_metadata` label), P-8 (the one timeout text
`operation timed out after {:?}`, COR-5) and P-9 (the env server treats only
a recoverable `EnvironmentException` as recoverable, COR-4) are owned by
[rules](rules.md#python-changes) and apply to this area. Core adds:

| ID      | Change                                                                                                                                                                                                                                                                                                                        | Where                                                                                       | Justification                                                                                                                                                         |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| COR-P1  | `describe` / `describe_json` raise `ValueError` when an explicit `kind` contradicts the object: `kind="env"` on an object without `make`, or `kind="model"` on an object without `predict` (the attribute auto-detection keys on, `py/_describe.py:242-245`). Today `kind` is honoured as given and selects the construction method (`make` for `"env"`, `load` for `"model"`, `:237-240`); an object lacking it falls through to the bare-callable path (`:589-604`, `:717-720`), producing a wrong envelope. Unchanged: bare callables (no `make`, no `predict`); a duck-typed policy *class* with `predict` but no `load`, which keeps describing with no params (no `load` signature to read; the class itself is the callable `_as_callable` returns, `:596-604`) and its corners by attribute presence (`:204-224`), under auto-detection or `kind="model"`; and a non-callable *instance* of such a class, which keeps raising `TypeError` (`not a factory, model, or callable`, `:717-720`) under either, COR-13's `TypeError` case. | `py/_describe.py:237-245`, `:589-604`, `:717-720` | Natively the target's type fixes the kind, so `kind` can only be a cross-check; a silent wrong artifact is worse than an error. The classmethods pass the matching kind (`py/_authoring.py:171-175`, `py/_models/base.py:521-525`). COR-14. |
| COR-P2  | `Model.serve(address, options=ServeOptions(workflow_edition=X))` adopts `X` as the call's value, ahead of `RLMESH_WORKFLOW_EDITION`, as `EnvServer` already does (`py/_server.py:283-290`) and as `Model.serve`'s docstring says (`py/_models/base.py:1491-1493`). Today the code passes it as the served peer's option, below the environment variable (`:1500`, `py/_editions.py:139-149,269-273`). | `py/_models/base.py:1498-1501`: `serve_options_declaring(options, call=options.workflow_edition if options is not None else None, declared=…)` | One rule for every serve call that takes `ServeOptions` (COR-6). Only observable with both the variable and the option set. |
| COR-P3  | `describe(rlmesh.Model(policy))` (a wrapping `Model`, `_policy is not self`) reads the type-level items (target qualname, edition declaration, params, variants, corners, class-level `native_chunk`) from the wrapped policy, exactly as `describe(policy)` would: from `type(policy)`, or from the callable itself for a bare callable. `spec` stays the model's. Today they come from the wrapper class `rlmesh.Model`: qualname `rlmesh._native:Model`, no edition, `params` of `Model.load`, corners `[]`. | `py/_describe.py:164-200,204-224,252-266,555-556,579-598` | Natively `Model` always wraps a `Policy`, so the wrapper's items are never what an author declared (D-16). A `Model` subclass instance (`_policy is self`) is unchanged. |
| COR-P4  | A `ModelException` raised from a Python model's predict corner becomes the Model class with code `INTERNAL` and the exception's `is_recoverable`, instead of `Internal` (`pyrs/model.rs:87-90`). `ModelException(message, *, is_recoverable=False)` is its authoring constructor; an authored one reports `code == "INTERNAL"`. In process the author's own exception is still re-raised unchanged (R-12). Any other exception stays the non-recoverable `INTERNAL` model failure it is today. The wire message is the raw one (COR-P11). | `pyrs/model.rs:87-90`, the `ModelException` class (P-1) | Rust (`Error::model_recoverable`) and C (`rlmesh_callback_set_error(msg, true)`) already let an author mark a model failure recoverable (R-21); Python and C++ gain the same (COR-22). |
| COR-P5  | `describe_json(…, served_env=[])` raises `ValueError` instead of `IndexError` (`served_env[0]`, `py/_describe.py:411-412`). | `py/_describe.py:411-412` | `IndexError` has no R-21 row (D-16); an empty lane list is an invalid argument in every language ([Env](env.md) `ServedEnv`, COR-23). |
| COR-P6  | A wrapping `rlmesh.Model(policy)` (`_policy is not self`) resolves its edition for run, session and serve with `declared=getattr(type(policy), "workflow_edition", None)`, read as COR-P3's describe reads it, instead of `type(self).workflow_edition` (the wrapper class's `None`). For run and session it is authored (warns when it floats) unless it wraps a bare callable (`_policy is None`); today `authored=self._policy is self` makes every wrapped policy object unauthored. Serve stays authored as a served peer (`serve_options_declaring`'s default, `py/_editions.py:257`). A `Model` subclass (`_policy is self`) is unchanged. | `py/_models/base.py:1559-1571` (`_declared_workflow_edition`), `:1500` (`serve`) | Natively the policy type is the declaration surface and the authored participant (`Policy::workflow_edition()`, the vtable field, the C++ static; [Model](model.md)), so `Model::new(Greedy, …)` reads Greedy's declaration and warns when it floats; Python's `Model(Greedy())` must too (D-16, COR-7). Keeps run, session and serve consistent with describe after COR-P3. |
| COR-P7  | `describe` reads `native_chunk` from the type, never from an instance attribute: `getattr(cls, "native_chunk", None)` where `cls` is the class itself for a class, `type(obj)` for an instance (the wrapped policy's type for a wrapping `Model`, COR-P3), and the callable itself for a bare callable. Today it reads `getattr(obj, "native_chunk", None)` on the object (`py/_describe.py:198`), so a `Model` subclass instance (`_policy is self`), which ran `load()` in `__init__` (`py/_models/base.py:613-617`), reports a K that `load()` set from the checkpoint (`:508-518`) instead of the class declaration. | `py/_describe.py:195-200` | The code's own comment says a K set in `load()` is invisible to describe on purpose (`:195-197`); every native instance form reports only the static declaration (COR-10, MDL-13), so the instance read is a Python quirk (D-16). |
| COR-P8  | `EnvironmentException` (and so `RecoverableEnvironmentException` and user subclasses) and `ModelException` format their text as `[CODE] message`: `__str__` returns `f"[{self.code}] {message}"`, where `message` is what `BaseException.__str__` returns. An authored `EnvironmentException("bad")` therefore reads `[INTERNAL] bad` (the class-level code, `pyrs/types.rs:95-101`) in process, including when R-12 re-raises it unchanged. The client builds the exceptions it raises with the raw message and sets `code` and `is_recoverable` on the instance, instead of preformatting `[CODE] message` into the message (`pyrs/types.rs:61`), so their `str` is unchanged and `args[0]` becomes the raw message. | `pyrs/types.rs:24-36,61-69` (the exception classes, `env_error_to_py`), the `ModelException` class (P-1) | R-22 makes `str(exc)` equal to Rust `Display`, C `rlmesh_last_error_message` and C++ `what()`, which read `[INTERNAL] bad` for the same authored error (`Error::environment("bad")`, `rlmesh_callback_set_error`, `EnvironmentException("bad")`); today Python's authored text is the bare `bad` while the same failure reads `[INTERNAL] bad` once it crosses the wire (D-16). Changelog entry: `str()` of an authored env or model exception gains its code prefix. |
| COR-P9  | `describe` of a model (class, instance or wrapping `Model`, read from the type as COR-P7 says) refuses a static declaration Model construction or resolve would refuse, instead of recording it: a class-level `native_chunk` with no `predict_chunk` / `predict_chunk_batch` raises `TypeError` with the construction text (`py/_models/base.py:727-740`), and a class-level int `native_chunk` below 1 raises `ValueError` with the resolve text `native_chunk must be >= 1 (or None when undeclared), got …` (`:1060-1063`). Today any int is recorded (`py/_describe.py:198-200`) and corners are not checked. | `py/_describe.py:195-200` | The native type form checks the same declaration (COR-10, COR-13): C's vtable is refused as `rlmesh_model_new` refuses it. A static declaration's validity does not depend on the run, unlike an edition refusal, which depends on `RLMESH_WORKFLOW_EDITION` and is recorded (COR-12); an envelope for a model that can never be built is a silent wrong artifact (as COR-P1). |
| COR-P10 | The metadata codec and `sanitize_metadata` turn an `rlmesh.Tensor` into the nested list of its elements, as they already turn a NumPy array (`tolist`), instead of raising `TypeError` (`value of type 'Tensor' is not supported in RLMesh metadata`; the codec's unwrapping probes only `_asdict` / `tolist` / `item`, which `rlmesh.Tensor` lacks, `pyrs/spaces/value_codec/normalization.rs:26-35`) or stringifying it. No new public `Tensor` method: the codec matches the class. | `pyrs/spaces/value_codec/metadata.rs:30-104` (`py_any_to_meta_value`), `py/_metadata.py:58-69` (`_unwrap`) | `rlmesh.Tensor` is a `Value` kind, and native `Metadata` carries it; COR-26 encodes it as Python encodes an array, so every language sends the same `info` (D-16). |
| COR-P11 | The env server and the model worker put the raw message of an author's exception on the wire (COR-27): for an `EnvironmentException` (and its subclasses) or a `ModelException`, `BaseException.__str__(exc)`, without the `[CODE] ` prefix COR-P8 gives `str(exc)`; for any other exception, `str(exc)`. Neither adds a phase or type-name prefix. Today the env server sends `f"{phase} failed: {e}"`, whose PyErr `Display` adds the type name (`step failed: ValueError: x`, `step failed: EnvironmentException: bad`, `pyrs/server/py_environment.rs:457`), and `PyPredict::internal` sends PyErr's `Display` (`ValueError: x`, `pyrs/model.rs:87-90`), so after COR-P8 the client would read `[INTERNAL] step failed: EnvironmentException: [INTERNAL] bad`. Texts the server adds for its own failures (`{phase} task panicked: …`, `{phase}: the serving thread has stopped`, `:445-454`) are unchanged. | `pyrs/server/py_environment.rs:457`, `pyrs/model.rs:87-90` | A Rust env already sends the raw message (`gym_error_to_env_error`, `rs/env/wire.rs:692-708`), and every receiver formats `[CODE] message` itself (`rs/error.rs:199,253`, `pyrs/types.rs:61`), so one rule keeps the remote text equal everywhere: an authored `EnvironmentException("bad")` from `step` reads `[INTERNAL] bad`, an author's `ValueError("x")` reads `[INTERNAL] x` (D-16). Changelog entry: the remote text of a Python env or model failure loses its `{phase} failed: <Type>: ` prefix. |

No other Python behaviour changes here. In particular, `describe_json`'s
`workflow_edition`, the resolver's rungs (D-7, P-6), the warning's category
and text, and the build info keep Python's behaviour (COR-6, COR-7, COR-12);
only the inputs a wrapping `Model` passes to the resolver change (COR-P6).
Metadata keeps Python's codec except the `Tensor` arm (COR-P10). The wire
text of an author's env or model failure follows COR-27 (COR-P11).

## Current differences

### Rust

- **Values.** No `rlmesh::Value`, `Metadata`, `InfoDict` or `Kwargs`. The
  facade exports `SpaceValue` (`rs/lib.rs:166`) and `MetaMap` / `MetaValue`
  (`rs/spaces.rs:25`); the adapter `Value` (`crates/rlmesh-adapters/src/apply/value.rs:22-30`)
  is internal and lacks `None`, `Bool`, `Int` and `Tuple` (X-2, G-17).
- **Error variants** (`rs/error.rs:74-89`: `Address`, `Connection`,
  `Timeout`, `Environment`, `Model`, `Server`, `Internal`): add
  `InvalidArgument(InvalidArgumentError)` (new, absorbing `Address`),
  `Protocol`, `Type`, `Runtime`, `Os`, `Cancelled`, `Error::class()` and
  `ErrorClass`; fold `Server` into `Internal` (R-21).
- **Codes.** `ErrorCode` (`rs/error.rs:12-31`) lacks `Unsupported`, so the wire
  value becomes `Internal` (`:191`, P-5); add `as_str()`. `ModelError`
  (`:50-55`) has no `code`; the wire code is dropped at `:252-255`.
- **Protocol failures** become `Internal` (`rs/error.rs:205-209,248`); route
  them to `Error::Protocol` (P-3).
- **Text** (`rs/error.rs:163-173`): drop the `connection error:`,
  `environment error:`, `model error:` and `internal error:` prefixes
  (Environment and Model text becomes `[CODE] message`, P-1); timeout text
  becomes `operation timed out after {:?}` (COR-5).
- **Wire error text.** The model server sends `message: error.to_string()`
  (`rs/model/wire.rs:36-46`, `model_error_value`), today's prefixed `Display`
  (`model error: …`, `internal error: …`); once `Display` is R-22's
  `[CODE] message`, the client would read `[INTERNAL] [INTERNAL] bad`. It
  sends `ModelError.message` (or `EnvironmentError.message`) for an
  Environment- or Model-class error and the R-22 text for any other
  (COR-27). The env server already sends the raw message
  (`rs/env/wire.rs:692-708`).
- **`Error::is_recoverable()`** returns true for `Timeout` and `Connection`
  (`rs/error.rs:113-121`), which no Python exception reports (COR-4).
- **Authoring.** Add `Error::environment` / `environment_recoverable` beside
  `model` / `model_recoverable` (`rs/error.rs:97-110`). Env authors return
  `EnvRuntimeError` (`crates/rlmesh-spaces/src/errors.rs:31`), re-exported at
  `rs/spaces.rs:19` and `rs/prelude.rs:15`; it moves to `binding` (R-21, R-25).
- **Plumbing impls** at `rs/error.rs:195,205,211,219` become `pub(crate)`
  functions (R-25).
- **Editions.** `CURRENT_WORKFLOW_EDITION` (`rs/lib.rs:149`) is the cohort
  spelling (`crates/rlmesh-proto/src/lib.rs:31`), not
  `current_workflow_edition()` (the base, `pyrs/lifecycle.rs:96-98`); add the
  function and move the constant, `parse_declared_edition`,
  `PROTOCOL_GENERATION` and `supported_workflow_editions` (`rs/lib.rs:149-155`)
  to `binding` (R-25). No resolver, environment variable or warning (G-19,
  G-21).
- `RemoteEnv` (`rs/env/client.rs:414-417`) cannot declare an edition and has
  no `selected_workflow_edition`; `RemoteVectorEnv` declares only through
  `connect_declaring` (`rs/env/client.rs:81`).
- `RemoteModel::selected_workflow_edition` (`rs/model/remote.rs:279`) has no
  Python counterpart; removed when `Session` lands (COR-9).
- `RunLocalOptions::workflow_edition` (`rs/model/worker.rs:93`) moves to
  `RunOptions` ([Running](running.md)).
- `ServeOptions::workflow_edition` (`rs/serve_options.rs:76`) is trimmed on
  conversion but never validated (COR-8).
- **Describe.** No `describe`, `describe_json`, `Describable`, options or
  `DescribeKind`. `DESCRIBE_METADATA_KEY` and `DESCRIBE_SCHEMA_VERSION` live
  only in `crates/rlmesh-adapters/src/keys.rs:26,32`.
- **Build info.** No `build_info()`, `BuildInfo`, `BuildSource` or `VERSION`.
  `ENV_RESET_OPTIONS_KEY` is already at the root (`rs/lib.rs:164`).

### C

- **Statuses** (`h:61-72`, `capi/abi/status.rs:16-27`): add `PROTOCOL` (9),
  `TYPE` (10), `RUNTIME` (11), `OS` (12); rename `TRANSPORT` to `CONNECTION` (5) with a
  deprecated alias; stop returning `INVALID_VALUE` (2), which today reports a
  wrong-kind read (`h:186,195`, `capi/abi/status.rs:46-48`) that becomes
  `RLMESH_ERR_TYPE`.
- **Mapping** (`capi/abi/status.rs:57-64`): `Server` maps to `TRANSPORT`
  (`:59`) and becomes `INTERNAL`; the `_ => Internal` arm hides
  Protocol, Type and Runtime.
- **Last-error slot** (`capi/abi/status.rs:73-81`) stores message, flag and
  status only, copied from the by-value `CapiError` (`:29-33`) that carries
  the error across the callback-thread hop; add class and code to
  `CapiError`, to its `From<Error>` (`:54-71`), to `store_last_error` and
  `guard` (`:77-81,113-127`) and to the slot, and add `rlmesh_last_error_class`,
  `rlmesh_last_error_code`, `rlmesh_last_error_code_str`,
  `rlmesh_error_code_str`, `rlmesh_model_error_code_str`,
  `rlmesh_error_class_name`, `rlmesh_error_class_is_subclass` (`h:74-83`),
  and the `RLMeshErrorCode` / `RLMeshModelErrorCode` enums (COR-3).
  The message is Rust's prefixed `Display` (`:67`), fixed by R-22. The flag
  comes from `Error::is_recoverable()` (`:56`, COR-4).
- **Values.** `RLMeshValueKind` (`h:143-152`) and its constructors and
  accessors (`h:139-224`) project `SpaceValue`; they re-base on `Value` (X-2)
  as the [Value API](#value-api) lists. Today's functions (`h:161-222`):
  constructors return `RLMeshValue*` or NULL and become `RLMeshStatus` with
  an out-parameter; `rlmesh_value_box` → `rlmesh_value_tensor`;
  `rlmesh_value_as_tensor` returns a new shared `RLMeshTensor**` instead of
  filling a borrowed `RLMeshTensor` view; `rlmesh_value_discrete` /
  `_as_discrete` → `rlmesh_value_int` / `_as_int`; `rlmesh_value_text` and
  `_as_text` take `RLMeshStr` instead of `(const char*, size_t)`;
  `rlmesh_value_multi_binary`, `_multi_discrete`, `_array_len`,
  `_copy_multi_binary` and `_copy_multi_discrete` are removed, because those
  values are `TENSOR` (Spaces SPC-10); `rlmesh_value_tuple_get` →
  `rlmesh_value_get` (LIST or TUPLE, status + out); `rlmesh_value_dict_key` /
  `_dict_get_at` (sorted order) → `rlmesh_value_dict_key_at` /
  `_dict_value_at` (insertion order, COR-2); `rlmesh_value_dict_get` takes
  `RLMeshStr` and returns status + out; `rlmesh_value_tuple` and
  `rlmesh_value_dict` share their children instead of adopting them on
  success, take `const RLMeshValue* const*`, and `rlmesh_value_dict` takes
  `RLMeshStr` keys and accepts a repeated key (COR-21). Add
  `rlmesh_value_none`, `_bool`, `_as_bool`, `_float`, `_as_float`, `_bytes`,
  `_as_bytes`, `_list`, `_dict_contains_key`, `rlmesh_value_clone` and
  `rlmesh_value_equals`.
- **Version** (`h:44-52`): `RLMESH_ABI_VERSION_MAJOR/MINOR/PATCH` and
  `rlmesh_abi_version_major/minor/patch` carry the package semver; replace
  them with `RLMESH_VERSION` and `rlmesh_build_info` (X-1, G-28).
- **Missing:** `rlmesh_current_workflow_edition`, `rlmesh_build_info`,
  `RLMeshBuildInfo`, `RLMeshBuildSource`, the describe functions and options,
  `RLMeshDescribeKind`, and the `RLMESH_DESCRIBE_*` and
  `RLMESH_ENV_RESET_OPTIONS_KEY` macros.
- **Editions.** `RLMeshRunOptions` (`h:449-463`) has no `workflow_edition`.
  `RLMeshServeOptions.workflow_edition` (`h:492`) already reads NULL and `""`
  as none, but the struct has no `struct_size` (G-23).
- **Served-env describe.** `rlmesh_env_describe_json(const RLMeshEnv*, …)`
  (`h:795`) reads the edition `rlmesh_env_bind` stored on the env
  (`capi/describe.rs:143-148`); with binding moved to `RLMeshEnvServer`
  ([Env](env.md)) it becomes the fenced
  `rlmesh_ext_env_server_describe_json(const RLMeshEnvServer*, …)` (COR-16).
  `RLMeshDescribeJsonOptions` takes lanes (`served_env` + `served_env_len`)
  or a vector env (`served_vector_env`) (COR-23). There is no model type form today; add
  `rlmesh_model_vtable_describe` and `_describe_json` (COR-10).

### C++

- **Errors.** One `Error` value class plus `Result<T>` (`hpp:45-69`, D-5,
  G-26) becomes the R-21 exception tree with `code()` and `is_recoverable()`.
- Wrapper-made errors use `RLMESH_ERR_INVALID_VALUE` (`hpp:366,642,716,729,779,800`)
  and `RLMESH_ERR_MODEL` for a misuse (`hpp:960`); each takes the class Python
  raises for the same check (`TypeError`, `ValueError` or `RuntimeError`).
- **Values.** `ValueRef` (`hpp:249`) is removed (R-27); the move-only `Value`
  (`hpp:378-400`) becomes copy-sharing over `rlmesh_value_clone`.
- `ServeOptions::workflow_edition` is `std::string` (`hpp:937`); it becomes
  `std::optional<std::string>`.
- `EnvServer::describe_json()` (`hpp:1439-1441`) becomes
  `rlmesh::ext::describe_json(const EnvServer&)` (COR-16).
- **Missing:** `Metadata` aliases, `build_info`, `BuildInfo`, `VERSION`,
  `current_workflow_edition`, the describe templates and the constants.

## Decisions

| ID     | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Status                 |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| COR-1  | `Value` has exactly Python's kinds: the six primitives, `Tensor`, `list`, `tuple`, `dict`. Space-specific kinds (`Box`, `Discrete`, `MultiBinary`, `MultiDiscrete`) are not value kinds: a Discrete action is an `Int`, a Box observation a `Tensor`. `bool` is not an `Int`. C kinds renumber 1-10 in ABI 5 (X-1).                                                                                                                                                                                                                                  | Settled                |
| COR-2  | `Metadata` (and so a dict `Value`) keeps insertion order, as a Python `dict` does; lookup and equality ignore order. Sampled and decoded Dict values therefore follow the space's declaration order (SPC-10; `pyrs/spaces/value_codec/codec.rs:55-64`). Maps that crossed the wire arrive in the wire's sorted order (`MetaMap`, `crates/rlmesh-spaces/src/meta.rs:15`), as in Python, with the wire's kinds (COR-26). Alternative: a sorted map (`BTreeMap`, `std::map`), which differs from Python in iteration order only. | Default — pending user |
| COR-3  | One code getter in C: `bool rlmesh_last_error_code(int32_t* out)` writes the wire number of whichever code the class carries (`RLMeshErrorCode` for the Environment family, `RLMeshModelErrorCode` for Model) and returns true; for any other class it returns false and leaves `*out` untouched, since 0 is the real wire value `UNSPECIFIED` in both enums and cannot mean "no code" (R-13). `rlmesh_last_error_code_str()` returns Python's `code` text, or NULL with no code. Like every last-error getter it does not change the slot. Both code enums (`RLMeshErrorCode`, `RLMeshModelErrorCode`, and C++ `ErrorCode` / `ModelErrorCode` with the same numbers; Errors table) are mirrors of the wire `EnvErrorCode` / `ModelErrorCode` (`crates/rlmesh-proto/proto/rlmesh/env/v1/interaction.proto:163-177`, `model/v1/interaction.proto:226-237`) and keep the wire numbers, `UNSPECIFIED = 0` included, so they are exempt from R-9's `*_INVALID = 0` although `rlmesh_error_code_str` / `rlmesh_model_error_code_str` take them as input: no call refuses a code, and each `_str` accepts any number (Python's spelling for a named value, `"UNSPECIFIED"` for 0, NULL for a number the enum does not name), as a decoder accepts an unknown wire value. Rust keeps the name `ErrorCode` for env codes; Python has no class name for either. | Settled |
| COR-4  | `Error::is_recoverable()` and `rlmesh_last_error_is_recoverable()` mean Python's `getattr(exc, "is_recoverable", False)`: the Environment or Model flag, false for every other class. Today's facade predicate (true also for `Timeout` and `Connection`, `rs/error.rs:113-121`) becomes a `pub(crate)` transport-retry function for its one in-facade user, `model_rpc` (`rs/model/local.rs:149`); the C flag stops reading it (`capi/abi/status.rs:56`, Current differences > C). The `is_recoverable` calls at `rs/model/local.rs:158` (`rlmesh_runtime::RuntimeError`) and `:227,247` (`rlmesh_grpc::error::Error`) are other crates' predicates and are unaffected.                                                                                                                                                                                                                                     | Default — pending user |
| COR-5  | Error text equals Python's `str(exc)` (R-22, whose table is authoritative): `[CODE] message` for Environment and Model (P-1), including an authored one in process before it crosses the wire (`[INTERNAL] bad` for `Error::environment("bad")`, C `rlmesh_callback_set_error("bad", …)`, C++ `EnvironmentException("bad")` / `ModelException("bad")` and, after COR-P8, Python's; the in-process R-12 rethrow keeps that text; the wire carries the raw message and the receiver adds the prefix once, COR-27); the raw message for Internal, Protocol, Type, Runtime, OSError and Connection; `invalid address: …` for an address `ValueError`, otherwise the raw message; `operation timed out after {:?}` of the `Duration` for Timeout, for every deadline (P-8; `pyrs/types.rs:44-58`); `cancelled` natively for a cancel (Python's signal-raised `KeyboardInterrupt` keeps the interpreter's empty text). Folding `Server` into `Internal` keeps Python's `server error: …` text. Errors Python's own code raises keep its text natively, except R-22's native texts that name `load` or `model` where Python names a qualname: the `LOAD_KEYS` unknown-key refusal `extra='passthrough' but load has no **kwargs to forward: a, b` and the default-`load` refusal `model received construction params (a, b) but does not override load(...) to apply them; implement load(**kwargs) to consume the binding` (names sorted), plus the Rust undefined-corner text. | Settled |
| COR-6  | One resolver in the facade (D-7, P-6). Rungs, highest first (`py/_editions.py:139-155`): (1) the call's own value: the `workflow_edition` field of `RunOptions`, `SessionOptions` or `RemoteEnvOptions`, and the `ServeOptions.workflow_edition` given to a serve call that takes `ServeOptions` (`EnvServer`, `Model.serve`), which is that call's caller-resolved answer and is adopted ahead of the environment variable (`py/_server.py:283-290`; [Env](env.md) construction step 6; Python's `Model.serve` aligns, COR-P2); (2) `RLMESH_WORKFLOW_EDITION`; (3) the served peer's option: `EnvFactory.serve(workflow_edition=)` / `EnvFactoryServeOptions::workflow_edition` (`py/serve.py:519-522`) and the CLI's `--workflow-edition`; (4) the type's declaration; then none. Rust, C and C++ read no project file; Python adds its `pyproject.toml` rung after the type's declaration (D-7). A blank call, option or declaration declares nothing and falls through; a set-but-blank environment variable stops resolution, floats, and does not warn. The winner is validated and trimmed; a refused edition fails with the `ValueError` class. `describe_json`'s `workflow_edition` is not a rung-1 value: when set it bypasses the resolver and is used as given, unvalidated (COR-12); when unset, describe runs rungs (2) and (4) only (Python adds `pyproject.toml`), never warns, and records a refusal instead of failing (`py/_describe.py:563-575`). | Settled |
| COR-7  | `WorkflowEditionWarning` fires at most once per process (`py/_editions.py:79,233-238`), only for an authored participant (the type that carries the declaration surface, as [Model](model.md) and [Env](env.md) define it, or a served peer) that floats, and not when a caller resolved it first (`resolved_by_caller`, `py/_editions.py:64`; the loopback server `run` stands up; natively, Running's env relay (RUN-25): a local env reached by `run` is resolved by the caller and does not warn). Natively a `Model`'s authored participant is its policy type; a `Model` built from a bare callable is not authored. Python today treats only a `Model` subclass as authored for run and session and reads the wrapper class's declaration (`authored=self._policy is self`, `declared=type(self).workflow_edition`, `py/_models/base.py:1559-1571`), so a wrapped policy object neither declares nor warns; COR-P6 aligns it, leaving only a wrapped bare callable unauthored. `describe` never warns (`py/_describe.py:570`). Category and handler are R-23. Text: Python keeps its own (`py/_editions.py:241-248`), which names the class attribute and `[tool.rlmesh]` in `pyproject.toml`, surfaces only Python has (D-7); its binding emits it with its caller attribution (P-6). Rust, C and C++ share one text naming their own surfaces: `no workflow edition declared: floating to '<edition>', which moves when rlmesh upgrades. Pin it with workflow_edition "<edition>" on the type or in the call's options; RLMESH_WORKFLOW_EDITION='' floats deliberately.` ("the type" is Rust's and C++'s `workflow_edition()`, C's vtable or attributes field). | Settled |
| COR-8  | Edition values in options structs are plain data (R-6): reading `ServeOptions::workflow_edition` back gives what was written, and validation runs at the consuming call (serve, bind, run, session, connect). Python validates and trims in its constructor; the error class is the same.                                                                                                                                                                                                                                                             | Settled                |
| COR-9  | `selected_workflow_edition` exists on `Session`, `RemoteEnv` and `RemoteVectorEnv` only, as in Python. On `Session` it connects first and fails with `Error::Runtime` for a local env, as Python raises `RuntimeError` (`py/_models/_eval.py:940-958`).                                                                                                                                                                                                                                                                                                | Settled                |
| COR-10 | Describe has a type form and an instance form, as Python does. Type form: reads only type-level items (edition, params, variants, corners, a type-level spec and `native_chunk`). Env spaces: one construction, `prepare()` and `make()` for the default branch, then one more of each per non-default `tag_params` branch, each reading that branch's `env_spec` for `env_contracts` (`py/_describe.py:184-187,340-352,461-477`). The type form default-constructs a fresh factory each time; the instance form runs `prepare()` / `make()` on the given instance each time (`:469`). A `served_env` replaces only the default branch's build (`:409-413`). Instance form: edition, params, variants, corners and `native_chunk` still come from the type; only `spec` comes from the instance (`:366-384`), plus, for an env factory, the builds above. `native_chunk` is always the type's static declaration, never a K that `load()` set on an instance (`py/_describe.py:195-197`; MDL-13; Python aligns, COR-P7). For a `Model`, the type is the wrapped policy's (COR-P3): `describe(&model)` reports what `describe::<P>()` reports, plus the model's `spec`. The envelope's `target`, `params`, `runtime` and badge `error_type` follow COR-25. The two type forms differ as Python's do. A model type form runs no user code and loads no weights (`py/_describe.py:586-589`, MDL-13): Rust `Describable::model::<P>()` reads `P`'s associated items; C `rlmesh_model_vtable_describe` / `_describe_json` read only the vtable's static fields (`spec`, `has_native_chunk` + `native_chunk`, `workflow_edition`, `load_keys` + `load_keys_len` as params, R-13's optional list: NULL = None, non-NULL with 0 = `[]`, and the non-NULL corner slots as corners), call no slot (not even `load`), need no `user_data`, and refuse a vtable `rlmesh_model_new` would refuse with the same status; C++ `describe<P>()` / `describe_json<P>()` for a policy type wrap them over the static `RLMeshModelVtable` the wrapper generates for `P`, constructing nothing. An env factory type form constructs, as Python's `obj()` does (`py/_describe.py:469`): Rust builds `F::default()` per build; C cannot construct a factory, so it describes a handle (ENV-19); C++ `describe<F>()` default-constructs one `F` and runs C's instance form on it, so its branches share that one factory, which differs from Python's fresh factory per build only when `prepare` / `make` leave state that changes a later branch. | Settled |
| COR-11 | `describe` and `describe_json` keep separate options structs, because their Python keyword sets differ. Python does not grow `served_env` / `workflow_edition` on `describe`.                                                                                                                                                                                                                                                                                                                                                                          | Settled                |
| COR-12 | `describe_json`'s `workflow_edition` is tri-state and never validated (`py/_describe.py:563-575`): None runs the resolver with no call or option value (environment variable, then declaration; Python adds `pyproject.toml`, D-7), without warning, and records a failure as `runtime.workflow_edition_error`; any other value is stripped, and a blank result means "the server resolved none", so the envelope reports `BuildInfo.workflow_edition` (the cohort spelling); a non-blank stripped value is used as given. | Settled |
| COR-13 | Describe is total for the target's own content: a failed env build or space read is badged in the envelope (`py/_describe.py:409-445`) and an invalid declaration is recorded (COR-12). The call itself fails, with the `ValueError` class, only for: an invalid `kind` (`py/_describe.py:237-239`) or one contradicting the target (COR-14, COR-P1); an invalid `generated_at` or any other envelope invariant failure (`invalid describe envelope: …`, `pyrs/adapters.rs:1030-1031`); an empty `served_env` lane list (COR-23, COR-P5); a model type form whose static declaration construction refuses, since there is no built model to badge: C `rlmesh_model_vtable_describe` / `_describe_json` fail with the status `rlmesh_model_new` returns for the same vtable (R-19's sizing statuses; `RLMESH_ERR_TYPE` for `context_corners` bits on NULL slots and for a static K without a chunk corner; `RLMESH_ERR_INVALID_ARGUMENT` for `has_native_chunk` with 0; [Model](model.md)), C++ `describe<P>()` throws the matching class, Rust `describe` / `describe_json` of `Describable::model::<P>()` (the constructor itself is infallible) fail with `Error::Type` for `NATIVE_CHUNK = Some(K)` without a chunk corner and with `Error::InvalidArgument` for `Some(0)`, and Python's describe raises `TypeError` / `ValueError` for the same declarations after COR-P9 (today it records any int `native_chunk` and checks no corners, `py/_describe.py:198-200`). Python also raises `TypeError` for an object it cannot classify or call (`py/_describe.py:246-249,717-720`); natively the target is typed, so those cases cannot be written. | Settled |
| COR-14 | `kind` stays in every native options struct for parity, but the target fixes the kind natively, so `kind` is a cross-check: a contradiction fails with the `ValueError` class, and Python adopts the same check (COR-P1). Alternative: keep Python's fall-through and drop `kind` natively.                                                                                                                                                                                                                                                             | Default — pending user |
| COR-15 | The `"module:Class"` entrypoint string is Python-only: resolving it is a Python import (`py/_describe.py:132-134`). Manifest status `exempt`, citing this decision.                                                                                                                                                                                                                                                                                                                                                                                    | Settled                |
| COR-16 | The served-env shorthand (`rlmesh_env_describe_json`, `EnvServer::describe_json()`) describes a served env with no factory, which Python cannot express, so it is a fenced extra (R-25) on the server in both languages, so C++ wraps C one to one (D-2): C `RLMeshStatus rlmesh_ext_env_server_describe_json(const RLMeshEnvServer* server, RLMeshBytes* out)`, C++ `std::string rlmesh::ext::describe_json(const EnvServer&)`; Rust has no such extra today and gains none (Rust callers describe with `served_env` before the env moves into the server). It equals `describe_json` with `served_env` = the server's lanes (or `served_vector_env` = its vector env) and `workflow_edition` = the edition the server resolved (COR-6), or `""` if none. The server owns binding in the target ([Env](env.md)), so the edition is read from it, not from an env handle (today `capi/describe.rs:143-148`). | Settled |
| COR-17 | Build info values are static and fixed at compile time. C returns borrowed `const char*` with process lifetime and frees nothing: R-20 reserves `RLMeshBytes` for owned text. `supported_workflow_editions` is the static table `SUPPORTED_WORKFLOW_EDITIONS` (`crates/rlmesh-proto/src/lib.rs:50`). `build_source` is an enum because its values are a closed set (R-9; `crates/rlmesh-proto/build.rs:199-234`). Rust `BuildInfo` stays `#[non_exhaustive]`: callers read it and never build it. Python defines no `__eq__`, so no equality (R-10). | Settled                |
| COR-18 | Native version constants use SemVer (`0.1.0-rc.12`), as `BuildInfo.version` does (`pyrs/lifecycle.rs:111-113`); Python's `__version__` is the PEP 440 spelling of the same release, a Python packaging quirk. `RLMESH_VERSION` is the header's version; `rlmesh_build_info()` reports the linked library's. A mise check keeps the hand-written macro equal to the workspace version.                                                                                                                                                                  | Settled                |
| COR-19 | `DESCRIBE_SCHEMA_VERSION` is `i64` in the facade (R-5: no Python check justifies a narrower type), though the envelope builder stores `u32` (`crates/rlmesh-adapters/src/keys.rs:32`).                                                                                                                                                                                                                                                                                                                                                                 | Settled                |
| COR-20 | C++ library-raised exceptions carry a code through an internal constructor. The public constructors take Python's authoring arguments: a message, plus `is_recoverable` for `ModelException` (COR-22); an authored throw sends code `INTERNAL` (R-21), and its `what()` is `[INTERNAL] message` (COR-5). `EnvironmentException` (and so its subclasses) and `ModelException` also keep the raw message in a private member that only the wrapper's trampoline reads (a `friend` in `rlmesh::detail`), and the trampoline passes that raw message, not `what()`, to `rlmesh_callback_set_error` (COR-27); there is no public accessor, since Python's counterpart is `args[0]` and R-25 adds no extra. | Settled |
| COR-21 | C `Value`s are immutable (R-11), so C builds a DICT in one call and has no insert or remove; Rust and C++ `Metadata` are mutable maps with Python `dict` semantics (Metadata rows). `rlmesh_value_dict` with a repeated key behaves as `dict(pairs)`: the first position, the last value. Today's C refuses a repeated key (`h:213-220`); Python has no such refusal. | Settled |
| COR-22 | Authored model failures follow R-21's authored-failure rule in every language: an author marks a model failure recoverable with Rust `Error::model_recoverable`, C `rlmesh_callback_set_error(msg, true)` in a model callback, C++ `throw rlmesh::ModelException(msg, true)` and Python `raise rlmesh.ModelException(msg, is_recoverable=True)` (COR-P4). The code sent is always `INTERNAL`, and the authored error's text is `[INTERNAL] message` in every language, in process and as the remote caller reads it (COR-5; Python's `__str__`, COR-P8); the wire itself carries the raw message, to which the receiver adds the prefix once (COR-27). Any other error from a model callback is a non-recoverable `INTERNAL` model failure on the wire and propagates unchanged in process (R-12). | Settled |
| COR-23 | `served_env` takes what Python accepts: an env, a vector env, or a list of lane envs, of which only lane 0 is read (`py/_describe.py:409-426`). `ServedEnv` (Rust enum, C++ view) and the C fields `served_env` + `served_env_len` and `served_vector_env` are defined once, in [Env](env.md) (R-27); Core only carries them in `DescribeJsonOptions`. An empty lane list fails with the `ValueError` class (COR-P5). A vector env contributes `single_*` spaces and `num_envs`. Rust `ServedEnv` borrows, C borrows for the call, C++ is a non-owning view. An env already moved into an `EnvServer` is described through COR-16 in C and C++; Rust describes it before the move. | Settled |
| COR-24 | Native `Value` equality (Rust `PartialEq`, C `rlmesh_value_equals`, C++ `operator==`) is kind-strict: values of different kinds are unequal, so `Int(1) != Float(1.0)` and `Bool(true) != Int(1)`, including inside lists, tuples and dicts. This is an intentional difference from Python's numeric `==` (`1 == 1.0 == True`, `{"a": 1} == {"a": 1.0}`): Python has no `Value` class and its `==` is the language's, while native `Value` carries typed kinds whose reads are strict (COR-1). Within a kind: `Float` compares as IEEE `f64` (NaN is unequal to itself, so there is no `Eq`); `List` and `Tuple` compare item by item; `Dict` ignores order (COR-2); `Tensor` uses Rust `Tensor`'s structural `PartialEq` (SPC-18). The E-3 fixtures compare values by kind and content, not with Python's `==`. | Settled |
| COR-25 | Native envelope pieces. Python fills four pieces by reflection (`py/_describe.py:174-178,252-266,437-445`); natives have no module, qualname, signature or exception type name, so each has one native spelling. **`target`**: `{"entrypoint": null, "qualname": …}` (natives have no import string, COR-15). Rust: `"rust:"` + `std::any::type_name::<T>()` of the described type (the policy type, for a `Model` its wrapped policy's, COR-P3; the factory type; for a closure-built `Model` or factory the closure's type, as Python names a bare function), advisory like Python's. C and C++ (C has no type name and C++ wraps C, R-22's reason): `"native:<id>"` for an env target, `<id>` being the served env's `spec.id` or, for a factory, its default-branch env's, `"UnknownEnv-v1"` when that build failed or set none (ENV-6; today's spelling, `capi/describe.rs:188`); `"native:model"` for a model. **`params`**: `{"param_spec": null, "signature_tier": [...]}` (`ParamSpec` stays deferred). A model lists its declared load keys (`LOAD_KEYS`, `load_keys`, MDL-19) in declaration order as `{"name", "type": "str", "default": null, "required"}`: a required key (`LoadKey::required`) is Python's entry for an unannotated keyword without a default (`"required": true`), an optional key (`LoadKey::optional`) Python's entry for an unannotated `key=None` (`"required": false`; `type` `"str"` and `default` `null` either way, `py/params/_resolve.py:171-180,412-424`); no declaration gives `[]`, as Python's `load(**kwargs)`. An env factory lists its `tag_params` in order as `{"name", "type", "default", "required": false}`, `type` named from the default's kind (`"bool"`, `"int"`, `"float"`, `"str"`, else `"str"`) and `default` its JSON form, Python's entry for an unannotated keyword with that default; no `tag_params` gives `[]`, as Python's `make(**kwargs)`. A Python signature can list more keywords and annotated types; natives declare only these. **`runtime`**: the edition handshake keys (`protocol_generation`, `supported_workflow_editions`, `preferred_workflow_edition`, and `workflow_edition_error` when recorded, COR-12), identical in every language, plus the describing host's PeerInfo under the handshake's field names (`crates/rlmesh-proto/src/lib.rs:891-950`): `component` (`"rlmesh-rust"`; `"rlmesh-capi"` for C and C++; Python `"rlmesh-python"`), `language` (`"rust"`, `"c"`), `package_version` (`BuildInfo.version`), `os`, `arch` (`std::env::consts`), and `language_version`, `os_version`, `framework_versions` only when the host supplied them (omitted today in C, `capi/describe.rs:194-203`). C++ describe reports C's runtime; only a served env's envelope carries the C++ host runtime, through ENV-12's `host_runtime_json` (COR-16). PeerInfo is advisory and never gates compatibility. **Badges**: an `env_spec` badge's `error_type` is the R-21 class name (`ErrorClass::name()`, C `rlmesh_error_class_name`), as [Env](env.md) states; Python's `type(exc).__name__` can also name a class outside R-21 (`ImportError`) or an author's subclass, which natives cannot raise. Every badge's `error` (and `variations_error`, `catalog_error`) is the error's R-22 text. | Settled |
| COR-26 | Metadata on the wire. A `Metadata` (or dict `Value`) encoded to the wire (env `info`, reset `options`, `env_contract.metadata`, every wire metadata map) is encoded as Python's metadata codec encodes a dict (`pyrs/spaces/value_codec/metadata.rs:30-104`; `MetaValue`, `crates/rlmesh-spaces/src/meta.rs:4-13`): `None` → Null; `Bool`, `Int`, `Float`, `Bytes` and `List` keep their kind; `Text` → String; `Dict` → Map; `Tuple` → List (`py/_metadata.py:28-31`, "tuples become lists, as in the codec"); `Tensor` → its elements as nested Lists by shape in C order, a 0-d tensor as its bare element, as NumPy's `tolist` (`normalization.rs:30-31`; Python's `rlmesh.Tensor` too after COR-P10), with `bool` elements as Bool, integer elements as Int and floating elements as Float (widened to `f64`). A `uint64` element above `i64::MAX` fails with the ValueError class and Python's text `metadata integer <n> does not fit int64; RLMesh metadata integers must be in [-2**63, 2**63) -- store a larger value as a string instead` (`metadata.rs:69-75`), where Python raises `OverflowError` (R-21's kept exception). Nothing is dropped or stringified; stringifying is `sanitize_metadata`'s opt-in (P-7). Decoding yields only the wire kinds: a Tuple comes back as a List, a Tensor as a List (or a bare element), a Map as a Dict in sorted key order (COR-2). | Settled |
| COR-27 | Wire error text. The `message` of a wire `EnvError` / `ModelError` that a server sends for an author's failure is never R-22's `[CODE] message`; the receiving client stores it as the error's `message` (Rust `EnvironmentError.message` / `ModelError.message`, as `rs/error.rs:199,253` already do; Python's exceptions after COR-P8) and adds the `[CODE] ` prefix exactly once when the error is read (R-22). What each sender puts there: for an Environment- or Model-class error, wherever it is raised, the raw message: Rust `EnvironmentError.message` / `ModelError.message`, never `Display`; C the text passed to `rlmesh_callback_set_error` or `rlmesh_callback_set_error_class`; C++ the raw message the exception keeps for the trampoline (COR-20), never `what()`; Python `BaseException.__str__(exc)` (COR-P11). For any other error a callback returns or raises, which the server sends as an `INTERNAL` failure (R-21), that error's R-22 text: Rust `Display`, C the text recorded with `rlmesh_callback_set_error_class`, C++ `what()` (a `std::exception` outside the R-21 tree included; `unknown C++ exception` for any other thrown type, X-10), Python `str(exc)`. No sender adds a phase, class or type name: an authored `EnvironmentException("bad")` raised in `step` reaches the remote caller as `[INTERNAL] bad`, and an author's `ValueError("x")` (`Error::InvalidArgument`, `rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_VALUE_ERROR, "x")`, `throw rlmesh::ValueError("x")`) as `[INTERNAL] x`, in every language. Texts a server adds for its own failures (a panicked task, a stopped serving thread) are outside this rule. Today Python wraps every env failure as `{phase} failed: <Type>: message` and Rust's model server sends `Display` (COR-P11, Current differences > Rust). | Settled |

### Review notes

- Applied r1-core 1: COR-10 separates class and instance describe.
- Applied r1-core 2: all eight `BuildInfo` fields are mapped with their types,
  `supported_workflow_editions` as a list and `git` as optional (COR-17).
- Applied r1-core 3, 4, 5, 6 through rules: R-21 and D-12 (misuse, type and
  model failures distinct; INVALID_VALUE keeps 2), R-23 and D-13 (warning
  categories, stderr default, warnings as errors), R-25 (plumbing impls and
  route types). Core only maps them.
- Applied r1-core 7: COR-12 strips the resolved edition.
- Applied r1-core defaults: `ModelException` (D-10), env failures raise
  `(Recoverable)EnvironmentException` (D-11), connection, timeout, value and
  type errors stay outside `RLMeshException` (R-21), `#[non_exhaustive]` kept
  on `Error` and `BuildInfo`, separate describe options (COR-11), deprecated
  aliases exempt (D-8), `sanitize_metadata` classified separately (D-9).
- Rejected review (r1-core 3, partial): one Rust variant per Python
  subclass; see rules.md review notes (flat enums cannot make the parent arm
  catch the subclass).
- Adopted r1-core default "native editions: no manifest rung" (user, D-7).
- Rejected review (round 0): generate `RLMESH_VERSION` from the workspace
  version. `rlmesh.h` is a hand-authored installed header, and generating it
  would need a separate `rlmesh_version.h` with its own packaging; a mise check
  catches drift instead (COR-18).
- Consistency pass: COR-2 now keeps insertion order, because SPC-10 (from
  r1-spaces 3) needs Dict values in declaration order and the sorted map
  contradicted it. The declaration row uses ENV-14's associated function on
  both authoring traits, and the `selected_workflow_edition` rows defer to
  Running and Remote (`RLMeshBytes`, RMT-17).
- Answered open questions removed: `ModelException` (D-10), the breaking env
  change (D-11), builtins under `RLMeshException` (R-21: no), merged describe
  options (COR-11). The `kind` question is COR-14.
- Applied r2 (17 of 18): `RLMeshException` row matches the subtree with
  `is_subclass_of` (R-21 builtins excluded); COR-6 makes a serve call's
  `ServeOptions.workflow_edition` the call's value everywhere (`EnvServer`'s
  verbatim adoption, which [Env](env.md) step 6 states, is the rule;
  `Model.serve` aligns, COR-P2), which also settles the separate EnvServer
  exception finding; COR-7 native warning text; `ServedEnv` defined (COR-23,
  COR-P5); COR-16 on `RLMeshEnvServer`; exact Value API and `Metadata` dict
  semantics (COR-21); COR-10 per-branch builds; wrapped-`Model` describe
  (COR-P3); recoverable authored model failures everywhere (COR-22, COR-P4,
  COR-20); COR-3 code getter without a 0 sentinel; COR-P1 describes Python's
  `load` selection; COR-12 rungs; COR-13 failure set; `model error:` prefix;
  `CapiError` carries class and code; constructor typo; Adapters detail cell.
  Rules r2 cross-file applied: `OSError` (`Error::Os`, status 12, 13 classes)
  and COR-5's text list.
- Rejected review (r2, the `EnvServer`-exception finding, as worded): it
  asked to keep `ServeOptions` below `RLMESH_WORKFLOW_EDITION` except for
  `EnvServer`. Two of Python's three texts already treat an explicit
  `ServeOptions.workflow_edition` as the caller's answer (`py/_server.py:283-290`
  comment and code, `py/_models/base.py:1491-1493` docstring), and `ServeOptions`
  is the serve call's own options struct (R-6), so the exception became the
  rule (COR-6) and only `Model.serve`'s code changes (COR-P2).
- Cross-file (r2): env.md — Current differences > C, `rlmesh_env_bind` bullet: `rlmesh_ext_env_describe_json` →
  `rlmesh_ext_env_server_describe_json(const RLMeshEnvServer*, RLMeshBytes* out)`
  (COR-16); construction step 6: "an explicit `options.workflow_edition` is
  the call's value (COR-6 rung 1), adopted ahead of `RLMESH_WORKFLOW_EDITION`"
  (wording only; behaviour unchanged).
- Cross-file (r2): model.md — `Model.serve` row: `ServeOptions.workflow_edition`
  is the call's value, ahead of `RLMESH_WORKFLOW_EDITION` (COR-6, COR-P2); add
  an authored-failure row: Python `raise ModelException(msg, is_recoverable=…)`,
  Rust `Err(Error::model(msg))` / `Error::model_recoverable(msg)`, C
  `rlmesh_callback_set_error(msg, recoverable)`, C++
  `throw rlmesh::ModelException(msg, recoverable)`; anything else is a
  non-recoverable `INTERNAL` model failure (COR-22); `Model.describe()` /
  instance describe reports the wrapped policy's declarations (COR-P3).
- Cross-file (r2): rules.md — R-21 "Authored envs and models": add Python
  `ModelException(message, *, is_recoverable=False)` and C++
  `ModelException(message, is_recoverable)` beside `model_recoverable`
  (COR-22); D-7 / P-6 rung 1 includes `ServeOptions.workflow_edition` on
  `EnvServer` and `Model.serve` (COR-6).
- Applied r3 (11 of 11, one partly): `served_env` row, COR-23, COR-13 and
  COR-16 defer to [Env](env.md)'s single `ServedEnv` definition (R-27), which
  keeps the lanes form, so the C fields (`served_env` + `served_env_len`,
  `served_vector_env`) and COR-P5's empty-lane `ValueError` match Env's
  table; COR-P6 aligns a wrapping Python `Model(policy)`'s run, session and
  serve edition with the policy type, and COR-7 states today's rule; COR-10
  adds the C model type form `rlmesh_model_vtable_describe` /
  `_describe_json` (static fields, no slot called), C++ `describe<P>()` wraps
  it without constructing, and the env-factory type form constructs; COR-6
  drops `DescribeJsonOptions` from rung 1 and states describe's bypass; C
  value migration list; Rust variant list with `Os` and the new
  `InvalidArgument`; COR-24 kind-strict `Value` equality;
  `Value(std::nullptr_t) = delete`; C++ `ValueKind`, C
  `rlmesh_value_dict_contains_key`; exact `describe_json` signatures; COR-P1
  qualified for non-callable instances.
- Resolved (r3 finding "dict contains return convention", restated r6):
  both C contains calls now return `RLMeshStatus` + `bool* out`. Core's
  `rlmesh_value_dict_contains_key` fails with `RLMESH_ERR_TYPE` on a
  non-DICT value (a wrong-kind read, Value API intro), as Rust's
  `v.as_dict()?.contains_key(k)` does, and Spaces'
  `rlmesh_dict_contains_key` fails with `RLMESH_ERR_TYPE` /
  `RLMESH_ERR_INVALID_ARGUMENT` the same way (spaces.md `key in d` row), so
  the r3 finding is applied, not partly rejected.
- Cross-file (r3): model.md — `Model.describe()` row and MDL-13: C is
  `rlmesh_model_vtable_describe(&vt, NULL, &out)` (Core's model type form,
  COR-10: the vtable's static fields, no slot called, no `user_data`), not
  the handle form, so "C has no type form" and the `ModelOptions.spec`
  difference go; C++ `Model::describe<P>()` wraps it over the static
  `RLMeshModelVtable` generated for `P` and constructs nothing (no `load`).
- Cross-file (r3): model.md — the `workflow_edition` declaration row (or its
  Python changes): Python's wrapping `Model(policy)` reads
  `type(policy).workflow_edition` and is authored unless it wraps a bare
  callable (COR-P6).
- Cross-file (r3): env.md — ENV-19 / the `describe()` classmethod row: C++
  `EnvFactory<F>::describe()` default-constructs one `F` and runs C's
  instance form on it, so its branches share that factory, unlike Python's
  and Rust's fresh factory per build (COR-10).
- Applied r4 (7 of 7): COR-10's instance form takes only `spec` from the
  instance, `native_chunk` always from the type, and COR-P7 makes Python's
  describe read it from the type (model.md's r3 cross-file note); COR-25
  fixes the native envelope `target`, `params`, `runtime` and badge
  `error_type` (with a Describe row); an authored Environment or Model error
  reads `[INTERNAL] message` in every language, in process too (COR-5,
  COR-20, COR-22, the note under the Errors table), and COR-P8 gives
  Python's `EnvironmentException` / `ModelException` that `__str__`; the
  Errors note names `rlmesh_callback_set_error_class` for every other class
  (rules' r3 cross-file note) and the C++ payload rethrow; COR-7 names
  Running's env relay (RUN-25; running.md's r2 cross-file note); the C++
  integral constructor deletes only `char`, `wchar_t` and `charN_t`, so
  `int8_t` / `uint8_t` build `Int`; the C Shape uses the model type form.
  Rules' r4 cross-file notes applied: C warning category name and
  `is_subclass` in the `WorkflowEditionWarning` row, COR-5 lists R-22's
  native load texts.
- Cross-file (r4): rules.md — R-22 table, Environment and Model rows: the
  `[CODE] message` text holds for an authored error in process too
  (`Error::environment("bad")` reads `[INTERNAL] bad`), with Python's
  `__str__` change (COR-P8) listed beside P-1; R-21 "Authored envs and
  models": the authored constructors' text is `[INTERNAL] message` (COR-5).
- Cross-file (r4): model.md — MDL-13 / `Model.describe()` row: the envelope's
  `params` lists `LOAD_KEYS` / `load_keys` and `target.qualname` is
  `rust:<type_name::<P>()>` or `native:model` (COR-25); Python's describe
  reads `native_chunk` from the type, never a K `load()` set (COR-P7).
- Cross-file (r4): env.md — `EnvFactory.describe` paragraph: cite COR-25 for
  `target` (`rust:<type_name::<F>()>`, `native:<id>`), `params` (from
  `tag_params`) and `runtime`; `error_type` as already stated.
- Cross-file (r4, optional): rules.md — R-25: C++ `describe_json<T>()`
  reports C's runtime (`language "c"`) because no shared describe options
  struct may carry a fenced `host_runtime_json`; if C++ describe should
  report `"c++"` as its served envelope does (ENV-12), R-25 needs an
  exception for the describe options or a process-wide fenced host-runtime
  setter.
- Applied r5 (8 of 8): COR-26 defines Metadata on the wire (Tuple → List,
  Tensor → its `tolist` form, `uint64` overflow the ValueError class; kinds
  come back as the wire's), with COR-P10 so Python encodes an
  `rlmesh.Tensor` as it encodes a NumPy array (today `TypeError`); the
  Metadata row derives `Clone, Debug, Default`; an `RLMeshStr` output past
  the end is `{NULL, 0}` with `RLMESH_OK`, and a present one always has
  non-NULL `data`; COR-13 lists the model type form's declaration refusals
  (C's `rlmesh_model_new` statuses, Rust's `NATIVE_CHUNK` checks), with
  COR-P9 so Python's describe refuses the same declarations; COR-25 spells
  required and optional load keys separately; COR-4 cites only `model_rpc`
  (`rs/model/local.rs:149`) and names the other crates' predicates as
  unaffected; the Python changes intro lists P-8 and P-9; `rlmesh.hpp`
  push/pops Xlib's `None` and `Bool` macros.
- Cross-file (r5): spaces.md — SPC-28: the facade `Tensor` derives or
  implements `Debug` (delegating to the core `Tensor`'s `Debug`), so
  `Value`'s `#[derive(Clone, PartialEq, Debug)]` compiles.
- Cross-file (r5): spaces.md — SPC-13 and the `rlmesh_dict_key_at` /
  `rlmesh_dict_value_at` row: an index past the end returns `RLMESH_OK`
  with `{NULL, 0}` / `NULL`, and a present `RLMeshStr` output always has
  non-NULL `data` (the empty key included), as Core's Value API now states.
- Cross-file (r5): rules.md — R-4: decide whether C++ enumerators that
  collide with common platform object-like macros (Xlib `None`, `Bool`,
  `True`, `False`, `Status`, `Success`; also `DType::Bool`) take a trailing
  `_` as reserved words do, or adopt Core's `push_macro` / `pop_macro`
  rule for every enumerator `rlmesh.hpp` declares.
- Cross-file (r5): rules.md — R-21's `OverflowError` note: a native
  `uint64` Tensor element above `i64::MAX` in wire metadata (COR-26) can be
  written natively and is the ValueError class there; Python keeps
  `OverflowError`.
- Cross-file (r5): model.md — MDL-13 and the `Model.describe()` row: Rust
  `Model::describe::<P>()` fails with `Error::Type` for `NATIVE_CHUNK =
  Some(K)` without a chunk corner and `Error::InvalidArgument` for
  `Some(0)`, as C's type form refuses the vtable (COR-13); Python's
  describe refuses the same declarations (COR-P9).
- Cross-file (r5, optional): env.md — the `info` and reset `options` rows
  (and `env_contract.metadata`) cite COR-26 for how native `Value` kinds
  cross the wire.
- Applied r6 (6 of 6): COR-27 defines the wire error text (an Environment-
  or Model-class error sends its raw message, any other error its R-22
  text, no phase or type-name prefix; the receiver adds `[CODE] ` once),
  referenced from COR-5, COR-20 (the C++ exception keeps the raw message
  privately for the trampoline), COR-22, COR-P4 and the Errors note, with
  COR-P11 for Python's env server and `PyPredict` and a Rust current
  difference for `rs/model/wire.rs:36-46`; COR-10 names the vtable's
  `load_keys` + `load_keys_len`; COR-13's Rust refusal comes from
  `describe` / `describe_json`, not the infallible `Describable::model`;
  the C and C++ code enums are spelled out with wire numbers and COR-3
  exempts them from R-9's `INVALID = 0`; the r3 dict-contains note is
  resolved; C citations corrected (`h:61-72`, `capi/abi/status.rs:54-71`).
- Cross-file (r6): rules.md — E-2 rule 6: exempt `RLMeshErrorCode` and
  `RLMeshModelErrorCode` (wire mirrors whose 0 is `UNSPECIFIED`, COR-3)
  beside `RLMeshStatus`; R-9's output-only-mirror sentence: such mirrors
  keep wire numbers even where a `_str` function takes them as input.
- Cross-file (r6): rules.md — R-12 / X-10: the C++ trampoline passes an
  `EnvironmentException` / `ModelException`'s private raw message (not
  `what()`) to `rlmesh_callback_set_error` (COR-20, COR-27); R-22: the wire
  `EnvError` / `ModelError` `message` is the raw message and the receiver
  adds `[CODE] ` once (COR-27); Python changes: list COR-P11 beside P-1 /
  COR-P8 (the remote text loses `{phase} failed: <Type>: `).
- Cross-file (r6, optional): env.md and model.md — the server / serve rows
  cite COR-27 for the wire text of an author's failure.
