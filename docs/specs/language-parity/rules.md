# Language parity: rules

Part of the [language parity spec](../language-parity.md). Status: Draft —
reviewed round 6.

These rules hold in every area file. An area that disagrees with a rule is
wrong; fix the area, or change the rule here. IDs are stable: `R-n` rules,
`D-n` decisions, `P-n` Python changes, `X-n` cross-cutting changes, `G-n`
gaps, `E-n` enforcement; area files use `<AREA>-n` decisions and
`<AREA>-P<n>` Python changes (`COR`, `SPC`, `ENV`, `MDL`, `RUN`, `RMT`, `ADP`).
Never renumber; retire an ID by marking it removed.

Paths: `py/` = `python/rlmesh/src/rlmesh/`, `pyrs/` =
`python/rlmesh/rust/src/`, `rs/` = `crates/rlmesh/src/`, `capi/` =
`crates/rlmesh-capi/src/`, `h` = `crates/rlmesh-capi/include/rlmesh.h`, `hpp`
= `crates/rlmesh-capi/include/rlmesh.hpp`.

## Rules

### R-1 One library

Every Stable Python symbol (`python/rlmesh/tests/api_surface/snapshots/api_surface.json`,
labels in [compatibility](../../compatibility.md)) has a counterpart in Rust,
C and C++ with the same name (cased by R-4), the same parameters and fields
under the same names, the same defaults and the same behaviour. A difference
that no rule explains is a bug.

### R-2 Python is the reference

Python's Stable surface defines the nouns, verbs, defaults and validation. The
other languages do not grow their own model. When the target changes Python's
behaviour, the change is listed explicitly (P-n below, or the area's "Python
changes" list); it is never implied by a table cell.

### R-3 Translation table

These differences are expected. Each applies the same way to every symbol.

| Concern                        | Python                            | Rust                                                                                                   | C++                                                        | C                                                   |
| ------------------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------- |
| Names                          | `snake_case`, `PascalCase`        | same (R-4)                                                                                             | same (R-4)                                                 | D-6                                                 |
| Modules                        | `rlmesh.spaces`                   | `rlmesh::spaces`                                                                                       | `rlmesh::spaces`                                           | D-6                                                 |
| Errors                         | exceptions                        | `Result<T, Error>`                                                                                     | exceptions                                                 | `RLMeshStatus` + thread-local last error            |
| Error classes                  | exception classes                 | `Error` variant + `ErrorClass` (R-21)                                                                  | classes with Python's inheritance (R-21)                   | status + `RLMeshErrorClass` (R-21)                  |
| Warnings                       | `warnings.warn(Category)`         | handler (R-23)                                                                                         | handler (R-23)                                             | handler (R-23)                                      |
| Lifetime                       | context manager, `close()`        | `Drop`, `close()`                                                                                      | RAII, `close()`                                            | `rlmesh_<t>_close`, `rlmesh_<t>_free`               |
| Keyword arguments              | keywords with defaults            | options struct, `Default` (R-6)                                                                        | aggregate, `with_<field>` setters; C++20 designators (R-6) | options struct, `RLMESH_<NAME>_INIT` (R-6, R-19)    |
| Optional keywords, short calls | keywords                          | positional `Option` (R-7, closed list)                                                                 | default arguments (R-7)                                    | pointer, NULL = default (R-7)                       |
| `*args` / `**kwargs`           | variadic                          | slice or `Vec` / `Kwargs` field                                                                        | `std::vector` / `Kwargs` field                             | pointer + count / `RLMeshValue*` dict               |
| Optional value                 | `None`                            | `Option<T>`                                                                                            | `std::optional<T>`                                         | pointer (NULL = None) or `has_<name>` (R-13)        |
| Optional list                  | `None` or list                    | `Option<Vec<T>>`                                                                                       | `rlmesh::OptionalList<T>`                                  | pointer + count; NULL = None, count 0 = `[]` (R-13) |
| Durations                      | `*_seconds: float`                | `Duration`, `_seconds` dropped                                                                         | `std::chrono::duration<double>`, `_seconds` dropped        | `double *_seconds`                                  |
| Integers                       | `int`                             | `i64`; `u64` where Python refuses negatives (R-5)                                                      | `int64_t` / `uint64_t`                                     | `int64_t` / `uint64_t`                              |
| Tuples returned                | tuple                             | tuple, or a struct named for the fields                                                                | `std::tuple` (structured bindings)                         | out-parameters in tuple order                       |
| Closed string vocabularies     | `str` from a fixed set            | enum, `as_str()` / `Display` = Python spelling (R-9)                                                   | `enum class`, PascalCase, `to_string()` (R-9)              | enum, `INVALID = 0`, `rlmesh_<enum>_str()` (R-9)    |
| Dunder methods                 | `__repr__`, `__eq__`, …           | traits (R-10)                                                                                          | operators (R-10)                                           | named functions (R-10)                              |
| Handles                        | object reference                  | move-only, or `Clone` shares (R-11)                                                                    | move-only, or copy shares (R-11)                           | `rlmesh_<t>_clone` returns a new reference (R-11)   |
| Frozen dataclasses             | `@dataclass(frozen=True)`         | value struct, `Clone` copies (R-11)                                                                    | value struct (R-11)                                        | value handle or struct (R-11)                       |
| Protocols, subclass hooks      | `Protocol`, overridable methods   | trait (R-12)                                                                                           | class detected by a templated adapter (R-12)               | vtable, NULL slot = not defined (R-12)              |
| Callables                      | callable                          | `impl FnMut(..) + Send + 'static`; `Fn + Send + Sync` where called concurrently or process-wide (R-23) | any callable                                               | function pointer + `user_data` + `free_user_data`   |
| Blocking and async             | sync                              | sync; `*_async` twins opt in (R-16)                                                                    | sync                                                       | sync                                                |
| Cancellation                   | `KeyboardInterrupt` (signal poll) | `ext::CancelToken`, future drop (R-18)                                                                 | `ext::CancelToken` (R-18)                                  | `RLMeshExtCancelToken` (R-18)                       |
| Language-only extras           | —                                 | `rlmesh::ext`, `rlmesh::binding` (R-25)                                                                | `rlmesh::ext` (R-25)                                       | `rlmesh_ext_` / `RLMeshExt` (R-25)                  |

### R-4 Names

- Rust and C++ keep Python's spelling: types `PascalCase`, functions, fields
  and parameters `snake_case`, constants `UPPER_SNAKE`. C++ enumerators are
  `PascalCase` (R-9). C naming is D-6.
- Options struct fields and parameters carry the Python keyword names. The
  only rename is the `_seconds` suffix, dropped in Rust and C++ where the type
  is a duration (`max_episode_seconds` → `max_episode: Option<Duration>`) and
  kept in C.
- A name that is a keyword or reserved word in Rust (strict, reserved or
  weak), C++ or C gets a trailing `_` in Rust, C++ and C fields alike:
  `export_`, `default_`, `type_` in every native language, even where only
  one of them reserves the word. Rust does not use `r#` raw identifiers. One
  spelling keeps the E-2 name check a single mapping per name.

### R-5 Scalars and text

- `int` → `i64` (C/C++ `int64_t`); `u64` where Python refuses negatives (seeds
  of spaces, counts). `usize` only for in-memory lengths and positions. An
  area that picks a narrower type cites the Python check that justifies it.
- `float` → `f64` / `double`; `bool` → `bool`.
- `str` → `String` / `&str` (C++ `std::string`, `std::string_view` for
  constants, except constants that initialize `std::string` fields, which
  are `inline constexpr const char*`: Adapters' roles, parts and keys,
  ADP-22). C text inputs are NUL-terminated UTF-8, except text data that
  Python lets contain NUL (text values, dict keys), which uses the
  length-bearing view Spaces defines. C text outputs are `RLMeshBytes`
  (owned, variable-length), with two exceptions that return `const char*`:
  the static spellings of closed vocabularies (`rlmesh_<enum>_str`,
  `rlmesh_error_class_name`, `rlmesh_warning_category_name`; static
  storage, NULL for a value with no spelling: `INVALID`, an `_OTHER`, a
  value outside the enum; R-9, SPC-20) and the
  last-error getters `rlmesh_last_error_message` and
  `rlmesh_last_error_code_str` (thread-local, valid until the next rlmesh
  call on that thread, NULL if none; R-22, Core COR-3).
- `bytes` → `Vec<u8>` / `&[u8]`, C pointer + length in, `RLMeshBytes` out.

### R-6 Keyword arguments become one options struct

- Required parameters stay positional, in Python's order. Every parameter with
  a default goes into one options struct, passed last, except the calls in
  R-7. The area table names each struct.
- Rust: a plain struct with public fields whose `Default` is Python's
  defaults. Options structs are **not** `#[non_exhaustive]`, which would forbid
  struct literals outside the crate. Callers must write
  `..Default::default()`. Cargo's SemVer reference counts a new public field
  on an all-public struct as major; by project policy, stated in the crate
  docs and in each options struct's doc comment, appending an options field
  is a minor change, and a full literal without `..` is unsupported. Pass
  `Default::default()` for "no keywords" (Rust has no default arguments).
- C++: an aggregate with default member initializers equal to Python's
  defaults, built portably with assignment or chainable `with_<field>` setters
  (lvalue and rvalue overloads); C++20 designated initializers remain a
  convenience. The struct stays an aggregate and the parameter defaults to `{}`.
- C: a struct whose first field is `size_t struct_size`, with an
  `RLMESH_<NAME>_INIT` macro holding Python's defaults. A NULL options pointer
  means the `_INIT` value. Sizing rules are R-19.
- `**kwargs` (`EnvFactory.serve(**make_kwargs)`, `Variant(**metadata)`)
  becomes a `Kwargs` field; `*args` (`Action(*components)`,
  `Session.reader(*items)`) becomes a positional slice.
- Options structs are plain data. Validation runs in the call that consumes
  the struct, with Python's checks and Python's error class.
- Exception: authored exception constructors take no options struct.
  `ModelException(message, *, is_recoverable=False)` is C++
  `explicit ModelException(std::string message, bool is_recoverable = false)`,
  Rust `Error::model(message)` / `Error::model_recoverable(message)` and C
  `rlmesh_callback_set_error(message, recoverable)` (R-21, COR-20, COR-22).
  The other authored constructors take only a message.

### R-7 Positional convenience calls (closed list)

Only these calls take Python's optional keywords as trailing positional
parameters, in Python's order. Everything else uses R-6. Adding a call here
is a rule change.

| #     | Python                                                                                               | Positional parameters |
| ----- | ---------------------------------------------------------------------------------------------------- | --------------------- |
| R-7.1 | `Session.reset(*, seed=None, trial_index=None)`                                                      | `seed, trial_index`   |
| R-7.2 | `RemoteEnv.reset`, `RemoteVectorEnv.reset`, `SandboxEnv.reset`, `SandboxVectorEnv.reset`             | `seed, options`       |
| R-7.3 | `RemoteEnv.render`, `RemoteVectorEnv.render`, `SandboxEnv.render`, `SandboxVectorEnv.render`         | `env_index`           |
| R-7.4 | `RemoteEnv.shutdown`, `RemoteVectorEnv.shutdown`, `SandboxEnv.shutdown`, `SandboxVectorEnv.shutdown` | `reason`              |
| R-7.5 | `EnvServer.wait(timeout=None)`                                                                       | `timeout`             |
| R-7.6 | `Space.seed(seed=None)` and every space type                                                         | `seed`                |
| R-7.7 | `trial_index(options, *, lane=None)`                                                                 | `lane`                |

The sandbox clients in R-7.2-R-7.4 are Experimental and deferred (Remote);
their rows fix the call shape that applies when sandbox parity lands, and
bind nothing before then.

- Rust: each listed parameter is `Option<T>`; `None` means Python's default,
  whatever it is (`render(None)` is `env_index=0`, `shutdown(None)` is
  `"owner shutdown"`).
- C: a pointer for text and for parameters whose Python type is optional (NULL
  = Python's default); a plain value for a non-optional scalar (`env_index`),
  where the caller passes Python's default.
- C++: default arguments equal to Python's defaults (`s.reset(0)`).
- Authored protocol methods (`Env.reset(seed, options)`,
  `Model.on_episode_end(episode_id)`) are not convenience calls: R-12 gives
  them Python's parameters positionally.

### R-8 Rust builders

Builders are allowed only as optional setters on an options struct:

- one setter per field, named exactly as the field, taking the field's inner
  type without the `Option`, returning `Self`. Scalar fields (integers,
  floats, `bool`, enums) take `T` exactly, so an integer literal infers the
  field's type (`impl Into<u64>` would leave `2` to fall back to `i32`, which
  has no `Into<u64>`: E0277). Fields whose inner type is a trait-object
  reference or pointer (`&mut dyn _`, `Arc<dyn _>`, `Box<dyn _>`, such as
  `RunOptions.hooks` and `RecorderCaptureOptions.session`) take that exact
  type, so unsizing coercion applies at the call site (`.hooks(&mut
progress)`; `impl Into<&mut dyn RunHooks>` has no `From` impl to use:
  E0277). Other fields (`String`, `Vec<_>`, `Duration`, `SpecArg`, …) take
  `impl Into<T>`;
- no `build()` step, no typestate, never on a handle (`Model`, `Session`,
  clients), never the only way to set a field.

`RunOptions::default().episodes(2).seeds([0, 1])` equals the struct literal.
Fallible `with_<field>` constructors exist only on value structs whose Python
form validates an `InitVar` (R-11). No other builder shapes.

### R-9 Closed vocabularies

A Python `str` attribute or parameter drawn from a fixed set (dtype, space
kind, `bounds_kind`, device, `EnvironmentException.code`, adapter
vocabularies, warning and error categories) becomes an enum:

- text form is Python's spelling: Rust `as_str()` and `Display`, C
  `rlmesh_<enum>_str()`, C++ `to_string()`. Exception: the two class
  vocabularies, `ErrorClass` (R-21) and `WarningCategory` (R-23), spell
  Python's `__name__` and use `name()` in Rust and C++ and
  `rlmesh_error_class_name` / `rlmesh_warning_category_name` in C, with no
  `as_str()` / `_str` of their own;
- numbers are the wire values where a wire enum exists;
- C input enums reserve `*_INVALID = 0`, refused by every call; real values
  start at 1. Output-only mirrors of wire enums keep the wire numbers.
  Exception: `RLMeshStatus`, which callbacks return (R-20), keeps
  `RLMESH_OK = 0` (`h:61-72`);
- a field that Python stores tolerantly (any string constructs and
  round-trips) holds `Tolerant<E>` over the closed enum: Rust and C++
  `Tolerant<E>` (`Known(E)` / `Other(String)`), C the enum's
  `RLMESH_<NAME>_OTHER` value plus a `<name>_other` text field. The
  canonical list of such fields and the type are Adapters' (ADP-4). A field whose text is not drawn from a fixed set
  stays a string (RUN-11).

### R-10 Dunder methods

Map only where Python defines the method:

| Python                   | Rust                                             | C++                         | C                                 |
| ------------------------ | ------------------------------------------------ | --------------------------- | --------------------------------- |
| `__repr__`, `__str__`    | `Display` (Python's `str`), `Debug`              | `operator<<`, `to_string()` | `rlmesh_<t>_repr` → `RLMeshBytes` |
| `__eq__`                 | `PartialEq`                                      | `operator==`                | `rlmesh_<t>_equals`               |
| `__hash__`               | `Eq + Hash`, only if the type has no float field | `std::hash`, same condition | —                                 |
| `__len__`, `__getitem__` | `len()`, `get(i)`                                | `size()`, `operator[]`      | `rlmesh_<t>_len`, `_get`          |
| `__iter__`               | `iter()`                                         | `begin()`/`end()`           | `_len` + `_get`                   |
| `__contains__`           | `contains()`                                     | `contains()`                | `rlmesh_<t>_contains`             |
| `__call__`               | `call()`                                         | `operator()`                | `rlmesh_<t>_call`                 |
| `__dlpack__`             | `to_dlpack()`                                    | `to_dlpack()`               | `rlmesh_<t>_to_dlpack`            |
| `__enter__` / `__exit__` | R-3 lifetime row                                 | RAII                        | `_close` / `_free`                |

C forms: `RLMeshStatus rlmesh_<t>_repr(const T*, RLMeshBytes* out)`.
`_equals` on a handle is `bool rlmesh_<t>_equals(const T* a, const T* b)`
(NULL → false). `_equals` over a sized struct (R-19), caller-built or a
library out-struct the caller holds, takes the status form
`RLMeshStatus rlmesh_<t>_equals(const T* a, const T* b, bool* out)`, so it
refuses a `struct_size` below the minimum, an `INVALID` enum and NULL as
every other call does (ADP-29).

Exceptions to the table:

- **`str` differs from `repr`.** `Display`, `operator<<` and `to_string()`
  print Python's `str`; the `repr` is a hand-written Rust `Debug`, a C++
  `std::string repr() const` and C `rlmesh_<t>_repr` (Adapters' `Advisory`,
  ADP-28). Where `str` is `repr` (the dataclasses), `Display` prints it and
  `Debug` is not compared (derived unless the area defines it).
- **Input leaves.** C input leaf structs reached only through a validated
  handle (Adapters' tag leaves: `RLMeshImage`, `RLMeshState`,
  `RLMeshActuator`, …) have no C `_equals` or `_repr`, although Python's
  frozen dataclasses define `__eq__` and `__repr__`; C compares and prints
  them at the handle (`rlmesh_model_spec_equals` / `_repr`,
  `rlmesh_env_tags_equals` / `_repr`; ADP-6). Rust and C++ keep the table.

Keyed mappings: on a type whose `__getitem__` takes a key and whose class
also has a value-membership `contains` (Spaces `Dict`, SPC-26), `__contains__`
is `contains_key()` / `rlmesh_<t>_contains_key`, `__getitem__` is `get(key)` /
`operator[](key)` / `rlmesh_<t>_get(RLMeshStr key)`, and positional C access
is `rlmesh_<t>_key_at` / `rlmesh_<t>_value_at`.

### R-11 Handles and value structs

- **Handles** (`Model`, `Session`, clients, servers) are objects with
  identity. A handle is copyable only if Python lets two references drive it
  at once; then a copy shares the object (Rust `Clone`, C++ copy, C
  `rlmesh_<t>_clone` returning a new reference, freed independently). A handle
  that needs exclusive use (`Session`, `RemoteEnv`, `RemoteVectorEnv`) is
  move-only.
- **Value structs** come from frozen dataclasses (`EpisodeResult`,
  `ObservationRoles`, the adapter tag leaves such as `Image` and `StateTag`).
  Fields are public and named as in
  Python; a copy is independent. Leaf checks run where Python's composite
  validates (`ModelSpec::new`, `EnvTags::new`) with Python's error class;
  `InitVar` sugar becomes a fallible `with_<name>`.
- **Immutable values** (`Value`, `Tensor`, and the validated composites
  `ModelSpec` / `EnvTags`, read through accessors, ADP-21) may share storage
  on copy, because sharing is unobservable. C++ `Value` is passed as
  `const Value&`.

### R-12 Protocols and authored types

- A Python protocol or overridable class hook becomes a Rust trait (Python's
  optional hooks are default methods), a C vtable (a NULL slot means "not
  defined") and a C++ class detected at compile time through a templated
  adapter; an abstract base class is used only where nothing static is read.
- The ordinary trait impl is the complete Rust path. A proc macro may only be
  sugar that writes what a hand-written impl can write; no capability needs
  it.
- Protocol methods take Python's parameters positionally in Python's order,
  optional ones as `Option` / NULL / `std::optional`.
- An authored callback's error propagates unchanged within one process, as
  Python re-raises the author's own exception (`pyrs/model.rs:76-78`,
  `pyrs/hooks.rs:21-24`): Rust returns the author's `Error` as is; C returns
  the class the callback set; the C++ wrapper rethrows the author's own
  exception object through the error payload channel below. Across the wire
  the peer sees R-21's Environment or Model class.
  Exception: Adapters' `check_inverse` self-check wraps an arm's error in
  `AdapterResolutionError` carrying the arm's text (ADP-27); per-step calls
  of the same arms propagate unchanged.
- **Error payload channel** (fenced, R-25; the C++ wrapper's transport).
  C errors travel by value from the callback's thread to the outermost export
  and are written to the caller's thread-local slot only there
  (`capi/abi/status.rs:1-5`), so a callback-thread or trampoline-object stash
  cannot carry an exception: the callback may run on a library thread, and
  one `Model` may serve sessions on several threads. The payload travels
  inside the error instead:
  - `void rlmesh_ext_callback_set_error_payload(void* payload, void
(*free_payload)(void*))`, called inside a callback, attaches an opaque
    payload to the error that callback reports (by `rlmesh_callback_set_error`
    or `_class`, before or after, or the default of a bare nonzero return). A
    second call in the same callback frees the first payload. A callback that
    returns `RLMESH_OK` has its payload freed.
  - The payload moves with the error by value to the export that returns
    the failure, which stores it in the caller's thread-local slot beside
    the message and class. Between the callback and the export the error is
    a facade `rlmesh::Error` (the capi `Policy` / `Env` adapter returns
    `rlmesh::Result`, `capi/model.rs:295-307`; it crosses the engine,
    `Session` and run and becomes a `CapiError` only at the export,
    `capi/model.rs:738`), so the payload rides on `rlmesh::Error` itself.
    An enum cannot hold a private field, so the attachment is a hidden
    variant, allowed by `#[non_exhaustive]` and reachable only through
    `binding` (R-25): `#[doc(hidden)] Error::Attached(binding::AttachedError)`,
    where `AttachedError { error: Box<Error>, payload:
Arc<binding::ErrorPayload> }` has private fields and is never nested
    (attaching to an attached error replaces its payload). `ErrorPayload:
Send + Sync` owns the pointer and `free_payload` in a
    `Mutex<Option<_>>` and calls `free_payload` on its last drop unless the
    pointer was taken. Only the capi creates it
    (`binding::attach_payload(Error, ErrorPayload) -> Error`) and removes it
    (`binding::take_payload(Error) -> (Error, Option<ErrorPayload>)`), so no
    shared-surface Rust call returns an attached error.
    - `class()`, `Display`, `is_recoverable()` and the detail accessors see
      through the attachment; facade code that matches variants matches
      `Error::unattached(&self) -> &Error` (`pub(crate)`).
    - `Clone` shares the `Arc`: the first take moves the pointer out, a
      later take of any clone gets none. `PartialEq` / `Eq` ignore the
      attachment, so `Error` replaces its derived `PartialEq` and `Eq`
      (`rs/error.rs:72`) with manual impls that compare `unattached()`;
      `Clone` and `Debug` stay derived (`AttachedError`'s manual `Debug`
      prints the inner error and `payload: ..`).
    - Facade code that rewraps or folds an error drops the attachment: the
      wire send (class, code and text only), ADP-27's `check_inverse`
      wrapping, `join_results`' fold of two failures (`rs/error.rs:127-136`;
      when only one side fails it passes unchanged, attachment included),
      and errors swallowed by Session Drop or RUN-19.
    - `CapiError::from(rlmesh::Error)` calls `take_payload` and moves the
      payload into `CapiError` (which gains the payload field), and the
      export stores it in the slot.
  - `void* rlmesh_ext_last_error_take_payload(void)`, on the caller's thread
    after a failing call, returns the payload of the last error and moves
    ownership to the taker (a second take returns NULL). An untaken payload
    is freed with `free_payload` when the slot is overwritten, cleared or its
    thread exits. A payload whose error does not reach a caller unchanged
    (sent across the wire, wrapped by ADP-27, swallowed by Session Drop or
    RUN-19) is freed by the library; `free_payload` may run on any thread.
  - C++: the `noexcept` trampoline catches, records class and text with R-12's
    calls (X-10), and attaches a heap `std::exception_ptr` with a deleter as
    the payload. After a failing C call the wrapper takes the payload and
    rethrows it with `std::rethrow_exception`; with none, it throws the C++
    class of `rlmesh_last_error_class()` (R-21), carrying the message, code
    and recoverable flag, since one status covers two classes (1 and 3). The
    same channel carries a warning handler's exception (R-23).
- A C callback sets its error with one of two calls before returning
  nonzero:
  - `void rlmesh_callback_set_error(const char* message, bool recoverable)`.
    Its class is fixed by the callback kind: `EnvironmentException` in an env
    callback (`RecoverableEnvironmentException` when `recoverable`),
    `ModelException` in a model callback (with `is_recoverable` =
    `recoverable`), and `RLMeshException` in every other callback (run hooks,
    `RunContext`, `AdapterBase`, transforms, variant sinks, the warning
    handler), where `recoverable` is ignored.
  - `void rlmesh_callback_set_error_class(RLMeshErrorClass error_class, const
char* message)`, which records any R-21 class, as a Python author may
    raise `ValueError` or `TypeError` from a hook. An Environment or Model
    class carries code `INTERNAL` and is not recoverable unless the class is
    `RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION`; a recoverable
    model failure uses `rlmesh_callback_set_error`.
    `RLMESH_ERROR_CLASS_INVALID` (0, R-9) or an unknown value records
    `RLMeshException`, because the callback has no status to receive a
    refusal.

  A callback that returns nonzero without setting an error fails with its
  kind's class and the text `callback failed with status <n>`. The last
  call before the return wins.

  The pending error and payload are per callback frame. A callback may make
  sync rlmesh calls on its thread (R-16), which may run further callbacks
  there, and the library clears the pending state before each callback
  (`capi/abi/status.rs:83-88`). So the library saves the thread's pending
  error and payload before invoking a callback and restores them after it
  returns, and a nested call cannot clobber an outer callback's pending
  error. The nested export's own failure, written to the thread-local slot
  as usual, does not count as the outer callback's error unless the
  callback records it again.

### R-13 Optional values

- No sentinels. C uses a pointer (NULL = None) or a `has_<name>` field; 0 is a
  value, refused where Python refuses it.
- An optional list in C is a pointer plus a `<name>_len` count: NULL = None,
  non-NULL with count 0 = `[]`, NULL with a nonzero count is refused. C++ uses
  `rlmesh::OptionalList<T>`, which accepts braced lists (`.seeds = {0, 1}`).

### R-14 Durations and limits

Each parameter keeps Python's own validation: `ServeOptions` timeouts refuse
`<= 0` (`pyrs/lifecycle.rs:231-239`), `EnvServer.wait(0)` is allowed
(`pyrs/server/py_env_server.rs:478-485`), client timeouts follow Remote. A native
language never reads 0 as "unset".

### R-15 Tuples and Gymnasium shapes

- A returned tuple stays a tuple in Rust and C++ (`let (obs, info) = ..`,
  `auto [obs, info] = ..`), or a struct named for its fields where Python
  names them. C returns out-parameters in tuple order.
- `reset` returns `(observation, info)` and `step` returns
  `(observation, reward, terminated, truncated, info)` on `Env`, `Session`,
  `RemoteEnv` and `RemoteVectorEnv` alike.

### R-16 Sync first, async twins

- Every language's primary API blocks, as Python does (D-1).
- Rust adds `<name>_async` for each blocking call that waits on I/O or the
  model. The twin has the same parameters, types, borrows and ownership as the
  sync call, and the same return type wrapped in a future. Both accept the
  same local-env representation (Running's `EnvTarget<'a>`); no twin takes an
  owned or `'static` input where the sync call borrows.
- **One driver, two ways to wait.** The sync call and its twin run the same
  driver future. The sync call drives it on the caller's thread in segments
  (R-17): a segment polls the driver until it yields the next event (a hook
  dispatch, a relayed env op, a queued warning) or completes. Between
  segments the caller handles that event outside any async execution
  context (the condition below), with the driving flag clear; the library
  runtime's `Handle::enter` guard (R-17) may still be active, so
  `Handle::try_current()` can succeed there. So a hook, env callback or warning handler
  under a sync call may make sync rlmesh calls (RUN-8, RUN-25). The twin
  handles the same events inside its own future, on the caller's executor,
  with the driving flag set, so a sync rlmesh call made there fails with
  `Error::Runtime` (RUN-8). This is the one behavioural difference allowed
  between a call and its twin.
- Exception: `EnvServer::serve_async` and `EnvFactoryExt::serve_async` run
  every lane, lane 0 included, on library threads and install no signal
  handlers; sync `serve()` keeps lane 0 on the caller and installs ENV-11's
  handlers (ENV-29). Sync env callbacks inside the future would block the
  caller's executor.
- **Refusal in an async context.** A sync call made from an async execution
  context fails with `Error::Runtime` naming the `_async` twin, before it
  blocks; it never panics (today it panics, `rs/model/worker.rs:270-277`).
  Tokio exposes no test for its own `block_on` panic condition (the entered
  flag is private: `tokio/src/runtime/context/runtime.rs:39-68`,
  `context/blocking.rs:17` is `pub(crate)`), so the condition is defined by
  public probes. The thread is in an async execution context when
  `tokio::task::try_id().is_some()` (it is polling a task of any tokio
  runtime or running a `spawn_blocking` closure) or the library's thread-local driving flag is set. The library
  sets that flag while its sync driver polls a segment on the thread and
  while a twin handles a hook, env op or warning inside its future. Allowed:
  plain threads (`std::thread::spawn`, the library's own callback threads)
  including those that entered a runtime (`Handle::enter`), and event
  handling between sync segments; not `Handle::try_current().is_ok()`, which
  would refuse both. Three known deviations from tokio's own condition, all
  tested (E-3): a `block_in_place` section inside a task is refused (`try_id`
  stays set there); a `spawn_blocking` closure is refused, because tokio runs
  it as a task with an id (`runtime/blocking/pool.rs` `spawn_blocking_inner`,
  `runtime/task/core.rs` `TaskIdGuard` in `Core::poll`; `tests/task_id.rs`
  `task_id_spawn_blocking`) and no public API tells it apart, so such code
  uses `std::thread::spawn` or the `_async` twin; and a caller's own
  `Runtime::block_on` / `Handle::block_on` root future is not a task and is
  not detected, so the call blocks that thread as any blocking call does
  there. None can panic, because the sync driver never calls tokio's
  `block_on` (R-17).
- **Library-run callbacks.** Authored callbacks that the library runs off the
  caller's thread (model corners and hooks behind a served or session-driven
  `Model`, env lanes, C and C++ callbacks alike) run on plain library
  threads, never inside a tokio task or `spawn_blocking` (today's
  `rs/model/engine.rs:1125-1251` and `capi/model.rs:365-494` use
  `spawn_blocking` and change), with the driving flag clear, so they may make
  sync rlmesh calls (a policy that queries a `RemoteModel`), as a Python
  `predict` can.
- C and C++ have no async API.

### R-17 Borrowing and `Session<'a>`

- A run borrows the model and the env: the model is not closed by a run
  (X-7); the env is closed only with `close_env`.
- Rust `Session<'a>` holds a shared model borrow, Running's `ModelTarget<'a>`
  (`&'a Model`, `&'a RemoteModel`, a sandbox model or `RANDOM_SAMPLE`; models
  serialize their own calls), and an env from Running's `EnvTarget<'a>`:
  `&'a mut` of a local env, vector env or factory, `&'a mut RemoteEnv` /
  `&'a mut RemoteVectorEnv`, a shared `&'a EnvServer`, or a connection it
  dialed itself from an address (owned). The `&mut` borrow rules out
  concurrent use of the env from Rust. It does not replace Python's runtime
  check: a remote handle that was driven with `reset` / `step` still holds the
  env's server-side session slot after the borrow ends, and a run or session
  on it fails at runtime with `RuntimeError` (`py/_models/base.py:1389-1397`,
  [Running](running.md)).
- Sandbox models in `ModelTarget` apply when sandbox parity lands (R-7 note).
- Borrowed futures are driven on the caller. The sync method polls the
  borrowed driver future on the calling thread one segment at a time (R-16),
  with the library's own thread-parking executor inside `Handle::enter()` of
  the library runtime. That runtime is multi-threaded, so its worker threads
  drive I/O and timers while the caller parks. The sync method never calls
  tokio's `Handle::block_on`, so no runtime nesting can panic (Python's client
  uses `Handle::block_on` today, `pyrs/client.rs:213-236`, from plain threads
  only). Hooks and relayed env ops run on the caller between segments (RUN-8,
  RUN-25). The async twin is polled by the caller's executor inside the
  library runtime's context (RMT-7) and handles those events inside its
  future. Borrowed state is never moved into `tokio::spawn`, which needs
  `'static` (today's `H: 'static` bound, `rs/model/worker.rs:263`).
- Only `'static` tasks that own everything they touch (connection drivers,
  the close of a connection the session dialed itself) run on library threads,
  and only they may outlive a call as detached cleanup. When a `session_async`
  / `run_async` future that dialed an address is dropped, the close of that
  owned connection is handed to such a detached task and still runs (R-18).
- Drop never blocks inside an async runtime (R-16's condition). Outside one,
  Drop finishes owned cleanup with a bound; inside one, it detaches the owned
  part. Explicit `close()` / `close_async()` is the way to observe close
  errors.
- **Session Drop.** Rust `Drop` of a `Session`, `rlmesh_session_free` and
  `~Session` act as `__exit__` on an exception ([Running](running.md) Session
  lifetime): no viewer hold, errors swallowed. For a local `Model`, the
  model's `on_episode_end` for an open episode is owned cleanup: Drop submits
  it through the `Model`'s own shared (`'static`) internals onto the model's
  serialized call queue (MDL-7), ahead of any later call on that model, so it
  runs although the session's `&'a Model` borrow ends; `Model::close` waits
  for it as it waits for in-flight predicts (X-7). Outside an async runtime
  Drop waits up to the bound for that call and for the owned connections'
  close (past the bound they still finish), and runs `close_env`'s close of
  the borrowed env on the dropping thread. Inside an async runtime Drop waits
  for nothing: `on_episode_end` runs later on a library thread, owned
  connections close on detached tasks, and a borrowed env is not closed even
  with `close_env`, because a borrow cannot be detached; its owner closes it,
  and `close_async()` closes it from async code. C and C++ have no runtime
  context, so `rlmesh_session_free` and `~Session` always take the outside
  path. "Inside" is R-16's condition, so a Rust Drop on a `spawn_blocking`
  thread takes the inside path (R-16's deviation) and does not close a
  borrowed env; drop the session on a plain thread or call `close()` first.
- C and C++ follow the same ownership: a session or run never frees what it
  borrowed; the caller frees handles after the session is freed.

### R-18 Cancellation

- One mechanism, fenced (R-25): Rust and C++ `rlmesh::ext::CancelToken`, C
  `RLMeshExtCancelToken` (`rlmesh_ext_cancel_token_new`,
  `rlmesh_ext_cancel_token_cancel`, `rlmesh_ext_cancel_token_free`), passed
  in a `cancel` options field or obtained from `cancel_handle()` on `Session`,
  `RemoteEnv` and `RemoteVectorEnv` (C `rlmesh_<t>_cancel_handle`, returning
  a new token reference). The `cancel` fields and `cancel_handle` members are
  one of the two fenced extras that sit on shared types, because a token must
  reach the call it cancels (R-25); the other is the serve options `ext`
  field (R-25). The token is the counterpart of Python's
  KeyboardInterrupt, which a 100 ms signal poll raises during blocked calls
  (`pyrs/client.rs:24,212-236`).
- **Token API.** The token is a shared handle (R-11): every copy is the
  same token, so `cancel()` on any copy fires it, and `cancel_handle()`
  returns such a copy by value.
  - Rust: `#[derive(Clone, Debug, Default)] pub struct CancelToken`
    (`Send + Sync`; `Clone` shares), `pub fn new() -> Self` (equal to
    `default()`), `pub fn cancel(&self)`.
  - C: `RLMeshStatus rlmesh_ext_cancel_token_new(RLMeshExtCancelToken**
out)`, `RLMeshStatus rlmesh_ext_cancel_token_clone(const
RLMeshExtCancelToken*, RLMeshExtCancelToken** out)` (a new reference,
    freed independently), `RLMeshStatus
rlmesh_ext_cancel_token_cancel(RLMeshExtCancelToken*)` and `void
rlmesh_ext_cancel_token_free(RLMeshExtCancelToken*)` (NULL is a no-op;
    the token lives until its last reference is freed). `_cancel` may be
    called from any thread, concurrently with the call it cancels; it is
    not async-signal-safe, so a SIGINT handler sets a flag that a thread
    turns into `_cancel`.
  - C++: `rlmesh::ext::CancelToken`, default-constructible (wraps `_new`);
    a copy shares the token (`_clone`), destruction frees one reference;
    `void cancel()`.
- **Firing and re-arming.** One `cancel()` fires once: it fails the call in
  flight on the token or, when none is, the next call made with that token,
  at its start. The token then re-arms, as one Ctrl-C raises one
  KeyboardInterrupt. Repeated `cancel()` before that call fails counts once,
  as Python's signal flag coalesces repeated signals before its poll. This
  holds the same way for `cancel_handle()` tokens and `cancel` options
  fields. Only `cancel` fields can put one token on concurrent calls
  (`Session` and the remote handles refuse overlapping calls); then one
  `cancel()` fails every call in flight on it.
- **State after a cancel.** A cancelled call fails with R-21's cancellation
  class and leaves the handle as Python leaves it after KeyboardInterrupt:
  later calls are allowed. A cancelled single call (`reset`, `predict`,
  `step`, a remote client call) is abandoned and does not end the episode;
  the open episode ends at the next `reset` or at close, as in Python
  (`py/_models/_eval.py:1008-1041`). A cancelled run (`run`, `Model.run`,
  `Session.run`) is a run that failed: it ends its open episode as on any
  other error (Python's `finally`, `py/_models/_eval.py:1469-1477`;
  [Running](running.md)) and fires `on_run_end` with the partial result.
- **Dropped futures.** Dropping an `_async` future is a cancellation with the
  same handle state, but it fires no run hook (`on_run_end`) and makes no
  `on_episode_end` call of its own. Hooks are dispatched to the caller
  (RUN-8), and Drop may run inside the caller's runtime, where nothing may
  block (R-17) and a hook making sync calls fails (R-16). An episode left
  open by a dropped `Session::run_async` stays open and ends at the session's
  next `reset`, `run` or close, Session Drop included (R-17). Dropping a
  `run_async` that built its own session drops that session, so R-17's
  Session Drop ends its episode. On drop, no new env or model call starts
  and the result of any call in flight is discarded; a model callback
  already running on a library thread, or a remote RPC already sent, may
  still run to completion (the server finishes it, RMT-6). The library's own
  tasks for the call own only `'static` state (R-17) and are held by
  abort-on-drop guards, since a plain `JoinHandle` drop would detach them.
  The handle is usable afterwards. R-17's detached close of an owned,
  self-dialed connection still runs.
- SIGINT is cancellation; SIGTERM stops a server gracefully and is not an
  error (Env owns the signal rules).

### R-19 C struct sizing

Applies to every C options struct, out-struct, vtable and lent struct.
`struct_size` is the first field.

- **Minimum.** Each struct has `RLMESH_<NAME>_MIN_SIZE`: its `sizeof` in ABI
  5 (X-1), or in the first later generation that adds the struct. Structs
  that already carry `struct_size` in ABI 4 (`RLMeshModelVtable`,
  `h:431-436`, read at `capi/model.rs:523-528`; `RLMeshEnvVtable` and
  `RLMeshEnvConfig`, `capi/env.rs:106-124`) change layout in ABI 5, so their
  ABI-4 `sizeof` is not a minimum. A
  `struct_size` below the minimum, including 0, is refused with
  `RLMESH_ERR_INVALID_ARGUMENT`, so `= {0}` fails loudly instead of meaning
  "zero episodes". Every field present at the minimum is mandatory to supply;
  a required field can only be added with an ABI bump.
- **Whole fields only.** A field exists for this call only if
  `offsetof(field) + sizeof(field) <= struct_size`. The library never reads or
  writes past `struct_size`, and never reads part of a field.
- **Smaller (older caller).** A field the caller's `struct_size` does not
  cover takes Python's default (its `_INIT` value). Fields added after the
  minimum must therefore be optional, and their all-zero bytes must mean
  Python's default (`has_*` false, NULL, a `has_*`-guarded enum).
- **Larger (newer caller).** Bytes past the library's own `sizeof` must be
  zero; otherwise the call fails with `RLMESH_ERR_TYPE` naming the struct and
  offset. Because appended fields are zero at their default, zero means "left
  at default" and is safe to ignore, while a nonzero byte is a setting this
  library cannot honour. Ignoring it would drop the setting silently; Python
  refuses an unknown keyword with TypeError, so C does the same.
- **Layout.** Appended fields start at or after the previous `sizeof` and
  leave no implicit padding (explicit `_reserved` members, zeroed by `_INIT`),
  so the zero check never reads indeterminate padding. A header test asserts
  both.
- **Out-structs.** The caller initializes with `_INIT` and sets
  `struct_size`. The library writes only the whole fields it knows that fit;
  everything else keeps the caller's `_INIT` value, so a newer caller sees
  `has_*` false for fields an older library did not fill. Below the minimum,
  nothing is written and the call fails.
- **Vtables.** As inputs: an uncovered slot is NULL ("not defined"), and a
  non-NULL slot past the library's `sizeof` is refused with `RLMESH_ERR_TYPE`
  naming the struct and offset, as a nonzero tail is, because the library
  would silently never call it.
- **Lent structs.** A sized struct the library fills or owns and lends by
  pointer, to a callback (`RLMeshEpisodeResult`, `RLMeshStepEvent`; RUN-29)
  or as a borrowed return (the leaves of `rlmesh_model_spec_input/_output`
  and `rlmesh_env_tags_observation/_action` trees; ADP-20), carries the
  library's own `sizeof` in `struct_size`. The reader uses a field only if
  `offsetof(field) + sizeof(field) <= struct_size` and treats any uncovered
  field as its `_INIT` value, so a newer reader sees `has_*` false for fields
  an older library does not know.
- No API takes an array of sized structs (use an array of handles or
  pointers), and no sized struct is passed by value. By-value structs are
  frozen per ABI generation.

### R-20 C conventions

- Constructors from a vtable take
  `(const Vtable*, void* user_data, const Options* /* NULL = INIT */, T** out)`;
  a type whose Python constructor takes no keywords (`RunHooks`,
  `RunContext`, `AdapterBase`) has no options parameter:
  `(const Vtable*, void* user_data, T** out)`.
- Callbacks return `RLMeshStatus`, nonzero after `rlmesh_callback_set_error`
  or `rlmesh_callback_set_error_class` (R-12).
  Every vtable with `user_data` has `void (*free_user_data)(void*)` (NULL =
  none), called once when the library drops its last copy. Every other call
  that takes a callback plus `user_data` takes `free_user_data` beside it
  (R-3 Callables row; `rlmesh_set_warning_handler`, R-23).
- Ownership of `user_data` passes to the library only when the call that
  takes it returns `RLMESH_OK`. A call that fails, including a vtable
  constructor, never calls `free_user_data`; the caller keeps ownership and
  frees it itself (the C++ wrapper deletes its trampoline object and throws).
- Collections are read with `rlmesh_<t>_len` and `rlmesh_<t>_get(i)`.
- The caller-buffer form (`out`, `cap`, `out_len`) is only for numeric arrays
  and raw bytes; owned variable-length text is `RLMeshBytes`.
- Every handle has `rlmesh_<t>_free` (NULL is a no-op); `_close` exists where
  Python has `close()`.

### R-21 Errors

Each exception class that rlmesh's Stable surface raises has exactly one row.
A Stable raise of a class neither listed nor folded into a row (below) is a
bug in this table. Experimental
symbols are out of scope: the `params` and `platform` modules and the sandbox
clients (`SandboxEnv`, `SandboxVectorEnv`, `SandboxModel`, `SandboxRuntime`,
`SandboxBuild`). The sandbox clients raise builtin `RuntimeError` for
infrastructure failures (Docker missing, pull failed, container exited;
`py/_sandbox/session.py:397-406,426-445,517-530`); before sandbox parity
lands that must move to a non-misuse class (`RLMeshException` or
`ConnectionError`), with a Python change.

| Python class (Python base)                                 | Raised for                                                                                                                                                                                                                                                                                                                                                                                                                 | Rust                                                            | C status (value)                                                                 | C++ class : base                                                 |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `RLMeshException` (`RuntimeError`)                         | an internal or server-side failure with no subclass                                                                                                                                                                                                                                                                                                                                                                        | `Error::Internal(String)`                                       | `RLMESH_ERR_INTERNAL` (99); C-only `RLMESH_ERR_PANIC` (7) maps here              | `rlmesh::RLMeshException : rlmesh::RuntimeError`                 |
| `ProtocolException` (`RLMeshException`)                    | generation mismatch, refused edition, failed handshake, malformed message                                                                                                                                                                                                                                                                                                                                                  | `Error::Protocol(String)`                                       | `RLMESH_ERR_PROTOCOL` (9, new)                                                   | `rlmesh::ProtocolException : RLMeshException`                    |
| `EnvironmentException` (`RLMeshException`)                 | the env reported a failure; carries `code`                                                                                                                                                                                                                                                                                                                                                                                 | `Error::Environment(EnvironmentError)`, `is_recoverable: false` | `RLMESH_ERR_ENVIRONMENT` (3)                                                     | `rlmesh::EnvironmentException : RLMeshException`                 |
| `RecoverableEnvironmentException` (`EnvironmentException`) | as above; the request failed, the session stays usable                                                                                                                                                                                                                                                                                                                                                                     | `Error::Environment(e)`, `e.is_recoverable: true`               | 3, `rlmesh_last_error_is_recoverable() != 0`                                     | `rlmesh::RecoverableEnvironmentException : EnvironmentException` |
| `ModelException` (`RLMeshException`), new (D-10)           | the model reported a failure; carries `code`, `is_recoverable`                                                                                                                                                                                                                                                                                                                                                             | `Error::Model(ModelError)`                                      | `RLMESH_ERR_MODEL` (4)                                                           | `rlmesh::ModelException : RLMeshException`                       |
| `ValueError`                                               | an invalid argument value, address or options field                                                                                                                                                                                                                                                                                                                                                                        | `Error::InvalidArgument(InvalidArgumentError)`                  | `RLMESH_ERR_INVALID_ARGUMENT` (1)                                                | `rlmesh::ValueError : std::invalid_argument`                     |
| `AdapterResolutionError` (`ValueError`)                    | adapters cannot bridge env and model                                                                                                                                                                                                                                                                                                                                                                                       | `Error::InvalidArgument(e)`, `e.adapter_resolution.is_some()`   | 1, `RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR`                                 | `rlmesh::AdapterResolutionError : ValueError`                    |
| `TypeError`                                                | the wrong kind of value or object (a value of another kind, a local env for `RemoteModel`, `view` misuse)                                                                                                                                                                                                                                                                                                                  | `Error::Type(String)`                                           | `RLMESH_ERR_TYPE` (10, new)                                                      | `rlmesh::TypeError : std::logic_error`                           |
| builtin `RuntimeError`                                     | misuse (closed handle, overlapping or cross-thread call, blocking call inside a runtime, a local `EnvServer` whose contract lacks spaces passed to `RemoteModel.session` (RMT-12; a remote peer's is `ProtocolException`)), local refusals (episode capacity), and a call to a `Model` corner that is not defined (Python raises the subclass `NotImplementedError`, `py/_models/base.py:785-851`; folded here, see below) | `Error::Runtime(String)`                                        | `RLMESH_ERR_RUNTIME` (11, new)                                                   | `rlmesh::RuntimeError : std::runtime_error`                      |
| builtin `OSError`                                          | a filesystem failure: `Recorder.export` refusing its target (Python raises the subclass `FileExistsError`, `py/recorder/export.py:62-74`; folded here, see below) and failed mkdir, write or copy                                                                                                                                                                                                                          | `Error::Os(String)`                                             | `RLMESH_ERR_OS` (12, new)                                                        | `rlmesh::OSError : std::runtime_error`                           |
| `ConnectionError` (`OSError`)                              | a transport failure                                                                                                                                                                                                                                                                                                                                                                                                        | `Error::Connection(String)`                                     | `RLMESH_ERR_CONNECTION` (5); `RLMESH_ERR_TRANSPORT` deprecated alias, same value | `rlmesh::ConnectionError : rlmesh::OSError`                      |
| `TimeoutError` (`OSError`)                                 | a connect or request deadline                                                                                                                                                                                                                                                                                                                                                                                              | `Error::Timeout(Duration)`                                      | `RLMESH_ERR_TIMEOUT` (6)                                                         | `rlmesh::TimeoutError : rlmesh::OSError`                         |
| `KeyboardInterrupt` (`BaseException`)                      | cancellation (R-18)                                                                                                                                                                                                                                                                                                                                                                                                        | `Error::Cancelled`                                              | `RLMESH_ERR_CANCELLED` (8)                                                       | `rlmesh::ext::CancelledError`, no standard base (D-15)           |
| —                                                          | C only, deprecated                                                                                                                                                                                                                                                                                                                                                                                                         | —                                                               | `RLMESH_ERR_INVALID_VALUE` (2): kept, deprecated, never returned from ABI 5      | —                                                                |

Status values 0-8 and 99 keep today's discriminants
(`capi/abi/status.rs:13-26`); new statuses take 9-12.

**Folded subclasses.** Where Python's own code raises an idiomatic
builtin subclass, `NotImplementedError` (an undefined `Model` corner) and
`FileExistsError` (an export target; likewise `PermissionError` and the other
`OSError` subclasses its filesystem calls raise), Python keeps it, and the native
languages raise the row's class: `RuntimeError` and `OSError`. They have no
`ErrorClass` entry. `except RuntimeError` / `except OSError` and the native
catch of the same class match both, so "Same inheritance" holds for every
class in this table; a clause naming the subclass itself has no native
counterpart. Likewise rlmesh's own Experimental `ValueError` subclasses
raised from Stable calls, `ParamError` and its subclasses
`MissingParamError` and `UnknownParamError` (`py/params/_resolve.py:36-56`;
raised by `Model` load through `from_config`, Model MDL-19, and by
`EnvFactory.make` with `tag_params` for a discriminant value outside its
choices, `py/_authoring.py:345-357`): Python keeps them, natively they are
the ValueError row with no `ErrorClass` entry, and an `except` naming them
has no native counterpart. E-3 compares every folded subclass by
`isinstance` against the row's class.

Exceptions that a Python protocol or the binding layer requires stay
Python-only and have no row: `KeyError` (only a `__getitem__` lookup on a
Dict space or Dict value; encoding a value that lacks a Dict-space key, single
or batched, is the ValueError class, and decoding a received value that
lacks one is `ProtocolException`, P-11), `BufferError` (`__dlpack__` refusals),
`OverflowError` (an integer outside the range of the native parameter type:
a negative value for an unsigned parameter, a value above `u32` / `i64`,
such as a metadata integer beyond int64; natively such a value cannot be
written, or the area's Python change makes it the ValueError class) and
`AttributeError` (an Adapter-only call on a custom adapter).
Their native forms are in Spaces SPC-27 and Adapters ADP-10. Importing a
trusted custom-input entrypoint in `resolve` no longer leaks `ImportError`
or `AttributeError` (P-10).

- **Same inheritance.** C++ classes copy Python's tree, so
  `catch (const rlmesh::RuntimeError&)` catches exactly what
  `except RuntimeError` catches from rlmesh, and
  `catch (const rlmesh::RLMeshException&)` excludes the builtins, as Python
  does. Catching a `std::` base is wider than any Python clause; that and a
  Python clause naming a folded subclass (above) are the only allowed
  differences.
- **Flat languages.** Rust and C cannot nest variants, so a subclass with no
  variant of its own (`RecoverableEnvironmentException`,
  `AdapterResolutionError`) uses its parent's variant or status, and matching
  the parent catches it, as `except Parent` does. Every error also answers
  its exact class: Rust `Error::class() -> ErrorClass`, C
  `rlmesh_last_error_class() -> RLMeshErrorClass`. `ErrorClass` is a closed
  vocabulary (R-9) of the 13 Python classes above (folded subclasses
  excluded), with `name()` (Python's spelling) and `is_subclass_of(other)`
  following Python's MRO, so `ConnectionError` and `TimeoutError` report
  `OSError` as a base; C has
  `rlmesh_error_class_name` and `rlmesh_error_class_is_subclass`.
- **Misuse is not a model failure.** Builtin `RuntimeError` keeps its own
  category; nothing that Python raises as `RuntimeError` maps to
  `Error::Model`. `TypeError` is distinct from `ValueError`.
- **Detail fields.** `EnvironmentError { code, message, is_recoverable }`,
  `ModelError { code, message, is_recoverable }` (the wire `ModelErrorCode`,
  dropped today at `rs/error.rs:252-255`) and `InvalidArgumentError
{ message, adapter_resolution }` are owned by [Core](core.md) and
  [Adapters](adapters.md). C reads codes with `rlmesh_last_error_code`.
- **Authored envs and models.** Only an Environment-class error's own flag
  makes an authored env failure recoverable, and the code sent is always
  `INTERNAL`. A `ModelException` (or any other class) raised inside an env is
  never recoverable; Python today honours any truthy `is_recoverable`
  attribute (`pyrs/server/py_environment.rs:418-429`) and changes (P-9).
  Authors raise them as: Python `EnvironmentException(message)`,
  `RecoverableEnvironmentException(message)` and
  `ModelException(message, *, is_recoverable=False)` (COR-22); Rust
  `Error::environment` / `Error::environment_recoverable` and `Error::model` /
  `Error::model_recoverable` (renamed constructors); C
  `rlmesh_callback_set_error(msg, recoverable)` (R-12); C++
  `throw rlmesh::EnvironmentException(msg)`,
  `throw rlmesh::RecoverableEnvironmentException(msg)` and
  `throw rlmesh::ModelException(msg, is_recoverable)` (R-6 exception,
  COR-20).
- Rust `Error` stays `#[non_exhaustive]`. `Address` folds into
  `InvalidArgument` and `Server` into `Internal`.

Python today (`pyrs/types.rs:42-78`): `Model`, `Server` and `Internal` raise
bare `RuntimeError`; protocol failures arrive as `Internal`
(`rs/error.rs:205-209,248`), so `ProtocolException` is never raised; env codes
`TIMEOUT`, `INVALID_ACTION`, `CANCELLED` raise `TimeoutError`, `ValueError`,
`RuntimeError`. P-1 to P-5 move Python onto the table.

### R-22 Error text

One formatter in the Rust facade produces the text. Rust `Display`, C
`rlmesh_last_error_message`, C++ `what()` and Python `str(exc)` are equal.
The canonical text per row (Core COR-5 holds the same list):

| Row                                                                            | Text                                                                                                                       |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `EnvironmentException`, `RecoverableEnvironmentException`                      | `[CODE] message` (`[INVALID_ACTION] bad action`)                                                                           |
| `ModelException`                                                               | `[CODE] message` (P-1)                                                                                                     |
| `RLMeshException`                                                              | the raw message; a folded `Server` failure keeps `server error: message`                                                   |
| `ProtocolException`, `TypeError`, `RuntimeError`, `OSError`, `ConnectionError` | the raw message                                                                                                            |
| `ValueError`, `AdapterResolutionError`                                         | the raw message; an address error is `invalid address: message`                                                            |
| `TimeoutError`                                                                 | `operation timed out after {:?}` of the `Duration` (`operation timed out after 5s`), for every deadline (P-8)              |
| `KeyboardInterrupt`                                                            | `cancelled` natively; Python's signal-raised `KeyboardInterrupt` keeps the interpreter's empty text, outside this equality |

Rust drops its `connection error:` style prefixes (`rs/error.rs:163-173`).
Errors that Python's own code raises (the folded `FileExistsError` of an
export refusal) have the same text natively, with these exceptions, each
because a native policy carries no Python type name (C's vtable has none and
C++ wraps C):

- the Rust undefined-corner text names the missing `CORNERS` entry
  ([Model](model.md));
- the `LOAD_KEYS` unknown-key refusal (ValueError class, MDL-19) is
  `extra='passthrough' but load has no **kwargs to forward: a, b` (names
  sorted), naming `load` where Python names the qualname `<Cls>.load`
  (`py/params/_resolve.py:150-157,455-456`);
- the default-`load` refusal (TypeError class, Model's Load) is `model
received construction params (a, b) but does not override load(...) to
apply them; implement load(**kwargs) to consume the binding` (names
  sorted), with `model` where Python names `{Cls}`
  (`py/_bootstrap/loaders.py:388-398`).

### R-23 Warnings

- **Categories.** A warning is `Warning { category, message }`.
  `WarningCategory` is a closed vocabulary: `UserWarning`, `RuntimeWarning`,
  `WorkflowEditionWarning` (a `UserWarning`, `py/_editions.py:53`), with
  `name()` and `is_subclass_of` (C `const char*
rlmesh_warning_category_name(RLMeshWarningCategory)` and `bool
rlmesh_warning_category_is_subclass(RLMeshWarningCategory cls,
RLMeshWarningCategory base)`, as R-21's error-class pair; R-9 exception).
  Each native warning keeps the category Python
  uses for the same event: `RuntimeWarning` for a short chunk
  (`py/_models/_chunk.py:136-141`), `UserWarning` for adapter notes
  (`py/adapters/tag.py:63`), `execution_horizon` without a chunk corner
  (`py/_models/base.py:1188,1448,1681`) and an undeclared `trial_index`
  (`py/_models/_eval.py:1048`), `WorkflowEditionWarning` for a floating
  edition (`py/_editions.py:241-248`).
- **Types.** A value struct (R-11) in each language:
  - Rust: `#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive] pub enum WarningCategory { UserWarning,
RuntimeWarning, WorkflowEditionWarning }` with `pub fn name(self) ->
&'static str` and `pub fn is_subclass_of(self, other: WarningCategory)
-> bool`; `#[derive(Clone, Debug, PartialEq, Eq)] pub struct Warning {
pub category: WarningCategory, pub message: String }`, `Display` =
    `<Category>: <message>`.
  - C: `typedef enum RLMeshWarningCategory {
RLMESH_WARNING_CATEGORY_INVALID = 0,
RLMESH_WARNING_CATEGORY_USER_WARNING = 1,
RLMESH_WARNING_CATEGORY_RUNTIME_WARNING = 2,
RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING = 3 }
RLMeshWarningCategory;` (R-9; no wire enum). A warning reaches C only as
    the handler's arguments, so there is no warning struct.
  - C++: `enum class WarningCategory { UserWarning = 1, RuntimeWarning = 2,
WorkflowEditionWarning = 3 }` (values of the C enum), with free
    `std::string_view name(WarningCategory)` and `bool
is_subclass_of(WarningCategory, WarningCategory)`; `struct Warning {
WarningCategory category; std::string message; }` with `operator==`
    and `operator<<`.
- **Handler.**
  - Rust: `pub fn set_warning_handler(handler: Option<Box<dyn Fn(&Warning)
-> Result<()> + Send + Sync>>)`.
  - C: `typedef RLMeshStatus (*RLMeshWarningHandlerFn)(void* user_data,
RLMeshWarningCategory category, const char* message);` (`message` is
    NUL-terminated UTF-8, borrowed for the call) and `RLMeshStatus
rlmesh_set_warning_handler(RLMeshWarningHandlerFn fn, void* user_data,
void (*free_user_data)(void*));`. It returns `RLMESH_OK`, or
    `RLMESH_ERR_INVALID_ARGUMENT` for a NULL `fn` with non-NULL `user_data`
    or `free_user_data`, which keeps the installed handler and leaves
    `user_data` with the caller (R-20). A failing `fn` reports its error
    with R-12's calls (`RLMeshException` kind).
  - C++: `void rlmesh::set_warning_handler(std::function<void(const
Warning&)> handler)`; an empty `std::function` restores the default.
  - Process-wide; `None` / NULL `fn` / empty restores the default.
    Installing or resetting is atomic with respect to emission: the library
    holds the handler by reference count, an emission uses the handler
    current when it starts, and `free_user_data` (NULL = none) runs once,
    after the replacement or reset and after every in-flight call of the
    old handler has returned (R-20's last copy), on whichever thread drops
    that copy.
- **Thread.** The handler runs synchronously on the thread that made the API
  call that warned (for an `_async` twin, the thread polling its future). A
  warning raised on a library thread is queued and delivered on that calling
  thread before the call returns, in emission order, so Python's binding sees
  its caller frames and a failing handler can abort that call. Only a
  warning with no API call in flight (a background server's request) runs
  the handler on a library thread; Python attributes it to no caller frame,
  and a failing handler fails that served request. Handlers must be
  thread-safe, since calls on different threads warn concurrently.
- **Default.** One line to stderr, `rlmesh: <Category>: <message>`, each
  distinct (category, message) once per process, the native stand-in for
  Python's default filter. No `tracing` autodetection (D-13).
- **Warnings as errors.** A handler that fails (Rust `Err`, C nonzero, C++
  throws) aborts the operation that warned, and the call returns that error
  (C++ rethrows the handler's exception, carried by R-12's error payload
  channel). The one exception is Running
  RUN-19: recorder capture warnings never abort a run, as in Python
  (`py/recorder/hooks.py:72-83`). Python's binding installs a handler
  that calls `warnings.warn` with the category and the first caller frame
  outside rlmesh (`py/_editions.py:215`); under `-W error` the raised
  `Warning` propagates out of the Python call that triggered it, unchanged.
- Warnings raised by Python code stay plain `warnings.warn`.

### R-24 Python-only bridges stay Python-only

NumPy, Gymnasium conversion, Torch, JAX, the `ValueBridge` parameters and the
`rlmesh.types` typing protocols have no counterpart and need none. A
parameter that exists only to select a bridge (`bridge=`, `input_bridge=`) or
to negotiate a Python protocol (`__dlpack__`'s `max_version`, `stream`;
SPC-19) is dropped natively.

### R-25 Language-only extras are fenced

- Rust: `rlmesh::ext` holds documented user-facing extras (`CancelToken`,
  address types, endpoint-token constants, `remove_stale_socket`; each area
  lists its own).
  `rlmesh::binding` is `#[doc(hidden)]` and semver-exempt, for the in-repo
  bindings (Python, C ABI). Nothing language-only stays at the crate root, in
  `rlmesh::prelude` or in a shared module.
- C: `rlmesh_ext_` functions and `RLMeshExt` types (D-6), such as
  `RLMeshExtCancelToken` and its `rlmesh_ext_cancel_token_*` calls. Two
  exceptions sit on shared types, in every native language, because the
  extra must reach the call that uses it; their type is fenced:
  - R-18's cancellation entry points: the `cancel` options fields and
    `cancel_handle()` members (C `rlmesh_<t>_cancel_handle`), typed as the
    fenced token;
  - the `ext` field of the serve options structs, `EnvServerOptions.ext`
    (ENV-12) and `ModelServeOptions.ext` (MDL-20): Rust
    `Option<ext::EnvServerOptions>` / `Option<ext::ModelServeOptions>`, C
    `const RLMeshExtEnvServerOptions*` / `const RLMeshExtModelServeOptions*`
    (NULL = none), C++ `std::optional<ext::EnvServerOptions>` /
    `std::optional<ext::ModelServeOptions>`.

  No other shared type carries a user-facing fenced member. Rust `Error`'s
  `#[doc(hidden)]` `Attached` variant (R-12's error payload channel) is
  `binding` plumbing, not an extra: no shared-surface call returns it and
  C and C++ have no counterpart. The ABI
  identification items `RLMESH_ABI_VERSION`, `rlmesh_abi_version` and
  `rlmesh_abi_check` keep their names, because a consumer must call them
  against any ABI generation.

- C++: `rlmesh::ext`.
- Trait impls cannot be crate-private: an impl is public wherever both types
  are nameable. Each public `From` impl from a plumbing type becomes a
  `pub(crate) fn` and call sites use `.map_err(Error::from_grpc)` and the
  like:

  | Today                                                                                                                | Becomes                                     |
  | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
  | `From<rlmesh_grpc::error::Error> for Error` (`rs/error.rs:219`)                                                      | `pub(crate) fn Error::from_grpc`            |
  | `From<rlmesh_grpc::error::ProtocolError> for Error` (`:205`)                                                         | `pub(crate) fn Error::from_protocol`        |
  | `From<rlmesh_adapters::v1::ApplyError> for Error` (`:211`)                                                           | `pub(crate) fn Error::from_apply`           |
  | `From<EnvError> for EnvironmentError` (`:195`)                                                                       | `pub(crate) fn EnvironmentError::from_wire` |
  | `From<ServeOptions> for rlmesh_grpc::ServeOptions` (`rs/serve_options.rs:91`)                                        | `pub(crate) fn ServeOptions::to_grpc`       |
  | `From<&ModelRouteContext> for AdapterContext` (`rs/model/wire.rs:200`)                                               | moves with `ModelRouteContext` to `binding` |
  | `From<ConnectAddress> for RunLocalOptions`, `From<BindAddress> for ServeModelOptions` (`rs/model/worker.rs:203,257`) | removed with their types                    |

- Top-level Rust exports to move (`rs/lib.rs:132-167`; the prelude re-exports
  at `rs/prelude.rs:11-20` follow):

  | Export                                                                                                                                                                                                                                                                                                                                     | Destination                                                                                              |
  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
  | `ModelHandler`, `ModelWorker`, `PredictFn`, `AdaptedModelHandler`, `RouteResolver`, `RouteConfig`, `RouteNeeds`, `ResolveOptions`, `ModelRouteSetup`, `ModelRouteContext`, `ModelHandlerRuntimeModel`, `EnvClientRuntimeEnv`, `ModelObservation`, `PredictFrames`, `HeldState`, `HistoryNeeds`, `HistoryFrame`, `EpisodeInfo` (`:142-148`) | `binding`                                                                                                |
  | `ResetRequest`, `ResetResult`, `StepRequest`, `StepResult`, `RenderRequest`, `RenderResult`, `CloseRequest`, `CloseResult`, `Vector*Request`, `Vector*Result`, `VectorCloseResult`, `EpisodeMetadata`, `EndpointPhases` (`:135-140`)                                                                                                       | `binding`                                                                                                |
  | `EnvRuntimeError`, `SpaceValue` (`:166`)                                                                                                                                                                                                                                                                                                   | `binding`                                                                                                |
  | `RenderFrame` (`:166`)                                                                                                                                                                                                                                                                                                                     | `binding` with `RenderRequest` / `RenderResult`; removed once `render` returns a `Value` ([Env](env.md)) |
  | `CURRENT_WORKFLOW_EDITION`, `parse_declared_edition` (`:149`); `PROTOCOL_GENERATION`, `PeerInfoOverride`, `peer_info_override`, `update_peer_info_override`, `supported_workflow_editions` (`:152-155`)                                                                                                                                    | `binding` (`build_info()` and `current_workflow_edition()` are the shared surface)                       |
  | `telemetry` (`:157`), `PeerCeiling`, `TRIAL_INDEX_OPTION` (`:164`)                                                                                                                                                                                                                                                                         | `binding`                                                                                                |
  | `remove_stale_socket` (`:132`), `BindAddress`, `ConnectAddress` (`:133`), `ENV_ENDPOINT_TOKEN_ENV`, `MODEL_ENDPOINT_TOKEN_ENV` (`:165`)                                                                                                                                                                                                    | `ext`                                                                                                    |
  | `async_trait` (`:134`), `CancellationToken` (`:167`)                                                                                                                                                                                                                                                                                       | removed; `ext::CancelToken` replaces the token                                                           |
  | `RuntimeHooks`, `HookError`, `NoopRuntimeHooks`, the `*Event` types (`:159-162`), `RuntimeReport` (`:164`), `RunLocalOptions`, `ServeModelOptions` (`:146`), `BoundEnvServer`, `BoundModelServer`                                                                                                                                          | `binding` until their replacements land, then removed                                                    |

  The prelude (`rs/prelude.rs:11-20`) also drops `async_trait` (`:11`) and
  `spaces::BinaryPayload` (`:12`), which moves to `binding` with the
  `rlmesh::spaces` wire extras ([Spaces](spaces.md)). Afterwards the prelude re-exports only shared-surface
  names an author needs: the authoring traits (`Env`, `VectorEnv`,
  `EnvFactory` with its `EnvFactoryExt`, `Policy`), every type named in
  those traits' signatures (for example `Value`, `Metadata` / `InfoDict` /
  `Kwargs`, `Space`, `VectorSeed`, `MadeEnv`, `TagParam`, `Variant`,
  `adapters::EnvTags`, `PredictContext`, `BatchPredictContext`, `Corners`,
  `SpecArg`; the traits are the list), and `Error`, `ErrorClass`, `Result`;
  nothing from `ext` or `binding`.

  `ENV_RESET_OPTIONS_KEY` and `predict_seed` stay: they are Stable Python
  symbols. The `rlmesh::spaces` wire extras (`rs/spaces.rs:17-40`) are listed
  in [Spaces](spaces.md).

### R-26 Full parity is the target

C, C++ and Rust each cover the whole Stable Python surface except R-24's
bridges. Until a symbol lands, its area table marks it missing; a missing
symbol is a gap (G-n), not a design choice.

### R-27 One home per shared type

- **One value type.** `rlmesh::Value` (Rust, C++) and `RLMeshValue*` (C)
  carry `rlmesh.types.Value`. No `SpaceValue`, `ModelValue`,
  `RLMeshModelValue` or C++ `ValueRef` on the shared surface.
- **One metadata map.** Rust `Metadata` with aliases `InfoDict` and `Kwargs`,
  a C `RLMeshValue*` of kind `DICT`, C++ `Metadata` / `InfoDict` / `Kwargs`;
  insertion-ordered like a Python `dict` (COR-2). No `MetaMap`,
  `RLMeshMetadata` or `RLMeshMetaMap`.
- **One contract type.** `rlmesh::specs::EnvContract`, C `RLMeshEnvContract`,
  owned by Env.
- Owners (the area that defines the symbol; others link to it):

  | Area                    | Owns                                                                                                                                                                                                                                                                                  |
  | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | rules.md                | error classes and statuses (R-21), C callback errors and the error payload channel (R-12), warnings and `Warning` (R-23), cancellation (R-18), C sizing (R-19)                                                                                                                        |
  | [Core](core.md)         | `Value`, `Metadata` / `InfoDict` / `Kwargs`, error detail fields, editions, describe, build info, `ENV_RESET_OPTIONS_KEY`                                                                                                                                                             |
  | [Spaces](spaces.md)     | spaces, `SpaceSpec`, `Tensor`, `DType`, `Device`, C text views                                                                                                                                                                                                                        |
  | [Env](env.md)           | `Env`, `VectorEnv`, `VectorSeed`, `EnvFactory`, `EnvFactoryExt`, C++ `EnvFactoryBase`, `EnvFactoryServeOptions`, `MadeEnv`, `TagParam`, `Variant`, `VariantOptions`, `EnvServer`, `EnvServerOptions`, `ServeOptions`, `Transport`, `EnvContract`, `ServedEnv`, `trial_index`, signals |
  | [Model](model.md)       | `Model`, `Policy`, `PredictContext`, `BatchPredictContext`, `SpecArg`, `NO_ADAPTER`, `predict_seed`, the entrypoint registry                                                                                                                                                          |
  | [Running](running.md)   | `run`, `session`, `Session`, `RunContext`, `RunHooks`, results, `TelemetryRow`, `StepEvent`, `Reader`, `FrameSources`, `ReadItem`, `View`, `Recorder` and its records (`WorkloadRecord`, `EpisodeRecord`, `MediaRef`), `RANDOM_SAMPLE`, `ModelTarget`, `EnvTarget`                    |
  | [Remote](remote.md)     | `RemoteEnv`, `RemoteVectorEnv`, `RemoteModel`, their options structs, sandbox clients, client timeouts                                                                                                                                                                                |
  | [Adapters](adapters.md) | `EnvTags`, `ModelSpec`, `ObservationRoles`, `tag`, `resolve`, `Adapter`, `Advisory`, `adapters.*` keys, `AdapterResolutionError` detail                                                                                                                                               |

## Decisions

| ID   | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Status                  |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| D-1  | Sync first. Every primary API blocks; Rust adds opt-in `*_async` twins (R-16).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | User                    |
| D-2  | Full parity in C. C covers the whole surface through handles and options structs; C++ wraps C one to one.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | User                    |
| D-3  | C defaults come from `RLMESH_<NAME>_INIT` macros guarded by `struct_size` (R-19).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | User                    |
| D-4  | `RemoteEnv`, `RemoteVectorEnv`, `RemoteModel` and `Session` exist in Rust, C and C++.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | User                    |
| D-5  | C++17 minimum with exceptions, tested at C++17 and C++20. One header shape: no standard-dependent layouts or member sets; mixed-standard translation units may share wrapper types. Detection traits replace concepts, `rlmesh::Span` replaces `std::span` in public signatures, and options remain aggregates with portable `with_<field>` setters. The C++20 part of D-5 is reversed on 2026-10-16 to reduce integration friction with Unreal 5.0–5.2, ROS 2, Omniverse and Godot. `Result<T>`, `RLMESH_TRY`, abort-on-error and `-fno-exceptions` support remain removed.                                                                                                                                                                                                                                                                                    | Settled                 |
| D-6  | C naming. Types and their functions use the type name without its module: `RLMeshBox` / `rlmesh_box_*`, `RLMeshEnvTags` / `rlmesh_env_tags_*`. Free functions and constants keep their module: `rlmesh_adapters_tag`, `rlmesh_spaces_space_from_spec`, `RLMESH_ADAPTERS_ENV_METADATA_KEY`; root ones take none (`rlmesh_run`). A type name used in two modules keeps the module for both (`RLMeshSpacesText`, `RLMeshAdaptersText`); today the only such pair is `Text`. A free function named after the type it builds is `rlmesh_<t>_new` (`rlmesh.session` builds a `Session` → `rlmesh_session_new`); other builders keep their module (`rlmesh_adapters_resolve` and `rlmesh_adapters_resolve_from_contract` build an `RLMeshAdapter`, whose `rlmesh_adapter_new` is the vtable constructor). Language-only extras use `rlmesh_ext_` / `RLMeshExt` (R-25). | Settled                 |
| D-7  | Project edition lookup. Rust, C and C++ resolve the workflow edition with Python's chain minus its project-file rung (Core COR-6: the call's own value, including the `ServeOptions.workflow_edition` given to `EnvServer` or `Model.serve`; `RLMESH_WORKFLOW_EDITION`; the served peer's option; the type's declaration); they read no project file. Python keeps its `pyproject.toml` `[tool.rlmesh] workflow_edition` rung (`py/_editions.py:162-190`) as a Python-only convenience (R-24).                                                                                                                                                                                                                                                                                                                                                                  | User                    |
| D-8  | Deprecated aliases are not ported. `rlmesh.__build__` (deprecated, `py/__init__.py:106-110`) has no counterpart and is not a gap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Settled                 |
| D-9  | `sanitize_metadata` (`py/_metadata.py:23`) is a Python bridge: it coerces arbitrary Python objects. Relabel it in compatibility.md and `api_metadata.json`; no native counterpart.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | User (accepted default) |
| D-10 | Add a Stable `rlmesh.ModelException(RLMeshException)` with `code` and `is_recoverable`, raised for model failures (P-1). Additive; existing `except RuntimeError` keeps matching. Builtin `RuntimeError` is reserved for misuse (D-12).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | User                    |
| D-11 | Env failures always raise `(Recoverable)EnvironmentException` with the code, including `TIMEOUT`, `INVALID_ACTION` and `CANCELLED` (P-4). Breaking for `except ValueError` / `except TimeoutError` around env calls; needs a changelog entry. The alternative, hybrid classes, has no flat Rust or C form. Follows D-16.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Settled                 |
| D-12 | Builtin `RuntimeError` and `TypeError` raised by rlmesh get their own categories (`Error::Runtime`, `Error::Type`), not `Model` and not `InvalidArgument` (R-21).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | User                    |
| D-13 | The native default warning handler writes to stderr. No `tracing` autodetection: a warning must not depend on whether a subscriber happens to be installed (R-23).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Settled                 |
| D-14 | A C struct larger than the library's must have zero unknown trailing bytes (R-19), mirroring Python's TypeError for an unknown keyword.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Settled                 |
| D-16 | Python is the reference for concepts, not for quirks. Where Python's current behaviour is accidental or inconsistent, Python changes to the target in the same minor release, with a changelog entry (listed as `P-n` / `<AREA>-P<n>`), instead of every language copying the quirk. Deliberate Python behaviour is matched exactly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | User                    |
| D-15 | C++ `rlmesh::ext::CancelledError` derives from no `std::` exception, as `KeyboardInterrupt` is not an `Exception`: `catch (const std::exception&)` does not swallow a cancel, as `except Exception` does not. It still has `what()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Settled                 |

### Review notes

- Rejected review (r1-core 3, partial): one Rust variant per Python subclass.
  A flat enum cannot make matching `Error::Environment` also catch a
  recoverable failure, which `except EnvironmentException` does; subclasses
  share the parent's variant and are told apart by `ErrorClass`, while C++
  keeps real subclasses. The rest of r1-core 3 is applied (D-12).
- r1-core default "native editions: no manifest rung" is adopted by the user
  (D-7).
- Consistency pass across the area files: R-17 names Running's
  `ModelTarget` / `EnvTarget` forms; R-20 lets keyword-less vtable types
  (`RunHooks`, `RunContext`, `AdapterBase`) omit options; R-21 lists the
  Python-only protocol exceptions (SPC-27, ADP-10); R-24 drops protocol
  negotiation parameters (SPC-19); R-27's owners match each area's "Owns"
  list; R-9 no longer points at an Adapters list of open enums (ADP-4); the
  Shapes wrap the spec in `SpecArg`. Accepted defaults are listed in the
  [index](../language-parity.md#decisions-accepted-by-the-user).
- Applied without change: r1-core 4 (R-23, D-13), r1-core 5 (R-25), r1-core 6
  (INVALID_VALUE keeps 2), r1-run 2 (R-16, R-17), r1-remote 1 (R-17), r1-remote
  2 (R-18), r1-remote 3 (R-21 protocol row, P-3).
- Round 2, applied: `NotImplementedError` folds into the builtin
  `RuntimeError` row and `FileExistsError` into a new builtin `OSError` row
  (`Error::Os`, `RLMESH_ERR_OS` 12, `rlmesh::OSError`, an `ErrorClass`
  member; 13 classes); R-4 reserves a word in every native language if any
  of Rust, C++ or C reserves it; R-17 keeps the remote session-slot check at
  runtime; R-17 / R-18 settle dropped futures (owned self-dialed close is
  detached, no `on_run_end`); R-16 defines "async execution context" as
  tokio's `block_on` condition; R-22 names each row's text (P-8); R-21's
  authored-env recoverability is a target (P-9); R-6 states the options-field
  SemVer policy; R-20 / R-23 add `free_user_data` to the warning handler and
  the failed-call ownership rule; D-7 / P-6 list COR-6's rungs; R-21 scopes
  out Experimental symbols, sandbox clients included (R-7 note, R-17);
  R-25 adds `RenderFrame` and the prelude extras and says what the prelude
  keeps; R-14 path fixed; R-23 delivers warnings on the calling thread and
  cites RUN-19.
- Cross-file (r2): model.md — line 299-300: Rust default corner bodies return
  `Error::Runtime` (`ErrorClass::RuntimeError`, `RLMESH_ERR_RUNTIME`,
  `rlmesh::RuntimeError`; Python's `NotImplementedError`, folded by R-21), not
  `Error::Type`.
- Cross-file (r2): running.md — `Recorder.export` (line 340 table, line 71
  Shape): Rust method is `export_` (R-4), and the row states its error class
  in every language: builtin `OSError` (Python `FileExistsError` for a
  refused target; `Error::Os`, `RLMESH_ERR_OS`, `rlmesh::OSError`).
- Cross-file (r2): running.md — line 409 "`on_run_end` fires exactly once,
  also on an error or a cancel": a token cancel; a dropped Rust `_async`
  future fires no hook (R-18).
- Cross-file (r2): core.md — COR-5 text list: add `OSError` (raw message),
  `cancelled` for `KeyboardInterrupt` natively, and "for every deadline
  (P-8)" to the Timeout text, matching R-22's table; line 89 error-class row:
  add `OSError` → `Error::Os(String)`, status 12.
- Cross-file (r2): remote.md — sandbox clients' infrastructure failures (Docker
  missing, pull failed, container exited) need a non-misuse class, e.g.
  `RLMeshException` or `ConnectionError`, with an `RMT-P<n>`, before sandbox
  parity lands (R-21 scope note); the client deadline produces
  `Error::Timeout` with R-22's text (P-8).
- Round 3, applied (all 23 findings; none rejected): R-16 / R-17 split the
  sync driver into `block_on`-free segments with hooks, env ops and warnings
  handled between them, and name the twin difference as the only one
  allowed (RUN-8, RUN-25); R-16 defines the async context by public probes
  (`task::try_id()` plus a library driving flag) with two tested deviations,
  since tokio's entered flag is private; R-16 records ENV-29's `serve_async`
  exception; R-17 states Session Drop inside and outside a runtime; R-18
  states token firing and re-arming (two findings), limits "the episode is
  not ended" to single calls, ends a cancelled run's episode, and replaces
  the drop claims with "no call starts after drop; in-flight results are
  discarded"; R-19 adds lent structs (RUN-29, ADP-20); R-9 adopts
  `Tolerant<E>` (ADP-4); R-12 fixes each C callback kind's class and adds
  `rlmesh_callback_set_error_class`; R-8 scalar setters take `T`; P-10 wraps
  trusted-entrypoint import failures in `AdapterResolutionError` (D-16); the
  C token is `RLMeshExtCancelToken`, with R-25's one exception for `cancel`
  fields and `cancel_handle`; P-8 lists all four texts; the RuntimeError row
  qualifies "contract without spaces" (RMT-12); R-5 (ADP-22), R-11 (ADP-21),
  R-21 authored constructors (COR-22), R-27 Env owners and D-7 / P-6 rung 1
  (COR-6) apply the r2 cross-file notes; R-6 exempts authored exception
  constructors instead of an R-7 row (R-7.8 is RMT-2's pending proposal); E-2
  leaves Rust `Default` values to E-3; R-3 callables row; R-25 prelude rule;
  D-6 narrowed to builders named after their type; index status.
- Cross-file (r3): running.md — RUN-29: cite R-19's "Lent structs" bullet
  instead of "R-19 covers only caller-supplied structs".
- Cross-file (r3): adapters.md — ADP-20: cite R-19's "Lent structs" bullet
  for the borrowed leaves' sizing rule.
- Cross-file (r3): running.md (lines 176, 192, 214), remote.md (lines 151,
  207, RMT-6), model.md (lines 367, 531, 670) — C token type
  `RLMeshCancelToken` → `RLMeshExtCancelToken`, `rlmesh_cancel_token_free` →
  `rlmesh_ext_cancel_token_free` (R-18, R-25); `rlmesh_<t>_cancel_handle`
  keeps its name.
- Cross-file (r3): remote.md — RMT-6: drop "(Cross-file (r2))"; R-18 now
  states firing and re-arming.
- Cross-file (r3): running.md — Session lifetime: Rust Drop inside an async
  runtime follows R-17 "Session Drop" (`on_episode_end` queued through the
  `Model`, a borrowed env not closed even with `close_env`); a cancelled run
  ends its open episode (R-18); C hook errors (line 247) take R-12's class
  (`RLMeshException` from `rlmesh_callback_set_error`, any R-21 class from
  `rlmesh_callback_set_error_class`); RUN-8 "block on the library runtime"
  means R-17's segment executor, not tokio's `block_on`.
- Cross-file (r3): model.md — MDL-7 / Lifetime: the serialized call queue is
  FIFO and accepts a `Session` Drop's detached `on_episode_end` (R-17);
  `close` waits for queued calls as for in-flight predicts; MDL-11 /
  `trust_entrypoints`: cite P-10 for Python's import-failure class.
- Cross-file (r3): adapters.md — `resolve` row / ADP-11: Python's
  trusted-entrypoint import failure becomes `AdapterResolutionError` (P-10).
- Cross-file (r3): env.md:138-144 — cite R-16's `serve_async` exception
  instead of saying R-16's rule "does not hold".
- Cross-file (r3): core.md — Errors table, C author cells: other classes are
  raised from a C callback with `rlmesh_callback_set_error_class` (R-12).
- Round 4, applied (all 14 findings; none rejected): R-16 makes
  `spawn_blocking` closures a third tested deviation (tokio runs them as
  tasks with an id, so `try_id` is set; no public API tells them apart) and
  requires library-run callbacks on plain threads so they may make sync
  calls; R-17 Session Drop and E-3 follow; R-16 says event handling between
  segments is outside any async execution context though the
  `Handle::enter` guard may be active; R-12 adds the fenced error payload
  channel (`rlmesh_ext_callback_set_error_payload`,
  `rlmesh_ext_last_error_take_payload`) that carries the C++
  `exception_ptr` with the error by value, and records ADP-27's
  `check_inverse` exception; X-10 maps caught C++ classes to both C setters;
  X-1 lists the `RLMESH_ERR_TRANSPORT` alias as additive; P-11 makes a
  missing Dict-space key at encode a `ValueError`, and R-21 narrows `KeyError`
  to `__getitem__` and widens the `OverflowError` note; R-9 exempts the class
  vocabularies' `name()` spelling and `RLMeshStatus`'s `RLMESH_OK = 0` (E-2
  rule 6 too); R-23 gives C's `rlmesh_warning_category_name` /
  `_is_subclass`; R-19 defines the minimum as the ABI-5 `sizeof` and refuses
  an out-of-range vtable slot with `RLMESH_ERR_TYPE`; R-10 adds the keyed
  mapping names (SPC-26); R-18 / R-25 / E-2 rule 4 allow the serve options
  `ext` field (ENV-12, MDL-20) as the second fenced member on shared types;
  R-27 adds `VariantOptions`; R-22 lists the native `LOAD_KEYS` and
  default-`load` texts.
- Not applied (r3 env.md cross-file, R-25 prelude half): `VariantOptions` is
  named in no authoring trait's signature (`Variant::new` is not a trait
  method), and the prelude rule says "the traits are the list", so it stays
  out of the prelude; only R-27 adds it.
- Cross-file (r4): model.md — Load: state the native default-`load` refusal
  text from R-22 (`model received construction params (a, b) but does not
override load(...) ...`, TypeError class); MDL-7 / Lifetime: the serialized
  call queue runs corners and hooks on plain library threads, not
  `spawn_blocking` (R-16 "Library-run callbacks"; today
  `rs/model/engine.rs:1125-1251`, `capi/model.rs:365-494`), and list that
  under current differences.
- Cross-file (r4): spaces.md — SPC-27: `KeyError` is only a `__getitem__`
  lookup; encoding a value missing a Dict-space key is the ValueError class
  (rules P-11); `OverflowError` also covers values above `u32` / `i64`
  (R-21 note). SPC-26: cite R-10's keyed-mapping note.
- Cross-file (r4): core.md — Warnings rows: C `rlmesh_warning_category_name`
  and `rlmesh_warning_category_is_subclass` (R-23); Errors table, C++ cell:
  the wrapper rethrows the author's exception via R-12's error payload
  channel; COR-5's text list: add R-22's two native load texts.
- Cross-file (r4): env.md — ENV-12: cite R-25's serve options `ext`
  exception (now stated in R-25 and R-18).
- Cross-file (r4): model.md — MDL-20: cite R-25's serve options `ext`
  exception.
- Round 5, applied (all 11 findings; none rejected): R-12's payload rides on
  `rlmesh::Error` between the capi adapter and the export (a C callback
  error is a facade `Error` there, `capi/model.rs:295-307,738`), as a hidden
  `Error::Attached` variant with `Arc`-shared, take-once payload, manual
  `PartialEq` / `Eq`, and dropped on rewrap or fold; X-1, R-25 and E-2 rule
  4 follow. R-12: the C++ wrapper with no payload throws the class of
  `rlmesh_last_error_class()` (statuses 1 and 3 cover two classes each); the
  pending callback error and payload are saved and restored per callback
  frame (`capi/abi/status.rs:83-88` clears before each callback). R-23 gives
  exact `Warning` / `WarningCategory` types in all three languages, the C
  handler typedef and setter with its status, and frees `user_data` only
  after the old handler's in-flight calls return. R-18 gives the exact
  token API (Rust `Clone` shares; C `_new` / `_clone` / `_cancel` /
  `_free`; C++ copy shares). R-19 lists `RLMeshModelVtable` among the ABI-4
  sized structs (`h:431-436`, `capi/model.rs:523-528`). X-10 defines the
  text for a non-`std::exception` throw. Shapes state which Rust `env`
  forms `&mut env` covers (running.md:145). R-25's prelude sentence moves
  `BinaryPayload` to `binding` (spaces.md:285). R-8 trait-object fields
  take their exact type.
- Cross-file (r5): spaces.md — the r3 request to fix R-25's
  `BinaryPayload` prelude sentence (spaces.md:550-554) is applied; mark it
  done.
- Cross-file (r5): running.md, remote.md — `cancel_handle()` returning
  `ext::CancelToken` by value relies on R-18's shared-handle token (now
  stated); C users who keep a token past its handle use
  `rlmesh_ext_cancel_token_clone` (R-18); no text change required beyond
  citing R-18 "Token API".
- Cross-file (r5): core.md — Warnings rows: cite R-23 "Types" and
  "Handler" for `Warning`, `WarningCategory` and `set_warning_handler`
  signatures; if core.md lists `Error`'s derives, note R-12's hidden
  `Attached` variant and manual `PartialEq` / `Eq`.
- Cross-file (r4): running.md — Session lifetime: Rust Drop on a
  `spawn_blocking` thread takes R-17's inside path (borrowed env not closed);
  RUN-8: a sync call from a `spawn_blocking` closure is refused (R-16).
- Round 6, applied (all 7 findings; none rejected): P-11 covers the
  batched encoder (`batch.rs:125-128`, per-lane `conform`, ValueError) and
  makes a decoded value missing a Dict-space key `ProtocolException`
  (`codec.rs:57-61`, `batch.rs:77-80`), so R-21's `KeyError` note holds.
  R-21 folds `ParamError` / `MissingParamError` / `UnknownParamError`
  (Experimental `ValueError` subclasses raised from Stable `Model` load and
  `EnvFactory.make`, `py/_authoring.py:345-357`) into the ValueError row;
  E-3 compares every folded subclass by `isinstance`. R-5 lists the
  `const char*` text outputs (vocabulary spellings, last-error getters).
  Status line now round 6. R-27 adds C++ `EnvFactoryBase` (env.md r5) and
  Running's `Reader`, `FrameSources`, `ReadItem`, `TelemetryRow` and
  recorder records. E-3 Cancellation adds ENV-32 / ENV-33 serving (env.md
  r5). R-10 gives exact C `_repr` / `_equals` forms, the status-form
  `_equals` over sized structs (ADP-29), the `str` ≠ `repr` split (ADP-28)
  and no C `_equals` / `_repr` on input leaves (ADP-6).
- Cross-file (r6): env.md — mark the r5 requests to rules.md (R-27
  `EnvFactoryBase`, E-3 ENV-32 / ENV-33) done.
- Cross-file (r6): adapters.md — mark the r5 R-10 requests (ADP-28 split,
  ADP-29 status form) done.
- Cross-file (r6): running.md — each C `_equals` over a sized struct takes
  R-10's status form `RLMeshStatus rlmesh_<t>_equals(const T*, const T*,
bool* out)`: `rlmesh_step_event_equals`, `rlmesh_episode_result_equals`,
  `rlmesh_workload_record_equals`, `rlmesh_episode_record_equals`,
  `rlmesh_media_ref_equals` (today `bool`, running.md:607-611);
  `rlmesh_frame_sources_equals` and other handle `_equals` stay `bool`.
- Cross-file (r6): spaces.md — SPC-27: a decoded value lacking a
  Dict-space key is `ProtocolException` and the batched encoder is covered
  (rules P-11).
- Cross-file (r6): remote.md, env.md, model.md — a received Dict value
  that lacks a key of its declared space is `ProtocolException` (a
  malformed message, rules P-11) in every language: state it where the
  native clients (observations) and servers (actions, observations) decode
  wire values.
- Cross-file (r6): model.md, env.md — `ParamError` raised by load
  (MDL-19) and by `EnvFactory.make` with `tag_params` is a folded subclass
  of the ValueError row (R-21); cite R-21's "Folded subclasses".

## Python changes

| ID   | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Compatibility                                                                                  |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| P-1  | `Error::Model` raises `ModelException` (D-10) with `code`, `is_recoverable` and text `[CODE] message`, instead of `RuntimeError("model error: …")` (`pyrs/types.rs:50`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Subclass of `RuntimeError`: catches keep working; text changes                                 |
| P-2  | `Internal` and server failures raise `RLMeshException` instead of bare `RuntimeError` (`pyrs/types.rs:51-54`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Subclass of `RuntimeError`                                                                     |
| P-3  | Protocol failures, including a failed handshake, raise `ProtocolException` (today `RuntimeError` via `Internal`; `rs/error.rs:205-209,248`). Transport failures stay `ConnectionError`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Subclass of `RuntimeError`                                                                     |
| P-4  | Env codes `TIMEOUT`, `INVALID_ACTION`, `CANCELLED` raise `(Recoverable)EnvironmentException` (D-11), not `TimeoutError` / `ValueError` / `RuntimeError` (`pyrs/types.rs:70-77`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Breaking (D-11); changelog entry                                                               |
| P-5  | The wire `UNSUPPORTED` env code reaches Python as `UNSUPPORTED` instead of `INTERNAL` (`rs/error.rs:179-191`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Value of `code` changes                                                                        |
| P-6  | The edition resolver (COR-6's rungs: the call's own value, including `ServeOptions.workflow_edition` on `EnvServer` and `Model.serve` (COR-P2); `RLMESH_WORKFLOW_EDITION`; the served peer's option; the type's declaration) moves into the Rust facade; Python layers its `pyproject.toml` rung after the type's declaration (D-7) and keeps its warning category and caller attribution (R-23 Thread).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | None intended                                                                                  |
| P-7  | `sanitize_metadata` is relabelled Bridge (D-9).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Label only                                                                                     |
| P-8  | Every timeout has the one R-22 text `operation timed out after {:?}`. The Python clients' own texts go, as the deadlines move into the facade clients (Remote): `remote environment call timed out after {:.3}s` (`pyrs/client.rs:243-246`), `remote environment connect timed out after {:.3}s` (`pyrs/client.rs:726-730`), `remote model connect timed out after {:.3}s` (`pyrs/model.rs:1038-1041`) and `remote model request timed out after {:.3}s` (`pyrs/model.rs:1127-1130`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Text of `TimeoutError` changes                                                                 |
| P-9  | The Python env server treats a raised exception as recoverable only if it is an `EnvironmentException` with a truthy `is_recoverable` (`RecoverableEnvironmentException`, or, after P-4, one re-raised from a recoverable remote env), not any object with that attribute (`pyrs/server/py_environment.rs:418-429`), so a `ModelException(is_recoverable=True)` raised in an env is not recoverable (R-21).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Narrower; only affects non-env exceptions carrying the flag                                    |
| P-10 | `resolve(..., trust_entrypoints=True)` (and `Model`, `run`, `session` resolving through it) raises `AdapterResolutionError` when a custom-input entrypoint cannot be imported or its attribute path does not resolve, chained `from` the original `ImportError` / `AttributeError`, instead of letting those escape (`py/adapters/resolver.py:459-468`, `py/_entrypoint.py:33-41`). Natively the same case is an unregistered entrypoint, already `AdapterResolutionError` (ADP-11, MDL-11). The malformed-string `ValueError` and non-callable `TypeError` stay (R-21 rows).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Breaking for `except ImportError` / `except AttributeError` around resolution; changelog entry |
| P-11 | Encoding a dict that lacks a key of its Dict space (a `step` action, a reset or step observation, any value checked against a Dict space) raises `ValueError` with the structural-conformance text native `conform` produces (`missing key '<k>'` at the value's path, `crates/rlmesh-spaces/src/spaces/composite/dict.rs` `conform`), as every other structural deviation already does, instead of `KeyError("missing dict key '<k>'")` from the unchecked encoder (`pyrs/spaces/value_codec/codec.rs:138-153,209-211`). The batched encoder `py_any_to_batched_space_values_with_backend` (`batch.rs:125-128`; `RemoteVectorEnv.step` actions `pyrs/client.rs:626`, batched model frames `pyrs/model.rs:277,344`, served vector-env observations `pyrs/server/py_environment.rs:830,955`) runs the same `conform` on each lane and raises the same `ValueError`; a call site that already rewraps the encoder's error keeps its rewrap (the batched-predict `TypeError`, `pyrs/model.rs:284-290`). Decoding a received value that lacks a key of its Dict space (single `codec.rs:57-61`, batched `dict_column` `batch.rs:77-80`; every caller decodes a wire value) raises `ProtocolException`, a malformed message (R-21), instead of `KeyError("missing RLMesh dict key '<k>'")`. `KeyError` stays only for `__getitem__` lookups (R-21, SPC-27). | Breaking for `except KeyError` around encoding and decoding; changelog entry                   |

Builtin `RuntimeError` for misuse (and `NotImplementedError`), `OSError`
(and `FileExistsError`), `ValueError`, `TypeError`, `ConnectionError`,
`TimeoutError` (class; text by P-8) and `KeyboardInterrupt` are unchanged.

## Cross-cutting changes

Each moves a native language onto the rules. Areas carry the details.

- **X-1 C ABI 5.** The value-type rebase, `RLMeshDType` struct → enum,
  `RLMeshTensor` → opaque, callbacks returning `RLMeshStatus`, new
  `free_user_data` fields, `struct_size` on `RLMeshServeOptions` and
  `RLMeshRunOptions`, enums with `INVALID = 0`, and the removed
  `rlmesh_abi_version_major/minor/patch` (`h:44-52`) break the ABI under the
  header's own rule (`h:30-35`). Bump `RLMESH_ABI_VERSION` 4 → 5 and the
  SONAME in one release. Additive in the same release: `RLMESH_ERR_CONNECTION`
  with `RLMESH_ERR_TRANSPORT` kept as a deprecated alias of the same value
  (R-21), the new statuses 9-12, and the error payload channel (R-12). On
  the Rust side the channel adds the hidden `Error::Attached` variant
  (additive under `#[non_exhaustive]`) and replaces `Error`'s derived
  `PartialEq` / `Eq` with manual impls of the same meaning; `Clone` and
  `Debug` stay derived (R-12).
- **X-2 One value type** (R-27). C `RLMeshValue` is a `SpaceValue` projection
  today (`capi/env.rs:432`); it re-bases on core's `Value` tree. Core owns it.
- **X-3 Typed metadata.** Info, reset options, tags and variants stop crossing
  C and C++ as JSON (`options_json`, `info_json`, `adapter_tags_json`;
  `h:523-552,718-732`; `hpp:1259-1328`).
- **X-4 No sentinels** (R-13). 0 means unset today for `max_episodes`,
  `max_episode_steps`, `max_episode_seconds`, `execution_horizon` and the serve
  timeouts (`h:450-458,483-490`; `hpp:891-922`).
- **X-5 C enums reserve 0** (R-9).
- **X-6 Vtable constructors** take `(vtable, user_data, options, out)` (R-20);
  today `rlmesh_env_new` takes the config before `user_data`.
- **X-7 A run borrows the model.** `RunLocalOptions::close_model` defaults to
  true (`rs/model/worker.rs:72,109`), `run_local` consumes the worker
  (`:275`), and C and C++ inherit it. Target: close only on `close()`,
  Drop/free, or serve stop; `close()` is idempotent, waits for in-flight
  predicts and refuses later runs.
- **X-8 One cancellation mechanism** (R-18). `rlmesh_model_cancel` is terminal
  today (`h:503-512`) and remote clients have no interrupt.
- **X-9 Fencing** (R-25).
- **X-10 C++17 with exceptions, tested also at C++20** (D-5). Callbacks throw; a `noexcept`
  trampoline catches and maps the caught C++ class to
  `rlmesh_callback_set_error` (the env and model classes, with the
  recoverable flag) or `rlmesh_callback_set_error_class` (every other R-21
  class, for example a thrown `rlmesh::TypeError` from a hook). An exception
  outside the R-21 tree records the callback kind's class: a
  `std::exception` with its `what()`, any other thrown type (`throw 42`,
  caught by `catch (...)`) with the text `unknown C++ exception`. Every
  case attaches the `std::exception_ptr` with
  `rlmesh_ext_callback_set_error_payload`; the wrapper rethrows it from
  `rlmesh_ext_last_error_take_payload` (R-12).

## Shapes

The same task in each language: wrap a policy, run two episodes with two
seeds, then drive one episode by hand. These are targets and agree with the
area tables; `env` is any env input [Running](running.md) accepts (a
`RemoteEnv`, a local env, an address). In the Rust block `env` is a local
`Env` or a `RemoteEnv` (`&mut env` converts through `EnvTarget`'s `From<&'a
mut E>` / `From<&'a mut RemoteEnv>`); for an address pass `"unix:///…"` or
`addr.as_str()`, for an `EnvServer` `&server`, and for a vector env or
factory `EnvTarget::vector(&mut v)` / `EnvTarget::factory(&mut f)`.

```python
class Greedy:
    def predict(self, observation):
        return 0

model = rlmesh.Model(Greedy(), spec=spec)
result = model.run(env, episodes=2, seeds=[0, 1])
with rlmesh.session(model, env) as s:
    obs, info = s.reset(seed=0)
    obs, reward, terminated, truncated, info = s.step(s.predict(obs))
```

```rust
struct Greedy;

impl rlmesh::Policy for Greedy {
    fn predict(&mut self, observation: Value) -> rlmesh::Result<Value> {
        Ok(Value::Int(0))
    }
}

let model = Model::new(Greedy, ModelOptions { spec: Some(spec.into()), ..Default::default() })?;
let result = model.run(
    &mut env,
    RunOptions { episodes: Some(2), seeds: Some(vec![0, 1]), ..Default::default() },
)?;
let mut s = rlmesh::session(&model, &mut env, SessionOptions::default())?;
let (obs, info) = s.reset(Some(0), None)?;
let action = s.predict(obs)?;
let (obs, reward, terminated, truncated, info) = s.step(action)?;
```

```cpp
struct Greedy {
  rlmesh::Value predict(const rlmesh::Value& observation) { return rlmesh::Value(0); }
};

auto model = rlmesh::Model(Greedy{}, {.spec = spec});
auto result = model.run(env, {.episodes = 2, .seeds = {0, 1}});
auto s = rlmesh::session(model, env);
auto [obs, info] = s.reset(0);
auto [obs2, reward, terminated, truncated, info2] = s.step(s.predict(obs));
```

```c
RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
vt.predict = greedy_predict;
RLMeshModelOptions mo = RLMESH_MODEL_OPTIONS_INIT;
mo.spec = (RLMeshSpecArg){RLMESH_SPEC_KIND_MODEL_SPEC, spec};
RLMeshModel* model = NULL;
rlmesh_model_new(&vt, NULL, &mo, &model);

const int64_t seeds[] = {0, 1};
RLMeshRunOptions ro = RLMESH_RUN_OPTIONS_INIT;
ro.has_episodes = true;
ro.episodes = 2;
ro.seeds = seeds;
ro.seeds_len = 2;
RLMeshRunResult* result = NULL;
/* env_target: an RLMeshEnvTarget (Running) */
rlmesh_model_run(model, env_target, &ro, &result);

RLMeshSession* s = NULL;
rlmesh_session_new(rlmesh_model_target(model), env_target, NULL, &s);
int64_t seed = 0;
RLMeshValue* obs = NULL;
RLMeshValue* info = NULL;
rlmesh_session_reset(s, &seed, NULL, &obs, &info);
```

- **Model.** Rust has a `Policy` trait and a concrete `Model`; the plain trait
  impl is the full path and any attribute macro is sugar (R-12). C has a
  vtable plus an `RLMeshModel*`; C++ detects members. All share Python's four
  inference corners and fallbacks, `on_episode_end` and `close`
  ([Model](model.md)).
- **Env.** Gymnasium-shaped everywhere (R-15); `EnvFactory` and `Variant`
  exist in every language ([Env](env.md)).
- **Results.** `RunResult`, `EpisodeResult`, `RunHooks`, `StepEvent` with
  Python's field names and units ([Running](running.md)).
- **Metadata** is typed, never JSON text (X-3).

## Known gaps

The October 2026 audit found these gaps before implementation. All are
closed: Python and Rust by the parity waves, C by ABI 5 and C++ by the C++17
header. `✓` closed (was a gap or partial), `—` matched already or does not
apply. File references describe the code as audited.

| ID   | Gap                                                                                                                                                                                                                                                                                                                                      | Python      | Rust | C   | C++ |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---- | --- | --- |
| G-1  | Episode budget is "until the env ends" (`max_episodes` unset), not Python's 1 or `len(seeds)` (`py/_models/_eval.py:178-195`)                                                                                                                                                                                                            | —           | ✓    | ✓   | ✓   |
| G-2  | A run closes the model (`close_model = true`, `rs/model/worker.rs:72,109`)                                                                                                                                                                                                                                                               | —           | ✓    | ✓   | ✓   |
| G-3  | Live episodes evicted at 4,096 (`capi/model.rs:73`); Python refuses past 65,536 (`py/_models/_episodes.py:25,110`)                                                                                                                                                                                                                       | —           | —    | ✓   | ✓   |
| G-4  | `episodes` / `seeds` named `max_episodes` / `episode_seeds` (`rs/model/worker.rs:47,54`; `h:450,457`)                                                                                                                                                                                                                                    | —           | ✓    | ✓   | ✓   |
| G-5  | Extra `seeded` / `base_seed` (`h:451-452`; `hpp:892-894`)                                                                                                                                                                                                                                                                                | —           | —    | ✓   | ✓   |
| G-6  | Model authoring is `ModelHandler` / `ModelWorker` / batch callback, not `Policy` + `Model`                                                                                                                                                                                                                                               | —           | ✓    | ✓   | ✓   |
| G-7  | No `Session`                                                                                                                                                                                                                                                                                                                             | —           | ✓    | ✓   | ✓   |
| G-8  | No remote clients                                                                                                                                                                                                                                                                                                                        | —           | —    | ✓   | ✓   |
| G-9  | Remote clients are async-only (`rs/env/client.rs:59-368`); no blocking primary                                                                                                                                                                                                                                                           | —           | ✓    | —   | —   |
| G-10 | Blocking calls panic inside a runtime (`rs/model/worker.rs:270-277`)                                                                                                                                                                                                                                                                     | —           | ✓    | —   | —   |
| G-11 | Env authoring uses request/result structs, not Gymnasium shape                                                                                                                                                                                                                                                                           | —           | ✓    | —   | —   |
| G-12 | Results are `RuntimeReport` or aggregates, not `RunResult` / `EpisodeResult`                                                                                                                                                                                                                                                             | —           | ✓    | ✓   | ✓   |
| G-13 | No `ModelSpec` or transforms; C has only resolve/describe of a plan (`h:334-359`)                                                                                                                                                                                                                                                        | —           | ✓    | ✓   | ✓   |
| G-14 | No `EnvFactory` / `Variant`                                                                                                                                                                                                                                                                                                              | —           | ✓    | ✓   | ✓   |
| G-15 | Spaces lack `sample` / `seed` on the shared surface (the sampler is internal, `crates/rlmesh-spaces/src/sample.rs`)                                                                                                                                                                                                                      | —           | ✓    | ✓   | ✓   |
| G-16 | Box bounds are `double` only (`h:272,298-302`); Rust carries typed bounds                                                                                                                                                                                                                                                                | —           | —    | ✓   | ✓   |
| G-17 | Two value types (`SpaceValue` vs the adapter `Value`)                                                                                                                                                                                                                                                                                    | —           | ✓    | ✓   | ✓   |
| G-18 | Metadata passed as JSON strings (X-3)                                                                                                                                                                                                                                                                                                    | —           | —    | ✓   | ✓   |
| G-19 | Editions: Rust has `workflow_edition` on serve and run (`rs/model/worker.rs:93`) and `selected_workflow_edition` on `RemoteVectorEnv` and `RemoteModel` (`rs/env/client.rs:139`, `rs/model/remote.rs:279`), but not on `RemoteEnv`, and no `current_workflow_edition()` or resolver chain; C and C++ declare only when serving (`h:492`) | —           | ✓    | ✓   | ✓   |
| G-20 | Error classes collapsed (R-21): no Protocol, Type, Runtime, Os; Server maps to `TRANSPORT` (`capi/abi/status.rs:59`)                                                                                                                                                                                                                     | ✓ (P-1–P-5) | ✓    | ✓   | ✓   |
| G-21 | No warning channel (R-23)                                                                                                                                                                                                                                                                                                                | —           | ✓    | ✓   | ✓   |
| G-22 | Cancellation: Rust re-exports `CancellationToken`; C `rlmesh_model_cancel` is terminal; no client interrupt                                                                                                                                                                                                                              | —           | ✓    | ✓   | ✓   |
| G-23 | `struct_size` missing on `RLMeshServeOptions` / `RLMeshRunOptions`, no minimum check anywhere (`capi/env.rs:575`, `capi/model.rs:527` refuse only 0)                                                                                                                                                                                     | —           | —    | ✓   | —   |
| G-24 | 0-as-unset sentinels (X-4)                                                                                                                                                                                                                                                                                                               | —           | —    | ✓   | ✓   |
| G-25 | C input enums use 0 as a real value (X-5)                                                                                                                                                                                                                                                                                                | —           | —    | ✓   | —   |
| G-26 | C++17, `Result<T>`, `-fno-exceptions` (`hpp:1-19`)                                                                                                                                                                                                                                                                                       | —           | —    | —   | ✓   |
| G-27 | Binding plumbing at the crate root and in the prelude (R-25)                                                                                                                                                                                                                                                                             | —           | ✓    | —   | —   |
| G-28 | `RLMESH_ABI_VERSION_MAJOR/MINOR/PATCH` carry the package semver (`h:44-52`); `RLMeshContract` not `RLMeshEnvContract`                                                                                                                                                                                                                    | —           | —    | ✓   | ✓   |

## Enforcement

**PLANNED. None of this exists yet.** Today the repo has the Python surface
snapshot and its tests (`python/rlmesh/tests/api_surface/`) and extractors for
the native surfaces used by the docs
(`tools/rlmesh_api_surface/src/rlmesh_api_surface/rust_surface.py` from
rustdoc JSON, `header_surface.py` from libclang). No Rust, C or C++ snapshot
is committed, and nothing compares the languages.

### E-1 Mapping manifest

`docs/specs/language-parity/manifest.toml`, one table per Stable Python
symbol, keyed by its snapshot path. Members and parameters nest under it:

```toml
["rlmesh:Session.reset"]
area = "running"
rule = ["R-7.1"]

["rlmesh:Session.reset".rust]
status = "present"              # present | missing | bridge | exempt
path = "rlmesh::Session::reset"
also = ["rlmesh::Session::reset_async"]
params = { seed = "seed", trial_index = "trial_index" }

["rlmesh:Session.reset".c]
status = "missing"
path = "rlmesh_session_reset"
gap = "G-7"

["rlmesh:Session.reset".cpp]
status = "missing"
path = "rlmesh::Session::reset"
gap = "G-7"
```

- `status = "bridge"` cites R-24; `"exempt"` cites a decision (D-8, D-9).
- `"missing"` requires a `gap` ID.
- Options structs map each Python keyword to its field, with the default
  written once as a Python literal (`default = "1"`).
- Rust-only and C-only exports are listed under `[ext]` and `[binding]` with
  their rule.

### E-2 Check command

`mise run parity:check`: loads the Python snapshot, extracts the Rust surface
(rustdoc JSON) and the two headers (libclang), and checks them against the
manifest. It runs in `mise run check` and in CI.

Acceptance criteria. The check fails when:

1. a Stable Python symbol, member, parameter or field has no manifest entry;
2. a `present` entry names a native item that does not exist, or a field or
   parameter whose name differs from Python's after R-4;
   The parameter-name check excludes only native parameters with no Python
   counterpart: the receiver/handle corresponding to `self`; C `out` and
   `out_*` parameters (R-15); pointer length/size companions (`*_len`, or
   `len`/`count` paired with a pointer, R-5/R-20); `struct_size` (R-19);
   callback state pointers named `user_data` or `ctx` (R-20); and the
   options-struct parameter itself (R-6). These are structural rules in the
   checker, not per-entry exemptions. A free callback names its parameter after
   the object it frees (`free_value(value)`, `free_payload(payload)`); only a
   callback that frees state uses `user_data` or `ctx`.
   Other positional parameter names must
   match Python after R-4. A deliberate translation stated by a D-/area
   decision must cite that decision in the manifest.
3. a `missing` entry's native item exists (stale gap);
4. a native public item is neither in the manifest nor under `ext` /
   `binding` / `rlmesh_ext_` (R-25), R-18's `cancel` fields and
   `cancel_handle` members and the serve options `ext` fields
   (`EnvServerOptions.ext`, `ModelServeOptions.ext`) being listed under
   `[ext]` and R-12's hidden `Error::Attached` variant under `[binding]`;
5. a C options struct lacks `struct_size` first, its `_INIT` macro, or its
   `_MIN_SIZE` (R-19); an `_INIT` macro's literal or a C++ default member
   initializer (both read by libclang) differs from the manifest's Python
   default. Rust `Default` values are not in rustdoc JSON; E-3's Defaults
   family checks them;
6. a C input enum's value 0 is not `*_INVALID` (`RLMeshStatus`, whose 0 is
   `RLMESH_OK`, is exempt, R-9), or a C enum has an
   `*_OTHER` value that R-9 / ADP-4 do not list as tolerant (R-9);
7. an `ErrorClass` / `RLMeshErrorClass` / C++ exception class is missing or
   has a different base than R-21.

### E-3 Behavioural fixtures

Snapshots check names and literals, not behaviour. Shared fixtures in
`tests/parity/fixtures/*.toml` describe scenarios once; a runner per language
(pytest, a Rust integration test, a C program and a C++ program built by the
capi tests) executes each against the same in-repo scripted env and model and
compares results. `mise run parity:fixtures` runs all four; it joins
`test:ci`.

Required scenario families:

- **Defaults:** a run with no options executes one episode; `seeds=[0, 1]`
  alone runs two; `episodes=2, seeds=[0]` raises the ValueError class in every
  language; each options struct's defaults equal Python's, with a Rust test
  generated from the manifest comparing every `Default::default()` field to
  the manifest's Python default (E-2 cannot read them).
- **Borrowing:** the model works after `run` and is not closed; the env is
  closed only with `close_env`; a second session on a busy env fails with
  Python's class, including a run on a remote handle already driven with
  `reset` / `step` (R-17); freeing a C session leaves borrowed handles valid;
  dropping a Rust session with an open episode fires `on_episode_end` once,
  and closes a borrowed env with `close_env` outside a runtime but not inside
  one (R-17 Session Drop).
- **Cancellation:** a cancel during a single `step` fails it with the
  cancellation class, leaves the episode open until the next `reset` and
  allows another call; a cancel during `Session.run` fires `on_run_end` with
  the partial result and ends the open episode (`on_episode_end` once); one
  `cancel()` fails exactly one call, a `cancel()` while idle fails the next
  call at its start, and the token is armed again afterwards; dropping a Rust
  `_async` future fires no run hook, no env or model call starts after the
  drop (the scripted env and model count call starts), and the handle is
  usable afterwards; only R-17's owned cleanup (a self-dialed connection's
  close, Session Drop) runs. Serving (ENV-32, ENV-33): dropping a polled
  `EnvServer::serve_async` stops the server, closes the env with
  `close_env_on_shutdown` and does not block the dropping task; dropping
  `wait_async` leaves the server running; dropping a polled
  `EnvFactoryExt::serve_async` closes the envs it made; a factory serve
  whose `cancel` fires drains, closes the envs and fails with the
  cancellation class, and one given an already-fired token runs no
  `prepare`.
- **Sync and async (Rust):** a sync call from a tokio task, from a
  `block_in_place` section, from a `spawn_blocking` closure and from a hook
  under an `_async` twin fails with `Error::Runtime` naming the twin; from a
  `std::thread::spawn` thread, from a thread inside `Handle::enter` and from
  a caller's own `block_on` root future it runs; a hook under the sync call
  and a model callback the library runs (a policy whose `predict` makes a
  sync `RemoteModel` call) may make sync calls; dropping a session with
  `close_env` inside a `spawn_blocking` closure leaves the borrowed env open;
  nothing panics (R-16, R-17).
- **Errors:** each R-21 row is produced once (scripted failures for env, model,
  protocol, transport, timeout, misuse, type, filesystem), checking class,
  `is_subclass_of`, `code`, `is_recoverable` and text (R-22); the folded
  subclasses compare by `isinstance` (R-21); a C hook that sets
  `rlmesh_callback_set_error_class(RLMESH_ERROR_CLASS_TYPE_ERROR, …)` fails
  the run with the TypeError class, as a Python hook raising `TypeError`
  does (R-12); a C++ hook or policy throwing its own exception type derived
  from an R-21 class is rethrown as that same object at the failing call, and
  with one `Model` serving two sessions on two threads, each failing call
  rethrows its own session's exception (R-12 error payload channel).
- **Warnings:** each category is delivered with its name, on the calling
  thread; a failing handler aborts the call, except RUN-19's recorder
  warnings; Python under `-W error` raises the warning attributed to the
  caller's frame.
- **C sizing:** `= {0}` refused; a struct of `_MIN_SIZE` takes defaults for
  later fields; a larger struct with zero tail is accepted and with a nonzero
  tail refused; an out-struct keeps unknown fields at `_INIT`; a lent struct
  (a callback's `RLMeshStepEvent`, a borrowed spec leaf) reports the
  library's `sizeof`, and a reader built with a larger header reads the
  uncovered fields as `_INIT` (R-19).
