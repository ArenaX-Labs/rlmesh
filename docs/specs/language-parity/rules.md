# Language parity: rules

Part of the [language parity spec](../language-parity.md). Status: Draft —
reviewed round 6. C#: Reviewed (Opus and GPT rounds 1–3), 2026-10-19; a
design record of the v1 surface, of which v0.2 implements only the
foundation and R-28's minimal surface (D-45), pending user review (D-17 to
D-45, R-28, X-11 to X-16, G-29).

These rules hold in every area file. An area that disagrees with a rule is
wrong; fix the area, or change the rule here. IDs are stable: `R-n` rules,
`D-n` decisions, `P-n` Python changes, `X-n` cross-cutting changes, `G-n`
gaps, `E-n` enforcement; area files use `<AREA>-n` decisions and
`<AREA>-P<n>` Python changes (`COR`, `SPC`, `ENV`, `MDL`, `RUN`, `RMT`, `ADP`).
Never renumber; retire an ID by marking it removed.

Paths: `py/` = `python/rlmesh/src/rlmesh/`, `pyrs/` =
`python/rlmesh/rust/src/`, `rs/` = `crates/rlmesh/src/`, `capi/` =
`crates/rlmesh-capi/src/`, `h` = `crates/rlmesh-capi/include/rlmesh.h`, `hpp`
= `crates/rlmesh-capi/include/rlmesh.hpp`, `cs/` = `csharp/src/RLMesh/` (the
hand-written C# layer), `csi` = `csharp/src/RLMesh/Interop/Native.g.cs` (the
generated raw P/Invoke layer, D-18).

## Rules

### R-1 One library

Every Stable Python symbol (`python/rlmesh/tests/api_surface/snapshots/api_surface.json`,
labels in [compatibility](../../compatibility.md)) has a counterpart in Rust,
C, C++ and C# with the same name (cased by R-4), the same parameters and fields
under the same names, the same defaults and the same behaviour. A difference
that no rule explains is a bug.

### R-2 Python is the reference

Python's Stable surface defines the nouns, verbs, defaults and validation. The
other languages do not grow their own model. When the target changes Python's
behaviour, the change is listed explicitly (P-n below, or the area's "Python
changes" list); it is never implied by a table cell.

### R-3 Translation table

These differences are expected. Each applies the same way to every symbol.

| Concern                        | Python                            | Rust                                                                                                   | C++                                                        | C                                                   | C#                                                                                                |
| ------------------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Names                          | `snake_case`, `PascalCase`        | same (R-4)                                                                                             | same (R-4)                                                 | D-6                                                 | .NET casing (R-4, D-20)                                                                           |
| Modules                        | `rlmesh.spaces`                   | `rlmesh::spaces`                                                                                       | `rlmesh::spaces`                                           | D-6                                                 | `RLMesh.Spaces` namespace; free functions on a static class (D-21)                                |
| Errors                         | exceptions                        | `Result<T, Error>`                                                                                     | exceptions                                                 | `RLMeshStatus` + thread-local last error            | exceptions                                                                                        |
| Error classes                  | exception classes                 | `Error` variant + `ErrorClass` (R-21)                                                                  | classes with Python's inheritance (R-21)                   | status + `RLMeshErrorClass` (R-21)                  | classes with Python's inheritance under .NET bases (R-21, D-25)                                   |
| Warnings                       | `warnings.warn(Category)`         | handler (R-23)                                                                                         | handler (R-23)                                             | handler (R-23)                                      | handler (R-23)                                                                                    |
| Lifetime                       | context manager, `close()`        | `Drop`, `close()`                                                                                      | RAII, `close()`                                            | `rlmesh_<t>_close`, `rlmesh_<t>_free`               | `IDisposable` (`using`), `Close()`; `SafeHandle` finalizer as a safety net (R-11, R-28)           |
| Keyword arguments              | keywords with defaults            | options struct, `Default` (R-6)                                                                        | aggregate, `with_<field>` setters; C++20 designators (R-6) | options struct, `RLMESH_<NAME>_INIT` (R-6, R-19)    | options class, settable properties initialized to Python's defaults; `options = null` (R-6, D-22) |
| Optional keywords, short calls | keywords                          | positional `Option` (R-7, closed list)                                                                 | default arguments (R-7)                                    | pointer, NULL = default (R-7)                       | optional parameters, `null` = Python's default (R-7, D-23)                                        |
| `*args` / `**kwargs`           | variadic                          | slice or `Vec` / `Kwargs` field                                                                        | `std::vector` / `Kwargs` field                             | pointer + count / `RLMeshValue*` dict               | `params T[]` / `Metadata` property (D-30)                                                         |
| Optional value                 | `None`                            | `Option<T>`                                                                                            | `std::optional<T>`                                         | pointer (NULL = None) or `has_<name>` (R-13)        | `T?`, nullable reference types enabled (R-13)                                                     |
| Optional list                  | `None` or list                    | `Option<Vec<T>>`                                                                                       | `rlmesh::OptionalList<T>`                                  | pointer + count; NULL = None, count 0 = `[]` (R-13) | `IReadOnlyList<T>?`; `null` = None, empty = `[]` (R-13)                                           |
| Durations                      | `*_seconds: float`                | `Duration`, `_seconds` dropped                                                                         | `std::chrono::duration<double>`, `_seconds` dropped        | `double *_seconds`                                  | `TimeSpan`, `_seconds` dropped (R-14, D-35)                                                       |
| Integers                       | `int`                             | `i64`; `u64` where Python refuses negatives (R-5)                                                      | `int64_t` / `uint64_t`                                     | `int64_t` / `uint64_t`                              | `long`; `ulong` where Python refuses negatives (R-5, D-34)                                        |
| Tuples returned                | tuple                             | tuple, or a struct named for the fields                                                                | `std::tuple` (structured bindings)                         | out-parameters in tuple order                       | `readonly struct` named for the fields with `Deconstruct`, or a named `ValueTuple` (R-15, D-29)   |
| Closed string vocabularies     | `str` from a fixed set            | enum, `as_str()` / `Display` = Python spelling (R-9)                                                   | `enum class`, PascalCase, `to_string()` (R-9)              | enum, `INVALID = 0`, `rlmesh_<enum>_str()` (R-9)    | `enum`, PascalCase, C's numbers; `AsString()` = Python spelling (R-9)                             |
| Dunder methods                 | `__repr__`, `__eq__`, …           | traits (R-10)                                                                                          | operators (R-10)                                           | named functions (R-10)                              | overrides and BCL interfaces (R-10, D-26)                                                         |
| Handles                        | object reference                  | move-only, or `Clone` shares (R-11)                                                                    | move-only, or copy shares (R-11)                           | `rlmesh_<t>_clone` returns a new reference (R-11)   | sealed class over a `SafeHandle`; a reference shares (R-11)                                       |
| Frozen dataclasses             | `@dataclass(frozen=True)`         | value struct, `Clone` copies (R-11)                                                                    | value struct (R-11)                                        | value handle or struct (R-11)                       | `sealed record`, init-only properties (R-11, D-28)                                                |
| Protocols, subclass hooks      | `Protocol`, overridable methods   | trait (R-12)                                                                                           | class detected by a templated adapter (R-12)               | vtable, NULL slot = not defined (R-12)              | interface plus optional one-member interfaces, detected with `is` (R-12, D-32)                    |
| Callables                      | callable                          | `impl FnMut(..) + Send + 'static`; `Fn + Send + Sync` where called concurrently or process-wide (R-23) | any callable                                               | function pointer + `user_data` + `free_user_data`   | delegate (`Func<..>`, `Action<..>`) or a one-method interface; called from library threads (R-28) |
| Blocking and async             | sync                              | sync; `*_async` twins opt in (R-16)                                                                    | sync                                                       | sync                                                | sync; `<Name>Async` twins return `Task` (R-16, D-24)                                              |
| Cancellation                   | `KeyboardInterrupt` (signal poll) | `ext::CancelToken`, future drop (R-18)                                                                 | `ext::CancelToken` (R-18)                                  | `RLMeshExtCancelToken` (R-18)                       | `Ext.CancelToken`; twins take a `CancellationToken` (R-18)                                        |
| Language-only extras           | —                                 | `rlmesh::ext`, `rlmesh::binding` (R-25)                                                                | `rlmesh::ext` (R-25)                                       | `rlmesh_ext_` / `RLMeshExt` (R-25)                  | `RLMesh.Ext` (R-25, D-33)                                                                         |

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
- **C#** (D-20) applies one fixed transform to Python's name, so E-2 stays a
  single mapping per name: types, interfaces' members, methods, properties,
  enum members and constants are `PascalCase` (`max_episode_steps` →
  `MaxEpisodeSteps`, `NO_ADAPTER` → `NoAdapter`, `DESCRIBE_METADATA_KEY` →
  `DescribeMetadataKey`); parameters are `camelCase` (`trial_index` →
  `trialIndex`); a protocol or capability interface is `I` + `PascalCase`
  (R-12). The `_seconds` suffix is dropped where the type is `TimeSpan`, as in
  Rust and C++. R-4's trailing `_` is not used: a `PascalCase` member is never
  a C# keyword (`Default`, `Type`, `Export`, `Params`, `Base`), and a
  `camelCase` parameter that is one is written as a verbatim identifier
  (`@params`, `@event`, `@base`, `@default`), whose metadata name is the plain
  word. A member whose `PascalCase` name equals its declaring type's name
  (CS0542) or an inherited `object` member (`Equals`, `GetHashCode`,
  `GetType`, `ToString`) takes the suffix `Value` (`Foo.Foo` → `Foo.FooValue`);
  the Stable surface has no such case today, and an area that meets one cites
  this bullet.
- C# member kinds: a Python attribute or property is a C# property and a
  Python method a C# method (`.shape` → `Shape`, `.copy()` → `Copy()`),
  except that a Python property whose read connects or can fail on I/O is a
  method with an `Async` twin (`Session.Contract()`,
  `Session.SelectedWorkflowEdition()`; Running RUN-33), because a .NET
  property is expected to be cheap and cannot have a twin, and except that a
  Python property whose value borrows native memory is a method returning
  an owning lease (`Tensor.buffer` → `Tensor.Buffer()`, Spaces SPC-31),
  because a .NET property should not hand out a resource the caller must
  dispose. The transform is mechanical: `tobytes` is `Tobytes()`, not
  .NET's `ToBytes()`.

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
- **C#** (D-34, D-36): `i64` → `long`, `u64` → `ulong`, `u32` → `uint`,
  `u16` → `ushort` (C's exact width, so E-2 compares one type); `usize`
  lengths, counts and positions → `int` (`Count`, indexers, `Span<T>`
  lengths), converted with a checked cast, because every BCL collection
  counts in `int`. The assembly is `[CLSCompliant(false)]`. `float` →
  `double`, `bool` → `bool` (the raw layer passes C `bool` as `byte`, R-28).
  `str` → `string`, marshalled by the hand layer to UTF-8; a C# string may
  hold U+0000, so a `string` bound for a NUL-terminated C input that contains
  one is refused with the ValueError class before the C call (D-36; the C++
  wrapper, which passes `c_str()` and truncates today, makes the same
  refusal in the same release), while text data that Python lets contain
  NUL travels as Spaces' length-bearing view. Unpaired UTF-16 surrogates are
  refused with the ValueError class, as invalid UTF-8 is in C. These
  refusals apply to arguments; text the library builds itself from an
  exception it reports to C is sanitized instead, never refused (X-11).
  Constants are `public const string` (compile-time, baked into callers;
  they are wire-frozen keys, so versioning does not change them) except
  where a constant is a handle (`RandomSample`, `NoAdapter`), which is
  `public static readonly`. C text outputs come in two ownerships, and the
  hand layer treats them differently: an owned `RLMeshBytes` is copied into
  a `string` (or `byte[]`) and released with `rlmesh_bytes_free` before the
  call returns; a borrowed `const char*` (the static vocabulary spellings,
  `rlmesh_last_error_message`, `rlmesh_last_error_code_str`) is copied and
  never freed, and the last-error strings, which live in the calling
  thread's slot until its next rlmesh call, are copied before the wrapper
  makes any other rlmesh call on that thread (R-28 Last error). `bytes` →
  `ReadOnlySpan<byte>` in (copied by C), `byte[]` out (an owned copy of
  `RLMeshBytes`).

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
- C# (D-22): a `public sealed class <Name>Options` with a public
  parameterless constructor and one `{ get; set; }` property per keyword,
  named by R-4, each initialized to Python's default
  (`public long? Episodes { get; set; } = null;`), built with an object
  initializer (`new RunOptions { Episodes = 2 }`) or by assignment. The
  parameter is `<Name>Options? options = null`, last, and `null` means a fresh
  default instance. Not a `record` (no synthesized equality, which Python's
  keywords do not have, R-25) and not `init`-only (a caller that assembles
  options conditionally assigns after construction). The call reads every
  property once, at entry, into the C struct (R-19), so mutating the object
  afterwards affects only later calls. Appending a property is a minor change,
  as R-6 states for Rust. Defaults are not C# optional parameters, which
  accept only compile-time constants and are copied into every caller at
  compile time, so a changed default would need every consumer rebuilt.
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
  The other authored constructors take only a message. C#:
  `public ModelException(string message, bool isRecoverable = false)` (a
  constant `false` default is safe to bake into callers, R-7's C# bullet).
- C# `**kwargs` is a `Metadata` property (D-30, R-27); `*args` is a
  `params T[]` positional parameter (`Session.Reader(params ReadItem[] items)`).

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
- C# (D-23): optional parameters. A parameter whose Python type is optional,
  and every text parameter, is nullable with default `null`, which means
  Python's default whatever it is, as Rust's `None`
  (`Shutdown(string? reason = null)` is `"owner shutdown"`), so no
  non-constant or changeable default is compiled into callers; a
  non-optional scalar takes Python's default as its constant
  (`Render(ulong envIndex = 0)`, a `u64` lane offset, Remote), as in C. Named arguments work as in Python
  (`s.Reset(trialIndex: 3)`).
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

C# has no builders: options are set with object initializers or assignment
(R-6), and value records with `with` expressions (R-11).

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
- C#: a `public enum` with `PascalCase` members whose values are C's numbers,
  `int`-backed unless C's enum needs another width; like C++, an input enum
  has no `Invalid` member, so `default(E)` is 0 and the C call refuses it
  with the ValueError class. The text form is an extension method
  `public static string AsString(this E value)` (Python's spelling) in
  `public static class EExtensions`, plus `public static E Parse(string text)`
  there only where Python accepts the text as input (refusing unknown text
  with the ValueError class);
  `ErrorClass` and `WarningCategory` have `Name()` instead (the exception
  above). `Enum.ToString()` prints the C# member name, not Python's
  spelling: .NET does not let a library override it, so it is a translation
  difference and no API prints an enum through it. A bit set (Model's
  `Corners`) is a `[Flags]` enum over C's `uint32_t` bits. `Tolerant<E>` is
  `public readonly struct Tolerant<E> : IEquatable<Tolerant<E>> where E :
struct, Enum` with `bool IsKnown`, `E? Known`, `string? Other`,
  `AsString()`, `ToString()` (Python's spelling) and an implicit conversion
  from `E` (ADP-4). C# cannot seal the constraint to the vocabulary enums as
  Rust's `Vocabulary` trait does, so a `Tolerant<E>` over any other enum
  throws the TypeError class on first use (Adapters).

### R-10 Dunder methods

Map only where Python defines the method:

| Python                   | Rust                                             | C++                         | C                                 | C#                                                     |
| ------------------------ | ------------------------------------------------ | --------------------------- | --------------------------------- | ------------------------------------------------------ |
| `__repr__`, `__str__`    | `Display` (Python's `str`), `Debug`              | `operator<<`, `to_string()` | `rlmesh_<t>_repr` → `RLMeshBytes` | `override string ToString()` (Python's `str`)          |
| `__eq__`                 | `PartialEq`                                      | `operator==`                | `rlmesh_<t>_equals`               | `IEquatable<T>`, `Equals(object?)`, `==` / `!=` (D-26) |
| `__hash__`               | `Eq + Hash`, only if the type has no float field | `std::hash`, same condition | —                                 | content `GetHashCode()`, float fields included (D-26)  |
| `__len__`, `__getitem__` | `len()`, `get(i)`                                | `size()`, `operator[]`      | `rlmesh_<t>_len`, `_get`          | `Count`, `this[int]` (`IReadOnlyList<T>`)              |
| `__iter__`               | `iter()`                                         | `begin()`/`end()`           | `_len` + `_get`                   | `IEnumerable<T>`                                       |
| `__contains__`           | `contains()`                                     | `contains()`                | `rlmesh_<t>_contains`             | `Contains()`                                           |
| `__call__`               | `call()`                                         | `operator()`                | `rlmesh_<t>_call`                 | `Invoke()`                                             |
| `__dlpack__`             | `to_dlpack()`                                    | `to_dlpack()`               | `rlmesh_<t>_to_dlpack`            | `Ext` (Spaces)                                         |
| `__enter__` / `__exit__` | R-3 lifetime row                                 | RAII                        | `_close` / `_free`                | `IDisposable`, `using`                                 |

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

C# forms (D-26):

- **`str` and `repr`.** `ToString()` returns Python's `str`; where `repr`
  differs, a `public string Repr()` method returns it, as C++'s `repr()`.
- **Equality.** Where Python defines `__eq__`, the type implements
  `IEquatable<T>`, overrides `Equals(object?)` and declares `==` / `!=`
  with the same meaning (a `sealed record` synthesizes all four). .NET
  requires `GetHashCode` to agree with `Equals`, so it is always overridden
  with it. Where Python also defines `__hash__` (the frozen dataclasses,
  which Python hashes even with float fields) it hashes the contents; R-10's
  float restriction is Rust's (`f64: !Eq`) and C++'s and does not apply,
  since `double.Equals` is reflexive for NaN. Where Python's type is
  unhashable (`Value`, `Metadata`) the hash is a coarse one that still
  agrees with `Equals` (the kind and the length), and the type documents
  that it is not meant as a dictionary key. Types with identity equality
  in Python (`SpaceSpec`, `Tensor`, SPC-18) keep reference equality.
- **Collections.** `__len__` / `__getitem__` / `__iter__` implement
  `IReadOnlyList<T>` (`Count`, `this[int]`, `GetEnumerator()`), or
  `IReadOnlyCollection<T>` where Python has no positional access. Index
  past the end throws `ArgumentOutOfRangeException`, the BCL contract
  `IReadOnlyList<T>` callers rely on, as C++'s `std::out_of_range`; an area
  that reads a missing position as absent (Core's Value API) offers
  `TryGet(int, out T)` beside it.
- **Keyed mappings** implement `IReadOnlyDictionary<string, T>`:
  `ContainsKey`, `this[string]` (missing key: `KeyNotFoundException`, the
  BCL contract, as C++'s `at`), `TryGetValue`, `Keys`, `Values` in
  insertion order; positional access is `KeyAt(int)` / `ValueAt(int)`; a
  value-membership `contains` stays `Contains(value)`.
- **Input leaves.** C# keeps the table, as Rust and C++ do.
- **No Python counterpart.** `Value` exposes `Count`, `this[int]` and
  `TryGet` for its list, tuple and dict kinds without implementing
  `IReadOnlyList<T>`, since a scalar has no length (Core COR-29). Its
  `ToString()`, which .NET requires and Python has no class for, is a debug
  form that E-3 does not compare.

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
- **C#.** A handle is a `public sealed class` that owns one private
  `SafeHandle` subclass whose `ReleaseHandle` calls `rlmesh_<t>_free`
  (R-28 says on which thread). A C# reference is already a shared
  reference, so a copy-sharing handle needs no `Clone()`, and a move-only
  handle cannot be enforced by the compiler; exclusive use stays Python's
  runtime check (an overlapping call fails with the RuntimeError class).
  `Dispose()` is `__exit__` (Session Drop semantics for `Session`, R-17):
  it releases the wrapper's native reference for every holder of the C#
  reference, as Python's one object is closed for every name bound to it.
  `Close()` is Python's `close()` and reports errors; `Dispose()` swallows
  them. A call on a disposed wrapper throws `ObjectDisposedException`
  (D-41): a C#-only state, like a C++ moved-from object, which Python and
  Rust cannot reach. A call on a closed but undisposed handle reaches C
  and fails as Python does. Value structs (D-28) are `public sealed record`
  classes: Python's required fields are parameters of an explicit
  constructor (`camelCase`, R-4; not positional-record syntax, whose
  parameters would be `PascalCase`) in Python's
  order, defaulted fields are `init` properties initialized to Python's
  defaults, and `with` gives a modified copy; list fields are
  `EquatableList<T>` (D-27) so the synthesized equality is Python's `==`.
  The constructor runs the leaf checks where Python's composite validates;
  `InitVar` sugar becomes an instance method `public T With<Name>(...)`
  returning a modified copy and throwing Python's class, as Rust's
  `with_<name>(self, …)` and C++'s `const` member are. Immutable values (`Value`, `Tensor`, `ModelSpec`,
  `EnvTags`) are sealed classes over a shared native reference.

### R-12 Protocols and authored types

- A Python protocol or overridable class hook becomes a Rust trait (Python's
  optional hooks are default methods), a C vtable (a NULL slot means "not
  defined") and a C++ class detected at compile time through a templated
  adapter; an abstract base class is used only where nothing static is read.
- The ordinary trait impl is the complete Rust path. A proc macro may only be
  sugar that writes what a hand-written impl can write; no capability needs
  it.
- C# (D-32): a protocol is a `public interface I<Name>` holding Python's
  required members. Each optional member or hook is its own one-member
  interface named by R-4 (`ILoad`, `IOnEpisodeEnd`, `IRender`), which the
  authored type implements or not; the wrapper detects each with `is` once,
  at construction, and fills or leaves NULL the matching C vtable slot. This
  is the counterpart of C's NULL slot and C++'s member detection, so
  "written" means "implemented", and an invalid declaration is refused at
  construction as C and C++ refuse it. No default interface methods (they
  cannot express "not defined", and Unity's IL2CPP support for them is not
  established) and no reflection over authored members (R-28). Authored
  types never need a base class, so a Unity `MonoBehaviour` or a Godot
  `Node` can implement them. A Python base class with no-op members whose
  presence is unobservable (`RunHooks`) is an `abstract class` with
  `virtual` no-ops implementing the interface, offered beside the interface,
  never instead of it. Implementing both the context-free and the context
  form of one corner, as any other contradictory declaration, is refused
  with the TypeError class (MDL-23). One `IClose { void Close(); }` serves
  every authored type with a `close` hook (envs, vector envs, factories,
  policies); it is not `IDisposable`, which is the owner's protocol, while
  `Close()` is a hook the library calls (MDL-23, ENV-36). Type-level declarations that Python reads from the
  class (`spec`, `native_chunk`, `load_keys`, `workflow_edition`,
  `tag_params`) cannot be static interface members in netstandard2.1, so
  they travel in a sealed declarations object the type returns from a
  one-member interface and that describe also accepts on its own (Model
  MDL-23 and Env ENV-36 give the shapes), the counterpart of C's static vtable
  fields.
- Protocol methods take Python's parameters positionally in Python's order,
  optional ones as `Option` / NULL / `std::optional` / C# `T?`.
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
  - Tags. A wrapper attaches its payload with
    `rlmesh_ext_callback_set_tagged_error_payload(payload, free_payload,
tag)`, `tag` being the address of a wrapper-owned static object, and
    takes it with `rlmesh_ext_last_error_take_payload_owned(&payload,
&free_payload, &tag)`. It reads a payload only when `tag` is its own and
    otherwise calls the returned `free_payload` without reading it, so the
    C++ and C# wrappers can share one process (a Unity player that also
    loads a C++ native plugin over RLMesh) without either misreading the
    other's payload.
  - C# (X-11): the trampoline catches `Exception`, records class and text
    with R-12's calls by X-11's mapping, and attaches
    `GCHandle.ToIntPtr(GCHandle.Alloc(ExceptionDispatchInfo.Capture(e)))`
    with the C# tag and a static `free_payload` that frees the `GCHandle`
    (cheap, safe on any thread). After a failing C call the wrapper takes
    the payload; with its own tag it calls `ExceptionDispatchInfo.Throw()`,
    which rethrows the author's own exception object with its original stack
    trace; with none, or a foreign tag, it throws the C# class of
    `rlmesh_last_error_class()` (R-21) with the message, code and
    recoverable flag. The same channel carries a warning handler's
    exception (R-23). The C# tag is one byte from `Marshal.AllocHGlobal`,
    allocated once and never freed, so its address is unique in the process.
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
- C# compiles with nullable reference types enabled. An optional value is
  `T?`: `Nullable<T>` for value types (`long?`, `TimeSpan?`, enums) and an
  annotated reference (`string?`, `Value?`, `Metadata?`) for classes;
  `null` is None. `Value.None` (Python's `None` as a value) and a `null`
  `Value?` (absent) stay distinct, as C++'s `Value()` and an empty
  `std::optional<Value>`. An optional list is `IReadOnlyList<T>?`: `null` =
  None, an empty list = `[]`; arrays and `List<T>` convert implicitly
  (`Seeds = new long[] { 0, 1 }`). A non-nullable reference parameter given
  `null` throws `ArgumentNullException` before the C call; it is a C#-only
  misuse (the annotation is compile-time only) and not an R-21 row.

### R-14 Durations and limits

Each parameter keeps Python's own validation: `ServeOptions` timeouts refuse
`<= 0` (`pyrs/lifecycle.rs:231-239`), `EnvServer.wait(0)` is allowed
(`pyrs/server/py_env_server.rs:478-485`), client timeouts follow Remote. A native
language never reads 0 as "unset".

C# (D-35): a duration is `TimeSpan` (`TimeSpan?` when optional), with the
`_seconds` suffix dropped, and converts to C's `double` seconds as
`span.Ticks / 1e7` (exact up to 2^53 ticks, about 28 years); a `double` read back from C
becomes `TimeSpan.FromTicks(checked((long)Math.Round(seconds * 1e7)))`, so
precision is 100 ns. R-14's checks run in C, unchanged: a negative or zero
`TimeSpan` is refused where Python refuses it, and
`Timeout.InfiniteTimeSpan` (−1 ms) is a negative value, never "no
timeout". A Python value `TimeSpan` cannot hold (infinity, NaN, beyond
`TimeSpan.MaxValue`) has no C# spelling; an area whose Python call accepts
infinity as "no limit" states the C# form it uses instead.

### R-15 Tuples and Gymnasium shapes

- A returned tuple stays a tuple in Rust and C++ (`let (obs, info) = ..`,
  `auto [obs, info] = ..`), or a struct named for its fields where Python
  names them. C returns out-parameters in tuple order.
- `reset` returns `(observation, info)` and `step` returns
  `(observation, reward, terminated, truncated, info)` on `Env`, `Session`,
  `RemoteEnv` and `RemoteVectorEnv` alike.
- C# (D-29): the Gymnasium results are named `public readonly struct`s with
  get-only properties in Python's order and a `Deconstruct` method, so
  `var (obs, info) = s.Reset(0);` and
  `var (obs, reward, terminated, truncated, info) = s.Step(action);` read
  like Gymnasium while `result.Reward` stays available: `ResetResult
{ Value Observation; Metadata Info; }`, `StepResult { Value Observation;
double Reward; bool Terminated; bool Truncated; Metadata Info; }`, and
  the vector forms the areas name (C++ uses the same names). An authored
  env returns them with their public constructors. Any other returned tuple
  is a `ValueTuple` with Python's element names (`(Value Observation,
Metadata Info)`), which deconstructs the same way. Hand-written structs
  rather than `record struct` keep them consumable by Unity's C# 9
  compiler (D-19).

### R-16 Sync first, async twins

- Every language's primary API blocks, as Python does (D-1). The one
  exception is the browser binding (R-29), which is async-first.
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
- **C# twins** (D-24; design record, D-45: v0.2 ships only
  `EnvServer.ShutdownAsync`, on the pool below). C#
  adds `<Name>Async` for exactly the calls that have a Rust `_async` twin, named by R-4 (`Session.ResetAsync`,
  `RemoteEnv.StepAsync`, `Model.RunAsync`, `EnvServer.WaitAsync`), except
  where a constructor or a builder cannot be async and D-21's builder rule
  names the twin instead (`RemoteEnv.ConnectAsync`, RMT-24;
  `RLMeshApi.SessionAsync`, RUN-32). The twin takes the sync call's
  parameters plus `CancellationToken cancellationToken = default` last, and
  returns `Task` or `Task<T>` of the sync call's result (not `ValueTask`: a
  twin always completes asynchronously, so `ValueTask` saves nothing and is
  easier to misuse). C has no async API, so a twin runs the sync C call on a
  library-owned worker thread, never a .NET thread-pool worker, since the
  call blocks:
  - a handle that refuses overlapping calls (`Session`, `RemoteEnv`,
    `RemoteVectorEnv`) has one worker, created on its first twin and stopped
    when the handle is released, so a 60 Hz `StepAsync` loop reuses one OS
    thread (and one Mono / IL2CPP thread attachment) instead of creating
    one per call. A twin issued while that worker is busy runs on the pool
    below, so its overlapping C call is refused by C with the RuntimeError
    class as a sync overlap is (Running), never queued behind the first;
  - every other twin (`RLMeshApi.RunAsync`, `Model.RunAsync`,
    `Model.SessionAsync`, `RemoteModel.SessionAsync`, `ConnectAsync`, the
    serve and wait twins) runs on a small library-owned pool that adds a
    thread whenever every pool thread is busy, so no twin waits behind a
    blocked call, and retires a thread idle for 30 s.

  Each twin is a C# `async` method, except the two serve twins that start
  a server before they return a task (`EnvServer.ServeAsync`, ENV-40;
  `EnvFactory.ServeAsync`, ENV-39): each of those is a plain method whose
  prelude does the synchronous start and then returns the task of an
  internal `async` helper, so a startup failure throws from the call
  itself while cancellation still ends the task as below. A twin checks
  `cancellationToken` first and
  throws `Ext.CancelledError` for an already-cancelled token without
  scheduling anything, awaits the worker's completion, and rethrows the
  worker's result. Because the `async` builder records the
  `OperationCanceledException` it is given, the task ends `Canceled` and
  `await` rethrows that same `Ext.CancelledError` object, so
  `catch (Ext.CancelledError)` after `await` works on CoreCLR and Mono
  (E-3); a `TaskCompletionSource.TrySetCanceled` would substitute a
  `TaskCanceledException`, and `Task.Factory.StartNew(f, token)` with a
  cancelled token would never run `f`, so neither is used. `await` resumes
  on the caller's `SynchronizationContext` as usual (Unity's main thread
  under `UnitySynchronizationContext`). The twin's driver is the sync call
  itself on the worker: hooks, relayed env ops and warnings run there
  between segments, outside any async context, so a sync rlmesh call made
  from them runs, where Rust's twin refuses it. Cancellation is R-18's C#
  bullet: each twin hands its C call a per-operation token explicitly, in
  X-12's `cancel` argument or the call's `cancel` field.

- **No refusal in C#.** A sync C# call never refuses because it is made
  from async code. .NET exposes no reliable test: a `SynchronizationContext`
  is present on every Unity main-thread frame, which must be allowed to
  block, and `Task.CurrentId` is set only inside some task bodies. A sync
  call on a thread-pool thread blocks that thread like any blocking .NET
  call. This, and the sync call allowed under a twin, are the C# translation
  differences from Rust's R-16; both are tested (E-3).
- **C# serve twins.** Every C# serve twin follows R-16's `serve_async`
  exception: every lane runs on library threads and no path that a twin
  takes installs a process-wide signal handler, so a twin is safe in an
  engine host. In the design record, `EnvServer.ServeAsync` is `Start()`
  plus `WaitAsync()` (ENV-40), and `EnvFactory.ServeAsync` is the
  background factory serve `rlmesh_ext_env_factory_start` (X-13), run on
  the caller's thread as `EnvServer.ServeAsync` runs `Start()`, so both
  throw startup failures before returning a task, plus `WaitAsync()`
  (ENV-39); `Model.ServeAsync` is the sync C model serve on a pool thread
  with `RLMeshExtModelServeOptions.no_signal_handlers` set (X-15; MDL-24).
  No twin is ever shipped installing signal handlers. The sync `Serve()`
  calls keep C's foreground behaviour (lane 0 on the caller, ENV-11's
  handlers), for console hosts only (R-28); v0.2 ships no C# `Serve()`.

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
- **C#** (design record for sessions and runs, D-45; the ownership rules
  are the v0.2 foundation's, R-28). The compiler cannot check a borrow, so
  the wrapper makes freeing out of order harmless instead. A `Session` (and
  a run, for its duration) holds strong references to the managed model
  and env wrappers it borrows and acquires a reference on each of their
  ownership records (R-28 Ownership records), released only after its own
  native free has finished. A model or env disposed while a session uses
  it is closed as an owner at once (its later C# calls throw
  `ObjectDisposedException`, D-41) and stays alive natively until the
  session releases it; the session keeps working. `Close()` is different:
  closing a model or env under a session is Python's `close()`, and the
  session's next call fails as Python's does. Python's runtime checks are
  unchanged (a second session on a busy env, a remote handle already driven
  with `reset` / `step`, R-17). `Session.Dispose()` is `rlmesh_session_free`,
  so R-17's Session Drop outside a runtime applies: errors swallowed,
  `on_episode_end` for an open episode, the borrowed env closed with
  `close_env`. It runs inline on the disposing thread when R-28's
  `Dispose()` rule allows (no reference outstanding, not inside a callback,
  not a library thread), and is then finished when `Dispose()` returns;
  otherwise the free is queued to the cleanup workers and
  `Ext.Lifetime.WaitFreed` waits for it. On every path
  `rlmesh_session_free` runs to completion first and only then are the
  borrowed records' references released, because the native free still
  uses them; a queued cleanup record owns the session pointer and those
  references together (R-28 Finalization).

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
  - C#: `public sealed class RLMesh.Ext.CancelToken : IDisposable` (a
    `SafeHandle` over one native reference, R-11), `public CancelToken()`
    (wraps `_new`), `public void Cancel()` (any thread). A C# reference
    shares the token; `CancelHandle()` returns a new wrapper over the new
    reference C returns. The `cancel` options property is
    `Ext.CancelToken? Cancel { get; set; }`.
- **Per-call cancel** (X-12, fenced; approved by the user 2026-10-19). A `cancel`
  field reaches only the calls whose options carry one, and a
  `cancel_handle()` token is shared by every call on its handle. Seven
  single calls gain an explicit per-call input:
  `rlmesh_ext_session_reset` / `_predict` / `_step`,
  `rlmesh_ext_remote_env_reset` / `_step` and
  `rlmesh_ext_remote_vector_env_reset` / `_step`, each the unfenced call
  plus a borrowed `cancel` token. The call watches it beside its handle's
  token, subscribes both once at native admission (after the busy and
  overlap checks, so a call refused there consumes no pending firing) and
  carries the subscription explicitly through its watcher until it
  returns; it fails if either fires, with this rule's firing, re-arming and
  state-after-cancel semantics. No call reads a thread-local cancel scope.
  C++ gains `rlmesh::ext` overloads; Rust needs none (it cancels by
  dropping futures and through `cancel` fields). Full contract:
  [C ABI additions for v0.2](../language-parity.md#c-abi-additions-for-v02).
- **C# and `CancellationToken`.** The bridge is v0.2 foundation (D-45),
  internal and unit-tested directly against the C ABI (E-3 Cancellation);
  every twin that would use it is design record, and no public v0.2 call
  takes a native token (`EnvServer.ShutdownAsync`'s token only stops the
  wait, ENV-40). A twin's `cancellationToken` bridges to a fresh
  `Ext.CancelToken` made for that one operation and passed explicitly: in
  X-12's `cancel` argument for the seven single calls, or in the call's
  `cancel` field for a call that has one. A registration on `cancellationToken` calls that token's `Cancel()`
  (any thread); the registration's callback catches every exception, so
  nothing escapes into the cancelling thread, often Unity's main thread.
  Teardown runs in a fixed order: the C call returns, the registration is
  disposed (which waits for a callback already running), and only then is
  the token freed, so `Cancel()` never runs on a freed token. An
  already-cancelled `cancellationToken` fails the twin before scheduling
  (R-16's C# twins bullet), as R-18's idle `cancel()` fails the next call.
  Because the token belongs to one operation and is never reused, neither
  race of the shared-token design exists (X-12: completion and admission);
  E-3 tests both deterministically.
  - **The caller's own `Cancel`.** A call whose options carry the caller's
    `Cancel` has one native `cancel` field, and ABI 5 composes no tokens
    natively (Lessons, cancellation). So in the design record a twin over
    such a call with the caller's `Cancel` set and a cancellable
    `cancellationToken` is refused with the ValueError class before
    scheduling, instead of one silently replacing the other; either alone
    works. Server twins are the exception: they cancel by shutting their
    server down, not through a token, so the caller's `Cancel` stays in its
    field and applies throughout (ENV-39, ENV-40). v2 composes sources
    natively.
  - **Which token fired.** A cancelled twin fails with `Ext.CancelledError`,
    an `OperationCanceledException` whose `CancellationToken` is the twin's
    `cancellationToken`, so the task ends `Canceled` as .NET expects. When
    the caller's `Cancel`, a handle's `CancelHandle()` token or the shutdown
    barrier fired instead, the error is the same class with the same
    `CancellationToken`, which then reads as not cancelled; the task still
    ends `Canceled`, because the `async` builder records any
    `OperationCanceledException`.
  - **Sync C# calls** (design record) pass no per-operation token, except
    the long-running calls with a `cancel` field (`Run`, `Session.Run`,
    `EnvFactory.Serve`, `Model.Serve`): with the caller's `Cancel` unset the
    wrapper puts a library-made token there so that R-28's shutdown barrier
    can cancel them; with it set, the barrier fires the caller's token
    (shutdown is terminal). A server, `EnvServer.Serve` included, needs no
    token: the barrier stops it by shutting the server down.
- **Firing and re-arming.** One `cancel()` fires once: it fails the call in
  flight on the token or, when none is, the next call made with that token,
  at its start. The token then re-arms, as one Ctrl-C raises one
  KeyboardInterrupt. Repeated `cancel()` before that call fails counts once,
  as Python's signal flag coalesces repeated signals before its poll. This
  holds the same way for `cancel_handle()` tokens and `cancel` options
  fields and X-12's explicit per-call arguments. These inputs can put one
  token on concurrent calls on different handles (`Session` and the remote
  handles refuse overlapping calls on the same handle); then one `cancel()`
  fails every call in flight on it.
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
- C# has no dropped futures: a twin's task cannot be dropped, only
  cancelled through its `CancellationToken`, which takes the token path
  above, so the twin's run fires `on_run_end` with the partial result like
  any cancelled run. Abandoning a task (not awaiting it) does not cancel
  it; the C call runs to completion on its thread.
- SIGINT is cancellation; SIGTERM stops a server gracefully and is not an
  error (Env owns the signal rules). C# keeps C's foreground `Serve()`
  installation (ENV-11): in a .NET process it temporarily replaces the
  runtime's own SIGINT / SIGTERM handling (`Console.CancelKeyPress` and
  `PosixSignalRegistration` handlers do not run while `Serve()` is in
  progress) and restores it on return. Engine hosts (Unity, Godot) never
  call a foreground `Serve()`: they use `Start()` or a serve twin, none of
  which installs anything (R-16's C# serve twins, R-28).

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
- **C#.** Sized structs exist only in the internal raw layer (D-18), as
  generated `[StructLayout(LayoutKind.Sequential)]` structs; no public C#
  type exposes `struct_size`, `_INIT` or `_MIN_SIZE`. The generator does not
  translate the `RLMESH_<NAME>_INIT` macros (nor `RLMESH_RANDOM_SAMPLE`'s
  compound literal), so the hand layer has one managed factory per struct,
  `internal static T <Name>Init()`, that sets `struct_size =
(nuint)sizeof(T)` and every field to the macro's value; the E-3 C sizing
  family compares each factory with C's `_INIT` field by field (named
  fields and explicit `_reserved` members, pointers for NULL-ness only),
  never as whole bytes, because implicit padding between fields (as around
  `has_port` / `port` in `RLMeshRemoteEnvOptions`) is unspecified in C and
  differs between two correct initializations. A call fills its struct from the options class
  (R-6), writing every field the library's generation knows, so it always
  sends the generated `sizeof`, at or above every `_MIN_SIZE`, and never
  sends a nonzero tail. Out-structs are initialized with the factory and
  read by R-19's whole-field rule; lent structs are read through
  `struct_size` the same way, so an older C# assembly reading a newer
  library's lent struct sees only the fields it knows. A `bool` field is a
  1-byte `byte` in the raw layer (R-28). The generated layout is checked
  per architecture: pointer-sized fields and `_reserved[sizeof(size_t) -
1]` arrays make it 64-bit specific, and only 64-bit targets ship (D-19).

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
- C# (R-28): every C callback is a `static` method of the hand layer
  marked `[MonoPInvokeCallback(typeof(D))]`, `D` being the generated
  `[UnmanagedFunctionPointer(CallingConvention.Cdecl)]` delegate type for a
  named callback typedef, or a hand-declared one for an anonymous vtable
  field (which the raw layer types `IntPtr`). Each delegate instance lives
  in a `static readonly` field, so the pointer from
  `Marshal.GetFunctionPointerForDelegate` is valid for the process.
  `user_data` is `GCHandle.ToIntPtr(GCHandle.Alloc(state))`, `state` being
  the authored object plus its trampoline bookkeeping; `free_user_data` is
  a static callback that frees the `GCHandle`. On a failed call (R-20's
  ownership rule) the wrapper frees the `GCHandle` itself before throwing.
  Every trampoline catches every exception (X-11) and returns a status; no
  managed exception crosses a native frame. Each `GCHandle` handed to C as
  `user_data`, `value` or `payload` is a root of an ownership record (R-28)
  from `Alloc` until its free callback has run and the native call that ran
  it has been joined, and each trampoline counts its active frames; R-28's
  shutdown barrier uses both (D-43).

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

| Python class (Python base)                                 | Raised for                                                                                                                                                                                                                                                                                                                                                                                                                 | Rust                                                            | C status (value)                                                                 | C++ class : base                                                 | C# class : base                                                        |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `RLMeshException` (`RuntimeError`)                         | an internal or server-side failure with no subclass                                                                                                                                                                                                                                                                                                                                                                        | `Error::Internal(String)`                                       | `RLMESH_ERR_INTERNAL` (99); C-only `RLMESH_ERR_PANIC` (7) maps here              | `rlmesh::RLMeshException : rlmesh::RuntimeError`                 | `RLMesh.RLMeshException : RLMesh.RuntimeError`                         |
| `ProtocolException` (`RLMeshException`)                    | generation mismatch, refused edition, failed handshake, malformed message                                                                                                                                                                                                                                                                                                                                                  | `Error::Protocol(String)`                                       | `RLMESH_ERR_PROTOCOL` (9, new)                                                   | `rlmesh::ProtocolException : RLMeshException`                    | `RLMesh.ProtocolException : RLMeshException`                           |
| `EnvironmentException` (`RLMeshException`)                 | the env reported a failure; carries `code`                                                                                                                                                                                                                                                                                                                                                                                 | `Error::Environment(EnvironmentError)`, `is_recoverable: false` | `RLMESH_ERR_ENVIRONMENT` (3)                                                     | `rlmesh::EnvironmentException : RLMeshException`                 | `RLMesh.EnvironmentException : RLMeshException`                        |
| `RecoverableEnvironmentException` (`EnvironmentException`) | as above; the request failed, the session stays usable                                                                                                                                                                                                                                                                                                                                                                     | `Error::Environment(e)`, `e.is_recoverable: true`               | 3, `rlmesh_last_error_is_recoverable() != 0`                                     | `rlmesh::RecoverableEnvironmentException : EnvironmentException` | `RLMesh.RecoverableEnvironmentException : EnvironmentException`        |
| `ModelException` (`RLMeshException`), new (D-10)           | the model reported a failure; carries `code`, `is_recoverable`                                                                                                                                                                                                                                                                                                                                                             | `Error::Model(ModelError)`                                      | `RLMESH_ERR_MODEL` (4)                                                           | `rlmesh::ModelException : RLMeshException`                       | `RLMesh.ModelException : RLMeshException`                              |
| `ValueError`                                               | an invalid argument value, address or options field                                                                                                                                                                                                                                                                                                                                                                        | `Error::InvalidArgument(InvalidArgumentError)`                  | `RLMESH_ERR_INVALID_ARGUMENT` (1)                                                | `rlmesh::ValueError : std::invalid_argument`                     | `RLMesh.ValueError : System.ArgumentException`                         |
| `AdapterResolutionError` (`ValueError`)                    | adapters cannot bridge env and model                                                                                                                                                                                                                                                                                                                                                                                       | `Error::InvalidArgument(e)`, `e.adapter_resolution.is_some()`   | 1, `RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR`                                 | `rlmesh::AdapterResolutionError : ValueError`                    | `RLMesh.AdapterResolutionError : ValueError`                           |
| `TypeError`                                                | the wrong kind of value or object (a value of another kind, a local env for `RemoteModel`, `view` misuse)                                                                                                                                                                                                                                                                                                                  | `Error::Type(String)`                                           | `RLMESH_ERR_TYPE` (10, new)                                                      | `rlmesh::TypeError : std::logic_error`                           | `RLMesh.TypeError : System.InvalidCastException`                       |
| builtin `RuntimeError`                                     | misuse (closed handle, overlapping or cross-thread call, blocking call inside a runtime, a local `EnvServer` whose contract lacks spaces passed to `RemoteModel.session` (RMT-12; a remote peer's is `ProtocolException`)), local refusals (episode capacity), and a call to a `Model` corner that is not defined (Python raises the subclass `NotImplementedError`, `py/_models/base.py:785-851`; folded here, see below) | `Error::Runtime(String)`                                        | `RLMESH_ERR_RUNTIME` (11, new)                                                   | `rlmesh::RuntimeError : std::runtime_error`                      | `RLMesh.RuntimeError : System.InvalidOperationException`               |
| builtin `OSError`                                          | a filesystem failure: `Recorder.export` refusing its target (Python raises the subclass `FileExistsError`, `py/recorder/export.py:62-74`; folded here, see below) and failed mkdir, write or copy                                                                                                                                                                                                                          | `Error::Os(String)`                                             | `RLMESH_ERR_OS` (12, new)                                                        | `rlmesh::OSError : std::runtime_error`                           | `RLMesh.OSError : System.IO.IOException`                               |
| `ConnectionError` (`OSError`)                              | a transport failure                                                                                                                                                                                                                                                                                                                                                                                                        | `Error::Connection(String)`                                     | `RLMESH_ERR_CONNECTION` (5); `RLMESH_ERR_TRANSPORT` deprecated alias, same value | `rlmesh::ConnectionError : rlmesh::OSError`                      | `RLMesh.ConnectionError : RLMesh.OSError`                              |
| `TimeoutError` (`OSError`)                                 | a connect or request deadline                                                                                                                                                                                                                                                                                                                                                                                              | `Error::Timeout(Duration)`                                      | `RLMESH_ERR_TIMEOUT` (6)                                                         | `rlmesh::TimeoutError : rlmesh::OSError`                         | `RLMesh.TimeoutError : RLMesh.OSError`                                 |
| `KeyboardInterrupt` (`BaseException`)                      | cancellation (R-18)                                                                                                                                                                                                                                                                                                                                                                                                        | `Error::Cancelled`                                              | `RLMESH_ERR_CANCELLED` (8)                                                       | `rlmesh::ext::CancelledError`, no standard base (D-15)           | `RLMesh.Ext.CancelledError : System.OperationCanceledException` (D-25) |
| —                                                          | C only, deprecated                                                                                                                                                                                                                                                                                                                                                                                                         | —                                                               | `RLMESH_ERR_INVALID_VALUE` (2): kept, deprecated, never returned from ABI 5      | —                                                                | —                                                                      |

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
- **C# inheritance** (D-25). C# copies Python's tree as C++ does, rooting
  each builtin at the nearest .NET exception so .NET idioms keep working:
  `ValueError` under `ArgumentException` (its `ParamName` stays `null`, so
  `Message` carries no `(Parameter '…')` suffix and equals R-22's text),
  `TypeError` under `InvalidCastException` (a wrong-kind `Value` read is the
  common case), `RuntimeError` under `InvalidOperationException` (so
  `ObjectDisposedException`, D-41, and the misuse class share a base),
  `OSError` under `IOException`. `TimeoutError` stays under `OSError`, as
  Python's, and so is not a `System.TimeoutException`. Catching a .NET base
  is wider than any Python clause, as a `std::` base is in C++. In
  particular `RLMeshException` derives from `RuntimeError`, as in Python, so
  every rlmesh failure class (`RLMeshException`, `ProtocolException`,
  `EnvironmentException`, `ModelException` and their subclasses) is also an
  `InvalidOperationException`: `catch (InvalidOperationException)` catches
  them together with misuse and `ObjectDisposedException`, as
  `except RuntimeError` catches them in Python. One
  difference has no C++ counterpart: every .NET exception derives from
  `System.Exception`, so `catch (Exception)` catches a cancellation, which
  `except Exception` and C++'s `catch (const std::exception&)` (D-15) do
  not. `Ext.CancelledError` derives from `OperationCanceledException`
  instead, so `Task` machinery reports a cancelled twin as `Canceled` and
  `catch (OperationCanceledException)` reads as it does elsewhere in .NET;
  D-15's intent is not reachable. Every C# exception class has the public
  constructors .NET expects (`()`, `(string message)`, `(string message,
Exception? inner)`) except where Core's authored constructors fix the
  shape (COR-20); `Message` is R-22's text; `ToString()` is .NET's (type,
  message, stack) and outside R-22.
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
  COR-20); C# `throw new EnvironmentException(msg)`,
  `throw new RecoverableEnvironmentException(msg)` and
  `throw new ModelException(msg, isRecoverable: true)` (X-11).
- Rust `Error` stays `#[non_exhaustive]`. `Address` folds into
  `InvalidArgument` and `Server` into `Internal`.

Python today (`pyrs/types.rs:42-78`): `Model`, `Server` and `Internal` raise
bare `RuntimeError`; protocol failures arrive as `Internal`
(`rs/error.rs:205-209,248`), so `ProtocolException` is never raised; env codes
`TIMEOUT`, `INVALID_ACTION`, `CANCELLED` raise `TimeoutError`, `ValueError`,
`RuntimeError`. P-1 to P-5 move Python onto the table.

### R-22 Error text

One formatter in the Rust facade produces the text. Rust `Display`, C
`rlmesh_last_error_message`, C++ `what()`, C# `Exception.Message` and Python
`str(exc)` are equal.
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
  - C#: `public enum WarningCategory { UserWarning = 1, RuntimeWarning = 2,
WorkflowEditionWarning = 3 }` with `public static string Name(this
WarningCategory)` and `public static bool IsSubclassOf(this
WarningCategory, WarningCategory other)` in `WarningCategoryExtensions`;
    `public sealed record Warning` with `public Warning(WarningCategory
category, string message)` and get-only `Category` / `Message`
    properties, whose `ToString()` is `<Category>: <message>`.
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
  - C#: `public static void SetWarningHandler(Action<Warning>? handler)` on
    `RLMesh.RLMeshApi` (D-21); `null` restores the default. One handler, not a
    .NET `event` (D-37): an event's several subscribers would make "the
    handler failed" ambiguous, and R-23's install is one atomic replacement.
    The `Action` is held through a `GCHandle` as `user_data`; a throwing
    handler aborts the call that warned, and the call rethrows the
    handler's own exception (R-12, X-11).
  - Process-wide; `None` / NULL `fn` / empty restores the default.
    Installing or resetting is atomic with respect to emission: the library
    holds the handler by reference count, an emission uses the handler
    current when it starts, and `free_user_data` (NULL = none) runs once,
    after the replacement or reset and after every in-flight call of the
    old handler has returned (R-20's last copy), on whichever thread drops
    that copy.
- **Thread.** The handler runs synchronously on the thread that made the API
  call that warned (for a Rust `_async` twin, the thread polling its future;
  for a C# `Async` twin, the twin's own thread, R-16). A
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
SPC-19) is dropped natively. C# has no counterpart either: no
`System.Numerics.Tensors`, `ML.NET`, Unity `NativeArray<T>` or Godot array
bridge sits on the shared surface. Interop with them goes through `Tensor`'s
spans (Spaces SPC-31); an engine-specific bridge belongs to its integration
(`integrations/unity/`), outside this spec.

### R-25 Language-only extras are fenced

- Rust: `rlmesh::ext` holds documented user-facing extras (`CancelToken`,
  address types, endpoint-token constants, `remove_stale_socket`; each area
  lists its own).
  `rlmesh::binding` is `#[doc(hidden)]` and semver-exempt, for the in-repo
  bindings (Python, C ABI). So is `rlmesh::lockstep`, the v2 session protocol
  engine behind the `lockstep` feature that the bindings drive; users reach
  it through the shared `Model`, `Binding` and `EnvServer`. The live viewer
  engine (`rlmesh::viewer`, behind `View`) and the recorder's video writer
  (`rlmesh::recorder::video`, behind `Recorder.capture`) are hidden plumbing
  too: a run and the Python binding drive them. Nothing language-only stays
  at the crate root, in `rlmesh::prelude` or in a shared module.
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
    `std::optional<ext::ModelServeOptions>`, C# `Ext.EnvServerOptions? Ext`
    / `Ext.ModelServeOptions? Ext`; and, under X-14, the factory serve
    options' `ext`, which only C (`const RLMeshExtEnvFactoryServeOptions*`)
    and C# (`Ext.EnvFactoryServeOptions? Ext`) have, because its one field
    is the host runtime that Rust knows itself and the C++ wrapper fills
    (ENV-12).

  No other shared type carries a user-facing fenced member. Rust `Error`'s
  `#[doc(hidden)]` `Attached` variant (R-12's error payload channel) is
  `binding` plumbing, not an extra: no shared-surface call returns it and
  C, C++ and C# have no counterpart. So are `RunHooks`' `#[doc(hidden)]`
  `transform_observation` and `transform_action`, which carry Python's
  Experimental `RunHooks` pair to the run, and `RunHooks::artifacts` with
  `RecorderCaptureOptions::record`, which carry Python's Experimental
  `Recorder.record` artifact tree; C, C++ and C# have no counterpart in
  v0.2. The ABI
  identification items `RLMESH_ABI_VERSION`, `rlmesh_abi_version` and
  `rlmesh_abi_check` keep their names, because a consumer must call them
  against any ABI generation.

- C++: `rlmesh::ext`.
- C# (D-33): the `RLMesh.Ext` namespace (`Ext.CancelToken`,
  `Ext.CancelledError`, `Ext.EnvServerOptions`, `Ext.EnvFactoryServeOptions`,
  `Ext.ModelServeOptions`, `Ext.EnvFactoryExtensions.Start`,
  `Ext.DlpackTensor` and the typed and raw tensor views, `Ext.Entrypoints`,
  `Ext.Lifetime`; each area lists its own), mirroring C++'s `rlmesh::ext`.
  `Tensor.ToDlpack` / `Tensor.FromDlpack` are Stable names whose types are
  the fenced `Ext.DlpackTensor`, as SPC-19 has it for the other languages. R-18's `cancel` and `CancelHandle()` members and the serve
  options `Ext` property are the same two exceptions on shared types. C#
  has no `binding` module: the raw P/Invoke layer (`RLMesh.Interop`, D-18)
  is `internal`, so it is not public surface at all, and the Unity
  integration uses only the public surface and `RLMesh.Ext` (no
  `InternalsVisibleTo`). The `Async` twins (R-16) are on shared types, as
  Rust's are, not fenced. C#-only conveniences that do not change behaviour
  (implicit conversions into `Value`, `Deconstruct` methods, BCL interface
  implementations required by R-10) are translation, not extras.
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

C, C++, C# and Rust each cover the whole Stable Python surface except R-24's
bridges. Until a symbol lands, its area table marks it missing; a missing
symbol is a gap (G-n), not a design choice. C# is the exception (D-17,
D-45): its column is a design record of the v1 surface, superseded by v2,
not a list of gaps (G-29).

### R-27 One home per shared type

- **One value type.** `rlmesh::Value` (Rust, C++), `RLMeshValue*` (C) and
  `RLMesh.Value` (C#) carry `rlmesh.types.Value`. No `SpaceValue`, `ModelValue`,
  `RLMeshModelValue` or C++ `ValueRef` on the shared surface.
- **One metadata map.** Rust `Metadata` with aliases `InfoDict` and `Kwargs`,
  a C `RLMeshValue*` of kind `DICT`, C++ `Metadata` / `InfoDict` / `Kwargs`;
  insertion-ordered like a Python `dict` (COR-2). No `MetaMap`,
  `RLMeshMetadata` or `RLMeshMetaMap`. C# has `RLMesh.Metadata` only: .NET
  has no exported type alias, so `InfoDict` and `Kwargs` are spelled
  `Metadata` (D-30).
- **One contract type.** `rlmesh::specs::EnvContract`, C `RLMeshEnvContract`,
  C# `RLMesh.Specs.EnvContract`, owned by Env. C# `SpaceSpec` has one
  home, `RLMesh.Spaces.SpaceSpec`: .NET cannot export one type under two
  namespaces, so `rlmesh.specs.SpaceSpec` maps to it (D-30, Spaces).
- Owners (the area that defines the symbol; others link to it):

  | Area                    | Owns                                                                                                                                                                                                                                                                                  |
  | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | rules.md                | error classes and statuses (R-21), C callback errors and the error payload channel (R-12), warnings and `Warning` (R-23), cancellation (R-18), C sizing (R-19), C# runtime constraints (R-28), the C# trampoline mapping (X-11)                                                       |
  | [Core](core.md)         | `Value`, `Metadata` / `InfoDict` / `Kwargs`, error detail fields, editions, describe, build info, `ENV_RESET_OPTIONS_KEY`                                                                                                                                                             |
  | [Spaces](spaces.md)     | spaces, `SpaceSpec`, `Tensor`, `DType`, `Device`, C text views                                                                                                                                                                                                                        |
  | [Env](env.md)           | `Env`, `VectorEnv`, `VectorSeed`, `EnvFactory`, `EnvFactoryExt`, C++ `EnvFactoryBase`, `EnvFactoryServeOptions`, `MadeEnv`, `TagParam`, `Variant`, `VariantOptions`, `EnvServer`, `EnvServerOptions`, `ServeOptions`, `Transport`, `EnvContract`, `ServedEnv`, `trial_index`, signals |
  | [Model](model.md)       | `Model`, `Policy`, `PredictContext`, `BatchPredictContext`, `SpecArg`, `NO_ADAPTER`, `predict_seed`, the entrypoint registry                                                                                                                                                          |
  | [Running](running.md)   | `run`, `session`, `Session`, `RunContext`, `RunHooks`, results, `TelemetryRow`, `StepEvent`, `Reader`, `FrameSources`, `ReadItem`, `View`, `Recorder` and its records (`WorkloadRecord`, `EpisodeRecord`, `MediaRef`), `RANDOM_SAMPLE`, `ModelTarget`, `EnvTarget`                    |
  | [Remote](remote.md)     | `RemoteEnv`, `RemoteVectorEnv`, `RemoteModel`, their options structs, sandbox clients, client timeouts                                                                                                                                                                                |
  | [Adapters](adapters.md) | `EnvTags`, `ModelSpec`, `ObservationRoles`, `tag`, `resolve`, `Adapter`, `Advisory`, `adapters.*` keys, `AdapterResolutionError` detail                                                                                                                                               |

### R-28 C# runtime constraints

C# runs on .NET 10, Mono (the Unity Editor) and Unity's IL2CPP (players)
(D-19), on 64-bit targets only; WebGL, which has no threads, and 32-bit
targets are unsupported (D-40). Every C# area mapping obeys these, and E-2 /
E-3 check them. This rule is the C# foundation that v0.2 implements (D-45);
where a bullet names a handle kind v0.2 does not build (`Session`, `Model`,
the remote clients, hooks, transforms), that part is design record.

- **v0.2 surface** (D-45). The `RLMesh` package is labelled Experimental
  (it joins [compatibility](../../compatibility.md)'s Native APIs list when
  it ships) and exposes what serving an authored env to a Python client
  needs, and nothing more. The list below is the whole public and protected
  surface of both assemblies, every type and member; E-2's v0.2 check fails
  when the dump differs from it. A member the area tables give one of these
  types but the list omits is design record. Names and semantics are the
  area tables'.
  - **Authoring** (Env ENV-36):
    - `interface IEnv`: `Space ObservationSpace { get; }`,
      `Space ActionSpace { get; }`,
      `ResetResult Reset(long? seed, Metadata? options)`,
      `StepResult Step(Value? action)`;
    - `interface IClose`: `void Close()`, called once when the server
      closes the env (`CloseEnvOnShutdown`, ENV-25);
    - `readonly struct ResetResult`:
      `ResetResult(Value observation, Metadata info)`,
      `Value Observation { get; }`, `Metadata Info { get; }`,
      `void Deconstruct(out Value observation, out Metadata info)` (D-29);
    - `readonly struct StepResult`: `StepResult(Value observation, double reward, bool terminated, bool truncated, Metadata info)`, the
      five get-only properties `Observation`, `Reward`, `Terminated`,
      `Truncated`, `Info`, and `Deconstruct` with the five matching `out`
      parameters (D-29). A `null` constructor argument throws
      `ArgumentNullException`; a `default` result returned by an env fails
      the call with the TypeError class.
  - **Handles:**
    - `sealed class Env : IDisposable`: `static Env For(IEnv env)`,
      `void Dispose()` (ENV-37). No `Close()`: a served env is closed by its
      server.
    - `sealed class EnvServer : IDisposable`:
      `EnvServer(Env env, EnvServerOptions? options = null)` and
      `EnvServer(IEnv env, EnvServerOptions? options = null)` (both bind,
      so a bind failure throws from the constructor, ENV-40),
      `string Address { get; }`, `void Start()`,
      `bool Wait(TimeSpan? timeout = null)` (D-23), `void Shutdown()`,
      `Task ShutdownAsync(CancellationToken cancellationToken = default)`
      (ENV-40), `void Dispose()`. No lane or vector constructors, no
      `EnvContract`, no `Serve()` (it installs signal handlers, R-18), no
      `ServeAsync` and no `WaitAsync`: `Start()` does not block, and an
      engine polls `Wait(TimeSpan.Zero)` from its tick.
    - `sealed class EnvServerOptions` (D-22):
      `string? Address { get; set; }` (`null` = `tcp://127.0.0.1:0`),
      `bool CloseEnvOnShutdown { get; set; } = true`,
      `Ext.EnvServerOptions? Ext { get; set; }`. No `Host`, `Port`, `Path`
      or `Transport` (the address spells all four), no `Options`
      (`ServeOptions` keeps Python's defaults) and no `Tags` (adapters do
      not ship).
  - **Values:**
    - `class Space`, with no public or protected constructor and no public
      member of its own, so no type outside the assembly can derive from it
      (SPC-33); `sealed class Box : Space`:
      `Box(Bound low, Bound high, IReadOnlyList<long> shape, BoxOptions? options = null)`; `sealed class BoxOptions`:
      `DType Dtype { get; set; } = DType.Float32`; `readonly struct Bound`
      with exactly SPC-35's implicit conversions and no other public member;
      `sealed class Discrete : Space`:
      `Discrete(long n, DiscreteOptions? options = null)`;
      `sealed class DiscreteOptions`: `long Start { get; set; } = 0`,
      `DType Dtype { get; set; } = DType.Int64`; `enum DType` with C's
      names and numbers (`Bool = 1`, `Uint8`, `Uint16`, `Uint32`, `Uint64`,
      `Int8`, `Int16`, `Int32`, `Int64`, `Float16`, `Float32`,
      `Float64 = 12`).
      `sealed class Dict : Space, IReadOnlyDictionary<string, Space>`:
      `Dict(IEnumerable<KeyValuePair<string, Space>> spaces)` (copies child
      specs; native construction sorts keys, then enumeration preserves that
      declaration order), `int Count { get; }`, `Space this[string key]`,
      `bool TryGetValue(string key, out Space value)`, `bool ContainsKey(string key)`,
      `IEnumerable<string> Keys { get; }`, `IEnumerable<Space> Values { get; }`,
      `IEnumerator<KeyValuePair<string, Space>> GetEnumerator()` and the explicit
      non-generic `IEnumerable.GetEnumerator()` (SPC-34). Missing keys throw
      `KeyNotFoundException`; repeated reads return the same cached child.
      No `SpaceSpec`, `Space(spec)`
      constructors, read-back members (`Kind`, `Shape`, `Low`, `N`, …),
      `Seed`, `Sample`, `Contains` or `Bound` accessors.
    - `sealed class Value : IEquatable<Value>`, Core's Value API (COR-29):
      the implicit conversions from `bool`, `sbyte`, `byte`, `short`,
      `ushort`, `int`, `uint`, `long`, `float`, `double`, `string`,
      `Tensor` and `Metadata`; the `[Obsolete(error: true)]` conversions
      from `char` and `ulong`; `static Value None { get; }`,
      `static Value FromUInt64(ulong v)`,
      `static Value Bytes(ReadOnlySpan<byte> data)`,
      `static Value List(params Value[] items)`,
      `static Value List(IEnumerable<Value> items)`, the two matching
      `Tuple` overloads, `static Value Dict(Metadata items)`;
      `ValueKind Kind { get; }`, `bool IsNone { get; }`, `AsBool()`,
      `AsInt()`, `AsFloat()`, `AsText()`, `AsBytes()`, `AsTensor()`,
      `AsList()`, `AsTuple()`, `AsDict()`; `int Count { get; }`,
      `Value this[int index] { get; }`,
      `bool TryGet(int index, out Value? item)`; `Equals(Value?)`,
      `Equals(object?)`, `GetHashCode()`, `==` and `!=` (COR-24, D-26).
      `enum ValueKind` with C's numbers (`None = 1` … `Dict = 10`).
    - `sealed class Metadata : IDictionary<string, Value>, IReadOnlyDictionary<string, Value>, IEquatable<Metadata>` (COR-30):
      `Metadata()`,
      `Metadata(IEnumerable<KeyValuePair<string, Value>> pairs)`,
      `Value this[string key] { get; set; }`,
      `bool TryGetValue(string key, out Value value)`,
      `bool ContainsKey(string key)`, `void Add(string key, Value value)`,
      `bool Remove(string key)`, `int Count { get; }`,
      `ICollection<string> Keys { get; }`,
      `ICollection<Value> Values { get; }`, `GetEnumerator()` (insertion
      order), `Equals(Metadata?)`, `Equals(object?)`, `GetHashCode()`, `==`
      and `!=` (D-26). The rest of `ICollection<KeyValuePair<string, Value>>` and `IReadOnlyDictionary`'s `Keys` / `Values` are explicit
      interface implementations.
    - `sealed class Tensor : IDisposable` (SPC-31, SPC-32):
      `Tensor(ReadOnlySpan<byte> buffer, IReadOnlyList<long> shape, DType dtype)` (copies), `IReadOnlyList<long> Shape { get; }`,
      `DType Dtype { get; }`, `byte[] Tobytes()`,
      `TensorLease<byte> Buffer()`, `void Dispose()`;
      `sealed class TensorLease<T> : IDisposable where T : unmanaged`:
      `ReadOnlySpan<T> Span { get; }`, `ReadOnlyMemory<T> Memory { get; }`,
      `void Dispose()` (v0.2 makes only `TensorLease<byte>`).
    - No `ToString()` override on any type.
  - **Errors** (R-21, D-25): `RLMeshException`, `ProtocolException`,
    `EnvironmentException` (plus `bool IsRecoverable { get; }`),
    `RecoverableEnvironmentException`, `ValueError`, `TypeError`,
    `RuntimeError`, `OSError`, `ConnectionError`, `TimeoutError` and
    `Ext.CancelledError`, each with exactly the public constructors
    `(string message)` and `(string message, Exception? innerException)`.
    No `ModelException`, `AdapterResolutionError`, `Code` property or error
    code enum: nothing in v0.2 raises or reads them, and a C status of
    another class maps to `RLMeshException`. `Ext.CancelledError` is what a
    cancelled `ShutdownAsync` throws (R-16).
  - **`RLMesh.Ext`:** `sealed class EnvServerOptions`:
    `string? HostRuntimeJson { get; set; }` (ENV-38; `Token`,
    `PeerInfoExtra` and `PublishDescribe` are design record);
    `CancelledError` (above); `static class Lifetime`:
    `bool Shutdown(TimeSpan timeout)`, `IReadOnlyList<string> Pending()`,
    `CancellationToken ShuttingDown { get; }`,
    `LifetimeState State { get; }`,
    `bool WaitFreed(Env env, TimeSpan timeout)`,
    `bool WaitFreed(EnvServer server, TimeSpan timeout)` (D-43);
    `enum LifetimeState { Open, ShuttingDown, Completed, Incomplete }`.

  Internal, never public: `Ext.CancelToken` and the `CancellationToken`
  bridge (R-18's C# bullet, unit-tested directly against the C ABI), the
  raw layer, `NativeGate` and `MonoPInvokeCallbackAttribute` (IL2CPP matches
  it by name, so it need not be public). Nothing else of the C# column
  ships: no vector envs, factories, models, sessions, runs, remote clients,
  adapters, recorder, DLPack, typed tensor extensions or warning handler.
  The foundation's tests reach the other C calls they need (a remote client
  driving the served env, a session over `RLMESH_RANDOM_SAMPLE`, the X-12
  calls under the bridge) through the internal raw layer, visible to the
  test assembly only.

- **No runtime code generation.** No `System.Reflection.Emit`,
  `DynamicMethod`, `Expression.Compile`, `dynamic` or runtime-generated
  marshalling stubs. P/Invoke signatures in the raw layer are blittable
  (pointers, `nint` / `nuint`, fixed-width integers, `double`, `byte` for C
  `bool`, generated blittable structs), with no `string`, `bool`,
  `StringBuilder` or `[MarshalAs]` parameters, so no marshalling stub is
  needed. C `bool` fields and parameters are `byte` (a .NET `bool` field is
  not blittable, and default marshalling would make it 4 bytes).
- **No reflection over authored code.** Authored members are found with
  interface checks (`is`), never `GetMethod`, attribute scans or
  `Activator.CreateInstance` (R-12, D-32), so IL2CPP and trimming cannot
  strip what the library calls. A type-level interface check
  (`typeof(I).IsAssignableFrom(typeof(T))`) is allowed: it reads only the
  type's interface list, which IL2CPP and trimming preserve (Model's
  describe type form, MDL-23). So is `GetType().Name` for advisory text
  (ADP-32). The net10.0-only unload hook additionally reads
  `typeof(LifetimeEngine).Assembly` to find its own AssemblyLoadContext. No
  other `System.Type` member is used (E-2's reference check).
- **Native library.** `[DllImport("rlmesh_capi", CallingConvention =
CallingConvention.Cdecl)]`, the name a `const string` in the raw layer;
  platforms that link the C library statically (iOS) are built with
  `__Internal` instead (D-40). Resolution never depends on
  `NativeLibrary`, `LibraryImport` or `UnmanagedCallersOnly`, none of which
  exists in netstandard2.1; the net10.0 build may add a
  `NativeLibrary.SetDllImportResolver` as an optional extra for custom
  install layouts (one of D-19's three allowed `#if` uses, with the
  internal net10.0 unload hook of the shutdown barrier and the
  `System.Half` overloads).
- **ABI gate.** `rlmesh_abi_check` is a `static inline` header helper
  (`h:397`), not an export, and netstandard2.1 offers no assembly-load hook
  that IL2CPP is known to honour, so the gate is a hand-written
  `internal static class NativeGate` with a lazy, thread-safe validation result
  (`Lazy<RuntimeError?>`), initialized after operation admission.
  It records the first failure of three checks as a RuntimeError class
  failure: loading the library (a `DllNotFoundException`,
  `BadImageFormatException` or `EntryPointNotFoundException` from the first
  call is caught and reported with the library name `rlmesh_capi` and the
  runtime's message); the generation (`rlmesh_abi_version()` against the
  hand layer's `AbiVersion` constant, 5, naming both); and the build
  (`rlmesh_build_info`'s `version` against the hand layer's
  `PackageVersion`, the header's `RLMESH_VERSION`, naming both). The build
  check is needed because X-12 to X-16 land in ABI 5 with no bump, so an
  older ABI-5 library passes the generation check and would fail mid-call;
  the package ships its own native build (D-40), so the two always match
  when installed as shipped. Both constants are hand-written, because D-18's
  generator flags emit no macro bindings; E-3's C sizing check compares
  them with the header. Every public entry point that reaches C calls
  `NativeGate.Ensure()` first, which evaluates validation once and rethrows
  the recorded failure on every later call, so no other C call is made
  against a library that failed it. An `EntryPointNotFoundException` that
  still escapes a later call (a library replaced by hand) is caught by the
  wrapper and reported as the RuntimeError class naming the missing symbol.
- **Callbacks** follow R-20's C# bullet: static methods with
  `[MonoPInvokeCallback]` (the library defines its own
  `MonoPInvokeCallbackAttribute`, which IL2CPP matches by name), delegates
  rooted for the process, `GCHandle` as `user_data`, every exception caught
  (X-11). `UnmanagedCallersOnly` and C# function pointers are not used:
  IL2CPP does not document them for reverse P/Invoke.
- **Ownership records** (D-39). Every native handle the library holds is
  one ownership record: the native pointer, its kind, the records it holds
  references on (a session's model and env, R-17), the managed roots it
  owns (`GCHandle`s and rooted delegates handed to C, R-20) and one
  interlocked 64-bit state word holding a state, a reference count `n` and
  an owner generation `g`. The states are `Idle`, `InUse(n)` (n call-scoped
  references, twins, borrowers and leases), `Transferring`, `Transferred`
  (its pointer given to C, Env ENV-37), `Closing(n)` (owner closed with n
  references outstanding), `Freeing` and `Freed`. Every transition is a
  compare-and-swap loop on that word, so calls, borrowing, disposal,
  finalization, adoption (ENV-37), transfer and the shutdown barrier share
  one state machine; no lock is held across native destruction or a
  callback. The complete table (any pair not listed is refused):

  | Event                     | From                                      | To                                                                                                 |
  | ------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
  | Acquire                   | `Idle`, `InUse(n)`                        | `InUse(n+1)`; any other state: `ObjectDisposedException` (D-41)                                    |
  | Release                   | `InUse(1)`                                | `Idle`                                                                                             |
  | Release                   | `InUse(n)`, n > 1                         | `InUse(n-1)`                                                                                       |
  | Release                   | `Closing(n)`, n > 1                       | `Closing(n-1)`                                                                                     |
  | Release                   | `Closing(1)`                              | `Freeing`; the releaser destroys (below)                                                           |
  | Owner close, generation g | `Idle` at g                               | `Freeing`; the closer destroys                                                                     |
  | Owner close, generation g | `InUse(n)` at g                           | `Closing(n)`                                                                                       |
  | Owner close, generation g | at another generation, or any later state | no change (a stale owner's close is a no-op)                                                       |
  | Adopt (ENV-37)            | `Idle` / `InUse(n)` at g                  | the same state at g+1                                                                              |
  | Transfer reserve          | `Idle`                                    | `Transferring`                                                                                     |
  | Transfer done / undone    | `Transferring`                            | `Transferred` / `Idle`                                                                             |
  | Barrier close             | `Idle`, `InUse(n)` at any generation      | `Freeing` / `Closing(n)`                                                                           |
  | Freed                     | `Freeing`                                 | `Freed`, after the native free returned and the record's references on other records were released |
  - **Owners.** `Dispose()` and the finalizer are the owner close, with the
    generation of the owner token they hold (ENV-37's adoption moves the
    owner to a new token at g+1, so a stale token's close changes
    nothing); a record with one fixed owner stays at generation 0.
    `Dispose()` never throws. It returns after the native free only when
    it made the free itself; otherwise the last release makes it, and
    `Ext.Lifetime.WaitFreed` waits for `Freed`.
  - **Where destruction runs.** An inert record (below) is destroyed inline
    by whichever thread took it to `Freeing`, its finalizer thread
    included: its free runs no authored code and does not block. A
    strong-table record is destroyed inline only by a `Dispose()` that took
    `Freeing` on a thread at callback depth 0 that is not library-owned (a
    twin worker, the pool, a cleanup worker, a teardown thread). Every other
    strong-table destruction (a last release at callback depth > 0 or on a
    library thread, a finalizer's close, a `Dispose()` inside a callback)
    becomes a cleanup record queued for the cleanup workers, so a corner
    that drops the last reference to its own `Model` never waits on itself
    inside `rlmesh_model_free`. The cleanup workers are a library-owned set
    that, like the twin pool (R-16), adds a worker whenever every worker is
    busy and a record is queued and retires one idle for 30 s, so a free
    stuck in a drain holds up only itself. Each cleanup record has an
    interlocked claim word (`Queued` → `Claimed` → `Done`): exactly one
    claimant runs its free, a cleanup worker or a shutdown teardown thread
    (D-43 step 5), and anyone else needing it waits for `Done`.
  - **Strong table.** Records whose free can run authored code or block, or
    that own managed roots, are kept in a strong table from creation to
    `Freed`, independent of wrapper reachability. The table holds the
    record, never the wrapper or the wrapper's owner token: the finalizable
    object of such a handle is its owner token, a `SafeHandle` subclass
    referenced only by the wrapper, whose release is the owner close at the
    token's generation. So an undisposed `Env` or `EnvServer` is finalized
    once unreachable and its free is queued, rather than kept alive by the
    table until shutdown. Every wrapper method ends with `GC.KeepAlive(this)`
    after its release, so a wrapper is never finalized while its own call
    runs. Inert records need no table: their `SafeHandle` is the record.

- **Handle lifetime in calls.** The raw layer passes handles as `IntPtr`,
  which keeps no `SafeHandle` alive, so a disposed check followed by a
  pointer read would race `Dispose()` and the finalizer. Every wrapper call
  acquires each record it passes as a raw pointer, inside a `try`, before
  reading the pointer, and releases it in the `finally`, after the outputs
  are copied, borrowed outputs included; a concurrent `Dispose()` or
  finalization then only defers the free to the last release. A twin
  acquires on the caller's thread before it schedules the call and
  releases when the C call has returned on the worker, so a cancelled or
  abandoned twin, or the shutdown barrier below, never frees a handle under
  a running call. A trampoline that wraps a handle passed to a callback
  acquires it for the callback's duration.
- **Threads.** Where a callback runs depends on the call, as in C:
  1. a sync run or session over a local env, vector env or factory
     (`Run`, `Session.Reset` / `Step` / `Run`) relays env ops, hooks and
     warnings to the thread that made the call (Running RUN-8, RUN-25), so
     a main-thread `Session.Step` over a `MonoBehaviour` env calls it
     inline on the main thread; under a twin that thread is the twin's
     worker (R-16);
  2. `EnvFactory` callbacks (`Prepare`, `Make`, `TagsFor`,
     `EnumerateVariants`, `Close`) run on the thread that called into the
     factory and are exclusive (`h:2758-2761`, ENV-34);
  3. served envs (every lane under `Start()` and the serve twins, every
     lane but lane 0 under `Serve()`), served models and the corners and
     hooks of a `Model` driven by a session or run run on library threads,
     several at once, from threads .NET did not create; .NET, Mono and
     IL2CPP attach such a thread on its first reverse call.

  The library never assumes a main thread, never captures a
  `SynchronizationContext` and never marshals a callback to another thread
  (D-38). An engine integration that must touch engine objects only on its
  main thread dispatches there itself: it calls the authored code inline
  when the callback already runs on the main thread (case 1, or case 2
  entered from the main thread), and otherwise queues the request and
  blocks the library thread until the main thread answers. It must then
  never make a blocking rlmesh call on the main thread that waits on a
  queued callback, or it deadlocks, and it fails its queued and new
  requests once the shutdown barrier starts (below). The foreground
  `Serve()` (which keeps lane 0 on the caller and installs signal handlers,
  R-18) is for console hosts; engine hosts use `Start()` or a serve twin
  (R-16).

- **Last error.** The thread-local last-error slot (R-22) is read on the
  same OS thread right after the failing call, and its strings are copied
  before any other rlmesh call on that thread (R-5). Sync wrappers never
  `await` between the two, and a twin reads it on its worker, so no thread
  switch can intervene. After every failing call the wrapper takes the
  error payload (R-12), so no C#-tagged payload is left in a thread's slot.
- **Finalization** (D-39). Every handle has a finalizable `SafeHandle` as a
  safety net (an inert handle's own, a strong-table handle's owner token);
  `Dispose()` / `using` is the contract. A finalizer performs
  `Dispose()`'s owner close on the record and never destroys on the
  finalizer thread when the free can run authored callbacks or block:
  `Env`, `EnvServer` and, in the design record, `Session`, `Model`,
  `RemoteEnv`, `RemoteVectorEnv`, `RemoteModel`, `Recorder` and its
  captures, `VectorEnv`, `EnvFactory`, `RunHooks`, `RunContext`,
  `AdapterBase`, the transforms and wrapped predicts, `ReadFn` /
  `RenderFn`, and an `Ext.DlpackTensor` that another producer exported (its
  deleter is unknown code, Spaces SPC-31). Their destruction is queued to
  the cleanup workers (Ownership records) as a cleanup record that owns the
  native pointer and the record's references on other records, and
  releases those only after the native free has finished, so no dependency
  is freed under it. Inert handles (values, tensors, spaces, specs, contracts, results,
  records, DLPack tensors the library exported) free inline on the
  finalizer thread: their free runs no authored code and does not block.

  An authored object held by a native vtable (`GCHandle` `user_data`) stays
  reachable until the native side frees it, so a served env is collected
  only after its server is released; an undisposed server is a leak, not a
  crash.

- **Per-value cost.** `Value`, `Tensor`, `Space`, `SpaceSpec` and the other
  immutable wrappers are two objects each: the sealed wrapper and its
  `SafeHandle`, which is also its inert record and the only finalizable
  object, freeing the native reference inline. No public member exposes the
  native pointer and every call holds a call-scoped reference (above), so a
  wrapper can be neither used after its free nor leaked past its
  collection. Children are wrapped when read, not eagerly: an indexer read
  makes one wrapper pair, `AsList()` / `AsTuple()` / `AsDict()` one pair
  per child, and a `Metadata` read from C (an info dict, reset options) one
  pair per value; an empty info dict makes no native wrapper. So one step
  of a C# env stepped through a session (a `Box` observation the env
  returns from a cache, an empty info, a reused action) makes two
  finalizable objects in the library: the action's `Value` handed to
  `IEnv.Step` and the observation's `Value` returned to the caller. E-3's
  benchmark holds this to a budget.

- **Shutdown barrier** (D-43). Before the domain or context that loaded
  RLMesh goes away (a Unity Editor domain reload under Mono, the unload of a
  collectible `AssemblyLoadContext` on .NET 10), no function pointer or
  `GCHandle` handed to C may remain in native hands and no reverse call may
  still be running managed code; IL2CPP players never reload. The holders
  are every record with managed roots: `Env` vtables, `EnvServer`s (their
  envs) and error payloads (`free_payload`, R-12) in v0.2; in the design
  record also `VectorEnv` and `EnvFactory` vtables, `Model` vtables and the
  `ModelOptions` hooks, `RunHooks`, `RunContext`, `AdapterBase`, obs and
  rotation transforms, wrapped predicts, `ReadFn` / `RenderFn`
  (`rlmesh_frame_sources_new`), episode-state values (`free_value`,
  MDL-25), entrypoint registrations (`Ext.Entrypoints`, MDL-11) and the
  warning handler (R-23). Per-operation tokens (R-18) hold no managed root,
  only a native reference, released in step 7. The API is
  `public static bool Ext.Lifetime.Shutdown(TimeSpan timeout)`,
  `IReadOnlyList<string> Ext.Lifetime.Pending()`, the
  `CancellationToken Ext.Lifetime.ShuttingDown`, `Ext.Lifetime.State` (an
  `Ext.LifetimeState`: `Open`, `ShuttingDown`, `Completed`, `Incomplete`)
  and `Ext.Lifetime.WaitFreed` (below). A negative `timeout`, including
  `Timeout.InfiniteTimeSpan`, is the ValueError class (D-35); `Zero`
  starts or resumes the teardown and reports whether it has completed.
  1. **Admission gate.** One interlocked word holds an open bit and the
     count of admitted operations. Every public entry point that reaches C
     (a sync call, a twin, a constructor) enters it before it schedules
     anything and leaves after it has adopted its outputs, published any
     record it made to the strong table and released its call-scoped
     references; a twin enters on the caller's thread and leaves on its
     worker. `Shutdown` clears the open bit with one compare-and-swap; every
     later entry at callback depth 0 throws `ObjectDisposedException`
     (D-41). The gate never reopens in that domain: shutdown is terminal.
     Never refused: a public call at callback depth > 0 (an `IEnv.Step`
     building its `Value` and `Tensor` results on a lane during the drain),
     which is counted as admitted without the open check, because its
     native frame is one that teardown already joins (the server's drain
     waits for the lane's callback); the internal steps of operations
     already admitted (a worker finishing its C call, a registration firing
     its token); trampolines' input and output conversion and error
     reporting (X-11); payload takes; cleanup and teardown work; the
     barrier's own calls; `Dispose()`, which only closes and releases and
     never throws; and `WaitFreed`, which makes no native call.
  2. **Refused contexts.** `Shutdown` returns `false` at once, without
     closing the gate, when it is called on a thread inside a trampoline (a
     per-thread callback depth above 0: a `ProcessExit` raised by
     `Environment.Exit` or `Application.Quit` inside an env callback, or a
     callback calling it), on a library-owned thread, or re-entrantly from
     a `ShuttingDown` registration; `Pending()` then names the refusal. It
     never waits on its own frame. A call from another thread while one is
     running waits for the same outcome, up to its own timeout.
  3. **Notify.** A teardown thread fires `ShuttingDown` (registrations run
     there, never on the caller's thread). An integration's main-thread
     dispatcher then fails every queued and later request (its trampoline
     reports the RuntimeError class; Threads above): it checks
     `ShuttingDown` and enqueues under one lock, so no request can be queued
     after the queue was failed, and no library thread stays blocked on the
     main thread.
  4. **Cancel.** Without blocking, fire the per-operation token of every
     admitted operation (R-18), the `CancelHandle()` token of every live
     `Session`, `RemoteEnv` and `RemoteVectorEnv`, and the caller-set
     `Cancel` of every admitted long-running call (servers are stopped in
     step 5). In v0.2 only the foundation's internal bridge makes
     per-operation tokens and no public call carries one, so this step
     fires tokens only in the bridge's own tests. Calls no token reaches are named by `Pending()`
     until they return: dials (RMT-6), `RemoteModel` calls, `Model.Close`
     waiting for in-flight predicts (X-7), `Recorder` export, and a callback
     blocked in authored code.
  5. **Teardown on tracked threads.** Everything that can block runs on
     teardown threads the barrier creates and tracks, never on the caller's
     thread: `rlmesh_env_server_shutdown` and `_free` for every server (an
     env stuck in `step` holds its lane, and the server's `Drop` waits for
     each lane's `finish`, `capi/env_server.rs`), every strong-table
     record's native free in dependency order (a record releases its
     references on others only after its own free returns), foreign DLPack
     deleters, `rlmesh_ext_unregister_entrypoint` by identity for every
     name C# registered (X-16), and the default warning handler restored
     with `rlmesh_set_warning_handler(NULL, NULL, NULL)`, which frees the
     old `user_data` only after the old handler's calls return (R-23).
     Teardown closes every record's owner (the barrier close, Ownership
     records); a record still referenced by an admitted operation frees on
     that operation's last release. Records without a dependency between
     them tear down in parallel: each queued cleanup record is claimed
     exactly once (Ownership records' claim word) by a cleanup worker or a
     teardown thread, whichever claims first, and a free already claimed
     and running is waited for, never run twice; so one stuck free holds up
     only what depends on it. The records come from the strong table, not
     from wrappers, so collected but unfinalized wrappers are included, and
     teardown re-reads the table until completion (below), so a record
     published after its first pass by an operation admitted before the
     gate closed (a constructor that had not yet published, a worker that
     had not yet adopted its output) joins the same teardown. While it
     waits, a teardown thread runs `GC.Collect()` /
     `GC.WaitForPendingFinalizers()` rounds and the cleanup workers keep
     draining, so finalizer-queued records finish too.
  6. **Completion accounting.** A managed root counts as released only
     when its free callback (`free_user_data`, `free_value`,
     `free_payload`, a foreign deleter) has run and the native call that
     ran it has returned to a thread the barrier joins: a teardown thread's
     or cleanup worker's free returned, or the native operation that
     dropped the root was itself joined by such a free (ABI 5 frees wait
     for the callbacks in flight on their state: env and server frees,
     ENV-7; model close and free, which also wait for the `on_episode_end`
     a session free left on the model's queue past its bound, X-7 and R-17;
     the warning-handler setter, R-23). An empty root set alone proves
     nothing: a free callback can still be returning, and a record's
     cleanup can be what causes the free. A reverse call from a native
     thread that no joined call covers is reported by `Pending()`, never
     assumed finished (v2 moves this accounting into the core: Lessons 7).
  7. **Release managed roots.** Free what the managed side still holds
     (payloads taken but not rethrown, per-operation tokens), stop the
     cleanup workers, the twin workers and the pool, and set `Completed`.

  **Completion.** `Shutdown` returns `true` and the state is `Completed`
  only when all of these hold at once, checked after step 7's stops: the
  gate is closed and its admitted-operation count is zero; the strong table
  is empty (every record `Freed`, including those published after teardown
  began); every managed root is released by step 6's rule; no cleanup
  record is `Queued` or `Claimed`; every teardown thread and cleanup worker
  has been joined and the twin workers and pool are stopped; the
  trampolines' active-frame count (entered first, left in each
  trampoline's outermost `finally`) reads zero; and, in the design record,
  every X-16 registration is removed and the warning handler reset. The
  admitted count reaching zero is a prerequisite of the rest, never
  inferred from them: an admitted constructor paused before publishing its
  record leaves the table empty but the count at one.

  One absolute deadline, `timeout` after the call starts, bounds every
  wait, and every wait is a timed wait on a completion; the caller's thread
  makes no native call, so it never blocks past the deadline. On expiry
  `Shutdown` returns `false` and the state is `Incomplete`: the gate stays
  closed; teardown threads keep running (never aborted); every unfinished
  record keeps its native pointer, its references and its roots, and every
  delegate stays rooted; `Pending()` lists each unfinished operation, each
  unfinished record with its kind and the call it waits in, the active
  trampoline frames and any cleanup failure (X-11's cleanup trampolines).
  A later `Shutdown` resumes waiting on the same teardown and returns
  `true` once it completes. `false` means the domain must not be unloaded.
  After `Completed`, `Shutdown` returns `true` at once.

  **`WaitFreed`.** `public static bool Ext.Lifetime.WaitFreed(Env env, TimeSpan timeout)` and the `EnvServer` overload (the design record adds
  one per callback-bearing handle type) wait until the handle's record is
  `Freed`: its native free has returned and its references on other records
  have been released. They return `true` then (at once if it already is),
  `false` when `timeout` passes first, including for a handle whose owner
  is still open. A negative `timeout` is the ValueError class; `Zero`
  polls. They make no native call and wait on a completion, so the gate
  never refuses them; at callback depth > 0 they return the current answer
  without waiting, because the free being waited for may itself wait for
  that callback.

  **Host preflight.** An unload notification cannot refuse an unload, so
  safety comes from the host running the barrier to `true` before it starts
  the unload, and deferring the unload when it returns `false`:
  - **Unity Editor.** The integration holds
    `EditorApplication.LockReloadAssemblies()` while RLMesh is in use. When
    a reload is wanted (scripts recompiled, Play Mode entered with domain
    reload enabled, its menu command) it calls `Shutdown(5 s)` on a worker
    outside any callback, while the main thread keeps pumping, and calls
    `UnlockReloadAssemblies()` only
    on `true`; on `false` it keeps the lock, so the reload stays deferred,
    logs `Pending()` and offers a retry. Whether the lock defers every
    reload path is what D-44's failed-barrier test checks. Exiting Play Mode
    and entering with domain reload disabled do not run this terminal barrier;
    component teardown stops its server without closing lifetime admission.
    An explicitly confirmed unsafe force-unlock menu is an escape hatch, never
    evidence of safe unload. SessionState records the lock count so an orphaned
    count surviving an unsafe unload can be released once in the new domain.
  - **.NET collectible contexts.** The host calls `Shutdown` and calls
    `AssemblyLoadContext.Unload()` only on `true`.
  - **Fallback diagnostics.** `AppDomain.DomainUnload` (Mono raises it at a
    reload), the net10.0 build's internal `AssemblyLoadContext.Unloading`
    hook and Unity's `AssemblyReloadEvents.beforeAssemblyReload` only
    report: when the barrier has not completed they log `Pending()` as an
    error, saying the unload is unsafe and may crash, and run `Shutdown`
    with a 1 s bound as damage limitation; they never make an unload safe.
    The barrier must finish before an unload starts because Mono aborts
    threads that run managed code when it unloads, and a
    `ThreadAbortException` re-raised at the end of a trampoline's `catch`
    would cross a native frame.
  - **Process exit.** `AppDomain.ProcessExit` calls `Shutdown(5 s)` for an
    orderly stop (servers drained); a process exit needs no unload safety.
    Unity also attempts the barrier on a worker with a bounded wait during normal
    quitting, logs `Pending()` on failure and accepts the original quit request
    regardless (the Editor offers "Quit anyway" or cancel). It never replaces
    the quit request with `EditorApplication.Exit`, bypasses save prompts, or
    drops a player's requested exit code.

  Library threads that have called managed code stay attached to the
  runtime after the barrier; whether Mono's reload waits on them is what
  D-44's Editor reload test checks before v0.2, and a native quiesce call is
  proposed only if that test shows a hang.

- **Memory.** No member of the shared surface returns a span over native
  memory. A `ReadOnlySpan<T>` does not keep its owner reachable, so with
  such a member `var s = MakeTensor().Buffer;` could read storage the
  finalizer had already freed, whatever its lexical scope. Native memory is
  reached through an owning lease, `TensorLease<T>` (Spaces SPC-31): it
  holds its own native tensor reference (`rlmesh_tensor_clone`) from
  creation to `Dispose()` and has no finalizer, so its `Span` and `Memory`
  stay valid until it is disposed even when the lease and the tensor become
  unreachable first; a lease that is never disposed keeps its reference
  for the life of the process (a leak, never a use-after-free). A raw span
  tied to the tensor's own lifetime is fenced in `RLMesh.Ext` as
  `DangerousAsSpan<T>`, valid only while the caller keeps the tensor
  reachable and undisposed (`GC.KeepAlive`). Inputs given as spans are
  copied by C before the call returns.
- **Language level** (D-19). The public surface is consumable from C# 9
  (Unity 2021.3 LTS through Unity 6) and uses no member that needs a later
  compiler or runtime to call: no `required` members, no static abstract
  interface members, no default interface methods, no generic math, no
  `ref` fields. The implementation may use newer syntax that compiles to
  netstandard2.1, with internal polyfills (`IsExternalInit`).

### R-29 Browser (JS) runtime constraints

The browser binding (the `rlmesh` npm package, `javascript/rlmesh`) is
Experimental and binds the v2 session (protocol generation 2), not the v1
surface the area tables map; it has no manifest column. Its constraints:

- **Async-first, the exception to R-16.** Every call that waits on the
  network returns a `Promise` (`connect`, `lease`, `send`, `arm`, `disarm`,
  `estop`), and `subscribe` is an async iterator. There are no sync twins: a
  browser main thread must never block, and a Worker's loop must keep
  draining its socket. Calls that do not wait (`latest`, `peek`, `status`,
  `held`, `drive`, `release`) return at once.
- **Capability profile.** A browser is a client of a session: it observes,
  holds leases on commandable parts and commands them, arms, disarms and
  e-stops, and estimates its clock. It serves no environments, declares no
  robot parts, publishes no samples and runs no edge for hardware (an edge
  must run next to its robot, where a process stall can be bounded). Its
  binding is the gateway's identity binding over the commandable parts, not
  a bound policy map.
- **One core.** It runs `rlmesh-session-core` (the wire and the state machines), the sans-IO
  core shared with the native session, compiled to
  `wasm32-unknown-unknown`: messages are the v2 wire's, and the lease, clock
  and receive rules are the core's. It adds no state machine of its own
  beyond delivery measurement.
- **Through a gateway, as a participant.** It reaches a session only
  through `rlmesh-gateway`, one WebSocket per traffic class. Each control
  connection is a participant of its own: the host grants and lapses its
  leases and maps its clock from its own heartbeats, and a disconnect ends
  the participant. Nothing is resumed by a reconnect.
- **Times and numbers at the boundary.** Times cross as `f64` milliseconds
  since the UNIX epoch (`i64` ns inside); lease tokens and seqs as `f64`,
  exact below 2^53. Modes, priorities and kinds are lowercase strings with
  no wire prefixes.
- **Borrowing.** A borrowed value (`peek`, the `frame` event) is a
  `Disposable` valid until released, and reading it afterwards throws, as
  R-28 rules out spans over native memory for C#; `latest` returns a copy the
  caller owns.

### R-30 Node (JS) native binding

The Node binding is the same `rlmesh` npm package's `default` export
condition (Node 22+, Electron's main process): `Host`, `Robot`, `Policy`, `Rate` and
`Recorder` over a Node-API addon (`javascript/rlmesh/native`, napi-rs) that
binds `rlmesh-session` (its edge included) directly, as PyO3 does. It is
Experimental, binds the v2 session like R-29, and has no manifest column.

- **Python's `rlmesh.v2`, one to one.** The same classes, verbs and
  options, with JS names (`now_ns` is `nowNs`, `edge_stats` is `edgeStats`,
  keyword arguments are an options object). Python's keyword fields
  (`publish("arm", pos=q, vel=dq)`) are an object (`publish('arm', { pos, vel
})`). It has the full capability profile: it hosts sessions, declares
  robots, runs their edges and runs bound policies. Every rule stays in the
  Rust crates; the wrapper only converts.
- **Async-first, the exception to R-16**, as R-29. Calls that wait on the
  network (`Host.open`, `Robot.connect`, `bind`, `arm`, `acquire`, `run`,
  `close`) return a `Promise`; `publish`, `due`, `observe` and `send` do not
  wait and return at once. `Rate.sleep` resolves off the JS thread. The run
  loop runs on a native thread and hands each start to `predict` on the JS
  thread; `predict` may return a `Promise`.
- **Arrays.** A tensor is `{ dtype, shape, data }` with a typed array of the
  dtype (`float16` as `Float16Array` where the runtime has one, else raw
  `Uint16Array` bits); actions are `{ data, shape }`, the dtype following the
  typed array. `publish` casts as NumPy's `same_kind` does and copies before
  it returns. Observation arrays are the caller's.
- **Times and numbers.** Times are `bigint` ns on `process.hrtime.bigint()`'s
  clock (CLOCK_MONOTONIC, Python's `time.monotonic_ns()`), and lease tokens
  are `bigint`; seqs and counters are `number`, exact below 2^53.
- **Errors.** `SessionException` and its subclasses `SessionRefused`
  (`status`), `BindingRefused` (`problems`), `StartSkipped` (`reason`,
  `waiting`, `stale`, `gaps`) and `SessionTimeout`; Python's `ValueError` is
  `RangeError` and its `TypeError` is `TypeError`. A run whose `predict`
  throws rejects with that error.

## Decisions

| ID   | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Status                           |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| D-1  | Sync first. Every primary API blocks; Rust adds opt-in `*_async` twins (R-16).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | User                             |
| D-2  | Full parity in C. C covers the whole surface through handles and options structs; C++ wraps C one to one.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | User                             |
| D-3  | C defaults come from `RLMESH_<NAME>_INIT` macros guarded by `struct_size` (R-19).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | User                             |
| D-4  | `RemoteEnv` and `RemoteVectorEnv` exist in Rust, C and C++. A model is bound to one env as Python's `Model.bind` does: a `Binding` (`rlmesh::bind`, `rlmesh_model_bind`, `Model::bind`); a model served elsewhere is a `Model` built from its participant address (`rlmesh_model_from_address`, `Model::from_address`; Rust's `RemoteModel`). Python's 0.1 `Session`, `session` and `RemoteModel` shapes, kept only to refuse with their 0.2 form, have no native counterpart (revised 2026-10-23, C ABI 6).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | User                             |
| D-5  | C++17 minimum with exceptions, tested at C++17 and C++20. One token-identical definition of every RLMesh entity at C++17 and C++20 (ODR); no standard/feature-macro branches inside or around RLMesh definitions. Only the top-level version/exceptions guards and `std::ranges::enable_borrowed_range` / `std::ranges::enable_view` specializations for `rlmesh::Span<T>` under `#if defined(__cpp_lib_ranges)` may depend on the standard; the specializations add no RLMesh members. Mixed-standard translation units may share wrapper types. Detection traits replace concepts, `rlmesh::Span` replaces `std::span` in public signatures, and options remain aggregates with portable `with_<field>` setters. The C++20 part of D-5 is reversed on 2026-10-16 to reduce integration friction with Unreal 5.0–5.2, ROS 2, Omniverse and Godot. `Result<T>`, `RLMESH_TRY`, abort-on-error and `-fno-exceptions` support remain removed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Settled                          |
| D-6  | C naming. Types and their functions use the type name without its module: `RLMeshBox` / `rlmesh_box_*`, `RLMeshEnvTags` / `rlmesh_env_tags_*`. Free functions and constants keep their module: `rlmesh_adapters_tag`, `rlmesh_spaces_space_from_spec`, `RLMESH_ADAPTERS_ENV_METADATA_KEY`; root ones take none (`rlmesh_run`). A type name used in two modules keeps the module for both (`RLMeshSpacesText`, `RLMeshAdaptersText`); today the only such pair is `Text`. A free function named after the type it builds is `rlmesh_<t>_new`; other builders keep their module (`rlmesh_adapters_resolve` and `rlmesh_adapters_resolve_from_contract` build an `RLMeshAdapter`, whose `rlmesh_adapter_new` is the vtable constructor). Language-only extras use `rlmesh_ext_` / `RLMeshExt` (R-25).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Settled                          |
| D-7  | Project edition lookup. Rust, C and C++ resolve the workflow edition with Python's chain minus its project-file rung (Core COR-6: the call's own value, including the `ServeOptions.workflow_edition` given to `EnvServer` or `Model.serve`; `RLMESH_WORKFLOW_EDITION`; the served peer's option; the type's declaration); they read no project file. Python keeps its `pyproject.toml` `[tool.rlmesh] workflow_edition` rung (`py/_editions.py:162-190`) as a Python-only convenience (R-24).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | User                             |
| D-8  | Deprecated aliases are not ported. `rlmesh.__build__` (deprecated, `py/__init__.py:106-110`) has no counterpart and is not a gap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Settled                          |
| D-9  | `sanitize_metadata` (`py/_metadata.py:23`) is a Python bridge: it coerces arbitrary Python objects. Relabel it in compatibility.md and `api_metadata.json`; no native counterpart.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | User (accepted default)          |
| D-10 | Add a Stable `rlmesh.ModelException(RLMeshException)` with `code` and `is_recoverable`, raised for model failures (P-1). Additive; existing `except RuntimeError` keeps matching. Builtin `RuntimeError` is reserved for misuse (D-12).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | User                             |
| D-11 | Env failures always raise `(Recoverable)EnvironmentException` with the code, including `TIMEOUT`, `INVALID_ACTION` and `CANCELLED` (P-4). Breaking for `except ValueError` / `except TimeoutError` around env calls; needs a changelog entry. The alternative, hybrid classes, has no flat Rust or C form. Follows D-16.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Settled                          |
| D-12 | Builtin `RuntimeError` and `TypeError` raised by rlmesh get their own categories (`Error::Runtime`, `Error::Type`), not `Model` and not `InvalidArgument` (R-21).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | User                             |
| D-13 | The native default warning handler writes to stderr. No `tracing` autodetection: a warning must not depend on whether a subscriber happens to be installed (R-23).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Settled                          |
| D-14 | A C struct larger than the library's must have zero unknown trailing bytes (R-19), mirroring Python's TypeError for an unknown keyword.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Settled                          |
| D-16 | Python is the reference for concepts, not for quirks. Where Python's current behaviour is accidental or inconsistent, Python changes to the target in the same minor release, with a changelog entry (listed as `P-n` / `<AREA>-P<n>`), instead of every language copying the quirk. Deliberate Python behaviour is matched exactly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | User                             |
| D-15 | C++ `rlmesh::ext::CancelledError` derives from no `std::` exception, as `KeyboardInterrupt` is not an `Exception`: `catch (const std::exception&)` does not swallow a cancel, as `except Exception` does not. It still has `what()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Settled                          |
| D-17 | C# is a fifth language (R-26), for game engines and simulators (Unity first, also Godot and Stride). The C# library wraps the C ABI one to one, as C++ does (D-2). Engine integrations (`integrations/unity/`) sit on top and are outside this spec; their constraints shape R-28. Amended by the user on 2026-10-19 (D-45): full parity on the v1 surface is a design record, superseded by v2; v0.2 ships the foundation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | User (amended by D-45)           |
| D-18 | C# layout and layering. `csharp/` at the repo root (sibling of `python/`): `csharp/src/RLMesh/` is the NuGet package and root namespace `RLMesh`. Two layers: a raw P/Invoke layer generated from `rlmesh.h` by ClangSharpPInvokeGenerator (pinned 21.1.8.4; `-x c -std c11 -n RLMesh.Interop -m Native -l rlmesh_capi -c codegen=compatible --generate fnptr-codegen=false --generate funcs-with-body=false --generate helper-types`, with zig's clang builtin include directory and `--no-resource-directory-detection`, as verified in the spike, plus `--with-access-specifier *=Internal` so nothing generated is public) into `csharp/src/RLMesh/Interop/Native.g.cs`, `internal`, never hand-edited; `mise run csharp:generate` writes it and `mise run csharp:generate -- --check` (in `mise run check:csharp` and C#-affected tests) fails when it is stale. Named callback typedefs become Cdecl delegate types; anonymous callback fields become `IntPtr`; no `delegate*`. The `_INIT` macros, static inline helpers and object-like macros (no macro-binding flag is passed) are not generated; the hand layer supplies them, `AbiVersion` and `PackageVersion` included, and E-3 checks them against the header (R-19, R-28's ABI gate). The hand-written idiomatic layer is everything public.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Pending (C#)                     |
| D-19 | C# targets `netstandard2.1` (Unity, Mono, Godot) and `net10.0`, the current LTS (.NET 8 support ends 2026-11-10). The public surface is consumable from C# 9 (R-28 language level). Only 64-bit targets ship: the generated layout is per-architecture (R-19). Both builds come from one source; `#if` is allowed only for the net10.0 `NativeLibrary.SetDllImportResolver` extra (R-28), the internal net10.0 `AssemblyLoadContext.Unloading` diagnostic hook of the shutdown barrier (D-43) and the `System.Half` overloads (SPC-31); the first two are internal, and E-2 diffs the two public surfaces against an allowlist holding only the third. The `System.Half` overloads belong to the typed tensor extensions, which are design record, so in v0.2 the allowlist is empty and the two public surfaces are identical. The SDK is pinned by `global.json`, installed on demand by C#-scoped mise configuration (`dotnet = "10.0.402"`, `core:dotnet`); it builds both targets, and the tests run on the .NET 10 runtime.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Pending (C#)                     |
| D-20 | C# naming is one transform of Python's name (R-4): `PascalCase` types, members and constants, `camelCase` parameters, `I`-prefixed interfaces, verbatim `@` for keyword parameters, no trailing `_`, the `Value` suffix for a member colliding with its type or `object`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pending (C#)                     |
| D-21 | C# free functions. A Python module maps to a namespace (`rlmesh.spaces` → `RLMesh.Spaces`, `rlmesh.adapters` → `RLMesh.Adapters`, `rlmesh.specs` → `RLMesh.Specs`); its free functions and constants are static members of one static class per module named `<Module>Api`: `RLMesh.RLMeshApi`, `RLMesh.Spaces.SpacesApi`, `RLMesh.Adapters.AdaptersApi` (`using static RLMesh.RLMeshApi;` gives `Run(model, env)`). A static class cannot share its namespace's name (`RLMesh.RLMesh` breaks name lookup in every consumer), a lowercase `rlmesh` class (TorchSharp's style) breaks .NET convention, and a bare `Api` collides with user types and breaks the `<Module>Api` pattern. A free function named after the type it builds is that type's constructor, as D-6 makes it `_new` (`rlmesh.session(model, env)` → `new Session(model, env)`). A Python classmethod or staticmethod stays a static method on its type (`Model.FromConfig`, `Tensor.FromDlpack`); a module-level builder stays on the module class (`SpacesApi.SpaceFromSpec`). A builder that must be async is a static `<Name>Async` beside the constructor (`RemoteEnv.ConnectAsync`, RMT-24; `RLMeshApi.SessionAsync`, RUN-32), the one exception to R-16's twin naming.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pending (C#)                     |
| D-22 | C# keyword arguments are options classes (R-6): sealed, mutable `{ get; set; }` properties initialized to Python's defaults, `options = null` last. Rejected: `record` with `init` properties (synthesized equality is an unreviewed extra, conditional assembly needs `with` chains, and C# 9 has no `required`), and C# optional parameters (compile-time constants only, baked into callers).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pending (C#)                     |
| D-23 | C# R-7 calls use optional parameters, `null` meaning Python's default, or Python's constant for a non-optional scalar (R-7).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Pending (C#)                     |
| D-24 | C# `Async` twins (R-16): the Rust twin list, `Task` / `Task<T>`, `CancellationToken` last. Each twin is a C# `async` method over the sync C call on a library-owned worker: one worker per exclusive handle (`Session`, `RemoteEnv`, `RemoteVectorEnv`), a small growing pool for the rest. Its token reaches only its own operation as an explicit per-operation token (X-12's `cancel` argument or the call's `cancel` field, R-18), so neither the completion nor the admission race exists; a twin given both the caller's `Cancel` and a cancellable token over a call with one `cancel` field is refused until v2 composes tokens natively; an already-cancelled token fails before scheduling, and `await` rethrows the same `Ext.CancelledError`. No refusal of sync calls in async code; hooks under a twin run on its worker and may make sync calls. Serve twins install no signal handlers (X-13, X-15). v0.2 ships only `EnvServer.ShutdownAsync`, on the pool (D-45); the rest, `EnvServer.ServeAsync` and `WaitAsync` included, is design record.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pending (C#)                     |
| D-25 | C# exception bases (R-21): `ValueError : ArgumentException`, `TypeError : InvalidCastException`, `RuntimeError : InvalidOperationException`, `OSError : IOException`, `Ext.CancelledError : OperationCanceledException`; the rest follow Python's tree, so `RLMeshException` and every rlmesh failure class below it are `InvalidOperationException`s too, and `catch (InvalidOperationException)` catches them with misuse and `ObjectDisposedException`, as `except RuntimeError` does in Python. `catch (Exception)` catches a cancellation (D-15 not reachable in .NET).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Pending (C#)                     |
| D-26 | C# equality and hashing (R-10): `IEquatable<T>` + `Equals` + `==` where Python has `__eq__`; `GetHashCode` always agrees with `Equals`, content-based where Python hashes (floats included), coarse where Python's type is unhashable; identity types keep reference equality.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Pending (C#)                     |
| D-27 | C# collections. Inputs take `IReadOnlyList<T>` (arrays and `List<T>` convert) or `ReadOnlySpan<T>` for numeric data, and are copied; `*args` is `params T[]`; outputs are `IReadOnlyList<T>`. List fields of value records are `RLMesh.EquatableList<T>` (immutable, `IReadOnlyList<T>`, structural `Equals` / `GetHashCode`, Python's list text), so a record's synthesized equality is Python's dataclass `==`; arrays and `List<T>` convert to it implicitly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pending (C#)                     |
| D-28 | C# value structs (R-11) are `sealed record` classes: required fields are parameters of an explicit `camelCase` constructor in Python's order (not positional-record syntax, whose `PascalCase` parameters E-2's name check would refuse), defaulted fields `init` properties with Python's defaults, `ToString()` overridden to Python's `str` (the synthesized record text differs), `With<Name>` instance methods returning a modified copy for `InitVar` sugar.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Pending (C#)                     |
| D-29 | C# returned tuples (R-15): Gymnasium results are hand-written `readonly struct`s (`ResetResult`, `StepResult`, vector forms) with named properties and `Deconstruct`; other tuples are named `ValueTuple`s.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Pending (C#)                     |
| D-30 | C# has no exported type or namespace alias, so every Python alias is spelled as its target: `InfoDict` and `Kwargs` are `Metadata`, Adapters' `StackSpec` is `IReadOnlyList<int>`, and `rlmesh.adapters.AdapterResolutionError` is `RLMesh.AdapterResolutionError`. A subclass of `Metadata` would make a `Metadata` argument fail to bind where `InfoDict` is declared. The manifest maps each alias to its target with this decision cited.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Pending (C#)                     |
| D-31 | C# tensor memory (R-28, SPC-31): no span over native memory on the shared surface. `Tensor.Buffer()` returns an owning `TensorLease<byte>` that holds its own native reference until disposed (no finalizer: an undisposed lease leaks, it never dangles); typed leases are `Ext.TensorExtensions.Lease<T>()`; raw spans tied to the tensor's lifetime are fenced as `Ext.TensorExtensions.DangerousAsSpan<T>()`; owned copies are `Tobytes()` / `ToArray<T>()`; inputs are copied. DLPack is fenced in `RLMesh.Ext`; a foreign producer's deleter runs on the disposing thread or a cleanup worker, never inline on the finalizer thread.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Pending (C#)                     |
| D-32 | C# authoring (R-12): a required interface per protocol plus one-member optional interfaces detected with `is` at construction; no base class required (engine components implement them); type-level declarations in a sealed declarations object (MDL-23, ENV-36); an abstract base with virtual no-ops only beside an interface (`RunHooks`). Not default interface methods: netstandard2.1 has them, but a default body cannot express C's "slot not defined" (the wrapper cannot tell an override from the default) and IL2CPP's support for them is not established; not static abstract members, which netstandard2.1 lacks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Pending (C#)                     |
| D-33 | C# fence (R-25) is the `RLMesh.Ext` namespace. No public `binding` module: the raw layer is `internal`, and the Unity integration uses only public and `Ext` members.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Pending (C#)                     |
| D-34 | C# integers keep C's widths (`long`, `ulong`, `uint`, `ushort`); in-memory counts and indices are `int`; the assembly is `[CLSCompliant(false)]` (R-5).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Pending (C#)                     |
| D-35 | C# durations are `TimeSpan` (100 ns precision), `_seconds` dropped; R-14's checks unchanged; `Timeout.InfiniteTimeSpan` is negative, not infinite.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Pending (C#)                     |
| D-36 | A C# string bound for a NUL-terminated C input that contains U+0000 is refused with the ValueError class before the call; unpaired surrogates likewise (R-5). The C++ wrapper, which truncates at `c_str()` today, makes the same refusal (`rlmesh::ValueError`) in the same release (v0.2), so the two wrappers over C agree. Text the wrapper builds from an exception it reports to C is sanitized instead (X-11).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Pending (C#)                     |
| D-37 | The C# warning handler is one `Action<Warning>?` set with `RLMeshApi.SetWarningHandler`, not a .NET `event` (R-23).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Pending (C#)                     |
| D-38 | C# callbacks run where C runs them (R-28 Threads): on the calling thread for a sync run or session over a local env and for factory callbacks, and on library threads, several at once, for served envs and models and a model behind a session or run. The library never marshals to a main thread or captures a `SynchronizationContext`; engine dispatch is the integration's job, calling inline when already on the main thread and failing pending requests when the shutdown barrier starts (D-43).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Pending (C#)                     |
| D-39 | C# ownership and finalization (R-28): one ownership record per native handle, with a compare-and-swap state word (`Idle`, `InUse(n)`, `Transferring`, `Transferred`, `Closing(n)`, `Freeing`, `Freed`, plus an owner generation) shared by calls, borrowing, disposal, finalization, adoption, transfer and the shutdown barrier, with R-28's complete transition table. `Dispose()` closes the owner at once and destroys a callback-bearing record inline only when no reference remains and its thread is at callback depth 0 and not a library thread; every other such destruction is a cleanup record queued to the library-owned cleanup workers (which grow so one stuck free holds up only itself), claimed exactly once, and `Ext.Lifetime.WaitFreed` waits for it. A finalizer-reached release of a handle whose free can run callbacks or block is always queued; the cleanup record owns the native pointer and the handle's dependency references and releases those only after the native free. Inert handles free inline wherever their last release happens, the finalizer thread included. Records with roots or blocking frees live in a strong table until freed, independent of wrapper reachability; their finalizable object is an owner token referenced only by the wrapper, so an undisposed handle is finalized, not kept alive by the table.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Pending (C#)                     |
| D-40 | C# distribution. NuGet `RLMesh` carries `lib/netstandard2.1`, `lib/net10.0` and `runtimes/<rid>/native/` for each released 64-bit target; 32-bit targets and WebGL (no threads) are unsupported. v0.2's RIDs are `linux-x64` and `win-x64`, built on the Linux release runner by `release:cxx:package` (the Windows one cross-compiled for MSVC, `docs/release.md`), and `osx-arm64` and `osx-x64`, served by one universal `librlmesh_capi.dylib` built, signed with ArenaX Labs' Developer ID and notarized as extra steps in the existing tag-only `wheels-macos` release job (approved by the user 2026-10-19; no new job, runner or trigger), the same dylib also attached to the GitHub Release as the macOS C/C++ package. The release policy was approved by the user on 2026-10-20: tags fail if Apple secrets are missing unless `vars.RLMESH_ALLOW_UNSIGNED_MACOS=true` explicitly enables ad-hoc signing without notarization (the workflow warns). iOS and Android are not built in v0.2. The raw layer is generated once, on Linux: `rlmesh.h` uses only fixed-width integers, `size_t`, pointers, enums and C `bool` (no `long`, `wchar_t` or bit-fields), so one layout holds on every 64-bit LP64 and LLP64 target; E-3 checks it per RID by compiling a generated C file of `_Static_assert`s on each struct's `sizeof` and field offsets for every shipped target triple (compile only, so it runs on the Linux runner). Unity consumes the prebuilt netstandard2.1 assembly plus the native plugins, not the sources, so the implementation's language version is ours (D-19): a UPM package `com.rlmesh.rlmesh` with `package.json`, `Runtime/RLMesh.dll` (the NuGet netstandard2.1 assembly, its importer set to any platform and auto-referenced) and `Runtime/Plugins/x86_64/{linux,windows}/` holding `librlmesh_capi.so` and `rlmesh_capi.dll`, each importer set to the Editor and Standalone for its OS on x86_64, and `Runtime/Plugins/macOS/librlmesh_capi.dylib` (universal), its importer set to the Editor and Standalone on macOS for any CPU. A second netstandard2.1 build with `__Internal` for statically linked platforms (iOS) is design record until such a target ships; it would be a separate assembly file of the same identity in its own UPM plugin folder, imported for iOS only. A served env reports `{"language":"csharp","framework_versions":{"dotnet" or "mono": "<version>"}}` (C's `host_runtime_json` refuses a `toolchain` other than gcc, clang or msvc, `capi/describe.rs`) through `Ext.EnvServerOptions.HostRuntimeJson` and, under X-14, `Ext.EnvFactoryServeOptions.HostRuntimeJson` (`null` = the wrapper's detected runtime, ENV-38; ENV-12, COR-25); the detection cannot tell IL2CPP from Mono, so the Unity integration sets the value. Describe's own runtime stays C's, as for C++. | Pending (C#)                     |
| D-41 | A call on a disposed C# wrapper throws `ObjectDisposedException` (an `InvalidOperationException`, as `RuntimeError` is, D-25): a C#-only state outside R-21, like a C++ moved-from object.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Pending (C#)                     |
| D-42 | C# enforcement in v0.2 (E-2, E-3 foundation parts): the generated interop's staleness check, the ABI gate test, both assemblies built (netstandard2.1 and net10.0) and their public surfaces read by a System.Reflection.Metadata tool, diffed against each other (the allowlist is empty in v0.2) and against R-28's v0.2 inventory, IL checks for callbacks and forbidden references, field-wise layout and `_INIT` checks, trampoline and exception-containment fixtures, cancellation race tests, the shutdown and collectible-context fixture, the fixture runner on the .NET 10 JIT against each assembly and, as a supplement, NativeAOT, and an advisory benchmark. Deferred to v2: the manifest's `csharp` column, C#'s E-1 entries and E-2 acceptance criteria 1 to 7 (C# surface parity), and the generated C# defaults test. NativeAOT is not evidence for Unity: D-44's Unity tests are.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Pending (C#)                     |
| D-43 | C# shutdown barrier (R-28): `Ext.Lifetime.Shutdown(TimeSpan timeout)` closes a terminal admission gate that counts operations from before scheduling through record publication, output adoption and reference release (public calls at callback depth > 0 counted but never refused; internal cleanup and error reporting exempt), refuses to run from callbacks, library threads and re-entrant notification, fires every operation and handle token, runs every potentially blocking native teardown (server shutdown and free, record frees, foreign deleters, X-16 unregistration by identity, the warning handler reset) on tracked teardown threads under one absolute deadline, in parallel for independent records with each cleanup record claimed exactly once, re-reads the strong table so records published by already-admitted operations join, drains finalizers and cleanup while waiting, and counts a managed root released only when its free has run and the native call that ran it has been joined. It completes only when R-28's completion predicate holds: zero admitted operations (a prerequisite, never inferred), an empty strong table, every root released, no queued or claimed cleanup, every worker joined and the trampolines' frame count at zero. `WaitFreed(handle, timeout)` waits for one handle's free. The caller never blocks past the deadline; on `false` everything unfinished is retained and `Pending()` names it. Hosts preflight: the Unity integration and .NET hosts run it to `true` before starting a reload or unload and defer it otherwise; unload notifications are fallback diagnostics. No native quiesce call: proposed only if D-44's Mono reload test shows a hang.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Pending (C#)                     |
| D-44 | Unity support is claimed only after three v0.2 release gates pass: an IL2CPP player smoke test (a built player serves a C# env to a Python client), a Mono Editor domain-reload test (serve, reload with the barrier, serve again, no crash or hang) and a failed-barrier reload test (an env blocked in `step` makes `Shutdown` return `false`; the reload stays deferred and the Editor responsive; after the env is released a retry returns `true` and the reload proceeds). The user approved dispatch-only GameCI on `depot-ubuntu-24.04-8` on 2026-10-20, using repository Unity activation secrets; NativeAOT and Mono-on-.NET fixture runs are supplements, not substitutes (E-3).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Approved (mechanism, 2026-10-20) |
| D-45 | C# scope for v0.2 (the user's scope change, 2026-10-19). The protocol is redesigned for v2 (clocked sessions, typed channels, timed action chunks, leases, edge runtime, Zenoh, a Rust core); v0.2 ships on the current wire and v2 is a clean break with every binding a thin wrapper over the core's C ABI. So the C# column of every area is a design record of how C# maps onto the v1 surface, not a backlog, and v0.2 does not implement its public API (no part ships behind a flag, CLAUDE.md). v0.2 implements: the C ABI additions X-12 to X-16 that are approved; and the C# foundation: native library loading and the ABI gate, the generated interop layer (D-18), ownership records and `SafeHandle` types (R-28, D-39), callback trampolines with exception containment (X-11), cancellation plumbing (R-18's C# bridge, internal and unit-tested against the C ABI; no public v0.2 call uses it), the shutdown barrier (D-43), NuGet and UPM packaging for `netstandard2.1` + `net10.0` (D-19, D-40), CI with the NativeAOT stand-in and the test harness (D-42). Its public surface is R-28's "v0.2 surface", the exhaustive inventory of what serving an authored env to a Python client needs, labelled Experimental. Lessons for the v2 C ABI are in the [index](../language-parity.md#lessons-for-the-v2-c-abi).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | User (scope); surface pending    |
| D-46 | Served-model batch settings (`BatchOptions`, `Model.serve`'s `batching`) are Python and Rust only in v0.2; C and C++ serve with `BatchOptions`' defaults, and their settings arrive with the next C ABI revision.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| D-47 | The recorder's video codec (`Recorder`'s `codec` and `ffmpeg`, Rust `RecorderOptions::codec` / `ffmpeg`) is Python and Rust only in v0.2; C, C++ and C# record AV1, and the options arrive with the next C ABI revision.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | User (2026-10-26)                |

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
- C# round 1, applied (two reviews against the draft commit, with the
  user's rulings): per-operation cancellation through a C cancel scope
  (X-12; R-16, R-18, D-24), replacing the shared-token bridge and both its
  races, with the caller's `Cancel` kept beside it; twins as `async`
  methods on per-handle workers and a growing pool (R-16); serve twins with
  no signal handlers through a background factory serve (X-13) and a model
  serve field (X-15), each twin dropped from v0.2 if its addition is
  rejected; factory serve host runtime (X-14) and a host runtime without
  `toolchain`, which C refuses for C# (ENV-38, D-40); one shutdown barrier
  in place of `DisposeAll()` covering every holder of a managed pointer,
  entrypoint unregistration (X-16) and the Unity gates (R-28, D-43, D-44);
  explicit and finalizer release paths with dependency-owning cleanup
  records (R-17, R-28, D-39); call-scoped handle references (R-28); the
  env registry rebuilt on native ownership with weak wrappers, `For`
  only, API ownership and factory-output transfer (ENV-37); tensor leases
  in place of borrowed spans and foreign DLPack deleters off the finalizer
  thread (R-28, SPC-31, D-31); the ABI gate as a lazy static check
  (R-28); exact `Value` conversions (COR-29); trampolines guarded end to end
  with a text policy (X-11); owned versus borrowed C text (R-5);
  `netstandard2.1` + `net10.0` with both assemblies checked (D-19, D-42,
  E-2, E-3); overload-aware manifest entries, IL checks, field-wise
  `_INIT` comparison and renumbered C# criteria (E-1, E-2, R-19);
  `RLMeshApi` (D-21), `SpaceSpec`'s one home (R-27), `SpecArg` in
  `RLMesh.Types`, settable `ModelDeclarations` (MDL-23), refused `null`
  declarations (COR-28), the three callback-thread cases (R-28, D-38), C++
  joining D-36, and the D-25, D-32, R-16 naming and R-17 ordering notes.
- C# round 1, not applied: making `IEnv.Step` take a non-nullable `Value`.
  ENV-3 (Settled) keeps `step` receiving an optional action in every
  language, because Python calls `step(None)` for a request with no action;
  C passes NULL for it (`capi/env_api.rs` `step`, `h:2693`), as Rust's `Option<Value>` and C++'s
  `std::optional<Value>` do, so `StepResult Step(Value? action)` is the
  matching C# form.
- Scope change (the user, 2026-10-19; D-45): v2 supersedes the v1-shaped
  API, so the C# column becomes a design record and v0.2 implements only
  the foundation and the C ABI additions; the pending-decision index splits
  into what v0.2 needs and design record; C#'s manifest column is deferred
  to v2; the index gains Lessons for the v2 C ABI.
- C# round 2, applied (two reviews against the round-1 commit; none
  rejected; the round-1 `IEnv.Step` rejection was upheld by both): the
  shutdown barrier redesigned as one admission gate, refused contexts,
  tracked teardown threads under one deadline, joined-free completion
  accounting, retained state on timeout and a host preflight protocol, with
  unload events as diagnostics and a failed-barrier Unity gate (R-28, D-43,
  D-44); one ownership state machine per handle for calls, borrowing,
  disposal, finalization, transfer and the barrier, with deferred frees
  queued off callbacks and library threads and `WaitFreed` (R-17, R-28,
  D-39, ENV-37); `For` independent of GC timing (ENV-37); the DLPack import
  state machine (SPC-31); X-12 replaced by seven explicit per-call cancel
  variants subscribed at admission, with no thread-local scope, the twin
  teardown order, the refused double-source case and the token reported on
  a cancel (R-18, D-24); X-13's per-call gate collection moved into the
  server, admission and scope released at return, the caller's `cancel`
  watched for the whole serve, synchronous start for both serve twins
  (ENV-39); X-14's validator described exactly; X-16 removes by identity
  and drops outside the lock; the benchmark redefined with an advisory
  timing ratio (E-3); `#if` for the internal unload hook (D-19); the factory
  ownership citation is `h:2758-2761`; Model's `RLMeshExtModelServeOptions`
  block follows the header's field order; C++ and C# signatures inside list
  items moved into code blocks.
- C# round 3, applied (two reviews against the round-2 commit, with the
  user's rulings; none rejected): R-28's v0.2 surface is now an exhaustive
  type and member inventory that E-2 checks the assemblies against,
  trimmed to what serving a C# env needs (no `IRender`, `ServeAsync`,
  `WaitAsync`, lane constructors, `EnvContract`, `Env.Close`, `Tags`,
  `Transport`, `ServeOptions`, `SpaceSpec`, `ModelException`
  or error codes; `ShutdownAsync` and `Ext.CancelledError` kept), with the
  decisions its types depend on moved to the index's "Needed for v0.2"
  (D-23, D-26, D-27, D-30, SPC-32, SPC-35). The subsequent user-approved
  authoring scope restores `Dict` and SPC-34 for v0.2, without `SpaceSpec`,
  positional child accessors or a `Spaces` alias. One
  owner and lifecycle design for `Env.For`: an owner generation in the
  record's one compare-and-swap word, a finalizable owner token referenced
  only by the wrapper, a `Building` reservation, wait-and-rebuild after
  retirement, refusal while the library uses the handle, retirement by
  record identity (ENV-37, R-28). D-43: zero admitted operations as a
  completion prerequisite with a full completion predicate, late records
  joining the teardown, R-28's complete transition table, growing cleanup
  workers with exactly-once claims, `WaitFreed`'s signature, public calls
  inside callbacks counted but never refused, timeout validation; cleanup
  trampolines (X-11). X-12, X-13 and X-15 stay in v0.2 by the user's
  decision, their consumers stated honestly (C and C++ hosts, a future
  Unreal host, C#'s internal bridge); the `CancellationToken` bridge is
  internal and unit-tested against the C ABI, and R-18's "except
  `EnvServer`'s" is gone. X-12's completion and idle-firing tests, argument
  order and C++ remote overloads, and the session's lazy make behind
  admission; X-13's single subscription from admission into the server's
  watcher and which of `wait` / `shutdown` reports a cancellation; X-14
  validation before step 0; X-16's empty name. The server binds in its
  constructor (ENV-40); the design record's serve twins are plain methods
  with an internal `async` task. The ABI gate catches load failures and
  pins the library build, with hand-written constants (R-28, D-18);
  packaging names v0.2's RIDs, the UPM layout and per-target layout checks
  (D-40). The trampoline fixture checks identity in process and ENV-35's
  class, code and text across the wire. Lessons for the v2 C ABI gain
  feature discovery, generator-friendly declarations, call error objects,
  per-function concurrency contracts and no implicit process-wide changes,
  and the DLPack, transfer and pending-callback rationales are corrected.

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
- **X-11 C# trampolines** (R-12, R-20, R-28). Every C# callback trampoline
  guards its whole body, not only the authored call: reading the native
  inputs into managed values, the authored call, converting and writing its
  outputs, and reporting the error all run inside one `try` /
  `catch (Exception e)`, and the reporting itself runs inside a second
  guard, so no managed exception crosses a native frame whatever fails. The
  caught class is mapped as X-10 does, most derived first:
  `RecoverableEnvironmentException` / `EnvironmentException` →
  `rlmesh_callback_set_error_class(RECOVERABLE_ENVIRONMENT_EXCEPTION` or
  `ENVIRONMENT_EXCEPTION, raw)`; `ModelException` →
  `rlmesh_callback_set_error(raw, e.IsRecoverable)` in a model callback,
  else `set_error_class(MODEL_EXCEPTION, raw)`; `ProtocolException`,
  `RLMeshException`, `AdapterResolutionError`, `ValueError`, `TypeError`,
  `RuntimeError`, `TimeoutError`, `ConnectionError`, `OSError`,
  `Ext.CancelledError` → `set_error_class` with their own class and
  `e.Message`. `raw` is the message without COR-P8's `[CODE] ` prefix.
  Any other exception, .NET's own included (`ArgumentException`,
  `InvalidOperationException`, `ObjectDisposedException`, a plain
  `OperationCanceledException`, `NullReferenceException`), is outside the
  R-21 tree and records the callback kind's class with
  `rlmesh_callback_set_error(e.Message, false)`: a non-recoverable
  `EnvironmentException` in an env callback (Python reads it as
  `EnvironmentException`, as the spike observed), a non-recoverable
  `ModelException` in a model callback, `RLMeshException` elsewhere. Every
  case then attaches the exception with the C# tag (R-12), so in process
  the caller gets the author's own object back; across the wire the peer
  sees R-21's Environment or Model class with that text.
  - **Text policy.** The reported text is built so that encoding cannot
    fail: a `Message` getter that throws is replaced by
    `<exception type name>: message unavailable`; U+0000 becomes U+FFFD
    (a NUL would truncate the C string) and each unpaired surrogate
    becomes U+FFFD (UTF-8 encoding with a replacement fallback, which
    never throws). D-36 refuses such text only in arguments; exception text
    is sanitized, because refusing it would lose the error. The in-process
    rethrow carries the author's original object, so its `Message` is
    unchanged there; only the C text and the wire text are sanitized.
  - **Fallback.** If building the text or the payload fails anyway (an
    out-of-memory, a `GCHandle.Alloc` failure), the trampoline records the
    callback kind's class with a preallocated, static UTF-8 text
    (`C# callback failed and its error could not be reported`) and no
    payload, through a path that allocates nothing, and returns a nonzero
    status. A payload already attached for the original exception is kept
    when only the text failed.
  - **Cleanup trampolines.** The `void` free callbacks (`free_user_data`,
    `free_payload`, `free_value`, a managed DLPack deleter) have their own
    policy, because native code calls them from destructors that may set no
    `CallbackFrame` (`capi/env_api.rs` `UserData::drop` calls the free
    directly) and may run inside another callback's frame, where reporting
    an error would overwrite that callback's error or leave an orphan
    payload. A cleanup trampoline guards its whole body, always enters and
    leaves the active-frame count (R-28), never calls
    `rlmesh_callback_set_error*` or attaches a payload, and never touches
    the thread's last-error slot. Its work is releasing the root (freeing
    the `GCHandle`, retiring an `Env.For` registration by record identity,
    ENV-37) and marking it released; an exception there is contained and
    recorded in the lifetime diagnostics that `Ext.Lifetime.Pending()`
    lists, and a root it failed to release stays unreleased, so the
    shutdown barrier stays incomplete rather than reporting a false
    success.

  Exceptions the runtime cannot catch (`StackOverflowException`, a
  fail-fast) end the process, as they would in any .NET host.

- **X-12 Per-call cancel inputs** (R-18; C ABI addition,
  Approved by the user, 2026-10-19). Seven `rlmesh_ext_` variants of the
  session and remote-client
  `reset` / `predict` / `step` calls take an explicit `cancel` token,
  subscribed once at native admission; C++ `rlmesh::ext` overloads; for C
  and C++ hosts (a future Unreal host among them; no in-repo code calls
  them yet) and C#'s internal cancellation bridge. Contract and tests:
  [C ABI additions for v0.2](../language-parity.md#c-abi-additions-for-v02).
- **X-13 Background factory serve** (ENV-39; Approved by the user, 2026-10-19).
  `rlmesh_ext_env_factory_start` runs the factory serve's pre-serve steps,
  publishes the describe envelope and returns a started `RLMeshEnvServer*`
  that owns the made envs and keeps the caller's `cancel` subscription,
  taken once at admission, through serving and cleanup. Only cancellation
  that wins the shutdown trigger race changes the result; local shutdown
  and drop probe the same subscription before triggering. C++
  `rlmesh::ext::start`; C#'s `Start` and `EnvFactory.ServeAsync` are design
  record.
- **X-14 Factory serve host runtime** (ENV-12, ENV-38;
  Approved by the user, 2026-10-19). `RLMeshEnvFactoryServeOptions.ext` with
  `RLMeshExtEnvFactoryServeOptions.host_runtime_json`, so a C++ factory
  serve reports its own runtime instead of C's.
- **X-15 Model serve without signal handlers** (MDL-24;
  Approved by the user, 2026-10-19).
  `RLMeshExtModelServeOptions.no_signal_handlers`, consumed by
  C++'s `ext::ModelServeOptions`; C#'s `Model.ServeAsync` is design record.
- **X-16 Entrypoint unregistration** (MDL-11, D-43; Approved by the user, 2026-10-19).
  `rlmesh_ext_unregister_entrypoint`, removing a registration only by its
  owner's identity, so an unloading plugin or C# domain drops its own
  transforms and no one else's; C++ and Rust gain the same `ext` call.

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

```csharp
sealed class Greedy : IPredict
{
    public Value Predict(Value observation) => 0;
}

var model = new Model(new Greedy(), new ModelOptions { Spec = spec });
var result = model.Run(env, new RunOptions { Episodes = 2, Seeds = new long[] { 0, 1 } });
using var s = new Session(model, env);
var (obs, info) = s.Reset(0);
var (obs2, reward, terminated, truncated, info2) = s.Step(s.Predict(obs));
```

In the C# block `env` converts implicitly to Running's `EnvTarget` from a
`RemoteEnv`, `RemoteVectorEnv`, `EnvServer`, address `string` or Env's
`Env` / `VectorEnv` handles; C# cannot convert implicitly from an interface
(CS0552), so an authored `IEnv` takes each entry point's `IEnv` overload, and
`IVectorEnv` / `IEnvFactory` go through `EnvTarget.Vector` /
`EnvTarget.Factory` (RUN-31). `0` converts to `Value` through the
exact `int` operator (COR-29); `Greedy` could equally be a `MonoBehaviour` (R-12).

- **Model.** Rust has a `Policy` trait and a concrete `Model`; the plain trait
  impl is the full path and any attribute macro is sugar (R-12). C has a
  vtable plus an `RLMeshModel*`; C++ detects members; C# detects
  implemented corner interfaces (MDL-23). All share Python's four
  inference corners and fallbacks, `on_episode_end` and `close`
  ([Model](model.md)).
- **Env.** Gymnasium-shaped everywhere (R-15); `EnvFactory` and `Variant`
  exist in every language ([Env](env.md)).
- **Results.** `RunResult`, `EpisodeResult`, `RunHooks`, `StepEvent` with
  Python's field names and units ([Running](running.md)).
- **Metadata** is typed, never JSON text (X-3).

## Known gaps

The October 2026 audit found these gaps before implementation. G-1 to G-28
are closed: Python and Rust by the parity waves, C by ABI 5 and C++ by the
C++17 header. They describe pre-existing code, so none applies to C#, which
starts at the target; G-29 records C# itself. `✓` closed (was a gap or
partial), `—` matched already or does not apply, `open` not yet closed. File references describe the code as audited.

| ID   | Gap                                                                                                                                                                                                                                                                                                                                      | Python      | Rust | C   | C++ | C#                   |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---- | --- | --- | -------------------- |
| G-1  | Episode budget is "until the env ends" (`max_episodes` unset), not Python's 1 or `len(seeds)` (`py/_models/_eval.py:178-195`)                                                                                                                                                                                                            | —           | ✓    | ✓   | ✓   | —                    |
| G-2  | A run closes the model (`close_model = true`, `rs/model/worker.rs:72,109`)                                                                                                                                                                                                                                                               | —           | ✓    | ✓   | ✓   | —                    |
| G-3  | Live episodes evicted at 4,096 (`capi/model.rs:73`); Python refuses past 65,536 (`py/_models/_episodes.py:25,110`)                                                                                                                                                                                                                       | —           | —    | ✓   | ✓   | —                    |
| G-4  | `episodes` / `seeds` named `max_episodes` / `episode_seeds` (`rs/model/worker.rs:47,54`; `h:450,457`)                                                                                                                                                                                                                                    | —           | ✓    | ✓   | ✓   | —                    |
| G-5  | Extra `seeded` / `base_seed` (`h:451-452`; `hpp:892-894`)                                                                                                                                                                                                                                                                                | —           | —    | ✓   | ✓   | —                    |
| G-6  | Model authoring is `ModelHandler` / `ModelWorker` / batch callback, not `Policy` + `Model`                                                                                                                                                                                                                                               | —           | ✓    | ✓   | ✓   | —                    |
| G-7  | No `Session`                                                                                                                                                                                                                                                                                                                             | —           | ✓    | ✓   | ✓   | —                    |
| G-8  | No remote clients                                                                                                                                                                                                                                                                                                                        | —           | —    | ✓   | ✓   | —                    |
| G-9  | Remote clients are async-only (`rs/env/client.rs:59-368`); no blocking primary                                                                                                                                                                                                                                                           | —           | ✓    | —   | —   | —                    |
| G-10 | Blocking calls panic inside a runtime (`rs/model/worker.rs:270-277`)                                                                                                                                                                                                                                                                     | —           | ✓    | —   | —   | —                    |
| G-11 | Env authoring uses request/result structs, not Gymnasium shape                                                                                                                                                                                                                                                                           | —           | ✓    | —   | —   | —                    |
| G-12 | Results are `RuntimeReport` or aggregates, not `RunResult` / `EpisodeResult`                                                                                                                                                                                                                                                             | —           | ✓    | ✓   | ✓   | —                    |
| G-13 | No `ModelSpec` or transforms; C has only resolve/describe of a plan (`h:334-359`)                                                                                                                                                                                                                                                        | —           | ✓    | ✓   | ✓   | —                    |
| G-14 | No `EnvFactory` / `Variant`                                                                                                                                                                                                                                                                                                              | —           | ✓    | ✓   | ✓   | —                    |
| G-15 | Spaces lack `sample` / `seed` on the shared surface (the sampler is internal, `crates/rlmesh-spaces/src/sample.rs`)                                                                                                                                                                                                                      | —           | ✓    | ✓   | ✓   | —                    |
| G-16 | Box bounds are `double` only (`h:272,298-302`); Rust carries typed bounds                                                                                                                                                                                                                                                                | —           | —    | ✓   | ✓   | —                    |
| G-17 | Two value types (`SpaceValue` vs the adapter `Value`)                                                                                                                                                                                                                                                                                    | —           | ✓    | ✓   | ✓   | —                    |
| G-18 | Metadata passed as JSON strings (X-3)                                                                                                                                                                                                                                                                                                    | —           | —    | ✓   | ✓   | —                    |
| G-19 | Editions: Rust has `workflow_edition` on serve and run (`rs/model/worker.rs:93`) and `selected_workflow_edition` on `RemoteVectorEnv` and `RemoteModel` (`rs/env/client.rs:139`, `rs/model/remote.rs:279`), but not on `RemoteEnv`, and no `current_workflow_edition()` or resolver chain; C and C++ declare only when serving (`h:492`) | —           | ✓    | ✓   | ✓   | —                    |
| G-20 | Error classes collapsed (R-21): no Protocol, Type, Runtime, Os; Server maps to `TRANSPORT` (`capi/abi/status.rs:59`)                                                                                                                                                                                                                     | ✓ (P-1–P-5) | ✓    | ✓   | ✓   | —                    |
| G-21 | No warning channel (R-23)                                                                                                                                                                                                                                                                                                                | —           | ✓    | ✓   | ✓   | —                    |
| G-22 | Cancellation: Rust re-exports `CancellationToken`; C `rlmesh_model_cancel` is terminal; no client interrupt                                                                                                                                                                                                                              | —           | ✓    | ✓   | ✓   | —                    |
| G-23 | `struct_size` missing on `RLMeshServeOptions` / `RLMeshRunOptions`, no minimum check anywhere (`capi/env.rs:575`, `capi/model.rs:527` refuse only 0)                                                                                                                                                                                     | —           | —    | ✓   | —   | —                    |
| G-24 | 0-as-unset sentinels (X-4)                                                                                                                                                                                                                                                                                                               | —           | —    | ✓   | ✓   | —                    |
| G-25 | C input enums use 0 as a real value (X-5)                                                                                                                                                                                                                                                                                                | —           | —    | ✓   | —   | —                    |
| G-26 | C++17, `Result<T>`, `-fno-exceptions` (`hpp:1-19`)                                                                                                                                                                                                                                                                                       | —           | —    | —   | ✓   | —                    |
| G-27 | Binding plumbing at the crate root and in the prelude (R-25)                                                                                                                                                                                                                                                                             | —           | ✓    | —   | —   | —                    |
| G-28 | `RLMESH_ABI_VERSION_MAJOR/MINOR/PATCH` carry the package semver (`h:44-52`); `RLMeshContract` not `RLMeshEnvContract`                                                                                                                                                                                                                    | —           | —    | ✓   | ✓   | —                    |
| G-29 | No v1-shaped C# library: the C# column of every area is a design record, superseded by v2, not a backlog (D-17, D-45); v0.2 ships the C# foundation and R-28's Experimental env-serving surface                                                                                                                                          | —           | —    | —   | —   | design record (D-45) |

The browser binding (R-29) has no column here: it binds the v2 session,
not the surface these gaps describe.

## Enforcement

**E-1, E-2 and E-3 are implemented for Python, Rust, C and C++;** the C#
additions in [C# enforcement](#c-enforcement) are a draft (D-42), of which
v0.2 builds the foundation checks. The reviewed mapping manifest and
`mise run parity:check` compare the committed Stable Python snapshot with live
Rust, C and C++ extraction. No native surface snapshot is committed. Shared
behavioural fixtures exercise the four public APIs and check Rust defaults
against the same manifest.

### E-1 Mapping manifest

`docs/specs/language-parity/manifest.toml`, one table per Stable Python
symbol, keyed by its snapshot path. Members and parameters nest under it:

```toml
["rlmesh:Session.reset"]
area = "running"
rule = ["R-7.1"]
parameters = { seed = { default = "None" }, trial_index = { default = "None" } }

["rlmesh:Session.reset".rust]
status = "present"              # present | missing | bridge | exempt
path = "rlmesh::Session::reset"
also = ["rlmesh::Session::reset_async"]
params = { seed = "rlmesh::Session::reset#seed", trial_index = "rlmesh::Session::reset#trial_index" }

["rlmesh:Session.reset".c]
status = "present"
path = "rlmesh_session_reset"
params = { seed = "rlmesh_session_reset#seed", trial_index = "rlmesh_session_reset#trial_index" }

["rlmesh:Session.reset".cpp]
status = "present"
path = "rlmesh::Session::reset"
params = { seed = "rlmesh::Session::reset#seed", trial_index = "rlmesh::Session::reset#trial_index" }
```

- `status = "bridge"` cites R-24; `"exempt"` cites a decision (D-8, D-9).
- `"missing"` requires a `gap` ID.
- Each symbol/member has its own snapshot-path table. Its `parameters` table
  records every Stable parameter; options structs map each keyword to its field
  as `Type#field`, with the default written once as a Python literal
  (`default = "1"`). An `expression` preserves the snapshot spelling for a
  symbolic default or a stub ellipsis; the resolved literal is source-reviewed.
- `also` inventories individual translation types, helpers and members, never
  a wildcard over a type's future members. `bridges` names R-24 parameters;
  `translations` cites decisions for deliberate parameter translations;
  `refused` cites the decision (D-4) for a 0.1 keyword Python keeps only to
  refuse with its 0.2 form, which has no native counterpart.
  `protocol` records C++ detected authoring methods (R-12), `injected` the
  MDL-3/MDL-4 authoring inputs, `collection` the native collection access form,
  and `factory_switch` the R-21 boolean translated into two Rust constructors.
- `python -m rlmesh_api_surface.parity_draft --surfaces <dump.json> --output <draft.toml>` proposes mappings using the area tables and extracted surfaces;
  it refuses to overwrite the reviewed manifest. Obtain review inputs with
  `python -m rlmesh_api_surface.parity --dump <dump.json>`. Resolve every
  `REVIEW` marker against the source before adopting a draft.
- Rust-only and C-only exports are listed under `[ext]` and `[binding]` with
  their rule.

### E-2 Check command

`mise run parity:check`: loads the Python snapshot, extracts the Rust surface
(rustdoc JSON) and the two headers (libclang), and checks them against the
manifest. It runs in `mise run check` and in CI. The task also runs the
checker's synthetic unit tests, including all seven rejection categories.
Rust extraction includes hidden items to inventory `Error::Attached`, but skips
`binding` module contents. Prelude and other re-export aliases are
inventoried individually alongside their canonical paths. The C vendored DLPack definitions are
explicitly inventoried as third-party binding content (SPC-19, SPC-30).

CI uses its default C/C++ drivers and the existing pinned Rust toolchain and
libclang dependency. A local host whose newer system headers cannot be read
by libclang 18 can set `RLMESH_PARITY_HEADER_CC` and
`RLMESH_PARITY_HEADER_CXX` to compatible drivers (for example
`mise exec zig@0.13.0 -- zig cc` / `zig c++`). These affect header parsing
only; Cargo and CI require no override.

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

Shared fixtures in `tests/parity/fixtures/*.toml` describe scenarios once.
`tests/parity/python_runner.py`, the Rust integration test
`crates/rlmesh/tests/parity_fixtures.rs`, and the C11 / C++17 programs in
`tests/parity/` execute the same deterministic two-step counter env and constant
model through their language's APIs. Pytest compares Python/C/C++ observations;
the Rust runner compares the same fixtures via a lossless JSON projection.
Every specified result, class, inheritance flag, code, recoverability flag and
error text is checked. Scenarios restricted to a subset of languages declare
`languages` and a required `why`; pytest includes that reason when skipping.
`mise run parity:fixtures` runs all four and is a dependency of `test:ci`;
`test:cxx` also builds and runs the native scenarios with its selected compilers
at C++17 and C++20. `--repeat 10 --load` repeats the complete gate under one
additional CPU worker. Endpoints are OS-assigned and synchronization uses
channels or bounded state/completion waits, without fixed ports or sleeps.

`tests/parity/generate_defaults.py` generates
`crates/rlmesh/tests/parity_defaults.rs` directly from the reviewed manifest.
It reads `[symbol.parameters.keyword].default` and the Rust `Type#field`
parameter mapping, including the inventoried wrapper field for a whole options
argument. It constructs each mapped options type once, checks every distinct
Python default (90 fields across 22 types), and rejects conflicting duplicate
mappings or unsupported literals. The fixture gate checks that the generated
file is current; generator tests compare the manifest literals with live Python
signatures. Required parameters, native-only cancellation/extension fields and
options fields with no Python default are outside this comparison. See
`tests/parity/README.md` for the runner protocol, CI prerequisites and
ABI-minimum boundary notes.

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

### C# enforcement

Draft (D-42). Nothing below exists yet. v0.2 builds the foundation's
checks: E-2's assembly checks and the E-3 fixtures marked v0.2. C# surface
parity (the manifest's `csharp` column, E-2's acceptance criteria for C#,
the generated defaults test and the fixture families that need the
v1-shaped surface) is deferred to v2: there is no v1-shaped C# surface to
compare (D-45), so until then `parity:check` reads no `csharp` column. The
deferred parts stay here as design record.

**E-1 manifest (deferred to v2).** Every symbol table would gain a `.csharp` table beside `.rust`,
`.c` and `.cpp`, with the same `status` / `path` / `also` / `params` /
`gap` keys and meanings, plus `overloads` where a member has several:

```toml
["rlmesh:Session.reset".csharp]
status = "present"
path = "RLMesh.Session.Reset"
also = ["RLMesh.Session.ResetAsync"]
params = { seed = "RLMesh.Session.Reset#seed", trial_index = "RLMesh.Session.Reset#trialIndex" }

["rlmesh:RunOptions.episodes".csharp]
status = "present"
path = "RLMesh.RunOptions#Episodes"

["rlmesh:EnvServer".csharp]
status = "present"
path = "RLMesh.EnvServer"
overloads = [
  "M:RLMesh.EnvServer.#ctor(RLMesh.Env,RLMesh.EnvServerOptions)",
  "M:RLMesh.EnvServer.#ctor(RLMesh.IEnv,RLMesh.EnvServerOptions)",
  "M:RLMesh.EnvServer.#ctor(System.Collections.Generic.IReadOnlyList{RLMesh.Env},RLMesh.EnvServerOptions)",
  # … every constructor overload, each listed once
]
```

- Paths are metadata names joined with `.`: namespace, type, member; a
  nested type joins with `+` (`RLMesh.Spaces.Box+Bounds`); a generic type
  keeps its arity (``RLMesh.Tolerant`1``); an options or record property is
  `Type#Property`; a parameter is `Method#parameterName` as metadata spells
  it (no `@`).
- **Overloads.** A path names a member group; each overload is identified by
  its normalized signature, the XML documentation ID the C# compiler
  writes (`M:` + type + member + parameter types, generic arity
  written as a backtick and a count, `#ctor` for constructors), which encodes parameter types,
  `ref` / `out` and generic arity exactly. A member with one overload needs
  no `overloads` key. A member with several lists every one: an overload
  missing from the assembly fails criterion 2, and an overload the
  assembly has but the list lacks fails criterion 4, so an unintended
  overload cannot hide behind a shared name. `params` then maps each
  Python parameter per overload (`params."M:…" = { env = "…#env" }`) when
  the overloads' parameter names differ; generic constraints are part of
  the inventoried signature (E-2) and compared as written.
- No `.csharp` tables are written in v0.2; a v2 C# binding drafts them
  with `parity_draft` from its own surface. Its `table_hints`
  reads only four-cell rows today, so with the C# column it reads no area
  table at all; it must accept the five-column rows (Python, Rust, C, C++,
  C#) before it is next used.
- Translations cite their decision in `translations`: `InfoDict` / `Kwargs`
  → `Metadata` (D-30), `rlmesh.specs.SpaceSpec` →
  `RLMesh.Spaces.SpaceSpec` (D-30, R-27), extension-method enum text
  (R-9), `TimeSpan` (D-35), constructors for type-named builders (D-21),
  `Tensor.Buffer()` as a lease method (R-4).
- `[ext]` lists `RLMesh.Ext.*` and the fenced members on shared types
  (R-25). There is no C# `[binding]` list: the raw layer is internal.

**E-2 extraction and check (v0.2, except the manifest comparison).** A
small .NET tool,
`tools/rlmesh_csharp_surface/` (net10.0 console app; no NuGet dependencies,
System.Reflection.Metadata is in the BCL), reads both built assemblies,
`lib/netstandard2.1/RLMesh.dll` (the one Unity consumes) and
`lib/net10.0/RLMesh.dll`, with `PEReader` / `MetadataReader`. It never loads
or runs them, so it needs no native library and works on any host. It
writes JSON in `parity.py`'s `Item` shape (`kind`, `signature`, `params`,
`fields`, `values`, `base`, `protocols`, `default`) per assembly, keyed by
the paths above, with one item per overload:

- every public and protected type: kind (`class`, `struct`, `interface`,
  `enum`, `delegate`; a record is detected by its compiler-generated
  `<Clone>$` and `EqualityContract`), base type and implemented interfaces
  (R-12 protocols, R-21 bases, R-10's BCL interfaces), generic parameters
  and their constraints;
- methods with their documentation ID, parameter names, types, `params`,
  `out` / `ref`, generic arity and constraints, and optional-parameter
  defaults from the `Constant` table, so R-7's defaults are checked from
  metadata; properties with their accessors (`init` is a
  `modreq(IsExternalInit)` setter); fields; enum members and values;
- nullability, decoded from `NullableAttribute` /
  `NullableContextAttribute` and `Nullable<T>`, so an R-13 optional that is
  not nullable (or the reverse) fails;
- `TupleElementNamesAttribute` for R-15's named tuples, `ObsoleteAttribute`
  (with its `error` flag, for COR-29's refused conversions), and
  `ExtensionAttribute` (the R-9 `AsString` / `Name` extensions are
  inventoried under their enum).

In v0.2 the tool runs three checks and the R-28 checks below. First, it
diffs the two dumps: a public item present in one assembly only, or with
another signature, fails unless it is on the allowlist in the manifest's
`[csharp.target_differences]` table, which is empty in v0.2 (the design
record adds the `System.Half` overloads, net10.0 only, SPC-31); the
net10.0 `SetDllImportResolver` extra and unload hook are internal and do
not appear. Second, it compares the netstandard2.1 dump with the checked-in
inventory `csharp/surface/v0.2.json`, which lists exactly R-28's v0.2
surface (every type, member, overload, base and nullability): an item in
one and not the other fails, so the public surface changes only with that
list. Third, the same task runs `mise run csharp:generate -- --check` (the
generated interop is current, D-18).

Deferred to v2: `parity.py` adds `csharp` to `LANGUAGES` and reads the
dumps through
`--csharp-surface <surface.json>`, or builds the assemblies and runs the
tool itself under `mise run dev:heavy`. Names map by R-4's C# transform
(`native_name` gains a C# branch: `PascalCase` members, `camelCase`
parameters, `_seconds` dropped for `TimeSpan`, no trailing `_`); `fenced()`
treats `RLMesh.Ext` as the fence. E-2's seven acceptance criteria apply to
C# with these readings:

1. a Stable Python symbol, member, parameter or field has no `.csharp`
   entry, as written;
2. a `present` entry names a C# item or overload that does not exist, or a
   parameter, property or field whose name differs from Python's after
   R-4's C# transform; the structural parameter exemptions add the
   `cancellationToken` of an `Async` twin (R-16);
3. a `missing` entry's C# item exists (a stale gap), as written;
4. a public C# type, member or overload is outside the manifest and outside
   `RLMesh.Ext`; compiler-generated record members (`<Clone>$`,
   `EqualityContract`, `PrintMembers`, `Deconstruct`), BCL interface
   implementations required by R-10, `Deconstruct` on R-15's result
   structs, and implicit conversions into `Value` are structural, not
   per-entry;
5. an R-7 optional parameter's constant differs from the manifest's Python
   default (read from metadata). Options-class property initializers are
   IL, not metadata; E-3's generated defaults test checks them;
6. a C# enum member's value differs from C's number for the same name, or
   an input enum declares an `Invalid` member (R-9);
7. a C# exception class is missing or has another base than R-21's C#
   column.

The R-28 checks on the assemblies' references and IL (v0.2), with the method
bodies decoded by System.Reflection.Metadata (`MethodBodyBlock`,
`GetILReader`):

- no type or member reference into `System.Reflection.Emit`,
  `System.Linq.Expressions` (`Compile`), `Microsoft.CSharp` (`dynamic`) or
  `System.Activator`, and no reference to `Type.GetMethod`,
  `GetProperty`, `GetField`, `GetMembers`, `MakeGenericType` or
  `MakeGenericMethod`; the only `System.Type` members referenced are
  `GetTypeFromHandle` (`typeof`), `IsAssignableFrom` and `get_Name`
  (`GetType().Name`), plus `get_Assembly` exclusively in the net10.0 unload
  hook (never in netstandard2.1);
- every `DllImport` names `rlmesh_capi` (or `__Internal` in the static
  build) with Cdecl and a blittable signature;
- every method whose address is handed to C carries
  `MonoPInvokeCallbackAttribute` naming its delegate type: the tool finds
  each `ldftn` that is followed by a `newobj` of a delegate type carrying
  `UnmanagedFunctionPointerAttribute`, and fails when the `ldftn` target
  lacks the attribute or is not `static`.

The build itself runs the trimming and AOT analyzers (`IsAotCompatible`,
`IsTrimmable`) with warnings as errors for the net10.0 target.

**E-3 fixtures (v0.2).** `tests/parity/csharp_runner/` is a net10.0
console project implementing the same two-step counter env through the
public C# surface (R-28's v0.2 surface) and speaking the same runner
protocol as `cpp_runner.cpp` (scenario in, JSON observation out). The model
side is the action-zero model of a small C helper compiled with the
fixtures, driven over the internal raw layer (a session, or a remote client
against the C# env served with `Start()`), because the C# model and session
surface is design record (D-45). `run.py` builds it once per
`parity:fixtures` (under the heavy lock, with the C library in the same
build mode as the Python extension, or `RLMESH_RELEASE_BUILD=1`, so the
protocol cohort fingerprints match) and runs it in these configurations,
each compared by pytest with Python, C and C++:

- **JIT, net10.0 build:** the runner references the library project.
- **JIT, netstandard2.1 build:** the same runner references the built
  `lib/netstandard2.1/RLMesh.dll` file (not the project, whose net10.0
  build NuGet would pick), so the assembly Unity consumes runs every
  fixture.
- **NativeAOT** (`PublishAot=true`, net10.0 build), once per gate: a
  supplement that proves no JIT, no `Reflection.Emit` and trimming safety.
  It does not stand in for IL2CPP: it does not read
  `MonoPInvokeCallbackAttribute` (E-2 does), its moving GC differs from
  Boehm's conservative one, and it runs no UnityLinker stripping or IL2CPP
  generic sharing. D-44's Unity tests are the gate for Unity support.
- **Mono, netstandard2.1 build** (optional: added when the .NET Mono
  runtime pack restores on the CI image; `UseMonoRuntime=true`,
  self-contained, Linux x64), the closest licence-free stand-in for the
  Unity Editor's JIT.

The fixture families that apply to the foundation, with their C# readings:

- **ABI gate:** a stub `rlmesh_capi` reporting another generation makes the
  first public call throw the RuntimeError class naming both generations,
  and every later call throw it again, with no other C call made; a stub of
  the right generation reporting another `rlmesh_build_info` version fails
  the same way naming both versions; a missing library and a library
  missing an entry point each fail the first call with the RuntimeError
  class naming `rlmesh_capi` and the cause (R-28).
- **Compile (C#):** a fixture project compiled with C# 9 against the
  netstandard2.1 assembly must fail exactly where the spec refuses at
  compile time (`Value v = 'a';`, `Value v = 1UL;`, COR-29) and compile
  every other conversion (each integer type into `Value`, each `Bound`
  source, SPC-35), checking each value's kind at run time (each `Bound`
  through the Box it builds, C sizing below).
- **Ownership:** a `Dispose()` racing a call on another thread defers the
  free and never faults; a last release inside a callback, or on a library
  thread, queues the free to the cleanup workers, and `WaitFreed` returns
  `true` once it finished, `false` before (a test hook holds the free) and
  at once without waiting at callback depth > 0; `Dispose()` with no
  reference outstanding returns after the native free; an undisposed,
  unreachable `EnvServer` is finalized (its owner token, not kept alive by
  the strong table) and frees on a cleanup worker; a free held by a test
  hook does not delay an unrelated queued free (a second cleanup worker
  starts); each transition of R-28's table, a `Closing(n)` releasing to
  `Closing(n-1)` and a stale owner token's close changing nothing included,
  is driven once; an inert `Tensor` released last inside a callback frees
  inline (R-28 Ownership records, Finalization).
- **Env handles (ENV-37):** `Env.For` on one object from many threads
  builds one native handle (a C helper counts live env handles), and a
  re-entrant `For` from the object's own space getter during the build
  throws the RuntimeError class; `For`, drop without `Dispose`, `For` again
  returns the same wrapper with no collection, and a wrapper over the same
  native handle when the old wrapper was collected but its owner token not
  yet finalized (a test hook holds the finalizer), after which the old
  token's finalizer changes nothing; when the old token was finalized
  first, `For` returns a wrapper over a new native handle, and the old
  handle's free returned before the new one was built (the live count never
  exceeds one); a finalizer held after reading its generation while `For`
  adopts ends with exactly one owner; no variant throws; `For` on an object
  that `new EnvServer(obj)` made a handle for throws the RuntimeError class
  while that server lives and, after `server.Dispose()` and
  `WaitFreed(server, …)`, builds a new handle; `For` after the caller's own
  `Dispose()` while a server still borrows the handle throws the
  RuntimeError class until the server ends; an old record's
  `free_user_data` never removes a newer registration of the same object.
- **Cancellation:** X-12's tests in C and C++, and in C# for the internal
  bridge, unit-tested directly against the C ABI with no public twin: a
  `CancellationToken` bridged to a per-operation token passed to
  `rlmesh_ext_remote_env_step` against a C#-served env that blocks on a
  barrier fails the call with the cancellation class, which the bridge
  reports as `Ext.CancelledError` carrying that `CancellationToken`; an
  already-fired token fails before any C call (the env counts call
  starts); the completion and admission races; a pool worker reused after
  a cancelled operation carries nothing over; `Cancel()` racing the
  registration's disposal never throws on the cancelling thread and never
  touches a freed token; the bridge frees each token only after the C call
  returned and the registration was disposed.
- **Serving (C#):** a Python client drives a C# env served with `Start()`;
  `Wait(TimeSpan.Zero)` returns `false` while it serves and `true` after
  `Shutdown()`; a bind failure (an address in use) throws
  `ConnectionError` from the `EnvServer(IEnv)` constructor, which releases
  the handle it made, so a later `Env.For` on the object succeeds; `ShutdownAsync` drains and closes per `CloseEnvOnShutdown` with
  `Shutdown()` on a pool thread, not the caller's; one cancelled while
  draining ends `Canceled` with `Ext.CancelledError` (caught after `await`
  on CoreCLR and Mono as the same object) while the drain finishes, and one
  given an already-cancelled token throws it without shutting down; a
  `Dispose()` during `ShutdownAsync` defers the free until the drain
  finishes; a C# server with default `Ext` options constructs, and the
  handshake's PeerInfo reads `language` `"csharp"` (ENV-38).
- **Errors (trampolines):** in process, through the internal session path
  over a C# env, each R-21 class the v0.2 surface defines, thrown by an env
  callback, comes back at the C# caller as the same object
  (reference-equal) with its original stack trace, and a .NET exception
  outside the tree as the same object too (R-12's payload). Across the
  wire, a Python client of a C# server sees ENV-35's authored failure for
  each thrown class: the `EnvironmentException` class with code `INTERNAL`,
  the text `[INTERNAL] step failed: <message>`, and
  `RecoverableEnvironmentException` only for a thrown
  `RecoverableEnvironmentException` (`env/sync_wire.rs`
  `authored_failure`). An exception whose `Message` holds U+0000 and an
  unpaired surrogate, and one whose `Message` getter throws, each fail the
  call with X-11's sanitized or fallback text, never crash; a cleanup
  trampoline that throws (a test hook) is contained, reported by
  `Pending()`, and leaves the caller's own error intact.
- **C sizing (C#):** each generated struct's `sizeof` and field offsets
  equal C's (a C helper compiled with the fixtures prints them), and the
  generated `_Static_assert` file compiles for every shipped target triple
  (D-40); each managed `_INIT` factory equals C's `_INIT` field by field,
  explicit `_reserved` members included and implicit padding excluded,
  pointers by NULL-ness (R-19); `AbiVersion` and `PackageVersion` equal the
  header's `RLMESH_ABI_VERSION` and `RLMESH_VERSION`; and real calls, not
  layouts, exercise the by-value and `bool` paths: a Box built with every
  `RLMeshBound` arm (bool, int, uint, float) reads its bounds back through
  the raw layer, every `bool` parameter, return and out pointer kind the
  v0.2 surface uses round-trips both values, and every callback typedef the
  v0.2 surface installs is invoked by C at least once.
- **C# runtime:** an env served from C# receives callbacks on several
  library threads at once; forcing `GC.Collect()` and
  `GC.WaitForPendingFinalizers()` while serving breaks nothing (delegates
  rooted, `GCHandle`s live); a tensor whose last managed use was `Buffer()`
  survives a forced collection while its lease is undisposed and its span
  reads the original bytes (R-28 Memory).
- **Shutdown barrier (C#):** a fixture loads `RLMesh.dll` into a
  collectible `AssemblyLoadContext`, serves an env with `Start()`, drives
  it from a C client, makes the env throw once (an error payload), and
  leaves one `Env` wrapper undisposed and unreachable (created in a
  non-inlined helper). As the host's preflight it calls
  `Ext.Lifetime.Shutdown`, expects `true`, then unloads the context and
  checks through a `WeakReference` that it is collected and through a C
  helper's callback counter that no callback ran after `Shutdown` returned.
  Variants: an env blocked forever in `step` makes `Shutdown(1 s)` return
  `false` within the deadline plus a scheduling margin (the caller is not
  blocked by the stuck free), `Pending()` names the server, the host does
  not unload, and after the env is released a second `Shutdown` returns
  `true` and the unload proceeds; `Shutdown` from inside an env callback,
  from a `ShuttingDown` registration and from a pool worker each return
  `false` at once; a negative timeout, `Timeout.InfiniteTimeSpan`
  included, is the ValueError class; a call admitted before `Shutdown`
  finishes or is cancelled, a call after it throws
  `ObjectDisposedException`, and `Dispose()` after it does not throw; an
  `EnvServer` constructor held by a test hook after admission and before
  publishing its record keeps `Shutdown` from completing (a short timeout
  returns `false` with the table empty but the operation counted), and
  once released its server is torn down by the same teardown and a later
  `Shutdown` returns `true`; likewise a worker held before adopting its
  output; an env callback that builds `Value` and `Tensor` results during
  the drain is not refused; two servers, one with an env blocked in `step`,
  tear down in parallel, the free stuck behind the blocked env holding up
  only itself; the `DomainUnload` / `Unloading` diagnostics log `Pending()`
  when the host skipped the preflight.
- **Benchmark (C#; v0.2):** one scenario, built twice. A local counter
  env (an `Env` handle, so its callbacks run on the calling thread) with a
  `Box(4,)` float32 observation is stepped through a session over
  `RLMESH_RANDOM_SAMPLE` with a reused action; the env returns
  one cached observation `Value` and one cached empty `Metadata` each step,
  so the trampoline only clones native references and the env side
  allocates nothing managed. The C# build drives `rlmesh_session_step`
  through the internal raw layer from the runner and wraps the returned
  observation as the public surface would; the C++ build uses the same env
  logic through `rlmesh.hpp`. Both are release builds
  (`RLMESH_RELEASE_BUILD=1`, the C# `Release` configuration) on the .NET 10
  JIT after 1,000 warm-up steps. Measured over 10,000 steps, with
  `GC.Collect()` / `GC.WaitForPendingFinalizers()` checkpoints before and
  after:
  - **Finalizable objects (gating):** a library counter of finalizable
    `SafeHandle` constructions, cumulative, read at the checkpoints, is at
    most 2 per step (the observation returned to the caller and the action
    handed to `IEnv.Step`; R-28 Per-value cost).
  - **Bytes (gating):** `GC.GetTotalAllocatedBytes(precise: true)`, which
    counts every thread, grows by at most 512 bytes per step.
  - **No leak (gating):** after 100,000 steps and a checkpoint, the
    library's live-handle count equals its value before the loop.
  - **Time (advisory):** five batches of 10,000 steps per build, C# and
    C++ batches interleaved back to back under `mise run dev:heavy`; the
    median C# time per step over the median C++ time per step is reported
    against a 1.25× target and recorded, never failing the gate, because
    this host is shared.

**Deferred to v2** (design record, not built): the generated C# defaults
test (`generate_defaults.py`'s C# target emitting
`csharp/tests/RLMesh.Tests/ParityDefaults.g.cs`), and the families that need
the v1-shaped surface: Defaults through the options classes, Borrowing
(a `Model` and env disposed under a session), the twins' Sync and async
behaviour (a sync call from a hook under a twin, 100 sequential
`StepAsync` calls on one worker), Warnings (a throwing C# handler), and
C#-model Errors (one `Model` serving two sessions).

CI: the user approved conditional C# steps in the existing Linux job. The
dotnet SDK is installed through the scoped C# mise config only when the
affected planner selects C# (D-19); no job, runner, matrix entry or trigger
is added to that workflow. D-44's separate dispatch-only GameCI workflow
and its `depot-ubuntu-24.04-8` runner were approved on 2026-10-20. Its
three Unity release gates still require recorded passing engine runs.
