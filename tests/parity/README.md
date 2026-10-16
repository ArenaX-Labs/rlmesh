# Shared behavioural parity fixtures

Run `mise run parity:fixtures`. For a repeatable stress check, use
`mise run parity:fixtures -- --repeat 10 --load`. The task takes the machine's
heavy lock once, builds the C ABI library and C11/C++17 consumers once, and runs
pytest plus Rust integration tests for each iteration. `--no-build` reuses the
native consumers while Cargo still checks that Rust binaries are current.

The TOML files describe inputs and expected observations once. An optional
`languages` list marks a language-specific contract (Rust futures, C sizing,
C++ exception identity, Python warnings-as-errors). Every restricted scenario
requires a `why` string, printed in pytest skip reasons. Factory serving with
active or already-fired cancel tokens runs in Rust/C/C++; dropping serve/wait
futures is Rust-only because the other APIs expose synchronous calls. No runner
receives expected values as inputs. Pytest compares Python and native JSON observations;
`parity_fixtures.rs` loads a lossless JSON projection of the same TOML and
compares every specified field. Unknown operations and missing result fields
fail. A native process has a 30-second deadline. Native serving announcements
may precede the final JSON observation on stdout.

All four scripts implement the same counter env (reset to zero, two steps per
episode, reward one) and action-zero model. They count env/model call starts,
closes, and episode ends. Ordinary scenarios instantiate these through each
language's public authoring API. Remote borrowing uses an actual ephemeral
loopback server. Rust's nested remote-model scenario binds its test endpoint
through binding plumbing so it can keep the OS-assigned listener, with no port
reservation race. Cancellation/drop synchronization uses channels, observable
state and bounded completion waits, never fixed ports or sleeps. The optional
stress worker occupies one CPU and is terminated and joined even on failure.

`test:ci` depends on this gate. `test:rust` excludes the two parity integration
binaries so the fixture and generated-default tests run once in CI. `test:cxx` also compiles and runs its native
scenarios with each selected compiler at C++17 and C++20, using the same flags
as the existing examples. CI's existing setup provides Rust, Python 3.11, Zig
and pytest; no additional packages, jobs, triggers or runners are needed.
Compiler scratch stays in `target/`; pytest receives no checkout-local TMPDIR.

`generate_defaults.py` emits `crates/rlmesh/tests/parity_defaults.rs` directly
from `docs/specs/language-parity/manifest.toml`. It constructs every mapped Rust data type with defaults (including View, adapters
and recorder records), and checks all distinct mapped Python defaults, including `None` callbacks
that cannot implement equality. The fixture gate runs its `--check`; use
`python tests/parity/generate_defaults.py` after an intentional manifest update.
Generator unit tests compare the literals to live Python signatures and reject
conflicting duplicate mappings or unsupported literals. The schema is `[symbol.parameters.keyword].default` for Python literals
and `[symbol.rust].params` with `Type#field` targets. A whole options argument
can map to a method parameter with its wrapper field inventoried in `also`.
Callable defaults without a field require an explicit `default_skip` reason.
Required keywords and native-only fields with no Python default (including
cancellation/extensions and DLPack bridge options) are outside this comparison.

C ABI 5 currently makes `_MIN_SIZE` equal to the full known struct size, so its
minimum-sized fixture supplies `_INIT` for all known fields. Extended input,
out-struct and lent-reader cases exercise unknown appended fields without
inventing a product field or ABI generation just for a test.

For manifest keys, rule citations, and failure repair, see
[When parity:check or parity:fixtures fails](../../docs/testing.md#when-paritycheck-or-parityfixtures-fails).
