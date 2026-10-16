# Temporary C++ compatibility surface

ABI 5's shared C API uses the facade Env/VectorEnv, Model, PredictContext,
EpisodeResult and RunResult forms. The separate C++20 wave has not replaced
the old `rlmesh::Space` / `SpaceRef`, `Model` / `Batch`, or `Environment` /
`EnvServer` wrappers. The items below are retained only because `rlmesh.hpp`
calls or reads them. They are fenced compatibility items, not a second shared
C authoring API. Do not add new callers.

The header audit in `src/tests.rs` requires the legacy function names in
`rlmesh.h` and `rlmesh.hpp` to match. Remove each orphan when its wrapper
migrates. The C/C++ gate compiles the wrapper surface and runs its old model
and env paths over gRPC, alongside the new C facade tests.

## Space functions retained

These 18 exports still serve the old spec-owning wrappers. New C code uses
Space handles and typed readers. The Experimental constructors also await
facade equivalents; adding them is outside this wave.

| Exact symbol                              | Wrapper dependency                                           |
| ----------------------------------------- | ------------------------------------------------------------ |
| `rlmesh_ext_legacy_space_box`             | `Space::box`, old double-bound spec constructor              |
| `rlmesh_ext_legacy_space_box_elementwise` | `Space::box`, old double-array bounds                        |
| `rlmesh_ext_legacy_space_discrete`        | `Space::discrete`, owned spec constructor                    |
| `rlmesh_ext_legacy_space_dict`            | `Space::dict`, adopts released child specs                   |
| `rlmesh_ext_legacy_space_multi_binary`    | `Space::multi_binary`, Experimental constructor              |
| `rlmesh_ext_legacy_space_multi_discrete`  | `Space::multi_discrete`, Experimental constructor            |
| `rlmesh_ext_legacy_space_text`            | `Space::text`, Experimental constructor                      |
| `rlmesh_ext_legacy_space_tuple`           | `Space::tuple`, adopts child specs; Experimental constructor |
| `rlmesh_ext_legacy_space_box_bounds`      | `SpaceRef::box_bounds`, per-element double bounds            |
| `rlmesh_ext_legacy_space_discrete_n`      | `SpaceRef::discrete`, paired n/start outputs                 |
| `rlmesh_ext_legacy_space_len`             | `SpaceRef::len`, Dict/Tuple child count                      |
| `rlmesh_ext_legacy_space_tuple_get`       | `SpaceRef::tuple_get`, borrowed spec child                   |
| `rlmesh_ext_legacy_space_dict_get`        | `SpaceRef::dict_get`, old C-string key lookup                |
| `rlmesh_ext_legacy_space_dict_get_at`     | `SpaceRef::dict_get_at`, borrowed spec child                 |
| `rlmesh_ext_legacy_space_dict_key`        | `SpaceRef::dict_key`, borrowed key bytes                     |
| `rlmesh_ext_legacy_space_text_length`     | `SpaceRef::text_length`, Experimental limits                 |
| `rlmesh_ext_legacy_space_text_charset`    | `SpaceRef::text_charset`, Experimental charset               |
| `rlmesh_ext_legacy_space_copy_nvec`       | `SpaceRef::nvec`, Experimental counts                        |

Removed: `rlmesh_ext_legacy_space_type`, `_dtype`, `_ndim`, `_copy_shape`.
The wrapper now calls `rlmesh_space_spec_kind`, `_dtype`, and `_shape`.
`rlmesh_space_free` frees only an owned Space handle: the ABI-4 spec-free
fallback and its global pointer registry are gone. Specs use
`rlmesh_space_spec_free`.

## Callback and result records retained

All six are directly read or written by the existing wrapper. Their layouts
remain frozen compatibility layouts, without claiming R-19 sized-record
semantics. Migrating them to the shared ABI-5 forms requires the C++ model/env
wrapper migration, not a rename of an otherwise identical record. No unused
callback/result record remains.

| Exact record                 | Why the current wrapper needs it                                             | ABI-5 replacement path                              |
| ---------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------- |
| `RLMeshExtLegacyEpisode`     | `Batch::episode` reads the fixed-stride episode array                        | Per-corner `RLMeshPredictContext` and episode state |
| `RLMeshExtLegacyObservation` | `Batch` borrows route metadata, observations and that episode array          | Policy vtable observation/context arguments         |
| `RLMeshExtLegacyResetArgs`   | `EnvServer::trampoline_reset` reads seed, trial and JSON options             | Env vtable seed plus typed Value dict options       |
| `RLMeshExtLegacyResetResult` | The reset trampoline writes an owned observation and borrowed JSON info      | Tuple-ordered owned observation/info Value outputs  |
| `RLMeshExtLegacyStepResult`  | The step trampoline writes observation, reward, flags and borrowed JSON info | Tuple-ordered typed Env step outputs                |
| `RLMeshExtLegacyRunReport`   | `Model::run_local` copies scalar aggregates into `RunReport`                 | RunResult handle and EpisodeResult records          |

Do not add `struct_size` to the fixed-stride episode array: R-19 forbids an
array of sized structs. The new Policy ABI already uses per-row context
pointers. The old callback records and their JSON bridge disappear with the
corresponding C++ trampolines.

## Other legacy functions retained

These 11 exports keep the old worker and combined env/server wrappers alive:

| Exact symbol                          | Wrapper dependency                                         |
| ------------------------------------- | ---------------------------------------------------------- |
| `rlmesh_ext_legacy_model_new`         | `Model::create`, old worker callback vtable                |
| `rlmesh_ext_legacy_model_run_local`   | `Model::run_local`, old RunOptions/RunReport               |
| `rlmesh_ext_legacy_model_serve`       | `Model::serve`, old worker server                          |
| `rlmesh_ext_legacy_model_cancel`      | `Model::cancel`, worker cancellation                       |
| `rlmesh_ext_legacy_model_free`        | Model destructor                                           |
| `rlmesh_ext_legacy_env_new_lanes`     | `EnvServer::create_lanes`, owned wrapper callback states   |
| `rlmesh_ext_legacy_env_bind`          | EnvServer bind step                                        |
| `rlmesh_ext_legacy_env_serve`         | EnvServer foreground serving                               |
| `rlmesh_ext_legacy_env_cancel`        | EnvServer cancellation                                     |
| `rlmesh_ext_legacy_env_free`          | EnvServer destructor and callback drain                    |
| `rlmesh_ext_legacy_env_describe_json` | Old `EnvServer::describe_json`; the new server uses COR-16 |

`rlmesh_ext_legacy_env_config_init` is one retained header-only helper, used by
the old EnvServer wrapper to declare its host runtime. These seven supporting
types remain with the wrapper paths above:

- `RLMeshExtLegacyEnv` and `RLMeshExtLegacyModel`: opaque wrapper handles.
- `RLMeshExtLegacyEnvVtable` and `RLMeshExtLegacyModelVtable`: wrapper callback slots.
- `RLMeshExtLegacyEnvConfig`: old space specs, JSON declarations and host runtime.
- `RLMeshExtLegacyRunOptions`: old worker run controls.
- `RLMeshExtLegacyServeOptions`: old model serving controls.

Each of the five sized supporting records retains its matching
`RLMESH_EXT_LEGACY_*_INIT` and `RLMESH_EXT_LEGACY_*_MIN_SIZE` macros (the stems
are `ENV_VTABLE`, `MODEL_VTABLE`, `ENV_CONFIG`, `RUN_OPTIONS`, `SERVE_OPTIONS`).
Their sized inputs already enforce ABI-5 minima and zero unknown tails.
Callback function-pointer declarations are part of those retained vtables.

Removed: the exported scalar `rlmesh_ext_legacy_env_new`; the wrapper only
needs the lane constructor. Scalar compatibility tests and examples now use
one lane. Total remainder: **29 exported legacy functions, one inline helper,
and the six callback/result records above**.
