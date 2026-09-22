# Describe Envelope: `rlmesh.describe.v1`

The **describe envelope** is the single, self-contained JSON artifact that fully
describes an environment factory or a model: its parameters, variants, IO
contract, obs/action spaces, and the runtime it was generated under. It is
generated once (build/generate-time), uploaded to the managed platform, listed in
a dashboard, and is forward-compatible with being baked into an OCI image label.

This document is the **cross-language contract**. The format (the field set,
versioning, ordering, and serialization) is owned by the Rust crate
`rlmesh-adapters` (`build_describe_envelope`). Any producer (the Python SDK
today; a future C++ or TypeScript SDK) emits a byte-identical envelope for the
same logical input by handing its gathered pieces to that one builder, or by
implementing this contract exactly.

## Versioning

- `schema_version` is an integer, **stamped by the builder**: a producer never
  sets it. It is `1` today.
- The wire discriminant is the metadata key **`rlmesh.describe.v1`** (the
  `DESCRIBE_METADATA_KEY` constant). Within `v1` the format evolves **additively
  only**: new optional fields with defaults. A breaking restructure ships under a
  new key (`rlmesh.describe.v2`) and bumps `schema_version`; a v2 reader keeps
  reading v1.
- Serialization is canonical: the builder serializes the whole tree through one
  `serde_json` pass, and object keys sort (`BTreeMap`), so the bytes do not
  depend on the producer's language or JSON formatting.

## Layers: who produces what

| Concern                                                                | Owner                             | Notes                                                                                                                                                                        |
| ---------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`, `kind`, ordering, final serialization                | **Rust builder**                  | Stamped/validated/serialized once; no producer can disagree.                                                                                                                 |
| `generated_at`                                                         | producer-supplied, Rust-validated | RFC-3339; omit for a content-addressable artifact (do **not** use wall-clock in a reproducible build).                                                                       |
| `target`, `params`, `variants`, `env_spec`, `env_contracts`, `runtime` | **per-language gatherer**         | Requires introspecting/executing the producer's own language (signature reflection, running the author's variant enumeration, constructing the env, reading local versions). |
| `env_tags`, `model_spec`, `env_spec.*` space dicts                     | shared codecs                     | Already canonical from their own serializers (`EnvTags`/`ModelSpec`/`SpaceSpec`); embedded as-is.                                                                            |

A producer's _only_ language-specific job is the gathering. Everything about the
_format_ is shared.

## Envelope shape

### Environment (`kind: "env"`)

```text
{
  "schema_version": 1,
  "kind": "env",
  "target": { "entrypoint": "mypkg.envs:Libero" | null, "qualname": "mypkg.envs:Libero" },
  "generated_at": "2026-06-28T19:30:00Z",
  "env_spec": {
    "observation_space": { ... },
    "action_space": { ... },
    "num_envs": 8
  },
  "env_tags": { ... } | null,
  "env_contracts": {
    "discriminants": ["action_type"],
    "branches": [
      { "params": { "action_type": "delta" }, "default": true, "env_tags": { ... }, "env_spec": { ... } },
      { "params": { "action_type": "abs" },   "default": false, "env_tags": { ... }, "env_spec": { ... } }
    ]
  },
  "params": {
    "param_spec": { "params": [ ... ], "extra": "forbid" } | null,
    "signature_tier": [ { "name": "...", "type": "...", "default": ..., "required": false } ]
  },
  "variants": {
    "catalog": [ { "id": "libero_10/0", "params": { ... }, "metadata": { ... } } ],
    "variations": { "seed": [0, 1, 2] }
  },
  "runtime": {
    "component": "rlmesh-python",
    "language": "python",
    "language_version": "3.11.8",
    "package_version": "0.1.0",
    "os": "linux", "os_version": "...", "arch": "x86_64",
    "framework_versions": { "numpy": "...", "torch": "..." }
  }
}
```

- `env_spec` is captured from **one representative** constructed env: one per
  _contract branch_. A factory's variants share spaces, so a factory with no
  declared contract discriminants has exactly one shape; a factory that declares
  them (`tag_params`) has one per branch, and the top-level `env_spec`/`env_tags`
  are the **default branch's**. For a vectorized env it carries `single_*` spaces
  plus `num_envs`.
- `env_spec.observation_space` / `action_space` are the `SpaceSpec` JSON form.
  The envelope is strict JSON, so a non-finite number anywhere in it is `null`:
  a `null` in a Box `low`/`high` means unbounded on that edge (`-inf` under
  `low`, `+inf` under `high`).
- `env_contracts` appears **only** for a factory that declares contract
  discriminants, so an envelope emitted for a single-contract env is byte-identical
  to one emitted before the field existed. It is self-describing: `discriminants`
  names the axes, and every branch carries its full binding (never a subset) plus
  the `env_tags`/`env_spec` that binding produces -- so a reader that cannot run
  the producer's code can still say which branch a contract belongs to, and a
  static check can name the branch it validated. Exactly one branch has
  `default: true`, and its `env_tags`/`env_spec` are the top-level ones.

### Model (`kind: "model"`)

Same wrapper; drops `env_spec`/`env_tags`, adds `model_spec`, `corners`, and
`native_chunk`:

```text
{
  "schema_version": 1,
  "kind": "model",
  "target": { ... },
  "model_spec": { "input": { ... }, "output": { ... } } | null,
  "corners": ["predict", "predict_chunk_batch"],
  "native_chunk": 30,
  "params": { ... },
  "variants": { ... },
  "runtime": { ... }
}
```

- `corners` lists the predict corners the model class actually defines, drawn
  from the closed set `["predict", "predict_chunk", "predict_batch",
"predict_chunk_batch"]` and reported in that order (general -> specific). It is
  **introspected, not declared**, so a packaging claim like `supportsBatching` can
  be checked against the code that ships. Omitted when the producer resolved no
  corner at all (a duck-typed callable it could not introspect); an empty list is
  never emitted.
- `native_chunk` is the model's own declaration of its chunk length K: how many
  per-step actions one chunk-corner call returns. Omitted for the elastic
  (undeclared) contract, in which the runtime takes the `min(len, h)` prefix of
  whatever comes back. It is an integer (`bool` is not accepted), read off the
  class, so a K a model only sets while loading its weights is deliberately
  invisible here -- describe runs without weights.

## Best-effort / error badges

Any gathered piece that fails (an env that needs a GPU to build, a `make()` that
needs unavailable args, a model spec that can't be published, a broken
`enumerate_*`) is replaced by an `{"error": "<message>"}` badge in place of that
field (`env_spec`, `model_spec`) or as a sibling `*_error` key (`catalog_error`,
`variations_error`). The envelope is **always** emitted: a no-GPU build still
produces a useful artifact with `env_spec: {"error": ...}`.

## Invariants enforced by the builder

- `kind` is a closed enum (`"env"` | `"model"`); anything else is rejected.
- Env-only fields (`env_spec`, `env_tags`, `env_contracts`) never appear on a
  `model` envelope, and the model-only fields (`model_spec`, `corners`,
  `native_chunk`) never appear on an `env` envelope. The builder rejects a
  misplaced field by name rather than dropping it.
- Unknown top-level fields are rejected (the key set is part of the contract).
- `generated_at`, if present, must be RFC-3339.
