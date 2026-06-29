# Describe Envelope — `rlmesh.describe.v1`

The **describe envelope** is the single, self-contained JSON artifact that fully
describes an environment factory or a model: its parameters, variants, IO
contract, obs/action spaces, and the runtime it was generated under. It is
generated once (build/generate-time), uploaded to a managed service, listed in a
dashboard, and is forward-compatible with being baked into an OCI image label.

This document is the **cross-language contract**. The format — the field set,
versioning, ordering, and serialization — is owned by the Rust crate
`rlmesh-adapters` (`build_describe_envelope`). Any producer (the Python SDK
today; a future C++ or TypeScript SDK) emits a byte-identical envelope for the
same logical input by handing its gathered pieces to that one builder, or by
implementing this contract exactly.

## Versioning

- `schema_version` is an integer, **stamped by the builder** — a producer never
  sets it. It is `1` today.
- The wire discriminant is the metadata key **`rlmesh.describe.v1`** (the
  `DESCRIBE_METADATA_KEY` constant). Within `v1` the format evolves **additively
  only**: new optional fields with defaults. A breaking restructure ships under a
  new key (`rlmesh.describe.v2`) and bumps `schema_version`; a v2 reader keeps
  reading v1.
- Serialization is canonical: the builder serializes the whole tree through one
  `serde_json` pass, and object keys sort (`BTreeMap`) — so the bytes do not
  depend on the producer's language or JSON formatting.

## Layers — who produces what

| Concern                                                 | Owner                             | Notes                                                                                                                                                                        |
| ------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`, `kind`, ordering, final serialization | **Rust builder**                  | Stamped/validated/serialized once; no producer can disagree.                                                                                                                 |
| `generated_at`                                          | producer-supplied, Rust-validated | RFC-3339; omit for a content-addressable artifact (do **not** use wall-clock in a reproducible build).                                                                       |
| `target`, `params`, `variants`, `env_spec`, `runtime`   | **per-language gatherer**         | Requires introspecting/executing the producer's own language (signature reflection, running the author's variant enumeration, constructing the env, reading local versions). |
| `env_tags`, `model_spec`, `env_spec.*` space dicts      | shared codecs                     | Already canonical from their own serializers (`EnvTags`/`ModelSpec`/`SpaceSpec`); embedded as-is.                                                                            |

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

- `env_spec` is captured from **one representative** constructed env (an
  `EnvFactory` is single-shape by its one `env_tags` contract; variants share
  spaces). For a vectorized env it carries `single_*` spaces plus `num_envs`.
- `env_spec.observation_space` / `action_space` are the `SpaceSpec` JSON form.

### Model (`kind: "model"`)

Same wrapper; drops `env_spec`/`env_tags`, adds `model_spec`:

```text
{
  "schema_version": 1,
  "kind": "model",
  "target": { ... },
  "model_spec": { "input": { ... }, "output": { ... } } | null,
  "params": { ... },
  "variants": { ... },
  "runtime": { ... }
}
```

## Best-effort / error badges

Any gathered piece that fails (an env that needs a GPU to build, a `make()` that
needs unavailable args, a model spec that can't be published, a broken
`enumerate_*`) is replaced by an `{"error": "<message>"}` badge in place of that
field (`env_spec`, `model_spec`) or as a sibling `*_error` key (`catalog_error`,
`variations_error`). The envelope is **always** emitted — a no-GPU build still
produces a useful artifact with `env_spec: {"error": ...}`.

## Invariants enforced by the builder

- `kind` is a closed enum (`"env"` | `"model"`); anything else is rejected.
- Env-only fields (`env_spec`, `env_tags`) never appear on a `model` envelope,
  and `model_spec` never appears on an `env` envelope.
- Unknown top-level fields are rejected (the key set is part of the contract).
- `generated_at`, if present, must be RFC-3339.
