# Adapter spec conformance vectors — v1

These vectors freeze the `v1` semantics of `rlmesh.adapters`: the JSON spec format, resolution outcomes (including exact error conditions), and the numeric behavior of plan application. Every implementation of the adapter core — the reference Python package today, the `rlmesh-adapters` Rust crate, and every language binding — must pass all cases in `cases/`.

## Versioning

The format follows protobuf-style package versioning. Specs travel under the metadata keys `rlmesh.adapters.v1.env_io_spec` / `rlmesh.adapters.v1.model_io_spec`. Within `v1`, changes are additive only: new optional fields with defaults that old readers may ignore and old writers may omit. Anything else is a `v2`: new keys, a new vectors directory, and publishers may dual-publish both versions during migration.

## Case format

One JSON file per case, dispatched on `kind`:

- `resolve` — `env_spec` + `model_spec`, expecting either `{"ok": true, "describe": <exact text>}` or `{"error_contains": <substring>}`. Error cases pin _resolve-time_ failure: an implementation that defers the failure to apply time fails the case.
- `serialization` — `side` (`env`|`model`) + `doc`: `from_dict(doc)` followed by `to_dict()` must reproduce `doc` exactly.
- `apply` — specs + `observation` + `model_output`, expecting the exact model payload and env action. Values are encoded as `{"kind": "array", dtype, shape, data}`, `{"kind": "list", data}`, `{"kind": "text", data}`, or `{"kind": "map", data}` (nested observations). Numeric comparison: exact dtype match, values within `atol` (default 1e-6).

## Updating (snapshot-style)

Expectations are machine-written, human-reviewed:

    UPDATE_VECTORS=1 cargo test -p rlmesh-adapters

This rewrites each case's `expect` block (and normalizes spec documents, so new defaulted fields propagate) from current behavior. Review the diff before committing — a changed vector is a semantic change to `v1` and must be additive: never delete a case within v1, and change expectations only as a deliberate, reviewed decision. Hand-curated `error_contains` substrings are kept as long as they still match.

To add a case: write the inputs by hand (specs, observation, model_output) with an empty `"expect": {}`, then run update mode once.

Cases with `"preserve_inputs": true` keep their spec documents verbatim across update runs. This is for defaults-pinning cases (e.g. `apply_minimal_spec_defaults`): their specs deliberately omit every optional field, so the expectations pin the missing-field defaults of every implementation — Python's `from_dict` and the core's serde must agree or the case fails on one side.

The PIL parity anchor for `bilinear_aa` lives in the Python suite (`test_bilinear_aa_resize_matches_pillow_within_one_step`, skipped when Pillow is absent), so it is checked continuously rather than only at authoring time.

## Resize algorithms

`ImageInput.resample` declares which of the two pinned resize algorithms the model's training pipeline used; resolution rejects anything else with a typed error (`resample` is a constrained string, not an enum, so future additive values degrade to a resolution error on older cores rather than a parse failure):

- `"bilinear"` — 4-tap bilinear with half-pixel centers (OpenCV/torch-compatible).
- `"bilinear_aa"` (default) — antialiased separable triangle filter: on downscale the filter support widens by the scale factor. PIL-compatible; the generator asserts parity within one uint8 step of real Pillow.

Both are specified as: weights computed in float64, both passes in float64, one final round-half-to-even, clip to [0, 255], uint8. Resize apply cases use `atol: 1.0` (one uint8 step) to absorb cross-language rounding at ties; all other apply cases use `atol: 1e-6`.
