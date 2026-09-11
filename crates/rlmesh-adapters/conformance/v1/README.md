# Adapter spec conformance vectors — v1

These vectors freeze the `v1` semantics of `rlmesh.adapters`: the JSON spec format, resolution outcomes (including exact error conditions), and the numeric behavior of plan application. Every implementation of the adapter core — the reference Python package today, the `rlmesh-adapters` Rust crate, and every language binding — must pass all cases in `cases/`.

## Versioning

The format follows protobuf-style package versioning. Specs travel under the metadata keys `rlmesh.adapters.v1.env_io_spec` / `rlmesh.adapters.v1.model_io_spec`. Within `v1`, changes are additive only: new optional fields with defaults that old readers may ignore and old writers may omit. Anything else is a `v2`: new keys, a new vectors directory, and publishers may dual-publish both versions during migration.

## Case format

One JSON file per case, dispatched on `kind`:

- `resolve` — `env_spec` + `model_spec`, expecting either `{"ok": true, "describe": <exact text>}` or `{"error_contains": <substring>}`. Error cases pin _resolve-time_ failure: an implementation that defers the failure to apply time fails the case. An optional `advisories_contain` lists substrings each of which must appear in some `"<severity>: <message>"` advisory line — hand-curated like `error_contains` (update mode carries it through rather than rewriting it), so a case pins only the advisory it is about.
- `serialization` — `side` (`env`|`model`) + `doc`: `from_dict(doc)` followed by `to_dict()` must reproduce `doc` exactly.
- `role_policy` — `side` (`env`|`model`) + `policy` (`passthrough`|`strict`|`forbid`) + `doc`: the publish-gate role tier, expecting acceptance (`{}`) or `{"error_contains": <substring>}`. Frozen like `serialization` — the policy table _is_ the contract, so update mode never rewrites these.
- `apply` — specs + `observation` + `model_output`, expecting the exact model payload and env action. Values are encoded as `{"kind": "array", dtype, shape, data}`, `{"kind": "list", data}`, `{"kind": "text", data}`, or `{"kind": "map", data}` (nested observations). Numeric comparison: exact dtype match, values within `atol` (default 1e-6).

## Updating (snapshot-style)

Expectations are machine-written, human-reviewed:

    UPDATE_VECTORS=1 cargo test -p rlmesh-adapters

This rewrites each case's `expect` block (and normalizes spec documents, so new defaulted fields propagate) from current behavior. Review the diff before committing — a changed vector is a semantic change to `v1` and must be additive: never delete a case within v1, and change expectations only as a deliberate, reviewed decision. Hand-curated `error_contains` substrings are kept as long as they still match.

To add a case: write the inputs by hand (specs, observation, model_output) with an empty `"expect": {}`, then run update mode once.

Cases with `"preserve_inputs": true` keep their spec documents verbatim across update runs. This is for defaults-pinning cases (e.g. `apply_minimal_spec_defaults`): their specs deliberately omit every optional field, so the expectations pin the missing-field defaults of every implementation — Python's `from_dict` and the core's serde must agree or the case fails on one side.

The library parity anchors live in the Python suite (`test_aa_resize_matches_pillow_within_one_step`, skipped when Pillow is absent; `test_area_resize_matches_opencv_within_one_step`, skipped when OpenCV is absent), so they are checked continuously rather than only at authoring time.

## Resize algorithms

`ImageInput.resample` declares which pinned resize algorithm the model's training pipeline used; resolution rejects anything else with a typed error (`resample` is a constrained string, not an enum, so future additive values degrade to a resolution error on older cores rather than a parse failure).

The names follow one rule: **un-suffixed is cv2/torch semantics, `_aa` is PIL semantics** — an antialiased filter whose support widens with the downscale factor. Bare `"bicubic"` and `"lanczos3"` are deliberately _not_ names, because the two libraries' kernels differ (cv2/torch cubic uses a = -0.75, PIL a = -0.5) and a spec that names one without saying which library must fail rather than silently get the other.

- `"bilinear"` (default) — 4-tap bilinear with half-pixel centers, no antialiasing (OpenCV `INTER_LINEAR` / torch `interpolate(antialias=False)`).
- `"bilinear_aa"` — PIL `BILINEAR`: separable triangle filter, support 1.
- `"bicubic_aa"` — PIL `BICUBIC`: separable Keys cubic with a = -0.5, support 2.
- `"lanczos3_aa"` — PIL `LANCZOS`: separable sinc windowed by a 3-lobe sinc, support 3.
- `"area"` — OpenCV `INTER_AREA`: the exact average of each output pixel's source footprint. Unlike the `_aa` filters it does not widen a fixed kernel; the footprint _is_ the kernel, so upscaling splits a sub-pixel span across the one or two source pixels it covers rather than interpolating.

The `_aa` filters share one weight builder: per output pixel, `center = (i + 0.5) * scale`, filter stretched by `max(scale, 1)`, taps snapped to the nearest pixel centers, weights normalized to sum to 1. `"area"` uses the same builder with the box integrated over each source pixel (so a partly covered edge pixel gets exactly its coverage) and a filter stretch of `scale` in both directions.

All of them are specified as: weights computed in float64, both passes in float64, the horizontal pass's output clipped to [0, 255] before the vertical pass — PIL's intermediate is an 8-bit image, so the negative lobes of the cubic and Lanczos kernels are clipped there, and a pipeline that clips only at the end drifts from Pillow by tens of levels on a hard edge — then one final round-half-to-even, clip to [0, 255], uint8. Resize apply cases use `atol: 1.0` (one uint8 step) to absorb cross-language rounding at ties; all other apply cases use `atol: 1e-6`.
