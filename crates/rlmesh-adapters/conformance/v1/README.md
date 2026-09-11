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

The library parity anchors live in the Python suite (`test_aa_resize_matches_pillow_within_one_step` and `test_zoom_crop_matches_pillow_box_resize_within_one_step`, skipped when Pillow is absent; `test_area_resize_matches_opencv_within_one_step`, skipped when OpenCV is absent), so they are checked continuously rather than only at authoring time.

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

## Cropping

`ImageInput.crop` (a side fraction) or `crop_area` (the same box as an area fraction, side = its square root) keeps a center box of the frame; declaring both is a resolve error, and each is bounded to `(0, 1]` at the wire. `crop_mode` says where the box meets the resize:

- `"zoom"` (default) — the _fractional_ box is handed straight to the resampler above, which samples it onto the target in one pass. Anchored on PIL's `Image.resize(size, box=...)`: `center = box_start + (i + 0.5) * (box_end - box_start) / dst`, taps clamped to the **whole frame**, so the filter still reaches past the box edge exactly as Pillow's does.
- `"slice"` — an _integer_ center box, `round(side * fraction)` pixels (clamped to at least 1), cut before the resize, which then sees only the cut.

`allow_upscale` is measured against the box, not the camera: a crop is what the resize actually reads, so cropping past the target is an upscale.

`channel_order = "bgr"` swaps red and blue after the spatial ops and before the dtype cast, and requires a 3-channel image. The full order is **upright → jpeg → crop → resize → channel swap → normalize/dtype → layout → lead dims**.

Crop, JPEG and channel-order steps are reported by `describe` (`jpeg q95`, `zoom 0.949 (crop 90.0% area)`, `crop 0.667 (slice) -> 320x320`, `bgr`) and carry **no advisory**: they are declared behavior, not a conversion the resolver chose.

## JPEG round-trip

`ImageInput.jpeg_quality` (1-100, the IJG scale) encodes the frame as JPEG and decodes it again, reproducing the codec artifacts a model trained on stored JPEGs saw. It runs on the **upright frame, before the crop and resize** — encoding after a crop would put the 8×8 block grid and the chroma subsampling at the wrong scale. It requires a 3-channel image (JPEG subsampling is defined on YCbCr), which resolution enforces.

**The profile is the contract**, because another engine reproducing `apply_jpeg_q95` has to write the same stream:

- **Baseline sequential** (`SOF0`), 8-bit samples, no restart intervals, no progressive scans.
- **4:2:0 chroma subsampling**: luma sampling factors 2×2, both chroma planes 1×1 — half resolution on each axis. Chroma is reduced by a **box average** over each 2×2 block (libjpeg's `h2v2_downsample`), not by picking a corner sample.
- **The standard IJG quantization tables** (the Annex K luma and chroma tables) scaled by the quality, and **the standard IJG Huffman tables** — not per-image optimized tables.
- **RGB → YCbCr** by the JFIF (BT.601) matrix, the JFIF density defaults, and no embedded ICC or Exif segments.

That is what `tf.io.encode_jpeg(..., quality=q)` and Pillow's `Image.save(..., "JPEG", quality=q)` write by default; the vectors are pinned against the encoder, and the Python suite's `test_jpeg_roundtrip_is_near_pillows_q95_jpeg` anchors it to Pillow (libjpeg) continuously. That anchor is **near**, not exact: the profile agrees, but libjpeg's IDCT and the decoder behind the vectors round differently in the last place, so it is pinned at _within 2 levels on at least 99% of pixels and 4 anywhere_ — on a natural frame it lands inside 2 everywhere. A conforming implementation reproduces `apply_jpeg_q95` exactly (the vectors are encoder-to-decoder within one engine); an engine using a different decoder should expect the same near-anchor tolerance rather than byte identity.

**Composition.** The round-trip composes with every other image step by position, not by special case: it sees the frame after any 180° rotation and before any crop, so `jpeg + crop + resize` (pinned by `resolve_describe_jpeg_steps`) encodes the full camera frame, then the crop and resize read the decoded result — identical to a pipeline that saved a JPEG, reopened it, and cropped. The codec is pinned by version (`jpeg-encoder = "=0.7.1"`), because a retuned encoder is a vector change, not a dependency update.
