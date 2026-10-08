# Publishing compute variants

One model or environment version can ship several container images: a PyTorch CUDA build, a ROCm build, a JAX build, a build that only runs on newer GPUs. One image can also run in several ways, such as osmesa or EGL rendering. RLMesh describes both structurally, so publishing them needs no platform-side configuration:

- **A version is an OCI image index.** Its non-attestation children are the **variants**: one image each, each declaring a `variant` block in its `dev.rlmesh.package` label.
- **Profiles are runtime configurations of one image.** They live in that image's label as `profiles[]` and differ only in environment variables, GPU count, resources, and hardware requirements.

`rlmesh registry publish` assembles the index from images you have already pushed. `rlmesh check-image` validates the `variant` and `profiles` blocks before you push, and reports what the platform will infer for an image that declares none. Neither touches the describe envelope or the wire protocol.

## The label schema

Both blocks sit inside the existing `dev.rlmesh.package` JSON label, next to `schemaVersion`, `name`, `tags`, `checkpoints`, and the rest:

```json
{
  "schemaVersion": 1,
  "variant": {
    "key": "cuda12",
    "facets": { "framework": "torch", "accel": "cuda", "render": "egl" },
    "requires": {
      "accel.vendor": "nvidia",
      "accel.compute": ">=8.0,<10.0",
      "accel.cuda": ">=12.4",
      "accel.driver": "550",
      "accel.vram_bytes": 24000000000
    },
    "priority": 10
  },
  "profiles": [
    {
      "key": "osmesa",
      "default": true,
      "facets": { "render": "osmesa" },
      "envVars": { "MUJOCO_GL": "osmesa" }
    },
    {
      "key": "egl",
      "facets": { "render": "egl" },
      "envVars": { "MUJOCO_GL": "egl" },
      "gpu": { "count": 1 }
    }
  ]
}
```

Each block is optional. A `variant` block needs a `key`, and `rlmesh registry publish` requires a `variant` block on every child, because a variant block is how a child is meant to declare itself. An image without one still runs: the platform infers what it needs (see [Inference](#inference-without-a-variant-block)).

| Field                  | Meaning                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `variant.key`          | Required. The variant's name within the version (`cuda12`), matching `^[a-z0-9][a-z0-9-]{0,31}$` (1-32 lowercase letters, digits, or dashes). Unique per version.  |
| `variant.facets`       | What the image is built on. Checkpoints match on facets (`requires: {framework: torch}`), and selection can pin them. Only the three keys in the next table exist. |
| `variant.requires`     | What hardware the image needs (see below). When omitted, the platform infers it from the image's markers.                                                          |
| `variant.priority`     | An integer in `[-1000, 1000]`, default 0. When several variants fit the same hardware, the higher one wins.                                                        |
| `profiles[].key`       | Required. Same pattern as `variant.key`, and unique within the image.                                                                                              |
| `profiles[].default`   | At most one profile is the default. When none is marked, the first one is the default. A request that names no profile runs it.                                    |
| `profiles[].facets`    | Override the variant's facets, key by key.                                                                                                                         |
| `profiles[].envVars`   | String environment variables the profile sets on the container. `RLMESH_ADDRESS` has no effect, since the platform assigns it per pod.                             |
| `profiles[].gpu.count` | GPUs the profile needs, 0 to 8.                                                                                                                                    |
| `profiles[].requires`  | Override the variant's `requires`, key by key: a profile's `accel.cuda` replaces the variant's rather than narrowing it.                                           |
| `profiles[].resources` | Runtime resources, as in the rest of the package label. The platform validates them against its ceilings.                                                          |

Facets:

| Facet       | Values                                                                                |
| ----------- | ------------------------------------------------------------------------------------- |
| `framework` | A lowercase token, `^[a-z0-9][a-z0-9_.-]{0,31}$`: `torch`, `jax`, `tensorflow`, ...   |
| `accel`     | The accelerator stack the image is built for: `cpu`, `cuda`, or `rocm`. Not a vendor. |
| `render`    | The GL backend an environment renders with: `osmesa`, `egl`, or `none`.               |

When a variant block declares both `facets.accel` and `requires`, they must agree: `cuda` with `accel.vendor` `nvidia`, `rocm` with `amd`, `cpu` with no vendor.

`requires` keys are a fixed set, each compared against the hardware the platform probes:

| Key                | Value                                                                     |
| ------------------ | ------------------------------------------------------------------------- |
| `accel.vendor`     | `"nvidia"` or `"amd"`.                                                    |
| `accel.compute`    | NVIDIA compute capability, a version constraint (`">=8.0,<10.0"`).        |
| `accel.cuda`       | The CUDA version the host driver supports, a version constraint.          |
| `accel.driver`     | The NVIDIA driver version, a version constraint (`"550"`, `">=535.104"`). |
| `accel.gfx`        | A non-empty list of AMD GPU targets: `["gfx942", "gfx90a"]`.              |
| `accel.vram_bytes` | The minimum VRAM per GPU, as a JSON integer in bytes: `24000000000`.      |

A version constraint is a string of comma-joined clauses. Each clause is an optional `>=`, `>`, `<=`, `<`, `==`, or `=` followed by a dotted version of up to three parts. A bare version means a minimum, so `"12.4"` is `">=12.4"`. Versions compare numerically part by part, so `12` equals `12.0`. `accel.vram_bytes` is a number, not a string: the platform refuses `">=24000000000"`, and a value it cannot decode costs the image its whole variant declaration. The platform stores it as a signed 64-bit integer, so it must lie between 1 and 9223372036854775807.

Every key other than `accel.vendor` needs `accel.vendor` in the same `requires` object. A profile that adds `accel.cuda` repeats `"accel.vendor": "nvidia"`. The NVIDIA keys (`accel.compute`, `accel.cuda`, `accel.driver`) cannot sit under `amd`, and `accel.gfx` cannot sit under `nvidia`.

### The index annotation

Version-level data that applies to every variant can go on the index itself, as a `dev.rlmesh.package` annotation; `rlmesh registry publish --index-package version.json` sets it. It may carry only `name`, `description`, `checkpoints`, `compatibility`, `capabilities`, and `inputArtifacts`, plus `schemaVersion` (1, the default) and `rev`. The platform ignores any other key, and `publish` warns about it. Where a child's label sets one of those keys differently, the annotation wins, and `publish` warns about that too. Children's `variant` blocks remain the primary declaration: `variant` and `profiles` never belong on the index.

Only an OCI image index carries annotations. docker assembles a Docker manifest list instead when every manifest the sources carry, images and attestations alike, is a Docker schema2 manifest, as a classic `docker build` and `docker push` produce, so `publish` refuses `--index-package` then and names the sources. It also checks the index docker would assemble before pushing, and the pushed one after, and fails if either is not an OCI image index carrying the annotation. Rebuild them with OCI media types: `docker buildx build --push` (its attestations make the push an OCI index), or `--provenance=false --output type=image,oci-mediatypes=true,push=true`. One OCI source among them is enough. Without `--index-package`, schema2 sources publish as a Docker manifest list, which the platform reads as a version too.

A single source that is itself an index, published without `--index-package`, is republished as is, so its own `dev.rlmesh.package` annotation becomes the version's. `publish` checks it by the same rules as `--index-package` (a `schemaVersion` other than 1 fails; other keys are warned about), says that it is kept, and compares it with the child's label. Pass `--index-package` to replace it. With several sources, a source index's annotation is not carried into the version, and `publish` warns that it is dropped.

## Inference without a variant block

An image without a variant block, or a variant block without `requires`, gets its requirements from its own markers, read in this order:

1. `CUDA_VERSION=12.4.1` gives `{"accel.vendor": "nvidia", "accel.cuda": ">=12.4"}` (the major.minor) and `facets.accel` `cuda`.
2. Without it, the `cuda>=X.Y` part of `NVIDIA_REQUIRE_CUDA` gives the same.
3. `ROCM_VERSION` gives `{"accel.vendor": "amd"}` and `facets.accel` `rocm`.
4. Without either marker, a torch build tag in the describe label's `runtime.framework_versions` decides: `2.3.0+cu121` is CUDA 12.1, and `2.3.0+rocm6.0` is ROCm 6.0.
5. Otherwise the image is a CPU image: `facets.accel` `cpu`, no requires.

An image with both CUDA and ROCm markers gets neither an accel facet nor requires. `facets.framework` is inferred from `framework_versions` too (`torch`, then `jax`/`jaxlib`, then `tensorflow`).

A variant block that declares `facets.accel` but no `requires`, for a stack the markers do not show, keeps only the vendor that stack implies. `check-image` reports the inference on its `variant:` line:

```text
ok    variant: none declared; inferred from CUDA_VERSION=12.4.1: facets accel=cuda; requires accel.cuda>=12.4, accel.vendor=nvidia
ok    rows: default
```

## Row keys

The platform records one row per runnable configuration of a version, and an evaluation request names a row. It derives the row keys itself:

| Image                                     | Row key                                                                                                                                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Declares a variant, no profiles           | `<variant.key>`                                                                                                                                                                                 |
| Declares a variant and profiles           | `<variant.key>-<profile.key>`, one per profile (`cuda12-egl`)                                                                                                                                   |
| Profiles but no variant block             | the bare profile key (`egl`)                                                                                                                                                                    |
| An undeclared single image (not an index) | `default`                                                                                                                                                                                       |
| An undeclared child of an index           | `default` when it is the only undeclared child on its platform, else named for its inferred stack: `cuda12`, `rocm6`, `cpu`                                                                     |
| A child that is not `linux/amd64`         | an undeclared child's key gets the platform appended (`default-arm64v8`). Any non-`linux/amd64` child is recorded with status `excluded` and never selected; a non-linux child fails its check. |

A derived row key must still match the key pattern, so a long variant key and a long profile key together can exceed 32 characters. `check-image` and `publish` fail on that, and `publish` fails when two children derive the same row key. A version needs at least one `linux/amd64` child. Its default row is the first `linux/amd64` child, at that child's default profile.

## Build and push each variant

Build each variant for `linux/amd64` and push it under its own tag. The label is easiest to keep in a JSON file per variant, passed in as a build argument, so one Dockerfile serves every variant:

```dockerfile
ARG RLMESH_PACKAGE
LABEL dev.rlmesh.package=${RLMESH_PACKAGE}
```

```bash
docker buildx build --platform linux/amd64 --push -t ns/pi0:v3-cuda12 \
  --build-arg RLMESH_PACKAGE="$(jq -c . variants/cuda12.json)" -f Dockerfile.cuda .
docker buildx build --platform linux/amd64 --push -t ns/pi0:v3-rocm6 \
  --build-arg RLMESH_PACKAGE="$(jq -c . variants/rocm6.json)" -f Dockerfile.rocm .
docker buildx build --platform linux/amd64 --push -t ns/pi0:v3-jax \
  --build-arg RLMESH_PACKAGE="$(jq -c . variants/jax.json)" -f Dockerfile.jax .
```

To check an image before pushing it, build it with `--load` and run `rlmesh check-image ns/pi0:v3-cuda12` (see [below](#what-check-image-validates)).

## Publish the version

```bash
rlmesh registry publish ns/pi0:v3 ns/pi0:v3-cuda12 ns/pi0:v3-rocm6 ns/pi0:v3-jax
```

The first reference is the version (`REPOSITORY:TAG`); the rest are the variants. Publishing:

1. Resolves each source and pins it by digest, so a tag that moves mid-publish cannot swap a child. Each source must be one linux image. A BuildKit push is an index of the image plus its attestation manifests, and the attestations are carried into the version. Each carried attestation must describe an image of the version (its `vnd.docker.reference.digest`); a source whose attestation names another image, or none, fails, and naming that source's image by digest (`ns/pi0@sha256:…`) leaves its attestations behind. When a source is an index, the platform its index declares for the image must match the image config's `os`/`architecture`, since the platform schedules by the one and runs the other.
2. Reads each image's config and runs the version's checks:
   - every child carries a `dev.rlmesh.package` label with a `variant` block that passes the same checks as `check-image`
   - variant keys and derived row keys are unique
   - every child serves the same kind (env or model, from the describe label or the `rlmesh.serve` command)
   - at least one child is `linux/amd64`

   A non-amd64 child is only a warning, since the platform records it as `excluded`. Every check that can run before pushing runs first (these, the assembled index, and step 3), every problem across all sources is reported at once, and nothing is pushed if there is one.

3. Checks where each tag points now, reading each manifest as the registry stores it. TARGET and each `--tag` name the version, so one that already points at a different index is refused unless you pass `--force`. One that already holds this very index (the same media type and `artifactType`, the same manifests in order with every descriptor field, the same annotations and `subject`; only whitespace and key order may differ) is left as is and not pushed again, so publishing again is safe and never moves TARGET's digest (another tag holding it at another digest is re-pointed at TARGET's in step 5). A tag is new only when the registry says it does not exist (under any spelling docker resolves to it, so `ubuntu:v1` and `docker.io/library/ubuntu:v1` are one tag); one that cannot be read (an authorization failure, a transport error, a registry error, output that does not parse) is refused unless you pass `--force`, with the reason. The `--channel` tag moves, which is what it is for, and the summary shows the digest it moves from, or why it could not be read; a channel that is also TARGET or a `--tag` is protected like them.
4. Creates the index with `docker buildx imagetools create` and pushes it to TARGET alone, unless TARGET already holds it, then reads it back as the registry stores it and checks it is the index assembled (and, with `--index-package`, an OCI image index carrying the annotation).
5. Points every other tag that does not already hold TARGET's digest at that digest (`imagetools create --tag TAG REPOSITORY@DIGEST`, which copies the manifest byte for byte), so TARGET, each `--tag`, and the channel are the same manifest, even when TARGET was published earlier with another serialization of the same index.

If the check in step 4 fails, only TARGET has moved; the other tags still point where they did. The error says what TARGET pointed at before and how to recover: put that digest back with `docker buildx imagetools create --tag TARGET REPOSITORY@DIGEST`, or, once the cause is fixed, publish again with `--force`, since TARGET now holds a different index. If pointing the other tags fails, TARGET holds the version and running the same publish again finishes the job.

| Flag                   | Effect                                                                                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--tag TAG`            | Another tag for the index in TARGET's repository; repeatable. Like TARGET, it does not move to a different index, or over a state that cannot be read, without `--force`.           |
| `--channel TAG`        | A moving channel tag such as `latest`. It moves without `--force`, and the summary prints the digest it moved from; when it is also TARGET or a `--tag`, it is protected like them. |
| `--index-package FILE` | Version-level JSON, set as the index's `dev.rlmesh.package` annotation. `schemaVersion` defaults to 1; keys outside the version-level set are kept but warned about.                |
| `--dry-run`            | Prints the per-variant summary, the index JSON, and each tag's state without pushing; fails on a tag the real run would refuse.                                                     |
| `--force`              | Pushes TARGET or a `--tag` that already points at a different index, or that cannot be read.                                                                                        |

The summary shows each child's effective facets and requires (marked `(inferred)` when they come from the image's markers), its row keys, the version's default row (`*`), and where each tag points now:

```text
$ rlmesh registry publish --dry-run --channel latest --index-package version.json \
    ns/pi0:v3 ns/pi0:v3-cuda12 ns/pi0:v3-rocm6 ns/pi0:v3-jax
Dry run: would publish ns/pi0:v3 (3 variants)

VARIANT       KIND   PLATFORM     PRIORITY  FACETS                       REQUIRES                                                   ROWS                 IMAGE
torch-cuda12  model  linux/amd64  10        accel=cuda, framework=torch  accel.compute>=8.0, accel.cuda>=12.4, accel.vendor=nvidia  torch-cuda12*        ns/pi0:v3-cuda12 (56992f07aee3)
torch-rocm6   model  linux/amd64  5         accel=rocm, framework=torch  accel.gfx in [gfx942,gfx90a], accel.vendor=amd             torch-rocm6          ns/pi0:v3-rocm6 (8cfb9d2f40c5)
jax           model  linux/amd64  0         accel=cpu, framework=jax     -                                                          jax-osmesa, jax-egl  ns/pi0:v3-jax (7a4d816428ad)
* the version's default row: the first linux/amd64 child, at its default profile

Index for ns/pi0:v3, ns/pi0:latest (nothing pushed):
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ ... ] }

Tags:
  ns/pi0:v3      new
  ns/pi0:latest  moves from sha256:2c5e1f0a9b7d... (channel)
```

Publishing goes through docker, so it authenticates the way `docker push` does. Against the managed platform's registry, run `rlmesh registry login` first.

## Profiles: osmesa and EGL

A MuJoCo environment that renders headless on a CPU with osmesa, or on an NVIDIA GPU with EGL, is one image with two profiles:

```json
{
  "schemaVersion": 1,
  "variant": { "key": "mujoco", "facets": { "accel": "cpu" } },
  "profiles": [
    {
      "key": "osmesa",
      "default": true,
      "facets": { "render": "osmesa" },
      "envVars": { "MUJOCO_GL": "osmesa" }
    },
    {
      "key": "egl",
      "facets": { "render": "egl" },
      "envVars": { "MUJOCO_GL": "egl" },
      "gpu": { "count": 1 },
      "requires": { "accel.vendor": "nvidia" }
    }
  ]
}
```

The image installs both osmesa and the EGL libraries. A profile switches between them with `MUJOCO_GL` and asks for a GPU only when it needs one, so the osmesa row (`mujoco-osmesa`) can be scheduled anywhere and the EGL row (`mujoco-egl`) lands on NVIDIA hardware.

## What check-image validates

`rlmesh check-image IMAGE` reports `variant:`, `profiles:`, and `rows:` lines next to its other checks, in the same failed / warnings / not checked buckets and the same `--json` shape. The rules are the platform's own.

It **fails** on what the platform's `variant_requires_schema` check fails:

- a block of the wrong type
- a missing or malformed key
- an unknown field or facet, or a facet value outside its vocabulary
- a `priority` that is not an integer in `[-1000, 1000]` (a string such as `"10"` or a fraction fails too)
- an unknown `requires` key, or a malformed constraint
- `accel.vram_bytes` that is not a positive integer within the signed 64-bit range
- `accel.gfx` that is not a list of `gfx…` targets
- a hardware key without `accel.vendor`, or one under the wrong vendor
- `facets.accel` contradicting `accel.vendor`
- duplicate profile keys, or more than one default profile
- `gpu.count` outside 0 to 8
- a derived row key that does not match the key pattern

It **warns** when a declaration contradicts the image's own markers, the cases the platform's `variant_requires_vs_env` check covers. For `accel.cuda` the CLI checks the whole constraint, upper bounds included, so it can warn where the platform does not:

- `accel.vendor` `amd` on a CUDA image, or `nvidia` on a ROCm image
- `accel.cuda` that admits drivers older than the image's CUDA runtime needs, because the constraint has no lower bound (`<13`) or its lower bound sits below the runtime (`>=12.2` on CUDA 12.4); or that admits none that can run it, because an upper bound or an `==` sits below the runtime (`<12`, `==12.2`)
- `facets.accel` naming a stack other than the one the image is built on
- `facets.framework` absent from the describe label's `framework_versions`

It also warns when no profile is marked default (the first one is), when a profile's overrides leave the merged `requires` incoherent (for example, `accel.vendor: amd` over an inherited `accel.cuda`), and on non-portable or platform-assigned `envVars`. With or without a variant block, it reports the effective requires (declared or [inferred](#inference-without-a-variant-block)) and the [row keys](#row-keys) the image gets when pushed on its own.

## What the managed platform does with it

The platform probes each variant of a version on each hardware class it runs, the same admission probe a single image gets. It records which rows run where. Given an evaluation and the hardware it lands on, the platform picks a row automatically:

1. Keep the rows whose `requires` the hardware meets.
2. Keep those whose facets match the requested checkpoint's requirements.
3. Prefer the highest `priority`.

An evaluation request can instead name one explicitly with `variant: <row key>`.
