# Publishing compute variants

One model or environment version can ship several container images, called **builds** or **variants**: a CUDA build and a CPU fallback, a build for older drivers and one for newer GPUs. The RLMesh platform picks the build that fits the hardware an evaluation lands on.

Who needs what:

- **You publish a single image.** Nothing new. Push it as before; the platform reads what it needs from the image's own CUDA or ROCm markers (see [Inference](#inference-without-a-variant-block)).
- **You run evaluations.** Nothing. Selection is automatic, and a request can optionally [pin a build](#how-the-platform-picks-a-build).
- **You publish two or more builds of one version.** Read the six concepts below. Everything after them is [Advanced](#advanced) and optional.

## The six concepts

1. **Build key.** Each build names itself with `variant.key` (`cuda12`, `cpu`).
2. **Requirements.** Each build says what hardware it needs in `variant.requires`: `accel.vendor`, `accel.compute`, `accel.cuda`, `accel.driver`, `accel.vram`.
3. **Constraint syntax.** Versions are constraints such as `">=12.4"`; VRAM is a quantity such as `"16Gi"`.
4. **One version is one OCI index.** The builds are pushed separately, then published together as one image index under one tag.
5. **A default build with fallback.** The first build is the default. The others are fallbacks for hardware the default cannot run on.
6. **Check, publish, promote.** `rlmesh check-image` each build, `rlmesh registry publish` the version, then move a channel such as `latest` onto it.

The walk-through below publishes `ns/pi0:v3` with two builds: `cuda12`, the default, for NVIDIA GPUs with at least 16Gi of VRAM, and `cpu`, which runs anywhere.

### 1. Build key

Each build declares a `variant` block inside its `dev.rlmesh.package` image label, next to the label's other keys (`schemaVersion`, `name`, `checkpoints`, ...). The one required field is `key`:

```json
{ "schemaVersion": 1, "variant": { "key": "cpu" } }
```

A key is 1-32 lowercase letters, digits, or dashes, starting with a letter or digit (`^[a-z0-9][a-z0-9-]{0,31}$`), and unique within the version. It is how the platform, its reports, and an optional pin name the build.

### 2. Requirements

`requires` lists what the hardware must offer. A build without it runs anywhere (or gets requirements [inferred](#inference-without-a-variant-block) from its CUDA markers):

```json
{
  "schemaVersion": 1,
  "variant": {
    "key": "cuda12",
    "requires": {
      "accel.vendor": "nvidia",
      "accel.compute": ">=8.0",
      "accel.cuda": ">=12.4",
      "accel.vram": "16Gi"
    }
  }
}
```

| Key             | Value                                                                     |
| --------------- | ------------------------------------------------------------------------- |
| `accel.vendor`  | `"nvidia"` or `"amd"`. Every other key needs it in the same `requires`.   |
| `accel.compute` | NVIDIA compute capability, a version constraint (`">=8.0,<10.0"`).        |
| `accel.cuda`    | The CUDA version the host driver supports, a version constraint.          |
| `accel.driver`  | The NVIDIA driver version, a version constraint (`"550"`, `">=535.104"`). |
| `accel.vram`    | The minimum VRAM per GPU, as a quantity string: `"16Gi"`.                 |

AMD builds use `accel.gfx` instead of the NVIDIA keys; see [AMD and ROCm](#amd-rocm-and-accelgfx).

### 3. Constraint syntax

A version constraint is a string of comma-joined clauses that must all hold. Each clause is an optional `>=`, `>`, `<=`, `<`, `==`, or `=` followed by a dotted version of up to three parts. A bare version means a minimum, so `"12.4"` is `">=12.4"`, and `">=8.0,<10.0"` is a range. Versions compare numerically part by part, so `12` equals `12.0`.

`accel.vram` is a quantity, the way Kubernetes writes memory: digits, an optional fraction, and an optional suffix, `Ki`, `Mi`, `Gi`, `Ti` (powers of 1024) or `K`, `M`, `G`, `T` (powers of 1000), no suffix meaning bytes. `"16Gi"`, `"24G"`, `"1.5Gi"`, and `"80000000000"` are quantities. It is always a minimum, so it takes no comparator (`">=24Gi"` fails), and it has no exponent (`"1e9"`), no milli suffix (`"100m"`), no space (`"24 Gi"`), and no other unit (`"24GB"`, `"24gi"`). It must be a positive whole number of bytes (`"1.5"` fails, `"1.5Gi"` does not) that fits a signed 64-bit integer. A JSON number such as `24` fails; write it as a quantity string. A value the platform cannot decode costs the image its whole variant declaration.

### 4. One version is one OCI index

Build each variant for `linux/amd64` and push it under its own tag. Keeping each label in a JSON file, passed in as a build argument, lets one Dockerfile serve every build:

```dockerfile
ARG RLMESH_PACKAGE
LABEL dev.rlmesh.package=${RLMESH_PACKAGE}
```

```bash
docker buildx build --platform linux/amd64 --push -t ns/pi0:v3-cuda12 \
  --build-arg RLMESH_PACKAGE="$(jq -c . variants/cuda12.json)" -f Dockerfile.cuda .
docker buildx build --platform linux/amd64 --push -t ns/pi0:v3-cpu \
  --build-arg RLMESH_PACKAGE="$(jq -c . variants/cpu.json)" -f Dockerfile.cpu .
```

`rlmesh registry publish` then assembles them into one OCI image index, the version, tagged `ns/pi0:v3`. Its children are the builds; the per-build tags are only how you hand them to `publish`.

### 5. A default build with fallback

The first build listed is the version's default. Put the build you want wherever it can run first, and add the others for hardware it cannot run on. In the example, an NVIDIA node with 16Gi of VRAM or more runs `cuda12`; a CPU-only node, or a GPU with less VRAM, falls back to `cpu`. Where both fit, the platform prefers the default, so a fallback only runs where the default cannot (see [selection](#how-the-platform-picks-a-build)).

### 6. Check, publish, promote

**Check** each build before pushing it: build it with `--load` and run

```bash
rlmesh check-image ns/pi0:v3-cuda12
```

It validates the `variant` block by the platform's rules and warns where it contradicts the image (an `accel.cuda` older than the image's CUDA runtime, say). See [What check-image validates](#what-check-image-validates).

**Publish** the version once both builds are pushed. `--dry-run` shows the plan without pushing:

```bash
rlmesh registry publish --dry-run ns/pi0:v3 ns/pi0:v3-cuda12 ns/pi0:v3-cpu
rlmesh registry publish ns/pi0:v3 ns/pi0:v3-cuda12 ns/pi0:v3-cpu
```

The first reference is the version (`REPOSITORY:TAG`); the rest are the builds, default first. Every build must carry a `variant` block, and every check runs before anything is pushed. Publishing authenticates the way `docker push` does; against the RLMesh platform's registry, run `rlmesh registry login` first.

**Promote** the version by moving a channel tag onto it, once the platform has probed each build and the results look right. Run the same publish with `--channel`:

```bash
rlmesh registry publish --channel latest ns/pi0:v3 ns/pi0:v3-cuda12 ns/pi0:v3-cpu
```

`ns/pi0:v3` already holds this index, so it is left as is and only `latest` moves; the summary prints the digest it moved from, which is what you point it back at to roll back. Passing `--channel` on the first publish promotes immediately.

## How the platform picks a build

The RLMesh platform probes each build of a version on each hardware class it runs, and records where each one runs. For an evaluation and the hardware it lands on, it picks a build in this order:

1. **An explicit pin wins.** A request that names a build (`variant: cuda12`) gets that one.
2. **Drop incompatible and failed builds:** those whose `requires` the hardware does not meet, and those whose probe failed on it.
3. **Prefer verified builds,** those whose probe passed on this hardware, over ones not yet probed there.
4. **Prefer the default build.**
5. **Fall back to a stable order,** so the same request on the same hardware always gets the same build.

Most evaluations never pin. Pin only to compare builds, or to reproduce a run on one build.

## Advanced

Nothing here is needed for the two-build path above.

### Full label schema

The `variant` block has fields beyond `key` and `requires`, and an image can also list runtime **profiles**: several ways to run the same image that differ only in environment variables, GPU count, resources, and hardware requirements. Both sit in the `dev.rlmesh.package` label:

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
      "accel.vram": "24Gi"
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
| `variant.requires`     | What hardware the image needs. When omitted, the platform infers it from the image's markers.                                                                      |
| `variant.priority`     | An integer in `[-1000, 1000]`, default 0. See [Priority](#priority).                                                                                               |
| `profiles[].key`       | Required. Same pattern as `variant.key`, and unique within the image.                                                                                              |
| `profiles[].default`   | At most one profile is the default. When none is marked, the first one is the default. A request that names no profile runs it.                                    |
| `profiles[].facets`    | Override the variant's facets, key by key.                                                                                                                         |
| `profiles[].envVars`   | String environment variables the profile sets on the container. `RLMESH_ADDRESS` has no effect, since the platform assigns it per pod.                             |
| `profiles[].gpu.count` | GPUs the profile needs, 0 to 8.                                                                                                                                    |
| `profiles[].requires`  | Override the variant's `requires`, key by key: a profile's `accel.cuda` replaces the variant's rather than narrowing it.                                           |
| `profiles[].resources` | Runtime resources, as in the rest of the package label. The platform validates them against its ceilings.                                                          |

### Facets

| Facet       | Values                                                                                |
| ----------- | ------------------------------------------------------------------------------------- |
| `framework` | A lowercase token, `^[a-z0-9][a-z0-9_.-]{0,31}$`: `torch`, `jax`, `tensorflow`, ...   |
| `accel`     | The accelerator stack the image is built for: `cpu`, `cuda`, or `rocm`. Not a vendor. |
| `render`    | The GL backend an environment renders with: `osmesa`, `egl`, or `none`.               |

When a variant block declares both `facets.accel` and `requires`, they must agree: `cuda` with `accel.vendor` `nvidia`, `rocm` with `amd`, `cpu` with no vendor. A checkpoint that requires a facet (`framework: torch`) makes a build without it incompatible, so it is dropped at step 2 of [selection](#how-the-platform-picks-a-build).

### AMD, ROCm, and `accel.gfx`

| Key         | Value                                                        |
| ----------- | ------------------------------------------------------------ |
| `accel.gfx` | A non-empty list of AMD GPU targets: `["gfx942", "gfx90a"]`. |

Every key other than `accel.vendor` needs `accel.vendor` in the same `requires` object. A profile that adds `accel.cuda` repeats `"accel.vendor": "nvidia"`. The NVIDIA keys (`accel.compute`, `accel.cuda`, `accel.driver`) cannot sit under `amd`, and `accel.gfx` cannot sit under `nvidia`. A ROCm build is declared like a CUDA one:

```json
{
  "schemaVersion": 1,
  "variant": {
    "key": "rocm6",
    "facets": { "framework": "torch", "accel": "rocm" },
    "requires": { "accel.vendor": "amd", "accel.gfx": ["gfx942", "gfx90a"] }
  }
}
```

### Priority

`variant.priority` is an integer in `[-1000, 1000]`, default 0. Among compatible, verified builds, a higher priority is preferred before the default is. Leave it at 0 unless a non-default build should beat the default on hardware where both fit; a default plus fallbacks never needs it.

### Profiles and rendering: osmesa and EGL

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

The image installs both osmesa and the EGL libraries. A profile switches between them with `MUJOCO_GL` and asks for a GPU only when it needs one. The default profile's row (`mujoco-osmesa`) runs unless a request pins the EGL row (`variant: mujoco-egl`), which lands on NVIDIA hardware.

### Inference without a variant block

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

### Row keys

The platform records one row per runnable configuration of a version, and a pin names a row. Without profiles, a row is a build and its key is the build key. The platform derives the row keys itself:

| Image                                     | Row key                                                                                                                                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Declares a variant, no profiles           | `<variant.key>`                                                                                                                                                                                 |
| Declares a variant and profiles           | `<variant.key>-<profile.key>`, one per profile (`cuda12-egl`)                                                                                                                                   |
| Profiles but no variant block             | the bare profile key (`egl`)                                                                                                                                                                    |
| An undeclared single image (not an index) | `default`                                                                                                                                                                                       |
| An undeclared child of an index           | `default` when it is the only undeclared child on its platform, else named for its inferred stack: `cuda12`, `rocm6`, `cpu`                                                                     |
| A child that is not `linux/amd64`         | an undeclared child's key gets the platform appended (`default-arm64v8`). Any non-`linux/amd64` child is recorded with status `excluded` and never selected; a non-linux child fails its check. |

A derived row key must still match the key pattern, so a long variant key and a long profile key together can exceed 32 characters. `check-image` and `publish` fail on that, and `publish` fails when two children derive the same row key. A version needs at least one `linux/amd64` child. Its default row is the first `linux/amd64` child, at that child's default profile.

### The index annotation (`--index-package`)

Version-level data that applies to every variant can go on the index itself, as a `dev.rlmesh.package` annotation; `rlmesh registry publish --index-package version.json` sets it. It may carry only `name`, `description`, `checkpoints`, `compatibility`, `capabilities`, and `inputArtifacts`, plus `schemaVersion` (1, the default) and `rev`. The platform ignores any other key, and `publish` warns about it. Where a child's label sets one of those keys differently, the annotation wins, and `publish` warns about that too. Children's `variant` blocks remain the primary declaration: `variant` and `profiles` never belong on the index.

Only an OCI image index carries annotations. docker assembles a Docker manifest list instead when every manifest the sources carry, images and attestations alike, is a Docker schema2 manifest, as a classic `docker build` and `docker push` produce, so `publish` refuses `--index-package` then and names the sources. It also checks the index docker would assemble before pushing, and the pushed one after, and fails if either is not an OCI image index carrying the annotation. Rebuild them with OCI media types: `docker buildx build --push` (its attestations make the push an OCI index), or `--provenance=false --output type=image,oci-mediatypes=true,push=true`. One OCI source among them is enough. Without `--index-package`, schema2 sources publish as a Docker manifest list, which the platform reads as a version too.

A single source that is itself an index, published without `--index-package`, is republished as is, so its own `dev.rlmesh.package` annotation becomes the version's. `publish` checks it by the same rules as `--index-package` (a `schemaVersion` other than 1 fails; other keys are warned about), says that it is kept, and compares it with the child's label. Pass `--index-package` to replace it. With several sources, a source index's annotation is not carried into the version, and `publish` warns that it is dropped.

### What publish does, and its flags (`--tag`, `--force`)

Publishing:

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

### What check-image validates

`rlmesh check-image IMAGE` reports `variant:`, `profiles:`, and `rows:` lines next to its other checks, in the same failed / warnings / not checked buckets and the same `--json` shape. The rules are the platform's own.

It **fails** on what the platform's `variant_requires_schema` check fails:

- a block of the wrong type
- a missing or malformed key
- an unknown field or facet, or a facet value outside its vocabulary
- a `priority` that is not an integer in `[-1000, 1000]` (a string such as `"10"` or a fraction fails too)
- an unknown `requires` key, or a malformed constraint
- `accel.vram` that is not a positive whole-byte quantity string within the signed 64-bit range
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
