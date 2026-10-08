# Publishing compute variants

One model or environment version can ship several container images: a PyTorch CUDA build, a ROCm build, a JAX build, a build that only runs on newer GPUs. One image can also run in several ways, such as osmesa or EGL rendering. RLMesh describes both structurally, so publishing them needs no platform-side configuration:

- **A version is an OCI image index.** Its non-attestation children are the **variants**: one image each, each declaring a `variant` block in its `dev.rlmesh.package` label.
- **Profiles are runtime configurations of one image.** They live in that image's label as `profiles[]` and differ only in environment variables, GPU count, and hardware requirements.

`rlmesh registry publish` assembles the index from images you have already pushed, and `rlmesh check-image` validates the `variant` and `profiles` blocks before you push. Neither touches the describe envelope or the wire protocol.

## The label schema

Both blocks sit inside the existing `dev.rlmesh.package` JSON label, next to `schemaVersion`, `name`, `tags`, `checkpoints`, and the rest:

```json
{
  "schemaVersion": 1,
  "variant": {
    "key": "torch-cuda12",
    "facets": { "framework": "torch", "accel": "nvidia", "render": "egl" },
    "requires": {
      "accel.vendor": "nvidia",
      "accel.compute": ">=8.0",
      "accel.gfx": ["gfx942"],
      "accel.cuda": ">=12.2",
      "accel.driver": ">=535",
      "accel.vram_bytes": ">=24000000000"
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
      "gpu": { "count": 1 },
      "requires": { "accel.vendor": "nvidia" }
    }
  ]
}
```

The example shows every `requires` key at once. A real label would not pair `accel.gfx` (an AMD GPU target) with `accel.vendor: nvidia`.

Every field is optional in the schema. `rlmesh registry publish` requires `variant.key` on each child, since the platform addresses a variant by its key.

| Field                  | Meaning                                                                                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `variant.key`          | The variant's name within the version (`torch-cuda12`). It is unique per version. Keep it a DNS label: lowercase letters, digits, and `-`, at most 63 characters. |
| `variant.facets`       | Free-form string tags. Checkpoints match on them (`requires: {framework: torch}`), and the platform displays them.                                                |
| `variant.requires`     | What hardware the image needs (see below).                                                                                                                        |
| `variant.priority`     | An integer. When several variants fit the same hardware, the higher one wins.                                                                                     |
| `profiles[].key`       | The profile's name, unique within the image.                                                                                                                      |
| `profiles[].default`   | Exactly one profile is the default. A request that names no profile runs it.                                                                                      |
| `profiles[].facets`    | Like `variant.facets`, for this profile.                                                                                                                          |
| `profiles[].envVars`   | Environment variables the profile sets on the container. `RLMESH_ADDRESS` has no effect, since the platform assigns it per pod.                                   |
| `profiles[].gpu.count` | GPUs the profile needs.                                                                                                                                           |
| `profiles[].requires`  | Hardware requirements on top of the variant's. They must be compatible with the variant's.                                                                        |

`requires` keys are a fixed set, each compared against the hardware the platform probes:

| Key                | Value                                               | Comparators                  |
| ------------------ | --------------------------------------------------- | ---------------------------- |
| `accel.vendor`     | `nvidia`, `amd`, or `intel`                         | equality, or a list (one of) |
| `accel.compute`    | NVIDIA compute capability, a dotted version (`8.0`) | `>=`, `<`, `=`, or a list    |
| `accel.gfx`        | AMD GPU target (`gfx942`)                           | equality, or a list (one of) |
| `accel.cuda`       | The CUDA version the host driver supports (`12.2`)  | `>=`, `<`, `=`, or a list    |
| `accel.driver`     | The GPU driver version (`535`, `535.104.05`)        | `>=`, `<`, `=`, or a list    |
| `accel.vram_bytes` | GPU memory in bytes                                 | `>=`, `<`, `=`, or a list    |

A value is a string: `">=12.2"`, `"<13"`, `"=12.4"`, or a bare `"nvidia"`, which means `=`. An array of bare strings means set membership. Only `>=`, `<`, and `=` exist. Versions compare numerically part by part, so `12` equals `12.0`. A JSON number is refused: write `">=24000000000"`, not `24000000000`.

Version-level data that applies to every variant, such as checkpoints and compatibility, can go on the index itself as a `dev.rlmesh.package` **annotation**. `rlmesh registry publish --index-package version.json` sets it.

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

1. Resolves each source and pins it by digest, so a tag that moves mid-publish cannot swap a child. Each source must be one linux image. A BuildKit push is an index of the image plus its attestation manifests, and the attestations are carried into the version.
2. Reads each image's config and checks that every child carries a `dev.rlmesh.package` label with a `variant.key`, that keys are unique, that every child serves the same kind (env or model, from the describe label or the `rlmesh.serve` command), and that each `variant` and `profiles` block passes the same checks as `check-image`. Every problem across all sources is reported at once, and nothing is pushed if there is one.
3. Creates the index with `docker buildx imagetools create` and pushes it under TARGET, each `--tag`, and `--channel`.

| Flag                   | Effect                                                                                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--tag TAG`            | Another tag for the index in TARGET's repository; repeatable.                                                                                                                    |
| `--channel TAG`        | A moving channel tag such as `latest`.                                                                                                                                           |
| `--index-package FILE` | Version-level JSON, set as the index's `dev.rlmesh.package` annotation. `schemaVersion` defaults to 1; `variant` and `profiles` are refused here, since they describe one image. |
| `--dry-run`            | Prints the per-variant summary and the index JSON without pushing.                                                                                                               |

```text
$ rlmesh registry publish --dry-run --channel latest --index-package version.json \
    ns/pi0:v3 ns/pi0:v3-cuda12 ns/pi0:v3-rocm6 ns/pi0:v3-jax
Dry run: would publish ns/pi0:v3 (3 variants)

VARIANT       KIND   PLATFORM     PRIORITY  FACETS                         REQUIRES                                                   PROFILES               IMAGE
torch-cuda12  model  linux/amd64  10        accel=nvidia, framework=torch  accel.compute>=8.0, accel.cuda>=12.4, accel.vendor=nvidia  -                      ns/pi0:v3-cuda12 (488f376057c6)
torch-rocm6   model  linux/amd64  5         accel=amd, framework=torch     accel.gfx in [gfx942,gfx90a], accel.vendor=amd             -                      ns/pi0:v3-rocm6 (9d96d96d9938)
jax           model  linux/amd64  -         framework=jax                  -                                                          osmesa (default), egl  ns/pi0:v3-jax (7a4d816428ad)

Index for ns/pi0:v3, ns/pi0:latest (nothing pushed):
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ ... ] }
```

Publishing goes through docker, so it authenticates the way `docker push` does. Against the managed platform's registry, run `rlmesh registry login` first.

## Profiles: osmesa and EGL

A MuJoCo environment that renders headless on a CPU with osmesa, or on an NVIDIA GPU with EGL, is one image with two profiles:

```json
{
  "schemaVersion": 1,
  "variant": { "key": "mujoco", "facets": { "sim": "mujoco" } },
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

The image installs both osmesa and the EGL libraries. A profile switches between them with `MUJOCO_GL` and asks for a GPU only when it needs one, so the CPU profile can be scheduled anywhere.

## What check-image validates

`rlmesh check-image IMAGE` reports `variant:` and `profiles:` findings next to its other checks, in the same failed / warnings / not checked buckets and the same `--json` shape.

It fails on:

- a block of the wrong type
- a `requires` key outside the fixed set
- a comparator other than `>=`, `<`, or `=`
- a value that is not a version or byte count where one is expected
- a JSON number used as a requirement
- duplicate or missing profile keys
- anything other than exactly one default profile
- a profile whose `requires` contradicts the variant's (`accel.vendor=amd` under a variant that requires `nvidia`)
- an `accel.cuda` range that no host can satisfy alongside the image's own CUDA floor

It also reads the image's `Env` and warns when `requires` disagrees with the stack the image was built on:

- **CUDA images** (`CUDA_VERSION`, `NVIDIA_REQUIRE_CUDA`): `accel.vendor` should admit `nvidia`. `accel.cuda` should not admit hosts below the image's floor. That floor is the `cuda>=X` in `NVIDIA_REQUIRE_CUDA`, which the NVIDIA container runtime enforces; without it, `CUDA_VERSION`'s major.minor. `accel.gfx` (AMD) is out of place.
- **ROCm images** (`ROCM_VERSION`): `accel.vendor` should admit `amd`, and `accel.cuda` / `accel.compute` are out of place.

Unknown fields, unknown vendors, and keys that are not DNS labels are warnings. A label with neither block gets none of these checks.

## What the managed platform does with it

The platform probes each variant of a version on each hardware class it runs, the same admission probe a single image gets. It records which variants run where. Given an evaluation and the hardware it lands on, the platform picks a variant automatically:

1. Keep the variants whose `requires` the hardware meets.
2. Keep those whose facets match the requested checkpoint's requirements.
3. Prefer the highest `priority`.

An evaluation request can instead name one explicitly with `variant: <key>`. Within the chosen image, the default profile runs unless the request selects another.
