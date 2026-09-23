# CLI Reference

The `rlmesh` binary signs in to a managed RLMesh platform, keeps that session fresh for scripts and for docker, and submits and watches evaluations. It ships standalone (`cargo install rlmesh-cli`) and inside the Python package as `python -m rlmesh` and the `rlmesh` console script.

`rlmesh --help` and `rlmesh <command> --help` are the authoritative flag lists. This page describes the behavior around them.

## Commands

| Command                                                                                | What it does                                                                                                                                   | Exit code                                                              |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `rlmesh version`                                                                       | Version, workflow edition, every edition the build can drive, and distribution (`standalone`, `python-wheel`, ...).                            | 0                                                                      |
| `rlmesh login [--platform URL]`                                                        | Signs in with the browser device flow and stores the session under the profile. Defaults to the hosted platform.                               | 1 if the sign-in was declined, expired, or the platform cannot be used |
| `rlmesh logout`                                                                        | Asks the platform to end the session, then deletes the stored credential.                                                                      | 0 even when the platform could not be reached                          |
| `rlmesh whoami [--json]`                                                               | The profile, its platform, sign-in state, and the identity the platform confirms.                                                              | 0 only for a verified session or API key                               |
| `rlmesh token [--json]`                                                                | A fresh access token for scripts: `Authorization: Bearer $(rlmesh token)`.                                                                     | 1 when there is no usable session                                      |
| `rlmesh profile list [--json]` / `use NAME` / `remove NAME`                            | Named profiles: which platform and session a command acts on.                                                                                  |                                                                        |
| `rlmesh org list [--json]` / `switch ID`                                               | The organizations the account belongs to; switch re-issues the session against another one.                                                    | 1 when the provider kept the current organization                      |
| `rlmesh registry login`                                                                | Registers the bundled `docker-credential-rlmesh` helper for the platform's image registry.                                                     | 1 when not signed in                                                   |
| `rlmesh eval submit REQUEST [--preview] [--wait] [--json]`                             | Launches (or, with `--preview`, sizes) an evaluation from a JSON file or `-` for stdin.                                                        | with `--wait`, 1 unless the evaluation completed                       |
| `rlmesh eval list [--status S] [--tag K:V]... [--q TEXT] [--limit N] [--json]`         | Newest evaluations first.                                                                                                                      |                                                                        |
| `rlmesh eval get ID`                                                                   | One evaluation as JSON.                                                                                                                        |                                                                        |
| `rlmesh eval wait ID`                                                                  | Polls until the evaluation finishes.                                                                                                           | 1 unless it completed                                                  |
| `rlmesh eval cancel ID`                                                                | Cancels a running evaluation.                                                                                                                  |                                                                        |
| `rlmesh check MODULE:CLASS [--json]`                                                   | Checks an env or model class for packaging and contract mistakes before its image is built.                                                    | 1 on a failure                                                         |
| `rlmesh check-image IMAGE [--platform-editions LIST \| --platform-version V] [--json]` | Checks a built image's config the way the platform's admission will, before pushing.                                                           | 1 on a failure                                                         |
| `rlmesh describe MODULE:CLASS [--label]`                                               | Prints the class's describe envelope, or with `--label` the `dev.rlmesh.describe=...` value for `docker build --label` (run inside the image). |                                                                        |

Every command that touches a platform takes `--profile NAME` (or `RLMESH_PROFILE`) to pick the profile; the default profile is the one `rlmesh profile use` chose, else the first one that signed in, else `default`. Usage errors exit 2.

## Pre-push checks

`rlmesh check` and `rlmesh check-image` sort what they find into three buckets and exit 1 only on the first: **failed** (the push would land as not-runnable, or the platform would reject a claim), **warnings** (a claim the platform trims, or intent it cannot see), and **not checked** (what could not be decided on this machine, and who decides it instead). `--json` prints `{"failed", "warnings", "not_checked", "passed"}`, each a list of strings.

`rlmesh check MODULE:CLASS` runs the Python describe on the class, in the interpreter `RLMESH_PYTHON` names (the `python -m rlmesh` entrypoint sets it to its own; else the first of `python3` / `python` that imports `rlmesh`). It fails on a model without a `spec` (the managed probe cannot synthesize inputs without a `ModelSpec`), an env without `tags`, a spec or tags that does not resolve, a broken `model_spec` / `env_tags`, a workflow edition declaration this build cannot run, or a class that does not import. It warns on ad hoc roles a curated publish gate would refuse and on best-effort badges elsewhere in the envelope. Building the env for its spaces is best-effort: when it needed a GPU, a display, or assets this machine lacks, that is reported as not checked, not failed.

`rlmesh check-image IMAGE` reads `docker image inspect` and checks: the command runs `python -m rlmesh.serve` with a `module:Class`, `--env` if and only if the describe label says the image is an env, no baked `--address` (warning: it would override the address the platform assigns per pod), no image-level `ENV RLMESH_ADDRESS` (warning), `EXPOSE 50051` (warning), `linux/amd64`, the `dev.rlmesh.describe` and `dev.rlmesh.package` labels when present (a missing describe label is a note: the platform reads describe off the handshake), and, when the describe label advertises editions, whether the image's rlmesh shares a workflow edition with the platform, negotiated the way the runtime does at bind time. The platform's editions come from `--platform-editions 2026.06,...` (the list `rlmesh version` prints as `Editions` on the platform's rlmesh), from `--platform-version VERSION`, or default to this CLI's own rlmesh. Two assumptions behind `--platform-version`: only this CLI's own version is known, since each release retains its own edition list (earlier prereleases offered only their own cohort; `2026.06` first appears in the builds that seal it), so any other version leaves the edition check as not checked and asks for `--platform-editions`; and version spellings are normalized, so `0.1.0rc15` (PyPI) and `0.1.0-rc.15` name the same build. A custom entrypoint (not `rlmesh.serve`) is not checked; the runtime probe verifies it.

`rlmesh describe MODULE:CLASS --label` is for an image with a custom entrypoint (one that does not run `python -m rlmesh.serve`, so the platform cannot read describe off the handshake). The envelope records the machine it ran on: its OS and architecture, package versions, editions, and whatever env the machine could build. The platform fails a label whose `runtime.os` is not `linux`, and `check-image` fails it the same way, so generate the label inside the built image and add it in a second, cached build:

```bash
docker build -t my-model:latest .
docker build --label "$(docker run --rm --entrypoint rlmesh my-model:latest describe my_pkg:Policy --label)" -t my-model:latest .
```

The describe envelope's `runtime` block carries what the served peer's handshake will send, under the handshake's names: `protocol_generation`, `supported_workflow_editions` (every edition its rlmesh can drive, newest first) and `preferred_workflow_edition` (what the server resolved: `--workflow-edition`, `RLMESH_WORKFLOW_EDITION`, the class declaration, or `[tool.rlmesh]`, else that build's newest); `runtime.workflow_edition_error` is present when the declaration names an edition that rlmesh cannot run, and both `check` and `check-image` fail on it since `rlmesh.serve` would refuse to start. See `docs/specs/describe.v1.md`. `python -m rlmesh._describe` is the module behind both commands: `TARGET [--label]`, `--check-entrypoint MODULE:CLASS`, `--check IMAGE`, `--check-labels FILE`, each with `--json`.

## Environment

| Variable                | Effect                                                                                                                                                                                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RLMESH_PROFILE`        | The profile to act on when `--profile` is absent.                                                                                                                                                                                                                     |
| `RLMESH_PYTHON`         | The interpreter `check` and `describe` run `python -m rlmesh._describe` in. Set by the `python -m rlmesh` entrypoint to its own interpreter; otherwise the first of `python3` / `python` that imports `rlmesh`.                                                       |
| `RLMESH_PLATFORM_URL`   | The platform for `rlmesh login --platform` and for `RLMESH_API_KEY`. A bare host gets `https://`; loopback hosts get `http://`.                                                                                                                                       |
| `RLMESH_API_KEY`        | A platform API key. `token`, `eval`, and `whoami` use it as the bearer credential instead of a profile session; the platform defaults to `https://api.rlmesh.dev` unless `RLMESH_PLATFORM_URL` is set. `login`, `logout`, `org`, `profile`, and `registry` ignore it. |
| `RLMESH_CONFIG_DIR`     | Where `config.toml` lives (default: the OS config directory, `rlmesh` subdirectory).                                                                                                                                                                                  |
| `RLMESH_DATA_DIR`       | Where `credentials.json` and the session lock files live (default: the OS local-data directory, `rlmesh` subdirectory).                                                                                                                                               |
| `RLMESH_KEYCHAIN`       | `off`, `0`, `false`, or `no` keeps credentials out of the OS keychain and in the 0600 file instead, for CI runners and containers.                                                                                                                                    |
| `NO_COLOR`, `TERM=dumb` | Plain output; color is also off whenever stdout is not a terminal.                                                                                                                                                                                                    |

## Files

- `config.toml` (config directory): the default profile and, per profile, the platform URL, the token endpoint the profile signed in against, the registry host handed to docker, and the identity the platform last confirmed. Never holds a credential.
- `credentials.json` (data directory, mode 0600): the access and refresh token per profile, used only when the OS keychain is unavailable or `RLMESH_KEYCHAIN=off`. Otherwise credentials live in the keychain under the service `rlmesh` with the profile name as the account.
- `<profile>.lock` (data directory): held for the duration of a session refresh so concurrent commands rotate the session once. A lock older than 30 seconds is treated as abandoned.
- `~/.docker/config.json` (or `$DOCKER_CONFIG/config.json`): `rlmesh registry login` adds a `credHelpers` entry mapping the registry host to `rlmesh`.

## Session lifecycle

The access token is a short-lived JWT. A command reuses it while it is more than a minute from its `exp` claim, so `rlmesh token`, `rlmesh eval ...`, and every docker pull cost no round trip to the identity provider. Only an expired token triggers the `refresh_token` grant, under the profile lock, and after re-reading the credential store in case another process already rotated it. A request the platform still rejects as unauthorized is retried once after a forced refresh. Refresh tokens are single use, so a session that two machines share stops working on one of them after the other refreshes; sign in on each machine instead.

`rlmesh org switch` re-issues the session with the identity provider's `organization_id` extension to the refresh grant, then confirms the active organization from the new token and from the platform.

`rlmesh logout` calls the platform's session-revocation route before deleting the local credential. A platform without that route, or one that cannot be reached, is reported in one muted line and the local sign-out proceeds.

## JSON output

- `rlmesh whoami --json`: `{"profile", "platform", "status", "verified", "identity", "error"}`. `status` is one of `signed_in`, `signed_out`, `incomplete`, `api_key`; `identity` is `{"userId", "email", "displayName", "organizationId", "organizationName"}` or `null`; `error` is the verification failure or `null`. The exit code follows `verified`.
- `rlmesh token --json`: `{"platform", "token", "expiresAt"}`. `expiresAt` is RFC 3339 UTC, or `null` for an API key or a token without an `exp` claim.
- `rlmesh profile list --json`: `[{"name", "platform", "status", "default"}]`.
- `rlmesh org list --json`: the platform's `organizations` array as served, each entry with `name`, `providerId`, `active`, and, when the organization is provisioned, `id` and `registryNamespace`.
- `rlmesh eval ... --json`: the platform's response as served.

## Platform contract

The CLI has no built-in identity provider. Everything it needs comes from the platform it signs in to, so a provider change is a platform deploy rather than a CLI upgrade. A platform that wants to be driven by this CLI serves:

- `GET /v1/info` (unauthenticated): `auth.cli.clientId` (the OAuth public client id the CLI presents), `auth.deviceAuthorizationEndpoint`, and `auth.tokenEndpoint`, plus `urls.dashboard` for the links `eval submit` prints. Other fields, `auth.issuer` included, are ignored. Endpoints must be `https://` (plain `http://` is accepted only for loopback hosts). The document only ever gains fields.
- The device flow of RFC 8628 at the advertised endpoints: `POST deviceAuthorizationEndpoint` with `client_id` returns `device_code`, `user_code`, `verification_uri` (optionally `verification_uri_complete`), `expires_in`, and `interval`; the CLI then polls `POST tokenEndpoint` with `grant_type=urn:ietf:params:oauth:grant-type:device_code` and honors `authorization_pending`, `slow_down`, `access_denied`, and `expired_token`.
- The `refresh_token` grant of RFC 6749 at the token endpoint, with one extension: an `organization_id` form field asks for the session to be re-issued against that organization. Refresh tokens rotate on use. The token response needs only `access_token` and `refresh_token`; anything else is ignored.
- Access tokens as JWTs whose payload carries the standard `exp` claim. The CLI reads it without verifying the signature, only to schedule refreshes; the platform validates every token it receives, and no other claim is read.
- `GET /v1/me` (bearer): `subject.id`, optionally `subject.email` and `subject.displayName`, and `organization.providerId` plus `organization.name`. The provider id is the id the CLI shows and switches on.
- `GET /v1/me/organizations` (bearer): `organizations[]` with `name`, `providerId`, `active`, and optionally `id` and `registryNamespace`.
- `DELETE /v1/me/session` (bearer): ends the session behind the token; `204` on success, idempotent. A `404` or `405` is treated as "not supported".
- `GET /v1/registry/info` (bearer): `host` and `namespaces[]` of the image registry, whose password is the access token.
- API keys as plain bearer credentials on every `/v1` route, for headless use.

The CLI pins the token endpoint's host per profile at sign-in. If `/v1/info` later advertises a token endpoint on another host, refreshes stop with an error until the user signs in again, so a compromised or misconfigured platform cannot redirect a stored refresh token. A profile stored by a CLI from before pinning existed has no pin and is asked to sign in once more rather than adopting whatever the platform advertises.
