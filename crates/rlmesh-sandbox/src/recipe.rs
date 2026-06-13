//! The language-neutral recipe schema and its Dockerfile deriver.
//!
//! These `serde` structs are the canonical parse of the recipe wire format; the
//! Python `rlmesh.recipes` dataclasses are typed views with the identical JSON
//! shape (snake_case keys, a `kind`-tagged `make` union). [`derive_dockerfile`]
//! implements the build-field -> Dockerfile-instruction contract (spec section
//! 5A): the neutral conformance surface a non-Python deriver (a future capi
//! consumer) must reproduce, guarded by golden-file tests.
//!
//! The deriver covers the full build vocabulary: base (+python symlink for a
//! non-python base), env/pythonpath/gpu, apt (`system` united with
//! `system_runtime`), the author's `project` tree (`COPY` + editable install),
//! third-party `fetch` (pinned git clone / checksummed url download), pip or uv
//! install steps, a `run_as` user drop, raw `commands`, and the verbatim-
//! Dockerfile trapdoor. `from_recipe` is inlined by the registry layer before the
//! deriver runs. `ProjectInstall` requires its source tree to be staged into the
//! build context under [`PROJECT_CONTEXT_DIR`] by the caller.
//!
//! System packages (`system`/`system_runtime`) are installed with **apt**, so a
//! structured build targets a **Debian/Ubuntu** base; [`render_system_packages`] is
//! the single point to generalize to another distro, and `build.dockerfile` is the
//! escape hatch for a non-Debian base today.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::DEFAULT_BASE_IMAGE;

const CONTAINER_PORT: u16 = 50051;
const WORKDIR: &str = "/opt/rlmesh";
const ENTRYPOINT: &str = "ENTRYPOINT [\"python\", \"-m\", \"rlmesh._bootstrap.sandbox_env\"]";

/// The build-context subdirectory a `ProjectInstall` source tree is staged into;
/// the deriver `COPY`s from here and `write_build_context` populates it.
pub const PROJECT_CONTEXT_DIR: &str = "project";
const DEFAULT_PROJECT_DEST: &str = "/opt/rlmesh/project";

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
    /// `FROM` image (None -> [`DEFAULT_BASE_IMAGE`]).
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
    /// Forward field: published adapter annotations.
    #[serde(default)]
    pub annotations: Option<serde_json::Value>,
    /// The schema version.
    #[serde(default = "default_recipe_version")]
    pub recipe_version: u32,
}

impl Recipe {
    /// Parse a recipe from its canonical JSON wire format. Parsing executes nothing.
    pub fn from_json(payload: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(payload)
    }
}

/// Quote a token for safe single-argument use in a `/bin/sh` command.
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
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
pub fn derive_dockerfile(recipe: &Recipe) -> Result<String, DeriveError> {
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
        out.push_str(ENTRYPOINT);
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

    let base = build.base.as_deref().unwrap_or(DEFAULT_BASE_IMAGE);
    validate_token("build.base", base)?;

    let mut out = String::new();
    out.push_str("# syntax=docker/dockerfile:1.7\n\n");
    out.push_str(&format!("FROM {base}\n\n"));

    // ENV block: standard vars, then gpu caps, then build.env, then PYTHONPATH.
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

    // system packages: system union system_runtime, order-preserving dedup, one layer.
    let apt = union_preserving_order(&build.system, &build.system_runtime);
    out.push_str(&render_system_packages(&apt)?);

    // A non-python base (e.g. nvidia/cuda) has no `python` on PATH; symlink it.
    if base_is_non_python(base) {
        out.push_str("RUN ln -sf \"$(command -v python3)\" /usr/local/bin/python\n\n");
    }

    // project: COPY the author's staged tree then install it (editable by default).
    if let Some(project) = &build.project {
        out.push_str(&render_project(project, verb)?);
    }

    // fetch: third-party git clones / url downloads, each a pinned RUN.
    for fetch in &build.fetch {
        out.push_str(&render_fetch(fetch, verb)?);
    }

    // pip: the stock preamble (bootstrap + rlmesh + gymnasium) then each step.
    out.push_str(&format!(
        "RUN {}\n\n",
        render_pip_chain(&build.pip, &build.installer)?
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

    out.push_str(&format!("EXPOSE {CONTAINER_PORT}\n"));
    out.push_str(ENTRYPOINT);
    out.push('\n');
    Ok(out)
}

/// Whether a base image lacks a `python` on PATH and needs the `python3` symlink.
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
            let mut chain = format!("git clone --depth=1 {repo_q} {dest_q}");
            if let Some(git_ref) = &fetch.ref_ {
                validate_token("fetch.ref", git_ref)?;
                let ref_q = shell_quote(git_ref);
                chain.push_str(&format!(
                    " && git -C {dest_q} fetch --depth=1 origin {ref_q} && git -C {dest_q} checkout {ref_q}"
                ));
            }
            if let Some(req) = &fetch.pip_requirements {
                validate_token("fetch.pip_requirements", req)?;
                chain.push_str(&format!(" && {verb} -r {dest_q}/{req}"));
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

/// Render the single pip `RUN` chain: installer preamble then each [`PipInstall`].
fn render_pip_chain(steps: &[PipInstall], installer: &str) -> Result<String, DeriveError> {
    let verb = install_verb(installer);
    let mut parts = Vec::new();
    if installer == "uv" {
        // Bootstrap uv itself with pip, then install everything through uv.
        parts.push("python -m pip install --no-cache-dir uv".to_string());
    } else {
        parts.push("python -m pip install --no-cache-dir --upgrade pip".to_string());
    }
    parts.push(format!("{verb} rlmesh"));
    parts.push(format!("{verb} gymnasium"));
    for step in steps {
        parts.push(render_pip_step(step, verb)?);
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

#[cfg(test)]
mod tests {
    use super::*;

    const GYM_RECIPE_JSON: &str = include_str!("../tests/golden/gym_atari.recipe.json");
    const GYM_GOLDEN: &str = include_str!("../tests/golden/gym_atari.dockerfile");
    const LIBERO_RECIPE_JSON: &str = include_str!("../tests/golden/libero.recipe.json");
    const LIBERO_GOLDEN: &str = include_str!("../tests/golden/libero.dockerfile");

    fn gym_recipe() -> Recipe {
        Recipe::from_json(GYM_RECIPE_JSON).expect("fixture parses")
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
        let derived = derive_dockerfile(&gym_recipe()).expect("derives");
        assert_eq!(
            derived, GYM_GOLDEN,
            "derived Dockerfile drifted from the checked-in golden"
        );
    }

    #[test]
    fn empty_build_uses_default_base() {
        let recipe = Recipe::from_json(r#"{"name":"a","make":{"kind":"gym","env_id":"E-v0"}}"#)
            .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert!(derived.contains(&format!("FROM {DEFAULT_BASE_IMAGE}")));
    }

    #[test]
    fn renders_apt_union_dedup_order() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"system":["cmake","g++"],"system_runtime":["g++","libglew2.2"]}}"#,
        )
        .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
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
        let derived = derive_dockerfile(&recipe).expect("derives");
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
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert!(derived.contains("ENV NVIDIA_DRIVER_CAPABILITIES=all"));
        assert!(derived.contains("ENV MUJOCO_GL=egl"));
        assert!(derived.contains("ENV PYTHONPATH=/opt/x:/opt/y"));
    }

    #[test]
    fn renders_run_as_user_and_commands() {
        let recipe =
            Recipe::from_json(r#"{"name":"a","build":{"run_as":1000,"commands":["echo built"]}}"#)
                .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert!(derived.contains("RUN sh -lc 'echo built'"));
        assert!(derived.contains("USER 1000"));
        assert!(derived.contains("useradd --create-home --uid 1000 rlmesh"));
    }

    #[test]
    fn dockerfile_trapdoor_emits_verbatim_with_entrypoint() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"dockerfile":"FROM scratch\nRUN echo hi\n"}}"#,
        )
        .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert_eq!(
            derived,
            "FROM scratch\nRUN echo hi\n\nENTRYPOINT [\"python\", \"-m\", \"rlmesh._bootstrap.sandbox_env\"]\n"
        );
    }

    #[test]
    fn renders_project_copy_and_editable_install() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"project":{"src":".","dest":"/opt/robot_env"}}}"#,
        )
        .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert!(derived.contains("COPY project /opt/robot_env"));
        assert!(derived.contains("RUN python -m pip install --no-cache-dir -e '/opt/robot_env'"));
    }

    #[test]
    fn renders_project_default_dest_and_non_editable() {
        let recipe = Recipe::from_json(r#"{"name":"a","build":{"project":{"editable":false}}}"#)
            .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
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
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert!(derived.contains(
            "RUN git clone --depth=1 'https://github.com/x/LIBERO.git' '/opt/LIBERO' && git -C '/opt/LIBERO' fetch --depth=1 origin 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' && git -C '/opt/LIBERO' checkout 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' && python -m pip install --no-cache-dir -e '/opt/LIBERO' && rm -rf '/opt/LIBERO'/.git"
        ));
    }

    #[test]
    fn renders_url_fetch_with_sha256_check() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"url","url":"https://x/a.tar.gz","dest":"/opt/a.tar.gz","sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}}"#,
        )
        .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
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
        assert!(matches!(
            derive_dockerfile(&recipe),
            Err(DeriveError::MissingField(_))
        ));
    }

    #[test]
    fn renders_uv_installer_path() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"installer":"uv","pip":[{"packages":["sapien"]}]}}"#,
        )
        .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert!(derived.contains("python -m pip install --no-cache-dir uv"));
        assert!(derived.contains("uv pip install --system --no-cache-dir rlmesh"));
        assert!(derived.contains("uv pip install --system --no-cache-dir 'sapien'"));
    }

    #[test]
    fn non_python_base_gets_python_symlink() {
        let recipe = Recipe::from_json(
            r#"{"name":"a","build":{"base":"nvidia/cuda:12.4.1-runtime-ubuntu22.04"}}"#,
        )
        .expect("parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
        assert!(derived.contains("RUN ln -sf \"$(command -v python3)\" /usr/local/bin/python"));
        // The default python base must NOT get the symlink.
        let py = Recipe::from_json(r#"{"name":"a","make":{"kind":"gym","env_id":"E-v0"}}"#)
            .expect("parses");
        assert!(!derive_dockerfile(&py).expect("derives").contains("ln -sf"));
    }

    #[test]
    fn libero_recipe_matches_golden_dockerfile() {
        let recipe = Recipe::from_json(LIBERO_RECIPE_JSON).expect("fixture parses");
        let derived = derive_dockerfile(&recipe).expect("derives");
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
}
