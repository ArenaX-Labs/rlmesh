//! The language-neutral recipe schema and its Dockerfile deriver.
//!
//! These `serde` structs are the canonical parse of the recipe wire format; the
//! Python `rlmesh.recipes` dataclasses are typed views with the identical JSON
//! shape (snake_case keys, a `kind`-tagged `make` union). [`derive_dockerfile`]
//! implements the build-field -> Dockerfile-instruction contract (spec section
//! 5A): the neutral conformance surface a non-Python deriver (a future capi
//! consumer) must reproduce, guarded by golden-file tests.
//!
//! Scope note: this slice derives the flat-gym subset plus apt packages, indexed
//! pip steps, env/pythonpath/gpu, a `run_as` user drop, raw commands, and the
//! verbatim-Dockerfile trapdoor. Build steps that require staging a host tree
//! into the build context (`project`, `fetch`) return [`DeriveError::Unsupported`]
//! rather than silently dropping an instruction; they land with the build-context
//! staging work.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::DEFAULT_BASE_IMAGE;

const CONTAINER_PORT: u16 = 50051;
const WORKDIR: &str = "/opt/rlmesh";
const ENTRYPOINT: &str = "ENTRYPOINT [\"python\", \"-m\", \"rlmesh._bootstrap.sandbox_env\"]";

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
}

/// The named factory (phase 3), tagged by `kind` in the wire format.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Setup {
    /// `os.environ` updates applied before `requires.imports`.
    pub env: BTreeMap<String, String>,
    /// File writes.
    pub files: Vec<FileWrite>,
}

/// Registration imports (gym/hf only).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Requires {
    /// Registration side-effect imports.
    pub imports: Vec<String>,
}

fn default_recipe_version() -> u32 {
    1
}

/// An inert environment recipe.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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

    // Build steps that require staging a host tree into the build context are not
    // yet derived; reject loudly rather than drop the instruction.
    if build.project.is_some() {
        return Err(DeriveError::Unsupported("project".to_string()));
    }
    if !build.fetch.is_empty() {
        return Err(DeriveError::Unsupported("fetch".to_string()));
    }
    if build.from_recipe.is_some() {
        return Err(DeriveError::Unsupported("from_recipe".to_string()));
    }
    if build.installer != "pip" {
        return Err(DeriveError::Unsupported(format!(
            "installer={}",
            build.installer
        )));
    }

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

    // apt: system union system_runtime, order-preserving dedup, one layer.
    let apt = union_preserving_order(&build.system, &build.system_runtime);
    if !apt.is_empty() {
        let mut names = Vec::with_capacity(apt.len());
        for name in &apt {
            validate_token("build.system", name)?;
            names.push(shell_quote(name));
        }
        out.push_str(&format!(
            "RUN apt-get update && apt-get install -y --no-install-recommends {} && rm -rf /var/lib/apt/lists/*\n\n",
            names.join(" ")
        ));
    }

    // pip: the stock preamble (pip upgrade + rlmesh + gymnasium) then each step.
    out.push_str(&format!("RUN {}\n\n", render_pip_chain(&build.pip)?));

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

/// Union two apt lists preserving first-occurrence order.
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

/// Render the single pip `RUN` chain: stock preamble then each [`PipInstall`].
fn render_pip_chain(steps: &[PipInstall]) -> Result<String, DeriveError> {
    let mut parts = vec![
        "python -m pip install --no-cache-dir --upgrade pip".to_string(),
        "python -m pip install --no-cache-dir rlmesh".to_string(),
        "python -m pip install --no-cache-dir gymnasium".to_string(),
    ];
    for step in steps {
        parts.push(render_pip_step(step)?);
    }
    Ok(parts.join(" && "))
}

/// Render one `python -m pip install` line with its own index/flag arguments.
fn render_pip_step(step: &PipInstall) -> Result<String, DeriveError> {
    let mut line = String::from("python -m pip install --no-cache-dir");
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

    const GYM_RECIPE_JSON: &str = include_str!("../tests/golden/safety_gymnasium.recipe.json");
    const GYM_GOLDEN: &str = include_str!("../tests/golden/safety_gymnasium.dockerfile");

    fn gym_recipe() -> Recipe {
        Recipe::from_json(GYM_RECIPE_JSON).expect("fixture parses")
    }

    #[test]
    fn deserializes_python_wire_shape() {
        let recipe = gym_recipe();
        assert_eq!(recipe.name, "safety/point-goal");
        assert_eq!(
            recipe.make,
            Some(Make::Gym {
                env_id: "SafetyPointGoal1-v0".to_string(),
                kwargs: BTreeMap::new(),
            })
        );
        assert_eq!(recipe.build.pip.len(), 1);
        assert_eq!(recipe.build.pip[0].packages, ["safety-gymnasium==1.0.0"]);
        assert_eq!(recipe.requires.imports, ["safety_gymnasium"]);
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
    fn rejects_project_and_fetch_loudly() {
        let project =
            Recipe::from_json(r#"{"name":"a","build":{"project":{"src":"."}}}"#).expect("parses");
        assert!(matches!(
            derive_dockerfile(&project),
            Err(DeriveError::Unsupported(_))
        ));

        let fetch = Recipe::from_json(
            r#"{"name":"a","build":{"fetch":[{"kind":"git","repo":"https://x/r.git"}]}}"#,
        )
        .expect("parses");
        assert!(matches!(
            derive_dockerfile(&fetch),
            Err(DeriveError::Unsupported(_))
        ));
    }

    #[test]
    fn rejects_uv_installer_for_now() {
        let recipe =
            Recipe::from_json(r#"{"name":"a","build":{"installer":"uv"}}"#).expect("parses");
        assert!(matches!(
            derive_dockerfile(&recipe),
            Err(DeriveError::Unsupported(_))
        ));
    }
}
