mod docker;
mod error;
mod hf;
pub mod recipe;
mod source;
mod wheel;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use error::SandboxError;
pub use source::{
    EnvironmentSourceRef, GymSourceRef, HfSourceRef, RecipeProvenance, RecipeSourceRef,
};
pub(crate) use wheel::ResolvedRlmeshPackage;

pub const DEFAULT_BASE_IMAGE: &str = "python:3.11-slim";
pub const DEFAULT_PACKAGE_NAME: &str = "rlmesh";
pub const BOOTSTRAP_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct SandboxOptions {
    pub base_image: Option<String>,
    pub rlmesh_package: Option<String>,
    pub packages: Vec<String>,
    pub imports: Vec<String>,
    pub kwargs: BTreeMap<String, serde_json::Value>,
    pub num_envs: usize,
    pub vectorization_mode: VectorizationMode,
    pub trust_remote_code: bool,
    pub allow_unpinned_hf: bool,
    /// The recipe author's source-tree root, used to resolve a relative
    /// `ProjectInstall.src` for build-context staging and content hashing. Only
    /// meaningful for a recipe source with a `build.project`.
    pub context_root: Option<PathBuf>,
}

impl Default for SandboxOptions {
    fn default() -> Self {
        Self {
            base_image: None,
            rlmesh_package: None,
            packages: Vec::new(),
            imports: Vec::new(),
            kwargs: BTreeMap::new(),
            num_envs: 1,
            vectorization_mode: VectorizationMode::Sync,
            trust_remote_code: false,
            allow_unpinned_hf: false,
            context_root: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorizationMode {
    Sync,
    Async,
}

impl VectorizationMode {
    pub fn parse(value: Option<&str>) -> std::result::Result<Self, SandboxError> {
        match value.unwrap_or("sync").trim() {
            "sync" => Ok(Self::Sync),
            "async" => Ok(Self::Async),
            other => Err(SandboxError::invalid_option(format!(
                "vectorization_mode must be 'sync' or 'async', got '{other}'"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::Async => "async",
        }
    }
}

impl SandboxOptions {
    pub fn resolved_base_image(&self) -> String {
        self.base_image
            .clone()
            .or_else(|| std::env::var("RLMESH_SANDBOX_BASE_IMAGE").ok())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_BASE_IMAGE.to_string())
    }

    fn resolved_rlmesh_package(&self, base_image: &str) -> Result<ResolvedRlmeshPackage> {
        let selected = self
            .rlmesh_package
            .clone()
            .or_else(|| {
                std::env::var("RLMESH_SANDBOX_RLMESH_PACKAGE")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or_else(default_rlmesh_package);

        wheel::resolve_rlmesh_package(validate_nonempty("rlmesh_package", selected)?, base_image)
    }
}

/// Details of a started sandbox container, returned by [`start_env`] and
/// [`start_env_async`].
///
/// Dropping this without recording `container_id` leaks a running container, so
/// it is `#[must_use]`. It is `#[non_exhaustive]` so future fields (extra
/// container metadata and ports can be added without breaking callers that
/// read fields by name.
#[derive(Debug, Clone)]
#[must_use = "dropping a RunResult without its container_id leaks the started container"]
#[non_exhaustive]
pub struct RunResult {
    pub requested_source: String,
    pub resolved_source: String,
    pub address: String,
    pub container_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EffectiveSandboxSpec {
    pub schema_version: u32,
    pub requested_source: EnvironmentSourceRef,
    pub resolved_source: source::ResolvedEnvironmentSourceRef,
    pub base_image: String,
    pub rlmesh_package: ResolvedRlmeshPackage,
    pub packages: Vec<String>,
    pub imports: Vec<String>,
    pub kwargs: BTreeMap<String, serde_json::Value>,
    pub num_envs: usize,
    pub vectorization_mode: VectorizationMode,
    /// The parsed recipe for a `recipe://` source; `None` for gym/hf. When set,
    /// its build phase drives the Dockerfile (the deriver) and its runtime phase
    /// (setup/make/requires) rides the bootstrap payload.
    pub recipe: Option<recipe::Recipe>,
    /// The recipe author's source-tree root, for `ProjectInstall` staging.
    pub context_root: Option<PathBuf>,
    pub build_hash: String,
}

impl EffectiveSandboxSpec {
    fn resolve(
        source: EnvironmentSourceRef,
        options: SandboxOptions,
    ) -> std::result::Result<Self, SandboxError> {
        // A recipe source carries its build phase in the document; parse it up
        // front, gate its build by provenance, and let build.base override the
        // image. Gym/hf sources leave `recipe` None.
        let recipe = match &source {
            EnvironmentSourceRef::Recipe(reference) => {
                let parsed =
                    recipe::Recipe::from_json(&reference.document.to_string()).map_err(|err| {
                        SandboxError::invalid_source(format!("invalid recipe document: {err}"))
                    })?;
                validate_recipe_build(
                    &parsed.build,
                    reference.provenance,
                    options.trust_remote_code,
                )
                .map_err(SandboxError::recipe_build_policy)?;
                Some(parsed)
            }
            _ => None,
        };
        // Provenance is build-relevant for a recipe (it changes the derived
        // Dockerfile, see BuildHashInput.provenance); gym/hf sources leave it None.
        let recipe_provenance = match &source {
            EnvironmentSourceRef::Recipe(reference) => Some(reference.provenance),
            _ => None,
        };
        let context_root = options.context_root.clone();

        let recipe_base = recipe.as_ref().and_then(|r| r.build.base.clone());
        let base_image = match recipe_base {
            Some(base) => validate_nonempty("base_image", base),
            None => validate_nonempty("base_image", options.resolved_base_image()),
        }
        .map_err(SandboxError::invalid_option)?;
        let rlmesh_package = options
            .resolved_rlmesh_package(&base_image)
            .map_err(SandboxError::wheel)?;
        let packages =
            validate_specs("packages", options.packages).map_err(SandboxError::invalid_option)?;
        let imports =
            validate_specs("imports", options.imports).map_err(SandboxError::invalid_option)?;
        validate_source_trust(
            &source,
            options.trust_remote_code,
            options.allow_unpinned_hf,
        )
        .map_err(SandboxError::huggingface_policy)?;
        let kwargs = options.kwargs;
        let num_envs = validate_num_envs(options.num_envs).map_err(SandboxError::invalid_option)?;
        let vectorization_mode = options.vectorization_mode;
        let resolved_source = resolve_source(&source).map_err(SandboxError::source_resolution)?;

        // (7.1A) When the recipe stages a host tree (ProjectInstall), fold a
        // content digest of that tree into the build hash so editing the source
        // -- with the recipe JSON unchanged -- rebuilds the image instead of
        // silently reusing a stale one.
        let content_digest =
            recipe::recipe_content_digest(recipe.as_ref(), context_root.as_deref())
                .map_err(SandboxError::invalid_option)?;

        // build_hash deliberately excludes runtime-only parameters (kwargs,
        // num_envs, vectorization_mode): they are delivered to the container at
        // `docker run` time via the bootstrap payload, never baked into the
        // image, so changing them must not produce a new image tag or trigger a
        // rebuild.
        let build_hash = build_hash(&BuildHashInput {
            schema_version: BOOTSTRAP_SCHEMA_VERSION,
            // A recipe's image is keyed by its build phase, not its task identity,
            // so a from_recipe family shares one image.
            source: if recipe.is_some() {
                None
            } else {
                Some(&resolved_source)
            },
            base_image: &base_image,
            rlmesh_package: &rlmesh_package,
            packages: &packages,
            imports: &imports,
            build: recipe.as_ref().map(|r| &r.build),
            provenance: recipe_provenance,
            content_digest: content_digest.as_deref(),
        })
        .map_err(SandboxError::invalid_option)?;

        Ok(Self {
            schema_version: BOOTSTRAP_SCHEMA_VERSION,
            requested_source: source,
            resolved_source,
            base_image,
            rlmesh_package,
            packages,
            imports,
            kwargs,
            num_envs,
            vectorization_mode,
            recipe,
            context_root,
            build_hash,
        })
    }

    pub(crate) fn slug(&self) -> String {
        self.resolved_source.slug()
    }

    /// The image-tag slug. For a recipe source this is a constant, so the image
    /// is keyed purely by `build_hash` (the build phase) and a from_recipe family
    /// shares one image; gym/hf keep their per-source slug.
    pub(crate) fn image_slug(&self) -> String {
        match &self.resolved_source {
            source::ResolvedEnvironmentSourceRef::Recipe(_) => "recipe".to_string(),
            _ => self.resolved_source.slug(),
        }
    }

    pub(crate) fn requested_display(&self) -> String {
        self.requested_source.to_string()
    }

    pub(crate) fn resolved_display(&self) -> String {
        self.resolved_source.to_string()
    }
}

#[derive(Serialize)]
struct BuildHashInput<'a> {
    schema_version: u32,
    /// The resolved source identity for gym/hf. `None` for a recipe source: a
    /// recipe's image is determined by its build phase alone (the per-task name,
    /// make, and setup ride the runtime bootstrap), so an N-task family with one
    /// inlined `from_recipe` build shares a single image/build_hash.
    source: Option<&'a source::ResolvedEnvironmentSourceRef>,
    base_image: &'a str,
    rlmesh_package: &'a ResolvedRlmeshPackage,
    packages: &'a [String],
    imports: &'a [String],
    /// The recipe build phase (None for gym/hf). A build change rebuilds.
    build: Option<&'a recipe::Build>,
    /// The recipe provenance (None for gym/hf). The deriver emits a DIFFERENT
    /// Dockerfile by provenance -- a Remote recipe skips the implicit unpinned
    /// `gymnasium` install -- so provenance must key the image too. Without this,
    /// an Installed and a Remote recipe with a byte-identical build phase would
    /// collide on one cached image tag and reuse the wrong Dockerfile.
    provenance: Option<RecipeProvenance>,
    /// A content digest of the staged `ProjectInstall` tree (7.1A).
    content_digest: Option<&'a str>,
}

/// Gate a recipe's build phase by provenance (spec section 2 / 7.1E / 7.1G).
///
/// `Installed` recipes build anything (a build is no more privileged than the
/// Dockerfile the package would otherwise hand-write). A `Remote` recipe's build
/// is pinned and restricted: free-form `commands`, `from_recipe`, and
/// `ProjectInstall` are rejected outright; every `Fetch` must be pinned; and
/// every `build.pip` step must be version-pinned with an allowlisted index and a
/// digest-pinned base.
fn validate_recipe_build(
    build: &recipe::Build,
    provenance: RecipeProvenance,
    _trust_remote_code: bool,
) -> Result<()> {
    if provenance == RecipeProvenance::Installed {
        return Ok(());
    }

    // Remote: free-form shell and host-tree references can never be pinned.
    anyhow::ensure!(
        build.commands.is_empty(),
        "a Remote recipe must not carry build.commands (no pinning a free-form shell line)"
    );
    anyhow::ensure!(
        build.from_recipe.is_none(),
        "a Remote recipe must not use build.from_recipe (name-confusion substitution vector)"
    );
    anyhow::ensure!(
        build.project.is_none(),
        "a Remote recipe must not use a ProjectInstall (there is no host tree to read)"
    );
    anyhow::ensure!(
        build.dockerfile.is_none(),
        "a Remote recipe must not carry a verbatim build.dockerfile (unpinnable)"
    );

    // Remote fetches must be pinned (a 40-char git ref / a url sha256).
    for fetch in &build.fetch {
        match fetch.kind.as_str() {
            "git" => anyhow::ensure!(
                fetch.ref_.as_deref().is_some_and(looks_like_full_git_sha),
                "a Remote recipe's git fetch must pin a full 40-character commit ref"
            ),
            "url" => anyhow::ensure!(
                fetch.sha256.is_some(),
                "a Remote recipe's url fetch must pin a sha256"
            ),
            other => anyhow::bail!("unknown fetch kind {other:?}"),
        }
    }

    // (7.1G) Remote build.pip must be version-pinned with an allowlisted index;
    // the base, if set, must be digest-pinned.
    for step in &build.pip {
        // A `-r requirements.txt` smuggles packages whose contents cannot be
        // version-pinned or index-allowlisted here, so it bypasses the gate
        // below; reject it outright for Remote recipes.
        anyhow::ensure!(
            step.requirements.is_none(),
            "a Remote recipe's build.pip must not use a -r requirements file (its contents cannot be version-pinned or index-allowlisted)"
        );
        for package in &step.packages {
            // A `name @ url` direct reference makes pip install from `url` and
            // ignore the index entirely, so it cannot be index-allowlisted; the
            // `==` inside the url would also spuriously satisfy the version-pin
            // check below. Reject it before that check (on the pre-marker
            // portion, since a marker may legitimately contain '@').
            let requirement = package.split(';').next().unwrap_or(package);
            // An option-shaped entry (a PEP 508 requirement never starts with
            // '-') smuggles a pip flag through `packages`: pip's optparse reads
            // `--extra-index-url=URL` as a real option, redirecting to an
            // un-allowlisted index. Reject any leading-dash token outright.
            anyhow::ensure!(
                !requirement.trim_start().starts_with('-'),
                "a Remote recipe's build.pip package must be a requirement, not a pip option/flag (got {package:?})"
            );
            anyhow::ensure!(
                !requirement.contains('@'),
                "a Remote recipe's build.pip must not use a direct URL/path reference ('name @ url'); it bypasses the index allowlist (got {package:?})"
            );
            anyhow::ensure!(
                requirement_is_version_pinned(package),
                "a Remote recipe's build.pip must version-pin every package (got {package:?})"
            );
        }
        for index in step.index_url.iter().chain(step.extra_index_urls.iter()) {
            anyhow::ensure!(
                is_allowlisted_index(index),
                "a Remote recipe's pip index {index:?} is not on the allowlist"
            );
        }
    }
    // A Remote recipe must DECLARE its base: omitting it would fall back to the
    // default (or caller) image -- a mutable tag like `python:3.11-slim` -- which
    // is not reproducible, defeating the pinning guarantee above.
    let Some(base) = &build.base else {
        anyhow::bail!(
            "a Remote recipe must declare a digest-pinned build.base (omitting it falls back to a mutable default tag)"
        );
    };
    anyhow::ensure!(
        base.contains("@sha256:"),
        "a Remote recipe's base image must be digest-pinned (got {base:?})"
    );

    Ok(())
}

/// The default pip-index allowlist for Remote recipes (PyPI + the two indices
/// real GPU recipes need). A hosted catalog may extend this.
fn is_allowlisted_index(url: &str) -> bool {
    const ALLOWED: [&str; 3] = [
        "https://pypi.org/simple",
        "https://download.pytorch.org/whl",
        "https://pypi.nvidia.com",
    ];
    ALLOWED.iter().any(|prefix| index_url_matches(url, prefix))
}

/// Whether `url` is the allowlisted `prefix` itself or a path under it, anchored
/// at a real `/` boundary so a sibling/look-alike host cannot slip through. An
/// unanchored `starts_with` would let `https://pypi.nvidia.com.evil.example/simple`
/// match `https://pypi.nvidia.com`; requiring the next char be `/` (or end of
/// string) closes that bypass.
fn index_url_matches(url: &str, prefix: &str) -> bool {
    match url.strip_prefix(prefix) {
        Some("") => true,
        Some(rest) => rest.starts_with('/'),
        None => false,
    }
}

/// Whether a PEP 508 requirement string version-pins its package NAME.
///
/// The check must look at the requirement portion only: an environment marker
/// (the part after `;`) like `torch ; os_name == 'posix'` contains a `==` that
/// would spuriously satisfy a naive `contains("==")` while leaving `torch`
/// itself unpinned. So strip the marker first, then require a real version pin
/// (`==` or `===`) in the requirement portion.
fn requirement_is_version_pinned(package: &str) -> bool {
    let requirement = package.split(';').next().unwrap_or(package);
    requirement.contains("==")
}

fn validate_source_trust(
    source: &EnvironmentSourceRef,
    trust_remote_code: bool,
    allow_unpinned_hf: bool,
) -> Result<()> {
    let EnvironmentSourceRef::Hf(source) = source else {
        return Ok(());
    };

    anyhow::ensure!(
        trust_remote_code,
        "hf:// sandbox sources execute remote code from env.py and requirements.txt; pass trust_remote_code=True only for sources you trust"
    );

    if allow_unpinned_hf {
        return Ok(());
    }

    let Some(revision) = source.revision.as_deref() else {
        anyhow::bail!(
            "hf:// sandbox sources must pin a full 40-character git SHA by default; pass allow_unpinned_hf=True to opt into branch/tag resolution"
        );
    };
    anyhow::ensure!(
        looks_like_full_git_sha(revision),
        "hf:// sandbox revision must be a full 40-character git SHA by default; pass allow_unpinned_hf=True to opt into branch/tag resolution"
    );

    Ok(())
}

fn resolve_source(source: &EnvironmentSourceRef) -> Result<source::ResolvedEnvironmentSourceRef> {
    match source {
        EnvironmentSourceRef::Gym(source) => {
            Ok(source::ResolvedEnvironmentSourceRef::Gym(source.clone()))
        }
        EnvironmentSourceRef::Hf(source) => {
            let resolved_revision = hf::resolve_revision(source)?;
            Ok(source::ResolvedEnvironmentSourceRef::Hf(
                source::ResolvedHfSourceRef {
                    repo: source.repo.clone(),
                    resolved_revision,
                    suite: source.suite.clone(),
                    task: source.task.clone(),
                },
            ))
        }
        // A recipe arrives already-structured; there is no remote revision to
        // resolve, so it passes through unchanged.
        EnvironmentSourceRef::Recipe(source) => {
            Ok(source::ResolvedEnvironmentSourceRef::Recipe(source.clone()))
        }
    }
}

/// Build the sandbox image and start a container for `source`.
///
/// This is a synchronous convenience wrapper around [`start_env_async`]. It
/// must not be called from within an existing tokio runtime: it creates its
/// own runtime internally and will panic ("Cannot start a runtime from within
/// a runtime") if one is already active. From async code, call
/// [`start_env_async`] directly.
pub fn start_env(
    source: EnvironmentSourceRef,
    options: SandboxOptions,
) -> std::result::Result<RunResult, SandboxError> {
    let runtime = tokio::runtime::Runtime::new().map_err(|err| {
        SandboxError::container_startup(format!("failed to create runtime: {err}"))
    })?;
    runtime.block_on(start_env_async(source, options))
}

/// Build the sandbox image and start a container for `source`.
///
/// This is the async-first entry point; it is safe to call from inside a tokio
/// runtime. The synchronous [`start_env`] wrapper is provided for convenience
/// and must not be called from an async context.
pub async fn start_env_async(
    source: EnvironmentSourceRef,
    options: SandboxOptions,
) -> std::result::Result<RunResult, SandboxError> {
    let spec = EffectiveSandboxSpec::resolve(source, options)?;
    let docker = docker::DockerBackend;
    let artifact = docker.ensure_image(&spec).map_err(|err| {
        SandboxError::from_docker_op(err, |m| SandboxError::ImageBuild { message: m })
    })?;
    let started = docker
        .run_container_async(&spec, &artifact)
        .await
        .map_err(|err| {
            SandboxError::from_docker_op(err, |m| SandboxError::ContainerStartup { message: m })
        })?;

    Ok(RunResult {
        requested_source: spec.requested_display(),
        resolved_source: spec.resolved_display(),
        address: started.address,
        container_id: started.container_id,
    })
}

/// Stop and remove a sandbox container by id.
pub fn stop_container(container_id: &str) -> std::result::Result<(), SandboxError> {
    docker::DockerBackend
        .stop_container(container_id)
        .map_err(|err| SandboxError::from_docker_op(err, |m| SandboxError::Docker { message: m }))
}

/// Best-effort reap of orphaned rlmesh-owned sandbox containers.
///
/// Only containers whose owner process has exited are removed, so this is safe
/// to call while other live rlmesh processes hold running sessions. Returns the
/// ids that were removed.
pub fn reap_orphaned_containers() -> std::result::Result<Vec<String>, SandboxError> {
    docker::DockerBackend
        .reap_orphaned_containers()
        .map_err(|err| SandboxError::from_docker_op(err, |m| SandboxError::Docker { message: m }))
}

pub fn default_rlmesh_package() -> String {
    format!(
        "{DEFAULT_PACKAGE_NAME}=={}",
        python_package_version(env!("CARGO_PKG_VERSION"))
    )
}

fn python_package_version(version: &str) -> String {
    if let Some((base, suffix)) = version.split_once("-alpha.") {
        return format!("{base}a{suffix}");
    }
    if let Some((base, suffix)) = version.split_once("-beta.") {
        return format!("{base}b{suffix}");
    }
    if let Some((base, suffix)) = version.split_once("-rc.") {
        return format!("{base}rc{suffix}");
    }
    version.to_string()
}

fn build_hash(input: &BuildHashInput<'_>) -> Result<String> {
    let raw = serde_json::to_vec(input)?;
    let mut hasher = Sha256::new();
    hasher.update(raw);
    Ok(hex(&hasher.finalize()))
}

/// Lowercase-hex encode a byte slice (e.g. a SHA-256 digest).
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut acc, byte| {
        let _ = write!(acc, "{byte:02x}");
        acc
    })
}

/// Quote a token for safe single-argument use in a `/bin/sh` command.
pub(crate) fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn validate_nonempty(label: &str, value: String) -> Result<String> {
    let value = value.trim().to_string();
    anyhow::ensure!(!value.is_empty(), "{label} must not be empty");
    anyhow::ensure!(
        !value.contains('\n') && !value.contains('\r'),
        "{label} must not contain newlines"
    );
    Ok(value)
}

fn validate_specs(label: &str, values: Vec<String>) -> Result<Vec<String>> {
    values
        .into_iter()
        .map(|value| validate_nonempty(label, value))
        .collect()
}

fn validate_num_envs(value: usize) -> Result<usize> {
    anyhow::ensure!(value > 0, "num_envs must be at least 1");
    Ok(value)
}

pub(crate) fn looks_like_full_git_sha(value: &str) -> bool {
    value.len() == 40 && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_version_uses_pep440_prereleases() {
        assert_eq!(python_package_version("0.1.0-alpha.1"), "0.1.0a1");
        assert_eq!(python_package_version("0.1.0-beta.2"), "0.1.0b2");
        assert_eq!(python_package_version("0.1.0-rc.3"), "0.1.0rc3");
        assert_eq!(python_package_version("0.1.0"), "0.1.0");
    }

    #[test]
    fn gym_sources_do_not_require_remote_code_trust() {
        let source = EnvironmentSourceRef::parse("CartPole-v1").unwrap();
        validate_source_trust(&source, false, false).unwrap();
    }

    #[test]
    fn hf_sources_require_explicit_remote_code_trust() {
        let source =
            EnvironmentSourceRef::parse("hf://org/repo@0123456789abcdef0123456789abcdef01234567")
                .unwrap();
        let err = validate_source_trust(&source, false, false).unwrap_err();
        assert!(err.to_string().contains("trust_remote_code=True"));
    }

    #[test]
    fn hf_sources_require_full_sha_unless_unpinned_is_allowed() {
        let source = EnvironmentSourceRef::parse("hf://org/repo@main").unwrap();
        let err = validate_source_trust(&source, true, false).unwrap_err();
        assert!(err.to_string().contains("40-character git SHA"));
        validate_source_trust(&source, true, true).unwrap();
    }

    #[test]
    fn hf_sources_accept_full_sha_when_trusted() {
        let source =
            EnvironmentSourceRef::parse("hf://org/repo@0123456789abcdef0123456789abcdef01234567")
                .unwrap();
        validate_source_trust(&source, true, false).unwrap();
    }

    #[test]
    fn build_hash_changes_when_inputs_change() {
        let source = EnvironmentSourceRef::parse("CartPole-v1").unwrap();
        let base =
            EffectiveSandboxSpec::resolve(source.clone(), SandboxOptions::default()).unwrap();
        let changed = EffectiveSandboxSpec::resolve(
            source,
            SandboxOptions {
                base_image: Some("python:3.12-slim".to_string()),
                ..SandboxOptions::default()
            },
        )
        .unwrap();

        assert_ne!(base.build_hash, changed.build_hash);
    }

    #[test]
    fn build_hash_keys_recipe_provenance() {
        // The deriver emits a different Dockerfile by provenance (a Remote recipe
        // skips the implicit unpinned gymnasium), so two specs with a byte-identical
        // build phase but different provenance must not share an image tag; otherwise
        // ensure_image would reuse the wrong (Installed/Remote) Dockerfile.
        let document = serde_json::json!({
            "name": "a/b",
            "make": {"kind": "gym", "env_id": "E-v0"},
            "build": {"base": "python@sha256:abc"},
        });
        let installed = EffectiveSandboxSpec::resolve(
            recipe_source(document.clone(), RecipeProvenance::Installed),
            SandboxOptions::default(),
        )
        .unwrap();
        let remote = EffectiveSandboxSpec::resolve(
            recipe_source(document, RecipeProvenance::Remote),
            SandboxOptions::default(),
        )
        .unwrap();
        assert_ne!(installed.build_hash, remote.build_hash);
    }

    #[test]
    fn public_errors_are_typed_and_discriminable() {
        // num_envs == 0 must surface as a typed InvalidOption, not a stringly error.
        let err = EffectiveSandboxSpec::resolve(
            EnvironmentSourceRef::parse("CartPole-v1").unwrap(),
            SandboxOptions {
                num_envs: 0,
                ..SandboxOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, SandboxError::InvalidOption { .. }));

        // Unpinned hf source surfaces as a HuggingFacePolicy error.
        let err = EffectiveSandboxSpec::resolve(
            EnvironmentSourceRef::parse("hf://org/repo@main").unwrap(),
            SandboxOptions {
                trust_remote_code: true,
                ..SandboxOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, SandboxError::HuggingFacePolicy { .. }));

        // vectorization_mode parse failures are typed too.
        let err = VectorizationMode::parse(Some("parallel")).unwrap_err();
        assert!(matches!(err, SandboxError::InvalidOption { .. }));

        // Source parse failures are typed.
        let err = EnvironmentSourceRef::parse("ftp://nope").unwrap_err();
        assert!(matches!(err, SandboxError::InvalidSource { .. }));
    }

    #[test]
    fn build_hash_is_stable_across_runtime_only_params() {
        // kwargs, num_envs, and vectorization_mode are delivered at run time
        // and must not change the image tag, otherwise every gym.make kwarg
        // tweak rebuilds the image and re-downloads the pip layers.
        let source = EnvironmentSourceRef::parse("CartPole-v1").unwrap();
        let base =
            EffectiveSandboxSpec::resolve(source.clone(), SandboxOptions::default()).unwrap();

        let mut kwargs = BTreeMap::new();
        kwargs.insert("render_mode".to_string(), serde_json::json!("rgb_array"));
        let with_runtime_params = EffectiveSandboxSpec::resolve(
            source,
            SandboxOptions {
                kwargs,
                num_envs: 8,
                vectorization_mode: VectorizationMode::Async,
                ..SandboxOptions::default()
            },
        )
        .unwrap();

        assert_eq!(base.build_hash, with_runtime_params.build_hash);
    }

    #[test]
    fn hf_task_changes_resolved_display_slug_and_build_hash() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let options = SandboxOptions {
            trust_remote_code: true,
            ..SandboxOptions::default()
        };
        let first = EffectiveSandboxSpec::resolve(
            EnvironmentSourceRef::parse(&format!("hf://org/repo@{revision}:suite/0")).unwrap(),
            options.clone(),
        )
        .unwrap();
        let second = EffectiveSandboxSpec::resolve(
            EnvironmentSourceRef::parse(&format!("hf://org/repo@{revision}:suite/1")).unwrap(),
            options,
        )
        .unwrap();

        assert_eq!(
            first.resolved_display(),
            format!("hf://org/repo@{revision}:suite/0")
        );
        assert_eq!(first.slug(), "org-repo-suite-0");
        assert_ne!(first.build_hash, second.build_hash);
    }

    fn recipe_source(
        document: serde_json::Value,
        provenance: RecipeProvenance,
    ) -> EnvironmentSourceRef {
        EnvironmentSourceRef::Recipe(RecipeSourceRef {
            name: "acme/env".to_string(),
            document,
            provenance,
        })
    }

    fn gym_recipe_document() -> serde_json::Value {
        serde_json::json!({
            "name": "acme/env",
            "make": {"kind": "gym", "env_id": "CartPole-v1"},
            "build": {"pip": [{"packages": ["pygame"]}]},
            "requires": {"imports": ["my_envs"]}
        })
    }

    #[test]
    fn recipe_source_resolves_and_derives_its_dockerfile() {
        let source = recipe_source(gym_recipe_document(), RecipeProvenance::Installed);
        let spec = EffectiveSandboxSpec::resolve(source, SandboxOptions::default()).unwrap();
        assert!(spec.recipe.is_some());
        assert_eq!(spec.slug(), "acme-env");
        assert_eq!(spec.resolved_display(), "recipe://acme/env");
    }

    #[test]
    fn recipe_base_overrides_default_image() {
        let mut document = gym_recipe_document();
        document["build"]["base"] = serde_json::json!("nvidia/cuda:12.4.1-runtime-ubuntu22.04");
        let source = recipe_source(document, RecipeProvenance::Installed);
        let spec = EffectiveSandboxSpec::resolve(source, SandboxOptions::default()).unwrap();
        assert_eq!(spec.base_image, "nvidia/cuda:12.4.1-runtime-ubuntu22.04");
    }

    #[test]
    fn installed_recipe_build_passes_the_gate() {
        let build = recipe::Build {
            commands: vec!["echo hi".to_string()],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&build, RecipeProvenance::Installed, false).is_ok());
    }

    #[test]
    fn remote_recipe_rejects_commands_and_unpinned_fetch() {
        let with_commands = recipe::Build {
            commands: vec!["echo hi".to_string()],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&with_commands, RecipeProvenance::Remote, true).is_err());

        let unpinned_fetch = recipe::Build {
            fetch: vec![recipe::Fetch {
                kind: "git".to_string(),
                repo: Some("https://x/r.git".to_string()),
                ref_: Some("main".to_string()),
                ..recipe::Fetch::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&unpinned_fetch, RecipeProvenance::Remote, true).is_err());
    }

    #[test]
    fn remote_recipe_rejects_unpinned_pip_and_bad_index() {
        let unpinned = recipe::Build {
            pip: vec![recipe::PipInstall {
                packages: vec!["torch".to_string()],
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&unpinned, RecipeProvenance::Remote, true).is_err());

        let bad_index = recipe::Build {
            pip: vec![recipe::PipInstall {
                packages: vec!["torch==2.0.0".to_string()],
                index_url: Some("https://attacker.example/simple".to_string()),
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&bad_index, RecipeProvenance::Remote, true).is_err());
    }

    #[test]
    fn remote_recipe_rejects_pip_requirements_file() {
        // A `-r requirements.txt` smuggles packages past the version-pin and
        // index-allowlist gate, so a Remote recipe must reject it even when its
        // explicit packages are pinned.
        let with_requirements = recipe::Build {
            pip: vec![recipe::PipInstall {
                packages: vec!["torch==2.0.0".to_string()],
                requirements: Some("requirements.txt".to_string()),
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&with_requirements, RecipeProvenance::Remote, true).is_err());
        // An Installed recipe is unaffected (the gate returns Ok early).
        assert!(
            validate_recipe_build(&with_requirements, RecipeProvenance::Installed, true).is_ok()
        );
    }

    #[test]
    fn remote_recipe_rejects_direct_url_reference() {
        // A `name @ url` direct reference makes pip install from the url and
        // ignore the allowlisted index; the `==` inside the url would also
        // spuriously satisfy the version-pin check, so it must be rejected.
        let direct_ref = recipe::Build {
            pip: vec![recipe::PipInstall {
                packages: vec!["evil @ https://attacker.example/evil==1.0.whl".to_string()],
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&direct_ref, RecipeProvenance::Remote, true).is_err());
        // A normal version-pinned package and an extras+marker spec still pass
        // (with the digest-pinned base a Remote recipe now requires).
        let normal = recipe::Build {
            base: Some("python@sha256:abc".to_string()),
            pip: vec![recipe::PipInstall {
                packages: vec![
                    "pkg==1.0".to_string(),
                    "pkg[extra]==1.0 ; os_name == 'posix'".to_string(),
                ],
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&normal, RecipeProvenance::Remote, true).is_ok());
    }

    #[test]
    fn remote_recipe_requires_a_declared_pinned_base() {
        // Omitting build.base for a Remote recipe is rejected: it would fall back
        // to a mutable default tag, defeating reproducibility.
        let no_base = recipe::Build {
            pip: vec![recipe::PipInstall {
                packages: vec!["pkg==1.0".to_string()],
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&no_base, RecipeProvenance::Remote, true).is_err());
        // A digest-pinned base passes; a mutable-tag base is still rejected.
        let pinned = recipe::Build {
            base: Some("python@sha256:abc".to_string()),
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&pinned, RecipeProvenance::Remote, true).is_ok());
        let mutable = recipe::Build {
            base: Some("python:3.11-slim".to_string()),
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&mutable, RecipeProvenance::Remote, true).is_err());
        // Installed recipes are unaffected by the base requirement.
        assert!(validate_recipe_build(&no_base, RecipeProvenance::Installed, false).is_ok());
    }

    #[test]
    fn remote_recipe_rejects_option_shaped_package() {
        // An option-shaped `packages` entry smuggles a pip flag: pip's optparse
        // reads `--extra-index-url=URL` as a real option (the trailing `==`
        // satisfies the version-pin check), redirecting to an un-allowlisted
        // index. A leading-dash token must be rejected before that check.
        for smuggled in [
            "--extra-index-url=https://evil.example/s==1.0",
            "--index-url=https://evil.example/s==1.0",
        ] {
            let build = recipe::Build {
                pip: vec![recipe::PipInstall {
                    packages: vec![smuggled.to_string()],
                    ..recipe::PipInstall::default()
                }],
                ..recipe::Build::default()
            };
            assert!(
                validate_recipe_build(&build, RecipeProvenance::Remote, true).is_err(),
                "{smuggled:?} should be rejected for a Remote recipe"
            );
        }
    }

    #[test]
    fn remote_recipe_accepts_fully_pinned_build() {
        let build = recipe::Build {
            base: Some("python@sha256:abc".to_string()),
            pip: vec![recipe::PipInstall {
                packages: vec!["torch==2.0.0".to_string()],
                index_url: Some("https://download.pytorch.org/whl/cu124".to_string()),
                ..recipe::PipInstall::default()
            }],
            fetch: vec![recipe::Fetch {
                kind: "git".to_string(),
                repo: Some("https://x/r.git".to_string()),
                ref_: Some("a".repeat(40)),
                dest: "/opt/r".to_string(),
                ..recipe::Fetch::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&build, RecipeProvenance::Remote, true).is_ok());
    }

    #[test]
    fn from_recipe_family_shares_build_hash_and_image_slug() {
        let shared_build = serde_json::json!({"base":"nvidia/cuda:12.4.1-runtime-ubuntu22.04","system":["cmake"],"gpu":true});
        let resolve_scene = |name: &str, entrypoint: &str, build: serde_json::Value| {
            let document = serde_json::json!({
                "name": name,
                "make": {"kind": "py", "entrypoint": entrypoint},
                "build": build,
            });
            EffectiveSandboxSpec::resolve(
                recipe_source(document, RecipeProvenance::Installed),
                SandboxOptions::default(),
            )
            .unwrap()
        };

        let scene1 = resolve_scene("droid/scene1", "robot_env:s1", shared_build.clone());
        let scene2 = resolve_scene("droid/scene2", "robot_env:s2", shared_build.clone());
        // Same inlined build -> one image, despite different task names/factories.
        assert_eq!(scene1.build_hash, scene2.build_hash);
        assert_eq!(scene1.image_slug(), "recipe");

        // A different build phase -> a different image.
        let other = resolve_scene(
            "droid/scene3",
            "robot_env:s3",
            serde_json::json!({"gpu": false}),
        );
        assert_ne!(scene1.build_hash, other.build_hash);
    }

    #[test]
    fn allowlist_anchors_at_a_path_boundary() {
        // Exact match and a real sub-path are allowed.
        assert!(is_allowlisted_index("https://pypi.nvidia.com"));
        assert!(is_allowlisted_index("https://pypi.nvidia.com/simple"));
        assert!(is_allowlisted_index(
            "https://download.pytorch.org/whl/cu124"
        ));
        // A look-alike host that merely has the allowlisted entry as a string
        // prefix must not match (the bypass this fix closes).
        assert!(!is_allowlisted_index(
            "https://pypi.nvidia.com.evil.example/simple"
        ));
        assert!(!is_allowlisted_index("https://attacker.example/simple"));
    }

    #[test]
    fn version_pin_check_is_marker_aware() {
        // A real pin passes.
        assert!(requirement_is_version_pinned("torch==2.0.0"));
        assert!(requirement_is_version_pinned("torch===2.0.0"));
        assert!(requirement_is_version_pinned(
            "torch==2.0.0 ; os_name == 'posix'"
        ));
        // An unpinned package whose only `==` lives in an environment marker must
        // be rejected (the bypass this fix closes).
        assert!(!requirement_is_version_pinned("torch ; os_name == 'posix'"));
        assert!(!requirement_is_version_pinned("torch"));
    }

    #[test]
    fn remote_recipe_rejects_marker_only_equals_as_unpinned() {
        // End-to-end through the gate: a marker `==` must not satisfy the pin.
        let build = recipe::Build {
            pip: vec![recipe::PipInstall {
                packages: vec!["torch ; os_name == 'posix'".to_string()],
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&build, RecipeProvenance::Remote, true).is_err());
    }

    #[test]
    fn remote_recipe_rejects_lookalike_allowlist_host() {
        // End-to-end through the gate: a look-alike host index must be rejected.
        let build = recipe::Build {
            pip: vec![recipe::PipInstall {
                packages: vec!["torch==2.0.0".to_string()],
                index_url: Some("https://pypi.nvidia.com.evil.example/simple".to_string()),
                ..recipe::PipInstall::default()
            }],
            ..recipe::Build::default()
        };
        assert!(validate_recipe_build(&build, RecipeProvenance::Remote, true).is_err());
    }
}
