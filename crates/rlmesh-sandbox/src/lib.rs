mod docker;
mod error;
mod hf;
mod source;
mod wheel;

use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use error::SandboxError;
pub use source::{EnvironmentSourceRef, GymSourceRef, HfSourceRef};
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
    /// Opt-in memory ceiling for the build. `None` builds via the default docker
    /// builder (today's behaviour). A docker size string (e.g. `"20g"`) or the
    /// literal `"auto"` routes the build through a bounded `docker-container`
    /// buildx builder so an OOM is a clean cgroup-local build failure instead of
    /// a host freeze. Host-relative, never baked, so it stays out of the build
    /// hash.
    pub build_memory: Option<String>,
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
            build_memory: None,
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

    /// The requested build memory ceiling: field, else
    /// `RLMESH_SANDBOX_BUILD_MEMORY`, else `None`. The raw value (a docker size
    /// string or the literal `"auto"`/`"off"`) is interpreted by the docker
    /// backend; this only applies the field-over-env precedence, mirroring
    /// [`resolved_base_image`](Self::resolved_base_image).
    pub fn resolved_build_memory(&self) -> Option<String> {
        self.build_memory
            .clone()
            .or_else(|| std::env::var("RLMESH_SANDBOX_BUILD_MEMORY").ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
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
    /// Opt-in build memory ceiling (see [`SandboxOptions::build_memory`]).
    /// Excluded from `build_hash` -- host-relative, never baked into the image.
    pub build_memory: Option<String>,
    pub build_hash: String,
}

impl EffectiveSandboxSpec {
    fn resolve(
        source: EnvironmentSourceRef,
        options: SandboxOptions,
    ) -> std::result::Result<Self, SandboxError> {
        let build_memory = options.resolved_build_memory();

        let base_image = validate_nonempty("base_image", options.resolved_base_image())
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

        // build_hash deliberately excludes runtime-only parameters (kwargs,
        // num_envs, vectorization_mode): they are delivered to the container at
        // `docker run` time via the bootstrap payload, never baked into the
        // image, so changing them must not produce a new image tag or trigger a
        // rebuild.
        let build_hash = build_hash(&BuildHashInput {
            schema_version: BOOTSTRAP_SCHEMA_VERSION,
            source: &resolved_source,
            base_image: &base_image,
            rlmesh_package: &rlmesh_package,
            packages: &packages,
            imports: &imports,
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
            build_memory,
            build_hash,
        })
    }

    pub(crate) fn slug(&self) -> String {
        self.resolved_source.slug()
    }

    /// The deterministic local image reference for this spec -- the single
    /// source of truth for both the build (`ensure_image`) and the export tag.
    pub(crate) fn image_tag(&self) -> String {
        format!(
            "rlmesh-sandbox-{}:{}",
            self.slug(),
            &self.build_hash[..12.min(self.build_hash.len())]
        )
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
    source: &'a source::ResolvedEnvironmentSourceRef,
    base_image: &'a str,
    rlmesh_package: &'a ResolvedRlmeshPackage,
    packages: &'a [String],
    imports: &'a [String],
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
    // Best-effort: sweep containers orphaned by a prior hard kill before
    // starting a new one. Label-keyed and env-agnostic, so this also reclaims
    // orphaned model containers. A reaper failure must never fail the start.
    if let Err(err) = docker.reap_orphaned_containers() {
        tracing::debug!("orphan reap before sandbox start failed: {err:#}");
    }
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
}
