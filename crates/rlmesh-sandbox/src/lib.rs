mod docker;
mod error;
mod hf;
pub mod recipe;
mod source;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use error::SandboxError;
pub use source::{EnvironmentSourceRef, GymSourceRef, HfSourceRef};

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

        resolve_rlmesh_package(validate_nonempty("rlmesh_package", selected)?, base_image)
    }
}

/// Details of a started sandbox container, returned by [`start_env`] and
/// [`start_env_async`].
///
/// Dropping this without recording `container_id` leaks a running container, so
/// it is `#[must_use]`. It is `#[non_exhaustive]` so future fields (extra
/// container metadata, ports, ...) can be added without breaking callers that
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
    pub build_hash: String,
}

impl EffectiveSandboxSpec {
    fn resolve(
        source: EnvironmentSourceRef,
        options: SandboxOptions,
    ) -> std::result::Result<Self, SandboxError> {
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
            build_hash,
        })
    }

    pub(crate) fn slug(&self) -> String {
        self.resolved_source.slug()
    }

    pub(crate) fn requested_display(&self) -> String {
        self.requested_source.requested_display()
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ResolvedRlmeshPackage {
    Pip {
        spec: String,
    },
    Wheel {
        source_path: PathBuf,
        install_path: String,
        sha256: String,
    },
}

impl ResolvedRlmeshPackage {
    pub(crate) fn install_ref(&self) -> &str {
        match self {
            Self::Pip { spec } => spec,
            Self::Wheel { install_path, .. } => install_path,
        }
    }

    pub(crate) fn source_path(&self) -> Option<&Path> {
        match self {
            Self::Pip { .. } => None,
            Self::Wheel { source_path, .. } => Some(source_path),
        }
    }
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
                    requested_revision: source.revision.clone(),
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
/// MUST NOT be called from within an existing tokio runtime: it creates its
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

fn resolve_rlmesh_package(value: String, base_image: &str) -> Result<ResolvedRlmeshPackage> {
    if value == "local" {
        let wheel = resolve_local_rlmesh_wheel(base_image)?;
        return resolved_wheel_package(&wheel);
    }

    let path = Path::new(&value);
    if !is_direct_url_package_spec(&value)
        && path.extension().and_then(|value| value.to_str()) == Some("whl")
    {
        return resolved_wheel_package(path);
    }

    Ok(ResolvedRlmeshPackage::Pip { spec: value })
}

fn is_direct_url_package_spec(value: &str) -> bool {
    value.contains("://")
}

fn resolved_wheel_package(path: &Path) -> Result<ResolvedRlmeshPackage> {
    let source_path = fs::canonicalize(path)
        .with_context(|| format!("failed to resolve RLMesh wheel path {}", path.display()))?;
    anyhow::ensure!(
        source_path.is_file(),
        "RLMesh wheel path must point to a file: {}",
        source_path.display()
    );
    anyhow::ensure!(
        source_path.extension().and_then(|value| value.to_str()) == Some("whl"),
        "RLMesh wheel path must end in .whl: {}",
        source_path.display()
    );

    let filename = source_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("RLMesh wheel path must have a UTF-8 filename"))?
        .to_string();
    let sha256 = file_sha256(&source_path)?;

    Ok(ResolvedRlmeshPackage::Wheel {
        source_path,
        install_path: format!("/opt/rlmesh/packages/{filename}"),
        sha256,
    })
}

fn resolve_local_rlmesh_wheel(base_image: &str) -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("failed to inspect current directory")?;
    let repo_root = find_repo_root(&cwd).ok_or_else(|| {
        anyhow::anyhow!(
            "rlmesh_package='local' must be run from inside an RLMesh checkout or use an explicit wheel path"
        )
    })?;
    let dist_dir = repo_root.join("python/rlmesh/dist");
    select_local_rlmesh_wheel(&dist_dir, base_image)
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|ancestor| {
        if ancestor.join("Cargo.toml").is_file()
            && ancestor.join("python/rlmesh/pyproject.toml").is_file()
        {
            Some(ancestor.to_path_buf())
        } else {
            None
        }
    })
}

/// The C library a base image links against, which determines whether a
/// manylinux (glibc) or musllinux (musl) wheel is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Libc {
    Glibc,
    Musl,
}

/// Compatibility constraints derived from a base image: the Python
/// `(major, minor)` version and the libc. Derived by parsing the image
/// reference robustly rather than substring-sniffing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImageCompat {
    python: (u32, u32),
    libc: Libc,
}

fn select_local_rlmesh_wheel(dist_dir: &Path, base_image: &str) -> Result<PathBuf> {
    let compat = resolve_image_compat(base_image)?;
    let entries = fs::read_dir(dist_dir).with_context(|| {
        format!(
            "failed to read RLMesh wheel directory {}",
            dist_dir.display()
        )
    })?;
    let mut candidates = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("whl"))
        .filter_map(|path| {
            let filename = path.file_name()?.to_str()?.to_string();
            wheel_rank(&filename, compat).map(|rank| (rank, filename, path))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    candidates
        .into_iter()
        .next()
        .map(|(_, _, path)| path)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "rlmesh_package='local' could not find a compatible Linux RLMesh wheel in {}; build one with `mise run release:python:wheels` or pass an explicit wheel path",
                dist_dir.display()
            )
        })
}

/// Derive the Python version and libc that wheels must be compatible with,
/// failing loudly when the base image does not declare a Python version that
/// can be parsed unambiguously (rather than guessing from a coincidental
/// substring like `12.3.10` in a CUDA tag).
fn resolve_image_compat(base_image: &str) -> Result<ImageCompat> {
    let python = parse_image_python_version(base_image).ok_or_else(|| {
        anyhow::anyhow!(
            "could not determine the Python version of base image '{base_image}' for rlmesh_package='local'; use an official python:X.Y image or pass an explicit wheel path"
        )
    })?;
    Ok(ImageCompat {
        python,
        libc: parse_image_libc(base_image),
    })
}

/// Parse the Python `(major, minor)` version from a base image reference.
///
/// Only recognizes unambiguous declarations: the tag of an official
/// `python`/`pypy` image (e.g. `python:3.11-slim`), or an explicit
/// `pythonX.Y` / `pyX.Y` token. Returns `None` otherwise so the caller can
/// fail loudly instead of matching a coincidental substring.
fn parse_image_python_version(base_image: &str) -> Option<(u32, u32)> {
    let (repo, tag) = match base_image.rsplit_once(':') {
        Some((repo, tag)) => (repo, tag),
        None => (base_image, ""),
    };

    // Official python image: the tag begins with the version (python:3.11-slim).
    let repo_name = repo.rsplit('/').next().unwrap_or(repo);
    if (repo_name == "python" || repo_name == "pypy")
        && !tag.is_empty()
        && let Some(version) = parse_version_prefix(tag)
    {
        return Some(version);
    }

    // Otherwise look for an explicit `python3.11` / `py3.11` token anywhere in
    // the reference, delimited so digits from an unrelated version cannot leak.
    for token in base_image.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '.')) {
        for prefix in ["python", "py"] {
            if let Some(rest) = token.strip_prefix(prefix)
                && let Some(version) = parse_version_prefix(rest)
            {
                return Some(version);
            }
        }
    }

    None
}

/// Parse a leading `X.Y` version from a string, ignoring any trailing suffix
/// (e.g. `3.11-slim` -> `(3, 11)`).
fn parse_version_prefix(value: &str) -> Option<(u32, u32)> {
    let mut chars = value.chars();
    let major: String = chars
        .by_ref()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    // `take_while` consumed the '.' separator; collect the minor digits.
    let minor: String = value
        .strip_prefix(&major)?
        .strip_prefix('.')?
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if major.is_empty() || minor.is_empty() {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Determine libc from delimited image-name tokens.
fn parse_image_libc(base_image: &str) -> Libc {
    let is_musl = base_image
        .split([':', '-', '/', '.', '_'])
        .any(|token| token == "alpine" || token == "musl");
    if is_musl { Libc::Musl } else { Libc::Glibc }
}

fn wheel_rank(filename: &str, compat: ImageCompat) -> Option<(usize, usize)> {
    let (python_tag, abi_tag, platform_tag) = wheel_tags(filename)?;
    let python_rank = python_wheel_rank(python_tag, abi_tag, compat.python)?;
    let platform_rank = platform_wheel_rank(platform_tag, compat.libc)?;
    Some((platform_rank, python_rank))
}

fn wheel_tags(filename: &str) -> Option<(&str, &str, &str)> {
    let stem = filename.strip_suffix(".whl")?;
    let parts = stem.split('-').collect::<Vec<_>>();
    if parts.len() < 5 || parts.first() != Some(&DEFAULT_PACKAGE_NAME) {
        return None;
    }
    Some((
        parts[parts.len() - 3],
        parts[parts.len() - 2],
        parts[parts.len() - 1],
    ))
}

/// Rank a wheel's python/abi tags against the target interpreter version.
/// A stable-ABI (`abi3`) wheel built for `cpXY` is accepted on any
/// interpreter `>= X.Y`; a version-specific (`cpXY`/`cpXY`) wheel must match
/// the interpreter exactly. Lower rank is preferred.
fn python_wheel_rank(python_tag: &str, abi_tag: &str, python: (u32, u32)) -> Option<usize> {
    let (target_major, target_minor) = python;
    let wheel_version = python_tag
        .strip_prefix("cp")
        .and_then(parse_cp_version)
        .or_else(|| python_tag.strip_prefix("pp").and_then(parse_cp_version))?;

    if abi_tag == "abi3" {
        // Stable ABI: forward-compatible with newer interpreters.
        let compatible = (target_major, target_minor) >= wheel_version;
        return compatible.then_some(0);
    }

    // Version-specific wheel: require an exact interpreter match.
    (wheel_version == (target_major, target_minor) && abi_tag == python_tag).then_some(1)
}

/// Parse a packed CPython version tag like `311` -> `(3, 11)` or `310` ->
/// `(3, 10)`. The major version is the first digit; the rest is the minor.
fn parse_cp_version(value: &str) -> Option<(u32, u32)> {
    if value.len() < 2 || !value.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let (major, minor) = value.split_at(1);
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn platform_wheel_rank(platform_tag: &str, libc: Libc) -> Option<usize> {
    if !platform_matches_host_arch(platform_tag) {
        return None;
    }
    match libc {
        Libc::Musl => platform_tag.starts_with("musllinux").then_some(0),
        Libc::Glibc => {
            if platform_tag.starts_with("manylinux") {
                Some(0)
            } else if platform_tag.starts_with("linux_") {
                Some(1)
            } else {
                None
            }
        }
    }
}

fn platform_matches_host_arch(platform_tag: &str) -> bool {
    match std::env::consts::ARCH {
        "aarch64" => platform_tag.contains("aarch64"),
        "x86_64" => platform_tag.contains("x86_64"),
        _ => false,
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read RLMesh wheel {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
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
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
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

fn looks_like_full_git_sha(value: &str) -> bool {
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
    fn resolves_explicit_wheel_package_and_hashes_contents() {
        let tempdir = tempfile::tempdir().unwrap();
        let wheel = tempdir
            .path()
            .join("rlmesh-0.1.0b2-cp311-abi3-manylinux_2_17_x86_64.whl");
        fs::write(&wheel, b"first").unwrap();
        let first = resolved_wheel_package(&wheel).unwrap();
        fs::write(&wheel, b"second").unwrap();
        let second = resolved_wheel_package(&wheel).unwrap();

        assert_eq!(
            first.install_ref(),
            "/opt/rlmesh/packages/rlmesh-0.1.0b2-cp311-abi3-manylinux_2_17_x86_64.whl"
        );
        assert_ne!(first, second);
    }

    #[test]
    fn preserves_direct_wheel_urls_as_pip_specs() {
        for spec in [
            "https://example.com/rlmesh-0.1.0b2-cp311-abi3-manylinux_2_17_x86_64.whl",
            "rlmesh @ https://example.com/rlmesh-0.1.0b2-cp311-abi3-manylinux_2_17_x86_64.whl",
        ] {
            let resolved = resolve_rlmesh_package(spec.to_string(), DEFAULT_BASE_IMAGE).unwrap();
            assert_eq!(
                resolved,
                ResolvedRlmeshPackage::Pip {
                    spec: spec.to_string()
                }
            );
        }
    }

    #[test]
    fn selects_local_manylinux_wheel_for_default_base_image() {
        let tempdir = tempfile::tempdir().unwrap();
        let arch = match std::env::consts::ARCH {
            "aarch64" => "aarch64",
            "x86_64" => "x86_64",
            _ => return,
        };
        fs::write(
            tempdir.path().join(format!(
                "rlmesh-0.1.0b2-cp311-abi3-musllinux_1_2_{arch}.whl"
            )),
            b"",
        )
        .unwrap();
        fs::write(
            tempdir.path().join(format!(
                "rlmesh-0.1.0b2-cp311-abi3-manylinux_2_17_{arch}.whl"
            )),
            b"",
        )
        .unwrap();

        let wheel = select_local_rlmesh_wheel(tempdir.path(), DEFAULT_BASE_IMAGE).unwrap();
        assert!(
            wheel
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("manylinux")
        );
    }

    #[test]
    fn parses_python_version_only_from_real_declarations() {
        assert_eq!(
            parse_image_python_version("python:3.11-slim"),
            Some((3, 11))
        );
        assert_eq!(parse_image_python_version("python:3.10"), Some((3, 10)));
        assert_eq!(
            parse_image_python_version("docker.io/library/python:3.12-bookworm"),
            Some((3, 12))
        );
        assert_eq!(
            parse_image_python_version("myimg:python3.11-cuda"),
            Some((3, 11))
        );
        // A coincidental "3.10" inside an unrelated version must NOT be read as
        // the Python version.
        assert_eq!(
            parse_image_python_version("nvidia/cuda:12.3.10-runtime-ubuntu22.04"),
            None
        );
        assert_eq!(parse_image_python_version("ubuntu:22.04"), None);
    }

    #[test]
    fn cuda_image_with_coincidental_version_substring_fails_loudly() {
        // base_image contains the substring "3.10" but is not a python image;
        // we must refuse to guess instead of demanding cp310 wheels.
        let err = resolve_image_compat("nvidia/cuda:12.3.10-runtime-ubuntu22.04").unwrap_err();
        assert!(
            err.to_string()
                .contains("could not determine the Python version")
        );
    }

    #[test]
    fn libc_detection_is_token_based() {
        assert_eq!(parse_image_libc("python:3.11-alpine"), Libc::Musl);
        assert_eq!(parse_image_libc("myrepo/img:musl"), Libc::Musl);
        assert_eq!(parse_image_libc("python:3.11-slim"), Libc::Glibc);
        // "musl" as part of a larger token (e.g. a hostname) is not musl libc.
        assert_eq!(
            parse_image_libc("muslorg.example.com/img:latest"),
            Libc::Glibc
        );
    }

    #[test]
    fn abi3_wheel_is_forward_compatible_with_newer_python() {
        // A cp311/abi3 wheel works on 3.11+ but not on an older interpreter.
        assert_eq!(python_wheel_rank("cp311", "abi3", (3, 12)), Some(0));
        assert_eq!(python_wheel_rank("cp311", "abi3", (3, 11)), Some(0));
        assert_eq!(python_wheel_rank("cp311", "abi3", (3, 10)), None);
        // A version-specific wheel requires an exact interpreter match.
        assert_eq!(python_wheel_rank("cp310", "cp310", (3, 10)), Some(1));
        assert_eq!(python_wheel_rank("cp310", "cp310", (3, 11)), None);
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
