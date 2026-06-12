mod docker;
mod hf;
mod source;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("sync").trim() {
            "sync" => Ok(Self::Sync),
            "async" => Ok(Self::Async),
            other => anyhow::bail!("vectorization_mode must be 'sync' or 'async', got '{other}'"),
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

#[derive(Debug, Clone)]
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
    fn resolve(source: EnvironmentSourceRef, options: SandboxOptions) -> Result<Self> {
        let base_image = validate_nonempty("base_image", options.resolved_base_image())?;
        let rlmesh_package = options.resolved_rlmesh_package(&base_image)?;
        let packages = validate_specs("packages", options.packages)?;
        let imports = validate_specs("imports", options.imports)?;
        validate_source_trust(
            &source,
            options.trust_remote_code,
            options.allow_unpinned_hf,
        )?;
        let kwargs = options.kwargs;
        let num_envs = validate_num_envs(options.num_envs)?;
        let vectorization_mode = options.vectorization_mode;
        let resolved_source = resolve_source(&source)?;
        let build_hash = build_hash(&BuildHashInput {
            schema_version: BOOTSTRAP_SCHEMA_VERSION,
            source: &resolved_source,
            base_image: &base_image,
            rlmesh_package: &rlmesh_package,
            packages: &packages,
            imports: &imports,
            kwargs: &kwargs,
            num_envs,
            vectorization_mode,
        })?;

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
    kwargs: &'a BTreeMap<String, serde_json::Value>,
    num_envs: usize,
    vectorization_mode: VectorizationMode,
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

pub fn start_env(source: EnvironmentSourceRef, options: SandboxOptions) -> Result<RunResult> {
    let spec = EffectiveSandboxSpec::resolve(source, options)?;
    let docker = docker::DockerBackend;
    let artifact = docker.ensure_image(&spec)?;
    let started = docker.run_container(&spec, &artifact)?;

    Ok(RunResult {
        requested_source: spec.requested_display(),
        resolved_source: spec.resolved_display(),
        address: started.address,
        container_id: started.container_id,
    })
}

pub fn stop_container(container_id: &str) -> Result<()> {
    docker::DockerBackend.stop_container(container_id)
}

/// Best-effort reap of rlmesh-owned sandbox containers left behind by hard
/// process kills (SIGKILL/OOM). Containers are identified by the
/// `rlmesh.sandbox` label that [`start_env`] stamps on every container it
/// starts. Returns the ids that were removed.
pub fn reap_orphaned_containers() -> Result<Vec<String>> {
    docker::DockerBackend.reap_orphaned_containers()
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

fn select_local_rlmesh_wheel(dist_dir: &Path, base_image: &str) -> Result<PathBuf> {
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
            wheel_rank(&filename, base_image).map(|rank| (rank, filename, path))
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

fn wheel_rank(filename: &str, base_image: &str) -> Option<(usize, usize)> {
    let (python_tag, abi_tag, platform_tag) = wheel_tags(filename)?;
    let python_rank = python_wheel_rank(python_tag, abi_tag, base_image)?;
    let platform_rank = platform_wheel_rank(platform_tag, base_image)?;
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

fn python_wheel_rank(python_tag: &str, abi_tag: &str, base_image: &str) -> Option<usize> {
    if base_image.contains("3.10") {
        return (python_tag == "cp310" && abi_tag == "cp310").then_some(0);
    }
    (python_tag == "cp311" && abi_tag == "abi3").then_some(0)
}

fn platform_wheel_rank(platform_tag: &str, base_image: &str) -> Option<usize> {
    if !platform_matches_host_arch(platform_tag) {
        return None;
    }
    if base_image.contains("alpine") || base_image.contains("musl") {
        return platform_tag.starts_with("musllinux").then_some(0);
    }
    if platform_tag.starts_with("manylinux") {
        return Some(0);
    }
    if platform_tag.starts_with("linux_") {
        return Some(1);
    }
    None
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
