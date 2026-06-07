mod docker;
mod hf;
mod source;

use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use source::{EnvironmentSourceRef, GymSourceRef, HfSourceRef};

pub const DEFAULT_BASE_IMAGE: &str = "python:3.11-slim";
pub const DEFAULT_PACKAGE_NAME: &str = "rlmesh";
pub const BOOTSTRAP_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct SandboxOptions {
    pub base_image: Option<String>,
    pub package_spec: Option<String>,
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
            package_spec: None,
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

    pub fn resolved_package_spec(&self) -> String {
        self.package_spec
            .clone()
            .or_else(|| std::env::var("RLMESH_SANDBOX_PACKAGE_SPEC").ok())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(default_package_spec)
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
    pub package_spec: String,
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
        let package_spec = validate_nonempty("package_spec", options.resolved_package_spec())?;
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
            package_spec: &package_spec,
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
            package_spec,
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
    package_spec: &'a str,
    packages: &'a [String],
    imports: &'a [String],
    kwargs: &'a BTreeMap<String, serde_json::Value>,
    num_envs: usize,
    vectorization_mode: VectorizationMode,
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

pub fn default_package_spec() -> String {
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
}
