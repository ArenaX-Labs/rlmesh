use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{DEFAULT_PACKAGE_NAME, hex};

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

pub(crate) fn resolve_rlmesh_package(
    value: String,
    base_image: &str,
) -> Result<ResolvedRlmeshPackage> {
    if value == "local" {
        let wheel = resolve_local_rlmesh_wheel(base_image)?;
        return resolved_wheel_package(&wheel);
    }

    let path = Path::new(&value);
    if !value.contains("://") && path.extension().and_then(|value| value.to_str()) == Some("whl") {
        return resolved_wheel_package(path);
    }

    Ok(ResolvedRlmeshPackage::Pip { spec: value })
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
/// reference from delimited tokens rather than substring matching.
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
    Ok(hex(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_BASE_IMAGE;

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
        // A coincidental "3.10" inside an unrelated version must not be read as
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
}
