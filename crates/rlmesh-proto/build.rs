use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_WORKFLOW_EDITION_BASE: &str = "2026.06";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let repo_root = root.parent().and_then(Path::parent).unwrap_or(&root);
    println!("cargo:rerun-if-env-changed=RLMESH_RELEASE_BUILD");
    println!("cargo:rerun-if-env-changed=RLMESH_WORKFLOW_EDITION_BASE");
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("rlmesh.toml").display()
    );
    let git_dir = repo_root.join(".git");
    if git_dir.exists() {
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
        emit_git_rerun_paths(repo_root);
    }

    let base = workflow_edition_base(repo_root);
    let version = std::env::var("CARGO_PKG_VERSION")?;
    let cohort = workflow_cohort(repo_root, &version);
    let current_edition = if let Some(dev) = cohort.dev_token {
        format!("{base}-dev.{dev}")
    } else if version.contains('-') {
        format!("{base}-{version}")
    } else {
        base.clone()
    };

    println!("cargo:rustc-env=RLMESH_WORKFLOW_EDITION_BASE={base}");
    println!("cargo:rustc-env=RLMESH_CURRENT_WORKFLOW_EDITION={current_edition}");
    println!("cargo:rustc-env=RLMESH_BUILD_COHORT={}", cohort.name);
    println!("cargo:rustc-env=RLMESH_BUILD_SOURCE={}", cohort.source);

    let spec = root.join("proto");
    tonic_prost_build::configure()
        .enum_attribute(
            "rlmesh.model.v1.JoinRequest.kind",
            "#[allow(clippy::large_enum_variant)]",
        )
        // The SpaceValue leaves carry tensor bytes; generate each leaf as a
        // refcounted `bytes::Bytes` so the codec can share the tensor's storage
        // (zero-copy) instead of copying element bytes into the message.
        .bytes(".rlmesh.spaces.v1.SpaceValue.leaves")
        .compile_protos(
            &[
                // Core
                spec.join("rlmesh/core/v1/contract.proto"),
                spec.join("rlmesh/core/v1/handshake.proto"),
                // Env
                spec.join("rlmesh/env/v1/interaction.proto"),
                spec.join("rlmesh/env/v1/service.proto"),
                // Spaces
                spec.join("rlmesh/spaces/v1/meta.proto"),
                spec.join("rlmesh/spaces/v1/spaces.proto"),
                spec.join("rlmesh/spaces/v1/types.proto"),
                spec.join("rlmesh/spaces/v1/value.proto"),
                // Model
                spec.join("rlmesh/model/v1/interaction.proto"),
                spec.join("rlmesh/model/v1/service.proto"),
            ],
            &[spec],
        )?;

    Ok(())
}

struct WorkflowCohort {
    name: String,
    source: String,
    dev_token: Option<String>,
}

fn workflow_edition_base(repo_root: &Path) -> String {
    if let Ok(base) = std::env::var("RLMESH_WORKFLOW_EDITION_BASE")
        && !base.trim().is_empty()
    {
        return base;
    }

    let manifest = repo_root.join("rlmesh.toml");
    let Ok(text) = std::fs::read_to_string(manifest) else {
        return DEFAULT_WORKFLOW_EDITION_BASE.to_string();
    };
    manifest_string_value(&text, "base_edition")
        .unwrap_or_else(|| DEFAULT_WORKFLOW_EDITION_BASE.to_string())
}

fn manifest_string_value(text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key} = ");
    text.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(&prefix))
        .and_then(|raw| raw.trim().strip_prefix('"')?.split_once('"'))
        .map(|(value, _)| value.to_string())
}

fn workflow_cohort(repo_root: &Path, version: &str) -> WorkflowCohort {
    if release_build_enabled() {
        return WorkflowCohort {
            name: release_cohort_name(version),
            source: "release".to_string(),
            dev_token: None,
        };
    }

    let Some(head) = git_output(repo_root, &["rev-parse", "--short=12", "HEAD"]) else {
        return WorkflowCohort {
            name: release_cohort_name(version),
            source: "package".to_string(),
            dev_token: None,
        };
    };

    let dirty = git_output(repo_root, &["status", "--porcelain=v1"])
        .is_some_and(|status| !status.trim().is_empty());
    let token = if dirty {
        format!("{head}.dirty.{:016x}", dirty_fingerprint(repo_root))
    } else {
        head
    };

    WorkflowCohort {
        name: format!("dev.{token}"),
        source: "git".to_string(),
        dev_token: Some(token),
    }
}

fn release_build_enabled() -> bool {
    std::env::var("RLMESH_RELEASE_BUILD").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn release_cohort_name(version: &str) -> String {
    if version.contains('-') {
        version.to_string()
    } else {
        "stable".to_string()
    }
}

fn git_output(repo_root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn dirty_fingerprint(repo_root: &Path) -> u64 {
    let mut hash = Fnv1a64::new();
    if let Ok(output) = Command::new("git")
        .args(["diff", "--binary", "HEAD", "--"])
        .current_dir(repo_root)
        .output()
    {
        hash.write(&output.stdout);
    }
    if let Ok(output) = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard", "-z"])
        .current_dir(repo_root)
        .output()
    {
        hash.write(&output.stdout);
    }
    hash.finish()
}

fn emit_git_rerun_paths(repo_root: &Path) {
    let Ok(output) = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(repo_root)
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    for raw in output.stdout.split(|byte| *byte == 0) {
        if raw.is_empty() {
            continue;
        }
        if let Ok(rel) = std::str::from_utf8(raw) {
            println!("cargo:rerun-if-changed={}", repo_root.join(rel).display());
        }
    }
}

struct Fnv1a64(u64);

impl Fnv1a64 {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}
