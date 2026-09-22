use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "build_manifest.rs"]
mod manifest;

use manifest::manifest_string_list;

const RETAINED_EDITIONS_FILE: &str = "supported_editions.txt";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let repo_root = root.parent().and_then(Path::parent).unwrap_or(&root);
    println!("cargo:rerun-if-env-changed=RLMESH_RELEASE_BUILD");
    println!("cargo:rerun-if-env-changed=RLMESH_WORKFLOW_EDITION_BASE");
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("rlmesh.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join(RETAINED_EDITIONS_FILE).display()
    );
    // `.git` is a file, not a directory, in a worktree checkout, so `.git/HEAD`
    // does not exist there and watching it would rerun this script on every
    // build. Ask git where HEAD actually lives and watch only a real path, and
    // only when this is the rlmesh checkout (see `workflow_cohort`).
    if repo_root.join("rlmesh.toml").exists()
        && let Some(head) = git_output(repo_root, &["rev-parse", "--git-path", "HEAD"])
    {
        let head = repo_root.join(head);
        if head.exists() {
            println!("cargo:rerun-if-changed={}", head.display());
        }
    }

    let retained = match retained_workflow_editions(&root, repo_root) {
        Ok(retained) => retained,
        Err(message) => {
            println!("cargo::error={message}");
            return Ok(());
        }
    };
    let base = workflow_edition_base(repo_root, &retained[0]);
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
    println!(
        "cargo:rustc-env=RLMESH_SUPPORTED_WORKFLOW_EDITIONS={}",
        supported_workflow_editions(retained, &current_edition).join(",")
    );
    println!("cargo:rustc-env=RLMESH_BUILD_COHORT={}", cohort.name);
    println!("cargo:rustc-env=RLMESH_BUILD_SOURCE={}", cohort.source);

    let spec = root.join("proto");
    // prost-build emits no rerun-if-changed for its inputs; watch the tree.
    println!("cargo:rerun-if-changed={}", spec.display());
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
                spec.join("rlmesh/core/v1/env_contract.proto"),
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

/// The manifest's `base_edition`, else the base of the packaged current edition
/// (a published crate ships no `rlmesh.toml`).
fn workflow_edition_base(repo_root: &Path, packaged_current: &str) -> String {
    if let Ok(base) = std::env::var("RLMESH_WORKFLOW_EDITION_BASE")
        && !base.trim().is_empty()
    {
        return base;
    }

    std::fs::read_to_string(repo_root.join("rlmesh.toml"))
        .ok()
        .and_then(|text| manifest_string_value(&text, "base_edition"))
        .or_else(|| packaged_current.split('-').next().map(String::from))
        .unwrap_or_default()
}

/// The crate's `supported_editions.txt`: the official current edition on the
/// first line, then every other retained edition (sealed ones under their bare
/// `YYYY.MM` names), one per line. Every build reads it; a repo build also
/// requires it to be `rlmesh.toml`'s `[workflow]` list in that order.
fn retained_workflow_editions(root: &Path, repo_root: &Path) -> Result<Vec<String>, String> {
    let retained_path = root.join(RETAINED_EDITIONS_FILE);
    let retained: Vec<String> = std::fs::read_to_string(&retained_path)
        .map_err(|err| format!("cannot read {}: {err}", retained_path.display()))?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(String::from)
        .collect();
    if retained.is_empty() {
        return Err(format!("{} lists no editions", retained_path.display()));
    }
    if let Ok(text) = std::fs::read_to_string(repo_root.join("rlmesh.toml")) {
        let current = manifest_string_value(&text, "current_edition");
        let declared: Vec<String> = current
            .iter()
            .cloned()
            .chain(
                manifest_string_list(&text, "supported_editions")
                    .into_iter()
                    .filter(|edition| Some(edition) != current.as_ref()),
            )
            .collect();
        if declared != retained {
            return Err(format!(
                "crates/rlmesh-proto/{RETAINED_EDITIONS_FILE} lists {retained:?} but \
                 rlmesh.toml [workflow] current_edition then supported_editions is \
                 {declared:?}; run `python scripts/bump_version.py --sync-editions`"
            ));
        }
    }
    Ok(retained)
}

/// Workflow editions this build offers, `current_edition` first, then the
/// retained editions. The official current edition (the file's first line) is
/// dropped from the tail: this build's cohort already leads the list, and a dev
/// or recohorted build must not claim the official release cohort.
fn supported_workflow_editions(retained: Vec<String>, current_edition: &str) -> Vec<String> {
    let mut editions = vec![current_edition.to_string()];
    editions.extend(
        retained
            .into_iter()
            .skip(1)
            .filter(|edition| edition != current_edition),
    );
    editions
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

    // A published crate ships build.rs but not the repo's rlmesh.toml. Without
    // this guard, git discovery walks up from the registry/vendor directory into
    // whatever repo happens to contain the consumer's build tree and stamps a dev
    // cohort from *their* commit, which no released peer can negotiate with.
    if !repo_root.join("rlmesh.toml").exists() {
        return WorkflowCohort {
            name: release_cohort_name(version),
            source: "package".to_string(),
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
