use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let repo_root = root.parent().and_then(Path::parent).unwrap_or(&root);
    println!("cargo:rerun-if-env-changed=RLMESH_RELEASE_BUILD");
    let checkout = repo_root.join("rlmesh.toml").exists();
    // `.git` is a file in a worktree, so `.git/HEAD` does not exist there: ask git where HEAD lives and watch only a real path.
    if checkout && let Some(head) = git_output(repo_root, &["rev-parse", "--git-path", "HEAD"]) {
        let head = repo_root.join(head);
        if head.exists() {
            println!("cargo:rerun-if-changed={}", head.display());
        }
        // HEAD on a branch is a symbolic ref that commits never rewrite; the
        // branch ref (or packed-refs once packed) is what moves.
        let mut refs = vec!["packed-refs".to_string()];
        refs.extend(git_output(repo_root, &["symbolic-ref", "-q", "HEAD"]));
        for name in refs {
            if let Some(path) = git_output(repo_root, &["rev-parse", "--git-path", &name]) {
                let path = repo_root.join(path);
                if path.exists() {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }

    let (source, git) = if release_build_enabled() {
        ("release", None)
    } else if let Some(head) = checkout
        .then(|| git_output(repo_root, &["rev-parse", "--short=12", "HEAD"]))
        .flatten()
    {
        ("git", Some(head))
    } else {
        ("package", None)
    };
    println!("cargo:rustc-env=RLMESH_BUILD_SOURCE={source}");
    println!(
        "cargo:rustc-env=RLMESH_BUILD_GIT={}",
        git.unwrap_or_default()
    );

    let spec = root.join("proto");
    println!("cargo:rerun-if-changed={}", spec.display());
    tonic_prost_build::configure()
        .build_client(false)
        .build_server(false)
        .enum_attribute(
            "rlmesh.model.v1.JoinRequest.kind",
            "#[allow(clippy::large_enum_variant)]",
        )
        .bytes(".rlmesh.spaces.v1.SpaceValue.leaves")
        .compile_protos(
            &[
                spec.join("rlmesh/core/v1/env_contract.proto"),
                spec.join("rlmesh/core/v1/handshake.proto"),
                spec.join("rlmesh/env/v1/interaction.proto"),
                spec.join("rlmesh/spaces/v1/meta.proto"),
                spec.join("rlmesh/spaces/v1/spaces.proto"),
                spec.join("rlmesh/spaces/v1/types.proto"),
                spec.join("rlmesh/spaces/v1/value.proto"),
                spec.join("rlmesh/model/v1/interaction.proto"),
            ],
            &[spec],
        )?;

    Ok(())
}

fn release_build_enabled() -> bool {
    std::env::var("RLMESH_RELEASE_BUILD").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
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
