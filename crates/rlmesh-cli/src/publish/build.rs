//! Building `rlmesh.toml`'s variants with `docker buildx build`, each pushed
//! under its own tag in the version's repository with its rendered label.
//! (Pushing by digest alone would be tidier, but buildx's default `docker`
//! driver cannot.)

use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::image_check::PACKAGE_LABEL;
use crate::manifest::VariantBuild;

/// Builds one variant and pushes it by digest; a test hands in a fake.
pub(crate) trait Builder {
    /// Run the build `args` (from [`build_args`]) and return the pushed
    /// image's digest.
    fn build(&self, args: &[String]) -> Result<String>;
}

/// The `docker buildx build` arguments for `variant`, pushed as `name` with
/// OCI media types (so the index can carry its annotation), writing its
/// metadata to `metadata`.
pub(crate) fn build_args(variant: &VariantBuild, name: &str, metadata: &Path) -> Vec<String> {
    let label = serde_json::to_string(&Value::Object(variant.label.clone()))
        .expect("a JSON map serializes");
    let mut args: Vec<String> = ["build", "--platform", "linux/amd64", "--file"]
        .map(str::to_owned)
        .into();
    args.push(variant.dockerfile.display().to_string());
    for (key, value) in &variant.build_args {
        args.push("--build-arg".to_owned());
        args.push(format!("{key}={value}"));
    }
    args.push("--label".to_owned());
    args.push(format!("{PACKAGE_LABEL}={label}"));
    args.push("--output".to_owned());
    args.push(format!(
        "type=image,name={name},push=true,oci-mediatypes=true"
    ));
    args.push("--metadata-file".to_owned());
    args.push(metadata.display().to_string());
    args.push(variant.context.display().to_string());
    args
}

/// The real builder: `docker buildx build`, its progress on stderr.
pub(crate) struct DockerBuilder;

impl Builder for DockerBuilder {
    fn build(&self, args: &[String]) -> Result<String> {
        let metadata = args
            .iter()
            .position(|arg| arg == "--metadata-file")
            .and_then(|at| args.get(at + 1))
            .context("build arguments without --metadata-file")?;
        let status = Command::new("docker")
            .arg("buildx")
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .status()
            .context("running docker buildx build (is docker with buildx installed?)")?;
        if !status.success() {
            bail!("docker buildx build failed ({status})");
        }
        let raw = std::fs::read_to_string(metadata)
            .with_context(|| format!("reading the build metadata {metadata}"))?;
        let _ = std::fs::remove_file(metadata);
        pushed_digest(&raw)
    }
}

/// The digest in a `--metadata-file`: what was pushed.
pub(crate) fn pushed_digest(metadata: &str) -> Result<String> {
    let metadata: Value =
        serde_json::from_str(metadata).context("the build metadata is not JSON")?;
    match metadata["containerimage.digest"].as_str() {
        Some(digest) if digest.starts_with("sha256:") => Ok(digest.to_owned()),
        _ => bail!("the build metadata names no pushed digest (containerimage.digest)"),
    }
}
