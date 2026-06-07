use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::source::HfSourceRef;

pub(crate) fn resolve_revision(source: &HfSourceRef) -> Result<String> {
    if let Some(revision) = &source.revision
        && looks_like_full_git_sha(revision)
    {
        return Ok(revision.to_string());
    }

    let url = hf_git_url(&source.repo);
    let target = source.revision.as_deref().unwrap_or("HEAD");
    let output = Command::new("git")
        .args(["ls-remote", &url, target])
        .output()
        .with_context(|| format!("failed to query {url}; is git installed?"))?;
    if !output.status.success() {
        bail!(
            "git ls-remote failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(line) = stdout.lines().next() else {
        bail!("unable to resolve revision '{target}' for {url}");
    };
    let Some((sha, _)) = line.split_once('\t') else {
        bail!("unexpected git ls-remote output for {url}");
    };
    Ok(sha.to_string())
}

pub(crate) fn materialize_source(
    source: &HfSourceRef,
    resolved_revision: &str,
    destination: &Path,
) -> Result<()> {
    if destination.exists() {
        fs::remove_dir_all(destination).with_context(|| {
            format!(
                "failed to clear existing HF source directory {}",
                destination.display()
            )
        })?;
    }

    let url = hf_git_url(&source.repo);
    let clone = Command::new("git")
        .args(["clone", "--quiet", &url])
        .arg(destination)
        .env("GIT_LFS_SKIP_SMUDGE", "1")
        .output()
        .with_context(|| format!("failed to clone {url}; is git installed?"))?;
    if !clone.status.success() {
        bail!(
            "git clone failed for {url}: {}",
            String::from_utf8_lossy(&clone.stderr).trim()
        );
    }

    let checkout = Command::new("git")
        .args(["checkout", "--quiet", resolved_revision])
        .current_dir(destination)
        .output()
        .with_context(|| format!("failed to checkout {resolved_revision}"))?;
    if !checkout.status.success() {
        bail!(
            "git checkout failed for {resolved_revision}: {}",
            String::from_utf8_lossy(&checkout.stderr).trim()
        );
    }

    let env_py = destination.join("env.py");
    if !env_py.exists() {
        bail!(
            "hf://{} must contain env.py at the repository root",
            source.repo
        );
    }

    let git_dir = destination.join(".git");
    if git_dir.exists() {
        fs::remove_dir_all(&git_dir).with_context(|| {
            format!(
                "failed to remove git metadata from HF source directory {}",
                git_dir.display()
            )
        })?;
    }

    Ok(())
}

fn hf_git_url(repo: &str) -> String {
    format!("https://huggingface.co/{repo}")
}

fn looks_like_full_git_sha(value: &str) -> bool {
    value.len() == 40 && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::looks_like_full_git_sha;

    #[test]
    fn detects_full_git_shas() {
        assert!(looks_like_full_git_sha(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(!looks_like_full_git_sha("abcdef0"));
        assert!(!looks_like_full_git_sha("main"));
    }
}
