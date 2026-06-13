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

    // Resolve the unpinned HEAD when no revision is requested.
    let Some(revision) = source.revision.as_deref() else {
        return resolve_head(&url);
    };

    validate_revision(revision)?;

    // Query the exact refs we accept, in git's checkout precedence order
    // (annotated/lightweight tag before branch), passing fully-qualified ref
    // patterns so a single user revision cannot expand to multiple refs and
    // cannot be reinterpreted as a branch when a same-named tag exists. The
    // patterns are anchored exact refs, not globs.
    for ref_pattern in [
        format!("refs/tags/{revision}"),
        format!("refs/heads/{revision}"),
    ] {
        if let Some(sha) = ls_remote_exact(&url, &ref_pattern)? {
            return Ok(sha);
        }
    }

    bail!("unable to resolve revision '{revision}' to a tag or branch for {url}");
}

/// Reject revisions that git could misinterpret. The exact-ref query already
/// prevents glob expansion, but we additionally reject option-looking and
/// glob-bearing revisions defensively so a hostile ref name can never be
/// reparsed as a `git` flag (e.g. `--upload-pack=...`).
fn validate_revision(revision: &str) -> Result<()> {
    if revision.starts_with('-') {
        bail!("revision must not start with '-': '{revision}'");
    }
    if revision.contains(['*', '?', '[', ']', '\\', '^', '~', ':', ' ']) {
        bail!(
            "revision contains characters that are not allowed in a tag or branch name: '{revision}'"
        );
    }
    Ok(())
}

/// Run `git ls-remote --` for an exact ref pattern and return the matching SHA,
/// rejecting ambiguous (multi-ref) results. The `--` terminates option parsing
/// so neither the URL nor the pattern can be treated as a flag.
fn ls_remote_exact(url: &str, ref_pattern: &str) -> Result<Option<String>> {
    let output = Command::new("git")
        .args(["ls-remote", "--", url, ref_pattern])
        .output()
        .with_context(|| format!("failed to query {url}; is git installed?"))?;
    if !output.status.success() {
        bail!(
            "git ls-remote failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    parse_ls_remote_unique(&String::from_utf8_lossy(&output.stdout), ref_pattern)
}

fn resolve_head(url: &str) -> Result<String> {
    let output = Command::new("git")
        .args(["ls-remote", "--", url, "HEAD"])
        .output()
        .with_context(|| format!("failed to query {url}; is git installed?"))?;
    if !output.status.success() {
        bail!(
            "git ls-remote failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    parse_ls_remote_unique(&String::from_utf8_lossy(&output.stdout), "HEAD")?
        .ok_or_else(|| anyhow::anyhow!("unable to resolve HEAD for {url}"))
}

/// Parse `git ls-remote` output, requiring at most one matching ref. Returns
/// the SHA when exactly one ref matched, `None` when none matched, and errors
/// when the pattern was ambiguous (more than one ref).
fn parse_ls_remote_unique(stdout: &str, ref_pattern: &str) -> Result<Option<String>> {
    let mut shas = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((sha, _ref_name)) = line.split_once('\t') else {
            bail!("unexpected git ls-remote output: {line:?}");
        };
        shas.push(sha.to_string());
    }
    shas.dedup();
    match shas.len() {
        0 => Ok(None),
        1 => Ok(Some(shas.remove(0))),
        _ => bail!("revision '{ref_pattern}' is ambiguous: it matched multiple refs"),
    }
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
    use super::{looks_like_full_git_sha, parse_ls_remote_unique, validate_revision};

    #[test]
    fn detects_full_git_shas() {
        assert!(looks_like_full_git_sha(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(!looks_like_full_git_sha("abcdef0"));
        assert!(!looks_like_full_git_sha("main"));
    }

    #[test]
    fn rejects_option_injection_and_globs() {
        assert!(validate_revision("--upload-pack=touch /tmp/x").is_err());
        assert!(validate_revision("-x").is_err());
        assert!(validate_revision("v1.*").is_err());
        assert!(validate_revision("release/[0-9]").is_err());
        assert!(validate_revision("a^b").is_err());
        // Ordinary tags/branches are accepted.
        assert!(validate_revision("v1.0").is_ok());
        assert!(validate_revision("main").is_ok());
        assert!(validate_revision("release/1.0").is_ok());
    }

    #[test]
    fn parse_ls_remote_returns_unique_sha() {
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let out = format!("{sha}\trefs/tags/v1.0\n");
        assert_eq!(
            parse_ls_remote_unique(&out, "refs/tags/v1.0").unwrap(),
            Some(sha.to_string())
        );
    }

    #[test]
    fn parse_ls_remote_returns_none_for_no_match() {
        assert_eq!(parse_ls_remote_unique("", "refs/tags/v1.0").unwrap(), None);
        assert_eq!(
            parse_ls_remote_unique("\n  \n", "refs/heads/x").unwrap(),
            None
        );
    }

    #[test]
    fn parse_ls_remote_rejects_ambiguous_match() {
        // Distinct SHAs under one (would-be-glob) pattern must be rejected,
        // not silently resolved to the first line.
        let out = "1111111111111111111111111111111111111111\trefs/tags/v1.0\n\
                   2222222222222222222222222222222222222222\trefs/tags/v1.0-rc1\n";
        let err = parse_ls_remote_unique(out, "refs/tags/v1.*").unwrap_err();
        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn parse_ls_remote_dedupes_identical_shas() {
        // A peeled annotated tag lists the same object twice (the `^{}` line);
        // identical SHAs are not ambiguous.
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let out = format!("{sha}\trefs/tags/v1.0\n{sha}\trefs/tags/v1.0^{{}}\n");
        assert_eq!(
            parse_ls_remote_unique(&out, "refs/tags/v1.0").unwrap(),
            Some(sha.to_string())
        );
    }
}
