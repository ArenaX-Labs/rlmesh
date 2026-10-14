//! Registry access through `docker buildx imagetools`, and reading what a
//! tag holds: a manifest, a confirmed absence, or an unreadable state.

use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::source::Reference;
use super::tags::Existing;

/// The registry operations publish needs, as `docker buildx imagetools`
/// provides them; a test hands in canned responses instead.
pub(crate) trait Imagetools {
    /// The JSON `imagetools inspect REF --format '{{json .Manifest}}'` prints:
    /// an index (with `manifests`) or a bare manifest descriptor.
    fn manifest(&self, reference: &str) -> Result<String>;
    /// The OCI image config of a single-image reference
    /// (`--format '{{json .Image}}'`).
    fn config(&self, reference: &str) -> Result<String>;
    /// The manifest exactly as the registry stores it (`inspect REF --raw`),
    /// every field included.
    fn raw(&self, reference: &str) -> Result<String>;
    /// Run `imagetools create ARGS`, returning its stdout (the index JSON
    /// under `--dry-run`).
    fn create(&self, args: &[String]) -> Result<String>;
}

/// The real registry, through the docker CLI.
pub(crate) struct DockerImagetools;

impl DockerImagetools {
    fn run(args: &[&str]) -> Result<String> {
        let output = Command::new("docker")
            .args(["buildx", "imagetools"])
            .args(args)
            .stdin(Stdio::null())
            .output()
            .context("running docker buildx imagetools (is docker with buildx installed?)")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "docker buildx imagetools {} failed: {}",
                args.first().copied().unwrap_or_default(),
                stderr.trim()
            );
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

impl Imagetools for DockerImagetools {
    fn manifest(&self, reference: &str) -> Result<String> {
        Self::run(&["inspect", reference, "--format", "{{json .Manifest}}"])
    }

    fn config(&self, reference: &str) -> Result<String> {
        Self::run(&["inspect", reference, "--format", "{{json .Image}}"])
    }

    fn raw(&self, reference: &str) -> Result<String> {
        Self::run(&["inspect", reference, "--raw"])
    }

    fn create(&self, args: &[String]) -> Result<String> {
        let args: Vec<&str> = std::iter::once("create")
            .chain(args.iter().map(String::as_str))
            .collect();
        Self::run(&args)
    }
}

/// Whether an inspect error is the registry saying `tag` does not exist, as
/// opposed to failing to say. Only the shapes a missing manifest takes count:
/// containerd's `REF: not found` (what a manifest 404 becomes), the
/// distribution `MANIFEST_UNKNOWN`/`NAME_UNKNOWN` codes, and a 404 on the
/// manifest URL itself. A 404 from a token endpoint, a credential helper
/// that is "not found", a 401, a 5xx, or a transport error is not absence.
pub(crate) fn absent(tag: &str, error: &str) -> bool {
    let error = error.trim();
    if not_found_line(tag, error) {
        return true;
    }
    // Past the exact form, a message naming credentials or a token is an
    // access failure, whatever status it reports.
    let lower = error.to_ascii_lowercase();
    if [
        "credential",
        "authoriz",
        "token",
        "denied",
        "insufficient_scope",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return false;
    }
    [
        "MANIFEST_UNKNOWN",
        "NAME_UNKNOWN",
        "manifest unknown",
        "name unknown",
    ]
    .iter()
    .any(|code| error.contains(code))
        || error.lines().any(|line| {
            line.contains("/v2/")
                && line.contains("/manifests/")
                && line.ends_with(": 404 Not Found")
        })
}

/// Whether the error ends with containerd's `REF: not found` for `tag`, the
/// two compared as docker resolves them (`ubuntu:v1` reads
/// `docker.io/library/ubuntu:v1`).
fn not_found_line(tag: &str, error: &str) -> bool {
    let Some(reference) = error.strip_suffix(": not found") else {
        return false;
    };
    let reference = reference
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or_default();
    canonical_reference(reference) == canonical_reference(tag)
}

/// `reference` the way docker resolves it: a first path component that is
/// not a host (no `.` or `:`, and not `localhost`) means Docker Hub, as do
/// `index.docker.io` and `registry-1.docker.io`; a one-component Docker Hub
/// name is under `library/`; no tag and no digest means `latest`.
pub(crate) fn canonical_reference(reference: &str) -> String {
    let Ok(Reference {
        repository,
        tag,
        digest,
    }) = Reference::parse(reference)
    else {
        return reference.to_owned();
    };
    let (host, path) = match repository.split_once('/') {
        Some((first, rest)) if first.contains(['.', ':']) || first == "localhost" => (first, rest),
        _ => ("docker.io", repository.as_str()),
    };
    let host = match host {
        "index.docker.io" | "registry-1.docker.io" => "docker.io",
        host => host,
    };
    let mut canonical = if host == "docker.io" && !path.contains('/') {
        format!("{host}/library/{path}")
    } else {
        format!("{host}/{path}")
    };
    if tag.is_none() && digest.is_none() {
        canonical.push_str(":latest");
    }
    if let Some(tag) = tag {
        canonical.push_str(&format!(":{tag}"));
    }
    if let Some(digest) = digest {
        canonical.push_str(&format!("@{digest}"));
    }
    canonical
}

/// What `tag` holds now. An error that is not a confirmed absence leaves it
/// unreadable, never new: a version tag is not pushed over what could not be
/// checked. The raw manifest is read by the digest the tag resolved to, not
/// by the tag again, so a tag that moves between the two reads cannot pair
/// one manifest's digest with another's contents. Only the tag's own read can
/// say it is absent: once it resolved, failing to read that digest is not.
pub(super) fn existing_manifest(tools: &impl Imagetools, tag: &str) -> Existing {
    let resolve = || -> Result<String> {
        let view: Value = serde_json::from_str(&tools.manifest(tag)?)
            .context("docker printed a manifest that is not JSON")?;
        Ok(view
            .get("digest")
            .and_then(Value::as_str)
            .context("docker printed a manifest without a digest")?
            .to_owned())
    };
    let digest = match resolve() {
        Ok(digest) => digest,
        Err(error) => {
            let error = format!("{error:#}");
            return if absent(tag, &error) {
                Existing::Absent
            } else {
                Existing::Unreadable(error)
            };
        }
    };
    let read_raw = || -> Result<Value> {
        let repository = Reference::parse(tag)?.repository;
        serde_json::from_str(&tools.raw(&format!("{repository}@{digest}"))?)
            .context("its raw manifest is not JSON")
    };
    match read_raw() {
        Ok(raw) => Existing::Found { digest, raw },
        Err(error) => Existing::Unreadable(format!("{error:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The messages `docker buildx imagetools inspect` (0.37) prints, each
    /// as the wrapper reports it.
    fn inspect_error(message: &str) -> String {
        format!("docker buildx imagetools inspect failed: ERROR: {message}")
    }

    /// A tag resolved to `sha256:old`, which then moves: reading by tag
    /// would give the new index, while the old one is still readable by
    /// digest; `raw` fails with `raw_error` when set.
    struct MovingTag {
        raw_error: Option<&'static str>,
        raws: std::cell::RefCell<Vec<String>>,
    }

    impl Imagetools for MovingTag {
        fn manifest(&self, _reference: &str) -> Result<String> {
            Ok(r#"{"digest": "sha256:old", "manifests": []}"#.to_owned())
        }
        fn config(&self, reference: &str) -> Result<String> {
            bail!("no config for {reference}")
        }
        fn raw(&self, reference: &str) -> Result<String> {
            self.raws.borrow_mut().push(reference.to_owned());
            if let Some(error) = self.raw_error {
                bail!("{error}");
            }
            Ok(match reference {
                "reg.example/ns/pi0@sha256:old" => r#"{"manifests": ["old"]}"#,
                _ => r#"{"manifests": ["new"]}"#,
            }
            .to_owned())
        }
        fn create(&self, _args: &[String]) -> Result<String> {
            bail!("nothing is created")
        }
    }

    #[test]
    fn what_a_tag_holds_is_read_by_the_digest_it_resolved_to() {
        let tools = MovingTag {
            raw_error: None,
            raws: Default::default(),
        };
        assert_eq!(
            existing_manifest(&tools, "reg.example/ns/pi0:v3"),
            Existing::Found {
                digest: "sha256:old".to_owned(),
                raw: serde_json::json!({"manifests": ["old"]}),
            }
        );
        assert_eq!(*tools.raws.borrow(), ["reg.example/ns/pi0@sha256:old"]);

        // Once the tag resolved, a digest the registry no longer knows is
        // not the tag's absence.
        let tools = MovingTag {
            raw_error: Some("ERROR: manifest unknown: manifest unknown"),
            raws: Default::default(),
        };
        assert!(matches!(
            existing_manifest(&tools, "reg.example/ns/pi0:v3"),
            Existing::Unreadable(_)
        ));
    }

    #[test]
    fn only_a_confirmed_absence_reads_as_a_new_tag() {
        let tag = "localhost:5055/ns/pi0:v1";
        for message in [
            // A manifest 404, and a 404 with a MANIFEST_UNKNOWN body.
            "localhost:5055/ns/pi0:v1: not found",
            "manifest unknown: manifest unknown",
            "unexpected status from HEAD request to http://localhost:5055/v2/ns/pi0/manifests/v1: \
             404 Not Found",
        ] {
            assert!(absent(tag, &inspect_error(message)), "{message}");
        }
        // docker normalizes a Docker Hub reference.
        assert!(absent(
            "ns/pi0:v1",
            &inspect_error("docker.io/ns/pi0:v1: not found")
        ));
        for message in [
            "unexpected status from HEAD request to http://localhost:5055/v2/ns/pi0/manifests/v1: \
             401 Unauthorized",
            "failed to authorize: failed to fetch anonymous token: unexpected status from GET \
             request to http://localhost:5055/token?scope=repository%3Ans%2Fpi0%3Apull: 404 Not \
             Found",
            "error getting credentials - err: exec: \"docker-credential-gone\": executable file \
             not found in $PATH, out: ``",
            "failed to do request: Head \"http://localhost:5999/v2/ns/pi0/manifests/v1\": dial \
             tcp 127.0.0.1:5999: connect: connection refused",
            "unexpected status from HEAD request to http://localhost:5055/v2/ns/pi0/manifests/v1: \
             500 Internal Server Error",
            "pull access denied, repository does not exist or may require authorization: server \
             message: insufficient_scope: authorization failed",
            // Another reference's absence is not this one's.
            "localhost:5055/ns/other:v1: not found",
        ] {
            assert!(!absent(tag, &inspect_error(message)), "{message}");
        }
        // Not running docker at all is not absence either.
        assert!(!absent(
            tag,
            "running docker buildx imagetools (is docker with buildx installed?): No such file \
             or directory (os error 2)"
        ));
    }

    #[test]
    fn references_compare_as_docker_resolves_them() {
        for (reference, canonical) in [
            ("ubuntu", "docker.io/library/ubuntu:latest"),
            ("ubuntu:24.04", "docker.io/library/ubuntu:24.04"),
            ("docker.io/ubuntu:24.04", "docker.io/library/ubuntu:24.04"),
            (
                "index.docker.io/library/ubuntu:24.04",
                "docker.io/library/ubuntu:24.04",
            ),
            (
                "registry-1.docker.io/ubuntu:24.04",
                "docker.io/library/ubuntu:24.04",
            ),
            ("ns/pi0:v1", "docker.io/ns/pi0:v1"),
            ("index.docker.io/ns/pi0:v1", "docker.io/ns/pi0:v1"),
            ("localhost/pi0:v1", "localhost/pi0:v1"),
            ("localhost:5055/ns/pi0:v1", "localhost:5055/ns/pi0:v1"),
            ("reg.example/pi0:v1", "reg.example/pi0:v1"),
            ("reg:5000/pi0", "reg:5000/pi0:latest"),
            (
                "reg.example/ns/pi0@sha256:ab",
                "reg.example/ns/pi0@sha256:ab",
            ),
        ] {
            assert_eq!(canonical_reference(reference), canonical, "{reference}");
        }
        // buildx 0.37 names every Docker Hub spelling by its canonical form.
        for tag in [
            "ubuntu:rl-none",
            "docker.io/ubuntu:rl-none",
            "index.docker.io/library/ubuntu:rl-none",
            "docker.io/library/ubuntu:rl-none",
        ] {
            assert!(
                absent(
                    tag,
                    &inspect_error("docker.io/library/ubuntu:rl-none: not found")
                ),
                "{tag}"
            );
        }
        assert!(absent(
            "ubuntu",
            &inspect_error("docker.io/library/ubuntu:latest: not found")
        ));
        for (tag, reported) in [
            // Another repository, tag, or registry is not this tag's absence.
            ("ns/pi0:v1", "docker.io/library/pi0:v1"),
            ("pi0:v1", "docker.io/ns/pi0:v1"),
            ("ubuntu:a", "docker.io/library/ubuntu:b"),
            ("other.example/ns/pi0:v1", "docker.io/ns/pi0:v1"),
            ("localhost/pi0:v1", "docker.io/library/pi0:v1"),
            ("localhost:5055/ns/pi0:v1", "localhost:5056/ns/pi0:v1"),
        ] {
            assert!(
                !absent(tag, &inspect_error(&format!("{reported}: not found"))),
                "{tag} vs {reported}"
            );
        }
    }
}
