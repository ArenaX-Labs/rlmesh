//! Resolving each source to the one image it holds, with the attestations
//! its index carries along.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::imagetools::Imagetools;
use super::media::{ATTESTATION, DOCKER_MEDIA_PREFIX, REFERENCE_DIGEST, REFERENCE_TYPE};
use super::tags::check_tag;
use crate::image_check::{ImageConfig, PACKAGE_LABEL};
use crate::variant;

/// An image reference split the way docker reads it:
/// `[host[:port]/]path[:tag][@digest]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reference {
    pub repository: String,
    pub tag: Option<String>,
    pub digest: Option<String>,
}

impl Reference {
    pub fn parse(raw: &str) -> Result<Self> {
        let raw = raw.trim();
        let (name, digest) = match raw.split_once('@') {
            Some((name, digest)) => (name, Some(digest.to_owned())),
            None => (raw, None),
        };
        let slash = name.rfind('/');
        let (repository, tag) = match name.rfind(':') {
            Some(colon) if slash.is_none_or(|slash| colon > slash) => {
                (&name[..colon], Some(name[colon + 1..].to_owned()))
            }
            _ => (name, None),
        };
        if repository.is_empty() || repository.ends_with('/') {
            bail!("{raw:?} is not an image reference");
        }
        if let Some(tag) = &tag {
            check_tag(tag)?;
        }
        Ok(Self {
            repository: repository.to_owned(),
            tag,
            digest,
        })
    }
}

/// One source resolved to the single image it holds.
#[derive(Debug, Clone)]
pub(crate) struct Child {
    /// The reference as given on the command line.
    pub source: String,
    /// `repository@digest` of what the source names (its index when it has
    /// one, so attestations come along).
    pub pinned: String,
    /// The digest of the image manifest itself.
    pub image_digest: String,
    /// The image manifest's media type (OCI, or Docker schema2).
    pub media_type: String,
    /// `os/architecture[/variant]`: the index descriptor's platform when the
    /// source is an index (checked to agree with the config), else the
    /// config's.
    pub platform: String,
    pub config: ImageConfig,
    /// The attestation manifests the source's index carries along.
    pub attestations: Vec<Attestation>,
    /// The `dev.rlmesh.package` annotation on the source's own index.
    pub index_annotation: Option<String>,
}

/// An attestation manifest riding in a source's index.
#[derive(Debug, Clone)]
pub(crate) struct Attestation {
    pub digest: String,
    pub media_type: String,
    /// The image it describes (`vnd.docker.reference.digest`).
    pub subject: Option<String>,
}

impl Child {
    /// Whether the fleet can select this child, from its checked platform.
    pub(super) fn selectable(&self) -> bool {
        let mut parts = self.platform.split('/');
        variant::selectable(parts.next().unwrap_or(""), parts.next().unwrap_or(""))
    }

    /// Whether every descriptor this child brings into the version, its image
    /// and each attestation, is a Docker (schema2) manifest.
    fn docker_media_only(&self) -> bool {
        std::iter::once(self.media_type.as_str())
            .chain(self.attestations.iter().map(|a| a.media_type.as_str()))
            .all(|media_type| media_type.starts_with(DOCKER_MEDIA_PREFIX))
    }
}

/// Resolve a source to its one linux image: an index may hold that image plus
/// attestation manifests, nothing else.
pub(super) fn resolve_source(tools: &impl Imagetools, source: &str) -> Result<Child> {
    let reference = Reference::parse(source)?;
    let (manifest, digest) = read_manifest(tools, source)?;
    let pinned = format!("{}@{digest}", reference.repository);
    let selected = select_image(source, &manifest, digest)?;
    let config = read_config(tools, source, &reference, &selected.digest)?;
    let platform = checked_platform(source, selected.platform, &config)?;
    require_linux(source, &config, &platform)?;
    Ok(Child {
        source: source.to_owned(),
        pinned,
        image_digest: selected.digest,
        media_type: selected.media_type,
        platform,
        config,
        attestations: selected.attestations,
        index_annotation: index_annotation(&manifest),
    })
}

/// The image a source holds, picked out of its index when it has one.
struct Selected {
    digest: String,
    media_type: String,
    /// The index descriptor's platform; `None` for a bare manifest.
    platform: Option<String>,
    attestations: Vec<Attestation>,
}

/// The source's manifest (an index or a bare image manifest) and its digest.
fn read_manifest(tools: &impl Imagetools, source: &str) -> Result<(Value, String)> {
    let manifest: Value = serde_json::from_str(&tools.manifest(source)?)
        .with_context(|| format!("parsing the manifest of {source}"))?;
    let digest = manifest
        .get("digest")
        .and_then(Value::as_str)
        .with_context(|| format!("the manifest of {source} carries no digest"))?
        .to_owned();
    Ok((manifest, digest))
}

/// A bare manifest is its own image; an index must hold exactly one linux
/// image, besides the attestations it carries along.
fn select_image(source: &str, manifest: &Value, digest: String) -> Result<Selected> {
    let Some(entries) = manifest.get("manifests").and_then(Value::as_array) else {
        return Ok(Selected {
            digest,
            media_type: media_type(manifest),
            platform: None,
            attestations: Vec::new(),
        });
    };
    let (images, attestations) = split_index(source, entries)?;
    let image = one_image(source, images)?;
    Ok(Selected {
        digest: image.digest,
        media_type: image.media_type,
        platform: Some(image.platform),
        attestations,
    })
}

/// An image entry of a source's index.
struct IndexImage {
    digest: String,
    media_type: String,
    platform: String,
}

/// Sort an index's entries into images and attestations: every entry needs a
/// digest, and every image a linux platform.
fn split_index(source: &str, entries: &[Value]) -> Result<(Vec<IndexImage>, Vec<Attestation>)> {
    let mut images = Vec::new();
    let mut attestations = Vec::new();
    for entry in entries {
        let annotation = |key: &str| {
            entry
                .get("annotations")
                .and_then(|annotations| annotations.get(key))
                .and_then(Value::as_str)
        };
        let digest = entry
            .get("digest")
            .and_then(Value::as_str)
            .with_context(|| format!("an entry in the index of {source} has no digest"))?;
        if annotation(REFERENCE_TYPE) == Some(ATTESTATION) {
            attestations.push(Attestation {
                digest: digest.to_owned(),
                media_type: media_type(entry),
                subject: annotation(REFERENCE_DIGEST).map(str::to_owned),
            });
            continue;
        }
        let platform = entry.get("platform").map(platform_name).unwrap_or_default();
        if !platform.starts_with("linux/") {
            bail!(
                "{source} holds a {} image; the platform runs linux images only (build \
                 with --platform linux/amd64)",
                platform_or_less(&platform)
            );
        }
        images.push(IndexImage {
            digest: digest.to_owned(),
            media_type: media_type(entry),
            platform,
        });
    }
    Ok((images, attestations))
}

/// Each source is one variant, so its index holds exactly one image.
fn one_image(source: &str, mut images: Vec<IndexImage>) -> Result<IndexImage> {
    match images.len() {
        1 => Ok(images.swap_remove(0)),
        0 => bail!("{source} is an index with no image in it"),
        n => bail!(
            "{source} is an index of {n} images ({}); each source must be one image, \
             since each child of the published index is one variant",
            images
                .iter()
                .map(|image| image.platform.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The selected image's config, which must parse.
fn read_config(
    tools: &impl Imagetools,
    source: &str,
    reference: &Reference,
    image_digest: &str,
) -> Result<ImageConfig> {
    ImageConfig::from_oci_config(
        &tools.config(&format!("{}@{image_digest}", reference.repository))?,
    )
    .map_err(anyhow::Error::msg)
    .with_context(|| format!("reading the image config of {source}"))
}

/// The platform schedules a child by its descriptor and the runtime runs the
/// config, so the two must name the same platform: eligibility (and the
/// default row) is decided from it. Returns the descriptor's platform, else
/// the config's.
fn checked_platform(
    source: &str,
    declared: Option<String>,
    config: &ImageConfig,
) -> Result<String> {
    let configured = if config.os.is_empty() {
        String::new()
    } else {
        format!("{}/{}", config.os, config.architecture)
    };
    let Some(declared) = declared else {
        return Ok(configured);
    };
    let os_architecture = declared
        .splitn(3, '/')
        .take(2)
        .collect::<Vec<_>>()
        .join("/");
    if os_architecture != configured {
        bail!(
            "{source}: its index declares the image {declared}, but the image config says \
             {}; rebuild it so they agree",
            if configured.is_empty() {
                "no platform"
            } else {
                configured.as_str()
            }
        );
    }
    Ok(declared)
}

/// The platform runs linux images only.
fn require_linux(source: &str, config: &ImageConfig, platform: &str) -> Result<()> {
    if config.os != "linux" {
        bail!(
            "{source} is a {} image; the platform runs linux images only (build with \
             --platform linux/amd64)",
            platform_or_less(platform)
        );
    }
    Ok(())
}

/// The `dev.rlmesh.package` annotation on the source's own index; a bare
/// manifest has none.
fn index_annotation(manifest: &Value) -> Option<String> {
    manifest
        .get("manifests")
        .and(manifest.get("annotations"))
        .and_then(|annotations| annotations.get(PACKAGE_LABEL))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn media_type(value: &Value) -> String {
    value
        .get("mediaType")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// A platform for a message: `platform-less` when there is none.
fn platform_or_less(platform: &str) -> &str {
    if platform.is_empty() {
        "platform-less"
    } else {
        platform
    }
}

/// Every carried attestation must describe one of the version's children: an
/// attestation whose subject is not among them would dangle in the index.
pub(crate) fn attestation_errors(children: &[Child]) -> Vec<String> {
    let selected: BTreeSet<&str> = children
        .iter()
        .map(|child| child.image_digest.as_str())
        .collect();
    children
        .iter()
        .flat_map(|child| {
            child.attestations.iter().filter_map(|attestation| {
                let subject = attestation.subject.as_deref();
                if subject.is_some_and(|subject| selected.contains(subject)) {
                    return None;
                }
                Some(format!(
                    "{}: attestation manifest {} {}, not an image of this version; push the \
                     source again, or name its image by digest ({}@{}) to leave the \
                     attestations behind",
                    child.source,
                    attestation.digest,
                    match subject {
                        Some(subject) => format!("describes {subject}"),
                        None => format!("names no image ({REFERENCE_DIGEST})"),
                    },
                    Reference::parse(&child.source)
                        .map(|reference| reference.repository)
                        .unwrap_or_default(),
                    child.image_digest
                ))
            })
        })
        .collect()
}

/// `docker buildx imagetools create` writes a Docker manifest list, which
/// carries no annotations, when every descriptor it assembles (images and
/// attestations alike) is a Docker schema2 manifest; the `--index-package`
/// annotation would then be dropped.
pub(crate) fn docker_list_error(children: &[Child]) -> Option<String> {
    (!children.is_empty() && children.iter().all(Child::docker_media_only)).then(|| {
        format!(
            "--index-package needs an OCI image index, but every manifest the sources carry \
             (images and attestations) is a Docker schema2 manifest ({}), so docker would \
             publish a Docker manifest list and drop the annotation; rebuild the sources with \
             OCI media types (docker buildx build --push, or --provenance=false --output \
             type=image,oci-mediatypes=true,push=true)",
            children
                .iter()
                .map(|child| child.source.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

fn platform_name(platform: &Value) -> String {
    let field = |name: &str| platform.get(name).and_then(Value::as_str).unwrap_or("");
    let mut name = format!("{}/{}", field("os"), field("architecture"));
    if !field("variant").is_empty() {
        name = format!("{name}/{}", field("variant"));
    }
    if name == "/" { String::new() } else { name }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publish::media::OCI_INDEX;
    use crate::publish::testing::*;
    use serde_json::json;

    #[test]
    fn references_split_like_docker() {
        let parse = |raw: &str| Reference::parse(raw).unwrap();
        assert_eq!(
            parse("localhost:5000/ns/pi0:v3"),
            Reference {
                repository: "localhost:5000/ns/pi0".to_owned(),
                tag: Some("v3".to_owned()),
                digest: None
            }
        );
        let pinned = parse(&format!("localhost:5000/ns/pi0@{}", digest('a')));
        assert_eq!(
            (pinned.repository.as_str(), pinned.tag),
            ("localhost:5000/ns/pi0", None)
        );
        assert_eq!(pinned.digest, Some(digest('a')));
        assert_eq!(parse("ns/pi0").tag, None);
        assert!(Reference::parse("ns/pi0:-bad").is_err());
        assert!(Reference::parse("").is_err());
    }

    #[test]
    fn docker_schema2_sources_cannot_carry_the_index_annotation() {
        let mut registry = FakeRegistry::default();
        for (reference, image) in [("r/x:a", 'a'), ("r/x:b", 'b')] {
            registry.add(reference, None, image, keyed(&reference[4..]));
            // A classic `docker push`: a Docker schema2 manifest.
            registry.insert(
                reference,
                &json!({"mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                        "digest": digest(image), "size": 525}),
            );
        }
        let dir = tempfile::tempdir().unwrap();
        let mut publish = args("r/x:v1", &["r/x:a", "r/x:b"]);
        publish.index_package = Some(version_json(&dir, r#"{"checkpoints":[]}"#));
        let (result, _) = run(&registry, &publish);
        let error = format!("{:#}", result.unwrap_err());
        for needle in [
            "--index-package needs an OCI image index, but every manifest the sources carry \
             (images and attestations) is a Docker schema2 manifest (r/x:a, r/x:b)",
            "drop the annotation",
            "--output type=image,oci-mediatypes=true,push=true",
        ] {
            assert!(error.contains(needle), "{needle:?} not in:\n{error}");
        }
        assert!(registry.created.borrow().is_empty(), "nothing is created");

        // Without an annotation, a Docker manifest list is a whole version.
        let (result, out) = run(&registry, &args("r/x:v1", &["r/x:a", "r/x:b"]));
        assert_eq!(result.unwrap(), 0, "{out}");

        // A Docker manifest list whose image and attestation are both schema2
        // carries nothing OCI either.
        const SCHEMA2: &str = "application/vnd.docker.distribution.manifest.v2+json";
        registry.add("r/x:s", Some('s'), 'e', keyed("s"));
        let mut list = buildkit_index(&digest('e'), ("linux", "amd64"));
        list["digest"] = Value::from(digest('s'));
        list["mediaType"] =
            Value::from("application/vnd.docker.distribution.manifest.list.v2+json");
        list["manifests"][0]["mediaType"] = Value::from(SCHEMA2);
        list["manifests"][1]["mediaType"] = Value::from(SCHEMA2);
        registry.insert("r/x:s", &list);
        publish.target = "r/x:v3".to_owned();
        publish.sources = vec!["r/x:a".to_owned(), "r/x:s".to_owned()];
        let error = format!("{:#}", run(&registry, &publish).0.unwrap_err());
        assert!(
            error.contains("Docker schema2 manifest (r/x:a, r/x:s)"),
            "{error}"
        );
        assert_eq!(
            registry.created.borrow().len(),
            2,
            "nothing more is created"
        );
        // With an OCI attestation, docker assembles an OCI index.
        list["manifests"][1]["mediaType"] =
            Value::from("application/vnd.oci.image.manifest.v1+json");
        registry.insert("r/x:s", &list);
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        let pushed: Value = serde_json::from_str(&registry.manifest("r/x:v3").unwrap()).unwrap();
        assert_eq!(pushed["mediaType"], OCI_INDEX);
        assert!(pushed["annotations"][PACKAGE_LABEL].is_string(), "{pushed}");
        publish.sources = vec!["r/x:a".to_owned(), "r/x:b".to_owned()];

        // One OCI child and docker assembles an OCI index, annotation kept.
        registry.add("r/x:c", None, 'c', keyed("c"));
        publish.target = "r/x:v2".to_owned();
        publish.sources.push("r/x:c".to_owned());
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            registry
                .created
                .borrow()
                .last()
                .unwrap()
                .contains(&"--annotation".to_owned()),
            "{out}"
        );
    }

    #[test]
    fn the_index_platform_must_match_the_image_config() {
        let mut registry = FakeRegistry::default();
        // The index says arm64, the image config amd64.
        registry.add("r/x:a", Some('i'), 'a', keyed("a"));
        let mut index = buildkit_index(&digest('a'), ("linux", "arm64"));
        index["digest"] = Value::from(digest('i'));
        registry.insert("r/x:a", &index);
        let error = format!("{:#}", resolve_source(&registry, "r/x:a").unwrap_err());
        assert!(
            error.contains(
                "r/x:a: its index declares the image linux/arm64, but the image config says \
                 linux/amd64; rebuild it so they agree"
            ),
            "{error}"
        );
        // An amd64 descriptor over an arm64 config cannot become the default row.
        let mut arm: Value = serde_json::from_str(&keyed("b")).unwrap();
        arm["architecture"] = Value::from("arm64");
        registry.add("r/x:b", Some('j'), 'b', arm.to_string());
        let (result, out) = run(&registry, &args("r/x:v1", &["r/x:b"]));
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(
                "r/x:b: its index declares the image linux/amd64, but the image config says \
                 linux/arm64"
            ),
            "{error}"
        );
        assert!(!out.contains('*'), "no default row: {out}");
        // A platform variant in the descriptor is fine when os/arch agree.
        index["manifests"][0]["platform"]["variant"] = Value::from("v8");
        registry.insert("r/x:a", &index);
        let mut arm: Value = serde_json::from_str(&keyed("a")).unwrap();
        arm["architecture"] = Value::from("arm64");
        registry
            .configs
            .insert(format!("r/x@{}", digest('a')), arm.to_string());
        let child = resolve_source(&registry, "r/x:a").unwrap();
        assert_eq!(child.platform, "linux/arm64/v8");
        assert!(!child.selectable());
    }

    #[test]
    fn carried_attestations_must_describe_a_child() {
        let mut registry = FakeRegistry::default();
        registry.add("r/x:a", Some('i'), 'c', keyed("a"));
        let mut index = buildkit_index(&digest('c'), ("linux", "amd64"));
        index["digest"] = Value::from(digest('i'));
        // An attestation of an image that is not in the version, and one of nothing.
        index["manifests"][1]["annotations"][REFERENCE_DIGEST] = Value::from(digest('z'));
        index["manifests"].as_array_mut().unwrap().push(json!({
            "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": digest('n'),
            "annotations": {REFERENCE_TYPE: ATTESTATION},
            "platform": {"os": "unknown", "architecture": "unknown"}}));
        registry.insert("r/x:a", &index);
        let (result, _) = run(&registry, &args("r/x:v1", &["r/x:a"]));
        let error = format!("{:#}", result.unwrap_err());
        for needle in [
            format!(
                "r/x:a: attestation manifest {} describes {}, not an image of this version",
                digest('a'),
                digest('z')
            ),
            format!("name its image by digest (r/x@{})", digest('c')),
            format!(
                "r/x:a: attestation manifest {} names no image (vnd.docker.reference.digest)",
                digest('n')
            ),
        ] {
            assert!(error.contains(&needle), "{needle:?} not in:\n{error}");
        }
        assert!(registry.created.borrow().is_empty());

        // An attestation of another source's image still describes a child.
        let mut index = buildkit_index(&digest('c'), ("linux", "amd64"));
        index["digest"] = Value::from(digest('i'));
        registry.insert("r/x:a", &index);
        registry.add("r/x:b", Some('j'), 'b', keyed("b"));
        let mut index = buildkit_index(&digest('b'), ("linux", "amd64"));
        index["digest"] = Value::from(digest('j'));
        index["manifests"][1]["annotations"][REFERENCE_DIGEST] = Value::from(digest('c'));
        registry.insert("r/x:b", &index);
        let children: Vec<Child> = ["r/x:a", "r/x:b"]
            .iter()
            .map(|source| resolve_source(&registry, source).unwrap())
            .collect();
        assert!(attestation_errors(&children).is_empty());
        assert_eq!(attestation_errors(&children[1..]).len(), 1);
    }
}
