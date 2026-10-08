//! `rlmesh registry publish`: assemble pushed per-variant images into one OCI
//! image index and push it as a version. The managed platform reads the index
//! as the version and its non-attestation children as compute variants, each
//! declared by the `variant` block of its `dev.rlmesh.package` label (see
//! [`crate::package`]).
//!
//! Registry access goes through `docker buildx imagetools` (inspect to read
//! each source, create to assemble and push), so docker's own credential
//! helpers, including `docker-credential-rlmesh`, authenticate it. Sources are
//! pinned by digest before the index is created, so a tag moving underneath a
//! publish cannot swap a child. A version tag (TARGET, `--tag`) is pushed
//! only over a tag the registry says is absent, or one already holding the
//! same index (which is left alone); a different index, or a tag that cannot
//! be read, needs `--force`. The channel tag moves. The index is pushed to
//! TARGET alone and checked as the registry stores it; only then are the
//! other tags pointed at TARGET's digest, so every tag is one manifest.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::cli::PublishArgs;
use crate::image_check::{
    self, CheckReport, DESCRIBE_LABEL, ImageConfig, Kind, PACKAGE_LABEL, parse_serve_command,
};
use crate::package::{self, ComputeBlocks, Requires};
use crate::render::Style;

/// The annotations BuildKit puts on an attestation manifest inside an index:
/// its type, and the digest of the image it describes.
const REFERENCE_TYPE: &str = "vnd.docker.reference.type";
const REFERENCE_DIGEST: &str = "vnd.docker.reference.digest";
const ATTESTATION: &str = "attestation-manifest";
/// The media-type prefix of Docker's own (non-OCI) manifests.
const DOCKER_MEDIA_PREFIX: &str = "application/vnd.docker.";
const OCI_INDEX: &str = "application/vnd.oci.image.index.v1+json";

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

pub(crate) fn publish(args: &PublishArgs, stdout: &mut impl Write, style: Style) -> Result<i32> {
    publish_with(&DockerImagetools, args, stdout, style)
}

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

fn check_tag(tag: &str) -> Result<()> {
    let mut chars = tag.chars();
    let valid = tag.len() <= 128
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    if !valid {
        bail!("{tag:?} is not a valid image tag");
    }
    Ok(())
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
    fn selectable(&self) -> bool {
        let mut parts = self.platform.split('/');
        package::selectable(parts.next().unwrap_or(""), parts.next().unwrap_or(""))
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
fn resolve_source(tools: &impl Imagetools, source: &str) -> Result<Child> {
    let reference = Reference::parse(source)?;
    let manifest: Value = serde_json::from_str(&tools.manifest(source)?)
        .with_context(|| format!("parsing the manifest of {source}"))?;
    let digest = manifest
        .get("digest")
        .and_then(Value::as_str)
        .with_context(|| format!("the manifest of {source} carries no digest"))?
        .to_owned();
    let media_type = |value: &Value| {
        value
            .get("mediaType")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let pinned = format!("{}@{digest}", reference.repository);
    let mut attestations = Vec::new();
    let (image_digest, image_media_type, platform) = match manifest
        .get("manifests")
        .and_then(Value::as_array)
    {
        None => (digest, media_type(&manifest), None),
        Some(entries) => {
            let mut images = Vec::new();
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
                        if platform.is_empty() {
                            "platform-less"
                        } else {
                            platform.as_str()
                        }
                    );
                }
                images.push((digest.to_owned(), media_type(entry), platform));
            }
            match images.len() {
                1 => {
                    let (digest, media_type, platform) = images.swap_remove(0);
                    (digest, media_type, Some(platform))
                }
                0 => bail!("{source} is an index with no image in it"),
                n => bail!(
                    "{source} is an index of {n} images ({}); each source must be one image, \
                         since each child of the published index is one variant",
                    images
                        .iter()
                        .map(|(_, _, platform)| platform.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }
    };
    let config = ImageConfig::from_oci_config(
        &tools.config(&format!("{}@{image_digest}", reference.repository))?,
    )
    .map_err(anyhow::Error::msg)
    .with_context(|| format!("reading the image config of {source}"))?;
    let configured = if config.os.is_empty() {
        String::new()
    } else {
        format!("{}/{}", config.os, config.architecture)
    };
    // The platform schedules a child by its descriptor and the runtime runs
    // the config, so the two must name the same platform: eligibility (and
    // the default row) is decided from it.
    let platform = match platform {
        Some(declared) => {
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
            declared
        }
        None => configured,
    };
    if config.os != "linux" {
        bail!(
            "{source} is a {} image; the platform runs linux images only (build with \
             --platform linux/amd64)",
            if platform.is_empty() {
                "platform-less"
            } else {
                platform.as_str()
            }
        );
    }
    let index_annotation = manifest
        .get("manifests")
        .and(manifest.get("annotations"))
        .and_then(|annotations| annotations.get(PACKAGE_LABEL))
        .and_then(Value::as_str)
        .map(str::to_owned);
    Ok(Child {
        source: source.to_owned(),
        pinned,
        image_digest,
        media_type: image_media_type,
        platform,
        config,
        attestations,
        index_annotation,
    })
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

/// A child with its package label read.
#[derive(Debug, Clone)]
pub(crate) struct VariantImage {
    pub child: Child,
    pub key: String,
    pub kind: Option<Kind>,
    pub blocks: ComputeBlocks,
    /// The requires the platform records for the base row: declared, or
    /// inferred from the image's markers.
    pub requires: Requires,
    pub requires_inferred: bool,
    pub facets: BTreeMap<String, String>,
    /// The row keys the platform derives (one per profile).
    pub rows: Vec<String>,
}

impl VariantImage {
    fn selectable(&self) -> bool {
        self.child.selectable()
    }
}

/// What the image serves: the describe label's kind, else the serve command's.
fn image_kind(config: &ImageConfig) -> Option<Kind> {
    config
        .labels
        .get(DESCRIBE_LABEL)
        .and_then(|raw| image_check::parse_describe_label(raw).ok())
        .and_then(|label| label.kind)
        .or_else(|| parse_serve_command(config).kind())
}

/// The version's default row: the first `linux/amd64` child, at its default
/// profile, as `(variant index, row index)`.
pub(crate) fn default_row(variants: &[VariantImage]) -> Option<(usize, usize)> {
    variants
        .iter()
        .position(VariantImage::selectable)
        .map(|index| (index, variants[index].blocks.default_profile().unwrap_or(0)))
}

/// Read and check each child's package label, and the set as a whole: every
/// child needs a label with a `variant.key`, its declaration must pass the
/// platform's checks, row keys are unique, kinds agree, and at least one
/// child is `linux/amd64`. Returns the variants, the warnings worth printing,
/// and every error found (all of them, so one run reports everything to fix).
pub(crate) fn check_variants(
    children: Vec<Child>,
) -> (Vec<VariantImage>, Vec<String>, Vec<String>) {
    let mut variants = Vec::new();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    for child in children {
        let source = child.source.clone();
        let Some(raw) = child
            .config
            .labels
            .get(PACKAGE_LABEL)
            .filter(|raw| !raw.trim().is_empty())
        else {
            errors.push(format!(
                "{source}: no {PACKAGE_LABEL} label; each child needs one declaring its \
                 variant.key"
            ));
            continue;
        };
        let package: Map<String, Value> = match serde_json::from_str(raw) {
            Ok(Value::Object(package)) => package,
            Ok(_) => {
                errors.push(format!("{source}: {PACKAGE_LABEL} must be a JSON object"));
                continue;
            }
            Err(err) => {
                errors.push(format!(
                    "{source}: {PACKAGE_LABEL} is not valid JSON: {err}"
                ));
                continue;
            }
        };
        if package.get("schemaVersion").and_then(Value::as_u64) != Some(1) {
            errors.push(format!(
                "{source}: {PACKAGE_LABEL} schemaVersion {} is not the supported version 1",
                package.get("schemaVersion").unwrap_or(&Value::Null)
            ));
        }
        let report: CheckReport = package::check_variant(&child.config);
        errors.extend(report.failed.iter().map(|m| format!("{source}: {m}")));
        warnings.extend(report.warnings.iter().map(|m| format!("{source}: {m}")));
        let (blocks, _) = package::parse_compute_blocks(&package);
        let Some(key) = blocks.variant.as_ref().and_then(|v| v.key.clone()) else {
            if blocks.variant.is_none() {
                errors.push(format!(
                    "{source}: {PACKAGE_LABEL} declares no variant block; each child of a \
                     version is addressed by its variant.key"
                ));
            }
            continue;
        };
        let kind = image_kind(&child.config);
        if kind.is_none() {
            warnings.push(format!(
                "{source}: kind unknown (no {DESCRIBE_LABEL} label and the command does not \
                 run -m rlmesh.serve); the platform's probe decides it"
            ));
        }
        if !child.selectable() {
            warnings.push(format!(
                "{source}: {} image; recorded with status excluded and never selected, since \
                 the fleet runs linux/amd64",
                child.platform
            ));
        }
        let inferred = package::infer(&child.config);
        let (requires, facets) = package::effective(&blocks, &inferred);
        let requires_inferred = blocks
            .variant
            .as_ref()
            .is_some_and(|variant| variant.requires.is_none());
        let rows = blocks.row_keys(&key);
        variants.push(VariantImage {
            child,
            key,
            kind,
            blocks,
            requires,
            requires_inferred,
            facets,
            rows,
        });
    }
    let mut by_key: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut by_row: BTreeMap<&str, Vec<&VariantImage>> = BTreeMap::new();
    for variant in &variants {
        by_key
            .entry(&variant.key)
            .or_default()
            .push(&variant.child.source);
        for row in &variant.rows {
            by_row.entry(row).or_default().push(variant);
        }
    }
    for (key, sources) in by_key.iter().filter(|(_, sources)| sources.len() > 1) {
        errors.push(format!(
            "variant.key {key:?} is declared by {}; keys must be unique within a version",
            sources.join(" and ")
        ));
    }
    for (row, owners) in &by_row {
        let keys: BTreeSet<&str> = owners.iter().map(|v| v.key.as_str()).collect();
        // A shared variant.key was reported above; this is a profile row
        // colliding with another child's key or row.
        if owners.len() > 1 && keys.len() > 1 {
            errors.push(format!(
                "row key {row:?} is derived by {}; rows are <variant.key>-<profile.key>, so \
                 rename a key",
                owners
                    .iter()
                    .map(|v| v.child.source.as_str())
                    .collect::<Vec<_>>()
                    .join(" and ")
            ));
        }
    }
    let mut by_kind: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for variant in &variants {
        if let Some(kind) = variant.kind {
            by_kind
                .entry(kind.name())
                .or_default()
                .push(&variant.child.source);
        }
    }
    if by_kind.len() > 1 {
        errors.push(format!(
            "mixed kinds: {}; every variant of a version serves the same kind",
            by_kind
                .iter()
                .map(|(kind, sources)| format!("{kind} ({})", sources.join(", ")))
                .collect::<Vec<_>>()
                .join(" vs ")
        ));
    }
    if !variants.is_empty() && default_row(&variants).is_none() {
        errors.push(
            "no child is linux/amd64, so nothing in this version can be scheduled".to_owned(),
        );
    }
    (variants, warnings, errors)
}

/// Read `--index-package`: a JSON object of version-level package data.
/// `schemaVersion` defaults to 1. The platform reads only the version-level
/// keys off the index ([`package::VERSION_PACKAGE_KEYS`]); anything else is
/// kept as written but warned about. Returns compact JSON and the warnings.
pub(crate) fn index_package(raw: &str, origin: &str) -> Result<(String, Vec<String>)> {
    let mut value: Value =
        serde_json::from_str(raw).with_context(|| format!("{origin} is not valid JSON"))?;
    let Value::Object(package) = &mut value else {
        bail!("{origin} must be a JSON object");
    };
    match package.get("schemaVersion") {
        None => {
            package.insert("schemaVersion".to_owned(), Value::from(1));
        }
        Some(version) if version.as_u64() == Some(1) => {}
        Some(version) => bail!("{origin} schemaVersion {version} is not the supported version 1"),
    }
    let warnings = package::index_annotation_warnings(package)
        .into_iter()
        .map(|warning| format!("{origin}: {warning}"))
        .collect();
    Ok((serde_json::to_string(&value)?, warnings))
}

/// Warn for each child that sets a version-level key differently from the
/// index annotation, which wins on the platform.
fn index_disagreements(annotation: &str, variants: &[VariantImage]) -> Vec<String> {
    let Ok(Value::Object(index)) = serde_json::from_str::<Value>(annotation) else {
        return Vec::new();
    };
    variants
        .iter()
        .filter_map(|variant| {
            let child = package::package_object(&variant.child.config)?;
            let differs: Vec<&str> = package::VERSION_PACKAGE_KEYS
                .iter()
                .copied()
                .filter(|key| matches!((child.get(*key), index.get(*key)), (Some(a), Some(b)) if a != b))
                .collect();
            (!differs.is_empty()).then(|| {
                format!(
                    "{}: sets {} differently from the index annotation, which wins",
                    variant.child.source,
                    differs.join(", ")
                )
            })
        })
        .collect()
}

/// The full target references: TARGET, each `--tag` in its repository, then
/// the channel, without repeats.
pub(crate) fn target_tags(args: &PublishArgs) -> Result<(Reference, Vec<String>)> {
    let target = Reference::parse(&args.target)?;
    if target.digest.is_some() || target.tag.is_none() {
        bail!(
            "target {:?} must be REPOSITORY:TAG (an index is published under a tag, not a digest)",
            args.target
        );
    }
    let mut tags = vec![format!(
        "{}:{}",
        target.repository,
        target.tag.as_deref().unwrap_or_default()
    )];
    for tag in args.tags.iter().chain(&args.channel) {
        let tag = tag.trim();
        check_tag(tag)?;
        let full = format!("{}:{tag}", target.repository);
        if !tags.contains(&full) {
            tags.push(full);
        }
    }
    Ok((target, tags))
}

/// The `imagetools create` arguments: every tag, the index annotation, and the
/// sources pinned by digest, in command-line order.
pub(crate) fn create_args(
    tags: &[String],
    annotation: Option<&str>,
    variants: &[VariantImage],
    dry_run: bool,
) -> Vec<String> {
    let mut args = Vec::new();
    if dry_run {
        args.push("--dry-run".to_owned());
    }
    for tag in tags {
        args.push("--tag".to_owned());
        args.push(tag.clone());
    }
    if let Some(annotation) = annotation {
        args.push("--annotation".to_owned());
        args.push(format!("index:{PACKAGE_LABEL}={annotation}"));
    }
    args.extend(variants.iter().map(|variant| variant.child.pinned.clone()));
    args
}

/// What a tag holds before the publish.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Existing {
    /// The registry says there is no such manifest.
    Absent,
    /// It could not be read (auth, transport, a registry error, output that
    /// does not parse), so nothing is known about it; the reason.
    Unreadable(String),
    /// A manifest at `digest`, as stored (`raw`).
    Found { digest: String, raw: Value },
}

/// Where a tag points before the publish, against the index being published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TagState {
    /// Nothing there yet.
    New,
    /// Not readable; the reason.
    Unreadable(String),
    /// Already this index, at this digest: left as is.
    Unchanged(String),
    /// Another manifest, at this digest.
    Moves(String),
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
/// checked.
fn existing_manifest(tools: &impl Imagetools, tag: &str) -> Existing {
    let read = || -> Result<Existing> {
        let view: Value = serde_json::from_str(&tools.manifest(tag)?)
            .context("docker printed a manifest that is not JSON")?;
        let digest = view
            .get("digest")
            .and_then(Value::as_str)
            .context("docker printed a manifest without a digest")?
            .to_owned();
        let raw = serde_json::from_str(&tools.raw(tag)?).context("its raw manifest is not JSON")?;
        Ok(Existing::Found { digest, raw })
    };
    match read() {
        Ok(existing) => existing,
        Err(error) => {
            let error = format!("{error:#}");
            if absent(tag, &error) {
                Existing::Absent
            } else {
                Existing::Unreadable(error)
            }
        }
    }
}

/// Whether an existing manifest is the index about to be published. Both are
/// compared as stored, field by field: the same media type and artifactType,
/// the same manifests in the same order (digest, size, platform, annotations,
/// every descriptor field), the same annotations and subject. Only the
/// serialization (whitespace, key order) may differ.
pub(crate) fn same_index(existing: &Value, index: &Value) -> bool {
    existing == index
}

/// Each tag's state against `index`, the index `imagetools create --dry-run`
/// assembles.
pub(crate) fn tag_states(existing: &[Existing], index: &Value) -> Vec<TagState> {
    existing
        .iter()
        .map(|existing| match existing {
            Existing::Absent => TagState::New,
            Existing::Unreadable(reason) => TagState::Unreadable(reason.clone()),
            Existing::Found { digest, raw } if same_index(raw, index) => {
                TagState::Unchanged(digest.clone())
            }
            Existing::Found { digest, .. } => TagState::Moves(digest.clone()),
        })
        .collect()
}

/// Print each tag's state and return the refusals: TARGET and `--tag` tags
/// name a version and do not move, or get pushed over a state that could not
/// be read, without `--force`; the channel tag (its full reference, unless it
/// is also a version tag) moves, which is its purpose, and says from where.
/// Every tag ends at TARGET's digest, so another tag holding the same index
/// at another digest (another serialization) is re-pointed.
fn report_tags(
    stdout: &mut impl Write,
    style: Style,
    tags: &[String],
    states: &[TagState],
    channel: Option<&str>,
    force: bool,
) -> Result<Vec<String>> {
    let mut refusals = Vec::new();
    let width = tags
        .iter()
        .map(|tag| tag.chars().count())
        .max()
        .unwrap_or(0);
    let version = match &states[0] {
        TagState::Unchanged(digest) => Some(digest.as_str()),
        _ => None,
    };
    writeln!(stdout)?;
    writeln!(stdout, "{}", style.bold("Tags:"))?;
    for (position, (tag, state)) in tags.iter().zip(states).enumerate() {
        let is_channel = channel == Some(tag.as_str());
        let note = match state {
            TagState::New => style.muted("new"),
            TagState::Unchanged(digest) if position == 0 || version == Some(digest.as_str()) => {
                style.muted(&format!("unchanged ({digest}), not pushed again"))
            }
            TagState::Unchanged(digest) => style.muted(&match version {
                Some(version) => {
                    format!("same index at {digest}, re-pointed at {version} (TARGET's digest)")
                }
                None => format!(
                    "same index at {digest}, re-pointed at TARGET's new digest if that differs"
                ),
            }),
            TagState::Moves(digest) if is_channel => format!("moves from {digest} (channel)"),
            TagState::Moves(digest) if force => {
                style.yellow(&format!("moves from {digest} (--force)"))
            }
            TagState::Moves(digest) => {
                refusals.push(format!(
                    "{tag} already points at {digest}, a different index; a version tag does \
                     not move without --force"
                ));
                style.red_bold(&format!("refused: points at {digest}"))
            }
            TagState::Unreadable(reason) if is_channel => {
                style.yellow(&format!("unreadable, overwritten (channel): {reason}"))
            }
            TagState::Unreadable(reason) if force => {
                style.yellow(&format!("unreadable, overwritten (--force): {reason}"))
            }
            TagState::Unreadable(reason) => {
                refusals.push(format!(
                    "{tag} could not be read ({reason}), so it may already hold another \
                     version; a version tag is pushed only where the registry says it is \
                     absent, so fix access to it or pass --force"
                ));
                style.red_bold("refused: could not be read")
            }
        };
        writeln!(stdout, "  {tag:<width$}  {note}")?;
    }
    Ok(refusals)
}

pub(crate) fn publish_with(
    tools: &impl Imagetools,
    args: &PublishArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let (target, tags) = target_tags(args)?;
    let mut warnings = Vec::new();
    let mut annotation = match &args.index_package {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let (annotation, index_warnings) = index_package(&raw, &path.display().to_string())?;
            warnings.extend(index_warnings);
            Some(annotation)
        }
        None => None,
    };

    let mut children = Vec::new();
    let mut errors = Vec::new();
    for source in &args.sources {
        match resolve_source(tools, source) {
            Ok(child) => children.push(child),
            Err(error) => errors.push(format!("{error:#}")),
        }
    }
    errors.extend(attestation_errors(&children));
    if annotation.is_some()
        && children.len() == args.sources.len()
        && let Some(error) = docker_list_error(&children)
    {
        errors.push(error);
    }
    // A lone index source is republished as is, its own annotation included,
    // so that annotation is the version's and is checked like --index-package.
    // Anywhere else, docker assembles a new index without it.
    let inherits = annotation.is_none() && args.sources.len() == 1;
    for child in &children {
        let Some(raw) = &child.index_annotation else {
            continue;
        };
        if !inherits {
            warnings.push(format!(
                "{}: the {PACKAGE_LABEL} annotation on its index is not carried into the version \
                 (only --index-package sets the version's)",
                child.source
            ));
            continue;
        }
        let origin = format!(
            "{}: the {PACKAGE_LABEL} annotation on its index",
            child.source
        );
        match index_package(raw, &origin) {
            Ok((inherited, index_warnings)) => {
                warnings.push(format!(
                    "{origin} is kept on the published version (a single index source is \
                     republished as is); pass --index-package to replace it"
                ));
                warnings.extend(index_warnings);
                annotation = Some(inherited);
            }
            Err(error) => errors.push(format!(
                "{error:#}; it would be kept on the published version, so fix it or pass \
                 --index-package to replace it"
            )),
        }
    }
    let (variants, check_warnings, check_errors) = check_variants(children);
    warnings.extend(check_warnings);
    if let Some(annotation) = &annotation {
        warnings.extend(index_disagreements(annotation, &variants));
    }
    errors.extend(check_errors);

    writeln!(
        stdout,
        "{}",
        style.bold(&format!(
            "{} {} ({} variant{})",
            if args.dry_run {
                "Dry run: would publish"
            } else {
                "Publishing"
            },
            args.target,
            variants.len(),
            if variants.len() == 1 { "" } else { "s" }
        ))
    )?;
    if !variants.is_empty() {
        writeln!(stdout)?;
        write_table(stdout, style, &variants)?;
    }
    if !warnings.is_empty() {
        writeln!(stdout)?;
        for warning in &warnings {
            writeln!(stdout, "{}  {warning}", style.yellow("warn"))?;
        }
    }
    let cannot_publish = |errors: &[String]| {
        anyhow::anyhow!(
            "cannot publish {}:\n{}",
            args.target,
            errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    if !errors.is_empty() {
        return Err(cannot_publish(&errors));
    }

    // The index as docker will assemble it: printed under --dry-run, checked
    // to carry the annotation, and compared with what each tag holds. Only
    // --index-package is passed as an annotation; an inherited one rides
    // along with its index.
    let explicit = args.index_package.as_ref().and(annotation.as_deref());
    let preview = tools.create(&create_args(&tags, explicit, &variants, true))?;
    if args.dry_run {
        writeln!(stdout)?;
        writeln!(
            stdout,
            "{}",
            style.muted(&format!("Index for {} (nothing pushed):", tags.join(", ")))
        )?;
        writeln!(stdout, "{}", preview.trim())?;
    }
    let index: Value = serde_json::from_str(&preview)
        .context("parsing the index docker buildx imagetools create --dry-run printed")?;
    if explicit.is_some()
        && let Some(error) = annotation_dropped(&index)
    {
        return Err(cannot_publish(&[format!("docker would assemble {error}")]));
    }
    let existing: Vec<Existing> = tags
        .iter()
        .map(|tag| existing_manifest(tools, tag))
        .collect();
    let states = tag_states(&existing, &index);
    let channel = args
        .channel
        .as_deref()
        .map(|channel| format!("{}:{}", target.repository, channel.trim()))
        .filter(|channel| {
            *channel != tags[0]
                && !args
                    .tags
                    .iter()
                    .any(|tag| *channel == format!("{}:{}", target.repository, tag.trim()))
        });
    let refusals = report_tags(
        stdout,
        style,
        &tags,
        &states,
        channel.as_deref(),
        args.force,
    )?;
    if !refusals.is_empty() {
        return Err(cannot_publish(&refusals));
    }
    if args.dry_run {
        return Ok(0);
    }

    // The version is TARGET's digest: the one it holds when it already holds
    // this index (never pushed again, since docker would re-serialize it),
    // else that of the index pushed to TARGET alone and checked as stored.
    // Only then do the other tags move, each pointed at that digest, which
    // docker copies byte for byte, so every alias is the same manifest and a
    // failed check leaves them where they were.
    let repository = &target.repository;
    let version = match &states[0] {
        TagState::Unchanged(digest) => digest.clone(),
        before => push_target(
            tools, repository, &tags[0], before, explicit, &variants, &index,
        )?,
    };
    let aliases: Vec<&String> = tags[1..]
        .iter()
        .zip(&states[1..])
        .filter(|(_, state)| **state != TagState::Unchanged(version.clone()))
        .map(|(tag, _)| tag)
        .collect();
    writeln!(stdout)?;
    if aliases.is_empty() && states[0] == TagState::Unchanged(version.clone()) {
        writeln!(
            stdout,
            "{}",
            style.success(&format!(
                "Already published {repository}@{version}; every tag holds this index, nothing \
                 pushed"
            ))
        )?;
        return Ok(0);
    }
    if !aliases.is_empty() {
        let mut args = Vec::new();
        for tag in &aliases {
            args.push("--tag".to_owned());
            args.push((*tag).clone());
        }
        args.push(format!("{repository}@{version}"));
        tools.create(&args).with_context(|| {
            format!(
                "{} holds the version at {version}, but pointing {} at it failed; run the same \
                 publish again to finish (TARGET is left as is and the other tags are pointed \
                 at it)",
                tags[0],
                aliases
                    .iter()
                    .map(|tag| tag.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    }
    writeln!(
        stdout,
        "{}",
        style.success(&format!("Published {repository}@{version}"))
    )?;
    for (tag, state) in tags.iter().zip(&states) {
        match state {
            TagState::Unchanged(digest) if *digest == version => writeln!(
                stdout,
                "  {} {tag} (already {digest})",
                style.muted("unchanged")
            )?,
            TagState::Unchanged(digest) => writeln!(
                stdout,
                "  {} {tag} (same index, from {digest})",
                style.muted("re-pointed")
            )?,
            _ => writeln!(stdout, "  {} {tag}", style.muted("tagged"))?,
        }
    }
    Ok(0)
}

/// Push the index to TARGET alone and check it as the registry stores it:
/// the index assembled (the same manifests, annotations and media type),
/// and, with `--index-package`, an OCI image index carrying that annotation.
/// Returns its digest. On a failed check only TARGET has moved, and the error
/// says how to recover from what it held `before`.
fn push_target(
    tools: &impl Imagetools,
    repository: &str,
    target: &str,
    before: &TagState,
    explicit: Option<&str>,
    variants: &[VariantImage],
    index: &Value,
) -> Result<String> {
    tools.create(&create_args(
        &[target.to_owned()],
        explicit,
        variants,
        false,
    ))?;
    let recovery = match before {
        TagState::Moves(previous) => format!(
            "{target} pointed at {previous} before this run; restore it with `docker buildx \
             imagetools create --tag {target} {repository}@{previous}`, or publish again with \
             --force once the cause is fixed"
        ),
        TagState::Unreadable(_) => format!(
            "{target} could not be read before this run, so what it held is unknown; publish \
             again with --force once the cause is fixed"
        ),
        TagState::New | TagState::Unchanged(_) => format!(
            "{target} did not exist before this run, so it holds only this push: publish again \
             with --force once the cause is fixed, or delete {target} from the registry first"
        ),
    };
    let stored = || -> Result<(String, Value)> {
        let view: Value = serde_json::from_str(&tools.manifest(target)?)
            .context("docker printed a manifest that is not JSON")?;
        let digest = view
            .get("digest")
            .and_then(Value::as_str)
            .context("docker printed a manifest without a digest")?
            .to_owned();
        let raw = serde_json::from_str(&tools.raw(&format!("{repository}@{digest}"))?)
            .context("its raw manifest is not JSON")?;
        Ok((digest, raw))
    };
    let (digest, raw) = stored().map_err(|error| {
        anyhow::anyhow!(
            "pushed {target}, but could not read it back to check it ({error:#}); no other tag \
             was moved. {recovery}"
        )
    })?;
    let problem = explicit.and_then(|_| annotation_dropped(&raw)).or_else(|| {
        (!same_index(&raw, index)).then(|| {
            "an index other than the one assembled (its manifests, annotations or media \
                 type differ)"
                .to_owned()
        })
    });
    if let Some(problem) = problem {
        bail!(
            "pushed {target} ({digest}), but the registry holds {problem}; no other tag was \
             moved. {recovery}"
        );
    }
    Ok(digest)
}

/// Why `index` would not carry the `--index-package` annotation, if it would
/// not: it is not an OCI image index (a Docker manifest list has no
/// annotations), or the annotation is missing from it.
fn annotation_dropped(index: &Value) -> Option<String> {
    let media_type = index
        .get("mediaType")
        .and_then(Value::as_str)
        .unwrap_or("(no media type)");
    if media_type != OCI_INDEX {
        return Some(format!(
            "a {media_type}, not an OCI image index, so the --index-package annotation is \
             dropped; rebuild the sources with OCI media types (docker buildx build --push)"
        ));
    }
    let kept = index
        .get("annotations")
        .and_then(|annotations| annotations.get(PACKAGE_LABEL))
        .is_some();
    (!kept).then(|| format!("an index without the {PACKAGE_LABEL} annotation --index-package sets"))
}

fn write_table(stdout: &mut impl Write, style: Style, variants: &[VariantImage]) -> Result<()> {
    let headers = [
        "VARIANT", "KIND", "PLATFORM", "PRIORITY", "FACETS", "REQUIRES", "ROWS", "IMAGE",
    ];
    let default = default_row(variants);
    let rows: Vec<[String; 8]> = variants
        .iter()
        .enumerate()
        .map(|(index, variant)| {
            let priority = variant
                .blocks
                .variant
                .as_ref()
                .map_or(0, |declared| declared.priority);
            let or_dash = |value: String| {
                if value.is_empty() {
                    "-".to_owned()
                } else {
                    value
                }
            };
            let requires = match (
                package::format_requires(&variant.requires),
                variant.requires_inferred,
            ) {
                (requires, _) if requires.is_empty() => "-".to_owned(),
                (requires, true) => format!("{requires} (inferred)"),
                (requires, false) => requires,
            };
            let row_keys = variant
                .rows
                .iter()
                .enumerate()
                .map(|(row, key)| {
                    if default == Some((index, row)) {
                        format!("{key}*")
                    } else {
                        key.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            [
                variant.key.clone(),
                variant.kind.map_or("?", Kind::name).to_owned(),
                if variant.selectable() {
                    variant.child.platform.clone()
                } else {
                    format!("{} (excluded)", variant.child.platform)
                },
                priority.to_string(),
                or_dash(package::format_facets(&variant.facets)),
                requires,
                row_keys,
                format!(
                    "{} {}",
                    variant.child.source,
                    short_digest(&variant.child.image_digest)
                ),
            ]
        })
        .collect();
    let widths: Vec<usize> = (0..headers.len())
        .map(|column| {
            rows.iter()
                .map(|row| row[column].chars().count())
                .chain([headers[column].len()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let line = |cells: &[String]| {
        cells
            .iter()
            .zip(&widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_owned()
    };
    let header: Vec<String> = headers.iter().map(|h| (*h).to_owned()).collect();
    writeln!(stdout, "{}", style.bold(&line(&header)))?;
    for row in &rows {
        writeln!(stdout, "{}", line(row))?;
    }
    if default.is_some() {
        writeln!(
            stdout,
            "{}",
            style.muted(
                "* the version's default row: the first linux/amd64 child, at its default profile"
            )
        )?;
    }
    Ok(())
}

fn short_digest(digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    format!("({})", &hex[..hex.len().min(12)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    use anyhow::anyhow;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    /// Canned registry contents, in the shapes `docker buildx imagetools
    /// inspect` prints for `.Manifest` and `.Image`.
    #[derive(Default)]
    struct FakeRegistry {
        manifests: RefCell<BTreeMap<String, String>>,
        raws: RefCell<BTreeMap<String, String>>,
        configs: BTreeMap<String, String>,
        created: RefCell<Vec<Vec<String>>>,
        /// Applied to each index a push assembles before it is stored, as a
        /// registry or docker that loses part of it would.
        mangle: Cell<Option<fn(&mut Value)>>,
    }

    /// The index `imagetools create ARGS` assembles, reduced to what the
    /// tests compare: one child per source, the index annotation. Like
    /// docker, it is a Docker manifest list, without annotations, when every
    /// descriptor the sources carry is a Docker manifest.
    fn assembled(registry: &FakeRegistry, args: &[String]) -> (Vec<String>, Value) {
        let (mut tags, mut manifests, mut annotations) = (Vec::new(), Vec::new(), Map::new());
        let mut docker_only = true;
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--dry-run" => {}
                "--tag" => tags.push(args.next().unwrap().clone()),
                "--annotation" => {
                    let (key, value) = args.next().unwrap()["index:".len()..]
                        .split_once('=')
                        .unwrap();
                    annotations.insert(key.to_owned(), Value::from(value));
                }
                source => {
                    let digest = source.split_once('@').unwrap().1;
                    docker_only &= registry
                        .media_types(digest)
                        .iter()
                        .all(|media_type| media_type.starts_with(DOCKER_MEDIA_PREFIX));
                    manifests.push(json!({"digest": digest}));
                }
            }
        }
        let mut index = json!({"schemaVersion": 2, "mediaType": OCI_INDEX, "manifests": manifests});
        if docker_only {
            index["mediaType"] =
                Value::from("application/vnd.docker.distribution.manifest.list.v2+json");
        } else if !annotations.is_empty() {
            index["annotations"] = Value::Object(annotations);
        }
        (tags, index)
    }

    impl Imagetools for FakeRegistry {
        fn manifest(&self, reference: &str) -> Result<String> {
            let key = self.resolve(reference);
            match self.manifests.borrow().get(&key) {
                // A canned failure other than "not found".
                Some(error) if error.starts_with("ERROR:") => Err(anyhow!("{error}")),
                Some(manifest) => Ok(manifest.clone()),
                None => Err(anyhow!("ERROR: {reference}: not found")),
            }
        }

        fn config(&self, reference: &str) -> Result<String> {
            self.configs
                .get(reference)
                .cloned()
                .ok_or_else(|| anyhow!("ERROR: {reference}: not found"))
        }

        /// The canned raw manifest when one was set, else the `.Manifest`
        /// view without the descriptor fields docker adds to it.
        fn raw(&self, reference: &str) -> Result<String> {
            if let Some(raw) = self.raws.borrow().get(&self.resolve(reference)) {
                return Ok(raw.clone());
            }
            let mut view: Value = serde_json::from_str(&self.manifest(reference)?)?;
            if let Some(view) = view.as_object_mut() {
                view.remove("digest");
                view.remove("size");
            }
            Ok(view.to_string())
        }

        /// Like docker: a lone index source without annotations is copied
        /// byte for byte; anything else is assembled into a new index.
        fn create(&self, args: &[String]) -> Result<String> {
            self.created.borrow_mut().push(args.to_vec());
            let (tags, mut index) = assembled(self, args);
            if args.iter().any(|a| a == "--dry-run") {
                return Ok(format!("{}\n", serde_json::to_string_pretty(&index)?));
            }
            let (mut sources, mut rest, mut annotated) = (Vec::new(), args.iter(), false);
            while let Some(arg) = rest.next() {
                match arg.as_str() {
                    "--tag" => _ = rest.next(),
                    "--annotation" => annotated = rest.next().is_some(),
                    source => sources.push(source),
                }
            }
            let copied = match sources.as_slice() {
                [source] if !annotated => {
                    let manifest: Value = serde_json::from_str(&self.manifest(source)?)?;
                    if manifest.get("manifests").is_some() {
                        Some(self.raw(source)?)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let raw = match copied {
                Some(raw) => raw,
                None => {
                    if let Some(mangle) = self.mangle.get() {
                        mangle(&mut index);
                    }
                    serde_json::to_string_pretty(&index)?
                }
            };
            for tag in tags {
                self.store(&tag, &raw);
            }
            Ok(String::new())
        }
    }

    fn digest(seed: char) -> String {
        format!("sha256:{}", seed.to_string().repeat(64))
    }

    /// The digest of a manifest stored as `raw`, as a registry computes it.
    fn sha256(raw: &str) -> String {
        let hex: String = Sha256::digest(raw.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        format!("sha256:{hex}")
    }

    /// A BuildKit push: an index of one image plus its attestation manifest.
    fn buildkit_index(image: &str, platform: (&str, &str)) -> Value {
        json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "digest": digest('i'),
            "manifests": [
                {"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": image,
                 "platform": {"os": platform.0, "architecture": platform.1}},
                {"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": digest('a'),
                 "annotations": {"vnd.docker.reference.digest": image,
                                 "vnd.docker.reference.type": "attestation-manifest"},
                 "platform": {"os": "unknown", "architecture": "unknown"}}
            ]
        })
    }

    fn oci_config(env: &[&str], cmd: &[&str], package: Option<Value>) -> String {
        let mut labels = Map::new();
        if let Some(package) = package {
            labels.insert(PACKAGE_LABEL.to_owned(), Value::from(package.to_string()));
        }
        json!({"os": "linux", "architecture": "amd64",
               "config": {"Env": env, "Cmd": cmd, "Labels": labels}})
        .to_string()
    }

    const MODEL: &[&str] = &["python", "-m", "rlmesh.serve", "pi0:Policy"];
    const ENV: &[&str] = &["python", "-m", "rlmesh.serve", "--env", "sim:Env"];

    impl FakeRegistry {
        /// Register `reference` as a BuildKit index (when `index_digest` is
        /// set) or a bare manifest, with `config` as its image's config.
        fn add(
            &mut self,
            reference: &str,
            index_digest: Option<char>,
            image: char,
            config: String,
        ) {
            let repository = Reference::parse(reference).unwrap().repository;
            let image = digest(image);
            let manifest = match index_digest {
                Some(seed) => {
                    let mut index = buildkit_index(&image, ("linux", "amd64"));
                    index["digest"] = Value::from(digest(seed));
                    index
                }
                None => json!({"mediaType": "application/vnd.oci.image.manifest.v1+json",
                               "digest": image, "size": 481}),
            };
            self.insert(reference, &manifest);
            self.configs.insert(format!("{repository}@{image}"), config);
        }

        /// The media types of the descriptors the manifest at `digest`
        /// brings into an index: its children, or itself.
        fn media_types(&self, digest: &str) -> Vec<String> {
            let manifests = self.manifests.borrow();
            let Some(manifest) = manifests
                .values()
                .filter_map(|raw| serde_json::from_str::<Value>(raw).ok())
                .find(|manifest| manifest["digest"] == digest)
            else {
                return vec![OCI_INDEX.to_owned()];
            };
            let media_type = |value: &Value| value["mediaType"].as_str().unwrap_or("").to_owned();
            match manifest["manifests"].as_array() {
                Some(children) => children.iter().map(media_type).collect(),
                None => vec![media_type(&manifest)],
            }
        }

        /// The key `reference` is stored under: itself, or for
        /// `repository@digest`, a tag of that repository holding the digest.
        fn resolve(&self, reference: &str) -> String {
            let manifests = self.manifests.borrow();
            if manifests.contains_key(reference) {
                return reference.to_owned();
            }
            let Some((repository, digest)) = reference.split_once('@') else {
                return reference.to_owned();
            };
            manifests
                .iter()
                .find(|(key, view)| {
                    key.starts_with(&format!("{repository}:"))
                        && serde_json::from_str::<Value>(view)
                            .is_ok_and(|view| view["digest"] == digest)
                })
                .map_or_else(|| reference.to_owned(), |(key, _)| key.clone())
        }

        /// Store `raw` under `reference` as a registry would: at the sha256
        /// of its bytes, docker's view adding that digest and the size.
        fn store(&self, reference: &str, raw: &str) {
            let mut view: Value = serde_json::from_str(raw).unwrap();
            view["digest"] = Value::from(sha256(raw));
            view["size"] = Value::from(raw.len());
            self.insert(reference, &view);
            self.raws
                .borrow_mut()
                .insert(reference.to_owned(), raw.to_owned());
        }

        /// The digest `reference` points at.
        fn digest_of(&self, reference: &str) -> String {
            let view: Value = serde_json::from_str(&self.manifest(reference).unwrap()).unwrap();
            view["digest"].as_str().unwrap().to_owned()
        }

        fn insert(&self, reference: &str, manifest: &Value) {
            self.insert_raw(reference, &manifest.to_string());
        }

        fn insert_raw(&self, reference: &str, raw: &str) {
            self.raws.borrow_mut().remove(reference);
            self.manifests
                .borrow_mut()
                .insert(reference.to_owned(), raw.to_owned());
        }

        /// The non-dry-run `create` calls.
        fn pushes(&self) -> usize {
            self.created
                .borrow()
                .iter()
                .filter(|args| !args.contains(&"--dry-run".to_owned()))
                .count()
        }
    }

    fn three_variants() -> FakeRegistry {
        let mut registry = FakeRegistry::default();
        registry.add(
            "reg.example/ns/pi0:v3-cuda12",
            Some('1'),
            'c',
            oci_config(
                &["CUDA_VERSION=12.4.1", "NVIDIA_REQUIRE_CUDA=cuda>=12.4"],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "torch-cuda12",
                    "facets": {"framework": "torch", "accel": "cuda"},
                    "requires": {"accel.vendor": "nvidia", "accel.cuda": ">=12.4"}, "priority": 10}})),
            ),
        );
        registry.add(
            "reg.example/ns/pi0:v3-rocm6",
            Some('2'),
            'r',
            oci_config(
                &["ROCM_VERSION=6.2.1"],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "torch-rocm6",
                    "requires": {"accel.vendor": "amd", "accel.gfx": ["gfx942", "gfx90a"]}, "priority": 5}})),
            ),
        );
        registry.add(
            "reg.example/ns/pi0:v3-jax",
            None,
            'j',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "jax", "facets": {"framework": "jax"}},
                    "profiles": [{"key": "osmesa", "default": true, "envVars": {"MUJOCO_GL": "osmesa"}},
                                 {"key": "egl", "envVars": {"MUJOCO_GL": "egl"}, "gpu": {"count": 1}}]})),
            ),
        );
        registry
    }

    fn args(target: &str, sources: &[&str]) -> PublishArgs {
        PublishArgs {
            target: target.to_owned(),
            sources: sources.iter().map(|s| (*s).to_owned()).collect(),
            tags: Vec::new(),
            channel: None,
            index_package: None,
            dry_run: false,
            force: false,
        }
    }

    fn run(registry: &FakeRegistry, args: &PublishArgs) -> (Result<i32>, String) {
        let mut out = Vec::new();
        let result = publish_with(registry, args, &mut out, Style::for_terminal(false));
        (result, String::from_utf8(out).unwrap())
    }

    const SOURCES: &[&str] = &[
        "reg.example/ns/pi0:v3-cuda12",
        "reg.example/ns/pi0:v3-rocm6",
        "reg.example/ns/pi0:v3-jax",
    ];

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
    fn dry_run_prints_the_summary_and_the_index_without_pushing() {
        let registry = three_variants();
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("version.json");
        std::fs::write(
            &package,
            r#"{"checkpoints":[{"name":"base","uri":"hf://org/pi0","default":true}]}"#,
        )
        .unwrap();
        let mut args = args("reg.example/ns/pi0:v3", SOURCES);
        args.dry_run = true;
        args.channel = Some("latest".to_owned());
        args.tags = vec!["v3.0".to_owned()];
        args.index_package = Some(package);
        let (result, out) = run(&registry, &args);
        assert_eq!(result.unwrap(), 0, "{out}");

        let created = registry.created.borrow();
        assert_eq!(created.len(), 1);
        assert_eq!(
            created[0],
            [
                "--dry-run",
                "--tag",
                "reg.example/ns/pi0:v3",
                "--tag",
                "reg.example/ns/pi0:v3.0",
                "--tag",
                "reg.example/ns/pi0:latest",
                "--annotation",
                r#"index:dev.rlmesh.package={"checkpoints":[{"default":true,"name":"base","uri":"hf://org/pi0"}],"schemaVersion":1}"#,
                // Indexes stay whole (attestations ride along); bare manifests as is.
                &format!("reg.example/ns/pi0@{}", digest('1')),
                &format!("reg.example/ns/pi0@{}", digest('2')),
                &format!("reg.example/ns/pi0@{}", digest('j')),
            ]
        );
        assert!(
            out.starts_with("Dry run: would publish reg.example/ns/pi0:v3 (3 variants)"),
            "{out}"
        );
        let header = out.lines().find(|l| l.starts_with("VARIANT")).unwrap();
        for column in [
            "KIND", "PLATFORM", "PRIORITY", "FACETS", "REQUIRES", "ROWS", "IMAGE",
        ] {
            assert!(header.contains(column), "{header}");
        }
        let row = out.lines().find(|l| l.starts_with("torch-cuda12")).unwrap();
        for cell in [
            "model",
            "linux/amd64",
            "10",
            "accel=cuda, framework=torch",
            "accel.cuda>=12.4",
            // The version's default row: the first linux/amd64 child.
            "torch-cuda12*",
        ] {
            assert!(row.contains(cell), "{cell:?} not in {row:?}");
        }
        assert!(
            row.ends_with("reg.example/ns/pi0:v3-cuda12 (cccccccccccc)"),
            "{row}"
        );
        let row = out.lines().find(|l| l.starts_with("torch-rocm6")).unwrap();
        assert!(row.contains("accel.gfx in [gfx942,gfx90a]"), "{row}");
        assert!(row.contains("accel=rocm"), "inferred facet: {row}");
        let row = out.lines().find(|l| l.starts_with("jax")).unwrap();
        // No requires declared and no markers: a CPU image, rows per profile.
        assert!(row.contains("accel=cpu, framework=jax"), "{row}");
        assert!(row.contains("jax-osmesa, jax-egl"), "{row}");
        assert!(out.contains("* the version's default row"), "{out}");
        assert!(out.contains("(nothing pushed)"), "{out}");
        assert!(
            out.contains("application/vnd.oci.image.index.v1+json"),
            "{out}"
        );
        assert!(!out.contains("warn"), "{out}");
    }

    #[test]
    fn publish_pushes_and_reports_the_index_digest() {
        let registry = three_variants();
        let (result, out) = run(&registry, &args("reg.example/ns/pi0:v3", SOURCES));
        assert_eq!(result.unwrap(), 0, "{out}");
        let created = registry.created.borrow();
        // The index is assembled first (to check it), then pushed.
        assert_eq!(created.len(), 2);
        assert!(created[0].contains(&"--dry-run".to_owned()));
        assert!(!created[1].contains(&"--dry-run".to_owned()));
        assert!(!created[1].contains(&"--annotation".to_owned()));
        let published = registry.digest_of("reg.example/ns/pi0:v3");
        assert_eq!(
            published,
            sha256(&registry.raw("reg.example/ns/pi0:v3").unwrap())
        );
        assert!(
            out.contains(&format!("Published reg.example/ns/pi0@{published}")),
            "{out}"
        );
        assert!(out.contains("tagged reg.example/ns/pi0:v3"), "{out}");
    }

    #[test]
    fn every_problem_is_reported_before_anything_is_created() {
        let mut registry = three_variants();
        // Same key as the CUDA build.
        registry.add(
            "reg.example/ns/pi0:dup",
            None,
            'd',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "torch-cuda12"}})),
            ),
        );
        // An env among models.
        registry.add(
            "reg.example/ns/pi0:env",
            None,
            'e',
            oci_config(
                &[],
                ENV,
                Some(json!({"schemaVersion": 1, "variant": {"key": "sim"}})),
            ),
        );
        registry.add(
            "reg.example/ns/pi0:nolabel",
            None,
            'n',
            oci_config(&[], MODEL, None),
        );
        // An empty label (`--build-arg RLMESH_PACKAGE=` left unset) reads as none.
        let mut empty: Value = serde_json::from_str(&oci_config(&[], MODEL, None)).unwrap();
        empty["config"]["Labels"][PACKAGE_LABEL] = Value::from("");
        registry.add("reg.example/ns/pi0:empty", None, 'm', empty.to_string());
        registry.add(
            "reg.example/ns/pi0:nokey",
            None,
            'k',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "profiles": []})),
            ),
        );
        registry.add(
            "reg.example/ns/pi0:badreq",
            None,
            'b',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1,
                "variant": {"key": "bad", "requires": {"accel.cuda": ">12"}}})),
            ),
        );
        let windows = buildkit_index(&digest('w'), ("windows", "amd64"));
        registry.insert("reg.example/ns/pi0:win", &windows);
        let mut multi = buildkit_index(&digest('x'), ("linux", "amd64"));
        multi["manifests"].as_array_mut().unwrap().push(
            json!({"digest": digest('y'), "platform": {"os": "linux", "architecture": "arm64"}}),
        );
        registry.insert("reg.example/ns/pi0:multi", &multi);

        let sources = [
            "reg.example/ns/pi0:v3-cuda12",
            "reg.example/ns/pi0:dup",
            "reg.example/ns/pi0:env",
            "reg.example/ns/pi0:nolabel",
            "reg.example/ns/pi0:empty",
            "reg.example/ns/pi0:nokey",
            "reg.example/ns/pi0:badreq",
            "reg.example/ns/pi0:win",
            "reg.example/ns/pi0:multi",
            "reg.example/ns/pi0:missing",
        ];
        let mut args = args("reg.example/ns/pi0:v3", &sources);
        args.dry_run = true;
        let (result, out) = run(&registry, &args);
        let error = format!("{:#}", result.unwrap_err());
        assert!(registry.created.borrow().is_empty(), "nothing is created");
        for needle in [
            "cannot publish reg.example/ns/pi0:v3:",
            "reg.example/ns/pi0:win holds a windows/amd64 image; the platform runs linux images only",
            "reg.example/ns/pi0:multi is an index of 2 images (linux/amd64, linux/arm64)",
            "reg.example/ns/pi0:missing: not found",
            "reg.example/ns/pi0:nolabel: no dev.rlmesh.package label",
            "reg.example/ns/pi0:empty: no dev.rlmesh.package label",
            "reg.example/ns/pi0:nokey: dev.rlmesh.package declares no variant block",
            "reg.example/ns/pi0:badreq: variant: requires accel.cuda needs accel.vendor",
            "variant.key \"torch-cuda12\" is declared by reg.example/ns/pi0:v3-cuda12 and reg.example/ns/pi0:dup",
            "mixed kinds: env (reg.example/ns/pi0:env) vs model (",
        ] {
            assert!(error.contains(needle), "{needle:?} not in:\n{error}");
        }
        // The variants that did resolve are still summarized.
        assert!(out.lines().any(|l| l.starts_with("sim ")), "{out}");
    }

    #[test]
    fn non_linux_bare_manifests_and_unknown_kinds() {
        let mut registry = FakeRegistry::default();
        let mut config: Value = serde_json::from_str(&oci_config(&[], &[], None)).unwrap();
        config["os"] = Value::from("windows");
        registry.add("r/x:win", None, 'w', config.to_string());
        let child = resolve_source(&registry, "r/x:win").unwrap_err();
        assert!(
            format!("{child:#}").contains("is a windows/amd64 image"),
            "{child:#}"
        );

        registry.add(
            "r/x:custom",
            None,
            'c',
            oci_config(
                &[],
                &["./serve.sh"],
                Some(json!({"schemaVersion": 1, "variant": {"key": "a"}})),
            ),
        );
        let child = resolve_source(&registry, "r/x:custom").unwrap();
        let (variants, warnings, errors) = check_variants(vec![child]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(variants[0].kind, None);
        assert!(warnings[0].contains("kind unknown"), "{warnings:?}");
    }

    #[test]
    fn index_package_is_version_level_only() {
        assert_eq!(
            index_package(r#"{"checkpoints":[]}"#, "f").unwrap(),
            (r#"{"checkpoints":[],"schemaVersion":1}"#.to_owned(), vec![])
        );
        let (_, warnings) =
            index_package(r#"{"rev":3,"variant":{"key":"a"},"tags":[]}"#, "f").unwrap();
        assert_eq!(
            warnings,
            ["f: index annotation keys ignored (they belong on the child images): tags, variant"]
        );
        for (raw, needle) in [
            ("[]", "must be a JSON object"),
            ("{", "not valid JSON"),
            (r#"{"schemaVersion":2}"#, "schemaVersion 2"),
        ] {
            let error = format!("{:#}", index_package(raw, "f").unwrap_err());
            assert!(error.contains(needle), "{raw}: {error}");
        }
    }

    #[test]
    fn the_index_annotation_wins_over_children_and_is_warned_about() {
        let mut registry = FakeRegistry::default();
        registry.add(
            "r/x:a",
            None,
            'a',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "name": "old",
                "variant": {"key": "a"}})),
            ),
        );
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("version.json");
        std::fs::write(&package, r#"{"name":"pi0","tags":["gpu"]}"#).unwrap();
        let mut publish = args("r/x:v1", &["r/x:a"]);
        publish.dry_run = true;
        publish.index_package = Some(package);
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            out.contains("index annotation keys ignored (they belong on the child images): tags"),
            "{out}"
        );
        assert!(
            out.contains("r/x:a: sets name differently from the index annotation, which wins"),
            "{out}"
        );
    }

    #[test]
    fn rows_are_unique_across_the_version_and_one_child_must_be_schedulable() {
        let mut registry = FakeRegistry::default();
        // "cuda12" with profile "egl" derives "cuda12-egl", another child's key.
        registry.add(
            "r/x:a",
            None,
            'a',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "cuda12"},
                "profiles": [{"key": "egl", "default": true}]})),
            ),
        );
        registry.add(
            "r/x:b",
            None,
            'b',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "cuda12-egl"}})),
            ),
        );
        let (result, _) = run(&registry, &args("r/x:v1", &["r/x:a", "r/x:b"]));
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains("row key \"cuda12-egl\" is derived by r/x:a and r/x:b"),
            "{error}"
        );

        // Only an arm64 child: excluded, so nothing can be scheduled.
        let mut arm: Value = serde_json::from_str(&oci_config(
            &[],
            MODEL,
            Some(json!({"schemaVersion": 1, "variant": {"key": "arm"}})),
        ))
        .unwrap();
        arm["architecture"] = Value::from("arm64");
        registry.add("r/x:arm", None, 'r', arm.to_string());
        let (result, out) = run(&registry, &args("r/x:v1", &["r/x:arm"]));
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("no child is linux/amd64"), "{error}");
        assert!(
            out.contains("linux/amd64") && out.contains("status excluded"),
            "{out}"
        );
        assert!(out.contains("linux/arm64 (excluded)"), "{out}");
    }

    /// A `variant`-only package label for a child keyed `key`.
    fn keyed(key: &str) -> String {
        oci_config(
            &[],
            MODEL,
            Some(json!({"schemaVersion": 1, "variant": {"key": key}})),
        )
    }

    fn version_json(dir: &tempfile::TempDir, raw: &str) -> std::path::PathBuf {
        let path = dir.path().join("version.json");
        std::fs::write(&path, raw).unwrap();
        path
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
    fn a_lone_index_source_passes_its_annotation_through_checked() {
        let mut registry = FakeRegistry::default();
        registry.add(
            "r/x:a",
            Some('i'),
            'a',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "name": "pi0", "variant": {"key": "a"}})),
            ),
        );
        let annotate = |registry: &FakeRegistry, package: Value| {
            let mut index = buildkit_index(&digest('a'), ("linux", "amd64"));
            index["digest"] = Value::from(digest('i'));
            index["annotations"] = json!({PACKAGE_LABEL: package.to_string()});
            registry.insert("r/x:a", &index);
        };
        let mut publish = args("r/x:v1", &["r/x:a"]);
        publish.dry_run = true;

        annotate(
            &registry,
            json!({"schemaVersion": 2, "checkpoints": [{"name": "stale"}]}),
        );
        let (result, _) = run(&registry, &publish);
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(
                "r/x:a: the dev.rlmesh.package annotation on its index schemaVersion 2 is not \
                 the supported version 1; it would be kept on the published version"
            ),
            "{error}"
        );
        assert!(registry.created.borrow().is_empty());

        // Valid, but carrying child-level blocks and a stale name: warned
        // about exactly as --index-package would be.
        annotate(
            &registry,
            json!({"schemaVersion": 1, "name": "old", "variant": {"key": "a"}, "profiles": []}),
        );
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        for needle in [
            "r/x:a: the dev.rlmesh.package annotation on its index is kept on the published \
             version",
            "index annotation keys ignored (they belong on the child images): profiles, variant",
            "r/x:a: sets name differently from the index annotation, which wins",
        ] {
            assert!(out.contains(needle), "{needle:?} not in:\n{out}");
        }
        // Docker keeps it; it is not passed again.
        assert!(
            !registry.created.borrow()[0].contains(&"--annotation".to_owned()),
            "{:?}",
            registry.created.borrow()
        );

        // With a second source, docker assembles a new index without it.
        registry.add("r/x:b", None, 'b', keyed("b"));
        publish.sources.push("r/x:b".to_owned());
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            out.contains(
                "r/x:a: the dev.rlmesh.package annotation on its index is not carried into the \
                 version"
            ),
            "{out}"
        );
        assert!(!out.contains("is kept"), "{out}");
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

    /// The `Tags:` line of `tag` in a publish's output.
    fn tag_line<'a>(out: &'a str, tag: &str) -> &'a str {
        out.lines()
            .find(|line| line.trim_start().starts_with(&format!("{tag} ")))
            .unwrap_or_else(|| panic!("no tag line for {tag} in:\n{out}"))
    }

    #[test]
    fn version_tags_do_not_move_without_force_and_channels_say_where_from() {
        let registry = three_variants();
        let mut publish = args("reg.example/ns/pi0:v3", SOURCES);
        publish.tags = vec!["v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3").ends_with(" new"),
            "{out}"
        );
        // TARGET is pushed alone, then the other tags point at its digest.
        let v3 = registry.digest_of("reg.example/ns/pi0:v3");
        for tag in ["reg.example/ns/pi0:v3.0", "reg.example/ns/pi0:latest"] {
            assert_eq!(registry.digest_of(tag), v3, "{tag}");
        }
        assert_eq!(
            registry.created.borrow()[2],
            [
                "--tag",
                "reg.example/ns/pi0:v3.0",
                "--tag",
                "reg.example/ns/pi0:latest",
                &format!("reg.example/ns/pi0@{v3}"),
            ]
        );
        assert_eq!(registry.pushes(), 2);
        // Publishing the same version again changes nothing and is allowed;
        // no tag is pushed again.
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3")
                .ends_with(&format!(" unchanged ({v3}), not pushed again")),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "Already published reg.example/ns/pi0@{v3}; every tag holds this index"
            )),
            "{out}"
        );
        assert_eq!(registry.pushes(), 2);

        // Other children under the same version tag: the dry run says what
        // would be refused, and the real run refuses it.
        let mut other = args("reg.example/ns/pi0:v3", &SOURCES[..2]);
        other.channel = Some("latest".to_owned());
        other.dry_run = true;
        let (result, out) = run(&registry, &other);
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(&format!(
                "reg.example/ns/pi0:v3 already points at {v3}, a different index; a version tag \
                 does not move without --force"
            )),
            "{error}"
        );
        assert!(
            out.contains("(nothing pushed)"),
            "the index is shown: {out}"
        );
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3").ends_with(&format!("refused: points at {v3}")),
            "{out}"
        );
        // The channel is not refused: moving is what it is for.
        assert!(
            tag_line(&out, "reg.example/ns/pi0:latest")
                .ends_with(&format!("moves from {v3} (channel)")),
            "{out}"
        );
        other.dry_run = false;
        assert!(run(&registry, &other).0.is_err());
        assert_eq!(registry.pushes(), 2, "nothing is pushed");
        other.force = true;
        let (result, out) = run(&registry, &other);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3")
                .ends_with(&format!("moves from {v3} (--force)")),
            "{out}"
        );
        assert_eq!(registry.pushes(), 4);
        assert_eq!(
            registry.digest_of("reg.example/ns/pi0:latest"),
            registry.digest_of("reg.example/ns/pi0:v3")
        );

        // A new version moves only the channel, without --force.
        let mut next = args("reg.example/ns/pi0:v4", SOURCES);
        next.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &next);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v4").ends_with(" new"),
            "{out}"
        );
        assert!(
            tag_line(&out, "reg.example/ns/pi0:latest").contains("moves from"),
            "{out}"
        );

        // A channel that is also the version tag is protected like one.
        let mut same = args("reg.example/ns/pi0:latest", &SOURCES[..1]);
        same.channel = Some("latest".to_owned());
        let error = format!("{:#}", run(&registry, &same).0.unwrap_err());
        assert!(error.contains("does not move without --force"), "{error}");
    }

    /// The messages `docker buildx imagetools inspect` (0.37) prints, each
    /// as the wrapper reports it.
    fn inspect_error(message: &str) -> String {
        format!("docker buildx imagetools inspect failed: ERROR: {message}")
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
    fn an_unreadable_version_tag_is_refused_without_force() {
        let registry = three_variants();
        let tag = "reg.example/ns/pi0:v5";
        for (canned, reason) in [
            (
                "ERROR: unexpected status from HEAD request to \
                 https://reg.example/v2/ns/pi0/manifests/v5: 401 Unauthorized",
                "401 Unauthorized",
            ),
            (
                "ERROR: failed to do request: Head \"https://reg.example/v2/ns/pi0/manifests/v5\": \
                 dial tcp: i/o timeout",
                "i/o timeout",
            ),
            (
                "ERROR: failed to authorize: failed to fetch oauth token: unexpected status from \
                 GET request to https://reg.example/token: 404 Not Found",
                "token: 404 Not Found",
            ),
            ("<html>bad gateway</html>", "is not JSON"),
        ] {
            registry.insert_raw(tag, canned);
            let mut publish = args(tag, SOURCES);
            publish.dry_run = true;
            let (result, out) = run(&registry, &publish);
            let error = format!("{:#}", result.unwrap_err());
            assert!(
                error.contains(&format!("{tag} could not be read (")) && error.contains(reason),
                "{reason:?} not in:\n{error}"
            );
            assert!(error.contains("pass --force"), "{error}");
            assert!(
                tag_line(&out, tag).ends_with("refused: could not be read"),
                "{out}"
            );
            publish.dry_run = false;
            assert!(run(&registry, &publish).0.is_err());
            assert_eq!(registry.pushes(), 0, "nothing is pushed");
        }
        // A real absence is new.
        registry.manifests.borrow_mut().remove(tag);
        let mut publish = args(tag, SOURCES);
        publish.dry_run = true;
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(tag_line(&out, tag).ends_with(" new"), "{out}");

        // --force pushes over it, and the channel is overwritten regardless;
        // both say why.
        registry.insert_raw(tag, "ERROR: unexpected status: 401 Unauthorized");
        let channel = "reg.example/ns/pi0:latest";
        registry.insert_raw(channel, "ERROR: unexpected status: 503 Service Unavailable");
        publish.dry_run = false;
        publish.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &publish);
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(&format!("{tag} could not be read")),
            "{error}"
        );
        assert!(
            !error.contains(channel),
            "the channel is not refused: {error}"
        );
        publish.force = true;
        let (result, out2) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out2}");
        assert!(
            tag_line(&out2, tag).ends_with(
                "unreadable, overwritten (--force): ERROR: unexpected status: 401 Unauthorized"
            ),
            "{out2}"
        );
        for out in [&out, &out2] {
            assert!(
                tag_line(out, channel).ends_with(
                    "unreadable, overwritten (channel): ERROR: unexpected status: 503 Service \
                     Unavailable"
                ),
                "{out}"
            );
        }
        assert_eq!(registry.pushes(), 2);
        assert_eq!(registry.digest_of(channel), registry.digest_of(tag));
    }

    #[test]
    fn an_existing_index_is_unchanged_only_when_every_field_matches() {
        let registry = three_variants();
        let target = "reg.example/ns/pi0:v3";
        let (result, out) = run(&registry, &args(target, SOURCES));
        assert_eq!(result.unwrap(), 0, "{out}");
        let stored: Value = serde_json::from_str(&registry.raw(target).unwrap()).unwrap();

        let assembled = registry.digest_of(target);

        // The same index, serialized differently (compact, other key order),
        // so at another digest: unchanged, and not pushed again, so its
        // digest stays put.
        let reordered = format!(
            "{{\"manifests\":{},\"schemaVersion\":2,\"mediaType\":\"{OCI_INDEX}\"}}",
            stored["manifests"]
        );
        let version = sha256(&reordered);
        assert_ne!(version, assembled);
        // The channel already holds the same index, as docker assembles it.
        registry.store("reg.example/ns/pi0:latest", &registry.raw(target).unwrap());
        registry.store(target, &reordered);
        let mut publish = args(target, SOURCES);
        publish.tags = vec!["v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(tag_line(&out, target).contains("unchanged"), "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:latest").ends_with(&format!(
                "same index at {assembled}, re-pointed at {version} (TARGET's digest)"
            )),
            "{out}"
        );
        // Only the other tags are pushed, pointed at TARGET's stored digest
        // rather than assembled again, so every tag is the same manifest.
        let created = registry.created.borrow().last().unwrap().clone();
        assert_eq!(
            created,
            [
                "--tag",
                "reg.example/ns/pi0:v3.0",
                "--tag",
                "reg.example/ns/pi0:latest",
                &format!("reg.example/ns/pi0@{version}"),
            ]
        );
        for tag in [
            target,
            "reg.example/ns/pi0:v3.0",
            "reg.example/ns/pi0:latest",
        ] {
            assert_eq!(registry.digest_of(tag), version, "{tag}");
            assert_eq!(registry.raw(tag).unwrap(), reordered, "{tag}");
        }
        assert!(
            out.contains(&format!("Published reg.example/ns/pi0@{version}")),
            "{out}"
        );
        assert!(
            out.contains(&format!("unchanged {target} (already {version})")),
            "{out}"
        );
        assert!(out.contains("tagged reg.example/ns/pi0:v3.0"), "{out}");
        assert!(
            out.contains("re-pointed reg.example/ns/pi0:latest"),
            "{out}"
        );
        // Now every tag holds the version's digest: nothing to push.
        let pushes = registry.pushes();
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(out.contains("Already published"), "{out}");
        assert_eq!(registry.pushes(), pushes);

        // An index carrying a subject or an artifactType the replacement would
        // drop is a different index, though docker's .Manifest view hides both.
        for (field, value) in [
            (
                "subject",
                json!({"mediaType": "application/vnd.oci.image.manifest.v1+json",
                       "digest": digest('s'), "size": 7}),
            ),
            ("artifactType", json!("application/vnd.example.sbom")),
        ] {
            let mut carrying = stored.clone();
            carrying[field] = value;
            registry
                .raws
                .borrow_mut()
                .insert(target.to_owned(), carrying.to_string());
            let error = format!(
                "{:#}",
                run(&registry, &args(target, SOURCES)).0.unwrap_err()
            );
            assert!(
                error.contains(&format!(
                    "{target} already points at {version}, a different index"
                )),
                "{field}: {error}"
            );
        }
        // So is one whose manifests differ only in a descriptor's size.
        let mut resized = stored.clone();
        resized["manifests"][0]["size"] = Value::from(1);
        registry
            .raws
            .borrow_mut()
            .insert(target.to_owned(), resized.to_string());
        assert!(run(&registry, &args(target, SOURCES)).0.is_err());
    }

    #[test]
    fn a_failed_check_after_the_push_moves_only_target_and_says_how_to_recover() {
        let registry = three_variants();
        let dir = tempfile::tempdir().unwrap();
        let (target, extra, channel) = (
            "reg.example/ns/pi0:v3",
            "reg.example/ns/pi0:v3.0",
            "reg.example/ns/pi0:latest",
        );
        let mut publish = args(target, SOURCES);
        publish.tags = vec!["v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        publish.index_package = Some(version_json(&dir, r#"{"checkpoints":[]}"#));

        // The registry loses the annotation on the way in.
        registry.mangle.set(Some(|index| {
            index.as_object_mut().unwrap().remove("annotations");
        }));
        let (result, _) = run(&registry, &publish);
        let error = format!("{:#}", result.unwrap_err());
        let stored = registry.digest_of(target);
        for needle in [
            format!(
                "pushed {target} ({stored}), but the registry holds an index without the \
                 dev.rlmesh.package annotation --index-package sets; no other tag was moved"
            ),
            format!(
                "{target} did not exist before this run, so it holds only this push: publish \
                 again with --force once the cause is fixed"
            ),
        ] {
            assert!(error.contains(&needle), "{needle:?} not in:\n{error}");
        }
        assert_eq!(registry.pushes(), 1, "only TARGET was pushed");
        for tag in [extra, channel] {
            assert!(registry.manifest(tag).is_err(), "{tag} did not move");
        }

        // A retry without --force is refused at TARGET, as the error said;
        // with it, the version is published and every tag is its digest.
        registry.mangle.set(None);
        let error = format!("{:#}", run(&registry, &publish).0.unwrap_err());
        assert!(
            error.contains(&format!("{target} already points at {stored}")),
            "{error}"
        );
        assert_eq!(registry.pushes(), 1);
        publish.force = true;
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        let version = registry.digest_of(target);
        assert_ne!(version, stored);
        for tag in [extra, channel] {
            assert_eq!(registry.digest_of(tag), version, "{tag}");
        }
        let pushed: Value = serde_json::from_str(&registry.raw(target).unwrap()).unwrap();
        assert!(pushed["annotations"][PACKAGE_LABEL].is_string(), "{pushed}");

        // Forcing over a published version: the error names the digest it
        // held and how to put it back, and the channel stays on it.
        registry.mangle.set(Some(|index| {
            index["manifests"].as_array_mut().unwrap().reverse();
        }));
        let mut over = args(target, &SOURCES[..2]);
        over.channel = Some("latest".to_owned());
        over.force = true;
        let error = format!("{:#}", run(&registry, &over).0.unwrap_err());
        for needle in [
            "but the registry holds an index other than the one assembled".to_owned(),
            format!(
                "{target} pointed at {version} before this run; restore it with `docker buildx \
                 imagetools create --tag {target} reg.example/ns/pi0@{version}`"
            ),
        ] {
            assert!(error.contains(&needle), "{needle:?} not in:\n{error}");
        }
        assert_eq!(registry.digest_of(channel), version);
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

    #[test]
    fn the_assembled_index_must_keep_the_annotation() {
        assert_eq!(
            annotation_dropped(&json!({"mediaType": OCI_INDEX,
                "annotations": {PACKAGE_LABEL: "{}"}})),
            None
        );
        let list = annotation_dropped(
            &json!({"mediaType": "application/vnd.docker.distribution.manifest.list.v2+json"}),
        )
        .unwrap();
        assert!(
            list.contains("not an OCI image index, so the --index-package annotation is dropped"),
            "{list}"
        );
        let bare = annotation_dropped(&json!({"mediaType": OCI_INDEX})).unwrap();
        assert!(
            bare.contains("without the dev.rlmesh.package annotation"),
            "{bare}"
        );
    }

    #[test]
    fn target_must_be_a_tag_and_extra_tags_join_its_repository() {
        let mut publish = args("localhost:5000/ns/pi0:v3", &["s"]);
        publish.tags = vec!["v3".to_owned(), "v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        let (_, tags) = target_tags(&publish).unwrap();
        assert_eq!(
            tags,
            [
                "localhost:5000/ns/pi0:v3",
                "localhost:5000/ns/pi0:v3.0",
                "localhost:5000/ns/pi0:latest"
            ]
        );
        assert!(target_tags(&args("ns/pi0", &["s"])).is_err());
        assert!(target_tags(&args(&format!("ns/pi0@{}", digest('a')), &["s"])).is_err());
        publish.channel = Some("not/a/tag".to_owned());
        assert!(target_tags(&publish).is_err());
    }
}
