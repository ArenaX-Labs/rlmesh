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
//! publish cannot swap a child.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::cli::PublishArgs;
use crate::image_check::{
    self, CheckReport, DESCRIBE_LABEL, ImageConfig, Kind, PACKAGE_LABEL, parse_serve_command,
};
use crate::package::{self, ComputeBlocks};
use crate::render::Style;

/// The annotation BuildKit puts on an attestation manifest inside an index.
const REFERENCE_TYPE: &str = "vnd.docker.reference.type";
const ATTESTATION: &str = "attestation-manifest";

/// The registry operations publish needs, as `docker buildx imagetools`
/// provides them; a test hands in canned responses instead.
pub(crate) trait Imagetools {
    /// The JSON `imagetools inspect REF --format '{{json .Manifest}}'` prints:
    /// an index (with `manifests`) or a bare manifest descriptor.
    fn manifest(&self, reference: &str) -> Result<String>;
    /// The OCI image config of a single-image reference
    /// (`--format '{{json .Image}}'`).
    fn config(&self, reference: &str) -> Result<String>;
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
    /// `os/architecture[/variant]`.
    pub platform: String,
    pub config: ImageConfig,
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
    let pinned = format!("{}@{digest}", reference.repository);
    let (image_digest, platform) = match manifest.get("manifests").and_then(Value::as_array) {
        None => (digest, None),
        Some(entries) => {
            let mut images = Vec::new();
            for entry in entries {
                let attestation = entry
                    .pointer(&format!("/annotations/{REFERENCE_TYPE}"))
                    .and_then(Value::as_str)
                    == Some(ATTESTATION);
                if attestation {
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
                let digest = entry
                    .get("digest")
                    .and_then(Value::as_str)
                    .with_context(|| format!("an entry in the index of {source} has no digest"))?;
                images.push((digest.to_owned(), platform));
            }
            match images.len() {
                1 => {
                    let (digest, platform) = images.swap_remove(0);
                    (digest, Some(platform))
                }
                0 => bail!("{source} is an index with no image in it"),
                n => bail!(
                    "{source} is an index of {n} images ({}); each source must be one image, \
                     since each child of the published index is one variant",
                    images
                        .iter()
                        .map(|(_, platform)| platform.as_str())
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
    let platform = platform.unwrap_or_else(|| {
        if config.os.is_empty() {
            String::new()
        } else {
            format!("{}/{}", config.os, config.architecture)
        }
    });
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
    Ok(Child {
        source: source.to_owned(),
        pinned,
        image_digest,
        platform,
        config,
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

/// Read and check each child's package label, and the set as a whole: every
/// child needs a label with a `variant.key`, keys are unique, kinds agree.
/// Returns the variants, the warnings worth printing, and every error found
/// (all of them, so one run reports everything to fix).
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
        let report: CheckReport = package::check_compute_blocks(&package, &child.config);
        errors.extend(report.failed.iter().map(|m| format!("{source}: {m}")));
        warnings.extend(
            report
                .warnings
                .iter()
                .filter(|m| !m.starts_with("variant: no key"))
                .map(|m| format!("{source}: {m}")),
        );
        let (blocks, _) = package::parse_compute_blocks(&package);
        let Some(key) = blocks.variant.as_ref().and_then(|v| v.key.clone()) else {
            errors.push(format!(
                "{source}: {PACKAGE_LABEL} declares no variant.key; each child of a version \
                 is addressed by its key"
            ));
            continue;
        };
        let kind = image_kind(&child.config);
        if kind.is_none() {
            warnings.push(format!(
                "{source}: kind unknown (no {DESCRIBE_LABEL} label and the command does not \
                 run -m rlmesh.serve); the platform's probe decides it"
            ));
        }
        if !child.config.architecture.is_empty() && child.config.architecture != "amd64" {
            warnings.push(format!(
                "{source}: {} image; the platform runs linux/amd64 today",
                child.platform
            ));
        }
        variants.push(VariantImage {
            child,
            key,
            kind,
            blocks,
        });
    }
    let mut by_key: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for variant in &variants {
        by_key
            .entry(&variant.key)
            .or_default()
            .push(&variant.child.source);
    }
    for (key, sources) in by_key.iter().filter(|(_, sources)| sources.len() > 1) {
        errors.push(format!(
            "variant.key {key:?} is declared by {}; keys must be unique within a version",
            sources.join(" and ")
        ));
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
    (variants, warnings, errors)
}

/// Read `--index-package`: a JSON object of version-level package data.
/// `schemaVersion` defaults to 1; `variant` and `profiles` describe one
/// image and belong in that image's label. Returns compact JSON.
pub(crate) fn index_package(raw: &str, origin: &str) -> Result<String> {
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
    for field in ["variant", "profiles"] {
        if package.contains_key(field) {
            bail!(
                "{origin} declares {field:?}, which describes one image; put it in that image's \
                 {PACKAGE_LABEL} label instead"
            );
        }
    }
    Ok(serde_json::to_string(&value)?)
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

pub(crate) fn publish_with(
    tools: &impl Imagetools,
    args: &PublishArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let (target, tags) = target_tags(args)?;
    let annotation = match &args.index_package {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some(index_package(&raw, &path.display().to_string())?)
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
    let (variants, warnings, check_errors) = check_variants(children);
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
    if !errors.is_empty() {
        bail!(
            "cannot publish {}:\n{}",
            args.target,
            errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let create = create_args(&tags, annotation.as_deref(), &variants, args.dry_run);
    let output = tools.create(&create)?;
    writeln!(stdout)?;
    if args.dry_run {
        writeln!(
            stdout,
            "{}",
            style.muted(&format!("Index for {} (nothing pushed):", tags.join(", ")))
        )?;
        writeln!(stdout, "{}", output.trim())?;
        return Ok(0);
    }
    let pushed: Value =
        serde_json::from_str(&tools.manifest(&tags[0])?).context("parsing the published index")?;
    let digest = pushed
        .get("digest")
        .and_then(Value::as_str)
        .unwrap_or("(digest unknown)");
    writeln!(
        stdout,
        "{}",
        style.success(&format!("Published {}@{digest}", target.repository))
    )?;
    for tag in &tags {
        writeln!(stdout, "  {} {tag}", style.muted("tagged"))?;
    }
    Ok(0)
}

fn write_table(stdout: &mut impl Write, style: Style, variants: &[VariantImage]) -> Result<()> {
    let headers = [
        "VARIANT", "KIND", "PLATFORM", "PRIORITY", "FACETS", "REQUIRES", "PROFILES", "IMAGE",
    ];
    let rows: Vec<[String; 8]> = variants
        .iter()
        .map(|variant| {
            let detail = variant.blocks.variant.clone().unwrap_or_default();
            let or_dash = |value: String| {
                if value.is_empty() {
                    "-".to_owned()
                } else {
                    value
                }
            };
            [
                variant.key.clone(),
                variant.kind.map_or("?", Kind::name).to_owned(),
                variant.child.platform.clone(),
                detail
                    .priority
                    .map_or_else(|| "-".to_owned(), |p| p.to_string()),
                or_dash(package::format_facets(&detail.facets)),
                or_dash(package::format_requires(&detail.requires)),
                or_dash(package::profile_summary(&variant.blocks.profiles)),
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
    Ok(())
}

fn short_digest(digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    format!("({})", &hex[..hex.len().min(12)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use anyhow::anyhow;
    use serde_json::json;

    /// Canned registry contents, in the shapes `docker buildx imagetools
    /// inspect` prints for `.Manifest` and `.Image`.
    #[derive(Default)]
    struct FakeRegistry {
        manifests: BTreeMap<String, String>,
        configs: BTreeMap<String, String>,
        created: RefCell<Vec<Vec<String>>>,
    }

    impl Imagetools for FakeRegistry {
        fn manifest(&self, reference: &str) -> Result<String> {
            self.manifests
                .get(reference)
                .cloned()
                .ok_or_else(|| anyhow!("ERROR: {reference}: not found"))
        }

        fn config(&self, reference: &str) -> Result<String> {
            self.configs
                .get(reference)
                .cloned()
                .ok_or_else(|| anyhow!("ERROR: {reference}: not found"))
        }

        fn create(&self, args: &[String]) -> Result<String> {
            self.created.borrow_mut().push(args.to_vec());
            Ok(if args.iter().any(|a| a == "--dry-run") {
                json!({"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
                       "manifests": []})
                .to_string()
            } else {
                String::new()
            })
        }
    }

    fn digest(seed: char) -> String {
        format!("sha256:{}", seed.to_string().repeat(64))
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
            self.manifests
                .insert(reference.to_owned(), manifest.to_string());
            self.configs.insert(format!("{repository}@{image}"), config);
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
                    "facets": {"framework": "torch", "accel": "nvidia"},
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
        registry.manifests.insert(
            "reg.example/ns/pi0:v3".to_owned(),
            json!({"digest": digest('p')}).to_string(),
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
            "KIND", "PLATFORM", "PRIORITY", "FACETS", "REQUIRES", "PROFILES", "IMAGE",
        ] {
            assert!(header.contains(column), "{header}");
        }
        let row = out.lines().find(|l| l.starts_with("torch-cuda12")).unwrap();
        for cell in [
            "model",
            "linux/amd64",
            "10",
            "accel=nvidia, framework=torch",
            "accel.cuda>=12.4",
        ] {
            assert!(row.contains(cell), "{cell:?} not in {row:?}");
        }
        assert!(
            row.ends_with("reg.example/ns/pi0:v3-cuda12 (cccccccccccc)"),
            "{row}"
        );
        let row = out.lines().find(|l| l.starts_with("torch-rocm6")).unwrap();
        assert!(row.contains("accel.gfx in [gfx942,gfx90a]"), "{row}");
        let row = out.lines().find(|l| l.starts_with("jax")).unwrap();
        assert!(row.contains("osmesa (default), egl"), "{row}");
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
        assert!(!created[0].contains(&"--dry-run".to_owned()));
        assert!(!created[0].contains(&"--annotation".to_owned()));
        assert!(
            out.contains(&format!("Published reg.example/ns/pi0@{}", digest('p'))),
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
        registry
            .manifests
            .insert("reg.example/ns/pi0:win".to_owned(), windows.to_string());
        let mut multi = buildkit_index(&digest('x'), ("linux", "amd64"));
        multi["manifests"].as_array_mut().unwrap().push(
            json!({"digest": digest('y'), "platform": {"os": "linux", "architecture": "arm64"}}),
        );
        registry
            .manifests
            .insert("reg.example/ns/pi0:multi".to_owned(), multi.to_string());

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
            "reg.example/ns/pi0:nokey: dev.rlmesh.package declares no variant.key",
            "reg.example/ns/pi0:badreq: variant: requires accel.cuda comparator",
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
            r#"{"checkpoints":[],"schemaVersion":1}"#
        );
        assert!(index_package(r#"{"schemaVersion":1}"#, "f").is_ok());
        for (raw, needle) in [
            ("[]", "must be a JSON object"),
            ("{", "not valid JSON"),
            (r#"{"schemaVersion":2}"#, "schemaVersion 2"),
            (r#"{"variant":{"key":"a"}}"#, "declares \"variant\""),
            (r#"{"profiles":[]}"#, "declares \"profiles\""),
        ] {
            let error = format!("{:#}", index_package(raw, "f").unwrap_err());
            assert!(error.contains(needle), "{raw}: {error}");
        }
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
