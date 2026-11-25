//! Pure checks over a built image's OCI config: the serve command and the
//! address it binds, the port it exposes, its platform, its rlmesh labels
//! (including the package label's `variant` / `profiles` blocks, see
//! [`crate::variant`]), and whether an [`EditionTarget`] can run it: the
//! session wire it speaks and the workflow edition it declares, against the
//! rlmesh this CLI was built with or a platform's editions named on the command
//! line.

mod serve;

pub use serve::{ServeCommand, check_serve_command, parse_serve_command, shell_words};

use std::collections::BTreeMap;

use rlmesh_proto::{Edition, PROTOCOL_GENERATION};
use rlmesh_session::wire::version::{WIRE_REVISION, wire_revision_name};
use serde::{Deserialize, Serialize};

/// OCI config label carrying the describe envelope (`python -m
/// rlmesh._describe`).
pub const DESCRIBE_LABEL: &str = "dev.rlmesh.describe";
/// OCI config label carrying packaged checkpoint declarations.
pub const PACKAGE_LABEL: &str = "dev.rlmesh.package";
/// The port `rlmesh.serve` hosts its session on when `RLMESH_ADDRESS` names
/// none: the one port every participant dials.
pub const SERVE_PORT: u16 = rlmesh_session::address::DEFAULT_PORT;
/// The env var `rlmesh.serve` binds.
pub const ADDRESS_ENV: &str = "RLMESH_ADDRESS";
/// The env var that declares a served peer's workflow edition, above the
/// class declaration; empty declares none.
pub const WORKFLOW_EDITION_ENV: &str = "RLMESH_WORKFLOW_EDITION";

/// What a check found, in the buckets the CLI prints.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckReport {
    #[serde(default)]
    pub failed: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub not_checked: Vec<String>,
    #[serde(default)]
    pub passed: Vec<String>,
}

impl CheckReport {
    /// Whether nothing failed (warnings and unchecked items do not block a push).
    pub fn ok(&self) -> bool {
        self.failed.is_empty()
    }

    /// Append another report's findings, bucket by bucket.
    pub fn extend(&mut self, other: CheckReport) {
        self.failed.extend(other.failed);
        self.warnings.extend(other.warnings);
        self.not_checked.extend(other.not_checked);
        self.passed.extend(other.passed);
    }
}

/// What an image serves, as the describe envelope names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Env,
    Model,
}

impl Kind {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "env" => Some(Self::Env),
            "model" => Some(Self::Model),
            _ => None,
        }
    }

    /// The describe envelope's spelling: `env` or `model`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Model => "model",
        }
    }
}

/// The parts of an OCI image config the checks read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageConfig {
    /// `Os`, e.g. `linux`; empty when unknown.
    pub os: String,
    /// `Architecture`, e.g. `amd64`; empty when unknown.
    pub architecture: String,
    /// `Config.Entrypoint` (empty when null).
    pub entrypoint: Vec<String>,
    /// `Config.Cmd` (empty when null).
    pub cmd: Vec<String>,
    /// `Config.Env` as `KEY=value` strings.
    pub env: Vec<String>,
    /// `Config.ExposedPorts` keys, e.g. `50051/tcp`.
    pub exposed_ports: Vec<String>,
    /// `Config.Labels`.
    pub labels: BTreeMap<String, String>,
}

impl ImageConfig {
    /// Parse the JSON `docker image inspect <ref>` prints (a one-element array; a
    /// bare object is accepted too).
    pub fn from_docker_inspect(json: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|err| format!("docker inspect JSON: {err}"))?;
        let entry = match value {
            serde_json::Value::Array(mut entries) => {
                if entries.is_empty() {
                    return Err("docker inspect returned no image".to_owned());
                }
                entries.swap_remove(0)
            }
            other => other,
        };
        let inspect: Inspect =
            serde_json::from_value(entry).map_err(|err| format!("docker inspect JSON: {err}"))?;
        Ok(Self::from_parts(
            inspect.os,
            inspect.architecture,
            inspect.config,
        ))
    }

    /// Parse an OCI image config as a registry serves it (what `docker buildx
    /// imagetools inspect REF --format '{{json .Image}}'` prints): lowercase `os` /
    /// `architecture` / `config`, with the `config` keys spelled as in `docker
    /// image inspect`.
    pub fn from_oci_config(json: &str) -> Result<Self, String> {
        let oci: OciConfig =
            serde_json::from_str(json).map_err(|err| format!("OCI image config JSON: {err}"))?;
        Ok(Self::from_parts(oci.os, oci.architecture, oci.config))
    }

    fn from_parts(
        os: Option<String>,
        architecture: Option<String>,
        config: Option<InspectConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();
        Self {
            os: os.unwrap_or_default(),
            architecture: architecture.unwrap_or_default(),
            entrypoint: config.entrypoint.unwrap_or_default(),
            cmd: config.cmd.unwrap_or_default(),
            env: config.env.unwrap_or_default(),
            exposed_ports: config
                .exposed_ports
                .unwrap_or_default()
                .into_keys()
                .collect(),
            labels: config.labels.unwrap_or_default(),
        }
    }

    /// The value of the `KEY=value` entry in `env`, when set.
    pub fn env_value(&self, key: &str) -> Option<&str> {
        self.env
            .iter()
            .find_map(|entry| entry.strip_prefix(key)?.strip_prefix('='))
    }

    /// Whether `exposed_ports` lists `port` on any protocol.
    pub fn exposes_port(&self, port: u16) -> bool {
        self.exposed_ports.iter().any(|entry| {
            let (number, protocol) = entry.split_once('/').unwrap_or((entry, "tcp"));
            protocol == "tcp" && number.parse() == Ok(port)
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Inspect {
    #[serde(default)]
    os: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    config: Option<InspectConfig>,
}

#[derive(Deserialize)]
struct OciConfig {
    #[serde(default)]
    os: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    config: Option<InspectConfig>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
struct InspectConfig {
    #[serde(default)]
    entrypoint: Option<Vec<String>>,
    #[serde(default)]
    cmd: Option<Vec<String>>,
    #[serde(default)]
    env: Option<Vec<String>>,
    #[serde(default)]
    exposed_ports: Option<BTreeMap<String, serde_json::Value>>,
    #[serde(default)]
    labels: Option<BTreeMap<String, String>>,
}

/// Check the image's OS/architecture against the one RLMesh deploys images to
/// (linux/amd64).
pub fn check_platform(config: &ImageConfig) -> CheckReport {
    let mut report = CheckReport::default();
    if config.os.is_empty() || config.architecture.is_empty() {
        report
            .not_checked
            .push("platform: the image config names no os/architecture".to_owned());
    } else if config.os != "linux" || config.architecture != "amd64" {
        report.failed.push(format!(
            "platform: image is {}/{}; RLMesh deploys linux/amd64 images (build with \
             --platform linux/amd64)",
            config.os, config.architecture
        ));
    } else {
        report.passed.push("platform: linux/amd64".to_owned());
    }
    report
}

/// What a describe envelope's `runtime` block says about the image's rlmesh.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescribeRuntime {
    /// The `rlmesh` package the image was described with.
    #[serde(default)]
    pub package_version: Option<String>,
    /// The wire generation the image speaks (`rlmesh-realtime-v1`).
    #[serde(default)]
    pub protocol_generation: Option<String>,
    /// The session wire revision the image's rlmesh speaks
    /// (`rlmesh_session::wire::version::WIRE_REVISION`): builds of one
    /// generation talk exactly when their revisions match.
    #[serde(default)]
    pub session_wire_revision: Option<u32>,
    /// The workflow edition the image declares, resolved when it was described.
    #[serde(default)]
    pub workflow_edition: Option<String>,
    /// The OS the envelope was generated on (`linux`); a label baked on a Mac
    /// carries `macos` and fails validation.
    #[serde(default)]
    pub os: Option<String>,
    /// The architecture the envelope was generated on (`x86_64` / `amd64`).
    #[serde(default)]
    pub arch: Option<String>,
}

/// What a `dev.rlmesh.describe` label says about its image.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DescribeLabel {
    /// `kind`.
    pub kind: Option<Kind>,
    /// `target.entrypoint`, else `target.qualname`: the `module:Class` described.
    pub target: Option<String>,
    /// `runtime`.
    pub runtime: DescribeRuntime,
}

#[derive(Deserialize, Default)]
struct RawDescribe {
    #[serde(default)]
    schema_version: Option<serde_json::Value>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    target: Option<RawTarget>,
    #[serde(default)]
    runtime: Option<DescribeRuntime>,
}

#[derive(Deserialize, Default)]
struct RawTarget {
    #[serde(default)]
    entrypoint: Option<String>,
    #[serde(default)]
    qualname: Option<String>,
}

/// Parse a describe label's wrapper: schema version 1, an env/model kind, the
/// target, and the runtime offer.
pub fn parse_describe_label(raw: &str) -> Result<DescribeLabel, String> {
    let describe: RawDescribe = serde_json::from_str(raw)
        .map_err(|err| format!("{DESCRIBE_LABEL} is not valid JSON: {err}"))?;
    if describe
        .schema_version
        .as_ref()
        .and_then(serde_json::Value::as_u64)
        != Some(1)
    {
        return Err(format!(
            "{DESCRIBE_LABEL} schema_version {} is not the supported version 1",
            describe.schema_version.unwrap_or(serde_json::Value::Null)
        ));
    }
    let kind = match describe.kind.as_deref() {
        Some(kind) => Some(
            Kind::parse(kind)
                .ok_or_else(|| format!("{DESCRIBE_LABEL} kind {kind:?} is not 'env' or 'model'"))?,
        ),
        None => None,
    };
    let target = describe.target.unwrap_or_default();
    Ok(DescribeLabel {
        kind,
        target: target
            .entrypoint
            .or(target.qualname)
            .filter(|target| !target.is_empty()),
        runtime: describe.runtime.unwrap_or_default(),
    })
}

fn missing_describe_note(config: &ImageConfig) -> String {
    let command = parse_serve_command(config);
    if command.serve.is_some() || command.argv.is_empty() {
        return format!(
            "labels: no {DESCRIBE_LABEL} label; a runner reads describe off the \
             rlmesh.serve handshake instead (run `rlmesh check <module:Class>` for the \
             class-level checks)"
        );
    }
    format!(
        "labels: no {DESCRIBE_LABEL} label; a runner reads describe off the served \
         peer's handshake, which a native (C/C++) env publishes through the C ABI when it \
         binds (a native model publishes none). To describe the image before it runs, bake \
         {DESCRIBE_LABEL} with the env's describe JSON (`rlmesh_env_describe_json`, C++ \
         `EnvServer::describe_json()`)"
    )
}

/// Check the rlmesh labels' wrappers and read the describe label.
pub fn check_labels(config: &ImageConfig) -> (CheckReport, Option<DescribeLabel>) {
    let mut report = CheckReport::default();
    let label = match config.labels.get(DESCRIBE_LABEL) {
        None => {
            report.not_checked.push(missing_describe_note(config));
            None
        }
        Some(raw) => match parse_describe_label(raw) {
            Ok(label) => {
                report.passed.push(format!(
                    "labels: {DESCRIBE_LABEL} describes {} {}",
                    label.kind.map_or("?", Kind::name),
                    label.target.as_deref().unwrap_or("?")
                ));
                Some(label)
            }
            Err(message) => {
                report.failed.push(format!("labels: {message}"));
                None
            }
        },
    };
    if let Some(raw) = config.labels.get(PACKAGE_LABEL) {
        match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(serde_json::Value::Object(package)) => {
                if package
                    .get("schemaVersion")
                    .and_then(serde_json::Value::as_u64)
                    != Some(1)
                {
                    report.failed.push(format!(
                        "labels: {PACKAGE_LABEL} schemaVersion {} is not the supported \
                         version 1 (readers ignore the label)",
                        package
                            .get("schemaVersion")
                            .unwrap_or(&serde_json::Value::Null)
                    ));
                }
            }
            Ok(_) => report
                .failed
                .push(format!("labels: {PACKAGE_LABEL} must be a JSON object")),
            Err(err) => report
                .failed
                .push(format!("labels: {PACKAGE_LABEL} is not valid JSON: {err}")),
        }
    }
    (report, label)
}

/// Check where a describe label was generated: reject a label whose
/// `runtime.os` is not `linux` (the host's spaces, versions, and editions are
/// not the image's), and warns on an architecture other than `x86_64` /
/// `amd64`.
pub fn check_label_runtime(runtime: &DescribeRuntime) -> CheckReport {
    let mut report = CheckReport::default();
    match runtime.os.as_deref().map(str::trim) {
        None | Some("") => {}
        Some("linux") => report
            .passed
            .push("labels: describe label generated on linux".to_owned()),
        Some(os) => report.failed.push(format!(
            "labels: the describe label was generated on {os}; a label whose runtime.os \
             is not linux is rejected when the image is admitted. Bake it inside the image (docker run ... rlmesh \
             describe --label), not on the host"
        )),
    }
    match runtime.arch.as_deref().map(str::trim) {
        None | Some("") | Some("x86_64") | Some("amd64") => {}
        Some(arch) => report.warnings.push(format!(
            "labels: the describe label was generated on {arch}; the image runs on amd64, \
             so what the host could build may not be what the image builds"
        )),
    }
    report
}

/// What `check-image` checks an image's edition against, and how its messages
/// name it: this rlmesh build (the default), or a target platform's editions
/// given on the command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditionTarget {
    /// The workflow editions the target knows.
    pub editions: Vec<String>,
    /// The target as messages name it (`this rlmesh build`).
    pub name: String,
    /// The session wire revision the target speaks, when known.
    pub wire_revision: Option<u32>,
}

impl EditionTarget {
    /// The rlmesh this CLI was built with.
    pub fn this_build() -> Self {
        Self {
            editions: known_editions(),
            name: "this rlmesh build".to_owned(),
            wire_revision: Some(WIRE_REVISION),
        }
    }

    /// A target platform that knows `editions`; its wire revision is not known.
    pub fn platform(editions: Vec<String>) -> Self {
        Self {
            editions,
            name: "the target platform".to_owned(),
            wire_revision: None,
        }
    }
}

fn known_editions() -> Vec<String> {
    Edition::ALL
        .iter()
        .map(|edition| edition.name().to_owned())
        .collect()
}

/// The editions a platform says it knows, trimmed, blanks dropped.
pub fn platform_editions(editions: &[String]) -> Vec<String> {
    editions
        .iter()
        .map(|edition| edition.trim().to_owned())
        .filter(|edition| !edition.is_empty())
        .collect()
}

/// The editions a platform on the `rlmesh` version `version` knows, when that
/// is this CLI's own version.
pub fn platform_editions_for_version(version: &str) -> Option<Vec<String>> {
    (normalize_version(version) == normalize_version(env!("CARGO_PKG_VERSION")))
        .then(known_editions)
}

/// Spell a package version the SemVer way this workspace does (`0.1.0-rc.13`),
/// accepting PEP 440 (`0.1.0rc13`), a loose `0.1.0-rc13` / `0.1.0.rc.13`, a `v`
/// prefix, and mixed case.
pub fn normalize_version(version: &str) -> String {
    let version = version.trim().to_ascii_lowercase();
    let version = version.strip_prefix('v').unwrap_or(&version);
    let base_end = version
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(version.len());
    let base = version[..base_end].trim_end_matches('.');
    let rest = version[base_end..].trim_start_matches(['-', '.', '_']);
    if rest.is_empty() {
        return base.to_owned();
    }
    let tag_end = rest
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(rest.len());
    let tag = match &rest[..tag_end] {
        "alpha" => "a",
        "beta" => "b",
        "c" | "pre" | "preview" => "rc",
        other => other,
    };
    let number = rest[tag_end..].trim_start_matches(['-', '.', '_']);
    if tag.is_empty() || number.is_empty() || !number.chars().all(|c| c.is_ascii_digit()) {
        return version.to_owned();
    }
    format!("{base}-{tag}.{number}")
}

/// Check that the image's rlmesh speaks the target's protocol generation and
/// session wire revision, and declares a workflow edition the target knows.
pub fn check_editions(runtime: &DescribeRuntime, target: &EditionTarget) -> CheckReport {
    let mut report = CheckReport::default();
    let name = &target.name;
    match runtime.protocol_generation.as_deref().map(str::trim) {
        None | Some("") => report.not_checked.push(
            "editions: describe carries no protocol_generation (built with an rlmesh before \
             the field); the session handshake checks it when the image runs"
                .to_owned(),
        ),
        Some("rlmesh-wire-v1") => report.failed.push(
            "editions: this rlmesh-wire-v1 image needs a RLMesh 0.1 runner; rebuild it with RLMesh 0.2 to use realtime sessions".to_owned(),
        ),
        Some(generation) if generation != PROTOCOL_GENERATION => report.failed.push(format!(
            "editions: the image speaks protocol generation {generation}; {name} speaks \
             {PROTOCOL_GENERATION}; rebuild the image with an rlmesh of that generation"
        )),
        Some(_) => {
            report.passed.push(format!(
                "editions: protocol generation {PROTOCOL_GENERATION}"
            ));
            check_wire_revision(runtime.session_wire_revision, target, &mut report);
        }
    }
    match runtime
        .workflow_edition
        .as_deref()
        .map(str::trim)
        .filter(|edition| !edition.is_empty())
    {
        None => report.not_checked.push(
            "editions: describe carries no workflow_edition (built with an rlmesh before the \
             field); the session checks it when the image runs"
                .to_owned(),
        ),
        Some(edition) if target.editions.iter().any(|known| known == edition) => {
            report.passed.push(format!(
                "editions: workflow edition {edition}, which {name} knows"
            ))
        }
        Some(edition) => report.failed.push(format!(
            "editions: the image declares workflow edition {edition}, which {name} does not \
             know (it knows {:?}); run it on an rlmesh release that knows {edition}, or rebuild \
             the image on a release whose edition {name} knows",
            target.editions,
        )),
    }
    report
}

fn check_wire_revision(image: Option<u32>, target: &EditionTarget, report: &mut CheckReport) {
    let Some(own) = target.wire_revision else {
        report.not_checked.push(format!(
            "editions: session wire revision {}: {} names no wire revision of its own; the \
             session checks it when the image runs",
            image.map_or("absent".to_owned(), |revision| revision.to_string()),
            target.name,
        ));
        return;
    };
    match image {
        None => report.warnings.push(format!(
            "editions: describe carries no session_wire_revision (an rlmesh 0.2 pre-release \
             from before the field, whose session host listened on four ports); this rlmesh \
             speaks revision {WIRE_REVISION}: rebuild the image against it"
        )),
        Some(revision) if revision != own => report.failed.push(format!(
            "editions: the image speaks session wire revision {revision} ({}), {} speaks \
             revision {own} ({}); rebuild both from the same release",
            wire_revision_name(revision),
            target.name,
            wire_revision_name(own),
        )),
        Some(_) => report
            .passed
            .push(format!("editions: session wire revision {own}")),
    }
}

/// The workflow edition the image's `ENV RLMESH_WORKFLOW_EDITION` declares
/// above whatever the describe label recorded; `None` when it sets none or a
/// blank one.
pub fn declared_workflow_edition(config: &ImageConfig) -> Option<String> {
    config
        .env_value(WORKFLOW_EDITION_ENV)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Run every image-config check: platform, serve command, labels, the compute
/// variant ([`crate::variant::check_variant`]: declared or inferred), and (when
/// the describe label is there) wire and edition compatibility with `target`,
/// or a `not_checked` note when the caller has no target to check against.
pub fn check_image(config: &ImageConfig, target: Option<&EditionTarget>) -> CheckReport {
    let mut report = check_platform(config);
    let (labels, describe) = check_labels(config);
    let kind = describe.as_ref().and_then(|label| label.kind);
    report.extend(check_serve_command(config, kind));
    report.extend(labels);
    report.extend(crate::variant::check_variant(config));
    match describe {
        Some(mut label) => {
            let command = parse_serve_command(config);
            if let (Some(served), Some(described)) = (&command.target, &label.target)
                && served != described
            {
                report.failed.push(format!(
                    "entrypoint: serves {served:?} but the describe label is for {described:?}"
                ));
            }
            report.extend(check_label_runtime(&label.runtime));
            if let Some(declared) = declared_workflow_edition(config) {
                report.passed.push(format!(
                    "editions: the image declares {declared} (above the label's {})",
                    label.runtime.workflow_edition.as_deref().unwrap_or("none")
                ));
                label.runtime.workflow_edition = Some(declared);
            }
            match target {
                Some(target) => report.extend(check_editions(&label.runtime, target)),
                None => report.not_checked.push(
                    "editions: no editions to check against; pass --platform-editions \
                     with the list `rlmesh version` prints on the target's rlmesh"
                        .to_owned(),
                ),
            }
        }
        None => report.not_checked.push(
            "editions: not checked without a describe label; the session handshake checks \
             them when the image runs"
                .to_owned(),
        ),
    }
    report
}

#[cfg(test)]
mod tests;
