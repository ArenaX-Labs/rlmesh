//! Pure checks over a built image's OCI config: the serve command and the address it
//! binds, the port it
//! exposes, its platform, its rlmesh labels (including the package label's
//! `variant` / `profiles` blocks, see [`crate::variant`]), and whether the
//! rlmesh it was built with can meet a platform's workflow editions.
//!
//! Nothing here touches docker, the network, or Python: the caller hands in an
//! [`ImageConfig`] (parsed from `docker image inspect`, a registry manifest, ...)
//! and reads a [`CheckReport`] back. Function names, report buckets and the
//! [`RuntimeOffer`] field names (which are the describe envelope's
//! `runtime.*` keys and the wire handshake's) are a contract.
//!
//! The deep describe-envelope checks (a model without a spec, an env without
//! tags, a spec that does not resolve) need the Python gatherer and live in
//! `rlmesh._describe`; this module only reads the envelope's kind, target, and
//! edition offer off the label.

mod serve;

pub use serve::{ServeCommand, check_serve_command, parse_serve_command, shell_words};

use std::collections::BTreeMap;

use rlmesh_proto::{
    EditionRefusal, PROTOCOL_GENERATION, SessionFloor, SessionOffer, negotiate_session_floor,
    supported_workflow_editions,
};
use rlmesh_session::wire::version::{WIRE_REVISION, wire_revision_name};
use serde::{Deserialize, Serialize};

/// OCI config label carrying the describe envelope (`python -m rlmesh._describe`).
pub const DESCRIBE_LABEL: &str = "dev.rlmesh.describe";
/// OCI config label carrying packaged checkpoint declarations.
pub const PACKAGE_LABEL: &str = "dev.rlmesh.package";
/// The port `rlmesh.serve` hosts its session on when `RLMESH_ADDRESS` names
/// none: the one port every participant dials.
pub const SERVE_PORT: u16 = rlmesh_session::address::DEFAULT_PORT;
/// The env var `rlmesh.serve` binds.
pub const ADDRESS_ENV: &str = "RLMESH_ADDRESS";
/// The env var that declares a served peer's workflow edition, above
/// `--workflow-edition` and the class declaration; empty declares none.
pub const WORKFLOW_EDITION_ENV: &str = "RLMESH_WORKFLOW_EDITION";

/// What a check found, in the buckets the CLI prints. `failed` means the push
/// contains an invalid or non-runnable image; `warnings` identify inconsistent
/// or unverifiable claims; `not_checked` names
/// what could not be decided here (a custom entrypoint, an envelope without
/// editions) and who decides it instead; `passed` is informational.
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

/// The parts of an OCI image config the checks read. Every field is what
/// `docker image inspect` reports under `Config` (plus the top-level `Os` /
/// `Architecture`); a caller with another source fills the same shape.
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
    /// Parse the JSON `docker image inspect <ref>` prints (a one-element array;
    /// a bare object is accepted too).
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

    /// Parse an OCI image config as a registry serves it (what
    /// `docker buildx imagetools inspect REF --format '{{json .Image}}'`
    /// prints): lowercase `os` / `architecture` / `config`, with the
    /// `config` keys spelled as in `docker image inspect`.
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

/// Check the image's OS/architecture against supported deployment targets.
pub fn check_platform(config: &ImageConfig) -> CheckReport {
    let mut report = CheckReport::default();
    if config.os.is_empty() || config.architecture.is_empty() {
        report
            .not_checked
            .push("platform: the image config names no os/architecture".to_owned());
    } else if config.os != "linux" || config.architecture != "amd64" {
        report.failed.push(format!(
            "platform: image is {}/{}; the platform runs linux/amd64 (build with \
             --platform linux/amd64)",
            config.os, config.architecture
        ));
    } else {
        report.passed.push("platform: linux/amd64".to_owned());
    }
    report
}

/// The edition handshake a describe envelope's `runtime` block advertises. The
/// field names are the envelope's `runtime.*` keys and the wire
/// `HandshakeRequest`'s: `protocol_generation`, `supported_workflow_editions`
/// (CAN), `preferred_workflow_edition` (WANT). Every field is optional so an
/// envelope from an rlmesh that predates them still parses.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOffer {
    /// The `rlmesh` package the image was described with.
    #[serde(default)]
    pub package_version: Option<String>,
    /// The wire generation the image speaks (`rlmesh-realtime-v1`).
    #[serde(default)]
    pub protocol_generation: Option<String>,
    /// The session wire revision the image's rlmesh speaks
    /// (`rlmesh_session::wire::version::WIRE_REVISION`, 1 for one port):
    /// a 0.2 pre-release bumps it on a wire break, so two builds of one
    /// generation that cannot talk differ here. Absent from an envelope an
    /// earlier pre-release described.
    #[serde(default)]
    pub session_wire_revision: Option<u32>,
    /// Every edition the image's rlmesh can drive a session at.
    #[serde(default)]
    pub supported_workflow_editions: Vec<String>,
    /// The edition the image declares (its ceiling).
    #[serde(default)]
    pub preferred_workflow_edition: Option<String>,
    /// Why the class's declared edition cannot be run by the rlmesh that
    /// described it (`rlmesh.serve` refuses to start on the same declaration);
    /// absent when the declaration is fine.
    #[serde(default)]
    pub workflow_edition_error: Option<String>,
    /// Nothing declares an edition, so the image floats to its rlmesh's
    /// newest; absent for a declaration or a deliberate float.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub workflow_edition_floats: bool,
    /// The OS the envelope was generated on (`linux`); a label baked on a
    /// Mac carries `macos` and fails validation.
    #[serde(default)]
    pub os: Option<String>,
    /// The architecture the envelope was generated on (`x86_64` / `amd64`).
    #[serde(default)]
    pub arch: Option<String>,
}

impl RuntimeOffer {
    /// The image's bind-time offer, or `None` when the envelope advertises no
    /// editions (an older rlmesh); the managed platform checks the handshake
    /// when the image is onboarded.
    pub fn session_offer(&self) -> Option<SessionOffer> {
        if self.supported_workflow_editions.is_empty() {
            return None;
        }
        Some(SessionOffer {
            editions: self.supported_workflow_editions.clone(),
            preferred: self
                .preferred_workflow_edition
                .as_deref()
                .map(str::trim)
                .filter(|edition| !edition.is_empty())
                .map(str::to_owned),
        })
    }
}

/// The nudge for a participant that declares no workflow edition, or `None`
/// when it declares one (or floats deliberately).
pub fn edition_float_warning(runtime: &RuntimeOffer) -> Option<String> {
    if !runtime.workflow_edition_floats {
        return None;
    }
    let floats_to = runtime
        .preferred_workflow_edition
        .as_deref()
        .map(str::trim)
        .filter(|edition| !edition.is_empty())
        .unwrap_or(rlmesh_proto::WORKFLOW_EDITION_BASE);
    let base = floats_to.split('-').next().unwrap_or(floats_to);
    Some(format!(
        "editions: no workflow edition declared: it floats to {floats_to}, which moves when \
         its rlmesh upgrades. Pin it before you upgrade: workflow_edition = \"{base}\" on the \
         class, under [tool.rlmesh] in pyproject.toml, or {WORKFLOW_EDITION_ENV}={base} in \
         the image ({WORKFLOW_EDITION_ENV}='' floats deliberately)"
    ))
}

/// What a `dev.rlmesh.describe` label says about its image.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DescribeLabel {
    /// `kind`.
    pub kind: Option<Kind>,
    /// `target.entrypoint`, else `target.qualname`: the `module:Class` described.
    pub target: Option<String>,
    /// `runtime`.
    pub runtime: RuntimeOffer,
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
    runtime: Option<RuntimeOffer>,
}

#[derive(Deserialize, Default)]
struct RawTarget {
    #[serde(default)]
    entrypoint: Option<String>,
    #[serde(default)]
    qualname: Option<String>,
}

/// Parse a describe label's wrapper: schema version 1, an env/model kind, the
/// target, and the runtime offer. Sub-pieces are not validated here.
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

/// The note for an image without a describe label, by what it serves: an
/// `rlmesh.serve` image is described on its handshake; a custom entrypoint is
/// most likely a native host, whose env describes itself through the C ABI.
fn missing_describe_note(config: &ImageConfig) -> String {
    let command = parse_serve_command(config);
    if command.serve.is_some() || command.argv.is_empty() {
        return format!(
            "labels: no {DESCRIBE_LABEL} label; the platform reads describe off the \
             rlmesh.serve handshake instead (run `rlmesh check <module:Class>` for the \
             class-level checks)"
        );
    }
    format!(
        "labels: no {DESCRIBE_LABEL} label; the platform reads describe off the served \
         peer's handshake, which a native (C/C++) env publishes through the C ABI when it \
         binds (a native model publishes none). To describe the image before it runs, bake \
         {DESCRIBE_LABEL} with the env's describe JSON (`rlmesh_env_describe_json`, C++ \
         `EnvServer::describe_json()`)"
    )
}

/// Check the rlmesh labels' wrappers and read the describe label. A missing
/// describe label is not a failure: the parser reads describe off the served
/// peer's handshake instead, which `rlmesh.serve` stamps, and a native (C/C++)
/// env stamps through the C ABI when it binds.
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
                         version 1 (the platform will ignore the label)",
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

/// Check where a describe label was generated: reject a label
/// whose `runtime.os` is not `linux` (the host's spaces, versions, and
/// editions are not the image's), and warns on an architecture other than
/// `x86_64` / `amd64`. Absent fields are not checked.
pub fn check_label_runtime(runtime: &RuntimeOffer) -> CheckReport {
    let mut report = CheckReport::default();
    match runtime.os.as_deref().map(str::trim) {
        None | Some("") => {}
        Some("linux") => report
            .passed
            .push("labels: describe label generated on linux".to_owned()),
        Some(os) => report.failed.push(format!(
            "labels: the describe label was generated on {os}; the platform fails a label \
             whose runtime.os is not linux. Bake it inside the image (docker run ... rlmesh \
             describe --label), not on the host"
        )),
    }
    match runtime.arch.as_deref().map(str::trim) {
        None | Some("") | Some("x86_64") | Some("amd64") => {}
        Some(arch) => report.warnings.push(format!(
            "labels: the describe label was generated on {arch}; the platform runs amd64, \
             so what the host could build may not be what the image builds"
        )),
    }
    report
}

/// A platform's offer from the editions it says it can drive (undeclared
/// WANT, so it admits everything it can).
pub fn platform_offer(editions: &[String]) -> SessionOffer {
    SessionOffer {
        editions: editions
            .iter()
            .map(|edition| edition.trim().to_owned())
            .filter(|edition| !edition.is_empty())
            .collect(),
        preferred: None,
    }
}

/// A platform's offer from the `rlmesh` version it runs. Only this CLI's own
/// version is known: a build's retained list is generated from its own
/// `rlmesh.toml`, and it changes between releases (the `0.1.0-rc.10` to
/// `rc.12` builds offered only their cohort, `2026.06` first appears in
/// `rc.13`), so any other version returns `None` and the caller asks for
/// `--platform-editions`. Spellings are normalized with [`normalize_version`],
/// so PyPI's `0.1.0rc13` names the same build as `0.1.0-rc.13`.
pub fn platform_offer_for_version(version: &str) -> Option<SessionOffer> {
    (normalize_version(version) == normalize_version(env!("CARGO_PKG_VERSION"))).then(|| {
        SessionOffer {
            editions: supported_workflow_editions(),
            preferred: None,
        }
    })
}

/// Spell a package version the SemVer way this workspace does
/// (`0.1.0-rc.13`), accepting PEP 440 (`0.1.0rc13`), a loose `0.1.0-rc13` /
/// `0.1.0.rc.13`, a `v` prefix, and mixed case. `alpha`/`beta` collapse to
/// `a`/`b` as PEP 440 does. A spelling this cannot read is returned trimmed
/// and lowercased, so it only ever matches itself.
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

/// Reconcile an image's offer with a platform's the way the runtime will at
/// bind time: the image sits in its own tier (env or model; an unknown kind
/// takes the env tier, the rule is symmetric), the platform stands in for the
/// runtime and for the peer on the other side, which it also supplies.
pub fn negotiate_image(
    image: &SessionOffer,
    kind: Option<Kind>,
    platform: &SessionOffer,
) -> Result<SessionFloor, EditionRefusal> {
    match kind {
        Some(Kind::Model) => negotiate_session_floor(platform, image, platform),
        _ => negotiate_session_floor(image, platform, platform),
    }
}

/// Check that the image's rlmesh speaks the platform's protocol generation and
/// shares a workflow edition with it. An envelope without the fields is
/// `not_checked`: the managed platform checks the handshake when the image is
/// onboarded.
pub fn check_editions(
    runtime: &RuntimeOffer,
    kind: Option<Kind>,
    platform: &SessionOffer,
) -> CheckReport {
    let mut report = CheckReport::default();
    match runtime.protocol_generation.as_deref().map(str::trim) {
        None | Some("") => report.not_checked.push(
            "editions: describe carries no protocol_generation (built with an rlmesh before \
             the field); the managed platform checks the handshake when you onboard it"
                .to_owned(),
        ),
        Some("rlmesh-wire-v1") => report.failed.push(
            "editions: this rlmesh-wire-v1 image needs a RLMesh 0.1 runner; rebuild it with RLMesh 0.2 to use realtime sessions".to_owned(),
        ),
        Some(generation) if generation != PROTOCOL_GENERATION => report.failed.push(format!(
            "editions: image speaks protocol generation {generation}, the platform speaks \
             {PROTOCOL_GENERATION}; rebuild against the platform's rlmesh"
        )),
        Some(_) => {
            report.passed.push(format!(
                "editions: protocol generation {PROTOCOL_GENERATION}"
            ));
            check_wire_revision(runtime.session_wire_revision, &mut report);
        }
    }
    if let Some(error) = runtime
        .workflow_edition_error
        .as_deref()
        .map(str::trim)
        .filter(|error| !error.is_empty())
    {
        report.failed.push(format!(
            "editions: the image declares a workflow edition its rlmesh cannot run, so \
             rlmesh.serve will refuse to start ({error})"
        ));
        return report;
    }
    let Some(image) = runtime.session_offer() else {
        report.not_checked.push(
            "editions: describe carries no supported_workflow_editions (built with an rlmesh \
             before the field); the managed platform checks the handshake when you onboard it"
                .to_owned(),
        );
        return report;
    };
    match negotiate_image(&image, kind, platform) {
        Ok(floor) => report.passed.push(format!(
            "editions: a session would run at {} (image can {:?}, wants {}; platform can {:?})",
            floor.selected_workflow_edition,
            image.editions,
            image.preferred.as_deref().unwrap_or("its newest"),
            platform.editions,
        )),
        Err(refusal) => report.failed.push(format!(
            "editions: the image and the platform share no workflow edition ({refusal}); \
             rebuild with an rlmesh that offers one of the platform's editions"
        )),
    }
    report
}

/// Check that an image of this protocol generation speaks this rlmesh's
/// session wire revision: a 0.2 pre-release's wire may break before 0.2.0
/// freezes it, and a pair on two revisions cannot talk.
fn check_wire_revision(image: Option<u32>, report: &mut CheckReport) {
    match image {
        None => report.warnings.push(format!(
            "editions: describe carries no session_wire_revision (an rlmesh 0.2 pre-release \
             from before the field, whose session host listened on four ports); this rlmesh \
             speaks revision {WIRE_REVISION}: rebuild the image against it"
        )),
        Some(revision) if revision != WIRE_REVISION => report.failed.push(format!(
            "editions: the image speaks session wire revision {revision} ({}), this rlmesh \
             speaks revision {WIRE_REVISION} ({}); rebuild both from the same release",
            wire_revision_name(revision),
            wire_revision_name(WIRE_REVISION),
        )),
        Some(_) => report
            .passed
            .push(format!("editions: session wire revision {WIRE_REVISION}")),
    }
}

/// The workflow edition the image's serve command declares above whatever the
/// describe label recorded: `ENV RLMESH_WORKFLOW_EDITION` (an empty value
/// declares none, deliberately), else a baked `--workflow-edition`. `None`
/// when neither is present, so the label's value stands. The value's
/// validity is the runtime's to judge; this only reads the precedence.
pub fn declared_workflow_edition(config: &ImageConfig) -> Option<Option<String>> {
    if let Some(value) = config.env_value(WORKFLOW_EDITION_ENV) {
        let value = value.trim();
        return Some((!value.is_empty()).then(|| value.to_owned()));
    }
    parse_serve_command(config).workflow_edition.map(|value| {
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

/// Run every image-config check: platform, serve command, labels, the compute
/// variant ([`crate::variant::check_variant`]: declared or inferred), and (when
/// the describe label advertises editions) edition compatibility with
/// `platform`, or a `not_checked` note when the caller has no platform offer
/// to check against. A `--workflow-edition` or `ENV RLMESH_WORKFLOW_EDITION`
/// on the image overrides the label's `preferred_workflow_edition`, as it
/// does for the served peer.
pub fn check_image(config: &ImageConfig, platform: Option<&SessionOffer>) -> CheckReport {
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
                    "editions: the image declares {} (above the label's {})",
                    declared
                        .as_deref()
                        .unwrap_or("no edition, floating to its newest"),
                    label
                        .runtime
                        .preferred_workflow_edition
                        .as_deref()
                        .unwrap_or("none")
                ));
                label.runtime.preferred_workflow_edition = declared;
                label.runtime.workflow_edition_floats = false;
            }
            report
                .warnings
                .extend(edition_float_warning(&label.runtime));
            match platform {
                Some(platform) => {
                    report.extend(check_editions(&label.runtime, label.kind, platform));
                }
                None => report.not_checked.push(
                    "editions: no platform editions to check against; pass \
                     --platform-editions with the list `rlmesh version` prints on the \
                     platform's rlmesh"
                        .to_owned(),
                ),
            }
        }
        None => report.not_checked.push(
            "editions: not checked without a describe label; the managed platform checks \
             the handshake when you onboard the image"
                .to_owned(),
        ),
    }
    report
}

#[cfg(test)]
mod tests;
