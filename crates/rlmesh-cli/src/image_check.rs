//! Pure checks over a built image's OCI config: the serve command and the
//! address it binds, the port it exposes, its platform, its rlmesh labels
//! (including the package label's `variant` / `profiles` blocks, see
//! [`crate::variant`]), and whether the rlmesh it was built with shares a
//! workflow edition with an [`EditionTarget`]: the rlmesh this CLI was built
//! with, or a platform's editions named on the command line.

mod serve;

pub use serve::{ServeCommand, check_serve_command, parse_serve_command, shell_words};

use std::collections::BTreeMap;

use rlmesh_proto::{
    EditionRefusal, PROTOCOL_GENERATION, SessionFloor, SessionOffer, negotiate_session_floor,
    supported_workflow_editions,
};
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
/// The env var that declares a served peer's workflow edition, above
/// `--workflow-edition` and the class declaration; empty declares none.
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

/// The edition handshake a describe envelope's `runtime` block advertises.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOffer {
    /// The `rlmesh` package the image was described with.
    #[serde(default)]
    pub package_version: Option<String>,
    /// The wire generation the image speaks (`rlmesh-realtime-v1`).
    #[serde(default)]
    pub protocol_generation: Option<String>,
    /// The session wire revision the image's rlmesh speaks
    /// (`rlmesh_session::wire::version::WIRE_REVISION`, 1 for one port): a 0.2
    /// pre-release bumps it on a wire break, so two builds of one generation that
    /// cannot talk differ here.
    #[serde(default)]
    pub session_wire_revision: Option<u32>,
    /// Every edition the image's rlmesh can drive a session at.
    #[serde(default)]
    pub supported_workflow_editions: Vec<String>,
    /// The edition the image declares (its ceiling).
    #[serde(default)]
    pub preferred_workflow_edition: Option<String>,
    /// Why the class's declared edition cannot be run by the rlmesh that described
    /// it (`rlmesh.serve` refuses to start on the same declaration); absent when
    /// the declaration is fine.
    #[serde(default)]
    pub workflow_edition_error: Option<String>,
    /// Nothing declares an edition, so the image floats to its rlmesh's newest;
    /// absent for a declaration or a deliberate float.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub workflow_edition_floats: bool,
    /// The OS the envelope was generated on (`linux`); a label baked on a Mac
    /// carries `macos` and fails validation.
    #[serde(default)]
    pub os: Option<String>,
    /// The architecture the envelope was generated on (`x86_64` / `amd64`).
    #[serde(default)]
    pub arch: Option<String>,
}

impl RuntimeOffer {
    /// The image's bind-time offer, or `None` when the envelope advertises no
    /// editions (an older rlmesh); the session handshake checks it when the image
    /// runs.
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
pub fn check_label_runtime(runtime: &RuntimeOffer) -> CheckReport {
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

/// What `check-image` checks an image's editions against, and how its messages
/// name it: this rlmesh build (the default), or a target platform's editions
/// given on the command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditionTarget {
    /// The editions the target can drive.
    pub offer: SessionOffer,
    /// The target as messages name it (`this rlmesh build`).
    pub name: String,
}

impl EditionTarget {
    /// The rlmesh this CLI was built with.
    pub fn this_build() -> Self {
        Self {
            offer: SessionOffer::this_build(None),
            name: "this rlmesh build".to_owned(),
        }
    }

    /// A target platform's offer.
    pub fn platform(offer: SessionOffer) -> Self {
        Self {
            offer,
            name: "the target platform".to_owned(),
        }
    }
}

/// A platform's offer from the editions it says it can drive (undeclared WANT,
/// so it admits everything it can).
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

/// A platform's offer from the `rlmesh` version it runs.
pub fn platform_offer_for_version(version: &str) -> Option<SessionOffer> {
    (normalize_version(version) == normalize_version(env!("CARGO_PKG_VERSION"))).then(|| {
        SessionOffer {
            editions: supported_workflow_editions(),
            preferred: None,
        }
    })
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

/// Reconcile an image's offer with a target's the way the runtime will at bind
/// time: the image sits in its own tier (env or model; an unknown kind takes
/// the env tier, the rule is symmetric), the target stands in for the runtime
/// and for the peer on the other side.
pub fn negotiate_image(
    image: &SessionOffer,
    kind: Option<Kind>,
    target: &SessionOffer,
) -> Result<SessionFloor, EditionRefusal> {
    match kind {
        Some(Kind::Model) => negotiate_session_floor(target, image, target),
        _ => negotiate_session_floor(image, target, target),
    }
}

/// Check that the image's rlmesh speaks the target's protocol generation and
/// shares a workflow edition with it.
pub fn check_editions(
    runtime: &RuntimeOffer,
    kind: Option<Kind>,
    target: &EditionTarget,
) -> CheckReport {
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
             before the field); the session handshake checks them when the image runs"
                .to_owned(),
        );
        return report;
    };
    let speaks = |offer: &SessionOffer| match offer.preferred.as_deref() {
        Some(preferred) => format!("{:?}, preferring {preferred}", offer.editions),
        None => format!("{:?}", offer.editions),
    };
    match negotiate_image(&image, kind, &target.offer) {
        Ok(floor) => report.passed.push(format!(
            "editions: a session would run at {} (the image speaks {}; {name} speaks {:?})",
            floor.selected_workflow_edition,
            speaks(&image),
            target.offer.editions,
        )),
        Err(refusal) => {
            let remedy = if refusal.involves_dev_build() {
                rlmesh_proto::DEV_BUILD_HINT.to_owned()
            } else {
                format!("rebuild the image with an rlmesh that offers one of {name}'s editions")
            };
            report.failed.push(format!(
                "editions: the image speaks {}; {name} speaks {:?}; no edition satisfies \
                 both, so a session would be refused: {remedy}",
                speaks(&image),
                target.offer.editions,
            ));
        }
    }
    report
}

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
/// declares none, deliberately), else a baked `--workflow-edition`.
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
/// the describe label advertises editions) edition compatibility with `target`,
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
            match target {
                Some(target) => {
                    report.extend(check_editions(&label.runtime, label.kind, target));
                }
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
