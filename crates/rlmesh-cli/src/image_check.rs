//! Pure checks over a built image's OCI config, the way the managed platform's
//! admission reads it: the serve command and the address it binds, the port it
//! exposes, its platform, its rlmesh labels, and whether the rlmesh it was
//! built with can meet a platform's workflow editions.
//!
//! Nothing here touches docker, the network, or Python: the caller hands in an
//! [`ImageConfig`] (parsed from `docker image inspect`, a registry manifest, ...)
//! and reads a [`CheckReport`] back. The managed platform's tooling calls these
//! functions too, so their names, the report's buckets, and the
//! [`RuntimeOffer`] field names (which are the describe envelope's
//! `runtime.*` keys and the wire handshake's) are a contract.
//!
//! The deep describe-envelope checks (a model without a spec, an env without
//! tags, a spec that does not resolve) need the Python gatherer and live in
//! `rlmesh._describe`; this module only reads the envelope's kind, target, and
//! edition offer off the label.

use std::collections::BTreeMap;

use rlmesh_proto::{
    EditionRefusal, PROTOCOL_GENERATION, SessionFloor, SessionOffer, negotiate_session_floor,
    supported_workflow_editions,
};
use serde::{Deserialize, Serialize};

/// OCI config label carrying the describe envelope (`python -m rlmesh._describe`).
pub const DESCRIBE_LABEL: &str = "dev.rlmesh.describe";
/// OCI config label carrying packaged checkpoint declarations.
pub const PACKAGE_LABEL: &str = "dev.rlmesh.package";
/// The port the platform connects to a served peer on.
pub const SERVE_PORT: &str = "50051";
/// The env var `rlmesh.serve` binds; the platform assigns it per pod.
pub const ADDRESS_ENV: &str = "RLMESH_ADDRESS";
/// The env var that declares a served peer's workflow edition, above
/// `--workflow-edition` and the class declaration; empty declares none.
pub const WORKFLOW_EDITION_ENV: &str = "RLMESH_WORKFLOW_EDITION";

/// What a check found, in the buckets the CLI prints. `failed` means the push
/// would land as not-runnable or the platform would reject a claim; `warnings`
/// are claims the platform trims or intent it cannot see; `not_checked` names
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

    fn name(self) -> &'static str {
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
        let config = inspect.config.unwrap_or_default();
        Ok(Self {
            os: inspect.os.unwrap_or_default(),
            architecture: inspect.architecture.unwrap_or_default(),
            entrypoint: config.entrypoint.unwrap_or_default(),
            cmd: config.cmd.unwrap_or_default(),
            env: config.env.unwrap_or_default(),
            exposed_ports: config
                .exposed_ports
                .unwrap_or_default()
                .into_keys()
                .collect(),
            labels: config.labels.unwrap_or_default(),
        })
    }

    /// The value of the `KEY=value` entry in `env`, when set.
    pub fn env_value(&self, key: &str) -> Option<&str> {
        self.env
            .iter()
            .find_map(|entry| entry.strip_prefix(key)?.strip_prefix('='))
    }

    /// Whether `exposed_ports` lists `port` on any protocol.
    pub fn exposes_port(&self, port: &str) -> bool {
        self.exposed_ports
            .iter()
            .any(|entry| entry.split('/').next() == Some(port))
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

/// The `python -m rlmesh.serve ...` invocation read off an image's
/// `Entrypoint` + `Cmd`. See [`parse_serve_command`] for the exact reading.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServeCommand {
    /// The command's words: Docker's argv as stored, with a shell-form
    /// script split into words (see [`parse_serve_command`]).
    pub argv: Vec<String>,
    /// Index of the `rlmesh.serve` word in the first `-m rlmesh.serve` pair;
    /// `None` when the command never runs the module (a custom entrypoint).
    pub serve: Option<usize>,
    /// Whether `--env` is present (the image serves an environment).
    pub env: bool,
    /// The `module:Class` the command serves.
    pub target: Option<String>,
    /// The value of a baked `--address`, when one is present.
    pub address: Option<String>,
    /// The value of a baked `--workflow-edition`, when one is present.
    pub workflow_edition: Option<String>,
}

impl ServeCommand {
    /// The kind the command serves, when it runs `rlmesh.serve` at all.
    pub fn kind(&self) -> Option<Kind> {
        self.serve
            .map(|_| if self.env { Kind::Env } else { Kind::Model })
    }
}

/// Read the serve command off an image config. This is how an image with no
/// describe label is classified; the platform then learns the rest from the
/// served peer's handshake. The platform's own parser applies these same
/// rules, so an image reads the same on both sides:
///
/// 1. The command is `Entrypoint` followed by `Cmd`, each element one word:
///    Docker's argument boundaries are kept, so an exec-form
///    `["python", "-m", "rlmesh.serve", "pkg:Policy", "--kwargs-json", "{\"a\": 1}"]`
///    carries the JSON as one word.
/// 2. Shell form only: when the first word is a shell (`sh`, `bash`, `dash`,
///    `ash`, `zsh`, with or without a directory) and the second is a `-c`
///    style flag (`-c`, `-lc`, `-ec`, ...), the third word is a script and
///    is split into words with POSIX quoting (single quotes literal, double
///    quotes with `\"` `\\` `\$` escapes, a backslash escaping the next
///    character; no expansion, `$@` stays a word, operators like `&&` stay
///    words). The words after the script (`sh -c '...' -- ARGS`) are appended
///    unchanged, which is how a `bash -c 'exec python "$@"' --` entrypoint
///    hands `Cmd` through.
/// 3. The serve module is the first `rlmesh.serve` word preceded by `-m`.
///    Nothing before it matters (`uv run`, `exec`, `source ... &&`).
/// 4. After it, the words are read in order: `--env X` / `--env=X` mark an
///    env image and its target; `--address [X]`, `--address=X`,
///    `--workflow-edition [X]`, `--workflow-edition=X` are recorded; any other
///    `--flag` skips one following word as its value unless that word starts
///    with `--` or the flag carries `=`; the **first** remaining bare word is
///    the model target (a model image's `pkg:Policy` comes first by
///    construction; a later bare word is a flag value the reader did not
///    recognize, never the target).
pub fn parse_serve_command(config: &ImageConfig) -> ServeCommand {
    let raw: Vec<&str> = config
        .entrypoint
        .iter()
        .chain(&config.cmd)
        .map(String::as_str)
        .collect();
    let argv: Vec<String> = match raw.as_slice() {
        [shell, flag, script, rest @ ..] if is_shell(shell) && is_command_flag(flag) => {
            shell_words(script)
                .into_iter()
                .chain(rest.iter().map(|word| (*word).to_owned()))
                .collect()
        }
        words => words.iter().map(|word| (*word).to_owned()).collect(),
    };
    let serve = argv
        .iter()
        .enumerate()
        .skip(1)
        .find(|(index, token)| *token == "rlmesh.serve" && argv[index - 1] == "-m")
        .map(|(index, _)| index);
    let mut command = ServeCommand {
        argv,
        serve,
        ..ServeCommand::default()
    };
    let Some(serve) = serve else {
        return command;
    };
    let args = &command.argv[serve + 1..];
    let mut positional = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--env" {
            command.env = true;
            if let Some(value) = args.get(index + 1) {
                command.target = Some(value.clone());
                index += 1;
            }
        } else if let Some(value) = arg.strip_prefix("--env=") {
            command.env = true;
            command.target = Some(value.to_owned());
        } else if arg == "--address" {
            command.address = Some(args.get(index + 1).cloned().unwrap_or_default());
            index += 1;
        } else if let Some(value) = arg.strip_prefix("--address=") {
            command.address = Some(value.to_owned());
        } else if arg == "--workflow-edition" {
            command.workflow_edition = Some(args.get(index + 1).cloned().unwrap_or_default());
            index += 1;
        } else if let Some(value) = arg.strip_prefix("--workflow-edition=") {
            command.workflow_edition = Some(value.to_owned());
        } else if arg.starts_with("--") {
            // `--flag value`: skip the value unless it is itself a flag.
            if !arg.contains('=')
                && args
                    .get(index + 1)
                    .is_some_and(|next| !next.starts_with("--"))
            {
                index += 1;
            }
        } else if positional.is_none() {
            positional = Some(arg.clone());
        }
        index += 1;
    }
    if !command.env {
        command.target = positional;
    }
    command
}

fn is_shell(word: &str) -> bool {
    matches!(
        word.rsplit('/').next().unwrap_or(word),
        "sh" | "bash" | "dash" | "ash" | "zsh"
    )
}

/// A `-c`-family shell flag: a single-dash cluster containing `c` (`-c`,
/// `-lc`, `-ec`, `-euxc`).
fn is_command_flag(word: &str) -> bool {
    word.len() > 1
        && word.starts_with('-')
        && !word.starts_with("--")
        && word[1..].chars().all(|c| c.is_ascii_alphabetic())
        && word.contains('c')
}

/// Split a shell command string into words the way `sh -c` reads it, before
/// any expansion: whitespace separates words; single quotes are literal;
/// inside double quotes a backslash escapes `"`, `\`, `$`, and a backtick;
/// outside quotes a backslash escapes the next character. `$@`, `&&`, `;`,
/// and `|` are left as ordinary words. An unterminated quote runs to the end.
pub fn shell_words(script: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = script.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_word = true;
                for inner in chars.by_ref() {
                    if inner == '\'' {
                        break;
                    }
                    word.push(inner);
                }
            }
            '"' => {
                in_word = true;
                while let Some(inner) = chars.next() {
                    match inner {
                        '"' => break,
                        '\\' => match chars.peek() {
                            Some(&escaped @ ('"' | '\\' | '$' | '`')) => {
                                word.push(escaped);
                                chars.next();
                            }
                            _ => word.push('\\'),
                        },
                        other => word.push(other),
                    }
                }
            }
            '\\' => {
                in_word = true;
                if let Some(escaped) = chars.next() {
                    word.push(escaped);
                }
            }
            c if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            other => {
                in_word = true;
                word.push(other);
            }
        }
    }
    if in_word {
        words.push(word);
    }
    words
}

/// Check the serve command, the address it binds, and the port it exposes.
/// `kind` is the describe label's kind when the image carries one, so `--env`
/// can be checked against what the envelope says the image is.
pub fn check_serve_command(config: &ImageConfig, kind: Option<Kind>) -> CheckReport {
    let mut report = CheckReport::default();
    let command = parse_serve_command(config);
    if command.argv.is_empty() {
        // The platform warns (a workload can still supply a command); match it.
        report.warnings.push(
            "entrypoint: image declares no Entrypoint/Cmd; the platform cannot see what it \
             serves"
                .to_owned(),
        );
        return report;
    }
    let Some(serve) = command.serve else {
        report.not_checked.push(format!(
            "entrypoint: command `{}` does not run -m rlmesh.serve (custom entrypoint); \
             the runtime probe verifies serving",
            command.argv.join(" ")
        ));
        return report;
    };
    if command.address.is_some() {
        // A warning, as on the platform: it assigns the address per pod, and a
        // baked flag that names the same port still serves.
        report.warnings.push(format!(
            "entrypoint: --address is baked into the command; it overrides the {ADDRESS_ENV} \
             the platform assigns per pod. Drop it (rlmesh.serve binds {ADDRESS_ENV}, \
             default 0.0.0.0:{SERVE_PORT})"
        ));
    }
    match (&command.target, kind) {
        (None, _) => report
            .failed
            .push("entrypoint: rlmesh.serve names no module:Class to serve".to_owned()),
        (Some(_), Some(kind)) if (kind == Kind::Env) != command.env => {
            report.failed.push(format!(
                "entrypoint: --env present={} but the describe label says kind {}",
                command.env,
                kind.name()
            ));
        }
        (Some(_), _) => report.passed.push(format!(
            "entrypoint: {}",
            command.argv[serve.saturating_sub(2)..].join(" ")
        )),
    }
    if let Some(address) = config.env_value(ADDRESS_ENV) {
        report.warnings.push(format!(
            "env: image-level ENV {ADDRESS_ENV}={address} is set; the platform assigns it \
             per pod, so it has no effect there"
        ));
    }
    if config.exposes_port(SERVE_PORT) {
        report.passed.push(format!("ports: EXPOSE {SERVE_PORT}"));
    } else {
        report.warnings.push(format!(
            "ports: EXPOSE {SERVE_PORT} is missing; the platform connects on {SERVE_PORT} \
             regardless, add it so the intent is visible"
        ));
    }
    report
}

/// Check the image's OS/architecture against what the platform runs.
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
    /// The wire generation the image speaks (`rlmesh-wire-v1`).
    #[serde(default)]
    pub protocol_generation: Option<String>,
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
    /// The OS the envelope was generated on (`linux`); a label baked on a
    /// Mac carries `macos` and the platform fails it.
    #[serde(default)]
    pub os: Option<String>,
    /// The architecture the envelope was generated on (`x86_64` / `amd64`).
    #[serde(default)]
    pub arch: Option<String>,
}

impl RuntimeOffer {
    /// The image's bind-time offer, or `None` when the envelope advertises no
    /// editions (an older rlmesh); the runtime probe decides then.
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

/// Check the rlmesh labels' wrappers and read the describe label. A missing
/// describe label is not a failure: the platform reads describe off the
/// `rlmesh.serve` handshake instead.
pub fn check_labels(config: &ImageConfig) -> (CheckReport, Option<DescribeLabel>) {
    let mut report = CheckReport::default();
    let label = match config.labels.get(DESCRIBE_LABEL) {
        None => {
            report.not_checked.push(format!(
                "labels: no {DESCRIBE_LABEL} label; the platform reads describe off the \
                 rlmesh.serve handshake instead (run `rlmesh check <module:Class>` for the \
                 class-level checks)"
            ));
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

/// Check where a describe label was generated: the platform fails a label
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
/// `not_checked`: the runtime probe verifies the handshake.
pub fn check_editions(
    runtime: &RuntimeOffer,
    kind: Option<Kind>,
    platform: &SessionOffer,
) -> CheckReport {
    let mut report = CheckReport::default();
    match runtime.protocol_generation.as_deref().map(str::trim) {
        None | Some("") => report.not_checked.push(
            "editions: describe carries no protocol_generation (built with an rlmesh before \
             the field); the runtime probe verifies the handshake"
                .to_owned(),
        ),
        Some(generation) if generation != PROTOCOL_GENERATION => report.failed.push(format!(
            "editions: image speaks protocol generation {generation}, the platform speaks \
             {PROTOCOL_GENERATION}; rebuild against the platform's rlmesh"
        )),
        Some(_) => report.passed.push(format!(
            "editions: protocol generation {PROTOCOL_GENERATION}"
        )),
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
             before the field); the runtime probe verifies the handshake"
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

/// Run every image-config check: platform, serve command, labels, and (when
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
            }
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
            "editions: not checked without a describe label; the runtime probe negotiates \
             the handshake"
                .to_owned(),
        ),
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_proto::CURRENT_WORKFLOW_EDITION;

    fn image(entrypoint: &[&str], cmd: &[&str], env: &[&str], ports: &[&str]) -> ImageConfig {
        ImageConfig {
            os: "linux".to_owned(),
            architecture: "amd64".to_owned(),
            entrypoint: entrypoint.iter().map(|s| (*s).to_owned()).collect(),
            cmd: cmd.iter().map(|s| (*s).to_owned()).collect(),
            env: env.iter().map(|s| (*s).to_owned()).collect(),
            exposed_ports: ports.iter().map(|s| (*s).to_owned()).collect(),
            labels: BTreeMap::new(),
        }
    }

    #[test]
    fn serve_command_table() {
        struct Case {
            name: &'static str,
            entrypoint: &'static [&'static str],
            cmd: &'static [&'static str],
            env: &'static [&'static str],
            ports: &'static [&'static str],
            kind: Option<Kind>,
            failed: &'static [&'static str],
            warnings: &'static [&'static str],
            not_checked: &'static [&'static str],
        }
        let cases = [
            Case {
                name: "model image, clean",
                entrypoint: &["python", "-m", "rlmesh.serve", "pkg:Policy"],
                cmd: &[],
                env: &[],
                ports: &["50051/tcp"],
                kind: Some(Kind::Model),
                failed: &[],
                warnings: &[],
                not_checked: &[],
            },
            Case {
                name: "env image via sh -c, no EXPOSE, baked env address",
                entrypoint: &["sh", "-c", "python -m rlmesh.serve --env pkg:Env"],
                cmd: &[],
                env: &["RLMESH_ADDRESS=0.0.0.0:6000"],
                ports: &[],
                kind: Some(Kind::Env),
                failed: &[],
                warnings: &["env: image-level ENV", "ports: EXPOSE 50051 is missing"],
                not_checked: &[],
            },
            Case {
                name: "--address baked in",
                entrypoint: &["python"],
                cmd: &[
                    "-m",
                    "rlmesh.serve",
                    "pkg:Policy",
                    "--address",
                    "0.0.0.0:7000",
                ],
                env: &[],
                ports: &["50051/tcp"],
                kind: None,
                failed: &[],
                warnings: &["entrypoint: --address is baked"],
                not_checked: &[],
            },
            Case {
                name: "--env on a model label",
                entrypoint: &["python", "-m", "rlmesh.serve", "--env=pkg:Env"],
                cmd: &[],
                env: &[],
                ports: &["50051/tcp"],
                kind: Some(Kind::Model),
                failed: &["--env present=true but the describe label says kind model"],
                warnings: &[],
                not_checked: &[],
            },
            Case {
                name: "no target",
                entrypoint: &[
                    "python",
                    "-m",
                    "rlmesh.serve",
                    "--workflow-edition",
                    "2026.06",
                ],
                cmd: &[],
                env: &[],
                ports: &["50051/tcp"],
                kind: None,
                failed: &["names no module:Class"],
                warnings: &[],
                not_checked: &[],
            },
            Case {
                name: "custom entrypoint",
                entrypoint: &["python", "serve.py"],
                cmd: &[],
                env: &[],
                ports: &["50051/tcp"],
                kind: None,
                failed: &[],
                warnings: &[],
                not_checked: &["custom entrypoint"],
            },
            Case {
                name: "no command at all",
                entrypoint: &[],
                cmd: &[],
                env: &[],
                ports: &[],
                kind: None,
                failed: &[],
                warnings: &["no Entrypoint/Cmd"],
                not_checked: &[],
            },
        ];
        for case in cases {
            let config = image(case.entrypoint, case.cmd, case.env, case.ports);
            let report = check_serve_command(&config, case.kind);
            for (bucket, expected) in [
                (&report.failed, case.failed),
                (&report.warnings, case.warnings),
                (&report.not_checked, case.not_checked),
            ] {
                assert_eq!(bucket.len(), expected.len(), "{}: {report:?}", case.name);
                for (message, needle) in bucket.iter().zip(expected) {
                    assert!(message.contains(needle), "{}: {message}", case.name);
                }
            }
        }
    }

    #[test]
    fn exec_form_keeps_docker_argument_boundaries() {
        // A JSON value with spaces is one word; the first bare word is the target.
        let config = image(
            &["python", "-m", "rlmesh.serve"],
            &["pkg:Policy", "--kwargs-json", "{\"a\": 1}"],
            &[],
            &[],
        );
        let command = parse_serve_command(&config);
        assert_eq!(command.serve, Some(2));
        assert_eq!(command.target.as_deref(), Some("pkg:Policy"));
        assert_eq!(command.argv.len(), 6);
        // A whitespace-bearing exec-form element is NOT a shell script: it stays one word.
        let config = image(&["python -m rlmesh.serve pkg:Policy"], &[], &[], &[]);
        assert_eq!(parse_serve_command(&config).serve, None);
    }

    #[test]
    fn shell_form_splits_the_script_and_appends_the_rest() {
        let config = image(
            &[
                "/bin/sh",
                "-c",
                "uv run python -m rlmesh.serve --env 'pkg:Env' --framework torch",
            ],
            &[],
            &[],
            &[],
        );
        let command = parse_serve_command(&config);
        assert_eq!(command.serve, Some(4));
        assert!(command.env);
        assert_eq!(command.target.as_deref(), Some("pkg:Env"));
        assert_eq!(command.kind(), Some(Kind::Env));
        assert_eq!(command.address, None);
        // `bash -c 'exec python "$@"' --` hands Cmd through unchanged.
        let config = image(
            &[
                "bash",
                "-c",
                "source /sim/setup.sh && exec /sim/bin/python3 \"$@\"",
                "--",
            ],
            &["-m", "rlmesh.serve", "--env", "franka_reach:FrankaReach"],
            &[],
            &[],
        );
        let command = parse_serve_command(&config);
        assert_eq!(command.target.as_deref(), Some("franka_reach:FrankaReach"));
        assert!(command.env);
    }

    #[test]
    fn shell_words_honor_posix_quoting() {
        assert_eq!(
            shell_words(
                r#"python -m rlmesh.serve pkg:Policy --kwargs-json '{"a": 1}' --x "a b" c\ d "$@""#
            ),
            [
                "python",
                "-m",
                "rlmesh.serve",
                "pkg:Policy",
                "--kwargs-json",
                r#"{"a": 1}"#,
                "--x",
                "a b",
                "c d",
                "$@"
            ]
        );
        assert_eq!(
            shell_words(r#""esc \" \\ \$x" 'lit \n'"#),
            ["esc \" \\ $x", "lit \\n"]
        );
        assert_eq!(shell_words("  "), Vec::<String>::new());
        assert_eq!(shell_words("''"), [""]);
    }

    #[test]
    fn platform_must_be_linux_amd64() {
        let mut config = image(&["python"], &[], &[], &[]);
        assert!(check_platform(&config).ok());
        config.architecture = "arm64".to_owned();
        let report = check_platform(&config);
        assert!(report.failed[0].contains("linux/arm64"), "{report:?}");
        config.architecture.clear();
        assert_eq!(check_platform(&config).not_checked.len(), 1);
    }

    #[test]
    fn docker_inspect_parses_nulls_and_ports() {
        let json = r#"[{"Os":"linux","Architecture":"amd64","Config":{"Entrypoint":null,
            "Cmd":["python","-m","rlmesh.serve","pkg:Policy"],"Env":["PATH=/usr/bin"],
            "ExposedPorts":{"50051/tcp":{}},"Labels":null}}]"#;
        let config = ImageConfig::from_docker_inspect(json).unwrap();
        assert_eq!(config.cmd.len(), 4);
        assert!(config.entrypoint.is_empty());
        assert!(config.exposes_port("50051"));
        assert_eq!(config.env_value("PATH"), Some("/usr/bin"));
        assert!(ImageConfig::from_docker_inspect("[]").is_err());
    }

    fn label(kind: &str, runtime: serde_json::Value) -> String {
        serde_json::json!({
            "schema_version": 1,
            "kind": kind,
            "target": {"entrypoint": "pkg:Policy", "qualname": "pkg:Policy"},
            "runtime": runtime,
        })
        .to_string()
    }

    #[test]
    fn describe_label_wrapper_is_read() {
        let parsed = parse_describe_label(&label(
            "model",
            serde_json::json!({"supported_workflow_editions": ["2026.06"]}),
        ))
        .unwrap();
        assert_eq!(parsed.kind, Some(Kind::Model));
        assert_eq!(parsed.target.as_deref(), Some("pkg:Policy"));
        assert_eq!(
            parsed.runtime.session_offer(),
            Some(SessionOffer::new(&["2026.06"]))
        );
        assert!(parse_describe_label("{nope").is_err());
        assert!(parse_describe_label(r#"{"schema_version":2,"kind":"model"}"#).is_err());
        assert!(parse_describe_label(r#"{"schema_version":1,"kind":"thing"}"#).is_err());
        assert!(
            parse_describe_label(r#"{"schema_version":1,"kind":"env"}"#)
                .unwrap()
                .runtime
                .session_offer()
                .is_none()
        );
    }

    #[test]
    fn missing_describe_label_is_a_note_not_a_failure() {
        let config = image(
            &["python", "-m", "rlmesh.serve", "pkg:Policy"],
            &[],
            &[],
            &[],
        );
        let (report, label) = check_labels(&config);
        assert!(label.is_none());
        assert!(report.ok());
        assert!(report.not_checked[0].contains("handshake"), "{report:?}");
    }

    #[test]
    fn edition_table() {
        let platform = platform_offer(&["2026.06-0.1.0-rc.13".to_owned(), "2026.06".to_owned()]);
        let offer =
            |generation: Option<&str>, editions: &[&str], preferred: Option<&str>| RuntimeOffer {
                protocol_generation: generation.map(str::to_owned),
                supported_workflow_editions: editions.iter().map(|e| (*e).to_owned()).collect(),
                preferred_workflow_edition: preferred.map(str::to_owned),
                ..RuntimeOffer::default()
            };
        // Sealed base in common: a newer or older prerelease still meets the platform.
        let report = check_editions(
            &offer(
                Some(PROTOCOL_GENERATION),
                &["2026.06-0.1.0-rc.12", "2026.06"],
                Some("2026.06"),
            ),
            Some(Kind::Model),
            &platform,
        );
        assert!(report.ok(), "{report:?}");
        assert!(report.passed.iter().any(|m| m.contains("run at 2026.06")));
        // Nothing in common: refused, naming both sides.
        let report = check_editions(
            &offer(Some(PROTOCOL_GENERATION), &["2031.01"], None),
            Some(Kind::Env),
            &platform,
        );
        assert_eq!(report.failed.len(), 1, "{report:?}");
        assert!(report.failed[0].contains("share no workflow edition"));
        // Wrong generation is a failure even when editions agree.
        let report = check_editions(
            &offer(Some("rlmesh-wire-v2"), &["2026.06"], None),
            None,
            &platform,
        );
        assert!(
            report.failed[0].contains("protocol generation"),
            "{report:?}"
        );
        // No fields at all: not checked, twice (generation and editions).
        let report = check_editions(&offer(None, &[], None), None, &platform);
        assert!(report.ok());
        assert_eq!(report.not_checked.len(), 2, "{report:?}");
        // A declaration the image's own rlmesh refused: fails without Python.
        let mut impossible = offer(Some(PROTOCOL_GENERATION), &["2026.06"], Some("2026.06"));
        impossible.workflow_edition_error = Some("cannot drive 1999.01".to_owned());
        let report = check_editions(&impossible, None, &platform);
        assert!(
            report.failed.iter().any(|m| m.contains("refuse to start")),
            "{report:?}"
        );
    }

    #[test]
    fn platform_offer_for_version_knows_only_this_build() {
        let own = platform_offer_for_version(env!("CARGO_PKG_VERSION")).unwrap();
        assert_eq!(own.editions, supported_workflow_editions());
        // PyPI's spelling of the same build.
        let pep440 = env!("CARGO_PKG_VERSION").replace("-rc.", "rc");
        assert_eq!(platform_offer_for_version(&pep440), Some(own));
        // Another release's retained list is not this build's to guess: rc.12
        // offered only its cohort, so adding this build's sealed base would
        // pass an image the real handshake refuses.
        assert_eq!(platform_offer_for_version("0.1.0-rc.12"), None);
        assert_eq!(platform_offer_for_version("0.1.0"), None);
    }

    #[test]
    fn version_spellings_normalize() {
        for (raw, expected) in [
            ("0.1.0-rc.13", "0.1.0-rc.13"),
            ("0.1.0rc13", "0.1.0-rc.13"),
            ("0.1.0-rc13", "0.1.0-rc.13"),
            ("0.1.0.rc.13", "0.1.0-rc.13"),
            ("v0.1.0RC13", "0.1.0-rc.13"),
            ("0.1.0", "0.1.0"),
            ("0.1.0a1", "0.1.0-a.1"),
            ("0.1.0-alpha.1", "0.1.0-a.1"),
            ("0.1.0-beta.3", "0.1.0-b.3"),
            ("0.1.0-dev.abc123", "0.1.0-dev.abc123"),
        ] {
            assert_eq!(normalize_version(raw), expected, "{raw}");
        }
    }

    #[test]
    fn host_baked_label_is_refused() {
        let runtime = |os: &str, arch: &str| RuntimeOffer {
            os: Some(os.to_owned()),
            arch: Some(arch.to_owned()),
            ..RuntimeOffer::default()
        };
        let report = check_label_runtime(&runtime("macos", "arm64"));
        assert!(
            report.failed[0].contains("generated on macos"),
            "{report:?}"
        );
        assert!(report.warnings[0].contains("arm64"), "{report:?}");
        assert!(check_label_runtime(&runtime("linux", "x86_64")).ok());
        assert!(
            check_label_runtime(&runtime("linux", "amd64"))
                .warnings
                .is_empty()
        );
        // An older envelope without the fields is not judged.
        assert_eq!(
            check_label_runtime(&RuntimeOffer::default()),
            CheckReport::default()
        );
        // And it is wired into check_image off the label's runtime block.
        let mut config = image(
            &["python", "-m", "rlmesh.serve", "pkg:Policy"],
            &[],
            &[],
            &[],
        );
        config.labels.insert(
            DESCRIBE_LABEL.to_owned(),
            label("model", serde_json::json!({"os": "macos", "arch": "arm64"})),
        );
        let report = check_image(&config, None);
        assert!(
            report
                .failed
                .iter()
                .any(|m| m.contains("generated on macos")),
            "{report:?}"
        );
    }

    #[test]
    fn image_declared_edition_overrides_the_label() {
        let mut config = image(
            &[
                "python",
                "-m",
                "rlmesh.serve",
                "pkg:Policy",
                "--workflow-edition",
                "2020.01",
            ],
            &[],
            &[],
            &["50051/tcp"],
        );
        config.labels.insert(
            DESCRIBE_LABEL.to_owned(),
            label(
                "model",
                serde_json::json!({
                    "protocol_generation": PROTOCOL_GENERATION,
                    "supported_workflow_editions": ["2026.06-0.1.0-rc.13", "2026.06"],
                    "preferred_workflow_edition": "2026.06",
                }),
            ),
        );
        let platform = platform_offer(&["2026.06".to_owned()]);
        // The flag is a ceiling below the one edition both can drive: refused,
        // even though the label alone would have passed.
        assert_eq!(
            declared_workflow_edition(&config),
            Some(Some("2020.01".to_owned()))
        );
        let report = check_image(&config, Some(&platform));
        assert!(
            report
                .failed
                .iter()
                .any(|m| m.contains("share no workflow edition")),
            "{report:?}"
        );
        // An empty ENV declares none: the peer floats to its newest, and a
        // sealed base in common lets the session run.
        config.env.push(format!("{WORKFLOW_EDITION_ENV}="));
        assert_eq!(declared_workflow_edition(&config), Some(None));
        let report = check_image(&config, Some(&platform));
        assert!(report.ok(), "{report:?}");
        assert!(
            report
                .passed
                .iter()
                .any(|m| m.contains("floating to its newest")),
            "{report:?}"
        );
        // A non-empty ENV wins over the flag.
        config.env.pop();
        config.env.push(format!("{WORKFLOW_EDITION_ENV}=2026.06"));
        assert_eq!(
            declared_workflow_edition(&config),
            Some(Some("2026.06".to_owned()))
        );
        assert!(check_image(&config, Some(&platform)).ok());
    }

    #[test]
    fn check_image_ties_the_label_to_the_command() {
        let mut config = image(
            &["python", "-m", "rlmesh.serve", "pkg:Other"],
            &[],
            &[],
            &["50051/tcp"],
        );
        config.labels.insert(
            DESCRIBE_LABEL.to_owned(),
            label(
                "model",
                serde_json::json!({
                    "protocol_generation": PROTOCOL_GENERATION,
                    "supported_workflow_editions": supported_workflow_editions(),
                    "preferred_workflow_edition": CURRENT_WORKFLOW_EDITION,
                }),
            ),
        );
        let report = check_image(&config, Some(&SessionOffer::this_build(None)));
        assert_eq!(report.failed.len(), 1, "{report:?}");
        assert!(report.failed[0].contains("serves \"pkg:Other\""));
        config.cmd.clear();
        config.entrypoint[3] = "pkg:Policy".to_owned();
        let report = check_image(&config, Some(&SessionOffer::this_build(None)));
        assert!(report.ok(), "{report:?}");
        assert!(report.not_checked.is_empty(), "{report:?}");
    }
}
