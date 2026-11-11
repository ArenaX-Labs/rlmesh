//! The serve command: parsing an image's entrypoint and command into the serve
//! invocation, and checking the address it binds and the ports it exposes.

use crate::image_check::{ADDRESS_ENV, CheckReport, ImageConfig, Kind, SERVE_PORT, SESSION_PORTS};

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
/// describe label is classified. The served peer's handshake supplies the
/// rest of its contract. Parsing follows these rules:
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

/// The base port an address (`host:port[/...]`, `:port`) names, if any.
fn address_port(address: &str) -> Option<u16> {
    let host = address.split(['/', '#']).next()?;
    host.rsplit_once(':')?.1.parse().ok()
}

/// The ports a session host on `base` listens on (fewer when they would run
/// past the last port).
fn session_ports(base: u16) -> Vec<u16> {
    (0..SESSION_PORTS)
        .filter_map(|offset| base.checked_add(offset))
        .collect()
}

/// Check the serve command, the address it binds, and the ports it exposes.
/// `kind` is the describe label's kind when the image carries one, so `--env`
/// can be checked against what the envelope says the image is.
pub fn check_serve_command(config: &ImageConfig, kind: Option<Kind>) -> CheckReport {
    let mut report = CheckReport::default();
    let command = parse_serve_command(config);
    if command.argv.is_empty() {
        // A workload can still supply a command, so this is a warning.
        report.warnings.push(
            "entrypoint: image declares no Entrypoint/Cmd; the platform cannot see what it \
             serves"
                .to_owned(),
        );
        return report;
    }
    let Some(serve) = command.serve else {
        report.not_checked.push(format!(
            "entrypoint: command `{}` does not run -m rlmesh.serve (custom entrypoint, e.g. \
             a native C/C++ host); the managed platform checks it when you onboard the image",
            command.argv.join(" ")
        ));
        return report;
    };
    if command.address.is_some() {
        // A warning: the address can be assigned at launch, and a
        // baked flag that names the same port still serves.
        report.warnings.push(format!(
            "entrypoint: --address is baked into the command; it overrides the {ADDRESS_ENV} \
             a launcher sets per container. Drop it (rlmesh.serve hosts its session on \
             {ADDRESS_ENV}, default 0.0.0.0:{SERVE_PORT})"
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
            "env: image-level ENV {ADDRESS_ENV}={address} is set; a launcher sets it per \
             container, which overrides it"
        ));
    }
    let base = command
        .address
        .as_deref()
        .or(config.env_value(ADDRESS_ENV))
        .and_then(address_port)
        .unwrap_or(SERVE_PORT);
    let ports = session_ports(base);
    let missing: Vec<u16> = ports
        .iter()
        .copied()
        .filter(|port| !config.exposes_port(*port))
        .collect();
    let range = format!("{base}-{}", base.saturating_add(SESSION_PORTS - 1));
    if missing.is_empty() {
        report.passed.push(format!("ports: EXPOSE {range}"));
    } else {
        let missing: Vec<String> = missing.iter().map(u16::to_string).collect();
        report.warnings.push(format!(
            "ports: EXPOSE {range} is incomplete (missing {}); rlmesh.serve hosts its session \
             on its base port {base} and the next {} (one per traffic class), and participants \
             dial HOST:{base}#KEY/NAME on all of them. A launcher publishes them regardless; \
             add `EXPOSE {range}` so the intent is visible",
            missing.join(", "),
            SESSION_PORTS - 1
        ));
    }
    report
}
