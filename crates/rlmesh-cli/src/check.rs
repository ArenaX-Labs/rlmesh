//! `rlmesh check`, `rlmesh check-image`, and `rlmesh describe`: the pre-push
//! checks. The image-config checks are the pure functions in
//! [`crate::image_check`]; everything that needs the Python gatherer (describe
//! envelopes, class-level checks, the deep label check) shells out to
//! `python -m rlmesh._describe`, the same bridge a bake uses. `RLMESH_PYTHON`
//! names the interpreter (the `python -m rlmesh` entrypoint sets it to its
//! own), else `python3`, else `python`.

use crate::cli::{CheckArgs, CheckImageArgs, DescribeArgs};
use crate::image_check::{self, CheckReport, DESCRIBE_LABEL, ImageConfig, PACKAGE_LABEL};
use crate::render::Style;

use anyhow::{Context, Result, bail};
use rlmesh_proto::SessionOffer;
use std::ffi::OsString;
use std::io::Write;
use std::process::{Command, Stdio};

const DESCRIBE_MODULE: &str = "rlmesh._describe";

pub(crate) fn check_image(
    args: &CheckImageArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let config = docker_inspect(&args.image)?;
    let (platform, against) = platform_offer(args)?;
    let mut report = image_check::check_image(&config, platform.as_ref());
    // The deep label checks (the envelope's class-level checks, the package
    // label's checkpoints) live in rlmesh._describe; run them whenever either
    // label is present, and say which validation was skipped when no Python
    // with rlmesh is reachable.
    let present: Vec<&str> = [DESCRIBE_LABEL, PACKAGE_LABEL]
        .into_iter()
        .filter(|label| config.labels.contains_key(*label))
        .collect();
    if !present.is_empty() {
        let labels = serde_json::to_string(&config.labels)?;
        match run_python_report(&["--check-labels", "-", "--json"], Some(&labels)) {
            Ok(deep) => merge_deep(&mut report, deep),
            Err(error) => report.not_checked.push(format!(
                "labels: {} not validated ({error:#}); install rlmesh in the Python on PATH \
                 or set RLMESH_PYTHON",
                present
                    .iter()
                    .map(|label| match *label {
                        PACKAGE_LABEL => format!("{PACKAGE_LABEL} checkpoints"),
                        other => format!("{other} contents"),
                    })
                    .collect::<Vec<_>>()
                    .join(" and ")
            )),
        }
    }
    render(
        &report,
        &format!("{} against {against}", args.image),
        args.json,
        stdout,
        style,
    )
}

/// Fold the Python label report into the image report without saying things
/// twice: Python re-reports the label wrappers (JSON, `schema_version`,
/// `kind`, the package label's shape), so the Rust findings on those go; the
/// Rust side owns the edition verdict in `check-image` (it applies the
/// image's own `--workflow-edition` and the platform offer), so Python's
/// `runtime:` lines go. Everything else from both sides is kept.
fn merge_deep(report: &mut CheckReport, mut deep: CheckReport) {
    let rust_wrapper = |message: &String| message.starts_with("labels: dev.rlmesh.");
    report.failed.retain(|message| !rust_wrapper(message));
    report.warnings.retain(|message| !rust_wrapper(message));
    let python_runtime = |message: &String| {
        message
            .strip_prefix(DESCRIBE_LABEL)
            .is_some_and(|rest| rest.starts_with(" runtime:"))
    };
    deep.failed.retain(|message| !python_runtime(message));
    deep.warnings.retain(|message| !python_runtime(message));
    // The missing-describe-label note is Rust's too (a package-only image
    // still gets the deep check for its checkpoints).
    deep.not_checked.retain(|message| {
        !python_runtime(message) && !message.starts_with(&format!("no {DESCRIBE_LABEL} label"))
    });
    deep.passed.retain(|message| !python_runtime(message));
    report.extend(deep);
}

pub(crate) fn check_target(args: &CheckArgs, stdout: &mut impl Write, style: Style) -> Result<i32> {
    let report = run_python_report(&["--check-entrypoint", &args.target, "--json"], None)?;
    render(&report, &args.target, args.json, stdout, style)
}

pub(crate) fn describe(
    args: &DescribeArgs,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> Result<i32> {
    describe_with(python_command()?, &args.target, args.label, stdout, stderr)
}

/// Pass `python -m rlmesh._describe TARGET [--label]` through: its stdout,
/// its stderr, and its exit code are the command's.
fn describe_with(
    mut python: Command,
    target: &str,
    label: bool,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> Result<i32> {
    let mut argv = vec![target];
    if label {
        argv.push("--label");
    }
    let output = python
        .arg("-m")
        .arg(DESCRIBE_MODULE)
        .args(&argv)
        .stdin(Stdio::null())
        .output()
        .context("running python -m rlmesh._describe")?;
    stdout.write_all(&output.stdout)?;
    stderr.write_all(&output.stderr)?;
    Ok(output.status.code().unwrap_or(1))
}

/// The platform offer to check editions against: `--platform-editions`, else
/// `--platform-version` (only this CLI's own version is known; any other is
/// `None`, reported as not checked), else the rlmesh this CLI was built with.
fn platform_offer(args: &CheckImageArgs) -> Result<(Option<SessionOffer>, String)> {
    if !args.platform_editions.is_empty() {
        let offer = image_check::platform_offer(&args.platform_editions);
        if offer.editions.is_empty() {
            bail!("--platform-editions names no edition");
        }
        let against = format!("platform editions {:?}", offer.editions);
        return Ok((Some(offer), against));
    }
    if let Some(version) = &args.platform_version {
        let offer = image_check::platform_offer_for_version(version);
        let against = match &offer {
            Some(_) => format!("a platform on rlmesh {} (this CLI's build)", version.trim()),
            None => format!(
                "a platform on rlmesh {} (not this CLI's {}, editions unknown)",
                version.trim(),
                env!("CARGO_PKG_VERSION")
            ),
        };
        return Ok((offer, against));
    }
    Ok((
        Some(SessionOffer::this_build(None)),
        format!("this CLI's rlmesh {}", env!("CARGO_PKG_VERSION")),
    ))
}

fn docker_inspect(image: &str) -> Result<ImageConfig> {
    let output = Command::new("docker")
        .args(["image", "inspect", image])
        .stdin(Stdio::null())
        .output()
        .context("running docker image inspect (is docker installed?)")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        bail!(
            "docker image inspect {image} failed: {}",
            if stderr.is_empty() {
                "is docker running?"
            } else {
                stderr
            }
        );
    }
    ImageConfig::from_docker_inspect(&String::from_utf8_lossy(&output.stdout))
        .map_err(anyhow::Error::msg)
}

/// Run `python -m rlmesh._describe <args>` and parse the JSON report it prints
/// on stdout. The exit code is not consulted: a report with failures exits
/// nonzero there and is still a report here.
fn run_python_report(args: &[&str], stdin: Option<&str>) -> Result<CheckReport> {
    run_python_report_with(python_command()?, args, stdin)
}

fn run_python_report_with(
    mut command: Command,
    args: &[&str],
    stdin: Option<&str>,
) -> Result<CheckReport> {
    command.arg("-m").arg(DESCRIBE_MODULE).args(args);
    let output = match stdin {
        Some(input) => {
            let mut child = command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .context("running python -m rlmesh._describe")?;
            child
                .stdin
                .take()
                .context("python stdin")?
                .write_all(input.as_bytes())?;
            child.wait_with_output()?
        }
        None => command
            .stdin(Stdio::null())
            .output()
            .context("running python -m rlmesh._describe")?,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).with_context(|| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        format!(
            "python -m rlmesh._describe {} exited {} without a report{}",
            args.join(" "),
            output.status.code().unwrap_or(-1),
            tail(stderr.trim()),
        )
    })
}

fn tail(stderr: &str) -> String {
    if stderr.is_empty() {
        return String::new();
    }
    let lines: Vec<&str> = stderr.lines().collect();
    let start = lines.len().saturating_sub(12);
    format!(":\n{}", lines[start..].join("\n"))
}

/// The interpreter carrying the rlmesh package: `RLMESH_PYTHON`, else the
/// first of `python3` / `python` that imports `rlmesh`.
fn python_command() -> Result<Command> {
    python_command_from(std::env::var_os("RLMESH_PYTHON"), imports_rlmesh)
}

/// The discovery rule behind [`python_command`]: a non-empty `configured`
/// (`RLMESH_PYTHON`) is taken as is, unprobed; otherwise the first candidate
/// on PATH for which `imports_rlmesh` holds.
fn python_command_from(
    configured: Option<OsString>,
    imports_rlmesh: impl Fn(&str) -> bool,
) -> Result<Command> {
    if let Some(python) = configured.filter(|value| !value.is_empty()) {
        return Ok(Command::new(python));
    }
    for candidate in ["python3", "python"] {
        if imports_rlmesh(candidate) {
            return Ok(Command::new(candidate));
        }
    }
    bail!(
        "no python with the rlmesh package on PATH (tried python3, python); set RLMESH_PYTHON \
         to the interpreter that has it"
    )
}

fn imports_rlmesh(candidate: &str) -> bool {
    Command::new(candidate)
        .arg("-c")
        .arg("import rlmesh")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Print a report: one line per finding, worst first, then a one-line
/// summary. Exit 1 when anything failed.
fn render(
    report: &CheckReport,
    subject: &str,
    json: bool,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    if json {
        writeln!(stdout, "{}", serde_json::to_string_pretty(report)?)?;
        return Ok(i32::from(!report.ok()));
    }
    for message in &report.failed {
        writeln!(stdout, "{}  {message}", style.red_bold("FAIL"))?;
    }
    for message in &report.warnings {
        writeln!(stdout, "{}  {message}", style.yellow("warn"))?;
    }
    for message in &report.not_checked {
        writeln!(stdout, "{}  {message}", style.muted("skip"))?;
    }
    for message in &report.passed {
        writeln!(stdout, "{}  {message}", style.green("ok  "))?;
    }
    let summary = format!(
        "{} failed, {} warnings, {} not checked",
        report.failed.len(),
        report.warnings.len(),
        report.not_checked.len()
    );
    if report.ok() {
        writeln!(
            stdout,
            "{}",
            style.success(&format!("{subject}: {summary}"))
        )?;
        Ok(0)
    } else {
        writeln!(stdout, "{} {subject}: {summary}", style.red_bold("✗"))?;
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in interpreter: a shell script that behaves like
    /// `python -m rlmesh._describe` would for the test at hand.
    #[cfg(unix)]
    fn fake_python(dir: &tempfile::TempDir, body: &str) -> Command {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.path().join("python");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        Command::new(path)
    }

    #[test]
    fn interpreter_discovery_prefers_rlmesh_python_then_probes_path() {
        let configured = python_command_from(Some(OsString::from("/opt/venv/bin/python")), |_| {
            panic!("configured interpreter is not probed")
        })
        .unwrap();
        assert_eq!(configured.get_program(), "/opt/venv/bin/python");
        // Empty RLMESH_PYTHON means unset.
        let probed = python_command_from(Some(OsString::new()), |c| c == "python").unwrap();
        assert_eq!(probed.get_program(), "python");
        let first = python_command_from(None, |_| true).unwrap();
        assert_eq!(first.get_program(), "python3");
        let error = python_command_from(None, |_| false).unwrap_err();
        assert!(error.to_string().contains("RLMESH_PYTHON"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn describe_passes_python_output_and_exit_code_through() {
        let dir = tempfile::tempdir().unwrap();
        let python = fake_python(&dir, r#"echo "args: $*"; echo "banner" >&2; exit 3"#);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = describe_with(python, "pkg:Policy", true, &mut out, &mut err).unwrap();
        assert_eq!(code, 3);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "args: -m rlmesh._describe pkg:Policy --label\n"
        );
        assert_eq!(String::from_utf8(err).unwrap(), "banner\n");
    }

    #[cfg(unix)]
    #[test]
    fn python_report_is_parsed_from_stdout_regardless_of_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        // Echo stdin back inside the report so the labels round trip is visible.
        let python = fake_python(
            &dir,
            r#"read input; printf '{"failed":["x: %s"],"warnings":[],"not_checked":[],"passed":[]}
' "$input"; exit 1"#,
        );
        let report =
            run_python_report_with(python, &["--check-labels", "-", "--json"], Some("labels\n"))
                .unwrap();
        assert_eq!(report.failed, ["x: labels"]);
        // No JSON on stdout: an error that carries the tail of stderr.
        let python = fake_python(&dir, r#"echo "Traceback" >&2; exit 1"#);
        let error =
            run_python_report_with(python, &["--check-entrypoint", "pkg:X"], None).unwrap_err();
        let text = format!("{error:#}");
        assert!(
            text.contains("exited 1 without a report") && text.contains("Traceback"),
            "{text}"
        );
    }

    #[test]
    fn platform_offer_defaults_to_this_build() {
        let args = |editions: &[&str], version: Option<&str>| CheckImageArgs {
            image: "img:tag".to_owned(),
            platform_editions: editions.iter().map(|e| (*e).to_owned()).collect(),
            platform_version: version.map(str::to_owned),
            json: false,
        };
        let (offer, against) = platform_offer(&args(&[], None)).unwrap();
        assert_eq!(offer, Some(SessionOffer::this_build(None)));
        assert!(against.contains(env!("CARGO_PKG_VERSION")), "{against}");
        let (offer, _) = platform_offer(&args(&["2026.06", " "], None)).unwrap();
        assert_eq!(offer, Some(SessionOffer::new(&["2026.06"])));
        assert!(platform_offer(&args(&[" "], None)).is_err());
        let (offer, against) = platform_offer(&args(&[], Some("0.0.1"))).unwrap();
        assert_eq!(offer, None);
        assert!(against.contains("editions unknown"), "{against}");
        let (offer, _) = platform_offer(&args(&[], Some(env!("CARGO_PKG_VERSION")))).unwrap();
        assert_eq!(
            offer.map(|o| o.editions),
            Some(rlmesh_proto::supported_workflow_editions())
        );
    }

    #[test]
    fn merge_deep_drops_what_the_other_side_already_said() {
        let mut report = CheckReport {
            failed: vec![
                "labels: dev.rlmesh.package schemaVersion 2 is not the supported version 1"
                    .to_owned(),
                "labels: the describe label was generated on macos".to_owned(),
            ],
            not_checked: vec![
                "editions: describe carries no supported_workflow_editions".to_owned(),
            ],
            ..CheckReport::default()
        };
        let deep = CheckReport {
            failed: vec![
                "dev.rlmesh.package schemaVersion 2 is not the supported version 1".to_owned(),
            ],
            warnings: vec![
                "dev.rlmesh.describe model_spec: x declares no class-level spec".to_owned(),
            ],
            not_checked: vec![
                "dev.rlmesh.describe runtime: describe carries no workflow editions".to_owned(),
                "no dev.rlmesh.describe label: the platform reads describe off the handshake"
                    .to_owned(),
            ],
            passed: vec!["dev.rlmesh.describe runtime: rlmesh-wire-v1".to_owned()],
        };
        merge_deep(&mut report, deep);
        assert_eq!(
            report.failed,
            [
                "labels: the describe label was generated on macos",
                "dev.rlmesh.package schemaVersion 2 is not the supported version 1",
            ]
        );
        assert_eq!(report.warnings.len(), 1);
        assert_eq!(report.not_checked.len(), 1, "{report:?}");
        assert!(report.passed.is_empty(), "{report:?}");
    }

    #[test]
    fn render_lists_every_bucket_and_exits_on_failures() {
        let report = CheckReport {
            failed: vec!["entrypoint: bad".to_owned()],
            warnings: vec!["ports: none".to_owned()],
            not_checked: vec!["editions: later".to_owned()],
            passed: vec!["platform: linux/amd64".to_owned()],
        };
        let mut out = Vec::new();
        let code = render(
            &report,
            "img:tag",
            false,
            &mut out,
            Style::for_terminal(false),
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(code, 1);
        assert!(text.contains("FAIL  entrypoint: bad"), "{text}");
        assert!(text.contains("warn  ports: none"), "{text}");
        assert!(text.contains("skip  editions: later"), "{text}");
        assert!(text.contains("ok    platform: linux/amd64"), "{text}");
        assert!(
            text.contains("img:tag: 1 failed, 1 warnings, 1 not checked"),
            "{text}"
        );

        let mut out = Vec::new();
        let code = render(
            &CheckReport::default(),
            "x",
            true,
            &mut out,
            Style::for_terminal(false),
        )
        .unwrap();
        assert_eq!(code, 0);
        let parsed: CheckReport = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed, CheckReport::default());
    }
}
