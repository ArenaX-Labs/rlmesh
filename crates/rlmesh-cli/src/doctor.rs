//! `rlmesh doctor`: run an image on this machine and probe
//! it with `rlmesh::binding::doctor`, the platform probe's checks.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use rlmesh::binding::doctor::{
    self, CheckStatus, EnvProbeOptions, ModelProbeOptions, PROBE_REPORT, ProbeCheck, ProbeReport,
    ProbeSubject,
};

use crate::check::docker_inspect;
use crate::cli::{DoctorArgs, ProbeImageArgs, ProbeKind};
use crate::container::Container;
use crate::image_check::{self, DESCRIBE_LABEL, ImageConfig, Kind};
use crate::render::Style;

pub(crate) fn doctor(args: &DoctorArgs, stdout: &mut impl Write, style: Style) -> Result<i32> {
    let config = docker_inspect(&args.image.image)?;
    let describe = describe_label(&config);
    let kind = args.kind.unwrap_or_else(|| image_kind(&config));
    let output = args.output.clone();
    std::fs::create_dir_all(&output)
        .with_context(|| format!("create the output directory {}", output.display()))?;
    let image = &args.image;
    let mut report = off_runtime(|| match kind {
        ProbeKind::Env => probe_env(image, describe, &output),
        ProbeKind::Model => probe_model(image, describe),
    })?;
    report.checks.insert(0, image_config_check(&config));
    if kind == ProbeKind::Env {
        report.env.repo = image.image.clone();
    }
    let report_path = output.join(PROBE_REPORT);
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    if image.json {
        writeln!(stdout, "{}", serde_json::to_string_pretty(&report)?)?;
    } else {
        render_checks(&report, &image.image, stdout, style)?;
        writeln!(
            stdout,
            "{}  {}",
            style.muted("report "),
            report_path.display()
        )?;
        if let Some(preview) = &report.preview {
            writeln!(
                stdout,
                "{}  {}",
                style.muted("preview"),
                output.join(&preview.video).display()
            )?;
        }
    }
    Ok(i32::from(failed(&report)))
}

/// The probe of an env image, served in a container of its own.
fn probe_env(
    image: &ProbeImageArgs,
    describe: Option<serde_json::Value>,
    output: &Path,
) -> Result<ProbeReport> {
    let env = start(&image.image, "env", image)?;
    let env = match env {
        Ok(env) => env,
        Err(error) => return Ok(doctor::failed_handshake(ProbeSubject::Environment, &error)),
    };
    Ok(doctor::probe_env(
        &env.address,
        &EnvProbeOptions {
            describe,
            output_dir: Some(output.to_path_buf()),
            connect_timeout: Duration::from_secs(image.timeout),
        },
    ))
}

/// The probe of a model image, beside its reference env when one is given.
fn probe_model(image: &ProbeImageArgs, describe: Option<serde_json::Value>) -> Result<ProbeReport> {
    let (model, reference) = start_model(image)?;
    let model = match model {
        Ok(model) => model,
        Err(error) => return Ok(doctor::failed_handshake(ProbeSubject::Model, &error)),
    };
    let mut report = doctor::probe_model(
        &model.address,
        &ModelProbeOptions {
            describe,
            reference_env: reference.as_ref().map(|env| env.address.clone()),
            connect_timeout: Duration::from_secs(image.timeout),
        },
    );
    if reference.is_some() {
        report.env.repo = image.env.clone().unwrap_or_default();
    }
    Ok(report)
}

/// The model container and, with `--env`, its reference env's. A model that
/// does not come up is the report's failed handshake; an env that does not is
/// an error, since nothing about the model was checked.
fn start_model(image: &ProbeImageArgs) -> Result<(Result<Container, String>, Option<Container>)> {
    let reference =
        match &image.env {
            Some(env) => Some(start(env, "env", image)?.map_err(|error| {
                anyhow::anyhow!("the reference env {env} did not start: {error}")
            })?),
            None => None,
        };
    Ok((start(&image.image, "model", image)?, reference))
}

/// Start `image` as participant `name` and wait for it to serve; the inner
/// error is the container's failure to come up, with its logs.
fn start(image: &str, name: &str, args: &ProbeImageArgs) -> Result<Result<Container, String>> {
    let container = Container::start(image, name, args.gpus.as_deref())?;
    let address = container.address.clone();
    let ready = |attempt: Duration| match name {
        "env" => doctor::env_answers(&address),
        _ => doctor::model_answers(&address, attempt),
    };
    Ok(
        match container.wait_ready(Duration::from_secs(args.timeout), ready) {
            Ok(()) => Ok(container),
            Err(error) => Err(format!("{error:#}")),
        },
    )
}

/// Run `work` on a thread of its own: the probe blocks, and a blocking
/// rlmesh call refuses to run on the CLI's async runtime.
fn off_runtime<T: Send>(work: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    // Every session this command joins is hosted in a container, behind its
    // published ports: join as a client the host routes for, as the
    // container's own participants do. As concurrent peers of one container
    // host, this process's participants trip zenoh's peer routing.
    rlmesh_session::address::join_as_client();
    std::thread::scope(|scope| {
        scope
            .spawn(work)
            .join()
            .map_err(|_| anyhow::anyhow!("the probe thread panicked"))?
    })
}

/// The image's describe envelope, when it carries a parseable label.
fn describe_label(config: &ImageConfig) -> Option<serde_json::Value> {
    serde_json::from_str(config.labels.get(DESCRIBE_LABEL)?).ok()
}

/// What the image serves: its describe label's kind, else its serve command's.
fn image_kind(config: &ImageConfig) -> ProbeKind {
    let labelled = config
        .labels
        .get(DESCRIBE_LABEL)
        .and_then(|raw| image_check::parse_describe_label(raw).ok())
        .and_then(|label| label.kind);
    match labelled {
        Some(Kind::Env) => ProbeKind::Env,
        Some(Kind::Model) => ProbeKind::Model,
        None if image_check::parse_serve_command(config).env => ProbeKind::Env,
        None => ProbeKind::Model,
    }
}

/// `check-image`'s static checks of the image config and labels, as the
/// report's first check: an invalid describe label fails here.
fn image_config_check(config: &ImageConfig) -> ProbeCheck {
    let report = image_check::check_image(config, None);
    let (status, lines) = if !report.failed.is_empty() {
        (CheckStatus::Fail, &report.failed)
    } else if !report.warnings.is_empty() {
        (CheckStatus::Warn, &report.warnings)
    } else {
        (CheckStatus::Pass, &report.passed)
    };
    ProbeCheck::new("image_config", status, lines.join("; "))
        .metric("failed", report.failed.len() as f64)
        .metric("warnings", report.warnings.len() as f64)
}

fn failed(report: &ProbeReport) -> bool {
    report
        .checks
        .iter()
        .any(|check| check.status == CheckStatus::Fail)
}

fn render_checks(
    report: &ProbeReport,
    image: &str,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let subject = match report.subject {
        ProbeSubject::Model => "model",
        ProbeSubject::Environment => "env",
    };
    let width = report
        .checks
        .iter()
        .map(|check| check.name.len())
        .max()
        .unwrap_or(0);
    let mut counts = [0usize; 4];
    for check in &report.checks {
        let (label, index) = match check.status {
            CheckStatus::Pass => (style.green("ok  "), 0),
            CheckStatus::Warn => (style.yellow("warn"), 1),
            CheckStatus::Fail => (style.red_bold("FAIL"), 2),
            CheckStatus::Skip => (style.muted("skip"), 3),
        };
        counts[index] += 1;
        writeln!(stdout, "{label}  {:<width$}  {}", check.name, check.detail)?;
    }
    let summary = format!(
        "{image} ({subject}): {} passed, {} warnings, {} failed, {} skipped",
        counts[0], counts[1], counts[2], counts[3]
    );
    match report.blocker() {
        Some(blocker) => writeln!(
            stdout,
            "{} {summary}; not runnable: {blocker}",
            style.red_bold("✗")
        )?,
        None if counts[2] > 0 => writeln!(stdout, "{} {summary}", style.red_bold("✗"))?,
        None => writeln!(stdout, "{}", style.success(&summary))?,
    }
    Ok(())
}
