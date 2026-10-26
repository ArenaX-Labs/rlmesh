//! `rlmesh doctor`: the managed platform's probe, run on
//! one machine against a served env or model.
//!
//! Ported from the managed runner's probe (`probe.rs` and `probe/` in
//! rlmesh-middleman). The checks keep their names and the report keeps the
//! platform's `probe-report.json` shape ([`ProbeReport`]), so the platform
//! probe can call this code and read the same verdicts. What changed is the
//! wire: a model is reached through the 0.2 binder ([`crate::bind`]) against a
//! scripted stand-in env, so resolution is the binder's own, and an env is
//! driven through a binding with the probe's actions.
//!
//! Hidden binding plumbing (R-25): the CLI and the platform drive it.

// The report types mirror the platform's JSON contract field for field; their
// docs live on the platform side.
#![allow(missing_docs)]

mod compare;
mod env;
mod model;
mod preview;
pub mod report;
mod synth;
#[cfg(test)]
mod tests;

use std::time::Duration;

use chrono::Utc;

pub use env::{EnvProbeOptions, probe_env};
pub use model::{ModelProbeOptions, probe_model};
pub use report::{
    BaselinePoint, CheckStatus, Observed, ProbeCheck, ProbeEnv, ProbePeer, ProbePreview,
    ProbeReport, ProbeSubject, Recommendation, RenderRequest, SCHEMA_VERSION, knee_batch_size,
};
pub use synth::SYNTHETIC_ENV_ID;

/// The file a report is written to beside its preview, as the platform names it.
pub const PROBE_REPORT: &str = "probe-report.json";

/// How long a served env or model gets to answer its first call.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(120);

/// Checks whose Fail says the image is not runnable at all; the platform's
/// `probe/verdict.go` carries the same set.
pub const BLOCKING_CHECK_NAMES: [&str; 7] = [
    "handshake",
    "resolve",
    "shape_single",
    "native_chunk_contract",
    "env_handshake",
    "env_contract",
    "env_step_conforms",
];

impl ProbeReport {
    /// The first blocking check that failed, as `name: detail`.
    pub fn blocker(&self) -> Option<String> {
        BLOCKING_CHECK_NAMES.iter().find_map(|name| {
            self.checks
                .iter()
                .find(|check| check.name == *name && check.status == CheckStatus::Fail)
                .map(|check| format!("{}: {}", check.name, check.detail))
        })
    }
}

/// The report of a subject that never came up to answer its handshake:
/// `detail` says why, and every other check is skipped.
pub fn failed_handshake(subject: ProbeSubject, detail: &str) -> ProbeReport {
    let (name, names) = match subject {
        ProbeSubject::Model => ("handshake", model::MODEL_CHECK_NAMES),
        ProbeSubject::Environment => ("env_handshake", env::ENV_CHECK_NAMES),
    };
    let mut probe = Probe::new(subject, names);
    probe.record(ProbeCheck::new(name, CheckStatus::Fail, detail).tag("connect_failed"));
    probe.skip_remaining(&format!("{name} failed"));
    probe.report(ProbeEnv::default(), ProbePeer::default())
}

/// The checks a probe records, in order, and what it has observed so far.
struct Probe {
    subject: ProbeSubject,
    names: &'static [&'static str],
    started_at: chrono::DateTime<Utc>,
    checks: Vec<ProbeCheck>,
    observed: Observed,
    baseline: Vec<BaselinePoint>,
    preview: Option<ProbePreview>,
}

impl Probe {
    fn new(subject: ProbeSubject, names: &'static [&'static str]) -> Self {
        Self {
            subject,
            names,
            started_at: Utc::now(),
            checks: Vec::new(),
            observed: Observed::default(),
            baseline: Vec::new(),
            preview: None,
        }
    }

    fn record(&mut self, check: ProbeCheck) {
        tracing::debug!(
            check = %check.name,
            status = ?check.status,
            detail = %check.detail,
            "doctor check"
        );
        self.checks.push(check);
    }

    /// Skips every check not yet recorded, naming why.
    fn skip_remaining(&mut self, reason: &str) {
        for name in self.names {
            if !self.checks.iter().any(|check| check.name == *name) {
                self.checks.push(ProbeCheck::skipped(name, reason));
            }
        }
    }

    fn report(self, env: ProbeEnv, peer: ProbePeer) -> ProbeReport {
        ProbeReport {
            schema_version: SCHEMA_VERSION,
            subject: self.subject,
            started_at: self.started_at,
            ended_at: Utc::now(),
            env,
            peer,
            recommendation: Recommendation::derive(&self.observed, &self.baseline),
            checks: self.checks,
            observed: self.observed,
            baseline: self.baseline,
            preview: self.preview,
        }
    }
}

/// The peer block of a report, read off a describe envelope's `runtime`.
fn peer_from_describe(describe: Option<&serde_json::Value>) -> ProbePeer {
    let runtime = describe.and_then(|envelope| envelope.get("runtime"));
    let text = |key: &str| {
        runtime
            .and_then(|runtime| runtime.get(key))
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    ProbePeer {
        package_version: text("package_version"),
        language: text("language"),
        language_version: text("language_version"),
        capabilities: Vec::new(),
        describe: describe.cloned(),
    }
}

/// Whether the env at `address` answers: one bind, closed at once. How a
/// caller that started the env waits for it to come up.
pub fn env_answers(address: &str) -> Result<(), String> {
    let mut binding = crate::bind(crate::RANDOM_SAMPLE, address, crate::BindOptions::default())
        .map_err(|error| error.to_string())?;
    let _ = binding.close();
    Ok(())
}

/// Whether the model at `address` answers a connect within `timeout`, as the
/// Python sandbox waits for a model container.
pub fn model_answers(address: &str, timeout: Duration) -> Result<(), String> {
    let id = format!("doctor-{}", uuid::Uuid::new_v4().simple());
    crate::run::drive::block_on_library(async {
        let connection = crate::lockstep::ModelConnection::connect(address, &id, timeout)
            .await
            .map_err(|error| error.to_string())?;
        connection.close().await;
        Ok(())
    })
}

/// Bind the random baseline to the env at `address`, retrying a connect that
/// fails while the env is still coming up (its session host can answer before
/// its `make()` returns) until `timeout` has passed.
fn bind_env(address: &str, timeout: Duration) -> crate::Result<crate::Binding<'static>> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match crate::bind(crate::RANDOM_SAMPLE, address, crate::BindOptions::default()) {
            Err(error)
                if error.class().is_subclass_of(crate::ErrorClass::OSError)
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(500));
            }
            bound => return bound,
        }
    }
}

/// `sorted[p]`, nearest rank; 0 for an empty slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return 0.0;
    };
    sorted[((last as f64 * p).round() as usize).min(last)]
}
