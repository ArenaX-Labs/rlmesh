//! No secret reaches a tracing subscriber. Hosting a session, a keyed join,
//! serving an env and serving a model (the Rust facade's serve and this
//! extension's) are captured at TRACE, Zenoh's targets included, through the
//! filter every subscriber this repo installs adds; neither the address key,
//! the launch identity nor the link password may appear in any event or span
//! field.
//!
//! It runs in a child process: the capture must be the process's first
//! global subscriber, and the identity rides the child's environment as a
//! launcher passes it.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

const CHILD: &str = "RLMESH_TEST_SECRET_TRACES_CHILD";
/// What the child prints once it checked every line.
const RAN: &str = "secret traces: checked";

/// Every event and span field, as text.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<String>>>);

impl Capture {
    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }

    fn saw(&self, needle: &str) -> bool {
        self.0
            .lock()
            .unwrap()
            .iter()
            .any(|line| line.contains(needle))
    }
}

struct Text<'a>(&'a mut String);

impl Visit for Text<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, " {}={value:?}", field.name());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        let _ = write!(self.0, " {}={value}", field.name());
    }
}

impl<S: tracing::Subscriber> Layer<S> for Capture {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _: &tracing::span::Id,
        _: Context<'_, S>,
    ) {
        let meta = attrs.metadata();
        let mut line = format!("span {} {}", meta.target(), meta.name());
        attrs.record(&mut Text(&mut line));
        self.0.lock().unwrap().push(line);
    }

    fn on_record(
        &self,
        _: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _: Context<'_, S>,
    ) {
        let mut line = "record".to_string();
        values.record(&mut Text(&mut line));
        self.0.lock().unwrap().push(line);
    }

    fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
        let meta = event.metadata();
        let mut line = format!("event {} {}", meta.level(), meta.target());
        event.record(&mut Text(&mut line));
        self.0.lock().unwrap().push(line);
    }
}

/// The secrets a host launched under `identity` holds, by what each is.
fn secrets(identity: &str) -> Vec<(&'static str, String)> {
    let key = rlmesh_session::address::identity_key(identity).expect("the identity");
    let text = key.to_string();
    let (secret, _host) = text.split_once('.').expect("{secret}.{host}");
    let mut secrets = vec![
        ("the identity", identity.to_string()),
        ("the address key", text.clone()),
        ("the key's secret", secret.to_string()),
        ("the link password", key.link_password().to_string()),
    ];
    for part in identity.split('.') {
        secrets.push(("a part of the identity", part.to_string()));
    }
    secrets
}

/// Waits up to 30 s for `capture` to see `needle`.
fn wait_for(capture: &Capture, needle: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !capture.saw(needle) {
        assert!(Instant::now() < deadline, "never saw {needle:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn no_secret_reaches_a_subscriber() {
    let (identity, _) = rlmesh_session::address::launch_identity().expect("keys");
    let scratch = std::env::temp_dir().join(format!("rlmesh-secret-traces-{}", std::process::id()));
    let mut dir = std::fs::DirBuilder::new();
    dir.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut dir, 0o700);
    dir.create(scratch.join("state")).expect("state dir");
    dir.create(scratch.join("runtime")).expect("runtime dir");
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "secret_traces::child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .env(
            rlmesh_session::address::HOST_IDENTITY_VAR,
            identity.as_str(),
        )
        .env("RLMESH_PROFILE_PHASES", "1")
        .env("RLMESH_STATE_DIR", scratch.join("state"))
        .env("RLMESH_RUNTIME_DIR", scratch.join("runtime"))
        .env_remove("RUST_LOG")
        .output()
        .expect("the child");
    let _ = std::fs::remove_dir_all(&scratch);
    // Its own messages never print a secret (see `child`).
    let shown = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "the child failed:\n{shown}");
    // libtest prints the child's line after its own `test … ... `.
    let ran = shown
        .lines()
        .find_map(|line| line.find(RAN).map(|at| &line[at..]));
    assert!(ran.is_some(), "the child ran nothing:\n{shown}");
    println!("{}", ran.unwrap_or_default());
}

#[test]
fn child() {
    let Ok(identity) = std::env::var(rlmesh_session::address::HOST_IDENTITY_VAR) else {
        return;
    };
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let capture = Capture::default();
    tracing_subscriber::registry()
        .with(tracing_subscriber::filter::filter_fn(
            rlmesh_session::logging::may_record,
        ))
        .with(capture.clone())
        .try_init()
        .expect("the first subscriber");

    // Serve an env: it hosts the session (under the launched identity).
    let (server, address) = Python::attach(|py| -> PyResult<_> {
        let env = py
            .import("gymnasium")?
            .call_method1("make", ("CartPole-v1",))?;
        let server = py
            .get_type::<crate::server::PyEnvServer>()
            .call1((env, "127.0.0.1:0/env"))?;
        server.call_method0("start")?;
        let address: String = server.call_method0("address")?.extract()?;
        Ok((server.unbind(), address))
    })
    .expect("serve the env");
    assert!(address.contains('#'), "a keyed address");
    wait_for(&capture, "EnvService serving");
    let session = address.rsplit_once('/').expect("participant").0.to_string();

    // A keyed join: a remote env client.
    Python::attach(|py| -> PyResult<()> {
        let client = py
            .get_type::<crate::client::PyEnvClient>()
            .call1((address.as_str(),))?;
        client.call_method0("close")?;
        Ok(())
    })
    .expect("join the env");

    // Serve a model through the Rust facade, until it is up.
    let model = rlmesh::Model::new(
        |_: rlmesh::Value| Ok(rlmesh::Value::Int(0)),
        rlmesh::ModelOptions::default(),
    )
    .expect("model");
    let cancel = rlmesh::ext::CancelToken::new();
    let serving = {
        let (session, cancel) = (session.clone(), cancel.clone());
        std::thread::spawn(move || {
            model.serve(
                &session,
                rlmesh::ModelServeOptions {
                    name: Some("policy".into()),
                    cancel: Some(cancel),
                    ..Default::default()
                },
            )
        })
    };
    wait_for(&capture, "model serving");
    cancel.cancel();
    let _ = serving.join().expect("the facade's serve");

    // Serve a model through this extension: stopped first, it returns once
    // bound.
    Python::attach(|py| -> PyResult<()> {
        let predict = py.eval(c"lambda observation: 0", None, None)?;
        let model = py.get_type::<crate::model::PyModel>().call1((predict,))?;
        model.call_method0("stop_serving")?;
        let _ = model.call_method1("serve", (session.as_str(), "policy2"));
        Ok(())
    })
    .expect("serve the extension's model");

    Python::attach(|py| server.bind(py).call_method0("shutdown").map(drop)).expect("shutdown");

    let lines = capture.lines();
    // What the run must have covered.
    for (start, needle) in [
        ("span ", " rlmesh.client.connect"),
        ("span ", " rlmesh.model.serve"),
        ("event DEBUG zenoh", ""),
    ] {
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with(start) && line.contains(needle)),
            "captured nothing like {start:?}…{needle:?}"
        );
    }
    let secrets = secrets(&identity);
    let mut leaks = Vec::new();
    for line in &lines {
        let found: Vec<&str> = secrets
            .iter()
            .filter(|(_, secret)| line.contains(secret.as_str()))
            .map(|(what, _)| *what)
            .collect();
        if !found.is_empty() {
            let mut shown = line.clone();
            for (_, secret) in &secrets {
                shown = shown.replace(secret.as_str(), "<SECRET>");
            }
            shown.truncate(shown.floor_char_boundary(240));
            leaks.push(format!("{}: {shown}", found.join(", ")));
        }
    }
    assert!(
        leaks.is_empty(),
        "{} captured lines hold a secret:\n{}",
        leaks.len(),
        leaks.join("\n")
    );
    println!("{RAN} {} lines", lines.len());
}
