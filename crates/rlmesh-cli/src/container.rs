//! One prebuilt rlmesh image run for `doctor` and `bench`, the way the Python
//! sandbox runs it (`rlmesh._sandbox.session.start_prebuilt_container`): the
//! container hosts its session on the host's own port numbers, published on
//! loopback, and serves in it under a participant name. Removed on drop.

use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rlmesh_session::address::JOIN_AS_CLIENT_VAR;
use rlmesh_session::transport::TrafficClass;

/// The longest one readiness attempt waits, so an image that died is told
/// apart between attempts.
const READY_ATTEMPT: Duration = Duration::from_secs(5);

/// The ports a session host listens on, one per traffic class, consecutive.
const SESSION_PORTS: u16 = TrafficClass::ALL.len() as u16;

pub struct Container {
    id: String,
    image: String,
    /// The participant address the container serves at.
    pub address: String,
}

impl Container {
    /// `docker run -d` `image`, serving as participant `name`, with the
    /// sandbox's hardening flags and owner labels.
    pub fn start(image: &str, name: &str, gpus: Option<&str>) -> Result<Self> {
        let port = free_session_ports()?;
        let last = port + SESSION_PORTS - 1;
        let mut command = Command::new("docker");
        command.args([
            "run",
            "-d",
            "-p",
            &format!("127.0.0.1:{port}-{last}:{port}-{last}"),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--label",
            "rlmesh.sandbox=1",
            "--label",
            &format!("rlmesh.sandbox.owner-pid={}", std::process::id()),
        ]);
        if let Some(gpus) = gpus {
            command.args(["--gpus", gpus]);
        }
        for (key, value) in [
            ("RLMESH_ADDRESS", format!("0.0.0.0:{port}")),
            ("RLMESH_NAME", name.to_string()),
            (JOIN_AS_CLIENT_VAR, "1".to_string()),
        ] {
            command.args(["-e", &format!("{key}={value}")]);
        }
        let output = command
            .arg(image)
            .stdin(Stdio::null())
            .output()
            .context("running docker run (is docker installed?)")?;
        if !output.status.success() {
            bail!(
                "docker run {image} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(Self {
            id: String::from_utf8_lossy(&output.stdout).trim().to_string(),
            image: image.to_string(),
            address: format!("127.0.0.1:{port}/{name}"),
        })
    }

    /// Wait until `ready` (one attempt, given how long it may take) says the
    /// container serves, or fail with its recent logs once it exits or
    /// `timeout` passes. A published port answers TCP before the container
    /// serves, and after it died, so only a real call says it is up.
    pub fn wait_ready(
        &self,
        timeout: Duration,
        ready: impl Fn(Duration) -> std::result::Result<(), String>,
    ) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let short = &self.id[..self.id.len().min(12)];
        loop {
            if !self.running() {
                bail!(
                    "container {short} for {} exited before serving; recent logs:\n{}",
                    self.image,
                    self.logs()
                );
            }
            let left = deadline.saturating_duration_since(Instant::now());
            let error = match ready(left.min(READY_ATTEMPT)) {
                Ok(()) => return Ok(()),
                Err(error) => error,
            };
            if Instant::now() >= deadline {
                bail!(
                    "container {short} for {} did not serve within {}s ({error}); recent logs:\n{}",
                    self.image,
                    timeout.as_secs(),
                    self.logs()
                );
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    fn running(&self) -> bool {
        Command::new("docker")
            .args(["inspect", "-f", "{{.State.Running}}", &self.id])
            .output()
            .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).trim() == "true")
    }

    /// The last lines the container wrote.
    pub fn logs(&self) -> String {
        Command::new("docker")
            .args(["logs", "--tail", "40", &self.id])
            .output()
            .map(|output| {
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            })
            .unwrap_or_default()
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["rm", "-f", &self.id])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// A base port on 127.0.0.1 with the session ports after it free. Free when
/// checked: `docker run` reports a port taken since.
fn free_session_ports() -> Result<u16> {
    for _ in 0..64 {
        let first = TcpListener::bind(("127.0.0.1", 0)).context("no free local port")?;
        let base = first.local_addr()?.port();
        let Some(last) = base.checked_add(SESSION_PORTS - 1) else {
            continue;
        };
        if (base + 1..=last).all(|port| TcpListener::bind(("127.0.0.1", port)).is_ok()) {
            return Ok(base);
        }
    }
    bail!("found no {SESSION_PORTS} consecutive free ports on 127.0.0.1")
}
