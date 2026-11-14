//! One prebuilt rlmesh image run for `doctor` and `bench`, the way the Python
//! sandbox runs it (`rlmesh._sandbox.session.start_prebuilt_container`): the
//! container hosts its session on the host's own port number, published on
//! loopback, under keys made here, and serves in it under a participant
//! name. Removed on drop.

use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rlmesh_session::address::{
    AddressKey, DEFAULT_SESSION, HOST_IDENTITY_VAR, SessionAddress, launch_identity, redact_text,
};

/// The longest one readiness attempt waits, so an image that died is told
/// apart between attempts.
const READY_ATTEMPT: Duration = Duration::from_secs(5);

/// What `docker run` adds for the image under probe, as a sandbox's run
/// options do: its `volumes`, and its params as `RLMESH_MAKE_KWARGS`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunExtras<'a> {
    pub volumes: &'a [String],
    pub kwargs_json: Option<&'a str>,
}

pub struct Container {
    id: String,
    image: String,
    /// The participant it serves as (`env`, `model`).
    role: String,
    /// The participant address the container serves at, with its host's
    /// key: a credential, so print it only through `redact`.
    pub address: String,
}

impl Container {
    /// `docker run -d` `image`, serving as participant `name`, with the
    /// sandbox's hardening flags and owner labels.
    pub fn start(
        image: &str,
        name: &str,
        gpus: Option<&str>,
        extras: RunExtras<'_>,
    ) -> Result<Self> {
        let port = free_session_port()?;
        let (identity, key) = launch_identity()?;
        let output = run_command(image, name, gpus, port, &identity, extras)
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
            role: name.to_string(),
            address: participant_address(port, key, name)?,
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
            if let Some(code) = self.exited() {
                bail!("{}", self.exit_report(code));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            let error = match ready(left.min(READY_ATTEMPT)) {
                Ok(()) => return Ok(()),
                Err(error) => error,
            };
            // A container that died during the attempt is its cause.
            if let Some(code) = self.exited() {
                bail!("{}", self.exit_report(code));
            }
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

    /// The container's exit code once it stopped (`Some(None)`: stopped,
    /// code unknown), `None` while it runs or when docker cannot say.
    fn exited(&self) -> Option<Option<i32>> {
        let output = Command::new("docker")
            .args([
                "inspect",
                "-f",
                "{{.State.Running}} {{.State.ExitCode}}",
                &self.id,
            ])
            .output()
            .ok()
            .filter(|output| output.status.success())?;
        let text = String::from_utf8_lossy(&output.stdout);
        let (running, code) = text.trim().split_once(' ').unwrap_or((text.trim(), ""));
        (running != "true").then(|| code.parse().ok())
    }

    /// The error of a container that exited before it served: its exit
    /// leads, then its last log lines.
    fn exit_report(&self, code: Option<i32>) -> String {
        let short = &self.id[..self.id.len().min(12)];
        exit_report(&self.role, &self.image, short, code, &self.logs())
    }

    /// The last lines the container wrote, any secret in them (its address
    /// key, its host identity) left out.
    pub fn logs(&self) -> String {
        Command::new("docker")
            .args(["logs", "--tail", "40", &self.id])
            .output()
            .map(|output| {
                redact_text(&format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ))
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

/// The error of `role`'s container (`short` id, running `image`) that
/// exited with `code` before it served, quoting its last log lines.
fn exit_report(role: &str, image: &str, short: &str, code: Option<i32>, logs: &str) -> String {
    let exited = match code {
        Some(code) => format!("the {role} container exited with code {code} during startup"),
        None => format!("the {role} container stopped during startup"),
    };
    let logs = logs.trim_end();
    if logs.is_empty() {
        format!("{exited} ({image}, container {short}); it logged nothing")
    } else {
        format!("{exited} ({image}, container {short}); its last log lines:\n{logs}")
    }
}

/// The `docker run -d` of `image` hosting its session on `port` under
/// `identity` ([`launch_identity`]'s text). Beyond loopback, as a
/// container's published port is, a session hosts only under keys its
/// launcher passed; the secret reaches docker through its environment,
/// never argv.
fn run_command(
    image: &str,
    name: &str,
    gpus: Option<&str>,
    port: u16,
    identity: &str,
    extras: RunExtras<'_>,
) -> Command {
    let mut command = Command::new("docker");
    command.env(HOST_IDENTITY_VAR, identity).args([
        "run",
        "-d",
        "-p",
        &format!("127.0.0.1:{port}:{port}"),
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
    for volume in extras.volumes {
        command.args(["-v", volume]);
    }
    if let Some(kwargs) = extras.kwargs_json {
        command.args(["-e", &format!("RLMESH_MAKE_KWARGS={kwargs}")]);
    }
    for (key, value) in [
        ("RLMESH_ADDRESS", format!("0.0.0.0:{port}")),
        ("RLMESH_NAME", name.to_string()),
    ] {
        command.args(["-e", &format!("{key}={value}")]);
    }
    command.args(["-e", HOST_IDENTITY_VAR]).arg(image);
    command
}

/// Participant `name`'s address in the session a container hosts on
/// loopback `port` under `key`.
fn participant_address(port: u16, key: AddressKey, name: &str) -> Result<String> {
    let session = SessionAddress {
        key: Some(key),
        ..SessionAddress::new(
            SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
            DEFAULT_SESSION,
        )?
    };
    Ok(session.participant_invite(name))
}

/// A free port on 127.0.0.1 to publish a container's session on. Free when
/// checked: `docker run` reports a port taken since.
fn free_session_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("no free local port")?;
    Ok(listener.local_addr()?.port())
}

#[cfg(test)]
mod tests {
    use rlmesh_session::address::Address;

    use super::*;

    #[test]
    fn an_exit_during_startup_leads_its_report() {
        assert_eq!(
            exit_report(
                "env",
                "sapien:gate",
                "abc123",
                Some(1),
                "libvulkan missing\n"
            ),
            "the env container exited with code 1 during startup (sapien:gate, container \
             abc123); its last log lines:\nlibvulkan missing"
        );
        assert_eq!(
            exit_report("model", "m:1", "abc123", None, ""),
            "the model container stopped during startup (m:1, container abc123); it logged nothing"
        );
    }

    #[test]
    fn run_passes_the_identity_by_name_and_dials_its_key() {
        let (identity, key) = launch_identity().unwrap();
        let command = run_command(
            "img:tag",
            "env",
            None,
            40000,
            &identity,
            RunExtras::default(),
        );
        let args: Vec<String> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(args.iter().all(|arg| !arg.contains(identity.as_str())));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-e", HOST_IDENTITY_VAR])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-e", "RLMESH_ADDRESS=0.0.0.0:40000"])
        );
        // One port: every traffic class dials it.
        let published: Vec<_> = args
            .windows(2)
            .filter(|pair| pair[0] == "-p")
            .map(|pair| pair[1].as_str())
            .collect();
        assert_eq!(published, ["127.0.0.1:40000:40000"]);
        assert_eq!(args.last().map(String::as_str), Some("img:tag"));
        let passed = command
            .get_envs()
            .find(|(name, _)| *name == HOST_IDENTITY_VAR)
            .and_then(|(_, value)| value);
        assert_eq!(passed.and_then(|v| v.to_str()), Some(identity.as_str()));

        let address = participant_address(40000, key.clone(), "env").unwrap();
        let parsed = Address::participant(&address).unwrap();
        assert_eq!(parsed.key, Some(key));
        assert_eq!(parsed.participant.as_deref(), Some("env"));
        assert_eq!(
            parsed.joining().unwrap().base,
            SocketAddr::from((Ipv4Addr::LOCALHOST, 40000))
        );
    }

    #[test]
    fn run_mounts_volumes_and_binds_params_as_a_sandbox_does() {
        let volumes = ["/host/weights:/weights:ro".to_owned()];
        let command = run_command(
            "img:tag",
            "model",
            None,
            40000,
            "identity",
            RunExtras {
                volumes: &volumes,
                kwargs_json: Some(r#"{"checkpoint":"/weights/a"}"#),
            },
        );
        let args: Vec<String> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-v", "/host/weights:/weights:ro"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-e", r#"RLMESH_MAKE_KWARGS={"checkpoint":"/weights/a"}"#])
        );
        assert_eq!(args.last().map(String::as_str), Some("img:tag"));
    }
}
