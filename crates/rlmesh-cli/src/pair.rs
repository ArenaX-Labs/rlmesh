//! `rlmesh pair` (design §3 item 2): on a laptop, `rlmesh pair HOST` runs
//! the pairing exchange with the robot's rlmeshd, shows the 6-digit code,
//! and once the owner approves on the robot keeps the host's key, id and
//! addresses with this machine's new device key in
//! `~/.config/rlmesh/hosts/{host id}.toml` ([`HostFile`]); joining that
//! host attaches with them from then on. On the robot, `rlmesh pair
//! approve [CODE]` approves the machine showing the code, which rlmeshd
//! enrolls in `devices.toml` as a `machine` (default cap `operator`).
//!
//! **The local channel.** `pair approve` and `devices provision` reach the
//! running rlmeshd over a Unix socket in its state directory
//! ([`SOCKET`], 0600 in a 0700 directory, so only the daemon's user, on
//! the robot, gets in): one JSON request and one JSON reply per
//! connection.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlmesh_session::address::Address;
use rlmesh_session::pairing::{self, HostFile};
use serde::{Deserialize, Serialize};

use crate::cli::{PairArgs, PairCommand};

/// The local channel's socket in rlmeshd's state directory.
pub const SOCKET: &str = "pair.sock";
/// A machine's default cap (design §12 decision 6).
const MACHINE_CAP: &str = "operator";

/// A request on the local channel.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Ask {
    /// The machines waiting for approval.
    Pending,
    /// Approve the machine showing `code`, at `max`.
    Approve { code: u32, max: String },
    /// The host's key, id and addresses, for a board.
    Host,
}

/// Its reply.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Answer {
    Pending {
        machines: Vec<Waiting>,
    },
    Approved {
        device: String,
        max: String,
    },
    Host {
        host_key: String,
        host_id: String,
        addresses: Vec<String>,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct Waiting {
    name: String,
    kind: String,
    age_s: u64,
}

/// `rlmesh pair`.
pub async fn pair(args: &PairArgs, stdout: &mut impl Write) -> Result<()> {
    match &args.command {
        Some(PairCommand::Approve {
            code,
            max,
            state_dir,
        }) => {
            let dir = crate::daemon::state_dir(state_dir.as_deref())?;
            approve(&dir, code.as_deref(), max.as_deref(), stdout)
        }
        None => {
            let host = args.host.as_deref().context("name the robot's host")?;
            pair_machine(host, args.name.as_deref(), stdout).await
        }
    }
}

async fn pair_machine(host: &str, name: Option<&str>, stdout: &mut impl Write) -> Result<()> {
    let address = Address::session(host)?;
    let Some(named) = address.host.clone() else {
        bail!("{host:?} names no host: give the robot's, as robot.local or 10.0.0.5:7447");
    };
    let base = address.joining()?.base;
    let dialed = rlmesh_session::address::HostName {
        port: Some(base.port()),
        ..named
    }
    .to_string();
    let name = match name {
        Some(name) => name.to_string(),
        None => hostname().unwrap_or_else(|| "machine".into()),
    };
    writeln!(stdout, "pairing with rlmeshd at {dialed} as {name}")?;
    stdout.flush()?;
    let paired = pairing::pair(base, &name, |sas| {
        let _ = writeln!(stdout, "Confirm on the robot: rlmesh pair approve {sas}");
        let _ = stdout.flush();
    })
    .await?;
    let file = HostFile::new(&paired, &dialed);
    let dir = pairing::hosts_dir().context("no home directory for ~/.config/rlmesh/hosts")?;
    let path = file.write(&dir)?;
    writeln!(
        stdout,
        "paired with {} as device {}: {}",
        file.host_id,
        file.device,
        path.display()
    )?;
    Ok(())
}

#[cfg(unix)]
fn hostname() -> Option<String> {
    crate::sys::hostname()
}

#[cfg(not(unix))]
fn hostname() -> Option<String> {
    std::env::var("COMPUTERNAME").ok().filter(|n| !n.is_empty())
}

fn approve(
    dir: &Path,
    code: Option<&str>,
    max: Option<&str>,
    stdout: &mut impl Write,
) -> Result<()> {
    let code = match code {
        Some(code) => code.to_string(),
        None => {
            let Answer::Pending { machines } = call(dir, &Ask::Pending)? else {
                bail!("rlmeshd answered something else");
            };
            if machines.is_empty() {
                bail!("no machine is waiting to pair: run rlmesh pair on it first");
            }
            for m in &machines {
                writeln!(
                    stdout,
                    "waiting: {} ({}), {} s ago",
                    m.name, m.kind, m.age_s
                )?;
            }
            write!(stdout, "Code the machine shows: ")?;
            stdout.flush()?;
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            line.trim().to_string()
        }
    };
    let digits = code.trim();
    let code = match digits.parse::<u32>() {
        Ok(code) if digits.len() == 6 && code < 1_000_000 => code,
        _ => bail!("the code is the 6 digits the machine shows"),
    };
    let max = max.unwrap_or(MACHINE_CAP).to_string();
    match call(dir, &Ask::Approve { code, max })? {
        Answer::Approved { device, max } => {
            writeln!(stdout, "paired device {device} at {max}")?;
            Ok(())
        }
        other => bail!("rlmeshd answered {other:?}"),
    }
}

/// The host's details a board is provisioned with, from the running
/// rlmeshd.
pub fn host_details(dir: &Path) -> Result<crate::provision::Home> {
    let Answer::Host {
        host_key,
        host_id,
        addresses,
    } = call(dir, &Ask::Host)?
    else {
        bail!("rlmeshd answered something else");
    };
    Ok(crate::provision::Home {
        host_key: rlmesh_session::auth::address::unbase32(&host_key)
            .context("rlmeshd's host key")?,
        host_id,
        addresses,
    })
}

/// One request to the rlmeshd whose state is in `dir`.
#[cfg(unix)]
fn call(dir: &Path, ask: &Ask) -> Result<Answer> {
    use std::io::Read;
    let path = dir.join(SOCKET);
    let mut stream = std::os::unix::net::UnixStream::connect(&path).with_context(|| {
        format!(
            "rlmeshd is not running with its state in {} (no {})",
            dir.display(),
            path.display()
        )
    })?;
    let mut request = serde_json::to_vec(ask)?;
    request.push(b'\n');
    stream.write_all(&request)?;
    stream.shutdown(std::net::Shutdown::Write)?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply)?;
    match serde_json::from_str(&reply).context("rlmeshd's reply")? {
        Answer::Error { message } => bail!("{message}"),
        answer => Ok(answer),
    }
}

// TODO: the local channel is a Unix socket; give rlmeshd a named pipe when
// it runs on Windows.
#[cfg(not(unix))]
fn call(_: &Path, _: &Ask) -> Result<Answer> {
    bail!("rlmeshd's local channel (pair approve, devices provision) needs Unix")
}

/// What rlmeshd's side of the local channel works with.
#[cfg(unix)]
pub struct Local {
    /// The state directory.
    pub dir: std::path::PathBuf,
    /// The pairing endpoint.
    pub service: std::sync::Arc<pairing::PairService>,
    /// The host's static key, id and addresses.
    pub host: rlmesh_session::wire::PairInfo,
    /// Asks the daemon to apply `devices.toml` now, and answers once it
    /// has.
    pub apply: tokio::sync::mpsc::Sender<tokio::sync::oneshot::Sender<()>>,
}

/// Serves the local channel until the returned task is aborted (on drop of
/// the daemon), the socket left behind removed at the next start.
#[cfg(unix)]
pub fn serve_local(local: Local) -> Result<tokio::task::JoinHandle<()>> {
    use std::os::unix::fs::PermissionsExt;
    let path = local.dir.join(SOCKET);
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err).with_context(|| path.display().to_string()),
    }
    let listener =
        tokio::net::UnixListener::bind(&path).with_context(|| path.display().to_string())?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    let local = std::sync::Arc::new(local);
    Ok(tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(answer_local(stream, local.clone()));
        }
    }))
}

#[cfg(unix)]
async fn answer_local(stream: tokio::net::UnixStream, local: std::sync::Arc<Local>) {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    let (read, mut write) = stream.into_split();
    let mut line = String::new();
    // One short request.
    if BufReader::new(read.take(4096))
        .read_line(&mut line)
        .await
        .is_err()
    {
        return;
    }
    let answer = match serde_json::from_str::<Ask>(&line) {
        Ok(Ask::Pending) => Answer::Pending {
            machines: local
                .service
                .pending()
                .into_iter()
                .map(|p| Waiting {
                    name: p.name,
                    kind: p.kind,
                    age_s: u64::try_from(p.age_ns / 1_000_000_000).unwrap_or_default(),
                })
                .collect(),
        },
        Ok(Ask::Approve { code, max }) => approve_local(&local, code, &max).await,
        Ok(Ask::Host) => Answer::Host {
            host_key: rlmesh_session::auth::address::base32(&local.host.host_key),
            host_id: local.host.host_id.clone(),
            addresses: local.host.addresses.clone(),
        },
        Err(err) => Answer::Error {
            message: format!("not a request: {err}"),
        },
    };
    if let Ok(mut reply) = serde_json::to_vec(&answer) {
        reply.push(b'\n');
        let _ = write.write_all(&reply).await;
    }
}

/// Approves the machine showing `code` at `max`: enrolls it, applies the
/// table, and sends the machine its result.
#[cfg(unix)]
async fn approve_local(local: &Local, code: u32, max: &str) -> Answer {
    use rlmesh_session::wire::{CtlError, CtlStatus, PairInfo};

    use crate::devices::{Entry, edit};
    let error = |message: String| Answer::Error { message };
    if let Err(err) = crate::devices::class(max) {
        return error(format!("{err:#}"));
    }
    let approved = match local.service.approve(code) {
        Ok(approved) => approved,
        Err(err) => return error(err.to_string()),
    };
    let public_key = rlmesh_session::auth::address::base32(&approved.public);
    let enrolled = edit(&local.dir, |table| {
        let id = table.free_id(&approved.name, "machine");
        let entry = Entry {
            public_key,
            kind: Some(approved.kind.clone()),
            name: Some(approved.name.clone()),
            max_priority: max.to_string(),
            parts: None,
            verbs: None,
        };
        table.devices.insert(id.clone(), entry);
        Ok(id)
    });
    let device = match enrolled {
        Ok(device) => device,
        Err(err) => {
            local.service.decide(
                approved.exchange,
                Err(CtlError::new(
                    CtlStatus::Unsupported,
                    "rlmeshd could not record the device",
                )),
            );
            return error(format!("{err:#}"));
        }
    };
    // Admitted before the machine hears, so it can attach at once.
    let (done, applied) = tokio::sync::oneshot::channel();
    if local.apply.send(done).await.is_ok() {
        let _ = applied.await;
    }
    let info = PairInfo {
        device: device.clone(),
        ..local.host.clone()
    };
    if !local.service.decide(approved.exchange, Ok(info)) {
        // Admitted for a moment: its id goes like a revoked one's.
        let _ = edit(&local.dir, |table| table.revoke(&device));
        return error("the machine's pairing expired before the approval".into());
    }
    Answer::Approved {
        device,
        max: max.to_string(),
    }
}
