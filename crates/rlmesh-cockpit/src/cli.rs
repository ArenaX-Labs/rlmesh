//! `rlmesh cockpit ADDRESS`: the cockpit window on one live session;
//! `rlmesh cockpit --replay FILE` on a recorded one. The same command is the
//! standalone `rlmesh-cockpit` binary.
//!
//! [`run`] is async so the `rlmesh` CLI can call it from its runtime: it
//! joins the session there, then runs the window on the calling thread
//! (`block_in_place`), which must be the process's main thread on macOS.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use rlmesh_session::SessionAddress;
use rlmesh_session::record::Rolling;
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role};

use crate::{CockpitApp, Feed, Model, ModelConfig, Replay};

/// The RLMesh cockpit: live channels, declared vs measured rates, why each
/// robot is or isn't moving, the lease and mode timeline, and arm and
/// e-stop, for one session.
#[derive(clap::Args, Clone)]
pub struct CockpitArgs {
    /// The session address, `[host/][session]` (what `rlmesh host` and
    /// `rlmesh.realtime.Host.address` print). Its key (`#…`) is a
    /// credential and the command line is world-readable: prefer
    /// RLMESH_COCKPIT_ADDRESS or --address-file.
    pub address: Option<String>,
    /// Read the session address from this file.
    #[arg(long, value_name = "FILE", conflicts_with = "address")]
    pub address_file: Option<PathBuf>,
    /// Replay a recording instead of joining a session: an MCAP file, or a
    /// directory of a session's rolling segments.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["address", "address_file"])]
    pub replay: Option<PathBuf>,
    /// This observer's participant id.
    #[arg(long, default_value = "cockpit")]
    pub id: String,
    /// How long to wait for the session host, in seconds.
    #[arg(long, default_value_t = 10.0)]
    pub timeout: f64,
    /// The recordings directory of a host on this machine (`rlmesh host
    /// --record-dir`), for episodes; the host's default when absent.
    #[arg(long, value_name = "DIR")]
    pub record_dir: Option<PathBuf>,
    /// The dataset episodes are collected into (`collected/{dataset}/`
    /// under the recordings directory).
    #[cfg(feature = "collect")]
    #[arg(long, default_value = "default")]
    pub dataset: String,
}

// The address may carry its key: `Debug` leaves it out.
impl std::fmt::Debug for CockpitArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CockpitArgs");
        debug
            .field(
                "address",
                &self.address.as_deref().map(rlmesh_session::address::redact),
            )
            .field("address_file", &self.address_file)
            .field("replay", &self.replay)
            .field("id", &self.id)
            .field("timeout", &self.timeout)
            .field("record_dir", &self.record_dir);
        #[cfg(feature = "collect")]
        debug.field("dataset", &self.dataset);
        debug.finish()
    }
}

/// The variable the session address comes from when neither the argument
/// nor `--address-file` gives it.
pub const ADDRESS_VAR: &str = "RLMESH_COCKPIT_ADDRESS";

/// The cockpit's error.
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Opens the cockpit window for `args` and returns when it closes. Call it
/// on a multi-threaded Tokio runtime, from the process's main thread.
pub async fn run(args: CockpitArgs) -> Result<(), Error> {
    if let Some(path) = &args.replay {
        return replay(path);
    }
    let address = session_address(&args, std::env::var(ADDRESS_VAR).ok())?;
    live(&args, &address).await
}

/// The session address from the argument, `--address-file` or
/// [`ADDRESS_VAR`] (`env`), in that order. Parsed here rather than by clap,
/// whose error would repeat the text, key and all.
pub fn session_address(args: &CockpitArgs, env: Option<String>) -> Result<SessionAddress, Error> {
    let (from, text) = match (&args.address, &args.address_file) {
        (Some(text), _) => ("ADDRESS".to_string(), text.clone()),
        (None, Some(file)) => (
            format!("--address-file {}", file.display()),
            std::fs::read_to_string(file)
                .map_err(|e| format!("--address-file {}: {e}", file.display()))?
                .trim()
                .to_string(),
        ),
        (None, None) => match env.filter(|text| !text.trim().is_empty()) {
            Some(text) => (ADDRESS_VAR.to_string(), text.trim().to_string()),
            None => {
                return Err(format!(
                    "a session address (ADDRESS, --address-file or {ADDRESS_VAR}) or --replay FILE"
                )
                .into());
            }
        },
    };
    // The session's own parse error already leaves the key out.
    text.parse::<SessionAddress>()
        .map_err(|e| format!("{from}: {e}").into())
}

fn options(title: String) -> eframe::NativeOptions {
    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size([1500.0, 900.0]),
        ..eframe::NativeOptions::default()
    }
}

/// Runs the window on this thread, out of the runtime's way.
fn window(
    title: String,
    app: impl FnOnce(&eframe::CreationContext<'_>) -> Result<CockpitApp, Error> + 'static,
) -> Result<(), Error> {
    let shown = || {
        eframe::run_native(
            "rlmesh-cockpit",
            options(title),
            Box::new(move |cc| Ok(Box::new(app(cc)?))),
        )
    };
    let shown = match tokio::runtime::Handle::try_current() {
        Ok(_) => tokio::task::block_in_place(shown),
        Err(_) => shown(),
    };
    shown.map_err(|err| err.to_string().into())
}

fn replay(path: &Path) -> Result<(), Error> {
    // Not a participant of the recording (its cockpit is likely `cockpit`):
    // replayed inputs are already session time, so no clock mapping of
    // `me` may apply to them.
    let me = ParticipantId::new("_replay")?;
    let replay = Replay::open(path, me, ModelConfig::default())?;
    let path = path.to_path_buf();
    window(format!("RLMesh cockpit · {}", path.display()), move |_| {
        Ok(CockpitApp::replay(replay, path))
    })
}

async fn live(args: &CockpitArgs, address: &SessionAddress) -> Result<(), Error> {
    let timeout = Duration::try_from_secs_f64(args.timeout)?;
    let participant = Arc::new(
        address
            .join(
                &args.id,
                vec![Role::Observer],
                PriorityClass::Operator,
                timeout,
            )
            .await?,
    );
    tracing::info!(session = %address, "attached as an observer");
    let me = ParticipantId::new(&args.id)?;
    // The window's context exists only once it opens; until then nothing
    // needs repainting.
    let ctx: Arc<OnceLock<egui::Context>> = Arc::default();
    let wake = Arc::new({
        let ctx = ctx.clone();
        move || {
            if let Some(ctx) = ctx.get() {
                ctx.request_repaint();
            }
        }
    });
    let feed = Feed::start(participant, wake).await?;
    let recordings = args.record_dir.clone().or_else(Rolling::default_dir);
    #[cfg(feature = "collect")]
    let dataset = args.dataset.clone();
    #[cfg(not(feature = "collect"))]
    let dataset = String::new();
    window(format!("RLMesh cockpit · {address}"), move |cc| {
        let _ = ctx.set(cc.egui_ctx.clone());
        Ok(CockpitApp::new(
            feed,
            Model::new(me, ModelConfig::default()),
            recordings.map(Rolling::new),
            dataset,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: CockpitArgs,
    }

    #[test]
    fn a_keyed_address_is_never_echoed() {
        let key = rlmesh_session::address::launch_identity()
            .expect("keys")
            .1
            .to_string();
        let parse = |line: &str| {
            Cli::try_parse_from(line.split_whitespace())
                .expect("args")
                .args
        };
        // Positional, malformed: our parse refuses it, not clap's echo.
        let args = parse(&format!("cockpit 127.0.0.1:x/s#{key}"));
        assert!(!format!("{args:?}").contains(&key));
        let err = session_address(&args, None)
            .expect_err("a bad port")
            .to_string();
        assert!(!err.contains(&key), "{err}");
        // From the environment, when nothing else names it.
        let keyed = format!("127.0.0.1:17600/s#{key}");
        let address =
            session_address(&parse("cockpit"), Some(keyed)).expect("from the environment");
        assert_eq!(address.key.map(|k| k.to_string()), Some(key.clone()));
        let err = session_address(&parse("cockpit"), Some(format!("127.0.0.1:x/s#{key}")))
            .expect_err("a bad port")
            .to_string();
        assert!(!err.contains(&key) && err.contains(ADDRESS_VAR), "{err}");
        assert!(session_address(&parse("cockpit"), None).is_err());
    }
}
