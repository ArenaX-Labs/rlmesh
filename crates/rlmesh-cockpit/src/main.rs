//! `rlmesh-cockpit ADDRESS`: the cockpit window on one live session;
//! `rlmesh-cockpit --replay FILE` on a recorded one.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use rlmesh_cockpit::{CockpitApp, Feed, Model, ModelConfig, Replay};
use rlmesh_session::SessionAddress;
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role};

/// The RLMesh cockpit: live channels, declared vs measured rates, the
/// lease and mode timeline, and arm and e-stop, for one session.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The session address, `[host/][session]` (what
    /// `rlmesh.v2.Host.address` prints).
    #[arg(required_unless_present = "replay")]
    address: Option<SessionAddress>,
    /// Replay a recording (MCAP) instead of joining a session.
    #[arg(long, value_name = "FILE", conflicts_with = "address")]
    replay: Option<PathBuf>,
    /// This observer's participant id.
    #[arg(long, default_value = "cockpit")]
    id: String,
    /// How long to wait for the session host, in seconds.
    #[arg(long, default_value_t = 10.0)]
    timeout: f64,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,rlmesh_cockpit=info".into()),
        )
        .init();
    let args = Args::parse();
    let result = match (&args.address, &args.replay) {
        (_, Some(path)) => replay(&args, path),
        (Some(address), None) => run(&args, address),
        (None, None) => unreachable!("clap requires one"),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn options(title: String) -> eframe::NativeOptions {
    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size([1500.0, 900.0]),
        ..eframe::NativeOptions::default()
    }
}

fn replay(args: &Args, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let me = ParticipantId::new(&args.id)?;
    let replay = Replay::open(path, me, ModelConfig::default())?;
    eframe::run_native(
        "rlmesh-cockpit",
        options(format!("RLMesh cockpit · {}", path.display())),
        Box::new(|_| Ok(Box::new(CockpitApp::replay(replay)))),
    )?;
    Ok(())
}

fn run(args: &Args, address: &SessionAddress) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let _entered = runtime.enter();
    let timeout = Duration::try_from_secs_f64(args.timeout)?;
    let participant = Arc::new(runtime.block_on(address.join(
        &args.id,
        vec![Role::Observer],
        PriorityClass::Operator,
        timeout,
    ))?);
    tracing::info!(session = %address, "attached as an observer");
    let me = ParticipantId::new(&args.id)?;
    eframe::run_native(
        "rlmesh-cockpit",
        options(format!("RLMesh cockpit · {address}")),
        Box::new(|cc| {
            let ctx = cc.egui_ctx.clone();
            let wake = Arc::new(move || ctx.request_repaint());
            let feed = runtime.block_on(Feed::start(participant, wake))?;
            Ok(Box::new(CockpitApp::new(
                feed,
                Model::new(me, ModelConfig::default()),
            )))
        }),
    )?;
    Ok(())
}
