//! `rlmesh-cockpit ADDRESS`: the cockpit window on one live session.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use rlmesh_cockpit::{CockpitApp, Feed, Model, ModelConfig};
use rlmesh_session::SessionAddress;
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role};

/// The RLMesh cockpit: live channels, declared vs measured rates, the
/// lease and mode timeline, and arm and e-stop, for one session.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The session address, `[host/][session]` (what
    /// `rlmesh.v2.Host.address` prints).
    address: SessionAddress,
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
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let _entered = runtime.enter();
    let timeout = Duration::try_from_secs_f64(args.timeout)?;
    let participant = Arc::new(runtime.block_on(args.address.join(
        &args.id,
        vec![Role::Observer],
        PriorityClass::Operator,
        timeout,
    ))?);
    tracing::info!(session = %args.address, "attached as an observer");
    let me = ParticipantId::new(&args.id)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(format!("RLMesh cockpit · {}", args.address))
            .with_inner_size([1500.0, 900.0]),
        ..eframe::NativeOptions::default()
    };
    eframe::run_native(
        "rlmesh-cockpit",
        options,
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
