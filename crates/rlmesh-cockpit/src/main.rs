//! `rlmesh-cockpit ADDRESS`: the cockpit window on one live session;
//! `rlmesh-cockpit --replay FILE` on a recorded one. The same as
//! `rlmesh cockpit`.

use std::process::ExitCode;

use clap::Parser;
use rlmesh_cockpit::cli::{CockpitArgs, run};

/// The RLMesh cockpit: live channels, declared vs measured rates, why each
/// robot is or isn't moving, the lease and mode timeline, and arm and
/// e-stop, for one session.
#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    #[command(flatten)]
    args: CockpitArgs,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!("{},rlmesh_cockpit=info", rlmesh_session::DEFAULT_LOG_FILTER).into()
            }),
        )
        .init();
    let args = Cli::parse().args;
    let result = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(Into::into)
        .and_then(|runtime| runtime.block_on(run(args)));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!("{err}");
            ExitCode::FAILURE
        }
    }
}
