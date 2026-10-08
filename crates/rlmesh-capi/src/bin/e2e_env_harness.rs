//! End-to-end loopback harness for the env side: spawn a compiled C/C++ env
//! binary (`<bin> 127.0.0.1:0`), read the address it prints
//! (`listening on <address>`), drive a model against it, then close the
//! binary's stdin and expect a graceful exit. The inverse of `e2e_harness`.
//!
//! Usage: `e2e_env_harness <env-binary>`. Asserts that the env published its
//! adapter tags and `trial_index` reset option, that three seeded, trial-indexed
//! episodes of two steps run, that every reset reached C with its seed and trial
//! index, that render returns a PNG, and that the env's close hook ran.
#![allow(clippy::print_stderr)]

use std::io::{BufRead, BufReader, Read};
use std::process::{Command, ExitCode, Stdio};

use async_trait::async_trait;
use rlmesh::spaces::{DType, MetaValue, SpaceValue, Tensor};
use rlmesh::{ModelHandler, ModelObservation, ModelWorker, RemoteEnv, RunLocalOptions};

const EPISODES: u64 = 3;
const BASE_SEED: i64 = 40;

/// A zero `float32[3]` action per row.
struct ZeroModel;

#[async_trait]
impl ModelHandler for ZeroModel {
    async fn predict(&mut self, obs: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        Ok((0..obs.num_envs)
            .map(|_| {
                SpaceValue::Box(
                    Tensor::from_vec(vec![0; 12], vec![3], DType::Float32).expect("zero action"),
                )
            })
            .collect())
    }
}

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("e2e_env_harness: {message}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let Some(binary) = std::env::args().nth(1) else {
        return fail("usage: e2e_env_harness <env-binary>");
    };
    let mut child = match Command::new(&binary)
        .arg("127.0.0.1:0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => return fail(format!("spawn {binary}: {err}")),
    };
    let mut stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
    let mut first = String::new();
    if stdout.read_line(&mut first).is_err() {
        let _ = child.kill();
        return fail("env printed nothing");
    }
    eprint!("{first}");
    let Some(address) = first
        .trim()
        .strip_prefix("listening on ")
        .map(str::to_string)
    else {
        let _ = child.kill();
        return fail(format!("expected `listening on <address>`, got {first:?}"));
    };

    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let checked = runtime.block_on(drive(&address));

    // Closing stdin is the env's cue to cancel, drain, close and exit.
    drop(child.stdin.take());
    let mut rest = String::new();
    let _ = stdout.read_to_string(&mut rest);
    eprint!("{rest}");
    let status = child.wait();
    if let Err(message) = checked {
        return fail(message);
    }
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => return fail(format!("env exited with {status}")),
        Err(err) => return fail(format!("wait for env: {err}")),
    }
    if !rest.contains("env closed") {
        return fail("the env's close hook did not run");
    }
    for k in 0..EPISODES {
        // The runtime derives each episode's seed from the base; the trial
        // index walks base, base + 1, ...
        if !rest.contains(&format!("trial={k}")) {
            return fail(format!("no reset reached C with trial index {k}"));
        }
    }
    // The direct probe reset, then one per run episode.
    if rest.matches("reset seed=").count() != EPISODES as usize + 1 {
        return fail(format!("expected {} resets", EPISODES + 1));
    }
    eprintln!("e2e_env_harness: OK");
    ExitCode::SUCCESS
}

async fn drive(address: &str) -> Result<(), String> {
    let mut client = RemoteEnv::connect(address)
        .await
        .map_err(|err| format!("connect: {err}"))?;
    let metadata = client
        .env_contract()
        .metadata
        .clone()
        .ok_or("the contract carries no metadata")?;
    if !matches!(
        metadata.get("rlmesh.adapters.v1.env_tags"),
        Some(MetaValue::Map(_))
    ) {
        return Err("adapter tags were not published as a map".into());
    }
    if !matches!(
        metadata.get(rlmesh::ENV_RESET_OPTIONS_KEY),
        Some(MetaValue::List(_))
    ) {
        return Err("the trial_index reset option was not declared".into());
    }
    client
        .reset(rlmesh::ResetRequest::default())
        .await
        .map_err(|err| format!("reset: {err}"))?;
    let frame = client
        .render(rlmesh::RenderRequest::default())
        .await
        .map_err(|err| format!("render: {err}"))?
        .frame
        .ok_or("render returned no frame")?;
    if !frame.frame.starts_with(b"\x89PNG") {
        return Err("render frame is not a PNG".into());
    }
    client
        .close()
        .await
        .map_err(|err| format!("close: {err}"))?;
    drop(client);

    let options = RunLocalOptions::parse(address)
        .map_err(|err| err.to_string())?
        .for_episodes(EPISODES)
        .base_seed(BASE_SEED)
        .trial_index_base(0);
    let report = ModelWorker::new(ZeroModel)
        .run_local_async(options)
        .await
        .map_err(|err| format!("run: {err}"))?;
    eprintln!(
        "run report: episodes={} steps={}",
        report.total_episodes, report.total_steps
    );
    if report.total_episodes != EPISODES as i64 || report.total_steps != 2 * EPISODES as i64 {
        return Err("unexpected run report".into());
    }
    Ok(())
}
