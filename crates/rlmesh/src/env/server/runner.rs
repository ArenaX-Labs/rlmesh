//! The serving thread: running the lanes or vector until a stop, cancelling
//! a background serve, and cleaning up what it held.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::lifecycle::{ShutdownTrigger, await_close_with_timeout};

use super::wrappers::Marked;
use super::{Listener, Resources, Served, Shared, in_callback, lock};
use crate::env::sync_wire::{Actor, ActorRunner, SyncLane, SyncVector};
use crate::env::wire::{WireEnvAdapter, WireLaneAdapter};
use crate::env::{Env, VectorEnv};
use crate::{Error, Result, ServeOptions};

/// The foreground job runner of a lane or a vector env, if this serve keeps
/// one on the calling thread.
enum Runner {
    None,
    Lane(ActorRunner<Box<dyn Env + Send>>),
    Vector(ActorRunner<Box<dyn VectorEnv + Send>>),
}

impl Runner {
    fn run(self) {
        match self {
            Runner::None => {}
            Runner::Lane(runner) => runner.run(),
            Runner::Vector(runner) => runner.run(),
        }
    }
}

/// A background factory serve owns its cancellation subscription until serving
/// ends. Joining the serve thread releases the watcher before returning its result.
pub(super) struct CancelProbe {
    pub(super) subscription: crate::ext::CancelSubscription,
    pub(super) trigger: ShutdownTrigger,
    pub(super) cancelled: Mutex<bool>,
}
impl CancelProbe {
    pub(super) fn sample(&self) {
        let mut cancelled = lock(&self.cancelled);
        if !*cancelled && self.subscription.fired() && self.trigger.trigger("cancelled") {
            *cancelled = true;
        }
    }
}
impl Shared {
    pub(super) fn trigger_shutdown(&self, reason: &str) {
        // A firing before this call must compete synchronously, even between
        // watcher polls. Once another trigger wins, later firings cannot alter
        // the serve result, including firings during drain/close.
        if let Some(probe) = lock(&self.background_cancel).upgrade() {
            probe.sample();
        }
        self.shutdown.trigger(reason);
    }
}
pub(super) struct BackgroundCancel {
    pub(super) stop: Arc<std::sync::atomic::AtomicBool>,
    pub(super) probe: Arc<CancelProbe>,
    pub(super) thread: Option<JoinHandle<()>>,
}
impl BackgroundCancel {
    pub(super) fn start(
        subscription: crate::ext::CancelSubscription,
        trigger: ShutdownTrigger,
    ) -> Result<Self> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = Arc::new(AtomicBool::new(false));
        let probe = Arc::new(CancelProbe {
            subscription,
            trigger,
            cancelled: Mutex::new(false),
        });
        let (stop_flag, watched) = (Arc::clone(&stop), Arc::clone(&probe));
        let thread = std::thread::Builder::new()
            .name("rlmesh-factory-cancel".into())
            .spawn(move || {
                while !stop_flag.load(Ordering::SeqCst) {
                    watched.sample();
                    std::thread::sleep(Duration::from_millis(20));
                }
                // Retain the subscription through drain/close, so later firings
                // remain consumed without changing the selected serve result.
            })
            .map_err(|err| {
                Error::Internal(format!("failed to spawn cancellation watcher: {err}"))
            })?;
        Ok(Self {
            stop,
            probe,
            thread: Some(thread),
        })
    }
    pub(super) fn stop(mut self) -> bool {
        self.finish();
        *lock(&self.probe.cancelled)
    }
    fn finish(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Drop for BackgroundCancel {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Serve until the shutdown trigger fires, then drain and close. In the
/// foreground the calling thread runs lane 0's (or the vector env's) jobs
/// while the transport runs on a helper thread.
pub(super) fn run(resources: Resources, shared: &Arc<Shared>, foreground: bool) -> Result<()> {
    run_inner(resources, shared, foreground)
}

fn run_inner(resources: Resources, shared: &Arc<Shared>, foreground: bool) -> Result<()> {
    let Resources {
        served,
        runtime,
        listener,
        options,
        close_timeout,
        close_env_on_shutdown,
    } = resources;
    let id = shared.id;
    let contract = shared.contract.clone();
    let parts = ServeParts {
        runtime,
        listener,
        options,
        close_timeout,
        close_env_on_shutdown,
        shutdown: shared.shutdown.clone(),
    };
    match served {
        Served::Lanes(lanes) => {
            let mut wire_lanes = Vec::with_capacity(lanes.len());
            let mut runner = Runner::None;
            for (index, lane) in lanes.into_iter().enumerate() {
                let lane: Box<dyn Env + Send> = Box::new(Marked {
                    id,
                    env: Some(lane),
                });
                if foreground && index == 0 {
                    let (relayed, lane_runner) = SyncLane::relayed(lane, &contract)?;
                    wire_lanes.push(relayed);
                    runner = Runner::Lane(lane_runner);
                } else {
                    wire_lanes.push(SyncLane::new(lane, &contract)?);
                }
            }
            let mut adapter = WireLaneAdapter::new(wire_lanes)
                .map_err(|err| Error::invalid_argument(err.to_string()))?;
            adapter.track_abandoned(Arc::clone(&shared.abandoned));
            drive(adapter, runner, parts)
        }
        Served::Vector(env) => {
            let env: Box<dyn VectorEnv + Send> = Box::new(Marked { id, env: Some(env) });
            let (actor, runner) = if foreground {
                let (actor, runner) = Actor::foreground(env);
                (actor, Runner::Vector(runner))
            } else {
                let (actor, thread) = Actor::spawn("rlmesh-vector-env".to_string(), env)?;
                lock(&shared.abandoned).push(thread);
                (actor, Runner::None)
            };
            drive(
                WireEnvAdapter::new(SyncVector::new(actor, &contract)?),
                runner,
                parts,
            )
        }
    }
}

/// What the transport needs to serve one env.
struct ServeParts {
    runtime: tokio::runtime::Runtime,
    listener: Listener,
    options: ServeOptions,
    close_timeout: Duration,
    close_env_on_shutdown: bool,
    shutdown: ShutdownTrigger,
}

/// What [`drive`] serves: a transport env, natively too with the `lockstep`
/// feature.
trait Driven: crate::lockstep::Served {}
impl<E: crate::lockstep::Served> Driven for E {}

/// Serve `environment`; with a foreground runner, the transport runs on a
/// helper thread while this thread runs the runner's jobs.
fn drive<E: Driven + Send + Sync + 'static>(
    environment: E,
    runner: Runner,
    parts: ServeParts,
) -> Result<()> {
    let ServeParts {
        runtime,
        listener,
        options,
        close_timeout,
        close_env_on_shutdown,
        shutdown,
    } = parts;
    let serve = move || -> Result<()> {
        runtime.block_on(async move {
            let environment = Arc::new(environment);
            let serve_result = match listener {
                Listener::Lockstep(binding) => {
                    let cancel = tokio_util::sync::CancellationToken::new();
                    let stop = {
                        let (shutdown, cancel) = (shutdown.clone(), cancel.clone());
                        tokio::spawn(async move {
                            shutdown.cancelled().await;
                            cancel.cancel();
                        })
                    };
                    let options = crate::lockstep::ServeOptions {
                        workflow_edition: options.workflow_edition.clone(),
                        record: std::env::var_os("RLMESH_LOCKSTEP_RECORD").map(Into::into),
                    };
                    let served = binding
                        .serve(Arc::clone(&environment) as _, options, cancel)
                        .await
                        .map_err(|err| Error::server(err.to_string()));
                    stop.abort();
                    served
                }
            };
            // A transport failure also begins stopping before close callbacks.
            // Later token firings cannot replace that selected serve error.
            shutdown.trigger("serve ended");
            let close_result = if close_env_on_shutdown {
                let close = async { crate::lockstep::Served::close(&*environment).await };
                match await_close_with_timeout(close, Some(close_timeout)).await {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(err)) => Err(Error::Internal(err.message)),
                    Err(timeout) => Err(Error::Internal(format!(
                        "close timed out after {}ms",
                        timeout.as_millis()
                    ))),
                }
            } else {
                Ok(())
            };
            // The env (and with it the lane threads and a foreground actor)
            // drops here, which ends a foreground runner.
            drop(environment);
            serve_failure(serve_result, close_result)
        })
    };

    if matches!(runner, Runner::None) {
        return serve();
    }
    let server = std::thread::Builder::new()
        .name("rlmesh-env-server".to_string())
        .spawn(serve)
        .map_err(|err| Error::Internal(format!("failed to spawn server thread: {err}")))?;
    runner.run();
    server.join().unwrap_or_else(|_| {
        Err(Error::Internal(
            "serve failed: the server thread panicked".into(),
        ))
    })
}

/// The one serve-failure text: `serve failed: <message>`.
fn serve_failure(serve: Result<()>, close: Result<()>) -> Result<()> {
    let text = |err: Error| match err {
        Error::Internal(message) => message,
        other => other.to_string(),
    };
    match (serve, close) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) | (Ok(()), Err(err)) => {
            Err(Error::Internal(format!("serve failed: {}", text(err))))
        }
        (Err(serve), Err(close)) => Err(Error::Internal(format!(
            "serve failed: {}; close failed: {}",
            text(serve),
            text(close)
        ))),
    }
}

/// Release a never-served server: release the listener (stopping a session host it runs) and close the envs (when
/// `close_env_on_shutdown`), bounded by `close_timeout`.
pub(super) fn cleanup_ready(resources: Resources, shared: &Arc<Shared>) -> Result<()> {
    let Resources {
        served,
        runtime,
        listener,
        close_timeout,
        close_env_on_shutdown,
        ..
    } = resources;
    listener.release(&runtime);
    drop(runtime);
    let id = shared.id;
    if !close_env_on_shutdown {
        in_callback(id, || drop(served));
        return Ok(());
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("rlmesh-env-close".to_string())
        .spawn(move || {
            let result = in_callback(id, || match served {
                Served::Lanes(mut lanes) => lanes.iter_mut().try_for_each(|lane| {
                    in_callback(id, || lane.close()).map_err(|err| {
                        super::super::sync_wire::authored_failure("close", &err).to_string()
                    })
                }),
                Served::Vector(mut env) => in_callback(id, || env.close()).map_err(|err| {
                    super::super::sync_wire::authored_failure("close", &err).to_string()
                }),
            });
            let _ = tx.send(result);
        })
        .map_err(|err| Error::Internal(format!("failed to spawn a close thread: {err}")))?;
    let result = rx.recv_timeout(close_timeout);
    if result.is_ok() {
        let _ = thread.join();
    } else {
        lock(&shared.abandoned).push(thread);
    }
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(Error::Internal(format!(
            "serve failed: {}",
            message.trim_start_matches("runtime error: ")
        ))),
        Err(_) => Err(Error::Internal(format!(
            "serve failed: close timed out after {}ms",
            close_timeout.as_millis()
        ))),
    }
}
