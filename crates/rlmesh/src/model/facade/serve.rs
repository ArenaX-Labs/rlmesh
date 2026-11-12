//! Serving a model on a session, and running it locally against an env.

use std::sync::Arc;
use std::time::Duration;

use crate::lifecycle::ShutdownTrigger;

use super::queue::Shared;
use super::{CancelWatcher, Model, ModelServeOptions};
use crate::editions::EditionSources;
use crate::{Error, Result};

/// One serve, blocking the calling thread: validate, resolve the edition,
/// join the session of the participant address as the model it names, as
/// Python's serve does, serve until stopped, then close the model.
pub(super) fn serve_blocking(
    shared: &Arc<Shared>,
    address: &str,
    options: ModelServeOptions,
    foreground: bool,
    dropped: Option<crate::ext::CancelToken>,
) -> Result<()> {
    let ModelServeOptions {
        name,
        options,
        cancel,
        ext,
        batching,
    } = options;
    if cancel.as_ref().is_some_and(crate::ext::CancelToken::take) {
        return Err(Error::Cancelled);
    }
    let participant = match name.as_deref() {
        Some("") => return Err(Error::invalid_argument("name must not be empty")),
        Some(name) => format!("{}/{name}", address.trim_end_matches('/')),
        None => address.to_string(),
    };
    let address = participant.as_str();
    if !crate::address::is_v2(address) {
        return Err(Error::invalid_argument(format!(
            "serve: {address:?} is a v1 model endpoint, which rlmesh 0.2 no longer serves; \
             serve on a participant address (e.g. 'lab:7447/policy')"
        )));
    }
    shared.admit()?;
    let serve = options.unwrap_or_default();
    serve.validate()?;
    let ext = ext.unwrap_or_default();
    if ext.predict_concurrency == Some(0) {
        return Err(Error::invalid_argument(
            "predict_concurrency must be at least 1",
        ));
    }
    let tokens: Vec<_> = cancel.into_iter().chain(dropped).collect();
    {
        if ext.predict_concurrency.is_some() {
            return Err(Error::invalid_argument(
                "predict_concurrency: a session model runs one forward at a time (gone in \
                 rlmesh 0.2 for a session address)",
            ));
        }
        // The call's edition wins over the environment variable (COR-6).
        let edition = crate::editions::resolve_native(EditionSources {
            call: serve.workflow_edition.as_deref(),
            option: None,
            declared: shared.decl.workflow_edition.as_deref(),
        })?;
        let join = serve.close_timeout.unwrap_or(SESSION_JOIN);
        let batching = batching.unwrap_or_default();
        serve_session(shared, address, edition, batching, join, foreground, tokens)
    }
}

/// How long a session model's join and close may take when the options set
/// no `close_timeout`, as Python's serve.
const SESSION_JOIN: Duration = Duration::from_secs(30);

/// Joins the session of the participant address `address` as the model and
/// answers whoever drives it there, until stopped.
fn serve_session(
    shared: &Arc<Shared>,
    address: &str,
    edition: Option<String>,
    batching: crate::BatchOptions,
    join: Duration,
    foreground: bool,
    tokens: Vec<crate::ext::CancelToken>,
) -> Result<()> {
    let handler = super::super::bridge::engine_handler(Arc::clone(shared));
    // The envelope a driver reads the model's own horizon (and spec) from.
    let describe = crate::describe::describe_json(
        crate::describe::Describable::built(shared),
        crate::describe::DescribeJsonOptions {
            workflow_edition: Some(edition.clone().unwrap_or_default()),
            ..Default::default()
        },
    )?;
    let service = Arc::new(
        crate::lockstep::ModelService::new(handler, describe)
            .declaring(edition.as_deref())
            .batching(batching)?,
    );
    let binding = crate::run::drive::block_on_library(crate::lockstep::ModelBinding::bind(address))
        .map_err(crate::env::client::lockstep_error)?;
    tracing::info!(address = %binding.address(), "model serving");
    let trigger = ShutdownTrigger::new();
    let stopped = trigger.clone();
    let serve = async move {
        let cancel = tokio_util::sync::CancellationToken::new();
        let stop = cancel.clone();
        let relay = tokio::spawn(async move {
            stopped.cancelled_owned().await;
            stop.cancel();
        });
        let result = binding.serve(service, join, cancel).await;
        relay.abort();
        result.map_err(crate::env::client::lockstep_error)
    };
    serve_until_stopped(shared, trigger, foreground, tokens, join, serve)
}

/// Serves until `trigger` fires (a signal, a cancel token, the model's
/// close), then closes the model: the cancellation class when a token or
/// SIGINT stopped it.
fn serve_until_stopped(
    shared: &Arc<Shared>,
    trigger: ShutdownTrigger,
    foreground: bool,
    tokens: Vec<crate::ext::CancelToken>,
    close_timeout: Duration,
    serve: impl std::future::Future<Output = Result<()>>,
) -> Result<()> {
    let serve_id = shared.register_serve(trigger.clone())?;
    let signals = foreground.then(|| crate::env::signals::enter(trigger.clone()));
    let watcher = CancelWatcher::start(tokens, trigger);
    let result = crate::run::drive::block_on_library(serve);
    shared.deregister_serve(serve_id);
    let cancelled = watcher.stop();
    let interrupted = signals.as_ref().is_some_and(|scope| scope.interrupted());
    drop(signals);
    let close = shared.close(Some(close_timeout));
    match (result, close) {
        (Err(err), _) => Err(err),
        (Ok(()), Err(err)) => Err(err),
        (Ok(()), Ok(())) if cancelled || interrupted => Err(Error::Cancelled),
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Drive `model` in-process against a remote env (the binding's run path
/// until `run` lands): the run borrows the model and never closes it.
pub(crate) async fn run_local(
    model: &Model,
    options: super::super::worker::RunLocalOptions,
    cancellation: tokio_util::sync::CancellationToken,
    hooks: Arc<dyn crate::run::runtime::RuntimeHooks>,
) -> Result<crate::run::runtime::RuntimeReport> {
    let shared = model.shared();
    shared.check_not_reentrant("run")?;
    shared.admit()?;
    let _slot = shared.claim_run()?;
    if options.execution_horizon > 1 && !shared.decl.has_chunk() {
        crate::warnings::warn(
            crate::WarningCategory::UserWarning,
            format!(
                "execution_horizon={} was requested but the model defines no predict_chunk(); \
                 running un-chunked.",
                options.execution_horizon
            ),
        )?;
    }
    let mut handler = super::super::bridge::engine_handler(Arc::clone(shared));
    super::super::local::run_local(&mut handler, options, cancellation, hooks).await
}
