use std::sync::{Arc, Mutex as StdMutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh::ServeOptions;
use rlmesh::binding::env::{WireEnvAdapter, WireLaneAdapter};
use rlmesh::binding::lifecycle::{ShutdownTrigger, await_close_with_timeout};
use rlmesh_spaces::EnvContract;

use super::py_environment::{
    PyServerEnv, build_lane_server_env, build_vector_server_env, run_foreground,
};
use crate::lifecycle::PyServeOptions;
use crate::spaces::env_contract_to_py;
use crate::types::to_py_err;

enum BoundListener {
    Lockstep(Box<rlmesh::lockstep::LockstepBinding>),
}

type ServeResult = Result<(), String>;

const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

struct ServerResources {
    env: PyServerEnv,
    runtime: tokio::runtime::Runtime,
    listener: BoundListener,
    options: ServeOptions,
    close_env_on_shutdown: bool,
    announce: Option<rlmesh::binding::Announce>,
}

enum ServerState {
    Ready(Box<ServerResources>),
    StartingBackground,
    RunningForeground,
    RunningBackground(JoinHandle<ServeResult>),
    Stopped,
}

/// Python wrapper for the RLMesh EnvService server.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyEnvServer {
    id: u64,
    state: StdMutex<ServerState>,
    address: String,
    env_contract: EnvContract,
    shutdown: ShutdownTrigger,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyVectorEnvServer {
    /// Create a new vectorized RLMesh environment server.
    /// # Arguments
    /// * `env` - Python gymnasium.vector.VectorEnv object
    /// * `address` - Optional bind address shortcut
    #[new]
    #[pyo3(signature = (env, address=None, *, options=None, native_values=false, close_env_on_shutdown=true, announce=true))]
    fn new(
        env: Py<PyAny>,
        address: Option<&str>,
        options: Option<PyServeOptions>,
        native_values: bool,
        close_env_on_shutdown: bool,
        announce: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: construct_server(
                build_vector_server_env(env, native_values)?,
                address,
                options,
                close_env_on_shutdown,
                announce,
            )?,
        })
    }

    fn address(&self) -> String {
        self.inner.address()
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "EnvContract", imports = ()))]
    fn env_contract<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.inner.env_contract(py)
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "EnvContract", imports = ()))]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.inner.spec(py)
    }

    fn serve(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.serve(py)
    }

    fn start(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.start(py)
    }

    #[pyo3(signature = (timeout=None))]
    fn wait(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        self.inner.wait(py, timeout)
    }

    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.shutdown(py)
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyVectorEnvServer {
    inner: PyEnvServer,
}

static NEXT_SERVER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn construct_server(
    mut py_env: PyServerEnv,
    address: Option<&str>,
    options: Option<PyServeOptions>,
    close_env_on_shutdown: bool,
    announce: bool,
) -> PyResult<PyEnvServer> {
    crate::telemetry::init_tracing("env_server");
    let options = options.map(PyServeOptions::into_rust).unwrap_or_default();
    let shutdown = ShutdownTrigger::new();
    let id = NEXT_SERVER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    py_env.set_server_id(id);

    let mut env_contract = py_env.env_contract().clone();
    env_contract.num_envs = u32::try_from(py_env.num_envs()).unwrap_or(u32::MAX);

    let runtime = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "failed to create tokio runtime: {}",
            e
        ))
    })?;

    let address = rlmesh::ext::ConnectAddress::env(address.unwrap_or(":0/env"))
        .map_err(to_py_err)?
        .as_str();
    let address = address.as_str();
    rlmesh::lockstep::check(env_contract.clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    // Without the GIL: the host it joins may be a Python thread of this process still starting.
    let binding =
        Python::attach(|py| py.detach(|| runtime.block_on(rlmesh::lockstep::bind(address))))
            .map_err(|e| match e {
                rlmesh::lockstep::GymError::Config(why) => {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(why)
                }
                other => PyErr::new::<pyo3::exceptions::PyConnectionError, _>(other.to_string()),
            })?;
    Ok(PyEnvServer {
        id,
        address: binding.address(),
        env_contract,
        state: StdMutex::new(ServerState::Ready(Box::new(ServerResources {
            env: py_env,
            runtime,
            listener: BoundListener::Lockstep(Box::new(binding)),
            options,
            close_env_on_shutdown,
            announce: announce.then(|| rlmesh::binding::Announce::from_env(address)),
        }))),
        shutdown,
    })
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyEnvServer {
    /// Create a new RLMesh environment server.
    /// # Arguments
    /// * `env` - Python gymnasium.Env object, or a list of them to serve as the
    ///   lanes of one endpoint (`num_envs = len(list)`)
    /// * `address` - Optional bind address shortcut
    #[new]
    #[pyo3(signature = (env, address=None, *, options=None, native_values=false, close_env_on_shutdown=true, announce=true))]
    fn new(
        py: Python<'_>,
        env: Py<PyAny>,
        address: Option<&str>,
        options: Option<PyServeOptions>,
        native_values: bool,
        close_env_on_shutdown: bool,
        announce: bool,
    ) -> PyResult<Self> {
        let lanes: Vec<Py<PyAny>> = match env.bind(py).cast::<pyo3::types::PyList>() {
            Ok(list) => list.iter().map(|lane| lane.unbind()).collect(),
            Err(_) => vec![env],
        };
        construct_server(
            build_lane_server_env(lanes, native_values)?,
            address,
            options,
            close_env_on_shutdown,
            announce,
        )
    }

    /// Get the server address.
    fn address(&self) -> String {
        self.address.clone()
    }

    /// Get the environment contract served by this endpoint.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "EnvContract", imports = ()))]
    fn env_contract<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        env_contract_to_py(py, &self.env_contract)
    }

    /// Alias for env_contract.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "EnvContract", imports = ()))]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        env_contract_to_py(py, &self.env_contract)
    }

    /// Start serving (blocking).
    /// Releases the GIL while running so other Python threads can execute.
    /// The env stays on this thread: every reset/step/render/close runs here,
    /// so an env built on the main thread is driven from it.
    fn serve(&self, py: Python<'_>) -> PyResult<()> {
        let resources = take_resources(&self.state, "serve", ServerState::RunningForeground)?;
        let address = self.address.clone();
        let shutdown = self.shutdown.clone();

        let result = py.detach(move || run_server(resources, address, shutdown, true));

        let mut state = self.state.lock().expect("server state mutex poisoned");
        *state = ServerState::Stopped;
        drop(state);

        result.map_err(serve_failed)
    }

    /// Start serving on a background thread.
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        let resources = take_resources(&self.state, "start", ServerState::StartingBackground)?;
        let address = self.address.clone();
        let shutdown = self.shutdown.clone();

        let handle = std::thread::Builder::new()
            .name("rlmesh-env-server".to_string())
            .spawn(move || run_server(resources, address, shutdown, false))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "failed to spawn server thread: {}",
                    e
                ))
            })?;

        let action = {
            let mut state = self.state.lock().expect("server state mutex poisoned");
            match std::mem::replace(&mut *state, ServerState::Stopped) {
                ServerState::StartingBackground => {
                    *state = ServerState::RunningBackground(handle);
                    None
                }
                ServerState::Stopped => Some(handle),
                previous => {
                    *state = previous;
                    None
                }
            }
        };

        if let Some(handle) = action {
            py.detach(|| join_background_server(handle))?;
        }

        Ok(())
    }

    /// Wait for a background server to stop.
    #[pyo3(signature = (timeout=None))]
    fn wait(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let timeout = parse_wait_timeout(timeout)?;
        py.detach(|| wait_background_server(&self.state, timeout))
    }

    /// Shutdown the server.
    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        self.shutdown.trigger("local shutdown");
        // Never join from an env callback: shutdown would wait for that same callback.
        if super::py_environment::in_server_callback(self.id) {
            let mut state = self.state.lock().expect("server state mutex poisoned");
            if matches!(*state, ServerState::RunningForeground) {
                *state = ServerState::Stopped;
            }
            return Ok(());
        }

        enum ShutdownAction {
            None,
            Cleanup(Box<ServerResources>),
            Join(JoinHandle<ServeResult>),
        }

        let action = {
            let mut state = self.state.lock().expect("server state mutex poisoned");
            match std::mem::replace(&mut *state, ServerState::Stopped) {
                ServerState::Ready(resources) => ShutdownAction::Cleanup(resources),
                ServerState::RunningBackground(handle) => ShutdownAction::Join(handle),
                ServerState::StartingBackground
                | ServerState::RunningForeground
                | ServerState::Stopped => ShutdownAction::None,
            }
        };

        match action {
            ShutdownAction::None => Ok(()),
            ShutdownAction::Cleanup(resources) => py
                .detach(|| cleanup_ready_resources(*resources))
                .map_err(serve_failed),
            ShutdownAction::Join(handle) => py.detach(|| join_background_server(handle)),
        }
    }
}

async fn shutdown_signal() -> &'static str {
    tokio::signal::ctrl_c()
        .await
        .expect("install Ctrl+C handler");
    tracing::info!(pid = std::process::id(), "shutdown requested via SIGINT");
    "sigint"
}

pub(crate) fn spawn_signal_shutdown(shutdown: ShutdownTrigger) {
    tokio::spawn(async move {
        let reason = shutdown_signal().await;
        shutdown.trigger(reason);
    });
}

impl Drop for PyEnvServer {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        self.shutdown.trigger("drop");

        // After interpreter finalization, attaching to Python below would be unsound.
        if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
            return;
        }

        if let Ok(mut state) = self.state.lock() {
            let previous = std::mem::replace(&mut *state, ServerState::Stopped);
            drop(state);

            if let ServerState::Ready(resources) = previous {
                Python::attach(|py| {
                    let _ = py.detach(|| cleanup_ready_resources(*resources));
                });
            }
        }
    }
}

fn py_runtime_err(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message.into())
}

fn py_value_err(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

fn parse_wait_timeout(timeout: Option<f64>) -> PyResult<Option<Duration>> {
    timeout
        .map(|value| {
            Duration::try_from_secs_f64(value)
                .map_err(|_| py_value_err("timeout must be a non-negative finite float or None"))
        })
        .transpose()
}

fn wait_background_server(
    state: &StdMutex<ServerState>,
    timeout: Option<Duration>,
) -> PyResult<bool> {
    let start = Instant::now();

    loop {
        let handle = {
            let mut guard = state.lock().expect("server state mutex poisoned");
            match &*guard {
                ServerState::RunningBackground(handle) if handle.is_finished() => {
                    match std::mem::replace(&mut *guard, ServerState::Stopped) {
                        ServerState::RunningBackground(handle) => Some(handle),
                        _ => unreachable!("server state changed while locked"),
                    }
                }
                ServerState::RunningBackground(_) | ServerState::StartingBackground => None,
                ServerState::Stopped => return Ok(true),
                ServerState::Ready(_) => {
                    return Err(py_runtime_err("wait() called before start()"));
                }
                ServerState::RunningForeground => {
                    return Err(py_runtime_err(
                        "wait() called while serve() is already running in the foreground",
                    ));
                }
            }
        };

        if let Some(handle) = handle {
            join_background_server(handle)?;
            return Ok(true);
        }

        let nap = match timeout {
            Some(timeout) => {
                let elapsed = start.elapsed();
                if elapsed >= timeout {
                    return Ok(false);
                }
                (timeout - elapsed).min(Duration::from_millis(10))
            }
            None => Duration::from_millis(10),
        };
        std::thread::sleep(nap);
        // Poll for a pending signal each iteration: no bytecode runs while the GIL is released,
        // so Ctrl-C would otherwise be lost.
        crate::client::check_signals()?;
    }
}

fn take_resources(
    state: &StdMutex<ServerState>,
    action: &str,
    next_state: ServerState,
) -> PyResult<ServerResources> {
    let mut guard = state.lock().expect("server state mutex poisoned");
    let previous = std::mem::replace(&mut *guard, next_state);

    match previous {
        ServerState::Ready(resources) => Ok(*resources),
        ServerState::StartingBackground => {
            *guard = ServerState::StartingBackground;
            Err(py_runtime_err(format!(
                "{action}() called while server is starting in the background"
            )))
        }
        ServerState::RunningForeground => {
            *guard = ServerState::RunningForeground;
            Err(py_runtime_err(format!(
                "{action}() called while server is already running"
            )))
        }
        ServerState::RunningBackground(handle) => {
            *guard = ServerState::RunningBackground(handle);
            Err(py_runtime_err(format!(
                "{action}() called while server is already running in the background"
            )))
        }
        ServerState::Stopped => {
            *guard = ServerState::Stopped;
            Err(py_runtime_err(format!(
                "{action}() called after the server has been stopped"
            )))
        }
    }
}

fn run_server(
    resources: ServerResources,
    address: String,
    shutdown: ShutdownTrigger,
    foreground: bool,
) -> ServeResult {
    let ServerResources {
        mut env,
        runtime,
        listener,
        options,
        close_env_on_shutdown,
        announce,
    } = resources;

    let jobs = foreground.then(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        env.dispatch_foreground(tx);
        rx
    });
    if !foreground {
        let (tx, rx) = std::sync::mpsc::channel();
        if env.dispatch_vector(tx) {
            std::thread::Builder::new()
                .name("rlmesh-vector-env".to_string())
                .spawn(move || run_foreground(rx, || {}))
                .map_err(|e| format!("failed to spawn the vector env thread: {e}"))?;
        }
    }
    let sigterm = foreground.then(|| rlmesh::binding::foreground_sigterm_scope(shutdown.clone()));
    let signal_shutdown = shutdown.clone();

    let serve = move || {
        runtime.block_on(async move {
            tracing::info!(
                "EnvService serving on {}",
                rlmesh_session::address::redact(&address)
            );
            if foreground {
                spawn_signal_shutdown(shutdown.clone());
            }
            match env {
                PyServerEnv::Lanes(lanes) => {
                    let lanes = WireLaneAdapter::new(lanes).map_err(|err| err.to_string())?;
                    run_env_server(
                        lanes,
                        listener,
                        options,
                        shutdown,
                        close_env_on_shutdown,
                        announce,
                    )
                    .await
                }
                PyServerEnv::Vector(env) => {
                    run_env_server(
                        WireEnvAdapter::new(*env),
                        listener,
                        options,
                        shutdown,
                        close_env_on_shutdown,
                        announce,
                    )
                    .await
                }
            }
        })
    };

    let Some(jobs) = jobs else {
        return serve();
    };
    let server = std::thread::Builder::new()
        .name("rlmesh-env-server".to_string())
        .spawn(serve)
        .map_err(|e| format!("failed to spawn server thread: {e}"))?;
    run_foreground(jobs, || {
        if crate::client::check_signals().is_err() {
            signal_shutdown.trigger("sigint");
        }
    });
    let result = server
        .join()
        .unwrap_or_else(|_| Err("server thread panicked".to_string()));
    drop(sigterm);
    result
}

async fn run_env_server<E>(
    env: E,
    listener: BoundListener,
    options: ServeOptions,
    shutdown: ShutdownTrigger,
    close_env_on_shutdown: bool,
    announce: Option<rlmesh::binding::Announce>,
) -> ServeResult
where
    E: rlmesh::lockstep::Served,
{
    let close_timeout = Some(options.close_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE));
    let env = Arc::new(env);
    let workflow_edition = options.workflow_edition.clone();
    let serve_result = match listener {
        BoundListener::Lockstep(binding) => {
            let cancel = tokio_util::sync::CancellationToken::new();
            let stop = {
                let (shutdown, cancel) = (shutdown.clone(), cancel.clone());
                tokio::spawn(async move {
                    shutdown.cancelled().await;
                    cancel.cancel();
                })
            };
            let served = binding
                .serve(
                    Arc::clone(&env) as _,
                    rlmesh::lockstep::ServeOptions {
                        workflow_edition,
                        record: std::env::var_os("RLMESH_LOCKSTEP_RECORD").map(Into::into),
                        announce,
                    },
                    cancel,
                )
                .await
                .map_err(|err| err.to_string());
            stop.abort();
            served
        }
    };

    let close_result = if close_env_on_shutdown {
        close_env(env, close_timeout).await
    } else {
        Ok(())
    };
    match (serve_result, close_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) => Err(err),
        (Ok(()), Err(err)) => Err(err),
        (Err(serve_err), Err(close_err)) => Err(format!(
            "environment server failed: {serve_err}; close hook failed: {close_err}"
        )),
    }
}

async fn close_env<E>(env: Arc<E>, close_timeout: Option<std::time::Duration>) -> ServeResult
where
    E: rlmesh::lockstep::Served,
{
    let close = async { rlmesh::lockstep::Served::close(env.as_ref()).await };
    await_close_with_timeout(close, close_timeout)
        .await
        .map_err(|timeout| format!("close timed out after {}ms", timeout.as_millis()))?
        .map_err(|err| err.message)
}

fn cleanup_ready_resources(resources: ServerResources) -> ServeResult {
    let ServerResources {
        mut env,
        runtime,
        listener,
        options,
        close_env_on_shutdown,
        ..
    } = resources;

    match listener {
        BoundListener::Lockstep(binding) => runtime.block_on(binding.close()),
    }

    if !close_env_on_shutdown {
        return Ok(());
    }
    let close_timeout = Some(options.close_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE));
    runtime.block_on(async move {
        await_close_with_timeout(env.close(), close_timeout)
            .await
            .map_err(|timeout| format!("close timed out after {}ms", timeout.as_millis()))?
            .map_err(|err| match err {
                rlmesh_spaces::errors::EnvRuntimeError::Runtime(message)
                | rlmesh_spaces::errors::EnvRuntimeError::Recoverable(message)
                | rlmesh_spaces::errors::EnvRuntimeError::InvalidValue(message)
                | rlmesh_spaces::errors::EnvRuntimeError::InvalidSpace(message) => message,
                other => other.to_string(),
            })
    })
}

fn join_background_server(handle: JoinHandle<ServeResult>) -> PyResult<()> {
    match handle.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(serve_failed(err)),
        Err(_) => Err(serve_failed(
            "the background server thread panicked".to_string(),
        )),
    }
}

fn serve_failed(message: String) -> PyErr {
    crate::types::RLMeshException::new_err(format!("serve failed: {message}"))
}
