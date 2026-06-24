//! PyEnvServer - Python-facing RLMesh environment server.

#[cfg(unix)]
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh::env::{ScalarEnvAdapter, WireEnvAdapter};
use rlmesh::{BindAddress, ServeOptions};
use rlmesh_grpc::env::{Environment, env_service_from_shared};
use rlmesh_grpc::lifecycle::{
    ShutdownTrigger, await_close_with_timeout, await_server_shutdown, start_idle_shutdown,
};
use rlmesh_spaces::EnvContract;
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use tokio_stream::wrappers::TcpListenerStream;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;

use super::py_environment::{PyServerEnv, build_scalar_server_env, build_vector_server_env};
use crate::lifecycle::PyServeOptions;
use crate::spaces::env_contract_to_py;
use crate::types::to_py_err;

enum BoundListener {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix {
        listener: UnixListener,
        path: PathBuf,
    },
}

type ServeResult = Result<(), String>;

/// Default shutdown cap for embedded Python servers.
const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

struct ServerResources {
    env: PyServerEnv,
    runtime: tokio::runtime::Runtime,
    listener: BoundListener,
    options: ServeOptions,
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
    #[pyo3(signature = (env, address=None, *, options=None))]
    fn new(
        env: Py<PyAny>,
        address: Option<&str>,
        options: Option<PyServeOptions>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: construct_server(env, address, options, build_vector_server_env)?,
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

fn construct_server(
    env: Py<PyAny>,
    address: Option<&str>,
    options: Option<PyServeOptions>,
    build_env: fn(Py<PyAny>) -> PyResult<PyServerEnv>,
) -> PyResult<PyEnvServer> {
    crate::telemetry::init_tracing("env_server");
    let shutdown = ShutdownTrigger::new();

    let py_env = build_env(env)?;
    let env_contract = py_env.env_contract().clone();

    let runtime = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "failed to create tokio runtime: {}",
            e
        ))
    })?;

    let bind_target = match address {
        Some(address) => BindAddress::parse(address).map_err(to_py_err)?,
        None => BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        },
    };

    let (address, listener) = match bind_target {
        BindAddress::Tcp { host, port } => {
            let host_owned = host.to_string();
            let listener: TcpListener = runtime
                .block_on(async { TcpListener::bind((host_owned.as_str(), port)).await })
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyConnectionError, _>(format!(
                        "failed to bind tcp listener: {}",
                        e
                    ))
                })?;
            let bound_addr = listener.local_addr().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "failed to read bound tcp address: {}",
                    e
                ))
            })?;

            (
                format!("tcp://{}", bound_addr),
                BoundListener::Tcp(listener),
            )
        }
        BindAddress::Unix { path: socket_path } => {
            #[cfg(not(unix))]
            {
                let _ = socket_path;
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "unix sockets are not supported on Windows; use tcp://host:port instead",
                ));
            }

            #[cfg(unix)]
            {
                if socket_path.exists() {
                    std::fs::remove_file(&socket_path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyConnectionError, _>(format!(
                            "failed to remove stale unix socket '{}': {}",
                            socket_path.display(),
                            e
                        ))
                    })?;
                }

                // UnixListener registration requires an entered Tokio runtime.
                let listener = {
                    let _guard = runtime.enter();
                    UnixListener::bind(&socket_path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyConnectionError, _>(format!(
                            "failed to bind unix socket '{}': {}",
                            socket_path.display(),
                            e
                        ))
                    })?
                };

                (
                    format!("unix://{}", socket_path.display()),
                    BoundListener::Unix {
                        listener,
                        path: socket_path,
                    },
                )
            }
        }
    };

    Ok(PyEnvServer {
        address,
        env_contract,
        state: StdMutex::new(ServerState::Ready(Box::new(ServerResources {
            env: py_env,
            runtime,
            listener,
            options: options.map(PyServeOptions::into_rust).unwrap_or_default(),
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
    /// * `env` - Python gymnasium.Env object
    /// * `address` - Optional bind address shortcut
    #[new]
    #[pyo3(signature = (env, address=None, *, options=None))]
    fn new(
        env: Py<PyAny>,
        address: Option<&str>,
        options: Option<PyServeOptions>,
    ) -> PyResult<Self> {
        construct_server(env, address, options, build_scalar_server_env)
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
    fn serve(&self, py: Python<'_>) -> PyResult<()> {
        let resources = take_resources(&self.state, "serve", ServerState::RunningForeground)?;
        let address = self.address.clone();
        let shutdown = self.shutdown.clone();

        let result = py.detach(move || run_server(resources, address, shutdown, true));

        let mut state = self.state.lock().expect("server state mutex poisoned");
        *state = ServerState::Stopped;
        drop(state);

        result.map_err(py_runtime_err)
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
                .map_err(py_runtime_err),
            ShutdownAction::Join(handle) => py.detach(|| join_background_server(handle)),
        }
    }
}

async fn shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!(pid = std::process::id(), "shutdown requested via SIGINT");
                "sigint"
            }
            _ = sigterm.recv() => {
                tracing::info!(
                    pid = std::process::id(),
                    "shutdown requested via SIGTERM; stopping EnvServer"
                );
                "sigterm"
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl+C handler");
        tracing::info!(pid = std::process::id(), "shutdown requested via Ctrl+C");
        "ctrl_c"
    }
}

fn spawn_signal_shutdown(shutdown: ShutdownTrigger) {
    tokio::spawn(async move {
        let reason = shutdown_signal().await;
        shutdown.trigger(reason);
    });
}

impl Drop for PyEnvServer {
    fn drop(&mut self) {
        self.shutdown.trigger("drop");

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

        if let Some(timeout) = timeout {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Ok(false);
            }

            std::thread::sleep((timeout - elapsed).min(Duration::from_millis(10)));
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
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
    install_signal_handlers: bool,
) -> ServeResult {
    let ServerResources {
        env,
        runtime,
        listener,
        options,
    } = resources;

    runtime.block_on(async move {
        tracing::info!("EnvService serving on {}", address);
        if install_signal_handlers {
            spawn_signal_shutdown(shutdown.clone());
        }

        match env {
            PyServerEnv::Single(env) => {
                run_env_server(
                    WireEnvAdapter::new(ScalarEnvAdapter::new(env)),
                    listener,
                    options,
                    shutdown,
                )
                .await
            }
            PyServerEnv::Vector(env) => {
                run_env_server(WireEnvAdapter::new(env), listener, options, shutdown).await
            }
        }
    })
}

async fn run_env_server<E>(
    env: WireEnvAdapter<E>,
    listener: BoundListener,
    options: ServeOptions,
    shutdown: ShutdownTrigger,
) -> ServeResult
where
    WireEnvAdapter<E>: Environment + 'static,
{
    let activity_tx = start_idle_shutdown(options.idle_timeout, shutdown.clone());
    // Bound both teardown phases so the background serve thread always terminates
    // and EnvServer.shutdown()'s join (and interpreter exit) can never hang on a
    // lingering client connection or a blocking close hook.
    let drain_timeout = Some(options.drain_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE));
    let close_timeout = Some(options.close_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE));
    let env = Arc::new(Mutex::new(env));
    let grpc_options = rlmesh_grpc::ServeOptions::from(options);
    let service = env_service_from_shared(
        Arc::clone(&env),
        shutdown.clone(),
        grpc_options,
        activity_tx,
    );
    let serve_result = match listener {
        BoundListener::Tcp(listener) => await_server_shutdown(
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming_shutdown(
                    TcpListenerStream::new(listener),
                    shutdown.cancelled_owned(),
                ),
            shutdown.clone(),
            drain_timeout,
        )
        .await
        .map_err(|err| err.to_string()),
        #[cfg(unix)]
        BoundListener::Unix { listener, path } => {
            let result = await_server_shutdown(
                tonic::transport::Server::builder()
                    .add_service(service)
                    .serve_with_incoming_shutdown(
                        UnixListenerStream::new(listener),
                        shutdown.cancelled_owned(),
                    ),
                shutdown.clone(),
                drain_timeout,
            )
            .await;
            let _ = std::fs::remove_file(path);
            result.map_err(|err| err.to_string())
        }
    };

    let close_result = close_env(env, close_timeout).await;
    match (serve_result, close_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) => Err(err),
        (Ok(()), Err(err)) => Err(err),
        (Err(serve_err), Err(close_err)) => Err(format!(
            "environment server failed: {serve_err}; close hook failed: {close_err}"
        )),
    }
}

async fn close_env<E>(env: Arc<Mutex<E>>, close_timeout: Option<std::time::Duration>) -> ServeResult
where
    E: Environment,
{
    let close = async { env.lock().await.close().await.map(|_| ()) };
    await_close_with_timeout(close, close_timeout)
        .await
        .map_err(|timeout| format!("close timed out after {}ms", timeout.as_millis()))?
        .map_err(|err| err.to_string())
}

fn cleanup_ready_resources(resources: ServerResources) -> ServeResult {
    let ServerResources {
        mut env,
        runtime,
        listener,
        options,
    } = resources;

    match listener {
        BoundListener::Tcp(listener) => {
            drop(listener);
        }
        #[cfg(unix)]
        BoundListener::Unix { listener, path } => {
            drop(listener);
            let _ = std::fs::remove_file(path);
        }
    }

    let close_timeout = Some(options.close_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE));
    runtime.block_on(async move {
        await_close_with_timeout(env.close(), close_timeout)
            .await
            .map_err(|timeout| format!("close timed out after {}ms", timeout.as_millis()))?
            .map_err(|err| err.to_string())
    })
}

fn join_background_server(handle: JoinHandle<ServeResult>) -> PyResult<()> {
    match handle.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(py_runtime_err(format!("serve failed: {}", err))),
        Err(_) => Err(py_runtime_err("background server thread panicked")),
    }
}
