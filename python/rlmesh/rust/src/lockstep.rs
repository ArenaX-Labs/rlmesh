//! PyO3 bindings for the lockstep clock (protocol v2 §12): a v2 robot that
//! serves the lockstep exchange from Python callbacks
//! (`rlmesh.v2.Robot(..., clock="lockstep")` wraps it). Gym envs, models and
//! `RemoteEnv` (a Gym env, or any robot through its Gym view) reach lockstep
//! through their own servers, runs and clients (participant addresses).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh::lockstep::view::{Frame, commands};
use rlmesh::lockstep::{GymError, LockstepHandler, Request};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::wire::{FieldSpec, LockstepReply, PartSpec, ParticipantId};
use tokio_util::sync::CancellationToken;

use crate::session::{SessionException, block_on, part_dict, runtime, timeout, value_to_py};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLockstepRobot>()?;
    Ok(())
}

pub(crate) fn gym_err(err: GymError) -> PyErr {
    SessionException::new_err(err.to_string())
}

/// One part's fields as a Python value: the array of a single-field part,
/// else a dict of arrays.
fn fields_value(fields: &[FieldSpec], values: &[Bytes]) -> Value {
    let tensor = |f: &FieldSpec, b: &Bytes| Tensor {
        dtype: f.dtype,
        shape: f.shape.clone(),
        data: b.to_vec(),
    };
    match (fields, values) {
        ([field], [bytes]) => Value::Tensor(tensor(field, bytes)),
        _ => Value::Dict(
            fields
                .iter()
                .zip(values)
                .map(|(f, b)| (f.name.clone(), tensor(f, b)))
                .collect(),
        ),
    }
}

/// The robot's Python callbacks, and the frame they publish into.
struct Callbacks {
    step: Py<PyAny>,
    reset: Py<PyAny>,
    parts: Vec<PartSpec>,
    frame: Arc<Mutex<Option<Frame>>>,
}

impl Callbacks {
    /// Runs one callback with a fresh frame, then takes the frame back.
    fn run(
        &self,
        call: impl FnOnce(Python<'_>) -> PyResult<(f64, bool, bool, String)>,
    ) -> Result<Frame, GymError> {
        *self
            .frame
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(Frame::new(self.parts.clone()));
        let outcome = Python::attach(call);
        let mut frame = self
            .frame
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .ok_or_else(|| GymError::Map("the frame vanished".into()))?;
        let (reward, terminated, truncated, info_json) =
            outcome.map_err(|err| GymError::Map(format!("the robot's callback raised: {err}")))?;
        frame.reward = reward;
        frame.terminated = terminated;
        frame.truncated = truncated;
        frame.info_json = info_json;
        Ok(frame)
    }
}

struct PyHandler(Arc<Callbacks>);

#[async_trait]
impl LockstepHandler for PyHandler {
    async fn handle(&self, request: Request) -> Result<LockstepReply, GymError> {
        let callbacks = self.0.clone();
        tokio::task::spawn_blocking(move || match request {
            Request::Reset(reset) => {
                let seed = reset.seeds.first().copied();
                let frame = callbacks.run(|py| {
                    callbacks
                        .reset
                        .call1(py, (seed, reset.options_json.as_str()))?
                        .extract(py)
                })?;
                Ok(frame.reply(reset.epoch, 0))
            }
            Request::Step(step) => {
                let commanded = commands(&callbacks.parts, &step)?;
                let frame = callbacks.run(|py| {
                    let d = PyDict::new(py);
                    for (path, fields) in &commanded {
                        let part = callbacks
                            .parts
                            .iter()
                            .find(|p| p.path.as_str() == path)
                            .and_then(|p| p.command.as_ref());
                        if let Some(command) = part {
                            d.set_item(
                                path,
                                value_to_py(py, fields_value(&command.fields, fields))?,
                            )?;
                        }
                    }
                    callbacks.step.call1(py, (d,))?.extract(py)
                })?;
                Ok(frame.reply(step.epoch, step.step))
            }
        })
        .await
        .map_err(|e| GymError::Closed(e.to_string()))?
    }
}

/// A v2 robot that serves the lockstep exchange (§12): every step is one
/// call of its Python `step(commands)`, which publishes the robot's parts.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2LockstepRobot", frozen)]
pub struct PyLockstepRobot {
    name: String,
    address: String,
    robot_json: String,
    parts: Vec<PartSpec>,
    binding: Mutex<Option<rlmesh::lockstep::LockstepBinding>>,
    frame: Arc<Mutex<Option<Frame>>>,
    stop: CancellationToken,
    served: Mutex<Option<tokio::task::JoinHandle<Result<(), GymError>>>>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyLockstepRobot {
    /// Binds the robot `name` in the session at `address`
    /// (`{ip}:{port}/{tenant}/{session}`) with the map `robot_json`; it
    /// declares and arms when it starts serving.
    #[new]
    fn new(py: Python<'_>, address: &str, name: &str, robot_json: String) -> PyResult<Self> {
        let robot = ParticipantId::new(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let robot_json = rlmesh::lockstep::lockstep_robot_map(&robot_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let parts = rlmesh_session::spec::normalise(
            &robot,
            rlmesh_session::wire::version::SPEC_VERSION,
            &robot_json,
        )
        .map_err(|e| PyValueError::new_err(e.message))?;
        // The session address with the robot: its participant address.
        let address = match address {
            "" => name.to_string(),
            address => format!("{address}/{name}"),
        };
        let binding = block_on(py, async move {
            rlmesh::lockstep::bind(&address).await.map_err(gym_err)
        })?;
        Ok(Self {
            name: name.to_string(),
            address: binding.address(),
            robot_json,
            parts,
            binding: Mutex::new(Some(binding)),
            frame: Arc::default(),
            stop: CancellationToken::new(),
            served: Mutex::new(None),
        })
    }

    /// The robot's participant id.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The robot's participant address, `{ip}:{port}[/{session}]/{name}`.
    #[getter]
    fn address(&self) -> &str {
        &self.address
    }

    /// The parts as normalised, in authored order.
    #[gen_stub(override_return_type(
        type_repr = "builtins.list[builtins.dict[builtins.str, typing.Any]]",
        imports = ("builtins", "typing")
    ))]
    fn parts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for part in &self.parts {
            list.append(part_dict(py, part)?)?;
        }
        Ok(list)
    }

    /// Declares, arms and serves: `step(commands)` and `reset(seed,
    /// options_json)` each return `(reward, terminated, truncated,
    /// info_json)`. With `background`, returns once serving.
    #[pyo3(signature = (step, reset, background=false, timeout_s=10.0))]
    fn serve(
        &self,
        py: Python<'_>,
        step: Py<PyAny>,
        reset: Py<PyAny>,
        background: bool,
        timeout_s: f64,
    ) -> PyResult<()> {
        let binding = self
            .binding
            .lock()
            .ok()
            .and_then(|mut b| b.take())
            .ok_or_else(|| SessionException::new_err("the robot already served"))?;
        let robot_json = self.robot_json.clone();
        let frame = self.frame.clone();
        let stop = self.stop.clone();
        let (declared_tx, declared_rx) = tokio::sync::oneshot::channel();
        let options = rlmesh::lockstep::ServeOptions::default();
        let task = runtime().spawn(async move {
            binding
                .serve_robot(&robot_json, options, stop, move |declared| {
                    let _ = declared_tx.send(());
                    Ok(Arc::new(PyHandler(Arc::new(Callbacks {
                        step,
                        reset,
                        parts: declared.parts,
                        frame,
                    }))) as Arc<dyn LockstepHandler>)
                })
                .await
        });
        let wait = timeout(timeout_s)?;
        if background {
            let task = block_on(py, async move {
                match tokio::time::timeout(wait, declared_rx).await {
                    Ok(Ok(())) => Ok(task),
                    Err(_) => Err(SessionException::new_err(
                        "the robot did not declare in time",
                    )),
                    // The serve task ended before serving: its own error says why.
                    Ok(Err(_)) => Err(match task.await {
                        Ok(Err(err)) => gym_err(err),
                        Ok(Ok(())) => SessionException::new_err("the robot stopped before serving"),
                        Err(join) => SessionException::new_err(join.to_string()),
                    }),
                }
            })?;
            *self
                .served
                .lock()
                .map_err(|_| SessionException::new_err("poisoned"))? = Some(task);
            return Ok(());
        }
        let stop = self.stop.clone();
        let outcome = block_on(py, async move {
            let guard = StopOnDrop(stop);
            let outcome = task.await;
            std::mem::forget(guard);
            Ok(outcome)
        });
        self.stop.cancel();
        match outcome? {
            Ok(result) => result.map_err(gym_err),
            Err(join) => Err(SessionException::new_err(join.to_string())),
        }
    }

    /// Inside a callback: publishes `path`'s fields (bytes-likes in field
    /// order, or a `str` for text) into this step's reply.
    fn publish(&self, py: Python<'_>, path: &str, fields: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let mut owned: Vec<Vec<u8>> = Vec::with_capacity(fields.len());
        for field in &fields {
            if let Ok(text) = field.extract::<String>() {
                owned.push(text.into_bytes());
            } else {
                let buffer = pyo3::buffer::PyBuffer::<u8>::get(field).map_err(|_| {
                    PyValueError::new_err(format!(
                        "{path}: a field must be a C-contiguous bytes-like or a str"
                    ))
                })?;
                owned.push(buffer.to_vec(py)?);
            }
        }
        let mut frame = self
            .frame
            .lock()
            .map_err(|_| SessionException::new_err("poisoned"))?;
        let frame = frame.as_mut().ok_or_else(|| {
            SessionException::new_err(
                "a lockstep robot publishes inside its step or reset callback",
            )
        })?;
        let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        frame.publish(path, &refs).map_err(gym_err)
    }

    /// Stops serving and detaches.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.stop.cancel();
        let task = self.served.lock().ok().and_then(|mut t| t.take());
        if let Some(task) = task {
            block_on(py, async move {
                let _ = task.await;
                Ok(())
            })?;
        }
        Ok(())
    }
}

struct StopOnDrop(CancellationToken);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
