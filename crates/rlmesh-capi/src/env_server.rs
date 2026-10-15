//! Facade-backed environment servers and sized serve options.
#![allow(unsafe_code)] // C strings, sized options and borrowed handles.
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_value},
};
use crate::codec::RLMeshBytes;
use crate::contract::RLMeshEnvContract;
use crate::env_api::{BorrowedEnv, BorrowedVectorEnv, Claim, RLMeshEnv, RLMeshVectorEnv};
use crate::ffi;
use crate::value::handle::{RLMeshValue, write_out};
use rlmesh::{EnvServer, EnvServerOptions, ServeOptions, Transport};
use std::ffi::c_char;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

// The adapter wave supplies authoring constructors for this opaque type. Env
// APIs copy and return it already, without routing tags through JSON.
#[derive(Clone)]
pub struct RLMeshEnvTags(pub(crate) rlmesh::adapters::EnvTags);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_tags_free(tags: *mut RLMeshEnvTags) {
    guard_value((), || {
        if !tags.is_null() {
            drop(unsafe { Box::from_raw(tags) });
        }
    });
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshServeOptions {
    pub struct_size: usize,
    pub allow_remote_shutdown: bool,
    pub has_idle_timeout_seconds: bool,
    pub has_drain_timeout_seconds: bool,
    pub has_close_timeout_seconds: bool,
    pub compress_responses: bool,
    pub _reserved: [u8; 3],
    pub idle_timeout_seconds: f64,
    pub drain_timeout_seconds: f64,
    pub close_timeout_seconds: f64,
    pub workflow_edition: *const c_char,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshExtEnvServerOptions {
    pub struct_size: usize,
    pub token: *const c_char,
    pub peer_info_extra: *const RLMeshValue,
    pub host_runtime_json: *const c_char,
    pub publish_describe: bool,
    pub _reserved: [u8; std::mem::size_of::<usize>() - 1],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEnvServerOptions {
    pub struct_size: usize,
    pub address: *const c_char,
    pub host: *const c_char,
    pub path: *const c_char,
    pub options: *const RLMeshServeOptions,
    pub tags: *const RLMeshEnvTags,
    pub ext: *const RLMeshExtEnvServerOptions,
    pub port: u16,
    pub has_port: bool,
    pub has_transport: bool,
    pub transport: i32,
    pub close_env_on_shutdown: bool,
    pub _reserved: [u8; 7],
}
pub(crate) unsafe fn serve_options(
    ptr: *const RLMeshServeOptions,
) -> Result<Option<ServeOptions>, CapiError> {
    let Some(o) = unsafe { sizing::read(ptr, "RLMeshServeOptions") }? else {
        return Ok(None);
    };
    let duration = |has, v, name| {
        if has {
            ffi::duration(v, name).map(Some)
        } else {
            Ok(None)
        }
    };
    Ok(Some(ServeOptions {
        allow_remote_shutdown: o.allow_remote_shutdown,
        idle_timeout: duration(
            o.has_idle_timeout_seconds,
            o.idle_timeout_seconds,
            "idle_timeout_seconds",
        )?,
        drain_timeout: duration(
            o.has_drain_timeout_seconds,
            o.drain_timeout_seconds,
            "drain_timeout_seconds",
        )?,
        close_timeout: duration(
            o.has_close_timeout_seconds,
            o.close_timeout_seconds,
            "close_timeout_seconds",
        )?,
        workflow_edition: unsafe { ffi::text(o.workflow_edition) }?,
        compress_responses: o.compress_responses,
    }))
}
unsafe fn options(
    ptr: *const RLMeshEnvServerOptions,
) -> Result<(EnvServerOptions, serde_json::Map<String, serde_json::Value>), CapiError> {
    let Some(o) = unsafe { sizing::read(ptr, "RLMeshEnvServerOptions") }? else {
        return Ok((EnvServerOptions::default(), serde_json::Map::new()));
    };
    let transport = if o.has_transport {
        Some(match o.transport {
            1 => Transport::Tcp,
            2 => Transport::Unix,
            _ => return Err(CapiError::invalid_arg("invalid transport")),
        })
    } else {
        None
    };
    let ext = unsafe { sizing::read(o.ext, "RLMeshExtEnvServerOptions") }?;
    let ext = ext
        .map(|e| -> Result<_, CapiError> {
            let peer_info_extra = ffi::dict(e.peer_info_extra)?
                .unwrap_or_default()
                .into_iter()
                .map(|(key, v)| {
                    let value = v
                        .as_text()
                        .map_err(|_| CapiError::invalid_arg("peer_info_extra values must be TEXT"))?
                        .to_string();
                    Ok((key, value))
                })
                .collect::<Result<_, CapiError>>()?;
            // The publication's host declaration is applied by the C binding
            // after the facade has validated and constructed the server.
            Ok((
                rlmesh::ext::EnvServerOptions {
                    token: unsafe { ffi::text(e.token) }?,
                    peer_info_extra,
                    publish_describe: e.publish_describe,
                },
                unsafe { ffi::text(e.host_runtime_json) }?,
            ))
        })
        .transpose()?;
    let host = ext
        .as_ref()
        .and_then(|(_, host)| host.as_deref())
        .map(crate::describe::host_runtime)
        .transpose()?
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    Ok((
        EnvServerOptions {
            address: unsafe { ffi::text(o.address) }?,
            host: unsafe { ffi::text(o.host) }?,
            path: unsafe { ffi::text(o.path) }?,
            port: o.has_port.then_some(o.port),
            transport,
            options: unsafe { serve_options(o.options) }?,
            tags: unsafe { o.tags.as_ref() }.map(|t| t.0.clone()),
            close_env_on_shutdown: o.close_env_on_shutdown,
            ext: ext.map(|(e, _)| e),
        },
        host,
    ))
}
pub struct RLMeshEnvServer {
    server: EnvServer,
    contract: RLMeshEnvContract,
    stopped: Arc<AtomicBool>,
    lanes: Vec<BorrowedEnv>,
    vector: Option<BorrowedVectorEnv>,
    _claims: Vec<Claim>,
}
impl Drop for RLMeshEnvServer {
    fn drop(&mut self) {
        let _ = self.server.shutdown();
        self.stopped.store(true, Ordering::Release);
        for lane in &self.lanes {
            lane.drain();
        }
        if let Some(v) = &self.vector {
            v.drain();
        }
    }
}
fn require<'a>(ptr: *const RLMeshEnvServer) -> Result<&'a RLMeshEnvServer, CapiError> {
    unsafe { ptr.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null server"))
}
fn output(
    server: EnvServer,
    stopped: Arc<AtomicBool>,
    lanes: Vec<BorrowedEnv>,
    vector: Option<BorrowedVectorEnv>,
    claims: Vec<Claim>,
    out: *mut *mut RLMeshEnvServer,
) -> Result<(), CapiError> {
    let contract = RLMeshEnvContract::new(rlmesh::binding::env_contract_to_wire(
        server.env_contract(),
    )?);
    write_out(
        out,
        Box::into_raw(Box::new(RLMeshEnvServer {
            server,
            contract,
            stopped,
            lanes,
            vector,
            _claims: claims,
        })),
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_server_new(
    envs: *const *mut RLMeshEnv,
    envs_len: usize,
    opts: *const RLMeshEnvServerOptions,
    out: *mut *mut RLMeshEnvServer,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let (opts, host) = unsafe { options(opts) }?;
        let envs = unsafe { crate::spaces::slice_arg(envs, envs_len, "envs") }?;
        if envs.is_empty() {
            return Err(CapiError::invalid_arg("envs must not be empty"));
        }
        let stopped = Arc::new(AtomicBool::new(false));
        let mut claims = Vec::new();
        let mut lanes = Vec::new();
        for &env in envs {
            let e =
                unsafe { env.as_ref() }.ok_or_else(|| CapiError::type_error("null env lane"))?;
            claims.push(Claim::new(&e.leased)?);
            lanes.push(BorrowedEnv {
                inner: Arc::clone(&e.inner),
                stopped: Arc::clone(&stopped),
            });
        }
        let server = EnvServer::lanes(lanes.clone(), opts)?;
        rlmesh::binding::env_server_binding_host(&server, "rlmesh-capi", "c", host)?;
        output(server, stopped, lanes, None, claims, out)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_server_new_vector(
    env: *mut RLMeshVectorEnv,
    opts: *const RLMeshEnvServerOptions,
    out: *mut *mut RLMeshEnvServer,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let (opts, host) = unsafe { options(opts) }?;
        let e = unsafe { env.as_ref() }.ok_or_else(|| CapiError::type_error("null vector env"))?;
        let claims = vec![Claim::new(&e.leased)?];
        let stopped = Arc::new(AtomicBool::new(false));
        let v = BorrowedVectorEnv {
            inner: Arc::clone(&e.inner),
            stopped: Arc::clone(&stopped),
        };
        let server = EnvServer::vector(v.clone(), opts)?;
        rlmesh::binding::env_server_binding_host(&server, "rlmesh-capi", "c", host)?;
        output(server, stopped, Vec::new(), Some(v), claims, out)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_server_address(
    server: *const RLMeshEnvServer,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, require(server)?.server.address().to_string()))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_server_env_contract(
    server: *const RLMeshEnvServer,
) -> *const RLMeshEnvContract {
    guard_value(std::ptr::null(), || {
        require(server).map_or(std::ptr::null(), |s| std::ptr::from_ref(&s.contract))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_server_repr(
    server: *const RLMeshEnvServer,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, require(server)?.server.to_string()))
}
macro_rules! lifecycle {
    ($name:ident,$method:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(server: *mut RLMeshEnvServer) -> RLMeshStatus {
            guard(|| {
                require(server)?.server.$method()?;
                Ok(())
            })
        }
    };
}
lifecycle!(rlmesh_env_server_start, start);
lifecycle!(rlmesh_env_server_serve, serve);
lifecycle!(rlmesh_env_server_shutdown, shutdown);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_server_wait(
    server: *mut RLMeshEnvServer,
    timeout: *const f64,
    out_stopped: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        if out_stopped.is_null() {
            return Err(CapiError::invalid_arg("null out_stopped"));
        }
        let timeout = unsafe { timeout.as_ref() }
            .map(|&v| ffi::duration(v, "timeout"))
            .transpose()?;
        write_out(out_stopped, require(server)?.server.wait(timeout)?)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_server_free(server: *mut RLMeshEnvServer) {
    guard_value((), || {
        if !server.is_null() {
            drop(unsafe { Box::from_raw(server) });
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_transport_str(value: i32) -> *const c_char {
    match value {
        1 => c"tcp".as_ptr(),
        2 => c"unix".as_ptr(),
        _ => std::ptr::null(),
    }
}
