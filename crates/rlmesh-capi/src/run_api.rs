//! Python-named run options and frozen by-value target unions.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard},
};
use crate::cancel::RLMeshExtCancelToken;
use crate::env_api::{BorrowedEnv, BorrowedVectorEnv, Claim, RLMeshEnv, RLMeshVectorEnv};
use crate::env_factory::{RLMeshEnvFactory, borrow_factory};
use crate::env_server::RLMeshEnvServer;
use crate::ffi;
use crate::model_api::{RLMeshModel, require};
use crate::run_hooks::{self, RLMeshRunHooks};
use crate::run_results::{self, RLMeshRunResult};
use crate::value::handle::write_out;
use std::ffi::{c_char, c_void};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshModelTarget {
    pub kind: i32,
    pub handle: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEnvTarget {
    pub kind: i32,
    pub handle: *const c_void,
    pub address: *const c_char,
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_target(p: *mut RLMeshModel) -> RLMeshModelTarget {
    RLMeshModelTarget {
        kind: 1,
        handle: p.cast(),
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_address_target(p: *const c_char) -> RLMeshEnvTarget {
    RLMeshEnvTarget {
        kind: 1,
        handle: std::ptr::null(),
        address: p,
    }
}
macro_rules! target {
    ($f:ident,$ty:ty,$kind:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $f(p: *const $ty) -> RLMeshEnvTarget {
            RLMeshEnvTarget {
                kind: $kind,
                handle: p.cast(),
                address: std::ptr::null(),
            }
        }
    };
}
target!(rlmesh_env_server_target, RLMeshEnvServer, 2);
target!(rlmesh_env_target, RLMeshEnv, 5);
target!(rlmesh_vector_env_target, RLMeshVectorEnv, 6);
target!(rlmesh_env_factory_target, RLMeshEnvFactory, 7);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRunOptions {
    pub struct_size: usize,
    pub seeds: *const i64,
    pub seeds_len: usize,
    pub hooks: *mut RLMeshRunHooks,
    pub workflow_edition: *const c_char,
    pub cancel: *const RLMeshExtCancelToken,
    pub episodes: u64,
    pub max_episode_steps: u64,
    pub max_episode_seconds: f64,
    pub prefetch_lead: u64,
    pub trial_index_base: u64,
    pub execution_horizon: u32,
    pub has_episodes: bool,
    pub has_max_episode_steps: bool,
    pub has_max_episode_seconds: bool,
    pub close_env: bool,
    pub has_trust_entrypoints: bool,
    pub trust_entrypoints: bool,
    pub _reserved: [u8; 2],
}
pub(crate) fn seconds(v: f64) -> Result<Option<Duration>, CapiError> {
    if v.is_nan() || v <= 0.0 {
        return Err(CapiError::invalid_arg("max_episode_seconds must be > 0"));
    }
    if v == f64::INFINITY {
        Ok(None)
    } else {
        Ok(Some(
            Duration::try_from_secs_f64(v).unwrap_or(Duration::MAX),
        ))
    }
}
pub(crate) unsafe fn options(
    p: *const RLMeshRunOptions,
) -> Result<rlmesh::RunOptions<'static>, CapiError> {
    let mut r = rlmesh::RunOptions::default();
    if let Some(o) = unsafe { sizing::read(p, "RLMeshRunOptions") }? {
        if o._reserved != [0; 2] {
            return Err(CapiError::type_error(
                "reserved run option bytes must be zero",
            ));
        }
        r.episodes = o.has_episodes.then_some(o.episodes);
        r.seeds = if o.seeds.is_null() {
            if o.seeds_len != 0 {
                return Err(CapiError::invalid_arg("null seeds with nonzero length"));
            }
            None
        } else {
            Some(unsafe { crate::spaces::slice_arg(o.seeds, o.seeds_len, "seeds") }?.to_vec())
        };
        r.max_episode_steps = o.has_max_episode_steps.then_some(o.max_episode_steps);
        r.max_episode = if o.has_max_episode_seconds {
            seconds(o.max_episode_seconds)?
        } else {
            None
        };
        r.close_env = o.close_env;
        r.trust_entrypoints = o.has_trust_entrypoints.then_some(o.trust_entrypoints);
        r.execution_horizon = o.execution_horizon;
        r.prefetch_lead = o.prefetch_lead;
        r.workflow_edition = unsafe { ffi::text(o.workflow_edition) }?;
        r.trial_index_base = o.trial_index_base;
        r.cancel = unsafe { o.cancel.as_ref() }.map(|c| c.0.clone());
    }
    Ok(r)
}
pub(crate) unsafe fn model<'a>(t: RLMeshModelTarget) -> Result<rlmesh::ModelTarget<'a>, CapiError> {
    match t.kind {
        1 => Ok((&require(t.handle.cast())?.model).into()),
        3 => Ok(rlmesh::RANDOM_SAMPLE.into()),
        _ => Err(CapiError::invalid_arg(
            "invalid or unsupported model target",
        )),
    }
}
pub(crate) unsafe fn with_env<R>(
    t: RLMeshEnvTarget,
    f: impl FnOnce(rlmesh::EnvTarget<'_>) -> Result<R, CapiError>,
) -> Result<R, CapiError> {
    match t.kind {
        1 => f(unsafe { ffi::text(t.address) }?
            .ok_or_else(|| CapiError::invalid_arg("null address"))?
            .into()),
        2 => f(unsafe { t.handle.cast::<RLMeshEnvServer>().as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null server"))?
            .facade()
            .into()),
        5 => {
            let h = unsafe { t.handle.cast::<RLMeshEnv>().as_ref() }
                .ok_or_else(|| CapiError::invalid_arg("null env"))?;
            let _claim = Claim::new(&h.leased)?;
            let mut env = BorrowedEnv {
                inner: Arc::clone(&h.inner),
                stopped: Arc::new(AtomicBool::new(false)),
            };
            let result = f((&mut env).into());
            env.stopped.store(true, Ordering::Release);
            env.finish(false);
            result
        }
        6 => {
            let h = unsafe { t.handle.cast::<RLMeshVectorEnv>().as_ref() }
                .ok_or_else(|| CapiError::invalid_arg("null vector env"))?;
            let _claim = Claim::new(&h.leased)?;
            let mut env = BorrowedVectorEnv {
                inner: Arc::clone(&h.inner),
                stopped: Arc::new(AtomicBool::new(false)),
            };
            let result = f(rlmesh::EnvTarget::vector(&mut env));
            env.stopped.store(true, Ordering::Release);
            env.finish(false);
            result
        }
        7 => {
            let mut factory = borrow_factory(t.handle.cast())?;
            let _scope = factory.scope();
            f(factory.target())
        }
        _ => Err(CapiError::invalid_arg("invalid or unsupported env target")),
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run(
    model_target: RLMeshModelTarget,
    env: RLMeshEnvTarget,
    p: *const RLMeshRunOptions,
    out: *mut *mut RLMeshRunResult,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let opts = unsafe { options(p) }?;
        let raw = unsafe { sizing::read(p, "RLMeshRunOptions") }?;
        let mut hooks =
            unsafe { run_hooks::borrow(raw.map_or(std::ptr::null_mut(), |o| o.hooks)) }?;
        let options = rlmesh::RunOptions {
            hooks: hooks.as_deref_mut().map(|h| h as &mut dyn rlmesh::RunHooks),
            ..opts
        };
        let m = unsafe { model(model_target) }?;
        let result = unsafe { with_env(env, |e| Ok(rlmesh::run(m, e, options)?)) }?;
        run_results::output(out, result)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_run(
    model: *mut RLMeshModel,
    env: RLMeshEnvTarget,
    p: *const RLMeshRunOptions,
    out: *mut *mut RLMeshRunResult,
) -> RLMeshStatus {
    unsafe { rlmesh_run(rlmesh_model_target(model), env, p, out) }
}
