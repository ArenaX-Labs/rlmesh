//! Immutable run/telemetry handles and sized episode results.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_ptr, guard_value},
};
use crate::codec::RLMeshBytes;
use crate::ffi;
use crate::value::handle::write_out;
use rlmesh::{EpisodeResult, RunResult, TelemetryRow};
use std::ffi::c_char;
use std::sync::Arc;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEpisodeResult {
    pub struct_size: usize,
    pub index: i64,
    pub has_seed: bool,
    pub seed: i64,
    pub steps: i64,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub has_success: bool,
    pub success: bool,
    pub duration_s: f64,
    pub has_predict_ms: bool,
    pub predict_ms: f64,
    pub has_step_ms: bool,
    pub step_ms: f64,
    pub has_trial: bool,
    pub trial: i64,
}
impl RLMeshEpisodeResult {
    pub(crate) fn from_core(e: &EpisodeResult) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            index: e.index,
            seed: e.seed.unwrap_or_default(),
            steps: e.steps,
            reward: e.reward,
            duration_s: e.duration_s,
            predict_ms: e.predict_ms.unwrap_or_default(),
            step_ms: e.step_ms.unwrap_or_default(),
            trial: e.trial.unwrap_or_default(),
            has_seed: e.seed.is_some(),
            terminated: e.terminated,
            truncated: e.truncated,
            has_success: e.success.is_some(),
            success: e.success.unwrap_or_default(),
            has_predict_ms: e.predict_ms.is_some(),
            has_step_ms: e.step_ms.is_some(),
            has_trial: e.trial.is_some(),
        }
    }
    pub(crate) unsafe fn read(ptr: *const Self) -> Result<EpisodeResult, CapiError> {
        let e = unsafe { sizing::read(ptr, "RLMeshEpisodeResult") }?
            .ok_or_else(|| CapiError::invalid_arg("null episode result"))?;
        Ok(EpisodeResult {
            index: e.index,
            seed: e.has_seed.then_some(e.seed),
            steps: e.steps,
            reward: e.reward,
            duration_s: e.duration_s,
            predict_ms: e.has_predict_ms.then_some(e.predict_ms),
            step_ms: e.has_step_ms.then_some(e.step_ms),
            trial: e.has_trial.then_some(e.trial),
            terminated: e.terminated,
            truncated: e.truncated,
            success: e.has_success.then_some(e.success),
        })
    }
}
#[derive(Clone)]
pub struct RLMeshRunResult(pub(crate) Arc<RunResult>, Arc<Vec<std::ffi::CString>>);
impl RLMeshRunResult {
    pub(crate) fn new(result: RunResult) -> Self {
        let messages = result
            .advisories
            .iter()
            .map(|a| std::ffi::CString::new(a.message.as_str()).unwrap_or_default())
            .collect();
        Self(Arc::new(result), Arc::new(messages))
    }
}

pub struct RLMeshTelemetryRow(pub(crate) Arc<TelemetryRow>);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRunResultOptions {
    pub struct_size: usize,
    pub episodes: *const *const RLMeshEpisodeResult,
    pub episodes_len: usize,
    pub telemetry: *const *const RLMeshTelemetryRow,
    pub telemetry_len: usize,
    pub advisories: *const *const crate::adapters::RLMeshAdvisory,
    pub advisories_len: usize,
}
pub(crate) fn require<'a>(p: *const RLMeshRunResult) -> Result<&'a RunResult, CapiError> {
    unsafe { p.as_ref() }
        .map(|r| r.0.as_ref())
        .ok_or_else(|| CapiError::invalid_arg("null run result"))
}
pub(crate) fn output(out: *mut *mut RLMeshRunResult, result: RunResult) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    write_out(out, Box::into_raw(Box::new(RLMeshRunResult::new(result))))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_result_new(
    options: *const RLMeshRunResultOptions,
    out: *mut *mut RLMeshRunResult,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let mut result = RunResult::default();
        if let Some(o) = unsafe { sizing::read(options, "RLMeshRunResultOptions") }? {
            for &p in
                unsafe { crate::spaces::slice_arg(o.advisories, o.advisories_len, "advisories") }?
            {
                result
                    .advisories
                    .push(unsafe { crate::adapters::parse_advisory(p) }?);
            }
            for &p in unsafe { crate::spaces::slice_arg(o.episodes, o.episodes_len, "episodes") }? {
                result
                    .episodes
                    .push(unsafe { RLMeshEpisodeResult::read(p) }?);
            }
            for &p in
                unsafe { crate::spaces::slice_arg(o.telemetry, o.telemetry_len, "telemetry") }?
            {
                result.telemetry.push(
                    unsafe { p.as_ref() }
                        .ok_or_else(|| CapiError::invalid_arg("null telemetry row"))?
                        .0
                        .as_ref()
                        .clone(),
                );
            }
        }
        output(out, result)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_result_clone(
    p: *const RLMeshRunResult,
) -> *mut RLMeshRunResult {
    guard_ptr(|| {
        let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null result"))?;
        Ok(Box::into_raw(Box::new(p.clone())))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_result_free(p: *mut RLMeshRunResult) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
    });
}
macro_rules! scalar {
    ($name:ident,$ty:ty,$default:expr,$body:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(p: *const RLMeshRunResult) -> $ty {
            guard_value($default, || require(p).map_or($default, $body))
        }
    };
}
scalar!(rlmesh_run_result_episodes_len, usize, 0, |r| r
    .episodes
    .len());
scalar!(
    rlmesh_run_result_num_episodes,
    usize,
    0,
    RunResult::num_episodes
);
scalar!(
    rlmesh_run_result_total_steps,
    i64,
    0,
    RunResult::total_steps
);
scalar!(
    rlmesh_run_result_mean_reward,
    f64,
    0.0,
    RunResult::mean_reward
);
scalar!(rlmesh_run_result_telemetry_len, usize, 0, |r| r
    .telemetry
    .len());
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_run_result_success_rate(p: *const RLMeshRunResult, out: *mut f64) -> bool {
    guard_value(false, || {
        require(p)
            .ok()
            .and_then(RunResult::success_rate)
            .is_some_and(|v| write_out(out, v).is_ok())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_result_episodes_get(
    p: *const RLMeshRunResult,
    i: usize,
    out: *mut RLMeshEpisodeResult,
) -> RLMeshStatus {
    guard(|| {
        let caller = unsafe { sizing::read(out, "RLMeshEpisodeResult") }?
            .ok_or_else(|| CapiError::invalid_arg("null out"))?;
        let e = require(p)?
            .episodes
            .get(i)
            .ok_or_else(|| CapiError::invalid_arg("episode index out of range"))?;
        let mut filled = RLMeshEpisodeResult::from_core(e);
        filled.struct_size = caller.struct_size;
        write_out(out, filled)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_run_result_telemetry_get(
    p: *const RLMeshRunResult,
    i: usize,
    out: *mut *mut RLMeshTelemetryRow,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let r = require(p)?
            .telemetry
            .get(i)
            .ok_or_else(|| CapiError::invalid_arg("telemetry index out of range"))?
            .clone();
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshTelemetryRow(Arc::new(r)))),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_run_result_format_telemetry(
    p: *const RLMeshRunResult,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, require(p)?.format_telemetry()))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_run_result_repr(
    p: *const RLMeshRunResult,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, require(p)?.to_string()))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_run_result_equals(
    a: *const RLMeshRunResult,
    b: *const RLMeshRunResult,
) -> bool {
    guard_value(false, || match (require(a), require(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_episode_result_equals(
    a: *const RLMeshEpisodeResult,
    b: *const RLMeshEpisodeResult,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        write_out(
            out,
            unsafe { RLMeshEpisodeResult::read(a) }? == unsafe { RLMeshEpisodeResult::read(b) }?,
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_episode_result_repr(
    p: *const RLMeshEpisodeResult,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, unsafe { RLMeshEpisodeResult::read(p) }?.to_string()))
}
fn row<'a>(p: *const RLMeshTelemetryRow) -> Result<&'a TelemetryRow, CapiError> {
    unsafe { p.as_ref() }
        .map(|r| r.0.as_ref())
        .ok_or_else(|| CapiError::invalid_arg("null telemetry row"))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_telemetry_row_new(
    op: *const c_char,
    component: *const c_char,
    metric: *const c_char,
    unit: *const c_char,
    count: i64,
    avg: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    out: *mut *mut RLMeshTelemetryRow,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let text = |p| unsafe { ffi::text(p) }?.ok_or_else(|| CapiError::invalid_arg("null text"));
        let row = TelemetryRow {
            op: text(op)?,
            component: text(component)?,
            metric: text(metric)?,
            unit: text(unit)?,
            count,
            avg,
            p50,
            p95,
            p99,
        };
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshTelemetryRow(Arc::new(row)))),
        )
    })
}
macro_rules! row_text {
    ($name:ident,$field:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(
            p: *const RLMeshTelemetryRow,
            out: *mut RLMeshBytes,
        ) -> RLMeshStatus {
            guard(|| ffi::bytes(out, row(p)?.$field.clone()))
        }
    };
}
row_text!(rlmesh_telemetry_row_op, op);
row_text!(rlmesh_telemetry_row_component, component);
row_text!(rlmesh_telemetry_row_metric, metric);
row_text!(rlmesh_telemetry_row_unit, unit);
macro_rules! row_scalar {
    ($name:ident,$ty:ty,$field:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(p: *const RLMeshTelemetryRow) -> $ty {
            guard_value(Default::default(), || {
                row(p).map_or(Default::default(), |r| r.$field)
            })
        }
    };
}
row_scalar!(rlmesh_telemetry_row_count, i64, count);
row_scalar!(rlmesh_telemetry_row_avg, f64, avg);
row_scalar!(rlmesh_telemetry_row_p50, f64, p50);
row_scalar!(rlmesh_telemetry_row_p95, f64, p95);
row_scalar!(rlmesh_telemetry_row_p99, f64, p99);
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_telemetry_row_repr(
    p: *const RLMeshTelemetryRow,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, row(p)?.to_string()))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_telemetry_row_equals(
    a: *const RLMeshTelemetryRow,
    b: *const RLMeshTelemetryRow,
) -> bool {
    guard_value(false, || match (row(a), row(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_telemetry_row_free(p: *mut RLMeshTelemetryRow) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_result_advisories_len(p: *const RLMeshRunResult) -> usize {
    guard_value(0, || require(p).map_or(0, |r| r.advisories.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_result_advisories_get(
    p: *const RLMeshRunResult,
    i: usize,
    out: *mut crate::adapters::RLMeshAdvisory,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null run result"))?;
        let note =
            h.0.advisories
                .get(i)
                .ok_or_else(|| CapiError::invalid_arg("advisory index out of range"))?;
        let mut arena = crate::adapters::Arena::default();
        let mut view = crate::adapters::view_advisory(&mut arena, note);
        view.message = h.1[i].as_ptr();
        unsafe { crate::adapters::out_struct(out, view, "RLMeshAdvisory") }
    })
}
