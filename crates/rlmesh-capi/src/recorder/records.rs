//! Immutable, independently owned snapshots of facade recorder records.
#![allow(unsafe_code)]
use super::*;
use crate::abi::status::guard_value;
use crate::spaces::slice_arg;
use rlmesh::recorder::{EpisodeRecord, MediaRef, WorkloadRecord};

pub struct RLMeshMediaRef(pub(crate) Arc<MediaRef>);
pub struct RLMeshEpisodeRecord(pub(crate) Arc<EpisodeRecord>);
pub struct RLMeshWorkloadRecord(pub(crate) Arc<WorkloadRecord>);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshMediaRefOptions {
    pub struct_size: usize,
    pub has_frame_count: bool,
    pub frame_count: i64,
    pub has_width: bool,
    pub width: i64,
    pub has_height: bool,
    pub height: i64,
    pub has_fps: bool,
    pub fps: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEpisodeRecordOptions {
    pub struct_size: usize,
    pub has_success: bool,
    pub success: bool,
    pub duration_s: f64,
    pub included_in_metrics: bool,
    pub media: *const *const RLMeshMediaRef,
    pub media_len: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshWorkloadRecordOptions {
    pub struct_size: usize,
    pub config: *const RLMeshValue,
    pub episodes: *const *const RLMeshEpisodeRecord,
    pub episodes_len: usize,
}
macro_rules! record {
    ($ty:ident, $native:ident, $require:ident, $free:ident, $equals:ident, $repr:ident, $dict:ident) => {
        fn $require<'a>(p: *const $ty) -> Result<&'a $native, CapiError> {
            unsafe { p.as_ref() }
                .map(|h| &*h.0)
                .ok_or_else(|| CapiError::invalid_arg(concat!("null ", stringify!($ty))))
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $free(p: *mut $ty) {
            guard_value((), || {
                if !p.is_null() {
                    drop(unsafe { Box::from_raw(p) });
                }
            });
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn $equals(a: *const $ty, b: *const $ty) -> bool {
            guard_value(false, || match ($require(a), $require(b)) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            })
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn $repr(p: *const $ty, out: *mut RLMeshBytes) -> RLMeshStatus {
            guard(|| ffi::bytes(out, $require(p)?.to_string()))
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn $dict(p: *const $ty, out: *mut *mut RLMeshValue) -> RLMeshStatus {
            guard(|| {
                write_out(out, null_mut())?;
                write_out(out, into_handle(Value::Dict($require(p)?.to_dict())))
            })
        }
    };
}
record!(
    RLMeshMediaRef,
    MediaRef,
    media,
    rlmesh_media_ref_free,
    rlmesh_media_ref_equals,
    rlmesh_media_ref_repr,
    rlmesh_media_ref_to_dict
);
record!(
    RLMeshEpisodeRecord,
    EpisodeRecord,
    episode,
    rlmesh_episode_record_free,
    rlmesh_episode_record_equals,
    rlmesh_episode_record_repr,
    rlmesh_episode_record_to_dict
);
record!(
    RLMeshWorkloadRecord,
    WorkloadRecord,
    workload,
    rlmesh_workload_record_free,
    rlmesh_workload_record_equals,
    rlmesh_workload_record_repr,
    rlmesh_workload_record_to_dict
);

macro_rules! scalar {
    ($fn:ident, $ty:ident, $require:ident, $out:ty, $value:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn(p: *const $ty) -> $out {
            guard_value(<$out>::default(), || {
                $require(p).map_or(<$out>::default(), $value)
            })
        }
    };
}
macro_rules! optional {
    ($fn:ident, $ty:ident, $require:ident, $out:ty, $value:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn(p: *const $ty, out: *mut $out) -> bool {
            guard_value(false, || {
                $require(p)
                    .ok()
                    .and_then($value)
                    .is_some_and(|v| write_out(out, v).is_ok())
            })
        }
    };
}
macro_rules! string {
    ($fn:ident, $ty:ident, $require:ident, $field:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn(p: *const $ty, out: *mut RLMeshBytes) -> RLMeshStatus {
            guard(|| ffi::bytes(out, $require(p)?.$field.clone()))
        }
    };
}
string!(rlmesh_media_ref_camera, RLMeshMediaRef, media, camera);
string!(rlmesh_media_ref_kind, RLMeshMediaRef, media, kind);
string!(rlmesh_media_ref_path, RLMeshMediaRef, media, path);
string!(rlmesh_media_ref_format, RLMeshMediaRef, media, format);
optional!(
    rlmesh_media_ref_frame_count,
    RLMeshMediaRef,
    media,
    i64,
    |r| r.frame_count
);
optional!(rlmesh_media_ref_width, RLMeshMediaRef, media, i64, |r| r
    .width);
optional!(rlmesh_media_ref_height, RLMeshMediaRef, media, i64, |r| r
    .height);
optional!(rlmesh_media_ref_fps, RLMeshMediaRef, media, f64, |r| r.fps);
scalar!(
    rlmesh_episode_record_index,
    RLMeshEpisodeRecord,
    episode,
    i64,
    |r| r.index
);
optional!(
    rlmesh_episode_record_seed,
    RLMeshEpisodeRecord,
    episode,
    i64,
    |r| r.seed
);
scalar!(
    rlmesh_episode_record_steps,
    RLMeshEpisodeRecord,
    episode,
    i64,
    |r| r.steps
);
scalar!(
    rlmesh_episode_record_reward,
    RLMeshEpisodeRecord,
    episode,
    f64,
    |r| r.reward
);
scalar!(
    rlmesh_episode_record_terminated,
    RLMeshEpisodeRecord,
    episode,
    bool,
    |r| r.terminated
);
scalar!(
    rlmesh_episode_record_truncated,
    RLMeshEpisodeRecord,
    episode,
    bool,
    |r| r.truncated
);
optional!(
    rlmesh_episode_record_success,
    RLMeshEpisodeRecord,
    episode,
    bool,
    |r| r.success
);
scalar!(
    rlmesh_episode_record_duration_s,
    RLMeshEpisodeRecord,
    episode,
    f64,
    |r| r.duration_s
);
scalar!(
    rlmesh_episode_record_included_in_metrics,
    RLMeshEpisodeRecord,
    episode,
    bool,
    |r| r.included_in_metrics
);
scalar!(
    rlmesh_episode_record_media_len,
    RLMeshEpisodeRecord,
    episode,
    usize,
    |r| r.media.len()
);
string!(
    rlmesh_workload_record_model,
    RLMeshWorkloadRecord,
    workload,
    model
);
string!(
    rlmesh_workload_record_env,
    RLMeshWorkloadRecord,
    workload,
    env
);
string!(
    rlmesh_workload_record_task,
    RLMeshWorkloadRecord,
    workload,
    task
);
scalar!(
    rlmesh_workload_record_episodes_len,
    RLMeshWorkloadRecord,
    workload,
    usize,
    |r| r.episodes.len()
);
scalar!(
    rlmesh_workload_record_mean_reward,
    RLMeshWorkloadRecord,
    workload,
    f64,
    WorkloadRecord::mean_reward
);
optional!(
    rlmesh_workload_record_success_rate,
    RLMeshWorkloadRecord,
    workload,
    f64,
    WorkloadRecord::success_rate
);
scalar!(
    rlmesh_workload_record_total_steps,
    RLMeshWorkloadRecord,
    workload,
    i64,
    WorkloadRecord::total_steps
);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_media_ref_new(
    camera: *const c_char,
    kind: *const c_char,
    path: *const c_char,
    format: *const c_char,
    options: *const RLMeshMediaRefOptions,
    out: *mut *mut RLMeshMediaRef,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, null_mut())?;
        let mut r = MediaRef {
            camera: unsafe { text(camera, "camera") }?,
            kind: unsafe { text(kind, "kind") }?,
            path: unsafe { text(path, "path") }?,
            format: unsafe { text(format, "format") }?,
            ..Default::default()
        };
        if let Some(o) = unsafe { sizing::read(options, "RLMeshMediaRefOptions") }? {
            r.frame_count = o.has_frame_count.then_some(o.frame_count);
            r.width = o.has_width.then_some(o.width);
            r.height = o.has_height.then_some(o.height);
            r.fps = o.has_fps.then_some(o.fps);
        }
        write_out(out, Box::into_raw(Box::new(RLMeshMediaRef(Arc::new(r)))))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_episode_record_new(
    index: i64,
    seed: *const i64,
    steps: i64,
    reward: f64,
    terminated: bool,
    truncated: bool,
    options: *const RLMeshEpisodeRecordOptions,
    out: *mut *mut RLMeshEpisodeRecord,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, null_mut())?;
        let mut r = EpisodeRecord {
            index,
            seed: unsafe { seed.as_ref() }.copied(),
            steps,
            reward,
            terminated,
            truncated,
            ..Default::default()
        };
        if let Some(o) = unsafe { sizing::read(options, "RLMeshEpisodeRecordOptions") }? {
            r.success = o.has_success.then_some(o.success);
            r.duration_s = o.duration_s;
            r.included_in_metrics = o.included_in_metrics;
            r.media = unsafe { slice_arg(o.media, o.media_len, "media") }?
                .iter()
                .map(|&p| media(p).cloned())
                .collect::<Result<_, _>>()?;
        }
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshEpisodeRecord(Arc::new(r)))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_workload_record_new(
    model: *const c_char,
    env: *const c_char,
    task: *const c_char,
    options: *const RLMeshWorkloadRecordOptions,
    out: *mut *mut RLMeshWorkloadRecord,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, null_mut())?;
        let mut r = WorkloadRecord {
            model: unsafe { text(model, "model") }?,
            env: unsafe { text(env, "env") }?,
            task: unsafe { text(task, "task") }?,
            ..Default::default()
        };
        if let Some(o) = unsafe { sizing::read(options, "RLMeshWorkloadRecordOptions") }? {
            r.config = ffi::dict(o.config)?.unwrap_or_default();
            r.episodes = unsafe { slice_arg(o.episodes, o.episodes_len, "episodes") }?
                .iter()
                .map(|&p| episode(p).cloned())
                .collect::<Result<_, _>>()?;
        }
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshWorkloadRecord(Arc::new(r)))),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_episode_record_media_get(
    p: *const RLMeshEpisodeRecord,
    i: usize,
    out: *mut *mut RLMeshMediaRef,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, null_mut())?;
        let r = episode(p)?
            .media
            .get(i)
            .cloned()
            .ok_or_else(|| CapiError::invalid_arg("media index out of range"))?;
        write_out(out, Box::into_raw(Box::new(RLMeshMediaRef(Arc::new(r)))))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_workload_record_episodes_get(
    p: *const RLMeshWorkloadRecord,
    i: usize,
    out: *mut *mut RLMeshEpisodeRecord,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, null_mut())?;
        let r = workload(p)?
            .episodes
            .get(i)
            .cloned()
            .ok_or_else(|| CapiError::invalid_arg("episode index out of range"))?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshEpisodeRecord(Arc::new(r)))),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_workload_record_config(
    p: *const RLMeshWorkloadRecord,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, null_mut())?;
        write_out(out, into_handle(Value::Dict(workload(p)?.config.clone())))
    })
}
