//! Explicit representation contracts for every sized FFI input.
#![allow(unsafe_code)]
use super::SizedInput;

macro_rules! input {
    ($ty:ty; first_appended = $min:ident; appended = [$($appended:ident),*] $(; bools = [$($bool:ident),*])?) => {
        // SAFETY: as below; all-zero is valid for every field (pointers,
        // integers, floats, bools).
        unsafe impl SizedInput for $ty {
            const BOOL_FIELDS: &'static [(usize, &'static str)] = &[
                $($((std::mem::offset_of!($ty, $bool), stringify!($bool)),)*)?
            ];
            // The first appended field starts at the ABI-5 sizeof.
            const MIN_SIZE: usize = std::mem::offset_of!($ty, $min);
            const APPENDED_ENDS: &'static [usize] = &[
                $(std::mem::offset_of!($ty, $appended)
                    + crate::abi::sizing::inputs::field_size(|v: &$ty| &v.$appended),)*
            ];
        }
    };
    ($ty:ty $(, $field:ident)*) => {
        // SAFETY: listed bools are the only restricted scalar representations;
        // remaining fields are C integers, floats, raw pointers or nullable callbacks.
        unsafe impl SizedInput for $ty {
            const BOOL_FIELDS: &'static [(usize, &'static str)] = &[
                $((std::mem::offset_of!($ty, $field), stringify!($field)),)*
            ];
        }
    };
}
input!(crate::adapters::RLMeshAction, has_clip);
input!(
    crate::adapters::RLMeshActuator,
    has_encoding,
    has_range,
    binary,
    has_scale,
    invert,
    has_threshold,
    clip,
    optional,
    has_frame,
    has_reference,
    has_offset
);
input!(crate::adapters::RLMeshAdaptersText, has_fill);
input!(crate::adapters::RLMeshAdvisory);
input!(crate::adapters::RLMeshConcat, has_pad_to, has_clip);
input!(crate::adapters::RLMeshConstant);
input!(crate::adapters::RLMeshCustom);
input!(crate::adapters::RLMeshCustomEncoding);
input!(
    crate::adapters::RLMeshField,
    has_range,
    has_frame,
    has_provenance
);
input!(
    crate::adapters::RLMeshImage,
    has_height,
    has_width,
    has_channels,
    upside_down,
    allow_upscale,
    optional,
    has_fill,
    has_crop,
    has_crop_area,
    has_jpeg_quality,
    has_render,
    has_stride
);
input!(crate::adapters::RLMeshImageTag, upside_down);
input!(crate::adapters::RLMeshObservationRoles);
input!(crate::adapters::RLMeshPreviousOptions);
input!(
    crate::adapters::RLMeshResolveOptions,
    trust_entrypoints,
    check_inverse
);
input!(crate::adapters::RLMeshRotation);
input!(crate::adapters::RLMeshSplit);
input!(
    crate::adapters::RLMeshState,
    has_dim,
    has_index,
    optional,
    has_range,
    has_pad_to,
    has_scale,
    has_offset,
    has_frame,
    has_clip
);
input!(
    crate::adapters::RLMeshStateTag,
    has_range,
    has_frame,
    has_provenance
);
input!(crate::adapters::RLMeshTagOptions, validate);
input!(crate::adapters::RLMeshTextTag);
input!(crate::adapters::runtime::RLMeshAdapterVtable);
input!(crate::describe_api::RLMeshDescribeJsonOptions, has_kind);
input!(crate::describe_api::RLMeshDescribeOptions, has_kind);
input!(crate::env_api::RLMeshEnvAttributes);
input!(crate::env_api::RLMeshEnvVtable);
input!(crate::env_api::RLMeshVectorEnvAttributes);
input!(crate::env_api::RLMeshVectorEnvVtable);
input!(crate::env_factory::RLMeshEnvFactoryAttributes);
#[cfg(target_pointer_width = "64")]
input!(crate::env_factory::RLMeshEnvFactoryServeOptions; first_appended = _reserved; appended = [_reserved, ext]);
#[cfg(target_pointer_width = "32")]
input!(crate::env_factory::RLMeshEnvFactoryServeOptions; first_appended = _reserved; appended = [_reserved, ext, _reserved1]);
input!(crate::env_factory::RLMeshExtEnvFactoryServeOptions);
input!(crate::env_factory::RLMeshEnvFactoryVtable);
input!(crate::env_factory::RLMeshTagParam);
input!(
    crate::env_server::RLMeshEnvServerOptions,
    has_port,
    has_transport,
    close_env_on_shutdown
);
input!(
    crate::env_server::RLMeshExtEnvServerOptions,
    publish_describe
);
input!(
    crate::env_server::RLMeshServeOptions,
    allow_remote_shutdown,
    has_idle_timeout_seconds,
    has_drain_timeout_seconds,
    has_close_timeout_seconds,
    compress_responses
);
input!(
    crate::model_api::RLMeshExtModelServeOptions;
    first_appended = no_signal_handlers;
    appended = [no_signal_handlers, _reserved1];
    bools = [has_predict_concurrency, no_signal_handlers]
);
input!(crate::model_api::RLMeshModelOptions, trust_entrypoints);
input!(crate::model_api::RLMeshModelServeOptions);
input!(
    crate::model_api::RLMeshModelVtable,
    allow_fusion,
    has_native_chunk
);
input!(
    crate::recorder::RLMeshRecorderAddOptions,
    included_in_metrics
);
input!(
    crate::recorder::RLMeshRecorderCaptureOptions,
    included_in_metrics
);
input!(
    crate::recorder::RLMeshRecorderExportOptions,
    has_archive,
    archive
);
input!(crate::recorder::RLMeshRecorderOptions);
input!(crate::recorder::RLMeshRecorderToDictOptions);
input!(
    crate::recorder::records::RLMeshEpisodeRecordOptions,
    has_success,
    success,
    included_in_metrics
);
input!(
    crate::recorder::records::RLMeshMediaRefOptions,
    has_frame_count,
    has_width,
    has_height,
    has_fps
);
input!(crate::recorder::records::RLMeshWorkloadRecordOptions);
input!(
    crate::remote_api::RLMeshRemoteEnvOptions;
    first_appended = _reserved;
    appended = [_reserved, api_key, tls_ca_file];
    bools = [has_port, has_transport, has_connect_timeout_seconds, has_request_timeout_seconds]
);
input!(
    crate::remote_api::RLMeshRemoteModelOptions;
    first_appended = api_key;
    appended = [api_key, tls_ca_file];
    bools = [has_port, has_transport, has_connect_timeout_seconds, has_request_timeout_seconds]
);
input!(
    crate::run_api::RLMeshRunOptions,
    has_episodes,
    has_max_episode_steps,
    has_max_episode_seconds,
    close_env,
    has_trust_entrypoints,
    trust_entrypoints
);
input!(crate::run_hooks::RLMeshRunContextVtable);
input!(crate::run_hooks::RLMeshRunHooksVtable);
input!(
    crate::run_hooks::RLMeshStepEvent,
    has_seed,
    terminated,
    truncated
);
input!(
    crate::run_results::RLMeshEpisodeResult,
    has_seed,
    terminated,
    truncated,
    has_success,
    success,
    has_predict_ms,
    has_step_ms,
    has_trial
);
input!(crate::run_results::RLMeshRunResultOptions);
input!(
    crate::session_api::RLMeshSessionOptions,
    close_env,
    has_trust_entrypoints,
    trust_entrypoints
);
input!(
    crate::session_api::RLMeshSessionRunOptions,
    has_episodes,
    has_max_episode_steps,
    has_max_episode_seconds
);
input!(crate::space_handle::RLMeshBoxOptions);
input!(crate::space_handle::RLMeshDiscreteOptions);
input!(
    crate::value::dlpack::RLMeshToDlpackOptions,
    copy,
    has_dl_device
);
input!(crate::variant::RLMeshVariantOptions);
input!(crate::view::RLMeshView, has_step_hz);

/// The size of the field `get` projects, in a const context.
pub(crate) const fn field_size<T, F>(_get: fn(&T) -> &F) -> usize {
    std::mem::size_of::<F>()
}

#[cfg(test)]
mod tests {
    use super::SizedInput;
    use crate::remote_api::{RLMeshRemoteEnvOptions, RLMeshRemoteModelOptions};
    use std::ffi::c_char;

    /// The ABI-5 layouts, as an ABI-5 caller compiled them.
    #[repr(C)]
    #[allow(dead_code)]
    struct Abi5Env {
        struct_size: usize,
        address: *const c_char,
        host: *const c_char,
        has_port: bool,
        port: u16,
        path: *const c_char,
        has_transport: bool,
        transport: i32,
        has_connect_timeout_seconds: bool,
        connect_timeout_seconds: f64,
        has_request_timeout_seconds: bool,
        request_timeout_seconds: f64,
        workflow_edition: *const c_char,
    }
    #[repr(C)]
    #[allow(dead_code)]
    struct Abi5Model {
        struct_size: usize,
        address: *const c_char,
        host: *const c_char,
        has_port: bool,
        port: u16,
        path: *const c_char,
        has_transport: bool,
        transport: i32,
        has_connect_timeout_seconds: bool,
        connect_timeout_seconds: f64,
        has_request_timeout_seconds: bool,
        request_timeout_seconds: f64,
    }

    #[test]
    fn appended_remote_fields_start_past_the_abi5_sizeof_on_this_target() {
        assert_eq!(
            <RLMeshRemoteEnvOptions as SizedInput>::MIN_SIZE,
            std::mem::size_of::<Abi5Env>()
        );
        assert_eq!(
            <RLMeshRemoteModelOptions as SizedInput>::MIN_SIZE,
            std::mem::size_of::<Abi5Model>()
        );
        assert!(
            std::mem::offset_of!(RLMeshRemoteEnvOptions, api_key) >= std::mem::size_of::<Abi5Env>()
        );
    }
}
