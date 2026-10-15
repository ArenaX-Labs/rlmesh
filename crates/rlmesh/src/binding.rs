//! Plumbing for the in-repo bindings (the Python extension and the C ABI).
//!
//! Hidden from the docs and exempt from semver: nothing here is part of the
//! shared surface, and it changes with the bindings.

/// Legacy DLPack capsules are Python protocol plumbing, not a native API.
pub mod dlpack {
    pub use crate::dlpack::{DLManagedTensor, export_legacy, import_legacy};
}

pub mod env;
pub mod model;
pub mod spaces;

/// The async-trait macro the binding seams (`ModelHandler`, the legacy env
/// traits) are written with.
pub use async_trait::async_trait;

pub use crate::editions::{
    EditionResolution, EditionSources, resolve_rungs as resolve_workflow_edition_rungs,
    validate_workflow_edition,
};
/// The build's wire identity and the host-supplied handshake `PeerInfo`
/// override, for a non-Rust host (the C ABI) that describes itself.
pub use rlmesh_proto::{
    CURRENT_WORKFLOW_EDITION, PROTOCOL_GENERATION, PeerInfoOverride, parse_declared_edition,
    peer_info_override, supported_workflow_editions, update_peer_info_override,
};

/// Resolve a native participant's workflow edition (COR-6): rungs 1 to 4,
/// warning once per process when an `authored` participant floats.
pub fn resolve_workflow_edition(
    sources: EditionSources<'_>,
    authored: bool,
) -> Result<Option<String>> {
    crate::editions::resolve_native(sources, authored)
}

use crate::{Error, Metadata, Result, ServeOptions, Tensor, Value};

/// The transport's options for `options` and an env endpoint `token` (the env
/// server the Python binding hosts itself).
pub fn serve_options_to_grpc(
    options: ServeOptions,
    token: Option<String>,
) -> rlmesh_grpc::ServeOptions {
    options.to_grpc(crate::serve_options::WireExtras {
        token,
        ..Default::default()
    })
}

/// The env endpoint token `RLMESH_ENV_ENDPOINT_TOKEN` sets, checked as every
/// env server checks it (an invalid value is the `RLMeshException` class).
pub fn env_endpoint_token() -> Result<Option<String>> {
    crate::serve_options::env_endpoint_token(None)
}

/// The shared [`crate::spaces::SpaceSpec`] for a wire spec.
pub fn space_spec_from_wire(spec: rlmesh_spaces::SpaceSpec) -> crate::spaces::SpaceSpec {
    crate::spaces::SpaceSpec::from_wire(spec)
}

/// The wire spec behind a shared [`crate::spaces::SpaceSpec`].
pub fn space_spec_wire(spec: &crate::spaces::SpaceSpec) -> &rlmesh_spaces::SpaceSpec {
    spec.wire()
}

/// The shared [`Tensor`] for a core tensor; refuses an unspecified dtype.
pub fn tensor_from_core(tensor: rlmesh_spaces::Tensor) -> Result<Tensor> {
    if tensor.dtype() == rlmesh_spaces::DType::Unspecified {
        return Err(Error::invalid_argument("tensor dtype must be specified"));
    }
    Ok(Tensor::wrap(tensor))
}

/// The core tensor behind a shared [`Tensor`].
pub fn tensor_core(tensor: &Tensor) -> &rlmesh_spaces::Tensor {
    tensor.core()
}

/// A per-element Box (the C and C++ `ext` builders; SPC-22): `low` and `high`
/// hold one bound per element of `shape`.
pub fn box_elementwise(
    low: &[crate::spaces::Bound],
    high: &[crate::spaces::Bound],
    shape: Vec<i64>,
    options: crate::spaces::BoxOptions,
) -> Result<crate::spaces::Box> {
    let spec = crate::spaces::box_space::box_elementwise_spec(low, high, shape, options.dtype)?;
    crate::spaces::Box::from_spec(spec)
}

/// Draw a wire value for binding plumbing. Shared C values instead use
/// [`crate::spaces::Space::sample`], which restores Dict declaration order.
pub fn space_sample_wire(space: &crate::spaces::Space) -> Result<rlmesh_spaces::SpaceValue> {
    space.sample_wire()
}

/// Strict wire membership for binding plumbing. Shared C values instead use
/// [`crate::spaces::Space::contains`] and its broader facade Value acceptance.
pub fn space_contains_wire(
    space: &crate::spaces::Space,
    value: &rlmesh_spaces::SpaceValue,
) -> bool {
    rlmesh_spaces::contains(space.spec().wire(), value).is_ok()
}

/// The shared [`Value`] for a wire value of the space `spec` (SPC-10).
pub fn value_from_space(
    spec: &rlmesh_spaces::SpaceSpec,
    value: rlmesh_spaces::SpaceValue,
) -> Value {
    crate::value::space_codec::value_from_space(spec, value)
}

/// The wire value of the space `spec` for `value` (SPC-5), unchecked.
pub fn value_to_space(
    spec: &rlmesh_spaces::SpaceSpec,
    value: &Value,
) -> Result<rlmesh_spaces::SpaceValue> {
    crate::value::space_codec::value_to_space(spec, value)
}

/// A [`Metadata`] encoded for the wire (COR-26).
pub fn metadata_to_wire(map: &Metadata) -> Result<rlmesh_spaces::MetaMap> {
    crate::value::meta_codec::metadata_to_wire(map)
}

/// A wire map decoded as [`Metadata`], in the wire's sorted order.
pub fn metadata_from_wire(map: rlmesh_spaces::MetaMap) -> Metadata {
    crate::value::meta_codec::metadata_from_wire(map)
}

/// One [`Value`] encoded for the wire (COR-26).
pub fn meta_value_to_wire(value: &Value) -> Result<rlmesh_spaces::MetaValue> {
    crate::value::meta_codec::value_to_wire(value)
}

/// One wire value decoded as a [`Value`].
pub fn meta_value_from_wire(value: rlmesh_spaces::MetaValue) -> Value {
    crate::value::meta_codec::value_from_wire(value)
}

/// The reserved reset-option key a reset's `trial_index` travels under.
pub use rlmesh_runtime::TRIAL_INDEX_OPTION;

/// The runtime driver's observer seam, its report and telemetry, and the
/// driver's cancellation token: the bindings' native-loop plumbing (R-25).
/// The shared surface is [`crate::RunHooks`], [`crate::RunResult`] and
/// [`crate::ext::CancelToken`].
pub use rlmesh_runtime::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError, NoopRuntimeHooks,
    ObservationEmittedEvent, PeerCeiling, RuntimeHooks, RuntimeReport, StepCompletedEvent,
    telemetry,
};
pub use tokio_util::sync::CancellationToken;

/// The env envelope a served env publishes (ENV-13), for the C ABI:
/// `qualname` overrides `native:<id>`; `component` and `language` name the
/// describing host, `host_runtime` adds its runtime fields.
pub fn env_describe_envelope(
    contract: &crate::EnvContract,
    qualname: Option<&str>,
    catalog: Option<&serde_json::Value>,
    component: &'static str,
    language: &'static str,
    host_runtime: serde_json::Map<String, serde_json::Value>,
    workflow_edition: Option<&str>,
) -> Result<String> {
    crate::env::envelope::env_envelope(
        contract,
        qualname,
        catalog,
        &crate::env::envelope::DescribingHost {
            component,
            language,
            extra: host_runtime,
        },
        workflow_edition,
        contract.num_envs() > 1,
    )
}

/// The shared contract for a wire contract (refuses one without spaces).
pub fn env_contract_from_wire(wire: &rlmesh_spaces::EnvContract) -> Result<crate::EnvContract> {
    crate::EnvContract::from_wire(wire)
}

/// The wire contract behind a shared one.
pub fn env_contract_to_wire(contract: &crate::EnvContract) -> Result<rlmesh_spaces::EnvContract> {
    contract.to_wire()
}

/// ENV-11's scoped signal installation for the Python binding's foreground
/// serve: SIGTERM stops the server through `trigger` and the previous
/// disposition is restored when the scope drops; SIGINT stays Python's.
pub fn foreground_sigterm_scope(
    trigger: rlmesh_grpc::lifecycle::ShutdownTrigger,
) -> crate::env::signals::SignalScope {
    crate::env::signals::enter_with(trigger, false)
}

/// PNG-encode a render frame by Python's lenient rule (ENV-5); `None` for a
/// value that is no image.
pub fn encode_render_frame(frame: &Value) -> Result<Option<Vec<u8>>> {
    crate::env::batch::encode_frame(frame)
}

/// Shared canonical adapter JSON formatter for language bindings.
pub fn adapter_json_format(value: &serde_json::Value) -> Result<String> {
    crate::adapters::format_json(value)
}

/// Resolve a read-only reader for a binding's authored run context.
pub fn reader_from_contract(
    contract: &crate::EnvContract,
    items: &[crate::ReadItem],
) -> Result<crate::Reader> {
    crate::run::read::reader(contract, items)
}

/// An error carrying an in-process binding payload across worker threads.
#[derive(Clone)]
pub struct AttachedError {
    pub(crate) error: Box<Error>,
    payload: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

impl std::fmt::Debug for AttachedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl AttachedError {
    /// Attach a host exception without changing its native classification.
    pub fn attach(error: Error, payload: impl std::any::Any + Send + Sync) -> Error {
        let error = match error {
            Error::Attached(attached) => attached.error,
            error => Box::new(error),
        };
        Error::Attached(Self {
            error,
            payload: std::sync::Arc::new(payload),
        })
    }

    /// Read the host's payload at its outer boundary.
    pub fn payload<T: std::any::Any>(&self) -> Option<&T> {
        self.payload.downcast_ref()
    }

    /// The error without its host payload.
    pub fn error(&self) -> &Error {
        &self.error
    }
}

/// Refuse an authored callback whose awaiting operation was dropped.
pub fn check_model_call() -> Result<()> {
    crate::warnings::check_cancelled()
}

impl PartialEq for AttachedError {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error && std::sync::Arc::ptr_eq(&self.payload, &other.payload)
    }
}
impl Eq for AttachedError {}

/// Emit a native warning through the caller-thread warning machinery.
pub fn emit_warning(category: crate::WarningCategory, message: String) -> Result<()> {
    crate::warnings::warn(category, message)
}

/// Apply a C/C++ host declaration to an unstarted server's published describe.
pub fn env_server_binding_host(
    server: &crate::EnvServer,
    component: &'static str,
    language: &'static str,
    extra: serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    server.binding_host(component, language, extra)
}

/// Validate a binding's factory attributes through the shared branch checker.
pub fn env_factory_check<F: crate::EnvFactory>() -> Result<()> {
    crate::env::factory::check_branch_table::<F>()
}

/// Describe a C/C++ binding's instance with the shared facade and native identity.
pub fn describe_native<'a>(
    obj: impl Into<crate::Describable<'a>>,
    options: crate::DescribeOptions,
) -> Result<crate::Metadata> {
    crate::describe::metadata(
        obj.into(),
        options,
        &crate::describe::Host {
            component: "rlmesh-capi",
            language: "c",
            extra: Default::default(),
        },
    )
}
/// The JSON form of [`describe_native`].
pub fn describe_native_json<'a>(
    obj: impl Into<crate::Describable<'a>>,
    options: crate::DescribeJsonOptions<'a>,
) -> Result<String> {
    crate::describe::envelope(
        obj.into(),
        options,
        &crate::describe::Host {
            component: "rlmesh-capi",
            language: "c",
            extra: Default::default(),
        },
    )
}
/// A binding's model type form: declaration validation without constructing a policy.
pub fn model_declaration_describable(
    decl: crate::binding::model::PolicyDecl,
) -> Result<crate::Describable<'static>> {
    crate::describe::Describable::from_declaration(decl)
}
/// COR-16's cached C/C++ server shorthand (no public Rust extra).
pub fn env_server_native_describe_json(server: &crate::EnvServer) -> Result<String> {
    server.binding_description()
}

/// Factory serving with the same C/C++ Describe identity as [`describe_native`].
pub fn env_factory_native_serve<F: crate::EnvFactory>(
    factory: &mut F,
    address: &str,
    options: crate::EnvFactoryServeOptions,
) -> Result<()> {
    crate::env::factory::serve_host(
        factory,
        address,
        options,
        &crate::describe::Host {
            component: "rlmesh-capi",
            language: "c",
            extra: Default::default(),
        },
    )
}
