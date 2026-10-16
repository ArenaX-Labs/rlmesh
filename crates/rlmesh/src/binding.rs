//! Plumbing for the in-repo bindings (the Python extension and the C ABI).
//!
//! Hidden from the docs and exempt from semver: nothing here is part of the
//! shared surface, and it changes with the bindings.

pub mod env;
pub mod spaces;

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

/// Draw a wire value from `space` (the C ABI's values are still wire values).
pub fn space_sample_wire(space: &crate::spaces::Space) -> Result<rlmesh_spaces::SpaceValue> {
    space.sample_wire()
}

/// Whether the wire `value` is a member of `space`.
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
