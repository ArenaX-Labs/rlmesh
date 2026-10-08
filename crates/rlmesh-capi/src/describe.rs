//! The describe envelope for a native env: the `rlmesh.describe.v1` artifact a
//! Python env gets from `rlmesh describe`, built here from the env's contract.
//!
//! A native host has no signature to reflect and no variant catalog, so its
//! envelope carries the target, the spaces, the published tags, and the runtime
//! edition handshake. The managed platform reads it off a baked
//! `dev.rlmesh.describe` image label, or, for an image without one, off the
//! handshake `PeerInfo.extra` the server stamps at bind, as `rlmesh.serve` does
//! for a Python env. Each bound env stamps its own server, so several envs in
//! one process each advertise their own contract.
#![allow(unsafe_code)] // FFI: the exported describe call.

use std::sync::Mutex;

use rlmesh_adapters::v1::{
    DESCRIBE_METADATA_KEY, ENV_METADATA_KEY, build_describe_envelope, space_spec_to_json,
};
use rlmesh_spaces::{EnvContract, SpaceSpec};
use serde_json::{Value, json};

use crate::abi::status::{CapiError, RLMeshStatus, guard};
use crate::adapters::meta_to_json;
use crate::codec::RLMeshBytes;
use crate::env::RLMeshEnv;

/// Component name the envelope's `runtime` reports for a C ABI host.
const COMPONENT: &str = "rlmesh-capi";

/// An env's describe envelope: built at `rlmesh_env_new` with the build's own
/// edition, rebuilt at bind with the edition the server declares.
pub(crate) struct Describe {
    contract: EnvContract,
    envelope: Mutex<String>,
}

impl Describe {
    pub(crate) fn new(contract: &EnvContract) -> Result<Self, CapiError> {
        Ok(Self {
            envelope: Mutex::new(envelope(contract, None)?),
            contract: contract.clone(),
        })
    }

    /// The current envelope JSON.
    pub(crate) fn json(&self) -> String {
        lock(&self.envelope).clone()
    }

    /// The envelope rebuilt with the edition the server declares, for bind to
    /// put on this env's own handshake `PeerInfo.extra` (see
    /// [`Describe::bound_extra`]). Nothing changes until bind succeeds and
    /// calls [`Describe::commit`].
    pub(crate) fn for_bind(&self, workflow_edition: Option<&str>) -> Result<String, CapiError> {
        envelope(&self.contract, workflow_edition)
    }

    /// The handshake `PeerInfo.extra` entry carrying `envelope`, set per server
    /// (on its serve options) so each endpoint in a process reports its own.
    pub(crate) fn bound_extra(envelope: &str) -> (String, String) {
        (DESCRIBE_METADATA_KEY.to_string(), envelope.to_string())
    }

    /// Keep the envelope a successful bind advertised as this env's describe.
    pub(crate) fn commit(&self, envelope: String) {
        *lock(&self.envelope) = envelope;
    }
}

/// Build the env envelope. `workflow_edition` is the declaration the server
/// sends (None or blank: the build's newest edition, as the handshake spells it).
pub(crate) fn envelope(
    contract: &EnvContract,
    workflow_edition: Option<&str>,
) -> Result<String, CapiError> {
    let space = |space: Option<&SpaceSpec>, which: &str| match space {
        Some(space) => space_spec_to_json(space)
            .map_err(|err| CapiError::invalid_arg(format!("{which}: {err}"))),
        None => Err(CapiError::invalid_arg(format!("{which} is missing"))),
    };
    let env_tags = contract
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get(ENV_METADATA_KEY))
        .map_or(Value::Null, meta_to_json);
    let preferred = workflow_edition
        .map(str::trim)
        .filter(|edition| !edition.is_empty())
        .unwrap_or(rlmesh::CURRENT_WORKFLOW_EDITION);
    let pieces = json!({
        "target": {"entrypoint": null, "qualname": format!("native:{}", contract.id)},
        "env_spec": {
            "observation_space": space(contract.observation_space.as_ref(), "observation_space")?,
            "action_space": space(contract.action_space.as_ref(), "action_space")?,
        },
        "env_tags": env_tags,
        "runtime": {
            "component": COMPONENT,
            "language": "c",
            "package_version": env!("CARGO_PKG_VERSION"),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "protocol_generation": rlmesh::PROTOCOL_GENERATION,
            "supported_workflow_editions": rlmesh::supported_workflow_editions(),
            "preferred_workflow_edition": preferred,
        },
    });
    build_describe_envelope("env", &pieces.to_string(), None)
        .map_err(|err| CapiError::internal(format!("invalid describe envelope: {err}")))
}

/// Write the env's describe envelope (UTF-8 JSON, not NUL-terminated) to `out`;
/// free with `rlmesh_bytes_free`. Valid before and after bind: after bind it
/// declares the edition the server was bound with.
///
/// # Safety
/// `env` must be a live handle; `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_describe_json(
    env: *const RLMeshEnv,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let handle = unsafe { env.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null env"))?;
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        *out = RLMeshBytes::from_vec(handle.describe().json().into_bytes());
        Ok(())
    })
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use rlmesh_spaces::{
        BoxBounds, BoxSpec, DType, DiscreteSpec, MetaMap, MetaValue, SpaceKind, UniformBounds,
    };

    use super::*;

    fn contract(tags: Option<MetaValue>) -> EnvContract {
        let mut metadata = MetaMap::new();
        if let Some(tags) = tags {
            metadata.insert(ENV_METADATA_KEY.to_string(), tags);
        }
        EnvContract {
            id: "Reach-v0".into(),
            observation_space: Some(SpaceSpec {
                shape: vec![3],
                dtype: DType::Float32,
                spec: Some(SpaceKind::Box(BoxSpec {
                    bounds: Some(BoxBounds::Uniform(UniformBounds {
                        low: f64::NEG_INFINITY,
                        high: 1.0,
                    })),
                })),
            }),
            action_space: Some(SpaceSpec {
                shape: vec![],
                dtype: DType::Int64,
                spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 2, start: 0 })),
            }),
            metadata: (!metadata.is_empty()).then_some(metadata),
            num_envs: 1,
            ..Default::default()
        }
    }

    fn built(result: Result<String, CapiError>) -> String {
        result.unwrap_or_else(|err| panic!("envelope failed: {}", err.message))
    }

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).expect("envelope is JSON")
    }

    #[test]
    fn envelope_carries_target_spaces_tags_and_runtime() {
        let tags = MetaValue::Map(
            [("observation".to_string(), MetaValue::Map(MetaMap::new()))]
                .into_iter()
                .collect(),
        );
        let text = built(envelope(&contract(Some(tags)), None));
        assert!(text.starts_with(r#"{"schema_version":1,"kind":"env","#));
        let value = parse(&text);
        assert_eq!(
            value["target"],
            json!({"entrypoint": null, "qualname": "native:Reach-v0"})
        );
        let obs = &value["env_spec"]["observation_space"];
        assert_eq!(obs["kind"], "box");
        assert_eq!(obs["details"]["low"], Value::Null, "-inf renders as null");
        assert_eq!(obs["details"]["high"], 1.0);
        assert_eq!(value["env_spec"]["action_space"]["details"]["n"], 2);
        assert_eq!(value["env_tags"], json!({"observation": {}}));
        let runtime = &value["runtime"];
        assert_eq!(runtime["language"], "c");
        assert_eq!(runtime["protocol_generation"], rlmesh::PROTOCOL_GENERATION);
        assert_eq!(
            runtime["preferred_workflow_edition"],
            rlmesh::CURRENT_WORKFLOW_EDITION
        );
        assert_eq!(
            runtime["supported_workflow_editions"],
            json!(rlmesh::supported_workflow_editions())
        );
    }

    #[test]
    fn untagged_env_has_null_tags_and_a_declared_edition_is_preferred() {
        let text = built(envelope(&contract(None), Some(" 2026.06 ")));
        let value = parse(&text);
        assert_eq!(value["env_tags"], Value::Null);
        assert_eq!(value["runtime"]["preferred_workflow_edition"], "2026.06");
    }

    unsafe extern "C" fn reset(
        _user_data: *mut std::ffi::c_void,
        _args: *const crate::env::RLMeshResetArgs,
        _out: *mut crate::env::RLMeshResetResult,
    ) -> std::ffi::c_int {
        1
    }

    unsafe extern "C" fn step(
        _user_data: *mut std::ffi::c_void,
        _action: *const crate::value::handle::RLMeshValue,
        _out: *mut crate::env::RLMeshStepResult,
    ) -> std::ffi::c_int {
        1
    }

    fn describe_json(env: *const RLMeshEnv) -> Value {
        let mut out = RLMeshBytes::from_vec(Vec::new());
        let status = unsafe { rlmesh_env_describe_json(env, &mut out) };
        assert_eq!(status, RLMeshStatus::Ok);
        parse(&String::from_utf8(unsafe { out.into_vec() }).expect("utf-8"))
    }

    const F32: crate::value::dtype::RLMeshDType = crate::value::dtype::RLMeshDType {
        code: 2,
        bits: 32,
        lanes: 1,
    };

    /// A C env `id` whose observation is Dict{eef_pos: f32[3] in [-bound, bound]}.
    fn native_env(id: &str, bound: f64) -> *mut RLMeshEnv {
        use std::ffi::CString;

        use crate::env::{RLMeshEnvConfig, RLMeshEnvVtable, rlmesh_env_new};
        use crate::spaces::{rlmesh_space_box, rlmesh_space_dict, rlmesh_space_free};

        let three = [3i64];
        let eef = unsafe { rlmesh_space_box(F32, three.as_ptr(), 1, -bound, bound) };
        let key = CString::new("eef_pos").unwrap();
        let obs = unsafe { rlmesh_space_dict([key.as_ptr()].as_ptr(), [eef].as_ptr(), 1) };
        let one = [1i64];
        let act = unsafe { rlmesh_space_box(F32, one.as_ptr(), 1, -1.0, 1.0) };
        let id = CString::new(id).unwrap();
        let tags = CString::new(
            r#"{"observation": {"eef_pos": {"type": "state", "role": "proprio/eef_pos"}},
                "action": {"components": [{"role": "action/gripper", "dim": 1}]}}"#,
        )
        .unwrap();
        let config = RLMeshEnvConfig {
            struct_size: std::mem::size_of::<RLMeshEnvConfig>(),
            id: id.as_ptr(),
            observation_space: obs,
            action_space: act,
            adapter_tags_json: tags.as_ptr(),
            reset_options: std::ptr::null(),
            num_reset_options: 0,
            render_mode: std::ptr::null(),
            metadata_json: std::ptr::null(),
            foreground: false,
        };
        let vtable = RLMeshEnvVtable {
            struct_size: std::mem::size_of::<RLMeshEnvVtable>(),
            reset: Some(reset),
            step: Some(step),
            render: None,
            close: None,
        };
        let mut env = std::ptr::null_mut();
        let status = unsafe { rlmesh_env_new(&vtable, &config, std::ptr::null_mut(), &mut env) };
        unsafe {
            rlmesh_space_free(obs);
            rlmesh_space_free(act);
        }
        assert_eq!(status, RLMeshStatus::Ok);
        env
    }

    /// Bind `env` at `address` declaring `edition`; the bound address on success.
    fn bind(env: *mut RLMeshEnv, address: &str, edition: &str) -> Result<String, RLMeshStatus> {
        use std::ffi::CString;

        use crate::env::rlmesh_env_bind;
        use crate::model::RLMeshServeOptions;

        let edition = CString::new(edition).unwrap();
        let options = RLMeshServeOptions {
            token: std::ptr::null(),
            allow_remote_shutdown: false,
            idle_timeout_ms: 0,
            drain_timeout_ms: 0,
            close_timeout_ms: 0,
            predict_concurrency: 0,
            workflow_edition: edition.as_ptr(),
        };
        let address = CString::new(address).unwrap();
        let mut out = RLMeshBytes::from_vec(Vec::new());
        match unsafe { rlmesh_env_bind(env, address.as_ptr(), &options, &mut out) } {
            RLMeshStatus::Ok => Ok(String::from_utf8(unsafe { out.into_vec() }).unwrap()),
            status => Err(status),
        }
    }

    #[test]
    fn the_c_export_describes_each_env_and_only_a_successful_bind_publishes() {
        use crate::env::rlmesh_env_free;

        // A host's own process-wide override fields are left as they are.
        rlmesh::update_peer_info_override(|info| info.os_version = "6.1".into());

        let first = native_env("Native-v0", 2.0);
        let second = native_env("Other-v0", 5.0);

        let before = describe_json(first);
        assert_eq!(before["kind"], "env");
        assert_eq!(before["target"]["qualname"], "native:Native-v0");
        let eef_space = &before["env_spec"]["observation_space"]["details"]["spaces"]["eef_pos"];
        assert_eq!(eef_space["details"]["low"], -2.0);
        assert_eq!(
            before["env_tags"]["observation"]["eef_pos"]["role"],
            "proprio/eef_pos"
        );
        assert_eq!(
            before["runtime"]["preferred_workflow_edition"],
            rlmesh::CURRENT_WORKFLOW_EDITION
        );

        // Bound with a declared edition: the envelope (and the handshake copy)
        // declares exactly what the server sends.
        let base = rlmesh::CURRENT_WORKFLOW_EDITION.split('-').next().unwrap();
        let first_address = bind(first, "127.0.0.1:0", base).expect("bind the first env");
        bind(second, "127.0.0.1:0", rlmesh::CURRENT_WORKFLOW_EDITION).expect("bind the second");
        let after = describe_json(first);
        assert_eq!(after["runtime"]["preferred_workflow_edition"], base);

        // A bind that fails (the first env's port is taken) publishes nothing:
        // its describe keeps the pre-bind edition and the others are untouched.
        let third = native_env("Failed-v0", 9.0);
        let third_before = describe_json(third);
        assert!(bind(third, &first_address, base).is_err());
        assert_eq!(describe_json(third), third_before);
        unsafe { rlmesh_env_free(third) };
        assert!(
            rlmesh::peer_info_override()
                .is_none_or(|info| !info.extra.contains_key(DESCRIBE_METADATA_KEY)),
            "the describe is per server, not process-wide"
        );

        // Each endpoint carries its own envelope on its own handshake (the
        // per-server `ServeOptions::peer_info_extra`, covered over the wire in
        // rlmesh-grpc); here the two bound envs keep distinct contracts.
        let second_after = describe_json(second);
        assert_eq!(second_after["target"]["qualname"], "native:Other-v0");
        let other_eef =
            &second_after["env_spec"]["observation_space"]["details"]["spaces"]["eef_pos"];
        assert_eq!(other_eef["details"]["low"], -5.0);
        assert_eq!(
            second_after["runtime"]["preferred_workflow_edition"],
            rlmesh::CURRENT_WORKFLOW_EDITION
        );
        assert_eq!(
            describe_json(first),
            after,
            "binding the second left the first alone"
        );
        assert_eq!(
            rlmesh::peer_info_override().map(|info| info.os_version),
            Some("6.1".into()),
            "other override fields are kept"
        );
        for env in [first, second] {
            unsafe { rlmesh_env_free(env) };
        }

        let mut out = RLMeshBytes::from_vec(Vec::new());
        let status = unsafe { rlmesh_env_describe_json(std::ptr::null(), &mut out) };
        assert_eq!(status, RLMeshStatus::InvalidArgument);
    }

    #[test]
    fn a_contract_without_spaces_is_refused() {
        let mut bare = contract(None);
        bare.action_space = None;
        let Err(err) = envelope(&bare, None) else {
            panic!("a spaceless contract built an envelope");
        };
        assert!(err.message.contains("action_space"), "{}", err.message);
    }
}
