//! The describe envelope for a native env: the `rlmesh.describe.v1` artifact a
//! Python env gets from `rlmesh describe`, built here from the env's contract.
//!
//! A native host has no signature to reflect and no variant catalog, so its
//! envelope carries the target, the spaces, the published tags, and the runtime
//! edition handshake. The managed platform reads it off a baked
//! `dev.rlmesh.describe` image label, or, for an image without one, off the
//! handshake `PeerInfo.extra` the server stamps at bind, exactly as
//! `rlmesh.serve` does for a Python env.
#![allow(unsafe_code)] // FFI: the exported describe call.

use std::sync::Mutex;

use rlmesh_adapters::v1::{
    DESCRIBE_METADATA_KEY, ENV_METADATA_KEY, build_describe_envelope, space_spec_to_json,
};
use rlmesh_spaces::{EnvContract, SpaceSpec};
use serde_json::{Value, json};

use crate::abi::status::{CapiError, RlmeshStatus, guard};
use crate::adapters::meta_to_json;
use crate::codec::RlmeshBytes;
use crate::env::RlmeshEnv;

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

    /// Rebuild the envelope with the edition the server declares and put it on
    /// the handshake `PeerInfo.extra` under [`DESCRIBE_METADATA_KEY`], beside
    /// whatever else the process override already reports. The override is
    /// process-wide, so with several envs in one process the last bound wins.
    pub(crate) fn publish(&self, workflow_edition: Option<&str>) -> Result<(), CapiError> {
        let envelope = envelope(&self.contract, workflow_edition)?;
        rlmesh::update_peer_info_override(|info| {
            info.extra
                .insert(DESCRIBE_METADATA_KEY.to_string(), envelope.clone());
        });
        *lock(&self.envelope) = envelope;
        Ok(())
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
    env: *const RlmeshEnv,
    out: *mut RlmeshBytes,
) -> RlmeshStatus {
    guard(|| {
        let handle = unsafe { env.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null env"))?;
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        *out = RlmeshBytes::from_vec(handle.describe().json().into_bytes());
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
        _args: *const crate::env::RlmeshResetArgs,
        _out: *mut crate::env::RlmeshResetResult,
    ) -> std::ffi::c_int {
        1
    }

    unsafe extern "C" fn step(
        _user_data: *mut std::ffi::c_void,
        _action: *const crate::value::handle::RlmeshValue,
        _out: *mut crate::env::RlmeshStepResult,
    ) -> std::ffi::c_int {
        1
    }

    fn describe_json(env: *const RlmeshEnv) -> Value {
        let mut out = RlmeshBytes::from_vec(Vec::new());
        let status = unsafe { rlmesh_env_describe_json(env, &mut out) };
        assert_eq!(status, RlmeshStatus::Ok);
        parse(&String::from_utf8(unsafe { out.into_vec() }).expect("utf-8"))
    }

    #[test]
    fn the_c_export_describes_the_env_and_bind_puts_it_on_the_handshake() {
        use std::ffi::CString;

        use crate::env::{
            RlmeshEnvConfig, RlmeshEnvVtable, rlmesh_env_bind, rlmesh_env_free, rlmesh_env_new,
        };
        use crate::model::RlmeshServeOptions;
        use crate::spaces::{rlmesh_space_box, rlmesh_space_dict, rlmesh_space_free};
        use crate::value::dtype::RlmeshDType;

        const F32: RlmeshDType = RlmeshDType {
            code: 2,
            bits: 32,
            lanes: 1,
        };
        // A host's own override fields survive the describe stamp.
        rlmesh::update_peer_info_override(|info| info.os_version = "6.1".into());

        let three = [3i64];
        let eef = unsafe { rlmesh_space_box(F32, three.as_ptr(), 1, -2.0, 2.0) };
        let key = CString::new("eef_pos").unwrap();
        let obs = unsafe { rlmesh_space_dict([key.as_ptr()].as_ptr(), [eef].as_ptr(), 1) };
        let one = [1i64];
        let act = unsafe { rlmesh_space_box(F32, one.as_ptr(), 1, -1.0, 1.0) };
        let id = CString::new("Native-v0").unwrap();
        let tags = CString::new(
            r#"{"observation": {"eef_pos": {"type": "state", "role": "proprio/eef_pos"}},
                "action": {"components": [{"role": "action/gripper", "dim": 1}]}}"#,
        )
        .unwrap();
        let config = RlmeshEnvConfig {
            struct_size: std::mem::size_of::<RlmeshEnvConfig>(),
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
        let vtable = RlmeshEnvVtable {
            struct_size: std::mem::size_of::<RlmeshEnvVtable>(),
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
        assert_eq!(status, RlmeshStatus::Ok);

        let before = describe_json(env);
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
        let edition = CString::new(base).unwrap();
        let options = RlmeshServeOptions {
            token: std::ptr::null(),
            allow_remote_shutdown: false,
            idle_timeout_ms: 0,
            drain_timeout_ms: 0,
            close_timeout_ms: 0,
            predict_concurrency: 0,
            workflow_edition: edition.as_ptr(),
        };
        let address = CString::new("127.0.0.1:0").unwrap();
        let status =
            unsafe { rlmesh_env_bind(env, address.as_ptr(), &options, std::ptr::null_mut()) };
        assert_eq!(status, RlmeshStatus::Ok);
        let after = describe_json(env);
        assert_eq!(after["runtime"]["preferred_workflow_edition"], base);
        let peer = rlmesh::peer_info_override().expect("an override is installed");
        let stamped = peer
            .extra
            .get(DESCRIBE_METADATA_KEY)
            .expect("describe on the handshake");
        assert_eq!(parse(stamped), after);
        assert_eq!(peer.os_version, "6.1", "other override fields are kept");
        unsafe { rlmesh_env_free(env) };

        let mut out = RlmeshBytes::from_vec(Vec::new());
        let status = unsafe { rlmesh_env_describe_json(std::ptr::null(), &mut out) };
        assert_eq!(status, RlmeshStatus::InvalidArgument);
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
