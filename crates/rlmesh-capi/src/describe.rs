//! The describe envelope for a native env: the `rlmesh.describe.v1` artifact a
//! Python env gets from `rlmesh describe`, built here from the env's contract.
//!
//! A native host has no signature to reflect, so its envelope carries the
//! target, the spaces, the published tags, the variant catalog the config
//! declares (if any), and the runtime edition handshake. The managed platform reads it off a baked
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
use crate::env::RLMeshExtLegacyEnv;

/// Component name the envelope's `runtime` reports for a C ABI host.
const COMPONENT: &str = "rlmesh-capi";

/// Validate the optional host declaration before it can override runtime fields.
/// Component, package, OS/arch and protocol/edition remain library-owned.
pub(crate) fn host_runtime(text: &str) -> Result<Value, CapiError> {
    let invalid = |message: String| CapiError::invalid_arg(format!("host_runtime_json: {message}"));
    let value: Value = serde_json::from_str(text).map_err(|err| invalid(err.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("expected a JSON object".into()))?;
    let non_empty = |value: &Value, field: &str| {
        value
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| invalid(format!("{field} must be a non-empty string")))
    };
    let mut normalized = serde_json::Map::new();
    for (key, value) in object {
        let value = match key.as_str() {
            "language" | "language_version" => Value::String(non_empty(value, key)?),
            "framework_versions" => {
                let versions = value
                    .as_object()
                    .ok_or_else(|| invalid("framework_versions must be an object".into()))?;
                let mut normalized_versions = serde_json::Map::new();
                for (name, version) in versions {
                    let name = name.trim();
                    if name.is_empty() {
                        return Err(invalid("framework names must be non-empty".into()));
                    }
                    let version = non_empty(version, "framework version")?;
                    if normalized_versions
                        .insert(name.to_owned(), Value::String(version))
                        .is_some()
                    {
                        return Err(invalid(format!(
                            "duplicate framework name {name:?} after trimming"
                        )));
                    }
                }
                Value::Object(normalized_versions)
            }
            "toolchain" => {
                let toolchain = value
                    .as_object()
                    .ok_or_else(|| invalid("toolchain must be an object".into()))?;
                if toolchain.len() != 2
                    || !toolchain.contains_key("name")
                    || !toolchain.contains_key("version")
                {
                    return Err(invalid(
                        "toolchain must contain only name and version".into(),
                    ));
                }
                let name = non_empty(&toolchain["name"], "toolchain.name")?;
                let version = non_empty(&toolchain["version"], "toolchain.version")?;
                if !matches!(name.as_str(), "gcc" | "clang" | "msvc") {
                    return Err(invalid("toolchain.name must be gcc, clang or msvc".into()));
                }
                if !version.split_once('.').is_some_and(|(major, minor)| {
                    !major.is_empty()
                        && !minor.is_empty()
                        && major.bytes().all(|b| b.is_ascii_digit())
                        && minor.bytes().all(|b| b.is_ascii_digit())
                }) {
                    return Err(invalid("toolchain.version must be major.minor".into()));
                }
                json!({"name": name, "version": version})
            }
            _ => return Err(invalid(format!("unknown field {key:?}"))),
        };
        normalized.insert(key.clone(), value);
    }
    Ok(Value::Object(normalized))
}

/// An env's describe envelope: built at `rlmesh_ext_legacy_env_new` with the build's own
/// edition, rebuilt at bind with the edition the server declares.
pub(crate) struct Describe {
    contract: EnvContract,
    /// The config's `variants_catalog_json`, checked: `variants.catalog`.
    catalog: Option<Value>,
    host_runtime: Option<Value>,
    envelope: Mutex<String>,
}

impl Describe {
    pub(crate) fn new(
        contract: &EnvContract,
        catalog: Option<Value>,
        host_runtime: Option<Value>,
    ) -> Result<Self, CapiError> {
        Ok(Self {
            envelope: Mutex::new(envelope(
                contract,
                catalog.as_ref(),
                host_runtime.as_ref(),
                None,
            )?),
            contract: contract.clone(),
            catalog,
            host_runtime,
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
        envelope(
            &self.contract,
            self.catalog.as_ref(),
            self.host_runtime.as_ref(),
            workflow_edition,
        )
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

/// Build the env envelope. `catalog` is the declared variant catalog;
/// `workflow_edition` is the declaration the server sends (None or blank: the
/// build's newest edition, as the handshake spells it).
pub(crate) fn envelope(
    contract: &EnvContract,
    catalog: Option<&Value>,
    host_runtime: Option<&Value>,
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
        .unwrap_or(rlmesh::binding::CURRENT_WORKFLOW_EDITION);
    let mut pieces = json!({
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
            "protocol_generation": rlmesh::binding::PROTOCOL_GENERATION,
            "supported_workflow_editions": rlmesh::binding::supported_workflow_editions(),
            "preferred_workflow_edition": preferred,
        },
    });
    if let Some(host_runtime) = host_runtime {
        for (key, value) in host_runtime.as_object().expect("validated host runtime") {
            pieces["runtime"][key] = value.clone();
        }
    }
    if let Some(catalog) = catalog {
        pieces["variants"] = json!({"catalog": catalog});
    }
    build_describe_envelope("env", &pieces.to_string(), None)
        .map_err(|err| CapiError::invalid_arg(format!("invalid describe envelope: {err}")))
}

/// Write the env's describe envelope (UTF-8 JSON, not NUL-terminated) to `out`;
/// free with `rlmesh_bytes_free`. Valid before and after bind: after bind it
/// declares the edition the server was bound with.
///
/// # Safety
/// `env` must be a live handle; `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_legacy_env_describe_json(
    env: *const RLMeshExtLegacyEnv,
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
        let text = built(envelope(&contract(Some(tags)), None, None, None));
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
        assert!(runtime.get("language_version").is_none());
        assert!(runtime.get("framework_versions").is_none());
        assert_eq!(
            runtime["protocol_generation"],
            rlmesh::binding::PROTOCOL_GENERATION
        );
        assert_eq!(
            runtime["preferred_workflow_edition"],
            rlmesh::binding::CURRENT_WORKFLOW_EDITION
        );
        assert_eq!(
            runtime["supported_workflow_editions"],
            json!(rlmesh::binding::supported_workflow_editions())
        );
    }

    #[test]
    fn untagged_env_has_null_tags_and_a_declared_edition_is_preferred() {
        let text = built(envelope(&contract(None), None, None, Some(" 2026.06 ")));
        let value = parse(&text);
        assert_eq!(value["env_tags"], Value::Null);
        assert_eq!(value["runtime"]["preferred_workflow_edition"], "2026.06");
        assert!(value.get("variants").is_none(), "no catalog, no variants");
    }

    #[test]
    fn a_declared_catalog_is_the_envelopes_variants_catalog() {
        let catalog = json!([{"id": "viper/firm", "params": {"vehicle": "viper"}}]);
        let value = parse(&built(envelope(
            &contract(None),
            Some(&catalog),
            None,
            None,
        )));
        assert_eq!(value["variants"], json!({"catalog": catalog}));
    }

    #[test]
    fn host_runtime_overrides_only_declared_fields_and_survives_bind() {
        let runtime = host_runtime(
            r#"{"language":"c++","language_version":"20","toolchain":{"name":"clang","version":"19.1"}}"#,
        )
        .unwrap_or_else(|err| panic!("{}", err.message));
        let describe = Describe::new(&contract(None), None, Some(runtime))
            .unwrap_or_else(|err| panic!("{}", err.message));
        for value in [
            parse(&describe.json()),
            parse(&built(describe.for_bind(Some("2026.06")))),
        ] {
            assert_eq!(value["runtime"]["language"], "c++");
            assert_eq!(value["runtime"]["language_version"], "20");
            assert_eq!(
                value["runtime"]["toolchain"],
                json!({"name":"clang","version":"19.1"})
            );
            assert_eq!(value["runtime"]["component"], COMPONENT);
            assert_eq!(
                value["runtime"]["package_version"],
                env!("CARGO_PKG_VERSION")
            );
        }
        let partial = host_runtime(r#"{"language_version":"17"}"#)
            .unwrap_or_else(|err| panic!("{}", err.message));
        let value = parse(&built(envelope(
            &contract(None),
            None,
            Some(&partial),
            None,
        )));
        assert_eq!(value["runtime"]["language"], "c");
        assert_eq!(value["runtime"]["language_version"], "17");
        assert!(value["runtime"].get("framework_versions").is_none());
        let empty = host_runtime("{}").unwrap_or_else(|err| panic!("{}", err.message));
        assert_eq!(
            built(envelope(&contract(None), None, Some(&empty), None)),
            built(envelope(&contract(None), None, None, None))
        );
    }

    unsafe extern "C" fn reset(
        _user_data: *mut std::ffi::c_void,
        _args: *const crate::env::RLMeshExtLegacyResetArgs,
        _out: *mut crate::env::RLMeshExtLegacyResetResult,
    ) -> std::ffi::c_int {
        1
    }

    unsafe extern "C" fn step(
        _user_data: *mut std::ffi::c_void,
        _action: *const crate::value::handle::RLMeshValue,
        _out: *mut crate::env::RLMeshExtLegacyStepResult,
    ) -> std::ffi::c_int {
        1
    }

    fn describe_json(env: *const RLMeshExtLegacyEnv) -> Value {
        let mut out = RLMeshBytes::from_vec(Vec::new());
        let status = unsafe { rlmesh_ext_legacy_env_describe_json(env, &mut out) };
        assert_eq!(status, RLMeshStatus::Ok);
        parse(&String::from_utf8(unsafe { out.into_vec() }).expect("utf-8"))
    }

    const F32: i32 = 11;

    /// A C env `id` whose observation is Dict{eef_pos: f32[3] in [-bound, bound]}.
    fn native_env(id: &str, bound: f64) -> *mut RLMeshExtLegacyEnv {
        native_env_with(id, bound, None).expect("rlmesh_ext_legacy_env_new")
    }

    /// [`native_env`] declaring `catalog` as its `variants_catalog_json`.
    fn native_env_with(
        id: &str,
        bound: f64,
        catalog: Option<&str>,
    ) -> Result<*mut RLMeshExtLegacyEnv, RLMeshStatus> {
        native_env_runtime(id, bound, catalog, None, false)
    }

    fn native_env_runtime(
        id: &str,
        bound: f64,
        catalog: Option<&str>,
        runtime: Option<&str>,
        old_config: bool,
    ) -> Result<*mut RLMeshExtLegacyEnv, RLMeshStatus> {
        native_env_runtime_bytes(id, bound, catalog, runtime.map(str::as_bytes), old_config)
    }

    fn native_env_runtime_bytes(
        id: &str,
        bound: f64,
        catalog: Option<&str>,
        runtime: Option<&[u8]>,
        old_config: bool,
    ) -> Result<*mut RLMeshExtLegacyEnv, RLMeshStatus> {
        use std::ffi::CString;

        use crate::env::{
            RLMeshExtLegacyEnvConfig, RLMeshExtLegacyEnvVtable, rlmesh_ext_legacy_env_new,
        };
        use crate::spaces::{rlmesh_space_box, rlmesh_space_dict, rlmesh_space_spec_free};

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
        let catalog = catalog.map(|catalog| CString::new(catalog).unwrap());
        let runtime = runtime.map(|runtime| CString::new(runtime).unwrap());
        let config_size = if old_config {
            std::mem::offset_of!(RLMeshExtLegacyEnvConfig, host_runtime_json)
        } else {
            std::mem::size_of::<RLMeshExtLegacyEnvConfig>()
        };
        let config = RLMeshExtLegacyEnvConfig {
            struct_size: config_size,
            id: id.as_ptr(),
            observation_space: obs,
            action_space: act,
            adapter_tags_json: tags.as_ptr(),
            reset_options: std::ptr::null(),
            num_reset_options: 0,
            render_mode: std::ptr::null(),
            metadata_json: std::ptr::null(),
            foreground: false,
            variants_catalog_json: catalog.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            host_runtime_json: runtime.as_ref().map_or(std::ptr::null(), |r| r.as_ptr()),
        };
        let vtable = RLMeshExtLegacyEnvVtable {
            struct_size: std::mem::size_of::<RLMeshExtLegacyEnvVtable>(),
            reset: Some(reset),
            step: Some(step),
            render: None,
            close: None,
        };
        let mut env = std::ptr::null_mut();
        // Allocate exactly the declared prefix, aligned like the config. An old
        // host never allocated the appended field; the library must not read it.
        let mut storage = vec![0usize; config_size / std::mem::size_of::<usize>()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const config).cast::<u8>(),
                storage.as_mut_ptr().cast::<u8>(),
                config_size,
            );
        }
        let status = unsafe {
            rlmesh_ext_legacy_env_new(
                &vtable,
                storage.as_ptr().cast(),
                std::ptr::null_mut(),
                &mut env,
            )
        };
        unsafe {
            rlmesh_space_spec_free(obs);
            rlmesh_space_spec_free(act);
        }
        if status == RLMeshStatus::Ok {
            Ok(env)
        } else {
            Err(status)
        }
    }

    /// Bind `env` at `address` declaring `edition`; the bound address on success.
    fn bind(
        env: *mut RLMeshExtLegacyEnv,
        address: &str,
        edition: &str,
    ) -> Result<String, RLMeshStatus> {
        use std::ffi::CString;

        use crate::env::rlmesh_ext_legacy_env_bind;
        use crate::model::RLMeshExtLegacyServeOptions;

        let edition = CString::new(edition).unwrap();
        let options = RLMeshExtLegacyServeOptions {
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
        match unsafe { rlmesh_ext_legacy_env_bind(env, address.as_ptr(), &options, &mut out) } {
            RLMeshStatus::Ok => Ok(String::from_utf8(unsafe { out.into_vec() }).unwrap()),
            status => Err(status),
        }
    }

    #[test]
    fn the_c_export_describes_each_env_and_only_a_successful_bind_publishes() {
        use crate::env::rlmesh_ext_legacy_env_free;

        // A host's own process-wide override fields are left as they are.
        rlmesh::binding::update_peer_info_override(|info| info.os_version = "6.1".into());

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
            rlmesh::binding::CURRENT_WORKFLOW_EDITION
        );

        // Bound with a declared edition: the envelope (and the handshake copy)
        // declares exactly what the server sends.
        let base = rlmesh::binding::CURRENT_WORKFLOW_EDITION
            .split('-')
            .next()
            .unwrap();
        let first_address = bind(first, "127.0.0.1:0", base).expect("bind the first env");
        bind(
            second,
            "127.0.0.1:0",
            rlmesh::binding::CURRENT_WORKFLOW_EDITION,
        )
        .expect("bind the second");
        let after = describe_json(first);
        assert_eq!(after["runtime"]["preferred_workflow_edition"], base);

        // A bind that fails (the first env's port is taken) publishes nothing:
        // its describe keeps the pre-bind edition and the others are untouched.
        let third = native_env("Failed-v0", 9.0);
        let third_before = describe_json(third);
        assert!(bind(third, &first_address, base).is_err());
        assert_eq!(describe_json(third), third_before);
        unsafe { rlmesh_ext_legacy_env_free(third) };
        assert!(
            rlmesh::binding::peer_info_override()
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
            rlmesh::binding::CURRENT_WORKFLOW_EDITION
        );
        assert_eq!(
            describe_json(first),
            after,
            "binding the second left the first alone"
        );
        assert_eq!(
            rlmesh::binding::peer_info_override().map(|info| info.os_version),
            Some("6.1".into()),
            "other override fields are kept"
        );
        for env in [first, second] {
            unsafe { rlmesh_ext_legacy_env_free(env) };
        }

        let mut out = RLMeshBytes::from_vec(Vec::new());
        let status = unsafe { rlmesh_ext_legacy_env_describe_json(std::ptr::null(), &mut out) };
        assert_eq!(status, RLMeshStatus::InvalidArgument);
    }

    #[test]
    fn the_c_export_describes_a_declared_catalog_and_refuses_a_malformed_one() {
        use crate::env::rlmesh_ext_legacy_env_free;

        let catalog = r#"[{"id": "viper/firm", "params": {"vehicle": "viper", "soil": "firm"},
                           "metadata": {"name": "Viper rover on firm soil"}},
                          {"id": "hmmwv/firm", "params": {"vehicle": "hmmwv"}}]"#;
        let env =
            native_env_with("Catalog-v0", 1.0, Some(catalog)).expect("rlmesh_ext_legacy_env_new");
        let value = describe_json(env);
        assert_eq!(value["variants"]["catalog"], parse(catalog));
        unsafe { rlmesh_ext_legacy_env_free(env) };

        for bad in [
            r#"{"id": "a"}"#,
            r#"[{"params": {}}]"#,
            r#"[{"id": "a"}, {"id": "a"}]"#,
            r#"[{"id": "a", "params": [1]}]"#,
            r#"[{"id": "a", "name": "typo"}]"#,
            "not json",
        ] {
            assert_eq!(
                native_env_with("Catalog-v0", 1.0, Some(bad)).err(),
                Some(RLMeshStatus::InvalidArgument),
                "{bad}"
            );
        }
    }

    #[test]
    fn c_config_runtime_is_copied_per_env_and_old_configs_keep_the_default() {
        use crate::env::rlmesh_ext_legacy_env_free;

        let cpp = native_env_runtime(
            "Cpp-v0",
            1.0,
            None,
            Some(
                r#"{"language":"c++","language_version":"17","toolchain":{"name":"gcc","version":"14.2"}}"#,
            ),
            false,
        )
        .expect("C++ runtime");
        let c = native_env_runtime(
            "C-v0",
            1.0,
            None,
            Some(
                r#"{"language":"c","language_version":"11","toolchain":{"name":"clang","version":"19.1"}}"#,
            ),
            false,
        )
        .expect("C runtime");
        // The field beyond the old config's allocation is ignored, even if the
        // new source config had a malformed declaration there.
        let old =
            native_env_runtime("Old-v0", 1.0, None, Some("not json"), true).expect("old config");
        let before = describe_json(cpp);
        assert_eq!(before["runtime"]["language"], "c++");
        assert_eq!(before["runtime"]["language_version"], "17");
        assert_eq!(
            before["runtime"]["toolchain"],
            json!({"name":"gcc","version":"14.2"})
        );
        bind(
            cpp,
            "127.0.0.1:0",
            rlmesh::binding::CURRENT_WORKFLOW_EDITION,
        )
        .expect("bind C++ env");
        assert_eq!(
            describe_json(cpp),
            before,
            "bind retained the copied runtime"
        );
        assert_eq!(describe_json(c)["runtime"]["language"], "c");
        assert_eq!(describe_json(c)["runtime"]["language_version"], "11");
        let legacy = describe_json(old);
        assert_eq!(legacy["runtime"]["language"], "c");
        assert!(legacy["runtime"].get("language_version").is_none());
        assert!(legacy["runtime"].get("framework_versions").is_none());
        for env in [cpp, c, old] {
            unsafe { rlmesh_ext_legacy_env_free(env) };
        }
    }

    #[test]
    fn c_config_refuses_malformed_host_runtime() {
        for bad in [
            "not json",
            "null",
            "[]",
            r#"{"language":42}"#,
            r#"{"language":" "}"#,
            r#"{"language_version":null}"#,
            r#"{"component":"host"}"#,
            r#"{"package_version":"1"}"#,
            r#"{"os":"host"}"#,
            r#"{"arch":"host"}"#,
            r#"{"protocol_generation":"host"}"#,
            r#"{"supported_workflow_editions":[]}"#,
            r#"{"preferred_workflow_edition":"host"}"#,
            r#"{"workflow_edition_error":"host"}"#,
            r#"{"toolchain":[]}"#,
            r#"{"toolchain":{}}"#,
            r#"{"toolchain":{"name":"gcc"}}"#,
            r#"{"toolchain":{"version":"16.2"}}"#,
            r#"{"toolchain":{"name":"gcc","version":"16.2","extra":"x"}}"#,
            r#"{"toolchain":{"name":42,"version":"16.2"}}"#,
            r#"{"toolchain":{"name":"gcc","version":42}}"#,
            r#"{"toolchain":{"name":" ","version":"16.2"}}"#,
            r#"{"toolchain":{"name":"gcc","version":" "}}"#,
            r#"{"toolchain":{"name":"unknown","version":"16.2"}}"#,
            r#"{"toolchain":{"name":"gcc","version":"16"}}"#,
            r#"{"toolchain":{"name":"gcc","version":"16.2.1"}}"#,
            r#"{"toolchain":{"name":"gcc","version":"16.x"}}"#,
            r#"{"toolchain":{"name":"gcc","version":".2"}}"#,
            r#"{"toolchain":{"name":"gcc","version":"16."}}"#,
            r#"{"framework_versions":{"torch":"1"," torch ":"2"}}"#,
            r#"{"framework_versions":[]}"#,
            r#"{"framework_versions":{"gcc":14}}"#,
            r#"{"framework_versions":{"":"1"}}"#,
            r#"{"framework_versions":{"gcc":""}}"#,
            r#"{"framework_versions":{"  ":"1"}}"#,
            r#"{"framework_versions":{"gcc":"  "}}"#,
        ] {
            assert_eq!(
                native_env_runtime("Bad-v0", 1.0, None, Some(bad), false).err(),
                Some(RLMeshStatus::InvalidArgument),
                "{bad}"
            );
        }
    }

    #[test]
    fn host_runtime_trims_all_declared_strings() {
        use crate::env::rlmesh_ext_legacy_env_free;

        let env = native_env_runtime("Trim-v0", 1.0, None, Some(
            r#"{"language":" c++ ","language_version":" 17 ","framework_versions":{" torch ":" 2.1 "},"toolchain":{"name":" gcc ","version":" 16.2 "}}"#
        ), false).expect("trimmed declaration");
        let runtime = &describe_json(env)["runtime"];
        assert_eq!(runtime["language"], "c++");
        assert_eq!(runtime["language_version"], "17");
        assert_eq!(runtime["framework_versions"], json!({"torch":"2.1"}));
        assert_eq!(runtime["toolchain"], json!({"name":"gcc","version":"16.2"}));
        unsafe { rlmesh_ext_legacy_env_free(env) };
    }

    #[test]
    fn c_config_refuses_invalid_utf8_and_oversized_declarations() {
        assert_eq!(
            native_env_runtime_bytes("Bad-v0", 1.0, None, Some(b"\xff"), false).err(),
            Some(RLMeshStatus::InvalidArgument)
        );
        let big = "x".repeat(4 * 1024 * 1024);
        let runtime = json!({"framework_versions":{"torch":big}}).to_string();
        assert_eq!(
            native_env_runtime("Big-v0", 1.0, None, Some(&runtime), false).err(),
            Some(RLMeshStatus::InvalidArgument)
        );
        let catalog = json!([{"id":"big","metadata":{"data":big}}]).to_string();
        assert_eq!(
            native_env_with("Big-v0", 1.0, Some(&catalog)).err(),
            Some(RLMeshStatus::InvalidArgument)
        );
    }

    #[test]
    fn a_contract_without_spaces_is_refused() {
        let mut bare = contract(None);
        bare.action_space = None;
        let Err(err) = envelope(&bare, None, None, None) else {
            panic!("a spaceless contract built an envelope");
        };
        assert!(err.message.contains("action_space"), "{}", err.message);
    }
}
