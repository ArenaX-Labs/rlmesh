//! Tag-driven IO adapter resolution over the C ABI. A model worker resolves
//! the env's published `EnvTags` against its own `ModelSpec` into an opaque,
//! immutable plan handle. Specs cross as JSON (the frozen v1 wire format is the
//! contract; the capi never mirrors the ~30 nested spec structs); the env's
//! observation/action spaces cross as the `RlmeshSpaceSpec` handles the contract
//! already exposes. Per-step apply (`transform_obs`/`transform_action`) is a
//! separate, not-yet-implemented surface.
#![allow(unsafe_code)] // FFI: raw pointers + C string in / owned buffer out.

use std::collections::BTreeSet;
use std::ffi::{CStr, c_char};

use rlmesh_adapters::v1::{
    ENV_METADATA_KEY, EnvTags, ModelSpec, ResolvedAdapter, SpaceView, resolve,
};
use rlmesh_spaces::MetaValue;

use crate::abi::status::{CapiError, RlmeshStatus, guard};
use crate::codec::RlmeshBytes;
use crate::spaces::{RlmeshContract, RlmeshSpaceSpec, spec_ref};

/// An opaque resolved adapter plan. Immutable; free with
/// `rlmesh_adapter_plan_free`.
#[repr(transparent)]
pub struct RlmeshAdapterPlan(ResolvedAdapter);

unsafe fn cstr<'a>(ptr: *const c_char, what: &str) -> Result<&'a str, CapiError> {
    if ptr.is_null() {
        return Err(CapiError::invalid_arg(format!("null {what}")));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| CapiError::invalid_arg(format!("{what} is not valid UTF-8")))
}

fn plan_ref<'a>(plan: *const RlmeshAdapterPlan) -> Result<&'a ResolvedAdapter, CapiError> {
    unsafe { plan.cast::<ResolvedAdapter>().as_ref() }
        .ok_or_else(|| CapiError::invalid_arg("null adapter plan"))
}

fn write_bytes(out: *mut RlmeshBytes, bytes: Vec<u8>) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    unsafe { *out = RlmeshBytes::from_vec(bytes) };
    Ok(())
}

/// The top-level observation entry a (possibly dotted) plan key lives under.
/// `"."` is the reserved flat/root observation, its own top-level key.
fn top_level_key(key: &str) -> &str {
    if key == "." {
        return ".";
    }
    key.split('.').next().unwrap_or(key)
}

/// Resolve env tags + spaces and a model spec into a plan handle.
///
/// `env_tags_json` is the env's `EnvTags` as JSON (see
/// `rlmesh_contract_adapter_tags_json`); `model_spec_json` is the model's
/// `ModelSpec`. The spaces are borrowed `RlmeshSpaceSpec` handles (e.g. from the
/// contract); they are not retained past the call. On success `*out_plan` owns a
/// plan freed with `rlmesh_adapter_plan_free`.
///
/// # Safety
/// `out_plan` must be a valid writable pointer; the string and space pointers,
/// when non-null, must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_resolve(
    env_tags_json: *const c_char,
    observation_space: *const RlmeshSpaceSpec,
    action_space: *const RlmeshSpaceSpec,
    model_spec_json: *const c_char,
    trust_entrypoints: bool,
    out_plan: *mut *mut RlmeshAdapterPlan,
) -> RlmeshStatus {
    guard(|| {
        if out_plan.is_null() {
            return Err(CapiError::invalid_arg("null out_plan"));
        }
        unsafe { *out_plan = std::ptr::null_mut() };
        let tags_json = unsafe { cstr(env_tags_json, "env_tags_json") }?;
        let spec_json = unsafe { cstr(model_spec_json, "model_spec_json") }?;
        let obs = spec_ref(observation_space)
            .ok_or_else(|| CapiError::invalid_arg("null observation_space"))?;
        let act =
            spec_ref(action_space).ok_or_else(|| CapiError::invalid_arg("null action_space"))?;
        let tags: EnvTags = serde_json::from_str(tags_json)
            .map_err(|err| CapiError::invalid_arg(format!("invalid env tags: {err}")))?;
        let model_spec: ModelSpec = serde_json::from_str(spec_json)
            .map_err(|err| CapiError::invalid_arg(format!("invalid model spec: {err}")))?;
        let adapter = resolve(
            &tags,
            &SpaceView::from(obs),
            &SpaceView::from(act),
            &model_spec,
            trust_entrypoints,
        )
        .map_err(|err| CapiError::invalid_value(err.message))?;
        unsafe { *out_plan = Box::into_raw(Box::new(RlmeshAdapterPlan(adapter))) };
        Ok(())
    })
}

/// Free a plan from `rlmesh_adapter_resolve`.
///
/// # Safety
/// `plan` must be a plan this thread owns and has not freed (NULL is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_plan_free(plan: *mut RlmeshAdapterPlan) {
    if !plan.is_null() {
        drop(unsafe { Box::from_raw(plan) });
    }
}

/// Write a human-readable summary of the plan to `out` (UTF-8, free with
/// `rlmesh_bytes_free`).
///
/// # Safety
/// `plan` and `out` must be valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_plan_describe(
    plan: *const RlmeshAdapterPlan,
    out: *mut RlmeshBytes,
) -> RlmeshStatus {
    guard(|| write_bytes(out, plan_ref(plan)?.describe().into_bytes()))
}

/// Write the top-level observation keys the plan reads as a JSON array of
/// strings to `out` (free with `rlmesh_bytes_free`). A host applying the plan
/// should encode only these keys.
///
/// # Safety
/// `plan` and `out` must be valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_plan_referenced_obs_keys(
    plan: *const RlmeshAdapterPlan,
    out: *mut RlmeshBytes,
) -> RlmeshStatus {
    guard(|| {
        let referenced = plan_ref(plan)?.referenced_obs_keys();
        let keys: BTreeSet<&str> = referenced.iter().map(|key| top_level_key(key)).collect();
        let json = serde_json::to_string(&keys)
            .map_err(|err| CapiError::internal(format!("serialize keys: {err}")))?;
        write_bytes(out, json.into_bytes())
    })
}

/// Write the env's adapter `EnvTags` (from the contract metadata) as JSON to
/// `out`, ready to pass to `rlmesh_adapter_resolve` (free with
/// `rlmesh_bytes_free`). Writes an empty buffer (status OK) when the env was
/// served without tags — the caller treats empty as "untagged env".
///
/// # Safety
/// `contract` and `out` must be valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_contract_adapter_tags_json(
    contract: *const RlmeshContract,
    out: *mut RlmeshBytes,
) -> RlmeshStatus {
    guard(|| {
        let contract = unsafe { RlmeshContract::as_ref(contract) }
            .ok_or_else(|| CapiError::invalid_arg("null contract"))?;
        let json = match contract
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get(ENV_METADATA_KEY))
        {
            Some(tags) => serde_json::to_string(&meta_to_json(tags))
                .map_err(|err| CapiError::internal(format!("serialize tags: {err}")))?,
            None => String::new(),
        };
        write_bytes(out, json.into_bytes())
    })
}

/// Faithfully render contract metadata as JSON (used by the tags accessor).
pub(crate) fn meta_to_json(value: &MetaValue) -> serde_json::Value {
    use serde_json::Value as Json;
    match value {
        MetaValue::Null => Json::Null,
        MetaValue::Bool(value) => Json::Bool(*value),
        MetaValue::Int(value) => Json::Number((*value).into()),
        MetaValue::Float(value) => {
            serde_json::Number::from_f64(*value).map_or(Json::Null, Json::Number)
        }
        MetaValue::String(value) => Json::String(value.clone()),
        MetaValue::Bytes(value) => {
            Json::Array(value.iter().map(|&b| Json::Number(b.into())).collect())
        }
        MetaValue::List(items) => Json::Array(items.iter().map(meta_to_json).collect()),
        MetaValue::Map(map) => Json::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), meta_to_json(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use rlmesh_spaces::{
        BoxSpec, DType, DictSpec, EnvContract, MetaMap, SpaceKind, SpaceSpec, TextSpec,
    };

    use super::*;

    const BASIC_PAIRING: &str =
        include_str!("../../rlmesh-adapters/conformance/v1/cases/resolve_basic_pairing.json");

    fn box_spec(shape: Vec<i64>, dtype: DType) -> SpaceSpec {
        SpaceSpec {
            shape,
            dtype,
            spec: Some(SpaceKind::Box(BoxSpec { bounds: None })),
        }
    }

    // Inverse of meta_to_json, to stage env tags into a contract as the env side
    // would have published them (EnvTags.to_metadata stores to_dict() verbatim).
    fn json_to_meta(value: &serde_json::Value) -> MetaValue {
        use serde_json::Value as Json;
        match value {
            Json::Null => MetaValue::Null,
            Json::Bool(value) => MetaValue::Bool(*value),
            Json::Number(number) => number.as_i64().map_or_else(
                || MetaValue::Float(number.as_f64().unwrap()),
                MetaValue::Int,
            ),
            Json::String(value) => MetaValue::String(value.clone()),
            Json::Array(items) => MetaValue::List(items.iter().map(json_to_meta).collect()),
            Json::Object(map) => MetaValue::Map(
                map.iter()
                    .map(|(k, v)| (k.clone(), json_to_meta(v)))
                    .collect(),
            ),
        }
    }

    // The observation/action SpaceSpecs whose SpaceView matches the
    // resolve_basic_pairing vector (Dict of cam/eef_pos/eef_quat/gripper/text;
    // Box[7] action; no declared bounds).
    fn basic_obs_space() -> SpaceSpec {
        SpaceSpec {
            shape: vec![],
            dtype: DType::Unspecified,
            spec: Some(SpaceKind::Dict(DictSpec {
                keys: ["cam", "eef_pos", "eef_quat", "gripper", "instruction"]
                    .map(String::from)
                    .to_vec(),
                spaces: vec![
                    box_spec(vec![8, 8, 3], DType::Uint8),
                    box_spec(vec![3], DType::Float32),
                    box_spec(vec![4], DType::Float32),
                    box_spec(vec![2], DType::Float32),
                    SpaceSpec {
                        shape: vec![],
                        dtype: DType::Unspecified,
                        spec: Some(SpaceKind::Text(TextSpec::default())),
                    },
                ],
            })),
        }
    }

    fn read_bytes(bytes: &RlmeshBytes) -> String {
        if bytes.data.is_null() {
            return String::new();
        }
        let slice = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) };
        String::from_utf8(slice.to_vec()).expect("utf-8")
    }

    #[test]
    fn resolve_basic_pairing_matches_conformance_vector() {
        let case: serde_json::Value = serde_json::from_str(BASIC_PAIRING).unwrap();
        let tags = CString::new(serde_json::to_string(&case["env_tags"]).unwrap()).unwrap();
        let model = CString::new(serde_json::to_string(&case["model_spec"]).unwrap()).unwrap();
        let expected = case["expect"]["describe"].as_str().unwrap();

        let obs = RlmeshSpaceSpec(basic_obs_space());
        let act = RlmeshSpaceSpec(box_spec(vec![7], DType::Float32));
        let mut plan: *mut RlmeshAdapterPlan = std::ptr::null_mut();

        let status = unsafe {
            rlmesh_adapter_resolve(tags.as_ptr(), &obs, &act, model.as_ptr(), false, &mut plan)
        };
        assert_eq!(status, RlmeshStatus::Ok);
        assert!(!plan.is_null());

        let mut out = RlmeshBytes {
            data: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            unsafe { rlmesh_adapter_plan_describe(plan, &mut out) },
            RlmeshStatus::Ok
        );
        assert_eq!(read_bytes(&out), expected);
        unsafe { crate::codec::rlmesh_bytes_free(out) };

        let mut keys = RlmeshBytes {
            data: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            unsafe { rlmesh_adapter_plan_referenced_obs_keys(plan, &mut keys) },
            RlmeshStatus::Ok
        );
        let parsed: BTreeSet<String> = serde_json::from_str(&read_bytes(&keys)).unwrap();
        assert!(
            parsed.contains("cam") && parsed.contains("eef_pos") && parsed.contains("instruction")
        );
        unsafe { crate::codec::rlmesh_bytes_free(keys) };

        unsafe { rlmesh_adapter_plan_free(plan) };
    }

    #[test]
    fn resolve_rejects_invalid_model_spec_json() {
        let obs = RlmeshSpaceSpec(basic_obs_space());
        let act = RlmeshSpaceSpec(box_spec(vec![7], DType::Float32));
        let tags =
            CString::new(r#"{"observation":{},"action":{"clip":null,"components":[]}}"#).unwrap();
        let bad = CString::new("{ not json").unwrap();
        let mut plan: *mut RlmeshAdapterPlan = std::ptr::null_mut();
        let status = unsafe {
            rlmesh_adapter_resolve(tags.as_ptr(), &obs, &act, bad.as_ptr(), false, &mut plan)
        };
        assert_eq!(status, RlmeshStatus::InvalidArgument);
        assert!(plan.is_null());
    }

    #[test]
    fn resolve_rejects_null_out_and_spaces() {
        let tags = CString::new("{}").unwrap();
        let model = CString::new("{}").unwrap();
        // null out_plan
        assert_eq!(
            unsafe {
                rlmesh_adapter_resolve(
                    tags.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    model.as_ptr(),
                    false,
                    std::ptr::null_mut(),
                )
            },
            RlmeshStatus::InvalidArgument
        );
        // null spaces
        let mut plan: *mut RlmeshAdapterPlan = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                rlmesh_adapter_resolve(
                    tags.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    model.as_ptr(),
                    false,
                    &mut plan,
                )
            },
            RlmeshStatus::InvalidArgument
        );
        assert!(plan.is_null());
    }

    #[test]
    fn contract_tags_json_roundtrips_through_resolve() {
        let case: serde_json::Value = serde_json::from_str(BASIC_PAIRING).unwrap();
        let mut metadata = MetaMap::new();
        metadata.insert(ENV_METADATA_KEY.to_owned(), json_to_meta(&case["env_tags"]));
        let contract = RlmeshContract(EnvContract {
            metadata: Some(metadata),
            ..Default::default()
        });

        let mut out = RlmeshBytes {
            data: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            unsafe { rlmesh_contract_adapter_tags_json(&contract, &mut out) },
            RlmeshStatus::Ok
        );
        let tags_json = read_bytes(&out);
        unsafe { crate::codec::rlmesh_bytes_free(out) };
        // The accessor's JSON must be exactly what resolve consumes.
        serde_json::from_str::<EnvTags>(&tags_json).expect("tags json parses as EnvTags");

        let tags = CString::new(tags_json).unwrap();
        let model = CString::new(serde_json::to_string(&case["model_spec"]).unwrap()).unwrap();
        let obs = RlmeshSpaceSpec(basic_obs_space());
        let act = RlmeshSpaceSpec(box_spec(vec![7], DType::Float32));
        let mut plan: *mut RlmeshAdapterPlan = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                rlmesh_adapter_resolve(tags.as_ptr(), &obs, &act, model.as_ptr(), false, &mut plan)
            },
            RlmeshStatus::Ok
        );
        assert!(!plan.is_null());
        unsafe { rlmesh_adapter_plan_free(plan) };
    }

    #[test]
    fn contract_tags_json_empty_when_untagged() {
        let contract = RlmeshContract(EnvContract::default());
        let mut out = RlmeshBytes {
            data: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            unsafe { rlmesh_contract_adapter_tags_json(&contract, &mut out) },
            RlmeshStatus::Ok
        );
        assert!(out.data.is_null() && out.len == 0);
        unsafe { crate::codec::rlmesh_bytes_free(out) };
    }
}
