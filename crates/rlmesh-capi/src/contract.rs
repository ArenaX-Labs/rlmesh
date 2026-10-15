//! The immutable EnvContract handle and typed reset-option helper.
#![allow(unsafe_code)] // FFI: owned handles, out parameters and borrowed spec pointers.

use crate::abi::status::{CapiError, RLMeshStatus, guard, guard_ptr, guard_value};
use crate::codec::RLMeshBytes;
use crate::spaces::RLMeshSpaceSpec;
use crate::value::handle::{RLMeshValue, into_handle, value_ref, write_out};
use rlmesh::{EnvContract, Value};
use std::sync::Arc;

#[derive(Clone)]
pub struct RLMeshEnvContract(Arc<Contract>);
struct Contract {
    wire: rlmesh_spaces::EnvContract,
    facade: rlmesh::Result<EnvContract>,
    metadata: Option<RLMeshValue>,
}
impl RLMeshEnvContract {
    pub(crate) fn new(wire: rlmesh_spaces::EnvContract) -> Self {
        let facade = rlmesh::binding::env_contract_from_wire(&wire);
        let metadata = wire
            .metadata
            .clone()
            .map(|m| RLMeshValue::new(Value::Dict(rlmesh::binding::metadata_from_wire(m))));
        Self(Arc::new(Contract {
            wire,
            facade,
            metadata,
        }))
    }
    /// A legacy handler can receive a contract with missing spaces. Keep its
    /// wire representation for adapter resolution; shared methods requiring
    /// the full contract return the facade's ProtocolException for that case.
    pub(crate) unsafe fn as_ref<'a>(ptr: *const Self) -> Option<&'a rlmesh_spaces::EnvContract> {
        unsafe { ptr.as_ref() }.map(|c| &c.0.wire)
    }
    fn facade(&self) -> Result<&EnvContract, CapiError> {
        self.0
            .facade
            .as_ref()
            .map_err(|e| CapiError::from(e.clone()))
    }
}
fn require<'a>(contract: *const RLMeshEnvContract) -> Result<&'a RLMeshEnvContract, CapiError> {
    unsafe { contract.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null contract"))
}
fn bytes(out: *mut RLMeshBytes, text: &str) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    write_out(out, RLMeshBytes::from_vec(text.as_bytes().to_vec()))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_clone(
    contract: *const RLMeshEnvContract,
) -> *mut RLMeshEnvContract {
    guard_ptr(|| Ok(Box::into_raw(Box::new(require(contract)?.clone()))))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_env(
    contract: *const RLMeshEnvContract,
) -> *mut RLMeshEnvContract {
    rlmesh_env_contract_clone(contract)
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_spec(
    contract: *const RLMeshEnvContract,
) -> *mut RLMeshEnvContract {
    rlmesh_env_contract_clone(contract)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_contract_free(contract: *mut RLMeshEnvContract) {
    guard_value((), || {
        if !contract.is_null() {
            drop(unsafe { Box::from_raw(contract) });
        }
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_id(
    contract: *const RLMeshEnvContract,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| bytes(out, require(contract)?.facade()?.id()))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_observation_space(
    contract: *const RLMeshEnvContract,
) -> *const RLMeshSpaceSpec {
    guard_value(std::ptr::null(), || {
        unsafe { RLMeshEnvContract::as_ref(contract) }
            .and_then(|c| c.observation_space.as_ref())
            .map_or(std::ptr::null(), |s| std::ptr::from_ref(s).cast())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_action_space(
    contract: *const RLMeshEnvContract,
) -> *const RLMeshSpaceSpec {
    guard_value(std::ptr::null(), || {
        unsafe { RLMeshEnvContract::as_ref(contract) }
            .and_then(|c| c.action_space.as_ref())
            .map_or(std::ptr::null(), |s| std::ptr::from_ref(s).cast())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_num_envs(contract: *const RLMeshEnvContract) -> u64 {
    guard_value(0, || {
        unsafe { RLMeshEnvContract::as_ref(contract) }.map_or(0, |c| u64::from(c.num_envs))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_render_mode(
    contract: *const RLMeshEnvContract,
    out_has: *mut bool,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        if out_has.is_null() || out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let mode = require(contract)?.facade()?.render_mode();
        write_out(out_has, mode.is_some())?;
        bytes(out, mode.unwrap_or_default())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_metadata(
    contract: *const RLMeshEnvContract,
) -> *const RLMeshValue {
    guard_value(std::ptr::null(), || {
        require(contract)
            .ok()
            .and_then(|c| c.0.metadata.as_ref())
            .map_or(std::ptr::null(), std::ptr::from_ref)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_to_dict(
    contract: *const RLMeshEnvContract,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        write_out(
            out,
            into_handle(Value::Dict(require(contract)?.facade()?.to_dict())),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_contract_repr(
    contract: *const RLMeshEnvContract,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| bytes(out, &require(contract)?.facade()?.to_string()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_trial_index(
    options: *const RLMeshValue,
    lane: *const usize,
    out_has: *mut bool,
    out: *mut i64,
) -> RLMeshStatus {
    guard(|| {
        if out_has.is_null() || out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let options = if options.is_null() {
            None
        } else {
            Some(value_ref(options)?.as_dict()?)
        };
        let index = rlmesh::trial_index(options, unsafe { lane.as_ref() }.copied())?;
        write_out(out_has, index.is_some())?;
        write_out(out, index.unwrap_or_default())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::handle::*;
    use RLMeshStatus as S;
    use rlmesh_spaces::spaces::DiscreteBuilder;
    use rlmesh_spaces::{EnvContract as WireContract, MetaValue};
    use std::ptr::{null, null_mut};
    fn contract() -> RLMeshEnvContract {
        let space = DiscreteBuilder::new(4).build().unwrap();
        RLMeshEnvContract::new(WireContract {
            id: "C-v1".into(),
            observation_space: Some(space.clone()),
            action_space: Some(space),
            render_mode: "rgb_array".into(),
            num_envs: 2,
            metadata: Some([("blob".into(), MetaValue::Bytes(vec![0, 1, 2]))].into()),
            ..Default::default()
        })
    }
    #[test]
    fn contract_reads_use_facade_and_clones_retain_borrowed_children() {
        let c = contract();
        let cloned = rlmesh_env_contract_clone(&c);
        let obs = rlmesh_env_contract_observation_space(&c);
        let metadata = rlmesh_env_contract_metadata(&c);
        assert_eq!(rlmesh_value_kind(metadata), RLMeshValueKind::Dict);
        let mut dict = null_mut();
        assert_eq!(rlmesh_env_contract_to_dict(&c, &mut dict), S::Ok);
        assert_eq!(
            value_ref(dict).ok().unwrap(),
            &Value::Dict(c.facade().ok().unwrap().to_dict())
        );
        let mut out = RLMeshBytes::from_vec(vec![]);
        let mut has = false;
        assert_eq!(
            rlmesh_env_contract_render_mode(&c, &mut has, &mut out),
            S::Ok
        );
        assert!(has);
        assert_eq!(
            unsafe { std::mem::replace(&mut out, RLMeshBytes::from_vec(vec![])).into_vec() },
            b"rgb_array"
        );
        drop(c);
        assert_eq!(rlmesh_env_contract_num_envs(cloned), 2);
        assert_eq!(obs, rlmesh_env_contract_observation_space(cloned));
        assert_eq!(metadata, rlmesh_env_contract_metadata(cloned));
        assert_eq!(rlmesh_env_contract_id(cloned, &mut out), S::Ok);
        assert_eq!(
            unsafe { std::mem::replace(&mut out, RLMeshBytes::from_vec(vec![])).into_vec() },
            b"C-v1"
        );
        assert_eq!(rlmesh_env_contract_repr(cloned, &mut out), S::Ok);
        assert!(
            unsafe { std::mem::replace(&mut out, RLMeshBytes::from_vec(vec![])).into_vec() }
                .starts_with(b"EnvContract(")
        );
        let env = rlmesh_env_contract_env(cloned);
        let spec = rlmesh_env_contract_spec(cloned);
        unsafe {
            for c in [cloned, env, spec] {
                rlmesh_env_contract_free(c);
            }
            rlmesh_value_free(dict);
        }
        assert_eq!(rlmesh_env_contract_num_envs(null()), 0);
    }
    #[test]
    fn optional_contract_fields_and_missing_wire_spaces() {
        let mut c = contract();
        let mut wire = c.0.wire.clone();
        wire.metadata = None;
        wire.render_mode.clear();
        c = RLMeshEnvContract::new(wire);
        assert!(rlmesh_env_contract_metadata(&c).is_null());
        let mut out = RLMeshBytes::from_vec(vec![]);
        let mut has = true;
        assert_eq!(
            rlmesh_env_contract_render_mode(&c, &mut has, &mut out),
            S::Ok
        );
        assert!(!has);
        assert_eq!(out.len, 0);
        unsafe {
            crate::codec::rlmesh_bytes_free(out);
        }
        let c = RLMeshEnvContract::new(WireContract::default());
        let mut dict = null_mut();
        assert_eq!(rlmesh_env_contract_to_dict(&c, &mut dict), S::Protocol);
        assert!(dict.is_null());
        assert_eq!(
            rlmesh_env_contract_id(null(), null_mut()),
            S::InvalidArgument
        );
    }
    #[test]
    fn trial_index_uses_typed_metadata_and_python_validation() {
        let mut has = true;
        let mut out = 99;
        assert_eq!(
            unsafe { rlmesh_trial_index(null(), null(), &mut has, &mut out) },
            S::Ok
        );
        assert!(!has);
        let map = |v| into_handle(Value::Dict([("trial_index", v)].into_iter().collect()));
        for (v, expected) in [
            (Value::Int(-3), Some(-3)),
            (Value::Float(4.0), Some(4)),
            (Value::Bool(true), None),
            (Value::Text("1".into()), None),
            (Value::Float(1.5), None),
        ] {
            let o = map(v);
            assert_eq!(
                unsafe { rlmesh_trial_index(o, null(), &mut has, &mut out) },
                S::Ok
            );
            assert_eq!(has, expected.is_some());
            if let Some(n) = expected {
                assert_eq!(out, n);
            }
            unsafe {
                rlmesh_value_free(o);
            }
        }
        let o = map(Value::List(vec![Value::Int(7), Value::Int(8)]));
        let lane = 1;
        assert_eq!(
            unsafe { rlmesh_trial_index(o, &lane, &mut has, &mut out) },
            S::Ok
        );
        assert!(has);
        assert_eq!(out, 8);
        assert_eq!(
            unsafe { rlmesh_trial_index(o, null(), &mut has, &mut out) },
            S::InvalidArgument
        );
        assert_eq!(
            unsafe { rlmesh_trial_index(o, &2, &mut has, &mut out) },
            S::InvalidArgument
        );
        assert_eq!(
            unsafe { rlmesh_trial_index(o, &lane, null_mut(), &mut out) },
            S::InvalidArgument
        );
        unsafe {
            rlmesh_value_free(o);
        }
        let o = into_handle(Value::Bool(false));
        assert_eq!(
            unsafe { rlmesh_trial_index(o, null(), &mut has, &mut out) },
            S::Type
        );
        unsafe {
            rlmesh_value_free(o);
        }
    }
}
