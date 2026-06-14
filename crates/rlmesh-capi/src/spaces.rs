//! Opaque `SpaceSpec` and `EnvContract` handles, plus the read accessors a C
//! model needs to find the spaces it must decode/encode against. Space *builders*
//! (the env-authoring side) are a separate, not-yet-implemented module.
#![allow(unsafe_code)] // FFI: raw pointers + repr(transparent) handles.

use rlmesh_spaces::{EnvContract, SpaceSpec};

use crate::abi::status::{CapiError, RlmeshStatus, guard, guard_value};
use crate::value::dtype::RlmeshDType;

/// An opaque environment contract (spaces, id, num_envs, autoreset).
#[repr(transparent)]
pub struct RlmeshContract(pub(crate) EnvContract);

impl RlmeshContract {
    /// # Safety
    /// `ptr` must be NULL or a valid `*const RlmeshContract` outliving `'a`.
    unsafe fn as_ref<'a>(ptr: *const Self) -> Option<&'a EnvContract> {
        unsafe { ptr.cast::<EnvContract>().as_ref() }
    }
}

/// An opaque space specification.
#[repr(transparent)]
pub struct RlmeshSpaceSpec(pub(crate) SpaceSpec);

/// Borrow the observation space (valid while `contract` lives), or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_contract_observation_space(
    contract: *const RlmeshContract,
) -> *const RlmeshSpaceSpec {
    space_ptr(contract, |contract| contract.observation_space.as_ref())
}

/// Borrow the action space (valid while `contract` lives), or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_contract_action_space(
    contract: *const RlmeshContract,
) -> *const RlmeshSpaceSpec {
    space_ptr(contract, |contract| contract.action_space.as_ref())
}

/// The contract's batch size (`num_envs`), or 0 if `contract` is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_contract_num_envs(contract: *const RlmeshContract) -> u32 {
    guard_value(0, || {
        unsafe { RlmeshContract::as_ref(contract) }.map_or(0, |contract| contract.num_envs)
    })
}

fn space_ptr(
    contract: *const RlmeshContract,
    pick: impl FnOnce(&EnvContract) -> Option<&SpaceSpec>,
) -> *const RlmeshSpaceSpec {
    guard_value(std::ptr::null(), || {
        match unsafe { RlmeshContract::as_ref(contract) } {
            Some(contract) => {
                pick(contract).map_or(std::ptr::null(), |spec| (spec as *const SpaceSpec).cast())
            }
            None => std::ptr::null(),
        }
    })
}

fn spec_ref<'a>(spec: *const RlmeshSpaceSpec) -> Option<&'a SpaceSpec> {
    unsafe { spec.cast::<SpaceSpec>().as_ref() }
}

/// The space kind as a `SpaceType` discriminant (Box=1 … Tuple=11), or 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_type(spec: *const RlmeshSpaceSpec) -> i32 {
    guard_value(0, || {
        spec_ref(spec).map_or(0, |spec| spec.space_type() as i32)
    })
}

/// The space's element dtype (`{0,0,0}` if unset/unsupported).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_dtype(spec: *const RlmeshSpaceSpec) -> RlmeshDType {
    guard_value(
        RlmeshDType {
            code: 0,
            bits: 0,
            lanes: 0,
        },
        || {
            spec_ref(spec)
                .and_then(|spec| RlmeshDType::from_core(spec.dtype))
                .unwrap_or(RlmeshDType {
                    code: 0,
                    bits: 0,
                    lanes: 0,
                })
        },
    )
}

/// The space's rank (number of dimensions), or 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_ndim(spec: *const RlmeshSpaceSpec) -> usize {
    guard_value(0, || spec_ref(spec).map_or(0, |spec| spec.shape.len()))
}

/// Copy the space's shape into `out` (capacity `cap`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_copy_shape(
    spec: *const RlmeshSpaceSpec,
    out: *mut i64,
    cap: usize,
) -> RlmeshStatus {
    guard(|| {
        let spec = spec_ref(spec).ok_or_else(|| CapiError::invalid_arg("null space spec"))?;
        if spec.shape.len() > cap {
            return Err(CapiError::invalid_arg("shape buffer too small"));
        }
        if spec.shape.is_empty() {
            return Ok(());
        }
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(out, spec.shape.len()) };
        dst.copy_from_slice(&spec.shape);
        Ok(())
    })
}
