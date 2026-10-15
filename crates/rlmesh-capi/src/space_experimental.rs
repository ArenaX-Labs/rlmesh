//! Experimental space constructors. Unlike the removed ABI 4 helpers these
//! return Space handles and share their sampler on clone.
#![allow(unsafe_code)]
use crate::abi::status::{CapiError, RLMeshStatus, guard};
use crate::space_handle::{RLMeshSpace, RLMeshStr, write_space};
use crate::spaces::{RLMeshSpaceSpec, slice_arg, spec_ref};
use crate::value::handle::write_out;
use rlmesh_spaces::spaces::{
    MultiBinaryBuilder, MultiDiscreteBuilder, TextBuilder, TupleSpaceBuilder,
};
use rlmesh_spaces::{SpaceKind, SpaceSpec};

fn built(
    spec: Result<SpaceSpec, rlmesh_spaces::errors::SpaceError>,
    out: *mut *mut RLMeshSpace,
) -> Result<(), CapiError> {
    let spec = spec.map_err(|e| CapiError::invalid_arg(e.to_string()))?;
    write_space(
        out,
        rlmesh::spaces::Space::new(rlmesh::binding::space_spec_from_wire(spec)),
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_multi_binary_new(
    shape: *const i64,
    n: usize,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        built(
            MultiBinaryBuilder::shape(unsafe { slice_arg(shape, n, "shape") }?.to_vec()).build(),
            out,
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_multi_discrete_new(
    nvec: *const i64,
    n: usize,
    dtype: i32,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let dtype = crate::value::dtype::dtype_from_c(dtype)
            .ok_or_else(|| CapiError::invalid_arg("invalid dtype"))?;
        let dtype = rlmesh_spaces::DType::try_from(dtype as i32)
            .map_err(|_| CapiError::invalid_arg("invalid dtype"))?;
        built(
            MultiDiscreteBuilder::vector(unsafe { slice_arg(nvec, n, "nvec") }?.to_vec())
                .dtype(dtype)
                .build(),
            out,
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_text_new(
    max_length: i64,
    min_length: i64,
    charset: *const std::ffi::c_char,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let mut b = TextBuilder::new(max_length).min_length(min_length);
        if let Some(s) = unsafe { crate::ffi::text(charset) }? {
            b = b.charset(s);
        }
        built(b.build(), out)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tuple_new(
    children: *const *const RLMeshSpaceSpec,
    n: usize,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let children = unsafe { slice_arg(children, n, "children") }?
            .iter()
            .map(|&s| {
                spec_ref(s)
                    .cloned()
                    .ok_or_else(|| CapiError::invalid_arg("null child"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        built(TupleSpaceBuilder::new().extend(children).build(), out)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_multi_discrete_nvec(
    space: *const RLMeshSpace,
    out: *mut i64,
    cap: usize,
    len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        let spec = spec_ref(crate::space_handle::rlmesh_space_spec(space))
            .ok_or_else(|| CapiError::invalid_arg("null space"))?;
        let Some(SpaceKind::MultiDiscrete(v)) = &spec.spec else {
            return Err(CapiError::type_error("not MultiDiscrete"));
        };
        write_out(len, v.nvec.len())?;
        if out.is_null() && cap == 0 {
            return Ok(());
        }
        if cap < v.nvec.len() || out.is_null() {
            return Err(CapiError::invalid_arg("short nvec buffer"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(v.nvec.as_ptr(), out, v.nvec.len());
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_text_limits(
    space: *const RLMeshSpace,
    min: *mut i64,
    max: *mut i64,
    charset: *mut RLMeshStr,
) -> RLMeshStatus {
    guard(|| {
        let spec = spec_ref(crate::space_handle::rlmesh_space_spec(space))
            .ok_or_else(|| CapiError::invalid_arg("null space"))?;
        let Some(SpaceKind::Text(v)) = &spec.spec else {
            return Err(CapiError::type_error("not Text"));
        };
        write_out(min, v.min_length)?;
        write_out(max, v.max_length)?;
        write_out(charset, RLMeshStr::borrowed(&v.charset))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tuple_len(
    space: *const RLMeshSpace,
    out: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        let spec = spec_ref(crate::space_handle::rlmesh_space_spec(space))
            .ok_or_else(|| CapiError::invalid_arg("null space"))?;
        let Some(SpaceKind::Tuple(v)) = &spec.spec else {
            return Err(CapiError::type_error("not Tuple"));
        };
        write_out(out, v.spaces.len())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tuple_get(
    space: *const RLMeshSpace,
    index: usize,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let spec = spec_ref(crate::space_handle::rlmesh_space_spec(space))
            .ok_or_else(|| CapiError::invalid_arg("null space"))?;
        let Some(SpaceKind::Tuple(v)) = &spec.spec else {
            return Err(CapiError::type_error("not Tuple"));
        };
        let child = v
            .spaces
            .get(index)
            .ok_or_else(|| CapiError::invalid_arg("tuple index out of range"))?;
        write_space(
            out,
            rlmesh::spaces::Space::new(rlmesh::binding::space_spec_from_wire(child.clone())),
        )
    })
}
