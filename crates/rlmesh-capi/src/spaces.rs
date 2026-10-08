//! Opaque `SpaceSpec` and `EnvContract` handles: the read accessors a C model
//! needs to find the spaces it must decode/encode against, and the builders a C
//! environment declares its spaces with.
#![allow(unsafe_code)] // FFI: raw pointers + repr(transparent) handles.

use std::ffi::{CStr, c_char};

use rlmesh_spaces::errors::SpaceError;
use rlmesh_spaces::spaces::{
    BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder, MultiBinaryBuilder, MultiDiscreteBuilder,
    TextBuilder, TupleSpaceBuilder,
};
use rlmesh_spaces::{
    BoxBounds, DType, EnvContract, SpaceKind, SpaceSpec, SpaceType, decode_scalars,
};

use crate::abi::status::{CapiError, RlmeshStatus, guard, guard_ptr, guard_value};
use crate::value::dtype::RlmeshDType;
use crate::value::handle::{RlmeshValueKind, write_out};

/// An opaque environment contract (spaces, id, num_envs, autoreset).
#[repr(transparent)]
pub struct RlmeshContract(pub(crate) EnvContract);

impl RlmeshContract {
    /// # Safety
    /// `ptr` must be NULL or a valid `*const RlmeshContract` outliving `'a`.
    pub(crate) unsafe fn as_ref<'a>(ptr: *const Self) -> Option<&'a EnvContract> {
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

pub(crate) fn spec_ref<'a>(spec: *const RlmeshSpaceSpec) -> Option<&'a SpaceSpec> {
    unsafe { spec.cast::<SpaceSpec>().as_ref() }
}

/// The space kind, or `Invalid` when `spec` is NULL or its kind is unspecified.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_type(spec: *const RlmeshSpaceSpec) -> RlmeshValueKind {
    guard_value(RlmeshValueKind::Invalid, || {
        spec_ref(spec).map_or(RlmeshValueKind::Invalid, |spec| match spec.space_type() {
            SpaceType::Box => RlmeshValueKind::Box,
            SpaceType::Discrete => RlmeshValueKind::Discrete,
            SpaceType::MultiBinary => RlmeshValueKind::MultiBinary,
            SpaceType::MultiDiscrete => RlmeshValueKind::MultiDiscrete,
            SpaceType::Text => RlmeshValueKind::Text,
            SpaceType::Dict => RlmeshValueKind::Dict,
            SpaceType::Tuple => RlmeshValueKind::Tuple,
            SpaceType::Unspecified => RlmeshValueKind::Invalid,
        })
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

/// Write the child count of a `Dict`/`Tuple` space to `out`. Any other kind is
/// `RLMESH_ERR_INVALID_VALUE`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_len(
    spec: *const RlmeshSpaceSpec,
    out: *mut usize,
) -> RlmeshStatus {
    guard(|| {
        let len = match &spec_ref(spec).ok_or_else(null_spec)?.spec {
            Some(SpaceKind::Dict(dict)) => dict.spaces.len(),
            Some(SpaceKind::Tuple(tuple)) => tuple.spaces.len(),
            _ => return Err(CapiError::invalid_value("space is not a Dict or Tuple")),
        };
        write_out(out, len)
    })
}

/// Borrow a `Tuple` space's child by index (valid while `spec` lives), or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_tuple_get(
    spec: *const RlmeshSpaceSpec,
    index: usize,
) -> *const RlmeshSpaceSpec {
    guard_value(std::ptr::null(), || {
        match spec_ref(spec).map(|spec| &spec.spec) {
            Some(Some(SpaceKind::Tuple(tuple))) => child_ptr(tuple.spaces.get(index)),
            _ => std::ptr::null(),
        }
    })
}

/// Borrow a `Dict` space's child by key (valid while `spec` lives), or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_dict_get(
    spec: *const RlmeshSpaceSpec,
    key: *const c_char,
) -> *const RlmeshSpaceSpec {
    guard_value(std::ptr::null(), || {
        if key.is_null() {
            return std::ptr::null();
        }
        let Some(Some(SpaceKind::Dict(dict))) = spec_ref(spec).map(|spec| &spec.spec) else {
            return std::ptr::null();
        };
        // SAFETY: `key` is non-NULL and, per the ABI contract, NUL-terminated.
        let Ok(key) = (unsafe { CStr::from_ptr(key) }).to_str() else {
            return std::ptr::null();
        };
        let index = dict.keys.iter().position(|candidate| candidate == key);
        child_ptr(index.and_then(|index| dict.spaces.get(index)))
    })
}

/// Borrow a `Dict` space's `index`-th child in declaration order -- the same
/// order as `rlmesh_space_dict_key`, so `key(i)` names `get_at(i)`. NULL for any
/// other kind or an out-of-range index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_dict_get_at(
    spec: *const RlmeshSpaceSpec,
    index: usize,
) -> *const RlmeshSpaceSpec {
    guard_value(std::ptr::null(), || {
        match spec_ref(spec).map(|spec| &spec.spec) {
            Some(Some(SpaceKind::Dict(dict))) => child_ptr(dict.spaces.get(index)),
            _ => std::ptr::null(),
        }
    })
}

/// Borrow a `Dict` space's `index`-th key: `*out_len` UTF-8 bytes, NOT
/// NUL-terminated, valid while `spec` lives. Keys are in declaration order,
/// parallel to the children.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_dict_key(
    spec: *const RlmeshSpaceSpec,
    index: usize,
    out_ptr: *mut *const c_char,
    out_len: *mut usize,
) -> RlmeshStatus {
    guard(|| {
        let Some(SpaceKind::Dict(dict)) = &spec_ref(spec).ok_or_else(null_spec)?.spec else {
            return Err(CapiError::invalid_value("space is not a Dict"));
        };
        let key = dict
            .keys
            .get(index)
            .ok_or_else(|| CapiError::invalid_arg("dict key index out of range"))?;
        write_out(out_ptr, key.as_ptr().cast::<c_char>())?;
        write_out(out_len, key.len())
    })
}

/// Write the `index`-th element's inclusive bounds of a `Box` space to
/// `out_low`/`out_high` (row-major order; a uniform bound broadcasts, an
/// undeclared bound is -inf/+inf). `index` must be below the shape's element
/// count. Either out-pointer may be NULL to skip it; on failure neither is
/// written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_box_bounds(
    spec: *const RlmeshSpaceSpec,
    index: usize,
    out_low: *mut f64,
    out_high: *mut f64,
) -> RlmeshStatus {
    guard(|| {
        let spec = spec_ref(spec).ok_or_else(null_spec)?;
        let Some(SpaceKind::Box(box_spec)) = &spec.spec else {
            return Err(CapiError::invalid_value("space is not a Box"));
        };
        let numel: usize = spec
            .shape
            .iter()
            .map(|&dim| usize::try_from(dim).unwrap_or(0))
            .product();
        if index >= numel {
            return Err(CapiError::invalid_arg("bounds index out of range"));
        }
        let (low, high) = bounds_at(box_spec.bounds.as_ref(), spec.dtype, index);
        write_opt(out_low, low);
        write_opt(out_high, high);
        Ok(())
    })
}

/// Write a `Discrete` space's category count and first value to `out_n` /
/// `out_start` (valid values are `start ..= start + n - 1`). Either out-pointer
/// may be NULL to skip it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_discrete_n(
    spec: *const RlmeshSpaceSpec,
    out_n: *mut i64,
    out_start: *mut i64,
) -> RlmeshStatus {
    guard(|| {
        let Some(SpaceKind::Discrete(discrete)) = &spec_ref(spec).ok_or_else(null_spec)?.spec
        else {
            return Err(CapiError::invalid_value("space is not Discrete"));
        };
        write_opt(out_n, discrete.n);
        write_opt(out_start, discrete.start);
        Ok(())
    })
}

/// Write a `Text` space's length limits (in characters) to `out_min`/`out_max`.
/// Either out-pointer may be NULL to skip it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_text_length(
    spec: *const RlmeshSpaceSpec,
    out_min: *mut i64,
    out_max: *mut i64,
) -> RlmeshStatus {
    guard(|| {
        let Some(SpaceKind::Text(text)) = &spec_ref(spec).ok_or_else(null_spec)?.spec else {
            return Err(CapiError::invalid_value("space is not Text"));
        };
        write_opt(out_min, text.min_length);
        write_opt(out_max, text.max_length);
        Ok(())
    })
}

/// Borrow a `Text` space's charset: `*out_len` UTF-8 bytes, NOT NUL-terminated,
/// valid while `spec` lives. An empty charset means any character is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_text_charset(
    spec: *const RlmeshSpaceSpec,
    out_ptr: *mut *const c_char,
    out_len: *mut usize,
) -> RlmeshStatus {
    guard(|| {
        let Some(SpaceKind::Text(text)) = &spec_ref(spec).ok_or_else(null_spec)?.spec else {
            return Err(CapiError::invalid_value("space is not Text"));
        };
        write_out(out_ptr, text.charset.as_ptr().cast::<c_char>())?;
        write_out(out_len, text.charset.len())
    })
}

/// Copy a `MultiDiscrete` space's per-element category counts into `out`
/// (capacity `cap`, one entry per element of the shape, row-major).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_copy_nvec(
    spec: *const RlmeshSpaceSpec,
    out: *mut i64,
    cap: usize,
) -> RlmeshStatus {
    guard(|| {
        let Some(SpaceKind::MultiDiscrete(md)) = &spec_ref(spec).ok_or_else(null_spec)?.spec else {
            return Err(CapiError::invalid_value("space is not MultiDiscrete"));
        };
        if md.nvec.len() > cap {
            return Err(CapiError::invalid_arg("nvec buffer too small"));
        }
        if md.nvec.is_empty() {
            return Ok(());
        }
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        // SAFETY: `out` is non-NULL and the caller declared capacity `cap >= len`.
        let dst = unsafe { std::slice::from_raw_parts_mut(out, md.nvec.len()) };
        dst.copy_from_slice(&md.nvec);
        Ok(())
    })
}

fn null_spec() -> CapiError {
    CapiError::invalid_arg("null space spec")
}

fn child_ptr(child: Option<&SpaceSpec>) -> *const RlmeshSpaceSpec {
    child.map_or(std::ptr::null(), |spec| (spec as *const SpaceSpec).cast())
}

/// The `index`-th element's bounds as `f64`, decoding byte-typed bounds with the
/// space dtype. An absent or malformed bound reads as unbounded.
fn bounds_at(bounds: Option<&BoxBounds>, dtype: DType, index: usize) -> (f64, f64) {
    match bounds {
        Some(BoxBounds::Uniform(uniform)) => (uniform.low, uniform.high),
        Some(BoxBounds::Elementwise(elementwise)) => (
            elementwise
                .low
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY),
            elementwise
                .high
                .get(index)
                .copied()
                .unwrap_or(f64::INFINITY),
        ),
        Some(BoxBounds::TypedUniform(typed)) => (
            typed_at(&typed.low, dtype, 0, f64::NEG_INFINITY),
            typed_at(&typed.high, dtype, 0, f64::INFINITY),
        ),
        Some(BoxBounds::TypedElementwise(typed)) => (
            typed_at(&typed.low, dtype, index, f64::NEG_INFINITY),
            typed_at(&typed.high, dtype, index, f64::INFINITY),
        ),
        Some(BoxBounds::Unbounded(_)) | None => (f64::NEG_INFINITY, f64::INFINITY),
    }
}

fn typed_at(bytes: &[u8], dtype: DType, index: usize, default: f64) -> f64 {
    decode_scalars(bytes, dtype)
        .ok()
        .and_then(|scalars| scalars.get(index).map(|scalar| scalar.to_f64(dtype)))
        .unwrap_or(default)
}

/// Write through an optional C out-pointer (NULL means "not wanted").
fn write_opt<T>(out: *mut T, value: T) {
    // SAFETY: `as_mut` rejects NULL; a non-NULL out-pointer being writable for
    // `T` is the caller's ABI contract.
    if let Some(slot) = unsafe { out.as_mut() } {
        *slot = value;
    }
}

// ---- builders (the env-authoring side) -------------------------------------

/// Hand a built space to C as an owned handle (free with `rlmesh_space_free`).
fn into_space_handle(spec: SpaceSpec) -> *mut RlmeshSpaceSpec {
    Box::into_raw(Box::new(RlmeshSpaceSpec(spec)))
}

fn built(result: Result<SpaceSpec, SpaceError>) -> Result<*mut RlmeshSpaceSpec, CapiError> {
    result
        .map(into_space_handle)
        .map_err(|err| CapiError::invalid_arg(format!("invalid space: {err}")))
}

/// # Safety
/// `ptr` is NULL only when `len == 0`; otherwise it points at `len` readable `T`s.
unsafe fn slice_arg<'a, T>(ptr: *const T, len: usize, what: &str) -> Result<&'a [T], CapiError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(CapiError::invalid_arg(format!("null {what}")));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn core_dtype(dtype: RlmeshDType) -> Result<DType, CapiError> {
    dtype
        .to_core()
        .ok_or_else(|| CapiError::invalid_arg("unsupported dtype"))
}

fn is_signed_int(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
    )
}

fn is_unsigned_int(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64
    )
}

/// An integer dtype's bound must be a finite whole number.
fn int_bound(value: f64) -> Result<i64, CapiError> {
    if value.is_finite() && value.fract() == 0.0 {
        #[allow(clippy::cast_possible_truncation)]
        Ok(value as i64)
    } else {
        Err(CapiError::invalid_arg(format!(
            "integer Box bound {value} is not a finite whole number"
        )))
    }
}

/// A `Box` space with one bound pair for every element. `low`/`high` may be
/// infinite for a float dtype (unbounded on that side); an integer dtype needs
/// finite whole-number bounds. Returns an owned space, or NULL on error.
///
/// # Safety
/// `shape` points at `ndim` readable `int64_t`s (NULL when `ndim == 0`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_box(
    dtype: RlmeshDType,
    shape: *const i64,
    ndim: usize,
    low: f64,
    high: f64,
) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| {
        let dtype = core_dtype(dtype)?;
        let shape = unsafe { slice_arg(shape, ndim, "shape") }?.to_vec();
        let builder = if is_unsigned_int(dtype) {
            let low = u64::try_from(int_bound(low)?)
                .map_err(|_| CapiError::invalid_arg("unsigned Box bound is negative"))?;
            let high = u64::try_from(int_bound(high)?)
                .map_err(|_| CapiError::invalid_arg("unsigned Box bound is negative"))?;
            BoxSpaceBuilder::uint_scalar(low, high, shape)
        } else if is_signed_int(dtype) {
            BoxSpaceBuilder::int_scalar(int_bound(low)?, int_bound(high)?, shape)
        } else {
            BoxSpaceBuilder::scalar(low, high, shape)
        };
        built(builder.dtype(dtype).build())
    })
}

/// A `Box` space with per-element bounds: `low`/`high` each hold the shape's
/// element count, row-major. Same dtype rules as `rlmesh_space_box`.
///
/// # Safety
/// `shape` points at `ndim` `int64_t`s; `low`/`high` at `numel` doubles each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_box_elementwise(
    dtype: RlmeshDType,
    shape: *const i64,
    ndim: usize,
    low: *const f64,
    high: *const f64,
) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| {
        let dtype = core_dtype(dtype)?;
        let shape = unsafe { slice_arg(shape, ndim, "shape") }?.to_vec();
        let numel = shape.iter().try_fold(1usize, |acc, &dim| {
            usize::try_from(dim)
                .ok()
                .and_then(|dim| acc.checked_mul(dim))
                .ok_or_else(|| CapiError::invalid_arg("Box shape has a negative dimension"))
        })?;
        let low = unsafe { slice_arg(low, numel, "low") }?;
        let high = unsafe { slice_arg(high, numel, "high") }?;
        let builder = if is_signed_int(dtype) || is_unsigned_int(dtype) {
            let ints = |bounds: &[f64]| -> Result<Vec<i64>, CapiError> {
                bounds.iter().map(|&b| int_bound(b)).collect()
            };
            BoxSpaceBuilder::int_tensor(ints(low)?, ints(high)?, shape)
        } else {
            BoxSpaceBuilder::tensor(low.to_vec(), high.to_vec(), shape)
        };
        built(builder.dtype(dtype).build())
    })
}

/// A `Discrete` space over `start ..= start + n - 1`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_discrete(n: i64, start: i64) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| built(DiscreteBuilder::new(n).start(start).build()))
}

/// A `MultiBinary` space of the given shape (`[n]` for the usual flat one).
///
/// # Safety
/// `shape` points at `ndim` readable `int64_t`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_multi_binary(
    shape: *const i64,
    ndim: usize,
) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| {
        let shape = unsafe { slice_arg(shape, ndim, "shape") }?.to_vec();
        built(MultiBinaryBuilder::shape(shape).build())
    })
}

/// A flat `MultiDiscrete` space with one category count per element.
///
/// # Safety
/// `nvec` points at `n` readable `int64_t`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_multi_discrete(
    nvec: *const i64,
    n: usize,
) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| {
        let nvec = unsafe { slice_arg(nvec, n, "nvec") }?.to_vec();
        built(MultiDiscreteBuilder::vector(nvec).build())
    })
}

/// A `Text` space of `min_length ..= max_length` characters drawn from
/// `charset` (NUL-terminated UTF-8; NULL or "" allows any character).
///
/// # Safety
/// `charset` is NULL or a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_text(
    min_length: i64,
    max_length: i64,
    charset: *const c_char,
) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| {
        let mut builder = TextBuilder::new(max_length).min_length(min_length);
        if !charset.is_null() {
            let charset = unsafe { CStr::from_ptr(charset) }
                .to_str()
                .map_err(|_| CapiError::invalid_arg("charset is not UTF-8"))?;
            builder = builder.charset(charset);
        }
        built(builder.build())
    })
}

/// Reclaim `n` owned child handles, all or nothing: the composite builders take
/// ownership only once every argument checks out.
///
/// # Safety
/// `children` points at `n` handles from the space constructors.
unsafe fn take_children(
    children: *const *mut RlmeshSpaceSpec,
    n: usize,
) -> Result<Vec<SpaceSpec>, CapiError> {
    let children = unsafe { slice_arg(children, n, "children") }?;
    if children.iter().any(|child| child.is_null()) {
        return Err(CapiError::invalid_arg("null child space"));
    }
    Ok(children
        .iter()
        .map(|&child| unsafe { Box::from_raw(child) }.0)
        .collect())
}

/// A `Dict` space of `n` (key, child) pairs, in declaration order. Takes
/// ownership of every child on success; on failure (NULL) takes none of them.
/// Keys must be unique.
///
/// # Safety
/// `keys` points at `n` C strings; `children` at `n` owned space handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_dict(
    keys: *const *const c_char,
    children: *const *mut RlmeshSpaceSpec,
    n: usize,
) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| {
        let raw_keys = unsafe { slice_arg(keys, n, "keys") }?;
        let mut names: Vec<String> = Vec::with_capacity(n);
        for &key in raw_keys {
            if key.is_null() {
                return Err(CapiError::invalid_arg("null dict key"));
            }
            let key = unsafe { CStr::from_ptr(key) }
                .to_str()
                .map_err(|_| CapiError::invalid_arg("dict key is not UTF-8"))?;
            if names.iter().any(|seen| seen == key) {
                return Err(CapiError::invalid_arg(format!(
                    "duplicate dict key {key:?}"
                )));
            }
            names.push(key.to_string());
        }
        let children = unsafe { take_children(children, n) }?;
        built(
            DictSpaceBuilder::new()
                .extend(names.into_iter().zip(children))
                .build(),
        )
    })
}

/// A `Tuple` space of `n` children. Takes ownership of every child on success;
/// on failure (NULL) takes none of them.
///
/// # Safety
/// `children` points at `n` owned space handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_tuple(
    children: *const *mut RlmeshSpaceSpec,
    n: usize,
) -> *mut RlmeshSpaceSpec {
    guard_ptr(|| {
        let children = unsafe { take_children(children, n) }?;
        built(TupleSpaceBuilder::new().extend(children).build())
    })
}

/// Free an owned space (from a constructor). Not for a borrowed contract space
/// or child. NULL is a no-op.
///
/// # Safety
/// `spec` is NULL or an owned handle this thread has not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_free(spec: *mut RlmeshSpaceSpec) {
    guard_value((), || {
        if !spec.is_null() {
            drop(unsafe { Box::from_raw(spec) });
        }
    });
}
