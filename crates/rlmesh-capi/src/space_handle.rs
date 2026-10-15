//! The stateful space handle (`RLMeshSpace`, with `RLMeshBox`, `RLMeshDiscrete`
//! and `RLMeshDict` as typedefs of it): Python's `rlmesh.spaces` over the Rust
//! facade's `rlmesh::spaces`, plus the spec readers that go with it.
#![allow(unsafe_code)] // FFI: raw pointers, a C union and repr(transparent) handles.

use std::ffi::c_char;

use rlmesh::spaces::{
    Bound, BoundsKind, Box as SpaceBox, BoxBound, BoxOptions, Dict, Discrete, DiscreteOptions,
    Space, SpaceKind, SpaceSpec,
};

use crate::abi::status::{CapiError, RLMeshStatus, guard, guard_ptr, guard_value};
use crate::codec::RLMeshBytes;
use crate::spaces::{RLMeshSpaceSpec, slice_arg, spec_ref};
use crate::value::dtype::{DTYPE_OTHER, dtype_from_c, dtype_to_c, static_str};
use crate::value::handle::{RLMeshValue, into_handle, write_out};

/// A space handle: a spec plus its sampler. Clones share the sampler.
#[repr(transparent)]
pub struct RLMeshSpace(pub(crate) Space);

/// Length-bearing UTF-8 text (`RLMeshStr`): `data` may be NULL only when `len`
/// is 0; outputs are borrowed and not NUL-terminated.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshStr {
    pub data: *const c_char,
    pub len: usize,
}

impl RLMeshStr {
    pub(crate) fn borrowed(text: &str) -> Self {
        Self {
            data: text.as_ptr().cast(),
            len: text.len(),
        }
    }

    /// # Safety
    /// `data` points at `len` readable bytes (or `len` is 0).
    pub(crate) unsafe fn read<'a>(self, what: &str) -> Result<&'a str, CapiError> {
        let bytes = unsafe { slice_arg(self.data.cast::<u8>(), self.len, what) }?;
        std::str::from_utf8(bytes)
            .map_err(|_| CapiError::invalid_arg(format!("{what} is not UTF-8")))
    }
}

/// `RLMeshBound`'s payload, read per `type_`. `b` is a byte so a caller's
/// non-0/1 value is not an invalid Rust `bool`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union RLMeshBoundValue {
    pub b: u8,
    pub i: i64,
    pub u: u64,
    pub f: f64,
}

/// One typed Box bound (`RLMeshBound`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshBound {
    pub type_: i32,
    pub value: RLMeshBoundValue,
}

const BOUND_BOOL: i32 = 1;
const BOUND_INT: i32 = 2;
const BOUND_UINT: i32 = 3;
const BOUND_FLOAT: i32 = 4;

impl RLMeshBound {
    fn read(self) -> Result<Bound, CapiError> {
        // SAFETY: each arm reads the member `type_` names; every member is
        // plain data valid for any bit pattern.
        Ok(match self.type_ {
            BOUND_BOOL => Bound::Bool(unsafe { self.value.b } != 0),
            BOUND_INT => Bound::Int(unsafe { self.value.i }),
            BOUND_UINT => Bound::Uint(unsafe { self.value.u }),
            BOUND_FLOAT => Bound::Float(unsafe { self.value.f }),
            other => {
                return Err(CapiError::invalid_arg(format!(
                    "invalid RLMeshBound type_ {other}"
                )));
            }
        })
    }

    fn write(bound: Bound) -> Self {
        let (type_, value) = match bound {
            Bound::Bool(b) => (BOUND_BOOL, RLMeshBoundValue { b: u8::from(b) }),
            Bound::Int(i) => (BOUND_INT, RLMeshBoundValue { i }),
            Bound::Uint(u) => (BOUND_UINT, RLMeshBoundValue { u }),
            Bound::Float(f) => (BOUND_FLOAT, RLMeshBoundValue { f }),
            _ => (0, RLMeshBoundValue { u: 0 }),
        };
        Self { type_, value }
    }
}

/// `Box`'s options (`RLMeshBoxOptions`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshBoxOptions {
    pub struct_size: usize,
    pub dtype: i32,
    pub _reserved: u32,
}

/// `Discrete`'s options (`RLMeshDiscreteOptions`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshDiscreteOptions {
    pub struct_size: usize,
    pub start: i64,
    pub dtype: i32,
    pub _reserved: u32,
}

/// Read a sized options struct by R-19: NULL is the defaults; a
/// `struct_size` below the first generation's size is refused; bytes past
/// this library's `sizeof` must be zero.
///
/// # Safety
/// `ptr` is NULL or points at `struct_size` readable bytes.
unsafe fn read_options<T: Copy>(
    ptr: *const T,
    min_size: usize,
    name: &str,
) -> Result<Option<T>, CapiError> {
    if ptr.is_null() {
        return Ok(None);
    }
    debug_assert_eq!(min_size, std::mem::size_of::<T>());
    unsafe { crate::abi::sizing::read(ptr, name) }
}

fn space_dtype(value: i32, field: &str) -> Result<rlmesh::DType, CapiError> {
    dtype_from_c(value).ok_or_else(|| {
        if value == DTYPE_OTHER {
            CapiError::invalid_arg(format!("{field}: RLMESH_DTYPE_OTHER is not a space dtype"))
        } else {
            CapiError::invalid_arg(format!("{field}: invalid dtype {value}"))
        }
    })
}

// ---- handle ownership -------------------------------------------------------

fn into_space_handle(space: Space) -> *mut RLMeshSpace {
    std::boxed::Box::into_raw(std::boxed::Box::new(RLMeshSpace(space)))
}

fn write_space(out: *mut *mut RLMeshSpace, space: Space) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    write_out(out, into_space_handle(space))
}

fn space_ref<'a>(space: *const RLMeshSpace) -> Option<&'a Space> {
    // SAFETY: repr(transparent) over Space; NULL maps to None.
    unsafe { space.cast::<Space>().as_ref() }
}

fn require_space<'a>(space: *const RLMeshSpace) -> Result<&'a Space, CapiError> {
    space_ref(space).ok_or_else(|| CapiError::invalid_arg("null space"))
}

pub(crate) fn require_spec(spec: *const RLMeshSpaceSpec) -> Result<SpaceSpec, CapiError> {
    spec_ref(spec)
        .map(|wire| rlmesh::binding::space_spec_from_wire(wire.clone()))
        .ok_or_else(|| CapiError::invalid_arg("null space spec"))
}

fn typed<'a, T>(
    space: *const RLMeshSpace,
    cast: impl FnOnce(&'a Space) -> Option<&'a T>,
    kind: &str,
) -> Result<&'a T, CapiError> {
    let space = require_space(space)?;
    let actual = space.kind();
    cast(space)
        .ok_or_else(|| CapiError::type_error(format!("expected a {kind} space, got {actual}")))
}

/// SPC-20's buffer rule: `*out_len` always gets the full length; `out == NULL`
/// with `cap == 0` is a length query; a short `cap` writes nothing.
pub(crate) fn copy_buffer<T: Copy>(
    src: &[T],
    out: *mut T,
    cap: usize,
    out_len: *mut usize,
) -> Result<(), CapiError> {
    write_out(out_len, src.len())?;
    if out.is_null() && cap == 0 {
        return Ok(());
    }
    if cap < src.len() {
        return Err(CapiError::invalid_arg(format!(
            "buffer holds {cap} entries, {} needed",
            src.len()
        )));
    }
    if !src.is_empty() {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        // SAFETY: `out` holds at least `cap >= src.len()` entries.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), out, src.len()) };
    }
    Ok(())
}

fn kind_to_c(kind: SpaceKind) -> i32 {
    match kind {
        SpaceKind::Box => 1,
        SpaceKind::Discrete => 2,
        SpaceKind::MultiBinary => 3,
        SpaceKind::MultiDiscrete => 4,
        SpaceKind::Text => 5,
        SpaceKind::Dict => 10,
        SpaceKind::Tuple => 11,
        _ => 0,
    }
}

const ALL_KINDS: [SpaceKind; 8] = [
    SpaceKind::Unknown,
    SpaceKind::Box,
    SpaceKind::Discrete,
    SpaceKind::MultiBinary,
    SpaceKind::MultiDiscrete,
    SpaceKind::Text,
    SpaceKind::Dict,
    SpaceKind::Tuple,
];

const ALL_BOUNDS_KINDS: [BoundsKind; 5] = [
    BoundsKind::Unbounded,
    BoundsKind::Uniform,
    BoundsKind::Elementwise,
    BoundsKind::TypedUniform,
    BoundsKind::TypedElementwise,
];

fn bounds_kind_to_c(kind: BoundsKind) -> i32 {
    ALL_BOUNDS_KINDS
        .iter()
        .position(|k| *k == kind)
        .map_or(0, |index| index as i32 + 1)
}

fn repr_bytes(out: *mut RLMeshBytes, text: String) -> Result<(), CapiError> {
    write_out(out, RLMeshBytes::from_vec(text.into_bytes()))
}

// ---- vocabularies -----------------------------------------------------------

/// Python's spelling of a space kind (static storage), NULL for an unknown value.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_kind_str(kind: i32) -> *const c_char {
    ALL_KINDS
        .iter()
        .find(|k| kind_to_c(**k) == kind)
        .map_or(std::ptr::null(), |k| static_str(k.as_str()))
}

/// Python's spelling of a bounds kind, NULL for `INVALID` or an unknown value.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_bounds_kind_str(kind: i32) -> *const c_char {
    ALL_BOUNDS_KINDS
        .iter()
        .find(|k| bounds_kind_to_c(**k) == kind)
        .map_or(std::ptr::null(), |k| static_str(k.as_str()))
}

// ---- SpaceSpec --------------------------------------------------------------

/// A new owned copy of `spec` (NULL for NULL), freed with `rlmesh_space_spec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_spec_clone(spec: *const RLMeshSpaceSpec) -> *mut RLMeshSpaceSpec {
    guard_value(std::ptr::null_mut(), || {
        spec_ref(spec).map_or(std::ptr::null_mut(), |wire| {
            std::boxed::Box::into_raw(std::boxed::Box::new(RLMeshSpaceSpec(wire.clone())))
        })
    })
}

/// The spec's kind; `UNKNOWN` (0) for NULL or a spec with no payload.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_spec_kind(spec: *const RLMeshSpaceSpec) -> i32 {
    guard_value(0, || {
        spec_ref(spec).map_or(0, |wire| {
            kind_to_c(rlmesh::binding::space_spec_from_wire(wire.clone()).kind())
        })
    })
}

/// The spec's shape (SPC-20 buffer rule).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_spec_shape(
    spec: *const RLMeshSpaceSpec,
    out: *mut i64,
    cap: usize,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        let wire = spec_ref(spec).ok_or_else(|| CapiError::invalid_arg("null space spec"))?;
        copy_buffer(&wire.shape, out, cap, out_len)
    })
}

/// The spec's dtype (`RLMeshDType`); a Dict reads `FLOAT32`; `INVALID` for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_spec_dtype(spec: *const RLMeshSpaceSpec) -> i32 {
    guard_value(0, || {
        spec_ref(spec).map_or(0, |wire| {
            dtype_to_c(rlmesh::binding::space_spec_from_wire(wire.clone()).dtype())
        })
    })
}

/// A fresh space (own sampler) for `spec`; accepts every kind.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_spec_to_space(
    spec: *const RLMeshSpaceSpec,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| write_space(out, require_spec(spec)?.to_space()))
}

/// The spec's Gymnasium-style text (Python's `repr`), as owned bytes.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_spec_repr(
    spec: *const RLMeshSpaceSpec,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| repr_bytes(out, require_spec(spec)?.to_string()))
}

/// Python's `space_from_spec`: refuses a spec with no payload.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_spaces_space_from_spec(
    spec: *const RLMeshSpaceSpec,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| write_space(out, rlmesh::spaces::space_from_spec(require_spec(spec)?)?))
}

// ---- Space ------------------------------------------------------------------

/// `Space(spec)`: never fails for a valid spec.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_new(
    spec: *const RLMeshSpaceSpec,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| write_space(out, Space::new(require_spec(spec)?)))
}

/// A new handle sharing `space`'s sampler (NULL for NULL).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_clone(space: *const RLMeshSpace) -> *mut RLMeshSpace {
    guard_ptr(
        || Ok(space_ref(space).map_or(std::ptr::null_mut(), |s| into_space_handle(s.clone()))),
    )
}

/// Free an owned Space handle. Specs use rlmesh_space_spec_free. Not for a
/// borrowed child. NULL is a no-op.
///
/// # Safety
/// `space` is NULL or an owned handle not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_free(space: *mut RLMeshSpace) {
    guard_value((), || {
        if space.is_null() {
            return;
        }
        drop(unsafe { std::boxed::Box::from_raw(space) });
    });
}

/// The space's spec, borrowed for as long as any handle sharing the space
/// lives; NULL for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_spec(space: *const RLMeshSpace) -> *const RLMeshSpaceSpec {
    guard_value(std::ptr::null(), || {
        space_ref(space).map_or(std::ptr::null(), |space| {
            let wire: *const rlmesh::binding::spaces::SpaceSpec =
                rlmesh::binding::space_spec_wire(space.spec());
            wire.cast()
        })
    })
}

/// The space's kind; `UNKNOWN` for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_kind(space: *const RLMeshSpace) -> i32 {
    guard_value(0, || {
        space_ref(space).map_or(0, |space| kind_to_c(space.kind()))
    })
}

/// The facade space's dtype; INVALID for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_dtype(space: *const RLMeshSpace) -> i32 {
    guard_value(0, || {
        space_ref(space).map_or(0, |space| dtype_to_c(space.dtype()))
    })
}

/// The space's shape (SPC-20 buffer rule).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_shape(
    space: *const RLMeshSpace,
    out: *mut i64,
    cap: usize,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| copy_buffer(require_space(space)?.shape(), out, cap, out_len))
}

/// `seed(seed)`: `seed == NULL` draws one from OS entropy. The seed applied
/// goes to `*out_seed` (may be NULL).
///
/// # Safety
/// `seed` is NULL or readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_seed(
    space: *const RLMeshSpace,
    seed: *const u64,
    out_seed: *mut u64,
) -> RLMeshStatus {
    guard(|| {
        let space = require_space(space)?;
        let used = space.seed(unsafe { seed.as_ref() }.copied());
        if !out_seed.is_null() {
            write_out(out_seed, used)?;
        }
        Ok(())
    })
}

/// Draw a value (an owned `RLMeshValue`, freed with `rlmesh_value_free`).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_sample(
    space: *const RLMeshSpace,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let value = require_space(space)?.sample()?;
        write_out(out, into_handle(value))
    })
}

/// Whether `value` is a member of `space`; false for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_contains(
    space: *const RLMeshSpace,
    value: *const RLMeshValue,
) -> bool {
    guard_value(false, || {
        // SAFETY: value is NULL or a live immutable value handle.
        let value = unsafe { value.as_ref() };
        match (space_ref(space), value) {
            (Some(space), Some(value)) => space.contains(value.value()),
            _ => false,
        }
    })
}

/// Python's `==` on spaces (normalized spec comparison); false for NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_equals(a: *const RLMeshSpace, b: *const RLMeshSpace) -> bool {
    guard_value(false, || match (space_ref(a), space_ref(b)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    })
}

/// The space's Gymnasium-style text (Python's `repr`), as owned bytes.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_space_repr(
    space: *const RLMeshSpace,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| repr_bytes(out, require_space(space)?.to_string()))
}

// ---- Box --------------------------------------------------------------------

const BOX_OPTIONS_MIN_SIZE: usize = std::mem::size_of::<RLMeshBoxOptions>();
const DISCRETE_OPTIONS_MIN_SIZE: usize = std::mem::size_of::<RLMeshDiscreteOptions>();

unsafe fn box_options(options: *const RLMeshBoxOptions) -> Result<BoxOptions, CapiError> {
    Ok(
        match unsafe { read_options(options, BOX_OPTIONS_MIN_SIZE, "RLMeshBoxOptions") }? {
            None => BoxOptions::default(),
            Some(options) => BoxOptions {
                dtype: space_dtype(options.dtype, "RLMeshBoxOptions.dtype")?,
            },
        },
    )
}

/// `Box(low, high, shape, dtype)`.
///
/// # Safety
/// `shape` holds `ndim` entries; `options` is NULL or a sized struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_box_new(
    low: RLMeshBound,
    high: RLMeshBound,
    shape: *const i64,
    ndim: usize,
    options: *const RLMeshBoxOptions,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let shape = unsafe { slice_arg(shape, ndim, "shape") }?.to_vec();
        let options = unsafe { box_options(options) }?;
        let space = SpaceBox::new(low.read()?, high.read()?, shape, options)?;
        write_space(out, space.into())
    })
}

/// Per-element `Box` (Rust-free extra, SPC-22): `low` and `high` hold
/// `bounds_len` entries, the shape's element count.
///
/// # Safety
/// `low`/`high` hold `bounds_len` entries; `shape` holds `ndim`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_box_elementwise_new(
    low: *const RLMeshBound,
    high: *const RLMeshBound,
    bounds_len: usize,
    shape: *const i64,
    ndim: usize,
    options: *const RLMeshBoxOptions,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let read = |bounds: &[RLMeshBound]| -> Result<Vec<Bound>, CapiError> {
            bounds.iter().map(|bound| bound.read()).collect()
        };
        let low = read(unsafe { slice_arg(low, bounds_len, "low") }?)?;
        let high = read(unsafe { slice_arg(high, bounds_len, "high") }?)?;
        let shape = unsafe { slice_arg(shape, ndim, "shape") }?.to_vec();
        let options = unsafe { box_options(options) }?;
        let space = rlmesh::binding::box_elementwise(&low, &high, shape, options)?;
        write_space(out, space.into())
    })
}

/// `Box(spec)`: another kind is `RLMESH_ERR_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_box_from_spec(
    spec: *const RLMeshSpaceSpec,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| write_space(out, SpaceBox::from_spec(require_spec(spec)?)?.into()))
}

fn box_bound_entries(bound: Option<BoxBound>) -> Vec<RLMeshBound> {
    match bound {
        None => Vec::new(),
        Some(BoxBound::Uniform(bound)) => vec![RLMeshBound::write(bound)],
        Some(BoxBound::Elementwise(bounds)) => bounds.into_iter().map(RLMeshBound::write).collect(),
        Some(_) => Vec::new(),
    }
}

/// The lower bound's entries (SPC-20 buffer rule): 0 when unbounded, 1 when
/// uniform, one per element otherwise; read `rlmesh_box_bounds_kind` for which.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_box_low(
    space: *const RLMeshSpace,
    out: *mut RLMeshBound,
    cap: usize,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        let space = typed(space, Space::as_box, "box")?;
        copy_buffer(&box_bound_entries(space.low()), out, cap, out_len)
    })
}

/// The upper bound's entries; as `rlmesh_box_low`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_box_high(
    space: *const RLMeshSpace,
    out: *mut RLMeshBound,
    cap: usize,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        let space = typed(space, Space::as_box, "box")?;
        copy_buffer(&box_bound_entries(space.high()), out, cap, out_len)
    })
}

/// How the bounds are stored: `*out_has` false when the spec has none.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_box_bounds_kind(
    space: *const RLMeshSpace,
    out_has: *mut bool,
    out: *mut i32,
) -> RLMeshStatus {
    guard(|| {
        let space = typed(space, Space::as_box, "box")?;
        let kind = space.bounds_kind();
        write_out(out_has, kind.is_some())?;
        write_out(out, kind.map_or(0, bounds_kind_to_c))
    })
}

// ---- Discrete ---------------------------------------------------------------

/// `Discrete(n, start, dtype)`.
///
/// # Safety
/// `options` is NULL or a sized struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_discrete_new(
    n: i64,
    options: *const RLMeshDiscreteOptions,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let options = match unsafe {
            read_options(options, DISCRETE_OPTIONS_MIN_SIZE, "RLMeshDiscreteOptions")
        }? {
            None => DiscreteOptions::default(),
            Some(options) => DiscreteOptions {
                start: options.start,
                dtype: space_dtype(options.dtype, "RLMeshDiscreteOptions.dtype")?,
            },
        };
        write_space(out, Discrete::new(n, options)?.into())
    })
}

/// `Discrete(spec)`: another kind is `RLMESH_ERR_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_discrete_from_spec(
    spec: *const RLMeshSpaceSpec,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| write_space(out, Discrete::from_spec(require_spec(spec)?)?.into()))
}

/// `n`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_discrete_n(space: *const RLMeshSpace, out: *mut i64) -> RLMeshStatus {
    guard(|| write_out(out, typed(space, Space::as_discrete, "discrete")?.n()))
}

/// `start`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_discrete_start(space: *const RLMeshSpace, out: *mut i64) -> RLMeshStatus {
    guard(|| write_out(out, typed(space, Space::as_discrete, "discrete")?.start()))
}

// ---- Dict -------------------------------------------------------------------

/// `Dict(spaces)`: copies each spec; keys sort at construction. A duplicate or
/// empty key is `RLMESH_ERR_INVALID_ARGUMENT`.
///
/// # Safety
/// `keys` and `specs` hold `n` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_dict_new(
    keys: *const RLMeshStr,
    specs: *const *const RLMeshSpaceSpec,
    n: usize,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let keys = unsafe { slice_arg(keys, n, "keys") }?;
        let specs = unsafe { slice_arg(specs, n, "specs") }?;
        let mut entries = Vec::with_capacity(n);
        for (key, spec) in keys.iter().zip(specs) {
            let key = unsafe { key.read("dict key") }?;
            entries.push((key.to_string(), require_spec(*spec)?));
        }
        write_space(out, Dict::new(entries)?.into())
    })
}

/// `Dict(spec)`: another kind is `RLMESH_ERR_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_dict_from_spec(
    spec: *const RLMeshSpaceSpec,
    out: *mut *mut RLMeshSpace,
) -> RLMeshStatus {
    guard(|| write_space(out, Dict::from_spec(require_spec(spec)?)?.into()))
}

/// `len(d)`.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_dict_len(space: *const RLMeshSpace, out: *mut usize) -> RLMeshStatus {
    guard(|| write_out(out, typed(space, Space::as_dict, "dict")?.len()))
}

/// The `i`-th key in declaration order, borrowed from the Dict; past the end
/// reads as absent (`RLMESH_OK`, `{NULL, 0}`).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_dict_key_at(
    space: *const RLMeshSpace,
    i: usize,
    out: *mut RLMeshStr,
) -> RLMeshStatus {
    guard(|| {
        let dict = typed(space, Space::as_dict, "dict")?;
        let key = dict.keys().nth(i).map_or(
            RLMeshStr {
                data: std::ptr::null(),
                len: 0,
            },
            RLMeshStr::borrowed,
        );
        write_out(out, key)
    })
}

/// The `i`-th child space in declaration order, borrowed from the Dict; past
/// the end reads as absent (`RLMESH_OK`, `*out = NULL`).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_dict_value_at(
    space: *const RLMeshSpace,
    i: usize,
    out: *mut *const RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let dict = typed(space, Space::as_dict, "dict")?;
        let child = dict
            .values()
            .nth(i)
            .map_or(std::ptr::null(), |child| (child as *const Space).cast());
        write_out(out, child)
    })
}

/// `d[key]`, borrowed; `*out = NULL` (and OK) when absent.
///
/// # Safety
/// `key` holds `key.len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_dict_get(
    space: *const RLMeshSpace,
    key: RLMeshStr,
    out: *mut *const RLMeshSpace,
) -> RLMeshStatus {
    guard(|| {
        let dict = typed(space, Space::as_dict, "dict")?;
        let key = unsafe { key.read("dict key") }?;
        let child = dict
            .get(key)
            .map_or(std::ptr::null(), |child| (child as *const Space).cast());
        write_out(out, child)
    })
}

/// `key in d` into `*out`: `RLMESH_ERR_TYPE` on another kind,
/// `RLMESH_ERR_INVALID_ARGUMENT` on a malformed key.
///
/// # Safety
/// `key` holds `key.len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_dict_contains_key(
    space: *const RLMeshSpace,
    key: RLMeshStr,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        let dict = typed(space, Space::as_dict, "dict")?;
        let key = unsafe { key.read("dict key") }?;
        write_out(out, dict.contains_key(key))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_dict_values_keep_remote_spec_order_and_nul_keys() {
        use crate::value::handle::{
            rlmesh_value_as_int, rlmesh_value_dict_get, rlmesh_value_dict_key_at,
            rlmesh_value_dict_value_at, rlmesh_value_free,
        };
        use rlmesh_spaces::{DType, DictSpec, DiscreteSpec, SpaceKind, SpaceSpec};

        // A remote spec may declare any key order. Dict::new sorts keys, so
        // sampling a locally built Dict alone cannot detect the old BTreeMap bug.
        let keys = ["z", "m\0key", "a"];
        let numbers = [7, 11, 13];
        let spec = RLMeshSpaceSpec(SpaceSpec {
            shape: vec![],
            dtype: DType::Float32,
            spec: Some(SpaceKind::Dict(DictSpec {
                keys: keys.iter().map(|s| (*s).to_owned()).collect(),
                spaces: numbers
                    .iter()
                    .map(|&start| SpaceSpec {
                        shape: vec![],
                        dtype: DType::Int64,
                        spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 1, start })),
                    })
                    .collect(),
            })),
        });
        let mut space = std::ptr::null_mut();
        assert_eq!(rlmesh_dict_from_spec(&spec, &mut space), RLMeshStatus::Ok);
        let mut sampled = std::ptr::null_mut();
        assert_eq!(rlmesh_space_sample(space, &mut sampled), RLMeshStatus::Ok);
        assert!(rlmesh_space_contains(space, sampled));
        unsafe { rlmesh_space_free(space) };
        drop(spec);
        for (i, expected) in keys.iter().enumerate() {
            let mut key = RLMeshStr {
                data: std::ptr::null(),
                len: 0,
            };
            assert_eq!(
                rlmesh_value_dict_key_at(sampled, i, &mut key),
                RLMeshStatus::Ok
            );
            assert_eq!(
                unsafe { std::slice::from_raw_parts(key.data.cast::<u8>(), key.len) },
                expected.as_bytes()
            );
            let mut by_key = std::ptr::null();
            let mut by_index = std::ptr::null();
            assert_eq!(
                unsafe {
                    rlmesh_value_dict_get(sampled, RLMeshStr::borrowed(expected), &mut by_key)
                },
                RLMeshStatus::Ok
            );
            assert_eq!(
                rlmesh_value_dict_value_at(sampled, i, &mut by_index),
                RLMeshStatus::Ok
            );
            assert!(!by_key.is_null());
            assert_eq!(by_key, by_index);
            let mut number = 0;
            assert_eq!(rlmesh_value_as_int(by_key, &mut number), RLMeshStatus::Ok);
            assert_eq!(number, numbers[i]);
        }
        unsafe { rlmesh_value_free(sampled) };
    }

    #[test]
    fn options_follow_struct_sizing() {
        let shape = [2i64];
        let low = RLMeshBound::write(Bound::Float(0.0));
        let high = RLMeshBound::write(Bound::Float(1.0));
        let mut out: *mut RLMeshSpace = std::ptr::null_mut();

        // Below the minimum (including 0) is refused.
        let zeroed = RLMeshBoxOptions {
            struct_size: 0,
            dtype: 11,
            _reserved: 0,
        };
        let status = unsafe { rlmesh_box_new(low, high, shape.as_ptr(), 1, &zeroed, &mut out) };
        assert_eq!(status, RLMeshStatus::InvalidArgument);

        // A larger caller struct with a nonzero tail is a TypeError.
        #[repr(C)]
        struct Newer {
            base: RLMeshBoxOptions,
            extra: u64,
        }
        let newer = Newer {
            base: RLMeshBoxOptions {
                struct_size: std::mem::size_of::<Newer>(),
                dtype: 11,
                _reserved: 0,
            },
            extra: 1,
        };
        let status = unsafe { rlmesh_box_new(low, high, shape.as_ptr(), 1, &newer.base, &mut out) };
        assert_eq!(status, RLMeshStatus::Type);

        // A zero tail is accepted.
        let newer = Newer { extra: 0, ..newer };
        let status = unsafe { rlmesh_box_new(low, high, shape.as_ptr(), 1, &newer.base, &mut out) };
        assert_eq!(status, RLMeshStatus::Ok);
        assert_eq!(rlmesh_space_kind(out), 1);
        unsafe { rlmesh_space_free(out) };
    }

    #[test]
    fn bound_type_invalid_is_refused() {
        let shape = [1i64];
        let invalid = RLMeshBound {
            type_: 0,
            value: RLMeshBoundValue { u: 0 },
        };
        let high = RLMeshBound::write(Bound::Float(1.0));
        let mut out: *mut RLMeshSpace = std::ptr::null_mut();
        let status =
            unsafe { rlmesh_box_new(invalid, high, shape.as_ptr(), 1, std::ptr::null(), &mut out) };
        assert_eq!(status, RLMeshStatus::InvalidArgument);
        assert!(out.is_null());
    }

    #[test]
    fn vocabulary_spellings() {
        let text = |ptr: *const c_char| unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(text(rlmesh_space_kind_str(3)), "multi_binary");
        assert_eq!(text(rlmesh_space_kind_str(0)), "unknown");
        assert!(rlmesh_space_kind_str(7).is_null());
        assert_eq!(text(rlmesh_bounds_kind_str(4)), "typed_uniform");
        assert!(rlmesh_bounds_kind_str(0).is_null());
    }
}
