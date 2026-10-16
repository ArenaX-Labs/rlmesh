//! The borrowed downcasts behind `Space::as_box` / `as_discrete` / `as_dict`.
//!
//! The one place in `rlmesh::spaces` that needs `unsafe`: each typed space is
//! a `#[repr(transparent)]` newtype over `Space`, so a `&Space` already known
//! to be of that kind is a valid `&T`. Callers check the kind first.
#![allow(unsafe_code)]

use super::{Box, Dict, Discrete, Space};

pub(super) fn as_box(space: &Space) -> &Box {
    // SAFETY: repr(transparent) over Space.
    unsafe { &*(space as *const Space).cast::<Box>() }
}

pub(super) fn as_discrete(space: &Space) -> &Discrete {
    // SAFETY: repr(transparent) over Space.
    unsafe { &*(space as *const Space).cast::<Discrete>() }
}

pub(super) fn as_dict(space: &Space) -> &Dict {
    // SAFETY: repr(transparent) over Space.
    unsafe { &*(space as *const Space).cast::<Dict>() }
}
