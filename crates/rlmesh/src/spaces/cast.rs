//! The borrowed downcasts behind `Space::as_box` / `as_discrete` / `as_dict`.
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
