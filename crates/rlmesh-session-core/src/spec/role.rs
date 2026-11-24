//! The binder's Gym door: a model bound to a Gym env by role.
//!
//! A Gym env publishes its spaces plus sparse v1 tags that say what each leaf means (a role:
//! `image/primary`, `proprio/eef_rot` in `quat_xyzw`, `action/gripper`); a model declares the
//! payload it wants (a v1 `ModelSpec`). `rlmesh-adapters` lowers both ([`input`]), and [`bind`]
//! matches every model leaf and env actuator by role, part and provenance. A declared target
//! size, encoding or range is the consent to convert into it; the lossy fits are the named
//! [`Projection`](crate::spec::Projection)s a spec must allow. The result is a [`Binding`] of
//! per-leaf plans that [`apply`] runs on every step.
//!
//! The v2 binder ([`spec::bind()`](crate::spec::bind())) runs the same conversions
//! ([`spec::rotation`](crate::spec::rotation), [`spec::image`](crate::spec::image),
//! [`spec::convert`](crate::spec::convert), [`spec::apply`](crate::spec::apply)) but matches by
//! kind and path, and needs `allow` for any conversion.
//!
//! - [`roles`]: the role vocabulary and its registry.
//! - [`frames`]: the `frame`, `reference` and `provenance` vocabularies.
//! - [`path`]: where a feature is read from and a model value lands.
//! - [`plan`]: the binding and its plans; [`apply`] runs them.

#[allow(unused_imports)]
use crate::prelude::*;

mod action;
pub mod apply;
mod bind;
mod custom;
mod describe;
mod error;
mod fmt;
pub mod frames;
mod image;
pub mod input;
pub mod path;
pub mod plan;
pub mod roles;
mod state;
mod text;

pub use bind::bind;
pub use error::{Advisory, Error, ErrorCode, Severity};
pub use input::{Env, Spec};
pub use path::{NodePath, PathSeg};
pub use plan::Binding;
