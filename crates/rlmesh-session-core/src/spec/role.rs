//! The binder's Gym door: a model bound to a Gym env by role.
//!
//! A Gym env publishes its spaces plus sparse v1 tags that say what each
//! leaf means (a role: `image/primary`, `proprio/eef_rot` in `quat_xyzw`,
//! `action/gripper`); a model declares the payload it wants (a v1
//! `ModelSpec`). `rlmesh-adapters` lowers both ([`input`]), and [`bind`]
//! matches every model leaf and every env actuator to its counterpart by
//! role, part and provenance, deciding each conversion the way v1 specs
//! always meant it: declaring a target size, encoding or range is the
//! consent to convert into it, and the lossy fits are the named
//! [`Projection`](crate::spec::Projection)s a spec must allow. The result
//! is a [`Binding`] of per-leaf plans that [`apply`] runs on every step, in
//! the model's process.
//!
//! The two doors run the same conversions
//! ([`spec::rotation`](crate::spec::rotation),
//! [`spec::image`](crate::spec::image), [`spec::convert`](crate::spec::convert)
//! and the element codec of [`spec::apply`](crate::spec::apply)). They match
//! differently, by role here and by kind and path in
//! [`spec::bind()`](crate::spec::bind()), and they decide differently because
//! the spec styles mean different things: a v1 declaration is the consent to
//! convert into it, a v2 entry needs `allow`.
//!
//! - [`roles`]: the role vocabulary and its registry.
//! - [`frames`]: the `frame`, `reference` and `provenance` vocabularies.
//! - [`path`]: where a feature is read from and a model value lands.
//! - [`plan`]: the binding and its plans; [`apply`] runs them.

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
