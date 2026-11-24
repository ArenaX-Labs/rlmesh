//! Semantic role vocabulary for matching env features to model inputs.
//!
//! Roles are opaque strings to the binder. Each domain module ships a `ROLES` table and the
//! [`registry`] is their union; [`parts`] suggests names for where a role repeats on a body.
//!
//! A registered role may pin its dim via [`registry::DimLaw`], only as validation: the author
//! always declares `dim=`. An unregistered (ad-hoc) role resolves on string agreement alone.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod body;
pub mod core;
pub mod manipulation;
pub mod parts;
pub mod registry;
